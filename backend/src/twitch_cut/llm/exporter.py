"""Собрать пакет файлов для загрузки в NotebookLM.

Выход в out_dir:
- transcript_001.md, ..._NNN.md — word-stream, ≤450k слов на файл.
- prompt.md — инструкция для LLM.
- schema.json — JSON Schema ожидаемого ответа (для документации/справки).
- README.md — краткая шпаргалка для пользователя: как загрузить и что делать
  с ответом.

Всё это пользователь руками:
1. Открывает notebooklm.google.com.
2. Создаёт новый notebook, загружает transcript_*.md как sources.
3. Копирует содержимое prompt.md в чат.
4. Копирует JSON-ответ в файл (напр. response.json).
5. Запускает `twitch-cut highlights-import ...`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..cache import write_json
from .prompt import RESPONSE_SCHEMA, build_prompt
from .segments import (
    WORD_LIMIT_DEFAULT,
    build_word_lines,
    chunk_bounds,
    chunk_by_word_limit,
)

README_MARKDOWN = """\
# NotebookLM highlights export

Этот каталог собран `twitch-cut highlights-export` и содержит всё, что нужно
для выбора highlights через NotebookLM.

## Что делать

1. Открой https://notebooklm.google.com и создай новый notebook.
2. Загрузи как sources все файлы `transcript_*.md` из этого каталога
   (лимит NotebookLM: 50 источников, 500 000 слов на источник, 200 MB).
3. Скопируй содержимое `prompt.md` в чат NotebookLM.
4. Модель ответит **строго JSON** внутри code-fence ```` ```json ... ``` ````.
   Сохрани ответ в файл (например `response.json` в этом же каталоге).
5. Вернись в терминал и запусти:

   ```
   twitch-cut highlights-import \\
       --decisions <path-to-decisions.json> \\
       --response <path-to-response.json> \\
       --output <path-to-decisions.json>   # можно тот же — перезапишется
   ```

## Про схему ответа

`schema.json` — JSON Schema ожидаемого ответа. Программа валидирует ответ по
Pydantic-модели `Highlight` (те же ограничения), а `schema.json` — для
справки, чтобы можно было руками поправить ответ, если LLM ошиблась.

## Метаданные экспорта

Диапазон покрытия и chunk map — в `manifest.json`.
"""


def _chunk_header(chunk_id: int, total: int, first_ts: str, last_ts: str, n_words: int) -> str:
    return (
        f"# transcript chunk {chunk_id}/{total}\n\n"
        f"- range: `{first_ts}` — `{last_ts}`\n"
        f"- words: {n_words:,}\n\n"
        "---\n\n"
    )


def _chunk_footer(chunk_id: int, total: int) -> str:
    return f"\n\n---\n\n_end of chunk {chunk_id}/{total}_\n"


def build_notebooklm_package(
    transcript: dict[str, Any],
    out_dir: Path | str,
    *,
    n_highlights: int = 5,
    max_words_per_chunk: int = WORD_LIMIT_DEFAULT,
) -> dict[str, Any]:
    """Записать все артефакты в out_dir и вернуть manifest.

    Manifest содержит: chunks[] (id/path/first_ts/last_ts/word_count),
    n_highlights, total_words, prompt_path, schema_path.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    lines = build_word_lines(transcript)
    if not lines:
        raise ValueError(
            "Transcript has no words — check that segments[].words[] is populated."
        )

    chunks = chunk_by_word_limit(lines, max_words=max_words_per_chunk)
    total = len(chunks)
    if total > 50:
        raise ValueError(
            f"Transcript produces {total} chunks — NotebookLM limit is 50 sources. "
            "Reduce transcript length or increase max_words_per_chunk (careful: "
            "500k words is the hard NotebookLM limit)."
        )

    chunk_manifest: list[dict[str, Any]] = []
    for i, chunk in enumerate(chunks, start=1):
        first_ts, last_ts = chunk_bounds(chunk)
        n_words = len(chunk)
        fname = f"transcript_{i:03d}.md"
        path = out_dir / fname
        body = (
            _chunk_header(i, total, first_ts, last_ts, n_words)
            + "\n".join(chunk)
            + _chunk_footer(i, total)
        )
        path.write_text(body, encoding="utf-8")
        chunk_manifest.append(
            {
                "id": i,
                "path": fname,
                "first_ts": first_ts,
                "last_ts": last_ts,
                "word_count": n_words,
            }
        )

    prompt_path = out_dir / "prompt.md"
    prompt_path.write_text(build_prompt(n_highlights), encoding="utf-8")

    schema_path = out_dir / "schema.json"
    write_json(schema_path, RESPONSE_SCHEMA)

    readme_path = out_dir / "README.md"
    readme_path.write_text(README_MARKDOWN, encoding="utf-8")

    manifest = {
        "chunks": chunk_manifest,
        "n_highlights": n_highlights,
        "total_words": len(lines),
        "max_words_per_chunk": max_words_per_chunk,
        "files": {
            "prompt": prompt_path.name,
            "schema": schema_path.name,
            "readme": readme_path.name,
        },
    }
    write_json(out_dir / "manifest.json", manifest)
    return manifest
