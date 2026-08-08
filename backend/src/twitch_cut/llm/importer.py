"""Разбор JSON-ответа от NotebookLM: извлечение code-fence, парсинг,
валидация типов/границ, сборка HighlightSet.

Дизайн ошибок: одно исключение MergeError со списком человекочитаемых причин.
CLI печатает их bullet-списком, ничего не роняет по первой ошибке — сразу
показываем все, чтобы пользователь мог одним заходом поправить.

Толерантный режим: если часть highlights не прошла валидацию, а остальные —
прошли, НЕ роняем весь импорт. Возвращаем (HighlightSet, warnings), где
warnings — список проблемных #N; UI показывает «принято 4 из 5, вот битые»,
чтобы пользователь поправил один элемент, а не переделывал ответ целиком.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from .model import Highlight, HighlightSet

# NotebookLM обычно оборачивает JSON в fenced code block. Ловим и с языком, и
# без. Флаг DOTALL — переносы строк внутри JSON допустимы.
_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)

# Допуски по длительности. Промпт просит 30..90с, но LLM иногда чуть выходит
# за границы — не отбрасываем сразу, а warn'аем. Жёсткий верх — 180с
# (highlight длиннее 3 минут = LLM не понял задачу, отклоняем).
DURATION_SOFT_MIN_S = 30.0
DURATION_SOFT_MAX_S = 90.0
DURATION_HARD_MIN_S = 10.0
DURATION_HARD_MAX_S = 180.0


class MergeError(ValueError):
    """Список причин, почему JSON нельзя смержить в decisions.json."""

    def __init__(self, reasons: list[str]):
        self.reasons = reasons
        super().__init__("\n".join(f"  - {r}" for r in reasons))


def _extract_json_blob(raw: str) -> str:
    """Достать JSON-объект: сначала пробуем fenced code, потом — весь raw."""
    stripped = raw.strip()
    m = _FENCE_RE.search(stripped)
    if m:
        return m.group(1)
    # Fallback: если пользователь скопировал только JSON без fence.
    if stripped.startswith("{"):
        return stripped
    raise MergeError(
        [
            "No JSON object found in response. Expected either a ```json fenced "
            "block or a raw {...} object at the start of the file."
        ]
    )


def parse_text(
    raw: str,
    *,
    transcript_range_s: tuple[float, float] | None = None,
    source: str = "notebooklm",
    transcript_hash: str | None = None,
) -> tuple[HighlightSet, list[str]]:
    """Распарсить СЫРОЙ текст ответа (вставленный из NotebookLM) → (set, warnings).

    Отличие от parse_response: работает с текстом напрямую (не файлом) и
    ТОЛЕРАНТНА к частичному браку — если хоть один highlight валиден, возвращает
    его + список warnings по отброшенным. Роняем MergeError только когда парсить
    вообще нечего (нет JSON / битый JSON / все элементы невалидны).

    :param transcript_range_s: (min, max) секунды транскрипта — отбрасываем
        highlights за пределами.
    :returns: (HighlightSet, warnings). warnings пуст, если всё чисто.
    """
    blob = _extract_json_blob(raw)

    try:
        data = json.loads(blob)
    except json.JSONDecodeError as e:
        raise MergeError([f"Invalid JSON at line {e.lineno}, col {e.colno}: {e.msg}"])

    if not isinstance(data, dict) or "highlights" not in data:
        raise MergeError(["Response must be an object with a 'highlights' array."])

    reasons: list[str] = []
    highlights: list[Highlight] = []

    for i, item in enumerate(data["highlights"]):
        try:
            h = Highlight.model_validate(item)
        except ValidationError as e:
            for err in e.errors():
                loc = ".".join(str(x) for x in err["loc"]) or "<root>"
                reasons.append(f"highlight #{i}: {loc}: {err['msg']}")
            continue

        dur = h.duration_s
        if dur < DURATION_HARD_MIN_S or dur > DURATION_HARD_MAX_S:
            reasons.append(
                f"highlight #{i} ({h.title!r}): duration {dur}s outside hard "
                f"limits [{DURATION_HARD_MIN_S}..{DURATION_HARD_MAX_S}]s"
            )
            continue

        if transcript_range_s is not None:
            lo, hi = transcript_range_s
            if h.start_s < lo - 0.5 or h.end_s > hi + 0.5:
                reasons.append(
                    f"highlight #{i} ({h.title!r}): [{h.start_s}, {h.end_s}] "
                    f"outside transcript range [{lo}, {hi}]"
                )
                continue

        highlights.append(h)

    if not highlights:
        # Ничего не спасли — это уже жёсткая ошибка (показываем все причины).
        raise MergeError(
            (reasons or ["Response contained no highlights."])
            + (["All highlights rejected — aborting merge."] if reasons else [])
        )

    hs = HighlightSet(
        highlights=highlights,
        source=source,
        transcript_hash=transcript_hash,
        model=source,
    )
    # reasons здесь — мягкие: часть прошла, часть отброшена. Возвращаем как
    # warnings, чтобы UI/CLI показали «принято N, отброшено M».
    return hs, reasons


def parse_response(
    path: Path | str,
    *,
    transcript_range_s: tuple[float, float] | None = None,
    source: str = "notebooklm",
    transcript_hash: str | None = None,
) -> HighlightSet:
    """Прочитать JSON-ответ NotebookLM из ФАЙЛА и вернуть валидный HighlightSet.

    Тонкая обёртка над parse_text: читает файл, отбрасывает warnings (CLI/сервер
    при необходимости зовут parse_text напрямую, чтобы их показать).

    :param transcript_range_s: (min, max) — секунды начала/конца транскрипта.
        Если передан, отбрасываем highlights за пределами.
    """
    path = Path(path)
    if not path.exists():
        raise MergeError([f"Response file not found: {path}"])

    raw = path.read_text(encoding="utf-8")
    hs, _warnings = parse_text(
        raw,
        transcript_range_s=transcript_range_s,
        source=source,
        transcript_hash=transcript_hash,
    )
    return hs


def merge_into_decisions(
    decisions: dict[str, Any],
    highlights: HighlightSet,
    *,
    stream_offset_s: float | None = None,
) -> dict[str, Any]:
    """Записать highlights в existing decisions.json dict под ключ 'highlights'.

    Не in-place: возвращаем новый dict (dict сохраняет порядок вставки; ключ
    'highlights' появится в конце, что удобно для diff).

    :param stream_offset_s: сдвиг ЛОКАЛЬНОГО времени (от начала транскрипта) →
        АБСОЛЮТНОГО времени исходника. Обычно = range_in в секундах. Если задан,
        к каждому highlight добавляем `stream_start_s`/`stream_end_s` — их читают
        рендер клипов и Vegas-маркеры (как mutes читают `stream_start`). None →
        offset берём из decisions.range_in автоматически; если и его нет — 0.
    """
    merged = dict(decisions)
    bundle = highlights.model_dump(exclude_none=True)

    offset = stream_offset_s
    if offset is None:
        offset = _offset_from_decisions(decisions)

    for h in bundle.get("highlights", []):
        h["stream_start_s"] = round(h["start_s"] + offset, 3)
        h["stream_end_s"] = round(h["end_s"] + offset, 3)

    merged["highlights"] = bundle
    return merged


def _offset_from_decisions(decisions: dict[str, Any]) -> float:
    """Локальное→stream смещение из decisions.range_in (таймкод/секунды)."""
    raw = decisions.get("range_in")
    if raw in (None, "", 0):
        return 0.0
    try:
        # range_in — таймкод "HH:MM:SS.mmm" ИЛИ число секунд.
        from ..timecode import seconds_float

        return seconds_float(raw)
    except Exception:  # noqa: BLE001
        return 0.0
