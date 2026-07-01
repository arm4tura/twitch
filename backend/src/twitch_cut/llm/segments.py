"""Из транскрипта → плоский поток строк `[HH:MM:SS.mmm] слово` и разбиение
на чанки под лимит NotebookLM (500k слов/источник).

Формат для NotebookLM: одна строка = одно слово с точным таймингом. Это
избыточно по размеру, но даёт модели однозначные тайминги для каждого слова,
и она может процитировать start/end с высокой точностью.

Ограничение NotebookLM: 500k слов на источник, 200MB, до 50 источников.
Берём 450k слов/чанк — запас на служебные строки шапки/футера.
"""

from typing import Any, Iterable

from ..transcription import iter_words

WORD_LIMIT_DEFAULT = 450_000


def _fmt_ts(seconds: float | None) -> str:
    """`123.456` → `00:02:03.456`. `None` → `??:??:??.???` (сохраняем видимость)."""
    if seconds is None:
        return "??:??:??.???"
    if seconds < 0:
        seconds = 0.0
    total_ms = int(round(seconds * 1000))
    hh, rem = divmod(total_ms, 3_600_000)
    mm, rem = divmod(rem, 60_000)
    ss, ms = divmod(rem, 1000)
    return f"{hh:02d}:{mm:02d}:{ss:02d}.{ms:03d}"


def build_word_lines(transcript: dict[str, Any]) -> list[str]:
    """Собрать плоский список строк `[HH:MM:SS.mmm] слово`.

    Пустые слова пропускаем (WhisperX иногда выдаёт `""` между repl'ами).
    Токенам без word-level таймингов подставляется segment_start (уже сделано
    в `iter_words`), но флаг `timing_source == 'segment_fallback'` при желании
    можно логировать снаружи.
    """
    lines: list[str] = []
    for w in iter_words(transcript):
        text = (w["text"] or "").strip()
        if not text:
            continue
        lines.append(f"[{_fmt_ts(w['start'])}] {text}")
    return lines


def chunk_by_word_limit(
    lines: list[str],
    max_words: int = WORD_LIMIT_DEFAULT,
) -> list[list[str]]:
    """Разбить на чанки не больше `max_words` слов каждый.

    Одна line = одно слово в нашем формате, поэтому `len(chunk)` == число слов.
    Оставляем возможность override для тестов (`max_words=10` и т.п.).
    """
    if max_words <= 0:
        raise ValueError("max_words must be positive")
    if not lines:
        return []
    chunks: list[list[str]] = []
    for i in range(0, len(lines), max_words):
        chunks.append(lines[i : i + max_words])
    return chunks


def chunk_bounds(chunk: Iterable[str]) -> tuple[str, str]:
    """Вернуть (first_ts, last_ts) строкой для шапки чанка."""
    first = last = None
    for line in chunk:
        if line.startswith("["):
            ts = line[1 : line.index("]")]
            if first is None:
                first = ts
            last = ts
    if first is None:
        return ("??:??:??.???", "??:??:??.???")
    return (first, last or first)
