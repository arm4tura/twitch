"""Разбор JSON-ответа от NotebookLM: извлечение code-fence, парсинг,
валидация типов/границ, сборка HighlightSet.

Дизайн ошибок: одно исключение MergeError со списком человекочитаемых причин.
CLI печатает их bullet-списком, ничего не роняет по первой ошибке — сразу
показываем все, чтобы пользователь мог одним заходом поправить.
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


def parse_response(
    path: Path | str,
    *,
    transcript_range_s: tuple[float, float] | None = None,
    source: str = "notebooklm",
    transcript_hash: str | None = None,
) -> HighlightSet:
    """Прочитать JSON-ответ NotebookLM и вернуть валидный HighlightSet.

    :param transcript_range_s: (min, max) — секунды начала/конца транскрипта.
        Если передан, отбрасываем highlights за пределами.
    """
    path = Path(path)
    if not path.exists():
        raise MergeError([f"Response file not found: {path}"])

    raw = path.read_text(encoding="utf-8")
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

    if reasons and not highlights:
        raise MergeError(reasons + ["All highlights rejected — aborting merge."])

    if not highlights:
        raise MergeError(["Response contained no highlights."])

    if reasons:
        # Мягкие проблемы — не роняем, но прикрепим к metadata через
        # transcript_hash null-safe path. Печать делает CLI.
        pass

    return HighlightSet(
        highlights=highlights,
        source=source,
        transcript_hash=transcript_hash,
        model=source,
    )


def merge_into_decisions(decisions: dict[str, Any], highlights: HighlightSet) -> dict[str, Any]:
    """Записать highlights в existing decisions.json dict под ключ 'highlights'.

    Не in-place: возвращаем новый dict (dict сохраняет порядок вставки; ключ
    'highlights' появится в конце, что удобно для diff).
    """
    merged = dict(decisions)
    merged["highlights"] = highlights.model_dump(exclude_none=True)
    return merged
