"""Слияние OfftopicSet в decisions.json под ключ 'offtopic'.

Зеркалит llm/importer.merge_into_decisions: тот же паттерн local→stream
пересчёта через offset (= range_in в секундах). Оффтоп-сегменты кладём в
СВОЙ ключ decisions['offtopic'] — не трогаем mutes/silences/highlights
(каждая фича живёт в своём ключе).

К каждому сегменту добавляем stream_start_s/stream_end_s (абсолютное время
исходника) — их читают рендер клипов и Vegas-маркеры, как highlights.
"""

from __future__ import annotations

from typing import Any

from .model import OfftopicSet


def merge_into_decisions(
    decisions: dict[str, Any],
    offtopic: OfftopicSet,
    *,
    stream_offset_s: float | None = None,
) -> dict[str, Any]:
    """Записать оффтоп-сегменты в decisions dict под ключ 'offtopic'.

    Не in-place: возвращаем новый dict (порядок вставки сохраняется, 'offtopic'
    появляется в конце — удобно для diff).

    :param stream_offset_s: сдвиг LOCAL-времени (от начала транскрипта) →
        АБСОЛЮТНОГО времени исходника. Обычно = range_in в секундах. None →
        берём из decisions.range_in; если и его нет — 0.
    """
    merged = dict(decisions)
    bundle = offtopic.model_dump(exclude_none=True)

    offset = stream_offset_s
    if offset is None:
        offset = _offset_from_decisions(decisions)

    for seg in bundle.get("segments", []):
        seg["stream_start_s"] = round(seg["start_s"] + offset, 3)
        seg["stream_end_s"] = round(seg["end_s"] + offset, 3)

    merged["offtopic"] = bundle
    return merged


def _offset_from_decisions(decisions: dict[str, Any]) -> float:
    """Локальное→stream смещение из decisions.range_in (таймкод/секунды)."""
    raw = decisions.get("range_in")
    if raw in (None, "", 0):
        return 0.0
    try:
        from ..timecode import seconds_float

        return seconds_float(raw)
    except Exception:  # noqa: BLE001
        return 0.0
