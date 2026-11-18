"""Фаза 4: контракт OfftopicSegment/OfftopicSet — новые поля и совместимость.

Проверяем:
- новые улики сегмента (pause_s/anchored) и метаданные набора (method,
  source_transcript_hash, source_model) с их дефолтами;
- обратную совместимость: JSON, сериализованный ДО Фазы 4 (без method и без
  pause_s/anchored), читается как оконный прогон без улик — не падает;
- валидацию границ полей (pause_s ≥ 0, anchored ∈ [0,1], method — из литерала);
- extra="forbid" остаётся в силе (фронт не протащит опечатку тихо);
- round-trip через model_dump(exclude_none) — то, что мержится в decisions.json.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from twitch_cut.llm_offtopic.model import OfftopicSegment, OfftopicSet


# --- Дефолты новых полей ------------------------------------------------------


def test_segment_new_fields_default_none():
    seg = OfftopicSegment(start_s=1.0, end_s=2.0, confidence=0.5, reason="x")
    assert seg.pause_s is None
    assert seg.anchored is None


def test_set_new_fields_defaults():
    s = OfftopicSet(source="mock")
    # method по умолчанию — оконный путь (обратная совместимость).
    assert s.method == "windows"
    assert s.source_transcript_hash is None
    assert s.source_model is None


# --- Обратная совместимость с до-Фаза-4 JSON ---------------------------------


def test_legacy_set_without_method_reads_as_windows():
    # Так выглядел OfftopicSet до Фазы 4: ни method, ни source_*.
    legacy = {
        "segments": [
            {"start_s": 10.0, "end_s": 20.0, "confidence": 0.8, "reason": "своё дело"}
        ],
        "source": "builtin:anthropic",
        "transcript_hash": "abc",
        "model": "claude-x",
    }
    s = OfftopicSet.model_validate(legacy)
    assert s.method == "windows"
    assert s.source_transcript_hash is None
    seg = s.segments[0]
    assert seg.pause_s is None
    assert seg.anchored is None


# --- Заполненные улики + round-trip ------------------------------------------


def test_source_anchored_roundtrip():
    s = OfftopicSet(
        segments=[
            OfftopicSegment(
                start_s=10.0,
                end_s=20.0,
                confidence=0.9,
                reason="ушёл в чат",
                pause_s=277.0,
                anchored=0.117,
            )
        ],
        source="builtin:anthropic",
        model="claude-opus-5",
        method="source_anchored",
        source_transcript_hash="src-hash",
        source_model="gigaam-v3",
    )
    # exclude_none — ровно то, что merge_into_decisions кладёт в decisions.json.
    dumped = s.model_dump(exclude_none=True)
    assert dumped["method"] == "source_anchored"
    assert dumped["source_transcript_hash"] == "src-hash"
    assert dumped["source_model"] == "gigaam-v3"
    seg = dumped["segments"][0]
    assert seg["pause_s"] == 277.0
    assert seg["anchored"] == 0.117

    # И обратно — переживает re-validate без потерь.
    again = OfftopicSet.model_validate(dumped)
    assert again.method == "source_anchored"
    assert again.segments[0].pause_s == 277.0
    assert again.segments[0].anchored == 0.117


def test_dump_exclude_none_omits_empty_source_fields():
    # У оконного/одно-проходного результата source_* пустые — их не должно быть
    # в decisions.json (иначе UI подумает, что был исходник).
    s = OfftopicSet(source="mock", method="single_pass")
    dumped = s.model_dump(exclude_none=True)
    assert dumped["method"] == "single_pass"
    assert "source_transcript_hash" not in dumped
    assert "source_model" not in dumped


# --- Валидация ----------------------------------------------------------------


def test_negative_pause_rejected():
    with pytest.raises(ValidationError):
        OfftopicSegment(start_s=1.0, end_s=2.0, confidence=0.5, reason="x", pause_s=-1.0)


@pytest.mark.parametrize("bad", [-0.1, 1.5])
def test_anchored_out_of_range_rejected(bad):
    with pytest.raises(ValidationError):
        OfftopicSegment(start_s=1.0, end_s=2.0, confidence=0.5, reason="x", anchored=bad)


def test_invalid_method_rejected():
    with pytest.raises(ValidationError):
        OfftopicSet(source="mock", method="bogus")


def test_extra_field_still_forbidden():
    # Фаза 4 добавила поля, но extra="forbid" остаётся — опечатка падает.
    with pytest.raises(ValidationError):
        OfftopicSet(source="mock", methdo="windows")
