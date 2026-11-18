"""Фаза 3: разбор СПИСКА сегментов и снап границ на реплики.

Путь по исходнику отличается от оконного тем, что модель возвращает не один
вердикт, а список интервалов с границами в MM:SS. Здесь проверяем ровно контракт
этого списка:

- пустой `segments` — валидный ответ «оффтопа нет», а НЕ pause_failed;
- отсутствие ключа / битый JSON / обрезка — parse_failed (батч повторяют);
- один сегмент с выдуманным таймкодом не обнуляет весь батч (его дропают);
- границы модели (MM:SS, точность 1с) снапятся на границы реплик, а сегмент,
  не попавший ни в одну реплику, выбрасывается — это галлюцинация.
"""

from __future__ import annotations

import pytest

from twitch_cut.llm_offtopic.prompt import (
    parse_llm_timecode,
    parse_outline_response,
    parse_segments_response,
)
from twitch_cut.llm_offtopic.turns import Turn, snap_span, turns_from_segments


# --- parse_llm_timecode -----------------------------------------------------


def test_parse_llm_timecode_formats():
    assert parse_llm_timecode("01:05") == pytest.approx(65.0)
    assert parse_llm_timecode("00:00") == 0.0
    assert parse_llm_timecode("1:02:03") == pytest.approx(3723.0)
    assert parse_llm_timecode(65) == 65.0
    assert parse_llm_timecode(12.5) == 12.5


def test_parse_llm_timecode_minutes_over_60():
    # На длинном VOD модель продолжает минуты за 60. Строгий parse_timecode
    # такое отвергает, но смысл однозначен — разбираем сами.
    assert parse_llm_timecode("185:23") == pytest.approx(185 * 60 + 23)
    assert parse_llm_timecode("120:00") == pytest.approx(7200.0)


def test_parse_llm_timecode_rejects_garbage():
    assert parse_llm_timecode("скоро") is None
    assert parse_llm_timecode("") is None
    assert parse_llm_timecode(None) is None
    assert parse_llm_timecode(True) is None  # bool — не таймкод
    assert parse_llm_timecode(-5) is None


# --- parse_segments_response: три состояния ---------------------------------


def test_segments_parsed_from_fence():
    ans = parse_segments_response(
        '```json\n{"segments": [{"start": "01:00", "end": "02:30", '
        '"confidence": 0.8, "reason": "своё дело", "quote": "я тут"}]}\n```'
    )
    assert ans.parse_failed is False
    assert len(ans.segments) == 1
    seg = ans.segments[0]
    assert (seg.start_s, seg.end_s) == (60.0, 150.0)
    assert seg.confidence == pytest.approx(0.8)
    assert seg.reason == "своё дело"
    assert seg.quote == "я тут"


def test_empty_segments_is_a_valid_no_offtopic_answer():
    # Пустой список — это ответ «оффтопа нет», НЕ отказ разбора.
    ans = parse_segments_response('{"segments": []}')
    assert ans.parse_failed is False
    assert ans.segments == ()


def test_missing_segments_key_is_parse_failed():
    # Ответ не той схемой / модель заболталась → батч надо повторить.
    ans = parse_segments_response('{"topic": "что-то", "note": "думаю"}')
    assert ans.parse_failed is True
    assert "segments" in ans.error


def test_no_json_at_all_is_parse_failed():
    ans = parse_segments_response("длинное рассуждение без JSON вообще")
    assert ans.parse_failed is True


def test_truncated_json_is_parse_failed():
    # Обрыв по max_tokens: незакрытый объект — не «пустой список».
    ans = parse_segments_response('```json\n{"segments": [{"start": "01:00", "end":')
    assert ans.parse_failed is True


def test_one_bad_timecode_does_not_kill_the_batch():
    ans = parse_segments_response(
        '{"segments": ['
        '{"start": "01:00", "end": "02:00", "confidence": 0.7, "reason": "ok"},'
        '{"start": "хз", "end": "потом", "confidence": 0.9, "reason": "битый"},'
        '{"start": "05:00", "end": "04:00", "confidence": 0.5, "reason": "end<start"}'
        ']}'
    )
    assert ans.parse_failed is False
    assert len(ans.segments) == 1  # выжил только валидный
    assert ans.dropped == 2
    assert len(ans.dropped_reasons) == 2


def test_confidence_clamped_and_reason_defaulted():
    ans = parse_segments_response(
        '{"segments": [{"start": "00:10", "end": "00:20", "confidence": 5}]}'
    )
    seg = ans.segments[0]
    assert seg.confidence == 1.0  # 5 → клампится в [0,1]
    assert seg.reason == "no reason given"


def test_segments_topic_extracted_for_single_pass():
    ans = parse_segments_response('{"topic": "обзор игры", "segments": []}')
    assert ans.topic == "обзор игры"


# --- parse_outline_response -------------------------------------------------


def test_outline_keeps_text_and_extracts_topic():
    raw = "ТЕМА: обзор нового телефона\n- 00:00 распаковка\n- 02:00 камера"
    outline = parse_outline_response(raw)
    assert outline.empty is False
    assert outline.topic == "обзор нового телефона"
    assert "распаковка" in outline.text


def test_empty_outline_is_empty():
    # Обрезка по max_tokens → плана нет; классификатор обязан повторить.
    outline = parse_outline_response("   ")
    assert outline.empty is True
    assert outline.topic == ""


# --- snap_span --------------------------------------------------------------


def _turns() -> list[Turn]:
    return [
        Turn(start_s=10.0, end_s=20.0, text="реплика раз", word_count=2),
        Turn(start_s=25.0, end_s=40.0, text="реплика два", word_count=2),
        Turn(start_s=100.0, end_s=110.0, text="реплика три", word_count=2),
    ]


def test_snap_stretches_to_covered_turns():
    # Модель дала 00:12–00:38 (внутри слов) — тянем к границам реплик, которые
    # сегмент реально пересёк: с 10.0 (начало первой) до 40.0 (конец второй).
    span = snap_span(12.0, 38.0, _turns())
    assert span is not None
    assert (span.start_s, span.end_s) == (10.0, 40.0)
    assert (span.first_index, span.last_index) == (0, 1)
    assert span.turn_count == 2


def test_snap_single_turn():
    span = snap_span(11.0, 15.0, _turns())
    assert (span.start_s, span.end_s) == (10.0, 20.0)
    assert span.turn_count == 1


def test_snap_returns_none_when_span_hits_no_turn():
    # 00:50–00:55 попадает в дыру между репликами (речь видео) → выдумано.
    assert snap_span(50.0, 55.0, _turns()) is None


def test_snap_returns_none_on_degenerate_or_empty():
    assert snap_span(30.0, 30.0, _turns()) is None
    assert snap_span(10.0, 20.0, []) is None


# --- turns_from_segments (деградированный путь) -----------------------------


def test_turns_from_segments_builds_and_sorts():
    transcript = {
        "segments": [
            {"start": 5.0, "end": 8.0, "text": "вторая фраза"},
            {"start": 0.0, "end": 3.0, "text": "первая фраза"},
            {"start": 9.0, "end": 9.5, "text": "   "},  # пустая — пропускается
        ]
    }
    turns = turns_from_segments(transcript)
    assert [t.text for t in turns] == ["первая фраза", "вторая фраза"]
    assert turns[0].start_s == 0.0


def test_turns_from_segments_reconstructs_text_from_words():
    transcript = {
        "segments": [
            {
                "start": 0.0,
                "end": 2.0,
                "words": [
                    {"word": "кот", "start": 0.0, "end": 0.5},
                    {"word": "сидел", "start": 0.6, "end": 1.0},
                ],
            }
        ]
    }
    turns = turns_from_segments(transcript)
    assert turns[0].text == "кот сидел"
    assert turns[0].word_count == 2
