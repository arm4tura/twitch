import json

import pytest

from twitch_cut.llm.importer import (
    MergeError,
    merge_into_decisions,
    parse_response,
    parse_text,
)


def _write(tmp_path, name, content):
    p = tmp_path / name
    p.write_text(content, encoding="utf-8")
    return p


VALID_ONE = {
    "highlights": [
        {
            "start_s": 10.5,
            "end_s": 55.5,
            "title": "первый пик",
            "reason": "стример вздрогнул",
            "score": 0.85,
            "quote": "аааа",
        }
    ]
}


def test_parse_response_accepts_fenced_json(tmp_path):
    p = _write(
        tmp_path,
        "r.json",
        "ok:\n```json\n" + json.dumps(VALID_ONE) + "\n```\ntail text",
    )
    hs = parse_response(p)
    assert len(hs.highlights) == 1
    assert hs.highlights[0].title == "первый пик"


def test_parse_response_accepts_raw_json(tmp_path):
    p = _write(tmp_path, "r.json", json.dumps(VALID_ONE))
    hs = parse_response(p)
    assert hs.highlights[0].duration_s == 45.0


def test_parse_response_rejects_no_json(tmp_path):
    p = _write(tmp_path, "r.json", "просто текст без JSON")
    with pytest.raises(MergeError, match="No JSON object found"):
        parse_response(p)


def test_parse_response_rejects_malformed_json(tmp_path):
    p = _write(tmp_path, "r.json", "{not json}")
    with pytest.raises(MergeError, match="Invalid JSON"):
        parse_response(p)


def test_parse_response_rejects_missing_highlights_key(tmp_path):
    p = _write(tmp_path, "r.json", json.dumps({"other": []}))
    with pytest.raises(MergeError, match="'highlights' array"):
        parse_response(p)


def test_parse_response_rejects_inverted_time(tmp_path):
    bad = {"highlights": [{**VALID_ONE["highlights"][0], "start_s": 100, "end_s": 50}]}
    p = _write(tmp_path, "r.json", json.dumps(bad))
    with pytest.raises(MergeError, match="end_s"):
        parse_response(p)


def test_parse_response_rejects_out_of_range(tmp_path):
    p = _write(tmp_path, "r.json", json.dumps(VALID_ONE))
    with pytest.raises(MergeError, match="outside transcript range"):
        parse_response(p, transcript_range_s=(0.0, 20.0))


def test_parse_response_rejects_too_long(tmp_path):
    bad = {"highlights": [{**VALID_ONE["highlights"][0], "start_s": 0, "end_s": 500}]}
    p = _write(tmp_path, "r.json", json.dumps(bad))
    with pytest.raises(MergeError, match="hard limits"):
        parse_response(p)


def test_parse_response_missing_file(tmp_path):
    with pytest.raises(MergeError, match="not found"):
        parse_response(tmp_path / "nope.json")


def test_merge_into_decisions_appends_highlights_key(tmp_path):
    p = _write(tmp_path, "r.json", json.dumps(VALID_ONE))
    hs = parse_response(p)
    dec = {"mutes": [], "cuts": []}
    merged = merge_into_decisions(dec, hs)
    assert "highlights" in merged
    assert merged["highlights"]["highlights"][0]["title"] == "первый пик"
    # Оригинал не тронут.
    assert "highlights" not in dec


# --- parse_text: толерантный разбор ------------------------------------------


def test_parse_text_returns_no_warnings_when_clean():
    hs, warnings = parse_text(json.dumps(VALID_ONE))
    assert len(hs.highlights) == 1
    assert warnings == []


def test_parse_text_keeps_valid_drops_broken():
    # Один годный highlight + один с инвертированным временем: годный проходит,
    # битый попадает в warnings (а не роняет весь разбор).
    mixed = {
        "highlights": [
            VALID_ONE["highlights"][0],
            {**VALID_ONE["highlights"][0], "start_s": 100, "end_s": 50},
        ]
    }
    hs, warnings = parse_text(json.dumps(mixed))
    assert len(hs.highlights) == 1
    assert hs.highlights[0].title == "первый пик"
    assert len(warnings) == 1
    assert "end_s" in warnings[0]


def test_parse_text_raises_when_all_rejected():
    # Все элементы битые → нечего спасать → MergeError.
    allbad = {
        "highlights": [
            {**VALID_ONE["highlights"][0], "start_s": 100, "end_s": 50},
        ]
    }
    with pytest.raises(MergeError):
        parse_text(json.dumps(allbad))


# --- merge: обогащение stream-координатами -----------------------------------


def test_merge_enriches_stream_coords_from_range_in():
    # range_in задаёт сдвиг локальное→абсолютное. stream_start_s/stream_end_s
    # должны появиться у каждого highlight (их читают рендер клипов и Vegas).
    hs, _ = parse_text(json.dumps(VALID_ONE))
    dec = {"mutes": [], "cuts": [], "range_in": "00:01:00.000"}
    merged = merge_into_decisions(dec, hs)
    h = merged["highlights"]["highlights"][0]
    assert h["stream_start_s"] == 70.5  # 10.5 + 60
    assert h["stream_end_s"] == 115.5  # 55.5 + 60


def test_merge_stream_coords_default_zero_offset():
    # Нет range_in → offset 0 → stream-координаты совпадают с локальными.
    hs, _ = parse_text(json.dumps(VALID_ONE))
    merged = merge_into_decisions({"mutes": []}, hs)
    h = merged["highlights"]["highlights"][0]
    assert h["stream_start_s"] == 10.5
    assert h["stream_end_s"] == 55.5


def test_merge_explicit_offset_overrides_range_in():
    hs, _ = parse_text(json.dumps(VALID_ONE))
    dec = {"range_in": "00:01:00.000"}
    merged = merge_into_decisions(dec, hs, stream_offset_s=5.0)
    h = merged["highlights"]["highlights"][0]
    assert h["stream_start_s"] == 15.5  # 10.5 + 5, не 60
