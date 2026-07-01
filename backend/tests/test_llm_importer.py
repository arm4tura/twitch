import json

import pytest

from twitch_cut.llm.importer import MergeError, merge_into_decisions, parse_response


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
