from twitch_cut.llm.segments import (
    build_word_lines,
    chunk_by_word_limit,
    chunk_bounds,
)


def _mk_transcript(words):
    return {
        "segments": [
            {
                "start": 0.0,
                "end": 60.0,
                "words": [
                    {"word": w, "start": s, "end": e} for w, s, e in words
                ],
            }
        ]
    }


def test_build_word_lines_formats_timestamps_ms():
    t = _mk_transcript([("привет", 0.1, 0.6), ("мир", 1.234, 1.5)])
    lines = build_word_lines(t)
    assert lines == ["[00:00:00.100] привет", "[00:00:01.234] мир"]


def test_build_word_lines_skips_empty_words():
    t = _mk_transcript([("привет", 0.1, 0.6), ("", 0.7, 0.8), ("мир", 1.0, 1.2)])
    lines = build_word_lines(t)
    assert len(lines) == 2
    assert "мир" in lines[1]


def test_build_word_lines_falls_back_to_segment_start_when_word_missing_time():
    # Слово без word-level start/end берёт segment_start (см. iter_words).
    t = {
        "segments": [
            {"start": 12.5, "end": 15.0, "words": [{"word": "ало"}]}
        ]
    }
    lines = build_word_lines(t)
    assert lines == ["[00:00:12.500] ало"]


def test_chunk_by_word_limit_splits_evenly():
    lines = [f"[00:00:00.{i:03d}] w{i}" for i in range(10)]
    chunks = chunk_by_word_limit(lines, max_words=4)
    assert [len(c) for c in chunks] == [4, 4, 2]


def test_chunk_by_word_limit_rejects_zero():
    import pytest

    with pytest.raises(ValueError):
        chunk_by_word_limit(["x"], max_words=0)


def test_chunk_bounds_returns_first_and_last_timestamp():
    chunk = ["[00:00:00.100] a", "[00:00:05.500] b", "[00:00:10.000] c"]
    assert chunk_bounds(chunk) == ("00:00:00.100", "00:00:10.000")


def test_chunk_bounds_handles_empty():
    assert chunk_bounds([]) == ("??:??:??.???", "??:??:??.???")
