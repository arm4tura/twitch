import pytest

from twitch_cut.profanity import RussianNormalizer, detect_profanity, load_banwords
from twitch_cut.timecode import parse_timecode


def test_load_banwords_ignores_comments(tmp_path):
    path = tmp_path / "banwords.txt"
    path.write_text("# comment\n\nБЛИН # inline\n", encoding="utf-8")
    normalizer = RussianNormalizer()
    entries = load_banwords(path, normalizer)
    assert len(entries) == 1
    assert entries[0].normalized == "блин"


def test_detect_surface_match(tmp_path):
    path = tmp_path / "banwords.txt"
    path.write_text("блин\n", encoding="utf-8")
    normalizer = RussianNormalizer()
    entries = load_banwords(path, normalizer)
    transcript = {
        "segments": [
            {
                "id": "seg_1",
                "start": 1.0,
                "end": 2.0,
                "words": [{"word": "Блин!", "start": 1.1, "end": 1.4, "score": 0.9}],
            }
        ]
    }
    matches = detect_profanity(transcript, entries, parse_timecode("00:01:00"), normalizer)
    assert len(matches) == 1
    assert matches[0].match_type == "surface"
    assert matches[0].local_start_ms == 1100
    assert matches[0].stream_start_ms == 61_100


def test_detect_lemma_match_when_pymorphy_available(tmp_path):
    normalizer = RussianNormalizer()
    if normalizer._morph is None:
        pytest.skip("pymorphy3 is not installed in this environment")
    path = tmp_path / "banwords.txt"
    path.write_text("плохой\n", encoding="utf-8")
    entries = load_banwords(path, normalizer)
    transcript = {
        "segments": [
            {"words": [{"word": "плохого", "start": 65.35, "end": 65.8, "score": 0.94}]}
        ]
    }
    matches = detect_profanity(transcript, entries, parse_timecode("00:01:00"), normalizer)
    assert len(matches) == 1
    assert matches[0].match_type == "lemma"
