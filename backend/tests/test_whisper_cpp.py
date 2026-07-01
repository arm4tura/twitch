"""Tests for whisper.cpp JSON normalization and BPE-style token gluing.

Focus areas:
 - Special timestamp tokens like [_TT_300], [_BEG_] must NOT stick to a word,
   even when whisper.cpp emits them WITHOUT a leading space (e.g. fused to
   the previous BPE-subword as one token "ладно[_TT_300]" or as a bare
   "[_TT_300]" right after a content token).
 - BPE-subword tokens are glued into words by leading-space boundaries.
 - Word-level timings (start/end in seconds) survive the normalization.
"""
from __future__ import annotations

from twitch_cut.whisper_cpp import (
    _build_words_from_tokens,
    _strip_special_tokens,
    normalize_whisper_cpp_json,
)


def _tok(text: str, t_from: int, t_to: int, p: float = 0.9) -> dict:
    return {"text": text, "offsets": {"from": t_from, "to": t_to}, "p": p}


def test_strip_special_tokens_handles_inline_attachment() -> None:
    # The exact bug: special token attached to a real word without space.
    assert _strip_special_tokens("ладно[_TT_300]") == "ладно"
    assert _strip_special_tokens("[_TT_300]ладно") == "ладно"
    assert _strip_special_tokens(" [_BEG_]") == " "
    assert _strip_special_tokens("[_TT_300]") == ""
    # <|...|>-style markers as well.
    assert _strip_special_tokens("ладно<|0.00|>") == "ладно"
    assert _strip_special_tokens("<|notimestamps|>привет") == "привет"
    # Plain text is untouched.
    assert _strip_special_tokens("ладно") == "ладно"


def test_build_words_basic_bpe_gluing() -> None:
    tokens = [
        _tok(" при", 0, 200, 0.9),
        _tok("вет", 200, 400, 0.8),
        _tok(" мир", 400, 700, 0.95),
    ]
    words = _build_words_from_tokens(tokens)
    assert [w["word"] for w in words] == ["привет", "мир"]
    assert words[0]["start"] == 0.0
    assert words[0]["end"] == 0.4
    assert words[1]["start"] == 0.4
    assert words[1]["end"] == 0.7


def test_special_token_does_not_stick_when_no_leading_space() -> None:
    # whisper.cpp may emit '[_TT_300]' as its own token, no leading space.
    # It must not become part of the previous word.
    tokens = [
        _tok(" ладно", 100, 350, 0.9),
        _tok("[_TT_300]", 350, 350, 0.0),
        _tok(" дальше", 360, 700, 0.9),
    ]
    words = _build_words_from_tokens(tokens)
    assert [w["word"] for w in words] == ["ладно", "дальше"]


def test_special_token_fused_inside_word_token_is_stripped() -> None:
    # The exact reported bug: special token fused to a content token as one
    # string 'ладно[_TT_300]'. The result must still be just "ладно".
    tokens = [
        _tok(" ладно[_TT_300]", 100, 350, 0.9),
        _tok(" дальше", 360, 700, 0.9),
    ]
    words = _build_words_from_tokens(tokens)
    assert [w["word"] for w in words] == ["ладно", "дальше"]
    assert words[0]["start"] == 0.1
    assert words[0]["end"] == 0.35


def test_angle_brace_special_tokens_are_stripped() -> None:
    tokens = [
        _tok("<|0.00|>", 0, 0, 0.0),
        _tok(" привет", 0, 300, 0.9),
        _tok("<|notimestamps|>", 300, 300, 0.0),
        _tok(" мир", 300, 600, 0.9),
    ]
    words = _build_words_from_tokens(tokens)
    assert [w["word"] for w in words] == ["привет", "мир"]


def test_normalize_whisper_cpp_json_full_segment() -> None:
    raw = {
        "transcription": [
            {
                "offsets": {"from": 0, "to": 700},
                "text": " привет мир",
                "tokens": [
                    _tok(" при", 0, 200, 0.9),
                    _tok("вет", 200, 400, 0.8),
                    _tok("[_TT_400]", 400, 400, 0.0),
                    _tok(" мир", 400, 700, 0.95),
                ],
            }
        ],
        "result": {"language": "ru"},
    }
    norm = normalize_whisper_cpp_json(raw)
    assert norm["backend"] == "whisper.cpp"
    assert norm["language"] == "ru"
    assert len(norm["segments"]) == 1
    seg = norm["segments"][0]
    assert seg["id"] == "seg_000000"
    assert seg["start"] == 0.0
    assert seg["end"] == 0.7
    assert [w["word"] for w in seg["words"]] == ["привет", "мир"]


def test_blank_and_empty_tokens_are_ignored() -> None:
    tokens = [
        _tok("", 0, 0, 0.0),
        _tok(" ", 0, 0, 0.0),
        _tok(" слово", 0, 200, 0.9),
    ]
    words = _build_words_from_tokens(tokens)
    assert [w["word"] for w in words] == ["слово"]


def test_score_is_averaged_across_subword_tokens() -> None:
    tokens = [
        _tok(" при", 0, 100, 1.0),
        _tok("вет", 100, 200, 0.5),
    ]
    words = _build_words_from_tokens(tokens)
    assert len(words) == 1
    # mean of 1.0 and 0.5 = 0.75
    assert abs(words[0]["score"] - 0.75) < 1e-9
