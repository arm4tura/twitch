from pathlib import Path

from twitch_cut.config import PipelineConfig
from twitch_cut.decisions import build_decisions
from twitch_cut.profanity import ProfanityMatch
from twitch_cut.timecode import parse_timecode


def test_build_decisions_uses_local_and_stream_times():
    match = ProfanityMatch(
        word="блин",
        normalized="блин",
        lemma="блин",
        banword="блин",
        banword_lemma="блин",
        local_start_ms=4300,
        local_end_ms=4700,
        stream_start_ms=64_300,
        stream_end_ms=64_700,
        confidence=0.96,
        segment_id="seg_1",
        match_type="surface",
        timing_source="word",
    )
    doc = build_decisions(
        stream_path=Path("stream.mp4"),
        original_path=Path("original_video.mp4"),
        range_in=parse_timecode("00:01:00"),
        range_out=parse_timecode("00:02:00"),
        matches=[match],
        config=PipelineConfig(mute_padding_before_ms=80, mute_padding_after_ms=120),
    )
    mute = doc["mutes"][0]
    cut = doc["cuts"][0]
    assert mute["start"] == 4.22
    assert mute["end"] == 4.82
    assert mute["stream_start"] == 64.22
    assert mute["stream_end"] == 64.82
    assert mute["intro_risk"] is True
    assert mute["status"] == "accepted"
    assert cut["action"] == "CUT"
    assert cut["target"] == "audio"
    assert cut["operation"] == "split_and_mute_audio"
    assert cut["stream_start"] == 64.22
    assert doc["summary"] == {"cuts": 1, "mutes": 1, "raw_matches": 1}


def _make_match() -> ProfanityMatch:
    return ProfanityMatch(
        word="блин",
        normalized="блин",
        lemma="блин",
        banword="блин",
        banword_lemma="блин",
        local_start_ms=1000,
        local_end_ms=1200,
        stream_start_ms=1000,
        stream_end_ms=1200,
        confidence=0.9,
        segment_id="seg_1",
        match_type="surface",
        timing_source="word",
    )


def test_source_reflects_whisperx_transcriber():
    doc = build_decisions(
        stream_path=Path("stream.mp4"),
        original_path=Path("original.mp4"),
        range_in=parse_timecode("00:00:00"),
        range_out=parse_timecode("00:01:00"),
        matches=[_make_match()],
        config=PipelineConfig(transcriber="whisperx"),
    )
    assert doc["mutes"][0]["source"] == "whisperx+pymorphy3"
    assert doc["cuts"][0]["source"] == "whisperx+pymorphy3"


def test_source_reflects_whispercpp_transcriber():
    doc = build_decisions(
        stream_path=Path("stream.mp4"),
        original_path=Path("original.mp4"),
        range_in=parse_timecode("00:00:00"),
        range_out=parse_timecode("00:01:00"),
        matches=[_make_match()],
        config=PipelineConfig(transcriber="whispercpp"),
    )
    assert doc["mutes"][0]["source"] == "whisper.cpp+pymorphy3"
    assert doc["cuts"][0]["source"] == "whisper.cpp+pymorphy3"


def _match_at(
    word: str,
    start_ms: int,
    end_ms: int,
    segment_id: str = "seg_1",
) -> ProfanityMatch:
    return ProfanityMatch(
        word=word,
        normalized=word,
        lemma=word,
        banword=word,
        banword_lemma=word,
        local_start_ms=start_ms,
        local_end_ms=end_ms,
        stream_start_ms=start_ms,
        stream_end_ms=end_ms,
        confidence=0.9,
        segment_id=segment_id,
        match_type="surface",
        timing_source="word",
    )


def test_gap_split_does_not_mute_clean_speech_between_distant_swears():
    # Два мата в одном сегменте, между ними ~4с чистой речи.
    # Старая группировка слепила бы их в один мьют first..last и
    # проглотила бы всю речь между ними. Gap-split должен разбить на два.
    first = _match_at("блять", 1000, 1300, segment_id="seg_1")
    second = _match_at("блять", 5300, 5600, segment_id="seg_1")
    doc = build_decisions(
        stream_path=Path("stream.mp4"),
        original_path=Path("original.mp4"),
        range_in=parse_timecode("00:00:00"),
        range_out=parse_timecode("00:10:00"),
        matches=[first, second],
        config=PipelineConfig(mute_join_gap_ms=600),
    )
    mutes = doc["mutes"]
    assert len(mutes) == 2, "далёкие маты должны дать два отдельных мьюта"
    # Ни один мьют не должен покрывать промежуток 1.3..5.3с чистой речи.
    for m in mutes:
        assert (m["end"] - m["start"]) < 1.0


def test_close_swears_in_one_outburst_stay_joined():
    # 'бля блять нахуй' одним выкриком — разрывы < gap, склеиваются в один.
    a = _match_at("бля", 1000, 1200, segment_id="seg_1")
    b = _match_at("блять", 1250, 1500, segment_id="seg_1")
    c = _match_at("нахуй", 1550, 1850, segment_id="seg_1")
    doc = build_decisions(
        stream_path=Path("stream.mp4"),
        original_path=Path("original.mp4"),
        range_in=parse_timecode("00:00:00"),
        range_out=parse_timecode("00:10:00"),
        matches=[a, b, c],
        config=PipelineConfig(mute_join_gap_ms=600),
    )
    assert len(doc["mutes"]) == 1
    assert doc["mutes"][0]["matched_token_count"] == 3


def test_broken_word_timing_is_capped_to_max_word_seconds():
    # whisper.cpp выдал слову 'блять' длительность 9.48с (end = конец
    # сегмента). В режиме 'word' хвост должен обрезаться до max_word_seconds.
    match = _match_at("блять", 1000, 10_480, segment_id="seg_1")
    doc = build_decisions(
        stream_path=Path("stream.mp4"),
        original_path=Path("original.mp4"),
        range_in=parse_timecode("00:00:00"),
        range_out=parse_timecode("00:10:00"),
        matches=[match],
        config=PipelineConfig(mute_extend_mode="word", mute_max_word_seconds=1.5),
    )
    mute = doc["mutes"][0]
    # start 1.0 - 0.08 padding = 0.92; конец = 1.0 + 1.5 + 0.12 padding = 2.62
    assert mute["end"] - mute["start"] < 2.0
