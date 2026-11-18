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
    word_index: int | None = None,
    prev_end_ms: int | None = None,
    next_start_ms: int | None = None,
    match_type: str = "surface",
    needs_review: bool = False,
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
        match_type=match_type,
        timing_source="word",
        needs_review=needs_review,
        prev_word_local_end_ms=prev_end_ms,
        prev_word_stream_end_ms=prev_end_ms,
        next_word_local_start_ms=next_start_ms,
        next_word_stream_start_ms=next_start_ms,
        word_start_index=word_index,
        word_end_index=word_index,
    )


def test_clean_word_between_swears_is_not_muted():
    # 'бля какой даун' — 'какой' (не мат) стоит СЛОВОМ между двумя матами.
    # По таймингам разрыв бля->даун мал (<600мс, 'какой' короткое), старая
    # gap-склейка проглотила бы 'какой'. Критерий соседства по индексу слова:
    # бля=w10, какой=w11 (чистое, нет матча), даун=w12 — индексы 10 и 12 НЕ
    # подряд => два отдельных мьюта, 'какой' не мьютится.
    # бля=w10 (следующее слово 'какой' стартует 5320), какой=w11 (чистое, нет
    # матча), даун=w12 (предыдущее слово 'какой' кончается 5450) — индексы 10 и
    # 12 НЕ подряд => два отдельных мьюта; клампы padding по границам 'какой'
    # не дают мьютам залезть на него.
    blya = _match_at("бля", 5240, 5320, segment_id="seg_1", word_index=10, next_start_ms=5320)
    daun = _match_at("даун", 5450, 5730, segment_id="seg_1", word_index=12, prev_end_ms=5450)
    doc = build_decisions(
        stream_path=Path("stream.mp4"),
        original_path=Path("original.mp4"),
        range_in=parse_timecode("00:00:00"),
        range_out=parse_timecode("00:10:00"),
        matches=[blya, daun],
        config=PipelineConfig(mute_join_gap_ms=600),
    )
    mutes = doc["mutes"]
    assert len(mutes) == 2, "чистое слово между матами => два отдельных мьюта"
    # Ни один мьют не должен покрывать интервал слова 'какой' (5.32..5.45).
    for m in mutes:
        assert not (m["start"] <= 5.38 <= m["end"]), "'какой' не должен быть в мьюте"


def test_gap_split_does_not_mute_clean_speech_between_distant_swears():
    # Два мата в одном сегменте, между ними ~4с чистой речи (несколько слов).
    # Старая группировка слепила бы их в один мьют first..last и
    # проглотила бы всю речь между ними. По индексам слов w5 и w12 не подряд.
    first = _match_at("блять", 1000, 1300, segment_id="seg_1", word_index=5)
    second = _match_at("блять", 5300, 5600, segment_id="seg_1", word_index=12)
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
    # 'бля блять нахуй' одним выкриком — слова идут ВПЛОТНУЮ (индексы подряд
    # w5,w6,w7, нет чистого слова между), склеиваются в один мьют.
    a = _match_at("бля", 1000, 1200, segment_id="seg_1", word_index=5)
    b = _match_at("блять", 1250, 1500, segment_id="seg_1", word_index=6)
    c = _match_at("нахуй", 1550, 1850, segment_id="seg_1", word_index=7)
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


def test_pause_between_adjacent_swears_splits_the_mute():
    # Соседство по индексу слова НЕ значит «вплотную»: в паузе слов нет, поэтому
    # индексы соседние даже через секунды тишины. На реальном стриме «блять»
    # 337.806 (w898) и «блять» 341.488 (w899) слиплись в мьют длиной 4.1 с, а
    # поджатие по звуку потом оставило только второе слово — первый мат остался
    # слышен. Разрыв больше mute_join_gap_ms => две отдельные группы.
    first = _match_at("блять", 337_806, 338_046, segment_id="seg_19", word_index=898)
    second = _match_at("блять", 341_488, 341_769, segment_id="seg_19", word_index=899)
    doc = build_decisions(
        stream_path=Path("stream.mp4"),
        original_path=Path("original.mp4"),
        range_in=parse_timecode("00:00:00"),
        range_out=parse_timecode("00:10:00"),
        matches=[first, second],
        config=PipelineConfig(mute_join_gap_ms=600),
    )
    mutes = doc["mutes"]
    assert len(mutes) == 2
    assert all(m["end"] - m["start"] < 1.0 for m in mutes)


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


def _one_mute(matches):
    doc = build_decisions(
        stream_path=Path("stream.mp4"),
        original_path=Path("original.mp4"),
        range_in=parse_timecode("00:00:00"),
        range_out=parse_timecode("00:10:00"),
        matches=matches,
        config=PipelineConfig(mute_join_gap_ms=600),
    )
    assert len(doc["mutes"]) == 1
    return doc["mutes"][0]


def _mute_with(cross_ms: int, *, padding_before_ms: int = 140):
    """Один мьют на слове 5.000..5.300, у которого предыдущее слово кончается
    РОВНО на его начале — типичная сетка ASR в связной речи."""
    match = _match_at("блять", 5000, 5300, word_index=7, prev_end_ms=5000)
    doc = build_decisions(
        stream_path=Path("stream.mp4"),
        original_path=Path("original.mp4"),
        range_in=parse_timecode("00:00:00"),
        range_out=parse_timecode("00:10:00"),
        matches=[match],
        config=PipelineConfig(
            mute_padding_before_ms=padding_before_ms,
            mute_head_cross_prev_ms=cross_ms,
        ),
    )
    assert len(doc["mutes"]) == 1
    return doc["mutes"][0]


def test_neighbour_clamp_eats_head_padding_when_words_are_contiguous():
    """Исходное поведение и причина ручки: сетка слов идёт вплотную, поэтому
    клэмп по концу предыдущего слова съедает ВЕСЬ головной padding.

    Это делало mute_padding_before_ms бездействующим ровно в связной речи —
    именно там, где человек потом расширял голову руками (12 из 12 расширений на
    vod_2847020280_20260817_2330)."""
    mute = _mute_with(0)
    assert mute["start"] == 5.0, "padding 140 мс схлопнулся в ноль — так и было"
    assert mute["head_cross_ms"] == 0


def test_head_padding_may_cross_into_previous_word_by_the_allowance():
    """С допуском голова заходит на разметку соседа ровно на разрешённые мс."""
    mute = _mute_with(100)
    assert mute["start"] == 4.9
    assert mute["head_cross_ms"] == 100


def test_head_cross_never_exceeds_the_padding_itself():
    """Допуск — это снятие клэмпа, а не второй padding: больше
    mute_padding_before_ms голова не растёт даже при огромном допуске."""
    mute = _mute_with(5000, padding_before_ms=140)
    assert mute["start"] == 4.86
    assert mute["head_cross_ms"] == 140


def test_solid_match_next_to_soft_fragment_is_still_muted():
    """Твёрдый матч в группе — мьют, даже если рядом мягкий обрубок.

    Раньше статус решал any(needs_review), и на vod_2847020280 группа «бля» +
    «ь» (surface + short_fragment) уходила в review — то есть мат оставался
    слышен, пока человек не переключит статус руками. Янтарный флаг при этом
    остаётся: 'accepted' + needs_review=True.
    """
    solid = _match_at("бля", 1000, 1200, word_index=5)
    soft = _match_at(
        "ь", 1250, 1350, word_index=6, match_type="short_fragment", needs_review=True
    )
    mute = _one_mute([solid, soft])
    assert mute["matched_token_count"] == 2
    assert mute["status"] == "accepted"
    assert mute["needs_review"] is True
    assert mute["match_type"] == "short_fragment+surface"


def test_group_of_only_soft_fragments_goes_to_review():
    """Обратная защита: группа, в которой кроме мягких матчей ничего нет, в
    экспорт по умолчанию не идёт (это те самые ложные 'её/ей/бы')."""
    soft = _match_at(
        "ей", 1000, 1200, word_index=5, match_type="short_fragment", needs_review=True
    )
    mute = _one_mute([soft])
    assert mute["status"] == "review"
    assert mute["needs_review"] is True
