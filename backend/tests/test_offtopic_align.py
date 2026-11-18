"""Фаза 2: привязка стрима к исходнику и расщепление дорожек.

Проверяем ровно то, на чём держится вся схема: слова, которые есть в исходнике,
уходят дорожке ВИДЕО, остальные — СТРИМЕРУ, а `pos_at` восстанавливает таймлайн
воспроизведения. Если привязка ломается молча, оффтопом становится весь VOD —
поэтому тесты есть и на вырожденные входы.
"""

from __future__ import annotations

from twitch_cut.llm_offtopic.align import (
    SHINGLE_N,
    Word,
    align_transcripts,
    align_words,
    build_shingle_index,
    normalize_word,
    transcript_words,
)
from twitch_cut.llm_offtopic.turns import smooth, split_tracks


def _words(text: str, *, start: float = 0.0, step: float = 0.5) -> list[Word]:
    """«раз два три» → поток Word с шагом step секунд."""
    out = []
    for i, raw in enumerate(text.split()):
        t = start + i * step
        out.append(Word(text=normalize_word(raw), raw=raw, start_s=t, end_s=t + step / 2))
    return out


def _transcript(text: str, *, start: float = 0.0, step: float = 0.5) -> dict:
    ws = _words(text, start=start, step=step)
    return {
        "segments": [
            {
                "start": ws[0].start_s,
                "end": ws[-1].end_s,
                "words": [
                    {"word": w.raw, "start": w.start_s, "end": w.end_s} for w in ws
                ],
            }
        ]
    }


# --- нормализация и разбор транскрипта --------------------------------------


def test_normalize_word_strips_punctuation_and_case():
    assert normalize_word("«Айфон,»") == "айфон"
    assert normalize_word("Ку-ку!") == "куку"
    assert normalize_word("iPhone") == "iphone"
    assert normalize_word("—") == ""


def test_transcript_words_skips_wordless_and_timeless_entries():
    t = {
        "segments": [
            {
                "words": [
                    {"word": "привет", "start": 1.0, "end": 1.5},
                    {"word": "…", "start": 2.0, "end": 2.1},  # без букв
                    {"word": "мир"},  # без таймингов
                    {"word": "снова", "start": 0.5, "end": 0.9},
                ]
            }
        ]
    }
    ws = transcript_words(t)
    assert [w.text for w in ws] == ["снова", "привет"]  # отсортировано по времени


def test_shingle_index_positions():
    idx = build_shingle_index(_words("а б в г д"), SHINGLE_N)
    assert idx[("а", "б", "в", "г")] == [0]
    assert idx[("б", "в", "г", "д")] == [1]


# --- привязка ---------------------------------------------------------------


def test_matched_words_belong_to_video_track():
    source = _words("кот сидел на подоконнике и смотрел в окно")
    # Стример повторяет фразу видео, потом говорит своё.
    stream = _words("кот сидел на подоконнике и смотрел в окно а у меня чайник вскипел")

    al = align_words(stream, source)

    matched_text = [w.text for w, m in zip(al.stream, al.matched) if m]
    own_text = [w.text for w, m in zip(al.stream, al.matched) if not m]
    assert matched_text == ["кот", "сидел", "на", "подоконнике", "и", "смотрел", "в", "окно"]
    assert own_text == ["а", "у", "меня", "чайник", "вскипел"]
    assert al.matched_ratio > 0.6


def test_pos_at_reconstructs_playback_timeline():
    source = _words("раз два три четыре пять шесть семь восемь")
    # Видео играет с 100-й секунды стрима, начиная с 5-го слова исходника.
    stream = _words("пять шесть семь восемь", start=100.0)

    al = align_words(stream, source)

    assert al.pos_at == [(100.0, 4)]
    assert al.anchors_in(99.0, 101.0) == [4]
    assert al.anchors_in(200.0, 300.0) == []
    assert al.last_anchor_before(150.0) == 4
    assert al.last_anchor_before(50.0) is None


def test_unrelated_streams_have_no_anchors():
    al = align_words(_words("совершенно другие слова тут"), _words("кот сидел на окне"))
    assert al.matched_ratio == 0.0
    assert al.pos_at == []
    assert al.anchored_ratio(0.0, 100.0) == 0.0


def test_short_source_yields_empty_alignment():
    # Исходник короче шингла — привязки нет, но и падения нет.
    al = align_words(_words("раз два три четыре пять"), _words("раз два"))
    assert al.matched == [False] * 5
    assert al.pos_at == []


def test_anchored_ratio_is_local():
    source = _words("кот сидел на подоконнике и смотрел в окно")
    stream = _words(
        "кот сидел на подоконнике и смотрел в окно "  # 0.0–3.5с: видео
        "а у меня чайник вскипел пойду налью чаю сейчас",  # 4.0–8.5с: своё
    )
    al = align_words(stream, source)
    assert al.anchored_ratio(0.0, 3.5) == 1.0
    assert al.anchored_ratio(4.0, 8.5) == 0.0


def test_align_transcripts_entry_point():
    al = align_transcripts(_transcript("раз два три четыре пять"), _transcript("раз два три четыре"))
    assert sum(al.matched) == 4


def test_source_text_clamps_and_trims():
    al = align_words(_words("раз два"), _words("раз два три четыре пять"))
    assert al.source_text(0, 3) == "раз два три"
    assert al.source_text(-10, 99) == "раз два три четыре пять"  # границы клампятся
    assert al.source_text(0, 5, max_chars=7) == "раз два"
    assert al.source_text(0, 5, max_chars=4, tail=True) == "пять"


# --- сглаживание и реплики --------------------------------------------------


def test_smooth_erases_short_runs_inside():
    # Одиночный «остров» True внутри False — ошибка ASR, не смена говорящего.
    assert smooth([False] * 4 + [True] + [False] * 4, min_run=3) == [False] * 9
    # Прогон длиной min_run остаётся.
    assert smooth([False, False, False, True, True, True, False, False, False]) == [
        False, False, False, True, True, True, False, False, False
    ]


def test_smooth_keeps_edge_runs():
    # У края нет «предыдущего» прогона: затирать нечем и незачем.
    assert smooth([True, False, False, False, False], min_run=3)[0] is True


def test_split_tracks_returns_streamer_turns_only():
    source = _words("кот сидел на подоконнике и смотрел в окно")
    stream = _words(
        "кот сидел на подоконнике и смотрел в окно "
        "слушайте а у меня тут чайник вскипел пойду налью себе чаю"
    )
    split = split_tracks(align_words(stream, source))

    assert split.video_words == 8
    assert split.streamer_words == split.total_words - 8
    assert len(split.turns) == 1
    turn = split.turns[0]
    assert turn.text.startswith("слушайте а у меня")
    assert "подоконнике" not in turn.text  # речь видео в реплику не попала
    assert turn.word_count == 11
    assert turn.duration_s > 0


def test_short_turns_are_dropped():
    source = _words("раз два три четыре пять шесть семь восемь девять")
    # «ага» между двумя фразами видео — не реплика, судить нечего.
    stream = _words("раз два три четыре ага пять шесть семь восемь девять")
    split = split_tracks(align_words(stream, source), min_run=1)
    assert split.turns == []
    assert split.dropped_short_turns == 1


def test_turn_survives_short_video_insert():
    source = _words("кот сидел на подоконнике")
    # Стример говорит, сквозь него проходит фраза видео — это ОДНА реплика.
    stream = (
        _words("значит смотрите какая тут история", start=0.0)
        + _words("кот сидел на подоконнике", start=2.5)
        + _words("вот примерно так это и было", start=4.5)
    )
    split = split_tracks(align_words(stream, source))
    assert len(split.turns) == 1
    assert split.merged_turns == 1
    assert "значит" in split.turns[0].text and "примерно" in split.turns[0].text
