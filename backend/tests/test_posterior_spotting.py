"""Тесты второго детектора: мат по посториорам CTC (`detect_posterior_profanity`).

Здесь проверяется не поиск (он в `test_keyword_spotting.py`), а стык: словарь
ключей, перевод находки в ProfanityMatch с контекстом транскрипта и правило
«дубли текстового детектора наружу не выходят». Посториоры подделаны: модель и
аудио не нужны.
"""

from __future__ import annotations

import numpy as np
import pytest

from twitch_cut.keyword_spotting import MIN_KEYWORD_LEN
from twitch_cut.posteriors import Posteriors, SegmentPosteriors
from twitch_cut.profanity import (
    PREFIX_ONLY_ROOTS,
    SPOT_KEYWORDS,
    ProfanityMatch,
    RussianNormalizer,
    detect_posterior_profanity,
    detect_profanity,
    load_banwords,
    roots_in,
)
from twitch_cut.timecode import TimeSpan, parse_timecode

VOCAB = ["е", "б", "а", "н", "п", "и", "с", "р", " "]
BLANK = len(VOCAB)
FRAME_SHIFT = 0.04


def _emission(n_frames: int, spikes: dict[int, tuple[str, float]]) -> np.ndarray:
    """[T, C] log-probs: почти везде blank, на кадрах из `spikes` — своя буква."""
    n_classes = len(VOCAB) + 1
    filler = 0.02
    probs = np.full((n_frames, n_classes), filler / (n_classes - 1), dtype=np.float32)
    probs[:, BLANK] = 1.0 - filler
    for frame, (char, prob) in spikes.items():
        probs[frame] = filler / (n_classes - 2)
        probs[frame, VOCAB.index(char)] = prob
        probs[frame, BLANK] = 1.0 - filler - prob
    return np.log(probs)


def _posteriors(emission, *, start: float) -> Posteriors:
    return Posteriors(
        vocab=VOCAB,
        blank_id=BLANK,
        segments=[
            SegmentPosteriors(
                index=0, start=start, frame_shift=FRAME_SHIFT, log_probs=emission
            )
        ],
    )


def _quiet_eban(*, first: int = 5) -> np.ndarray:
    """«ебан» тихими пиками: argmax на этих кадрах — blank, то есть в ТЕКСТЕ
    распознавания корня нет и найти его может только спотинг."""
    return _emission(40, {first + i * 2: (char, 0.35) for i, char in enumerate("ебан")})


def test_spot_keywords_are_long_enough_to_search():
    """Короткие ключи запрещены в самом поиске — в словаре их быть не должно,
    иначе спотинг молча искал бы не то, что написано в списке."""
    assert SPOT_KEYWORDS
    assert all(len(key) >= MIN_KEYWORD_LEN for key in SPOT_KEYWORDS)


def test_every_spot_keyword_is_profanity_for_the_text_detector():
    """Один источник правды: всё, что ищется по посториорам, текстовый детектор
    обязан считать матом. Разъедься эти два списка — стадия предлагала бы
    человеку слова, которые сама же считает чистыми."""
    for key in SPOT_KEYWORDS:
        assert roots_in(key), key


def test_prefix_only_roots_are_not_searched_in_posteriors():
    """У prefix-only корня есть невинные омографы (рецепт, команда, скипидар), а
    у спотинга нет ни начала слова, ни словаря — отличить он их не может.
    Поэтому такие корни в посториорах не ищутся вообще."""
    for root in PREFIX_ONLY_ROOTS:
        assert root not in SPOT_KEYWORDS, root


def test_epta_is_caught_by_the_text_detector(tmp_path):
    """Дырка словаря, найденная на реальном прогоне: «ёпта» доезжала до текста, а
    корня под неё не было. «епт» — prefix-only: ни одна из 5.14 млн словоформ с
    него не начинается, зато внутри слова он сидит в рецепте и шептать."""
    banwords = tmp_path / "banwords.txt"
    banwords.write_text("", encoding="utf-8")
    normalizer = RussianNormalizer()
    entries = load_banwords(banwords, normalizer)
    transcript = {
        "segments": [
            {
                "id": "seg_1",
                "words": [
                    {"word": "ёпта", "start": 1.0, "end": 1.3, "score": 0.9},
                    {"word": "рецепт", "start": 1.5, "end": 1.9, "score": 0.9},
                    {"word": "шептать", "start": 2.0, "end": 2.4, "score": 0.9},
                ],
            }
        ]
    }

    matches = detect_profanity(transcript, entries, TimeSpan(0), normalizer)

    assert [m.word for m in matches] == ["ёпта"]
    assert matches[0].banword == "<stem:епт>"


def test_posterior_candidate_carries_context_and_goes_to_review():
    """Находка спотинга должна выглядеть как обычный матч: с segment_id,
    границами фразы и соседними словами (без них окно мьюта считалось бы иначе,
    чем у текстовых матчей). Но статусом — только кандидат для человека."""
    transcript = {
        "segments": [
            {
                "id": "seg_7",
                "start": 10.0,
                "end": 13.0,
                "words": [
                    {"word": "вот", "start": 10.0, "end": 10.2, "score": 0.9},
                    {"word": "епанисир", "start": 10.2, "end": 10.6, "score": 0.4},
                    {"word": "и", "start": 11.0, "end": 11.1, "score": 0.9},
                ],
            }
        ]
    }

    matches = detect_posterior_profanity(
        _posteriors(_quiet_eban(), start=10.2),
        transcript,
        parse_timecode("00:01:00"),
        min_score=0.3,
    )

    assert [m.banword for m in matches] == ["<spot:ебан>"]
    match = matches[0]
    assert match.match_type == "posterior"
    assert match.needs_review is True
    assert match.timing_source == "posterior+align"
    # Что услышало распознавание на этом месте — человеку в списке мьютов.
    assert match.word == "епанисир"
    assert match.segment_id == "seg_7"
    assert match.segment_local_start_ms == 10_000
    assert match.prev_word_local_end_ms == 10_200
    assert match.next_word_local_start_ms == 11_000
    # Абсолютное время: начало сегмента посториоров + границы forced_align по
    # подставленной гипотезе + range_in. Кадры эмиссии дали бы 10_400: первая
    # буква вспыхивает на кадре 5, а начало слова лежит раньше всплеска.
    assert match.local_start_ms == 10_320
    assert match.stream_start_ms == 70_320


def test_candidate_on_top_of_a_text_match_is_dropped():
    """Спотинг видит те же слова, что и текст: без этого правила на каждый
    честно найденный мат приезжал бы его дубль-review (на реальном прогоне ~75
    дублей против одной новой находки)."""
    transcript = {"segments": [{"id": "seg_1", "start": 0.0, "end": 3.0, "words": []}]}
    known = ProfanityMatch(
        word="ебанский",
        normalized="ебанский",
        lemma="ебанский",
        banword="<stem:еб>",
        banword_lemma="еб",
        local_start_ms=200,
        local_end_ms=700,
        stream_start_ms=200,
        stream_end_ms=700,
        confidence=0.9,
        segment_id="seg_1",
        match_type="stem",
        timing_source="whisper",
    )

    args = (_posteriors(_quiet_eban(), start=0.0), transcript, TimeSpan(0))
    assert [m.banword for m in detect_posterior_profanity(*args, min_score=0.3)] == [
        "<spot:ебан>"
    ]
    assert detect_posterior_profanity(*args, min_score=0.3, known=[known]) == []


def test_candidate_in_a_gap_of_the_transcript_survives():
    """Спотинг умеет и там, где транскрипт молчит (ASR выбросил слово целиком).
    Тогда показывать нечего, кроме самого ключа, — но кандидат обязан доехать."""
    matches = detect_posterior_profanity(
        _posteriors(_quiet_eban(), start=100.0),
        {"segments": []},
        TimeSpan(0),
        min_score=0.3,
    )

    assert [m.word for m in matches] == ["ебан?"]
    # Сегмента нет — id не выдумываем: по нему склеиваются мьюты одной фразы.
    assert matches[0].segment_id is None
    assert matches[0].local_start_ms == 100_120


def test_candidate_bounds_come_from_forced_align_and_spare_the_neighbours():
    """Ключ подставляется в гипотезу ОТДЕЛЬНЫМ словом, поэтому его границы не
    растекаются на соседей: слева громкие «ср», справа «пи» — greedy их слышит,
    в гипотезе они остаются, и forced_align обязан отдать ключу только его
    кадры. Без пробелов вокруг ключа буквы соседей слиплись бы с ним в одно
    слово и окно кандидата уехало бы на чужую речь."""
    spikes = {0: ("с", 0.9), 2: ("р", 0.9), 30: ("п", 0.9), 32: ("и", 0.9)}
    spikes.update({10 + i * 2: (char, 0.35) for i, char in enumerate("ебан")})

    matches = detect_posterior_profanity(
        _posteriors(_emission(40, spikes), start=0.0),
        {"segments": []},
        TimeSpan(0),
        min_score=0.3,
    )

    assert [m.banword for m in matches] == ["<spot:ебан>"]
    match = matches[0]
    assert match.timing_source == "posterior+align"
    # Всплески букв ключа — кадры 10–16, то есть 400–680 мс: окно обязано их
    # накрыть...
    assert match.local_start_ms <= 400
    assert match.local_end_ms >= 680
    # ...и не залезть на соседей: «р» гаснет на кадре 3, «п» вспыхивает на 30.
    assert match.local_start_ms >= 120
    assert match.local_end_ms <= 1_200


def test_candidate_falls_back_to_emission_frames_without_a_space_in_the_vocab():
    """Откат обязан работать молча. Нет пробела в словаре — подставить ключ
    отдельным словом нечем, и границы остаются по кадрам эмиссии первой и
    последней буквы. Прогон не рушится, а `timing_source` честно говорит,
    откуда тайминги, — по нему audit не диагностирует чужую проблему."""
    posteriors = Posteriors(
        vocab=[char if char != " " else "_" for char in VOCAB],
        blank_id=BLANK,
        segments=[
            SegmentPosteriors(
                index=0, start=100.0, frame_shift=FRAME_SHIFT, log_probs=_quiet_eban()
            )
        ],
    )

    matches = detect_posterior_profanity(
        posteriors, {"segments": []}, TimeSpan(0), min_score=0.3
    )

    assert [m.timing_source for m in matches] == ["posterior"]
    assert matches[0].local_start_ms == 100_200  # кадр 5 × 40 мс


def test_silence_gives_no_candidates():
    """Порог обязан молчать на тишине: иначе стадия завалит человека
    кандидатами на пустом месте."""
    assert (
        detect_posterior_profanity(
            _posteriors(_emission(60, {}), start=0.0),
            {"segments": []},
            TimeSpan(0),
            min_score=0.3,
        )
        == []
    )
