"""Тесты `alignment.align_words` — границы слов по пути forced alignment.

Проверять надо на бумаге, поэтому здесь синтетические log-probs: CTC пиковый,
значит эмиссия — это одиночные всплески на фоне blank. Расставляем всплески
руками и знаем правильный ответ заранее. Модель и аудио не нужны.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from twitch_cut.alignment import MAX_EDGE_EXTEND_MS, align_words, forced_align, merge_tokens

VOCAB = ["а", "б", "в", "г", " "]
BLANK = len(VOCAB)
FRAME_SHIFT = 0.04  # как у GigaAM v3: 40 мс на кадр энкодера


def to_char(token_id: int) -> str:
    return VOCAB[token_id]


def _emission(n_frames: int, spikes: dict[int, int]) -> np.ndarray:
    """[T, C] log-probs: blank почти везде, на кадрах из `spikes` — своя буква."""
    n_classes = len(VOCAB) + 1
    probs = np.full((n_frames, n_classes), 0.02 / (n_classes - 1), dtype=np.float32)
    probs[:, BLANK] = 0.98
    for frame, token_id in spikes.items():
        probs[frame] = 0.02 / (n_classes - 2)
        probs[frame, token_id] = 0.9
        probs[frame, BLANK] = 0.08
    return np.log(probs)


def _two_words(gap_start: int = 12, n_frames: int = 20):
    """«аб» (всплески 2 и 4), пробел (7) и «вг» (gap_start, gap_start+1)."""
    spikes = {2: 0, 4: 1, 7: 4, gap_start: 2, gap_start + 1: 3}
    ids = [0, 1, 4, 2, 3]
    return _emission(n_frames, spikes), ids


def _align(emission, ids, **kwargs):
    return align_words(
        emission,
        ids,
        blank_id=BLANK,
        to_char=to_char,
        frame_shift=FRAME_SHIFT,
        **kwargs,
    )


def test_blank_frames_belong_to_the_word():
    """Главное отличие от кадров эмиссии: провал blank вокруг всплеска — это
    тоже звук слова, поэтому окно ШИРЕ, а не равно точкам эмиссии."""
    words = _align(*_two_words())

    assert [w.text for w in words] == ["аб", "вг"]
    # «аб»: всплески 2..4 → путь [2,5); слева провал 2 кадра → +1, справа 7 → +3.
    assert words[0].start == pytest.approx((2 - 1) * FRAME_SHIFT)
    assert words[0].end == pytest.approx((5 + 3) * FRAME_SHIFT)
    # Кадры эмиссии дали бы 0.08..0.20 — то есть на 40 и 120 мс уже.
    assert words[0].start < 2 * FRAME_SHIFT
    assert words[0].end > (4 + 1) * FRAME_SHIFT


def test_neighbours_cannot_overlap():
    """Расширяемся не больше чем на ПОЛОВИНУ провала — иначе мьют одного слова
    залезет на соседнее чистое, и человек будет откатывать границы обратно."""
    words = _align(*_two_words())

    assert words[0].end <= words[1].start
    assert words[1].start == pytest.approx((12 - 3) * FRAME_SHIFT)


def test_long_pause_is_not_swallowed():
    """Перед секундной паузой слово не должно «съесть» тишину: расширение
    ограничено MAX_EDGE_EXTEND_MS, а не половиной провала."""
    words = _align(*_two_words(gap_start=40, n_frames=60))

    cap_frames = round(MAX_EDGE_EXTEND_MS / 1000 / FRAME_SHIFT)
    assert words[0].end == pytest.approx((5 + cap_frames) * FRAME_SHIFT)
    assert words[1].start == pytest.approx((40 - cap_frames) * FRAME_SHIFT)


def test_word_score_appears_where_gigaam_had_none():
    """У GigaAM в транскрипте `score: null`. Путь forced_align даёт скор на
    каждый кадр — отсюда настоящая уверенность по слову."""
    words = _align(*_two_words())

    for word in words:
        assert word.score is not None
        assert 0.5 < word.score <= 1.0
    assert words[0].score == pytest.approx(0.9, abs=0.05)


def test_boundaries_stay_inside_the_segment():
    """Всплеск на первом и последнем кадре: расширять некуда, за пределы
    сегмента выходить нельзя (иначе мьют уедет в чужой VAD-фрагмент)."""
    emission = _emission(4, {0: 0, 3: 1})
    words = _align(emission, [0, 1])

    assert len(words) == 1
    assert words[0].start == 0.0
    assert words[0].end == pytest.approx(4 * FRAME_SHIFT)


def test_no_tokens_is_empty_not_a_crash():
    assert _align(_emission(5, {}), []) == []


def test_impossible_alignment_falls_back_instead_of_raising():
    """Букв больше, чем кадров — выровнять нельзя. Контракт: None, чтобы
    вызывающий код взял тайминги по кадрам эмиссии, а прогон не рухнул."""
    assert _align(_emission(2, {0: 0, 1: 1}), [0, 1, 2, 3]) is None


def test_frame_shift_scales_everything():
    """frame_shift считается на каждый сегмент (аудио-семплы / кадры энкодера),
    поэтому границы обязаны быть линейны по нему, а не по константе 40 мс."""
    emission, ids = _two_words()
    slow = align_words(
        emission, ids, blank_id=BLANK, to_char=to_char, frame_shift=0.08,
        max_extend_ms=MAX_EDGE_EXTEND_MS * 2,
    )
    fast = _align(emission, ids)

    assert slow is not None and fast is not None
    for a, b in zip(slow, fast):
        assert a.start == pytest.approx(b.start * 2)
        assert a.end == pytest.approx(b.end * 2)


def test_new_window_contains_the_old_one():
    """Инвариант против регрессии: новая граница обязана быть НЕ УЖЕ старой —
    иначе мы теряем то, что кадры эмиссии уже покрывали."""
    emission, ids = _two_words()
    words = _align(emission, ids)
    old = [(2 * FRAME_SHIFT, 5 * FRAME_SHIFT), (12 * FRAME_SHIFT, 14 * FRAME_SHIFT)]

    for word, (start, end) in zip(words, old):
        assert word.start <= start + 1e-9 and word.end >= end - 1e-9


def test_probabilities_are_real_not_logs():
    """Скор — вероятность 0..1, а не log-prob: в JSON транскрипта поле `score`
    читает фронт и downstream, минус там означал бы «сломано»."""
    emission = _emission(6, {1: 0, 3: 1})
    words = _align(emission, [0, 1])

    assert words[0].score == pytest.approx(math.exp(math.log(0.9)), abs=1e-6)


# --- сам Витерби (forced_align / merge_tokens) ------------------------------


def test_repeated_letter_needs_a_blank_between():
    """«аа» в CTC — две буквы, только если между ними blank; на двух кадрах
    такую гипотезу не уложить, на трёх — путь обязан пройти через blank."""
    emission = _emission(3, {0: 0, 2: 0})
    with pytest.raises(ValueError):
        forced_align(emission[:1], [0, 0], blank=BLANK)

    path, _ = forced_align(emission, [0, 0], blank=BLANK)
    assert path.tolist() == [0, BLANK, 0]


def test_blank_in_hypothesis_is_rejected():
    with pytest.raises(ValueError):
        forced_align(_emission(5, {}), [0, BLANK], blank=BLANK)


def test_merge_tokens_drops_blanks_and_averages_scores():
    path = np.array([BLANK, 0, 0, BLANK, 1, BLANK])
    scores = np.log(np.array([0.9, 0.8, 0.6, 0.9, 0.5, 0.9], dtype=np.float32))
    spans = merge_tokens(path, scores, blank=BLANK)

    assert [(s.token, s.start, s.end) for s in spans] == [(0, 1, 3), (1, 4, 5)]
    assert spans[0].score == pytest.approx((math.log(0.8) + math.log(0.6)) / 2, abs=1e-6)


def test_matches_torchaudio_frame_for_frame():
    """Путь и скоры — как у torchaudio.functional.forced_align: те же ничьи,
    то же окно допустимых состояний. Нужен torch (dev-набор `[export]`)."""
    torch = pytest.importorskip("torch")
    ta = pytest.importorskip("torchaudio.functional")

    rng = np.random.default_rng(0)
    for _ in range(400):
        n_classes = int(rng.integers(3, 35))
        blank = n_classes - 1
        n_frames = int(rng.integers(1, 80))
        n_tokens = int(rng.integers(1, n_frames + 1))
        targets = rng.integers(0, blank, size=n_tokens)
        if rng.random() < 0.3 and n_tokens > 1:
            targets[1::2] = targets[0::2][: len(targets[1::2])]
        logits = rng.normal(size=(n_frames, n_classes)).astype(np.float32) * 3
        if rng.random() < 0.2:
            logits = np.round(logits)  # ничьи
        emission = torch.log_softmax(torch.from_numpy(logits), -1)
        try:
            ref_path, ref_scores = ta.forced_align(
                emission[None], torch.tensor([targets.tolist()], dtype=torch.int32), blank=blank
            )
        except RuntimeError:
            with pytest.raises(ValueError):
                forced_align(emission.numpy(), targets.tolist(), blank=blank)
            continue
        path, scores = forced_align(emission.numpy(), targets.tolist(), blank=blank)
        np.testing.assert_array_equal(path, ref_path[0].numpy())
        np.testing.assert_array_equal(scores, ref_scores[0].numpy())
