"""Тесты `posteriors` — запись и чтение log-probs рядом с транскриптом.

Файл посториоров — это кэш, который переживает прогон и читается ДРУГОЙ стадией,
поэтому проверяем именно контракт: что положили, то и достали, время сегментов не
съехало, а битый или неполный вход не превращается в тихо неверные данные.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from twitch_cut.posteriors import read_posteriors, to_float16, write_posteriors

np = pytest.importorskip("numpy")

VOCAB = ["а", "б", "в", " "]
BLANK = len(VOCAB)


def _chunk(n_frames: int, seed: int) -> "np.ndarray":
    """Правдоподобные log-probs: log-softmax по случайным логитам."""
    rng = np.random.default_rng(seed)
    logits = rng.normal(size=(n_frames, len(VOCAB) + 1)).astype(np.float32)
    return logits - np.log(np.exp(logits).sum(axis=1, keepdims=True))


def _write(path: Path, chunks, starts):
    return write_posteriors(path, chunks, starts=starts, vocab=VOCAB, blank_id=BLANK)


def test_round_trip_keeps_segments_apart(tmp_path: Path):
    """Сегменты склеены в один массив, поэтому главное — что `offsets` режут его
    обратно ровно там же: иначе кадры одного сегмента уедут в другой."""
    first, second = _chunk(7, 1), _chunk(3, 2)

    _write(tmp_path / "p.npz", [(first, 0.04), (second, 0.05)], [0.0, 12.5])
    got = read_posteriors(tmp_path / "p.npz")

    assert [s.log_probs.shape[0] for s in got.segments] == [7, 3]
    assert [s.start for s in got.segments] == [0.0, 12.5]
    assert [s.frame_shift for s in got.segments] == pytest.approx([0.04, 0.05])
    assert np.asarray(got.segments[1].log_probs) == pytest.approx(second, abs=0.01)


def test_float16_does_not_move_the_score():
    """На диске float16: шаг сетки в диапазоне log-prob около 0.01, то есть
    меньше 1% по вероятности — порог спотинга такого не замечает."""
    chunk = _chunk(64, 3)

    back = np.asarray(to_float16(chunk), dtype=np.float32)

    assert np.abs(np.exp(back) - np.exp(chunk)).max() < 0.005


def test_letters_map_to_their_token_ids(tmp_path: Path):
    """Спотингу нужен тот же словарь, что был у модели: иначе он будет искать
    буквы по чужим индексам и найдёт что угодно. `char_to_id` — единственный
    способ спотинга узнать индекс буквы."""
    _write(tmp_path / "p.npz", [(_chunk(4, 4), 0.04)], [1.0])

    got = read_posteriors(tmp_path / "p.npz")

    assert got.blank_id == BLANK
    assert [got.char_to_id(c) for c in "абв "] == [0, 1, 2, 3]
    assert got.char_to_id("z") is None


def test_segments_without_starts_are_not_saved(tmp_path: Path):
    """Посториоры без времени бесполезны: лучше не сохранить ничего, чем
    сохранить кадры, привязанные к чужому сегменту."""
    assert _write(tmp_path / "p.npz", [(_chunk(4, 5), 0.04)], [0.0, 1.0]) is None
    assert not (tmp_path / "p.npz").exists()


def test_missing_file_is_a_none_not_a_crash(tmp_path: Path):
    """Кэша может не быть (старый прогон, чистка) — стадия спотинга просто не
    запускается, прогон из-за этого падать не должен."""
    assert read_posteriors(tmp_path / "nope.npz") is None


def test_garbage_file_is_a_none_not_a_crash(tmp_path: Path):
    """То же для битого файла: недописанный `.npz` не должен рушить пайплайн."""
    path = tmp_path / "broken.npz"
    path.write_bytes(b"PK\x03\x04 not really an npz")

    assert read_posteriors(path) is None
