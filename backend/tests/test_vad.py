"""Тесты `vad` — VAD pyannote на onnxruntime и нарезка речи под ASR.

Модель здесь не нужна: сессию onnxruntime подменяет двойник, который
«слышит речь» там, где у сигнала большая амплитуда. Так проверяется весь
конвейер вокруг модели — окна, склейка Хэммингом, обрезка хвоста, гистерезис.
Если в окружении есть pyannote.audio (dev-набор `[export]`), склейка и
гистерезис дополнительно сверяются с оригиналом бит в бит.
"""

from __future__ import annotations

from itertools import combinations
from types import SimpleNamespace

import numpy as np
import pytest

from twitch_cut.vad import SpeechDetector, segment_audio

SR = 100  # 100 Гц хватает: модель-двойник смотрит на амплитуду, а не на звук
FRAME = 0.1  # кадр двойника — 10 сэмплов


def _powerset_mapping(num_classes: int = 3, max_set: int = 2) -> list[list[int]]:
    rows = []
    for size in range(max_set + 1):
        for subset in combinations(range(num_classes), size):
            rows.append([1 if c in subset else 0 for c in range(num_classes)])
    return rows


META = {
    "sample_rate": SR,
    "duration": 10.0,
    "step": 1.0,
    "window_samples": 10 * SR,
    "warm_up": [0.0, 0.0],
    "frame_start": 0.0,
    "frame_duration": FRAME,
    "frame_step": FRAME,
    "num_frames": 100,
    "powerset_mapping": _powerset_mapping(),
    "onset": 0.5,
    "offset": 0.5,
}


class _LoudnessModel:
    """Двойник ONNX-сессии: кадр — речь (класс «спикер 1»), если средняя
    амплитуда его сэмплов больше 0.5, иначе «никого» (класс 0)."""

    def get_inputs(self):
        return [SimpleNamespace(name="waveform")]

    def run(self, _outputs, feeds):
        batch = feeds["waveform"][:, 0, :]
        per_frame = int(FRAME * SR)
        frames = batch.reshape(batch.shape[0], -1, per_frame)
        loud = np.abs(frames).mean(axis=-1) > 0.5
        logits = np.zeros((*loud.shape, 7), dtype=np.float32)
        logits[..., 0] = np.where(loud, 0.0, 5.0)
        logits[..., 1] = np.where(loud, 5.0, 0.0)
        return [logits]


def _audio(seconds: float, speech: list[tuple[float, float]]) -> np.ndarray:
    audio = np.zeros(int(seconds * SR), dtype=np.float32)
    for start, end in speech:
        audio[int(start * SR) : int(end * SR)] = 1.0
    return audio


def test_finds_speech_where_it_is():
    detector = SpeechDetector(_LoudnessModel(), META)
    regions = detector.speech_regions(_audio(30.0, [(8.0, 14.5)]))

    assert len(regions) == 1
    start, end = regions[0]
    # Время — середины кадров (как у pyannote Binarize), отсюда допуск в кадр.
    assert start == pytest.approx(8.0, abs=FRAME)
    assert end == pytest.approx(14.5, abs=FRAME)


def test_tail_window_is_padded_and_cropped():
    """Длина не кратна шагу окна: последнее окно дополняется нулями, а то, что
    вылезло за конец аудио, отрезается — речь у самого конца не теряется и
    не тянется в несуществующее время."""
    detector = SpeechDetector(_LoudnessModel(), META)
    audio = _audio(30.5, [(27.0, 30.5)])
    regions = detector.speech_regions(audio)

    assert len(regions) == 1
    assert regions[0][0] == pytest.approx(27.0, abs=FRAME)
    assert regions[0][1] <= len(audio) / SR + FRAME


def test_audio_shorter_than_one_window():
    detector = SpeechDetector(_LoudnessModel(), META)
    regions = detector.speech_regions(_audio(4.0, [(1.0, 2.0)]))

    assert len(regions) == 1
    assert regions[0][0] == pytest.approx(1.0, abs=FRAME)
    assert regions[0][1] == pytest.approx(2.0, abs=FRAME)


def test_silence_gives_no_regions():
    detector = SpeechDetector(_LoudnessModel(), META)
    assert detector.speech_regions(_audio(25.0, [])) == []


def test_reports_chunk_progress():
    detector = SpeechDetector(_LoudnessModel(), META)
    ticks: list[tuple[int, int]] = []
    detector.speech_regions(_audio(45.5, [(5.0, 6.0)]), on_chunk=lambda d, t: ticks.append((d, t)))

    assert ticks[-1][0] == ticks[-1][1], "последний тик — все окна готовы"


# --- нарезка под ASR (gigaam.vad_utils.segment_audio_file) -----------------


def test_short_regions_are_glued_into_one_chunk():
    audio = np.zeros(60 * SR, dtype=np.float32)
    regions = [(1.0, 3.0), (4.0, 6.0), (7.0, 9.0)]
    pieces, boundaries = segment_audio(audio, regions, SR)

    assert boundaries == [(1.0, 9.0)]
    assert len(pieces[0]) == 8 * SR


def test_chunk_is_closed_before_it_outgrows_max_duration():
    """Кусок закрывается, когда следующий участок вытолкнул бы его за 22 с."""
    audio = np.zeros(80 * SR, dtype=np.float32)
    regions = [(0.0, 12.0), (13.0, 25.0), (26.0, 30.0)]
    _, boundaries = segment_audio(audio, regions, SR)

    assert boundaries == [(0.0, 12.0), (13.0, 30.0)]


def test_long_speech_is_split_evenly_under_the_hard_limit():
    """Сплошная речь длиннее 30 с режется на равные куски не длиннее 30 с."""
    audio = np.zeros(100 * SR, dtype=np.float32)
    _, boundaries = segment_audio(audio, [(0.0, 70.0)], SR)

    assert len(boundaries) == 3
    assert all(end - start <= 30.0 for start, end in boundaries)
    assert boundaries[0][0] == 0.0 and boundaries[-1][1] == pytest.approx(70.0)


def test_tiny_blip_is_dropped():
    audio = np.zeros(10 * SR, dtype=np.float32)
    assert segment_audio(audio, [(2.0, 2.1)], SR) == ([], [])


# --- сверка с pyannote, если он установлен ---------------------------------


def test_aggregate_and_binarize_match_pyannote():
    """Склейка окон и гистерезис — операция в операцию как в pyannote 3.4."""
    pytest.importorskip("pyannote.audio")
    from pyannote.audio.core.inference import Inference
    from pyannote.audio.utils.signal import Binarize
    from pyannote.core import Segment, SlidingWindow, SlidingWindowFeature

    real = {
        **META,
        "sample_rate": 16000,
        "window_samples": 160000,
        "frame_duration": 0.0619375,
        "frame_step": 0.016875,
        "num_frames": 589,
    }
    detector = SpeechDetector(_LoudnessModel(), real)
    rng = np.random.default_rng(3)
    for num_chunks in (1, 7, 40):
        # Речь кусками: случайные включения, растянутые на десятки кадров.
        raw = rng.random((num_chunks, 589)) < 0.02
        scores = (np.cumsum(raw, axis=1) % 2).astype(np.float32)[..., None]
        audio_seconds = 10.0 + (num_chunks - 1) * 1.0 - 0.37

        mine = detector._crop_loose(detector._aggregate(scores.copy()), audio_seconds)
        mine_regions = detector._binarize(mine)

        chunks = SlidingWindow(start=0.0, duration=10.0, step=1.0)
        frames = SlidingWindow(start=0.0, duration=0.0619375, step=0.016875)
        theirs = Inference.aggregate(
            SlidingWindowFeature(scores.copy(), chunks),
            frames,
            warm_up=(0.0, 0.0),
            hamming=True,
            missing=0.0,
        )
        theirs.data = theirs.crop(Segment(0.0, audio_seconds), mode="loose")
        np.testing.assert_array_equal(mine, theirs.data)

        annotation = Binarize(onset=0.5, offset=0.5)(theirs)
        expected = [(s.start, s.end) for s in annotation.get_timeline().support()]
        assert mine_regions == expected
