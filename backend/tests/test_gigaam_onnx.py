"""Тесты `gigaam_onnx` — признаки, батч и greedy-декод вокруг ONNX-графа.

Сам граф подменяет двойник сессии: проверяется всё, что раньше делал пакет
gigaam на torch и что теперь наше — длины признаков, дополнение батча,
обрезка по длинам энкодера, greedy CTC и frame_shift. Если стоит torchaudio,
лог-мел дополнительно сверяется с ним.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from twitch_cut.gigaam_onnx import GigaAMOnnx, LogMel, greedy_tokens, mel_filterbank

VOCAB = [" ", "а", "б", "в"]
BLANK = len(VOCAB)
META = {
    "sample_rate": 16000,
    "features": 64,
    "n_fft": 320,
    "win_length": 320,
    "hop_length": 160,
    "center": False,
    "vocab": VOCAB,
    "blank_id": BLANK,
}


def test_greedy_collapses_repeats_and_drops_blank():
    labels = [BLANK, 1, 1, BLANK, 1, 2, 2, BLANK, 0, 3]
    log_probs = np.full((len(labels), BLANK + 1), -10.0, dtype=np.float32)
    log_probs[np.arange(len(labels)), labels] = 0.0

    ids, frames = greedy_tokens(log_probs, BLANK)

    assert ids == [1, 1, 2, 0, 3]
    assert frames == [1, 4, 5, 8, 9], "кадр — ПЕРВЫЙ кадр всплеска"


def test_log_mel_shape_and_lengths():
    mel = LogMel(META)
    wavs = np.zeros((2, 16000), dtype=np.float32)
    feats = mel(wavs)

    assert feats.shape == (2, 64, (16000 - 320) // 160 + 1)
    assert feats.dtype == np.float32
    np.testing.assert_array_equal(mel.out_len(np.array([16000, 8000])), [99, 49])
    # Тишина — это пол log(1e-9), а не -inf: модель видела именно его.
    assert np.allclose(feats, np.log(1e-9))


def test_log_mel_matches_torchaudio():
    torch = pytest.importorskip("torch")
    torchaudio = pytest.importorskip("torchaudio")

    rng = np.random.default_rng(1)
    t = np.arange(32000) / 16000
    wav = (0.3 * np.sin(2 * np.pi * 440 * t) + 0.01 * rng.normal(size=t.shape)).astype(np.float32)
    reference = torchaudio.transforms.MelSpectrogram(
        sample_rate=16000, n_mels=64, win_length=320, hop_length=160, n_fft=320, center=False
    )(torch.from_numpy(wav)[None])
    reference = torch.log(reference.clamp(1e-9, 1e9)).numpy()

    mine = LogMel(META)(wav[None])
    assert np.abs(mine - reference).max() < 1e-3
    np.testing.assert_allclose(
        mel_filterbank(16000, 320, 64),
        torchaudio.functional.melscale_fbanks(161, 0.0, 8000.0, 64, 16000, None, "htk").numpy(),
        atol=1e-5,
    )


class _FakeEncoder:
    """Двойник сессии: 4 кадра признаков → 1 кадр энкодера, на кадре t
    «выстреливает» буква (t % 3) + 1, хвост за длиной — мусор."""

    def get_inputs(self):
        return [
            SimpleNamespace(name="features", type="tensor(float)"),
            SimpleNamespace(name="feature_lengths", type="tensor(int64)"),
        ]

    def run(self, _outputs, feeds):
        feats, lens = feeds["features"], feeds["feature_lengths"]
        self.seen = (feats.shape, lens.tolist())
        n_out = feats.shape[2] // 4
        enc_lens = lens // 4
        log_probs = np.full((feats.shape[0], n_out, BLANK + 1), -20.0, dtype=np.float32)
        for t in range(n_out):
            log_probs[:, t, (t % 3) + 1] = 0.0
        return [log_probs, enc_lens]


def test_decode_batch_pads_and_trims_by_encoder_length():
    session = _FakeEncoder()
    model = GigaAMOnnx(session, META)
    long_wav = np.zeros(16000, dtype=np.float32)
    short_wav = np.zeros(8000, dtype=np.float32)

    long_seg, short_seg = model.decode_batch([long_wav, short_wav])

    # Батч дополнен до самого длинного куска, длины признаков — свои.
    assert session.seen == ((2, 64, 99), [99, 49])
    assert long_seg.log_probs.shape == (24, BLANK + 1)
    assert short_seg.log_probs.shape == (12, BLANK + 1), "хвост за длиной отрезан"
    assert short_seg.text == "абвабвабвабв"
    # Секунд на кадр — по НАСТОЯЩЕЙ длине куска, как у gigaam.
    assert short_seg.frame_shift == pytest.approx(0.5 / 12)
    assert long_seg.frame_shift == pytest.approx(1.0 / 24)


def test_empty_batch():
    assert GigaAMOnnx(_FakeEncoder(), META).decode_batch([]) == []
