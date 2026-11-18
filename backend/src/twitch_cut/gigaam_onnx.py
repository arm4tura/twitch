"""GigaAM v3_ctc в onnxruntime: мел-спектрограмма на numpy + greedy CTC.

ЗАЧЕМ. Это то, что раньше делал пакет gigaam на torch: признаки → энкодер →
CTC-голова → greedy-декод. Энкодер с головой экспортированы одним графом
(`tools/export_onnx.py`), всё до и после графа — здесь:

  - лог-мел как у torchaudio.MelSpectrogram (Hann periodic, power 2, мел-шкала
    HTK без нормировки, center=False) и `log(clamp(1e-9, 1e9))` — как
    `gigaam.preprocess.FeatureExtractor`;
  - батч дополняется нулями до самого длинного куска, длины признаков
    считаются так же, как в gigaam: маска внутри энкодера отрезает хвост, и
    результат по куску совпадает с прогоном без батча;
  - greedy-декод: argmax по кадру, blank выброшен, повторы склеены — кадры
    первых всплесков букв уходят дальше как кадры эмиссии.

Параметры признаков, словарь и blank лежат в метаданных .onnx.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np


@dataclass
class DecodedSegment:
    """Результат распознавания одного куска аудио."""

    text: str
    token_ids: list[int]
    token_frames: list[int]
    """Кадр энкодера, на котором выстрелил каждый токен (greedy)."""
    log_probs: np.ndarray
    """[кадры, классы] float32 — log-softmax CTC только по настоящим кадрам."""
    frame_shift: float
    """Секунд на кадр энкодера (у каждого куска своё — как у gigaam)."""


def mel_filterbank(sample_rate: int, n_fft: int, n_mels: int) -> np.ndarray:
    """Треугольные мел-фильтры [n_fft//2+1, n_mels] — torchaudio.functional.
    melscale_fbanks(f_min=0, f_max=sr/2, norm=None, mel_scale='htk'), float32."""
    n_freqs = n_fft // 2 + 1
    all_freqs = np.linspace(0, sample_rate // 2, n_freqs, dtype=np.float32)
    m_min = 2595.0 * math.log10(1.0 + 0.0 / 700.0)
    m_max = 2595.0 * math.log10(1.0 + (sample_rate / 2) / 700.0)
    m_pts = np.linspace(m_min, m_max, n_mels + 2, dtype=np.float32)
    f_pts = (700.0 * (10.0 ** (m_pts / np.float32(2595.0)) - 1.0)).astype(np.float32)
    f_diff = f_pts[1:] - f_pts[:-1]
    slopes = f_pts[None, :] - all_freqs[:, None]
    down_slopes = (-1.0 * slopes[:, :-2]) / f_diff[:-1]
    up_slopes = slopes[:, 2:] / f_diff[1:]
    return np.maximum(np.float32(0.0), np.minimum(down_slopes, up_slopes)).astype(np.float32)


class LogMel:
    """Лог-мел признаки GigaAM: [B, samples] → [B, features, кадры]."""

    def __init__(self, meta: dict[str, Any]):
        self.sample_rate = int(meta["sample_rate"])
        self.n_fft = int(meta["n_fft"])
        self.win_length = int(meta["win_length"])
        self.hop_length = int(meta["hop_length"])
        self.center = bool(meta["center"])
        if self.center or self.win_length != self.n_fft:
            # У v3 center=False и окно = n_fft; другой конфиг значит другую
            # модель, и молча считать признаки «примерно так же» нельзя.
            raise ValueError("Поддерживаются только признаки с center=False и win_length == n_fft")
        # Окно и фильтры — из модели (там они округлены до float16, и модель
        # обучена на признаках именно с ними, см. tools/export_onnx.py).
        # Посчитанные заново — только запасной путь для модели без них.
        if meta.get("window") is not None:
            self.window = np.asarray(meta["window"], dtype=np.float32)
        else:
            n = np.arange(self.win_length, dtype=np.float64)
            self.window = (0.5 - 0.5 * np.cos(2.0 * math.pi * n / self.win_length)).astype(
                np.float32
            )
        if meta.get("mel_fb") is not None:
            self.fb = np.asarray(meta["mel_fb"], dtype=np.float32)
        else:
            self.fb = mel_filterbank(self.sample_rate, self.n_fft, int(meta["features"]))

    def out_len(self, lengths: np.ndarray) -> np.ndarray:
        return (lengths - self.win_length) // self.hop_length + 1

    def __call__(self, wavs: np.ndarray) -> np.ndarray:
        frames = np.lib.stride_tricks.sliding_window_view(wavs, self.n_fft, axis=-1)
        frames = frames[:, :: self.hop_length] * self.window
        spectrum = np.fft.rfft(frames, n=self.n_fft, axis=-1)
        power = (spectrum.real**2 + spectrum.imag**2).astype(np.float32)
        # numpy с Accelerate (macOS) ставит ложные флаги FP-исключений на
        # matmul float32 — результат верный, а предупреждения засоряют лог.
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            mel = power @ self.fb
        return np.log(np.clip(mel, 1e-9, 1e9)).transpose(0, 2, 1).astype(np.float32)


class GigaAMOnnx:
    """Энкодер+CTC GigaAM одной ONNX-сессией."""

    def __init__(self, session: Any, meta: dict[str, Any]):
        self.session = session
        self.meta = meta
        self.sample_rate = int(meta["sample_rate"])
        self.vocab: list[str] = [str(ch) for ch in meta["vocab"]]
        self.blank_id = int(meta["blank_id"])
        self.features = LogMel(meta)
        inputs = session.get_inputs()
        self._feat_name = inputs[0].name
        self._len_name = inputs[1].name
        self._feat_dtype = np.float16 if "float16" in inputs[0].type else np.float32

    def id_to_str(self, token_id: int) -> str:
        if 0 <= token_id < len(self.vocab):
            return self.vocab[token_id]
        return ""

    def decode_batch(self, wavs: Sequence[np.ndarray]) -> list[DecodedSegment]:
        """Распознать куски аудио одним батчем (как DataLoader у gigaam)."""
        if not wavs:
            return []
        lengths = np.asarray([len(w) for w in wavs], dtype=np.int64)
        batch = np.zeros((len(wavs), int(lengths.max())), dtype=np.float32)
        for i, wav in enumerate(wavs):
            batch[i, : len(wav)] = wav
        features = self.features(batch)
        feat_lens = self.features.out_len(lengths).astype(np.int64)
        log_probs, enc_lens = self.session.run(
            None,
            {self._feat_name: features.astype(self._feat_dtype), self._len_name: feat_lens},
        )
        log_probs = np.asarray(log_probs, dtype=np.float32)
        enc_lens = np.asarray(enc_lens, dtype=np.int64).reshape(-1)

        out: list[DecodedSegment] = []
        for i in range(len(wavs)):
            n_frames = int(min(max(enc_lens[i], 0), log_probs.shape[1]))
            seg_probs = np.ascontiguousarray(log_probs[i, :n_frames])
            ids, frames = greedy_tokens(seg_probs, self.blank_id)
            out.append(
                DecodedSegment(
                    text="".join(self.id_to_str(t) for t in ids),
                    token_ids=ids,
                    token_frames=frames,
                    log_probs=seg_probs,
                    frame_shift=(int(lengths[i]) / self.sample_rate / n_frames) if n_frames else 0.0,
                )
            )
        return out


def greedy_tokens(log_probs: np.ndarray, blank_id: int) -> tuple[list[int], list[int]]:
    """Greedy CTC: (id токенов, кадры их первых всплесков) — как
    `gigaam.decoding.CTCGreedyDecoding.decode`."""
    if log_probs.shape[0] == 0:
        return [], []
    labels = log_probs.argmax(axis=-1)
    keep = labels != blank_id
    keep[1:] &= labels[1:] != labels[:-1]
    frames = np.nonzero(keep)[0]
    return labels[frames].astype(int).tolist(), frames.astype(int).tolist()
