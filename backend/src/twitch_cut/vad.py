"""VAD без torch: pyannote segmentation-3.0 в onnxruntime + нарезка под ASR.

ЗАЧЕМ. GigaAM режет длинное аудио на куски по речи (`gigaam.vad_utils`):
pyannote VoiceActivityDetection находит речь, потом соседние участки
склеиваются в куски 15–22 с (жёсткий потолок 30 с). Сама модель pyannote
экспортирована в ONNX (`tools/export_onnx.py`), а всё вокруг неё —
скользящее окно, перевод powerset-классов в «речь/не речь», склейка окон
весами Хэмминга и гистерезис — повторено здесь на numpy, операция в
операцию как в pyannote.audio 3.4 (`Inference.slide`, `Inference.aggregate`,
`Binarize`). Поэтому границы кусков те же, что давал torch-путь, а значит и
транскрипт тот же.

Параметры (окно 10 с, шаг 1 с, рецептивное поле кадра, таблица powerset)
лежат в метаданных самого .onnx — модуль их не хардкодит.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Optional

import numpy as np

logger = logging.getLogger(__name__)

VAD_BATCH_SIZE = 32
"""Окон за один прогон модели — как у pyannote Inference по умолчанию."""


class SpeechDetector:
    """pyannote VoiceActivityDetection(segmentation-3.0) поверх onnxruntime."""

    def __init__(self, session: Any, meta: dict[str, Any]):
        self.session = session
        self.sample_rate = int(meta["sample_rate"])
        self.window = int(meta["window_samples"])
        self.duration = float(meta["duration"])
        self.step = float(meta["step"])
        self.warm_up = tuple(float(x) for x in meta.get("warm_up", (0.0, 0.0)))
        self.frame_duration = float(meta["frame_duration"])
        self.frame_step = float(meta["frame_step"])
        self.onset = float(meta.get("onset", 0.5))
        self.offset = float(meta.get("offset", self.onset))
        mapping = np.asarray(meta["powerset_mapping"], dtype=np.float32)
        # Класс powerset «речь есть», если в его множестве хоть один спикер.
        # pyannote делает one_hot(argmax) @ mapping и берёт max по спикерам —
        # это ровно «argmax попал в непустое множество».
        self._speech_class = mapping.max(axis=1) > 0
        self._input = session.get_inputs()[0].name

    def speech_regions(
        self,
        audio: np.ndarray,
        on_chunk: Optional[Callable[[int, int], None]] = None,
    ) -> list[tuple[float, float]]:
        """Участки речи [(start, end)] в секундах, по возрастанию."""
        scores = self._slide(audio, on_chunk)
        return self._binarize(scores)

    # --- Inference.slide ----------------------------------------------------

    def _slide(
        self, audio: np.ndarray, on_chunk: Optional[Callable[[int, int], None]]
    ) -> np.ndarray:
        num_samples = int(audio.shape[0])
        window = self.window
        step_size = round(self.step * self.sample_rate)

        if num_samples >= window:
            num_chunks = (num_samples - window) // step_size + 1
        else:
            num_chunks = 0
        has_last_chunk = num_samples < window or (num_samples - window) % step_size > 0
        total = num_chunks + int(has_last_chunk)

        outputs: list[np.ndarray] = []
        for c in range(0, num_chunks, VAD_BATCH_SIZE):
            n = min(VAD_BATCH_SIZE, num_chunks - c)
            batch = np.stack(
                [audio[(c + k) * step_size : (c + k) * step_size + window] for k in range(n)]
            )
            outputs.append(self._infer(batch))
            if on_chunk is not None:
                on_chunk(c + n, total)
        if has_last_chunk:
            last = audio[num_chunks * step_size :]
            padded = np.zeros(window, dtype=np.float32)
            padded[: last.shape[0]] = last
            outputs.append(self._infer(padded[None]))
            if on_chunk is not None:
                on_chunk(total, total)

        if not outputs:
            return np.zeros((0, 1), dtype=np.float32)
        chunk_scores = np.vstack(outputs)
        aggregated = self._aggregate(chunk_scores)
        if has_last_chunk:
            aggregated = self._crop_loose(aggregated, num_samples / self.sample_rate)
        return aggregated

    def _infer(self, batch: np.ndarray) -> np.ndarray:
        """[B, samples] → [B, кадры, 1]: 1.0 там, где речь, иначе 0.0."""
        logits = self.session.run(None, {self._input: batch[:, None, :].astype(np.float32)})[0]
        speech = self._speech_class[np.argmax(logits, axis=-1)]
        return speech.astype(np.float32)[..., None]

    # --- Inference.aggregate (hamming=True, missing=0.0) -------------------

    def _aggregate(self, scores: np.ndarray) -> np.ndarray:
        num_chunks, num_frames_per_chunk, num_classes = scores.shape
        # Окна кадров отсчитываются от начала первого окна (chunks.start = 0),
        # а не от frame_start модели — так делает pyannote.
        frame_start = 0.0
        epsilon = 1e-12

        hamming_window = np.hamming(num_frames_per_chunk).reshape(-1, 1)
        warm_up_window = np.ones((num_frames_per_chunk, 1))
        warm_up_left = round(self.warm_up[0] / self.duration * num_frames_per_chunk)
        warm_up_window[:warm_up_left] = epsilon
        warm_up_right = round(self.warm_up[1] / self.duration * num_frames_per_chunk)
        warm_up_window[num_frames_per_chunk - warm_up_right :] = epsilon

        def closest(t: float) -> int:
            return int(
                np.rint((t - frame_start - 0.5 * self.frame_duration) / self.frame_step)
            )

        num_frames = (
            closest(0.0 + self.duration + (num_chunks - 1) * self.step + 0.5 * self.frame_duration)
            + 1
        )
        aggregated_output = np.zeros((num_frames, num_classes), dtype=np.float32)
        overlapping_chunk_count = np.zeros((num_frames, num_classes), dtype=np.float32)
        aggregated_mask = np.zeros((num_frames, num_classes), dtype=np.float32)

        for index in range(num_chunks):
            score = scores[index]
            chunk_start = 0.0 + index * self.step
            mask = 1 - np.isnan(score)
            np.nan_to_num(score, copy=False, nan=0.0)
            start_frame = closest(chunk_start + 0.5 * self.frame_duration)
            end_frame = start_frame + num_frames_per_chunk
            aggregated_output[start_frame:end_frame] += score * mask * hamming_window * warm_up_window
            overlapping_chunk_count[start_frame:end_frame] += mask * hamming_window * warm_up_window
            aggregated_mask[start_frame:end_frame] = np.maximum(
                aggregated_mask[start_frame:end_frame], mask
            )

        average = aggregated_output / np.maximum(overlapping_chunk_count, epsilon)
        average[aggregated_mask == 0.0] = 0.0
        return average

    def _crop_loose(self, scores: np.ndarray, end: float) -> np.ndarray:
        """SlidingWindowFeature.crop(Segment(0, end), mode='loose') — хвост,
        выпавший за конец аудио из-за дополненного нулями последнего окна."""
        j = int(np.floor((end - 0.0) / self.frame_step))
        return scores[: min(j + 1, scores.shape[0])]

    # --- Binarize(onset, offset), min_duration_on/off = 0 -------------------

    def _binarize(self, scores: np.ndarray) -> list[tuple[float, float]]:
        num_frames = scores.shape[0]
        if num_frames == 0:
            return []
        timestamps = [
            0.5 * ((i * self.frame_step) + (i * self.frame_step + self.frame_duration))
            for i in range(num_frames)
        ]
        k_scores = scores[:, 0]
        regions: list[tuple[float, float]] = []
        start = timestamps[0]
        is_active = k_scores[0] > self.onset
        t = start
        for t, y in zip(timestamps[1:], k_scores[1:]):
            if is_active:
                if y < self.offset:
                    regions.append((start, t))
                    start = t
                    is_active = False
            else:
                if y > self.onset:
                    start = t
                    is_active = True
        if is_active:
            regions.append((start, t))
        return [(a, b) for a, b in regions if b - a >= 1e-6]


def segment_audio(
    audio: np.ndarray,
    regions: list[tuple[float, float]],
    sample_rate: int,
    max_duration: float = 22.0,
    min_duration: float = 15.0,
    strict_limit_duration: float = 30.0,
    new_chunk_threshold: float = 0.2,
) -> tuple[list[np.ndarray], list[tuple[float, float]]]:
    """Склеить участки речи в куски под ASR — `gigaam.vad_utils.segment_audio_file`.

    Соседние участки копятся, пока кусок не перерастёт `min_duration` или
    очередной участок не вытолкнет его за `max_duration`; кусок длиннее
    `strict_limit_duration` режется поровну. Возвращает сами куски сэмплов и
    их границы (секунды от начала аудио).
    """
    sr = sample_rate
    segments: list[np.ndarray] = []
    boundaries: list[tuple[float, float]] = []
    audio_end = audio.shape[0] / sr

    def _update_segments(curr_start: float, curr_end: float, curr_duration: float) -> None:
        if curr_duration > strict_limit_duration:
            max_segments = int(curr_duration / strict_limit_duration) + 1
            segment_duration = curr_duration / max_segments
            curr_end = curr_start + segment_duration
            for _ in range(max_segments - 1):
                segments.append(audio[int(curr_start * sr) : int(curr_end * sr)])
                boundaries.append((curr_start, curr_end))
                curr_start = curr_end
                curr_end += segment_duration
        segments.append(audio[int(curr_start * sr) : int(curr_end * sr)])
        boundaries.append((curr_start, curr_end))

    curr_duration = 0.0
    curr_start = 0.0
    curr_end = 0.0
    for region_start, region_end in regions:
        start = max(0, region_start)
        end = min(audio_end, region_end)
        if curr_duration == 0.0:
            curr_start = start
        elif curr_duration > new_chunk_threshold and (
            curr_duration + (end - curr_end) > max_duration or curr_duration > min_duration
        ):
            _update_segments(curr_start, curr_end, curr_duration)
            curr_start = start
        curr_end = end
        curr_duration = curr_end - curr_start

    if curr_duration > new_chunk_threshold:
        _update_segments(curr_start, curr_end, curr_duration)

    return segments, boundaries
