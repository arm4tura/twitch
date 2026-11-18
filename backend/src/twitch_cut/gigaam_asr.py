"""GigaAM v3 транскрипция — дефолтный ASR-движок проекта.

ЗАЧЕМ отдельно от transcription.py (WhisperX)
--------------------------------------------
GigaAM (Salute) на русском ловит мат точнее и даёт более узкие пословные
тайминги, чем faster-whisper (WhisperX растягивает mute-слово на 1–5 c —
см. сравнительный пробник examples/compare_asr.py). Поэтому это движок по
умолчанию.

КАК ЗАПУСКАЕТСЯ. Без torch: модель и VAD экспортированы в ONNX
(`tools/export_onnx.py`) и идут через onnxruntime (`onnx_models.py`).
Раньше здесь жила пачка monkey-patch'ей поверх пакета gigaam (блэнк на MPS,
путь к VAD-весам, ffmpeg в PATH, перехват `_decode` ради таймингов и
прогресса). Теперь весь long-form путь написан явно и по шагам:

  1. аудио → float32 16 кГц (`ffmpeg_tools.load_audio_16k`);
  2. VAD pyannote segmentation-3.0 → участки речи → куски 15–22 с
     (`vad.py`, повтор `gigaam.vad_utils.segment_audio_file`);
  3. куски батчами по 16 → лог-мел → энкодер+CTC → greedy-декод
     (`gigaam_onnx.py`);
  4. границы слов — forced alignment по тем же log-probs (`alignment.py`),
     отсюда и `score` слова; не выровнялось — кадры эмиссии CTC.

Модель по умолчанию — `v3_ctc`: CTC-декодер (не e2e) даёт «сырой» фонетический
текст без нормализации-цензуры, что важно для детекта мата.

Выход идентичен по форме transcribe_audio (WhisperX):
    {"segments": [{"id", "start", "end", "words": [{"word","start","end","score"}]}]}
чтобы downstream (iter_words, detect_profanity, decisions) работал без изменений.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

from .alignment import align_words
from .cache import file_fingerprint, read_json, stable_hash, write_json
from .posteriors import posteriors_path, to_float16, write_posteriors

logger = logging.getLogger(__name__)

# Колбэк дробного прогресса транскрибации: (fraction 0..1, message).
# Дробь считается по АУДИО-СЕКУНДАМ обработанных VAD-сегментов, а не по их
# количеству: сегменты неравной длины, и счёт «сегмент из N» даёт рваную
# оценку на записи, где в начале много коротких реплик.
TranscribeProgressFn = Callable[[float, str], None]

ASR_BATCH_SIZE = 16
"""Кусков за один прогон энкодера — как `fr_batch_size` у gigaam."""

SUPPORTED_MODELS = ("v3_ctc",)
"""Что есть в ONNX. RNN-T/e2e не экспортированы: декодер у них — цикл по
кадрам с состоянием LSTM, а детекту мата нужен именно сырой CTC-текст."""

# Кэш загруженных сессий в пределах процесса: открыть 400 МБ модели —
# секунды, делать это на каждый прогон незачем. Ключ — (model_name, device).
_MODEL_CACHE: dict[tuple[str, str], "_Engine"] = {}


def _normalize_gigaam_model(model_name: str) -> str:
    """Приводим короткие/пользовательские имена к каноничным.

    UI и дефолты оперируют 'v3_ctc'. Разрешаем также короткое 'ctc', и на
    всякий случай прогоняем 'large-v3' (дефолт WhisperX-поля) в 'v3_ctc',
    чтобы случайно прокинутое значение из расширенного режима не роняло прогон.
    """
    name = (model_name or "").strip()
    if not name or name in {"large-v3", "gigaam", "default", "ctc"}:
        return "v3_ctc"
    return name


@dataclass
class _Engine:
    """Открытые ONNX-сессии: VAD + GigaAM."""

    vad: Any
    asr: Any

    @property
    def accelerator(self) -> str:
        """Провайдер, который считает GigaAM сейчас (после сбоя ускорителя — CPU)."""
        return self.asr.session.get_providers()[0].replace("ExecutionProvider", "")


def _load_engine(model_name: str, device: str) -> _Engine:
    if model_name not in SUPPORTED_MODELS:
        raise RuntimeError(
            f"Модель GigaAM '{model_name}' не поддерживается: доступна только "
            f"{', '.join(SUPPORTED_MODELS)}."
        )
    cache_key = (model_name, device)
    cached = _MODEL_CACHE.get(cache_key)
    if cached is not None:
        return cached

    from .gigaam_onnx import GigaAMOnnx
    from .onnx_models import GIGAAM_MODEL, VAD_MODEL, create_session, ensure_model, read_meta
    from .vad import SpeechDetector

    vad_path = ensure_model(VAD_MODEL)
    asr_path = ensure_model(GIGAAM_MODEL)
    logger.info("Загружаю GigaAM %s (onnxruntime, device=%s)", model_name, device)
    # VAD гоняет КАЖДУЮ секунду аудио десять раз (окно 10 с, шаг 1 с) — на
    # трёхчасовом стриме это минуты CPU, поэтому на GPU он едет вместе с ASR.
    # Решение у VAD жёсткое (argmax по классам), поэтому погрешность GPU его
    # почти не двигает.
    vad_session = create_session(vad_path, device)
    asr_session = create_session(asr_path, device)
    engine = _Engine(
        vad=SpeechDetector(vad_session, read_meta(vad_session)),
        asr=GigaAMOnnx(asr_session, read_meta(asr_session)),
    )
    _MODEL_CACHE[cache_key] = engine
    return engine


class _SpeechProgress:
    """Дробный прогресс распознавания по аудио-секундам речи.

    ЗАЧЕМ. Распознавание — это 90% времени обработки, и без тиков UI
    показывает «12%» весь час: ETA считать не из чего (нужны минимум две
    точки), а пользователь не понимает, живой процесс или подвис.

    `found` зовётся после VAD (знаем ОБЩИЙ объём работы в секундах речи),
    `advance` — после каждого батча. Дробь = обработанные секунды / общие:
    сегменты сильно разной длины, счёт по штукам врёт на записях, где в
    начале много коротких реплик. Без речи не тикаем вовсе — делить не на что.
    """

    def __init__(
        self,
        on_progress: Optional[TranscribeProgressFn],
        on_status: Optional[Callable[[str], None]] = None,
    ):
        self._emit = on_progress
        self._status = on_status
        self._durations: list[float] = []
        self._total = 0.0
        self._done_idx = 0
        self._done_sec = 0.0
        self._last_emit = 0.0

    def searching(self, done: int, total: int) -> None:
        """Тик во время VAD — в `on_status`, а не в дробь: на длинном стриме
        поиск речи идёт минутами, и видно должно быть, что процесс жив, но
        ETA распознавания по нему считать нельзя (UI ставит якорь скорости на
        первом тике дроби — см. TRANSCRIBE_ENTER_PCT в pipeline.py)."""
        if self._status is None or not total:
            return
        now = time.monotonic()
        if done < total and now - self._last_emit < 0.5:
            return
        self._last_emit = now
        self._status(f"поиск речи: {done * 100 // total}%")

    def found(self, boundaries: list[tuple[float, float]]) -> None:
        self._durations = [max(0.0, float(end) - float(start)) for start, end in boundaries]
        self._total = sum(self._durations)
        if self._emit is not None and self._total > 0:
            self._emit(0.0, f"речь найдена: {len(self._durations)} фрагментов")

    def advance(self, batch_len: int) -> None:
        if self._emit is None or not self._total:
            return
        idx = self._done_idx
        self._done_sec += sum(self._durations[idx : idx + batch_len])
        self._done_idx = idx + batch_len
        fraction = min(1.0, self._done_sec / self._total)
        # Троттлим до ~2 событий в секунду: батч короткой речи может проходить
        # за миллисекунды, а каждый emit — это WS-пуш всем подписчикам.
        now = time.monotonic()
        if fraction >= 1.0 or now - self._last_emit >= 0.5:
            self._last_emit = now
            done_min = self._done_sec / 60.0
            total_min = self._total / 60.0
            self._emit(fraction, f"распознано {done_min:.1f} / {total_min:.1f} мин речи")


def _write_posteriors_for(
    path: Path,
    chunks: list[tuple[Any, float]],
    transcript: dict[str, Any],
    vocab: list[str],
    blank_id: int,
) -> None:
    """Сложить собранные посториоры в `.npz` рядом с транскриптом.

    Начала сегментов берём из ГОТОВОГО транскрипта, порядок сегментов у
    сборщика и у транскрипта один. Сегменты, для которых log-probs получить
    не удалось, выпадают — по ним просто не будет спотинга.
    """
    if not chunks:
        logger.warning("посториоры: сборщик пуст — спотинга по этому прогону не будет")
        return
    segments = transcript.get("segments") or []
    if len(chunks) != len(segments):
        logger.warning(
            "посториоры: %d сегментов у декода против %d в транскрипте — не сохраняю",
            len(chunks),
            len(segments),
        )
        return
    pairs = [
        (chunk, shift, float(seg.get("start") or 0.0))
        for (chunk, shift), seg in zip(chunks, segments)
        if chunk is not None
    ]
    if not pairs:
        return
    written = write_posteriors(
        path,
        [(chunk, shift) for chunk, shift, _ in pairs],
        starts=[start for _, _, start in pairs],
        vocab=vocab,
        blank_id=blank_id,
    )
    if written is not None:
        size_mb = written.stat().st_size / (1024 * 1024)
        logger.info(
            "Посториоры сохранены: %d сегментов, %.1f МБ (%s)",
            len(pairs),
            size_mb,
            written.name,
        )


def _emission_words(
    asr: Any, token_ids: list[int], token_frames: list[int], frame_shift: float
) -> list[tuple[str, float, float]]:
    """Слова по кадрам эмиссии — запасной путь, если выравнивание не сошлось
    (`gigaam.timestamps_utils.frames_to_words`)."""
    words: list[tuple[str, float, float]] = []
    chars: list[str] = []
    frames: list[int] = []

    def commit() -> None:
        text = "".join(chars).strip()
        if text:
            words.append((text, frames[0] * frame_shift, (frames[-1] + 1) * frame_shift))
        chars.clear()
        frames.clear()

    for token_id, frame in zip(token_ids, token_frames):
        char = asr.id_to_str(token_id)
        if char.startswith("▁"):
            commit()
            char = char[1:]
        elif char == " ":
            commit()
            continue
        chars.append(char)
        frames.append(frame)
    commit()
    return words


def _segment_entry(
    index: int, decoded: Any, asr: Any, seg_start: float, seg_end: float
) -> dict[str, Any]:
    """Один сегмент транскрипта: слова с АБСОЛЮТНЫМИ таймингами и скором."""
    aligned = None
    if decoded.log_probs.shape[0] > 0:
        aligned = align_words(
            decoded.log_probs,
            decoded.token_ids,
            blank_id=asr.blank_id,
            to_char=asr.id_to_str,
            frame_shift=decoded.frame_shift,
        )
    if aligned is not None:
        # Скор — средняя вероятность букв слова на пути forced alignment.
        raw = [(w.text, w.start, w.end, w.score) for w in aligned]
    else:
        # Не выровнялось — тайминги по кадрам эмиссии, скора нет (downstream
        # это переносит: неверный confidence хуже отсутствующего).
        raw = [
            (text, start, end, None)
            for text, start, end in _emission_words(
                asr, decoded.token_ids, decoded.token_frames, decoded.frame_shift
            )
        ]
    return {
        "id": f"seg_{index:06d}",
        "start": float(seg_start),
        "end": float(seg_end),
        "text": decoded.text,
        "words": [
            {
                "word": text,
                "start": float(round(start + seg_start, 3)),
                "end": float(round(end + seg_start, 3)),
                "score": score,
            }
            for text, start, end, score in raw
        ],
    }


def transcribe_with_gigaam(
    audio_path: Path,
    workdir: Path,
    model_name: str = "v3_ctc",
    device: str = "cuda",
    force: bool = False,
    on_progress: Optional[TranscribeProgressFn] = None,
    keep_posteriors: bool = False,
    on_status: Optional[Callable[[str], None]] = None,
) -> tuple[dict[str, Any], str, Path]:
    """Транскрипция через GigaAM v3 с пословными таймингами.

    Сигнатура и возвращаемый контракт совпадают с transcription.transcribe_audio:
    возвращает (transcript_dict, cache_key, cache_path). Кэш инвалидируется по
    отпечатку аудио + модели + устройства.

    device: 'cuda' | 'mps' | 'cpu' — см. `onnx_models` (GPU через CUDA/DirectML;
    'mps' на Apple — CPU). Не поднялся или упал ускоритель — считаем на CPU.

    on_progress: необязательный колбэк (fraction 0..1, message) — тикает по мере
    прохода VAD-сегментов, чтобы UI мог показать ETA самой длинной стадии.

    keep_posteriors: сохранить log-probs рядом с транскриптом (`posteriors_<key>.npz`,
    см. `posteriors.py`) — их читает поиск мата по посториорам. Если файла нет, а
    он нужен, транскрипт из кэша НЕ берём: посториоры второй раз без прогона
    модели не получить.

    on_status: необязательный колбэк строки состояния ДО распознавания (поиск
    речи) — без дроби, чтобы не попасть в замер скорости для ETA.
    """
    model_name = _normalize_gigaam_model(model_name)

    key = stable_hash(
        {
            "stage": "gigaam_transcription",
            "audio": file_fingerprint(audio_path),
            "model": model_name,
            "device": device,
            "word_timestamps": True,
            # Версия способа получить границы слова: forced alignment по
            # log-probs вместо кадров эмиссии CTC (alignment.py). Меняется —
            # старые транскрипты в кэше должны стать невалидными, иначе прогон
            # тихо останется на прежних таймингах. Переезд torch → onnxruntime
            # ключ НЕ меняет: граф, признаки, VAD и выравнивание те же, и
            # пересчитывать уже распознанные стримы незачем.
            "timings": "forced_align_v1",
        }
    )
    cache_path = workdir / "cache" / f"transcript_{key}.json"
    npz_path = posteriors_path(workdir, key)

    if cache_path.exists() and not force and (npz_path.exists() or not keep_posteriors):
        logger.info("GigaAM-транскрипт уже есть в cache: %s", cache_path)
        return read_json(cache_path), key, cache_path

    from .ffmpeg_tools import load_audio_16k
    from .vad import segment_audio

    engine = _load_engine(model_name, device)
    asr = engine.asr
    progress = _SpeechProgress(on_progress, on_status)

    logger.info("GigaAM (%s): %s", engine.accelerator, audio_path.name)
    audio = load_audio_16k(audio_path, asr.sample_rate)
    regions = engine.vad.speech_regions(audio, on_chunk=progress.searching)
    pieces, boundaries = segment_audio(audio, regions, asr.sample_rate)
    del audio
    progress.found(boundaries)

    segments: list[dict[str, Any]] = []
    # Посториоры нужны только когда их просят сохранить (иначе это десятки
    # мегабайт в памяти впустую). float16 — сразу на выходе декода.
    chunks: list[tuple[Any, float]] = []
    for batch_start in range(0, len(pieces), ASR_BATCH_SIZE):
        batch = pieces[batch_start : batch_start + ASR_BATCH_SIZE]
        for offset, decoded in enumerate(asr.decode_batch(batch)):
            index = batch_start + offset
            seg_start, seg_end = boundaries[index]
            segments.append(_segment_entry(index, decoded, asr, seg_start, seg_end))
            if keep_posteriors:
                chunk = to_float16(decoded.log_probs) if decoded.log_probs.shape[0] else None
                chunks.append((chunk, decoded.frame_shift))
        progress.advance(len(batch))

    transcript = {"segments": segments}
    if keep_posteriors:
        _write_posteriors_for(npz_path, chunks, transcript, asr.vocab, asr.blank_id)

    n_words = sum(len(s["words"]) for s in segments)
    n_scored = sum(1 for s in segments for w in s["words"] if w.get("score") is not None)
    logger.info(
        "GigaAM: %d сегментов, %d слов с таймингами (forced alignment: %d)",
        len(segments),
        n_words,
        n_scored,
    )

    write_json(cache_path, transcript)
    return transcript, key, cache_path
