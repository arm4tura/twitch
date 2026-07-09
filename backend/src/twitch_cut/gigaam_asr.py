"""GigaAM v3 транскрипция — дефолтный ASR-движок проекта.

ЗАЧЕМ отдельно от transcription.py (WhisperX)
--------------------------------------------
GigaAM (Salute) на русском ловит мат точнее и даёт более узкие пословные
тайминги, чем faster-whisper (WhisperX растягивает mute-слово на 1–5 c —
см. сравнительный пробник examples/compare_asr.py). Плюс он не тянет за собой
хрупкий CTranslate2/cuDNN-стек, который на Windows постоянно падает на
`cudnn_ops_infer64_8.dll`. Поэтому это движок по умолчанию.

Модель по умолчанию — `v3_ctc`: CTC-декодер (не e2e) даёт «сырой» фонетический
текст без нормализации-цензуры, что важно для детекта мата. `transcribe_longform`
сам режет длинное аудио по VAD и возвращает сегменты с АБСОЛЮТНЫМИ пословными
таймингами (start/end уже сдвинуты на начало сегмента) — ровно та форма, что
нужна detect_profanity / build_decisions.

Выход этой функции идентичен по форме transcribe_audio (WhisperX):
    {"segments": [{"id", "start", "end", "words": [{"word","start","end","score"}]}]}
чтобы downstream (iter_words, detect_profanity, decisions) работал без изменений.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from .cache import file_fingerprint, read_json, stable_hash, write_json
from .config import MODELS_DIR

logger = logging.getLogger(__name__)

# Кэш загруженной модели в пределах процесса: веса GigaAM ~1–2 GB, грузить их
# на каждый прогон дорого. Ключ — (model_name, device).
_MODEL_CACHE: dict[tuple[str, str], Any] = {}


def _normalize_gigaam_model(model_name: str) -> str:
    """Приводим короткие/пользовательские имена к каноничным для gigaam.load_model.

    UI и дефолты оперируют 'v3_ctc'. Разрешаем также короткое 'ctc'/'rnnt'
    (gigaam сам достроит до v3_*), и на всякий случай прогоняем 'large-v3'
    (дефолт WhisperX-поля) в 'v3_ctc', чтобы случайно прокинутое значение
    из расширённого режима не роняло прогон.
    """
    name = (model_name or "").strip()
    if not name or name in {"large-v3", "gigaam", "default"}:
        return "v3_ctc"
    return name


def _patch_gigaam_vad_segmentation() -> None:
    """Чиним загрузку VAD-модели pyannote внутри gigaam.transcribe_longform.

    gigaam.vad_utils.resolve_local_segmentation_path возвращает директорию
    снапшота HF (`.../snapshots/<sha>/`), и эта директория уходит в
    pyannote `Model.from_pretrained`. Но тот принимает только ФАЙЛ-чекпоинт:
    директорию он не распознаёт (`os.path.isfile` == False) и трактует как
    HF repo_id → `validate_repo_id` падает с HFValidationError на Windows-пути
    (двоеточие, бэкслеши, длина > 96).

    Оборачиваем резолвер так, чтобы он возвращал путь к самому файлу весов
    (`pytorch_model.bin` / `model.safetensors`) внутри снапшота. Идемпотентно.
    """
    try:
        from gigaam import vad_utils
    except Exception:  # noqa: BLE001 — gigaam может не иметь vad_utils в старых сборках
        return

    if getattr(vad_utils, "_twitchcut_seg_patched", False):
        return

    _orig_resolve = vad_utils.resolve_local_segmentation_path

    def _resolve_to_file(model_id: str) -> str:
        resolved = Path(_orig_resolve(model_id=model_id))
        if resolved.is_dir():
            for name in ("pytorch_model.bin", "model.safetensors", "model.bin"):
                candidate = resolved / name
                if candidate.exists():
                    logger.info("VAD segmentation checkpoint: %s", candidate)
                    return str(candidate)
        return str(resolved)

    vad_utils.resolve_local_segmentation_path = _resolve_to_file
    vad_utils._twitchcut_seg_patched = True


def _load_gigaam_model(model_name: str, device: str) -> Any:
    cache_key = (model_name, device)
    cached = _MODEL_CACHE.get(cache_key)
    if cached is not None:
        return cached

    import gigaam  # локальный импорт: тяжёлый, нужен только при реальном прогоне

    # torch 2.6+: gigaam грузит checkpoint через torch.load; наши патчи ставят
    # weights_only=False и регистрируют safe globals, иначе unpickle падает.
    from .transcription import apply_whisperx_patches

    apply_whisperx_patches()
    # Чиним pyannote-VAD путь (директория снапшота → файл весов), иначе
    # transcribe_longform падает на HFValidationError.
    _patch_gigaam_vad_segmentation()

    # Держим веса GigaAM в общем models/ рядом с HF-кэшем, а не в ~/.cache/gigaam,
    # чтобы пользователь видел все веса в одном месте (и prefetch клал туда же).
    download_root = str((MODELS_DIR / "gigaam").resolve())
    (MODELS_DIR / "gigaam").mkdir(parents=True, exist_ok=True)

    logger.info("Загружаю GigaAM model=%s device=%s (download_root=%s)", model_name, device, download_root)
    try:
        model = gigaam.load_model(model_name, device=device, download_root=download_root)
    except TypeError:
        # Очень старые сборки gigaam без download_root/device kwargs.
        model = gigaam.load_model(model_name)

    _MODEL_CACHE[cache_key] = model
    return model


def _to_transcript(longform_result: Any) -> dict[str, Any]:
    """LongformTranscriptionResult -> dict формата WhisperX (segments/words)."""
    segments: list[dict[str, Any]] = []
    for seg_index, seg in enumerate(getattr(longform_result, "segments", []) or []):
        words: list[dict[str, Any]] = []
        for w in getattr(seg, "words", None) or []:
            words.append(
                {
                    "word": w.text,
                    "start": float(w.start),
                    "end": float(w.end),
                    # У GigaAM нет confidence-скора на слово — оставляем None,
                    # downstream это переносит (score опционален).
                    "score": None,
                }
            )
        # Если у сегмента нет пословных таймингов (word_timestamps выключен или
        # пустой сегмент) — всё равно кладём сам сегмент с его границами.
        segments.append(
            {
                "id": f"seg_{seg_index:06d}",
                "start": float(getattr(seg, "start", 0.0) or 0.0),
                "end": float(getattr(seg, "end", 0.0) or 0.0),
                "text": getattr(seg, "text", "") or "",
                "words": words,
            }
        )
    return {"segments": segments}


def transcribe_with_gigaam(
    audio_path: Path,
    workdir: Path,
    model_name: str = "v3_ctc",
    device: str = "cuda",
    force: bool = False,
) -> tuple[dict[str, Any], str, Path]:
    """Транскрипция через GigaAM v3 с пословными таймингами.

    Сигнатура и возвращаемый контракт совпадают с transcription.transcribe_audio:
    возвращает (transcript_dict, cache_key, cache_path). Кэш инвалидируется по
    отпечатку аудио + модели + устройства.

    device: 'cuda' | 'cpu'. compute_type WhisperX-а тут не нужен — GigaAM сам
    выбирает fp16 на GPU / fp32 на CPU.
    """
    model_name = _normalize_gigaam_model(model_name)

    key = stable_hash(
        {
            "stage": "gigaam_transcription",
            "audio": file_fingerprint(audio_path),
            "model": model_name,
            "device": device,
            "word_timestamps": True,
        }
    )
    cache_path = workdir / "cache" / f"transcript_{key}.json"

    if cache_path.exists() and not force:
        logger.info("GigaAM-транскрипт уже есть в cache: %s", cache_path)
        return read_json(cache_path), key, cache_path

    try:
        import gigaam  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "GigaAM не установлен. Установите git-версию (с word-timestamps):\n"
            "  backend\\.venv\\Scripts\\pip install --no-deps "
            "git+https://github.com/salute-developers/GigaAM.git\n"
            "или переключите движок на WhisperX (--transcriber whisperx)."
        ) from exc

    model = _load_gigaam_model(model_name, device)

    logger.info("Запускаю GigaAM transcribe_longform (word_timestamps=True): %s", audio_path.name)
    try:
        result = model.transcribe_longform(str(audio_path), word_timestamps=True)
    except TypeError as exc:
        # PyPI-релиз 0.1.0 без word-timestamps: transcribe(wav) -> str.
        raise RuntimeError(
            "Установленная версия GigaAM не поддерживает пословные тайминги "
            "(нужна git-версия). Переставьте: pip install --no-deps "
            "git+https://github.com/salute-developers/GigaAM.git"
        ) from exc

    transcript = _to_transcript(result)

    n_segments = len(transcript["segments"])
    n_words = sum(len(s["words"]) for s in transcript["segments"])
    logger.info("GigaAM: %d сегментов, %d слов с таймингами", n_segments, n_words)

    write_json(cache_path, transcript)
    return transcript, key, cache_path
