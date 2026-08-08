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
import os
import shutil
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


# Перехощенные веса pyannote segmentation-3.0 (лицензия MIT). Лежат в
# GitHub-релизе проекта, чтобы не тянуть с HuggingFace: там segmentation-3.0
# gated и требует принять условия + auth-токен, что ломает офлайн-прогон у
# пользователей. GitHub отдаёт файл по прямой ссылке без токена.
VAD_SEGMENTATION_URL = (
    "https://github.com/arm4tura/twitch/releases/download/models-v1/pytorch_model.bin"
)
VAD_SEGMENTATION_PATH = (MODELS_DIR / "vad" / "segmentation-3.0.bin").resolve()


def _ensure_vad_segmentation() -> str | None:
    """Скачать веса pyannote segmentation-3.0 из GitHub-релиза (один раз, с кэшем).

    Возвращает путь к локальному .bin или None, если скачать не удалось — тогда
    вызывающий код падает обратно на оригинальный резолвер gigaam (он попробует
    HF). Файл — pytorch-lightning чекпоинт segmentation-3.0; pyannote
    Model.from_pretrained принимает его напрямую как ФАЙЛ, без config.yaml.
    """
    target = VAD_SEGMENTATION_PATH
    if target.exists() and target.stat().st_size > 0:
        return str(target)
    try:
        import urllib.request

        target.parent.mkdir(parents=True, exist_ok=True)
        # Качаем во временный .part и атомарно переименовываем — чтобы битый
        # (оборванный) файл не выглядел как валидный кэш при следующем запуске.
        tmp = target.with_name(target.name + ".part")
        logger.info("Скачиваю VAD segmentation-3.0: %s", VAD_SEGMENTATION_URL)
        urllib.request.urlretrieve(VAD_SEGMENTATION_URL, tmp)
        if tmp.stat().st_size == 0:
            tmp.unlink(missing_ok=True)
            raise OSError("скачанный файл пуст")
        tmp.replace(target)
        logger.info("VAD segmentation-3.0 готов: %s", target)
        return str(target)
    except Exception as exc:  # noqa: BLE001 — сеть/диск; fallback на HF-резолвер
        logger.warning(
            "Не удалось скачать VAD-веса из GitHub (%s) — fallback на HuggingFace", exc
        )
        return None


def _patch_gigaam_blank_token_crash() -> None:
    """Фиксим ДВЕ болячки CTC-декодера GigaAM на MPS (Apple Metal).

    (A) КРАШ: на device='mps' булева маска `labels[skip_mask]` в CTCGreedyDecoding
        (decoding.py) иногда пропускает blank_id=33, и он доходит до
        tokenizer.decode → vocab[33] → ConfigIndexError на 33-элементном vocab.

    (B) МУСОРНЫЙ ТЕКСТ (главное): даже когда краша нет, boolean-indexing в torch
        MPS (2.6, macOS) на реальных log_probs модели даёт НЕВЕРНУЮ выборку
        токенов — `labels[skip_mask]` возвращает не те элементы, что на CPU.
        Слова получаются перемешанной кашей («вооббеввююч»), а детектор мата
        лепит ложные заглушки там, где мата нет. На CPU та же маска корректна
        (проверено: транскрипт читаемый).

    Решение:
      1. Оборачиваем `CTCGreedyDecoding.decode` так, чтобы весь ПОСТ-энкодерный
         счёт (argmax → skip_mask → выборка токенов) шёл на CPU. Тяжёлый энкодер
         остаётся на MPS; сюда приходит уже маленький `encoded` ([B, T, D]),
         перенос на CPU дешёвый, а результат совпадает с чистым CPU-прогоном.
      2. Страховка от краша (A): фильтруем токены >= len(vocab) и коэрсим vocab
         ListConfig→list в Tokenizer.decode/id_to_str.

    Патч живёт в проектном слое (устойчив к pip reinstall), идемпотентен.
    """
    try:
        from gigaam.decoding import CTCGreedyDecoding, Tokenizer
    except Exception:  # noqa: BLE001
        return

    if getattr(Tokenizer, "_twitchcut_blank_patched", False):
        return

    # --- (1) CTC greedy decode на CPU: чинит мусорный текст + краш разом -------
    _orig_ctc_decode = CTCGreedyDecoding.decode

    def cpu_ctc_decode(self, head, encoded, lengths):
        """Считаем декод на CPU: MPS boolean-indexing выбирает не те токены.

        Переносим сам модуль head и входы на CPU на время декода, потом
        возвращаем head обратно на исходное устройство (модель может
        переиспользоваться между прогонами). encoded здесь уже маленький.
        """
        try:
            import torch  # noqa: F401
        except Exception:  # noqa: BLE001
            return _orig_ctc_decode(self, head, encoded, lengths)

        enc_dev = getattr(encoded, "device", None)
        if enc_dev is None or enc_dev.type != "mps":
            # Не MPS — оригинальный путь без накладных расходов.
            return _orig_ctc_decode(self, head, encoded, lengths)

        head_was = next(head.parameters()).device if any(True for _ in head.parameters()) else None
        try:
            head_cpu = head.to("cpu")
            enc_cpu = encoded.detach().to("cpu")
            len_cpu = lengths.detach().to("cpu")
            return _orig_ctc_decode(self, head_cpu, enc_cpu, len_cpu)
        finally:
            if head_was is not None:
                head.to(head_was)

    CTCGreedyDecoding.decode = cpu_ctc_decode

    # --- (2) Страховка Tokenizer: отсекаем out-of-vocab токены (краш A) --------
    _orig_decode = Tokenizer.decode
    _orig_id_to_str = Tokenizer.id_to_str

    def safe_decode(self, tokens: list[int]) -> str:
        """Отсекаем out-of-vocab токены (включая blank) перед декодом."""
        if self.charwise:
            # Конвертируем OmegaConf ListConfig в обычный list — защита от
            # ConfigIndexError при любой ошибке индексации.
            if not isinstance(self.vocab, list):
                self.vocab = list(self.vocab)
            # Фильтруем токены >= len(vocab): blank_id всегда == len(vocab),
            # любые id выше — тоже битые (не должны доходить сюда по идее).
            clean = [t for t in tokens if 0 <= t < len(self.vocab)]
            return _orig_decode(self, clean)
        return _orig_decode(self, tokens)

    def safe_id_to_str(self, token_id: int) -> str:
        """Заглушка на случай, если frames_to_words пытается конвертить blank."""
        if self.charwise:
            if not isinstance(self.vocab, list):
                self.vocab = list(self.vocab)
            if token_id < 0 or token_id >= len(self.vocab):
                return ""  # blank или out-of-vocab → пустая строка
            return self.vocab[token_id]
        return _orig_id_to_str(self, token_id)

    Tokenizer.decode = safe_decode
    Tokenizer.id_to_str = safe_id_to_str
    Tokenizer._twitchcut_blank_patched = True


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
        # 1) Наши перехощенные веса из GitHub-релиза (MIT, без HF-токена).
        #    Если скачались/уже в кэше — отдаём их и HF не трогаем вовсе.
        hosted = _ensure_vad_segmentation()
        if hosted:
            logger.info("VAD segmentation checkpoint (GitHub): %s", hosted)
            return hosted
        # 2) Fallback: оригинальный резолвер gigaam (снапшот HF). Чиним
        #    «папка→файл», как раньше.
        resolved = Path(_orig_resolve(model_id=model_id))
        if resolved.is_dir():
            for name in ("pytorch_model.bin", "model.safetensors", "model.bin"):
                candidate = resolved / name
                if candidate.exists():
                    logger.info("VAD segmentation checkpoint (HF): %s", candidate)
                    return str(candidate)
        return str(resolved)

    vad_utils.resolve_local_segmentation_path = _resolve_to_file
    vad_utils._twitchcut_seg_patched = True


def _ensure_ffmpeg_on_path() -> None:
    """Даём gigaam найти ffmpeg по голой команде `"ffmpeg"`.

    ЗАЧЕМ. gigaam.preprocess.load_audio (её зовёт vad_utils.segment_audio_file
    внутри transcribe_longform) вызывает ffmpeg как `subprocess.run(["ffmpeg", ...])`,
    т.е. резолвит бинарь через PATH. На Windows ffmpeg обычно в PATH — работает.
    На macOS Electron-приложение стартует с урезанным PATH (без Homebrew /opt/homebrew/bin),
    поэтому голый вызов падает с FileNotFoundError: 'ffmpeg'.

    Наш собственный код (ffmpeg_tools.py) от этого не страдает — он берёт бандл
    через imageio_ffmpeg.get_ffmpeg_exe(). Тот же бандл-бинарь подкладываем и для
    gigaam: делаем симлинк с именем `ffmpeg` на versioned-бинарь
    (ffmpeg-macos-aarch64-v7.1) и добавляем его каталог в начало PATH. Идемпотентно.
    """
    # Если ffmpeg уже резолвится (Windows, или пользователь поставил его в PATH) —
    # ничего не трогаем.
    if shutil.which("ffmpeg"):
        return

    try:
        import imageio_ffmpeg

        exe = Path(imageio_ffmpeg.get_ffmpeg_exe())
    except Exception as exc:  # noqa: BLE001 — imageio_ffmpeg может отсутствовать
        logger.warning("Не удалось получить бандл ffmpeg из imageio_ffmpeg: %s", exc)
        return

    if not exe.exists():
        logger.warning("Бандл ffmpeg не найден по пути: %s", exe)
        return

    # Каталог для симлинка держим рядом с весами, чтобы не мусорить в temp и
    # переиспользовать между прогонами.
    bin_dir = (MODELS_DIR / "ffmpeg_bin").resolve()
    bin_dir.mkdir(parents=True, exist_ok=True)
    link = bin_dir / ("ffmpeg.exe" if os.name == "nt" else "ffmpeg")

    try:
        # Пересоздаём симлинк, если он битый или указывает не туда.
        if link.is_symlink() or link.exists():
            if link.is_symlink() and Path(os.readlink(link)) == exe and link.exists():
                pass  # уже правильный — оставляем
            else:
                link.unlink()
                link.symlink_to(exe)
        else:
            link.symlink_to(exe)
    except OSError as exc:
        # На некоторых системах симлинки запрещены (напр. Windows без прав).
        # Фолбэк: просто добавим каталог самого бандла в PATH — но там имя
        # versioned, поэтому голый `ffmpeg` так не найдётся. Пробуем копию.
        logger.warning("Симлинк ffmpeg не создан (%s), пробую копию", exc)
        try:
            shutil.copy2(exe, link)
        except OSError as copy_exc:  # noqa: BLE001
            logger.warning("Не удалось подготовить ffmpeg для gigaam: %s", copy_exc)
            return

    os.environ["PATH"] = str(bin_dir) + os.pathsep + os.environ.get("PATH", "")
    logger.info("ffmpeg для gigaam подготовлен: %s -> %s", link, exe)


def _resolve_device(requested: str) -> str:
    """Пробуем MPS; если недоступен или падает — фолбэк на CPU с варнингом."""
    if requested != "mps":
        return requested
    try:
        import torch
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS недоступен на этом устройстве")
        # Быстрый smoke-test: если ядра не компилируются — узнаем сразу
        torch.zeros(1, device="mps")
        return "mps"
    except Exception as exc:  # noqa: BLE001
        logger.warning("MPS недоступен, переключаюсь на CPU: %s", exc)
        return "cpu"


def _load_gigaam_model(model_name: str, device: str) -> Any:
    device = _resolve_device(device)
    cache_key = (model_name, device)
    cached = _MODEL_CACHE.get(cache_key)
    if cached is not None:
        return cached

    import gigaam  # локальный импорт: тяжёлый, нужен только при реальном прогоне

    # torch 2.6+: gigaam грузит checkpoint через torch.load; наши патчи ставят
    # weights_only=False и регистрируют safe globals, иначе unpickle падает.
    from .transcription import apply_whisperx_patches

    apply_whisperx_patches()
    # Чиним blank-токен на MPS: булева индексация в torch 2.6 MPS пропускает
    # blank_id=33 через skip_mask, и он доходит до vocab[33] → ConfigIndexError.
    _patch_gigaam_blank_token_crash()
    # Чиним pyannote-VAD путь (директория снапшота → файл весов), иначе
    # transcribe_longform падает на HFValidationError.
    _patch_gigaam_vad_segmentation()
    # Кладём бандл-ffmpeg в PATH: gigaam зовёт `"ffmpeg"` голой командой, а на
    # macOS Electron PATH урезан → FileNotFoundError: 'ffmpeg'.
    _ensure_ffmpeg_on_path()

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
            "GigaAM не установлен. Обычно его ставит установщик приложения при "
            "первом запуске — если этой ошибки не должно быть, перезапустите "
            "подготовку окружения (кнопка «Повторить» в окне первого запуска).\n"
            "Ручная установка (git-версия — только она умеет пословные тайминги):\n"
            "  pip install hydra-core onnx\n"
            "  pip install --no-deps git+https://github.com/salute-developers/GigaAM.git\n"
            "или переключите движок на WhisperX."
        ) from exc

    model = _load_gigaam_model(model_name, device)

    logger.info("Запускаю GigaAM transcribe_longform (word_timestamps=True): %s", audio_path.name)
    try:
        result = model.transcribe_longform(str(audio_path), word_timestamps=True)
    except TypeError as exc:
        # PyPI-релиз 0.2.0 без word-timestamps: transcribe(wav) -> str.
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
