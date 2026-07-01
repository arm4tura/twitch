from __future__ import annotations

import inspect
import logging
from pathlib import Path
from typing import Any

from .cache import file_fingerprint, read_json, stable_hash, write_json

logger = logging.getLogger(__name__)


_TORCH_SAFE_GLOBALS_PATCHED = False
_TORCH_LOAD_PATCHED = False
_SPEECHBRAIN_K2_STUBBED = False


def _stub_missing_speechbrain_optionals() -> None:
    """Обезвредить LazyModule в speechbrain, чтобы inspect.stack() не падал.

    ПРОБЛЕМА: pyannote → pytorch_lightning вызывает inspect.stack() внутри
    Model.load_from_checkpoint() (для детектирования JIT-режима). Стек проходит
    по всем sys.modules и обращается к __file__ / attr каждого. speechbrain
    1.0+ регистрирует LazyModule-обёртки для десятков опциональных интеграций
    (k2_fsa, kenlm, flair, nemo, ...). Обращение к ЛЮБОМУ атрибуту LazyModule
    триггерит реальный импорт, который падает ImportError'ом на отсутствующем
    optional-пакете. Ошибка прокидывается наружу и валит pyannote.

    ФИКС: monkey-patch метода `__getattr__` у самого класса LazyModule
    (и DeprecatedModuleRedirect). Если реальный ленивый импорт падает —
    возвращаем безопасное значение вместо raise:
      - __file__ / __name__ / __path__ / __spec__ / __loader__ — строки/None
      - всё остальное — AttributeError (штатное поведение)
    Это универсальнее, чем стабить модули по именам поштучно: покрывает
    все нынешние и будущие проблемные интеграции сразу.

    Идемпотентно.
    """
    global _SPEECHBRAIN_K2_STUBBED
    if _SPEECHBRAIN_K2_STUBBED:
        return

    try:
        from speechbrain.utils import importutils as _sb_importutils
    except ImportError:
        # speechbrain не установлен — стабить нечего.
        _SPEECHBRAIN_K2_STUBBED = True
        return

    LazyModule = getattr(_sb_importutils, "LazyModule", None)
    if LazyModule is None:
        _SPEECHBRAIN_K2_STUBBED = True
        return

    _original_getattr = LazyModule.__getattr__

    # Атрибуты, которые inspect / importlib / pickle трогают на любом модуле —
    # для них возвращаем безопасные заглушки, а не поднимаем ImportError.
    _SAFE_MODULE_ATTRS = {
        "__file__", "__name__", "__path__", "__spec__", "__loader__",
        "__package__", "__doc__", "__all__", "__dict__",
    }

    def _safe_getattr(self, attr: str) -> Any:
        try:
            return _original_getattr(self, attr)
        except ImportError:
            # Ленивый импорт упал — optional-пакет не установлен.
            # Для attrs, которые нужны inspect/importlib, отдаём безопасные
            # дефолты. Для всех прочих — обычный AttributeError (getattr
            # снаружи получит AttributeError, а не ImportError, и не будет
            # ронять всю программу).
            if attr == "__file__":
                return f"<lazy-stub:{self.target}>"
            if attr == "__name__":
                return self.target
            if attr == "__package__":
                return self.target.rsplit(".", 1)[0] if "." in self.target else ""
            if attr in _SAFE_MODULE_ATTRS:
                return None
            raise AttributeError(
                f"lazy module {self.target!r} could not be loaded (missing optional dependency)"
            )

    LazyModule.__getattr__ = _safe_getattr  # type: ignore[assignment]

    # DeprecatedModuleRedirect наследуется от LazyModule и переопределяет
    # ensure_module — но НЕ __getattr__ (он унаследован). Ловим на всякий случай
    # если это когда-нибудь изменится.
    Deprecated = getattr(_sb_importutils, "DeprecatedModuleRedirect", None)
    if Deprecated is not None and "__getattr__" in Deprecated.__dict__:
        _dep_original = Deprecated.__getattr__

        def _dep_safe(self, attr: str) -> Any:
            try:
                return _dep_original(self, attr)
            except ImportError:
                if attr == "__file__":
                    return f"<lazy-stub:{self.target}>"
                if attr == "__name__":
                    return self.target
                if attr in _SAFE_MODULE_ATTRS:
                    return None
                raise AttributeError(
                    f"deprecated lazy module {self.target!r} could not be loaded"
                )

        Deprecated.__getattr__ = _dep_safe  # type: ignore[assignment]

    _SPEECHBRAIN_K2_STUBBED = True
    logger.info(
        "speechbrain LazyModule.__getattr__ обёрнут: ImportError на optional-пакетах "
        "(flair/k2/kenlm/nemo/...) больше не будет ронять pyannote"
    )


def _patch_torch_load_weights_only_false() -> None:
    """Monkey-patch torch.load, чтобы pyannote VAD checkpoint загружался.

    ПРОБЛЕМА: PyTorch 2.6+ по умолчанию грузит с weights_only=True. Pyannote
    checkpoint содержит десятки нестандартных типов (ListConfig, DictConfig,
    typing.Any, pathlib.PosixPath, и т.д.) — whitelist через add_safe_globals
    бесконечен: whitelist один тип, вылезает следующий.

    Прошлая версия патча пыталась определять pyannote по имени файла, но
    lightning_fabric/pyannote открывает файл через fsspec ДО torch.load,
    и в torch.load уже прилетает file-object без нормального пути в .name.
    Поэтому теперь патчим ГЛОБАЛЬНО: любой torch.load без явного
    weights_only=True получает weights_only=False.

    Безопасность: единственные torch.load-ы в нашем pipeline идут внутри
    WhisperX/pyannote/lightning для загрузки VAD и alignment моделей —
    все качаются с trusted HuggingFace-репозиториев. Whisper-веса грузятся
    faster-whisper'ом через ctranslate2 и НЕ идут через python torch.load.
    Так что глобальный откат к старому pre-2.6 поведению безопасен для
    этого конкретного проекта.

    Идемпотентно.
    """
    global _TORCH_LOAD_PATCHED
    if _TORCH_LOAD_PATCHED:
        return
    try:
        import torch
    except ImportError:
        return

    _original_load = torch.load

    def _patched_load(*args: Any, **kwargs: Any) -> Any:
        # ВАЖНО: pyannote/lightning_fabric ЯВНО передают weights_only=True в
        # torch.load (см. lightning_fabric/utilities/cloud_io.py — там
        # torch.load(f, weights_only=weights_only, ...) с weights_only=True
        # для локальных файлов). setdefault() не помог бы — ключ уже есть.
        # Поэтому перезаписываем всегда: пользователь этой обёртки сам решил,
        # что грузит trusted checkpoint, и хочет pre-2.6 поведение.
        kwargs["weights_only"] = False
        return _original_load(*args, **kwargs)

    torch.load = _patched_load  # type: ignore[assignment]
    _TORCH_LOAD_PATCHED = True
    logger.info(
        "torch.load обёрнут: weights_only=False по умолчанию (для загрузки pyannote VAD/alignment checkpoints)"
    )


def _patch_torch_safe_globals_for_pyannote() -> None:
    """Разрешить unpickle классов, нужных pyannote/whisperx чекпоинтам.

    PyTorch 2.6+ по умолчанию грузит с weights_only=True и падает на
    omegaconf.listconfig.ListConfig внутри pyannote VAD checkpoint.
    Ошибка: `Unsupported global: GLOBAL omegaconf.listconfig.ListConfig`.

    Whitelist'им конкретные классы через add_safe_globals — это безопаснее,
    чем ставить torch.load(..., weights_only=False) глобально: unpickle
    остаётся ограниченным, просто расширяем список допустимых типов.

    Идемпотентно: повторные вызовы — no-op. На старых torch (<2.6) без
    add_safe_globals функция ничего не делает.
    """
    global _TORCH_SAFE_GLOBALS_PATCHED
    if _TORCH_SAFE_GLOBALS_PATCHED:
        return

    try:
        import torch
    except ImportError:
        return

    add_safe_globals = getattr(torch.serialization, "add_safe_globals", None)
    if add_safe_globals is None:
        # torch <2.6 — там weights_only=False по умолчанию, патч не нужен.
        _TORCH_SAFE_GLOBALS_PATCHED = True
        return

    safe_classes: list[Any] = []

    # omegaconf.listconfig.ListConfig — конкретный класс из трейса.
    try:
        from omegaconf.listconfig import ListConfig  # type: ignore
        safe_classes.append(ListConfig)
    except ImportError:
        pass

    # По опыту сообщества pyannote/whisperx тянет ещё несколько omegaconf
    # и pyannote-типов в разных версиях чекпоинтов — регистрируем всё, что
    # доступно. Если класса нет — просто пропускаем.
    _optional_imports: list[tuple[str, str]] = [
        ("omegaconf.dictconfig", "DictConfig"),
        ("omegaconf.base", "ContainerMetadata"),
        ("omegaconf.base", "Metadata"),
        ("omegaconf.nodes", "AnyNode"),
        ("omegaconf.nodes", "IntegerNode"),
        ("omegaconf.nodes", "FloatNode"),
        ("omegaconf.nodes", "StringNode"),
        ("omegaconf.nodes", "BooleanNode"),
    ]
    for module_name, cls_name in _optional_imports:
        try:
            module = __import__(module_name, fromlist=[cls_name])
            safe_classes.append(getattr(module, cls_name))
        except (ImportError, AttributeError):
            continue

    if safe_classes:
        add_safe_globals(safe_classes)
        logger.info(
            "torch.serialization.add_safe_globals: разрешено %d классов для pyannote checkpoint (%s)",
            len(safe_classes),
            ", ".join(c.__name__ for c in safe_classes),
        )

    _TORCH_SAFE_GLOBALS_PATCHED = True


def load_mock_transcript(path: Path) -> dict[str, Any]:
    data = read_json(path)
    if "segments" not in data:
        raise ValueError("Mock transcript must contain a 'segments' array")
    return data


def apply_whisperx_patches() -> None:
    """Применить все monkey-patches, нужные для WhisperX+pyannote+PyTorch 2.6.

    Идемпотентно (сами _patch_* функции стоят на _*_applied флагах).

    Обязательно вызывать ДО whisperx.load_model — WhisperX внутри load_model
    инициализирует pyannote VAD, который делает torch.load с weights_only=True
    и падает без наших патчей на unpickle omegaconf-объектов.

    Тройная защита:
    1) add_safe_globals — регистрируем известные omegaconf-типы для
       torch.load weights_only=True path.
    2) monkey-patch torch.load — форсим weights_only=False для pyannote путей,
       которые явно передают weights_only=True в вызов (обходит add_safe_globals).
    3) stub speechbrain optional integrations — иначе inspect.stack() внутри
       pytorch_lightning падает на LazyModule для k2_fsa, kenlm и т.п.
    """
    _patch_torch_safe_globals_for_pyannote()
    _patch_torch_load_weights_only_false()
    _stub_missing_speechbrain_optionals()


def _load_whisperx_model(
    whisperx: Any,
    model_name: str,
    device: str,
    compute_type: str,
    language: str,
    vad_method: str,
    asr_options: dict[str, Any] | None,
) -> Any:
    """Загружает WhisperX-модель с anti-hallucination asr_options.

    ГЛАВНЫЙ FIX: без asr_options={"condition_on_previous_text": False, ...}
    faster-whisper на длинных стримах галлюцинирует и сжимает тайминги на
    5+ секунд, теряя целые фразы (см. detection_2b4b1a5c89ffaa3a — фраза
    "Так, не я ещё папа. Тут одиннадцатиклассник..." была пропущена).

    Старые версии WhisperX могут не принимать asr_options или vad_method —
    делаем последовательный degrade: сначала пробуем всё, потом только
    asr_options, потом только vad_method, потом ничего.
    """

    # Fix PyTorch 2.6+ unpickle крэша при загрузке pyannote VAD checkpoint.
    # Обязательно вызвать ДО whisperx.load_model. Подробности в docstring
    # apply_whisperx_patches().
    apply_whisperx_patches()

    def _try(**extra: Any) -> Any:
        return whisperx.load_model(
            model_name,
            device=device,
            compute_type=compute_type,
            language=language,
            **extra,
        )

    attempts: list[dict[str, Any]] = []
    if asr_options:
        attempts.append({"vad_method": vad_method, "asr_options": asr_options})
        attempts.append({"asr_options": asr_options})
    attempts.append({"vad_method": vad_method})
    attempts.append({})

    last_error: TypeError | None = None
    for extra in attempts:
        try:
            model = _try(**extra)
            missing = []
            if asr_options and "asr_options" not in extra:
                missing.append("asr_options (anti-hallucination profile НЕ применён)")
            if vad_method != "pyannote" and "vad_method" not in extra:
                missing.append(f"vad_method={vad_method}")
            if missing:
                logger.warning(
                    "Установленный WhisperX не поддерживает: %s. Использую то, что удалось.",
                    ", ".join(missing),
                )
            return model
        except TypeError as exc:
            last_error = exc
            continue
    # Не должны сюда попасть — последний attempt пустой и должен работать.
    raise RuntimeError("Не удалось загрузить WhisperX ни с одним набором аргументов") from last_error


def _transcribe_with_compatible_args(
    model: Any,
    audio: Any,
    language: str,
    batch_size: int,
    vad_filter: bool,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {"language": language, "batch_size": batch_size}
    try:
        parameters = inspect.signature(model.transcribe).parameters
    except (TypeError, ValueError):
        parameters = {}

    if "vad_filter" in parameters:
        kwargs["vad_filter"] = vad_filter
    elif not vad_filter:
        logger.warning(
            "Installed WhisperX transcribe() does not support vad_filter; "
            "--no-vad-filter cannot disable VAD for this version."
        )

    return model.transcribe(audio, **kwargs)


def transcribe_audio(
    audio_path: Path,
    workdir: Path,
    model_name: str = "large-v3",
    language: str = "ru",
    device: str = "cuda",
    compute_type: str = "float16",
    batch_size: int = 16,
    vad_filter: bool = True,
    vad_method: str = "pyannote",
    asr_options: dict[str, Any] | None = None,
    force: bool = False,
) -> tuple[dict[str, Any], str, Path]:
    key_data = {
        "stage": "whisperx_transcription",
        "audio": file_fingerprint(audio_path),
        "model": model_name,
        "language": language,
        "device": device,
        "compute_type": compute_type,
        "batch_size": batch_size,
        "vad_filter": vad_filter,
        "vad_method": vad_method,
        # asr_options попадают в cache key — смена anti-hallucination профиля
        # должна инвалидировать старые кэши, иначе будем читать битый транскрипт.
        "asr_options": asr_options or {},
    }
    key = stable_hash(key_data)
    cache_path = workdir / "cache" / f"transcript_{key}.json"

    if cache_path.exists() and not force:
        logger.info("Транскрипт уже есть в cache: %s", cache_path)
        return read_json(cache_path), key, cache_path

    try:
        import whisperx  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "WhisperX не установлен. Установите зависимости из requirements.txt "
            "или используйте --mock-transcript для smoke-теста."
        ) from exc

    logger.info(
        "Загружаю WhisperX model=%s device=%s compute=%s vad_method=%s asr_options=%s",
        model_name,
        device,
        compute_type,
        vad_method,
        sorted(asr_options.keys()) if asr_options else [],
    )
    model = _load_whisperx_model(
        whisperx, model_name, device, compute_type, language, vad_method, asr_options
    )
    audio = whisperx.load_audio(str(audio_path))

    logger.info("Запускаю transcription, language=%s batch_size=%s vad_filter=%s", language, batch_size, vad_filter)
    result = _transcribe_with_compatible_args(model, audio, language, batch_size, vad_filter)

    logger.info("Запускаю forced alignment для word-level таймингов")
    align_model, metadata = whisperx.load_align_model(language_code=language, device=device)
    aligned = whisperx.align(
        result["segments"],
        align_model,
        metadata,
        audio,
        device,
        return_char_alignments=False,
    )

    write_json(cache_path, aligned)
    return aligned, key, cache_path


def iter_words(transcript: dict[str, Any]):
    for segment_index, segment in enumerate(transcript.get("segments", [])):
        segment_id = segment.get("id", f"seg_{segment_index:06d}")
        segment_start = segment.get("start")
        segment_end = segment.get("end")
        for word_index, word in enumerate(segment.get("words", [])):
            yield {
                "segment_id": segment_id,
                "segment_index": segment_index,
                "word_index": word_index,
                "text": word.get("word") or word.get("text") or "",
                "start": word.get("start", segment_start),
                "end": word.get("end", segment_end),
                "segment_start": segment_start,
                "segment_end": segment_end,
                "score": word.get("score", word.get("confidence")),
                "timing_source": "word" if "start" in word and "end" in word else "segment_fallback",
            }
