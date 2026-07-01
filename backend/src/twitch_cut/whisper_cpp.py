"""whisper.cpp backend.

Вызывает whisper-cli.exe (бинарь из ggerganov/whisper.cpp release) через
subprocess, читает его JSON output (--output-json-full) и нормализует
в наш transcript-формат:

    {
      "segments": [
        {"id": "seg_000000", "start": 1.2, "end": 4.7,
         "words": [{"word": "бля", "start": 1.20, "end": 1.45, "score": 0.91}, ...]},
        ...
      ]
    }

После этого transcription.iter_words() и detect_profanity() работают
одинаково, независимо от backend-а.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

from .cache import file_fingerprint, read_json, stable_hash, write_json

logger = logging.getLogger(__name__)


class WhisperCppError(RuntimeError):
    """Ошибка вызова whisper.cpp."""


# whisper.cpp служебные токены, которые ВНЕ зависимости от наличия ведущего
# пробела не должны попадать в текст слова. Сюда входят:
#   [_TT_300], [_BEG_], [_EOT_], [_SOT_], [_TRANSLATE_], [_TRANSCRIBE_], ...
#   <|notimestamps|>, <|0.00|>, <|ru|>, <|en|>, <|startoftranscript|>, ...
# Применяем как .sub() к сырому тексту токена — таким образом обрабатывается и
# случай, когда спецтокен ПРИКЛЕЕН к слову (например 'ладно[_TT_300]' или
# '[_TT_300]ладно') без ведущего пробела, что иначе ломает BPE-склейку.
_SPECIAL_TOKEN_INLINE_RE = re.compile(r"\[_[A-Z0-9_]+\]|<\|[^|]*\|>")
# whisper.cpp иногда выдаёт пустые токены или один пробел.
_BLANK_TOKEN_RE = re.compile(r"^\s*$")


def _strip_special_tokens(text: str) -> str:
    """Удалить все служебные whisper.cpp токены из строки.

    Работает с сырым текстом, до lstrip и без зависимости от ведущего пробела:
    спецтокен может быть как самостоятельным '[_TT_300]', так и прилипшим
    к слову 'ладно[_TT_300]'/'[_TT_300]ладно' — в обоих случаях он удаляется,
    а оставшиеся буквы корректно собираются BPE-склейкой.
    """
    return _SPECIAL_TOKEN_INLINE_RE.sub("", text)


def _resolve_binary(explicit: Path | None) -> Path:
    """Найти whisper-cli бинарь: явный путь → PATH → ошибка."""
    if explicit is not None:
        path = explicit.expanduser()
        if not path.exists():
            raise WhisperCppError(
                f"--whisper-cpp-bin: файл не найден: {path}"
            )
        return path.resolve()
    for candidate in ("whisper-cli", "whisper-cli.exe", "main", "main.exe"):
        found = shutil.which(candidate)
        if found:
            return Path(found).resolve()
    raise WhisperCppError(
        "Не указан --whisper-cpp-bin и в PATH не найден whisper-cli/whisper-cli.exe. "
        "Скачайте release-бинарь с https://github.com/ggerganov/whisper.cpp/releases "
        "(CUDA-сборка для Windows) и передайте его путь через --whisper-cpp-bin."
    )


def _resolve_model(explicit: Path) -> Path:
    path = explicit.expanduser()
    if not path.exists():
        raise WhisperCppError(
            f"--whisper-cpp-model: файл не найден: {path}. "
            "Скачайте ggml-large-v3.bin с "
            "https://huggingface.co/ggerganov/whisper.cpp/tree/main"
        )
    return path.resolve()


def _ms_to_seconds(value: Any) -> float | None:
    """whisper.cpp пишет таймстампы как int миллисекунд."""
    if value is None:
        return None
    try:
        return float(value) / 1000.0
    except (TypeError, ValueError):
        return None


def _build_words_from_tokens(tokens: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """whisper.cpp выдаёт subword-токены — собираем их в слова по пробелам.

    Каждый токен:
        {"text": "бля", "offsets": {"from": 1200, "to": 1450}, "p": 0.91}
    Слово начинается с пробельного токена (или с самого первого токена)
    и продолжается non-space токенами BPE.
    """
    words: list[dict[str, Any]] = []
    current_text = ""
    current_start_ms: int | None = None
    current_end_ms: int | None = None
    current_probs: list[float] = []

    def flush() -> None:
        nonlocal current_text, current_start_ms, current_end_ms, current_probs
        text = current_text.strip()
        if text and current_start_ms is not None and current_end_ms is not None:
            score = sum(current_probs) / len(current_probs) if current_probs else None
            words.append(
                {
                    "word": text,
                    "start": current_start_ms / 1000.0,
                    "end": current_end_ms / 1000.0,
                    "score": float(score) if score is not None else None,
                }
            )
        current_text = ""
        current_start_ms = None
        current_end_ms = None
        current_probs = []

    for token in tokens:
        raw = str(token.get("text", ""))
        if not raw:
            continue
        # Вырезаем служебные whisper.cpp токены ('[_TT_*]', '[_BEG_]', '<|...|>')
        # из текста ДО любых других проверок. Это критично, потому что они
        # могут быть приклеены к слову без ведущего пробела
        # (например 'ладно[_TT_300]') и тогда без удаления попадают в слово.
        had_content = bool(raw.strip())
        raw = _strip_special_tokens(raw)
        if not raw:
            # Был непустой текст, но он целиком состоял из спецтокенов —
            # это не граница слова, просто пропускаем.
            if had_content:
                continue
            continue
        offsets = token.get("offsets") or {}
        t_from = offsets.get("from")
        t_to = offsets.get("to")
        if t_from is None or t_to is None:
            # без тайминга токен бесполезен для word-level mute.
            continue
        prob = token.get("p")

        starts_new_word = raw.startswith(" ") or raw.startswith(" ")
        clean = raw.lstrip()

        if starts_new_word and current_text:
            flush()

        # Пустой токен после strip — это голая граница слова, пропускаем.
        if _BLANK_TOKEN_RE.match(clean):
            continue

        if not current_text:
            current_start_ms = int(t_from)
        current_text += clean
        current_end_ms = int(t_to)
        if prob is not None:
            try:
                current_probs.append(float(prob))
            except (TypeError, ValueError):
                pass

    flush()
    return words


def normalize_whisper_cpp_json(raw: dict[str, Any]) -> dict[str, Any]:
    """Привести JSON whisper.cpp к формату WhisperX-подобного transcript."""
    transcription = raw.get("transcription")
    if not isinstance(transcription, list):
        raise WhisperCppError(
            "Неожиданный JSON whisper.cpp: нет массива 'transcription'"
        )
    segments: list[dict[str, Any]] = []
    for idx, segment in enumerate(transcription):
        if not isinstance(segment, dict):
            continue
        offsets = segment.get("offsets") or {}
        # whisper-cli даёт offsets в миллисекундах; timestamps.from/to — это
        # отформатированные строки "HH:MM:SS,mmm" (для людей). Берём offsets.
        seg_start = _ms_to_seconds(offsets.get("from"))
        seg_end = _ms_to_seconds(offsets.get("to"))
        text = str(segment.get("text", "")).strip()
        tokens = segment.get("tokens") or []
        words = _build_words_from_tokens(tokens) if isinstance(tokens, list) else []
        # Если whisper.cpp не выдал tokens (старые бинари), оставим словарь
        # без word-timing — detector упадёт на segment-fallback таймингах.
        segments.append(
            {
                "id": f"seg_{idx:06d}",
                "start": seg_start,
                "end": seg_end,
                "text": text,
                "words": words,
            }
        )
    return {
        "segments": segments,
        "language": raw.get("result", {}).get("language") or raw.get("language"),
        "backend": "whisper.cpp",
    }


def _run_whisper_cpp(
    binary: Path,
    model: Path,
    audio_path: Path,
    language: str,
    workdir: Path,
    threads: int | None,
    extra_args: list[str],
) -> dict[str, Any]:
    """Запускает whisper-cli, читает <audio>.json, возвращает raw JSON."""
    output_dir = workdir / "cache" / "whispercpp"
    output_dir.mkdir(parents=True, exist_ok=True)
    # whisper-cli пишет <output-file>.json при --output-json-full.
    # Имя файла привязываем к фингерпринту аудио чтобы не плодить мусор.
    fp = file_fingerprint(audio_path)
    output_prefix = output_dir / f"out_{stable_hash(fp)}"
    json_path = output_prefix.with_suffix(".json")
    if json_path.exists():
        json_path.unlink()

    cmd: list[str] = [
        str(binary),
        "-m", str(model),
        "-f", str(audio_path),
        "-l", language,
        "--output-json-full",
        "--output-file", str(output_prefix),
        "--print-progress",
    ]
    if threads is not None and threads > 0:
        cmd += ["-t", str(threads)]
    cmd += list(extra_args)

    logger.info("Запускаю whisper.cpp: %s", " ".join(cmd))
    env = os.environ.copy()
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            # whisper-cli.exe пишет stdout/stderr в UTF-8 (прогресс, метаданные
            # модели и т.п.). На русской Windows локаль по умолчанию cp1251 —
            # и без явной кодировки Python падает на первом не-cp1251 байте
            # с UnicodeDecodeError в reader-треде subprocess. errors=replace
            # гарантирует, что мы дочитаем поток до конца, даже если whisper
            # выплюнет что-то странное.
            encoding="utf-8",
            errors="replace",
            check=False,
            env=env,
        )
    except FileNotFoundError as exc:
        raise WhisperCppError(f"Не удалось запустить {binary}: {exc}") from exc

    if proc.returncode != 0:
        tail_stdout = (proc.stdout or "").strip().splitlines()[-20:]
        tail_stderr = (proc.stderr or "").strip().splitlines()[-20:]
        raise WhisperCppError(
            "whisper.cpp завершился с ошибкой "
            f"(exit={proc.returncode}).\nstdout:\n"
            + "\n".join(tail_stdout)
            + "\nstderr:\n"
            + "\n".join(tail_stderr)
        )

    if not json_path.exists():
        raise WhisperCppError(
            f"whisper.cpp не создал JSON файл {json_path}. "
            "Возможно, ваш бинарь не поддерживает --output-json-full; "
            "обновите whisper.cpp до свежего релиза."
        )

    return read_json(json_path)


def transcribe_with_whisper_cpp(
    audio_path: Path,
    workdir: Path,
    binary: Path | None,
    model_path: Path,
    language: str = "ru",
    threads: int | None = None,
    extra_args: list[str] | None = None,
    force: bool = False,
) -> tuple[dict[str, Any], str, Path]:
    """Запускает whisper.cpp на готовом WAV и возвращает (transcript, key, cache_path)."""
    binary_path = _resolve_binary(binary)
    model_resolved = _resolve_model(model_path)

    key_data = {
        "stage": "whispercpp_transcription",
        "audio": file_fingerprint(audio_path),
        "binary": file_fingerprint(binary_path),
        "model": file_fingerprint(model_resolved),
        "language": language,
        "threads": threads,
        "extra_args": extra_args or [],
    }
    key = stable_hash(key_data)
    cache_path = workdir / "cache" / f"transcript_whispercpp_{key}.json"

    if cache_path.exists() and not force:
        logger.info("whisper.cpp transcript уже есть в cache: %s", cache_path)
        return read_json(cache_path), key, cache_path

    raw = _run_whisper_cpp(
        binary=binary_path,
        model=model_resolved,
        audio_path=audio_path,
        language=language,
        workdir=workdir,
        threads=threads,
        extra_args=extra_args or [],
    )
    normalized = normalize_whisper_cpp_json(raw)
    write_json(cache_path, normalized)
    return normalized, key, cache_path
