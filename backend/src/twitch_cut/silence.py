"""Детект тишины / мёртвого эфира через ffmpeg silencedetect.

Возвращает список временны́х сегментов [(start_sec, end_sec), ...], где уровень
звука ниже заданного порога дольше минимальной длительности. Используется для
автоматического вырезания пауз/мёртвых моментов из стримов — аналогично детекту
мата (profanity.py), но по громкости, а не по банвордам.

Кэширование через cache.py — hash аудио + параметры (threshold/duration) → список
сегментов, чтобы повторный детект на том же файле летал без запуска ffmpeg.
"""

from __future__ import annotations

import logging
import re
import subprocess
from pathlib import Path
from typing import Optional

from .cache import file_fingerprint, stable_hash, write_json, read_json
from .ffmpeg_tools import get_ffmpeg_path

logger = logging.getLogger(__name__)

# ffmpeg silencedetect выдаёт:
#   [silencedetect @ 0x...] silence_start: 12.345
#   [silencedetect @ 0x...] silence_end: 15.678 | silence_duration: 3.333
_SILENCE_START_RE = re.compile(r"silence_start:\s*(?P<start>[\d.]+)")
_SILENCE_END_RE = re.compile(r"silence_end:\s*(?P<end>[\d.]+)")


def detect_silence_ffmpeg(
    audio_path: Path,
    threshold_db: float = -30.0,
    min_duration: float = 0.5,
    workdir: Optional[Path] = None,
    force: bool = False,
) -> list[tuple[float, float]]:
    """Находит сегменты тишины через ffmpeg silencedetect.

    Args:
        audio_path: путь к аудио (wav/mp3/любой формат, который ffmpeg понимает)
        threshold_db: порог RMS в dBFS; всё ниже = тишина (обычно −25…−40)
        min_duration: минимальная длительность сегмента (сек); отсекает короткие паузы
        workdir: каталог для кэша (по умолчанию — родитель audio_path)
        force: пропустить кэш, перезапустить детект

    Returns:
        Список (start, end) в секундах (float). Пример:
            [(12.345, 15.678), (42.1, 48.9)]

    Raises:
        RuntimeError: если ffmpeg упал или не нашёл тишины (тогда список пустой, но
                      не ошибка — возможно аудио без пауз).
    """
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    if workdir is None:
        workdir = audio_path.parent / "cache"
    cache_dir = workdir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Кэш-ключ: fingerprint файла + параметры детекта.
    key_data = {
        "stage": "detect_silence_ffmpeg",
        "audio": file_fingerprint(audio_path),
        "threshold_db": threshold_db,
        "min_duration": min_duration,
    }
    key = stable_hash(key_data)
    cache_path = cache_dir / f"silence_{key}.json"

    if cache_path.exists() and not force:
        logger.info("Silence segments уже детектены, используем cache: %s", cache_path)
        cached = read_json(cache_path)
        return [tuple(seg) for seg in cached["segments"]]

    ffmpeg = get_ffmpeg_path()
    logger.info("Детектим тишину через ffmpeg silencedetect: %s", audio_path)

    # ffmpeg -i audio.wav -af silencedetect=n=-30dB:d=0.5 -f null - 2>&1
    # -vn: игнорировать видео (если есть)
    # -f null: не записываем output, только анализ
    cmd = [
        ffmpeg,
        "-i",
        str(audio_path),
        "-vn",
        "-af",
        f"silencedetect=n={threshold_db}dB:d={min_duration}",
        "-f",
        "null",
        "-",
    ]

    try:
        completed = subprocess.run(
            cmd,
            check=False,  # silencedetect не падает с ненулевым кодом, если тишины нет
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.strip() if exc.stderr else ""
        raise RuntimeError(f"ffmpeg silencedetect failed: {stderr}") from exc

    # Парсинг stderr: ищем пары silence_start / silence_end.
    # ffmpeg пишет start, потом end — но могут быть неполные пары (файл обрывается
    # на тишине). Собираем аккуратно: start → pending, end → закрываем.
    output = f"{completed.stdout}\n{completed.stderr}"
    segments: list[tuple[float, float]] = []
    pending_start: Optional[float] = None

    for line in output.splitlines():
        start_match = _SILENCE_START_RE.search(line)
        if start_match:
            pending_start = float(start_match.group("start"))
            continue

        end_match = _SILENCE_END_RE.search(line)
        if end_match and pending_start is not None:
            end = float(end_match.group("end"))
            segments.append((pending_start, end))
            pending_start = None

    # Если файл кончился на тишине — pending_start висит без конца. Можно игнорить
    # (консервативно) или достроить до конца файла. Пока игнорим — нужен duration.
    if pending_start is not None:
        logger.warning(
            "Silence segment at %.2f has no end (file ends in silence?), dropping",
            pending_start,
        )

    logger.info("Найдено %d silence segments", len(segments))
    write_json(
        cache_path,
        {
            "stage": "detect_silence_ffmpeg",
            "key": key,
            "inputs": key_data,
            "segments": segments,
        },
    )
    return segments


def merge_close_segments(
    segments: list[tuple[float, float]],
    gap_threshold: float = 0.5,
) -> list[tuple[float, float]]:
    """Склеивает близкие сегменты тишины в один (gap ≤ threshold).

    Пример: [(10, 12), (12.3, 15)] при gap_threshold=0.5 → [(10, 15)]
    Полезно, чтобы не плодить десятки мелких вырезов из-за коротких всплесков.

    Args:
        segments: список (start, end), отсортированный по start
        gap_threshold: максимальный разрыв (сек) для склейки

    Returns:
        Объединённый список (start, end).
    """
    if not segments:
        return []

    # Сортируем на всякий (ffmpeg всегда даёт sorted, но на входе может быть что угодно)
    sorted_segs = sorted(segments, key=lambda x: x[0])
    merged: list[tuple[float, float]] = [sorted_segs[0]]

    for curr_start, curr_end in sorted_segs[1:]:
        prev_start, prev_end = merged[-1]
        gap = curr_start - prev_end
        if gap <= gap_threshold:
            # Склеиваем: расширяем предыдущий до max(prev_end, curr_end)
            merged[-1] = (prev_start, max(prev_end, curr_end))
        else:
            merged.append((curr_start, curr_end))

    return merged


def add_padding(
    segments: list[tuple[float, float]],
    padding_before: float = 0.1,
    padding_after: float = 0.1,
) -> list[tuple[float, float]]:
    """Добавляет padding до/после каждого сегмента (аналог mute_padding).

    Args:
        segments: список (start, end)
        padding_before: сколько секунд добавить до start (не уходит <0)
        padding_after: сколько секунд добавить после end

    Returns:
        Список (start - pad_before, end + pad_after), clamped к [0, +∞).
    """
    return [
        (max(0.0, s - padding_before), e + padding_after) for s, e in segments
    ]
