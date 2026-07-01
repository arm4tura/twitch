"""Извлечение peaks для отрисовки waveform в UI-таймлайне.

Стратегия:
- ffmpeg → raw PCM s16le mono 8000 Hz (маленький sample rate — качества хватает
  для визуализации, ускоряет обработку часового стрима до ~200 ms).
- Полученные samples делятся на N bucket'ов (`peaks`, по умолчанию 1024).
  В каждом — max(abs(sample)) — так пики визуально сохраняются, а тихие места
  выглядят тонкой линией.
- Нормализация к [-1, 1] через деление на 32768 (max int16).
- Итог: JSON `{peaks: [float × N], duration_s, sample_rate}`.

Кэш:
- Ключ: stable_hash({stream_path, mtime, size, peaks}). Меняется путь, файл
  или запрошенное разрешение — пересчёт.
- Файл кэша: `<workdir>/cache/waveform_<hash>.json` если стрим лежит внутри
  workdir'a джобы. Иначе — глобальный `~/.cache/twitch_cut/waveform_<hash>.json`
  (создаётся при первом использовании; тесты подменяют через env).

Осознанно НЕ пишем WebAudio-совместимый формат — фронт принимает голый массив
float'ов, передаёт wavesurfer'у через `peaks:` option (см. TimelineScreen).
"""

from __future__ import annotations

import array
import logging
import os
import subprocess
from pathlib import Path
from typing import Optional

from ..cache import read_json, stable_hash, write_json
from ..ffmpeg_tools import get_ffmpeg_path, probe_media_duration

logger = logging.getLogger(__name__)

# Даунсэмпл: 8kHz достаточно для peaks, а декодировать в 44.1 kHz для
# визуализации — потеря 5-6× времени на ffmpeg-transcode.
_PCM_SAMPLE_RATE = 8000
_PCM_BYTES_PER_SAMPLE = 2  # s16le
_MAX_INT16 = 32768.0


def _cache_file(stream_path: Path, peaks: int, cache_root: Optional[Path] = None) -> Path:
    """Собрать путь к кэшу peaks-файла. Ключ — путь+mtime+size+peaks."""
    stat = stream_path.stat()
    key = stable_hash(
        {
            "stage": "waveform",
            "path": str(stream_path.resolve()),
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "peaks": peaks,
            "sample_rate": _PCM_SAMPLE_RATE,
        }
    )
    root = cache_root or _default_cache_root()
    root.mkdir(parents=True, exist_ok=True)
    return root / f"waveform_{key}.json"


def _default_cache_root() -> Path:
    """Каталог глобального кэша peaks. Можно переопределить env-переменной."""
    override = os.environ.get("TWITCH_CUT_WAVEFORM_CACHE")
    if override:
        return Path(override)
    return Path.home() / ".cache" / "twitch_cut" / "waveform"


def _run_ffmpeg_pcm(stream_path: Path) -> bytes:
    """Прогнать ffmpeg → raw PCM s16le 8kHz mono, вернуть весь bytestream.

    Для часового mp4 это ~57 MB — влезает в память без проблем. Стриминг
    не нужен: peaks считаются сразу, никакого пайплайна нет.
    """
    ffmpeg = get_ffmpeg_path()
    cmd = [
        ffmpeg,
        "-nostdin",
        "-hide_banner",
        "-loglevel", "error",
        "-i", str(stream_path),
        "-vn",
        "-ac", "1",
        "-ar", str(_PCM_SAMPLE_RATE),
        "-f", "s16le",
        "-acodec", "pcm_s16le",
        "pipe:1",
    ]
    completed = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if completed.returncode != 0:
        stderr = completed.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"ffmpeg waveform extraction failed: {stderr}")
    return completed.stdout


def _bucket_peaks(pcm: bytes, buckets: int) -> list[float]:
    """Свести PCM в N bucket'ов, взяв max(abs) в каждом. Нормализация [-1..1].

    Если samples < buckets — расширяем массив пиков нулями до нужной длины
    (waveform окажется короче реальной длительности, но фронт не сломается).
    """
    if buckets <= 0:
        raise ValueError("buckets must be > 0")
    if not pcm:
        return [0.0] * buckets
    samples = array.array("h")
    samples.frombytes(pcm)
    total = len(samples)
    if total == 0:
        return [0.0] * buckets
    step = total / buckets
    out: list[float] = []
    for i in range(buckets):
        lo = int(i * step)
        hi = int((i + 1) * step)
        if hi <= lo:
            hi = lo + 1
        if hi > total:
            hi = total
        # max(abs) — сохраняет визуальный «удар» пика в bucket'e; если брать
        # avg(abs), волна получается «пухлой» и нечитаемой на тихих участках.
        chunk_max = 0
        for j in range(lo, hi):
            v = samples[j]
            av = -v if v < 0 else v
            if av > chunk_max:
                chunk_max = av
        out.append(chunk_max / _MAX_INT16)
    return out


def compute_waveform(
    stream_path: Path,
    peaks: int = 1024,
    *,
    cache_root: Optional[Path] = None,
    force: bool = False,
) -> dict:
    """Публичный API: вернуть peaks JSON. Использует кэш если возможно.

    :param force: игнорировать кэш и пересчитать (для тестов и «Обновить»).
    """
    if not stream_path.exists():
        raise FileNotFoundError(f"waveform source not found: {stream_path}")
    cache_file = _cache_file(stream_path, peaks, cache_root=cache_root)
    if cache_file.exists() and not force:
        try:
            cached = read_json(cache_file)
            # На всякий: если формат старый — пересчитаем.
            if isinstance(cached, dict) and isinstance(cached.get("peaks"), list):
                return cached
        except Exception as exc:  # noqa: BLE001
            logger.warning("waveform cache read failed (%s), recomputing", exc)

    pcm = _run_ffmpeg_pcm(stream_path)
    peaks_arr = _bucket_peaks(pcm, peaks)
    duration_s = len(pcm) / _PCM_BYTES_PER_SAMPLE / _PCM_SAMPLE_RATE
    # Fallback duration через ffmpeg probe, если PCM пустой (не должно случаться).
    if duration_s <= 0:
        try:
            duration_s = probe_media_duration(stream_path).to_seconds()
        except Exception:
            duration_s = 0.0

    doc = {
        "peaks": peaks_arr,
        "duration_s": duration_s,
        "sample_rate": _PCM_SAMPLE_RATE,
        "source": str(stream_path.resolve()),
    }
    try:
        write_json(cache_file, doc)
    except Exception as exc:  # noqa: BLE001
        logger.warning("waveform cache write failed: %s", exc)
    return doc
