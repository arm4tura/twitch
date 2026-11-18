"""Извлечение peaks для отрисовки waveform в UI-таймлайне.

Два режима:
1. `compute_waveform` — ВЕСЬ стрим в N бакетов («базовый слой»). Кэшируется на
   диск, грузится один раз при открытии таймлайна.
2. `compute_waveform_range` — ТОЛЬКО видимое окно в N бакетов («слой детали»).
   Так работают Audacity/Reaper: на зуме подгружается крошка именно того
   диапазона, что на экране. НЕ кэшируется на диск (комбинаций бесконечно) —
   ffmpeg с `-ss` декодирует узкий слайс за десятки ms.

Стратегия:
- ffmpeg → raw PCM s16le mono. Базовый слой — 8000 Hz (качества хватает на
  весь стрим, час обрабатывается за ~200 ms). Слой детали поднимает sample rate
  адаптивно (до 48 kHz), чтобы на глубоком зуме в бакете оставалось ≥8 семплов.
- В каждом бакете считаем ЧЕТЫРЕ величины (одним проходом numpy):
    * min(sample) / max(sample) — настоящие границы контура. Именно
      асимметричный min/max, а не зеркальный ±max(|.|), даёт узнаваемую
      «рваную» форму Audacity; зеркало выглядит гладким блобом.
    * RMS = sqrt(mean(sample²)) — энергия участка: рисуется плотным ядром
      внутри контура (фирменный вид Audacity — контур + яркая середина).
    * max(|sample|) — legacy-поле `peaks` для обратной совместимости.
- Нормализация к [-1, 1] через деление на 32768 (max int16).
- Итог: JSON `{peaks, min, max, rms, duration_s, sample_rate}`. Поля
  `min`/`max`/`rms` опциональны для фронта (старые кэши их не содержат —
  тогда рисуется один симметричный контур по `peaks`).

Кэш (только базовый слой):
- Ключ: stable_hash({stream_path, mtime, size, peaks, format}). Меняется путь,
  файл, разрешение или формат полей — пересчёт.
- Файл кэша: `<workdir>/cache/waveform_<hash>.json` если стрим лежит внутри
  workdir'a джобы. Иначе — глобальный `~/.cache/twitch_cut/waveform_<hash>.json`
  (создаётся при первом использовании; тесты подменяют через env).
"""

from __future__ import annotations

import logging
import os
import subprocess
from pathlib import Path
from typing import Optional

import numpy as np

from ..cache import read_json, stable_hash, write_json
from ..ffmpeg_tools import get_ffmpeg_path, probe_media_duration

logger = logging.getLogger(__name__)

# Даунсэмпл: 8kHz достаточно для peaks, а декодировать в 44.1 kHz для
# визуализации — потеря 5-6× времени на ffmpeg-transcode.
_PCM_SAMPLE_RATE = 8000
_PCM_BYTES_PER_SAMPLE = 2  # s16le
_MAX_INT16 = 32768.0

# --- слой детали (окно зума) -------------------------------------------------
# На глубоком зуме 8 kHz мало: 0.4 с окна = 3200 семплов на ~2000 бакетов, т.е.
# 1.6 семпла на бакет — контур вырождается. Поднимаем sample rate так, чтобы в
# бакете осталось ≥ _RANGE_MIN_SPB семплов, но не выше _RANGE_MAX_SR (декод
# дороже, а на глаз разницы уже нет).
_RANGE_MIN_SPB = 8
_RANGE_MAX_SR = 48000


def _cache_file(stream_path: Path, peaks: int, cache_root: Optional[Path] = None) -> Path:
    """Собрать путь к кэшу peaks-файла. Ключ — путь+mtime+size+peaks."""
    stat = stream_path.stat()
    key = stable_hash(
        {
            "stage": "waveform",
            # format=3 добавил min[]/max[] (настоящий контур вместо зеркального
            # max(|.|)); format=2 добавлял rms[]. Bump ключа → старые кэши не
            # отдаются, а пересчитываются с полным набором полей.
            "format": 3,
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


def _bucket_waveform(
    pcm: bytes, buckets: int
) -> tuple[list[float], list[float], list[float], list[float]]:
    """Свести PCM в N bucket'ов: min, max, RMS и |peak|. Нормализация [-1..1].

    Возвращает `(mins, maxs, rms, peaks)` одинаковой длины `buckets`:
    - mins[i]  = min(sample)/32768 — нижняя граница контура (≤ 0);
    - maxs[i]  = max(sample)/32768 — верхняя граница контура (≥ 0);
    - rms[i]   = sqrt(mean(sample²))/32768 — энергия (плотное ядро);
    - peaks[i] = max(|sample|)/32768 — legacy-поле (симметричная волна).

    ИМЕННО min/max, а не ±max(|.|): зеркальная волна выглядит гладким «блобом»,
    а настоящая асимметрия — это и есть узнаваемая форма Audacity.

    Если samples < buckets — массивы дополняются нулями до нужной длины (waveform
    окажется короче реальной длительности, но фронт не сломается).
    """
    if buckets <= 0:
        raise ValueError("buckets must be > 0")
    zeros = [0.0] * buckets
    if not pcm:
        return zeros, list(zeros), list(zeros), list(zeros)
    # s16le → int16 БЕЗ копии буфера: count режет по числу элементов, а не
    # слайсом bytes (иначе часовой стрим копировался бы на 170+ MB).
    count = len(pcm) // _PCM_BYTES_PER_SAMPLE
    if count == 0:
        return zeros, list(zeros), list(zeros), list(zeros)
    samples = np.frombuffer(pcm, dtype="<i2", count=count)  # little-endian s16
    total = samples.shape[0]
    step = total / buckets
    mins_out: list[float] = []
    maxs_out: list[float] = []
    rms_out: list[float] = []
    peaks_out: list[float] = []
    for i in range(buckets):
        lo = int(i * step)
        hi = int((i + 1) * step)
        if hi <= lo:
            hi = lo + 1
        if hi > total:
            hi = total
        # Каждый bucket кастуем в float отдельно — пик памяти = один bucket, а не
        # весь стрим в float32 (часовой = сотни MB).
        chunk = samples[lo:hi].astype(np.float32)
        vmin = float(chunk.min())
        vmax = float(chunk.max())
        rms = float(np.sqrt(np.mean(chunk * chunk)))
        mins_out.append(vmin / _MAX_INT16)
        maxs_out.append(vmax / _MAX_INT16)
        rms_out.append(rms / _MAX_INT16)
        peaks_out.append(max(abs(vmin), abs(vmax)) / _MAX_INT16)
    return mins_out, maxs_out, rms_out, peaks_out


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
    mins_arr, maxs_arr, rms_arr, peaks_arr = _bucket_waveform(pcm, peaks)
    duration_s = len(pcm) / _PCM_BYTES_PER_SAMPLE / _PCM_SAMPLE_RATE
    # Fallback duration через ffmpeg probe, если PCM пустой (не должно случаться).
    if duration_s <= 0:
        try:
            duration_s = probe_media_duration(stream_path).to_seconds()
        except Exception:
            duration_s = 0.0

    doc = {
        "peaks": peaks_arr,
        "min": mins_arr,
        "max": maxs_arr,
        "rms": rms_arr,
        "duration_s": duration_s,
        "sample_rate": _PCM_SAMPLE_RATE,
        "source": str(stream_path.resolve()),
    }
    try:
        write_json(cache_file, doc)
    except Exception as exc:  # noqa: BLE001
        logger.warning("waveform cache write failed: %s", exc)
    return doc


def _range_sample_rate(span_s: float, buckets: int) -> int:
    """Подобрать sample rate так, чтобы в бакете было ≥ _RANGE_MIN_SPB семплов."""
    if span_s <= 0 or buckets <= 0:
        return _PCM_SAMPLE_RATE
    target = buckets * _RANGE_MIN_SPB / span_s
    return int(max(_PCM_SAMPLE_RATE, min(target, _RANGE_MAX_SR)))


def _range_buckets(span_s: float, buckets: int, sample_rate: int) -> int:
    """Урезать число бакетов до того, что реально обеспечено семплами.

    На максимальном зуме фронт просит до 8192 бакетов, а sample rate упирается в
    потолок 48 kHz: 0.54 с → 25920 семплов, т.е. ~3 семпла на бакет вместо
    обещанных 8. Раньше потолок выигрывал молча — контур на самом важном зуме
    вырождался, а в JSON улетало ~400 KB на полсекунды звука.

    Честнее отдать МЕНЬШЕ бакетов, но настоящих: фронт умеет рисовать любое их
    число (при <1 бакета на столбик интерполирует), а «бакет из 3 семплов» — это
    уже не огибающая, а шум формы.
    """
    if span_s <= 0 or sample_rate <= 0:
        return max(1, buckets)
    supported = int(span_s * sample_rate / _RANGE_MIN_SPB)
    return max(1, min(buckets, supported))


def _run_ffmpeg_pcm_range(
    stream_path: Path, start_s: float, span_s: float, sample_rate: int
) -> bytes:
    """Декодировать ТОЛЬКО слайс [start, start+span] → raw PCM s16le mono.

    `-ss` ДО `-i` — input seeking: ffmpeg прыгает по индексу контейнера, а не
    декодирует стрим с нуля. Для узкого окна это десятки ms вместо секунд
    (современный ffmpeg при этом остаётся сэмпл-точным — доводит от ближайшего
    keyframe). `-t` ограничивает длительность, поэтому объём вывода известен и
    мал: 2 с при 48 kHz = 192 KB.
    """
    ffmpeg = get_ffmpeg_path()
    cmd = [
        ffmpeg,
        "-nostdin",
        "-hide_banner",
        "-loglevel", "error",
        "-ss", f"{max(0.0, start_s):.6f}",
        "-t", f"{max(0.0, span_s):.6f}",
        "-i", str(stream_path),
        "-vn",
        "-ac", "1",
        "-ar", str(sample_rate),
        "-f", "s16le",
        "-acodec", "pcm_s16le",
        "pipe:1",
    ]
    completed = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if completed.returncode != 0:
        stderr = completed.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"ffmpeg waveform range extraction failed: {stderr}")
    return completed.stdout


def compute_waveform_range(
    stream_path: Path,
    start_s: float,
    end_s: float,
    buckets: int = 2048,
) -> dict:
    """Публичный API: peaks ТОЛЬКО для окна [start_s, end_s] («слой детали»).

    Без дискового кэша: комбинаций (start, end, buckets) бесконечно, а декод
    узкого слайса и так дешёвый. Фронт кэширует тайлы в памяти.

    Возвращает фактически покрытый диапазон в `start_s`/`end_s` — он может
    оказаться короче запрошенного (например запрос уехал за конец файла), и
    фронт обязан рисовать по ФАКТИЧЕСКИМ границам, иначе волна съедет.

    Число бакетов тоже может быть меньше запрошенного (см. `_range_buckets`) —
    фронт по этому признаку понимает, что глубже детализация не станет.
    """
    if not stream_path.exists():
        raise FileNotFoundError(f"waveform source not found: {stream_path}")
    if buckets <= 0:
        raise ValueError("buckets must be > 0")
    start = max(0.0, float(start_s))
    end = float(end_s)
    if end <= start:
        raise ValueError("end_s must be > start_s")
    span = end - start
    sample_rate = _range_sample_rate(span, buckets)
    buckets = _range_buckets(span, buckets, sample_rate)
    pcm = _run_ffmpeg_pcm_range(stream_path, start, span, sample_rate)
    mins_arr, maxs_arr, rms_arr, peaks_arr = _bucket_waveform(pcm, buckets)
    # Фактическая длительность слайса: у конца файла ffmpeg отдаёт меньше, чем
    # просили. Отдаём реальные границы, чтобы фронт не растягивал их на всё окно.
    actual_span = len(pcm) / _PCM_BYTES_PER_SAMPLE / sample_rate
    if actual_span <= 0:
        actual_span = span
    return {
        "peaks": peaks_arr,
        "min": mins_arr,
        "max": maxs_arr,
        "rms": rms_arr,
        "start_s": start,
        "end_s": start + actual_span,
        "sample_rate": sample_rate,
        "source": str(stream_path.resolve()),
    }
