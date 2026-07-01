from __future__ import annotations

import logging
import re
import subprocess
from pathlib import Path

import imageio_ffmpeg

from .cache import file_fingerprint, stable_hash, write_json
from .timecode import TimeSpan, parse_timecode

logger = logging.getLogger(__name__)
_DURATION_RE = re.compile(r"Duration:\s*(?P<duration>\d{1,2}:\d{2}:\d{2}(?:[\.,]\d{1,3})?)")


def get_ffmpeg_path() -> str:
    return imageio_ffmpeg.get_ffmpeg_exe()


def probe_media_duration(media_path: Path) -> TimeSpan:
    """Return media duration using ffmpeg metadata output."""

    ffmpeg = get_ffmpeg_path()
    cmd = [ffmpeg, "-hide_banner", "-i", str(media_path)]
    completed = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    output = f"{completed.stdout}\n{completed.stderr}"
    match = _DURATION_RE.search(output)
    if not match:
        raise RuntimeError(f"Could not detect media duration for {media_path}")
    return parse_timecode(match.group("duration"))


def extract_audio_range(
    stream_path: Path,
    range_in: TimeSpan,
    range_out: TimeSpan,
    workdir: Path,
    force: bool = False,
) -> tuple[Path, str]:
    if range_out <= range_in:
        raise ValueError("range_out must be greater than range_in")

    extract_dir = workdir / "extracted"
    cache_dir = workdir / "cache"
    extract_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    key_data = {
        "stage": "extract_audio",
        "stream": file_fingerprint(stream_path),
        "range_in_ms": range_in.ms,
        "range_out_ms": range_out.ms,
        "format": "wav_pcm_s16le_16khz_mono",
    }
    key = stable_hash(key_data)
    wav_path = extract_dir / f"stream_range_{key}.wav"
    meta_path = cache_dir / f"extract_{key}.json"

    if wav_path.exists() and meta_path.exists() and not force:
        logger.info("Аудио уже извлечено, используем cache: %s", wav_path)
        return wav_path, key

    ffmpeg = get_ffmpeg_path()
    logger.info("Извлекаю аудио через ffmpeg: %s", wav_path)
    duration = range_out - range_in
    cmd = [
        ffmpeg,
        "-y",
        "-ss",
        f"{range_in.to_seconds():.3f}",
        "-i",
        str(stream_path),
        "-t",
        f"{duration.to_seconds():.3f}",
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-acodec",
        "pcm_s16le",
        str(wav_path),
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.strip() if exc.stderr else ""
        raise RuntimeError(f"ffmpeg extraction failed: {stderr}") from exc

    write_json(
        meta_path,
        {
            "stage": "extract_audio",
            "key": key,
            "inputs": key_data,
            "output": str(wav_path),
        },
    )
    return wav_path, key
