"""On-demand thumbnail generation from stream video.

Generates a JPEG snapshot at ~10% of the video's duration (past intros) and
caches by file fingerprint. Designed for the Dashboard project card — the
`<img>` loads lazily and falls back to the Film placeholder on 404.

Security: only the source video of a registered project, or files inside a
project directory / DATA_ROOT, are authorized.
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

from fastapi import HTTPException

from ..cache import file_fingerprint, stable_hash

# ffmpeg is bundled via imageio-ffmpeg.
from ..ffmpeg_tools import get_ffmpeg_path, probe_media_duration

logger = logging.getLogger(__name__)


def _thumb_cache_dir() -> Path:
    from ..config import DATA_ROOT

    d = DATA_ROOT / "cache" / "thumbs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _pick_seek(stream: Path) -> float:
    """Pick a representative timestamp (seconds) for the thumbnail frame.

    Probes the video's own duration and seeks 10% in — past intros, before the
    end. Probing is reliable regardless of where decisions.json lives (the
    source video usually sits outside work/). On any probe failure falls back to
    a small fixed offset that is safe even for short clips.
    """
    try:
        duration_s = probe_media_duration(stream).to_seconds()
        if duration_s > 0:
            # 10% in, but never deeper than 60s (long VODs shouldn't seek minutes).
            return min(duration_s * 0.1, 60.0)
    except Exception:
        pass
    return 3.0  # short safe offset — past a black first frame


def _extract_frame(stream: Path, seek_s: float, width: int, out: Path) -> None:
    """Extract a single JPEG frame via ffmpeg.  out must not already exist."""
    out.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg = get_ffmpeg_path()
    # -ss before -i for fast seek (no decode of preceding frames).
    cmd = [
        ffmpeg,
        "-y",
        "-ss",
        f"{seek_s:.3f}",
        "-i",
        str(stream),
        "-frames:v",
        "1",
        "-vf",
        f"scale={width}:-2",
        "-q:v",
        "4",
        "-loglevel",
        "error",
        str(out),
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.strip() if exc.stderr else ""
        raise RuntimeError(f"ffmpeg frame extraction failed: {stderr}") from exc


def generate_thumbnail(stream_path: str, width: int = 640) -> Path:
    """Generate (or retrieve cached) thumbnail for *stream_path*.  Returns the
    JPEG path on disk, or raises HTTPException on failure."""
    from ..config import DATA_ROOT
    from .projects import registered_roots, registered_stream_paths

    stream = Path(stream_path).expanduser().resolve()

    # --- authorization --------------------------------------------------------
    # Легальны: (1) файлы внутри DATA_ROOT / папки любого проекта, ИЛИ (2) сам
    # исходник видео, привязанный к одному из «недавних» проектов — он обычно
    # лежит вне work/ (там, куда пользователь его закинул), поэтому проверки по
    # каталогам проектов недостаточно.
    allowed_roots = {DATA_ROOT.resolve()} | registered_roots()
    in_root = any(root in stream.parents or root == stream for root in allowed_roots)
    is_source = stream in registered_stream_paths()
    if not (in_root or is_source):
        raise HTTPException(status_code=403, detail=f"stream not authorized: {stream}")

    if not stream.exists() or not stream.is_file():
        raise HTTPException(status_code=404, detail=f"stream not found: {stream}")

    # --- cache key ------------------------------------------------------------
    fp = file_fingerprint(stream)
    key_data = {"stage": "thumbnail", "stream": fp, "width": width}
    key = stable_hash(key_data)
    cached = _thumb_cache_dir() / f"{key}.jpg"

    if cached.exists():
        logger.debug("thumbnail cache hit: %s", cached)
        return cached

    # --- generate -------------------------------------------------------------
    seek = _pick_seek(stream)
    logger.info("generating thumbnail for %s at %.1fs, %dpx", stream, seek, width)
    try:
        _extract_frame(stream, seek, width, cached)
    except RuntimeError as exc:
        logger.error("thumbnail generation failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"thumbnail generation failed: {exc}")

    return cached
