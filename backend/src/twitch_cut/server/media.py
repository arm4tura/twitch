"""HTTP Range-стриминг медиа-файлов для HTML5 `<audio>`/`<video>`.

Задача:
- Отдавать байты произвольного файла (mp4/wav/mp3) через FastAPI так, чтобы
  браузер мог сикать (`audio.currentTime = 42`).
- Обязательная поддержка Range: без `Content-Range` + 206 браузер не сможет
  прыгать в середину файла, playback будет только с самого начала.

Безопасность:
- Файл должен быть в белом списке (`allowed_media_paths` в JobStore) — иначе
  любой сайт, открывшийся в дефолтном браузере пользователя, смог бы через
  `<audio src="http://127.0.0.1:PORT/media?path=/etc/passwd">` вычитать
  произвольный файл. Whitelist пополняется автоматически при создании job'а
  (stream/original попадают в allowed) и вручную через отдельный POST-хук
  (не в этом коммите — фронт добавляет пути через API-вызов при загрузке
  проекта из Dashboard).
- Никакого directory traversal: сравниваем `Path.resolve()` строго с элементами
  whitelist'а.
"""

from __future__ import annotations

import logging
import mimetypes
import os
import re
from pathlib import Path
from typing import Iterator, Optional

from fastapi import HTTPException
from fastapi.responses import StreamingResponse

logger = logging.getLogger(__name__)

# 256 KB на chunk — стандартный размер для audio streaming: короче — CPU-overhead,
# длиннее — заметная задержка при seek на медленных ФС.
_CHUNK_SIZE = 256 * 1024

_RANGE_RE = re.compile(r"bytes=(?P<start>\d*)-(?P<end>\d*)$")


def _stream_chunks(path: Path, start: int, end_inclusive: int) -> Iterator[bytes]:
    """Ленивый reader между байтовыми оффсетами. end_inclusive — как в RFC 7233."""
    remaining = end_inclusive - start + 1
    with path.open("rb") as f:
        f.seek(start)
        while remaining > 0:
            chunk = f.read(min(_CHUNK_SIZE, remaining))
            if not chunk:
                break
            remaining -= len(chunk)
            yield chunk


def _parse_range(range_header: str, size: int) -> Optional[tuple[int, int]]:
    """Распарсить `Range: bytes=start-end`. Возвращает (start, end_inclusive)
    или None если формат не «bytes=…».

    Спека:
      - `bytes=0-`   → весь файл начиная с 0
      - `bytes=-500` → последние 500 байт
      - `bytes=100-200` → диапазон
      - `bytes=100-999999` при файле < end → clamp до size-1
    """
    m = _RANGE_RE.match(range_header.strip())
    if not m:
        return None
    start_s = m.group("start")
    end_s = m.group("end")
    if not start_s and not end_s:
        return None
    if not start_s:
        # Suffix range: последние N байт.
        length = int(end_s)
        if length <= 0:
            return None
        start = max(0, size - length)
        end = size - 1
    else:
        start = int(start_s)
        end = int(end_s) if end_s else size - 1
    if start >= size:
        raise HTTPException(
            status_code=416,
            detail="range not satisfiable",
            headers={"Content-Range": f"bytes */{size}"},
        )
    end = min(end, size - 1)
    if end < start:
        raise HTTPException(status_code=416, detail="invalid range")
    return start, end


def _guess_media_type(path: Path) -> str:
    """Определить Content-Type по расширению; fallback — octet-stream.

    mimetypes НЕ знает про webm audio, но mp4/mp3/wav/flac закрывает — этого
    хватает для нашей аудио-задачи.
    """
    guessed, _ = mimetypes.guess_type(path.name)
    return guessed or "application/octet-stream"


def build_media_response(
    path: Path,
    range_header: Optional[str],
) -> StreamingResponse:
    """Собрать StreamingResponse с корректным Range/Content-Range.

    Ошибки:
      - файла нет → 404
      - Range не парсится → игнорируем и отдаём 200 (RFC разрешает).
      - Range за пределами файла → 416 с `Content-Range: bytes */size`.
    """
    if not path.exists() or not path.is_file():
        raise HTTPException(status_code=404, detail=f"media not found: {path}")
    size = path.stat().st_size
    media_type = _guess_media_type(path)

    if range_header:
        parsed = _parse_range(range_header, size)
        if parsed is not None:
            start, end = parsed
            headers = {
                "Content-Range": f"bytes {start}-{end}/{size}",
                "Accept-Ranges": "bytes",
                "Content-Length": str(end - start + 1),
                # Кэш ~1 час — файлы стрима не меняются, но при пересборке
                # decisions.json путь тот же. Кэш дольше — риск stale.
                "Cache-Control": "private, max-age=3600",
            }
            return StreamingResponse(
                _stream_chunks(path, start, end),
                status_code=206,
                media_type=media_type,
                headers=headers,
            )

    # Полный файл: 200 + Accept-Ranges (браузер сможет ре-запросить с Range).
    headers = {
        "Accept-Ranges": "bytes",
        "Content-Length": str(size),
        "Cache-Control": "private, max-age=3600",
    }
    return StreamingResponse(
        _stream_chunks(path, 0, size - 1),
        status_code=200,
        media_type=media_type,
        headers=headers,
    )


def resolve_and_authorize(path_str: str, whitelist: set[Path]) -> Path:
    """Проверить путь: должен существовать И быть в whitelist'e.

    Возвращает Path.resolve() (готовый к open). Кидает 403/404 если запрещено.
    """
    p = Path(path_str).expanduser().resolve()
    if not p.exists():
        raise HTTPException(status_code=404, detail=f"file not found: {p}")
    # Резолвим whitelist тоже — иначе `~/…` в реестре не смэтчит абсолютный p.
    allowed_resolved = {Path(x).expanduser().resolve() for x in whitelist}
    if p not in allowed_resolved:
        raise HTTPException(status_code=403, detail=f"path not in media whitelist: {p}")
    return p
