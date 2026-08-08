"""Обёртка над TwitchDownloaderCLI (github.com/lay295/TwitchDownloader).

Скачивание Twitch VOD как subprocess — по тому же образцу, что `ffmpeg_tools.py`
зовёт ffmpeg. Весь ТОЧНЫЙ синтаксис CLI (имена флагов, режимы, формат прогресса)
СОЗНАТЕЛЬНО собран в одном месте — см. блок «=== CLI CONTRACT (VERIFY) ===».
Если версия TwitchDownloaderCLI поменяет флаг — правка здесь, в одной функции,
а не по всему коду.

ffmpeg НЕ дублируем: TwitchDownloaderCLI требует ffmpeg для мукса .ts→.mp4,
но у нас уже есть бинарник через imageio_ffmpeg — передаём его путь флагом
--ffmpeg-path (см. get_ffmpeg_path в ffmpeg_tools).
"""

from __future__ import annotations

import json
import logging
import os
import platform
import re
import subprocess
import sys
from pathlib import Path
from typing import Callable, Optional

from .config import DATA_ROOT
from .ffmpeg_tools import get_ffmpeg_path

logger = logging.getLogger(__name__)


class TdcliError(RuntimeError):
    """Ошибка запуска/выполнения TwitchDownloaderCLI с человекочитаемым текстом."""


# ============================================================================
# Расположение бинарника
# ============================================================================

def _platform_binary_name() -> str:
    return "TwitchDownloaderCLI.exe" if os.name == "nt" else "TwitchDownloaderCLI"


def _platform_dir() -> str:
    """Подкаталог per-platform, куда electron-builder кладёт нужный бинарник.

    Совпадает с ассетами релизов TwitchDownloader: win-x64 / linux-x64 /
    osx-x64 / osx-arm64.
    """
    machine = platform.machine().lower()
    is_arm = machine in {"arm64", "aarch64"}
    if sys.platform.startswith("win"):
        return "win-x64"
    if sys.platform == "darwin":
        return "osx-arm64" if is_arm else "osx-x64"
    return "linux-x64"


def get_tdcli_path(*, auto_download: bool = True) -> str:
    """Найти исполняемый TwitchDownloaderCLI.

    Порядок:
      1. Явный override TWITCH_CUT_TDCLI (полный путь к бинарнику) — для dev/тестов.
      2. Каталог из TWITCH_CUT_TDCLI_DIR (его выставляет electron, указывая на
         распакованные extraResources), + per-platform подкаталог.
      3. <DATA_ROOT>/vendor/tdcli/<platform>/<binary> — dev/first-run расположение.
      4. Голое имя в PATH (системная установка).
      5. Автозагрузка с GitHub Releases в <DATA_ROOT>/vendor/tdcli/<platform>/
         (если auto_download=True) — чтобы на свежем клоне и при первом запуске
         VOD-downloader работал без ручной раскладки бинарника.

    Кидает TdcliError, если ничего не найдено и автозагрузка не удалась.

    auto_download=False отключает шаг 5 — для мест, где скачивание в фоне
    нежелательно (тесты, проверка «установлен ли downloader» без побочек).
    """
    explicit = os.environ.get("TWITCH_CUT_TDCLI")
    if explicit and Path(explicit).exists():
        return explicit

    binary = _platform_binary_name()
    plat = _platform_dir()

    candidates: list[Path] = []
    res_dir = os.environ.get("TWITCH_CUT_TDCLI_DIR")
    if res_dir:
        candidates.append(Path(res_dir) / plat / binary)
        candidates.append(Path(res_dir) / binary)
    candidates.append(DATA_ROOT / "vendor" / "tdcli" / plat / binary)
    candidates.append(DATA_ROOT / "vendor" / "tdcli" / binary)

    for c in candidates:
        if c.exists():
            # На unix убеждаемся, что бит исполняемости стоит (после распаковки
            # из архива он может слететь).
            if os.name != "nt":
                try:
                    mode = c.stat().st_mode
                    if not (mode & 0o111):
                        c.chmod(mode | 0o111)
                except OSError:
                    pass
            return str(c)

    # PATH как предпоследний шанс (системная установка).
    from shutil import which
    found = which(binary)
    if found:
        return found

    # Автозагрузка — последний шанс. Тянет нужный per-platform asset и кладёт в
    # DATA_ROOT/vendor/tdcli/<platform>/ (см. fetch_tdcli.ensure_tdcli).
    if auto_download:
        from .fetch_tdcli import ensure_tdcli

        downloaded = ensure_tdcli()
        if downloaded:
            return downloaded

    raise TdcliError(
        "TwitchDownloaderCLI не найден, и автоматически скачать его не удалось "
        "(нет сети или GitHub недоступен). Ожидается в "
        f"vendor/tdcli/{plat}/{binary} или в PATH. Проверьте соединение и "
        "повторите — при первом обращении бинарник скачается сам."
    )


# ============================================================================
# === CLI CONTRACT (VERIFY) ==================================================
# Точный синтаксис TwitchDownloaderCLI. ЕДИНСТВЕННОЕ место, где зашиты имена
# флагов и режимов. Свериться с github.com/lay295/TwitchDownloader при смене
# версии. Проверено под линейку TwitchDownloaderCLI 1.55–1.56.x.
# ============================================================================

# Режим скачивания VOD. В старых версиях был другой алиас — сверить.
_MODE_VIDEO = "videodownload"
# Режим метаданных (список качеств/длительность).
_MODE_INFO = "info"
# Формат вывода info. ВНИМАНИЕ: JSON НЕ поддерживается (в 1.56.x
# `--format Json` кидает NotImplementedException «JSON format is not yet
# supported»). Валидные значения: Raw, Table, M3U8. Мы берём Raw — это Twitch
# GraphQL-ответ (title/lengthSeconds/owner/thumbnails), плюс отдельно M3U8 для
# списка качеств.
_INFO_FORMAT_META = "Raw"
_INFO_FORMAT_QUALITIES = "M3U8"


def _build_download_args(
    *,
    video_id: str,
    output: Path,
    quality: Optional[str],
    beginning: Optional[str],
    ending: Optional[str],
    oauth: Optional[str],
    temp_path: Optional[Path],
    threads: int,
    ffmpeg_path: str,
) -> list[str]:
    """Собрать argv для скачивания VOD.

    Флаги (VERIFY):
      videodownload
      --id <id>              ID видео (или URL — CLI принимает оба)
      -o <file>              выходной файл (расширение задаёт контейнер, .mp4)
      -q <quality>           напр. "1080p60", "720p", "Source"; пусто → лучшее
      -b <time>              crop с начала; формат HH:MM:SS или секунды
      -e <time>              crop до; тот же формат
      -t <n>                 число потоков загрузки
      --oauth <token>        для sub-only VOD
      --temp-path <dir>      временные .ts-сегменты (для resume)
      --ffmpeg-path <file>   свой ffmpeg (у нас imageio_ffmpeg)
      --collision <mode>     поведение при существующем файле: Overwrite
    """
    args = [
        get_tdcli_path(),
        _MODE_VIDEO,
        "--id", video_id,
        "-o", str(output),
        "-t", str(threads),
        "--ffmpeg-path", ffmpeg_path,
        "--collision", "Overwrite",
    ]
    if quality:
        args += ["-q", quality]
    if beginning:
        args += ["-b", beginning]
    if ending:
        args += ["-e", ending]
    if oauth:
        args += ["--oauth", oauth]
    if temp_path:
        args += ["--temp-path", str(temp_path)]
    return args


def _build_info_args(video_id: str, fmt: str) -> list[str]:
    """argv для получения метаданных VOD.

    Флаги (VERIFY):
      info --id <id> --format <Raw|M3U8|Table> --banner false

    JSON-формат CLI НЕ поддерживает (см. _INFO_FORMAT_META). Raw даёт JSON-ответ
    Twitch GraphQL первой строкой (после [STATUS]-строки), M3U8 — плейлист с
    качествами. `--banner false` убирает копирайт-баннер из stdout.
    """
    return [
        get_tdcli_path(),
        _MODE_INFO,
        "--id",
        video_id,
        "--format",
        fmt,
        "--banner",
        "false",
    ]


# Прогресс TwitchDownloaderCLI печатает статус-строками вида
#   [STATUS] - Downloading 45%
#   [STATUS] - Combining Parts 100%
# обновляя их через '\r'. Тянем ЛЮБОЙ процент из строки — устойчиво к смене
# формулировок. Стадию берём по ключевым словам.
_PCT_RE = re.compile(r"(\d{1,3})\s*%")
_STAGE_KEYWORDS = [
    ("download", ("downloading",)),
    ("verify", ("verifying",)),
    ("combine", ("combining", "finalizing")),
    ("info", ("fetching", "video info")),
]


def parse_progress_line(line: str) -> Optional[tuple[str, Optional[float]]]:
    """Разобрать строку вывода CLI → (stage, percent|None). None если не прогресс."""
    low = line.lower()
    stage = None
    for name, kws in _STAGE_KEYWORDS:
        if any(k in low for k in kws):
            stage = name
            break
    m = _PCT_RE.search(line)
    pct = None
    if m:
        try:
            pct = max(0.0, min(100.0, float(m.group(1))))
        except ValueError:
            pct = None
    if stage is None and pct is None:
        return None
    return (stage or "download", pct)


# ============================================================================
# Классификация ошибок → человекочитаемый текст (для UI)
# ============================================================================

def classify_error(stderr: str, returncode: int) -> str:
    """Свести stderr TwitchDownloaderCLI к понятному пользователю сообщению.

    Строки-триггеры — эвристики (VERIFY по реальным логам). Неизвестное —
    отдаём как есть, обрезав.
    """
    low = stderr.lower()
    if any(k in low for k in ("subscriber", "sub-only", "unauthorized", "401", "must be", "oauth")):
        return (
            "VOD доступен только подписчикам. Нужен OAuth-токен авторизованного "
            "аккаунта с подпиской."
        )
    if any(k in low for k in ("not found", "404", "does not exist", "unavailable")):
        return "VOD недоступен — возможно, удалён или истёк срок хранения."
    if any(k in low for k in ("forbidden", "403")):
        return "Twitch отклонил запрос (403). Попробуйте позже или укажите OAuth-токен."
    if any(k in low for k in ("quality", "resolution")) and "not" in low:
        return "Запрошенное качество недоступно для этого VOD."
    if any(k in low for k in ("network", "timed out", "timeout", "connection")):
        return "Сбой сети при скачивании. Проверьте соединение и повторите."
    if "space" in low or "enospc" in low:
        return "Недостаточно места на диске."
    snippet = stderr.strip().splitlines()[-1] if stderr.strip() else ""
    return f"Скачивание не удалось (код {returncode}). {snippet}".strip()


# ============================================================================
# Публичные операции
# ============================================================================

def fetch_vod_info(video_id: str, *, timeout: float = 30.0) -> dict:
    """Вернуть метаданные VOD: {title, channel, duration_ms, thumbnail_url, qualities}.

    Делает ДВА вызова CLI:
      1. `info --format Raw`  → JSON Twitch GraphQL (title/lengthSeconds/owner/
         thumbnails). Первой строкой идёт [STATUS]-строка — её пропускаем.
      2. `info --format M3U8` → плейлист с доступными качествами.

    JSON-формат CLI не поддерживает (кидает NotImplementedException), поэтому
    Raw+M3U8 — единственный рабочий путь. Парсинг защищённый: если качества не
    достали — отдаём дефолтный список; фронт умеет деградировать.
    """
    meta_raw = _run_info(video_id, _INFO_FORMAT_META, timeout=timeout)
    data = _parse_raw_json(meta_raw)

    # Качества — отдельным вызовом M3U8. Не критично: если упадёт, показываем
    # дефолт (Source), скачивание всё равно возможно.
    qualities: list[dict] = []
    try:
        m3u8 = _run_info(video_id, _INFO_FORMAT_QUALITIES, timeout=timeout)
        qualities = _parse_m3u8_qualities(m3u8)
    except TdcliError:
        logger.warning("tdcli: не удалось получить список качеств VOD %s", video_id)

    return _normalize_info(data, video_id, qualities)


def _run_info(video_id: str, fmt: str, *, timeout: float) -> str:
    """Запустить `info --format <fmt>` и вернуть stdout. Кидает TdcliError."""
    args = _build_info_args(video_id, fmt)
    try:
        proc = subprocess.run(
            args, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        raise TdcliError("Таймаут получения метаданных VOD.") from exc
    except FileNotFoundError as exc:
        raise TdcliError(str(exc)) from exc

    if proc.returncode != 0:
        # stderr иногда пуст, а текст ошибки уезжает в stdout — склеиваем.
        raise TdcliError(classify_error(proc.stderr or proc.stdout, proc.returncode))
    return proc.stdout


def _parse_raw_json(stdout: str) -> dict:
    """Достать JSON-объект метаданных из вывода `info --format Raw`.

    Вывод: одна или несколько [STATUS]-строк, затем JSON-объект(ы). Нас
    интересует первый объект, где есть data.video с title/lengthSeconds. Берём
    первую строку, начинающуюся с '{' и парсящуюся как JSON с таким узлом.
    """
    for line in stdout.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        video = (
            obj.get("data", {}).get("video")
            if isinstance(obj.get("data"), dict)
            else None
        )
        if isinstance(video, dict) and ("lengthSeconds" in video or "title" in video):
            return video
        # Иногда полезный объект — сам верхний уровень (старые версии).
        if "lengthSeconds" in obj or "title" in obj:
            return obj
    raise TdcliError("Не удалось разобрать метаданные VOD (нет JSON в ответе).")


# Строки M3U8: пара из EXT-X-MEDIA (имя/группа) и следующего EXT-X-STREAM-INF
# (разрешение/fps). Значение для `-q` у TwitchDownloaderCLI — человекочитаемое
# NAME ("1080p60", "720p60", "Audio Only"); Source == группа "chunked".
_M3U8_MEDIA_RE = re.compile(r'#EXT-X-MEDIA:[^\n]*GROUP-ID="([^"]+)"[^\n]*NAME="([^"]+)"')


def _parse_m3u8_qualities(m3u8: str) -> list[dict]:
    """Разобрать качества из плейлиста → [{value,label}], лучшее первым.

    value — то, что передаётся в `-q` (имя качества). label — человекочитаемое,
    с пометкой Source для группы chunked.
    """
    out: list[dict] = []
    for m in _M3U8_MEDIA_RE.finditer(m3u8):
        group, name = m.group(1), m.group(2)
        is_source = group.lower() == "chunked"
        label = f"{name} (Source)" if is_source else name
        out.append({"value": name, "label": label, "_source": is_source})
    # chunked (Source) — вперёд; остальное в порядке появления (Twitch отдаёт по
    # убыванию качества).
    out.sort(key=lambda q: 0 if q.get("_source") else 1)
    for q in out:
        q.pop("_source", None)
    return out


def _first(d: dict, *keys, default=None):
    """Вернуть первое присутствующее значение из candidate-ключей (регистронезав.)."""
    lower = {k.lower(): v for k, v in d.items()} if isinstance(d, dict) else {}
    for k in keys:
        if k.lower() in lower and lower[k.lower()] is not None:
            return lower[k.lower()]
    return default


def _normalize_info(node: dict, video_id: str, qualities: list[dict]) -> dict:
    """Привести JSON info (узел data.video) к нашему контракту (VodInfo)."""
    title = _first(node, "title", "name", default="") or ""

    channel = ""
    ch = _first(node, "owner", "channel", "streamer", "user")
    if isinstance(ch, dict):
        channel = _first(ch, "displayName", "name", "login", default="") or ""
    elif isinstance(ch, str):
        channel = ch

    # Длительность.
    dur = _first(node, "lengthSeconds", "length", "duration", "durationSeconds", default=0)
    duration_ms = 0
    if isinstance(dur, (int, float)):
        duration_ms = int(dur * 1000)
    elif isinstance(dur, str):
        if dur.isdigit():
            duration_ms = int(dur) * 1000
        else:
            parts = dur.split(":")
            try:
                secs = 0
                for p in parts:
                    secs = secs * 60 + int(p)
                duration_ms = secs * 1000
            except ValueError:
                duration_ms = 0

    # Превью: thumbnailURLs — массив (берём первый), либо одиночные ключи.
    thumb = None
    thumbs = _first(node, "thumbnailURLs", "thumbnailUrls")
    if isinstance(thumbs, list) and thumbs:
        thumb = thumbs[0]
    if not thumb:
        t = _first(node, "thumbnailUrl", "thumbnail", "previewThumbnailUrl")
        if isinstance(t, dict):
            t = _first(t, "url", "medium", "large")
        thumb = t

    if not qualities:
        qualities = [{"value": "chunked", "label": "Source (максимальное)"}]

    return {
        "id": str(video_id),
        "title": title,
        "channel": channel,
        "duration_ms": duration_ms,
        "thumbnail_url": thumb if isinstance(thumb, str) else None,
        "qualities": qualities,
    }


def download_vod(
    *,
    video_id: str,
    output: Path,
    quality: Optional[str] = None,
    beginning: Optional[str] = None,
    ending: Optional[str] = None,
    oauth: Optional[str] = None,
    temp_path: Optional[Path] = None,
    threads: int = 4,
    on_progress: Optional[Callable[[str, Optional[float]], None]] = None,
) -> Path:
    """Скачать VOD, стримя прогресс в on_progress(stage, percent).

    Блокирующая (запускать в asyncio.to_thread из runner'а). Кидает TdcliError
    с человекочитаемым текстом при сбое. Возвращает путь к готовому файлу.
    """
    output.parent.mkdir(parents=True, exist_ok=True)
    if temp_path:
        temp_path.mkdir(parents=True, exist_ok=True)

    args = _build_download_args(
        video_id=video_id,
        output=output,
        quality=quality,
        beginning=beginning,
        ending=ending,
        oauth=oauth,
        temp_path=temp_path,
        threads=threads,
        ffmpeg_path=get_ffmpeg_path(),
    )

    logger.info("tdcli download: %s", " ".join(a for a in args if a != oauth))  # не логируем токен

    try:
        proc = subprocess.Popen(
            args,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # прогресс TDCLI идёт вперемешку — сливаем
            text=True,
            bufsize=1,
        )
    except FileNotFoundError as exc:
        raise TdcliError(str(exc)) from exc

    tail: list[str] = []  # последние строки — на случай ошибки
    assert proc.stdout is not None
    # CLI обновляет строку через '\r'; читаем посимвольно-безопасно построчно,
    # разбивая и по '\r', и по '\n'.
    buf = ""
    while True:
        chunk = proc.stdout.read(1)
        if chunk == "" and proc.poll() is not None:
            break
        if chunk in ("\r", "\n"):
            line = buf.strip()
            buf = ""
            if not line:
                continue
            tail.append(line)
            if len(tail) > 20:
                tail.pop(0)
            if on_progress:
                parsed = parse_progress_line(line)
                if parsed:
                    on_progress(parsed[0], parsed[1])
        else:
            buf += chunk

    rc = proc.wait()
    if rc != 0:
        raise TdcliError(classify_error("\n".join(tail), rc))

    if not output.exists():
        raise TdcliError("Скачивание завершилось, но файл не создан.")
    return output
