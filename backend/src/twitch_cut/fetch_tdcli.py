"""Автоматическая загрузка бинарника TwitchDownloaderCLI.

Бинарник TwitchDownloaderCLI (github.com/lay295/TwitchDownloader) большой и
лицензируется отдельно, поэтому в репозиторий он НЕ коммитится (в
`desktop/build/tdcli/<platform>/` лежат только `.gitkeep`). Раньше это значило,
что на свежем клоне VOD-downloader молча не работал: `/vod/info` отдавал 502,
UI показывал «превью недоступно», и пользователь должен был вручную качать
бинарник и раскладывать его по папкам.

Этот модуль убирает ручной шаг: если бинарник не найден ни в одном из известных
мест, `tdcli.get_tdcli_path()` зовёт `ensure_tdcli()` — тот скачивает нужный
per-platform asset из GitHub Releases, распаковывает и кладёт в
`DATA_ROOT/vendor/tdcli/<platform>/`. Дальнейшие запуски находят его сразу.

Для СБОРКИ (electron-builder) отдельный путь — `desktop/scripts/prepare-tdcli.js`
раскладывает бинарники по `desktop/build/tdcli/*` перед упаковкой. Этот модуль —
про dev и про first-run уже установленного приложения.

Разрешение имени asset'а: НЕ хардкодим точное имя файла (оно меняется между
версиями TwitchDownloader). Спрашиваем у GitHub API список asset'ов последнего
релиза и матчим по ключевым словам платформы. Если сеть/квота API недоступны —
падаём на закреплённую версию с известным шаблоном имени (PINNED_*).
"""

from __future__ import annotations

import io
import json
import logging
import os
import platform
import ssl
import stat
import sys
import tempfile
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_REPO = "lay295/TwitchDownloader"
_LATEST_API = f"https://api.github.com/repos/{_REPO}/releases/latest"

# Закреплённая версия — фолбэк, если GitHub API недоступен (rate-limit без
# токена — 60 запросов/час на IP; за корпоративным NAT легко упереться). Имена
# asset'ов TwitchDownloader стабильны в пределах схемы «...-<OS>-<arch>.zip».
# Обновлять при апгрейде проверенной линейки (см. CLI CONTRACT в tdcli.py).
_PINNED_TAG = "1.55.5"
_PINNED_ASSETS = {
    "win-x64": f"TwitchDownloaderCLI-{_PINNED_TAG}-Windows-x64.zip",
    "linux-x64": f"TwitchDownloaderCLI-{_PINNED_TAG}-Linux-x64.zip",
    "osx-x64": f"TwitchDownloaderCLI-{_PINNED_TAG}-MacOS-x64.zip",
    "osx-arm64": f"TwitchDownloaderCLI-{_PINNED_TAG}-MacOS-Arm64.zip",
}
_PINNED_URL = f"https://github.com/{_REPO}/releases/download/{_PINNED_TAG}/{{asset}}"

# Ключевые слова для матчинга asset'а по платформе (регистронезависимо). Порядок
# важен: osx-arm64 должен требовать И mac, И arm — иначе поймает x64-сборку.
_PLATFORM_KEYWORDS: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    # platform_dir: (must_contain_any_of_os, must_contain_any_of_arch)
    "win-x64": (("windows", "win"), ("x64", "amd64", "x86_64")),
    "linux-x64": (("linux",), ("x64", "amd64", "x86_64")),
    "osx-x64": (("macos", "osx", "mac"), ("x64", "amd64", "x86_64")),
    "osx-arm64": (("macos", "osx", "mac"), ("arm64", "arm", "aarch64")),
}


def _platform_binary_name() -> str:
    return "TwitchDownloaderCLI.exe" if os.name == "nt" else "TwitchDownloaderCLI"


def _platform_dir() -> str:
    """Подкаталог per-platform (совпадает с tdcli._platform_dir)."""
    machine = platform.machine().lower()
    is_arm = machine in {"arm64", "aarch64"}
    if sys.platform.startswith("win"):
        return "win-x64"
    if sys.platform == "darwin":
        return "osx-arm64" if is_arm else "osx-x64"
    return "linux-x64"


def _vendor_dir() -> Path:
    """Куда кладём скачанный бинарник (writable, тот же корень, что и модели)."""
    from .config import DATA_ROOT

    return DATA_ROOT / "vendor" / "tdcli" / _platform_dir()


def _ssl_context() -> ssl.SSLContext:
    """SSL-контекст с надёжным набором корневых сертификатов.

    На macOS Python с python.org (и собранный из него бэкенд) по умолчанию НЕ
    использует связку сертификатов Keychain — `urllib` падает с
    `CERTIFICATE_VERIFY_FAILED`, пока пользователь вручную не запустит
    «Install Certificates.command». Чтобы не заставлять его это делать, берём
    CA-bundle из `certifi` — он в прямых зависимостях проекта и едет внутри
    собранного бэкенда. Им же качает модели `onnx_models`. Если certifi
    почему-то нет — откатываемся на дефолтный контекст (лучше попробовать, чем
    гарантированно упасть)."""
    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except Exception:  # noqa: BLE001 — certifi отсутствует/битый → дефолт
        return ssl.create_default_context()


def _http_get(url: str, *, timeout: float, accept: Optional[str] = None) -> bytes:
    headers = {"User-Agent": "twitch-cut/tdcli-fetch"}
    if accept:
        headers["Accept"] = accept
    # Позволяем прокинуть токен, если пользователь упёрся в rate-limit.
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, headers=headers)
    ctx = _ssl_context()
    with urllib.request.urlopen(req, timeout=timeout, context=ctx) as resp:  # noqa: S310 (github only)
        return resp.read()


def _resolve_asset_url(plat: str, *, timeout: float) -> str:
    """Найти URL asset'а для платформы через GitHub API; при сбое — PINNED.

    Матчим по ключевым словам, а не по точному имени — устойчиво к смене версии
    и мелким переименованиям («MacOS» vs «macOS», «Arm64» vs «arm64»).
    """
    os_kws, arch_kws = _PLATFORM_KEYWORDS[plat]
    try:
        raw = _http_get(
            _LATEST_API, timeout=timeout, accept="application/vnd.github+json"
        )
        data = json.loads(raw)
        assets = data.get("assets", []) if isinstance(data, dict) else []
        for asset in assets:
            name = str(asset.get("name", ""))
            low = name.lower()
            if not low.endswith(".zip"):
                continue
            if any(k in low for k in os_kws) and any(k in low for k in arch_kws):
                url = asset.get("browser_download_url")
                if url:
                    logger.info("tdcli asset resolved via API: %s", name)
                    return str(url)
        logger.warning(
            "tdcli: подходящий asset для %s не найден в latest-релизе, "
            "падаю на закреплённую версию %s",
            plat,
            _PINNED_TAG,
        )
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError) as exc:
        logger.warning(
            "tdcli: GitHub API недоступен (%s), падаю на закреплённую версию %s",
            exc,
            _PINNED_TAG,
        )

    return _PINNED_URL.format(asset=_PINNED_ASSETS[plat])


def _extract_binary(zip_bytes: bytes, dest: Path) -> None:
    """Достать из zip именно исполняемый файл CLI и записать в dest (атомарно).

    В архиве бинарник может лежать в корне или во вложенной папке; берём первый
    файл, чьё basename == ожидаемому имени (или, для unix, любой файл без
    расширения, помеченный исполняемым в zip-атрибутах, — как последний шанс).
    """
    want = _platform_binary_name()
    dest.parent.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        member = None
        for info in zf.infolist():
            if info.is_dir():
                continue
            base = Path(info.filename).name
            if base == want:
                member = info
                break
        # unix-фолбэк: единственный исполняемый файл в архиве.
        if member is None and os.name != "nt":
            execs = [
                i
                for i in zf.infolist()
                if not i.is_dir() and (i.external_attr >> 16) & 0o111
            ]
            if len(execs) == 1:
                member = execs[0]

        if member is None:
            raise RuntimeError(
                f"в архиве TwitchDownloaderCLI не найден бинарник '{want}'"
            )

        payload = zf.read(member)

    # Пишем во временный файл рядом и атомарно переименовываем — безопасно при
    # гонке двух параллельных first-run запросов /vod/info.
    fd, tmp_name = tempfile.mkstemp(dir=str(dest.parent), prefix=".tdcli-", suffix=".part")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(payload)
        if os.name != "nt":
            st = os.stat(tmp_name)
            os.chmod(tmp_name, st.st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
        os.replace(tmp_name, dest)
    finally:
        if os.path.exists(tmp_name):
            try:
                os.remove(tmp_name)
            except OSError:
                pass


def ensure_tdcli(*, timeout: float = 120.0) -> Optional[str]:
    """Гарантировать наличие бинарника; вернуть путь или None при неудаче.

    Идемпотентно: если файл уже скачан — сразу возвращает путь, ничего не
    качает. Любая ошибка сети/распаковки логируется и возвращает None (вызывающий
    код в tdcli.get_tdcli_path сам решит, что делать — обычно кинет TdcliError с
    человекочитаемым текстом).
    """
    plat = _platform_dir()
    dest = _vendor_dir() / _platform_binary_name()
    if dest.exists():
        return str(dest)

    logger.info(
        "TwitchDownloaderCLI не найден — скачиваю для %s в %s …", plat, dest.parent
    )
    try:
        url = _resolve_asset_url(plat, timeout=min(timeout, 30.0))
        logger.info("tdcli: загружаю %s", url)
        zip_bytes = _http_get(url, timeout=timeout)
        _extract_binary(zip_bytes, dest)
    except (urllib.error.URLError, TimeoutError, OSError, RuntimeError, zipfile.BadZipFile) as exc:
        logger.error("tdcli: автозагрузка не удалась: %s", exc)
        return None

    if dest.exists():
        logger.info("tdcli: готов — %s", dest)
        return str(dest)
    return None
