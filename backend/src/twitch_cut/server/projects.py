"""Реестр «недавних проектов» для Dashboard-экрана.

Модель: один JSON-файл со списком путей к `decisions.json`. Каждый раз, когда
runner завершает `process` job, путь `req.decisions` добавляется в этот реестр
(дедупликация по абсолютному пути, LRU-топ 100).

`GET /projects` эндпоинт читает реестр, для каждого пути читает
`decisions.json` (быстро — это ~несколько KB), собирает метаданные (кол-во
регионов, mtime, длительность из таймкодов `range_in`/`range_out`). Мёртвые/битые пути пропускает
молча — они удалятся при следующем `register()`.

Почему JSON, а не БД: у нас single-user desktop, никаких конкурентных запросов,
файл ~1KB. Sqlite здесь — ovkerkill.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


def _default_config_dir() -> Path:
    """Легаси-каталог конфига (XDG / Windows APPDATA / mac Application Support).

    Использовался как основное место реестра до переезда в DATA_ROOT; теперь —
    только фолбэк для старых установок (см. default_projects_file). Каталог
    здесь не создаём — вызовется явно перед записью.
    """
    if os.name == "nt":
        base = Path(os.environ.get("APPDATA") or (Path.home() / "AppData" / "Roaming"))
    else:
        base = Path(os.environ.get("XDG_CONFIG_HOME") or (Path.home() / ".config"))
    return base / "twitch_cut"


def default_projects_file() -> Path:
    """Путь к реестру. Приоритет:
      1. env TWITCH_CUT_PROJECTS_FILE (тесты подсовывают свой tmp_path),
      2. <DATA_ROOT>/projects.json — рядом с work/ и models/,
      3. XDG/APPDATA — легаси-фолбэк, только если там уже лежит старый файл.

    Реестр хранит пути к decisions.json внутри DATA_ROOT/work/. Держим его в
    том же DATA_ROOT, чтобы portable-папка данных переносилась целиком (иначе
    после переноса «недавние проекты» указывали бы в пустоту). Легаси-путь
    подхватываем, только если новый ещё не создан и старый реально существует —
    так старые установки не теряют список после обновления.
    """
    override = os.environ.get("TWITCH_CUT_PROJECTS_FILE")
    if override:
        return Path(override)

    from ..config import DATA_ROOT

    primary = DATA_ROOT / "projects.json"
    if primary.exists():
        return primary

    legacy = _default_config_dir() / "projects.json"
    if legacy.exists():
        return legacy

    return primary


# --- registry I/O ------------------------------------------------------------

MAX_REGISTRY_ENTRIES = 100


def registered_roots(*, store_path: Optional[Path] = None) -> set[Path]:
    """Родительские каталоги всех зарегистрированных decisions.json (resolved).

    Используется авторизацией data-эндпоинтов (/decisions, /transcript): файлы
    внутри папки любого «недавнего» проекта считаются легальными для чтения и
    записи. Битые записи реестра пропускаются молча.
    """
    store = store_path or default_projects_file()
    roots: set[Path] = set()
    for raw in _load_registry(store):
        try:
            roots.add(Path(raw).expanduser().resolve().parent)
        except Exception:  # noqa: BLE001
            continue
    return roots


def registered_stream_paths(*, store_path: Optional[Path] = None) -> set[Path]:
    """Пути к исходным видео (`source`/`stream`) всех зарегистрированных проектов.

    Исходник живёт ТАМ, куда пользователь его положил (обычно вне work/), а не
    в папке проекта — там только decisions.json + кэш. Авторизация /thumbnail
    строится по этому списку: превью легально только для видео, реально
    привязанного к одному из «недавних» проектов. Пути в decisions.json могут
    быть относительными (напр. "./stream.mp4") — резолвим их относительно папки
    самого decisions.json, а не cwd процесса. Битые записи пропускаем молча.
    """
    store = store_path or default_projects_file()
    out: set[Path] = set()
    for raw in _load_registry(store):
        dpath = Path(raw).expanduser()
        try:
            doc = json.loads(dpath.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        if not isinstance(doc, dict):
            continue
        for key in ("source", "original", "stream"):
            v = doc.get(key)
            if not isinstance(v, str) or not v:
                continue
            p = Path(v).expanduser()
            # Относительные пути — от папки decisions.json (см. build_decisions,
            # source может быть "./stream.mp4").
            if not p.is_absolute():
                p = dpath.resolve().parent / p
            try:
                out.add(p.resolve())
            except Exception:  # noqa: BLE001
                continue
    return out


def _load_registry(store_path: Path) -> list[str]:
    """Прочитать реестр или вернуть пустой список. Ошибки — тихо в лог."""
    if not store_path.exists():
        return []
    try:
        data = json.loads(store_path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        logger.warning("projects registry corrupt (%s), starting empty", exc)
        return []
    if not isinstance(data, dict):
        return []
    paths = data.get("paths")
    if not isinstance(paths, list):
        return []
    # На всякий: фильтруем не-строки, приводим к str.
    return [str(p) for p in paths if isinstance(p, (str, os.PathLike))]


def _save_registry(store_path: Path, paths: list[str]) -> None:
    store_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = store_path.with_suffix(store_path.suffix + ".tmp")
    tmp.write_text(json.dumps({"paths": paths}, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(store_path)


def register_project(decisions_path: str | Path, *, store_path: Optional[Path] = None) -> None:
    """Добавить путь к decisions.json в топ реестра. Идемпотентно.

    Дедуп по resolved-str, MRU-порядок (последний зарегистрированный — первый).
    Кэп на MAX_REGISTRY_ENTRIES — держим реестр компактным.
    """
    store = store_path or default_projects_file()
    key = str(Path(decisions_path).expanduser().resolve())
    paths = _load_registry(store)
    # Убрать старую позицию, если есть, и положить в начало.
    paths = [p for p in paths if p != key]
    paths.insert(0, key)
    if len(paths) > MAX_REGISTRY_ENTRIES:
        paths = paths[:MAX_REGISTRY_ENTRIES]
    try:
        _save_registry(store, paths)
    except Exception as exc:  # noqa: BLE001
        # Не падаем: регистрация — best-effort side-effect. Основная работа
        # (сам job) уже завершилась к этому моменту.
        logger.warning("failed to save projects registry: %s", exc)


# --- listing -----------------------------------------------------------------


def _duration_ms(doc: dict[str, Any]) -> Optional[int]:
    """Длительность обработанного куска в миллисекундах, или None.

    Схема 1.1 (актуальная, см. decisions.build_decisions) хранит границы как
    ТАЙМКОДЫ-строки в ключах `range_in`/`range_out` ("00:01:04.220") плюс готовый
    `local_duration`. Раньше _project_meta искал только числовые `range_*_ms`,
    которых в реальных файлах нет — поэтому длительность в Dashboard всегда была
    пустой. Разбираем оба формата:
      1) `local_duration` — таймкод-строка, самый прямой источник;
      2) `range_in`/`range_out` — таймкоды-строки, берём разницу;
      3) числовые `range_in_ms`/`range_out_ms` или вложенный `range: {in_ms,out_ms}`
         — legacy / тестовый формат.
    """
    from ..timecode import parse_timecode

    local = doc.get("local_duration")
    if isinstance(local, str) and local:
        try:
            return parse_timecode(local).ms
        except Exception:  # noqa: BLE001
            pass

    r_in, r_out = doc.get("range_in"), doc.get("range_out")
    if isinstance(r_in, str) and isinstance(r_out, str):
        try:
            return parse_timecode(r_out).ms - parse_timecode(r_in).ms
        except Exception:  # noqa: BLE001
            pass

    nested = doc.get("range") if isinstance(doc.get("range"), dict) else {}
    in_ms = doc.get("range_in_ms", nested.get("in_ms"))
    out_ms = doc.get("range_out_ms", nested.get("out_ms"))
    if isinstance(in_ms, (int, float)) and isinstance(out_ms, (int, float)):
        return int(out_ms) - int(in_ms)

    return None


def _project_meta(decisions_path: Path) -> Optional[dict[str, Any]]:
    """Собрать метаданные проекта из decisions.json. Возвращает None если битый.

    Тонкость: `updated_at` берём из mtime самого decisions.json (перезаписался
    после сохранения из UI — mtime обновился), а не из содержимого. Это важно
    для сортировки «недавно правил».
    """
    try:
        stat = decisions_path.stat()
        raw = decisions_path.read_text(encoding="utf-8")
        doc = json.loads(raw)
    except Exception:
        return None
    if not isinstance(doc, dict):
        return None
    mutes = doc.get("mutes") or []
    cuts = doc.get("cuts") or []
    highlights_bundle = doc.get("highlights") or {}
    highlights = (
        highlights_bundle.get("highlights") if isinstance(highlights_bundle, dict) else []
    ) or []
    caches = doc.get("caches") or {}
    # В decisions.json путь к исходному видео пишется в "source" (см. build_decisions,
    # schema 1.1). Ключ "stream" — legacy, оставлен как fallback для старых файлов.
    # Кладём в stream_path — как ожидает фронт (ProjectMeta.stream_path).
    # Путь может быть ОТНОСИТЕЛЬНЫМ ("source": "./stream.mp4" — так пишет
    # build_decisions). Резолвим его от папки decisions.json, иначе фронт
    # отправит "./stream.mp4" в /thumbnail, а generate_thumbnail разрешит его от
    # cwd процесса → файл не найдётся (404) или не пройдёт авторизацию (403).
    # Тот же приём — в registered_stream_paths().
    stream_path: Optional[str] = None
    for k in ("source", "stream"):
        v = doc.get(k)
        if isinstance(v, str) and v:
            stream_path = v
            break
    if stream_path:
        p = Path(stream_path).expanduser()
        if not p.is_absolute():
            p = decisions_path.resolve().parent / p
        try:
            stream_path = str(p.resolve())
        except Exception:  # noqa: BLE001
            stream_path = None
    workdir = None
    if isinstance(caches, dict):
        # workdir нигде явно не хранится, но audio-cache лежит внутри него;
        # берём родителя от файла аудио-кэша если он есть.
        audio_cache = caches.get("audio") if isinstance(caches.get("audio"), str) else None
        if audio_cache:
            try:
                workdir = str(Path(audio_cache).parent.parent)
            except Exception:
                workdir = None

    duration_ms = _duration_ms(doc)

    return {
        "decisions_path": str(decisions_path),
        "name": decisions_path.stem or decisions_path.parent.name,
        "workdir": workdir,
        "stream_path": stream_path,
        "updated_at_ms": int(stat.st_mtime * 1000),
        "duration_ms": duration_ms,
        "mutes_count": len(mutes) if isinstance(mutes, list) else 0,
        "cuts_count": len(cuts) if isinstance(cuts, list) else 0,
        "highlights_count": len(highlights) if isinstance(highlights, list) else 0,
    }


def list_projects(*, store_path: Optional[Path] = None) -> list[dict[str, Any]]:
    """Прочитать реестр и вернуть массив project-meta. Мёртвые записи скрыты,
    но НЕ удаляются из реестра — иначе `list_projects` был бы write-side-effect'ом.
    """
    store = store_path or default_projects_file()
    paths = _load_registry(store)
    out: list[dict[str, Any]] = []
    for raw in paths:
        p = Path(raw)
        if not p.exists():
            continue
        meta = _project_meta(p)
        if meta is not None:
            out.append(meta)
    return out


def prune_registry(*, store_path: Optional[Path] = None) -> int:
    """Убрать из реестра пути к несуществующим файлам. Возвращает число удалённых.

    Отдельная функция — вызывается только при явном действии пользователя
    (пункт «Очистить историю» в настройках; в этой фазе — не подключено).
    """
    store = store_path or default_projects_file()
    paths = _load_registry(store)
    kept = [p for p in paths if Path(p).exists()]
    removed = len(paths) - len(kept)
    if removed:
        _save_registry(store, kept)
    return removed


# --- deletion ----------------------------------------------------------------


def _looks_like_project_dir(path: Path) -> bool:
    """Грубая защита от rmtree по чему попало.

    Удаляем папку проекта только если она достаточно «глубокая» (не корень
    диска и не домашний каталог) — т.е. это подпапка внутри work/, созданная
    пайплайном. Домашний каталог и корни трогать запрещаем.
    """
    try:
        resolved = path.expanduser().resolve()
    except Exception:  # noqa: BLE001
        return False
    if not resolved.is_dir():
        return False
    home = Path.home().resolve()
    # Не корень диска (у корня нет "настоящего" родителя) и не сам home.
    if resolved == resolved.parent or resolved == home:
        return False
    # Достаточно вложенная: минимум 2 сегмента пути после якоря.
    if len(resolved.relative_to(resolved.anchor).parts) < 2:
        return False
    return True


def unregister_project(
    decisions_path: str | Path,
    *,
    delete_files: bool = False,
    store_path: Optional[Path] = None,
) -> dict[str, Any]:
    """Убрать проект из реестра и (опционально) удалить его файлы с диска.

    Реестр — источник правды для Dashboard. При delete_files=True дополнительно
    удаляем папку проекта (родитель decisions.json — обычно подпапка в work/,
    куда пайплайн кладёт decisions.json + mutes.cs + cache). Удаление файлов —
    best-effort: если папка «подозрительная» (корень/домашний каталог) или
    удаление упало, реестр всё равно чистим, а причину возвращаем в ответе.

    Возвращает {"unregistered": bool, "deleted_dir": str|None, "error": str|None}.
    """
    store = store_path or default_projects_file()
    key = str(Path(decisions_path).expanduser().resolve())

    paths = _load_registry(store)
    before = len(paths)
    paths = [p for p in paths if p != key]
    unregistered = len(paths) != before
    if unregistered:
        try:
            _save_registry(store, paths)
        except Exception as exc:  # noqa: BLE001
            logger.warning("failed to save projects registry after unregister: %s", exc)

    deleted_dir: Optional[str] = None
    error: Optional[str] = None
    if delete_files:
        import shutil

        project_dir = Path(key).parent
        if _looks_like_project_dir(project_dir):
            try:
                shutil.rmtree(project_dir)
                deleted_dir = str(project_dir)
            except Exception as exc:  # noqa: BLE001
                error = f"не удалось удалить папку {project_dir}: {exc}"
                logger.warning(error)
        else:
            error = f"папка {project_dir} не похожа на папку проекта — файлы не тронуты"
            logger.warning(error)

    return {"unregistered": unregistered, "deleted_dir": deleted_dir, "error": error}

