"""FastAPI приложение для desktop UI.

Дизайн:
- Всё что делает CLI (Фаза 1 + Фаза 3) вынесено в `runner.py`.
- Здесь только транспорт: JSON in → JobStore.create + create_task → job_id.
  Прогресс/финал через WS. Отмена — DELETE.
- Никакой БД, никакой персистентности между запусками процесса.
- CORS открыт для localhost:5173 (Vite dev-server). В prod фронт монтируется
  как статика из `desktop/dist/` — тогда CORS не нужен вообще (same-origin).
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any, Optional

from fastapi import Body, FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from fastapi import Request

from ..cache import read_json, write_json
from .media import build_media_response, resolve_and_authorize
from .projects import list_projects as _list_projects_registry
from .settings import (
    load_settings as _load_settings,
    logs_dir as _logs_dir,
    save_settings as _save_settings,
)
from .waveform import compute_waveform
from .runner import (
    run_export_vegas_job,
    run_highlights_export_job,
    run_highlights_import_job,
    run_process_job,
)
from .schemas import (
    DecisionsPayload,
    ExportVegasRequest,
    HighlightsExportRequest,
    HighlightsImportRequest,
    JobResponse,
    ProcessJobRequest,
)
from .state import JobKind, JobStore

logger = logging.getLogger(__name__)


def _state_to_response(state) -> JobResponse:
    """Snapshot JobState в JobResponse (Pydantic validation on the wire)."""
    return JobResponse(
        id=state.id,
        kind=state.kind.value,
        status=state.status.value,
        progress=state.progress,
        stage=state.stage,
        message=state.message,
        result=state.result,
        error=state.error,
        created_at=state.created_at,
        finished_at=state.finished_at,
    )


def create_app(
    *,
    store: Optional[JobStore] = None,
    static_dir: Optional[Path] = None,
    dev_cors: bool = True,
) -> FastAPI:
    """Собрать FastAPI-приложение.

    :param store: инъекция JobStore (для тестов). По умолчанию создаётся новый.
    :param static_dir: каталог со статикой фронта (desktop/dist). Если задан
        и существует — маунтится под `/`. index.html становится корнем.
    :param dev_cors: разрешить CORS с localhost:5173 (для Vite dev).
    """
    app = FastAPI(title="twitch-cut backend", version="0.1.0")
    app.state.store = store or JobStore()

    if dev_cors:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
            allow_methods=["*"],
            allow_headers=["*"],
        )

    # --- health -------------------------------------------------------------

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {"ok": True, "jobs": len(app.state.store.list())}

    # --- jobs: create -------------------------------------------------------

    def _spawn(kind: JobKind, coro_factory):
        state = app.state.store.create(kind)
        task = asyncio.get_event_loop().create_task(coro_factory(state))
        app.state.store.attach_task(state.id, task)
        return state

    @app.post("/jobs/process", response_model=JobResponse)
    async def create_process_job(req: ProcessJobRequest):
        # Пополняем медиа-whitelist сразу: как только пользователь запустил job,
        # ему легально смотреть исходный stream/оригинал через /media (для
        # Timeline после завершения). original может быть None (простой режим).
        app.state.store.allow_media(req.stream, req.original or "")
        state = _spawn(
            JobKind.PROCESS,
            lambda s: run_process_job(app.state.store, s, req),
        )
        return _state_to_response(state)

    @app.post("/jobs/export-vegas", response_model=JobResponse)
    async def create_export_vegas_job(req: ExportVegasRequest):
        state = _spawn(
            JobKind.EXPORT_VEGAS,
            lambda s: run_export_vegas_job(app.state.store, s, req),
        )
        return _state_to_response(state)

    @app.post("/jobs/highlights-export", response_model=JobResponse)
    async def create_highlights_export_job(req: HighlightsExportRequest):
        state = _spawn(
            JobKind.HIGHLIGHTS_EXPORT,
            lambda s: run_highlights_export_job(app.state.store, s, req),
        )
        return _state_to_response(state)

    @app.post("/jobs/highlights-import", response_model=JobResponse)
    async def create_highlights_import_job(req: HighlightsImportRequest):
        state = _spawn(
            JobKind.HIGHLIGHTS_IMPORT,
            lambda s: run_highlights_import_job(app.state.store, s, req),
        )
        return _state_to_response(state)

    # --- jobs: query --------------------------------------------------------

    @app.get("/jobs", response_model=list[JobResponse])
    async def list_jobs():
        return [_state_to_response(s) for s in app.state.store.list()]

    @app.get("/jobs/{job_id}", response_model=JobResponse)
    async def get_job(job_id: str):
        state = app.state.store.get(job_id)
        if state is None:
            raise HTTPException(status_code=404, detail=f"job not found: {job_id}")
        return _state_to_response(state)

    @app.delete("/jobs/{job_id}")
    async def cancel_job(job_id: str):
        if app.state.store.get(job_id) is None:
            raise HTTPException(status_code=404, detail=f"job not found: {job_id}")
        cancelled = app.state.store.cancel(job_id)
        return {"cancelled": cancelled}

    # --- jobs: events (WS) --------------------------------------------------

    @app.websocket("/jobs/{job_id}/events")
    async def job_events(ws: WebSocket, job_id: str):
        await ws.accept()
        if app.state.store.get(job_id) is None:
            await ws.send_text(json.dumps({"type": "error", "detail": "job not found"}))
            await ws.close()
            return
        try:
            async for event in app.state.store.subscribe(job_id):
                await ws.send_text(json.dumps(event))
        except WebSocketDisconnect:
            logger.info("ws client disconnected job=%s", job_id)
            return
        finally:
            try:
                await ws.close()
            except Exception:  # noqa: BLE001
                pass

    # --- decisions I/O ------------------------------------------------------

    @app.get("/decisions")
    async def read_decisions(path: str = Query(...)):
        p = Path(path)
        if not p.exists():
            raise HTTPException(status_code=404, detail=f"decisions file not found: {p}")
        return JSONResponse(read_json(p))

    @app.put("/decisions")
    async def write_decisions_endpoint(path: str = Query(...), payload: DecisionsPayload = Body(...)):
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        write_json(p, payload.decisions)
        # Регистрируем в реестре Recent Projects — покрывает случай, когда
        # пользователь правит существующий decisions.json из UI (Timeline).
        try:
            from .projects import register_project
            register_project(p)
        except Exception:  # noqa: BLE001
            pass
        return {"ok": True, "path": str(p)}

    # --- projects registry --------------------------------------------------

    @app.get("/projects")
    async def list_projects():
        """Список недавних проектов из реестра.

        Формат каждого элемента см. `projects._project_meta`. Битые/удалённые
        пути пропускаются молча (не удаляются из реестра — это делает
        отдельная кнопка «Очистить историю» в настройках).
        """
        return _list_projects_registry()

    @app.delete("/projects")
    async def delete_project(
        path: str = Query(..., description="Путь к decisions.json проекта"),
        delete_files: bool = Query(
            False, description="Также удалить папку проекта с диска"
        ),
    ):
        """Удалить проект из реестра недавних.

        По умолчанию убирает только запись из реестра (файлы на диске остаются).
        При delete_files=true дополнительно удаляет папку проекта (родитель
        decisions.json) — best-effort, с защитой от rmtree по корню/домашнему
        каталогу. Результат см. `projects.unregister_project`.
        """
        from .projects import unregister_project

        return unregister_project(path, delete_files=delete_files)

    # --- settings -----------------------------------------------------------

    @app.get("/settings")
    async def get_settings():
        """Прочитать пользовательские настройки. Формат плоский dict; см.
        `settings.py`. Отсутствие файла — не ошибка, просто {}.

        Дефолты device/compute_type НЕ хардкодим на 'cuda': если backend
        запущен в CPU-режиме (TWITCH_CUT_CPU=1 — GPU не найден), подставляем
        cpu/int8, чтобы форма NewJob не предлагала заведомо падающий cuda.
        Явно сохранённые пользователем значения не трогаем.
        """
        from ..config import default_compute_type, default_device

        data = _load_settings()
        data.setdefault("default_device", default_device())
        data.setdefault("default_compute_type", default_compute_type())
        return data

    @app.put("/settings")
    async def put_settings(payload: dict[str, Any] = Body(...)):
        """Полная перезапись settings.json тем, что прислал клиент.

        Никакой валидации ключей — фронт сам знает свою схему, backend только
        хранит. Merge не делаем: клиент присылает полный документ, включая
        неизвестные ему поля из будущих версий (они приходили в GET и должны
        уйти обратно as-is для round-trip).
        """
        _save_settings(payload)
        return {"ok": True}

    @app.get("/settings/logs_dir")
    async def get_logs_dir():
        """Путь к папке логов / конфига. Frontend показывает кнопку «Открыть»
        и передаёт этот путь в electron shell.openPath().
        """
        return {"path": str(_logs_dir())}

    @app.get("/paths/suggest_workdir")
    async def suggest_workdir(stream: str = Query("", description="Путь к stream-файлу (опционально)")):
        """Рекомендованный workdir для нового job'а.

        Формат: ~/twitch_cut/projects/{basename}_{yyyymmdd_hhmm}/
        Гарантирует уникальность (в имени — минута), чтобы два запуска подряд
        не перезаписывали decisions.json друг друга. Если stream не передан —
        используем 'job' как basename.

        Возвращает только рекомендацию — папку НЕ создаёт (создастся при
        первом write в pipeline). UI показывает путь в поле workdir, юзер может
        отредактировать.
        """
        from .paths import suggested_workdir

        return {"path": str(suggested_workdir(stream or None))}


    @app.get("/transcript")
    async def read_transcript(path: str = Query(...)):
        p = Path(path)
        if not p.exists():
            raise HTTPException(status_code=404, detail=f"transcript file not found: {p}")
        return JSONResponse(read_json(p))

    # --- waveform / media (Timeline screen) ---------------------------------

    @app.get("/waveform")
    async def get_waveform(
        path: str = Query(..., description="Абсолютный путь к stream.mp4/mp3/wav"),
        peaks: int = Query(1024, ge=64, le=8192),
    ):
        """Peaks для отрисовки waveform в UI. Кэшируется.

        Разрешает только whitelisted пути — тот же список, что и /media.
        Иначе левый сайт мог бы вычислить peaks произвольного файла и по
        характеристикам понять что там (attack: peaks-инфа приватна).
        """
        allowed = app.state.store.allowed_media()
        resolved = resolve_and_authorize(path, allowed)
        try:
            return await asyncio.to_thread(compute_waveform, resolved, peaks)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc))
        except Exception as exc:  # noqa: BLE001
            logger.exception("waveform failed for %s", resolved)
            raise HTTPException(status_code=500, detail=f"waveform failed: {exc}")

    @app.post("/waveform/allow")
    async def allow_media_path(payload: dict[str, Any] = Body(...)):
        """Добавить путь в whitelist медиа (для Timeline при открытии проекта).

        Тело: `{"paths": ["/abs/a.mp4", "/abs/b.mp3"]}`. Валидируем что каждый
        путь существует — тогда добавляем.
        """
        raw = payload.get("paths") or []
        if not isinstance(raw, list):
            raise HTTPException(status_code=422, detail="'paths' must be a list")
        added: list[str] = []
        for item in raw:
            if not isinstance(item, str):
                continue
            p = Path(item).expanduser()
            if p.exists():
                app.state.store.allow_media(str(p))
                added.append(str(p.resolve()))
        return {"allowed": added}

    @app.get("/media")
    async def get_media(request: Request, path: str = Query(...)):
        """Стриминг медиа-файла с поддержкой HTTP Range для `<audio>`.currentTime."""
        allowed = app.state.store.allowed_media()
        resolved = resolve_and_authorize(path, allowed)
        return build_media_response(resolved, request.headers.get("range"))

    # --- static (prod) ------------------------------------------------------

    if static_dir is not None and Path(static_dir).exists():
        # html=True → отдавать index.html на корень и любой неизвестный путь
        # (нужно React-роутеру, если появится client-side routing).
        app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="static")

    return app
