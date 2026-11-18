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
    SettingsValidationError,
    load_settings as _load_settings,
    logs_dir as _logs_dir,
    save_settings as _save_settings,
    validate_settings as _validate_settings,
)
from .waveform import compute_waveform, compute_waveform_range
from .runner import (
    run_detect_offtopic_job,
    run_detect_silence_job,
    run_download_vod_job,
    run_export_vegas_highlights_job,
    run_export_vegas_job,
    run_highlights_export_job,
    run_highlights_import_job,
    run_process_job,
    run_render_highlights_job,
    run_render_job,
)
from .schemas import (
    DecisionsPayload,
    DetectOfftopicRequest,
    DetectSilenceRequest,
    DownloadVodRequest,
    ExportVegasHighlightsRequest,
    ExportVegasRequest,
    HighlightsExportRequest,
    HighlightsImportRequest,
    JobResponse,
    ProcessJobRequest,
    RenderHighlightsRequest,
    RenderRequest,
    TestConnectionRequest,
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


def _authorize_data_path(path_str: str, *, must_exist: bool = True) -> Path:
    """Авторизация путей data-эндпоинтов (/decisions, /transcript).

    В отличие от /media (whitelist в JobStore), эти эндпоинты работают с
    JSON-артефактами пайплайна. Легальны только пути, лежащие внутри:
      - DATA_ROOT (work/ — туда пайплайн пишет прогоны сам), либо
      - каталога любого зарегистрированного «недавнего» проекта.

    Без этой проверки GET/PUT /decisions и GET /transcript были бы
    произвольным чтением и записью любого файла на диске: сайт, открытый в
    браузере пользователя, мог бы дёрнуть 127.0.0.1:PORT/decisions?path=...
    и вычитать или перезаписать чужой файл (DNS-rebinding обходит loopback).

    must_exist=False — для PUT: файл ещё может не существовать, проверяем
    только что каталог назначения разрешён.
    """
    from ..config import DATA_ROOT
    from .projects import registered_roots

    p = Path(path_str).expanduser().resolve()
    if must_exist and not p.exists():
        raise HTTPException(status_code=404, detail=f"file not found: {p}")

    allowed_roots = {DATA_ROOT.resolve()} | registered_roots()
    if not any(root in p.parents or root == p for root in allowed_roots):
        raise HTTPException(status_code=403, detail=f"path not authorized: {p}")
    return p


def _previous_decisions(path: Path) -> dict[str, Any] | None:
    """Что лежит в decisions.json прямо сейчас — для журнала правок.

    Нужно ДО перезаписи. Журнал — необязательная диагностика, поэтому любая
    проблема с чтением (нет файла, битый json) означает «сравнивать не с чем»,
    а не ошибку сохранения.
    """
    if not _edit_journal_enabled():
        return None
    try:
        data = read_json(path)
    except Exception:  # noqa: BLE001
        return None
    return data if isinstance(data, dict) else None


def _edit_journal_enabled() -> bool:
    try:
        return bool(_load_settings().get("edit_journal"))
    except Exception:  # noqa: BLE001
        return False


def _journal_save(path: Path, doc: Any, previous: dict[str, Any] | None) -> None:
    """Дописать журнал правок. НИКОГДА не мешает сохранению — только лог."""
    if not isinstance(doc, dict) or not _edit_journal_enabled():
        return
    try:
        from ..edit_journal import record_save

        record_save(path, doc, previous)
    except Exception:  # noqa: BLE001
        logger.warning("edit journal write failed for %s", path, exc_info=True)


def _extract_vod_id(raw: str) -> str:
    """Извлечь числовой Twitch VOD ID из URL или голого ID.

    Дублирует `parseTwitchVodInput` во фронте (desktop/src/lib/twitch.ts), но
    минимально: фронт уже провалидировал и мог прислать как полный URL, так и
    голый ID. Клипы/каналы отсекаем — их TwitchDownloaderCLI так не качает.

    Кидает ValueError с человекочитаемым текстом, если ID не извлекается.
    """
    import re
    from urllib.parse import urlparse

    s = (raw or "").strip()
    if not s:
        raise ValueError("Пустая ссылка на VOD.")

    # Голый числовой ID.
    if s.isdigit():
        return s

    # Пытаемся как URL (добавляем схему, если её нет).
    parsed = urlparse(s if re.match(r"^https?://", s, re.I) else f"https://{s}")
    host = (parsed.hostname or "").lower()
    if not re.search(r"(^|\.)twitch\.tv$", host):
        raise ValueError("Это не ссылка на Twitch. Ожидается twitch.tv/videos/<id>.")

    m = re.match(r"^/videos/(\d+)/?$", parsed.path)
    if not m:
        raise ValueError(
            "Не похоже на ссылку на VOD. Клипы и трансляции не поддерживаются — "
            "нужна ссылка вида twitch.tv/videos/<id>."
        )
    return m.group(1)


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
    app = FastAPI(title="twitch-cut backend", version="0.2.0")
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

    @app.post("/jobs/render-highlights", response_model=JobResponse)
    async def create_render_highlights_job(req: RenderHighlightsRequest):
        # Вход (исходник) разрешаем сразу; выходные клипы whitelist'ит сам
        # runner по завершении (их имена известны только после рендера).
        app.state.store.allow_media(req.input_path)
        state = _spawn(
            JobKind.RENDER_HIGHLIGHTS,
            lambda s: run_render_highlights_job(app.state.store, s, req),
        )
        return _state_to_response(state)

    @app.post("/jobs/export-vegas-highlights", response_model=JobResponse)
    async def create_export_vegas_highlights_job(req: ExportVegasHighlightsRequest):
        state = _spawn(
            JobKind.EXPORT_VEGAS_HIGHLIGHTS,
            lambda s: run_export_vegas_highlights_job(app.state.store, s, req),
        )
        return _state_to_response(state)

    @app.post("/jobs/detect-silence", response_model=JobResponse)
    async def create_detect_silence_job(req: DetectSilenceRequest):
        # Разрешаем читать это медиа через /media (Timeline будет
        # накладывать найденные сегменты на дорожку).
        app.state.store.allow_media(req.video_path)
        state = _spawn(
            JobKind.DETECT_SILENCE,
            lambda s: run_detect_silence_job(app.state.store, s, req),
        )
        return _state_to_response(state)

    @app.post("/jobs/render", response_model=JobResponse)
    async def create_render_job(req: RenderRequest):
        # Whitelist и вход (читаем), и выход (после рендера Timeline/плеер
        # захочет его проиграть через /media).
        app.state.store.allow_media(req.input_path, req.output_path)
        state = _spawn(
            JobKind.RENDER,
            lambda s: run_render_job(app.state.store, s, req),
        )
        return _state_to_response(state)

    @app.post("/jobs/detect-offtopic", response_model=JobResponse)
    async def create_detect_offtopic_job(req: DetectOfftopicRequest):
        # Секрет (req.api_key) едет в теле по loopback, используется джобой
        # транзитно и не персистится (см. runner._run_detect_offtopic_sync).
        # Путь по исходнику читает локальный файл — заносим его в whitelist,
        # как вход остальных джоб (resolve_source_path отвергает URL). Транскрипт
        # стрима не whitelist'им: он берётся из decisions, не отдаётся в /media.
        if req.source_audio:
            app.state.store.allow_media(req.source_audio)
        state = _spawn(
            JobKind.DETECT_OFFTOPIC,
            lambda s: run_detect_offtopic_job(app.state.store, s, req),
        )
        return _state_to_response(state)

    # --- VOD download (Twitch) ----------------------------------------------

    @app.get("/vod/info")
    async def get_vod_info(url: str = Query(..., description="Twitch VOD URL или ID")):
        """Метаданные VOD (title/длительность/качества) без скачивания.

        Зовёт `TwitchDownloaderCLI info`. Быстрый вызов для превью-карточки —
        фронт дебаунсит его на вводе ссылки. Если бинарник не найден или CLI
        вернул ошибку — 4xx с человекочитаемым detail; фронт деградирует к
        режиму «метаданные недоступны, скачать всё равно можно».
        """
        from ..tdcli import TdcliError, fetch_vod_info

        try:
            video_id = _extract_vod_id(url)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

        try:
            info = await asyncio.to_thread(fetch_vod_info, video_id)
        except TdcliError as exc:
            raise HTTPException(status_code=502, detail=str(exc))
        return info

    @app.post("/jobs/download-vod", response_model=JobResponse)
    async def create_download_vod_job(req: DownloadVodRequest):
        """Запустить скачивание VOD. Прогресс/финал — через тот же WS, что и
        остальные джобы. По завершении result.stream — путь к готовому mp4,
        который можно сразу скормить в /jobs/process.
        """
        from .paths import suggested_workdir

        try:
            video_id = _extract_vod_id(req.url)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

        # Резолвим workdir и имя выходного файла ЗДЕСЬ (не в runner'е), чтобы
        # сразу вернуть предсказуемый путь и добавить его в whitelist. Имя
        # файла — vod_<id>.mp4 внутри уникальной по минуте подпапки work/.
        workdir = (
            Path(req.workdir).expanduser()
            if req.workdir
            else suggested_workdir(f"vod_{video_id}")
        )
        output = workdir / f"vod_{video_id}.mp4"

        state = _spawn(
            JobKind.DOWNLOAD_VOD,
            lambda s: run_download_vod_job(app.state.store, s, req, video_id, output),
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
        p = _authorize_data_path(path)
        return JSONResponse(read_json(p))

    @app.put("/decisions")
    async def write_decisions_endpoint(path: str = Query(...), payload: DecisionsPayload = Body(...)):
        p = _authorize_data_path(path, must_exist=False)
        p.parent.mkdir(parents=True, exist_ok=True)
        # Читаем предыдущее состояние ДО записи: журнал правок сравнивает
        # «что было на диске» с «что пришло». В сам документ ничего не
        # добавляем — decisions.json должен остаться байт-в-байт тем, что
        # прислал фронт (см. test_decisions_read_write_roundtrip).
        previous = _previous_decisions(p)
        write_json(p, payload.decisions)
        _journal_save(p, payload.decisions, previous)
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

        Известные ключи валидируются по типу/диапазону (см.
        `settings.validate_settings`), неизвестные сохраняются as-is для
        round-trip полей из будущих версий. Валидация защищает от кривого
        клиента и постороннего POST на localhost, способного записать в
        settings.json значения не того типа и уронить старт приложения.
        Merge не делаем: клиент присылает полный документ.
        """
        try:
            validated = _validate_settings(payload)
        except SettingsValidationError as exc:
            raise HTTPException(status_code=422, detail=str(exc))
        _save_settings(validated)
        return {"ok": True}

    # --- AI provider/template config (без секретов) -------------------------

    @app.get("/ai/providers")
    async def get_ai_providers():
        """Встроенные пресеты + сохранённые пользовательские (из settings.json).

        Секретов не содержит: провайдер хранит только api_key_ref (ссылку на
        запись в safeStorage), не сам ключ.
        """
        from ..llm_offtopic import BUILTIN_PROVIDERS

        builtins = [p.model_dump() for p in BUILTIN_PROVIDERS]
        saved = _load_settings().get("ai_providers") or []
        return {"builtin": builtins, "custom": saved}

    @app.get("/ai/templates")
    async def get_ai_templates():
        """Встроенные шаблоны промптов + сохранённые пользовательские клоны."""
        from ..llm_offtopic import BUILTIN_TEMPLATES

        builtins = [t.model_dump() for t in BUILTIN_TEMPLATES]
        saved = _load_settings().get("ai_templates") or []
        return {"builtin": builtins, "custom": saved}

    @app.post("/ai/test-connection")
    async def test_ai_connection(req: TestConnectionRequest):
        """Проверить доступность провайдера (лёгкий ping-запрос к LLM).

        Синхронная проверка (не job): результат нужен UI сразу. Секрет
        (req.api_key) используется транзитно и не персистится/не логируется.
        Сетевой round-trip — в to_thread, чтобы не блокировать event-loop.
        """
        from ..llm_offtopic import AIProvider, LLMClient, builtin_provider

        provider = builtin_provider(req.provider_id)
        if provider is None:
            if not req.provider:
                raise HTTPException(
                    status_code=400,
                    detail=f"unknown provider_id={req.provider_id!r} and no custom `provider`",
                )
            try:
                provider = AIProvider.model_validate(req.provider)
            except Exception as exc:  # noqa: BLE001
                raise HTTPException(status_code=422, detail=f"invalid provider: {exc}")

        client = LLMClient(provider, api_key=req.api_key)
        ok, detail = await asyncio.to_thread(client.test_connection)
        return {"ok": ok, "detail": detail}

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
        p = _authorize_data_path(path)
        return JSONResponse(read_json(p))

    @app.get("/prompt")
    async def get_prompt(n_highlights: int = Query(5, ge=1, le=50)):
        """Готовый промпт для NotebookLM — фронт даёт кнопку «Скопировать промпт».

        Тот же текст, что exporter кладёт в prompt.md пакета, но получить его
        можно и без сборки пакета: пользователь копирует промпт, идёт в
        NotebookLM (или любую другую LLM) и вставляет рядом с транскриптом.
        n_highlights — сколько моментов просить (дефолт 5, как у экспортёра).
        """
        from ..llm.prompt import build_prompt

        return {"prompt": build_prompt(n_highlights), "n_highlights": n_highlights}

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

    @app.get("/waveform/range")
    async def get_waveform_range(
        path: str = Query(..., description="Абсолютный путь к stream.mp4/mp3/wav"),
        start: float = Query(..., ge=0.0, description="Начало окна, сек"),
        end: float = Query(..., gt=0.0, description="Конец окна, сек"),
        res: int = Query(2048, ge=64, le=8192, description="Бакетов в окне"),
    ):
        """Peaks ТОЛЬКО для видимого окна («слой детали», как в Audacity/Reaper).

        НЕ кэшируется на диск: комбинаций (start, end, res) бесконечно, а декод
        узкого слайса дешёвый (`-ss` до `-i`). Фронт кэширует тайлы в памяти.

        Whitelist тот же, что у /waveform и /media — иначе левый сайт мог бы
        снять peaks произвольного файла (peaks-инфа приватна).
        """
        allowed = app.state.store.allowed_media()
        resolved = resolve_and_authorize(path, allowed)
        if end <= start:
            raise HTTPException(status_code=422, detail="end must be > start")
        try:
            return await asyncio.to_thread(
                compute_waveform_range, resolved, start, end, res
            )
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc))
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc))
        except Exception as exc:  # noqa: BLE001
            logger.exception("waveform range failed for %s", resolved)
            raise HTTPException(status_code=500, detail=f"waveform range failed: {exc}")

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

    @app.get("/thumbnail")
    async def get_thumbnail(
        path: str = Query(..., description="Абсолютный путь к записи стрима"),
        w: int = Query(640, ge=160, le=1920, description="Ширина превью в px"),
    ):
        """JPEG-превью кадра записи для карточки проекта на Dashboard.

        Кадр берётся на 10% обработанного диапазона (после интро) и кэшируется
        по фингерпринту файла. Авторизация — только пути внутри зарегистрированных
        проектов. Если исходник удалён/перемещён — 404, и фронт откатывается на
        Film-плейсхолдер.
        """
        from fastapi.responses import FileResponse
        from .thumbnail import generate_thumbnail

        jpg = await asyncio.to_thread(generate_thumbnail, path, w)
        return FileResponse(
            jpg,
            media_type="image/jpeg",
            headers={"Cache-Control": "private, max-age=86400"},
        )

    # --- static (prod) ------------------------------------------------------

    if static_dir is not None and Path(static_dir).exists():
        # html=True → отдавать index.html на корень и любой неизвестный путь
        # (нужно React-роутеру, если появится client-side routing).
        app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="static")

    return app
