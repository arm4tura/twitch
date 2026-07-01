"""JobStore: in-process реестр джоб + очереди событий на подписчиков.

Никакой БД. Джоба живёт пока живёт процесс `uvicorn`. Это desktop-приложение
на одного пользователя — сессия = один запуск Electron; всё, что нужно
пережить перезапуск, пишется в decisions.json / workdir/cache/ на диск.

Модель конкурентности:
- Каждая джоба — `asyncio.Task`. Отмена = `task.cancel()` внутри runner ловится
  как `CancelledError` и переводит статус в 'cancelled'.
- Подписчики (WS-клиенты) получают события через `asyncio.Queue`. Очередь одна
  на джобу — runner пушит, listener читает. Дублируем в `queue.put(None)` как
  sentinel, чтобы WS понял «финал, закрывай сокет».
- Позднее подключение к запущенной джобе: отдаём текущий snapshot через
  синхронный HTTP GET и подписываемся на будущие события. Историю событий
  не хранию — прогресс coarse-grained, потеря 1-2 промежуточных tick'ов
  не критична; клиент всё равно получит финальный state.
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    CANCELLED = "cancelled"


class JobKind(str, Enum):
    PROCESS = "process"
    EXPORT_VEGAS = "export_vegas"
    HIGHLIGHTS_EXPORT = "highlights_export"
    HIGHLIGHTS_IMPORT = "highlights_import"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class JobState(BaseModel):
    """Snapshot состояния джобы. Изменяемое поле — все атрибуты BaseModel.

    Мутабельность здесь — сознательный выбор: runner в hot-path патчит
    `state.progress = ...` в цикле, копировать каждый раз через `.model_copy`
    было бы дороже. Если понадобится иммутабельность (напр. для очередей
    событий), делаем `state.model_copy(deep=True)` в точке фиксации.
    """

    model_config = ConfigDict(extra="forbid")

    id: str
    kind: JobKind
    status: JobStatus = JobStatus.PENDING
    progress: float = 0.0
    stage: str = ""
    message: str = ""
    result: Optional[dict[str, Any]] = None
    error: Optional[str] = None
    created_at: str = Field(default_factory=_now_iso)
    finished_at: Optional[str] = None

    def is_terminal(self) -> bool:
        return self.status in {JobStatus.DONE, JobStatus.FAILED, JobStatus.CANCELLED}


class _JobRecord:
    """Внутренняя запись в реестре: state + fan-out очереди + фоновая asyncio.Task."""

    __slots__ = ("state", "subscribers", "task", "final_event")

    def __init__(self, state: JobState) -> None:
        self.state = state
        # Fan-out: у КАЖДОГО подписчика — своя очередь. Runner пушит через
        # store.emit → каждая очередь получает копию. Это критично, потому что:
        #   1) UI может открыть/закрыть/переоткрыть JobScreen — новый consumer
        #      не должен «съедать» события у старого;
        #   2) при переподключении WS новый handler создаёт новую очередь и
        #      получает catch-up snapshot первым событием (см. subscribe).
        # Ключ — id объекта очереди; значение — сама очередь. Set удобен для
        # удаления при выходе из subscribe().
        self.subscribers: set[asyncio.Queue[Optional[dict[str, Any]]]] = set()
        self.task: Optional[asyncio.Task[Any]] = None
        # Финальное событие держим отдельно, чтобы поздние подписчики получили
        # именно final (со всеми полями result/error), а не просто «терминальный
        # snapshot». Заполняется в первом emit type=final.
        self.final_event: Optional[dict[str, Any]] = None


class JobStore:
    """Реестр всех джоб в процессе.

    Публичный API:
      - `create(kind) -> JobState`  создать пустую запись, вернуть snapshot.
      - `get(id) -> JobState | None`
      - `list() -> list[JobState]`
      - `cancel(id) -> bool`         попросить отмену (не блокирует).
      - `attach_task(id, task)`      привязать asyncio.Task, созданный runner'ом.
      - `emit(id, event)`            пуш события в очередь (для runner'а).
      - `subscribe(id)`              async-генератор событий (для WS handler'а).
    """

    def __init__(self) -> None:
        self._records: dict[str, _JobRecord] = {}
        # Whitelist медиа-путей, которые разрешено отдавать через /media.
        # Пополняется при создании job'а (stream/original) и при загрузке
        # проекта (decisions.stream). Без этого /media был бы open-file-read
        # для любого HTTP-клиента на localhost.
        self._allowed_media: set[str] = set()

    # ---- media whitelist -------------------------------------------------

    def allow_media(self, *paths: str) -> None:
        """Разрешить отдачу перечисленных абсолютных путей через /media."""
        from pathlib import Path as _Path

        for p in paths:
            if not p:
                continue
            self._allowed_media.add(str(_Path(p).expanduser().resolve()))

    def allowed_media(self) -> set[str]:
        return set(self._allowed_media)

    # ---- CRUD ------------------------------------------------------------

    def create(self, kind: JobKind) -> JobState:
        job_id = uuid.uuid4().hex
        state = JobState(id=job_id, kind=kind)
        self._records[job_id] = _JobRecord(state)
        return state

    def get(self, job_id: str) -> Optional[JobState]:
        rec = self._records.get(job_id)
        return rec.state if rec else None

    def list(self) -> list[JobState]:
        return [rec.state for rec in self._records.values()]

    def _record(self, job_id: str) -> Optional[_JobRecord]:
        return self._records.get(job_id)

    # ---- lifecycle -------------------------------------------------------

    def attach_task(self, job_id: str, task: asyncio.Task[Any]) -> None:
        rec = self._records.get(job_id)
        if rec is None:
            raise KeyError(f"Job not found: {job_id}")
        rec.task = task

    def cancel(self, job_id: str) -> bool:
        """Попросить отмену. Возвращает True, если было что отменять.

        Отмену производит `runner`: `CancelledError` бросается внутри `to_thread`,
        runner ловит и переводит status в CANCELLED через `finish()`. Здесь мы
        только зовём `task.cancel()`.
        """
        rec = self._records.get(job_id)
        if rec is None:
            return False
        if rec.state.is_terminal():
            return False
        if rec.task is not None and not rec.task.done():
            rec.task.cancel()
            return True
        # Джоба ещё не стартовала (пример: сразу после create). Отметим отменённой.
        rec.state.status = JobStatus.CANCELLED
        rec.state.finished_at = _now_iso()
        # Разблокируем возможных подписчиков — пусть увидят final и закроют сокет.
        final = {"type": "final", "state": rec.state.model_dump(mode="json")}
        rec.final_event = final
        for q in rec.subscribers:
            q.put_nowait(final)
            q.put_nowait(None)
        return True

    # ---- events ----------------------------------------------------------

    def emit(self, job_id: str, event: dict[str, Any]) -> None:
        """Пуш события ВСЕМ подписчикам. Never blocks (unbounded queues)."""
        rec = self._records.get(job_id)
        if rec is None:
            return
        # Запомним final, чтобы поздние подписчики получили его как первое
        # событие (см. subscribe: catch-up).
        if event.get("type") == "final":
            rec.final_event = event
        for q in rec.subscribers:
            q.put_nowait(event)

    def close_stream(self, job_id: str) -> None:
        """Sentinel для ВСЕХ подписчиков — «больше событий не будет»."""
        rec = self._records.get(job_id)
        if rec is None:
            return
        for q in rec.subscribers:
            q.put_nowait(None)

    async def subscribe(self, job_id: str):
        """Async-итератор событий джобы. Отдаёт финальный snapshot и закрывается.

        Поведение:
          - при подписке СРАЗУ отдаём текущий state как type='snapshot' — это
            «catch-up» для позднего/переподключившегося клиента: UI мгновенно
            видит актуальный progress/stage без ожидания следующего tick'а.
          - если джоба уже терминальна, отдаём сохранённый final и выходим.
          - иначе — регистрируем свою очередь в rec.subscribers, слушаем.
            Финализация: убираем очередь из subscribers, чтобы emit не пушил
            в мёртвую ссылку.
        """
        rec = self._records.get(job_id)
        if rec is None:
            return
        # Всегда отдаём snapshot первым — UI получает актуальные stage/progress
        # мгновенно, даже если после этого не будет ни одного нового события
        # (напр. джоба зависла между tick'ами).
        yield {"type": "snapshot", "state": rec.state.model_dump(mode="json")}
        # Late-subscribe fast-path: терминальные — сразу final, дальше ничего.
        if rec.state.is_terminal():
            if rec.final_event is not None:
                yield rec.final_event
            return
        q: asyncio.Queue[Optional[dict[str, Any]]] = asyncio.Queue()
        rec.subscribers.add(q)
        try:
            while True:
                event = await q.get()
                if event is None:
                    return
                yield event
                if event.get("type") == "final":
                    return
        finally:
            # Отвязываемся ВСЕГДА, даже при CancelledError (WS disconnect).
            # Иначе emit() будет пушить в осиротевшую очередь и медленно
            # утекать память при частых reconnect'ах.
            rec.subscribers.discard(q)
