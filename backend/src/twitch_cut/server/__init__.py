"""FastAPI backend для desktop UI (Фаза 4).

Модуль оборачивает существующий CLI-пайплайн (Фаза 1 + Фаза 3) в HTTP/WS
транспорт для Electron-приложения. Ничего в CLI-логике не меняется — только
новый транспортный слой.
"""

from .app import create_app
from .schemas import (
    ExportVegasRequest,
    HighlightsExportRequest,
    HighlightsImportRequest,
    JobEvent,
    JobResponse,
    ProcessJobRequest,
)
from .state import JobKind, JobState, JobStatus, JobStore

__all__ = [
    "create_app",
    "ExportVegasRequest",
    "HighlightsExportRequest",
    "HighlightsImportRequest",
    "JobEvent",
    "JobResponse",
    "ProcessJobRequest",
    "JobKind",
    "JobState",
    "JobStatus",
    "JobStore",
]
