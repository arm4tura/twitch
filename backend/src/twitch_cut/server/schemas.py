"""Pydantic request/response модели для FastAPI-эндпоинтов.

Соглашения:
- `extra='forbid'` — не даём фронту тихо протаскивать неизвестные ключи.
- Все пути — строки (`str`), не `pathlib.Path`. FastAPI умеет Path, но
  сериализация через JSON и обратно всё равно превратит их в строку —
  проще держать один тип на границе.
- Опции пайплайна намеренно суженное подмножество `PipelineConfig`: только
  то, что пользователь реально трогает через UI. Всё остальное берётся
  из дефолтов конфига.
"""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


# --- Job request bodies ------------------------------------------------------


class ProcessJobRequest(BaseModel):
    """Запрос на полный прогон Фазы 1: stream → transcript → mutes → decisions.

    Обязателен только `stream`. Всё остальное (словарь мата, рабочая папка,
    пути выходных файлов, оригинал реакции) бэкенд подставляет сам — это
    «простой режим» UI: пользователь выбирает одну запись и жмёт «Обработать».
    «Расширенный режим» присылает эти поля явно.
    """

    model_config = ConfigDict(extra="forbid")

    stream: str = Field(..., description="Путь к исходному stream.mp4")
    original: Optional[str] = Field(None, description="Путь к оригинальному видео реакции")
    banwords: Optional[str] = Field(None, description="Путь к словарю мата; None → встроенный")
    workdir: Optional[str] = Field(None, description="Рабочая папка cache/checkpoints; None → авто")
    decisions: Optional[str] = Field(None, description="Куда записать decisions.json; None → в workdir")
    vegas: Optional[str] = Field(None, description="Куда записать Vegas C# script; None → в workdir")

    range_in: Optional[str] = Field(None, description="Timecode начала (HH:MM:SS.mmm)")
    range_out: Optional[str] = Field(None, description="Timecode конца")

    # Опциональные overrides — если не заданы, берутся дефолты PipelineConfig
    # (device/compute_type — из окружения: CPU-режим на машине без NVIDIA).
    # Движок по умолчанию — GigaAM v3 (точнее по русскому мату, не тянет
    # CTranslate2/cuDNN). 'model' относится к WhisperX; для GigaAM — gigaam_model.
    transcriber: str = "gigaam"
    gigaam_model: str = "v3_ctc"
    model: str = "large-v3"
    language: str = "ru"
    device: Optional[str] = None
    compute_type: Optional[str] = None
    batch_size: int = Field(16, ge=1)
    vad_filter: bool = True
    vad_method: str = "pyannote"

    mute_padding_before_ms: int = Field(80, ge=0)
    mute_padding_after_ms: int = Field(120, ge=0)
    mute_extend_mode: str = "word"
    mute_max_seconds: float = Field(6.0, gt=0)
    mute_join_gap_ms: int = Field(600, ge=0)

    force_extract: bool = False
    force_transcribe: bool = False
    force_detect: bool = False

    # Smoke-mode: пропустить extract+transcribe, взять готовый JSON.
    # Используется в тестах И как быстрый способ прогнать пайплайн без CUDA.
    mock_transcript: Optional[str] = None


class ExportVegasRequest(BaseModel):
    """Регенерация .cs скрипта из уже существующего decisions.json."""

    model_config = ConfigDict(extra="forbid")

    decisions: str
    vegas: str


class HighlightsExportRequest(BaseModel):
    """Собрать пакет transcript_*.md + prompt.md + schema.json для NotebookLM."""

    model_config = ConfigDict(extra="forbid")

    decisions: str = Field(..., description="decisions.json — оттуда берём transcript path")
    out_dir: str
    transcript: Optional[str] = None
    n_highlights: int = Field(5, ge=1, le=50)


class HighlightsImportRequest(BaseModel):
    """Провалидировать JSON-ответ NotebookLM и записать в decisions.json."""

    model_config = ConfigDict(extra="forbid")

    decisions: str
    response: str = Field(..., description="Путь к JSON-файлу с ответом NotebookLM")
    output: str = Field(..., description="Куда записать обновлённый decisions.json")
    transcript: Optional[str] = None


# --- Job responses / events --------------------------------------------------


JobStatusLiteral = Literal["pending", "running", "done", "failed", "cancelled"]
JobKindLiteral = Literal[
    "process", "export_vegas", "highlights_export", "highlights_import"
]


class JobResponse(BaseModel):
    """Snapshot состояния джобы для HTTP-эндпоинтов."""

    model_config = ConfigDict(extra="forbid")

    id: str
    kind: JobKindLiteral
    status: JobStatusLiteral
    progress: float = Field(0.0, ge=0.0, le=100.0)
    stage: str = ""
    message: str = ""
    result: Optional[dict[str, Any]] = None
    error: Optional[str] = None
    created_at: str
    finished_at: Optional[str] = None


class JobEvent(BaseModel):
    """Событие, прилетающее в WebSocket подписчикам джобы.

    type='progress' — обычный tick стадии.
    type='final'    — джоба завершилась (done/failed/cancelled); `state` содержит
                      финальный snapshot. После final сокет сервер закрывает.
    """

    model_config = ConfigDict(extra="forbid")

    type: Literal["progress", "final"]
    state: JobResponse


# --- Decisions I/O -----------------------------------------------------------


class DecisionsPayload(BaseModel):
    """Тело для PUT /decisions — просто «сохрани мне вот этот JSON»."""

    model_config = ConfigDict(extra="forbid")

    decisions: dict[str, Any]
