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


class DownloadVodRequest(BaseModel):
    """Скачивание Twitch VOD через bundled TwitchDownloaderCLI (subprocess).

    Обязателен только `url` — полный URL VOD или числовой ID (бэкенд извлекает
    ID сам). Всё остальное опционально: качество (None → источник/максимум),
    обрезка по таймкодам (None → целиком), OAuth-токен (для sub-only VOD),
    рабочая папка (None → бэкенд создаёт подпапку в work/).
    """

    model_config = ConfigDict(extra="forbid")

    url: str = Field(..., description="Twitch VOD URL или числовой ID")
    quality: Optional[str] = Field(None, description="Значение качества из VodInfo; None → источник")
    range_in: Optional[str] = Field(None, description="Timecode начала обрезки (HH:MM:SS); None → с начала")
    range_out: Optional[str] = Field(None, description="Timecode конца обрезки (HH:MM:SS); None → до конца")
    oauth: Optional[str] = Field(None, description="OAuth-токен для sub-only VOD; None если публичный")
    workdir: Optional[str] = Field(None, description="Куда качать; None → авто-подпапка в work/")


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
    response: str = Field(
        "",
        description="Путь к JSON-файлу с ответом NotebookLM (альтернатива response_text)",
    )
    response_text: Optional[str] = Field(
        None,
        description="Сырой текст ответа (вставленный из NotebookLM). Приоритетнее response.",
    )
    output: str = Field(..., description="Куда записать обновлённый decisions.json")
    transcript: Optional[str] = None


class DetectSilenceRequest(BaseModel):
    """Детект тишины через ffmpeg silencedetect."""

    model_config = ConfigDict(extra="forbid")

    video_path: str = Field(..., description="Путь к видео/аудио")
    threshold_db: float = Field(-30.0, description="Порог RMS в dBFS")
    min_duration: float = Field(0.5, ge=0.1, description="Минимальная длительность сегмента (сек)")
    merge_gap: float = Field(0.5, ge=0.0, description="Склеивать сегменты с зазором ≤ N сек")
    padding_before: float = Field(0.1, ge=0.0, description="Padding до начала (сек)")
    padding_after: float = Field(0.1, ge=0.0, description="Padding после конца (сек)")
    workdir: Optional[str] = Field(None, description="Рабочая папка для кэша")
    force: bool = Field(False, description="Пропустить кэш")


class RenderRequest(BaseModel):
    """Рендер видео с вырезанными сегментами."""

    model_config = ConfigDict(extra="forbid")

    input_path: str = Field(..., description="Путь к исходному видео")
    output_path: str = Field(..., description="Куда записать результат")
    segments_to_remove: list[tuple[float, float]] = Field(
        ..., description="[(start, end), ...] в секундах — что удалить"
    )
    encoder: Optional[str] = Field(None, description="H.264-энкодер; None → авто")
    audio_bitrate: str = Field("192k", description="Битрейт аудио")
    quality: Literal["fast", "balanced", "max"] = Field(
        "balanced", description="Пресет скорость↔качество энкодера"
    )
    range_start: Optional[float] = Field(
        None, description="Обрезать вывод слева (абс. секунды исходника); None → с начала"
    )
    range_end: Optional[float] = Field(
        None, description="Обрезать вывод справа (абс. секунды исходника); None → до конца"
    )
    force: bool = Field(False, description="Перезаписать output если существует")


class RenderHighlightsRequest(BaseModel):
    """Нарезать highlights из decisions.json в отдельные mp4-клипы."""

    model_config = ConfigDict(extra="forbid")

    decisions: str = Field(..., description="decisions.json с highlights (bundle)")
    input_path: str = Field(..., description="Исходное видео (абс. секунды stream'а)")
    out_dir: str = Field(..., description="Каталог для клипов (создаётся)")
    quality: Literal["fast", "balanced", "max"] = Field(
        "balanced", description="Пресет скорость↔качество энкодера"
    )
    padding_s: float = Field(
        0.0, ge=0.0, le=10.0, description="Доп. секунды до/после каждого клипа"
    )
    force: bool = Field(False, description="Перезаписать существующие клипы")


class ExportVegasHighlightsRequest(BaseModel):
    """Собрать Vegas C#-скрипт с Region'ами на highlights."""

    model_config = ConfigDict(extra="forbid")

    decisions: str
    vegas: str = Field(..., description="Куда записать .cs скрипт маркеров")


# --- Job responses / events --------------------------------------------------


JobStatusLiteral = Literal["pending", "running", "done", "failed", "cancelled"]
JobKindLiteral = Literal[
    "process",
    "export_vegas",
    "highlights_export",
    "highlights_import",
    "render_highlights",
    "export_vegas_highlights",
    "detect_silence",
    "render",
    "download_vod",
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
