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

from pydantic import BaseModel, ConfigDict, Field, field_validator

from ..config import SENSITIVITY_DEFAULT, SENSITIVITY_LEVELS


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

    # Чувствительность детекции — одна ручка на числа ниже (padding, склейка,
    # поджим по RMS, расширение окна на тишине, допуск голове зайти на соседа,
    # порог спотинга); см. config.SENSITIVITY_LEVELS. Сами эти поля — Optional:
    # None означает
    # «взять из уровня», а не «оставить как было». Числа здесь НЕ дублируются
    # намеренно: раньше те же 80/120/600 лежали и тут, и в CLI, и в
    # PipelineConfig, и разъезжались.
    sensitivity: str = SENSITIVITY_DEFAULT

    mute_padding_before_ms: Optional[int] = Field(None, ge=0)
    mute_padding_after_ms: Optional[int] = Field(None, ge=0)
    mute_extend_mode: str = "word"
    mute_max_seconds: float = Field(6.0, gt=0)
    mute_join_gap_ms: Optional[int] = Field(None, ge=0)
    # Потолок длительности ОДНОГО слова: защита от битого word-timing, когда
    # ASR ставит слову end = конец сегмента. Раньше не пробрасывался из API
    # вообще (жёстко 1.5 с в PipelineConfig). Уровнем чувствительности НЕ
    # управляется — это предохранитель, а не выбор «шире/уже».
    mute_max_word_seconds: float = Field(1.5, gt=0)
    # Подгонка границ мьюта по энергии аудио (см. mute_refine). Выключение
    # возвращает старое поведение «границы только из ASR».
    mute_trim_to_audio: bool = True
    mute_trim_peak_ratio: Optional[float] = Field(None, gt=0.0, lt=1.0)
    mute_trim_guard_ms: Optional[int] = Field(None, ge=0)
    mute_trim_min_ms: int = Field(150, ge=0)
    # Расширение назад окна, стоящего на тишине (см.
    # config.mute_expand_on_silence_ms): единственная операция, которая делает
    # окно ШИРЕ. 0 — выключить.
    mute_expand_on_silence_ms: Optional[int] = Field(None, ge=0)
    # Допуск головному отступу зайти на разметку предыдущего слова (см.
    # config.mute_head_cross_prev_ms). 0 — запрет, как было: в связной речи
    # клэмп по соседу схлопывает mute_padding_before_ms в ноль.
    mute_head_cross_prev_ms: Optional[int] = Field(None, ge=0)

    # Поиск мата в посториорах CTC (только gigaam) — находки идут как
    # review-кандидаты. Раньше из API не пробрасывались вообще.
    spot_posteriors: bool = True
    spot_min_score: Optional[float] = Field(None, gt=0.0, le=1.0)

    force_extract: bool = False
    force_transcribe: bool = False
    force_detect: bool = False

    # Smoke-mode: пропустить extract+transcribe, взять готовый JSON.
    # Используется в тестах И как быстрый способ прогнать пайплайн без CUDA.
    mock_transcript: Optional[str] = None

    # Имя уровня проверяем на границе API, а не внутри джобы: иначе опечатка в
    # запросе превращается в упавшую джобу (и человек видит её в списке как
    # ошибку обработки), вместо 422 со списком допустимых значений.
    @field_validator("sensitivity")
    @classmethod
    def _known_sensitivity(cls, value: str) -> str:
        if value not in SENSITIVITY_LEVELS:
            known = ", ".join(SENSITIVITY_LEVELS)
            raise ValueError(f"sensitivity must be one of: {known}")
        return value


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
    segments_to_mute: list[tuple[float, float]] = Field(
        default_factory=list,
        description="[(start, end), ...] в секундах — заглушить аудио в ноль, кадры оставить (маты)",
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


class DetectOfftopicRequest(BaseModel):
    """AI-детекция оффтоп-сегментов: транскрипт → decisions['offtopic'].

    `api_key` — ТРАНЗИТНЫЙ секрет: Electron main расшифровывает его из
    safeStorage и кладёт в тело запроса на loopback. Бэкенд использует ключ
    на время джобы и НИКОГДА не пишет его на диск / в лог / в result. Здесь он
    просто поле запроса — не сохраняется в settings.json.

    `provider`/`topic_template`/`classification_template` — полные dict'ы
    кастомных клонов (валидируются в runner через AIProvider/PromptTemplate).
    Если не заданы — берётся встроенный пресет по `provider_id` / дефолтные
    шаблоны. Поля тюнинга (window_s и т.п.) None → классификатор берёт свои
    дефолты (см. llm_offtopic.classifier).

    Путь классификации выбирает `method` (см. llm_offtopic.classifier):
      None            — авто: есть исходник → source, иначе → windows (не меняем
                        поведение старых вызовов, у которых источника нет);
      "source"        — по исходнику (нужен source_audio ИЛИ source_transcript);
      "single-pass"   — один проход без исходника;
      "windows"       — старый оконный путь.
    `source_audio` — локальный файл исходного видео (URL не поддерживается,
    см. source.resolve_source_path); его путь эндпоинт заносит в whitelist
    (`store.allow_media`), как вход остальных джоб. `source_transcript` —
    готовый JSON транскрипта исходника, если он уже посчитан.
    """

    model_config = ConfigDict(extra="forbid")

    decisions: str
    output: str = Field(..., description="Куда записать decisions.json с ключом 'offtopic'")
    transcript: Optional[str] = None

    provider_id: str = Field("builtin:anthropic", description="ID провайдера (builtin:* или кастомного)")
    provider: Optional[dict[str, Any]] = Field(
        None, description="Полный dict кастомного AIProvider (если provider_id не встроенный)"
    )
    api_key: Optional[str] = Field(
        None, description="ТРАНЗИТНЫЙ API-ключ из safeStorage. Не персистится."
    )

    cheap_model: Optional[str] = Field(None, description="[windows] Модель 1-й ступени; None → дефолт провайдера")
    expensive_model: Optional[str] = Field(None, description="[windows] Модель 2-й ступени каскада; None → без каскада")
    # LLM-модель пути по исходнику / одного прохода (у windows своя пара cheap/expensive).
    model: Optional[str] = Field(None, description="[source/single-pass] Модель LLM; None → дефолт провайдера")

    # Путь классификации и исходник (Фаза 4). Все опциональны — без них джоба
    # ведёт себя как раньше (оконный путь).
    method: Optional[str] = Field(
        None, description="None|source|single-pass|windows (см. docstring)"
    )
    source_audio: Optional[str] = Field(
        None, description="Локальный файл исходника — распознаём и используем как ground truth"
    )
    source_transcript: Optional[str] = Field(
        None, description="Готовый JSON транскрипта исходника (вместо распознавания source_audio)"
    )
    source_model: Optional[str] = Field(
        None, description="Модель ASR исходника; должна совпадать с моделью стрима. None → дефолт"
    )
    source_device: Optional[str] = Field(
        None, description="Устройство ASR исходника (cpu безопаснее). None → дефолт"
    )

    topic_template: Optional[dict[str, Any]] = Field(None, description="Кастомный шаблон определения темы")
    classification_template: Optional[dict[str, Any]] = Field(
        None, description="Кастомный шаблон классификации оффтопа"
    )

    # Тюнинг пайплайна. None → дефолт классификатора (не переопределяем).
    window_s: Optional[float] = Field(None, gt=0, description="Длина окна классификации (сек)")
    overlap_s: Optional[float] = Field(None, ge=0, description="Перекрытие окон (сек)")
    chunk_s: Optional[float] = Field(None, gt=0, description="Размер чанка пере-извлечения темы (сек)")
    merge_gap_s: Optional[float] = Field(None, ge=0, description="Склеивать оффтоп-окна с зазором ≤ N сек")
    cascade_threshold: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Ниже этого confidence — дозапрос дорогой модели"
    )


class TestConnectionRequest(BaseModel):
    """Проверка доступности провайдера из UI настроек.

    Как и в DetectOfftopicRequest, `api_key` — транзитный секрет, не персистится.
    """

    model_config = ConfigDict(extra="forbid")

    provider_id: str = Field("builtin:anthropic", description="ID провайдера (builtin:* или кастомного)")
    provider: Optional[dict[str, Any]] = Field(None, description="Полный dict кастомного AIProvider")
    api_key: Optional[str] = Field(None, description="ТРАНЗИТНЫЙ API-ключ. Не персистится.")


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
    "detect_offtopic",
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
