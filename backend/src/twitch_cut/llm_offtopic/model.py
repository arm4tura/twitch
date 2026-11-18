"""Pydantic-модели оффтоп-сегментов и OfftopicSet.

Зеркалит llm/model.py (Highlight/HighlightSet) — те же конвенции:
- границы в LOCAL-времени (секунды от начала транскрипта), как decisions.mutes;
- extra="forbid", валидатор end>start, свойство duration_s;
- при merge в decisions.json пересчёт в stream-время делает merger.py.

Отличие от Highlight: оффтоп-сегмент несёт `confidence` (уверенность
классификатора) и опциональный `model`/`stage` для audit trail каскада
(дешёвая модель → дорогая при низкой уверенности).
"""

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class OfftopicSegment(BaseModel):
    """Один фрагмент, помеченный классификатором как оффтоп."""

    model_config = ConfigDict(extra="forbid")

    # Границы в секундах ОТ НАЧАЛА ТРАНСКРИПТА (local-время), как mutes[i].start.
    start_s: float = Field(ge=0.0)
    end_s: float = Field(gt=0.0)

    # Уверенность классификатора 0..1. В отличие от highlights.score — здесь это
    # рабочий сигнал: каскад дозапрашивает дорогую модель при confidence ниже
    # порога, а UI может фильтровать по нему.
    confidence: float = Field(ge=0.0, le=1.0)
    # Объяснение — почему фрагмент оффтоп (для ручного триажа в UI).
    reason: str = Field(min_length=1, max_length=2000)

    # Тема, относительно которой классифицировали (для контекста в UI). Может
    # отличаться между чанками — тему пере-извлекаем на каждый чанк.
    topic: Optional[str] = Field(default=None, max_length=500)
    # Короткая цитата из фрагмента — для быстрой сверки, что не галлюцинация.
    quote: Optional[str] = Field(default=None, max_length=500)

    # Audit: какая модель вынесла финальный вердикт и на какой ступени каскада
    # ("cheap" | "expensive"). None для не-каскадного прогона.
    model: Optional[str] = Field(default=None, max_length=120)
    stage: Optional[str] = Field(default=None, max_length=40)

    # --- Улики пути по исходнику (classify_with_source) ---------------------
    # Заполняются ТОЛЬКО когда сегмент найден по исходнику: у оконного и
    # одно-проходного путей привязки нет, там оба None.
    #
    # pause_s — сколько секунд видео МОЛЧАЛО во время сегмента (0.0 = играло
    # всю реплику). Длина паузы и есть сила улики: минута тишины видео при
    # говорящем стримере — почти всегда уход в свои дела (см. pause.py).
    pause_s: Optional[float] = Field(default=None, ge=0.0)
    # anchored — доля слов сегмента, привязанных к исходнику (0..1). Независимая
    # от LLM улика: внутри оффтопа она низкая (на замеренном VOD 0.117 против
    # 0.667 вне). Это кандидат, а не вердикт — стример мог говорить поверх
    # играющего видео (низкая привязка, но по теме), поэтому решение за моделью.
    anchored: Optional[float] = Field(default=None, ge=0.0, le=1.0)

    @field_validator("end_s")
    @classmethod
    def _end_after_start(cls, v: float, info) -> float:
        start = info.data.get("start_s")
        if start is not None and v <= start:
            raise ValueError(f"end_s ({v}) must be > start_s ({start})")
        return v

    @property
    def duration_s(self) -> float:
        return round(self.end_s - self.start_s, 3)


class OfftopicSet(BaseModel):
    """Набор оффтоп-сегментов + метаданные о прогоне."""

    model_config = ConfigDict(extra="forbid")

    # Пустой список валиден: «оффтопа не найдено» — легитимный результат
    # (в отличие от highlights, где min_length=1). Верх — 1000: транскрипт
    # многочасового стрима может дать много коротких оффтоп-кусков.
    segments: list[OfftopicSegment] = Field(default_factory=list, max_length=1000)

    # Откуда пришло: id провайдера/модели или 'mock' (audit trail).
    source: str = Field(min_length=1)
    # Хеш транскрипта — если пользователь перегенерил транскрипт, сегменты
    # стухают (детектируется по несовпадению хеша), как у HighlightSet.
    transcript_hash: Optional[str] = None
    # Модель, которой прогоняли (для UI/лога). None если mock.
    model: Optional[str] = None

    # Каким путём получен результат — нужно UI, чтобы объяснить вердикты:
    #   "windows"        — старый оконный путь (судит СМЕСЬ речи, потолок точности);
    #   "single_pass"    — один проход без исходника (границы от модели);
    #   "source_anchored"— по исходнику (расщепление дорожек + план видео + пауза).
    # Дефолт "windows" — обратная совместимость: у сериализованных ДО Фазы 4
    # OfftopicSet поля method нет, и они должны читаться как оконный прогон.
    method: Literal["windows", "single_pass", "source_anchored"] = "windows"
    # Хеш транскрипта ИСХОДНИКА (путь по исходнику) — тем же образом стухает,
    # если исходник перераспознали. None вне пути по исходнику.
    source_transcript_hash: Optional[str] = None
    # Модель ASR, которой распознан исходник (должна совпадать с моделью стрима).
    # Для audit/воспроизводимости; None вне пути по исходнику.
    source_model: Optional[str] = None

    @property
    def total_offtopic_s(self) -> float:
        """Суммарная длительность оффтопа в секундах."""
        return round(sum(s.duration_s for s in self.segments), 3)
