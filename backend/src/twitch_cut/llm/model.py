"""Pydantic-модели highlights и HighlightSet.

Highlights приходят из NotebookLM в JSON. Валидация двухслойная:
1. Pydantic: типы, обязательные поля, диапазоны.
2. Importer поверх: соответствие таймингов границам стрима и осмысленная
   длительность highlight'а.
"""

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class Highlight(BaseModel):
    """Один яркий момент, выбранный LLM."""

    model_config = ConfigDict(extra="forbid")

    # Границы в секундах ОТ НАЧАЛА ТРАНСКРИПТА (то есть в local-времени,
    # не в stream-времени). Согласовано с decisions.mutes[i].start — те тоже
    # в local-времени. При merge в decisions.json пересчёт не нужен.
    start_s: float = Field(ge=0.0)
    end_s: float = Field(gt=0.0)

    # Короткое название момента для UI (типа "стример увидел скример").
    title: str = Field(min_length=1, max_length=200)
    # Развёрнутое объяснение — почему этот момент интересен.
    reason: str = Field(min_length=1, max_length=2000)
    # Уверенность LLM 0..1. Не используем как жёсткий фильтр, но показываем
    # пользователю для ручного триажа.
    score: float = Field(ge=0.0, le=1.0)
    # Опциональная короткая цитата из транскрипта — для быстрой сверки, что
    # LLM говорит о реальном фрагменте, а не галлюцинирует. NotebookLM обычно
    # умеет процитировать точно.
    quote: Optional[str] = Field(default=None, max_length=500)

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


class HighlightSet(BaseModel):
    """Набор highlights + метаданные о том, откуда они пришли."""

    model_config = ConfigDict(extra="forbid")

    highlights: list[Highlight] = Field(min_length=1, max_length=50)
    # 'notebooklm' | 'mock' — источник (для audit trail в decisions.json).
    source: str = Field(min_length=1)
    # Хеш транскрипта, из которого сгенерирован пакет для LLM. Если пользователь
    # перегенерирует транскрипт (например, сменил модель) — highlights стухают,
    # можно детектировать по несовпадению хеша.
    transcript_hash: Optional[str] = None
    # Модель / версия / что успели поймать. NotebookLM UI не отдаёт номер
    # модели явно, поэтому чаще всего "notebooklm" без версии.
    model: Optional[str] = None
