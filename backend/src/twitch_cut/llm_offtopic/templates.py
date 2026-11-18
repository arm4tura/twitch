"""Шаблоны промптов для оффтоп-детекции: редактируемый текст + валидация плейсхолдеров.

Виды промптов по двум путям классификации:

Оконный путь (старый, работает без исходника):

- ``topic``           — «о чём этот стрим/чанк?» → короткое описание темы.
                        Плейсхолдеры: {transcript}.
- ``classification``  — «этот фрагмент оффтоп относительно темы?» → вердикт.
                        Плейсхолдеры: {topic}, {segment}.

Путь по исходнику (Фаза 3): тема не угадывается по бормотанию стримера, а
берётся из транскрипта самого видео, и модель сама расставляет границы:

- ``source_outline``        — план исходника (ТЕМА + блоки с таймкодами) =
                              ground truth. Плейсхолдеры: {transcript}.
- ``source_classification`` — реплики стримера с контекстом видео → список
                              оффтоп-сегментов. Плейсхолдеры: {outline}, {turns}.
- ``single_pass``           — деградация без исходника: один проход по
                              ASR-сегментам, модель сама задаёт границы.
                              Плейсхолдеры: {transcript}.

Шаблон хранит СЫРОЙ текст с плейсхолдерами `{name}`. Рендер — обычный
str.format с проверенным набором ключей (см. prompt.py). Валидация здесь
следит, чтобы пользователь не сломал шаблон: обязательные плейсхолдеры на
месте, лишних `{...}` нет (иначе .format упадёт KeyError в рантайме).

BUILTIN_TEMPLATES — read-only (`builtin:*`). UI клонирует их в редактируемые
копии (built-in→clone, как у providers.py). Поэтому новые виды добавляются
НОВЫМИ шаблонами: старые `builtin:topic-ru` / `builtin:classification-ru`
трогать нельзя — на них могут ссылаться уже сделанные клоны.
"""

from __future__ import annotations

import re
import string
from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class TemplateKind(str, Enum):
    """Роль шаблона в пайплайне."""

    TOPIC = "topic"
    CLASSIFICATION = "classification"
    # Путь по исходнику (Фаза 3).
    SOURCE_OUTLINE = "source_outline"
    SOURCE_CLASSIFICATION = "source_classification"
    SINGLE_PASS = "single_pass"


# Обязательные плейсхолдеры по видам. Рендер подставляет ровно эти ключи;
# отсутствие любого → шаблон бесполезен (модель не увидит транскрипт/тему).
TOPIC_PLACEHOLDERS: frozenset[str] = frozenset({"transcript"})
CLASSIFICATION_PLACEHOLDERS: frozenset[str] = frozenset({"topic", "segment"})
SOURCE_OUTLINE_PLACEHOLDERS: frozenset[str] = frozenset({"transcript"})
SOURCE_CLASSIFICATION_PLACEHOLDERS: frozenset[str] = frozenset({"outline", "turns"})
SINGLE_PASS_PLACEHOLDERS: frozenset[str] = frozenset({"transcript"})

_REQUIRED_BY_KIND: dict[TemplateKind, frozenset[str]] = {
    TemplateKind.TOPIC: TOPIC_PLACEHOLDERS,
    TemplateKind.CLASSIFICATION: CLASSIFICATION_PLACEHOLDERS,
    TemplateKind.SOURCE_OUTLINE: SOURCE_OUTLINE_PLACEHOLDERS,
    TemplateKind.SOURCE_CLASSIFICATION: SOURCE_CLASSIFICATION_PLACEHOLDERS,
    TemplateKind.SINGLE_PASS: SINGLE_PASS_PLACEHOLDERS,
}


def _extract_placeholders(text: str) -> set[str]:
    """Достать имена плейсхолдеров `{name}` из шаблона.

    Используем string.Formatter — он разбирает ровно так же, как .format в
    рантайме, поэтому «что распарсили = что подставится». `{{` / `}}`
    (экранированные скобки) корректно игнорируются как литералы.
    """
    names: set[str] = set()
    for _literal, field_name, _spec, _conv in string.Formatter().parse(text):
        if field_name is None:
            continue
        # `{obj.attr}` / `{obj[0]}` — берём корневое имя до первого . или [.
        root = re.split(r"[.\[]", field_name, maxsplit=1)[0].strip()
        if root:
            names.add(root)
    return names


class PromptTemplate(BaseModel):
    """Редактируемый шаблон промпта одного вида (см. TemplateKind)."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1, max_length=120)
    label: str = Field(min_length=1, max_length=120)
    kind: TemplateKind
    # Сырой текст с плейсхолдерами `{name}`.
    text: str = Field(min_length=1, max_length=20_000)
    builtin: bool = False

    @field_validator("text")
    @classmethod
    def _text_parseable(cls, v: str) -> str:
        # Ловим синтаксически битый шаблон СРАЗУ (незакрытая `{`, и т.п.),
        # иначе .format упадёт лишь в рантайме на реальном транскрипте.
        try:
            list(string.Formatter().parse(v))
        except ValueError as e:
            raise ValueError(f"malformed template (bad {{...}}): {e}") from e
        return v

    @model_validator(mode="after")
    def _placeholders_match_kind(self) -> "PromptTemplate":
        found = _extract_placeholders(self.text)
        required = _REQUIRED_BY_KIND[self.kind]

        missing = required - found
        if missing:
            raise ValueError(
                f"template kind '{self.kind.value}' requires placeholders "
                f"{sorted(required)}; missing {sorted(missing)}"
            )

        extra = found - required
        if extra:
            # Лишний плейсхолдер = KeyError при .format. Запрещаем — лучше
            # упасть на сохранении шаблона, чем на обработке реального стрима.
            raise ValueError(
                f"template kind '{self.kind.value}' allows only placeholders "
                f"{sorted(required)}; unknown {sorted(extra)}"
            )
        return self

    def placeholders(self) -> set[str]:
        return _extract_placeholders(self.text)


# --- Встроенные шаблоны -----------------------------------------------------
#
# Текст на русском — целевой контент русскоязычные реакшн-стримы. Промпты
# просят модель вернуть СТРОГО заданный формат; парсинг вердикта — на нашей
# стороне (classifier.py), не полагаемся на «модель поняла».

_BUILTIN_TOPIC_TEXT = """\
Ты анализируешь транскрипт отрезка стрима (реакция на видео/игру/контент).
Ниже — слова с таймингами, одна строка = одно слово: `[HH:MM:SS.mmm] слово`.

Определи ОСНОВНУЮ ТЕМУ этого отрезка одной-двумя фразами: на что реагирует
стример, что за контент в фокусе. Не пересказывай сюжет — назови предмет.

Ответь СТРОГО одной строкой в формате:
ТЕМА: <короткое описание темы>

Транскрипт:
{transcript}
"""

_BUILTIN_CLASSIFICATION_TEXT = """\
Основная тема стрима: {topic}

Ниже — фрагмент транскрипта (слова с таймингами). Определи, является ли этот
фрагмент ОФФТОПОМ — то есть отходом от основной темы. Оффтоп это:

- личные отступления, не связанные с контентом (бытовуха, планы на день);
- обсуждение доната/технических моментов стрима (звук, чат, донат-алерты);
- реклама, самопиар, обсуждение других стримеров вне контекста;
- длинные паузы-болтовня ни о чём.

НЕ оффтоп: реакция на контент по теме, разбор, эмоции по поводу происходящего
в видео/игре — даже если стример отвлекается на секунду.

Ответь СТРОГО JSON-объектом в одном code-fence ```json:

```json
{{
  "offtopic": true,
  "confidence": 0.0,
  "reason": "краткое объяснение (1-2 предложения)"
}}
```

Где `offtopic` — true/false, `confidence` — 0..1 (уверенность), `reason` —
почему. Никакого текста вне code-fence.

Фрагмент:
{segment}
"""


# --- Путь по исходнику ------------------------------------------------------
#
# Формулировки взяты из прототипа, который на реальном VOD дал 12.1 мин оффтопа
# против 1.4 мин у оконного пути. Правки текста ниже — это правки поведения
# детектора, а не косметика: два правила в _BUILTIN_SOURCE_CLASSIFICATION_TEXT
# помечены как обязательные, без них схема ломается известным образом.

_BUILTIN_SOURCE_OUTLINE_TEXT = """\
Ниже транскрипт видео, на которое реагирует стример. Составь краткий план: о чём
видео и какие в нём смысловые блоки. 5–10 пунктов, каждый одной строкой в виде
`- <таймкод начала> <о чём блок>`. Перед планом одной строкой: `ТЕМА: <суть видео>`.

Транскрипт:
{transcript}
"""

_BUILTIN_SOURCE_CLASSIFICATION_TEXT = """\
{outline}

Выше — план видео, на которое реагирует стример. Ниже — только СОБСТВЕННАЯ речь
стримера (речь самого видео уже вычтена), с указанием, что играло в видео в этот
момент. `<ПАУЗА>` означает, что видео в этот момент не воспроизводилось.

Найди ОФФТОП — реплики, где стример ушёл от видео: личные/бытовые темы, техника
стрима (звук, чат, донаты), реклама и самопиар, посторонние разговоры.

НЕ оффтоп: реакция, комментарий, скепсис, эмоции ПО ПОВОДУ того, что сейчас в
кадре, — включая короткую личную аналогию, если стример тут же возвращается.

ВАЖНО про совпадение темы. Само по себе совпадение по ключевому слову с видео
НЕ делает реплику по теме. Если стример перестал смотреть и занялся своим делом
(пишет ИИ-ассистенту, ковыряет свой телефон, решает свою задачу) — это оффтоп,
даже если предмет тот же, что в видео. Критерий: комментирует ли он КАДР или
делает своё.

`<ПАУЗА Nс>` означает, что во время реплики видео не воспроизводилось. Чем
длиннее пауза, тем сильнее признак оффтопа: минута и больше тишины видео при
говорящем стримере — это почти всегда уход в свои дела. Но короткая пауза сама
по себе не доказательство: стример может ставить паузу и обсуждать кадр.

Ответь СТРОГО одним JSON в code-fence:

```json
{{"segments": [{{"start": "MM:SS", "end": "MM:SS", "confidence": 0.0,
  "reason": "почему оффтоп", "quote": "цитата"}}]}}
```

Пустой `segments` — валидный ответ, если оффтопа нет. Никакого текста вне fence.

Реплики:
{turns}
"""

_BUILTIN_SINGLE_PASS_TEXT = """\
Ниже — транскрипт стрима (реакция на видео/игру). Каждая строка: `[MM:SS] речь`.

ШАГ 1. Определи основную тему стрима — на что реагирует стример.

ШАГ 2. Найди все ОФФТОП-фрагменты — куски, где стример уходит от основной темы:
- личные/бытовые отступления (здоровье, планы, еда, усталость, свои покупки);
- техника стрима: звук, чат, донат-алерты, задержка, «слышно меня?»;
- реклама, самопиар, обсуждение других стримеров вне контекста;
- болтовня ни о чём, не связанная с контентом.

НЕ оффтоп: реакция на контент, разбор, эмоции, скепсис по поводу видео —
даже если стример на секунду отвлекается.

ГРАНИЦЫ ВАЖНЫ: start/end — реальные границы отступления по таймкодам строк,
а не «весь блок, где оно встретилось». Короткая реплика (< 8 сек) в потоке
по теме — НЕ отдельный сегмент, пропускай её.

Ответь СТРОГО одним JSON в code-fence:

```json
{{
  "topic": "основная тема одной фразой",
  "segments": [
    {{"start": "MM:SS", "end": "MM:SS", "confidence": 0.0,
      "reason": "почему оффтоп (1 предложение)",
      "quote": "короткая цитата из фрагмента"}}
  ]
}}
```

Пустой `segments` — валидный ответ, если оффтопа нет. Никакого текста вне fence.

Транскрипт:
{transcript}
"""


BUILTIN_TEMPLATES: list[PromptTemplate] = [
    PromptTemplate(
        id="builtin:topic-ru",
        label="Определение темы (RU)",
        kind=TemplateKind.TOPIC,
        text=_BUILTIN_TOPIC_TEXT,
        builtin=True,
    ),
    PromptTemplate(
        id="builtin:classification-ru",
        label="Классификация оффтопа (RU)",
        kind=TemplateKind.CLASSIFICATION,
        text=_BUILTIN_CLASSIFICATION_TEXT,
        builtin=True,
    ),
    PromptTemplate(
        id="builtin:source-outline-ru",
        label="План исходника (RU)",
        kind=TemplateKind.SOURCE_OUTLINE,
        text=_BUILTIN_SOURCE_OUTLINE_TEXT,
        builtin=True,
    ),
    PromptTemplate(
        id="builtin:source-classification-ru",
        label="Оффтоп по исходнику (RU)",
        kind=TemplateKind.SOURCE_CLASSIFICATION,
        text=_BUILTIN_SOURCE_CLASSIFICATION_TEXT,
        builtin=True,
    ),
    PromptTemplate(
        id="builtin:single-pass-ru",
        label="Оффтоп одним проходом, без исходника (RU)",
        kind=TemplateKind.SINGLE_PASS,
        text=_BUILTIN_SINGLE_PASS_TEXT,
        builtin=True,
    ),
]


def builtin_template(template_id: str) -> Optional[PromptTemplate]:
    """Найти встроенный шаблон по id. None если не встроенный."""
    for t in BUILTIN_TEMPLATES:
        if t.id == template_id:
            return t
    return None


def builtin_template_for_kind(kind: TemplateKind) -> PromptTemplate:
    """Дефолтный встроенный шаблон нужного вида (для fallback в пайплайне)."""
    for t in BUILTIN_TEMPLATES:
        if t.kind == kind:
            return t
    raise KeyError(f"no builtin template for kind {kind!r}")
