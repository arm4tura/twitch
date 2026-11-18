"""Рендер промптов из шаблонов + разбор ответов классификатора.

Оконный путь: render_topic_prompt / render_classification_prompt подставляют
транскрипт/тему/фрагмент в шаблон (PromptTemplate). Шаблон уже провалидирован в
templates.py (нужные плейсхолдеры на месте, лишних нет), поэтому str.format не
упадёт.

Путь по исходнику (Фаза 3): render_source_outline_prompt (план исходника),
render_source_classification_prompt (реплики стримера + контекст видео),
render_single_pass_prompt (деградация без исходника). Ответ там — не вердикт по
одному окну, а СПИСОК сегментов с границами в MM:SS; разбирает его
parse_segments_response.

parse_topic_response / parse_classification_response / parse_segments_response —
толерантный разбор ответа LLM. Модель просят вернуть строгий формат (ТЕМА: … /
JSON в fence), но LLM бывает болтлива — вытаскиваем нужное регэкспом/из fence,
не роняя пайплайн на лишнем тексте вокруг.

Толерантность НЕ значит «придумать вердикт»: если разобрать не удалось,
parse_classification_response возвращает вердикт с parse_failed=True (третье
состояние), а не «не оффтоп». То же и у списка сегментов: пустой `segments` —
это ответ «оффтопа нет», а неразобранный ответ — `parse_failed=True`, и
классификатор обязан повторить запрос. Разница принципиальная — см.
ClassificationVerdict.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Iterable, Optional, Sequence

from ..timecode import parse_timecode
from .templates import (
    PromptTemplate,
    TemplateKind,
    builtin_template_for_kind,
)

# Тот же приём, что в llm/importer.py: JSON часто в fenced-блоке. Ловим с
# языком и без, DOTALL — переносы внутри объекта допустимы.
_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)
# "ТЕМА: ..." — до конца строки. IGNORECASE на случай "Тема:"/"тема:".
_TOPIC_RE = re.compile(r"ТЕМА\s*:\s*(.+)", re.IGNORECASE)
# «185:23» — модель продолжила минуты за 60 на длинном VOD. parse_timecode
# такое отвергает (minutes >= 60), но смысл однозначен.
_BIG_MINUTES_RE = re.compile(r"^(?P<m>\d{2,}):(?P<s>[0-5]?\d(?:[.,]\d+)?)$")

# Ограничения на текст, который приходит от модели в результат.
MAX_REASON_CHARS = 2000
MAX_QUOTE_CHARS = 500
MAX_OUTLINE_CHARS = 8000



def render_topic_prompt(
    transcript: str,
    *,
    template: Optional[PromptTemplate] = None,
) -> str:
    """Промпт «определи тему» из шаблона (или встроенного дефолта)."""
    tpl = template or builtin_template_for_kind(TemplateKind.TOPIC)
    if tpl.kind is not TemplateKind.TOPIC:
        raise ValueError(f"expected TOPIC template, got {tpl.kind.value}")
    return tpl.text.format(transcript=transcript)


def render_classification_prompt(
    topic: str,
    segment: str,
    *,
    template: Optional[PromptTemplate] = None,
) -> str:
    """Промпт «оффтоп ли фрагмент» из шаблона (или встроенного дефолта)."""
    tpl = template or builtin_template_for_kind(TemplateKind.CLASSIFICATION)
    if tpl.kind is not TemplateKind.CLASSIFICATION:
        raise ValueError(f"expected CLASSIFICATION template, got {tpl.kind.value}")
    return tpl.text.format(topic=topic, segment=segment)


def render_source_outline_prompt(
    transcript: str,
    *,
    template: Optional[PromptTemplate] = None,
) -> str:
    """Промпт «составь план исходника» — ТЕМА + блоки с таймкодами.

    Это ground truth всего пути: тема берётся из транскрипта самого видео, а не
    угадывается по речи стримера (в смеси речи стримера и видео угадывание и
    ломалось).
    """
    tpl = template or builtin_template_for_kind(TemplateKind.SOURCE_OUTLINE)
    if tpl.kind is not TemplateKind.SOURCE_OUTLINE:
        raise ValueError(f"expected SOURCE_OUTLINE template, got {tpl.kind.value}")
    return tpl.text.format(transcript=transcript)


def render_source_classification_prompt(
    outline: str,
    turns: str,
    *,
    template: Optional[PromptTemplate] = None,
) -> str:
    """Промпт «найди оффтоп» по репликам стримера с контекстом видео."""
    tpl = template or builtin_template_for_kind(TemplateKind.SOURCE_CLASSIFICATION)
    if tpl.kind is not TemplateKind.SOURCE_CLASSIFICATION:
        raise ValueError(
            f"expected SOURCE_CLASSIFICATION template, got {tpl.kind.value}"
        )
    return tpl.text.format(outline=outline, turns=turns)


def render_single_pass_prompt(
    transcript: str,
    *,
    template: Optional[PromptTemplate] = None,
) -> str:
    """Промпт «один проход по транскрипту» — деградация без исходника.

    Хуже пути по исходнику (тема угадывается по смеси речи), но лучше окон:
    границы сегментов задаёт модель, а не сетка 45с.
    """
    tpl = template or builtin_template_for_kind(TemplateKind.SINGLE_PASS)
    if tpl.kind is not TemplateKind.SINGLE_PASS:
        raise ValueError(f"expected SINGLE_PASS template, got {tpl.kind.value}")
    return tpl.text.format(transcript=transcript)


# --- Тела промптов ----------------------------------------------------------


def format_timecode(seconds: float, *, with_hours: bool = False) -> str:
    """Секунды → MM:SS (или HH:MM:SS). Точность 1с — как просим у модели."""
    total = max(0, int(seconds))
    if with_hours or total >= 3600:
        h, rem = divmod(total, 3600)
        m, s = divmod(rem, 60)
        return f"{h:02d}:{m:02d}:{s:02d}"
    m, s = divmod(total, 60)
    return f"{m:02d}:{s:02d}"


def format_asr_lines(
    segments: Iterable[dict[str, Any]],
    *,
    with_hours: bool = False,
) -> str:
    """ASR-сегменты → строки `[MM:SS] речь`.

    Гранулярность ASR-сегмента, а не слова на строку: тот же VOD — 37K символов
    против 123K у вертикального столбика слов, и читается как речь. Если у
    сегмента нет готового `text`, склеиваем его слова.
    """
    lines: list[str] = []
    for seg in segments:
        text = str(seg.get("text", "")).strip()
        if not text:
            text = " ".join(
                str(w.get("word", "")).strip()
                for w in (seg.get("words") or [])
                if str(w.get("word", "")).strip()
            ).strip()
        if not text:
            continue
        start = seg.get("start")
        if start is None:
            continue
        lines.append(f"[{format_timecode(float(start), with_hours=with_hours)}] {text}")
    return "\n".join(lines)


def format_turn_lines(
    turns: Sequence[Any],
    contexts: Sequence[str],
    *,
    with_hours: bool = False,
) -> str:
    """Реплики стримера + что играло в видео.

    Каждая реплика — две строки: своя речь и `В ВИДЕО:` (либо текст исходника,
    либо `<ПАУЗА Nс …>`). Пауза приходит уже отрендеренной числом — см. pause.py.
    """
    if len(turns) != len(contexts):
        raise ValueError(f"turns/contexts length mismatch: {len(turns)} != {len(contexts)}")
    lines: list[str] = []
    for turn, ctx in zip(turns, contexts):
        a = format_timecode(turn.start_s, with_hours=with_hours)
        b = format_timecode(turn.end_s, with_hours=with_hours)
        lines.append(f"[{a}–{b}] СТРИМЕР: {turn.text}")
        lines.append(f"    В ВИДЕО: {ctx}")
    return "\n".join(lines)


def parse_topic_response(raw: str) -> str:
    """Вытащить тему из ответа. Fallback: вся строка, если формат не соблюдён.

    Модель просят ответить `ТЕМА: <...>`. Если она это сделала — берём хвост.
    Иначе не роняем пайплайн: обрезаем ответ до первой непустой строки и
    отдаём как есть (лучше приблизительная тема, чем падение).
    """
    m = _TOPIC_RE.search(raw)
    if m:
        topic = m.group(1).strip()
        if topic:
            return topic[:500]
    # Fallback: первая непустая строка.
    for line in raw.splitlines():
        line = line.strip()
        if line:
            return line[:500]
    return "неизвестно"


class ClassificationVerdict:
    """Результат разбора classification-ответа (без Pydantic — внутренний DTO).

    ТРИ состояния, а не два. Раньше их было два (offtopic=True / False), и
    любой неразобранный ответ становился `False/0.0` — то есть обрезанный по
    max_tokens ответ выглядел как честное «фрагмент по теме». На реальном VOD
    так потерялось 13 окон из 30. Теперь неразобранный ответ помечается
    `parse_failed=True`: вердикта НЕТ, и классификатор обязан либо повторить
    запрос, либо записать окно как неклассифицированное.
    """

    __slots__ = ("offtopic", "confidence", "reason", "parse_failed")

    def __init__(
        self,
        offtopic: bool,
        confidence: float,
        reason: str,
        *,
        parse_failed: bool = False,
    ):
        self.offtopic = offtopic
        self.confidence = confidence
        self.reason = reason
        self.parse_failed = parse_failed

    def __repr__(self) -> str:  # pragma: no cover - диагностика
        return (
            f"ClassificationVerdict(offtopic={self.offtopic}, "
            f"confidence={self.confidence}, reason={self.reason!r}, "
            f"parse_failed={self.parse_failed})"
        )


def _no_verdict(reason: str) -> ClassificationVerdict:
    """Ответа нет. offtopic=False здесь — не вердикт, а заполнитель поля:
    смотреть надо на parse_failed."""
    return ClassificationVerdict(False, 0.0, reason, parse_failed=True)


# Строки, которые модель иногда присылает вместо bool. Всё остальное (включая
# "false" — bool("false") в Python это True!) считаем неразобранным.
_TRUE_WORDS = frozenset({"true", "yes", "y", "1", "да", "оффтоп", "offtopic"})
_FALSE_WORDS = frozenset({"false", "no", "n", "0", "нет"})


def _as_bool(value: object) -> Optional[bool]:
    """Привести значение `offtopic` к bool. None = не смогли (нет вердикта)."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        v = value.strip().lower()
        if v in _TRUE_WORDS:
            return True
        if v in _FALSE_WORDS:
            return False
    return None


def parse_classification_response(raw: str) -> ClassificationVerdict:
    """Разобрать JSON-вердикт классификатора. Толерантен к тексту вокруг fence.

    Ожидаем `{"offtopic": bool, "confidence": 0..1, "reason": str}`. Если JSON
    не найден / битый / без поля `offtopic` — возвращаем parse_failed-вердикт.
    Это НЕ «не оффтоп»: consumer обязан различать (см. ClassificationVerdict).
    """
    blob = _extract_json_blob(raw)
    if blob is None:
        return _no_verdict("no JSON verdict in response")

    try:
        data = json.loads(blob)
    except json.JSONDecodeError:
        return _no_verdict("malformed JSON verdict")

    if not isinstance(data, dict):
        return _no_verdict("verdict is not an object")

    offtopic = _as_bool(data.get("offtopic"))
    if offtopic is None:
        # Ключа нет или в нём что-то нечитаемое: модель могла ответить
        # объяснением, планом или чужой схемой. Вердикта из этого не сделать.
        return _no_verdict(f"verdict without usable 'offtopic' field: {blob[:200]}")

    reason = str(data.get("reason", "")).strip()[:2000] or "no reason given"

    conf_raw = data.get("confidence", 0.0)
    try:
        confidence = float(conf_raw)
    except (TypeError, ValueError):
        confidence = 0.0
    # Клампим в [0,1] — модель иногда отдаёт 1.2 / -0.1 / проценты.
    confidence = max(0.0, min(1.0, confidence))

    return ClassificationVerdict(offtopic, confidence, reason)


def _extract_json_blob(raw: str) -> Optional[str]:
    """JSON-объект: сначала fenced-блок, потом первый `{...}` в тексте."""
    stripped = raw.strip()
    m = _FENCE_RE.search(stripped)
    if m:
        return m.group(1)
    if stripped.startswith("{"):
        return stripped
    # Последний шанс: первый сбалансированный-ish объект в тексте.
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start != -1 and end != -1 and end > start:
        return stripped[start : end + 1]
    return None


# --- Путь по исходнику: план и список сегментов ------------------------------


@dataclass(frozen=True)
class SourceOutline:
    """План исходника: полный текст (уходит в промпт классификации) + тема."""

    text: str
    topic: str

    @property
    def empty(self) -> bool:
        return not self.text.strip()


def parse_outline_response(raw: str) -> SourceOutline:
    """Разобрать план исходника.

    Текст плана используется как есть — модель сама решает, сколько блоков и
    как их назвать. Отдельно вытаскиваем строку `ТЕМА:` для поля topic у
    сегментов. Пустой ответ (обрезка по max_tokens) → empty=True, и
    классификатор обязан повторить запрос, а не работать без темы.
    """
    text = raw.strip()[:MAX_OUTLINE_CHARS]
    if not text:
        return SourceOutline(text="", topic="")
    return SourceOutline(text=text, topic=parse_topic_response(text))


def parse_llm_timecode(value: Any) -> Optional[float]:
    """`MM:SS` / `HH:MM:SS` / `185:23` / секунды → секунды. None если не таймкод.

    Модель просят отвечать MM:SS (точность 1с). `185:23` — минуты за 60 на
    длинном VOD: строгий parse_timecode такое отвергает, но смысл однозначен,
    поэтому разбираем сами вместо того, чтобы терять сегмент.
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value) if value >= 0 else None
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    try:
        return parse_timecode(text).to_seconds()
    except (ValueError, ArithmeticError):
        pass
    m = _BIG_MINUTES_RE.match(text)
    if m:
        return int(m.group("m")) * 60 + float(m.group("s").replace(",", "."))
    return None


@dataclass(frozen=True)
class RawSegment:
    """Сегмент, как его назвала модель: границы в секундах, до снапа."""

    start_s: float
    end_s: float
    confidence: float
    reason: str
    quote: str


@dataclass(frozen=True)
class SegmentsAnswer:
    """Ответ со СПИСКОМ сегментов. Те же три состояния, что у вердикта.

    - `parse_failed=False`, segments пуст  → «оффтопа здесь нет» (валидный ответ);
    - `parse_failed=True`                 → ответа НЕТ (обрезка, битый JSON,
      чужая схема). Батч надо повторить, а не молча счесть «всё по теме»;
    - `dropped`                           → сколько элементов выброшено из-за
      нечитаемых границ. Не ошибка ответа, но повод залогировать.
    """

    segments: tuple[RawSegment, ...] = ()
    topic: str = ""
    parse_failed: bool = False
    error: str = ""
    dropped: int = 0
    dropped_reasons: tuple[str, ...] = ()


def _failed_segments(error: str) -> SegmentsAnswer:
    """Ответа нет. Пустой segments здесь — не «нет оффтопа», смотреть на
    parse_failed."""
    return SegmentsAnswer(parse_failed=True, error=error)


def parse_segments_response(raw: str) -> SegmentsAnswer:
    """Разобрать `{"segments": [{"start": "MM:SS", "end": "MM:SS", …}]}`.

    Ключ `segments` ОБЯЗАН быть: пустой список — это ответ «оффтопа нет», а
    отсутствие ключа значит, что модель ответила не тем (или ответ обрезан) —
    такое помечаем parse_failed, чтобы батч повторили.

    Отдельные элементы с нечитаемыми/вывернутыми границами выбрасываются и
    считаются в `dropped`: один сегмент с галлюцинированным таймкодом не должен
    обнулять весь батч.
    """
    blob = _extract_json_blob(raw)
    if blob is None:
        return _failed_segments("no JSON object in response")

    try:
        data = json.loads(blob)
    except json.JSONDecodeError:
        return _failed_segments("malformed JSON")

    if not isinstance(data, dict):
        return _failed_segments("answer is not an object")

    items = data.get("segments")
    if items is None:
        return _failed_segments(f"answer without 'segments' key: {blob[:200]}")
    if not isinstance(items, list):
        return _failed_segments("'segments' is not a list")

    topic = str(data.get("topic", "")).strip()[:500]
    out: list[RawSegment] = []
    dropped: list[str] = []

    for item in items:
        if not isinstance(item, dict):
            dropped.append(f"not an object: {str(item)[:80]}")
            continue
        start = parse_llm_timecode(item.get("start"))
        end = parse_llm_timecode(item.get("end"))
        if start is None or end is None:
            dropped.append(
                f"unreadable timecodes: start={item.get('start')!r} end={item.get('end')!r}"
            )
            continue
        if end <= start:
            dropped.append(f"end <= start: {item.get('start')!r}–{item.get('end')!r}")
            continue

        try:
            confidence = float(item.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0
        confidence = max(0.0, min(1.0, confidence))

        reason = str(item.get("reason", "")).strip()[:MAX_REASON_CHARS] or "no reason given"
        quote = str(item.get("quote", "")).strip()[:MAX_QUOTE_CHARS]

        out.append(
            RawSegment(
                start_s=start,
                end_s=end,
                confidence=confidence,
                reason=reason,
                quote=quote,
            )
        )

    return SegmentsAnswer(
        segments=tuple(out),
        topic=topic,
        dropped=len(dropped),
        dropped_reasons=tuple(dropped[:10]),
    )
