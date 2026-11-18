"""Классификатор оффтопа: транскрипт → OfftopicSet.

ДВА пути, и выбирает между ними наличие исходника (см. run_* в server/runner
и detect-offtopic в CLI):

`classify_with_source` — путь по исходнику (основной). Транскрипт стрима это
СМЕСЬ речи видео и речи стримера; оконный путь ниже судит эту смесь и поэтому
принимает пересказ видео за «по теме». Здесь смесь расщепляется: тема берётся
из транскрипта самого видео (план = ground truth), на вход модели идут только
реплики стримера, к каждой приложено, что играло в видео в этот момент, а
границы сегментов задаёт модель, а не сетка 45с.

`classify_single_pass` — деградация, когда исходника нет: один проход по
ASR-сегментам, границы всё ещё за моделью. Хуже пути по исходнику (тема
угадывается по смеси), но на реальном VOD дал 10 сегментов против 2 у окон.

`classify_transcript` — старый оконный путь. Оставлен как есть: на него
ссылаются существующие джобы и настройки, и он единственный не требует ни
исходника, ни длинного контекста.

Оконный пайплайн (всё синхронно — зовётся из asyncio.to_thread на уровне джобы):

  1. build_word_lines(transcript) → плоский поток `[ts] слово` (llm/segments).
  2. Режем на ОКНА по времени (window_s) с перекрытием (overlap_s). Перекрытие
     решает проблему границ: фраза на стыке окон не теряется.
  3. Для каждого чанка (группа окон / крупный отрезок) пере-извлекаем ТЕМУ —
     тема стрима дрейфует, единая тема на весь VOD даёт мусор.
  4. Каждое окно классифицируем относительно темы чанка. Ответ без вердикта
     (обрезан по max_tokens / битый JSON) — НЕ «по теме»: один ретрай с
     увеличенным бюджетом, потом окно считается неклассифицированным и попадает
     в лог. Молча зачесть его как «чисто» нельзя — так терялось 13 окон из 30.
  5. Каскад: при confidence < cascade_threshold дозапрашиваем дорогую модель
     (если задана) — дёшево отсеиваем очевидное, дорого проверяем спорное.
  6. Соседние оффтоп-окна сливаем в один OfftopicSegment (dedup + smoothing):
     перекрытие/зазор ≤ merge_gap_s → один сегмент, confidence = максимум.

Всё, что режем/ограничиваем (короткие окна, пустые чанки, окна без вердикта,
батчи без ответа, выдуманные таймкоды), логируется через переданный logger —
«тихих» усечений нет.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional, Sequence

from ..llm.segments import build_word_lines
from .align import Alignment, align_transcripts
from .client import LLMClient
from .model import OfftopicSegment, OfftopicSet
from .pause import PAUSE_RATE_PER_S, turn_context, video_context
from .prompt import (
    ClassificationVerdict,
    SegmentsAnswer,
    SourceOutline,
    format_asr_lines,
    format_timecode,
    format_turn_lines,
    parse_classification_response,
    parse_outline_response,
    parse_segments_response,
    parse_topic_response,
    render_classification_prompt,
    render_single_pass_prompt,
    render_source_classification_prompt,
    render_source_outline_prompt,
    render_topic_prompt,
)
from .templates import PromptTemplate
from .turns import (
    DEFAULT_MERGE_GAP_S as DEFAULT_TURN_MERGE_GAP_S,
    DEFAULT_MIN_RUN,
    DEFAULT_MIN_TURN_WORDS,
    Turn,
    snap_span,
    split_tracks,
    turns_from_segments,
)

logger = logging.getLogger(__name__)

# Дефолты пайплайна. Подобраны под реакшн-стримы: окно 45с — типичная длина
# оффтоп-отступления; перекрытие 10с ловит фразы на стыке; чанк 900с (15мин) —
# в пределах одной темы, разумный размер контекста для topic-извлечения.
DEFAULT_WINDOW_S = 45.0
DEFAULT_OVERLAP_S = 10.0
DEFAULT_CHUNK_S = 900.0
DEFAULT_MERGE_GAP_S = 8.0
# Ниже этого confidence — дозапрашиваем дорогую модель (если она задана).
DEFAULT_CASCADE_THRESHOLD = 0.6
# Окна короче — не классифицируем (обрывок, нечего анализировать).
MIN_WINDOW_WORDS = 3

# Бюджеты ответа. Щедрее прежних 128/256: у моделей с extended thinking скрытые
# токены расходуют ТОТ ЖЕ max_tokens, и на 256 ответ обрывался, не дойдя до
# JSON-вердикта. RETRY — единственный повтор при отсутствии вердикта.
TOPIC_MAX_TOKENS = 512
CLASSIFY_MAX_TOKENS = 1024
CLASSIFY_RETRY_MAX_TOKENS = 4096

# Путь по исходнику. Батч = 30 реплик: один промпт на все реплики VOD (43K
# символов) прокси не отдала — вернулся ПУСТОЙ ответ, без stop_reason и без
# ошибки. Ответ здесь — список сегментов, а не один вердикт, поэтому бюджеты
# на порядок больше: обрыв JSON посередине = потерянный батч.
DEFAULT_TURN_BATCH = 30
OUTLINE_MAX_TOKENS = 2048
OUTLINE_RETRY_MAX_TOKENS = 4096
SEGMENTS_MAX_TOKENS = 8192
SEGMENTS_RETRY_MAX_TOKENS = 16384
# Деградированный путь: сколько символов транскрипта класть в один промпт.
SINGLE_PASS_MAX_CHARS = 30_000
# Доля слов стрима, нашедшихся в исходнике. Ниже — привязки фактически нет
# (не тот файл, другая дорожка, другой язык): вся речь уедет стримеру и весь
# VOD станет «оффтопом». Это не повод падать, но повод крикнуть в лог.
MIN_MATCHED_RATIO_WARN = 0.05
# Больше OfftopicSet не примет (валидатор модели). Режем с логом, не молча.
MAX_SEGMENTS = 1000

# Тема, когда извлечь не удалось. Раньше это значение приходило и от исключения
# внутри _extract_topic, и от неразобранного ответа, и никуда не логировалось —
# в проде так и осталось topic="неизвестно" без следов причины.
UNKNOWN_TOPIC = "неизвестно"
# План исходника не получен: путь продолжает работать (реплики и паузы на
# месте), но без ground truth. В промпт уходит явная заглушка, а не пустота.
NO_OUTLINE_TEXT = "ТЕМА: неизвестно (план исходника получить не удалось)"


@dataclass
class _TimedWord:
    """Слово с началом в секундах (local-время) — для нарезки по времени."""

    start_s: float
    text: str


@dataclass
class _Window:
    """Окно классификации: границы + текст для промпта."""

    start_s: float
    end_s: float
    lines: list[str]

    @property
    def word_count(self) -> int:
        return len(self.lines)

    def as_prompt_block(self) -> str:
        return "\n".join(self.lines)


def _parse_ts(line: str) -> Optional[float]:
    """`[HH:MM:SS.mmm] слово` → начало в секундах. None если формат чужой."""
    if not line.startswith("["):
        return None
    close = line.find("]")
    if close == -1:
        return None
    ts = line[1:close]
    try:
        hh, mm, rest = ts.split(":")
        ss = float(rest)
        return int(hh) * 3600 + int(mm) * 60 + ss
    except (ValueError, TypeError):
        return None


def _timed_words(transcript: dict[str, Any]) -> list[_TimedWord]:
    """Плоский поток слов с таймингами (переиспользуем build_word_lines)."""
    out: list[_TimedWord] = []
    for line in build_word_lines(transcript):
        ts = _parse_ts(line)
        if ts is None:
            continue
        # текст = всё после "] "
        close = line.find("]")
        text = line[close + 1 :].strip()
        out.append(_TimedWord(start_s=ts, text=text))
    return out


def _make_windows(
    words: list[_TimedWord],
    *,
    window_s: float,
    overlap_s: float,
) -> list[_Window]:
    """Скользящие окна по времени с перекрытием.

    Шаг = window_s - overlap_s. Каждое окно собирает слова, попавшие в
    [cursor, cursor+window_s). Окна короче MIN_WINDOW_WORDS отбрасываем
    (с логом), чтобы не слать в LLM обрывки.
    """
    if not words:
        return []
    if window_s <= 0:
        raise ValueError("window_s must be positive")
    step = window_s - overlap_s
    if step <= 0:
        raise ValueError("overlap_s must be smaller than window_s")

    t_end = words[-1].start_s
    windows: list[_Window] = []
    dropped = 0
    cursor = words[0].start_s
    while cursor <= t_end:
        hi = cursor + window_s
        bucket = [w for w in words if cursor <= w.start_s < hi]
        if len(bucket) >= MIN_WINDOW_WORDS:
            lines = [f"[{_fmt(w.start_s)}] {w.text}" for w in bucket]
            windows.append(
                _Window(start_s=bucket[0].start_s, end_s=bucket[-1].start_s, lines=lines)
            )
        elif bucket:
            dropped += 1
        cursor += step

    if dropped:
        logger.info("offtopic: dropped %d too-short windows (< %d words)", dropped, MIN_WINDOW_WORDS)
    return windows


def _fmt(seconds: float) -> str:
    """Секунды → `HH:MM:SS.mmm` (для строк окна, как в segments._fmt_ts)."""
    if seconds < 0:
        seconds = 0.0
    total_ms = int(round(seconds * 1000))
    hh, rem = divmod(total_ms, 3_600_000)
    mm, rem = divmod(rem, 60_000)
    ss, ms = divmod(rem, 1000)
    return f"{hh:02d}:{mm:02d}:{ss:02d}.{ms:03d}"


def _chunk_windows(windows: list[_Window], chunk_s: float) -> list[list[_Window]]:
    """Сгруппировать окна в чанки по chunk_s — единица пере-извлечения темы."""
    if not windows:
        return []
    chunks: list[list[_Window]] = []
    current: list[_Window] = []
    anchor = windows[0].start_s
    for w in windows:
        if w.start_s - anchor >= chunk_s and current:
            chunks.append(current)
            current = []
            anchor = w.start_s
        current.append(w)
    if current:
        chunks.append(current)
    return chunks


@dataclass
class _Verdict:
    """Оффтоп-вердикт по окну (промежуточный, до слияния)."""

    window: _Window
    confidence: float
    reason: str
    topic: str
    model: str
    stage: str


@dataclass
class _WindowResult:
    """Что стало с окном после классификации.

    Три исхода, и их нельзя путать:
      verdict != None            — ОФФТОП;
      verdict None, failure None — модель сказала «по теме»;
      verdict None, failure!=None— вердикта нет (обрыв, битый JSON, ошибка сети).
    Последнее — не «чисто», а пробел в покрытии; он идёт в лог.
    """

    verdict: Optional[_Verdict]
    failure: Optional[str] = None


@dataclass
class _Attempt:
    """Одна попытка получить вердикт по окну (с ретраем внутри)."""

    verdict: Optional[ClassificationVerdict]
    model: str
    failure: Optional[str] = None


def classify_transcript(
    transcript: dict[str, Any],
    *,
    client: LLMClient,
    topic_template: Optional[PromptTemplate] = None,
    classification_template: Optional[PromptTemplate] = None,
    cheap_model: Optional[str] = None,
    expensive_model: Optional[str] = None,
    window_s: float = DEFAULT_WINDOW_S,
    overlap_s: float = DEFAULT_OVERLAP_S,
    chunk_s: float = DEFAULT_CHUNK_S,
    merge_gap_s: float = DEFAULT_MERGE_GAP_S,
    cascade_threshold: float = DEFAULT_CASCADE_THRESHOLD,
    transcript_hash: Optional[str] = None,
) -> OfftopicSet:
    """Прогнать транскрипт через оффтоп-классификатор → OfftopicSet.

    :param cheap_model: модель первой ступени. None → дефолт провайдера.
    :param expensive_model: модель второй ступени каскада. None → каскада нет
        (одна ступень), спорные окна остаются с вердиктом дешёвой модели.
    """
    words = _timed_words(transcript)
    if not words:
        logger.info("offtopic: empty transcript, nothing to classify")
        return OfftopicSet(segments=[], source=_source_label(client), transcript_hash=transcript_hash)

    windows = _make_windows(words, window_s=window_s, overlap_s=overlap_s)
    chunks = _chunk_windows(windows, chunk_s)
    logger.info("offtopic: %d words → %d windows → %d chunks", len(words), len(windows), len(chunks))

    verdicts: list[_Verdict] = []
    unclassified: list[tuple[float, str]] = []
    for ci, chunk in enumerate(chunks):
        topic = _extract_topic(chunk, client, topic_template, cheap_model)
        logger.info("offtopic: chunk %d/%d topic=%r", ci + 1, len(chunks), topic)
        for w in chunk:
            res = _classify_window(
                w,
                topic=topic,
                client=client,
                template=classification_template,
                cheap_model=cheap_model,
                expensive_model=expensive_model,
                cascade_threshold=cascade_threshold,
            )
            if res.verdict is not None:
                verdicts.append(res.verdict)
            elif res.failure is not None:
                unclassified.append((w.start_s, res.failure))

    if unclassified:
        # Это НЕ «чисто»: окна остались без вердикта. Пишем сколько и первые
        # причины — иначе следующий отказ снова будет невидим.
        logger.warning(
            "offtopic: %d/%d окон без вердикта (не зачтены как «по теме»); первые: %s",
            len(unclassified),
            len(windows),
            "; ".join(f"{_fmt(t)} {why}" for t, why in unclassified[:5]),
        )

    segments = _merge_verdicts(verdicts, merge_gap_s=merge_gap_s)
    logger.info("offtopic: %d offtopic windows → %d merged segments", len(verdicts), len(segments))

    return OfftopicSet(
        segments=segments,
        source=_source_label(client),
        transcript_hash=transcript_hash,
        model=expensive_model or cheap_model or client.provider.default_model(),
    )


def _extract_topic(
    chunk: list[_Window],
    client: LLMClient,
    template: Optional[PromptTemplate],
    model: Optional[str],
) -> str:
    """Пере-извлечь тему чанка. При ошибке LLM — UNKNOWN_TOPIC (не роняем прогон).

    Каждая деградация до UNKNOWN_TOPIC логируется с причиной: без этого в проде
    оставалось topic="неизвестно" без следов того, что произошло.
    """
    transcript_text = "\n".join(line for w in chunk for line in w.lines)
    prompt = render_topic_prompt(transcript_text, template=template)
    try:
        reply = client.complete(prompt, model=model, max_tokens=TOPIC_MAX_TOKENS)
    except Exception as e:  # noqa: BLE001 - тема не критична, деградируем мягко
        logger.warning("offtopic: topic extraction failed: %s", e)
        return UNKNOWN_TOPIC
    if reply.empty:
        logger.warning("offtopic: topic extraction без текста (%s)", reply.diagnostics())
        return UNKNOWN_TOPIC
    if reply.truncated:
        # Тема — одна строка, обрыв означает, что бюджет ушёл в thinking;
        # первую строку всё же попробуем разобрать.
        logger.info("offtopic: topic ответ обрезан (%s)", reply.diagnostics())
    return parse_topic_response(reply.text)


def _ask_verdict(
    prompt: str,
    *,
    client: LLMClient,
    model: Optional[str],
    window_start_s: float,
) -> _Attempt:
    """Спросить вердикт по окну: запрос + один ретрай при отсутствии вердикта.

    Ретраим ровно один раз и только по причине «вердикта нет» (пустой/обрезанный
    ответ, битый JSON), с увеличенным бюджетом. Сетевая ошибка не ретраится —
    её причина не в бюджете, а повтор удвоит время прогона на ровном месте.
    """
    used_model = model or (client.provider.default_model() or "?")
    budgets = (CLASSIFY_MAX_TOKENS, CLASSIFY_RETRY_MAX_TOKENS)
    last_failure = "no attempt"

    for attempt, max_tokens in enumerate(budgets, 1):
        try:
            reply = client.complete(prompt, model=model, max_tokens=max_tokens)
        except Exception as e:  # noqa: BLE001 - окно пропускаем, не роняем прогон
            return _Attempt(None, used_model, failure=f"llm error: {type(e).__name__}: {e}")

        if reply.empty:
            last_failure = f"пустой ответ ({reply.diagnostics()})"
        else:
            verdict = parse_classification_response(reply.text)
            if not verdict.parse_failed:
                return _Attempt(verdict, reply.model or used_model)
            last_failure = f"{verdict.reason} ({reply.diagnostics()})"

        if attempt < len(budgets):
            logger.info(
                "offtopic: нет вердикта @ %.1fs (%s) — повтор с max_tokens=%d",
                window_start_s,
                last_failure,
                budgets[attempt],
            )

    return _Attempt(None, used_model, failure=last_failure)


def _classify_window(
    window: _Window,
    *,
    topic: str,
    client: LLMClient,
    template: Optional[PromptTemplate],
    cheap_model: Optional[str],
    expensive_model: Optional[str],
    cascade_threshold: float,
) -> _WindowResult:
    """Классифицировать одно окно → _WindowResult (оффтоп / чисто / нет вердикта).

    Каскад: сперва дешёвая модель. Если она сказала «оффтоп», но неуверенно
    (confidence < threshold), и задана дорогая — перепроверяем дорогой. Её
    вердикт финальный (в т.ч. может отменить оффтоп); если дорогая не ответила,
    остаётся вердикт дешёвой — он у нас уже есть, терять его не за что.
    """
    prompt = render_classification_prompt(topic, window.as_prompt_block(), template=template)

    first = _ask_verdict(
        prompt, client=client, model=cheap_model, window_start_s=window.start_s
    )
    if first.verdict is None:
        logger.warning(
            "offtopic: окно @ %.1fs без вердикта: %s", window.start_s, first.failure
        )
        return _WindowResult(None, failure=first.failure)

    verdict = first.verdict
    used_model = first.model
    stage = "cheap"

    need_recheck = (
        verdict.offtopic
        and expensive_model is not None
        and expensive_model != cheap_model
        and verdict.confidence < cascade_threshold
    )
    if need_recheck:
        second = _ask_verdict(
            prompt, client=client, model=expensive_model, window_start_s=window.start_s
        )
        if second.verdict is not None:
            verdict = second.verdict
            used_model = second.model
            stage = "expensive"
        else:
            logger.warning(
                "offtopic: каскад не дал вердикта @ %.1fs (%s) — оставляю вердикт дешёвой модели",
                window.start_s,
                second.failure,
            )

    if not verdict.offtopic:
        return _WindowResult(None)

    return _WindowResult(
        _Verdict(
            window=window,
            confidence=verdict.confidence,
            reason=verdict.reason,
            topic=topic,
            model=used_model,
            stage=stage,
        )
    )


def _merge_verdicts(verdicts: list[_Verdict], *, merge_gap_s: float) -> list[OfftopicSegment]:
    """Слить пересекающиеся/близкие оффтоп-окна в OfftopicSegment.

    Окна перекрываются by design (overlap), плюс подряд идущий оффтоп даёт
    цепочку окон — всё это должно стать ОДНИМ сегментом. Правило слияния:
    следующее окно начинается не позже (конец текущего + merge_gap_s).
    confidence сегмента = максимум по вошедшим окнам (самый уверенный сигнал).
    """
    if not verdicts:
        return []
    ordered = sorted(verdicts, key=lambda v: v.window.start_s)

    segments: list[OfftopicSegment] = []
    cur_start = ordered[0].window.start_s
    cur_end = ordered[0].window.end_s
    cur_best = ordered[0]

    def flush(start: float, end: float, best: _Verdict) -> None:
        # Гарантируем end>start: одиночное окно может иметь start==end
        # (все слова в одну миллисекунду) — раздвигаем на 1мс.
        if end <= start:
            end = start + 0.001
        segments.append(
            OfftopicSegment(
                start_s=round(start, 3),
                end_s=round(end, 3),
                confidence=best.confidence,
                reason=best.reason,
                topic=best.topic,
                model=best.model,
                stage=best.stage,
            )
        )

    for v in ordered[1:]:
        if v.window.start_s <= cur_end + merge_gap_s:
            cur_end = max(cur_end, v.window.end_s)
            if v.confidence > cur_best.confidence:
                cur_best = v
        else:
            flush(cur_start, cur_end, cur_best)
            cur_start = v.window.start_s
            cur_end = v.window.end_s
            cur_best = v
    flush(cur_start, cur_end, cur_best)
    return segments


def _source_label(client: LLMClient) -> str:
    """Строка-источник для audit trail в OfftopicSet.source."""
    return f"llm:{client.provider.id}"


# --- Путь по исходнику ------------------------------------------------------


@dataclass
class _Found:
    """Найденный сегмент до слияния: границы уже снапнуты на реплики."""

    start_s: float
    end_s: float
    confidence: float
    reason: str
    quote: str
    topic: str
    model: str
    stage: str
    # Улики пути по исходнику (см. model.OfftopicSegment). None вне него.
    pause_s: Optional[float] = None
    anchored: Optional[float] = None


@dataclass
class _SegmentsAttempt:
    """Попытка получить СПИСОК сегментов (с ретраем внутри)."""

    answer: Optional[SegmentsAnswer]
    model: str
    failure: Optional[str] = None


def _ask_segments(
    prompt: str,
    *,
    client: LLMClient,
    model: Optional[str],
    label: str,
) -> _SegmentsAttempt:
    """Спросить список сегментов: запрос + один ретрай, если ОТВЕТА нет.

    Ретрай ровно один и только по причине «ответа нет» (пустой ответ, битый
    JSON, чужая схема) — с увеличенным бюджетом; сетевая ошибка не ретраится
    (её причина не в бюджете). Пустой `segments` при разобранном JSON — это
    ОТВЕТ «оффтопа здесь нет», его не ретраим: платить дважды не за что.

    Батч без ответа возвращается как failure и учитывается вызывающим. Молча
    пропустить его нельзя: 30 реплик, посчитанные «по теме» из-за пустого
    ответа прокси, — это ровно тот способ терять оффтоп, из-за которого
    появилась Фаза 1.
    """
    used_model = model or (client.provider.default_model() or "?")
    budgets = (SEGMENTS_MAX_TOKENS, SEGMENTS_RETRY_MAX_TOKENS)
    last_failure = "no attempt"

    for attempt, max_tokens in enumerate(budgets, 1):
        try:
            reply = client.complete(prompt, model=model, max_tokens=max_tokens)
        except Exception as e:  # noqa: BLE001 - батч теряем, прогон не роняем
            return _SegmentsAttempt(
                None, used_model, failure=f"llm error: {type(e).__name__}: {e}"
            )

        if reply.empty:
            last_failure = f"пустой ответ ({reply.diagnostics()})"
        else:
            answer = parse_segments_response(reply.text)
            if not answer.parse_failed:
                return _SegmentsAttempt(answer, reply.model or used_model)
            last_failure = f"{answer.error} ({reply.diagnostics()})"

        if attempt < len(budgets):
            logger.info(
                "offtopic: %s без ответа (%s) — повтор с max_tokens=%d",
                label,
                last_failure,
                budgets[attempt],
            )

    return _SegmentsAttempt(None, used_model, failure=last_failure)


def _extract_outline(
    source_transcript: dict[str, Any],
    *,
    client: LLMClient,
    template: Optional[PromptTemplate],
    model: Optional[str],
    with_hours: bool,
) -> SourceOutline:
    """План исходника — один вызов на весь VOD, ground truth для классификации.

    Заменяет _extract_topic: тема больше не угадывается по речи стримера. Если
    план получить не удалось, путь продолжает работать без него (реплики и
    паузы на месте), но это громкий warning, а не тихая деградация.
    """
    body = format_asr_lines(source_transcript.get("segments") or [], with_hours=with_hours)
    if not body.strip():
        logger.warning("offtopic: транскрипт исходника без текста — плана не будет")
        return SourceOutline(text="", topic="")

    prompt = render_source_outline_prompt(body, template=template)
    budgets = (OUTLINE_MAX_TOKENS, OUTLINE_RETRY_MAX_TOKENS)

    for attempt, max_tokens in enumerate(budgets, 1):
        try:
            reply = client.complete(prompt, model=model, max_tokens=max_tokens)
        except Exception as e:  # noqa: BLE001 - без плана хуже, но не смертельно
            logger.warning("offtopic: план исходника не получен: %s", e)
            return SourceOutline(text="", topic="")

        outline = parse_outline_response(reply.text)
        if not outline.empty:
            logger.info(
                "offtopic: план исходника получен (%d симв.), ТЕМА=%r",
                len(outline.text),
                outline.topic,
            )
            return outline

        if attempt < len(budgets):
            logger.info(
                "offtopic: план исходника пуст (%s) — повтор с max_tokens=%d",
                reply.diagnostics(),
                budgets[attempt],
            )
        else:
            logger.warning(
                "offtopic: план исходника пуст и после повтора (%s)", reply.diagnostics()
            )

    return SourceOutline(text="", topic="")


def _snap_found(
    answer: SegmentsAnswer,
    turns: Sequence[Turn],
    *,
    topic: str,
    model: str,
    stage: str,
    label: str,
    alignment: Optional[Alignment] = None,
    with_hours: bool = False,
) -> list[_Found]:
    """Ответ модели → сегменты с границами по репликам.

    Модель отвечает MM:SS, реплики посчитаны по словам — снапим на реплики.
    Сегмент, не пересёкший ни одной реплики, выбрасываем: такой таймкод
    выдуман. И то, и другое идёт в лог с количеством.
    """
    if answer.dropped:
        logger.warning(
            "offtopic: %s — %d элементов с нечитаемыми границами: %s",
            label,
            answer.dropped,
            "; ".join(answer.dropped_reasons),
        )

    out: list[_Found] = []
    hallucinated = 0
    for raw in answer.segments:
        span = snap_span(raw.start_s, raw.end_s, turns)
        if span is None:
            hallucinated += 1
            logger.info(
                "offtopic: %s — сегмент %s–%s не попал ни в одну реплику, выброшен",
                label,
                format_timecode(raw.start_s, with_hours=with_hours),
                format_timecode(raw.end_s, with_hours=with_hours),
            )
            continue

        # Пауза и привязка — независимые от LLM улики решения. С Фазы 4 они
        # едут в OfftopicSegment (pause_s/anchored), а не только в лог, чтобы
        # UI мог объяснить вердикт. Вне пути по исходнику (alignment=None)
        # улик нет — там оба None.
        pause_s: Optional[float] = None
        anchored: Optional[float] = None
        if alignment is not None:
            ctx = video_context(alignment, span.start_s, span.end_s)
            pause_s = ctx.pause_s
            anchored = round(alignment.anchored_ratio(span.start_s, span.end_s), 4)
            logger.info(
                "offtopic: %s → %s–%s conf=%.2f реплик=%d привязка=%.2f %s: %s",
                label,
                format_timecode(span.start_s, with_hours=with_hours),
                format_timecode(span.end_s, with_hours=with_hours),
                raw.confidence,
                span.turn_count,
                anchored,
                (
                    f"ПАУЗА {ctx.pause_s:.0f}с"
                    if not ctx.playing
                    else f"видео играет (привязок {ctx.anchors})"
                ),
                raw.reason[:120],
            )

        out.append(
            _Found(
                start_s=span.start_s,
                end_s=span.end_s,
                confidence=raw.confidence,
                reason=raw.reason,
                quote=raw.quote,
                topic=topic,
                model=model,
                stage=stage,
                pause_s=pause_s,
                anchored=anchored,
            )
        )

    if hallucinated:
        logger.warning(
            "offtopic: %s — выброшено %d сегментов с выдуманными таймкодами", label, hallucinated
        )
    return out


def classify_with_source(
    transcript: dict[str, Any],
    source_transcript: dict[str, Any],
    *,
    client: LLMClient,
    outline_template: Optional[PromptTemplate] = None,
    classification_template: Optional[PromptTemplate] = None,
    model: Optional[str] = None,
    turn_batch: int = DEFAULT_TURN_BATCH,
    merge_gap_s: float = DEFAULT_MERGE_GAP_S,
    min_run: int = DEFAULT_MIN_RUN,
    turn_merge_gap_s: float = DEFAULT_TURN_MERGE_GAP_S,
    min_turn_words: int = DEFAULT_MIN_TURN_WORDS,
    pause_rate_per_s: float = PAUSE_RATE_PER_S,
    transcript_hash: Optional[str] = None,
    source_transcript_hash: Optional[str] = None,
    source_model: Optional[str] = None,
) -> OfftopicSet:
    """Классификация по исходнику: план видео + реплики стримера с контекстом.

    Схема (вызовов LLM = 1 + число батчей):

      1. Привязка стрима к исходнику (align) → какие слова принадлежат видео.
      2. Расщепление дорожек (split_tracks) → реплики СТРИМЕРА.
      3. План исходника (1 вызов) — тема и блоки, ground truth вместо догадки.
      4. Реплики батчами по ~30 + к каждой то, что играло в видео (или
         `<ПАУЗА Nс>`) → модель сама возвращает границы оффтопа.
      5. Снап границ на реплики, слияние близких сегментов.

    :param source_transcript: транскрипт ИСХОДНИКА (того же движка ASR — иначе
        тексты несравнимы, см. source.transcribe_source).
    :param model: модель для обоих шагов. None → дефолт провайдера.
    :param source_transcript_hash: хеш транскрипта исходника — стухание сегментов
        при перераспознавании исходника (audit trail в OfftopicSet).
    :param source_model: модель ASR исходника (для audit/воспроизводимости).
    """
    alignment = align_transcripts(transcript, source_transcript)
    label_source = _source_label(client)
    used_model = model or client.provider.default_model()
    # Общие для всех выходов метаданные пути по исходнику (в т.ч. ранние
    # пустые результаты — они тоже «source_anchored», просто без сегментов).
    source_meta: dict[str, Any] = {
        "method": "source_anchored",
        "source_transcript_hash": source_transcript_hash,
        "source_model": source_model,
    }

    if not alignment.stream:
        logger.info("offtopic: пустой транскрипт стрима, классифицировать нечего")
        return OfftopicSet(
            segments=[], source=label_source, transcript_hash=transcript_hash, **source_meta
        )

    if alignment.matched_ratio < MIN_MATCHED_RATIO_WARN:
        # Не падаем: возможно, стример правда почти не даёт звук видео. Но чаще
        # это не тот исходник — и тогда весь VOD уедет в «оффтоп».
        logger.warning(
            "offtopic: привязка к исходнику почти не сработала — совпало %.1f%% слов стрима. "
            "Похоже, это не тот файл или другая звуковая дорожка; результат ниже недостоверен",
            100.0 * alignment.matched_ratio,
        )

    split = split_tracks(
        alignment,
        min_run=min_run,
        merge_gap_s=turn_merge_gap_s,
        min_turn_words=min_turn_words,
    )
    turns = split.turns
    if not turns:
        logger.warning(
            "offtopic: собственной речи стримера не нашлось (совпало %.1f%% слов) — "
            "оффтопа нет по построению",
            100.0 * alignment.matched_ratio,
        )
        return OfftopicSet(
            segments=[], source=label_source, transcript_hash=transcript_hash, **source_meta
        )

    with_hours = max(t.end_s for t in turns) >= 3600.0
    outline = _extract_outline(
        source_transcript,
        client=client,
        template=outline_template,
        model=model,
        with_hours=with_hours,
    )
    outline_text = outline.text or NO_OUTLINE_TEXT
    topic = outline.topic or UNKNOWN_TOPIC

    if turn_batch <= 0:
        raise ValueError("turn_batch must be positive")
    batches = [turns[i : i + turn_batch] for i in range(0, len(turns), turn_batch)]
    logger.info(
        "offtopic: путь по исходнику — реплик %d (%.1f мин) → батчей %d",
        len(turns),
        split.streamer_seconds / 60.0,
        len(batches),
    )

    found: list[_Found] = []
    failed: list[str] = []
    for bi, batch in enumerate(batches, 1):
        contexts = [
            turn_context(alignment, t, pause_rate_per_s=pause_rate_per_s).render() for t in batch
        ]
        body = format_turn_lines(batch, contexts, with_hours=with_hours)
        prompt = render_source_classification_prompt(
            outline_text, body, template=classification_template
        )
        label = (
            f"батч {bi}/{len(batches)} "
            f"{format_timecode(batch[0].start_s, with_hours=with_hours)}–"
            f"{format_timecode(batch[-1].end_s, with_hours=with_hours)}"
        )
        logger.info("offtopic: %s, реплик %d, промпт %d символов", label, len(batch), len(prompt))

        attempt = _ask_segments(prompt, client=client, model=model, label=label)
        if attempt.answer is None:
            logger.warning("offtopic: %s без ответа: %s", label, attempt.failure)
            failed.append(f"{label}: {attempt.failure}")
            continue

        found.extend(
            _snap_found(
                attempt.answer,
                batch,
                topic=topic,
                model=attempt.model,
                stage="source",
                label=label,
                alignment=alignment,
                with_hours=with_hours,
            )
        )

    if failed:
        # Реплики этих батчей НЕ проверены — это пробел в покрытии, а не «чисто».
        logger.warning(
            "offtopic: %d/%d батчей остались без ответа (их реплики не проверены): %s",
            len(failed),
            len(batches),
            "; ".join(failed[:5]),
        )

    segments = _merge_found(found, merge_gap_s=merge_gap_s)
    logger.info(
        "offtopic: путь по исходнику — %d сегментов от модели → %d после слияния (%.1f мин)",
        len(found),
        len(segments),
        sum(s.end_s - s.start_s for s in segments) / 60.0,
    )
    return OfftopicSet(
        segments=segments,
        source=label_source,
        transcript_hash=transcript_hash,
        model=used_model,
        **source_meta,
    )


def classify_single_pass(
    transcript: dict[str, Any],
    *,
    client: LLMClient,
    template: Optional[PromptTemplate] = None,
    model: Optional[str] = None,
    merge_gap_s: float = DEFAULT_MERGE_GAP_S,
    max_chars: int = SINGLE_PASS_MAX_CHARS,
    transcript_hash: Optional[str] = None,
) -> OfftopicSet:
    """Деградация без исходника: один проход по ASR-сегментам.

    Тема здесь снова угадывается по смеси речи (исходника нет — расщепить
    дорожки нечем), но границы задаёт модель, а не сетка 45с. Гранулярность —
    ASR-сегмент (`[MM:SS] речь`), а не слово на строку: тот же VOD влезает в
    один-два запроса вместо 94.
    """
    label_source = _source_label(client)
    used_model = model or client.provider.default_model()

    turns = turns_from_segments(transcript)
    if not turns:
        logger.info("offtopic: пустой транскрипт, классифицировать нечего")
        return OfftopicSet(
            segments=[],
            source=label_source,
            transcript_hash=transcript_hash,
            method="single_pass",
        )

    with_hours = max(t.end_s for t in turns) >= 3600.0
    chunks = _chunk_turns_by_chars(turns, max_chars=max_chars, with_hours=with_hours)
    logger.info(
        "offtopic: один проход — %d ASR-сегментов → %d запрос(ов)", len(turns), len(chunks)
    )

    found: list[_Found] = []
    failed: list[str] = []
    for ci, chunk in enumerate(chunks, 1):
        body = _turn_lines(chunk, with_hours=with_hours)
        prompt = render_single_pass_prompt(body, template=template)
        label = (
            f"проход {ci}/{len(chunks)} "
            f"{format_timecode(chunk[0].start_s, with_hours=with_hours)}–"
            f"{format_timecode(chunk[-1].end_s, with_hours=with_hours)}"
        )
        logger.info("offtopic: %s, строк %d, промпт %d символов", label, len(chunk), len(prompt))

        attempt = _ask_segments(prompt, client=client, model=model, label=label)
        if attempt.answer is None:
            logger.warning("offtopic: %s без ответа: %s", label, attempt.failure)
            failed.append(f"{label}: {attempt.failure}")
            continue

        found.extend(
            _snap_found(
                attempt.answer,
                chunk,
                topic=attempt.answer.topic or UNKNOWN_TOPIC,
                model=attempt.model,
                stage="single-pass",
                label=label,
                with_hours=with_hours,
            )
        )

    if failed:
        logger.warning(
            "offtopic: %d/%d проходов без ответа (их отрезки не проверены): %s",
            len(failed),
            len(chunks),
            "; ".join(failed[:5]),
        )

    segments = _merge_found(found, merge_gap_s=merge_gap_s)
    logger.info(
        "offtopic: один проход — %d сегментов от модели → %d после слияния", len(found), len(segments)
    )
    return OfftopicSet(
        segments=segments,
        source=label_source,
        transcript_hash=transcript_hash,
        model=used_model,
        method="single_pass",
    )


def _turn_lines(turns: Sequence[Turn], *, with_hours: bool) -> str:
    """Реплики/ASR-сегменты → строки `[MM:SS] речь` (тело single-pass промпта)."""
    return "\n".join(
        f"[{format_timecode(t.start_s, with_hours=with_hours)}] {t.text}" for t in turns
    )


def _chunk_turns_by_chars(
    turns: Sequence[Turn],
    *,
    max_chars: int,
    with_hours: bool,
) -> list[list[Turn]]:
    """Разбить транскрипт на промпты по символьному бюджету.

    Бюджет, а не число строк: на 43K символов прокси вернула ПУСТОЙ ответ без
    ошибки, поэтому размер промпта — управляемая величина. Одна строка длиннее
    бюджета всё равно едет одна (обрезать речь ради лимита — хуже).
    """
    chunks: list[list[Turn]] = []
    current: list[Turn] = []
    size = 0
    for t in turns:
        line_len = len(t.text) + 12  # + `[MM:SS] ` и перевод строки
        if current and size + line_len > max_chars:
            chunks.append(current)
            current = []
            size = 0
        current.append(t)
        size += line_len
    if current:
        chunks.append(current)
    return chunks


def _merge_found(found: Sequence[_Found], *, merge_gap_s: float) -> list[OfftopicSegment]:
    """Слить пересекающиеся/близкие найденные сегменты в OfftopicSegment.

    Батчи и проходы независимы, поэтому один и тот же уход в свои дела может
    прийти дважды (реплика на границе батчей). Правило то же, что у оконного
    пути: следующий начинается не позже (конец текущего + merge_gap_s);
    confidence = максимум, а reason/quote берём у самого уверенного.
    """
    if not found:
        return []
    ordered = sorted(found, key=lambda f: (f.start_s, f.end_s))

    merged: list[_Found] = []
    for item in ordered:
        if merged and item.start_s <= merged[-1].end_s + merge_gap_s:
            cur = merged[-1]
            best = item if item.confidence > cur.confidence else cur
            merged[-1] = _Found(
                start_s=cur.start_s,
                end_s=max(cur.end_s, item.end_s),
                confidence=max(cur.confidence, item.confidence),
                reason=best.reason,
                quote=best.quote,
                topic=best.topic,
                model=best.model,
                stage=best.stage,
                # Улики берём у самого уверенного вошедшего сегмента — как
                # reason/quote. Пересчитать по слитому интервалу нечем: alignment
                # сюда не передаётся, а границы после слияния всё равно снапнуты.
                pause_s=best.pause_s,
                anchored=best.anchored,
            )
        else:
            merged.append(item)

    if len(merged) > MAX_SEGMENTS:
        logger.warning(
            "offtopic: сегментов %d, в результат уйдут первые %d (лимит модели)",
            len(merged),
            MAX_SEGMENTS,
        )
        merged = merged[:MAX_SEGMENTS]

    return [
        OfftopicSegment(
            start_s=round(f.start_s, 3),
            end_s=round(max(f.end_s, f.start_s + 0.001), 3),
            confidence=f.confidence,
            reason=f.reason,
            topic=f.topic,
            quote=f.quote or None,
            model=f.model,
            stage=f.stage,
            pause_s=f.pause_s,
            anchored=f.anchored,
        )
        for f in merged
    ]
