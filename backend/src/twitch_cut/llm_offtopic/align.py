"""Привязка транскрипта стрима к транскрипту исходника — чистый текст, без LLM.

ЗАЧЕМ. Стример проигрывает видео в эфир, поэтому речь источника попадает в
аудио стрима и распознаётся тем же ASR вперемешку с его комментарием. Значит
транскрипт стрима — это СМЕСЬ двух дорожек, и классификатор, который судит
смесь, обречён видеть «по теме» и «оффтоп» в одном окне.

Имея транскрипт исходника, смесь можно расщепить: строим индекс шинглов по 4
слова из исходника и помечаем слова стрима, попавшие в найденный шингл. Такое
слово — это ВИДЕО; остальные — СТРИМЕР (расщепление в turns.py).

Побочный продукт важнее самой разметки: `pos_at` — это восстановленный таймлайн
воспроизведения, пары «время стрима → позиция в исходнике». По нему видно, что
играло в видео в конкретный момент и играло ли вообще (pause.py).

N=4: на 3 словах частотные фразы («и вот он такой») дают ложные совпадения, на
5 теряются короткие вставки. Всё детерминированно и бесплатно — никаких
запросов к модели.
"""

from __future__ import annotations

import bisect
import logging
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Длина шингла в словах.
SHINGLE_N = 4

# Нормализация слова: только буквы/цифры, регистр вниз. Пунктуация и дефисы
# уходят — ASR ставит их непредсказуемо, и без нормализации совпадений почти нет.
_NON_WORD_RE = re.compile(r"[^а-яёa-z0-9]")


def normalize_word(raw: str) -> str:
    """`«Айфон,»` → `айфон`. Пустая строка, если значимых символов нет."""
    return _NON_WORD_RE.sub("", raw.lower())


@dataclass(frozen=True)
class Word:
    """Слово с таймингами. `text` нормализовано, `raw` — как сказал ASR."""

    text: str
    raw: str
    start_s: float
    end_s: float


def transcript_words(transcript: dict[str, Any]) -> list[Word]:
    """Плоский поток слов из транскрипта (формат WhisperX/GigaAM: segments→words).

    Слова без пословных таймингов и без значимых символов пропускаются: они не
    участвуют ни в шинглах, ни в подсчёте привязки. Порядок гарантируем
    сортировкой по времени — на него опираются bisect-запросы в Alignment.
    """
    out: list[Word] = []
    skipped = 0
    for seg in transcript.get("segments") or []:
        for w in seg.get("words") or []:
            raw = str(w.get("word", "") or "").strip()
            text = normalize_word(raw)
            if not text:
                skipped += 1
                continue
            try:
                start = float(w["start"])
            except (KeyError, TypeError, ValueError):
                skipped += 1
                continue
            try:
                end = float(w.get("end", start))
            except (TypeError, ValueError):
                end = start
            out.append(Word(text=text, raw=raw, start_s=start, end_s=max(start, end)))
    if skipped:
        logger.info("align: пропущено %d слов без таймингов/без букв", skipped)
    out.sort(key=lambda x: x.start_s)
    return out


@dataclass
class Alignment:
    """Результат привязки стрима к исходнику.

    :param matched: по слову стрима — попало ли оно в шингл, найденный в
        исходнике (то есть «это говорит видео»).
    :param pos_at: пары (время стрима, индекс слова в исходнике), отсортированы
        по времени. Это и есть восстановленный таймлайн воспроизведения.
    """

    stream: list[Word]
    source: list[Word]
    matched: list[bool]
    pos_at: list[tuple[float, int]]
    shingle_n: int = SHINGLE_N

    # Служебные проекции для bisect (строятся в __post_init__).
    _anchor_ts: list[float] = field(default_factory=list, repr=False)
    _stream_ts: list[float] = field(default_factory=list, repr=False)

    def __post_init__(self) -> None:
        self._anchor_ts = [t for t, _ in self.pos_at]
        self._stream_ts = [w.start_s for w in self.stream]

    # --- агрегаты -----------------------------------------------------------

    @property
    def matched_ratio(self) -> float:
        """Доля слов стрима, отданных дорожке ВИДЕО. На реакшн-VOD ~0.5."""
        if not self.stream:
            return 0.0
        return sum(self.matched) / len(self.stream)

    def anchored_ratio(self, start_s: float, end_s: float) -> float:
        """Доля привязанных слов стрима внутри интервала.

        Независимая от LLM улика: на замеренном VOD внутри найденного оффтопа
        0.117, вне него 0.667. Кандидат, а не вердикт — стример может говорить
        поверх играющего видео (тогда привязка низкая, но это не оффтоп).
        """
        lo = bisect.bisect_left(self._stream_ts, start_s)
        hi = bisect.bisect_right(self._stream_ts, end_s)
        if hi <= lo:
            return 0.0
        return sum(self.matched[lo:hi]) / (hi - lo)

    # --- таймлайн воспроизведения -------------------------------------------

    def anchors_in(self, start_s: float, end_s: float) -> list[int]:
        """Позиции исходника, привязанные ВНУТРИ интервала (может быть пусто)."""
        lo = bisect.bisect_left(self._anchor_ts, start_s)
        hi = bisect.bisect_right(self._anchor_ts, end_s)
        return [p for _, p in self.pos_at[lo:hi]]

    def last_anchor_before(self, t_s: float) -> Optional[int]:
        """Последняя позиция исходника до момента t — «что играло перед паузой»."""
        idx = bisect.bisect_left(self._anchor_ts, t_s)
        if idx <= 0:
            return None
        return self.pos_at[idx - 1][1]

    def source_text(
        self, lo: int, hi: int, *, max_chars: Optional[int] = None, tail: bool = False
    ) -> str:
        """Текст исходника по диапазону индексов слов (границы клампятся).

        :param tail: обрезать не начало, а хвост (для «что было перед паузой»).
        """
        if not self.source:
            return ""
        lo = max(0, min(lo, len(self.source)))
        hi = max(lo, min(hi, len(self.source)))
        text = " ".join(w.text for w in self.source[lo:hi])
        if max_chars is not None and len(text) > max_chars:
            return text[-max_chars:] if tail else text[:max_chars]
        return text


def build_shingle_index(
    words: list[Word], n: int = SHINGLE_N
) -> dict[tuple[str, ...], list[int]]:
    """Индекс шинглов: кортеж из n нормализованных слов → позиции в тексте."""
    index: dict[tuple[str, ...], list[int]] = defaultdict(list)
    for i in range(len(words) - n + 1):
        index[tuple(w.text for w in words[i : i + n])].append(i)
    return index


def align_words(
    stream: list[Word], source: list[Word], *, n: int = SHINGLE_N
) -> Alignment:
    """Привязать поток слов стрима к потоку слов исходника."""
    if n < 2:
        raise ValueError("shingle_n must be >= 2")

    matched = [False] * len(stream)
    pos_at: list[tuple[float, int]] = []
    if not stream or len(source) < n:
        if stream and source:
            logger.warning(
                "align: исходник короче шингла (%d слов < %d) — привязки не будет",
                len(source),
                n,
            )
        return Alignment(
            stream=stream, source=source, matched=matched, pos_at=pos_at, shingle_n=n
        )

    index = build_shingle_index(source, n)
    hits = 0
    for i in range(len(stream) - n + 1):
        key = tuple(w.text for w in stream[i : i + n])
        positions = index.get(key)
        if not positions:
            continue
        hits += 1
        # Первая позиция: повторы фразы в исходнике редки, а выбор «самой
        # правильной» из них требует глобального выравнивания — оно тут не нужно,
        # нам достаточно знать, что видео в этот момент звучит.
        pos_at.append((stream[i].start_s, positions[0]))
        for j in range(i, i + n):
            matched[j] = True

    logger.info(
        "align: стрим %d слов, исходник %d слов, шинглов совпало %d, "
        "слов привязано %d (%.1f%%)",
        len(stream),
        len(source),
        hits,
        sum(matched),
        100.0 * sum(matched) / len(stream),
    )
    return Alignment(
        stream=stream, source=source, matched=matched, pos_at=pos_at, shingle_n=n
    )


def align_transcripts(
    stream_transcript: dict[str, Any],
    source_transcript: dict[str, Any],
    *,
    n: int = SHINGLE_N,
) -> Alignment:
    """align_words для двух транскриптов (удобная точка входа для пайплайна)."""
    return align_words(
        transcript_words(stream_transcript), transcript_words(source_transcript), n=n
    )
