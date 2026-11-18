"""Расщепление дорожек: что говорит ВИДЕО, а что — СТРИМЕР.

Вход — Alignment (align.py): по каждому слову стрима известно, попало ли оно в
шингл, найденный в исходнике. Слово из шингла — ВИДЕО, остальное — СТРИМЕР.
Выход — реплики стримера (Turn): непрерывные отрезки его собственной речи, они
и подаются классификатору вместо смеси.

Два сглаживания обязательны — без них получается шум, а не реплики:

1. `smooth(min_run=3)` — одиночное слово-остров внутри чужой дорожки это ошибка
   ASR («и» распозналось в обеих дорожках), а не смена говорящего.
2. Склейка реплик, разорванных короткой (< 2.5с) вставкой видео: стример
   говорит сквозь фразу из видео, это одна реплика, а не три.

Реплики короче 4 слов отбрасываются: судить «ага», «ну да» нечего, а в промпте
они занимают место.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Sequence

from .align import Alignment, Word

logger = logging.getLogger(__name__)

# Прогонов короче — затираем значением предыдущего прогона (ошибка ASR).
DEFAULT_MIN_RUN = 3
# Вставка видео короче — реплику стримера не разрывает.
DEFAULT_MERGE_GAP_S = 2.5
# Реплики короче — выбрасываем.
DEFAULT_MIN_TURN_WORDS = 4


@dataclass
class Turn:
    """Реплика стримера: непрерывный отрезок его собственной речи."""

    start_s: float
    end_s: float
    text: str
    word_count: int

    @property
    def duration_s(self) -> float:
        return round(self.end_s - self.start_s, 3)


@dataclass
class TrackSplit:
    """Результат расщепления: разметка по словам + реплики стримера."""

    is_video: list[bool]
    turns: list[Turn]
    video_words: int
    streamer_words: int
    dropped_short_turns: int = 0
    merged_turns: int = 0

    @property
    def total_words(self) -> int:
        return self.video_words + self.streamer_words

    @property
    def streamer_seconds(self) -> float:
        return round(sum(t.duration_s for t in self.turns), 3)


def smooth(flags: list[bool], min_run: int = DEFAULT_MIN_RUN) -> list[bool]:
    """Затереть прогоны короче min_run значением предыдущего прогона.

    Прогоны на самом краю списка не трогаем: у первого нет «предыдущего», а
    затирать последний по предыдущему — значит выдумывать данные там, где запись
    просто закончилась. Затирание идёт слева направо по уже изменённому списку,
    поэтому цепочка коротких прогонов сливается в один — так и надо: это шум.
    """
    if min_run <= 1 or not flags:
        return list(flags)
    out = list(flags)
    i = 0
    while i < len(out):
        j = i
        while j < len(out) and out[j] == out[i]:
            j += 1
        if j - i < min_run and i > 0 and j < len(out):
            for k in range(i, j):
                out[k] = out[i - 1]
        i = j
    return out


def build_turns(
    words: list[Word],
    is_video: list[bool],
    *,
    merge_gap_s: float = DEFAULT_MERGE_GAP_S,
    min_turn_words: int = DEFAULT_MIN_TURN_WORDS,
) -> tuple[list[Turn], int, int]:
    """Собрать реплики стримера из размеченных слов.

    :return: (реплики, сколько склеено, сколько выброшено коротких)
    """
    raw: list[dict] = []
    current: Optional[dict] = None
    for w, video in zip(words, is_video):
        if video:
            if current is not None:
                raw.append(current)
                current = None
            continue
        if current is None:
            current = {"start": w.start_s, "end": w.end_s, "words": [w.raw]}
        else:
            current["end"] = w.end_s
            current["words"].append(w.raw)
    if current is not None:
        raw.append(current)

    merged: list[dict] = []
    merges = 0
    for turn in raw:
        if merged and turn["start"] - merged[-1]["end"] < merge_gap_s:
            merged[-1]["end"] = turn["end"]
            merged[-1]["words"] += turn["words"]
            merges += 1
        else:
            merged.append(dict(turn))

    turns: list[Turn] = []
    dropped = 0
    for turn in merged:
        if len(turn["words"]) < min_turn_words:
            dropped += 1
            continue
        turns.append(
            Turn(
                start_s=round(turn["start"], 3),
                end_s=round(max(turn["end"], turn["start"] + 0.001), 3),
                text=" ".join(turn["words"]),
                word_count=len(turn["words"]),
            )
        )
    return turns, merges, dropped


def split_tracks(
    alignment: Alignment,
    *,
    min_run: int = DEFAULT_MIN_RUN,
    merge_gap_s: float = DEFAULT_MERGE_GAP_S,
    min_turn_words: int = DEFAULT_MIN_TURN_WORDS,
) -> TrackSplit:
    """Alignment → дорожки + реплики стримера."""
    is_video = smooth(alignment.matched, min_run=min_run)
    turns, merges, dropped = build_turns(
        alignment.stream,
        is_video,
        merge_gap_s=merge_gap_s,
        min_turn_words=min_turn_words,
    )
    video_words = sum(is_video)
    streamer_words = len(is_video) - video_words

    logger.info(
        "turns: %d слов → ВИДЕО %d (%.0f%%), СТРИМЕР %d (%.0f%%); "
        "реплик %d (%.1f мин), склеено %d, отброшено коротких %d",
        len(is_video),
        video_words,
        100.0 * video_words / len(is_video) if is_video else 0.0,
        streamer_words,
        100.0 * streamer_words / len(is_video) if is_video else 0.0,
        len(turns),
        sum(t.duration_s for t in turns) / 60.0,
        merges,
        dropped,
    )
    return TrackSplit(
        is_video=is_video,
        turns=turns,
        video_words=video_words,
        streamer_words=streamer_words,
        dropped_short_turns=dropped,
        merged_turns=merges,
    )


# --- Снап границ ------------------------------------------------------------


@dataclass(frozen=True)
class SnappedSpan:
    """Границы сегмента после снапа на реплики + какие реплики он накрыл."""

    start_s: float
    end_s: float
    first_index: int
    last_index: int

    @property
    def turn_count(self) -> int:
        return self.last_index - self.first_index + 1


def snap_span(
    start_s: float,
    end_s: float,
    turns: Sequence[Turn],
) -> Optional[SnappedSpan]:
    """Подтянуть границы модели к границам реплик, которые она накрыла.

    Модель отвечает в MM:SS — точность 1с, плюс она смотрит на таймкоды строк,
    а не на слова. Оставлять её числа как есть значит резать по середине слова.
    Берём реплики, которые сегмент реально пересекает, и растягиваем границы до
    первой и последней из них: границы реплик посчитаны по словам ASR.

    None, если сегмент не пересёк НИ ОДНОЙ реплики: такой таймкод модель
    выдумала (или он попал целиком в речь видео), и в результат он не идёт.
    """
    if end_s <= start_s or not turns:
        return None

    hits = [
        i
        for i, t in enumerate(turns)
        # Пересечение, а не вложенность: модель часто даёт границы «примерно».
        if t.end_s > start_s and t.start_s < end_s
    ]
    if not hits:
        return None

    first, last = hits[0], hits[-1]
    return SnappedSpan(
        start_s=round(turns[first].start_s, 3),
        end_s=round(turns[last].end_s, 3),
        first_index=first,
        last_index=last,
    )


def turns_from_segments(
    transcript: dict,
    *,
    min_turn_words: int = 1,
) -> list[Turn]:
    """ASR-сегменты → «реплики» для деградированного пути (без исходника).

    Расщепления дорожек здесь нет (исходника нет — сравнивать не с чем), но снап
    границ нужен тот же: модель видит строки `[MM:SS] речь` и отвечает MM:SS,
    поэтому её границы подтягиваем к границам ASR-сегментов.
    """
    out: list[Turn] = []
    for seg in transcript.get("segments") or []:
        words_raw = seg.get("words") or []
        text = str(seg.get("text", "")).strip()
        if not text:
            # Транскрипт без готового `text` (только слова) — склеиваем сами.
            text = " ".join(
                str(w.get("word", "")).strip()
                for w in words_raw
                if str(w.get("word", "")).strip()
            ).strip()
        start, end = seg.get("start"), seg.get("end")
        if start is None and words_raw:
            start = words_raw[0].get("start")
        if end is None and words_raw:
            end = words_raw[-1].get("end")
        if not text or start is None or end is None:
            continue
        start_f, end_f = float(start), float(end)
        if end_f <= start_f:
            continue
        words = text.split()
        if len(words) < min_turn_words:
            continue
        out.append(
            Turn(
                start_s=round(start_f, 3),
                end_s=round(end_f, 3),
                text=text,
                word_count=len(words),
            )
        )
    out.sort(key=lambda t: t.start_s)
    return out
