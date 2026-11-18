"""audit.py — сверка «где мат на самом деле» с «где его нашло приложение».

Зачем это существует. Проверять качество детекции на слух и пересказывать
результат словами («вот тут не нашло, а тут замьютило лишнее») — дорого и
невоспроизводимо. `twitch-cut audit <workdir>` собирает всё, что можно
проверить машинно, в один отчёт audit.md рядом с decisions.json.

ЧТО ПРОВЕРЯЕТСЯ АВТОМАТИЧЕСКИ:

1. Пропуски, видимые в транскрипте: слово с обсценным корнем есть, а мьюта на
   нём нет. Корни и матчер — ОБЩИЕ с детектором (profanity.roots_in), поэтому
   «отчёт знает корень, а детектор нет» здесь больше не бывает: остаются только
   настоящие потери — статус, группировка, потолки, границы.
2. Ложные срабатывания: мьют стоит, а ни в одном перекрытом слове обсценного
   корня нет.
3. Границы («рамки») — по WAV, а не по таймингам ASR: сколько тишины в голове
   и в хвосте мьюта, и не обрывается ли речь ровно на границе (тогда мат,
   скорее всего, торчит наружу).
4. Дыры ASR: громкие участки без единого слова. Мат в них не мог быть найден
   в принципе — их надо слушать руками.
5. Ручные правки против решения программы — если включён «Журнал правок»
   (edit_journal.py пишет `<decisions>.edits.json`). Это единственный источник,
   который видит НЕПРАВИЛЬНО УСЛЫШАННЫЕ слова: человек поставил мьют там, где в
   транскрипте мата нет.

ЧЕГО МАШИНА НЕ МОЖЕТ: слово, которое ASR услышал НЕПРАВИЛЬНО («блять» →
«плять», «сука» → «сумка»). В транскрипте его нет, энергия там обычная речь.
Для этого рядом кладётся ground_truth.txt — список реальных таймкодов от
человека; audit сверяет с ним мьюты и печатает попадания/промахи/лишнее.
Формат — см. GROUND_TRUTH_HELP ниже.

Все времена в отчёте ЛОКАЛЬНЫЕ (от range_in), как на таймлайне «Правки».
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

from .edit_journal import (
    baseline_source,
    journal_path,
    net_changes,
    read_journal,
    split_added_by_origin,
)
from .mute_refine import (
    FLOOR_MULT,
    FLOOR_PCT,
    FRAME_MS,
    MARGIN_MS,
    REANCHOR_MAX_VOICED,
    THR_CAP,
    WavEnergy,
    _percentile,
    global_threshold,
    open_energy,
)
from .profanity import (
    CORE_ROOTS as _CORE_ROOTS,
    SPICY_ROOTS as _SPICY_ROOTS,
    RussianNormalizer,
    match_root,
    roots_in,
)
from .transcription import iter_words

# ЗАМЕТКА ДЛЯ ЧЕЛОВЕКА-ПОДСТАВЬ-СВОИ-ТАЙМКОДЫ
GROUND_TRUTH_HELP = """\
# ground_truth.txt — реальные таймкоды мата, по одному в строке.
# Время локальное (от начала диапазона обработки), как на таймлайне «Правки».
#   MM:SS.mmm            — точка
#   HH:MM:SS.mmm         — точка
#   01:23.4-01:24.1      — интервал
# После времени можно дописать слово/комментарий, после # — комментарий.
# Пример:
#   00:05.9 хуйня
#   02:11.2-02:12.0 бля  # тихо, на фоне музыки
"""

# --- корни: ЕДИНЫЙ словарь с детектором -------------------------------------
# Списки и матчер живут в profanity.py и используются обоими: детектором и этим
# отчётом. Раньше здесь была своя копия, заведомо шире, и она разъехалась с
# детектором — отчёт знал «гандон», детектор нет. Реэкспорт оставлен, чтобы не
# переписывать обращения audit_mod.CORE_ROOTS в тестах и скриптах.
CORE_ROOTS = _CORE_ROOTS
SPICY_ROOTS = _SPICY_ROOTS


# --- транскрипт --------------------------------------------------------------


@dataclass(frozen=True)
class Word:
    idx: int
    text: str
    norm: str
    start_ms: int
    end_ms: int
    segment_id: str
    timing_source: str

    @property
    def core(self) -> list[str]:
        return roots_in(self.norm, CORE_ROOTS)

    @property
    def spicy(self) -> list[str]:
        return roots_in(self.norm, SPICY_ROOTS)


def _as_ms(value: Any, fallback: int) -> int:
    """Секунды из транскрипта → мс. None/мусор → fallback (у слова может не
    быть своих таймингов, тогда наследуем границы сегмента)."""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if math.isfinite(value):
            return int(round(float(value) * 1000))
    return fallback


def load_words(
    transcript: dict[str, Any], normalizer: RussianNormalizer | None = None
) -> list[Word]:
    normalizer = normalizer or RussianNormalizer()
    words: list[Word] = []
    cursor = 0
    for idx, raw in enumerate(iter_words(transcript)):
        seg_start = _as_ms(raw.get("segment_start"), cursor)
        seg_end = _as_ms(raw.get("segment_end"), seg_start)
        start = _as_ms(raw.get("start"), seg_start)
        end = _as_ms(raw.get("end"), max(seg_end, start))
        if end < start:
            end = start
        text = str(raw.get("text", "")).strip()
        words.append(
            Word(
                idx=idx,
                text=text,
                norm=normalizer.normalize(text),
                start_ms=start,
                end_ms=end,
                segment_id=str(raw.get("segment_id", "")),
                timing_source=str(raw.get("timing_source", "word")),
            )
        )
        cursor = end
    return words


def fmt_ms(ms: int) -> str:
    ms = max(0, int(ms))
    total, frac = divmod(ms, 1000)
    hours, rem = divmod(total, 3600)
    minutes, seconds = divmod(rem, 60)
    head = f"{hours}:{minutes:02d}:{seconds:02d}" if hours else f"{minutes:02d}:{seconds:02d}"
    return f"{head}.{frac:03d}"


def context(words: Sequence[Word], first: int, last: int, radius: int = 3) -> str:
    """«… это была [хуйня] такая …» — слово в окружении соседей."""
    lo = max(0, first - radius)
    hi = min(len(words), last + radius + 1)
    parts: list[str] = []
    if lo > 0:
        parts.append("…")
    for i in range(lo, hi):
        text = words[i].text or "·"
        parts.append(f"[{text}]" if first <= i <= last else text)
    if hi < len(words):
        parts.append("…")
    return " ".join(parts)


# --- энергия: как окно мьюта сидит на речи -----------------------------------

EDGE_FRAMES = 3
"""Сколько кадров (по 10 мс) считаем «краем» окна при проверке обрыва речи."""

VOICE_LEFT_PCT = 0.2
"""С какой доли звука в НЕзамьюченной части слова считаем, что мат слышно.

Не ноль: один-два кадра над порогом даёт затухающий хвост гласной и щелчки
склейки, на слух это не мат. На реальном стриме у «мьют поджат по звуку» выходит
0.00–0.05, у настоящего торчащего мата — 0.4 и выше, так что порог посередине.
"""

SILENT_SPAN_RATIO = 0.08
"""Насколько тише окрестности должно быть внутри слова, чтобы считать, что
тайминг ASR указывает НЕ на звук.

Абсолютным порогом это не измерить: `global_threshold` — уровень громкой речи по
всему файлу, и тихо сказанное слово его законно не достаёт. Поэтому сравниваем
пик внутри слова с пиком в ±SPAN_MARGIN_MS вокруг него. На реальном стриме
разделение чистое: у слов, чьи границы уехали в тишину, отношение 0.00–0.05, у
тихо, но реально произнесённых — 0.13 и выше.
"""

SPAN_MARGIN_MS = 700
"""Окрестность слова для оценки «а звучит ли тут вообще что-нибудь»."""

LOUD_SPAN_RATIO = 0.3
"""Доля пика окрестности, по которой отмечаем «вот здесь реально звучит речь» —
чтобы в отчёте показать, куда съехал тайминг."""

REANCHORED_VOICE_PCT = 0.6
"""Какую долю ИЗМЕРЕННОГО звука слова должен накрыть мьют, чтобы уехавший тайминг
перестал быть пропуском.

Нужно потому, что пайплайн умеет ДОТЯГИВАТЬ окно до звука: окно, стоящее на
тишине, он расширяет назад (`config.mute_expand_on_silence_ms`), а перенос на
ближайшую речь (`mute_refine.reanchor_window`) существует, но по замеру выключен.
И в том и в другом случае мьют СПЕЦИАЛЬНО не совпадает с границами слова — а
именно по ним отчёт ищет пропуски. Без этой проверки удачное расширение читалось
бы как «мат звучит незакрытым», то есть отчёт звал бы человека править то, что
программа уже починила."""


@dataclass
class EnergyFit:
    fill: float
    """Доля кадров окна выше порога речи. Низкая = мьютим в основном тишину."""
    head_ms: int
    tail_ms: int
    hot_left: bool
    """Речь идёт непрерывно ЧЕРЕЗ левую границу — слово началось раньше мьюта."""
    hot_right: bool
    """То же справа: речь продолжается за правой границей."""
    peak: float
    thr: float


def _threshold(frames: Sequence[float], window: Sequence[float]) -> tuple[float, float]:
    """Порог речи по той же формуле, что в mute_refine.trim_window (пол по всему
    участку с запасом, пик — по самому окну). Возвращает (peak, thr)."""
    peak = max(window) if window else 0.0
    if peak <= 0.0:
        return 0.0, 0.0
    floor = _percentile(list(frames), FLOOR_PCT)
    return peak, min(peak * THR_CAP, max(peak * 0.12, floor * FLOOR_MULT))


def fit_window(energy: WavEnergy, start_ms: int, end_ms: int) -> EnergyFit | None:
    if end_ms <= start_ms:
        return None
    frames, base_ms = energy.rms_frames(max(0, start_ms - MARGIN_MS), end_ms + MARGIN_MS)
    if not frames:
        return None
    lo = max(0, int((start_ms - base_ms) / FRAME_MS))
    hi = min(len(frames), int(math.ceil((end_ms - base_ms) / FRAME_MS)))
    if hi <= lo:
        return None
    window = frames[lo:hi]
    peak, thr = _threshold(frames, window)
    if peak <= 0.0:
        return EnergyFit(0.0, end_ms - start_ms, end_ms - start_ms, False, False, 0.0, 0.0)

    hot = [v >= thr for v in window]
    first = next((i for i, v in enumerate(hot) if v), None)
    if first is None:
        return EnergyFit(0.0, end_ms - start_ms, end_ms - start_ms, False, False, peak, thr)
    last = next(i for i in range(len(hot) - 1, -1, -1) if hot[i])

    outside_left = frames[max(0, lo - EDGE_FRAMES) : lo]
    outside_right = frames[hi : hi + EDGE_FRAMES]
    return EnergyFit(
        fill=sum(1 for v in hot if v) / len(hot),
        head_ms=first * FRAME_MS,
        tail_ms=(len(hot) - 1 - last) * FRAME_MS,
        hot_left=first < EDGE_FRAMES
        and bool(outside_left)
        and all(v >= thr for v in outside_left),
        hot_right=last >= len(hot) - EDGE_FRAMES
        and bool(outside_right)
        and all(v >= thr for v in outside_right),
        peak=peak,
        thr=thr,
    )


def voiced_ratio(
    energy: WavEnergy, start_ms: int, end_ms: int, threshold: float
) -> float | None:
    """Доля кадров окна выше АБСОЛЮТНОГО порога (`mute_refine.global_threshold`).

    Не то же самое, что `EnergyFit.fill`: тот считает порог от пика САМОГО окна,
    и в тишине пиком становится шум — порог падает под него, и «речь» находится
    в пустом месте (на реальном прогоне у мьюта, стоявшего целиком в тишине,
    fill = 42%). Поэтому вопрос «а глушит ли этот мьют хоть что-нибудь» можно
    задать только абсолютным порогом — тем же, которым это решает пайплайн
    (`mute_refine.reanchor_window`).
    """
    if threshold <= 0.0 or end_ms <= start_ms:
        return None
    frames, _ = energy.rms_frames(start_ms, end_ms)
    if not frames:
        return None
    return sum(1 for value in frames if value >= threshold) / len(frames)


def on_silence(voiced: float | None) -> bool:
    """Окно стоит на тишине — глушить там нечего.

    Порог общий с пайплайном (`mute_refine.REANCHOR_MAX_VOICED`) намеренно:
    отчёт должен называть «мьют встал не на звук» ровно там, где так же считает
    обработка. Иначе отчёт советует лечить padding и словарь в местах, где
    пайплайн уже умеет расширить окно назад, а настоящую причину не называет
    вовсе.
    """
    return voiced is not None and voiced <= REANCHOR_MAX_VOICED


def _subtract(
    span: tuple[int, int], covers: Sequence[tuple[int, int]]
) -> list[tuple[int, int]]:
    """Что осталось от `span` после вычитания `covers`."""
    rest = [span]
    for c_start, c_end in covers:
        nxt: list[tuple[int, int]] = []
        for start, end in rest:
            if c_end <= start or c_start >= end:
                nxt.append((start, end))
                continue
            if c_start > start:
                nxt.append((start, min(c_start, end)))
            if c_end < end:
                nxt.append((max(c_end, start), end))
        rest = nxt
    return [(s, e) for s, e in rest if e > s]


def loudest_ms(energy: WavEnergy, start_ms: int, end_ms: int) -> int | None:
    """Середина самого громкого кадра на [start_ms, end_ms) — «где слово звучит».

    Мьют, не накрывший этот кадр, не накрыл главное: даже если вне мьюта по доле
    кадров тихо, самое слышное место мата осталось открытым.
    """
    if end_ms <= start_ms:
        return None
    frames, base_ms = energy.rms_frames(start_ms, end_ms)
    if not frames:
        return None
    best = max(range(len(frames)), key=lambda i: frames[i])
    if frames[best] <= 0.0:
        return None
    return base_ms + best * FRAME_MS + FRAME_MS // 2


def uncovered_voice(
    energy: WavEnergy,
    start_ms: int,
    end_ms: int,
    covers: Sequence[tuple[int, int]],
) -> float | None:
    """Доля ЗВУКА в тех частях слова, которые мьют НЕ накрыл.

    Нужна, чтобы не обвинять пайплайн в пропуске там, где он прав. Тайминги ASR
    систематически длиннее самого слова (хвост уезжает в паузу), а подгонка по
    аудио режет мьют ровно по звуку — формально «серединa слова вне мьюта», а на
    слух мат закрыт целиком. Различить это можно только по звуку.

    Порог речи — тот же, что у подгонки (`mute_refine`), и пик считается по
    ВСЕМУ слову. Поэтому 0.0 читается как «вне мьюта тишина», а заметно больше
    нуля — как «за границей мьюта осталась речь, мат слышно». None — считать не
    по чему (нет аудио или нечего вычитать).
    """
    rest = _subtract((start_ms, end_ms), covers)
    if not rest or end_ms <= start_ms:
        return None
    frames, base_ms = energy.rms_frames(max(0, start_ms - MARGIN_MS), end_ms + MARGIN_MS)
    if not frames:
        return None
    lo = max(0, int((start_ms - base_ms) / FRAME_MS))
    hi = min(len(frames), int(math.ceil((end_ms - base_ms) / FRAME_MS)))
    if hi <= lo:
        return None
    peak, thr = _threshold(frames, frames[lo:hi])
    if peak <= 0.0:
        return 0.0
    total = voiced = 0
    for span_start, span_end in rest:
        i0 = max(lo, int((span_start - base_ms) / FRAME_MS))
        i1 = min(hi, int(math.ceil((span_end - base_ms) / FRAME_MS)))
        for value in frames[i0:i1]:
            total += 1
            voiced += value >= thr
    if total == 0:
        return None
    return voiced / total


def span_voice_ratio(
    energy: WavEnergy,
    start_ms: int,
    end_ms: int,
    *,
    margin_ms: int = SPAN_MARGIN_MS,
) -> float | None:
    """Во сколько раз внутри слова тише, чем в его окрестности (0..1).

    Отвечает на вопрос «а тайминг ASR вообще указывает на звук?». Окно мьюта
    считается из таймингов ASR, и если границы слова уехали в тишину, мат рядом
    останется слышен: подгонка по аудио умеет только сужать окно. Спасти такое
    окно может лишь операция, делающая его ШИРЕ — расширение назад
    (`config.mute_expand_on_silence_ms`; перенос по замеру выключен) — поэтому «в
    границах слова тишина» ещё не значит «мат не заглушён», см.
    REANCHORED_VOICE_PCT.
    Отличить уехавший тайминг от «слово сказано тихо» можно только сравнением с
    окрестностью: см. SILENT_SPAN_RATIO.

    None — считать не по чему (нет кадров или вокруг тоже тишина).
    """
    if end_ms <= start_ms:
        return None
    inside, _ = energy.rms_frames(start_ms, end_ms)
    around, _ = energy.rms_frames(max(0, start_ms - margin_ms), end_ms + margin_ms)
    if not inside or not around:
        return None
    near_peak = max(around)
    if near_peak <= 0.0:
        return None
    return max(inside) / near_peak


def loud_span(
    energy: WavEnergy,
    start_ms: int,
    end_ms: int,
    *,
    margin_ms: int = SPAN_MARGIN_MS,
) -> tuple[int, int] | None:
    """Где в окрестности слова реально звучит речь — разбег громких кадров.

    Нужно для отчёта: когда тайминг слова уехал в тишину, показываем, куда
    именно, иначе «мьют стоит не на звуке» невозможно проверить на слух.
    """
    frames, base_ms = energy.rms_frames(max(0, start_ms - margin_ms), end_ms + margin_ms)
    if not frames:
        return None
    peak = max(frames)
    if peak <= 0.0:
        return None
    hot = [i for i, v in enumerate(frames) if v >= peak * LOUD_SPAN_RATIO]
    if not hot:
        return None
    return base_ms + hot[0] * FRAME_MS, base_ms + (hot[-1] + 1) * FRAME_MS


# --- дыры ASR: громко, но ни одного слова ------------------------------------


@dataclass
class Hole:
    start_ms: int
    end_ms: int
    voiced: float

    @property
    def duration_ms(self) -> int:
        return self.end_ms - self.start_ms


def find_holes(
    energy: WavEnergy,
    words: Sequence[Word],
    thr: float,
    *,
    min_ms: int = 1200,
    min_voiced: float = 0.35,
) -> list[Hole]:
    """Промежутки без слов, в которых звук всё-таки есть."""
    total = energy.duration_ms
    if total <= 0 or thr <= 0.0:
        return []
    gaps: list[tuple[int, int]] = []
    cursor = 0
    for word in words:
        if word.start_ms - cursor >= min_ms:
            gaps.append((cursor, word.start_ms))
        cursor = max(cursor, word.end_ms)
    if total - cursor >= min_ms:
        gaps.append((cursor, total))

    holes: list[Hole] = []
    for start, end in gaps:
        frames, _ = energy.rms_frames(start, min(end, total))
        if not frames:
            continue
        voiced = sum(1 for v in frames if v >= thr) / len(frames)
        if voiced >= min_voiced:
            holes.append(Hole(start, min(end, total), voiced))
    return holes


# --- ground_truth.txt: то, что слышит человек --------------------------------


@dataclass
class TruthPoint:
    start_ms: int
    end_ms: int
    note: str
    line_no: int

    def label(self) -> str:
        span = fmt_ms(self.start_ms)
        if self.end_ms > self.start_ms:
            span += f"–{fmt_ms(self.end_ms)}"
        return f"{span} {self.note}".strip()


def parse_timestamp(token: str) -> int | None:
    """`SS.mmm` / `MM:SS.mmm` / `HH:MM:SS.mmm` → мс. Не разобрали — None."""
    parts = token.replace(",", ".").split(":")
    if not parts or len(parts) > 3:
        return None
    try:
        seconds = float(parts[-1])
        minutes = int(parts[-2]) if len(parts) >= 2 else 0
        hours = int(parts[-3]) if len(parts) == 3 else 0
    except ValueError:
        return None
    if seconds < 0 or minutes < 0 or hours < 0:
        return None
    return int(round((hours * 3600 + minutes * 60 + seconds) * 1000))


def parse_ground_truth(path: Path) -> tuple[list[TruthPoint], list[str]]:
    """Разбор ground_truth.txt. Возвращает (точки, непонятые строки)."""
    points: list[TruthPoint] = []
    problems: list[str] = []
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        head, _, note = line.partition(" ")
        # Интервал пишется как 01:23.4-01:24.1; знак минуса внутри первого токена.
        left, dash, right = head.partition("-")
        start = parse_timestamp(left)
        end = parse_timestamp(right) if dash and right else start
        if start is None or end is None:
            problems.append(f"строка {line_no}: {raw.strip()}")
            continue
        points.append(
            TruthPoint(start, max(start, end), note.strip(), line_no)
        )
    points.sort(key=lambda p: p.start_ms)
    return points, problems


# --- разбор прогона ----------------------------------------------------------

SILENCE_FLAG_MS = 250
"""С какой тишины на краю мьюта это уже видно глазом на таймлайне."""
LOW_FILL = 0.35
LONG_MUTE_MS = 1500


class AuditError(RuntimeError):
    pass


@dataclass
class MuteAudit:
    mute: dict[str, Any]
    start_ms: int
    end_ms: int
    words: list[Word]
    fit: EnergyFit | None
    flags: list[str] = field(default_factory=list)

    @property
    def duration_ms(self) -> int:
        return self.end_ms - self.start_ms

    @property
    def word_text(self) -> str:
        return str(self.mute.get("word") or "").strip()


@dataclass
class Missed:
    word: Word
    roots: list[str]
    known_stem: str | None
    kind: str  # core | spicy
    mute_span: tuple[int, int] | None = None
    """Перекрывающий слово мьют (самый широкий по перекрытию), если он есть."""
    covered_ms: int = 0
    voiced_outside: float | None = None
    """Доля звука вне мьюта в границах слова: см. `uncovered_voice`."""
    peak_covered: bool | None = None
    """Накрыт ли мьютом самый громкий кадр слова: см. `loudest_ms`."""
    span_ratio: float | None = None
    """Пик внутри слова относительно окрестности: см. `span_voice_ratio`."""
    voice_span: tuple[int, int] | None = None
    """Где рядом реально звучит речь (когда тайминг слова уехал в тишину)."""
    voice_covered: bool | None = None
    """Накрыл ли мьют этот звук (когда тайминг уехал): значит окно дотянулось."""

    @property
    def diagnosis(self) -> str:
        """`asr_timing_off` — границы слова указывают в тишину, звук рядом, и на
        звук ничего не встало: мат слышен (подгонка умеет только сужать окно, а
        расширение назад до звука не дотянулось — см. `reanchor` и
        `expand_start_ms` в decisions.json).
        `reanchored` — тайминг слова тоже уехал в тишину, но мьют накрывает
        измеренный звук: это НЕ пропуск, а починенный случай. Имя класса
        историческое — сейчас окно дотягивается расширением назад
        (`mute_expand_on_silence_ms`), перенос по замеру выключен.
        `no_mute` — мьюта на слове нет вовсе, дырка до/в детекции.
        `trimmed_off` — мьют есть, но мат слышно: вне мьюта осталась речь или
        самое громкое место слова.
        `trimmed_to_speech` — мьют есть, вне него тишина: тайминг ASR длиннее
        самого слова, и это не пропуск (обвинять пайплайн тут не за что).
        `partial` — мьют есть, но звук не проверен (нет кэша аудио).
        """
        if self.span_ratio is not None and self.span_ratio < SILENT_SPAN_RATIO:
            return "reanchored" if self.voice_covered else "asr_timing_off"
        if self.mute_span is None:
            return "no_mute"
        if self.voiced_outside is None and self.peak_covered is None:
            return "partial"
        if self.peak_covered is False:
            return "trimmed_off"
        return (
            "trimmed_off"
            if (self.voiced_outside or 0.0) >= VOICE_LEFT_PCT
            else "trimmed_to_speech"
        )

    @property
    def real(self) -> bool:
        """Настоящий пропуск: мата не закрыли или закрыли не до конца."""
        return self.diagnosis not in ("trimmed_to_speech", "reanchored")


@dataclass
class EditFinding:
    """Одна ручная правка против решения программы (журнал `edit_journal`).

    `kind`: added — человек добавил мьют там, где программы не было;
    removed / off — снял (удалил или перевёл в 'rejected');
    moved — сдвинул границы; on — включил обратно снятый программой.
    """

    kind: str
    start_ms: int
    end_ms: int
    mute: dict[str, Any]
    words: list[Word]
    roots: list[str]
    known_stem: str | None
    verdict: str
    base_start_ms: int | None = None
    base_end_ms: int | None = None
    fit: EnergyFit | None = None
    base_fit: EnergyFit | None = None
    neighbour_gap_ms: int | None = None
    """Зазор до ближайшего мьюта ПРОГРАММЫ (0 — окна пересекаются), None — если
    базовых мьютов нет. Нужен только добавленным мьютам: он отличает «человек
    раздвинул наше окно» от «мы не услышали мат» (см. NEIGHBOUR_GAP_MS)."""
    voiced: float | None = None
    """Доля озвученных кадров в окне правки по АБСОЛЮТНОМУ порогу (`voiced_ratio`).
    У снятого мьюта это окно ПРОГРАММЫ: низкое значение = мьют стоял на тишине."""
    base_voiced: float | None = None
    """То же для окна программы ДО сдвига (только у moved)."""
    neighbour_voiced: float | None = None
    """То же для ближайшего мьюта программы (только у added, рядом с
    `neighbour_gap_ms`): если тот стоял на тишине, человек не раздвигал наше
    окно, а переставил его на звук."""

    @property
    def on_silence(self) -> bool:
        """Окно самой правки стоит на тишине по абсолютному порогу."""
        return on_silence(self.voiced)

    @property
    def d_start_ms(self) -> int:
        return self.start_ms - (self.base_start_ms if self.base_start_ms is not None else self.start_ms)

    @property
    def d_end_ms(self) -> int:
        return self.end_ms - (self.base_end_ms if self.base_end_ms is not None else self.end_ms)


@dataclass
class AuditResult:
    decisions_path: Path
    doc: dict[str, Any]
    words: list[Word]
    mutes: list[MuteAudit]
    missed: list[Missed]
    holes: list[Hole]
    audio_path: Path | None
    audio_ok: bool
    truth_path: Path | None
    truth: list[TruthPoint]
    truth_problems: list[str]
    truth_hits: list[tuple[TruthPoint, MuteAudit]] = field(default_factory=list)
    truth_misses: list[TruthPoint] = field(default_factory=list)
    truth_extras: list[MuteAudit] = field(default_factory=list)
    tolerance_ms: int = 1000
    # Журнал ручных правок (<decisions>.edits.json). Пусто — журнал выключен
    # или человек ничего не менял.
    edits_path: Path | None = None
    journal: dict[str, Any] | None = None
    edits: list[EditFinding] = field(default_factory=list)
    edit_notes: list[str] = field(default_factory=list)
    """Компактные строки по слоям, где разбор не текстовый: тишина и оффтоп."""


def _seconds_to_ms(value: Any) -> int:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return int(round(float(value) * 1000))
    return 0


def _resolve(value: Any, base: Path) -> Path | None:
    if not value:
        return None
    path = Path(str(value))
    return path if path.is_absolute() else (base / path)


def _words_in(words: Sequence[Word], start_ms: int, end_ms: int) -> list[Word]:
    return [w for w in words if w.end_ms > start_ms and w.start_ms < end_ms]


EDGE_WORD_TOL_MS = 60
"""Насколько ASR может расходиться со звуком, прежде чем это «слово вылезло»."""
NEIGHBOR_GAP_MS = 150
"""Если следующее слово начинается позже, чем через столько после границы, то
звук за границей — это ХВОСТ ТЕКУЩЕГО слова, а не начало соседнего."""


def _edge_reasons(
    item: MuteAudit, words: Sequence[Word], core: Sequence[Word]
) -> list[str]:
    """Что означает «речь идёт через границу мьюта».

    Сама непрерывность энергии на границе не значит ничего: в беглой речи слова
    идут вплотную, и «речь на левой границе» — это обычно предыдущее слово.
    Флаг ставим ТОЛЬКО когда звук за границей не объясняется соседним словом:
    либо ASR считает матерное слово длиннее мьюта, либо соседнего слова там нет.
    """
    fit = item.fit
    if fit is None:
        return []
    out: list[str] = []
    core_ids = {w.idx for w in core}

    if fit.hot_right:
        last_end = max((w.end_ms for w in core), default=None)
        neighbor = any(
            w.idx not in core_ids
            and w.start_ms < item.end_ms + NEIGHBOR_GAP_MS
            and w.end_ms > item.end_ms
            for w in words
        )
        if last_end is not None and last_end > item.end_ms + EDGE_WORD_TOL_MS:
            out.append(
                f"мат по ASR тянется до {fmt_ms(last_end)}, мьют кончается раньше "
                "— хвост слова слышно"
            )
        elif not neighbor:
            out.append(
                "за правой границей звук есть, а слова там нет — похоже на "
                "недорезанный хвост"
            )

    if fit.hot_left:
        first_start = min((w.start_ms for w in core), default=None)
        neighbor = any(
            w.idx not in core_ids
            and w.end_ms > item.start_ms - NEIGHBOR_GAP_MS
            and w.start_ms < item.start_ms
            for w in words
        )
        if first_start is not None and first_start < item.start_ms - EDGE_WORD_TOL_MS:
            out.append(
                f"мат по ASR начинается в {fmt_ms(first_start)} — раньше мьюта, "
                "начало слышно"
            )
        elif not neighbor:
            out.append(
                "за левой границей звук есть, а слова там нет — похоже на "
                "недорезанное начало"
            )
    return out


def _flags_for(
    item: MuteAudit, words: Sequence[Word], normalizer: RussianNormalizer
) -> list[str]:
    flags: list[str] = []
    core = [w for w in item.words if w.core]
    if not core:
        # Слово мьюта («ебтвою», «хули») могло быть склеено или взято из словаря —
        # тогда в перекрытых токенах корня нет, а мат есть.
        recorded = roots_in(normalizer.normalize(item.word_text))
        spicy = [w for w in item.words if w.spicy]
        if recorded:
            pass
        elif spicy:
            flags.append("только грубость на грани, ядра мата нет")
        elif item.words:
            flags.append("НЕТ обсценного корня в перекрытых словах")
        else:
            flags.append("под мьютом вообще нет слов транскрипта")
    if str(item.mute.get("match_type")) in {"short_fragment", "censored"}:
        flags.append(f"матч по обрывку/цензуре ({item.mute.get('match_type')})")
    if any(w.timing_source != "word" for w in core):
        flags.append("тайминги унаследованы от сегмента")
    if item.duration_ms >= LONG_MUTE_MS:
        flags.append(f"длинный мьют {item.duration_ms} мс (похоже на потолок)")
    fit = item.fit
    if fit is not None:
        if fit.tail_ms >= SILENCE_FLAG_MS:
            flags.append(f"хвост тишины {fit.tail_ms} мс")
        if fit.head_ms >= SILENCE_FLAG_MS:
            flags.append(f"тишина в голове {fit.head_ms} мс")
        if fit.fill < LOW_FILL:
            flags.append(f"речь занимает лишь {fit.fill * 100:.0f}% окна")
    flags.extend(_edge_reasons(item, words, core))
    if str(item.mute.get("status")) == "review":
        flags.append("status=review (в экспорт не попадёт без решения)")
    return flags


# --- журнал ручных правок ----------------------------------------------------


def _roots_of(words: Sequence[Word]) -> tuple[list[str], str | None]:
    """Обсценные корни во всех словах отрезка + знал ли корень сам детектор."""
    roots: list[str] = []
    known: str | None = None
    for word in words:
        for root in word.core + word.spicy:
            if root not in roots:
                roots.append(root)
        known = known or match_root(word.norm)
    return roots, known


ADDED_DIAGNOSES: tuple[str, ...] = (
    "asr_timing_off",
    "no_words",
    "lost_after_detection",
    "spicy_only",
    "neighbour_window",
    "asr_misheard",
)
"""Почему человеку пришлось ставить мьют руками — взаимоисключающие причины.
Ключи (а не текст вердикта) — то, что считает `twitch-cut eval`."""

NEIGHBOUR_GAP_MS = 700
"""Насколько близко к мьюту машины должен стоять добавленный мьют, чтобы считать
его не пропуском мата, а раздвинутым окном.

Замерено на реальном прогоне (15 мин, 25 добавленных мьютов). Из 16, которые
числились «ASR услышал не то», 12 стоят в 6–662 мс от мьюта машины, а текст под
ними невинный («это вообще», «даже», «склеили», «распоряжении») — человек не
нашёл там мат, он раздвинул или разбил соседний мьют. Настоящие пропуски лежат
далеко: 1718, 1937, 3278, 3855 и 12006 мс. Зазор 700 мс делит эти две группы без
пересечения, и разница между ними больше порядка — не подгонка под данные.

Разделять их обязательно: иначе отчёт советует чинить качество ASR (16 случаев!)
там, где чинить надо padding и склейку окон, а единственный настоящий пропуск
мата тонет в этой пачке."""

MATCH_TOLERANCE_MS = 250
"""Допуск сопоставления «мьют человека ↔ мьют машины» по ЗАЗОРУ между окнами.

Нужен там, где журнала правок нет и окна сопоставляются геометрически — сверка
двух прогонов (`tools/sensitivity_levels.py`), где `id` мьюта позиционный и
поэтому бесполезен. Правило «есть пересечение» для этого не годится: окна мата
шириной 130–470 мс, и расхождение границ на 150 мс уничтожает пересечение
целиком — одно и то же слово становится сразу пропуском (у человека) и ложным (у
машины), двойная плата за ошибку границы.

Замерено 2026-08-17 на трёх правленых записях (254 мьюта человека, 19 без
пересечения): зазоры до ближайшего окна машины — 19, 22, 45, 46, 50, 57, 61, 63,
76, 86, 98, 138, 183, 233, 251, 401, 662, 1807, 12006 мс. Медиана 86 мс, в
пределах 250 мс — 14 из 19. То есть большинство «пропусков» этой метрики было
расхождением границ, а не ненайденным словом.

Почему 250, а не 700 как у `NEIGHBOUR_GAP_MS`: тот порог отвечает на вопрос
«человек раздвигал наше окно?», его читает человек, и он может быть щедрым.
Здесь порогом считается ПОЛНОТА, а 700 мс от окна шириной 130 мс — это уже
соседнее слово."""

REMOVED_DIAGNOSES: tuple[str, ...] = (
    "empty",
    "on_silence",
    "deliberate",
    "false_positive",
)
"""Почему человек снял мьют программы."""

ADDED_LABELS: dict[str, str] = {
    "no_words": "речь не распознана (VAD/ASR)",
    "asr_timing_off": "окно машины рядом стояло на ТИШИНЕ — уехал тайминг ASR",
    "lost_after_detection": "корень детектор знал — потеряно ПОСЛЕ детекции",
    "spicy_only": "только грубость из SPICY_ROOTS — не пропуск, а выбор",
    "neighbour_window": "вплотную к мьюту машины — раздвинутое окно, не пропуск",
    "asr_misheard": "мата в тексте нет — ASR услышал не то",
}
REMOVED_LABELS: dict[str, str] = {
    "empty": "под мьютом не было слов",
    "on_silence": "мьют стоял на ТИШИНЕ — глушить было нечего",
    "deliberate": "мат есть и слышен, снято осознанно",
    "false_positive": "ЛОЖНОЕ срабатывание — корня нет",
}

ADDED_CHARGE: dict[str, str] = {
    "no_words": "missed",
    "asr_misheard": "missed",
    "lost_after_detection": "missed",
    "asr_timing_off": "misplaced",
    "neighbour_window": "misplaced",
    "spicy_only": "policy",
}
"""Чего добавленный мьют стоит ПОЛНОТЕ. Под словом «пропуск» лежат три разные
вещи, и складывать их в одно число — значит мерить сумму несравнимого.

* `missed` — слова машина не нашла. Сюда идут и «ASR услышал не то», и «речь не
  распознана»: это ограничение модели, а ограничение модели — ровно то, что
  полнота и меряет. Вынести их из полноты значит получить число, которое растёт
  от того, что модель стала слышать хуже.
* `misplaced` — слово машина нашла, но окно встало не там (мьют машины рядом:
  либо на тишине — уехал тайминг, либо человек его раздвинул). Поиск сработал;
  платить за это полнотой нельзя, ошибка живёт в медианах `d_start`/`d_end`.
* `policy` — грубость из `SPICY_ROOTS`, которую детектор не мьютит ОСОЗНАННО.
  Не ошибка вообще: такой мьют уходит из полноты целиком, а не в пропуски.

Ключи — из `classify_added`, поэтому текст в `audit.md` и число в
`twitch-cut eval` объясняют одно и то же (см. ADDED_LABELS)."""

CHARGE_LABELS: dict[str, str] = {
    "missed": "слово НЕ НАЙДЕНО — пропуск поиска",
    "misplaced": "слово найдено, окно встало не там — ошибка границ",
    "policy": "не пропуск: детектор такое не мьютит осознанно",
}


def classify_added(
    words: Sequence[Word],
    roots: Sequence[str],
    known: str | None,
    neighbour_gap_ms: int | None = None,
    neighbour_voiced: float | None = None,
) -> str:
    """Ключ причины для добавленного человеком мьюта (см. ADDED_DIAGNOSES).

    `neighbour_gap_ms` — зазор до ближайшего мьюта машины (None, если считать не
    из чего), `neighbour_voiced` — озвученность ТОГО окна по абсолютному порогу.
    Вместе они разделяют три разных случая рядом стоящего машинного мьюта:

    * машинное окно на звуке, мата в тексте нет → человек раздвинул НАШЕ окно
      (`neighbour_window`, лечится padding и склейкой);
    * машинное окно на ТИШИНЕ → мы глушили воздух, а человек переставил мьют на
      речь (`asr_timing_off`: машина такое окно помечает янтарным и расширяет
      НАЗАД — mute_expand_on_silence_ms, — но переносить его за человека по
      замеру не берётся, mute_reanchor_move);
    * машинного окна рядом нет → мы правда не услышали (`asr_misheard`).

    Тишина проверяется РАНЬШЕ всех остальных признаков, даже раньше пустого
    транскрипта: это измерение, а не вывод из текста, и оно объясняет правку
    целиком. `no_words` в таком месте прямо врёт («речь не распознана») — речь
    распознана, уехали её тайминги, и доказательство этому наш же мьют рядом,
    глушащий тишину. `lost_after_detection` («потерялось после детекции: статус,
    группировка, потолки или подгонка») и `neighbour_window` («лечится padding»)
    тоже зовут не туда: границы окна тут не при чём, окно стояло не в том месте.
    """
    if (
        neighbour_gap_ms is not None
        and neighbour_gap_ms <= NEIGHBOUR_GAP_MS
        and on_silence(neighbour_voiced)
    ):
        return "asr_timing_off"
    if not words:
        return "no_words"
    if roots and known:
        return "lost_after_detection"
    if roots:
        return "spicy_only"
    if neighbour_gap_ms is not None and neighbour_gap_ms <= NEIGHBOUR_GAP_MS:
        return "neighbour_window"
    return "asr_misheard"


def classify_removed(
    words: Sequence[Word], roots: Sequence[str], voiced: float | None = None
) -> str:
    """Ключ причины для снятого человеком мьюта (см. REMOVED_DIAGNOSES).

    `voiced` — озвученность окна мьюта по АБСОЛЮТНОМУ порогу (`voiced_ratio`).
    Без неё «мат в тексте есть» автоматически означало «снято осознанно, детектор
    не виноват» — и мьют, простоявший всю свою длину в тишине, отчёт записывал в
    выбор человека. На реальном прогоне так потерялась ровно та правка, которая
    и была настоящим дефектом пайплайна.

    Ложное срабатывание проверяется раньше тишины: если обсценного корня нет
    вовсе, чинить надо словарь/нормализацию, а не тайминги.
    """
    if not words:
        return "empty"
    if not roots:
        return "false_positive"
    if on_silence(voiced):
        return "on_silence"
    return "deliberate"


def _verdict_added(
    words: Sequence[Word],
    roots: Sequence[str],
    known: str | None,
    neighbour_gap_ms: int | None = None,
    neighbour_voiced: float | None = None,
) -> str:
    """Человек поставил мьют там, где программа ничего не нашла. Почему?

    Диагнозы лечатся в разных местах: пустой транскрипт → ASR/VAD, окно машины
    рядом стояло в тишине → тайминги ASR и перенос окна, корень ядра → потери
    ПОСЛЕ детекции, только грубость → сознательный выбор детектора, мьют вплотную
    к машинному → padding и склейка окон, мата в тексте нет → ASR услышал не то.
    Их ключи — `classify_added`, по ним же считает метрику `twitch-cut eval`,
    поэтому текст и число всегда объясняют одно и то же.

    Классов «дыра словаря» и «поиск смотрит только на префикс» здесь больше нет:
    словарь и матчер у детектора и у отчёта общие (profanity.roots_in), поэтому
    корень, который видит отчёт, детектор видит тоже — by construction.
    """
    diag = classify_added(words, roots, known, neighbour_gap_ms, neighbour_voiced)
    if diag == "asr_timing_off":
        return (
            f"мьют машины рядом ({neighbour_gap_ms} мс) стоял на ТИШИНЕ "
            f"({(neighbour_voiced or 0.0) * 100:.0f}% озвученных кадров) — мат мы "
            "нашли, но заглушили воздух: тайминги слова у ASR уехали. Ты переставил "
            "окно на речь; машина такое окно помечает янтарным и расширяет НАЗАД "
            "(mute_expand_on_silence_ms), но не переносит — см. mute_reanchor_move. "
            "Если звук ушёл ВПЕРЁД от разметки, расширение назад его не достаёт. "
            "Padding и словарь тут не при чём"
        )
    if diag == "no_words":
        return (
            "в транскрипте здесь НЕТ СЛОВ — речь не распознана (VAD/ASR). "
            "Детектор физически не мог тут ничего найти"
        )
    if diag == "lost_after_detection":
        return (
            f"детектор корень ЗНАЛ (`{known}`) — мьют потерялся ПОСЛЕ детекции: "
            "статус review, группировка, потолки или подгонка границ"
        )
    if diag == "spicy_only":
        return (
            f"в тексте только грубость на грани ({', '.join(roots)}) — она в "
            "SPICY_ROOTS, детектор её НЕ мьютит осознанно: это выбор пользователя, "
            "а не пропуск"
        )
    if diag == "neighbour_window":
        return (
            f"мата в тексте нет, но мьют машины рядом — {neighbour_gap_ms} мс: "
            "человек раздвинул или разбил ЕГО окно, а не нашёл новый мат. "
            "Лечится padding и склейкой, качество ASR тут не при чём"
        )
    return (
        "мата в тексте нет — ASR УСЛЫШАЛ НЕ ТО. По транскрипту такое не ловится "
        "ни детектором, ни этим отчётом; лечится спотингом по посториорам "
        "(spot_posteriors) или ground_truth"
    )


def _verdict_removed(
    words: Sequence[Word],
    roots: Sequence[str],
    mute: dict[str, Any],
    voiced: float | None = None,
) -> str:
    """Человек снял мьют, который поставила программа."""
    match = str(mute.get("match_type") or "?")
    banword = str(mute.get("banword") or "?")
    diag = classify_removed(words, roots, voiced)
    if diag == "empty":
        return "под мьютом вообще нет слов — мьют стоял на пустом месте"
    if diag == "on_silence":
        return (
            f"мат в тексте есть ({', '.join(roots)}), но в окне мьюта ТИШИНА "
            f"({(voiced or 0.0) * 100:.0f}% озвученных кадров по абсолютному порогу) "
            "— мьют глушил воздух, а сам мат остался слышен. Снято правильно: "
            "виноваты тайминги ASR. Машина такое окно помечает янтарным и "
            "расширяет назад (mute_expand_on_silence_ms), но не переносит — см. "
            "mute_reanchor_move"
        )
    if diag == "deliberate":
        return (
            f"мат в тексте ЕСТЬ ({', '.join(roots)}) и звук в окне есть — снято "
            "осознанно (уже зацензурено, не слышно, не нужно), детектор не виноват"
        )
    return (
        f"обсценного корня в перекрытых словах НЕТ — ложное срабатывание "
        f"(match={match}, banword={banword}): смотреть нормализацию/лемматизацию"
    )


def _verdict_moved(finding: EditFinding) -> str:
    """Человек сдвинул границы. Диагноз — из направления сдвига и из звука."""
    parts: list[str] = []
    d_start, d_end = finding.d_start_ms, finding.d_end_ms
    if d_start < 0:
        parts.append(f"начало сдвинуто РАНЬШЕ на {abs(d_start)} мс — мьют опаздывал")
    elif d_start > 0:
        parts.append(f"начало сдвинуто ПОЗЖЕ на {d_start} мс — мьют начинался рано")
    if d_end > 0:
        parts.append(f"конец сдвинут ПОЗЖЕ на {d_end} мс — хвост мата было слышно")
    elif d_end < 0:
        parts.append(f"конец сдвинут РАНЬШЕ на {abs(d_end)} мс — мьют съедал лишнее")
    # Сдвиг С тишины НА звук — это не «неточная граница», а уехавший тайминг ASR:
    # окно программы не глушило ничего. Мерим абсолютным порогом, потому что
    # относительный (`fit.fill`) в тишине находит речь в шуме.
    if on_silence(finding.base_voiced) and not finding.on_silence:
        parts.append(
            f"окно программы стояло на ТИШИНЕ ({(finding.base_voiced or 0.0) * 100:.0f}% "
            f"озвученных кадров), твоё — на речи ({(finding.voiced or 0.0) * 100:.0f}%): "
            "уехали тайминги ASR, а не границы (padding тут не при чём). Машина "
            "такое окно помечает янтарным и расширяет назад, но не переносит — "
            "см. mute_reanchor_move"
        )
    trim = _trim_pair(finding.mute)
    if trim and (d_start < 0 or d_end > 0):
        parts.append(f"подгонка по звуку срезала {trim[0]}/{trim[1]} мс — возможно, лишнего")
    return "; ".join(parts) or "границы совпали с точностью до округления"


def _trim_pair(mute: dict[str, Any]) -> tuple[int, int] | None:
    left = mute.get("audio_trim_start_ms")
    right = mute.get("audio_trim_end_ms")
    if not isinstance(left, int) or not isinstance(right, int):
        return None
    return (left, right) if (left or right) else None


def _machine_spans(journal: dict[str, Any]) -> list[tuple[int, int]]:
    """Окна мьютов, которые поставила ПРОГРАММА (снимок baseline журнала).

    Только `accepted`: review-мьют в Vegas не уезжает, звук на нём остаётся, и
    считать соседний ручной мьют «раздвинутым окном» тут нельзя — там человек
    закрывал то, что программа как раз НЕ закрыла."""
    baseline = journal.get("baseline")
    mutes = (baseline or {}).get("mutes") if isinstance(baseline, dict) else None
    spans: list[tuple[int, int]] = []
    for mute in mutes or []:
        if not isinstance(mute, dict):
            continue
        if str(mute.get("status") or "accepted") != "accepted":
            continue
        start = _seconds_to_ms(mute.get("start"))
        end = _seconds_to_ms(mute.get("end"))
        if end > start:
            spans.append((start, end))
    return sorted(spans)


def _machine_gap(span: tuple[int, int], finding: EditFinding) -> int:
    """Зазор между окном правки и окном программы (0 — пересекаются)."""
    return max(0, max(span[0] - finding.end_ms, finding.start_ms - span[1]))


def _gap_to_machine(
    spans: Sequence[tuple[int, int]], finding: EditFinding
) -> int | None:
    """Зазор от правки до ближайшего мьюта программы: 0 — окна пересекаются,
    None — сравнивать не с чем (в снимке мьютов нет)."""
    if not spans:
        return None
    return min(_machine_gap(span, finding) for span in spans)


def _nearest_machine(
    spans: Sequence[tuple[int, int]], finding: EditFinding
) -> tuple[int, int] | None:
    """Само окно ближайшего мьюта программы — чтобы измерить, был ли в НЁМ звук."""
    if not spans:
        return None
    return min(spans, key=lambda span: _machine_gap(span, finding))


def _neighbour_context(
    finding: EditFinding,
    machine: Sequence[tuple[int, int]],
    energy: WavEnergy | None,
    threshold: float,
) -> None:
    """Заполнить у ДОБАВЛЕННОГО мьюта зазор до ближайшего окна программы и
    озвученность ТОГО окна: вместе они отличают «раздвинул наше окно» от «мы
    глушили тишину, а он переставил мьют на звук» (см. `classify_added`)."""
    finding.neighbour_gap_ms = _gap_to_machine(machine, finding)
    span = _nearest_machine(machine, finding)
    if span is not None and energy is not None:
        finding.neighbour_voiced = voiced_ratio(energy, span[0], span[1], threshold)


def _edit_findings(
    journal: dict[str, Any],
    doc: dict[str, Any],
    words: Sequence[Word],
    energy: WavEnergy | None,
    threshold: float = 0.0,
) -> tuple[list[EditFinding], list[str]]:
    """Итоговая разница «пайплайн → текущий decisions.json» с диагнозом.

    Считаем от BASELINE журнала, а не от предыдущего сохранения: интересует
    расхождение человека с ПРОГРАММОЙ, а не история метаний.

    `threshold` — абсолютный порог речи (`mute_refine.global_threshold`), один на
    весь отчёт: им же диагнозы проверяют, стоял ли мьют на звуке.
    """
    net = net_changes(journal, doc)
    layer = net.get("mutes")
    layer = layer if isinstance(layer, dict) else {}
    out: list[EditFinding] = []
    machine = _machine_spans(journal)

    def neighbour(finding: EditFinding) -> None:
        _neighbour_context(finding, machine, energy, threshold)

    for entry in layer.get("added") or []:
        item = entry.get("after") or {}
        finding = _finding("added", item, energy, words, threshold=threshold)
        if finding is not None:
            neighbour(finding)
            finding.verdict = _verdict_added(
                finding.words,
                finding.roots,
                finding.known_stem,
                finding.neighbour_gap_ms,
                finding.neighbour_voiced,
            )
            out.append(finding)

    for entry in layer.get("removed") or []:
        item = entry.get("before") or {}
        finding = _finding("removed", item, energy, words, threshold=threshold)
        if finding is not None:
            finding.verdict = _verdict_removed(
                finding.words, finding.roots, item, finding.voiced
            )
            out.append(finding)

    for entry in layer.get("moved") or []:
        item = entry.get("after") or {}
        base = entry.get("before") or {}
        if _relocated(base, item):
            out.extend(
                _relocation_findings(base, item, energy, words, machine, threshold)
            )
            continue
        finding = _finding("moved", item, energy, words, base=base, threshold=threshold)
        if finding is not None:
            finding.verdict = _verdict_moved(finding)
            out.append(finding)

    for entry in layer.get("status") or []:
        item = entry.get("after") or {}
        to = str(entry.get("status_to") or "")
        frm = str(entry.get("status_from") or "")
        kind = "off" if to == "rejected" else "on"
        finding = _finding(kind, item, energy, words, threshold=threshold)
        if finding is None:
            continue
        if kind == "off":
            finding.verdict = "снят статусом (" + f"{frm} → {to}): " + _verdict_removed(
                finding.words, finding.roots, item, finding.voiced
            )
        else:
            finding.verdict = _verdict_enabled(frm, to, item, finding)
        out.append(finding)

    out.sort(key=lambda f: f.start_ms)
    return out, _edit_notes(journal, net)


def _relocated(base: dict[str, Any], item: dict[str, Any]) -> bool:
    """Мьют не подровняли по краю, а ПЕРЕНЕСЛИ в другое место.

    Журнал сопоставляет мьюты по `id`, поэтому «человек перетащил мьют на другое
    слово» приходит тем же видом правки, что и «подвинул границу на 40 мс». Для
    отчёта и метрики это разные вещи: на реальном прогоне 13 таких переносов
    (до 900 мс) дали медиану ошибки границ 116 мс вместо 71 и «худшую» 2.7 с —
    цифра описывала перетаскивания, а не точность окна.

    Критерий простой и не требует порогов на вкус: если у окон нет ОБЩЕГО звука,
    человек глушил другое место.
    """
    start, end = _seconds_to_ms(item.get("start")), _seconds_to_ms(item.get("end"))
    base_start, base_end = _seconds_to_ms(base.get("start")), _seconds_to_ms(base.get("end"))
    if end <= start or base_end <= base_start:
        return False
    return min(end, base_end) <= max(start, base_start)


def _relocation_findings(
    base: dict[str, Any],
    item: dict[str, Any],
    energy: WavEnergy | None,
    words: Sequence[Word],
    machine: Sequence[tuple[int, int]] = (),
    threshold: float = 0.0,
) -> list[EditFinding]:
    """Перенос = машина заглушила не то (ложное) + человек нашёл пропуск.

    Именно так это и надо лечить: два разных диагноза в двух разных местах
    пайплайна, а не один «сдвиг границ».

    Зазор до мьютов машины половине «добавлено» нужен ровно так же, как обычному
    добавленному мьюту, и считается по тому же снимку baseline. Без него перенос
    на 200 мс в сторону выглядел в метрике как «ASR услышал не то»: на реальном
    прогоне так потерялись 11 из 12 случаев раздвинутого окна — диагноз звал
    чинить распознавание там, где человек просто перетащил окно на соседний слог.
    Свой прежний бокс мьюта в снимке остался, поэтому зазор чаще всего мал — это
    и есть правда про перенос.
    """
    out: list[EditFinding] = []
    removed = _finding("removed", base, energy, words, threshold=threshold)
    if removed is not None:
        removed.verdict = "мьют перенесён в другое место: " + _verdict_removed(
            removed.words, removed.roots, base, removed.voiced
        )
        out.append(removed)
    added = _finding("added", item, energy, words, threshold=threshold)
    if added is not None:
        _neighbour_context(added, machine, energy, threshold)
        added.verdict = "мьют перенесён сюда: " + _verdict_added(
            added.words,
            added.roots,
            added.known_stem,
            added.neighbour_gap_ms,
            added.neighbour_voiced,
        )
        out.append(added)
    return out


def _finding(
    kind: str,
    item: dict[str, Any],
    energy: WavEnergy | None,
    words: Sequence[Word],
    *,
    base: dict[str, Any] | None = None,
    threshold: float = 0.0,
) -> EditFinding | None:
    start = _seconds_to_ms(item.get("start"))
    end = _seconds_to_ms(item.get("end"))
    if end <= start:
        return None
    base_start = _seconds_to_ms(base.get("start")) if base else None
    base_end = _seconds_to_ms(base.get("end")) if base else None
    # Слова берём по ОБЪЕДИНЕНИЮ окон: при сдвиге объяснение почти всегда лежит
    # в той части, которую человек добавил или отрезал.
    lo = min(start, base_start) if base_start is not None else start
    hi = max(end, base_end) if base_end is not None else end
    inside = _words_in(words, lo, hi)
    roots, known = _roots_of(inside)
    return EditFinding(
        kind=kind,
        start_ms=start,
        end_ms=end,
        mute=item,
        words=list(inside),
        roots=roots,
        known_stem=known,
        verdict="",
        base_start_ms=base_start,
        base_end_ms=base_end,
        fit=fit_window(energy, start, end) if energy else None,
        base_fit=(
            fit_window(energy, base_start, base_end)
            if energy and base_start is not None and base_end is not None
            else None
        ),
        voiced=voiced_ratio(energy, start, end, threshold) if energy else None,
        base_voiced=(
            voiced_ratio(energy, base_start, base_end, threshold)
            if energy and base_start is not None and base_end is not None
            else None
        ),
    )


_LAYER_LABELS = {"silences": "тишина", "offtopic": "оффтоп"}
_KIND_LABELS = {"added": "добавлено", "removed": "убрано", "moved": "сдвинуто", "status": "смена статуса"}


def _plural(n: int, one: str, few: str, many: str) -> str:
    """`5 сегментов` вместо `5 сегментов`-на-всё: отчёт читает человек."""
    tail100, tail10 = n % 100, n % 10
    if 11 <= tail100 <= 14:
        word = many
    elif tail10 == 1:
        word = one
    elif 2 <= tail10 <= 4:
        word = few
    else:
        word = many
    return f"{n} {word}"


def _edit_notes(journal: dict[str, Any], net: dict[str, Any]) -> list[str]:
    """Слои, где текстового диагноза нет: тишина и оффтоп — только счётчики.

    Тишина считается по энергии (не по словам), оффтоп — LLM по репликам;
    объяснять их корнями бессмысленно, но знать, что человек их переделывал,
    полезно: это отдельные пороги в настройках.

    Добавленное делим по провенансу: авто-детект тишины и AI-детект оффтопа
    сыпят сегменты уже ПОСЛЕ базового снимка, и без этого деления их работа
    читалась бы как правка человека.
    """
    notes: list[str] = []
    for layer, label in _LAYER_LABELS.items():
        changes = net.get(layer)
        if not isinstance(changes, dict):
            continue
        manual_added, auto_added = split_added_by_origin(changes.get("added"))
        parts = []
        if manual_added:
            parts.append(f"добавлено вручную: {len(manual_added)}")
        parts += [
            f"{_KIND_LABELS[kind]}: {len(changes.get(kind) or [])}"
            for kind in ("removed", "moved", "status")
            if changes.get(kind)
        ]
        if parts:
            notes.append(f"{label} — " + ", ".join(parts))
        if auto_added:
            notes.append(
                f"{label} — ещё {_plural(len(auto_added), 'сегмент', 'сегмента', 'сегментов')} "
                "насыпал авто-детект уже после обработки: это работа программы, "
                "а не твоя правка"
            )
    baseline = journal.get("baseline")
    refreshed = baseline.get("refreshed") if isinstance(baseline, dict) else None
    if isinstance(refreshed, dict):
        for layer, info in refreshed.items():
            if not isinstance(info, dict):
                continue
            label = _LAYER_LABELS.get(layer, layer)
            notes.append(
                f"базовый снимок «{label}» обновлён отдельно от обработки: "
                f"{info.get('note')} ({info.get('at')})"
            )
    totals = journal.get("totals")
    if isinstance(totals, dict) and totals.get("saves"):
        saves = int(totals.get("saves") or 0)
        # Считаем только заходы человека: сессии `pipeline`/`machine` — это
        # обновления базового снимка, а не правки.
        sessions = sum(
            1
            for s in journal.get("sessions") or []
            if isinstance(s, dict) and s.get("kind") == "save"
        )
        notes.append(
            f"{_plural(saves, 'сохранение', 'сохранения', 'сохранений')} с правками "
            f"(история в журнале: {_plural(sessions, 'сессия', 'сессии', 'сессий')})"
        )
    return notes


def _verdict_enabled(frm: str, to: str, item: dict[str, Any], finding: EditFinding) -> str:
    """Человек включил мьют, который программа отложила или сняла."""
    head = f"включён вручную ({frm or '—'} → {to or '—'})"
    if bool(item.get("needs_review")):
        return (
            f"{head}: детектор сомневался (needs_review) и был НЕ прав — "
            "порог/причина review слишком строгие"
        )
    if finding.roots:
        return f"{head}: мат в тексте есть ({', '.join(finding.roots)})"
    return f"{head}: обсценного корня в тексте нет — доверять только слуху"


def analyze(
    decisions_path: Path,
    *,
    truth_path: Path | None = None,
    tolerance_ms: int = 1000,
    transcript_path: Path | None = None,
    audio_path: Path | None = None,
    edits_path: Path | None = None,
) -> AuditResult:
    decisions_path = Path(decisions_path)
    if decisions_path.is_dir():
        decisions_path = decisions_path / "decisions.json"
    if not decisions_path.exists():
        raise AuditError(f"не нашёл decisions.json: {decisions_path}")
    doc = json.loads(decisions_path.read_text(encoding="utf-8"))
    base = decisions_path.parent
    caches = doc.get("caches") or {}

    tpath = _resolve(transcript_path or caches.get("transcript"), base)
    if tpath is None or not tpath.exists():
        raise AuditError(
            "не нашёл кэш транскрипта (caches.transcript в decisions.json). "
            "Без транскрипта сверять нечего."
        )
    normalizer = RussianNormalizer()
    words = load_words(json.loads(tpath.read_text(encoding="utf-8")), normalizer)

    apath = _resolve(audio_path or caches.get("audio"), base)
    energy = open_energy(apath)
    # Журнал правок: по умолчанию ищем рядом с решениями. Его может не быть —
    # это необязательная часть отчёта.
    jpath = Path(edits_path) if edits_path else journal_path(decisions_path)
    journal = read_journal(jpath) if jpath.exists() else None
    try:
        mutes: list[MuteAudit] = []
        for mute in doc.get("mutes") or []:
            start = _seconds_to_ms(mute.get("start"))
            end = _seconds_to_ms(mute.get("end"))
            item = MuteAudit(
                mute=mute,
                start_ms=start,
                end_ms=end,
                words=_words_in(words, start, end),
                fit=fit_window(energy, start, end) if energy else None,
            )
            item.flags = _flags_for(item, words, normalizer)
            mutes.append(item)

        spans = [(m.start_ms, m.end_ms) for m in mutes]
        missed: list[Missed] = []
        for word in words:
            core, spicy = word.core, word.spicy
            if not core and not spicy:
                continue
            mid = (word.start_ms + word.end_ms) // 2
            if any(s <= mid <= e for s, e in spans):
                continue
            # Мьют может быть — и всё равно не накрывать середину слова: подгонка
            # по аудио режет окно по звуку, а тайминг ASR почти всегда длиннее
            # самого слова. Поэтому смотрим, что осталось вне мьюта: речь (мат
            # слышно) или тишина (претензий нет). Без этого пропуском числился
            # аккуратно поджатый мьют, и диагноз врал про «статус/группировку».
            covers = [(s, e) for s, e in spans if s < word.end_ms and e > word.start_ms]
            covered_ms = sum(
                min(e, word.end_ms) - max(s, word.start_ms) for s, e in covers
            )
            widest = max(covers, key=lambda sp: min(sp[1], word.end_ms) - max(sp[0], word.start_ms), default=None)
            loud_ms = (
                loudest_ms(energy, word.start_ms, word.end_ms) if energy and covers else None
            )
            # Отдельно от мьюта: указывают ли границы самого слова на звук. Если
            # нет — виноват тайминг ASR, а не группировка/потолки, и мьют стоит в
            # тишине независимо от того, «покрыта» ли середина слова.
            ratio = span_voice_ratio(energy, word.start_ms, word.end_ms) if energy else None
            # Куда съехал звук — и накрыл ли его мьют. Второе обязательно: пайплайн
            # расширяет назад окно, стоящее на тишине (а с включённым переносом —
            # двигает его), и тогда окно НЕ совпадает с границами слова (в них
            # тишина). Считать такой случай пропуском значит звать человека править
            # уже починенное.
            voice_span = (
                loud_span(energy, word.start_ms, word.end_ms)
                if energy and ratio is not None and ratio < SILENT_SPAN_RATIO
                else None
            )
            voice_covered = (
                sum(
                    max(0, min(e, voice_span[1]) - max(s, voice_span[0]))
                    for s, e in spans
                )
                >= REANCHORED_VOICE_PCT * (voice_span[1] - voice_span[0])
                if voice_span and voice_span[1] > voice_span[0]
                else None
            )
            missed.append(
                Missed(
                    word=word,
                    roots=core or spicy,
                    known_stem=match_root(word.norm),
                    kind="core" if core else "spicy",
                    mute_span=widest,
                    covered_ms=covered_ms,
                    voiced_outside=(
                        uncovered_voice(energy, word.start_ms, word.end_ms, covers)
                        if energy and covers
                        else None
                    ),
                    peak_covered=(
                        any(s <= loud_ms <= e for s, e in covers)
                        if loud_ms is not None
                        else None
                    ),
                    span_ratio=ratio,
                    voice_span=voice_span,
                    voice_covered=voice_covered,
                )
            )

        # Абсолютный порог речи считается ОДИН раз на отчёт (40 проб по файлу) и
        # уходит и в поиск дыр, и в разбор правок: «тут вообще есть звук» должно
        # означать одно и то же во всех разделах и в пайплайне.
        threshold = global_threshold(energy) if energy else 0.0
        holes = find_holes(energy, words, threshold) if energy else []
        edits, edit_notes = (
            _edit_findings(journal, doc, words, energy, threshold)
            if journal
            else ([], [])
        )
    finally:
        if energy:
            energy.close()

    result = AuditResult(
        decisions_path=decisions_path,
        doc=doc,
        words=words,
        mutes=mutes,
        missed=missed,
        holes=holes,
        audio_path=apath,
        audio_ok=energy is not None,
        truth_path=truth_path,
        truth=[],
        truth_problems=[],
        tolerance_ms=tolerance_ms,
        edits_path=jpath if journal else None,
        journal=journal,
        edits=edits,
        edit_notes=edit_notes,
    )

    if truth_path and Path(truth_path).exists():
        result.truth, result.truth_problems = parse_ground_truth(Path(truth_path))
        matched: set[int] = set()
        for point in result.truth:
            hit = next(
                (
                    m
                    for m in mutes
                    if m.start_ms - tolerance_ms <= point.end_ms
                    and m.end_ms + tolerance_ms >= point.start_ms
                ),
                None,
            )
            if hit is None:
                result.truth_misses.append(point)
            else:
                result.truth_hits.append((point, hit))
                matched.add(id(hit))
        result.truth_extras = [m for m in mutes if id(m) not in matched]
    return result


# --- отчёт -------------------------------------------------------------------


def _fit_text(item: MuteAudit) -> str:
    fit = item.fit
    if fit is None:
        return "аудио не проверено"
    return (
        f"речь {fit.fill * 100:.0f}% окна, голова {fit.head_ms} мс, "
        f"хвост {fit.tail_ms} мс"
    )


def _trim_text(item: MuteAudit) -> str:
    left = item.mute.get("audio_trim_start_ms")
    right = item.mute.get("audio_trim_end_ms")
    if left is None and right is None:
        return "trim —"
    return f"trim {left or 0}/{right or 0} мс"


def _mute_head(item: MuteAudit) -> str:
    word = item.word_text or "?"
    return (
        f"**{fmt_ms(item.start_ms)}–{fmt_ms(item.end_ms)}** ({item.duration_ms} мс) "
        f"«{word}»"
    )


def _detail_block(item: MuteAudit, words: Sequence[Word]) -> list[str]:
    lines = [
        f"- {_mute_head(item)} — {_trim_text(item)}, {_fit_text(item)}, "
        f"timing={item.mute.get('timing_source', '—')}, match={item.mute.get('match_type', '—')}"
    ]
    if item.words:
        first, last = item.words[0].idx, item.words[-1].idx
        lines.append(f"  - текст: {context(words, first, last)}")
    for flag in item.flags:
        lines.append(f"  - ⚠ {flag}")
    return lines


def _compact_line(item: MuteAudit) -> str:
    fit = item.fit
    tail = f", хвост {fit.tail_ms}" if fit else ""
    fill = f", речь {fit.fill * 100:.0f}%" if fit else ""
    return (
        f"- {fmt_ms(item.start_ms)} +{item.duration_ms} мс «{item.word_text}»"
        f"{fill}{tail}"
    )


FLAG_MARKERS: tuple[tuple[str, str], ...] = (
    ("НЕТ обсценного корня", "мьют без мата в тексте (ложное срабатывание?)"),
    ("только грубость", "только грубость на грани"),
    ("вообще нет слов", "под мьютом нет слов транскрипта"),
    ("обрывку/цензуре", "матч по обрывку слова или цензуре"),
    ("хвост тишины", f"хвост тишины ≥{SILENCE_FLAG_MS} мс"),
    ("тишина в голове", f"тишина в голове ≥{SILENCE_FLAG_MS} мс"),
    ("речь занимает лишь", f"речь <{LOW_FILL * 100:.0f}% окна"),
    ("хвост слова слышно", "мат по ASR длиннее мьюта справа"),
    ("недорезанный хвост", "звук за правой границей, слова там нет"),
    ("начало слышно", "мат по ASR начинается раньше мьюта"),
    ("недорезанное начало", "звук за левой границей, слова там нет"),
    ("длинный мьют", f"длинный мьют ≥{LONG_MUTE_MS} мс"),
    ("унаследованы от сегмента", "тайминги от сегмента, не от слова"),
    ("status=review", "status=review"),
)


def _header(result: AuditResult) -> list[str]:
    doc = result.doc
    settings = doc.get("settings") or {}
    statuses = Counter(str(m.mute.get("status")) for m in result.mutes)
    summary = doc.get("summary") or {}
    out = [
        f"# Аудит мата — {result.decisions_path.parent.name}",
        "",
        f"- decisions: `{result.decisions_path}`",
        f"- диапазон: {doc.get('range_in')} → {doc.get('range_out')} "
        f"(локально {doc.get('local_duration')})",
        f"- слов в транскрипте: {len(result.words)}",
        f"- мьютов: {len(result.mutes)} "
        f"({', '.join(f'{k}={v}' for k, v in sorted(statuses.items()))}); "
        f"сырых матчей детектора: {summary.get('raw_matches', '—')}",
        f"- аудио: "
        + (
            f"`{result.audio_path}`"
            if result.audio_ok
            else f"НЕ прочитано ({result.audio_path}) — проверки границ по звуку пропущены"
        ),
    ]
    knobs = {
        k: v
        for k, v in settings.items()
        if k.startswith("mute_") or k in {"raw_mute", "banwords"}
    }
    if knobs:
        out.append("- настройки мьюта: " + ", ".join(f"`{k}`={v}" for k, v in knobs.items()))
    out.append("")
    return out


def _stats(result: AuditResult) -> list[str]:
    out = ["## Сводка по границам", ""]
    if not result.mutes:
        return out + ["Мьютов нет.", ""]
    durations = sorted(m.duration_ms for m in result.mutes)
    mid = durations[len(durations) // 2]
    out.append(
        f"- длительность мьюта: мин {durations[0]}, медиана {mid}, макс {durations[-1]} мс"
    )
    fits = [m.fit for m in result.mutes if m.fit is not None]
    if fits:
        out.append(
            f"- речь в окне (медиана): {sorted(f.fill for f in fits)[len(fits) // 2] * 100:.0f}%; "
            f"хвост тишины (медиана): {sorted(f.tail_ms for f in fits)[len(fits) // 2]} мс"
        )
    trimmed = [
        (m.mute.get("audio_trim_start_ms") or 0) + (m.mute.get("audio_trim_end_ms") or 0)
        for m in result.mutes
    ]
    nonzero = [t for t in trimmed if t > 0]
    out.append(
        f"- поджато по звуку: {len(nonzero)} из {len(result.mutes)}"
        + (f", в среднем на {sum(nonzero) / len(nonzero):.0f} мс" if nonzero else "")
    )
    out.append("")
    for marker, label in FLAG_MARKERS:
        count = sum(1 for m in result.mutes if any(marker in f for f in m.flags))
        if count:
            out.append(f"- {label}: **{count}**")
    out.append("")
    return out


def _truth_section(result: AuditResult) -> list[str]:
    out = ["## Сверка с ground_truth.txt", ""]
    if not result.truth_path:
        return out + [
            "Файла нет — сверять с человеческим слухом нечего. Чтобы он появился:",
            "",
            "```",
            GROUND_TRUTH_HELP.rstrip(),
            "```",
            "",
            f"Положи его как `{result.decisions_path.parent / 'ground_truth.txt'}` "
            "и запусти audit снова.",
            "",
        ]
    if not Path(result.truth_path).exists():
        return out + [f"Файл `{result.truth_path}` не найден.", ""]
    out.append(
        f"Файл: `{result.truth_path}`, точек: {len(result.truth)}, "
        f"допуск ±{result.tolerance_ms} мс."
    )
    for problem in result.truth_problems:
        out.append(f"- ⚠ не понял {problem}")
    out.append("")
    out.append(
        f"- **попаданий: {len(result.truth_hits)}** из {len(result.truth)}"
    )
    out.append(f"- **промахов: {len(result.truth_misses)}** (мат есть, мьюта нет)")
    out.append(
        f"- мьютов вне списка: {len(result.truth_extras)} "
        "(если список полный — это лишние мьюты)"
    )
    out.append("")
    if result.truth_misses:
        out.append("### Промахи — мат есть, мьюта нет")
        out.append("")
        for point in result.truth_misses:
            near = [
                w for w in result.words
                if w.end_ms > point.start_ms - 1500 and w.start_ms < point.end_ms + 1500
            ]
            text = context(result.words, near[0].idx, near[-1].idx, radius=0) if near else "(слов нет)"
            out.append(f"- **{point.label()}** — что услышал ASR: {text}")
        out.append("")
    if result.truth_extras:
        out.append("### Мьюты, не отмеченные в списке")
        out.append("")
        out.extend(_compact_line(m) for m in result.truth_extras)
        out.append("")
    return out


MISSED_LABELS: list[tuple[str, str]] = [
    ("no_mute", "мьюта нет вовсе"),
    ("asr_timing_off", "мьют стоит не на звуке: границы слова у ASR уехали в тишину"),
    ("trimmed_off", "мьют есть, но мат закрыт не целиком"),
    ("partial", "мьют есть, но звук не проверен — нет кэша аудио"),
]
"""Классы настоящих пропусков в порядке убывания «виноват пайплайн»."""


def _why_missed(item: Missed) -> str:
    """Причина пропуска — по звуку, а не по догадке.

    Раньше здесь для КАЖДОГО непокрытого слова печаталось «корень детектор знал,
    значит потерялось дальше (статус/группировка/потолки)». На реальном стриме
    это врало дважды: у 8 из 10 слов мьют был и закрывал звук целиком (просто
    тайминг ASR длиннее слова, и середина слова оказывалась в тишине за мьютом),
    а у остальных виноват был не «статус/группировка», а тайминг ASR.
    """
    diagnosis = item.diagnosis
    if diagnosis == "asr_timing_off":
        where = (
            f"; звук на самом деле {fmt_ms(item.voice_span[0])}–{fmt_ms(item.voice_span[1])}"
            if item.voice_span
            else ""
        )
        return (
            f"в границах слова тишина (пик {(item.span_ratio or 0.0) * 100:.0f}% от "
            f"окрестности){where}. Мьют стоит по таймингам ASR, и звук остался вне "
            "него. Что видит машина, скажут поля `reanchor` и `expand_start_ms` в "
            "decisions.json: 'silent' = речь рядом есть, окно расширено назад на "
            "expand_start_ms мс (автоперенос выключен по замеру, mute_reanchor_move), "
            "'no_speech' = речи в радиусе поиска нет"
        )
    if diagnosis == "no_mute":
        if item.known_stem:
            return (
                f"детектор корень ЗНАЛ (`{item.known_stem}`) — значит мьют потерялся "
                "дальше (статус/группировка/потолки)"
            )
        return (
            "корня ядра в слове нет — это грубость из SPICY_ROOTS, детектор её не "
            "мьютит осознанно (или слово чистое, а корень нашёлся у соседа)"
        )
    if diagnosis == "trimmed_off":
        left = (
            f"вне мьюта осталось {(item.voiced_outside or 0.0) * 100:.0f}% звука"
            if (item.voiced_outside or 0.0) >= VOICE_LEFT_PCT
            else "самое громкое место слова осталось вне мьюта"
        )
        span = item.mute_span
        return (
            f"мьют {fmt_ms(span[0])}–{fmt_ms(span[1])} накрыл {item.covered_ms} мс "
            f"слова, {left}"
            if span
            else left
        )
    return "мьют есть, но проверить по звуку нечем (аудио не прочитано)"


def _not_missed_notes(trimmed: Sequence[Missed], moved: Sequence[Missed]) -> list[str]:
    """Слова, которые формально «не накрыты», но пропуском не являются.

    Печатать их обязательно: без объяснения человек видит в отчёте «мьют не на
    слове» и идёт править окно, которое либо и так закрывает звук, либо программа
    его уже дотянула до звука.
    """
    notes: list[str] = []
    if trimmed:
        notes.append(
            f"у {_plural(len(trimmed), 'слова', 'слов', 'слов')} мьют формально не "
            "накрывает середину слова, но вне мьюта тишина — это тайминг ASR длиннее слова"
        )
    if moved:
        notes.append(
            f"у {_plural(len(moved), 'слова', 'слов', 'слов')} тайминги ASR уехали в "
            "тишину, но мьют дотянулся до измеренного звука и накрывает его — править "
            "окно не нужно"
        )
    return notes


def _missed_section(result: AuditResult, limit: int) -> list[str]:
    out = ["## Пропуски, видимые прямо в транскрипте", ""]
    core = [m for m in result.missed if m.kind == "core"]
    spicy = [m for m in result.missed if m.kind == "spicy"]
    real = [m for m in core if m.real]
    trimmed = [m for m in core if m.diagnosis == "trimmed_to_speech"]
    moved = [m for m in core if m.diagnosis == "reanchored"]
    notes = _not_missed_notes(trimmed, moved)
    if not core:
        out.append("Ядра мата без мьюта в транскрипте нет — по тексту всё закрыто.")
        out.append("")
    elif not real:
        tail = ": " + "; ".join(notes) + "." if notes else "."
        out.append("Настоящих пропусков нет" + tail)
        out.append("")
    else:
        out.append(f"**{len(real)}** слов с обсценным корнем звучат незакрытыми:")
        out.append("")
        shown = 0
        for diagnosis, label in MISSED_LABELS:
            items = [m for m in real if m.diagnosis == diagnosis]
            if not items:
                continue
            out.append(f"**{label}** — {len(items)}:")
            out.append("")
            for item in items[: max(0, limit - shown)]:
                word = item.word
                out.append(
                    f"- **{fmt_ms(word.start_ms)}–{fmt_ms(word.end_ms)}** «{word.text}» "
                    f"(корни: {', '.join(item.roots)}) — {_why_missed(item)}"
                )
                out.append(f"  - текст: {context(result.words, word.idx, word.idx)}")
            shown += len(items)
            if shown > limit:
                out.append(f"- … и ещё {shown - limit}")
            out.append("")
        for note in notes:
            out.append(f"Ещё {note}.")
            out.append("")
    if spicy:
        out.append(
            f"Грубость на грани (мьютить или нет — решаешь ты), {len(spicy)}: "
            + ", ".join(
                f"{fmt_ms(m.word.start_ms)} «{m.word.text}»" for m in spicy[:limit]
            )
            + (f" … и ещё {len(spicy) - limit}" if len(spicy) > limit else "")
        )
        out.append("")
    return out


def _suspicious_section(result: AuditResult, limit: int) -> list[str]:
    flagged = [m for m in result.mutes if m.flags]
    out = ["## Мьюты с подозрительными границами", ""]
    if not flagged:
        return out + ["Все мьюты сидят на речи чисто.", ""]
    out.append(f"{len(flagged)} из {len(result.mutes)}. Сначала самые проблемные:")
    out.append("")
    flagged.sort(key=lambda m: (-len(m.flags), -(m.fit.tail_ms if m.fit else 0)))
    for item in flagged[:limit]:
        out.extend(_detail_block(item, result.words))
    if len(flagged) > limit:
        out.append(f"- … и ещё {len(flagged) - limit} (подними --max-details)")
    out.append("")
    return out


def _holes_section(result: AuditResult, limit: int = 25) -> list[str]:
    out = ["## Дыры ASR — звук есть, слов нет", ""]
    if not result.audio_ok:
        return out + ["Аудио не прочитано — проверка пропущена.", ""]
    if not result.holes:
        return out + ["Громких участков без слов нет.", ""]
    out.append(
        f"{len(result.holes)} участков. Мат здесь не мог быть найден в принципе "
        "(речь не распознана) — это то, что стоит послушать руками:"
    )
    out.append("")
    for hole in sorted(result.holes, key=lambda h: -h.duration_ms)[:limit]:
        out.append(
            f"- **{fmt_ms(hole.start_ms)}–{fmt_ms(hole.end_ms)}** "
            f"({hole.duration_ms / 1000:.1f} с, звук в {hole.voiced * 100:.0f}% кадров)"
        )
    if len(result.holes) > limit:
        out.append(f"- … и ещё {len(result.holes) - limit}")
    out.append("")
    return out


def _clean_section(result: AuditResult, limit: int) -> list[str]:
    clean = [m for m in result.mutes if not m.flags]
    out = ["## Остальные мьюты (чистые)", ""]
    if not clean:
        return out + ["—", ""]
    out.extend(_compact_line(m) for m in clean[:limit])
    if len(clean) > limit:
        out.append(f"- … и ещё {len(clean) - limit}")
    out.append("")
    return out


def _edit_fit_text(fit: EnergyFit | None, voiced: float | None = None) -> str:
    """Звук в окне двумя мерами. `fit.fill` — по ОТНОСИТЕЛЬНОМУ порогу (от пика
    самого окна), `абс.` — по абсолютному (`voiced_ratio`). Показываем обе рядом
    именно потому, что они расходятся: в тишине относительный порог опускается до
    шума и даёт «речь 42% окна» там, где по абсолютному озвучено 0%. Диагноз
    считается по абсолютному, и число под ним должно быть видно."""
    abs_part = f" (абс. {voiced * 100:.0f}%)" if voiced is not None else ""
    if fit is None:
        return f"озвучено{abs_part}" if voiced is not None else "звук не проверен"
    return (
        f"речь {fit.fill * 100:.0f}% окна{abs_part}, "
        f"голова {fit.head_ms} мс, хвост {fit.tail_ms} мс"
    )


def _edit_line(finding: EditFinding, words: Sequence[Word]) -> list[str]:
    span = f"{fmt_ms(finding.start_ms)}–{fmt_ms(finding.end_ms)}"
    if finding.base_start_ms is not None and finding.base_end_ms is not None:
        span = (
            f"{fmt_ms(finding.base_start_ms)}–{fmt_ms(finding.base_end_ms)} → {span}"
        )
    title = str(finding.mute.get("word") or "").strip()
    out = [f"- **{span}**" + (f" «{title}»" if title else "") + f" — {finding.verdict}"]
    if finding.words:
        first, last = finding.words[0].idx, finding.words[-1].idx
        out.append(f"  - текст: {context(words, first, last)}")
    if finding.base_fit is not None or finding.base_voiced is not None:
        out.append(
            f"  - звук: программа — "
            f"{_edit_fit_text(finding.base_fit, finding.base_voiced)}; "
            f"ты — {_edit_fit_text(finding.fit, finding.voiced)}"
        )
    elif finding.fit is not None or finding.voiced is not None:
        out.append(f"  - звук: {_edit_fit_text(finding.fit, finding.voiced)}")
    diag = _edit_diag(finding.mute)
    if diag:
        # Для добавленного вручную мьюта «как решала программа» — вранье: она
        # его не решала вовсе.
        label = "поля твоего мьюта" if finding.kind == "added" else "как решала программа"
        out.append(f"  - {label}: {diag}")
    return out


def _edit_diag(mute: dict[str, Any]) -> str:
    """Поля, по которым видно механику решения программы для этого мьюта."""
    bits: list[str] = []
    for key, label in (
        ("timing_source", "тайминг"),
        ("match_type", "match"),
        ("banword", "banword"),
        ("status", "статус"),
    ):
        value = mute.get(key)
        if value:
            bits.append(f"{label}={value}")
    trim = _trim_pair(mute)
    if trim:
        bits.append(f"подгонка по звуку −{trim[0]}/−{trim[1]} мс")
    raw_start, raw_end = mute.get("raw_local_start"), mute.get("raw_local_end")
    if isinstance(raw_start, (int, float)) and isinstance(raw_end, (int, float)):
        bits.append(f"ASR отдал {raw_start:.3f}–{raw_end:.3f}")
    return ", ".join(bits)


def _median(values: Sequence[int]) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return int(round((ordered[mid - 1] + ordered[mid]) / 2))


def _move_summary(moved: Sequence[EditFinding], settings: dict[str, Any]) -> list[str]:
    """Куда человек систематически двигает границы — и какая настройка за это
    отвечает. Одна правка ничего не значит, поэтому смотрим медиану."""
    if not moved:
        return []
    d_start = [f.d_start_ms for f in moved]
    d_end = [f.d_end_ms for f in moved]
    med_start, med_end = _median(d_start), _median(d_end)
    out = [
        f"Сдвигов: {len(moved)}. Медиана: начало {med_start:+d} мс, конец {med_end:+d} мс. "
        f"Начало двигал раньше: {sum(1 for v in d_start if v < 0)} из {len(moved)}; "
        f"конец двигал позже: {sum(1 for v in d_end if v > 0)} из {len(moved)}.",
    ]
    hints: list[str] = []
    before = settings.get("mute_padding_before_ms")
    after = settings.get("mute_padding_after_ms")
    guard = settings.get("mute_trim_guard_ms")
    if med_start <= -40:
        hints.append(
            f"начало систематически опаздывает → поднять `mute_padding_before_ms` "
            f"(сейчас {before}) или `mute_trim_guard_ms` (сейчас {guard})"
        )
    if med_end >= 40:
        hints.append(
            f"хвост систематически не дорезан → поднять `mute_padding_after_ms` "
            f"(сейчас {after}); если у этих мьютов есть подгонка по звуку — она режет лишнее"
        )
    if med_start >= 40 and med_end <= -40:
        hints.append(
            f"мьюты стабильно ШИРЕ нужного → padding великоват "
            f"({before}/{after} мс), либо подгонка по звуку выключена"
        )
    if hints:
        out.append("")
        out.extend(f"- {h}" for h in hints)
    return out


_BASELINE_NOTE = {
    "pipeline": "снимок решения ПАЙПЛАЙНА — сравнение честное",
    "first_save": (
        "снимок первого УВИДЕННОГО сохранения (журнал включили после обработки) — "
        "правки, сделанные до этого, в сравнение не попали"
    ),
    "unknown": "источник снимка неизвестен — сравнение приблизительное",
}


def _edits_section(result: AuditResult, limit: int) -> list[str]:
    out = ["## Твои правки против того, что сделала программа", ""]
    journal = result.journal
    if journal is None:
        return out + [
            "Журнала правок нет. Включи «Журнал правок» в настройках — тогда каждое "
            "сохранение из «Правки» будет дописываться в "
            f"`{journal_path(result.decisions_path).name}`, и этот раздел покажет, "
            "где человек и программа разошлись и почему.",
            "",
        ]
    source = baseline_source(journal)
    baseline = journal.get("baseline") if isinstance(journal.get("baseline"), dict) else {}
    out.append(
        f"Журнал: `{result.edits_path}` — базовый снимок от {baseline.get('captured_at')}, "
        f"{_BASELINE_NOTE.get(source, source)}."
    )
    if not result.edits and not result.edit_notes:
        return out + ["", "Расхождений с программой нет — ты ничего не менял.", ""]
    by_kind = {kind: [f for f in result.edits if f.kind == kind] for kind in ("added", "removed", "off", "moved", "on")}
    out.append("")
    if any(by_kind.values()):
        out.append(
            "Мьюты: добавлено вручную — {added}, снято — {off}, сдвинуто — {moved}, "
            "включено обратно — {on}.".format(
                added=len(by_kind["added"]),
                off=len(by_kind["removed"]) + len(by_kind["off"]),
                moved=len(by_kind["moved"]),
                on=len(by_kind["on"]),
            )
        )
    else:
        out.append("Мьюты программы ты не трогал.")
    for note in result.edit_notes:
        out.append(f"- {note}")
    out.append("")
    out += _edits_added(by_kind["added"], result, limit)
    out += _edits_dropped(by_kind["removed"] + by_kind["off"], result, limit)
    out += _edits_moved(by_kind["moved"], result, limit)
    out += _edits_enabled(by_kind["on"], result, limit)
    return out


def _edits_added(items: Sequence[EditFinding], result: AuditResult, limit: int) -> list[str]:
    if not items:
        return []
    out = [
        f"### Программа НЕ нашла — ты добавил мьют ({len(items)})",
        "",
        "Самая ценная часть: каждая такая правка — это либо дыра в словаре, либо "
        "ошибка ASR, либо потеря уже найденного мата.",
        "",
    ]
    for finding in items[:limit]:
        out += _edit_line(finding, result.words)
    if len(items) > limit:
        out.append(f"- … и ещё {len(items) - limit}")
    out += [
        "",
        "Готовые строки для `ground_truth.txt` (чтобы эти места проверялись "
        "автоматически на следующих прогонах):",
        "",
        "```",
    ]
    for finding in items:
        word = str(finding.mute.get("word") or "").strip()
        out.append(
            f"{fmt_ms(finding.start_ms)}-{fmt_ms(finding.end_ms)}"
            + (f"  {word}" if word else "")
        )
    out += ["```", ""]
    return out


def _edits_dropped(items: Sequence[EditFinding], result: AuditResult, limit: int) -> list[str]:
    if not items:
        return []
    out = [
        f"### Ты снял мьют ({len(items)})",
        "",
        "Если корня мата в перекрытых словах нет — это ложное срабатывание "
        "детектора, и его надо лечить в словаре/нормализации.",
        "",
    ]
    for finding in sorted(items, key=lambda f: f.start_ms)[:limit]:
        out += _edit_line(finding, result.words)
    if len(items) > limit:
        out.append(f"- … и ещё {len(items) - limit}")
    out.append("")
    return out


def _edits_moved(items: Sequence[EditFinding], result: AuditResult, limit: int) -> list[str]:
    if not items:
        return []
    out = [f"### Ты поправил границы ({len(items)})", ""]
    settings = result.doc.get("settings")
    out += _move_summary(items, settings if isinstance(settings, dict) else {})
    out.append("")
    for finding in items[:limit]:
        out += _edit_line(finding, result.words)
    if len(items) > limit:
        out.append(f"- … и ещё {len(items) - limit}")
    out.append("")
    return out


def _edits_enabled(items: Sequence[EditFinding], result: AuditResult, limit: int) -> list[str]:
    if not items:
        return []
    out = [
        f"### Ты включил мьют, отложенный программой ({len(items)})",
        "",
        "Это про статус `review`: детектор нашёл мат, но не решился мьютить сам.",
        "",
    ]
    for finding in items[:limit]:
        out += _edit_line(finding, result.words)
    if len(items) > limit:
        out.append(f"- … и ещё {len(items) - limit}")
    out.append("")
    return out


def render_report(
    result: AuditResult, *, max_details: int = 40, max_clean: int = 150
) -> str:
    lines: list[str] = []
    lines += _header(result)
    lines += _stats(result)
    lines += _truth_section(result)
    lines += _edits_section(result, max_details)
    lines += _missed_section(result, max_details)
    lines += _suspicious_section(result, max_details)
    lines += _holes_section(result)
    lines += _clean_section(result, max_clean)
    lines += [
        "---",
        "",
        "Чего этот отчёт не видит САМ: слово, которое ASR УСЛЫШАЛ неправильно. "
        "В транскрипте его нет, по энергии это обычная речь. Такие места ловятся "
        "через ground_truth.txt, через «Дыры ASR» выше или через журнал правок "
        "(мьют, добавленный вручную там, где в тексте мата нет).",
        "",
    ]
    return "\n".join(lines)


def write_report(
    result: AuditResult,
    out_path: Path | None = None,
    *,
    max_details: int = 40,
    max_clean: int = 150,
) -> Path:
    target = Path(out_path) if out_path else result.decisions_path.parent / "audit.md"
    target.write_text(
        render_report(result, max_details=max_details, max_clean=max_clean),
        encoding="utf-8",
    )
    return target
