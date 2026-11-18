"""Что играло в видео во время реплики — и играло ли вообще.

Здесь был баг прототипа v3, поэтому модуль отдельный (и на него есть тест).

БАГ. Контекст брался окном ±25с вокруг реплики. У длинной реплики (замеренный
случай — 276с монолога) привязки есть только по её КРАЯМ: стример что-то сказал
до паузы и после неё. Окно цепляло эти краевые привязки и склеивало ~5 минут
текста исходника, хотя внутри реплики видео молчало. Модель видела «в видео
говорят про айфоны», совпадение по ключевому слову — и засчитывала болтовню про
пароль от айфона как реакцию на кадр.

ИСПРАВЛЕНИЕ. Пауза измеряется ВНУТРИ реплики: `привязок / длительность`. Ниже
порога (0.08 привязки в секунду ≈ одна привязка на 12с) считаем, что видео
молчит. Краевые привязки в счёт не идут — их время лежит вне интервала.

Пауза отдаётся модели ЧИСЛОМ (`<ПАУЗА 277с>`), а не флагом: длительность и есть
сила улики. Минута тишины видео при говорящем стримере — почти всегда уход в
свои дела; две секунды — ничего не значат, стример мог поставить паузу, чтобы
обсудить кадр. Флаг заставил бы модель одинаково относиться к обоим случаям.
"""

from __future__ import annotations

from dataclasses import dataclass

from .align import Alignment
from .turns import Turn

# Привязок в секунду; ниже — видео считаем молчащим. 0.08 ≈ одна привязка на
# 12 секунд: реальная речь видео даёт кратно больше, а редкие ложные шинглы
# внутри длинного монолога порог не пробивают.
PAUSE_RATE_PER_S = 0.08

# Совсем короткую реплику не делим на её длительность: на 0.5с любая единственная
# привязка дала бы rate=2 и «играет». Знаменатель не меньше секунды.
MIN_DURATION_S = 1.0

# Сколько слов исходника дать модели, когда видео играет: от первой привязки до
# последней + хвост (реплика обычно начинается чуть раньше, чем ASR ловит шингл).
PLAYING_TAIL_WORDS = 12
PLAYING_MAX_CHARS = 260

# Что играло ПЕРЕД паузой: окно вокруг последней привязки, обрезанное с начала —
# модели нужен конец фразы, на которой видео остановили.
BEFORE_PAUSE_BACK_WORDS = 25
BEFORE_PAUSE_AHEAD_WORDS = 5
BEFORE_PAUSE_MAX_CHARS = 200

# Подставляется, если привязок до реплики нет вообще (видео ещё не начиналось).
NO_SOURCE_TEXT = "—"


@dataclass(frozen=True)
class VideoContext:
    """Состояние видео на интервале реплики.

    :param playing: видео звучало (привязок хватило, чтобы пробить порог).
    :param anchors: сколько привязок ВНУТРИ интервала — сырая улика для лога.
    :param duration_s: длительность интервала (не меньше MIN_DURATION_S при
        расчёте rate, здесь — фактическая).
    :param rate: привязок в секунду.
    :param pause_s: длительность паузы; 0.0, если видео играло.
    :param text: если играло — текст исходника этого места; если пауза — что
        играло перед ней.
    """

    playing: bool
    anchors: int
    duration_s: float
    rate: float
    pause_s: float
    text: str

    def render(self) -> str:
        """Строка для промпта. Формулировки замерены на VOD — менять с оглядкой."""
        if self.playing:
            return f"<играет, привязок {self.anchors}> {self.text}".rstrip()
        return (
            f"<ПАУЗА {self.pause_s:.0f}с — видео молчит, привязок {self.anchors}>. "
            f"Перед паузой в видео было: …{self.text or NO_SOURCE_TEXT}"
        )


def video_context(
    alignment: Alignment,
    start_s: float,
    end_s: float,
    *,
    pause_rate_per_s: float = PAUSE_RATE_PER_S,
) -> VideoContext:
    """Посчитать состояние видео на [start_s, end_s].

    Смотрим ТОЛЬКО привязки внутри интервала — в этом весь смысл (см. модульный
    docstring). За его пределы выходим единственный раз: чтобы показать, на чём
    видео остановили, если внутри пауза.
    """
    inside = alignment.anchors_in(start_s, end_s)
    duration = max(0.0, end_s - start_s)
    rate = len(inside) / max(MIN_DURATION_S, duration)

    if rate < pause_rate_per_s:
        last = alignment.last_anchor_before(start_s)
        before_text = (
            ""
            if last is None
            else alignment.source_text(
                last - BEFORE_PAUSE_BACK_WORDS,
                last + BEFORE_PAUSE_AHEAD_WORDS,
                max_chars=BEFORE_PAUSE_MAX_CHARS,
                tail=True,
            )
        )
        return VideoContext(
            playing=False,
            anchors=len(inside),
            duration_s=round(duration, 3),
            rate=round(rate, 4),
            pause_s=round(max(MIN_DURATION_S, duration), 3),
            text=before_text,
        )

    lo, hi = min(inside), max(inside)
    return VideoContext(
        playing=True,
        anchors=len(inside),
        duration_s=round(duration, 3),
        rate=round(rate, 4),
        pause_s=0.0,
        text=alignment.source_text(
            lo, hi + PLAYING_TAIL_WORDS, max_chars=PLAYING_MAX_CHARS
        ),
    )


def turn_context(
    alignment: Alignment,
    turn: Turn,
    *,
    pause_rate_per_s: float = PAUSE_RATE_PER_S,
) -> VideoContext:
    """video_context для реплики стримера (turns.Turn)."""
    return video_context(
        alignment, turn.start_s, turn.end_s, pause_rate_per_s=pause_rate_per_s
    )
