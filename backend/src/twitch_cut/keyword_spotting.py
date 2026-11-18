"""keyword_spotting.py — поиск мата по ПОСТЕРИОРАМ CTC, а не по тексту ASR.

ПРОБЛЕМА, которую это лечит. Детектор читает ТЕКСТ распознавания, поэтому он
бессилен там, где ASR услышал не то: «епанисир», «изачльно поугарал», «епта»
(мат с «п» вместо «б») — корня в тексте нет, значит нет и мьюта. На замере
`twitch-cut eval` это самый большой остаток ошибки: 17 из 25 мьютов, которые
человек добавил руками, приходятся именно на такие места (см. `evaluate.py`).
Границы слов чинить дальше бессмысленно — чинить надо СЛУХ.

ЧТО ДЕЛАЕМ. Greedy-декод CTC на каждом кадре берёт САМЫЙ вероятный токен и
выбрасывает остальные — а вероятность нужной буквы там часто вторая-третья, и
её видно в log-probs. Поэтому ищем корень мата прямо в посториорах, не спрашивая
у декодера, что он там «услышал».

КАК. CTC пиковый: каждая буква выстреливает одиночным всплеском, между
всплесками blank. Значит слово в посториорах — это ЦЕПОЧКА ПИКОВ по буквам
корня, идущих по порядку и близко друг к другу, А В РАЗРЫВАХ МЕЖДУ НИМИ НЕ
ДОЛЖНО БЫТЬ ДРУГИХ БУКВ. Ищем такую цепочку динамическим программированием:
`dp[i][t]` — лучшая сумма log-prob для первых i букв, если i-я буква выстрелила
на кадре t. Переход — из любого кадра в пределах `max_gap_frames` назад, и за
каждый кадр внутри разрыва добавляется log-prob того, что там blank (или повтор
той же буквы — CTC склеивает подряд идущие одинаковые метки). Ответ — максимум
по последней букве; скор кандидата = сумма, поделённая на ЧИСЛО БУКВ и взятая
как exp, то есть средняя геометрическая вероятность на букву — та же величина,
что `score` слова из `alignment.py`, и её можно сравнивать с ней напрямую.

ПОЧЕМУ РАЗРЫВЫ ПРИХОДИТСЯ ПРОВЕРЯТЬ. Первая версия их не оценивала вовсе —
и на реальном прогоне выдала «ебат» со скором 1.00 внутри слова «брат», «ебло»
внутри «было»: б, а, т стоят по порядку и близко, а «е» нашлась в соседнем
слове. Без blank в разрывах ищется не слово, а ПОДПОСЛЕДОВАТЕЛЬНОСТЬ букв, и в
беглой русской речи она находится почти всюду. Кадр с чужой буквой даёт в
разрыве log-prob около -3 и цепочку убивает, кадр blank — около -0.02 и на скор
не влияет; ровно этого и хотелось.

ПОЧЕМУ НЕ ПОЛНЫЙ CTC-ФОРВАРД. Полный форвард считает вероятность ВСЕХ путей и
нормирует её на длину пути, а длина у локального совпадения не фиксирована —
поэтому blank-кадры (у них вероятность под 0.98) начинают тянуть скор вверх и
короткие «совпадения» в тишине выглядят увереннее настоящих. Здесь нормировка
только на число БУКВ: blank-кадры разрыва могут скор лишь опустить, поднять его
не может ничто, и порог остаётся сравнимым между ключами разной длины.

ЧЕГО ЭТО НЕ ДЕЛАЕТ. Кандидат — не приговор: «ебал» может собраться из «е бал»,
а на границе слов blank стоит законно. Поэтому порог `min_score` выбран по
замеру на реальном прогоне (см. `SPOT_MIN_SCORE` в `config.py`), а сами
кандидаты идут в решение как `review`, а не как готовый мьют.

Модуль намеренно не знает ни про gigaam, ни про мат: на вход — матрица log-probs,
список строк-ключей и функция «буква → id токена». Поэтому проверяется
синтетическими пиками, без модели и без аудио. Считается на numpy во float32 —
те же операции и та же точность, что были у прежней torch-версии.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
import math
from typing import Any, Callable, Sequence

import numpy as np

logger = logging.getLogger(__name__)

MAX_GAP_FRAMES = 6
"""Максимальный разрыв между всплесками соседних букв, в кадрах энкодера.

Кадр GigaAM v3 ≈ 40 мс, то есть 6 кадров — это 240 мс на букву. Раньше здесь
стояло 3: пока кадры разрыва не оценивались, длинный разрыв означал цепочку из
букв разных слов. Теперь разрыв обязан быть blank, и держать его коротким
незачем — замер на реальном прогоне: с 3 до 6 кадров ложных не прибавилось
вообще (4 при пороге 0.3), зато нашлось растянутое «е…банский», у которого
между «е» и «п» шесть кадров чистой тишины. Дальше 6 кривая упирается: 8, 10 и
12 дают ровно то же самое."""

NEG_INF = -1e30
"""«Так нельзя» для DP. Не `-inf`: сумма с ним даёт NaN на некоторых бэкендах."""

MIN_KEYWORD_LEN = 4
"""Короче четырёх букв искать нельзя.

Замерено на реальном прогоне: трёхбуквенные ключи («бля», «хуй», «хуя») дают
кандидатов пачками на невинной речи — трёх пиков подряд слишком мало, чтобы
отличить корень от случайного созвучия. Полные формы («блят», «хуйн») ловят то
же самое, но уже без мусора."""


@dataclass
class SpottedKeyword:
    """Найденная в посториорах цепочка букв корня.

    `start`/`end` — секунды от начала сегмента (как у `alignment.AlignedWord`),
    границы по КАДРАМ ЭМИССИИ первой и последней буквы. Шире их не делаем: CTC
    выстреливает с задержкой, но за задержку отвечает padding мьюта, а он
    настраиваемый и одинаковый для всех источников.
    """

    keyword: str
    start: float
    end: float
    score: float
    """Средняя геометрическая вероятность на букву корня (0..1) — сравнима с
    `score` слова из forced alignment напрямую. В сумму входит и плата за кадры
    разрыва (там должен быть blank), поэтому чужая буква между буквами корня
    роняет скор, а тишина между ними — почти нет."""
    start_frame: int
    end_frame: int


def spot_keywords(
    log_probs: Any,
    keywords: Sequence[str],
    *,
    char_to_id: Callable[[str], int | None],
    blank_id: int,
    frame_shift: float,
    min_score: float,
    max_gap_frames: int = MAX_GAP_FRAMES,
    require_unheard: bool = True,
) -> list[SpottedKeyword]:
    """Кандидаты на мат в посториорах ОДНОГО сегмента.

    `log_probs` — [T, C] log-softmax CTC (numpy, float32), `keywords` —
    корни строками (уже нормализованные под словарь: нижний регистр, ё→е),
    `char_to_id` — «буква → id токена» или None, если буквы нет в словаре,
    `blank_id` — id blank-токена (по нему проверяются разрывы между буквами),
    `frame_shift` — секунд на кадр энкодера, `min_score` — порог по средней
    вероятности буквы (0..1).

    `require_unheard` — отдавать только то, чего НЕТ в тексте распознавания:
    если greedy-декод на этом же месте сам выдал буквы ключа, кандидат выкидывается
    (см. `_greedy_contains`). Ровно за этим стадия и нужна: то, что попало в текст,
    разбирает текстовый детектор — со словарём, леммами и списком омографов, то
    есть лучше. Выключать стоит только для отладки («а видит ли спотинг то, что
    нашёл текст»).

    Возвращает НЕПЕРЕСЕКАЮЩИЕСЯ кандидаты по времени: если в одном месте
    сработало несколько корней («хуйн» и «хуйня»), остаётся самый уверенный.
    Пустой список — не ошибка, а нормальный ответ «мата здесь не слышно».
    """
    if not keywords or min_score <= 0.0 or min_score > 1.0:
        return []
    try:
        emission = np.asarray(log_probs, dtype=np.float32)
    except Exception as exc:  # noqa: BLE001
        logger.warning("keyword spotting: не читаются log-probs (%s)", exc)
        return []
    if emission.ndim != 2 or emission.shape[0] == 0:
        return []
    if not 0 <= int(blank_id) < int(emission.shape[1]):
        logger.warning("keyword spotting: blank_id %s вне словаря", blank_id)
        return []

    found: list[SpottedKeyword] = []
    for keyword in keywords:
        ids = _keyword_ids(keyword, char_to_id)
        if ids is None or len(keyword) < MIN_KEYWORD_LEN:
            continue
        found.extend(
            _spot_one(
                emission,
                keyword,
                ids,
                blank_id=int(blank_id),
                frame_shift=frame_shift,
                min_score=min_score,
                max_gap_frames=max(1, int(max_gap_frames)),
                require_unheard=require_unheard,
            )
        )
    return _suppress_overlaps(found)


def _keyword_ids(keyword: str, char_to_id: Callable[[str], int | None]) -> list[int] | None:
    """Буквы ключа → id токенов. Хоть одной буквы нет в словаре → None (искать
    нечего: у charwise-модели такой буквы просто не бывает на выходе)."""
    ids: list[int] = []
    for char in keyword:
        token_id = char_to_id(char)
        if token_id is None:
            return None
        ids.append(int(token_id))
    return ids


def _spot_one(
    emission: np.ndarray,
    keyword: str,
    ids: Sequence[int],
    *,
    blank_id: int,
    frame_shift: float,
    min_score: float,
    max_gap_frames: int,
    require_unheard: bool,
) -> list[SpottedKeyword]:
    """Все места, где цепочка пиков по буквам ключа набрала выше порога."""
    n_frames = int(emission.shape[0])
    if n_frames < len(ids):
        return []
    blank = emission[:, blank_id]
    neg_inf = np.float32(NEG_INF)

    # dp[t] — лучшая сумма log-prob, если ПОСЛЕДНЯЯ разобранная буква
    # выстрелила на кадре t; starts[t] — кадр первой буквы этой же цепочки.
    dp = emission[:, ids[0]].copy()
    starts = np.arange(n_frames, dtype=np.int64)
    for i in range(1, len(ids)):
        # Повтор буквы («нн») в CTC требует blank между всплесками, иначе он
        # декодируется как ОДНА буква. Поэтому соседние кадры для повтора
        # запрещены (минимальный разрыв 2), и в разрыве допустим только blank.
        repeat = ids[i] == ids[i - 1]
        min_shift = 2 if repeat else 1
        # Чем «оплачивается» кадр внутри разрыва: blank или (если буквы разные)
        # затянувшийся всплеск предыдущей буквы — подряд идущие одинаковые
        # метки CTC склеивает в одну, так что это тот же путь.
        gap = blank if repeat else np.maximum(blank, emission[:, ids[i - 1]])

        best = np.full(n_frames, neg_inf, dtype=np.float32)
        best_start = np.zeros(n_frames, dtype=np.int64)
        # run[t] — сумма gap по кадрам, которые окажутся ВНУТРИ разрыва длины
        # shift, то есть t-1 … t-shift+1. Набирается по одному кадру на шаг.
        run = np.zeros(n_frames, dtype=np.float32)
        for shift in range(1, max_gap_frames + 1):
            if shift >= n_frames:
                break
            if shift > 1:
                inner = np.zeros(n_frames, dtype=np.float32)
                inner[shift - 1 :] = gap[: n_frames - shift + 1]
                run = run + inner
            if shift < min_shift:
                continue
            cand = np.full(n_frames, neg_inf, dtype=np.float32)
            cand[shift:] = dp[:-shift] + run[shift:]
            cand_start = np.zeros(n_frames, dtype=np.int64)
            cand_start[shift:] = starts[:-shift]
            better = cand > best
            best = np.where(better, cand, best)
            best_start = np.where(better, cand_start, best_start)
        dp = emission[:, ids[i]] + best
        starts = best_start

    # Порог по СРЕДНЕЙ вероятности буквы: сумма log-prob сравнивается с
    # len(ids) * log(min_score) — так порог не зависит от длины ключа.
    threshold = np.float32(len(ids) * math.log(min_score))
    hits: list[SpottedKeyword] = []
    for frame in np.nonzero(dp >= threshold)[0].tolist():
        first = int(starts[int(frame)])
        last = int(frame)
        if require_unheard and _greedy_contains(
            emission, first, last + 1, ids, blank_id=blank_id
        ):
            continue
        hits.append(
            SpottedKeyword(
                keyword=keyword,
                start=first * frame_shift,
                end=(last + 1) * frame_shift,
                score=round(math.exp(float(dp[last]) / len(ids)), 4),
                start_frame=first,
                end_frame=last + 1,
            )
        )
    return _suppress_overlaps(hits)


def greedy_path(
    emission: Any,
    *,
    blank_id: int,
    start_frame: int = 0,
    end_frame: int | None = None,
) -> list[tuple[int, int]]:
    """Greedy-декод как [(кадр, id метки)]: blank выброшен, повторы склеены.

    Кадр — первый кадр всплеска буквы. Он нужен не только фильтру «ключ уже
    есть в тексте» ниже, но и подстановке кандидата в гипотезу сегмента
    (`profanity._aligned_hit_bounds`): без кадров непонятно, какие токены
    выбрасывать на место находки, а forced_align монотонен и требует, чтобы
    порядок токенов совпадал с порядком кадров.
    """
    labels = np.asarray(emission)[start_frame:end_frame].argmax(axis=1).tolist()
    path: list[tuple[int, int]] = []
    previous: int | None = None
    for offset, label in enumerate(labels):
        # Правило CTC: подряд идущие одинаковые метки — одна буква, blank
        # разрывает повтор (поэтому «previous» помнит и blank тоже).
        if label != previous and label != blank_id:
            path.append((start_frame + offset, label))
        previous = label
    return path


def _greedy_contains(
    emission: Any,
    start_frame: int,
    end_frame: int,
    ids: Sequence[int],
    *,
    blank_id: int,
) -> bool:
    """Есть ли ключ в ТЕКСТЕ распознавания на этом же месте?

    Текст — это greedy-декод: argmax на кадре, blank выброшен, подряд идущие
    одинаковые метки склеены. Если ключ нашёлся и там, кандидат не нужен: это
    же слово уже видит текстовый детектор — со словарём, леммами и списком
    омографов, то есть он рассудит лучше. Так уходит «блят» внутри
    «оскорблять»: буквы там идут подряд, и они правда невинны. А «ебан» внутри
    «епанисир» остаётся — greedy на месте «б» настаивает на «п», то есть в
    тексте корня нет и рассудить его текстовому детектору нечем.

    Сравнение ПОДРЯД идущих меток — не придирка, а суть: DP разрешает разрывы
    между буквами, greedy-путь — нет. Поэтому фильтр снимает только буквальные
    вхождения ключа в текст, а невинное слово, где буквы ключа раскиданы
    («нефрибеты» против ключа «ебет» — е,б,е,т там есть, но не подряд), сюда не
    попадает и доезжает до человека кандидатом. Такие остатки гасит не этот
    фильтр, а статус review: в Vegas кандидаты не уезжают.
    """
    collapsed = [
        label
        for _, label in greedy_path(
            emission, blank_id=blank_id, start_frame=start_frame, end_frame=end_frame
        )
    ]
    n = len(ids)
    return any(collapsed[i : i + n] == list(ids) for i in range(len(collapsed) - n + 1))


def _suppress_overlaps(hits: list[SpottedKeyword]) -> list[SpottedKeyword]:
    """Оставляем самого уверенного кандидата на каждое место.

    DP даёт всплеск скора на нескольких соседних кадрах подряд (и разные корни
    ловят одно и то же слово), поэтому без подавления один «ебаный» приезжает
    десятком кандидатов. Жадно по убыванию скора: берём кандидата, если его
    окно ещё не занято.
    """
    kept: list[SpottedKeyword] = []
    for hit in sorted(hits, key=lambda h: (-h.score, h.start_frame)):
        if any(
            hit.start_frame < other.end_frame and other.start_frame < hit.end_frame
            for other in kept
        ):
            continue
        kept.append(hit)
    return sorted(kept, key=lambda h: h.start_frame)
