"""alignment.py — пословные границы из forced alignment, а не из кадров эмиссии.

ПРОБЛЕМА. GigaAM отдаёт тайминги слова так (`gigaam/timestamps_utils.py`):
начало = кадр, на котором CTC ВЫСТРЕЛИЛ первой буквой, конец = кадр последней
буквы + 1. CTC пиковый: он выдаёт букву одним кадром и молчит (blank) вокруг,
причём выстреливает ПОЗЖЕ настоящего начала звука — модели нужен контекст.
Значит окно слова получается и сдвинутым вправо, и слишком узким: хвост звука
(гласная, шипящая) остаётся за границей. На реальном прогоне человек правил
границы у 87% мьютов (`twitch-cut eval`: медиана 100 мс по началу и 77 по
концу, сдвиг со знаком −30 и −19 мс — машина опаздывала).

ЧТО ДЕЛАЕМ. Берём те же log-probs CTC и гоним forced alignment (Витерби, см.
`forced_align` ниже — повтор `torchaudio.functional.forced_align` на numpy)
по гипотезе модели. Виттерби-путь даёт РАЗМЕТКУ ВСЕХ КАДРОВ: где какой токен,
где blank. И вот главное: **кадры blank между буквами — это тоже звук слова**,
их нельзя выбрасывать. Поэтому границу слова расширяем в соседний blank-провал,
но не дальше его середины — тогда окна соседних слов не могут пересечься, — и
не дальше `MAX_EDGE_EXTEND_MS` (иначе слово перед секундной паузой «съест»
полсекунды тишины).

ПОЧЕМУ ИМЕННО forced_align, если путь Виттерби по собственной гипотезе модели
совпадает с argmax-путём. Честно: сами КАДРЫ ЭМИССИИ он не меняет (argmax-путь
максимален по каждому кадру, значит он же и оптимален под свою же гипотезу).
Выигрыш даёт разметка кадров blank и длина прогонов токенов, которых у
`frames_to_words` просто нет. Плюс forced_align даёт СКОР на каждый кадр —
отсюда настоящий confidence слова (у GigaAM его не было, в транскрипте лежал
`score: null`) — и позволяет позже выровнять ИСПРАВЛЕННЫЙ текст по тому же
аудио (поиск мата по посториорам, если ASR услышал не то).

Модуль намеренно не знает про gigaam: на вход — тензор log-probs, id токенов и
функция «id → символ». Поэтому проверяется синтетическими пиками в тестах, без
модели и без аудио.

ЧТО ЭТО ДАЛО НА ЗАМЕРЕ. Контрольный A/B на 15 минутах реального стрима: один и
тот же код, во втором прогоне `align_words` подменён на заглушку-None (тот самый
путь отката ниже), правда — таймлайн человека (142 мьюта), сверка по пересечению
во времени на 123 общих совпадениях.

* текст побайтово тот же, ни одно окно не сузилось: начало −40 мс медиана,
  конец +40 (максимум −121/+160);
* `score` появился у 2122 слов из 2122 — до этого не было ни у одного;
* по СЫРЫМ окнам слов утечка мата (окно не достаёт до границы человека)
  19.4 → 14.6 с суммарно, медиана слева 51 → 24 мс, справа 27 → 0;
* по ГОТОВЫМ мьютам утечка 15.2 → 12.7 с, «не течёт ни с одной стороны»
  29 → 38 из 123, |Δконец| медиана 73 → 53 мс, а системный сдвиг со знаком
  −40/+17 мс стал 0/0 — опоздание границ исчезло;
* поджим по RMS срабатывает ВТРОЕ реже (127 → 52 слева, 137 → 67 справа): он и
  нужен был как костыль под неверные окна ASR.

ВАЖНО ПРО ГРАНИЦЫ МЬЮТА. Само окно мьюта от расширения слова почти не растёт:
`decisions._compute_mute_window` уже прижимает padding к концу предыдущего и
началу следующего слова, а расширенное слово одновременно двигает и эту границу.
Две механики делают одно и то же, поэтому |Δначало| у мьютов не изменилось
(78 мс против 79). Выигрыш — в правде на уровне СЛОВА (её потребляют поджим по
RMS, audit и будущий поиск мата по посториорам) и в `score`.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any, Callable, Sequence

import numpy as np

logger = logging.getLogger(__name__)

MAX_EDGE_EXTEND_MS = 120
"""Насколько слову разрешено расширяться в blank-провал с каждой стороны.

Кадр CTC у GigaAM ≈ 40 мс, задержка эмиссии обычно 1–3 кадра, замеренная
ошибка границ на реальном прогоне — 77–100 мс. Отсюда 120 мс: хватает покрыть
задержку и хвост звука, но слово перед долгой паузой не залезает в тишину.
Дальше окно всё равно поджимает `mute_audio_trim` по RMS (он умеет только
СУЖАТЬ), поэтому щедрая граница безопаснее скупой."""


@dataclass
class TokenSpan:
    """Прогон одного токена на пути выравнивания: кадры [start, end)."""

    token: int
    start: int
    end: int
    score: float
    """Средний log-prob токена по его кадрам."""


def forced_align(
    emission: np.ndarray, targets: Sequence[int], *, blank: int
) -> tuple[np.ndarray, np.ndarray]:
    """CTC forced alignment: метка и её log-prob на КАЖДОМ кадре.

    Повтор CPU-ядра `torchaudio.functional.forced_align` (torchaudio 2.6) —
    тот же Витерби по расширенной blank'ами цепочке, то же окно допустимых
    состояний на каждом кадре и тот же порядок выбора при равенстве (прыжок
    через blank, потом сосед, потом остаться), всё во float32. Поэтому путь
    совпадает с torchaudio кадр в кадр, а значит и границы слов.

    `emission` — [T, C] log-softmax, `targets` — гипотеза без blank.
    Кадров меньше, чем нужно гипотезе (с blank'ом между повторами), →
    ValueError: выровнять такую гипотезу нельзя в принципе.
    """
    emission = np.asarray(emission, dtype=np.float32)
    tg = np.asarray(list(targets), dtype=np.int64)
    n_frames = int(emission.shape[0])
    n_targets = int(tg.shape[0])
    if n_targets == 0:
        raise ValueError("пустая гипотеза")
    if np.any(tg == blank):
        raise ValueError("в гипотезе не должно быть blank")
    n_states = 2 * n_targets + 1
    repeats = int(np.sum(tg[1:] == tg[:-1])) if n_targets > 1 else 0
    if n_frames < n_targets + repeats:
        raise ValueError(
            f"кадров {n_frames} меньше, чем нужно гипотезе ({n_targets} токенов, {repeats} повторов)"
        )

    labels = np.full(n_states, blank, dtype=np.int64)
    labels[1::2] = tg
    # Прыжок через blank (i-2 → i) — только на букву и только если она не
    # повтор предыдущей: между одинаковыми буквами blank обязателен.
    can_skip = np.zeros(n_states, dtype=bool)
    if n_targets > 1:
        odd = np.arange(3, n_states, 2)
        can_skip[odd] = tg[odd // 2] != tg[odd // 2 - 1]

    neg_inf = np.float32(-np.inf)
    alphas = np.full(n_states, neg_inf, dtype=np.float32)
    back = np.zeros((n_frames, n_states), dtype=np.int8)
    span = n_targets + repeats
    start = 0 if n_frames - span > 0 else 1
    end = 1 if n_states == 1 else 2
    for i in range(start, end):
        alphas[i] = emission[0, labels[i]]

    for t in range(1, n_frames):
        if n_frames - t <= span:
            if start % 2 == 1 and start // 2 + 1 < n_targets and tg[start // 2] != tg[start // 2 + 1]:
                start += 1
            start += 1
        if t <= span:
            if end % 2 == 0 and end < 2 * n_targets and tg[end // 2 - 1] != tg[end // 2]:
                end += 1
            end += 1
        current = np.full(n_states, neg_inf, dtype=np.float32)
        loop_start = start
        if start == 0:
            current[0] = alphas[0] + emission[t, blank]
            back[t, 0] = 0
            loop_start = 1
        if loop_start < end:
            idx = np.arange(loop_start, end)
            x0 = alphas[idx]
            x1 = alphas[idx - 1]
            x2 = np.where(can_skip[idx], alphas[np.maximum(idx - 2, 0)], neg_inf)
            take2 = (x2 > x1) & (x2 > x0)
            take1 = ~take2 & (x1 > x0) & (x1 > x2)
            best = np.where(take2, x2, np.where(take1, x1, x0))
            current[idx] = best + emission[t, labels[idx]]
            back[t, idx] = np.where(take2, 2, np.where(take1, 1, 0))
        alphas = current

    state = n_states - 1 if alphas[n_states - 1] > alphas[n_states - 2] else n_states - 2
    path = np.empty(n_frames, dtype=np.int64)
    for t in range(n_frames - 1, -1, -1):
        path[t] = labels[state]
        state -= int(back[t, state])
    scores = emission[np.arange(n_frames), path]
    return path, scores


def merge_tokens(path: np.ndarray, scores: np.ndarray, *, blank: int) -> list[TokenSpan]:
    """Путь по кадрам → прогоны токенов без blank (`torchaudio.functional.merge_tokens`)."""
    path = np.asarray(path)
    if path.size == 0:
        return []
    changes = np.nonzero(np.diff(path, prepend=-1, append=-1) != 0)[0].tolist()
    spans: list[TokenSpan] = []
    for start, end in zip(changes[:-1], changes[1:]):
        token = int(path[start])
        if token == blank:
            continue
        spans.append(
            TokenSpan(
                token=token,
                start=int(start),
                end=int(end),
                score=float(np.mean(scores[start:end], dtype=np.float32)),
            )
        )
    return spans


@dataclass
class AlignedWord:
    """Слово с границами в СЕКУНДАХ от начала сегмента (как у gigaam.Word)."""

    text: str
    start: float
    end: float
    score: float | None = None
    """Средняя вероятность букв слова (0..1) — из скоров пути forced_align."""


def align_words(
    log_probs: Any,
    token_ids: Sequence[int],
    *,
    blank_id: int,
    to_char: Callable[[int], str],
    frame_shift: float,
    max_extend_ms: int = MAX_EDGE_EXTEND_MS,
) -> list[AlignedWord] | None:
    """Границы слов по пути forced alignment.

    `log_probs` — [T, C] log-softmax ОДНОГО сегмента (numpy, float32),
    `token_ids` — гипотеза модели теми же id, что вернул greedy-декод,
    `to_char` — «id → строка» словаря (для charwise-словаря это буква,
    для sentencepiece — кусок с префиксом «▁» на границе слова),
    `frame_shift` — секунд на кадр энкодера.

    Возвращает None, если выровнять не удалось: вызывающий код тогда падает
    обратно на кадры эмиссии — тайминги хуже, но прогон не рушится.
    """
    ids = [int(t) for t in token_ids]
    if not ids:
        return []

    try:
        emission = np.asarray(log_probs, dtype=np.float32)
        if emission.ndim != 2:
            return None
        n_frames = int(emission.shape[0])
        path, scores = forced_align(emission, ids, blank=blank_id)
        spans = merge_tokens(path, scores, blank=blank_id)
    except Exception as exc:  # noqa: BLE001 — короткий сегмент, повторы, битый blank
        logger.warning("forced_align не сошёлся (%s) — тайминги по кадрам эмиссии", exc)
        return None

    if len(spans) != len(ids):
        # Путь обязан выдать ровно нашу гипотезу; если нет — молчать нельзя,
        # но и рушить прогон незачем.
        logger.warning(
            "forced_align вернул %d токенов вместо %d — тайминги по кадрам эмиссии",
            len(spans),
            len(ids),
        )
        return None

    groups = _group_into_words(ids, spans, to_char)
    if not groups:
        return []

    max_extend = max(0, int(round(max_extend_ms / 1000.0 / frame_shift))) if frame_shift > 0 else 0
    words: list[AlignedWord] = []
    for i, (text, first, last, probs) in enumerate(groups):
        prev_end = groups[i - 1][2] if i else 0
        next_start = groups[i + 1][1] if i + 1 < len(groups) else n_frames
        # Половина провала — не «на всякий случай», а гарантия: окна соседних
        # слов от такого расширения пересечься не могут.
        left = min(max_extend, max(0, (first - prev_end) // 2))
        right = min(max_extend, max(0, (next_start - last) // 2))
        words.append(
            AlignedWord(
                text=text,
                start=max(0.0, (first - left) * frame_shift),
                end=min(n_frames * frame_shift, (last + right) * frame_shift),
                score=round(sum(probs) / len(probs), 4) if probs else None,
            )
        )
    return words


def _group_into_words(
    ids: Sequence[int],
    spans: Sequence[Any],
    to_char: Callable[[int], str],
) -> list[tuple[str, int, int, list[float]]]:
    """Токены → слова: (текст, первый кадр, последний кадр+1, вероятности букв).

    Правило границы слова — то же, что у `gigaam.timestamps_utils.frames_to_words`
    (пробел или префикс «▁»), иначе текст слов разъедется с транскриптом.
    """
    import math

    out: list[tuple[str, int, int, list[float]]] = []
    chars: list[str] = []
    first = 0
    last = 0
    probs: list[float] = []

    def commit() -> None:
        nonlocal chars, probs
        text = "".join(chars).strip()
        chars = []
        saved, probs = probs, []
        if text:
            out.append((text, first, last, saved))

    for token_id, span in zip(ids, spans):
        char = to_char(token_id)
        if char.startswith("▁"):
            commit()
            char = char[1:]
        elif char == " ":
            commit()
            continue
        if not chars:
            first = int(span.start)
        last = int(span.end)
        chars.append(char)
        probs.append(math.exp(float(span.score)))

    commit()
    return out
