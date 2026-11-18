"""Тесты `keyword_spotting.spot_keywords` — поиск мата по посториорам CTC.

Проверять надо на бумаге, поэтому здесь синтетические log-probs: CTC пиковый,
значит буква — это одиночный всплеск на фоне blank. Расставляем всплески руками
и знаем правильный ответ заранее. Модель и аудио не нужны.
"""

from __future__ import annotations

import numpy as np
import pytest

from twitch_cut.keyword_spotting import MAX_GAP_FRAMES, MIN_KEYWORD_LEN, spot_keywords

VOCAB = ["б", "л", "я", "т", "х", "у", "й", "н", "а", " "]
IDX = {char: i for i, char in enumerate(VOCAB)}
BLANK = len(VOCAB)
FRAME_SHIFT = 0.04  # как у GigaAM v3: 40 мс на кадр энкодера


def char_to_id(char: str) -> int | None:
    return IDX.get(char)


def _emission(n_frames: int, spikes: dict[int, tuple[str, float]]) -> np.ndarray:
    """[T, C] log-probs: blank почти везде, на кадрах из `spikes` — своя буква.

    Вероятность всплеска задаём явно: при 0.9 буква — argmax (её увидел бы и
    greedy-декод), при 0.3 argmax остаётся blank, и буква видна ТОЛЬКО в
    посториорах — ровно тот случай, за которым весь модуль.
    """
    n_classes = len(VOCAB) + 1
    filler = 0.02
    probs = np.full((n_frames, n_classes), filler / (n_classes - 1), dtype=np.float32)
    probs[:, BLANK] = 1.0 - filler
    for frame, (char, prob) in spikes.items():
        probs[frame] = filler / (n_classes - 2)
        probs[frame, IDX[char]] = prob
        probs[frame, BLANK] = 1.0 - filler - prob
    return np.log(probs)


def _chain(word: str, first: int, step: int, prob: float) -> dict[int, tuple[str, float]]:
    """Всплески по буквам слова: каждые `step` кадров, все с одной силой."""
    return {first + i * step: (char, prob) for i, char in enumerate(word)}


def _spot(emission, keywords, *, min_score: float = 0.5, **kwargs):
    return spot_keywords(
        emission,
        keywords,
        char_to_id=char_to_id,
        blank_id=BLANK,
        frame_shift=FRAME_SHIFT,
        min_score=min_score,
        **kwargs,
    )


def _dp_only(emission, keywords, *, min_score: float = 0.5, **kwargs):
    """То же, но без фильтра «чего нет в тексте».

    Механику DP (стоимость разрывов, ограничение на длину разрыва, подавление
    пересечений, границы по кадрам) удобно мерить на ГРОМКИХ пиках 0.9: тогда
    скор считается на бумаге и не зависит от того, что там решил argmax. Но
    громкая буква — это буква, которую слышит и greedy-декод, то есть
    `require_unheard` такого кандидата законно выкинет. Поэтому тесты механики
    просят DP без фильтра, а сам фильтр проверяют тесты ниже — на тихих пиках,
    как в жизни."""
    return _spot(emission, keywords, min_score=min_score, require_unheard=False, **kwargs)


def test_finds_the_root_the_greedy_decode_threw_away():
    """Главный смысл модуля: argmax на всех кадрах — blank, то есть в ТЕКСТЕ
    ASR здесь пусто, а корень в посториорах виден и находится."""
    emission = _emission(20, _chain("блят", 5, 2, 0.3))

    hits = _spot(emission, ["блят"], min_score=0.25)

    assert [h.keyword for h in hits] == ["блят"]
    assert hits[0].score == pytest.approx(0.3, abs=0.01)
    # Границы — по кадрам эмиссии первой и последней буквы: 5 и 11 (+1 кадр).
    assert hits[0].start == pytest.approx(5 * FRAME_SHIFT)
    assert hits[0].end == pytest.approx(12 * FRAME_SHIFT)


def test_score_is_dragged_down_by_the_weakest_letter():
    """Скор — средняя геометрическая по буквам, поэтому одна невнятная буква
    роняет весь кандидат. Три буквы по 0.9 и одна 0.3 дают 0.684, минус копейка
    за три blank-кадра в разрывах (по 0.98) — 0.674."""
    spikes = _chain("блят", 5, 2, 0.9)
    spikes[9] = ("я", 0.3)
    emission = _emission(20, spikes)

    assert _spot(emission, ["блят"], min_score=0.6)[0].score == pytest.approx(0.674, abs=0.005)
    assert _spot(emission, ["блят"], min_score=0.7) == []


def test_a_foreign_letter_inside_the_gap_kills_the_chain():
    """Главная защита от ложных: в разрыве между буквами корня должен быть
    blank. Иначе «ебат» находится внутри «брат» (б, а, т идут по порядку, «е» —
    из соседнего слова), а «ебло» — внутри «было»: именно так первая версия
    выдала на реальном прогоне 16 ложных кандидатов со скором до 1.00.

    Здесь ровно этот случай на бумаге: ищем «блят» там, где сказано «б_я_т», а
    на месте «л» выстрелила «х» — чужая буква, оплатить её blank-ом нельзя.
    """
    brat = _emission(20, {5: ("б", 0.9), 7: ("х", 0.9), 9: ("я", 0.9), 11: ("т", 0.9)})

    assert _spot(brat, ["блят"], min_score=0.3) == []


def test_blank_gap_is_almost_free_but_a_letter_is_not():
    """Кадр blank в разрыве стоит около -0.02 (вероятность 0.98) и на скор почти
    не влияет; кадр с чужой буквой — около -3, и цепочка проваливается под
    порог. Один и тот же ключ на одном и том же месте: с тишиной в разрыве
    находится, с чужой буквой — нет."""
    quiet = _emission(20, _chain("хуйн", 5, 3, 0.9))
    noisy = _emission(20, {**_chain("хуйн", 5, 3, 0.9), 6: ("а", 0.9)})

    assert _dp_only(quiet, ["хуйн"])[0].score == pytest.approx(0.87, abs=0.02)
    assert _dp_only(noisy, ["хуйн"]) == []


def test_letters_too_far_apart_are_not_a_word():
    """Разрыв между всплесками соседних букв ограничен `MAX_GAP_FRAMES` (6
    кадров ≈ 240 мс). Шаг ровно на границе ещё слово, шаг за границей — уже
    буквы из разных слов, и цепочка обязана рассыпаться, даже если между ними
    чистая тишина: иначе корень соберётся через полсекунды паузы."""
    inside = _emission(60, _chain("блят", 5, MAX_GAP_FRAMES, 0.9))
    outside = _emission(60, _chain("блят", 5, MAX_GAP_FRAMES + 1, 0.9))

    assert len(_dp_only(inside, ["блят"])) == 1
    assert _dp_only(outside, ["блят"]) == []


def test_one_candidate_per_place_not_a_dozen():
    """DP набирает выше порога на нескольких соседних кадрах подряд; наружу
    должен выходить один — самый уверенный — кандидат."""
    spikes = _chain("блят", 5, 2, 0.9)
    spikes[12] = ("т", 0.8)  # второй возможный конец цепочки, окна пересекаются

    hits = _dp_only(_emission(20, spikes), ["блят"])

    assert len(hits) == 1
    assert hits[0].end_frame == 12  # выиграл конец с более сильной «т» (кадр 11)


def test_two_roots_on_one_word_leave_the_stronger_one():
    """«хуйн» и «уйна» ловят одно и то же место. Наружу должен выйти один
    кандидат, иначе один «хуйня» превратится в пачку мьютов."""
    emission = _emission(24, _chain("хуйна", 5, 2, 0.9))

    hits = _dp_only(emission, ["хуйн", "уйна"])

    assert len(hits) == 1


def test_short_keys_are_not_searched_at_all():
    """Трёхбуквенные ключи на реальном прогоне дают кандидатов пачками на
    невинной речи — искать их запрещено даже при идеальных пиках."""
    emission = _emission(20, _chain("бля", 5, 2, 0.95))

    assert MIN_KEYWORD_LEN == 4
    assert _spot(emission, ["бля"], min_score=0.3) == []


def test_letter_outside_the_vocab_is_skipped_not_a_crash():
    """Словарь charwise-модели не содержит латиницы: такой ключ искать нечем,
    но прогон из-за него падать не должен."""
    emission = _emission(20, _chain("блят", 5, 2, 0.9))

    assert _dp_only(emission, ["fuck", "блят"]) != []
    assert [h.keyword for h in _dp_only(emission, ["fuck"])] == []


def test_repeated_letter_needs_a_blank_between_the_peaks():
    """Правило CTC: две одинаковые буквы подряд без blank между ними
    декодируются как ОДНА. Значит цепочка «аа» на соседних кадрах невозможна."""
    glued = _emission(20, {5: ("а", 0.9), 6: ("а", 0.9), 8: ("б", 0.9), 10: ("л", 0.9)})
    spaced = _emission(20, {5: ("а", 0.9), 7: ("а", 0.9), 9: ("б", 0.9), 11: ("л", 0.9)})

    assert _dp_only(glued, ["аабл"]) == []
    assert len(_dp_only(spaced, ["аабл"])) == 1


def test_frame_shift_scales_the_boundaries():
    """frame_shift считается на каждый сегмент, поэтому границы обязаны быть
    линейны по нему, а не по константе 40 мс."""
    emission = _emission(20, _chain("блят", 5, 2, 0.9))

    slow = spot_keywords(
        emission,
        ["блят"],
        char_to_id=char_to_id,
        blank_id=BLANK,
        frame_shift=0.08,
        min_score=0.5,
        require_unheard=False,
    )
    fast = _dp_only(emission, ["блят"])

    assert slow[0].start == pytest.approx(fast[0].start * 2)
    assert slow[0].end == pytest.approx(fast[0].end * 2)


def test_silence_gives_no_candidates():
    """Порог должен молчать там, где ничего не звучало: иначе спотинг завалит
    таймлайн мьютами на тишине."""
    assert _spot(_emission(50, {}), ["блят", "хуйн"], min_score=0.2) == []


def test_what_the_text_already_heard_is_not_a_candidate():
    """`require_unheard` — граница между стадиями: то, что попало в текст ASR,
    разбирает текстовый детектор (словарь, леммы, омографы), и второй раз это
    место предлагать нельзя, иначе один «блять» приедет двумя мьютами.

    Одни и те же пики, разная громкость: на 0.9 буквы — argmax, то есть они
    ЕСТЬ в тексте, и кандидата не будет; на 0.3 argmax — blank, в тексте пусто,
    и кандидат нужен."""
    loud = _emission(20, _chain("блят", 5, 2, 0.9))
    quiet = _emission(20, _chain("блят", 5, 2, 0.3))

    assert _spot(loud, ["блят"], min_score=0.25) == []
    assert [h.keyword for h in _spot(quiet, ["блят"], min_score=0.25)] == ["блят"]


def test_a_letter_the_greedy_path_squeezed_in_still_reaches_review():
    """Фильтр сравнивает ПОДРЯД идущие метки greedy-пути, и это осознанно: DP
    разрешает разрывы, greedy — нет. Здесь в разрыве стоит невнятная чужая буква
    (0.50 против blank 0.48): argmax её берёт, поэтому в тексте стоит «блхят» и
    ключа там нет, а DP оплачивает тот же кадр как blank и кандидата отдаёт.

    Так на реальном прогоне выжил единственный ложный кандидат («ебет» внутри
    «нефрибеты»). Тест держит документацию честной: фильтр снимает буквальные
    вхождения ключа в текст, а не всё созвучное, — остаток гасит статус review.
    """
    spikes = _chain("блят", 5, 2, 0.9)
    spikes[8] = ("х", 0.50)  # между «л» (7) и «я» (9): argmax — «х», но blank рядом

    hits = _spot(_emission(20, spikes), ["блят"], min_score=0.5)

    assert [h.keyword for h in hits] == ["блят"]
