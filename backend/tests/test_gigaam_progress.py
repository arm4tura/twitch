"""Тесты прогресса распознавания GigaAM (`_SpeechProgress`).

Раньше прогресс снимался патчем внутренностей пакета gigaam (обёртки на
`model._decode` и `vad_utils.segment_audio_file`), теперь long-form путь наш,
и счётчик зовётся явно: `found` после VAD, `advance` после каждого батча.
Контракт прежний: дробь считается по аудио-секундам речи, без речи не тикаем,
сообщения человекочитаемые.
"""

from __future__ import annotations

import pytest

from twitch_cut.gigaam_asr import _SpeechProgress


def _play(progress: _SpeechProgress, boundaries, batches) -> None:
    """Проиграть прогон: VAD нашёл куски, потом батчи распознавания."""
    progress.found(boundaries)
    for size in batches:
        progress.advance(size)


def test_fraction_weighted_by_audio_seconds_not_segment_count():
    """Дробь = обработанные аудио-секунды / общие, а НЕ «сегмент N из M».

    Сегменты сильно разной длины: 60 + 10 + 10 + 160 = 240 c. После первого
    батча (3 сегмента) обработано 80 c = 33%, хотя по штукам это было бы 75%.
    Счёт по штукам врал бы на записях, где в начале много коротких реплик —
    ETA скакала бы.
    """
    ticks: list[tuple[float, str]] = []
    boundaries = [(0.0, 60.0), (60.0, 70.0), (70.0, 80.0), (80.0, 240.0)]
    _play(_SpeechProgress(lambda f, m="": ticks.append((f, m))), boundaries, batches=[3, 1])

    fractions = [f for f, _ in ticks]
    assert fractions[0] == 0.0, "первый тик — «речь найдена», работа ещё не начата"
    assert fractions[1] == pytest.approx(80.0 / 240.0)
    assert fractions[-1] == pytest.approx(1.0)
    assert fractions == sorted(fractions), "дробь не должна убывать"


def test_no_callback_is_a_noop():
    """on_progress=None — счётчик просто молчит, ничего не падает."""
    _play(_SpeechProgress(None), [(0.0, 60.0)], batches=[1])


def test_zero_speech_does_not_divide_by_zero():
    """Аудио без речи: boundaries пустые, total = 0 — деления на ноль нет."""
    ticks: list[float] = []
    _play(_SpeechProgress(lambda f, m="": ticks.append(f)), boundaries=[], batches=[1])

    assert ticks == [], "без речи тикать нечем"


def test_last_batch_always_reports_completion():
    """Троттлинг (~2 тика в секунду) не должен съесть финальные 100%:
    батчи здесь идут быстрее полсекунды, но последний тик — всегда 1.0."""
    ticks: list[float] = []
    boundaries = [(float(i), float(i + 1)) for i in range(10)]
    _play(_SpeechProgress(lambda f, m="": ticks.append(f)), boundaries, batches=[1] * 10)

    assert ticks[-1] == pytest.approx(1.0)


def test_message_reports_minutes_of_speech():
    """Сообщение тика человекочитаемое: «распознано X / Y мин речи».

    Оно уходит в лог джобы и в UI, поэтому формат — часть контракта.
    """
    ticks: list[tuple[float, str]] = []
    boundaries = [(0.0, 60.0), (60.0, 120.0)]
    _play(_SpeechProgress(lambda f, m="": ticks.append((f, m))), boundaries, batches=[2])

    assert "фрагмент" in ticks[0][1], "первый тик сообщает, сколько речи найдено"
    assert ticks[-1][1] == "распознано 2.0 / 2.0 мин речи"


def test_speech_search_ticks_go_to_status_not_fraction():
    """Пока идёт VAD, тики уходят строкой состояния, а не дробью — иначе UI
    поставил бы якорь ETA на поиске речи и посчитал скорость распознавания
    по нему. Последний тик поиска приходит всегда."""
    fractions: list[float] = []
    statuses: list[str] = []
    progress = _SpeechProgress(lambda f, m="": fractions.append(f), statuses.append)
    for done in range(1, 11):
        progress.searching(done, 10)

    assert fractions == []
    assert statuses[-1] == "поиск речи: 100%"
