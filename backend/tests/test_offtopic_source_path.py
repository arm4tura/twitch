"""Фаза 3: путь классификации по исходнику (без сети — LLMClient подменён).

Собираем настоящий Alignment из пары транскриптов, а вызовы LLM подменяем
фейком, который отвечает по типу промпта (план / классификация / один проход).
Проверяем то, ради чего путь и существует:

- батчинг реплик и один вызов на план (число вызовов LLM = 1 + число батчей);
- батч без ответа ПОВТОРЯЕТСЯ, а на повторном отказе учитывается как пробел
  покрытия, а не как «по теме» (регрессия, из-за которой появилась Фаза 1);
- сегмент с выдуманным таймкодом (не попал ни в одну реплику) выбрасывается;
- границы модели снапятся на реплики;
- деградация без исходника (classify_single_pass) всё равно даёт сегменты.
"""

from __future__ import annotations

import logging
import re

import pytest

from twitch_cut.llm_offtopic import (
    classify_single_pass,
    classify_with_source,
)
from twitch_cut.llm_offtopic.client import Completion
from twitch_cut.llm_offtopic.providers import AIProvider, ProviderFormat


def _norm(w: str) -> str:
    return re.sub(r"[^а-яёa-z0-9]", "", w.lower())


def _transcript(pairs: list[tuple[str, float]], *, step: float = 1.0) -> dict:
    """[(слово, начало)] → transcript-dict с одним сегментом."""
    words = [
        {"word": w, "start": t, "end": t + step / 2} for w, t in pairs
    ]
    return {
        "segments": [
            {
                "start": words[0]["start"],
                "end": words[-1]["end"],
                "words": words,
            }
        ]
    }


def _seq(prefix: str, n: int, start: float, *, step: float = 1.0):
    return [(f"{prefix}{i}", start + i * step) for i in range(n)]


# Исходник: 24 уникальных слова. Стрим: играет видео (эхо исходника) → монолог
# стримера своими словами → снова видео. Между дорожками — 4-граммы для привязки.
SOURCE_WORDS = [f"вид{i}" for i in range(24)]
SOURCE = _transcript([(w, i * 1.0) for i, w in enumerate(SOURCE_WORDS)])

# Стрим: вид0..вид9 (0–9с, видео), свой0..свой9 (10–19с, реплика), вид10..вид19 (20–29с).
STREAM = _transcript(
    [(f"вид{i}", float(i)) for i in range(10)]
    + [(f"свой{i}", 10.0 + i) for i in range(10)]
    + [(f"вид{i}", 20.0 + (i - 10)) for i in range(10, 20)]
)


class _FakeClient:
    """LLMClient-заглушка: отвечает по типу промпта и номеру попытки.

    :param outline: Completion для промпта плана.
    :param classify: список Completion по попыткам классификационного промпта
        (первый элемент — первая попытка, второй — ретрай, …). Если список
        короче числа попыток, повторяется последний.
    """

    def __init__(self, *, outline: Completion, classify: list[Completion]):
        self.provider = AIProvider(
            id="fake",
            label="Fake",
            format=ProviderFormat.ANTHROPIC,
            models=["fake-model"],
        )
        self._outline = outline
        self._classify = classify
        self.calls: list[str] = []
        self._classify_attempts = 0

    def complete(self, prompt: str, *, model=None, max_tokens=None, **kw) -> Completion:
        if "Составь краткий план" in prompt:
            self.calls.append("outline")
            return self._outline
        # и source_classification, и single_pass — это «список сегментов».
        self.calls.append("classify")
        i = min(self._classify_attempts, len(self._classify) - 1)
        self._classify_attempts += 1
        return self._classify[i]


def _outline_ok() -> Completion:
    return Completion(text="ТЕМА: счёт по-русски\n- 00:00 перечисление чисел")


def _segments(json_body: str) -> Completion:
    return Completion(text=f"```json\n{json_body}\n```")


# --- classify_with_source ---------------------------------------------------


def test_source_path_finds_and_snaps_segment(caplog):
    # Классификация возвращает 00:10–00:20 — ровно вокруг монолога стримера.
    client = _FakeClient(
        outline=_outline_ok(),
        classify=[
            _segments(
                '{"segments": [{"start": "00:10", "end": "00:20", '
                '"confidence": 0.85, "reason": "занялся своим", "quote": "свой0"}]}'
            )
        ],
    )
    with caplog.at_level(logging.INFO, logger="twitch_cut.llm_offtopic.classifier"):
        result = classify_with_source(STREAM, SOURCE, client=client, turn_batch=30)

    # 1 план + 1 батч (реплик мало).
    assert client.calls == ["outline", "classify"]
    assert len(result.segments) == 1
    seg = result.segments[0]
    # Снап на реплику: начало первой её реплики ≈ 10с.
    assert seg.start_s == pytest.approx(10.0, abs=0.6)
    assert seg.stage == "source"
    assert seg.topic == "счёт по-русски"
    assert result.model == "fake-model"
    # Фаза 4: путь помечает себя, а независимые улики (привязка/пауза) едут
    # в сам сегмент, а не только в лог.
    assert result.method == "source_anchored"
    assert seg.anchored is not None and 0.0 <= seg.anchored <= 1.0
    assert seg.pause_s is not None and seg.pause_s >= 0.0


def test_source_meta_propagates_and_survives_empty():
    # Хеши/модель исходника — audit trail, должны доехать до OfftopicSet даже
    # на пустом результате (метод у пустого результата всё равно source_anchored).
    client = _FakeClient(outline=_outline_ok(), classify=[_segments('{"segments": []}')])
    result = classify_with_source(
        STREAM,
        SOURCE,
        client=client,
        turn_batch=30,
        transcript_hash="strh",
        source_transcript_hash="srch",
        source_model="gigaam-v3",
    )
    assert result.segments == []
    assert result.method == "source_anchored"
    assert result.transcript_hash == "strh"
    assert result.source_transcript_hash == "srch"
    assert result.source_model == "gigaam-v3"


def test_source_path_batches_turns():
    # turn_batch=1 при (здесь) одной реплике = 1 батч; проверяем, что дробление
    # по батчам вообще работает и не теряет вызов плана.
    client = _FakeClient(outline=_outline_ok(), classify=[_segments('{"segments": []}')])
    result = classify_with_source(STREAM, SOURCE, client=client, turn_batch=1)
    assert client.calls.count("outline") == 1
    assert client.calls.count("classify") >= 1
    assert result.segments == []


def test_empty_batch_answer_is_retried_then_counted(caplog):
    # Первая попытка — пустой ответ (как отдавала прокси на большом промпте),
    # вторая — валидный список. Батч НЕ теряется благодаря ретраю.
    client = _FakeClient(
        outline=_outline_ok(),
        classify=[
            Completion(text="", stop_reason="max_tokens"),  # пусто → ретрай
            _segments(
                '{"segments": [{"start": "00:10", "end": "00:19", '
                '"confidence": 0.7, "reason": "своё"}]}'
            ),
        ],
    )
    result = classify_with_source(STREAM, SOURCE, client=client, turn_batch=30)
    # два обращения к классификации по одному батчу = попытка + ретрай.
    assert client.calls == ["outline", "classify", "classify"]
    assert len(result.segments) == 1


def test_failed_batch_is_not_silently_clean(caplog):
    # Оба обращения пусты → батч без ответа. Его реплики НЕ проверены — это
    # должно попасть в лог как пробел, а не молча стать «по теме».
    client = _FakeClient(
        outline=_outline_ok(),
        classify=[Completion(text="", stop_reason="max_tokens")],
    )
    with caplog.at_level(logging.WARNING, logger="twitch_cut.llm_offtopic.classifier"):
        result = classify_with_source(STREAM, SOURCE, client=client, turn_batch=30)

    assert result.segments == []
    assert any("не проверены" in r.getMessage() for r in caplog.records)


def test_hallucinated_timecode_is_dropped(caplog):
    # Модель выдала сегмент 05:00–05:30 — там нет ни одной реплики стримера
    # (это время вообще вне транскрипта). Такой сегмент выбрасываем.
    client = _FakeClient(
        outline=_outline_ok(),
        classify=[
            _segments(
                '{"segments": [{"start": "05:00", "end": "05:30", '
                '"confidence": 0.9, "reason": "выдумано"}]}'
            )
        ],
    )
    with caplog.at_level(logging.INFO, logger="twitch_cut.llm_offtopic.classifier"):
        result = classify_with_source(STREAM, SOURCE, client=client, turn_batch=30)

    assert result.segments == []
    assert any("не попал ни в одну реплику" in r.getMessage() for r in caplog.records)


def test_low_match_ratio_warns_but_does_not_crash(caplog):
    # Стрим и «исходник» без общих слов: привязок нет, вся речь уедет стримеру.
    # Не падаем, но громко предупреждаем — иначе весь VOD станет оффтопом молча.
    unrelated_source = _transcript([(f"чужое{i}", i * 1.0) for i in range(24)])
    client = _FakeClient(outline=_outline_ok(), classify=[_segments('{"segments": []}')])
    with caplog.at_level(logging.WARNING, logger="twitch_cut.llm_offtopic.classifier"):
        classify_with_source(STREAM, unrelated_source, client=client, turn_batch=30)
    assert any("не тот файл" in r.getMessage() for r in caplog.records)


# --- classify_single_pass (деградация) --------------------------------------


def test_single_pass_finds_segment_without_source():
    transcript = {
        "segments": [
            {"start": 0.0, "end": 30.0, "text": "разбираем видео про котиков"},
            {"start": 30.0, "end": 90.0, "text": "а вообще мне надо ответить в чате про донат"},
            {"start": 90.0, "end": 120.0, "text": "ладно вернёмся к котикам"},
        ]
    }
    client = _FakeClient(
        outline=_outline_ok(),
        classify=[
            _segments(
                '{"topic": "котики", "segments": [{"start": "00:30", "end": "01:30", '
                '"confidence": 0.8, "reason": "чат и донат"}]}'
            )
        ],
    )
    result = classify_single_pass(transcript, client=client)
    # Плана здесь нет — только классификация.
    assert client.calls == ["classify"]
    assert len(result.segments) == 1
    seg = result.segments[0]
    assert seg.stage == "single-pass"
    assert seg.topic == "котики"
    # Снап на ASR-сегмент 30–90.
    assert (seg.start_s, seg.end_s) == pytest.approx((30.0, 90.0))
    # Фаза 4: одиночный проход помечает метод; независимых улик нет (нет
    # исходника) → pause_s/anchored пустые.
    assert result.method == "single_pass"
    assert seg.pause_s is None
    assert seg.anchored is None


def test_single_pass_empty_transcript_returns_empty():
    client = _FakeClient(outline=_outline_ok(), classify=[_segments('{"segments": []}')])
    result = classify_single_pass({"segments": []}, client=client)
    assert result.segments == []
    assert result.method == "single_pass"
    assert client.calls == []  # ничего не спрашивали
