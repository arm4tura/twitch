"""Фаза 1: обрезанный ответ модели — это НЕ «по теме».

Замер на реальном VOD: 13 окон из 30 закончились на stop_reason=max_tokens с
пустым text (бюджет ушёл в скрытый thinking), и все 13 молча зачлись как «нет
оффтопа». Тесты ниже фиксируют три состояния вердикта и то, что классификатор
пробел в покрытии не выдаёт за чистое окно.
"""

from __future__ import annotations

import logging

import pytest

from twitch_cut.llm_offtopic import Completion
from twitch_cut.llm_offtopic.classifier import classify_transcript
from twitch_cut.llm_offtopic.prompt import parse_classification_response
from twitch_cut.llm_offtopic.providers import AIProvider, ProviderFormat


# --- parse_classification_response: три состояния ---------------------------


def test_verdict_offtopic_true():
    v = parse_classification_response(
        '```json\n{"offtopic": true, "confidence": 0.9, "reason": "болтает про кота"}\n```'
    )
    assert (v.offtopic, v.parse_failed) == (True, False)
    assert v.confidence == pytest.approx(0.9)
    assert v.reason == "болтает про кота"


def test_verdict_offtopic_false_is_a_real_verdict():
    v = parse_classification_response('{"offtopic": false, "confidence": 0.8}')
    assert v.offtopic is False
    assert v.parse_failed is False


@pytest.mark.parametrize(
    "raw",
    [
        "",  # пустой ответ (весь бюджет ушёл в thinking)
        "Сейчас подумаю над этим фрагментом…",  # обрыв до JSON
        "```json\n{\"offtopic\": tru\n```",  # обрыв внутри JSON
        '{"segments": []}',  # чужая схема
        "[1, 2, 3]",  # не объект
        '{"confidence": 0.9, "reason": "…"}',  # нет поля offtopic
    ],
)
def test_unparseable_answer_is_not_a_verdict(raw):
    v = parse_classification_response(raw)
    assert v.parse_failed is True, raw
    # offtopic=False здесь — заполнитель поля, а не решение модели.
    assert v.offtopic is False


def test_string_false_is_not_coerced_to_true():
    # bool("false") в Python это True — поле разбирается явно.
    assert parse_classification_response('{"offtopic": "false"}').offtopic is False
    assert parse_classification_response('{"offtopic": "да"}').offtopic is True
    assert parse_classification_response('{"offtopic": "может быть"}').parse_failed


def test_confidence_is_clamped_and_tolerant():
    assert parse_classification_response(
        '{"offtopic": true, "confidence": 1.7}'
    ).confidence == pytest.approx(1.0)
    assert parse_classification_response(
        '{"offtopic": true, "confidence": "ага"}'
    ).confidence == pytest.approx(0.0)


# --- Completion: причина остановки доходит до вызывающего -------------------


def test_completion_truncated_and_empty():
    c = Completion(text="", stop_reason="max_tokens", thinking="x" * 900)
    assert c.truncated is True
    assert c.empty is True
    assert "max_tokens" in c.diagnostics()
    assert "900ch" in c.diagnostics()  # видно, куда ушёл бюджет

    ok = Completion(text='{"offtopic": false}', stop_reason="end_turn")
    assert (ok.truncated, ok.empty) == (False, False)


def test_completion_openai_length_counts_as_truncated():
    # openai-совместимые отдают finish_reason='length' вместо 'max_tokens'.
    assert Completion(text="полов", stop_reason="length").truncated is True


# --- classifier: окно без вердикта не становится «чистым» -------------------


class _FakeClient:
    """LLMClient-заглушка: отдаёт заранее заданные Completion по очереди."""

    def __init__(self, replies: list[Completion]):
        self.provider = AIProvider(
            id="fake", label="Fake", format=ProviderFormat.ANTHROPIC, models=["m-cheap"]
        )
        self._replies = list(replies)
        self.calls: list[int] = []

    def complete(self, prompt, *, model=None, max_tokens=0, **kw) -> Completion:
        self.calls.append(max_tokens)
        if self._replies:
            return self._replies.pop(0)
        return Completion(text="", stop_reason="max_tokens")


def _transcript(words: list[tuple[str, float]]) -> dict:
    return {
        "segments": [
            {
                "start": words[0][1],
                "end": words[-1][1] + 0.5,
                "words": [
                    {"word": w, "start": t, "end": t + 0.4} for w, t in words
                ],
            }
        ]
    }


_ONE_WINDOW = _transcript([("раз", 0.0), ("два", 1.0), ("три", 2.0), ("четыре", 3.0)])


def test_truncated_window_is_reported_not_silently_clean(caplog):
    """Обрыв → ретрай с большим бюджетом → окно в лог как без вердикта."""
    client = _FakeClient(
        [
            Completion(text="ТЕМА: разбор видео", stop_reason="end_turn"),  # topic
            Completion(text="", stop_reason="max_tokens", thinking="…"),  # обрыв
            Completion(text="", stop_reason="max_tokens", thinking="…"),  # и ретрай
        ]
    )
    with caplog.at_level(logging.WARNING, logger="twitch_cut.llm_offtopic.classifier"):
        result = classify_transcript(_ONE_WINDOW, client=client)

    assert result.segments == []  # оффтопа не нашли — но и не соврали, что чисто
    warnings = "\n".join(r.getMessage() for r in caplog.records)
    assert "без вердикта" in warnings
    # Ретрай был, и он был с БОЛЬШИМ бюджетом, чем первая попытка.
    topic_call, first, retry = client.calls
    assert retry > first


def test_verdict_after_retry_is_used():
    """Ретрай с увеличенным бюджетом даёт вердикт — окно классифицировано."""
    client = _FakeClient(
        [
            Completion(text="ТЕМА: разбор видео", stop_reason="end_turn"),
            Completion(text="", stop_reason="max_tokens"),  # первая попытка обрыв
            Completion(
                text='{"offtopic": true, "confidence": 0.9, "reason": "про свой телефон"}',
                stop_reason="end_turn",
            ),
        ]
    )
    result = classify_transcript(_ONE_WINDOW, client=client)
    assert len(result.segments) == 1
    assert result.segments[0].reason == "про свой телефон"


def test_network_error_is_not_retried():
    """Сетевую ошибку не ретраим: причина не в бюджете, повтор удвоит прогон."""

    class _Boom(_FakeClient):
        def complete(self, prompt, *, model=None, max_tokens=0, **kw):
            self.calls.append(max_tokens)
            if len(self.calls) == 1:
                return Completion(text="ТЕМА: разбор", stop_reason="end_turn")
            raise ConnectionError("proxy died")

    client = _Boom([])
    result = classify_transcript(_ONE_WINDOW, client=client)
    assert result.segments == []
    assert len(client.calls) == 2  # тема + одна попытка классификации, без повтора
