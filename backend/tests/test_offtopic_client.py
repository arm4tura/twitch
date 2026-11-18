"""Фаза 1: LLMClient отдаёт причину остановки, а не только текст.

Здесь нет сети: SDK подменяется заглушками. Проверяем контракт разбора ответа
(stop_reason / thinking / usage), стриминг по умолчанию и то, что в
openai-совместимый запрос НЕ уезжает `stream_options` — на неизвестном параметре
локальные endpoints (Ollama, часть прокси) отвечают 400.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from twitch_cut.llm_offtopic.client import (
    Completion,
    LLMClient,
    LLMError,
    _completion_from_anthropic,
    _completion_from_anthropic_stream,
    _completion_from_openai,
    _completion_from_openai_stream,
)
from twitch_cut.llm_offtopic.providers import (
    AIProvider,
    AuthScheme,
    ProviderFormat,
    builtin_provider,
)

ANTHROPIC = AIProvider(
    id="anthropic-test",
    label="Anthropic",
    format=ProviderFormat.ANTHROPIC,
    models=["model-a"],
)
OPENAI = AIProvider(
    id="openai-test",
    label="OpenAI-compatible",
    format=ProviderFormat.OPENAI,
    base_url="http://127.0.0.1:11434/v1",
    models=["model-o"],
)


# --- разбор ответов SDK ------------------------------------------------------


def _anthropic_msg(*blocks, stop_reason="end_turn", model="model-a"):
    return SimpleNamespace(
        content=list(blocks),
        stop_reason=stop_reason,
        model=model,
        usage=SimpleNamespace(input_tokens=1234, output_tokens=56),
    )


def test_anthropic_thinking_is_kept_apart_from_text():
    msg = _anthropic_msg(
        SimpleNamespace(type="thinking", thinking="долго думаю "),
        SimpleNamespace(type="text", text=' {"offtopic": false} '),
    )
    c = _completion_from_anthropic(msg, "model-a")
    assert c.text == '{"offtopic": false}'
    assert c.thinking == "долго думаю "
    assert (c.input_tokens, c.output_tokens) == (1234, 56)
    assert c.empty is False


def test_anthropic_budget_eaten_by_thinking_is_truncated_not_empty_verdict():
    msg = _anthropic_msg(
        SimpleNamespace(type="thinking", thinking="x" * 4000),
        stop_reason="max_tokens",
    )
    c = _completion_from_anthropic(msg, "model-a")
    assert (c.empty, c.truncated) == (True, True)


def test_anthropic_redacted_thinking_does_not_crash():
    msg = _anthropic_msg(SimpleNamespace(type="redacted_thinking", data="opaque"))
    assert _completion_from_anthropic(msg, "model-a").thinking == ""


def test_anthropic_missing_usage_does_not_crash():
    msg = SimpleNamespace(content=[SimpleNamespace(type="text", text="ok")])
    c = _completion_from_anthropic(msg, "model-a")
    assert (c.text, c.input_tokens, c.model) == ("ok", 0, "model-a")


def _ev(type_: str, **fields):
    """SSE-событие anthropic-стрима как его отдаёт SDK (SimpleNamespace)."""
    return SimpleNamespace(type=type_, **fields)


def _anthropic_events(text: str = "ok", *, stop_reason: str = "end_turn"):
    """Нормальная последовательность событий: start → блок → дельта → stop."""
    return [
        _ev(
            "message_start",
            message=SimpleNamespace(
                model="model-a",
                stop_reason=None,
                usage=SimpleNamespace(input_tokens=1234, output_tokens=1),
            ),
        ),
        _ev("content_block_start", content_block=SimpleNamespace(type="text", text="")),
        _ev("content_block_delta", delta=SimpleNamespace(type="text_delta", text=text)),
        _ev("content_block_stop"),
        _ev(
            "message_delta",
            delta=SimpleNamespace(stop_reason=stop_reason),
            usage=SimpleNamespace(output_tokens=56),
        ),
        _ev("message_stop"),
    ]


def test_anthropic_stream_glues_text_with_stop_reason_and_usage():
    c = _completion_from_anthropic_stream(iter(_anthropic_events(' {"ok":1} ')), "model-a")
    assert c.text == '{"ok":1}'
    assert c.stop_reason == "end_turn"
    assert (c.input_tokens, c.output_tokens) == (1234, 56)
    assert c.model == "model-a"


def test_anthropic_stream_survives_null_deltas_and_service_events():
    """Регрессия agentrouter: null-события не должны обнулять ответ.

    Его SSE присылает `delta:null`/`content_block:null`/`message:null`,
    служебные `billing.summary` и просто `data: null` вперемешку с нормальными
    событиями. На них аккумулятор SDK копил 0 текста при 8192 израсходованных
    токенах — весь батч терялся.
    """
    events = [
        _ev("message_start", message=None),
        _ev("content_block_start", content_block=None),
        _ev("content_block_delta", delta=None),
        {"object": "billing.summary", "amount": 1},  # служебное событие агрегатора
        None,  # `data: null`
        _ev("content_block_delta", delta=SimpleNamespace(type="text_delta", text='{"seg')),
        _ev("content_block_delta", delta=SimpleNamespace(type="text_delta", text='ments":[]}')),
        _ev(
            "message_delta",
            delta=SimpleNamespace(stop_reason="end_turn"),
            usage=SimpleNamespace(output_tokens=8192),
        ),
    ]
    c = _completion_from_anthropic_stream(iter(events), "model-a")
    assert c.text == '{"segments":[]}'
    assert (c.stop_reason, c.output_tokens) == ("end_turn", 8192)


def test_anthropic_stream_keeps_thinking_apart_from_text():
    events = [
        _ev("content_block_delta", delta=SimpleNamespace(type="thinking_delta", thinking="думаю")),
        _ev("content_block_delta", delta=SimpleNamespace(type="text_delta", text="ответ")),
    ]
    c = _completion_from_anthropic_stream(iter(events), "model-a")
    assert (c.text, c.thinking) == ("ответ", "думаю")


def test_anthropic_stream_interrupted_keeps_partial_text():
    """Обрыв сокета не должен выбрасывать уже накопленное.

    agentrouter рубит соединение на длинных ответах (incomplete chunked read).
    Частичный текст отдаём с пометкой — разбирать/повторять решает вызывающий.
    """

    def _broken():
        yield _ev("content_block_delta", delta=SimpleNamespace(text='{"segments":'))
        raise ConnectionError("peer closed connection without sending complete message body")

    c = _completion_from_anthropic_stream(_broken(), "model-a")
    assert c.text == '{"segments":'
    assert c.stop_reason == "stream_interrupted"


def test_anthropic_stream_interrupted_with_nothing_is_llm_error():
    def _broken():
        raise ConnectionError("incomplete chunked read")
        yield  # pragma: no cover - делает функцию генератором

    with pytest.raises(LLMError, match="stream failed"):
        _completion_from_anthropic_stream(_broken(), "model-a")


def test_anthropic_stream_closes_socket_on_break():
    """Раннее завершение итерации закрывает соединение, а не бросает его."""
    closed: list[bool] = []

    class _Events:
        def __iter__(self):
            yield _ev("content_block_delta", delta=SimpleNamespace(text="частично"))
            raise ConnectionError("cut")

        def close(self):
            closed.append(True)

    assert _completion_from_anthropic_stream(_Events(), "model-a").text == "частично"
    assert closed == [True]


def _chunk(content=None, *, reasoning=None, finish=None, usage=None, model="model-o"):
    return SimpleNamespace(
        model=model,
        usage=usage,
        choices=[
            SimpleNamespace(
                delta=SimpleNamespace(content=content, reasoning_content=reasoning),
                finish_reason=finish,
            )
        ],
    )


def test_openai_stream_is_glued_with_finish_reason():
    chunks = [
        _chunk('{"offtopic":'),
        _chunk(" true}", reasoning="думаю"),
        _chunk(finish="length"),
    ]
    c = _completion_from_openai_stream(iter(chunks), "model-o")
    assert c.text == '{"offtopic": true}'
    assert c.thinking == "думаю"
    assert c.truncated is True  # finish_reason='length' — тот же обрыв


def test_openai_stream_break_becomes_llm_error():
    def _broken():
        yield _chunk("нача")
        raise ConnectionResetError("proxy died")

    with pytest.raises(LLMError, match="stream failed"):
        _completion_from_openai_stream(_broken(), "model-o")


def test_openai_non_stream_without_choices():
    c = _completion_from_openai(SimpleNamespace(choices=[]), "model-o")
    assert (c.text, c.stop_reason) == ("", "no_choices")


# --- заглушки SDK -----------------------------------------------------------


class _FakeAnthropic:
    """Заглушка SDK. Стрим-путь = create(stream=True) → поток событий.

    Не-стримовый — тот же create(), но без `stream`, и отдаёт цельный Message.
    Разделение важно: путь выбирается флагом провайдера, и подмена одного
    другим — та самая ошибка, из-за которой agentrouter отвечал 504.
    """

    def __init__(self, msg=None, events=None):
        self.calls: list[dict] = []
        outer = self

        class _Messages:
            def create(self, **kw):
                outer.calls.append(kw)
                return iter(events or []) if kw.get("stream") else msg

        self.messages = _Messages()

    @property
    def stream_calls(self) -> list[dict]:
        return [c for c in self.calls if c.get("stream")]

    @property
    def create_calls(self) -> list[dict]:
        return [c for c in self.calls if not c.get("stream")]


class _FakeOpenAI:
    def __init__(self, chunks):
        self.calls: list[dict] = []
        outer = self

        class _Completions:
            def create(self, **kw):
                outer.calls.append(kw)
                return iter(chunks) if kw.get("stream") else chunks[0]

        self.chat = SimpleNamespace(completions=_Completions())


# --- поведение complete() ---------------------------------------------------


def test_anthropic_complete_streams_by_default():
    fake = _FakeAnthropic(events=_anthropic_events("привет"))
    client = LLMClient(ANTHROPIC)
    client._anthropic_client = lambda: fake  # type: ignore[method-assign]

    reply = client.complete("ping", max_tokens=64)

    assert reply.text == "привет"
    assert len(fake.stream_calls) == 1
    assert not fake.create_calls  # не-стримовый путь не задействован
    assert fake.stream_calls[0]["max_tokens"] == 64
    assert fake.stream_calls[0]["model"] == "model-a"  # дефолт провайдера


def test_anthropic_complete_stream_false_uses_create():
    fake = _FakeAnthropic(msg=_anthropic_msg(SimpleNamespace(type="text", text="ok")))
    client = LLMClient(ANTHROPIC)
    client._anthropic_client = lambda: fake  # type: ignore[method-assign]

    assert client.complete("ping", stream=False).text == "ok"

    assert len(fake.create_calls) == 1
    assert not fake.stream_calls


def test_provider_stream_preference_applies_when_arg_omitted():
    """Транспорт выбирает провайдер: у endpoint'ов без SSE — цельный ответ."""
    no_sse = ANTHROPIC.model_copy(update={"stream": False})
    fake = _FakeAnthropic(msg=_anthropic_msg(SimpleNamespace(type="text", text="ok")))
    client = LLMClient(no_sse)
    client._anthropic_client = lambda: fake  # type: ignore[method-assign]

    assert client.complete("ping").text == "ok"

    assert len(fake.create_calls) == 1
    assert not fake.stream_calls


def test_agentrouter_preset_streams_over_bearer():
    """Контракт транспорта agentrouter: стрим + Bearer.

    Не-стрим его nginx буферизует и рвёт по 504 (замер: батч 8192 токена — 6
    минут и 504), а битые SSE-события лечит терпимый разбор в client.py.
    """
    p = builtin_provider("builtin:agentrouter")
    assert p is not None
    assert p.stream is True
    assert p.auth_scheme is AuthScheme.BEARER
    assert p.format is ProviderFormat.ANTHROPIC


def test_openai_stream_request_has_no_stream_options():
    fake = _FakeOpenAI([_chunk("ok", finish="stop")])
    client = LLMClient(OPENAI)
    client._openai_client = lambda: fake  # type: ignore[method-assign]

    assert client.complete("ping").text == "ok"
    (kwargs,) = fake.calls
    assert kwargs["stream"] is True
    assert "stream_options" not in kwargs  # иначе Ollama/прокси → 400


def test_sdk_exception_becomes_llm_error():
    class _Boom:
        messages = SimpleNamespace(
            create=lambda **kw: (_ for _ in ()).throw(RuntimeError("no route"))
        )

    client = LLMClient(ANTHROPIC)
    client._anthropic_client = lambda: _Boom()  # type: ignore[method-assign]

    with pytest.raises(LLMError, match="anthropic request failed"):
        client.complete("ping")


def test_stream_break_is_not_double_wrapped():
    """Обрыв стрима — уже доменная ошибка; второй раз её не заворачиваем."""

    def _cut():
        raise ConnectionError("incomplete chunked read")
        yield  # pragma: no cover - делает функцию генератором

    fake = _FakeAnthropic(events=_cut())
    client = LLMClient(ANTHROPIC)
    client._anthropic_client = lambda: fake  # type: ignore[method-assign]

    with pytest.raises(LLMError, match="anthropic stream failed") as e:
        client.complete("ping")
    assert "request failed" not in str(e.value)


# --- test_connection --------------------------------------------------------


def test_connection_ok_when_text_present(monkeypatch):
    client = LLMClient(ANTHROPIC)
    monkeypatch.setattr(client, "complete", lambda *a, **kw: Completion(text="pong"))
    assert client.test_connection() == (True, "ok")


def test_connection_ok_when_budget_went_to_thinking(monkeypatch):
    """Ключ и endpoint рабочие: текста нет из-за бюджета, а не из-за отказа."""
    client = LLMClient(ANTHROPIC)
    monkeypatch.setattr(
        client,
        "complete",
        lambda *a, **kw: Completion(text="", stop_reason="max_tokens", thinking="…"),
    )
    ok, detail = client.test_connection()
    assert ok is True
    assert "max_tokens" in detail


def test_connection_fails_on_empty_answer_without_reason(monkeypatch):
    client = LLMClient(ANTHROPIC)
    monkeypatch.setattr(
        client, "complete", lambda *a, **kw: Completion(text="", stop_reason="end_turn")
    )
    ok, detail = client.test_connection()
    assert ok is False
    assert "empty response" in detail


def test_connection_reports_error_text(monkeypatch):
    client = LLMClient(ANTHROPIC)

    def _boom(*a, **kw):
        raise LLMError("bad api key")

    monkeypatch.setattr(client, "complete", _boom)
    assert client.test_connection() == (False, "bad api key")
