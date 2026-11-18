"""Унифицированный LLM-клиент: один интерфейс для anthropic и openai-совместимых.

Клиент stateless по секретам: api_key передаётся в конструктор на время
прогона джобы и НИКОГДА не пишется на диск/в лог (ключи живут в Electron
safeStorage, приходят в backend по loopback — см. providers.py). Здесь ключ
только держится в памяти инстанса и уходит в SDK.

SDK (`anthropic`, `openai`) импортируются ЛЕНИВО, внутри методов. Причина:
модуль должен импортироваться даже когда optional-deps не поставлены (dev-venv
без gigaam-стека — см. память twitch-cut-dev-venv-gigaam). Отсутствие пакета →
понятная ошибка в момент вызова, а не ImportError на импорте llm_offtopic.

Публичный контракт:
- complete(prompt, ...) -> Completion  — запрос: текст + ПРИЧИНА остановки.
- test_connection() -> (ok, detail)    — проверка ключа/endpoint для UI.
- list_models() -> list[str]           — доступные модели (для выпадашки).

ПОЧЕМУ Completion, а не str. Модели с extended thinking расходуют СКРЫТЫЕ
токены из того же max_tokens: ответ может закончиться на
`stop_reason="max_tokens"`, не начав text-блок. Вызывающий обязан отличать
«модель ответила: не оффтоп» от «бюджета не хватило» — иначе обрезание молча
превращается в честный вердикт (замер на реальном VOD: 13 из 30 окон
обрезаны, и все зачлись как «по теме»). Раньше здесь склеивались только
text-блоки, а stop_reason и thinking выбрасывались — причина отказа не доходила
до вызывающего вообще.

ПОЧЕМУ streaming по умолчанию. На длинных ответах (батч реплик, план
исходника) не-стримовый запрос рвётся: у agentrouter его nginx буферизует ответ
целиком и отдаёт 504 Gateway Time-out, не дождавшись генерации (замер: батч на
8192 токена — 6 минут и 504). Стрим держит соединение живым, потому что байты
текут непрерывно, и отдаёт stop_reason/usage тем же путём. stream=False
оставлен для endpoint'ов без SSE.

ПОЧЕМУ свой разбор стрима, а не messages.stream(). Аккумулятор SDK строгий: он
ждёт валидные события Anthropic. Агрегаторы (agentrouter) вперемешку с ними
присылают `delta: null`, `content_block: null` и служебные `billing.summary` —
на них аккумулятор копит 0 текста или спотыкается, и ответ теряется целиком
(замер: 8192 сгенерированных токена → text=0ch). Поэтому события читаются
здесь вручную, через getattr с дефолтом: чего нет — пропускаем, что есть —
копим. См. _completion_from_anthropic_stream.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from .providers import AIProvider, AuthScheme, ProviderFormat

# Дефолты запроса. max_tokens с запасом: у моделей с thinking скрытые токены
# едят ТОТ ЖЕ бюджет, и «короткого ответа» (JSON-вердикт) недостаточно, чтобы
# бюджет был скромным. temperature низкая — классификация должна быть
# стабильной, не творческой.
DEFAULT_MAX_TOKENS = 2048
DEFAULT_TEMPERATURE = 0.0
# 180с, а не 60: чанк транскрипта на 15 минут в минуту не укладывается —
# особенно через прокси и с thinking. Таймаут должен рвать зависшее
# соединение, а не нормальный длинный запрос.
DEFAULT_TIMEOUT_S = 180.0

# Дефолтная модель anthropic-формата, если провайдер не задал (см. claude-api
# skill — актуальный id на 2026-08). Путь по исходнику держит в контексте план
# видео + батч реплик и сам расставляет границы — на этой задаче Opus 5 заметно
# сильнее 4.8, поэтому дефолт здесь и в providers.py именно он.
DEFAULT_ANTHROPIC_MODEL = "claude-opus-5"

# Причины остановки, означающие «ответ оборван на середине». anthropic отдаёт
# 'max_tokens', openai-совместимые — finish_reason='length'.
TRUNCATED_STOP_REASONS = frozenset({"max_tokens", "length"})


class LLMError(RuntimeError):
    """Ошибка обращения к LLM (сеть, авторизация, отсутствующий SDK)."""


@dataclass(frozen=True)
class Completion:
    """Ответ модели вместе с причиной остановки и расходом бюджета.

    `text` может быть ПУСТЫМ при валидном ответе — если весь max_tokens ушёл в
    thinking. Тогда `truncated` истинно, и это НЕ вердикт «нет», а нехватка
    бюджета: вызывающий обязан либо повторить с большим max_tokens, либо
    записать окно как неклассифицированное.
    """

    text: str
    stop_reason: Optional[str] = None
    # Текст thinking-блоков (если модель их вернула). Нужен не для показа, а
    # для диагностики: len(thinking) объясняет, куда ушёл бюджет.
    thinking: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    # Модель, которая реально ответила (прокси может подменить).
    model: Optional[str] = None

    @property
    def truncated(self) -> bool:
        """Ответ оборван по лимиту токенов."""
        return self.stop_reason in TRUNCATED_STOP_REASONS

    @property
    def empty(self) -> bool:
        """Текста нет (мог быть съеден thinking или оборван на первом блоке)."""
        return not self.text.strip()

    def diagnostics(self) -> str:
        """Однострочная диагностика для лога: почему ответ такой, какой он есть."""
        return (
            f"stop_reason={self.stop_reason} text={len(self.text)}ch "
            f"thinking={len(self.thinking)}ch "
            f"tokens_in={self.input_tokens} tokens_out={self.output_tokens}"
        )


class LLMClient:
    """Обёртка над одним провайдером. Создаётся на джобу, живёт в памяти."""

    def __init__(
        self,
        provider: AIProvider,
        *,
        api_key: Optional[str] = None,
        timeout_s: float = DEFAULT_TIMEOUT_S,
    ):
        """:param api_key: секрет из safeStorage (расшифрован в Electron main).
        None допустим только для провайдеров без auth (локальная Ollama).
        """
        self.provider = provider
        self._api_key = api_key
        self._timeout_s = timeout_s

    # --- Публичный интерфейс ------------------------------------------------

    def complete(
        self,
        prompt: str,
        *,
        model: Optional[str] = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
        system: Optional[str] = None,
        stream: Optional[bool] = None,
    ) -> Completion:
        """Одиночный запрос → Completion. Синхронный (зовётся из to_thread).

        :param stream: держать SSE-соединение. None (дефолт) → берём предпочтение
            провайдера (`provider.stream`): у нормальных endpoint'ов стрим держит
            длинный ответ живым, у agentrouter'а стрим бракованный и провайдер
            просит не-стрим. Явные True/False перекрывают предпочтение.
        """
        model = model or self._default_model()
        if stream is None:
            stream = self.provider.stream
        if self.provider.format is ProviderFormat.ANTHROPIC:
            return self._complete_anthropic(
                prompt, model, max_tokens, temperature, system, stream
            )
        return self._complete_openai(
            prompt, model, max_tokens, temperature, system, stream
        )

    def test_connection(self) -> tuple[bool, str]:
        """Лёгкая проверка для UI: (ok, человекочитаемая деталь).

        Не роняем исключение — возвращаем (False, reason), чтобы UI показал
        причину, а не стектрейс.
        """
        try:
            reply = self.complete("ping", max_tokens=64)
        except LLMError as e:
            return False, str(e)
        except Exception as e:  # noqa: BLE001 - в UI нужен любой текст ошибки
            return False, f"{type(e).__name__}: {e}"
        if not reply.empty:
            return True, "ok"
        if reply.truncated:
            # Ключ и endpoint рабочие, текста нет потому что бюджет ушёл в
            # thinking. Для проверки соединения это успех, а не отказ.
            return True, f"ok (текст пуст: {reply.diagnostics()})"
        return False, f"empty response ({reply.diagnostics()})"

    def list_models(self) -> list[str]:
        """Список моделей. Anthropic/OpenAI — через /models; при ошибке —
        падаем на статический список провайдера (лучше, чем пусто)."""
        try:
            if self.provider.format is ProviderFormat.ANTHROPIC:
                return self._list_models_anthropic()
            return self._list_models_openai()
        except Exception:  # noqa: BLE001 - фолбэк на статику всегда безопасен
            return list(self.provider.models)

    # --- anthropic формат ---------------------------------------------------

    def _anthropic_client(self):
        try:
            import anthropic  # noqa: PLC0415 - ленивый импорт (optional dep)
        except ImportError as e:
            raise LLMError(
                "anthropic SDK не установлен — `pip install anthropic`"
            ) from e
        kwargs: dict = {"timeout": self._timeout_s}
        if self._api_key:
            # Один и тот же секрет, разный заголовок: BEARER → auth_token
            # (Authorization: Bearer, нужен agentrouter'у), иначе → api_key
            # (x-api-key, нативный Anthropic). Ключ уходит в SDK, не в headers.
            if self.provider.auth_scheme is AuthScheme.BEARER:
                kwargs["auth_token"] = self._api_key
            else:
                kwargs["api_key"] = self._api_key
        if self.provider.base_url:
            kwargs["base_url"] = self.provider.base_url
        if self.provider.headers:
            kwargs["default_headers"] = dict(self.provider.headers)
        return anthropic.Anthropic(**kwargs)

    def _complete_anthropic(
        self,
        prompt: str,
        model: str,
        max_tokens: int,
        temperature: float,
        system: Optional[str],
        stream: bool,
    ) -> Completion:
        client = self._anthropic_client()
        req: dict = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            req["system"] = system
        try:
            if stream:
                # НЕ messages.stream(): его аккумулятор не переваривает
                # null-дельты агрегаторов (см. модульный docstring). Берём
                # сырые события и копим сами.
                events = client.messages.create(stream=True, **req)
                return _completion_from_anthropic_stream(events, model)
            msg = client.messages.create(**req)
        except LLMError:
            raise  # уже доменная (обрыв стрима) — не заворачиваем второй раз
        except Exception as e:  # noqa: BLE001 - оборачиваем в доменную ошибку
            raise LLMError(f"anthropic request failed: {e}") from e
        return _completion_from_anthropic(msg, model)

    def _list_models_anthropic(self) -> list[str]:
        client = self._anthropic_client()
        page = client.models.list(limit=100)
        ids = [m.id for m in getattr(page, "data", [])]
        return ids or list(self.provider.models)

    # --- openai-совместимый формат ------------------------------------------

    def _openai_client(self):
        try:
            from openai import OpenAI  # noqa: PLC0415 - ленивый импорт
        except ImportError as e:
            raise LLMError("openai SDK не установлен — `pip install openai`") from e
        if not self.provider.base_url:
            raise LLMError("openai-формат требует base_url у провайдера")
        # Некоторые локальные endpoints (Ollama) не требуют ключа — SDK всё
        # равно хочет непустую строку, подставляем плейсхолдер.
        api_key = self._api_key or "not-needed"
        kwargs: dict[str, Any] = {
            "api_key": api_key,
            "base_url": self.provider.base_url,
            "timeout": self._timeout_s,
        }
        # Доп. заголовки провайдера (агрегаторы вроде agentrouter опознают клиента
        # по ним). Authorization/Content-Type сюда не попадают — их чистит валидатор.
        if self.provider.headers:
            kwargs["default_headers"] = dict(self.provider.headers)
        return OpenAI(**kwargs)

    def _complete_openai(
        self,
        prompt: str,
        model: str,
        max_tokens: int,
        temperature: float,
        system: Optional[str],
        stream: bool,
    ) -> Completion:
        client = self._openai_client()
        messages: list[dict] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        req: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": messages,
        }
        try:
            if stream:
                # stream_options={"include_usage": True} СОЗНАТЕЛЬНО не
                # передаём: локальные endpoints (Ollama, часть прокси) отвечают
                # на неизвестный параметр 400. Цена — usage=0 в стриме.
                chunks = client.chat.completions.create(stream=True, **req)
                return _completion_from_openai_stream(chunks, model)
            resp = client.chat.completions.create(**req)
        except Exception as e:  # noqa: BLE001
            raise LLMError(f"openai request failed: {e}") from e
        return _completion_from_openai(resp, model)

    def _list_models_openai(self) -> list[str]:
        client = self._openai_client()
        page = client.models.list()
        ids = [m.id for m in getattr(page, "data", [])]
        return ids or list(self.provider.models)

    # --- helpers ------------------------------------------------------------

    def _default_model(self) -> str:
        m = self.provider.default_model()
        if m:
            return m
        if self.provider.format is ProviderFormat.ANTHROPIC:
            return DEFAULT_ANTHROPIC_MODEL
        raise LLMError(
            f"provider {self.provider.id!r} has no models and no default"
        )


# --- разбор ответов SDK в Completion ----------------------------------------
#
# Всё через getattr с дефолтом: у прокси и у разных версий SDK поля приходят
# неполными, и падать на отсутствующем usage — худшее, что можно сделать в
# середине часового прогона.


def _completion_from_anthropic(msg: Any, model: str) -> Completion:
    """Message SDK anthropic → Completion (text + thinking + stop_reason)."""
    text_parts: list[str] = []
    think_parts: list[str] = []
    for block in getattr(msg, "content", None) or []:
        btype = getattr(block, "type", None)
        if btype == "text":
            text_parts.append(getattr(block, "text", "") or "")
        elif btype in ("thinking", "redacted_thinking"):
            # У redacted_thinking текста нет (только зашифрованный data) —
            # длина всё равно попадёт в диагностику как 0, это честно.
            think_parts.append(getattr(block, "thinking", "") or "")
    usage = getattr(msg, "usage", None)
    return Completion(
        text="".join(text_parts).strip(),
        stop_reason=getattr(msg, "stop_reason", None),
        thinking="".join(think_parts),
        input_tokens=_int_attr(usage, "input_tokens"),
        output_tokens=_int_attr(usage, "output_tokens"),
        model=getattr(msg, "model", None) or model,
    )


def _completion_from_anthropic_stream(events: Any, model: str) -> Completion:
    """Склеить SSE-события Anthropic вручную — терпимо к мусору агрегаторов.

    Аккумулятор SDK (`messages.stream()`) ждёт валидные события и на
    `delta: null` / `content_block: null` / служебных `billing.summary`
    (agentrouter присылает всё это вперемешку с нормальными) копит 0 текста.
    Здесь каждое событие читается через getattr с дефолтом: чего нет —
    пропускаем, что есть — копим. Никакое одно битое событие не может обнулить
    ответ.

    Обрыв посреди стрима НЕ выбрасывает уже накопленное: возвращаем частичный
    текст со `stop_reason="stream_interrupted"`. Дальше решает вызывающий
    (_ask_segments / _ask_verdict): дочитанный JSON разберётся, недочитанный
    провалит разбор и уйдёт в штатный повтор. Если же не накопилось ничего,
    спасать нечего — это ошибка запроса (LLMError).
    """
    text_parts: list[str] = []
    think_parts: list[str] = []
    stop_reason: Optional[str] = None
    input_tokens = 0
    output_tokens = 0
    actual_model: Optional[str] = None
    interrupted: Optional[BaseException] = None

    try:
        for event in events:
            # message_start: модель и input_tokens. message_stop — без полей.
            msg = getattr(event, "message", None)
            if msg is not None:
                actual_model = getattr(msg, "model", None) or actual_model
                stop_reason = getattr(msg, "stop_reason", None) or stop_reason
                mu = getattr(msg, "usage", None)
                input_tokens = _int_attr(mu, "input_tokens") or input_tokens
                output_tokens = _int_attr(mu, "output_tokens") or output_tokens
            # content_block_start: у блока уже может быть начальный текст.
            block = getattr(event, "content_block", None)
            if block is not None:
                text_parts.append(getattr(block, "text", None) or "")
                think_parts.append(getattr(block, "thinking", None) or "")
            # content_block_delta: text_delta/thinking_delta.
            # message_delta: здесь приходит финальный stop_reason.
            delta = getattr(event, "delta", None)
            if delta is not None:
                text_parts.append(getattr(delta, "text", None) or "")
                think_parts.append(getattr(delta, "thinking", None) or "")
                stop_reason = getattr(delta, "stop_reason", None) or stop_reason
            # usage на уровне события (message_delta) — итоговый расход.
            eu = getattr(event, "usage", None)
            if eu is not None:
                input_tokens = _int_attr(eu, "input_tokens") or input_tokens
                output_tokens = _int_attr(eu, "output_tokens") or output_tokens
    except Exception as e:  # noqa: BLE001 - обрыв разбираем ниже, а не роняем
        interrupted = e
    finally:
        # Закрыть сокет, если SDK дал такую возможность: при раннем выходе из
        # итерации соединение иначе висит до сборки мусора.
        close = getattr(events, "close", None)
        if callable(close):
            try:
                close()
            except Exception:  # noqa: BLE001 - закрытие best-effort
                pass

    text = "".join(text_parts).strip()
    if interrupted is not None:
        if not text:
            raise LLMError(f"anthropic stream failed: {interrupted}") from interrupted
        if stop_reason is None:
            stop_reason = "stream_interrupted"

    return Completion(
        text=text,
        stop_reason=stop_reason,
        thinking="".join(think_parts),
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        model=actual_model or model,
    )


def _completion_from_openai(resp: Any, model: str) -> Completion:
    """ChatCompletion → Completion. reasoning_content (DeepSeek/vLLM) → thinking."""
    choices = getattr(resp, "choices", None) or []
    if not choices:
        return Completion(text="", stop_reason="no_choices", model=model)
    choice = choices[0]
    message = getattr(choice, "message", None)
    usage = getattr(resp, "usage", None)
    return Completion(
        text=(getattr(message, "content", None) or "").strip(),
        stop_reason=getattr(choice, "finish_reason", None),
        thinking=getattr(message, "reasoning_content", None) or "",
        input_tokens=_int_attr(usage, "prompt_tokens"),
        output_tokens=_int_attr(usage, "completion_tokens"),
        model=getattr(resp, "model", None) or model,
    )


def _completion_from_openai_stream(chunks: Any, model: str) -> Completion:
    """Склеить стрим openai-чанков в Completion.

    finish_reason приходит в последнем чанке с непустым choices; usage —
    только если провайдер прислал его сам (мы его не запрашиваем).
    """
    text_parts: list[str] = []
    think_parts: list[str] = []
    stop_reason: Optional[str] = None
    usage: Any = None
    actual_model: Optional[str] = None
    try:
        for chunk in chunks:
            actual_model = getattr(chunk, "model", None) or actual_model
            usage = getattr(chunk, "usage", None) or usage
            for choice in getattr(chunk, "choices", None) or []:
                delta = getattr(choice, "delta", None)
                if delta is not None:
                    text_parts.append(getattr(delta, "content", None) or "")
                    think_parts.append(getattr(delta, "reasoning_content", None) or "")
                stop_reason = getattr(choice, "finish_reason", None) or stop_reason
    except Exception as e:  # noqa: BLE001 - обрыв посреди стрима = ошибка запроса
        raise LLMError(f"openai stream failed: {e}") from e
    return Completion(
        text="".join(text_parts).strip(),
        stop_reason=stop_reason,
        thinking="".join(think_parts),
        input_tokens=_int_attr(usage, "prompt_tokens"),
        output_tokens=_int_attr(usage, "completion_tokens"),
        model=actual_model or model,
    )


def _int_attr(obj: Any, name: str) -> int:
    """getattr → int, устойчиво к None/строкам/отсутствию объекта usage."""
    if obj is None:
        return 0
    try:
        return int(getattr(obj, name, 0) or 0)
    except (TypeError, ValueError):
        return 0
