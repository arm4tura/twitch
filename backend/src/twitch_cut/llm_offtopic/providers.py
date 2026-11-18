"""Модели AI-провайдеров: как ходить в LLM (формат API, base_url, модели).

Два формата запросов, которые мы поддерживаем:

- ``anthropic``       — нативный Anthropic Messages API (SDK `anthropic`).
- ``openai``          — любой OpenAI-совместимый `/v1/chat/completions`
                        (OpenRouter, Ollama, LM Studio, vLLM, DeepSeek, Groq…).

Провайдер НЕ хранит API-ключ. Ключи живут в Electron safeStorage (шифруются
OS-keychain'ом), расшифровываются в main-процессе и приходят в backend в теле
запроса на loopback (127.0.0.1). Backend их использует, но НИКОГДА не пишет на
диск и не логирует. Поэтому здесь только `api_key_ref` — символическая ссылка
(«какой ключ спросить у Electron»), а не сам секрет.

BUILTIN_PROVIDERS — read-only пресеты (id с префиксом ``builtin:``). UI делает
из них редактируемые клоны (см. templates.py — та же схема built-in→clone).
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ProviderFormat(str, Enum):
    """Формат протокола запроса к LLM."""

    ANTHROPIC = "anthropic"
    OPENAI = "openai"


class AuthScheme(str, Enum):
    """Как класть секрет в запрос — важно только для anthropic-формата.

    Нативный Anthropic авторизуется заголовком ``x-api-key``; агрегаторы,
    прикидывающиеся Anthropic Messages API (agentrouter и часть прокси), ждут
    ``Authorization: Bearer``. SDK anthropic умеет оба: ``api_key=`` → x-api-key,
    ``auth_token=`` → Bearer (см. client.py). Для openai-формата поле не влияет —
    openai SDK всегда шлёт Bearer из api_key.
    """

    X_API_KEY = "x-api-key"
    BEARER = "bearer"


class AIProvider(BaseModel):
    """Описание одного AI-провайдера (endpoint + список моделей).

    Секрета здесь нет — только `api_key_ref` (имя записи в Electron safeStorage).
    """

    model_config = ConfigDict(extra="forbid")

    # Стабильный идентификатор. `builtin:*` — read-only пресет; пользовательские
    # клоны получают собственный uuid-подобный id на фронте.
    id: str = Field(min_length=1, max_length=120)
    # Человекочитаемое имя для UI ("OpenRouter", "Локальная Ollama").
    label: str = Field(min_length=1, max_length=120)
    format: ProviderFormat

    # Базовый URL API. Для anthropic можно None (SDK берёт дефолт
    # https://api.anthropic.com). Для openai — обязателен (некуда ходить иначе).
    base_url: Optional[str] = Field(default=None, max_length=500)

    # Список model id, которые этот провайдер отдаёт (для выпадашки в UI).
    # Может быть пустым — тогда UI покажет свободный ввод / позовёт list_models.
    models: list[str] = Field(default_factory=list, max_length=200)

    # Символическая ссылка на ключ в safeStorage ("builtin:anthropic" и т.п.).
    # None → провайдер ключа не требует (локальная Ollama без auth).
    api_key_ref: Optional[str] = Field(default=None, max_length=120)

    # Схема авторизации для anthropic-формата: x-api-key (нативный Anthropic) или
    # Bearer (agentrouter и прокси, прикидывающиеся Messages API). Секрет тот же
    # (api_key_ref), меняется только заголовок — SDK ставит его сам из ключа.
    # Для openai-формата не используется (там всегда Bearer).
    auth_scheme: AuthScheme = AuthScheme.X_API_KEY

    # Доп. HTTP-заголовки к КАЖДОМУ запросу (default_headers у SDK). Нужны
    # агрегаторам, опознающим клиента по заголовкам: HTTP-Referer/X-Title у
    # OpenRouter-совместимых, фингерпринт клиента у agentrouter. Секрета тут НЕТ —
    # Authorization ставит SDK из api_key, сам ключ сюда не пишем и не персистим.
    headers: dict[str, str] = Field(default_factory=dict, max_length=50)

    # Держать ли SSE-стрим. True почти всегда: пока байты текут, длинный ответ
    # не рвётся по таймауту шлюза. False — только для endpoint'ов без SSE; на
    # agentrouter не-стрим даёт 504 (его nginx буферизует ответ целиком и не
    # дожидается генерации: замер — батч на 8192 токена, 6 минут и 504).
    # Битые SSE-события агрегаторов лечит НЕ этот флаг, а терпимый разбор
    # стрима в client.py (_completion_from_anthropic_stream).
    stream: bool = True

    # True → пресет из BUILTIN_PROVIDERS, UI запрещает правку (только клон).
    builtin: bool = False

    @field_validator("base_url")
    @classmethod
    def _base_url_shape(cls, v: Optional[str], info) -> Optional[str]:
        if v is None:
            return v
        v = v.strip()
        if not v:
            return None
        if not (v.startswith("http://") or v.startswith("https://")):
            raise ValueError("base_url must start with http:// or https://")
        # Нормализуем: без хвостового слэша (клиент сам добавит путь).
        return v.rstrip("/")

    @field_validator("models")
    @classmethod
    def _models_nonempty_strings(cls, v: list[str]) -> list[str]:
        cleaned = [m.strip() for m in v if m and m.strip()]
        return cleaned

    @field_validator("headers")
    @classmethod
    def _headers_clean(cls, v: dict[str, str]) -> dict[str, str]:
        # Отбрасываем пустые и те, что SDK ставит сам: Authorization берётся из
        # api_key, Content-Type проставляется на JSON-запрос. Дать им затереться
        # своим значением = сломать auth/сериализацию.
        managed = {"authorization", "content-type"}
        out: dict[str, str] = {}
        for k, val in (v or {}).items():
            key = str(k).strip()
            sval = str(val).strip()
            if not key or not sval or key.lower() in managed:
                continue
            out[key] = sval
        return out

    def default_model(self) -> Optional[str]:
        """Первая модель из списка — дефолт для UI. None если список пуст."""
        return self.models[0] if self.models else None


# --- Встроенные пресеты -----------------------------------------------------
#
# Anthropic-модели — актуальные id на 2026-08 (см. claude-api skill). Не
# используем старые `claude-3-5-*` — они устарели. Дефолт (первый в списке) —
# Opus 5: путь по исходнику держит в контексте план видео и батч реплик и сам
# расставляет границы, и на этой задаче 4.8 заметно слабее. Заодно оживает
# каскад: cheap=Sonnet 5 → expensive=Opus 5 (раньше обе ступени были 4.8).
#
# Модели у openai-совместимых провайдеров меняются часто — держим короткий
# список ходовых, остальное UI дотянет через list_models().

BUILTIN_PROVIDERS: list[AIProvider] = [
    AIProvider(
        id="builtin:anthropic",
        label="Anthropic (Claude)",
        format=ProviderFormat.ANTHROPIC,
        base_url=None,
        models=[
            "claude-opus-5",
            "claude-opus-4-8",
            "claude-sonnet-5",
            "claude-haiku-4-5",
        ],
        api_key_ref="builtin:anthropic",
        builtin=True,
    ),
    AIProvider(
        id="builtin:openai",
        label="OpenAI",
        format=ProviderFormat.OPENAI,
        base_url="https://api.openai.com/v1",
        models=["gpt-4o", "gpt-4o-mini"],
        api_key_ref="builtin:openai",
        builtin=True,
    ),
    AIProvider(
        id="builtin:openrouter",
        label="OpenRouter",
        format=ProviderFormat.OPENAI,
        base_url="https://openrouter.ai/api/v1",
        models=[
            "anthropic/claude-opus-4.8",
            "openai/gpt-4o-mini",
            "deepseek/deepseek-chat",
        ],
        api_key_ref="builtin:openrouter",
        builtin=True,
    ),
    AIProvider(
        id="builtin:ollama",
        label="Ollama (локально)",
        format=ProviderFormat.OPENAI,
        base_url="http://127.0.0.1:11434/v1",
        models=["claude-opus-4-8", "llama3.1", "qwen2.5"],
        api_key_ref=None,  # локальная Ollama без auth
        builtin=True,
    ),
    # AgentRouter — агрегатор, говорящий по нативному Anthropic Messages API
    # (тот же путь /v1/messages, который SDK anthropic добавляет к base_url).
    # Авторизация — Authorization: Bearer (НЕ x-api-key), поэтому auth_scheme=BEARER:
    # SDK положит ключ в Bearer через auth_token. Сервис опознаёт клиента по набору
    # заголовков Roo Code — шлём их через default_headers (они перекрывают дефолтные
    # User-Agent/X-Stainless самого SDK). Секрета в заголовках нет — Authorization
    # ставит SDK из ключа, сам ключ сюда не пишем и не персистим.
    # Стрим оставляем включённым (stream=True по умолчанию): не-стримовый запрос
    # его nginx буферизует и рвёт по 504. Его SSE присылает `delta:null`,
    # `content_block:null` и служебные `billing.summary` — это переваривает
    # терпимый разбор стрима в client.py, а не флаг stream.
    # Модели: первая в списке — дефолт (default_model), поэтому Opus 5 впереди —
    # тот же выбор, что у нативного Anthropic-пресета. `gpt-5.6-sol` не опечатка:
    # агрегатор отдаёт и не-Anthropic модели по тому же Messages API, формат
    # запроса от id модели не зависит.
    AIProvider(
        id="builtin:agentrouter",
        label="AgentRouter",
        format=ProviderFormat.ANTHROPIC,
        base_url="https://agentrouter.org",
        models=["claude-opus-5", "claude-opus-4-8", "gpt-5.6-sol"],
        api_key_ref="builtin:agentrouter",
        auth_scheme=AuthScheme.BEARER,
        headers={
            "User-Agent": "RooCode/3.34.8",
            "X-Title": "Roo Code",
            "HTTP-Referer": "https://github.com/RooVetGit/Roo-Cline",
            "X-Stainless-Runtime-Version": "v22.20.0",
            "X-Stainless-Runtime": "node",
            "X-Stainless-Arch": "x64",
            "X-Stainless-OS": "Linux",
            "X-Stainless-Lang": "js",
        },
        builtin=True,
    ),
]


def builtin_provider(provider_id: str) -> Optional[AIProvider]:
    """Найти встроенный пресет по id. None если не встроенный."""
    for p in BUILTIN_PROVIDERS:
        if p.id == provider_id:
            return p
    return None
