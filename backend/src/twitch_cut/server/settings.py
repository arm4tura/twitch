"""Пользовательские настройки Desktop-приложения.

Хранение: один JSON-файл рядом с `projects.json` (тот же каталог XDG /
APPDATA / macOS Application Support). Схема плоская, все поля опциональные —
неизвестные ключи из будущих версий сохраняются at-write (round-trip
без потерь), незнакомые значения игнорируются в UI.

Почему не БД / не ini: файл 200 байт, редактируется раз в месяц, single-user.
JSON тривиально мигрируется руками если понадобится.

Пример файла:
```
{
  "default_model": "large-v3",
  "default_language": "ru",
  "default_device": "cuda",
  "default_compute_type": "float16",
  "default_batch_size": 16,
  "default_vad_method": "pyannote",
  "default_vad_filter": true
}
```

Никакой валидации значений не делаем — они пробрасываются в форму NewJob как
defaults, там уже WhisperX / ffmpeg разберутся что валидно, что нет.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


def _default_config_dir() -> Path:
    """Тот же каталог что у projects.json — не хочу дублировать логику XDG."""
    if os.name == "nt":
        base = Path(os.environ.get("APPDATA") or (Path.home() / "AppData" / "Roaming"))
    else:
        base = Path(os.environ.get("XDG_CONFIG_HOME") or (Path.home() / ".config"))
    return base / "twitch_cut"


def default_settings_file() -> Path:
    """Путь к settings.json. Env TWITCH_CUT_SETTINGS_FILE для тестов."""
    override = os.environ.get("TWITCH_CUT_SETTINGS_FILE")
    if override:
        return Path(override)
    return _default_config_dir() / "settings.json"


def logs_dir() -> Path:
    """Каталог логов. Сюда `serve` пишет backend.log (см.
    logging_setup.setup_server_logging), здесь же лежат projects.json и
    settings.json — UI-кнопка «Открыть папку логов» открывает именно его.

    Лог отдельной джобы лежит не здесь, а рядом с её результатами:
    `<workdir>/logs/<job>.log` (logging_setup.job_file_log).
    """
    return _default_config_dir()


# --- validation -------------------------------------------------------------

# Известные ключи и их типы. Неизвестные ключи НЕ отвергаем (round-trip для
# полей из будущих версий), но известные — проверяем, чтобы кривой клиент или
# посторонний POST на localhost не превратил settings.json в мусор, ломающий
# запуск. bool проверяем ДО int: в Python bool — подкласс int.
_KNOWN_STR_KEYS = frozenset({
    "default_transcriber",
    "default_device",
    "default_compute_type",
    "default_model",
    "default_language",
    "default_vad_method",
})
_MAX_KEYS = 100
_MAX_STR_LEN = 1000

# Ключи AI-конфига в settings.json. Хранят ТОЛЬКО конфиг провайдеров/шаблонов
# без секретов — сами ключи API живут в Electron safeStorage и сюда не пишутся.
_AI_LIST_KEYS = frozenset({"ai_providers", "ai_templates"})
_MAX_AI_ITEMS = 100

# Имена полей, которые НИКОГДА не должны попасть в settings.json (это секреты).
# Проверяем рекурсивно по всем provider/template-объектам: даже случайный POST
# на localhost не сможет персистнуть токен. `api_key_ref` — символическая
# ссылка, не секрет, поэтому в стоп-лист не входит (проверяем точное имя).
_FORBIDDEN_SECRET_KEYS = frozenset({"api_key", "apikey", "key", "token", "secret"})


class SettingsValidationError(ValueError):
    """Полезная нагрузка PUT /settings не прошла проверку."""


def _assert_no_secret_keys(obj: Any, *, path: str = "") -> None:
    """Рекурсивно убедиться, что в объекте нет полей-секретов.

    Кидает SettingsValidationError на первом же подозрительном ключе. Так
    settings.json физически не может стать местом хранения API-ключа.
    """
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(k, str) and k.strip().lower() in _FORBIDDEN_SECRET_KEYS:
                where = f"{path}.{k}" if path else k
                raise SettingsValidationError(
                    f"secret-bearing key '{where}' is not allowed in settings — "
                    "API keys live in the OS keychain (safeStorage), not settings.json"
                )
            _assert_no_secret_keys(v, path=f"{path}.{k}" if path else str(k))
    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            _assert_no_secret_keys(item, path=f"{path}[{i}]")


def _validate_ai_config(key: str, value: Any) -> None:
    """Проверить ai_providers / ai_templates: список объектов без секретов,
    каждый элемент — валидный AIProvider / PromptTemplate.
    """
    if not isinstance(value, list):
        raise SettingsValidationError(f"'{key}' must be a list")
    if len(value) > _MAX_AI_ITEMS:
        raise SettingsValidationError(f"'{key}' has too many items (>{_MAX_AI_ITEMS})")

    # 1) Секреты — под запретом (до любой другой валидации).
    _assert_no_secret_keys(value, path=key)

    # 2) Структурная валидация каждого элемента через pydantic-модели пакета
    #    llm_offtopic (чистый pydantic, без SDK — импорт безопасен и ленив).
    from ..llm_offtopic import AIProvider, PromptTemplate

    model = AIProvider if key == "ai_providers" else PromptTemplate
    for i, item in enumerate(value):
        if not isinstance(item, dict):
            raise SettingsValidationError(f"'{key}[{i}]' must be an object")
        try:
            model.model_validate(item)
        except Exception as exc:  # noqa: BLE001 - pydantic ValidationError → 422
            raise SettingsValidationError(f"'{key}[{i}]' invalid: {exc}") from exc


def validate_settings(payload: Any) -> dict[str, Any]:
    """Проверить тело PUT /settings. Возвращает тот же dict, если всё ок.

    Правила:
      - корень — объект (dict), не список/строка;
      - не больше _MAX_KEYS ключей (защита от разбухания файла);
      - известные строковые ключи — строки не длиннее _MAX_STR_LEN;
      - default_batch_size — целое 1..512;
      - default_vad_filter — bool;
      - rename_project_to_title — bool;
      - edit_journal — bool;
      - неизвестные ключи пропускаются as-is (forward-compat).

    Кидает SettingsValidationError с человекочитаемым сообщением.
    """
    if not isinstance(payload, dict):
        raise SettingsValidationError("settings must be a JSON object")
    if len(payload) > _MAX_KEYS:
        raise SettingsValidationError(f"too many keys (>{_MAX_KEYS})")

    for key in _KNOWN_STR_KEYS:
        if key in payload:
            val = payload[key]
            if not isinstance(val, str):
                raise SettingsValidationError(f"'{key}' must be a string")
            if len(val) > _MAX_STR_LEN:
                raise SettingsValidationError(f"'{key}' too long")

    if "default_vad_filter" in payload:
        if not isinstance(payload["default_vad_filter"], bool):
            raise SettingsValidationError("'default_vad_filter' must be a boolean")

    if "rename_project_to_title" in payload:
        if not isinstance(payload["rename_project_to_title"], bool):
            raise SettingsValidationError("'rename_project_to_title' must be a boolean")

    # Журнал правок (edit_journal.py): пишет <decisions>.edits.json рядом с
    # решениями. Тумблер, а не путь — файл всегда рядом с проектом.
    if "edit_journal" in payload:
        if not isinstance(payload["edit_journal"], bool):
            raise SettingsValidationError("'edit_journal' must be a boolean")

    if "default_batch_size" in payload:
        bs = payload["default_batch_size"]
        # bool — подкласс int, но батч true/false бессмысленен.
        if isinstance(bs, bool) or not isinstance(bs, int):
            raise SettingsValidationError("'default_batch_size' must be an integer")
        if not (1 <= bs <= 512):
            raise SettingsValidationError("'default_batch_size' out of range (1..512)")

    for ai_key in _AI_LIST_KEYS:
        if ai_key in payload:
            _validate_ai_config(ai_key, payload[ai_key])

    return payload


# --- I/O --------------------------------------------------------------------


def load_settings(*, store_path: Optional[Path] = None) -> dict[str, Any]:
    """Прочитать settings.json. Битый / отсутствующий файл → пустой dict.

    Возвращаем СЫРОЙ dict, а не объект — форма фронта сама решает какие поля
    подставлять как defaults. Это упрощает добавление новых полей: backend
    не нужно править.
    """
    store = store_path or default_settings_file()
    if not store.exists():
        return {}
    try:
        data = json.loads(store.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        logger.warning("settings file corrupt (%s), returning empty", exc)
        return {}
    if not isinstance(data, dict):
        return {}
    return data


def save_settings(
    settings: dict[str, Any], *, store_path: Optional[Path] = None
) -> None:
    """Атомарно записать settings.json (write → tmp → replace)."""
    store = store_path or default_settings_file()
    store.parent.mkdir(parents=True, exist_ok=True)
    tmp = store.with_suffix(store.suffix + ".tmp")
    tmp.write_text(
        json.dumps(settings, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    tmp.replace(store)
