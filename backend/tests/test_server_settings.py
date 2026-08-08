"""Тесты валидации PUT /settings.

Известные ключи проверяются по типу/диапазону; неизвестные сохраняются as-is
(round-trip для полей из будущих версий). Изоляция settings.json — через
autouse-фикстуру в conftest (env TWITCH_CUT_SETTINGS_FILE → tmp).
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from twitch_cut.server.app import create_app
from twitch_cut.server.settings import (
    SettingsValidationError,
    validate_settings,
)

import pytest


# --- unit: validate_settings ------------------------------------------------


def test_validate_accepts_full_valid_payload():
    payload = {
        "default_transcriber": "gigaam",
        "default_device": "cuda",
        "default_compute_type": "float16",
        "default_model": "large-v3",
        "default_language": "ru",
        "default_batch_size": 16,
        "default_vad_method": "pyannote",
        "default_vad_filter": True,
    }
    assert validate_settings(payload) == payload


def test_validate_preserves_unknown_keys():
    """Неизвестные ключи не отвергаются — forward-compat round-trip."""
    payload = {"future_flag": {"nested": 1}, "default_language": "en"}
    assert validate_settings(payload) == payload


def test_validate_rejects_non_object():
    for bad in ([], "x", 5, None):
        with pytest.raises(SettingsValidationError):
            validate_settings(bad)


def test_validate_rejects_wrong_type_known_key():
    with pytest.raises(SettingsValidationError):
        validate_settings({"default_language": 123})


def test_validate_rejects_batch_size_bool_and_range():
    with pytest.raises(SettingsValidationError):
        validate_settings({"default_batch_size": True})  # bool не int
    with pytest.raises(SettingsValidationError):
        validate_settings({"default_batch_size": 0})
    with pytest.raises(SettingsValidationError):
        validate_settings({"default_batch_size": 10_000})


def test_validate_rejects_vad_filter_non_bool():
    with pytest.raises(SettingsValidationError):
        validate_settings({"default_vad_filter": "yes"})


def test_validate_rejects_too_many_keys():
    with pytest.raises(SettingsValidationError):
        validate_settings({f"k{i}": i for i in range(101)})


# --- endpoint: PUT /settings ------------------------------------------------


def test_put_settings_ok_roundtrips_via_get():
    client = TestClient(create_app())
    body = {"default_language": "en", "default_batch_size": 8, "future_flag": 1}
    r = client.put("/settings", json=body)
    assert r.status_code == 200, r.text

    g = client.get("/settings").json()
    assert g["default_language"] == "en"
    assert g["default_batch_size"] == 8
    assert g["future_flag"] == 1  # неизвестный ключ сохранён


def test_put_settings_invalid_returns_422_and_does_not_write():
    client = TestClient(create_app())
    # сначала валидное значение
    assert client.put("/settings", json={"default_language": "ru"}).status_code == 200
    # затем кривой тип — должен быть отклонён, файл не перезаписан мусором
    r = client.put("/settings", json={"default_batch_size": "huge"})
    assert r.status_code == 422, r.text
    g = client.get("/settings").json()
    assert g.get("default_language") == "ru"
    assert "default_batch_size" not in g or g["default_batch_size"] != "huge"
