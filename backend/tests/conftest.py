"""Общие фикстуры для тестов backend.

Главное здесь — автоматическая изоляция пользовательского состояния. Реестр
недавних проектов (`projects.json`) и настройки (`settings.json`) по умолчанию
живут в `%APPDATA%\\twitch_cut\\` (или `~/.config/twitch_cut/`). Некоторые
эндпоинты (например `PUT /decisions` → `register_project`) пишут туда как
побочный эффект. Без изоляции прогон тестов засорял бы РЕАЛЬНЫЙ список проектов
пользователя путями вида `.../pytest-of-.../decisions.json`.

Фикстура ниже (autouse) на время КАЖДОГО теста перенаправляет оба файла во
временный каталог через env-override, которые уже поддерживают
`server/projects.py` и `server/settings.py`. После теста переменные окружения
восстанавливаются.
"""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def isolate_user_state(tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch):
    """Увести projects.json / settings.json во временный каталог на время теста."""
    state_dir: Path = tmp_path_factory.mktemp("twitch_cut_state")
    monkeypatch.setenv("TWITCH_CUT_PROJECTS_FILE", str(state_dir / "projects.json"))
    monkeypatch.setenv("TWITCH_CUT_SETTINGS_FILE", str(state_dir / "settings.json"))
    yield
    # monkeypatch сам откатит env после теста.
