"""Тесты реестра недавних проектов (server/projects.py).

Изолируем от системного XDG_CONFIG_HOME/APPDATA через env TWITCH_CUT_PROJECTS_FILE.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from twitch_cut.server import projects as pj


@pytest.fixture()
def registry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    store = tmp_path / "projects.json"
    monkeypatch.setenv("TWITCH_CUT_PROJECTS_FILE", str(store))
    return store


def _make_decisions(path: Path, *, mutes=3, cuts=1, highlights=2, range_in_ms=0, range_out_ms=60000) -> None:
    path.write_text(
        json.dumps(
            {
                "range_in_ms": range_in_ms,
                "range_out_ms": range_out_ms,
                "mutes": [{"start_ms": i, "end_ms": i + 10} for i in range(mutes)],
                "cuts": [{"start_ms": i, "end_ms": i + 10} for i in range(cuts)],
                "highlights": {"highlights": [{"start_s": 0, "end_s": 1, "title": f"h{i}"} for i in range(highlights)]},
                "caches": {"audio": str(path.parent / "cache" / "audio.wav")},
            }
        ),
        encoding="utf-8",
    )


def test_register_creates_file(registry: Path, tmp_path: Path) -> None:
    d = tmp_path / "d.json"
    _make_decisions(d)
    pj.register_project(d)
    assert registry.exists()
    data = json.loads(registry.read_text(encoding="utf-8"))
    assert data["paths"] == [str(d.resolve())]


def test_register_dedup_and_mru(registry: Path, tmp_path: Path) -> None:
    a = tmp_path / "a.json"; _make_decisions(a)
    b = tmp_path / "b.json"; _make_decisions(b)
    pj.register_project(a)
    pj.register_project(b)
    pj.register_project(a)  # снова a — должно всплыть в топ
    data = json.loads(registry.read_text(encoding="utf-8"))
    assert data["paths"] == [str(a.resolve()), str(b.resolve())]


def test_register_caps_length(registry: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pj, "MAX_REGISTRY_ENTRIES", 3)
    for i in range(5):
        p = tmp_path / f"d{i}.json"
        _make_decisions(p)
        pj.register_project(p)
    data = json.loads(registry.read_text(encoding="utf-8"))
    assert len(data["paths"]) == 3
    # Последние 3 (mru порядок — d4, d3, d2).
    expected = [str((tmp_path / f"d{i}.json").resolve()) for i in (4, 3, 2)]
    assert data["paths"] == expected


def test_list_projects_skips_missing(registry: Path, tmp_path: Path) -> None:
    real = tmp_path / "real.json"; _make_decisions(real)
    ghost = tmp_path / "ghost.json"
    # Прямая запись в реестр: два пути, один битый.
    registry.write_text(json.dumps({"paths": [str(ghost), str(real)]}), encoding="utf-8")
    out = pj.list_projects()
    assert len(out) == 1
    assert out[0]["decisions_path"] == str(real)
    assert out[0]["mutes_count"] == 3
    assert out[0]["cuts_count"] == 1
    assert out[0]["highlights_count"] == 2
    assert out[0]["duration_ms"] == 60000


def test_duration_ms_from_schema_1_1_timecodes(registry: Path, tmp_path: Path) -> None:
    """Регрессия: реальный decisions.json (schema 1.1) хранит границы как
    таймкоды-строки (range_in/range_out/local_duration), а НЕ как числовые
    range_*_ms. Раньше _project_meta читал только *_ms → duration_ms всегда
    был None для настоящих проектов в Dashboard."""
    real = tmp_path / "real.json"
    real.write_text(
        json.dumps(
            {
                "schema_version": "1.1",
                "source": str(tmp_path / "stream.mp4"),
                "range_in": "00:00:10.000",
                "range_out": "00:05:10.000",
                "local_duration": "00:05:00.000",
                "mutes": [{"stream_start": 1.0, "stream_end": 1.5}],
                "cuts": [],
                "caches": {},
            }
        ),
        encoding="utf-8",
    )
    registry.write_text(json.dumps({"paths": [str(real)]}), encoding="utf-8")
    out = pj.list_projects()
    assert len(out) == 1
    assert out[0]["duration_ms"] == 300_000  # 5 минут


def test_list_projects_empty_when_no_file(registry: Path) -> None:
    assert not registry.exists()
    assert pj.list_projects() == []


def test_list_projects_ignores_corrupt_json(registry: Path, tmp_path: Path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text("not valid json {", encoding="utf-8")
    registry.write_text(json.dumps({"paths": [str(bad)]}), encoding="utf-8")
    assert pj.list_projects() == []


def test_prune_removes_missing(registry: Path, tmp_path: Path) -> None:
    real = tmp_path / "real.json"; _make_decisions(real)
    ghost = tmp_path / "ghost.json"
    registry.write_text(json.dumps({"paths": [str(ghost), str(real)]}), encoding="utf-8")
    removed = pj.prune_registry()
    assert removed == 1
    data = json.loads(registry.read_text(encoding="utf-8"))
    assert data["paths"] == [str(ghost.__class__(real))]


def test_list_projects_via_http(registry: Path, tmp_path: Path) -> None:
    from fastapi.testclient import TestClient
    from twitch_cut.server.app import create_app

    d = tmp_path / "d.json"; _make_decisions(d, mutes=7, cuts=2, highlights=1)
    pj.register_project(d)

    client = TestClient(create_app())
    r = client.get("/projects")
    assert r.status_code == 200
    body = r.json()
    assert len(body) == 1
    item = body[0]
    assert item["decisions_path"] == str(d.resolve())
    assert item["mutes_count"] == 7
    assert item["cuts_count"] == 2
    assert item["highlights_count"] == 1


def test_put_decisions_registers_project(registry: Path, tmp_path: Path, monkeypatch) -> None:
    from fastapi.testclient import TestClient
    from twitch_cut.server.app import create_app

    # PUT /decisions авторизует пути внутри DATA_ROOT; в тесте DATA_ROOT = tmp_path.
    monkeypatch.setattr("twitch_cut.config.DATA_ROOT", tmp_path)
    p = tmp_path / "edited.json"
    client = TestClient(create_app())
    r = client.put(
        f"/decisions?path={p}",
        json={"decisions": {"mutes": [], "cuts": [], "caches": {}}},
    )
    assert r.status_code == 200
    data = json.loads(registry.read_text(encoding="utf-8"))
    assert data["paths"] == [str(p.resolve())]


# --- deletion ----------------------------------------------------------------


def test_unregister_only_keeps_files(registry: Path, tmp_path: Path) -> None:
    """delete_files=False: запись уходит из реестра, файлы на диске остаются."""
    proj = tmp_path / "work" / "stream_20260101_1200"
    proj.mkdir(parents=True)
    d = proj / "decisions.json"
    _make_decisions(d)
    pj.register_project(d)

    res = pj.unregister_project(d, delete_files=False)
    assert res["unregistered"] is True
    assert res["deleted_dir"] is None
    assert res["error"] is None
    # Файл и папка на месте.
    assert d.exists()
    assert proj.exists()
    # Реестр пуст.
    assert pj.list_projects() == []


def test_unregister_with_files_removes_dir(registry: Path, tmp_path: Path) -> None:
    """delete_files=True: папка проекта (родитель decisions.json) удаляется."""
    proj = tmp_path / "work" / "stream_20260101_1200"
    (proj / "cache").mkdir(parents=True)
    d = proj / "decisions.json"
    _make_decisions(d)
    (proj / "cache" / "audio.wav").write_bytes(b"\x00\x01")
    pj.register_project(d)

    res = pj.unregister_project(d, delete_files=True)
    assert res["unregistered"] is True
    assert res["deleted_dir"] == str(proj)
    assert res["error"] is None
    assert not proj.exists()
    assert pj.list_projects() == []


def test_unregister_missing_is_idempotent(registry: Path, tmp_path: Path) -> None:
    """Удаление незарегистрированного пути не падает: unregistered=False."""
    d = tmp_path / "never_registered.json"
    _make_decisions(d)
    res = pj.unregister_project(d, delete_files=False)
    assert res["unregistered"] is False
    assert res["deleted_dir"] is None
    # Файл не трогали (delete_files=False).
    assert d.exists()


def test_unregister_refuses_home_dir(registry: Path, tmp_path: Path, monkeypatch) -> None:
    """Защита: не rmtree'им домашний каталог, даже если decisions.json лежит
    прямо в нём. Реестр всё равно чистим, но deleted_dir=None + error."""
    # Прикидываем home = tmp_path, decisions.json лежит прямо в home.
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    d = tmp_path / "decisions.json"
    _make_decisions(d)
    pj.register_project(d)

    res = pj.unregister_project(d, delete_files=True)
    assert res["unregistered"] is True
    assert res["deleted_dir"] is None
    assert res["error"] is not None
    # Домашний каталог и файл целы.
    assert d.exists()


def test_delete_project_via_http(registry: Path, tmp_path: Path) -> None:
    from fastapi.testclient import TestClient
    from twitch_cut.server.app import create_app

    proj = tmp_path / "work" / "stream_20260101_1200"
    proj.mkdir(parents=True)
    d = proj / "decisions.json"
    _make_decisions(d)
    pj.register_project(d)

    client = TestClient(create_app())
    r = client.delete(f"/projects?path={d}&delete_files=true")
    assert r.status_code == 200
    body = r.json()
    assert body["unregistered"] is True
    assert body["deleted_dir"] == str(proj)
    assert body["error"] is None
    assert not proj.exists()
    # Список после удаления пуст.
    assert client.get("/projects").json() == []


def test_delete_project_via_http_unregister_only(registry: Path, tmp_path: Path) -> None:
    from fastapi.testclient import TestClient
    from twitch_cut.server.app import create_app

    proj = tmp_path / "work" / "stream_20260101_1200"
    proj.mkdir(parents=True)
    d = proj / "decisions.json"
    _make_decisions(d)
    pj.register_project(d)

    client = TestClient(create_app())
    # delete_files по умолчанию False — не передаём флаг.
    r = client.delete(f"/projects?path={d}")
    assert r.status_code == 200
    body = r.json()
    assert body["unregistered"] is True
    assert body["deleted_dir"] is None
    assert d.exists()
    assert proj.exists()
    assert client.get("/projects").json() == []
