"""E2E тесты FastAPI-приложения через TestClient.

Проверяем HTTP-контракт: health, CRUD джоб, чтение/запись decisions, mount статики.
Runner-логика тестируется отдельно в test_server_runner.py — здесь стабим её
через фейковый JobStore/coro, чтобы не тянуть whisperx.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from twitch_cut.server.app import create_app
from twitch_cut.server.state import JobKind, JobStatus, JobStore


def test_health_ok():
    client = TestClient(create_app())
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert body["jobs"] == 0


def test_get_job_404_when_missing():
    client = TestClient(create_app())
    r = client.get("/jobs/deadbeef")
    assert r.status_code == 404


def test_list_jobs_empty():
    client = TestClient(create_app())
    r = client.get("/jobs")
    assert r.status_code == 200
    assert r.json() == []


def test_cancel_missing_job_404():
    client = TestClient(create_app())
    r = client.delete("/jobs/nope")
    assert r.status_code == 404


def test_decisions_read_write_roundtrip(tmp_path: Path, monkeypatch):
    # /decisions авторизует пути внутри DATA_ROOT; в тесте DATA_ROOT = tmp_path.
    monkeypatch.setattr("twitch_cut.config.DATA_ROOT", tmp_path)
    client = TestClient(create_app())
    p = tmp_path / "decisions.json"
    payload = {"mutes": [{"start_ms": 0, "end_ms": 100}], "cuts": [], "caches": {}}

    r = client.put(f"/decisions?path={p}", json={"decisions": payload})
    assert r.status_code == 200
    assert r.json()["path"] == str(p)
    assert p.exists()

    r = client.get(f"/decisions?path={p}")
    assert r.status_code == 200
    assert r.json() == payload


def test_decisions_read_404_when_missing(tmp_path: Path):
    client = TestClient(create_app())
    r = client.get(f"/decisions?path={tmp_path / 'nope.json'}")
    assert r.status_code == 404


def test_decisions_put_rejects_extra_fields(tmp_path: Path):
    client = TestClient(create_app())
    p = tmp_path / "d.json"
    # DecisionsPayload has extra=forbid — top-level extra key must 422.
    r = client.put(f"/decisions?path={p}", json={"decisions": {}, "extra": 1})
    assert r.status_code == 422


def test_transcript_read(tmp_path: Path, monkeypatch):
    monkeypatch.setattr("twitch_cut.config.DATA_ROOT", tmp_path)
    client = TestClient(create_app())
    p = tmp_path / "t.json"
    p.write_text(json.dumps({"segments": []}), encoding="utf-8")
    r = client.get(f"/transcript?path={p}")
    assert r.status_code == 200
    assert r.json() == {"segments": []}


def test_static_dir_mounts_index(tmp_path: Path):
    static = tmp_path / "dist"
    static.mkdir()
    (static / "index.html").write_text("<html><body>hello</body></html>", encoding="utf-8")
    client = TestClient(create_app(static_dir=static))
    r = client.get("/")
    assert r.status_code == 200
    assert "hello" in r.text


def test_process_request_forbids_extra_fields():
    client = TestClient(create_app())
    r = client.post(
        "/jobs/process",
        json={
            "stream": "a.mp4", "original": "b.mp4", "banwords": "b.txt",
            "workdir": "w", "decisions": "d.json", "vegas": "v.cs",
            "not_a_field": 1,  # extra=forbid → 422
        },
    )
    assert r.status_code == 422


def test_highlights_import_request_validates_required():
    client = TestClient(create_app())
    r = client.post("/jobs/highlights-import", json={"decisions": "d.json"})
    # missing response + output → 422
    assert r.status_code == 422


def test_cancel_pending_job_marks_cancelled(tmp_path: Path):
    """Дёрнуть DELETE до старта таски — джоба должна стать cancelled.

    Симулируем ситуацию через JobStore напрямую: создаём job без attach_task.
    """
    store = JobStore()
    state = store.create(JobKind.PROCESS)
    ok = store.cancel(state.id)
    assert ok is True
    assert store.get(state.id).status == JobStatus.CANCELLED


def test_cancel_terminal_job_returns_false():
    store = JobStore()
    state = store.create(JobKind.PROCESS)
    state.status = JobStatus.DONE
    assert store.cancel(state.id) is False


def test_decisions_rejects_unauthorized_path(tmp_path: Path, monkeypatch):
    """GET/PUT /decisions и GET /transcript должны отклонять пути вне
    DATA_ROOT и вне зарегистрированных проектов (защита от чтения/записи
    произвольного файла через localhost)."""
    # DATA_ROOT указывает в отдельный каталог; secret лежит СНАРУЖИ него.
    data_root = tmp_path / "data"
    data_root.mkdir()
    monkeypatch.setattr("twitch_cut.config.DATA_ROOT", data_root)

    secret = tmp_path / "secret.json"
    secret.write_text('{"token": "leak"}', encoding="utf-8")

    client = TestClient(create_app())

    # чтение чужого файла — 403, а не 200 с содержимым
    r = client.get(f"/decisions?path={secret}")
    assert r.status_code == 403, r.text

    r = client.get(f"/transcript?path={secret}")
    assert r.status_code == 403, r.text

    # запись поверх чужого файла — 403, файл не тронут
    r = client.put(
        f"/decisions?path={secret}",
        json={"decisions": {"mutes": [], "cuts": [], "caches": {}}},
    )
    assert r.status_code == 403, r.text
    assert secret.read_text(encoding="utf-8") == '{"token": "leak"}'
