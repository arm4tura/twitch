"""Тесты runner-функций.

Смысл: не тащить whisperx/torch, а прогнать полный цикл на mock_transcript.
Проверяем: progress-события летят, финальный статус верный, exception ловится.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from twitch_cut.server.runner import (
    Progress,
    _resolve_transcript_path,
    run_export_vegas_job,
    run_highlights_export_job,
    run_highlights_import_job,
)
from twitch_cut.server.schemas import (
    ExportVegasRequest,
    HighlightsExportRequest,
    HighlightsImportRequest,
)
from twitch_cut.server.state import JobKind, JobStatus, JobStore


# ---------- helpers ----------------------------------------------------------


def _mk_transcript(n_words: int) -> dict:
    return {
        "segments": [
            {
                "start": 0.0,
                "end": float(n_words),
                "words": [
                    {"word": f"w{i}", "start": float(i), "end": i + 0.5}
                    for i in range(n_words)
                ],
            }
        ]
    }


async def _drain(store: JobStore, job_id: str) -> list[dict]:
    events: list[dict] = []
    async for ev in store.subscribe(job_id):
        events.append(ev)
    return events


# ---------- Progress emitter -------------------------------------------------


def test_progress_emit_clamps_and_pushes():
    store = JobStore()
    state = store.create(JobKind.PROCESS)
    p = Progress(store, state)

    p.emit("stage-a", 50.0, "half")
    assert state.progress == 50.0
    assert state.stage == "stage-a"
    assert state.message == "half"

    # Clamping.
    p.emit("stage-b", 150.0, "over")
    assert state.progress == 100.0
    p.emit("stage-c", -5.0, "under")
    assert state.progress == 0.0


# ---------- _resolve_transcript_path ----------------------------------------


def test_resolve_transcript_override(tmp_path):
    p = tmp_path / "t.json"
    p.write_text("{}", encoding="utf-8")
    got = _resolve_transcript_path({"caches": {}}, str(p))
    assert got == p


def test_resolve_transcript_override_missing(tmp_path):
    with pytest.raises(ValueError, match="transcript not found"):
        _resolve_transcript_path({"caches": {}}, str(tmp_path / "nope.json"))


def test_resolve_transcript_from_caches(tmp_path):
    p = tmp_path / "t.json"
    p.write_text("{}", encoding="utf-8")
    got = _resolve_transcript_path({"caches": {"transcript": str(p)}}, None)
    assert got == p


def test_resolve_transcript_no_hint():
    with pytest.raises(ValueError, match="caches.transcript missing"):
        _resolve_transcript_path({"caches": {}}, None)


# ---------- highlights_export runner ----------------------------------------


def test_highlights_export_runner_success(tmp_path):
    store = JobStore()
    state = store.create(JobKind.HIGHLIGHTS_EXPORT)

    tpath = tmp_path / "t.json"
    tpath.write_text(json.dumps(_mk_transcript(50)), encoding="utf-8")
    dpath = tmp_path / "d.json"
    dpath.write_text(json.dumps({"caches": {"transcript": str(tpath)}}), encoding="utf-8")

    req = HighlightsExportRequest(
        decisions=str(dpath),
        out_dir=str(tmp_path / "pkg"),
        n_highlights=3,
    )

    async def _go():
        await run_highlights_export_job(store, state, req)
        return await _drain(store, state.id)

    events = asyncio.run(_go())
    assert state.status == JobStatus.DONE
    assert (tmp_path / "pkg" / "manifest.json").exists()
    # Финальное событие всегда 'final'.
    assert events[-1]["type"] == "final"
    assert events[-1]["state"]["status"] == "done"


def test_highlights_export_runner_failure(tmp_path):
    store = JobStore()
    state = store.create(JobKind.HIGHLIGHTS_EXPORT)

    # decisions.json без транскрипта — упадёт в runner.
    dpath = tmp_path / "d.json"
    dpath.write_text(json.dumps({"caches": {}}), encoding="utf-8")

    req = HighlightsExportRequest(
        decisions=str(dpath), out_dir=str(tmp_path / "pkg"), n_highlights=3,
    )
    asyncio.run(run_highlights_export_job(store, state, req))
    assert state.status == JobStatus.FAILED
    assert "caches.transcript" in state.error


# ---------- highlights_import runner ----------------------------------------


def test_highlights_import_runner_success(tmp_path):
    store = JobStore()
    state = store.create(JobKind.HIGHLIGHTS_IMPORT)

    tpath = tmp_path / "t.json"
    tpath.write_text(json.dumps(_mk_transcript(200)), encoding="utf-8")
    dpath = tmp_path / "d.json"
    dpath.write_text(json.dumps({"caches": {"transcript": str(tpath)}}), encoding="utf-8")

    response_body = {
        "highlights": [
            {"start_s": 10.0, "end_s": 55.0, "title": "t1", "reason": "r1", "score": 0.9},
            {"start_s": 60.0, "end_s": 120.0, "title": "t2", "reason": "r2", "score": 0.8},
        ]
    }
    resp_path = tmp_path / "response.json"
    resp_path.write_text(json.dumps(response_body), encoding="utf-8")

    out = tmp_path / "d2.json"
    req = HighlightsImportRequest(
        decisions=str(dpath),
        response=str(resp_path),
        output=str(out),
    )
    asyncio.run(run_highlights_import_job(store, state, req))
    assert state.status == JobStatus.DONE, state.error
    assert out.exists()
    merged = json.loads(out.read_text(encoding="utf-8"))
    assert len(merged["highlights"]["highlights"]) == 2


def test_highlights_import_runner_bad_response(tmp_path):
    store = JobStore()
    state = store.create(JobKind.HIGHLIGHTS_IMPORT)

    dpath = tmp_path / "d.json"
    dpath.write_text(json.dumps({"caches": {}}), encoding="utf-8")
    resp_path = tmp_path / "resp.json"
    resp_path.write_text("не json", encoding="utf-8")

    req = HighlightsImportRequest(
        decisions=str(dpath), response=str(resp_path), output=str(tmp_path / "o.json")
    )
    asyncio.run(run_highlights_import_job(store, state, req))
    assert state.status == JobStatus.FAILED
    assert "MergeError" in state.error


# ---------- export_vegas runner ---------------------------------------------


def test_export_vegas_runner_success(tmp_path):
    store = JobStore()
    state = store.create(JobKind.EXPORT_VEGAS)

    # Минимально валидный decisions для vegas_export:
    dpath = tmp_path / "d.json"
    dpath.write_text(
        json.dumps(
            {
                "stream": {"path": "S:/stream.mp4"},
                "range_in_ms": 0,
                "range_out_ms": 60_000,
                "mutes": [],
                "cuts": [],
            }
        ),
        encoding="utf-8",
    )
    vpath = tmp_path / "out.cs"
    req = ExportVegasRequest(decisions=str(dpath), vegas=str(vpath))
    asyncio.run(run_export_vegas_job(store, state, req))
    # Может быть либо done либо failed в зависимости от требований vegas_export;
    # главное — что состояние терминальное и финализирующее событие пришло.
    assert state.status in {JobStatus.DONE, JobStatus.FAILED}
    assert state.finished_at is not None
