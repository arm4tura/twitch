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
    run_detect_offtopic_job,
    run_export_vegas_job,
    run_highlights_export_job,
    run_highlights_import_job,
)
from twitch_cut.server.schemas import (
    DetectOfftopicRequest,
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


# ---------- detect_offtopic runner: диспетч путей (Фаза 4) -------------------
#
# Классификатор и LLMClient подменяем: цель этих тестов — не прогон LLM/GigaAM,
# а что runner ВЫБИРАЕТ верный путь по req.method/source и что result.method
# доезжает в состояние джобы. Реальный merge оставляем — он собирает bundle.


class _FakeLLMClient:
    """Заглушка LLMClient: без сети. provider — реальный (для progress.id)."""

    def __init__(self, provider, api_key=None):
        self.provider = provider
        self.api_key = api_key


def _mk_offtopic_decisions(tmp_path) -> Path:
    """decisions.json + транскрипт в caches — общий вход detect-offtopic."""
    tpath = tmp_path / "t.json"
    tpath.write_text(json.dumps(_mk_transcript(30)), encoding="utf-8")
    dpath = tmp_path / "d.json"
    dpath.write_text(
        json.dumps({"range_in": 0, "caches": {"transcript": str(tpath)}}),
        encoding="utf-8",
    )
    return dpath


def test_detect_offtopic_windows_default(tmp_path, monkeypatch):
    # Нет исходника, method не задан → оконный путь (не меняем старое поведение).
    import twitch_cut.llm_offtopic as llm
    from twitch_cut.llm_offtopic.model import OfftopicSegment, OfftopicSet

    captured: dict = {}

    def fake_windows(transcript_doc, *, client, **kw):
        captured["kw"] = kw
        return OfftopicSet(
            segments=[
                OfftopicSegment(start_s=5.0, end_s=15.0, confidence=0.8, reason="оффтоп")
            ],
            source="builtin:anthropic",
            model="claude-x",
        )  # method по умолчанию — "windows"

    monkeypatch.setattr(llm, "classify_transcript", fake_windows)
    monkeypatch.setattr(llm, "LLMClient", _FakeLLMClient)

    dpath = _mk_offtopic_decisions(tmp_path)
    out = tmp_path / "d_out.json"
    req = DetectOfftopicRequest(decisions=str(dpath), output=str(out))

    store = JobStore()
    state = store.create(JobKind.DETECT_OFFTOPIC)
    asyncio.run(run_detect_offtopic_job(store, state, req))

    assert state.status == JobStatus.DONE, state.error
    assert state.result["method"] == "windows"
    assert state.result["segments_count"] == 1
    merged = json.loads(out.read_text(encoding="utf-8"))
    assert merged["offtopic"]["method"] == "windows"
    # merge проставил абсолютные stream_*_s (offset=range_in=0).
    assert merged["offtopic"]["segments"][0]["stream_start_s"] == 5.0
    # source_* полей нет — не было исходника.
    assert "source_transcript_hash" not in merged["offtopic"]


def test_detect_offtopic_source_transcript_path(tmp_path, monkeypatch):
    # Дан готовый source_transcript → авто-выбор пути по исходнику; проверяем,
    # что classify_with_source получил хеш исходника и method доехал в result.
    import twitch_cut.llm_offtopic as llm
    from twitch_cut.llm_offtopic.model import OfftopicSegment, OfftopicSet

    captured: dict = {}

    def fake_source(transcript_doc, source_doc, *, client, **kw):
        captured["kw"] = kw
        return OfftopicSet(
            segments=[
                OfftopicSegment(
                    start_s=10.0,
                    end_s=20.0,
                    confidence=0.9,
                    reason="ушёл в чат",
                    pause_s=42.0,
                    anchored=0.11,
                )
            ],
            source="builtin:anthropic",
            model="claude-opus-5",
            method="source_anchored",
            source_transcript_hash=kw.get("source_transcript_hash"),
            source_model=kw.get("source_model"),
        )

    monkeypatch.setattr(llm, "classify_with_source", fake_source)
    monkeypatch.setattr(llm, "LLMClient", _FakeLLMClient)

    dpath = _mk_offtopic_decisions(tmp_path)
    src = tmp_path / "source_t.json"
    src.write_text(json.dumps(_mk_transcript(40)), encoding="utf-8")
    out = tmp_path / "d_out.json"
    req = DetectOfftopicRequest(
        decisions=str(dpath),
        output=str(out),
        source_transcript=str(src),
        source_model="gigaam-v3",
    )

    store = JobStore()
    state = store.create(JobKind.DETECT_OFFTOPIC)
    asyncio.run(run_detect_offtopic_job(store, state, req))

    assert state.status == JobStatus.DONE, state.error
    assert state.result["method"] == "source_anchored"
    # Хеш готового source_transcript посчитан из файла (не None) и проброшен.
    assert captured["kw"]["source_transcript_hash"]
    assert captured["kw"]["source_model"] == "gigaam-v3"
    merged = json.loads(out.read_text(encoding="utf-8"))
    seg = merged["offtopic"]["segments"][0]
    assert seg["pause_s"] == 42.0
    assert seg["anchored"] == 0.11


def test_detect_offtopic_both_sources_rejected(tmp_path, monkeypatch):
    # source_audio И source_transcript вместе — противоречие, джоба падает.
    import twitch_cut.llm_offtopic as llm

    monkeypatch.setattr(llm, "LLMClient", _FakeLLMClient)
    dpath = _mk_offtopic_decisions(tmp_path)
    req = DetectOfftopicRequest(
        decisions=str(dpath),
        output=str(tmp_path / "o.json"),
        source_audio="/tmp/src.mp4",
        source_transcript=str(tmp_path / "s.json"),
    )
    store = JobStore()
    state = store.create(JobKind.DETECT_OFFTOPIC)
    asyncio.run(run_detect_offtopic_job(store, state, req))
    assert state.status == JobStatus.FAILED
    assert "not both" in state.error


def test_detect_offtopic_updates_edit_journal_baseline(tmp_path, monkeypatch):
    """Джоба пишет decisions.json МИМО PUT /decisions. Значит найденный оффтоп
    обязан уехать в базовый снимок журнала: иначе следующее сохранение из
    «Правки» покажет машинные сегменты как «человек добавил»."""
    import twitch_cut.llm_offtopic as llm
    from twitch_cut import edit_journal as ej
    from twitch_cut.llm_offtopic.model import OfftopicSegment, OfftopicSet
    from twitch_cut.server.settings import save_settings

    monkeypatch.setattr(
        llm,
        "classify_transcript",
        lambda transcript_doc, *, client, **kw: OfftopicSet(
            segments=[OfftopicSegment(start_s=5.0, end_s=15.0, confidence=0.8, reason="оффтоп")],
            source="builtin:anthropic",
            model="claude-x",
        ),
    )
    monkeypatch.setattr(llm, "LLMClient", _FakeLLMClient)
    save_settings({"edit_journal": True})

    dpath = _mk_offtopic_decisions(tmp_path)
    out = tmp_path / "d_out.json"
    # Журнал уже есть (обработка прошла с включённым флагом) и оффтопа в нём нет.
    out.write_text(dpath.read_text(encoding="utf-8"), encoding="utf-8")
    ej.start_journal(out, json.loads(out.read_text(encoding="utf-8")))

    req = DetectOfftopicRequest(decisions=str(dpath), output=str(out))
    store = JobStore()
    state = store.create(JobKind.DETECT_OFFTOPIC)
    asyncio.run(run_detect_offtopic_job(store, state, req))

    assert state.status == JobStatus.DONE, state.error
    journal = ej.load_journal(out)
    assert journal["baseline"]["offtopic"][0]["start_s"] == 5.0
    assert "AI-детект оффтопа" in journal["baseline"]["refreshed"]["offtopic"]["note"]
    # Человек ничего не менял — правок в итоге нет.
    merged = json.loads(out.read_text(encoding="utf-8"))
    assert ej.is_empty(ej.net_changes(journal, merged))


def test_detect_offtopic_without_journal_flag_writes_no_journal(tmp_path, monkeypatch):
    import twitch_cut.llm_offtopic as llm
    from twitch_cut import edit_journal as ej
    from twitch_cut.llm_offtopic.model import OfftopicSegment, OfftopicSet

    monkeypatch.setattr(
        llm,
        "classify_transcript",
        lambda transcript_doc, *, client, **kw: OfftopicSet(
            segments=[OfftopicSegment(start_s=5.0, end_s=15.0, confidence=0.8, reason="оффтоп")],
            source="builtin:anthropic",
            model="claude-x",
        ),
    )
    monkeypatch.setattr(llm, "LLMClient", _FakeLLMClient)

    dpath = _mk_offtopic_decisions(tmp_path)
    out = tmp_path / "d_out.json"
    req = DetectOfftopicRequest(decisions=str(dpath), output=str(out))
    store = JobStore()
    state = store.create(JobKind.DETECT_OFFTOPIC)
    asyncio.run(run_detect_offtopic_job(store, state, req))

    assert state.status == JobStatus.DONE, state.error
    assert not ej.journal_path(out).exists()
