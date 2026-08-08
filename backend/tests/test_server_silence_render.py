"""Тесты server endpoints для detect-silence и render jobs."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from twitch_cut.server.runner import run_detect_silence_job, run_render_job
from twitch_cut.server.schemas import DetectSilenceRequest, RenderRequest
from twitch_cut.server.state import JobKind, JobStatus, JobStore


async def _drain(store: JobStore, job_id: str) -> list[dict]:
    events: list[dict] = []
    async for ev in store.subscribe(job_id):
        events.append(ev)
    return events


# --- detect_silence runner ---------------------------------------------------


def test_detect_silence_runner_success(tmp_path, monkeypatch):
    """Detect-silence job: успешный детект с merge + padding."""
    store = JobStore()
    state = store.create(JobKind.DETECT_SILENCE)

    video = tmp_path / "in.mp4"
    video.write_bytes(b"fake mp4")

    # Mock detect_silence_ffmpeg → два сегмента
    def _fake_detect(audio_path, threshold_db, min_duration, workdir, force):
        return [(1.0, 3.0), (5.0, 7.0)]

    monkeypatch.setattr("twitch_cut.server.runner.detect_silence_ffmpeg", _fake_detect)

    req = DetectSilenceRequest(
        video_path=str(video),
        threshold_db=-30.0,
        min_duration=0.5,
        merge_gap=0.5,
        padding_before=0.1,
        padding_after=0.2,
    )

    async def _go():
        await run_detect_silence_job(store, state, req)
        return await _drain(store, state.id)

    events = asyncio.run(_go())
    assert state.status == JobStatus.DONE, state.error
    result = state.result
    assert result["count"] == 2
    # Padding применён: (1.0-0.1, 3.0+0.2), (5.0-0.1, 7.0+0.2)
    assert result["segments"] == [(0.9, 3.2), (4.9, 7.2)]
    assert events[-1]["type"] == "final"


def test_detect_silence_runner_no_segments(tmp_path, monkeypatch):
    """Detect-silence: файл без тишины."""
    store = JobStore()
    state = store.create(JobKind.DETECT_SILENCE)

    video = tmp_path / "in.mp4"
    video.write_bytes(b"fake mp4")

    monkeypatch.setattr("twitch_cut.server.runner.detect_silence_ffmpeg", lambda *a, **k: [])

    req = DetectSilenceRequest(video_path=str(video))
    asyncio.run(run_detect_silence_job(store, state, req))
    assert state.status == JobStatus.DONE
    assert state.result["count"] == 0
    assert state.result["segments"] == []


def test_detect_silence_runner_missing_file(tmp_path):
    """Detect-silence: файл не существует → FAILED."""
    store = JobStore()
    state = store.create(JobKind.DETECT_SILENCE)

    req = DetectSilenceRequest(video_path=str(tmp_path / "nope.mp4"))
    asyncio.run(run_detect_silence_job(store, state, req))
    assert state.status == JobStatus.FAILED
    assert "video not found" in state.error


# --- render runner -----------------------------------------------------------


def test_render_runner_success(tmp_path, monkeypatch):
    """Render job: успешный рендер с прогрессом."""
    store = JobStore()
    state = store.create(JobKind.RENDER)

    inp = tmp_path / "in.mp4"
    inp.write_bytes(b"fake input")
    out = tmp_path / "out.mp4"

    # Mock render_video + probe_media_duration
    def _fake_probe(p):
        from twitch_cut.timecode import TimeSpan
        return TimeSpan.from_seconds(60.0)

    def _fake_render(input_path, output_path, segments_to_remove, **kwargs):
        # Вызываем progress callback несколько раз
        progress = kwargs.get("progress")
        if progress:
            progress(0.0, "start")
            progress(50.0, "encoding")
            progress(100.0, "done")
        output_path.write_bytes(b"fake rendered")
        return output_path

    monkeypatch.setattr("twitch_cut.server.runner.probe_media_duration", _fake_probe)
    monkeypatch.setattr("twitch_cut.server.runner._render_video", _fake_render)

    req = RenderRequest(
        input_path=str(inp),
        output_path=str(out),
        segments_to_remove=[(10.0, 20.0)],
    )

    async def _go():
        # Drain events concurrently while job runs
        drain_task = asyncio.create_task(_drain(store, state.id))
        await run_render_job(store, state, req)
        events = await drain_task
        return events

    events = asyncio.run(_go())
    assert state.status == JobStatus.DONE, state.error
    assert out.exists()
    result = state.result
    assert result["output_size"] == len(b"fake rendered")
    assert result["removed_count"] == 1
    assert result["kept_duration"] == 50.0  # 60 - 10
    # Проверим что прогресс репортился
    progress_events = [e for e in events if e["type"] == "progress"]
    assert len(progress_events) > 0


def test_render_runner_missing_input(tmp_path):
    """Render: входной файл не существует → FAILED."""
    store = JobStore()
    state = store.create(JobKind.RENDER)

    req = RenderRequest(
        input_path=str(tmp_path / "nope.mp4"),
        output_path=str(tmp_path / "out.mp4"),
        segments_to_remove=[(1.0, 2.0)],
    )
    asyncio.run(run_render_job(store, state, req))
    assert state.status == JobStatus.FAILED
    assert "input video not found" in state.error


def test_render_runner_remove_everything(tmp_path, monkeypatch):
    """Render: удаляем весь файл → FAILED (ValueError)."""
    store = JobStore()
    state = store.create(JobKind.RENDER)

    inp = tmp_path / "in.mp4"
    inp.write_bytes(b"fake")

    def _fake_probe(p):
        from twitch_cut.timecode import TimeSpan
        return TimeSpan.from_seconds(10.0)

    def _fake_render(*args, **kwargs):
        raise ValueError("Nothing to keep")

    monkeypatch.setattr("twitch_cut.server.runner.probe_media_duration", _fake_probe)
    monkeypatch.setattr("twitch_cut.server.runner._render_video", _fake_render)

    req = RenderRequest(
        input_path=str(inp),
        output_path=str(tmp_path / "out.mp4"),
        segments_to_remove=[(0.0, 10.0)],
    )
    asyncio.run(run_render_job(store, state, req))
    assert state.status == JobStatus.FAILED
    assert "Nothing to keep" in state.error
