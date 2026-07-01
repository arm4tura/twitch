"""Job runners: обёртки над CLI-функциями для FastAPI.

Каждый runner:
1. Строит `Progress` объект. Progress.emit(stage, percent, message) обновляет
   `state` джобы И пушит событие в её очередь (для WS-подписчиков).
2. Крутит существующие CLI-функции внутри `asyncio.to_thread(...)` — они
   CPU-bound и заблокировали бы event-loop.
3. В finally: ставит финальный статус, пушит `type='final'` event и sentinel.

Прогресс coarse-grained: extract_audio(0→10) → transcribe(10→75) →
detect_profanity(75→85) → build_decisions(85→95) → write_outputs(95→100).
Внутри transcribe нет callback'а — шлём только «начало/конец стадии».
Тонкую детализацию оставим на Фазу 2 (там свои прогрессы, тот же механизм).
"""

from __future__ import annotations

import asyncio
import logging
import traceback
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable

from ..cache import file_content_hash, read_json, stable_hash, write_json
from ..config import DEFAULT_ASR_OPTIONS, PipelineConfig
from ..decisions import build_decisions, write_decisions
from ..ffmpeg_tools import extract_audio_range, probe_media_duration
from ..llm.exporter import build_notebooklm_package
from ..llm.importer import MergeError, merge_into_decisions, parse_response
from ..profanity import ProfanityMatch, RussianNormalizer, detect_profanity, load_banwords
from ..timecode import TimeSpan, parse_timecode
from ..transcription import load_mock_transcript, transcribe_audio
from ..vegas_export import VegasExportError, write_vegas_script
from ..whisper_cpp import WhisperCppError, transcribe_with_whisper_cpp
from .projects import register_project
from .schemas import (
    ExportVegasRequest,
    HighlightsExportRequest,
    HighlightsImportRequest,
    ProcessJobRequest,
)
from .state import JobState, JobStatus, JobStore

logger = logging.getLogger(__name__)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class Progress:
    """Прогресс-эмиттер. Хранит только ссылки — state сам живёт в JobStore."""

    def __init__(self, store: JobStore, state: JobState) -> None:
        self._store = store
        self._state = state

    def emit(self, stage: str, percent: float, message: str = "") -> None:
        self._state.stage = stage
        self._state.progress = max(0.0, min(100.0, float(percent)))
        self._state.message = message
        self._store.emit(
            self._state.id,
            {"type": "progress", "state": self._state.model_dump(mode="json")},
        )


# --- utilities ---------------------------------------------------------------


def _resolve_range(stream: Path, range_in: str | None, range_out: str | None) -> tuple[TimeSpan, TimeSpan]:
    start = parse_timecode(range_in) if range_in else TimeSpan(0)
    end = parse_timecode(range_out) if range_out else probe_media_duration(stream)
    if end <= start:
        raise ValueError("range_out must be greater than range_in")
    return start, end


def _detection_cache_path(workdir: Path, key: str) -> Path:
    return workdir / "cache" / f"detection_{key}.json"


def _write_detection_cache(path: Path, key: str, matches: list[ProfanityMatch]) -> None:
    write_json(
        path,
        {
            "stage": "profanity_detection",
            "key": key,
            "matches": [asdict(m) for m in matches],
        },
    )


def _load_detection_cache(path: Path) -> list[ProfanityMatch]:
    payload = read_json(path)
    return [ProfanityMatch(**item) for item in payload["matches"]]


def _finalize(store: JobStore, state: JobState, status: JobStatus, *, error: str | None = None,
              result: dict[str, Any] | None = None) -> None:
    """Общее завершение: статус + финальный event + sentinel."""
    state.status = status
    state.error = error
    if result is not None:
        state.result = result
    state.finished_at = _now_iso()
    store.emit(state.id, {"type": "final", "state": state.model_dump(mode="json")})
    store.close_stream(state.id)


# --- process job -------------------------------------------------------------


def _run_process_sync(req: ProcessJobRequest, progress: Progress) -> dict[str, Any]:
    """Синхронная реализация. Гоняется в to_thread. Отсюда нельзя await."""
    stream = Path(req.stream)
    original = Path(req.original)
    banwords = Path(req.banwords)
    workdir = Path(req.workdir)
    decisions_out = Path(req.decisions)
    vegas_out = Path(req.vegas)

    workdir.mkdir(parents=True, exist_ok=True)

    config = PipelineConfig(
        language=req.language,
        model=req.model,
        device=req.device,
        compute_type=req.compute_type,
        batch_size=req.batch_size,
        vad_filter=req.vad_filter,
        vad_method=req.vad_method,
        asr_options=dict(DEFAULT_ASR_OPTIONS),
        mute_padding_before_ms=req.mute_padding_before_ms,
        mute_padding_after_ms=req.mute_padding_after_ms,
        mute_extend_mode=req.mute_extend_mode,
        mute_max_seconds=req.mute_max_seconds,
        mute_join_gap_ms=req.mute_join_gap_ms,
    )
    config.validate()

    start, end = _resolve_range(stream, req.range_in, req.range_out)

    audio_path: Path | None = None
    audio_key = "mock"

    if req.mock_transcript is None:
        progress.emit("extract_audio", 2.0, "ffmpeg → wav")
        audio_path, audio_key = extract_audio_range(
            stream, start, end, workdir, force=req.force_extract
        )
        progress.emit("extract_audio", 10.0, f"audio ready: {audio_path.name}")

        progress.emit("transcribe", 12.0, f"whisperx model={req.model}")
        transcript, transcript_key, transcript_cache = transcribe_audio(
            audio_path=audio_path,
            workdir=workdir,
            model_name=req.model,
            language=req.language,
            device=req.device,
            compute_type=req.compute_type,
            batch_size=req.batch_size,
            vad_filter=req.vad_filter,
            vad_method=req.vad_method,
            asr_options=config.asr_options,
            force=req.force_transcribe,
        )
        progress.emit("transcribe", 75.0, f"transcript ready ({len(transcript.get('segments', []))} segments)")
    else:
        progress.emit("mock_transcript", 40.0, f"reading {req.mock_transcript}")
        mock_path = Path(req.mock_transcript)
        transcript = load_mock_transcript(mock_path)
        transcript_key = stable_hash(
            {
                "stage": "mock_transcript",
                "path": str(mock_path.resolve()),
                "content": file_content_hash(mock_path),
            }
        )
        transcript_cache = mock_path
        progress.emit("mock_transcript", 75.0, "mock transcript loaded")

    detection_key = stable_hash(
        {
            "stage": "profanity_detection",
            "transcript_key": transcript_key,
            "banwords": {"path": str(banwords.resolve()), "content": file_content_hash(banwords)},
            "range_in_ms": start.ms,
            "range_out_ms": end.ms,
            "normalization": "lowercase+yo+pymorphy3",
        }
    )
    detection_cache = _detection_cache_path(workdir, detection_key)

    progress.emit("detect_profanity", 78.0, "load banwords + detect")
    if detection_cache.exists() and not req.force_detect:
        matches = _load_detection_cache(detection_cache)
    else:
        normalizer = RussianNormalizer()
        entries = load_banwords(banwords, normalizer=normalizer)
        matches = detect_profanity(transcript, entries, start, normalizer=normalizer)
        _write_detection_cache(detection_cache, detection_key, matches)
    progress.emit("detect_profanity", 85.0, f"{len(matches)} matches")

    progress.emit("build_decisions", 88.0, "compose decisions.json")
    decisions_doc = build_decisions(
        stream_path=stream,
        original_path=original,
        range_in=start,
        range_out=end,
        matches=matches,
        config=config,
        transcript_cache=transcript_cache,
        audio_cache=audio_path,
    )
    decisions_doc["caches"]["audio_key"] = audio_key
    decisions_doc["caches"]["transcript_key"] = transcript_key
    decisions_doc["caches"]["detection"] = str(detection_cache)
    decisions_doc["caches"]["detection_key"] = detection_key
    progress.emit("build_decisions", 95.0, f"{len(decisions_doc.get('mutes', []))} mutes")

    progress.emit("write_outputs", 97.0, "write decisions + vegas")
    write_decisions(decisions_out, decisions_doc)
    write_vegas_script(vegas_out, decisions_doc)
    progress.emit("write_outputs", 100.0, "done")

    return {
        "decisions_path": str(decisions_out),
        "vegas_path": str(vegas_out),
        "mutes_count": len(decisions_doc.get("mutes", [])),
        "cuts_count": len(decisions_doc.get("cuts", [])),
    }


async def run_process_job(store: JobStore, state: JobState, req: ProcessJobRequest) -> None:
    progress = Progress(store, state)
    state.status = JobStatus.RUNNING
    progress.emit("init", 0.0, "starting process job")
    try:
        result = await asyncio.to_thread(_run_process_sync, req, progress)
        # Регистрируем в реестре Recent Projects — best-effort, ошибку глотает
        # сама register_project. Делаем это ПОСЛЕ успешного write_decisions,
        # чтобы битые/незавершённые пути в Dashboard не всплывали.
        register_project(req.decisions)
        _finalize(store, state, JobStatus.DONE, result=result)
    except asyncio.CancelledError:
        _finalize(store, state, JobStatus.CANCELLED, error="cancelled by user")
        raise
    except Exception as exc:  # noqa: BLE001
        logger.exception("process job failed")
        _finalize(store, state, JobStatus.FAILED, error=f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}")


# --- export_vegas job --------------------------------------------------------


def _run_export_vegas_sync(req: ExportVegasRequest, progress: Progress) -> dict[str, Any]:
    progress.emit("read", 20.0, "read decisions.json")
    decisions_doc = read_json(Path(req.decisions))
    progress.emit("write", 60.0, "write vegas .cs")
    try:
        write_vegas_script(Path(req.vegas), decisions_doc)
    except VegasExportError as exc:
        raise ValueError(f"vegas export: {exc}") from exc
    progress.emit("write", 100.0, "done")
    return {"vegas_path": req.vegas}


async def run_export_vegas_job(store: JobStore, state: JobState, req: ExportVegasRequest) -> None:
    progress = Progress(store, state)
    state.status = JobStatus.RUNNING
    progress.emit("init", 0.0, "starting export")
    try:
        result = await asyncio.to_thread(_run_export_vegas_sync, req, progress)
        _finalize(store, state, JobStatus.DONE, result=result)
    except asyncio.CancelledError:
        _finalize(store, state, JobStatus.CANCELLED, error="cancelled by user")
        raise
    except Exception as exc:  # noqa: BLE001
        logger.exception("export_vegas job failed")
        _finalize(store, state, JobStatus.FAILED, error=f"{type(exc).__name__}: {exc}")


# --- highlights_export job ---------------------------------------------------


def _resolve_transcript_path(decisions_doc: dict[str, Any], override: str | None) -> Path:
    if override is not None:
        p = Path(override)
        if not p.exists():
            raise ValueError(f"transcript not found: {p}")
        return p
    caches = decisions_doc.get("caches") or {}
    cached = caches.get("transcript")
    if not cached:
        raise ValueError("decisions.caches.transcript missing — pass `transcript` explicitly")
    p = Path(cached)
    if not p.exists():
        raise ValueError(f"transcript from decisions.caches.transcript not found: {p}")
    return p


def _run_highlights_export_sync(req: HighlightsExportRequest, progress: Progress) -> dict[str, Any]:
    progress.emit("read", 10.0, "read decisions + transcript")
    decisions_doc = read_json(Path(req.decisions))
    transcript_path = _resolve_transcript_path(decisions_doc, req.transcript)
    transcript_doc = read_json(transcript_path)

    progress.emit("build_package", 40.0, "compose notebooklm package")
    manifest = build_notebooklm_package(
        transcript_doc, Path(req.out_dir), n_highlights=req.n_highlights
    )
    progress.emit("build_package", 100.0, f"{len(manifest['chunks'])} chunks")
    return {"out_dir": req.out_dir, "manifest": manifest}


async def run_highlights_export_job(store: JobStore, state: JobState, req: HighlightsExportRequest) -> None:
    progress = Progress(store, state)
    state.status = JobStatus.RUNNING
    progress.emit("init", 0.0, "starting highlights-export")
    try:
        result = await asyncio.to_thread(_run_highlights_export_sync, req, progress)
        _finalize(store, state, JobStatus.DONE, result=result)
    except asyncio.CancelledError:
        _finalize(store, state, JobStatus.CANCELLED, error="cancelled by user")
        raise
    except Exception as exc:  # noqa: BLE001
        logger.exception("highlights_export job failed")
        _finalize(store, state, JobStatus.FAILED, error=f"{type(exc).__name__}: {exc}")


# --- highlights_import job ---------------------------------------------------


def _run_highlights_import_sync(req: HighlightsImportRequest, progress: Progress) -> dict[str, Any]:
    progress.emit("read", 10.0, "read decisions + response")
    decisions_doc = read_json(Path(req.decisions))

    range_s: tuple[float, float] | None = None
    transcript_hash: str | None = None
    try:
        tpath = _resolve_transcript_path(decisions_doc, req.transcript)
        tdoc = read_json(tpath)
        segments = tdoc.get("segments") or []
        if segments:
            first = segments[0].get("start") or 0.0
            last = segments[-1].get("end") or first
            range_s = (float(first), float(last))
        transcript_hash = file_content_hash(tpath)
    except ValueError:
        # No transcript — skip range check (mirrors CLI behavior).
        progress.emit("read", 20.0, "transcript missing — skipping range check")

    progress.emit("parse", 50.0, "parse + validate response")
    try:
        hs = parse_response(
            Path(req.response),
            transcript_range_s=range_s,
            transcript_hash=transcript_hash,
        )
    except MergeError as exc:
        raise ValueError("MergeError:\n" + "\n".join(f" • {r}" for r in exc.reasons)) from exc

    progress.emit("merge", 80.0, "merge into decisions.json")
    merged = merge_into_decisions(decisions_doc, hs)

    progress.emit("write", 95.0, "write output")
    write_json(Path(req.output), merged)
    progress.emit("write", 100.0, f"{len(hs.highlights)} highlights merged")

    return {
        "output": req.output,
        "highlights_count": len(hs.highlights),
        "highlights": [h.model_dump() for h in hs.highlights],
    }


async def run_highlights_import_job(store: JobStore, state: JobState, req: HighlightsImportRequest) -> None:
    progress = Progress(store, state)
    state.status = JobStatus.RUNNING
    progress.emit("init", 0.0, "starting highlights-import")
    try:
        result = await asyncio.to_thread(_run_highlights_import_sync, req, progress)
        _finalize(store, state, JobStatus.DONE, result=result)
    except asyncio.CancelledError:
        _finalize(store, state, JobStatus.CANCELLED, error="cancelled by user")
        raise
    except Exception as exc:  # noqa: BLE001
        logger.exception("highlights_import job failed")
        _finalize(store, state, JobStatus.FAILED, error=f"{type(exc).__name__}: {exc}")


# --- dispatch ----------------------------------------------------------------

# Единая точка входа для app.py — принимает request-модель и создаёт task.
JobRunner = Callable[[JobStore, JobState, Any], Awaitable[None]]

RUNNERS: dict[str, JobRunner] = {
    "process": run_process_job,
    "export_vegas": run_export_vegas_job,
    "highlights_export": run_highlights_export_job,
    "highlights_import": run_highlights_import_job,
}
