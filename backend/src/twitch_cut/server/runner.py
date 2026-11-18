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
import weakref
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..cache import file_content_hash, read_json, write_json
from ..config import DEFAULT_ASR_OPTIONS, PipelineConfig
from ..ffmpeg_tools import probe_media_duration
from ..llm.exporter import build_notebooklm_package
from ..llm.importer import MergeError, merge_into_decisions, parse_text
from ..render import _safe_slug
from ..render import render_clip as _render_clip
from ..render import render_video as _render_video
from ..silence import add_padding, detect_silence_ffmpeg, merge_close_segments
from ..vegas_export import (
    VegasExportError,
    write_vegas_highlights_script,
    write_vegas_script,
)
from .projects import register_project
from .schemas import (
    DetectOfftopicRequest,
    DetectSilenceRequest,
    DownloadVodRequest,
    ExportVegasHighlightsRequest,
    ExportVegasRequest,
    HighlightsExportRequest,
    HighlightsImportRequest,
    ProcessJobRequest,
    RenderHighlightsRequest,
    RenderRequest,
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


# --- тяжёлые джобы: ровно одна за раз ---------------------------------------

HEAVY_JOB_SLOTS = 1
"""Сколько ASR-обработок разрешено крутить одновременно.

Одна. GigaAM грузит ~2 ГБ весов на MPS/CUDA, и N параллельных сессий не
ускоряют работу, а отбирают друг у друга GPU и память: на 16 ГБ Mac десяток
инстансов уводит машину в swap, и НИ ОДИН не доезжает.

Хуже того, без этого лимита переполняется дефолтный пул `asyncio.to_thread`
(min(32, cpu+4) — на 10-ядерном Mac это 14 воркеров). Джоба номер 15 висит в
очереди executor'а: статус running, стадия init, 0 %, ни строчки в логе — с
виду «застряло на Подготовке» навсегда. Ждать в семафоре честнее: мы хотя бы
можем сказать пользователю, что он в очереди.
"""

_heavy_gates: "weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, asyncio.Semaphore]" = (
    weakref.WeakKeyDictionary()
)


def _heavy_gate() -> asyncio.Semaphore:
    """Семафор тяжёлых джоб, привязанный к ТЕКУЩЕМУ event-loop.

    Не модульная константа: `asyncio.Semaphore` привязывается к первому
    использовавшему его loop'у, а тесты гоняют каждый прогон через свой
    `asyncio.run(...)` — общий объект падал бы с «bound to a different event
    loop». В проде loop один, так что словарь всегда из одной записи.
    """
    loop = asyncio.get_running_loop()
    gate = _heavy_gates.get(loop)
    if gate is None:
        gate = asyncio.Semaphore(HEAVY_JOB_SLOTS)
        _heavy_gates[loop] = gate
    return gate


# --- process job -------------------------------------------------------------


def _run_process_sync(req: ProcessJobRequest, progress: Progress) -> dict[str, Any]:
    """Синхронная реализация. Гоняется в to_thread. Отсюда нельзя await.

    Пустые пути (простой режим UI) заполняются дефолтами: workdir — уникальная
    папка в ~/twitch_cut/projects, decisions/vegas — внутри неё, banwords —
    встроенный словарь, original — сам stream (реакция == запись, если второго
    видео нет).

    Сам пайплайн живёт в `pipeline.run_transcription_pipeline` — общий с CLI,
    чтобы движки и порядок стадий не расходились. Тут только резолв дефолтов
    UI-режима (через общий `resolve_job_defaults`) и адаптер прогресса под WS.
    """
    from ..pipeline import run_transcription_pipeline
    from .paths import resolve_job_defaults

    d = resolve_job_defaults(
        stream=req.stream,
        original=req.original,
        banwords=req.banwords,
        workdir=req.workdir,
        decisions=req.decisions,
        vegas=req.vegas,
        device=req.device,
        compute_type=req.compute_type,
    )

    config = PipelineConfig(
        transcriber=req.transcriber,
        language=req.language,
        model=req.model,
        device=d.device,
        compute_type=d.compute_type,
        batch_size=req.batch_size,
        vad_filter=req.vad_filter,
        vad_method=req.vad_method,
        asr_options=dict(DEFAULT_ASR_OPTIONS),
        sensitivity=req.sensitivity,
        mute_padding_before_ms=req.mute_padding_before_ms,
        mute_padding_after_ms=req.mute_padding_after_ms,
        mute_extend_mode=req.mute_extend_mode,
        mute_max_seconds=req.mute_max_seconds,
        mute_join_gap_ms=req.mute_join_gap_ms,
        mute_max_word_seconds=req.mute_max_word_seconds,
        mute_trim_to_audio=req.mute_trim_to_audio,
        mute_trim_peak_ratio=req.mute_trim_peak_ratio,
        mute_trim_guard_ms=req.mute_trim_guard_ms,
        mute_trim_min_ms=req.mute_trim_min_ms,
        mute_expand_on_silence_ms=req.mute_expand_on_silence_ms,
        # Без этой строки поле есть в схеме, видно в UI и не работает: конфиг тут
        # собирается перечислением. Страж — test_sensitivity.py::
        # test_every_knob_from_the_request_reaches_the_pipeline.
        mute_head_cross_prev_ms=req.mute_head_cross_prev_ms,
        # Спотинг по посториорам: из API не пробрасывался вообще, то есть
        # выключить его или сдвинуть порог из UI было нельзя.
        spot_posteriors=req.spot_posteriors,
        spot_min_score=req.spot_min_score,
    )
    config = config.normalized()
    config.validate()

    return run_transcription_pipeline(
        stream=Path(req.stream),
        original=d.original,
        banwords=d.banwords,
        workdir=d.workdir,
        decisions_out=d.decisions,
        vegas_out=d.vegas,
        config=config,
        range_in=req.range_in,
        range_out=req.range_out,
        gigaam_model=req.gigaam_model,
        mock_transcript=Path(req.mock_transcript) if req.mock_transcript else None,
        force_extract=req.force_extract,
        force_transcribe=req.force_transcribe,
        force_detect=req.force_detect,
        progress=progress.emit,
    )


def _start_edit_journal(decisions_path: str | Path) -> None:
    """Снять базовый снимок для журнала правок — решение ПАЙПЛАЙНА.

    Только когда журнал включён в настройках. Best-effort: диагностика не имеет
    права ронять только что успешно завершённую обработку.
    """
    try:
        from .settings import load_settings

        if not bool(load_settings().get("edit_journal")):
            return
        from ..edit_journal import start_journal

        path = Path(decisions_path)
        start_journal(path, read_json(path))
    except Exception:  # noqa: BLE001
        logger.warning("edit journal baseline failed for %s", decisions_path, exc_info=True)


def _refresh_edit_journal(
    decisions_path: str | Path, doc: dict[str, Any], layers: tuple[str, ...], note: str
) -> None:
    """Записать в baseline журнала слои, которые только что посчитала машина.

    Джоба дописывает сегменты в decisions.json напрямую, минуя PUT /decisions;
    без этого они попали бы в отчёт как «человек добавил». Best-effort и только
    при включённом журнале — как и `_start_edit_journal`.
    """
    try:
        from .settings import load_settings

        if not bool(load_settings().get("edit_journal")):
            return
        from ..edit_journal import refresh_baseline

        refresh_baseline(Path(decisions_path), doc, layers, note=note)
    except Exception:  # noqa: BLE001
        logger.warning("edit journal refresh failed for %s", decisions_path, exc_info=True)


async def run_process_job(store: JobStore, state: JobState, req: ProcessJobRequest) -> None:
    progress = Progress(store, state)
    state.status = JobStatus.RUNNING
    progress.emit("init", 0.0, "starting process job")
    gate = _heavy_gate()
    try:
        # Ждём своей очереди ВНУТРИ try: отмена стоящей в очереди джобы должна
        # давать 'cancelled', а не тихо висеть.
        if gate.locked():
            progress.emit("init", 0.0, "в очереди: уже идёт другая обработка")
        async with gate:
            progress.emit("init", 0.0, "starting process job")
            result = await asyncio.to_thread(_run_process_sync, req, progress)
        # Регистрируем в реестре Recent Projects — best-effort, ошибку глотает
        # сама register_project. Делаем это ПОСЛЕ успешного write_decisions,
        # чтобы битые/незавершённые пути в Dashboard не всплывали. Берём путь
        # из результата (req.decisions мог быть None — простой режим).
        register_project(result["decisions_path"])
        _start_edit_journal(result["decisions_path"])
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
    # Источник ответа: вставленный текст (response_text) приоритетнее файла.
    # Так UI может слать содержимое textarea, не заставляя сохранять .json.
    if req.response_text is not None and req.response_text.strip():
        raw = req.response_text
    elif req.response:
        rpath = Path(req.response)
        if not rpath.exists():
            raise ValueError(f"response file not found: {rpath}")
        raw = rpath.read_text(encoding="utf-8")
    else:
        raise ValueError("Provide either `response` (file path) or `response_text`.")

    try:
        hs, warnings = parse_text(
            raw,
            transcript_range_s=range_s,
            transcript_hash=transcript_hash,
        )
    except MergeError as exc:
        raise ValueError("MergeError:\n" + "\n".join(f" • {r}" for r in exc.reasons)) from exc

    progress.emit("merge", 80.0, "merge into decisions.json")
    merged = merge_into_decisions(decisions_doc, hs)

    progress.emit("write", 95.0, "write output")
    write_json(Path(req.output), merged)
    tail = f" ({len(warnings)} skipped)" if warnings else ""
    progress.emit("write", 100.0, f"{len(hs.highlights)} highlights merged{tail}")

    # Отдаём обогащённые highlights (со stream_start_s/stream_end_s из merge),
    # чтобы UI сразу показал панель без повторного чтения decisions.json.
    return {
        "output": req.output,
        "highlights_count": len(hs.highlights),
        "highlights": merged["highlights"]["highlights"],
        "warnings": warnings,
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


# --- render_highlights job ---------------------------------------------------


def _read_highlights_bundle(decisions_doc: dict[str, Any]) -> list[dict[str, Any]]:
    """Достать список highlights из decisions bundle; None → пустой."""
    bundle = decisions_doc.get("highlights") or {}
    if not isinstance(bundle, dict):
        return []
    items = bundle.get("highlights")
    return items if isinstance(items, list) else []


def _run_render_highlights_sync(
    req: RenderHighlightsRequest, progress: Progress
) -> dict[str, Any]:
    """Нарезать каждый highlight из decisions.json в отдельный mp4."""
    progress.emit("read", 5.0, "read decisions")
    decisions_doc = read_json(Path(req.decisions))
    highlights = _read_highlights_bundle(decisions_doc)
    if not highlights:
        raise ValueError("decisions.json не содержит highlights — нечего нарезать")

    input_path = Path(req.input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"input video not found: {input_path}")

    out_dir = Path(req.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    clips: list[dict[str, Any]] = []
    total = len(highlights)
    for i, h in enumerate(highlights):
        start = float(h.get("stream_start_s", h.get("start_s")))
        end = float(h.get("stream_end_s", h.get("end_s")))
        # Padding наружу (не короче границ исходника).
        pad = req.padding_s
        s = max(0.0, start - pad)
        e = end + pad
        if e <= s:
            raise ValueError(f"highlight #{i} ({h.get('title', '?')}): end <= start")

        title = str(h.get("title", "highlight"))
        fname = f"{i + 1:02d}_{_safe_slug(title)}.mp4"
        out_path = out_dir / fname

        progress.emit(
            "render", 5.0 + (i / total) * 90.0, f"[{i + 1}/{total}] {title}"
        )

        def _adapter(pct: float, msg: str = "", _i: int = i, _t: int = total) -> None:
            # map render_clip 0-100 → наш слайс стадии
            progress.emit("render", 5.0 + ((_i + pct / 100.0) / _t) * 90.0, msg)

        _render_clip(
            input_path,
            out_path,
            s,
            e,
            quality=req.quality,
            progress=_adapter,
            force=req.force,
        )
        clips.append(
            {
                "index": i,
                "title": title,
                "start_s": s,
                "end_s": e,
                "path": str(out_path),
            }
        )

    progress.emit("done", 100.0, f"{len(clips)} clips rendered")
    return {"out_dir": str(out_dir), "clips": clips, "count": len(clips)}


async def run_render_highlights_job(store: JobStore, state: JobState, req: RenderHighlightsRequest) -> None:
    progress = Progress(store, state)
    state.status = JobStatus.RUNNING
    progress.emit("init", 0.0, "starting render-highlights")
    try:
        result = await asyncio.to_thread(_run_render_highlights_sync, req, progress)
        # Клипы теперь существуют — разрешаем их отдачу через /media, чтобы
        # плеер в панели highlights мог проиграть каждый.
        store.allow_media(*[c["path"] for c in result.get("clips", [])])
        _finalize(store, state, JobStatus.DONE, result=result)
    except asyncio.CancelledError:
        _finalize(store, state, JobStatus.CANCELLED, error="cancelled by user")
        raise
    except Exception as exc:  # noqa: BLE001
        logger.exception("render_highlights job failed")
        _finalize(store, state, JobStatus.FAILED, error=f"{type(exc).__name__}: {exc}")


# --- export_vegas_highlights job ---------------------------------------------


def _run_export_vegas_highlights_sync(
    req: ExportVegasHighlightsRequest, progress: Progress
) -> dict[str, Any]:
    """Собрать Vegas-скрипт с Region'ами на highlights."""
    progress.emit("read", 30.0, "read decisions")
    decisions_doc = read_json(Path(req.decisions))
    out_path = Path(req.vegas)
    progress.emit("build", 60.0, "compose highlights markers script")
    try:
        write_vegas_highlights_script(out_path, decisions_doc)
    except VegasExportError as exc:
        raise ValueError(str(exc)) from exc
    progress.emit("done", 100.0, f"script: {out_path.name}")
    return {"output": str(out_path)}


async def run_export_vegas_highlights_job(
    store: JobStore, state: JobState, req: ExportVegasHighlightsRequest
) -> None:
    progress = Progress(store, state)
    state.status = JobStatus.RUNNING
    progress.emit("init", 0.0, "starting export-vegas-highlights")
    try:
        result = await asyncio.to_thread(_run_export_vegas_highlights_sync, req, progress)
        _finalize(store, state, JobStatus.DONE, result=result)
    except asyncio.CancelledError:
        _finalize(store, state, JobStatus.CANCELLED, error="cancelled by user")
        raise
    except Exception as exc:  # noqa: BLE001
        logger.exception("export_vegas_highlights job failed")
        _finalize(store, state, JobStatus.FAILED, error=f"{type(exc).__name__}: {exc}")


# --- detect_silence job ------------------------------------------------------

def _run_detect_silence_sync(req: DetectSilenceRequest, progress: Progress) -> dict[str, Any]:
    """Детект тишины через ffmpeg silencedetect."""
    progress.emit("detect", 10.0, "starting silence detection")

    video_path = Path(req.video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"video not found: {video_path}")

    workdir = Path(req.workdir) if req.workdir else video_path.parent / "cache"

    progress.emit("detect", 30.0, f"ffmpeg silencedetect threshold={req.threshold_db}dB")
    segments = detect_silence_ffmpeg(
        video_path,
        threshold_db=req.threshold_db,
        min_duration=req.min_duration,
        workdir=workdir,
        force=req.force,
    )

    progress.emit("postprocess", 70.0, f"found {len(segments)} segments")
    if segments:
        segments = merge_close_segments(segments, gap_threshold=req.merge_gap)
        progress.emit("postprocess", 85.0, f"after merge: {len(segments)}")
        segments = add_padding(
            segments,
            padding_before=req.padding_before,
            padding_after=req.padding_after,
        )

    progress.emit("done", 100.0, f"final: {len(segments)} silence segments")

    return {
        "video_path": str(video_path),
        "segments": segments,
        "count": len(segments),
        "total_duration": sum(e - s for s, e in segments),
    }


async def run_detect_silence_job(store: JobStore, state: JobState, req: DetectSilenceRequest) -> None:
    progress = Progress(store, state)
    state.status = JobStatus.RUNNING
    progress.emit("init", 0.0, "starting detect-silence")
    try:
        result = await asyncio.to_thread(_run_detect_silence_sync, req, progress)
        _finalize(store, state, JobStatus.DONE, result=result)
    except asyncio.CancelledError:
        _finalize(store, state, JobStatus.CANCELLED, error="cancelled by user")
        raise
    except Exception as exc:  # noqa: BLE001
        logger.exception("detect_silence job failed")
        _finalize(store, state, JobStatus.FAILED, error=f"{type(exc).__name__}: {exc}")


# --- render job --------------------------------------------------------------


def _run_render_sync(req: RenderRequest, progress: Progress) -> dict[str, Any]:
    """Рендер видео с вырезанными сегментами."""
    progress.emit("prepare", 5.0, "validate paths")

    input_path = Path(req.input_path)
    output_path = Path(req.output_path)

    if not input_path.exists():
        raise FileNotFoundError(f"input video not found: {input_path}")

    total_duration = probe_media_duration(input_path).to_seconds()
    # Оценка «останется» с учётом диапазона: если задан range_start/range_end,
    # база — длина окна, а не всего файла (иначе прогресс/лог врут).
    lo = req.range_start if req.range_start is not None else 0.0
    hi = req.range_end if req.range_end is not None else total_duration
    window = max(0.0, hi - lo)
    # Вырезанное внутри окна: клампим каждый сегмент к [lo, hi], иначе сегмент
    # на границе окна раздул бы оценку.
    removed_in_window = sum(
        max(0.0, min(e, hi) - max(s, lo)) for s, e in req.segments_to_remove
    )
    kept_duration = max(0.0, window - removed_in_window)

    progress.emit("render", 10.0, f"rendering {kept_duration:.1f}s / {total_duration:.1f}s")

    def _progress_adapter(pct: float, msg: str = "") -> None:
        # Map render_video 0-100 → our 10-100
        mapped = 10.0 + (pct * 0.9)
        progress.emit("render", mapped, msg)

    _render_video(
        input_path,
        output_path,
        req.segments_to_remove,
        segments_to_mute=req.segments_to_mute,
        total_duration=total_duration,
        encoder=req.encoder,
        audio_bitrate=req.audio_bitrate,
        quality=req.quality,
        range_start=req.range_start,
        range_end=req.range_end,
        progress=_progress_adapter,
        force=req.force,
    )

    output_size = output_path.stat().st_size
    progress.emit("done", 100.0, f"rendered: {output_path.name} ({output_size} bytes)")

    return {
        "input_path": str(input_path),
        "output_path": str(output_path),
        "output_size": output_size,
        "removed_count": len(req.segments_to_remove),
        "muted_count": len(req.segments_to_mute),
        "kept_duration": kept_duration,
    }


async def run_render_job(store: JobStore, state: JobState, req: RenderRequest) -> None:
    progress = Progress(store, state)
    state.status = JobStatus.RUNNING
    progress.emit("init", 0.0, "starting render")
    try:
        result = await asyncio.to_thread(_run_render_sync, req, progress)
        _finalize(store, state, JobStatus.DONE, result=result)
    except asyncio.CancelledError:
        _finalize(store, state, JobStatus.CANCELLED, error="cancelled by user")
        raise
    except Exception as exc:  # noqa: BLE001
        logger.exception("render job failed")
        _finalize(store, state, JobStatus.FAILED, error=f"{type(exc).__name__}: {exc}")


# --- download_vod job --------------------------------------------------------


def _run_download_vod_sync(
    req: DownloadVodRequest, video_id: str, output: Path, progress: Progress
) -> dict[str, Any]:
    """Скачать VOD через TwitchDownloaderCLI, стримя прогресс в WS.

    Стадии CLI (download → verify → combine) маппим в единый 0-100 progress:
    сам процент — это процент текущей стадии, стадию несём в `stage`. Точного
    «сколько всего осталось» CLI не даёт, поэтому показываем процент стадии
    и её имя — этого хватает, чтобы прогресс-бар двигался осмысленно.
    """
    from ..tdcli import download_vod, get_tdcli_path

    _STAGE_LABEL = {
        "download": "Скачиваю сегменты",
        "verify": "Проверяю сегменты",
        "combine": "Собираю видео",
        "info": "Получаю данные",
    }

    def _on_progress(stage: str, percent: float | None) -> None:
        # combine — финальная стадия: показываем 95-100, чтобы бар не «прыгал»
        # обратно с 100% download на 0% combine. Остальные — как есть (0-95).
        label = _STAGE_LABEL.get(stage, stage)
        if stage == "combine":
            pct = 95.0 + (percent or 0.0) * 0.05
        else:
            pct = min(95.0, (percent or 0.0) * 0.95)
        progress.emit(stage, pct, label)

    # Гарантируем наличие бинарника ДО старта скачивания. На свежем клоне/первом
    # запуске это скачает TwitchDownloaderCLI (~50MB) — показываем это отдельной
    # стадией, чтобы прогресс-бар не «висел» молча перед стартом CLI.
    progress.emit("prepare", 0.0, "Подготавливаю загрузчик…")
    get_tdcli_path()

    progress.emit("download", 0.0, "Запускаю TwitchDownloaderCLI")
    download_vod(
        video_id=video_id,
        output=output,
        quality=req.quality,
        beginning=req.range_in or None,
        ending=req.range_out or None,
        oauth=req.oauth or None,
        temp_path=output.parent / "_tdcli_temp",
        on_progress=_on_progress,
    )
    progress.emit("done", 100.0, f"Готово: {output.name}")
    return {
        "video_id": video_id,
        "output": str(output),
        "stream": str(output),
        "output_size": output.stat().st_size if output.exists() else 0,
    }


async def run_download_vod_job(
    store: JobStore, state: JobState, req: DownloadVodRequest, video_id: str, output: Path
) -> None:
    progress = Progress(store, state)
    state.status = JobStatus.RUNNING
    progress.emit("init", 0.0, "starting download-vod")
    try:
        result = await asyncio.to_thread(
            _run_download_vod_sync, req, video_id, output, progress
        )
        # Скачанный файл теперь можно смотреть/обрабатывать — в whitelist для
        # /media (превью) и как готовый stream.mp4 для последующего process-job'а.
        store.allow_media(result["output"])
        _finalize(store, state, JobStatus.DONE, result=result)
    except asyncio.CancelledError:
        _finalize(store, state, JobStatus.CANCELLED, error="cancelled by user")
        raise
    except Exception as exc:  # noqa: BLE001
        logger.exception("download_vod job failed")
        _finalize(store, state, JobStatus.FAILED, error=f"{type(exc).__name__}: {exc}")


# --- detect_offtopic job -----------------------------------------------------


def _run_detect_offtopic_sync(
    req: DetectOfftopicRequest, progress: Progress
) -> dict[str, Any]:
    """AI-классификация оффтопа: транскрипт → decisions['offtopic'].

    Зеркалит _run_highlights_import_sync: читаем decisions + транскрипт,
    прогоняем через классификатор, мержим в свой ключ, пишем output.

    Путь выбирает req.method (см. _resolve_offtopic_method): по исходнику
    (нужен source_audio/source_transcript), один проход или старый оконный.

    Секрет (req.api_key) используется только для конструирования LLMClient и
    НИКОГДА не попадает в progress-сообщения, логи или result.
    """
    # Ленивый импорт: SDK anthropic/openai опциональны ([ai] extra). Так
    # runner импортируется без них — падаем только при реальном вызове LLM.
    from ..llm_offtopic import (
        AIProvider,
        LLMClient,
        PromptTemplate,
        builtin_provider,
        classify_single_pass,
        classify_transcript,
    )
    from ..llm_offtopic import merge_into_decisions as merge_offtopic

    if req.source_audio and req.source_transcript:
        raise ValueError("pass either source_audio or source_transcript, not both")

    progress.emit("read", 8.0, "read decisions + transcript")
    decisions_doc = read_json(Path(req.decisions))
    transcript_path = _resolve_transcript_path(decisions_doc, req.transcript)
    transcript_doc = read_json(transcript_path)
    transcript_hash = file_content_hash(transcript_path)

    # Провайдер: встроенный пресет по id, иначе — кастомный dict из запроса.
    provider = builtin_provider(req.provider_id)
    if provider is None:
        if not req.provider:
            raise ValueError(
                f"unknown provider_id={req.provider_id!r} and no custom `provider` supplied"
            )
        provider = AIProvider.model_validate(req.provider)

    topic_tpl = (
        PromptTemplate.model_validate(req.topic_template) if req.topic_template else None
    )
    class_tpl = (
        PromptTemplate.model_validate(req.classification_template)
        if req.classification_template
        else None
    )

    client = LLMClient(provider, api_key=req.api_key)

    have_source = bool(req.source_audio or req.source_transcript)
    method = _resolve_offtopic_method(req.method, have_source)

    if method == "source":
        # Кастомные шаблоны из запроса — под ОКОННЫЙ путь (topic/classification);
        # у пути по исходнику другие виды шаблонов, поэтому берём встроенные.
        result = _classify_source_path(
            req,
            progress,
            client=client,
            transcript_doc=transcript_doc,
            transcript_hash=transcript_hash,
        )
    elif method == "single-pass":
        progress.emit("classify", 15.0, f"classify offtopic (single-pass) via {provider.id}")
        result = classify_single_pass(
            transcript_doc,
            client=client,
            model=req.model,
            transcript_hash=transcript_hash,
        )
    else:  # windows — старый оконный путь
        # Тюнинг: пробрасываем только non-None, чтобы классификатор держал дефолты.
        tuning: dict[str, Any] = {}
        for name in ("window_s", "overlap_s", "chunk_s", "merge_gap_s", "cascade_threshold"):
            val = getattr(req, name)
            if val is not None:
                tuning[name] = val
        progress.emit("classify", 15.0, f"classify offtopic via {provider.id}")
        result = classify_transcript(
            transcript_doc,
            client=client,
            topic_template=topic_tpl,
            classification_template=class_tpl,
            cheap_model=req.cheap_model,
            expensive_model=req.expensive_model,
            transcript_hash=transcript_hash,
            **tuning,
        )

    progress.emit("merge", 90.0, "merge into decisions.json")
    merged = merge_offtopic(decisions_doc, result)

    progress.emit("write", 95.0, "write output")
    write_json(Path(req.output), merged)
    # Оффтоп посчитала МАШИНА и записала мимо PUT /decisions — обновляем базовый
    # снимок журнала, иначе следующее сохранение выглядело бы как «человек
    # добавил N сегментов оффтопа».
    _refresh_edit_journal(
        req.output,
        merged,
        ("offtopic",),
        note=f"AI-детект оффтопа ({result.method}) — сегменты посчитала программа",
    )
    progress.emit(
        "write",
        100.0,
        f"{len(result.segments)} offtopic segments ({result.total_offtopic_s:.1f}s)",
    )

    # Отдаём обогащённые сегменты (со stream_start_s/stream_end_s из merge),
    # чтобы будущая UI-панель рендерила без повторного чтения decisions.json.
    # api_key СОЗНАТЕЛЬНО отсутствует в result — секрет не покидает процесс.
    return {
        "output": req.output,
        "method": result.method,
        "segments_count": len(result.segments),
        "total_offtopic_s": result.total_offtopic_s,
        "offtopic": merged["offtopic"]["segments"],
    }


_VALID_OFFTOPIC_METHODS = ("source", "single-pass", "windows")


def _resolve_offtopic_method(requested: str | None, have_source: bool) -> str:
    """Выбрать путь классификации.

    None → авто: есть исходник → по исходнику, иначе → оконный (не меняем
    поведение старых вызовов без исходника). Явный метод честно валидируем.
    """
    if requested is None:
        return "source" if have_source else "windows"
    if requested not in _VALID_OFFTOPIC_METHODS:
        raise ValueError(
            f"unknown method={requested!r}; expected one of {list(_VALID_OFFTOPIC_METHODS)}"
        )
    if requested == "source" and not have_source:
        raise ValueError("method=source requires source_audio or source_transcript")
    return requested


def _classify_source_path(
    req: DetectOfftopicRequest,
    progress: Progress,
    *,
    client: Any,
    transcript_doc: dict[str, Any],
    transcript_hash: str,
) -> Any:
    """Путь по исходнику: получить транскрипт исходника и классифицировать.

    Транскрипция исходника — единственная долгая стадия здесь (минуты GPU/CPU),
    поэтому прогресс GigaAM пробрасываем в джоб-progress, иначе UI выглядит
    зависшим. Готовый source_transcript эту стадию пропускает.
    """
    from ..llm_offtopic import classify_with_source
    from ..llm_offtopic.source import DEFAULT_SOURCE_MODEL, SourceError, transcribe_source

    if req.source_transcript:
        progress.emit("read_source", 12.0, "read source transcript")
        src_path = Path(req.source_transcript)
        if not src_path.exists():
            raise ValueError(f"source transcript not found: {src_path}")
        source_doc = read_json(src_path)
        source_hash: str | None = file_content_hash(src_path)
        # Чем распознан готовый транскрипт — не знаем; сохраняем, что передали.
        source_model = req.source_model
    else:
        src_model = req.source_model or DEFAULT_SOURCE_MODEL
        src_device = req.source_device or "cpu"
        workdir = Path(req.decisions).resolve().parent

        def _tick(fraction: float, message: str = "") -> None:
            # 10..70% джобы — на распознавание исходника.
            progress.emit(
                "transcribe_source",
                10.0 + max(0.0, min(1.0, fraction)) * 60.0,
                message or "распознаю исходник",
            )

        try:
            src = transcribe_source(
                req.source_audio,
                workdir,
                model_name=src_model,
                device=src_device,
                on_progress=_tick,
            )
        except SourceError as exc:
            # Внятная причина (ссылка вместо файла, пустой ASR) — наружу как ValueError.
            raise ValueError(f"source transcription failed: {exc}") from exc
        source_doc = src.transcript
        source_hash = src.key
        source_model = src_model

    progress.emit("classify", 72.0, f"classify offtopic (source) via {client.provider.id}")
    return classify_with_source(
        transcript_doc,
        source_doc,
        client=client,
        model=req.model,
        transcript_hash=transcript_hash,
        source_transcript_hash=source_hash,
        source_model=source_model,
    )


async def run_detect_offtopic_job(
    store: JobStore, state: JobState, req: DetectOfftopicRequest
) -> None:
    from ..logging_setup import job_file_log

    progress = Progress(store, state)
    state.status = JobStatus.RUNNING
    progress.emit("init", 0.0, "starting detect-offtopic")
    # Лог джобы лежит рядом с её decisions.json. Классификатор пишет туда тему
    # каждого чанка и каждое окно, оставшееся без вердикта; без файла эти записи
    # в серверном процессе просто исчезали.
    workdir = Path(req.decisions).resolve().parent
    with job_file_log(workdir, name=f"offtopic_{state.id}") as log_path:
        if log_path is not None:
            logger.info("detect_offtopic job %s → log %s", state.id, log_path)
        try:
            result = await asyncio.to_thread(_run_detect_offtopic_sync, req, progress)
            if log_path is not None:
                # Путь в result — чтобы UI мог открыть лог прогона, а не искать его.
                result["log"] = str(log_path)
            _finalize(store, state, JobStatus.DONE, result=result)
        except asyncio.CancelledError:
            _finalize(store, state, JobStatus.CANCELLED, error="cancelled by user")
            raise
        except Exception as exc:  # noqa: BLE001
            # Логируем тип+сообщение исключения. Секрет в них не попадает: LLMClient
            # не кладёт ключ в текст ошибок (см. llm_offtopic.client).
            logger.exception("detect_offtopic job failed")
            _finalize(store, state, JobStatus.FAILED, error=f"{type(exc).__name__}: {exc}")
