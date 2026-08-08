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
        mute_padding_before_ms=req.mute_padding_before_ms,
        mute_padding_after_ms=req.mute_padding_after_ms,
        mute_extend_mode=req.mute_extend_mode,
        mute_max_seconds=req.mute_max_seconds,
        mute_join_gap_ms=req.mute_join_gap_ms,
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


async def run_process_job(store: JobStore, state: JobState, req: ProcessJobRequest) -> None:
    progress = Progress(store, state)
    state.status = JobStatus.RUNNING
    progress.emit("init", 0.0, "starting process job")
    try:
        result = await asyncio.to_thread(_run_process_sync, req, progress)
        # Регистрируем в реестре Recent Projects — best-effort, ошибку глотает
        # сама register_project. Делаем это ПОСЛЕ успешного write_decisions,
        # чтобы битые/незавершённые пути в Dashboard не всплывали. Берём путь
        # из результата (req.decisions мог быть None — простой режим).
        register_project(result["decisions_path"])
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
