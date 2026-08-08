"""Единое ядро пайплайна Фазы 1: stream → transcript → мат → decisions → Vegas.

Раньше эта логика жила в ДВУХ местах — `cli.py:process` и
`server/runner.py:_run_process_sync` — и успела разойтись: CLI не знал про
движок `gigaam` (дефолтный!) и падал с BadParameter на нём, а server уже умел.
Теперь оба вызывают `run_transcription_pipeline`, а расхождение движков
невозможно по построению.

Разделение обязанностей:
  - Вызывающий (CLI или runner) строит `PipelineConfig` со своими дефолтами и
    флагами (`--raw-mute`, `--disable-asr-hardening`, device из окружения…),
    резолвит пути и передаёт сюда.
  - Ядро делает саму работу и репортит прогресс через callback `progress`
    (CLI подсовывает логгер, runner — Progress.emit для WS). None → тишина.
  - Ядро кидает доменные исключения (ValueError, WhisperCppError). Перевод в
    typer.BadParameter / JobStatus.FAILED — забота вызывающего.
"""

from __future__ import annotations

import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Optional

from .cache import file_content_hash, read_json, stable_hash, write_json
from .config import PipelineConfig
from .decisions import build_decisions, write_decisions
from .ffmpeg_tools import extract_audio_range, probe_media_duration
from .profanity import ProfanityMatch, RussianNormalizer, detect_profanity, load_banwords
from .timecode import TimeSpan, parse_timecode
from .transcription import load_mock_transcript, transcribe_audio
from .vegas_export import write_vegas_script

logger = logging.getLogger(__name__)

# Прогресс-callback: (stage, percent, message). Прогресс coarse-grained —
# extract(2→10) → transcribe(12→75) → detect(78→85) → build(88→95) → write(97→100).
ProgressFn = Callable[[str, float, str], None]


def _noop_progress(stage: str, percent: float, message: str = "") -> None:
    pass


def resolve_range(
    stream: Path, range_in: Optional[str], range_out: Optional[str]
) -> tuple[TimeSpan, TimeSpan]:
    """Границы обработки. range_out по умолчанию — длительность файла."""
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


def _transcribe(
    *,
    config: PipelineConfig,
    audio_path: Path,
    workdir: Path,
    gigaam_model: str,
    whisper_cpp_bin: Optional[Path],
    whisper_cpp_model: Optional[Path],
    whisper_cpp_threads: Optional[int],
    whisper_cpp_extra: Optional[list[str]],
    force_transcribe: bool,
    progress: ProgressFn,
) -> tuple[dict[str, Any], str, Path]:
    """Диспетчер движков. ЕДИНСТВЕННОЕ место, где выбирается транскрайбер —
    поэтому CLI и server физически не могут разойтись по набору движков.

    Тяжёлые движковые модули импортируются лениво: без обращения к gigaam/
    whispercpp здесь не тянется ни torch, ни subprocess-обвязка.
    """
    if config.transcriber == "gigaam":
        # Дефолт: GigaAM v3 (Salute). Точнее по русскому мату, узкие пословные
        # тайминги, не требует CTranslate2/cuDNN.
        from .gigaam_asr import transcribe_with_gigaam

        progress("transcribe", 12.0, f"gigaam model={gigaam_model}")
        return transcribe_with_gigaam(
            audio_path=audio_path,
            workdir=workdir,
            model_name=gigaam_model,
            device=config.device,
            force=force_transcribe,
        )

    if config.transcriber == "whisperx":
        progress("transcribe", 12.0, f"whisperx model={config.model}")
        return transcribe_audio(
            audio_path=audio_path,
            workdir=workdir,
            model_name=config.model,
            language=config.language,
            device=config.device,
            compute_type=config.compute_type,
            batch_size=config.batch_size,
            vad_filter=config.vad_filter,
            vad_method=config.vad_method,
            asr_options=config.asr_options or None,
            force=force_transcribe,
        )

    if config.transcriber == "whispercpp":
        from .whisper_cpp import transcribe_with_whisper_cpp

        if whisper_cpp_model is None:
            raise ValueError(
                "whisper_cpp_model is required when transcriber == 'whispercpp'"
            )
        progress("transcribe", 12.0, "whispercpp")
        return transcribe_with_whisper_cpp(
            audio_path=audio_path,
            workdir=workdir,
            binary=whisper_cpp_bin,
            model_path=whisper_cpp_model,
            language=config.language,
            threads=whisper_cpp_threads or None,
            extra_args=whisper_cpp_extra or [],
            force=force_transcribe,
        )

    # config.validate() уже это ловит; страховка на случай прямого вызова.
    raise ValueError(f"unknown transcriber: {config.transcriber!r}")


def run_transcription_pipeline(
    *,
    stream: Path,
    original: Path,
    banwords: Path,
    workdir: Path,
    decisions_out: Path,
    vegas_out: Path,
    config: PipelineConfig,
    range_in: Optional[str] = None,
    range_out: Optional[str] = None,
    gigaam_model: str = "v3_ctc",
    whisper_cpp_bin: Optional[Path] = None,
    whisper_cpp_model: Optional[Path] = None,
    whisper_cpp_threads: Optional[int] = None,
    whisper_cpp_extra: Optional[list[str]] = None,
    mock_transcript: Optional[Path] = None,
    force_extract: bool = False,
    force_transcribe: bool = False,
    force_detect: bool = False,
    progress: Optional[ProgressFn] = None,
) -> dict[str, Any]:
    """Прогнать полный пайплайн и записать decisions.json + Vegas .cs.

    Ожидает уже провалидированный `config` (вызови config.validate() до) и
    готовые пути. Возвращает сводку: decisions_path / vegas_path / mutes_count /
    cuts_count.
    """
    emit = progress or _noop_progress

    workdir.mkdir(parents=True, exist_ok=True)
    start, end = resolve_range(stream, range_in, range_out)

    audio_path: Optional[Path] = None
    audio_key = "mock"

    if mock_transcript is None:
        emit("extract_audio", 2.0, "ffmpeg -> wav")
        audio_path, audio_key = extract_audio_range(
            stream, start, end, workdir, force=force_extract
        )
        emit("extract_audio", 10.0, f"audio ready: {audio_path.name}")

        transcript, transcript_key, transcript_cache = _transcribe(
            config=config,
            audio_path=audio_path,
            workdir=workdir,
            gigaam_model=gigaam_model,
            whisper_cpp_bin=whisper_cpp_bin,
            whisper_cpp_model=whisper_cpp_model,
            whisper_cpp_threads=whisper_cpp_threads,
            whisper_cpp_extra=whisper_cpp_extra,
            force_transcribe=force_transcribe,
            progress=emit,
        )
        emit(
            "transcribe",
            75.0,
            f"transcript ready ({len(transcript.get('segments', []))} segments)",
        )
    else:
        emit("mock_transcript", 40.0, f"reading {mock_transcript}")
        transcript = load_mock_transcript(mock_transcript)
        transcript_key = stable_hash(
            {
                "stage": "mock_transcript",
                "path": str(mock_transcript.resolve()),
                "content": file_content_hash(mock_transcript),
            }
        )
        transcript_cache = mock_transcript
        emit("mock_transcript", 75.0, "mock transcript loaded")

    detection_key = stable_hash(
        {
            "stage": "profanity_detection",
            "transcript_key": transcript_key,
            "banwords": {
                "path": str(banwords.resolve()),
                "content": file_content_hash(banwords),
            },
            "range_in_ms": start.ms,
            "range_out_ms": end.ms,
            "normalization": "lowercase+yo+pymorphy3",
        }
    )
    detection_cache = _detection_cache_path(workdir, detection_key)

    emit("detect_profanity", 78.0, "load banwords + detect")
    if detection_cache.exists() and not force_detect:
        matches = _load_detection_cache(detection_cache)
    else:
        normalizer = RussianNormalizer()
        entries = load_banwords(banwords, normalizer=normalizer)
        matches = detect_profanity(transcript, entries, start, normalizer=normalizer)
        _write_detection_cache(detection_cache, detection_key, matches)
    emit("detect_profanity", 85.0, f"{len(matches)} matches")

    emit("build_decisions", 88.0, "compose decisions.json")
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
    emit("build_decisions", 95.0, f"{len(decisions_doc.get('mutes', []))} mutes")

    emit("write_outputs", 97.0, "write decisions + vegas")
    write_decisions(decisions_out, decisions_doc)
    write_vegas_script(vegas_out, decisions_doc)
    emit("write_outputs", 100.0, "done")

    return {
        "decisions_path": str(decisions_out),
        "vegas_path": str(vegas_out),
        "mutes_count": len(decisions_doc.get("mutes", [])),
        "cuts_count": len(decisions_doc.get("cuts", [])),
    }
