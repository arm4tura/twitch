"""Единое ядро пайплайна Фазы 1: stream → transcript → мат → decisions → Vegas.

Раньше эта логика жила в ДВУХ местах — `cli.py:process` и
`server/runner.py:_run_process_sync` — и успела разойтись: CLI не знал про
движок `gigaam` (дефолтный!) и падал с BadParameter на нём, а server уже умел.
Теперь оба вызывают `run_transcription_pipeline`, а расхождение движков
невозможно по построению.

Разделение обязанностей:
  - Вызывающий (CLI или runner) строит `PipelineConfig` из своих флагов
    (`--raw-mute`, `--disable-asr-hardening`, device из окружения…), резолвит
    пути и передаёт сюда. Числа мьюта он НЕ выдумывает: передаёт имя уровня
    чувствительности, а значения выводит сам конфиг (config.SENSITIVITY_LEVELS)
    — иначе те же 80/120/600 снова разъедутся между CLI, API и UI.
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
from .keyword_spotting import MAX_GAP_FRAMES
from .posteriors import posteriors_path, read_posteriors
from .profanity import (
    SPOT_KEYWORDS,
    ProfanityMatch,
    RussianNormalizer,
    decision_lists,
    detect_posterior_profanity,
    detect_profanity,
    load_banwords,
)
from .timecode import TimeSpan, parse_timecode
from .transcription import load_mock_transcript, transcribe_audio
from .vegas_export import write_vegas_script

logger = logging.getLogger(__name__)

# Прогресс-callback: (stage, percent, message). Прогресс coarse-grained —
# extract(2→10) → transcribe(12→75) → detect(78→85) → build(88→95) → write(97→100).
ProgressFn = Callable[[str, float, str], None]

# Границы стадии transcribe на общей шкале. Вынесены в константы, потому что
# используются в двух местах: адаптер дробного прогресса GigaAM растягивает
# сюда свои тики, а run_transcription_pipeline добивает хвост до END после
# возврата. Разъедься эти числа — прогресс-бар прыгнул бы назад.
TRANSCRIBE_START_PCT = 12.0
TRANSCRIBE_END_PCT = 75.0

# «Вхожу в стадию» — ДО загрузки весов модели и VAD-сегментации.
#
# Почему не сразу TRANSCRIBE_START_PCT: UI ставит якорь ETA на первом тике с
# progress >= 12 и от него меряет скорость. Между входом в стадию и первым
# реальным батчем проходит загрузка весов GigaAM (1–2 GB) + VAD — до минуты и
# больше на холодном старте. Заякорись UI на входе, этот простой попал бы в
# замер скорости, и первая оценка вышла бы завышенной в разы (а монотонный
# clamp вниз потом долго сползал бы к правде).
#
# Поэтому вход репортим НИЖЕ порога якоря: 11% (> 10% конца extract_audio, так
# что монотонность шкалы сохранена). Якорем становится первый тик батч-цикла,
# который приходит уже ровно на 12% — см. _gigaam_progress. Двигать это число
# в 12+ нельзя, не поправив порог в JobScreen.
TRANSCRIBE_ENTER_PCT = 11.0


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


def _spotting_cache_path(workdir: Path, key: str) -> Path:
    """Кэш стадии спотинга — отдельный от detection_*.json файл.

    Отдельный, потому что ключ у него другой: спотинг не зависит от банвордов,
    зато зависит от порога и списка корней. Общий кэш заставлял бы пересчитывать
    текстовый детектор при каждой правке порога — и наоборот."""
    return workdir / "cache" / f"spotting_{key}.json"


def _write_detection_cache(
    path: Path, key: str, matches: list[ProfanityMatch], stage: str = "profanity_detection"
) -> None:
    write_json(
        path,
        {
            "stage": stage,
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
    keep_posteriors: bool,
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

        # Вход в стадию репортим на 11%, а не на 12: см. TRANSCRIBE_ENTER_PCT.
        # Загрузка весов (1–2 GB) и VAD не должны попасть в замер скорости ETA.
        progress("transcribe", TRANSCRIBE_ENTER_PCT, f"gigaam model={gigaam_model}")

        # GigaAM тикает дробным прогрессом по мере обхода VAD-сегментов —
        # растягиваем его на диапазон стадии (12→75%), чтобы UI мог считать ETA.
        # Хвост до 75 добивает сам run_transcription_pipeline после возврата.
        def _gigaam_progress(fraction: float, message: str = "") -> None:
            progress(
                "transcribe",
                TRANSCRIBE_START_PCT
                + max(0.0, min(1.0, fraction)) * (TRANSCRIBE_END_PCT - TRANSCRIBE_START_PCT),
                message,
            )

        return transcribe_with_gigaam(
            audio_path=audio_path,
            workdir=workdir,
            model_name=gigaam_model,
            device=config.device,
            force=force_transcribe,
            on_progress=_gigaam_progress,
            # Посториоры нужны стадии спотинга. Просим их ДО распознавания:
            # log-probs живут один миг внутри декода, второй раз без прогона
            # модели их не получить (см. posteriors.py).
            keep_posteriors=keep_posteriors,
            # Поиск речи (VAD) на длинном стриме идёт минутами — показываем,
            # что процесс жив, но НИЖЕ порога якоря ETA, как и вход в стадию.
            on_status=lambda message: progress("transcribe", TRANSCRIBE_ENTER_PCT, message),
        )

    if config.transcriber == "whisperx":
        progress("transcribe", TRANSCRIBE_START_PCT, f"whisperx model={config.model}")
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
        progress("transcribe", TRANSCRIBE_START_PCT, "whispercpp")
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


def _spot_stage(
    *,
    config: PipelineConfig,
    workdir: Path,
    transcript: dict[str, Any],
    transcript_key: str,
    detection_key: str,
    matches: list[ProfanityMatch],
    range_in: TimeSpan,
    force: bool,
    progress: ProgressFn,
) -> tuple[list[ProfanityMatch], Optional[str], Optional[Path]]:
    """Стадия «мат по посториорам»: дополняет текстовые матчи кандидатами.

    Возвращает (объединённые матчи, ключ кэша, путь кэша). Посториоров нет —
    возвращаем текстовые матчи как есть: это не ошибка прогона, а «стадии не
    было» (старый кэш транскрипта, whisperx, mock-транскрипт).

    Кэш свой, отдельный от detection_*.json: спотинг не зависит от банвордов,
    зато зависит от порога, списка корней и от того, что нашёл текстовый
    детектор (его находки глушат кандидатов-дубликатов) — поэтому в ключе и
    detection_key. Смысл кэша тот же, что у посториоров: правка порога должна
    стоить секунды DP, а не повторное распознавание.
    """
    npz_path = posteriors_path(workdir, transcript_key)
    key = stable_hash(
        {
            "stage": "posterior_spotting",
            "transcript_key": transcript_key,
            "detection_key": detection_key,
            "keywords": list(SPOT_KEYWORDS),
            "min_score": round(float(config.spot_min_score), 4),
            "max_gap_frames": MAX_GAP_FRAMES,
            # Источник границ кандидата: кадры эмиссии или forced_align по
            # подставленной гипотезе. Меняет тайминги, значит обязан менять ключ.
            "bounds": "forced_align",
        }
    )
    cache_path = _spotting_cache_path(workdir, key)
    if cache_path.exists() and not force:
        spotted = _load_detection_cache(cache_path)
    else:
        posteriors = read_posteriors(npz_path)
        if posteriors is None:
            logger.warning(
                "посториоров нет (%s) — поиск мата по посториорам пропущен",
                npz_path.name,
            )
            return matches, None, None
        spotted = detect_posterior_profanity(
            posteriors,
            transcript,
            range_in,
            min_score=config.spot_min_score,
            known=matches,
        )
        _write_detection_cache(cache_path, key, spotted, stage="posterior_spotting")
    progress("spot_posteriors", 87.0, f"{len(spotted)} кандидатов по посториорам")
    # Порядок по времени обязателен: decisions.py склеивает соседние маты в один
    # мьют, идя по списку подряд (см. _group_by_segment).
    merged = sorted(matches + spotted, key=lambda m: m.local_start_ms)
    return merged, key, cache_path


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
    # Спотинг по посториорам умеет только gigaam (это CTC-модель, log-probs есть
    # только у неё), и только на реальном аудио: у mock-транскрипта посториоров
    # нет по определению. Флаг решает и то, просить ли их у ASR, — иначе на диск
    # ложились бы мегабайты, которые никто не прочитает.
    spotting_on = (
        config.spot_posteriors and config.transcriber == "gigaam" and mock_transcript is None
    )

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
            keep_posteriors=spotting_on,
            progress=emit,
        )
        emit(
            "transcribe",
            TRANSCRIBE_END_PCT,
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
            # Словарь корней и белые списки живут в коде, а не в файле банвордов,
            # поэтому их правка иначе не инвалидировала бы кэш: добавили корень
            # «епт» — прогон с готовым detection_*.json остался бы без него, и
            # «ёпта» по-прежнему уезжала бы в ручную правку. Все такие списки
            # собраны в profanity.decision_lists() — новый добавляется там.
            "roots": decision_lists(),
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

    spotting_key = None
    spotting_cache = None
    if spotting_on:
        emit("spot_posteriors", 86.0, "поиск мата по посториорам")
        matches, spotting_key, spotting_cache = _spot_stage(
            config=config,
            workdir=workdir,
            transcript=transcript,
            transcript_key=transcript_key,
            detection_key=detection_key,
            matches=matches,
            range_in=start,
            force=force_detect,
            progress=emit,
        )

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
    if spotting_key is not None:
        decisions_doc["caches"]["spotting"] = str(spotting_cache)
        decisions_doc["caches"]["spotting_key"] = spotting_key
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
