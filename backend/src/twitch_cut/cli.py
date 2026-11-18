from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

# ВАЖНО: configure_hf_cache() ДОЛЖЕН вызваться до импорта .transcription
# (который тянет whisperx → torch → huggingface_hub). HuggingFace-либы читают
# HF_HOME один раз при импорте — поменять его позже не получится.
from .config import (
    DEFAULT_ASR_OPTIONS,
    SENSITIVITY_DEFAULT,
    PipelineConfig,
    configure_hf_cache,
)

configure_hf_cache()

from .cache import file_content_hash, read_json, write_json  # noqa: E402
from .audit import AuditError, analyze as analyze_audit, write_report  # noqa: E402
from .evaluate import (  # noqa: E402
    evaluate as evaluate_edits,
    read_metrics,
    render_metrics,
    write_metrics,
)
from .llm.exporter import build_notebooklm_package  # noqa: E402
from .llm.importer import MergeError, merge_into_decisions, parse_text  # noqa: E402
from .logging_setup import setup_logging, setup_server_logging  # noqa: E402
from .pipeline import run_transcription_pipeline  # noqa: E402
from .silence import add_padding, detect_silence_ffmpeg, merge_close_segments  # noqa: E402
from .vegas_export import VegasExportError, exportable_mutes, write_vegas_script  # noqa: E402
from .whisper_cpp import WhisperCppError  # noqa: E402

app = typer.Typer(
    help="CLI-ядро Фазы 1: stream → ASR → мат → decisions.json → Vegas .cs",
    no_args_is_help=True,
)
console = Console()
logger = logging.getLogger(__name__)

@app.callback()
def main() -> None:
    """Phase 1 command group."""


@app.command()
def process(
    stream: Path = typer.Option(..., exists=True, file_okay=True, dir_okay=False, help="Исходный stream.mp4"),
    original: Optional[Path] = typer.Option(None, exists=True, file_okay=True, dir_okay=False, help="Оригинальное видео реакции; если не указано — сам stream"),
    range_in: Optional[str] = typer.Option(None, "--range-in", help="Начало обработки в исходном stream.mp4; если не указано, 00:00:00"),
    range_out: Optional[str] = typer.Option(None, "--range-out", help="Конец обработки в исходном stream.mp4; если не указано, конец видео"),
    banwords: Optional[Path] = typer.Option(None, exists=True, file_okay=True, dir_okay=False, help="Словарь мата/банвордов; если не указано — встроенный словарь"),
    workdir: Optional[Path] = typer.Option(None, help="Рабочая папка cache/checkpoints; если не указано — уникальная папка в work/"),
    decisions: Optional[Path] = typer.Option(None, help="Куда записать decisions.json; если не указано — в workdir"),
    vegas: Optional[Path] = typer.Option(None, help="Куда записать Vegas C# script; если не указано — в workdir"),
    model: str = typer.Option("large-v3", help="WhisperX/faster-whisper model"),
    language: str = typer.Option("ru", help="Язык транскрипции"),
    device: Optional[str] = typer.Option(None, help="cuda или cpu; если не указано — авто-детект (CPU без NVIDIA)"),
    compute_type: Optional[str] = typer.Option(None, help="float16, float32 или int8; если не указано — авто (int8 на CPU, float16 на GPU)"),
    batch_size: int = typer.Option(16, min=1, help="WhisperX batch size"),
    vad_filter: bool = typer.Option(True, help="Включить VAD внутри faster-whisper/WhisperX"),
    vad_method: str = typer.Option(
        "pyannote",
        help=(
            "VAD backend для WhisperX: 'pyannote' (default, стабильнее для длинных стримов) "
            "или 'silero' (агрессивнее режет, коррелирует с hallucination-drift Whisper)"
        ),
    ),
    disable_asr_hardening: bool = typer.Option(
        False,
        "--disable-asr-hardening",
        help=(
            "Отключить anti-hallucination профиль faster-whisper "
            "(condition_on_previous_text=False, temperature fallback и т.д.). "
            "По умолчанию включён — защищает от потери 5+ секунд речи на длинных стримах."
        ),
    ),
    transcriber: str = typer.Option(
        "gigaam",
        "--transcriber",
        help=(
            "Backend транскрипции: 'gigaam' (default, GigaAM v3 — точнее по русскому мату, "
            "не тянет CTranslate2/cuDNN), 'whisperx' (PyTorch+CUDA) или "
            "'whispercpp' (subprocess whisper-cli.exe)"
        ),
    ),
    gigaam_model: str = typer.Option(
        "v3_ctc",
        "--gigaam-model",
        help="Модель GigaAM при --transcriber gigaam (например 'v3_ctc').",
    ),
    whisper_cpp_bin: Optional[Path] = typer.Option(
        None,
        "--whisper-cpp-bin",
        help="Путь к whisper-cli.exe (release с github.com/ggerganov/whisper.cpp). Если не указан, ищется в PATH.",
    ),
    whisper_cpp_model: Optional[Path] = typer.Option(
        None,
        "--whisper-cpp-model",
        help="Путь к ggml-large-v3.bin (huggingface.co/ggerganov/whisper.cpp).",
    ),
    whisper_cpp_threads: int = typer.Option(
        0,
        "--whisper-cpp-threads",
        min=0,
        help="Число CPU-потоков для whisper.cpp (0 = по умолчанию).",
    ),
    whisper_cpp_extra: Optional[str] = typer.Option(
        None,
        "--whisper-cpp-extra",
        help="Доп. флаги для whisper-cli.exe через пробел, например '--no-fallback --best-of 5'.",
    ),
    sensitivity: str = typer.Option(
        SENSITIVITY_DEFAULT,
        "--sensitivity",
        help=(
            "Чувствительность детекции: low | balanced | high. Одна ручка вместо "
            "восьми чисел (padding, склейка, поджим по RMS, расширение окна на "
            "тишине, допуск голове зайти на соседа, порог спотинга): low — "
            "не глушить лишнего, high — не пропустить мат. Отдельные --mute-*/"
            "--spot-min-score перебивают уровень."
        ),
    ),
    mute_padding_before_ms: Optional[int] = typer.Option(
        None, min=0, help="Padding перед словом (мс). По умолчанию — из --sensitivity."
    ),
    mute_padding_after_ms: Optional[int] = typer.Option(
        None, min=0, help="Padding после слова (мс). По умолчанию — из --sensitivity."
    ),
    mute_extend_mode: str = typer.Option(
        "word",
        "--mute-extend-mode",
        help="Как расширять mute: 'word' (точно по слову, default) или 'segment-tail' (до конца фразы — только если нет word-level таймингов)",
    ),
    mute_max_seconds: float = typer.Option(
        6.0,
        "--mute-max-seconds",
        min=0.1,
        help="Жёсткий потолок длительности одного mute (защита от длинных сегментов)",
    ),
    mute_join_gap_ms: Optional[int] = typer.Option(
        None,
        "--mute-join-gap-ms",
        min=0,
        help="Макс. разрыв (мс) между соседними матами для склейки в один mute. Больше — два отдельных mute, чистая речь между ними не мьютится. По умолчанию — из --sensitivity.",
    ),
    raw_mute: bool = typer.Option(
        False,
        "--raw-mute",
        help="Диагностический режим: мьютить СТРОГО по таймингам whisper для каждого банворда (без padding, без склейки, без extend, без cap). Покажет, что выдаёт whisper без нашей обработки.",
    ),
    spot_posteriors: bool = typer.Option(
        True,
        "--spot-posteriors/--no-spot-posteriors",
        help="Искать мат ещё и в посториорах CTC — там, где ASR услышал не то («епанисир» вместо «ебанский»). Только gigaam; находки идут как review-кандидаты.",
    ),
    spot_min_score: Optional[float] = typer.Option(
        None,
        "--spot-min-score",
        min=0.01,
        max=1.0,
        help="Порог спотинга: средняя вероятность буквы корня. По умолчанию — из --sensitivity (замерен на реальном прогоне); ниже — больше ложных кандидатов, выше — теряются находки.",
    ),
    force_extract: bool = typer.Option(False, help="Пересоздать WAV extraction"),
    force_transcribe: bool = typer.Option(False, help="Пересоздать transcript cache"),
    force_detect: bool = typer.Option(False, help="Пересоздать cache детекции мата"),
    auto_silence: bool = typer.Option(
        False,
        "--auto-silence",
        help=(
            "После детекта мата прогнать ffmpeg silencedetect по извлечённому "
            "аудио и записать найденную тишину в decisions.json (status=accepted). "
            "Аналог кнопки «Найти тишину» в UI — экономит ручной проход по таймлайну."
        ),
    ),
    silence_threshold_db: float = typer.Option(
        -30.0,
        "--silence-threshold-db",
        help="Порог тишины в dBFS (всё ниже — тишина). Только при --auto-silence.",
    ),
    silence_min_duration: float = typer.Option(
        0.5,
        "--silence-min-duration",
        min=0.1,
        help="Мин. длительность сегмента тишины (сек). Только при --auto-silence.",
    ),
    silence_merge_gap: float = typer.Option(
        0.5,
        "--silence-merge-gap",
        min=0.0,
        help="Склеивать сегменты тишины с зазором ≤ N сек. Только при --auto-silence.",
    ),
    silence_padding: float = typer.Option(
        0.1,
        "--silence-padding",
        min=0.0,
        help="Padding до/после каждого сегмента тишины (сек). Только при --auto-silence.",
    ),
    mock_transcript: Optional[Path] = typer.Option(None, exists=True, file_okay=True, dir_okay=False, help="JSON transcript для smoke-теста без WhisperX"),
    verbose: bool = typer.Option(False, help="Подробный лог"),
) -> None:
    """Run Phase 1 pipeline and generate editable decisions + Vegas audio mutes."""

    # Резолвим дефолты «простого режима» тем же кодом, что и HTTP-runner
    # (server/paths.py), чтобы CLI и UI не расходились: workdir → уникальная
    # папка в work/, original → сам stream, banwords → встроенный словарь,
    # device/compute_type → авто-детект (CPU без NVIDIA).
    from .server.paths import resolve_job_defaults

    d = resolve_job_defaults(
        stream=stream,
        original=original,
        banwords=banwords,
        workdir=workdir,
        decisions=decisions,
        vegas=vegas,
        device=device,
        compute_type=compute_type,
    )
    workdir = d.workdir
    original = d.original
    banwords = d.banwords
    decisions = d.decisions
    vegas = d.vegas
    device = d.device
    compute_type = d.compute_type

    workdir.mkdir(parents=True, exist_ok=True)
    setup_logging(workdir, verbose=verbose)

    # whispercpp требует явный путь к модели — ловим до старта пайплайна,
    # чтобы дать понятный param_hint вместо падения в середине.
    if transcriber == "whispercpp" and whisper_cpp_model is None:
        raise typer.BadParameter(
            "--whisper-cpp-model обязателен при --transcriber whispercpp",
            param_hint="--whisper-cpp-model",
        )

    # Anti-hallucination профиль faster-whisper (см. config.DEFAULT_ASR_OPTIONS).
    # По умолчанию включён — без него Whisper на длинных стримах галлюцинирует
    # и сжимает тайминги на 5+ секунд, теряя целые фразы (regression fix для
    # detection_2b4b1a5c89ffaa3a). Флаг --disable-asr-hardening возвращает
    # старое поведение для сравнения/отладки. Для gigaam/whispercpp профиль
    # не применяется (это опции faster-whisper), но хранить его в config
    # безвредно.
    asr_options: dict[str, object] = (
        {} if disable_asr_hardening else dict(DEFAULT_ASR_OPTIONS)
    )

    config = PipelineConfig(
        transcriber=transcriber,
        language=language,
        model=model,
        device=device,
        compute_type=compute_type,
        batch_size=batch_size,
        vad_filter=vad_filter,
        vad_method=vad_method,
        asr_options=asr_options,
        sensitivity=sensitivity,
        mute_padding_before_ms=mute_padding_before_ms,
        mute_padding_after_ms=mute_padding_after_ms,
        mute_extend_mode=mute_extend_mode,
        mute_max_seconds=mute_max_seconds,
        mute_join_gap_ms=mute_join_gap_ms,
        raw_mute=raw_mute,
        spot_posteriors=spot_posteriors,
        spot_min_score=spot_min_score,
    )
    # config.validate() кидает ValueError на неизвестном движке/параметре —
    # переводим в typer.BadParameter для человекочитаемого CLI-вывода.
    # normalized() до этого мягко чинит mps/cuda под платформу и движок.
    config = config.normalized()
    try:
        config.validate()
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

    console.rule("[bold cyan]Фаза 1: stream -> Vegas audio mutes")
    console.print(f"Движок: [bold]{transcriber}[/bold]")

    extra_args = whisper_cpp_extra.split() if whisper_cpp_extra else None

    try:
        result = run_transcription_pipeline(
            stream=stream,
            original=original,
            banwords=banwords,
            workdir=workdir,
            decisions_out=decisions,
            vegas_out=vegas,
            config=config,
            range_in=range_in,
            range_out=range_out,
            gigaam_model=gigaam_model,
            whisper_cpp_bin=whisper_cpp_bin,
            whisper_cpp_model=whisper_cpp_model,
            whisper_cpp_threads=whisper_cpp_threads or None,
            whisper_cpp_extra=extra_args,
            mock_transcript=mock_transcript,
            force_extract=force_extract,
            force_transcribe=force_transcribe,
            force_detect=force_detect,
            progress=lambda stage, pct, msg="": logger.info("[%s %.0f%%] %s", stage, pct, msg),
        )
    except WhisperCppError as exc:
        raise typer.BadParameter(str(exc), param_hint="--transcriber") from exc
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

    if auto_silence:
        _run_auto_silence(
            decisions_path=Path(result["decisions_path"]),
            threshold_db=silence_threshold_db,
            min_duration=silence_min_duration,
            merge_gap=silence_merge_gap,
            padding=silence_padding,
        )

    console.print("\n[green]Готово.[/green]")
    console.print(f"decisions.json: [bold]{result['decisions_path']}[/bold]")
    console.print(f"Vegas script:    [bold]{result['vegas_path']}[/bold]")
    console.print(f"Mute markers:    [bold]{result['mutes_count']}[/bold]")


def _run_auto_silence(
    decisions_path: Path,
    threshold_db: float,
    min_duration: float,
    merge_gap: float,
    padding: float,
) -> None:
    """Детект тишины после основного пайплайна и запись в decisions.json.

    Мирроит серверный `_run_detect_silence_sync` и фронтовый контракт
    SilenceRecord: {id, start, end, status}. Аудио берётся из уже извлечённого
    кэша (`caches.audio` в decisions.json) — те же audio-секунды, что и waveform,
    поэтому конвертации координат не нужно. Все найденные сегменты пишутся со
    status="accepted" (по умолчанию вырезаются; ревьюер снимет лишнее в «Правке»).
    """
    doc = read_json(decisions_path)
    audio_ref = (doc.get("caches") or {}).get("audio")
    if not audio_ref:
        console.print(
            _fmt_warn(
                "auto-silence пропущен: в decisions.json нет caches.audio "
                "(извлечённого аудио)."
            )
        )
        return

    audio_path = Path(audio_ref)
    if not audio_path.exists():
        console.print(
            _fmt_warn(f"auto-silence пропущен: аудио не найдено — {audio_path}")
        )
        return

    segments = detect_silence_ffmpeg(
        audio_path,
        threshold_db=threshold_db,
        min_duration=min_duration,
    )
    if segments:
        segments = merge_close_segments(segments, gap_threshold=merge_gap)
        segments = add_padding(segments, padding_before=padding, padding_after=padding)

    doc["silences"] = [
        {
            "id": f"silence_auto_{round(s * 1000)}",
            "start": s,
            "end": e,
            "status": "accepted",
        }
        for s, e in segments
    ]
    write_json(decisions_path, doc)

    console.print(
        _fmt_ok(f"auto-silence: найдено сегментов тишины — {len(segments)}")
    )


@app.command("export")
def export_decisions(
    decisions: Path = typer.Option(..., exists=True, file_okay=True, dir_okay=False, help="Отредактированный decisions.json"),
    vegas: Optional[Path] = typer.Option(None, help="Куда записать Vegas C# script; если не указано — mutes.cs рядом с decisions.json"),
) -> None:
    """Regenerate a Vegas script from an edited decisions.json."""

    # Дефолт — рядом с decisions.json (это workdir прогона), как в основном
    # пайплайне (<workdir>/mutes.cs). Не тянем стухший относительный output/.
    if vegas is None:
        vegas = decisions.parent / "mutes.cs"

    decisions_doc = read_json(decisions)
    try:
        exported_mutes = exportable_mutes(decisions_doc)
        write_vegas_script(vegas, decisions_doc)
    except VegasExportError as exc:
        raise typer.BadParameter(str(exc), param_hint="--decisions") from exc

    console.print("[green]Vegas script regenerated.[/green]")
    console.print(f"source decisions: [bold]{decisions}[/bold]")
    console.print(f"Vegas script:     [bold]{vegas}[/bold]")
    console.print(f"Exported mutes:   [bold]{len(exported_mutes)}[/bold]")


@app.command()
def audit(
    project: Path = typer.Argument(..., exists=True, help="Папка прогона или decisions.json"),
    out: Optional[Path] = typer.Option(None, help="Куда писать отчёт; по умолчанию audit.md рядом с decisions.json"),
    truth: Optional[Path] = typer.Option(None, help="ground_truth.txt с реальными таймкодами; по умолчанию берётся из папки прогона, если есть"),
    tolerance: float = typer.Option(1.0, help="Допуск сверки с ground_truth, секунды"),
    edits: Optional[Path] = typer.Option(None, help="Журнал ручных правок; по умолчанию <decisions>.edits.json рядом с решениями"),
    max_details: int = typer.Option(40, help="Сколько подозрительных мьютов расписывать подробно"),
) -> None:
    """Сверить найденный мат с реальным: пропуски, ложные срабатывания, границы.

    Отчёт — markdown рядом с decisions.json. Что именно проверяется и чего
    проверить нельзя — см. модуль audit.py и шапку самого отчёта.
    """

    # Дефолт для ground_truth: файл в папке прогона. Так его достаточно один раз
    # положить рядом, и дальше `audit <workdir>` подхватывает сам.
    decisions_path = project / "decisions.json" if project.is_dir() else project
    if truth is None:
        candidate = decisions_path.parent / "ground_truth.txt"
        truth = candidate if candidate.exists() else None

    try:
        result = analyze_audit(
            decisions_path,
            truth_path=truth,
            tolerance_ms=int(round(tolerance * 1000)),
            edits_path=edits,
        )
    except AuditError as exc:
        raise typer.BadParameter(str(exc), param_hint="PROJECT") from exc

    report = write_report(result, out, max_details=max_details)

    missed_core = [m for m in result.missed if m.kind == "core"]
    real_missed = [m for m in missed_core if m.real]
    flagged = sum(1 for m in result.mutes if m.flags)
    console.print(f"Отчёт: [bold]{report}[/bold]")
    console.print(f"Мьютов: [bold]{len(result.mutes)}[/bold], из них с замечаниями: [bold]{flagged}[/bold]")
    # Считаем именно СЛЫШНЫЕ пропуски. Остальные не покрыты только формально, и
    # причины у них РАЗНЫЕ: либо тайминг ASR длиннее слова и подгонка срезала
    # тишину, либо окно уехало в тишину и было перенесено на речь (см.
    # Missed.diagnosis). Одной фразой их описывать нельзя: «вне мьюта тишина»
    # про перенос — неправда, там мьют просто стоит в другом месте.
    excuses = {
        "trimmed_to_speech": "вне мьюта тишина",
        "reanchored": "мьют перенесён на звук",
    }
    not_real = [
        f"{sum(1 for m in missed_core if m.diagnosis == key)} {text}"
        for key, text in excuses.items()
        if any(m.diagnosis == key for m in missed_core)
    ]
    console.print(
        f"Мат слышно (настоящие пропуски): [bold]{len(real_missed)}[/bold]"
        + (f" (+{', '.join(not_real)})" if not_real else "")
    )
    console.print(f"Дыры ASR (звук без слов): [bold]{len(result.holes)}[/bold]")
    if result.journal is not None:
        console.print(
            f"Ручных правок против программы: [bold]{len(result.edits)}[/bold] "
            f"(журнал: {result.edits_path})"
        )
    if result.truth:
        console.print(
            f"ground_truth: попаданий [bold]{len(result.truth_hits)}[/bold]/"
            f"{len(result.truth)}, промахов [bold]{len(result.truth_misses)}[/bold]"
        )
    else:
        console.print(
            "[yellow]ground_truth.txt не найден[/yellow] — как его заполнить, "
            "написано в отчёте"
        )


@app.command("eval")
def eval_edits(
    project: Path = typer.Argument(..., exists=True, help="Папка прогона или decisions.json"),
    edits: Optional[Path] = typer.Option(None, help="Журнал правок; по умолчанию <decisions>.edits.json"),
    json_out: Optional[Path] = typer.Option(None, "--json", help="Куда сохранить замер для сравнения прогонов"),
    baseline: Optional[Path] = typer.Option(None, help="JSON прошлого замера: покажет, что изменилось"),
) -> None:
    """Оценить пайплайн по ручным правкам: точность, полнота, ошибка границ.

    Правки человека — единственная разметка, которая есть бесплатно: журнал
    (`<decisions>.edits.json`) помнит, что решила машина ДО правок. Отчёт
    `audit` объясняет каждую правку словами, а эта команда сводит их в числа,
    которые можно сравнить до и после изменений в пайплайне.
    """
    decisions_path = project / "decisions.json" if project.is_dir() else project
    try:
        result = evaluate_edits(decisions_path, edits_path=edits)
    except AuditError as exc:
        raise typer.BadParameter(str(exc), param_hint="PROJECT") from exc

    prev = read_metrics(baseline) if baseline else None
    if baseline and prev is None:
        console.print(f"[yellow]Прошлый замер не прочитан: {baseline}[/yellow]")
    console.print(render_metrics(result, prev))
    if json_out:
        console.print(f"Замер сохранён: [bold]{write_metrics(result, json_out)}[/bold]")
    if not result.trustworthy:
        console.print(
            "[yellow]Цифры оптимистичнее правды: baseline снят не обработкой "
            "(см. пометку выше)[/yellow]"
        )


# =============================================================================
# doctor & prefetch — health-check и предзагрузка моделей.
# Помогают новому пользователю понять, почему что-то не работает, ДО того как
# он запустит process на 40-минутном стриме и упадёт через 5 минут ожидания.
# =============================================================================


def _fmt_ok(text: str) -> str:
    return f"[green]OK[/green]     {text}"


def _fmt_warn(text: str) -> str:
    return f"[yellow]WARN[/yellow]   {text}"


def _fmt_fail(text: str) -> str:
    return f"[red]FAIL[/red]   {text}"


# Версии необязательного WhisperX-стека (extra [whisperx]). doctor сверяет их,
# только если WhisperX установлен: патчи в transcription.py написаны под
# именно эти версии.
_WHISPERX_EXPECTED_VERSIONS: dict[str, str] = {
    "whisperx": "3.4.5",
    "torch": "2.6.0",
    "faster_whisper": "1.2.1",
    "pyannote.audio": "3.4.0",
    "speechbrain": "1.1.0",
    "pytorch_lightning": "2.6.5",
    "transformers": "5.12.1",
    "huggingface_hub": "1.21.0",
    "ctranslate2": "4.4.0",
    "omegaconf": "2.3.1",
}


def _whisperx_installed() -> bool:
    from importlib.util import find_spec

    try:
        return find_spec("whisperx") is not None
    except (ImportError, ValueError):
        return False


@app.command()
def doctor() -> None:
    """Проверить окружение: Python, onnxruntime, ускоритель, ffmpeg, модели."""
    import shutil
    import sys
    from importlib import metadata

    from .config import MODELS_DIR, default_device

    console.print("[bold]twitch-cut doctor[/bold] — проверка окружения\n")

    problems = 0

    # --- Python версия ---
    py_ver = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    if (3, 10) <= sys.version_info[:2] < (3, 13):
        console.print(_fmt_ok(f"Python {py_ver}"))
    else:
        console.print(_fmt_fail(f"Python {py_ver} — нужен 3.10, 3.11 или 3.12"))
        problems += 1

    # --- onnxruntime + ускоритель ---
    try:
        import onnxruntime as ort  # noqa: WPS433

        from .onnx_models import accelerator_label

        providers = ", ".join(p.replace("ExecutionProvider", "") for p in ort.get_available_providers())
        device = default_device()
        accel = accelerator_label(device)
        line = f"onnxruntime {ort.__version__} (провайдеры: {providers}); device={device} → {accel}"
        if accel == "CPU" and device != "cpu":
            console.print(
                _fmt_warn(f"{line}. Ускоритель не найден — распознавание пойдёт на CPU (медленнее).")
            )
        else:
            console.print(_fmt_ok(line))
    except ImportError:
        console.print(_fmt_fail("onnxruntime не установлен — распознавание не запустится"))
        problems += 1

    # --- необязательный WhisperX-стек ---
    if _whisperx_installed():
        for pkg, expected in _WHISPERX_EXPECTED_VERSIONS.items():
            try:
                installed = metadata.version(pkg)
            except metadata.PackageNotFoundError:
                console.print(_fmt_warn(f"{pkg} не установлен (WhisperX ждёт =={expected})"))
                continue
            # Сравниваем без учёта локального суффикса типа +cu126.
            installed_base = installed.split("+", 1)[0]
            if installed_base == expected:
                console.print(_fmt_ok(f"{pkg}=={installed}"))
            else:
                console.print(
                    _fmt_warn(
                        f"{pkg}=={installed}, ожидалось =={expected}. "
                        "Патчи в transcription.py могут сработать некорректно."
                    )
                )
    else:
        console.print(_fmt_ok("WhisperX не установлен — он не нужен движку по умолчанию (GigaAM)"))

    # --- ffmpeg ---
    try:
        import imageio_ffmpeg

        ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
        if Path(ffmpeg_path).exists():
            console.print(_fmt_ok(f"ffmpeg: {ffmpeg_path}"))
        else:
            console.print(_fmt_fail(f"imageio_ffmpeg вернул несуществующий путь: {ffmpeg_path}"))
            problems += 1
    except Exception as exc:  # noqa: BLE001
        console.print(_fmt_fail(f"imageio-ffmpeg сломан: {exc}"))
        problems += 1

    # --- ONNX-модели ---
    from .onnx_models import ALL_MODELS, is_present

    for model_file in ALL_MODELS:
        if is_present(model_file):
            size_mb = model_file.path.stat().st_size / (1024 * 1024)
            console.print(_fmt_ok(f"{model_file.name} ({size_mb:.0f} МБ)"))
        else:
            console.print(
                _fmt_warn(f"{model_file.name} не скачана. Запустите:  twitch-cut prefetch")
            )

    # --- свободное место на диске моделей ---
    disk = shutil.disk_usage(MODELS_DIR.parent if MODELS_DIR.parent.exists() else Path.cwd())
    free_gb = disk.free / (1024**3)
    if free_gb < 2:
        console.print(_fmt_warn(f"Свободно на диске: {free_gb:.1f} GB — модели весят ~0.5 GB"))
    else:
        console.print(_fmt_ok(f"Свободно на диске: {free_gb:.1f} GB"))

    # --- Фаза 4: FastAPI backend для desktop UI ------------------------------
    # Три проверки: (a) fastapi+uvicorn установлены — FAIL, без них `serve`
    # не запустится; (b) node в PATH — WARN, без него UI не соберётся, но
    # CLI работает; (c) desktop/dist/index.html существует — WARN, без него
    # `serve --static-dir desktop/dist` вернёт пустую страницу.
    for pkg in ("fastapi", "uvicorn"):
        try:
            v = metadata.version(pkg)
            console.print(_fmt_ok(f"{pkg}=={v}"))
        except metadata.PackageNotFoundError:
            console.print(_fmt_fail(f"{pkg} не установлен — команда `twitch-cut serve` не запустится"))
            problems += 1

    # WebSocket-стек — без него /jobs/{id}/events отдаёт 404, ProgressScreen
    # в UI показывает "Failed to fetch". uvicorn[standard] должен ставить его
    # автоматически, но если venv собран без extras — молча ломается.
    try:
        ws_ver = metadata.version("websockets")
        console.print(_fmt_ok(f"websockets=={ws_ver}"))
    except metadata.PackageNotFoundError:
        try:
            wp_ver = metadata.version("wsproto")
            console.print(_fmt_ok(f"wsproto=={wp_ver} (WS-стек)"))
        except metadata.PackageNotFoundError:
            console.print(_fmt_fail(
                "websockets/wsproto не установлены — WebSocket в UI не заработает "
                "(GET /jobs/{id}/events → 404). Fix: "
                "pip install 'uvicorn[standard]==0.32.1' 'websockets==14.1'"
            ))
            problems += 1

    # cuDNN 8 — нужен ТОЛЬКО ctranslate2/whisperx на Windows (GigaAM идёт через
    # onnxruntime и cuDNN не трогает). torch 2.6+cu126 тянет свой cuDNN 9 в
    # torch\lib, но ctranslate2 ищет `cudnn_ops_infer64_8.dll` через
    # LoadLibrary → нужен либо в PATH, либо в site-packages\nvidia\cudnn\bin
    # (пакет nvidia-cudnn-cu12). Если не найден — backend упадёт с
    # STATUS_STACK_BUFFER_OVERRUN сразу как whisperx попытается загрузить модель.
    if sys.platform == "win32" and _whisperx_installed():
        import os

        cudnn_found_via: Optional[str] = None
        # 1) pip-пакет nvidia-cudnn-cu12
        try:
            import nvidia.cudnn  # type: ignore  # noqa: WPS433
            cudnn_bin = Path(nvidia.cudnn.__file__).parent / "bin"
            if cudnn_bin.exists() and any(cudnn_bin.glob("cudnn_ops_infer64_*.dll")):
                cudnn_found_via = f"pip: {cudnn_bin}"
            elif cudnn_bin.exists() and any(cudnn_bin.glob("cudnn*.dll")):
                cudnn_found_via = f"pip (cuDNN 9): {cudnn_bin}"
        except ImportError:
            pass
        # 2) PATH
        if cudnn_found_via is None:
            for p in os.environ.get("PATH", "").split(os.pathsep):
                if not p:
                    continue
                pp = Path(p)
                if pp.exists() and any(pp.glob("cudnn_ops_infer64_*.dll")):
                    cudnn_found_via = f"PATH: {pp}"
                    break

        if cudnn_found_via:
            console.print(_fmt_ok(f"cuDNN DLL найден ({cudnn_found_via})"))
        else:
            console.print(_fmt_warn(
                "cuDNN DLL не найден. При GPU-транскрипции backend упадёт с "
                "STATUS_STACK_BUFFER_OVERRUN (exit 3221226505). Fix: "
                "pip install nvidia-cudnn-cu12==8.9.7.29  "
                "(Electron main.ts автоматически подхватит site-packages\\nvidia\\cudnn\\bin в PATH)."
            ))
    # На Linux cuDNN обычно из системного пакета — не проверяем.

    node_bin = shutil.which("node")
    if node_bin:
        console.print(_fmt_ok(f"node: {node_bin}"))
    else:
        console.print(_fmt_warn(
            "Node.js не найден в PATH — desktop UI собрать нельзя. "
            "Скачай LTS с https://nodejs.org (нужен только для сборки; CLI работает без него)."
        ))

    # Пробуем найти собранный фронт относительно репозитория (backend/../desktop/dist)
    # или относительно текущей директории.
    dist_candidates = [
        Path(__file__).resolve().parents[3] / "desktop" / "dist" / "index.html",
        Path.cwd() / "desktop" / "dist" / "index.html",
    ]
    dist_found = next((p for p in dist_candidates if p.exists()), None)
    if dist_found is not None:
        console.print(_fmt_ok(f"desktop UI собран: {dist_found}"))
    else:
        console.print(_fmt_warn(
            "desktop/dist/index.html не найден — UI не собран. "
            "Собери:  cd desktop && npm ci && npm run build"
        ))

    console.print()
    if problems == 0:
        console.print("[bold green]Всё ок.[/bold green] Можно запускать `twitch-cut process ...`")
        raise typer.Exit(0)
    else:
        console.print(f"[bold red]Проблем: {problems}.[/bold red] Смотри FAIL строки выше.")
        raise typer.Exit(1)


@app.command()
def accel() -> None:
    """Какой ускоритель реально поднимается на этой машине — одна строка JSON.

    Зовёт установщик приложения после prefetch, чтобы показать бейдж GPU/CPU.
    Проверка честная: открываем настоящую сессию на маленькой VAD-модели с
    устройством по умолчанию и смотрим, какой провайдер она получила (у
    DirectML, например, провайдер «есть» и на машине без видеокарты — понять,
    поднимется ли он, можно только создав сессию).
    """
    import json

    from .config import default_device
    from .onnx_models import VAD_MODEL, create_session, is_present

    device = default_device()
    accelerator = "CPU"
    if is_present(VAD_MODEL):
        session = create_session(VAD_MODEL.path, device)
        accelerator = session.get_providers()[0].replace("ExecutionProvider", "")
    print(json.dumps({"device": device, "accelerator": accelerator}), flush=True)


@app.command()
def prefetch(
    with_whisperx: bool = typer.Option(
        False,
        "--whisperx",
        help="Ещё и модели WhisperX (~4 GB; нужен extra [whisperx]).",
    ),
    model: str = typer.Option("large-v3", help="WhisperX model для предзагрузки"),
    language: str = typer.Option("ru", help="Язык alignment-модели (wav2vec2)"),
    device: str = typer.Option("cpu", help="Куда временно грузить веса WhisperX (cpu безопаснее)"),
) -> None:
    """Скачать ONNX-модели (GigaAM v3 + VAD, ~430 MB) в models/onnx/.

    Полезно вызвать один раз при установке — тогда первый `process` не
    будет молча качать модели в середине пайплайна. Установщик приложения
    зовёт именно эту команду и показывает её вывод в окне первого запуска.
    """
    from .config import MODELS_DIR
    from .onnx_models import ALL_MODELS, ensure_model

    setup_logging()

    console.print(f"[bold]Prefetch[/bold] моделей в: {MODELS_DIR}\n")

    # 1. ONNX-модели — без них распознавание не запустится, поэтому сбой здесь
    # роняет prefetch (а с ним и первый запуск: кнопка «Повторить» в сплэше).
    last_print = 0.0

    def _report(name: str, done: int, total: int) -> None:
        # Строка на секунду, а не на каждый мегабайт: сплэш показывает
        # последнюю строку вывода, а лог установки не должен пухнуть.
        nonlocal last_print
        now = time.monotonic()
        if now - last_print < 1.0 and done != total:
            return
        last_print = now
        done_mb = done / (1024 * 1024)
        if total:
            print(f"  {name}: {done * 100 // total}% ({done_mb:.0f} / {total / (1024 * 1024):.0f} МБ)", flush=True)
        else:
            print(f"  {name}: {done_mb:.0f} МБ", flush=True)

    for model_file in ALL_MODELS:
        console.print(f"→ {model_file.name} (~{model_file.size_mb} МБ)...")
        try:
            ensure_model(model_file, on_progress=_report)
        except Exception as exc:  # noqa: BLE001
            console.print(_fmt_fail(f"{model_file.name}: {exc}"))
            raise typer.Exit(1) from exc
        console.print(_fmt_ok(model_file.name))

    if with_whisperx:
        _prefetch_whisperx(model=model, language=language, device=device)

    console.print()
    console.print("[bold green]Готово.[/bold green] Модели закэшированы, первый прогон не будет скачивать.")


def _prefetch_whisperx(*, model: str, language: str, device: str) -> None:
    """Модели необязательного движка WhisperX: faster-whisper + wav2vec2."""
    # Заглушаем verbose HTTP-логи, иначе они топят tqdm-прогресс-бар и
    # пользователь думает что всё зависло. httpx/hf_hub логгеры сами по себе
    # ничего полезного не показывают — реальная информация в tqdm-баре.
    for noisy in ("httpx", "huggingface_hub", "urllib3", "filelock"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    if not _whisperx_installed():
        console.print(_fmt_fail('WhisperX не установлен: pip install -e ".[whisperx]"'))
        raise typer.Exit(1)

    # 1. WhisperX ASR (faster-whisper через ctranslate2).
    try:
        import whisperx  # noqa: WPS433

        # ВАЖНО: apply_whisperx_patches() до whisperx.load_model. Без него
        # PyTorch 2.6 упадёт на weights_only=True при загрузке pyannote VAD
        # checkpoint (omegaconf.ListConfig не в whitelist). Патчи идемпотентны.
        from .transcription import apply_whisperx_patches
        apply_whisperx_patches()

        console.print(f"→ Скачиваю WhisperX {model} (~3 GB для large-v3, это долго)...")
        # compute_type=int8 позволяет load_model отработать даже без CUDA.
        whisperx.load_model(model, device=device, compute_type="int8", language=language)
        console.print(_fmt_ok(f"WhisperX {model}"))
    except Exception as exc:  # noqa: BLE001
        console.print(_fmt_fail(f"WhisperX {model}: {exc}"))
        raise typer.Exit(1) from exc

    # 2. Alignment (wav2vec2 для нужного языка).
    #
    # Alignment нужен ТОЛЬКО движку WhisperX (word-level тайминги). Движок по
    # умолчанию — GigaAM, у него своя выдача, alignment ему не требуется.
    # Поэтому сбой этой стадии НЕ должен ронять весь первый запуск (bootstrap):
    # раньше `raise typer.Exit(1)` здесь блокировал установку целиком, даже
    # пользователям на GigaAM. Делаем 3 попытки (HF временами отдаёт 5xx/таймаут
    # на больших файлах), а если так и не встало — предупреждаем и продолжаем.
    align_ok = False
    last_exc: Exception | None = None
    for attempt in range(1, 4):
        try:
            suffix = "" if attempt == 1 else f" (попытка {attempt}/3)"
            console.print(f"→ Скачиваю alignment-модель для '{language}' (~1 GB){suffix}...")
            whisperx.load_align_model(language_code=language, device=device)
            console.print(_fmt_ok(f"alignment ({language})"))
            align_ok = True
            break
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            console.print(_fmt_fail(f"alignment: {exc}"))
            if attempt < 3:
                time.sleep(2 * attempt)  # 2s, 4s — простой бэкофф

    if not align_ok:
        # Не Exit(1): alignment докачается при первом WhisperX-прогоне, а
        # GigaAM (движок по умолчанию) от него не зависит.
        console.print(
            f"[yellow]![/yellow] alignment-модель ('{language}') не установилась "
            f"({last_exc}).\n"
            "  Это нужно только для движка WhisperX — GigaAM (по умолчанию) работает без неё.\n"
            "  Программа запустится; alignment докачается при первом WhisperX-прогоне."
        )

    # pyannote VAD для WhisperX грузится внутри whisperx.load_model выше,
    # отдельно предзагружать не нужно.


# =============================================================================
# highlights-export / highlights-import — Фаза 3, интеграция с NotebookLM.
# NotebookLM не даёт API, поэтому обмен через файлы: программа готовит пакет
# → пользователь руками работает в UI → программа валидирует ответ.
# =============================================================================


def _resolve_transcript_path(decisions_doc: dict, override: Optional[Path]) -> Path:
    if override is not None:
        return override
    caches = decisions_doc.get("caches") or {}
    cached = caches.get("transcript")
    if not cached:
        raise typer.BadParameter(
            "decisions.json не содержит caches.transcript — передай --transcript явно."
        )
    path = Path(cached)
    if not path.exists():
        raise typer.BadParameter(
            f"Транскрипт из decisions.caches.transcript не найден: {path}. "
            "Передай --transcript явно, если файл переехал."
        )
    return path


@app.command("highlights-export")
def highlights_export(
    decisions: Path = typer.Option(
        ..., exists=True, file_okay=True, dir_okay=False,
        help="decisions.json — берём отсюда путь к транскрипту.",
    ),
    out_dir: Path = typer.Option(
        ..., help="Каталог, куда положить пакет для NotebookLM (создаётся).",
    ),
    transcript: Optional[Path] = typer.Option(
        None, exists=True, file_okay=True, dir_okay=False,
        help="Путь к транскрипту (переопределяет decisions.caches.transcript).",
    ),
    n_highlights: int = typer.Option(5, min=1, max=50, help="Сколько highlights попросить у LLM."),
) -> None:
    """Собрать пакет `transcript_*.md + prompt.md + schema.json` для NotebookLM."""

    decisions_doc = read_json(decisions)
    transcript_path = _resolve_transcript_path(decisions_doc, transcript)
    transcript_doc = read_json(transcript_path)

    manifest = build_notebooklm_package(
        transcript_doc, out_dir, n_highlights=n_highlights,
    )

    console.print(f"[green]NotebookLM package готов.[/green] → [bold]{out_dir}[/bold]")
    console.print(f"chunks:       [bold]{len(manifest['chunks'])}[/bold]")
    console.print(f"total words:  [bold]{manifest['total_words']:,}[/bold]")
    console.print(f"n_highlights: [bold]{manifest['n_highlights']}[/bold]")
    console.print()
    console.print("Дальше:")
    console.print("  1. Открой [bold]https://notebooklm.google.com[/bold], создай notebook.")
    console.print(f"  2. Загрузи как sources все [bold]transcript_*.md[/bold] из {out_dir}.")
    console.print(f"  3. Вставь в чат содержимое [bold]{out_dir / 'prompt.md'}[/bold].")
    console.print("  4. Сохрани JSON-ответ модели в файл (например response.json).")
    console.print(
        "  5. Запусти [bold]twitch-cut highlights-import "
        "--decisions ... --response ... --output ...[/bold]"
    )


@app.command("highlights-import")
def highlights_import(
    decisions: Path = typer.Option(
        ..., exists=True, file_okay=True, dir_okay=False,
        help="Существующий decisions.json, в который вольём highlights.",
    ),
    response: Path = typer.Option(
        ..., exists=True, file_okay=True, dir_okay=False,
        help="JSON-ответ от NotebookLM (можно с markdown code-fence).",
    ),
    output: Path = typer.Option(
        ..., help="Куда записать обновлённый decisions.json (можно тот же путь).",
    ),
    transcript: Optional[Path] = typer.Option(
        None, exists=True, file_okay=True, dir_okay=False,
        help="Транскрипт для расчёта диапазона (по умолчанию — из decisions.caches).",
    ),
) -> None:
    """Провалидировать JSON-ответ NotebookLM и записать highlights в decisions.json."""

    decisions_doc = read_json(decisions)

    # Диапазон транскрипта — для проверки, что LLM не улетела за границы.
    # Если транскрипт недоступен, пропускаем эту проверку.
    range_s: tuple[float, float] | None = None
    transcript_hash: Optional[str] = None
    try:
        transcript_path = _resolve_transcript_path(decisions_doc, transcript)
        transcript_doc = read_json(transcript_path)
        segments = transcript_doc.get("segments") or []
        if segments:
            first = segments[0].get("start") or 0.0
            last = segments[-1].get("end") or first
            range_s = (float(first), float(last))
        transcript_hash = file_content_hash(transcript_path)
    except typer.BadParameter:
        console.print(
            "[yellow]WARN[/yellow] транскрипт не найден — пропускаю range-check."
        )

    try:
        hs, warnings = parse_text(
            response.read_text(encoding="utf-8"),
            transcript_range_s=range_s,
            transcript_hash=transcript_hash,
        )
    except MergeError as exc:
        console.print("[red]Не могу смержить highlights:[/red]")
        for r in exc.reasons:
            console.print(f"  • {r}")
        raise typer.Exit(code=1)

    if warnings:
        console.print(f"[yellow]Отброшено {len(warnings)} highlight(s):[/yellow]")
        for w in warnings:
            console.print(f"  • {w}")

    merged = merge_into_decisions(decisions_doc, hs)
    write_json(output, merged)

    console.print(f"[green]Highlights смержены.[/green] → [bold]{output}[/bold]")
    console.print(f"count:  [bold]{len(hs.highlights)}[/bold]")
    for h in hs.highlights:
        console.print(
            f"  • [{h.start_s:.1f}s → {h.end_s:.1f}s] "
            f"(score={h.score:.2f}) [bold]{h.title}[/bold]"
        )


@app.command("detect-offtopic")
def detect_offtopic(
    decisions: Path = typer.Option(
        ..., exists=True, file_okay=True, dir_okay=False,
        help="decisions.json — берём отсюда транскрипт и range_in.",
    ),
    output: Path = typer.Option(
        ..., help="Куда записать decisions.json с ключом 'offtopic' (можно тот же путь).",
    ),
    provider_id: str = typer.Option(
        "builtin:anthropic", "--provider",
        help="ID встроенного провайдера (builtin:anthropic|openai|openrouter|ollama).",
    ),
    api_key: Optional[str] = typer.Option(
        None, "--api-key", envvar="TWITCH_CUT_AI_KEY",
        help="API-ключ провайдера. В проде приходит из Electron safeStorage; "
        "для CLI-теста задай флагом или env TWITCH_CUT_AI_KEY.",
    ),
    method: str = typer.Option(
        "auto", "--method",
        help="auto|source|single-pass|windows. auto: есть исходник → source, "
        "иначе → single-pass. windows — старый оконный путь (совместимость).",
    ),
    source_audio: Optional[Path] = typer.Option(
        None, "--source-audio", exists=True, file_okay=True, dir_okay=False,
        help="Локальный файл исходного видео — распознаём тем же движком и "
        "используем как ground truth (путь по исходнику). URL не поддерживается.",
    ),
    source_transcript: Optional[Path] = typer.Option(
        None, "--source-transcript", exists=True, file_okay=True, dir_okay=False,
        help="Готовый транскрипт исходника (JSON) — вместо распознавания --source-audio.",
    ),
    source_model: str = typer.Option(
        "v3_ctc", "--source-model",
        help="Модель ASR для исходника. Должна совпадать с моделью стрима — "
        "иначе тексты несравнимы и привязка ломается.",
    ),
    source_device: str = typer.Option(
        "cpu", "--source-device", help="Устройство ASR исходника (cpu безопаснее).",
    ),
    model: Optional[str] = typer.Option(
        None, "--model",
        help="Модель LLM для пути по исходнику/одного прохода (по умолчанию — дефолт провайдера).",
    ),
    cheap_model: Optional[str] = typer.Option(
        None, "--cheap-model", help="[windows] Модель первой ступени (дефолт провайдера)."
    ),
    expensive_model: Optional[str] = typer.Option(
        None, "--expensive-model", help="[windows] Модель второй ступени каскада (None → без каскада)."
    ),
    transcript: Optional[Path] = typer.Option(
        None, exists=True, file_okay=True, dir_okay=False,
        help="Транскрипт стрима (по умолчанию — из decisions.caches.transcript).",
    ),
) -> None:
    """Прогнать транскрипт через AI-классификатор оффтопа и записать в decisions.json.

    Три пути (см. --method):
      source       — по транскрипту исходника: тема из видео, на вход только речь
                     стримера + что играло в кадре. Нужен --source-audio/-transcript.
      single-pass  — деградация без исходника: один проход, границы задаёт модель.
      windows      — старый оконный путь (совместимость): судит СМЕСЬ речи.
    """

    from .llm_offtopic import (
        LLMClient,
        LLMError,
        builtin_provider,
        classify_single_pass,
        classify_transcript,
        classify_with_source,
    )
    from .llm_offtopic import merge_into_decisions as merge_offtopic

    valid_methods = {"auto", "source", "single-pass", "windows"}
    if method not in valid_methods:
        raise typer.BadParameter(f"--method должен быть одним из {sorted(valid_methods)}")
    if source_audio is not None and source_transcript is not None:
        raise typer.BadParameter("Задай либо --source-audio, либо --source-transcript, не оба.")

    provider = builtin_provider(provider_id)
    if provider is None:
        raise typer.BadParameter(
            f"Неизвестный provider_id={provider_id!r}. Встроенные: "
            "builtin:anthropic, builtin:openai, builtin:openrouter, builtin:ollama."
        )

    decisions_doc = read_json(decisions)
    transcript_path = _resolve_transcript_path(decisions_doc, transcript)
    transcript_doc = read_json(transcript_path)
    transcript_hash = file_content_hash(transcript_path)

    have_source = source_audio is not None or source_transcript is not None
    if method == "auto":
        resolved_method = "source" if have_source else "single-pass"
    else:
        resolved_method = method
    if resolved_method == "source" and not have_source:
        raise typer.BadParameter(
            "--method source требует --source-audio или --source-transcript."
        )

    client = LLMClient(provider, api_key=api_key)
    console.print(
        f"[cyan]Классифицирую оффтоп[/cyan] method=[bold]{resolved_method}[/bold] "
        f"provider=[bold]{provider.id}[/bold]"
    )

    try:
        if resolved_method == "source":
            source_doc, source_hash, resolved_source_model = _load_source_transcript(
                source_audio=source_audio,
                source_transcript=source_transcript,
                workdir=decisions.parent,
                model_name=source_model,
                device=source_device,
            )
            result = classify_with_source(
                transcript_doc,
                source_doc,
                client=client,
                model=model,
                transcript_hash=transcript_hash,
                source_transcript_hash=source_hash,
                source_model=resolved_source_model,
            )
        elif resolved_method == "single-pass":
            result = classify_single_pass(
                transcript_doc,
                client=client,
                model=model,
                transcript_hash=transcript_hash,
            )
        else:  # windows — старый путь
            console.print(
                f"  cheap=[bold]{cheap_model or provider.default_model()}[/bold] "
                f"expensive=[bold]{expensive_model or '—'}[/bold]"
            )
            result = classify_transcript(
                transcript_doc,
                client=client,
                cheap_model=cheap_model,
                expensive_model=expensive_model,
                transcript_hash=transcript_hash,
            )
    except LLMError as exc:
        console.print(f"[red]LLM ошибка:[/red] {exc}")
        raise typer.Exit(code=1)

    merged = merge_offtopic(decisions_doc, result)
    write_json(output, merged)
    # Если рядом есть журнал правок — оффтоп посчитала МАШИНА, и это должно
    # лежать в базовом снимке, а не всплыть потом как «человек добавил».
    try:
        from .edit_journal import refresh_baseline

        refresh_baseline(
            output,
            merged,
            ("offtopic",),
            note=f"detect-offtopic ({result.method}) — сегменты посчитала программа",
        )
    except Exception:  # noqa: BLE001
        pass

    console.print(f"[green]Оффтоп смержен.[/green] → [bold]{output}[/bold]")
    console.print(
        f"segments: [bold]{len(result.segments)}[/bold]  "
        f"total: [bold]{result.total_offtopic_s:.1f}s[/bold]"
    )
    for s in result.segments:
        console.print(
            f"  • [{s.start_s:.1f}s → {s.end_s:.1f}s] "
            f"(conf={s.confidence:.2f}, {s.stage}) {s.reason[:80]}"
        )


def _load_source_transcript(
    *,
    source_audio: Optional[Path],
    source_transcript: Optional[Path],
    workdir: Path,
    model_name: str,
    device: str,
) -> tuple[dict, Optional[str], Optional[str]]:
    """Транскрипт исходника + провенанс: (doc, source_transcript_hash, source_model).

    Готовый JSON — хеш по содержимому файла, модель ASR неизвестна (None). При
    распознавании --source-audio — ключ кэша GigaAM как хеш и фактическая модель.
    """
    if source_transcript is not None:
        return read_json(source_transcript), file_content_hash(source_transcript), None

    from .llm_offtopic import SourceError, transcribe_source

    console.print(
        f"[cyan]Распознаю исходник[/cyan] [bold]{source_audio}[/bold] "
        f"(model={model_name}, device={device})…"
    )
    try:
        result = transcribe_source(
            source_audio,
            workdir,
            model_name=model_name,
            device=device,
            on_progress=lambda f, m: None,
        )
    except SourceError as exc:
        console.print(f"[red]Исходник:[/red] {exc}")
        raise typer.Exit(code=1)
    console.print(
        f"  исходник распознан: слов [bold]{result.word_count}[/bold], "
        f"{result.duration_s / 60.0:.1f} мин"
    )
    return result.transcript, result.key, model_name



# =============================================================================
# serve — FastAPI backend для Electron desktop UI (Фаза 4).
# =============================================================================


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", help="Хост для uvicorn (по умолчанию только loopback)."),
    port: int = typer.Option(
        0, help="Порт uvicorn. 0 = выбрать свободный; актуальный порт печатается как TWITCH_CUT_PORT=NNNN."
    ),
    static_dir: Optional[Path] = typer.Option(
        None,
        "--static-dir",
        help="Каталог со статикой фронта (обычно desktop/dist). Если не задан, / не отдаёт index.html.",
    ),
    reload: bool = typer.Option(False, help="uvicorn --reload (dev only, замедляет старт)."),
) -> None:
    """Запустить FastAPI backend. Electron грепает `TWITCH_CUT_PORT=` из stdout."""

    import sys

    import uvicorn

    from .server.app import create_app
    from .server.settings import logs_dir

    # Логи backend-процесса: файл в каталоге настроек + stderr. Без этого
    # обработчиков у логгера пакета нет вообще (uvicorn настраивает только свои),
    # и диагностика джоб — включая «окно без вердикта» у оффтоп-классификатора —
    # не доходит никуда.
    log_path = setup_server_logging(logs_dir())
    if log_path is not None:
        print(f"TWITCH_CUT_LOG={log_path}", flush=True)

    application = create_app(static_dir=static_dir)

    # uvicorn при port=0 биндит на свободный порт, но по умолчанию не сообщает
    # какой. Хук через lifespan: как только сокет создан, uvicorn заполняет
    # `server.servers[0].sockets[0]` — оттуда достаём реальный порт и печатаем
    # маркерную строку в stdout, чтобы Electron main мог её грепать.
    config = uvicorn.Config(
        application,
        host=host,
        port=port,
        log_level="info",
        reload=reload,
    )
    server = uvicorn.Server(config)

    original_startup = server.startup

    async def _startup_with_port_broadcast(sockets=None):
        await original_startup(sockets=sockets)
        try:
            srv = server.servers[0]  # type: ignore[attr-defined]
            sock = list(srv.sockets)[0]
            actual_port = sock.getsockname()[1]
            print(f"TWITCH_CUT_PORT={actual_port}", flush=True)
            sys.stdout.flush()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not determine bound port: %s", exc)

    server.startup = _startup_with_port_broadcast  # type: ignore[assignment]
    server.run()


if __name__ == "__main__":
    app()

