from __future__ import annotations

import logging
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

# ВАЖНО: configure_hf_cache() ДОЛЖЕН вызваться до импорта .transcription
# (который тянет whisperx → torch → huggingface_hub). HuggingFace-либы читают
# HF_HOME один раз при импорте — поменять его позже не получится.
from .config import DEFAULT_ASR_OPTIONS, PipelineConfig, configure_hf_cache

configure_hf_cache()

from .cache import file_content_hash, read_json, stable_hash, write_json  # noqa: E402
from .decisions import build_decisions, write_decisions  # noqa: E402
from .ffmpeg_tools import extract_audio_range, probe_media_duration  # noqa: E402
from .llm.exporter import build_notebooklm_package  # noqa: E402
from .llm.importer import MergeError, merge_into_decisions, parse_response  # noqa: E402
from .logging_setup import setup_logging  # noqa: E402
from .profanity import ProfanityMatch, RussianNormalizer, detect_profanity, load_banwords  # noqa: E402
from .timecode import TimeSpan, parse_timecode  # noqa: E402
from .transcription import load_mock_transcript, transcribe_audio  # noqa: E402
from .vegas_export import VegasExportError, exportable_mutes, write_vegas_script  # noqa: E402
from .whisper_cpp import WhisperCppError, transcribe_with_whisper_cpp  # noqa: E402

app = typer.Typer(
    help="CLI-ядро Фазы 1: stream → WhisperX → мат → decisions.json → Vegas .cs",
    no_args_is_help=True,
)
console = Console()
logger = logging.getLogger(__name__)

@app.callback()
def main() -> None:
    """Phase 1 command group."""


def _load_detection_cache(path: Path) -> list[ProfanityMatch]:
    payload = read_json(path)
    return [ProfanityMatch(**item) for item in payload["matches"]]


def _write_detection_cache(path: Path, key: str, matches: list[ProfanityMatch]) -> None:
    write_json(
        path,
        {
            "stage": "profanity_detection",
            "key": key,
            "matches": [asdict(match) for match in matches],
        },
    )


def _resolve_range(stream: Path, range_in: Optional[str], range_out: Optional[str]) -> tuple[TimeSpan, TimeSpan]:
    start = parse_timecode(range_in) if range_in else TimeSpan(0)
    end = parse_timecode(range_out) if range_out else probe_media_duration(stream)
    if end <= start:
        raise typer.BadParameter("--range-out должен быть больше --range-in")
    return start, end


@app.command()
def process(
    stream: Path = typer.Option(..., exists=True, file_okay=True, dir_okay=False, help="Исходный stream.mp4"),
    original: Path = typer.Option(..., exists=True, file_okay=True, dir_okay=False, help="Оригинальное видео реакции"),
    range_in: Optional[str] = typer.Option(None, "--range-in", help="Начало обработки в исходном stream.mp4; если не указано, 00:00:00"),
    range_out: Optional[str] = typer.Option(None, "--range-out", help="Конец обработки в исходном stream.mp4; если не указано, конец видео"),
    banwords: Path = typer.Option(..., exists=True, file_okay=True, dir_okay=False, help="Словарь мата/банвордов"),
    workdir: Path = typer.Option(Path("work/job_001"), help="Рабочая папка cache/checkpoints"),
    decisions: Path = typer.Option(Path("output/decisions.json"), help="Куда записать decisions.json"),
    vegas: Path = typer.Option(Path("output/vegas_build.cs"), help="Куда записать Vegas C# script"),
    model: str = typer.Option("large-v3", help="WhisperX/faster-whisper model"),
    language: str = typer.Option("ru", help="Язык транскрипции"),
    device: str = typer.Option("cuda", help="cuda или cpu"),
    compute_type: str = typer.Option("float16", help="float16, float32 или int8"),
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
        "whisperx",
        "--transcriber",
        help="Backend транскрипции: 'whisperx' (default, PyTorch+CUDA) или 'whispercpp' (subprocess whisper-cli.exe)",
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
    mute_padding_before_ms: int = typer.Option(80, min=0, help="Padding перед словом"),
    mute_padding_after_ms: int = typer.Option(120, min=0, help="Padding после слова"),
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
    mute_join_gap_ms: int = typer.Option(
        600,
        "--mute-join-gap-ms",
        min=0,
        help="Макс. разрыв (мс) между соседними матами для склейки в один mute. Больше — два отдельных mute, чистая речь между ними не мьютится.",
    ),
    raw_mute: bool = typer.Option(
        False,
        "--raw-mute",
        help="Диагностический режим: мьютить СТРОГО по таймингам whisper для каждого банворда (без padding, без склейки, без extend, без cap). Покажет, что выдаёт whisper без нашей обработки.",
    ),
    force_extract: bool = typer.Option(False, help="Пересоздать WAV extraction"),
    force_transcribe: bool = typer.Option(False, help="Пересоздать WhisperX transcript cache"),
    force_detect: bool = typer.Option(False, help="Пересоздать cache детекции мата"),
    mock_transcript: Optional[Path] = typer.Option(None, exists=True, file_okay=True, dir_okay=False, help="JSON transcript для smoke-теста без WhisperX"),
    verbose: bool = typer.Option(False, help="Подробный лог"),
) -> None:
    """Run Phase 1 pipeline and generate editable decisions + Vegas audio mutes."""

    workdir.mkdir(parents=True, exist_ok=True)
    setup_logging(workdir, verbose=verbose)

    # Anti-hallucination профиль faster-whisper (см. config.DEFAULT_ASR_OPTIONS).
    # По умолчанию включён — без него Whisper на длинных стримах галлюцинирует
    # и сжимает тайминги на 5+ секунд, теряя целые фразы (regression fix для
    # detection_2b4b1a5c89ffaa3a). Флаг --disable-asr-hardening возвращает
    # старое поведение для сравнения/отладки.
    asr_options: dict[str, object] | None = (
        None if disable_asr_hardening else dict(DEFAULT_ASR_OPTIONS)
    )

    config = PipelineConfig(
        language=language,
        model=model,
        device=device,
        compute_type=compute_type,
        batch_size=batch_size,
        vad_filter=vad_filter,
        vad_method=vad_method,
        asr_options=asr_options or {},
        mute_padding_before_ms=mute_padding_before_ms,
        mute_padding_after_ms=mute_padding_after_ms,
        mute_extend_mode=mute_extend_mode,
        mute_max_seconds=mute_max_seconds,
        mute_join_gap_ms=mute_join_gap_ms,
        transcriber=transcriber,
        raw_mute=raw_mute,
    )
    config.validate()

    start, end = _resolve_range(stream, range_in, range_out)

    console.rule("[bold cyan]Фаза 1: stream → Vegas audio mutes")
    console.print(f"Диапазон: [bold]{start.format()} → {end.format()}[/bold]")

    if transcriber not in {"whisperx", "whispercpp"}:
        raise typer.BadParameter("--transcriber должен быть 'whisperx' или 'whispercpp'")
    if transcriber == "whispercpp" and whisper_cpp_model is None:
        raise typer.BadParameter(
            "--whisper-cpp-model обязателен при --transcriber whispercpp",
            param_hint="--whisper-cpp-model",
        )

    audio_path = None
    audio_key = "mock"
    if mock_transcript is None:
        audio_path, audio_key = extract_audio_range(stream, start, end, workdir, force=force_extract)
        if transcriber == "whisperx":
            transcript, transcript_key, transcript_cache = transcribe_audio(
                audio_path=audio_path,
                workdir=workdir,
                model_name=model,
                language=language,
                device=device,
                compute_type=compute_type,
                batch_size=batch_size,
                vad_filter=vad_filter,
                vad_method=vad_method,
                asr_options=asr_options,
                force=force_transcribe,
            )
        else:
            extra_args = whisper_cpp_extra.split() if whisper_cpp_extra else []
            threads_arg: Optional[int] = whisper_cpp_threads or None
            try:
                transcript, transcript_key, transcript_cache = transcribe_with_whisper_cpp(
                    audio_path=audio_path,
                    workdir=workdir,
                    binary=whisper_cpp_bin,
                    model_path=whisper_cpp_model,
                    language=language,
                    threads=threads_arg,
                    extra_args=extra_args,
                    force=force_transcribe,
                )
            except WhisperCppError as exc:
                raise typer.BadParameter(str(exc), param_hint="--transcriber") from exc
    else:
        logger.info("Использую mock transcript: %s", mock_transcript)
        transcript = load_mock_transcript(mock_transcript)
        transcript_key = stable_hash(
            {
                "stage": "mock_transcript",
                "path": str(mock_transcript.resolve()),
                "content": file_content_hash(mock_transcript),
            }
        )
        transcript_cache = mock_transcript

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
    detection_cache = workdir / "cache" / f"detection_{detection_key}.json"

    if detection_cache.exists() and not force_detect:
        logger.info("Детекция мата уже есть в cache: %s", detection_cache)
        matches = _load_detection_cache(detection_cache)
    else:
        normalizer = RussianNormalizer()
        entries = load_banwords(banwords, normalizer=normalizer)
        logger.info("Загружено banwords: %s", len(entries))
        matches = detect_profanity(transcript, entries, start, normalizer=normalizer)
        _write_detection_cache(detection_cache, detection_key, matches)

    logger.info("Найдено mute-маркеров: %s", len(matches))

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

    write_decisions(decisions, decisions_doc)
    write_vegas_script(vegas, decisions_doc)

    console.print("\n[green]Готово.[/green]")
    console.print(f"decisions.json: [bold]{decisions}[/bold]")
    console.print(f"Vegas script:    [bold]{vegas}[/bold]")
    console.print(f"Mute markers:    [bold]{len(matches)}[/bold]")


@app.command("export")
def export_decisions(
    decisions: Path = typer.Option(..., exists=True, file_okay=True, dir_okay=False, help="Отредактированный decisions.json"),
    vegas: Path = typer.Option(Path("output/vegas_build.cs"), help="Куда записать Vegas C# script"),
) -> None:
    """Regenerate a Vegas script from an edited decisions.json."""

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


# Ожидаемые версии критичных зависимостей — держим синхронно с requirements-gpu.txt.
# doctor сравнивает установленное с этим списком: если разошлось — WARN
# (патчи в transcription.py написаны под именно эти версии).
_EXPECTED_VERSIONS: dict[str, str] = {
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


@app.command()
def doctor() -> None:
    """Проверить окружение: Python, CUDA, версии либ, ffmpeg, кэш моделей."""
    import os
    import shutil
    import sys
    from importlib import metadata

    from .config import MODELS_DIR

    console.print("[bold]twitch-cut doctor[/bold] — проверка окружения\n")

    problems = 0

    # --- Python версия ---
    py_ver = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    if (3, 10) <= sys.version_info[:2] < (3, 13):
        console.print(_fmt_ok(f"Python {py_ver}"))
    else:
        console.print(_fmt_fail(f"Python {py_ver} — нужен 3.10, 3.11 или 3.12"))
        problems += 1

    # --- torch + CUDA ---
    try:
        import torch  # noqa: WPS433

        cuda_avail = torch.cuda.is_available()
        torch_ver = torch.__version__
        if cuda_avail:
            gpu_name = torch.cuda.get_device_name(0)
            cuda_ver = torch.version.cuda
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            console.print(
                _fmt_ok(f"torch {torch_ver}, CUDA {cuda_ver}, GPU: {gpu_name} ({vram_gb:.1f} GB)")
            )
        else:
            console.print(
                _fmt_fail(
                    f"torch {torch_ver}, но CUDA НЕ доступна. "
                    "Скорее всего установлена CPU-версия torch. "
                    "Переустановите:  pip install -r requirements-gpu.txt"
                )
            )
            problems += 1
    except ImportError:
        console.print(_fmt_fail("torch не установлен"))
        problems += 1

    # --- версии критичных либ ---
    for pkg, expected in _EXPECTED_VERSIONS.items():
        try:
            installed = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            console.print(_fmt_fail(f"{pkg} не установлен (ожидалось =={expected})"))
            problems += 1
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
            # WARN, не FAIL — иногда работает и на других версиях.

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

    # --- кэш моделей ---
    if MODELS_DIR.exists():
        size_gb = sum(f.stat().st_size for f in MODELS_DIR.rglob("*") if f.is_file()) / (1024**3)
        console.print(_fmt_ok(f"models dir: {MODELS_DIR} ({size_gb:.2f} GB)"))
        if size_gb < 0.5:
            console.print(
                _fmt_warn(
                    "Моделей меньше 500 MB — вероятно не скачаны. "
                    "Запустите:  twitch-cut prefetch"
                )
            )
    else:
        console.print(
            _fmt_warn(
                f"models dir не создан: {MODELS_DIR}. "
                "Запустите:  twitch-cut prefetch"
            )
        )

    # --- свободное место на диске моделей ---
    disk = shutil.disk_usage(MODELS_DIR.parent if MODELS_DIR.parent.exists() else Path.cwd())
    free_gb = disk.free / (1024**3)
    if free_gb < 6:
        console.print(_fmt_warn(f"Свободно на диске: {free_gb:.1f} GB — WhisperX+модели весят ~4 GB"))
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

    # cuDNN 8 — нужен ctranslate2/whisperx на Windows. torch 2.6+cu126 тянет
    # свой cuDNN 9 в torch\lib, но ctranslate2 ищет `cudnn_ops_infer64_8.dll`
    # через LoadLibrary → нужен либо в PATH, либо в site-packages\nvidia\cudnn\bin
    # (пакет nvidia-cudnn-cu12). Если не найден — backend упадёт с
    # STATUS_STACK_BUFFER_OVERRUN сразу как whisperx попытается загрузить модель.
    if sys.platform == "win32":
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
def prefetch(
    model: str = typer.Option("large-v3", help="WhisperX model для предзагрузки"),
    language: str = typer.Option("ru", help="Язык alignment-модели (wav2vec2)"),
    device: str = typer.Option("cpu", help="Куда временно грузить веса (cpu безопаснее)"),
) -> None:
    """Скачать WhisperX + wav2vec2 alignment + pyannote VAD в ./models/.

    Полезно вызвать один раз при установке — тогда первый `process` не
    будет молча качать 4 GB в середине пайплайна.
    """
    from .config import MODELS_DIR

    setup_logging()

    # Заглушаем verbose HTTP-логи, иначе они топят tqdm-прогресс-бар и
    # пользователь думает что всё зависло. httpx/hf_hub логгеры сами по себе
    # ничего полезного не показывают — реальная информация в tqdm-баре.
    for noisy in ("httpx", "huggingface_hub", "urllib3", "filelock"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    console.print(f"[bold]Prefetch[/bold] моделей в: {MODELS_DIR}\n")
    console.print(
        "[dim]Скачивается ~4 GB. Если прогресс-бар не появляется 2-3 минуты — "
        "убей процесс (Ctrl+C) и проверь размер:\n"
        f"  Get-ChildItem -Recurse {MODELS_DIR} | Measure-Object -Sum Length\n"
        "Если размер растёт — просто ждёшь; если нет — открой issue.[/dim]\n"
    )

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
    try:
        console.print(f"→ Скачиваю alignment-модель для '{language}' (~1 GB)...")
        whisperx.load_align_model(language_code=language, device=device)
        console.print(_fmt_ok(f"alignment ({language})"))
    except Exception as exc:  # noqa: BLE001
        console.print(_fmt_fail(f"alignment: {exc}"))
        raise typer.Exit(1) from exc

    # 3. pyannote VAD — грузится внутри whisperx.load_model выше, отдельно
    # предзагружать не нужно. Если бы понадобилось — pyannote.audio.Pipeline
    # требует HF token, а WhisperX использует форк без токена.

    console.print()
    console.print("[bold green]Готово.[/bold green] Модели закэшированы, первый прогон не будет скачивать.")


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
        hs = parse_response(
            response, transcript_range_s=range_s, transcript_hash=transcript_hash,
        )
    except MergeError as exc:
        console.print("[red]Не могу смержить highlights:[/red]")
        for r in exc.reasons:
            console.print(f"  • {r}")
        raise typer.Exit(code=1)

    merged = merge_into_decisions(decisions_doc, hs)
    write_json(output, merged)

    console.print(f"[green]Highlights смержены.[/green] → [bold]{output}[/bold]")
    console.print(f"count:  [bold]{len(hs.highlights)}[/bold]")
    for h in hs.highlights:
        console.print(
            f"  • [{h.start_s:.1f}s → {h.end_s:.1f}s] "
            f"(score={h.score:.2f}) [bold]{h.title}[/bold]"
        )


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

