"""Транскрипт исходного видео — ground truth для привязки (align.py).

ГЛАВНОЕ ТРЕБОВАНИЕ: исходник распознаётся ТЕМ ЖЕ движком, моделью и
устройством, что и стрим. Привязка работает на точных совпадениях шинглов из 4
нормализованных слов; два разных ASR запишут одну и ту же фразу по-разному
(«щас» / «сейчас», «айфон» / «iPhone»), совпадений не будет, и расщепление
дорожек молча выродится в «всё сказал стример» — без единой ошибки в логе.
Поэтому здесь только GigaAM, тот же путь, что у стрима
(`gigaam_asr.transcribe_with_gigaam`), а модель и устройство обязан передать
вызывающий, взяв их из прогона стрима.

ВХОД — локальный файл (mp3/mp4/wav/mkv — что прочитает ffmpeg). Скачивание по
ссылке (yt-dlp и подобное) здесь сознательно не делается: это отдельная задача
со своими ошибками, прокси и возрастными ограничениями. Ссылку отвергаем с
внятным сообщением, а не пытаемся угадать.

Аудио всегда прогоняется через ffmpeg в 16 кГц моно PCM — тот же формат, что у
стрима (`ffmpeg_tools.extract_audio_range`), чтобы на входе модели не было
разницы в частоте дискретизации. Оба шага кэшируются по отпечатку файла, так что
повторный запуск джобы исходник не перераспознаёт (а это единственная дорогая
стадия здесь — минуты GPU против секунд на всё остальное).
"""

from __future__ import annotations

import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

from ..cache import file_fingerprint, stable_hash, write_json
from ..ffmpeg_tools import get_ffmpeg_path

logger = logging.getLogger(__name__)

# Тот же формат, что у стрима: 16 кГц, моно, PCM s16le.
SOURCE_AUDIO_FORMAT = "wav_pcm_s16le_16khz_mono"

# Модель по умолчанию — как у стрима (см. gigaam_asr._normalize_gigaam_model).
DEFAULT_SOURCE_MODEL = "v3_ctc"

# Доля прогресса, отданная извлечению аудио; остальное — транскрипция.
_EXTRACT_PROGRESS_SHARE = 0.05

_URL_PREFIXES = ("http://", "https://", "rtmp://", "rtmps://", "//")

# (fraction 0..1, message) — та же форма, что gigaam_asr.TranscribeProgressFn.
SourceProgressFn = Callable[[float, str], None]


class SourceError(ValueError):
    """С исходником что-то не так: ссылка вместо файла, файла нет, пустой ASR."""


@dataclass(frozen=True)
class SourceTranscript:
    """Распознанный исходник + пути, по которым его можно найти повторно."""

    transcript: dict[str, Any]
    key: str
    transcript_path: Path
    audio_path: Path
    source_path: Path

    @property
    def segments(self) -> list[dict[str, Any]]:
        return list(self.transcript.get("segments") or [])

    @property
    def word_count(self) -> int:
        return sum(len(s.get("words") or []) for s in self.segments)

    @property
    def duration_s(self) -> float:
        """Конец последнего сегмента. 0.0 для пустого транскрипта."""
        segs = self.segments
        if not segs:
            return 0.0
        try:
            return round(float(segs[-1].get("end") or 0.0), 3)
        except (TypeError, ValueError):
            return 0.0


def resolve_source_path(source: str | Path) -> Path:
    """Проверить, что на входе локальный файл, и вернуть абсолютный путь.

    Ссылки отвергаем явно: скачивание — не наша задача, а молчаливое падение
    ffmpeg на URL даёт непонятную пользователю ошибку.
    """
    raw = str(source).strip()
    if not raw:
        raise SourceError("путь к исходнику пуст")
    lowered = raw.lower()
    if lowered.startswith(_URL_PREFIXES):
        raise SourceError(
            "исходник задан ссылкой — скачивание не поддерживается. "
            "Скачайте видео/аудио сами (например, yt-dlp) и укажите файл на диске"
        )
    path = Path(raw).expanduser()
    try:
        path = path.resolve()
    except OSError as exc:  # noqa: BLE001 - битый путь/симлинк
        raise SourceError(f"не удалось разобрать путь к исходнику: {exc}") from exc
    if not path.exists():
        raise SourceError(f"файл исходника не найден: {path}")
    if not path.is_file():
        raise SourceError(f"исходник должен быть файлом, а не каталогом: {path}")
    return path


def extract_source_audio(
    source_path: Path, workdir: Path, *, force: bool = False
) -> tuple[Path, str]:
    """Исходник → 16 кГц моно wav. Возвращает (путь, ключ кэша).

    Своя функция, а не `extract_audio_range`: там обязательны границы диапазона,
    а исходник нужен целиком, и лишние `-ss/-t` только добавили бы шанс
    рассинхрона таймингов с индексом слов.
    """
    extract_dir = workdir / "extracted"
    cache_dir = workdir / "cache"
    extract_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    key_data = {
        "stage": "extract_source_audio",
        "source": file_fingerprint(source_path),
        "format": SOURCE_AUDIO_FORMAT,
    }
    key = stable_hash(key_data)
    wav_path = extract_dir / f"source_{key}.wav"
    meta_path = cache_dir / f"extract_source_{key}.json"

    if wav_path.exists() and meta_path.exists() and not force:
        logger.info("source: аудио исходника уже извлечено: %s", wav_path)
        return wav_path, key

    ffmpeg = get_ffmpeg_path()
    logger.info("source: извлекаю аудио исходника через ffmpeg -> %s", wav_path.name)
    cmd = [
        ffmpeg,
        "-y",
        "-i",
        str(source_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-acodec",
        "pcm_s16le",
        str(wav_path),
    ]
    try:
        subprocess.run(
            cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or "").strip()
        # Хвост stderr: ffmpeg пишет туда баннер и мету, причина всегда в конце.
        raise SourceError(
            f"ffmpeg не смог извлечь аудио из исходника: {stderr[-500:]}"
        ) from exc

    write_json(
        meta_path,
        {
            "stage": "extract_source_audio",
            "key": key,
            "inputs": key_data,
            "output": str(wav_path),
        },
    )
    return wav_path, key


def transcribe_source(
    source: str | Path,
    workdir: Path,
    *,
    model_name: str = DEFAULT_SOURCE_MODEL,
    device: str = "cpu",
    force: bool = False,
    on_progress: Optional[SourceProgressFn] = None,
) -> SourceTranscript:
    """Распознать исходник тем же GigaAM, что и стрим.

    :param model_name: та же модель, что у стрима (иначе тексты несравнимы).
    :param device: то же устройство, что у стрима.
    :param force: перераспознать, игнорируя кэш.

    GigaAM импортируется лениво — модуль обязан оставаться импортируемым там,
    где torch не установлен (валидация настроек, тесты промптов).
    """
    path = resolve_source_path(source)
    workdir = Path(workdir)

    def tick(fraction: float, message: str) -> None:
        if on_progress is not None:
            on_progress(max(0.0, min(1.0, fraction)), message)

    tick(0.0, f"исходник: {path.name}")
    audio_path, _ = extract_source_audio(path, workdir, force=force)
    tick(_EXTRACT_PROGRESS_SHARE, "аудио исходника готово")

    from ..gigaam_asr import transcribe_with_gigaam

    def _asr_progress(fraction: float, message: str = "") -> None:
        span = 1.0 - _EXTRACT_PROGRESS_SHARE
        tick(_EXTRACT_PROGRESS_SHARE + fraction * span, message)

    transcript, key, transcript_path = transcribe_with_gigaam(
        audio_path=audio_path,
        workdir=workdir,
        model_name=model_name,
        device=device,
        force=force,
        on_progress=_asr_progress,
    )
    tick(1.0, "исходник распознан")

    result = SourceTranscript(
        transcript=transcript,
        key=key,
        transcript_path=transcript_path,
        audio_path=audio_path,
        source_path=path,
    )
    if result.word_count == 0:
        # Дальше молча получилось бы «привязок 0» на всём стриме, то есть
        # «стример говорит сам» — и оффтопом стал бы весь VOD.
        raise SourceError(
            f"в транскрипте исходника нет слов с таймингами ({transcript_path}) — "
            "проверьте, что в файле есть речь"
        )
    logger.info(
        "source: %s — %d сегментов, %d слов, %.1f мин (модель %s, устройство %s)",
        path.name,
        len(result.segments),
        result.word_count,
        result.duration_s / 60.0,
        model_name,
        device,
    )
    return result
