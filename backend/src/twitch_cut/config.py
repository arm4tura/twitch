from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# =============================================================================
# Пути к данным (модели и т.п.).
#
# Мы храним HuggingFace-кэш ЛОКАЛЬНО (./models/) вместо дефолтного
# ~/.cache/huggingface/. Причины:
#   1. Пользователю очевидно, где лежат 4 GB весов — можно посмотреть/удалить.
#   2. Не мешает другим проектам того же пользователя.
#   3. `twitch-cut prefetch` кладёт всё в предсказуемое место — можно заранее
#      скачать на машине с быстрым интернетом, а потом скопировать models/
#      на офлайн-машину монтажёра.
#
# Корень данных (DATA_ROOT):
#   - В dev (запуск из репо) — корень репозитория. Считается от этого файла:
#       backend/src/twitch_cut/config.py  →  parents[3] = <repo>/
#   - В упакованном приложении сам код лежит в read-only каталоге
#     (Program Files\...\resources\backend) — туда писать НЕЛЬЗЯ. Electron
#     задаёт TWITCH_CUT_DATA_DIR, указывающий на writable-каталог данных
#     (userData или portable-папка рядом с exe). Тогда models/ живёт там.
# =============================================================================
def _resolve_data_root() -> Path:
    override = os.environ.get("TWITCH_CUT_DATA_DIR")
    if override:
        return Path(override).expanduser().resolve()
    # dev-фолбэк: корень репозитория (backend/src/twitch_cut/config.py → parents[3]).
    return Path(__file__).resolve().parents[3]


DATA_ROOT: Path = _resolve_data_root()
MODELS_DIR: Path = (DATA_ROOT / "models").resolve()


def _mps_available() -> bool:
    """Есть ли рабочий Apple Metal (MPS) backend у torch.

    Импорт torch тяжёлый — зовём эту функцию только на darwin и только когда
    реально выбираем device (не на импорте модуля). Любая ошибка → False:
    лучше тихо уехать на CPU, чем упасть на детекте.
    """
    if sys.platform != "darwin":
        return False
    try:
        import torch

        return bool(torch.backends.mps.is_available())
    except Exception:  # noqa: BLE001
        return False


def default_device() -> str:
    """Устройство по умолчанию.

    Приоритет:
      1. TWITCH_CUT_DEVICE — явный оверрайд (cuda|cpu|mps), уважаем как есть;
      2. TWITCH_CUT_CPU=1 — bootstrap выставляет, если NVIDIA GPU не найден
         (Windows/Linux без CUDA) → форсим CPU;
      3. macOS с рабочим MPS → 'mps' (Apple Silicon, ускорение GigaAM);
      4. иначе — 'cuda' (Windows/Linux с NVIDIA).

    Важно: MPS реально ускоряет только GigaAM (torch-путь). whisperx/whispercpp
    идут через ctranslate2, который MPS не умеет — для них PipelineConfig.validate
    запретит mps, а вызывающий код должен откатиться на cpu.
    """
    override = os.environ.get("TWITCH_CUT_DEVICE")
    if override:
        return override
    if os.environ.get("TWITCH_CUT_CPU") == "1":
        return "cpu"
    if _mps_available():
        return "mps"
    return "cuda"


def default_compute_type() -> str:
    """compute_type для faster-whisper. На CPU float16 не поддерживается —
    берём int8 (быстрее и работает); на GPU остаётся float16.

    На MPS этот параметр не используется (GigaAM не читает compute_type
    faster-whisper), но держим валидное значение — float16, как на GPU."""
    if os.environ.get("TWITCH_CUT_DEVICE") == "mps" or (
        os.environ.get("TWITCH_CUT_CPU") != "1" and _mps_available()
    ):
        return "float16"
    return "int8" if os.environ.get("TWITCH_CUT_CPU") == "1" else "float16"


def default_banwords_path() -> Path:
    """Встроенный словарь мата.

    Используется «простым режимом» UI, когда пользователь не выбирал свой
    файл банвордов, — чтобы человеку не нужно было искать/составлять словарь
    руками. Лежит в backend/banwords.txt (рядом с исходниками; копируется в
    resources/backend при сборке installer'а). Fallback — banwords.example.txt.
    """
    base = Path(__file__).resolve().parents[2]  # .../backend
    for name in ("banwords.txt", "banwords.example.txt"):
        candidate = base / name
        if candidate.exists():
            return candidate
    return base / "banwords.txt"


def configure_hf_cache() -> Path:
    """Выставить HF_HOME до первого импорта whisperx/torch/pyannote.

    Вызывать ОБЯЗАТЕЛЬНО в самом начале точки входа (cli.py: до всех остальных
    импортов из twitch_cut) — HuggingFace-библиотеки читают HF_HOME один раз
    при импорте, поменять его позже не получится.

    Идемпотентно. Если пользователь уже выставил HF_HOME снаружи (например,
    хочет общий кэш) — уважаем его выбор и ничего не трогаем.
    """
    if os.environ.get("HF_HOME"):
        return Path(os.environ["HF_HOME"])
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(MODELS_DIR)
    # Старые версии huggingface_hub читают TRANSFORMERS_CACHE отдельно.
    os.environ.setdefault("TRANSFORMERS_CACHE", str(MODELS_DIR))
    # torch.hub тоже уводим в тот же корень — pyannote иногда через него грузит.
    os.environ.setdefault("TORCH_HOME", str(MODELS_DIR / "torch"))
    # Отключаем hf_xet (экспериментальный CDN-протокол HuggingFace).
    # На Windows без Developer Mode/админ-прав он стабильно виснет на первом
    # chunk крупных LFS-файлов (наблюдали на faster-whisper-large-v3 ~3 GB) —
    # прогресс-бар вообще не появляется, потому что xet использует свой
    # download-путь мимо стандартного tqdm. Стандартный HTTP-скачок надёжнее.
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    # Windows без Developer Mode не умеет в symlinks — HF валит предупреждение
    # на каждую загрузку. Реального вреда нет (файлы копируются вместо
    # симлинков), только мусор в логах.
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    return MODELS_DIR


# ASR-параметры faster-whisper по умолчанию настроены на защиту от
# hallucination drift: whisper из-за condition_on_previous_text=True (родной
# дефолт faster-whisper) склонен «додумывать» текст на длинных стримах,
# сжимая тайминги на 5+ секунд и теряя целые фразы. Отключаем context-carry
# и добавляем temperature fallback + порог no_speech, которые уже прошли
# проверку в Colab-эксперименте и дали стабильные тайминги.
DEFAULT_ASR_OPTIONS: dict[str, Any] = {
    "condition_on_previous_text": False,
    "no_speech_threshold": 0.6,
    "temperatures": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
    "beam_size": 5,
    "best_of": 5,
}


@dataclass(frozen=True)
class PipelineConfig:
    # Движок ASR:
    #   'gigaam'    — GigaAM v3 (Salute), DEFAULT. Лучше ловит русский мат,
    #                 узкие пословные тайминги, не требует CTranslate2/cuDNN.
    #   'whisperx'  — faster-whisper через WhisperX (PyTorch+CUDA).
    #   'whispercpp'— subprocess whisper-cli.exe.
    transcriber: str = "gigaam"
    language: str = "ru"
    model: str = "large-v3"
    # Авто-детект: cuda (NVIDIA) / mps (Apple Silicon) / cpu — см. default_device.
    device: str = field(default_factory=default_device)
    compute_type: str = field(default_factory=default_compute_type)
    batch_size: int = 16
    vad_filter: bool = True
    # 'pyannote' даёт более стабильные границы речи для длинных стримов;
    # silero режет агрессивнее и на практике коррелирует с hallucination-drift
    # (Whisper теряет 5+ секунд речи). Оставляем возможность переключить.
    vad_method: str = "pyannote"
    # Опции faster-whisper. По умолчанию — anti-hallucination profile
    # (см. DEFAULT_ASR_OPTIONS). Если WhisperX старой версии не поддерживает
    # asr_options — код в transcription.py делает graceful fallback.
    asr_options: dict[str, Any] = field(default_factory=lambda: dict(DEFAULT_ASR_OPTIONS))
    mute_padding_before_ms: int = 80
    mute_padding_after_ms: int = 120
    # 'word' — мьютим только слово-мат (точно по word-level таймингам). Default.
    #          Подходит, когда whisper/whisper.cpp выдают точные тайминги слов
    #          (--output-json-full). Padding ±80/120мс ловит хвост гласной.
    # 'segment-tail' — мьютим от слова до конца Whisper-сегмента (фразы).
    #          Был дефолтом, пока тайминги слов были ненадёжны. Сейчас
    #          раздувает мьют на 5+ секунд чистого текста, если в сегменте
    #          мат только в начале. Использовать только если word-level
    #          таймингов нет вообще.
    mute_extend_mode: str = "word"
    # Жёсткий потолок длительности одного мьюта, секунды. Защита от
    # аномально длинных whisper-сегментов (например, 30-секундная "фраза").
    mute_max_seconds: float = 6.0
    # Максимальный разрыв (мс) между двумя соседними матами, при котором их
    # склеивают в ОДИН мьют. Если между матами больше — это два отдельных
    # мьюта, а чистая речь между ними НЕ мьютится.
    #
    # Раньше группировка склеивала все маты одного whisper-сегмента в одно
    # окно first..last — и если внутри фразы было два мата с предложением
    # чистой речи между ними ('блять ... убил школьницу ... блять'), то
    # мьютилось всё предложение целиком. 600мс ≈ типичная пауза-вдох между
    # словами; маты в одном выкрике ('бля блять нахуй') идут плотнее.
    mute_join_gap_ms: int = 600
    # Максимальная правдоподобная длительность ОДНОГО слова, секунды.
    # whisper.cpp иногда выдаёт битый word-timing: последнему слову сегмента
    # ставит end = конец всего сегмента (слово 'блять' → 9.48с). При
    # mute_extend_mode='word' это раздувало мьют до потолка. Если слово
    # длиннее порога — обрезаем его до start + mute_max_word_seconds.
    mute_max_word_seconds: float = 1.5
    # Диагностический режим: мьютить СТРОГО по таймингам whisper для каждого
    # отдельного банворда, без padding, без склейки соседних, без extend,
    # без word-cap и без segment-cap. Нужен чтобы понять, что именно врёт —
    # сам whisper или наша обработка вокруг его таймингов.
    raw_mute: bool = False

    def normalized(self) -> "PipelineConfig":
        """Вернуть копию с device, приведённым под платформу и движок.

        Вызывать ДО validate(). default_device() на маке возвращает 'mps' не
        зная транскрайбера; если пользователь при этом выбрал whisperx/whispercpp
        (которые MPS не умеют), молча откатываемся на 'cpu' с варнингом, а не
        роняем job валидацией. Явный device='cuda' на маке тоже даунгрейдим —
        CUDA на darwin нет. Датакласс frozen, поэтому возвращаем новый инстанс.
        """
        import dataclasses
        import logging

        log = logging.getLogger(__name__)
        device = self.device
        if device == "mps" and self.transcriber != "gigaam":
            log.warning(
                "device=mps не поддерживается транскрайбером %s → откат на cpu",
                self.transcriber,
            )
            device = "cpu"
        if device == "cuda" and sys.platform == "darwin":
            log.warning("device=cuda недоступен на macOS → откат на cpu")
            device = "cpu"
        if device == self.device:
            return self
        return dataclasses.replace(self, device=device)

    def validate(self) -> None:
        if self.device not in {"cuda", "cpu", "mps"}:
            raise ValueError("device must be 'cuda', 'cpu', or 'mps'")
        # MPS (Apple Metal) ускоряет только torch-путь GigaAM. whisperx идёт
        # через ctranslate2, whispercpp — через whisper-cli; ни тот, ни другой
        # MPS не понимают. Ловим тут, чтобы не свалиться внутри загрузки модели.
        if self.device == "mps" and self.transcriber != "gigaam":
            raise ValueError(
                f"device 'mps' поддерживается только для transcriber 'gigaam', "
                f"а не '{self.transcriber}' — используйте 'cpu'"
            )
        if self.compute_type not in {"float16", "float32", "int8"}:
            raise ValueError("compute_type must be float16, float32, or int8")
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")
        if self.vad_method not in {"silero", "pyannote"}:
            raise ValueError("vad_method must be 'silero' or 'pyannote'")
        if self.mute_padding_before_ms < 0 or self.mute_padding_after_ms < 0:
            raise ValueError("mute padding cannot be negative")
        if self.mute_extend_mode not in {"word", "segment-tail"}:
            raise ValueError("mute_extend_mode must be 'word' or 'segment-tail'")
        if self.mute_max_seconds <= 0:
            raise ValueError("mute_max_seconds must be positive")
        if self.mute_join_gap_ms < 0:
            raise ValueError("mute_join_gap_ms cannot be negative")
        if self.mute_max_word_seconds <= 0:
            raise ValueError("mute_max_word_seconds must be positive")
        if self.transcriber not in {"gigaam", "whisperx", "whispercpp"}:
            raise ValueError("transcriber must be 'gigaam', 'whisperx' or 'whispercpp'")
