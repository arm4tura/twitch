"""ONNX-модели: где лежат, откуда качаются, как открываются.

ЗАЧЕМ. Распознавание (GigaAM v3_ctc) и VAD (pyannote segmentation-3.0) идут
через onnxruntime, а не torch: это сотни мегабайт рантайма вместо гигабайтов
torch+CUDA, и никаких cuDNN-DLL. Сами .onnx получаются из torch-чекпоинтов
один раз (`tools/export_onnx.py`) и лежат в GitHub-релизе моделей проекта;
`twitch-cut prefetch` скачивает их в `models/onnx/` и сверяет sha256.

УСКОРИТЕЛЬ. `device` проекта ('cuda' | 'mps' | 'cpu') переводится в execution
provider onnxruntime:
  - 'cuda' — GPU на Windows/Linux: CUDA EP (если стоит onnxruntime-gpu) или
    DirectML (onnxruntime-directml: любая видеокарта под Windows — NVIDIA,
    AMD, Intel — и никаких CUDA/cuDNN);
  - 'mps'  — Apple: CoreML EP;
  - 'cpu'  — CPU.
Не поднялся ускоритель — тихо уходим на CPU с варнингом: медленнее, но прогон
не падает.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

from .config import MODELS_DIR

logger = logging.getLogger(__name__)

META_KEY = "twitch_cut"
"""Ключ metadata_props, под которым в .onnx лежит JSON-описание модели."""

GIGAAM_ONNX_NAME = "gigaam_v3_ctc.onnx"
VAD_ONNX_NAME = "vad_segmentation3.onnx"

MODELS_RELEASE_URL = "https://github.com/arm4tura/twitch/releases/download/models-v2"
"""GitHub-релиз с ONNX-моделями: без токенов и gated-репозиториев HF."""


@dataclass(frozen=True)
class ModelFile:
    name: str
    sha256: str
    size_mb: int
    """Примерный размер — только для сообщений пользователю."""

    @property
    def path(self) -> Path:
        return onnx_dir() / self.name

    @property
    def url(self) -> str:
        return f"{MODELS_RELEASE_URL}/{self.name}"


# sha256 — ровно тех файлов, что выдал tools/export_onnx.py и что лежат в релизе.
# Перевыгрузили модели → обновить хеши здесь, иначе prefetch их не примет.
GIGAAM_MODEL = ModelFile(
    name=GIGAAM_ONNX_NAME,
    sha256="b8e05989cf9e559509376aa0d7dc3109673687789ad91f13975eae586fa1ceda",
    size_mb=424,
)
VAD_MODEL = ModelFile(
    name=VAD_ONNX_NAME,
    sha256="26c48e66d33d72c3e349bc9209bd5ad50f9ae58abf819b6fa3b72c64f82cd52e",
    size_mb=6,
)
ALL_MODELS: tuple[ModelFile, ...] = (VAD_MODEL, GIGAAM_MODEL)

DownloadProgressFn = Callable[[str, int, int], None]
"""(имя файла, скачано байт, всего байт; 0 — размер неизвестен)."""


def onnx_dir() -> Path:
    return MODELS_DIR / "onnx"


def is_present(model: ModelFile) -> bool:
    path = model.path
    return path.exists() and path.stat().st_size > 0


def ensure_model(
    model: ModelFile,
    *,
    on_progress: Optional[DownloadProgressFn] = None,
    retries: int = 3,
) -> Path:
    """Вернуть путь к модели, скачав её при необходимости.

    Качаем во временный `.part` и переименовываем только после сверки sha256 —
    оборванный или подменённый файл не должен выглядеть как готовый кэш.
    Модель, уже лежащую на диске, не перепроверяем: хеш 400 МБ на каждом
    запуске — лишние секунды, а битой она могла стать только руками.
    """
    target = model.path
    if is_present(model):
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".part")
    last_exc: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            _download(model.url, tmp, model.name, on_progress)
            digest = _sha256(tmp)
            if model.sha256 and digest != model.sha256:
                raise OSError(
                    f"контрольная сумма не сошлась: {digest[:12]}… вместо {model.sha256[:12]}…"
                )
            tmp.replace(target)
            return target
        except Exception as exc:  # noqa: BLE001 — сеть/диск; пробуем ещё раз
            last_exc = exc
            tmp.unlink(missing_ok=True)
            logger.warning(
                "Не удалось скачать %s (попытка %d/%d): %s", model.name, attempt, retries, exc
            )
            if attempt < retries:
                time.sleep(2 * attempt)
    raise RuntimeError(f"Не удалось скачать модель {model.name}: {last_exc}") from last_exc


def _download(
    url: str, dest: Path, name: str, on_progress: Optional[DownloadProgressFn]
) -> None:
    # CA-bundle из certifi: собранный бэкенд на macOS системных сертификатов
    # не видит (см. fetch_tdcli._ssl_context).
    from .fetch_tdcli import _ssl_context

    with urllib.request.urlopen(url, timeout=60, context=_ssl_context()) as response, dest.open(
        "wb"
    ) as out:
        total = int(response.headers.get("Content-Length") or 0)
        done = 0
        while True:
            block = response.read(1 << 20)
            if not block:
                break
            out.write(block)
            done += len(block)
            if on_progress is not None:
                on_progress(name, done, total)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


# --- сессии onnxruntime ------------------------------------------------------


def _providers_for(device: str, available: list[str]) -> list[Any]:
    """Список execution providers под `device` проекта, CPU всегда последним."""
    wanted: list[Any] = []
    if device == "cuda":
        if "CUDAExecutionProvider" in available:
            wanted.append(("CUDAExecutionProvider", {"cudnn_conv_algo_search": "HEURISTIC"}))
        if "DmlExecutionProvider" in available:
            wanted.append("DmlExecutionProvider")
    elif device == "mps":
        if "CoreMLExecutionProvider" in available:
            # Кэш скомпилированной CoreML-модели: без него 400 МБ графа
            # перекомпилируются при каждом открытии сессии. Neural Engine не
            # берём (CPUAndGPU, а не ALL): он считает только во float16, а
            # GPU с float32-накоплением держит log-probs ближе к CPU-пути.
            cache_dir = MODELS_DIR / "coreml_cache"
            cache_dir.mkdir(parents=True, exist_ok=True)
            wanted.append(
                (
                    "CoreMLExecutionProvider",
                    {
                        "ModelFormat": "MLProgram",
                        "MLComputeUnits": "CPUAndGPU",
                        "ModelCacheDirectory": str(cache_dir),
                    },
                )
            )
    wanted.append("CPUExecutionProvider")
    return wanted


def accelerator_label(device: str) -> str:
    """Что реально получит `device` на этой машине — для doctor и логов."""
    import onnxruntime as ort

    first = _providers_for(device, ort.get_available_providers())[0]
    name = first[0] if isinstance(first, tuple) else first
    return name.replace("ExecutionProvider", "")


def create_session(path: Path, device: str) -> Any:
    """InferenceSession под `device`. Ускоритель не поднялся → CPU."""
    import onnxruntime as ort

    providers = _providers_for(device, ort.get_available_providers())

    def _options(dml: bool) -> Any:
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.log_severity_level = 3
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        if dml:
            # DirectML не умеет memory pattern — без этого флага сессия не создаётся.
            opts.enable_mem_pattern = False
        threads = os.environ.get("TWITCH_CUT_ORT_THREADS")
        if threads and threads.isdigit():
            opts.intra_op_num_threads = int(threads)
        return opts

    dml = "DmlExecutionProvider" in providers
    try:
        session = ort.InferenceSession(str(path), sess_options=_options(dml), providers=providers)
    except Exception as exc:  # noqa: BLE001 — драйвер/EP; CPU есть всегда
        if providers == ["CPUExecutionProvider"]:
            raise
        logger.warning("%s: ускоритель не поднялся (%s) — работаю на CPU", path.name, exc)
        session = ort.InferenceSession(
            str(path), sess_options=_options(False), providers=["CPUExecutionProvider"]
        )
    logger.info("%s: onnxruntime %s", path.name, ", ".join(session.get_providers()))
    return session


def read_meta(session: Any) -> dict[str, Any]:
    """JSON-описание модели из metadata_props (кладёт tools/export_onnx.py)."""
    raw = session.get_modelmeta().custom_metadata_map.get(META_KEY)
    if not raw:
        raise RuntimeError(
            "В ONNX-модели нет описания twitch_cut — файл не из tools/export_onnx.py"
        )
    return json.loads(raw)
