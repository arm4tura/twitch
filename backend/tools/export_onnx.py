"""Экспорт GigaAM v3_ctc и VAD pyannote segmentation-3.0 в ONNX.

Зачем. Рабочее приложение больше не тянет torch: распознавание и VAD идут
через onnxruntime (`gigaam_onnx.py`, `vad.py`). Сами ONNX-файлы получаются
ОДИН раз из исходных torch-чекпоинтов — этим скриптом, на машине разработчика,
где torch/gigaam/pyannote ещё стоят (`pip install -e ".[export]"`). Готовые
файлы заливаются в GitHub-релиз моделей, оттуда их качает `twitch-cut prefetch`.

Каждый .onnx самоописывающий: в `metadata_props` под ключом `twitch_cut` лежит
JSON со всем, что нужно рантайму без torch — словарь и blank у GigaAM,
параметры мел-спектрограммы, окно/шаг/рецептивное поле у VAD. Так файл модели
и его параметры не могут разъехаться.

Запуск (из backend/):
    python tools/export_onnx.py                  # в ../models/onnx
    python tools/export_onnx.py --out /tmp/onnx  # куда угодно

После экспорта:
    python tools/onnx_parity.py <аудио>          # сверка с torch-путём
и залить оба .onnx в релиз (имена файлов — в `twitch_cut/onnx_models.py`).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twitch_cut.config import MODELS_DIR  # noqa: E402
from twitch_cut.onnx_models import (  # noqa: E402
    GIGAAM_ONNX_NAME,
    META_KEY,
    VAD_ONNX_NAME,
)

GIGAAM_CKPT = MODELS_DIR / "gigaam" / "v3_ctc.ckpt"
VAD_CKPT = MODELS_DIR / "vad" / "segmentation-3.0.bin"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _attach_meta(path: Path, meta: dict) -> None:
    """Положить JSON-описание модели в metadata_props самого .onnx."""
    import onnx

    model = onnx.load(str(path))
    del model.metadata_props[:]
    entry = model.metadata_props.add()
    entry.key = META_KEY
    entry.value = json.dumps(meta, ensure_ascii=False)
    onnx.save(model, str(path))


def _store_weights_fp16(path: Path, min_elements: int = 1024) -> int:
    """Хранить веса в .onnx как float16, считать — по-прежнему во float32.

    Веса GigaAM в чекпоинте уже лежат во float16 (420 МБ), а torch-экспорт
    пишет их float32 — файл вырастает вдвое. Поэтому каждый крупный
    float32-инициализатор, который ТОЧНО представим во float16, заменяем на
    float16-копию + узел Cast→float32 перед графом. onnxruntime сворачивает
    такой Cast при создании сессии (constant folding), то есть считает граф с
    теми же float32-весами, бит в бит. Инициализаторы, которые экспорт уже
    пересчитал (свёртка BatchNorm в веса и т.п.) и которые во float16 не
    влезают без потерь, остаются float32.

    Возвращает число переведённых инициализаторов.
    """
    import numpy as np
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    model = onnx.load(str(path))
    graph = model.graph
    casts = []
    kept = []
    converted = 0
    for init in graph.initializer:
        if init.data_type != TensorProto.FLOAT:
            kept.append(init)
            continue
        weights = numpy_helper.to_array(init)
        if weights.size < min_elements:
            kept.append(init)
            continue
        half = weights.astype(np.float16)
        if not np.array_equal(half.astype(np.float32), weights):
            kept.append(init)
            continue
        stored = f"{init.name}__fp16"
        kept.append(numpy_helper.from_array(half, stored))
        casts.append(helper.make_node("Cast", [stored], [init.name], to=TensorProto.FLOAT))
        converted += 1
    del graph.initializer[:]
    graph.initializer.extend(kept)
    nodes = list(graph.node)
    del graph.node[:]
    graph.node.extend(casts + nodes)
    onnx.save(model, str(path))
    return converted


def export_gigaam(out_dir: Path) -> Path:
    """GigaAM v3_ctc: энкодер + CTC-голова одним графом, на выходе log-probs.

    Штатный экспорт gigaam (`GigaAMASR._to_onnx`): входы — мел-признаки
    [B, 64, T] и их длины, выходы — log_probs [B, T', 34] и длины после
    сабсэмплинга. Мел-спектрограмма в граф НЕ входит: её считает numpy
    (`gigaam_onnx.log_mel`), так граф остаётся тем, что поддерживает gigaam.
    """
    import gigaam

    # Имя, а не путь к .ckpt: путь gigaam считает дообученным чекпоинтом
    # другого формата. Лежащий в download_root файл он возьмёт без скачивания.
    model = gigaam.load_model(
        "v3_ctc",
        fp16_encoder=False,
        device="cpu",
        download_root=str(GIGAAM_CKPT.parent),
    )
    model.eval()
    cfg = model.cfg
    pre = cfg.preprocessor
    with tempfile.TemporaryDirectory() as tmp:
        model.to_onnx(tmp)
        produced = Path(tmp) / f"{cfg.model_name}.onnx"
        target = out_dir / GIGAAM_ONNX_NAME
        target.write_bytes(produced.read_bytes())
    converted = _store_weights_fp16(target)
    print(f"{target.name}: {converted} инициализаторов хранятся во float16")

    vocab = [str(ch) for ch in cfg.decoding.vocabulary]
    # Окно Ханна и мел-фильтры берём из САМОЙ модели, а не считаем заново:
    # в чекпоинте они лежат округлёнными до float16, и torch-путь считает
    # признаки именно по ним. Посчитанное «честное» окно гасит утечку спектра
    # лучше, и в тихих полосах (низ и верх спектра) лог-мел уезжает на единицы —
    # модель такого входа не видела, текст начинает расходиться.
    featurizer = model.preprocessor.featurizer[0]
    window = featurizer.spectrogram.window.detach().float().cpu().numpy()
    mel_fb = featurizer.mel_scale.fb.detach().float().cpu().numpy()
    meta = {
        "format": 1,
        "kind": "gigaam_ctc",
        "model_name": str(cfg.model_name),
        "sample_rate": int(pre.sample_rate),
        "features": int(pre.features),
        "n_fft": int(pre.get("n_fft", pre.sample_rate // 40)),
        "win_length": int(pre.get("win_length", pre.sample_rate // 40)),
        "hop_length": int(pre.get("hop_length", pre.sample_rate // 100)),
        "center": bool(pre.get("center", True)),
        "vocab": vocab,
        "blank_id": len(vocab),
        # float32 → float64 → JSON → float32 восстанавливается без потерь.
        "window": [float(x) for x in window],
        "mel_fb": [[float(x) for x in row] for row in mel_fb],
        "source_md5": hashlib.md5(GIGAAM_CKPT.read_bytes()).hexdigest()
        if GIGAAM_CKPT.exists()
        else None,
    }
    _attach_meta(target, meta)
    return target


def export_vad(out_dir: Path) -> Path:
    """pyannote segmentation-3.0 (PyanNet): окно 10 с сырого звука → logits
    powerset-классов по кадрам. Остальной VAD-пайплайн (окна, агрегация,
    гистерезис) повторён на numpy в `vad.py`, параметры — в метаданных."""
    import torch
    from pyannote.audio import Model
    from pyannote.audio.core.task import Problem, Resolution, Specifications
    from pyannote.audio.utils.powerset import Powerset
    from torch.torch_version import TorchVersion

    with torch.serialization.safe_globals([TorchVersion, Problem, Specifications, Resolution]):
        model = Model.from_pretrained(str(VAD_CKPT))
    model.eval()
    spec = model.specifications
    duration = float(spec.duration)
    sample_rate = int(model.audio.sample_rate)
    window = model.audio.get_num_samples(duration)
    warm_up = tuple(float(x) for x in spec.warm_up)
    # Шаг окна — как у pyannote Inference по умолчанию: 10% окна, если нет warm-up.
    step = 0.1 * duration if warm_up[0] == 0.0 else warm_up[0]
    frames = model.receptive_field

    target = out_dir / VAD_ONNX_NAME
    dummy = torch.zeros(1, 1, window)
    with torch.no_grad(), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        torch.onnx.export(
            model,
            (dummy,),
            str(target),
            input_names=["waveform"],
            output_names=["logits"],
            dynamic_axes={"waveform": {0: "batch"}, "logits": {0: "batch"}},
            opset_version=17,
            dynamo=False,
        )
        num_frames = int(model(dummy).shape[1])

    mapping = Powerset(len(spec.classes), spec.powerset_max_classes).mapping
    meta = {
        "format": 1,
        "kind": "pyannote_segmentation",
        "sample_rate": sample_rate,
        "duration": duration,
        "step": step,
        "window_samples": int(window),
        "warm_up": list(warm_up),
        "frame_start": float(frames.start),
        "frame_duration": float(frames.duration),
        "frame_step": float(frames.step),
        "num_frames": num_frames,
        "powerset_mapping": mapping.int().tolist(),
        # VoiceActivityDetection для powerset-модели: пороги жёстко 0.5.
        "onset": 0.5,
        "offset": 0.5,
    }
    _attach_meta(target, meta)
    return target


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=MODELS_DIR / "onnx")
    parser.add_argument("--skip-gigaam", action="store_true")
    parser.add_argument("--skip-vad", action="store_true")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    produced: list[Path] = []
    if not args.skip_vad:
        produced.append(export_vad(args.out))
    if not args.skip_gigaam:
        produced.append(export_gigaam(args.out))
    for path in produced:
        size_mb = path.stat().st_size / (1024 * 1024)
        print(f"{path.name}: {size_mb:.1f} МБ, sha256 {_sha256(path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
