"""Сверка ONNX-пути распознавания с исходным torch-путём gigaam + pyannote.

Зачем. После `tools/export_onnx.py` (или обновления onnxruntime) нужно
убедиться, что рантайм без torch выдаёт то же самое, что выдавал torch:
те же участки речи у VAD, те же log-probs, тот же текст и те же границы слов.
Скрипт гонит оба пути на одном аудио и печатает расхождения по каждой стадии.

Нужен dev-набор с torch (`pip install -e ".[export]"`), запуск из backend/:
    python tools/onnx_parity.py <аудио> [--device cpu|mps|cuda]

Что считается нормой (CPU, fp32): VAD — границы совпадают точно; log-probs —
отличия ~1e-4 (другой порядок сложений в матмулах); argmax по кадрам
совпадает почти везде, текст сегментов совпадает. На GPU-провайдерах
расхождения больше (fp16/TF32 внутри), но текст должен совпадать.
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twitch_cut.config import MODELS_DIR  # noqa: E402
from twitch_cut.ffmpeg_tools import load_audio_16k  # noqa: E402
from twitch_cut.gigaam_asr import ASR_BATCH_SIZE, _load_engine  # noqa: E402
from twitch_cut.vad import segment_audio  # noqa: E402


def _torch_reference(audio_path: Path, audio: np.ndarray):
    """VAD-куски и log-probs по каждому куску — исходным torch-путём."""
    import gigaam
    import torch
    from gigaam import vad_utils

    vad_ckpt = MODELS_DIR / "vad" / "segmentation-3.0.bin"
    vad_utils.resolve_local_segmentation_path = lambda model_id: str(vad_ckpt)
    model = gigaam.load_model(
        "v3_ctc", fp16_encoder=False, device="cpu", download_root=str(MODELS_DIR / "gigaam")
    )
    model.eval()

    from twitch_cut.ffmpeg_tools import get_ffmpeg_path
    import os

    os.environ["PATH"] = str(Path(get_ffmpeg_path()).parent) + os.pathsep + os.environ["PATH"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pipeline = vad_utils.get_pipeline(torch.device("cpu"))
        speech = pipeline(str(audio_path))
    regions = [(s.start, s.end) for s in speech.get_timeline().support()]
    pieces, boundaries = segment_audio(audio, regions, 16000)

    log_probs: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(pieces), ASR_BATCH_SIZE):
            batch = pieces[start : start + ASR_BATCH_SIZE]
            lengths = torch.tensor([len(p) for p in batch])
            pad = torch.zeros(len(batch), int(lengths.max()))
            for i, piece in enumerate(batch):
                pad[i, : len(piece)] = torch.from_numpy(piece)
            encoded, enc_len = model.forward(pad, lengths)
            probs = model.head(encoder_output=encoded).float().numpy()
            for i in range(len(batch)):
                log_probs.append(probs[i, : int(enc_len[i])])
    return regions, boundaries, log_probs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("audio", type=Path)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    audio = load_audio_16k(args.audio)
    engine = _load_engine("v3_ctc", args.device)
    print(f"аудио {len(audio) / 16000:.1f} с, ускоритель ONNX: {engine.accelerator}")

    regions = engine.vad.speech_regions(audio)
    pieces, boundaries = segment_audio(audio, regions, 16000)
    ref_regions, ref_boundaries, ref_probs = _torch_reference(args.audio, audio)

    same_regions = len(regions) == len(ref_regions) and np.allclose(
        regions, ref_regions, atol=1e-9
    ) if regions else not ref_regions
    print(f"VAD: участков {len(regions)} / torch {len(ref_regions)}, совпадают: {same_regions}")
    print(
        f"куски под ASR: {len(boundaries)} / torch {len(ref_boundaries)}, "
        f"совпадают: {boundaries == ref_boundaries}"
    )
    if boundaries != ref_boundaries:
        return 1

    max_diff = 0.0
    frames = mismatched_frames = 0
    texts_equal = 0
    asr = engine.asr
    from twitch_cut.gigaam_onnx import greedy_tokens

    decoded = []
    for start in range(0, len(pieces), ASR_BATCH_SIZE):
        decoded.extend(asr.decode_batch(pieces[start : start + ASR_BATCH_SIZE]))
    for mine, ref in zip(decoded, ref_probs):
        if mine.log_probs.shape != ref.shape:
            print(f"разная длина кадров: {mine.log_probs.shape} против {ref.shape}")
            return 1
        max_diff = max(max_diff, float(np.abs(mine.log_probs - ref).max()))
        frames += ref.shape[0]
        mismatched_frames += int((mine.log_probs.argmax(-1) != ref.argmax(-1)).sum())
        ref_ids, _ = greedy_tokens(ref, asr.blank_id)
        texts_equal += int(ref_ids == mine.token_ids)
    print(f"log-probs: max |Δ| = {max_diff:.2e}")
    print(f"argmax по кадрам: расходится {mismatched_frames} из {frames}")
    print(f"текст сегментов совпал: {texts_equal} из {len(decoded)}")
    for item in decoded[:3]:
        print("  >", item.text[:100])
    return 0 if texts_equal == len(decoded) else 2


if __name__ == "__main__":
    raise SystemExit(main())
