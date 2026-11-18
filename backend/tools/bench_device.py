"""Сравнить ускорители GigaAM на одном аудио: скорость и расхождения с CPU.

Зачем. Устройство по умолчанию выбирается по замеру, а не на глаз: на Windows
это вопрос «DirectML ('cuda') быстрее CPU и не двигает текст?». CPU-прогон —
эталон (он сверен с torch-путём, см. tools/onnx_parity.py), остальные
устройства меряются против него. (Так и выяснилось, что CoreML с GigaAM не
работает: сессия создаётся, а первый прогон падает — см. onnx_models.)

Запуск (из backend/):
    python tools/bench_device.py <аудио.wav>                    # cpu + cuda
    python tools/bench_device.py <аудио.wav> --devices cpu,cuda

Время загрузки модели и время прогона печатаются раздельно; каждое устройство
гоняется дважды и в зачёт идёт второй прогон (первый прогревает драйвер).
Печатается и провайдер, который реально считал: если ускоритель упал, там
будет CPU.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twitch_cut import gigaam_asr  # noqa: E402
from twitch_cut.gigaam_asr import transcribe_with_gigaam  # noqa: E402


def _words(transcript: dict) -> list[dict]:
    return [w for s in transcript["segments"] for w in s["words"]]


def _run(audio: Path, device: str) -> tuple[dict, float, str]:
    with tempfile.TemporaryDirectory() as workdir:
        started = time.perf_counter()
        transcript, _, _ = transcribe_with_gigaam(audio, Path(workdir), device=device, force=True)
        elapsed = time.perf_counter() - started
    engine = gigaam_asr._MODEL_CACHE[("v3_ctc", device)]
    return transcript, elapsed, engine.accelerator


def _compare(reference: dict, other: dict) -> str:
    ref_words, other_words = _words(reference), _words(other)
    if len(ref_words) != len(other_words):
        return f"слов {len(other_words)} против {len(ref_words)} — текст разошёлся"
    text = sum(1 for a, b in zip(ref_words, other_words) if a["word"] != b["word"])
    timing = max(
        (max(abs(a["start"] - b["start"]), abs(a["end"] - b["end"])) for a, b in zip(ref_words, other_words)),
        default=0.0,
    )
    return f"другой текст у {text} из {len(ref_words)} слов, макс. сдвиг границы {timing * 1000:.0f} мс"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("audio", type=Path)
    parser.add_argument("--devices", default=None, help="через запятую; по умолчанию cpu + ускоритель платформы")
    args = parser.parse_args()

    devices = args.devices.split(",") if args.devices else ["cpu", "cuda"]
    results: dict[str, tuple[dict, float, str]] = {}
    for device in devices:
        load_started = time.perf_counter()
        gigaam_asr._load_engine("v3_ctc", device)
        load = time.perf_counter() - load_started
        _run(args.audio, device)  # прогрев
        transcript, elapsed, accel = _run(args.audio, device)
        results[device] = (transcript, elapsed, accel)
        print(f"{device}: провайдер {accel}, загрузка {load:.1f} с, прогон {elapsed:.1f} с", flush=True)

    reference = results.get("cpu")
    if reference is not None:
        for device, (transcript, elapsed, accel) in results.items():
            if device == "cpu":
                continue
            speedup = reference[1] / elapsed if elapsed else 0.0
            print(f"{device} против cpu: в {speedup:.2f} раза быстрее; {_compare(reference[0], transcript)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
