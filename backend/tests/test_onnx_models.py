"""Тесты `onnx_models` — скачивание моделей и выбор ускорителя.

Сеть не нужна: urllib умеет file://, поэтому «релиз» — это локальный каталог.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

import twitch_cut.onnx_models as onnx_models
from twitch_cut.onnx_models import ModelFile, _providers_for, ensure_model


@pytest.fixture
def release(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Каталог-«релиз» с одним файлом и пустой каталог моделей."""
    src = tmp_path / "release"
    src.mkdir()
    (src / "model.onnx").write_bytes(b"onnx-bytes" * 1000)
    monkeypatch.setattr(onnx_models, "MODELS_RELEASE_URL", src.as_uri())
    monkeypatch.setattr(onnx_models, "onnx_dir", lambda: tmp_path / "models")
    return src


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def test_downloads_and_verifies_checksum(release: Path):
    data = (release / "model.onnx").read_bytes()
    ticks: list[tuple[str, int, int]] = []
    path = ensure_model(
        ModelFile("model.onnx", _sha(data), 1), on_progress=lambda *a: ticks.append(a)
    )

    assert path.read_bytes() == data
    assert ticks and ticks[-1][1] == len(data)
    assert not path.with_name("model.onnx.part").exists()


def test_bad_checksum_leaves_nothing_behind(release: Path):
    model = ModelFile("model.onnx", "0" * 64, 1)
    with pytest.raises(RuntimeError, match="model.onnx"):
        ensure_model(model, retries=1)

    assert not model.path.exists(), "битый файл не должен выглядеть как готовый кэш"
    assert not model.path.with_name("model.onnx.part").exists()


def test_present_model_is_not_downloaded_again(release: Path):
    model = ModelFile("model.onnx", "0" * 64, 1)
    model.path.parent.mkdir(parents=True)
    model.path.write_bytes(b"already here")

    assert ensure_model(model, retries=1).read_bytes() == b"already here"


@pytest.mark.parametrize(
    ("device", "available", "first"),
    [
        ("cpu", ["DmlExecutionProvider", "CPUExecutionProvider"], "CPUExecutionProvider"),
        ("cuda", ["DmlExecutionProvider", "CPUExecutionProvider"], "DmlExecutionProvider"),
        ("cuda", ["CUDAExecutionProvider", "DmlExecutionProvider", "CPUExecutionProvider"], "CUDAExecutionProvider"),
        ("cuda", ["CPUExecutionProvider"], "CPUExecutionProvider"),
        ("mps", ["CoreMLExecutionProvider", "CPUExecutionProvider"], "CoreMLExecutionProvider"),
        ("mps", ["CPUExecutionProvider"], "CPUExecutionProvider"),
    ],
)
def test_device_maps_to_provider_with_cpu_fallback(device, available, first, tmp_path, monkeypatch):
    monkeypatch.setattr(onnx_models, "MODELS_DIR", tmp_path)
    providers = _providers_for(device, available)
    names = [p[0] if isinstance(p, tuple) else p for p in providers]

    assert names[0] == first
    assert names[-1] == "CPUExecutionProvider", "CPU — всегда последний запасной"
