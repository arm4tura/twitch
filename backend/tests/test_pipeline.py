"""Тесты общего ядра пайплайна (`twitch_cut.pipeline`).

Через mock_transcript гоняем полный цикл БЕЗ whisperx/gigaam/torch — так же,
как это делает CLI и server-runner. Отдельно закрываем регрессию: раньше
CLI `process` отвергал движок 'gigaam' (дефолтный!), потому что его список
допустимых движков разошёлся с config.validate() и server-ом.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from twitch_cut.cli import app
from twitch_cut.config import PipelineConfig
from twitch_cut.pipeline import resolve_range, run_transcription_pipeline

runner = CliRunner()


def _write_mock_transcript(path: Path) -> None:
    """Мок с одним матерным словом ('сука') и пословными таймингами."""
    path.write_text(
        json.dumps(
            {
                "segments": [
                    {
                        "id": "seg_1",
                        "start": 4.0,
                        "end": 6.0,
                        "text": "ну сука вот это поворот",
                        "words": [
                            {"word": "ну", "start": 4.1, "end": 4.25, "score": 0.98},
                            {"word": "сука", "start": 4.3, "end": 4.7, "score": 0.96},
                            {"word": "вот", "start": 4.8, "end": 5.0, "score": 0.98},
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )


def _write_banwords(path: Path) -> None:
    path.write_text("сука\n", encoding="utf-8")


def _fake_media(path: Path) -> None:
    # build_decisions только пишет путь в метаданные — реальный контейнер не
    # нужен, но файл должен существовать (mock-режим не зовёт ffmpeg).
    path.write_bytes(b"\x00")


# --- resolve_range -----------------------------------------------------------


def test_resolve_range_rejects_inverted(tmp_path: Path):
    stream = tmp_path / "s.mp4"
    _fake_media(stream)
    with pytest.raises(ValueError, match="range_out must be greater"):
        resolve_range(stream, "00:00:10", "00:00:05")


# --- full pipeline via mock transcript --------------------------------------


def _run(tmp_path: Path, transcriber: str) -> dict:
    stream = tmp_path / "stream.mp4"
    _fake_media(stream)
    banwords = tmp_path / "banwords.txt"
    _write_banwords(banwords)
    mock = tmp_path / "mock.json"
    _write_mock_transcript(mock)

    workdir = tmp_path / "work"
    decisions_out = tmp_path / "decisions.json"
    vegas_out = tmp_path / "mutes.cs"

    config = PipelineConfig(transcriber=transcriber, device="cpu", asr_options={})
    config.validate()

    return run_transcription_pipeline(
        stream=stream,
        original=stream,
        banwords=banwords,
        workdir=workdir,
        decisions_out=decisions_out,
        vegas_out=vegas_out,
        config=config,
        range_out="00:00:30",
        mock_transcript=mock,
    )


@pytest.mark.parametrize("transcriber", ["gigaam", "whisperx", "whispercpp"])
def test_pipeline_mock_all_engines(tmp_path: Path, transcriber: str):
    """mock_transcript обходит выбор движка — все три должны отработать
    одинаково и найти один мат."""
    result = _run(tmp_path, transcriber)

    assert result["mutes_count"] == 1
    assert Path(result["decisions_path"]).exists()
    assert Path(result["vegas_path"]).exists()
    doc = json.loads(Path(result["decisions_path"]).read_text(encoding="utf-8"))
    assert doc["mutes"], "ожидали хотя бы один mute"


def test_pipeline_emits_progress(tmp_path: Path):
    events: list[tuple[str, float]] = []

    stream = tmp_path / "stream.mp4"
    _fake_media(stream)
    banwords = tmp_path / "banwords.txt"
    _write_banwords(banwords)
    mock = tmp_path / "mock.json"
    _write_mock_transcript(mock)

    config = PipelineConfig(transcriber="gigaam", device="cpu", asr_options={})
    config.validate()
    run_transcription_pipeline(
        stream=stream,
        original=stream,
        banwords=banwords,
        workdir=tmp_path / "work",
        decisions_out=tmp_path / "d.json",
        vegas_out=tmp_path / "m.cs",
        config=config,
        range_out="00:00:30",
        mock_transcript=mock,
        progress=lambda stage, pct, msg="": events.append((stage, pct)),
    )
    # финальная стадия обязана долить до 100%
    assert events, "прогресс не эмитился"
    assert events[-1][1] == 100.0


# --- CLI regression: 'gigaam' больше не отвергается --------------------------


def test_cli_process_accepts_gigaam_default(tmp_path: Path):
    """Регрессия: CLI `process` раньше падал BadParameter на движке 'gigaam'
    (дефолтном). Гоняем через mock_transcript, движок явно не указываем —
    берётся дефолт 'gigaam'."""
    stream = tmp_path / "stream.mp4"
    _fake_media(stream)
    banwords = tmp_path / "banwords.txt"
    _write_banwords(banwords)
    mock = tmp_path / "mock.json"
    _write_mock_transcript(mock)
    decisions_out = tmp_path / "decisions.json"
    vegas_out = tmp_path / "mutes.cs"

    result = runner.invoke(
        app,
        [
            "process",
            "--stream", str(stream),
            "--original", str(stream),
            "--banwords", str(banwords),
            "--workdir", str(tmp_path / "work"),
            "--decisions", str(decisions_out),
            "--vegas", str(vegas_out),
            "--mock-transcript", str(mock),
            "--range-out", "00:00:30",
            "--device", "cpu",
        ],
    )
    assert result.exit_code == 0, result.output
    assert decisions_out.exists()
    assert "Готово" in result.output


def test_cli_process_rejects_unknown_transcriber(tmp_path: Path):
    stream = tmp_path / "stream.mp4"
    _fake_media(stream)
    banwords = tmp_path / "banwords.txt"
    _write_banwords(banwords)
    mock = tmp_path / "mock.json"
    _write_mock_transcript(mock)

    result = runner.invoke(
        app,
        [
            "process",
            "--stream", str(stream),
            "--original", str(stream),
            "--banwords", str(banwords),
            "--workdir", str(tmp_path / "work"),
            "--mock-transcript", str(mock),
            "--transcriber", "bogus",
            "--device", "cpu",
        ],
    )
    assert result.exit_code != 0
