"""Тесты детекта тишины (`twitch_cut.silence`).

subprocess.run мокается — реальный ffmpeg не запускается. Проверяем парсинг
stderr silencedetect, кэш, склейку и padding.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from twitch_cut import silence


# Типовой вывод ffmpeg silencedetect (пишется в stderr).
_FFMPEG_OUTPUT = """\
[silencedetect @ 0x7f] silence_start: 12.345
[silencedetect @ 0x7f] silence_end: 15.678 | silence_duration: 3.333
[silencedetect @ 0x7f] silence_start: 42.1
[silencedetect @ 0x7f] silence_end: 48.9 | silence_duration: 6.8
size=N/A time=00:01:00.00 bitrate=N/A
"""


def _fake_run_factory(stderr: str):
    """Возвращает функцию-замену subprocess.run с заданным stderr."""

    def _fake_run(cmd, **kwargs):
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr=stderr)

    return _fake_run


@pytest.fixture
def fake_audio(tmp_path: Path) -> Path:
    audio = tmp_path / "clip.wav"
    audio.write_bytes(b"RIFF....fake wav")
    return audio


def test_detect_parses_pairs(fake_audio, tmp_path, monkeypatch):
    monkeypatch.setattr(subprocess, "run", _fake_run_factory(_FFMPEG_OUTPUT))
    monkeypatch.setattr(silence, "get_ffmpeg_path", lambda: "/fake/ffmpeg")

    segments = silence.detect_silence_ffmpeg(
        fake_audio, threshold_db=-30.0, min_duration=0.5, workdir=tmp_path
    )

    assert segments == [(12.345, 15.678), (42.1, 48.9)]


def test_detect_dangling_start_dropped(fake_audio, tmp_path, monkeypatch):
    """Файл кончился на тишине — silence_start без end игнорируется."""
    output = (
        "[silencedetect @ 0x] silence_start: 10.0\n"
        "[silencedetect @ 0x] silence_end: 12.0 | silence_duration: 2.0\n"
        "[silencedetect @ 0x] silence_start: 55.0\n"  # без end
    )
    monkeypatch.setattr(subprocess, "run", _fake_run_factory(output))
    monkeypatch.setattr(silence, "get_ffmpeg_path", lambda: "/fake/ffmpeg")

    segments = silence.detect_silence_ffmpeg(fake_audio, workdir=tmp_path)
    assert segments == [(10.0, 12.0)]


def test_detect_no_silence(fake_audio, tmp_path, monkeypatch):
    monkeypatch.setattr(subprocess, "run", _fake_run_factory("size=N/A time=00:01:00\n"))
    monkeypatch.setattr(silence, "get_ffmpeg_path", lambda: "/fake/ffmpeg")

    segments = silence.detect_silence_ffmpeg(fake_audio, workdir=tmp_path)
    assert segments == []


def test_detect_uses_cache(fake_audio, tmp_path, monkeypatch):
    """Второй вызов с теми же параметрами читает кэш, не зовёт ffmpeg."""
    calls = {"n": 0}

    def _counting_run(cmd, **kwargs):
        calls["n"] += 1
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr=_FFMPEG_OUTPUT)

    monkeypatch.setattr(subprocess, "run", _counting_run)
    monkeypatch.setattr(silence, "get_ffmpeg_path", lambda: "/fake/ffmpeg")

    first = silence.detect_silence_ffmpeg(fake_audio, workdir=tmp_path)
    second = silence.detect_silence_ffmpeg(fake_audio, workdir=tmp_path)

    assert first == second
    assert calls["n"] == 1  # ffmpeg вызван только один раз


def test_detect_force_bypasses_cache(fake_audio, tmp_path, monkeypatch):
    calls = {"n": 0}

    def _counting_run(cmd, **kwargs):
        calls["n"] += 1
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr=_FFMPEG_OUTPUT)

    monkeypatch.setattr(subprocess, "run", _counting_run)
    monkeypatch.setattr(silence, "get_ffmpeg_path", lambda: "/fake/ffmpeg")

    silence.detect_silence_ffmpeg(fake_audio, workdir=tmp_path)
    silence.detect_silence_ffmpeg(fake_audio, workdir=tmp_path, force=True)

    assert calls["n"] == 2


def test_detect_missing_file_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(silence, "get_ffmpeg_path", lambda: "/fake/ffmpeg")
    with pytest.raises(FileNotFoundError):
        silence.detect_silence_ffmpeg(tmp_path / "nope.wav", workdir=tmp_path)


# --- merge_close_segments ---


def test_merge_close_segments():
    segs = [(10.0, 12.0), (12.3, 15.0), (20.0, 22.0)]
    merged = silence.merge_close_segments(segs, gap_threshold=0.5)
    assert merged == [(10.0, 15.0), (20.0, 22.0)]


def test_merge_gap_too_big_no_merge():
    segs = [(10.0, 12.0), (13.0, 15.0)]
    merged = silence.merge_close_segments(segs, gap_threshold=0.5)
    assert merged == [(10.0, 12.0), (13.0, 15.0)]


def test_merge_empty():
    assert silence.merge_close_segments([]) == []


def test_merge_unsorted_input():
    segs = [(20.0, 22.0), (10.0, 12.0)]
    merged = silence.merge_close_segments(segs, gap_threshold=0.5)
    assert merged == [(10.0, 12.0), (20.0, 22.0)]


def test_merge_overlapping():
    """Перекрывающиеся сегменты тоже склеиваются (gap отрицательный ≤ threshold)."""
    segs = [(10.0, 15.0), (13.0, 18.0)]
    merged = silence.merge_close_segments(segs, gap_threshold=0.5)
    assert merged == [(10.0, 18.0)]


# --- add_padding ---


def test_add_padding():
    segs = [(10.0, 12.0), (20.0, 22.0)]
    padded = silence.add_padding(segs, padding_before=0.1, padding_after=0.2)
    assert padded == [(9.9, 12.2), (19.9, 22.2)]


def test_add_padding_clamps_at_zero():
    segs = [(0.05, 2.0)]
    padded = silence.add_padding(segs, padding_before=0.1, padding_after=0.1)
    assert padded == [(0.0, 2.1)]
