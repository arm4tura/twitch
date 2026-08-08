"""Тесты рендера (`twitch_cut.render`).

subprocess мокается — ffmpeg не запускается. Проверяем инверсию сегментов,
выбор энкодера, построение фильтра и оркестрацию render_video (прогресс, ошибки).
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from twitch_cut import render


# --- invert_segments ---


def test_invert_basic():
    keeps = render.invert_segments([(10.0, 20.0), (30.0, 40.0)], total_duration=60.0)
    assert keeps == [(0.0, 10.0), (20.0, 30.0), (40.0, 60.0)]


def test_invert_empty_remove():
    assert render.invert_segments([], total_duration=60.0) == [(0.0, 60.0)]


def test_invert_remove_from_start():
    keeps = render.invert_segments([(0.0, 10.0)], total_duration=60.0)
    assert keeps == [(10.0, 60.0)]


def test_invert_remove_to_end():
    keeps = render.invert_segments([(50.0, 60.0)], total_duration=60.0)
    assert keeps == [(0.0, 50.0)]


def test_invert_overlapping_removes_merged():
    keeps = render.invert_segments([(10.0, 20.0), (15.0, 25.0)], total_duration=60.0)
    assert keeps == [(0.0, 10.0), (25.0, 60.0)]


def test_invert_unsorted():
    keeps = render.invert_segments([(30.0, 40.0), (10.0, 20.0)], total_duration=60.0)
    assert keeps == [(0.0, 10.0), (20.0, 30.0), (40.0, 60.0)]


def test_invert_remove_everything():
    keeps = render.invert_segments([(0.0, 60.0)], total_duration=60.0)
    assert keeps == []


def test_invert_clamps_beyond_duration():
    keeps = render.invert_segments([(50.0, 100.0)], total_duration=60.0)
    assert keeps == [(0.0, 50.0)]


def test_invert_drops_tiny_slivers():
    # keep между двумя вплотную идущими removes меньше 1мс — отбрасывается
    keeps = render.invert_segments([(10.0, 20.0), (20.0005, 30.0)], total_duration=60.0)
    assert keeps == [(0.0, 10.0), (30.0, 60.0)]


# --- get_best_encoder ---


def test_encoder_macos(monkeypatch):
    render.get_best_encoder.cache_clear()
    monkeypatch.setattr(render.platform, "system", lambda: "Darwin")
    assert render.get_best_encoder() == "h264_videotoolbox"
    render.get_best_encoder.cache_clear()


def test_encoder_nvenc(monkeypatch):
    render.get_best_encoder.cache_clear()
    monkeypatch.setattr(render.platform, "system", lambda: "Windows")
    monkeypatch.setattr(render, "get_ffmpeg_path", lambda: "/fake/ffmpeg")

    def _fake_run(cmd, **kwargs):
        return subprocess.CompletedProcess(
            cmd, 0, stdout="V..... h264_nvenc\nV..... h264_qsv\n", stderr=""
        )

    monkeypatch.setattr(subprocess, "run", _fake_run)
    assert render.get_best_encoder() == "h264_nvenc"
    render.get_best_encoder.cache_clear()


def test_encoder_cpu_fallback(monkeypatch):
    render.get_best_encoder.cache_clear()
    monkeypatch.setattr(render.platform, "system", lambda: "Linux")
    monkeypatch.setattr(render, "get_ffmpeg_path", lambda: "/fake/ffmpeg")

    def _fake_run(cmd, **kwargs):
        return subprocess.CompletedProcess(cmd, 0, stdout="V..... libx264\n", stderr="")

    monkeypatch.setattr(subprocess, "run", _fake_run)
    assert render.get_best_encoder() == "libx264"
    render.get_best_encoder.cache_clear()


# --- _build_concat_inputs ---


def test_build_concat_inputs():
    input_args, fc = render._build_concat_inputs(
        Path("/src.mp4"), [(0.0, 10.0), (20.0, 30.0)]
    )
    # Один seeked-инпут (-ss/-t/-i) на каждый keep-сегмент.
    assert input_args.count("-ss") == 2
    assert input_args.count("-i") == 2
    assert "0.000" in input_args and "20.000" in input_args  # старты сегментов
    # -t несёт ДЛИТЕЛЬНОСТЬ (e - s), а не абсолютный конец.
    t_values = [input_args[i + 1] for i, a in enumerate(input_args) if a == "-t"]
    assert t_values == ["10.000", "10.000"]
    assert fc == "[0:v][0:a][1:v][1:a]concat=n=2:v=1:a=1[v][a]"


# --- render_video (fake Popen) ---


class _FakePopen:
    """Фейковый Popen: отдаёт заранее заданные строки stderr и код возврата."""

    def __init__(self, lines, returncode=0):
        self.stdout = iter(lines)
        self._returncode = returncode

    def wait(self):
        return self._returncode


@pytest.fixture
def fake_video(tmp_path: Path) -> Path:
    v = tmp_path / "in.mp4"
    v.write_bytes(b"fake mp4")
    return v


def test_render_success_reports_progress(fake_video, tmp_path, monkeypatch):
    out = tmp_path / "out.mp4"
    lines = [
        "frame=1 time=00:00:05.00 bitrate=1000\n",
        "frame=2 time=00:00:10.00 bitrate=1000\n",
    ]
    monkeypatch.setattr(render, "get_ffmpeg_path", lambda: "/fake/ffmpeg")
    monkeypatch.setattr(
        render.subprocess, "Popen", lambda *a, **k: _FakePopen(lines, 0)
    )

    seen = []
    render.render_video(
        fake_video,
        out,
        segments_to_remove=[(20.0, 30.0)],  # keep 0-20, 30-60 => 50s
        total_duration=60.0,
        encoder="libx264",
        progress=lambda p, m="": seen.append(p),
    )

    assert seen[0] == 0.0
    assert seen[-1] == 100.0
    assert any(0 < p < 100 for p in seen)  # были промежуточные проценты


def test_render_reads_only_kept_ranges(fake_video, tmp_path, monkeypatch):
    """cmd подаёт один seeked-инпут (-ss) на keep-сегмент, а не весь исходник."""
    out = tmp_path / "out.mp4"
    captured: dict[str, list] = {}

    def _capture_popen(cmd, *a, **k):
        captured["cmd"] = cmd
        return _FakePopen(["frame=1 time=00:00:05.00 bitrate=1000\n"], 0)

    monkeypatch.setattr(render, "get_ffmpeg_path", lambda: "/fake/ffmpeg")
    monkeypatch.setattr(render.subprocess, "Popen", _capture_popen)

    render.render_video(
        fake_video,
        out,
        segments_to_remove=[(20.0, 30.0)],  # keep 0-20, 30-60 => 2 сегмента
        total_duration=60.0,
        encoder="libx264",
    )

    cmd = captured["cmd"]
    assert cmd.count("-ss") == 2  # по одному seeked-инпуту на keep
    # concat-фильтр лежит внутри значения -filter_complex, а не отдельным
    # элементом — ищем подстроку в этом аргументе.
    fc = cmd[cmd.index("-filter_complex") + 1]
    assert "concat=n=2:v=1:a=1[v][a]" in fc


def test_render_ffmpeg_failure_raises(fake_video, tmp_path, monkeypatch):
    out = tmp_path / "out.mp4"
    monkeypatch.setattr(render, "get_ffmpeg_path", lambda: "/fake/ffmpeg")
    monkeypatch.setattr(
        render.subprocess,
        "Popen",
        lambda *a, **k: _FakePopen(["Error: boom\n"], 1),
    )

    with pytest.raises(RuntimeError, match="ffmpeg render failed"):
        render.render_video(
            fake_video, out, [(10.0, 20.0)], total_duration=60.0, encoder="libx264"
        )


def test_render_remove_everything_raises(fake_video, tmp_path, monkeypatch):
    out = tmp_path / "out.mp4"
    monkeypatch.setattr(render, "get_ffmpeg_path", lambda: "/fake/ffmpeg")
    with pytest.raises(ValueError, match="Nothing to keep"):
        render.render_video(
            fake_video, out, [(0.0, 60.0)], total_duration=60.0, encoder="libx264"
        )


def test_render_missing_input_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(render, "get_ffmpeg_path", lambda: "/fake/ffmpeg")
    with pytest.raises(FileNotFoundError):
        render.render_video(
            tmp_path / "nope.mp4", tmp_path / "o.mp4", [(1.0, 2.0)], total_duration=60.0
        )


def test_render_existing_output_skipped(fake_video, tmp_path, monkeypatch):
    out = tmp_path / "out.mp4"
    out.write_bytes(b"already rendered")

    def _boom(*a, **k):
        raise AssertionError("Popen should not be called when output exists")

    monkeypatch.setattr(render, "get_ffmpeg_path", lambda: "/fake/ffmpeg")
    monkeypatch.setattr(render.subprocess, "Popen", _boom)

    result = render.render_video(
        fake_video, out, [(10.0, 20.0)], total_duration=60.0, force=False
    )
    assert result == out
