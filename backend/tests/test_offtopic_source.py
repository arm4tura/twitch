"""Фаза 2: транскрипция исходника (без ffmpeg и без GigaAM — оба подменены).

Проверяем контракт вокруг тяжёлой стадии: ссылка отвергается сразу, кэш не
перераспознаёт, пустой транскрипт — это ошибка, а не «стример говорил всё время
сам» (иначе оффтопом стал бы весь VOD).
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from twitch_cut.llm_offtopic import source as src_mod
from twitch_cut.llm_offtopic.source import (
    SourceError,
    extract_source_audio,
    resolve_source_path,
    transcribe_source,
)


@pytest.fixture
def media(tmp_path: Path) -> Path:
    f = tmp_path / "source.mp3"
    f.write_bytes(b"not really audio")
    return f


# --- проверка входа ---------------------------------------------------------


@pytest.mark.parametrize(
    "url",
    [
        "https://www.youtube.com/watch?v=abc",
        "http://example.com/video.mp4",
        "  HTTPS://EXAMPLE.COM/x.mp4  ",
    ],
)
def test_url_is_rejected_with_a_hint(url):
    with pytest.raises(SourceError, match="скачивание не поддерживается"):
        resolve_source_path(url)


def test_missing_file_is_rejected(tmp_path: Path):
    with pytest.raises(SourceError, match="не найден"):
        resolve_source_path(tmp_path / "нет.mp4")


def test_directory_is_rejected(tmp_path: Path):
    with pytest.raises(SourceError, match="не каталогом"):
        resolve_source_path(tmp_path)


def test_empty_path_is_rejected():
    with pytest.raises(SourceError, match="пуст"):
        resolve_source_path("   ")


# --- извлечение аудио -------------------------------------------------------


def _fake_ffmpeg(monkeypatch, *, fail: bool = False) -> list[list[str]]:
    """Подменить ffmpeg: записываем команды, создаём выходной wav."""
    calls: list[list[str]] = []
    monkeypatch.setattr(src_mod, "get_ffmpeg_path", lambda: "/fake/ffmpeg")

    def _run(cmd, **kw):
        calls.append(list(cmd))
        if fail:
            raise subprocess.CalledProcessError(
                1, cmd, stderr="ffmpeg version …\nInvalid data found when processing input"
            )
        Path(cmd[-1]).write_bytes(b"RIFF")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(src_mod.subprocess, "run", _run)
    return calls


def test_extract_audio_is_16k_mono_and_cached(tmp_path: Path, media: Path, monkeypatch):
    calls = _fake_ffmpeg(monkeypatch)

    wav, key = extract_source_audio(media, tmp_path)
    assert wav.exists()
    cmd = calls[0]
    # Тот же формат, что у стрима: 16 кГц моно PCM, без видео.
    assert cmd[:3] == ["/fake/ffmpeg", "-y", "-i"]
    for flag, value in (("-ar", "16000"), ("-ac", "1"), ("-acodec", "pcm_s16le")):
        assert cmd[cmd.index(flag) + 1] == value
    assert "-vn" in cmd

    again, key2 = extract_source_audio(media, tmp_path)
    assert (again, key2) == (wav, key)
    assert len(calls) == 1  # второй раз ffmpeg не звали


def test_extract_audio_failure_becomes_source_error(tmp_path: Path, media: Path, monkeypatch):
    _fake_ffmpeg(monkeypatch, fail=True)
    with pytest.raises(SourceError, match="Invalid data found"):
        extract_source_audio(media, tmp_path)


# --- транскрипция -----------------------------------------------------------


def _fake_asr(monkeypatch, transcript: dict) -> list[dict]:
    """Подменить GigaAM (он импортируется внутри transcribe_source лениво)."""
    calls: list[dict] = []
    import twitch_cut.gigaam_asr as gigaam_asr

    def _transcribe(*, audio_path, workdir, model_name, device, force, on_progress):
        calls.append(
            {
                "audio_path": audio_path,
                "model_name": model_name,
                "device": device,
                "force": force,
            }
        )
        on_progress(0.5, "половина")
        path = Path(workdir) / "cache" / "transcript_fake.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}", encoding="utf-8")
        return transcript, "fake-key", path

    monkeypatch.setattr(gigaam_asr, "transcribe_with_gigaam", _transcribe)
    return calls


_TRANSCRIPT = {
    "segments": [
        {
            "start": 0.0,
            "end": 3.0,
            "words": [
                {"word": "кот", "start": 0.0, "end": 0.5},
                {"word": "сидел", "start": 0.6, "end": 1.0},
            ],
        },
        {
            "start": 3.0,
            "end": 120.5,
            "words": [{"word": "на", "start": 3.0, "end": 3.2}],
        },
    ]
}


def test_transcribe_source_uses_same_engine_params(tmp_path: Path, media: Path, monkeypatch):
    _fake_ffmpeg(monkeypatch)
    asr_calls = _fake_asr(monkeypatch, _TRANSCRIPT)
    ticks: list[tuple[float, str]] = []

    result = transcribe_source(
        media,
        tmp_path,
        model_name="v3_ctc",
        device="mps",
        on_progress=lambda f, m: ticks.append((f, m)),
    )

    # Модель и устройство идут в ASR как передали: иначе тексты несравнимы.
    assert asr_calls[0]["model_name"] == "v3_ctc"
    assert asr_calls[0]["device"] == "mps"
    # На вход ASR ушёл извлечённый wav, а не исходный mp3.
    assert asr_calls[0]["audio_path"].suffix == ".wav"

    assert result.source_path == media.resolve()
    assert result.word_count == 3
    assert result.duration_s == pytest.approx(120.5)
    assert result.key == "fake-key"

    fractions = [f for f, _ in ticks]
    assert fractions == sorted(fractions)  # прогресс монотонный
    assert fractions[0] == 0.0 and fractions[-1] == 1.0


def test_transcribe_source_rejects_wordless_transcript(tmp_path: Path, media: Path, monkeypatch):
    _fake_ffmpeg(monkeypatch)
    _fake_asr(monkeypatch, {"segments": []})
    with pytest.raises(SourceError, match="нет слов с таймингами"):
        transcribe_source(media, tmp_path)


def test_transcribe_source_passes_url_check_first(tmp_path: Path):
    # Ссылку отвергаем ДО ffmpeg и ДО загрузки модели.
    with pytest.raises(SourceError, match="скачивание не поддерживается"):
        transcribe_source("https://youtu.be/xyz", tmp_path)
