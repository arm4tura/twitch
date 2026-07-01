"""Тесты для новых Timeline-эндпоинтов: /waveform, /media (Range), /waveform/allow.

Стратегия — не гоняем реальный ffmpeg (медленно, требует бинарь), а мокаем
`_run_ffmpeg_pcm`, возвращая контролируемый PCM. Для теста Range достаточно
plain-файла без аудио-контента.
"""

from __future__ import annotations

import array
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from twitch_cut.server import waveform as waveform_mod
from twitch_cut.server.app import create_app


# --- helpers ----------------------------------------------------------------


def _make_pcm_sine(duration_s: float, sample_rate: int = 8000, amp: int = 20000) -> bytes:
    """Собрать PCM s16le mono с постоянной амплитудой (для проверки нормализации).

    Синус реальной формы для теста избыточен — достаточно, чтобы max(abs) был
    предсказуем: используем «пилу» +amp/-amp, проверим что нормализованный
    пик близок к amp/32768.
    """
    total = int(duration_s * sample_rate)
    arr = array.array("h", [amp if (i % 2 == 0) else -amp for i in range(total)])
    return arr.tobytes()


# --- /waveform --------------------------------------------------------------


def test_waveform_returns_normalized_peaks(tmp_path: Path):
    """Пилообразный PCM 1 с при peaks=128 → 128 значений, max ≈ amp/32768."""
    src = tmp_path / "sine.wav"
    src.write_bytes(b"placeholder")  # реальный контент не нужен — ffmpeg замокан
    cache = tmp_path / "cache"

    fake_pcm = _make_pcm_sine(duration_s=1.0)
    client = TestClient(create_app())
    client.app.state.store.allow_media(str(src))

    with patch.object(waveform_mod, "_run_ffmpeg_pcm", return_value=fake_pcm), \
         patch.object(waveform_mod, "_default_cache_root", return_value=cache):
        r = client.get(f"/waveform?path={src}&peaks=128")

    assert r.status_code == 200, r.text
    body = r.json()
    assert isinstance(body["peaks"], list)
    assert len(body["peaks"]) == 128
    assert body["sample_rate"] == 8000
    # 1 сек PCM 8000 Hz s16le → duration_s = 1.0
    assert body["duration_s"] == pytest.approx(1.0, rel=0.01)
    peak = max(abs(x) for x in body["peaks"])
    assert 0.55 < peak < 0.70  # amp=20000 → 20000/32768 ≈ 0.61


def test_waveform_caches_second_call(tmp_path: Path):
    """Второй GET /waveform с теми же параметрами не должен звать ffmpeg."""
    src = tmp_path / "clip.wav"
    src.write_bytes(b"placeholder")
    cache = tmp_path / "cache"

    fake_pcm = _make_pcm_sine(0.5)
    client = TestClient(create_app())
    client.app.state.store.allow_media(str(src))

    with patch.object(
        waveform_mod, "_run_ffmpeg_pcm", return_value=fake_pcm
    ) as m_ffmpeg, patch.object(waveform_mod, "_default_cache_root", return_value=cache):
        r1 = client.get(f"/waveform?path={src}&peaks=64")
        r2 = client.get(f"/waveform?path={src}&peaks=64")

    assert r1.status_code == 200
    assert r2.status_code == 200
    assert r1.json()["peaks"] == r2.json()["peaks"]
    # ffmpeg вызван только один раз — второй ответ из кэша.
    assert m_ffmpeg.call_count == 1


def test_waveform_requires_whitelist(tmp_path: Path):
    """Путь не в whitelist → 403, даже если файл существует."""
    src = tmp_path / "leaked.wav"
    src.write_bytes(b"x")
    client = TestClient(create_app())  # allow_media НЕ вызван
    r = client.get(f"/waveform?path={src}&peaks=128")
    assert r.status_code == 403


def test_waveform_404_when_missing(tmp_path: Path):
    """Whitelist разрешает несуществующий путь → 404 (не 500)."""
    src = tmp_path / "nope.wav"
    client = TestClient(create_app())
    client.app.state.store.allow_media(str(src))  # в whitelist, но файла нет
    r = client.get(f"/waveform?path={src}&peaks=128")
    assert r.status_code == 404


# --- /media -----------------------------------------------------------------


def test_media_full_returns_200(tmp_path: Path):
    src = tmp_path / "a.mp3"
    src.write_bytes(b"A" * 5000)
    client = TestClient(create_app())
    client.app.state.store.allow_media(str(src))
    r = client.get(f"/media?path={src}")
    assert r.status_code == 200
    assert r.content == b"A" * 5000
    assert r.headers.get("accept-ranges") == "bytes"


def test_media_range_returns_206(tmp_path: Path):
    """Range: bytes=0-1023 → 206 + Content-Range + первые 1024 байта."""
    src = tmp_path / "a.mp3"
    body = bytes(range(256)) * 20  # 5120 B
    src.write_bytes(body)
    client = TestClient(create_app())
    client.app.state.store.allow_media(str(src))
    r = client.get(f"/media?path={src}", headers={"Range": "bytes=0-1023"})
    assert r.status_code == 206
    assert r.headers["content-range"] == f"bytes 0-1023/{len(body)}"
    assert r.headers["content-length"] == "1024"
    assert r.content == body[:1024]


def test_media_range_suffix(tmp_path: Path):
    """`Range: bytes=-500` → последние 500 байт."""
    src = tmp_path / "a.mp3"
    body = b"Z" * 2000
    src.write_bytes(body)
    client = TestClient(create_app())
    client.app.state.store.allow_media(str(src))
    r = client.get(f"/media?path={src}", headers={"Range": "bytes=-500"})
    assert r.status_code == 206
    assert r.content == body[-500:]


def test_media_range_out_of_bounds_416(tmp_path: Path):
    src = tmp_path / "a.mp3"
    src.write_bytes(b"x" * 100)
    client = TestClient(create_app())
    client.app.state.store.allow_media(str(src))
    r = client.get(f"/media?path={src}", headers={"Range": "bytes=999-1999"})
    assert r.status_code == 416
    assert "content-range" in r.headers


def test_media_rejects_unlisted_path(tmp_path: Path):
    src = tmp_path / "secret.mp4"
    src.write_bytes(b"nope")
    client = TestClient(create_app())  # без allow_media
    r = client.get(f"/media?path={src}")
    assert r.status_code == 403


def test_media_404_when_file_missing(tmp_path: Path):
    """Whitelist разрешил путь, но файла нет → 404."""
    src = tmp_path / "phantom.mp4"
    client = TestClient(create_app())
    client.app.state.store.allow_media(str(src))
    r = client.get(f"/media?path={src}")
    # resolve_and_authorize кидает 404 до Range-логики.
    assert r.status_code == 404


# --- /waveform/allow --------------------------------------------------------


def test_allow_media_extends_whitelist(tmp_path: Path):
    """POST /waveform/allow добавляет пути и после этого /media работает."""
    src = tmp_path / "new.mp3"
    src.write_bytes(b"data")
    client = TestClient(create_app())

    r_forbidden = client.get(f"/media?path={src}")
    assert r_forbidden.status_code == 403

    r_allow = client.post("/waveform/allow", json={"paths": [str(src)]})
    assert r_allow.status_code == 200
    assert str(src.resolve()) in r_allow.json()["allowed"]

    r_ok = client.get(f"/media?path={src}")
    assert r_ok.status_code == 200


def test_allow_media_ignores_missing_paths(tmp_path: Path):
    """Несуществующие пути молча выбрасываются из allowed-ответа."""
    client = TestClient(create_app())
    r = client.post(
        "/waveform/allow",
        json={"paths": [str(tmp_path / "ghost.mp3"), 42, None]},  # + мусор
    )
    assert r.status_code == 200
    assert r.json()["allowed"] == []


def test_allow_media_rejects_wrong_shape():
    client = TestClient(create_app())
    r = client.post("/waveform/allow", json={"paths": "not-a-list"})
    assert r.status_code == 422


# --- process job pre-fills whitelist ---------------------------------------


def test_process_job_prefills_media_whitelist(tmp_path: Path):
    """После POST /jobs/process оба медиа-пути должны быть доступны через /media."""
    stream = tmp_path / "s.mp4"
    original = tmp_path / "o.mp4"
    for p in (stream, original):
        p.write_bytes(b"x" * 32)
    banwords = tmp_path / "b.txt"
    banwords.write_text("")
    workdir = tmp_path / "wd"
    workdir.mkdir()
    decisions = workdir / "d.json"
    vegas = workdir / "v.cs"

    client = TestClient(create_app())
    r = client.post(
        "/jobs/process",
        json={
            "stream": str(stream),
            "original": str(original),
            "banwords": str(banwords),
            "workdir": str(workdir),
            "decisions": str(decisions),
            "vegas": str(vegas),
            "mock_transcript": str(tmp_path / "t.json"),
        },
    )
    assert r.status_code == 200
    # Оба медиа теперь в whitelist — /media их отдаёт.
    r_stream = client.get(f"/media?path={stream}")
    r_original = client.get(f"/media?path={original}")
    assert r_stream.status_code == 200
    assert r_original.status_code == 200
