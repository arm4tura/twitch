"""Тесты Timeline-эндпоинтов: /waveform, /waveform/range, /media (Range), /waveform/allow.

Стратегия — не гоняем реальный ffmpeg (медленно, требует бинарь), а мокаем
`_run_ffmpeg_pcm` / `_run_ffmpeg_pcm_range`, возвращая контролируемый PCM. Для
теста Range достаточно plain-файла без аудио-контента.
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


def _make_pcm_asymmetric(
    duration_s: float, sample_rate: int = 8000, hi: int = 24000, lo: int = -6000
) -> bytes:
    """PCM с РАЗНЫМ верхом и низом — проверяем что контур несимметричный.

    Ради этого и появился format=3: настоящие min/max вместо зеркального
    ±max(|.|). На таком сигнале max≈0.73, а min≈-0.18 — зеркало бы их сравняло.
    """
    total = int(duration_s * sample_rate)
    arr = array.array("h", [hi if (i % 2 == 0) else lo for i in range(total)])
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


def test_waveform_includes_rms(tmp_path: Path):
    """Ответ несёт rms[] той же длины, каждый rms ≤ peak, энергия > 0.

    Пила ±amp: в каждом бакете max(abs)=amp и RMS=amp → rms[i]≈peaks[i]≈0.61.
    Ключевой инвариант формы Audacity — ядро (rms) не вылезает за контур (peak).
    """
    src = tmp_path / "sine.wav"
    src.write_bytes(b"placeholder")
    cache = tmp_path / "cache"

    fake_pcm = _make_pcm_sine(duration_s=1.0)
    client = TestClient(create_app())
    client.app.state.store.allow_media(str(src))

    with patch.object(waveform_mod, "_run_ffmpeg_pcm", return_value=fake_pcm), \
         patch.object(waveform_mod, "_default_cache_root", return_value=cache):
        r = client.get(f"/waveform?path={src}&peaks=128")

    assert r.status_code == 200, r.text
    body = r.json()
    assert isinstance(body["rms"], list)
    assert len(body["rms"]) == len(body["peaks"]) == 128
    # RMS ≤ пик в каждом бакете (иначе ядро вылезло бы за контур).
    assert all(rm <= pk + 1e-4 for rm, pk in zip(body["rms"], body["peaks"]))
    assert max(body["rms"]) > 0.5  # энергия присутствует (не нули)


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


# --- /waveform/range (слой детали для зума) ---------------------------------


def test_waveform_range_returns_window(tmp_path: Path):
    """Окно [10, 12] при res=256 → 256 бакетов, границы = фактически покрытые."""
    src = tmp_path / "clip.wav"
    src.write_bytes(b"placeholder")
    client = TestClient(create_app())
    client.app.state.store.allow_media(str(src))

    # span=2.0, res=256 → _range_sample_rate = clamp(256*8/2) = 8000.
    fake_pcm = _make_pcm_sine(duration_s=2.0, sample_rate=8000)
    with patch.object(waveform_mod, "_run_ffmpeg_pcm_range", return_value=fake_pcm):
        r = client.get(f"/waveform/range?path={src}&start=10&end=12&res=256")

    assert r.status_code == 200, r.text
    body = r.json()
    assert len(body["min"]) == len(body["max"]) == len(body["rms"]) == 256
    assert len(body["peaks"]) == 256
    assert body["start_s"] == pytest.approx(10.0)
    assert body["end_s"] == pytest.approx(12.0, rel=0.01)
    assert body["sample_rate"] == 8000
    # Контур: min ≤ 0 ≤ max в каждом бакете (иначе форма перевернётся).
    assert all(mn <= 0.0 for mn in body["min"])
    assert all(mx >= 0.0 for mx in body["max"])
    # Ядро не вылезает за контур.
    assert all(rm <= pk + 1e-4 for rm, pk in zip(body["rms"], body["peaks"]))


def test_waveform_range_contour_is_asymmetric(tmp_path: Path):
    """min и max независимы: сигнал +24000/-6000 не должен стать зеркальным."""
    src = tmp_path / "clip.wav"
    src.write_bytes(b"placeholder")
    client = TestClient(create_app())
    client.app.state.store.allow_media(str(src))

    fake_pcm = _make_pcm_asymmetric(duration_s=1.0, sample_rate=8000)
    with patch.object(waveform_mod, "_run_ffmpeg_pcm_range", return_value=fake_pcm):
        r = client.get(f"/waveform/range?path={src}&start=0&end=1&res=64")

    assert r.status_code == 200, r.text
    body = r.json()
    assert max(body["max"]) == pytest.approx(24000 / 32768, rel=0.01)
    assert min(body["min"]) == pytest.approx(-6000 / 32768, rel=0.01)
    # Именно это ломало вид: зеркало ±max(|.|) сделало бы обе границы 0.73.
    assert abs(body["max"][0]) - abs(body["min"][0]) > 0.4


def test_waveform_range_raises_sample_rate_on_deep_zoom(tmp_path: Path):
    """Узкое окно → ffmpeg зовётся с поднятым sample rate (≥8 семплов/бакет)."""
    src = tmp_path / "clip.wav"
    src.write_bytes(b"placeholder")
    client = TestClient(create_app())
    client.app.state.store.allow_media(str(src))

    fake_pcm = _make_pcm_sine(duration_s=0.5, sample_rate=32768)
    with patch.object(
        waveform_mod, "_run_ffmpeg_pcm_range", return_value=fake_pcm
    ) as m_ffmpeg:
        r = client.get(f"/waveform/range?path={src}&start=100&end=100.5&res=2048")

    assert r.status_code == 200, r.text
    # span=0.5, buckets=2048 → 2048*8/0.5 = 32768 (в коридоре [8000, 48000]).
    _path, start_arg, span_arg, sr_arg = m_ffmpeg.call_args.args
    assert start_arg == pytest.approx(100.0)
    assert span_arg == pytest.approx(0.5)
    assert sr_arg == 32768
    assert r.json()["sample_rate"] == 32768


def test_waveform_range_reports_actual_span(tmp_path: Path):
    """У конца файла ffmpeg отдаёт меньше — end_s должен сказать правду.

    Иначе фронт растянет 2 секунды данных на 10-секундное окно и волна съедет.
    """
    src = tmp_path / "clip.wav"
    src.write_bytes(b"placeholder")
    client = TestClient(create_app())
    client.app.state.store.allow_media(str(src))

    # Просим 10 с, ffmpeg «упёрся в конец» и вернул 2 с при 8000 Hz.
    short_pcm = _make_pcm_sine(duration_s=2.0, sample_rate=8000)
    with patch.object(waveform_mod, "_run_ffmpeg_pcm_range", return_value=short_pcm):
        r = client.get(f"/waveform/range?path={src}&start=0&end=10&res=128")

    assert r.status_code == 200, r.text
    body = r.json()
    assert body["end_s"] == pytest.approx(2.0, rel=0.01)
    assert len(body["max"]) == 128  # res соблюдается независимо от объёма PCM


def test_waveform_range_caps_buckets_to_sample_budget(tmp_path: Path):
    """На максимальном зуме бакетов отдаётся меньше запрошенного — но честных.

    span=0.54 с при потолке 48 kHz = 25920 семплов. На 8192 бакета это ~3 семпла
    на бакет вместо обещанных 8 — раньше потолок выигрывал молча. Теперь число
    бакетов урезается до обеспеченного (~0.54*48000/8 ≈ 3240; точное значение
    зависит от float-вычитания границ, поэтому проверяем инвариант, а не число),
    и фронт по разнице «просили/получили» понимает, что глубже детализации нет.
    """
    src = tmp_path / "clip.wav"
    src.write_bytes(b"placeholder")
    client = TestClient(create_app())
    client.app.state.store.allow_media(str(src))

    fake_pcm = _make_pcm_sine(duration_s=0.54, sample_rate=48000)
    with patch.object(
        waveform_mod, "_run_ffmpeg_pcm_range", return_value=fake_pcm
    ) as m_ffmpeg:
        r = client.get(f"/waveform/range?path={src}&start=10&end=10.54&res=8192")

    assert r.status_code == 200, r.text
    body = r.json()
    assert body["sample_rate"] == 48000  # упёрлись в потолок
    n = len(body["max"])
    assert n < 8192  # запрос урезан — фронт увидит это и не пойдёт по кругу
    assert n == pytest.approx(3240, abs=2)
    assert len(body["min"]) == len(body["rms"]) == len(body["peaks"]) == n
    # Инвариант держится: ≥8 семплов на бакет.
    _p, _s, span_arg, sr_arg = m_ffmpeg.call_args.args
    assert span_arg * sr_arg / n >= 8.0
    # Диапазон не пострадал — урезали детализацию, а не окно.
    assert body["start_s"] == pytest.approx(10.0)
    assert body["end_s"] == pytest.approx(10.54, rel=0.01)


def test_range_buckets_keeps_request_when_affordable():
    """Если семплов хватает — число бакетов не трогаем."""
    # 2 с при 8000 Hz = 16000 семплов; 256 бакетов по 62 семпла — с запасом.
    assert waveform_mod._range_buckets(2.0, 256, 8000) == 256
    # 0.5 с при 32768 Hz = 16384 семпла → ровно 2048 бакетов по 8.
    assert waveform_mod._range_buckets(0.5, 2048, 32768) == 2048
    # Просят больше, чем обеспечено → урезаем.
    assert waveform_mod._range_buckets(0.5, 8192, 48000) == 3000
    # Мусор на входе не должен ронять расчёт.
    assert waveform_mod._range_buckets(0.0, 512, 48000) == 512
    assert waveform_mod._range_buckets(1.0, 512, 0) == 512


def test_waveform_range_rejects_inverted_window(tmp_path: Path):
    """end ≤ start → 422, а не пустой массив или 500."""
    src = tmp_path / "clip.wav"
    src.write_bytes(b"placeholder")
    client = TestClient(create_app())
    client.app.state.store.allow_media(str(src))
    r = client.get(f"/waveform/range?path={src}&start=5&end=3&res=128")
    assert r.status_code == 422


def test_waveform_range_requires_whitelist(tmp_path: Path):
    """Диапазонные пики — тоже приватные данные: путь не в whitelist → 403."""
    src = tmp_path / "leaked.wav"
    src.write_bytes(b"x")
    client = TestClient(create_app())  # allow_media НЕ вызван
    r = client.get(f"/waveform/range?path={src}&start=0&end=1&res=128")
    assert r.status_code == 403


def test_range_sample_rate_scales_with_zoom():
    """Адаптивный sample rate: растёт на зуме, зажат в [8000, 48000]."""
    # Широкое окно — базовых 8 kHz с запасом хватает.
    assert waveform_mod._range_sample_rate(600.0, 2000) == 8000
    # Глубокий зум — поднимаем до нужного (2000*8/0.4 = 40000).
    assert waveform_mod._range_sample_rate(0.4, 2000) == 40000
    # Ещё глубже — упираемся в потолок, а не декодируем 96 kHz зря.
    assert waveform_mod._range_sample_rate(0.1, 2000) == 48000
    # Мусор на входе не должен делить на ноль.
    assert waveform_mod._range_sample_rate(0.0, 2000) == 8000


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
