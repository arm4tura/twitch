"""Тесты подгонки границ мьюта по аудио (mute_refine + интеграция в decisions).

Что именно проверяем. Жалоба была: мьют — красная плита, тянущаяся далеко по
тишине после того, как звук затух (битый word-timing → потолок 1.5 с →
константные 1.70 с независимо от аудио). Поэтому главный тест — «плита
поджимается к звуку», а рядом — гарантии, что подгонка НИКОГДА не делает окно
шире и не падает на мусорном входе.
"""

from __future__ import annotations

from array import array
import math
from pathlib import Path
import wave

import pytest

from twitch_cut.config import PipelineConfig
from twitch_cut.decisions import build_decisions
from twitch_cut.mute_refine import (
    WavEnergy,
    global_threshold,
    open_energy,
    reanchor_window,
    silence_before_ms,
    trim_window,
)
from twitch_cut.profanity import ProfanityMatch
from twitch_cut.timecode import parse_timecode

SR = 16000


def _write_wav(
    path: Path,
    duration_ms: int,
    loud: list[tuple[int, int]] | list[tuple[int, int, int]],
    *,
    amp: int = 12000,
    noise: int = 40,
    channels: int = 1,
    width: int = 2,
) -> Path:
    """WAV s16 mono 16 kHz: тихий шум везде + «речь» (тон 200 Гц) в loud-окнах.

    Тон, а не константа: RMS константы равен её амплитуде и не зависит от
    длины кадра, на таком сигнале порог сработал бы даже при ошибке в расчёте
    кадров. Тон 200 Гц при кадре 10 мс даёт 2 периода — честный RMS ≈ amp/√2.

    Элемент loud может быть (start_ms, end_ms) или (start_ms, end_ms, amp) —
    третий элемент нужен, чтобы сделать одно слово ГРОМЧЕ другого.
    """
    total = int(duration_ms * SR / 1000)
    samples = array("h", [0]) * total
    for i in range(total):
        samples[i] = noise if (i % 2 == 0) else -noise
    for span in loud:
        start_ms, end_ms = span[0], span[1]
        span_amp = span[2] if len(span) > 2 else amp
        s0 = max(0, int(start_ms * SR / 1000))
        s1 = min(total, int(end_ms * SR / 1000))
        for i in range(s0, s1):
            samples[i] = int(span_amp * math.sin(2 * math.pi * 200 * i / SR))
    with wave.open(str(path), "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(width)
        w.setframerate(SR)
        w.writeframes(samples.tobytes())
    return path


# --- trim_window -------------------------------------------------------------


def test_trim_window_cuts_silent_tail(tmp_path: Path):
    """Ровно жалоба со скриншота: окно 0…1700, звук только 200…600."""
    wav = _write_wav(tmp_path / "a.wav", 3000, [(200, 600)])
    with WavEnergy(wav) as energy:
        start, end = trim_window(energy, 0, 1700, guard_ms=40, min_ms=150)
    # Границы садятся на звук ± защитная кромка (кадр анализа 10 мс).
    assert start == pytest.approx(160, abs=15)
    assert end == pytest.approx(640, abs=15)
    # Плита 1.70 с превратилась в ~0.48 с — это и есть «как должно быть».
    assert (end - start) < 600


def test_trim_window_keeps_continuous_speech(tmp_path: Path):
    """Речь на всё окно — поджимать нечего, окно остаётся как решил ASR."""
    wav = _write_wav(tmp_path / "b.wav", 2000, [(0, 2000)])
    with WavEnergy(wav) as energy:
        start, end = trim_window(energy, 300, 1200)
    assert (start, end) == (300, 1200)


def test_trim_window_never_grows(tmp_path: Path):
    """Звук шире окна → окно НЕ расширяется.

    Это load-bearing: расширение по энергии обнулило бы защиту prev/next-word
    в _compute_mute_window — соседнее чистое слово это тоже энергия.
    """
    wav = _write_wav(tmp_path / "c.wav", 4000, [(0, 4000)])
    with WavEnergy(wav) as energy:
        start, end = trim_window(energy, 1000, 1500)
    assert start >= 1000
    assert end <= 1500


def test_trim_window_keeps_window_on_pure_silence(tmp_path: Path):
    """В окне вообще нет звука → подгонять не по чему, оставляем окно ASR."""
    wav = _write_wav(tmp_path / "d.wav", 2000, [], noise=0)
    with WavEnergy(wav) as energy:
        assert trim_window(energy, 200, 900) == (200, 900)


def test_trim_window_respects_min_duration(tmp_path: Path):
    """Короткий всплеск не превращается в 20-мс щелчок — тянем до min_ms."""
    wav = _write_wav(tmp_path / "e.wav", 2000, [(500, 530)])
    with WavEnergy(wav) as energy:
        start, end = trim_window(energy, 300, 1500, guard_ms=0, min_ms=300)
    assert end - start >= 300
    # …но всё ещё внутри исходного окна.
    assert start >= 300 and end <= 1500


def test_trim_window_handles_window_past_eof(tmp_path: Path):
    """Окно за концом файла не должно ронять расчёт."""
    wav = _write_wav(tmp_path / "f.wav", 500, [(0, 500)])
    with WavEnergy(wav) as energy:
        start, end = trim_window(energy, 2000, 2500)
    assert (start, end) == (2000, 2500)


def test_trim_window_ignores_inverted_window(tmp_path: Path):
    wav = _write_wav(tmp_path / "g.wav", 1000, [(0, 1000)])
    with WavEnergy(wav) as energy:
        assert trim_window(energy, 800, 800) == (800, 800)


def test_trim_window_does_not_drift_onto_louder_neighbour(tmp_path: Path):
    """Регрессия: мат тихий, соседнее слово в padding — ГРОМЧЕ.

    Порог считался как доля от пика ВСЕГО окна, поэтому пик громкого соседа
    задирал порог выше энергии самого мата: речь «находилась» только на соседе,
    и окно поджималось к нему — мат оставался слышен целиком. На реальном
    стриме так уехало 10 мьютов из 133 («блять» 6.866–7.105 → мьют
    7.075–7.225). Лечится привязкой поиска к core — таймингам самого мата.
    """
    wav = _write_wav(
        tmp_path / "neighbour.wav",
        4000,
        [(1000, 1240, 1200), (1300, 1700, 16000)],
    )
    word = (1000, 1240)
    window = (920, 1360)  # слово + padding 80/120

    with WavEnergy(wav) as energy:
        drifted = trim_window(energy, *window)
        anchored = trim_window(energy, *window, cores=[word])

    # Как было: окно уехало на соседа, от мата почти ничего не перекрыто.
    assert drifted[0] > word[0] + 100
    # Как надо: окно целиком накрывает слово (± защитная кромка).
    assert anchored[0] <= word[0]
    assert anchored[1] >= word[1]
    # И по-прежнему только сужение — за исходное окно не вылезли.
    assert anchored[0] >= window[0] and anchored[1] <= window[1]


def test_trim_window_keeps_sound_inside_padding(tmp_path: Path):
    """Главный инвариант правки: звук, попавший в padding, из мьюта НЕ срезается.

    Разметка слова врёт на ±160 мс (квартили против независимой линейки: onset
    после паузы ≥150 мс, 5309 слов), и padding существует ровно затем, чтобы эту
    ошибку впитывать. Старое правило искало речь ТОЛЬКО внутри разметки и
    добавляло guard, то есть padding выбрасывало целиком: здесь оно вернуло бы
    960 вместо 920 и хвост мата остался бы слышен. Замер на четырёх записях
    (55 минут, 328 окон): 17.4 → 14.7 с слышимого мата.
    """
    wav = _write_wav(tmp_path / "pad.wav", 4000, [(900, 1240)])
    word = (1000, 1240)          # разметка опоздала на 100 мс
    window = (920, 1360)         # слово + padding 80/120

    with WavEnergy(wav) as energy:
        start, end = trim_window(energy, *window, cores=[word], guard_ms=40)

    # Речь идёт от самого края окна — срезать нечего.
    assert start == window[0]
    # А тишина после слова срезана: поджатие свою работу делает.
    assert end < window[1]


def test_trim_window_gives_padding_sound_back_after_a_real_pause(tmp_path: Path):
    """Исключение из инварианта выше: если звук в padding отделён от слова
    ПАУЗОЙ, это соседнее слово, и оно из мьюта убирается.

    Порог паузы (`SILENCE_GAP_MS` = 60 мс) подобран замером: короче — правило
    принимает смычку согласного за паузу и отдаёт соседу начало самого мата
    (утечка 14.7 → 15.1 с при пороге 0), длиннее — лишнего замьюченного звука
    больше при той же утечке.
    """
    # Пауза 80 мс — сосед отдельный.
    far = _write_wav(tmp_path / "far.wav", 4000, [(1000, 1240), (1320, 1900)])
    # Пауза 40 мс — короче смычки, различить нельзя, звук остаётся в мьюте.
    near = _write_wav(tmp_path / "near.wav", 4000, [(1000, 1240), (1280, 1900)])
    word, window = (1000, 1240), (920, 1400)

    with WavEnergy(far) as energy:
        _, cut = trim_window(energy, *window, cores=[word], guard_ms=40)
    with WavEnergy(near) as energy:
        _, kept = trim_window(energy, *window, cores=[word], guard_ms=40)

    assert cut <= 1320                    # на соседа не заехали
    assert cut >= word[1]                 # но и мат перекрыт целиком
    assert kept == window[1]              # сомнительное остаётся замьюченным


def test_trim_window_core_outside_window_falls_back(tmp_path: Path):
    """cores за пределами окна (битые тайминги) → ищем по всему окну, не падаем."""
    wav = _write_wav(tmp_path / "core_out.wav", 3000, [(200, 600)])
    with WavEnergy(wav) as energy:
        with_core = trim_window(
            energy, 0, 1700, cores=[(2500, 2900)], guard_ms=40, min_ms=150
        )
        plain = trim_window(energy, 0, 1700, guard_ms=40, min_ms=150)
    assert with_core == plain


def test_trim_window_keeps_quiet_word_of_a_group(tmp_path: Path):
    """Регрессия: в группе два мата, первый ТИШЕ второго.

    Порог считался от пика всего разбега слов, поэтому тихое слово его не
    достигало — речь «находилась» только на громком, и мьют начинался уже после
    тихого мата: он оставался слышен целиком. На реальном стриме так потерялись
    «блять» 337.8 («блять блять» одной группой), «бля» 521.1 и «сука» 656.6
    («сука ебать» — «сука» тише). Лечится порогом на КАЖДОЕ слово отдельно.
    """
    wav = _write_wav(
        tmp_path / "group.wav",
        4000,
        [(1000, 1200, 900), (1320, 1560, 16000)],
    )
    quiet, loud = (1000, 1200), (1320, 1560)
    window = (920, 1680)  # оба слова + padding 80/120

    with WavEnergy(wav) as energy:
        hull = trim_window(energy, *window, cores=[quiet, loud])
        merged = trim_window(energy, *window, cores=[(quiet[0], loud[1])])

    # Как было (одно ядро на весь разбег): тихий мат за границей мьюта.
    assert merged[0] > quiet[1]
    # Как надо: оболочка накрывает оба слова, но всё ещё внутри окна.
    assert hull[0] <= quiet[0] and hull[1] >= loud[1]
    assert hull[0] >= window[0] and hull[1] <= window[1]


def test_trim_window_keeps_silent_word_of_a_group(tmp_path: Path):
    """Слово, на месте которого в аудио тишина, из мьюта не выпадает: поджимать
    по нему нечего, а выкинуть — значит оставить мат слышным, если врёт как раз
    аудио-часть (кэш обрезан, канал не тот), а не ASR."""
    wav = _write_wav(tmp_path / "silent_word.wav", 4000, [(1320, 1560)], noise=0)
    silent, loud = (1000, 1200), (1320, 1560)

    with WavEnergy(wav) as energy:
        hull = trim_window(energy, 920, 1680, cores=[silent, loud])

    assert hull[0] <= silent[0]
    assert hull[1] >= loud[1]


# --- reanchor_window ---------------------------------------------------------


def test_reanchor_only_flags_by_default(tmp_path: Path):
    """По умолчанию перенос НЕ двигает окно, а помечает его: вердикт 'silent',
    границы те же. Замер на 40 минутах правленого руками материала: из 25
    переносов 10 увели мьют С ТОГО звука, который замьютил человек, и только 2
    привели на него (см. docstring reanchor_window и config.mute_reanchor_move).
    Признак «окно на тишине» при этом рабочий — его и оставляем человеку."""
    wav = _write_wav(tmp_path / "silence.wav", 4000, [(1800, 2050)])
    with WavEnergy(wav) as energy:
        thr = global_threshold(energy)
        assert reanchor_window(energy, 1220, 1579, threshold=thr, guard_ms=80) == (
            1220,
            1579,
            "silent",
        )


def test_reanchor_moves_window_off_silence_onto_speech(tmp_path: Path):
    """Числа с реального прогона (vod_2847020280): окно «бля» 261.612–261.971 —
    0% озвученных кадров, звук на 262.192–262.442. Поджатие такое окно починить
    не может по определению (оно только сужает), человек руками сдвинул мьют на
    +543 мс. С `move=True` перенос делает это сам."""
    wav = _write_wav(tmp_path / "silence.wav", 4000, [(1800, 2050)])
    with WavEnergy(wav) as energy:
        thr = global_threshold(energy)
        start, end, verdict = reanchor_window(
            energy, 1220, 1579, threshold=thr, guard_ms=80, move=True
        )
    assert verdict == "reanchored"
    # Окно накрыло реальный звук…
    assert start <= 1800 and end >= 2050
    # …и не стало шире, чем решил ASR-путь (359 мс).
    assert (end - start) == 359


def test_reanchor_does_not_move_across_a_long_gap(tmp_path: Path):
    """Речь в радиусе поиска есть, но за широкой тишиной (700 мс): чей это звук —
    неизвестно, и мьют там с той же вероятностью накроет чужое слово. Именно такие
    переносы (зазор от 300 мс) уводили мьют с мата на замере, поэтому включённый
    перенос ограничен REANCHOR_MAX_GAP_MS — окно остаётся на месте с флагом."""
    wav = _write_wav(tmp_path / "far.wav", 4000, [(2200, 2600)])
    with WavEnergy(wav) as energy:
        thr = global_threshold(energy)
        assert reanchor_window(
            energy, 1100, 1500, threshold=thr, guard_ms=40, move=True
        ) == (1100, 1500, "no_speech")


def test_reanchor_keeps_window_that_already_has_sound(tmp_path: Path):
    wav = _write_wav(tmp_path / "onsound.wav", 4000, [(1000, 1500)])
    with WavEnergy(wav) as energy:
        thr = global_threshold(energy)
        assert reanchor_window(energy, 1050, 1450, threshold=thr) == (1050, 1450, "ok")


def test_reanchor_does_not_move_quiet_mat_onto_loud_neighbour(tmp_path: Path):
    """Тот же баг, от которого защищают `cores` в поджатии, но со стороны
    переноса: мат тише 15% от уровня громкой речи, поэтому для АБСОЛЮТНОГО порога
    он выглядит тишиной. Если бы перенос смотрел только на долю озвученных
    кадров, окно уехало бы с тихого мата на громкого соседа. Правило «речь
    задевает окно → не двигаем» это закрывает."""
    wav = _write_wav(
        tmp_path / "quiet_mat.wav", 4000, [(1000, 1240, 1200), (1300, 2000, 16000)]
    )
    with WavEnergy(wav) as energy:
        thr = global_threshold(energy)
        frames, _ = energy.rms_frames(1000, 1240)
        assert max(frames) < thr, "тихий мат должен быть НИЖЕ абсолютного порога"
        assert reanchor_window(energy, 920, 1360, threshold=thr) == (920, 1360, "ok")


def test_reanchor_reports_no_speech_when_there_is_none(tmp_path: Path):
    """Речи в радиусе поиска нет — окно остаётся как решил ASR (ослаблять мьют
    по подозрению нельзя), но вызывающий получает вердикт для флага «проверь»."""
    wav = _write_wav(tmp_path / "empty.wav", 4000, [(3500, 3800)])
    with WavEnergy(wav) as energy:
        thr = global_threshold(energy)
        assert reanchor_window(energy, 1000, 1400, threshold=thr) == (1000, 1400, "no_speech")


def test_reanchor_ignores_short_blips(tmp_path: Path):
    """Щелчок/стук короче REANCHOR_MIN_RUN_MS словом не считается."""
    wav = _write_wav(tmp_path / "blip.wav", 4000, [(1800, 1840), (3500, 3800)])
    with WavEnergy(wav) as energy:
        thr = global_threshold(energy)
        _, _, verdict = reanchor_window(energy, 1000, 1400, threshold=thr)
    assert verdict == "no_speech"


def test_reanchor_is_off_without_threshold_or_radius(tmp_path: Path):
    wav = _write_wav(tmp_path / "off.wav", 4000, [(1800, 2050)])
    with WavEnergy(wav) as energy:
        thr = global_threshold(energy)
        assert reanchor_window(energy, 1220, 1579, threshold=0.0) == (1220, 1579, "ok")
        assert reanchor_window(energy, 1220, 1579, threshold=thr, search_ms=0) == (
            1220,
            1579,
            "ok",
        )


def test_reanchor_takes_the_nearest_run(tmp_path: Path):
    """Ближайший участок, а не самый громкий: окно не должно перелетать через
    полсекунды тишины на более заметный звук."""
    wav = _write_wav(
        tmp_path / "two_runs.wav", 5000, [(1600, 1900, 6000), (2400, 2900, 16000)]
    )
    with WavEnergy(wav) as energy:
        thr = global_threshold(energy)
        start, _, verdict = reanchor_window(
            energy, 1200, 1500, threshold=thr, guard_ms=40, move=True
        )
    assert verdict == "reanchored"
    assert start == pytest.approx(1560, abs=20)


def test_reanchor_stays_inside_the_file(tmp_path: Path):
    """Речь у самого конца записи: окно не должно уехать за длину файла."""
    wav = _write_wav(tmp_path / "eof.wav", 2000, [(1700, 2000)])
    with WavEnergy(wav) as energy:
        thr = global_threshold(energy)
        start, end, verdict = reanchor_window(
            energy, 900, 1500, threshold=thr, guard_ms=40, move=True
        )
    assert verdict == "reanchored"
    assert end <= energy.duration_ms
    assert end - start == 600


def test_wav_energy_rejects_stereo(tmp_path: Path):
    """Кэш пайплайна всегда mono s16; иное — не наш файл, не угадываем."""
    wav = _write_wav(tmp_path / "h.wav", 200, [], channels=2)
    with pytest.raises(ValueError):
        WavEnergy(wav)


def test_open_energy_is_forgiving(tmp_path: Path):
    """Подгонка — улучшение точности, а не обязательный этап: любая проблема
    с файлом даёт None, а не исключение в пайплайне."""
    assert open_energy(None) is None
    assert open_energy(tmp_path / "missing.wav") is None
    broken = tmp_path / "broken.wav"
    broken.write_bytes(b"not a wav at all")
    assert open_energy(broken) is None


# --- silence_before_ms -------------------------------------------------------


def test_silence_before_ms_is_zero_on_a_voiced_neighbour(tmp_path: Path):
    """Звук идёт вплотную к `at_ms` → отодвигать голову некуда, возвращаем 0.

    Это ветка связной речи: ровно она гасит слепой допуск в чистого соседа.
    """
    wav = _write_wav(tmp_path / "voiced.wav", 4000, [(500, 1850)])
    with WavEnergy(wav) as energy:
        thr = global_threshold(energy)
        assert silence_before_ms(energy, 1800, 300, thr) == 0


def test_silence_before_ms_returns_the_silent_run_up_to_the_cap(tmp_path: Path):
    """Перед `at_ms` только тишина → отдаём её всю, но не дальше `max_ms`."""
    wav = _write_wav(tmp_path / "voiced.wav", 4000, [(500, 1850)])
    with WavEnergy(wav) as energy:
        thr = global_threshold(energy)
        # 2700–3000 — сплошная тишина (звук кончился в 1850): упираемся в потолок.
        assert silence_before_ms(energy, 3000, 300, thr) == 300


def test_silence_before_ms_stops_at_the_first_voiced_frame(tmp_path: Path):
    """Тишина 1850–2000, дальше назад звук: считаем ровно тишину, на звуке стоп."""
    wav = _write_wav(tmp_path / "voiced.wav", 4000, [(500, 1850)])
    with WavEnergy(wav) as energy:
        thr = global_threshold(energy)
        # Из 300 мс окна назад тишины только 150 (1850–2000), дальше хвост звука.
        assert silence_before_ms(energy, 2000, 300, thr) == pytest.approx(150, abs=15)


def test_silence_before_ms_is_inert_on_degenerate_input(tmp_path: Path):
    """Контракт «опция, но не фатальна»: без порога/окна/позиции — тихий ноль,
    а не исключение. Ровно так страж инертен без энергии или порога."""
    wav = _write_wav(tmp_path / "voiced.wav", 4000, [(500, 1850)])
    with WavEnergy(wav) as energy:
        thr = global_threshold(energy)
        assert silence_before_ms(energy, 2000, 0, thr) == 0  # окна нет
        assert silence_before_ms(energy, 0, 300, thr) == 0  # позиции нет
        assert silence_before_ms(energy, 2000, 300, 0.0) == 0  # порога нет


# --- интеграция в build_decisions -------------------------------------------


def _match(
    local_start: int,
    local_end: int,
    *,
    timing_source: str = "segment_fallback",
    prev_word_end: int | None = None,
) -> ProfanityMatch:
    """`prev_word_end` — конец разметки ПРЕДЫДУЩЕГО слова (мс, локальная шкала).
    Именно он включает клэмп головного padding в _compute_mute_window."""
    return ProfanityMatch(
        word="блять",
        normalized="блять",
        lemma="блять",
        banword="блять",
        banword_lemma="блять",
        local_start_ms=local_start,
        local_end_ms=local_end,
        stream_start_ms=local_start + 60_000,
        stream_end_ms=local_end + 60_000,
        confidence=0.9,
        segment_id="seg_1",
        match_type="surface",
        timing_source=timing_source,
        prev_word_local_end_ms=prev_word_end,
        prev_word_stream_end_ms=(
            None if prev_word_end is None else prev_word_end + 60_000
        ),
    )


def _decisions(
    tmp_path: Path,
    audio: Path | None,
    matches: list[ProfanityMatch] | None = None,
    **cfg,
) -> dict:
    return build_decisions(
        stream_path=Path("stream.mp4"),
        original_path=Path("original.mp4"),
        range_in=parse_timecode("00:01:00"),
        range_out=parse_timecode("00:02:00"),
        # Битый timing: слову отдали границы всего VAD-сегмента (2000…9480).
        matches=matches if matches is not None else [_match(2000, 9480)],
        config=PipelineConfig(**cfg),
        audio_cache=audio,
    )


def test_build_decisions_trims_mute_to_audio(tmp_path: Path):
    """Сквозной сценарий скриншота: 1.70 с плиты → окно по фактическому звуку."""
    # Звук есть только 2.0…2.45 с, дальше тишина — а ASR отдал окно до 9.48 с.
    wav = _write_wav(tmp_path / "stream.wav", 12_000, [(2000, 2450)])

    doc = _decisions(tmp_path, wav)
    mute = doc["mutes"][0]
    # Без подгонки было бы 1.92…3.62 (потолок 1.5 с + padding) = 1.70 с.
    assert mute["end"] - mute["start"] < 0.8
    assert mute["start"] == pytest.approx(1.95, abs=0.08)
    assert mute["end"] == pytest.approx(2.49, abs=0.08)
    # Stream-шкала сдвинута на ту же дельту (обе из одних и тех же слов).
    assert mute["stream_start"] - mute["start"] == pytest.approx(60.0, abs=0.01)
    assert mute["stream_end"] - mute["end"] == pytest.approx(60.0, abs=0.01)
    # Audit trail: видно, что подгонка сработала и на сколько.
    assert mute["audio_trim_end_ms"] > 1000
    assert doc["cuts"][0]["end"] == mute["end"]
    # raw_* остаются нетронутыми — это «что сказал ASR», их не подгоняем.
    assert doc["mutes"][0]["raw_local_end"] == 9.48


def test_build_decisions_without_audio_keeps_old_behaviour(tmp_path: Path):
    """Нет кэша аудио (mock_transcript-режим) → ровно прежние границы."""
    doc = _decisions(tmp_path, None)
    mute = doc["mutes"][0]
    assert mute["start"] == 1.92
    assert mute["end"] == pytest.approx(3.62, abs=0.001)
    assert mute["audio_trim_start_ms"] == 0
    assert mute["audio_trim_end_ms"] == 0


def test_build_decisions_trim_can_be_disabled(tmp_path: Path):
    """mute_trim_to_audio=False → аудио не читается, границы как раньше."""
    wav = _write_wav(tmp_path / "stream.wav", 12_000, [(2000, 2450)])
    doc = _decisions(tmp_path, wav, mute_trim_to_audio=False)
    assert doc["mutes"][0]["end"] == pytest.approx(3.62, abs=0.001)
    assert doc["mutes"][0]["audio_trim_end_ms"] == 0


def test_raw_mute_is_not_trimmed(tmp_path: Path):
    """raw_mute — диагностика «что выдал ASR»; подгонка тут врала бы о причине."""
    wav = _write_wav(tmp_path / "stream.wav", 12_000, [(2000, 2450)])
    doc = _decisions(tmp_path, wav, raw_mute=True)
    mute = doc["mutes"][0]
    assert (mute["start"], mute["end"]) == (2.0, 9.48)
    assert mute["audio_trim_end_ms"] == 0


def test_build_decisions_keeps_mute_on_quiet_mat_next_to_loud_word(tmp_path: Path):
    """Сквозная регрессия того же бага, что в trim_window: мат тихий, следующее
    слово громкое. Мьют обязан остаться на мате, а не уехать на соседа.

    Числа из реального прогона (decisions.json, vod_2842397545): «блять»
    6.866–7.105 превращался в мьют 7.075–7.225 — то есть мат целиком оставался
    слышен, а замьючено было чистое слово после него.
    """
    wav = _write_wav(
        tmp_path / "stream.wav",
        9000,
        [(6866, 7105, 1200), (7180, 7900, 16000)],
    )
    doc = _decisions(tmp_path, wav, matches=[_match(6866, 7105, timing_source="word")])
    mute = doc["mutes"][0]
    # Мат перекрыт полностью.
    assert mute["start"] <= 6.866
    assert mute["end"] >= 7.105
    # И на громкого соседа мьют не съехал.
    assert mute["end"] <= 7.18
    # Stream-шкала едет ровно на ту же дельту.
    assert mute["stream_start"] - mute["start"] == pytest.approx(60.0, abs=0.01)
    assert mute["stream_end"] - mute["end"] == pytest.approx(60.0, abs=0.01)
    # Перенос тут не при чём: речь задевает окно, значит это работа поджатия.
    assert mute["reanchor"] == "ok"


def test_build_decisions_flags_mute_that_sits_on_silence(tmp_path: Path):
    """Сквозной сценарий с реального прогона (vod_2847020280, «бля» 261.612):
    тайминги слова у ASR уехали в тишину, звук на 580 мс позже. Мьют глушил
    воздух, мат был слышен целиком — поджатие такое не лечит (оно только сужает).

    Переносить окно за человека пайплайн не берётся: автоперенос на замере
    (40 минут правленого руками материала) уводил мьют с мата чаще, чем приводил,
    см. config.mute_reanchor_move. Что он делает — помечает окно янтарным и
    расширяет его НАЗАД (config.mute_expand_on_silence_ms).

    Здесь звук ПОЗЖЕ разметки, поэтому расширение назад его не накрывает: этот
    случай так и остаётся за человеком по янтарной метке. Расширение целится в
    противоположный, системный для CTC сдвиг — сетка слов опаздывает (см.
    test_build_decisions_expands_window_onto_speech_before_the_markup)."""
    wav = _write_wav(tmp_path / "stream.wav", 6000, [(1800, 2050)])
    # Значение расширения задаём явно: тест про механизм, а не про уровень
    # (числа уровней стережёт test_sensitivity).
    doc = _decisions(
        tmp_path,
        wav,
        matches=[_match(1300, 1500, timing_source="word")],
        mute_expand_on_silence_ms=300,
    )
    mute = doc["mutes"][0]
    assert mute["reanchor"] == "silent"
    assert mute["reanchor_offset_ms"] == 0
    # Мьют остаётся мьютом (ослаблять по подозрению нельзя), но янтарным.
    assert mute["status"] == "accepted"
    assert mute["needs_review"] is True
    # Окно расширено назад на 300 мс (1.22 → 0.92) и НЕ поджато: поджимать в
    # тишине не по чему, а относительный порог там считался бы от пика шума.
    assert mute["start"] == pytest.approx(0.92, abs=0.001)
    assert mute["expand_start_ms"] == 300
    assert mute["audio_trim_start_ms"] == 0 and mute["audio_trim_end_ms"] == 0
    # Хвост на месте (ASR-граница 1.5 + padding 120 мс), обе шкалы на одной дельте.
    assert mute["end"] == pytest.approx(1.62, abs=0.001)
    assert mute["stream_start"] - mute["start"] == pytest.approx(60.0, abs=0.01)


def test_build_decisions_expands_window_onto_speech_before_the_markup(tmp_path: Path):
    """Главный сценарий расширения: сетка слов опоздала на всю длину слова, и
    разметка ПРЕДЫДУЩЕГО слова стоит ровно на звуке нашего мата (геометрия с
    реального прогона vod_2847020280, 10 минут: звук мата 76.262–76.701, разметка
    ASR 76.701–77.021).

    Из этого получался двойной капкан: клэмп головного padding не пускал мьют в
    предыдущее слово (то есть на звук мата), и окно целиком стояло на тишине.
    Поджатие такое не лечит, перенос выключен — мат был слышен полностью, а
    поднять mute_padding_before_ms не помогало вообще (прибавку съедал клэмп).

    Про сам тот случай честно: расширение его НЕ починило. Тут звук отрезан от
    окна начисто, а на реальной записи звонкий участок мата заходит в окно на
    несколько кадров — и вердикт там `ok` («речь задевает окно»), то есть триггер
    не срабатывает. Замер: после расширения таких окон остаётся 52 из 63
    непокрытых, и растягивание на задетый участок им не помогает (+41 мс
    медианно, 0 накрытых целиком) — это класс про сетку слов, не про энергию.
    Расширение лечит другую половину: окна, у которых звука в окне нет вовсе.
    """
    wav = _write_wav(tmp_path / "stream.wav", 6000, [(1000, 1380)])
    match = _match(1400, 1700, timing_source="word", prev_word_end=1400)

    # Допуск головы задаём нулём явно: тест про механизм расширения, а про их
    # взаимодействие — отдельный тест ниже (числа уровней стережёт
    # test_sensitivity).
    doc = _decisions(
        tmp_path,
        wav,
        matches=[match],
        mute_expand_on_silence_ms=300,
        mute_head_cross_prev_ms=0,
    )
    mute = doc["mutes"][0]
    # Клэмп снят, головной padding 80 + 300 → окно накрывает звук почти целиком.
    assert mute["start"] == pytest.approx(1.02, abs=0.001)
    assert mute["expand_start_ms"] == 380
    assert mute["reanchor"] == "silent"
    assert mute["needs_review"] is True
    # Обе шкалы уехали на одну дельту, raw_* (что сказал ASR) не тронуты.
    assert mute["stream_start"] - mute["start"] == pytest.approx(60.0, abs=0.01)
    assert mute["raw_local_start"] == 1.4

    # Как было до ручки: клэмп прижимал начало к 1.4 — к концу звука, — и мьют
    # стоял на чистой тишине 1.4…1.82.
    before = _decisions(
        tmp_path,
        wav,
        matches=[match],
        mute_expand_on_silence_ms=0,
        mute_head_cross_prev_ms=0,
    )["mutes"][0]
    assert before["start"] == pytest.approx(1.4, abs=0.001)
    assert before["expand_start_ms"] == 0


def test_head_allowance_can_switch_the_expansion_off_and_that_is_measured(
    tmp_path: Path,
):
    """Допуск головы и расширение на тишине — конкурирующие ручки. Так и надо.

    Геометрия та же, что в тесте выше (звук 1.0–1.38, разметка 1.4–1.7). Допуск
    пускает голову на 60 мс в разметку соседа, окно начинается с 1.34 и задевает
    звук — вердикт становится 'ok', и расширение на 300 мс не срабатывает вовсе.
    То есть 60 мс головы отменяют 380.

    Выглядит как ошибка порядка операций, но это ЗАМЕРЕННЫЙ выбор. Обратный
    вариант (судить вердикт по окну в геометрии «без допуска», чтобы допуск не мог
    его сменить) прогнан на шести правленых записях: вердикт 'на тишине'
    возвращается 7 → 10 окон на 10 минутах и 4 → 6 на 25, а линейка утечки не
    двигается (1.56 → 1.55 с) — зато съеденных чистых слов +5 на 10 минутах и +2
    на 20. Причина простая: если допуск уже достал до звука, посылка расширения
    («звука в окне нет вовсе») ложна, и 300 мс уходят в чистую речь соседа.

    Тест стоит здесь, чтобы это взаимодействие менялось осознанно, а не
    обнаруживалось потом по съеденным словам.

    Страж по звуку здесь ВЫКЛЮЧЕН явно (mute_head_cross_voiced_guard=False): он —
    отдельный механизм, появившийся позже этого замера, и без него допуск-против-
    расширения читается ровно так, как его мерили. Что делает с этой же геометрией
    страж по умолчанию — в test_voiced_guard_lets_expansion_cover_a_late_onset
    ниже (короткий ответ: голова не лезет вслепую в звук, и окно накрывает мат
    ещё полнее — через расширение)."""
    wav = _write_wav(tmp_path / "stream.wav", 6000, [(1000, 1380)])
    match = _match(1400, 1700, timing_source="word", prev_word_end=1400)
    mute = _decisions(
        tmp_path,
        wav,
        matches=[match],
        mute_expand_on_silence_ms=300,
        mute_head_cross_prev_ms=60,
        mute_head_cross_voiced_guard=False,
    )["mutes"][0]
    assert mute["start"] == pytest.approx(1.34, abs=0.001)
    assert mute["head_cross_ms"] == 60
    assert mute["expand_start_ms"] == 0
    assert mute["reanchor"] == "ok"


def test_expansion_does_not_touch_a_window_that_has_sound(tmp_path: Path):
    """Обратная гарантия: где звук в окне ЕСТЬ, расширение не работает, а клэмп
    по предыдущему слову держит голову. Иначе расширение обнулило бы защиту
    соседней чистой речи на всех окнах, а не только на стоящих на тишине."""
    wav = _write_wav(tmp_path / "stream.wav", 6000, [(1000, 1400)])
    match = _match(1000, 1400, timing_source="word", prev_word_end=980)
    doc = _decisions(tmp_path, wav, matches=[match], mute_head_cross_prev_ms=0)
    mute = doc["mutes"][0]
    assert mute["reanchor"] == "ok"
    assert mute["expand_start_ms"] == 0 and mute["expand_end_ms"] == 0
    # Начало не залезло в предыдущее слово (padding 80 мс схлопнут клэмпом до
    # 0.98) — и поджатие по звуку тут работает как обычно.
    assert mute["start"] >= 0.98
    assert mute["head_cross_ms"] == 0

    # С допуском голова заходит на соседа, но ровно на разрешённые мс — и это
    # видно в head_cross_ms, а не только в границе.
    with_cross = _decisions(
        tmp_path, wav, matches=[match], mute_head_cross_prev_ms=60
    )["mutes"][0]
    assert with_cross["start"] >= 0.98 - 0.060
    assert with_cross["head_cross_ms"] <= 60


def test_voiced_guard_collapses_the_head_cross_on_a_contiguous_neighbour(
    tmp_path: Path,
):
    """Ровно та правка, из-за которой всё затевалось: связная речь, чистое слово
    ВПЛОТНУЮ к мату, форсированное выравнивание уже поставило начало мата верно.

    Геометрия — норма сетки ASR: звук идёт непрерывно 0.7–1.4, разметка соседа
    кончается в 1.0, там же начало мата (prev_word_end == onset == 1.0). Допуск
    головы 60 мс снимает клэмп и пускает мьют назад в соседа — но эти 60 мс НЕ
    тишина, а звучащий хвост чистого слова. Поджатие их не трогает (это звук), и
    человек снимал их руками запись за записью.

    Со стражем (по умолчанию) перед началом мата тишины нет — 0.94–1.0 звучит, —
    поэтому допуск урезается в ноль, и голова прижимается к началу мата: чистое
    слово соседа не замьючено. Без стража (как было) голова заходит на ~60 мс в
    его звук. Разница видна в head_cross_ms, которого больше нигде нет.
    """
    wav = _write_wav(tmp_path / "stream.wav", 6000, [(700, 1400)])
    match = _match(1000, 1400, timing_source="word", prev_word_end=1000)

    guarded = _decisions(
        tmp_path, wav, matches=[match], mute_head_cross_prev_ms=60
    )["mutes"][0]
    blind = _decisions(
        tmp_path,
        wav,
        matches=[match],
        mute_head_cross_prev_ms=60,
        mute_head_cross_voiced_guard=False,
    )["mutes"][0]

    # Страж: голова у начала мата, в звучащего соседа не залезла.
    assert guarded["head_cross_ms"] == 0
    assert guarded["start"] == pytest.approx(1.0, abs=0.02)
    # Без стража: те самые ~60 мс чистого слова под мьютом.
    assert blind["head_cross_ms"] == pytest.approx(60, abs=12)
    assert blind["start"] < guarded["start"]
    # Мат при этом накрыт в обоих случаях — страж ослабляет не мьют мата, а только
    # захват соседа (сам мат звучит с 1.0).
    assert guarded["end"] >= 1.4 and blind["end"] >= 1.4


def test_voiced_guard_lets_expansion_cover_a_late_onset(tmp_path: Path):
    """Обратный случай к тесту про допуск-против-расширения: там страж ВЫКЛЮЧЕН,
    здесь он по умолчанию ВКЛЮЧЁН, геометрия одна (звук 1.0–1.38, разметка 1.4–1.7,
    сетка слов опоздала на всю длину слова).

    Смысл: страж не мешает накрыть уехавший мат, он лишь запрещает ГОЛОВЕ лезть в
    звук вслепую. Перед началом мата (1.4) тишины только 20 мс (1.38–1.4), поэтому
    допуск 60 мс урезается до 20 — голова не достаёт до звука, вердикт остаётся
    'на тишине', и мат накрывает РАСШИРЕНИЕ (оно судит окно без клэмпов и стража).
    В итоге окно уезжает к 1.02 — ДАЛЬШЕ, чем слепой допуск (1.34), то есть звук
    мата 1.0–1.38 накрыт полнее, а не хуже. Это и есть «страж уступает расширению»,
    а не «страж мешает».
    """
    wav = _write_wav(tmp_path / "stream.wav", 6000, [(1000, 1380)])
    match = _match(1400, 1700, timing_source="word", prev_word_end=1400)
    mute = _decisions(
        tmp_path,
        wav,
        matches=[match],
        mute_expand_on_silence_ms=300,
        mute_head_cross_prev_ms=60,
    )["mutes"][0]

    assert mute["reanchor"] == "silent"
    assert mute["start"] == pytest.approx(1.02, abs=0.02)
    assert mute["expand_start_ms"] > 300
    # Накрыто ПОЛНЕЕ слепого допуска (тот давал 1.34) — начало звука 1.0 почти у края.
    assert mute["start"] < 1.34


def test_build_decisions_flags_mute_with_no_speech_around(tmp_path: Path):
    """Окно на тишине, речи в радиусе поиска нет вовсе: статус не понижаем
    (ослаблять мьют по подозрению нельзя), янтарный флаг ставим — слушать это
    человеку.

    Расширяем и здесь: «звука нет» по АБСОЛЮТНОМУ порогу — это и «мат сказан
    тише порога» тоже (на правленых записях у пяти окон, накрытых человеком
    целиком, звука по этому порогу «нет»), а цена ошибки расширения — лишние
    300 мс мьюта."""
    wav = _write_wav(tmp_path / "stream.wav", 6000, [(4000, 4300)])
    doc = _decisions(
        tmp_path,
        wav,
        matches=[_match(1300, 1500, timing_source="word")],
        mute_expand_on_silence_ms=300,
    )
    mute = doc["mutes"][0]
    assert mute["reanchor"] == "no_speech"
    assert mute["reanchor_offset_ms"] == 0
    assert mute["status"] == "accepted"
    assert mute["needs_review"] is True
    # Расширено назад на 300 мс и не поджато (в тишине поджимать не по чему —
    # ровно та причина, по которой «стоит ли мьют на звуке» нельзя спросить у
    # поджатия и нужен абсолютный порог).
    assert mute["start"] == pytest.approx(0.92, abs=0.001)
    assert mute["expand_start_ms"] == 300
    assert mute["audio_trim_start_ms"] == 0 and mute["audio_trim_end_ms"] == 0
    assert mute["end"] <= 1.62


def test_expansion_can_be_disabled(tmp_path: Path):
    """mute_expand_on_silence_ms=0 (уровень low) → прежнее поведение: окно на
    тишине только помечается, границы остаются в разметке ASR."""
    wav = _write_wav(tmp_path / "stream.wav", 6000, [(1800, 2050)])
    doc = _decisions(
        tmp_path,
        wav,
        matches=[_match(1300, 1500, timing_source="word")],
        mute_expand_on_silence_ms=0,
    )
    mute = doc["mutes"][0]
    assert mute["reanchor"] == "silent"
    assert mute["needs_review"] is True
    assert mute["expand_start_ms"] == 0 and mute["expand_end_ms"] == 0
    assert mute["start"] >= 1.22 and mute["end"] <= 1.62


def test_build_decisions_reanchors_mute_when_move_is_enabled(tmp_path: Path):
    """`mute_reanchor_move=True` — перенос для тех, кто замерил его на своём
    материале: окно едет на речь целиком, обе шкалы на одну дельту."""
    wav = _write_wav(tmp_path / "stream.wav", 6000, [(1800, 2050)])
    doc = _decisions(
        tmp_path,
        wav,
        matches=[_match(1300, 1500, timing_source="word")],
        mute_reanchor_move=True,
    )
    mute = doc["mutes"][0]
    assert mute["reanchor"] == "reanchored"
    assert mute["reanchor_offset_ms"] > 0
    # Мьют накрыл реальный звук.
    assert mute["start"] <= 1.81
    assert mute["end"] >= 2.05
    # Обе шкалы уехали на одну дельту, raw_* (что сказал ASR) не тронуты.
    assert mute["stream_start"] - mute["start"] == pytest.approx(60.0, abs=0.01)
    assert mute["stream_end"] - mute["end"] == pytest.approx(60.0, abs=0.01)
    assert mute["raw_local_start"] == 1.3
    # Мьют остаётся мьютом: перенос не понижает статус.
    assert mute["status"] == "accepted"
    assert doc["cuts"][0]["start"] == mute["start"]


def test_build_decisions_reanchor_can_be_disabled(tmp_path: Path):
    """mute_reanchor_ms=0 → абсолютный порог не считается, окна не двигаются."""
    wav = _write_wav(tmp_path / "stream.wav", 6000, [(1800, 2050)])
    doc = _decisions(
        tmp_path,
        wav,
        matches=[_match(1300, 1500, timing_source="word")],
        mute_reanchor_ms=0,
    )
    mute = doc["mutes"][0]
    assert mute["reanchor"] == "ok"
    assert mute["reanchor_offset_ms"] == 0
    # Окно осталось на месте (в границах ASR 1.22–1.62), звук на 1.8–2.05 так и
    # не заглушён — то самое «как было до переноса».
    assert mute["start"] >= 1.22
    assert mute["end"] <= 1.62
    assert mute["needs_review"] is False
