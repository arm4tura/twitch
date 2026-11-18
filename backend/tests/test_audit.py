"""Тесты `twitch-cut audit` — сверки «где мат» с «где нашло приложение».

Аудит нужен, чтобы не проверять детекцию на слух каждый раз. Сам он тоже может
врать, поэтому здесь синтетика с ЗАРАНЕЕ известным ответом: WAV из тона и
тишины, транскрипт с точными таймингами, decisions с нарочно кривыми мьютами.
"""

from __future__ import annotations

from array import array
import json
import math
from pathlib import Path
import wave

from twitch_cut import audit as audit_mod

RATE = 16000


def _write_wav(
    path: Path, spans: list[tuple[float, float]], duration: float, *, hum: int = 0
) -> None:
    """WAV 16 kHz mono: `spans` — участки громкого тона, остальное тишина.

    `hum` — амплитуда ровного фона вместо цифрового нуля. Нужен там, где важна
    разница между двумя порогами: относительный порог в фоне «находит» речь
    (пиком становится сам фон), абсолютный — нет.
    """
    total = int(duration * RATE)
    samples = array("h", [0]) * total
    if hum:
        for i in range(total):
            samples[i] = int(hum * math.sin(2 * math.pi * 110 * i / RATE))
    for start, end in spans:
        s0 = max(0, int(start * RATE))
        s1 = min(total, int(end * RATE))
        for i in range(s0, s1):
            samples[i] = int(12000 * math.sin(2 * math.pi * 220 * i / RATE))
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(RATE)
        wav.writeframes(samples.tobytes())


def _transcript(words: list[tuple[str, float, float]]) -> dict:
    return {
        "segments": [
            {
                "id": "seg_000000",
                "start": words[0][1],
                "end": words[-1][2],
                "words": [
                    {"word": text, "start": start, "end": end}
                    for text, start, end in words
                ],
            }
        ]
    }


def _mute(start: float, end: float, word: str, **extra) -> dict:
    base = {
        "id": f"mute_{int(start * 1000):06d}",
        "start": start,
        "end": end,
        "action": "MUTE",
        "status": "accepted",
        "word": word,
        "timing_source": "word",
        "match_type": "surface",
        "audio_trim_start_ms": 0,
        "audio_trim_end_ms": 0,
    }
    base.update(extra)
    return base


def _project(
    tmp_path: Path,
    *,
    words: list[tuple[str, float, float]],
    mutes: list[dict],
    voiced: list[tuple[float, float]],
    duration: float = 10.0,
    hum: int = 0,
) -> Path:
    transcript_path = tmp_path / "transcript.json"
    transcript_path.write_text(json.dumps(_transcript(words)), encoding="utf-8")
    audio_path = tmp_path / "audio.wav"
    _write_wav(audio_path, voiced, duration, hum=hum)
    doc = {
        "schema_version": "1.1",
        "range_in": "00:00:00.000",
        "range_out": "00:00:10.000",
        "local_duration": "00:00:10.000",
        "settings": {"mute_trim_to_audio": True},
        "summary": {"raw_matches": len(mutes)},
        "mutes": mutes,
        "cuts": [],
        "caches": {"transcript": str(transcript_path), "audio": str(audio_path)},
    }
    decisions = tmp_path / "decisions.json"
    decisions.write_text(json.dumps(doc), encoding="utf-8")
    return decisions


WORDS = [
    ("это", 0.5, 0.8),
    ("нихуя", 1.0, 1.4),
    ("привет", 3.0, 3.3),
    ("блять", 5.0, 5.4),
    ("конец", 8.6, 8.9),
]
VOICED = [(0.5, 0.8), (1.0, 1.4), (3.0, 3.3), (5.0, 5.4), (6.6, 8.0), (8.6, 8.9)]


def _default_project(tmp_path: Path) -> Path:
    return _project(
        tmp_path,
        words=WORDS,
        mutes=[
            _mute(2.9, 3.4, "привет"),  # мьют на чистом слове — ложное срабатывание
            _mute(5.0, 6.4, "блять"),  # мат закрыт, но хвост тянется по тишине
        ],
        voiced=VOICED,
    )


def test_detector_and_audit_share_one_matcher():
    """Раньше отчёт видел корень внутри слова («нихуя»), а детектор — нет, и
    расхождение приходилось объяснять диагнозами. Теперь матчер ОДИН, поэтому
    «отчёт знает корень, которого не знает детектор» невозможно by construction."""
    from twitch_cut.profanity import match_root, roots_in

    assert match_root("нихуя") == "хуя"
    assert audit_mod.roots_in("нихуя") == ["хуя"]
    assert audit_mod.roots_in is roots_in


def test_innocent_homographs_are_not_reported():
    assert audit_mod.roots_in("щебень") == []
    assert audit_mod.roots_in("ребенок") == []
    assert audit_mod.roots_in("барсука", audit_mod.SPICY_ROOTS) == []
    # «бля» внутри длинного глагола — не мат.
    assert audit_mod.roots_in("оскорблять") == []
    assert audit_mod.roots_in("потребляет") == []
    # А сам мат и его формы — мат.
    assert audit_mod.roots_in("блять") == ["бля"]
    assert audit_mod.roots_in("блядский") == ["бля"]


def test_missed_profanity_is_found(tmp_path):
    result = audit_mod.analyze(_default_project(tmp_path))

    missed = [m for m in result.missed if m.kind == "core"]
    assert [m.word.text for m in missed] == ["нихуя"]
    # Корень известен детектору (матчер общий) → это потеря в пайплайне, а не
    # дырка в словаре. Раньше здесь был None и диагноз «дыра словаря».
    assert missed[0].known_stem == "хуя"


def test_mute_without_obscene_root_is_flagged(tmp_path):
    result = audit_mod.analyze(_default_project(tmp_path))

    clean_word_mute = result.mutes[0]
    assert clean_word_mute.word_text == "привет"
    assert any("НЕТ обсценного корня" in f for f in clean_word_mute.flags)
    # А мьют на настоящем мате в этом не обвиняется.
    assert not any("НЕТ обсценного корня" in f for f in result.mutes[1].flags)


def test_recorded_word_saves_a_mute_whose_covered_text_looks_clean(tmp_path):
    """`word` мьюта — находка ДЕТЕКТОРА, и отчёт на неё опирается как на факт.

    Под окном могут лежать чистые токены (тайминги ASR уехали, мат склеен,
    услышано не то), и тогда корень ищется в записанном слове мьюта — иначе
    честный мьют попадает в «подозрительные границы» как ложное срабатывание.
    Ровно это и происходило, когда экран правки пересчитывал `word` из
    транскрипта на каждый сдвиг окна: на реальном прогоне (vod_2847020280) все 5
    тронутых человеком мьютов теряли слово детектора, и 2 из 35 получали флаг
    «НЕТ обсценного корня» — то есть правка границ фабриковала ложные диагнозы.
    """
    words = [("с", 1.0, 1.2), ("этим", 1.2, 1.5)]
    detected = audit_mod.analyze(
        _project(tmp_path, words=words, mutes=[_mute(1.0, 1.5, "бля")], voiced=[(1.0, 1.5)])
    )

    item = detected.mutes[0]
    assert [w.text for w in item.words] == ["с", "этим"]
    assert not [w for w in item.words if w.core]  # в перекрытых словах корня нет
    assert not any("НЕТ обсценного корня" in f for f in item.flags)

    # Тот же мьют, но со словом, затёртым накрытым текстом, — уже «ложное».
    second = tmp_path / "overwritten"
    second.mkdir()
    overwritten = audit_mod.analyze(
        _project(
            second,
            words=words,
            mutes=[_mute(1.0, 1.5, "с этим")],
            voiced=[(1.0, 1.5)],
        )
    )
    assert any("НЕТ обсценного корня" in f for f in overwritten.mutes[0].flags)


def test_tail_silence_is_measured_from_audio(tmp_path):
    result = audit_mod.analyze(_default_project(tmp_path))

    fit = result.mutes[1].fit
    assert fit is not None
    # Речь 5.0–5.4, окно до 6.4 → хвоста ровно секунда (±кадр).
    assert 900 <= fit.tail_ms <= 1100
    assert fit.head_ms <= 50
    assert any("хвост тишины" in f for f in result.mutes[1].flags)


def test_speech_crossing_the_right_edge_is_flagged(tmp_path):
    """Мьют кончился, а слово по ASR длиннее — верный признак торчащего мата."""
    decisions = _project(
        tmp_path,
        words=[("хуйня", 1.0, 2.0), ("дальше", 2.2, 2.6)],
        mutes=[_mute(1.0, 1.5, "хуйня")],
        voiced=[(1.0, 2.0), (2.2, 2.6)],
        duration=4.0,
    )

    result = audit_mod.analyze(decisions)

    fit = result.mutes[0].fit
    assert fit is not None and fit.hot_right
    assert any("хвост слова слышно" in f for f in result.mutes[0].flags)


def test_contiguous_neighbour_speech_is_not_a_complaint(tmp_path):
    """Слова вплотную — норма, а не «мат торчит». Иначе флаг горел бы на всём:
    padding мьюта почти всегда заезжает в соседнее слово."""
    decisions = _project(
        tmp_path,
        words=[("это", 1.0, 1.4), ("хуйня", 1.4, 1.8), ("дальше", 1.8, 2.2)],
        mutes=[_mute(1.32, 1.9, "хуйня")],
        voiced=[(1.0, 2.2)],
        duration=4.0,
    )

    result = audit_mod.analyze(decisions)

    fit = result.mutes[0].fit
    # Энергия на границах действительно непрерывна…
    assert fit is not None and fit.hot_left and fit.hot_right
    # …но объясняется соседними словами, поэтому замечаний нет.
    assert result.mutes[0].flags == []


def test_asr_holes_are_found(tmp_path):
    result = audit_mod.analyze(_default_project(tmp_path))

    assert result.audio_ok
    holes = result.holes
    assert len(holes) == 1, [(h.start_ms, h.end_ms) for h in holes]
    # Дыра — весь промежуток без слов (5.4–8.6), громкий кусок внутри 6.6–8.0.
    assert 5300 <= holes[0].start_ms <= 5500
    assert 8500 <= holes[0].end_ms <= 8700
    assert holes[0].voiced > 0.35


# --- почему слово осталось без мьюта: вердикт по звуку ------------------------


def _missed_one(tmp_path: Path, *, mute: tuple[float, float], voiced, word=(1.0, 1.4)):
    """Проект с одним матом, одним мьютом и заданной картиной звука.

    Мьют нарочно не накрывает СЕРЕДИНУ слова — по прежнему критерию это
    считалось пропуском всегда, каким бы ни был звук.
    """
    decisions = _project(
        tmp_path,
        words=[("это", 0.4, 0.7), ("блять", word[0], word[1]), ("дальше", 2.4, 2.8)],
        mutes=[_mute(mute[0], mute[1], "блять")],
        voiced=voiced,
        duration=4.0,
    )
    result = audit_mod.analyze(decisions)
    core = [m for m in result.missed if m.kind == "core"]
    assert len(core) == 1, [(m.word.text, m.diagnosis) for m in core]
    return result, core[0]


def test_mute_trimmed_to_speech_is_not_called_a_miss(tmp_path):
    """Тайминг ASR длиннее слова: звук 1.0–1.15, ASR отдал 1.0–1.4, мьют поджат
    по звуку. Середина слова (1.2) формально вне мьюта, но там тишина — мат
    закрыт, и обвинять пайплайн не за что."""
    _, item = _missed_one(
        tmp_path, mute=(0.95, 1.18), voiced=[(0.4, 0.7), (1.0, 1.15), (2.4, 2.8)]
    )

    assert item.diagnosis == "trimmed_to_speech"
    assert item.real is False
    assert item.voiced_outside == 0.0
    assert item.peak_covered is True


def test_mute_that_misses_the_loud_part_is_a_real_miss(tmp_path):
    """Зеркало предыдущего: звук слова 1.25–1.4, а мьют встал на 0.95–1.15 —
    громкая часть мата осталась открытой."""
    _, item = _missed_one(
        tmp_path, mute=(0.95, 1.15), voiced=[(0.4, 0.7), (1.25, 1.4), (2.4, 2.8)]
    )

    assert item.diagnosis == "trimmed_off"
    assert item.real is True
    assert item.peak_covered is False
    assert (item.voiced_outside or 0.0) > audit_mod.VOICE_LEFT_PCT


def test_word_timing_pointing_into_silence_is_named_as_such(tmp_path):
    """Границы слова уехали в тишину: звук 0.55–0.95, ASR отдал 1.0–1.4, мьют
    встал по ASR — то есть в тишину, а мат звучит рядом и слышен.

    Взято с реального стрима: «блять» 337.806 по ASR, а звук 337.32–337.71.
    Прежний отчёт про такое писал «мьют потерялся дальше (статус/группировка/
    потолки)» — неверный адрес: и статус, и группировка тут в порядке."""
    result, item = _missed_one(
        tmp_path, mute=(1.0, 1.15), voiced=[(0.55, 0.95), (2.4, 2.8)]
    )

    assert item.diagnosis == "asr_timing_off"
    assert item.real is True
    assert item.span_ratio is not None and item.span_ratio < audit_mod.SILENT_SPAN_RATIO
    # В отчёте видно, куда именно уехал тайминг — иначе на слух не проверить.
    assert item.voice_span is not None
    assert 500 <= item.voice_span[0] <= 600
    assert 950 <= item.voice_span[1] <= 1050
    report = audit_mod.render_report(result)
    assert "уехали в тишину" in report
    assert "звук на самом деле 00:00.5" in report
    assert "статус/группировка" not in report


def test_reanchored_mute_is_not_reported_as_a_miss(tmp_path):
    """Тот же уехавший тайминг, но окно ДОТЯНУЛОСЬ до звука — это не пропуск.

    Пайплайн умеет расширять назад мьют, стоящий на тишине
    (`config.mute_expand_on_silence_ms`), а с включённым `mute_reanchor_move` —
    и переносить его на ближайшую речь. В обоих случаях окно СПЕЦИАЛЬНО не
    совпадает с границами слова: в них тишина. Отчёт ищет пропуски именно по
    границам слова, поэтому удачная починка читалась как «мат звучит незакрытым» —
    на реальном прогоне (vod_2847020280, «бля» 261.612 → 262.172) отчёт звал
    человека править окно, которое программа уже поставила ровно на тот звук, куда
    сама и указала.
    """
    result, item = _missed_one(
        tmp_path, mute=(0.55, 0.95), voiced=[(0.55, 0.95), (2.4, 2.8)]
    )

    assert item.diagnosis == "reanchored"
    assert item.real is False
    assert item.voice_covered is True
    report = audit_mod.render_report(result)
    assert "Настоящих пропусков нет" in report
    assert "дотянулся до измеренного звука" in report
    assert "править окно не нужно" in report
    # Починка объясняется, но НЕ как раздел пропусков: чинить тут нечего.
    assert "мьют стоит не на звуке" not in report


def test_report_does_not_blame_pipeline_for_a_trimmed_mute(tmp_path):
    """Вердикт из отчёта, а не только из полей: у аккуратно поджатого мьюта
    претензий нет вовсе."""
    result, _ = _missed_one(
        tmp_path, mute=(0.95, 1.18), voiced=[(0.4, 0.7), (1.0, 1.15), (2.4, 2.8)]
    )

    report = audit_mod.render_report(result)
    assert "Настоящих пропусков нет" in report
    assert "статус/группировка" not in report


def test_word_without_any_mute_still_blames_detection(tmp_path):
    """Обратная страховка: если мьюта на слове нет и звук на месте — это
    настоящий пропуск, и адрес прежний (детекция/словарь)."""
    decisions = _project(
        tmp_path,
        words=[("это", 0.4, 0.7), ("нихуя", 1.0, 1.4)],
        mutes=[],
        voiced=[(0.4, 0.7), (1.0, 1.4)],
        duration=4.0,
    )
    result = audit_mod.analyze(decisions)

    item = next(m for m in result.missed if m.kind == "core")
    assert item.diagnosis == "no_mute"
    assert item.real is True
    # Корень «хуя» детектору известен (матчер общий), поэтому адрес — потери
    # после детекции, а не prefix-only поиск.
    assert item.known_stem == "хуя"
    assert "детектор корень ЗНАЛ" in audit_mod.render_report(result)


def test_span_measurements_survive_missing_audio(tmp_path):
    """Без аудио вердикт не выдумывается: мьют есть — 'partial', мьюта нет —
    'no_mute'."""
    decisions = _project(
        tmp_path,
        words=[("блять", 1.0, 1.4)],
        mutes=[_mute(0.95, 1.18, "блять")],
        voiced=[(1.0, 1.15)],
        duration=4.0,
    )
    doc = json.loads(decisions.read_text(encoding="utf-8"))
    doc["caches"]["audio"] = str(tmp_path / "missing.wav")
    decisions.write_text(json.dumps(doc), encoding="utf-8")

    result = audit_mod.analyze(decisions)

    assert result.audio_ok is False
    item = next(m for m in result.missed if m.kind == "core")
    assert item.diagnosis == "partial"
    assert item.span_ratio is None


def test_well_fitted_mute_has_no_complaints(tmp_path):
    """Контроль-позитив: если мьют аккуратно обнимает мат, замечаний быть НЕ
    должно. Без этого теста флаги могли бы срабатывать всегда и на всё."""
    decisions = _project(
        tmp_path,
        words=[("это", 0.4, 0.8), ("хуйня", 1.5, 1.9), ("дальше", 3.0, 3.4)],
        mutes=[_mute(1.42, 1.98, "хуйня")],
        voiced=[(0.4, 0.8), (1.5, 1.9), (3.0, 3.4)],
        duration=5.0,
    )

    result = audit_mod.analyze(decisions)

    assert result.mutes[0].flags == []
    assert [m.word.text for m in result.missed] == []


def test_silent_gap_is_not_a_hole(tmp_path):
    """Дыра — это ЗВУК без слов. Просто пауза дырой считаться не должна."""
    decisions = _project(
        tmp_path,
        words=[("это", 0.4, 0.8), ("конец", 8.0, 8.4)],
        mutes=[],
        voiced=[(0.4, 0.8), (8.0, 8.4)],
        duration=9.0,
    )

    result = audit_mod.analyze(decisions)

    assert result.audio_ok
    assert result.holes == []


def test_timestamp_parsing():
    assert audit_mod.parse_timestamp("5.9") == 5900
    assert audit_mod.parse_timestamp("01:23.400") == 83400
    assert audit_mod.parse_timestamp("1:02:03,5") == 3723500
    assert audit_mod.parse_timestamp("привет") is None


def test_ground_truth_diff(tmp_path):
    decisions = _default_project(tmp_path)
    truth = tmp_path / "ground_truth.txt"
    truth.write_text(
        "# реальные таймкоды\n"
        "00:05.1 блять\n"  # попадание — мьют 5.0–6.4
        "00:01.2 нихуя   # это ASR услышал, но мьюта нет\n"
        "мусор\n",
        encoding="utf-8",
    )

    result = audit_mod.analyze(decisions, truth_path=truth, tolerance_ms=300)

    assert [p.note for p, _ in result.truth_hits] == ["блять"]
    assert [p.note for p in result.truth_misses] == ["нихуя"]
    # Мьют на «привет» в список не попал — при полном списке это лишний мьют.
    assert [m.word_text for m in result.truth_extras] == ["привет"]
    assert result.truth_problems and "мусор" in result.truth_problems[0]


def test_report_is_written_and_readable(tmp_path):
    result = audit_mod.analyze(_default_project(tmp_path))

    report = audit_mod.write_report(result)

    assert report == tmp_path / "audit.md"
    text = report.read_text(encoding="utf-8")
    assert "## Пропуски, видимые прямо в транскрипте" in text
    assert "нихуя" in text
    assert "## Дыры ASR" in text
    # Без ground_truth отчёт обязан объяснить, как его завести.
    assert "ground_truth.txt" in text
    assert "MM:SS.mmm" in text


def test_missing_transcript_is_a_clear_error(tmp_path):
    decisions = _default_project(tmp_path)
    doc = json.loads(decisions.read_text(encoding="utf-8"))
    doc["caches"]["transcript"] = str(tmp_path / "nope.json")
    decisions.write_text(json.dumps(doc), encoding="utf-8")

    try:
        audit_mod.analyze(decisions)
    except audit_mod.AuditError as exc:
        assert "транскрипт" in str(exc)
    else:  # pragma: no cover - защита от молчаливого прохода
        raise AssertionError("ожидали AuditError")


def test_audio_absence_does_not_break_the_audit(tmp_path):
    decisions = _default_project(tmp_path)
    doc = json.loads(decisions.read_text(encoding="utf-8"))
    doc["caches"]["audio"] = str(tmp_path / "gone.wav")
    decisions.write_text(json.dumps(doc), encoding="utf-8")

    result = audit_mod.analyze(decisions)

    assert not result.audio_ok
    assert all(m.fit is None for m in result.mutes)
    assert result.holes == []
    # Текстовая часть проверок при этом работает.
    assert [m.word.text for m in result.missed if m.kind == "core"] == ["нихуя"]
    assert "проверки границ по звуку пропущены" in audit_mod.render_report(result)


# --- журнал ручных правок ----------------------------------------------------

PROGRAM_MUTES = [
    _mute(2.9, 3.4, "привет"),  # ложное срабатывание
    _mute(5.0, 6.4, "блять"),  # хвост тянется по тишине
]


def _edited_project(tmp_path: Path) -> Path:
    """Проект, где человек уже всё поправил, + журнал с решением программы.

    Правки покрывают все четыре вида диагноза: снятое ложное, поджатый хвост,
    добавленный инфиксный мат (детектор его не знает) и добавленный мьют там,
    где ASR не распознал ни одного слова.
    """
    from twitch_cut.edit_journal import start_journal

    edited = [
        # «привет» снят целиком.
        _mute(5.0, 5.5, "блять"),  # хвост поджат вручную (id тот же)
        _mute(1.0, 1.4, "нихуя", id="mute_manual_1000", source="manual"),
        _mute(6.8, 7.2, "", id="mute_manual_6800", source="manual"),
    ]
    decisions = _project(tmp_path, words=WORDS, mutes=edited, voiced=VOICED)
    start_journal(decisions, {"mutes": PROGRAM_MUTES, "silences": [], "offtopic": {"segments": []}})
    return decisions


def test_edits_are_picked_up_and_diagnosed(tmp_path):
    result = audit_mod.analyze(_edited_project(tmp_path))

    assert result.journal is not None
    by_kind = {}
    for finding in result.edits:
        by_kind.setdefault(finding.kind, []).append(finding)

    # Снятый мьют на чистом слове — назван ложным срабатыванием.
    (removed,) = by_kind["removed"]
    assert removed.start_ms == 2900
    assert "ложное срабатывание" in removed.verdict

    added = sorted(by_kind["added"], key=lambda f: f.start_ms)
    # Инфиксный мат: корень детектору известен, значит мьют потерялся дальше.
    assert added[0].start_ms == 1000
    assert "детектор корень ЗНАЛ (`хуя`)" in added[0].verdict
    # Место, где ASR не распознал слов вообще.
    assert added[1].start_ms == 6800
    assert "НЕТ СЛОВ" in added[1].verdict

    # Поджатый хвост — это сдвиг, а не пара «удалил/добавил».
    (moved,) = by_kind["moved"]
    assert (moved.base_start_ms, moved.base_end_ms) == (5000, 6400)
    assert moved.d_end_ms == -900
    assert "съедал лишнее" in moved.verdict


def test_edits_section_lands_in_the_report(tmp_path):
    result = audit_mod.analyze(_edited_project(tmp_path))

    text = audit_mod.render_report(result)

    assert "## Твои правки против того, что сделала программа" in text
    assert "снимок решения ПАЙПЛАЙНА" in text
    assert "### Программа НЕ нашла — ты добавил мьют (2)" in text
    assert "### Ты снял мьют (1)" in text
    assert "### Ты поправил границы (1)" in text
    # Готовые строки для ground_truth.txt — чтобы правки проверялись машинно.
    assert "00:01.000-00:01.400  нихуя" in text


def test_report_without_journal_explains_how_to_enable_it(tmp_path):
    text = audit_mod.render_report(audit_mod.analyze(_default_project(tmp_path)))
    assert "Журнала правок нет" in text
    assert "decisions.edits.json" in text


def test_explicit_edits_path_is_used(tmp_path):
    """`audit --edits` — журнал можно принести откуда угодно (например, из копии
    папки прогона)."""
    from twitch_cut.edit_journal import start_journal

    decisions = _project(
        tmp_path,
        words=WORDS,
        mutes=[_mute(5.0, 5.5, "блять")],
        voiced=VOICED,
    )
    elsewhere = tmp_path / "sub"
    elsewhere.mkdir()
    start_journal(
        elsewhere / "decisions.json",
        {"mutes": PROGRAM_MUTES, "silences": [], "offtopic": {"segments": []}},
    )

    result = audit_mod.analyze(decisions, edits_path=elsewhere / "decisions.edits.json")

    assert result.journal is not None
    assert {f.kind for f in result.edits} == {"removed", "moved"}


def _save(decisions: Path, mutate) -> None:
    """Сохранить документ так, как это делает PUT /decisions (с журналом)."""
    from twitch_cut.edit_journal import record_save

    previous = json.loads(decisions.read_text(encoding="utf-8"))
    doc = json.loads(json.dumps(previous))
    mutate(doc)
    decisions.write_text(json.dumps(doc), encoding="utf-8")
    record_save(decisions, doc, previous)


def test_auto_detected_silences_are_not_called_human_edits(tmp_path):
    """Тишину насыпает авто-детект в «Правке», а сохраняет её человек. Отчёт не
    имеет права записывать эту работу программы человеку в правки — иначе по
    журналу не понять, что человек на самом деле переделал."""
    from twitch_cut.edit_journal import start_journal

    decisions = _project(tmp_path, words=WORDS, mutes=[], voiced=VOICED)
    start_journal(decisions, json.loads(decisions.read_text(encoding="utf-8")))

    def add_silences(doc):
        doc["silences"] = [
            {"id": "silence_auto_6600", "start": 6.6, "end": 8.0, "status": "accepted"},
            {"id": "silence_manual_3400", "start": 3.4, "end": 3.9, "status": "accepted"},
        ]

    _save(decisions, add_silences)

    text = audit_mod.render_report(audit_mod.analyze(decisions))
    assert "тишина — добавлено вручную: 1" in text
    assert "насыпал авто-детект" in text


def test_machine_offtopic_lands_in_baseline_not_in_edits(tmp_path):
    """AI-детект оффтопа пишет decisions.json мимо PUT — после обновления
    базового снимка его сегменты не выглядят как «человек добавил»."""
    from twitch_cut.edit_journal import refresh_baseline, start_journal

    decisions = _project(tmp_path, words=WORDS, mutes=[], voiced=VOICED)
    start_journal(decisions, json.loads(decisions.read_text(encoding="utf-8")))

    doc = json.loads(decisions.read_text(encoding="utf-8"))
    doc["offtopic"] = {"segments": [{"start_s": 3.0, "end_s": 6.0, "stage": "cheap"}]}
    decisions.write_text(json.dumps(doc), encoding="utf-8")
    refresh_baseline(decisions, doc, ("offtopic",), note="AI-детект оффтопа (source)")

    result = audit_mod.analyze(decisions)

    assert not any("оффтоп — добавлено" in note for note in result.edit_notes)
    text = audit_mod.render_report(result)
    assert "базовый снимок «оффтоп» обновлён отдельно" in text
    assert "AI-детект оффтопа (source)" in text


# --- диагноз добавленного человеком мьюта ------------------------------------


def _word(text: str) -> audit_mod.Word:
    return audit_mod.Word(
        idx=0,
        text=text,
        norm=text,
        start_ms=0,
        end_ms=300,
        segment_id="seg_000000",
        timing_source="word",
    )


def test_added_mute_with_known_root_blames_the_pipeline_not_the_dictionary():
    """«гандоны» и «нехуевая» — корень детектору ИЗВЕСТЕН (общий словарь, общий
    матчер, инфикс разрешён). Значит мьюта нет не из-за поиска, а из-за потерь
    ПОСЛЕ детекции: статус review, группировка, потолки, подгонка границ."""
    for text, root in (("гандоны", "гандон"), ("нехуевая", "хуе"), ("заебись", "заеб")):
        word = _word(text)
        roots, known = audit_mod._roots_of([word])

        assert roots and known == root, text
        verdict = audit_mod._verdict_added([word], roots, known)
        assert f"`{root}`" in verdict and "ПОСЛЕ детекции" in verdict, text
        # старых диагнозов больше не существует: расхождение невозможно
        assert "дыра словаря" not in verdict and "префикс" not in verdict, text


def test_added_mute_on_spicy_only_says_it_is_a_deliberate_choice():
    """Грубость на грани («жопа») детектор не мьютит осознанно — это не пропуск."""
    word = _word("жопа")
    roots, known = audit_mod._roots_of([word])

    assert roots == ["жоп"] and known is None
    verdict = audit_mod._verdict_added([word], roots, known)
    assert "SPICY_ROOTS" in verdict and "НЕ мьютит осознанно" in verdict


def test_added_mute_without_any_root_blames_asr():
    word = _word("привет")
    roots, known = audit_mod._roots_of([word])

    assert roots == [] and known is None
    assert "ASR УСЛЫШАЛ НЕ ТО" in audit_mod._verdict_added([word], roots, known)


# --- «мата в тексте нет»: услышали не то или раздвинули наше же окно? ---------


def test_clean_word_next_to_a_machine_mute_is_a_window_not_a_mishearing():
    """Один и тот же чистый текст даёт РАЗНЫЙ диагноз в зависимости от зазора.

    Вплотную к мьюту машины — человек раздвинул или разбил ЕГО окно; далеко —
    мат, которого нет в транскрипте. Лечится это в разных местах пайплайна
    (padding/склейка против качества ASR), поэтому смешивать их нельзя.
    """
    word = _word("это вообще")
    roots, known = audit_mod._roots_of([word])
    assert roots == [] and known is None

    near = audit_mod.NEIGHBOUR_GAP_MS - 100
    assert audit_mod.classify_added([word], roots, known, near) == "neighbour_window"
    assert "padding" in audit_mod._verdict_added([word], roots, known, near)

    far = audit_mod.NEIGHBOUR_GAP_MS + 100
    assert audit_mod.classify_added([word], roots, known, far) == "asr_misheard"
    # Зазора нет (в снимке ни одного мьюта) — гадать не о чем, диагноз прежний.
    assert audit_mod.classify_added([word], roots, known, None) == "asr_misheard"


def test_known_root_outranks_the_neighbour_gap():
    """Соседство не должно прятать настоящую потерю: если корень детектору
    известен, мьют потерялся ПОСЛЕ детекции, даже когда окна пересекаются."""
    word = _word("заебись")
    roots, known = audit_mod._roots_of([word])

    assert audit_mod.classify_added([word], roots, known, 0) == "lost_after_detection"


def test_machine_spans_count_only_what_the_pipeline_actually_muted():
    """Review-мьют звук не глушит (в Vegas не уезжает), поэтому ручной мьют рядом
    с ним — это НЕ раздвинутое окно, а работа за программу."""
    journal = {
        "baseline": {
            "mutes": [
                {"start": 1.0, "end": 1.5, "status": "accepted"},
                {"start": 5.0, "end": 5.4, "status": "review"},
                {"start": 8.0, "end": 8.0, "status": "accepted"},  # пустое окно
                "мусор",
            ]
        }
    }

    assert audit_mod._machine_spans(journal) == [(1000, 1500)]
    assert audit_mod._machine_spans({}) == []


def test_gap_to_machine_is_zero_on_overlap_and_none_without_baseline():
    spans = [(1000, 1500), (5000, 5400)]
    finding = lambda s, e: audit_mod.EditFinding(  # noqa: E731
        kind="added",
        start_ms=s,
        end_ms=e,
        mute={},
        words=[],
        roots=[],
        known_stem=None,
        verdict="",
    )

    assert audit_mod._gap_to_machine(spans, finding(1200, 1300)) == 0  # внутри
    assert audit_mod._gap_to_machine(spans, finding(1400, 1600)) == 0  # пересекается
    assert audit_mod._gap_to_machine(spans, finding(1700, 1900)) == 200  # справа
    assert audit_mod._gap_to_machine(spans, finding(700, 900)) == 100  # слева
    # Ближайший из двух, а не первый попавшийся.
    assert audit_mod._gap_to_machine(spans, finding(4800, 4900)) == 100
    assert audit_mod._gap_to_machine([], finding(1200, 1300)) is None
    # Озвученность мерим по ТОМУ окну, по которому посчитан зазор.
    assert audit_mod._nearest_machine(spans, finding(4800, 4900)) == (5000, 5400)
    assert audit_mod._nearest_machine(spans, finding(1200, 1300)) == (1000, 1500)
    assert audit_mod._nearest_machine([], finding(1200, 1300)) is None


# --- звук в окне: «снято осознанно» против «мьют стоял на тишине» -------------


def test_two_thresholds_disagree_and_the_absolute_one_is_right(tmp_path):
    """Почему диагнозам нужен именно АБСОЛЮТНЫЙ порог.

    Относительный порог (`EnergyFit.fill`, как в mute_refine.trim_window) считает
    порог от пика самого окна. В окне без речи пиком становится фон — порог падает
    под него, и «речь» находится на 100% окна. Абсолютный порог (`voiced_ratio`)
    на том же окне даёт 0%. Диагноз, построенный на первом, обязан врать.
    """
    audio = tmp_path / "hum.wav"
    _write_wav(audio, [(3.0, 3.4)], 10.0, hum=200)
    with audit_mod.WavEnergy(audio) as energy:
        thr = audit_mod.global_threshold(energy)
        fit = audit_mod.fit_window(energy, 900, 1500)
        voiced = audit_mod.voiced_ratio(energy, 900, 1500, thr)
        loud = audit_mod.voiced_ratio(energy, 3000, 3400, thr)

    assert fit is not None and fit.fill == 1.0  # относительный порог: «речь везде»
    assert voiced == 0.0 and audit_mod.on_silence(voiced)  # абсолютный: тишина
    assert loud == 1.0 and not audit_mod.on_silence(loud)


def test_removed_mute_standing_on_silence_is_not_called_deliberate():
    """Мат в тексте есть, но в окне мьюта тишина — снято правильно, и виноват
    пайплайн (тайминги ASR), а не человек. Раньше отчёт писал «снято осознанно,
    детектор не виноват» и терял единственный настоящий дефект прогона."""
    word = _word("нихуя")
    roots, known = audit_mod._roots_of([word])
    mute = {"match_type": "surface", "banword": "нихуя"}

    assert audit_mod.classify_removed([word], roots, 0.0) == "on_silence"
    verdict = audit_mod._verdict_removed([word], roots, mute, 0.0)
    assert "ТИШИНА" in verdict and "mute_reanchor_move" in verdict
    assert "снято осознанно" not in verdict

    # Звук в окне есть — это действительно выбор человека, диагноз прежний.
    assert audit_mod.classify_removed([word], roots, 0.9) == "deliberate"
    assert "снято осознанно" in audit_mod._verdict_removed([word], roots, mute, 0.9)
    # Мерить нечем (нет WAV) — не выдумываем, остаётся прежний диагноз.
    assert audit_mod.classify_removed([word], roots, None) == "deliberate"


def test_false_positive_outranks_silence():
    """Обсценного корня нет вовсе — чинить надо словарь и нормализацию, даже если
    мьют вдобавок стоял на тишине."""
    word = _word("привет")
    roots, _known = audit_mod._roots_of([word])

    assert audit_mod.classify_removed([word], roots, 0.0) == "false_positive"
    assert audit_mod.classify_removed([], [], 0.0) == "empty"


def test_added_mute_next_to_a_silent_machine_window_is_timing_drift():
    """Один и тот же зазор до мьюта машины значит разное в зависимости от того,
    был ли в ТОМ окне звук: раздвинутое окно лечится padding, уехавший тайминг —
    переносом окна (его делает человек по янтарной метке, см. mute_reanchor_move).
    Смешивать их нельзя, это разные места пайплайна."""
    word = _word("это вообще")
    roots, known = audit_mod._roots_of([word])
    near = audit_mod.NEIGHBOUR_GAP_MS - 100
    far = audit_mod.NEIGHBOUR_GAP_MS + 100

    assert audit_mod.classify_added([word], roots, known, near, 0.0) == "asr_timing_off"
    verdict = audit_mod._verdict_added([word], roots, known, near, 0.0)
    assert "ТИШИНЕ" in verdict and "mute_reanchor_move" in verdict

    # Тот же зазор, но окно машины на звуке — прежний диагноз про padding.
    assert audit_mod.classify_added([word], roots, known, near, 0.8) == "neighbour_window"
    # Тишина далеко ничего не объясняет: мьют машины там не при чём.
    assert audit_mod.classify_added([word], roots, known, far, 0.0) == "asr_misheard"
    # Не измерено — прежнее поведение.
    assert audit_mod.classify_added([word], roots, known, near, None) == "neighbour_window"


def test_silent_machine_window_outranks_guesses_from_the_text():
    """Замер сильнее вывода из текста. «Слов тут нет» (речь не распознана) и
    «корень мы знали» (потерялось после детекции) объясняют правку хуже, чем наш
    же мьют рядом, глушащий тишину: слово распознано, уехали его тайминги."""
    word = _word("заебись")
    roots, known = audit_mod._roots_of([word])

    assert audit_mod.classify_added([word], roots, known, 0, 0.0) == "asr_timing_off"
    assert audit_mod.classify_added([], [], None, 0, 0.0) == "asr_timing_off"
    # Без замера — прежние диагнозы, ничего не потеряно.
    assert audit_mod.classify_added([word], roots, known, 0, None) == "lost_after_detection"
    assert audit_mod.classify_added([], [], None, 0, None) == "no_words"


def test_report_names_timing_drift_instead_of_a_deliberate_removal(tmp_path):
    """Сквозной случай vod_2847020280: мьют «бля» стоял в тишине, человек перетащил
    его на речь на 500 мс позже. Отчёт обязан назвать это уехавшим таймингом — по
    тому же замеру звука, который он и так печатает."""
    from twitch_cut.edit_journal import start_journal

    words = [("это", 0.5, 0.8), ("нихуя", 1.0, 1.4), ("привет", 3.0, 3.3)]
    voiced = [(0.5, 0.8), (1.9, 2.3), (3.0, 3.3)]  # «нихуя» звучит на 500 мс позже
    program = [_mute(0.9, 1.5, "нихуя")]
    human = [_mute(1.85, 2.35, "нихуя", id=program[0]["id"])]  # тот же мьют, перетащен
    decisions = _project(tmp_path, words=words, mutes=human, voiced=voiced, hum=200)
    start_journal(
        decisions, {"mutes": program, "silences": [], "offtopic": {"segments": []}}
    )

    result = audit_mod.analyze(decisions)
    by_kind = {f.kind: f for f in result.edits}

    # Половина «снято»: окно программы стояло на тишине — снято правильно.
    dropped = by_kind["removed"]
    assert (dropped.start_ms, dropped.end_ms) == (900, 1500)
    assert dropped.voiced == 0.0 and dropped.on_silence
    assert "ТИШИНА" in dropped.verdict and "mute_reanchor_move" in dropped.verdict
    assert "снято осознанно" not in dropped.verdict

    # Половина «добавлено»: человек поставил мьют на настоящий звук.
    added = by_kind["added"]
    assert (added.start_ms, added.end_ms) == (1850, 2350)
    assert added.voiced is not None and added.voiced > 0.5 and not added.on_silence
    assert added.neighbour_voiced == 0.0 and added.neighbour_gap_ms == 350
    assert "mute_reanchor_move" in added.verdict
    # Прежние (неверные) советы из отчёта ушли.
    assert "padding и склейкой" not in added.verdict
    assert "НЕТ СЛОВ" not in added.verdict

    # В отчёте видны ОБА порога рядом — иначе диагнозу нечем себя подтвердить.
    text = audit_mod.render_report(result)
    assert "(абс. 0%)" in text


def test_moved_mute_off_silence_is_named_timing_drift_not_a_padding_tweak(tmp_path):
    """Тот же дефект, но человек сдвинул границы, не потеряв общий звук: журнал
    отдаёт это как `moved`, и вердикт всё равно обязан назвать причину."""
    from twitch_cut.edit_journal import start_journal

    words = [("нихуя", 1.0, 1.6), ("привет", 3.0, 3.3)]
    voiced = [(1.55, 1.9), (3.0, 3.3)]  # звук в самом конце ASR-границ и дальше
    program = [_mute(1.0, 1.6, "нихуя")]
    human = [_mute(1.5, 2.1, "нихуя", id=program[0]["id"])]
    decisions = _project(tmp_path, words=words, mutes=human, voiced=voiced, hum=200)
    start_journal(
        decisions, {"mutes": program, "silences": [], "offtopic": {"segments": []}}
    )

    (moved,) = [f for f in audit_mod.analyze(decisions).edits if f.kind == "moved"]

    # В окне программы звук только задевает край (последние 50 мс), в окне
    # человека речь занимает большую часть — сдвиг С тишины НА речь.
    assert audit_mod.on_silence(moved.base_voiced) and not moved.on_silence
    assert moved.base_voiced is not None and moved.base_voiced < 0.1
    assert moved.voiced is not None and moved.voiced > 0.5
    assert "окно программы стояло на ТИШИНЕ" in moved.verdict
    assert "mute_reanchor_move" in moved.verdict


