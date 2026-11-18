"""Тесты `twitch-cut eval` — метрика пайплайна по ручным правкам.

Метрика должна быть проверяемой на бумаге, поэтому здесь проект с ТРЕМЯ
машинными мьютами и заранее известным ответом: один снят человеком (ложное),
один сдвинут (границы), один принят как есть, плюс один добавлен руками.
Отсюда точность 2/3, полнота 2/3 и ошибка конца −900 мс.
"""

from __future__ import annotations

from array import array
import json
import math
from pathlib import Path
import wave

import pytest

from twitch_cut import evaluate as eval_mod
from twitch_cut.audit import AuditError

RATE = 16000

WORDS = [
    ("это", 0.5, 0.8),
    ("нихуя", 1.0, 1.4),
    ("привет", 3.0, 3.3),
    ("блять", 5.0, 5.4),
    # Чистое слово вплотную к машинному мьюту: на него человек «съезжает», когда
    # правит окно, — см. test_relocation_next_to_a_machine_mute_is_not_an_asr_miss.
    ("ладно", 8.15, 8.4),
    ("хуйня", 8.6, 8.9),
]
VOICED = [(0.5, 0.8), (1.0, 1.4), (3.0, 3.3), (5.0, 5.4), (8.15, 8.4), (8.6, 8.9)]

# Что решила программа: ложное срабатывание на «привет», растянутый хвост на
# «блять» и точный мьют на «хуйня».
MACHINE = [(2.9, 3.4, "привет"), (5.0, 6.4, "блять"), (8.6, 8.9, "хуйня")]


def _write_wav(path: Path, spans: list[tuple[float, float]], duration: float) -> None:
    total = int(duration * RATE)
    samples = array("h", [0]) * total
    for start, end in spans:
        for i in range(max(0, int(start * RATE)), min(total, int(end * RATE))):
            samples[i] = int(12000 * math.sin(2 * math.pi * 220 * i / RATE))
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(RATE)
        wav.writeframes(samples.tobytes())


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
    mutes: list[dict],
    *,
    words: list[tuple[str, float, float]] | None = None,
    voiced: list[tuple[float, float]] | None = None,
) -> Path:
    words = words if words is not None else WORDS
    voiced = voiced if voiced is not None else VOICED
    transcript = tmp_path / "transcript.json"
    transcript.write_text(
        json.dumps(
            {
                "segments": [
                    {
                        "id": "seg_000000",
                        "start": words[0][1],
                        "end": words[-1][2],
                        "words": [
                            {"word": t, "start": s, "end": e} for t, s, e in words
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    audio = tmp_path / "audio.wav"
    _write_wav(audio, voiced, 10.0)
    decisions = tmp_path / "decisions.json"
    decisions.write_text(
        json.dumps(
            {
                "schema_version": "1.1",
                "range_in": "00:00:00.000",
                "range_out": "00:00:10.000",
                "local_duration": "00:00:10.000",
                "settings": {"mute_trim_to_audio": True},
                "summary": {"raw_matches": len(mutes)},
                "mutes": mutes,
                "cuts": [],
                "caches": {"transcript": str(transcript), "audio": str(audio)},
            }
        ),
        encoding="utf-8",
    )
    return decisions


def _edited(
    tmp_path: Path,
    *,
    human: list[dict] | None = None,
    machine: list[tuple[float, float, str]] | None = None,
    words: list[tuple[str, float, float]] | None = None,
    voiced: list[tuple[float, float]] | None = None,
) -> Path:
    """Проект после правок + журнал со снимком ПРОГРАММЫ (source=pipeline).

    По умолчанию человек: снял ложный мьют на «привет», поджал хвост «блять»
    (6.4 → 5.5), оставил «хуйня» как есть и добавил руками «нихуя».
    """
    from twitch_cut.edit_journal import start_journal

    edited = human if human is not None else [
        _mute(5.0, 5.5, "блять"),  # id тот же — это сдвиг, а не пара удалил/добавил
        _mute(8.6, 8.9, "хуйня"),  # принято как есть
        _mute(1.0, 1.4, "нихуя", id="mute_manual_1000", source="manual"),
    ]
    decisions = _project(tmp_path, edited, words=words, voiced=voiced)
    start_journal(
        decisions,
        {
            "mutes": [_mute(s, e, w) for s, e, w in (machine if machine is not None else MACHINE)],
            "silences": [],
            "offtopic": {"segments": []},
        },
    )
    return decisions


def test_metrics_are_countable_on_paper(tmp_path):
    result = eval_mod.evaluate(_edited(tmp_path))

    assert (result.machine, result.human) == (3, 3)
    assert (result.removed, result.added, result.moved, result.untouched) == (1, 1, 1, 1)
    assert result.kept == 2
    # 2 из 3 мьютов человек оставил → точность 2/3; пропустили один → полнота 2/3.
    assert result.precision == pytest.approx(2 / 3, abs=1e-4)
    assert result.recall == pytest.approx(2 / 3, abs=1e-4)
    assert result.f1 == pytest.approx(2 / 3, abs=1e-4)
    assert result.trustworthy is True and result.baseline_source == "pipeline"


def test_boundary_error_keeps_the_sign(tmp_path):
    """Знак важнее величины: он отвечает, машина ОПАЗДЫВАЕТ или РЕЖЕТ лишнее."""
    result = eval_mod.evaluate(_edited(tmp_path))

    assert result.d_end.n == 1
    assert result.d_end.signed_median == -900  # хвост укоротили на 900 мс
    assert result.d_end.median == 900 and result.d_end.worst == 900
    assert result.d_start.signed_median == 0
    assert result.d_duration.signed_median == -900


def test_muted_clean_speech_is_not_cancelled_by_the_net_width(tmp_path):
    """Переглушку по ГОЛОВЕ окна нетто-длина прячет — из-за этого её не видели.

    Человек сдвинул всё окно «блять» на +120 мс вперёд: начало +120 (срезал
    голову — 120 мс чистой речи соседа), конец +120 (отпустил хвост). По длине
    правка нулевая (d_duration = 0), и ровно так переглушка терялась: ruler
    «слышно» считает только НЕДОкрытие, а нетто-длина гасит «голову подрезал,
    хвост отпустил» в ноль. muted_clean_head обязан показать эти 120 мс, которых
    больше нигде не видно; хвост при этом расширили — это НЕ переглушка (0).
    """
    decisions = _edited(
        tmp_path,
        human=[
            _mute(2.9, 3.4, "привет"),
            _mute(5.12, 5.52, "блять", id="mute_005000"),  # окно целиком сдвинуто
            _mute(8.6, 8.9, "хуйня"),
        ],
        machine=[(2.9, 3.4, "привет"), (5.0, 5.4, "блять"), (8.6, 8.9, "хуйня")],
    )
    result = eval_mod.evaluate(decisions)

    assert result.moved == 1
    assert result.d_duration.signed_median == 0  # нетто-длина переглушку НЕ видит
    assert result.muted_clean_head.median == 120  # а вот и она — 120 мс головы
    assert result.muted_clean_head.total == 120
    assert result.muted_clean_tail.total == 0  # хвост расширили — не переглушка
    text = eval_mod.render_metrics(result)
    assert "заглушено ЧИСТОЙ речи" in text



    """Числа и текст в audit.md должны объяснять одно и то же, поэтому ключи
    диагнозов берутся из audit.classify_*, а не считаются здесь заново."""
    result = eval_mod.evaluate(_edited(tmp_path))

    # «нихуя» — корень детектору известен, значит потеря ПОСЛЕ детекции.
    assert result.added_by_diagnosis == {"lost_after_detection": 1}
    # «привет» — чистое слово, мьют был ложным.
    assert result.removed_by_diagnosis == {"false_positive": 1}


def test_rounding_jitter_is_not_a_boundary_error(tmp_path):
    """Сдвиг в 10 мс — округление round3 и дрожь мыши. Если считать его правкой,
    метрика начнёт ухудшаться от самого факта пересохранения файла."""
    decisions = _edited(
        tmp_path,
        human=[
            _mute(2.9, 3.4, "привет"),
            _mute(5.0, 6.41, "блять"),  # +10 мс
            _mute(8.6, 8.9, "хуйня"),
        ],
    )
    result = eval_mod.evaluate(decisions)

    assert result.moved == 0
    assert result.untouched == 3
    assert result.precision == 1.0 and result.recall == 1.0


def test_relocated_mute_is_not_a_boundary_error(tmp_path):
    """Человек перетащил мьют с «привет» на «нихуя» — общего звука у окон нет.

    Журнал сопоставляет по `id`, поэтому такая правка приходит как «сдвиг», и
    без разбора она вносит в ошибку границ 1.9 с. Правильный разбор: машина
    заглушила не то (ложное) + человек нашёл пропуск.
    """
    decisions = _edited(
        tmp_path,
        human=[
            # id машинного мьюта на «привет», а окно уже на «нихуя».
            _mute(1.0, 1.4, "нихуя", id="mute_002900"),
            _mute(5.0, 6.4, "блять"),
            _mute(8.6, 8.9, "хуйня"),
        ],
    )
    result = eval_mod.evaluate(decisions)

    assert result.moved == 0
    assert (result.removed, result.added) == (1, 1)
    assert result.d_start.n == 0 and result.d_end.n == 0
    # Ложное на «привет» и пропуск «нихуя» — два разных диагноза.
    assert result.removed_by_diagnosis == {"false_positive": 1}
    assert result.added_by_diagnosis == {"lost_after_detection": 1}


def test_relocation_next_to_a_machine_mute_is_not_an_asr_miss(tmp_path):
    """Перенос мьюта на соседнее слово — правка ОКНА, а не пропуск распознавания.

    Регрессия: диагноз зазора сначала считался только у обычных «добавленных»
    мьютов, а половина «добавлено» у переноса шла без него. На реальном прогоне
    так и терялись 11 из 12 случаев: метрика печатала «ASR услышал не то» и звала
    чинить распознавание там, где человек перетащил наше же окно на 200 мс.
    """
    decisions = _edited(
        tmp_path,
        human=[
            _mute(2.9, 3.4, "привет"),
            _mute(5.0, 6.4, "блять"),
            # id машинного мьюта на «хуйня», а окно уехало на чистое «ладно» —
            # общего звука нет, поэтому журнал отдаёт это как перенос.
            _mute(8.1, 8.4, "ладно", id="mute_008600"),
        ],
    )
    result = eval_mod.evaluate(decisions)

    assert (result.removed, result.added, result.moved) == (1, 1, 0)
    assert result.added_by_diagnosis == {"neighbour_window": 1}


def test_silence_under_a_mute_reaches_the_metric_too(tmp_path):
    """Замер звука обязан доезжать до метрики, а не оставаться в тексте отчёта.

    Ключи диагнозов у eval и audit.md общие ровно для того, чтобы число и вердикт
    объясняли одно и то же. Если audit.md пишет «мьют стоял на ТИШИНЕ, снято
    правильно», а метрика считает это «снято осознанно» — расхождение вернётся.
    """
    from twitch_cut.edit_journal import start_journal

    words = [("нихуя", 1.0, 1.4), ("привет", 3.0, 3.3)]
    voiced = [(1.9, 2.3), (3.0, 3.3)]  # «нихуя» звучит на 500 мс позже границ ASR
    machine = [_mute(0.9, 1.5, "нихуя")]
    human = [_mute(1.85, 2.35, "нихуя", id=machine[0]["id"])]  # перетащен на звук
    decisions = _project(tmp_path, human, words=words, voiced=voiced)
    start_journal(
        decisions, {"mutes": machine, "silences": [], "offtopic": {"segments": []}}
    )

    result = eval_mod.evaluate(decisions)

    assert result.removed_by_diagnosis == {"on_silence": 1}
    assert result.added_by_diagnosis == {"asr_timing_off": 1}
    text = eval_mod.render_metrics(result)
    assert "мьют стоял на ТИШИНЕ" in text and "уехал тайминг ASR" in text


def test_a_far_away_added_mute_still_blames_asr(tmp_path):
    """Обратная сторона порога: далеко от мьютов машины чистый текст остаётся
    «ASR услышал не то» — иначе новый диагноз проглотил бы настоящие пропуски."""
    decisions = _edited(
        tmp_path,
        human=[
            _mute(2.9, 3.4, "привет"),
            _mute(5.0, 6.4, "блять"),
            _mute(8.6, 8.9, "хуйня"),
            # 0.5–0.8 «это»: до ближайшего мьюта машины 2.1 с — не наше окно.
            _mute(0.5, 0.8, "это", id="mute_manual_500", source="manual"),
        ],
    )
    result = eval_mod.evaluate(decisions)

    assert result.added_by_diagnosis == {"asr_misheard": 1}


def test_a_moved_window_is_not_charged_to_the_search_recall(tmp_path):
    """Двойная плата за одну правку — то, из-за чего полнота врала.

    Человек перетащил НАШЕ окно на 200 мс: журнал отдаёт это как «снял + добавил»,
    и строгая полнота считает пропуск, а точность — ложное. Слово при этом
    найдено, и цена ошибки уже взята медианами d_start/d_end. Замер 2026-08-17 на
    трёх правленых записях: 14 из 19 «пропусков» отстояли от окна машины на
    19–251 мс (медиана 86 мс) — то есть были вот этим, а не ненайденным словом.
    """
    decisions = _edited(
        tmp_path,
        human=[
            _mute(2.9, 3.4, "привет"),
            _mute(5.0, 6.4, "блять"),
            _mute(8.1, 8.4, "ладно", id="mute_008600"),
        ],
    )
    result = eval_mod.evaluate(decisions)

    assert (result.missed, result.misplaced, result.policy) == (0, 1, 0)
    assert result.recall == pytest.approx(2 / 3, abs=1e-4)  # строгая платит
    assert result.recall_search == pytest.approx(1.0, abs=1e-4)  # поиск не платит
    assert result.missed_by_diagnosis == {}
    text = eval_mod.render_metrics(result)
    assert "полнота ПОИСКА" in text and "окно встало не там" in text


def test_a_word_the_model_never_heard_still_costs_the_search_recall(tmp_path):
    """Ограничение модели остаётся пропуском — иначе полноту нечем мерить.

    Тихий, скороговоркой или перекрытый чужой речью мат человек слышит, а машина
    нет. Соблазн вынести это из полноты («не вина человека») ломает саму метрику:
    полнота начнёт расти от того, что модель стала слышать хуже. Поэтому такие
    случаи получают ОТДЕЛЬНЫЙ СЧЁТ (missed_by_diagnosis), но из полноты не
    уходят — ни из строгой, ни из поисковой.
    """
    decisions = _edited(
        tmp_path,
        human=[
            _mute(2.9, 3.4, "привет"),
            _mute(5.0, 6.4, "блять"),
            _mute(8.6, 8.9, "хуйня"),
            _mute(0.5, 0.8, "это", id="mute_manual_500", source="manual"),
        ],
    )
    result = eval_mod.evaluate(decisions)

    assert (result.missed, result.misplaced, result.policy) == (1, 0, 0)
    assert result.recall == pytest.approx(3 / 4, abs=1e-4)
    assert result.recall_search == pytest.approx(3 / 4, abs=1e-4)
    assert result.missed_by_diagnosis == {"asr_misheard": 1}


def test_a_vad_hole_is_a_miss_with_its_own_name(tmp_path):
    """Звук есть, слов нет: модель не «не услышала», а НЕ СЛУШАЛА.

    Это тоже пропуск (заглушить было надо, и не заглушили), но лечится он в
    сегментации VAD, а не порогом поиска — поэтому у него свой ключ, и он виден
    числом рядом с полнотой.
    """
    decisions = _edited(
        tmp_path,
        human=[
            _mute(2.9, 3.4, "привет"),
            _mute(5.0, 6.4, "блять"),
            _mute(8.6, 8.9, "хуйня"),
            # 9.2–9.6: в WAV звук есть, в транскрипте слов нет вовсе.
            _mute(9.2, 9.6, "", id="mute_manual_9200", source="manual"),
        ],
        voiced=[*VOICED, (9.2, 9.6)],
    )
    result = eval_mod.evaluate(decisions)

    assert result.missed_by_diagnosis == {"no_words": 1}
    assert result.recall_search == pytest.approx(3 / 4, abs=1e-4)


def test_spicy_policy_is_not_a_miss_at_all(tmp_path):
    """Грубость из SPICY_ROOTS детектор не мьютит ОСОЗНАННО.

    Человек её глушит по своему вкусу, и записывать это в пропуск поиска — значит
    мерить полноту против чужой политики: порог поиска такой «пропуск» не лечит
    никогда. Из полноты поиска такой мьют уходит целиком, а не в пропуски.
    """
    words = [("сука", 0.5, 0.9), ("блять", 5.0, 5.4)]
    voiced = [(0.5, 0.9), (5.0, 5.4)]
    decisions = _edited(
        tmp_path,
        human=[
            _mute(5.0, 5.4, "блять"),
            _mute(0.5, 0.9, "сука", id="mute_manual_500", source="manual"),
        ],
        machine=[(5.0, 5.4, "блять")],
        words=words,
        voiced=voiced,
    )
    result = eval_mod.evaluate(decisions)

    assert result.added_by_diagnosis == {"spicy_only": 1}
    assert (result.missed, result.misplaced, result.policy) == (0, 0, 1)
    assert result.recall == pytest.approx(1 / 2, abs=1e-4)  # строгая платит
    assert result.recall_search == pytest.approx(1.0, abs=1e-4)  # поиск не при чём


def test_charge_table_covers_every_diagnosis(tmp_path):
    """Новый диагноз обязан получить цену. Без этого он молча падал бы в
    «пропуск» и менял полноту поиска, пока никто не смотрит."""
    from twitch_cut.audit import ADDED_CHARGE, ADDED_DIAGNOSES, CHARGE_LABELS

    assert set(ADDED_CHARGE) == set(ADDED_DIAGNOSES)
    assert set(ADDED_CHARGE.values()) <= set(CHARGE_LABELS)


def test_eval_without_journal_says_what_to_turn_on(tmp_path):
    with pytest.raises(AuditError) as exc:
        eval_mod.evaluate(_project(tmp_path, [_mute(5.0, 5.4, "блять")]))

    assert "Журнал" in str(exc.value) or "журнал" in str(exc.value)


def test_baseline_not_from_the_pipeline_is_flagged_not_hidden(tmp_path):
    """Журнал, заведённый с первого сохранения, уже содержит часть правок
    человека. Метрика тогда завышена, и это должно быть написано, а не скрыто."""
    from twitch_cut.edit_journal import record_save

    machine_doc = {"mutes": [_mute(s, e, w) for s, e, w in MACHINE]}
    decisions = _project(tmp_path, [_mute(8.6, 8.9, "хуйня")])
    record_save(decisions, json.loads(decisions.read_text(encoding="utf-8")), machine_doc)

    result = eval_mod.evaluate(decisions)

    assert result.baseline_source == "first_save"
    assert result.trustworthy is False
    assert any("ЗАВЫШЕНЫ" in note for note in result.notes)


def test_json_survives_a_roundtrip_and_shows_the_delta(tmp_path):
    """Замер нужен ровно для сравнения «до/после», поэтому JSON обязан читаться
    обратно, а рендер — считать дельту к прошлому прогону."""
    result = eval_mod.evaluate(_edited(tmp_path))
    path = eval_mod.write_metrics(result, tmp_path / "eval.json")
    prev = eval_mod.read_metrics(path)

    assert prev is not None and prev["schema_version"] == eval_mod.SCHEMA_VERSION
    assert "без изменений" in eval_mod.render_metrics(result, prev)

    worse = dict(prev, recall=0.5)
    text = eval_mod.render_metrics(result, worse)
    assert "+16.7 п.п." in text  # 2/3 против 1/2
    assert eval_mod.read_metrics(tmp_path / "нет-такого-файла.json") is None


def test_cli_eval_prints_the_summary(tmp_path):
    from typer.testing import CliRunner

    from twitch_cut.cli import app

    decisions = _edited(tmp_path)
    out_json = tmp_path / "metrics.json"
    res = CliRunner().invoke(app, ["eval", str(decisions.parent), "--json", str(out_json)])

    assert res.exit_code == 0, res.output
    assert "точность" in res.output and "полнота" in res.output
    assert out_json.exists()
