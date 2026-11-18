"""Одна ручка чувствительности вместо россыпи чисел (config.SENSITIVITY_LEVELS).

Эти тесты стерегут не арифметику, а обещание: «level → набор согласованных
чисел, и ровно одно место, где они написаны». Ломается оно молча — уровень
применился наполовину, а прогон выглядит нормально.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from twitch_cut.cli import app
from twitch_cut.config import (
    SENSITIVITY_DEFAULT,
    SENSITIVITY_LEVELS,
    PipelineConfig,
)
from twitch_cut.decisions import build_decisions
from twitch_cut.profanity import ProfanityMatch
from twitch_cut.server.schemas import ProcessJobRequest
from twitch_cut.timecode import parse_timecode

# Поля, которые уровень обязан заполнить. Список задан явно, а не выведен из
# SENSITIVITY_LEVELS: иначе тест «уровень заполняет свои поля» стал бы
# тавтологией и пропустил бы поле, забытое во всех уровнях сразу.
LEVEL_FIELDS = (
    "mute_padding_before_ms",
    "mute_padding_after_ms",
    "mute_join_gap_ms",
    "mute_trim_peak_ratio",
    "mute_trim_guard_ms",
    "mute_expand_on_silence_ms",
    "mute_head_cross_prev_ms",
    "spot_min_score",
)


def cfg(**kwargs) -> PipelineConfig:
    """Конфиг с фиксированным устройством: замер уровней не зависит от железа."""
    kwargs.setdefault("device", "cpu")
    kwargs.setdefault("compute_type", "float32")
    return PipelineConfig(**kwargs)


def test_every_level_fills_every_knob_and_passes_validation():
    for name in SENSITIVITY_LEVELS:
        config = cfg(sensitivity=name)
        config.validate()
        for field in LEVEL_FIELDS:
            assert getattr(config, field) is not None, f"{name}.{field} остался None"


def test_levels_declare_exactly_the_optional_fields():
    """Уровень не может завести «своё» поле или забыть чужое.

    Ключ уровня, которому нет None-поля в PipelineConfig, применяется в никуда;
    None-поле, которого нет в уровне, остаётся None и роняет прогон на первом
    же сравнении. И то и другое — молча.
    """
    optional = {
        f.name for f in dataclasses.fields(PipelineConfig) if f.default is None
    }
    assert optional == set(LEVEL_FIELDS)
    for name, level in SENSITIVITY_LEVELS.items():
        assert set(level) == set(LEVEL_FIELDS), f"уровень {name} расходится с полями"


def test_balanced_keeps_the_numbers_measured_before_the_knob():
    """Страховка от дрейфа: `balanced` — не «середина», а старые дефолты.

    Числа выписаны литералами намеренно. Сверять их с SENSITIVITY_LEVELS
    бессмысленно (тест сравнил бы значение с самим собой) — они здесь ровно для
    того, чтобы правка уровня, меняющая поведение по умолчанию, была видна как
    падение теста, а не как «стало чуть иначе» на следующем прогоне.
    """
    config = cfg(sensitivity="balanced")
    assert config.mute_padding_before_ms == 80
    assert config.mute_padding_after_ms == 120
    assert config.mute_join_gap_ms == 600
    assert config.mute_trim_peak_ratio == 0.12
    assert config.mute_trim_guard_ms == 40
    assert config.spot_min_score == 0.45
    # Это число появилось ПОЗЖЕ ручки (расширение окна, стоящего на тишине), но
    # стоит здесь по той же причине: молчаливая смена дефолтного поведения должна
    # быть видна как падение теста.
    assert config.mute_expand_on_silence_ms == 200
    # И это тоже позже (допуск голове зайти на разметку соседа): 60 из 80 мс
    # padding — измеренная точка, а не «немножко».
    assert config.mute_head_cross_prev_ms == 60
    assert SENSITIVITY_DEFAULT == "balanced"
    # Дефолтный конфиг = balanced: молчаливой смены поведения быть не должно.
    assert cfg() == config


def test_levels_move_along_one_axis():
    """low → high — один шаг «шире мьют», а не набор независимых настроек.

    Если хоть одно число пойдёт против оси, ручка перестанет быть ручкой:
    пользователь выберет «не пропускать мат» и получит более злой поджим границ.
    """
    low, mid, high = (SENSITIVITY_LEVELS[n] for n in ("low", "balanced", "high"))
    for field in (
        "mute_padding_before_ms",
        "mute_padding_after_ms",
        "mute_join_gap_ms",
        # Расширение окна, стоящего на тишине: 0 → 200 → 300. Шаг по этой оси
        # покупает накрытие мата съеденными чистыми словами, и цена за окно с
        # каждым шагом растёт — потому уровни и режут кривую в разных точках.
        "mute_expand_on_silence_ms",
    ):
        assert low[field] < mid[field] < high[field], field
    # Эти два — наоборот: чем выше чувствительность, тем МЕНЬШЕ значение
    # (поджим по RMS осторожнее, порог спотинга ниже).
    for field in ("mute_trim_peak_ratio", "spot_min_score"):
        assert low[field] > mid[field] > high[field], field
    # Защитная зона вокруг найденной речи растёт вместе с padding.
    assert low["mute_trim_guard_ms"] < mid["mute_trim_guard_ms"] < high["mute_trim_guard_ms"]
    # Допуск голове зайти на соседа — НЕ строгая ось: 0 → 60 → 60. С появлением
    # стража по звуку (mute_head_cross_voiced_guard) заход глубже на соседа
    # перестал быть «детекцией мата» и стал ИЗМЕРЯЕМОЙ ЦЕНОЙ (замьюченная чистая
    # звонкая речь), а страж и так схлопывает потолок в тишину при любом значении.
    # Двусторонний замер (tools/head_sweep.py, 8 записей) показал {60,100,140}
    # неразличимыми под стражем, а вне стража (raw_mute) — 100 переглушает вдвое.
    # Поэтому `high` = `balanced` = 60, а агрессия `high` живёт в полях выше.
    # Строго растёт только нижняя граница: low не должен пускать голову на соседа.
    assert low["mute_head_cross_prev_ms"] < mid["mute_head_cross_prev_ms"], "low"
    assert mid["mute_head_cross_prev_ms"] <= high["mute_head_cross_prev_ms"], "mid≤high"


def test_explicit_value_beats_the_level_without_touching_the_rest():
    config = cfg(sensitivity="high", mute_padding_after_ms=15)
    assert config.mute_padding_after_ms == 15
    assert config.mute_padding_before_ms == SENSITIVITY_LEVELS["high"]["mute_padding_before_ms"]
    assert config.spot_min_score == SENSITIVITY_LEVELS["high"]["spot_min_score"]


def test_unknown_level_is_rejected_by_name():
    """И именно по имени: без этой проверки первым падал бы None < 0."""
    with pytest.raises(ValueError) as exc:
        cfg(sensitivity="paranoid").validate()
    message = str(exc.value)
    assert "sensitivity" in message
    for name in SENSITIVITY_LEVELS:
        assert name in message


def test_resolved_numbers_survive_normalized():
    """normalized() пересобирает конфиг через replace — числа не должны потеряться."""
    config = cfg(sensitivity="high", transcriber="whisperx", device="mps")
    fixed = config.normalized()
    assert fixed.device == "cpu"
    assert fixed.sensitivity == "high"
    for field in LEVEL_FIELDS:
        assert getattr(fixed, field) == SENSITIVITY_LEVELS["high"][field]


def test_decisions_settings_carry_the_level_and_the_resolved_numbers():
    """decisions.json должен объяснять прогон целиком.

    Одного имени уровня мало: уровни со временем поменяются, а старый прогон
    обязан остаться читаемым. Одних чисел тоже мало — по ним не видно, что
    человек выбрал.
    """
    match = ProfanityMatch(
        word="блин",
        normalized="блин",
        lemma="блин",
        banword="блин",
        banword_lemma="блин",
        local_start_ms=4300,
        local_end_ms=4700,
        stream_start_ms=64_300,
        stream_end_ms=64_700,
        confidence=0.96,
        segment_id="seg_1",
        match_type="surface",
        timing_source="word",
    )
    doc = build_decisions(
        stream_path=Path("stream.mp4"),
        original_path=Path("original_video.mp4"),
        range_in=parse_timecode("00:01:00"),
        range_out=parse_timecode("00:02:00"),
        matches=[match],
        config=cfg(sensitivity="high"),
    )
    settings = doc["settings"]
    assert settings["sensitivity"] == "high"
    for field in LEVEL_FIELDS:
        assert settings[field] == SENSITIVITY_LEVELS["high"][field]


def test_high_level_widens_the_window_low_narrows_it():
    """Уровень обязан быть виден в результате, а не только в settings."""
    match = ProfanityMatch(
        word="хуйня",
        normalized="хуйня",
        lemma="хуйня",
        banword="хуйня",
        banword_lemma="хуйня",
        local_start_ms=4300,
        local_end_ms=4700,
        stream_start_ms=4300,
        stream_end_ms=4700,
        confidence=0.96,
        segment_id="seg_1",
        match_type="surface",
        timing_source="word",
    )
    widths = []
    for name in ("low", "balanced", "high"):
        doc = build_decisions(
            stream_path=Path("stream.mp4"),
            original_path=Path("original_video.mp4"),
            range_in=parse_timecode("00:00:00"),
            range_out=parse_timecode("00:01:00"),
            matches=[match],
            config=cfg(sensitivity=name),
        )
        mute = doc["mutes"][0]
        widths.append(round(mute["end"] - mute["start"], 3))
    assert widths[0] < widths[1] < widths[2], widths


# --- дорога от края наружу к прогону ----------------------------------------
#
# Ручка бесполезна, если её значение не доходит до пайплайна. Раньше так и было
# со спотингом: поля жили в PipelineConfig, а из API не пробрасывались вообще —
# и заметить это по тестам конфига было нельзя.


def _mock_run_args(tmp_path: Path, name: str) -> tuple[list[str], Path]:
    stream = tmp_path / "stream.mp4"
    stream.write_bytes(b"\x00")  # mock-режим не зовёт ffmpeg, но файл нужен
    banwords = tmp_path / "banwords.txt"
    banwords.write_text("сука\n", encoding="utf-8")
    mock = tmp_path / "mock.json"
    mock.write_text(
        json.dumps(
            {
                "segments": [
                    {
                        "id": "seg_1",
                        "start": 4.0,
                        "end": 6.0,
                        "text": "ну сука вот",
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
    decisions_out = tmp_path / f"decisions_{name}.json"
    return (
        [
            "process",
            "--stream", str(stream),
            "--original", str(stream),
            "--banwords", str(banwords),
            "--workdir", str(tmp_path / "work"),
            "--decisions", str(decisions_out),
            "--vegas", str(tmp_path / f"mutes_{name}.cs"),
            "--mock-transcript", str(mock),
            "--range-out", "00:00:30",
            "--device", "cpu",
            "--sensitivity", name,
        ],
        decisions_out,
    )


@pytest.mark.parametrize("name", ["low", "high"])
def test_cli_sensitivity_reaches_the_written_decisions(tmp_path: Path, name: str):
    args, decisions_out = _mock_run_args(tmp_path, name)
    result = CliRunner().invoke(app, args)
    assert result.exit_code == 0, result.output
    settings = json.loads(decisions_out.read_text(encoding="utf-8"))["settings"]
    assert settings["sensitivity"] == name
    for field in LEVEL_FIELDS:
        assert settings[field] == SENSITIVITY_LEVELS[name][field]


def test_cli_rejects_unknown_sensitivity(tmp_path: Path):
    args, _ = _mock_run_args(tmp_path, "paranoid")
    result = CliRunner().invoke(app, args)
    assert result.exit_code != 0
    assert "sensitivity" in result.output


def test_api_request_leaves_the_numbers_to_the_level():
    """Схема API не должна повторять числа: её дефолт — None, а не 80/120/600."""
    request = ProcessJobRequest(stream="a.mp4")
    assert request.sensitivity == SENSITIVITY_DEFAULT
    for field in LEVEL_FIELDS:
        assert getattr(request, field) is None, field


def test_api_request_rejects_unknown_level():
    """422 на границе, а не упавшая джоба в списке обработок."""
    with pytest.raises(ValueError, match="sensitivity"):
        ProcessJobRequest(stream="a.mp4", sensitivity="paranoid")


def test_api_job_hands_the_level_and_spotting_to_the_pipeline(tmp_path: Path, monkeypatch):
    """Джоба API должна донести уровень и параметры спотинга до пайплайна.

    Регрессия на «поле есть, но его никто не передаёт»: `spot_posteriors` и
    `spot_min_score` жили в PipelineConfig, а в схеме запроса отсутствовали —
    из UI выключить спотинг или подвинуть порог было нельзя, и тесты конфига
    этого не видели.
    """
    from twitch_cut.server import runner as runner_module

    seen: dict[str, PipelineConfig] = {}

    def fake_pipeline(**kwargs):
        seen["config"] = kwargs["config"]
        return {"decisions_path": str(tmp_path / "d.json")}

    monkeypatch.setattr(
        "twitch_cut.pipeline.run_transcription_pipeline", fake_pipeline
    )
    stream = tmp_path / "stream.mp4"
    stream.write_bytes(b"\x00")
    request = ProcessJobRequest(
        stream=str(stream),
        workdir=str(tmp_path / "work"),
        decisions=str(tmp_path / "d.json"),
        vegas=str(tmp_path / "v.cs"),
        banwords=None,
        device="cpu",
        compute_type="float32",
        sensitivity="high",
        spot_min_score=0.9,
        spot_posteriors=False,
    )

    class _Silent:
        def emit(self, *_args, **_kwargs) -> None:
            return None

    runner_module._run_process_sync(request, _Silent())  # type: ignore[arg-type]

    config = seen["config"]
    assert config.sensitivity == "high"
    assert config.spot_posteriors is False
    # Явное значение из запроса живёт, остальные пять берутся из уровня.
    assert config.spot_min_score == 0.9
    assert config.mute_padding_after_ms == SENSITIVITY_LEVELS["high"]["mute_padding_after_ms"]
    assert config.mute_trim_guard_ms == SENSITIVITY_LEVELS["high"]["mute_trim_guard_ms"]


def test_every_knob_from_the_request_reaches_the_pipeline(tmp_path: Path, monkeypatch):
    """Каждое поле уровня, заданное в запросе, обязано дойти до PipelineConfig.

    Тест на КЛАСС ошибки, а не на конкретную ручку: поле легко завести в схеме и
    забыть в `runner._run_process_sync`, где конфиг собирается перечислением
    вручную. Тогда UI показывает настройку, запрос её несёт, валидация проходит —
    а прогон молча берёт значение из уровня. Ровно так и уехал
    `mute_head_cross_prev_ms`. Значения ниже выбраны так, чтобы не совпадать ни с
    одним уровнем: иначе тест прошёл бы и на непереданном поле.
    """
    from twitch_cut.server import runner as runner_module

    explicit = {
        "mute_padding_before_ms": 33,
        "mute_padding_after_ms": 44,
        "mute_join_gap_ms": 55,
        "mute_trim_peak_ratio": 0.33,
        "mute_trim_guard_ms": 66,
        "mute_expand_on_silence_ms": 77,
        "mute_head_cross_prev_ms": 88,
        "spot_min_score": 0.66,
    }
    assert set(explicit) == set(LEVEL_FIELDS), "новое поле уровня — добавь его сюда"
    for name, level in SENSITIVITY_LEVELS.items():
        for field, value in explicit.items():
            assert level[field] != value, f"{name}.{field} совпал с пробным значением"

    seen: dict[str, PipelineConfig] = {}

    def fake_pipeline(**kwargs):
        seen["config"] = kwargs["config"]
        return {"decisions_path": str(tmp_path / "d.json")}

    monkeypatch.setattr("twitch_cut.pipeline.run_transcription_pipeline", fake_pipeline)
    stream = tmp_path / "stream.mp4"
    stream.write_bytes(b"\x00")
    request = ProcessJobRequest(
        stream=str(stream),
        workdir=str(tmp_path / "work"),
        decisions=str(tmp_path / "d.json"),
        vegas=str(tmp_path / "v.cs"),
        banwords=None,
        device="cpu",
        compute_type="float32",
        **explicit,
    )

    class _Silent:
        def emit(self, *_args, **_kwargs) -> None:
            return None

    runner_module._run_process_sync(request, _Silent())  # type: ignore[arg-type]

    config = seen["config"]
    for field, value in explicit.items():
        assert getattr(config, field) == value, f"{field} не доехал до пайплайна"
