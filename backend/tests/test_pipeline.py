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
from twitch_cut.timecode import parse_timecode
from twitch_cut import profanity as profanity_mod

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


def test_detection_cache_key_tracks_in_code_word_lists(tmp_path: Path, monkeypatch):
    """Правка списка решений обязана делать detection_*.json невалидным.

    Списки корней и белые списки живут в коде, а кэш детекции — на диске. Список,
    забытый в `profanity.decision_lists()`, не меняет ключ, и прогон по готовому
    воркдиру тихо остаётся со старыми матчами — от «правка не сработала» это
    неотличимо. Так и было с `_SAFE_SHORT_TOKENS`: «ха» в whitelist не убирало
    ложный мьют на перепрогоне.
    """

    def cache_name(where: Path) -> str:
        where.mkdir(parents=True, exist_ok=True)
        _run(where, "gigaam")
        (cache,) = (where / "work" / "cache").glob("detection_*.json")
        return cache.name

    before = cache_name(tmp_path / "a")
    monkeypatch.setattr(
        profanity_mod,
        "_SAFE_SHORT_TOKENS",
        profanity_mod._SAFE_SHORT_TOKENS | {"зю"},
    )
    assert cache_name(tmp_path / "b") != before


def test_decision_lists_are_stable_between_calls():
    """Ключ кэша считается по этим спискам, поэтому порядок обязан быть
    детерминированным: перебор `frozenset`/`set` зависит от хеша процесса, и без
    сортировки ключ менялся бы между запусками — кэш не попадал бы никогда."""
    assert profanity_mod.decision_lists() == profanity_mod.decision_lists()
    lists = profanity_mod.decision_lists()
    assert "ха" in lists["safe_short"]
    assert lists["safe_short"] == sorted(lists["safe_short"])


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


# --- ETA: дробный прогресс транскрибации -------------------------------------


def test_gigaam_fractional_progress_maps_onto_stage_range(tmp_path: Path, monkeypatch):
    """Тики GigaAM (0..1) растягиваются на диапазон стадии transcribe.

    Ради ETA в UI: без этих промежуточных точек оценщик видит один тик на
    12% и молчит весь час распознавания. Проверяем именно арифметику
    отображения, а не сам движок — transcribe_with_gigaam подменяем заглушкой,
    чтобы не тянуть torch/gigaam.
    """
    import twitch_cut.gigaam_asr as gigaam_asr
    from twitch_cut.pipeline import (
        TRANSCRIBE_END_PCT,
        TRANSCRIBE_ENTER_PCT,
        TRANSCRIBE_START_PCT,
        _transcribe,
    )

    events: list[tuple[str, float]] = []

    def _fake_transcribe(*, on_progress=None, on_status=None, **kwargs):
        # Имитируем поиск речи (VAD), потом обход сегментов: старт, середина, конец.
        assert on_progress is not None, "pipeline обязан прокинуть колбэк"
        assert on_status is not None, "и строку состояния для поиска речи"
        on_status("поиск речи: 50%")
        on_progress(0.0, "речь найдена")
        on_progress(0.5, "половина")
        on_progress(1.0, "готово")
        return {"segments": []}, "key", tmp_path / "cache.json"

    monkeypatch.setattr(gigaam_asr, "transcribe_with_gigaam", _fake_transcribe)

    config = PipelineConfig(transcriber="gigaam", device="cpu", asr_options={})
    config.validate()
    _transcribe(
        config=config,
        audio_path=tmp_path / "a.wav",
        workdir=tmp_path,
        gigaam_model="v3_ctc",
        whisper_cpp_bin=None,
        whisper_cpp_model=None,
        whisper_cpp_threads=None,
        whisper_cpp_extra=None,
        force_transcribe=False,
        keep_posteriors=False,
        progress=lambda stage, pct, msg="": events.append((stage, pct)),
    )

    percents = [pct for stage, pct in events if stage == "transcribe"]
    # Вход в стадию и поиск речи — НИЖЕ порога якоря ETA (загрузка весов и VAD
    # не должны попасть в замер скорости), дальше три тика движка внутри
    # диапазона стадии.
    assert percents[0] == TRANSCRIBE_ENTER_PCT
    assert percents[1] == TRANSCRIBE_ENTER_PCT
    assert percents[-3] == TRANSCRIBE_START_PCT
    assert percents[-2] == pytest.approx(
        (TRANSCRIBE_START_PCT + TRANSCRIBE_END_PCT) / 2
    )
    assert percents[-1] == TRANSCRIBE_END_PCT
    # Монотонность: прогресс-бар не должен прыгать назад.
    assert percents == sorted(percents)


def test_transcribe_enter_tick_stays_below_eta_anchor(tmp_path: Path, monkeypatch):
    """Контракт с UI: вход в стадию не должен становиться якорем ETA.

    JobScreen ставит якорь на первом тике с progress >= TRANSCRIBE_START_PCT и
    от него меряет скорость. Между входом в стадию и первым батчем проходит
    загрузка весов GigaAM (1–2 GB) + VAD — до минуты. Попади этот простой в
    замер, первая оценка была бы завышена в разы. Поэтому вход строго ниже
    порога, но выше конца extract_audio (10%), чтобы шкала не пошла назад.
    """
    from twitch_cut.pipeline import TRANSCRIBE_ENTER_PCT, TRANSCRIBE_START_PCT

    assert TRANSCRIBE_ENTER_PCT < TRANSCRIBE_START_PCT
    assert TRANSCRIBE_ENTER_PCT > 10.0  # конец стадии extract_audio


def test_gigaam_progress_clamps_out_of_range_fractions(tmp_path: Path, monkeypatch):
    """Кривая дробь от движка не должна выносить прогресс за пределы стадии.

    Если gigaam насчитает >1.0 (например, сегментов в батче больше, чем в
    плане), бар не имеет права уехать в detect/build-диапазон и потом
    прыгнуть назад.
    """
    import twitch_cut.gigaam_asr as gigaam_asr
    from twitch_cut.pipeline import (
        TRANSCRIBE_END_PCT,
        TRANSCRIBE_ENTER_PCT,
        TRANSCRIBE_START_PCT,
        _transcribe,
    )

    events: list[float] = []

    def _fake_transcribe(*, on_progress=None, **kwargs):
        on_progress(-0.5, "отрицательная")
        on_progress(3.0, "больше единицы")
        return {"segments": []}, "key", tmp_path / "cache.json"

    monkeypatch.setattr(gigaam_asr, "transcribe_with_gigaam", _fake_transcribe)

    config = PipelineConfig(transcriber="gigaam", device="cpu", asr_options={})
    config.validate()
    _transcribe(
        config=config,
        audio_path=tmp_path / "a.wav",
        workdir=tmp_path,
        gigaam_model="v3_ctc",
        whisper_cpp_bin=None,
        whisper_cpp_model=None,
        whisper_cpp_threads=None,
        whisper_cpp_extra=None,
        force_transcribe=False,
        keep_posteriors=False,
        progress=lambda stage, pct, msg="": events.append(pct),
    )

    assert events[-2] == TRANSCRIBE_START_PCT
    assert events[-1] == TRANSCRIBE_END_PCT
    assert all(TRANSCRIBE_ENTER_PCT <= p <= TRANSCRIBE_END_PCT for p in events)


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


# --- стадия спотинга по посториорам -----------------------------------------


def test_mock_run_does_not_pretend_to_spot(tmp_path: Path):
    """У mock-транскрипта посториоров нет по определению, поэтому стадии не
    было — и в decisions.json не должно быть её следов. Иначе прогон выглядел бы
    так, будто спотинг всё проверил и ничего не нашёл."""
    result = _run(tmp_path, "gigaam")

    caches = json.loads(Path(result["decisions_path"]).read_text(encoding="utf-8"))["caches"]
    assert "spotting" not in caches
    assert "spotting_key" not in caches


def _quiet_posteriors(workdir: Path, transcript_key: str, word: str = "ебан") -> Path:
    """Посториоры с ОДНИМ тихим корнем: пики 0.35 — argmax там blank, то есть в
    тексте распознавания слова нет и найти его может только спотинг."""
    import numpy as np

    from twitch_cut.posteriors import posteriors_path, write_posteriors

    vocab = sorted({*word, "п", "и", "с", "р", " "})
    blank = len(vocab)
    filler = 0.02
    probs = np.full((40, len(vocab) + 1), filler / len(vocab), dtype=np.float32)
    probs[:, blank] = 1.0 - filler
    for i, char in enumerate(word):
        frame = 5 + i * 2
        probs[frame] = filler / (len(vocab) - 1)
        probs[frame, vocab.index(char)] = 0.35
        probs[frame, blank] = 1.0 - filler - 0.35
    path = posteriors_path(workdir, transcript_key)
    written = write_posteriors(
        path,
        [(np.log(probs), 0.04)],
        starts=[10.2],
        vocab=vocab,
        blank_id=blank,
    )
    assert written is not None
    return path


def _spot(tmp_path: Path, *, matches=(), force: bool = False):
    from twitch_cut.pipeline import _spot_stage

    transcript = {
        "segments": [
            {
                "id": "seg_7",
                "start": 10.0,
                "end": 13.0,
                "words": [{"word": "епанисир", "start": 10.2, "end": 10.6, "score": 0.4}],
            }
        ]
    }
    # Порог ниже дефолтного: пики в подделке тихие (0.35), а меряем здесь стык
    # стадии с кэшем, а не калибровку порога (она в config.SPOT_MIN_SCORE).
    config = PipelineConfig(
        transcriber="gigaam", device="cpu", asr_options={}, spot_min_score=0.3
    )
    config.validate()
    return _spot_stage(
        config=config,
        workdir=tmp_path / "work",
        transcript=transcript,
        transcript_key="k1",
        detection_key="d1",
        matches=list(matches),
        range_in=parse_timecode("00:00:00"),
        force=force,
        progress=lambda stage, pct, msg="": None,
    )


def test_spot_stage_without_posteriors_changes_nothing(tmp_path: Path):
    """Посториоров нет (старый кэш транскрипта, whisperx, mock) — это не ошибка
    прогона: стадии просто не было. Матчи текстового детектора обязаны дойти до
    decisions без изменений, и ключа кэша в отчёте быть не должно."""
    matches, key, cache = _spot(tmp_path)

    assert matches == []
    assert key is None
    assert cache is None


def test_spot_stage_adds_the_candidate_and_caches_it(tmp_path: Path):
    """Стадия дополняет текстовые матчи кандидатами и складывает результат в
    свой кэш: правка порога должна стоить секунды DP, а не повторный ASR."""
    npz = _quiet_posteriors(tmp_path / "work", "k1")

    matches, key, cache = _spot(tmp_path)

    assert [m.banword for m in matches] == ["<spot:ебан>"]
    assert key is not None and cache is not None and cache.exists()

    # Кэш читается вместо повторного DP — убираем посториоры и повторяем вызов.
    npz.unlink()
    again, again_key, _ = _spot(tmp_path)
    assert [m.banword for m in again] == ["<spot:ебан>"]
    assert again_key == key


@pytest.mark.parametrize("spot_posteriors", [True, False])
def test_posteriors_are_requested_exactly_when_the_stage_will_run(
    tmp_path: Path, monkeypatch, spot_posteriors: bool
):
    """Связка тумблера и ASR: посториоры просим у движка тогда и только тогда,
    когда стадия спотинга будет их читать.

    Разъедься эти два места — обе стороны ломаются молча: с тумблером включённым
    и без просьбы стадия каждый раз находила бы пустоту («посториоров нет»), а с
    выключенным и с просьбой на диск ложились бы мегабайты, которые никто не
    прочитает.
    """
    import twitch_cut.gigaam_asr as gigaam_asr
    import twitch_cut.pipeline as pipeline

    stream = tmp_path / "stream.mp4"
    _fake_media(stream)
    banwords = tmp_path / "banwords.txt"
    _write_banwords(banwords)
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"\x00")
    asked: list[bool] = []

    def _fake_extract(stream_path, start, end, workdir, force=False):
        return audio, "audio-key"

    def _fake_transcribe(*, keep_posteriors=False, on_progress=None, **kwargs):
        asked.append(keep_posteriors)
        return {"segments": []}, "tkey", tmp_path / "cache.json"

    monkeypatch.setattr(pipeline, "extract_audio_range", _fake_extract)
    monkeypatch.setattr(gigaam_asr, "transcribe_with_gigaam", _fake_transcribe)

    config = PipelineConfig(
        transcriber="gigaam", device="cpu", asr_options={}, spot_posteriors=spot_posteriors
    )
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
    )

    assert asked == [spot_posteriors]
