"""Тесты журнала ручных правок (edit_journal + раздел в audit + PUT /decisions).

Смысл журнала — ответить на вопрос «человек решил иначе, ПОЧЕМУ программа
решила так». Поэтому тесты держат три вещи:

1. Baseline — это решение ПАЙПЛАЙНА, а не предыдущее сохранение. Иначе после
   десяти сохранений сравнивать будет не с чем.
2. Сдвиг границ опознаётся как сдвиг, а не как «удалили один мьют, добавили
   другой» — сопоставление идёт по id.
3. Журнал НИКОГДА не ломает сохранение decisions.json и ничего в него не
   дописывает (иначе поехал бы round-trip контракт PUT /decisions).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from twitch_cut import edit_journal as ej
from twitch_cut.server.app import create_app
from twitch_cut.server.settings import save_settings


def _mute(mid: str, start: float, end: float, **extra) -> dict:
    base = {
        "id": mid,
        "start": start,
        "end": end,
        "word": "блять",
        "status": "accepted",
        "match_type": "surface",
        "timing_source": "word",
        "banword": "блять",
    }
    base.update(extra)
    return base


def _doc(mutes: list[dict], silences: list[dict] | None = None) -> dict:
    return {
        "schema_version": "1.1",
        "mutes": mutes,
        "cuts": [],
        "silences": silences or [],
        "caches": {},
    }


# --- diff --------------------------------------------------------------------


def test_journal_path_sits_next_to_decisions(tmp_path: Path):
    assert ej.journal_path(tmp_path / "decisions.json") == tmp_path / "decisions.edits.json"


def test_added_and_removed(tmp_path: Path):
    dec = tmp_path / "decisions.json"
    ej.start_journal(dec, _doc([_mute("mute_000001", 1.0, 1.5)]))
    ej.record_save(
        dec,
        _doc([_mute("mute_manual_2000", 2.0, 2.4, source="manual", word="хуй")]),
        _doc([_mute("mute_000001", 1.0, 1.5)]),
    )
    journal = ej.load_journal(dec)
    assert journal is not None
    net = ej.net_changes(journal, _doc([_mute("mute_manual_2000", 2.0, 2.4)]))
    assert [m["after"]["id"] for m in net["mutes"]["added"]] == ["mute_manual_2000"]
    assert [m["before"]["id"] for m in net["mutes"]["removed"]] == ["mute_000001"]
    assert ej.count_changes(net) == {"added": 1, "removed": 1, "moved": 0, "status": 0}


def test_moved_is_not_add_plus_remove(tmp_path: Path):
    """Сдвиг узнаётся по id: иначе в отчёте вместо «границы поехали» была бы пара
    «удалил/добавил», и диагноз по padding посчитать было бы нечем."""
    dec = tmp_path / "decisions.json"
    before = _doc([_mute("mute_000001", 7.075, 7.225)])
    after = _doc([_mute("mute_000001", 6.88, 7.15)])
    ej.start_journal(dec, before)
    ej.record_save(dec, after, before)
    net = ej.net_changes(ej.load_journal(dec), after)
    moved = net["mutes"]["moved"]
    assert len(moved) == 1
    assert moved[0]["d_start"] == pytest.approx(-0.195, abs=0.001)
    assert moved[0]["d_end"] == pytest.approx(-0.075, abs=0.001)
    assert not net["mutes"]["added"] and not net["mutes"]["removed"]


def test_rounding_is_not_an_edit(tmp_path: Path):
    """round3 при сохранении даёт микросдвиги — они не должны попадать в журнал."""
    dec = tmp_path / "decisions.json"
    before = _doc([_mute("mute_000001", 1.0, 1.5)])
    after = _doc([_mute("mute_000001", 1.001, 1.4995)])
    ej.start_journal(dec, before)
    ej.record_save(dec, after, before)
    journal = ej.load_journal(dec)
    assert journal["sessions"] == []
    assert ej.is_empty(ej.net_changes(journal, after))


def test_status_change_is_separate_from_move(tmp_path: Path):
    dec = tmp_path / "decisions.json"
    before = _doc([_mute("mute_000001", 1.0, 1.5)])
    after = _doc([_mute("mute_000001", 1.0, 1.5, status="rejected")])
    ej.start_journal(dec, before)
    ej.record_save(dec, after, before)
    net = ej.net_changes(ej.load_journal(dec), after)
    assert net["mutes"]["moved"] == []
    assert net["mutes"]["status"][0]["status_from"] == "accepted"
    assert net["mutes"]["status"][0]["status_to"] == "rejected"


def test_baseline_is_pipeline_after_many_saves(tmp_path: Path):
    """Главный инвариант: сравнение идёт с машинным решением, а не с предыдущим
    сохранением. Иначе после нескольких заходов правки «растворяются»."""
    dec = tmp_path / "decisions.json"
    pipeline = _doc([_mute("mute_000001", 1.0, 1.5)])
    ej.start_journal(dec, pipeline)
    step1 = _doc([_mute("mute_000001", 1.2, 1.5)])
    step2 = _doc([_mute("mute_000001", 1.4, 1.9)])
    ej.record_save(dec, step1, pipeline)
    ej.record_save(dec, step2, step1)
    journal = ej.load_journal(dec)
    assert len(journal["sessions"]) == 2  # история метаний сохранена
    net = ej.net_changes(journal, step2)
    moved = net["mutes"]["moved"][0]
    # Итог считается от пайплайна: 1.0 → 1.4 и 1.5 → 1.9.
    assert moved["d_start"] == pytest.approx(0.4, abs=0.001)
    assert moved["d_end"] == pytest.approx(0.4, abs=0.001)
    assert journal["totals"]["saves"] == 2


def test_reprocess_keeps_history_and_resets_baseline(tmp_path: Path):
    """Повторная обработка того же workdir обновляет baseline, но история правок
    не теряется, а изменение самого пайплайна видно отдельной сессией."""
    dec = tmp_path / "decisions.json"
    first = _doc([_mute("mute_000001", 1.0, 1.5)])
    ej.start_journal(dec, first)
    edited = _doc([_mute("mute_000001", 1.0, 1.7)])
    ej.record_save(dec, edited, first)
    reprocessed = _doc([_mute("mute_000001", 1.05, 1.45)])
    ej.start_journal(dec, reprocessed)

    journal = ej.load_journal(dec)
    kinds = [s["kind"] for s in journal["sessions"]]
    assert kinds == ["save", "pipeline"]
    assert journal["baseline"]["mutes"][0]["end"] == 1.45
    assert ej.baseline_source(journal) == "pipeline"


def test_first_save_baseline_is_labelled_honestly(tmp_path: Path):
    """Журнал включили после обработки — сравниваем с первым увиденным файлом и
    НЕ выдаём это за решение пайплайна."""
    dec = tmp_path / "decisions.json"
    ej.record_save(dec, _doc([_mute("m1", 1.0, 1.6)]), _doc([_mute("m1", 1.0, 1.5)]))
    journal = ej.load_journal(dec)
    assert ej.baseline_source(journal) == "first_save"
    assert journal["baseline"]["mutes"][0]["end"] == 1.5


def test_silences_and_offtopic_are_journalled(tmp_path: Path):
    dec = tmp_path / "decisions.json"
    before = {"mutes": [], "silences": [{"id": "s1", "start": 10.0, "end": 12.0}], "offtopic": {"segments": [{"start_s": 30.0, "end_s": 60.0}]}}
    after = {"mutes": [], "silences": [{"id": "s1", "start": 10.5, "end": 12.0}], "offtopic": {"segments": []}}
    ej.start_journal(dec, before)
    ej.record_save(dec, after, before)
    net = ej.net_changes(ej.load_journal(dec), after)
    assert net["silences"]["moved"][0]["d_start"] == pytest.approx(0.5, abs=0.001)
    assert len(net["offtopic"]["removed"]) == 1


# --- машинное не выдаём за ручное --------------------------------------------


def test_origin_tells_hand_from_auto_detect():
    assert ej.origin({"id": "silence_auto_1000"}) == "auto"
    assert ej.origin({"id": "silence_manual_1000"}) == "manual"
    assert ej.origin({"id": "mute_000001", "source": "manual"}) == "manual"
    # Оффтоп id не переживает сохранение — там улика в stage.
    assert ej.origin({"start_s": 1.0, "stage": "manual"}) == "manual"
    assert ej.origin({"start_s": 1.0, "stage": "cheap", "model": "gpt"}) == "auto"


def test_auto_detected_silences_are_not_counted_as_human_edits(tmp_path: Path):
    """Детект тишины в «Правке» насыпает сегменты уже после снимка пайплайна, а
    сохраняет их человек. В diff они попадают, но провенанс отделяет их от рук."""
    dec = tmp_path / "decisions.json"
    pipeline = _doc([], silences=[])
    ej.start_journal(dec, pipeline)
    saved = _doc(
        [],
        silences=[
            {"id": "silence_auto_5000", "start": 5.0, "end": 6.0, "status": "accepted"},
            {"id": "silence_manual_9000", "start": 9.0, "end": 9.8, "status": "accepted"},
        ],
    )
    ej.record_save(dec, saved, pipeline)

    net = ej.net_changes(ej.load_journal(dec), saved)
    manual, auto = ej.split_added_by_origin(net["silences"]["added"])
    assert [m["after"]["id"] for m in manual] == ["silence_manual_9000"]
    assert [a["after"]["id"] for a in auto] == ["silence_auto_5000"]


def test_refresh_baseline_folds_machine_layer_into_baseline(tmp_path: Path):
    """AI-детект оффтопа пишет decisions.json мимо PUT /decisions. После
    refresh_baseline его сегменты — часть машинного решения, а не правка."""
    dec = tmp_path / "decisions.json"
    pipeline = _doc([_mute("mute_000001", 1.0, 1.5)])
    ej.start_journal(dec, pipeline)

    with_offtopic = {
        **pipeline,
        "offtopic": {"segments": [{"start_s": 30.0, "end_s": 60.0, "stage": "cheap"}]},
    }
    ej.refresh_baseline(dec, with_offtopic, ("offtopic",), note="AI-детект оффтопа")

    journal = ej.load_journal(dec)
    net = ej.net_changes(journal, with_offtopic)
    assert ej.is_empty(net)  # ничего человек не менял
    assert journal["baseline"]["offtopic"][0]["start_s"] == 30.0
    # Слой обновлён отдельно — это видно, а мьюты пайплайна не тронуты.
    assert journal["baseline"]["refreshed"]["offtopic"]["note"] == "AI-детект оффтопа"
    assert journal["baseline"]["mutes"][0]["id"] == "mute_000001"
    assert [s["kind"] for s in journal["sessions"]] == ["machine"]


def test_refresh_baseline_keeps_human_edits_of_other_layers(tmp_path: Path):
    dec = tmp_path / "decisions.json"
    pipeline = _doc([_mute("mute_000001", 1.0, 1.5)])
    ej.start_journal(dec, pipeline)
    edited = _doc([_mute("mute_000001", 1.0, 1.2)])
    ej.record_save(dec, edited, pipeline)

    with_offtopic = {**edited, "offtopic": {"segments": [{"start_s": 5.0, "end_s": 9.0}]}}
    ej.refresh_baseline(dec, with_offtopic, ("offtopic",), note="AI-детект оффтопа")

    net = ej.net_changes(ej.load_journal(dec), with_offtopic)
    assert net["mutes"]["moved"][0]["d_end"] == pytest.approx(-0.3, abs=0.001)
    assert ej.count_changes(net) == {"added": 0, "removed": 0, "moved": 1, "status": 0}


def test_refresh_baseline_without_journal_creates_nothing(tmp_path: Path):
    """Журнал не включён — не создаём его задним числом: baseline из текущего
    файла соврал бы, что все ручные правки сделала программа."""
    dec = tmp_path / "decisions.json"
    assert ej.refresh_baseline(dec, _doc([]), ("offtopic",), note="AI") is None
    assert not ej.journal_path(dec).exists()


def test_corrupt_journal_does_not_raise(tmp_path: Path):
    dec = tmp_path / "decisions.json"
    ej.journal_path(dec).write_text("{not json", encoding="utf-8")
    assert ej.load_journal(dec) is None
    # Битый файл перезаписывается новым журналом, а не роняет сохранение.
    ej.record_save(dec, _doc([]), _doc([]))
    assert ej.load_journal(dec) is not None


def test_sessions_are_capped(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(ej, "MAX_SESSIONS", 3)
    dec = tmp_path / "decisions.json"
    ej.start_journal(dec, _doc([_mute("m1", 1.0, 1.5)]))
    for i in range(6):
        before = _doc([_mute("m1", 1.0, 1.5 + i * 0.1)])
        after = _doc([_mute("m1", 1.0, 1.6 + i * 0.1)])
        ej.record_save(dec, after, before)
    journal = ej.load_journal(dec)
    assert len(journal["sessions"]) == 3
    assert journal["dropped_sessions"] == 3


# --- PUT /decisions ----------------------------------------------------------


def _client(tmp_path: Path, monkeypatch) -> TestClient:
    monkeypatch.setattr("twitch_cut.config.DATA_ROOT", tmp_path)
    return TestClient(create_app())


def test_put_decisions_writes_journal_when_enabled(tmp_path: Path, monkeypatch):
    save_settings({"edit_journal": True})
    client = _client(tmp_path, monkeypatch)
    dec = tmp_path / "decisions.json"
    first = _doc([_mute("mute_000001", 1.0, 1.5)])
    client.put(f"/decisions?path={dec}", json={"decisions": first})
    second = _doc([_mute("mute_000001", 0.9, 1.5)])
    client.put(f"/decisions?path={dec}", json={"decisions": second})

    journal = ej.load_journal(dec)
    assert journal is not None
    assert journal["sessions"][0]["changes"]["mutes"]["moved"][0]["d_start"] == pytest.approx(-0.1, abs=0.001)
    # В сам decisions.json журнал не добавил НИ ОДНОГО поля.
    assert json.loads(dec.read_text(encoding="utf-8")) == second


def test_put_decisions_without_flag_writes_nothing(tmp_path: Path, monkeypatch):
    client = _client(tmp_path, monkeypatch)
    dec = tmp_path / "decisions.json"
    client.put(f"/decisions?path={dec}", json={"decisions": _doc([])})
    assert not ej.journal_path(dec).exists()


def test_journal_failure_does_not_break_save(tmp_path: Path, monkeypatch):
    """Диагностика не имеет права ронять сохранение проекта."""
    save_settings({"edit_journal": True})
    client = _client(tmp_path, monkeypatch)

    def boom(*args, **kwargs):
        raise RuntimeError("disk on fire")

    monkeypatch.setattr(ej, "record_save", boom)
    dec = tmp_path / "decisions.json"
    r = client.put(f"/decisions?path={dec}", json={"decisions": _doc([])})
    assert r.status_code == 200
    assert dec.exists()
