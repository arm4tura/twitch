"""edit_journal.py — журнал ручных правок: «программа решила так, человек решил иначе».

Зачем. Качество детекции видно только в сравнении с человеком, но пересказывать
правки словами («вот тут я сдвинул начало, а тут снял мьют») — дорого и
невоспроизводимо. Когда в настройках включён «Журнал правок», каждое сохранение
из экрана «Правка» дописывает в ОТДЕЛЬНЫЙ файл (`<decisions>.edits.json`), что
именно изменилось относительно решения программы. `twitch-cut audit` потом
превращает это в раздел отчёта с диагнозом по каждой правке.

Три вещи, которые делают журнал полезным, а не просто дампом:

1. BASELINE — снимок того, что выдал ПАЙПЛАЙН, а не предыдущее сохранение.
   Пишется по завершении обработки (`start_journal`), поэтому сравнение всегда
   идёт с машинным решением, даже после десяти ручных сохранений подряд.
2. Сессии — что изменилось между сохранениями. По ним видно, что человек
   передумал (сдвинул и вернул), а не только итог.
3. Диагностические поля мьюта (`timing_source`, `match_type`, `audio_trim_*`)
   едут в снимок вместе с границами: без них по журналу нельзя понять, ПОЧЕМУ
   программа так решила.
4. Машинное не выдаётся за ручное. Что программа дописала в decisions.json уже
   после обработки, уезжает в baseline (`refresh_baseline`), а сегменты
   авто-детекта, которые сохраняет человек, узнаются по провенансу (`origin`).

Журнал НИКОГДА не должен ломать сохранение: любая ошибка логируется и глотается
вызывающей стороной (см. `server/app.py::write_decisions_endpoint`).

Времена в журнале — как в decisions.json, то есть СЕКУНДЫ (mutes/silences —
`start`/`end`, оффтоп — `start_s`/`end_s`).
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import logging
from pathlib import Path
from typing import Any, Callable, Sequence

logger = logging.getLogger(__name__)

SCHEMA_VERSION = "1.0"

MOVE_EPS_S = 0.005
"""Сдвиг меньше 5 мс — это округление round3 при сохранении, а не правка."""

MAX_SESSIONS = 300
"""Потолок истории: журнал должен оставаться читаемым файлом, а не логом на гиг."""

_MUTE_FIELDS: tuple[str, ...] = (
    "id",
    "start",
    "end",
    "word",
    "status",
    "reason",
    "banword",
    "match_type",
    "timing_source",
    "confidence",
    "needs_review",
    "source",
    "segment_id",
    "raw_local_start",
    "raw_local_end",
    "audio_trim_start_ms",
    "audio_trim_end_ms",
    # Что машина думала про звук под окном и что с окном из-за этого сделала.
    # Без этих полей снимок не даёт померить, помогает ли вердикт «окно на
    # тишине» и расширение по нему: в eval/audit они просто не доезжают.
    "reanchor",
    "reanchor_offset_ms",
    "expand_start_ms",
    "expand_end_ms",
    # Сколько мьюта стоит на разметке предыдущего слова (mute_head_cross_prev_ms).
    # Без этого поля в снимке нельзя померить цену ручки: правки человека по
    # голове окна и так видны, а вот сколько за них заплачено соседним словом —
    # нет.
    "head_cross_ms",
)
"""Что тащим в снимок мьюта. Не весь объект: остальное не помогает объяснить
решение программы, а файл раздувает."""

_SILENCE_FIELDS: tuple[str, ...] = ("id", "start", "end", "status", "source")

_OFFTOPIC_FIELDS: tuple[str, ...] = (
    "start_s",
    "end_s",
    "status",
    "topic",
    "reason",
    "confidence",
    "model",
    "stage",
    "anchored",
)


def journal_path(decisions_path: Path | str) -> Path:
    """`.../decisions.json` → `.../decisions.edits.json` (рядом с решениями)."""
    p = Path(decisions_path)
    return p.with_name(f"{p.stem}.edits.json")


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _pick(item: Any, fields: Sequence[str]) -> dict[str, Any]:
    if not isinstance(item, dict):
        return {}
    return {k: item[k] for k in fields if k in item}


def _snapshot(doc: dict[str, Any]) -> dict[str, Any]:
    """Срез документа решений: только то, что человек правит на таймлайне."""
    offtopic = doc.get("offtopic")
    segments = offtopic.get("segments") if isinstance(offtopic, dict) else None
    return {
        "mutes": [_pick(m, _MUTE_FIELDS) for m in _as_list(doc.get("mutes"))],
        "silences": [_pick(s, _SILENCE_FIELDS) for s in _as_list(doc.get("silences"))],
        "offtopic": [_pick(o, _OFFTOPIC_FIELDS) for o in _as_list(segments)],
    }


def _as_list(value: Any) -> list[Any]:
    return [v for v in value if isinstance(v, dict)] if isinstance(value, list) else []


def _f(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _key_by_id(item: dict[str, Any], start_field: str) -> str:
    """Идентификатор для сопоставления «до/после».

    Приоритет у `id` — он переживает сдвиг границ, поэтому подвинутый мьют
    опознаётся как подвинутый, а не как «удалили один, добавили другой».
    Оффтоп-сегменты id не имеют, для них ключ — начало с точностью до мс.
    """
    ident = item.get("id")
    if isinstance(ident, str) and ident:
        return ident
    start = _f(item.get(start_field))
    return f"@{round(start * 1000)}" if start is not None else "@?"


def _changes(
    old: list[dict[str, Any]],
    new: list[dict[str, Any]],
    *,
    fields: Sequence[str],
    start_field: str,
    end_field: str,
) -> dict[str, list[dict[str, Any]]]:
    """Что стало с каждым элементом: добавлен / убран / сдвинут / сменил статус."""
    key: Callable[[dict[str, Any]], str] = lambda item: _key_by_id(item, start_field)
    old_idx = {key(i): i for i in old}
    new_idx = {key(i): i for i in new}

    out: dict[str, list[dict[str, Any]]] = {
        "added": [],
        "removed": [],
        "moved": [],
        "status": [],
    }
    for k, cur in new_idx.items():
        prev = old_idx.get(k)
        if prev is None:
            out["added"].append({"key": k, "after": _pick(cur, fields)})
            continue
        _diff_pair(out, k, prev, cur, fields=fields, start_field=start_field, end_field=end_field)
    for k, prev in old_idx.items():
        if k not in new_idx:
            out["removed"].append({"key": k, "before": _pick(prev, fields)})
    return out


def _diff_pair(
    out: dict[str, list[dict[str, Any]]],
    key: str,
    prev: dict[str, Any],
    cur: dict[str, Any],
    *,
    fields: Sequence[str],
    start_field: str,
    end_field: str,
) -> None:
    """Один и тот же элемент до/после: границы и статус проверяем независимо —
    человек может и сдвинуть, и снять мьют одним сохранением."""
    d_start = _delta(prev.get(start_field), cur.get(start_field))
    d_end = _delta(prev.get(end_field), cur.get(end_field))
    if d_start is not None or d_end is not None:
        out["moved"].append(
            {
                "key": key,
                "before": _pick(prev, fields),
                "after": _pick(cur, fields),
                "d_start": d_start or 0.0,
                "d_end": d_end or 0.0,
            }
        )
    prev_status = prev.get("status")
    cur_status = cur.get("status")
    if prev_status != cur_status:
        out["status"].append(
            {
                "key": key,
                "after": _pick(cur, fields),
                "status_from": prev_status,
                "status_to": cur_status,
            }
        )


def _delta(before: Any, after: Any) -> float | None:
    """Знаковая разница в секундах или None, если сдвига нет (или нет чисел)."""
    a, b = _f(before), _f(after)
    if a is None or b is None:
        return None
    diff = b - a
    return round(diff, 3) if abs(diff) > MOVE_EPS_S else None


_LAYERS: tuple[tuple[str, tuple[str, ...], str, str], ...] = (
    ("mutes", _MUTE_FIELDS, "start", "end"),
    ("silences", _SILENCE_FIELDS, "start", "end"),
    ("offtopic", _OFFTOPIC_FIELDS, "start_s", "end_s"),
)


def origin(item: Any) -> str:
    """`manual` — сегмент поставила рука, `auto` — авто-детект.

    Зачем. Не всё, что появилось ПОСЛЕ базового снимка, сделал человек: детект
    тишины в «Правке» и AI-детект оффтопа тоже досыпают сегменты, и без
    провенанса они читались бы как «человек добавил 40 участков». Улики в
    decisions.json — соглашение об id из UI (`*_manual_*`) и поля
    `source`/`stage` («manual» у ручных).
    """
    if not isinstance(item, dict):
        return "auto"
    for name in ("source", "stage"):
        value = item.get(name)
        if isinstance(value, str) and value.strip().lower() == "manual":
            return "manual"
    ident = item.get("id")
    if isinstance(ident, str) and "_manual_" in ident:
        return "manual"
    return "auto"


def split_added_by_origin(
    entries: Any, *, side: str = "after"
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Корзину `added` — на ручные и машинные. Осмысленно ТОЛЬКО для `added`:
    снять или подвинуть машинный сегмент может лишь человек, там провенанс
    самого сегмента ничего не говорит о том, кто его правил."""
    manual: list[dict[str, Any]] = []
    auto: list[dict[str, Any]] = []
    for entry in entries if isinstance(entries, list) else []:
        if not isinstance(entry, dict):
            continue
        (manual if origin(entry.get(side)) == "manual" else auto).append(entry)
    return manual, auto


def diff_snapshots(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    """Разница двух срезов: по слою — added/removed/moved/status."""
    return {
        layer: _changes(
            _as_list(before.get(layer)),
            _as_list(after.get(layer)),
            fields=fields,
            start_field=start_field,
            end_field=end_field,
        )
        for layer, fields, start_field, end_field in _LAYERS
    }


def count_changes(diff: dict[str, Any]) -> dict[str, int]:
    """Сколько правок в разнице — по видам, суммарно по всем слоям."""
    totals = {"added": 0, "removed": 0, "moved": 0, "status": 0}
    for layer in diff.values():
        if not isinstance(layer, dict):
            continue
        for kind in totals:
            totals[kind] += len(_as_list(layer.get(kind)))
    return totals


def is_empty(diff: dict[str, Any]) -> bool:
    return not any(count_changes(diff).values())


# --- файл журнала ------------------------------------------------------------


def load_journal(decisions_path: Path | str) -> dict[str, Any] | None:
    """Прочитать журнал рядом с решениями. Нет файла или битый — None."""
    return read_journal(journal_path(decisions_path))


def read_journal(path: Path | str) -> dict[str, Any] | None:
    """То же, но по явному пути (для `audit --edits`)."""
    path = Path(path)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (OSError, ValueError) as exc:
        logger.warning("edit journal unreadable (%s): %s", path, exc)
        return None
    return data if isinstance(data, dict) else None


def _write_journal(path: Path, journal: dict[str, Any]) -> None:
    """Атомарно: сохранение решений уже прошло, недописанный журнал не нужен."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(journal, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def _blank(decisions_path: Path, baseline: dict[str, Any], source: str, at: str) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "decisions": str(decisions_path),
        "created_at": at,
        "updated_at": at,
        "baseline": {"captured_at": at, "source": source, **baseline},
        "sessions": [],
        "totals": {"added": 0, "removed": 0, "moved": 0, "status": 0, "saves": 0},
        "dropped_sessions": 0,
    }


def start_journal(
    decisions_path: Path | str,
    doc: dict[str, Any],
    *,
    at: str | None = None,
) -> Path:
    """Запомнить решение ПАЙПЛАЙНА как базовый снимок (вызывается по завершении
    обработки). Именно с ним потом сравниваются все ручные правки.

    Повторная обработка того же workdir не стирает историю: старый baseline
    уезжает в сессию с пометкой `pipeline`, так что видно ещё и как изменился
    сам пайплайн между прогонами.
    """
    path = journal_path(decisions_path)
    stamp = at or _now_iso()
    fresh = _snapshot(doc)
    journal = load_journal(decisions_path)
    if journal is None:
        journal = _blank(Path(decisions_path), fresh, "pipeline", stamp)
    else:
        old = journal.get("baseline")
        changes = diff_snapshots(old if isinstance(old, dict) else {}, fresh)
        journal["baseline"] = {"captured_at": stamp, "source": "pipeline", **fresh}
        if not is_empty(changes):
            _append_session(
                journal,
                {
                    "at": stamp,
                    "kind": "pipeline",
                    "note": "обработка прогнана заново — базовый снимок обновлён",
                    "counts": count_changes(changes),
                    "changes": changes,
                },
            )
    journal["updated_at"] = stamp
    _write_journal(path, journal)
    return path


def refresh_baseline(
    decisions_path: Path | str,
    doc: dict[str, Any],
    layers: Sequence[str],
    *,
    note: str,
    at: str | None = None,
) -> Path | None:
    """Обновить baseline ТОЛЬКО по указанным слоям — их посчитала машина.

    Нужно там, где программа дописывает в decisions.json уже ПОСЛЕ обработки
    (AI-детект оффтопа). Без этого машинные сегменты попадали бы в отчёт как
    «человек добавил»: в базовом снимке их нет, а в текущем файле есть.

    Журнала нет — ничего не создаём и возвращаем None: baseline тогда снимется
    при первом сохранении, и машинные сегменты уже будут внутри (`record_save`).
    """
    journal = load_journal(decisions_path)
    if journal is None:
        return None
    stamp = at or _now_iso()
    fresh = _snapshot(doc)
    picked = {layer: fresh[layer] for layer in layers if layer in fresh}
    if not picked:
        return None

    baseline = journal.get("baseline")
    baseline = dict(baseline) if isinstance(baseline, dict) else {}
    changes = diff_snapshots({k: baseline.get(k) for k in picked}, picked)
    baseline.update(picked)
    # Слой обновлён отдельно от снимка пайплайна — помечаем, чтобы в отчёте не
    # выглядело, будто всё это выдала обработка одним заходом.
    refreshed = baseline.get("refreshed")
    refreshed = dict(refreshed) if isinstance(refreshed, dict) else {}
    for layer in picked:
        refreshed[layer] = {"at": stamp, "note": note}
    baseline["refreshed"] = refreshed
    journal["baseline"] = baseline

    if not is_empty(changes):
        _append_session(
            journal,
            {
                "at": stamp,
                "kind": "machine",
                "note": note,
                "layers": sorted(picked),
                "counts": count_changes(changes),
                "changes": changes,
            },
        )
    journal["updated_at"] = stamp
    _write_journal(journal_path(decisions_path), journal)
    return journal_path(decisions_path)


def _append_session(journal: dict[str, Any], session: dict[str, Any]) -> None:
    sessions = journal.get("sessions")
    if not isinstance(sessions, list):
        sessions = []
    sessions.append(session)
    if len(sessions) > MAX_SESSIONS:
        dropped = len(sessions) - MAX_SESSIONS
        sessions = sessions[dropped:]
        journal["dropped_sessions"] = int(journal.get("dropped_sessions") or 0) + dropped
    journal["sessions"] = sessions


def record_save(
    decisions_path: Path | str,
    new_doc: dict[str, Any],
    previous: dict[str, Any] | None = None,
    *,
    at: str | None = None,
) -> Path:
    """Дописать в журнал, что изменилось этим сохранением.

    `previous` — документ, который лежал на диске ДО сохранения (None, если
    файла не было). Диффим именно с ним: сессия должна отвечать на вопрос «что
    человек сделал за этот заход», а сравнение с машинным решением даёт
    baseline.

    Если журнала ещё нет (обработка шла без включённого журнала), baseline
    берётся из `previous` с пометкой `first_save` — сравнение будет с первым
    увиденным состоянием, а не с пайплайном. Это честнее, чем врать про
    источник.
    """
    path = journal_path(decisions_path)
    stamp = at or _now_iso()
    after = _snapshot(new_doc)
    before = _snapshot(previous) if previous is not None else None

    journal = load_journal(decisions_path)
    if journal is None:
        base = before if before is not None else after
        source = "first_save" if before is not None else "unknown"
        journal = _blank(Path(decisions_path), base, source, stamp)

    if before is not None:
        changes = diff_snapshots(before, after)
        if not is_empty(changes):
            counts = count_changes(changes)
            _append_session(
                journal,
                {"at": stamp, "kind": "save", "counts": counts, "changes": changes},
            )
            totals = journal.get("totals")
            if not isinstance(totals, dict):
                totals = {}
            for kind, value in counts.items():
                totals[kind] = int(totals.get(kind) or 0) + value
            totals["saves"] = int(totals.get("saves") or 0) + 1
            journal["totals"] = totals

    journal["current"] = {layer: len(_as_list(after.get(layer))) for layer, *_ in _LAYERS}
    journal["updated_at"] = stamp
    _write_journal(path, journal)
    return path


def net_changes(journal: dict[str, Any], doc: dict[str, Any]) -> dict[str, Any]:
    """ИТОГ: baseline (что решила программа) против текущего decisions.json.

    Это то, что нужно для разбора «как должно быть vs как сделала программа»:
    промежуточные метания по сессиям тут не мешают.
    """
    baseline = journal.get("baseline")
    return diff_snapshots(baseline if isinstance(baseline, dict) else {}, _snapshot(doc))


def baseline_source(journal: dict[str, Any]) -> str:
    baseline = journal.get("baseline")
    source = baseline.get("source") if isinstance(baseline, dict) else None
    return source if isinstance(source, str) else "unknown"
