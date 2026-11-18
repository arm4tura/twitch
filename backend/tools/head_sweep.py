"""Замер головного допуска мьюта и стража по звуку — против таймлайна человека.

Зачем. `mute_head_cross_prev_ms` (config.py) и новый страж
`mute_head_cross_voiced_guard` тюнились «слышно»-линейкой (утечка внутри окна
человека), а та СТРУКТУРНО слепа к обратной ошибке — замьюченной ЧИСТОЙ речи
соседа. Поэтому каждый прогон линейки мог только поднимать допуск. Здесь допуск
мерится ДВУСТОРОННЕ: и недокрытие (мат слышен), и переглушка (съеден сосед).

Чем отличается от sensitivity_levels.py и от `twitch-cut eval`:
* sensitivity_levels сравнивает ГОТОВЫЕ прогоны машины с чужим таймлайном —
  здесь прогонов нет, окна ПЕРЕСОБИРАЮТСЯ из кэша детекции с разными config, а
  распознавание и детекция не трогаются (кэш общий, ключ детекции неизменен —
  см. pipeline._detection_cache_path). Одна запись, много кандидатов.
* eval читает журнал правок по `id` — он не годится для чужого набора окон
  (свежая сборка не совпадает id с правками человека). Поэтому сверка здесь
  ПО ПЕРЕСЕЧЕНИЮ/ЗАЗОРУ (та же мера, что в sensitivity_levels.measure).

Истина — принятые мьюты из ТЕКУЩЕГО decisions.json (человек их уже поправил).
Машина-кандидат — свежая сборка build_decisions с подставленным config. Голова
переглушки на паре «человек↔машина»:
    head = max(0, human_start - machine_start)   # машина начала РАНЬШЕ человека
    tail = max(0, machine_end - human_end)        # машина кончила ПОЗЖЕ человека
Это ровно eval.muted_clean_head/tail, посчитанные по пересечению, а не по id.
Человек, наоборот, РАСШИРИВШИЙ окно (нашёл пропущенный мат вплотную), даёт
head = 0 и уходит в недокрытие — он не должен тянуть допуск вверх, поэтому
`полн.поиск` печатается рядом как страж: страж режет только тишину и мат НЕ
может размьютить, значит полнота поиска обязана держаться при любом допуске.

Запуск (кэши берутся из самой рабочей папки, ничего не пишется):
    python tools/head_sweep.py WORKDIR [WORKDIR ...]
    python tools/head_sweep.py "$HOME/Library/Application Support/Electron/work"/vod_*/
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import sensitivity_levels as sl  # noqa: E402

from twitch_cut.config import PipelineConfig  # noqa: E402
from twitch_cut.decisions import build_decisions  # noqa: E402
from twitch_cut.pipeline import _load_detection_cache  # noqa: E402
from twitch_cut.profanity import ProfanityMatch  # noqa: E402
from twitch_cut.timecode import parse_timecode  # noqa: E402

Span = sl.Span

# Сетка допуска. 0 — клэмп как встарь; 100/140 — реальные значения high; между —
# чтобы увидеть форму кривой, а не две точки. Страж мерится на каждом значении.
CROSS_GRID = (0, 60, 100, 140)


def _config_from_settings(settings: dict) -> PipelineConfig:
    """Восстановить config записи из decisions.json['settings'].

    Фильтруем по полям PipelineConfig: старые записи могут не иметь новых полей
    (тогда берётся дефолт — например, страж), а снятые поля не должны ронять
    сборку. Так кандидат отличается от боевого прогона ТОЛЬКО подставленными
    ручками, всё остальное — как было у записи (уровень, padding, движок)."""
    known = {f.name for f in dataclasses.fields(PipelineConfig)}
    clean = {k: v for k, v in settings.items() if k in known}
    return PipelineConfig(**clean)


def _reconstruct_matches(doc: dict, workdir: Path) -> list[ProfanityMatch]:
    """Матчи, которые видел build_decisions: детекция + спотинг, по времени.

    Ровно как в pipeline._spot_stage: merged = sorted(detection + spotted).
    Пути берём из caches самой записи, а не гадаем — иначе ключ детекции (а с ним
    и набор находок) мог бы разойтись с тем, что записано в decisions.json.
    """
    caches = doc.get("caches") or {}
    matches: list[ProfanityMatch] = []
    det = caches.get("detection")
    if det and Path(det).exists():
        matches.extend(_load_detection_cache(Path(det)))
    spot = caches.get("spotting")
    if spot and Path(spot).exists():
        matches.extend(_load_detection_cache(Path(spot)))
    matches.sort(key=lambda m: m.local_start_ms)
    return matches


def _spans(doc: dict, statuses: tuple[str, ...]) -> list[Span]:
    return sorted(
        (sl._ms(m["start"]), sl._ms(m["end"]), str(m.get("word", "")))
        for m in doc.get("mutes") or []
        if str(m.get("status")) in statuses
    )


def _overmute(human: list[Span], machine: list[Span]) -> dict:
    """Двусторонняя переглушка по парам «человек↔машина» с пересечением.

    Каждое окно человека сопоставляем машинному с НАИБОЛЬШИМ пересечением и
    смотрим, насколько машина вылезла ЗА него слева (голова) и справа (хвост).
    Пары без пересечения сюда не входят вовсе — их цена в полноте, не здесь.
    """
    heads: list[int] = []
    tails: list[int] = []
    for h in human:
        overlapping = [(sl.overlap_ms(h, m), m) for m in machine if sl.overlap_ms(h, m) > 0]
        if not overlapping:
            continue
        _, m = max(overlapping, key=lambda t: t[0])
        heads.append(max(0, h[0] - m[0]))
        tails.append(max(0, m[1] - h[1]))
    charged = [x for x in heads if x > 0]
    return {
        "head_total_s": sum(heads) / 1000.0,
        "head_median_ms": statistics.median(charged) if charged else 0.0,
        "head_windows": len(charged),
        "tail_total_s": sum(tails) / 1000.0,
        "pairs": len(heads),
    }


def _evaluate_candidate(
    doc: dict,
    matches: list[ProfanityMatch],
    human: list[Span],
    *,
    cross_ms: int,
    guard: bool,
) -> dict:
    base = _config_from_settings(doc.get("settings") or {})
    cfg = dataclasses.replace(
        base,
        mute_head_cross_prev_ms=cross_ms,
        mute_head_cross_voiced_guard=guard,
    )
    caches = doc.get("caches") or {}
    audio = caches.get("audio")
    built = build_decisions(
        stream_path=Path(doc.get("source") or "stream.mp4"),
        original_path=Path(doc.get("original") or "original.mp4"),
        range_in=parse_timecode(doc["range_in"]),
        range_out=parse_timecode(doc["range_out"]),
        matches=matches,
        config=cfg,
        audio_cache=Path(audio) if audio and Path(audio).exists() else None,
    )
    machine = _spans(built, ("accepted",))
    review = _spans(built, ("review",))
    row = sl.measure(human, machine, review)
    row.update(_overmute(human, machine))
    return row


def _analyze_workdir(workdir: Path) -> dict | None:
    dec = workdir / "decisions.json"
    if not dec.exists():
        return None
    doc = json.loads(dec.read_text(encoding="utf-8"))
    human = _spans(doc, ("accepted",))
    if not human:
        return None
    matches = _reconstruct_matches(doc, workdir)
    own_cross = int((doc.get("settings") or {}).get("mute_head_cross_prev_ms") or 0)
    rows: dict[tuple[int, bool], dict] = {}
    for cross_ms in sorted(set(CROSS_GRID) | {own_cross}):
        for guard in (False, True):
            rows[(cross_ms, guard)] = _evaluate_candidate(
                doc, matches, human, cross_ms=cross_ms, guard=guard
            )
    return {
        "workdir": workdir.name,
        "human": len(human),
        "own_cross": own_cross,
        "sensitivity": (doc.get("settings") or {}).get("sensitivity"),
        "rows": rows,
    }


def _print_workdir(rep: dict) -> None:
    print(f"\n=== {rep['workdir']}  (уровень {rep['sensitivity']}, "
          f"допуск записи {rep['own_cross']} мс, мьютов человека {rep['human']}) ===")
    print(f"{'допуск':>6} {'страж':>6} {'точн':>5} {'полн':>5} {'п.поиск':>7} "
          f"{'голова,с':>9} {'мед,мс':>7} {'окон':>5} {'хвост,с':>8} {'лишн,с':>7}")
    for (cross_ms, guard), row in sorted(rep["rows"].items()):
        print(f"{cross_ms:6d} {('да' if guard else 'нет'):>6} "
              f"{row['precision']:5.3f} {row['recall']:5.3f} {row['recall_search']:7.3f} "
              f"{row['head_total_s']:9.2f} {row['head_median_ms']:7.0f} "
              f"{row['head_windows']:5d} {row['tail_total_s']:8.2f} {row['extra_s']:7.2f}")


def _print_headline(reports: list[dict]) -> None:
    """Заголовок: страж off→on при СОБСТВЕННОМ допуске записи.

    Это ответ на вопрос пользователя: с тем же уровнем, что стоял, страж должен
    срезать голову переглушки и НЕ уронить полноту поиска (мат он не размьучивает).
    """
    print("\n" + "=" * 78)
    print("СТРАЖ off→on ПРИ СОБСТВЕННОМ ДОПУСКЕ ЗАПИСИ (голова переглушки / полнота поиска)")
    print("=" * 78)
    print(f"{'запись':32} {'голова off→on, с':>18} {'п.поиск off→on':>16}")
    tot_off = tot_on = 0.0
    rs_min = 1.0
    for rep in reports:
        c = rep["own_cross"]
        off = rep["rows"][(c, False)]
        on = rep["rows"][(c, True)]
        tot_off += off["head_total_s"]
        tot_on += on["head_total_s"]
        rs_min = min(rs_min, on["recall_search"] - off["recall_search"])
        print(f"{rep['workdir'][:32]:32} "
              f"{off['head_total_s']:7.2f} → {on['head_total_s']:6.2f}   "
              f"{off['recall_search']:6.3f} → {on['recall_search']:6.3f}")
    print("-" * 78)
    print(f"{'ИТОГО голова переглушки':32} {tot_off:7.2f} → {tot_on:6.2f} с   "
          f"(снято {tot_off - tot_on:.2f} с)")
    print(f"худшая просадка полноты поиска от стража: {rs_min:+.3f} "
          f"(≥0 — страж мат не размьютил ни на одной записи)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workdirs", nargs="+", type=Path, help="рабочие папки записей")
    args = parser.parse_args()

    reports: list[dict] = []
    for wd in args.workdirs:
        if not wd.is_dir():
            continue
        try:
            rep = _analyze_workdir(wd)
        except Exception as exc:  # noqa: BLE001 — одна битая папка не должна ронять свип
            print(f"[!] {wd.name}: {type(exc).__name__}: {exc}", file=sys.stderr)
            continue
        if rep is None:
            continue
        reports.append(rep)
        _print_workdir(rep)

    if not reports:
        parser.error("не найдено ни одной папки с decisions.json и принятыми мьютами")
    _print_headline(reports)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
