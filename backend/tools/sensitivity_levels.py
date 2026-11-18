"""Замер уровней чувствительности: прогон машины против таймлайна человека.

Зачем. `SENSITIVITY_LEVELS` в config.py — три набора чисел, и единственный способ
не выдумывать их «на глаз» — посчитать, что каждый уровень даёт на реальном
прогоне, где мьюты человека приняты за истину. Таблица в комментарии к
`SENSITIVITY_LEVELS` получена именно этим скриптом; правишь уровни — пересчитай.

Чем отличается от `twitch-cut eval`. eval читает ЖУРНАЛ ПРАВОК внутри одной
рабочей папки (что человек поменял в нашем выводе) и потому меряет один прогон.
Здесь сравниваются РАЗНЫЕ прогоны с одним и тем же человеческим таймлайном из
другой папки — иначе уровни между собой не сопоставить.

Что считается и почему не только F1:
* точность/полнота/F1 по ПЕРЕКРЫТИЮ окон — сопоставимо с прежними замерами, но
  к ширине окна почти нечувствительно (перекрытие в 1 мс уже «попадание»);
* точность/полнота ПОИСКА по ЗАЗОРУ (`audit.MATCH_TOLERANCE_MS`) — отвечают на
  другой вопрос: нашла ли машина слово, пусть и не тем окном. Правило по
  пересечению на окнах шириной 130–470 мс жёстче объявленного допуска ±250 мс:
  расхождение границ на 150 мс уничтожает пересечение целиком, и одно слово
  становится сразу пропуском (у человека) и ложным (у машины) — двойная плата за
  ошибку границы. Замер, из которого взят допуск, записан у константы;
* НАКРЫТИЕ — сколько мьютов человека заглушены целиком и какова медиана
  незаглушенного остатка: слышимый хвост мата виден здесь и больше нигде;
* ЛИШНЕЕ — сумма машинного мьюта вне окон человека, то есть цена в чистой речи.
Асимметрия цены ошибки (слышимый мат хуже лишних 100 мс) — выбор человека, и
осмысленным он становится только когда обе стороны цены посчитаны.

Каждый мьют человека попадает РОВНО В ОДИН класс, и классы печатаются числами:

* `накрыт`  — есть пересечение с принятым мьютом машины;
* `рядом`   — пересечения нет, но принятый мьют машины в допуске: слово найдено,
              окно встало не там (цена — в медианах границ, не в полноте);
* `review`  — накрыт только кандидатом «на проверку»: поиск слово нашёл, но в
              Vegas оно не уехало. Это потеря ПОСЛЕ детекции, отдельный класс;
* `пропуск` — в допуске нет ничего: слова машина не нашла.

Класса «политика» (грубость из `SPICY_ROOTS`, которую детектор не мьютит
осознанно) здесь нет: для него нужен транскрипт, и считает его `twitch-cut eval`
по `audit.ADDED_CHARGE`. На сверке двух прогонов такой мьют попадёт в «пропуск» —
помните об этом, когда сравниваете число отсюда с числом оттуда.

`--breakdown` отвечает на отдельный вопрос: можно ли полосу «на проверку»
сделать порогом по уверенности вместо категорий match_type. Печатает точность
по каждой категории и то, что порог по confidence отрезал бы. На этом корпусе
ответ отрицательный — см. ProfanityMatch.needs_review.

Запуск (прогоны делаются заранее, транскрипт и посториоры берутся из кэша):

    for L in low balanced high; do
      twitch-cut process --stream STREAM --workdir WORK \\
        --range-in 00:00:00 --range-out 00:15:00 --sensitivity $L \\
        --decisions WORK/dec_$L.json --vegas WORK/vegas_$L.cs
    done
    python tools/sensitivity_levels.py --human HUMAN/decisions.json \\
        --run low=WORK/dec_low.json --run balanced=WORK/dec_balanced.json \\
        --run high=WORK/dec_high.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twitch_cut.audit import MATCH_TOLERANCE_MS  # noqa: E402

Span = tuple[int, int, str]


def _ms(value: object) -> int:
    return int(round(float(value) * 1000))  # type: ignore[arg-type]


def load_mutes(path: Path, statuses: tuple[str, ...] = ("accepted",)) -> list[Span]:
    """Мьюты из decisions.json в миллисекундах, отсортированные по времени.

    По умолчанию только `accepted`: `review`-кандидаты в Vegas не экспортируются,
    значит и в точность/полноту им нельзя — они считаются отдельно.
    """
    doc = json.loads(path.read_text(encoding="utf-8"))
    return sorted(
        (_ms(m["start"]), _ms(m["end"]), str(m.get("word", "")))
        for m in doc.get("mutes") or []
        if str(m.get("status")) in statuses
    )


def load_records(path: Path) -> list[dict]:
    """Мьюты как есть: разбивке нужны `match_type` и `confidence`, а их нет в Span."""
    doc = json.loads(path.read_text(encoding="utf-8"))
    return list(doc.get("mutes") or [])


def overlap_ms(a: Span, b: Span) -> int:
    return max(0, min(a[1], b[1]) - max(a[0], b[0]))


def gap_ms(a: Span, b: Span) -> int:
    """Зазор между окнами: 0 — пересекаются или соприкасаются."""
    return max(0, max(a[0], b[0]) - min(a[1], b[1]))


def classify(human: Span, machine: list[Span], review: list[Span], tolerance: int) -> str:
    """Класс одного мьюта человека: накрыт / рядом / review / пропуск.

    Порядок проверок — от самого сильного свидетельства к самому слабому, классы
    взаимоисключающие: сумма четырёх счётчиков всегда равна числу мьютов человека,
    иначе разбивка врала бы про своё же основание.
    """
    if any(overlap_ms(human, m) > 0 for m in machine):
        return "covered"
    if any(gap_ms(human, m) <= tolerance for m in machine):
        return "near"
    if any(gap_ms(human, r) <= tolerance for r in review):
        return "review"
    return "missed"


def union_ms(spans: list[tuple[int, int]]) -> int:
    """Длина объединения отрезков: два мьюта на одном слове не считаются дважды."""
    total = 0
    cur_start = cur_end = None
    for start, end in sorted(spans):
        if cur_end is None:
            cur_start, cur_end = start, end
        elif start <= cur_end:
            cur_end = max(cur_end, end)
        else:
            total += cur_end - cur_start  # type: ignore[operator]
            cur_start, cur_end = start, end
    if cur_end is not None:
        total += cur_end - cur_start  # type: ignore[operator]
    return total


def covered_ms(window: Span, machine: list[Span]) -> int:
    """Сколько миллисекунд окна человека машина реально заглушила."""
    return union_ms(
        [
            (max(window[0], m[0]), min(window[1], m[1]))
            for m in machine
            if overlap_ms(window, m) > 0
        ]
    )


def measure(
    human: list[Span],
    machine: list[Span],
    review: list[Span],
    tolerance: int = MATCH_TOLERANCE_MS,
) -> dict[str, float]:
    hits = [m for m in machine if any(overlap_ms(m, h) > 0 for h in human)]
    # Попадания «по допуску»: пересечения нет, но окно человека рядом. Считаются
    # отдельным числом, чтобы послабление правила было ВИДНО, а не растворилось
    # в точности.
    near_hits = [
        m
        for m in machine
        if not any(overlap_ms(m, h) > 0 for h in human)
        and any(gap_ms(m, h) <= tolerance for h in human)
    ]
    classes = [classify(h, machine, review, tolerance) for h in human]
    count = lambda key: sum(1 for c in classes if c == key)  # noqa: E731

    precision = len(hits) / len(machine) if machine else 0.0
    # Полнота — по мьютам ЧЕЛОВЕКА: два машинных мьюта на одном его слове не
    # должны давать полноту больше правды.
    recall = count("covered") / len(human) if human else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    # Поисковые числа: «нашла слово» вместо «накрыла окно». Пропуском остаётся
    # только то, чего в допуске нет вовсе.
    precision_search = (len(hits) + len(near_hits)) / len(machine) if machine else 0.0
    recall_search = (len(human) - count("missed")) / len(human) if human else 0.0

    gaps = [(h[1] - h[0]) - covered_ms(h, machine) for h in human]
    inside = sum(covered_ms(h, machine) for h in human)
    extra_ms = union_ms([(s, e) for s, e, _ in machine]) - inside
    return {
        "mutes": len(machine),
        "hits": len(hits),
        "false": len(machine) - len(hits),
        # «проп» первой таблицы — по-прежнему СТРОГО по пересечению, иначе старые
        # замеры перестали бы сравниваться с новыми, никого не предупредив.
        "missed": len(human) - count("covered"),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "near_hits": len(near_hits),
        "precision_search": precision_search,
        "recall_search": recall_search,
        "covered": count("covered"),
        "near": count("near"),
        "review_only": count("review"),
        "missed_search": count("missed"),
        "fully_covered": sum(1 for g in gaps if g <= 0),
        "gap_median_ms": statistics.median(gaps) if gaps else 0.0,
        "extra_s": extra_ms / 1000.0,
        "review": len(review),
        "review_hits": sum(1 for r in review if any(overlap_ms(r, h) > 0 for h in human)),
    }


def print_breakdown(human: list[Span], records: list[dict]) -> None:
    """Разбивка находок по КАТЕГОРИИ матча и проверка порога по confidence.

    Отвечает на вопрос «а нельзя ли вместо категорий (`needs_review` по
    match_type) поставить один порог по уверенности?». Ответ виден в двух
    последних блоках: у категорий точность расходится (обрывки и посториоры
    против surface/lemma/stem/glued), а у confidence — нет.
    """

    def hit(rec: dict) -> bool:
        span = (_ms(rec["start"]), _ms(rec["end"]), "")
        return any(overlap_ms(span, h) > 0 for h in human)

    groups: dict[tuple[str, str], list[dict]] = {}
    for rec in records:
        key = (str(rec.get("match_type") or "?"), str(rec.get("status")))
        groups.setdefault(key, []).append(rec)

    print(f"\n{'match_type':16} {'статус':9} {'n':>4} {'попали':>6} {'точн':>6}")
    for key in sorted(groups, key=lambda k: -len(groups[k])):
        rows = groups[key]
        hits = sum(1 for r in rows if hit(r))
        print(f"{key[0]:16} {key[1]:9} {len(rows):4} {hits:6} {hits / len(rows):6.2f}")

    accepted = [r for r in records if str(r.get("status")) == "accepted"]
    good = [r["confidence"] for r in accepted if hit(r) and r.get("confidence") is not None]
    bad = [r["confidence"] for r in accepted if not hit(r) and r.get("confidence") is not None]
    if not good or not bad:
        return
    print(
        f"\nconfidence попавших: n={len(good)} медиана={statistics.median(good):.3f} "
        f"мин={min(good):.3f}\n"
        f"confidence лишних  : n={len(bad)} медиана={statistics.median(bad):.3f} "
        f"мин={min(bad):.3f}"
    )
    for threshold in (0.5, 0.6, 0.7, 0.8, 0.9):
        print(
            f"  порог {threshold:.1f}: уберёт лишних "
            f"{sum(1 for c in bad if c < threshold)}/{len(bad)}, "
            f"настоящих {sum(1 for c in good if c < threshold)}/{len(good)}"
        )
    print(
        "Если второе число растёт раньше первого — порога по confidence быть не "
        "может, полоса review остаётся категорией (см. ProfanityMatch.needs_review)"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--human", required=True, type=Path, help="decisions.json человека")
    parser.add_argument(
        "--run",
        action="append",
        default=[],
        metavar="LABEL=PATH",
        help="Прогон машины: имя уровня и путь к его decisions.json",
    )
    parser.add_argument(
        "--tolerance-ms",
        type=int,
        default=MATCH_TOLERANCE_MS,
        help=f"допуск сопоставления по зазору окон (по умолчанию {MATCH_TOLERANCE_MS})",
    )
    parser.add_argument(
        "--breakdown",
        action="store_true",
        help="Дополнительно: разбивка по match_type и проверка порога по confidence",
    )
    args = parser.parse_args()

    human = load_mutes(args.human)
    if not human:
        parser.error(f"в {args.human} нет мьютов со статусом accepted")

    print(
        f"{'уровень':10} {'мьютов':>6} {'попали':>6} {'лишние':>6} {'проп':>5} "
        f"{'точн':>5} {'полн':>5} {'F1':>5} {'накрыто':>9} {'дырка':>7} "
        f"{'лишнего':>8} {'review':>8}"
    )
    parsed: list[tuple[str, Path]] = []
    for item in args.run:
        label, _, raw_path = item.partition("=")
        if not raw_path:
            parser.error(f"--run ожидает LABEL=PATH, получено {item!r}")
        parsed.append((label, Path(raw_path)))
    rows: list[tuple[str, dict[str, float]]] = []
    for label, path in parsed:
        row = measure(
            human,
            load_mutes(path),
            load_mutes(path, statuses=("review",)),
            tolerance=args.tolerance_ms,
        )
        rows.append((label, row))
        print(
            f"{label:10} {row['mutes']:6.0f} {row['hits']:6.0f} {row['false']:6.0f} "
            f"{row['missed']:5.0f} {row['precision']:5.3f} {row['recall']:5.3f} "
            f"{row['f1']:5.3f} {row['fully_covered']:4.0f}/{len(human):<4d} "
            f"{row['gap_median_ms']:5.0f}мс {row['extra_s']:7.1f}с "
            f"{row['review']:4.0f}/{row['review_hits']:<3.0f}"
        )

    print(f"\nПо ЗАЗОРУ, допуск ±{args.tolerance_ms} мс (нашла ли машина слово):")
    print(
        f"{'уровень':10} {'точн':>5} {'полн':>5} {'накрыт':>7} {'рядом':>6} "
        f"{'review':>7} {'пропуск':>8}"
    )
    for label, row in rows:
        print(
            f"{label:10} {row['precision_search']:5.3f} {row['recall_search']:5.3f} "
            f"{row['covered']:7.0f} {row['near']:6.0f} {row['review_only']:7.0f} "
            f"{row['missed_search']:8.0f}"
        )
    print(
        "накрыт — есть пересечение; рядом — слово найдено, окно встало не там "
        "(цена в границах, не в полноте); review — нашёл, но отправил на "
        "проверку, в Vegas не уехало; пропуск — в допуске нет ничего.\n"
        "Классы взаимоисключающие, их сумма = мьютов человека. Грубость из "
        "SPICY_ROOTS попадёт в «пропуск»: её отделяет только twitch-cut eval, "
        "которому доступен транскрипт."
    )
    print(
        f"\nмьютов человека: {len(human)}, суммарно "
        f"{union_ms([(s, e) for s, e, _ in human]) / 1000:.1f} с"
    )
    print(
        "накрыто — окон человека заглушено целиком; дырка — медиана остатка "
        "внутри окна; лишнего — мьют вне окон человека; review — кандидаты "
        "спотинга / из них по делу"
    )
    if args.breakdown:
        for label, path in parsed:
            print(f"\n=== {label} ===")
            print_breakdown(human, load_records(path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
