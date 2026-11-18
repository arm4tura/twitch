"""evaluate.py — правки человека как МЕТРИКА пайплайна.

Зачем. Про качество детекции можно спорить словами бесконечно, а сравнить два
прогона — нельзя: «стало лучше» никто не измерял. Но человек уже даёт разметку,
причём бесплатно: каждый раз, когда он правит таймлайн, он говорит «должно быть
вот так». Журнал правок (`<decisions>.edits.json`, см. edit_journal.py) хранит
машинный снимок ДО правок, поэтому разница «машина → человек» и есть метрика.

Что считаем. Мьюты машины сверяются с принятым человеком таймлайном:

* снял мьют      → ЛОЖНОЕ срабатывание (точность);
* добавил мьют   → зависит от ПРИЧИНЫ, см. ниже;
* сдвинул границы→ мат нашли, но окно не то (|Δначало|, |Δконец| в мс);
* не тронул      → машина угадала полностью.

Отдельный случай — ПЕРЕНОС: человек перетащил мьют так, что общего звука со
старым окном не осталось (журнал сопоставляет по `id`, поэтому приходит как
«сдвиг»). Это не ошибка границы, а «заглушили не то» + «нашли пропуск», и
считается именно так (см. `audit._relocated`): иначе одна такая правка вносит
в статистику границ полсекунды и врёт про системный сдвиг.

Из этого выводятся precision / recall / F1 и распределение ошибки границ. Плюс
разбивка пропусков по причинам (`audit.classify_added`) — та же, что печатает
`audit.md`, ключами из одного места, чтобы число и текст не разъезжались.

Добавленный мьют — не автоматически пропуск. `audit.ADDED_CHARGE` делит их на
три класса, и полнота поэтому считается ДВАЖДЫ: это два разных вопроса.

* `полнота (строгая)` — сколько мьютов человека машина накрыла своими. Отвечает
  «сколько работы осталось человеку» и сравнима со всеми прежними замерами.
* `полнота поиска` — сколько матерных слов детектор НАШЁЛ. Класс `misplaced`
  (машина нашла слово, но окно встало рядом: либо на тишине — уехал тайминг,
  либо человек его раздвинул) в пропуски не идёт — это ошибка границ, и она уже
  посчитана в медианах `d_start`/`d_end`, второй раз платить нельзя. Класс
  `policy` (грубость из `SPICY_ROOTS`, которую детектор не мьютит осознанно)
  уходит из полноты целиком: это разница политики, а не ошибка.

Чего второе число НЕ делает: оно не прощает ограничения модели. «ASR услышал не
то» и «речь не распознана» остаются пропуском. Если человек слышит мат, а машина
нет — это провал полноты, даже когда слово сказано тихо, скороговоркой или
перекрыто чужой речью; убрать такие случаи из полноты значит завести метрику,
которая растёт от того, что модель стала слышать хуже. Их отдельный счёт
(`missed_by_diagnosis`) для этого и печатается — чтобы ограничение модели было
видно числом, а не тонуло в общем F1.

Чего метрика НЕ значит:

* Она измеряет пайплайн ТОЛЬКО если baseline снят обработкой (`source:
  pipeline`). Если журнал завёлся с первого сохранения (`first_save`), часть
  правок уже внутри снимка, и цифры оптимистичнее правды — это печатается
  явным предупреждением, а не прячется.
* «Сдвинул границы» — не всегда вина машины: человек мог подровнять окно на
  свой вкус. Поэтому границы отчёт даёт распределением (медиана/p90), а не
  средним, и отдельно СО ЗНАКОМ — знак показывает системный сдвиг, а вкус
  систематическим не бывает.
* Слои тишины и оффтопа не оцениваются: там нет «правильного ответа», по
  которому можно считать точность. Их правки печатаются счётчиками.

Запуск: `twitch-cut eval <workdir>` — печатает сводку;
`--json out.json` — машинный формат, чтобы сравнить с прогоном после правок
пайплайна (`--baseline prev.json` покажет дельту).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
import statistics
from typing import Any, Iterable, Sequence

from .audit import (
    ADDED_CHARGE,
    ADDED_LABELS,
    CHARGE_LABELS,
    REMOVED_LABELS,
    AuditError,
    AuditResult,
    EditFinding,
    analyze,
    classify_added,
    classify_removed,
)
from .edit_journal import baseline_source, count_changes, net_changes

SCHEMA_VERSION = "1.2"

MOVED_TOUCH_MS = 30
"""Сдвиг меньше 30 мс не считаем правкой границ: это округление и дрожь мыши,
а не «машина ошиблась». Журнал пишет всё, метрика — только осмысленное."""


@dataclass
class Spread:
    """Распределение одной величины в мс. Медиана и p90 вместо среднего: пара
    правок на секунду не должна выглядеть как общая беда."""

    n: int = 0
    median: int = 0
    p90: int = 0
    worst: int = 0
    signed_median: int = 0
    """Медиана СО ЗНАКОМ: систематический сдвиг машины (минус — опаздывает)."""
    total: int = 0
    """Сумма ВЕЛИЧИН (по модулю). Для утечки/переглушки — сколько всего секунд
    набежало: медиана говорит «систематически ли», total — «сколько итого»."""

    @classmethod
    def of(cls, values: Sequence[int]) -> "Spread":
        if not values:
            return cls()
        magnitudes = sorted(abs(v) for v in values)
        idx = min(len(magnitudes) - 1, int(round(0.9 * (len(magnitudes) - 1))))
        return cls(
            n=len(values),
            median=int(round(statistics.median(magnitudes))),
            p90=magnitudes[idx],
            worst=magnitudes[-1],
            signed_median=int(round(statistics.median(values))),
            total=sum(magnitudes),
        )


@dataclass
class EvalResult:
    decisions_path: Path
    baseline_source: str
    trustworthy: bool
    """False — baseline снят не обработкой, метрика оптимистичнее правды."""

    machine: int = 0
    human: int = 0
    kept: int = 0
    untouched: int = 0
    added: int = 0
    removed: int = 0
    moved: int = 0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    missed: int = 0
    """Добавленные мьюты, где слово НЕ НАЙДЕНО (`audit.ADDED_CHARGE == missed`)."""
    misplaced: int = 0
    """Добавленные мьюты, где слово найдено, но окно машины встало рядом."""
    policy: int = 0
    """Добавленные мьюты на грубости, которую детектор не мьютит осознанно."""
    recall_search: float = 0.0
    """Полнота ПОИСКА: `misplaced` не пропуск, `policy` вне метрики вообще.

    Второго F1 рядом нет намеренно: точность считается по-прежнему строго, и
    гармоническое среднее строгой точности с нестрогой полнотой читалось бы как
    самостоятельная правда, хотя это смесь двух правил. Сравнивать прогоны между
    собой достаточно парой (`precision`, `recall_search`) плюс счётчиками
    классов."""
    d_start: Spread = field(default_factory=Spread)
    d_end: Spread = field(default_factory=Spread)
    d_duration: Spread = field(default_factory=Spread)
    muted_clean_head: Spread = field(default_factory=Spread)
    """Сколько ЧИСТОЙ РЕЧИ машина заглушила В ГОЛОВЕ окна — человек подрезал
    начало (d_start > 0). d_start/d_duration это прячут: первый мешает правку и
    ошибку под одним знаком, вторая — нетто по длине (голова минус хвост), и
    точная правка «подрезал голову, отпустил хвост» гасится в ноль. Здесь только
    сужение головы, односторонне: расширение — это НЕДОКРЫТИЕ, оно уже в d_start
    со знаком, второй раз платить нельзя. Это и есть та переглушка, из-за которой
    человек правит окна каждый раз (см. mute_head_cross_prev_ms в config.py)."""
    muted_clean_tail: Spread = field(default_factory=Spread)
    """То же для ХВОСТА окна: человек подрезал конец (d_end < 0)."""
    added_by_diagnosis: dict[str, int] = field(default_factory=dict)
    added_by_charge: dict[str, int] = field(default_factory=dict)
    missed_by_diagnosis: dict[str, int] = field(default_factory=dict)
    """Только настоящие пропуски, по причинам: ограничение модели числом."""
    removed_by_diagnosis: dict[str, int] = field(default_factory=dict)
    other_layers: dict[str, dict[str, int]] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    @property
    def boundary_ok(self) -> int:
        """Мьюты, где мат найден И границы человек не трогал."""
        return self.untouched

    def to_json(self) -> dict[str, Any]:
        data = asdict(self)
        data["decisions_path"] = str(self.decisions_path)
        data["schema_version"] = SCHEMA_VERSION
        return data


def evaluate(
    decisions_path: Path,
    *,
    edits_path: Path | None = None,
    audit_result: AuditResult | None = None,
) -> EvalResult:
    """Посчитать метрику по журналу правок.

    Разбор берём у `audit.analyze`: там уже прочитаны транскрипт, WAV и журнал,
    а каждая правка получила слова и корни. Считать это второй раз — значит
    завести второй источник истины и дать ему разъехаться с отчётом.
    """
    result = audit_result or analyze(decisions_path, edits_path=edits_path)
    if result.journal is None:
        raise AuditError(
            "журнала правок нет — метрику считать не из чего. Включите «Журнал "
            "правок» в настройках и переровняйте таймлайн: сравнение идёт с "
            f"машинным снимком (ожидался файл {result.edits_path or '<decisions>.edits.json'})"
        )

    journal = result.journal
    source = baseline_source(journal)
    baseline = journal.get("baseline") if isinstance(journal.get("baseline"), dict) else {}
    machine = len(baseline.get("mutes") or [])
    human = len([m for m in result.doc.get("mutes") or [] if isinstance(m, dict)])

    by_kind: dict[str, list[EditFinding]] = {}
    for finding in result.edits:
        by_kind.setdefault(finding.kind, []).append(finding)

    added = by_kind.get("added", [])
    removed = by_kind.get("removed", []) + by_kind.get("off", [])
    enabled = by_kind.get("on", [])
    moved = [
        f
        for f in by_kind.get("moved", [])
        if abs(f.d_start_ms) >= MOVED_TOUCH_MS or abs(f.d_end_ms) >= MOVED_TOUCH_MS
    ]

    kept = max(0, machine - len(removed))
    untouched = max(0, kept - len(moved) - len(enabled))
    precision = kept / machine if machine else 0.0
    recall = kept / (kept + len(added)) if (kept + len(added)) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    # Добавленные мьюты — три разных класса под одним словом «пропуск»
    # (audit.ADDED_CHARGE). Классифицируем КАЖДЫЙ по одному разу: и разбивка по
    # причинам, и полнота поиска обязаны считаться с одного и того же вердикта,
    # иначе текст отчёта и число метрики разъедутся на первой же правке причин.
    # Зазор до мьюта машины и озвученность ТОГО окна уже посчитаны в audit при
    # разборе журнала — здесь их только передаём (см. audit._verdict_added).
    diagnosed = [
        (
            key,
            ADDED_CHARGE.get(key, "missed"),
        )
        for key in (
            classify_added(
                f.words, f.roots, f.known_stem, f.neighbour_gap_ms, f.neighbour_voiced
            )
            for f in added
        )
    ]
    by_charge: dict[str, list[str]] = {}
    for key, charge in diagnosed:
        by_charge.setdefault(charge, []).append(key)
    missed = by_charge.get("missed", [])
    misplaced = by_charge.get("misplaced", [])
    policy = by_charge.get("policy", [])

    # Слово, найденное с уехавшим окном, — находка, а не пропуск: цена этой
    # ошибки уже взята медианами d_start/d_end, и второй раз брать её полнотой
    # значит платить дважды за одну правку.
    found = kept + len(misplaced)
    recall_search = found / (found + len(missed)) if (found + len(missed)) else 0.0

    other: dict[str, dict[str, int]] = {}
    net = net_changes(journal, result.doc)
    for layer in ("silences", "offtopic"):
        changes = net.get(layer)
        if isinstance(changes, dict):
            counts = count_changes({layer: changes})
            if any(counts.values()):
                other[layer] = counts

    # Насколько человек РАСШИРИЛ окно: плюс — машина накрывала мало.
    durations = [
        (f.end_ms - f.start_ms) - (f.base_end_ms - f.base_start_ms)
        for f in moved
        if f.base_end_ms is not None and f.base_start_ms is not None
    ]

    # Двусторонняя цена: сколько ЧИСТОЙ РЕЧИ человек ВЫРЕЗАЛ из окна машины.
    # durations (нетто) это гасит — правка «голову подрезал, хвост отпустил»
    # выходит нулевой по длине. Берём каждую сторону односторонне, только
    # СУЖЕНИЕ (max(0, ...)): расширение — недокрытие, оно уже в d_start/d_end.
    # d_start > 0 — начало сдвинули позже (срезали голову); d_end < 0 — конец
    # сдвинули раньше (срезали хвост), см. audit.EditFinding.d_*_ms.
    muted_clean_head = [max(0, f.d_start_ms) for f in moved if f.base_start_ms is not None]
    muted_clean_tail = [max(0, -f.d_end_ms) for f in moved if f.base_end_ms is not None]

    out = EvalResult(
        decisions_path=result.decisions_path,
        baseline_source=source,
        trustworthy=source == "pipeline",
        machine=machine,
        human=human,
        kept=kept,
        untouched=untouched,
        added=len(added),
        removed=len(removed),
        moved=len(moved),
        precision=round(precision, 4),
        recall=round(recall, 4),
        f1=round(f1, 4),
        missed=len(missed),
        misplaced=len(misplaced),
        policy=len(policy),
        recall_search=round(recall_search, 4),
        d_start=Spread.of([f.d_start_ms for f in moved]),
        d_end=Spread.of([f.d_end_ms for f in moved]),
        d_duration=Spread.of(durations),
        muted_clean_head=Spread.of(muted_clean_head),
        muted_clean_tail=Spread.of(muted_clean_tail),
        added_by_diagnosis=_tally(key for key, _charge in diagnosed),
        added_by_charge=_tally(charge for _key, charge in diagnosed),
        missed_by_diagnosis=_tally(missed),
        removed_by_diagnosis=_count_removed(removed),
        other_layers=other,
        notes=_notes(source, result, enabled),
    )
    return out


def _tally(keys: Iterable[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for key in keys:
        counts[key] = counts.get(key, 0) + 1
    return counts


def _count_removed(findings: Sequence[EditFinding]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for f in findings:
        # f.voiced — озвученность окна МАШИНЫ (снятый мьют), по ней отделяется
        # «снято осознанно» от «мьют стоял на тишине» (см. audit.classify_removed).
        key = classify_removed(f.words, f.roots, f.voiced)
        counts[key] = counts.get(key, 0) + 1
    return counts


def _notes(source: str, result: AuditResult, enabled: Sequence[EditFinding]) -> list[str]:
    notes: list[str] = []
    if source != "pipeline":
        notes.append(
            f"baseline снят НЕ обработкой (source={source}): часть ручных правок "
            "уже внутри снимка, поэтому точность и полнота ЗАВЫШЕНЫ. Чтобы мерить "
            "честно, включите журнал ДО обработки и прогоните process заново"
        )
    if not result.audio_ok:
        notes.append("WAV не прочитан — границы считаны только по таймингам ASR")
    if enabled:
        notes.append(
            f"{len(enabled)} мьютов человек включил обратно: детектор их нашёл, но "
            "сам не применил (статус review/rejected) — это потери после детекции, "
            "а не пропуск поиска"
        )
    return notes


# --- вывод -------------------------------------------------------------------


def render_metrics(result: EvalResult, baseline: dict[str, Any] | None = None) -> str:
    """Текстовая сводка. `baseline` — JSON прошлого замера для дельты."""
    lines: list[str] = []
    pct = lambda v: f"{v * 100:.1f}%"  # noqa: E731

    lines.append(f"Метрика по правкам: {result.decisions_path}")
    lines.append(
        f"  мьютов у машины: {result.machine}, у человека после правок: {result.human}"
    )
    lines.append(
        f"  точность (не ложные): {pct(result.precision)}"
        f"{_delta(baseline, 'precision', result.precision, as_pct=True)}"
        f"   — снято человеком: {result.removed}"
    )
    lines.append(
        f"  полнота, строгая (не пропустил): {pct(result.recall)}"
        f"{_delta(baseline, 'recall', result.recall, as_pct=True)}"
        f"   — добавлено человеком: {result.added}"
    )
    lines.append(f"  F1: {pct(result.f1)}{_delta(baseline, 'f1', result.f1, as_pct=True)}")
    lines.append(
        f"  полнота ПОИСКА (нашёл слово): {pct(result.recall_search)}"
        f"{_delta(baseline, 'recall_search', result.recall_search, as_pct=True)}"
        f"   — не найдено: {result.missed}"
    )
    if result.added_by_charge:
        lines.append("    из добавленного человеком:")
        for key in ("missed", "misplaced", "policy"):
            count = result.added_by_charge.get(key, 0)
            if count:
                lines.append(f"      {count:3d}  {CHARGE_LABELS.get(key, key)}")
    lines.append(
        f"  окно угадано без правок: {result.untouched}/{result.machine}"
        f" ({pct(result.untouched / result.machine if result.machine else 0.0)}),"
        f" границы правил: {result.moved}"
    )

    if result.moved:
        lines.append("  ошибка границ, мс (медиана / p90 / худшая):")
        for title, spread, key in (
            ("начало", result.d_start, "d_start"),
            ("конец", result.d_end, "d_end"),
            ("длина", result.d_duration, "d_duration"),
        ):
            sign = "+" if spread.signed_median > 0 else ""
            lines.append(
                f"    {title:7s} {spread.median:5d} / {spread.p90:5d} / {spread.worst:5d}"
                f"   сдвиг со знаком: {sign}{spread.signed_median}"
                f"{_delta(baseline, key, spread.median, nested='median')}"
            )

        # Переглушка: сколько ЧИСТОЙ речи человек вырезал из окна. Печатаем ВСЕГДА
        # при наличии правок границ (ноль — тоже ответ: «чистого не резали»), иначе
        # сравнение до/после теряет главный сигнал этой правки. total — сумма по
        # модулю (сколько итого набежало), median — систематична ли переглушка.
        lines.append("  заглушено ЧИСТОЙ речи, мс (медиана / всего) — переглушка окна:")
        for title, spread, key in (
            ("голова", result.muted_clean_head, "muted_clean_head"),
            ("хвост", result.muted_clean_tail, "muted_clean_tail"),
        ):
            lines.append(
                f"    {title:7s} {spread.median:5d} / {spread.total:6d}"
                f"{_delta(baseline, key, spread.total, nested='total')}"
            )

    if result.added_by_diagnosis:
        lines.append("  почему человек добавлял мьюты:")
        for key, count in sorted(result.added_by_diagnosis.items(), key=lambda kv: -kv[1]):
            charged = " ← пропуск поиска" if ADDED_CHARGE.get(key) == "missed" else ""
            lines.append(f"    {count:3d}  {ADDED_LABELS.get(key, key)}{charged}")
    if result.removed_by_diagnosis:
        lines.append("  почему снимал:")
        for key, count in sorted(result.removed_by_diagnosis.items(), key=lambda kv: -kv[1]):
            lines.append(f"    {count:3d}  {REMOVED_LABELS.get(key, key)}")

    for layer, counts in result.other_layers.items():
        shown = ", ".join(f"{k}={v}" for k, v in counts.items() if v)
        lines.append(f"  слой {layer} (в метрику не входит): {shown}")

    for note in result.notes:
        lines.append(f"  ! {note}")
    return "\n".join(lines)


def _delta(
    baseline: dict[str, Any] | None,
    key: str,
    value: float,
    *,
    as_pct: bool = False,
    nested: str | None = None,
) -> str:
    """`(+2.1 п.п.)` или `(-40 мс)` — насколько лучше прошлого замера."""
    if not baseline:
        return ""
    prev = baseline.get(key)
    if nested and isinstance(prev, dict):
        prev = prev.get(nested)
    if not isinstance(prev, (int, float)) or isinstance(prev, bool):
        return ""
    diff = value - prev
    if as_pct:
        if abs(diff) < 0.0005:
            return " (без изменений)"
        return f" ({diff * 100:+.1f} п.п.)"
    if diff == 0:
        return " (без изменений)"
    return f" ({diff:+.0f} мс)"


def read_metrics(path: Path) -> dict[str, Any] | None:
    """Прошлый замер для сравнения. Битый или отсутствующий — None."""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def write_metrics(result: EvalResult, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(result.to_json(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return path
