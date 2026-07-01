import { useEffect, useState } from "react";
import { Trash2, X } from "lucide-react";
import { Button } from "../ui/Button";
import { Card, CardHeader } from "../ui/Card";
import { Input } from "../ui/Input";
import { fmtMsExact, parseTimecode } from "../../lib/format";
import { cn } from "../../lib/cn";
import type { RegionKind, WFRegion } from "./Waveform";

/**
 * RegionEditor — правая панель редактора одного региона.
 *
 * Панель, а не Sheet: у нас fixed-layout таймлайна (волна + канвасы), боковой
 * выезд перекроет их и введёт лишнюю анимацию. Панель просто занимает 320px
 * справа, когда есть выбранный регион, и коллапсируется когда нет.
 *
 * Форма — контролируемая, но с локальным state'ом: пользователь может вбить
 * невалидный таймкод по буквам, мы не бросаем каждый keystroke в undo-стек.
 * Commit по blur или ⏎.
 *
 * Валидация: start < end, оба >= 0, оба ≤ duration. При невалидности — красный
 * ring, кнопка «Применить» disabled.
 */

export interface EditorRegionInput extends WFRegion {
  reason?: string;
  score?: number;
  words?: string[];
}

export interface RegionEditorProps {
  region: EditorRegionInput | null;
  durationMs: number;
  onCommit: (patch: {
    id: string;
    start: number;
    end: number;
    reason?: string;
    score?: number;
  }) => void;
  onDelete: (id: string) => void;
  onClose: () => void;
  className?: string;
}

const KIND_LABEL: Record<RegionKind, string> = {
  mute: "Мут",
  cut: "Вырез",
  highlight: "Хайлайт",
};

const KIND_DOT: Record<RegionKind, string> = {
  mute: "bg-err",
  cut: "bg-warn",
  highlight: "bg-brand-from",
};

export function RegionEditor({
  region,
  durationMs,
  onCommit,
  onDelete,
  onClose,
  className,
}: RegionEditorProps) {
  const [start, setStart] = useState("");
  const [end, setEnd] = useState("");
  const [reason, setReason] = useState("");
  const [score, setScore] = useState("");

  // Сброс формы при смене региона. Игнорируем повторные обновления того же id,
  // чтобы drag'н'drop не затирал inflight-правки в input'e.
  useEffect(() => {
    if (!region) return;
    setStart(fmtMsExact(region.start * 1000));
    setEnd(fmtMsExact(region.end * 1000));
    setReason(region.reason ?? "");
    setScore(region.score != null ? String(region.score) : "");
  }, [region?.id]);

  if (!region) {
    return (
      <Card
        variant="surface"
        padding="lg"
        className={cn(
          "flex h-full flex-col items-center justify-center text-center",
          className
        )}
      >
        <div className="mb-2 text-sm font-medium text-muted">Регион не выбран</div>
        <div className="text-xs text-subtle">
          Клик по цветной области на волне — откроет редактор.
        </div>
      </Card>
    );
  }

  const startMs = parseTimecode(start);
  const endMs = parseTimecode(end);
  const startValid = startMs !== null && startMs >= 0 && startMs <= durationMs;
  const endValid = endMs !== null && endMs > 0 && endMs <= durationMs;
  const rangeValid = startValid && endValid && startMs! < endMs!;

  const scoreNum = score.trim() === "" ? null : Number(score);
  const scoreValid = scoreNum === null || (Number.isFinite(scoreNum) && scoreNum >= 0 && scoreNum <= 1);

  const canApply = rangeValid && scoreValid;

  const apply = () => {
    if (!canApply) return;
    onCommit({
      id: region.id,
      start: startMs! / 1000,
      end: endMs! / 1000,
      reason: reason.trim() || undefined,
      score: scoreNum ?? undefined,
    });
  };

  return (
    <Card variant="elevated" padding="md" className={cn("flex h-full flex-col", className)}>
      <CardHeader className="mb-3">
        <div className="flex min-w-0 items-center gap-2">
          <span className={cn("h-2.5 w-2.5 rounded-full", KIND_DOT[region.kind])} />
          <div className="text-sm font-semibold text-fg">
            {KIND_LABEL[region.kind]}
          </div>
          <div className="ml-2 truncate font-mono text-[11px] text-subtle">
            #{region.id.slice(0, 8)}
          </div>
        </div>
        <Button
          size="sm"
          variant="ghost"
          onClick={onClose}
          className="h-7 w-7 p-0"
          aria-label="Закрыть"
        >
          <X className="h-4 w-4" />
        </Button>
      </CardHeader>

      <div className="flex-1 space-y-4 overflow-y-auto">
        <Field label="Начало">
          <Input
            value={start}
            onChange={(e) => setStart(e.target.value)}
            onBlur={apply}
            onKeyDown={(e) => e.key === "Enter" && apply()}
            invalid={!startValid && start !== ""}
            placeholder="HH:MM:SS.mmm"
            className="font-mono"
          />
        </Field>
        <Field label="Конец">
          <Input
            value={end}
            onChange={(e) => setEnd(e.target.value)}
            onBlur={apply}
            onKeyDown={(e) => e.key === "Enter" && apply()}
            invalid={(!endValid && end !== "") || (rangeValid === false && startValid && endValid)}
            placeholder="HH:MM:SS.mmm"
            className="font-mono"
          />
        </Field>

        <Field
          label={
            region.kind === "highlight"
              ? "Описание / причина"
              : region.kind === "cut"
                ? "Причина выреза"
                : "Слова / источник"
          }
        >
          {region.kind === "mute" && region.words?.length ? (
            <div className="flex flex-wrap gap-1 rounded-lg border border-white/5 bg-black/30 px-2 py-1.5">
              {region.words.map((w, i) => (
                <span key={i} className="rounded bg-err/20 px-1.5 py-0.5 text-[11px] text-err">
                  {w}
                </span>
              ))}
            </div>
          ) : (
            <Input
              value={reason}
              onChange={(e) => setReason(e.target.value)}
              onBlur={apply}
              placeholder={region.kind === "highlight" ? "О чём фрагмент" : "Причина (необязательно)"}
            />
          )}
        </Field>

        {region.kind === "highlight" && (
          <Field label="Оценка (0..1)">
            <Input
              value={score}
              onChange={(e) => setScore(e.target.value)}
              onBlur={apply}
              onKeyDown={(e) => e.key === "Enter" && apply()}
              invalid={!scoreValid}
              inputMode="decimal"
              placeholder="0.85"
            />
          </Field>
        )}

        <div className="rounded-lg border border-white/5 bg-black/20 p-3 text-xs text-muted">
          <div className="flex justify-between">
            <span className="text-subtle">Длительность</span>
            <span className="font-mono text-fg">
              {rangeValid ? fmtMsExact(endMs! - startMs!) : "—"}
            </span>
          </div>
        </div>
      </div>

      <div className="mt-4 flex items-center justify-between gap-2">
        <Button
          variant="destructive"
          size="sm"
          onClick={() => onDelete(region.id)}
        >
          <Trash2 className="mr-1.5 h-3.5 w-3.5" /> Удалить
        </Button>
        <Button
          variant="primary"
          size="sm"
          onClick={apply}
          disabled={!canApply}
        >
          Применить
        </Button>
      </div>
    </Card>
  );
}

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <label className="block">
      <div className="mb-1.5 text-[11px] font-medium uppercase tracking-wider text-subtle">
        {label}
      </div>
      {children}
    </label>
  );
}
