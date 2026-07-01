import { useEffect, useMemo, useState } from "react";
import {
  AlertCircle,
  ArrowRight,
  Ban,
  CheckCircle2,
  Copy,
  ExternalLink,
  Loader2,
  Terminal,
} from "lucide-react";
import { cancelJob, type JobState } from "../api";
import { useJobEvents, type JobLogEntry } from "../hooks/useJobEvents";
import { Button } from "../components/ui/Button";
import { Card, CardTitle } from "../components/ui/Card";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "../components/ui/Dialog";
import { ProgressRing } from "../components/ui/ProgressBar";
import { ScrollArea } from "../components/ui/ScrollArea";
import { StatusBadge } from "../components/ui/StatusBadge";
import { PROCESS_STEPS, Stepper } from "../components/ui/Stepper";
import { fmtRelativeMs } from "../lib/format";

/**
 * JobScreen — «диспетчерская» одной джобы.
 *
 * Layout: hero-блок (ProgressRing 200px + status/message) + двухколоночный
 * grid — Stepper слева, live-лог справа. Внизу — CTA-панель, меняющаяся по
 * статусу (Отмена / «Открыть таймлайн» / «Показать ошибку»).
 *
 * Стриминг делает useJobEvents-хук. При final=failed открываем Dialog с
 * traceback'ом; при final=done показываем зелёный CTA. При running — красная
 * кнопка «Отменить», подтверждение через отдельный Dialog.
 */
export function JobScreen({
  jobId,
  onDone,
}: {
  jobId: string;
  onDone: (decisionsPath: string | null) => void;
}) {
  const { state, log, error, wsReady } = useJobEvents(jobId);
  const [cancelling, setCancelling] = useState(false);
  const [errorOpen, setErrorOpen] = useState(false);
  const [cancelOpen, setCancelOpen] = useState(false);

  // Автооткрытие error-диалога при переходе в failed.
  useEffect(() => {
    if (state?.status === "failed") setErrorOpen(true);
  }, [state?.status]);

  const doCancel = async () => {
    setCancelling(true);
    try {
      await cancelJob(jobId);
    } finally {
      setCancelling(false);
      setCancelOpen(false);
    }
  };

  const decisionsPath = (state?.result as any)?.decisions_path ?? null;

  return (
    <div className="mx-auto flex max-w-6xl flex-col gap-6 p-8">
      <Header state={state} wsReady={wsReady} />

      {/* Hero + Stepper + Log */}
      <div className="grid grid-cols-1 gap-6 lg:grid-cols-[280px_minmax(0,1fr)]">
        {/* Hero: ProgressRing */}
        <div className="flex flex-col items-center justify-start gap-4">
          <ProgressRing value={state?.progress ?? 0} size={200} stroke={10}>
            <div className="flex flex-col items-center">
              <div className="font-mono-tabular text-4xl font-semibold text-fg tabular-nums">
                {Math.round(state?.progress ?? 0)}
                <span className="ml-0.5 text-xl text-subtle">%</span>
              </div>
              <div className="mt-1 max-w-[150px] truncate text-center text-[11px] uppercase tracking-wider text-subtle">
                {state?.stage || "запуск…"}
              </div>
            </div>
          </ProgressRing>
          <div className="w-full max-w-[220px] text-center text-xs text-muted">
            {state?.message || "Ждём событий с backend…"}
          </div>
        </div>

        {/* Stepper */}
        <Card variant="surface" padding="lg">
          <div className="mb-4 flex items-center justify-between">
            <CardTitle>Стадии</CardTitle>
            <span className="text-xs text-subtle">
              runner.py · <span className="font-mono-tabular">process</span>
            </span>
          </div>
          <Stepper
            steps={PROCESS_STEPS}
            currentStage={state?.stage ?? ""}
            completed={state?.status === "done"}
            failed={state?.status === "failed"}
          />
        </Card>
      </div>

      {/* Live-log */}
      <Card variant="surface" padding="none">
        <div className="flex items-center justify-between border-b border-white/5 px-5 py-3">
          <div className="flex items-center gap-2 text-sm">
            <Terminal className="h-4 w-4 text-brand-from" />
            <span className="font-medium text-fg">Живой лог</span>
            <span className="text-xs text-subtle">{log.length} строк</span>
          </div>
          <button
            type="button"
            onClick={() => {
              const text = log
                .map((l) => `[${l.stage}] ${l.progress.toFixed(1)}%  ${l.message}`)
                .join("\n");
              void navigator.clipboard.writeText(text);
            }}
            className="inline-flex items-center gap-1.5 rounded px-2 py-1 text-xs text-muted transition-colors hover:bg-white/5 hover:text-fg"
            title="Скопировать лог"
          >
            <Copy className="h-3 w-3" /> Копировать
          </button>
        </div>
        <ScrollArea className="h-[280px]">
          <div className="flex flex-col divide-y divide-white/[0.04] px-5 py-2 font-mono-tabular text-[12px]">
            {log.length === 0 ? (
              <div className="py-8 text-center text-subtle">пока пусто</div>
            ) : (
              log.map((l, i) => <LogRow key={i} entry={l} />)
            )}
          </div>
        </ScrollArea>
      </Card>

      {/* CTA-панель */}
      <FooterBar
        state={state}
        cancelling={cancelling}
        onRequestCancel={() => setCancelOpen(true)}
        onOpenTimeline={() => onDone(decisionsPath)}
        onShowError={() => setErrorOpen(true)}
      />

      {error && !state && (
        <Card
          variant="surface"
          className="border-err/30 bg-err/5 text-sm text-err"
        >
          <div className="font-medium">Не удалось подключиться к job'у</div>
          <div className="mt-1 break-all font-mono-tabular text-[11px] text-err/90">
            {error}
          </div>
        </Card>
      )}

      {/* Dialogs */}
      <Dialog open={cancelOpen} onOpenChange={setCancelOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Отменить джобу?</DialogTitle>
            <DialogDescription>
              Текущая стадия будет прервана. Кэш уже посчитанных стадий сохранится —
              следующий запуск с теми же параметрами продолжит с чекпоинта.
            </DialogDescription>
          </DialogHeader>
          <DialogFooter>
            <Button variant="secondary" onClick={() => setCancelOpen(false)}>
              Не отменять
            </Button>
            <Button
              variant="destructive"
              onClick={doCancel}
              loading={cancelling}
            >
              <Ban className="h-4 w-4" /> Отменить
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      <Dialog open={errorOpen} onOpenChange={setErrorOpen}>
        <DialogContent className="max-w-2xl">
          <DialogHeader>
            <DialogTitle className="flex items-center gap-2 text-err">
              <AlertCircle className="h-5 w-5" /> Job упал
            </DialogTitle>
            <DialogDescription>
              Traceback ниже. Полный лог тоже сохранён — скопируйте через кнопку «Копировать».
            </DialogDescription>
          </DialogHeader>
          <ScrollArea className="max-h-[400px] rounded-lg border border-white/5 bg-black/40 p-3">
            <pre className="whitespace-pre-wrap break-all font-mono-tabular text-[11px] text-err/90">
              {state?.error || "(пусто)"}
            </pre>
          </ScrollArea>
          <DialogFooter>
            <Button
              variant="secondary"
              onClick={() => {
                if (state?.error) void navigator.clipboard.writeText(state.error);
              }}
            >
              <Copy className="h-4 w-4" /> Копировать traceback
            </Button>
            <Button onClick={() => setErrorOpen(false)}>Закрыть</Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}

function Header({
  state,
  wsReady,
}: {
  state: JobState | null;
  wsReady: boolean;
}) {
  const createdMs = useMemo(() => {
    if (!state?.created_at) return null;
    const t = Date.parse(state.created_at);
    return Number.isNaN(t) ? null : t;
  }, [state?.created_at]);

  return (
    <header className="flex flex-wrap items-start justify-between gap-3">
      <div>
        <div className="flex items-center gap-3">
          <h1 className="text-2xl font-semibold tracking-tight text-fg">
            {stageLabel(state?.stage)}
          </h1>
          {state && <StatusBadge status={state.status} />}
        </div>
        <div className="mt-1 flex flex-wrap items-center gap-3 text-xs text-subtle">
          <span>
            job <span className="font-mono-tabular text-muted">{state?.id.slice(0, 8) ?? "…"}</span>
          </span>
          {createdMs != null && <span>создан {fmtRelativeMs(createdMs)}</span>}
          {!wsReady && state?.status !== "done" && state?.status !== "failed" && (
            <span className="inline-flex items-center gap-1 text-brand-from">
              <Loader2 className="h-3 w-3 animate-spin" /> подключение…
            </span>
          )}
        </div>
      </div>
    </header>
  );
}

function FooterBar({
  state,
  cancelling,
  onRequestCancel,
  onOpenTimeline,
  onShowError,
}: {
  state: JobState | null;
  cancelling: boolean;
  onRequestCancel: () => void;
  onOpenTimeline: () => void;
  onShowError: () => void;
}) {
  if (!state) return null;
  if (state.status === "done") {
    return (
      <Card
        variant="elevated"
        className="flex flex-wrap items-center justify-between gap-3 border-ok/30 bg-ok/5"
      >
        <div className="flex items-center gap-3">
          <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-ok/15 text-ok">
            <CheckCircle2 className="h-5 w-5" />
          </div>
          <div>
            <div className="text-sm font-medium text-fg">Готово</div>
            <div className="text-xs text-muted">
              {(state.result as any)?.mutes_count ?? 0} mutes ·{" "}
              {(state.result as any)?.cuts_count ?? 0} cuts
            </div>
          </div>
        </div>
        <Button onClick={onOpenTimeline}>
          Открыть таймлайн <ArrowRight className="h-4 w-4" />
        </Button>
      </Card>
    );
  }
  if (state.status === "failed") {
    return (
      <Card
        variant="elevated"
        className="flex flex-wrap items-center justify-between gap-3 border-err/30 bg-err/5"
      >
        <div className="flex items-center gap-3">
          <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-err/15 text-err">
            <AlertCircle className="h-5 w-5" />
          </div>
          <div className="min-w-0">
            <div className="text-sm font-medium text-fg">Ошибка</div>
            <div className="truncate text-xs text-err/80">
              {firstLine(state.error) || "см. traceback"}
            </div>
          </div>
        </div>
        <Button variant="destructive" onClick={onShowError}>
          <ExternalLink className="h-4 w-4" /> Показать traceback
        </Button>
      </Card>
    );
  }
  if (state.status === "cancelled") {
    return (
      <Card
        variant="surface"
        className="flex items-center justify-between gap-3 border-warn/30"
      >
        <div className="text-sm text-warn">Отменено пользователем.</div>
      </Card>
    );
  }
  // pending / running
  return (
    <Card
      variant="surface"
      className="flex flex-wrap items-center justify-between gap-3"
    >
      <div className="text-sm text-muted">
        Можно закрыть окно — job продолжит выполняться на backend'е. Живой лог
        подхватится при возврате.
      </div>
      <Button
        variant="destructive"
        onClick={onRequestCancel}
        loading={cancelling}
      >
        <Ban className="h-4 w-4" /> Отменить
      </Button>
    </Card>
  );
}

function LogRow({ entry }: { entry: JobLogEntry }) {
  const color =
    entry.level === "error"
      ? "text-err"
      : entry.level === "warn"
        ? "text-warn"
        : "text-muted";
  return (
    <div className="grid grid-cols-[70px_140px_60px_1fr] gap-3 py-1 leading-relaxed">
      <span className="text-subtle tabular-nums">
        {new Date(entry.ts).toLocaleTimeString(undefined, { hour12: false })}
      </span>
      <span className="truncate text-brand-from">{entry.stage}</span>
      <span className="text-right tabular-nums text-subtle">
        {entry.progress.toFixed(1)}%
      </span>
      <span className={`min-w-0 break-words ${color}`}>{entry.message || "—"}</span>
    </div>
  );
}

function stageLabel(stage: string | undefined) {
  if (!stage) return "Прогресс job'a";
  const step = PROCESS_STEPS.find(
    (s) => s.key === stage || s.aliases?.includes(stage)
  );
  return step?.title ?? "Прогресс job'a";
}

function firstLine(s: string | null | undefined) {
  if (!s) return "";
  const line = s.split("\n", 1)[0];
  return line.length > 120 ? line.slice(0, 120) + "…" : line;
}
