import { useEffect, useMemo, useRef, useState } from "react";
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
import { cancelJob, createProcessJob, getDefaultSensitivity, type JobState } from "../api";
import { useJobEvents, type JobLogEntry } from "../hooks/useJobEvents";
import { Button } from "../components/ui/Button";
import { Card } from "../components/ui/Card";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "../components/ui/Dialog";
import { ProgressBar } from "../components/ui/ProgressBar";
import { ScrollArea } from "../components/ui/ScrollArea";
import { StatusBadge } from "../components/ui/StatusBadge";
import { Stepper, stepsForKind, stageLabel } from "../components/ui/Stepper";
import { fmtDurationMs, fmtRelativeMs } from "../lib/format";

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
  onReady,
  onChainJob,
}: {
  jobId: string;
  onDone: (decisionsPath: string | null) => void;
  /**
   * Джоба завершилась и оставила decisions.json. В отличие от `onDone` (клик
   * по «Открыть таймлайн») НЕ переключает экран — только сообщает родителю
   * путь, чтобы тот разблокировал «Правку»/«Экспорт». Иначе вкладки оставались
   * бы серыми до клика, хотя всё уже готово.
   */
  onReady?: (decisionsPath: string) => void;
  /**
   * Цепочка: download_vod по завершении сам стартует process-джобу и просит
   * родителя перецепить экран на неё. Один поток «скачал → обработал →
   * таймлайн» без кликов. Опционально — при отсутствии просто покажем «скачано».
   */
  onChainJob?: (jobId: string) => void;
}) {
  const { state, log, error, wsReady } = useJobEvents(jobId);
  const [cancelling, setCancelling] = useState(false);
  const [errorOpen, setErrorOpen] = useState(false);
  const [cancelOpen, setCancelOpen] = useState(false);
  // Авто-цепочка download→process. `chaining` — идёт старт process-джобы (между
  // done скачивания и перецепкой экрана): показываем переходное состояние вместо
  // мёртвой кнопки. `chainError` — не удалось стартовать обработку. Ref-гард от
  // повторного запуска: final может прилететь и по WS, и повторным ре-рендером.
  const [chaining, setChaining] = useState(false);
  const [chainError, setChainError] = useState<string | null>(null);
  const chainStartedRef = useRef(false);
  // onChainJob тоже в ref, и по той же причине: App передаёт инлайн-стрелку,
  // новую на каждый свой рендер. В deps эффекта цепочки она была фитилём —
  // старт process-джобы менял activeJobId → App ре-рендерился → новая стрелка →
  // эффект перезапускался, а гард как раз обнулялся сменой jobId. Итог: цепочка
  // стартовала заново, пока не доезжал GET нового состояния — десятки
  // process-джоб на один скачанный VOD.
  const onChainJobRef = useRef(onChainJob);
  onChainJobRef.current = onChainJob;
  // onReady в ref: App передаёт инлайн-стрелку, она новая на каждый его рендер.
  // Через ref эффект «готово» зависит только от статуса джобы, а не от
  // идентичности пропа.
  const onReadyRef = useRef(onReady);
  onReadyRef.current = onReady;

  // Оценка «сколько осталось». Наивная формула elapsed/progress от старта job'а
  // дёргается вверх на прогреве (ffmpeg инициализирует энкодер, первые time=
  // шумные, стадия render мапит 0-100→10-100). Поэтому:
  //  1) Считаем скорость от ЯКОРЯ — первого надёжного тика (progress ≥ 12%),
  //     а не от created_at. Прогрев в расчёт не попадает.
  //  2) Держим оценку МОНОТОННОЙ вниз (min с прошлой) пока прогресс растёт —
  //     число только убывает, как ждёт пользователь.
  // Якорь и последняя оценка сбрасываются при смене job'а / падении прогресса.
  const [etaMs, setEtaMs] = useState<number | null>(null);
  const anchorRef = useRef<{ atMs: number; progress: number } | null>(null);
  const lastEtaRef = useRef<number | null>(null);
  const etaJobRef = useRef<string | null>(null);

  useEffect(() => {
    const p = state?.progress ?? 0;
    const running = state?.status === "running";

    // Сброс при смене job'а или откате прогресса (новый прогон в том же экране).
    if (etaJobRef.current !== (state?.id ?? null)) {
      etaJobRef.current = state?.id ?? null;
      anchorRef.current = null;
      lastEtaRef.current = null;
      setEtaMs(null);
    }
    if (!running || p >= 99.5) {
      if (!running) {
        anchorRef.current = null;
        lastEtaRef.current = null;
      }
      setEtaMs(null);
      return;
    }
    // Ставим якорь на первом надёжном тике — рендер реально пошёл (не init/prepare).
    if (!anchorRef.current) {
      if (p < 12) return; // ещё прогрев — рано оценивать
      anchorRef.current = { atMs: Date.now(), progress: p };
      return; // нужна вторая точка, чтобы померить скорость
    }
    const a = anchorRef.current;
    const dp = p - a.progress;
    if (dp <= 0.1) return; // прогресс ещё не сдвинулся заметно
    const elapsed = Date.now() - a.atMs;
    const remaining = (elapsed / dp) * (100 - p);
    if (!Number.isFinite(remaining) || remaining <= 0) return;
    // Монотонность вниз: показываем не больше прошлой оценки.
    const clamped =
      lastEtaRef.current != null ? Math.min(remaining, lastEtaRef.current) : remaining;
    lastEtaRef.current = clamped;
    setEtaMs(clamped);
  }, [state?.id, state?.progress, state?.status]);

  // Автооткрытие error-диалога при переходе в failed.
  useEffect(() => {
    if (state?.status === "failed") setErrorOpen(true);
  }, [state?.status]);

  // Сброс состояния цепочки при смене джобы (в т.ч. когда мы сами перецепили
  // экран на process-джобу — та уже не download_vod, цепочку не повторяем).
  useEffect(() => {
    chainStartedRef.current = false;
    setChaining(false);
    setChainError(null);
  }, [jobId]);

  // Авто-цепочка: download_vod завершился → стартуем обработку скачанного файла
  // и просим родителя перецепить экран на новую джобу. Так «обычный режим» =
  // один сквозной поток (скачал → распознал → мат → правки → таймлайн).
  useEffect(() => {
    if (state?.kind !== "download_vod" || state.status !== "done") return;
    // Снапшот обязан относиться к ТЕКУЩЕЙ джобе. Хук сбрасывает state при смене
    // jobId, но эта сверка — дешёвый структурный замок: она делает эффект
    // невосприимчивым к любому чужому/устаревшему снапшоту, а не только к тому
    // одному случаю, который мы уже починили в useJobEvents.
    if (state.id !== jobId) return;
    const chain = onChainJobRef.current;
    if (!chain || chainStartedRef.current) return;
    const stream =
      ((state.result as any)?.stream as string) ||
      ((state.result as any)?.output as string) ||
      "";
    if (!stream) {
      // Файла нет — цепочку не запустить; покажем это как ошибку цепочки, но
      // файл всё равно скачан (FooterBar даст «Показать файл»).
      setChainError("Скачанный файл не найден — обработку не запустить.");
      return;
    }
    chainStartedRef.current = true;
    setChaining(true);
    setChainError(null);
    (async () => {
      try {
        // Уровень чувствительности из настроек — цепочка «скачал → обработал»
        // это обычный режим работы, и настройка обязана действовать и здесь,
        // а не только в форме новой обработки.
        const sensitivity = await getDefaultSensitivity();
        const next = await createProcessJob({
          stream,
          ...(sensitivity ? { sensitivity } : {}),
        });
        chain(next.id);
      } catch (err) {
        chainStartedRef.current = false;
        setChaining(false);
        setChainError(err instanceof Error ? err.message : String(err));
      }
    })();
  }, [state?.id, state?.kind, state?.status, state?.result, jobId]);

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

  // Готово и есть разметка → сразу отпускаем вкладки «Правка»/«Экспорт», не
  // дожидаясь клика по «Открыть таймлайн». Ref-гард: `state` обновляется на
  // каждом WS-событии, а дёрнуть родителя достаточно один раз на джобу —
  // повторные вызовы плодили бы лишние re-render'ы всего приложения.
  const readyNotifiedRef = useRef<string | null>(null);
  useEffect(() => {
    if (state?.status !== "done" || !decisionsPath) return;
    if (readyNotifiedRef.current === jobId) return;
    readyNotifiedRef.current = jobId;
    onReadyRef.current?.(decisionsPath);
  }, [state?.status, decisionsPath, jobId]);

  return (
    <div className="screen-page flex flex-col gap-5">
      <Header state={state} wsReady={wsReady} />

      <Card variant="surface" padding="lg" className="processing-panel">
        <Stepper className="processing-steps" steps={stepsForKind(state?.kind)} currentStage={state?.stage ?? ""} completed={state?.status === "done"} failed={state?.status === "failed"} />
        <div className="my-7 border-t border-white/10" />
        <div className="mb-4 flex items-end justify-between gap-4">
          <div>
            <p className="mb-2 text-sm text-muted">{stageLabel(state?.kind, state?.stage)}</p>
            <p className="text-sm text-fg">{state?.message || "Ожидаем начало обработки…"}</p>
          </div>
          <span className="font-mono-tabular text-4xl text-fg">{Math.round(state?.progress ?? 0)}<span className="text-xl text-muted">%</span></span>
        </div>
        <ProgressBar value={state?.progress ?? 0} label="Прогресс обработки" />
        <p className="mt-4 text-xs text-muted">{etaMs != null ? `Примерно осталось: ${fmtDurationMs(etaMs)}` : "Оценка времени появится по ходу обработки"}</p>
      </Card>

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
          <div
            className="flex flex-col divide-y divide-white/[0.04] px-5 py-2 font-mono-tabular text-[12px]"
            aria-live="polite"
            aria-relevant="additions"
          >
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
        chaining={chaining}
        chainError={chainError}
        onRequestCancel={() => setCancelOpen(true)}
        onOpenTimeline={() => onDone(decisionsPath)}
        onShowError={() => setErrorOpen(true)}
      />

      {/* Ошибка авто-цепочки: файл скачан, но обработку не удалось стартовать —
          показываем отдельно от WS-ошибки подключения. */}
      {chainError && (
        <Card
          variant="surface"
          className="border-err/30 bg-err/5 text-sm text-err"
        >
          <div className="font-medium">Скачано, но обработка не запустилась</div>
          <div className="mt-1 break-all font-mono-tabular text-[11px] text-err/90">
            {chainError}
          </div>
        </Card>
      )}

      {error && !state && (
        <Card
          variant="surface"
          className="border-err/30 bg-err/5 text-sm text-err"
        >
          <div className="font-medium">Не удалось подключиться к обработке</div>
          <div className="mt-1 break-all font-mono-tabular text-[11px] text-err/90">
            {error}
          </div>
        </Card>
      )}

      {/* Dialogs */}
      <Dialog open={cancelOpen} onOpenChange={setCancelOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Отменить обработку?</DialogTitle>
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
              <AlertCircle className="h-5 w-5" /> Обработка прервалась
            </DialogTitle>
            <DialogDescription>
              Ниже — технические детали ошибки. Их можно скопировать кнопкой
              «Копировать» и приложить к обращению в поддержку.
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
              <Copy className="h-4 w-4" /> Копировать детали
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
            {stageLabel(state?.kind, state?.stage)}
          </h1>
          {state && <StatusBadge status={state.status} />}
        </div>
        <div className="mt-1 flex flex-wrap items-center gap-3 text-xs text-subtle">
          <span>
            обработка <span className="font-mono-tabular text-muted">{state?.id.slice(0, 8) ?? "…"}</span>
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
  chaining,
  chainError,
  onRequestCancel,
  onOpenTimeline,
  onShowError,
}: {
  state: JobState | null;
  cancelling: boolean;
  chaining: boolean;
  chainError: string | null;
  onRequestCancel: () => void;
  onOpenTimeline: () => void;
  onShowError: () => void;
}) {
  if (!state) return null;
  if (state.status === "done") {
    // download_vod — не финал потока: файл скачан, дальше авто-стартует
    // обработка. Пока цепочка запускается — переходное состояние; если она
    // сорвалась (chainError) — даём «Показать файл» как запасной выход.
    if (state.kind === "download_vod") {
      const out = String(
        (state.result as any)?.stream ?? (state.result as any)?.output ?? ""
      );
      if (chainError) {
        return (
          <Card
            variant="elevated"
            className="flex flex-wrap items-center justify-between gap-3 border-ok/30 bg-ok/5"
          >
            <div className="flex min-w-0 items-center gap-3">
              <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg bg-ok/15 text-ok">
                <CheckCircle2 className="h-5 w-5" />
              </div>
              <div className="min-w-0">
                <div className="text-sm font-medium text-fg">Запись скачана</div>
                <div className="truncate font-mono-tabular text-xs text-muted">
                  {out || "файл сохранён"}
                </div>
              </div>
            </div>
            {out && (
              <Button
                onClick={async () => {
                  await window.twitchCut.showInFolder(out);
                }}
              >
                <ExternalLink className="h-4 w-4" /> Показать файл
              </Button>
            )}
          </Card>
        );
      }
      return (
        <Card
          variant="surface"
          className="flex flex-wrap items-center justify-between gap-3 border-ok/30 bg-ok/5"
        >
          <div className="flex min-w-0 items-center gap-3">
            <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg bg-ok/15 text-ok">
              <CheckCircle2 className="h-5 w-5" />
            </div>
            <div className="min-w-0">
              <div className="text-sm font-medium text-fg">Запись скачана</div>
              <div className="truncate text-xs text-muted">
                {chaining ? "Запускаю обработку…" : "Готовим обработку…"}
              </div>
            </div>
          </div>
          <Button disabled loading>
            <Loader2 className="h-4 w-4" /> Обработка
          </Button>
        </Card>
      );
    }
    // Рендер/детект завершаются файлом, а не таймлайном — своя сводка и CTA.
    if (state.kind === "render") {
      const out = String((state.result as any)?.output_path ?? "");
      return (
        <Card
          variant="elevated"
          className="flex flex-wrap items-center justify-between gap-3 border-ok/30 bg-ok/5"
        >
          <div className="flex min-w-0 items-center gap-3">
            <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg bg-ok/15 text-ok">
              <CheckCircle2 className="h-5 w-5" />
            </div>
            <div className="min-w-0">
              <div className="text-sm font-medium text-fg">Видео готово</div>
              <div className="truncate font-mono-tabular text-xs text-muted">
                {out || "рендер завершён"}
              </div>
            </div>
          </div>
          {out && (
            <Button
              onClick={async () => {
                const ok = await window.twitchCut.showInFolder(out);
                if (!ok) return;
              }}
            >
              <ExternalLink className="h-4 w-4" /> Показать файл
            </Button>
          )}
        </Card>
      );
    }
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
        Можно закрыть окно — обработка продолжится в фоне. Живой лог
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

function firstLine(s: string | null | undefined) {
  if (!s) return "";
  const line = s.split("\n", 1)[0];
  return line.length > 120 ? line.slice(0, 120) + "…" : line;
}
