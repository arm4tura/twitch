import { useEffect, useRef, useState } from "react";
import { getJob, subscribeJob, type JobEvent, type JobState } from "../api";

/**
 * useJobEvents — подписка на конкретную job.
 *
 * Стратегия:
 * 1. При маунте — GET /jobs/:id для мгновенного snapshot (WS-соединение может
 *    провозиться несколько сотен ms, а нам сразу нужны stage/progress для UI).
 * 2. Затем — WebSocket-подписка через `subscribeJob`. Каждое событие обновляет
 *    `state`; событие `final` также вызывает `onFinal` callback (это нужно
 *    JobScreen'у, чтобы, например, зарегистрировать проект в реестре или
 *    автоматически перейти на Timeline через таймаут).
 * 3. Log — накапливаем строки на клиенте из полей `stage/message` каждого
 *    события. Дедупим одинаковые последовательные записи (Progress.emit
 *    вызывается часто с тем же сообщением на разных %).
 *
 * Возвращаем `{state, log, error, wsReady}`. `error` — ошибка WS/HTTP,
 * `wsReady=false` до первого сообщения WS ИЛИ пока идёт первичный GET.
 */

export interface JobLogEntry {
  ts: number;
  stage: string;
  message: string;
  level: "info" | "warn" | "error";
  progress: number;
}

export interface UseJobEventsResult {
  state: JobState | null;
  log: JobLogEntry[];
  error: string | null;
  wsReady: boolean;
}

export function useJobEvents(
  jobId: string | null,
  onFinal?: (state: JobState) => void
): UseJobEventsResult {
  const [state, setState] = useState<JobState | null>(null);
  const [log, setLog] = useState<JobLogEntry[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [wsReady, setWsReady] = useState(false);
  // Держим последний stage+message, чтобы не дублировать. Ref, не state:
  // изменение не должно триггерить render.
  const lastLogRef = useRef<string>("");
  // onFinal тоже в ref — иначе перепоставка подписки при каждом ре-рендере родителя.
  const onFinalRef = useRef(onFinal);
  onFinalRef.current = onFinal;

  useEffect(() => {
    if (!jobId) {
      setState(null);
      setLog([]);
      setWsReady(false);
      return;
    }
    let alive = true;
    let unsub: (() => void) | null = null;
    setError(null);
    lastLogRef.current = "";

    const appendLog = (s: JobState) => {
      const key = `${s.stage}::${s.message}`;
      if (key === lastLogRef.current) return;
      lastLogRef.current = key;
      const level: JobLogEntry["level"] =
        s.status === "failed" ? "error" : s.status === "cancelled" ? "warn" : "info";
      const entry: JobLogEntry = {
        ts: Date.now(),
        stage: s.stage || "—",
        message: s.message || "",
        level,
        progress: s.progress,
      };
      setLog((prev) => {
        // Кэп 500 строк — простая защита от «завтра пришлось листать 2 ГБ лога».
        const next = [...prev, entry];
        return next.length > 500 ? next.slice(-500) : next;
      });
    };

    (async () => {
      try {
        const initial = await getJob(jobId);
        if (!alive) return;
        setState(initial);
        appendLog(initial);
      } catch (err) {
        if (!alive) return;
        setError(err instanceof Error ? err.message : String(err));
      }
    })();

    (async () => {
      try {
        unsub = await subscribeJob(
          jobId,
          (ev: JobEvent) => {
            if (!alive) return;
            setWsReady(true);
            setState(ev.state);
            appendLog(ev.state);
            if (ev.type === "final") {
              onFinalRef.current?.(ev.state);
            }
          },
          (err) => {
            if (!alive) return;
            setError(err instanceof Error ? err.message : "WebSocket error");
          }
        );
      } catch (err) {
        if (!alive) return;
        setError(err instanceof Error ? err.message : String(err));
      }
    })();

    return () => {
      alive = false;
      unsub?.();
    };
  }, [jobId]);

  return { state, log, error, wsReady };
}
