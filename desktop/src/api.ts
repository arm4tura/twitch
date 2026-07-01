/**
 * Тонкая обёртка над FastAPI backend'ом (backend/src/twitch_cut/server/*).
 *
 * Все методы возвращают уже распаршенный JSON (или бросают Error). WS-подписка
 * на события джобы — отдельный `subscribeJob()` через нативный WebSocket.
 *
 * Базовый URL резолвится один раз на старте через preload (`getBackendPort`).
 * До резолва все вызовы `await`ают одну общую промис-инициализацию.
 */

export type JobStatus = "pending" | "running" | "done" | "failed" | "cancelled";
export type JobKind =
  | "process"
  | "export_vegas"
  | "highlights_export"
  | "highlights_import";

export interface JobState {
  id: string;
  kind: JobKind;
  status: JobStatus;
  progress: number;
  stage: string;
  message: string;
  result: Record<string, any> | null;
  error: string | null;
  created_at: string;
  finished_at: string | null;
}

export interface JobEvent {
  // `snapshot` — catch-up event, отправляется первым при каждом (re)connect
  // WS: пересинхронизирует UI без ожидания следующего runner-tick'а.
  // `progress` — обычный tick от runner'а. `final` — джоба завершена, WS
  // сейчас закроется.
  type: "snapshot" | "progress" | "final";
  state: JobState;
}

let baseUrlPromise: Promise<string> | null = null;
// Синхронно доступный base URL — заполняется после первого успешного getBaseUrl().
// Нужен для mediaUrl(): HTML5 <audio> не умеет ждать промис.
let _cachedBase: string | null = null;

/** Один раз спросить у preload порт и запомнить полный http://... base. */
export function getBaseUrl(): Promise<string> {
  if (baseUrlPromise) return baseUrlPromise;
  baseUrlPromise = window.twitchCut.getBackendPort().then((port) => {
    _cachedBase = `http://127.0.0.1:${port}`;
    return _cachedBase;
  });
  return baseUrlPromise;
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const base = await getBaseUrl();
  const res = await fetch(base + path, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      ...(init?.headers ?? {}),
    },
  });
  if (!res.ok) {
    let detail = "";
    try {
      const body = await res.json();
      detail = body.detail ? JSON.stringify(body.detail) : JSON.stringify(body);
    } catch {
      detail = await res.text();
    }
    throw new Error(`HTTP ${res.status} ${path}: ${detail}`);
  }
  return (await res.json()) as T;
}

// --- health -----------------------------------------------------------------

export function health(): Promise<{ ok: boolean; jobs: number }> {
  return request("/health");
}

// --- jobs -------------------------------------------------------------------

export function listJobs(): Promise<JobState[]> {
  return request("/jobs");
}

export function getJob(id: string): Promise<JobState> {
  return request(`/jobs/${id}`);
}

export function cancelJob(id: string): Promise<{ cancelled: boolean }> {
  return request(`/jobs/${id}`, { method: "DELETE" });
}

// Тела запросов повторяют backend/src/twitch_cut/server/schemas.py — если
// добавляешь поле там, добавь и здесь; extra=forbid отшибёт лишние на бэке.

export interface ProcessJobRequest {
  stream: string;
  original: string;
  banwords: string;
  workdir: string;
  decisions: string;
  vegas: string;
  range_in?: string | null;
  range_out?: string | null;
  model?: string;
  language?: string;
  device?: string;
  compute_type?: string;
  batch_size?: number;
  vad_filter?: boolean;
  vad_method?: string;
  mock_transcript?: string | null;
}

export function createProcessJob(req: ProcessJobRequest): Promise<JobState> {
  return request("/jobs/process", {
    method: "POST",
    body: JSON.stringify(req),
  });
}

export interface HighlightsExportRequest {
  decisions: string;
  out_dir: string;
  transcript?: string | null;
  n_highlights?: number;
}

export function createHighlightsExportJob(
  req: HighlightsExportRequest
): Promise<JobState> {
  return request("/jobs/highlights-export", {
    method: "POST",
    body: JSON.stringify(req),
  });
}

export interface HighlightsImportRequest {
  decisions: string;
  response: string;
  output: string;
  transcript?: string | null;
}

export function createHighlightsImportJob(
  req: HighlightsImportRequest
): Promise<JobState> {
  return request("/jobs/highlights-import", {
    method: "POST",
    body: JSON.stringify(req),
  });
}

export interface ExportVegasRequest {
  decisions: string;
  vegas: string;
}

export function createExportVegasJob(
  req: ExportVegasRequest
): Promise<JobState> {
  return request("/jobs/export-vegas", {
    method: "POST",
    body: JSON.stringify(req),
  });
}

// --- decisions / transcript -------------------------------------------------

export function readDecisions(path: string): Promise<any> {
  return request(`/decisions?path=${encodeURIComponent(path)}`);
}

export function writeDecisions(
  path: string,
  decisions: any
): Promise<{ ok: boolean; path: string }> {
  return request(`/decisions?path=${encodeURIComponent(path)}`, {
    method: "PUT",
    body: JSON.stringify({ decisions }),
  });
}

export function readTranscript(path: string): Promise<any> {
  return request(`/transcript?path=${encodeURIComponent(path)}`);
}

// --- projects registry ------------------------------------------------------

/**
 * Мета проекта из реестра «недавних» (backend/server/projects.py).
 *
 * Единственная нетривиальная тонкость — `updated_at_ms` считается из mtime
 * файла decisions.json, а не из его содержимого. Это позволяет сортировать
 * список по «последнему редактированию» после сохранения из Timeline.
 */
export interface ProjectMeta {
  decisions_path: string;
  name: string;
  workdir: string | null;
  stream_path: string | null;
  updated_at_ms: number;
  duration_ms: number | null;
  mutes_count: number;
  cuts_count: number;
  highlights_count: number;
}

export function listProjects(): Promise<ProjectMeta[]> {
  return request("/projects");
}

// --- settings ---------------------------------------------------------------

/**
 * Настройки — плоский dict. Frontend знает свои ключи (см. SettingsScreen),
 * backend просто хранит. Незнакомые поля из будущих версий сохраняются
 * round-trip.
 */
export type Settings = Record<string, unknown>;

export function getSettings(): Promise<Settings> {
  return request("/settings");
}

export function putSettings(settings: Settings): Promise<{ ok: boolean }> {
  return request("/settings", {
    method: "PUT",
    body: JSON.stringify(settings),
  });
}

export function getLogsDir(): Promise<{ path: string }> {
  return request("/settings/logs_dir");
}

/**
 * Рекомендованный путь workdir для нового джоба. Backend строит
 * `~/twitch_cut/projects/{basename}_{yyyymmdd_hhmm}/` — уникально по минуте,
 * никогда не перезаписывает предыдущий проект.
 */
export function suggestWorkdir(streamPath?: string): Promise<{ path: string }> {
  const qs = streamPath
    ? `?stream=${encodeURIComponent(streamPath)}`
    : "";
  return request(`/paths/suggest_workdir${qs}`);
}

// --- waveform / media (Timeline screen) -------------------------------------

/** Peaks-массив в формате, готовом к передаче wavesurfer.js через `peaks:` opt. */
export interface WaveformData {
  peaks: number[];
  duration_s: number;
  sample_rate: number;
  source: string;
}

/**
 * Получить peaks для отрисовки waveform.
 *
 * Backend кэширует по (path, mtime, size, peaks) — повторные вызовы дешёвые.
 * Для 40-минутного стрима первый вызов занимает ~200-400ms (ffmpeg → 8kHz PCM
 * → downsample). Фронту разумно кэшировать сам объект в React state, чтобы
 * при переоткрытии таймлайна не дёргать даже кэшированный HTTP.
 */
export function getWaveform(
  streamPath: string,
  peaks: number = 1024
): Promise<WaveformData> {
  return request(
    `/waveform?path=${encodeURIComponent(streamPath)}&peaks=${peaks}`
  );
}

/**
 * Полный URL к /media для `<audio src="...">`.
 *
 * Не `Promise` — HTML5 audio нужен sync src в первый render. `getBaseUrl()`
 * гарантированно резолвится ко времени, когда TimelineScreen монтируется
 * (после того как App.tsx уже успел спросить /health).
 *
 * Синхронный доступ — читаем закэшированный порт из глобальной переменной
 * (см. patch в getBaseUrl'e ниже).
 */
export function mediaUrl(streamPath: string): string {
  if (!_cachedBase) {
    throw new Error(
      "mediaUrl called before backend base URL resolved. Await getBaseUrl() first."
    );
  }
  return `${_cachedBase}/media?path=${encodeURIComponent(streamPath)}`;
}

/**
 * Разрешить набор путей для /media и /waveform. Вызывается при открытии
 * проекта из Dashboard: пути, попавшие в decisions.json, нужно легально
 * подгружать даже если процесс backend'а был перезапущен и потерял whitelist.
 */
export function allowMediaPaths(
  paths: string[]
): Promise<{ allowed: string[] }> {
  return request("/waveform/allow", {
    method: "POST",
    body: JSON.stringify({ paths }),
  });
}

// --- WebSocket подписка на события джобы ------------------------------------

/**
 * Подписаться на события джобы с автоматическим reconnect.
 *
 * Возвращает функцию отписки. Пока она НЕ вызвана — при обрыве соединения
 * (idle-таймаут OS/proxy, backend перезапуск, sleep-wake ноутбука) хук сам
 * переподключается с exponential backoff (100ms → 200 → 400 → 800 → 1600 → 3000
 * capped). На каждом (re)connect бэкенд первым событием отдаёт catch-up
 * snapshot текущего state — UI мгновенно синхронизируется.
 *
 * Callback стиль без RxJS: две страницы UI обходятся, не тащим зависимость.
 */
export async function subscribeJob(
  id: string,
  onEvent: (ev: JobEvent) => void,
  onError?: (err: Event | Error) => void
): Promise<() => void> {
  const base = await getBaseUrl();
  const wsUrl = base.replace(/^http/, "ws") + `/jobs/${id}/events`;

  let closedByUser = false;
  let currentWs: WebSocket | null = null;
  let reconnectTimer: number | null = null;
  let attempt = 0;

  const connect = () => {
    if (closedByUser) return;
    const ws = new WebSocket(wsUrl);
    currentWs = ws;
    ws.onopen = () => {
      // Успешное соединение сбрасывает backoff.
      attempt = 0;
    };
    ws.onmessage = (msg) => {
      try {
        const parsed = JSON.parse(msg.data) as JobEvent;
        onEvent(parsed);
      } catch (err) {
        onError?.(err as Error);
      }
    };
    ws.onerror = (e) => onError?.(e);
    ws.onclose = () => {
      currentWs = null;
      if (closedByUser) return;
      // Backoff: 100 * 2^attempt, capped at 3000ms. attempt++ ДО расчёта,
      // чтобы первый reconnect подождал 100ms (а не 0).
      attempt = Math.min(attempt + 1, 6);
      const delay = Math.min(100 * 2 ** (attempt - 1), 3000);
      reconnectTimer = window.setTimeout(connect, delay);
    };
  };

  connect();

  return () => {
    closedByUser = true;
    if (reconnectTimer !== null) {
      window.clearTimeout(reconnectTimer);
      reconnectTimer = null;
    }
    currentWs?.close();
    currentWs = null;
  };
}
