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
  | "highlights_import"
  | "render_highlights"
  | "export_vegas_highlights"
  | "detect_silence"
  | "render"
  | "download_vod"
  | "detect_offtopic";

/** Highlight из decisions.json (bundle: highlights[].highlights). */
export interface Highlight {
  start_s: number;
  end_s: number;
  title: string;
  reason: string;
  score: number;
  quote?: string | null;
  /** Абсолютные секунды исходника — проставляет merge (stream_offset). */
  stream_start_s?: number;
  stream_end_s?: number;
}

export interface JobState {
  id: string;
  kind: JobKind;
  status: JobStatus;
  progress: number;
  stage: string;
  message: string;
  result: Record<string, unknown> | null;
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
  original?: string | null;
  banwords?: string | null;
  workdir?: string | null;
  decisions?: string | null;
  vegas?: string | null;
  range_in?: string | null;
  range_out?: string | null;
  // Движок ASR. По умолчанию (не передаём) бэкенд берёт GigaAM v3. Расширенный
  // режим шлёт 'whisperx', т.к. настраивает именно его параметры.
  transcriber?: string;
  gigaam_model?: string;
  model?: string;
  language?: string;
  device?: string | null;
  compute_type?: string | null;
  batch_size?: number;
  vad_filter?: boolean;
  vad_method?: string;
  // Чувствительность детекции мата: 'low' | 'balanced' | 'high'. Одна ручка
  // вместо шести чисел (отступы мьюта, склейка, поджим границ по звуку, порог
  // спотинга) — их значения выводит бэкенд (config.SENSITIVITY_LEVELS). Здесь
  // намеренно НЕТ самих чисел: они не должны разъезжаться между UI и бэком.
  sensitivity?: string;
  mock_transcript?: string | null;
}

export function createProcessJob(req: ProcessJobRequest): Promise<JobState> {
  return request("/jobs/process", {
    method: "POST",
    body: JSON.stringify(req),
  });
}

// --- download VOD -----------------------------------------------------------
// Скачивание Twitch VOD через bundled TwitchDownloaderCLI (subprocess на бэке).
// Тело повторяет DownloadVodRequest в schemas.py.

export interface DownloadVodRequest {
  /** Twitch VOD URL или числовой ID. Бэкенд извлекает ID сам. */
  url: string;
  /** Значение качества из VodInfo.qualities[].value. По умолчанию источник. */
  quality?: string;
  /** Обрезка (таймкоды HH:MM:SS). Пусто/None → с начала / до конца. */
  range_in?: string | null;
  range_out?: string | null;
  /** OAuth-токен для sub-only VOD. Пусто, если VOD публичный. */
  oauth?: string | null;
  /** Куда качать; None → бэкенд создаёт подпапку в work/. */
  workdir?: string | null;
}

/** Одно доступное качество VOD (из info-режима CLI). */
export interface VodQualityInfo {
  value: string;
  label: string;
}

/** Метаданные VOD для превью перед скачиванием. */
export interface VodInfo {
  id: string;
  title: string;
  channel: string;
  /** Длительность в мс — граница для полей обрезки. */
  duration_ms: number;
  /** URL превьюшки (может быть null). */
  thumbnail_url: string | null;
  qualities: VodQualityInfo[];
}

/**
 * Запросить метаданные VOD (title/длительность/качества) БЕЗ скачивания.
 * Бэкенд зовёт `TwitchDownloaderCLI info`. Быстрый вызов для превью-карточки.
 */
export function getVodInfo(url: string): Promise<VodInfo> {
  return request(`/vod/info?url=${encodeURIComponent(url)}`);
}

export function createDownloadVodJob(req: DownloadVodRequest): Promise<JobState> {
  return request("/jobs/download-vod", {
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

/** Один source-файл транскрипта в собранном пакете (manifest.chunks[]). */
export interface PackageChunk {
  id: number;
  /** Имя файла относительно out_dir, напр. "transcript_001.md". */
  path: string;
  first_ts: number;
  last_ts: number;
  word_count: number;
}

/**
 * Результат highlights-export job (state.result). out_dir — абсолютный каталог
 * пакета; manifest.chunks — source-файлы транскрипта для загрузки в NotebookLM.
 */
export interface HighlightsExportResult {
  out_dir: string;
  manifest: {
    chunks: PackageChunk[];
    n_highlights: number;
    total_words: number;
    files: { prompt: string; schema: string; readme: string };
  };
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
  /** Путь к файлу ответа (альтернатива response_text). */
  response?: string;
  /** Сырой текст ответа NotebookLM — вставленный в textarea. Приоритетнее response. */
  response_text?: string;
  output: string;
  transcript?: string | null;
}

export interface HighlightsImportResult {
  output: string;
  highlights_count: number;
  /** Обогащённые highlights (с stream_start_s/stream_end_s). */
  highlights: Highlight[];
  /** Проблемные элементы, отброшенные при толерантном парсинге. */
  warnings?: string[];
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

// --- highlights render / vegas markers ---------------------------------------

export interface RenderHighlightsRequest {
  decisions: string;
  input_path: string;
  out_dir: string;
  quality?: "fast" | "balanced" | "max";
  padding_s?: number;
  force?: boolean;
}

export interface HighlightClip {
  index: number;
  title: string;
  start_s: number;
  end_s: number;
  path: string;
}

export interface RenderHighlightsResult {
  out_dir: string;
  clips: HighlightClip[];
  count: number;
}

export function createRenderHighlightsJob(
  req: RenderHighlightsRequest
): Promise<JobState> {
  return request("/jobs/render-highlights", {
    method: "POST",
    body: JSON.stringify(req),
  });
}

export interface ExportVegasHighlightsRequest {
  decisions: string;
  vegas: string;
}

export function createExportVegasHighlightsJob(
  req: ExportVegasHighlightsRequest
): Promise<JobState> {
  return request("/jobs/export-vegas-highlights", {
    method: "POST",
    body: JSON.stringify(req),
  });
}

// --- detect-silence ---------------------------------------------------------

/**
 * Запрос на детект тишины. Времена/пороги — в секундах и дБ (см.
 * DetectSilenceRequest в schemas.py). Все поля кроме video_path имеют
 * дефолты на бэке — шлём только те, что реально хотим переопределить.
 */
export interface DetectSilenceRequest {
  video_path: string;
  threshold_db?: number;
  min_duration?: number;
  merge_gap?: number;
  padding_before?: number;
  padding_after?: number;
  workdir?: string | null;
  force?: boolean;
}

/**
 * Результат detect-silence job (state.result по завершении). Сегменты — пары
 * [start, end] в секундах в координатах video_path (audio-space исходника).
 */
export interface DetectSilenceResult {
  video_path: string;
  segments: [number, number][];
  count: number;
  total_duration: number;
}

export function createDetectSilenceJob(
  req: DetectSilenceRequest
): Promise<JobState> {
  return request("/jobs/detect-silence", {
    method: "POST",
    body: JSON.stringify(req),
  });
}

// --- render -----------------------------------------------------------------

/**
 * Запрос на рендер: вырезать segments_to_remove ([start,end] в секундах) из
 * input_path и записать в output_path. encoder=null → бэкенд выберет сам.
 */
export interface RenderRequest {
  input_path: string;
  output_path: string;
  segments_to_remove: [number, number][];
  /** [start,end] в секундах — заглушить аудио в ноль, кадры оставить (маты). */
  segments_to_mute?: [number, number][];
  encoder?: string | null;
  audio_bitrate?: string;
  /** Пресет скорость↔качество энкодера. Дефолт на бэке — "balanced". */
  quality?: "fast" | "balanced" | "max";
  /** Обрезать вывод по диапазону обработки (абс. секунды исходника). */
  range_start?: number | null;
  range_end?: number | null;
  force?: boolean;
}

export interface RenderResult {
  input_path: string;
  output_path: string;
  output_size: number;
  removed_count: number;
  muted_count?: number;
  kept_duration: number;
}

export function createRenderJob(req: RenderRequest): Promise<JobState> {
  return request("/jobs/render", {
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

/**
 * Готовый промпт для NotebookLM (кнопка «Скопировать промпт» в AI Tiktok).
 * n_highlights — сколько моментов просить у модели (бэкенд валидирует 1..50).
 */
export function getPrompt(
  nHighlights = 5
): Promise<{ prompt: string; n_highlights: number }> {
  return request(`/prompt?n_highlights=${nHighlights}`);
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
  /** Настоящий тайтл стрима с Twitch, если сохранён в decisions.json. UI
   *  предпочитает его имени папки vod_<id>. */
  title?: string | null;
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

/**
 * Удалить проект из реестра недавних. По умолчанию убирает только запись
 * (файлы остаются); deleteFiles=true — ещё и удаляет папку проекта с диска.
 */
export function deleteProject(
  decisionsPath: string,
  deleteFiles = false
): Promise<{ unregistered: boolean; deleted_dir: string | null; error: string | null }> {
  const qs = `?path=${encodeURIComponent(decisionsPath)}&delete_files=${deleteFiles}`;
  return request(`/projects${qs}`, { method: "DELETE" });
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
 * Уровень чувствительности к мату из настроек — для запуска обработки.
 *
 * Best-effort: настройки не прочитались (бэкенд ещё не поднялся, файла нет) →
 * null, и запрос уходит без поля, а бэкенд берёт свой `balanced`. Вынесено сюда,
 * потому что обработка запускается из двух мест — формы новой обработки и
 * авто-цепочки после скачивания VOD, — и настройка обязана действовать в обоих.
 */
export function getDefaultSensitivity(): Promise<string | null> {
  return getSettings()
    .then((s) =>
      typeof s.default_sensitivity === "string" && s.default_sensitivity
        ? s.default_sensitivity
        : null
    )
    .catch(() => null);
}

// --- AI: провайдеры / шаблоны / детекция оффтопа ----------------------------
//
// БЕЗОПАСНОСТЬ: api_key НИКОГДА не проходит через renderer. GET-эндпоинты
// (providers/templates) секрета не несут — идут напрямую через request().
// А testConnection / detectOfftopic требуют ключ: renderer передаёт только
// `api_key_ref`, а расшифровку и вставку api_key делает Electron main
// (window.twitchCut.aiTestConnection / aiDetectOfftopic → loopback-POST).
// Поэтому эти две обёртки идут через IPC-мост, а не через request().

/** Провайдер ИИ (зеркалит llm_offtopic/providers.py AIProvider). Без секрета. */
export interface AiProvider {
  id: string;
  label: string;
  format: "anthropic" | "openai";
  base_url?: string | null;
  models: string[];
  /** Символическая ссылка на ключ в safeStorage (НЕ сам ключ). */
  api_key_ref?: string | null;
  /**
   * Схема авторизации для anthropic-формата: "x-api-key" (нативный Anthropic)
   * или "bearer" (agentrouter и прокси на Messages API). Меняет только
   * заголовок — ключ тот же, ставит его SDK. Для openai не используется.
   */
  auth_scheme?: "x-api-key" | "bearer";
  /** Доп. HTTP-заголовки к каждому запросу (default_headers у SDK). Без секрета. */
  headers?: Record<string, string> | null;
  /**
   * Держать ли SSE-стрим. По умолчанию true. false — у провайдеров с битым
   * стримом (agentrouter: delta:null/billing.summary ломают парсер) — берётся
   * один цельный не-стримовый ответ.
   */
  stream?: boolean;
  builtin: boolean;
}

/** Шаблон промпта (зеркалит llm_offtopic/templates.py PromptTemplate). */
export interface AiTemplate {
  id: string;
  label: string;
  kind: "topic" | "classification";
  text: string;
  builtin: boolean;
}

/** Оффтоп-сегмент (зеркалит llm_offtopic/model.py OfftopicSegment + stream_*). */
export interface OfftopicSegment {
  /** Границы в LOCAL-секундах (от начала транскрипта), как mutes[i].start. */
  start_s: number;
  end_s: number;
  confidence: number;
  reason: string;
  topic?: string | null;
  quote?: string | null;
  /** Audit: модель/ступень каскада, вынёсшие вердикт. */
  model?: string | null;
  stage?: string | null;
  /** Секунд видео МОЛЧАЛО в сегменте (путь source; 0 = играло всю реплику). */
  pause_s?: number | null;
  /** Доля слов сегмента, привязанных к исходнику 0..1 (путь source). */
  anchored?: number | null;
  /** Абсолютные секунды исходника — проставляет merger (stream_offset). */
  stream_start_s?: number;
  stream_end_s?: number;
}

/**
 * Тело detect-offtopic (зеркалит schemas.py DetectOfftopicRequest), НО без
 * `api_key`: renderer шлёт `api_key_ref`, main достаёт ключ и вставляет api_key.
 */
export interface DetectOfftopicRequest {
  decisions: string;
  output: string;
  transcript?: string | null;
  provider_id?: string;
  provider?: Record<string, unknown> | null;
  /** Ссылка на ключ в safeStorage; main расшифрует и подставит api_key. */
  api_key_ref?: string | null;
  cheap_model?: string | null;
  expensive_model?: string | null;
  /** [source/single-pass] Модель LLM; None → дефолт провайдера. */
  model?: string | null;
  /** null|source|single-pass|windows (см. schemas.py DetectOfftopicRequest). */
  method?: string | null;
  /** Локальный файл исходника — распознаём и используем как ground truth. */
  source_audio?: string | null;
  /** Готовый JSON транскрипта исходника (вместо распознавания source_audio). */
  source_transcript?: string | null;
  /** Модель ASR исходника; должна совпадать с моделью стрима. */
  source_model?: string | null;
  /** Устройство ASR исходника (cpu безопаснее). */
  source_device?: string | null;
  topic_template?: Record<string, unknown> | null;
  classification_template?: Record<string, unknown> | null;
  window_s?: number | null;
  overlap_s?: number | null;
  chunk_s?: number | null;
  merge_gap_s?: number | null;
  cascade_threshold?: number | null;
}

/** Тело test-connection (без api_key — только ref, main подставит ключ). */
export interface TestConnectionRequest {
  provider_id?: string;
  provider?: Record<string, unknown> | null;
  api_key_ref?: string | null;
}

/** Результат detect-offtopic job (state.result по завершении). */
export interface DetectOfftopicResult {
  output: string;
  segments_count: number;
  total_offtopic_s: number;
  offtopic: OfftopicSegment[];
}

/** Встроенные + пользовательские провайдеры (без секретов). */
export function getAiProviders(): Promise<{ builtin: AiProvider[]; custom: AiProvider[] }> {
  return request("/ai/providers");
}

/** Встроенные + пользовательские шаблоны промптов. */
export function getAiTemplates(): Promise<{ builtin: AiTemplate[]; custom: AiTemplate[] }> {
  return request("/ai/templates");
}

/**
 * Проверить соединение с провайдером. Идёт через Electron main (ключ по
 * api_key_ref достаётся и вставляется main-side; renderer ключ не держит).
 * Возвращает распарсенный ответ backend { ok, detail } или бросает Error.
 */
export async function testAiConnection(
  req: TestConnectionRequest
): Promise<{ ok: boolean; detail: string }> {
  const res = await window.twitchCut.aiTestConnection(req as unknown as Record<string, unknown>);
  if (!res.ok) {
    // res.ok=false тут = транспортная/IPC-ошибка (не «плохой ключ» — тот
    // приходит в data с HTTP 200 и { ok:false, detail }).
    throw new Error(res.error || `test-connection failed (HTTP ${res.status ?? "?"})`);
  }
  return res.data as { ok: boolean; detail: string };
}

/**
 * Запустить джобу AI-детекции оффтопа. Идёт через Electron main (ключ
 * инжектится main-side). Возвращает снапшот джобы (JobState) — дальше UI
 * подписывается через subscribeJob(state.id).
 */
export async function createDetectOfftopicJob(
  req: DetectOfftopicRequest
): Promise<JobState> {
  const res = await window.twitchCut.aiDetectOfftopic(req as unknown as Record<string, unknown>);
  if (!res.ok) {
    const data = res.data as { detail?: unknown } | null;
    const detail = data?.detail ? JSON.stringify(data.detail) : res.error;
    throw new Error(detail || `detect-offtopic failed (HTTP ${res.status ?? "?"})`);
  }
  return res.data as JobState;
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
  /**
   * Настоящие границы контура по бакетам (min ≤ 0 ≤ max). Асимметричный min/max
   * даёт узнаваемую «рваную» форму Audacity; зеркальный ±peaks — гладкий блоб.
   * Опционально: кэши format<3 их не содержат → фронт падает на `peaks`.
   */
  min?: number[];
  max?: number[];
  /**
   * RMS-энергия по тем же бакетам, что и peaks (0..1). Рисуется как яркое ядро
   * внутри контура пика — фирменный вид Audacity. Опционально: старые кэши
   * (format<2) его не содержат, фронт тогда рисует один контур.
   */
  rms?: number[];
  duration_s: number;
  sample_rate: number;
  source: string;
}

/**
 * «Слой детали» — peaks ТОЛЬКО для видимого окна (см. /waveform/range).
 *
 * `start_s`/`end_s` — ФАКТИЧЕСКИ покрытый диапазон: у конца файла ffmpeg отдаёт
 * меньше, чем просили, и рисовать нужно по этим границам, иначе волна съедет.
 */
export interface WaveformRange {
  peaks: number[];
  min: number[];
  max: number[];
  rms: number[];
  start_s: number;
  end_s: number;
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
 * Получить peaks ТОЛЬКО для окна [startS, endS] — «слой детали» при зуме.
 *
 * Так работают Audacity/Reaper: чем глубже зум, тем мельче запрошенный диапазон,
 * поэтому крошка волны видна на любом масштабе (базовые 8192 точки на весь
 * стрим при зуме в 5 секунд дали бы ~20 точек = гладкий блоб).
 *
 * НЕ кэшируется на бэкенде (комбинаций бесконечно) — декод узкого слайса стоит
 * десятки ms. Вызывающий обязан дебаунсить и отменять устаревшие запросы через
 * `signal`, иначе быстрый зум-жест наплодит очередь.
 */
export function getWaveformRange(
  streamPath: string,
  startS: number,
  endS: number,
  res: number = 2048,
  signal?: AbortSignal
): Promise<WaveformRange> {
  const qs =
    `path=${encodeURIComponent(streamPath)}` +
    `&start=${startS.toFixed(6)}&end=${endS.toFixed(6)}&res=${res}`;
  return request(`/waveform/range?${qs}`, { signal });
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
 * Полный URL к /thumbnail для `<img src="...">` в карточке проекта.
 *
 * Синхронный (как mediaUrl) — <img> нужен src в первый render. Возвращает null,
 * если base URL ещё не резолвился: карточка тогда показывает Film-плейсхолдер
 * до готовности бэкенда. При 404 (исходник удалён) <img> сам стриггерит onError.
 */
export function thumbnailUrl(streamPath: string, width = 640): string | null {
  if (!_cachedBase) return null;
  return `${_cachedBase}/thumbnail?path=${encodeURIComponent(streamPath)}&w=${width}`;
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
  let receivedFinal = false;
  let currentWs: WebSocket | null = null;
  let reconnectTimer: number | null = null;
  let attempt = 0;

  const isTerminal = (ev: JobEvent): boolean =>
    ev.type === "final" ||
    ev.state?.status === "done" ||
    ev.state?.status === "failed" ||
    ev.state?.status === "cancelled";

  const connect = () => {
    if (closedByUser || receivedFinal) return;
    const ws = new WebSocket(wsUrl);
    currentWs = ws;
    ws.onopen = () => {
      // Успешное соединение сбрасывает backoff.
      attempt = 0;
    };
    ws.onmessage = (msg) => {
      try {
        const parsed = JSON.parse(msg.data) as JobEvent;
        // Джоба уже терминальна: бэкенд отдаёт финальный snapshot и закрывает
        // сокет. Больше событий не будет — глушим авто-reconnect, иначе клиент
        // в бесконечном цикле пере-подключается к завершённой джобе (open/close
        // спам в логах backend'а).
        if (isTerminal(parsed)) {
          receivedFinal = true;
        }
        onEvent(parsed);
      } catch (err) {
        onError?.(err as Error);
      }
    };
    ws.onerror = (e) => onError?.(e);
    ws.onclose = () => {
      currentWs = null;
      if (closedByUser || receivedFinal) return;
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
