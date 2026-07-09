/**
 * Точная схема decisions.json / transcript.json — как их реально пишет backend
 * (см. backend/src/twitch_cut/decisions.py::build_decisions и gigaam_asr.py).
 *
 * ВАЖНО про единицы: в отличие от старого `types/decisions.ts` (там были
 * выдуманные `start_ms`), настоящий файл хранит время в СЕКУНДАХ и поля
 * называются `start`/`end` (локальные, от range_in) + `stream_start`/`stream_end`
 * (абсолютные, в исходном stream). `word` — строка (одно/несколько matched-слов),
 * `status` — судьба мьюта в экспорте.
 *
 * Эти типы использует новый экран «Правка» (ревью матов). Старый
 * `types/decisions.ts` оставлен для Dashboard/Export, которым важна только длина
 * массивов.
 */

/** Статус мьюта. Vegas-экспорт глушит только `accepted`; всё остальное пропускает. */
export type MuteStatus = "accepted" | "rejected" | "review" | string;

/** Один мьют из decisions.json (реальная схема 1.1). Лишние поля сохраняем as-is. */
export interface MuteRecord {
  id: string;
  /** Локальные секунды (от range_in) — совпадают с координатами transcript и cache-audio. */
  start: number;
  end: number;
  /** Абсолютные секунды в исходном stream — то, что читает Vegas-экспорт. */
  stream_start?: number;
  stream_end?: number;
  /** Пойманное слово/фраза (matched-токены через пробел). */
  word?: string;
  status?: MuteStatus;
  reason?: string;
  banword?: string;
  confidence?: number | null;
  needs_review?: boolean;
  segment_id?: string;
  /** Всё, что не перечислено выше, переживает round-trip без потерь. */
  [k: string]: unknown;
}

/** Кэши путей (аудио-извлечение диапазона + транскрипт). */
export interface DecisionsCaches {
  audio?: string | null;
  transcript?: string | null;
  [k: string]: unknown;
}

/** Корневой документ decisions.json. Незнакомые поля сохраняем. */
export interface DecisionsDoc {
  schema_version?: string;
  /** Путь к исходному stream (полный файл). */
  source?: string;
  /** Legacy-ключ на случай старых файлов. */
  stream?: string;
  original?: string;
  range_in?: string;
  range_out?: string;
  mutes?: MuteRecord[];
  cuts?: unknown[];
  highlights?: unknown;
  caches?: DecisionsCaches;
  _meta?: { stream_path?: string; project?: string };
  [k: string]: unknown;
}

// --- transcript --------------------------------------------------------------

/** Слово с пословным таймингом (секунды). score у GigaAM может быть null. */
export interface TranscriptWord {
  word: string;
  start: number;
  end: number;
  score?: number | null;
}

export interface TranscriptSegment {
  start: number;
  end: number;
  text?: string;
  words?: TranscriptWord[];
}

export interface TranscriptDoc {
  segments?: TranscriptSegment[];
  word_segments?: TranscriptWord[];
  [k: string]: unknown;
}
