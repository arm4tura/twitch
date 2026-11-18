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

/**
 * Один silence-сегмент из decisions.json. Пишется после ревью тишины на
 * таймлайне: `accepted` → вырежется при рендере, `rejected` → останется.
 * Времена — audio-секунды исходника (те же координаты, что и waveform/media).
 */
export interface SilenceRecord {
  id: string;
  start: number;
  end: number;
  status: "accepted" | "rejected";
  /** Всё, что не перечислено выше, переживает round-trip без потерь. */
  [k: string]: unknown;
}

/**
 * Один оффтоп-сегмент из decisions["offtopic"].segments (реальная схема —
 * backend/src/twitch_cut/llm_offtopic/model.py::OfftopicSegment + stream_*_s,
 * которые проставляет merger.py). Границы `start_s`/`end_s` — LOCAL-секунды
 * (от range_in), как у mutes[i].start; `stream_*_s` — абсолютные в исходнике.
 * Лишние поля сохраняем as-is (round-trip).
 */
export interface OfftopicSegment {
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
  [k: string]: unknown;
}

/**
 * Бандл AI-детекции оффтопа в decisions.json (ключ "offtopic"). Пишет
 * merger.py при merge результата джобы detect-offtopic. Пустой segments —
 * легитимный результат («оффтопа не найдено»).
 */
export interface OfftopicBundle {
  segments: OfftopicSegment[];
  /** Id провайдера/модели или 'mock' (audit trail). */
  source?: string;
  /** Хеш транскрипта — рассинхрон = сегменты стухли. */
  transcript_hash?: string;
  model?: string;
  /** Путь классификации, породивший бандл (см. model.py::OfftopicSet.method). */
  method?: string;
  /** Хеш транскрипта исходника (путь source). */
  source_transcript_hash?: string;
  /** Модель ASR исходника (путь source). */
  source_model?: string;
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
  /** Настоящий тайтл стрима с Twitch (если включён тумблер переименования и его
   *  вытащил детект оффтопа). Приоритетнее имени папки vod_<id> в UI. */
  title?: string;
  /** Путь к исходному stream (полный файл). */
  source?: string;
  /** Legacy-ключ на случай старых файлов. */
  stream?: string;
  original?: string;
  range_in?: string;
  range_out?: string;
  mutes?: MuteRecord[];
  /** Обнаруженная тишина после ревью (audio-секунды). Пишет TimelineScreen. */
  silences?: SilenceRecord[];
  /** AI-детекция оффтопа (бандл segments + метаданные). Пишет detect-offtopic. */
  offtopic?: OfftopicBundle;
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
