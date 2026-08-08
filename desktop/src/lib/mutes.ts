import type { MuteRecord } from "../types/project";

/**
 * Общая модель заглушки для экрана правки + хуков редактирования.
 *
 * UiMute — UI-модель мьюта: локальные секунды + судьба. Исходную запись
 * (raw) храним для round-trip при сохранении (stream_start/end и прочие поля).
 */
export interface UiMute {
  id: string;
  start: number;
  end: number;
  word: string;
  status: string;
  review: boolean;
  raw: MuteRecord;
}

export const SILENCED = "accepted"; // статус, который Vegas реально глушит
export const KEPT = "rejected"; // «оставить» — Vegas пропускает
export const MIN_MUTE_S = 0.03; // единый минимум длины мьюта (волна/кнопки/клик)

export function toUiMute(m: MuteRecord): UiMute {
  const status = typeof m.status === "string" ? m.status : SILENCED;
  return {
    id: m.id,
    start: Number(m.start) || 0,
    end: Number(m.end) || 0,
    word: typeof m.word === "string" ? m.word : "",
    status,
    review: status === "review" || !!m.needs_review,
    raw: m,
  };
}
