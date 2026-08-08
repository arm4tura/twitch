import { useCallback } from "react";
import type { Undoable } from "./useUndoable";

/**
 * useSilenceEditing — чистые правки списка silence-сегментов поверх undoable-стора.
 *
 * Аналог useMuteEditing, но для обнаруженной тишины: каждый silence-сегмент
 * хранит [start, end, status, id]. Status: "accepted" → будет вырезан при рендере,
 * "rejected" → пропускаем (оставляем как есть).
 *
 * Все правки идут через `store.set` — так accept/reject/resize/delete/add
 * попадают в общую историю undo/redo рядом с мьютами (⌘Z/⌘⇧Z ходят по одному
 * стеку через сам Undoable). UI-побочки (выделение, тосты, центрирование) —
 * оркестрирует экран, здесь только чистые правки массива.
 *
 * Координаты — audio-секунды (как хранит SilenceSegment). Минимальная длина
 * валидного сегмента — MIN_SILENCE_S (0.1s, соответствует порогу silence-детекта).
 */

export const MIN_SILENCE_S = 0.1; // минимальная длительность валидного silence-сегмента
export type SilenceStatus = "accepted" | "rejected";

export interface SilenceSegment {
  id: string;
  start: number; // audio-секунды
  end: number; // audio-секунды
  status: SilenceStatus;
}

/**
 * adaptiveTrim — АДАПТИВНОЕ вырезание тишины: не режем паузы под чистую (иначе
 * речь звучит пулемётом, будто человек говорит без остановки), а оставляем
 * «дыхание» длиной maxPause + защитный отступ `pad` от речи с каждого края.
 *
 * Для каждой обнаруженной паузы:
 *  - слишком короткая (после вычета дыхания и отступов резать нечего) → оставляем;
 *  - длинная → сохраняем maxPause естественной тишины (по maxPause/2 с каждого
 *    края) ПЛЮС ещё `pad` секунд отступа, чтобы блок не заезжал на тихий хвост
 *    речи (ffmpeg по RMS иногда метит затухание/придыхание как тишину). Вырезаем
 *    только оставшуюся СЕРЕДИНУ.
 *
 * maxPause=0, pad=0 → классическое «под чистую».
 * Возвращаем сырые [start, end] в audio-секундах — их дальше нормализует replaceAll.
 */
export function adaptiveTrim(
  raw: Array<{ start: number; end: number }>,
  maxPause: number,
  pad = 0.15
): Array<{ start: number; end: number }> {
  const keep = Math.max(0, maxPause);
  const p = Math.max(0, pad);
  const inset = keep / 2 + p; // сколько тишины сохраняем у каждого края паузы
  const out: Array<{ start: number; end: number }> = [];
  for (const seg of raw) {
    const dur = seg.end - seg.start;
    // Резать имеет смысл только если после двух отступов остаётся ≥ MIN_SILENCE_S.
    if (dur <= inset * 2 + MIN_SILENCE_S) continue;
    out.push({ start: seg.start + inset, end: seg.end - inset });
  }
  return out;
}

export interface SilenceEditing {
  /** Точечный patch одного сегмента (низкоуровневый — им пользуются экран и хелперы ниже). */
  updateOne: (id: string, patch: Partial<SilenceSegment>) => void;
  /** Свободная правка границ (драг края/тела блока), с клэмпом в MIN_SILENCE_S. */
  resize: (id: string, newStart: number, newEnd: number) => void;
  /** Пометить «вырезать». */
  accept: (id: string) => void;
  /** Пометить «оставить». */
  reject: (id: string) => void;
  /** Переключить accepted↔rejected. */
  toggleStatus: (id: string) => void;
  /** Удалить сегмент. */
  deleteSegment: (id: string) => void;
  /** Добавить сегмент вручную; возвращает id созданного (или null, если схлопнулся). */
  addSegment: (start: number, end: number, status?: SilenceStatus) => string | null;
  /**
   * Полностью заменить список (результат авто-детекта). Одна запись в историю
   * undo — весь набор откатывается/возвращается ⌘Z разом. `keepManual` сохраняет
   * ранее добавленные вручную/отредактированные сегменты поверх свежего детекта.
   */
  replaceAll: (
    segments: Array<{ start: number; end: number; status?: SilenceStatus; id?: string }>,
    opts?: { keepManual?: boolean }
  ) => void;
}

export function useSilenceEditing(
  store: Undoable<SilenceSegment[]>
): SilenceEditing {
  const updateOne = useCallback(
    (id: string, patch: Partial<SilenceSegment>) => {
      store.set((prev) => prev.map((s) => (s.id === id ? { ...s, ...patch } : s)));
    },
    [store]
  );

  const resize = useCallback(
    (id: string, newStart: number, newEnd: number) => {
      const start = Math.max(0, newStart);
      const end = Math.max(start + MIN_SILENCE_S, newEnd);
      if (end - start < MIN_SILENCE_S) return;
      updateOne(id, { start, end });
    },
    [updateOne]
  );

  const accept = useCallback(
    (id: string) => updateOne(id, { status: "accepted" }),
    [updateOne]
  );

  const reject = useCallback(
    (id: string) => updateOne(id, { status: "rejected" }),
    [updateOne]
  );

  const toggleStatus = useCallback(
    (id: string) => {
      const seg = store.present.find((s) => s.id === id);
      if (!seg) return;
      updateOne(id, {
        status: seg.status === "accepted" ? "rejected" : "accepted",
      });
    },
    [store, updateOne]
  );

  const deleteSegment = useCallback(
    (id: string) => {
      store.set((prev) => prev.filter((s) => s.id !== id));
    },
    [store]
  );

  const addSegment = useCallback(
    (start: number, end: number, status: SilenceStatus = "accepted") => {
      const s = Math.max(0, start);
      const e = Math.max(s + MIN_SILENCE_S, end);
      if (e - s < MIN_SILENCE_S) return null;
      const id = `silence_manual_${Math.round(s * 1000)}`;
      const newSeg: SilenceSegment = { id, start: s, end: e, status };
      store.set((prev) =>
        [...prev, newSeg].sort((a, b) => a.start - b.start)
      );
      return id;
    },
    [store]
  );

  const replaceAll = useCallback(
    (
      segments: Array<{ start: number; end: number; status?: SilenceStatus; id?: string }>,
      opts?: { keepManual?: boolean }
    ) => {
      const clean: SilenceSegment[] = segments
        .map((seg) => {
          const s = Math.max(0, seg.start);
          const e = Math.max(s + MIN_SILENCE_S, seg.end);
          return {
            id: seg.id ?? `silence_auto_${Math.round(s * 1000)}`,
            start: s,
            end: e,
            status: seg.status ?? "accepted",
          };
        })
        .filter((seg) => seg.end - seg.start >= MIN_SILENCE_S);

      store.set((prev) => {
        // keepManual — не затираем ручные/правленые (id начинается с silence_manual_).
        const manual = opts?.keepManual
          ? prev.filter((s) => s.id.startsWith("silence_manual_"))
          : [];
        const manualIds = new Set(manual.map((s) => s.id));
        const merged = [...manual, ...clean.filter((s) => !manualIds.has(s.id))];
        return merged.sort((a, b) => a.start - b.start);
      });
    },
    [store]
  );

  return {
    updateOne,
    resize,
    accept,
    reject,
    toggleStatus,
    deleteSegment,
    addSegment,
    replaceAll,
  };
}
