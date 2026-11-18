import { useCallback } from "react";
import type { Undoable } from "./useUndoable";

/**
 * useOfftopicEditing — чистые правки списка оффтоп-сегментов поверх undoable-стора.
 *
 * Аналог useSilenceEditing, но для AI-детекции оффтопа. Каждый сегмент, помимо
 * границ и статуса, несёт метаданные классификатора (confidence/reason/topic/
 * quote/model/stage) — они read-only для пользователя, но обязаны переживать
 * accept/reject/resize (иначе инспектор потеряет «почему это оффтоп»).
 *
 * Все правки идут через `store.set` — accept/reject/resize/delete/add попадают
 * в общую историю undo/redo рядом с мьютами и тишиной (⌘Z/⌘⇧Z ходят по одному
 * стеку через сам Undoable). UI-побочки (выделение, тосты) — оркестрирует экран.
 *
 * Координаты — LOCAL-секунды (от range_in), как mutes[i].start (НЕ audio-секунды
 * как у silence): бэкенд отдаёт start_s/end_s именно в local-времени.
 */

export const MIN_OFFTOPIC_S = 0.1; // минимальная длительность валидного сегмента
export type OfftopicStatus = "accepted" | "rejected";

export interface OfftopicSegmentUi {
  id: string;
  start: number; // local-секунды (от range_in)
  end: number; // local-секунды
  status: OfftopicStatus;
  // --- метаданные классификатора (read-only для пользователя) ---
  confidence: number;
  reason: string;
  topic?: string | null;
  quote?: string | null;
  model?: string | null;
  stage?: string | null;
  // Улики пути по исходнику (см. model.py::OfftopicSegment). Заполнены только
  // когда сегмент найден по исходнику; иначе null. pauseS — сколько секунд видео
  // молчало в сегменте, anchored — доля слов, привязанных к исходнику (0..1).
  pauseS?: number | null;
  anchored?: number | null;
}

export interface OfftopicEditing {
  /** Точечный patch одного сегмента (низкоуровневый). */
  updateOne: (id: string, patch: Partial<OfftopicSegmentUi>) => void;
  /** Свободная правка границ (драг края/тела блока), с клэмпом в MIN_OFFTOPIC_S. */
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
  addSegment: (start: number, end: number, status?: OfftopicStatus) => string | null;
  /**
   * Полностью заменить список (результат AI-детекции). Одна запись в историю
   * undo — весь набор откатывается/возвращается ⌘Z разом. `keepManual` сохраняет
   * ранее добавленные вручную сегменты поверх свежего детекта.
   */
  replaceAll: (
    segments: Array<{
      start: number;
      end: number;
      status?: OfftopicStatus;
      id?: string;
      confidence?: number;
      reason?: string;
      topic?: string | null;
      quote?: string | null;
      model?: string | null;
      stage?: string | null;
    }>,
    opts?: { keepManual?: boolean }
  ) => void;
}

export function useOfftopicEditing(
  store: Undoable<OfftopicSegmentUi[]>
): OfftopicEditing {
  const updateOne = useCallback(
    (id: string, patch: Partial<OfftopicSegmentUi>) => {
      store.set((prev) => prev.map((s) => (s.id === id ? { ...s, ...patch } : s)));
    },
    [store]
  );

  const resize = useCallback(
    (id: string, newStart: number, newEnd: number) => {
      const start = Math.max(0, newStart);
      const end = Math.max(start + MIN_OFFTOPIC_S, newEnd);
      if (end - start < MIN_OFFTOPIC_S) return;
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
    (start: number, end: number, status: OfftopicStatus = "accepted") => {
      const s = Math.max(0, start);
      const e = Math.max(s + MIN_OFFTOPIC_S, end);
      if (e - s < MIN_OFFTOPIC_S) return null;
      const id = `offtopic_manual_${Math.round(s * 1000)}`;
      // Ручной сегмент несёт нейтральные метаданные — пользователь сам знает почему.
      const newSeg: OfftopicSegmentUi = {
        id,
        start: s,
        end: e,
        status,
        confidence: 1,
        reason: "Добавлено вручную",
        topic: null,
        quote: null,
        model: null,
        stage: "manual",
      };
      store.set((prev) => [...prev, newSeg].sort((a, b) => a.start - b.start));
      return id;
    },
    [store]
  );

  const replaceAll = useCallback(
    (
      segments: Array<{
        start: number;
        end: number;
        status?: OfftopicStatus;
        id?: string;
        confidence?: number;
        reason?: string;
        topic?: string | null;
        quote?: string | null;
        model?: string | null;
        stage?: string | null;
        pauseS?: number | null;
        anchored?: number | null;
      }>,
      opts?: { keepManual?: boolean }
    ) => {
      const clean: OfftopicSegmentUi[] = segments
        .map((seg) => {
          const s = Math.max(0, seg.start);
          const e = Math.max(s + MIN_OFFTOPIC_S, seg.end);
          return {
            id: seg.id ?? `offtopic_auto_${Math.round(s * 1000)}`,
            start: s,
            end: e,
            status: seg.status ?? "accepted",
            confidence: seg.confidence ?? 0,
            reason: seg.reason ?? "",
            topic: seg.topic ?? null,
            quote: seg.quote ?? null,
            model: seg.model ?? null,
            stage: seg.stage ?? null,
            pauseS: seg.pauseS ?? null,
            anchored: seg.anchored ?? null,
          };
        })
        .filter((seg) => seg.end - seg.start >= MIN_OFFTOPIC_S);

      store.set((prev) => {
        // keepManual — не затираем ручные (id начинается с offtopic_manual_).
        const manual = opts?.keepManual
          ? prev.filter((s) => s.id.startsWith("offtopic_manual_"))
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
