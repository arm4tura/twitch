import { useCallback } from "react";
import type { Undoable } from "./useUndoable";
import {
  snapToWordBoundary,
  wordsInRange,
  type WordIndex,
} from "../lib/transcript";
import { KEPT, MIN_MUTE_S, SILENCED, type UiMute } from "../lib/mutes";

/**
 * useMuteEditing — чистые правки списка заглушек поверх undoable-стора.
 *
 * Всё здесь — правки границ/статуса/слова-подписи ОДНОГО мьюта, без UI-побочек
 * (без выделения, центрирования окна, тостов — это оркестрирует экран). Слово-
 * подпись всегда пересчитывается из транскрипта по новым границам, чтобы
 * инспектор показывал реально накрытый текст.
 *
 * Координаты — ЛОКАЛЬНЫЕ секунды (как хранит UiMute). Индекс слов приходит
 * ref'ом: транскрипт грузится асинхронно и может смениться.
 */
export interface MuteEditing {
  /** Точечный patch одного мьюта (низкоуровневый — им пользуются экран и хелперы ниже). */
  updateMute: (id: string, patch: Partial<UiMute>) => void;
  /** Переключить глушим/оставляем (снимает флаг review). */
  toggleMuted: (m: UiMute) => void;
  /** Клик по слову в транскрипте: снап границ к словам + пересчёт слова-подписи. */
  adjustMute: (id: string, start: number, end: number) => void;
  /** Точный сдвиг границ на дельту (сек), БЕЗ снапа — для кнопок −0.1/+0.1с. */
  nudgeMute: (id: string, dStart: number, dEnd: number) => void;
  /** Свободная правка границ ТОЧНО как задал пользователь (драг по волне), БЕЗ снапа. */
  adjustMuteFree: (id: string, start: number, end: number) => void;
}

function coveredWord(idx: WordIndex, s: number, e: number): string {
  return wordsInRange(idx, s, e)
    .map((w) => w.word)
    .join(" ")
    .trim();
}

export function useMuteEditing(
  store: Undoable<UiMute[]>,
  wordIndexRef: React.MutableRefObject<WordIndex>
): MuteEditing {
  const updateMute = useCallback(
    (id: string, patch: Partial<UiMute>) => {
      store.set((prev) => prev.map((m) => (m.id === id ? { ...m, ...patch } : m)));
    },
    [store]
  );

  const toggleMuted = useCallback(
    (m: UiMute) => {
      updateMute(m.id, { status: m.status === SILENCED ? KEPT : SILENCED, review: false });
    },
    [updateMute]
  );

  const adjustMute = useCallback(
    (id: string, start: number, end: number) => {
      const idx = wordIndexRef.current;
      const s = snapToWordBoundary(idx, start);
      const e = snapToWordBoundary(idx, end);
      if (e - s < MIN_MUTE_S) return; // защита от схлопывания
      const covered = coveredWord(idx, s, e);
      updateMute(id, { start: s, end: e, word: covered || undefined });
    },
    [updateMute, wordIndexRef]
  );

  const nudgeMute = useCallback(
    (id: string, dStart: number, dEnd: number) => {
      const m = store.present.find((x) => x.id === id);
      if (!m) return;
      const idx = wordIndexRef.current;
      let s = Math.max(0, m.start + dStart);
      let e = m.end + dEnd;
      if (e - s < MIN_MUTE_S) {
        // не даём краю перескочить через противоположный: упираем в MIN.
        if (dEnd !== 0) e = s + MIN_MUTE_S;
        else s = e - MIN_MUTE_S;
      }
      const covered = coveredWord(idx, s, e);
      updateMute(id, { start: s, end: e, word: covered || undefined });
    },
    [store, updateMute, wordIndexRef]
  );

  const adjustMuteFree = useCallback(
    (id: string, start: number, end: number) => {
      const idx = wordIndexRef.current;
      let s = Math.max(0, start);
      let e = end;
      if (e - s < MIN_MUTE_S) e = s + MIN_MUTE_S;
      const covered = coveredWord(idx, s, e);
      updateMute(id, { start: s, end: e, word: covered || undefined });
    },
    [updateMute, wordIndexRef]
  );

  return { updateMute, toggleMuted, adjustMute, nudgeMute, adjustMuteFree };
}
