import { useCallback } from "react";
import type { Undoable } from "./useUndoable";
import { snapToWordBoundary, type WordIndex } from "../lib/transcript";
import { KEPT, MIN_MUTE_S, SILENCED, type UiMute } from "../lib/mutes";

/**
 * useMuteEditing — чистые правки списка заглушек поверх undoable-стора.
 *
 * Всё здесь — правки границ/статуса ОДНОГО мьюта, без UI-побочек (без выделения,
 * центрирования окна, тостов — это оркестрирует экран).
 *
 * Чего здесь НЕТ и быть не должно: правки `word`. `word` — то, что ПОЙМАЛ
 * детектор (matched-токены), и сдвиг окна его не меняет. Раньше слово-подпись
 * пересчитывалась из транскрипта на каждую правку границ, и человек, поправив
 * окно, затирал слово детектора накрытым текстом («бля» → «с этим»). Затёртое
 * уезжало в decisions.json, в журнал правок и в audit.md, где мьют уже назывался
 * чужим словом, — то есть правка окна уничтожала единственную запись о том, что
 * тут вообще нашли. Реально накрытый текст — величина ПРОИЗВОДНАЯ от границ и
 * транскрипта, поэтому он не хранится, а считается при показе (MuteInspector,
 * из тех же слов, что подсвечены красным).
 *
 * Координаты — ЛОКАЛЬНЫЕ секунды (как хранит UiMute). Индекс слов приходит
 * ref'ом: транскрипт грузится асинхронно и может смениться.
 */
export interface MuteEditing {
  /** Точечный patch одного мьюта (низкоуровневый — им пользуются экран и хелперы ниже). */
  updateMute: (id: string, patch: Partial<UiMute>) => void;
  /** Переключить глушим/оставляем (снимает флаг review). */
  toggleMuted: (m: UiMute) => void;
  /** Клик по слову в транскрипте: снап границ к границам слов. */
  adjustMute: (id: string, start: number, end: number) => void;
  /** Точный сдвиг границ на дельту (сек), БЕЗ снапа — для кнопок −0.1/+0.1с. */
  nudgeMute: (id: string, dStart: number, dEnd: number) => void;
  /** Свободная правка границ ТОЧНО как задал пользователь (драг по волне), БЕЗ снапа. */
  adjustMuteFree: (id: string, start: number, end: number) => void;
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
      updateMute(id, { start: s, end: e });
    },
    [updateMute, wordIndexRef]
  );

  const nudgeMute = useCallback(
    (id: string, dStart: number, dEnd: number) => {
      const m = store.present.find((x) => x.id === id);
      if (!m) return;
      let s = Math.max(0, m.start + dStart);
      let e = m.end + dEnd;
      if (e - s < MIN_MUTE_S) {
        // не даём краю перескочить через противоположный: упираем в MIN.
        if (dEnd !== 0) e = s + MIN_MUTE_S;
        else s = e - MIN_MUTE_S;
      }
      updateMute(id, { start: s, end: e });
    },
    [store, updateMute]
  );

  const adjustMuteFree = useCallback(
    (id: string, start: number, end: number) => {
      let s = Math.max(0, start);
      let e = end;
      if (e - s < MIN_MUTE_S) e = s + MIN_MUTE_S;
      updateMute(id, { start: s, end: e });
    },
    [updateMute]
  );

  return { updateMute, toggleMuted, adjustMute, nudgeMute, adjustMuteFree };
}
