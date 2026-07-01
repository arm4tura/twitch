import { useCallback, useMemo, useRef, useState } from "react";

/**
 * useUndoable — стек undo/redo произвольного immutable-состояния.
 *
 * Юзкейс — Timeline: mutes/cuts/highlights массивы. Каждый пользовательский
 * edit (drag region, delete, split) кладёт новый snapshot в стек. ⌘Z/⌘⇧Z
 * ходят по стеку. Стек ограничен: слишком длинная история отгрызла бы RAM
 * при 500+ регионах, а UX-выгода уже иллюзорна.
 *
 * Дизайн:
 * - Храним ВЕСЬ снапшот, не diff. Для тысячи регионов это ~50 KB — приемлемо.
 * - Push игнорируется если новое значение === текущему (referential equality)
 *   — редьюсеры вызывают set часто, но не всегда меняют содержимое.
 * - `reset(next)` очищает историю и делает next начальным — используется
 *   при загрузке нового decisions.json.
 * - `dirty` — есть ли изменения с момента последнего mark().
 */

export interface Undoable<T> {
  present: T;
  set: (next: T | ((prev: T) => T)) => void;
  undo: () => void;
  redo: () => void;
  reset: (next: T) => void;
  /** Отметить текущее состояние как «сохранённое» — `dirty` станет false. */
  mark: () => void;
  canUndo: boolean;
  canRedo: boolean;
  dirty: boolean;
  size: number;
}

const DEFAULT_LIMIT = 50;

export function useUndoable<T>(initial: T, limit: number = DEFAULT_LIMIT): Undoable<T> {
  // past — самое старое сверху, новое снизу. present — отдельно. future — redo-стек.
  const [past, setPast] = useState<T[]>([]);
  const [present, setPresent] = useState<T>(initial);
  const [future, setFuture] = useState<T[]>([]);
  // Индекс «сохранённого» состояния относительно past.length; -1 = ничего не сохраняли.
  const savedAtRef = useRef<number>(0);

  const set = useCallback(
    (next: T | ((prev: T) => T)) => {
      setPresent((prev) => {
        const value = typeof next === "function" ? (next as (p: T) => T)(prev) : next;
        if (Object.is(value, prev)) return prev;
        setPast((p) => {
          const appended = [...p, prev];
          // Кэп размера — если превысили limit, откусываем от начала и
          // сдвигаем saved-указатель, иначе он «уплывёт» в отрицательный.
          if (appended.length > limit) {
            const trim = appended.length - limit;
            savedAtRef.current -= trim;
            return appended.slice(trim);
          }
          return appended;
        });
        setFuture([]); // любое новое действие обнуляет redo-стек
        return value;
      });
    },
    [limit]
  );

  const undo = useCallback(() => {
    setPast((p) => {
      if (p.length === 0) return p;
      const prev = p[p.length - 1];
      setFuture((f) => [present, ...f]);
      setPresent(prev);
      return p.slice(0, -1);
    });
  }, [present]);

  const redo = useCallback(() => {
    setFuture((f) => {
      if (f.length === 0) return f;
      const next = f[0];
      setPast((p) => [...p, present]);
      setPresent(next);
      return f.slice(1);
    });
  }, [present]);

  const reset = useCallback((next: T) => {
    setPast([]);
    setFuture([]);
    setPresent(next);
    savedAtRef.current = 0;
  }, []);

  const mark = useCallback(() => {
    savedAtRef.current = past.length;
  }, [past.length]);

  return useMemo(
    () => ({
      present,
      set,
      undo,
      redo,
      reset,
      mark,
      canUndo: past.length > 0,
      canRedo: future.length > 0,
      dirty: past.length !== savedAtRef.current,
      size: past.length + 1 + future.length,
    }),
    [present, set, undo, redo, reset, mark, past.length, future.length]
  );
}
