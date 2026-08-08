import { useEffect, useRef, useSyncExternalStore } from "react";

/**
 * Глобальный реестр «текущих действий Timeline'а» для CommandPalette.
 *
 * Проблема: палитра ⌘K живёт в App.tsx, а handler'ы save/undo/redo — внутри
 * TimelineScreen. Пробрасывать пропсы вверх через onMount-callback хрупко
 * (StrictMode дважды монтирует). Здесь простой store с subscribe/getSnapshot —
 * Timeline на маунте вызывает setActions(...), палитра подписывается
 * useSyncExternalStore. При unmount'e — reset обратно к пустоте.
 *
 * Cовместимо с server-rendering (не используем в SSR).
 */

export interface TimelineActions {
  onSave?: () => void;
  onUndo?: () => void;
  onRedo?: () => void;
  canUndo: boolean;
  canRedo: boolean;
  dirty: boolean;
}

const EMPTY: TimelineActions = { canUndo: false, canRedo: false, dirty: false };

let current: TimelineActions = EMPTY;
const listeners = new Set<() => void>();

function emit() {
  for (const l of listeners) l();
}

export function setTimelineActions(next: TimelineActions | null) {
  current = next ?? EMPTY;
  emit();
}

function subscribe(cb: () => void) {
  listeners.add(cb);
  return () => {
    listeners.delete(cb);
  };
}

function getSnapshot() {
  return current;
}

export function useTimelineActions(): TimelineActions {
  return useSyncExternalStore(subscribe, getSnapshot, getSnapshot);
}

/**
 * Хук для TimelineScreen: публикует свои handler'ы в глобальный store и
 * очищает их при unmount'e.
 *
 * ВАЖНО про идентичность: `save`/`undo`/`redo` часто пересоздаются на каждый
 * рендер (их useCallback зависит от sliceUndoable-объектов, которые не
 * мемоизируются). Если гонять их через deps эффекта, получаем цикл:
 *   publish → emit → useTimelineActions в App ре-рендерит App → TimelineScreen
 *   ре-рендерит → новые handler-идентичности → эффект снова → publish → …
 *   → «Maximum update depth exceeded».
 * Поэтому свежие handler'ы держим в ref (без эффекта), а публикуем ТОЛЬКО когда
 * меняются примитивные флаги canUndo/canRedo/dirty. Так emit срабатывает лишь
 * на реальное изменение состояния кнопок, а не на каждую смену идентичности.
 */
export function usePublishTimelineActions(actions: TimelineActions) {
  const ref = useRef(actions);
  ref.current = actions;

  useEffect(() => {
    // Публикуем стабильные обёртки, читающие свежие handler'ы через ref, +
    // текущие примитивные флаги. Пересобираем только на смену флагов.
    setTimelineActions({
      onSave: () => ref.current.onSave?.(),
      onUndo: () => ref.current.onUndo?.(),
      onRedo: () => ref.current.onRedo?.(),
      canUndo: actions.canUndo,
      canRedo: actions.canRedo,
      dirty: actions.dirty,
    });
    return () => setTimelineActions(null);
  }, [actions.canUndo, actions.canRedo, actions.dirty]);
}
