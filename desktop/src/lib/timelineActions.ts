import { useEffect, useSyncExternalStore } from "react";

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
 * очищает их при unmount'e. Меняющиеся значения (dirty/canUndo/canRedo)
 * подтягиваются в реальном времени — deps'ы стандартные.
 */
export function usePublishTimelineActions(actions: TimelineActions) {
  useEffect(() => {
    setTimelineActions(actions);
    return () => setTimelineActions(null);
  }, [actions.onSave, actions.onUndo, actions.onRedo, actions.canUndo, actions.canRedo, actions.dirty]);
}
