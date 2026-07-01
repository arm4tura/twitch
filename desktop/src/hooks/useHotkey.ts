import { useEffect, useRef } from "react";

/**
 * useHotkey — глобальная привязка комбинации к обработчику.
 *
 * Комбинации: 'mod+z', 'mod+shift+z', ' ' (space), 'escape', 'k', 'meta+k'.
 * `mod` = Cmd на mac, Ctrl везде ещё. Регистр не важен.
 *
 * По умолчанию игнорируем, если фокус внутри <input>/<textarea>/contenteditable
 * — иначе space в поле поиска будет ставить/снимать playback. Флаг
 * `allowInInput` перезаписывает поведение (нужно для ⌘K, чтобы можно было
 * вызвать палитру даже если курсор в текстовом поле).
 *
 * `deps` пробрасываются в useEffect — так же как в стандартных hook'ах.
 */

export type HotkeyHandler = (e: KeyboardEvent) => void;

export interface HotkeyOptions {
  allowInInput?: boolean;
  /** Событие вызывается на element'e (по умолчанию — window). */
  target?: HTMLElement | Window | Document | null;
  /** Отключить на время (динамика без ре-регистрации). */
  enabled?: boolean;
}

function isEditable(el: EventTarget | null): boolean {
  if (!(el instanceof HTMLElement)) return false;
  const tag = el.tagName;
  if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT") return true;
  if (el.isContentEditable) return true;
  return false;
}

function matches(e: KeyboardEvent, combo: string): boolean {
  const parts = combo.toLowerCase().split("+").map((s) => s.trim());
  const needsShift = parts.includes("shift");
  const needsAlt = parts.includes("alt") || parts.includes("option");
  const needsMod = parts.includes("mod") || parts.includes("cmd") || parts.includes("ctrl") || parts.includes("meta");
  const key = parts[parts.length - 1];

  const hasMod = e.metaKey || e.ctrlKey;
  if (needsMod && !hasMod) return false;
  if (!needsMod && hasMod) return false;
  if (needsShift !== e.shiftKey) return false;
  if (needsAlt !== e.altKey) return false;

  const evKey = e.key.toLowerCase();
  // Спецкейсы для читаемости хоткея.
  if (key === "space") return evKey === " ";
  if (key === "esc") return evKey === "escape";
  return evKey === key;
}

export function useHotkey(
  combo: string | string[],
  handler: HotkeyHandler,
  opts: HotkeyOptions = {}
): void {
  // Хендлер в ref — не перепривязываем listener при каждом ре-рендере родителя.
  const handlerRef = useRef(handler);
  handlerRef.current = handler;
  const enabled = opts.enabled !== false;

  useEffect(() => {
    if (!enabled) return;
    const target = (opts.target ?? window) as EventTarget;
    const combos = Array.isArray(combo) ? combo : [combo];
    const listener = (ev: Event) => {
      const e = ev as KeyboardEvent;
      if (!opts.allowInInput && isEditable(e.target)) return;
      for (const c of combos) {
        if (matches(e, c)) {
          handlerRef.current(e);
          return;
        }
      }
    };
    target.addEventListener("keydown", listener);
    return () => target.removeEventListener("keydown", listener);
  }, [Array.isArray(combo) ? combo.join("|") : combo, opts.allowInInput, opts.target, enabled]);
}
