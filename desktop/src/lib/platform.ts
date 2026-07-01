/**
 * Platform-detect для UI-подсказок клавиш.
 *
 * Хоткеи-рантайм (useHotkey) уже кроссплатформенный: комбо "mod+k" резолвится
 * в Meta на Mac и Ctrl на Windows/Linux. Но в UI-строках у нас захардкожены
 * Mac-глифы ⌘ ⇧ ⌥ ⏎ — на Windows это выглядит как «⌘K» рядом с надписью
 * «⌘K», по которой Ctrl+K ничего не делает интуитивно.
 *
 * Единый выход: одна функция `platformizeShortcut(raw)` разбирает строку,
 * подменяет модификаторы, возвращает то что рендерится в <Kbd>. Все
 * hardcoded «⌘K» в компонентах остаются как есть — они превращаются в
 * «Ctrl+K» уже на выводе.
 *
 * Детект: navigator.platform + userAgent — в Electron всегда доступен.
 * Кешируем в модульной переменной, платформа не меняется в рантайме.
 */

function detectIsMac(): boolean {
  // Electron: navigator.platform валидный ("MacIntel" | "Win32" | "Linux x86_64").
  // Guard на SSR / node-only контексты (тесты).
  if (typeof navigator === "undefined") return false;
  const p = navigator.platform || "";
  if (p.startsWith("Mac")) return true;
  // Fallback на userAgent — на новых macOS platform может быть "MacIntel"
  // всегда, но лишний sanity-check не помешает.
  const ua = navigator.userAgent || "";
  return /Mac|iPhone|iPad|iPod/.test(ua);
}

export const IS_MAC = detectIsMac();

/** Клавиша-модификатор для рендера: "⌘" на Mac, "Ctrl" везде. */
export const MOD_KEY = IS_MAC ? "⌘" : "Ctrl";

/** Shift: на Mac глиф ⇧, на PC — слово. */
export const SHIFT_KEY = IS_MAC ? "⇧" : "Shift";

/** Alt/Option. */
export const ALT_KEY = IS_MAC ? "⌥" : "Alt";

/** Enter/Return. */
export const ENTER_KEY = IS_MAC ? "⏎" : "Enter";

/**
 * Преобразовать строку-подсказку в платформенно-корректный вид.
 *
 * Правила замены:
 *   ⌘ → "⌘" на Mac, "Ctrl+" на Win/Linux
 *   ⇧ → "⇧" на Mac, "Shift+" на Win/Linux
 *   ⌥ → "⌥" на Mac, "Alt+"   на Win/Linux
 *   ⏎ → "⏎"   на Mac, "Enter" на Win/Linux
 *
 * После модификатора «+» ставится только если следом идёт другой символ.
 * "⌘⇧Z" → "Ctrl+Shift+Z", "⏎" → "Enter", "⌘⏎" → "Ctrl+Enter".
 *
 * На Mac всё возвращается без изменений — глифы уже правильные.
 */
export function platformizeShortcut(raw: string): string {
  if (IS_MAC) return raw;
  // Обрабатываем по одному символу, склеиваем модификаторы через "+".
  const out: string[] = [];
  for (const ch of raw) {
    switch (ch) {
      case "⌘":
        out.push("Ctrl");
        break;
      case "⇧":
        out.push("Shift");
        break;
      case "⌥":
        out.push("Alt");
        break;
      case "⏎":
        out.push("Enter");
        break;
      default:
        out.push(ch);
    }
  }
  // Склейка: между модификаторами и следующим символом ставим "+".
  // Простая эвристика: если предыдущий токен — слово-модификатор (Ctrl/Shift/Alt),
  // а следующий не пустой и не сам модификатор, вставляем "+".
  const MODS = new Set(["Ctrl", "Shift", "Alt"]);
  const parts: string[] = [];
  for (let i = 0; i < out.length; i++) {
    parts.push(out[i]);
    if (i < out.length - 1 && MODS.has(out[i])) {
      parts.push("+");
    }
  }
  return parts.join("");
}
