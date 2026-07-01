import { cn } from "../../lib/cn";
import { platformizeShortcut } from "../../lib/platform";

/**
 * <Kbd>⌘K</Kbd> — визуальные клавиши для подсказок. Моно-шрифт, тёмный chip.
 *
 * ВАЖНО: строку children пропускаем через `platformizeShortcut`, чтобы
 * hardcoded Mac-глифы (⌘ ⇧ ⌥ ⏎) на Windows/Linux превратились в
 * Ctrl/Shift/Alt/Enter. Всё что не строка — рендерим как есть (например,
 * `<Kbd>↑↓</Kbd>` для стрелок, которые одинаковы на всех платформах).
 */
export function Kbd({
  children,
  className,
}: {
  children: React.ReactNode;
  className?: string;
}) {
  const content =
    typeof children === "string" ? platformizeShortcut(children) : children;
  return (
    <kbd
      className={cn(
        "inline-flex h-5 min-w-[20px] items-center justify-center rounded border border-white/10 " +
          "bg-white/5 px-1 font-mono-tabular text-[10px] text-muted",
        className
      )}
    >
      {content}
    </kbd>
  );
}
