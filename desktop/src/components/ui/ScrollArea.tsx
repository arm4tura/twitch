import * as ScrollAreaPrimitive from "@radix-ui/react-scroll-area";
import { cn } from "../../lib/cn";

/**
 * ScrollArea — стилизованный скролл-контейнер поверх Radix. Использую для
 * лога job-экрана и для длинного списка регионов на Timeline (в след. коммитах).
 *
 * Скроллбар — тонкая полоска brand-цвета, появляется по hover'у.
 * У корневого элемента ВСЕГДА нужна явная высота — Radix не даёт её сам,
 * иначе скролл никогда не сработает.
 */

export function ScrollArea({
  className,
  children,
  ...props
}: React.ComponentProps<typeof ScrollAreaPrimitive.Root>) {
  return (
    <ScrollAreaPrimitive.Root
      className={cn("relative overflow-hidden", className)}
      {...props}
    >
      <ScrollAreaPrimitive.Viewport className="h-full w-full rounded-[inherit]">
        {children}
      </ScrollAreaPrimitive.Viewport>
      <ScrollBar />
      <ScrollAreaPrimitive.Corner />
    </ScrollAreaPrimitive.Root>
  );
}

function ScrollBar({
  orientation = "vertical",
}: {
  orientation?: "vertical" | "horizontal";
}) {
  return (
    <ScrollAreaPrimitive.Scrollbar
      orientation={orientation}
      className={cn(
        "flex touch-none select-none transition-colors",
        orientation === "vertical"
          ? "h-full w-1.5 border-l border-l-transparent p-[1px]"
          : "h-1.5 flex-col border-t border-t-transparent p-[1px]"
      )}
    >
      <ScrollAreaPrimitive.Thumb className="relative flex-1 rounded-full bg-white/15 transition-colors hover:bg-white/25" />
    </ScrollAreaPrimitive.Scrollbar>
  );
}
