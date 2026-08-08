import * as DropdownMenuPrimitive from "@radix-ui/react-dropdown-menu";
import { cn } from "../../lib/cn";

/**
 * DropdownMenu — тонкая обёртка над @radix-ui/react-dropdown-menu в стиле
 * проекта (см. Dialog.tsx — тот же паттерн Portal + анимированный Content).
 *
 * Используется для контекстного меню карточки проекта на Dashboard'е:
 * триггер «⋯» → пункт «Удалить» (variant="danger"). Radix берёт на себя
 * фокус-менеджмент, Escape, клик-вне и клавиатурную навигацию по пунктам.
 */

export const DropdownMenu = DropdownMenuPrimitive.Root;
export const DropdownMenuTrigger = DropdownMenuPrimitive.Trigger;

export function DropdownMenuContent({
  className,
  align = "end",
  sideOffset = 4,
  ...props
}: React.ComponentProps<typeof DropdownMenuPrimitive.Content>) {
  return (
    <DropdownMenuPrimitive.Portal>
      <DropdownMenuPrimitive.Content
        align={align}
        sideOffset={sideOffset}
        className={cn(
          "z-50 min-w-[10rem] overflow-hidden rounded-card border border-white/10 " +
            "bg-surface p-1 shadow-card animate-slide-up",
          className
        )}
        {...props}
      />
    </DropdownMenuPrimitive.Portal>
  );
}

export function DropdownMenuItem({
  className,
  variant = "default",
  ...props
}: React.ComponentProps<typeof DropdownMenuPrimitive.Item> & {
  variant?: "default" | "danger";
}) {
  return (
    <DropdownMenuPrimitive.Item
      className={cn(
        "flex cursor-pointer select-none items-center gap-2 rounded-md px-2.5 py-2 text-sm " +
          "outline-none transition-colors data-[disabled]:pointer-events-none data-[disabled]:opacity-50",
        variant === "danger"
          ? "text-err data-[highlighted]:bg-err/15 data-[highlighted]:text-err"
          : "text-fg data-[highlighted]:bg-white/[0.06]",
        className
      )}
      {...props}
    />
  );
}

export function DropdownMenuSeparator({
  className,
  ...props
}: React.ComponentProps<typeof DropdownMenuPrimitive.Separator>) {
  return (
    <DropdownMenuPrimitive.Separator
      className={cn("my-1 h-px bg-white/10", className)}
      {...props}
    />
  );
}
