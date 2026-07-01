import * as TooltipPrimitive from "@radix-ui/react-tooltip";
import { HelpCircle } from "lucide-react";
import { cn } from "../../lib/cn";

export const TooltipProvider = TooltipPrimitive.Provider;
export const Tooltip = TooltipPrimitive.Root;
export const TooltipTrigger = TooltipPrimitive.Trigger;

export function TooltipContent({
  className,
  sideOffset = 6,
  ...props
}: React.ComponentProps<typeof TooltipPrimitive.Content>) {
  return (
    <TooltipPrimitive.Portal>
      <TooltipPrimitive.Content
        sideOffset={sideOffset}
        className={cn(
          "z-50 max-w-xs rounded-md border border-white/10 bg-elevated px-2.5 py-1.5 text-xs text-fg shadow-lg " +
            "animate-fade-in data-[state=closed]:animate-fade-out",
          className
        )}
        {...props}
      />
    </TooltipPrimitive.Portal>
  );
}

/**
 * HelpTip — иконка «?» рядом с Label настройки. На hover/focus показывает
 * пояснение из `children`. Используется в NewJobScreen/SettingsScreen для
 * необрушивающих подсказок «что это за поле, когда трогать».
 *
 * Внутри — Tooltip.Trigger asChild + <button>, чтобы hover ловился и на
 * тач-устройствах через click. Не рендерим текст-заголовок — предполагается,
 * что HelpTip кладут прямо после <Label>.
 */
export function HelpTip({ children }: { children: React.ReactNode }) {
  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <button
          type="button"
          aria-label="Что это?"
          className="inline-flex h-4 w-4 items-center justify-center rounded-full text-subtle transition-colors hover:text-fg focus:outline-none focus:text-fg"
        >
          <HelpCircle className="h-3.5 w-3.5" />
        </button>
      </TooltipTrigger>
      <TooltipContent side="top">{children}</TooltipContent>
    </Tooltip>
  );
}

