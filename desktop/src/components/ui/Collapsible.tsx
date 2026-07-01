import * as CollapsiblePrimitive from "@radix-ui/react-collapsible";
import { ChevronDown } from "lucide-react";
import { cn } from "../../lib/cn";

/**
 * Collapsible — обёртка над Radix для секций типа «Продвинутые параметры».
 * Триггер — кнопка с шевроном, который поворачивается в data-[state=open].
 * Контент анимируется через utility-класс `animate-collapsible` (см. styles.css).
 */

export const Collapsible = CollapsiblePrimitive.Root;

export function CollapsibleTrigger({
  className,
  children,
  ...props
}: React.ComponentProps<typeof CollapsiblePrimitive.Trigger>) {
  return (
    <CollapsiblePrimitive.Trigger
      className={cn(
        "group flex w-full items-center justify-between gap-2 rounded-lg px-3 py-2 " +
          "text-sm text-muted transition-colors hover:bg-white/5 hover:text-fg " +
          "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-brand-from",
        className
      )}
      {...props}
    >
      {children}
      <ChevronDown className="h-4 w-4 shrink-0 transition-transform duration-200 group-data-[state=open]:rotate-180" />
    </CollapsiblePrimitive.Trigger>
  );
}

export function CollapsibleContent({
  className,
  ...props
}: React.ComponentProps<typeof CollapsiblePrimitive.Content>) {
  return (
    <CollapsiblePrimitive.Content
      className={cn(
        "overflow-hidden data-[state=open]:animate-collapsible-down data-[state=closed]:animate-collapsible-up",
        className
      )}
      {...props}
    />
  );
}
