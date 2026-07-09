import { forwardRef, type InputHTMLAttributes, type ReactNode } from "react";
import { Check } from "lucide-react";
import { cn } from "../../lib/cn";

/**
 * Checkbox — управляемый чекбокс в стиле проекта. Нативный <input type=checkbox>
 * спрятан (sr-only) для доступности/клавиатуры, поверх — свой квадрат с галочкой.
 * Клик по всей области (обёртка <label>) переключает.
 *
 * Отмеченное состояние — brand-заливка. Используется для «Заглушить (в ноль)»
 * в раскрытой строке заглушки: галочка = звук режется на этом тайминге.
 */

export interface CheckboxProps
  extends Omit<InputHTMLAttributes<HTMLInputElement>, "type" | "size"> {
  label?: ReactNode;
  /** Подпись помельче под основной меткой. */
  hint?: ReactNode;
}

export const Checkbox = forwardRef<HTMLInputElement, CheckboxProps>(
  ({ className, label, hint, checked, disabled, ...props }, ref) => (
    <label
      className={cn(
        "inline-flex cursor-pointer select-none items-center gap-2 text-sm",
        disabled && "cursor-not-allowed opacity-50",
        className
      )}
    >
      <span className="relative inline-flex h-4 w-4 shrink-0 items-center justify-center">
        <input
          ref={ref}
          type="checkbox"
          checked={checked}
          disabled={disabled}
          className="peer sr-only"
          {...props}
        />
        <span
          className={cn(
            "flex h-4 w-4 items-center justify-center rounded border transition-colors",
            "border-white/20 bg-black/30",
            "peer-checked:border-brand-from peer-checked:bg-brand-from",
            "peer-focus-visible:ring-2 peer-focus-visible:ring-brand-from/40"
          )}
        >
          {checked && <Check className="h-3 w-3 text-white" strokeWidth={3} />}
        </span>
      </span>
      {(label || hint) && (
        <span className="flex flex-col leading-tight">
          {label && <span className="text-fg">{label}</span>}
          {hint && <span className="text-[11px] text-subtle">{hint}</span>}
        </span>
      )}
    </label>
  )
);
Checkbox.displayName = "Checkbox";
