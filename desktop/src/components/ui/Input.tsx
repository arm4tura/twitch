import {
  forwardRef,
  type InputHTMLAttributes,
  type ReactNode,
  type SelectHTMLAttributes,
  type TextareaHTMLAttributes,
} from "react";
import { cn } from "../../lib/cn";

const base =
  "w-full rounded-lg border bg-black/30 px-3 py-2 text-sm text-fg placeholder:text-subtle " +
  "border-white/10 transition-colors " +
  "focus:outline-none focus:border-brand-from focus:ring-2 focus:ring-brand-from/30 " +
  "disabled:opacity-50 disabled:cursor-not-allowed";

export interface InputProps extends InputHTMLAttributes<HTMLInputElement> {
  leftIcon?: ReactNode;
  rightSlot?: ReactNode;
  invalid?: boolean;
}

/**
 * Input — text/number/email. `leftIcon` вставляется абсолютным позиционированием
 * с автоматическим левым padding'ом; `rightSlot` — для кнопок «Browse…» и т.п.
 */
export const Input = forwardRef<HTMLInputElement, InputProps>(
  ({ className, leftIcon, rightSlot, invalid, ...props }, ref) => (
    <div className="relative flex w-full items-stretch">
      {leftIcon && (
        <span className="pointer-events-none absolute left-3 top-1/2 -translate-y-1/2 text-muted">
          {leftIcon}
        </span>
      )}
      <input
        ref={ref}
        className={cn(
          base,
          leftIcon && "pl-9",
          rightSlot && "pr-2",
          invalid && "border-err focus:border-err focus:ring-err/30",
          className
        )}
        {...props}
      />
      {rightSlot && <div className="ml-2 flex items-center">{rightSlot}</div>}
    </div>
  )
);
Input.displayName = "Input";

export interface TextareaProps
  extends TextareaHTMLAttributes<HTMLTextAreaElement> {
  invalid?: boolean;
}

export const Textarea = forwardRef<HTMLTextAreaElement, TextareaProps>(
  ({ className, invalid, ...props }, ref) => (
    <textarea
      ref={ref}
      className={cn(
        base,
        "min-h-[80px] resize-y font-mono-tabular text-[13px]",
        invalid && "border-err focus:border-err focus:ring-err/30",
        className
      )}
      {...props}
    />
  )
);
Textarea.displayName = "Textarea";

export interface SelectProps
  extends SelectHTMLAttributes<HTMLSelectElement> {
  invalid?: boolean;
  options: Array<{ value: string; label: string }>;
}

/**
 * Select — нативный `<select>` в стиле проекта. Не тащим radix-select ради
 * ~40 kB — dropdowns у нас с фиксированными списками (модели/языки/устройства),
 * системный вид на всех платформах уместен. Стрелка — SVG-фон (иначе на разных
 * OS выглядит по-разному).
 */
export const Select = forwardRef<HTMLSelectElement, SelectProps>(
  ({ className, invalid, options, ...props }, ref) => (
    <select
      ref={ref}
      className={cn(
        base,
        // appearance-none + свой каре, чтобы совпадало с Input.
        "appearance-none pr-8 bg-[right_0.5rem_center] bg-no-repeat",
        "bg-[url('data:image/svg+xml;utf8,<svg xmlns=%22http://www.w3.org/2000/svg%22 fill=%22none%22 viewBox=%220 0 24 24%22 stroke=%22white%22 stroke-opacity=%220.5%22><path stroke-linecap=%22round%22 stroke-linejoin=%22round%22 stroke-width=%222%22 d=%22M19 9l-7 7-7-7%22/></svg>')]",
        invalid && "border-err focus:border-err focus:ring-err/30",
        className
      )}
      {...props}
    >
      {options.map((o) => (
        // Опции рендерятся в системном layer'е — стили ограничены OS, но
        // задаём фон/цвет для темных тем Windows/Linux.
        <option key={o.value} value={o.value} className="bg-elevated text-fg">
          {o.label}
        </option>
      ))}
    </select>
  )
);
Select.displayName = "Select";


/** Label — для форм. Приглушённый цвет, sm-size, uppercase-подпись опционально. */
export function Label({
  className,
  eyebrow,
  ...props
}: React.LabelHTMLAttributes<HTMLLabelElement> & { eyebrow?: boolean }) {
  return (
    <label
      className={cn(
        "block text-sm text-muted mb-1.5",
        eyebrow &&
          "text-[11px] uppercase tracking-wider text-subtle font-medium",
        className
      )}
      {...props}
    />
  );
}
