import { type ReactNode } from "react";
import { cn } from "../../lib/cn";
import { Card } from "./Card";

/**
 * StatCard — компактный KPI-плашка для Dashboard / Job-header'а.
 *
 * Дизайн: icon-chip брендового оттенка + числовое значение крупным моно-шрифтом
 * с tabular-nums (чтобы прыгающие цифры не перерисовывали layout) + label
 * приглушённым цветом. Опциональный `hint` под label — контекст (напр.
 * «за 7 дней»).
 *
 * Всё в одну строку по вертикали, чтобы 3–4 таких карточки красиво уложились
 * в grid из 3–4 колонок без переносов.
 */

export interface StatCardProps {
  icon?: ReactNode;
  label: string;
  value: ReactNode;
  hint?: ReactNode;
  /** Опциональный delta-tag справа (напр. «+3 за неделю»). */
  trend?: ReactNode;
  className?: string;
}

export function StatCard({
  icon,
  label,
  value,
  hint,
  trend,
  className,
}: StatCardProps) {
  return (
    <Card variant="surface" padding="md" className={cn("flex flex-col gap-3", className)}>
      <div className="flex items-start justify-between gap-3">
        <div className="flex items-center gap-2 text-xs font-medium uppercase tracking-wider text-subtle">
          {icon && (
            <span className="flex h-6 w-6 items-center justify-center rounded-md bg-brand/10 text-brand-from">
              {icon}
            </span>
          )}
          <span>{label}</span>
        </div>
        {trend && <div className="text-[11px] text-muted">{trend}</div>}
      </div>
      <div className="font-mono-tabular text-2xl font-semibold text-fg leading-none">
        {value}
      </div>
      {hint && <div className="text-xs text-muted">{hint}</div>}
    </Card>
  );
}
