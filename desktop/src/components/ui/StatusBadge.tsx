import { cva, type VariantProps } from "class-variance-authority";
import type { JobStatus } from "../../api";
import { cn } from "../../lib/cn";

const badgeStyles = cva(
  "inline-flex items-center gap-1.5 rounded-full px-2.5 py-0.5 text-[11px] font-medium uppercase tracking-wider",
  {
    variants: {
      status: {
        pending: "bg-white/5 text-muted",
        running:
          "bg-brand-from/15 text-brand-from ring-1 ring-brand-from/30",
        done: "bg-ok/15 text-ok ring-1 ring-ok/30",
        failed: "bg-err/15 text-err ring-1 ring-err/30",
        cancelled: "bg-warn/15 text-warn ring-1 ring-warn/30",
      },
    },
    defaultVariants: { status: "pending" },
  }
);

const RU: Record<JobStatus, string> = {
  pending: "ожидание",
  running: "работает",
  done: "готово",
  failed: "ошибка",
  cancelled: "отменено",
};

export interface StatusBadgeProps
  extends VariantProps<typeof badgeStyles> {
  status: JobStatus;
  className?: string;
  /** Опциональный подписи-override (иначе — из RU-map). */
  label?: string;
}

export function StatusBadge({ status, label, className }: StatusBadgeProps) {
  return (
    <span className={cn(badgeStyles({ status }), className)}>
      {status === "running" && (
        <span className="h-1.5 w-1.5 rounded-full bg-brand-from animate-pulse-dot" />
      )}
      {label ?? RU[status]}
    </span>
  );
}
