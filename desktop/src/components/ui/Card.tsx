import { forwardRef, type HTMLAttributes } from "react";
import { cva, type VariantProps } from "class-variance-authority";
import { cn } from "../../lib/cn";

/**
 * Card — базовая поверхность. Три уровня:
 * - surface: приглушённый zinc-900 фон, чуть светлее body.
 * - elevated: заметная тень + внутренний highlight, для CTA-карточек Dashboard.
 * - glass: полупрозрачный zinc + backdrop-blur, для sidebar / toolbar.
 */
const cardStyles = cva("rounded-card border transition-colors", {
  variants: {
    variant: {
      surface: "bg-surface/60 border-white/5",
      elevated: "bg-surface border-white/[0.06] shadow-card",
      glass: "bg-surface/40 border-white/5 backdrop-blur-xl",
    },
    padding: {
      none: "p-0",
      sm: "p-3",
      md: "p-5",
      lg: "p-6",
    },
  },
  defaultVariants: { variant: "surface", padding: "md" },
});

export interface CardProps
  extends HTMLAttributes<HTMLDivElement>,
    VariantProps<typeof cardStyles> {}

export const Card = forwardRef<HTMLDivElement, CardProps>(
  ({ className, variant, padding, ...props }, ref) => (
    <div
      ref={ref}
      className={cn(cardStyles({ variant, padding, className }))}
      {...props}
    />
  )
);
Card.displayName = "Card";

export function CardHeader({ className, ...props }: HTMLAttributes<HTMLDivElement>) {
  return (
    <div
      className={cn("mb-4 flex items-start justify-between gap-3", className)}
      {...props}
    />
  );
}

export function CardTitle({ className, ...props }: HTMLAttributes<HTMLHeadingElement>) {
  return (
    <h3
      className={cn("text-base font-semibold text-fg tracking-tight", className)}
      {...props}
    />
  );
}

export function CardDescription({
  className,
  ...props
}: HTMLAttributes<HTMLParagraphElement>) {
  return <p className={cn("text-sm text-muted", className)} {...props} />;
}
