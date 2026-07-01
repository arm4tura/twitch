import { forwardRef, type ButtonHTMLAttributes } from "react";
import { Slot } from "@radix-ui/react-slot";
import { cva, type VariantProps } from "class-variance-authority";
import { Loader2 } from "lucide-react";
import { cn } from "../../lib/cn";

/**
 * Button — 4 варианта × 3 размера, поддержка `asChild` (для <a> с кнопочным
 * стилем через Radix Slot) и `loading` со спиннером-иконкой поверх контента.
 *
 * Primary — brand-градиент с top-highlight для edge-lit эффекта.
 * Secondary — surface + border.
 * Ghost — прозрачный, hover:surface.
 * Destructive — красный, для «Удалить» / «Отменить джобу».
 */
const buttonStyles = cva(
  "relative inline-flex items-center justify-center gap-2 whitespace-nowrap rounded-lg font-medium " +
    "transition-[background,transform,box-shadow,color] duration-150 ease-out " +
    "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-brand-from focus-visible:ring-offset-2 focus-visible:ring-offset-bg " +
    "disabled:pointer-events-none disabled:opacity-50 " +
    "active:scale-[0.98]",
  {
    variants: {
      variant: {
        primary:
          "bg-brand text-white shadow-[0_1px_0_rgb(255_255_255/0.15)_inset,0_10px_25px_-10px_rgb(139_92_246/0.5)] " +
            "hover:brightness-110 hover:shadow-glow",
        secondary:
          "bg-surface/60 text-fg border border-white/10 backdrop-blur " +
            "hover:bg-surface hover:border-white/20",
        ghost: "text-muted hover:text-fg hover:bg-white/5",
        destructive:
          "bg-err/90 text-white hover:bg-err shadow-[0_1px_0_rgb(255_255_255/0.1)_inset]",
      },
      size: {
        sm: "h-8 px-3 text-xs",
        md: "h-10 px-4 text-sm",
        lg: "h-12 px-6 text-[15px]",
        icon: "h-9 w-9 p-0",
      },
    },
    defaultVariants: { variant: "primary", size: "md" },
  }
);

export interface ButtonProps
  extends ButtonHTMLAttributes<HTMLButtonElement>,
    VariantProps<typeof buttonStyles> {
  asChild?: boolean;
  loading?: boolean;
}

export const Button = forwardRef<HTMLButtonElement, ButtonProps>(
  ({ className, variant, size, asChild, loading, disabled, children, ...props }, ref) => {
    const Comp: any = asChild ? Slot : "button";
    return (
      <Comp
        ref={ref}
        className={cn(buttonStyles({ variant, size, className }))}
        disabled={disabled ?? loading}
        {...props}
      >
        {loading && (
          <span className="absolute inset-0 flex items-center justify-center">
            <Loader2 className="h-4 w-4 animate-spin" />
          </span>
        )}
        <span className={cn("inline-flex items-center gap-2", loading && "opacity-0")}>
          {children}
        </span>
      </Comp>
    );
  }
);
Button.displayName = "Button";
