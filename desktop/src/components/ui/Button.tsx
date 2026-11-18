import { forwardRef, type ButtonHTMLAttributes, type ElementType } from "react";
import { Slot } from "@radix-ui/react-slot";
import { cva, type VariantProps } from "class-variance-authority";
import { Loader2 } from "lucide-react";
import { cn } from "../../lib/cn";

/**
 * Button — 4 варианта × 3 размера, поддержка `asChild` (для <a> с кнопочным
 * стилем через Radix Slot) и `loading` со спиннером-иконкой поверх контента.
 *
 * Primary — однотонный фирменный акцент.
 * Secondary — surface + border.
 * Ghost — прозрачный, hover:surface.
 * Destructive — красный, для «Удалить» / «Отменить джобу».
 */
const buttonStyles = cva(
  "relative inline-flex items-center justify-center gap-2 whitespace-nowrap rounded-md font-medium " +
    "transition-[background,transform,box-shadow,color] duration-150 ease-out " +
    "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-brand-from focus-visible:ring-offset-2 focus-visible:ring-offset-bg " +
    "disabled:pointer-events-none disabled:opacity-50 " +
    "active:scale-[0.98]",
  {
    variants: {
      variant: {
        primary:
          "bg-brand text-white hover:brightness-110",
        secondary:
          "bg-elevated/50 text-fg border border-white/10 " +
            "hover:bg-surface hover:border-white/20",
        ghost: "text-muted hover:text-fg hover:bg-white/5",
        destructive:
          "bg-err/90 text-white hover:bg-err shadow-[0_1px_0_rgb(255_255_255/0.1)_inset]",
      },
      size: {
        sm: "h-8 px-3 text-xs",
        md: "h-9 px-3.5 text-sm",
        lg: "h-10 px-4 text-sm",
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
    const Comp: ElementType = asChild ? Slot : "button";
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
