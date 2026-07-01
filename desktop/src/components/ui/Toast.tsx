import { Toaster as SonnerToaster, toast } from "sonner";

/**
 * Обёртка над sonner с настроенной тёмной темой.
 * `toast()` re-export'ится — импортируй `import { toast } from ".../Toast"`.
 */
export function Toaster() {
  return (
    <SonnerToaster
      theme="dark"
      position="bottom-right"
      richColors
      closeButton
      toastOptions={{
        className:
          "!bg-elevated !border !border-white/10 !text-fg !rounded-lg !shadow-card",
      }}
    />
  );
}

export { toast };
