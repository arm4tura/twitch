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
        // break-words + фикс. ширина: без этого длинный путь/эхо ошибки без
        // пробелов растягивает тост во весь экран.
        className:
          "!bg-elevated !border !border-white/10 !text-fg !rounded-lg !shadow-card !max-w-[380px] [&_[data-description]]:!break-words [&_[data-title]]:!break-words",
      }}
    />
  );
}

export { toast };
