// Дизайн-токены Twitch Cut Desktop.
//
// ВАЖНО: файл именно .cjs, не .ts — Tailwind 3.4 подгружает TS-конфиги через
// внутренний jiti, и под Windows этот путь иногда молча возвращает пустой
// объект (content: [] → purge выкашивает все классы, итоговый CSS ~1.5 KB).
// CJS парсится напрямую Node'ом и работает одинаково на всех платформах.
//
// Стратегия: zinc-палитра + violet→indigo brand-gradient. Все токены сидят в
// CSS-переменных `:root` (см. styles.css), чтобы позже подключить light-theme
// без пересборки Tailwind. Sem-цвета (ok/warn/err) — тонкие обёртки над
// emerald/amber/rose для читаемости (`text-ok` вместо `text-emerald-500`).
//
// darkMode "class" — включаем `.dark` на <html> и потом сможем добавить
// light-toggle без правки конфига. Пока рендерим всегда dark.
/** @type {import('tailwindcss').Config} */
module.exports = {
  darkMode: "class",
  content: ["./index.html", "./src/**/*.{ts,tsx,js,jsx}"],
  theme: {
    extend: {
      colors: {
        // Поверхности — через var(), значения в styles.css.
        bg: "rgb(var(--bg) / <alpha-value>)",
        surface: "rgb(var(--surface) / <alpha-value>)",
        elevated: "rgb(var(--elevated) / <alpha-value>)",
        border: "rgb(var(--border) / <alpha-value>)",
        // Текст.
        fg: "rgb(var(--fg) / <alpha-value>)",
        muted: "rgb(var(--muted) / <alpha-value>)",
        subtle: "rgb(var(--subtle) / <alpha-value>)",
        // Семантика — прямые токены.
        ok: "rgb(var(--ok) / <alpha-value>)",
        warn: "rgb(var(--warn) / <alpha-value>)",
        err: "rgb(var(--err) / <alpha-value>)",
        // Brand — используется в accent-градиенте, focus ring, active nav-item.
        brand: {
          from: "rgb(var(--brand-from) / <alpha-value>)",
          to: "rgb(var(--brand-to) / <alpha-value>)",
          DEFAULT: "rgb(var(--brand-from) / <alpha-value>)",
        },
      },
      fontFamily: {
        sans: [
          "Inter Variable",
          "Inter",
          "system-ui",
          "-apple-system",
          "Segoe UI",
          "Roboto",
          "sans-serif",
        ],
        mono: [
          "JetBrains Mono",
          "ui-monospace",
          "Cascadia Mono",
          "Menlo",
          "monospace",
        ],
      },
      borderRadius: {
        card: "10px",
      },
      boxShadow: {
        card:
          "0 1px 0 rgb(255 255 255 / 0.03) inset, 0 20px 40px -20px rgb(0 0 0 / 0.6)",
        glow: "0 0 0 1px rgb(139 92 246 / 0.5), 0 8px 24px -8px rgb(139 92 246 / 0.35)",
      },
      backgroundImage: {
        brand:
          "linear-gradient(135deg, rgb(var(--brand-from)) 0%, rgb(var(--brand-to)) 100%)",
        "top-glow":
          "linear-gradient(to bottom, rgb(255 255 255 / 0.08), rgb(255 255 255 / 0) 40%)",
      },
      animation: {
        "pulse-dot": "pulse-dot 1.6s ease-in-out infinite",
        "fade-in": "fade-in 150ms ease-out",
        "fade-out": "fade-out 120ms ease-in",
        "slide-up": "slide-up 200ms cubic-bezier(0.16, 1, 0.3, 1)",
        "collapsible-down": "collapsible-down 200ms ease-out",
        "collapsible-up": "collapsible-up 200ms ease-out",
      },
      keyframes: {
        "pulse-dot": {
          "0%, 100%": { opacity: "1", transform: "scale(1)" },
          "50%": { opacity: "0.5", transform: "scale(0.9)" },
        },
        "fade-in": {
          from: { opacity: "0" },
          to: { opacity: "1" },
        },
        "fade-out": {
          from: { opacity: "1" },
          to: { opacity: "0" },
        },
        "slide-up": {
          from: { opacity: "0", transform: "translateY(6px)" },
          to: { opacity: "1", transform: "translateY(0)" },
        },
        "collapsible-down": {
          from: { height: "0" },
          to: { height: "var(--radix-collapsible-content-height)" },
        },
        "collapsible-up": {
          from: { height: "var(--radix-collapsible-content-height)" },
          to: { height: "0" },
        },
      },
    },
  },
  plugins: [],
};
