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
        card: "6px",
      },
      boxShadow: {
        card:
          "none",
        glow: "none",
      },
      backgroundImage: {
        brand:
          "none",
        "top-glow":
          "linear-gradient(to bottom, rgb(255 255 255 / 0.08), rgb(255 255 255 / 0) 40%)",
      },
      animation: {
        "pulse-dot": "pulse-dot 1.6s ease-in-out infinite",
        "fade-in": "fade-in 150ms ease-out",
        "fade-out": "fade-out 120ms ease-in",
        "slide-up": "slide-up 200ms cubic-bezier(0.16, 1, 0.3, 1)",
        "dialog-in": "dialog-in 200ms cubic-bezier(0.16, 1, 0.3, 1)",
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
        // Для центрированных диалогов: центрирование живёт через
        // -translate-x-1/2 -translate-y-1/2 на элементе, поэтому keyframe
        // обязан включать эти -50% в transform — иначе на время анимации
        // transform перебивает центрирование и окно «прыгает» в центр после
        // её окончания.
        "dialog-in": {
          from: {
            opacity: "0",
            transform: "translate(-50%, -50%) translateY(6px) scale(0.98)",
          },
          to: {
            opacity: "1",
            transform: "translate(-50%, -50%) translateY(0) scale(1)",
          },
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
