import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import pkg from "./package.json" with { type: "json" };

// Vite config для React renderer.
// - `base: "./"` — итоговые пути в index.html становятся относительными,
//   иначе Electron `file://` не найдёт /assets/*.js после `npm run build`.
// - Dev-порт 5173 фиксирован — Electron main.ts грепает его в NODE_ENV=development.
// - VITE_APP_VERSION — подставляется в SettingsScreen. Читаем из package.json,
//   чтобы не дублировать строку и не забывать бампить руками.
export default defineConfig({
  plugins: [react()],
  base: "./",
  define: {
    "import.meta.env.VITE_APP_VERSION": JSON.stringify(pkg.version),
  },
  server: {
    port: 5173,
    strictPort: true,
  },
  build: {
    outDir: "dist",
    sourcemap: true,
    // Electron умеет ES2022 → не транспилируем в древний ES.
    target: "es2022",
  },
});
