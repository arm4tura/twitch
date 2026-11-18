#!/usr/bin/env node
/**
 * build-backend.js — собирает Python-бэкенд в самостоятельный каталог
 * `backend/dist/<arch>/twitch-cut-backend/` ПЕРЕД `electron-builder`.
 *
 * Зачем: установщик больше не ставит Python/venv/torch при первом запуске —
 * бэкенд едет внутри приложения готовым исполняемым файлом (PyInstaller,
 * backend/packaging/twitch-cut-backend.spec). electron-builder берёт этот
 * каталог как extraResources → resources/backend (см. package.json; `${arch}`
 * там — x64/arm64, те же имена, что process.arch у Node).
 *
 * Запускается как часть `predist` (см. package.json). В CI бэкенд собирается
 * отдельным шагом до этого скрипта — тогда он видит готовый каталог и выходит
 * (повторная сборка — минута впустую). Пересобрать принудительно: `--force`.
 *
 * Python для сборки: $TWITCH_CUT_BUILD_PYTHON, иначе backend/.venv, иначе
 * `python` из PATH. В этом окружении должны стоять зависимости из
 * requirements-win.txt / requirements-mac.txt и pyinstaller
 * (`pip install -e "backend[build]"`). PyInstaller не умеет кросс-сборку:
 * бэкенд под macOS x64 собирается на Intel-маке, под arm64 — на Apple Silicon.
 */

"use strict";

const fs = require("fs");
const path = require("path");
const { spawnSync } = require("child_process");

const BACKEND = path.resolve(__dirname, "..", "..", "backend");
const SPEC = path.join(BACKEND, "packaging", "twitch-cut-backend.spec");
const DIST = path.join(BACKEND, "dist", process.arch);
const OUT_DIR = path.join(DIST, "twitch-cut-backend");
const EXE = path.join(
  OUT_DIR,
  process.platform === "win32" ? "twitch-cut-backend.exe" : "twitch-cut-backend"
);

function findPython() {
  if (process.env.TWITCH_CUT_BUILD_PYTHON) return process.env.TWITCH_CUT_BUILD_PYTHON;
  const venv = path.join(BACKEND, ".venv");
  const venvPython =
    process.platform === "win32"
      ? path.join(venv, "Scripts", "python.exe")
      : path.join(venv, "bin", "python");
  return fs.existsSync(venvPython) ? venvPython : "python";
}

function main() {
  const force = process.argv.includes("--force");
  if (fs.existsSync(EXE) && !force) {
    console.log(`[build-backend] уже собран: ${OUT_DIR} (пересобрать: --force)`);
    return;
  }
  const python = findPython();
  console.log(`[build-backend] PyInstaller через ${python}`);
  const res = spawnSync(
    python,
    [
      "-m",
      "PyInstaller",
      "--noconfirm",
      "--clean",
      "--distpath",
      DIST,
      "--workpath",
      path.join(BACKEND, "build", `pyinstaller-${process.arch}`),
      SPEC,
    ],
    { cwd: BACKEND, stdio: "inherit" }
  );
  if (res.status !== 0) {
    console.error(
      "[build-backend] сборка упала. Нужен pyinstaller в окружении сборки: " +
        'pip install -e "backend[build]"'
    );
    process.exit(res.status || 1);
  }
  if (!fs.existsSync(EXE)) {
    console.error(`[build-backend] PyInstaller отработал, но ${EXE} нет`);
    process.exit(1);
  }
  // Дымовой тест: собранный бэкенд хотя бы стартует и отвечает на --help.
  // Ловит забытые hidden imports раньше, чем пользователь увидит сплэш с ошибкой.
  const smoke = spawnSync(EXE, ["--help"], { encoding: "utf-8" });
  if (smoke.status !== 0) {
    console.error("[build-backend] собранный бэкенд не запускается:\n" + (smoke.stderr || smoke.stdout));
    process.exit(1);
  }
  console.log(`[build-backend] готово: ${OUT_DIR}`);
}

main();
