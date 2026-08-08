#!/usr/bin/env node
/**
 * prepare-tdcli.js — раскладывает бинарники TwitchDownloaderCLI по
 * `desktop/build/tdcli/<platform>/` ПЕРЕД `electron-builder`.
 *
 * Зачем: бинарник TwitchDownloader большой и лицензируется отдельно, поэтому в
 * репозиторий он не коммитится (там только `.gitkeep`). Без этого шага dist-сборка
 * либо падала (electron-builder не любит пустые extraResources), либо
 * упаковывала пустую папку — и у пользователя VOD-downloader не работал.
 *
 * Запускается как `predist` (см. package.json), поэтому `npm run dist` сам
 * подтягивает бинарники. В dev это НЕ нужно: backend умеет докачать бинарник
 * на первом обращении (backend/src/twitch_cut/fetch_tdcli.py).
 *
 * Платформы:
 *   • по умолчанию качаем только под ХОСТ-платформу (быстро, для локального dist);
 *   • `--all` — под все четыре (нужно для CI-матрицы и mac-сборки, которая шлёт
 *     и arm64, и x64);
 *   • `--platforms=osx-arm64,win-x64` — явный список.
 *
 * Идемпотентно: если бинарник уже лежит — пропускаем.
 *
 * Только Node built-ins + системный распаковщик (unzip/tar) — новых npm-зависимостей
 * не тянем.
 */

"use strict";

const fs = require("fs");
const os = require("os");
const path = require("path");
const https = require("https");
const { execFileSync } = require("child_process");

const REPO = "lay295/TwitchDownloader";
const LATEST_API = `https://api.github.com/repos/${REPO}/releases/latest`;

// Закреплённая версия — фолбэк, если GitHub API недоступен (rate-limit и т.п.).
// Держим в синхроне с _PINNED_TAG в backend/src/twitch_cut/fetch_tdcli.py.
const PINNED_TAG = "1.55.5";
const PINNED_ASSETS = {
  "win-x64": `TwitchDownloaderCLI-${PINNED_TAG}-Windows-x64.zip`,
  "linux-x64": `TwitchDownloaderCLI-${PINNED_TAG}-Linux-x64.zip`,
  "osx-x64": `TwitchDownloaderCLI-${PINNED_TAG}-MacOS-x64.zip`,
  "osx-arm64": `TwitchDownloaderCLI-${PINNED_TAG}-MacOS-Arm64.zip`,
};
const pinnedUrl = (asset) =>
  `https://github.com/${REPO}/releases/download/${PINNED_TAG}/${asset}`;

// Матчинг asset'а по ключевым словам: (os-слова, arch-слова). Регистронезависимо.
const PLATFORM_KEYWORDS = {
  "win-x64": [["windows", "win"], ["x64", "amd64", "x86_64"]],
  "linux-x64": [["linux"], ["x64", "amd64", "x86_64"]],
  "osx-x64": [["macos", "osx", "mac"], ["x64", "amd64", "x86_64"]],
  "osx-arm64": [["macos", "osx", "mac"], ["arm64", "arm", "aarch64"]],
};

const BINARY_NAME = (plat) =>
  plat === "win-x64" ? "TwitchDownloaderCLI.exe" : "TwitchDownloaderCLI";

const BUILD_TDCLI_DIR = path.resolve(__dirname, "..", "build", "tdcli");

function hostPlatform() {
  const arch = process.arch; // 'arm64' | 'x64' | ...
  if (process.platform === "win32") return "win-x64";
  if (process.platform === "darwin")
    return arch === "arm64" ? "osx-arm64" : "osx-x64";
  return "linux-x64";
}

function parseTargets() {
  const args = process.argv.slice(2);
  if (args.includes("--all")) return Object.keys(PLATFORM_KEYWORDS);
  const explicit = args.find((a) => a.startsWith("--platforms="));
  if (explicit) {
    return explicit
      .slice("--platforms=".length)
      .split(",")
      .map((s) => s.trim())
      .filter((s) => s in PLATFORM_KEYWORDS);
  }
  // Дефолт — хост-платформа. На macOS mac-таргет electron-builder'а шлёт СРАЗУ
  // arm64 и x64 (см. build.mac.target), поэтому качаем обе mac-архитектуры —
  // иначе на Intel-маке у пользователя не будет бинарника до первого запуска.
  const host = hostPlatform();
  if (process.platform === "darwin") return ["osx-arm64", "osx-x64"];
  return [host];
}

/** GET с редиректами; resolve → Buffer. */
function httpGet(url, { accept } = {}) {
  return new Promise((resolve, reject) => {
    const headers = { "User-Agent": "twitch-cut/prepare-tdcli" };
    if (accept) headers["Accept"] = accept;
    const token = process.env.GITHUB_TOKEN || process.env.GH_TOKEN;
    if (token) headers["Authorization"] = `Bearer ${token}`;

    https
      .get(url, { headers }, (res) => {
        // Редиректы (release assets уводят на объектное хранилище).
        if (
          res.statusCode &&
          res.statusCode >= 300 &&
          res.statusCode < 400 &&
          res.headers.location
        ) {
          res.resume();
          httpGet(res.headers.location, { accept }).then(resolve, reject);
          return;
        }
        if (res.statusCode !== 200) {
          res.resume();
          reject(new Error(`HTTP ${res.statusCode} для ${url}`));
          return;
        }
        const chunks = [];
        res.on("data", (c) => chunks.push(c));
        res.on("end", () => resolve(Buffer.concat(chunks)));
      })
      .on("error", reject);
  });
}

async function resolveAssetUrl(plat) {
  const [osKws, archKws] = PLATFORM_KEYWORDS[plat];
  try {
    const raw = await httpGet(LATEST_API, {
      accept: "application/vnd.github+json",
    });
    const data = JSON.parse(raw.toString("utf8"));
    const assets = Array.isArray(data.assets) ? data.assets : [];
    for (const asset of assets) {
      const name = String(asset.name || "");
      const low = name.toLowerCase();
      if (!low.endsWith(".zip")) continue;
      if (osKws.some((k) => low.includes(k)) && archKws.some((k) => low.includes(k))) {
        if (asset.browser_download_url) {
          console.log(`  asset (API): ${name}`);
          return asset.browser_download_url;
        }
      }
    }
    console.warn(
      `  подходящий asset для ${plat} не найден в latest — беру закреплённый ${PINNED_TAG}`
    );
  } catch (err) {
    console.warn(
      `  GitHub API недоступен (${err.message}) — беру закреплённый ${PINNED_TAG}`
    );
  }
  return pinnedUrl(PINNED_ASSETS[plat]);
}

/** Распаковать zip из буфера и вытащить бинарник в destFile. */
function extractBinary(zipBuf, plat, destFile) {
  const want = BINARY_NAME(plat);
  const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), "tdcli-"));
  const zipPath = path.join(tmpDir, "cli.zip");
  fs.writeFileSync(zipPath, zipBuf);

  try {
    // Кросс-платформенная распаковка без npm-зависимостей:
    //   - unix: `unzip` есть везде;
    //   - windows: `tar` (bsdtar) в Windows 10+ умеет zip.
    if (process.platform === "win32") {
      execFileSync("tar", ["-xf", zipPath, "-C", tmpDir], { stdio: "inherit" });
    } else {
      execFileSync("unzip", ["-o", "-q", zipPath, "-d", tmpDir], {
        stdio: "inherit",
      });
    }

    // Ищем бинарник по имени рекурсивно (может лежать во вложенной папке).
    const found = findFile(tmpDir, want);
    if (!found) {
      throw new Error(`в архиве не найден бинарник '${want}'`);
    }

    fs.mkdirSync(path.dirname(destFile), { recursive: true });
    fs.copyFileSync(found, destFile);
    if (process.platform !== "win32") {
      fs.chmodSync(destFile, 0o755);
    }
  } finally {
    fs.rmSync(tmpDir, { recursive: true, force: true });
  }
}

function findFile(dir, name) {
  for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
    const full = path.join(dir, entry.name);
    if (entry.isDirectory()) {
      const nested = findFile(full, name);
      if (nested) return nested;
    } else if (entry.name === name) {
      return full;
    }
  }
  return null;
}

async function prepareOne(plat) {
  const destFile = path.join(BUILD_TDCLI_DIR, plat, BINARY_NAME(plat));
  if (fs.existsSync(destFile)) {
    console.log(`✓ ${plat}: уже есть (${path.relative(process.cwd(), destFile)})`);
    return;
  }
  console.log(`↓ ${plat}: скачиваю TwitchDownloaderCLI …`);
  const url = await resolveAssetUrl(plat);
  const zipBuf = await httpGet(url);
  extractBinary(zipBuf, plat, destFile);
  console.log(`✓ ${plat}: готов (${path.relative(process.cwd(), destFile)})`);
}

async function main() {
  const targets = parseTargets();
  console.log(`prepare-tdcli: платформы = ${targets.join(", ")}`);
  let failed = 0;
  for (const plat of targets) {
    try {
      await prepareOne(plat);
    } catch (err) {
      failed++;
      console.error(`✗ ${plat}: ${err.message}`);
    }
  }
  if (failed > 0) {
    console.error(
      `\nprepare-tdcli: ${failed} платформ(а) не подготовлено. ` +
        `Сборка продолжится, но для этих платформ VOD-download работать не будет ` +
        `(backend попробует докачать бинарник на первом запуске).`
    );
    // Не валим сборку жёстко: backend умеет докачать бинарник в рантайме.
    // Если нужно строго — раскомментируй следующую строку.
    // process.exit(1);
  }
}

main().catch((err) => {
  console.error("prepare-tdcli: неожиданная ошибка:", err);
  process.exit(1);
});
