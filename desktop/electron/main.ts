/**
 * Electron main process.
 *
 * Ответственности:
 *   1. Найти Python из backend/.venv и запустить `twitch-cut serve --port 0`
 *      как child_process. Порт выбирается uvicorn'ом сам; мы грепаем stdout
 *      на маркер `TWITCH_CUT_PORT=NNNN` и передаём его в renderer через
 *      preload → window.twitchCut.backendPort.
 *   2. Создать BrowserWindow с загрузкой либо http://localhost:5173 (dev),
 *      либо file://../dist/index.html (prod).
 *   3. Диалог открытия файлов (native OS dialog) — renderer не имеет прямого
 *      доступа к fs, только через IPC. Это стандартный Electron-паттерн:
 *      contextIsolation=true + preload.
 *   4. Гарантированно убить backend при закрытии окна (else zombie uvicorn).
 */

import { app, BrowserWindow, dialog, ipcMain, shell } from "electron";
import { spawn, ChildProcessWithoutNullStreams } from "child_process";
import * as path from "path";
import * as fs from "fs";
import {
  runBootstrap,
  isBootstrapped,
  BootstrapError,
  type Paths,
  type BootstrapProgress,
} from "./bootstrap";

// --- paths ------------------------------------------------------------------

/**
 * Writable корень данных. В упакованном приложении сам код read-only
 * (внутри установки), поэтому venv/models/логи живут отдельно:
 *   - portable: папка <exe_dir>/TwitchCutter-Data, если она писабельна;
 *   - обычная установка: app.getPath("userData") (%APPDATA%\Twitch Cutter).
 * В dev (не упаковано) — тоже userData, чтобы отделить от исходников.
 */
function resolveDataDir(): string {
  if (app.isPackaged) {
    const exeDir = path.dirname(app.getPath("exe"));
    const portable = path.join(exeDir, "TwitchCutter-Data");
    try {
      fs.mkdirSync(portable, { recursive: true });
      fs.accessSync(exeDir, fs.constants.W_OK);
      return portable;
    } catch {
      /* Program Files не писабелен — уходим в userData */
    }
  }
  return app.getPath("userData");
}

/** Каталог backend: в prod — resources/backend (extraResources), в dev — <repo>/backend. */
function resolveBackendDir(): string {
  if (app.isPackaged) return path.join(process.resourcesPath, "backend");
  return path.resolve(__dirname, "..", "..", "..", "backend");
}

/** Каталог статики фронта (desktop/dist). В prod лежит рядом с main.js внутри app. */
function resolveStaticDir(): string {
  if (app.isPackaged) return path.join(app.getAppPath(), "dist");
  return path.resolve(__dirname, "..", "..", "dist");
}

/**
 * Собрать все пути окружения.
 *
 * В prod venv живёт в writable dataDir/env (его создаёт bootstrap).
 * В dev, если у разработчика уже есть рабочий backend/.venv — используем его
 * напрямую и пропускаем bootstrap (иначе `npm run dev` качал бы Python и 4 ГБ
 * моделей заново в userData, игнорируя готовое окружение).
 */
function resolvePaths(): Paths {
  const dataDir = resolveDataDir();

  // dev: предпочитаем существующий backend/.venv.
  if (!app.isPackaged) {
    const devVenv = path.join(resolveBackendDir(), ".venv");
    const devPython =
      process.platform === "win32"
        ? path.join(devVenv, "Scripts", "python.exe")
        : path.join(devVenv, "bin", "python");
    if (fs.existsSync(devPython)) {
      return {
        dataDir,
        envDir: devVenv,
        binDir: path.join(dataDir, "bin"),
        logsDir: path.join(dataDir, "logs"),
        backendDir: resolveBackendDir(),
        venvPython: devPython,
      };
    }
  }

  const envDir = path.join(dataDir, "env");
  const venvPython =
    process.platform === "win32"
      ? path.join(envDir, "Scripts", "python.exe")
      : path.join(envDir, "bin", "python");
  return {
    dataDir,
    envDir,
    binDir: path.join(dataDir, "bin"),
    logsDir: path.join(dataDir, "logs"),
    backendDir: resolveBackendDir(),
    venvPython,
  };
}

let PATHS: Paths;
let GPU_MODE = true; // уточняется после bootstrap; показываем бейдж в UI.

// --- helpers ----------------------------------------------------------------

/** Python из bootstrap-окружения (dataDir/env), fallback на `python` из PATH. */
function findPython(): { cmd: string; args: string[]; venvRoot: string | null } {
  if (fs.existsSync(PATHS.venvPython)) {
    return { cmd: PATHS.venvPython, args: ["-m", "twitch_cut.cli"], venvRoot: PATHS.envDir };
  }
  // Fallback — полагаемся на PATH. Пользователю подскажет doctor.
  return { cmd: "python", args: ["-m", "twitch_cut.cli"], venvRoot: null };
}

/**
 * Собрать PATH для backend-процесса: включить `nvidia\...\bin` из venv, чтобы
 * ctranslate2/whisperx нашли `cudnn_ops_infer64_8.dll`, `cublas64_12.dll` и т.д.
 *
 * Проблема: torch 2.6 (cu126) тянет cuDNN 9 в собственную папку torch\lib и её
 * же прописывает в DLL-путь через `os.add_dll_directory`. Но ctranslate2
 * загружается по обычному LoadLibrary — ему нужен PATH. Если pip-пакет
 * `nvidia-cudnn-cu12` установлен, его DLL лежат в site-packages\nvidia\cudnn\bin
 * — прокидываем эту папку в PATH при спавне.
 *
 * Возвращаем расширенный env для передачи в spawn(). Linux-случай безопасно
 * no-op: путей не существует, `existsSync` вернёт false, ничего не добавится.
 */
function buildBackendEnv(venvRoot: string | null): NodeJS.ProcessEnv {
  const env: NodeJS.ProcessEnv = {
    ...process.env,
    PYTHONUNBUFFERED: "1",
    // Backend пишет модели/кэш в writable дата-каталог, а не рядом с кодом.
    TWITCH_CUT_DATA_DIR: PATHS.dataDir,
  };
  // Если GPU не найден при bootstrap — сообщаем backend, чтобы doctor не ругался
  // FAIL'ом на отсутствие CUDA и pipeline знал, что работает в CPU-режиме.
  if (!GPU_MODE) env.TWITCH_CUT_CPU = "1";
  if (!venvRoot) return env;

  const sitePkgs = process.platform === "win32"
    ? path.join(venvRoot, "Lib", "site-packages")
    : path.join(venvRoot, "lib");

  // Кандидаты — все известные CUDA-либы, которые pip-пакеты `nvidia-*` кладут
  // в свои bin. Добавляем только существующие, чтобы PATH не рос без нужды.
  const cudaBinRoots = [
    path.join(sitePkgs, "nvidia", "cudnn", "bin"),
    path.join(sitePkgs, "nvidia", "cublas", "bin"),
    path.join(sitePkgs, "nvidia", "cuda_runtime", "bin"),
    path.join(sitePkgs, "nvidia", "cufft", "bin"),
    path.join(sitePkgs, "nvidia", "curand", "bin"),
    // torch кладёт свои cuDNN 9 сюда — на случай если ctranslate2 подхватит их.
    path.join(sitePkgs, "torch", "lib"),
  ].filter((p) => fs.existsSync(p));

  if (cudaBinRoots.length > 0) {
    const sep = process.platform === "win32" ? ";" : ":";
    const prev = env.PATH ?? env.Path ?? "";
    // Prepend — наши пути должны победить системные (иначе Windows возьмёт
    // случайный cudnn64_8.dll из %SystemRoot%\System32, если он там есть).
    env.PATH = cudaBinRoots.join(sep) + sep + prev;
    console.log("[backend] cuda dll dirs prepended to PATH:", cudaBinRoots);
  }
  return env;
}

// --- backend lifecycle ------------------------------------------------------

let backendProc: ChildProcessWithoutNullStreams | null = null;
let backendPort: number | null = null;
const portListeners: Array<(p: number) => void> = [];

function startBackend(): Promise<number> {
  return new Promise((resolve, reject) => {
    const { cmd, args, venvRoot } = findPython();
    const staticDir = resolveStaticDir();
    const fullArgs = [
      ...args,
      "serve",
      "--host", "127.0.0.1",
      "--port", "0",
      "--static-dir", staticDir,
    ];
    console.log("[backend] spawn:", cmd, fullArgs.join(" "));

    const proc = spawn(cmd, fullArgs, {
      cwd: PATHS.backendDir,
      env: buildBackendEnv(venvRoot),
      windowsHide: true,
    });
    backendProc = proc;

    let settled = false;
    const timer = setTimeout(() => {
      if (!settled) {
        settled = true;
        reject(new Error("backend startup timeout (30s)"));
      }
    }, 30_000);

    proc.stdout.on("data", (chunk: Buffer) => {
      const text = chunk.toString("utf-8");
      process.stdout.write("[backend] " + text);
      const m = text.match(/TWITCH_CUT_PORT=(\d+)/);
      if (m && !settled) {
        settled = true;
        clearTimeout(timer);
        backendPort = parseInt(m[1], 10);
        portListeners.splice(0).forEach((cb) => cb(backendPort!));
        resolve(backendPort);
      }
    });

    proc.stderr.on("data", (chunk: Buffer) => {
      process.stderr.write("[backend:err] " + chunk.toString("utf-8"));
    });

    proc.on("exit", (code, signal) => {
      console.log(`[backend] exited code=${code} signal=${signal}`);
      backendProc = null;
      if (!settled) {
        settled = true;
        clearTimeout(timer);
        reject(new Error(`backend exited before ready (code=${code})`));
      }
    });
  });
}

function stopBackend() {
  if (backendProc && !backendProc.killed) {
    console.log("[backend] terminating");
    // SIGTERM даёт uvicorn шанс закрыть сокеты; на Windows kill() посылает
    // TerminateProcess, что тоже приемлемо (нет чистого shutdown-хука).
    backendProc.kill();
  }
}

// --- window -----------------------------------------------------------------

function createWindow() {
  // Иконка окна и taskbar. Файл лежит в desktop/build/icon.ico — это же
  // multi-size ICO пойдёт в electron-builder-конфиг в Фазе 6.
  // Путь: dist из tsc — desktop/electron/dist/main.js, отсюда build/ — на два
  // уровня вверх от __dirname (electron/dist → electron → desktop → build).
  const iconPath = path.resolve(__dirname, "..", "..", "build", "icon.ico");
  const win = new BrowserWindow({
    width: 1200,
    height: 800,
    show: false,
    icon: fs.existsSync(iconPath) ? iconPath : undefined,
    webPreferences: {
      preload: path.join(__dirname, "preload.js"),
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: false, // preload использует path/fs через `require` — sandbox их блокирует.
    },
  });

  win.once("ready-to-show", () => win.show());

  if (process.env.NODE_ENV === "development") {
    win.loadURL("http://localhost:5173");
    win.webContents.openDevTools({ mode: "detach" });
  } else {
    win.loadFile(path.join(resolveStaticDir(), "index.html"));
  }
}

// --- splash + bootstrap -----------------------------------------------------

let splashWin: BrowserWindow | null = null;

/** Небольшое окно первого запуска с прогрессом установки. */
function createSplash(): BrowserWindow {
  const iconPath = path.resolve(__dirname, "..", "..", "build", "icon.ico");
  const win = new BrowserWindow({
    width: 520,
    height: 340,
    frame: false,
    resizable: false,
    show: true,
    center: true,
    icon: fs.existsSync(iconPath) ? iconPath : undefined,
    backgroundColor: "#0b0b0f",
    webPreferences: {
      preload: path.join(__dirname, "preload.js"),
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: false,
    },
  });
  if (process.env.NODE_ENV === "development") {
    win.loadURL("http://localhost:5173/splash.html");
  } else {
    win.loadFile(path.join(resolveStaticDir(), "splash.html"));
  }
  return win;
}

/**
 * Прогнать bootstrap, показывая прогресс в splash. Возвращает true при успехе.
 * При ошибке оставляет splash открытым (там кнопки «Показать лог» / «Повторить»)
 * и резолвится false.
 */
async function ensureEnvironment(): Promise<boolean> {
  // dev: если рабочий backend/.venv уже есть — считаем окружение готовым и НЕ
  // запускаем bootstrap (не качаем Python/модели поверх готового venv).
  const devVenvReady = !app.isPackaged && fs.existsSync(PATHS.venvPython);

  if (devVenvReady || isBootstrapped(PATHS)) {
    // Окружение уже есть — GPU-режим определяем по наличию флага .cpu-mode.
    GPU_MODE = !fs.existsSync(path.join(PATHS.dataDir, ".cpu-mode"));
    return true;
  }

  splashWin = createSplash();
  // Дожидаемся готовности renderer, чтобы первые события прогресса не потерялись.
  await new Promise<void>((res) => {
    if (!splashWin) return res();
    splashWin.webContents.once("did-finish-load", () => res());
  });

  const send = (p: BootstrapProgress) => {
    splashWin?.webContents.send("bootstrap:progress", p);
  };

  try {
    const { gpu } = await runBootstrap(PATHS, send);
    GPU_MODE = gpu;
    // Запоминаем режим, чтобы при следующих запусках знать без повторного детекта.
    try {
      if (gpu) fs.rmSync(path.join(PATHS.dataDir, ".cpu-mode"), { force: true });
      else fs.writeFileSync(path.join(PATHS.dataDir, ".cpu-mode"), "1", "utf-8");
    } catch {
      /* best-effort */
    }
    return true;
  } catch (err) {
    const logPath = err instanceof BootstrapError ? err.logPath : "";
    send({
      stage: "error",
      label: err instanceof Error ? err.message : String(err),
      percent: 0,
      detail: logPath ? `Лог: ${logPath}` : undefined,
    });
    return false;
  }
}

// --- IPC --------------------------------------------------------------------

ipcMain.handle("get-backend-port", async () => {
  if (backendPort !== null) return backendPort;
  return new Promise<number>((resolve) => portListeners.push(resolve));
});

/** Режим GPU/CPU — renderer показывает бейдж «CPU mode». */
ipcMain.handle("get-gpu-mode", async () => GPU_MODE);

/** Открыть лог bootstrap в системном редакторе (кнопка в сплэше). */
ipcMain.handle("bootstrap:openLog", async () => {
  const logPath = path.join(PATHS.logsDir, "bootstrap.log");
  if (fs.existsSync(logPath)) {
    await shell.openPath(logPath);
    return true;
  }
  return false;
});

/** Повторить bootstrap (кнопка «Повторить» после ошибки). */
ipcMain.handle("bootstrap:retry", async () => {
  const ok = await ensureEnvironment();
  if (ok) {
    try {
      await startBackend();
      createWindow();
      splashWin?.close();
      splashWin = null;
    } catch (err) {
      dialog.showErrorBox("Backend не запустился", String(err));
      return false;
    }
  }
  return ok;
});

ipcMain.handle("dialog:openFile", async (_e, opts?: Electron.OpenDialogOptions) => {
  const res = await dialog.showOpenDialog({
    properties: ["openFile"],
    ...opts,
  });
  if (res.canceled || res.filePaths.length === 0) return null;
  return res.filePaths[0];
});

ipcMain.handle("dialog:openDirectory", async () => {
  const res = await dialog.showOpenDialog({ properties: ["openDirectory"] });
  if (res.canceled || res.filePaths.length === 0) return null;
  return res.filePaths[0];
});

ipcMain.handle("dialog:saveFile", async (_e, opts?: Electron.SaveDialogOptions) => {
  const res = await dialog.showSaveDialog(opts ?? {});
  if (res.canceled || !res.filePath) return null;
  return res.filePath;
});

/**
 * Открыть системный проводник с выделенным файлом. Renderer передаёт
 * абсолютный путь. Если файл не существует — пробуем `openPath` на
 * родительский каталог (шоу-ин-фолдер без селекта). Возвращаем true
 * только если реально что-то открылось.
 */
ipcMain.handle("shell:showInFolder", async (_e, targetPath: string) => {
  if (typeof targetPath !== "string" || !targetPath) return false;
  try {
    if (fs.existsSync(targetPath)) {
      shell.showItemInFolder(targetPath);
      return true;
    }
    const dir = path.dirname(targetPath);
    if (fs.existsSync(dir)) {
      const err = await shell.openPath(dir);
      return err === "";
    }
    return false;
  } catch (e) {
    console.warn("[ipc] showInFolder failed:", e);
    return false;
  }
});

/**
 * Открыть путь как есть — если это папка, Explorer/Finder покажет её
 * содержимое; если файл, откроется в ассоциированном приложении.
 * Для «Открыть папку логов» из настроек. Если папки нет — молча создаём
 * (backend её создаст при первом save, но у нас может быть свежий инстанс).
 */
ipcMain.handle("shell:openPath", async (_e, targetPath: string) => {
  if (typeof targetPath !== "string" || !targetPath) return false;
  try {
    if (!fs.existsSync(targetPath)) {
      fs.mkdirSync(targetPath, { recursive: true });
    }
    const err = await shell.openPath(targetPath);
    return err === "";
  } catch (e) {
    console.warn("[ipc] openPath failed:", e);
    return false;
  }
});

/**
 * Открыть внешний URL. Фильтруем протокол: http/https/mailto — можно,
 * всё остальное (в частности file:, javascript:) — молча false. Renderer
 * не должен уметь через это API запустить произвольный локальный
 * бинарь через .desktop / .lnk-ссылку.
 */
ipcMain.handle("shell:openExternal", async (_e, url: string) => {
  if (typeof url !== "string" || !url) return false;
  const lower = url.trim().toLowerCase();
  const ok = lower.startsWith("http://") || lower.startsWith("https://") || lower.startsWith("mailto:");
  if (!ok) {
    console.warn("[ipc] openExternal rejected non-http url:", url.slice(0, 60));
    return false;
  }
  try {
    await shell.openExternal(url);
    return true;
  } catch (e) {
    console.warn("[ipc] openExternal failed:", e);
    return false;
  }
});

// --- app lifecycle ----------------------------------------------------------

app.whenReady().then(async () => {
  PATHS = resolvePaths();

  // 1. Первый запуск: подготовить Python-окружение и модели (со сплэшем).
  const ready = await ensureEnvironment();
  if (!ready) {
    // Splash остался открытым с ошибкой и кнопками «Показать лог»/«Повторить».
    // Не выходим — пользователь может нажать «Повторить» (bootstrap:retry).
    return;
  }

  // 2. Backend.
  try {
    await startBackend();
  } catch (err) {
    console.error("failed to start backend:", err);
    dialog.showErrorBox(
      "Backend не запустился",
      String(err) +
        "\n\nПопробуй перезапустить приложение. Лог: " +
        path.join(PATHS.logsDir, "bootstrap.log")
    );
    app.quit();
    return;
  }

  // 3. Главное окно, splash закрываем.
  createWindow();
  splashWin?.close();
  splashWin = null;

  app.on("activate", () => {
    if (BrowserWindow.getAllWindows().length === 0) createWindow();
  });
});

app.on("window-all-closed", () => {
  stopBackend();
  if (process.platform !== "darwin") app.quit();
});

app.on("before-quit", stopBackend);
