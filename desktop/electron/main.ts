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

// --- helpers ----------------------------------------------------------------

/** Ищем python в backend/.venv, fallback на `python` из PATH. */
function findPython(): { cmd: string; args: string[]; venvRoot: string | null } {
  const repoRoot = path.resolve(__dirname, "..", "..", "..");
  const winPy = path.join(repoRoot, "backend", ".venv", "Scripts", "python.exe");
  const nixPy = path.join(repoRoot, "backend", ".venv", "bin", "python");
  if (process.platform === "win32" && fs.existsSync(winPy)) {
    return {
      cmd: winPy,
      args: ["-m", "twitch_cut.cli"],
      venvRoot: path.join(repoRoot, "backend", ".venv"),
    };
  }
  if (fs.existsSync(nixPy)) {
    return {
      cmd: nixPy,
      args: ["-m", "twitch_cut.cli"],
      venvRoot: path.join(repoRoot, "backend", ".venv"),
    };
  }
  // Fallback — polагаемся на PATH. Пользователю подскажет doctor.
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
  const env: NodeJS.ProcessEnv = { ...process.env, PYTHONUNBUFFERED: "1" };
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
    const repoRoot = path.resolve(__dirname, "..", "..", "..");
    const staticDir = path.join(repoRoot, "desktop", "dist");
    const fullArgs = [
      ...args,
      "serve",
      "--host", "127.0.0.1",
      "--port", "0",
      "--static-dir", staticDir,
    ];
    console.log("[backend] spawn:", cmd, fullArgs.join(" "));

    const proc = spawn(cmd, fullArgs, {
      cwd: repoRoot,
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
    win.loadFile(path.join(__dirname, "..", "..", "dist", "index.html"));
  }
}

// --- IPC --------------------------------------------------------------------

ipcMain.handle("get-backend-port", async () => {
  if (backendPort !== null) return backendPort;
  return new Promise<number>((resolve) => portListeners.push(resolve));
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
  try {
    await startBackend();
  } catch (err) {
    console.error("failed to start backend:", err);
    dialog.showErrorBox(
      "Backend не запустился",
      String(err) + "\n\nПроверь, что установлен twitch-cut (см. install.ps1) и запусти `twitch-cut doctor`."
    );
    app.quit();
    return;
  }
  createWindow();

  app.on("activate", () => {
    if (BrowserWindow.getAllWindows().length === 0) createWindow();
  });
});

app.on("window-all-closed", () => {
  stopBackend();
  if (process.platform !== "darwin") app.quit();
});

app.on("before-quit", stopBackend);
