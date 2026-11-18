/**
 * Electron main process.
 *
 * Ответственности:
 *   1. Запустить бэкенд `serve --port 0` как child_process: в собранном
 *      приложении это готовый исполняемый файл resources/backend/twitch-cut-backend
 *      (PyInstaller, Python внутри), в dev — python из backend/.venv.
 *      Порт выбирается uvicorn'ом сам; мы грепаем stdout
 *      на маркер `TWITCH_CUT_PORT=NNNN` и передаём его в renderer через
 *      preload → window.twitchCut.backendPort.
 *   2. Создать BrowserWindow с загрузкой либо http://localhost:5173 (dev),
 *      либо file://../dist/index.html (prod).
 *   3. Диалог открытия файлов (native OS dialog) — renderer не имеет прямого
 *      доступа к fs, только через IPC. Это стандартный Electron-паттерн:
 *      contextIsolation=true + preload.
 *   4. Гарантированно убить backend при закрытии окна (else zombie uvicorn).
 */

import { app, BrowserWindow, dialog, ipcMain, nativeImage, safeStorage, shell } from "electron";
import { spawn, ChildProcessWithoutNullStreams } from "child_process";
import * as path from "path";
import * as fs from "fs";
import {
  runBootstrap,
  isBootstrapped,
  savedAccel,
  BootstrapError,
  type Accel,
  type Paths,
  type BootstrapProgress,
} from "./bootstrap";

// --- paths ------------------------------------------------------------------

/**
 * Writable корень данных. В упакованном приложении сам код read-only
 * (внутри установки), поэтому модели/логи/настройки живут отдельно:
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

/**
 * Каталог backend: в prod — resources/backend (собранный PyInstaller-каталог,
 * кладётся через extraResources), в dev — <repo>/backend.
 */
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
 * Каталог с бинарником TwitchDownloaderCLI (скачивание Twitch VOD).
 *
 * В prod — resources/tdcli (кладётся через extraResources), внутри —
 * per-platform подкаталоги win-x64 / osx-arm64 / osx-x64 / linux-x64. Питон
 * (tdcli.get_tdcli_path) сам добавит нужный подкаталог по своей платформе,
 * поэтому отдаём родительскую папку. В dev — <repo>/backend/vendor/tdcli, если
 * она есть (можно положить бинарник вручную для отладки); иначе null — и тогда
 * бэкенд ищет бинарник в PATH.
 */
function resolveTdcliDir(): string | null {
  if (app.isPackaged) return path.join(process.resourcesPath, "tdcli");
  const devDir = path.resolve(__dirname, "..", "..", "..", "backend", "vendor", "tdcli");
  return fs.existsSync(devDir) ? devDir : null;
}

/**
 * Иконка окна/taskbar. В prod — реальный файл resources/icon.ico (кладётся
 * через extraResources): грузить .ico для taskbar из-под asar на Windows
 * ненадёжно, поэтому держим на диске рядом, как backend. В dev — desktop/build.
 */
function resolveIconPath(): string {
  if (process.platform === "darwin") return resolveMacIconPath();
  if (process.platform !== "win32") return resolveDragIconPath();
  if (app.isPackaged) return path.join(process.resourcesPath, "icon.ico");
  return path.resolve(__dirname, "..", "..", "build", "icon.ico");
}

function resolveMacIconPath(): string {
  if (app.isPackaged) return path.join(process.resourcesPath, "icon-macos.png");
  return path.resolve(__dirname, "..", "..", "public", "icon-macos.png");
}

/**
 * Иконка курсора для drag-out (webContents.startDrag). ВАЖНО: это НЕ то же, что
 * resolveIconPath — там .ico для taskbar. Здесь нужен PNG: macOS
 * nativeImage.createFromPath НЕ декодирует .ico (возвращает пустой image), а
 * startDrag на macOS с пустой/битой иконкой кидает исключение — из-за чего drag
 * «даже не тянется». PNG декодируется на всех платформах.
 *   - prod: resources/icon.png (кладётся через extraResources);
 *   - dev:  desktop/public/icon.png (есть в репозитории).
 */
function resolveDragIconPath(): string {
  if (app.isPackaged) return path.join(process.resourcesPath, "icon.png");
  return path.resolve(__dirname, "..", "..", "public", "icon.png");
}

/**
 * Собрать все пути окружения и команду запуска бэкенда.
 *
 * В prod бэкенд — собранный исполняемый файл в resources/backend (Python и
 * зависимости внутри), модели докачивает bootstrap в dataDir/models.
 * В dev — python из backend/.venv (`python -m twitch_cut.cli`); без venv —
 * `python` из PATH, а подскажет, чего не хватает, `twitch-cut doctor`.
 */
function resolvePaths(): Paths {
  const dataDir = resolveDataDir();
  const backendDir = resolveBackendDir();
  const common = { dataDir, logsDir: path.join(dataDir, "logs"), backendDir };

  if (app.isPackaged) {
    const exe = process.platform === "win32" ? "twitch-cut-backend.exe" : "twitch-cut-backend";
    return { ...common, backendCmd: path.join(backendDir, exe), backendArgs: [], venvRoot: null };
  }

  const devVenv = path.join(backendDir, ".venv");
  const devPython =
    process.platform === "win32"
      ? path.join(devVenv, "Scripts", "python.exe")
      : path.join(devVenv, "bin", "python");
  if (fs.existsSync(devPython)) {
    return { ...common, backendCmd: devPython, backendArgs: ["-m", "twitch_cut.cli"], venvRoot: devVenv };
  }
  return { ...common, backendCmd: "python", backendArgs: ["-m", "twitch_cut.cli"], venvRoot: null };
}

let PATHS: Paths;
let GPU_MODE = true; // уточняется после bootstrap; показываем бейдж в UI.

/**
 * Показывать ли предупреждающий бейдж «CPU». На macOS — нет: там CPU и есть
 * режим по умолчанию (GigaAM на Apple Silicon быстрый), а CoreML включают руками.
 */
function gpuModeFor(accel: Accel | null): boolean {
  if (process.platform === "darwin") return true;
  return accel !== "cpu";
}

// --- helpers ----------------------------------------------------------------

/**
 * Собрать PATH для backend-процесса (только dev-venv с WhisperX): включить
 * `nvidia\...\bin`, чтобы ctranslate2/whisperx нашли `cudnn_ops_infer64_8.dll`,
 * `cublas64_12.dll` и т.д. Движку по умолчанию (GigaAM на onnxruntime) это не
 * нужно, а у собранного бэкенда venv нет вовсе — тогда venvRoot = null.
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
    // UTF-8 mode: без него дочерний Python берёт stdout в системной локали
    // (на русской Windows — cp1251), и первый же rich-print с символом типа
    // «→» валит процесс UnicodeEncodeError'ом. Electron читает stdout как
    // utf-8 (chunk.toString("utf-8")), так что форсируем ту же кодировку.
    PYTHONUTF8: "1",
    PYTHONIOENCODING: "utf-8",
    // Backend пишет модели/кэш в writable дата-каталог, а не рядом с кодом.
    TWITCH_CUT_DATA_DIR: PATHS.dataDir,
  };
  // Путь к бандлу TwitchDownloaderCLI (скачивание VOD). Питон подставит
  // per-platform подкаталог сам. В dev может быть null — тогда не выставляем,
  // и бэкенд ищет бинарник в PATH.
  const tdcliDir = resolveTdcliDir();
  if (tdcliDir) env.TWITCH_CUT_TDCLI_DIR = tdcliDir;
  // Ускоритель бэкенд выбирает сам (onnxruntime: DirectML/CUDA/CoreML, нет —
  // CPU), поэтому TWITCH_CUT_CPU больше не выставляем.
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
    const cmd = PATHS.backendCmd;
    const staticDir = resolveStaticDir();
    const fullArgs = [
      ...PATHS.backendArgs,
      "serve",
      "--host", "127.0.0.1",
      "--port", "0",
      "--static-dir", staticDir,
    ];
    console.log("[backend] spawn:", cmd, fullArgs.join(" "));

    const proc = spawn(cmd, fullArgs, {
      cwd: PATHS.backendDir,
      env: buildBackendEnv(PATHS.venvRoot),
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

/**
 * Остановить backend и дождаться, пока процесс реально умрёт.
 *
 * Нужно перед удалением дата-каталога: на Windows нельзя удалить файлы,
 * которые держит живой процесс (модели, логи — иначе EBUSY/EPERM). kill()
 * сам по себе асинхронный — ждём событие exit (или таймаут как страховку).
 */
function stopBackendAndWait(timeoutMs = 8000): Promise<void> {
  return new Promise((resolve) => {
    const proc = backendProc;
    if (!proc || proc.killed || proc.exitCode !== null) {
      backendProc = null;
      return resolve();
    }
    let done = false;
    const finish = () => {
      if (done) return;
      done = true;
      backendProc = null;
      resolve();
    };
    proc.once("exit", finish);
    const timer = setTimeout(finish, timeoutMs);
    if (typeof timer.unref === "function") timer.unref();
    proc.kill();
  });
}

// --- window -----------------------------------------------------------------

function createWindow() {
  // Иконка окна и taskbar (см. resolveIconPath): в prod — resources/icon.ico,
  // в dev — desktop/build/icon.ico.
  const iconPath = resolveIconPath();
  const win = new BrowserWindow({
    width: 1200,
    height: 800,
    show: false,
    backgroundColor: "#161719",
    ...(process.platform === "darwin"
      ? { titleBarStyle: "hidden" as const, trafficLightPosition: { x: 16, y: 14 } }
      : {}),
    icon: fs.existsSync(iconPath) ? iconPath : undefined,
    webPreferences: {
      preload: path.join(__dirname, "preload.js"),
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: true,
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
  const iconPath = resolveIconPath();
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
      sandbox: true,
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
  // запускаем bootstrap (модели бэкенд докачает сам при первом распознавании).
  const devVenvReady = !app.isPackaged && PATHS.venvRoot !== null;

  if (devVenvReady || isBootstrapped(PATHS)) {
    // Окружение уже есть — ускоритель знаем из прошлого bootstrap (.accel).
    GPU_MODE = gpuModeFor(savedAccel(PATHS));
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
    // Ускоритель bootstrap сам запоминает в .accel — при следующих запусках
    // он известен без повторной проверки.
    const { accel } = await runBootstrap(PATHS, send);
    GPU_MODE = gpuModeFor(accel);
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

// --- AI-ключи (Electron safeStorage) ---------------------------------------
//
// API-ключи провайдеров ИИ шифруются через OS keychain (safeStorage) и лежат
// в ai-keys.enc внутри PATHS.dataDir (тот же корень, что стирает data:wipe).
// Расшифровка происходит ТОЛЬКО здесь, в main-процессе: renderer никогда не
// получает открытый ключ обратно (мост не отдаёт `get`). Для запуска джобы /
// теста соединения main сам достаёт ключ, вставляет его в тело loopback-POST
// и делает fetch — открытый ключ не покидает main.
//
// Формат файла: JSON-словарь { [ref]: base64(safeStorage.encryptString) }.
// `ref` — символическая ссылка (не секрет): её можно хранить в settings.json.

/** Путь к зашифрованному хранилищу ключей. */
function aiKeysFile(): string {
  return path.join(PATHS.dataDir, "ai-keys.enc");
}

/**
 * Санитизация ref: только [A-Za-z0-9:_-]. Иначе renderer мог бы протащить
 * `../` (path traversal нам не грозит — ref это ключ JSON, не путь — но так же
 * отсекаем инъекцию произвольных JSON-ключей) или пустую строку.
 */
function sanitizeKeyRef(ref: unknown): string | null {
  if (typeof ref !== "string") return null;
  const trimmed = ref.trim();
  if (!trimmed || trimmed.length > 200) return null;
  if (!/^[A-Za-z0-9:_-]+$/.test(trimmed)) return null;
  return trimmed;
}

/** Прочитать словарь ключей. Отсутствующий/битый файл → пустой словарь. */
function readKeyStore(): Record<string, string> {
  const file = aiKeysFile();
  try {
    if (!fs.existsSync(file)) return {};
    const raw = fs.readFileSync(file, "utf-8");
    const data = JSON.parse(raw);
    if (data && typeof data === "object" && !Array.isArray(data)) {
      return data as Record<string, string>;
    }
    return {};
  } catch (e) {
    console.warn("[ai-keys] store read failed:", e);
    return {};
  }
}

/** Атомарно записать словарь ключей (tmp → rename). */
function writeKeyStore(store: Record<string, string>): void {
  const file = aiKeysFile();
  fs.mkdirSync(path.dirname(file), { recursive: true });
  const tmp = file + ".tmp";
  fs.writeFileSync(tmp, JSON.stringify(store), { encoding: "utf-8", mode: 0o600 });
  fs.renameSync(tmp, file);
}

/**
 * Достать открытый ключ по ref (только для main-side использования: run/test).
 * Возвращает null, если ref не сохранён или шифрование недоступно. НЕ
 * экспонируется в renderer.
 */
function decryptKey(ref: string): string | null {
  if (!safeStorage.isEncryptionAvailable()) return null;
  const store = readKeyStore();
  const b64 = store[ref];
  if (typeof b64 !== "string" || !b64) return null;
  try {
    return safeStorage.decryptString(Buffer.from(b64, "base64"));
  } catch (e) {
    console.warn("[ai-keys] decrypt failed:", e);
    return null;
  }
}

ipcMain.handle("ai-keys:set", async (_e, ref: unknown, plaintextKey: unknown) => {
  const cleanRef = sanitizeKeyRef(ref);
  if (!cleanRef) return { ok: false, error: "невалидный ref ключа" };
  if (typeof plaintextKey !== "string" || !plaintextKey) {
    return { ok: false, error: "пустой ключ" };
  }
  if (!safeStorage.isEncryptionAvailable()) {
    return { ok: false, error: "шифрование недоступно в этой ОС (нет keychain)" };
  }
  try {
    const encrypted = safeStorage.encryptString(plaintextKey);
    const store = readKeyStore();
    store[cleanRef] = encrypted.toString("base64");
    writeKeyStore(store);
    return { ok: true };
  } catch (e) {
    const msg = e instanceof Error ? e.message : String(e);
    console.error("[ai-keys] set failed:", msg);
    return { ok: false, error: msg };
  }
});

ipcMain.handle("ai-keys:has", async (_e, ref: unknown) => {
  const cleanRef = sanitizeKeyRef(ref);
  if (!cleanRef) return { ok: false, error: "невалидный ref ключа" };
  const store = readKeyStore();
  return { ok: true, present: typeof store[cleanRef] === "string" && !!store[cleanRef] };
});

ipcMain.handle("ai-keys:list", async () => {
  const store = readKeyStore();
  // Только refs — никогда значения.
  return { ok: true, refs: Object.keys(store) };
});

ipcMain.handle("ai-keys:delete", async (_e, ref: unknown) => {
  const cleanRef = sanitizeKeyRef(ref);
  if (!cleanRef) return { ok: false, error: "невалидный ref ключа" };
  try {
    const store = readKeyStore();
    if (cleanRef in store) {
      delete store[cleanRef];
      writeKeyStore(store);
    }
    return { ok: true };
  } catch (e) {
    const msg = e instanceof Error ? e.message : String(e);
    console.error("[ai-keys] delete failed:", msg);
    return { ok: false, error: msg };
  }
});

// --- AI-операции через loopback (ключ вставляется main-side) ----------------
//
// Renderer вызывает эти хендлеры БЕЗ api_key: он передаёт только api_key_ref
// (символическую ссылку). main достаёт открытый ключ из safeStorage, вставляет
// его в тело запроса и делает loopback-fetch на backend. Так открытый ключ
// физически не появляется в renderer (и в его devtools network-табе).

/** Дождаться порта backend (тот же механизм, что get-backend-port). */
function awaitBackendPort(): Promise<number> {
  if (backendPort !== null) return Promise.resolve(backendPort);
  return new Promise<number>((resolve) => portListeners.push(resolve));
}

/**
 * Общий помощник: взять тело запроса от renderer (с api_key_ref, без ключа),
 * достать ключ, POST на backend по loopback, вернуть распарсенный JSON.
 * Возвращает { ok, status, data } или { ok:false, error }.
 */
async function loopbackAiPost(
  route: string,
  body: Record<string, unknown>,
): Promise<{ ok: boolean; status?: number; data?: unknown; error?: string }> {
  if (!body || typeof body !== "object") {
    return { ok: false, error: "невалидное тело запроса" };
  }
  // api_key_ref → расшифрованный ключ (main-side). Сам ref в тело backend не идёт.
  const ref = sanitizeKeyRef((body as { api_key_ref?: unknown }).api_key_ref);
  const { api_key_ref: _drop, ...rest } = body as Record<string, unknown>;
  const payload: Record<string, unknown> = { ...rest };
  if (ref) {
    const key = decryptKey(ref);
    if (!key) {
      return { ok: false, error: "ключ не найден или не расшифровывается — сохраните ключ заново в настройках" };
    }
    payload.api_key = key;
  }
  try {
    const port = await awaitBackendPort();
    const res = await fetch(`http://127.0.0.1:${port}${route}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await res.json().catch(() => null);
    return { ok: res.ok, status: res.status, data };
  } catch (e) {
    const msg = e instanceof Error ? e.message : String(e);
    console.error(`[ai] loopback ${route} failed:`, msg);
    return { ok: false, error: msg };
  }
}

ipcMain.handle("ai:testConnection", async (_e, req: Record<string, unknown>) => {
  return loopbackAiPost("/ai/test-connection", req);
});

ipcMain.handle("ai:detectOfftopic", async (_e, req: Record<string, unknown>) => {
  return loopbackAiPost("/jobs/detect-offtopic", req);
});

/**
 * Иконка курсора для drag-out (webContents.startDrag). startDrag на macOS
 * кидает исключение, если icon пустой/невалидный (тогда drag «даже не тянется»),
 * а генерировать превью из видео дорого и асинхронно (drag должен стартовать
 * синхронно в dragstart). Поэтому:
 *   1. пробуем PNG-файл иконки приложения (resolveDragIconPath — именно PNG,
 *      т.к. macOS не декодирует .ico);
 *   2. фолбэк — сгенерированный в рантайме 64×64 непрозрачный bitmap через
 *      createFromBitmap. Без внешних файлов и base64-констант: гарантированно
 *      валиден на всех ОС, а 64px — безопасный минимум (16px на Retina macOS
 *      startDrag отвергает). Кэшируем.
 */
let _dragIconCache: Electron.NativeImage | null = null;
function dragIcon(): Electron.NativeImage {
  if (_dragIconCache && !_dragIconCache.isEmpty()) return _dragIconCache;
  // 1. PNG-иконка приложения с диска (prod: resources, dev: public).
  try {
    const iconPath = resolveDragIconPath();
    if (fs.existsSync(iconPath)) {
      const img = nativeImage.createFromPath(iconPath);
      if (!img.isEmpty()) {
        _dragIconCache = img.resize({ width: 64, height: 64 });
        return _dragIconCache;
      }
    }
  } catch {
    /* fall through to generated fallback */
  }
  // 2. Фолбэк: рисуем 64×64 BGRA-битмап в рантайме (opaque серый). Не зависит
  //    ни от файла, ни от PNG-декодера — createFromBitmap всегда валиден.
  const W = 64;
  const H = 64;
  const buf = Buffer.alloc(W * H * 4);
  for (let i = 0; i < W * H; i++) {
    const o = i * 4;
    buf[o] = 95; // B
    buf[o + 1] = 90; // G
    buf[o + 2] = 90; // R
    buf[o + 3] = 255; // A (opaque — обязательно, иначе macOS отвергает)
  }
  _dragIconCache = nativeImage.createFromBitmap(buf, { width: W, height: H });
  return _dragIconCache;
}

/**
 * Drag-out файла ИЗ приложения в браузер/файловую систему/сайт.
 *
 * Renderer в обработчике `dragstart` шлёт сюда абсолютный путь; мы вызываем
 * webContents.startDrag — дальше ОС ведёт нативный drag (сайт/загрузчик видит
 * настоящий файл). Используем ipcMain.on (send, не invoke): startDrag обязан
 * стартовать синхронно в контексте dragstart, promise round-trip его ломает.
 * Путь проверяем на существование — иначе startDrag на Windows кидает throw.
 */
ipcMain.on("shell:startDrag", (event, filePath: string) => {
  if (typeof filePath !== "string" || !filePath) return;
  try {
    if (!fs.existsSync(filePath)) {
      console.warn("[ipc] startDrag: file not found:", filePath);
      event.sender.send("shell:startDragError", {
        path: filePath,
        reason: "not-found",
      });
      return;
    }
    event.sender.startDrag({ file: filePath, icon: dragIcon() });
  } catch (e) {
    // Раньше ошибка молча гасилась в console.warn — из-за чего плитка «даже не
    // тянулась» без единого признака в UI. Теперь сообщаем в renderer, чтобы
    // показать тост, а не оставлять пользователя в неведении.
    console.warn("[ipc] startDrag failed:", e);
    event.sender.send("shell:startDragError", {
      path: filePath,
      reason: "start-failed",
      message: e instanceof Error ? e.message : String(e),
    });
  }
});

/**
 * Полное удаление скачанных данных: модели, логи, настройки — всё, что
 * появилось в дата-каталоге ПОСЛЕ установки и что штатный деинсталлятор не
 * отслеживает (у старых версий там же лежали venv с torch, python, uv-cache).
 *
 * Сам код программы (в Program Files / Applications) НЕ трогаем — его удаляет
 * системный деинсталлятор. После wipe при следующем запуске снова пройдёт
 * bootstrap. По умолчанию перезапускаем приложение, чтобы сплэш заново поднял
 * окружение; при quit=true — просто выходим (пользователь дальше деинсталлит).
 *
 * Порядок важен: сперва глушим backend и ЖДЁМ его смерти, иначе на Windows
 * файлы моделей заняты и rmSync падает EBUSY.
 */
ipcMain.handle("data:wipe", async (_e, opts?: { quit?: boolean }) => {
  const target = PATHS?.dataDir;
  if (!target) return { ok: false, error: "dataDir не определён" };

  // В dev backend/.venv лежит внутри репозитория (см. resolvePaths). Стирать
  // исходники по кнопке нельзя — это данные разработчика, не пользовательские.
  if (!app.isPackaged) {
    return {
      ok: false,
      error:
        "Полное удаление доступно только в собранном приложении (в dev данные лежат в репозитории).",
    };
  }

  const res = await dialog.showMessageBox({
    type: "warning",
    buttons: ["Удалить всё", "Отмена"],
    defaultId: 1,
    cancelId: 1,
    title: "Полное удаление данных",
    message: "Удалить все скачанные данные Twitch Cut?",
    detail:
      "Будут удалены модели распознавания, логи и настройки из:\n" +
      target +
      "\n\nСама программа останется — удалите её отдельно через системный деинсталлятор. " +
      "При следующем запуске всё скачается заново.",
    noLink: true,
  });
  if (res.response !== 0) return { ok: false, cancelled: true };

  try {
    await stopBackendAndWait();
    backendPort = null;
    fs.rmSync(target, { recursive: true, force: true });
    // На macOS дополнительно чистим Logs и Caches — они остаются после удаления .app.
    // Оба пути app-specific: logs = ~/Library/Logs/<имя>, cache = ~/Library/Caches/<имя>.
    // Caches строим ЯВНО через имя приложения — app.getPath("cache") вернул бы
    // КОРЕНЬ ~/Library/Caches (общий для всех программ), его стирать нельзя.
    if (process.platform === "darwin") {
      const appName = app.getName();
      const extras = [
        app.getPath("logs"),
        path.join(app.getPath("home"), "Library", "Caches", appName),
      ];
      for (const dir of extras) {
        // страховка: удаляем только каталог, чьё имя == имени приложения
        if (path.basename(dir) === appName) {
          fs.rmSync(dir, { recursive: true, force: true });
        }
      }
    }
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    console.error("[ipc] data:wipe failed:", msg);
    return { ok: false, error: msg };
  }

  // Перезапуск, чтобы заново пройти bootstrap; либо выходим по запросу.
  if (opts?.quit) {
    app.quit();
  } else {
    app.relaunch();
    app.exit(0);
  }
  return { ok: true };
});

// --- app lifecycle ----------------------------------------------------------
app.whenReady().then(async () => {
  PATHS = resolvePaths();
  if (process.platform === "darwin") {
    app.dock?.setIcon(nativeImage.createFromPath(resolveMacIconPath()));
  }

  // Windows: без явного AppUserModelID система не связывает окна процесса с
  // иконкой/ярлыком приложения и показывает дефолтную иконку Electron в
  // панели задач. Должен совпадать с build.appId в package.json.
  if (process.platform === "win32") app.setAppUserModelId("com.arm4tura.twitchcut");

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
