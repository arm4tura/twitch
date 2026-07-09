/**
 * First-run bootstrap: готовит Python-окружение БЕЗ участия пользователя.
 *
 * Раньше это делал install.ps1 руками (нужен системный Python + Node + ручной
 * prefetch). Теперь всё это происходит при первом запуске приложения:
 *   1. Скачиваем standalone `uv` (один бинарь) — менеджер Python/venv/пакетов.
 *   2. `uv python install 3.12` — uv сам качает нужный CPython.
 *   3. `uv venv <envDir>` — создаём venv в writable дата-каталоге.
 *   4. GPU-детект (nvidia-smi): выбираем requirements-gpu.txt vs requirements.txt.
 *   5. `uv pip install -r ...` + `uv pip install -e backend --no-deps`.
 *   6. `python -m twitch_cut.cli prefetch` — скачиваем модели (~4 GB).
 *   7. Пишем sentinel `<envDir>/.bootstrap-ok`, чтобы больше не повторять.
 *
 * Прогресс стримится в splash через onProgress. Полный лог пишется в
 * <logsDir>/bootstrap.log (кнопка «Показать лог» в сплэше открывает его).
 *
 * Идемпотентно: если sentinel есть и python на месте — возвращаемся сразу.
 * При ошибке любой стадии кидаем BootstrapError с путём к логу.
 */

import { spawn } from "child_process";
import * as fs from "fs";
import * as path from "path";

// --- типы -------------------------------------------------------------------

export interface Paths {
  /** Writable корень данных (userData или portable-папка). */
  dataDir: string;
  /** Python venv: <dataDir>/env */
  envDir: string;
  /** Бинарники (uv.exe): <dataDir>/bin */
  binDir: string;
  /** Логи: <dataDir>/logs */
  logsDir: string;
  /** Каталог backend (resources/backend в prod, <repo>/backend в dev). */
  backendDir: string;
  /** Путь до python.exe внутри env. */
  venvPython: string;
}

export interface BootstrapProgress {
  /** Машинный ключ стадии. */
  stage:
    | "check"
    | "uv"
    | "python"
    | "venv"
    | "gpu"
    | "deps"
    | "package"
    | "models"
    | "done"
    | "error";
  /** Человекочитаемая строка для сплэша (RU). */
  label: string;
  /** Общий прогресс 0..100 (грубая оценка по стадиям). */
  percent: number;
  /** Последняя строка лога/детали (хвост процесса). */
  detail?: string;
  /** Определён ли GPU (заполняется после стадии gpu). */
  gpu?: boolean;
}

export type ProgressCb = (p: BootstrapProgress) => void;

export class BootstrapError extends Error {
  constructor(message: string, public readonly logPath: string) {
    super(message);
    this.name = "BootstrapError";
  }
}

// --- константы --------------------------------------------------------------

/** Standalone uv для Windows x64 (zip с uv.exe + uvx.exe). */
const UV_WIN_URL =
  "https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-pc-windows-msvc.zip";

/** Версия схемы bootstrap — если поменяем шаги, бампаем и sentinel протухает. */
const BOOTSTRAP_VERSION = "2";

// Грубые веса стадий для общего прогресс-бара (сумма ≈ 100).
const STAGE_BASE: Record<BootstrapProgress["stage"], number> = {
  check: 0,
  uv: 3,
  python: 10,
  venv: 15,
  gpu: 18,
  deps: 25, // самая долгая — torch+cu126 ~2-3 GB
  package: 70,
  models: 75, // ~4 GB, тоже долгая
  done: 100,
  error: 0,
};

// --- утилиты процессов ------------------------------------------------------

function isWindows(): boolean {
  return process.platform === "win32";
}

function appendLog(logPath: string, text: string): void {
  try {
    fs.appendFileSync(logPath, text);
  } catch {
    /* лог — best-effort, не роняем bootstrap из-за него */
  }
}

/**
 * Запустить процесс, стримя stdout/stderr в лог и последнюю строку — в onProgress.
 * Резолвится с полным stdout; реджектится, если exit code != 0.
 */
function run(
  cmd: string,
  args: string[],
  opts: { cwd?: string; env?: NodeJS.ProcessEnv; logPath: string },
  onLine?: (line: string) => void
): Promise<string> {
  return new Promise((resolve, reject) => {
    appendLog(opts.logPath, `\n$ ${cmd} ${args.join(" ")}\n`);
    const child = spawn(cmd, args, {
      cwd: opts.cwd,
      env: opts.env ?? process.env,
      windowsHide: true,
    });
    let out = "";
    const handle = (chunk: Buffer, toErr: boolean) => {
      const text = chunk.toString("utf-8");
      out += text;
      appendLog(opts.logPath, (toErr ? "[err] " : "") + text);
      const lines = text.split(/\r?\n/).filter((l) => l.trim().length > 0);
      const last = lines[lines.length - 1];
      if (last && onLine) onLine(last.slice(0, 160));
    };
    child.stdout.on("data", (c) => handle(c, false));
    child.stderr.on("data", (c) => handle(c, true));
    child.on("error", (e) => reject(e));
    child.on("exit", (code) => {
      if (code === 0) resolve(out);
      else reject(new Error(`${path.basename(cmd)} exited with code ${code}`));
    });
  });
}

/** Запустить PowerShell-команду (для скачивания/распаковки на Windows). */
function pwsh(script: string, logPath: string, onLine?: (l: string) => void): Promise<string> {
  return run(
    "powershell",
    ["-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", script],
    { logPath },
    onLine
  );
}

// --- детект состояния -------------------------------------------------------

/** Готово ли окружение: python на месте + свежий sentinel. */
export function isBootstrapped(paths: Paths): boolean {
  const sentinel = path.join(paths.envDir, ".bootstrap-ok");
  if (!fs.existsSync(paths.venvPython) || !fs.existsSync(sentinel)) return false;
  try {
    return fs.readFileSync(sentinel, "utf-8").trim() === BOOTSTRAP_VERSION;
  } catch {
    return false;
  }
}

/** Есть ли NVIDIA GPU (nvidia-smi отработал с кодом 0). */
async function detectGpu(logPath: string): Promise<boolean> {
  try {
    await run("nvidia-smi", ["-L"], { logPath });
    return true;
  } catch {
    return false;
  }
}

// --- получение uv -----------------------------------------------------------

/** Вернуть путь к uv.exe, скачав его при необходимости. */
async function ensureUv(paths: Paths, logPath: string, emit: ProgressCb): Promise<string> {
  const uvExe = path.join(paths.binDir, isWindows() ? "uv.exe" : "uv");
  if (fs.existsSync(uvExe)) return uvExe;

  if (!isWindows()) {
    // На не-Windows полагаемся на системный uv (dev-сценарий).
    return "uv";
  }

  fs.mkdirSync(paths.binDir, { recursive: true });
  const zip = path.join(paths.binDir, "uv.zip");
  emit({ stage: "uv", label: "Скачиваю установщик uv…", percent: STAGE_BASE.uv });

  // Скачивание + распаковка средствами Windows PowerShell (без сторонних зависимостей).
  await pwsh(
    `$ProgressPreference='SilentlyContinue';` +
      `Invoke-WebRequest -Uri '${UV_WIN_URL}' -OutFile '${zip}';` +
      `Expand-Archive -Path '${zip}' -DestinationPath '${paths.binDir}' -Force;` +
      `Remove-Item '${zip}' -Force`,
    logPath,
    (l) => emit({ stage: "uv", label: "Скачиваю uv…", percent: STAGE_BASE.uv, detail: l })
  );

  if (!fs.existsSync(uvExe)) {
    throw new BootstrapError("uv.exe не найден после распаковки", logPath);
  }
  return uvExe;
}

// --- главная процедура ------------------------------------------------------

/**
 * Полный bootstrap. Возвращает { gpu } для отображения режима в UI.
 * Кидает BootstrapError при неудаче (в нём путь к логу).
 */
export async function runBootstrap(paths: Paths, onProgress: ProgressCb): Promise<{ gpu: boolean }> {
  fs.mkdirSync(paths.dataDir, { recursive: true });
  fs.mkdirSync(paths.logsDir, { recursive: true });
  const logPath = path.join(paths.logsDir, "bootstrap.log");
  appendLog(logPath, `\n===== bootstrap ${new Date().toISOString()} =====\n`);

  const emit: ProgressCb = (p) => onProgress(p);

  try {
    emit({ stage: "check", label: "Проверяю окружение…", percent: STAGE_BASE.check });

    // 1. uv
    const uv = await ensureUv(paths, logPath, emit);
    // Общий env для uv: uv хранит кэш Python внутри dataDir, чтобы не мусорить в системе.
    const uvEnv: NodeJS.ProcessEnv = {
      ...process.env,
      UV_PYTHON_INSTALL_DIR: path.join(paths.dataDir, "python"),
      UV_CACHE_DIR: path.join(paths.dataDir, "uv-cache"),
    };

    // 2. Python 3.12
    emit({ stage: "python", label: "Скачиваю Python 3.12…", percent: STAGE_BASE.python });
    await run(uv, ["python", "install", "3.12"], { logPath, env: uvEnv }, (l) =>
      emit({ stage: "python", label: "Скачиваю Python 3.12…", percent: STAGE_BASE.python, detail: l })
    );

    // 3. venv
    emit({ stage: "venv", label: "Создаю виртуальное окружение…", percent: STAGE_BASE.venv });
    // --clear: повторный bootstrap не должен падать на «venv уже существует».
    await run(
      uv,
      ["venv", paths.envDir, "--python", "3.12", "--clear"],
      { logPath, env: uvEnv },
      (l) => emit({ stage: "venv", label: "Создаю окружение…", percent: STAGE_BASE.venv, detail: l })
    );

    // 4. GPU-детект
    emit({ stage: "gpu", label: "Проверяю видеокарту…", percent: STAGE_BASE.gpu });
    const gpu = await detectGpu(logPath);
    emit({
      stage: "gpu",
      label: gpu ? "NVIDIA GPU найден" : "GPU не найден — режим CPU (медленнее)",
      percent: STAGE_BASE.gpu,
      gpu,
    });

    // uv pip работает в контексте созданного venv через --python.
    // --index-strategy unsafe-best-match: lock-файл — это `pip freeze` с
    // --extra-index-url на pytorch. uv по умолчанию берёт версии только с
    // первого индекса, где нашёлся пакет (first-index), и падает на пакетах,
    // которые pytorch зеркалит другой версией (certifi, idna, ...). pip так не
    // делает — ищет лучшую версию по всем индексам. Возвращаем это поведение;
    // безопасно, т.к. оба индекса (PyPI + официальный pytorch.org) доверенные.
    const pipBase = [
      "pip",
      "install",
      "--python",
      paths.venvPython,
      "--index-strategy",
      "unsafe-best-match",
    ];

    // 5. Зависимости (torch+cu126 для GPU, CPU-torch иначе)
    const reqFile = path.join(
      paths.backendDir,
      gpu ? "requirements-gpu.txt" : "requirements.txt"
    );
    emit({
      stage: "deps",
      label: gpu
        ? "Устанавливаю зависимости (torch+CUDA, ~2–3 ГБ)…"
        : "Устанавливаю зависимости (CPU-версия)…",
      percent: STAGE_BASE.deps,
      gpu,
    });
    await run(uv, [...pipBase, "-r", reqFile], { logPath, env: uvEnv }, (l) =>
      emit({ stage: "deps", label: "Устанавливаю зависимости…", percent: STAGE_BASE.deps, detail: l, gpu })
    );

    // 6. Делаем пакет twitch_cut импортируемым.
    //
    // РАНЬШЕ тут был `uv pip install -e backend --no-deps`. Он не работает для
    // упакованного приложения: backend лежит в read-only `Program Files`, а
    // setuptools (и uv, собирающий локальные пакеты ПРЯМО в папке-источнике)
    // пытается создать `src\*.egg-info` рядом с исходниками → «Отказано в
    // доступе». Копировать backend в writable-каталог ради editable-install —
    // лишнее: приложение зовёт backend только как `python -m twitch_cut.cli`
    // (см. main.ts serve и prefetch ниже), console-script и dist-метаданные не
    // используются. Поэтому просто кладём .pth с путём к src в site-packages —
    // ничего не собираем и не пишем в Program Files.
    emit({ stage: "package", label: "Устанавливаю twitch-cut…", percent: STAGE_BASE.package, gpu });
    const srcDir = path.join(paths.backendDir, "src");
    const writePthScript =
      "import sysconfig, os, sys\n" +
      "sp = sysconfig.get_paths()['purelib']\n" +
      "open(os.path.join(sp, 'twitch_cut.pth'), 'w', encoding='utf-8').write(sys.argv[1] + '\\n')\n" +
      "print('twitch_cut.pth ->', sp)\n";
    await run(
      paths.venvPython,
      ["-c", writePthScript, srcDir],
      { logPath, env: uvEnv },
      (l) => emit({ stage: "package", label: "Устанавливаю twitch-cut…", percent: STAGE_BASE.package, detail: l, gpu })
    );

    // 7. Модели (~4 GB). Прогресс идёт из tqdm в stderr — прокидываем как detail.
    emit({ stage: "models", label: "Скачиваю модели (~4 ГБ, это долго)…", percent: STAGE_BASE.models, gpu });
    const modelsEnv: NodeJS.ProcessEnv = {
      ...process.env,
      TWITCH_CUT_DATA_DIR: paths.dataDir,
      PYTHONUNBUFFERED: "1",
      // См. main.ts buildBackendEnv: на русской Windows дочерний Python иначе
      // берёт cp1251 и падает UnicodeEncodeError на rich-print «→» в prefetch.
      PYTHONUTF8: "1",
      PYTHONIOENCODING: "utf-8",
    };
    await run(
      paths.venvPython,
      ["-m", "twitch_cut.cli", "prefetch"],
      { logPath, env: modelsEnv, cwd: paths.backendDir },
      (l) => emit({ stage: "models", label: "Скачиваю модели…", percent: STAGE_BASE.models, detail: l, gpu })
    );

    // 8. Sentinel
    fs.writeFileSync(path.join(paths.envDir, ".bootstrap-ok"), BOOTSTRAP_VERSION, "utf-8");
    emit({ stage: "done", label: "Готово", percent: 100, gpu });
    return { gpu };
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    appendLog(logPath, `\n[FATAL] ${msg}\n`);
    emit({ stage: "error", label: `Ошибка установки: ${msg}`, percent: 0, detail: `Лог: ${logPath}` });
    throw new BootstrapError(msg, logPath);
  }
}
