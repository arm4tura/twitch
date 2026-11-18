/**
 * First-run bootstrap: докачивает модели распознавания БЕЗ участия пользователя.
 *
 * Раньше здесь при первом запуске качались uv и CPython, создавался venv и
 * ставились torch+CUDA (~10 ГБ, 10–15 минут). Теперь бэкенд собирается заранее
 * (PyInstaller, backend/packaging/) и едет внутри установщика — Python и все
 * зависимости уже на месте. Первый запуск только:
 *   1. убирает окружение старых версий (venv с torch, CPython, uv, веса
 *      torch-моделей) — новая версия ими не пользуется, а это ~10 ГБ;
 *   2. `twitch-cut-backend prefetch` — качает ONNX-модели (~430 МБ);
 *   3. `twitch-cut-backend accel` — узнаёт, какой ускоритель поднимается
 *      (бейдж GPU/CPU в сплэше и в сайдбаре);
 *   4. пишет sentinel `<dataDir>/.bootstrap-ok`, чтобы больше не повторять.
 *
 * Прогресс стримится в splash через onProgress. Полный лог пишется в
 * <logsDir>/bootstrap.log (кнопка «Показать лог» в сплэше открывает его).
 *
 * Идемпотентно: если sentinel свежий и бэкенд на месте — возвращаемся сразу.
 * При ошибке любой стадии кидаем BootstrapError с путём к логу.
 */

import { spawn } from "child_process";
import * as fs from "fs";
import * as path from "path";

// --- типы -------------------------------------------------------------------

export interface Paths {
  /** Writable корень данных (userData или portable-папка). */
  dataDir: string;
  /** Логи: <dataDir>/logs */
  logsDir: string;
  /** Рабочий каталог бэкенда (resources/backend в prod, <repo>/backend в dev). */
  backendDir: string;
  /** Исполняемый файл бэкенда: собранный twitch-cut-backend или python из dev-venv. */
  backendCmd: string;
  /** Аргументы перед командой: [] у собранного, ["-m", "twitch_cut.cli"] у python. */
  backendArgs: string[];
  /** Корень dev-venv (для CUDA-DLL WhisperX в PATH); null у собранного бэкенда. */
  venvRoot: string | null;
}

/** Ускоритель для бейджа: cuda — любой GPU (DirectML/CUDA), mps — CoreML. */
export type Accel = "cuda" | "mps" | "cpu";

export interface BootstrapProgress {
  /** Машинный ключ стадии. */
  stage: "check" | "cleanup" | "models" | "accel" | "done" | "error";
  /** Человекочитаемая строка для сплэша (RU). */
  label: string;
  /** Общий прогресс 0..100. */
  percent: number;
  /** Последняя строка лога/детали (хвост процесса). */
  detail?: string;
  /** Есть ли ускоритель (заполняется после стадии accel). */
  gpu?: boolean;
  /** Тип ускорителя для бейджа. */
  accel?: Accel;
}

export type ProgressCb = (p: BootstrapProgress) => void;

export class BootstrapError extends Error {
  constructor(message: string, public readonly logPath: string) {
    super(message);
    this.name = "BootstrapError";
  }
}

// --- константы --------------------------------------------------------------

/** Версия схемы bootstrap — если поменяем шаги, бампаем и sentinel протухает. */
const BOOTSTRAP_VERSION = "4";

/**
 * Что оставили версии с torch-окружением (BOOTSTRAP_VERSION ≤ 3) — пути
 * относительно dataDir. Всё это рантайм и кэши моделей, которыми новая
 * версия не пользуется; пользовательских данных (проекты, настройки, ключи,
 * models/onnx) здесь нет.
 */
const LEGACY_DIRS = [
  "env", // venv с torch/whisperx/pyannote
  "python", // CPython от uv
  "uv-cache",
  "bin", // uv
  path.join("models", "gigaam"), // torch-чекпоинт GigaAM
  path.join("models", "vad"), // torch-чекпоинт pyannote
  path.join("models", "hub"), // HF-кэш WhisperX (faster-whisper, wav2vec2)
  path.join("models", "torch"),
  path.join("models", "ffmpeg_bin"),
];
const LEGACY_FILES = [".cpu-mode"];

// --- утилиты процессов ------------------------------------------------------

/**
 * Убрать ANSI escape-последовательности (цвет/стиль от rich) из строки.
 * В лог пишем сырой текст, а в UI (splash detail) отдаём уже очищенный — иначе
 * пользователь видит хвосты вида `[2m…[0m` (сам ESC-байт невидим, коды остаются).
 */
function stripAnsi(text: string): string {
  // eslint-disable-next-line no-control-regex
  return text.replace(/\x1b\[[0-9;?]*[a-zA-Z]/g, "");
}

function appendLog(logPath: string, text: string): void {
  try {
    fs.appendFileSync(logPath, text);
  } catch {
    /* лог — best-effort, не роняем bootstrap из-за него */
  }
}

/**
 * Запустить процесс, стримя stdout/stderr в лог и последнюю строку — в onLine.
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
      if (!toErr) out += text;
      appendLog(opts.logPath, (toErr ? "[err] " : "") + text);
      const lines = text.split(/\r?\n/).filter((l) => l.trim().length > 0);
      const last = lines[lines.length - 1];
      if (last && onLine) onLine(stripAnsi(last).slice(0, 160));
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

/** Окружение для запуска бэкенда из bootstrap (как у serve в main.ts). */
function backendEnv(paths: Paths): NodeJS.ProcessEnv {
  return {
    ...process.env,
    TWITCH_CUT_DATA_DIR: paths.dataDir,
    PYTHONUNBUFFERED: "1",
    // См. main.ts buildBackendEnv: на русской Windows дочерний Python иначе
    // берёт cp1251 и падает UnicodeEncodeError на rich-print «→» в prefetch.
    PYTHONUTF8: "1",
    PYTHONIOENCODING: "utf-8",
  };
}

// --- детект состояния -------------------------------------------------------

function sentinelPath(paths: Paths): string {
  return path.join(paths.dataDir, ".bootstrap-ok");
}

/** Готово ли окружение: бэкенд на месте + свежий sentinel. */
export function isBootstrapped(paths: Paths): boolean {
  if (!fs.existsSync(paths.backendCmd) || !fs.existsSync(sentinelPath(paths))) return false;
  try {
    return fs.readFileSync(sentinelPath(paths), "utf-8").trim() === BOOTSTRAP_VERSION;
  } catch {
    return false;
  }
}

/** Ускоритель, который узнал прошлый bootstrap (файл .accel), или null. */
export function savedAccel(paths: Paths): Accel | null {
  try {
    const value = fs.readFileSync(path.join(paths.dataDir, ".accel"), "utf-8").trim();
    return value === "cuda" || value === "mps" || value === "cpu" ? value : null;
  } catch {
    return null;
  }
}

/** Имя провайдера onnxruntime (`accel` бэкенда) → бейдж. */
function accelFromProvider(provider: string): Accel {
  if (provider === "CoreML") return "mps";
  if (provider === "CPU") return "cpu";
  return "cuda"; // Dml, CUDA, TensorRT…
}

// --- стадии -----------------------------------------------------------------

/** Снести окружение версий с torch. Ошибки не фатальны — только место на диске. */
function removeLegacyRuntime(paths: Paths, logPath: string, emit: ProgressCb): void {
  const targets = [...LEGACY_DIRS, ...LEGACY_FILES]
    .map((rel) => path.join(paths.dataDir, rel))
    .filter((p) => fs.existsSync(p));
  if (targets.length === 0) return;
  emit({
    stage: "cleanup",
    label: "Удаляю окружение прошлой версии (torch, ~10 ГБ)…",
    percent: 2,
  });
  for (const target of targets) {
    try {
      fs.rmSync(target, { recursive: true, force: true });
      appendLog(logPath, `removed legacy: ${target}\n`);
    } catch (e) {
      appendLog(logPath, `[warn] не удалось удалить ${target}: ${String(e)}\n`);
    }
  }
}

// --- главная процедура ------------------------------------------------------

/**
 * Полный bootstrap. Возвращает ускоритель для отображения режима в UI.
 * Кидает BootstrapError при неудаче (в нём путь к логу).
 */
export async function runBootstrap(
  paths: Paths,
  onProgress: ProgressCb
): Promise<{ gpu: boolean; accel: Accel }> {
  fs.mkdirSync(paths.dataDir, { recursive: true });
  fs.mkdirSync(paths.logsDir, { recursive: true });
  const logPath = path.join(paths.logsDir, "bootstrap.log");
  appendLog(logPath, `\n===== bootstrap ${new Date().toISOString()} =====\n`);

  const emit: ProgressCb = (p) => onProgress(p);
  const env = backendEnv(paths);

  try {
    emit({ stage: "check", label: "Проверяю окружение…", percent: 0 });
    if (!fs.existsSync(paths.backendCmd) && path.isAbsolute(paths.backendCmd)) {
      throw new BootstrapError(`Не найден бэкенд: ${paths.backendCmd}`, logPath);
    }

    // 1. Окружение прошлых версий (venv с torch и т.п.).
    removeLegacyRuntime(paths, logPath, emit);

    // 2. Модели (~430 МБ). prefetch печатает строки «имя: NN% (x / y МБ)» —
    //    проценты переводим в общий прогресс 5..90.
    emit({ stage: "models", label: "Скачиваю модели распознавания (~430 МБ)…", percent: 5 });
    await run(
      paths.backendCmd,
      [...paths.backendArgs, "prefetch"],
      { logPath, env, cwd: paths.backendDir },
      (line) => {
        const m = line.match(/(gigaam|vad)\S*: (\d+)%/);
        let percent = 5;
        if (m) {
          // VAD крошечный (6 МБ), почти весь объём — GigaAM.
          percent = m[1] === "vad" ? 5 : 5 + Math.round((parseInt(m[2], 10) * 85) / 100);
        }
        emit({ stage: "models", label: "Скачиваю модели распознавания…", percent, detail: line });
      }
    );

    // 3. Какой ускоритель реально поднимается (DirectML/CUDA/CoreML/CPU).
    emit({ stage: "accel", label: "Проверяю ускоритель…", percent: 92 });
    let accel: Accel = "cpu";
    try {
      const out = await run(paths.backendCmd, [...paths.backendArgs, "accel"], {
        logPath,
        env,
        cwd: paths.backendDir,
      });
      const json = out
        .split(/\r?\n/)
        .reverse()
        .find((l) => l.trim().startsWith("{"));
      if (json) accel = accelFromProvider(String(JSON.parse(json).accelerator ?? "CPU"));
    } catch (e) {
      // Бейдж — не повод валить установку: распознавание само уйдёт на CPU.
      appendLog(logPath, `[warn] accel: ${String(e)}\n`);
    }
    const gpu = accel !== "cpu";
    emit({
      stage: "accel",
      label:
        accel === "cuda"
          ? "Видеокарта найдена — распознавание на GPU"
          : accel === "mps"
          ? "Apple — ускорение CoreML"
          : "Распознавание на CPU",
      percent: 96,
      gpu,
      accel,
    });

    // 4. Sentinel
    fs.writeFileSync(path.join(paths.dataDir, ".accel"), accel, "utf-8");
    fs.writeFileSync(sentinelPath(paths), BOOTSTRAP_VERSION, "utf-8");
    emit({ stage: "done", label: "Готово", percent: 100, gpu, accel });
    return { gpu, accel };
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    appendLog(logPath, `\n[FATAL] ${msg}\n`);
    emit({ stage: "error", label: `Ошибка установки: ${msg}`, percent: 0, detail: `Лог: ${logPath}` });
    throw err instanceof BootstrapError ? err : new BootstrapError(msg, logPath);
  }
}
