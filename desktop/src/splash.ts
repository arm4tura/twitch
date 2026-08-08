/**
 * Splash-скрипт первого запуска. Отдельный vite-вход (splash.html), НЕ тянет
 * React/основной бандл — стартует мгновенно. Слушает bootstrap:progress из main
 * через window.twitchCut.onBootstrapProgress и рисует прогресс.
 *
 * На стадии "error" показывает кнопки «Повторить» / «Показать лог».
 * На "done" main сам закроет это окно и откроет главное.
 */

interface Progress {
  stage: string;
  label: string;
  percent: number;
  detail?: string;
  gpu?: boolean;
  accel?: "cuda" | "mps" | "cpu";
}

/** Последний известный ускоритель — чтобы стадии без accel не сбрасывали бейдж. */
let lastAccel: "cuda" | "mps" | "cpu" | null = null;

const $ = (id: string) => document.getElementById(id)!;
const stageEl = $("stage");
const barEl = $("bar");
const detailEl = $("detail");
const badgeEl = $("badge");
const actionsEl = $("actions");
const appEl = $("app");

/** Убрать ANSI escape-коды (rich/tqdm) — иначе в detail видны хвосты `[2m…[0m`. */
function stripAnsi(text: string): string {
  // eslint-disable-next-line no-control-regex
  return text.replace(/\x1b\[[0-9;?]*[a-zA-Z]/g, "");
}

function render(p: Progress): void {
  stageEl.textContent = p.label;
  barEl.style.width = `${Math.max(0, Math.min(100, p.percent))}%`;
  detailEl.textContent = p.detail ? stripAnsi(p.detail) : "";

  // Бейдж ускорителя. accel (cuda|mps|cpu) точнее булева gpu; для обратной
  // совместимости выводим gpu→cuda/cpu, если accel не прислали. Бейдж «липкий»:
  // раз показанный ускоритель не сбрасывается на стадиях без accel/gpu.
  const accel: "cuda" | "mps" | "cpu" | null =
    p.accel ?? (typeof p.gpu === "boolean" ? (p.gpu ? "cuda" : "cpu") : null);
  if (accel) lastAccel = accel;

  if (lastAccel) {
    badgeEl.style.display = "inline-block";
    badgeEl.classList.remove("cpu");
    if (lastAccel === "cuda") {
      badgeEl.textContent = "GPU (CUDA)";
    } else if (lastAccel === "mps") {
      badgeEl.textContent = "GPU (Apple Metal)";
    } else {
      badgeEl.textContent = "CPU — медленнее";
      badgeEl.classList.add("cpu");
    }
  }

  if (p.stage === "error") {
    appEl.classList.add("error");
    actionsEl.classList.add("show");
  } else {
    appEl.classList.remove("error");
    actionsEl.classList.remove("show");
  }
}

window.twitchCut.onBootstrapProgress((payload) => render(payload as Progress));

$("retry").addEventListener("click", async () => {
  actionsEl.classList.remove("show");
  appEl.classList.remove("error");
  stageEl.textContent = "Повторяю…";
  await window.twitchCut.retryBootstrap();
});

$("log").addEventListener("click", () => {
  void window.twitchCut.openBootstrapLog();
});
