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
}

const $ = (id: string) => document.getElementById(id)!;
const stageEl = $("stage");
const barEl = $("bar");
const detailEl = $("detail");
const badgeEl = $("badge");
const actionsEl = $("actions");
const appEl = $("app");

function render(p: Progress): void {
  stageEl.textContent = p.label;
  barEl.style.width = `${Math.max(0, Math.min(100, p.percent))}%`;
  detailEl.textContent = p.detail ?? "";

  // Бейдж GPU/CPU появляется после стадии детекта.
  if (typeof p.gpu === "boolean") {
    badgeEl.style.display = "inline-block";
    if (p.gpu) {
      badgeEl.textContent = "GPU (CUDA)";
      badgeEl.classList.remove("cpu");
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
