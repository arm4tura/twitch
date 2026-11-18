import { useEffect, useRef, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { AppShell } from "./AppShell";
import { DashboardScreen } from "./screens/DashboardScreen";
import { NewJobScreen } from "./screens/NewJobScreen";
import { JobScreen } from "./screens/JobScreen";
import { TimelineScreen } from "./screens/TimelineScreen";
import { ExportScreen } from "./screens/ExportScreen";
import { AiTiktokScreen } from "./screens/AiTiktokScreen";
import { SettingsScreen } from "./screens/SettingsScreen";
import { Toaster } from "./components/ui/Toast";
import { ErrorBoundary } from "./components/ErrorBoundary";
import { CommandPalette } from "./components/CommandPalette";
import { useTimelineActions } from "./lib/timelineActions";
import { useHotkey } from "./hooks/useHotkey";
import { listJobs } from "./api";

/**
 * App — root shell. Никакого SPA-router'a: пять экранов + useState<Screen>.
 * Навигация допускается только когда данные готовы (JobScreen требует
 * activeJobId; Timeline/Export — decisionsPath). AppShell рендерит sidebar
 * с disabled-состояниями по этим же флагам.
 *
 * Коммит 5: framer-motion `<AnimatePresence>` для fade-slide между экранами,
 * ⌘K CommandPalette через глобальный TimelineActions-store, sonner Toaster.
 */
export type Screen = "dashboard" | "new" | "job" | "timeline" | "export" | "clips" | "settings";

// Единый variant — крошечное движение, никакого «wow»: mac-style crossfade
// с 4px vertical drift. 180ms — достаточно чтобы взгляд заметил, слишком мало
// чтобы раздражать при частом переключении. prefers-reduced-motion уважается
// глобально через styles.css (transition-duration 0.001ms).
const screenVariants = {
  initial: { opacity: 0, y: 4 },
  animate: { opacity: 1, y: 0 },
  exit: { opacity: 0, y: -4 },
};

export function App() {
  const [screen, setScreen] = useState<Screen>("dashboard");
  const [activeJobId, setActiveJobId] = useState<string | null>(null);
  const [decisionsPath, setDecisionsPath] = useState<string | null>(null);
  // Единственная точка правды для sidebar-badge: опрашиваем /jobs каждые 3с.
  // Dashboard-экран рендерит свои StatCard'ы из того же эндпоинта, но тянет
  // их независимо — badge важен даже когда пользователь на Timeline/Export.
  const [runningJobs, setRunningJobs] = useState(0);
  // «Тяжёлая» обработка = всё, что перезаписывает decisions.json целиком
  // (process/download/render/export). Только она блокирует «Правку»/«Экспорт».
  // Детект оффтопа/тишины запускается ИЗ самой «Правки» — блокировать её им
  // нельзя, иначе пользователь застревает (ушёл в TikTok — назад уже никак).
  const [busy, setBusy] = useState(false);
  const timelineActions = useTimelineActions();
  // Ручка «опросить /jobs прямо сейчас». Нужна, чтобы после финала джобы не
  // ждать до 3с очередного тика: `busy` гасит «Правку»/«Экспорт», и без
  // немедленного опроса вкладки оставались бы серыми ещё пару секунд после
  // того, как обработка уже закончилась.
  const refreshJobsRef = useRef<(() => void) | null>(null);

  useEffect(() => {
    let alive = true;
    const tick = async () => {
      try {
        const jobs = await listJobs();
        if (!alive) return;
        const running = jobs.filter(
          (j) => j.status === "running" || j.status === "pending"
        );
        setRunningJobs(running.length);
        setBusy(
          running.some(
            (j) => j.kind !== "detect_offtopic" && j.kind !== "detect_silence"
          )
        );
      } catch {
        /* backend не готов — тихо, покажем при следующем тике */
      }
    };
    refreshJobsRef.current = () => void tick();
    tick();
    const timer = window.setInterval(tick, 3000);
    return () => {
      alive = false;
      refreshJobsRef.current = null;
      window.clearInterval(timer);
    };
  }, []);

  const goToJob = (jobId: string, decisions?: string) => {
    setActiveJobId(jobId);
    if (decisions) setDecisionsPath(decisions);
    setScreen("job");
  };

  const goToTimeline = (path: string) => {
    setDecisionsPath(path);
    setScreen("timeline");
  };

  const navigate = (target: Screen) => {
    // CommandPalette может позвать экран, для которого нет контекста.
    // Здесь центральный guard: молча игнорируем неготовые переходы.
    if (target === "job" && !activeJobId) return;
    if ((target === "timeline" || target === "export" || target === "clips") && !decisionsPath)
      return;
    setScreen(target);
  };

  // Глобальный хоткей "открыть настройки" — стандарт macOS/Chrome: ⌘,
  // На Windows/Linux — Ctrl+, (useHotkey сам маппит mod→ctrl).
  // allowInInput=true: пользователь ждёт что настройки откроются даже если
  // курсор в поле поиска на Dashboard'е.
  useHotkey("mod+,", () => setScreen("settings"), { allowInInput: true });

  return (
    <>
      <AppShell
        screen={screen}
        onNavigate={setScreen}
        runningJobs={runningJobs}
        busy={busy}
        hasActiveJob={!!activeJobId}
        hasDecisions={!!decisionsPath}
      >
        <AnimatePresence mode="wait" initial={false}>
          <motion.div
            key={screen}
            variants={screenVariants}
            initial="initial"
            animate="animate"
            exit="exit"
            transition={{ duration: 0.18, ease: [0.16, 1, 0.3, 1] }}
            className="h-full"
          >
            <ErrorBoundary label={screen}>
            {screen === "dashboard" && (
              <DashboardScreen
                onNew={() => setScreen("new")}
                onOpen={(path) => goToTimeline(path)}
                onExport={(path) => { setDecisionsPath(path); setScreen("export"); }}
              />
            )}
            {screen === "new" && <NewJobScreen onJobStarted={goToJob} />}
            {screen === "job" && activeJobId && (
              <JobScreen
                jobId={activeJobId}
                onDone={(p) => p && goToTimeline(p)}
                // Джоба завершилась и оставила decisions.json — открываем
                // «Правку»/«Экспорт» СРАЗУ, не дожидаясь клика по «Открыть
                // таймлайн». Экран не переключаем: пользователь мог уйти на
                // другую вкладку, дёргать его оттуда нельзя. Плюс немедленный
                // re-poll /jobs, чтобы `busy` снялся сейчас, а не через 3с.
                onReady={(p) => {
                  setDecisionsPath(p);
                  refreshJobsRef.current?.();
                }}
                // Цепочка джоб: download_vod по завершении сам стартует
                // process-джобу и перецепляет экран на неё (один поток «скачал →
                // обработал → таймлайн», ноль кликов). goToJob сбрасывает лог и
                // ETA, т.к. это уже другая джоба.
                onChainJob={(id) => goToJob(id)}
              />
            )}
            {screen === "timeline" && decisionsPath && (
              <TimelineScreen decisionsPath={decisionsPath} />
            )}
            {screen === "export" && decisionsPath && (
              <ExportScreen
                decisionsPath={decisionsPath}
                onJobStarted={(id) => goToJob(id)}
              />
            )}
            {screen === "clips" && decisionsPath && (
              <AiTiktokScreen
                decisionsPath={decisionsPath}
                // НЕ уводим на экран «Прогресс»: весь результат (плитки для
                // перетаскивания, список моментов) показывается прямо здесь.
                // Уход бы размонтировал экран и потерял локальный стейт job'а.
                // Только регистрируем job — бейдж/таб «Прогресс» подхватят его
                // (App и так поллит /jobs каждые 3с).
                onJobStarted={(id) => setActiveJobId(id)}
              />
            )}
            {screen === "settings" && <SettingsScreen />}
            </ErrorBoundary>
          </motion.div>
        </AnimatePresence>
      </AppShell>

      <CommandPalette
        screen={screen}
        hasActiveJob={!!activeJobId}
        hasDecisions={!!decisionsPath}
        decisionsPath={decisionsPath}
        onNavigate={navigate}
        onSave={timelineActions.onSave}
        onUndo={timelineActions.onUndo}
        onRedo={timelineActions.onRedo}
        canUndo={timelineActions.canUndo}
        canRedo={timelineActions.canRedo}
        dirty={timelineActions.dirty}
      />
      <Toaster />
    </>
  );
}
