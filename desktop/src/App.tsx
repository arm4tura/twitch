import { useEffect, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { AppShell } from "./AppShell";
import { DashboardScreen } from "./screens/DashboardScreen";
import { NewJobScreen } from "./screens/NewJobScreen";
import { JobScreen } from "./screens/JobScreen";
import { TimelineScreen } from "./screens/TimelineScreen";
import { ExportScreen } from "./screens/ExportScreen";
import { SettingsScreen } from "./screens/SettingsScreen";
import { Toaster } from "./components/ui/Toast";
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
export type Screen = "dashboard" | "new" | "job" | "timeline" | "export" | "settings";

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
  const timelineActions = useTimelineActions();

  useEffect(() => {
    let alive = true;
    const tick = async () => {
      try {
        const jobs = await listJobs();
        if (!alive) return;
        setRunningJobs(
          jobs.filter((j) => j.status === "running" || j.status === "pending").length
        );
      } catch {
        /* backend не готов — тихо, покажем при следующем тике */
      }
    };
    tick();
    const timer = window.setInterval(tick, 3000);
    return () => {
      alive = false;
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
    if ((target === "timeline" || target === "export") && !decisionsPath) return;
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
            {screen === "dashboard" && (
              <DashboardScreen
                onNew={() => setScreen("new")}
                onOpen={(path) => goToTimeline(path)}
              />
            )}
            {screen === "new" && <NewJobScreen onJobStarted={goToJob} />}
            {screen === "job" && activeJobId && (
              <JobScreen
                jobId={activeJobId}
                onDone={(p) => p && goToTimeline(p)}
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
            {screen === "settings" && <SettingsScreen />}
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
