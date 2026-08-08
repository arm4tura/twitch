import { type ReactNode, useEffect, useState } from "react";
import { LayoutDashboard, Activity, Film, Send, Sparkles, Settings } from "lucide-react";
import { cn } from "./lib/cn";
import { Kbd } from "./components/ui/Kbd";
import type { Screen } from "./App";

/**
 * AppShell — persistent glass-sidebar + content-area.
 *
 * Sidebar фиксированной ширины 240px (без collapse в этом коммите —
 * добавим ⌘\ в Polish-фазе). Nav-items — vertical stack, активный элемент
 * подсвечен brand-градиентом слева.
 *
 * Content-area — flex-1, скроллится; переходы между экранами добавлю в
 * коммите 5 через framer-motion <AnimatePresence>.
 */

interface NavItem {
  id: Screen;
  label: string;
  icon: ReactNode;
  /** Заблокирован пока нет активной джобы / decisions.json. */
  disabled?: boolean;
  /** Почему пункт заблокирован — показывается как title при наведении. */
  disabledReason?: string;
  /** Показать бейдж с количеством (running jobs и т.п.). */
  badge?: number;
  shortcut?: string;
  /** Нарисовать разделитель НАД этим пунктом (визуально отделяет группу). */
  dividerBefore?: boolean;
}

export interface AppShellProps {
  screen: Screen;
  onNavigate: (screen: Screen) => void;
  runningJobs: number;
  hasActiveJob: boolean;
  hasDecisions: boolean;
  children: ReactNode;
}

export function AppShell({
  screen,
  onNavigate,
  runningJobs,
  hasActiveJob,
  hasDecisions,
  children,
}: AppShellProps) {
  // Пока идёт обработка — правку и экспорт блокируем: результата ещё нет,
  // а старый decisions.json может перезаписаться на середине.
  const busy = runningJobs > 0;
  const items: NavItem[] = [
    { id: "dashboard", label: "Проекты", icon: <LayoutDashboard className="h-4 w-4" />, shortcut: "⌘1" },
    {
      id: "job",
      label: "Прогресс",
      icon: <Activity className="h-4 w-4" />,
      disabled: !hasActiveJob,
      disabledReason: "Нет активной обработки — запустите новую в «Проектах»",
      badge: runningJobs || undefined,
      shortcut: "⌘3",
    },
    {
      id: "timeline",
      label: "Правка",
      icon: <Film className="h-4 w-4" />,
      disabled: !hasDecisions || busy,
      disabledReason: busy
        ? "Идёт обработка — правка станет доступна после завершения"
        : "Сначала обработайте запись, чтобы появилась разметка",
      shortcut: "⌘4",
    },
    {
      id: "export",
      label: "Экспорт",
      icon: <Send className="h-4 w-4" />,
      disabled: !hasDecisions || busy,
      disabledReason: busy
        ? "Идёт обработка — экспорт станет доступен после завершения"
        : "Сначала обработайте запись, чтобы было что выгружать",
      shortcut: "⌘5",
    },
    {
      id: "clips",
      label: "AI Tiktok",
      icon: <Sparkles className="h-4 w-4" />,
      disabled: !hasDecisions,
      disabledReason: "Откройте проект — нарезка на клипы доступна после обработки",
      shortcut: "⌘6",
      dividerBefore: true,
    },
  ];

  return (
    <div className="grid h-full grid-cols-[240px_1fr]">
      {/* Sidebar */}
      <aside className="flex flex-col border-r border-white/5 bg-surface/40 backdrop-blur-xl">
        <div className="px-5 pt-5 pb-4">
          <div className="flex items-center gap-2.5">
            {/*
              Логотип: пытаемся показать .ico (пользовательская иконка), с
              автоматическим fallback на .png на случай, если конкретная сборка
              Chromium не декодирует ICO (Windows-only фичи форматов курсоров
              иногда ломают декодер). onError переставляет src на PNG-вариант,
              который заведомо есть рядом.
            */}
            <img
              src="./icon.ico"
              alt="Twitch Cut"
              onError={(e) => {
                const el = e.currentTarget;
                if (!el.src.endsWith(".png")) el.src = "./icon.png";
              }}
              className="h-8 w-8 rounded-lg shadow-glow"
              draggable={false}
            />
            <div>
              <div className="text-sm font-semibold text-fg leading-tight">Twitch Cut</div>
              <div className="text-[10px] text-subtle uppercase tracking-wider">Desktop</div>
            </div>
            <CpuBadge />
          </div>
        </div>

        <div className="mt-2 px-2">
          <div className="mb-2 px-3 text-[10px] font-medium uppercase tracking-wider text-subtle">
            Навигация
          </div>
          <nav className="flex flex-col gap-0.5">
            {items.map((item) => (
              <div key={item.id}>
                {item.dividerBefore && (
                  <div className="my-2 border-t border-white/5" aria-hidden />
                )}
                <NavButton
                  item={item}
                  active={screen === item.id}
                  onClick={() => !item.disabled && onNavigate(item.id)}
                />
              </div>
            ))}
          </nav>
        </div>

        <div className="mt-auto p-2">
          <button
            type="button"
            onClick={() => onNavigate("settings")}
            className={cn(
              "flex w-full items-center gap-3 rounded-lg px-3 py-2 text-sm transition-colors",
              "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-brand-from",
              screen === "settings"
                ? "bg-white/[0.06] text-fg"
                : "text-muted hover:bg-white/5 hover:text-fg"
            )}
          >
            <Settings
              className={cn(
                "h-4 w-4",
                screen === "settings" ? "text-brand-from" : "text-muted"
              )}
            />
            <span className="flex-1 text-left">Настройки</span>
            <Kbd className="opacity-70">⌘,</Kbd>
          </button>
        </div>
      </aside>

      {/* Content */}
      <main className="flex min-w-0 flex-col overflow-hidden">
        <div className="flex-1 overflow-y-auto">{children}</div>
      </main>
    </div>
  );
}

/**
 * Бейдж режима вычислений. Виден только когда backend работает на CPU
 * (GPU не найден при первом запуске) — предупреждает, что обработка медленнее.
 * В GPU-режиме ничего не рисуем, чтобы не засорять сайдбар.
 */
function CpuBadge() {
  const [cpu, setCpu] = useState(false);
  useEffect(() => {
    window.twitchCut
      ?.getGpuMode?.()
      .then((gpu) => setCpu(!gpu))
      .catch(() => {});
  }, []);
  if (!cpu) return null;
  return (
    <span
      title="Видеокарта NVIDIA не найдена — обработка идёт на CPU (медленнее)"
      className="ml-auto rounded-full bg-amber-500/15 px-2 py-0.5 text-[10px] font-medium text-amber-400"
    >
      CPU
    </span>
  );
}

function NavButton({
  item,
  active,
  onClick,
}: {
  item: NavItem;
  active: boolean;
  onClick: () => void;
}) {
  const button = (
    <button
      type="button"
      onClick={onClick}
      disabled={item.disabled}
      className={cn(
        "group relative flex w-full items-center gap-3 rounded-lg px-3 py-2 text-sm transition-colors",
        "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-brand-from",
        active
          ? "bg-white/[0.06] text-fg"
          : "text-muted hover:bg-white/[0.03] hover:text-fg",
        item.disabled && "opacity-40 cursor-not-allowed hover:bg-transparent hover:text-muted"
      )}
    >
      {active && (
        <span className="absolute left-0 top-1/2 h-5 w-0.5 -translate-y-1/2 rounded-r-full bg-brand shadow-glow" />
      )}
      <span className={cn("shrink-0", active ? "text-brand-from" : "text-muted")}>
        {item.icon}
      </span>
      <span className="flex-1 text-left">{item.label}</span>
      {item.badge != null && item.badge > 0 && (
        <span className="rounded-full bg-brand-from/20 px-1.5 py-0.5 text-[10px] font-medium text-brand-from">
          {item.badge}
        </span>
      )}
      {item.shortcut && !item.disabled && (
        <Kbd className="opacity-0 transition-opacity group-hover:opacity-100">
          {item.shortcut}
        </Kbd>
      )}
    </button>
  );

  // disabled-кнопка в Chromium не всегда показывает native title (не ловит
  // hover-события), поэтому вешаем подсказку на обёртку — она перехватывает
  // наведение и объясняет, почему пункт недоступен.
  if (item.disabled && item.disabledReason) {
    return <span title={item.disabledReason}>{button}</span>;
  }
  return button;
}
