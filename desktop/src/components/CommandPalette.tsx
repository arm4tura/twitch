import { useEffect, useMemo, useRef, useState } from "react";
import * as DialogPrimitive from "@radix-ui/react-dialog";
import {
  ArrowRight,
  Film,
  FolderOpen,
  LayoutDashboard,
  Plus,
  Redo2,
  Save,
  Search,
  Send,
  Sparkles,
  Undo2,
} from "lucide-react";
import { cn } from "../lib/cn";
import { Kbd } from "./ui/Kbd";
import { useHotkey } from "../hooks/useHotkey";
import type { Screen } from "../App";

/**
 * CommandPalette — ⌘K overlay поверх всего приложения.
 *
 * Хранит список команд трёх типов:
 *   1. Навигация — «Открыть Dashboard/Timeline/…», disabled если контекст пуст
 *      (например Timeline без decisionsPath).
 *   2. Контекстные действия — «Сохранить», «Undo», «Redo», «Показать decisions.json
 *      в проводнике». Активны только когда предоставлен соответствующий handler.
 *   3. Внешние — «Открыть NotebookLM».
 *
 * Реализация — Radix Dialog (фокус-трап + esc + backdrop-click готовы), сверху
 * наш Input + список. Fuzzy-фильтр упрощён до case-insensitive substring поиска
 * — команд <20, полноценный fuzzy не нужен и добавил бы вес в bundle.
 *
 * Хоткей ⌘K регистрируется здесь же и работает даже когда фокус в <input> —
 * так и должно быть.
 */

export interface CommandContext {
  screen: Screen;
  hasActiveJob: boolean;
  hasDecisions: boolean;
  decisionsPath: string | null;
  /** Handlers, которые компонент-владелец Timeline'а вешает через ref/prop. */
  onNavigate: (screen: Screen) => void;
  onSave?: () => void;
  onUndo?: () => void;
  onRedo?: () => void;
  canUndo?: boolean;
  canRedo?: boolean;
  dirty?: boolean;
}

interface Command {
  id: string;
  title: string;
  hint?: string;
  section: "Навигация" | "Действия" | "Внешние";
  icon: React.ReactNode;
  shortcut?: string;
  disabled?: boolean;
  run: () => void;
}

export function CommandPalette(props: CommandContext) {
  const [open, setOpen] = useState(false);
  const [query, setQuery] = useState("");
  const inputRef = useRef<HTMLInputElement>(null);
  const [activeIdx, setActiveIdx] = useState(0);

  // Открытие ⌘K — глобально, работает даже в <input>.
  useHotkey(["mod+k"], (e) => {
    e.preventDefault();
    setOpen((v) => !v);
  }, { allowInInput: true });

  // Сброс поиска и подсветки при закрытии, автофокус при открытии.
  useEffect(() => {
    if (open) {
      setQuery("");
      setActiveIdx(0);
      // Focus после того как Radix смонтирует контент.
      queueMicrotask(() => inputRef.current?.focus());
    }
  }, [open]);

  const commands = useMemo<Command[]>(() => {
    const nav = (id: Screen, title: string, icon: React.ReactNode, shortcut: string, disabled = false): Command => ({
      id: `nav-${id}`,
      title,
      section: "Навигация",
      icon,
      shortcut,
      disabled,
      run: () => props.onNavigate(id),
    });

    const list: Command[] = [
      nav("dashboard", "Открыть проекты", <LayoutDashboard className="h-4 w-4" />, "⌘1"),
      nav("new", "Новый job", <Plus className="h-4 w-4" />, "⌘2"),
      nav("job", "Прогресс", <Sparkles className="h-4 w-4" />, "⌘3", !props.hasActiveJob),
      nav("timeline", "Таймлайн", <Film className="h-4 w-4" />, "⌘4", !props.hasDecisions),
      nav("export", "Экспорт", <Send className="h-4 w-4" />, "⌘5", !props.hasDecisions),
    ];

    if (props.onSave) {
      list.push({
        id: "act-save",
        title: props.dirty ? "Сохранить изменения" : "Сохранено — правок нет",
        section: "Действия",
        icon: <Save className="h-4 w-4" />,
        shortcut: "⌘S",
        disabled: !props.dirty,
        run: () => props.onSave!(),
      });
    }
    if (props.onUndo) {
      list.push({
        id: "act-undo",
        title: "Отменить",
        section: "Действия",
        icon: <Undo2 className="h-4 w-4" />,
        shortcut: "⌘Z",
        disabled: !props.canUndo,
        run: () => props.onUndo!(),
      });
    }
    if (props.onRedo) {
      list.push({
        id: "act-redo",
        title: "Вернуть",
        section: "Действия",
        icon: <Redo2 className="h-4 w-4" />,
        shortcut: "⌘⇧Z",
        disabled: !props.canRedo,
        run: () => props.onRedo!(),
      });
    }
    if (props.decisionsPath) {
      list.push({
        id: "act-reveal",
        title: "Показать decisions.json в проводнике",
        section: "Действия",
        icon: <FolderOpen className="h-4 w-4" />,
        run: () => {
          void window.twitchCut.showInFolder(props.decisionsPath!);
        },
      });
    }

    list.push({
      id: "ext-notebooklm",
      title: "Открыть NotebookLM в браузере",
      hint: "notebooklm.google.com",
      section: "Внешние",
      icon: <ArrowRight className="h-4 w-4" />,
      run: () => {
        void window.twitchCut.openExternal("https://notebooklm.google.com/");
      },
    });

    return list;
  }, [
    props.hasActiveJob,
    props.hasDecisions,
    props.decisionsPath,
    props.dirty,
    props.canUndo,
    props.canRedo,
    props.onSave,
    props.onUndo,
    props.onRedo,
    props.onNavigate,
  ]);

  // Фильтр — простой substring, регистронезависимый. Дизейбл сохраняем,
  // но пропускаем в конец списка через сортировку.
  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase();
    const res = q
      ? commands.filter((c) => c.title.toLowerCase().includes(q))
      : commands;
    return [...res].sort((a, b) => Number(!!a.disabled) - Number(!!b.disabled));
  }, [commands, query]);

  // Держим activeIdx в границах при смене фильтра/списка.
  useEffect(() => {
    setActiveIdx((i) => Math.min(Math.max(0, i), Math.max(0, filtered.length - 1)));
  }, [filtered.length]);

  const runAt = (i: number) => {
    const cmd = filtered[i];
    if (!cmd || cmd.disabled) return;
    setOpen(false);
    // requestAnimationFrame — чтобы диалог успел закрыться до навигации.
    // Иначе Radix может дважды дёрнуть focus-return.
    requestAnimationFrame(cmd.run);
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "ArrowDown") {
      e.preventDefault();
      setActiveIdx((i) => Math.min(filtered.length - 1, i + 1));
    } else if (e.key === "ArrowUp") {
      e.preventDefault();
      setActiveIdx((i) => Math.max(0, i - 1));
    } else if (e.key === "Enter") {
      e.preventDefault();
      runAt(activeIdx);
    }
  };

  // Группируем по секциям, сохраняя порядок внутри filtered.
  const sections = useMemo(() => {
    const map = new Map<string, Command[]>();
    for (const c of filtered) {
      const arr = map.get(c.section) ?? [];
      arr.push(c);
      map.set(c.section, arr);
    }
    return Array.from(map.entries());
  }, [filtered]);

  // Плоский индекс → item — чтобы подсветить строку по activeIdx.
  const flatIndexOf = (cmd: Command) => filtered.indexOf(cmd);

  return (
    <DialogPrimitive.Root open={open} onOpenChange={setOpen}>
      <DialogPrimitive.Portal>
        <DialogPrimitive.Overlay className="fixed inset-0 z-50 bg-black/60 backdrop-blur-sm data-[state=open]:animate-fade-in data-[state=closed]:animate-fade-out" />
        <DialogPrimitive.Content
          className="fixed left-1/2 top-[18%] z-50 w-full max-w-xl -translate-x-1/2 overflow-hidden rounded-card border border-white/10 bg-surface/95 shadow-card backdrop-blur-xl data-[state=open]:animate-slide-up"
          onKeyDown={handleKeyDown}
        >
          <DialogPrimitive.Title className="sr-only">
            Командная палитра
          </DialogPrimitive.Title>

          <div className="flex items-center gap-2 border-b border-white/8 px-3 py-2.5">
            <Search className="h-4 w-4 shrink-0 text-subtle" />
            <input
              ref={inputRef}
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Что сделать? — навигация, действия, ссылки…"
              className="flex-1 bg-transparent text-sm text-fg outline-none placeholder:text-subtle"
              autoComplete="off"
              spellCheck={false}
            />
            <Kbd>Esc</Kbd>
          </div>

          <div className="max-h-[50vh] overflow-y-auto py-2">
            {filtered.length === 0 && (
              <div className="px-4 py-8 text-center text-sm text-subtle">
                Ничего не найдено
              </div>
            )}
            {sections.map(([section, cmds]) => (
              <div key={section} className="mb-1 last:mb-0">
                <div className="px-4 pb-1 pt-2 text-[10px] font-medium uppercase tracking-wider text-subtle">
                  {section}
                </div>
                {cmds.map((cmd) => {
                  const idx = flatIndexOf(cmd);
                  const active = idx === activeIdx;
                  return (
                    <button
                      key={cmd.id}
                      type="button"
                      disabled={cmd.disabled}
                      onMouseEnter={() => setActiveIdx(idx)}
                      onClick={() => runAt(idx)}
                      className={cn(
                        "flex w-full items-center gap-3 px-4 py-2 text-left text-sm transition-colors",
                        "focus-visible:outline-none",
                        active && !cmd.disabled && "bg-brand-from/10 text-fg",
                        !active && !cmd.disabled && "text-muted hover:bg-white/[0.03] hover:text-fg",
                        cmd.disabled && "cursor-not-allowed opacity-40"
                      )}
                    >
                      <span className={cn("shrink-0", active ? "text-brand-from" : "text-muted")}>
                        {cmd.icon}
                      </span>
                      <span className="flex-1 truncate">{cmd.title}</span>
                      {cmd.hint && (
                        <span className="truncate font-mono text-[10px] text-subtle">
                          {cmd.hint}
                        </span>
                      )}
                      {cmd.shortcut && <Kbd>{cmd.shortcut}</Kbd>}
                    </button>
                  );
                })}
              </div>
            ))}
          </div>

          <div className="flex items-center justify-between border-t border-white/8 bg-black/30 px-3 py-2 text-[10px] text-subtle">
            <div className="flex items-center gap-3">
              <span className="flex items-center gap-1"><Kbd>↑↓</Kbd> навигация</span>
              <span className="flex items-center gap-1"><Kbd>↵</Kbd> запуск</span>
            </div>
            <span className="flex items-center gap-1"><Kbd>⌘K</Kbd> открыть/закрыть</span>
          </div>
        </DialogPrimitive.Content>
      </DialogPrimitive.Portal>
    </DialogPrimitive.Root>
  );
}
