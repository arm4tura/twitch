import { useEffect, type ReactNode } from "react";
import { GripVertical } from "lucide-react";
import { cn } from "../../lib/cn";
import { toast } from "./Toast";

/**
 * DraggableFile — плитка файла, которую можно вытащить ИЗ приложения наружу:
 * в браузер (форма загрузки на YouTube/TikTok), в проводник, на рабочий стол.
 *
 * Механика: в onDragStart зовём preload `window.twitchCut.startDrag(path)`,
 * который в main-процессе триггерит нативный `webContents.startDrag`. Дальше
 * перетаскиванием управляет ОС — целевое приложение получает НАСТОЯЩИЙ файл
 * (не URL, не blob), как если бы его тащили из Finder/Explorer.
 *
 * draggable=true на самом div обязателен, иначе dragstart не выстрелит. В
 * dragstart зовём e.preventDefault(), чтобы погасить стандартный HTML5-drag
 * (он тащит text/uri, а не файл) — иначе он перебивает нативный startDrag и
 * целевое приложение получает URL вместо файла. После preventDefault Electron
 * берёт управление и отдаёт ОС настоящий файл.
 *
 * Клик по плитке (не drag) — showInFolder: показать файл в системном
 * проводнике. Двойное назначение удобно: перетащил или открыл папку.
 */
export interface DraggableFileProps {
  /** Абсолютный путь к файлу на диске. */
  path: string;
  /** Заголовок (обычно имя клипа/файла). */
  title: ReactNode;
  /** Подзаголовок — длительность, размер и т.п. */
  subtitle?: ReactNode;
  /** Иконка слева. */
  icon?: ReactNode;
  className?: string;
  /** Правый слот — например мини-превью или кнопка ▶. */
  trailing?: ReactNode;
}

export function DraggableFile({
  path,
  title,
  subtitle,
  icon,
  className,
  trailing,
}: DraggableFileProps) {
  // Ошибка нативного drag приходит из main асинхронно (startDrag — fire-and-
  // forget, вернуть её синхронно из onDragStart нельзя). Подписываемся и
  // показываем тост, если файла нет или startDrag упал в main — иначе плитка
  // просто «не тянется» без объяснения.
  useEffect(() => {
    const off = window.twitchCut?.onStartDragError?.((info) => {
      if (info.path !== path) return; // не наша плитка
      if (info.reason === "not-found") {
        toast.error("Файл не найден", { description: path });
      } else {
        toast.error("Не удалось начать перетаскивание", {
          description: info.message ?? path,
        });
      }
    });
    return off;
  }, [path]);

  const onDragStart = (e: React.DragEvent) => {
    // КЛЮЧЕВОЙ момент: гасим стандартный HTML5-drag. Без preventDefault браузер
    // начинает свой drag (тащит text/uri, а не файл) и перебивает нативный
    // startDrag — целевое приложение (NotebookLM, форма загрузки) получает URL,
    // а не файл, и «не реагирует». С preventDefault Electron.startDrag берёт
    // управление и отдаёт ОС НАСТОЯЩИЙ файл — ровно как перетаскивание из Finder.
    e.preventDefault();
    try {
      window.twitchCut.startDrag(path);
    } catch {
      /* startDrag недоступен (напр. web-preview) — деградируем в no-op */
    }
  };

  const reveal = async () => {
    const ok = await window.twitchCut.showInFolder(path);
    if (!ok) {
      /* тихо: карточка и так подскажет через заголовок */
    }
  };

  return (
    <div
      draggable
      onDragStart={onDragStart}
      onClick={reveal}
      title="Перетащите файл в браузер/папку или кликните, чтобы открыть в проводнике"
      className={cn(
        "group flex cursor-grab items-center gap-3 rounded-lg border border-white/8 bg-black/20 p-2.5",
        "transition-colors hover:border-brand-from/40 hover:bg-black/30 active:cursor-grabbing",
        "select-none",
        className
      )}
    >
      <GripVertical className="h-4 w-4 shrink-0 text-subtle transition-colors group-hover:text-brand-from" />
      {icon && <span className="shrink-0 text-brand-from">{icon}</span>}
      <div className="min-w-0 flex-1">
        <div className="truncate text-sm font-medium text-fg">{title}</div>
        {subtitle != null && (
          <div className="truncate text-[11px] text-subtle">{subtitle}</div>
        )}
      </div>
      {trailing}
    </div>
  );
}
