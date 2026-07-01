import {
  Pause,
  Play,
  RotateCcw,
  RotateCw,
  Save,
  ZoomIn,
  ZoomOut,
  Maximize2,
} from "lucide-react";
import { Button } from "../ui/Button";
import { Kbd } from "../ui/Kbd";
import { StatusBadge } from "../ui/StatusBadge";
import { Tabs, TabsList, TabsTrigger } from "../ui/Tabs";
import { fmtMs } from "../../lib/format";
import { cn } from "../../lib/cn";

/**
 * TimelineToolbar — верхняя панель таймлайна.
 *
 * Layout:
 *   [Play/Pause] [current / total]      [Mutes|Cuts|Highlights|All]      [Zoom-]  [Zoom]  [Zoom+]  [Fit]      [Undo] [Redo] [Save]
 *
 * Всё поднято выше на самом экране, чтобы держать волну на «холсте».
 * Play отдельно (кнопка-хиро), Save отдельно (правый край), середина —
 * контекстные фильтры + zoom.
 */

export type TimelineTab = "mutes" | "cuts" | "highlights" | "all";

const TAB_ITEMS: Array<{ value: TimelineTab; label: string; count?: number }> = [
  { value: "all", label: "Все" },
  { value: "mutes", label: "Muted" },
  { value: "cuts", label: "Cuts" },
  { value: "highlights", label: "Highlights" },
];

export interface TimelineToolbarProps {
  isPlaying: boolean;
  onPlayPause: () => void;
  currentMs: number;
  totalMs: number;
  tab: TimelineTab;
  onTabChange: (t: TimelineTab) => void;
  counts: Record<Exclude<TimelineTab, "all">, number>;
  zoom: number;
  onZoomIn: () => void;
  onZoomOut: () => void;
  onFit: () => void;
  canUndo: boolean;
  canRedo: boolean;
  onUndo: () => void;
  onRedo: () => void;
  dirty: boolean;
  onSave: () => void;
  saving?: boolean;
  className?: string;
}

export function TimelineToolbar(props: TimelineToolbarProps) {
  const {
    isPlaying, onPlayPause, currentMs, totalMs,
    tab, onTabChange, counts,
    zoom, onZoomIn, onZoomOut, onFit,
    canUndo, canRedo, onUndo, onRedo,
    dirty, onSave, saving,
    className,
  } = props;

  return (
    <div
      className={cn(
        "flex flex-wrap items-center gap-3 rounded-card border border-white/8 bg-surface/70 px-3 py-2 shadow-card backdrop-blur-md",
        className
      )}
    >
      {/* Play + time */}
      <div className="flex items-center gap-2 pr-3">
        <Button
          size="sm"
          variant="primary"
          onClick={onPlayPause}
          className="h-9 w-9 p-0"
          aria-label={isPlaying ? "Пауза" : "Играть"}
        >
          {isPlaying ? <Pause className="h-4 w-4" /> : <Play className="h-4 w-4" />}
        </Button>
        <div className="font-mono text-xs tabular-nums text-muted">
          <span className="text-fg">{fmtMs(currentMs)}</span>
          <span className="mx-1 text-subtle">/</span>
          <span>{fmtMs(totalMs)}</span>
        </div>
      </div>

      {/* Filters */}
      <Tabs value={tab} onValueChange={(v) => onTabChange(v as TimelineTab)}>
        <TabsList>
          {TAB_ITEMS.map((it) => (
            <TabsTrigger key={it.value} value={it.value}>
              {it.value === "all"
                ? it.label
                : `${it.label} · ${counts[it.value as Exclude<TimelineTab, "all">]}`}
            </TabsTrigger>
          ))}
        </TabsList>
      </Tabs>

      <div className="ml-auto flex items-center gap-2">
        {/* Zoom */}
        <div className="flex items-center gap-1 rounded-md border border-white/8 bg-black/30 px-1 py-1">
          <Button
            size="sm"
            variant="ghost"
            onClick={onZoomOut}
            className="h-7 w-7 p-0"
            aria-label="Zoom out"
          >
            <ZoomOut className="h-3.5 w-3.5" />
          </Button>
          <span className="min-w-[3.5rem] text-center font-mono text-[11px] tabular-nums text-muted">
            {Math.round(zoom)} px/s
          </span>
          <Button
            size="sm"
            variant="ghost"
            onClick={onZoomIn}
            className="h-7 w-7 p-0"
            aria-label="Zoom in"
          >
            <ZoomIn className="h-3.5 w-3.5" />
          </Button>
          <Button
            size="sm"
            variant="ghost"
            onClick={onFit}
            className="h-7 w-7 p-0"
            aria-label="Fit"
          >
            <Maximize2 className="h-3.5 w-3.5" />
          </Button>
        </div>

        {/* Undo/Redo */}
        <div className="flex items-center gap-1 rounded-md border border-white/8 bg-black/30 px-1 py-1">
          <Button
            size="sm"
            variant="ghost"
            onClick={onUndo}
            disabled={!canUndo}
            className="h-7 w-7 p-0"
            aria-label="Undo"
          >
            <RotateCcw className="h-3.5 w-3.5" />
          </Button>
          <Button
            size="sm"
            variant="ghost"
            onClick={onRedo}
            disabled={!canRedo}
            className="h-7 w-7 p-0"
            aria-label="Redo"
          >
            <RotateCw className="h-3.5 w-3.5" />
          </Button>
        </div>

        {/* Save */}
        <Button
          size="sm"
          variant={dirty ? "primary" : "secondary"}
          onClick={onSave}
          disabled={!dirty || saving}
          loading={saving}
        >
          <Save className="mr-1.5 h-3.5 w-3.5" />
          Сохранить <Kbd className="ml-2">⌘S</Kbd>
        </Button>

        {dirty && !saving && (
          <StatusBadge status="pending" label="есть правки" className="ml-1" />
        )}
      </div>
    </div>
  );
}
