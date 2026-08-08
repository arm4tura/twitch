import { memo, useEffect, useRef } from "react";
import { cn } from "../../lib/cn";

/**
 * SilenceOverlay — цветные блоки обнаруженной тишины поверх волны таймлайна.
 *
 * Аналог MuteOverlay, но для silence-сегментов: accepted (будут вырезаны) →
 * жёлто-оранжевый полупрозрачный блок; rejected (пропускаем) → приглушённый серый.
 * Клик по блоку → выбор; выделенный подсвечивается обводкой.
 *
 * Границы правятся родителем через onResize (драг края/тела) — здесь только
 * отрисовка по готовым координатам (audio-секунды).
 *
 * ВАЖНО про жесты: блоки кликабельны (pointer-events-auto), из-за чего колесо/
 * трекпад-жест над оранжевым блоком «съедался» оверлеем и не доходил до
 * TimelineCanvas снизу (зум/пан не работали именно над тишиной). Поэтому wheel-
 * события мы перехватываем на оверлее и вручную перекидываем на канвас-svg
 * (data-timeline-surface) — там живёт слушатель зума/пана.
 */

export interface SilenceBlock {
  id: string;
  start: number; // audio-сек
  end: number; // audio-сек
  active: boolean; // accepted → будет вырезан; rejected → приглушён
  selected: boolean;
}

export interface SilenceOverlayProps {
  blocks: SilenceBlock[];
  durationS: number;
  view: { start: number; end: number };
  onSelect?: (id: string) => void;
  className?: string;
}

const VBW = 1000; // виртуальная ширина viewBox
const VBH = 220;

function SilenceOverlayImpl({
  blocks,
  durationS,
  view,
  onSelect,
  className,
}: SilenceOverlayProps) {
  const dur = Math.max(durationS, 0.001);
  const vStart = Math.max(0, Math.min(view.start, dur));
  const vEnd = Math.max(vStart + 0.0001, Math.min(view.end, dur));
  const span = vEnd - vStart;

  const timeToX = (t: number) => ((t - vStart) / span) * VBW;

  // Перекидываем колесо/трекпад-жест с оверлея на канвас-svg под ним: находим
  // соседний [data-timeline-surface] в том же контейнере и ре-диспатчим wheel.
  // Иначе зум/пан не работают над оранжевыми блоками (они pointer-events-auto).
  const svgRef = useRef<SVGSVGElement | null>(null);
  useEffect(() => {
    const el = svgRef.current;
    if (!el) return;
    const onWheel = (e: WheelEvent) => {
      const host = el.parentElement;
      const surface = host?.querySelector<HTMLElement>("[data-timeline-surface]");
      if (!surface) return;
      e.preventDefault();
      // Клонируем событие 1:1 (deltaX/Y, ctrl=pinch, shift=pan, координаты) —
      // слушатель канваса читает ровно эти поля.
      surface.dispatchEvent(
        new WheelEvent("wheel", {
          deltaX: e.deltaX,
          deltaY: e.deltaY,
          deltaMode: e.deltaMode,
          clientX: e.clientX,
          clientY: e.clientY,
          ctrlKey: e.ctrlKey,
          shiftKey: e.shiftKey,
          bubbles: false,
          cancelable: true,
        })
      );
    };
    el.addEventListener("wheel", onWheel, { passive: false });
    return () => el.removeEventListener("wheel", onWheel);
  }, []);

  return (
    <svg
      ref={svgRef}
      viewBox={`0 0 ${VBW} ${VBH}`}
      className={cn("absolute inset-0 block h-full w-full pointer-events-none", className)}
      preserveAspectRatio="none"
    >
      <g className="pointer-events-auto">
        {blocks.map((b) => {
          const x1 = timeToX(b.start);
          const x2 = timeToX(b.end);
          const w = Math.max(x2 - x1, 0.5);
          // Видимость: блок пересекается с окном.
          if (x2 < 0 || x1 > VBW) return null;

          const fillColor = b.active
            ? "rgba(251, 191, 36, 0.3)" // amber-400 — будет вырезан
            : "rgba(148, 163, 184, 0.15)"; // slate-400 приглушённый — пропускаем
          const strokeColor = b.selected
            ? "rgba(251, 191, 36, 0.9)" // amber-400 — выделен
            : b.active
            ? "rgba(251, 191, 36, 0.5)"
            : "rgba(148, 163, 184, 0.3)";
          const strokeWidth = b.selected ? 1.5 : 0.75;

          return (
            <rect
              key={b.id}
              x={x1}
              y={0}
              width={w}
              height={VBH}
              fill={fillColor}
              stroke={strokeColor}
              strokeWidth={strokeWidth}
              className="cursor-pointer transition-colors duration-100"
              onClick={(e) => {
                e.stopPropagation();
                onSelect?.(b.id);
              }}
            />
          );
        })}
      </g>
    </svg>
  );
}

export const SilenceOverlay = memo(SilenceOverlayImpl);
