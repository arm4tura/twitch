import { useMemo } from "react";
import { cn } from "../../lib/cn";
import type { WFRegion } from "./Waveform";

/**
 * Minimap — сжатый обзор всей длительности с маркерами регионов и viewport'ом.
 *
 * Реализация без wavesurfer minimap-плагина — тот перерисовывает peaks
 * заново каждые ~секунду при zoom > 200 и сжирает CPU. Здесь мы просто рисуем
 * SVG:
 *   - фоновая линия (лёгкая волна из peaks через canvas — необязательна),
 *   - цветные тонкие полоски для каждого региона (max 500, дальше визуально
 *     сливается),
 *   - прямоугольник viewport'а (scroll + visible width).
 *
 * Клик — прыжок в этот момент.
 */

export interface MinimapProps {
  durationS: number;
  regions: WFRegion[];
  currentS: number;
  /** Секунда, с которой начинается видимая область в основной волне. */
  viewportStartS: number;
  /** Ширина видимой области в секундах. */
  viewportWidthS: number;
  onSeek: (t: number) => void;
  className?: string;
}

const KIND_STROKE = {
  mute: "rgb(244 63 94)",
  cut: "rgb(245 158 11)",
  highlight: "rgb(139 92 246)",
};

export function Minimap({
  durationS,
  regions,
  currentS,
  viewportStartS,
  viewportWidthS,
  onSeek,
  className,
}: MinimapProps) {
  const height = 40;
  const width = 1000; // виртуальный viewport — реальная ширина растянет SVG

  // Дедуп бэрров: если >500 регионов, оставляем каждый n-ный. UX всё равно
  // не различает более плотное расположение при таком масштабе.
  const displayRegions = useMemo(() => {
    if (regions.length <= 500) return regions;
    const step = Math.ceil(regions.length / 500);
    return regions.filter((_, i) => i % step === 0);
  }, [regions]);

  const timeToX = (t: number) => (t / Math.max(durationS, 0.001)) * width;

  const handleClick = (e: React.MouseEvent<SVGSVGElement>) => {
    const rect = (e.currentTarget as SVGSVGElement).getBoundingClientRect();
    const x = e.clientX - rect.left;
    const frac = Math.max(0, Math.min(1, x / rect.width));
    onSeek(frac * durationS);
  };

  return (
    <svg
      className={cn(
        "block h-10 w-full cursor-pointer rounded-md border border-white/8 bg-black/30",
        className
      )}
      viewBox={`0 0 ${width} ${height}`}
      preserveAspectRatio="none"
      onClick={handleClick}
    >
      {/* Base line */}
      <line
        x1={0}
        y1={height / 2}
        x2={width}
        y2={height / 2}
        stroke="rgb(255 255 255 / 0.06)"
        strokeWidth={1}
      />

      {/* Regions */}
      {displayRegions.map((r) => {
        const x1 = timeToX(r.start);
        const x2 = Math.max(timeToX(r.end), x1 + 1);
        return (
          <line
            key={r.id}
            x1={x1}
            y1={height / 2 - 8}
            x2={x1}
            y2={height / 2 + 8}
            stroke={KIND_STROKE[r.kind]}
            strokeWidth={Math.max(1, x2 - x1)}
            opacity={0.7}
          />
        );
      })}

      {/* Viewport rectangle */}
      <rect
        x={timeToX(viewportStartS)}
        y={2}
        width={Math.max(4, timeToX(viewportWidthS))}
        height={height - 4}
        fill="rgb(139 92 246 / 0.08)"
        stroke="rgb(139 92 246 / 0.7)"
        strokeWidth={1}
        rx={2}
      />

      {/* Playhead */}
      <line
        x1={timeToX(currentS)}
        y1={0}
        x2={timeToX(currentS)}
        y2={height}
        stroke="rgb(244 244 245)"
        strokeWidth={1.2}
      />
    </svg>
  );
}
