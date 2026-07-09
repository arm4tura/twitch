import { useMemo } from "react";
import { cn } from "../../lib/cn";

/**
 * OverviewStrip — обзорная полоса всей записи (архетип «Overview» из
 * Audition/Reaper). Не зумится, всегда показывает целое: где по таймлайну
 * стоят заглушки, где сейчас playhead, куда кликнуть чтобы прыгнуть.
 *
 * Рисуем сами (SVG), без wavesurfer-minimap: тот пересчитывает peaks при
 * зуме и жрёт CPU. Пик-бар из готового массива + тонкие полоски-маркеры
 * мьютов, окрашенные по статусу (accepted=красный, rejected=серый).
 *
 * Клик по любому месту — seek. Клик по маркеру мьюта — выбрать его.
 */

export interface OverviewMarker {
  id: string;
  start: number;
  end: number;
  /** accepted → активный красный; иначе приглушённый (оставлен). */
  active: boolean;
  selected: boolean;
}

export interface OverviewStripProps {
  durationS: number;
  peaks: number[];
  markers: OverviewMarker[];
  currentS: number;
  onSeek: (t: number) => void;
  onPickMarker?: (id: string) => void;
  /** Текущее окно зума главного таймлайна — рисуем поверх как рамку-навигатор. */
  view?: { start: number; end: number };
  className?: string;
}

const VBW = 1000; // виртуальная ширина viewBox — SVG растянется по контейнеру
const VBH = 56;

export function OverviewStrip({
  durationS,
  peaks,
  markers,
  currentS,
  onSeek,
  onPickMarker,
  view,
  className,
}: OverviewStripProps) {
  const dur = Math.max(durationS, 0.001);
  const timeToX = (t: number) => (t / dur) * VBW;

  // Даунсэмпл peaks до ~500 столбиков — плотнее глаз не различает на этой ширине.
  const bars = useMemo(() => {
    if (!peaks.length) return [];
    const N = Math.min(500, peaks.length);
    const step = peaks.length / N;
    const out: number[] = [];
    for (let i = 0; i < N; i++) {
      // max по окну — сохраняем пики, а не среднее (иначе волна «проваливается»).
      let m = 0;
      const from = Math.floor(i * step);
      const to = Math.floor((i + 1) * step);
      for (let j = from; j < to && j < peaks.length; j++) {
        const v = Math.abs(peaks[j]);
        if (v > m) m = v;
      }
      out.push(m);
    }
    return out;
  }, [peaks]);

  const maxPeak = useMemo(() => bars.reduce((a, b) => Math.max(a, b), 0.0001), [bars]);
  const barW = VBW / (bars.length || 1);

  const handleClick = (e: React.MouseEvent<SVGSVGElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const frac = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
    onSeek(frac * dur);
  };

  return (
    <svg
      className={cn(
        "block h-14 w-full cursor-pointer rounded-lg border border-white/8 bg-black/30",
        className
      )}
      viewBox={`0 0 ${VBW} ${VBH}`}
      preserveAspectRatio="none"
      onClick={handleClick}
    >
      {/* Волна */}
      {bars.map((v, i) => {
        const h = (v / maxPeak) * (VBH - 8);
        return (
          <rect
            key={i}
            x={i * barW}
            y={(VBH - h) / 2}
            width={Math.max(barW - 0.4, 0.4)}
            height={h}
            fill="rgb(148 163 184 / 0.4)"
          />
        );
      })}

      {/* Маркеры мьютов */}
      {markers.map((m) => {
        const x = timeToX(m.start);
        const w = Math.max(timeToX(m.end) - x, 1.5);
        const color = m.active ? "rgb(244 63 94)" : "rgb(148 163 184)";
        return (
          <rect
            key={m.id}
            x={x}
            y={m.selected ? 0 : 4}
            width={w}
            height={m.selected ? VBH : VBH - 8}
            fill={color}
            opacity={m.selected ? 0.55 : m.active ? 0.4 : 0.25}
            onClick={(e) => {
              e.stopPropagation();
              onPickMarker?.(m.id);
            }}
            style={{ cursor: onPickMarker ? "pointer" : "inherit" }}
          />
        );
      })}

      {/* Рамка текущего окна зума главного таймлайна (навигатор) */}
      {view && view.end - view.start < dur - 0.001 && (
        <g pointerEvents="none">
          <rect
            x={timeToX(Math.max(0, view.start))}
            y={0}
            width={Math.max(timeToX(Math.min(dur, view.end)) - timeToX(Math.max(0, view.start)), 2)}
            height={VBH}
            fill="rgb(255 255 255 / 0.08)"
            stroke="rgb(255 255 255 / 0.5)"
            strokeWidth={1}
          />
        </g>
      )}

      {/* Playhead */}
      <line
        x1={timeToX(currentS)}
        y1={0}
        x2={timeToX(currentS)}
        y2={VBH}
        stroke="rgb(244 244 245)"
        strokeWidth={1.2}
      />
    </svg>
  );
}
