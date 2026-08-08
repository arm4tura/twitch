import { memo, useCallback, useEffect, useLayoutEffect, useMemo, useRef } from "react";
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
  /** silence=красный (заглушим) | keep=серый (оставим) | review=янтарный (не проверено). */
  state: 'silence' | 'keep' | 'review';
  selected: boolean;
}

export interface OverviewStripProps {
  durationS: number;
  peaks: number[];
  markers: OverviewMarker[];
  currentS?: number;
  currentRef?: React.MutableRefObject<number>;
  onSeek: (t: number) => void;
  onPickMarker?: (id: string) => void;
  /** Текущее окно зума главного таймлайна — рисуем поверх как рамку-навигатор. */
  view?: { start: number; end: number };
  className?: string;
}

const VBW = 1000; // виртуальная ширина viewBox — SVG растянется по контейнеру
const VBH = 56;

function OverviewStripImpl({
  durationS,
  peaks,
  markers,
  currentS,
  currentRef,
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

  // Волну (500 столбиков) рисуем на <canvas>: как SVG-rect она реконсилилась на
  // каждый зум-кадр рамки и на каждую правку мьюта. Canvas статичен — перерисов
  // только при смене peaks/размера. Маркеры/рамка/плейхед остаются SVG.
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const boxRef = useRef<HTMLDivElement | null>(null);
  const drawWave = useCallback(() => {
    const canvas = canvasRef.current;
    const box = boxRef.current;
    if (!canvas || !box) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const r = box.getBoundingClientRect();
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const w = Math.max(1, Math.round(r.width));
    const h = Math.max(1, Math.round(r.height));
    const pxW = Math.round(w * dpr);
    const pxH = Math.round(h * dpr);
    if (canvas.width !== pxW || canvas.height !== pxH) {
      canvas.width = pxW;
      canvas.height = pxH;
    }
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);
    if (!bars.length) return;
    const bw = w / bars.length;
    const barVis = Math.max(bw - 0.4 * (w / VBW), 0.4);
    ctx.fillStyle = "rgba(148,163,184,0.4)";
    ctx.beginPath();
    for (let i = 0; i < bars.length; i++) {
      const bh = (bars[i] / maxPeak) * (h - 8 * (h / VBH));
      ctx.rect(i * bw, (h - bh) / 2, barVis, bh);
    }
    ctx.fill();
  }, [bars, maxPeak]);

  useLayoutEffect(() => {
    const box = boxRef.current;
    if (!box) return;
    drawWave();
    const ro = new ResizeObserver(drawWave);
    ro.observe(box);
    return () => ro.disconnect();
  }, [drawWave]);

  // Самоанимирующийся плейхед: если родитель дал currentRef, двигаем линию
  // прямо в DOM через rAF — мини-карта (500 rect) не перерисовывается на каждом
  // кадре воспроизведения. Без ref — статичная линия по currentS (fallback).
  const playheadRef = useRef<SVGLineElement | null>(null);
  useEffect(() => {
    if (!currentRef) return;
    let raf = 0;
    let lastX = NaN;
    const tick = () => {
      const line = playheadRef.current;
      if (line) {
        const x = (currentRef.current / dur) * VBW;
        if (x !== lastX) {
          line.setAttribute("x1", String(x));
          line.setAttribute("x2", String(x));
          lastX = x;
        }
      }
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [currentRef, dur]);

  const handleClick = (e: React.MouseEvent<SVGSVGElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const frac = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
    onSeek(frac * dur);
  };

  return (
    <div
      ref={boxRef}
      className={cn(
        "relative block h-14 w-full overflow-hidden rounded-lg border border-white/8 bg-black/30",
        className
      )}
    >
      <canvas ref={canvasRef} className="pointer-events-none absolute inset-0 h-full w-full" />
      <svg
        className="absolute inset-0 block h-full w-full cursor-pointer"
        viewBox={`0 0 ${VBW} ${VBH}`}
        preserveAspectRatio="none"
        onClick={handleClick}
      >
        {/* Маркеры мьютов */}
        {markers.map((m) => {
          const x = timeToX(m.start);
          const w = Math.max(timeToX(m.end) - x, 1.5);
          const color =
            m.state === "silence"
              ? "rgb(244 63 94)"
              : m.state === "review"
              ? "rgb(251 191 36)"
              : "rgb(148 163 184)";
          const opacity = m.selected
            ? 0.55
            : m.state === "silence"
            ? 0.4
            : m.state === "review"
            ? 0.45
            : 0.25;
          return (
            <rect
              key={m.id}
              x={x}
              y={m.selected ? 0 : 4}
              width={w}
              height={m.selected ? VBH : VBH - 8}
              fill={color}
              opacity={opacity}
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

        {/* Playhead — при наличии currentRef координаты пишет rAF (без ре-рендера) */}
        <line
          ref={playheadRef}
          x1={timeToX(currentS ?? 0)}
          y1={0}
          x2={timeToX(currentS ?? 0)}
          y2={VBH}
          stroke="rgb(244 244 245)"
          strokeWidth={1.2}
          pointerEvents="none"
        />
      </svg>
    </div>
  );
}

/**
 * memo: мини-карта зависит только от peaks/markers/view. Плейхед анимируется
 * сам через currentRef, поэтому throttle-обновления currentS у родителя её не
 * трогают. Отсекает основную массу ре-рендеров во время воспроизведения.
 */
export const OverviewStrip = memo(OverviewStripImpl);
