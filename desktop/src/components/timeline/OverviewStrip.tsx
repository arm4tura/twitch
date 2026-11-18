import { memo, useCallback, useEffect, useLayoutEffect, useRef } from "react";
import { cn } from "../../lib/cn";
import { drawWaveTile, SHOW_RMS_CORE, type WaveTile } from "./waveTile";

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
  /**
   * ВСЕГДА базовый тайл (весь стрим). Мини-карта не зумится, поэтому крошку
   * видимого окна ей подгружать незачем — она показывает целое.
   */
  tile: WaveTile;
  /** Глобальная нормировка амплитуды — та же, что у главного канваса. */
  norm: number;
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
  tile,
  norm,
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

  // Волну рисуем на <canvas>: как SVG-rect она реконсилилась на каждый зум-кадр
  // рамки и на каждую правку мьюта. Canvas статичен — перерисовка только при
  // смене тайла/размера. Маркеры/рамка/плейхед остаются SVG.
  // Даунсэмпл не нужен: drawWaveTile сам агрегирует бакеты в столбик на пиксель.
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
    drawWaveTile(ctx, tile, {
      w,
      h,
      viewStart: 0,
      viewEnd: dur, // мини-карта всегда показывает целое
      norm,
      padPx: 4 * (h / VBH),
      // Приглушённее главного канваса — обзорная полоса не должна перетягивать
      // внимание. Ядро RMS, как и на главном канвасе, отключено.
      contourColor: "rgba(139,153,173,0.55)",
      coreColor: SHOW_RMS_CORE ? "rgba(181,195,214,0.6)" : undefined,
      minHalfPx: 0.4,
    });
  }, [tile, norm, dur]);

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
        "relative block h-14 w-full overflow-hidden rounded-lg border border-white/[0.08] bg-black/30",
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
