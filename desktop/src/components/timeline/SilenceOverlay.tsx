import { memo, useEffect, useRef, useState } from "react";
import { cn } from "../../lib/cn";

/**
 * SilenceOverlay — цветные блоки обнаруженной тишины поверх волны таймлайна.
 *
 * Аналог MuteOverlay, но для silence-сегментов: accepted (будут вырезаны) →
 * жёлто-оранжевый полупрозрачный блок; rejected (пропускаем) → приглушённый серый.
 * Клик по блоку → выбор; выделенный подсвечивается обводкой.
 *
 * Тайминги правятся прямо на таймлайне: тянем левый/правый край (resize) или
 * тело блока целиком (move) — ровно как у матов (см. TimelineCanvas.beginDrag).
 * Драг ведём в audio-секундах (координаты блока), на отпускании отдаём границы
 * в onResize; клик без сдвига → onSelect.
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
  /** Правка границ драгом (audio-секунды). Без него блоки только кликабельны. */
  onResize?: (id: string, start: number, end: number) => void;
  className?: string;
}

const VBW = 1000; // виртуальная ширина viewBox
const VBH = 220;
const MIN_DUR = 0.05; // минимальная длительность при драге (audio-сек)
const HANDLE_HIT = 14; // ширина зоны захвата края (в координатах viewBox)
const HANDLE_VIS = 3; // видимая полоска-ручка
const CLICK_SLOP = 0.02; // < этого сдвига (audio-сек) считаем кликом, не драгом

type DragMode = "start" | "end" | "body";
interface DragState {
  id: string;
  mode: DragMode;
  grabT: number;
  origStart: number;
  origEnd: number;
  moved: boolean;
}

function SilenceOverlayImpl({
  blocks,
  durationS,
  view,
  onSelect,
  onResize,
  className,
}: SilenceOverlayProps) {
  const dur = Math.max(durationS, 0.001);
  const vStart = Math.max(0, Math.min(view.start, dur));
  const vEnd = Math.max(vStart + 0.0001, Math.min(view.end, dur));
  const span = vEnd - vStart;

  const timeToX = (t: number) => ((t - vStart) / span) * VBW;

  // --- драг границ (audio-сек) ----------------------------------------------
  const svgRef = useRef<SVGSVGElement | null>(null);
  const dragRef = useRef<DragState | null>(null);
  const [preview, setPreview] = useState<{ id: string; start: number; end: number } | null>(null);

  // Пиксель события → время (audio-сек) через ширину svg на экране.
  const eventTime = (clientX: number): number => {
    const el = svgRef.current;
    if (!el) return vStart;
    const rect = el.getBoundingClientRect();
    const frac = rect.width > 0 ? (clientX - rect.left) / rect.width : 0;
    return vStart + frac * span;
  };

  const beginDrag = (b: SilenceBlock, mode: DragMode, ev: React.PointerEvent) => {
    if (!onResize) return;
    ev.preventDefault();
    ev.stopPropagation();
    try {
      svgRef.current?.setPointerCapture(ev.pointerId);
    } catch {
      /* капчер мог не встать — ок */
    }
    dragRef.current = {
      id: b.id,
      mode,
      grabT: eventTime(ev.clientX),
      origStart: b.start,
      origEnd: b.end,
      moved: false,
    };
    setPreview({ id: b.id, start: b.start, end: b.end });
  };

  const onPointerMove = (ev: React.PointerEvent) => {
    const d = dragRef.current;
    if (!d) return;
    const t = eventTime(ev.clientX);
    if (Math.abs(t - d.grabT) > CLICK_SLOP) d.moved = true;
    let ns = d.origStart;
    let ne = d.origEnd;
    if (d.mode === "start") {
      ns = Math.max(0, Math.min(t, d.origEnd - MIN_DUR));
    } else if (d.mode === "end") {
      ne = Math.min(dur, Math.max(t, d.origStart + MIN_DUR));
    } else {
      const delta = t - d.grabT;
      ns = Math.max(0, d.origStart + delta);
      ne = ns + (d.origEnd - d.origStart);
    }
    setPreview({ id: d.id, start: ns, end: ne });
  };

  const endDrag = (ev: React.PointerEvent) => {
    const d = dragRef.current;
    if (!d) return;
    dragRef.current = null;
    try {
      svgRef.current?.releasePointerCapture(ev.pointerId);
    } catch {
      /* ок */
    }
    const p = preview;
    setPreview(null);
    if (!d.moved) {
      onSelect?.(d.id);
      return;
    }
    if (p && onResize) onResize(d.id, p.start, p.end);
  };

  // Перекидываем колесо/трекпад-жест с оверлея на канвас-svg под ним: находим
  // соседний [data-timeline-surface] в том же контейнере и ре-диспатчим wheel.
  // Иначе зум/пан не работают над оранжевыми блоками (они pointer-events-auto).
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
      onPointerMove={onPointerMove}
      onPointerUp={endDrag}
      onPointerCancel={endDrag}
    >
      <g className="pointer-events-auto">
        {blocks.map((b) => {
          const pv = preview && preview.id === b.id ? preview : null;
          const s = pv ? pv.start : b.start;
          const e = pv ? pv.end : b.end;
          const x1 = timeToX(s);
          const x2 = timeToX(e);
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
          const handleColor = b.active
            ? "rgba(251, 191, 36, 0.9)"
            : "rgba(148, 163, 184, 0.6)";

          return (
            <g key={b.id}>
              {/* тело — тянуть целиком / клик выбрать */}
              <rect
                x={x1}
                y={0}
                width={w}
                height={VBH}
                fill={fillColor}
                stroke={strokeColor}
                strokeWidth={strokeWidth}
                className="transition-colors duration-100"
                style={{ cursor: onResize ? "grab" : "pointer" }}
                onPointerDown={(ev) => beginDrag(b, "body", ev)}
                onClick={(ev) => {
                  // Драг обрабатывается в endDrag; голый клик (без onResize) —
                  // старое поведение выбора.
                  if (onResize) return;
                  ev.stopPropagation();
                  onSelect?.(b.id);
                }}
              />
              {onResize && (
                <>
                  {/* левый край */}
                  <g
                    style={{ cursor: "ew-resize" }}
                    onPointerDown={(ev) => beginDrag(b, "start", ev)}
                  >
                    <rect x={x1 - HANDLE_HIT / 2} y={0} width={HANDLE_HIT} height={VBH} fill="transparent" />
                    <rect
                      x={x1 - HANDLE_VIS / 2}
                      y={0}
                      width={HANDLE_VIS}
                      height={VBH}
                      fill={handleColor}
                      opacity={b.selected ? 1 : 0.7}
                    />
                  </g>
                  {/* правый край */}
                  <g
                    style={{ cursor: "ew-resize" }}
                    onPointerDown={(ev) => beginDrag(b, "end", ev)}
                  >
                    <rect x={x2 - HANDLE_HIT / 2} y={0} width={HANDLE_HIT} height={VBH} fill="transparent" />
                    <rect
                      x={x2 - HANDLE_VIS / 2}
                      y={0}
                      width={HANDLE_VIS}
                      height={VBH}
                      fill={handleColor}
                      opacity={b.selected ? 1 : 0.7}
                    />
                  </g>
                </>
              )}
            </g>
          );
        })}
      </g>
    </svg>
  );
}

export const SilenceOverlay = memo(SilenceOverlayImpl);
