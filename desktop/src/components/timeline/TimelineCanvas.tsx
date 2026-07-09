import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { cn } from "../../lib/cn";

/**
 * TimelineCanvas — главный DAW-таймлайн всего стрима (архетип Vegas/Reaper).
 *
 * Один зумируемый холст со сплошной волной; заглушки лежат на нём цветными
 * блоками. Колесо — зум вокруг курсора, shift+колесо — пан. Клик по пустой
 * волне — перемотка, клик по блоку — выбор, драг края/тела блока — правка
 * границ БЕЗ снапа (волна = покадровая точность; снап к словам живёт только в
 * транскрипте инспектора).
 *
 * ВСЕ времена в пропсах — АУДИО-секунды (как peaks и durationS). Родитель сам
 * конвертирует в/из локального пространства через toAudio/toLocal.
 *
 * Производительность: волну рисуем ОДНИМ <path> (набор залитых столбиков), а не
 * тысячей <rect> — иначе React дёргается на каждом тике зума. Видимое окно
 * ресэмплим до ~1200 столбиков. Изменения view коалесцируем через rAF, чтобы
 * серия wheel-событий за кадр давала один setState у родителя.
 */

export interface TimelineBlock {
  id: string;
  start: number; // audio-сек
  end: number; // audio-сек
  active: boolean; // accepted → красный; иначе приглушённый (оставлен)
  selected: boolean;
}

export interface TimelineView {
  start: number;
  end: number;
}

export interface TimelineCanvasProps {
  peaks: number[];
  durationS: number;
  blocks: TimelineBlock[];
  view: TimelineView;
  currentS: number; // playhead, audio-сек
  onViewChange: (view: TimelineView) => void;
  onSeek: (audioT: number) => void;
  onSelect: (id: string) => void;
  /** Коммит новых границ блока (на отпускании драга), audio-сек, БЕЗ снапа. */
  onResize: (id: string, aStart: number, aEnd: number) => void;
  /** Короткий блип «слышно край» по ходу драга, audio-сек. */
  onScrub?: (audioT: number) => void;
  className?: string;
}

const VBW = 1000; // виртуальная ширина viewBox — SVG растянется по контейнеру
const VBH = 220;
const TARGET_BARS = 1200; // столбиков волны в видимом окне
const MIN_SPAN = 0.4; // минимальная ширина окна зума, сек
const MIN_DUR = 0.03; // минимальная длина блока, сек
const HANDLE_HIT = 12; // ширина зоны захвата края, vb-единицы
const HANDLE_VIS = 2; // видимая толщина края, vb-единицы
const CLICK_SLOP = 0.01; // сдвиг меньше (сек) — трактуем как клик, а не драг
const SCRUB_MS = 140; // порог между скраб-блипами по реальному времени, мс
const ZOOM_STEP = 1.2; // множитель зума на «щелчок» колеса

type DragMode = "start" | "end" | "body";
interface DragState {
  id: string;
  mode: DragMode;
  grabT: number;
  origStart: number;
  origEnd: number;
  moved: boolean;
}

export function TimelineCanvas({
  peaks,
  durationS,
  blocks,
  view,
  currentS,
  onViewChange,
  onSeek,
  onSelect,
  onResize,
  onScrub,
  className,
}: TimelineCanvasProps) {
  const svgRef = useRef<SVGSVGElement | null>(null);
  const dragRef = useRef<DragState | null>(null);
  const lastScrubRef = useRef(0);
  const [preview, setPreview] = useState<{ id: string; start: number; end: number } | null>(null);

  const dur = Math.max(durationS, 0.001);
  const vStart = Math.max(0, Math.min(view.start, dur));
  const vEnd = Math.max(vStart + MIN_SPAN * 0.001, Math.min(view.end, dur));
  const span = vEnd - vStart;

  const timeToX = useCallback((t: number) => ((t - vStart) / span) * VBW, [vStart, span]);

  // px клиента → audio-время в координатах окна.
  const eventTime = useCallback(
    (clientX: number) => {
      const rect = svgRef.current?.getBoundingClientRect();
      if (!rect || rect.width === 0) return vStart;
      const frac = Math.max(0, Math.min(1, (clientX - rect.left) / rect.width));
      return vStart + frac * span;
    },
    [vStart, span]
  );

  // Глобальный максимум пиков — нормируем высоту по нему, чтобы амплитуда была
  // одинаковой на любом зуме (а не «раздувалась» под окно).
  const maxPeak = useMemo(() => {
    let m = 0.0001;
    for (const p of peaks) {
      const v = Math.abs(p);
      if (v > m) m = v;
    }
    return m;
  }, [peaks]);

  // Волна видимого окна как один залитый <path>. Ресэмпл до TARGET_BARS:
  // max-в-bucket на плотных участках, линейная интерполяция на разреженных.
  const wavePath = useMemo(() => {
    if (!peaks.length) return "";
    const i0 = Math.max(0, Math.floor((vStart / dur) * peaks.length));
    const i1 = Math.min(peaks.length, Math.ceil((vEnd / dur) * peaks.length));
    const slice: number[] = [];
    for (let i = i0; i < i1; i++) slice.push(Math.abs(peaks[i]));
    if (!slice.length) return "";
    const cy = VBH / 2;
    const barW = VBW / TARGET_BARS;
    const half = barW * 0.4; // половина видимой ширины столбика
    let d = "";
    for (let i = 0; i < TARGET_BARS; i++) {
      const p0 = (i / TARGET_BARS) * slice.length;
      const p1 = ((i + 1) / TARGET_BARS) * slice.length;
      const a = Math.floor(p0);
      const b = Math.ceil(p1);
      let v: number;
      if (b - a <= 1) {
        const f = p0 - a;
        const v0 = slice[Math.min(a, slice.length - 1)] ?? 0;
        const v1 = slice[Math.min(a + 1, slice.length - 1)] ?? v0;
        v = v0 + (v1 - v0) * f;
      } else {
        let m = 0;
        for (let j = a; j < b && j < slice.length; j++) if (slice[j] > m) m = slice[j];
        v = m;
      }
      const h = Math.max((v / maxPeak) * (VBH - 24), 1.2);
      const cx = (i + 0.5) * barW;
      const x0 = cx - half;
      const x1 = cx + half;
      const y0 = cy - h / 2;
      const y1 = cy + h / 2;
      d += `M${x0.toFixed(2)} ${y0.toFixed(2)}L${x1.toFixed(2)} ${y0.toFixed(2)}L${x1.toFixed(2)} ${y1.toFixed(2)}L${x0.toFixed(2)} ${y1.toFixed(2)}Z`;
    }
    return d;
  }, [peaks, vStart, vEnd, dur, maxPeak]);

  // --- зум/пан колесом (native listener: React onWheel пассивный, не даёт
  // preventDefault). Читаем актуальные view/handler через refs. ---
  const viewRef = useRef(view);
  viewRef.current = view;
  const durRef = useRef(dur);
  durRef.current = dur;
  const onViewChangeRef = useRef(onViewChange);
  onViewChangeRef.current = onViewChange;
  // Коалесценция: pending хранит последнее посчитанное окно до коммита в rAF.
  const pendingRef = useRef<TimelineView | null>(null);
  const rafRef = useRef<number | null>(null);

  // Сброс pending, когда родитель принял новое view (или его сменили кнопками).
  useEffect(() => {
    pendingRef.current = null;
  }, [view.start, view.end]);

  const commitView = useCallback((v: TimelineView) => {
    pendingRef.current = v;
    if (rafRef.current == null) {
      rafRef.current = requestAnimationFrame(() => {
        rafRef.current = null;
        if (pendingRef.current) onViewChangeRef.current(pendingRef.current);
      });
    }
  }, []);

  useEffect(() => {
    const el = svgRef.current;
    if (!el) return;
    const clampView = (s: number, e: number): TimelineView => {
      const d = durRef.current;
      let width = Math.max(MIN_SPAN, Math.min(e - s, d));
      let ns = s;
      let ne = s + width;
      if (ns < 0) {
        ns = 0;
        ne = width;
      }
      if (ne > d) {
        ne = d;
        ns = Math.max(0, d - width);
      }
      return { start: ns, end: ne };
    };
    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      const cur = pendingRef.current ?? viewRef.current;
      const cs = Math.max(0, Math.min(cur.start, durRef.current));
      const ce = Math.max(cs + 0.001, Math.min(cur.end, durRef.current));
      const curSpan = ce - cs;
      const rect = el.getBoundingClientRect();
      const isPan = e.shiftKey || Math.abs(e.deltaX) > Math.abs(e.deltaY);
      if (isPan) {
        const raw = Math.abs(e.deltaX) > Math.abs(e.deltaY) ? e.deltaX : e.deltaY;
        const delta = (raw / (rect.width || 1)) * curSpan;
        commitView(clampView(cs + delta, ce + delta));
      } else {
        const frac = rect.width ? Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width)) : 0.5;
        const cursorT = cs + frac * curSpan;
        const factor = e.deltaY > 0 ? ZOOM_STEP : 1 / ZOOM_STEP;
        const newSpan = Math.max(MIN_SPAN, Math.min(curSpan * factor, durRef.current));
        const ns = cursorT - frac * newSpan;
        commitView(clampView(ns, ns + newSpan));
      }
    };
    el.addEventListener("wheel", onWheel, { passive: false });
    return () => el.removeEventListener("wheel", onWheel);
  }, [commitView]);

  useEffect(
    () => () => {
      if (rafRef.current != null) cancelAnimationFrame(rafRef.current);
    },
    []
  );

  // --- драг блоков ----------------------------------------------------------
  const maybeScrub = (t: number) => {
    if (!onScrub) return;
    const now = performance.now();
    if (now - lastScrubRef.current >= SCRUB_MS) {
      lastScrubRef.current = now;
      onScrub(t);
    }
  };

  const beginDrag = (b: TimelineBlock, mode: DragMode, ev: React.PointerEvent) => {
    ev.preventDefault();
    ev.stopPropagation();
    svgRef.current?.setPointerCapture(ev.pointerId);
    const grabT = eventTime(ev.clientX);
    dragRef.current = {
      id: b.id,
      mode,
      grabT,
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
      maybeScrub(ns);
    } else if (d.mode === "end") {
      ne = Math.max(t, d.origStart + MIN_DUR);
      maybeScrub(ne);
    } else {
      const delta = t - d.grabT;
      ns = Math.max(0, d.origStart + delta);
      ne = ns + (d.origEnd - d.origStart);
      maybeScrub(ns);
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
      /* капчер мог не встать — ок */
    }
    const p = preview;
    setPreview(null);
    if (!d.moved) {
      // Клик без сдвига — выбрать блок (родитель ещё и проиграет мат).
      onSelect(d.id);
      return;
    }
    if (p) onResize(d.id, p.start, p.end);
  };

  // Клик по пустой волне — перемотка.
  const onBackgroundClick = (ev: React.MouseEvent) => {
    if (dragRef.current) return;
    onSeek(eventTime(ev.clientX));
  };

  const cy = VBH / 2;
  const playheadX = timeToX(currentS);
  const playheadVisible = currentS >= vStart && currentS <= vEnd;

  return (
    <svg
      ref={svgRef}
      className={cn(
        "block h-full w-full touch-none select-none rounded-lg border border-white/8 bg-black/30",
        className
      )}
      viewBox={`0 0 ${VBW} ${VBH}`}
      preserveAspectRatio="none"
      onPointerMove={onPointerMove}
      onPointerUp={endDrag}
      onPointerCancel={endDrag}
    >
      {/* Фон для клика-перемотки (позади всего) */}
      <rect
        x={0}
        y={0}
        width={VBW}
        height={VBH}
        fill="transparent"
        style={{ cursor: "text" }}
        onClick={onBackgroundClick}
      />

      {/* Центральная ось */}
      <line x1={0} y1={cy} x2={VBW} y2={cy} stroke="rgb(255 255 255 / 0.06)" strokeWidth={1} />

      {/* Волна одним path */}
      {wavePath && <path d={wavePath} fill="rgb(148 163 184 / 0.5)" pointerEvents="none" />}

      {/* Блоки заглушек */}
      {blocks.map((b) => {
        const pv = preview && preview.id === b.id ? preview : null;
        const s = pv ? pv.start : b.start;
        const e = pv ? pv.end : b.end;
        if (e < vStart || s > vEnd) return null; // вне окна
        const xs = timeToX(s);
        const xe = timeToX(e);
        const w = Math.max(xe - xs, 1);
        const color = b.active ? "rgb(244 63 94)" : "rgb(148 163 184)";
        const bodyOpacity = b.selected ? 0.28 : b.active ? 0.16 : 0.1;
        return (
          <g key={b.id}>
            {/* тело — тянуть целиком / клик выбрать */}
            <rect
              x={xs}
              y={0}
              width={w}
              height={VBH}
              fill={color}
              opacity={bodyOpacity}
              stroke={color}
              strokeOpacity={b.selected ? 0.9 : 0.5}
              strokeWidth={b.selected ? 1.5 : 0.75}
              style={{ cursor: "grab" }}
              onPointerDown={(ev) => beginDrag(b, "body", ev)}
            />
            {/* левый край */}
            <g style={{ cursor: "ew-resize" }} onPointerDown={(ev) => beginDrag(b, "start", ev)}>
              <rect x={xs - HANDLE_HIT / 2} y={0} width={HANDLE_HIT} height={VBH} fill="transparent" />
              <rect x={xs - HANDLE_VIS / 2} y={0} width={HANDLE_VIS} height={VBH} fill={color} opacity={b.selected ? 1 : 0.7} />
            </g>
            {/* правый край */}
            <g style={{ cursor: "ew-resize" }} onPointerDown={(ev) => beginDrag(b, "end", ev)}>
              <rect x={xe - HANDLE_HIT / 2} y={0} width={HANDLE_HIT} height={VBH} fill="transparent" />
              <rect x={xe - HANDLE_VIS / 2} y={0} width={HANDLE_VIS} height={VBH} fill={color} opacity={b.selected ? 1 : 0.7} />
            </g>
          </g>
        );
      })}

      {/* Playhead */}
      {playheadVisible && (
        <line
          x1={playheadX}
          y1={0}
          x2={playheadX}
          y2={VBH}
          stroke="rgb(244 244 245)"
          strokeWidth={1.2}
          pointerEvents="none"
        />
      )}
    </svg>
  );
}
