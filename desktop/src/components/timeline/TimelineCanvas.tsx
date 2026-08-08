import { memo, useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
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
  /** Состояние маркера: 'silence' (красный) | 'keep' (серый) | 'review' (янтарный пунктир) */
  state: 'silence' | 'keep' | 'review';
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
  /**
   * Живой плейхед (audio-сек) через ref — читается собственным rAF канваса и
   * пишется прямо в DOM линии, БЕЗ ре-рендера React. Так волна/блоки не
   * реконсилятся 60 раз/сек во время воспроизведения (главный источник лагов).
   */
  currentRef: React.MutableRefObject<number>;
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

type DragMode = "start" | "end" | "body";
interface DragState {
  id: string;
  mode: DragMode;
  grabT: number;
  origStart: number;
  origEnd: number;
  moved: boolean;
}

function TimelineCanvasImpl({
  peaks,
  durationS,
  blocks,
  view,
  currentRef,
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

  // Предрасчёт: |peaks| один раз + глобальный максимум. Раньше и то, и другое
  // считалось на КАЖДОМ кадре зума внутри построения волны (Math.abs по всему
  // массиву + аллокация slice) — на длинном стриме это и есть главный лаг.
  // Теперь волна рисуется на <canvas> прямыми fillRect по этому массиву.
  const { absPeaks, maxPeak } = useMemo(() => {
    const arr = new Float32Array(peaks.length);
    let m = 0.0001;
    for (let i = 0; i < peaks.length; i++) {
      const v = Math.abs(peaks[i]);
      arr[i] = v;
      if (v > m) m = v;
    }
    return { absPeaks: arr, maxPeak: m };
  }, [peaks]);

  // --- волна на <canvas> ----------------------------------------------------
  // fillRect в цикле на порядок дешевле, чем строить ~50 KB SVG <path> и заново
  // тесселировать его каждый кадр. Canvas 2D одинаков в Chromium на Mac и Windows.
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const canvasBoxRef = useRef<HTMLDivElement | null>(null);
  const sizeRef = useRef({ w: 0, h: 0, dpr: 1 });

  const drawWave = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const { w, h, dpr } = sizeRef.current;
    if (w === 0 || h === 0) return;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);
    // Центральная ось.
    ctx.fillStyle = "rgba(255,255,255,0.06)";
    ctx.fillRect(0, h / 2 - 0.5, w, 1);
    const n = absPeaks.length;
    if (!n) return;
    const ve = vStart + span;
    const cy = h / 2;
    const amp = h - 24 * (h / VBH); // тот же запас сверху/снизу, что был в SVG
    const bars = Math.max(1, Math.min(TARGET_BARS, Math.floor(w)));
    const barW = w / bars;
    const barVis = Math.max(barW * 0.8, 1);
    const off = (barW - barVis) / 2;
    const i0f = (vStart / dur) * n;
    const i1f = (ve / dur) * n;
    const sliceLen = Math.max(i1f - i0f, 1);
    // Один path из всех столбиков, одна заливка — заметно дешевле, чем fillRect
    // на каждый бар. Цвет в rgba(): canvas 2D не парсит CSS-Color-4 «/ alpha».
    ctx.beginPath();
    for (let i = 0; i < bars; i++) {
      const p0 = i0f + (i / bars) * sliceLen;
      const p1 = i0f + ((i + 1) / bars) * sliceLen;
      const a = Math.floor(p0);
      const b = Math.ceil(p1);
      let v: number;
      if (b - a <= 1) {
        const f = p0 - a;
        const v0 = absPeaks[Math.max(0, Math.min(a, n - 1))] ?? 0;
        const v1 = absPeaks[Math.max(0, Math.min(a + 1, n - 1))] ?? v0;
        v = v0 + (v1 - v0) * f;
      } else {
        let m = 0;
        for (let j = Math.max(0, a); j < b && j < n; j++) if (absPeaks[j] > m) m = absPeaks[j];
        v = m;
      }
      const bh = Math.max((v / maxPeak) * amp, 1.2);
      ctx.rect(i * barW + off, cy - bh / 2, barVis, bh);
    }
    ctx.fillStyle = "rgba(148,163,184,0.5)";
    ctx.fill();
  }, [absPeaks, maxPeak, dur, vStart, span]);

  // Ресайз canvas под контейнер (DPR-aware) + перерисовка. RO создаём один раз:
  // измерение и перерисовку зовём через ref, иначе смена drawWave на каждый
  // зум-кадр пересоздавала бы ResizeObserver.
  const measureAndDrawRef = useRef<() => void>(() => {});
  measureAndDrawRef.current = () => {
    const box = canvasBoxRef.current;
    const canvas = canvasRef.current;
    if (!box || !canvas) return;
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
    sizeRef.current = { w, h, dpr };
    drawWave();
  };

  useLayoutEffect(() => {
    const box = canvasBoxRef.current;
    if (!box) return;
    measureAndDrawRef.current();
    const ro = new ResizeObserver(() => measureAndDrawRef.current());
    ro.observe(box);
    return () => ro.disconnect();
  }, []);

  // Перерисовка при смене окна/данных (размер не трогаем — он уже верный).
  useEffect(() => {
    drawWave();
  }, [drawWave]);

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
      // Нормируем deltaY: в line-mode (обычная мышь) шаг ~3 строки, домножаем до
      // пикселей, чтобы у мыши и тачпада была общая шкала «щелчка».
      const unit = e.deltaMode === 1 ? 16 : e.deltaMode === 2 ? rect.height || 400 : 1;
      const dy = e.deltaY * unit;
      const dx = e.deltaX * unit;
      // macOS pinch-to-zoom на тачпаде приходит как wheel с ctrlKey=true.
      const isPinch = e.ctrlKey;
      const isPan = !isPinch && (e.shiftKey || Math.abs(dx) > Math.abs(dy));
      if (isPan) {
        const raw = Math.abs(dx) > Math.abs(dy) ? dx : dy;
        const delta = (raw / (rect.width || 1)) * curSpan;
        commitView(clampView(cs + delta, ce + delta));
        return;
      }
      // Зум вокруг курсора. Величина ПРОПОРЦИОНАЛЬНА жесту (экспонента от dy),
      // поэтому щипок/скролл тачпада плавные, а не прыгают фиксированным шагом.
      // Для пинча коэффициент круче (жест короче), для колеса — мягче и
      // ограничен по модулю, чтобы один щелчок мыши не проскакивал пол-стрима.
      const frac = rect.width ? Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width)) : 0.5;
      const cursorT = cs + frac * curSpan;
      const k = isPinch ? 0.01 : 0.0025;
      const exponent = Math.max(-0.7, Math.min(0.7, dy * k));
      const factor = Math.exp(exponent);
      const newSpan = Math.max(MIN_SPAN, Math.min(curSpan * factor, durRef.current));
      const ns = cursorT - frac * newSpan;
      commitView(clampView(ns, ns + newSpan));
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

  // Самоанимирующийся плейхед: собственный rAF читает currentRef и пишет x/opacity
  // прямо в DOM линии. React не участвует — волна и блоки не перерисовываются во
  // время воспроизведения. vStart/span берём из свежего view через ref.
  const playheadLineRef = useRef<SVGLineElement | null>(null);
  const geomRef = useRef({ vStart, span });
  geomRef.current = { vStart, span };
  useEffect(() => {
    let raf = 0;
    let lastX = NaN;
    let lastVis = -1;
    const tick = () => {
      const line = playheadLineRef.current;
      if (line) {
        const { vStart: gs, span: gsp } = geomRef.current;
        const t = currentRef.current;
        const x = ((t - gs) / gsp) * VBW;
        const vis = t >= gs && t <= gs + gsp ? 1 : 0;
        if (x !== lastX) {
          line.setAttribute("x1", String(x));
          line.setAttribute("x2", String(x));
          lastX = x;
        }
        if (vis !== lastVis) {
          line.style.opacity = vis ? "1" : "0";
          lastVis = vis;
        }
      }
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [currentRef]);

  return (
    <div
      ref={canvasBoxRef}
      className={cn(
        "relative block h-full w-full overflow-hidden rounded-lg border border-white/8 bg-black/30",
        className
      )}
    >
      {/* Волна на canvas — под SVG-слоем взаимодействия */}
      <canvas
        ref={canvasRef}
        className="pointer-events-none absolute inset-0 h-full w-full"
      />
      <svg
        ref={svgRef}
        data-timeline-surface
        className="absolute inset-0 block h-full w-full touch-none select-none"
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

      {/* Блоки заглушек */}
      {blocks.map((b) => {
        const pv = preview && preview.id === b.id ? preview : null;
        const s = pv ? pv.start : b.start;
        const e = pv ? pv.end : b.end;
        if (e < vStart || s > vEnd) return null; // вне окна
        const xs = timeToX(s);
        const xe = timeToX(e);
        const w = Math.max(xe - xs, 1);
        // Три состояния маркера: silence=красный (заглушим), keep=серый
        // (оставим), review=янтарный пунктир (ещё не проверено).
        const color =
          b.state === "silence"
            ? "rgb(244 63 94)"
            : b.state === "review"
            ? "rgb(251 191 36)"
            : "rgb(148 163 184)";
        const bodyOpacity = b.selected
          ? 0.28
          : b.state === "silence"
          ? 0.16
          : b.state === "review"
          ? 0.14
          : 0.1;
        // review рисуем пунктиром, чтобы «непроверенное» читалось даже на цвет-
        // слепом мониторе; принятые/отклонённые — сплошной обводкой.
        const dashed = b.state === "review";
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
              strokeDasharray={dashed ? "4 3" : undefined}
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

      {/* Playhead — координаты пишет rAF-цикл прямо в DOM (без ре-рендера) */}
      <line
        ref={playheadLineRef}
        x1={0}
        y1={0}
        x2={0}
        y2={VBH}
        stroke="rgb(244 244 245)"
        strokeWidth={1.2}
        pointerEvents="none"
        style={{ opacity: 0 }}
      />
      </svg>
    </div>
  );
}

/**
 * memo: экран правки перерисовывается на throttle-тик таймкода (~15 Гц) и на
 * каждый setView. Волна/блоки зависят только от peaks/view/blocks — плейхед
 * анимируется сам через ref. Мемо отсекает ре-рендеры, где сменился лишь
 * currentAudioS у родителя.
 */
export const TimelineCanvas = memo(TimelineCanvasImpl);
