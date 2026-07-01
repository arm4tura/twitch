import { useEffect, useImperativeHandle, useRef, forwardRef, useCallback } from "react";
import WaveSurfer from "wavesurfer.js";
import RegionsPlugin, {
  type Region,
} from "wavesurfer.js/dist/plugins/regions.js";

/**
 * Waveform — обёртка над wavesurfer.js@7 + Regions plugin.
 *
 * Ответственности:
 * 1. Инициализация wavesurfer с peaks-массивом (передаём с бэка, локально
 *    ffmpeg не считаем — там 200-400ms и требует бинарь).
 * 2. Синхронизация регионов из props: полный diff по id, mismatched → пересоздаём.
 *    Wavesurfer regions не reactive; ручной clear+add — самый надёжный путь.
 * 3. Проброс событий наружу через callbacks и через imperativeHandle
 *    (play/pause/seek/zoom — родителю нужен прямой контроль).
 *
 * НЕ ответственности:
 * - Загрузка audio — делает <audio> элемент родителя (HTML5 media element для
 *   Range-запросов). Wavesurfer только рисует peaks и региональный overlay,
 *   его `media` — синхронизирован с нашим <audio> через `setMediaElement`.
 */

export type RegionKind = "mute" | "cut" | "highlight";

export interface WFRegion {
  id: string;
  kind: RegionKind;
  start: number; // секунды
  end: number;
  label?: string;
}

const KIND_COLORS: Record<RegionKind, string> = {
  // Полупрозрачные — чтобы волна под ними просвечивала.
  mute: "rgba(244, 63, 94, 0.28)", // rose-500
  cut: "rgba(245, 158, 11, 0.28)", // amber-500
  highlight: "rgba(139, 92, 246, 0.32)", // violet-500 (brand)
};

const KIND_BORDER: Record<RegionKind, string> = {
  mute: "rgba(244, 63, 94, 0.9)",
  cut: "rgba(245, 158, 11, 0.9)",
  highlight: "rgba(139, 92, 246, 0.95)",
};

export interface WaveformProps {
  /** HTMLAudioElement для воспроизведения (родитель создаёт <audio>). */
  media: HTMLAudioElement | null;
  /** Peaks-массив с бэка (`/waveform`). */
  peaks: number[];
  /** Полная длительность в секундах (важна: wavesurfer не может её вывести из peaks). */
  duration: number;
  regions: WFRegion[];
  /** Пользователь двинул/растянул регион — обновить в store. */
  onRegionChange?: (id: string, start: number, end: number) => void;
  /** Клик по региону — выделить его в правой Sheet-панели. */
  onRegionClick?: (id: string) => void;
  /** currentTime сместилось — сообщить родителю (для отображения playhead и т.п.). */
  onTimeUpdate?: (t: number) => void;
  /** Готовность к вызовам play/pause/seek. */
  onReady?: () => void;
  /** Уровень зума 1..1000 (пикселей на секунду). */
  zoom?: number;
  className?: string;
}

export interface WaveformHandle {
  playPause(): void;
  isPlaying(): boolean;
  seek(timeS: number): void;
  getCurrentTime(): number;
  zoom(pxPerSec: number): void;
  scrollToTime(timeS: number): void;
}

export const Waveform = forwardRef<WaveformHandle, WaveformProps>(function Waveform(
  {
    media,
    peaks,
    duration,
    regions,
    onRegionChange,
    onRegionClick,
    onTimeUpdate,
    onReady,
    zoom = 50,
    className,
  },
  ref
) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const wsRef = useRef<WaveSurfer | null>(null);
  const regionsPluginRef = useRef<ReturnType<typeof RegionsPlugin.create> | null>(null);
  // Стабильные callbacks — иначе effect'ы пересоздавали бы wavesurfer.
  const cbRef = useRef({ onRegionChange, onRegionClick, onTimeUpdate, onReady });
  cbRef.current = { onRegionChange, onRegionClick, onTimeUpdate, onReady };

  // --- init + destroy -------------------------------------------------------

  useEffect(() => {
    if (!containerRef.current || !media || !peaks.length || !duration) return;

    const regionsPlugin = RegionsPlugin.create();
    const ws = WaveSurfer.create({
      container: containerRef.current,
      height: 96,
      waveColor: "rgb(148 163 184 / 0.55)", // slate-400/55
      progressColor: "rgb(139 92 246 / 0.9)", // brand violet
      cursorColor: "rgb(244 244 245)", // zinc-50
      cursorWidth: 1,
      barWidth: 2,
      barGap: 1,
      barRadius: 1,
      normalize: false,
      peaks: [peaks],
      duration,
      media,
      interact: true,
      hideScrollbar: false,
      minPxPerSec: zoom,
      plugins: [regionsPlugin],
    });
    wsRef.current = ws;
    regionsPluginRef.current = regionsPlugin;

    const disposers: Array<() => void> = [];
    disposers.push(ws.on("ready", () => cbRef.current.onReady?.()));
    disposers.push(ws.on("timeupdate", (t) => cbRef.current.onTimeUpdate?.(t)));
    disposers.push(
      regionsPlugin.on("region-updated", (r: Region) => {
        cbRef.current.onRegionChange?.(r.id, r.start, r.end);
      })
    );
    disposers.push(
      regionsPlugin.on("region-clicked", (r: Region, e: MouseEvent) => {
        e.stopPropagation();
        cbRef.current.onRegionClick?.(r.id);
      })
    );

    return () => {
      disposers.forEach((d) => d());
      ws.destroy();
      wsRef.current = null;
      regionsPluginRef.current = null;
    };
    // Пересоздаём wavesurfer только при смене источника (media/peaks/duration).
    // Зум и регионы синхронизируются в отдельных effect'ах — без destroy'я.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [media, peaks, duration]);

  // --- regions sync ---------------------------------------------------------

  useEffect(() => {
    const plug = regionsPluginRef.current;
    if (!plug) return;
    plug.clearRegions();
    for (const r of regions) {
      plug.addRegion({
        id: r.id,
        start: r.start,
        end: r.end,
        color: KIND_COLORS[r.kind],
        drag: true,
        resize: true,
        content: r.label
          ? Object.assign(document.createElement("div"), {
              className: "text-[10px] font-medium leading-none px-1 py-0.5 text-white",
              innerText: r.label,
            })
          : undefined,
      });
    }
  }, [regions]);

  // --- zoom sync ------------------------------------------------------------

  useEffect(() => {
    const ws = wsRef.current;
    if (!ws) return;
    try {
      ws.zoom(zoom);
    } catch {
      // Wavesurfer иногда бросает если контейнер ещё не смонтирован — молча ok.
    }
  }, [zoom]);

  // --- imperative API -------------------------------------------------------

  const seek = useCallback((t: number) => {
    const ws = wsRef.current;
    if (!ws || !duration) return;
    ws.setTime(Math.max(0, Math.min(duration, t)));
  }, [duration]);

  useImperativeHandle(
    ref,
    () => ({
      playPause: () => wsRef.current?.playPause(),
      isPlaying: () => !!wsRef.current?.isPlaying(),
      seek,
      getCurrentTime: () => wsRef.current?.getCurrentTime() ?? 0,
      zoom: (pxPerSec: number) => wsRef.current?.zoom(pxPerSec),
      scrollToTime: (t: number) => {
        const ws = wsRef.current;
        if (!ws) return;
        // Wavesurfer 7: явного scrollToTime нет — двигаем через setScroll.
        const px = t * (ws.options.minPxPerSec ?? zoom);
        const container = containerRef.current;
        if (!container) return;
        const half = container.clientWidth / 2;
        container.scrollLeft = Math.max(0, px - half);
      },
    }),
    [seek, zoom]
  );

  // --- markup ---------------------------------------------------------------

  return (
    <div
      ref={containerRef}
      className={className}
      // Радиус + фон, чтобы выглядело как «панель», а не голый canvas.
      style={{ borderRadius: 10, background: "rgb(24 24 27 / 0.7)" }}
      data-region-border-mute={KIND_BORDER.mute}
      data-region-border-cut={KIND_BORDER.cut}
      data-region-border-highlight={KIND_BORDER.highlight}
    />
  );
});
