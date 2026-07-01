import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { AlertCircle, Loader2 } from "lucide-react";
import { Card } from "../components/ui/Card";
import { EmptyState } from "../components/ui/EmptyState";
import { Button } from "../components/ui/Button";
import { Waveform, type WaveformHandle, type WFRegion } from "../components/timeline/Waveform";
import { TimelineToolbar, type TimelineTab } from "../components/timeline/TimelineToolbar";
import { RegionEditor, type EditorRegionInput } from "../components/timeline/RegionEditor";
import { Minimap } from "../components/timeline/Minimap";
import { useUndoable } from "../hooks/useUndoable";
import { useHotkey } from "../hooks/useHotkey";
import { usePublishTimelineActions } from "../lib/timelineActions";
import { toast } from "../components/ui/Toast";
import { platformizeShortcut } from "../lib/platform";
import {
  allowMediaPaths,
  getWaveform,
  mediaUrl,
  readDecisions,
  writeDecisions,
  type WaveformData,
} from "../api";
import type { Cut, Decisions, Highlight, Mute } from "../types/decisions";

/**
 * TimelineScreen — флагманский экран Фазы 5.
 *
 * Данные:
 * - decisions.json из `readDecisions(path)` — конвертируем в WFRegion[] +
 *   держим оригинал в `original`, чтобы обратно сериализовать при сохранении.
 * - Peaks — GET /waveform, кэшируется бэком; для «клика по Timeline снова»
 *   мы сами кэшируем WaveformData в useRef, чтобы не делать даже 304-запрос.
 * - HTML5 <audio> — `mediaUrl(streamPath)`; wavesurfer подписан на этот
 *   элемент через `media:` — playback идёт через один источник.
 *
 * Undo/redo — `useUndoable<Regions>` со стеком 50. Save дёргает writeDecisions
 * и делает `mark()`. ⌘Z / ⌘⇧Z / Space привязаны через useHotkey.
 */

interface Regions {
  mutes: WFRegion[];
  cuts: WFRegion[];
  highlights: WFRegion[];
}

const EMPTY: Regions = { mutes: [], cuts: [], highlights: [] };

export function TimelineScreen({ decisionsPath }: { decisionsPath: string }) {
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [wave, setWave] = useState<WaveformData | null>(null);
  const [streamPath, setStreamPath] = useState<string | null>(null);
  const [projectName, setProjectName] = useState<string>("");
  const originalRef = useRef<Decisions | null>(null);
  const wfRef = useRef<WaveformHandle | null>(null);
  const audioRef = useRef<HTMLAudioElement | null>(null);

  // --- undoable state -------------------------------------------------------
  const store = useUndoable<Regions>(EMPTY);

  // --- playback state -------------------------------------------------------
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentS, setCurrentS] = useState(0);
  const [wsReady, setWsReady] = useState(false);
  const [tab, setTab] = useState<TimelineTab>("all");
  const [zoom, setZoom] = useState<number>(50);
  const [viewportStartS, setViewportStartS] = useState(0);
  const [viewportWidthS, setViewportWidthS] = useState(60);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);

  // --- initial load ---------------------------------------------------------

  useEffect(() => {
    let alive = true;
    (async () => {
      setLoading(true);
      setError(null);
      try {
        const doc = (await readDecisions(decisionsPath)) as Decisions & {
          _meta?: { stream_path?: string; project?: string };
          source?: string;
          stream?: string;
        };
        if (!alive) return;
        originalRef.current = doc;
        setProjectName(doc._meta?.project ?? extractProjectName(decisionsPath));

        // Streaming source: приоритет
        //   1) _meta.stream_path — если UI/пользователь явно проставил;
        //   2) source — как build_decisions пишет в реальности (schema 1.1);
        //   3) stream — legacy-ключ, если попадётся старый файл;
        //   4) sibling-файл рядом (stream.mp4 и т.п.).
        const stream =
          doc._meta?.stream_path ??
          (typeof doc.source === "string" ? doc.source : undefined) ??
          (typeof doc.stream === "string" ? doc.stream : undefined) ??
          (await guessStreamPath(decisionsPath));
        if (!stream) {
          throw new Error(
            "Не нашли исходное видео/аудио. Проверьте, что рядом с decisions.json лежит stream.mp4/mkv, либо укажите путь в поле \"source\" внутри decisions.json."
          );
        }
        setStreamPath(stream);

        // Разрешить путь на бэке (whitelist) — обязательно ДО /waveform и /media.
        await allowMediaPaths([stream]).catch(() => {
          /* эндпоинт может ещё не быть — не критично, /waveform всё равно
             проверит whitelist сам и вернёт 403 с понятной ошибкой */
        });

        const w = await getWaveform(stream, 2048);
        if (!alive) return;
        setWave(w);
        setViewportWidthS(Math.min(60, w.duration_s));

        // Конвертация decisions → regions.
        store.reset({
          mutes: (doc.mutes ?? []).map((m, i) => muteToRegion(m, i)),
          cuts: (doc.cuts ?? []).map((c, i) => cutToRegion(c, i)),
          highlights: (doc.highlights?.highlights ?? []).map((h, i) =>
            highlightToRegion(h, i)
          ),
        });
      } catch (err) {
        if (!alive) return;
        setError(err instanceof Error ? err.message : String(err));
      } finally {
        if (alive) setLoading(false);
      }
    })();
    return () => {
      alive = false;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [decisionsPath]);

  // --- computed regions per active tab -------------------------------------

  const visibleRegions = useMemo<WFRegion[]>(() => {
    const { mutes, cuts, highlights } = store.present;
    switch (tab) {
      case "mutes": return mutes;
      case "cuts": return cuts;
      case "highlights": return highlights;
      default: return [...mutes, ...cuts, ...highlights];
    }
  }, [store.present, tab]);

  const counts = useMemo(
    () => ({
      mutes: store.present.mutes.length,
      cuts: store.present.cuts.length,
      highlights: store.present.highlights.length,
    }),
    [store.present]
  );

  const selectedRegion = useMemo<EditorRegionInput | null>(() => {
    if (!selectedId) return null;
    const all = [
      ...store.present.mutes,
      ...store.present.cuts,
      ...store.present.highlights,
    ];
    const r = all.find((x) => x.id === selectedId);
    if (!r) return null;
    // Обогащаем reason/score/words из оригинала.
    const doc = originalRef.current;
    const extras: Partial<EditorRegionInput> = {};
    if (r.kind === "mute") {
      const m = doc?.mutes?.[parseIdx(r.id)];
      if (m?.words) extras.words = m.words;
    } else if (r.kind === "cut") {
      const c = doc?.cuts?.[parseIdx(r.id)];
      if (c?.reason) extras.reason = c.reason;
    } else if (r.kind === "highlight") {
      const h = doc?.highlights?.highlights?.[parseIdx(r.id)];
      if (h) {
        extras.reason = h.reason;
        extras.score = h.score;
      }
    }
    return { ...r, ...extras };
  }, [selectedId, store.present]);

  // --- audio wiring ---------------------------------------------------------

  useEffect(() => {
    const el = audioRef.current;
    if (!el) return;
    const onPlay = () => setIsPlaying(true);
    const onPause = () => setIsPlaying(false);
    el.addEventListener("play", onPlay);
    el.addEventListener("pause", onPause);
    return () => {
      el.removeEventListener("play", onPlay);
      el.removeEventListener("pause", onPause);
    };
  }, [streamPath]);

  const onTimeUpdate = useCallback((t: number) => {
    setCurrentS(t);
    // Простейшая логика viewport'а для minimap: держим playhead в центре
    // (60 сек показываем; sliding window).
    setViewportStartS((prev) => {
      const half = viewportWidthS / 2;
      if (t < prev + 4 || t > prev + viewportWidthS - 4) {
        return Math.max(0, Math.min((wave?.duration_s ?? t) - viewportWidthS, t - half));
      }
      return prev;
    });
  }, [viewportWidthS, wave?.duration_s]);

  // --- region edits (through store.set) -------------------------------------

  const applyRegionChange = useCallback(
    (id: string, patch: Partial<WFRegion>) => {
      store.set((prev) => ({
        mutes: prev.mutes.map((r) => (r.id === id ? { ...r, ...patch } : r)),
        cuts: prev.cuts.map((r) => (r.id === id ? { ...r, ...patch } : r)),
        highlights: prev.highlights.map((r) => (r.id === id ? { ...r, ...patch } : r)),
      }));
    },
    [store]
  );

  const deleteRegion = useCallback(
    (id: string) => {
      store.set((prev) => ({
        mutes: prev.mutes.filter((r) => r.id !== id),
        cuts: prev.cuts.filter((r) => r.id !== id),
        highlights: prev.highlights.filter((r) => r.id !== id),
      }));
      setSelectedId(null);
      toast.success("Регион удалён", {
        description: `${platformizeShortcut("⌘Z")} чтобы вернуть`,
      });
    },
    [store]
  );

  const commitEditor = useCallback(
    (p: { id: string; start: number; end: number; reason?: string; score?: number }) => {
      // Сохраняем reason/score в originalRef, чтобы при save они попали в json.
      const doc = originalRef.current;
      if (doc) {
        const i = parseIdx(p.id);
        if (p.id.startsWith("cut-") && doc.cuts?.[i]) {
          doc.cuts[i].reason = p.reason;
        } else if (p.id.startsWith("hl-") && doc.highlights?.highlights?.[i]) {
          doc.highlights.highlights[i].reason = p.reason ?? doc.highlights.highlights[i].reason;
          if (p.score != null) doc.highlights.highlights[i].score = p.score;
        }
      }
      applyRegionChange(p.id, { start: p.start, end: p.end });
    },
    [applyRegionChange]
  );

  // --- save -----------------------------------------------------------------

  const save = useCallback(async () => {
    if (saving || !store.dirty) return;
    const doc = originalRef.current ?? {};
    const nextDoc: Decisions = {
      ...doc,
      mutes: store.present.mutes.map((r) => regionToMute(r, doc.mutes ?? [])),
      cuts: store.present.cuts.map((r) => regionToCut(r, doc.cuts ?? [])),
      highlights: {
        ...(doc.highlights ?? {}),
        highlights: store.present.highlights.map((r) =>
          regionToHighlight(r, doc.highlights?.highlights ?? [])
        ),
      },
    };
    setSaving(true);
    try {
      await writeDecisions(decisionsPath, nextDoc);
      originalRef.current = nextDoc;
      store.mark();
      toast.success("Сохранено", { description: decisionsPath });
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      setError(msg);
      toast.error("Не удалось сохранить", { description: msg });
    } finally {
      setSaving(false);
    }
  }, [saving, store, decisionsPath]);

  // --- hotkeys --------------------------------------------------------------

  useHotkey(" ", (e) => {
    e.preventDefault();
    wfRef.current?.playPause();
  });
  useHotkey("mod+z", (e) => {
    e.preventDefault();
    store.undo();
  });
  useHotkey(["mod+shift+z", "mod+y"], (e) => {
    e.preventDefault();
    store.redo();
  });
  useHotkey("mod+s", (e) => {
    e.preventDefault();
    save();
  });
  useHotkey("delete", () => {
    if (selectedId) deleteRegion(selectedId);
  });
  useHotkey("escape", () => setSelectedId(null));

  // Экспонируем текущие action'ы в глобальный store, чтобы CommandPalette
  // (⌘K из App.tsx) могла их вызвать без прокидывания callback'ов через
  // всю иерархию.
  usePublishTimelineActions({
    onSave: save,
    onUndo: store.undo,
    onRedo: store.redo,
    canUndo: store.canUndo,
    canRedo: store.canRedo,
    dirty: store.dirty,
  });

  // --- zoom controls --------------------------------------------------------

  const zoomIn = useCallback(() => setZoom((z) => Math.min(1000, Math.round(z * 1.5))), []);
  const zoomOut = useCallback(() => setZoom((z) => Math.max(5, Math.round(z / 1.5))), []);
  const fit = useCallback(() => {
    if (!wave) return;
    // Умещаем всю длительность в ~1000px (типичный монитор).
    const target = Math.max(5, Math.round(1000 / Math.max(wave.duration_s, 1)));
    setZoom(target);
  }, [wave]);

  // --- viewport width recompute on zoom ------------------------------------

  useEffect(() => {
    if (!wave) return;
    setViewportWidthS(Math.min(wave.duration_s, Math.max(5, 1000 / zoom)));
  }, [zoom, wave]);

  // --- render ---------------------------------------------------------------

  if (loading) {
    return (
      <div className="flex h-full items-center justify-center p-12">
        <div className="flex items-center gap-3 text-muted">
          <Loader2 className="h-5 w-5 animate-spin" />
          <span className="text-sm">Загружаем decisions и waveform…</span>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="mx-auto max-w-3xl p-8">
        <EmptyState
          icon={<AlertCircle className="h-6 w-6 text-err" />}
          title="Не удалось открыть проект"
          description={error}
        />
      </div>
    );
  }

  if (!wave || !streamPath) {
    return null;
  }

  const durationMs = wave.duration_s * 1000;
  const audioSrc = mediaUrl(streamPath);

  return (
    <div className="flex h-full min-h-0 flex-col gap-3 p-4">
      {/* Header */}
      <div className="flex items-baseline justify-between gap-4">
        <div className="min-w-0">
          <h1 className="truncate text-lg font-semibold text-fg">{projectName || "Проект"}</h1>
          <p className="mt-0.5 truncate font-mono text-[11px] text-subtle">{streamPath}</p>
        </div>
        {!wsReady && (
          <span className="flex items-center gap-1.5 text-xs text-subtle">
            <Loader2 className="h-3 w-3 animate-spin" /> инициализация волны
          </span>
        )}
      </div>

      {/* Toolbar */}
      <TimelineToolbar
        isPlaying={isPlaying}
        onPlayPause={() => wfRef.current?.playPause()}
        currentMs={currentS * 1000}
        totalMs={durationMs}
        tab={tab}
        onTabChange={setTab}
        counts={counts}
        zoom={zoom}
        onZoomIn={zoomIn}
        onZoomOut={zoomOut}
        onFit={fit}
        canUndo={store.canUndo}
        canRedo={store.canRedo}
        onUndo={store.undo}
        onRedo={store.redo}
        dirty={store.dirty}
        onSave={save}
        saving={saving}
      />

      {/* Main split: waveform + editor */}
      <div className="grid min-h-0 flex-1 grid-cols-1 gap-3 xl:grid-cols-[minmax(0,1fr)_320px]">
        <div className="flex min-h-0 flex-col gap-3">
          <Card variant="elevated" padding="sm" className="min-h-0 flex-1">
            <div className="h-full min-h-[220px] overflow-x-auto">
              <Waveform
                ref={wfRef}
                media={audioRef.current}
                peaks={wave.peaks}
                duration={wave.duration_s}
                regions={visibleRegions}
                zoom={zoom}
                onRegionChange={(id, s, e) => applyRegionChange(id, { start: s, end: e })}
                onRegionClick={(id) => setSelectedId(id)}
                onTimeUpdate={onTimeUpdate}
                onReady={() => setWsReady(true)}
                className="h-full w-full"
              />
            </div>
          </Card>
          <Minimap
            durationS={wave.duration_s}
            regions={[
              ...store.present.mutes,
              ...store.present.cuts,
              ...store.present.highlights,
            ]}
            currentS={currentS}
            viewportStartS={viewportStartS}
            viewportWidthS={viewportWidthS}
            onSeek={(t) => wfRef.current?.seek(t)}
          />
          {/* Hidden HTML5 audio — wavesurfer подписан на него как на media */}
          <audio ref={audioRef} src={audioSrc} preload="auto" className="hidden" />
        </div>

        <RegionEditor
          region={selectedRegion}
          durationMs={durationMs}
          onCommit={commitEditor}
          onDelete={deleteRegion}
          onClose={() => setSelectedId(null)}
        />
      </div>

      {store.dirty && (
        <div className="flex items-center justify-end gap-2 pr-1 text-[11px] text-subtle">
          В истории: {store.size} состояний. {platformizeShortcut("⌘Z")} / {platformizeShortcut("⌘⇧Z")} / {platformizeShortcut("⌘S")}.
        </div>
      )}

      {error && (
        <Card variant="surface" padding="sm" className="border-err/40 bg-err/10">
          <div className="flex items-start gap-2 text-sm text-err">
            <AlertCircle className="mt-0.5 h-4 w-4 shrink-0" />
            <span className="flex-1">{error}</span>
            <Button size="sm" variant="ghost" onClick={() => setError(null)}>
              ok
            </Button>
          </div>
        </Card>
      )}
    </div>
  );
}

// --- helpers ----------------------------------------------------------------

/** ID кодирует kind + index в оригинальном массиве — так восстанавливаем extras. */
function muteToRegion(m: Mute, i: number): WFRegion {
  return {
    id: `mute-${i}`,
    kind: "mute",
    start: m.start_ms / 1000,
    end: m.end_ms / 1000,
    label: m.words?.join(" · "),
  };
}
function cutToRegion(c: Cut, i: number): WFRegion {
  return {
    id: `cut-${i}`,
    kind: "cut",
    start: c.start_ms / 1000,
    end: c.end_ms / 1000,
    label: c.reason,
  };
}
function highlightToRegion(h: Highlight, i: number): WFRegion {
  return {
    id: `hl-${i}`,
    kind: "highlight",
    start: h.start_s,
    end: h.end_s,
    label: h.title,
  };
}
function parseIdx(id: string): number {
  const m = id.match(/(\d+)$/);
  return m ? Number(m[1]) : -1;
}

function regionToMute(r: WFRegion, originals: Mute[]): Mute {
  const orig = originals[parseIdx(r.id)];
  return {
    ...(orig ?? {}),
    start_ms: Math.round(r.start * 1000),
    end_ms: Math.round(r.end * 1000),
  };
}
function regionToCut(r: WFRegion, originals: Cut[]): Cut {
  const orig = originals[parseIdx(r.id)];
  return {
    ...(orig ?? {}),
    start_ms: Math.round(r.start * 1000),
    end_ms: Math.round(r.end * 1000),
  };
}
function regionToHighlight(r: WFRegion, originals: Highlight[]): Highlight {
  const orig = originals[parseIdx(r.id)];
  return {
    ...(orig ?? { title: r.label ?? "", reason: "", score: 0 }),
    start_s: r.start,
    end_s: r.end,
  };
}

function extractProjectName(decisionsPath: string): string {
  const sep = decisionsPath.includes("\\") ? "\\" : "/";
  const parts = decisionsPath.split(sep).filter(Boolean);
  // .../workdir/decisions.json → workdir
  if (parts.length >= 2) return parts[parts.length - 2];
  return decisionsPath;
}

/**
 * Пытаемся угадать stream-file рядом с decisions.json. Backend может дать
 * подсказку через `_meta.stream_path`, а если её нет — берём первый файл в
 * той же папке с расширением из белого списка. Простая эвристика, но покрывает
 * дефолтный сценарий «process → decisions.json + stream.mp4 в workdir'e».
 */
async function guessStreamPath(decisionsPath: string): Promise<string | null> {
  const sep = decisionsPath.includes("\\") ? "\\" : "/";
  const dir = decisionsPath.split(sep).slice(0, -1).join(sep);
  // К сожалению, у нас нет /listdir эндпоинта — просто пробуем набор
  // очевидных имён; если ни один не подошёл — вернём null и покажем понятную
  // ошибку в UI. Пользователь тогда откроет json и дополнит `_meta.stream_path`.
  const candidates = ["stream.mp4", "stream.mkv", "stream.m4a", "stream.mp3", "original.mp4"];
  for (const name of candidates) {
    const guess = `${dir}${sep}${name}`;
    // Разрешаем на бэке и полагаемся на успешный allow (`allowed` включает путь).
    try {
      const res = await allowMediaPaths([guess]);
      if (res.allowed.some((p) => p.endsWith(name))) return guess;
    } catch {
      /* try next */
    }
  }
  return null;
}
