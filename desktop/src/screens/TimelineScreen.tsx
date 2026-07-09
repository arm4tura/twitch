import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  AlertCircle,
  ChevronLeft,
  ChevronRight,
  Loader2,
  Maximize2,
  Plus,
  RotateCcw,
  RotateCw,
  Save,
  VolumeX,
  ZoomIn,
  ZoomOut,
} from "lucide-react";
import { Card } from "../components/ui/Card";
import { EmptyState } from "../components/ui/EmptyState";
import { Button } from "../components/ui/Button";
import { Kbd } from "../components/ui/Kbd";
import { OverviewStrip, type OverviewMarker } from "../components/timeline/OverviewStrip";
import {
  TimelineCanvas,
  type TimelineBlock,
  type TimelineView,
} from "../components/timeline/TimelineCanvas";
import { MuteInspector } from "../components/timeline/MuteInspector";
import { useUndoable } from "../hooks/useUndoable";
import { useHotkey } from "../hooks/useHotkey";
import { usePublishTimelineActions } from "../lib/timelineActions";
import { toast } from "../components/ui/Toast";
import { platformizeShortcut } from "../lib/platform";
import { fmtMs } from "../lib/format";
import {
  buildWordIndex,
  contextWords,
  playWindow,
  snapToWordBoundary,
  wordsInRange,
  type WordIndex,
} from "../lib/transcript";
import {
  allowMediaPaths,
  getWaveform,
  mediaUrl,
  readDecisions,
  readTranscript,
  writeDecisions,
  type WaveformData,
} from "../api";
import type { DecisionsDoc, MuteRecord, TranscriptDoc } from "../types/project";

/**
 * TimelineScreen — экран «Правка» = РЕВЬЮ ЗАГЛУШЕК на DAW-таймлайне.
 *
 * Задача пользователя тут одна: пройтись по найденным матам и решить, что
 * глушить, а что оставить (плюс поправить границы / добавить пропущенное).
 * Раскладка — как в Vegas/Reaper, один зумируемый таймлайн всего стрима:
 *
 *  - OverviewStrip (архетип A): вся запись + рамка текущего окна зума (мини-карта).
 *  - Нав-бар: ◀/▶ между матами, зум −/+/весь стрим, добавить мат.
 *  - TimelineCanvas: сплошная волна всего стрима, маты — цветные блоки; зум
 *    колесом, пан shift+колесом, драг краёв блока правит границы без снапа.
 *  - MuteInspector: контекст ВЫБРАННОГО мата — статус, ±0.1с, превью, транскрипт
 *    (клик по слову = снап к словам).
 *
 * Cuts/highlights здесь НЕ показываем — это NotebookLM-конвейер, ему место в
 * «Экспорт → Дополнительно». Мы читаем/пишем только `mutes` реальной схемы
 * (start/end в секундах, status accepted|rejected|review).
 *
 * Координаты: правим в ЛОКАЛЬНОМ пространстве (start/end — от range_in; совпадает
 * с транскриптом и извлечённым cache-audio). Волна/блоки/плейбек живут в АУДИО-
 * пространстве (== локальному для cache-audio, со сдвигом offset для legacy
 * source). Конвертация — toAudio/toLocal. При сохранении пересчитываем
 * stream_start/stream_end — их читает Vegas-экспорт.
 */

/** UI-модель мьюта: локальные секунды + судьба. Исходную запись храним для round-trip. */
interface UiMute {
  id: string;
  start: number;
  end: number;
  word: string;
  status: string;
  review: boolean;
  raw: MuteRecord;
}

const SILENCED = "accepted"; // статус, который Vegas реально глушит
const KEPT = "rejected"; // «оставить» — Vegas пропускает
const MIN_MUTE_S = 0.03; // единый минимум длины мьюта (волна/кнопки/клик) — чтобы не схлопнуть в ноль
const VIEW_MIN_SPAN = 0.4; // минимальная ширина окна зума таймлайна, сек
const ZOOM_IN = 1 / 1.6; // множитель span при «приблизить»
const ZOOM_OUT = 1.6; // множитель span при «отдалить»

function toUiMute(m: MuteRecord): UiMute {
  const status = typeof m.status === "string" ? m.status : SILENCED;
  return {
    id: m.id,
    start: Number(m.start) || 0,
    end: Number(m.end) || 0,
    word: typeof m.word === "string" ? m.word : "",
    status,
    review: status === "review" || !!m.needs_review,
    raw: m,
  };
}

export function TimelineScreen({ decisionsPath }: { decisionsPath: string }) {
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [wave, setWave] = useState<WaveformData | null>(null);
  const [projectName, setProjectName] = useState("");

  // Источник аудио + система координат.
  const [audioPath, setAudioPath] = useState<string | null>(null);
  const localCoordsRef = useRef(true); // true → audio == локальное пространство
  const offsetRef = useRef(0); // source-space: audioTime = localTime + offset
  const docRef = useRef<DecisionsDoc | null>(null);
  const wordIndexRef = useRef<WordIndex>({ words: [] });

  const audioRef = useRef<HTMLAudioElement | null>(null);
  const playUntilRef = useRef<number | null>(null); // audio-секунда, на которой стоп
  const stopTimerRef = useRef<number | null>(null); // setTimeout-страховка стопа
  // Реалтайм-глушение при превью «как в экспорте»: во время playMute бегает rAF
  // и глушит <audio> ровно над заглушёнными участками. Скраб/«прослушать
  // выделение» — наоборот, играют оригинал (нужно услышать мат, чтобы поймать
  // границы), поэтому applyMutesRef там false.
  const rafRef = useRef<number | null>(null);
  const applyMutesRef = useRef(false);
  const muteRegionsRef = useRef<Array<[number, number]>>([]);

  // Undoable-список мьютов.
  const store = useUndoable<UiMute[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [currentAudioS, setCurrentAudioS] = useState(0);
  const [playingMuteId, setPlayingMuteId] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);
  // Окно зума главного таймлайна (АУДИО-секунды). null до загрузки волны.
  const [view, setView] = useState<TimelineView | null>(null);

  // --- координатные хелперы -------------------------------------------------
  const toAudio = useCallback(
    (localT: number) => (localCoordsRef.current ? localT : localT + offsetRef.current),
    []
  );
  const toLocal = useCallback(
    (audioT: number) => (localCoordsRef.current ? audioT : audioT - offsetRef.current),
    []
  );

  // --- загрузка -------------------------------------------------------------
  useEffect(() => {
    let alive = true;
    (async () => {
      setLoading(true);
      setError(null);
      try {
        const doc = (await readDecisions(decisionsPath)) as DecisionsDoc;
        if (!alive) return;
        docRef.current = doc;
        setProjectName(doc._meta?.project ?? extractProjectName(decisionsPath));

        const rawMutes = Array.isArray(doc.mutes) ? doc.mutes : [];
        const ui = rawMutes.map(toUiMute).sort((a, b) => a.start - b.start);

        // Offset (range_in в секундах) — из первого мьюта, где есть stream_start.
        const withStream = rawMutes.find(
          (m) => Number.isFinite(m.stream_start) && Number.isFinite(m.start)
        );
        offsetRef.current = withStream
          ? Number(withStream.stream_start) - Number(withStream.start)
          : 0;

        // Аудио-источник: извлечённый диапазон (локальные координаты) в приоритете.
        const cacheAudio = doc.caches?.audio ?? null;
        const source =
          doc._meta?.stream_path ??
          (typeof doc.source === "string" ? doc.source : undefined) ??
          (typeof doc.stream === "string" ? doc.stream : undefined) ??
          null;
        const media = cacheAudio ?? source;
        if (!media) {
          throw new Error(
            "Не нашли аудио для проигрывания. Проверьте caches.audio или source в decisions.json."
          );
        }
        localCoordsRef.current = !!cacheAudio;
        setAudioPath(media);

        await allowMediaPaths([media]).catch(() => {
          /* whitelist проверится и на /waveform — не критично */
        });

        // Транскрипт (вариант B) — не блокирующий: если нет, список всё равно работает.
        const transcriptPath = doc.caches?.transcript ?? null;
        if (transcriptPath) {
          try {
            const t = (await readTranscript(transcriptPath)) as TranscriptDoc;
            if (alive) wordIndexRef.current = buildWordIndex(t);
          } catch {
            /* транскрипт опционален */
          }
        }

        // Просим максимум пиков (8192): мини-волна одного мьюта вырезает узкое
        // окно из общего массива, и на 2048 в него попадала пара точек — рисовались
        // «кирпичи», а не волна. OverviewStrip всё равно даунсэмплит до 500, так что
        // на общую полосу это не влияет по скорости.
        const w = await getWaveform(media, 8192);
        if (!alive) return;
        setWave(w);
        setView({ start: 0, end: w.duration_s }); // старт — весь стрим
        store.reset(ui);
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

  const mutes = store.present;

  // --- производные ----------------------------------------------------------
  const silencedCount = useMemo(
    () => mutes.filter((m) => m.status === SILENCED).length,
    [mutes]
  );

  const markers = useMemo<OverviewMarker[]>(
    () =>
      mutes.map((m) => ({
        id: m.id,
        start: toAudio(m.start),
        end: toAudio(m.end),
        active: m.status === SILENCED,
        selected: m.id === selectedId,
      })),
    [mutes, selectedId, toAudio]
  );

  const selected = useMemo(
    () => mutes.find((m) => m.id === selectedId) ?? null,
    [mutes, selectedId]
  );

  const selectedContext = useMemo(() => {
    if (!selected) return [];
    // Прочие заглушки — их слова полностью убираем из контекста этой строки.
    const others = mutes
      .filter((m) => m.id !== selected.id && m.status === SILENCED)
      .map((m) => ({ start: m.start, end: m.end }));
    // Окно контекста = 1 слово до + мат + 2 слова после (в пределах сегмента).
    return contextWords(wordIndexRef.current, selected.start, selected.end, 1, 2, others);
  }, [selected, mutes]);

  const currentLocalS = toLocal(currentAudioS);

  // --- аудио-плейбек --------------------------------------------------------
  // Единая точка остановки. Дёргается из двух источников: браузерного
  // `timeupdate` (редкий, ~4 раза/сек, может проскочить границу) и из точного
  // setTimeout-таймера, заведённого при старте (см. playAudioRange). Любой,
  // кто сработал первым, глушит воспроизведение — второй становится no-op.
  const stopPlayback = useCallback(() => {
    const el = audioRef.current;
    if (stopTimerRef.current != null) {
      window.clearTimeout(stopTimerRef.current);
      stopTimerRef.current = null;
    }
    if (rafRef.current != null) {
      cancelAnimationFrame(rafRef.current);
      rafRef.current = null;
    }
    applyMutesRef.current = false;
    playUntilRef.current = null;
    if (el) {
      if (!el.paused) el.pause();
      el.muted = false; // снять реалтайм-глушение
    }
    setPlayingMuteId(null);
  }, []);

  /**
   * rAF-скан во время превью «как в экспорте»: чаще, чем timeupdate (~4 Гц),
   * поэтому граница мьюта не «протекает». Глушит <audio>.muted, когда
   * currentTime попадает в заглушённый участок, и снимает глушение вне их.
   * Также двигает плейхед плавно и ловит стоп по playUntil.
   */
  const scanMutes = useCallback(() => {
    const el = audioRef.current;
    if (!el) return;
    const t = el.currentTime;
    setCurrentAudioS(t);
    if (applyMutesRef.current) {
      let inMute = false;
      for (const [a, b] of muteRegionsRef.current) {
        if (t >= a && t < b) {
          inMute = true;
          break;
        }
      }
      if (el.muted !== inMute) el.muted = inMute;
    }
    if (playUntilRef.current != null && t >= playUntilRef.current) {
      stopPlayback();
      return;
    }
    rafRef.current = requestAnimationFrame(scanMutes);
  }, [stopPlayback]);

  // Обработчики вешаются как пропсы на сам <audio> (см. render), а не через
  // addEventListener в useEffect: элемент монтируется только когда готовы и
  // wave, и audioPath, и раньше слушатель успевал привязаться до монтирования
  // и больше не перепривязывался — стоп не срабатывал вовсе.
  const onAudioTime = useCallback(() => {
    const el = audioRef.current;
    if (!el) return;
    setCurrentAudioS(el.currentTime);
    if (playUntilRef.current != null && el.currentTime >= playUntilRef.current) {
      stopPlayback();
    }
  }, [stopPlayback]);

  const onAudioPause = useCallback(() => {
    // Ручная пауза (не наша граница) — снять подсветку играющей строки.
    if (playUntilRef.current == null) setPlayingMuteId(null);
  }, []);

  // Снять таймер/rAF при размонтировании экрана.
  useEffect(
    () => () => {
      if (stopTimerRef.current != null) window.clearTimeout(stopTimerRef.current);
      if (rafRef.current != null) cancelAnimationFrame(rafRef.current);
    },
    []
  );

  /**
   * Проиграть аудио-отрезок [aStart, aEnd] (в audio-координатах).
   * `pad` — сколько добавить с каждого края (контекст). Для превью мьюта — 0.25с,
   * для «слышно край» (скраб) — почти ноль, чтобы получился короткий блип.
   * `applyMutes` — глушить ли заглушённые участки в реальном времени (превью
   * «как в экспорте»). Для скраба/поиска границ — false (нужен оригинал).
   */
  const playAudioRange = useCallback(
    (aStart: number, aEnd: number, pad = 0.25, applyMutes = false) => {
      const el = audioRef.current;
      if (!el) return;
      const from = Math.max(0, aStart - pad);
      const until = aEnd + pad;
      el.currentTime = from;
      playUntilRef.current = until;
      // Реалтайм-глушение: список заглушённых участков в audio-координатах.
      applyMutesRef.current = applyMutes;
      el.muted = false;
      if (applyMutes) {
        muteRegionsRef.current = store.present
          .filter((m) => m.status === SILENCED)
          .map((m) => [toAudio(m.start), toAudio(m.end)] as [number, number]);
      } else {
        muteRegionsRef.current = [];
      }
      // Точная страховка: timeupdate/rAF могут не совпасть с концом окна —
      // дублируем стоп таймером на длину окна (playbackRate = 1).
      if (stopTimerRef.current != null) window.clearTimeout(stopTimerRef.current);
      const ms = Math.max(0, (until - from) * 1000);
      stopTimerRef.current = window.setTimeout(() => {
        stopTimerRef.current = null;
        stopPlayback();
      }, ms);
      // Плавный плейхед + точное глушение через rAF (частит, в отличие от timeupdate).
      if (rafRef.current != null) cancelAnimationFrame(rafRef.current);
      rafRef.current = requestAnimationFrame(scanMutes);
      void el.play().catch(() => {
        /* автоплей может быть заблокирован до первого жеста — игнор */
      });
    },
    [stopPlayback, scanMutes, store, toAudio]
  );

  const playMute = useCallback(
    (m: UiMute) => {
      const el = audioRef.current;
      if (!el) return;
      if (playingMuteId === m.id && !el.paused) {
        stopPlayback();
        return;
      }
      setPlayingMuteId(m.id);
      // Проигрываем мат в контексте: 1 слово до + сам мат + 2 слова после.
      // applyMutes=true → превью «как в экспорте»: заглушённые участки звучат
      // тишиной. Так «Заглушить (в ноль)» слышно сразу, а не только в Vegas.
      const win = playWindow(wordIndexRef.current, m.start, m.end, 1, 2);
      playAudioRange(toAudio(win.start), toAudio(win.end), 0.25, true);
    },
    [playingMuteId, playAudioRange, stopPlayback, toAudio]
  );

  const seekAudio = useCallback((audioT: number) => {
    const el = audioRef.current;
    if (!el) return;
    if (stopTimerRef.current != null) {
      window.clearTimeout(stopTimerRef.current);
      stopTimerRef.current = null;
    }
    if (rafRef.current != null) {
      cancelAnimationFrame(rafRef.current);
      rafRef.current = null;
    }
    applyMutesRef.current = false;
    playUntilRef.current = null;
    el.muted = false;
    el.currentTime = Math.max(0, audioT);
    setCurrentAudioS(el.currentTime);
  }, []);

  /** Скраб в АУДИО-координатах (драг края блока на таймлайне) — короткий блип. */
  const scrubAtAudio = useCallback(
    (audioT: number) => {
      playAudioRange(audioT, audioT, 0.05);
    },
    [playAudioRange]
  );

  // --- зум / пан / центрирование окна таймлайна -----------------------------
  /** Ставим окно [start,end], клэмпя ширину в [MIN, dur] и края в [0, dur]. */
  const setViewClamped = useCallback(
    (start: number, end: number) => {
      if (!wave) return;
      const dur = Math.max(wave.duration_s, 0.001);
      const span = Math.max(VIEW_MIN_SPAN, Math.min(end - start, dur));
      let s = start;
      let e = start + span;
      if (s < 0) {
        s = 0;
        e = span;
      }
      if (e > dur) {
        e = dur;
        s = Math.max(0, dur - span);
      }
      setView({ start: s, end: e });
    },
    [wave]
  );

  /**
   * Центрировать окно на мате [aStart,aEnd] (audio-сек). Текущий зум сохраняем,
   * но если мат длиннее окна — расширяем окно, чтобы он влез с запасом.
   */
  const centerViewOn = useCallback(
    (aStart: number, aEnd: number) => {
      if (!wave) return;
      const dur = Math.max(wave.duration_s, 0.001);
      const len = Math.max(aEnd - aStart, 0.001);
      const curSpan = view ? view.end - view.start : dur;
      let span = curSpan;
      if (len > curSpan * 0.6) span = Math.min(dur, len * 2.5 + 0.5);
      span = Math.min(span, dur);
      const c = (aStart + aEnd) / 2;
      setViewClamped(c - span / 2, c + span / 2);
    },
    [wave, view, setViewClamped]
  );

  /** Зум вокруг плейхеда (если он в окне) или центра окна. */
  const zoomBy = useCallback(
    (factor: number) => {
      if (!wave || !view) return;
      const dur = Math.max(wave.duration_s, 0.001);
      const curSpan = view.end - view.start;
      const anchor =
        currentAudioS >= view.start && currentAudioS <= view.end
          ? currentAudioS
          : (view.start + view.end) / 2;
      const frac = curSpan > 0 ? (anchor - view.start) / curSpan : 0.5;
      const span = Math.max(VIEW_MIN_SPAN, Math.min(curSpan * factor, dur));
      setViewClamped(anchor - frac * span, anchor - frac * span + span);
    },
    [wave, view, currentAudioS, setViewClamped]
  );

  const fitAll = useCallback(() => {
    if (!wave) return;
    setView({ start: 0, end: wave.duration_s });
  }, [wave]);

  /** Выбрать мат: подсветить, опц. центрировать окно и/или проиграть. */
  const selectMute = useCallback(
    (id: string, opts?: { play?: boolean; center?: boolean }) => {
      const m = store.present.find((x) => x.id === id);
      setSelectedId(id);
      if (!m) return;
      if (opts?.center) centerViewOn(toAudio(m.start), toAudio(m.end));
      if (opts?.play) playMute(m);
    },
    [store, centerViewOn, toAudio, playMute]
  );

  /** Прыжок к пред/след мату по времени (список отсортирован по start). */
  const goToMute = useCallback(
    (dir: 1 | -1) => {
      const list = store.present;
      if (!list.length) return;
      const i = list.findIndex((m) => m.id === selectedId);
      const next =
        i === -1
          ? dir > 0
            ? 0
            : list.length - 1
          : Math.max(0, Math.min(list.length - 1, i + dir));
      selectMute(list[next].id, { center: true });
    },
    [store, selectedId, selectMute]
  );

  // --- редактирование мьютов ------------------------------------------------
  const updateMute = useCallback(
    (id: string, patch: Partial<UiMute>) => {
      store.set((prev) =>
        prev.map((m) => (m.id === id ? { ...m, ...patch } : m))
      );
    },
    [store]
  );

  const toggleMuted = useCallback(
    (m: UiMute) => {
      updateMute(m.id, { status: m.status === SILENCED ? KEPT : SILENCED, review: false });
    },
    [updateMute]
  );

  /** Клик по слову в транскрипте: снап границ к словам + пересчёт слова-подписи. */
  const adjustMute = useCallback(
    (id: string, start: number, end: number) => {
      const idx = wordIndexRef.current;
      const s = snapToWordBoundary(idx, start);
      const e = snapToWordBoundary(idx, end);
      if (e - s < MIN_MUTE_S) return; // защита от схлопывания
      const covered = wordsInRange(idx, s, e)
        .map((w) => w.word)
        .join(" ")
        .trim();
      updateMute(id, { start: s, end: e, word: covered || undefined });
    },
    [updateMute]
  );

  /**
   * Точный сдвиг границ на дельту (секунды), БЕЗ снапа к словам — для кнопок
   * −0.1/+0.1с. Клэмпим минимальную длину и переисчисляем слово-подпись.
   */
  const nudgeMute = useCallback(
    (id: string, dStart: number, dEnd: number) => {
      const m = store.present.find((x) => x.id === id);
      if (!m) return;
      const idx = wordIndexRef.current;
      let s = Math.max(0, m.start + dStart);
      let e = m.end + dEnd;
      if (e - s < MIN_MUTE_S) {
        // не даём краю перескочить через противоположный: упираем в MIN.
        if (dEnd !== 0) e = s + MIN_MUTE_S;
        else s = e - MIN_MUTE_S;
      }
      const covered = wordsInRange(idx, s, e)
        .map((w) => w.word)
        .join(" ")
        .trim();
      updateMute(id, { start: s, end: e, word: covered || undefined });
    },
    [store, updateMute]
  );

  /**
   * Свободная правка границ ТОЧНО как задал пользователь (драг по волне), БЕЗ
   * снапа к словам. Волна обещает покадровую точность — снап бы отбрасывал
   * подрезку назад к краю слова. Снап к словам оставлен только для клика по
   * слову в транскрипте (adjustMute).
   */
  const adjustMuteFree = useCallback(
    (id: string, start: number, end: number) => {
      const idx = wordIndexRef.current;
      let s = Math.max(0, start);
      let e = end;
      if (e - s < MIN_MUTE_S) e = s + MIN_MUTE_S;
      const covered = wordsInRange(idx, s, e)
        .map((w) => w.word)
        .join(" ")
        .trim();
      updateMute(id, { start: s, end: e, word: covered || undefined });
    },
    [updateMute]
  );

  const deleteMute = useCallback(
    (id: string) => {
      store.set((prev) => prev.filter((m) => m.id !== id));
      if (selectedId === id) setSelectedId(null);
      toast.success("Заглушка удалена", {
        description: `${platformizeShortcut("⌘Z")} чтобы вернуть`,
      });
    },
    [store, selectedId]
  );

  const addMuteHere = useCallback(() => {
    const local = currentLocalS;
    const idx = wordIndexRef.current;
    // По умолчанию накрываем слово под курсором (если есть) или 0.6с.
    const hit = wordsInRange(idx, local, local + 0.01)[0];
    const start = hit ? hit.start : local;
    const end = hit ? hit.end : local + 0.6;
    const covered = wordsInRange(idx, start, end)
      .map((w) => w.word)
      .join(" ")
      .trim();
    const id = `mute_manual_${Math.round(start * 1000)}`;
    const created: UiMute = {
      id,
      start,
      end,
      word: covered,
      status: SILENCED,
      review: false,
      raw: {
        id,
        start,
        end,
        word: covered,
        status: SILENCED,
        reason: "profanity",
        source: "manual",
      },
    };
    store.set((prev) =>
      [...prev, created].sort((a, b) => a.start - b.start)
    );
    setSelectedId(id);
    centerViewOn(toAudio(start), toAudio(end));
    toast.success("Заглушка добавлена", { description: fmtMs(start * 1000) });
  }, [currentLocalS, store, centerViewOn, toAudio]);

  // --- сохранение -----------------------------------------------------------
  const save = useCallback(async () => {
    if (saving || !store.dirty) return;
    const doc = docRef.current ?? {};
    const off = offsetRef.current;
    // Собираем mutes обратно: правим start/end/word/status, пересчитываем
    // stream_start/stream_end (их читает Vegas), сохраняем прочие поля.
    const nextMutes: MuteRecord[] = store.present.map((m) => {
      const streamStart =
        Number.isFinite(m.raw.stream_start) || off !== 0 ? m.start + off : undefined;
      const streamEnd =
        Number.isFinite(m.raw.stream_end) || off !== 0 ? m.end + off : undefined;
      return {
        ...m.raw,
        id: m.id,
        start: round3(m.start),
        end: round3(m.end),
        word: m.word,
        status: m.status,
        ...(streamStart != null ? { stream_start: round3(streamStart) } : {}),
        ...(streamEnd != null ? { stream_end: round3(streamEnd) } : {}),
      };
    });
    const nextDoc: DecisionsDoc = { ...doc, mutes: nextMutes };
    setSaving(true);
    try {
      await writeDecisions(decisionsPath, nextDoc);
      docRef.current = nextDoc;
      store.mark();
      toast.success("Сохранено", {
        description: `Заглушим ${store.present.filter((m) => m.status === SILENCED).length} из ${store.present.length}`,
      });
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      setError(msg);
      toast.error("Не удалось сохранить", { description: msg });
    } finally {
      setSaving(false);
    }
  }, [saving, store, decisionsPath]);

  // --- навигация по матам клавишами -----------------------------------------
  useHotkey(["arrowdown", "arrowright"], (e) => {
    e.preventDefault();
    goToMute(1);
  });
  useHotkey(["arrowup", "arrowleft"], (e) => {
    e.preventDefault();
    goToMute(-1);
  });
  useHotkey(" ", (e) => {
    e.preventDefault();
    const m = mutes.find((x) => x.id === selectedId) ?? mutes[0];
    if (m) playMute(m);
  });
  useHotkey(["m", "enter"], () => {
    const m = mutes.find((x) => x.id === selectedId);
    if (m) toggleMuted(m);
  });
  useHotkey("delete", () => {
    if (selectedId) deleteMute(selectedId);
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
  useHotkey("escape", () => setSelectedId(null));

  usePublishTimelineActions({
    onSave: save,
    onUndo: store.undo,
    onRedo: store.redo,
    canUndo: store.canUndo,
    canRedo: store.canRedo,
    dirty: store.dirty,
  });

  // --- render ---------------------------------------------------------------
  if (loading) {
    return (
      <div className="flex h-full items-center justify-center p-12">
        <div className="flex items-center gap-3 text-muted">
          <Loader2 className="h-5 w-5 animate-spin" />
          <span className="text-sm">Загружаем заглушки и транскрипт…</span>
        </div>
      </div>
    );
  }

  if (error && !wave) {
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

  if (!wave || !audioPath || !view) return null;

  const audioSrc = mediaUrl(audioPath);
  const selectedIndex = selected ? mutes.findIndex((m) => m.id === selected.id) : -1;

  return (
    <div className="flex h-full min-h-0 flex-col gap-3 p-4">
      {/* Header */}
      <div className="flex items-center justify-between gap-4">
        <div className="min-w-0">
          <h1 className="truncate text-lg font-semibold text-fg">
            {projectName || "Проект"}
          </h1>
          <p className="mt-0.5 text-sm text-muted">
            Проверьте заглушки: что глушим, что оставляем.{" "}
            <span className="text-fg">
              Заглушим {silencedCount} из {mutes.length}
            </span>
          </p>
        </div>
        <div className="flex items-center gap-2">
          <div className="flex items-center gap-1 rounded-md border border-white/8 bg-black/30 p-1">
            <Button
              size="sm"
              variant="ghost"
              onClick={store.undo}
              disabled={!store.canUndo}
              className="h-7 w-7 p-0"
              aria-label="Отменить"
            >
              <RotateCcw className="h-3.5 w-3.5" />
            </Button>
            <Button
              size="sm"
              variant="ghost"
              onClick={store.redo}
              disabled={!store.canRedo}
              className="h-7 w-7 p-0"
              aria-label="Повторить"
            >
              <RotateCw className="h-3.5 w-3.5" />
            </Button>
          </div>
          <Button
            size="sm"
            variant={store.dirty ? "primary" : "secondary"}
            onClick={save}
            disabled={!store.dirty || saving}
            loading={saving}
          >
            <Save className="mr-1.5 h-3.5 w-3.5" /> Сохранить <Kbd className="ml-2">⌘S</Kbd>
          </Button>
        </div>
      </div>

      {/* Мини-карта: весь стрим + рамка окна зума */}
      <OverviewStrip
        durationS={wave.duration_s}
        peaks={wave.peaks}
        markers={markers}
        currentS={currentAudioS}
        view={view}
        onSeek={seekAudio}
        onPickMarker={(id) => selectMute(id, { center: true })}
      />

      {/* Нав-бар: переход по матам · зум · добавить */}
      <div className="flex flex-wrap items-center gap-2">
        <div className="flex items-center gap-1 rounded-md border border-white/8 bg-black/30 p-1">
          <Button
            size="sm"
            variant="ghost"
            onClick={() => goToMute(-1)}
            disabled={!mutes.length}
            className="h-7 w-7 p-0"
            aria-label="Предыдущий мат"
          >
            <ChevronLeft className="h-4 w-4" />
          </Button>
          <span className="min-w-[74px] text-center font-mono text-[11px] tabular-nums text-subtle">
            {selectedIndex >= 0 ? `мат ${selectedIndex + 1}/${mutes.length}` : `матов ${mutes.length}`}
          </span>
          <Button
            size="sm"
            variant="ghost"
            onClick={() => goToMute(1)}
            disabled={!mutes.length}
            className="h-7 w-7 p-0"
            aria-label="Следующий мат"
          >
            <ChevronRight className="h-4 w-4" />
          </Button>
        </div>

        <div className="flex items-center gap-1 rounded-md border border-white/8 bg-black/30 p-1">
          <Button
            size="sm"
            variant="ghost"
            onClick={() => zoomBy(ZOOM_OUT)}
            className="h-7 w-7 p-0"
            aria-label="Отдалить"
          >
            <ZoomOut className="h-3.5 w-3.5" />
          </Button>
          <Button
            size="sm"
            variant="ghost"
            onClick={() => zoomBy(ZOOM_IN)}
            className="h-7 w-7 p-0"
            aria-label="Приблизить"
          >
            <ZoomIn className="h-3.5 w-3.5" />
          </Button>
          <Button
            size="sm"
            variant="ghost"
            onClick={fitAll}
            className="h-7 px-2 text-xs"
            aria-label="Весь стрим"
          >
            <Maximize2 className="mr-1 h-3.5 w-3.5" /> весь стрим
          </Button>
        </div>

        <Button size="sm" variant="secondary" onClick={addMuteHere}>
          <Plus className="mr-1.5 h-3.5 w-3.5" /> мат на плейхеде
        </Button>

        <span className="font-mono text-[11px] tabular-nums text-subtle">
          {fmtMs(currentAudioS * 1000)} / {fmtMs(wave.duration_s * 1000)}
        </span>

        <span className="ml-auto flex items-center gap-1.5 text-xs text-subtle">
          <VolumeX className="h-3.5 w-3.5 text-err" /> {silencedCount} заглушим
          <span className="mx-1 text-white/15">·</span>
          {mutes.length - silencedCount} оставим
        </span>
      </div>

      {/* Главный таймлайн */}
      {mutes.length === 0 && (
        <Card variant="surface" padding="sm">
          <p className="text-center text-xs text-subtle">
            Заглушек нет — мат не найден или все убраны. «+ мат на плейхеде» добавит вручную.
          </p>
        </Card>
      )}
      <div className="min-h-0 flex-1">
        <TimelineCanvas
          peaks={wave.peaks}
          durationS={wave.duration_s}
          blocks={markers as TimelineBlock[]}
          view={view}
          currentS={currentAudioS}
          onViewChange={setView}
          onSeek={seekAudio}
          onSelect={(id) => selectMute(id, { play: true })}
          onResize={(id, aStart, aEnd) => adjustMuteFree(id, toLocal(aStart), toLocal(aEnd))}
          onScrub={scrubAtAudio}
        />
      </div>

      {/* Инспектор выбранного мата */}
      {selected && (
        <MuteInspector
          index={selectedIndex + 1}
          total={mutes.length}
          start={selected.start}
          end={selected.end}
          word={selected.word}
          muted={selected.status === SILENCED}
          review={selected.review}
          playing={playingMuteId === selected.id}
          contextWords={selectedContext}
          currentS={currentLocalS}
          onToggleMuted={() => toggleMuted(selected)}
          onDelete={() => deleteMute(selected.id)}
          onPlayExport={() => playMute(selected)}
          onPlayOriginal={() => {
            setPlayingMuteId(selected.id);
            const win = playWindow(wordIndexRef.current, selected.start, selected.end, 1, 2);
            playAudioRange(toAudio(win.start), toAudio(win.end), 0.06);
          }}
          onAdjust={(s, e) => adjustMute(selected.id, s, e)}
          onPlayRange={(s, e) => {
            setPlayingMuteId(selected.id);
            playAudioRange(toAudio(s), toAudio(e), 0.06);
          }}
          onNudge={(ds, de) => nudgeMute(selected.id, ds, de)}
        />
      )}

      {/* Hidden audio */}
      <audio
        ref={audioRef}
        src={audioSrc}
        preload="auto"
        className="hidden"
        onTimeUpdate={onAudioTime}
        onPause={onAudioPause}
      />

      <div className="flex items-center justify-between gap-2 pr-1 text-[11px] text-subtle">
        <span>
          ←→/↑↓ — между матами · колесо — зум · Shift+колесо — панорама · Space — прослушать · M/Enter — глушить/оставить · Del — удалить
        </span>
        {store.dirty && (
          <span>
            Не сохранено. {platformizeShortcut("⌘S")} — сохранить.
          </span>
        )}
      </div>

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

function round3(n: number): number {
  return Math.round(n * 1000) / 1000;
}

function extractProjectName(decisionsPath: string): string {
  const sep = decisionsPath.includes("\\") ? "\\" : "/";
  const parts = decisionsPath.split(sep).filter(Boolean);
  if (parts.length >= 2) return parts[parts.length - 2];
  return decisionsPath;
}
