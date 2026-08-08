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
  Scissors,
  Settings2,
  Trash2,
  VolumeX,
  Wand2,
  ZoomIn,
  ZoomOut,
} from "lucide-react";
import { Card } from "../components/ui/Card";
import { cn } from "../lib/cn";
import { EmptyState } from "../components/ui/EmptyState";
import { Button } from "../components/ui/Button";
import { Kbd } from "../components/ui/Kbd";
import { OverviewStrip, type OverviewMarker } from "../components/timeline/OverviewStrip";
import {
  TimelineCanvas,
  type TimelineView,
} from "../components/timeline/TimelineCanvas";
import { MuteInspector } from "../components/timeline/MuteInspector";
import { SilenceOverlay, type SilenceBlock } from "../components/timeline/SilenceOverlay";
import { useUndoable, sliceUndoable } from "../hooks/useUndoable";
import { useHotkey } from "../hooks/useHotkey";
import { useTimelineCoords } from "../hooks/useTimelineCoords";
import { useTimelinePlayback } from "../hooks/useTimelinePlayback";
import { usePublishTimelineActions } from "../lib/timelineActions";
import { toast } from "../components/ui/Toast";
import { platformizeShortcut } from "../lib/platform";
import { fmtMs } from "../lib/format";
import {
  buildWordIndex,
  contextWords,
  playWindow,
  wordsInRange,
  type WordIndex,
} from "../lib/transcript";
import {
  allowMediaPaths,
  createDetectSilenceJob,
  getWaveform,
  mediaUrl,
  readDecisions,
  readTranscript,
  writeDecisions,
  type DetectSilenceResult,
  type JobState,
  type WaveformData,
} from "../api";
import { useJobEvents } from "../hooks/useJobEvents";
import type { DecisionsDoc, MuteRecord, TranscriptDoc } from "../types/project";
import { useMuteEditing } from "../hooks/useMuteEditing";
import { adaptiveTrim, useSilenceEditing, type SilenceSegment } from "../hooks/useSilenceEditing";
import { SILENCED, toUiMute, type UiMute } from "../lib/mutes";
import type { SilenceRecord } from "../types/project";

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

const VIEW_MIN_SPAN = 0.4; // минимальная ширина окна зума таймлайна, сек
const ZOOM_IN = 1 / 1.6; // множитель span при «приблизить»
const ZOOM_OUT = 1.6; // множитель span при «отдалить»
// Стабильная пустая ссылка: в режиме «Тишина» на волну не кладём редактируемые
// блоки матов, но проп у мемо-канваса должен сохранять идентичность между
// рендерами, иначе TimelineCanvas реконсилится зря.
const EMPTY_BLOCKS: OverviewMarker[] = [];

/** Комбинированное undoable-состояние экрана: заглушки + тишина в одной истории. */
interface EditState {
  mutes: UiMute[];
  silences: SilenceSegment[];
}

export function TimelineScreen({ decisionsPath }: { decisionsPath: string }) {
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [wave, setWave] = useState<WaveformData | null>(null);
  const [projectName, setProjectName] = useState("");

  // Источник аудио + система координат.
  const [audioPath, setAudioPath] = useState<string | null>(null);
  const coords = useTimelineCoords();
  const { toAudio, toLocal, offsetRef } = coords;
  const docRef = useRef<DecisionsDoc | null>(null);
  const wordIndexRef = useRef<WordIndex>({ words: [] });

  // Undoable-стор всего экрана: mutes + silences в ОДНОЙ истории — ⌘Z/⌘⇧Z
  // ходят по единому стеку поверх обоих списков. Редакторы работают со срезами
  // через sliceUndoable (см. muteStore/silenceStore ниже).
  const store = useUndoable<EditState>({ mutes: [], silences: [] });
  const muteStore = sliceUndoable(store, "mutes");
  const silenceStore = sliceUndoable(store, "silences");
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [selectedSilenceId, setSelectedSilenceId] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);
  // Режим «Правки»: маты или тишина. Каждый режим показывает только свой
  // инструментарий (тулбар, блоки на волне, инспектор) — вдвое меньше шума.
  const [mode, setMode] = useState<"mutes" | "silence">("mutes");
  // Авто-детект тишины: id job'а бэка. null = не запускали / завершилось.
  const [silenceJobId, setSilenceJobId] = useState<string | null>(null);
  // Адаптивность: сколько «дыхания» тишины оставлять (сек). Паузы короче —
  // не трогаем; длиннее — режем только излишек сверх maxPause. 0 = под чистую.
  const [maxPause, setMaxPause] = useState(2);
  // Порог тишины (dBFS): всё тише = тишина. Ниже (−40) → строже, тихие хвосты
  // речи считаются звуком; выше (−25) → агрессивнее. −38 — консервативный дефолт.
  const [thresholdDb, setThresholdDb] = useState(-38);
  // Параметры поиска (пауза/порог) свёрнуты за шестерёнкой — по умолчанию скрыты,
  // чтобы основной тулбар тишины оставался одной кнопкой.
  const [showSilenceParams, setShowSilenceParams] = useState(false);
  // Окно зума главного таймлайна (АУДИО-секунды). null до загрузки волны.
  const [view, setView] = useState<TimelineView | null>(null);

  // Аудио-плейбек: скрытый <audio>, точная остановка, реалтайм-глушение, rAF.
  // Заглушённые участки для превью «как в экспорте» отдаём в audio-координатах.
  const playback = useTimelinePlayback(
    useCallback(
      () =>
        muteStore.present
          .filter((m) => m.status === SILENCED)
          .map((m) => [toAudio(m.start), toAudio(m.end)] as [number, number]),
      [muteStore, toAudio]
    ),
    // Вырезаемые при рендере участки (accepted silence, audio-сек): непрерывное
    // превью (пробел) перепрыгивает их, чтобы звучало как финальный рендер.
    useCallback(
      () =>
        silenceStore.present
          .filter((s) => s.status === "accepted")
          .map((s) => [s.start, s.end] as [number, number]),
      [silenceStore]
    )
  );
  const {
    audioRef,
    currentAudioS,
    currentAudioRef,
    playingMuteId,
    setPlayingMuteId,
    stopPlayback,
    playAudioRange,
    playFrom,
    seekAudio,
    audioProps,
  } = playback;

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

        // Silence-сегменты (audio-секунды) — уже отревьюенные из прошлой сессии.
        const rawSilences = Array.isArray(doc.silences) ? doc.silences : [];
        const uiSilences: SilenceSegment[] = rawSilences
          .filter(
            (s) => Number.isFinite(s.start) && Number.isFinite(s.end) && s.end > s.start
          )
          .map((s) => ({
            id: String(s.id ?? `silence_${Math.round(Number(s.start) * 1000)}`),
            start: Number(s.start),
            end: Number(s.end),
            status: (s.status === "rejected" ? "rejected" : "accepted") as SilenceSegment["status"],
          }))
          .sort((a, b) => a.start - b.start);

        // Offset (range_in в секундах) — из первого мьюта, где есть stream_start.
        const withStream = rawMutes.find(
          (m) => Number.isFinite(m.stream_start) && Number.isFinite(m.start)
        );
        const offset = withStream
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
        coords.configure({ localCoords: !!cacheAudio, offset });
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
        store.reset({ mutes: ui, silences: uiSilences });
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

  const mutes = muteStore.present;
  const silences = silenceStore.present;

  // --- производные ----------------------------------------------------------
  const silencedCount = useMemo(
    () => mutes.filter((m) => m.status === SILENCED).length,
    [mutes]
  );

  // Сколько матов ещё не проверено (review) — единственный «действенный» счётчик
  // в заголовке. Раскладка по статусам (заглушим/оставим/тишина) — в правой
  // сводке тулбара, счётчик списка — в табе режима.
  const reviewCount = useMemo(
    () => mutes.filter((m) => m.review).length,
    [mutes]
  );

  const markers = useMemo<OverviewMarker[]>(
    () =>
      mutes.map((m) => ({
        id: m.id,
        start: toAudio(m.start),
        end: toAudio(m.end),
        // Три состояния маркера: silence=красный (заглушим), keep=серый (оставим),
        // review=янтарный (не проверено). Данные уже трёхстатусные — чисто отрисовка.
        state: m.review ? "review" : m.status === SILENCED ? "silence" : "keep",
        selected: m.id === selectedId,
      })),
    [mutes, selectedId, toAudio]
  );

  // Silence-блоки для оверлея. Координаты уже audio-секунды (как хранит
  // SilenceSegment и ждёт SilenceOverlay) — конвертация не нужна. active =
  // «будет вырезан» (accepted).
  const silenceBlocks = useMemo<SilenceBlock[]>(
    () =>
      silences.map((s) => ({
        id: s.id,
        start: s.start,
        end: s.end,
        active: s.status === "accepted",
        selected: s.id === selectedSilenceId,
      })),
    [silences, selectedSilenceId]
  );

  const acceptedSilenceCount = useMemo(
    () => silences.filter((s) => s.status === "accepted").length,
    [silences]
  );

  const selectedSilence = useMemo(
    () => silences.find((s) => s.id === selectedSilenceId) ?? null,
    [silences, selectedSilenceId]
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
  // stopPlayback / playAudioRange / seekAudio / currentAudioS и обработчики
  // <audio> живут в useTimelinePlayback (см. выше). Здесь остаются только
  // callbacks, которым нужны транскрипт/store: playMute и scrubAtAudio.
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
      const m = muteStore.present.find((x) => x.id === id);
      setSelectedId(id);
      setSelectedSilenceId(null); // выбор мата снимает выбор тишины — инспектор один
      if (!m) return;
      if (opts?.center) centerViewOn(toAudio(m.start), toAudio(m.end));
      if (opts?.play) playMute(m);
    },
    [muteStore, centerViewOn, toAudio, playMute]
  );

  /** Выбрать тишину: подсветить, снять выбор мата (инспектор один на двоих). */
  const selectSilence = useCallback((id: string) => {
    setSelectedSilenceId(id);
    setSelectedId(null);
  }, []);

  /** Сменить режим (маты/тишина) — снимаем выбор, чтобы чужой инспектор не висел. */
  const switchMode = useCallback((next: "mutes" | "silence") => {
    setMode(next);
    setSelectedId(null);
    setSelectedSilenceId(null);
  }, []);

  /** Прыжок к пред/след мату по времени (список отсортирован по start). */
  const goToMute = useCallback(
    (dir: 1 | -1) => {
      const list = muteStore.present;
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
    [muteStore, selectedId, selectMute]
  );

  // --- редактирование мьютов ------------------------------------------------
  // Чистые правки границ/статуса/слова живут в useMuteEditing. UI-побочки
  // (выделение, центрирование, тосты, добавление/удаление, сохранение) —
  // остаются здесь, потому что дёргают view/selection/doc.
  const { toggleMuted, adjustMute, nudgeMute, adjustMuteFree } = useMuteEditing(
    muteStore,
    wordIndexRef
  );

  // Правки silence-сегментов — тот же undoable-стор (общая история с мьютами).
  const silenceEdit = useSilenceEditing(silenceStore);

  // Авто-детект тишины прямо в «Правке»: гоним ffmpeg-silencedetect по ТОМУ ЖЕ
  // audioPath, что играет таймлайн. Поэтому сегменты приходят уже в audio-
  // секундах (0-based для извлечённого cache-audio) — ровно как ждёт оверлей,
  // конвертация не нужна. Результат кладём в silenceStore (общая undo-история:
  // ⌘Z уберёт весь набор). keepManual — не затираем ранее правленные вручную.
  const silenceEditRef = useRef(silenceEdit);
  silenceEditRef.current = silenceEdit;
  // maxPause читаем через ref — onFinal мемоизирован с [], а значение может
  // смениться между запуском job'а и его завершением.
  const maxPauseRef = useRef(maxPause);
  maxPauseRef.current = maxPause;
  const silenceJob = useJobEvents(
    silenceJobId,
    useCallback((state: JobState) => {
      if (state.status === "done" && state.result) {
        const r = state.result as unknown as DetectSilenceResult;
        // Адаптив: из каждой найденной паузы вырезаем только излишек сверх
        // maxPause (короткие паузы не трогаем вовсе). Так речь не звучит
        // «пулемётом» — остаётся естественное дыхание.
        const trimmed = adaptiveTrim(
          r.segments.map(([start, end]) => ({ start, end })),
          maxPauseRef.current
        );
        silenceEditRef.current.replaceAll(
          trimmed.map(({ start, end }) => ({ start, end, status: "accepted" as const })),
          { keepManual: true }
        );
        setSilenceJobId(null);
        toast.success(
          trimmed.length > 0
            ? `Тишина: ${trimmed.length} участков к вырезу (пауза >${maxPauseRef.current}с) — проверьте и сохраните`
            : `Длинных пауз (>${maxPauseRef.current}с) не найдено`
        );
      } else if (state.status === "failed") {
        setSilenceJobId(null);
        toast.error("Детект тишины упал", { description: state.error ?? undefined });
      }
    }, [])
  );
  const detectingSilence =
    silenceJob.state?.status === "running" || silenceJob.state?.status === "pending";

  /** Запустить авто-детект тишины по текущему аудио таймлайна. */
  const runDetectSilence = useCallback(async () => {
    if (!audioPath || detectingSilence) return;
    try {
      // min_duration ловим паузы уже от maxPause (короче — резать нечего, их
      // отсеет и adaptiveTrim). padding=0 и merge_gap=0: адаптивную обрезку и
      // «дыхание» считаем на клиенте по сырым границам, чтобы не сдвигать край.
      const job = await createDetectSilenceJob({
        video_path: audioPath,
        threshold_db: thresholdDb,
        min_duration: Math.max(0.3, maxPause + 0.05),
        merge_gap: 0,
        padding_before: 0,
        padding_after: 0,
      });
      setSilenceJobId(job.id);
      toast("Ищу тишину…", { description: `паузы длиннее ${maxPause}с · порог ${thresholdDb}dB` });
    } catch (e) {
      toast.error("Не удалось запустить детект тишины", { description: String(e) });
    }
  }, [audioPath, detectingSilence, maxPause, thresholdDb]);

  const deleteMute = useCallback(
    (id: string) => {
      muteStore.set((prev) => prev.filter((m) => m.id !== id));
      if (selectedId === id) setSelectedId(null);
      toast.success("Заглушка удалена", {
        description: `${platformizeShortcut("⌘Z")} чтобы вернуть`,
      });
    },
    [muteStore, selectedId]
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
    muteStore.set((prev) =>
      [...prev, created].sort((a, b) => a.start - b.start)
    );
    setSelectedId(id);
    setSelectedSilenceId(null);
    centerViewOn(toAudio(start), toAudio(end));
    toast.success("Заглушка добавлена", { description: fmtMs(start * 1000) });
  }, [currentLocalS, muteStore, centerViewOn, toAudio]);

  // --- сохранение -----------------------------------------------------------
  const save = useCallback(async () => {
    if (saving || !store.dirty) return;
    const doc = docRef.current ?? {};
    const off = offsetRef.current;
    // Собираем mutes обратно: правим start/end/word/status, пересчитываем
    // stream_start/stream_end (их читает Vegas), сохраняем прочие поля.
    const nextMutes: MuteRecord[] = muteStore.present.map((m) => {
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
    // Silence-сегменты — audio-секунды, отсортированы; их читает ExportScreen
    // (не пере-детектит) и рендер вырезает `accepted`.
    const nextSilences: SilenceRecord[] = silenceStore.present
      .slice()
      .sort((a, b) => a.start - b.start)
      .map((s) => ({
        id: s.id,
        start: round3(s.start),
        end: round3(s.end),
        status: s.status,
      }));
    const nextDoc: DecisionsDoc = { ...doc, mutes: nextMutes, silences: nextSilences };
    setSaving(true);
    try {
      await writeDecisions(decisionsPath, nextDoc);
      docRef.current = nextDoc;
      store.mark();
      toast.success("Сохранено", {
        description: `Заглушим ${muteStore.present.filter((m) => m.status === SILENCED).length} из ${muteStore.present.length}`,
      });
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      setError(msg);
      toast.error("Не удалось сохранить", { description: msg });
    } finally {
      setSaving(false);
    }
  }, [saving, store, muteStore, silenceStore, decisionsPath]);

  // --- навигация по матам клавишами -----------------------------------------
  useHotkey(["arrowdown", "arrowright"], (e) => {
    e.preventDefault();
    goToMute(1);
  });
  useHotkey(["arrowup", "arrowleft"], (e) => {
    e.preventDefault();
    goToMute(-1);
  });
  // Пробел = toggle: нажал — играет НЕПРЕРЫВНО от текущего плейхеда со сквозным
  // глушением «как в экспорте» до конца дорожки; нажал ещё раз — стоп.
  // Пер-момент playMute (клик по строке/блоку) не трогаем — он живёт отдельно.
  useEffect(() => {
    const isEditable = (el: EventTarget | null): boolean => {
      if (!(el instanceof HTMLElement)) return false;
      const tag = el.tagName;
      return (
        tag === "INPUT" ||
        tag === "TEXTAREA" ||
        tag === "SELECT" ||
        el.isContentEditable
      );
    };
    // Пробел — переключатель (toggle): нажал — играет от плейхеда до конца
    // дорожки со сквозным глушением «как в экспорте»; нажал ещё раз (или дошло
    // до конца) — стоп. Пер-момент playMute (клик по строке/блоку) не трогаем.
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key !== " " && e.code !== "Space") return;
      if (isEditable(e.target)) return;
      if (e.metaKey || e.ctrlKey || e.altKey) return;
      e.preventDefault();
      if (e.repeat) return; // автоповтор при удержании — игнорируем
      const el = audioRef.current;
      const playing = !!el && !el.paused;
      if (playing) {
        stopPlayback();
      } else {
        // Играем от живого положения плейхеда до конца дорожки.
        playFrom(currentAudioRef.current, true);
      }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => {
      window.removeEventListener("keydown", onKeyDown);
    };
  }, [playFrom, stopPlayback, currentAudioRef, audioRef]);
  useHotkey(["m", "enter"], () => {
    // Тишина в фокусе — переключаем её; иначе мат.
    if (selectedSilenceId) {
      silenceEdit.toggleStatus(selectedSilenceId);
      return;
    }
    const m = mutes.find((x) => x.id === selectedId);
    if (m) toggleMuted(m);
  });
  useHotkey("delete", () => {
    if (selectedSilenceId) {
      silenceEdit.deleteSegment(selectedSilenceId);
      setSelectedSilenceId(null);
      return;
    }
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
  useHotkey("escape", () => {
    setSelectedId(null);
    setSelectedSilenceId(null);
  });

  usePublishTimelineActions({
    onSave: save,
    onUndo: store.undo,
    onRedo: store.redo,
    canUndo: store.canUndo,
    canRedo: store.canRedo,
    dirty: store.dirty,
  });

  // Стабильные обработчики для мемоизированных TimelineCanvas/OverviewStrip:
  // инлайн-стрелки в JSX меняли бы идентичность на каждый throttle-тик таймкода
  // и сводили бы memo на нет. onResize/onSelect/onPickMarker дёргают свежие
  // selectMute/adjustMuteFree через ref — сами остаются стабильными.
  const selectMuteRef = useRef(selectMute);
  selectMuteRef.current = selectMute;
  const adjustFreeRef = useRef(adjustMuteFree);
  adjustFreeRef.current = adjustMuteFree;
  const toLocalRef = useRef(toLocal);
  toLocalRef.current = toLocal;

  const handleSelectPlay = useCallback((id: string) => {
    selectMuteRef.current(id, { play: true });
  }, []);
  const handlePickMarker = useCallback((id: string) => {
    selectMuteRef.current(id, { center: true });
  }, []);
  const handleResize = useCallback((id: string, aStart: number, aEnd: number) => {
    adjustFreeRef.current(id, toLocalRef.current(aStart), toLocalRef.current(aEnd));
  }, []);

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
            {reviewCount > 0 ? (
              <span className="text-amber-400">осталось {reviewCount} проверить</span>
            ) : (
              <span className="text-fg">всё проверено</span>
            )}
          </p>
        </div>
        <div className="flex items-center gap-2">
          <div className="flex items-center gap-1 rounded-md border border-white/5 bg-black/30 p-1">
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
        currentRef={currentAudioRef}
        view={view}
        onSeek={seekAudio}
        onPickMarker={handlePickMarker}
      />

      {/* Переключатель режима: маты / тишина — каждый показывает свой тулбар */}
      <div className="flex items-center gap-1 self-start rounded-lg border border-white/10 bg-white/5 p-1">
        <button
          type="button"
          onClick={() => switchMode("mutes")}
          className={cn(
            "inline-flex h-8 items-center gap-2 rounded-md border px-3 text-xs font-medium transition-colors",
            mode === "mutes"
              ? "border-white/15 bg-surface text-fg shadow-[0_1px_0_rgb(255_255_255/0.05)_inset]"
              : "border-transparent text-muted hover:text-fg"
          )}
        >
          <VolumeX className="h-3.5 w-3.5" /> Маты
          <span className="rounded bg-white/8 px-1.5 py-px font-mono text-[10px] tabular-nums text-subtle">
            {mutes.length}
          </span>
        </button>
        <button
          type="button"
          onClick={() => switchMode("silence")}
          className={cn(
            "inline-flex h-8 items-center gap-2 rounded-md border px-3 text-xs font-medium transition-colors",
            mode === "silence"
              ? "border-white/15 bg-surface text-fg shadow-[0_1px_0_rgb(255_255_255/0.05)_inset]"
              : "border-transparent text-muted hover:text-fg"
          )}
        >
          <Scissors className="h-3.5 w-3.5" /> Тишина
          {silences.length > 0 && (
            <span className="rounded bg-white/8 px-1.5 py-px font-mono text-[10px] tabular-nums text-subtle">
              {silences.length}
            </span>
          )}
        </button>
      </div>

      {/* Тулбар режима «Маты»: переход по матам · зум · добавить · сводка */}
      {mode === "mutes" && (
        <div className="flex flex-wrap items-center gap-2">
          <div className="flex items-center gap-1 rounded-md border border-white/5 bg-black/30 p-1">
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

          <div className="flex items-center gap-1 rounded-md border border-white/5 bg-black/30 p-1">
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

          <span className="ml-auto flex items-center gap-1.5 text-xs text-subtle">
            {reviewCount > 0 && (
              <>
                <span className="text-amber-400">{reviewCount} проверить</span>
                <span className="mx-1 text-white/15">·</span>
              </>
            )}
            <VolumeX className="h-3.5 w-3.5 text-err" /> {silencedCount} заглушим
            <span className="mx-1 text-white/15">·</span>
            {mutes.length - silencedCount} оставим
          </span>

          <span className="font-mono text-[11px] tabular-nums text-subtle">
            {fmtMs(currentAudioS * 1000)} / {fmtMs(wave.duration_s * 1000)}
          </span>
        </div>
      )}

      {/* Тулбар режима «Тишина»: найти тишину · (шестерёнка) параметры · сводка */}
      {mode === "silence" && (
        <div className="flex flex-wrap items-center gap-2">
          <Button
            size="sm"
            variant="secondary"
            onClick={runDetectSilence}
            disabled={detectingSilence}
            loading={detectingSilence}
            aria-label="Найти тишину автоматически"
          >
            <Wand2 className="mr-1.5 h-3.5 w-3.5" />
            {detectingSilence ? "ищу тишину…" : "найти тишину"}
          </Button>

          <Button
            size="sm"
            variant={showSilenceParams ? "secondary" : "ghost"}
            onClick={() => setShowSilenceParams((v) => !v)}
            className="h-8 px-2"
            aria-label="Параметры поиска тишины"
          >
            <Settings2 className="mr-1.5 h-3.5 w-3.5" />
            пауза ≤{maxPause}с · {thresholdDb}dB
          </Button>

          {/* Параметры (пауза/порог) — по умолчанию свёрнуты за шестерёнкой */}
          {showSilenceParams && (
            <div className="flex items-center gap-1 rounded-md border border-white/5 bg-black/30 p-1">
              <span className="pl-1 text-[11px] text-subtle">пауза ≤</span>
              <Button
                size="sm"
                variant="ghost"
                onClick={() => setMaxPause((v) => Math.max(0, round1(v - 0.5)))}
                disabled={detectingSilence || maxPause <= 0}
                className="h-7 w-7 p-0"
                aria-label="Меньше паузы"
              >
                <ZoomOut className="h-3.5 w-3.5" />
              </Button>
              <span className="min-w-[38px] text-center font-mono text-[11px] tabular-nums text-fg">
                {maxPause}с
              </span>
              <Button
                size="sm"
                variant="ghost"
                onClick={() => setMaxPause((v) => Math.min(10, round1(v + 0.5)))}
                disabled={detectingSilence || maxPause >= 10}
                className="h-7 w-7 p-0"
                aria-label="Больше паузы"
              >
                <ZoomIn className="h-3.5 w-3.5" />
              </Button>
              <span className="pl-1 text-[11px] text-subtle">порог</span>
              <Button
                size="sm"
                variant="ghost"
                onClick={() => setThresholdDb((v) => Math.max(-50, v - 2))}
                disabled={detectingSilence || thresholdDb <= -50}
                className="h-7 w-7 p-0"
                aria-label="Строже (тише порог)"
              >
                <ZoomOut className="h-3.5 w-3.5" />
              </Button>
              <span className="min-w-[46px] text-center font-mono text-[11px] tabular-nums text-fg">
                {thresholdDb}dB
              </span>
              <Button
                size="sm"
                variant="ghost"
                onClick={() => setThresholdDb((v) => Math.min(-15, v + 2))}
                disabled={detectingSilence || thresholdDb >= -15}
                className="h-7 w-7 p-0"
                aria-label="Агрессивнее (громче порог)"
              >
                <ZoomIn className="h-3.5 w-3.5" />
              </Button>
            </div>
          )}

          <span className="ml-auto flex items-center gap-1.5 text-xs text-subtle">
            {silences.length > 0 ? (
              <>
                <Scissors className="h-3.5 w-3.5 text-amber-400" />
                <span className="text-amber-400">{acceptedSilenceCount}</span> вырежем
                <span className="mx-1 text-white/15">·</span>
                {silences.length - acceptedSilenceCount} оставим
              </>
            ) : (
              <span className="text-subtle">тишина не искалась</span>
            )}
          </span>

          <span className="font-mono text-[11px] tabular-nums text-subtle">
            {fmtMs(currentAudioS * 1000)} / {fmtMs(wave.duration_s * 1000)}
          </span>
        </div>
      )}

      {/* Главный таймлайн */}
      {mode === "mutes" && mutes.length === 0 && (
        <Card variant="surface" padding="sm">
          <p className="text-center text-xs text-subtle">
            Заглушек нет — мат не найден или все убраны. «+ мат на плейхеде» добавит вручную.
          </p>
        </Card>
      )}
      {mode === "silence" && silences.length === 0 && (
        <Card variant="surface" padding="sm">
          <p className="text-center text-xs text-subtle">
            Тишина ещё не искалась. «найти тишину» подберёт длинные паузы к вырезу.
          </p>
        </Card>
      )}
      <div className="relative min-h-0 flex-1">
        <TimelineCanvas
          peaks={wave.peaks}
          durationS={wave.duration_s}
          blocks={mode === "mutes" ? markers : EMPTY_BLOCKS}
          view={view}
          currentRef={currentAudioRef}
          onViewChange={setView}
          onSeek={seekAudio}
          onSelect={handleSelectPlay}
          onResize={handleResize}
          onScrub={scrubAtAudio}
        />
        {mode === "silence" && silenceBlocks.length > 0 && (
          <SilenceOverlay
            blocks={silenceBlocks}
            durationS={wave.duration_s}
            view={view}
            onSelect={selectSilence}
          />
        )}
      </div>

      {/* Инспектор выбранного мата */}
      {mode === "mutes" && selected && (
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

      {/* Инспектор выбранного участка тишины */}
      {mode === "silence" && selectedSilence && (
        <Card variant="surface" padding="sm">
          <div className="flex flex-wrap items-center gap-3 text-sm">
            <span className="flex items-center gap-1.5 font-medium">
              <Scissors className="h-4 w-4 text-amber-400" /> Тишина
            </span>
            <span className="font-mono text-xs tabular-nums text-subtle">
              {fmtMs(selectedSilence.start * 1000)} – {fmtMs(selectedSilence.end * 1000)}
              <span className="ml-1.5 text-white/40">
                ({(selectedSilence.end - selectedSilence.start).toFixed(2)}с)
              </span>
            </span>
            <span
              className={
                selectedSilence.status === "accepted"
                  ? "rounded bg-amber-400/15 px-1.5 py-0.5 text-[11px] text-amber-400"
                  : "rounded bg-white/8 px-1.5 py-0.5 text-[11px] text-subtle"
              }
            >
              {selectedSilence.status === "accepted" ? "вырежем" : "оставим"}
            </span>

            <div className="ml-auto flex items-center gap-1.5">
              <Button
                size="sm"
                variant="ghost"
                onClick={() => {
                  playAudioRange(selectedSilence.start, selectedSilence.end, 0.06);
                }}
              >
                прослушать
              </Button>
              <Button
                size="sm"
                variant={selectedSilence.status === "accepted" ? "secondary" : "primary"}
                onClick={() => silenceEdit.toggleStatus(selectedSilence.id)}
              >
                {selectedSilence.status === "accepted" ? "оставить" : "вырезать"}
              </Button>
              <Button
                size="sm"
                variant="ghost"
                onClick={() => {
                  silenceEdit.deleteSegment(selectedSilence.id);
                  setSelectedSilenceId(null);
                }}
                aria-label="Удалить участок тишины"
              >
                <Trash2 className="h-3.5 w-3.5" />
              </Button>
            </div>
          </div>
        </Card>
      )}

      {/* Hidden audio */}
      <audio
        ref={audioRef}
        src={audioSrc}
        preload="auto"
        className="hidden"
        onTimeUpdate={audioProps.onTimeUpdate}
        onPause={audioProps.onPause}
        onEnded={audioProps.onEnded}
      />

      <div className="flex items-center justify-between gap-2 pr-1 text-[11px] text-subtle">
        <span>
          ←→/↑↓ — между матами · щипок/колесо — зум · две пальца/Shift+колесо — панорама · Space — прослушать · M/Enter — глушить/оставить (мат) или вырезать/оставить (тишина) · Del — удалить · Esc — снять выбор
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

function round1(n: number): number {
  return Math.round(n * 10) / 10;
}

function extractProjectName(decisionsPath: string): string {
  const sep = decisionsPath.includes("\\") ? "\\" : "/";
  const parts = decisionsPath.split(sep).filter(Boolean);
  if (parts.length >= 2) return parts[parts.length - 2];
  return decisionsPath;
}
