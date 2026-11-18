import { PreviewDeck } from "../components/timeline/PreviewDeck";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  AlertCircle,
  Bot,
  ChevronDown,
  ChevronLeft,
  ChevronRight,
  FolderOpen,
  Loader2,
  Maximize2,
  MessageSquareOff,
  Plus,
  RotateCcw,
  RotateCw,
  Save,
  Scissors,
  Settings2,
  Sparkles,
  Trash2,
  VolumeX,
  Wand2,
  ZoomIn,
  ZoomOut,
} from "lucide-react";
import { ReviewList } from "../components/timeline/ReviewList";
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
import { OfftopicOverlay, type OfftopicBlock } from "../components/timeline/OfftopicOverlay";
import { ProgressBar } from "../components/ui/ProgressBar";
import { stageLabel } from "../components/ui/Stepper";
import { useUndoable, sliceUndoable } from "../hooks/useUndoable";
import { useHotkey } from "../hooks/useHotkey";
import { useTimelineCoords } from "../hooks/useTimelineCoords";
import { useTimelinePlayback } from "../hooks/useTimelinePlayback";
import { useWaveformTile } from "../hooks/useWaveformTile";
import type { WaveTile } from "../components/timeline/waveTile";
import { usePublishTimelineActions } from "../lib/timelineActions";
import { toast } from "../components/ui/Toast";
import { platformizeShortcut } from "../lib/platform";
import { fmtMs, parseTimecode } from "../lib/format";
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
  createDetectOfftopicJob,
  getAiProviders,
  getSettings,
  getVodInfo,
  getWaveform,
  listJobs,
  mediaUrl,
  readDecisions,
  readTranscript,
  writeDecisions,
  type AiProvider,
  type DetectSilenceResult,
  type JobState,
  type WaveformData,
} from "../api";
import { useJobEvents } from "../hooks/useJobEvents";
import { twitchVodUrl } from "../lib/twitch";
import type { DecisionsDoc, MuteRecord, TranscriptDoc } from "../types/project";
import type { OfftopicSegment as OfftopicSegmentDoc } from "../types/project";
import { useMuteEditing } from "../hooks/useMuteEditing";
import { adaptiveTrim, useSilenceEditing, type SilenceSegment } from "../hooks/useSilenceEditing";
import { useOfftopicEditing, type OfftopicSegmentUi } from "../hooks/useOfftopicEditing";
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

// Стабильный фолбэк окна, пока view ещё null (волна не загрузилась). Отдельная
// константа, а не литерал в JSX: иначе новый объект каждый рендер триггерил бы
// эффект догрузки крошки волны (см. useWaveformTile).
const EMPTY_VIEW = { start: 0, end: 0 };

// Пустой тайл волны на время до загрузки (рендер доходит до канваса только когда
// wave уже есть, но тип требует не-null; заодно стабильная ссылка для мемо).
const EMPTY_TILE: WaveTile = {
  start: 0,
  end: 1,
  min: new Float32Array(0),
  max: new Float32Array(0),
  rms: null,
};

// Форматы файла-исходника для детекта оффтопа (видео, на которое идёт реакция).
// Зеркалит NewJobScreen.VIDEO_EXT + аудио-контейнеры (исходник может быть звуком).
const SOURCE_VIDEO_EXT = [
  "mp4",
  "mkv",
  "mov",
  "ts",
  "avi",
  "webm",
  "mp3",
  "wav",
  "m4a",
];

/** Короткое имя файла из полного пути (для отображения выбранного исходника). */
function baseName(p: string): string {
  const parts = p.replace(/\\/g, "/").split("/");
  return parts[parts.length - 1] || p;
}

/** Комбинированное undoable-состояние экрана: заглушки + тишина + оффтоп в одной истории. */
interface EditState {
  mutes: UiMute[];
  silences: SilenceSegment[];
  offtopic: OfftopicSegmentUi[];
}

/** Слои таймлайна: какие оверлеи видимы. Все по умолчанию включены. */
interface LayerVisibility {
  mutes: boolean;
  silence: boolean;
  offtopic: boolean;
}

/**
 * Активный слой редактирования — определяет, ЧЕЙ инспектор показать при клике и
 * какой тулбар над волной. Видимость (LayerVisibility) и активность независимы:
 * можно смотреть все три слоя разом, но править (клик→инспектор, добавление,
 * детект) — только активный.
 */
type ActiveLayer = "mutes" | "silence" | "offtopic";

/**
 * Пер-проектная память UI-слоёв «Правки»: активный слой + видимость оверлеев.
 * TimelineScreen размонтируется при уходе на другой экран (App рендерит его
 * условно `screen === "timeline"`), из-за чего локальный стейт слоёв сбрасывался
 * — вернувшись, пользователь всегда видел «Маты». Держим состояние в модульном
 * сторе с ключом decisionsPath: возврат в ТОТ ЖЕ проект восстанавливает ту же
 * вкладку и те же чекбоксы; другой проект стартует с дефолтов. Живёт до
 * перезапуска приложения (память процесса, не диск) — этого достаточно.
 */
interface LayerUiMemory {
  activeLayer: ActiveLayer;
  layers: LayerVisibility;
  // Зум/скролл таймлайна (audio-секунды). Возврат в тот же проект восстанавливает
  // ту же позицию — раньше сбрасывался в «весь стрим» при каждом заходе на экран.
  view?: TimelineView | null;
}
const layerUiMemory = new Map<string, LayerUiMemory>();
const DEFAULT_LAYERS: LayerVisibility = { mutes: true, silence: true, offtopic: true };

export function TimelineScreen({ decisionsPath }: { decisionsPath: string }) {
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [wave, setWave] = useState<WaveformData | null>(null);
  const [projectName, setProjectName] = useState("");

  // Источник аудио + система координат.
  const [previewSource, setPreviewSource] = useState<string | null>(null);
  const [audioPath, setAudioPath] = useState<string | null>(null);
  const coords = useTimelineCoords();
  const { toAudio, toLocal, offsetRef } = coords;
  const docRef = useRef<DecisionsDoc | null>(null);
  const wordIndexRef = useRef<WordIndex>({ words: [] });

  // Undoable-стор всего экрана: mutes + silences + offtopic в ОДНОЙ истории —
  // ⌘Z/⌘⇧Z ходят по единому стеку поверх всех трёх списков. Редакторы работают
  // со срезами через sliceUndoable (см. muteStore/silenceStore/offtopicStore).
  const store = useUndoable<EditState>({ mutes: [], silences: [], offtopic: [] });
  const muteStore = sliceUndoable(store, "mutes");
  const silenceStore = sliceUndoable(store, "silences");
  const offtopicStore = sliceUndoable(store, "offtopic");
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [selectedSilenceId, setSelectedSilenceId] = useState<string | null>(null);
  const [selectedOfftopicId, setSelectedOfftopicId] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);
  // Слои таймлайна: три оверлея (маты/тишина/оффтоп) видимы одновременно, каждый
  // со своим чекбоксом. `activeLayer` — кого правим (клик→инспектор, тулбар,
  // детект). Раньше был один `mode` (маты|тишина, взаимоисключающие) — заменён на
  // видимость + активность, чтобы видеть все находки разом (см. LayerVisibility).
  const [layers, setLayers] = useState<LayerVisibility>(
    () => layerUiMemory.get(decisionsPath)?.layers ?? DEFAULT_LAYERS
  );
  const [activeLayer, setActiveLayer] = useState<ActiveLayer>(
    () => layerUiMemory.get(decisionsPath)?.activeLayer ?? "mutes"
  );
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
  // AI-детект оффтопа: id job'а (null = не запускали / завершилось), список
  // провайдеров из настроек и выбранный провайдер, флаг открытия поповера.
  const [offtopicJobId, setOfftopicJobId] = useState<string | null>(null);
  const [aiProviders, setAiProviders] = useState<AiProvider[]>([]);
  const [selectedProviderId, setSelectedProviderId] = useState<string>("");
  // Карта api_key_ref → есть ли сохранённый ключ. Считаем один раз при загрузке
  // провайдеров, чтобы в поповере показывать ТОЛЬКО настроенные (с ключом) —
  // а не весь список builtin, где пользователь ничего не вводил.
  const [keyPresence, setKeyPresence] = useState<Record<string, boolean>>({});
  // Есть ли сохранённый ключ у выбранного провайдера (safeStorage, main-side).
  // null = ещё не проверяли; false/true — результат aiKeyHas.
  const [providerHasKey, setProviderHasKey] = useState<boolean | null>(null);
  const [showDetectPopover, setShowDetectPopover] = useState(false);
  // Параметры детекта, задаваемые в поповере (per-run, не персистятся).
  // sourcePath — локальный файл исходника (видео, на которое идёт реакция):
  // задан → путь по исходнику (точнее), пусто → старый оконный режим.
  // detectModel — LLM для source/single-pass; cheap/expensive — каскад windows.
  const [sourcePath, setSourcePath] = useState<string | null>(null);
  const [detectModel, setDetectModel] = useState<string>("");
  const [showDetectAdvanced, setShowDetectAdvanced] = useState(false);
  const [cheapModel, setCheapModel] = useState<string>("");
  const [expensiveModel, setExpensiveModel] = useState<string>("");
  // Порог автопринятия: сегменты с confidence ниже него грузятся как «оставим»
  // (rejected), а не «вырежем». Живёт только в сессии (ref — чтобы async
  // onFinal читал свежее значение без stale-замыкания).
  const [autoAcceptConf, setAutoAcceptConf] = useState(0.7);
  const autoAcceptConfRef = useRef(autoAcceptConf);
  autoAcceptConfRef.current = autoAcceptConf;
  // Окно зума главного таймлайна (АУДИО-секунды). Инициализируем из памяти
  // лениво: возврат в тот же проект восстанавливает прошлый зум/скролл. null —
  // только при первом заходе (до загрузки волны фолбэк ставится на «весь стрим»).
  const [view, setView] = useState<TimelineView | null>(
    () => layerUiMemory.get(decisionsPath)?.view ?? null
  );
  // Запоминаем активный слой + видимость + зум/скролл на случай ухода с экрана и
  // возврата (см. layerUiMemory). Пишем на каждое изменение — стор дешёвый (Map
  // в памяти). view лениво инициализирован выше, поэтому первый прогон эффекта не
  // затирает сохранённый зум стартовым null.
  useEffect(() => {
    layerUiMemory.set(decisionsPath, { activeLayer, layers, view });
  }, [decisionsPath, activeLayer, layers, view]);

  // Ширина канваса в CSS-px — определяет, СКОЛЬКО бакетов просить у бэкенда для
  // видимого окна. Измеряем ResizeObserver'ом; обновляем только на заметное
  // изменение (>8 px), иначе дробный ресайз окна плодил бы фетчи волны.
  const waveBoxRef = useRef<HTMLDivElement | null>(null);
  const [canvasW, setCanvasW] = useState(1200);
  const waveReady = !!wave && !!audioPath && !!view;
  useEffect(() => {
    const el = waveBoxRef.current;
    if (!el) return;
    const measure = () => {
      const w = Math.round(el.getBoundingClientRect().width);
      if (w > 0) setCanvasW((prev) => (Math.abs(prev - w) > 8 ? w : prev));
    };
    measure();
    const ro = new ResizeObserver(measure);
    ro.observe(el);
    return () => ro.disconnect();
  }, [waveReady]);

  // Крошка волны под текущий зум (см. useWaveformTile). Базовые 8192 точки на
  // весь стрим при зуме в 5 сек дают ~23 точки — гладкий блоб; хук дозапрашивает
  // именно видимое окно, поэтому форма остаётся чёткой на любом масштабе.
  const {
    base: waveBase,
    tile: waveTile,
    norm: waveNorm,
  } = useWaveformTile({
    mediaPath: audioPath,
    wave,
    view: view ?? EMPTY_VIEW,
    canvasWidth: canvasW,
  });

  // Аудио-плейбек: скрытый <audio>, точная остановка, реалтайм-глушение, rAF.
  // Заглушённые участки для превью «как в экспорте» отдаём в audio-координатах.
  // Список ЖИВОЙ: пересобирается на правке мьютов (muteStore.present меняет
  // identity только при реальном изменении), и гейт слышит новую границу сразу —
  // без этого правка границы во время проигрывания не была слышна до перезапуска.
  const muteRegions = useMemo<Array<[number, number]>>(
    () =>
      muteStore.present
        .filter((m) => m.status === SILENCED)
        .map((m) => [toAudio(m.start), toAudio(m.end)] as [number, number]),
    [muteStore.present, toAudio]
  );
  const playback = useTimelinePlayback(
    muteRegions,
    // Вырезаемые при рендере участки (accepted silence + accepted offtopic, в
    // audio-сек): непрерывное превью (пробел) перепрыгивает их, чтобы звучало как
    // финальный рендер. Silence уже в audio-сек; offtopic в local — через toAudio.
    useCallback(
      () => {
        const cuts: [number, number][] = silenceStore.present
          .filter((s) => s.status === "accepted")
          .map((s) => [s.start, s.end] as [number, number]);
        for (const o of offtopicStore.present) {
          if (o.status === "accepted") {
            cuts.push([toAudio(o.start), toAudio(o.end)]);
          }
        }
        return cuts;
      },
      [silenceStore, offtopicStore, toAudio]
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
        // Приоритет имени: сохранённый настоящий тайтл стрима → старое _meta.project
        // → имя папки (vod_<id>). title появляется, если включён тумблер
        // переименования и детект оффтопа его вытащил (см. done-handler ниже).
        setProjectName(
          (doc.title ?? "").trim() || doc._meta?.project || extractProjectName(decisionsPath)
        );

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

        // Оффтоп-сегменты (LOCAL-секунды start_s/end_s, как mutes) — из бандла
        // doc.offtopic.segments, записанного AI-детекцией. status по умолчанию
        // accepted (будет вырезан); метаданные классификатора несём для инспектора.
        const uiOfftopic = mapOfftopicSegments(doc.offtopic?.segments);

        // Offset (range_in в секундах) — из первого мьюта, где есть stream_start.
        const withStream = rawMutes.find(
          (m) => Number.isFinite(m.stream_start) && Number.isFinite(m.start)
        );
        const offset = withStream
          ? Number(withStream.stream_start) - Number(withStream.start)
          : typeof doc.range_in === "string" ? (parseTimecode(doc.range_in) ?? 0) / 1000 : 0;

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
        setPreviewSource(source);

        await allowMediaPaths(source ? [media, source] : [media]).catch(() => {
          /* whitelist проверится и на /waveform — не критично */
        });

        // Транскрипт (вариант B) — не блокирующий: если нет, список всё равно работает.
        wordIndexRef.current = { words: [] };
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
        // Восстановленный из памяти зум/скролл сохраняем (подрезав под реальную
        // длину дорожки на случай устаревшего значения); первый заход — весь стрим.
        setView((prev) => {
          if (prev && prev.end > prev.start) {
            const start = Math.max(0, Math.min(prev.start, w.duration_s));
            const end = Math.max(start + 0.001, Math.min(prev.end, w.duration_s));
            return { start, end };
          }
          return { start: 0, end: w.duration_s }; // старт — весь стрим
        });
        store.reset({ mutes: ui, silences: uiSilences, offtopic: uiOfftopic });
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
  const offtopic = offtopicStore.present;

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

  // Диапазоны оффтопа в АУДИО-секундах — пока слой оффтопа видим. Мат/тишина, чей
  // центр попал внутрь такого диапазона, на таймлайне НЕ рисуем: оффтоп — «крышка»
  // на кусок, детали под ней (заглушки/паузы) — визуальный шум. Прячем под ЛЮБЫМ
  // сегментом — и «вырежем» (accepted), и «оставим» (rejected), хоть 32%, хоть 99%:
  // статус/уверенность тут ни при чём, иначе у слабых находок маты/тишина торчат, а
  // у сильных нет. Вернуть — удалить оффтоп или спрятать слой. Данные в сторе не
  // трогаем: скрытие чисто визуальное, save/плейбек/undo видят всё.
  const offtopicCoverRanges = useMemo<[number, number][]>(
    () =>
      layers.offtopic
        ? offtopic.map((o) => [toAudio(o.start), toAudio(o.end)] as [number, number])
        : [],
    [layers.offtopic, offtopic, toAudio]
  );

  const markers = useMemo<OverviewMarker[]>(
    () =>
      mutes
        .map((m): OverviewMarker => ({
          id: m.id,
          start: toAudio(m.start),
          end: toAudio(m.end),
          // Три состояния маркера: silence=красный (заглушим), keep=серый (оставим),
          // review=янтарный (не проверено). Данные уже трёхстатусные — чисто отрисовка.
          state: m.review ? "review" : m.status === SILENCED ? "silence" : "keep",
          selected: m.id === selectedId,
        }))
        // Мат под любым сегментом оффтопа скрываем (см. offtopicCoverRanges).
        .filter((mk) => !centerInsideAny(mk.start, mk.end, offtopicCoverRanges)),
    [mutes, selectedId, toAudio, offtopicCoverRanges]
  );

  // Silence-блоки для оверлея. Координаты уже audio-секунды (как хранит
  // SilenceSegment и ждёт SilenceOverlay) — конвертация не нужна. active =
  // «будет вырезан» (accepted).
  const silenceBlocks = useMemo<SilenceBlock[]>(
    () =>
      silences
        // Тишину под любым сегментом оффтопа скрываем (см. offtopicCoverRanges).
        .filter((s) => !centerInsideAny(s.start, s.end, offtopicCoverRanges))
        .map((s) => ({
          id: s.id,
          start: s.start,
          end: s.end,
          active: s.status === "accepted",
          selected: s.id === selectedSilenceId,
        })),
    [silences, selectedSilenceId, offtopicCoverRanges]
  );

  const acceptedSilenceCount = useMemo(
    () => silences.filter((s) => s.status === "accepted").length,
    [silences]
  );

  const selectedSilence = useMemo(
    () => silences.find((s) => s.id === selectedSilenceId) ?? null,
    [silences, selectedSilenceId]
  );

  // Оффтоп-блоки для оверлея. Координаты сегмента — LOCAL-секунды (как mutes),
  // а оверлей рисует в audio-пространстве → конвертируем через toAudio. active =
  // «будет вырезан» (accepted).
  const offtopicBlocks = useMemo<OfftopicBlock[]>(
    () =>
      offtopic.map((o) => ({
        id: o.id,
        start: toAudio(o.start),
        end: toAudio(o.end),
        active: o.status === "accepted",
        selected: o.id === selectedOfftopicId,
        pauseS: o.pauseS ?? null,
        anchored: o.anchored ?? null,
      })),
    [offtopic, selectedOfftopicId, toAudio]
  );

  const acceptedOfftopicCount = useMemo(
    () => offtopic.filter((o) => o.status === "accepted").length,
    [offtopic]
  );

  const selectedOfftopic = useMemo(
    () => offtopic.find((o) => o.id === selectedOfftopicId) ?? null,
    [offtopic, selectedOfftopicId]
  );

  // Сколько матов/тишины «поглотил» выбранный оффтоп — для инспектора: показываем,
  // что скрытые внутри находки не потеряны и вернутся при удалении/скрытии слоя.
  // Считаем ровно теми же правилами, что прячут (центр внутри диапазона), и только
  // когда реально прячем: слой оффтопа видим (статус не важен — прячем под любым).
  const hiddenInSelectedOfftopic = useMemo(() => {
    if (!selectedOfftopic || !layers.offtopic) return null;
    const rng: [number, number][] = [
      [toAudio(selectedOfftopic.start), toAudio(selectedOfftopic.end)],
    ];
    const muteCount = mutes.filter((m) =>
      centerInsideAny(toAudio(m.start), toAudio(m.end), rng)
    ).length;
    const silenceCount = silences.filter((s) =>
      centerInsideAny(s.start, s.end, rng)
    ).length;
    return muteCount + silenceCount > 0 ? { mutes: muteCount, silences: silenceCount } : null;
  }, [selectedOfftopic, layers.offtopic, mutes, silences, toAudio]);

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
      // Выбор одного слоя снимает выбор остальных — инспектор один на троих.
      setSelectedSilenceId(null);
      setSelectedOfftopicId(null);
      setActiveLayer("mutes");
      if (!m) return;
      if (opts?.center) centerViewOn(toAudio(m.start), toAudio(m.end));
      if (opts?.play) playMute(m);
    },
    [muteStore, centerViewOn, toAudio, playMute]
  );

  /** Выбрать тишину: подсветить, снять выбор мата/оффтопа (инспектор один). */
  const selectSilence = useCallback((id: string) => {
    setSelectedSilenceId(id);
    setSelectedId(null);
    setSelectedOfftopicId(null);
    setActiveLayer("silence");
  }, []);

  /** Выбрать оффтоп-сегмент: подсветить, снять выбор мата/тишины. */
  const selectOfftopic = useCallback((id: string) => {
    setSelectedOfftopicId(id);
    setSelectedId(null);
    setSelectedSilenceId(null);
    setActiveLayer("offtopic");
  }, []);

  /**
   * Переключить активный слой (маты/тишина/оффтоп) — снимаем выбор, чтобы чужой
   * инспектор не висел. Видимость слоёв (layers) при этом не трогаем: активность
   * и видимость независимы (см. LayerVisibility).
   */
  const switchLayer = useCallback((next: ActiveLayer) => {
    setActiveLayer(next);
    setSelectedId(null);
    setSelectedSilenceId(null);
    setSelectedOfftopicId(null);
    // Переключение слоя закрывает открытые панели прошлого слоя.
    setShowDetectPopover(false);
    setShowSilenceParams(false);
  }, []);

  /** Тумблер видимости оверлея (чекбокс слоя). Активность не меняет. */
  const toggleLayerVisible = useCallback((layer: ActiveLayer) => {
    setLayers((prev) => ({ ...prev, [layer]: !prev[layer] }));
  }, []);

  /**
   * Прыжок к пред/след сегменту АКТИВНОГО слоя по времени. Раньше навигация
   * (стрелки + кнопки) всегда ходила по матам, поэтому на вкладке тишины/оффтопа
   * «следующий сегмент» прыгал к мату, а не к сегменту слоя. Теперь список,
   * выбранный id и функция выбора берутся по activeLayer. Каждый список
   * сортируем по своему start (маты/оффтоп — local-секунды, тишина — audio) —
   * внутри слоя это монотонно по времени; центрируем окно на цели.
   */
  const goToSegment = useCallback(
    (dir: 1 | -1) => {
      const step = <T extends { id: string; start: number }>(
        list: readonly T[],
        selId: string | null
      ): T | null => {
        if (!list.length) return null;
        const sorted = [...list].sort((a, b) => a.start - b.start);
        const i = sorted.findIndex((x) => x.id === selId);
        const next =
          i === -1
            ? dir > 0
              ? 0
              : sorted.length - 1
            : Math.max(0, Math.min(sorted.length - 1, i + dir));
        return sorted[next];
      };
      if (activeLayer === "silence") {
        const t = step(silenceStore.present, selectedSilenceId);
        if (!t) return;
        selectSilence(t.id);
        centerViewOn(t.start, t.end); // тишина уже в audio-координатах
        return;
      }
      if (activeLayer === "offtopic") {
        const t = step(offtopicStore.present, selectedOfftopicId);
        if (!t) return;
        selectOfftopic(t.id);
        centerViewOn(toAudio(t.start), toAudio(t.end)); // оффтоп — local-секунды
        return;
      }
      const t = step(muteStore.present, selectedId);
      if (t) selectMute(t.id, { center: true });
    },
    [
      activeLayer,
      muteStore,
      silenceStore,
      offtopicStore,
      selectedId,
      selectedSilenceId,
      selectedOfftopicId,
      selectMute,
      selectSilence,
      selectOfftopic,
      centerViewOn,
      toAudio,
    ]
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

  // Правки оффтоп-сегментов — тот же undoable-стор (общая история). accept/reject/
  // resize/delete/replaceAll ходят по единому ⌘Z-стеку рядом с матами и тишиной.
  const offtopicEdit = useOfftopicEditing(offtopicStore);

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

  // --- AI-детект оффтопа -----------------------------------------------------
  // Провайдеров грузим один раз (builtin + custom из настроек); для каждого с
  // api_key_ref спрашиваем main, есть ли сохранённый ключ. Выбранным ставим
  // первый НАСТРОЕННЫЙ (с ключом или вовсе без ключа). Ключ инжектится main-side
  // по api_key_ref — renderer его не держит (см. createDetectOfftopicJob →
  // window.twitchCut.aiDetectOfftopic).
  useEffect(() => {
    let alive = true;
    (async () => {
      try {
        const { builtin, custom } = await getAiProviders();
        if (!alive) return;
        const all = [...builtin, ...custom];
        setAiProviders(all);

        // Пресенс ключей: уникальные ref → aiKeyHas. Провайдеры без ref в карту
        // не попадают (они «готовы» и так — фильтр это учитывает отдельно).
        const refs = Array.from(
          new Set(
            all
              .map((p) => p.api_key_ref)
              .filter((r): r is string => typeof r === "string" && r.length > 0)
          )
        );
        const entries = await Promise.all(
          refs.map(async (ref) => {
            const res = await window.twitchCut.aiKeyHas(ref);
            return [ref, Boolean(res.ok && res.present)] as const;
          })
        );
        if (!alive) return;
        const presence = Object.fromEntries(entries);
        setKeyPresence(presence);

        // Настроенный = кастомный keyless (добавлен вручную) ИЛИ ключ сохранён.
        // Встроенный keyless (Ollama) настроенным не считаем — см. configuredProviders.
        const isConfigured = (p: AiProvider) => {
          const ref = p.api_key_ref;
          if (ref) return Boolean(presence[ref]);
          return !p.builtin;
        };
        setSelectedProviderId((prev) =>
          all.some((p) => p.id === prev && isConfigured(p))
            ? prev
            : all.find(isConfigured)?.id ?? ""
        );
      } catch {
        /* провайдеры опциональны — кнопка детекта просто будет задизейблена */
      }
    })();
    return () => {
      alive = false;
    };
  }, []);

  // Re-attach живых detect-джоб при (пере)открытии «Правки». *JobId живут в
  // component-стейте: уход с экрана размонтирует Timeline и сбрасывает их в null,
  // из-за чего вернувшись пользователь не видел, что детект ещё идёт (кнопка не
  // залочена, спиннера нет), а результат по завершении молча не применялся.
  // На маунте подхватываем самую свежую running/pending detect-джобу каждого
  // вида обратно — useJobEvents переподпишется и вернёт индикатор + onFinal.
  useEffect(() => {
    let alive = true;
    (async () => {
      try {
        const jobs = await listJobs();
        if (!alive) return;
        const latestLive = (kind: JobState["kind"]) =>
          jobs
            .filter(
              (j) =>
                j.kind === kind &&
                (j.status === "running" || j.status === "pending")
            )
            .sort((a, b) => b.created_at.localeCompare(a.created_at))[0];
        const off = latestLive("detect_offtopic");
        if (off) setOfftopicJobId((cur) => cur ?? off.id);
        const sil = latestLive("detect_silence");
        if (sil) setSilenceJobId((cur) => cur ?? sil.id);
      } catch {
        /* нет бэка/джоб — не критично, просто не переподпишемся */
      }
    })();
    return () => {
      alive = false;
    };
  }, []);

  // Только настроенные провайдеры идут в выпадашку. «Настроен» = пользователь
  // его реально задал: для провайдера с ключом — ключ сохранён; для кастомного
  // (его добавили вручную) — всегда. Встроенные БЕЗ ключа (локальная Ollama) в
  // список НЕ кладём, пока пользователь их не настроил своим custom-провайдером:
  // иначе Ollama маячит в списке, даже если её нет и её не трогали.
  const configuredProviders = useMemo(
    () =>
      aiProviders.filter((p) => {
        const ref = p.api_key_ref;
        if (ref) return Boolean(keyPresence[ref]); // нужен ключ → должен быть сохранён
        return !p.builtin; // keyless: показываем только пользовательские (custom)
      }),
    [aiProviders, keyPresence]
  );

  const selectedProvider = useMemo(
    () => aiProviders.find((p) => p.id === selectedProviderId) ?? null,
    [aiProviders, selectedProviderId]
  );
  // ref ключа выбранного провайдера (у builtin = id; у custom = собственный id).
  const providerKeyRef = selectedProvider?.api_key_ref ?? null;
  const providerNeedsKey =
    providerKeyRef !== null && providerKeyRef !== undefined && providerKeyRef !== "";

  // Есть ли сохранённый ключ — спрашиваем main (safeStorage) при смене провайдера.
  // Провайдеры без ключа (локальная Ollama) считаем готовыми (providerHasKey=true).
  useEffect(() => {
    let alive = true;
    if (!selectedProvider) {
      setProviderHasKey(null);
      return;
    }
    if (!providerNeedsKey) {
      setProviderHasKey(true);
      return;
    }
    setProviderHasKey(null);
    (async () => {
      const res = await window.twitchCut.aiKeyHas(providerKeyRef as string);
      if (!alive) return;
      setProviderHasKey(Boolean(res.ok && res.present));
    })();
    return () => {
      alive = false;
    };
  }, [selectedProvider, providerNeedsKey, providerKeyRef]);

  // Детект-джоба: на done читаем decisions.offtopic заново (merger записал бандл
  // с stream_*_s) и кладём сегменты в offtopicStore через replaceAll (одна запись
  // в undo). keepManual — не затираем добавленные вручную.
  const offtopicEditRef = useRef(offtopicEdit);
  offtopicEditRef.current = offtopicEdit;
  const offtopicJob = useJobEvents(
    offtopicJobId,
    useCallback(
      (state: JobState) => {
        if (state.status === "done") {
          // Перечитываем decisions с диска: merger дописал бандл offtopic со
          // stream_*_s. Берём segments оттуда (не из state.result), чтобы UI и
          // файл на диске точно совпадали.
          (async () => {
            try {
              const fresh = (await readDecisions(decisionsPath)) as DecisionsDoc;
              docRef.current = fresh;
              const segs = fresh.offtopic?.segments ?? [];
              offtopicEditRef.current.replaceAll(
                segs.map((s) => ({
                  start: Number(s.start_s),
                  end: Number(s.end_s),
                  // Гейт автопринятия: ниже порога — грузим как «оставим»
                  // (rejected), показываем, но не режем. Порог читаем из ref,
                  // чтобы async-колбэк видел свежее значение слайдера.
                  status: (Number(s.confidence) || 0) >= autoAcceptConfRef.current
                    ? ("accepted" as const)
                    : ("rejected" as const),
                  confidence: s.confidence,
                  reason: s.reason,
                  topic: s.topic ?? null,
                  quote: s.quote ?? null,
                  model: s.model ?? null,
                  stage: s.stage ?? null,
                  pauseS: s.pause_s ?? null,
                  anchored: s.anchored ?? null,
                })),
                { keepManual: true }
              );
              toast.success(
                segs.length > 0
                  ? `Оффтоп: ${segs.length} участков — проверьте и сохраните`
                  : "Оффтопа не найдено"
              );
              // Переименование проекта в настоящий тайтл стрима с Twitch — ровно
              // тогда, когда пользователь ищет оффтоп (как он и просил). Тянем id
              // из имени файла-источника (vod_<id>.mp4) и достаём тайтл через
              // getVodInfo, пишем его в decisions.json (top-level "title"). Best-
              // effort: сеть/CLI могут упасть, тайтл не критичен — тогда молча
              // остаётся имя папки, а результат детекции не страдает.
              try {
                const settings = await getSettings();
                const wantRename = settings.rename_project_to_title !== false; // дефолт — вкл
                const already = (fresh.title ?? "").trim();
                if (wantRename && !already) {
                  const src = String(fresh.source ?? fresh.stream ?? "");
                  const idMatch = src.match(/vod_(\d+)/i) ?? decisionsPath.match(/vod_(\d+)/i);
                  if (idMatch) {
                    const info = await getVodInfo(twitchVodUrl(idMatch[1]));
                    const title = (info.title ?? "").trim();
                    if (title) {
                      const withTitle = { ...fresh, title };
                      await writeDecisions(decisionsPath, withTitle);
                      docRef.current = withTitle; // чтобы save() (spread docRef) не потерял тайтл
                      setProjectName(title);
                      toast.success(`Проект переименован: ${title}`);
                    }
                  }
                }
              } catch {
                /* тайтл не критичен — оставляем имя папки */
              }
            } catch (e) {
              toast.error("Не удалось прочитать результат детекции", {
                description: e instanceof Error ? e.message : String(e),
              });
            } finally {
              setOfftopicJobId(null);
            }
          })();
        } else if (state.status === "failed") {
          setOfftopicJobId(null);
          toast.error("Детект оффтопа упал", { description: state.error ?? undefined });
        }
      },
      [decisionsPath]
    )
  );
  const detectingOfftopic =
    offtopicJob.state?.status === "running" || offtopicJob.state?.status === "pending";

  // Готовность запуска детекта: провайдер выбран и (если нужен) ключ сохранён.
  const canDetectOfftopic = Boolean(selectedProvider) && providerHasKey === true;

  /** Запустить AI-детект оффтопа по текущему decisions/провайдеру. */
  const runDetectOfftopic = useCallback(async () => {
    if (!selectedProvider || detectingOfftopic || !canDetectOfftopic) return;
    setShowDetectPopover(false);
    try {
      const job = await createDetectOfftopicJob({
        decisions: decisionsPath,
        output: decisionsPath, // пишем обратно в тот же decisions.json (бандл offtopic)
        provider_id: selectedProvider.id,
        // Кастомный провайдер шлём целиком (backend возьмёт base_url/format);
        // builtin достаточно по id. api_key_ref — main достанет ключ и вставит.
        provider: selectedProvider.builtin
          ? null
          : (selectedProvider as unknown as Record<string, unknown>),
        api_key_ref: providerKeyRef ?? undefined,
        // Исходник (если выбран) → backend сам его транскрибирует и включит путь
        // по исходнику. Не задан → method не шлём, backend уйдёт в оконный режим.
        source_audio: sourcePath ?? undefined,
        model: detectModel || undefined,
        cheap_model: cheapModel || undefined,
        expensive_model: expensiveModel || undefined,
      });
      setOfftopicJobId(job.id);
      toast("Ищу оффтоп…", { description: `провайдер ${selectedProvider.label}` });
    } catch (e) {
      toast.error("Не удалось запустить детект оффтопа", {
        description: e instanceof Error ? e.message : String(e),
      });
    }
  }, [
    selectedProvider,
    detectingOfftopic,
    canDetectOfftopic,
    decisionsPath,
    providerKeyRef,
    sourcePath,
    detectModel,
    cheapModel,
    expensiveModel,
  ]);

  /** Выбрать локальный файл исходника (видео/аудио) для пути по исходнику. */
  const pickSource = useCallback(async () => {
    const picked = await window.twitchCut.openFile({
      filters: [{ name: "Видео/аудио", extensions: SOURCE_VIDEO_EXT }],
      title: "Видео, на которое идёт реакция",
    });
    if (picked) setSourcePath(picked);
  }, []);

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
      // У ручного мьюта нет находки детектора, поэтому подписью служит текст под
      // окном НА МОМЕНТ СОЗДАНИЯ. Дальше правки границ его не переписывают (см.
      // useMuteEditing): расхождение видно в инспекторе строкой «накрыто».
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
    setSelectedOfftopicId(null);
    centerViewOn(toAudio(start), toAudio(end));
    toast.success("Заглушка добавлена", { description: fmtMs(start * 1000) });
  }, [currentLocalS, muteStore, centerViewOn, toAudio]);

  // --- сохранение -----------------------------------------------------------
  const save = useCallback(async () => {
    if (saving || !store.dirty) return;
    const doc = docRef.current ?? {};
    const off = offsetRef.current;
    // Собираем mutes обратно: правим start/end/status, пересчитываем
    // stream_start/stream_end (их читает Vegas), сохраняем прочие поля. `word`
    // едет как есть — это находка детектора, а не подпись под окном.
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
    // Оффтоп-сегменты — LOCAL-секунды (start_s/end_s), пересчитываем stream_*_s
    // из start+offset ровно как mutes. Метаданные бандла (source/transcript_hash/
    // model) сохраняем из прежнего doc.offtopic. status кладём в сегмент, чтобы
    // ExportScreen мог отличить accepted (вырезать) от rejected (оставить).
    const nextOfftopicSegs: OfftopicSegmentDoc[] = offtopicStore.present
      .slice()
      .sort((a, b) => a.start - b.start)
      .map((o) => ({
        start_s: round3(o.start),
        end_s: round3(o.end),
        stream_start_s: round3(o.start + off),
        stream_end_s: round3(o.end + off),
        confidence: o.confidence,
        reason: o.reason,
        topic: o.topic ?? null,
        quote: o.quote ?? null,
        model: o.model ?? null,
        stage: o.stage ?? null,
        // Улики пути по исходнику переживают round-trip только если их вернуть.
        ...(o.pauseS != null ? { pause_s: o.pauseS } : {}),
        ...(o.anchored != null ? { anchored: o.anchored } : {}),
        status: o.status,
      }));
    const nextDoc: DecisionsDoc = {
      ...doc,
      mutes: nextMutes,
      silences: nextSilences,
      offtopic: { ...doc.offtopic, segments: nextOfftopicSegs },
    };
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
  }, [saving, store, muteStore, silenceStore, offtopicStore, decisionsPath]);

  // --- навигация по сегментам активного слоя клавишами ----------------------
  useHotkey(["arrowdown", "arrowright"], (e) => {
    e.preventDefault();
    goToSegment(1);
  });
  useHotkey(["arrowup", "arrowleft"], (e) => {
    e.preventDefault();
    goToSegment(-1);
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
    // Оффтоп в фокусе — переключаем его; иначе тишина; иначе мат.
    if (selectedOfftopicId) {
      offtopicEdit.toggleStatus(selectedOfftopicId);
      return;
    }
    if (selectedSilenceId) {
      silenceEdit.toggleStatus(selectedSilenceId);
      return;
    }
    const m = mutes.find((x) => x.id === selectedId);
    if (m) toggleMuted(m);
  });
  useHotkey("delete", () => {
    if (selectedOfftopicId) {
      offtopicEdit.deleteSegment(selectedOfftopicId);
      setSelectedOfftopicId(null);
      return;
    }
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
    setSelectedOfftopicId(null);
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
  // Тишина живёт в audio-секундах — передаём границы как есть.
  const handleSilenceResize = useCallback((id: string, aStart: number, aEnd: number) => {
    silenceEditRef.current.resize(id, aStart, aEnd);
  }, []);
  // Оффтоп рисуется в audio-space (toAudio), но хранится в local — конвертируем обратно.
  const handleOfftopicResize = useCallback((id: string, aStart: number, aEnd: number) => {
    offtopicEditRef.current.resize(id, toLocalRef.current(aStart), toLocalRef.current(aEnd));
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
  // Индексы для счётчиков навигации по тишине/оффтопу — в том же порядке (по
  // start), в котором ходит goToSegment, чтобы «N/M» совпадал с прыжками стрелок.
  const silenceIndex = selectedSilence
    ? [...silences].sort((a, b) => a.start - b.start).findIndex((s) => s.id === selectedSilence.id)
    : -1;
  const offtopicIndex = selectedOfftopic
    ? [...offtopic].sort((a, b) => a.start - b.start).findIndex((o) => o.id === selectedOfftopic.id)
    : -1;

  return (
    <div className="timeline-page flex h-full min-h-0 flex-col gap-3 p-4">
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

      {/* Слои: маты (крас.) · тишина (жёлт.) · оффтоп (син.) — видны разом, каждый
          со своим чекбоксом видимости. Клик по телу чипа делает слой активным
          (его тулбар/инспектор). Действия слоя живут в его тулбаре ниже. */}
      <div className="flex flex-wrap items-center gap-2">
        <div className="flex items-center gap-1 rounded-lg border border-white/10 bg-white/5 p-1">
          <LayerChip
            label="Маты"
            icon={<VolumeX className="h-3.5 w-3.5" />}
            swatchClass="bg-err"
            count={mutes.length}
            visible={layers.mutes}
            active={activeLayer === "mutes"}
            onToggleVisible={() => toggleLayerVisible("mutes")}
            onActivate={() => switchLayer("mutes")}
          />
          <LayerChip
            label="Тишина"
            icon={<Scissors className="h-3.5 w-3.5" />}
            swatchClass="bg-amber-400"
            count={silences.length}
            visible={layers.silence}
            active={activeLayer === "silence"}
            onToggleVisible={() => toggleLayerVisible("silence")}
            onActivate={() => switchLayer("silence")}
          />
          <LayerChip
            label="Оффтоп"
            icon={<MessageSquareOff className="h-3.5 w-3.5" />}
            swatchClass="bg-blue-500"
            count={offtopic.length}
            visible={layers.offtopic}
            active={activeLayer === "offtopic"}
            onToggleVisible={() => toggleLayerVisible("offtopic")}
            onActivate={() => switchLayer("offtopic")}
          />
        </div>
      </div>

      {/* Тулбар активного слоя «Маты»: переход по матам · зум · добавить · сводка */}
      {activeLayer === "mutes" && (
        <div className="flex flex-wrap items-center gap-2">
          <div className="flex items-center gap-1 rounded-md border border-white/5 bg-black/30 p-1">
            <Button
              size="sm"
              variant="ghost"
              onClick={() => goToSegment(-1)}
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
              onClick={() => goToSegment(1)}
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

      {/* Тулбар активного слоя «Тишина»: найти тишину · (шестерёнка) параметры · сводка */}
      {activeLayer === "silence" && (
        <div className="flex flex-wrap items-center gap-2">
          <div className="flex items-center gap-1 rounded-md border border-white/5 bg-black/30 p-1">
            <Button
              size="sm"
              variant="ghost"
              onClick={() => goToSegment(-1)}
              disabled={!silences.length}
              className="h-7 w-7 p-0"
              aria-label="Предыдущая тишина"
            >
              <ChevronLeft className="h-4 w-4" />
            </Button>
            <span className="min-w-[92px] text-center font-mono text-[11px] tabular-nums text-subtle">
              {silenceIndex >= 0
                ? `тишина ${silenceIndex + 1}/${silences.length}`
                : `тишин ${silences.length}`}
            </span>
            <Button
              size="sm"
              variant="ghost"
              onClick={() => goToSegment(1)}
              disabled={!silences.length}
              className="h-7 w-7 p-0"
              aria-label="Следующая тишина"
            >
              <ChevronRight className="h-4 w-4" />
            </Button>
          </div>

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
            {maxPause}с · {thresholdDb}dB
          </Button>

          {/* Параметры (пауза/порог) — два слайдера, разворачиваются своей строкой */}
          {showSilenceParams && (
            <div className="order-last w-full rounded-md border border-white/5 bg-black/30 px-3 py-2.5">
              <div className="grid grid-cols-[auto_1fr_auto] items-center gap-x-3 gap-y-2.5">
                <label htmlFor="silence-pause" className="text-xs text-muted">
                  Пауза
                </label>
                <input
                  id="silence-pause"
                  type="range"
                  min={0}
                  max={10}
                  step={0.5}
                  value={maxPause}
                  disabled={detectingSilence}
                  onChange={(e) => setMaxPause(round1(Number(e.target.value)))}
                  className="h-1.5 w-full cursor-pointer appearance-none rounded-full bg-white/10 accent-amber-400 disabled:opacity-50"
                  aria-label="Максимальная длина паузы к вырезу, секунд"
                />
                <span className="min-w-[52px] text-right font-mono text-[11px] tabular-nums text-fg">
                  {maxPause} с
                </span>

                <label htmlFor="silence-threshold" className="text-xs text-muted">
                  Порог
                </label>
                <input
                  id="silence-threshold"
                  type="range"
                  min={-50}
                  max={-15}
                  step={1}
                  value={thresholdDb}
                  disabled={detectingSilence}
                  onChange={(e) => setThresholdDb(Number(e.target.value))}
                  className="h-1.5 w-full cursor-pointer appearance-none rounded-full bg-white/10 accent-amber-400 disabled:opacity-50"
                  aria-label="Порог тишины в dBFS: тише порога = тишина"
                />
                <span className="min-w-[52px] text-right font-mono text-[11px] tabular-nums text-fg">
                  {thresholdDb} dB
                </span>
              </div>
              <p className="mt-2 text-[11px] leading-snug text-subtle">
                Паузы длиннее выбранной и тише порога попадают в вырез. Ниже
                порог — строже (меньше ложных срабатываний).
              </p>
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

      {/* Тулбар активного слоя «Оффтоп»: найти ИИ · добавить вручную · сводка */}
      {activeLayer === "offtopic" && (
        <div className="flex flex-wrap items-center gap-2">
          <div className="flex items-center gap-1 rounded-md border border-white/5 bg-black/30 p-1">
            <Button
              size="sm"
              variant="ghost"
              onClick={() => goToSegment(-1)}
              disabled={!offtopic.length}
              className="h-7 w-7 p-0"
              aria-label="Предыдущий оффтоп"
            >
              <ChevronLeft className="h-4 w-4" />
            </Button>
            <span className="min-w-[96px] text-center font-mono text-[11px] tabular-nums text-subtle">
              {offtopicIndex >= 0
                ? `оффтоп ${offtopicIndex + 1}/${offtopic.length}`
                : `оффтопов ${offtopic.length}`}
            </span>
            <Button
              size="sm"
              variant="ghost"
              onClick={() => goToSegment(1)}
              disabled={!offtopic.length}
              className="h-7 w-7 p-0"
              aria-label="Следующий оффтоп"
            >
              <ChevronRight className="h-4 w-4" />
            </Button>
          </div>

          {/* AI-детект оффтопа: поповер с выбором провайдера (только настроенные).
              Задизейблен, если у провайдера нет ключа — тултип уводит в Настройки. */}
          <div className="relative">
            <Button
              size="sm"
              variant="secondary"
              onClick={() => setShowDetectPopover((v) => !v)}
              disabled={detectingOfftopic}
              loading={detectingOfftopic}
              aria-label="Найти оффтоп с помощью ИИ"
            >
              <Sparkles className="mr-1.5 h-3.5 w-3.5" />
              {detectingOfftopic ? "ищу оффтоп…" : "найти оффтоп (ИИ)"}
            </Button>
            {showDetectPopover && !detectingOfftopic && (
              <div className="absolute left-0 top-full z-20 mt-1 w-80 rounded-lg border border-white/10 bg-surface p-3 shadow-xl">
                <div className="mb-2 flex items-center gap-1.5 text-xs font-medium text-fg">
                  <Bot className="h-3.5 w-3.5 text-brand-from" /> Провайдер ИИ
                </div>
                {configuredProviders.length === 0 ? (
                  <p className="text-xs text-subtle">
                    Нет настроенных провайдеров. Задайте ключ в{" "}
                    <span className="text-fg">Настройки → Провайдеры ИИ</span>.
                  </p>
                ) : (
                  <>
                    <select
                      value={selectedProviderId}
                      onChange={(e) => {
                        setSelectedProviderId(e.target.value);
                        // Список моделей у каждого провайдера свой — сбрасываем выбор.
                        setDetectModel("");
                        setCheapModel("");
                        setExpensiveModel("");
                      }}
                      className="w-full rounded-md border border-white/10 bg-black/30 px-2 py-1.5 text-sm text-fg outline-none focus:border-brand-from/50"
                    >
                      {configuredProviders.map((p) => (
                        <option key={p.id} value={p.id}>
                          {p.builtin ? p.label : `${p.label} (свой)`}
                        </option>
                      ))}
                    </select>

                    {/* Исходник: локальный файл видео/аудио, на которое идёт
                        реакция. Задан → путь по исходнику (точнее); пусто →
                        старый оконный режим. Кнопку запуска НЕ блокирует. */}
                    <div className="mt-3">
                      <div className="mb-1 text-[11px] font-medium text-muted">
                        Исходник (видео реакции)
                      </div>
                      {sourcePath ? (
                        <div className="flex items-center gap-1.5 rounded-md border border-white/10 bg-black/30 px-2 py-1.5">
                          <FolderOpen className="h-3.5 w-3.5 shrink-0 text-brand-from" />
                          <span
                            className="min-w-0 flex-1 truncate text-xs text-fg"
                            title={sourcePath}
                          >
                            {baseName(sourcePath)}
                          </span>
                          <button
                            type="button"
                            onClick={() => setSourcePath(null)}
                            className="shrink-0 text-[11px] text-subtle hover:text-fg"
                          >
                            убрать
                          </button>
                        </div>
                      ) : (
                        <Button
                          size="sm"
                          variant="ghost"
                          onClick={pickSource}
                          className="w-full justify-start"
                        >
                          <FolderOpen className="mr-1.5 h-3.5 w-3.5" /> выбрать файл…
                        </Button>
                      )}
                      <p className="mt-1 text-[11px] leading-snug text-subtle">
                        {sourcePath
                          ? "Распознаем исходник и разметим оффтоп по нему (точнее)."
                          : "Без исходника — старый оконный режим (менее точно)."}
                      </p>
                    </div>

                    {/* Модель LLM: пусто → дефолт провайдера. Для source/single-pass. */}
                    {(selectedProvider?.models?.length ?? 0) > 0 && (
                      <div className="mt-3">
                        <div className="mb-1 text-[11px] font-medium text-muted">
                          Модель
                        </div>
                        <select
                          value={detectModel}
                          onChange={(e) => setDetectModel(e.target.value)}
                          className="w-full rounded-md border border-white/10 bg-black/30 px-2 py-1.5 text-sm text-fg outline-none focus:border-brand-from/50"
                        >
                          <option value="">— модель по умолчанию —</option>
                          {selectedProvider?.models?.map((m) => (
                            <option key={m} value={m}>
                              {m}
                            </option>
                          ))}
                        </select>
                      </div>
                    )}

                    {/* «Дополнительно» — каскад cheap/expensive для оконного пути
                        (без исходника). Свёрнуто по умолчанию, как параметры тишины. */}
                    {(selectedProvider?.models?.length ?? 0) > 0 && (
                      <div className="mt-2">
                        <button
                          type="button"
                          onClick={() => setShowDetectAdvanced((v) => !v)}
                          className="flex items-center gap-1 text-[11px] text-subtle hover:text-fg"
                        >
                          <ChevronDown
                            className={cn(
                              "h-3 w-3 transition-transform",
                              showDetectAdvanced && "rotate-180"
                            )}
                          />
                          Дополнительно
                        </button>
                        {showDetectAdvanced && (
                          <div className="mt-2 space-y-2 rounded-md border border-white/5 bg-black/20 p-2">
                            <div>
                              <div className="mb-1 text-[11px] text-muted">
                                Дешёвая модель
                              </div>
                              <select
                                value={cheapModel}
                                onChange={(e) => setCheapModel(e.target.value)}
                                className="w-full rounded-md border border-white/10 bg-black/30 px-2 py-1.5 text-xs text-fg outline-none focus:border-brand-from/50"
                              >
                                <option value="">— по умолчанию —</option>
                                {selectedProvider?.models?.map((m) => (
                                  <option key={m} value={m}>
                                    {m}
                                  </option>
                                ))}
                              </select>
                            </div>
                            <div>
                              <div className="mb-1 text-[11px] text-muted">
                                Дорогая модель
                              </div>
                              <select
                                value={expensiveModel}
                                onChange={(e) => setExpensiveModel(e.target.value)}
                                className="w-full rounded-md border border-white/10 bg-black/30 px-2 py-1.5 text-xs text-fg outline-none focus:border-brand-from/50"
                              >
                                <option value="">— по умолчанию —</option>
                                {selectedProvider?.models?.map((m) => (
                                  <option key={m} value={m}>
                                    {m}
                                  </option>
                                ))}
                              </select>
                            </div>
                            <p className="text-[11px] leading-snug text-subtle">
                              Каскад дёшево→дорого — только для оконного режима
                              (без исходника).
                            </p>
                          </div>
                        )}
                      </div>
                    )}

                    <p className="mt-2 text-[11px] text-subtle">
                      {providerHasKey === null
                        ? "Проверяю ключ…"
                        : canDetectOfftopic
                        ? "Детект перезапишет прежние сегменты оффтопа."
                        : "Нет сохранённого ключа. Задайте его в Настройки → Провайдеры ИИ."}
                    </p>
                    <div className="mt-3 flex justify-end">
                      <Button
                        size="sm"
                        variant="primary"
                        onClick={runDetectOfftopic}
                        disabled={!canDetectOfftopic}
                      >
                        <Sparkles className="mr-1.5 h-3.5 w-3.5" /> запустить
                      </Button>
                    </div>
                  </>
                )}
              </div>
            )}
          </div>

          {/* Прогресс детекта — своей строкой. detect_offtopic крутится ИНЛАЙНОМ
              (не в JobScreen), а стадия «Транскрипция исходника» тянется минутами,
              поэтому показываем бар + подпись, иначе выглядит зависшим. */}
          {detectingOfftopic && (
            <div className="order-last w-full rounded-md border border-white/5 bg-black/30 px-3 py-2.5">
              <div className="mb-1.5 flex items-center gap-2 text-xs">
                <Loader2 className="h-3.5 w-3.5 animate-spin text-brand-from" />
                <span className="font-medium text-fg">
                  {stageLabel("detect_offtopic", offtopicJob.state?.stage)}
                </span>
                {offtopicJob.state?.message && (
                  <span className="min-w-0 flex-1 truncate text-subtle">
                    {offtopicJob.state.message}
                  </span>
                )}
                <span className="ml-auto font-mono text-[11px] tabular-nums text-subtle">
                  {Math.round(offtopicJob.state?.progress ?? 0)}%
                </span>
              </div>
              <ProgressBar value={offtopicJob.state?.progress ?? 0} />
            </div>
          )}

          <Button
            size="sm"
            variant="ghost"
            onClick={() => {
              const id = offtopicEdit.addSegment(currentLocalS, currentLocalS + 2);
              if (id) {
                selectOfftopic(id);
                toast.success("Сегмент оффтопа добавлен", { description: fmtMs(currentLocalS * 1000) });
              }
            }}
          >
            <Plus className="mr-1.5 h-3.5 w-3.5" /> оффтоп на плейхеде
          </Button>

          {/* Порог автопринятия: сегменты слабее порога грузятся как «оставим».
              Живёт в сессии; применяется к СЛЕДУЮЩЕМУ детекту (не ретро-гейтит). */}
          <div className="flex items-center gap-2">
            <label htmlFor="offtopic-conf" className="text-xs text-muted">
              Порог
            </label>
            <input
              id="offtopic-conf"
              type="range"
              min={0}
              max={1}
              step={0.05}
              value={autoAcceptConf}
              disabled={detectingOfftopic}
              onChange={(e) => setAutoAcceptConf(Number(e.target.value))}
              className="h-1.5 w-24 cursor-pointer appearance-none rounded-full bg-white/10 accent-blue-400 disabled:opacity-50"
              aria-label="Порог автопринятия: сегменты слабее порога помечаются «оставить»"
              title="Уверенность ниже порога → сегмент грузится как «оставим». Применится к следующему детекту."
            />
            <span className="min-w-[34px] text-right font-mono text-[11px] tabular-nums text-fg">
              {Math.round(autoAcceptConf * 100)}%
            </span>
          </div>

          <span className="ml-auto flex items-center gap-1.5 text-xs text-subtle">
            {offtopic.length > 0 ? (
              <>
                <MessageSquareOff className="h-3.5 w-3.5 text-blue-400" />
                <span className="text-blue-400">{acceptedOfftopicCount}</span> вырежем
                <span className="mx-1 text-white/15">·</span>
                {offtopic.length - acceptedOfftopicCount} оставим
              </>
            ) : (
              <span className="text-subtle">оффтоп не искался</span>
            )}
          </span>

          <span className="font-mono text-[11px] tabular-nums text-subtle">
            {fmtMs(currentAudioS * 1000)} / {fmtMs(wave.duration_s * 1000)}
          </span>
        </div>
      )}

      <div className="editor-workspace">
        <ReviewList layer={activeLayer} items={activeLayer === "mutes" ? mutes.map((m) => ({
          id: m.id, start: m.start, end: m.end, label: m.word || "Заглушка", selected: selectedId === m.id,
          detail: m.review ? "Проверить" : m.status === SILENCED ? "Заглушить" : "Оставить",
          onSelect: () => { selectMute(m.id, { center: true }); seekAudio(toAudio(m.start)); },
        })) : activeLayer === "silence" ? silences.map((s) => ({
          id: s.id, start: s.start, end: s.end, label: "Тишина", selected: selectedSilenceId === s.id,
          detail: `${(s.end - s.start).toFixed(1)} с`,
          onSelect: () => { selectSilence(s.id); centerViewOn(s.start, s.end); seekAudio(s.start); },
        })) : offtopic.map((o) => ({
          id: o.id, start: o.start, end: o.end, label: "Оффтоп", selected: selectedOfftopicId === o.id,
          detail: o.reason || `${(o.end - o.start).toFixed(1)} с`,
          onSelect: () => { selectOfftopic(o.id); centerViewOn(toAudio(o.start), toAudio(o.end)); seekAudio(toAudio(o.start)); },
        }))} />
        <div className="editor-main">
          <PreviewDeck key={audioPath} source={previewSource} audioRef={audioRef} videoOffset={offsetRef.current - toAudio(0)} toLocal={toLocal} toAudio={toAudio} words={wordIndexRef.current} onSeek={seekAudio} onPlay={(s) => playFrom(s, true)} onStop={stopPlayback} />
          <div className="editor-lower"><div className="editor-wave-column">
      {/* Главный таймлайн */}
      {activeLayer === "mutes" && mutes.length === 0 && (
        <Card variant="surface" padding="sm">
          <p className="text-center text-xs text-subtle">
            Заглушек нет — мат не найден или все убраны. «+ мат на плейхеде» добавит вручную.
          </p>
        </Card>
      )}
      {activeLayer === "silence" && silences.length === 0 && (
        <Card variant="surface" padding="sm">
          <p className="text-center text-xs text-subtle">
            Тишина ещё не искалась. «найти тишину» подберёт длинные паузы к вырезу.
          </p>
        </Card>
      )}
      {activeLayer === "offtopic" && offtopic.length === 0 && (
        <Card variant="surface" padding="sm">
          <p className="text-center text-xs text-subtle">
            Оффтоп ещё не искался. «найти оффтоп (ИИ)» разметит болтовню не по теме.
          </p>
        </Card>
      )}
      <div className="timeline-ruler" aria-label="Временная шкала">
        {Array.from({ length: 7 }, (_, i) => <span key={i}>{fmtMs((view.start + (view.end - view.start) * i / 6) * 1000)}</span>)}
      </div>
      <div ref={waveBoxRef} className="wave-workspace relative min-h-0 flex-1">
        <TimelineCanvas
          tile={waveTile ?? EMPTY_TILE}
          underlay={waveBase}
          norm={waveNorm}
          durationS={wave.duration_s}
          blocks={layers.mutes ? markers : EMPTY_BLOCKS}
          view={view}
          currentRef={currentAudioRef}
          onViewChange={setView}
          onSeek={seekAudio}
          onSelect={handleSelectPlay}
          onResize={handleResize}
          onScrub={scrubAtAudio}
        />
        {/* Оверлеи тишины/оффтопа стекуются поверх канваса по видимости слоёв.
            Оффтоп рисуем последним → его блоки сверху (клик приоритетнее). */}
        {layers.silence && silenceBlocks.length > 0 && (
          <SilenceOverlay
            blocks={silenceBlocks}
            durationS={wave.duration_s}
            view={view}
            onSelect={selectSilence}
            onResize={activeLayer === "silence" ? handleSilenceResize : undefined}
          />
        )}
        {layers.offtopic && offtopicBlocks.length > 0 && (
          <OfftopicOverlay
            blocks={offtopicBlocks}
            durationS={wave.duration_s}
            view={view}
            onSelect={selectOfftopic}
            onResize={activeLayer === "offtopic" ? handleOfftopicResize : undefined}
          />
        )}
      </div>

            {/* Мини-карта: весь стрим + рамка окна зума */}
      <OverviewStrip
        durationS={wave.duration_s}
        tile={waveBase ?? EMPTY_TILE}
        norm={waveNorm}
        markers={markers}
        currentRef={currentAudioRef}
        view={view}
        onSeek={seekAudio}
        onPickMarker={handlePickMarker}
      />


      </div><aside className="editor-inspector" aria-label="Параметры выбранного фрагмента">
      {!selected && !selectedSilence && !selectedOfftopic && <p className="p-4 text-xs text-muted">Выберите фрагмент в списке или на волне, чтобы изменить его границы и действие.</p>}
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

      {/* Инспектор выбранного участка тишины */}
      {selectedSilence && (
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

      {/* Инспектор выбранного оффтоп-сегмента: метаданные классификатора
          (тема/причина/цитата/уверенность) read-only + вырезать/оставить/удалить. */}
      {selectedOfftopic && (
        <Card variant="surface" padding="sm">
          <div className="flex flex-col gap-2 text-sm">
            <div className="flex flex-wrap items-center gap-3">
              <span className="flex items-center gap-1.5 font-medium">
                <MessageSquareOff className="h-4 w-4 text-blue-400" /> Оффтоп
              </span>
              <span className="font-mono text-xs tabular-nums text-subtle">
                {fmtMs(selectedOfftopic.start * 1000)} – {fmtMs(selectedOfftopic.end * 1000)}
                <span className="ml-1.5 text-white/40">
                  ({(selectedOfftopic.end - selectedOfftopic.start).toFixed(2)}с)
                </span>
              </span>
              {selectedOfftopic.topic && (
                <span className="rounded bg-blue-500/15 px-1.5 py-0.5 text-[11px] text-blue-300">
                  {selectedOfftopic.topic}
                </span>
              )}
              <span className="font-mono text-[11px] tabular-nums text-subtle">
                увер. {(selectedOfftopic.confidence * 100).toFixed(0)}%
              </span>
              <span
                className={
                  selectedOfftopic.status === "accepted"
                    ? "rounded bg-blue-500/15 px-1.5 py-0.5 text-[11px] text-blue-300"
                    : "rounded bg-white/8 px-1.5 py-0.5 text-[11px] text-subtle"
                }
              >
                {selectedOfftopic.status === "accepted" ? "вырежем" : "оставим"}
              </span>

              <div className="ml-auto flex items-center gap-1.5">
                <Button
                  size="sm"
                  variant="ghost"
                  onClick={() => {
                    playAudioRange(
                      toAudio(selectedOfftopic.start),
                      toAudio(selectedOfftopic.end),
                      0.06
                    );
                  }}
                >
                  прослушать
                </Button>
                <Button
                  size="sm"
                  variant={selectedOfftopic.status === "accepted" ? "secondary" : "primary"}
                  onClick={() => offtopicEdit.toggleStatus(selectedOfftopic.id)}
                >
                  {selectedOfftopic.status === "accepted" ? "оставить" : "вырезать"}
                </Button>
                <Button
                  size="sm"
                  variant="ghost"
                  onClick={() => {
                    offtopicEdit.deleteSegment(selectedOfftopic.id);
                    setSelectedOfftopicId(null);
                  }}
                  aria-label="Удалить сегмент оффтопа"
                >
                  <Trash2 className="h-3.5 w-3.5" />
                </Button>
              </div>
            </div>

            {hiddenInSelectedOfftopic && (
              <p className="text-[11px] leading-snug text-blue-300/70">
                Внутри скрыто: {hiddenInSelectedOfftopic.mutes} матов ·{" "}
                {hiddenInSelectedOfftopic.silences} тишины. Появятся снова, если
                удалить оффтоп или спрятать слой.
              </p>
            )}
            {selectedOfftopic.reason && (
              <p className="text-xs text-muted">{selectedOfftopic.reason}</p>
            )}
            {selectedOfftopic.quote && (
              <p className="border-l-2 border-blue-500/40 pl-2 text-xs italic text-subtle">
                «{selectedOfftopic.quote}»
              </p>
            )}
            {/* Улики пути по исходнику: пауза видео и доля привязки — независимая
                (не-LLM) опора вердикта. Есть только у сегментов из этого пути. */}
            {selectedOfftopic.pauseS != null && (
              <p className="font-mono text-[10px] tabular-nums text-blue-300/60">
                пауза {selectedOfftopic.pauseS.toFixed(0)}с
                {selectedOfftopic.anchored != null && (
                  <> · привязка {(selectedOfftopic.anchored * 100).toFixed(0)}%</>
                )}
              </p>
            )}
            {(selectedOfftopic.model || selectedOfftopic.stage) && (
              <p className="font-mono text-[10px] tabular-nums text-white/30">
                {[selectedOfftopic.stage, selectedOfftopic.model].filter(Boolean).join(" · ")}
              </p>
            )}
          </div>
        </Card>
      )}

        </aside></div></div>
      </div>

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
          ←→/↑↓ — между матами · щипок/колесо — зум · две пальца/Shift+колесо — панорама · Space — прослушать · M/Enter — глушить/вырезать/оставить (выбранное) · Del — удалить · Esc — снять выбор
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

/**
 * Попадает ли ЦЕНТР отрезка [a,b] внутрь одного из диапазонов (те же единицы)?
 * По центру, а не по полному вхождению: сегмент оффтопа обычно много больше
 * отдельного мата/паузы, а край может слегка вылезать — «свой» тот, чья
 * середина внутри. Используется, чтобы прятать маты/тишину под сегментом оффтопа.
 */
function centerInsideAny(a: number, b: number, ranges: readonly [number, number][]): boolean {
  if (ranges.length === 0) return false;
  const c = (a + b) / 2;
  for (const [s, e] of ranges) {
    if (c >= s && c <= e) return true;
  }
  return false;
}

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

/**
 * Маппинг сегментов из decisions.offtopic.segments → UI-модель (LOCAL-секунды).
 * Границы берём из start_s/end_s (как mutes[i].start); метаданные несём для
 * инспектора. status по умолчанию accepted (вырезать), если в файле не сказано
 * иначе. Валидные (end>start) и отсортированы по началу.
 */
function mapOfftopicSegments(
  segs: OfftopicSegmentDoc[] | undefined
): OfftopicSegmentUi[] {
  if (!Array.isArray(segs)) return [];
  return segs
    .filter(
      (s) =>
        Number.isFinite(s.start_s) && Number.isFinite(s.end_s) && s.end_s > s.start_s
    )
    .map((s) => {
      const start = Number(s.start_s);
      const rawStatus = (s as { status?: unknown }).status;
      return {
        id: String(
          (s as { id?: unknown }).id ?? `offtopic_auto_${Math.round(start * 1000)}`
        ),
        start,
        end: Number(s.end_s),
        status: (rawStatus === "rejected" ? "rejected" : "accepted") as OfftopicSegmentUi["status"],
        confidence: Number.isFinite(s.confidence) ? Number(s.confidence) : 0,
        reason: typeof s.reason === "string" ? s.reason : "",
        topic: s.topic ?? null,
        quote: s.quote ?? null,
        model: s.model ?? null,
        stage: s.stage ?? null,
        pauseS: s.pause_s ?? null,
        anchored: s.anchored ?? null,
      };
    })
    .sort((a, b) => a.start - b.start);
}

/**
 * LayerChip — чип слоя таймлайна: чекбокс видимости оверлея + цветной свотч +
 * иконка/название + счётчик. Клик по телу делает слой активным (подсветка),
 * клик по чекбоксу — показывает/прячет оверлей (активность не трогает).
 */
function LayerChip({
  label,
  icon,
  swatchClass,
  count,
  visible,
  active,
  onToggleVisible,
  onActivate,
}: {
  label: string;
  icon: React.ReactNode;
  swatchClass: string;
  count: number;
  visible: boolean;
  active: boolean;
  onToggleVisible: () => void;
  onActivate: () => void;
}) {
  return (
    <div
      className={cn(
        "inline-flex h-8 items-center gap-2 rounded-md border px-2 text-xs font-medium transition-colors",
        active
          ? "border-white/15 bg-surface text-fg shadow-[0_1px_0_rgb(255_255_255/0.05)_inset]"
          : "border-transparent text-muted hover:text-fg"
      )}
    >
      <input
        type="checkbox"
        checked={visible}
        onChange={onToggleVisible}
        onClick={(e) => e.stopPropagation()}
        className="h-3.5 w-3.5 rounded border-white/10 bg-black/30 accent-brand-from"
        aria-label={`Показывать слой «${label}»`}
      />
      <button
        type="button"
        onClick={onActivate}
        className="inline-flex items-center gap-2"
      >
        <span className={cn("h-2.5 w-2.5 rounded-sm", swatchClass, !visible && "opacity-30")} />
        {icon} {label}
        <span className="rounded bg-white/8 px-1.5 py-px font-mono text-[10px] tabular-nums text-subtle">
          {count}
        </span>
      </button>
    </div>
  );
}
