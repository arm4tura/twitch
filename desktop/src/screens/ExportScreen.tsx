import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  Download,
  FileCode2,
  FolderOpen,
  Loader2,
  Send,
  Sparkles,
  Video,
} from "lucide-react";
import {
  createDetectSilenceJob,
  createExportVegasJob,
  createRenderJob,
  readDecisions,
  type DetectSilenceRequest,
  type DetectSilenceResult,
  type JobState,
  type RenderRequest,
} from "../api";
import { useJobEvents } from "../hooks/useJobEvents";
import { Button } from "../components/ui/Button";
import { Card, CardDescription, CardTitle } from "../components/ui/Card";
import { FileField } from "../components/ui/FileField";
import { Label } from "../components/ui/Input";
import { toast } from "../components/ui/Toast";
import { fmtMs, parseTimecode } from "../lib/format";

/**
 * ExportScreen — выгрузка готового проекта.
 *
 * Главное действие — сборка скрипта для Sony Vegas Pro (мут-регионы как takes).
 * Это то, ради чего пользователь сюда пришёл, поэтому карточка Vegas стоит
 * первой и на всю ширину. Второе действие — финальный рендер (вырезать
 * заглушки + опц. тишину).
 *
 * Нарезка на клипы и весь NotebookLM-конвейер переехали в отдельную вкладку
 * «AI Tiktok» (AiTiktokScreen) — здесь остаётся только напоминание-ссылка.
 *
 * Каждая карточка — самодостаточная форма: поля пути, «Запустить», после
 * успешного создания job'а — toast с «Показать» через `showInFolder` IPC.
 */

export interface ExportScreenProps {
  decisionsPath: string;
  onJobStarted: (jobId: string) => void;
}

interface Summary {
  mutes: number;
  cuts: number;
  highlights: number;
  /** Сам массив highlights (для панели) — stream-координаты приходят из merge. */
  highlightItems: any[];
  workdir: string | null;
  stream: string | null;
  /** Сегменты заглушённых мьютов в АБСОЛЮТНЫХ (stream) секундах — для рендера. */
  mutedSegments: [number, number][];
  /**
   * Сегменты тишины, отревьюенные в «Правке» (status=accepted), в АБСОЛЮТНЫХ
   * (stream) секундах — уже готовы к вырезанию. Именно их ждёт пользователь:
   * «я всё сделал в Правке — просто вырежи это», без повторного детекта.
   */
  savedSilenceSegments: [number, number][];
  /**
   * Диапазон обработки (range_in/range_out) в АБСОЛЮТНЫХ секундах исходника,
   * либо null если границы нет (обрабатывали весь файл). Рендер обрежет вывод
   * по этому окну — иначе в итог попадёт всё видео, а не только размеченное.
   */
  rangeStartS: number | null;
  rangeEndS: number | null;
}

const LS_KEY = "twitchCut.export.v1";

interface FormState {
  vegasPath: string;
  renderOutputPath: string;
}

const DEFAULT_FORM: FormState = {
  vegasPath: "",
  renderOutputPath: "",
};

function loadForm(): FormState {
  try {
    const raw = localStorage.getItem(LS_KEY);
    if (!raw) return DEFAULT_FORM;
    return { ...DEFAULT_FORM, ...JSON.parse(raw) };
  } catch {
    return DEFAULT_FORM;
  }
}

function extractSummary(doc: any): Summary {
  const meta = (doc?._meta ?? {}) as Record<string, unknown>;
  const rawMutes: any[] = Array.isArray(doc?.mutes) ? doc.mutes : [];
  // Рендер режет исходный stream, поэтому нужны АБСОЛЮТНЫЕ координаты
  // (stream_start/stream_end). Если их нет (простой режим без сдвига) —
  // start/end и так уже абсолютны. Берём только реально глушимые (accepted).
  const mutedSegments = rawMutes
    .filter((m) => m?.status === "accepted")
    .map((m): [number, number] => {
      const s = Number.isFinite(m.stream_start) ? Number(m.stream_start) : Number(m.start) || 0;
      const e = Number.isFinite(m.stream_end) ? Number(m.stream_end) : Number(m.end) || 0;
      return [s, e];
    })
    .filter(([s, e]) => e > s)
    .sort((a, b) => a[0] - b[0]);

  // Сдвиг AUDIO→STREAM. Тишина (doc.silences) хранится в АУДИО-секундах, а рендер
  // режет исходный stream (АБСОЛЮТНЫЕ секунды). Мьюты содержат обе координаты,
  // поэтому offset = stream_start - start берём из любого мьюта, где есть обе.
  // Нет мьютов со сдвигом — значит режим без сдвига (audio == stream), offset=0.
  let offset = 0;
  for (const m of rawMutes) {
    if (Number.isFinite(m?.stream_start) && Number.isFinite(m?.start)) {
      offset = Number(m.stream_start) - Number(m.start);
      break;
    }
  }
  const rawSilences: any[] = Array.isArray(doc?.silences) ? doc.silences : [];
  const savedSilenceSegments = rawSilences
    .filter((s) => s?.status === "accepted")
    .map((s): [number, number] => {
      const st = (Number(s.start) || 0) + offset;
      const en = (Number(s.end) || 0) + offset;
      return [Math.round(st * 1000) / 1000, Math.round(en * 1000) / 1000];
    })
    .filter(([s, e]) => e > s)
    .sort((a, b) => a[0] - b[0]);

  // Диапазон обработки: range_in/range_out хранятся таймкодами ("HH:MM:SS.mmm")
  // в АБСОЛЮТНЫХ секундах исходника. Рендер обрежет вывод по этому окну.
  // parseTimecode → мс|null; null-граница = «без ограничения» с этой стороны.
  // range_in == 0 трактуем как «с начала» (null) — не создаёт лишнего клипа.
  const rangeInMs = typeof doc?.range_in === "string" ? parseTimecode(doc.range_in) : null;
  const rangeOutMs = typeof doc?.range_out === "string" ? parseTimecode(doc.range_out) : null;
  const rangeStartS = rangeInMs != null && rangeInMs > 0 ? rangeInMs / 1000 : null;
  const rangeEndS = rangeOutMs != null && rangeOutMs > 0 ? rangeOutMs / 1000 : null;

  // Highlights хранятся bundle'ом {highlights: [...], source, ...} (см.
  // merge_into_decisions), НЕ голым массивом. Старый Array.isArray(doc.highlights)
  // всегда давал 0. Читаем вложенный массив.
  const hlBundle = doc?.highlights;
  const hlArray: any[] = Array.isArray(hlBundle?.highlights)
    ? hlBundle.highlights
    : Array.isArray(hlBundle)
      ? hlBundle
      : [];

  return {
    mutes: rawMutes.length,
    cuts: Array.isArray(doc?.cuts) ? doc.cuts.length : 0,
    highlights: hlArray.length,
    highlightItems: hlArray,
    workdir: (meta.workdir as string | undefined) ?? null,
    // Путь к исходному видео. Ищем в нескольких местах — как TimelineScreen:
    // _meta.stream_path (Electron-схема) → source/stream/original (CLI-схема).
    // Иначе на CLI-decisions.json (без _meta) экспорт зря ругался «нет stream_path».
    stream:
      (meta.stream_path as string | undefined) ??
      (typeof doc?.source === "string" ? doc.source : undefined) ??
      (typeof doc?.stream === "string" ? doc.stream : undefined) ??
      (typeof doc?.original === "string" ? doc.original : undefined) ??
      null,
    mutedSegments,
    savedSilenceSegments,
    rangeStartS,
    rangeEndS,
  };
}

/** Суммарная длительность набора сегментов, мс (для человекочитаемого вывода). */
function segmentsDurationMs(segs: [number, number][]): number {
  return segs.reduce((sum, [s, e]) => sum + (e - s), 0) * 1000;
}

/** Секунды → mm:ss (для показа границ диапазона). */
function fmtS(secs: number): string {
  const m = Math.floor(secs / 60);
  const s = Math.floor(secs % 60);
  return `${m}:${String(s).padStart(2, "0")}`;
}

export function ExportScreen({ decisionsPath, onJobStarted }: ExportScreenProps) {
  const [form, setForm] = useState<FormState>(loadForm);
  const [summary, setSummary] = useState<Summary | null>(null);
  const [loadingSummary, setLoadingSummary] = useState(false);
  const [busy, setBusy] = useState<null | "vegas" | "render">(null);

  // --- рендер + удаление тишины ---------------------------------------------
  const [removeSilence, setRemoveSilence] = useState(false);
  // Пресет скорость↔качество энкодера. balanced по умолчанию — как на бэке.
  const [quality, setQuality] = useState<"fast" | "balanced" | "max">("balanced");
  // Однократно включаем «удалить тишину», если в проекте уже есть размеченная
  // в «Правке» тишина — иначе труд пользователя молча проигнорировался бы.
  // Гвард по decisionsPath, чтобы не перебивать ручное снятие галочки.
  const autoSilenceRef = useRef<string | null>(null);
  // job'ы рендера/детекта отслеживаем ЛОКАЛЬНО (нужен прогресс прямо в карточке),
  // а не только через глобальный onJobStarted — детект→рендер это два шага.
  const [silenceJobId, setSilenceJobId] = useState<string | null>(null);
  const [silenceResult, setSilenceResult] = useState<DetectSilenceResult | null>(null);
  const [renderJobId, setRenderJobId] = useState<string | null>(null);
  const [renderDone, setRenderDone] = useState<string | null>(null); // output_path

  const silenceJob = useJobEvents(
    silenceJobId,
    useCallback((state: JobState) => {
      if (state.status === "done" && state.result) {
        const r = state.result as unknown as DetectSilenceResult;
        setSilenceResult(r);
        toast.success(`Тишина: найдено ${r.count} сегментов`);
      } else if (state.status === "failed") {
        toast.error("Детект тишины упал", { description: state.error ?? undefined });
      }
    }, [])
  );

  const renderJob = useJobEvents(
    renderJobId,
    useCallback((state: JobState) => {
      if (state.status === "done" && state.result) {
        const out = String((state.result as Record<string, unknown>).output_path ?? "");
        setRenderDone(out);
        setBusy(null);
        toast.success("Рендер готов", {
          description: out,
          action: out
            ? {
                label: "Показать",
                onClick: async () => {
                  const ok = await window.twitchCut.showInFolder(out);
                  if (!ok) toast.error("Не удалось открыть каталог");
                },
              }
            : undefined,
        });
      } else if (state.status === "failed") {
        setBusy(null);
        toast.error("Рендер упал", { description: state.error ?? undefined });
      }
    }, [])
  );

  useEffect(() => {
    localStorage.setItem(LS_KEY, JSON.stringify(form));
  }, [form]);

  useEffect(() => {
    let alive = true;
    setLoadingSummary(true);
    readDecisions(decisionsPath)
      .then((doc) => {
        if (!alive) return;
        setSummary(extractSummary(doc));
      })
      .catch(() => {
        if (!alive) return;
        setSummary(null);
      })
      .finally(() => alive && setLoadingSummary(false));
    return () => {
      alive = false;
    };
  }, [decisionsPath]);

  // Есть размеченная в «Правке» тишина → включаем «удалить тишину» по умолчанию
  // (один раз на проект). Пользователь может снять галочку — повторно не жмём.
  useEffect(() => {
    if (!summary) return;
    if (autoSilenceRef.current === decisionsPath) return;
    autoSilenceRef.current = decisionsPath;
    if (summary.savedSilenceSegments.length > 0) setRemoveSilence(true);
  }, [summary, decisionsPath]);

  // Дефолтные пути в save-dialog'ах — рядом с decisions.json. Разделитель
  // угадываем по самому пути, чтобы работало и на Windows, и на POSIX.
  const { sep, dirOf } = useMemo(() => {
    const s = decisionsPath.includes("\\") ? "\\" : "/";
    const idx = decisionsPath.lastIndexOf(s);
    return { sep: s, dirOf: idx > 0 ? decisionsPath.slice(0, idx) : "" };
  }, [decisionsPath]);

  const vegasDefault = dirOf ? `${dirOf}${sep}vegas.cs` : undefined;
  const renderDefault = dirOf ? `${dirOf}${sep}render.mp4` : undefined;

  // Тишина из «Правки» уже отревьюена — при наличии используем ЕЁ напрямую,
  // без повторного детекта. Отдельный детект остаётся фолбэком (в проекте нет
  // сохранённой тишины, либо пользователь хочет перепроверить с нуля).
  const hasSavedSilence = (summary?.savedSilenceSegments.length ?? 0) > 0;

  // Итоговая тишина на вырезание: сохранённая из Правки ИЛИ (фолбэк) детект.
  const silenceSegments = useMemo<[number, number][]>(() => {
    if (!removeSilence) return [];
    if (hasSavedSilence) return summary!.savedSilenceSegments;
    return silenceResult ? silenceResult.segments : [];
  }, [removeSilence, hasSavedSilence, summary, silenceResult]);

  // Финальный список сегментов на вырезание: заглушки + (опц.) тишина.
  // Сортируем по началу — render_video ожидает упорядоченные непересекающиеся
  // интервалы (бэкенд их валидирует, но клиенту дешевле отдать сразу чистые).
  const renderSegments = useMemo<[number, number][]>(() => {
    const base = summary?.mutedSegments ?? [];
    return [...base, ...silenceSegments].sort((a, b) => a[0] - b[0]);
  }, [summary, silenceSegments]);

  // Что останется в итоге. При заданном диапазоне база — длина окна
  // (range_out−range_in), иначе неизвестна (весь файл) → null, не показываем.
  const keptDurationMs = useMemo<number | null>(() => {
    if (!summary || summary.rangeEndS == null) return null;
    const windowS = summary.rangeEndS - (summary.rangeStartS ?? 0);
    // Вырезаем только то, что попадает в окно.
    const lo = summary.rangeStartS ?? 0;
    const hi = summary.rangeEndS;
    const removedS = renderSegments.reduce((sum, [s, e]) => {
      const cs = Math.max(s, lo);
      const ce = Math.min(e, hi);
      return sum + Math.max(0, ce - cs);
    }, 0);
    return Math.max(0, (windowS - removedS) * 1000);
  }, [summary, renderSegments]);

  const silenceRunning =
    silenceJob.state?.status === "running" || silenceJob.state?.status === "pending";
  const renderRunning =
    renderJob.state?.status === "running" || renderJob.state?.status === "pending";

  const setF = <K extends keyof FormState>(k: K, v: FormState[K]) =>
    setForm((s) => ({ ...s, [k]: v }));

  /**
   * После успешного создания job'а — тост с «Показать» на итоговый файл/каталог.
   * showInFolder работает и по несуществующему пути (открывает родителя) — так
   * что клик до завершения job'а не сломается.
   */
  const toastReveal = (path: string, label: string) => {
    toast.success(label, {
      description: path,
      action: {
        label: "Показать",
        onClick: async () => {
          const ok = await window.twitchCut.showInFolder(path);
          if (!ok) toast.error("Не удалось открыть каталог");
        },
      },
    });
  };

  const runVegas = async () => {
    if (!form.vegasPath) {
      toast.error("Укажи, куда сохранить .cs");
      return;
    }
    setBusy("vegas");
    try {
      const job = await createExportVegasJob({
        decisions: decisionsPath,
        vegas: form.vegasPath,
      });
      toastReveal(form.vegasPath, "Vegas-скрипт собирается");
      onJobStarted(job.id);
    } catch (e) {
      toast.error("Ошибка запуска", { description: String(e) });
    } finally {
      setBusy(null);
    }
  };

  /** Запустить detect-silence job. Результат приедет через useJobEvents→onFinal. */
  const runDetectSilence = async () => {
    if (!summary?.stream) {
      toast.error("Не найден путь к видео (stream_path) в decisions.json");
      return;
    }
    setSilenceResult(null);
    setSilenceJobId(null);
    try {
      const req: DetectSilenceRequest = {
        video_path: summary.stream,
        threshold_db: -30,
        min_duration: 0.5,
        merge_gap: 0.5,
        padding_before: 0.1,
        padding_after: 0.1,
      };
      const job = await createDetectSilenceJob(req);
      setSilenceJobId(job.id);
    } catch (e) {
      toast.error("Не удалось запустить детект тишины", { description: String(e) });
    }
  };

  /** Запустить финальный рендер: вырезать заглушки (+ опц. тишину) из stream. */
  const runRender = async () => {
    if (!summary?.stream) {
      toast.error("Не найден путь к видео (stream_path) в decisions.json");
      return;
    }
    if (!form.renderOutputPath) {
      toast.error("Укажи, куда сохранить итоговое видео");
      return;
    }
    if (renderSegments.length === 0) {
      toast.error("Нет сегментов на вырезание — ни заглушек, ни тишины");
      return;
    }
    setRenderDone(null);
    setRenderJobId(null);
    setBusy("render");
    try {
      const req: RenderRequest = {
        input_path: summary.stream,
        output_path: form.renderOutputPath,
        segments_to_remove: renderSegments,
        quality,
        range_start: summary.rangeStartS,
        range_end: summary.rangeEndS,
        force: true,
      };
      const job = await createRenderJob(req);
      setRenderJobId(job.id);
      onJobStarted(job.id);
    } catch (e) {
      setBusy(null);
      toast.error("Не удалось запустить рендер", { description: String(e) });
    }
  };

  return (
    <div className="mx-auto max-w-6xl space-y-6 p-8">
      <header className="space-y-1">
        <div className="flex items-center gap-2 text-[11px] font-medium uppercase tracking-wider text-subtle">
          <Send className="h-3.5 w-3.5" /> Экспорт
        </div>
        <h1 className="text-2xl font-semibold tracking-tight text-fg">
          Выгрузка проекта
        </h1>
        <p className="max-w-2xl text-sm text-muted">
          Соберите скрипт для Sony Vegas Pro — это перенесёт все заглушенные
          места прямо на таймлайн. Либо соберите финальное видео здесь же.
        </p>
      </header>

      <Card variant="glass" padding="md" className="flex flex-wrap items-center gap-4">
        <div className="min-w-0 flex-1">
          <div className="text-[11px] uppercase tracking-wider text-subtle">
            Источник
          </div>
          <div className="truncate font-mono text-xs text-fg">{decisionsPath}</div>
        </div>
        <div className="flex items-center gap-3 text-xs">
          <Stat color="text-err" label="Мутов" value={summary?.mutes ?? "—"} />
          <Stat color="text-warn" label="Вырезов" value={summary?.cuts ?? "—"} />
          <Stat color="text-brand-from" label="Хайлайтов" value={summary?.highlights ?? "—"} />
        </div>
        {summary?.workdir && (
          <Button
            variant="ghost"
            size="sm"
            onClick={async () => {
              const ok = await window.twitchCut.showInFolder(summary.workdir!);
              if (!ok) toast.error("Не удалось открыть workdir");
            }}
          >
            <FolderOpen className="h-3.5 w-3.5" /> Открыть workdir
          </Button>
        )}
        {loadingSummary && (
          <span className="text-[11px] text-subtle">загрузка…</span>
        )}
      </Card>

      {/* Главное действие — скрипт для Vegas. На всю ширину, с крупной кнопкой. */}
      <Card
        variant="elevated"
        padding="lg"
        className="flex flex-col gap-5 border-warn/20 md:flex-row md:items-center"
      >
        <div className="flex-1 space-y-3">
          <CardHead
            icon={<FileCode2 className="h-4 w-4" />}
            gradient="from-amber-500/25 to-amber-500/0"
            iconColor="text-warn"
            title="Скрипт для Sony Vegas"
            desc="Перенесёт все заглушенные места на таймлайн Vegas. Откройте свой проект в Vegas, запустите скрипт — регионы расставятся сами."
          />
          <div className="space-y-2">
            <Label eyebrow>Куда сохранить</Label>
            <FileField
              kind="save"
              value={form.vegasPath}
              onChange={(v) => setF("vegasPath", v)}
              defaultPath={vegasDefault}
              filters={[{ name: "Vegas script", extensions: ["cs"] }]}
              placeholder="…/vegas.cs"
            />
          </div>
        </div>
        <Button
          size="lg"
          className="shrink-0 md:self-end"
          onClick={runVegas}
          disabled={!form.vegasPath}
          loading={busy === "vegas"}
        >
          <Download className="h-4 w-4" /> Собрать скрипт для Vegas
        </Button>
      </Card>

      {/* Финальный рендер с опциональным удалением тишины */}
      <Card
        variant="elevated"
        padding="lg"
        className="flex flex-col gap-5 border-green-500/20"
      >
        <div className="space-y-4">
          <CardHead
            icon={<Video className="h-4 w-4" />}
            gradient="from-green-500/25 to-green-500/0"
            iconColor="text-green-400"
            title="Финальный рендер"
            desc="Вырезать заглушки и опционально тишину, смонтировать готовое видео."
          />

          {/* Опция: удалить тишину */}
          <div className="rounded-lg bg-black/20 p-4">
            <label className="flex cursor-pointer items-start gap-3">
              <input
                type="checkbox"
                checked={removeSilence}
                onChange={(e) => {
                  setRemoveSilence(e.target.checked);
                  if (!e.target.checked) {
                    setSilenceJobId(null);
                    setSilenceResult(null);
                  }
                }}
                className="mt-0.5 h-4 w-4 rounded border-white/20 bg-black/30 text-primary focus:ring-2 focus:ring-primary focus:ring-offset-0"
              />
              <div className="flex-1">
                <span className="text-sm font-medium text-fg">Удалить тишину</span>
                {hasSavedSilence ? (
                  <p className="mt-0.5 text-xs text-muted">
                    Возьмём паузы, которые вы отметили в «Правке» — заново
                    анализировать не нужно.
                  </p>
                ) : (
                  <p className="mt-0.5 text-xs text-muted">
                    В «Правке» тишина не размечена. Можно найти паузы
                    автоматически (порог -30dB, мин. 0.5с).
                  </p>
                )}

                {/* Есть отревьюенная тишина из Правки — показываем её, без детекта. */}
                {removeSilence && hasSavedSilence && (
                  <div className="mt-2 rounded bg-amber-500/10 px-2 py-1.5 text-xs text-amber-200">
                    Из «Правки»: {summary!.savedSilenceSegments.length} сегментов ·{" "}
                    {fmtMs(segmentsDurationMs(summary!.savedSilenceSegments))} вырежем
                  </div>
                )}

                {/* Фолбэк: сохранённой тишины нет — предлагаем разовый детект. */}
                {removeSilence && !hasSavedSilence && !silenceResult && (
                  <Button
                    size="sm"
                    variant="secondary"
                    className="mt-2"
                    onClick={runDetectSilence}
                    disabled={silenceRunning}
                    loading={silenceRunning}
                  >
                    <Sparkles className="h-3.5 w-3.5" /> Найти паузы
                  </Button>
                )}
                {removeSilence && !hasSavedSilence && silenceResult && (
                  <div className="mt-2 rounded bg-amber-500/10 px-2 py-1.5 text-xs text-amber-200">
                    Найдено {silenceResult.count} сегментов ·{" "}
                    {fmtMs(silenceResult.total_duration * 1000)} вырежем
                  </div>
                )}
                {removeSilence && !hasSavedSilence &&
                  silenceJob.state?.status === "running" && (
                    <div className="mt-2 flex items-center gap-2 text-xs text-muted">
                      <Loader2 className="h-3.5 w-3.5 animate-spin" />
                      <span>
                        {silenceJob.state.stage || "детект"} ·{" "}
                        {Math.round(silenceJob.state.progress)}%
                      </span>
                    </div>
                  )}
              </div>
            </label>
          </div>

          {/* Диапазон обработки: показываем, если range_out задан (не весь файл). */}
          {summary?.rangeEndS != null && (
            <div className="rounded-lg bg-blue-500/10 px-3 py-2 text-xs text-blue-200">
              Рендерим только обработанный диапазон: {fmtS(summary.rangeStartS ?? 0)}–{fmtS(summary.rangeEndS!)} · ≈{fmtMs((summary.rangeEndS! - (summary.rangeStartS ?? 0)) * 1000)} исходника — вне диапазона не попадёт в итог.
            </div>
          )}

          {/* Пресет скорость↔качество энкодера */}
          <div className="rounded-lg bg-black/20 p-3">
            <div className="mb-2 text-[11px] font-medium uppercase tracking-wider text-subtle">
              Скорость / Качество
            </div>
            <div className="flex gap-1.5">
              {(["fast", "balanced", "max"] as const).map((q) => (
                <button
                  key={q}
                  type="button"
                  onClick={() => setQuality(q)}
                  className={[
                    "flex-1 rounded-md px-2 py-1.5 text-xs font-medium transition-colors",
                    quality === q
                      ? "bg-surface text-fg shadow-[0_1px_0_rgb(255_255_255/0.05)_inset]"
                      : "bg-white/5 text-muted hover:bg-white/10 hover:text-fg",
                  ].join(" ")}
                >
                  {q === "fast" ? "⚡ Быстро" : q === "balanced" ? "⚖ Баланс" : "🎯 Качество"}
                </button>
              ))}
            </div>
            <div className="mt-1.5 text-[10px] text-subtle">
              {quality === "fast"
                ? "Ниже битрейт, быстрый пресет. Для черновой проверки."
                : quality === "balanced"
                ? "Золотая середина. Хорошее качество за разумное время."
                : "Максимальное качество. Медленнее, но чище картинка."}
            </div>
          </div>

          {/* Итоговая статистика */}
          <div className="rounded-lg bg-black/20 px-4 py-3">
            <div className="flex items-baseline gap-2 text-xs text-muted">
              <span>Заглушки:</span>
              <span className="font-mono text-sm text-fg">
                {summary?.mutedSegments.length ?? 0}
              </span>
              <span>·</span>
              <span>{fmtMs(segmentsDurationMs(summary?.mutedSegments ?? []))}</span>
            </div>
            {removeSilence && silenceSegments.length > 0 && (
              <div className="mt-1 flex items-baseline gap-2 text-xs text-muted">
                <span>Тишина:</span>
                <span className="font-mono text-sm text-amber-400">
                  {silenceSegments.length}
                </span>
                <span>·</span>
                <span>{fmtMs(segmentsDurationMs(silenceSegments))}</span>
              </div>
            )}
            <div className="mt-2 flex items-baseline gap-2 border-t border-white/8 pt-2 text-xs font-semibold text-fg">
              <span>Всего вырежем:</span>
              <span className="font-mono text-sm">{renderSegments.length}</span>
              <span>·</span>
              <span>{fmtMs(segmentsDurationMs(renderSegments))}</span>
            </div>
            {keptDurationMs != null && (
              <div className="mt-1 flex items-baseline gap-2 text-xs text-green-300">
                <span>Останется:</span>
                <span className="font-mono text-sm">≈{fmtMs(keptDurationMs)}</span>
              </div>
            )}
          </div>

          {/* Путь выхода */}
          <div className="space-y-2">
            <Label eyebrow>Итоговое видео</Label>
            <FileField
              kind="save"
              value={form.renderOutputPath}
              onChange={(v) => setF("renderOutputPath", v)}
              defaultPath={renderDefault}
              filters={[
                { name: "MP4 video", extensions: ["mp4"] },
                { name: "Все", extensions: ["*"] },
              ]}
              placeholder="…/render.mp4"
            />
          </div>

          {/* Прогресс рендера */}
          {renderJob.state?.status === "running" && (
            <div className="flex items-center gap-2 rounded-lg bg-green-500/10 px-3 py-2 text-xs text-green-200">
              <Loader2 className="h-4 w-4 animate-spin" />
              <span>
                {renderJob.state.stage || "render"} ·{" "}
                {Math.round(renderJob.state.progress)}%
              </span>
            </div>
          )}

          {/* Результат */}
          {renderDone && (
            <div className="rounded-lg bg-green-500/10 px-3 py-2 text-xs text-green-200">
              Готово: {renderDone}
            </div>
          )}
        </div>

        <Button
          size="lg"
          onClick={runRender}
          disabled={
            !form.renderOutputPath ||
            renderRunning ||
            (removeSilence && !hasSavedSilence && !silenceResult) ||
            renderSegments.length === 0
          }
          loading={busy === "render"}
        >
          <Video className="h-4 w-4" /> Рендер
        </Button>
      </Card>

      {/* Нарезка на клипы и NotebookLM переехали во вкладку «AI Tiktok». */}
      <Card variant="surface" padding="md" className="flex items-center gap-3">
        <div className="inline-flex h-9 w-9 shrink-0 items-center justify-center rounded-lg bg-gradient-to-br from-violet-500/25 to-violet-500/0 text-brand-from">
          <Sparkles className="h-4 w-4" />
        </div>
        <div className="min-w-0 flex-1">
          <div className="text-sm font-medium text-fg">Нарезка на клипы и хайлайты</div>
          <div className="text-xs text-muted">
            Сборка пакета NotebookLM, импорт моментов и нарезка коротких клипов теперь
            во вкладке «AI Tiktok» слева.
          </div>
        </div>
      </Card>
    </div>
  );
}

function CardHead({
  icon,
  gradient,
  iconColor,
  title,
  desc,
}: {
  icon: React.ReactNode;
  gradient: string;
  iconColor: string;
  title: string;
  desc: string;
}) {
  return (
    <div>
      <div
        className={`mb-3 inline-flex h-9 w-9 items-center justify-center rounded-lg bg-gradient-to-br ${gradient} ${iconColor}`}
      >
        {icon}
      </div>
      <CardTitle>{title}</CardTitle>
      <CardDescription className="mt-1">{desc}</CardDescription>
    </div>
  );
}

function Stat({
  label,
  value,
  color,
}: {
  label: string;
  value: number | string;
  color: string;
}) {
  return (
    <div className="flex items-baseline gap-1.5">
      <span className={`font-mono text-sm font-semibold ${color}`}>{value}</span>
      <span className="text-[11px] uppercase tracking-wider text-subtle">
        {label}
      </span>
    </div>
  );
}
