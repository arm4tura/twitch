import { useCallback, useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import {
  ChevronRight,
  FileText,
  FileVideo,
  FolderOpen,
  Play,
  Rocket,
  Settings2,
  Sparkles,
  UploadCloud,
  Volume2,
  X,
} from "lucide-react";
import { createProcessJob, getSettings, type JobState } from "../api";
import { Button } from "../components/ui/Button";
import { Card, CardDescription, CardTitle } from "../components/ui/Card";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "../components/ui/Collapsible";
import { FileField } from "../components/ui/FileField";
import { Input, Label, Select } from "../components/ui/Input";
import { HelpTip } from "../components/ui/Tooltip";
import { cn } from "../lib/cn";

/**
 * Опции для select-полей продвинутых настроек. Списки отражают то, что реально
 * поддерживает WhisperX/pipeline'ы. Держим тут же — это не переиспользуется
 * нигде ещё; если понадобится — вынесем в constants/.
 */
const MODEL_OPTIONS = [
  { value: "tiny", label: "tiny — самая быстрая, низкое качество" },
  { value: "base", label: "base" },
  { value: "small", label: "small" },
  { value: "medium", label: "medium" },
  { value: "large-v2", label: "large-v2" },
  { value: "large-v3", label: "large-v3 — лучшее качество, медленно" },
];
// Движок распознавания речи. GigaAM v3 — русскоязычный дефолт (быстрее, не
// требует cuDNN-стека). WhisperX — мультиязычный, с тонкими VAD-параметрами.
const TRANSCRIBER_OPTIONS = [
  { value: "gigaam", label: "GigaAM v3 — русский, по умолчанию" },
  { value: "whisperx", label: "WhisperX — мультиязычный, тонкая настройка" },
];
const LANGUAGE_OPTIONS = [
  { value: "ru", label: "ru — русский" },
  { value: "en", label: "en — английский" },
  { value: "uk", label: "uk — украинский" },
  { value: "auto", label: "auto — автоопределение" },
];
const DEVICE_OPTIONS = [
  { value: "cuda", label: "cuda — GPU (NVIDIA)" },
  { value: "cpu", label: "cpu" },
];
const COMPUTE_OPTIONS = [
  { value: "float16", label: "float16 — быстрее, нужно ≥ 8 GB VRAM" },
  { value: "int8", label: "int8 — экономно, годится для 4-6 GB VRAM" },
  { value: "float32", label: "float32 — максимум точности, медленно" },
];
const VAD_OPTIONS = [
  { value: "pyannote", label: "pyannote — точнее, но медленнее" },
  { value: "silero", label: "silero — быстрый VAD" },
];

/**
 * FieldLabel — Label с иконкой «?» справа. Локальный хелпер для NewJobScreen:
 * форма насыщенная, много терминов (VAD, compute_type, batch_size), tooltip
 * рядом с каждым сильно снижает cognitive load.
 */
function FieldLabel({
  children,
  help,
}: {
  children: React.ReactNode;
  help: React.ReactNode;
}) {
  return (
    <div className="mb-1.5 flex items-center gap-1.5">
      <Label className="mb-0">{children}</Label>
      <HelpTip>{help}</HelpTip>
    </div>
  );
}

/**
 * NewJobScreen — запуск обработки записи. Два режима:
 *
 * - «Обычный» (по умолчанию): пользователь бросает запись стрима в окно (или
 *   выбирает файл) и жмёт «Обработать». Всё остальное — словарь мата, рабочая
 *   папка, пути выходных файлов, устройство (GPU/CPU) — подставляет бэкенд.
 *   Это путь для человека, который «не разбирается».
 * - «Расширенный»: полная форма со всеми путями и параметрами WhisperX/VAD.
 *   Для тех, кому нужен свой словарь, диапазон, модель и т.п.
 *
 * Обе ветки делят одно состояние формы (переключение режима ничего не теряет)
 * и общий submit; отличается только собираемый payload.
 */

const LS_KEY = "twitchCut.newJob.v1";
const LS_MODE = "twitchCut.newJob.mode";

type Mode = "simple" | "advanced";

interface FormState {
  stream: string;
  original: string;
  banwords: string;
  workdir: string;
  decisions: string;
  vegas: string;
  // advanced
  transcriber: string;
  range_in: string;
  range_out: string;
  model: string;
  language: string;
  device: string;
  compute_type: string;
  batch_size: string;
  vad_method: string;
  vad_filter: boolean;
  mock_transcript: string;
}

const DEFAULT_FORM: FormState = {
  stream: "",
  original: "",
  banwords: "",
  workdir: "",
  decisions: "",
  vegas: "",
  transcriber: "gigaam",
  range_in: "",
  range_out: "",
  model: "large-v3",
  language: "ru",
  device: "cuda",
  compute_type: "float16",
  batch_size: "16",
  vad_method: "pyannote",
  vad_filter: true,
  mock_transcript: "",
};

function loadForm(): FormState {
  try {
    const raw = localStorage.getItem(LS_KEY);
    if (!raw) return DEFAULT_FORM;
    const parsed = JSON.parse(raw);
    return { ...DEFAULT_FORM, ...parsed };
  } catch {
    return DEFAULT_FORM;
  }
}

function saveForm(form: FormState) {
  try {
    localStorage.setItem(LS_KEY, JSON.stringify(form));
  } catch {
    /* private mode / quota — ok, форма всё ещё живёт в памяти сессии */
  }
}

function loadMode(): Mode {
  try {
    return localStorage.getItem(LS_MODE) === "advanced" ? "advanced" : "simple";
  } catch {
    return "simple";
  }
}

/** Короткое имя файла из полного пути (для отображения выбранной записи). */
function baseName(p: string): string {
  const parts = p.replace(/\\/g, "/").split("/");
  return parts[parts.length - 1] || p;
}

function parseTimecodeMs(v: string): number | null {
  if (!v) return null;
  const parts = v.split(":");
  if (parts.length < 2 || parts.length > 3) return null;
  const [hStr, mStr, sPart] = parts.length === 3 ? parts : ["0", parts[0], parts[1]];
  const [sStr, msStr = "0"] = sPart.split(".");
  const h = Number(hStr);
  const m = Number(mStr);
  const s = Number(sStr);
  const ms = Number(msStr.padEnd(3, "0").slice(0, 3));
  if ([h, m, s, ms].some((n) => Number.isNaN(n))) return null;
  return ((h * 3600 + m * 60 + s) * 1000) + ms;
}

function fmtDurationMs(ms: number): string {
  const s = Math.round(ms / 1000);
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const sec = s % 60;
  return h > 0
    ? `${h}ч ${m}мин`
    : m > 0
      ? `${m}мин ${sec}с`
      : `${sec}с`;
}

const VIDEO_EXT = ["mp4", "mkv", "mov", "ts", "avi", "webm"];

export function NewJobScreen({
  onJobStarted,
}: {
  onJobStarted: (jobId: string, decisionsPath?: string) => void;
}) {
  const [mode, setMode] = useState<Mode>(() => loadMode());
  const [form, setForm] = useState<FormState>(() => loadForm());
  const [advOpen, setAdvOpen] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const set = useCallback(
    <K extends keyof FormState>(k: K, v: FormState[K]) =>
      setForm((f) => ({ ...f, [k]: v })),
    []
  );

  const switchMode = useCallback((m: Mode) => {
    setMode(m);
    try {
      localStorage.setItem(LS_MODE, m);
    } catch {
      /* ok */
    }
  }, []);

  // Debounced-persist в localStorage.
  useEffect(() => {
    const t = window.setTimeout(() => saveForm(form), 400);
    return () => window.clearTimeout(t);
  }, [form]);

  // Подтянуть дефолты из настроек: движок распознавания и device/compute_type.
  // На машине без GPU сервер отдаёт cpu/int8 (TWITCH_CUT_CPU=1) — иначе форма
  // предлагала бы заведомо падающий cuda. Применяем только к незаполненной
  // пользователем форме (нет сохранённого выбора), чтобы не перетирать выбор.
  useEffect(() => {
    const hasSaved = (() => {
      try {
        return !!localStorage.getItem(LS_KEY);
      } catch {
        return false;
      }
    })();
    if (hasSaved) return;
    let cancelled = false;
    getSettings()
      .then((s) => {
        if (cancelled) return;
        const tr = s.default_transcriber;
        const dev = s.default_device;
        const ct = s.default_compute_type;
        setForm((f) => ({
          ...f,
          transcriber: typeof tr === "string" ? tr : f.transcriber,
          device: typeof dev === "string" ? dev : f.device,
          compute_type: typeof ct === "string" ? ct : f.compute_type,
        }));
      })
      .catch(() => {
        /* backend ещё не готов — оставляем дефолты формы */
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const requiredFilled =
    !!form.stream && !!form.banwords;

  const rangeMs = useMemo(() => {
    const a = parseTimecodeMs(form.range_in);
    const b = parseTimecodeMs(form.range_out);
    if (a == null || b == null || b <= a) return null;
    return b - a;
  }, [form.range_in, form.range_out]);

  // decisions/vegas/workdir больше не заполняются в форме — бэкенд сам создаёт
  // подпапку в work/ и кладёт туда decisions.json + mutes.cs.

  /**
   * Запуск обработки.
   *
   * В простом режиме шлём только `stream` — бэкенд сам подставит словарь мата,
   * рабочую папку (подпапка в work/), пути выходов, оригинал и устройство.
   * В расширенном — те же входы плюс выбор движка и его параметры; workdir и
   * пути выходов бэкенд по-прежнему создаёт сам. Навигацию на job'у делаем без
   * decisionsPath — JobScreen возьмёт итоговый путь из результата после завершения.
   */
  const submit = async (simple: boolean) => {
    setError(null);
    setSubmitting(true);
    try {
      const whisperx = form.transcriber === "whisperx";
      const req = simple
        ? { stream: form.stream }
        : {
            stream: form.stream,
            banwords: form.banwords,
            // Оригинал видео скрыт в UI — шлём только если раньше был выбран.
            ...(form.original ? { original: form.original } : {}),
            range_in: form.range_in || null,
            range_out: form.range_out || null,
            transcriber: form.transcriber,
            // GigaAM-ветка тонких параметров не имеет — они относятся к WhisperX,
            // и бэкенд их игнорирует, если движок gigaam. Шлём только когда нужно.
            ...(whisperx
              ? {
                  model: form.model,
                  language: form.language,
                  device: form.device,
                  compute_type: form.compute_type,
                  batch_size: Number(form.batch_size) || 16,
                  vad_method: form.vad_method,
                  vad_filter: form.vad_filter,
                }
              : {}),
            mock_transcript: form.mock_transcript || null,
          };
      const state: JobState = await createProcessJob(req);
      onJobStarted(state.id);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className="mx-auto flex max-w-7xl flex-col gap-6 p-8">
      <header className="flex flex-wrap items-start justify-between gap-4">
        <div>
          <h1 className="text-2xl font-semibold tracking-tight text-fg">
            Новая обработка
          </h1>
          <p className="mt-1 text-sm text-muted">
            {mode === "simple"
              ? "Выберите запись стрима — приложение найдёт мат и подготовит разметку для Vegas Pro."
              : "Полный контроль: свои файлы, словарь, диапазон и параметры распознавания."}
          </p>
        </div>
        <ModeToggle mode={mode} onChange={switchMode} />
      </header>

      {mode === "simple" ? (
        <SimpleForm
          stream={form.stream}
          onStream={(v) => set("stream", v)}
          submitting={submitting}
          error={error}
          onSubmit={() => submit(true)}
        />
      ) : (
        <AdvancedForm
          form={form}
          set={set}
          advOpen={advOpen}
          setAdvOpen={setAdvOpen}
          requiredFilled={requiredFilled}
          rangeMs={rangeMs}
          submitting={submitting}
          error={error}
          onSubmit={() => submit(false)}
        />
      )}
    </div>
  );
}

// ============================================================================
// Переключатель режима
// ============================================================================

function ModeToggle({ mode, onChange }: { mode: Mode; onChange: (m: Mode) => void }) {
  return (
    <div className="inline-flex rounded-lg border border-white/10 bg-black/20 p-0.5 text-sm">
      {(["simple", "advanced"] as Mode[]).map((m) => (
        <button
          key={m}
          type="button"
          onClick={() => onChange(m)}
          className={cn(
            "rounded-md px-3 py-1.5 font-medium transition-colors",
            "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-brand-from",
            mode === m ? "bg-white/[0.08] text-fg" : "text-muted hover:text-fg"
          )}
        >
          {m === "simple" ? "Обычный" : "Расширенный"}
        </button>
      ))}
    </div>
  );
}

// ============================================================================
// Простой режим
// ============================================================================

function SimpleForm({
  stream,
  onStream,
  submitting,
  error,
  onSubmit,
}: {
  stream: string;
  onStream: (v: string) => void;
  submitting: boolean;
  error: string | null;
  onSubmit: () => void;
}) {
  const [dragOver, setDragOver] = useState(false);
  const dragDepth = useRef(0);

  const pick = async () => {
    const picked = await window.twitchCut.openFile({
      filters: [{ name: "Видео", extensions: VIDEO_EXT }],
      title: "Выберите запись стрима",
    });
    if (picked) onStream(picked);
  };

  const onDrop = (e: React.DragEvent) => {
    e.preventDefault();
    dragDepth.current = 0;
    setDragOver(false);
    const file = e.dataTransfer.files?.[0];
    if (!file) return;
    try {
      const p = window.twitchCut.getPathForFile(file);
      if (p) onStream(p);
    } catch {
      /* не удалось получить путь — пользователь может выбрать вручную */
    }
  };

  return (
    <div className="mx-auto flex w-full max-w-2xl flex-col gap-5">
      {/* Drop-зона */}
      <div
        role="button"
        tabIndex={0}
        onClick={pick}
        onKeyDown={(e) => (e.key === "Enter" || e.key === " ") && pick()}
        onDragEnter={(e) => {
          e.preventDefault();
          dragDepth.current += 1;
          setDragOver(true);
        }}
        onDragOver={(e) => e.preventDefault()}
        onDragLeave={(e) => {
          e.preventDefault();
          dragDepth.current -= 1;
          if (dragDepth.current <= 0) setDragOver(false);
        }}
        onDrop={onDrop}
        className={cn(
          "group flex cursor-pointer flex-col items-center justify-center gap-4 rounded-2xl border-2 border-dashed px-6 py-16 text-center transition-colors",
          "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-brand-from",
          dragOver
            ? "border-brand-from bg-brand-from/10"
            : "border-white/12 bg-surface/30 hover:border-white/25 hover:bg-surface/50"
        )}
      >
        {stream ? (
          <>
            <div className="flex h-14 w-14 items-center justify-center rounded-2xl bg-brand/15 text-brand-from">
              <FileVideo className="h-7 w-7" />
            </div>
            <div className="min-w-0">
              <div className="truncate text-base font-medium text-fg" title={stream}>
                {baseName(stream)}
              </div>
              <div className="mt-1 truncate font-mono text-[11px] text-subtle" title={stream}>
                {stream}
              </div>
            </div>
            <button
              type="button"
              onClick={(e) => {
                e.stopPropagation();
                onStream("");
              }}
              className="inline-flex items-center gap-1 rounded-md px-2 py-1 text-xs text-muted transition-colors hover:bg-white/5 hover:text-fg"
            >
              <X className="h-3.5 w-3.5" /> Выбрать другую
            </button>
          </>
        ) : (
          <>
            <div className="flex h-14 w-14 items-center justify-center rounded-2xl bg-white/5 text-muted transition-colors group-hover:text-brand-from">
              <UploadCloud className="h-7 w-7" />
            </div>
            <div>
              <div className="text-base font-medium text-fg">
                Перетащите запись стрима сюда
              </div>
              <div className="mt-1 text-sm text-muted">
                или нажмите, чтобы выбрать файл ({VIDEO_EXT.slice(0, 4).join(", ")}…)
              </div>
            </div>
          </>
        )}
      </div>

      {/* Пояснение — что произойдёт, простым языком */}
      <Card variant="glass" padding="md">
        <div className="mb-3 flex items-center gap-2">
          <Sparkles className="h-4 w-4 text-brand-from" />
          <CardTitle>Что произойдёт</CardTitle>
        </div>
        <ol className="flex flex-col gap-2 text-sm text-muted">
          <PreviewLine enabled={!!stream} text="Распознавание речи в записи" />
          <PreviewLine enabled={!!stream} text="Поиск мата по встроенному словарю" />
          <PreviewLine enabled={!!stream} text="Готовая разметка + скрипт для Vegas Pro" />
        </ol>
        <p className="mt-3 text-xs text-subtle">
          Результат появится в разделе «Проекты». Ничего настраивать не нужно —
          словарь и папки приложение подберёт само.
        </p>
      </Card>

      {error && <ErrorCard error={error} />}

      <Button
        size="lg"
        onClick={onSubmit}
        loading={submitting}
        disabled={!stream}
        className="w-full"
      >
        Обработать <ChevronRight className="h-4 w-4" />
      </Button>
      {!stream && (
        <p className="text-center text-xs text-subtle">
          Сначала выберите запись стрима.
        </p>
      )}
    </div>
  );
}

// ============================================================================
// Расширенный режим (прежняя полная форма)
// ============================================================================

function AdvancedForm({
  form,
  set,
  advOpen,
  setAdvOpen,
  requiredFilled,
  rangeMs,
  submitting,
  error,
  onSubmit,
}: {
  form: FormState;
  set: <K extends keyof FormState>(k: K, v: FormState[K]) => void;
  advOpen: boolean;
  setAdvOpen: (v: boolean) => void;
  requiredFilled: boolean;
  rangeMs: number | null;
  submitting: boolean;
  error: string | null;
  onSubmit: () => void;
}) {
  return (
    <div className="grid grid-cols-1 gap-6 xl:grid-cols-[minmax(0,1fr)_320px]">
      {/* Форма */}
      <div className="flex flex-col gap-5">
        {/* --- Входы --- */}
        <Card variant="elevated">
          <div className="mb-4 flex items-start gap-3">
            <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-brand/10 text-brand-from">
              <FolderOpen className="h-4 w-4" />
            </div>
            <div>
              <CardTitle>Файлы</CardTitle>
              <CardDescription>
                Запись стрима и словарь мата. Папку для результата приложение
                создаст само.
              </CardDescription>
            </div>
          </div>

          <div className="flex flex-col gap-3">
            <FilePickerCard
              label="Запись Twitch"
              icon={<FileVideo className="h-4 w-4" />}
              value={form.stream}
              onChange={(v) => set("stream", v)}
              kind="file"
              filters={[{ name: "Видео", extensions: VIDEO_EXT }]}
              title="Выберите запись стрима"
            />
            {/* Оригинал видео скрыт по просьбе: бэкенд подставляет оригинал сам.
                Расскомментировать, если снова понадобится ручной выбор. */}
            {/* <FilePickerCard
              label="Оригинал видео"
              icon={<FileVideo className="h-4 w-4" />}
              value={form.original}
              onChange={(v) => set("original", v)}
              kind="file"
              filters={[{ name: "Видео", extensions: VIDEO_EXT }]}
              title="Выберите видео-реакцию"
            /> */}
            <FilePickerCard
              label="Банворды (словарь мата)"
              icon={<FileText className="h-4 w-4" />}
              value={form.banwords}
              onChange={(v) => set("banwords", v)}
              kind="file"
              filters={[{ name: "Текст", extensions: ["txt", "json", "yaml", "yml"] }]}
              title="Выберите файл со словами"
            />
          </div>
        </Card>

        {/* --- Продвинутые --- */}
        <Card variant="surface" padding="sm">
          <Collapsible open={advOpen} onOpenChange={setAdvOpen}>
            <CollapsibleTrigger>
              <span className="flex items-center gap-2 text-fg">
                <Settings2 className="h-4 w-4 text-brand-from" />
                <span className="font-medium">Продвинутые параметры</span>
                <span className="text-xs text-subtle">
                  ({form.transcriber === "whisperx"
                    ? `WhisperX · ${form.model} · ${form.device}`
                    : "GigaAM v3"})
                </span>
              </span>
            </CollapsibleTrigger>
            <CollapsibleContent>
              <div className="grid grid-cols-1 gap-4 p-3 pt-4 md:grid-cols-2">
                <div className="md:col-span-2">
                  <FieldLabel help="Движок распознавания речи. GigaAM v3 — русскоязычная модель по умолчанию: быстрее и не требует установки CUDA/cuDNN. WhisperX — мультиязычный движок с тонкой настройкой модели, VAD и точности.">
                    Движок распознавания
                  </FieldLabel>
                  <Select
                    value={form.transcriber}
                    onChange={(e) => set("transcriber", e.target.value)}
                    options={TRANSCRIBER_OPTIONS}
                  />
                </div>

                {form.transcriber === "whisperx" && (
                  <>
                    <div>
                      <FieldLabel help="Размер модели WhisperX. Больше — точнее, но больше VRAM и медленнее. large-v3 — топ качества, tiny — для быстрых прогонов на CPU.">
                        Модель WhisperX
                      </FieldLabel>
                      <Select
                        value={form.model}
                        onChange={(e) => set("model", e.target.value)}
                        options={MODEL_OPTIONS}
                      />
                    </div>
                    <div>
                      <FieldLabel help="Язык стрима. auto — WhisperX сам определит по первым секундам. Явное указание точнее.">
                        Язык
                      </FieldLabel>
                      <Select
                        value={form.language}
                        onChange={(e) => set("language", e.target.value)}
                        options={LANGUAGE_OPTIONS}
                      />
                    </div>
                    <div>
                      <FieldLabel help="cuda — GPU NVIDIA (нужны CUDA-драйверы). cpu — работает везде, но в 10-30× медленнее.">
                        Device
                      </FieldLabel>
                      <Select
                        value={form.device}
                        onChange={(e) => set("device", e.target.value)}
                        options={DEVICE_OPTIONS}
                      />
                    </div>
                    <div>
                      <FieldLabel help="Точность вычислений. float16 — быстро, нужно 8+ GB VRAM. int8 — влезает в 4-6 GB. float32 — эталон, только для CPU или большого GPU.">
                        compute_type
                      </FieldLabel>
                      <Select
                        value={form.compute_type}
                        onChange={(e) => set("compute_type", e.target.value)}
                        options={COMPUTE_OPTIONS}
                      />
                    </div>
                    <div>
                      <FieldLabel help="Размер батча для WhisperX. Больше = быстрее, но растёт потребление VRAM. 16 — безопасный дефолт для 8 GB GPU.">
                        batch_size
                      </FieldLabel>
                      <Input
                        value={form.batch_size}
                        onChange={(e) => set("batch_size", e.target.value)}
                        inputMode="numeric"
                      />
                    </div>
                    <div>
                      <FieldLabel help="Voice Activity Detection — режет тишину до транскрипции. pyannote точнее (хуже с музыкой), silero быстрее и стабильнее.">
                        VAD method
                      </FieldLabel>
                      <Select
                        value={form.vad_method}
                        onChange={(e) => set("vad_method", e.target.value)}
                        options={VAD_OPTIONS}
                      />
                    </div>
                  </>
                )}

                <div>
                  <FieldLabel help="С какой секунды начинать обработку стрима. Формат HH:MM:SS.mmm — например «00:05:00» пропустит первые 5 минут.">
                    range_in
                  </FieldLabel>
                  <Input
                    value={form.range_in}
                    onChange={(e) => set("range_in", e.target.value)}
                    placeholder="00:00:00.000"
                    className="font-mono-tabular"
                  />
                </div>
                <div>
                  <FieldLabel help="До какой секунды обрабатывать. Оставь пустым — до конца файла. Формат тот же: HH:MM:SS.mmm.">
                    range_out
                  </FieldLabel>
                  <Input
                    value={form.range_out}
                    onChange={(e) => set("range_out", e.target.value)}
                    placeholder="01:20:00.000"
                    className="font-mono-tabular"
                  />
                </div>
                <div className="md:col-span-2">
                  <FieldLabel help="Дебаг: подсунуть готовый transcript.json вместо запуска распознавания. Полезно на машине без CUDA — тестируешь UI/pipeline без ждать ASR.">
                    mock_transcript
                  </FieldLabel>
                  <FileField
                    value={form.mock_transcript}
                    onChange={(v) => set("mock_transcript", v)}
                    kind="file"
                    placeholder="Опционально — для отладки без CUDA"
                    filters={[{ name: "JSON", extensions: ["json"] }]}
                  />
                </div>
                {form.transcriber === "whisperx" && (
                  <label className="flex items-center gap-2 text-sm text-muted md:col-span-2">
                    <input
                      type="checkbox"
                      checked={form.vad_filter}
                      onChange={(e) => set("vad_filter", e.target.checked)}
                      className="h-4 w-4 rounded border-white/10 bg-black/30 accent-brand-from"
                    />
                    Включить VAD-фильтр (WhisperX)
                    <HelpTip>
                      Дополнительная фильтрация тишины ВНУТРИ WhisperX (помимо
                      VAD method). Убирает ложные срабатывания на дыхание/шум.
                      Немного замедляет, но обычно улучшает качество.
                    </HelpTip>
                  </label>
                )}
              </div>
            </CollapsibleContent>
          </Collapsible>
        </Card>
      </div>

      {/* --- Live-панель --- */}
      {/* self-start + sticky: на большом экране панель «приклеивается» и едет
          вниз вместе со скроллом, а не остаётся в самом верху, когда форма
          длинная (продвинутые параметры раскрыты). */}
      <aside className="flex flex-col gap-4 self-start xl:sticky xl:top-8">
        <Card variant="glass">
          <div className="mb-3 flex items-center gap-2">
            <Sparkles className="h-4 w-4 text-brand-from" />
            <CardTitle>Что произойдёт</CardTitle>
          </div>
          <ol className="flex flex-col gap-2 text-sm text-muted">
            <PreviewLine
              enabled={!!form.stream}
              text={
                form.mock_transcript
                  ? "Пропуск ASR (mock_transcript задан)"
                  : "Извлечение аудио из записи"
              }
            />
            <PreviewLine
              enabled={!!form.stream}
              text={
                form.transcriber === "whisperx"
                  ? `Транскрипция WhisperX ${form.model} (${form.language})`
                  : "Транскрипция GigaAM v3 (русский)"
              }
            />
            <PreviewLine enabled={!!form.banwords} text="Поиск матов по банвордам" />
            <PreviewLine enabled={requiredFilled} text="Сборка разметки + скрипт для Vegas" />
          </ol>
          <div className="mt-4 grid grid-cols-2 gap-2 text-xs">
            <Chip
              icon={<Rocket className="h-3 w-3" />}
              label={form.transcriber === "whisperx" ? form.device : "gigaam"}
            />
            <Chip
              icon={<Volume2 className="h-3 w-3" />}
              label={form.transcriber === "whisperx" ? form.vad_method : "v3"}
            />
            {rangeMs != null && (
              <Chip
                icon={<Play className="h-3 w-3" />}
                label={fmtDurationMs(rangeMs)}
                className="col-span-2"
              />
            )}
          </div>
        </Card>

        {error && <ErrorCard error={error} />}

        <Button
          size="lg"
          onClick={onSubmit}
          loading={submitting}
          disabled={!requiredFilled}
          className="w-full"
        >
          Запустить <ChevronRight className="h-4 w-4" />
        </Button>
        {!requiredFilled && (
          <p className="text-center text-xs text-subtle">
            Выберите запись и банворды.
          </p>
        )}
      </aside>
    </div>
  );
}

// ============================================================================
// Общие мелкие компоненты
// ============================================================================

/**
 * FilePickerCard — крупная «плитка» выбора файла для расширенного режима:
 * закруглённый прямоугольник с иконкой, подписью-меткой и именем выбранного
 * файла (полный путь — мелким моноширинным под ним). Клик по всей плитке
 * открывает системный диалог; выбранный файл можно сбросить крестиком.
 */
function FilePickerCard({
  label,
  icon,
  value,
  onChange,
  kind,
  filters,
  title,
}: {
  label: string;
  icon: ReactNode;
  value: string;
  onChange: (v: string) => void;
  kind: "file";
  filters?: { name: string; extensions: string[] }[];
  title?: string;
}) {
  const [busy, setBusy] = useState(false);

  const pick = async () => {
    setBusy(true);
    try {
      const picked = await window.twitchCut.openFile({ filters, title });
      if (picked) onChange(picked);
    } finally {
      setBusy(false);
    }
  };
  // kind сейчас всегда "file" — оставлен в сигнатуре на случай будущих папок.
  void kind;

  const filled = !!value;
  return (
    <div
      role="button"
      tabIndex={0}
      onClick={pick}
      onKeyDown={(e) => (e.key === "Enter" || e.key === " ") && pick()}
      className={cn(
        "group flex items-center gap-3 rounded-xl border px-4 py-3 text-left transition-colors",
        "cursor-pointer focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-brand-from",
        filled
          ? "border-white/12 bg-surface/50 hover:border-white/25"
          : "border-dashed border-white/15 bg-black/20 hover:border-white/30 hover:bg-surface/40"
      )}
    >
      <div
        className={cn(
          "flex h-10 w-10 shrink-0 items-center justify-center rounded-lg transition-colors",
          filled ? "bg-brand/15 text-brand-from" : "bg-white/5 text-muted group-hover:text-brand-from"
        )}
      >
        {icon}
      </div>
      <div className="min-w-0 flex-1">
        <div className="text-xs font-medium uppercase tracking-wider text-subtle">
          {label}
        </div>
        {filled ? (
          <>
            <div className="truncate text-sm font-medium text-fg" title={value}>
              {baseName(value)}
            </div>
            <div className="truncate font-mono text-[10px] text-subtle" title={value}>
              {value}
            </div>
          </>
        ) : (
          <div className="text-sm text-muted">Нажмите, чтобы выбрать файл</div>
        )}
      </div>
      {filled ? (
        <button
          type="button"
          onClick={(e) => {
            e.stopPropagation();
            onChange("");
          }}
          className="shrink-0 rounded-md p-1.5 text-muted transition-colors hover:bg-white/5 hover:text-fg"
          title="Сбросить"
        >
          <X className="h-4 w-4" />
        </button>
      ) : (
        <span className="shrink-0 rounded-md px-2.5 py-1 text-xs text-muted transition-colors group-hover:bg-white/5 group-hover:text-fg">
          {busy ? "…" : "Выбрать…"}
        </span>
      )}
    </div>
  );
}

/**
 * ErrorCard — человекочитаемая ошибка запуска. Понятный заголовок сверху,
 * технические детали (сырой ответ backend'а) — под спойлером, чтобы не пугать
 * стектрейсом того, кто «не разбирается», но оставить их тому, кто полезет
 * разбираться.
 */
function ErrorCard({ error }: { error: string }) {
  const [open, setOpen] = useState(false);
  return (
    <Card variant="surface" className="border-err/30 bg-err/5 text-sm">
      <div className="font-medium text-err">Не удалось запустить обработку</div>
      <p className="mt-1 text-xs text-muted">
        Проверьте, что файл существует и открыт не в другой программе, и
        попробуйте ещё раз.
      </p>
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        className="mt-2 text-[11px] text-brand-from hover:underline"
      >
        {open ? "Скрыть детали" : "Показать детали"}
      </button>
      {open && (
        <div className="mt-2 max-h-40 overflow-auto break-all rounded bg-black/30 p-2 font-mono-tabular text-[11px] text-err/90">
          {error}
        </div>
      )}
    </Card>
  );
}

function PreviewLine({ enabled, text }: { enabled: boolean; text: string }) {
  return (
    <li className="flex items-start gap-2">
      <span
        className={
          enabled
            ? "mt-1.5 h-1.5 w-1.5 shrink-0 rounded-full bg-brand-from"
            : "mt-1.5 h-1.5 w-1.5 shrink-0 rounded-full bg-white/10"
        }
      />
      <span className={enabled ? "text-fg" : "text-subtle"}>{text}</span>
    </li>
  );
}

function Chip({
  icon,
  label,
  className,
}: {
  icon: React.ReactNode;
  label: string;
  className?: string;
}) {
  return (
    <span
      className={
        "inline-flex items-center gap-1.5 rounded-md border border-white/10 bg-black/30 px-2 py-1 text-muted " +
        (className ?? "")
      }
    >
      <span className="text-brand-from">{icon}</span>
      <span className="truncate">{label}</span>
    </span>
  );
}
