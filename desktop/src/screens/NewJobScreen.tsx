import { useCallback, useEffect, useMemo, useState } from "react";
import {
  ChevronRight,
  FileVideo,
  Folder,
  FolderOpen,
  Play,
  Rocket,
  Settings2,
  Sparkles,
  Volume2,
  Wand2,
} from "lucide-react";
import { createProcessJob, suggestWorkdir, type JobState } from "../api";
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
 * NewJobScreen — форма запуска process-job'а. Наследник ImportScreen'а из
 * прошлой версии, полностью переделан:
 *
 * - 6 первичных путей → FileField-ы, сгруппированы в 2 карточки (Входы / Выходы).
 * - Продвинутые опции (модель, VAD, mock_transcript, range) — в Collapsible.
 * - Правая колонка — live-панель «Что произойдёт» с расчётом длительности,
 *   бейджами device/model/язык.
 * - Значения формы кэшируются в localStorage, чтобы не набирать 6 путей
 *   каждый раз при разработке; отдельно для «продвинутых» — тоже сохраняем.
 * - Валидация на клиенте: 4 обязательных пути должны быть заполнены. Кнопка
 *   Submit disabled иначе.
 */

const LS_KEY = "twitchCut.newJob.v1";

interface FormState {
  stream: string;
  original: string;
  banwords: string;
  workdir: string;
  decisions: string;
  vegas: string;
  // advanced
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

export function NewJobScreen({
  onJobStarted,
}: {
  onJobStarted: (jobId: string, decisionsPath?: string) => void;
}) {
  const [form, setForm] = useState<FormState>(() => loadForm());
  const [advOpen, setAdvOpen] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const set = useCallback(
    <K extends keyof FormState>(k: K, v: FormState[K]) =>
      setForm((f) => ({ ...f, [k]: v })),
    []
  );

  // Debounced-persist в localStorage.
  useEffect(() => {
    const t = window.setTimeout(() => saveForm(form), 400);
    return () => window.clearTimeout(t);
  }, [form]);

  const requiredFilled =
    !!form.stream && !!form.original && !!form.banwords && !!form.workdir;

  const rangeMs = useMemo(() => {
    const a = parseTimecodeMs(form.range_in);
    const b = parseTimecodeMs(form.range_out);
    if (a == null || b == null || b <= a) return null;
    return b - a;
  }, [form.range_in, form.range_out]);

  // Автоподстановка decisions/vegas когда указан workdir и они пусты.
  useEffect(() => {
    if (form.workdir && !form.decisions) {
      const sep = form.workdir.includes("\\") ? "\\" : "/";
      set("decisions", `${form.workdir}${sep}decisions.json`);
    }
    if (form.workdir && !form.vegas) {
      const sep = form.workdir.includes("\\") ? "\\" : "/";
      set("vegas", `${form.workdir}${sep}mutes.cs`);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [form.workdir]);

  const submit = async () => {
    setError(null);
    setSubmitting(true);
    try {
      const req = {
        stream: form.stream,
        original: form.original,
        banwords: form.banwords,
        workdir: form.workdir,
        decisions: form.decisions,
        vegas: form.vegas,
        range_in: form.range_in || null,
        range_out: form.range_out || null,
        model: form.model,
        language: form.language,
        device: form.device,
        compute_type: form.compute_type,
        batch_size: Number(form.batch_size) || 16,
        vad_method: form.vad_method,
        vad_filter: form.vad_filter,
        mock_transcript: form.mock_transcript || null,
      };
      const state: JobState = await createProcessJob(req);
      onJobStarted(state.id, form.decisions);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className="mx-auto flex max-w-7xl flex-col gap-6 p-8">
      <header>
        <h1 className="text-2xl font-semibold tracking-tight text-fg">Новый job</h1>
        <p className="mt-1 text-sm text-muted">
          Стрим, оригинал, банворды → <span className="font-mono-tabular text-fg">decisions.json</span> + Vegas
          .cs. Продвинутые параметры — под спойлером.
        </p>
      </header>

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
                <CardTitle>Входы</CardTitle>
                <CardDescription>Три файла, всё как в CLI-пайплайне.</CardDescription>
              </div>
            </div>

            <div className="flex flex-col gap-4">
              <div>
                <Label>Stream (запись Twitch)</Label>
                <FileField
                  value={form.stream}
                  onChange={(v) => set("stream", v)}
                  kind="file"
                  leftIcon={<FileVideo className="h-4 w-4" />}
                  placeholder="…/stream.mp4"
                  filters={[
                    { name: "Video", extensions: ["mp4", "mkv", "mov", "ts"] },
                  ]}
                />
              </div>
              <div>
                <Label>Оригинал (реакция)</Label>
                <FileField
                  value={form.original}
                  onChange={(v) => set("original", v)}
                  kind="file"
                  leftIcon={<FileVideo className="h-4 w-4" />}
                  placeholder="…/reaction.mp4"
                  filters={[
                    { name: "Video", extensions: ["mp4", "mkv", "mov"] },
                  ]}
                />
              </div>
              <div>
                <Label>Банворды</Label>
                <FileField
                  value={form.banwords}
                  onChange={(v) => set("banwords", v)}
                  kind="file"
                  placeholder="…/banwords.txt"
                  filters={[
                    { name: "Text", extensions: ["txt", "json", "yaml", "yml"] },
                  ]}
                />
              </div>
            </div>
          </Card>

          {/* --- Выходы --- */}
          <Card variant="elevated">
            <div className="mb-4 flex items-start gap-3">
              <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-brand/10 text-brand-from">
                <Folder className="h-4 w-4" />
              </div>
              <div>
                <CardTitle>Выходы</CardTitle>
                <CardDescription>
                  Workdir — кэш и чекпоинты. Decisions/Vegas подставятся автоматически.
                </CardDescription>
              </div>
            </div>

            <div className="flex flex-col gap-4">
              <div>
                <div className="mb-1.5 flex items-center justify-between gap-2">
                  <Label className="mb-0">Workdir</Label>
                  <button
                    type="button"
                    onClick={async () => {
                      // Auto-workdir: спрашиваем backend "какой путь предложить".
                      // Передаём stream — если он уже выбран, basename попадёт
                      // в имя папки для читаемости. Если нет — backend вернёт
                      // 'job_yyyymmdd_hhmm'.
                      try {
                        const { path } = await suggestWorkdir(form.stream || undefined);
                        set("workdir", path);
                      } catch {
                        /* тихо: backend может ещё не подняться */
                      }
                    }}
                    className="inline-flex items-center gap-1 rounded px-2 py-0.5 text-[11px] text-brand-from transition-colors hover:bg-brand-from/10"
                    title="Сгенерировать уникальный путь ~/twitch_cut/projects/{имя}_{дата}"
                  >
                    <Wand2 className="h-3 w-3" />
                    Auto
                  </button>
                </div>
                <FileField
                  value={form.workdir}
                  onChange={(v) => set("workdir", v)}
                  kind="directory"
                  placeholder="…/twitch_cache/stream-01"
                />
              </div>
              <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
                <div>
                  <Label>decisions.json</Label>
                  <FileField
                    value={form.decisions}
                    onChange={(v) => set("decisions", v)}
                    kind="save"
                    placeholder="…/decisions.json"
                    filters={[{ name: "JSON", extensions: ["json"] }]}
                  />
                </div>
                <div>
                  <Label>Vegas .cs</Label>
                  <FileField
                    value={form.vegas}
                    onChange={(v) => set("vegas", v)}
                    kind="save"
                    placeholder="…/mutes.cs"
                    filters={[{ name: "C# script", extensions: ["cs"] }]}
                  />
                </div>
              </div>
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
                    ({form.model} · {form.device} · {form.vad_method})
                  </span>
                </span>
              </CollapsibleTrigger>
              <CollapsibleContent>
                <div className="grid grid-cols-1 gap-4 p-3 pt-4 md:grid-cols-2">
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
                    <FieldLabel help="Дебаг: подсунуть готовый transcript.json вместо запуска WhisperX. Полезно на машине без CUDA — тестируешь UI/pipeline без ждать ASR.">
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
                </div>
              </CollapsibleContent>
            </Collapsible>
          </Card>
        </div>

        {/* --- Live-панель --- */}
        <aside className="flex flex-col gap-4">
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
                    : "Извлечение аудио из stream"
                }
              />
              <PreviewLine
                enabled={!!form.stream}
                text={`Транскрипция WhisperX ${form.model} (${form.language})`}
              />
              <PreviewLine
                enabled={!!form.banwords}
                text="Поиск матов по банвордам"
              />
              <PreviewLine
                enabled={requiredFilled}
                text="Сборка decisions.json + Vegas .cs"
              />
            </ol>
            <div className="mt-4 grid grid-cols-2 gap-2 text-xs">
              <Chip icon={<Rocket className="h-3 w-3" />} label={form.device} />
              <Chip icon={<Volume2 className="h-3 w-3" />} label={form.vad_method} />
              {rangeMs != null && (
                <Chip
                  icon={<Play className="h-3 w-3" />}
                  label={fmtDurationMs(rangeMs)}
                  className="col-span-2"
                />
              )}
            </div>
          </Card>

          {error && (
            <Card
              variant="surface"
              className="border-err/30 bg-err/5 text-sm text-err"
            >
              <div className="font-medium">Backend вернул ошибку</div>
              <div className="mt-1 break-all font-mono-tabular text-[11px] text-err/90">
                {error}
              </div>
            </Card>
          )}

          <Button
            size="lg"
            onClick={submit}
            loading={submitting}
            disabled={!requiredFilled}
            className="w-full"
          >
            Запустить <ChevronRight className="h-4 w-4" />
          </Button>
          {!requiredFilled && (
            <p className="text-center text-xs text-subtle">
              Заполните stream, оригинал, банворды и workdir.
            </p>
          )}
        </aside>
      </div>
    </div>
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
