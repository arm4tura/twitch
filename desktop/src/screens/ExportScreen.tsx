import { useEffect, useMemo, useState } from "react";
import {
  Download,
  FileCode2,
  FolderOpen,
  Package,
  Send,
  Sparkles,
  Upload,
} from "lucide-react";
import {
  createExportVegasJob,
  createHighlightsExportJob,
  createHighlightsImportJob,
  readDecisions,
} from "../api";
import { Button } from "../components/ui/Button";
import { Card, CardDescription, CardTitle } from "../components/ui/Card";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "../components/ui/Collapsible";
import { FileField } from "../components/ui/FileField";
import { Label } from "../components/ui/Input";
import { toast } from "../components/ui/Toast";

/**
 * ExportScreen — выгрузка готового проекта.
 *
 * Главное действие — сборка скрипта для Sony Vegas Pro (мут-регионы как takes).
 * Это то, ради чего пользователь сюда пришёл, поэтому карточка Vegas стоит
 * первой и на всю ширину.
 *
 * Остальное спрятано под «Дополнительно» (свёрнуто по умолчанию), т.к. нужно
 * лишь продвинутым пользователям:
 *  - NotebookLM package — zip с транскриптом для Google NotebookLM;
 *  - Import NotebookLM response — забрать ответ ИИ и обогатить проект тегами.
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
  workdir: string | null;
  stream: string | null;
}

const LS_KEY = "twitchCut.export.v1";

interface FormState {
  vegasPath: string;
  packageDir: string;
  responsePath: string;
}

const DEFAULT_FORM: FormState = { vegasPath: "", packageDir: "", responsePath: "" };

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
  return {
    mutes: Array.isArray(doc?.mutes) ? doc.mutes.length : 0,
    cuts: Array.isArray(doc?.cuts) ? doc.cuts.length : 0,
    highlights: Array.isArray(doc?.highlights) ? doc.highlights.length : 0,
    workdir: (meta.workdir as string | undefined) ?? null,
    stream: (meta.stream_path as string | undefined) ?? null,
  };
}

export function ExportScreen({ decisionsPath, onJobStarted }: ExportScreenProps) {
  const [form, setForm] = useState<FormState>(loadForm);
  const [summary, setSummary] = useState<Summary | null>(null);
  const [loadingSummary, setLoadingSummary] = useState(false);
  const [busy, setBusy] = useState<null | "vegas" | "package" | "import">(null);
  const [moreOpen, setMoreOpen] = useState(false);

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

  // Дефолтные пути в save-dialog'ах — рядом с decisions.json. Разделитель
  // угадываем по самому пути, чтобы работало и на Windows, и на POSIX.
  const { sep, dirOf } = useMemo(() => {
    const s = decisionsPath.includes("\\") ? "\\" : "/";
    const idx = decisionsPath.lastIndexOf(s);
    return { sep: s, dirOf: idx > 0 ? decisionsPath.slice(0, idx) : "" };
  }, [decisionsPath]);

  const vegasDefault = dirOf ? `${dirOf}${sep}vegas.cs` : undefined;
  const packageDefault = dirOf || undefined;

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

  const runPackage = async () => {
    if (!form.packageDir) {
      toast.error("Укажи каталог для NotebookLM-пакета");
      return;
    }
    setBusy("package");
    try {
      const job = await createHighlightsExportJob({
        decisions: decisionsPath,
        out_dir: form.packageDir,
      });
      toastReveal(form.packageDir, "NotebookLM-пакет собирается");
      onJobStarted(job.id);
    } catch (e) {
      toast.error("Ошибка запуска", { description: String(e) });
    } finally {
      setBusy(null);
    }
  };

  const runImport = async () => {
    if (!form.responsePath) {
      toast.error("Укажи файл ответа NotebookLM");
      return;
    }
    setBusy("import");
    try {
      const job = await createHighlightsImportJob({
        decisions: decisionsPath,
        response: form.responsePath,
        // По умолчанию перезаписываем исходный decisions.json — пользователь
        // ждёт «обогащённый» тем же путём. Отдельное поле для output пока
        // не нужно; если добавим — сделаем «Save As».
        output: decisionsPath,
      });
      toastReveal(decisionsPath, "Импорт хайлайтов запущен");
      onJobStarted(job.id);
    } catch (e) {
      toast.error("Ошибка запуска", { description: String(e) });
    } finally {
      setBusy(null);
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
          места прямо на таймлайн. Остальные способы выгрузки — под «Дополнительно».
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

      {/* Всё остальное — только продвинутым, свёрнуто по умолчанию. */}
      <Collapsible open={moreOpen} onOpenChange={setMoreOpen}>
        <CollapsibleTrigger>
          <span className="flex items-center gap-2">
            <Sparkles className="h-4 w-4" /> Дополнительно — NotebookLM
          </span>
        </CollapsibleTrigger>
        <CollapsibleContent>
          <div className="grid gap-5 pt-4 md:grid-cols-2">
            <Card variant="elevated" padding="md" className="flex flex-col">
          <CardHead
            icon={<Package className="h-4 w-4" />}
            gradient="from-violet-500/25 to-violet-500/0"
            iconColor="text-brand-from"
            title="NotebookLM package"
            desc="ZIP с транскриптом и метаданными для загрузки в Google NotebookLM. ИИ вернёт список моментов-хайлайтов."
          />
          <div className="mt-4 flex-1 space-y-3">
            <Label eyebrow>Каталог для пакета</Label>
            <FileField
              kind="directory"
              value={form.packageDir}
              onChange={(v) => setF("packageDir", v)}
              defaultPath={packageDefault}
              placeholder="…/notebooklm/"
            />
            <button
              type="button"
              onClick={async () => {
                const ok = await window.twitchCut.openExternal(
                  "https://notebooklm.google.com/"
                );
                if (!ok) toast.error("Не удалось открыть браузер");
              }}
              className="text-[11px] text-brand-from hover:underline"
            >
              → Открыть NotebookLM
            </button>
          </div>
          <Button
            className="mt-5"
            onClick={runPackage}
            disabled={!form.packageDir}
            loading={busy === "package"}
          >
            <Sparkles className="h-4 w-4" /> Собрать пакет
          </Button>
        </Card>

        <Card variant="elevated" padding="md" className="flex flex-col">
          <CardHead
            icon={<Upload className="h-4 w-4" />}
            gradient="from-emerald-500/25 to-emerald-500/0"
            iconColor="text-ok"
            title="Import NotebookLM response"
            desc="Забрать ответ NotebookLM (txt/json/md) и обогатить decisions.json тегами highlight'ов с оценками."
          />
          <div className="mt-4 flex-1 space-y-3">
            <Label eyebrow>Файл ответа</Label>
            <FileField
              kind="file"
              value={form.responsePath}
              onChange={(v) => setF("responsePath", v)}
              filters={[
                { name: "NotebookLM response", extensions: ["txt", "json", "md"] },
                { name: "Все", extensions: ["*"] },
              ]}
              placeholder="…/notebooklm-response.txt"
            />
          </div>
          <Button
            className="mt-5"
            onClick={runImport}
            disabled={!form.responsePath}
            loading={busy === "import"}
          >
            <Upload className="h-4 w-4" /> Импортировать
          </Button>
        </Card>
          </div>
        </CollapsibleContent>
      </Collapsible>
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
