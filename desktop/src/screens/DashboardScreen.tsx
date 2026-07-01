import { useCallback, useEffect, useMemo, useState } from "react";
import {
  LayoutDashboard,
  Plus,
  RefreshCw,
  FolderOpen,
  Volume2,
  Scissors,
  Sparkles,
  Film,
  Clock,
  FileText,
  AlertCircle,
} from "lucide-react";
import { EmptyState } from "../components/ui/EmptyState";
import { Button } from "../components/ui/Button";
import { Card } from "../components/ui/Card";
import { StatCard } from "../components/ui/StatCard";
import { fmtDurationHuman, fmtRelativeMs } from "../lib/format";
import { listProjects, listJobs, type ProjectMeta, type JobState } from "../api";

/**
 * DashboardScreen — реализация коммита 2.
 *
 * Верхний ряд: три StatCard'а (проекты, активные джобы, суммарные правки).
 * Ниже — responsive grid карточек проектов из GET /projects. Пустой список →
 * центральный EmptyState с CTA на /new.
 *
 * Опрос: раз при монтировании + poll running-jobs каждые 3 с (лёгкий пинг,
 * не для прогресс-баров — те живут через WS в JobScreen). Кнопка «Обновить»
 * дёргает оба запроса вручную.
 */
export function DashboardScreen({
  onNew,
  onOpen,
}: {
  onNew: () => void;
  onOpen: (decisionsPath: string) => void;
}) {
  const [projects, setProjects] = useState<ProjectMeta[] | null>(null);
  const [jobs, setJobs] = useState<JobState[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [reloading, setReloading] = useState(false);

  const reload = useCallback(async () => {
    setReloading(true);
    setError(null);
    try {
      const [p, j] = await Promise.all([listProjects(), listJobs()]);
      setProjects(p);
      setJobs(j);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setReloading(false);
    }
  }, []);

  useEffect(() => {
    reload();
    // Лёгкий poll только для jobs-счётчика — не для проектов (те меняются
    // редко и через явные действия, которые могут дёрнуть reload напрямую).
    const timer = window.setInterval(async () => {
      try {
        const j = await listJobs();
        setJobs(j);
      } catch {
        /* игнор — покажем при явном обновлении */
      }
    }, 3000);
    return () => window.clearInterval(timer);
  }, [reload]);

  const runningCount = useMemo(
    () => jobs.filter((j) => j.status === "running" || j.status === "pending").length,
    [jobs]
  );

  const totalEdits = useMemo(
    () =>
      (projects ?? []).reduce(
        (acc, p) => acc + p.mutes_count + p.cuts_count + p.highlights_count,
        0
      ),
    [projects]
  );

  return (
    <div className="mx-auto max-w-6xl p-8">
      <header className="mb-8 flex items-end justify-between gap-6">
        <div>
          <h1 className="text-2xl font-semibold tracking-tight text-fg">Проекты</h1>
          <p className="mt-1 text-sm text-muted">
            Последние обработанные стримы и их decisions.json.
          </p>
        </div>
        <div className="flex items-center gap-2">
          <Button
            variant="secondary"
            size="md"
            onClick={reload}
            loading={reloading}
            aria-label="Обновить"
          >
            <RefreshCw className="h-4 w-4" /> Обновить
          </Button>
          <Button onClick={onNew}>
            <Plus className="h-4 w-4" /> Новый проект
          </Button>
        </div>
      </header>

      {/* Stat row */}
      <section className="mb-8 grid grid-cols-1 gap-3 sm:grid-cols-3">
        <StatCard
          icon={<FolderOpen className="h-3.5 w-3.5" />}
          label="Проектов"
          value={projects === null ? "—" : projects.length}
          hint={projects && projects.length > 0 ? "в реестре недавних" : "начните с нового"}
        />
        <StatCard
          icon={<Film className="h-3.5 w-3.5" />}
          label="Активных задач"
          value={runningCount}
          hint={
            runningCount > 0
              ? "в очереди или выполняются"
              : jobs.length > 0
                ? `${jobs.length} завершено`
                : "backend простаивает"
          }
        />
        <StatCard
          icon={<Sparkles className="h-3.5 w-3.5" />}
          label="Правок всего"
          value={totalEdits}
          hint="mutes + cuts + highlights"
        />
      </section>

      {/* Error banner */}
      {error && (
        <Card
          variant="surface"
          padding="sm"
          className="mb-4 flex items-center gap-3 border-err/40 bg-err/10 text-sm text-fg"
        >
          <AlertCircle className="h-4 w-4 shrink-0 text-err" />
          <span className="flex-1">Backend не отвечает: {error}</span>
          <Button variant="ghost" size="sm" onClick={reload}>
            Повторить
          </Button>
        </Card>
      )}

      {/* Projects grid */}
      {projects === null ? (
        <ProjectsSkeleton />
      ) : projects.length === 0 ? (
        <EmptyState
          icon={<LayoutDashboard className="h-6 w-6" />}
          title="Пока пусто"
          description="Первый проект появится здесь после запуска обработки. Реестр обновится автоматически, как только job завершится."
          action={
            <Button onClick={onNew}>
              <Plus className="h-4 w-4" /> Запустить первый job
            </Button>
          }
        />
      ) : (
        <div className="grid grid-cols-1 gap-4 md:grid-cols-2 xl:grid-cols-3">
          {projects.map((p) => (
            <ProjectCard key={p.decisions_path} project={p} onOpen={onOpen} />
          ))}
        </div>
      )}
    </div>
  );
}

// -------- ProjectCard --------------------------------------------------------

function ProjectCard({
  project,
  onOpen,
}: {
  project: ProjectMeta;
  onOpen: (path: string) => void;
}) {
  const title = deriveTitle(project);
  const durationLabel =
    project.duration_ms != null ? fmtDurationHuman(project.duration_ms / 1000) : null;

  return (
    <Card
      variant="elevated"
      padding="none"
      className="group flex flex-col overflow-hidden transition-transform hover:-translate-y-0.5 hover:border-white/10"
    >
      {/* Cover strip — плейсхолдер под будущий preview кадра (Vision-фаза). */}
      <div className="relative h-24 overflow-hidden bg-gradient-to-br from-brand-from/20 via-brand-to/10 to-transparent">
        <div
          className="absolute inset-0 opacity-40"
          style={{
            backgroundImage:
              "radial-gradient(600px circle at 30% 20%, rgb(139 92 246 / 0.35), transparent 50%)",
          }}
        />
        <div className="absolute inset-0 flex items-center justify-center">
          <Film className="h-8 w-8 text-white/40" />
        </div>
        {durationLabel && (
          <div className="absolute right-2 top-2 rounded-md bg-black/50 px-1.5 py-0.5 font-mono-tabular text-[11px] text-white backdrop-blur">
            {durationLabel}
          </div>
        )}
      </div>

      {/* Body */}
      <div className="flex flex-1 flex-col gap-3 p-4">
        <div className="min-w-0">
          <h3 className="truncate text-sm font-semibold text-fg" title={title}>
            {title}
          </h3>
          <div className="mt-0.5 flex items-center gap-1.5 text-[11px] text-subtle">
            <Clock className="h-3 w-3" />
            <span>{fmtRelativeMs(project.updated_at_ms)}</span>
          </div>
        </div>

        {/* Meta rows */}
        <div className="flex items-center gap-3 text-[11px] text-muted">
          <span className="flex items-center gap-1" title="Mutes">
            <Volume2 className="h-3 w-3 text-brand-from" />
            <span className="font-mono-tabular">{project.mutes_count}</span>
          </span>
          <span className="flex items-center gap-1" title="Cuts">
            <Scissors className="h-3 w-3 text-warn" />
            <span className="font-mono-tabular">{project.cuts_count}</span>
          </span>
          <span className="flex items-center gap-1" title="Highlights">
            <Sparkles className="h-3 w-3 text-ok" />
            <span className="font-mono-tabular">{project.highlights_count}</span>
          </span>
        </div>

        <div
          className="min-w-0 flex items-center gap-1.5 text-[11px] text-subtle"
          title={project.decisions_path}
        >
          <FileText className="h-3 w-3 shrink-0" />
          <span className="truncate font-mono text-[10px]">
            {shortenPath(project.decisions_path)}
          </span>
        </div>

        <div className="mt-auto pt-1">
          <Button
            variant="secondary"
            size="sm"
            className="w-full"
            onClick={() => onOpen(project.decisions_path)}
          >
            Открыть таймлайн
          </Button>
        </div>
      </div>
    </Card>
  );
}

// -------- Skeleton ----------------------------------------------------------

function ProjectsSkeleton() {
  return (
    <div className="grid grid-cols-1 gap-4 md:grid-cols-2 xl:grid-cols-3">
      {Array.from({ length: 3 }).map((_, i) => (
        <Card
          key={i}
          variant="surface"
          padding="none"
          className="animate-pulse overflow-hidden"
        >
          <div className="h-24 bg-elevated/40" />
          <div className="flex flex-col gap-2.5 p-4">
            <div className="h-3 w-2/3 rounded bg-elevated/60" />
            <div className="h-2 w-1/3 rounded bg-elevated/40" />
            <div className="mt-2 h-2 w-1/2 rounded bg-elevated/40" />
            <div className="mt-2 h-8 w-full rounded bg-elevated/40" />
          </div>
        </Card>
      ))}
    </div>
  );
}

// -------- helpers -----------------------------------------------------------

/** Название проекта: имя папки над decisions.json, если оно осмысленное; иначе stem файла. */
function deriveTitle(p: ProjectMeta): string {
  // Берём имя родительской директории — обычно это имя стрима/эпизода.
  const parts = p.decisions_path.replace(/\\/g, "/").split("/");
  if (parts.length >= 2) {
    const parent = parts[parts.length - 2];
    if (parent && parent !== "." && !/^(cache|workdir|tmp)$/i.test(parent)) {
      return parent;
    }
  }
  return p.name;
}

/** `.../deep/nested/path/decisions.json` → `…/nested/path/decisions.json` (2 сегмента). */
function shortenPath(path: string): string {
  const parts = path.replace(/\\/g, "/").split("/");
  if (parts.length <= 3) return path;
  return "…/" + parts.slice(-3).join("/");
}
