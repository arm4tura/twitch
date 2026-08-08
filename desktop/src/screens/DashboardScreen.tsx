import { useCallback, useEffect, useMemo, useState } from "react";
import {
  LayoutDashboard,
  Plus,
  RefreshCw,
  FolderOpen,
  Sparkles,
  Film,
  Clock,
  FileText,
  AlertCircle,
  MoreVertical,
  Trash2,
} from "lucide-react";
import { EmptyState } from "../components/ui/EmptyState";
import { Button } from "../components/ui/Button";
import { Card } from "../components/ui/Card";
import { StatCard } from "../components/ui/StatCard";
import { Checkbox } from "../components/ui/Checkbox";
import {
  DropdownMenu,
  DropdownMenuTrigger,
  DropdownMenuContent,
  DropdownMenuItem,
} from "../components/ui/DropdownMenu";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
  DialogFooter,
} from "../components/ui/Dialog";
import { toast } from "../components/ui/Toast";
import { fmtDurationHuman, fmtRelativeMs } from "../lib/format";
import {
  listProjects,
  listJobs,
  deleteProject,
  thumbnailUrl,
  type ProjectMeta,
  type JobState,
} from "../api";

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

  // Удаление проекта: карточка, для которой открыт диалог подтверждения
  // (null = диалог закрыт), флаг «удалить файлы с диска» и in-flight-стейт.
  const [deleting, setDeleting] = useState<ProjectMeta | null>(null);
  const [deleteFiles, setDeleteFiles] = useState(true);
  const [deleteBusy, setDeleteBusy] = useState(false);

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

  // Открыть диалог удаления. Чекбокс «удалить файлы» каждый раз сбрасываем в
  // true — деструктивный дефолт по требованию, но пользователь видит его явно
  // и может снять перед подтверждением.
  const openDelete = useCallback((project: ProjectMeta) => {
    setDeleting(project);
    setDeleteFiles(true);
  }, []);

  const confirmDelete = useCallback(async () => {
    if (!deleting) return;
    setDeleteBusy(true);
    try {
      const res = await deleteProject(deleting.decisions_path, deleteFiles);
      // Реестр чистится всегда; удаление файлов — best-effort. Если бэкенд
      // вернул error (например, папка залочена на Windows или «не похожа на
      // проект»), проект из списка всё равно ушёл — показываем как warning.
      if (res.error) {
        toast.warning("Проект убран из списка, но файлы не удалены", {
          description: res.error,
        });
      } else if (res.deleted_dir) {
        toast.success("Проект и его файлы удалены");
      } else {
        toast.success("Проект убран из списка");
      }
      setDeleting(null);
      await reload();
    } catch (err) {
      toast.error("Не удалось удалить проект", {
        description: err instanceof Error ? err.message : String(err),
      });
    } finally {
      setDeleteBusy(false);
    }
  }, [deleting, deleteFiles, reload]);

  const runningCount = useMemo(
    () => jobs.filter((j) => j.status === "running" || j.status === "pending").length,
    [jobs]
  );
  const hasActiveJobs = runningCount > 0;

  // Первичная загрузка — один раз при монтировании.
  useEffect(() => {
    reload();
  }, [reload]);

  // Poll jobs-счётчика. Пока есть активные задачи — часто (3с), в простое —
  // редкий heartbeat (15с), чтобы не дёргать backend каждые 3 секунды вечно,
  // когда ничего не происходит. Эффект пересоздаётся только при смене
  // hasActiveJobs (а не на каждый tick), поэтому таймер не сбрасывается зря.
  useEffect(() => {
    const intervalMs = hasActiveJobs ? 3000 : 15000;
    const timer = window.setInterval(async () => {
      try {
        const j = await listJobs();
        setJobs(j);
      } catch {
        /* игнор — покажем при явном обновлении */
      }
    }, intervalMs);
    return () => window.clearInterval(timer);
  }, [hasActiveJobs]);

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
            Обработанные записи — открой любую, чтобы поправить и выгрузить.
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
          {/* «Новый проект» показываем только когда список непустой — при пустом
              реестре главный CTA живёт в EmptyState по центру, дублировать не нужно. */}
          {projects && projects.length > 0 && (
            <Button onClick={onNew}>
              <Plus className="h-4 w-4" /> Новый проект
            </Button>
          )}
        </div>
      </header>

      {/* Stat row */}
      <section className="mb-8 grid grid-cols-1 gap-3 sm:grid-cols-3">
        <StatCard
          icon={<FolderOpen className="h-3.5 w-3.5" />}
          label="Проектов"
          value={projects === null ? "—" : projects.length}
          hint={projects && projects.length > 0 ? "завершено" : "начните с нового"}
        />
        <StatCard
          icon={<Film className="h-3.5 w-3.5" />}
          label="Активных задач"
          value={runningCount}
          hint={
            runningCount > 0
              ? "в очереди или выполняются"
              : "нет активных обработок"
          }
        />
        <StatCard
          icon={<Sparkles className="h-3.5 w-3.5" />}
          label="Всего правок"
          value={totalEdits}
          hint="заглушки, вырезы и хайлайты"
        />
      </section>

      {/* Error banner */}
      {error && <DashboardError error={error} onRetry={reload} />}

      {/* Projects grid */}
      {projects === null ? (
        <ProjectsSkeleton />
      ) : projects.length === 0 ? (
        <EmptyState
          icon={<LayoutDashboard className="h-6 w-6" />}
          title="Пока пусто"
          description="Здесь появится первая обработанная запись. Список обновится сам, как только обработка закончится."
          action={
            <Button onClick={onNew}>
              <Plus className="h-4 w-4" /> Обработать первую запись
            </Button>
          }
        />
      ) : (
        <div className="grid grid-cols-1 gap-4 md:grid-cols-2 xl:grid-cols-3">
          {projects.map((p) => (
            <ProjectCard
              key={p.decisions_path}
              project={p}
              onOpen={onOpen}
              onDelete={openDelete}
            />
          ))}
        </div>
      )}

      {/* Диалог подтверждения удаления. Открыт, пока deleting !== null.
          onOpenChange закрывает его только когда не идёт удаление — чтобы
          случайный клик-вне не оборвал запрос на полпути. */}
      <Dialog
        open={deleting !== null}
        onOpenChange={(open) => {
          if (!open && !deleteBusy) setDeleting(null);
        }}
      >
        <DialogContent className="max-w-md">
          <DialogHeader>
            <DialogTitle>
              Удалить проект «{deleting ? deriveTitle(deleting) : ""}»?
            </DialogTitle>
            <DialogDescription>
              Проект исчезнет из списка недавних. Это действие нельзя отменить.
            </DialogDescription>
          </DialogHeader>

          <div className="rounded-lg border border-white/10 bg-black/20 p-3">
            <Checkbox
              checked={deleteFiles}
              onChange={(e) => setDeleteFiles(e.target.checked)}
              label="Также удалить файлы с диска"
              hint={deleting ? shortenPath(projectDir(deleting.decisions_path)) : undefined}
            />
          </div>

          <DialogFooter>
            <Button
              variant="ghost"
              onClick={() => setDeleting(null)}
              disabled={deleteBusy}
            >
              Отмена
            </Button>
            <Button
              variant="destructive"
              onClick={confirmDelete}
              loading={deleteBusy}
            >
              <Trash2 className="h-4 w-4" /> Удалить
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}

// -------- ProjectCard --------------------------------------------------------

function ProjectCard({
  project,
  onOpen,
  onDelete,
}: {
  project: ProjectMeta;
  onOpen: (path: string) => void;
  onDelete: (project: ProjectMeta) => void;
}) {
  const { title, timestamp } = deriveMeta(project);
  const durationLabel =
    project.duration_ms != null ? fmtDurationHuman(project.duration_ms / 1000) : null;

  // Превью-кадр из исходной записи. Пробуем загрузить только если знаем путь и
  // base URL уже резолвился; при 404 (исходник удалён/перемещён — типично для
  // старых проектов) или битом кадре откатываемся на Film-плейсхолдер.
  const thumbSrc = project.stream_path ? thumbnailUrl(project.stream_path) : null;
  const [thumbFailed, setThumbFailed] = useState(false);
  const showThumb = thumbSrc != null && !thumbFailed;

  return (
    <Card
      variant="elevated"
      padding="none"
      className="group flex flex-col overflow-hidden transition-transform hover:-translate-y-0.5 hover:border-white/10"
    >
      {/* Cover — превью кадра записи во всю ширину, с откатом на Film-плейсхолдер.
          16:9 блок: object-cover заполняет его без чёрных полос по бокам. */}
      <div className="relative aspect-video w-full overflow-hidden bg-gradient-to-br from-brand-from/20 via-brand-to/10 to-transparent">
        <div
          className="absolute inset-0 opacity-40"
          style={{
            backgroundImage:
              "radial-gradient(600px circle at 30% 20%, rgb(139 92 246 / 0.35), transparent 50%)",
          }}
        />
        {showThumb ? (
          <img
            src={thumbSrc}
            alt=""
            loading="lazy"
            draggable={false}
            onError={() => setThumbFailed(true)}
            className="absolute inset-0 h-full w-full object-cover"
          />
        ) : (
          <div className="absolute inset-0 flex items-center justify-center">
            <Film className="h-8 w-8 text-white/40" />
          </div>
        )}
        {durationLabel && (
          <div className="absolute right-2 top-2 rounded-md bg-black/50 px-1.5 py-0.5 font-mono-tabular text-[11px] text-white backdrop-blur">
            {durationLabel}
          </div>
        )}

        {/* Меню действий «⋯» — левый верхний угол, чтобы не налезать на бейдж
            длительности справа. Появляется на hover карточки (и всегда виден
            при открытом меню — data-[state=open]). */}
        <DropdownMenu>
          <DropdownMenuTrigger
            aria-label="Действия с проектом"
            className="absolute left-2 top-2 rounded-md bg-black/50 p-1 text-white/80 opacity-0
                       backdrop-blur transition-opacity hover:bg-black/70 hover:text-white
                       focus-visible:opacity-100 focus-visible:outline-none focus-visible:ring-2
                       focus-visible:ring-brand-from group-hover:opacity-100
                       data-[state=open]:opacity-100"
          >
            <MoreVertical className="h-4 w-4" />
          </DropdownMenuTrigger>
          <DropdownMenuContent align="start">
            <DropdownMenuItem variant="danger" onSelect={() => onDelete(project)}>
              <Trash2 className="h-4 w-4" /> Удалить проект
            </DropdownMenuItem>
          </DropdownMenuContent>
        </DropdownMenu>
      </div>

      {/* Body */}
      <div className="flex flex-1 flex-col gap-3 p-4">
        <div className="min-w-0">
          <h3 className="truncate text-sm font-semibold text-fg" title={title}>
            {title}
          </h3>
          <div className="mt-0.5 flex items-center justify-between gap-2 text-[11px] text-subtle">
            <span className="flex items-center gap-1.5">
              <Clock className="h-3 w-3" />
              <span>{fmtRelativeMs(project.updated_at_ms)}</span>
            </span>
            {timestamp && (
              <span className="text-[11px] text-subtle">{timestamp}</span>
            )}
          </div>
        </div>

        {/* Нижний блок: путь к файлу */}
        <div className="mt-auto flex min-w-0 flex-col gap-1.5 border-t border-white/5 pt-3 text-[11px] text-subtle">
          <div
            className="flex min-w-0 items-center gap-1.5"
            title={project.decisions_path}
          >
            <FileText className="h-3 w-3 shrink-0" />
            <span className="truncate font-mono text-[10px]">
              {shortenPath(project.decisions_path)}
            </span>
          </div>
        </div>

        <div className="pt-1">
          <Button
            variant="secondary"
            size="sm"
            className="w-full"
            onClick={() => onOpen(project.decisions_path)}
          >
            Открыть
          </Button>
        </div>
      </div>
    </Card>
  );
}

// -------- Error banner -------------------------------------------------------

/**
 * DashboardError — баннер ошибки загрузки списка. Различает два случая:
 *  - похоже на «backend ещё стартует» (fetch failed / connection refused) —
 *    показываем мягкий текст про запуск;
 *  - иначе это настоящая ошибка (500 и т.п.) — честно говорим об этом и даём
 *    развернуть технические детали, а не прячем их за «подождите пару секунд».
 */
function DashboardError({ error, onRetry }: { error: string; onRetry: () => void }) {
  const [open, setOpen] = useState(false);
  // Грубая эвристика: сетевые сбои до готовности backend'а не содержат HTTP-кода.
  const looksLikeStarting =
    /failed to fetch|networkerror|load failed|connection refused|econnrefused/i.test(error) &&
    !/HTTP \d/.test(error);

  return (
    <Card
      variant="surface"
      padding="sm"
      className="mb-4 flex flex-col gap-2 border-err/40 bg-err/10 text-sm text-fg"
    >
      <div className="flex items-center gap-3">
        <AlertCircle className="h-4 w-4 shrink-0 text-err" />
        <span className="flex-1">
          {looksLikeStarting
            ? "Приложение ещё запускается или не отвечает. Подождите пару секунд и нажмите «Повторить»."
            : "Не удалось загрузить список проектов."}
        </span>
        {!looksLikeStarting && (
          <button
            type="button"
            onClick={() => setOpen((v) => !v)}
            className="shrink-0 text-[11px] text-muted transition-colors hover:text-fg"
          >
            {open ? "Скрыть детали" : "Детали"}
          </button>
        )}
        <Button variant="ghost" size="sm" onClick={onRetry}>
          Повторить
        </Button>
      </div>
      {!looksLikeStarting && open && (
        <div className="max-h-32 overflow-auto break-all rounded bg-black/30 p-2 font-mono-tabular text-[11px] text-err/90">
          {error}
        </div>
      )}
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
          <div className="aspect-video w-full bg-elevated/40" />
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
  return deriveMeta(p).title;
}

/**
 * Разбор имени папки прогона `{basename}_{yyyymmdd_hhmm}` (см. backend
 * suggested_workdir) на человекочитаемое имя + отдельную метку даты/времени.
 * Метку показываем в нижнем блоке карточки, а не приклеиваем к заголовку.
 * Если суффикса-таймстампа нет — timestamp === null.
 */
function deriveMeta(p: ProjectMeta): { title: string; timestamp: string | null } {
  const parts = p.decisions_path.replace(/\\/g, "/").split("/");
  let folder = p.name;
  if (parts.length >= 2) {
    const parent = parts[parts.length - 2];
    if (parent && parent !== "." && !/^(cache|workdir|tmp)$/i.test(parent)) {
      folder = parent;
    }
  }
  const m = folder.match(/^(.*)_(\d{8})_(\d{4})$/);
  if (!m) return { title: folder, timestamp: null };
  const [, base, ymd, hm] = m;
  const y = Number(ymd.slice(0, 4));
  const mo = Number(ymd.slice(4, 6));
  const d = Number(ymd.slice(6, 8));
  const hh = hm.slice(0, 2);
  const mm = hm.slice(2, 4);
  const date = new Date(y, mo - 1, d);
  if (Number.isNaN(date.getTime())) return { title: folder, timestamp: null };
  const stamp = date.toLocaleDateString("ru-RU", { day: "numeric", month: "short" });
  return { title: base || folder, timestamp: `${stamp}, ${hh}:${mm}` };
}

/** `.../deep/nested/path/decisions.json` → `…/nested/path/decisions.json` (2 сегмента). */
function shortenPath(path: string): string {
  const parts = path.replace(/\\/g, "/").split("/");
  if (parts.length <= 3) return path;
  return "…/" + parts.slice(-3).join("/");
}

/** Папка проекта — родитель decisions.json (то, что удалит delete_files=true). */
function projectDir(decisionsPath: string): string {
  const norm = decisionsPath.replace(/\\/g, "/");
  const idx = norm.lastIndexOf("/");
  return idx > 0 ? norm.slice(0, idx) : norm;
}
