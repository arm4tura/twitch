import { SourcePoster } from "../components/SourcePoster";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  Clapperboard,
  ClipboardCopy,
  ExternalLink,
  FileText,
  FolderOpen,
  Loader2,
  MousePointer2,
  Package,
  Pause,
  Play,
  Sparkles,
} from "lucide-react";
import {
  allowMediaPaths,
  createHighlightsExportJob,
  createHighlightsImportJob,
  createRenderHighlightsJob,
  getPrompt,
  mediaUrl,
  thumbnailUrl,
  readDecisions,
  type Highlight,
  type HighlightsExportResult,
  type JobState,
  type RenderHighlightsResult,
} from "../api";
import { useJobEvents } from "../hooks/useJobEvents";
import { Button } from "../components/ui/Button";
import { Card, CardDescription, CardTitle } from "../components/ui/Card";
import { DraggableFile } from "../components/ui/DraggableFile";
import { FileField } from "../components/ui/FileField";
import { Label } from "../components/ui/Input";
import { toast } from "../components/ui/Toast";
import { fmtS } from "../lib/format";

/**
 * AiTiktokScreen — всё что связано с NotebookLM и нарезкой клипов.
 *
 * Поток работы:
 *   1. Собрать пакет (ZIP с транскриптом) → загрузить в NotebookLM.
 *   2. Скопировать промпт → вставить в NotebookLM → получить JSON.
 *   3. Вставить ответ → импортировать хайлайты.
 *   4. Нарезать клипы → перетащить готовые mp4 прямо в браузер/TikTok/YouTube.
 *
 * Клипы после нарезки показываются как DraggableFile-плитки: тащи в браузер
 * или кликни «Показать в проводнике».
 */

export interface AiTiktokScreenProps {
  decisionsPath: string;
  onJobStarted: (jobId: string) => void;
}

interface DocSummary {
  highlights: Highlight[];
  stream: string | null;
  workdir: string | null;
}

function extractSummary(doc: any): DocSummary {
  const meta = (doc?._meta ?? {}) as Record<string, unknown>;
  const hlBundle = doc?.highlights;
  const hlArray: any[] = Array.isArray(hlBundle?.highlights)
    ? hlBundle.highlights
    : Array.isArray(hlBundle)
      ? hlBundle
      : [];
  return {
    highlights: hlArray as Highlight[],
    stream:
      (meta.stream_path as string | undefined) ??
      (typeof doc?.source === "string" ? doc.source : undefined) ??
      (typeof doc?.stream === "string" ? doc.stream : undefined) ??
      (typeof doc?.original === "string" ? doc.original : undefined) ??
      null,
    workdir: (meta.workdir as string | undefined) ?? null,
  };
}

function absBounds(h: Highlight): [number, number] {
  return [h.stream_start_s ?? h.start_s, h.stream_end_s ?? h.end_s];
}

function terse(e: unknown, max = 160): string {
  let msg = e instanceof Error ? e.message : String(e);
  const colon = msg.indexOf(": ");
  if (colon >= 0 && colon < 40) msg = msg.slice(colon + 2);
  msg = msg.replace(/\s+/g, " ").trim();
  return msg.length > max ? msg.slice(0, max - 1) + "…" : msg;
}

export function AiTiktokScreen({ decisionsPath, onJobStarted }: AiTiktokScreenProps) {
  const [summary, setSummary] = useState<DocSummary | null>(null);
  const [reloadTick, setReloadTick] = useState(0);

  // Пути для пакета/клипов — дефолты рядом с decisions.json.
  const { sep, dirOf } = useMemo(() => {
    const s = decisionsPath.includes("\\") ? "\\" : "/";
    const idx = decisionsPath.lastIndexOf(s);
    return { sep: s, dirOf: idx > 0 ? decisionsPath.slice(0, idx) : "" };
  }, [decisionsPath]);

  // Каталог пакета выводим из папки проекта (пользователю его не показываем —
  // техническая деталь). Каталог клипов остаётся настраиваемым ниже.
  const packageDir = useMemo(
    () => (dirOf ? `${dirOf}${sep}notebooklm` : ""),
    [dirOf, sep]
  );
  const [outDir, setOutDir] = useState(() =>
    dirOf ? `${dirOf}${sep}highlights` : ""
  );
  const [pasteText, setPasteText] = useState("");
  const [nHighlights, setNHighlights] = useState(5);
  const [copyingPrompt, setCopyingPrompt] = useState(false);
  const [busy, setBusy] = useState<null | "package" | "import">(null);
  const [packageJobId, setPackageJobId] = useState<string | null>(null);
  // Файлы собранного пакета для drag-out в NotebookLM: transcript_*.md как
  // источники. prompt.md отдельно не показываем — его копируют кнопкой.
  const [packageFiles, setPackageFiles] = useState<{
    dir: string;
    sources: { path: string; title: string; subtitle: string }[];
  } | null>(null);
  const [renderJobId, setRenderJobId] = useState<string | null>(null);
  const [clips, setClips] = useState<RenderHighlightsResult["clips"] | null>(null);

  // Превью-плеер.
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const [previewIdx, setPreviewIdx] = useState<number | null>(null);
  const [playingIdx, setPlayingIdx] = useState<number | null>(null);
  const stopAtRef = useRef<number>(0);
  const mediaAllowedRef = useRef<string | null>(null);

  useEffect(() => {
    let alive = true;
    readDecisions(decisionsPath)
      .then((doc) => alive && setSummary(extractSummary(doc)))
      .catch(() => alive && setSummary(null));
    return () => { alive = false; };
  }, [decisionsPath, reloadTick]);

  useEffect(() => {
    const stream = summary?.stream;
    if (!stream || mediaAllowedRef.current === stream) return;
    mediaAllowedRef.current = stream;
    allowMediaPaths([stream]).catch(() => {});
  }, [summary?.stream]);

  const sorted = useMemo(
    () =>
      (summary?.highlights ?? [])
        .map((h, i) => ({ h, i }))
        .sort((a, b) => b.h.score - a.h.score),
    [summary?.highlights]
  );

  const renderJob = useJobEvents(
    renderJobId,
    useCallback((state: JobState) => {
      if (state.status === "done" && state.result) {
        const r = state.result as unknown as RenderHighlightsResult;
        setClips(r.clips);
        toast.success(`Нарезано клипов: ${r.count}`, {
          description: r.out_dir,
          action: {
            label: "Показать",
            onClick: async () => {
              const ok = await window.twitchCut.showInFolder(r.out_dir);
              if (!ok) toast.error("Не удалось открыть каталог");
            },
          },
        });
      } else if (state.status === "failed") {
        toast.error("Нарезка клипов упала", { description: state.error ?? undefined });
      }
    }, [])
  );

  const renderRunning =
    renderJob.state?.status === "running" || renderJob.state?.status === "pending";

  // Пакет для NotebookLM: по завершении собираем абсолютные пути source-файлов
  // (transcript_*.md) + prompt.md, чтобы показать их как DraggableFile-плитки.
  const packageJob = useJobEvents(
    packageJobId,
    useCallback(
      (state: JobState) => {
        if (state.status === "done" && state.result) {
          const r = state.result as unknown as HighlightsExportResult;
          const dir = r.out_dir;
          const join = (name: string) => `${dir}${sep}${name}`;
          const sources = (r.manifest?.chunks ?? []).map((c) => ({
            path: join(c.path),
            title: c.path,
            subtitle: `${(c.word_count ?? 0).toLocaleString("ru")} слов · тащи в NotebookLM`,
          }));
          setPackageFiles({
            dir,
            sources,
          });
          toast.success(`Пакет собран: ${sources.length} файл(ов)`, {
            description: "Перетащите источники в NotebookLM",
          });
        } else if (state.status === "failed") {
          toast.error("Сборка пакета упала", { description: state.error ?? undefined });
        }
      },
      [sep]
    )
  );

  const packageRunning =
    packageJob.state?.status === "running" || packageJob.state?.status === "pending";

  // --- превью ----------------------------------------------------------------
  const togglePlay = async (idx: number, h: Highlight) => {
    const el = videoRef.current;
    const stream = summary?.stream;
    if (!el || !stream) return;
    if (playingIdx === idx) { el.pause(); setPlayingIdx(null); return; }
    const [s, e] = absBounds(h);
    setPreviewIdx(idx);
    stopAtRef.current = e;
    if (mediaAllowedRef.current !== stream) {
      mediaAllowedRef.current = stream;
      await allowMediaPaths([stream]).catch(() => {});
    }
    if (!el.src) el.src = mediaUrl(stream);
    el.currentTime = s;
    try { await el.play(); setPlayingIdx(idx); }
    catch (err) { setPlayingIdx(null); toast.error("Не удалось воспроизвести", { description: terse(err) }); }
  };

  const onTimeUpdate = () => {
    const el = videoRef.current;
    if (el && el.currentTime >= stopAtRef.current) { el.pause(); setPlayingIdx(null); }
  };

  // --- копировать промпт -----------------------------------------------------
  const copyPrompt = async () => {
    setCopyingPrompt(true);
    try {
      const { prompt } = await getPrompt(nHighlights);
      await navigator.clipboard.writeText(prompt);
      toast.success("Промпт скопирован", { description: `${nHighlights} моментов` });
    } catch (e) {
      toast.error("Не удалось скопировать промпт", { description: terse(e) });
    } finally {
      setCopyingPrompt(false);
    }
  };

  // --- собрать пакет ---------------------------------------------------------
  const runPackage = async () => {
    if (!packageDir) { toast.error("Укажите каталог для пакета"); return; }
    setBusy("package");
    setPackageFiles(null);
    setPackageJobId(null);
    try {
      const job = await createHighlightsExportJob({
        decisions: decisionsPath,
        out_dir: packageDir,
        n_highlights: nHighlights,
      });
      setPackageJobId(job.id);
      onJobStarted(job.id);
    } catch (e) {
      toast.error("Ошибка сборки пакета", { description: terse(e) });
    } finally {
      setBusy(null);
    }
  };

  // --- импорт ответа ---------------------------------------------------------
  const runImport = async () => {
    if (!pasteText.trim()) { toast.error("Вставьте ответ в поле"); return; }
    setBusy("import");
    try {
      const job = await createHighlightsImportJob({
        decisions: decisionsPath,
        response_text: pasteText,
        output: decisionsPath,
      });
      onJobStarted(job.id);
      toast.success("Импорт запущен");
      setPasteText("");
      setTimeout(() => setReloadTick((t) => t + 1), 1200);
    } catch (e) {
      toast.error("Ошибка импорта", { description: terse(e) });
    } finally {
      setBusy(null);
    }
  };

  // --- нарезка клипов --------------------------------------------------------
  const runClips = async () => {
    const stream = summary?.stream;
    if (!stream) { toast.error("Путь к видео не найден"); return; }
    if (!outDir) { toast.error("Укажите каталог для клипов"); return; }
    setClips(null);
    setRenderJobId(null);
    try {
      const job = await createRenderHighlightsJob({
        decisions: decisionsPath,
        input_path: stream,
        out_dir: outDir,
        quality: "balanced",
        padding_s: 0.3,
      });
      setRenderJobId(job.id);
      onJobStarted(job.id);
    } catch (e) {
      toast.error("Не удалось запустить нарезку", { description: terse(e) });
    }
  };

  const hasHighlights = (summary?.highlights.length ?? 0) > 0;
  const stream = summary?.stream ?? null;

  return (
    <div className="screen-page clips-page">
      <header className="clips-heading"><h1 className="text-2xl font-semibold tracking-tight">Клипы</h1><p className="mt-1 text-sm text-muted">От транскрипта к готовым моментам — через NotebookLM.</p></header>
      <ol className="clip-stages"><li data-complete={!!packageFiles}>01 <span>Подготовить транскрипт</span></li><li data-complete={hasHighlights}>02 <span>Импортировать ответ</span></li><li data-complete={!!clips}>03 <span>Нарезать клипы</span></li></ol>
      <div className="clip-preview">
        <div className="mb-3 text-sm font-medium">Предпросмотр {previewIdx != null ? `· ${summary?.highlights[previewIdx]?.title ?? ""}` : "момента"}</div>
        {previewIdx == null && <SourcePoster key={stream} path={stream} />}
        <video hidden={previewIdx == null} ref={videoRef} poster={stream ? thumbnailUrl(stream) ?? undefined : undefined} onPlay={() => setPlayingIdx(previewIdx)} controls={previewIdx != null} onTimeUpdate={onTimeUpdate} onPause={() => setPlayingIdx(null)} onEnded={() => setPlayingIdx(null)} className="aspect-video w-full rounded-md bg-black" />
        <p className="mt-3 text-xs text-muted">{hasHighlights ? "Нажмите ▶ у момента, чтобы просмотреть его в исходной записи." : "После импорта здесь появится предпросмотр выбранного клипа."}</p>
      </div>

      {/* ── Шаг 1: Подготовить транскрипт для ИИ ── */}
      <Card variant="elevated" padding="md" className="clip-prepare">
        <details open={!hasHighlights}><summary className="cursor-pointer text-sm font-semibold">1. Подготовить транскрипт</summary>

        <CardDescription className="mt-1">
          Подготовьте файлы для NotebookLM, загрузите их и отправьте промпт для поиска моментов.
        </CardDescription>

        {/* Пока файлы не собраны — компактная настройка + одна кнопка. */}
        {!packageFiles && (
          <div className="mt-4 space-y-3">
            <div className="space-y-1.5">
              <Label eyebrow>Сколько моментов искать</Label>
              <input
                type="number"
                min={1}
                max={50}
                value={nHighlights}
                onChange={(e) => setNHighlights(Math.max(1, Math.min(50, Number(e.target.value) || 5)))}
                className="block w-24 rounded-lg border border-white/10 bg-black/30 px-2 py-1.5 text-center font-mono text-sm text-fg focus:border-brand-from/50 focus:outline-none focus:ring-2 focus:ring-brand-from/20"
              />
            </div>
            <Button
              onClick={runPackage}
              disabled={!packageDir}
              loading={busy === "package" || packageRunning}
            >
              <Package className="h-4 w-4" /> Подготовить транскрипт
            </Button>
          </div>
        )}

        {packageRunning && (
          <div className="mt-3 flex items-center gap-2 rounded-lg bg-brand-from/10 px-3 py-2 text-xs text-brand-from">
            <Loader2 className="h-4 w-4 animate-spin" />
            <span>{packageJob.state?.message || "Собираю транскрипт…"}</span>
          </div>
        )}

        {/* После сборки — файлы становятся центром экрана с пошаговой инструкцией. */}
        {packageFiles && packageFiles.sources.length > 0 && (
          <div className="mt-5 space-y-4">
            {/* Под-шаг A: перетащить файлы */}
            <div className="rounded-lg border border-brand-from/20 bg-brand-from/[0.04] p-3">
              <div className="mb-2 flex items-center gap-2 text-sm font-medium text-fg">
                <span className="flex h-5 w-5 items-center justify-center rounded-full bg-brand-from/20 text-[11px] font-bold text-brand-from">
                  A
                </span>
                Перетащите эти файлы в NotebookLM
              </div>
              <p className="mb-3 flex items-center gap-1.5 text-[11px] text-subtle">
                <MousePointer2 className="h-3 w-3" />
                Перетащите файл в NotebookLM или загрузите его из папки проекта.
              </p>
              <div className="space-y-2">
                {packageFiles.sources.map((f) => (
                  <DraggableFile
                    key={f.path}
                    path={f.path}
                    title={f.title}
                    subtitle={f.subtitle}
                    icon={<FileText className="h-3.5 w-3.5" />}
                    className="border-brand-from/40 hover:border-white/50"
                  />
                ))}
              </div>
              <div className="mt-3 flex flex-wrap gap-2">
                <Button
                  size="sm"
                  onClick={() => window.twitchCut.openExternal("https://notebooklm.google.com/")}
                >
                  <ExternalLink className="h-3.5 w-3.5" /> Открыть NotebookLM
                </Button>
                <Button
                  size="sm"
                  variant="secondary"
                  onClick={async () => {
                    const ok = await window.twitchCut.showInFolder(packageFiles.dir);
                    if (!ok) toast.error("Не удалось открыть папку");
                  }}
                >
                  <FolderOpen className="h-3.5 w-3.5" /> Открыть папку
                </Button>
              </div>
            </div>

            {/* Под-шаг B: скопировать промпт */}
            <div className="rounded-lg border border-brand-from/20 bg-brand-from/[0.04] p-3">
              <div className="mb-2 flex items-center gap-2 text-sm font-medium text-fg">
                <span className="flex h-5 w-5 items-center justify-center rounded-full bg-brand-from/20 text-[11px] font-bold text-brand-from">
                  B
                </span>
                Скопируйте промпт и вставьте его в чат NotebookLM
              </div>
              <p className="mb-3 text-[11px] text-subtle">
                Вставьте промпт в чат NotebookLM. Полученный ответ импортируйте в шаге 2.
              </p>
              <Button size="sm" variant="secondary" onClick={copyPrompt} loading={copyingPrompt}>
                <ClipboardCopy className="h-3.5 w-3.5" /> Скопировать промпт
              </Button>
            </div>

            <button
              type="button"
              onClick={runPackage}
              className="text-[11px] text-subtle underline-offset-2 transition-colors hover:text-fg hover:underline"
            >
              Пересобрать транскрипт заново
            </button>
          </div>
        )}
        </details>
      </Card>

      {/* ── Шаг 2: Вставить ответ ── */}
      <Card variant="elevated" padding="md" className="clip-import">
        <details open={!hasHighlights}><summary className="cursor-pointer text-sm font-semibold">2. Вставить ответ ИИ</summary>

        <CardDescription className="mt-1">
          Скопируйте ответ модели целиком — JSON с пояснениями принимается.
          Частично битый ответ разберём по годным моментам.
        </CardDescription>
        <textarea
          value={pasteText}
          onChange={(e) => setPasteText(e.target.value)}
          placeholder={'Вставьте ответ (```json { "highlights": [...] } ```)'}
          rows={4}
          className="mt-3 w-full resize-y rounded-lg border border-white/10 bg-black/30 px-3 py-2 font-mono text-xs text-fg placeholder:text-subtle focus:border-brand-from/50 focus:outline-none focus:ring-2 focus:ring-brand-from/20"
        />
        <div className="mt-3 flex justify-end">
          <Button onClick={runImport} disabled={!pasteText.trim()} loading={busy === "import"}>
            Импортировать
          </Button>
        </div>
        </details>
      </Card>

      {/* ── Список моментов ── */}
      {hasHighlights ? (
        <Card variant="elevated" padding="md" className="clip-moments">
          <div className="mb-4 flex items-center justify-between">
            <div>
              <CardTitle>Моменты ({summary!.highlights.length})</CardTitle>
              <CardDescription className="mt-0.5">
                По оценке ↓ · ▶ — превью из исходника · тащите клип в браузер
              </CardDescription>
            </div>
          </div>
          <div className="space-y-2">
            {sorted.map(({ h, i }) => {
              const [s, e] = absBounds(h);
              const dur = e - s;
              const isPlaying = playingIdx === i;
              // Если клип уже нарезан — показываем DraggableFile поверх карточки.
              const clip = clips?.find((c) => c.index === i);
              return (
                <div key={i} className="flex items-start gap-3 rounded-lg border border-white/5 bg-black/20 p-3">
                  <button
                    type="button"
                    onClick={() => togglePlay(i, h)}
                    disabled={!stream}
                    className="mt-0.5 flex h-9 w-9 shrink-0 items-center justify-center rounded-full bg-brand-from/20 text-brand-from transition-colors hover:bg-brand-from/30 disabled:opacity-40"
                    aria-label={isPlaying ? "Пауза" : "Играть"}
                  >
                    {isPlaying ? <Pause className="h-4 w-4" /> : <Play className="h-4 w-4 translate-x-0.5" />}
                  </button>
                  <div className="min-w-0 flex-1 space-y-1.5">
                    <div className="flex items-center gap-2">
                      <span className="truncate text-sm font-medium text-fg">{h.title}</span>
                      <span className="shrink-0 font-mono text-[11px] text-subtle">
                        {fmtS(s)}–{fmtS(e)} · {dur.toFixed(0)}с
                      </span>
                    </div>
                    {h.reason && <p className="line-clamp-2 text-xs text-muted">{h.reason}</p>}
                    {h.quote && <p className="truncate text-[11px] italic text-subtle">«{h.quote}»</p>}
                    <div className="flex items-center gap-2">
                      <div className="h-1 flex-1 overflow-hidden rounded-full bg-white/8">
                        <div
                          className="h-full rounded-full bg-brand"
                          style={{ width: `${Math.round(h.score * 100)}%` }}
                        />
                      </div>
                      <span className="w-9 shrink-0 text-right font-mono text-[11px] text-subtle">
                        {h.score.toFixed(2)}
                      </span>
                    </div>
                    {clip && (
                      <DraggableFile
                        path={clip.path}
                        title={clip.title}
                        subtitle={`${dur.toFixed(0)}с · тащи в браузер или TikTok`}
                        icon={<Clapperboard className="h-3.5 w-3.5" />}
                      />
                    )}
                  </div>
                </div>
              );
            })}
          </div>
        </Card>
      ) : (
        <Card variant="surface" padding="lg" className="clip-moments text-center">
          <Sparkles className="mx-auto mb-2 h-6 w-6 text-subtle" />
          <div className="text-sm font-medium text-muted">Хайлайтов пока нет</div>
          <div className="mt-1 text-xs text-subtle">
            Соберите пакет, получите ответ ИИ и импортируйте его выше.
          </div>
        </Card>
      )}

      {/* ── Шаг 3: Нарезать клипы ── */}
      {hasHighlights && (
        <Card variant="elevated" padding="md" className="clip-render flex flex-col">
          <div className="mb-3 inline-flex h-9 w-9 items-center justify-center rounded-lg bg-brand-from/10 text-brand-from">
            <Clapperboard className="h-4 w-4" />
          </div>
          <CardTitle>3. Нарезать клипы</CardTitle>
          <CardDescription className="mt-1">
            Каждый момент → отдельный mp4. После нарезки тащите клип прямо в браузер.
          </CardDescription>
          <div className="mt-4 flex-1 space-y-2">
            <Label eyebrow>Каталог для клипов</Label>
            <FileField
              kind="directory"
              value={outDir}
              onChange={setOutDir}
              defaultPath={dirOf || undefined}
              placeholder="…/highlights/"
            />
          </div>
          {renderRunning && (
            <div className="mt-3 flex items-center gap-2 rounded-lg bg-brand-from/10 px-3 py-2 text-xs text-brand-from">
              <Loader2 className="h-4 w-4 animate-spin" />
              <span>{renderJob.state?.message || `${Math.round(renderJob.state?.progress ?? 0)}%`}</span>
            </div>
          )}
          {clips && (
            <div className="mt-3 rounded-lg bg-ok/10 px-3 py-2 text-xs text-ok">
              Готово: {clips.length} клип(ов). Тащите карточки выше в браузер.
            </div>
          )}
          <Button
            className="mt-4 self-start"
            onClick={runClips}
            disabled={!stream || !outDir || renderRunning}
            loading={renderRunning}
          >
            <Clapperboard className="h-4 w-4" /> Нарезать {summary!.highlights.length} клип(ов)
          </Button>
          {!stream && (
            <p className="mt-2 text-[11px] text-warn">
              Путь к исходному видео не найден в decisions.json.
            </p>
          )}
        </Card>
      )}
    </div>
  );
}
