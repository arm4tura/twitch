import { useMemo } from "react";
import { Play, Pause, Trash2, Volume2, VolumeX } from "lucide-react";
import { cn } from "../../lib/cn";
import { fmtMs } from "../../lib/format";
import { Button } from "../ui/Button";
import { TranscriptContext } from "./TranscriptContext";
import type { ContextWord } from "../../lib/transcript";

/**
 * MuteInspector — панель ОДНОГО выбранного мата под таймлайном (заменяет
 * раскрывающуюся часть прежнего MuteRow). Это не список, а контекст текущего
 * выбора: статус, глушить/оставить, точная подстройка краёв, превью и
 * транскрипт-контекст со снапом к словам.
 *
 * В шапке ДВА текста, и это не дубль: `word` — что ПОЙМАЛ детектор (лежит в
 * decisions.json и не меняется от правки границ), «накрыто» — что реально попадает
 * под текущее окно. Второе считается здесь из тех же слов, что подсвечены красным
 * ниже, а не хранится: производную от границ и транскрипта нельзя держать в файле,
 * иначе она разъезжается с границами (см. useMuteEditing — раньше она затирала
 * слово детектора). Расходятся эти два текста ровно тогда, когда человек сдвинул
 * окно, — и тогда оба видны.
 *
 * Всё в ЛОКАЛЬНЫХ секундах (как start/end мьюта и транскрипт). Границы правятся
 * тремя путями, как и раньше:
 *   - клик по слову в транскрипте → onAdjust (снап к словам);
 *   - кнопки ±0.1с → onNudge (точный сдвиг без снапа);
 *   - драг на таймлайне (в родителе) → adjustMuteFree.
 */

const NUDGE = 0.1; // шаг точной подстройки, сек

export interface MuteInspectorProps {
  /** Порядковый номер выбранного (1-based) и всего матов — для «мат N из M». */
  index: number;
  total: number;
  start: number; // локальные секунды
  end: number;
  /** Что поймал детектор (matched-токены). Правка границ его не меняет. */
  word?: string;
  /** true → будет заглушено (accepted); false → оставлено (rejected). */
  muted: boolean;
  review?: boolean;
  /** Играет ли сейчас этот мат (превью «как в экспорте»). */
  playing: boolean;
  /** Слова контекста вокруг мата (из родителя). */
  contextWords: ContextWord[];
  /** Плейхед в локальных секундах — подсветить играющее слово. */
  currentS?: number;
  onToggleMuted: () => void;
  onDelete: () => void;
  /** ▶ Как в экспорте — мат в контексте с реалтайм-заглушением. */
  onPlayExport: () => void;
  /** ▶ Оригинал — тот же отрезок, но со звуком (поймать мат). */
  onPlayOriginal: () => void;
  /** Правка со снапом к словам (клик по слову). */
  onAdjust: (start: number, end: number) => void;
  /** Проиграть отрезок оригинала (двойной клик по слову), локальные секунды. */
  onPlayRange: (start: number, end: number) => void;
  /** Точный сдвиг границ на дельту (секунды), без снапа. */
  onNudge: (deltaStart: number, deltaEnd: number) => void;
  className?: string;
}

export function MuteInspector({
  index,
  total,
  start,
  end,
  word,
  muted,
  review,
  playing,
  contextWords,
  currentS,
  onToggleMuted,
  onDelete,
  onPlayExport,
  onPlayOriginal,
  onAdjust,
  onPlayRange,
  onNudge,
  className,
}: MuteInspectorProps) {
  // Реально накрытый текст — из тех же слов, что TranscriptContext подсветит
  // красным ниже: один источник, поэтому шапка и транскрипт не могут разойтись.
  const covered = useMemo(
    () =>
      contextWords
        .filter((w) => w.inMute)
        .map((w) => w.word)
        .join(" ")
        .trim(),
    [contextWords]
  );
  const caught = word?.trim() ?? "";
  // Пустое «накрыто» не печатаем: окно в тишине и так видно по транскрипту, где
  // ни одно слово не подсвечено.
  const showCovered = covered !== "" && covered !== caught;

  return (
    <div
      className={cn(
        "mute-inspector space-y-3 rounded-xl border border-white/[0.08] bg-surface/50 p-3",
        className
      )}
    >
      {/* Статусная строка */}
      <div className="flex flex-wrap items-center gap-x-3 gap-y-1.5">
        <span className="font-mono text-[11px] text-subtle">
          мат {index} из {total}
        </span>
        <span
          className={cn(
            "min-w-0 truncate text-sm font-medium",
            muted ? "text-fg" : "text-muted line-through"
          )}
          title={word}
        >
          {caught || "— без слова —"}
        </span>
        {showCovered && (
          <span
            className="min-w-0 truncate text-[11px] text-subtle"
            title={`Под окном сейчас: ${covered}`}
          >
            накрыто: {covered}
          </span>
        )}
        <span className="font-mono text-[11px] tabular-nums text-subtle">
          {fmtMs(start * 1000)} · {Math.round((end - start) * 1000)} мс
        </span>
        {review && (
          <span className="rounded-full bg-warn/15 px-2 py-0.5 text-[10px] font-medium uppercase tracking-wider text-warn">
            под вопросом
          </span>
        )}


      </div>

      {/* Подстройка краёв + глушить/оставить + удалить */}
      <div className="flex flex-wrap items-center gap-x-4 gap-y-2">
        <div className="inspector-actions flex items-center gap-2">
          <Button
            size="sm"
            variant={muted ? "secondary" : "ghost"}
            onClick={onToggleMuted}
            className={cn(muted ? "text-err" : "text-muted")}
            aria-label={muted ? "Оставить (не глушить)" : "Глушить"}
          >
            {muted ? (
              <>
                <VolumeX className="mr-1.5 h-3.5 w-3.5" /> глушим
              </>
            ) : (
              <>
                <Volume2 className="mr-1.5 h-3.5 w-3.5" /> оставили
              </>
            )}
          </Button>
          <Button
            size="sm"
            variant="ghost"
            onClick={onDelete}
            className="text-subtle hover:text-err"
          >
            <Trash2 className="mr-1.5 h-3.5 w-3.5" /> Удалить
          </Button>
        </div>
        <EdgeNudge
          label="Начало"
          time={start}
          onMinus={() => onNudge(-NUDGE, 0)}
          onPlus={() => onNudge(NUDGE, 0)}
        />
        <EdgeNudge
          label="Конец"
          time={end}
          onMinus={() => onNudge(0, -NUDGE)}
          onPlus={() => onNudge(0, NUDGE)}
        />


      </div>
        <div className="ml-auto flex items-center gap-2">
          <Button
            size="sm"
            variant={playing ? "primary" : "secondary"}
            onClick={onPlayExport}
            aria-label="Прослушать как в экспорте"
          >
            {playing ? <Pause className="mr-1.5 h-3.5 w-3.5" /> : <Play className="mr-1.5 h-3.5 w-3.5" />}
            Как в экспорте
          </Button>
          <Button size="sm" variant="ghost" onClick={onPlayOriginal} aria-label="Прослушать оригинал">
            <Play className="mr-1.5 h-3.5 w-3.5" /> Оригинал
          </Button>
        </div>
      {/* Транскрипт-контекст (снап к словам) */}
      <TranscriptContext
        words={contextWords}
        muteStart={start}
        muteEnd={end}
        currentS={currentS}
        onAdjust={onAdjust}
        onPlayRange={onPlayRange}
      />

    </div>
  );
}

/**
 * EdgeNudge — метка края + текущий таймкод + кнопки −0.1/+0.1с. Точная
 * подстройка для тех, кому неудобно попадать мышью по краю блока.
 */
function EdgeNudge({
  label,
  time,
  onMinus,
  onPlus,
}: {
  label: string;
  time: number;
  onMinus: () => void;
  onPlus: () => void;
}) {
  return (
    <div className="flex items-center gap-1.5">
      <span className="text-[11px] text-subtle">{label}</span>
      <Button
        size="sm"
        variant="secondary"
        onClick={onMinus}
        className="h-7 px-2 font-mono text-xs"
        aria-label={`${label}: −0.1 секунды`}
      >
        −0.1с
      </Button>
      <span className="min-w-[64px] text-center font-mono text-[11px] tabular-nums text-fg">
        {fmtMs(time * 1000)}
      </span>
      <Button
        size="sm"
        variant="secondary"
        onClick={onPlus}
        className="h-7 px-2 font-mono text-xs"
        aria-label={`${label}: +0.1 секунды`}
      >
        +0.1с
      </Button>
    </div>
  );
}
