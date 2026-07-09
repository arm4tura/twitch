import { useMemo } from "react";
import { Volume2, VolumeX } from "lucide-react";
import { cn } from "../../lib/cn";
import type { ContextWord } from "../../lib/transcript";

/**
 * TranscriptContext — вариант B: читаемая фраза вокруг мьюта вместо голой волны.
 *
 * Показываем реальные слова из ASR (пословный тайминг). Слова, попавшие под
 * мьют, подсвечены красным (их не будет слышно). Клик по слову РЕДАКТИРУЕТ
 * границы мьюта, прилипая к границам этого слова («снап к словам»):
 *   - слово вне мьюта → мьют расширяется, чтобы его накрыть;
 *   - крайнее слово внутри мьюта → мьют сжимается, отпуская его;
 *   - внутреннее слово → без изменений (двойной клик — прослушать).
 *
 * Двойной клик по любому слову — проиграть именно его (превью).
 *
 * Так пользователь правит «какие слова глушим» на уровне слов, а не вбивает
 * таймкоды руками.
 */

export interface TranscriptContextProps {
  words: ContextWord[];
  /** Текущие границы мьюта (секунды) — для вычисления снапа. */
  muteStart: number;
  muteEnd: number;
  /** Секунда воспроизведения — подсветить играющее слово. */
  currentS?: number;
  /** Новый интервал мьюта после клика по слову. */
  onAdjust: (start: number, end: number) => void;
  /** Проиграть отрезок (двойной клик по слову — превью одного слова). */
  onPlayRange?: (start: number, end: number) => void;
  className?: string;
}

export function TranscriptContext({
  words,
  muteStart,
  muteEnd,
  currentS,
  onAdjust,
  onPlayRange,
  className,
}: TranscriptContextProps) {
  // Индексы первого/последнего слова под мьютом — нужны для логики «край vs нутро».
  const { firstHit, lastHit } = useMemo(() => {
    let f = -1;
    let l = -1;
    words.forEach((w, i) => {
      if (w.inMute) {
        if (f === -1) f = i;
        l = i;
      }
    });
    return { firstHit: f, lastHit: l };
  }, [words]);

  const handleClick = (w: ContextWord, i: number) => {
    if (!w.inMute) {
      // Слово вне мьюта → расширяем оболочку, чтобы его накрыть (снап к слову).
      onAdjust(Math.min(muteStart, w.start), Math.max(muteEnd, w.end));
      return;
    }
    // Слово внутри: отпускаем только с края, иначе (внутреннее) не трогаем.
    if (i === firstHit && firstHit !== lastHit) {
      onAdjust(words[i + 1]?.start ?? w.end, muteEnd);
    } else if (i === lastHit && firstHit !== lastHit) {
      onAdjust(muteStart, words[i - 1]?.end ?? w.start);
    }
    // firstHit === lastHit (единственное слово) — не даём «схлопнуть» мьют в ноль.
  };

  if (!words.length) {
    return (
      <div className={cn("text-xs text-subtle", className)}>
        Нет транскрипта для этого места — правьте границы на волне.
      </div>
    );
  }

  return (
    <div className={cn("space-y-2", className)}>
      <div className="flex items-center gap-1.5 text-[11px] uppercase tracking-wider text-subtle">
        <VolumeX className="h-3 w-3 text-err" /> заглушено
        <span className="mx-1 text-white/15">·</span>
        <Volume2 className="h-3 w-3 text-muted" /> слышно
        <span className="ml-auto normal-case tracking-normal text-subtle/70">
          клик — глушить/вернуть слово · двойной — прослушать
        </span>
      </div>
      <div className="flex flex-wrap gap-1 leading-relaxed">
        {words.map((w, i) => {
          const playing =
            currentS != null && currentS >= w.start && currentS < w.end;
          return (
            <button
              key={`${w.start}-${i}`}
              type="button"
              onClick={() => handleClick(w, i)}
              onDoubleClick={() => onPlayRange?.(w.start, w.end)}
              title={`${w.start.toFixed(2)}–${w.end.toFixed(2)}s`}
              className={cn(
                "rounded px-1.5 py-0.5 text-sm transition-colors",
                w.inMute
                  ? "bg-err/20 text-err hover:bg-err/30 line-through decoration-err/50"
                  : "text-fg/80 hover:bg-white/10",
                playing && "ring-1 ring-white/60"
              )}
            >
              {w.word}
            </button>
          );
        })}
      </div>
    </div>
  );
}
