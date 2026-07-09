/**
 * Утилиты над транскриптом для экрана «Правка».
 *
 * Ядро вариантов B (контекст-транскрипт) и «снап к словам»: у нас есть
 * пословный тайминг из ASR (GigaAM/WhisperX), и мы используем его чтобы
 *  1) показать реальные слова вокруг мьюта (а не голую волну),
 *  2) прилипать границами мьюта к границам слов при подрезке.
 *
 * Все времена — СЕКУНДЫ (как в decisions.json и transcript.json).
 */

import type { TranscriptDoc, TranscriptWord } from "../types/project";

/** Слово индекса + номер сегмента, из которого оно пришло (для клэмпа контекста). */
export interface IndexedWord extends TranscriptWord {
  /** Индекс ASR-сегмента. -1 если транскрипт без сегментов (только word_segments). */
  seg: number;
}

/** Плоский, отсортированный по времени список слов — индекс для всех операций. */
export interface WordIndex {
  words: IndexedWord[];
}

/**
 * Собирает плоский индекс: приоритет segments[].words (несёт границы сегмента —
 * нужны чтобы контекст не залезал в соседний момент), иначе — плоский
 * word_segments (тогда seg=-1, клэмпа по сегменту нет).
 */
export function buildWordIndex(doc: TranscriptDoc | null | undefined): WordIndex {
  if (!doc) return { words: [] };
  let flat: IndexedWord[] = [];
  if (Array.isArray(doc.segments) && doc.segments.length) {
    doc.segments.forEach((seg, si) => {
      if (Array.isArray(seg.words)) {
        for (const w of seg.words) flat.push({ ...w, seg: si });
      }
    });
  } else if (Array.isArray(doc.word_segments) && doc.word_segments.length) {
    flat = doc.word_segments.map((w) => ({ ...w, seg: -1 }));
  }
  // Оставляем только слова с валидным таймингом и сортируем по началу.
  flat = flat
    .filter((w) => Number.isFinite(w.start) && Number.isFinite(w.end) && w.end > w.start)
    .sort((a, b) => a.start - b.start);
  return { words: flat };
}

/** Индекс первого слова, чьё `end` > t (нижняя граница). Бинарный поиск. */
function lowerBound(words: IndexedWord[], t: number): number {
  let lo = 0;
  let hi = words.length;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (words[mid].end > t) hi = mid;
    else lo = mid + 1;
  }
  return lo;
}

/**
 * Слова, пересекающиеся с интервалом [start, end] (даже частично).
 * Используется чтобы подсветить, что именно попадает под мьют.
 */
export function wordsInRange(
  index: WordIndex,
  start: number,
  end: number
): IndexedWord[] {
  const { words } = index;
  const out: IndexedWord[] = [];
  for (let i = lowerBound(words, start); i < words.length; i++) {
    const w = words[i];
    if (w.start >= end) break;
    if (w.end > start && w.start < end) out.push(w);
  }
  return out;
}

export interface ContextWord extends IndexedWord {
  /** Пересекается ли слово с текущим интервалом мьюта (→ подсветка красным). */
  inMute: boolean;
}

/**
 * Контекст вокруг мьюта: `padBefore` слов слева и `padAfter` справа от [start,end]
 * плюс сами попавшие слова. Даёт вариант B — «читаемая фраза».
 *
 * Контекст НЕ выходит за границы сегмента, в котором лежит мьют (сегмент ASR ≈
 * предложение). Иначе в панель затекают слова из соседнего момента.
 *
 * Слова, попавшие под ДРУГИЕ мьюты (`otherRanges`), полностью исключаются — они
 * относятся к своим заглушкам и не должны мозолить глаз в этой строке.
 */
export function contextWords(
  index: WordIndex,
  start: number,
  end: number,
  padBefore = 1,
  padAfter = 2,
  otherRanges: Array<{ start: number; end: number }> = []
): ContextWord[] {
  const { words } = index;
  if (!words.length) return [];
  // Границы попадания.
  let firstHit = -1;
  let lastHit = -1;
  for (let i = 0; i < words.length; i++) {
    const w = words[i];
    if (w.end > start && w.start < end) {
      if (firstHit === -1) firstHit = i;
      lastHit = i;
    }
    if (w.start >= end) break;
  }
  // Если ни одно слово не попало (мьют в тишине) — центрируемся по времени.
  if (firstHit === -1) {
    const center = lowerBound(words, (start + end) / 2);
    firstHit = Math.min(center, words.length - 1);
    lastHit = firstHit - 1; // пустой диапазон попадания
  }
  const anchor = firstHit;
  // Сегмент якорного слова — контекст не должен из него вылезать.
  const seg = words[anchor]?.seg ?? -1;
  const inSeg = (i: number) => seg < 0 || words[i]?.seg === seg;

  let from = Math.max(0, firstHit - padBefore);
  let to = Math.min(words.length - 1, (lastHit === -1 ? firstHit : lastHit) + padAfter);
  // Клэмп по сегменту (если транскрипт сегментирован).
  while (from < anchor && !inSeg(from)) from++;
  while (to > anchor && !inSeg(to)) to--;

  const isOther = (w: IndexedWord) =>
    otherRanges.some((r) => w.end > r.start && w.start < r.end);

  const out: ContextWord[] = [];
  for (let i = from; i <= to; i++) {
    const w = words[i];
    const inMute = w.end > start && w.start < end;
    // Чужие заглушки — выкидываем целиком (кроме своих inMute-слов, их не бывает
    // в otherRanges по построению).
    if (!inMute && isOther(w)) continue;
    out.push({ ...w, inMute });
  }
  return out;
}

/**
 * Временнóе окно для прослушивания мьюта: `before` слов до и `after` слов после
 * заглушки, в пределах сегмента. Возвращает локальные секунды (совпадают с
 * координатами транскрипта). Так «плей» даёт мат в контексте, а не голый обрубок.
 */
export function playWindow(
  index: WordIndex,
  start: number,
  end: number,
  before = 1,
  after = 2
): { start: number; end: number } {
  const { words } = index;
  if (!words.length) return { start, end };
  let firstHit = -1;
  let lastHit = -1;
  for (let i = 0; i < words.length; i++) {
    const w = words[i];
    if (w.end > start && w.start < end) {
      if (firstHit === -1) firstHit = i;
      lastHit = i;
    }
    if (w.start >= end) break;
  }
  if (firstHit === -1) return { start, end };
  const seg = words[firstHit].seg;
  const inSeg = (i: number) => seg < 0 || words[i].seg === seg;

  let a = firstHit;
  for (let n = before; n > 0 && a - 1 >= 0 && inSeg(a - 1); n--) a--;
  let b = lastHit;
  for (let n = after; n > 0 && b + 1 < words.length && inSeg(b + 1); n--) b++;

  return {
    start: Math.min(start, words[a].start),
    end: Math.max(end, words[b].end),
  };
}

/**
 * Снап значения времени к ближайшей границе слова, если она в пределах
 * `tolerance` секунд. Иначе возвращает исходное `t`. Границы = start и end
 * каждого слова. Так подрезка мьюта «прилипает» к речи.
 */
export function snapToWordBoundary(
  index: WordIndex,
  t: number,
  tolerance = 0.15
): number {
  const { words } = index;
  let best = t;
  let bestDist = tolerance;
  // Линейный проход дёшев (слов обычно тысячи, вызывается на drag-commit, не в кадре).
  for (const w of words) {
    for (const b of [w.start, w.end]) {
      const d = Math.abs(b - t);
      if (d < bestDist) {
        bestDist = d;
        best = b;
      }
    }
    if (w.start > t + tolerance) break; // дальше только больше — words отсортированы
  }
  return best;
}

/** Расширить мьют до полной оболочки перекрытых слов (кнопка «выделить слово целиком»). */
export function expandToWords(
  index: WordIndex,
  start: number,
  end: number
): { start: number; end: number } | null {
  const hit = wordsInRange(index, start, end);
  if (!hit.length) return null;
  return {
    start: Math.min(start, hit[0].start),
    end: Math.max(end, hit[hit.length - 1].end),
  };
}
