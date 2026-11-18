import type { WaveformData, WaveformRange } from "../../api";

/**
 * waveTile — единая модель данных волны + общая отрисовка «как в Audacity».
 *
 * ЗАЧЕМ ТАЙЛЫ. Базовый слой (весь стрим в 8192 точки) на зуме вырождается: окно
 * 5 сек 30-минутного VOD = ~23 точки на ~1200 px, линейная интерполяция даёт
 * гладкий «блоб». Audacity/Reaper решают это подгрузкой крошки ИМЕННО видимого
 * диапазона. Здесь и базовый слой, и слой детали — один тип `WaveTile`
 * (диапазон + бакеты), поэтому рисовалка одна и та же, а выбор «какой тайл
 * лучше» — отдельное дешёвое решение (см. pickTile).
 *
 * ФОРМА. Контур рисуем по настоящим min/max (асимметричный, «рваный»), а НЕ по
 * ±max(|.|): зеркало выглядит гладким. Внутри контура может рисоваться ядро RMS
 * (энергия) — сейчас оно ОТКЛЮЧЕНО, см. SHOW_RMS_CORE.
 *
 * СЛОИ. База и деталь рисуются в один канвас: сначала база с `skipStart/skipEnd`
 * (только там, где детали нет), потом деталь. Поэтому резкий пан больше не
 * «откатывает» всю волну к грубой базе — грубым остаётся только тот край, куда
 * данные ещё не доехали.
 *
 * НОРМИРОВКА. Всегда по ГЛОБАЛЬНОМУ максимуму стрима (`norm`), а не по максимуму
 * тайла: иначе при подгрузке нового окна амплитуда «дышала» бы — волна прыгает
 * в высоту на каждый зум. Это load-bearing, не оптимизация.
 */

export interface WaveTile {
  /** Фактически покрытый диапазон, сек (audio-время). */
  start: number;
  end: number;
  /** Нижняя граница контура по бакетам, ≤ 0. */
  min: Float32Array;
  /** Верхняя граница контура по бакетам, ≥ 0. */
  max: Float32Array;
  /** Энергия по бакетам, ≥ 0. null — старый кэш без rms (рисуем один контур). */
  rms: Float32Array | null;
  /**
   * true — бэкенд отдал МЕНЬШЕ бакетов, чем просили: на этом диапазоне детальнее
   * уже не будет (упёрлись в потолок sample rate). Хук обязан это уважать, иначе
   * получится вечный цикл «мало бакетов → фетч → снова мало» на максимальном
   * зуме. Защита структурная, а не «с запасом по порогу».
   */
  maxed?: boolean;
}

/** Глобальный максимум |амплитуды| — стабильная нормировка для всех тайлов. */
export function tileMaxAbs(tile: WaveTile): number {
  let m = 0.0001;
  const { min, max } = tile;
  for (let i = 0; i < max.length; i++) {
    const hi = max[i];
    if (hi > m) m = hi;
    const lo = -min[i];
    if (lo > m) m = lo;
  }
  return m;
}

/**
 * Базовый тайл из полного waveform-ответа. Если бэкенд отдал старый кэш без
 * min/max — падаем на симметричный ±peaks (та же волна, что была до Фазы 2).
 */
export function baseTile(wave: WaveformData): WaveTile {
  const n = wave.peaks.length;
  const hasMinMax =
    !!wave.min && !!wave.max && wave.min.length === n && wave.max.length === n;
  const min = new Float32Array(n);
  const max = new Float32Array(n);
  if (hasMinMax) {
    for (let i = 0; i < n; i++) {
      min[i] = wave.min![i];
      max[i] = wave.max![i];
    }
  } else {
    for (let i = 0; i < n; i++) {
      const v = Math.abs(wave.peaks[i]);
      min[i] = -v;
      max[i] = v;
    }
  }
  let rms: Float32Array | null = null;
  if (wave.rms && wave.rms.length === n) {
    rms = new Float32Array(n);
    for (let i = 0; i < n; i++) rms[i] = Math.abs(wave.rms[i]);
  }
  return { start: 0, end: Math.max(wave.duration_s, 0.001), min, max, rms };
}

/**
 * Тайл детали из /waveform/range. Границы берём ФАКТИЧЕСКИЕ (см. WaveformRange).
 *
 * `requestedRes` — сколько бакетов просили. Если пришло меньше, бэкенд упёрся в
 * свой потолок; помечаем тайл `maxed`, чтобы хук не пытался догрузиться глубже.
 */
export function rangeTile(r: WaveformRange, requestedRes?: number): WaveTile {
  const n = r.max.length;
  const min = new Float32Array(n);
  const max = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    min[i] = r.min[i];
    max[i] = r.max[i];
  }
  let rms: Float32Array | null = null;
  if (r.rms && r.rms.length === n) {
    rms = new Float32Array(n);
    for (let i = 0; i < n; i++) rms[i] = Math.abs(r.rms[i]);
  }
  return {
    start: r.start_s,
    end: r.end_s,
    min,
    max,
    rms,
    maxed: requestedRes != null && n < requestedRes,
  };
}

/** Бакетов тайла, приходящихся на окно [vs, ve] — метрика «достаточно ли крошки». */
export function bucketsInView(tile: WaveTile, vs: number, ve: number): number {
  const tileSpan = tile.end - tile.start;
  if (tileSpan <= 0) return 0;
  const span = Math.max(ve - vs, 0);
  return tile.max.length * (span / tileSpan);
}

/** Покрывает ли тайл окно целиком (с микро-допуском на float). */
export function covers(tile: WaveTile, vs: number, ve: number): boolean {
  return vs >= tile.start - 1e-6 && ve <= tile.end + 1e-6;
}

/** Какая доля окна [vs, ve] покрыта тайлом: 1 — целиком, 0 — не пересекаются. */
export function coverFraction(tile: WaveTile, vs: number, ve: number): number {
  const span = ve - vs;
  if (span <= 0) return 0;
  const from = Math.max(tile.start, vs);
  const to = Math.min(tile.end, ve);
  return Math.max(0, to - from) / span;
}

/**
 * Показывать ли ядро RMS (второй, более светлый слой внутри контура).
 *
 * Выключено по решению пользователя: на тёмном фоне «ядро» читается как белая
 * заливка внутри волны, а не как энергия («зачем нужен внутри белый цвет»).
 * В Audacity двухтоновость работает потому, что там тёмная волна на светлом
 * фоне; при инверсии внутренний слой неизбежно выглядит светлее контура.
 *
 * Данные rms продолжают приходить и лежать в тайле — вернуть двухтоновость
 * можно этим одним флагом, ничего больше менять не нужно.
 */
export const SHOW_RMS_CORE = false;

export interface DrawWaveOpts {
  /** Размер в CSS-пикселях (трансформа DPR уже применена к ctx). */
  w: number;
  h: number;
  /** Окно, которое рисуем (audio-сек). */
  viewStart: number;
  viewEnd: number;
  /** Глобальная нормировка амплитуды (см. tileMaxAbs базового тайла). */
  norm: number;
  /** Вертикальный запас сверху/снизу, CSS-px. */
  padPx: number;
  contourColor: string;
  /** Цвет ядра RMS. undefined — ядро не рисуем (см. SHOW_RMS_CORE). */
  coreColor?: string;
  /** Минимальная полутолщина контура (тишина = линия, а не разрыв), CSS-px. */
  minHalfPx?: number;
  /**
   * Не рисовать столбики, попадающие в этот диапазон (audio-сек). Нужно для
   * подложки: база рисуется ТОЛЬКО там, где нет тайла детали.
   *
   * Почему не «нарисовать базу целиком, а деталь поверх»: бакет базы
   * агрегирует экстремумы по более широкому интервалу, поэтому её огибающая
   * всегда ≥ детальной и торчала бы вокруг детали призрачным ореолом.
   */
  skipStart?: number;
  skipEnd?: number;
}

/**
 * Нарисовать волну тайла в окне. Один проход по столбикам, ДВЕ заливки Path2D:
 * тусклый контур min/max, поверх — яркое ядро RMS.
 *
 * Gapless: один столбик на CSS-пиксель + перехлёст rectW, поэтому соседние
 * прямоугольники в одном Path2D сливаются в сплошную заливку (без «частокола»
 * и без швов внутри — швы появляются только при отдельных fill на бар).
 */
export function drawWaveTile(
  ctx: CanvasRenderingContext2D,
  tile: WaveTile,
  opts: DrawWaveOpts
): void {
  const { w, h, viewStart, viewEnd, norm, padPx, minHalfPx = 0.5 } = opts;
  const n = tile.max.length;
  if (!n || w <= 0 || h <= 0) return;
  const tileSpan = tile.end - tile.start;
  if (tileSpan <= 0) return;
  const viewSpan = viewEnd - viewStart;
  if (viewSpan <= 0) return;

  const cy = h / 2;
  const half = Math.max((h - padPx * 2) / 2, 1);
  const inv = 1 / Math.max(norm, 0.0001);
  const bars = Math.max(1, Math.floor(w));
  const colW = w / bars;
  const rectW = colW + 1; // перехлёст → сплошная заливка
  // Время → индекс бакета тайла.
  const tToIdx = (t: number) => ((t - tile.start) / tileSpan) * n;

  const contour = new Path2D();
  const rmsArr = tile.rms;
  const core = rmsArr && opts.coreColor ? new Path2D() : null;
  const skipStart = opts.skipStart;
  const skipEnd = opts.skipEnd;
  const hasSkip = skipStart != null && skipEnd != null && skipEnd > skipStart;

  for (let i = 0; i < bars; i++) {
    const t0 = viewStart + (i / bars) * viewSpan;
    const t1 = viewStart + ((i + 1) / bars) * viewSpan;
    // Столбик отдан другому слою (деталь) — пропускаем. Решаем по середине
    // столбика: на стыке ровно один слой рисует его, без щели и без наложения.
    if (hasSkip) {
      const mid = (t0 + t1) / 2;
      if (mid >= skipStart! && mid <= skipEnd!) continue;
    }
    const f0 = tToIdx(t0);
    const f1 = tToIdx(t1);
    // Столбик целиком вне тайла — пропускаем (бывает в кадре между сменой
    // тайлов при быстром пане; лучше пустота, чем растянутый мусор).
    if (f1 <= 0 || f0 >= n) continue;
    const a = Math.floor(f0);
    const b = Math.ceil(f1);
    let lo: number;
    let hi: number;
    let rv = 0;
    if (b - a <= 1) {
      // < 1 бакета на столбик (глубокий зум) — интерполируем между соседями,
      // чтобы контур ехал плавно, а не ступенькой.
      const i0 = Math.max(0, Math.min(a, n - 1));
      const i1 = Math.max(0, Math.min(a + 1, n - 1));
      const f = f0 - a;
      lo = tile.min[i0] + (tile.min[i1] - tile.min[i0]) * f;
      hi = tile.max[i0] + (tile.max[i1] - tile.max[i0]) * f;
      if (rmsArr) rv = rmsArr[i0] + (rmsArr[i1] - rmsArr[i0]) * f;
    } else {
      // > 1 бакета — агрегируем окно: контур = экстремумы (пик НЕ теряется),
      // энергия = sqrt(mean(rms²)) (арифметическое среднее занижало бы её).
      const from = Math.max(0, a);
      const to = Math.min(b, n);
      lo = 0;
      hi = 0;
      let sq = 0;
      let cnt = 0;
      for (let j = from; j < to; j++) {
        const mn = tile.min[j];
        const mx = tile.max[j];
        if (mn < lo) lo = mn;
        if (mx > hi) hi = mx;
        if (rmsArr) sq += rmsArr[j] * rmsArr[j];
        cnt++;
      }
      if (rmsArr && cnt > 0) rv = Math.sqrt(sq / cnt);
    }

    const x = i * colW;
    // Контур: y растёт вниз, поэтому max → выше оси, min (≤0) → ниже.
    let yTop = cy - hi * inv * half;
    let yBot = cy - lo * inv * half;
    if (yBot - yTop < minHalfPx * 2) {
      // Тишина: рисуем тонкую линию по центру, иначе волна «рвётся».
      yTop = cy - minHalfPx;
      yBot = cy + minHalfPx;
    }
    contour.rect(x, yTop, rectW, yBot - yTop);

    if (core) {
      const rh = rv * inv * half;
      if (rh > 0.35) core.rect(x, cy - rh, rectW, rh * 2);
    }
  }

  ctx.fillStyle = opts.contourColor;
  ctx.fill(contour);
  if (core && opts.coreColor) {
    ctx.fillStyle = opts.coreColor;
    ctx.fill(core);
  }
}
