import { useEffect, useMemo, useRef, useState } from "react";
import { getWaveformRange, type WaveformData } from "../api";
import {
  baseTile,
  bucketsInView,
  coverFraction,
  covers,
  rangeTile,
  tileMaxAbs,
  type WaveTile,
} from "../components/timeline/waveTile";

/**
 * useWaveformTile — подгрузка крошки волны под текущее окно зума (как Audacity).
 *
 * ПРОБЛЕМА. Базовый слой — весь стрим в 8192 точки. Для 30-минутного VOD это
 * ~4.5 точки/сек; окно 5 сек = ~23 точки на ~1200 px → линейная интерполяция
 * даёт гладкий «блоб» вместо волны. Никакой цвет этого не лечит: не хватает
 * ДАННЫХ, а не оформления.
 *
 * РЕШЕНИЕ. При зуме дозапрашиваем /waveform/range ровно на видимое окно (+запас
 * на пан) и рисуем из него. Базовый слой остаётся фолбэком, поэтому волна на
 * экране есть ВСЕГДА — подгрузка её только уточняет.
 *
 * ПОЧЕМУ НЕ ЛАГАЕТ (это и просил пользователь):
 * - дебаунс: во время самого жеста запросов нет, только после паузы;
 * - PAD-запас: мелкий пан внутри уже загруженного тайла не триггерит фетч;
 * - AbortController: устаревший запрос отменяется, ответы не накапливаются;
 * - тайл живёт в ref + версии-счётчике: пока фетч в полёте, рисуется предыдущий
 *   тайл, никакого мигания и пустых кадров;
 * - при резком пане тайл НЕ выбрасывается, пока покрывает большую часть окна:
 *   деталь остаётся, а грубая база подкладывается только в непокрытый край
 *   (см. PARTIAL_MIN_COVER и skipStart/skipEnd);
 * - фетч НЕ трогает view/плейхед, поэтому зум/пан остаются 60 fps.
 */

/** Запас по краям окна (доля span) — даёт пану «пробег» без нового фетча. */
const PAD = 0.35;
/** Пауза после последнего изменения окна, мс. Меньше — дёргает бэкенд зря. */
const DEBOUNCE_MS = 120;
/** Целевая плотность: бакетов на CSS-пиксель канваса. */
const BUCKETS_PER_PX = 2;
/** Ниже этой плотности в окне тайл считаем грубым → фетч. */
const MIN_BUCKETS_PER_PX = 0.9;
/** Выше — тайл сильно мельче нужного (мы зумились ОТ него) → фетч подешевле. */
const MAX_BUCKETS_PER_PX = 8;
const RES_MIN = 256;
const RES_MAX = 8192;
/**
 * Минимальная доля окна, при которой деталь всё ещё лучше базы.
 *
 * Раньше тайл детали использовался только при ПОЛНОМ покрытии окна, и любой
 * резкий пан за край запаса мгновенно откатывал всю волну к грубой базе — это
 * и выглядело как «оно меняется на старую версию». Теперь при частичном
 * покрытии деталь остаётся, а канвас подкладывает базу только в непокрытый
 * край (см. skipStart/skipEnd в drawWaveTile).
 *
 * Порог нужен для зума ОТ волны: там деталь превращается в узкий островок в
 * центре широкого окна, и «остров крупной крошки в грубом поле» выглядит хуже,
 * чем ровная база. Пан даёт покрытие ~0.7–0.95, зум-аут — сильно меньше.
 */
const PARTIAL_MIN_COVER = 0.6;

export interface UseWaveformTileArgs {
  /** Путь к медиа для /waveform/range. null — фетч выключен. */
  mediaPath: string | null;
  /** Полный waveform (базовый слой). null — ещё не загружен. */
  wave: WaveformData | null;
  /** Текущее окно зума главного канваса (audio-сек). */
  view: { start: number; end: number };
  /** Ширина канваса в CSS-px — определяет запрашиваемое разрешение. */
  canvasWidth: number;
}

export interface UseWaveformTileResult {
  /** Базовый тайл (весь стрим) — для мини-карты. null пока нет wave. */
  base: WaveTile | null;
  /** Лучший доступный тайл для главного канваса (деталь или база). */
  tile: WaveTile | null;
  /** Глобальная нормировка амплитуды (по базовому слою). */
  norm: number;
  /** Идёт ли догрузка детали (для необязательного индикатора). */
  loading: boolean;
}

export function useWaveformTile({
  mediaPath,
  wave,
  view,
  canvasWidth,
}: UseWaveformTileArgs): UseWaveformTileResult {
  const base = useMemo(() => (wave ? baseTile(wave) : null), [wave]);
  // Нормировка — по базовому слою, один раз. По тайлу детали нормировать нельзя:
  // амплитуда «задышит» на каждой подгрузке (см. waveTile.ts).
  const norm = useMemo(() => (base ? tileMaxAbs(base) : 1), [base]);

  // Тайл детали держим в ref, а перерисовку триггерим счётчиком: сам Float32Array
  // в state положил бы большой объект в реконсиляцию React на каждую подгрузку.
  const detailRef = useRef<WaveTile | null>(null);
  const [version, setVersion] = useState(0);
  const [loading, setLoading] = useState(false);

  // Сброс детали при смене источника — иначе на новом видео покажется старая волна.
  useEffect(() => {
    detailRef.current = null;
    setVersion((v) => v + 1);
  }, [mediaPath, wave?.duration_s]);

  const vStart = view.start;
  const vEnd = view.end;

  useEffect(() => {
    if (!mediaPath || !base) return;
    const dur = base.end;
    const span = Math.max(vEnd - vStart, 0.0001);
    const wCss = Math.max(canvasWidth, 1);

    // Базовый слой уже достаточно плотный (обычно на полном отзуме) — деталь не
    // нужна, роняем её, чтобы не держать память и не рисовать лишний тайл.
    if (bucketsInView(base, vStart, vEnd) >= wCss * MIN_BUCKETS_PER_PX) {
      if (detailRef.current) {
        detailRef.current = null;
        setVersion((v) => v + 1);
      }
      return;
    }

    // Нужен ли новый фетч? Текущая деталь годится, если покрывает окно и её
    // плотность в этом окне в разумном коридоре. `maxed` — тайл уже максимально
    // подробный из возможных: даже если бакетов мало, глубже не станет, и
    // повторный фетч дал бы вечный цикл на максимальном зуме.
    const cur = detailRef.current;
    if (cur && covers(cur, vStart, vEnd)) {
      const bpp = bucketsInView(cur, vStart, vEnd) / wCss;
      if (bpp <= MAX_BUCKETS_PER_PX && (bpp >= MIN_BUCKETS_PER_PX || cur.maxed)) return;
    }

    // Окно с запасом, зажатое в границы стрима.
    const padded = span * PAD;
    const rStart = Math.max(0, vStart - padded);
    const rEnd = Math.min(dur, vEnd + padded);
    if (rEnd - rStart <= 0) return;
    const res = Math.round(
      Math.max(RES_MIN, Math.min(wCss * BUCKETS_PER_PX * ((rEnd - rStart) / span), RES_MAX))
    );

    const ac = new AbortController();
    const timer = setTimeout(() => {
      setLoading(true);
      getWaveformRange(mediaPath, rStart, rEnd, res, ac.signal)
        .then((r) => {
          if (ac.signal.aborted) return;
          detailRef.current = rangeTile(r, res);
          setVersion((v) => v + 1);
        })
        .catch(() => {
          // Отмена или сетевая ошибка — молча остаёмся на прежнем тайле. Волна
          // на экране уже есть, ругаться пользователю не за что.
        })
        .finally(() => {
          if (!ac.signal.aborted) setLoading(false);
        });
    }, DEBOUNCE_MS);

    return () => {
      clearTimeout(timer);
      ac.abort();
    };
  }, [mediaPath, base, vStart, vEnd, canvasWidth]);

  // version в зависимостях — намеренно: он и есть сигнал «деталь обновилась».
  const tile = useMemo(() => {
    const d = detailRef.current;
    if (!d) return base;
    if (covers(d, vStart, vEnd)) return d;
    // Частичное покрытие: деталь всё равно лучше базы на своей части окна,
    // непокрытый край закроет подложка. Ниже порога — ровная база (см.
    // PARTIAL_MIN_COVER).
    if (base && coverFraction(d, vStart, vEnd) < PARTIAL_MIN_COVER) return base;
    return d;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [base, vStart, vEnd, version]);

  return { base, tile, norm, loading };
}
