import { forwardRef, useCallback, useMemo } from "react";
import { Input, type InputProps } from "./Input";

/**
 * TimecodeInput — масковый ввод таймкода `HH:MM:SS` (опц. `.mmm`) с жёсткими
 * ограничениями:
 *
 *   • минуты и секунды — максимум 2 цифры и ≤ 59 (нельзя ввести «третью» цифру
 *     или значение вроде 75);
 *   • часы — до `maxHoursDigits` цифр (по умолчанию 2 → до 99ч);
 *   • миллисекунды (если включены) — до 3 цифр;
 *   • значение клампится в диапазон `[minMs, maxMs]`, если границы заданы.
 *
 * Раньше range_in/range_out были обычными <Input> без валидации — можно было
 * ввести «99:99:99» или время больше длительности видео. Этот компонент —
 * единая замена и для них, и для новых полей скачивания VOD (там maxMs =
 * длительность VOD, поэтому нельзя выйти за конец записи).
 *
 * Контракт: value/onChange работают со СТРОКОЙ таймкода (как хранит форма),
 * а не с миллисекундами — чтобы дропин-замена <Input> ничего не ломала выше.
 * onChange отдаёт уже нормализованную (промасканную и, при blur, клампнутую)
 * строку.
 */

export interface TimecodeInputProps
  extends Omit<InputProps, "value" | "onChange" | "type"> {
  /** Текущее значение — строка таймкода, напр. "01:23:45" или "01:23:45.500". */
  value: string;
  /** Колбэк с уже нормализованной строкой. */
  onChange: (next: string) => void;
  /** Показывать поле миллисекунд (.mmm). По умолчанию false — секундной точности хватает. */
  withMillis?: boolean;
  /** Верхняя граница в мс (напр. длительность VOD). Больше ввести/склампить нельзя. */
  maxMs?: number | null;
  /** Нижняя граница в мс. По умолчанию 0. */
  minMs?: number | null;
  /** Сколько цифр разрешаем в часах. 2 → 99ч. */
  maxHoursDigits?: number;
}

/**
 * Разбор строки таймкода в мс, СОГЛАСОВАННЫЙ с раскладкой маски (слева направо,
 * начиная с часов). Группы делятся по ':':
 *   3 группы → H:M:S
 *   2 группы → H:M  (секунды = 0) — юзер ещё не дошёл до секунд
 *   1 группа → H    (минуты/секунды = 0)
 * Возвращает null, если строка пустая/битая.
 */
function parseTimecodeGroups(
  value: string
): { h: string; m: string; s: string; ms: string } | null {
  if (!value) return null;
  const [intPart, msRaw = "0"] = value.split(".");
  const parts = intPart.split(":");
  if (parts.length > 3) return null;
  let h = "0";
  let m = "0";
  let s = "0";
  if (parts.length === 3) {
    [h, m, s] = parts;
  } else if (parts.length === 2) {
    [h, m] = parts;
  } else {
    [h] = parts;
  }
  return { h: h || "0", m: m || "0", s: s || "0", ms: msRaw || "0" };
}

/** Считает мс из групп цифр. Пустые группы = 0. Диапазон ≤59 обеспечивает маска. */
function groupsToMs(h: string, m: string, s: string, ms: string): number {
  const hh = Number(h || "0");
  const mm = Number(m || "0");
  const ss = Number(s || "0");
  const mmm = Number((ms || "0").padEnd(3, "0").slice(0, 3));
  return ((hh * 3600 + mm * 60 + ss) * 1000) + mmm;
}

function msToTimecode(ms: number, withMillis: boolean, maxHoursDigits: number): string {
  const safe = Math.max(0, Math.floor(ms));
  const msPart = safe % 1000;
  const totalSec = Math.floor(safe / 1000);
  const s = totalSec % 60;
  const m = Math.floor(totalSec / 60) % 60;
  const hCap = Math.pow(10, maxHoursDigits) - 1;
  const h = Math.min(hCap, Math.floor(totalSec / 3600));
  const pad = (n: number, w = 2) => n.toString().padStart(w, "0");
  const hms = `${pad(h)}:${pad(m)}:${pad(s)}`;
  return withMillis ? `${hms}.${pad(msPart, 3)}` : hms;
}

/**
 * Маскирование ввода. Берём сырую строку (что напечатал юзер), вытаскиваем все
 * цифры и раскладываем их СЛЕВА НАПРАВО в HH:MM:SS(.mmm): первые цифры — часы,
 * затем минуты, затем секунды. Двоеточия расставляются сами по мере заполнения
 * групп — юзеру не нужно печатать разделители, и не появляется «ведущих»
 * двоеточий вроде «:1:» (из-за которых курсор разъезжался и «1 1» давало «:1:11»).
 *
 * Раскладка групп:
 *   часы   — maxHoursDigits цифр (по умолчанию 2);
 *   минуты — 2 цифры, ≤ 59;
 *   секунды— 2 цифры, ≤ 59.
 * Лишние цифры сверх секунд отбрасываются.
 */
function maskTimecode(raw: string, withMillis: boolean, maxHoursDigits: number): string {
  // Отделяем миллисекундную часть (после точки), если разрешена.
  let intPart = raw;
  let msDigits = "";
  const hasDot = withMillis && raw.includes(".");
  if (withMillis) {
    const dot = raw.indexOf(".");
    if (dot >= 0) {
      intPart = raw.slice(0, dot);
      msDigits = raw.slice(dot + 1).replace(/\D/g, "").slice(0, 3);
    }
  }

  // Единый поток цифр из целой части — разделители/мусор выкидываем.
  const digits = intPart.replace(/\D/g, "");

  // Раскладываем слева направо: часы (maxHoursDigits) → минуты (2) → секунды (2).
  const h = digits.slice(0, maxHoursDigits);
  const m = digits.slice(maxHoursDigits, maxHoursDigits + 2);
  const s = digits.slice(maxHoursDigits + 2, maxHoursDigits + 4);

  // Клампим готовые 2-значные группы минут/секунд к ≤ 59 (напр. «75» → «59»).
  const clampGroup = (g: string) => {
    if (g.length < 2) return g;
    const n = Number(g);
    return n > 59 ? "59" : g;
  };
  const mm = clampGroup(m);
  const ss = clampGroup(s);

  // Собираем строку, добавляя двоеточие ТОЛЬКО перед непустой следующей группой.
  // Это не даёт «ведущих» двоеточий, пока юзер ещё печатает часы.
  let out = h;
  if (mm.length || (m.length === 0 && digits.length > maxHoursDigits)) {
    out += `:${mm}`;
  }
  if (ss.length || (s.length === 0 && digits.length > maxHoursDigits + 2)) {
    out += `:${ss}`;
  }

  if (withMillis && (msDigits || hasDot)) {
    out += `.${msDigits}`;
  }
  return out;
}

export const TimecodeInput = forwardRef<HTMLInputElement, TimecodeInputProps>(
  (
    {
      value,
      onChange,
      withMillis = false,
      maxMs = null,
      minMs = 0,
      maxHoursDigits = 2,
      onBlur,
      invalid,
      placeholder,
      ...rest
    },
    ref
  ) => {
    // Мс-значение текущей строки — для проверки границ и подсветки invalid.
    const currentMs = useMemo(() => {
      const g = parseTimecodeGroups(value);
      if (!g) return null;
      // Пока часы/минуты не введены совсем — значение неопределённо (не 0),
      // чтобы не подсвечивать пустое поле как out-of-range.
      if (value.replace(/\D/g, "") === "") return null;
      return groupsToMs(g.h, g.m, g.s, g.ms);
    }, [value]);

    const outOfRange = useMemo(() => {
      if (currentMs == null) return false;
      const lo = minMs ?? 0;
      if (currentMs < lo) return true;
      if (maxMs != null && currentMs > maxMs) return true;
      return false;
    }, [currentMs, minMs, maxMs]);

    const handleChange = useCallback(
      (e: React.ChangeEvent<HTMLInputElement>) => {
        // Маскируем на каждый keystroke — лишние цифры/переполнение отсекаются
        // ещё до попадания в состояние, поэтому «3 цифры в минуты» невозможны.
        const masked = maskTimecode(e.target.value, withMillis, maxHoursDigits);
        onChange(masked);
      },
      [onChange, withMillis, maxHoursDigits]
    );

    const handleBlur = useCallback(
      (e: React.FocusEvent<HTMLInputElement>) => {
        // На blur — доводим до канонического HH:MM:SS и клампим в [min,max].
        // Клампим именно тут (а не на keystroke), чтобы не мешать печатать:
        // ввод «4:3…» временно < min не должен переписываться на лету.
        const g = value ? parseTimecodeGroups(value) : null;
        if (g) {
          let totalMs = groupsToMs(g.h, g.m, g.s, g.ms);
          const lo = minMs ?? 0;
          if (totalMs < lo) totalMs = lo;
          if (maxMs != null && totalMs > maxMs) totalMs = maxMs;
          onChange(msToTimecode(totalMs, withMillis, maxHoursDigits));
        }
        onBlur?.(e);
      },
      [value, onChange, onBlur, withMillis, maxHoursDigits, minMs, maxMs]
    );

    return (
      <Input
        ref={ref}
        value={value}
        onChange={handleChange}
        onBlur={handleBlur}
        inputMode="numeric"
        placeholder={placeholder ?? (withMillis ? "00:00:00.000" : "00:00:00")}
        invalid={invalid || outOfRange}
        className="font-mono-tabular"
        {...rest}
      />
    );
  }
);
TimecodeInput.displayName = "TimecodeInput";

/** Экспорт хелперов — переиспользуются в карточке скачивания для проверки диапазона. */
export function timecodeToMs(value: string): number | null {
  const g = parseTimecodeGroups(value);
  if (!g) return null;
  if (value.replace(/\D/g, "") === "") return null;
  const n = [g.h, g.m, g.s].map(Number);
  if (n.some((x) => Number.isNaN(x))) return null;
  return groupsToMs(g.h, g.m, g.s, g.ms);
}

export { msToTimecode };
