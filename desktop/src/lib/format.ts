/**
 * Форматирование таймкодов и длительностей.
 *
 * Обычно нам нужны либо миллисекунды (mutes/cuts в decisions.json хранят
 * `start_ms`/`end_ms`), либо секунды (highlights, длительность аудио). Разные
 * форматы для разных мест — под таймлайном хочется `H:MM:SS`, в редакторе
 * региона — `HH:MM:SS.mmm`.
 */

export function fmtMs(ms: number): string {
  if (!Number.isFinite(ms) || ms < 0) return "—";
  const total = Math.floor(ms / 100); // десятые доли секунды
  const tenths = total % 10;
  const totalSec = Math.floor(total / 10);
  const s = totalSec % 60;
  const m = Math.floor(totalSec / 60) % 60;
  const h = Math.floor(totalSec / 3600);
  const pad = (n: number) => n.toString().padStart(2, "0");
  return h > 0
    ? `${h}:${pad(m)}:${pad(s)}.${tenths}`
    : `${m}:${pad(s)}.${tenths}`;
}

export function fmtS(s: number): string {
  return fmtMs(s * 1000);
}

/** Полная точность для inputs таймкода в редакторе региона. */
export function fmtMsExact(ms: number): string {
  if (!Number.isFinite(ms) || ms < 0) return "00:00:00.000";
  const msPart = Math.floor(ms) % 1000;
  const totalSec = Math.floor(ms / 1000);
  const s = totalSec % 60;
  const m = Math.floor(totalSec / 60) % 60;
  const h = Math.floor(totalSec / 3600);
  const pad = (n: number, w = 2) => n.toString().padStart(w, "0");
  return `${pad(h)}:${pad(m)}:${pad(s)}.${pad(msPart, 3)}`;
}

/** Обратное: HH:MM:SS.mmm → ms; вернёт null если формат сломан. */
export function parseTimecode(str: string): number | null {
  const m = str.trim().match(/^(?:(\d+):)?(\d{1,2}):(\d{1,2})(?:\.(\d{1,3}))?$/);
  if (!m) return null;
  const h = Number(m[1] ?? 0);
  const mm = Number(m[2]);
  const ss = Number(m[3]);
  const ms = Number((m[4] ?? "0").padEnd(3, "0"));
  if (mm > 59 || ss > 59) return null;
  return ((h * 3600 + mm * 60 + ss) * 1000) + ms;
}

/**
 * Человекочитаемая длительность из миллисекунд по-русски: «2ч 5м» / «1м 20с» /
 * «45с». Единый формат для ETA обработки (JobScreen) и оценки диапазона
 * (NewJobScreen) — раньше в каждом экране была своя чуть-разная копия.
 */
export function fmtDurationMs(ms: number): string {
  if (!Number.isFinite(ms) || ms < 0) return "—";
  const totalSec = Math.round(ms / 1000);
  const h = Math.floor(totalSec / 3600);
  const m = Math.floor((totalSec % 3600) / 60);
  const s = totalSec % 60;
  if (h > 0) return `${h}ч ${m}м`;
  if (m > 0) return `${m}м ${s}с`;
  return `${s}с`;
}

/** "1h 24m", "24m", "42s" — для карточек проектов. */
export function fmtDurationHuman(seconds: number): string {
  if (!Number.isFinite(seconds) || seconds < 0) return "—";
  const total = Math.floor(seconds);
  const h = Math.floor(total / 3600);
  const m = Math.floor((total % 3600) / 60);
  const s = total % 60;
  // Показываем только значимые единицы: секунды опускаем если ==0, минуты
  // если ==0 при наличии часов. «4m 32s», «4m», «1h», «1h 5m», «1h 5m 3s», «45s».
  if (h > 0) {
    const parts = [`${h}h`];
    if (m > 0) parts.push(`${m}m`);
    if (s > 0) parts.push(`${s}s`);
    return parts.join(" ");
  }
  if (m > 0) return s > 0 ? `${m}m ${s}s` : `${m}m`;
  return `${s}s`;
}

/** ISO-строка → "5 минут назад" / "вчера" / "3 июля". Простая локальная реализация без i18n-либы. */
export function fmtRelative(iso: string): string {
  const then = new Date(iso).getTime();
  return fmtRelativeMs(then, iso);
}

/** Number-версия для API-полей *_ms (unix ms). */
export function fmtRelativeMs(ms: number, fallback = "—"): string {
  if (!Number.isFinite(ms)) return fallback;
  const diffMs = Date.now() - ms;
  const sec = Math.floor(diffMs / 1000);
  if (sec < 45) return "только что";
  const min = Math.floor(sec / 60);
  const hr = Math.floor(min / 60);
  // Пока не прошло полного часа — показываем минуты (иначе 45–59 мин давали
  // «0 ч назад»). Минимум 1 мин, чтобы не мигало «0 мин назад».
  if (hr < 1) return `${Math.max(1, min)} мин назад`;
  if (hr < 22) return `${hr} ч назад`;
  const day = Math.floor(hr / 24);
  if (day === 1) return "вчера";
  if (day < 7) return `${day} дн назад`;
  return new Date(ms).toLocaleDateString("ru-RU", {
    day: "numeric",
    month: "short",
    year: day > 300 ? "numeric" : undefined,
  });
}
