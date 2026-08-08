/**
 * Парсинг и валидация ссылок на Twitch VOD.
 *
 * Пользователь может вставить что угодно: полный URL, URL с таймкодом `?t=`,
 * ссылку с трекинг-параметрами, или просто числовой ID. Наша задача — вытащить
 * чистый числовой video ID (или сказать, что ссылка не распознана), ДО того как
 * дёргать backend. Это даёт мгновенную inline-валидацию поля ввода.
 *
 * Поддерживаемые формы:
 *   https://www.twitch.tv/videos/123456789
 *   https://twitch.tv/videos/123456789?t=1h2m3s
 *   twitch.tv/videos/123456789
 *   https://m.twitch.tv/videos/123456789
 *   123456789            (голый ID)
 *
 * НЕ поддерживаем (осознанно, вернём null): клипы (/clip/...), каналы вживую
 * (twitch.tv/<channel>) — это не VOD и TwitchDownloaderCLI их так не качает.
 */

export interface ParsedVod {
  /** Числовой ID видео (строкой — ID длинные, Number терять точность не должен). */
  id: string;
  /** Стартовое смещение из `?t=` в секундах, если было. Для подсказки range_in. */
  startOffsetSec: number | null;
}

/** Парсит `?t=1h2m3s` / `?t=45m` / `?t=90s` / `?t=120` → секунды. */
function parseTParam(t: string | null): number | null {
  if (!t) return null;
  // Чисто числовой t — это секунды.
  if (/^\d+$/.test(t)) return Number(t);
  const m = t.match(/^(?:(\d+)h)?(?:(\d+)m)?(?:(\d+)s)?$/i);
  if (!m || (!m[1] && !m[2] && !m[3])) return null;
  const h = Number(m[1] ?? 0);
  const min = Number(m[2] ?? 0);
  const s = Number(m[3] ?? 0);
  return h * 3600 + min * 60 + s;
}

/**
 * Извлекает VOD-ID из произвольного пользовательского ввода. Возвращает null,
 * если это не распознаваемая ссылка на VOD.
 */
export function parseTwitchVodInput(raw: string): ParsedVod | null {
  const input = raw.trim();
  if (!input) return null;

  // Голый числовой ID.
  if (/^\d+$/.test(input)) {
    return { id: input, startOffsetSec: null };
  }

  // Пытаемся распарсить как URL. Добавляем протокол, если его нет, чтобы
  // конструктор URL не падал на "twitch.tv/videos/123".
  let url: URL;
  try {
    url = new URL(/^https?:\/\//i.test(input) ? input : `https://${input}`);
  } catch {
    return null;
  }

  // Хост должен быть твичевским (в т.ч. www./m./go.).
  if (!/(^|\.)twitch\.tv$/i.test(url.hostname)) return null;

  // Путь вида /videos/<id>.
  const m = url.pathname.match(/^\/videos\/(\d+)\/?$/);
  if (!m) return null;

  return {
    id: m[1],
    startOffsetSec: parseTParam(url.searchParams.get("t")),
  };
}

/** Каноническая ссылка на VOD по ID — для отображения/логов. */
export function twitchVodUrl(id: string): string {
  return `https://www.twitch.tv/videos/${id}`;
}
