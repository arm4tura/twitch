import { useCallback, useEffect, useRef, useState } from "react";

/**
 * useTimelinePlayback — весь аудио-плейбек экрана правки: скрытый <audio>,
 * точная остановка на границе окна, реалтайм-глушение «как в экспорте» и
 * плавный плейхед через rAF.
 *
 * Работает В АУДИО-КООРДИНАТАХ (секунды исходного <audio>). Конвертацию
 * локальное↔аудио делает вызывающий (useTimelineCoords) — сюда всё приходит
 * уже в аудио-секундах. `getMuteRegions` возвращает заглушённые участки в
 * аудио-координатах: их читаем на старте превью, чтобы глушить в реальном
 * времени. `getSkipRegions` (опц.) — ВЫРЕЗАННЫЕ участки (accepted silence): во
 * время непрерывного превью (playFrom) плейхед перепрыгивает их, чтобы звучало
 * ровно как финальный рендер.
 *
 * Обработчики onTimeUpdate/onPause вешаются пропсами на сам <audio> в render —
 * см. audioProps. Раньше их вешали через addEventListener в useEffect, и
 * слушатель успевал привязаться до монтирования элемента и больше не
 * перепривязывался — стоп не срабатывал вовсе.
 */
export interface TimelinePlayback {
  audioRef: React.MutableRefObject<HTMLAudioElement | null>;
  currentAudioS: number;
  /**
   * Живое значение плейхеда (audio-сек), обновляется КАЖДЫЙ кадр rAF. В отличие
   * от `currentAudioS` (стейт, throttle ~15 Гц для текста), этот ref читают
   * самоанимирующиеся плейхеды таймлайна — так линия едет плавно 60 Гц, но React
   * не перерисовывает волну/блоки/мини-карту на каждом кадре.
   */
  currentAudioRef: React.MutableRefObject<number>;
  playingMuteId: string | null;
  setPlayingMuteId: React.Dispatch<React.SetStateAction<string | null>>;
  stopPlayback: () => void;
  /** Проиграть аудио-отрезок [aStart,aEnd] (audio-сек) с паддингом pad и опц. реалтайм-глушением. */
  playAudioRange: (aStart: number, aEnd: number, pad?: number, applyMutes?: boolean) => void;
  /**
   * Играть НЕПРЕРЫВНО от aStart до естественного конца дорожки (без фикс-окна
   * стопа) — для hold-to-play на пробел. Останавливать вызовом stopPlayback.
   */
  playFrom: (aStart: number, applyMutes?: boolean) => void;
  seekAudio: (audioT: number) => void;
  audioProps: {
    onTimeUpdate: () => void;
    onPause: () => void;
    onEnded: () => void;
  };
}

export function useTimelinePlayback(
  getMuteRegions: () => Array<[number, number]>,
  getSkipRegions?: () => Array<[number, number]>
): TimelinePlayback {
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const playUntilRef = useRef<number | null>(null); // audio-секунда, на которой стоп
  const stopTimerRef = useRef<number | null>(null); // setTimeout-страховка стопа
  // Реалтайм-глушение при превью «как в экспорте»: во время playMute бегает rAF
  // и глушит <audio> ровно над заглушёнными участками. Скраб/«прослушать
  // выделение» — наоборот, играют оригинал, поэтому applyMutesRef там false.
  const rafRef = useRef<number | null>(null);
  const applyMutesRef = useRef(false);
  const muteRegionsRef = useRef<Array<[number, number]>>([]);
  // Пропуск вырезанных участков (accepted silence) при непрерывном превью: во
  // время playFrom rAF перепрыгивает currentTime через эти окна, чтобы звучало
  // «как в финальном рендере» — вырезанное не слышно вовсе (в отличие от mute,
  // который лишь глушит). skipTargetRef хранит цель уже выданного seek, чтобы не
  // перезапускать его на каждом кадре, пока он не завершился.
  const applySkipsRef = useRef(false);
  const skipRegionsRef = useRef<Array<[number, number]>>([]);
  const skipTargetRef = useRef<number | null>(null);
  // Живой геттер вырезанных участков в ref — чтобы seekAudio/playFrom брали
  // актуальный список (getSkipRegions пересоздаётся при правках) без пересборки
  // колбэков и лишних ре-рендеров.
  const getSkipRegionsRef = useRef(getSkipRegions);
  getSkipRegionsRef.current = getSkipRegions;

  const [currentAudioS, setCurrentAudioSRaw] = useState(0);
  const [playingMuteId, setPlayingMuteId] = useState<string | null>(null);

  // Плейхед: `currentAudioRef` — живое значение (каждый кадр rAF, читают
  // самоанимирующиеся плейхеды), `currentAudioS` — throttle-стейт для текста
  // (таймкод, слово в транскрипте). Обновлять стейт 60 Гц заставляло React
  // перерисовывать весь экран правки на каждом кадре — главный источник лагов
  // на длинных стримах. Throttle до ~15 Гц: глазу таймкода хватает.
  const currentAudioRef = useRef(0);
  const lastStateTsRef = useRef(0);
  const setCurrentAudioS = useCallback((t: number) => {
    currentAudioRef.current = t;
    const now = performance.now();
    if (now - lastStateTsRef.current >= 66) {
      lastStateTsRef.current = now;
      setCurrentAudioSRaw(t);
    }
  }, []);
  // Форсированный коммит стейта (стоп/seek): текст должен встать ровно на месте.
  const flushCurrentAudioS = useCallback((t: number) => {
    currentAudioRef.current = t;
    lastStateTsRef.current = performance.now();
    setCurrentAudioSRaw(t);
  }, []);

  // Единая точка остановки. Дёргается из двух источников: браузерного
  // `timeupdate` (редкий, ~4 раза/сек, может проскочить границу) и из точного
  // setTimeout-таймера, заведённого при старте. Любой, кто сработал первым,
  // глушит воспроизведение — второй становится no-op.
  const stopPlayback = useCallback(() => {
    const el = audioRef.current;
    if (stopTimerRef.current != null) {
      window.clearTimeout(stopTimerRef.current);
      stopTimerRef.current = null;
    }
    if (rafRef.current != null) {
      cancelAnimationFrame(rafRef.current);
      rafRef.current = null;
    }
    applyMutesRef.current = false;
    playUntilRef.current = null;
    // Снять пропуск вырезанных участков, чтобы повторный playFrom стартовал чисто.
    applySkipsRef.current = false;
    skipRegionsRef.current = [];
    skipTargetRef.current = null;
    if (el) {
      if (!el.paused) el.pause();
      el.muted = false; // снять реалтайм-глушение
      // Догнать throttled-стейт до финальной позиции: пока играли, текст
      // обновлялся ~15 Гц, на стопе он должен встать ровно.
      flushCurrentAudioS(el.currentTime);
    }
    setPlayingMuteId(null);
  }, [flushCurrentAudioS]);

  /**
   * rAF-скан во время превью «как в экспорте»: чаще, чем timeupdate (~4 Гц),
   * поэтому граница мьюта не «протекает». Глушит <audio>.muted, когда
   * currentTime попадает в заглушённый участок, и снимает глушение вне их.
   * Также двигает плейхед плавно и ловит стоп по playUntil.
   */
  const scanMutes = useCallback(() => {
    const el = audioRef.current;
    if (!el) return;
    let t = el.currentTime;
    // Пропуск вырезанных участков: если плейхед въехал в accepted-silence,
    // прыгаем на его конец — вырезанное не проигрывается. skipTargetRef гасит
    // повторный seek, пока предыдущий не «докрутился» (currentTime асинхронен).
    if (applySkipsRef.current) {
      if (skipTargetRef.current != null && t < skipTargetRef.current - 0.001) {
        // Seek ещё едет к цели — ждём, плейхед не двигаем в дырку.
        rafRef.current = requestAnimationFrame(scanMutes);
        return;
      }
      skipTargetRef.current = null;
      for (const [a, b] of skipRegionsRef.current) {
        if (t >= a && t < b - 0.001) {
          el.currentTime = b;
          skipTargetRef.current = b;
          t = b;
          break;
        }
      }
    }
    setCurrentAudioS(t);
    if (applyMutesRef.current) {
      let inMute = false;
      for (const [a, b] of muteRegionsRef.current) {
        if (t >= a && t < b) {
          inMute = true;
          break;
        }
      }
      if (el.muted !== inMute) el.muted = inMute;
    }
    if (playUntilRef.current != null && t >= playUntilRef.current) {
      stopPlayback();
      return;
    }
    rafRef.current = requestAnimationFrame(scanMutes);
  }, [stopPlayback]);

  const onAudioTime = useCallback(() => {
    const el = audioRef.current;
    if (!el) return;
    setCurrentAudioS(el.currentTime);
    if (playUntilRef.current != null && el.currentTime >= playUntilRef.current) {
      stopPlayback();
    }
  }, [stopPlayback]);

  const onAudioPause = useCallback(() => {
    // Ручная пауза (не наша граница) — снять подсветку играющей строки.
    if (playUntilRef.current == null) setPlayingMuteId(null);
  }, []);

  // Естественный конец дорожки при hold-to-play (playFrom): playUntil не
  // выставлен, окно-стоп не сработает — глушим rAF/подсветку явно.
  const onAudioEnded = useCallback(() => {
    stopPlayback();
  }, [stopPlayback]);

  // Снять таймер/rAF при размонтировании экрана.
  useEffect(
    () => () => {
      if (stopTimerRef.current != null) window.clearTimeout(stopTimerRef.current);
      if (rafRef.current != null) cancelAnimationFrame(rafRef.current);
    },
    []
  );

  /**
   * Проиграть аудио-отрезок [aStart, aEnd] (в audio-координатах).
   * `pad` — сколько добавить с каждого края (контекст). Для превью мьюта — 0.25с,
   * для «слышно край» (скраб) — почти ноль, чтобы получился короткий блип.
   * `applyMutes` — глушить ли заглушённые участки в реальном времени (превью
   * «как в экспорте»). Для скраба/поиска границ — false (нужен оригинал).
   */
  const playAudioRange = useCallback(
    (aStart: number, aEnd: number, pad = 0.25, applyMutes = false) => {
      const el = audioRef.current;
      if (!el) return;
      const from = Math.max(0, aStart - pad);
      const until = aEnd + pad;
      el.currentTime = from;
      playUntilRef.current = until;
      // Реалтайм-глушение: список заглушённых участков в audio-координатах.
      applyMutesRef.current = applyMutes;
      el.muted = false;
      muteRegionsRef.current = applyMutes ? getMuteRegions() : [];
      // Точная страховка: timeupdate/rAF могут не совпасть с концом окна —
      // дублируем стоп таймером на длину окна (playbackRate = 1).
      if (stopTimerRef.current != null) window.clearTimeout(stopTimerRef.current);
      const ms = Math.max(0, (until - from) * 1000);
      stopTimerRef.current = window.setTimeout(() => {
        stopTimerRef.current = null;
        stopPlayback();
      }, ms);
      // Плавный плейхед + точное глушение через rAF (частит, в отличие от timeupdate).
      if (rafRef.current != null) cancelAnimationFrame(rafRef.current);
      rafRef.current = requestAnimationFrame(scanMutes);
      void el.play().catch(() => {
        /* автоплей может быть заблокирован до первого жеста — игнор */
      });
    },
    [stopPlayback, scanMutes, getMuteRegions]
  );

  /**
   * Играть НЕПРЕРЫВНО от aStart до естественного конца дорожки — для
   * hold-to-play на пробел. В отличие от playAudioRange, НЕ ставит playUntil и
   * setTimeout-страховку: играем до конца <audio> или до ручного stopPlayback
   * (отпускание пробела). Реалтайм-глушение работает так же, как в превью
   * «как в экспорте», поэтому applyMutes по умолчанию true.
   */
  const playFrom = useCallback(
    (aStart: number, applyMutes = true) => {
      const el = audioRef.current;
      if (!el) return;
      el.currentTime = Math.max(0, aStart);
      playUntilRef.current = null; // нет границы окна — играем до конца дорожки
      applyMutesRef.current = applyMutes;
      el.muted = false;
      muteRegionsRef.current = applyMutes ? getMuteRegions() : [];
      // Пропуск вырезанных участков (accepted silence) — чтобы превью звучало
      // ровно как финальный рендер: вырезанное не проигрывается, а перепрыгивается.
      applySkipsRef.current = true;
      skipRegionsRef.current = getSkipRegionsRef.current?.() ?? [];
      skipTargetRef.current = null;
      // Фикс-таймер стопа не нужен: конец придёт по 'ended' или по stopPlayback
      // (отпускание пробела). Снимаем оставшийся от прошлого превью, если был.
      if (stopTimerRef.current != null) {
        window.clearTimeout(stopTimerRef.current);
        stopTimerRef.current = null;
      }
      if (rafRef.current != null) cancelAnimationFrame(rafRef.current);
      rafRef.current = requestAnimationFrame(scanMutes);
      void el.play().catch(() => {
        /* автоплей может быть заблокирован до первого жеста — игнор */
      });
    },
    [scanMutes, getMuteRegions]
  );

  const seekAudio = useCallback(
    (audioT: number) => {
      const el = audioRef.current;
      if (!el) return;
      if (stopTimerRef.current != null) {
        window.clearTimeout(stopTimerRef.current);
        stopTimerRef.current = null;
      }
      // Было ли активное проигрывание (rAF-цикл). Если да — клик-перемотка
      // ДОЛЖНА его сохранить: меняем позицию и продолжаем играть (и пропускать
      // вырезанное/глушить), а не глушим цикл (это и был «лаг линии» — rAF
      // убивался, линия ехала только редкими timeupdate).
      const wasPlaying = rafRef.current != null;
      if (rafRef.current != null) {
        cancelAnimationFrame(rafRef.current);
        rafRef.current = null;
      }
      // Ручная перемотка отменяет старую границу окна (от playAudioRange): после
      // клика продолжаем как непрерывное проигрывание до конца дорожки.
      playUntilRef.current = null;
      // Если играли — состояние проигрывания (пропуск вырезанного, глушение)
      // остаётся включённым; если стоп — глушим и пропуск, и глушение.
      if (!wasPlaying) {
        applyMutesRef.current = false;
        applySkipsRef.current = false;
        skipRegionsRef.current = [];
        skipTargetRef.current = null;
      }
      el.muted = false;
      el.currentTime = Math.max(0, audioT);
      // Seek в середину вырезанного куска (accepted silence) — приземлиться на его
      // конец, как если бы этот кусок уже был удалён в финальном видео. Берём
      // свежий список (getSkipRegionsRef), а не остатки прошлого playFrom.
      const skip = getSkipRegionsRef.current?.() ?? [];
      for (const [a, b] of skip) {
        if (audioT >= a && audioT < b) {
          el.currentTime = b;
          break;
        }
      }
      flushCurrentAudioS(el.currentTime);
      // Клик во время проигрывания — продолжить с новой позиции: перезапустить
      // rAF-цикл, чтобы линия ехала плавно и вырезанное/глушение работали.
      if (wasPlaying) {
        if (skipTargetRef.current == null) skipTargetRef.current = el.currentTime;
        rafRef.current = requestAnimationFrame(scanMutes);
      }
    },
    [flushCurrentAudioS, scanMutes]
  );

  return {
    audioRef,
    currentAudioS,
    currentAudioRef,
    playingMuteId,
    setPlayingMuteId,
    stopPlayback,
    playAudioRange,
    playFrom,
    seekAudio,
    audioProps: { onTimeUpdate: onAudioTime, onPause: onAudioPause, onEnded: onAudioEnded },
  };
}
