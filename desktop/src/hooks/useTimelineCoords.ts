import { useCallback, useRef } from "react";

/**
 * useTimelineCoords — конвертация между ЛОКАЛЬНЫМ и АУДИО пространствами времени.
 *
 * Правки мьютов живут в локальных секундах (start/end — от range_in, совпадает
 * с транскриптом и извлечённым cache-audio). Волна/блоки/плейбек живут в аудио-
 * пространстве: для cache-audio оно == локальному, для legacy source сдвинуто на
 * offset (audioTime = localTime + offset).
 *
 * `configure` вызывается один раз при загрузке проекта, когда известно, играем
 * ли мы извлечённый диапазон (localCoords=true) или исходный стрим со сдвигом.
 * `offsetRef` открыт наружу: его читает сохранение (пересчёт stream_start/end).
 */
export interface TimelineCoords {
  toAudio: (localT: number) => number;
  toLocal: (audioT: number) => number;
  offsetRef: React.MutableRefObject<number>;
  configure: (opts: { localCoords: boolean; offset: number }) => void;
}

export function useTimelineCoords(): TimelineCoords {
  const localCoordsRef = useRef(true); // true → audio == локальное пространство
  const offsetRef = useRef(0); // source-space: audioTime = localTime + offset

  const toAudio = useCallback(
    (localT: number) => (localCoordsRef.current ? localT : localT + offsetRef.current),
    []
  );
  const toLocal = useCallback(
    (audioT: number) => (localCoordsRef.current ? audioT : audioT - offsetRef.current),
    []
  );
  const configure = useCallback((opts: { localCoords: boolean; offset: number }) => {
    localCoordsRef.current = opts.localCoords;
    offsetRef.current = opts.offset;
  }, []);

  return { toAudio, toLocal, offsetRef, configure };
}
