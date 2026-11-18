import { useEffect, useRef, useState, type MutableRefObject } from "react";
import { Pause, Play, Film } from "lucide-react";
import { mediaUrl } from "../../api";
import { fmtMs } from "../../lib/format";
import type { WordIndex } from "../../lib/transcript";

/** Audio remains the playback master so cuts and mute previews keep their semantics. */
export function PreviewDeck({ source, audioRef, videoOffset, toLocal, toAudio, words, onSeek, onPlay, onStop }: {
  source: string | null;
  audioRef: MutableRefObject<HTMLAudioElement | null>;
  videoOffset: number;
  toLocal: (s: number) => number;
  toAudio: (s: number) => number;
  words: WordIndex;
  onSeek: (s: number) => void;
  onPlay: (s: number) => void;
  onStop: () => void;
}) {
  const video = useRef<HTMLVideoElement>(null);
  const [failed, setFailed] = useState(false);
  const [playing, setPlaying] = useState(false);
  const [time, setTime] = useState(0);
  useEffect(() => {
    const audio = audioRef.current;
    if (!audio) return;
    let frame = 0;
    const sync = () => {
      const el = video.current;
      if (!el || el.readyState < 1) return;
      const target = Math.max(0, audio.currentTime + videoOffset);
      if (Math.abs(el.currentTime - target) > .15) el.currentTime = target;
      el.playbackRate = audio.playbackRate;
    };
    const tick = () => { sync(); frame = requestAnimationFrame(tick); };
    const update = () => { setTime(audio.currentTime); sync(); };
    const play = () => {
      setPlaying(true); sync();
      void video.current?.play().catch(() => {});
      cancelAnimationFrame(frame); frame = requestAnimationFrame(tick);
    };
    const pause = () => { setPlaying(false); video.current?.pause(); cancelAnimationFrame(frame); update(); };
    const ready = () => { sync(); if (!audio.paused) void video.current?.play().catch(() => {}); };
    const el = video.current;
    el?.addEventListener("loadedmetadata", ready);
    audio.addEventListener("play", play);
    audio.addEventListener("pause", pause);
    audio.addEventListener("timeupdate", update);
    audio.addEventListener("seeking", update);
    audio.addEventListener("ratechange", sync);
    update(); if (!audio.paused) play();
    return () => {
      cancelAnimationFrame(frame); el?.pause();
      el?.removeEventListener("loadedmetadata", ready);
      audio.removeEventListener("play", play); audio.removeEventListener("pause", pause);
      audio.removeEventListener("timeupdate", update); audio.removeEventListener("seeking", update);
      audio.removeEventListener("ratechange", sync);
    };
  }, [audioRef, source, videoOffset]);
  const local = toLocal(time);
  // Keep long recordings bounded; the transcript follows the current playback position.
  let lo = 0, hi = words.words.length;
  while (lo < hi) { const mid = (lo + hi) >>> 1; if (words.words[mid].end < local) lo = mid + 1; else hi = mid; }
  const first = Math.max(0, Math.floor(lo / 36) * 36 - 12);
  const excerpt = words.words.slice(first, first + 84);
  return <section className="preview-deck" aria-label="Видео и транскрипт">
    <div className="editor-video-panel">
      <div className="editor-video">
        {source && !failed ? <video ref={video} src={mediaUrl(source)} muted playsInline preload="metadata" onError={() => setFailed(true)} /> :
          <div className="video-unavailable"><Film className="h-7 w-7" /><span>Видео недоступно · аудио и правки доступны</span></div>}
      </div>
      <div className="preview-transport">
        <button type="button" aria-label={playing ? "Пауза" : "Воспроизвести с правками"} onClick={() => playing ? onStop() : onPlay(audioRef.current?.currentTime ?? 0)}>{playing ? <Pause size={15} /> : <Play size={15} />}</button>
        <span>{fmtMs(time * 1000)}</span><span className="ml-auto text-subtle">Предпросмотр</span>
      </div>
    </div>
    <div className="editor-transcript">
      <div className="transcript-heading">Транскрипт <span>Нажмите слово для перехода</span></div>
      <div className="transcript-words">{excerpt.length ? excerpt.map((w, i) => <button key={first + i} type="button" aria-current={local >= w.start && local < w.end ? "true" : undefined} onClick={() => onSeek(toAudio(w.start))}>{w.word}{" "}</button>) : <p className="text-muted">В этом проекте нет пословного транскрипта.</p>}</div>
    </div>
  </section>;
}
