"""Сравнительный пробник ASR: WhisperX vs GigaAM на НЕСКОЛЬКИХ кусках стрима.

ЗАЧЕМ
-----
Прежде чем переписывать backend под GigaAM, надо проверить на СВОИХ данных
две вещи, важные именно этому проекту:
  1. Кто лучше ловит мат (detect_profanity с твоим banwords.txt).
  2. У кого точнее пословные тайминги start/end (Vegas режет по границам слова).

Скрипт берёт несколько таймкодов стрима (--starts), гоняет каждый через ОБА
движка и выдаёт:
  * report.txt        — по каждому сегменту: тексты, слова с таймингами,
                        таблица мата бок-о-бок; в конце — СВОДНАЯ таблица.
  * transcript_*.json — полные транскрипты каждого движка.
  * profanity_*.json  — плоские списки мата с таймингами.
  * vegas_whisperx.cs / vegas_gigaam.cs — Vegas Pro скрипт ОТ КАЖДОГО движка
                        (mute по абсолютному времени в полном stream.mp4).
                        Открой оба в Vegas и лично сравни, чей mute точнее.

Vegas-скрипты строятся тем же прод-кодом (build_decisions + generate_vegas_script),
поэтому mute-логика (padding, склейка, cap) идентична реальному пайплайну.
Статусы форсятся в 'accepted', чтобы в .cs попал КАЖДЫЙ детект (для ручной проверки).

GigaAM подключён через официальный API (`transcribe(path, word_timestamps=True)`,
лимит 25 c) + silence-aware нарезка по тишине (ffmpeg silencedetect).

УСТАНОВКА GIGAAM (в тот же backend/.venv)
-----------------------------------------
    ВАЖНО: word-timestamps есть только в git-версии, НЕ в PyPI-релизе 0.1.0.
    Ставить с --no-deps, иначе gigaam откатит torch на CPU и сломает CUDA:

    backend\\.venv\\Scripts\\pip install --no-deps ^
        "git+https://github.com/salute-developers/GigaAM.git"

ЗАПУСК (три отрезка по 2 минуты)
--------------------------------
    backend\\.venv\\Scripts\\python backend\\examples\\compare_asr.py ^
        --input input\\stream.mp4 ^
        --starts 00:15:00,00:30:00,00:45:00 --duration 120 ^
        --gigaam-model ctc --device cuda
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

# --- Подключаем пакет проекта (backend/src) независимо от cwd -----------------
_HERE = Path(__file__).resolve()
_SRC = _HERE.parents[1] / "src"
if _SRC.exists() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from twitch_cut.config import DEFAULT_ASR_OPTIONS, PipelineConfig  # noqa: E402
from twitch_cut.decisions import build_decisions  # noqa: E402
from twitch_cut.ffmpeg_tools import get_ffmpeg_path  # noqa: E402
from twitch_cut.profanity import (  # noqa: E402
    ProfanityMatch,
    RussianNormalizer,
    detect_profanity,
    load_banwords,
)
from twitch_cut.timecode import TimeSpan, parse_timecode  # noqa: E402
from twitch_cut.vegas_export import generate_vegas_script  # noqa: E402


# =============================================================================
# ffmpeg-утилиты
# =============================================================================
def extract_working_wav(input_path: Path, start: TimeSpan, duration: TimeSpan, out_wav: Path) -> None:
    """Один WAV на оба движка — аудио идентично для честного сравнения."""
    ffmpeg = get_ffmpeg_path()
    cmd = [
        ffmpeg, "-y",
        "-ss", f"{start.to_seconds():.3f}",
        "-i", str(input_path),
        "-t", f"{duration.to_seconds():.3f}",
        "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le",
        str(out_wav),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0 or not out_wav.exists():
        raise RuntimeError(f"ffmpeg extraction failed:\n{proc.stderr[-2000:]}")


def _media_duration(media: Path) -> float:
    ffmpeg = get_ffmpeg_path()
    proc = subprocess.run([ffmpeg, "-hide_banner", "-i", str(media)], capture_output=True, text=True)
    m = re.search(r"Duration:\s*(\d+):(\d+):(\d+\.\d+)", proc.stderr)
    if not m:
        return 0.0
    return int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))


def probe_duration(wav: Path) -> float:
    try:
        import soundfile as sf
        return float(sf.info(str(wav)).duration)
    except Exception:  # noqa: BLE001
        return _media_duration(wav)


# =============================================================================
# WhisperX (прод-код transcribe_audio)
# =============================================================================
def _add_cudnn_to_dll_path() -> None:
    """Windows: CTranslate2 (движок WhisperX) ищет cudnn_ops_infer64_8.dll в PATH.

    nvidia-cudnn-cu12 кладёт DLL в site-packages/nvidia/cudnn/bin, но эта папка
    не в DLL-search → 'Could not locate cudnn_ops_infer64_8.dll' и жёсткий краш.
    Регистрируем каталоги cudnn/cublas до импорта whisperx.
    """
    if not hasattr(os, "add_dll_directory"):
        return
    for pkg in ("cudnn", "cublas"):
        for base in sys.path:
            cand = Path(base) / "nvidia" / pkg / "bin"
            if cand.is_dir():
                try:
                    os.add_dll_directory(str(cand))
                except OSError:
                    pass
                os.environ["PATH"] = str(cand) + os.pathsep + os.environ.get("PATH", "")
                break


def run_whisperx(wav: Path, workdir: Path, model_name: str, device: str, compute_type: str) -> dict[str, Any]:
    _add_cudnn_to_dll_path()
    from twitch_cut.transcription import transcribe_audio

    transcript, _key, _cache = transcribe_audio(
        audio_path=wav, workdir=workdir, model_name=model_name, language="ru",
        device=device, compute_type=compute_type, batch_size=16,
        vad_filter=True, vad_method="pyannote",
        asr_options=dict(DEFAULT_ASR_OPTIONS), force=False,
    )
    return transcript


# =============================================================================
# GigaAM: silence-aware чанки <= 25с + официальный word_timestamps
# =============================================================================
def _detect_silences(wav: Path, noise_db: int, min_sil: float) -> list[tuple[float, float]]:
    ffmpeg = get_ffmpeg_path()
    proc = subprocess.run(
        [ffmpeg, "-hide_banner", "-i", str(wav),
         "-af", f"silencedetect=noise={noise_db}dB:d={min_sil}", "-f", "null", "-"],
        capture_output=True, text=True,
    )
    starts = [float(x) for x in re.findall(r"silence_start:\s*([\d.]+)", proc.stderr)]
    ends = [float(x) for x in re.findall(r"silence_end:\s*([\d.]+)", proc.stderr)]
    return list(zip(starts, ends))


def _plan_chunks(duration: float, silences: list[tuple[float, float]],
                 max_len: float, min_len: float) -> list[tuple[float, float]]:
    cut_points = sorted((s + e) / 2 for s, e in silences)
    chunks: list[tuple[float, float]] = []
    start = 0.0
    while start < duration - 1e-3:
        target = start + max_len
        if target >= duration:
            chunks.append((start, duration))
            break
        cands = [c for c in cut_points if start + min_len < c <= target]
        cut = max(cands) if cands else target
        chunks.append((start, cut))
        start = cut
    return chunks


_GIGAAM_MODEL_CACHE: dict[str, Any] = {}  # грузим модель один раз на все сегменты


def _load_gigaam(model_name: str, device: str) -> Any:
    if model_name in _GIGAAM_MODEL_CACHE:
        return _GIGAAM_MODEL_CACHE[model_name]
    import gigaam
    from twitch_cut.transcription import apply_whisperx_patches
    apply_whisperx_patches()  # torch 2.6: weights_only=False, иначе unpickle падает
    try:
        model = gigaam.load_model(model_name, device=device)
    except TypeError:
        model = gigaam.load_model(model_name)
    _GIGAAM_MODEL_CACHE[model_name] = model
    return model


def run_gigaam(wav: Path, model_name: str, device: str,
               max_chunk: float, noise_db: int, min_sil: float) -> dict[str, Any]:
    model = _load_gigaam(model_name, device)
    duration = probe_duration(wav)
    silences = _detect_silences(wav, noise_db, min_sil)
    chunks = _plan_chunks(duration, silences, max_chunk, min_len=2.0)
    print(f"  GigaAM: {len(chunks)} чанк(ов) по тишине (<= {max_chunk:.0f}c)")

    ffmpeg = get_ffmpeg_path()
    segments: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory() as td:
        for i, (c_start, c_end) in enumerate(chunks):
            chunk_wav = Path(td) / f"chunk_{i:04d}.wav"
            subprocess.run(
                [ffmpeg, "-y", "-ss", f"{c_start:.3f}", "-i", str(wav),
                 "-t", f"{c_end - c_start:.3f}", "-ar", "16000", "-ac", "1",
                 "-c:a", "pcm_s16le", str(chunk_wav)],
                capture_output=True, text=True,
            )
            result = model.transcribe(str(chunk_wav), word_timestamps=True)
            words = []
            for w in getattr(result, "words", []) or []:
                words.append({
                    "word": w.text,
                    "start": float(w.start) + c_start,  # offset чанка -> локальное время сегмента
                    "end": float(w.end) + c_start,
                    "score": getattr(w, "confidence", None),
                })
            if words:
                segments.append({
                    "id": f"seg_{i:06d}",
                    "start": words[0]["start"],
                    "end": words[-1]["end"],
                    "words": words,
                })
    return {"segments": segments}


# =============================================================================
# Анализ
# =============================================================================
def transcript_text(t: dict[str, Any]) -> str:
    return " ".join(
        str(w.get("word") or w.get("text") or "").strip()
        for seg in t.get("segments", [])
        for w in seg.get("words", [])
    ).strip()


def pair_matches(mx: list[ProfanityMatch], mg: list[ProfanityMatch], tol_s: float = 1.5):
    """Пары одинаковых матов по banword и ближайшему старту.

    Возвращает список (whisperx|None, gigaam|None): None означает 'поймал только
    один движок'.
    """
    pairs: list[tuple[ProfanityMatch | None, ProfanityMatch | None]] = []
    used: set[int] = set()
    for x in mx:
        best = None
        bd = tol_s
        for j, g in enumerate(mg):
            if j in used or g.banword != x.banword:
                continue
            d = abs(g.local_start_ms - x.local_start_ms) / 1000
            if d < bd:
                bd, best = d, (j, g)
        if best:
            used.add(best[0])
            pairs.append((x, best[1]))
        else:
            pairs.append((x, None))
    for j, g in enumerate(mg):
        if j not in used:
            pairs.append((None, g))
    return pairs


# =============================================================================
# Vegas-экспорт (прод-код) — один .cs на движок, mute по всему stream.mp4
# =============================================================================
def build_vegas_for_engine(
    input_path: Path, per_segment_matches: list[tuple[TimeSpan, TimeSpan, list[ProfanityMatch]]],
    transcriber_id: str, device: str, out_cs: Path, out_json: Path,
) -> int:
    """Собираем decisions по каждому сегменту (реальный range_in -> абсолютный
    stream_start) и клеим mutes в один Vegas-скрипт. Статусы -> accepted, чтобы
    в .cs попал каждый детект для ручной проверки."""
    config = PipelineConfig(transcriber=transcriber_id, device=device)
    all_mutes: list[dict[str, Any]] = []
    for range_in, range_out, matches in per_segment_matches:
        if not matches:
            continue
        decisions = build_decisions(
            stream_path=input_path, original_path=input_path,
            range_in=range_in, range_out=range_out, matches=matches, config=config,
        )
        for mute in decisions.get("mutes", []):
            mute["status"] = "accepted"  # форсим, чтобы всё экспортировалось
            all_mutes.append(mute)

    all_mutes.sort(key=lambda m: float(m.get("stream_start", 0)))
    merged = {
        "schema_version": "1.1",
        "source": str(input_path),
        "original": str(input_path),
        "mutes": all_mutes,
        "cuts": [],
    }
    out_json.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")
    out_cs.write_text(generate_vegas_script(merged), encoding="utf-8")
    return len(all_mutes)


# =============================================================================
def main() -> int:
    ap = argparse.ArgumentParser(description="WhisperX vs GigaAM на нескольких кусках стрима")
    ap.add_argument("--input", required=True, type=Path, help="stream.mp4 или .wav")
    ap.add_argument("--starts", default=None,
                    help="таймкоды через запятую, напр. 00:15:00,00:30:00,00:45:00")
    ap.add_argument("--start", default="0", help="одиночный таймкод (если не задан --starts)")
    ap.add_argument("--duration", default="120", help="длительность каждого сегмента, сек")
    ap.add_argument("--banwords", type=Path, default=_HERE.parents[1] / "banwords.txt")
    ap.add_argument("--engines", default="both", choices=["both", "whisperx", "gigaam"])
    ap.add_argument("--whisperx-model", default="large-v3")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--compute-type", default="float16")
    ap.add_argument("--gigaam-model", default="ctc",
                    help="ctc | rnnt | v3_ctc | v3_e2e_ctc (для мата бери *_ctc без e2e)")
    ap.add_argument("--gigaam-max-chunk", type=float, default=22.0)
    ap.add_argument("--silence-db", type=int, default=-30)
    ap.add_argument("--silence-min", type=float, default=0.3)
    ap.add_argument("--workdir", type=Path, default=_HERE.parents[1].parent / "work" / "_asr_probe")
    args = ap.parse_args()

    starts = [parse_timecode(s.strip()) for s in args.starts.split(",")] if args.starts \
        else [parse_timecode(args.start)]
    duration = TimeSpan.from_seconds(float(args.duration))
    args.workdir.mkdir(parents=True, exist_ok=True)
    src_dur = _media_duration(args.input)

    print(f"Вход: {args.input}  (длина {src_dur/60:.1f} мин)")
    print(f"Сегментов: {len(starts)} по {args.duration}c   устройство: {args.device}")

    normalizer = RussianNormalizer()
    banwords = load_banwords(args.banwords, normalizer)
    print(f"Банвордов: {len(banwords)}  (из {args.banwords})")

    want = {"both": ("WhisperX", "GigaAM"), "whisperx": ("WhisperX",), "gigaam": ("GigaAM",)}[args.engines]

    # Накопители: по каждому движку -> список (range_in, range_out, matches) на сегмент
    per_engine_segments: dict[str, list[tuple[TimeSpan, TimeSpan, list[ProfanityMatch]]]] = {
        e: [] for e in want
    }
    # Для отчёта: детальные данные по сегментам
    seg_reports: list[dict[str, Any]] = []
    full_transcripts: dict[str, dict[str, Any]] = {e: {"segments": []} for e in want}
    full_profanity: dict[str, list[dict[str, Any]]] = {e: [] for e in want}

    for si, start in enumerate(starts):
        range_out = start + duration
        if range_out.ms > int(src_dur * 1000) + 500:
            print(f"\n[сегмент {si+1}] {start} — за пределами входа, пропускаю")
            continue
        print(f"\n===== Сегмент {si+1}/{len(starts)}: {start} (+{args.duration}c) =====")
        wav = args.workdir / f"probe_{si:02d}_{str(start).replace(':', '-')}.wav"
        extract_working_wav(args.input, start, duration, wav)
        if probe_duration(wav) < 0.5:
            print("  пустой WAV — пропускаю"); continue

        seg_entry: dict[str, Any] = {"idx": si, "start": start, "engines": {}}

        for eng in want:
            print(f"  -> {eng} ...")
            try:
                if eng == "WhisperX":
                    tr = run_whisperx(wav, args.workdir, args.whisperx_model, args.device, args.compute_type)
                    tid = "whisperx"
                else:
                    tr = run_gigaam(wav, args.gigaam_model, args.device,
                                    args.gigaam_max_chunk, args.silence_db, args.silence_min)
                    tid = "gigaam"
                # detect с РЕАЛЬНЫМ range_in -> stream-тайминги абсолютны (для Vegas)
                matches = detect_profanity(tr, banwords, start, normalizer)
                per_engine_segments[eng].append((start, range_out, matches))
                seg_entry["engines"][eng] = {"transcript": tr, "matches": matches, "tid": tid}
                # копим сквозные транскрипты/маты со сдвигом в абсолютное время
                for seg in tr.get("segments", []):
                    s2 = dict(seg)
                    s2["_stream_offset"] = start.to_seconds()
                    full_transcripts[eng]["segments"].append(s2)
                for m in matches:
                    full_profanity[eng].append({
                        "word": m.word, "banword": m.banword, "match_type": m.match_type,
                        "stream_start": round(m.stream_start_ms / 1000, 3),
                        "stream_end": round(m.stream_end_ms / 1000, 3),
                        "local_start": round(m.local_start_ms / 1000, 3),
                        "local_end": round(m.local_end_ms / 1000, 3),
                    })
                print(f"     мат: {len(matches)}")
            except ImportError:
                print("     SKIPPED GigaAM: pip install --no-deps "
                      "git+https://github.com/salute-developers/GigaAM.git")
            except TypeError as exc:
                if "word_timestamps" in str(exc):
                    print("     SKIPPED GigaAM: PyPI-релиз без word-timestamps -> ставь git-версию")
                else:
                    print(f"     SKIPPED {eng}: TypeError: {exc}")
            except Exception as exc:  # noqa: BLE001
                print(f"     SKIPPED {eng}: {type(exc).__name__}: {exc}")

        seg_reports.append(seg_entry)

    # ------------------------------------------------------------------ отчёт
    lines: list[str] = []

    def emit(s: str = "") -> None:
        print(s)
        lines.append(s)

    emit("\n" + "=" * 72)
    emit("ДЕТАЛИ ПО СЕГМЕНТАМ")
    for seg in seg_reports:
        emit(f"\n########## Сегмент {seg['idx']+1}: {seg['start']} ##########")
        for eng, data in seg["engines"].items():
            m = data["matches"]
            emit(f"\n--- {eng}: текст ---")
            emit(transcript_text(data["transcript"]))
            emit(f"\n--- {eng}: пойманный мат ({len(m)}) [локальное время сегмента] ---")
            if not m:
                emit("  (ничего не найдено)")
            for x in m:
                emit(f"  [{x.local_start_ms/1000:7.2f} - {x.local_end_ms/1000:7.2f}]  "
                     f"{x.word!r:18} banword={x.banword!r} type={x.match_type}")
        # пара X/G
        if "WhisperX" in seg["engines"] and "GigaAM" in seg["engines"]:
            mx = seg["engines"]["WhisperX"]["matches"]
            mg = seg["engines"]["GigaAM"]["matches"]
            emit(f"\n--- Сегмент {seg['idx']+1}: WhisperX vs GigaAM (длительность mute на слово) ---")
            emit(f"  {'слово':12} | {'WhX старт-конец (dur)':26} | {'GigaAM старт-конец (dur)':26} | коммент")
            for x, g in pair_matches(mx, mg):
                if x and g:
                    dx = (x.local_end_ms - x.local_start_ms) / 1000
                    dg = (g.local_end_ms - g.local_start_ms) / 1000
                    note = ""
                    if dx > 0.8 and dx > dg * 2.5:
                        note = f"WhX растянул до {dx:.1f}c (GigaAM {dg:.2f}c)"
                    emit(f"  {x.word[:12]:12} | {x.local_start_ms/1000:6.2f}-{x.local_end_ms/1000:6.2f} "
                         f"({dx:4.2f}c){' ':4} | {g.local_start_ms/1000:6.2f}-{g.local_end_ms/1000:6.2f} "
                         f"({dg:4.2f}c) | {note}")
                elif x:
                    emit(f"  {x.word[:12]:12} | {x.local_start_ms/1000:6.2f} только WhisperX | — | GigaAM пропустил")
                else:
                    emit(f"  {g.word[:12]:12} | — | {g.local_start_ms/1000:6.2f} только GigaAM | WhisperX пропустил")

    # ------------------------------------------------------- СВОДНАЯ таблица
    emit("\n" + "=" * 72)
    emit("СВОДКА ПО ВСЕМ СЕГМЕНТАМ")
    import statistics as st
    totals = {e: sum(len(m) for _, _, m in per_engine_segments[e]) for e in want}
    emit(f"\nПоймано матов всего:  " + "   ".join(f"{e}={totals[e]}" for e in want))

    if "WhisperX" in want and "GigaAM" in want:
        only_x = only_g = both = 0
        drifts: list[float] = []
        wx_over = 0
        wx_durs: list[float] = []
        gg_durs: list[float] = []
        for (ri, ro, mx), (_, _, mg) in zip(per_engine_segments["WhisperX"], per_engine_segments["GigaAM"]):
            for x, g in pair_matches(mx, mg):
                if x and g:
                    both += 1
                    drifts.append(abs(x.local_start_ms - g.local_start_ms) / 1000)
                    dx = (x.local_end_ms - x.local_start_ms) / 1000
                    dg = (g.local_end_ms - g.local_start_ms) / 1000
                    wx_durs.append(dx)
                    gg_durs.append(dg)
                    if dx > 0.8 and dx > dg * 2.5:
                        wx_over += 1
                elif x:
                    only_x += 1
                else:
                    only_g += 1
        emit(f"Совпало (оба поймали):        {both}")
        emit(f"Только WhisperX (GigaAM мимо): {only_x}")
        emit(f"Только GigaAM (WhisperX мимо): {only_g}")
        if drifts:
            emit(f"\nРасхождение СТАРТА мата (по совпавшим):")
            emit(f"  среднее {st.mean(drifts)*1000:5.0f} мс   медиана {st.median(drifts)*1000:5.0f} мс   "
                 f"макс {max(drifts)*1000:5.0f} мс")
        if wx_durs and gg_durs:
            emit(f"\nДлительность одного mute-слова:")
            emit(f"  WhisperX: средн {st.mean(wx_durs):.2f}c  макс {max(wx_durs):.2f}c")
            emit(f"  GigaAM:   средн {st.mean(gg_durs):.2f}c  макс {max(gg_durs):.2f}c")
            emit(f"  Случаев, где WhisperX неестественно растянул mate (>0.8c и >2.5x GigaAM): {wx_over}")

    # ------------------------------------------------------- запись артефактов
    saved: list[Path] = []
    report_path = args.workdir / "report.txt"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    saved.append(report_path)

    for eng in want:
        suffix = eng.lower()
        tp = args.workdir / f"transcript_{suffix}.json"
        tp.write_text(json.dumps(full_transcripts[eng], ensure_ascii=False, indent=2), encoding="utf-8")
        pp = args.workdir / f"profanity_{suffix}.json"
        pp.write_text(json.dumps(full_profanity[eng], ensure_ascii=False, indent=2), encoding="utf-8")
        saved.extend([tp, pp])

    # Vegas-скрипт от каждого движка
    emit("\n" + "=" * 72)
    emit("VEGAS-СКРИПТЫ (открой в Vegas Pro: Tools -> Scripting -> Run Script)")
    for eng in want:
        tid = "whisperx" if eng == "WhisperX" else "gigaam"
        cs = args.workdir / f"vegas_{eng.lower()}.cs"
        dj = args.workdir / f"decisions_{eng.lower()}.json"
        try:
            n = build_vegas_for_engine(
                args.input, per_engine_segments[eng], tid, args.device, cs, dj,
            )
            emit(f"  {eng}: {n} mute -> {cs}")
            saved.extend([cs, dj])
        except Exception as exc:  # noqa: BLE001
            emit(f"  {eng}: Vegas-экспорт не удался: {type(exc).__name__}: {exc}")

    report_path.write_text("\n".join(lines), encoding="utf-8")  # перезапись с Vegas-секцией

    print("\nФайлы:")
    for p in saved:
        print(f"  {p}")
    print("\nПроверка: открой в Vegas оба vegas_*.cs на полном stream.mp4 и сравни,")
    print("чей mute точнее садится на слово. Тайминги также видно в report.txt.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
