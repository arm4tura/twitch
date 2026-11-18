"""Финальный рендер видео с вырезанными сегментами.

Принимает ГОТОВЫЙ список сегментов на удаление (start, end) — ему всё равно,
откуда они: тишина (silence.py), мат (decisions.py) или ручная правка. Вырезает
их через ffmpeg: каждый «оставленный» кусок подаётся отдельным seeked-инпутом
(-ss/-t) и склеивается concat-фильтром в один проход, кодируя аппаратно, если
железо позволяет (VideoToolbox на Mac, NVENC/QSV/AMF на Win/Linux).

Разделение обязанностей (см. заметку об архитектуре «на будущее»):
  - detect (silence.py / decisions.py) → список сегментов
  - render (здесь) → чистая функция «видео + список → видео»
  - прогресс репортится через callback (parse ffmpeg stderr time=), как в
    pipeline.py — CLI подсовывает логгер, server — Progress.emit для WS.
"""

from __future__ import annotations

import logging
import platform
import re
import subprocess
from functools import lru_cache
from pathlib import Path
from typing import Callable, Optional

from .ffmpeg_tools import get_ffmpeg_path, probe_media_duration

logger = logging.getLogger(__name__)

# Прогресс-callback: (percent, message). None → тишина.
RenderProgressFn = Callable[[float, str], None]

# ffmpeg пишет прогресс в stderr: frame=... time=00:01:23.45 bitrate=...
_TIME_RE = re.compile(r"time=(?P<h>\d{2}):(?P<m>\d{2}):(?P<s>\d{2})[.,](?P<ms>\d{1,3})")

# Приоритет аппаратных H.264-энкодеров по платформам. libx264 — CPU-фолбэк,
# работает везде.
_ENCODER_PRIORITY = ["h264_nvenc", "h264_qsv", "h264_amf"]


@lru_cache(maxsize=1)
def get_best_encoder() -> str:
    """Определяет лучший доступный H.264-энкодер ОДИН раз (кэшируется).

    macOS → всегда h264_videotoolbox (Apple Silicon/Intel — аппаратный энкодер).
    Иначе пробуем nvenc→qsv→amf по факту наличия в `ffmpeg -encoders`, потом libx264.

    Возвращает имя энкодера для `-c:v`. Не проверяет, что энкодер реально
    инициализируется на этом железе (например nvenc без NVIDIA GPU) — только
    наличие в сборке ffmpeg. Реальный фолбэк на libx264 при ошибке кодирования —
    ответственность render_video (retry).
    """
    if platform.system() == "Darwin":
        return "h264_videotoolbox"

    try:
        completed = subprocess.run(
            [get_ffmpeg_path(), "-hide_banner", "-encoders"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        available = completed.stdout
    except Exception as exc:  # noqa: BLE001
        logger.warning("Не смог перечислить энкодеры ffmpeg (%s), фолбэк libx264", exc)
        return "libx264"

    for enc in _ENCODER_PRIORITY:
        if enc in available:
            logger.info("Выбран аппаратный энкодер: %s", enc)
            return enc

    logger.info("Аппаратный энкодер не найден, используем CPU libx264")
    return "libx264"


# Пресет «скорость↔качество». fast — жертвуем качеством ради времени,
# max — наоборот, balanced — золотая середина (дефолт). Регулируем через
# per-encoder флаги, т.к. аппаратные (VideoToolbox/NVENC) и CPU (libx264)
# понимают РАЗНЫЕ ключи скорости.
QualityLiteral = str  # "fast" | "balanced" | "max"


def _encoder_args(enc: str, quality: str) -> list[str]:
    """Флаги `-c:v ...` под конкретный энкодер и пресет качества.

    Возвращает список аргументов ПОСЛЕ `-c:v <enc>` (сам `-c:v enc` ставит
    вызывающий). Для неизвестных энкодеров — пустой список (дефолты ffmpeg).
    """
    q = quality if quality in ("fast", "balanced", "max") else "balanced"

    if enc == "h264_videotoolbox":
        # Аппаратный энкодер Apple: скорость почти не регулируется пресетом,
        # но realtime-флаг и уровень качества (-q:v, шкала 0..100) дают эффект.
        # fast → realtime + ниже качество; max → выше качество, без realtime.
        if q == "fast":
            return ["-realtime", "1", "-q:v", "45"]
        if q == "max":
            return ["-q:v", "75"]
        return ["-q:v", "60"]

    if enc == "h264_nvenc":
        # NVENC пресеты p1(fastest)..p7(slowest). -tune hq + -rc vbr + -cq.
        preset = {"fast": "p2", "balanced": "p4", "max": "p7"}[q]
        cq = {"fast": "28", "balanced": "23", "max": "19"}[q]
        return ["-preset", preset, "-tune", "hq", "-rc", "vbr", "-cq", cq]

    if enc == "h264_qsv":
        preset = {"fast": "veryfast", "balanced": "medium", "max": "slower"}[q]
        gq = {"fast": "28", "balanced": "23", "max": "19"}[q]
        return ["-preset", preset, "-global_quality", gq]

    if enc == "h264_amf":
        # AMF: quality-пресет balanced/speed/quality + rc cqp.
        amf_q = {"fast": "speed", "balanced": "balanced", "max": "quality"}[q]
        return ["-quality", amf_q, "-rc", "cqp", "-qp_i", "22", "-qp_p", "22"]

    if enc == "libx264":
        # CPU-фолбэк: preset регулирует скорость↔сжатие, crf — качество,
        # -threads 0 = все ядра (прямой ответ на «оптимизировать рендер»).
        preset = {"fast": "veryfast", "balanced": "medium", "max": "slow"}[q]
        crf = {"fast": "26", "balanced": "21", "max": "18"}[q]
        return ["-preset", preset, "-crf", crf, "-threads", "0"]

    return []


def _clamp_keeps_to_range(
    keeps: list[tuple[float, float]],
    range_start: Optional[float],
    range_end: Optional[float],
) -> list[tuple[float, float]]:
    """Пересекает keep-интервалы с окном [range_start, range_end].

    Нужно, когда проект обработан с range_in/range_out: в итог должен попасть
    ТОЛЬКО этот диапазон, а не весь исходник. mute/silence внутри диапазона
    уже вырезаны (они в keeps), здесь мы лишь отсекаем всё вне окна.
    None-граница = «без ограничения» с этой стороны.
    """
    lo = range_start if range_start is not None else float("-inf")
    hi = range_end if range_end is not None else float("inf")
    if lo == float("-inf") and hi == float("inf"):
        return keeps
    out: list[tuple[float, float]] = []
    for s, e in keeps:
        cs = max(s, lo)
        ce = min(e, hi)
        if ce - cs > 1e-3:
            out.append((cs, ce))
    return out


def invert_segments(
    remove: list[tuple[float, float]],
    total_duration: float,
) -> list[tuple[float, float]]:
    """Инвертирует список «на удаление» в список «оставить».

    Пример: remove=[(10, 20), (30, 40)], total=60
             → keep=[(0, 10), (20, 30), (40, 60)]

    Сегменты сортируются и мёржатся перекрытия перед инверсией — на входе может
    быть что угодно. Возвращает только непустые keep-интервалы.
    """
    if not remove:
        return [(0.0, total_duration)] if total_duration > 0 else []

    # Сортируем и склеиваем перекрытия, чтобы инверсия была корректной.
    sorted_rm = sorted(remove, key=lambda x: x[0])
    merged: list[tuple[float, float]] = [sorted_rm[0]]
    for s, e in sorted_rm[1:]:
        ps, pe = merged[-1]
        if s <= pe:
            merged[-1] = (ps, max(pe, e))
        else:
            merged.append((s, e))

    keeps: list[tuple[float, float]] = []
    cursor = 0.0
    for s, e in merged:
        s = max(0.0, s)
        e = min(total_duration, e)
        if s > cursor:
            keeps.append((cursor, s))
        cursor = max(cursor, e)
    if cursor < total_duration:
        keeps.append((cursor, total_duration))

    # Отсекаем вырожденные (нулевой/отрицательной длины) интервалы.
    return [(s, e) for s, e in keeps if e - s > 1e-3]


def _project_mutes_to_output(
    mutes: list[tuple[float, float]],
    keeps: list[tuple[float, float]],
) -> list[tuple[float, float]]:
    """Спроецировать диапазоны-«заглушки» (координаты ИСХОДНИКА) на выходной таймлайн.

    В отличие от `segments_to_remove`, заглушки НЕ удаляют кадры — их аудио гасим
    в ноль, а видео остаётся. Но в итоговом файле время сжато вырезами (keeps —
    это то, что реально склеивается), поэтому мат [ms,me] в координатах исходника
    надо пересчитать в позицию внутри склеенного выхода, где t начинается с 0.

    Для каждого keep-сегмента (в порядке склейки) берём пересечение с мьютом и
    отображаем в выходное время: base + (overlap_start - keep_start). Мьют,
    целиком попавший в вырезанный участок, проекции не даёт (его там уже нет);
    мьют, пересекающий рез, даёт два выходных куска — это корректно. keeps уже
    обрезаны диапазоном (см. _clamp_keeps_to_range), поэтому проекция сама
    учитывает range_in/range_out — выход начинается с первого keep'а.
    """
    out: list[tuple[float, float]] = []
    base = 0.0
    for ks, ke in keeps:
        for ms, me in mutes:
            os_ = max(ms, ks)
            oe = min(me, ke)
            if oe - os_ > 1e-3:
                out.append((base + (os_ - ks), base + (oe - ks)))
        base += ke - ks
    out.sort(key=lambda x: x[0])
    return out


def _volume_mute_filter(mute_out: list[tuple[float, float]]) -> str:
    """ffmpeg audio-фильтр: гасит громкость в ноль на выходных диапазонах mute_out.

    enable-выражение — сумма `between(t,a,b)`; ненулевое значение = фильтр активен
    (volume=0), иначе аудио проходит без изменений. Одинарные кавычки защищают
    запятые внутри between() при разборе filtergraph (аргумент уходит одним argv,
    без шелла, — их интерпретирует сам ffmpeg).
    """
    betweens = "+".join(f"between(t,{a:.3f},{b:.3f})" for a, b in mute_out)
    return f"volume=enable='{betweens}':volume=0"


def _build_concat_inputs(
    input_path: Path,
    keeps: list[tuple[float, float]],
    mute_out_ranges: Optional[list[tuple[float, float]]] = None,
) -> tuple[list[str], str]:
    """Строит seeked-инпуты и concat-фильтр для склейки keep-интервалов.

    Каждый keep-кусок подаётся ОТДЕЛЬНЫМ инпутом с `-ss {start} -t {dur}`, т.е.
    ffmpeg декодирует только оставляемые куски (быстрый keyframe-seek + догон до
    точки), а не весь исходник, как делал старый select/aselect-фильтр. Куски
    склеиваются `concat`-фильтром в один непрерывный ре-энкод — без рассинхрона
    A/V на стыках (в отличие от concat со stream-copy, где AAC-кадры не бьются
    по границам реза).

    Возвращает (input_args, filter_complex):
      - input_args — плоский список `-ss S -t D -i PATH` на каждый keep;
      - filter_complex — `[0:v][0:a]...[N-1:v][N-1:a]concat=n=N:v=1:a=1[v][a]`.

    Пустой keeps невозможен — render_video это ловит раньше.

    Замечание: длина команды растёт с числом сегментов (N инпутов). Для реальных
    объёмов (десятки–сотни) это в пределах ARG_MAX (~1 МБ на macOS).
    """
    input_args: list[str] = []
    labels = ""
    src = str(input_path)
    for i, (s, e) in enumerate(keeps):
        input_args += ["-ss", f"{s:.3f}", "-t", f"{e - s:.3f}", "-i", src]
        labels += f"[{i}:v][{i}:a]"
    filter_complex = f"{labels}concat=n={len(keeps)}:v=1:a=1[v][a]"
    if mute_out_ranges:
        # Маты не режем — гасим их аудио в ноль: concat отдаёт [a0], затем
        # volume=0 на выходных диапазонах матов → [a]. Видео [v] не трогаем.
        filter_complex = (
            f"{labels}concat=n={len(keeps)}:v=1:a=1[v][a0];"
            f"[a0]{_volume_mute_filter(mute_out_ranges)}[a]"
        )
    return input_args, filter_complex


def render_video(
    input_path: Path,
    output_path: Path,
    segments_to_remove: list[tuple[float, float]],
    *,
    segments_to_mute: Optional[list[tuple[float, float]]] = None,
    total_duration: Optional[float] = None,
    encoder: Optional[str] = None,
    audio_bitrate: str = "192k",
    quality: str = "balanced",
    range_start: Optional[float] = None,
    range_end: Optional[float] = None,
    progress: Optional[RenderProgressFn] = None,
    force: bool = False,
) -> Path:
    """Рендерит видео, вырезав `segments_to_remove`.

    Args:
        input_path: исходное видео
        output_path: куда писать результат
        segments_to_remove: [(start, end), ...] в секундах — что убрать
        segments_to_mute: [(start, end), ...] в секундах — заглушить аудио в ноль,
            КАДРЫ ОСТАВИТЬ (для матов: слово не слышно, но видео на месте). None/[]
            → ничего не глушим. Координаты — исходника, как у segments_to_remove.
        total_duration: длительность исходника (сек); если None — пробим ffmpeg'ом
        encoder: имя H.264-энкодера; если None — get_best_encoder()
        audio_bitrate: битрейт аудио
        quality: пресет скорость↔качество ("fast" | "balanced" | "max")
        range_start: если задан — обрезать вывод слева (абс. секунды исходника)
        range_end: если задан — обрезать вывод справа (абс. секунды исходника)
        progress: callback(percent 0..100, message)
        force: перезаписать output, если существует

    Returns:
        Путь к отрендеренному видео (output_path).

    Raises:
        FileNotFoundError: нет входного файла
        ValueError: удаляются все сегменты (нечего оставить)
        RuntimeError: ffmpeg упал
    """
    if not input_path.exists():
        raise FileNotFoundError(f"Input video not found: {input_path}")
    if output_path.exists() and not force:
        logger.info("Output уже существует, используем как есть: %s", output_path)
        return output_path

    emit = progress or (lambda p, m="": None)

    if total_duration is None:
        total_duration = probe_media_duration(input_path).to_seconds()

    keeps = invert_segments(segments_to_remove, total_duration)
    # Обрезка по диапазону обработки (range_in/range_out): в итог попадает
    # только окно, которое пользователь размечал в «Правке», а не весь исходник.
    keeps = _clamp_keeps_to_range(keeps, range_start, range_end)
    if not keeps:
        raise ValueError(
            "Nothing to keep — segments_to_remove covers the entire video"
        )

    kept_duration = sum(e - s for s, e in keeps)
    enc = encoder or get_best_encoder()
    enc_args = _encoder_args(enc, quality)
    # Маты (segments_to_mute) не режем — проецируем на выходной таймлайн и гасим
    # их аудио в ноль поверх склейки keeps.
    mute_out = _project_mutes_to_output(segments_to_mute or [], keeps)
    input_args, filter_complex = _build_concat_inputs(input_path, keeps, mute_out)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    emit(0.0, f"render start: {len(keeps)} segments, encoder={enc}, quality={quality}")

    ffmpeg = get_ffmpeg_path()
    cmd = [
        ffmpeg,
        "-y",
        *input_args,
        "-filter_complex",
        filter_complex,
        "-map",
        "[v]",
        "-map",
        "[a]",
        "-c:v",
        enc,
        *enc_args,
        "-c:a",
        "aac",
        "-b:a",
        audio_bitrate,
        str(output_path),
    ]

    logger.info("Рендер: %s → %s (encoder=%s)", input_path.name, output_path.name, enc)
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    # Стримим stderr построчно, парсим time= для прогресса относительно
    # длительности РЕЗУЛЬТАТА (kept_duration), а не исходника.
    assert proc.stdout is not None
    last_lines: list[str] = []
    for line in proc.stdout:
        last_lines.append(line)
        if len(last_lines) > 20:
            last_lines.pop(0)
        m = _TIME_RE.search(line)
        if m and kept_duration > 0:
            secs = (
                int(m.group("h")) * 3600
                + int(m.group("m")) * 60
                + int(m.group("s"))
                + int(m.group("ms").ljust(3, "0")) / 1000.0
            )
            # У самого хвоста time= замирает у kept_duration, пока ffmpeg
            # флашит мукс — показываем «финализация…», а не застывшие цифры.
            if secs >= kept_duration - 0.5:
                emit(99.0, "финализация…")
            else:
                pct = min(99.0, secs / kept_duration * 100.0)
                emit(pct, f"encoding {secs:.1f}/{kept_duration:.1f}s")

    ret = proc.wait()
    if ret != 0:
        tail = "".join(last_lines).strip()
        raise RuntimeError(f"ffmpeg render failed (code {ret}):\n{tail}")

    emit(100.0, f"done: {output_path.name}")
    logger.info("Рендер завершён: %s", output_path)
    return output_path


def _safe_slug(text: str, max_len: int = 48) -> str:
    """Заголовок highlight'а → безопасное имя файла (кириллица сохраняется).

    Убираем только символы, ломающие файловые системы (/ \\ : * ? " < > |) и
    управляющие; схлопываем пробелы в подчёркивания. Пусто → 'clip'.
    """
    cleaned = re.sub(r'[\\/:*?"<>|\x00-\x1f]', "", text).strip()
    cleaned = re.sub(r"\s+", "_", cleaned)
    cleaned = cleaned[:max_len].strip("_.")
    return cleaned or "clip"


def render_clip(
    input_path: Path,
    output_path: Path,
    start: float,
    end: float,
    *,
    encoder: Optional[str] = None,
    audio_bitrate: str = "192k",
    quality: str = "balanced",
    progress: Optional[RenderProgressFn] = None,
    force: bool = False,
) -> Path:
    """Вырезать ОДИН непрерывный фрагмент [start, end] из видео в отдельный файл.

    Это «обратная» операция к render_video: там мы удаляем сегменты и склеиваем
    остаток, здесь — сохраняем ровно один кусок (highlight → клип для шортса).
    Один seeked-инпут (`-ss start -t dur`), ре-энкод тем же энкодером/пресетом,
    что и финальный рендер, чтобы клипы были консистентны с итоговым видео.

    Args:
        input_path: исходное видео (абсолютные секунды stream'а)
        output_path: путь к mp4 клипа
        start, end: границы фрагмента в АБСОЛЮТНЫХ секундах исходника
        force: перезаписать output, если существует

    Raises:
        FileNotFoundError: нет входного файла
        ValueError: end <= start
        RuntimeError: ffmpeg упал
    """
    if not input_path.exists():
        raise FileNotFoundError(f"Input video not found: {input_path}")
    dur = end - start
    if dur <= 1e-3:
        raise ValueError(f"clip end ({end}) must be > start ({start})")
    if output_path.exists() and not force:
        logger.info("Клип уже существует, используем как есть: %s", output_path)
        return output_path

    emit = progress or (lambda p, m="": None)
    enc = encoder or get_best_encoder()
    enc_args = _encoder_args(enc, quality)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    ffmpeg = get_ffmpeg_path()
    # ВАЖНО про порядок ключей (баг «клип длиной в час, чёрный экран до момента»):
    #   -ss ДО -i — быстрый keyframe-seek (декодируем не весь файл).
    #   -t   ПОСЛЕ -i — output-опция: ограничивает ДЛИТЕЛЬНОСТЬ вывода ровно
    #        `dur` секундами, отсчитывая от точки seek'а. Если поставить -t до
    #        -i (как было), ffmpeg с некоторыми муксерами/энкодерами (в т.ч.
    #        h264_videotoolbox на macOS) сохраняет ИСХОДНЫЕ PTS: плеер видит
    #        клип, который «начинается» на 40-й минуте, показывает чёрный кадр
    #        от 0 до этой точки, и итоговая длительность = длине всего стрима.
    #   -avoid_negative_ts make_zero + -reset_timestamps 1 — сдвигают тайм-базу
    #        вывода к нулю, чтобы клип гарантированно стартовал с 00:00 на всех
    #        платформах (Mac VideoToolbox / Win NVENC/QSV/AMF / libx264).
    cmd = [
        ffmpeg,
        "-y",
        "-ss",
        f"{max(0.0, start):.3f}",
        "-i",
        str(input_path),
        "-t",
        f"{dur:.3f}",
        "-avoid_negative_ts",
        "make_zero",
        "-c:v",
        enc,
        *enc_args,
        "-c:a",
        "aac",
        "-b:a",
        audio_bitrate,
        str(output_path),
    ]

    emit(0.0, f"clip start: {dur:.1f}s, encoder={enc}")
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
    )
    assert proc.stdout is not None
    last_lines: list[str] = []
    for line in proc.stdout:
        last_lines.append(line)
        if len(last_lines) > 20:
            last_lines.pop(0)
        m = _TIME_RE.search(line)
        if m and dur > 0:
            secs = (
                int(m.group("h")) * 3600
                + int(m.group("m")) * 60
                + int(m.group("s"))
                + int(m.group("ms").ljust(3, "0")) / 1000.0
            )
            emit(min(99.0, secs / dur * 100.0), f"encoding {secs:.1f}/{dur:.1f}s")

    ret = proc.wait()
    if ret != 0:
        tail = "".join(last_lines).strip()
        raise RuntimeError(f"ffmpeg clip failed (code {ret}):\n{tail}")

    emit(100.0, f"done: {output_path.name}")
    return output_path
