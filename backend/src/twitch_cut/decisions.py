from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from typing import Any

from .config import PipelineConfig
from .profanity import ProfanityMatch
from .timecode import TimeSpan


def _seconds(ms: int) -> float:
    return round(ms / 1000.0, 3)


def _group_by_segment(
    matches: list[ProfanityMatch],
    join_gap_ms: int = 600,
) -> list[list[ProfanityMatch]]:
    """Группирует подряд идущие матчи с одинаковым segment_id В ОДИН мьют,
    но только если они идут плотно — разрыв между концом предыдущего мата
    и началом следующего не больше join_gap_ms.

    Если в одном whisper-сегменте два мата, а между ними пауза с чистой
    речью ('блять ... убил школьницу ... блять'), они попадают в РАЗНЫЕ
    группы — иначе мьют first..last проглотил бы предложение между ними.

    Матчи без segment_id остаются каждый в своей группе."""
    groups: list[list[ProfanityMatch]] = []
    for match in matches:
        prev = groups[-1][-1] if groups else None
        same_segment = (
            prev is not None
            and match.segment_id is not None
            and prev.segment_id == match.segment_id
        )
        gap_ok = (
            prev is not None
            and match.local_start_ms - prev.local_end_ms <= join_gap_ms
        )
        if same_segment and gap_ok:
            groups[-1].append(match)
        else:
            groups.append([match])
    return groups


def _compute_mute_window(
    group: list[ProfanityMatch],
    config: PipelineConfig,
) -> tuple[int, int, int, int]:
    """Считает (local_start_ms, local_end_ms, stream_start_ms, stream_end_ms)
    для одного объединённого mute по группе матчей одного сегмента.

    Учитывает mute_extend_mode и mute_max_seconds."""
    first = group[0]
    last = group[-1]

    # raw_mute: возвращаем тайминги whisper КАК ЕСТЬ для первого матча группы.
    # Без padding, без extend, без word-cap, без segment-cap, без склейки с
    # соседними. Диагностический режим — посмотреть, что реально выдаёт whisper.
    if config.raw_mute:
        return (
            first.local_start_ms,
            first.local_end_ms,
            first.stream_start_ms,
            first.stream_end_ms,
        )

    # База: от start первого мата до end последнего мата.
    base_local_start = first.local_start_ms
    base_local_end = last.local_end_ms
    base_stream_start = first.stream_start_ms
    base_stream_end = last.stream_end_ms

    if config.mute_extend_mode == "segment-tail":
        # Расширяем правую границу до конца сегмента, если он известен.
        if first.segment_local_end_ms is not None:
            base_local_end = max(base_local_end, first.segment_local_end_ms)
        if first.segment_stream_end_ms is not None:
            base_stream_end = max(base_stream_end, first.segment_stream_end_ms)
    else:
        # Режим 'word': доверяем word-таймингам. Но whisper.cpp иногда ставит
        # последнему слову сегмента end = конец всего сегмента (слово на 9.48с).
        # Обрезаем хвостовое слово до правдоподобной длительности, иначе мьют
        # раздувается до потолка на одном слове.
        max_word_ms = int(config.mute_max_word_seconds * 1000)
        last_local_dur = last.local_end_ms - last.local_start_ms
        if last_local_dur > max_word_ms:
            base_local_end = last.local_start_ms + max_word_ms
        last_stream_dur = last.stream_end_ms - last.stream_start_ms
        if last_stream_dur > max_word_ms:
            base_stream_end = last.stream_start_ms + max_word_ms

    # Применяем padding.
    padded_local_start = max(0, base_local_start - config.mute_padding_before_ms)
    padded_local_end = base_local_end + config.mute_padding_after_ms
    padded_stream_start = max(0, base_stream_start - config.mute_padding_before_ms)
    padded_stream_end = base_stream_end + config.mute_padding_after_ms

    # Ограничиваем before-padding концом предыдущего слова — мьют не должен
    # залезать на соседнюю чистую речь, когда слова идут вплотную ('это блять'
    # без паузы между ними). В паузах padding сохраняется (предыдущее слово
    # кончилось раньше, чем start - padding), а во вплотную идущих словах
    # автоматически схлопывается до начала мата.
    prev_local_end = getattr(first, "prev_word_local_end_ms", None)
    if prev_local_end is not None and padded_local_start < prev_local_end:
        padded_local_start = min(prev_local_end, base_local_start)
    prev_stream_end = getattr(first, "prev_word_stream_end_ms", None)
    if prev_stream_end is not None and padded_stream_start < prev_stream_end:
        padded_stream_start = min(prev_stream_end, base_stream_start)

    # Жёсткий потолок длительности — чтобы аномальный 30-секундный сегмент
    # не превратился в 30-секундный мьют. Считаем от начала первого слова.
    max_ms = int(config.mute_max_seconds * 1000)
    cap_local = padded_local_start + max_ms
    cap_stream = padded_stream_start + max_ms
    if padded_local_end > cap_local:
        padded_local_end = cap_local
    if padded_stream_end > cap_stream:
        padded_stream_end = cap_stream

    return padded_local_start, padded_local_end, padded_stream_start, padded_stream_end


def build_decisions(
    stream_path: Path,
    original_path: Path,
    range_in: TimeSpan,
    range_out: TimeSpan,
    matches: list[ProfanityMatch],
    config: PipelineConfig,
    transcript_cache: Path | None = None,
    audio_cache: Path | None = None,
) -> dict[str, Any]:
    cuts: list[dict[str, Any]] = []
    mutes: list[dict[str, Any]] = []

    groups = _group_by_segment(matches, join_gap_ms=config.mute_join_gap_ms)
    # raw_mute: разбиваем все группы на одиночные матчи — никакой склейки,
    # каждый банворд = отдельный мьют по чистым таймингам whisper.
    if config.raw_mute:
        groups = [[m] for m in matches]
    # Source строка для каждого cut/mute: отражает реальный backend транскрипции
    # ('whisperx' | 'whispercpp') + нормализатор для banword-match (pymorphy3).
    # Используется в decisions.json и в Vegas-экспорте как audit trail.
    transcriber_label = {
        "whisperx": "whisperx",
        "whispercpp": "whisper.cpp",
    }.get(config.transcriber, config.transcriber)
    source_label = f"{transcriber_label}+pymorphy3"
    for index, group in enumerate(groups, start=1):
        first = group[0]
        last = group[-1]
        (
            padded_local_start,
            padded_local_end,
            padded_stream_start,
            padded_stream_end,
        ) = _compute_mute_window(group, config)

        # Защитная сетка для оставшихся коротких/цензурированных фрагментов:
        # если хоть один матч в группе флагнут как needs_review — весь mute
        # уходит в status='review', и vegas_export.exportable_mutes его
        # НЕ экспортирует в Vegas (см. фильтр на строке 30). Флаг остаётся
        # в decisions.json — пользователь может вручную поставить 'accepted',
        # если хочет замьютить.
        #
        # Раньше здесь стоял default_status='accepted' для всех — комментарий
        # утверждал что «не доверяем score/уверенности», но по факту это
        # превращало 15 ложных срабатываний на 'её/ей/бы' в реальные мьюты
        # в Vegas. См. detection_2b4b1a5c89ffaa3a.json.
        needs_review = any(bool(getattr(m, "needs_review", False)) for m in group)
        default_status = "review" if needs_review else "accepted"

        # Объединённое слово/контекст: все matched-токены через пробел.
        joined_word = " ".join(m.word for m in group if m.word).strip() or first.word
        joined_lemma = " ".join(m.lemma for m in group if m.lemma).strip() or first.lemma
        joined_banword = ", ".join(sorted({m.banword for m in group if m.banword}))
        match_types = sorted({m.match_type for m in group})
        avg_confidence = None
        confidences = [m.confidence for m in group if m.confidence is not None]
        if confidences:
            avg_confidence = round(sum(confidences) / len(confidences), 4)

        common = {
            "word": joined_word,
            "reason": "profanity",
            "intro_risk": first.local_start_ms < 60_000,
            "match_type": match_types[0] if len(match_types) == 1 else "+".join(match_types),
            "needs_review": needs_review,
            "lemma": joined_lemma,
            "banword": joined_banword,
            "confidence": avg_confidence,
            "timing_source": first.timing_source,
            "source": source_label,
            "segment_id": first.segment_id,
            "matched_token_count": len(group),
            "extend_mode": config.mute_extend_mode,
        }
        cuts.append(
            {
                "id": f"cut_{index:06d}",
                "start": _seconds(padded_local_start),
                "end": _seconds(padded_local_end),
                "action": "CUT",
                "target": "audio",
                "operation": "split_and_mute_audio",
                "status": default_status,
                "stream_start": _seconds(padded_stream_start),
                "stream_end": _seconds(padded_stream_end),
                **common,
            }
        )
        mutes.append(
            {
                "id": f"mute_{index:06d}",
                "start": _seconds(padded_local_start),
                "end": _seconds(padded_local_end),
                "action": "MUTE",
                "status": default_status,
                "stream_start": _seconds(padded_stream_start),
                "stream_end": _seconds(padded_stream_end),
                "raw_local_start": _seconds(first.local_start_ms),
                "raw_local_end": _seconds(last.local_end_ms),
                "raw_stream_start": _seconds(first.stream_start_ms),
                "raw_stream_end": _seconds(last.stream_end_ms),
                "segment_local_end": (
                    _seconds(first.segment_local_end_ms)
                    if first.segment_local_end_ms is not None
                    else None
                ),
                "segment_stream_end": (
                    _seconds(first.segment_stream_end_ms)
                    if first.segment_stream_end_ms is not None
                    else None
                ),
                **common,
            }
        )

    return {
        "schema_version": "1.1",
        "source": str(stream_path),
        "original": str(original_path),
        "range_in": range_in.format(),
        "range_out": range_out.format(),
        "local_duration": (range_out - range_in).format(),
        "cuts": cuts,
        "mutes": mutes,
        "settings": asdict(config),
        "caches": {
            "audio": str(audio_cache) if audio_cache else None,
            "transcript": str(transcript_cache) if transcript_cache else None,
        },
        "summary": {
            "cuts": len(cuts),
            "mutes": len(mutes),
            "raw_matches": len(matches),
        },
    }


def write_decisions(path: Path, decisions: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(decisions, ensure_ascii=False, indent=2), encoding="utf-8")
