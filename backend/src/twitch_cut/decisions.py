from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from typing import Any

from .config import PipelineConfig
from .mute_refine import (
    WavEnergy,
    global_threshold,
    open_energy,
    reanchor_window,
    silence_before_ms,
    trim_window,
)
from .profanity import ProfanityMatch
from .timecode import TimeSpan


def _seconds(ms: int) -> float:
    return round(ms / 1000.0, 3)


def _group_by_segment(
    matches: list[ProfanityMatch],
    join_gap_ms: int = 600,
) -> list[list[ProfanityMatch]]:
    """Группирует матчи В ОДИН мьют ТОЛЬКО когда они идут словами ВПЛОТНУЮ —
    между ними нет ни одного чистого (не-матного) слова транскрипта И нет
    заметной паузы.

    Критерий соседства — по индексу слова, а не по временно́му разрыву. Раньше
    склейка шла по join_gap_ms (≤600мс между концом и началом), и это глотало
    невинное слово, если оно короткое: 'бля какой даун' — 'какой' быстрое,
    разрыв бля→даун <600мс, мьют бля..даун проглатывал 'какой'. Теперь два
    мата склеиваются, только если второй начинается на слове сразу за первым
    (word_start_index == prev.word_end_index + 1). Если между ними есть слово —
    это разные группы, чистая речь между матами НЕ мьютится.

    Но одного соседства по индексу мало: соседние по индексу слова могут стоять
    в СЕКУНДАХ друг от друга (пауза, вздох, тишина — слов там нет, поэтому
    индексы подряд). Такая склейка давала мьют на весь разбег: на реальном
    стриме «блять» 337.8 и «блять» 341.5 (индексы 898 и 899) слиплись в мьют
    длиной 4.1 с. Поэтому join_gap_ms снова участвует — но как НЕОБХОДИМОЕ
    условие вместе с соседством, а не вместо него, иначе вернулся бы старый баг
    с проглоченным 'какой'.

    Требования склейки: тот же segment_id И индексы слов идут подряд И разрыв
    между словами ≤ join_gap_ms. Матчи без segment_id или без индексов остаются
    каждый в своей группе (консервативно)."""
    groups: list[list[ProfanityMatch]] = []
    for match in matches:
        prev = groups[-1][-1] if groups else None
        same_segment = (
            prev is not None
            and match.segment_id is not None
            and prev.segment_id == match.segment_id
        )
        # Соседство слов: текущий мат начинается ровно на следующем слове после
        # последнего слова предыдущего мата. Нет индексов — не склеиваем.
        adjacent = (
            prev is not None
            and prev.word_end_index is not None
            and match.word_start_index is not None
            and match.word_start_index == prev.word_end_index + 1
        )
        # И между ними нет паузы: индексы подряд ещё не значит «вплотную».
        close_enough = (
            prev is not None and match.local_start_ms - prev.local_end_ms <= join_gap_ms
        )
        if same_segment and adjacent and close_enough:
            groups[-1].append(match)
        else:
            groups.append([match])
    return groups


def _compute_mute_window(
    group: list[ProfanityMatch],
    config: PipelineConfig,
    *,
    expand_head_ms: int = 0,
    clear_of_neighbours: bool = True,
    energy: WavEnergy | None = None,
    speech_threshold: float = 0.0,
) -> tuple[int, int, int, int]:
    """Считает (local_start_ms, local_end_ms, stream_start_ms, stream_end_ms)
    для одного объединённого mute по группе матчей одного сегмента.

    Учитывает mute_extend_mode и mute_max_seconds.

    `expand_head_ms` добавляется к головному padding, `clear_of_neighbours=False`
    снимает клэмпы по соседним словам. Оба нужны ОДНОВРЕМЕННО и только для окна,
    стоящего на тишине (см. config.mute_expand_on_silence_ms): по отдельности
    прибавка к padding упирается в клэмп, а снятый клэмп возвращает всего
    padding, которого этому окну и не хватало."""
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
    head_ms = config.mute_padding_before_ms + max(0, expand_head_ms)
    padded_local_start = max(0, base_local_start - head_ms)
    padded_local_end = base_local_end + config.mute_padding_after_ms
    padded_stream_start = max(0, base_stream_start - head_ms)
    padded_stream_end = base_stream_end + config.mute_padding_after_ms

    # Ограничиваем before-padding концом предыдущего слова — мьют не должен
    # залезать на соседнюю чистую речь, когда слова идут вплотную ('это блять'
    # без паузы между ними). В паузах padding сохраняется (предыдущее слово
    # кончилось раньше, чем start - padding), а во вплотную идущих словах
    # автоматически схлопывается до начала мата.
    #
    # Посылка клэмпа — «разметка соседнего слова стоит на соседнем звуке». Под
    # уехавшей сеткой слов она ложна, и тогда клэмп снимают целиком: см.
    # clear_of_neighbours и config.mute_expand_on_silence_ms.
    #
    # `mute_head_cross_prev_ms` — третий, промежуточный вариант: клэмп остаётся,
    # но с допуском. Он нужен потому, что «вплотную идущие слова» — это не
    # редкий случай, а норма сетки ASR: конец предыдущего слова равен началу
    # мата, поэтому головной padding схлопывается в ноль в связной речи, и ручка
    # mute_padding_before_ms там не действует вообще. Замер (12 из 12 ручных
    # расширений головы человеком — ровно на этом клэмпе) — в config.py.
    if clear_of_neighbours:
        cross_ms = max(0, int(getattr(config, "mute_head_cross_prev_ms", 0) or 0))
        # СТРАЖ ПО ЗВУКУ. Допуск `mute_head_cross_prev_ms` снимает клэмп по концу
        # предыдущего слова вслепую — на фиксированную константу. Но сетка ASR в
        # связной речи идёт ВПЛОТНУЮ (конец соседа == начало мата), и тогда весь
        # допуск уходит в ЗВУЧАЩУЮ речь соседа: 100 мс `high` — это 100 мс чистого
        # слова перед матом, замьюченного в каждом таком окне. Поджатие по RMS их
        # не убирает (это звук), и человек снимал их руками запись за записью.
        #
        # Форсированное выравнивание уже поставило `base_local_start` на границу
        # мата (gap до соседа делится пополам, cap 120 мс — alignment.py), поэтому
        # перед этой границей мата уже нет: каждый мс допуска — в половину соседа.
        # Единственный вопрос — тишина там или голос соседа, и на него отвечает
        # тот же абсолютный порог, что и на «стоит ли окно на тишине». Урезаем
        # допуск до фактически доступной тишины перед началом мата: в связной речи
        # она ~0 (голову к началу мата и прижимает), в реальной паузе — остаётся, и
        # поведение прежнее (padding уходит в тишину, поджатие потом решит).
        #
        # Страж только УМЕНЬШАЕТ cross_ms, то есть может лишь ПОДНЯТЬ floor и
        # заглушить МЕНЬШЕ — ослабить мьют мата он не может по построению. Без
        # энергии (raw_mute, нет аудио-кэша) или без порога он инертен, и всё
        # работает ровно как раньше — тот же контракт «опция, но не фатальна», что
        # у поджатия и переноса.
        if (
            energy is not None
            and speech_threshold > 0.0
            and cross_ms > 0
            and getattr(config, "mute_head_cross_voiced_guard", False)
        ):
            cross_ms = min(
                cross_ms,
                silence_before_ms(energy, base_local_start, cross_ms, speech_threshold),
            )
        prev_local_end = getattr(first, "prev_word_local_end_ms", None)
        if prev_local_end is not None:
            floor_local = prev_local_end - cross_ms
            if padded_local_start < floor_local:
                padded_local_start = min(floor_local, base_local_start)
        prev_stream_end = getattr(first, "prev_word_stream_end_ms", None)
        if prev_stream_end is not None:
            floor_stream = prev_stream_end - cross_ms
            if padded_stream_start < floor_stream:
                padded_stream_start = min(floor_stream, base_stream_start)

        # Симметрично: ограничиваем after-padding началом СЛЕДУЮЩЕГО слова — мьют
        # не должен залезать на соседнюю чистую речь справа, когда слова идут
        # вплотную ('бля какой' без паузы — after-padding 'бля' не глотает начало
        # 'какой'). Берём next-слово ПОСЛЕДНЕГО мата группы (padding применяется к
        # хвосту). В паузах padding сохраняется (следующее слово начинается позже,
        # чем end + padding), во вплотную идущих словах схлопывается до конца мата.
        next_local_start = getattr(last, "next_word_local_start_ms", None)
        if next_local_start is not None and padded_local_end > next_local_start:
            padded_local_end = max(next_local_start, base_local_end)
        next_stream_start = getattr(last, "next_word_stream_start_ms", None)
        if next_stream_start is not None and padded_stream_end > next_stream_start:
            padded_stream_end = max(next_stream_start, base_stream_end)

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


def _trim_window_to_audio(
    energy: WavEnergy,
    window: tuple[int, int, int, int],
    config: PipelineConfig,
    cores: list[tuple[int, int]] | None = None,
) -> tuple[int, int, int, int, int, int]:
    """Поджать окно мьюта к фактической речи в аудио.

    Возвращает то же окно (local_start, local_end, stream_start, stream_end)
    плюс сколько мс срезано слева и справа — это идёт в decisions.json как
    audit trail: по нему видно, сработала подгонка или окно уже было точным.

    Считаем по ЛОКАЛЬНОЙ шкале: кэш аудио — это wav ровно на [range_in,
    range_out], его t=0 совпадает с local 0. На stream-шкалу переносим ту же
    дельту, потому что оба тайминга получены из одних и тех же слов.

    `cores` — локальные интервалы самих матерных слов (без padding), по одному
    на слово группы. Из них берётся порог тишины для каждого края (поэтому
    подгонка не уезжает на громкого соседа в padding и не теряет тихое слово
    рядом с громким) и по ним же видно, отделён ли звук в padding настоящей
    паузой, то есть чужой он или свой: см. trim_window.
    """
    local_start, local_end, stream_start, stream_end = window
    new_local_start, new_local_end = trim_window(
        energy,
        local_start,
        local_end,
        cores=cores,
        peak_ratio=config.mute_trim_peak_ratio,
        guard_ms=config.mute_trim_guard_ms,
        min_ms=config.mute_trim_min_ms,
    )
    trim_start = max(0, new_local_start - local_start)
    trim_end = max(0, local_end - new_local_end)
    return (
        new_local_start,
        new_local_end,
        max(0, stream_start + trim_start),
        max(0, stream_end - trim_end),
        trim_start,
        trim_end,
    )


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
    # Подгонка границ по аудио: открываем кэш один раз на весь проход. В
    # raw_mute не трогаем ничего — это диагностический режим «что выдал ASR».
    energy = (
        open_energy(audio_cache)
        if config.mute_trim_to_audio and not config.raw_mute
        else None
    )
    # Абсолютный порог «тут вообще есть звук» — один на весь проход (40 проб по
    # 0.7 с). Нужен только для проверки окон, стоящих на тишине; относительный
    # порог поджатия на этот вопрос ответить не может (см. mute_refine).
    speech_threshold = (
        global_threshold(energy)
        if energy is not None and config.mute_reanchor_ms > 0
        else 0.0
    )
    try:
        _build_entries(
            cuts=cuts,
            mutes=mutes,
            groups=groups,
            config=config,
            source_label=source_label,
            energy=energy,
            speech_threshold=speech_threshold,
        )
    finally:
        if energy is not None:
            energy.close()

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


def _build_entries(
    *,
    cuts: list[dict[str, Any]],
    mutes: list[dict[str, Any]],
    groups: list[list[ProfanityMatch]],
    config: PipelineConfig,
    source_label: str,
    energy: WavEnergy | None,
    speech_threshold: float = 0.0,
) -> None:
    """Собрать cut/mute записи по группам матчей (мутирует переданные списки)."""
    for index, group in enumerate(groups, start=1):
        first = group[0]
        last = group[-1]
        (
            padded_local_start,
            padded_local_end,
            padded_stream_start,
            padded_stream_end,
        ) = _compute_mute_window(
            group, config, energy=energy, speech_threshold=speech_threshold
        )

        # Ядра — сами матерные слова группы по таймингам ASR, без padding,
        # КАЖДОЕ отдельно. Подгонка ищет речь только в них и потому не может ни
        # уехать на соседнее слово, попавшее в padding, ни выкинуть тихое слово
        # группы рядом с громким.
        cores = [(m.local_start_ms, m.local_end_ms) for m in group]

        # Проверка «окно стоит на ТИШИНЕ» (тайминги слова у ASR уехали). Сама она
        # окно НЕ двигает: признак рабочий, но абсолютный порог равен уровню
        # громкой речи, поэтому тихий мат под ним тоже выглядит тишиной — и
        # автоперенос уводил мьют с мата чаще, чем приводил (замер в mute_refine и
        # config.mute_reanchor_move). Её вердикт делает две вещи: помечает окно
        # янтарным ('silent' → needs_review) и включает расширение ниже — ту самую
        # операцию, которой раньше не было ни одной.
        # С включённым mute_reanchor_move окно едет на найденную речь. Ядра едут
        # вместе с окном, иначе следующая за этим подгонка искала бы речь по
        # старым, уже неверным границам слов.
        reanchor = "ok"
        reanchor_offset_ms = 0
        if energy is not None and speech_threshold > 0.0:
            # Судим ПОЛНОЕ окно, вместе с допуском головы
            # (mute_head_cross_prev_ms), и это измеренный выбор, а не порядок
            # строк. Допуск может задеть звук соседа и тем самым сменить вердикт
            # 'silent' → 'ok', то есть выключить расширение на 300 мс. Проверено
            # обратное («судить окно в геометрии без допуска», шесть правленых
            # записей): вердикт возвращается 7 → 10 окон на 10 минутах, а
            # линейка не двигается (утечка 1.56 → 1.55 с) ценой +5 съеденных
            # чистых слов там же и +2 на 20 минутах. Причина: если допуск уже
            # достал до звука, посылка расширения («звука в окне нет вовсе»)
            # ложна, и 300 мс уходят в чистую речь соседа.
            new_local_start, new_local_end, reanchor = reanchor_window(
                energy,
                padded_local_start,
                padded_local_end,
                threshold=speech_threshold,
                search_ms=config.mute_reanchor_ms,
                guard_ms=config.mute_trim_guard_ms,
                move=config.mute_reanchor_move,
            )
            if reanchor == "reanchored":
                reanchor_offset_ms = new_local_start - padded_local_start
                # Конец двигаем своей дельтой: у конца файла перенос упирается в
                # длину записи, и сдвиг сторон перестаёт быть одинаковым.
                end_delta = new_local_end - padded_local_end
                padded_stream_start = max(0, padded_stream_start + reanchor_offset_ms)
                padded_stream_end = max(0, padded_stream_end + end_delta)
                padded_local_start = new_local_start
                padded_local_end = new_local_end
                cores = [
                    (a + reanchor_offset_ms, b + reanchor_offset_ms) for a, b in cores
                ]

        # Окно стоит на тишине → его РАСШИРЯЮТ назад, а не переносят.
        #
        # Переносить нельзя (замер выше), но и оставить как есть — значит оставить
        # мат слышимым: у пайплайна больше нет ни одного механизма, который делает
        # окно шире, а сетка слов у CTC опаздывает системно. Расширение назад
        # накрывает звук, стоящий ПЕРЕД разметкой, и стоит всего лишних мс мьюта —
        # поэтому тот же признак, что для переноса был слишком грубым, здесь
        # годится (см. config.mute_expand_on_silence_ms).
        #
        # Клэмпы по соседям при этом снимаются: они и съедают прибавку, а их
        # посылка «рядом чистая речь» под уехавшей сеткой ложна — разметка
        # предыдущего слова стоит как раз на звуке мата.
        expand_start_ms = 0
        expand_end_ms = 0
        expanded = False
        if reanchor in ("silent", "no_speech") and config.mute_expand_on_silence_ms > 0:
            (
                wide_local_start,
                wide_local_end,
                wide_stream_start,
                wide_stream_end,
            ) = _compute_mute_window(
                group,
                config,
                expand_head_ms=config.mute_expand_on_silence_ms,
                clear_of_neighbours=False,
            )
            expand_start_ms = max(0, padded_local_start - wide_local_start)
            expand_end_ms = max(0, wide_local_end - padded_local_end)
            expanded = expand_start_ms > 0 or expand_end_ms > 0
            padded_local_start = min(padded_local_start, wide_local_start)
            padded_local_end = max(padded_local_end, wide_local_end)
            padded_stream_start = min(padded_stream_start, wide_stream_start)
            padded_stream_end = max(padded_stream_end, wide_stream_end)

        # Подгонка по аудио. Даёт ТОЛЬКО сужение (см. mute_refine), поэтому
        # никакая из проверок выше не может быть ею отменена.
        #
        # Расширенное окно не поджимаем вовсе: звука выше абсолютного порога в нём
        # нет (за это его и расширили), а относительный порог поджатия в таком
        # окне считается от пика шума — то есть вернул бы окно примерно туда,
        # откуда его только что расширили.
        trim_start_ms = 0
        trim_end_ms = 0
        if energy is not None and not expanded:
            (
                padded_local_start,
                padded_local_end,
                padded_stream_start,
                padded_stream_end,
                trim_start_ms,
                trim_end_ms,
            ) = _trim_window_to_audio(
                energy,
                (
                    padded_local_start,
                    padded_local_end,
                    padded_stream_start,
                    padded_stream_end,
                ),
                config,
                cores,
            )

        # Сколько мс готового мьюта в итоге стоит на разметке ПРЕДЫДУЩЕГО слова.
        # Считается после поджатия — то есть это факт, а не намерение: поджатие
        # умеет отдать этот кусок назад соседу, если между ним и матом есть пауза
        # (mute_refine._gap_cut). Ноль при mute_head_cross_prev_ms=0 и в паузах.
        prev_end_local = getattr(first, "prev_word_local_end_ms", None)
        head_cross_ms = (
            max(0, prev_end_local - padded_local_start)
            if prev_end_local is not None
            else 0
        )

        # Защитная сетка для коротких/цензурированных фрагментов: если ВСЕ
        # матчи группы «мягкие» (needs_review), мьют уходит в status='review',
        # и vegas_export.exportable_mutes его НЕ экспортирует в Vegas (см.
        # фильтр на строке 30). Флаг остаётся в decisions.json — пользователь
        # может вручную поставить 'accepted', если хочет замьютить.
        #
        # Раньше здесь стоял default_status='accepted' для всех — комментарий
        # утверждал что «не доверяем score/уверенности», но по факту это
        # превращало 15 ложных срабатываний на 'её/ей/бы' в реальные мьюты
        # в Vegas. См. detection_2b4b1a5c89ffaa3a.json.
        #
        # Затем правило ушло в другую крайность: any() снимал с экспорта и те
        # группы, где рядом с мягким фрагментом стоял ТВЁРДЫЙ матч. На
        # vod_2847020280 так осталась незаглушённой «бля ь» (surface +
        # short_fragment): мат в кадре есть, мьюта нет, пока человек не
        # кликнет. Поэтому статус и флаг разведены:
        #   * статус решает all(...) — в review уходят только группы, в
        #     которых кроме мягких фрагментов ничего нет (защита 'её/ей/бы'
        #     сохраняется: там группа состоит из одного мягкого матча);
        #   * needs_review остаётся any(...) — это флаг для UI и отчёта.
        #     'accepted' + needs_review=True = «глушим, но подсвечено янтарным»
        #     (см. desktop/src/lib/mutes.ts::toUiMute).
        soft = [bool(getattr(m, "needs_review", False)) for m in group]
        # Окно стоит на ТИШИНЕ: 'silent' — речь рядом есть, но переносить окно за
        # человека мы не берёмся (mute_reanchor_move выключен по замеру);
        # 'no_speech' — речи в радиусе поиска нет вообще. Статус НЕ понижаем
        # (ослаблять мьют по подозрению нельзя — мат мог просто прозвучать тише
        # порога), но янтарным флагом такое окно помечаем: это единственный класс,
        # который машина честно не умеет починить, и слушать его надо человеку.
        needs_review = any(soft) or reanchor in ("no_speech", "silent")
        default_status = "review" if all(soft) else "accepted"

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
            # Сколько мс срезала подгонка по аудио с каждой стороны. 0/0 —
            # окно ASR уже совпало со звуком (или подгонка выключена).
            "audio_trim_start_ms": trim_start_ms,
            "audio_trim_end_ms": trim_end_ms,
            # Проверка «окно на тишине»: вердикт и сдвиг в мс.
            #   'ok'         — в окне есть звук;
            #   'silent'     — звука нет, речь рядом есть, окно НЕ двинуто
            #                  (mute_reanchor_move выключен) → needs_review;
            #   'no_speech'  — звука нет и рядом речи нет → needs_review;
            #   'reanchored' — окно перенесено на речь (только при включённом
            #                  mute_reanchor_move), сдвиг в reanchor_offset_ms.
            # Отчёт (audit.py) читает эти поля, чтобы не диагностировать
            # «уехавшие тайминги» там, где пайплайн их уже починил.
            "reanchor": reanchor,
            "reanchor_offset_ms": reanchor_offset_ms,
            # Сколько мс ДОБАВИЛО расширение окна, стоящего на тишине — зеркало
            # audio_trim_*: одна пара полей про сужение, другая про рост. 0/0 —
            # окно не расширяли (в нём есть звук, или ручка выключена, см.
            # config.mute_expand_on_silence_ms). Расширенное окно по построению
            # НЕ поджато: audio_trim_* у него нули.
            "expand_start_ms": expand_start_ms,
            "expand_end_ms": expand_end_ms,
            # Сколько мс мьюта стоит на разметке предыдущего слова — цена ручки
            # mute_head_cross_prev_ms в чистой речи соседа, окно за окном.
            "head_cross_ms": head_cross_ms,
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


def write_decisions(path: Path, decisions: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(decisions, ensure_ascii=False, indent=2), encoding="utf-8")
