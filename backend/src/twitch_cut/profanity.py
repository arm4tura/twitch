from __future__ import annotations

from dataclasses import dataclass
import re
from pathlib import Path
from typing import Iterable

from .timecode import TimeSpan
from .transcription import iter_words

# Strip everything that is not a letter/digit/dash. We keep cyrillic + latin.
_TOKEN_RE = re.compile(r"[^0-9a-zа-яё-]+", re.IGNORECASE)
# After we drop all asterisks, this regex tells us if a token had censorship.
_CENSOR_RE = re.compile(r"[\*]+")
# Plain letters only — used for stem matching after we drop dashes/digits too.
_LETTERS_RE = re.compile(r"[^a-zа-яё]+", re.IGNORECASE)


# Curated list of Russian obscene stems. A token that starts with any of these
# (after normalization) is flagged as profanity even if it is a rare inflection
# Whisper produced or a partial word. Keep short stems only when they cannot
# match obvious non-obscene words.
OBSCENE_STEMS: tuple[str, ...] = (
    "бля",      # бля, блядь, блять, блядина, блядство, блядский...
    "бляд",
    "блят",
    "хуй",      # хуй, хуйня, хуйло, хуёвый, хуярить, нахуй...
    "хуе",      # охуеть, охуенно, хуевый
    "хуё",
    "хуя",      # хуярить, хуяк
    "нахуй",
    "нахуя",
    "похуй",
    "пизд",     # пизда, пиздец, пиздос, пиздабол, пиздюк...
    "ебал",     # ебал, ебала, ебали
    "ебат",     # ебать, ебаться
    "ебан",     # ебанный, ебаный, ебанутый
    "ебаш",     # ебашить
    "ебл",      # ебло, еблан
    "ебуч",
    "ёб",       # normalized to "еб"
    "еб",       # standalone — used carefully via fragment glue logic
    "заеб",
    "наеб",
    "съеб",
    "доеб",
    "проеб",
    "поеб",
    "отъеб",
    "въеб",     # въебать, въебал, въебите, въёбанный (ё→е при нормализации)
    "вьеб",     # whisper часто пишет въеб через мягкий знак вместо твёрдого
    "взъеб",    # взъебать, взъебка
    "разъеб",   # разъебать, разъёбанный
    "подъеб",   # подъебать, подъёб, подъёбка
    "объеб",    # объебать, объёбанный
    "охуе",
    "охуи",
    "охуя",
    "охуё",
    "уеба",
    "уебо",
    "уеби",
    "уебищ",
    "распизд",
)

# Stems shorter than this letter count are too risky to flag alone — they
# need additional evidence (censorship, exact match, or glued context).
_SAFE_STEM_MIN_LEN = 3

# Single-letter or two-letter normalized tokens are almost always Whisper
# fragments. We do NOT auto-mute these. We flag them for review only when
# they look obscene (start of an obscene stem, e.g. "б", "е", "х").
#
# Префиксы 'н' и 'п' были выпилены: они ловили частицы 'на/не/но/ну/по',
# раздувая список review-матчей в 2 раза без какой-либо ценности — это
# самые частые служебные слова, а не цензурированный мат. Цензурированные
# 'нх'/'пз' попадают в censored_pattern в шаге #3, а не сюда.
_SUSPICIOUS_SHORT_PREFIXES: tuple[str, ...] = ("б", "е", "х")

# Whitelist коротких слов, которые начинаются на 'б/е/х' но матом НЕ являются.
# Без него шаг #6 ловит 15+ ложных срабатываний на 10 минут стрима:
# 'её' → нормализуется в 'ее' → 2 буквы, начинается на 'е' → match.
# Аналогично 'ей', 'бы', 'бо', 'ею', 'их', 'им'. И decisions.py по умолчанию
# ставит status='accepted' даже для needs_review — эти слова реально мьютятся
# в Vegas. См. detection_2b4b1a5c89ffaa3a.json: 8×'ей', 4×'её', 1×'бы'.
_SAFE_SHORT_TOKENS: frozenset[str] = frozenset({
    # местоимения (после ё→е)
    "ее", "ей", "ею", "их", "им", "ем", "ел", "ем",
    # частицы/союзы/предлоги
    "бы", "бо", "би", "ех",
    # междометия
    "эх", "ох", "ах", "ух", "эй",
})


@dataclass(frozen=True)
class BanwordEntry:
    raw: str
    normalized: str
    lemma: str


@dataclass
class ProfanityMatch:
    word: str
    normalized: str
    lemma: str
    banword: str
    banword_lemma: str
    local_start_ms: int
    local_end_ms: int
    stream_start_ms: int
    stream_end_ms: int
    confidence: float | None
    segment_id: str | None
    match_type: str               # surface | lemma | stem | censored | glued
    timing_source: str
    needs_review: bool = False    # true for ambiguous short fragments
    # Границы whisper-сегмента, в котором находится слово. Нужны для расширения
    # мьюта от первого мата до конца сегмента (фразы).
    segment_local_start_ms: int | None = None
    segment_local_end_ms: int | None = None
    segment_stream_start_ms: int | None = None
    segment_stream_end_ms: int | None = None
    # Конец предыдущего слова в транскрипте (для ограничения before-padding,
    # чтобы мьют не залезал на соседнюю чистую речь, когда слова идут вплотную).
    prev_word_local_end_ms: int | None = None
    prev_word_stream_end_ms: int | None = None
    # Начало следующего слова в транскрипте (для ограничения after-padding —
    # симметрично prev_word_*_end_ms — чтобы after-padding мата не залезал на
    # следующее чистое слово: 'бля какой' без паузы, after-padding 'бля' не
    # должен глотать начало 'какой'). См. _compute_mute_window.
    next_word_local_start_ms: int | None = None
    next_word_stream_start_ms: int | None = None
    # Индексы слов транскрипта, которые покрывает матч (позиция в общем списке
    # iter_words). Обычный матч — одно слово (start==end). Glued-матч склеен из
    # двух соседних слов (end = start+1). Нужны для склейки соседних матов в один
    # мьют ТОЛЬКО когда между ними нет чистого слова — иначе 'бля какой даун'
    # проглотил бы невинное 'какой'. См. _group_by_segment.
    word_start_index: int | None = None
    word_end_index: int | None = None


class RussianNormalizer:
    def __init__(self) -> None:
        self._morph = None
        try:
            import pymorphy3  # type: ignore

            self._morph = pymorphy3.MorphAnalyzer()
        except Exception:
            self._morph = None

    def normalize(self, text: str) -> str:
        """Lowercase, yo→e, drop punctuation and asterisks."""
        cleaned = text.lower().replace("ё", "е").strip()
        cleaned = _TOKEN_RE.sub("", cleaned)
        return cleaned

    def normalize_keep_censor(self, text: str) -> tuple[str, bool]:
        """Return (letters_only_normalized, had_censorship).
        Asterisks are detected first, then stripped together with all non-letters.
        """
        had_censor = bool(_CENSOR_RE.search(text))
        cleaned = text.lower().replace("ё", "е").strip()
        cleaned = _LETTERS_RE.sub("", cleaned)
        return cleaned, had_censor

    def lemma(self, token: str) -> str:
        normalized = self.normalize(token)
        if not normalized:
            return ""
        if self._morph is None:
            return normalized
        parsed = self._morph.parse(normalized)
        if not parsed:
            return normalized
        return parsed[0].normal_form.replace("ё", "е")


def load_banwords(path: Path, normalizer: RussianNormalizer | None = None) -> list[BanwordEntry]:
    normalizer = normalizer or RussianNormalizer()
    entries: list[BanwordEntry] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        normalized = normalizer.normalize(line)
        if not normalized:
            continue
        entries.append(BanwordEntry(raw=line, normalized=normalized, lemma=normalizer.lemma(normalized)))
    return entries


def _matches_stem(normalized: str) -> str | None:
    """Return the stem if normalized token starts with a known obscene stem."""
    if not normalized:
        return None
    for stem in OBSCENE_STEMS:
        if len(stem) < _SAFE_STEM_MIN_LEN:
            # Short stems require exact prefix AND token length >= stem+1
            if normalized.startswith(stem) and len(normalized) >= len(stem) + 1:
                return stem
        else:
            if normalized.startswith(stem):
                return stem
    return None


def _build_match(
    word_obj: dict,
    normalized: str,
    lemma: str,
    entry: BanwordEntry,
    match_type: str,
    range_in: TimeSpan,
    needs_review: bool = False,
    override_text: str | None = None,
    override_start: float | None = None,
    override_end: float | None = None,
    prev_word_end: float | None = None,
    next_word_start: float | None = None,
    word_start_index: int | None = None,
    word_end_index: int | None = None,
) -> ProfanityMatch | None:
    start = override_start if override_start is not None else word_obj.get("start")
    end = override_end if override_end is not None else word_obj.get("end")
    if start is None or end is None:
        return None
    local_start = TimeSpan.from_seconds(start)
    local_end = TimeSpan.from_seconds(end)
    if local_end <= local_start:
        local_end = TimeSpan(local_start.ms + 200)
    stream_start = range_in + local_start
    stream_end = range_in + local_end
    confidence = word_obj.get("score")
    text = override_text if override_text is not None else str(word_obj.get("text", "")).strip()
    seg_start = word_obj.get("segment_start")
    seg_end = word_obj.get("segment_end")
    seg_local_start_ms = None
    seg_local_end_ms = None
    seg_stream_start_ms = None
    seg_stream_end_ms = None
    if seg_start is not None:
        sls = TimeSpan.from_seconds(seg_start)
        seg_local_start_ms = sls.ms
        seg_stream_start_ms = (range_in + sls).ms
    if seg_end is not None:
        sle = TimeSpan.from_seconds(seg_end)
        seg_local_end_ms = sle.ms
        seg_stream_end_ms = (range_in + sle).ms
    prev_word_local_end_ms = None
    prev_word_stream_end_ms = None
    if prev_word_end is not None:
        pwe = TimeSpan.from_seconds(prev_word_end)
        prev_word_local_end_ms = pwe.ms
        prev_word_stream_end_ms = (range_in + pwe).ms
    next_word_local_start_ms = None
    next_word_stream_start_ms = None
    if next_word_start is not None:
        nws = TimeSpan.from_seconds(next_word_start)
        next_word_local_start_ms = nws.ms
        next_word_stream_start_ms = (range_in + nws).ms
    return ProfanityMatch(
        word=text,
        normalized=normalized,
        lemma=lemma,
        banword=entry.raw,
        banword_lemma=entry.lemma,
        local_start_ms=local_start.ms,
        local_end_ms=local_end.ms,
        stream_start_ms=stream_start.ms,
        stream_end_ms=stream_end.ms,
        confidence=float(confidence) if confidence is not None else None,
        segment_id=word_obj.get("segment_id"),
        match_type=match_type,
        timing_source=str(word_obj.get("timing_source", "word")),
        needs_review=needs_review,
        segment_local_start_ms=seg_local_start_ms,
        segment_local_end_ms=seg_local_end_ms,
        segment_stream_start_ms=seg_stream_start_ms,
        segment_stream_end_ms=seg_stream_end_ms,
        prev_word_local_end_ms=prev_word_local_end_ms,
        prev_word_stream_end_ms=prev_word_stream_end_ms,
        next_word_local_start_ms=next_word_local_start_ms,
        next_word_stream_start_ms=next_word_stream_start_ms,
        word_start_index=word_start_index,
        word_end_index=word_end_index if word_end_index is not None else word_start_index,
    )


def detect_profanity(
    transcript: dict,
    banwords: Iterable[BanwordEntry],
    range_in: TimeSpan,
    normalizer: RussianNormalizer | None = None,
) -> list[ProfanityMatch]:
    normalizer = normalizer or RussianNormalizer()
    banword_list = list(banwords)
    surfaces = {entry.normalized: entry for entry in banword_list}
    lemmas = {entry.lemma: entry for entry in banword_list}

    # Synthetic banword entry for stem-only hits — we still want a banword name.
    def _stem_entry(stem: str) -> BanwordEntry:
        return BanwordEntry(raw=f"<stem:{stem}>", normalized=stem, lemma=stem)

    matches: list[ProfanityMatch] = []
    # We need lookahead/lookbehind for glued tokens — collect first.
    words = list(iter_words(transcript))
    used_glued: set[int] = set()

    for idx, word in enumerate(words):
        raw_text = str(word.get("text", ""))
        normalized = normalizer.normalize(raw_text)
        letters_only, had_censor = normalizer.normalize_keep_censor(raw_text)

        # Конец предыдущего слова в том же сегменте — нужен, чтобы before-padding
        # мьюта не залезал на соседнюю чистую речь, когда слова идут вплотную
        # ('это блять' без паузы). Берём предыдущее слово только если оно в том же
        # whisper-сегменте (иначе пауза между сегментами и ограничивать нечего).
        prev_word_end = None
        if idx > 0:
            prev = words[idx - 1]
            if prev.get("segment_id") == word.get("segment_id"):
                prev_word_end = prev.get("end")

        # Начало следующего слова в том же сегменте — симметрично prev_word_end,
        # ограничивает after-padding, чтобы мьют не залезал на следующее чистое
        # слово ('бля какой' без паузы — after-padding 'бля' не глотает 'какой').
        # Для glued-матча (склеен из idx и idx+1) следующее слово — idx+2; его
        # считаем отдельно в шаге #5.
        next_word_start = None
        if idx + 1 < len(words):
            nxt_w = words[idx + 1]
            if nxt_w.get("segment_id") == word.get("segment_id"):
                next_word_start = nxt_w.get("start")

        # ---------- 1. exact surface match ----------
        entry = surfaces.get(normalized) if normalized else None
        if entry is not None:
            lemma = normalizer.lemma(normalized)
            m = _build_match(word, normalized, lemma, entry, "surface", range_in, prev_word_end=prev_word_end, next_word_start=next_word_start, word_start_index=idx)
            if m:
                matches.append(m)
            continue

        # ---------- 2. exact lemma match ----------
        lemma = normalizer.lemma(normalized) if normalized else ""
        entry = lemmas.get(lemma) if lemma else None
        if entry is not None:
            m = _build_match(word, normalized, lemma, entry, "lemma", range_in, prev_word_end=prev_word_end, next_word_start=next_word_start, word_start_index=idx)
            if m:
                matches.append(m)
            continue

        # ---------- 3. censored match (еб***, ***ь, б***, ***) ----------
        if had_censor:
            if letters_only:
                stem = _matches_stem(letters_only)
                if stem is not None:
                    m = _build_match(
                        word, letters_only, letters_only,
                        _stem_entry(stem), "censored", range_in,
                        needs_review=False,
                        prev_word_end=prev_word_end,
                        next_word_start=next_word_start,
                        word_start_index=idx,
                    )
                    if m:
                        matches.append(m)
                    continue
                # Very short censored fragments like **ь — only review flag.
                if 1 <= len(letters_only) <= 2 and letters_only.startswith(_SUSPICIOUS_SHORT_PREFIXES):
                    m = _build_match(
                        word, letters_only, letters_only,
                        _stem_entry("censor"), "censored", range_in,
                        needs_review=True,
                        prev_word_end=prev_word_end,
                        next_word_start=next_word_start,
                        word_start_index=idx,
                    )
                    if m:
                        matches.append(m)
                    continue
            # Pure *** with no letters — still mark for review.
            m = _build_match(
                word, "", "",
                _stem_entry("censor"), "censored", range_in,
                needs_review=True,
                override_text=raw_text.strip(),
                prev_word_end=prev_word_end,
                next_word_start=next_word_start,
                word_start_index=idx,
            )
            if m:
                matches.append(m)
            continue

        # ---------- 4. stem match ----------
        if normalized:
            stem = _matches_stem(normalized)
            if stem is not None:
                m = _build_match(
                    word, normalized, lemma or normalized,
                    _stem_entry(stem), "stem", range_in,
                    prev_word_end=prev_word_end,
                    next_word_start=next_word_start,
                    word_start_index=idx,
                )
                if m:
                    matches.append(m)
                continue

        # ---------- 5. glued-token (Whisper split а+бля+дь) ----------
        # Only consider glue when current token is short and same segment as next.
        if idx in used_glued:
            continue
        if normalized and len(normalized) <= 3 and idx + 1 < len(words):
            nxt = words[idx + 1]
            if nxt.get("segment_id") == word.get("segment_id"):
                next_norm = normalizer.normalize(str(nxt.get("text", "")))
                glued = normalized + next_norm
                if glued:
                    entry_g = surfaces.get(glued) or lemmas.get(normalizer.lemma(glued))
                    stem_g = _matches_stem(glued)
                    if entry_g is not None or stem_g is not None:
                        start = word.get("start")
                        end = nxt.get("end")
                        if start is not None and end is not None:
                            used_glued.add(idx + 1)
                            # Для glued следующее чистое слово — idx+2 (idx и
                            # idx+1 уже съедены склейкой), в том же сегменте.
                            glued_next_start = None
                            if idx + 2 < len(words):
                                nxt2 = words[idx + 2]
                                if nxt2.get("segment_id") == word.get("segment_id"):
                                    glued_next_start = nxt2.get("start")
                            m = _build_match(
                                word, glued, normalizer.lemma(glued),
                                entry_g if entry_g is not None else _stem_entry(stem_g or ""),
                                "glued", range_in,
                                override_text=f"{raw_text}{nxt.get('text', '')}".strip(),
                                override_start=start,
                                override_end=end,
                                prev_word_end=prev_word_end,
                                next_word_start=glued_next_start,
                                word_start_index=idx,
                                word_end_index=idx + 1,
                            )
                            if m:
                                matches.append(m)
                            continue

        # ---------- 6. suspicious short fragment (б, ь, е) ----------
        if normalized and len(normalized) <= 2:
            # Whitelist: 'её/ей/бы/эх/ох' — не мат, а частотные служебные слова.
            # Без этой проверки шаг #6 замьютит их всех (см. комментарий у
            # _SAFE_SHORT_TOKENS).
            if normalized in _SAFE_SHORT_TOKENS:
                continue
            if normalized.startswith(_SUSPICIOUS_SHORT_PREFIXES) or normalized in {"ь", "ъ"}:
                m = _build_match(
                    word, normalized, normalized,
                    _stem_entry("short"), "short_fragment", range_in,
                    needs_review=True,
                    prev_word_end=prev_word_end,
                    next_word_start=next_word_start,
                    word_start_index=idx,
                )
                if m:
                    matches.append(m)
                continue

    return matches
