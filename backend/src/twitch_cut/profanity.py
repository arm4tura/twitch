from __future__ import annotations

from dataclasses import dataclass
import logging
import re
from pathlib import Path
from typing import Any, Iterable, Sequence

from .alignment import align_words
from .keyword_spotting import MIN_KEYWORD_LEN, greedy_path, spot_keywords
from .timecode import TimeSpan
from .transcription import iter_words

logger = logging.getLogger(__name__)

# Strip everything that is not a letter/digit/dash. We keep cyrillic + latin.
_TOKEN_RE = re.compile(r"[^0-9a-zа-яё-]+", re.IGNORECASE)
# After we drop all asterisks, this regex tells us if a token had censorship.
_CENSOR_RE = re.compile(r"[\*]+")
# Plain letters only — used for stem matching after we drop dashes/digits too.
_LETTERS_RE = re.compile(r"[^a-zа-яё]+", re.IGNORECASE)


# --- ЕДИНЫЙ СЛОВАРЬ КОРНЕЙ ---------------------------------------------------
# ОДИН источник истины: этими списками и функцией roots_in() пользуется И
# детектор (detect_profanity), И audit-отчёт. До 2026-08-14 списков было два —
# у отчёта заведомо шире — и они разъехались: отчёт знал «гандон», а детектор
# нет, поэтому мат уезжал в ручную правку, а отчёт при этом рапортовал
# «пропуск» без возможности его закрыть.
#
# Политика поиска у каждого корня своя, и выбрана она НЕ на глаз: каждый
# кандидат прогнан по 5.1 млн словоформ словаря pymorphy3
# (`tools/roots_selfcheck.py`). Корень ищется ВНУТРИ слова, только если на
# словаре у него нет ни одного невинного совпадения — так ловится инфиксный
# мат («нихуя», «дохуя», «ахуенно», «нехуевая», «распиздяй»), который
# prefix-only поиск пропускал.
CORE_ROOTS: tuple[str, ...] = (
    # хуй-семья: инфикс обязателен — нихуя/дохуя/охуеть/ахуенно/нехуевый.
    # Ложные (страхуй, психует, сухую, лопоухую) снимает INNOCENT_SUBSTRINGS.
    "хуй", "хуе", "хуя", "хую", "хуи",
    "пизд",                                     # пизда, пиздец, распиздяй
    # еб-семья: приставочные формы уникальны, внутри слова ложных на словаре
    # нет вообще, поэтому ищем где угодно («да ты заебал», «нихуя не съеб»).
    "заеб", "наеб", "уеб", "съеб", "отъеб", "въеб", "вьеб", "взъеб",
    "подъеб", "разъеб", "объеб", "доеб", "проеб", "поеб",
    # приставки, забытые в первом наборе, — найдены пропуском на живой записи:
    # ASR расслышал «мозг уже выебала» правильно, а детектор не знал слова, и
    # мьют встал на соседнее «блять» в 200 мс правее (vod_2847020280, 2026-08-18).
    # «еб» здесь только с начала слова, поэтому приставочную форму ловит лишь
    # свой корень. Замер по 5.14 млн словоформ pymorphy3: невинных слов с этими
    # подстроками НОЛЬ (tools/roots_selfcheck.py), инфикс безопасен
    # («повыебываться», «поприебался»). «недоеб» не нужен — его ловит «доеб».
    "выеб", "приеб", "перееб", "изъеб",
    "ебуч", "ебач", "ебыр", "ебтв",
    # ругательства: у всех ноль ложных на словаре
    "пидор", "педор", "гандон", "гондон", "мудак", "мудил",
    "залуп", "дроч", "шлюх", "сучар",
    # только с начала слова, см. PREFIX_ONLY_ROOTS
    "еб", "бля", "епт", "пидар", "мудо", "манда", "хул",
)
"""Ядро мата: если корень нашёлся — это мат, вопросов нет."""

SPICY_ROOTS: tuple[str, ...] = (
    "сука", "суки", "суку", "сукой", "сучк",
    "хер", "говн", "дерьм", "жоп", "мраз", "гнид", "задрот", "чмо", "долбо",
)
"""Грубость на грани: мьютить или нет — решение пользователя, не баг детектора.
Детектор их НЕ ловит, отчёт показывает отдельным разделом."""

PREFIX_ONLY_ROOTS: frozenset[str] = frozenset(
    {"еб", "бля", "епт", "пидар", "мудо", "манда", "хул"}
)
"""Корни, которые ВНУТРИ слова почти всегда невинны — замерено по словарю:
«бля» живёт в рубля/сабля/дубляж/скобля (2782 формы), «еб» — в хлебать/
погребать/колебаться/судебный/щебень (десятки тысяч), «епт» — в рецепт/концепт/
скептик/шептать (441 форма), «пидар» — в скипидар/лапидарный, «мудо» — в
талмуд/махмуд/бермуды, «манда» — в команда/мандарин/мандат, «хул» — в
богохульник. Мат с этими корнями стоит в начале слова (блять, ебать, ебло,
еблан, ёпта, пидарас, мудозвон, хули), поэтому префикса хватает, а приставочный
мат ловят отдельные корни (заеб, наеб, съеб...).

«епт» добавлен по замеру на реальном прогоне: «ёпта» звучит в стриме и ПОПАДАЕТ
в текст распознавания, но ни одного корня в нём не было — мат уезжал в ручную
правку на ровном месте. Слов, начинающихся на «епт», в словаре pymorphy3 нет
вообще (0 форм из 5.14 млн), так что префикс безопасен."""

_SAFE_ROOT_MIN_LEN = 3
"""Корень короче трёх букв («еб») требует, чтобы слово было хотя бы на букву
длиннее корня: «ебу», «ебло» — да, само «еб» — нет, это обрывок ASR, и его
разбирает шаг короткого фрагмента."""

# --- КЛЮЧИ ДЛЯ ПОИСКА ПО ПОСТОРИОРАМ -----------------------------------------
# Спотинг (`keyword_spotting.py`) ищет цепочку букв прямо в посториорах CTC, где
# нет ни слов, ни пробелов, ни словаря — а значит нет и политики «только с начала
# слова». Поэтому список ключей НЕ равен CORE_ROOTS, но выводится из него:
#
#   * корни от четырёх букв, у которых в словаре pymorphy3 нет ни одной невинной
#     формы с такой подстрокой (заеб, съеб, пизд, пидор, гандон, дроч, шлюх,
#     залуп…), берутся АВТОМАТИЧЕСКИ: добавили корень в CORE_ROOTS — спотинг
#     узнал о нём сам, разъехаться списки не могут;
#   * корни короче четырёх букв (еб, бля, хуй, хуе, хуя, уеб) и корни из
#     PREFIX_ONLY_ROOTS в посториорах искать нечем — нужны их развёрнутые формы,
#     они перечислены руками в SPOT_EXTRA_FORMS.
#
# Цена развёрнутых форм измерена по тем же 5.14 млн словоформ, что и корни:
# «ебат» сидит внутри 405 невинных форм (дебаты, хлебать, колебать), «ебет» —
# 660 (щебет, дебет, гребет), «епта» — 441 (лепта, шептал, адепта), «блят» — 79
# (скоблят, оскорблять, влюблять). Спотинг это переживает не потому, что список
# чистый, а потому что `require_unheard` выбрасывает кандидата, если буквы ключа
# уже есть в тексте распознавания: услышанный «оскорблять» разбирает текстовый
# детектор — со словарём, леммами и INNOCENT_SUBSTRINGS, — а спотингу остаётся
# только НЕуслышанное. Плюс сами кандидаты идут в decisions.json как review.
SPOT_EXTRA_FORMS: tuple[str, ...] = (
    "блят", "бляд", "бляя",
    "ебан", "ебал", "ебат", "ебет", "ебут", "ебло", "ебуть", "епта",
    "хуйн", "хуйл", "хуйк", "хуйц", "хуев", "хуес", "охуе", "ахуе",
    "нихуя", "дохуя", "похуй",
    "уеба", "мудоз",
)
"""Формы для корней, которые сам спотинг искать не может: короче четырёх букв
или живут внутри невинных слов. Список откалиброван на реальном прогоне (см.
`SPOT_MIN_SCORE` в `config.py`), а не выписан по интуиции."""

SPOT_KEYWORDS: tuple[str, ...] = tuple(
    sorted(
        {
            root
            for root in CORE_ROOTS
            if len(root) >= MIN_KEYWORD_LEN and root not in PREFIX_ONLY_ROOTS
        }
        | set(SPOT_EXTRA_FORMS)
    )
)
"""Что спотинг ищет в посториорах. Один источник истины с CORE_ROOTS: ключ,
которого текстовый детектор не считает матом, — это баг, и его ловит тест."""

INNOCENT_SUBSTRINGS: tuple[str, ...] = (
    # -страховать / штриховать / психовать / шаховать / сплоховать:
    # страхуй, застрахуем, психует, штрихуя, шахуемый, сплохуем
    "страх", "штрих", "псих", "шаху", "шахов", "плох",
    # прилагательные на -хий в винительном: сухую, тихую, глухую, ветхую,
    # лихую, островерхую, белобрюхую, бухую, лопоухую, безухую
    "сух", "тих", "глух", "ветх", "лих", "верх", "брюх", "бух", "оух", "зух",
    # -манд-: команда, командарм, мандат, мандарин, мандраж, мандолина
    "команд", "мандат", "мандар", "мандраж", "мандол", "мандант",
    # хул-: богохульник, хулиган, хулить, хуление
    "хулиган", "хулит", "хулен", "хулим",
    # омографы spicy-корней: сучок, чмокать, херувим, Херсон, Сукачёв, надолбы
    "сучок", "сучков", "чмок", "херув", "херсон", "херт", "херес",
    "сукачев", "сукугун", "надолб", "барсук", "сукно",
    # spicy-корень внутри невинного слова: межоперационный, корчма, Кучма,
    # трясучка, присучка, шхеры, мохер, шухер, Блюхер, Теймраз, Йокосука
    "межопер", "корчм", "кучм", "яречм", "теймраз", "экономраз", "промраз",
    "трясучк", "присучк", "рассучк", "нансук", "йокосук", "карасук",
    "шхер", "мохер", "шухер", "махер", "блюхер", "минхер",
    # старые омографы еб-семьи оставлены как страховка на случай возврата
    # инфиксного поиска: щебень, ребёнок, небеса, лебедь, серебро, хребет
    "щебен", "хлебн", "требн", "гребн", "дебет", "ребен", "небес", "лебед",
    "жеребе", "серебр", "требо", "хребе", "хребт", "лохмат", "молох",
)
"""Слова-омографы, внутри которых корень встречается случайно. Слово с любой из
этих подстрок считается чистым целиком. Список выведен из прогона по словарю
pymorphy3, а не придуман: см. `tools/roots_selfcheck.py`."""

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
    "ее", "ей", "ею", "их", "им", "ем", "ел",
    # частицы/союзы/предлоги
    "бы", "бо", "би", "ех",
    # междометия
    "эх", "ох", "ах", "ух", "эй",
    # смех и хмыканье: 'ха/хе/хи' начинаются на 'х', но матерного корня в них
    # нет ни при какой лемматизации. Замер по vod_2847020280 (15 минут,
    # 35 окон): весь шаг #6 дал бы×5, ей×2, ее×1, ха×2 — то есть 'ха' было
    # единственным словом из шага #6, которое ещё не было в whitelist, и
    # человек снял оба таких мьюта руками. 'хм' добавлено по тому же признаку.
    "ха", "хе", "хи", "хм",
})


def decision_lists() -> dict[str, list[str]]:
    """Все списки, по которым детектор РЕШАЕТ, — для ключа кэша детекции.

    Кэш детекции лежит на диске (`cache/detection_*.json`), а эти списки живут в
    коде и в ключ иначе не попадают: добавили «ха» в безопасные короткие токены —
    прогон с готовым detection_*.json остался бы с прежним ложным мьютом, и
    правка выглядела бы «не работающей» (ровно так и было с `_SAFE_SHORT_TOKENS`).
    Поэтому НОВЫЙ список решений добавляется здесь, а не только в свой шаг:
    тогда кэш становится невалидным сам, без ручной чистки воркдиров.

    Сортировка обязательна: `frozenset`/`set` перебираются в произвольном
    порядке, и без неё ключ менялся бы от запуска к запуску (кэш никогда не
    попадал бы).

    `rules` — то же самое для решений, которые списком не выражаются: когда
    меняется ПРАВИЛО (а не словарь), кэш обязан стать невалидным ровно так же.
    Иначе правка проверяется на готовом detection_*.json и «не работает».
    """
    return {
        "core": list(CORE_ROOTS),
        "prefix_only": sorted(PREFIX_ONLY_ROOTS),
        "innocent": list(INNOCENT_SUBSTRINGS),
        "safe_short": sorted(_SAFE_SHORT_TOKENS),
        "suspicious_short": list(_SUSPICIOUS_SHORT_PREFIXES),
        "rules": [
            # Склейка короткого слова с соседом не применяется, если сосед мат
            # сам по себе; съеденное склейкой слово не даёт второго матча.
            "glue-only-when-neighbour-not-profane-2026-08-18",
            # Записи словаря короче 3 букв после нормализации не грузятся:
            # «б***» → «б» мьютил одиночные обрубки ASR без флага проверки.
            "banword-entries-shorter-than-3-letters-ignored-2026-08-18",
        ],
    }


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
    # Полоса «на проверку». Решается КАТЕГОРИЕЙ (как слово найдено), а не числом,
    # и это результат замера, а не лени. На 15 минутах реального стрима
    # confidence не разделяет находки вообще: у 140 матчей, попавших в правки
    # монтажёра, медиана 0.956, у 14 лишних — 0.952. Любой порог по confidence
    # сначала выкидывает настоящий мат: 0.8 → минус 4 попадания и ни одного
    # лишнего, 0.9 → минус 19 попаданий против 2 лишних. Лишние — это
    # 'блять'/'пиздец'/'ебать' с confidence 0.95+, которые человек просто не стал
    # глушить, и никакое число в пайплайне об этом не знает. Категория же
    # разделяет: surface/lemma/stem/glued дают 0.88–1.00 точности и идут в мьют,
    # а short_fragment (0/2) и posterior (1/2) — сюда, на проверку.
    # Пересчитать: tools/sensitivity_levels.py --breakdown.
    needs_review: bool = False
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


_MIN_BANWORD_LEN = 3
"""Минимальная длина НОРМАЛИЗОВАННОЙ записи словаря.

Замаскированные строки («б***ь», «б***») после нормализации схлопываются в
«бь» и «б», и дальше surface-матч ловит ЛЮБОЙ обрубок ASR: одиночное «б»
приезжает готовым мьютом с needs_review=False — мимо whitelist коротких слов
(_SAFE_SHORT_TOKENS) и мимо шага #6, который для таких токенов по замыслу
ставит только флаг «на проверку». Свежий прогон приложения на
vod_2847020280 (10 минут) дал два таких окна: 124.4 «б» и 594.4 «б» —
человек снял оба руками. Самоцензуру в самой расшифровке («б***ь» словом)
ловит отдельная ветка по «*», словарь ей не нужен, так что здесь не теряется
ничего.
"""


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
        if len(normalized) < _MIN_BANWORD_LEN:
            continue
        entries.append(BanwordEntry(raw=line, normalized=normalized, lemma=normalizer.lemma(normalized)))
    return entries


def roots_in(normalized: str, roots: Iterable[str] = CORE_ROOTS) -> list[str]:
    """Корни из списка, найденные в нормализованном слове (пусто — слово чистое).

    ЕДИНСТВЕННЫЙ матчер: им пользуются и детектор, и audit-отчёт, поэтому
    «отчёт знает корень, а детектор нет» стало невозможно by construction.
    Политика на корень — из PREFIX_ONLY_ROOTS, исключения — INNOCENT_SUBSTRINGS.
    """
    if not normalized or any(bad in normalized for bad in INNOCENT_SUBSTRINGS):
        return []
    hits: list[str] = []
    for root in roots:
        if root in PREFIX_ONLY_ROOTS:
            if not normalized.startswith(root):
                continue
            if len(root) < _SAFE_ROOT_MIN_LEN and len(normalized) < len(root) + 1:
                continue
        elif root not in normalized:
            continue
        hits.append(root)
    return hits


def match_root(normalized: str, roots: Iterable[str] = CORE_ROOTS) -> str | None:
    """Первый найденный корень или None — быстрый путь для детектора."""
    hits = roots_in(normalized, roots)
    return hits[0] if hits else None


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


def _matches_alone(
    normalized: str,
    surfaces: dict[str, BanwordEntry],
    lemmas: dict[str, BanwordEntry],
    normalizer: RussianNormalizer,
) -> bool:
    """Матчится ли слово САМО, без склейки с соседом — по тем трём путям, что
    дают готовый мьют: словоформа, лемма, корень (шаги #1, #2, #4).

    Review-пути (#3 цензура, #6 короткий фрагмент) сюда НЕ входят намеренно:
    если сосед сам по себе даёт только кандидата на проверку, склейка полезнее —
    она ровно за этим и написана (ASR рвёт «блядь» на «бля»+«дь», и по
    отдельности это два мягких фрагмента, а вместе — мат).
    """
    if not normalized:
        return False
    if normalized in surfaces:
        return True
    lemma = normalizer.lemma(normalized)
    if lemma and lemma in lemmas:
        return True
    return match_root(normalized) is not None


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
        # Слово, уже съеденное склейкой на прошлой итерации (шаг #5), пропускаем
        # ЦЕЛИКОМ. Раньше эта проверка стояла ВНУТРИ шага #5 — и шаги #1–#4
        # успевали выдать на ту же вторую половину склейки ещё один, отдельный
        # матч. Две находки на одно слово группировка не сливает (соседство
        # требует word_start_index == prev.word_end_index + 1, а здесь индексы
        # совпадают), и на выходе получались ДВА вложенных мьюта с общим концом.
        # Замер на vod_2847020280_20260817_2330 (20 минут): 7 таких пар из 59
        # окон, и 7 из 10 съеденных мьютом чистых слов — это первые половины
        # склеек («я», «да», «он», «то», «он», «ну», «это»).
        if idx in used_glued:
            continue
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
                stem = match_root(letters_only)
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
            stem = match_root(normalized)
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
        #
        # И только когда сосед сам по себе НЕ мат. Иначе склейка не добавляет
        # находку, а портит её: окно растягивается на текущее чистое слово, а
        # корень ищется поперёк границы слов и находится ложный. Все 7 дублей на
        # vod_2847020280_20260817_2330 такие — «ну»+«ебать» дало <stem:уеб>,
        # «то»+«поебанные» → <stem:поеб>, хотя настоящий мат стоит вторым словом
        # и матчится сам.
        if normalized and len(normalized) <= 3 and idx + 1 < len(words):
            nxt = words[idx + 1]
            next_norm = normalizer.normalize(str(nxt.get("text", "")))
            if (
                nxt.get("segment_id") == word.get("segment_id")
                and not _matches_alone(next_norm, surfaces, lemmas, normalizer)
            ):
                glued = normalized + next_norm
                if glued:
                    entry_g = surfaces.get(glued) or lemmas.get(normalizer.lemma(glued))
                    stem_g = match_root(glued)
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


def detect_posterior_profanity(
    posteriors: Any,
    transcript: dict,
    range_in: TimeSpan,
    *,
    min_score: float,
    keywords: Sequence[str] = SPOT_KEYWORDS,
    known: Iterable[ProfanityMatch] = (),
    slack_ms: int = 300,
) -> list[ProfanityMatch]:
    """Мат, которого НЕТ в тексте распознавания: поиск по посториорам CTC.

    Второй детектор на тех же правах, что `detect_profanity`, но входом ему
    служат не слова, а log-probs модели (`posteriors.read_posteriors`). Нужен
    там, где ASR услышал не то: «епанисир» вместо «ебанский» — корня в тексте
    нет, а в посториорах цепочка букв видна. Поиск и порог живут в
    `keyword_spotting.py`, здесь — только перевод находок в ProfanityMatch.

    `known` — уже найденные текстовым детектором матчи: кандидат, попавший в
    их окно (± `slack_ms`), выбрасывается. Иначе на каждый честно найденный мат
    приезжал бы его же дубль-review: спотинг видит те же слова, что и текст, и
    на реальном прогоне таких дублей было ~75 против одной новой находки.
    Запас в 300 мс закрывает padding мьюта и задержку эмиссии CTC.

    Все матчи помечены `needs_review=True` и `match_type="posterior"`: это
    кандидат для человека, а не готовый мьют (в Vegas такие не экспортируются).
    """
    segments = transcript.get("segments") or []
    words = [
        word
        for word in iter_words(transcript)
        if word.get("start") is not None and word.get("end") is not None
    ]
    busy = [
        (m.local_start_ms - slack_ms, m.local_end_ms + slack_ms) for m in known
    ]
    found: list[ProfanityMatch] = []
    aligned_bounds = 0
    for segment in getattr(posteriors, "segments", []):
        hits = spot_keywords(
            segment.log_probs,
            keywords,
            char_to_id=posteriors.char_to_id,
            blank_id=posteriors.blank_id,
            frame_shift=segment.frame_shift,
            min_score=min_score,
        )
        for hit in hits:
            bounds = _aligned_hit_bounds(posteriors, segment, hit)
            match = _posterior_match(
                hit,
                segment_start=segment.start,
                segments=segments,
                words=words,
                busy=busy,
                range_in=range_in,
                bounds=bounds,
            )
            if match is not None:
                found.append(match)
                aligned_bounds += bounds is not None
    found.sort(key=lambda m: m.local_start_ms)
    logger.info(
        "спотинг по посториорам: %d кандидатов (порог %.2f), границы по "
        "forced_align у %d из них",
        len(found),
        min_score,
        aligned_bounds,
    )
    return found


def _aligned_hit_bounds(posteriors: Any, segment: Any, hit: Any) -> tuple[float, float] | None:
    """Границы находки спотинга через forced_align: ключ ПОДСТАВЛЯЕТСЯ в гипотезу.

    То же, что делает контекстный биасинг (CTC-WS, arXiv 2406.07096): кандидат
    вставляется в гипотезу сегмента на место кадров, где его нашёл DP, и
    сегмент перевыравнивается по тем же log-probs. Так кандидат получает
    границы из того же источника, что и текстовые слова, а не по кадрам эмиссии
    первой и последней буквы: эмиссия CTC опаздывает на 1–3 кадра, и
    forced_align это опоздание снимает (замер в `alignment.py`: медиана слева
    51 → 24 мс, справа 27 → 0).

    Возвращает секунды от начала сегмента — тот же отсчёт, что у `hit.start`.
    None — выровнять не удалось (в словаре нет пробела, гипотеза не сошлась,
    слова с ключом в ответе нет): вызывающий остаётся на кадрах эмиссии.
    """
    space_id = posteriors.char_to_id(" ")
    key_ids = [posteriors.char_to_id(char) for char in hit.keyword]
    if space_id is None or any(token_id is None for token_id in key_ids):
        return None
    path = greedy_path(segment.log_probs, blank_id=posteriors.blank_id)
    hypothesis = _splice_keyword(
        prefix=[label for frame, label in path if frame < hit.start_frame],
        keyword=[int(token_id) for token_id in key_ids],
        suffix=[label for frame, label in path if frame >= hit.end_frame],
        space_id=int(space_id),
    )
    aligned = align_words(
        segment.log_probs,
        hypothesis,
        blank_id=posteriors.blank_id,
        to_char=lambda token_id: str(posteriors.vocab[token_id]),
        frame_shift=segment.frame_shift,
    )
    if not aligned:
        return None
    same = [word for word in aligned if word.text == hit.keyword]
    if not same:
        return None
    # Ключ может встретиться в сегменте и сам по себе (greedy услышал его в
    # другом месте) — берём тот, что ближе к кадрам находки.
    middle = (hit.start + hit.end) / 2.0
    word = min(same, key=lambda w: abs((w.start + w.end) / 2.0 - middle))
    return word.start, word.end


def _splice_keyword(
    *,
    prefix: Sequence[int],
    keyword: Sequence[int],
    suffix: Sequence[int],
    space_id: int,
) -> list[int]:
    """Гипотеза сегмента с ключом вместо того, что greedy услышал на его кадрах.

    Ключ обязан стать ОТДЕЛЬНЫМ словом: `alignment._group_into_words` режет
    слова по пробелу, и без него буквы соседей слиплись бы с ключом, а границы
    уехали бы на них. Двойные пробелы не нужны — forced_align потребовал бы
    между ними blank, а взять его негде, если пауза короткая.
    """
    out: list[int] = list(prefix)
    for chunk in (keyword, suffix):
        if not chunk:
            continue
        if out and out[-1] != space_id and chunk[0] != space_id:
            out.append(space_id)
        out.extend(chunk)
    return out


def _spot_entry(keyword: str) -> BanwordEntry:
    """Синтетический банворд для находки в посториорах: в decisions.json видно,
    что мьют пришёл не из текста, а из спотинга, и по какому ключу."""
    return BanwordEntry(raw=f"<spot:{keyword}>", normalized=keyword, lemma=keyword)


def _posterior_match(
    hit: Any,
    *,
    segment_start: float,
    segments: Sequence[dict],
    words: Sequence[dict],
    busy: Sequence[tuple[int, int]],
    range_in: TimeSpan,
    bounds: tuple[float, float] | None = None,
) -> ProfanityMatch | None:
    """Находка спотинга → ProfanityMatch с контекстом сегмента и соседних слов.

    Контекст (segment_id, конец предыдущего слова, начало следующего) берётся из
    транскрипта ПО ВРЕМЕНИ, а не по номеру сегмента: сегменты без посториоров в
    кэш не попадают, поэтому нумерация там своя. Без соседних слов окно мьюта
    считалось бы иначе, чем у текстовых матчей, — padding заезжал бы на чистую
    речь (см. `_compute_mute_window`).

    `bounds` — границы от `_aligned_hit_bounds` (секунды от начала сегмента).
    Есть — тайминги идут из forced_align, как у текстовых слов, и это видно в
    `timing_source`. Нет — остаются кадры эмиссии первой и последней буквы."""
    start = segment_start + (bounds[0] if bounds else hit.start)
    end = segment_start + (bounds[1] if bounds else hit.end)
    start_ms = TimeSpan.from_seconds(start).ms
    end_ms = TimeSpan.from_seconds(end).ms
    if any(min(b_end, end_ms) - max(b_start, start_ms) > 0 for b_start, b_end in busy):
        return None
    index, segment = _segment_at(segments, (start + end) / 2.0)
    near = [word for word in words if word.get("segment_index") == index]
    prev_word_end = max(
        (word["end"] for word in near if word["end"] <= start), default=None
    )
    next_word_start = min(
        (word["start"] for word in near if word["start"] >= end), default=None
    )
    # Что распознавание услышало на этом месте — то, что человек увидит в
    # списке мьютов («епанисир», а ключ уйдёт в banword). Ничего не услышало
    # (спотинг умеет и в паузу) — показываем сам ключ со знаком вопроса.
    # Пересечение проверяется без запаса: границы кандидата теперь из того же
    # forced_align, что и границы слов, и запас (стоял 150 мс под задержку
    # кадров эмиссии) начал приписывать кандидату соседнее слово.
    heard = " ".join(
        str(word["text"]).strip()
        for word in near
        if word["end"] > start and word["start"] < end
    ).strip()
    # Сегмента нет (пустой транскрипт) → segment_id тоже нет: выдумывать id
    # нельзя, по нему decisions.py решает, что мьюты из одной фразы (см.
    # `_group_by_segment`), и общий фальшивый id склеил бы чужие находки.
    word_obj = {
        "text": heard or f"{hit.keyword}?",
        "start": start,
        "end": end,
        "score": hit.score,
        "segment_id": segment.get("id", f"seg_{index:06d}") if segment else None,
        "segment_start": (segment or {}).get("start"),
        "segment_end": (segment or {}).get("end"),
        "timing_source": "posterior+align" if bounds else "posterior",
    }
    return _build_match(
        word_obj,
        hit.keyword,
        hit.keyword,
        _spot_entry(hit.keyword),
        "posterior",
        range_in,
        needs_review=True,
        prev_word_end=prev_word_end,
        next_word_start=next_word_start,
    )


def _segment_at(segments: Sequence[dict], moment: float) -> tuple[int, dict | None]:
    """Сегмент транскрипта, накрывающий момент времени, иначе ближайший.
    Пустой транскрипт → (-1, None): мьют всё равно соберётся, просто без
    segment_id и границ фразы."""
    nearest = (-1, None)
    best = None
    for index, segment in enumerate(segments):
        seg_start = segment.get("start")
        seg_end = segment.get("end")
        if seg_start is None or seg_end is None:
            continue
        if seg_start <= moment <= seg_end:
            return index, segment
        distance = min(abs(moment - seg_start), abs(moment - seg_end))
        if best is None or distance < best:
            best = distance
            nearest = (index, segment)
    return nearest


