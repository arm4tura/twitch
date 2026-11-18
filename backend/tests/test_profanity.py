import pytest

from twitch_cut.profanity import (
    CORE_ROOTS,
    PREFIX_ONLY_ROOTS,
    RussianNormalizer,
    detect_profanity,
    load_banwords,
    match_root,
    roots_in,
)
from twitch_cut.timecode import parse_timecode

# Слова, которые ОБЯЗАН ловить корневой матчер, и корень, которым он их ловит.
# Половина списка — инфиксный мат: раньше поиск смотрел только на начало слова,
# и всё это проходило мимо детектора (правки человека это подтвердили).
MUST_MATCH: tuple[tuple[str, str], ...] = (
    ("хуйня", "хуй"), ("нихуя", "хуя"), ("дохуя", "хуя"), ("похуй", "хуй"),
    ("ахуенное", "хуе"), ("нехуевая", "хуе"), ("охуеть", "хуе"),
    ("пиздец", "пизд"), ("распиздяй", "пизд"),
    ("заебись", "заеб"), ("заебал", "заеб"), ("наебал", "наеб"),
    ("съебался", "съеб"), ("уебище", "уеб"), ("подъебка", "подъеб"),
    # приставочные формы, которых не было в первом наборе: «еб» ищется только с
    # начала слова, поэтому без своего корня они уезжали мимо детектора, а мьют
    # вставал на соседнее слово (замер на vod_2847020280, 2026-08-18).
    ("выебала", "выеб"), ("выебывается", "выеб"), ("приебался", "приеб"),
    ("переебал", "перееб"), ("изъебался", "изъеб"),
    ("ебать", "еб"), ("ебанный", "еб"),
    ("блять", "бля"), ("бля", "бля"),
    ("гандоны", "гандон"), ("пидорас", "пидор"), ("мудак", "мудак"),
    ("залупа", "залуп"), ("дрочить", "дроч"), ("шлюха", "шлюх"),
)

# Обычные слова русского языка: ни один корень не должен в них найтись. Это
# те самые ложные, которыми инфиксный поиск ломает стрим, если политика корня
# выбрана неверно (см. tools/roots_selfcheck.py — та же проверка, но по всем
# 5.1 млн словоформ pymorphy3).
MUST_NOT_MATCH: tuple[str, ...] = (
    "команда", "команды", "мандарин", "мандат",
    "неплохую", "сухую", "тихую", "глухую", "страхуй", "психует", "лопоухую",
    "хлебать", "судебный", "щебень", "ребенок", "погребать", "требовать",
    "рубля", "сабля", "дубляж", "скобля", "оскорблять", "потребляет",
    "скипидар", "лапидарный", "талмуд", "махмуд", "бермуды",
    "богохульник", "хулиган", "барсука", "сукно", "лохматый", "молох",
    "серебро", "лебедь", "жеребец", "хребет", "небеса", "дебет",
    "приемник", "приезд", "переезд", "изъездил", "выезд", "приемлемый",
)


def test_roots_matcher_finds_profanity_including_infix():
    for word, root in MUST_MATCH:
        assert match_root(word) == root, f"{word} → ожидался корень {root}"


def test_roots_matcher_leaves_innocent_words_alone():
    for word in MUST_NOT_MATCH:
        assert roots_in(word) == [], f"{word} — ложное срабатывание"


def test_prefix_only_roots_are_the_ones_with_innocent_homographs():
    """Корень ищется внутри слова, только если по словарю у него нет ложных.
    Здесь фиксируем сам контракт: у prefix-only корня инфикс не срабатывает."""
    assert PREFIX_ONLY_ROOTS <= set(CORE_ROOTS)
    for root in PREFIX_ONLY_ROOTS:
        # «ка…ка» — нейтральная обёртка: она не склеивается с приставочными
        # корнями (в отличие от «про…», где «про»+«еб» даёт настоящий корень).
        assert roots_in(f"ка{root}ка") == [], root
        assert root in roots_in(root + "а"), root


def test_load_banwords_ignores_comments(tmp_path):
    path = tmp_path / "banwords.txt"
    path.write_text("# comment\n\nБЛИН # inline\n", encoding="utf-8")
    normalizer = RussianNormalizer()
    entries = load_banwords(path, normalizer)
    assert len(entries) == 1
    assert entries[0].normalized == "блин"


def test_masked_banword_entries_do_not_become_one_letter_words(tmp_path):
    """«б***» в словаре после нормализации — это «б», и surface-матч начинал
    мьютить ЛЮБОЕ одиночное «б» из расшифровки без флага проверки (свежий
    прогон vod_2847020280: два окна, человек снял оба). Такие записи не
    грузятся, а обрубок остаётся делом шага #6 — мягким матчем на проверку."""
    path = tmp_path / "banwords.txt"
    path.write_text("***\nб***ь\nб***\nблять\n", encoding="utf-8")
    normalizer = RussianNormalizer()
    entries = load_banwords(path, normalizer)
    assert [e.normalized for e in entries] == ["блять"]

    transcript = {
        "segments": [
            {
                "id": "seg_1",
                "words": [
                    {"word": "хотя", "start": 1.0, "end": 1.2, "score": 0.9},
                    {"word": "б", "start": 1.4, "end": 1.6, "score": 0.9},
                    {"word": "постеснялись", "start": 1.8, "end": 2.4, "score": 0.9},
                ],
            }
        ]
    }
    matches = detect_profanity(transcript, entries, parse_timecode("00:00:00"), normalizer)
    assert [m.word for m in matches] == ["б"]
    assert matches[0].match_type == "short_fragment"
    assert matches[0].needs_review


def test_detect_surface_match(tmp_path):
    path = tmp_path / "banwords.txt"
    path.write_text("блин\n", encoding="utf-8")
    normalizer = RussianNormalizer()
    entries = load_banwords(path, normalizer)
    transcript = {
        "segments": [
            {
                "id": "seg_1",
                "start": 1.0,
                "end": 2.0,
                "words": [{"word": "Блин!", "start": 1.1, "end": 1.4, "score": 0.9}],
            }
        ]
    }
    matches = detect_profanity(transcript, entries, parse_timecode("00:01:00"), normalizer)
    assert len(matches) == 1
    assert matches[0].match_type == "surface"
    assert matches[0].local_start_ms == 1100
    assert matches[0].stream_start_ms == 61_100


def test_detect_lemma_match_when_pymorphy_available(tmp_path):
    normalizer = RussianNormalizer()
    if normalizer._morph is None:
        pytest.skip("pymorphy3 is not installed in this environment")
    path = tmp_path / "banwords.txt"
    path.write_text("плохой\n", encoding="utf-8")
    entries = load_banwords(path, normalizer)
    transcript = {
        "segments": [
            {"words": [{"word": "плохого", "start": 65.35, "end": 65.8, "score": 0.94}]}
        ]
    }
    matches = detect_profanity(transcript, entries, parse_timecode("00:01:00"), normalizer)
    assert len(matches) == 1
    assert matches[0].match_type == "lemma"


def test_detect_infix_profanity_without_any_banword(tmp_path):
    """Сквозная проверка того, ради чего затевался единый матчер: «нихуя» и
    «ахуенное» ловятся корнем внутри слова, без записи в banwords.txt, а
    «команда» рядом остаётся чистой."""
    path = tmp_path / "banwords.txt"
    path.write_text("", encoding="utf-8")
    normalizer = RussianNormalizer()
    entries = load_banwords(path, normalizer)
    transcript = {
        "segments": [
            {
                "id": "seg_1",
                "words": [
                    {"word": "команда", "start": 1.0, "end": 1.3, "score": 0.9},
                    {"word": "нихуя", "start": 1.4, "end": 1.8, "score": 0.9},
                    {"word": "ахуенное", "start": 2.0, "end": 2.5, "score": 0.9},
                ],
            }
        ]
    }
    matches = detect_profanity(transcript, entries, parse_timecode("00:00:00"), normalizer)
    assert [m.word for m in matches] == ["нихуя", "ахуенное"]
    assert {m.match_type for m in matches} == {"stem"}
    assert [m.banword for m in matches] == ["<stem:хуя>", "<stem:хуе>"]


def _short_fragment_matches(tmp_path, words: list[str]):
    """Шаг #6 (короткие подозрительные фрагменты) на списке слов по 0.2 с."""
    path = tmp_path / "banwords.txt"
    path.write_text("", encoding="utf-8")
    normalizer = RussianNormalizer()
    entries = load_banwords(path, normalizer)
    transcript = {
        "segments": [
            {
                "id": "seg_1",
                "words": [
                    {"word": w, "start": 1.0 + 0.4 * i, "end": 1.2 + 0.4 * i, "score": 0.9}
                    for i, w in enumerate(words)
                ],
            }
        ]
    }
    return detect_profanity(transcript, entries, parse_timecode("00:00:00"), normalizer)


def test_glue_is_skipped_when_neighbour_is_profane_on_its_own(tmp_path):
    """«ну ебать»: сосед — мат сам по себе, склейка не нужна и вредна.

    Раньше выходило ДВА матча: glued «нуебать» (корень <stem:уеб> нашёлся
    поперёк границы слов) и surface/stem «ебать». Группировка их не сливает —
    соседство требует word_start_index == prev.word_end_index + 1, а здесь
    индексы совпадают, — и в decisions.json получались два вложенных мьюта с
    общим концом. Замер на vod_2847020280_20260817_2330: 7 таких пар из 59 окон,
    и первое слово склейки («ну») мьютилось зря."""
    matches = _short_fragment_matches(tmp_path, ["ну", "ебать"])
    assert [m.word for m in matches] == ["ебать"]
    assert matches[0].match_type != "glued"


def test_glue_still_recovers_a_word_split_by_asr(tmp_path):
    """Обратная сторона: там, где сосед САМ по себе не мат, склейка остаётся —
    за этим она и написана. «бл»+«ять» по отдельности не дают ни одного твёрдого
    матча (только мягкий фрагмент «бл»), вместе — «блять»."""
    matches = _short_fragment_matches(tmp_path, ["бл", "ять"])
    assert [m.word for m in matches] == ["блять"]
    assert matches[0].match_type == "glued"
    assert not matches[0].needs_review


def test_word_eaten_by_glue_does_not_produce_a_second_match(tmp_path):
    """Вторая половина склейки не должна матчиться ещё и сама.

    «бл»+«ять*» → glued «блять*». Слово «ять*» съедено склейкой, но шаг #3
    (цензура) ловил его ещё и отдельным мягким матчем: проверка `used_glued`
    стояла в начале шага #5, то есть шаги #1–#4 успевали отработать на уже
    съеденном слове. Пара выбрана именно цензурная: после правки «сосед — мат
    сам по себе» твёрдые пути (#1/#2/#4) до этого места уже не доходят, а
    цензура и короткий фрагмент — доходят."""
    matches = _short_fragment_matches(tmp_path, ["бл", "ять*"])
    assert [m.word for m in matches] == ["блять*"]
    assert matches[0].match_type == "glued"


def test_laughter_is_not_a_suspicious_short_fragment(tmp_path):
    """«ха-ха» — не мат, а смех: шаг #6 ловил его только потому, что 'х' стоит
    в _SUSPICIOUS_SHORT_PREFIXES. На vod_2847020280 это дало два мьюта на
    реальном звуке, которые человек снял руками."""
    assert _short_fragment_matches(tmp_path, ["Ха", "ха", "хе", "хи", "хм"]) == []


def test_suspicious_short_fragment_still_flagged_for_review(tmp_path):
    """Обратная сторона whitelist: обрубки мата шаг #6 всё ещё ловит — мягким
    матчем (needs_review), а не мьютом по умолчанию."""
    matches = _short_fragment_matches(tmp_path, ["бл", "ь"])
    assert [m.word for m in matches] == ["бл", "ь"]
    assert {m.match_type for m in matches} == {"short_fragment"}
    assert all(m.needs_review for m in matches)
