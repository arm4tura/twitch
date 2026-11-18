"""Самопроверка словаря корней по 5.1 млн словоформ pymorphy3.

Зачем. Списки CORE_ROOTS / SPICY_ROOTS / PREFIX_ONLY_ROOTS / INNOCENT_SUBSTRINGS
в profanity.py решают, что программа замьютит. Ошибка в одном корне («манда»
внутри слова) мьютит слово «команда» на каждом стриме, и заметить это можно
только слушая. Скрипт отвечает машинно: какие НЕВИННЫЕ слова русского языка
ловит текущий словарь.

Запуск:
    python tools/roots_selfcheck.py            # сводка + список ложных
    python tools/roots_selfcheck.py --verbose   # ещё и все формы по корню

Что делать с выводом: если в списке появилось обычное слово — либо корню нужна
политика «только с начала слова» (PREFIX_ONLY_ROOTS), либо в
INNOCENT_SUBSTRINGS не хватает подстроки-омографа. Ожидаемые совпадения
(сам мат и редкие иностранные имена) перечислены в ACCEPTED ниже.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twitch_cut.profanity import (  # noqa: E402
    CORE_ROOTS,
    PREFIX_ONLY_ROOTS,
    SPICY_ROOTS,
    roots_in,
)

ACCEPTED: tuple[str, ...] = (
    # иностранные имена и патронимы из словаря pymorphy3: «хуе» внутри них
    # настоящее, а щит на них ставить нельзя — «ахуе»/«ахуй» это подстрока
    # живого мата (нахуй, ахуенно).
    "хуевич", "хуевна", "хуевен", "айхуе", "айхуа", "ахуй", "ахуи", "ахуя",
    "ахуе", "ахую", "нхулна", "забихул", "хуан", "хуарес", "хулио",
    "алемайехуев", "элиахуев",
    # погребище: единственная форма на «ебищ», мат «уебище» ловится корнем «уеб»
    "погребищ",
    # словарный мусор и фамилии, которых в живой речи не бывает: Аговна,
    # Уговна, Яговна, Гуговна — щит на них добавлять дороже, чем принять.
    "аговн", "уговн", "яговн", "гуговн", "гоговн", "агговн", "посучкаст",
    # это настоящее оскорбление, отчёт должен его показывать
    "черножоп",
    # отчество на -говна (Олеговна, Ареговна, Хьюговна): «говн» — spicy-корень,
    # детектор его не мьютит, а щит ставить нельзя — «говна» живое слово.
    "еговн", "рговн", "нговн", "юговн", "йговн", "иговн", "оговн", "ыговн",
    # грузинские топонимы Говхери / Сачхере, имя Пульхерия
    "говхер", "сачхер", "пульхер",
    # это МАТ, найденный внутри слова, — ровно то, ради чего инфиксный поиск и
    # заведён. В списке, чтобы «ложных нет» означало «ничего нового не появилось».
    "охуен", "охуени", "нехуй", "похер", "подерьмов",
    # редкие книжные слова с «хер» внутри: штрейкбрехер, Вормсбехер,
    # Влахернская, Секешфехервар. Корень spicy — детектор его не мьютит.
    "брехер", "бехер", "хернск", "фехерв",
)
"""Совпадения, которые мы принимаем осознанно: либо это мат, либо имя, которое
в русской речи стримера не встречается, а щит на него сломал бы живой мат."""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    import pymorphy3

    morph = pymorphy3.MorphAnalyzer()
    roots = tuple(CORE_ROOTS) + tuple(SPICY_ROOTS)
    hits: dict[str, set[str]] = defaultdict(set)
    total = 0
    for form in morph.dictionary.words.iterkeys(""):
        total += 1
        norm = form.lower().replace("ё", "е")
        if not norm.isalpha():
            continue
        for root in roots_in(norm, roots):
            hits[root].add(norm)

    print(f"словарь: {total} форм, корней: {len(roots)}")
    suspicious: dict[str, set[str]] = {}
    for root in roots:
        forms = hits.get(root, set())
        left = {f for f in forms if not f.startswith(root) and not _accepted(f)}
        policy = "начало" if root in PREFIX_ONLY_ROOTS else "любое место"
        mark = "  ← ПРОВЕРЬ" if left else ""
        print(f"{root:8s} [{policy:11s}] форм: {len(forms):5d}, из них не с корня: {len(left)}{mark}")
        if left:
            suspicious[root] = left
        if args.verbose and forms:
            print("      " + ", ".join(sorted(forms, key=len)[:40]))

    if not suspicious:
        print("\nЛОЖНЫХ НЕТ: ни одно словарное слово не поймано корнем не с начала.")
        return 0
    print("\nВОЗМОЖНЫЕ ЛОЖНЫЕ (корень найден ВНУТРИ слова):")
    for root, forms in suspicious.items():
        print(f"  {root}: {', '.join(sorted(forms, key=len)[:30])}")
    return 1


def _accepted(form: str) -> bool:
    return any(token in form for token in ACCEPTED)


if __name__ == "__main__":
    raise SystemExit(main())
