"""Точка входа собранного бэкенда: `twitch-cut-backend <команда>` == `twitch-cut <команда>`.

Electron запускает `twitch-cut-backend serve ...`, установщик — `prefetch` и
`accel`. freeze_support нужен на Windows: без него любой дочерний процесс
multiprocessing в собранном exe перезапускал бы всё приложение заново.
"""

import multiprocessing
import sys


def _force_utf8_stdio() -> None:
    """stdout/stderr — всегда UTF-8, что бы ни думала система.

    Собранный интерпретатор не читает PYTHONUTF8 из окружения, а на Windows
    stdout без него идёт в cp1251/cp1252 — и первый же rich-print русского
    текста или «→» валит процесс UnicodeEncodeError'ом. Electron и установщик
    читают вывод как UTF-8, так что это ровно то, что им нужно; errors=replace —
    чтобы символ, который всё же не вышло записать, не ронял бэкенд.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (OSError, ValueError):
                pass


if __name__ == "__main__":
    multiprocessing.freeze_support()
    _force_utf8_stdio()

    from twitch_cut.cli import app

    app(prog_name="twitch-cut")
