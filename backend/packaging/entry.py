"""Точка входа собранного бэкенда: `twitch-cut-backend <команда>` == `twitch-cut <команда>`.

Electron запускает `twitch-cut-backend serve ...`, установщик — `prefetch` и
`accel`. freeze_support нужен на Windows: без него любой дочерний процесс
multiprocessing в собранном exe перезапускал бы всё приложение заново.
"""

import multiprocessing

if __name__ == "__main__":
    multiprocessing.freeze_support()

    from twitch_cut.cli import app

    app(prog_name="twitch-cut")
