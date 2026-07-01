from __future__ import annotations

import logging
from pathlib import Path

from rich.logging import RichHandler


def setup_logging(workdir: Path | None = None, verbose: bool = False) -> logging.Logger:
    level = logging.DEBUG if verbose else logging.INFO
    handlers: list[logging.Handler] = [RichHandler(rich_tracebacks=True, show_time=True)]

    if workdir is not None:
        log_dir = workdir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_dir / "process.log", encoding="utf-8")
        file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        handlers.append(file_handler)

    logging.basicConfig(level=level, format="%(message)s", handlers=handlers, force=True)
    return logging.getLogger("twitch_cut")
