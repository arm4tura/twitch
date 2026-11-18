from __future__ import annotations

import contextlib
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Iterator, Optional

from rich.logging import RichHandler

# Логгер пакета. Все модули берут logging.getLogger(__name__), то есть
# 'twitch_cut.<...>' — значит хендлер, повешенный сюда, ловит весь backend и
# ничего лишнего (uvicorn/httpx/torch сюда не пишут).
PACKAGE_LOGGER = "twitch_cut"

_FILE_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"

# Имена наших хендлеров — по ним setup_server_logging остаётся идемпотентным.
_SERVER_FILE_HANDLER = "twitch_cut.server.file"
_SERVER_STREAM_HANDLER = "twitch_cut.server.stream"


def setup_logging(workdir: Path | None = None, verbose: bool = False) -> logging.Logger:
    """Логирование для CLI: rich в консоль + опционально файл в workdir/logs."""
    level = logging.DEBUG if verbose else logging.INFO
    handlers: list[logging.Handler] = [RichHandler(rich_tracebacks=True, show_time=True)]

    if workdir is not None:
        log_dir = workdir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_dir / "process.log", encoding="utf-8")
        file_handler.setFormatter(logging.Formatter(_FILE_FORMAT))
        handlers.append(file_handler)

    logging.basicConfig(level=level, format="%(message)s", handlers=handlers, force=True)
    return logging.getLogger(PACKAGE_LOGGER)


def setup_server_logging(log_dir: Path, *, verbose: bool = False) -> Optional[Path]:
    """Логирование backend-процесса (`serve`): файл в каталоге настроек + stderr.

    ЗАЧЕМ. Сервер не звал setup_logging вообще, а uvicorn настраивает только
    свои логгеры — у root-логгера обработчиков не было, поэтому INFO/WARNING из
    twitch_cut никуда не доходили. Практический итог: классификатор оффтопа
    честно писал «topic=неизвестно» и «окно без вердикта», а увидеть это было
    невозможно — отсюда и диагноз «прод молча возвращает 2 сегмента».

    Файл лежит там же, где settings.json — на этот каталог уже указывает кнопка
    UI «Открыть папку логов» (GET /settings/logs_dir).

    Идемпотентно: повторный вызов не добавляет второй хендлер.

    :return: путь к лог-файлу; None если каталог недоступен для записи (тогда
        остаётся только stderr — сервер из-за лога падать не должен).
    """
    level = logging.DEBUG if verbose else logging.INFO
    pkg = logging.getLogger(PACKAGE_LOGGER)
    pkg.setLevel(level)

    existing = {h.get_name() for h in pkg.handlers}

    if _SERVER_STREAM_HANDLER not in existing:
        stream = logging.StreamHandler()
        stream.set_name(_SERVER_STREAM_HANDLER)
        stream.setFormatter(logging.Formatter(_FILE_FORMAT))
        pkg.addHandler(stream)

    if _SERVER_FILE_HANDLER in existing:
        for h in pkg.handlers:
            if h.get_name() == _SERVER_FILE_HANDLER:
                return Path(getattr(h, "baseFilename", ""))
        return None

    path = log_dir / "backend.log"
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        # Ротация: процесс живёт всё время работы приложения, лог без предела
        # рос бы вместе с ним.
        file_handler = RotatingFileHandler(
            path, maxBytes=5_000_000, backupCount=3, encoding="utf-8"
        )
    except OSError as exc:
        pkg.warning("не смог открыть лог-файл %s (%s) — пишу только в stderr", path, exc)
        return None
    file_handler.set_name(_SERVER_FILE_HANDLER)
    file_handler.setFormatter(logging.Formatter(_FILE_FORMAT))
    pkg.addHandler(file_handler)
    return path


@contextlib.contextmanager
def job_file_log(
    workdir: Path, *, name: str, verbose: bool = False
) -> Iterator[Optional[Path]]:
    """Файловый лог одной джобы: `<workdir>/logs/<name>.log` на время джобы.

    ЗАЧЕМ отдельно от серверного лога: лог джобы должен лежать рядом с её
    результатами (decisions.json, cache/), чтобы разбор «почему получилось
    именно это» не требовал искать общий backend.log и вырезать из него нужный
    интервал.

    ОГРАНИЧЕНИЕ. Хендлер цепляется к логгеру пакета, а не к конкретной джобе:
    при двух одновременных джобах записи попадут в оба файла. Цена сознательная
    — альтернатива это протаскивание logger-объекта через все стадии пайплайна.

    Не роняем джобу, если лог открыть не удалось: yield-им None.
    """
    level = logging.DEBUG if verbose else logging.INFO
    pkg = logging.getLogger(PACKAGE_LOGGER)
    log_dir = workdir / "logs"
    path = log_dir / f"{name}.log"

    handler: Optional[logging.Handler] = None
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(path, encoding="utf-8")
    except OSError as exc:
        pkg.warning("не смог открыть лог джобы %s (%s) — продолжаю без него", path, exc)

    if handler is None:
        yield None
        return

    handler.setFormatter(logging.Formatter(_FILE_FORMAT))
    handler.setLevel(level)
    prev_level = pkg.level
    # NOTSET (0) означает «наследовать от root», а у root в серверном процессе
    # это WARNING — INFO-записи не дошли бы до нашего хендлера.
    if prev_level == logging.NOTSET or prev_level > level:
        pkg.setLevel(level)
    pkg.addHandler(handler)
    try:
        yield path
    finally:
        pkg.removeHandler(handler)
        handler.close()
        pkg.setLevel(prev_level)
