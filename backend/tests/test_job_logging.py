"""Фаза 1: лог джобы и лог сервера.

ЗАЧЕМ тест. У серверного процесса не было ни одного обработчика логов (uvicorn
настраивает только свои логгеры), поэтому диагностика классификатора —
«topic=неизвестно», «окно без вердикта» — не доходила никуда. Проверяем, что
записи действительно попадают в файл и что обвязка не течёт: хендлер снимается,
уровень логгера восстанавливается.
"""

from __future__ import annotations

import logging
from pathlib import Path

from twitch_cut.logging_setup import (
    PACKAGE_LOGGER,
    job_file_log,
    setup_server_logging,
)


def _reset_package_logger():
    pkg = logging.getLogger(PACKAGE_LOGGER)
    for h in list(pkg.handlers):
        pkg.removeHandler(h)
        h.close()
    pkg.setLevel(logging.NOTSET)
    return pkg


def test_job_log_captures_info_records(tmp_path: Path):
    pkg = _reset_package_logger()
    try:
        with job_file_log(tmp_path, name="offtopic_job7") as path:
            assert path == tmp_path / "logs" / "offtopic_job7.log"
            logging.getLogger("twitch_cut.llm_offtopic.classifier").info(
                "offtopic: %d/%d окон без вердикта", 13, 30
            )
        text = path.read_text(encoding="utf-8")
        assert "13/30 окон без вердикта" in text
        assert "twitch_cut.llm_offtopic.classifier" in text
    finally:
        _reset_package_logger()


def test_job_log_restores_logger_state(tmp_path: Path):
    pkg = _reset_package_logger()
    try:
        pkg.setLevel(logging.WARNING)
        handlers_before = list(pkg.handlers)

        with job_file_log(tmp_path, name="job") as path:
            # На время джобы уровень поднят до INFO — иначе диагностика не дойдёт.
            assert pkg.level == logging.INFO
            assert len(pkg.handlers) == len(handlers_before) + 1

        assert pkg.level == logging.WARNING
        assert pkg.handlers == handlers_before
        assert path.exists()
    finally:
        _reset_package_logger()


def test_job_log_does_not_break_job_when_dir_unwritable(tmp_path: Path):
    # workdir — существующий ФАЙЛ: mkdir внутри упадёт, джоба должна продолжиться.
    blocker = tmp_path / "not_a_dir"
    blocker.write_text("x", encoding="utf-8")
    _reset_package_logger()
    try:
        with job_file_log(blocker, name="job") as path:
            assert path is None
            logging.getLogger("twitch_cut.pipeline").info("работаю дальше")
    finally:
        _reset_package_logger()


def test_server_logging_is_idempotent(tmp_path: Path):
    pkg = _reset_package_logger()
    try:
        first = setup_server_logging(tmp_path)
        assert first == tmp_path / "backend.log"
        count = len(pkg.handlers)

        second = setup_server_logging(tmp_path)
        assert second == first
        assert len(pkg.handlers) == count  # второй вызов не удваивает хендлеры

        logging.getLogger("twitch_cut.server.runner").info("джоба стартовала")
        assert "джоба стартовала" in first.read_text(encoding="utf-8")
    finally:
        _reset_package_logger()
