"""LLM/NotebookLM интеграция для Фазы 3 — выбор ярких моментов.

Не онлайн-клиент, а offline pipeline: программа готовит пакет для загрузки в
notebooklm.google.com, пользователь руками получает JSON-ответ, программа
валидирует и мержит в decisions.json.

Причина такого дизайна: NotebookLM даёт 1M+ токенов контекста и хорошо держит
длинные тексты — по цене подписки, а не за токен. Полный стрим на 3–4 часа
влезает в один источник (лимит 500k слов ≈ 64 часа речи).
"""

from .exporter import build_notebooklm_package
from .importer import MergeError, merge_into_decisions, parse_response
from .model import Highlight, HighlightSet
from .segments import build_word_lines, chunk_by_word_limit, chunk_bounds

__all__ = [
    "build_notebooklm_package",
    "parse_response",
    "merge_into_decisions",
    "MergeError",
    "Highlight",
    "HighlightSet",
    "build_word_lines",
    "chunk_by_word_limit",
    "chunk_bounds",
]
