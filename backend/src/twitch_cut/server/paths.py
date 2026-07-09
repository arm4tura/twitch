"""Единая логика вычисления рекомендованных путей.

Используется и HTTP-эндпоинтом `/paths/suggest_workdir` (UI показывает путь
в поле workdir расширенного режима), и runner'ом (простой режим, когда UI
вообще не присылает workdir — бэкенд генерит его сам).
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from ..config import DATA_ROOT


def suggested_workdir(stream: str | None = None) -> Path:
    """Рекомендованный workdir для нового прогона.

    Формат: <DATA_ROOT>/work/{basename}_{yyyymmdd_hhmm}/

    Кладём внутрь общей папки work/ рядом с исходниками проекта (в dev —
    корень репо, в упакованном приложении — writable data-каталог из
    TWITCH_CUT_DATA_DIR). Так все прогоны лежат в одном предсказуемом месте,
    а не разбросаны по домашней папке. Минута в имени гарантирует уникальность
    — два запуска подряд не перезаписывают decisions.json друг друга. Папку НЕ
    создаёт (создастся при первом write в пайплайне). Если stream не задан —
    basename 'job'.
    """
    raw = Path(stream).stem if stream else "job"
    # Вычищаем только то, что ломает пути; пробелы/кириллицу ФС спокойно ест.
    safe = "".join(c if c not in '<>:"/\\|?*' else "_" for c in raw).strip() or "job"
    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    return DATA_ROOT / "work" / f"{safe}_{stamp}"
