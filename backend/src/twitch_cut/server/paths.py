"""Единая логика вычисления рекомендованных путей.

Используется и HTTP-эндпоинтом `/paths/suggest_workdir` (UI показывает путь
в поле workdir расширенного режима), и runner'ом (простой режим, когда UI
вообще не присылает workdir — бэкенд генерит его сам).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

from ..config import (
    DATA_ROOT,
    default_banwords_path,
    default_compute_type,
    default_device,
)


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


@dataclass
class JobDefaults:
    """Разрешённые пути и опции устройства для одного прогона пайплайна."""

    workdir: Path
    original: Path
    banwords: Path
    decisions: Path
    vegas: Path
    device: str
    compute_type: str


def resolve_job_defaults(
    *,
    stream: str | Path,
    original: Optional[str | Path] = None,
    banwords: Optional[str | Path] = None,
    workdir: Optional[str | Path] = None,
    decisions: Optional[str | Path] = None,
    vegas: Optional[str | Path] = None,
    device: Optional[str] = None,
    compute_type: Optional[str] = None,
) -> JobDefaults:
    """Единый резолв дефолтов «простого режима» для CLI и HTTP-runner'а.

    Раньше эта логика была скопирована в `runner._run_process_sync` и в
    `cli.process` и успела разойтись (CLI застрял на `work/job_001` + cuda,
    runner ушёл на `suggested_workdir()` + авто-device). Держим в одном месте,
    чтобы дефолты обоих адаптеров не расходились.

    Правила:
      - workdir      → уникальная папка `suggested_workdir(stream)`
      - original     → сам stream (реакция == запись, если второго видео нет)
      - banwords     → встроенный словарь `default_banwords_path()`
      - decisions    → `<workdir>/decisions.json`
      - vegas        → `<workdir>/mutes.cs`
      - device       → авто-детект `default_device()` (CPU без NVIDIA)
      - compute_type → авто-детект `default_compute_type()`
    """
    resolved_workdir = Path(workdir) if workdir else suggested_workdir(str(stream))
    return JobDefaults(
        workdir=resolved_workdir,
        original=Path(original) if original else Path(stream),
        banwords=Path(banwords) if banwords else default_banwords_path(),
        decisions=Path(decisions) if decisions else resolved_workdir / "decisions.json",
        vegas=Path(vegas) if vegas else resolved_workdir / "mutes.cs",
        device=device or default_device(),
        compute_type=compute_type or default_compute_type(),
    )
