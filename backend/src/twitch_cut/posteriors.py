"""posteriors.py — хранение посториоров CTC рядом с транскриптом.

ЗАЧЕМ ХРАНИТЬ. Поиск мата по посториорам (`keyword_spotting`) работает не с
текстом, а с log-probs модели, а они живут ровно один миг внутри декода. Если
считать спотинг в том же проходе, то любая правка порога или списка корней
означала бы ПОВТОРНОЕ распознавание — часы на длинном стриме. Поэтому log-probs
кладём в кэш рядом с транскриптом: спотинг становится отдельной дешёвой стадией,
которую можно крутить сколько нужно.

ФОРМАТ. Один `.npz` на прогон:
  probs   float16 [сумма кадров, C]  — log-probs всех VAD-сегментов подряд
  offsets int64   [сегментов + 1]    — границы сегментов в этом массиве
  starts  float32 [сегментов]        — АБСОЛЮТНОЕ начало сегмента, секунды
  shifts  float32 [сегментов]        — секунд на кадр (у каждого сегмента своё)
  vocab   <U8     [C - 1]            — буквы словаря, индекс = id токена
  blank   int64                      — id blank-токена

float16 выбран сознательно: log-prob лежит в диапазоне примерно [-30, 0], где
шаг float16 около 0.01 — это меньше 1% по вероятности, то есть на порог спотинга
не влияет, зато файл вдвое меньше. Пятнадцать минут речи ≈ 1.5 МБ.

Абсолютное начало сегмента хранится здесь, потому что сам декод его не знает:
границы сегментов считает VAD внутри `transcribe_longform`, а `_decode` видит
только батч. Порядок сегментов в файле — тот же, что в транскрипте.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Any, Iterator, Sequence

logger = logging.getLogger(__name__)


@dataclass
class SegmentPosteriors:
    """log-probs одного VAD-сегмента плюс всё, чтобы перевести кадры в секунды."""

    index: int
    start: float
    """Абсолютное начало сегмента в локальной шкале (как `start` у сегмента
    транскрипта), секунды."""
    frame_shift: float
    log_probs: Any
    """[T, C] float32, numpy."""


@dataclass
class Posteriors:
    """Прочитанный файл посториоров: словарь + сегменты."""

    vocab: list[str]
    blank_id: int
    segments: list[SegmentPosteriors]

    def char_to_id(self, char: str) -> int | None:
        """Буква → id токена. Нет в словаре → None (искать её нечем)."""
        try:
            return self._index[char]
        except KeyError:
            return None

    def __post_init__(self) -> None:
        self._index = {char: i for i, char in enumerate(self.vocab)}

    def __iter__(self) -> Iterator[SegmentPosteriors]:
        return iter(self.segments)


def posteriors_path(workdir: Path, transcript_key: str) -> Path:
    """Где лежат посториоры прогона. Ключ — тот же, что у транскрипта: файлы
    парные, и разъехаться они не должны (стадия спотинга ищет посториоры по
    ключу транскрипта, ничего не зная о том, как их писали)."""
    return workdir / "cache" / f"posteriors_{transcript_key}.npz"


def write_posteriors(
    path: Path,
    chunks: Sequence[tuple[Any, float]],
    *,
    starts: Sequence[float],
    vocab: Sequence[str],
    blank_id: int,
) -> Path | None:
    """Сложить посториоры сегментов в один `.npz`.

    `chunks` — по сегменту: (log_probs [T, C] numpy, frame_shift),
    `starts` — абсолютные начала сегментов в том же порядке. Длины не сошлись →
    не пишем ничего: посториоры без времени бесполезны, а врать нельзя.

    Возвращает путь или None, если записать не удалось (нет numpy, нет места).
    Спотинг тогда просто не запустится — распознавание из-за этого не рушим.
    """
    if len(chunks) != len(starts):
        logger.warning(
            "посториоры: %d сегментов против %d начал — не сохраняю",
            len(chunks),
            len(starts),
        )
        return None
    try:
        import numpy as np

        arrays = [_as_float16(chunk, np) for chunk, _ in chunks]
        arrays = [a for a in arrays if a is not None]
        if len(arrays) != len(chunks) or not arrays:
            return None
        offsets = np.zeros(len(arrays) + 1, dtype=np.int64)
        for i, array in enumerate(arrays):
            offsets[i + 1] = offsets[i] + array.shape[0]
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            probs=np.concatenate(arrays, axis=0),
            offsets=offsets,
            starts=np.asarray(starts, dtype=np.float32),
            shifts=np.asarray([shift for _, shift in chunks], dtype=np.float32),
            vocab=np.asarray(list(vocab), dtype="<U8"),
            blank=np.asarray(int(blank_id), dtype=np.int64),
        )
        return path
    except Exception as exc:  # noqa: BLE001 — диск/numpy; распознавание важнее
        logger.warning("посториоры не сохранены (%s)", exc)
        return None


def to_float16(log_probs: Any) -> Any | None:
    """[T, C] log-probs → numpy float16. Не вышло → None.

    Приводить к float16 стоит СРАЗУ на выходе декода: посториоры целого стрима
    в float32 — это гигабайты в памяти, а на диск они всё равно лягут половинкой.
    """
    try:
        import numpy as np
    except Exception as exc:  # noqa: BLE001 — numpy есть всегда, но не рушим ASR
        logger.warning("посториоры: нет numpy (%s)", exc)
        return None
    return _as_float16(log_probs, np)


def _as_float16(log_probs: Any, np: Any) -> Any | None:
    try:
        array = np.asarray(log_probs, dtype=np.float16)
        return array if array.ndim == 2 and array.shape[0] > 0 else None
    except Exception as exc:  # noqa: BLE001
        logger.warning("посториоры: сегмент не привести к float16 (%s)", exc)
        return None


def read_posteriors(path: Path) -> Posteriors | None:
    """Прочитать `.npz` обратно. Файла нет или он битый → None.

    log-probs отдаём как float32: спотинг считает суммы, и складывать float16
    незачем — точность внутри DP дешевле, чем на диске.
    """
    if not path.exists():
        return None
    try:
        import numpy as np

        with np.load(path, allow_pickle=False) as data:
            probs = data["probs"]
            offsets = data["offsets"]
            starts = data["starts"]
            shifts = data["shifts"]
            vocab = [str(char) for char in data["vocab"]]
            blank_id = int(data["blank"])
        n = len(offsets) - 1
        if n <= 0 or len(starts) != n or len(shifts) != n:
            logger.warning("посториоры %s: размеры не сходятся", path.name)
            return None
        segments = [
            SegmentPosteriors(
                index=i,
                start=float(starts[i]),
                frame_shift=float(shifts[i]),
                log_probs=_to_float32(probs[int(offsets[i]) : int(offsets[i + 1])], np),
            )
            for i in range(n)
        ]
        return Posteriors(vocab=vocab, blank_id=blank_id, segments=segments)
    except Exception as exc:  # noqa: BLE001 — битый кэш = «спотинга не будет»
        logger.warning("посториоры %s не читаются (%s)", path.name, exc)
        return None


def _to_float32(array: Any, np: Any) -> Any:
    """numpy float16 → numpy float32."""
    return np.asarray(array, dtype=np.float32)
