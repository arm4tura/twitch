from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
import re

_TIMECODE_RE = re.compile(
    r"^(?:(?P<h>\d+):)?(?P<m>\d{1,2}):(?P<s>\d{1,2})(?:[\.,](?P<ms>\d{1,3}))?$"
)


@dataclass(frozen=True, order=True)
class TimeSpan:
    """Milliseconds-based duration to avoid floating point drift."""

    ms: int

    def __post_init__(self) -> None:
        if self.ms < 0:
            raise ValueError("TimeSpan cannot be negative")

    @classmethod
    def from_seconds(cls, value: float | int | Decimal) -> "TimeSpan":
        decimal_value = Decimal(str(value))
        ms = (decimal_value * Decimal(1000)).quantize(Decimal("1"), rounding=ROUND_HALF_UP)
        return cls(int(ms))

    def to_seconds(self) -> float:
        return self.ms / 1000.0

    def __add__(self, other: "TimeSpan") -> "TimeSpan":
        return TimeSpan(self.ms + other.ms)

    def __sub__(self, other: "TimeSpan") -> "TimeSpan":
        if self.ms < other.ms:
            raise ValueError("TimeSpan subtraction would be negative")
        return TimeSpan(self.ms - other.ms)

    def format(self) -> str:
        hours, rem = divmod(self.ms, 3_600_000)
        minutes, rem = divmod(rem, 60_000)
        seconds, millis = divmod(rem, 1000)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{millis:03d}"

    def __str__(self) -> str:
        return self.format()


def parse_timecode(value: str | int | float | Decimal | TimeSpan) -> TimeSpan:
    if isinstance(value, TimeSpan):
        return value
    if isinstance(value, (int, float, Decimal)):
        return TimeSpan.from_seconds(value)

    raw = str(value).strip()
    if not raw:
        raise ValueError("Empty timecode")

    if ":" not in raw:
        try:
            return TimeSpan.from_seconds(Decimal(raw.replace(",", ".")))
        except Exception as exc:
            raise ValueError(f"Invalid seconds value: {value!r}") from exc

    match = _TIMECODE_RE.match(raw)
    if not match:
        raise ValueError(f"Invalid timecode: {value!r}")

    hours = int(match.group("h") or 0)
    minutes = int(match.group("m"))
    seconds = int(match.group("s"))
    millis_raw = match.group("ms") or "0"
    millis = int(millis_raw.ljust(3, "0"))

    if minutes >= 60 or seconds >= 60:
        raise ValueError(f"Invalid timecode field range: {value!r}")

    return TimeSpan(hours * 3_600_000 + minutes * 60_000 + seconds * 1000 + millis)


def format_timecode(value: TimeSpan | int | float | Decimal | str) -> str:
    return parse_timecode(value).format()


def seconds_float(value: TimeSpan | int | float | Decimal | str) -> float:
    return parse_timecode(value).to_seconds()
