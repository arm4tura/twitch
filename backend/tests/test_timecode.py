from decimal import Decimal

import pytest

from twitch_cut.timecode import TimeSpan, format_timecode, parse_timecode, seconds_float


def test_parse_full_timecode_with_millis():
    ts = parse_timecode("01:02:03.456")
    assert ts.ms == 3_723_456
    assert ts.format() == "01:02:03.456"


def test_parse_mm_ss_and_seconds():
    assert parse_timecode("02:03").format() == "00:02:03.000"
    assert parse_timecode("12.345").ms == 12_345
    assert parse_timecode(Decimal("1.235")).ms == 1_235


def test_offset_math_is_millisecond_based():
    base = parse_timecode("00:12:30")
    local = TimeSpan.from_seconds(4.3)
    assert (base + local).format() == "00:12:34.300"
    assert seconds_float(base + local) == 754.3


def test_invalid_timecodes():
    with pytest.raises(ValueError):
        parse_timecode("")
    with pytest.raises(ValueError):
        parse_timecode("00:99:00")
    with pytest.raises(ValueError):
        format_timecode("bad")
