from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from magnetic_app.domain.clock import PlaybackClock


def test_elapsed_time_and_speed_change():
    ticks = [0.]
    clock = PlaybackClock(634., lambda: ticks[0])
    clock.set_playing(True)
    ticks[0] = 1
    assert clock.current() == 5
    clock.set_speed(1)
    ticks[0] = 3
    assert clock.current() == 7
    clock.set_playing(False)
    ticks[0] = 20
    assert clock.current() == 7


def test_latest_seek_wins_and_drag_pauses_clock():
    ticks = [0.]
    clock = PlaybackClock(634., lambda: ticks[0])
    clock.set_playing(True)
    clock.begin_seek()
    for position in [2, 5, 70, 400, 410.123]:
        clock.seek(position)
    ticks[0] = 10
    assert clock.current() == pytest.approx(410.123)
    clock.end_seek()
    ticks[0] += .2
    assert clock.current() == pytest.approx(411.123)


def test_end_and_negative_positions():
    ticks = [0.]
    clock = PlaybackClock(10., lambda: ticks[0])
    clock.seek(-1)
    assert clock.current() == 0
    clock.set_playing(True)
    ticks[0] = 100
    assert clock.current() == 10
