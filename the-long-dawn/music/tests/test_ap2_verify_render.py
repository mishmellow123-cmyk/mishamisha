"""verify_ap2_render's measuring functions on SYNTHETIC stems (no render needed): each check passes the case it is
built for and fails its corruption.

    python -m pytest the-long-dawn/music/tests/test_ap2_verify_render.py -q
"""
import numpy as np
import pytest

import verify_ap2_render as V

FR = V.FR
LAN, GO = 4880, 5180


def step(at_frame, n=8000, amp=0.2, pre=8):
    """a one-shot's stem (offset, array): a 4 ms (8-sample) pre-roll ramp, then a decaying hit"""
    t = np.arange(n, dtype=np.float32)
    y = amp * np.exp(-t / 2000.0).astype(np.float32)
    y[:pre] *= np.linspace(0.0, 0.05, pre, dtype=np.float32)
    return int(at_frame * FR) - pre, np.stack([y, y], 1)


def test_entry_on_the_setoff_passes_and_its_corruptions_fail():
    ok = V.entry([step(GO)], LAN, GO)
    assert ok["ok"] and ok["quiet_peak"] == 0.0 and abs(ok["offset_frames"]) <= 1.0
    late = V.entry([step(GO + 2)], LAN, GO)                            # two frames late
    assert not late["ok"] and late["offset_frames"] == pytest.approx(2.0, abs=0.05)
    early = V.entry([step(GO - 2)], LAN, GO)                           # two frames early: it sounds in the stillness
    assert not early["ok"] and early["quiet_peak"] > 0.0
    standing = V.entry([step(4960), step(GO)], LAN, GO)                # a step while the line stands
    assert not standing["ok"] and standing["quiet_peak"] > 0.1


def test_a_tail_of_a_step_before_the_lantern_is_not_the_walk():
    # the stillness window starts at the lantern: a sound ending before it is outside, one ringing into it is not
    before = V.entry([step(LAN - 40, n=2000), step(GO)], LAN, GO)
    assert before["ok"]
    rings_in = V.entry([step(LAN - 5, n=40000), step(GO)], LAN, GO)
    assert not rings_in["ok"]


def test_window_sums_stems_at_their_offsets():
    a = (10, np.ones((5, 2), np.float32))
    b = (12, 2 * np.ones((5, 2), np.float32))
    w = V.window([a, b], 8, 20)
    assert w[:, 0].tolist() == [0, 0, 1, 1, 3, 3, 3, 2, 2, 0, 0, 0]
    assert V.window([a], 0, 10).max() == 0.0                           # control: nothing before the offset
