"""verify_c5_render.py on SYNTHETIC renders (no real render exists on the machine it was written on): every verdict is
shown both ways, so that a pass on the real render means the measurement ran, not that it could not fail."""
import json

import numpy as np
import pytest

import verify_c5_render as V

SR, FR = V.SR, V.FR
ATTACK = 0.12                              # a smooth attack: the 6 dB-under-peak point falls ATTACK / 3 after onset


def add_tone(y, offset, start, dur=1.0):
    """mono, in place: a 440 Hz tone whose sine-shaped attack puts its arrival (-6 dB under the sustain) at
    start + ATTACK / 3 (seconds; y's sample 0 is sample `offset` of the film)"""
    i0 = int(start * SR) - offset
    t = np.arange(int(dur * SR)) / SR
    env = np.sin(np.pi / 2 * np.minimum(1.0, t / ATTACK))
    y[i0:i0 + len(t)] += (0.5 * env * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    return y


def stem_arriving_at(frames, lead=100, tail=60):
    """-> (offset, mono stem) whose entries ARRIVE on the given frames, stored like render_v3's cropped part stems"""
    offset = (min(frames) - lead) * FR
    y = np.zeros((max(frames) + tail) * FR - offset, np.float32)
    for f in frames:
        add_tone(y, offset, f / V.FPS - ATTACK / 3)
    return offset, y


def test_arrival_on_the_frame():
    a = V.arrival(*stem_arriving_at([2880])[::-1], 2880)
    assert a["status"] == "ON" and abs(a["delta_frames"]) <= 1.0, a


def test_arrival_three_frames_late_is_off():
    a = V.arrival(*stem_arriving_at([2883])[::-1], 2880)
    assert a["status"] == "OFF" and 2.0 <= a["delta_frames"] <= 4.0, a


def test_arrival_not_isolated_is_not_a_pass():
    off, y = stem_arriving_at([2880])
    add_tone(y, off, 2880 / V.FPS - 1.6)
    assert V.arrival(y, off, 2880)["status"] == "NOT ISOLATED"


def test_arrival_silent():
    assert V.arrival(np.zeros(3000 * FR, np.float32), 0, 2880)["status"] == "SILENT"


def test_arrival_outside_the_stored_stem_reads_as_silence():
    off, y = stem_arriving_at([2880])
    assert V.arrival(y, off + 400 * FR, 2880)["status"] == "SILENT"     # the same stem stored 400 frames later


def test_silence_check_both_ways():
    y = np.zeros(((V.RING_CUT - V.SHUTDOWN) * FR, 2), np.float32)
    y[:FR // 2] = 0.3                                      # the cut lands inside frame 3848: reported, not judged
    r = V.check_silence(y, "x", -90.0, first_frame=V.SHUTDOWN)
    assert r["ok"] and r["first_frame_peak_db"] > -20
    y[(3900 - V.SHUTDOWN) * FR + 7] = 1e-3                  # -60 dBFS inside the silence
    r = V.check_silence(y, "x", -90.0, first_frame=V.SHUTDOWN)
    assert not r["ok"] and list(r["over_floor"]) == [3900]
    r = V.check_silence(y[:10 * FR], "x", -90.0, first_frame=V.SHUTDOWN)   # a truncated file is not silence
    assert not r["ok"] and r["frames_missing"]


def test_not_before_both_ways():
    r = V.check_not_before(*stem_arriving_at([3990])[::-1], 4000, -90.0)
    assert not r["ok"] and 3980 <= r["first_sound_frame"] < 3990
    r = V.check_not_before(*stem_arriving_at([4010], lead=8)[::-1], 4000, -90.0)
    assert r["ok"] and r["first_sound_frame"] is None


def test_length_both_ways():
    assert V.check_length(5920 * FR, 5920, "x")["ok"]
    assert not V.check_length(7200 * FR, 5920, "x")["ok"]


@pytest.fixture
def fake_render(tmp_path):
    """a C5P2 'render': a silent full-length premaster (sparse), part stems whose entries arrive on the bar map's
    measured frames, the Ring's entry after the cut, and no master"""
    ev, bm = V.barmap_frames("C5P2")
    cache, parts, out = tmp_path / "cache", tmp_path / "cache" / "parts", tmp_path / "out"
    for d in (cache, parts, out):
        d.mkdir(exist_ok=True)
    np.lib.format.open_memmap(cache / "premaster_score_final_C5P2.npy", mode="w+", dtype=np.float32,
                              shape=(bm["frames"] * FR, 2)).flush()
    want = {}
    for eid, pn in V.ARRIVALS:
        want.setdefault(pn, []).append(int(ev[eid]))
    manifest = {}
    for pn, frames in list(want.items()) + [(pn, [4010]) for pn in V.RING_ENTRY]:
        off, y = stem_arriving_at(frames, lead=8 if pn in V.RING_ENTRY else 100)
        np.save(parts / f"{pn}__k.npy", y)
        (parts / f"{pn}__k.json").write_text(json.dumps(dict(offset=off, n=len(y))))
        manifest[pn] = "k"
    (cache / "manifest_final_C5P2.json").write_text(json.dumps(manifest))
    return dict(cache=str(cache), out_dir=str(out), parts_dir=str(parts)), parts


def test_run_on_a_fake_render(fake_render):
    kw, _ = fake_render
    rep = V.run("C5P2", **kw)
    bad = [r for r in rep if not r["ok"]]
    assert [(r["check"], r["what"]) for r in bad] == [("inputs", "final_C5P2.wav")]    # the only thing missing
    arr = [r for r in rep if r["check"] == "arrival"]
    assert len(arr) == len(V.ARRIVALS) and all(r["status"] == "ON" for r in arr)
    assert all(r["ok"] for r in rep if r["check"] in ("length", "silence", "not before"))


def test_run_catches_a_late_voice(fake_render):
    kw, parts = fake_render
    ev, _ = V.barmap_frames("C5P2")
    off, y = stem_arriving_at([int(ev["reveal"]) + 4, int(ev["map_beacon_3"])])  # the far voice 4 frames late
    np.save(parts / "hn_far__k.npy", y)
    (parts / "hn_far__k.json").write_text(json.dumps(dict(offset=off, n=len(y))))
    rep = V.run("C5P2", **kw)
    late = [r for r in rep if r["check"] == "arrival" and r["part"] == "hn_far" and r["event"] == "reveal"]
    assert late and late[0]["status"] == "OFF" and not late[0]["ok"]
