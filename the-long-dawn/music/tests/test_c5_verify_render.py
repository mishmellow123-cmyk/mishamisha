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
    return stem_with([(f, "arrival") for f in frames], lead, tail)


def add_swell(y, offset, t0, rise=0.8):
    """mono, in place: a 440 Hz tone rising from -20 dB to 0 dB, linear in dB, over `rise` s from t0, then held: its
    -8 dB point is at t0 + 0.6 * rise and its -6 dB point at t0 + 0.7 * rise"""
    i0 = int(t0 * SR) - offset
    t = np.arange(int((rise + 0.6) * SR)) / SR
    lvl = 10 ** ((-20 + 20 * np.minimum(1.0, t / rise)) / 20)
    y[i0:i0 + len(t)] += (0.5 * lvl * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    return y


def stem_with(entries, lead=100, tail=60):
    """-> (offset, mono stem) with each (frame, kind) entry shaped as its check expects to find it ON:
    'arrival' arrives on the frame, 'bloom' starts on it, 'swell' puts the frame midway between its -8 and -6 dB
    points (early_s moves a swell earlier)"""
    frames = [e[0] for e in entries]
    offset = (min(frames) - lead) * FR
    y = np.zeros((max(frames) + tail) * FR - offset, np.float32)
    for e in entries:
        f, kind, early = e[0], e[1], (e[2] if len(e) > 2 else 0.0)
        if kind == "arrival":
            add_tone(y, offset, f / V.FPS - ATTACK / 3)
        elif kind == "bloom":
            add_tone(y, offset, f / V.FPS)
        else:
            add_swell(y, offset, f / V.FPS - 0.52 - early)
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
    for kind, pairs in (("arrival", V.ARRIVALS), ("swell", V.SWELLS), ("bloom", V.BLOOMS)):
        for eid, pn in pairs:
            want.setdefault(pn, []).append((int(ev[eid]), kind))
    manifest = {}
    for pn, entries in list(want.items()) + [(pn, [(4010, "arrival")]) for pn in V.RING_ENTRY]:
        off, y = stem_with(entries, lead=8 if pn in V.RING_ENTRY else 100)
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
    sw = [r for r in rep if r["check"] == "swell"]
    assert len(sw) == len(V.SWELLS) and all(r["status"] == "ON" for r in sw), sw
    bl = [r for r in rep if r["check"] == "bloom onset"]
    assert len(bl) == len(V.BLOOMS) and all(r["status"] == "ON" for r in bl), bl
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


def _replace_stem(parts, pn, entries):
    off, y = stem_with(entries)
    np.save(parts / f"{pn}__k.npy", y)
    (parts / f"{pn}__k.json").write_text(json.dumps(dict(offset=off, n=len(y))))


def test_run_catches_a_late_bloom(fake_render):
    """a bloom is judged by its ONSET: the same stem started 3 frames late is OFF"""
    kw, parts = fake_render
    ev, _ = V.barmap_frames("C5P2")
    entries = [(int(ev[e]), "arrival") for e, p in V.ARRIVALS if p == "tuba"] + [(int(ev["all_lit"]) + 3, "bloom")]
    _replace_stem(parts, "tuba", entries)
    rep = V.run("C5P2", **kw)
    r = [x for x in rep if x["check"] == "bloom onset" and x["part"] == "tuba"]
    assert r and r[0]["status"] == "OFF" and not r[0]["ok"] and r[0]["delta_frames"] > 2


def test_a_bloom_is_not_judged_by_its_arrival():
    """the same bloom, on its frame: ON by onset, but its -6 dB point (the arrival) is a few ms later; a swell's
    arrival lands frames later still, which is why the brass on all_lit is a bloom check, not an arrival check"""
    off, y = stem_with([(3791, "bloom")])
    assert V.arrival(y, off, 3791, under_db=V.BLOOM_ONSET_DB)["status"] == "ON"
    off, y = stem_with([(3791, "swell")])
    late = V.arrival(y, off, 3791)
    assert late["status"] == "ON" or late["delta_frames"] > 0


def test_run_catches_a_swell_wholly_early(fake_render):
    """a swell whose whole arrival zone lies before its frame (here 0.17 s early) is OFF"""
    kw, parts = fake_render
    ev, _ = V.barmap_frames("C5P2")
    _replace_stem(parts, "vln2", [(int(ev["pulls_ahead"]), "swell", 0.17)])
    rep = V.run("C5P2", **kw)
    r = [x for x in rep if x["check"] == "swell" and x["part"] == "vln2"]
    assert r and r[0]["status"] == "OFF" and not r[0]["ok"] and r[0]["zone_frames"][1] < -1.0
