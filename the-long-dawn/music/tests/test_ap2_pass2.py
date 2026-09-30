"""A score pass 2 (AP2) on the NOTE DATA: the walk waits for the crossing's measured set-off, enters on it, builds with
the front's measured acceleration, and everything else is pass 1's build exactly.

    python -m pytest the-long-dawn/music/tests/test_ap2_pass2.py -q

No samples are loaded: conftest's capped attack-time stub makes pass 1's anticipations the same on every machine. The
walk's two parts (a one-shot pizzicato and a hand drum) take no anticipation from it. Every assertion below is paired
with a corruption that must make it fail (each was seen failing when written). A note-data proof says where a note
starts; verify_ap2_render.py measures where it sounds in the rendered files.
"""
import copy
import hashlib
import json
import os

import pytest

from conftest import MUSIC, _stub_sampler_deps

V3 = os.path.join(MUSIC, "v3")


@pytest.fixture(scope="module")
def built():
    _stub_sampler_deps()
    import score_v3_A as A
    import score_v3_AP2 as P
    from timeline_v3 import BarMap
    bm = BarMap("AP2")
    return A.build(BarMap("A")), P.build(bm), bm


@pytest.fixture
def pair(built):
    a, p, bm = built
    return a, copy.deepcopy(p), bm


def measured(sid):
    ms = json.load(open(os.path.join(V3, "events_AP2_measured.json")))
    return next(r for r in ms["new_sync"] if r["id"] == sid)["frame"]


def walk(score, pn, lo, hi):
    """the walk instrument pn's SOUNDING notes in [lo, hi): pass 1's part and, in AP2, its _go part"""
    import score_v3_AP2 as P
    return [n for _, n in P.walk_notes(score, pn, lo, hi)]


# ---------------------------------------------------------------------------------------------------- the anchors
def test_the_map_carries_the_measured_setoff_and_pace():
    from timeline_v3 import BarMap
    bm = BarMap("AP2")
    assert bm.event("walk_setoff")["frame"] == measured("walk_setoff") == 5180
    assert bm.event("walk_full")["frame"] == measured("walk_full") == 5276
    raw = {e["id"]: e for e in json.load(open(os.path.join(V3, "barmap_AP2.json")))["sync"]}
    assert raw["walk_setoff"]["source"] == raw["walk_full"]["source"] == "measured"
    # control: pass 1's own map has no set-off (its walk starts on the lantern), so nothing here can pass on A
    with pytest.raises(KeyError):
        BarMap("A").event("walk_setoff")


# ---------------------------------------------------------------------------------------------------- 1: no walk
def test_no_walking_pulse_between_the_lantern_and_the_setoff(pair):
    import score_v3_AP2 as P
    a, p, bm = pair
    lan, go = bm.ev("lantern"), bm.ev("walk_setoff")
    for pn in P.WALK:
        early = [n for n in walk(p, pn, lan, go)]
        assert early == [], f"{pn}: {[n.start * 20 for n in early]}"
        # the earliest sound of every step (anticipation, humanising and pre-roll included) is not before the set-off
        # by more than one frame
        assert all(P.onset_frames(p, pn, n)[0] >= go * 20 - 1.0 for n in walk(p, pn, lan, bm.ev("narrowest")))
    assert not any(e.startswith("WALK WHILE THE LINE STANDS") for e in P.walk_problems(p, bm))
    # negative controls: pass 1 itself (eight steps per part from the lantern, 4880-5160), and AP2 with one step put
    # back at 5160 (beat 258), and with a later step anticipated into the standing stretch
    bad = P.walk_problems(a, bm)
    for pn in P.WALK:
        assert sum(e.startswith(f"WALK WHILE THE LINE STANDS: {pn}") for e in bad) == 8
        assert len(walk(a, pn, lan, go)) == 8
    extra = copy.deepcopy(p)
    extra.P("feet").n(60, 258.0, 2.0, 0.2)
    assert any(e.startswith("WALK WHILE THE LINE STANDS: feet") for e in P.walk_problems(extra, bm))
    unmuted = copy.deepcopy(p)                                   # one of pass 1's standing steps left sounding
    next(n for n in unmuted.P("cb_pizz").notes if abs(n.start * 20 - 4960) < 1e-6).gain_db = 0.0
    assert any(e.startswith("WALK WHILE THE LINE STANDS: cb_pizz") for e in P.walk_problems(unmuted, bm))
    hidden = copy.deepcopy(p)
    n = walk(hidden, "cb_pizz", bm.ev("walk_full"), bm.ev("narrowest"))[0]
    n.kw["antic"] = (n.start - go) * 20 / 24 + 0.1          # sounds before the set-off although written after it
    assert any(e.startswith("WALK WHILE THE LINE STANDS: cb_pizz") for e in P.walk_problems(hidden, bm))


# ---------------------------------------------------------------------------------------------------- 2: the entry
def test_the_first_onset_is_within_one_frame_of_the_measured_setoff(pair):
    import score_v3_AP2 as P
    a, p, bm = pair
    f_go = measured("walk_setoff")
    for pn in P.WALK:
        part, first = P.walk_notes(p, pn, bm.ev("lantern"), bm.ev("narrowest"))[0]
        lo, hi = P.onset_frames(p, part, first)
        assert part == P.GO[pn] and f_go - 1.0 <= lo and hi <= f_go + 1.0, (pn, part, lo, hi)
        assert first.sync and first.kw.get("antic") == 0.0
    assert not any(e.startswith("FIRST STEP OFF THE SET-OFF") for e in P.walk_problems(p, bm))
    # negative controls: the entry two frames late, two frames early, and pass 1's entry (the lantern, 300 frames early)
    for d in (+2, -2):
        late = copy.deepcopy(p)
        walk(late, "feet", bm.ev("lantern"), bm.ev("narrowest"))[0].start += d / 20.0
        assert any(e.startswith("FIRST STEP OFF THE SET-OFF: feet") for e in P.walk_problems(late, bm))
    assert any(e.startswith("FIRST STEP OFF THE SET-OFF: cb_pizz") for e in P.walk_problems(a, bm))


def test_the_setoff_step_is_pass_1s_root_on_the_measured_frame(pair):
    import score_v3_AP2 as P
    a, p, bm = pair
    go = bm.ev("walk_setoff")
    for pn, want in (("cb_pizz", 34), ("feet", 60)):                  # Bb1: watch-fire 1's root; the hand drum
        first = walk(p, pn, bm.ev("lantern"), bm.ev("narrowest"))[0]
        src = walk(a, pn, go, bm.ev("narrowest"))[0]                   # pass 1's first step at or after the set-off
        assert first.start == go and first.pitch == src.pitch == want and src.start == 260.0
    # control: the same checks on pass 1's first step of the crossing (the lantern's D2) fail
    lan0 = walk(a, "cb_pizz", bm.ev("lantern"), bm.ev("narrowest"))[0]
    assert (lan0.start, lan0.pitch) != (go, 34)


# ---------------------------------------------------------------------------------------------------- 3: the build
def test_the_build_follows_the_fronts_measured_acceleration(pair):
    import score_v3_AP2 as P
    a, p, bm = pair
    assert P.build_problems(p, a, bm) == []
    go, full = bm.ev("walk_setoff"), bm.ev("walk_full")
    for pn in P.WALK:
        mine = walk(p, pn, go, bm.ev("narrowest"))
        src = walk(a, pn, go, bm.ev("narrowest"))
        ratio = [m.vel / s.vel for m, s in zip(mine, src)]
        # 5180 at rest (speed 0): ENTRY; 5240: 0.625 of pace -> 0.8125; 5280 and 5320 (past 5276): pass 1's level
        assert ratio == pytest.approx([0.5, 0.8125, 1.0, 1.0])
        gaps = [(b.start - x.start) * 20 for x, b in zip(mine, mine[1:])]
        assert gaps == pytest.approx([60.0, 40.0, 40.0])             # the steps close up to pass 1's pace
    # negative controls: the walk at full level from the set-off (no build), and a step moved ahead of the pace
    flat = copy.deepcopy(p)
    for m, s_ in zip(walk(flat, "cb_pizz", go, bm.ev("narrowest")), walk(a, "cb_pizz", go, bm.ev("narrowest"))):
        m.vel = s_.vel
    assert any(e.startswith("BUILD OFF THE FRONT'S ACCELERATION: cb_pizz") for e in P.build_problems(flat, a, bm))
    rush = copy.deepcopy(p)
    walk(rush, "feet", go, bm.ev("narrowest"))[1].start = go + 1.0     # 20 frames after the set-off: faster than pace
    assert any(e.startswith("A STEP FASTER THAN THE PACE") for e in P.walk_problems(rush, bm))
    assert full * 20 == 5276


# ---------------------------------------------------------------------------------------------------- 4: pass 1
def test_pass_1_note_for_note_outside_the_walks_window(pair):
    import score_v3_AP2 as P
    a, p, bm = pair
    assert P.preservation_problems(a, p, bm) == []
    # every part but the walk's is pass 1's exactly; the walk's own parts are pass 1's note for note, in the same
    # order (so the sampler's per-part variation stream is pass 1's), their steps from the lantern to walk_full muted
    lan, full = bm.ev("lantern"), bm.ev("walk_full")
    for pn in a.parts:
        if pn not in P.WALK:
            assert a.P(pn).to_dict() == p.P(pn).to_dict(), pn
    for pn in P.WALK:
        xa, xp = a.P(pn).to_dict()["notes"], p.P(pn).to_dict()["notes"]
        assert len(xa) == len(xp) and [x["start"] for x in xa] == [x["start"] for x in xp]
        for x, y in zip(xa, xp):
            muted = lan - 1e-6 <= x["start"] < full - 1e-6
            assert y == dict(x, gain_db=P.MUTE_DB) if muted else y == x
    assert set(p.parts) - set(a.parts) == set(P.GO.values())
    # negative controls, one per exclusion edge: the lantern's CALL (4880, inside the window but not the walk), the
    # cycle (the thinking cycle is not the walk), a walk step after walk_full (5520), and the mixer contract
    for pn, frame in (("call_vlaq", 4880), ("cycle", 5000), ("cb_pizz", 5520)):
        broken = copy.deepcopy(p)
        next(n for n in broken.P(pn).notes if abs(n.start * 20 - frame) < 1e-6).pitch += 1
        assert f"PASS-1 DATA CHANGED OUTSIDE THE WALK'S WINDOW: {pn}" in P.preservation_problems(a, broken, bm)
    broken = copy.deepcopy(p)
    broken.groups = {}
    assert "PASS-1 SCORE CONTRACT CHANGED: groups" in P.preservation_problems(a, broken, bm)


def test_the_lantern_call_and_the_standing_stretch_are_pass_1s(pair):
    a, p, bm = pair
    lan, go = bm.ev("lantern"), bm.ev("walk_setoff")
    for pn in ("call_vlaq", "vln1_q", "vln2_q", "watch", "cycle"):
        mine = [n.to_dict() for n in p.P(pn).notes if lan - 1e-6 <= n.start < go]
        assert mine and mine == [n.to_dict() for n in a.P(pn).notes if lan - 1e-6 <= n.start < go], pn
    call = [n for n in p.P("call_vlaq").notes if abs(n.start - lan) < 1e-6]
    assert [n.pitch for n in call] == [50]                             # the lantern's CALL begins on D3, as in pass 1
    # control: a missing lantern CALL is seen
    gone = copy.deepcopy(p)
    gone.P("call_vlaq").notes = [n for n in gone.P("call_vlaq").notes if abs(n.start - lan) > 1e-6]
    assert not [n for n in gone.P("call_vlaq").notes if abs(n.start - lan) < 1e-6]


def test_pass_1_itself_is_unchanged(built):
    a, _, _ = built
    # measured by the Codex lane on this worktree (A under the stub): 1,169 notes, this digest of every part's data
    data = {name: part.to_dict() for name, part in a.parts.items()}
    dig = hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()
    assert dig == "c53a08e9fc2bdfd0583be25777b79bc9820b9aa4afb8288dc21b23a031f4fa81"
    assert sum(len(p.notes) for p in a.used().values()) == 1169
    data["watch"]["notes"][0]["pitch"] += 1
    assert hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest() != dig


def test_sync_table_only_the_two_relabelled_entries(pair):
    import score_v3_AP2 as P
    a, p, bm = pair
    by = {x[1]: x for x in p.sync}
    assert by[P.SETOFF_LABEL][0] == pytest.approx(bm.event("walk_setoff")["t"]) and by[P.SETOFF_LABEL][2] == "feet_go"
    assert by[P.WF1_LABEL][2:] == ("vln2_q", 0.1, "pitch:65") and by[P.WF1_LABEL][0] == pytest.approx(210.0)
    assert len(p.sync) == len(a.sync)
    broken = copy.deepcopy(p)
    broken.sync = [x for x in broken.sync if x[1] != P.WF1_LABEL] + [(1.0, "an invented sync point", "vla", 0.2, "hit")]
    assert "PASS-1 SYNC TABLE CHANGED OUTSIDE THE TWO RELABELLED ENTRIES" in P.preservation_problems(a, broken, bm)


def test_page_checks_pass(built):
    import score_v3_AP2 as P
    a, p, bm = built
    assert P.check(p, bm, a) == []
    assert P.check(a, bm, a) != []                                     # control: pass 1 fails AP2's page checks


# ---------------------------------------------------------------------------------------------------- the map
def test_effects_resolve_to_As_frames(tmp_path):
    """AP2 plays A's effects: every one of A's sync frames, beds, events and breaths is A's in AP2's map"""
    from timeline_v3 import BarMap
    A, P = BarMap("A"), BarMap("AP2")
    fa = {e["id"]: e["frame"] for e in A.events}
    fp = {e["id"]: e["frame"] for e in P.events}
    assert {k: fp[k] for k in fa} == fa and set(fp) - set(fa) == {"walk_setoff", "walk_full"}
    assert A.d["ambience"] == P.d["ambience"] and A.d["sfx"] == P.d["sfx"] and A.breaths == P.breaths
    # control: a moved entry of A in AP2's map (watch-fire 1 moved to the set-off) resolves A's watch-fire effect
    # elsewhere, and this comparison sees it
    d = json.load(open(os.path.join(V3, "barmap_AP2.json")))
    next(e for e in d["sync"] if e["id"] == "watchfire_1")["f"] = 5180
    moved = tmp_path / "barmap_AP2.json"
    moved.write_text(json.dumps(d))
    M = BarMap("AP2", str(moved), cues=os.path.join(V3, "cues_AP2.json"))
    assert M.d["sfx"] != A.d["sfx"]


def test_generator_is_current_and_never_moves_an_entry_of_A(tmp_path, monkeypatch):
    import barmap_ap2 as G
    assert G.main(["--check"]) == 0
    stale = tmp_path / "barmap_AP2.json"
    stale.write_text("{}\n")
    monkeypatch.setattr(G, "OUT_MAP", str(stale))
    assert G.main(["--check"]) == 1
    ms = json.load(open(G.MEASURED))
    ms["new_sync"].append(dict(id="watchfire_1", frame=5180, what="moved", evidence="none"))
    bad = tmp_path / "moved.json"
    bad.write_text(json.dumps(ms))
    monkeypatch.setattr(G, "MEASURED", str(bad))
    with pytest.raises(ValueError, match="never moves one of A's"):
        G.build()


def test_a_muted_step_renders_exactly_silent():
    """sampler.render_note scales a note by float32(10 ** ((gain_db + gain_var + patch gain) / 20)): at MUTE_DB that
    underflows to 0.0 for any variation the sampler draws (|gain_var| is a few dB), so a muted step adds exact zeros"""
    import numpy as np
    import score_v3_AP2 as P
    for var in (-3.0, 0.0, 3.0, 12.0):
        assert np.float32(10 ** ((P.MUTE_DB + var) / 20.0)) == 0.0
    # control: a merely quiet note (the audibility floor used by the checks) is not silent
    assert np.float32(10 ** ((P.AUDIBLE_DB + 0.0) / 20.0)) > 0.0
