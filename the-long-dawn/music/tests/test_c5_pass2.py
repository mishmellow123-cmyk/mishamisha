"""Score pass 2 (score_v3_C5P2 on barmap_C5P2): its timing invariants, proven on the note data, since nothing can be
rendered on the machine it was written on (no VSCO-2-CE samples, no `soxr`). Pass 1 (score_v3_C5 on barmap_C5) is
the control: it must stay byte-identical in its note data, and it must FAIL the checks pass 2 was written to pass."""
import hashlib
import json
import math
import os
import subprocess
import sys

import pytest

from conftest import SRC

FPS, BEAT_F = 24, 20                     # 72 BPM at 24 fps: one beat is 20 frames


def beat(frame):
    return frame / BEAT_F


def notes_at(S, pn, b, tol=1e-6):
    return [n for n in S.P(pn).notes if n.pitch is not None and abs(n.start - b) < tol]


def digest(S):
    parts = S.used()
    blob = json.dumps(dict(parts=[parts[k].to_dict() for k in sorted(parts)],
                           breaths=[[a, b, sorted(e)] for a, b, e in S.breaths], fader=S.fader),
                      sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()


# pass 1's note data (every part, note, dynamic point, breath and the fader) under the worst-case anticipation stub,
# at 69a1788 (the pass-1 inputs are unchanged on this branch): pass 2 is a separate module and must never move it
PASS1_DIGEST = "e1cf33b8eeeab849f6ae77277bd969133bc3dc686bd5df056622322eb726f128"


def test_pass1_is_reproducible(scores):
    S1, _ = scores["C5"]
    assert digest(S1) == PASS1_DIGEST


def test_barmap_p2_is_generated_from_the_measured_table():
    r = subprocess.run([sys.executable, os.path.join(SRC, "barmap_c5p2.py"), "--check"], cwd=SRC,
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr


def test_barmap_p2_events_are_the_measured_frames(scores):
    _, bm = scores["C5P2"]
    f = {e["id"]: e["frame"] for e in bm.events}
    assert (f["low_fire"], f["surge"], f["flares_back"], f["pulls_ahead"], f["leaders_reach"]) == (2423, 2483, 2546,
                                                                                                 2588, 2625)
    assert (f["reveal"], f["one_dark"], f["last_beacon"], f["all_lit"], f["forges_cold"]) == (2880, 3722, 3786, 3791,
                                                                                             3848)
    assert [f[f"map_beacon_{k}"] for k in range(2, 8)] == [3468, 3523, 3584, 3647, 3680, 3718]
    assert f["storm_gone"] == 4181 and f["ring_unfinished"] == 4000


def test_pass1_and_pass2_share_everything_outside_the_resynchronised_span(scores, patches):
    """the note lists are identical before the Trap (2320) and after the Ring's entry (4000); dynamics of notes
    outside differ only where a curve interpolates into the changed span, by < 0.01 dB"""
    import score_v3_C5P2 as P2
    from dsl import dyn_at
    from timeline_v3 import BEAT_S
    (S1, _), (S2, _) = scores["C5"], scores["C5P2"]
    u1, u2 = S1.used(), S2.used()
    assert set(u1) == set(u2)
    lo, hi = beat(2320), beat(4000) + 0.03

    def outside(parts, pn):
        return sorted(json.dumps(n.to_dict(), sort_keys=True) for n in parts[pn].notes
                      if not lo <= n.start < hi)
    for pn in u1:
        assert outside(u1, pn) == outside(u2, pn), pn
    worst = 0.0
    for pn in u1:
        d1, d2 = u1[pn].dyn, u2[pn].dyn
        if d1 == d2:
            continue
        for n in u1[pn].notes:
            if n.pitch is None or lo <= n.start < hi:
                continue
            a, b = (x / BEAT_S for x in P2.sounding(S1, pn, n, patches))
            k = max(2, int((b - a) / 0.01))
            for i in range(k + 1):
                x = a + (b - a) * i / k
                worst = max(worst, abs(30 * math.log10(max(dyn_at(d1, x), .03)) -
                                       30 * math.log10(max(dyn_at(d2, x), .03))))
    assert worst < 0.01, worst


def test_two_voices_answer_at_the_paired_first_fires(scores):
    S2, _ = scores["C5P2"]
    rev = beat(2880)
    assert rev == 144.0
    near = sorted((n.start, n.pitch) for n in S2.P("hn3").notes if rev - 1e-6 <= n.start < rev + 4)
    far = sorted((n.start, n.pitch) for n in S2.P("hn_far").notes if rev - 1e-6 <= n.start < rev + 4)
    assert near == [(144.0, 62), (145.0, 69), (146.0, 74)]          # D4 A4 D5: hers, near
    assert far == [(144.0, 57), (145.0, 64), (146.0, 69)]           # A3 E4 A4: the rival's, far, a fifth lower
    assert not any(n.legato for n in notes_at(S2, "hn3", rev) + notes_at(S2, "hn_far", rev))
    assert S2.P("hn_far").pan > 0 and S2.P("hn3").pan < S2.P("hn_far").pan    # the rival's fire is to the right


def test_pass1_has_one_voice_at_the_first_fires(scores):
    """the control for the rule above: pass 1 has a single call there, and the rule says so"""
    import score_v3_C5P2 as P2
    (S1, bm1), (S2, bm2) = scores["C5"], scores["C5P2"]
    assert not notes_at(S1, "hn3", beat(2880))
    assert P2.two_voice_problems(S1, bm1) and not P2.two_voice_problems(S2, bm2)


def test_holdout_resolves_on_the_measured_catch(scores):
    (S1, _), (S2, _) = scores["C5"], scores["C5P2"]
    gm7, sus, lit = beat(3722) - 4, beat(3722), beat(3791)
    assert (gm7, sus, lit) == (182.1, 186.1, 189.55)
    assert sorted(n.pitch for n in notes_at(S2, "vla", gm7)) == [58]            # Gm7: Bb3
    assert sorted(n.pitch for n in notes_at(S2, "vla", sus)) == [55, 62]        # A7sus4: G3 D4, through the pause
    assert sorted(n.pitch for n in notes_at(S2, "vla", lit)) == [54, 57]        # D: F#3 A3, all eight lit
    assert sorted(n.pitch for n in notes_at(S2, "cb", lit)) == [38]
    assert not notes_at(S2, "vla", beat(3760) + 0.5)                            # the retired 3760 catch is gone
    assert notes_at(S1, "vla", beat(3760) + 0.5)                                # (pass 1 keeps it)
    lit_sync = [s for s in S2.sync if "every beacon lit" in s[1]]
    assert lit_sync and abs(lit_sync[0][0] * FPS - 3791) < 1e-6


def test_far_beacons_answer_on_their_measured_catches(scores):
    S2, _ = scores["C5P2"]
    b3, b4 = beat(3523), beat(3584)
    assert (b3, b4) == (176.15, 179.2)
    assert notes_at(S2, "hn_far", b3) and notes_at(S2, "hn_farther", b4)


def test_trap_follows_the_measured_forges(scores):
    S2, _ = scores["C5P2"]
    low, surge, back, ahead, reach = (beat(f) for f in (2423, 2483, 2546, 2588, 2625))
    assert [n.pitch for n in notes_at(S2, "hn3", low)] == [69]                  # the lone horn tries the refusal
    assert notes_at(S2, "tbn", surge) and notes_at(S2, "tbn2", surge)           # the corrupted call returns
    call = sorted((round(n.start, 4), round(n.dur, 4), n.pitch) for n in S2.P("hn3").notes
                  if back - 1e-6 <= n.start < ahead)
    assert call == [(127.3, 0.525, 62), (127.825, 0.525, 68), (128.35, 1.05, 74)]
    assert call[-1][0] + call[-1][1] == pytest.approx(ahead)                    # the call ends as the climb starts
    climb = sorted((round(n.start, 4), n.pitch) for n in S2.P("vln1").notes if ahead - 1e-6 <= n.start <= reach + 1e-6)
    assert climb[0] == (129.4, 74) and climb[-1] == (131.25, 81)                # D5 ... A5 as they reach the Ring
    assert not [n for n in S2.P("vc_sp").notes if low + 1e-6 < n.start < surge - 1e-6]   # no ostinato while low


def test_hard_silence_pass2_clean_and_pass1_flagged(scores, patches):
    import score_v3_C5P2 as P2
    (S1, bm1), (S2, bm2) = scores["C5"], scores["C5P2"]
    assert P2.silence_problems(S2, bm2, patches) == []
    p1 = P2.silence_problems(S1, bm1, patches)
    starts = sorted(x.split()[4] for x in p1 if x.startswith("STARTS IN THE SILENCE"))
    assert starts == ["cb_q", "vc_q", "vla_q"]                                   # pass 1's anticipated Ring entry
    assert any(x.startswith("SOUNDS INTO THE SILENCE") for x in p1)             # and its swell's tails
    cold, ring = beat(3848), beat(4000)
    for pn, p in S2.used().items():
        for n in p.notes:
            if n.pitch is not None:
                assert not cold - 1e-9 <= n.start < ring, (pn, n.start)
                if n.start < cold:
                    lo_s, hi_s = P2.sounding(S2, pn, n, patches)
                    assert hi_s <= 3848 / FPS + 1e-9, (pn, n.start, hi_s * FPS)


def test_ring_entry_starts_on_the_cut(scores):
    S2, _ = scores["C5P2"]
    ring = beat(4000)
    entry = [(pn, n) for pn in ("cb_q", "vc_q", "vla_q") for n in S2.P(pn).notes
             if n.pitch is not None and abs(n.start - ring - 0.02) < 1e-9]
    assert len(entry) == 3 and sorted(S2.pinned) == ["cb_q", "vc_q", "vla_q"]
    for pn, n in entry:
        assert n.kw["antic"] == 0.0 and n.sync, pn


def test_swell_is_cut_on_the_shutdown(scores):
    import score_v3_C5P2 as P2
    S2, _ = scores["C5P2"]
    assert len(S2.cut_at_shutdown) == 15
    for pn, start, old_end, new_end in S2.cut_at_shutdown:
        assert start == pytest.approx(beat(3791)) or start == pytest.approx(beat(3791) + 0.5), pn
        assert old_end == pytest.approx(beat(3848) + 0.25) and new_end < beat(3848)
    for pn in ("vla", "hn", "timp_roll"):
        cut = [n for n in S2.P(pn).notes if n.kw.get("rel") == P2.CUT_REL_S]
        assert cut, pn


def test_check_is_clean_for_pass2(scores, patches):
    import score_v3_C5P2 as P2
    import kit_v3 as K
    S2, bm2 = scores["C5P2"]
    parts = S2.used()
    assert K.check_notes(parts, bm2.bars * 4) + K.check_rates(parts) == []
    assert P2.check(S2, bm2, patches) == []


# ------------------------------------------------------------------ negative tests: the checks must go red
def test_check_flags_a_stroke_on_a_measured_ignition(fresh_scores, patches):
    import score_v3_C5P2 as P2
    import score_v3_C as C1
    S, bm = fresh_scores("C5P2")
    pn = next(p for p in C1.STROKES if p in S.parts)
    S.P(pn).n("C5", bm.ev("map_beacon_5"), 1.0, 0.3)
    assert any(w.startswith("STROKE ON map_beacon_5") for w in P2.check(S, bm, patches))


def test_check_flags_a_missing_far_voice(fresh_scores, patches):
    import score_v3_C5P2 as P2
    S, bm = fresh_scores("C5P2")
    for n in notes_at(S, P2.FAR, bm.ev("reveal")):
        n.pitch = None
    assert any("TWO VOICES" in w for w in P2.check(S, bm, patches))


def test_check_flags_a_note_in_the_silence(fresh_scores, patches):
    import score_v3_C5P2 as P2
    S, bm = fresh_scores("C5P2")
    S.P("vla").n("D4", beat(3900), 1.0, 0.3)
    assert any(w.startswith("STARTS IN THE SILENCE: vla") for w in P2.check(S, bm, patches))


def test_check_flags_a_tail_into_the_silence(fresh_scores, patches):
    import score_v3_C5P2 as P2
    S, bm = fresh_scores("C5P2")
    n = next(n for n in S.P("vla").notes if abs(n.start - beat(3791)) < 1e-6)
    n.dur += 0.3
    assert any(w.startswith("SOUNDS INTO THE SILENCE: vla") for w in P2.check(S, bm, patches))


def test_silence_window_requires_the_breath(fresh_scores):
    import score_v3_C5P2 as P2
    _, bm = fresh_scores("C5P2")
    bm.breath_beats = lambda: []
    with pytest.raises(ValueError, match="no breath at forges_cold"):
        P2.silence_window(bm)


def test_check_admits_guessed_releases(scores, monkeypatch):
    """without the sampler's patch table the silence check must say it fell back, never pass silently"""
    import score_v3_C5P2 as P2
    S2, bm2 = scores["C5P2"]
    monkeypatch.setattr(P2, "_patches", lambda: None)
    assert any("DEFAULT RELEASES" in w for w in P2.check(S2, bm2))
