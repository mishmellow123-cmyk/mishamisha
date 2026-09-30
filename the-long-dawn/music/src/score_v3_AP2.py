"""THE LONG DAWN v3 - A score PASS 2 (AP2, 29 Sep): pass 1 on the restaged crossing's MEASURED set-off.

    python barmap_ap2.py              # (re)generate v3/barmap_AP2.json + cues_AP2.json from the measured table
    python score_v3_AP2.py            # page checks: no walk before the set-off, the walk's entry, pass-1 equality
    python render_v3.py AP2           # -> out/v3/final_AP2*.wav, score_AP2.wav, cache/v3/premaster_score_final_AP2.npy;
                                      #    pass 1's final_A* are untouched (a separate cut id, caches and outputs)
    python sound_v3.py AP2            # -> out/v3/sound_AP2*.wav: A's recorded effects under this score
    python verify_ap2_render.py       # the rendered files: length, the walk's audible entry, A outside the edit

Pass 1 (score_v3_A.py, COMPOSER-A) is the music. A18 THE CROSSING now plays the RUN-A4 restage
(renders/cand_crossing_both_decal): the line stands at the first watch-fire through the close and medium shots and
sets off, front first, as the sky begins to wheel. Pass 1 was written for a line that walks from the match cut, so
its WALK starts on the lantern (4880): twelve seconds of footsteps over a line standing still. Pass 2 changes only
that, and everything else is pass 1's own build() on A's own map, unchanged:

  1. THE WALK WAITS FOR THE SET-OFF. The walk is the kit's WALK (kit_v3.walk): the pizzicato basses and the soft hand
     drum (`feet`) on 1 and 3. Measured (music/src/measure_ap2_crossing.py; v3/events_AP2_measured.json): the front
     bearer's arc along its path is constant through the image at 5180 and rises from that instant (the renderer's
     speed ramp starts there at 0), first displaced image 5181, a foot 1 px (full size) at 5185, the body 1 px at 5190;
     the delivered frames' terrain slides past the line after 5180 as that set-off predicts and a standing line
     cannot. So no step sounds from the lantern until walk_setoff (5180, on 65 b4: the line leaves on an upbeat), where
     pass 1's first step at or after it (its 5200 step, the root Bb1) is placed; from there pass 1's own steps (5240,
     5280, 5320, then its narrowest-stretch rest and its return at 5520) play as written, 5240 at the build's level (2).
  2. THE BUILD FOLLOWS THE FRONT'S ACCELERATION. The line's speed rises linearly from 0 at the set-off to walking
     pace at walk_full (5276, measured: half pace 5228). Each step between them sounds at ENTRY + (1 - ENTRY) x the
     front's speed fraction of pass 1's level (5180: 0.5; 5240: 0.8125; 5280 on: 1.0, pass 1's step); the first step
     interval is 60 frames, then pass 1's 40: the steps close up as the line gets going, and reach pass 1's pace and
     level on watch-fire 2. ENTRY is a composition choice (not measured): the first step half pass 1's level, -9 dB in
     the sampler's level curve (30 log10), on its softer sample layers, while the motion is still under a pixel.
     The build follows the FRONT (the bearers and their great lantern, whose pace the walk has always kept:
     crossing.py, "The line walks at the pace of the great lantern's pool of light"), not the whole line, whose 42
     figures set off front to back from 5184 to 5367 (half by 5330) and reach 0.9 of walking pace from 5267 to 5549
     (half by 5414, inside the walk's narrowest-stretch rest, 5360-5520).
  3. THE STANDING STRETCH (4880 to the set-off) is carried by what pass 1 already plays there, unchanged: the lantern's
     CALL in the violas at 4880 (it stays), the violins' held pad (D sus2, Bbmaj7 from 5040), the watch tone, the
     second CALL at 5040 and the thinking cycle in quarter notes. The cycle is not the walk: it has ticked in quarter
     notes since cycle_locks (4640), over towers that stand still, and it runs on across the match cut as the heart of
     the fire becomes the lantern's light ("the same light, now inside a great lantern", crossing.py); pass 1 calls it
     "walking in step" because it keeps step with the walk once there is one, which from 5180 there is. No new
     material.
  4. WATCH-FIRE 1 (5040) IS RELABELLED, NOT MOVED. Measured: the line does not pass it at 5040 (it has stood beside
     it since 4880; the lantern is 638 px from it there, full size); the front bearer passes it at 5304 and the lantern
     at 5368 (screen x), just behind it (0.45 m farther from the camera). Pass 1's harmony change and CALL at 5040 are
     part of the standing stretch (3); following the real pass would put Bb after watch-fire 2's F (5280), the
     lantern's pass inside the narrowest stretch (from 5360), and A's effects are anchored on that frame. Watch-fires 2-4 are confirmed where pass 1 has them (the fire within 15 px of the lantern over
     5242-5358, 5484-5616, 5207-5839), and 'ridges' (3610) drives no note of pass 1 (the report has its re-measurement).

HOW THE WALK IS CHANGED, AND WHY THIS WAY. The sampler draws each part's per-note variations (+-6 cents, a gain
offset of 0.5 dB sd, a <= 3 ms start trim, the humanising) from one seeded stream in note order, and a note's level
picks its sample layers, each with its own round-robin counter. Deleting pass 1's standing steps, or re-levelling them
in place, would therefore change how every LATER step of that part sounds (the first render of this pass did: the same
notes from 5520 on, different variations and samples). So pass 1's steps from the lantern to walk_full stay in their
parts, note for note, muted (gain MUTE_DB: the sampler's gain underflows to exactly 0.0); the two steps before
walk_full sound from two new parts on the same seats (cb_pizz_go, feet_go: pass 1's 5200 step on the set-off at ENTRY,
its 5240 step at the build's level); and pass 1's own parts then play every step from walk_full on (5280, 5320,
5520-5800) exactly as in final_A, the same samples included. Every other part is pass 1's, note for note and cache key
for cache key.
"""
import kit_v3 as K
import score_v3_A as P1
from timeline_v3 import BEAT_S, BarMap

WALK = ("cb_pizz", "feet")          # kit_v3.walk's two parts: the pizzicato basses and the hand drum (the feet)
GO = {"cb_pizz": "cb_pizz_go", "feet": "feet_go"}     # AP2's steps before walk_full, on the same seats
MUTE_DB = -999.0                    # a pass-1 step kept for its place in the part's variation stream, silent
AUDIBLE_DB = -120.0                 # a note with gain_db at or under this is not a sounding note
ENTRY = 0.5                         # the first step's level, relative to pass 1's step (composition choice, above)
PACE_BEATS = 2.0                    # pass 1's walk: a step every two beats (40 frames) at walking pace
SETOFF_LABEL = "THE CROSSING: the line sets off (measured 5180): the walk's first step (the drum)"
WF1_LABEL = "watch-fire 1, the line standing at it: the harmony moves (violins E4 -> F4)"
PRE_S = 0.004                       # sampler: the pre-roll kept before a sample's detected onset


def anchors(bm):
    lan, go, full, narrow = (bm.ev(k) for k in ("lantern", "walk_setoff", "walk_full", "narrowest"))
    if not lan < go < full <= narrow:
        raise ValueError("AP2 needs lantern < walk_setoff < walk_full <= narrowest")
    return lan, go, full, narrow


def speed_frac(bm, beat):
    """the front's measured speed as a fraction of walking pace: 0 at the set-off, rising linearly to 1 at walk_full
    (crossing.lantern_s: constant acceleration SPEED / TAU_GO), then 1"""
    _, go, full, _ = anchors(bm)
    return min(1.0, max(0.0, (beat - go) / (full - go)))


def build_level(bm, beat):
    return ENTRY + (1.0 - ENTRY) * speed_frac(bm, beat)


def sounding(part, lo=-1e9, hi=1e9):
    """a part's sounding notes in [lo, hi) beats, by start"""
    return sorted((n for n in part.notes if n.pitch is not None and n.gain_db > AUDIBLE_DB and lo - 1e-6 <= n.start
                   < hi - 1e-6), key=lambda n: n.start)


def walk_notes(S, pn, lo=-1e9, hi=1e9):
    """(part name, note) of the walk instrument pn (pass 1's part and AP2's _go part) sounding in [lo, hi), by start"""
    out = [(pn, n) for n in sounding(S.P(pn), lo, hi)]
    if GO[pn] in S.parts:
        out += [(GO[pn], n) for n in sounding(S.P(GO[pn]), lo, hi)]
    return sorted(out, key=lambda x: x[1].start)


def walk_on_setoff(S, bm):
    """1 + 2: pass 1's steps from the lantern to walk_full are muted in place; its first step at or after the set-off
    sounds ON the set-off, and the next ones before walk_full at the front's speed, from the _go parts.
    Returns {part: [(pass-1 beat, AP2 beat, level factor), ...]} for the steps that sound differently."""
    import copy
    lan, go, full, narrow = anchors(bm)
    done = {}
    for pn in WALK:
        p = S.P(pn)
        span = sounding(p, lan, narrow)
        after = [n for n in span if n.start >= go - 1e-6]
        if not after or after[0].start >= go + PACE_BEATS - 1e-6:
            raise ValueError(f"{pn}: pass 1 has no step within one step of the set-off")
        if abs(((after[0].start - lan) / PACE_BEATS) % 2) > 1e-6:
            raise ValueError(f"{pn}: the step moved onto the set-off must be one of pass 1's steps on 1 (the root)")
        q = S.add(GO[pn], pn)
        rec = []
        for n in span:
            if n.start >= full - 1e-6:
                continue                       # walking pace: pass 1's own step, untouched
            if n.start >= go - 1e-6:
                c = copy.deepcopy(n)
                if n is after[0]:
                    c.start = go
                    c.sync = True              # no humanising: it sounds on the measured frame
                    c.kw["antic"] = 0.0        # a one-shot: its detected onset lands on the note (pre-roll only)
                f = build_level(bm, c.start)
                c.vel = c.vel * f
                q.notes.append(c)
                rec.append((round(n.start, 4), round(c.start, 4), round(f, 4)))
            n.gain_db = MUTE_DB
        q.notes.sort(key=lambda n: n.start)
        done[pn] = rec
    sync = []
    for t, label, part, w, kind in S.sync:
        if label == "THE CROSSING: the walk's first step (the drum)":
            t, label, part = go * BEAT_S, SETOFF_LABEL, GO["feet"]
        elif label.startswith("watch-fire 1: the harmony moves"):
            label, part, kind = WF1_LABEL, "vln2_q", "pitch:65"
        sync.append((t, label, part, w, kind))
    S.sync = sync
    return done


def build(bm):
    S = P1.build(bm)
    S.cut = "AP2"
    S.walk_moves = walk_on_setoff(S, bm)
    return S


# ---------------------------------------------------------------------------
# the page checks: the walk's timing on the note data, and pass 1 everywhere else
# ---------------------------------------------------------------------------
def onset_frames(S, pn, n):
    """(earliest, latest) frame at which note n's detected onset can land: its anticipation (the patch default when
    the note sets none), the sampler's pre-roll before it, and its humanising (1.5 sigma cap) unless it is a sync note"""
    import sampler
    p = S.P(pn)
    antic = float(n.kw.get("antic", sampler.P[n.art or p.inst]["antic"]))
    hum = 0.0 if n.sync else 1.5 * p.humanize_ms / 1000.0
    t = n.start * BEAT_S - antic
    return (t - hum - PRE_S) * 24.0, (t + hum) * 24.0


def walk_problems(S, bm):
    lan, go, full, narrow = anchors(bm)
    f_go = go * 20.0
    out = []
    for pn in WALK:
        notes = walk_notes(S, pn, lan, narrow)
        for part, n in notes:
            lo, _ = onset_frames(S, part, n)
            if lo < f_go - 1.0:
                out.append(f"WALK WHILE THE LINE STANDS: {pn} ({part}) sounds from frame {lo:.2f} (set-off {f_go:.0f})")
        if not notes:
            out.append(f"NO WALK AFTER THE SET-OFF: {pn}")
            continue
        lo, hi = onset_frames(S, *notes[0])
        if lo < f_go - 1.0 or hi > f_go + 1.0:
            out.append(f"FIRST STEP OFF THE SET-OFF: {pn} lands {lo:.2f}-{hi:.2f}, set-off {f_go:.0f} (+-1 frame)")
        st = [n.start for _, n in notes]
        gaps = [b - a for a, b in zip(st, st[1:]) if b <= full + PACE_BEATS]
        if any(g < PACE_BEATS - 1e-6 for g in gaps):
            out.append(f"A STEP FASTER THAN THE PACE WHILE THE LINE GETS GOING: {pn} gaps {gaps}")
        if any(b > a + 1e-6 for a, b in zip(gaps, gaps[1:])):
            out.append(f"THE STEPS DO NOT CLOSE UP AS THE LINE ACCELERATES: {pn} gaps {gaps}")
    return out


def build_problems(S, S1, bm):
    """from the set-off to the narrowest stretch the walk sounds pass 1's own steps from the set-off on, in order, pitch
    for pitch; the first on the set-off; each below walk_full at ENTRY + (1 - ENTRY) x the front's speed of pass 1's
    level (from the _go part), each from walk_full on at pass 1's level exactly (from pass 1's part)"""
    _, go, full, narrow = anchors(bm)
    out = []
    for pn in WALK:
        p1, mine = sounding(S1.P(pn), go, narrow), walk_notes(S, pn, go, narrow)
        if len(p1) != len(mine):
            out.append(f"THE WALK FROM THE SET-OFF IS NOT PASS 1'S STEPS: {pn}: {len(mine)} steps, pass 1 {len(p1)}")
            continue
        for k, (a, (part, n)) in enumerate(zip(p1, mine)):
            start = go if k == 0 else a.start
            want = a.vel * build_level(bm, start) if start < full - 1e-6 else a.vel
            where = GO[pn] if start < full - 1e-6 else pn
            if (abs(n.start - start) > 1e-9 or n.pitch != a.pitch or abs(n.vel - want) > 1e-9 or part != where):
                out.append(f"BUILD OFF THE FRONT'S ACCELERATION: {pn} step {k} at beat {n.start:.3f} in {part} (want "
                           f"{start:.3f} in {where}), pitch {n.pitch} (pass 1 {a.pitch}), vel {n.vel:.4f} "
                           f"(want {want:.4f})")
    return out


def preservation_problems(S1, S, bm):
    """pass 1's actual build, exactly, in every part (the walk's parts too: only the gain of their steps from the
    lantern to walk_full, now MUTE_DB); the _go parts hold only steps in [set-off, walk_full); pass 1's score
    contract (breaths, eq, groups, push, levels, centroid, rules) and its sync table but the two relabelled entries"""
    lan, go, full, _ = anchors(bm)
    out = []
    if set(S.parts) != set(S1.parts) | set(GO.values()):
        return ["PART SET CHANGED: " + ", ".join(sorted(set(S.parts) ^ (set(S1.parts) | set(GO.values()))))]
    for pn in S1.parts:
        a, b = S1.P(pn).to_dict(), S.P(pn).to_dict()
        if pn in WALK:
            for x in a["notes"]:
                if x["pitch"] is not None and lan - 1e-6 <= x["start"] < full - 1e-6:
                    x["gain_db"] = MUTE_DB
        if a != b:
            out.append(f"PASS-1 DATA CHANGED OUTSIDE THE WALK'S WINDOW: {pn}")
    for pn in GO.values():
        stray = [n.start for n in S.P(pn).notes if not go - 1e-6 <= n.start < full - 1e-6]
        if stray:
            out.append(f"A _go STEP OUTSIDE [set-off, walk_full): {pn} at {stray}")
    for attr in ("breaths", "eq", "groups", "push", "levels", "centroid", "rules"):
        if getattr(S1, attr) != getattr(S, attr):
            out.append(f"PASS-1 SCORE CONTRACT CHANGED: {attr}")
    keep = [x for x in S1.sync if not x[1].startswith(("THE CROSSING: the walk's first step", "watch-fire 1:"))]
    mine = [x for x in S.sync if x[1] not in (SETOFF_LABEL, WF1_LABEL)]
    if keep != mine:
        out.append("PASS-1 SYNC TABLE CHANGED OUTSIDE THE TWO RELABELLED ENTRIES")
    return out


def check(S, bm, S1=None):
    S1 = S1 if S1 is not None else P1.build(BarMap("A"))
    return (K.check_notes(S.used(), bm.bars * 4) + K.check_rates(S.used()) + walk_problems(S, bm)
            + build_problems(S, S1, bm) + preservation_problems(S1, S, bm))


if __name__ == "__main__":
    bm_ = BarMap("AP2")
    S_ = build(bm_)
    parts = S_.used()
    print(f"AP2: {len(parts)} parts, {sum(len(p.notes) for p in parts.values())} notes, {len(S_.sync)} sync points; "
          f"the walk (pass-1 beat, AP2 beat, level factor): {S_.walk_moves}")
    w = check(S_, bm_)
    print(f"page checks: {len(w)} warning(s)")
    for x in w[:80]:
        print("  " + x)
