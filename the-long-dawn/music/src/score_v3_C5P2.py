"""THE LONG DAWN v3 - C5 score PASS 2 (SOUND-SCORE-C, 29 Sep): pass 1 re-synchronised to the MEASURED picture.

    python barmap_c5p2.py                 # (re)generate v3/barmap_C5P2.json + cues_C5P2.json from the measured table
    python score_v3_C5P2.py               # page checks, incl. the hard silence and the measured ignitions
    python render_v3.py C5P2              # -> out/v3/final_C5P2.wav (+ premaster_score_final_C5P2.npy); pass 1's
                                          #    final_C5 is untouched (a separate cut id, separate caches and outputs)

NOT RENDERED ON THE NIGHT IT WAS WRITTEN: the machine had no VSCO-2-CE samples, no SFZ mappings and no `soxr` (every
part here is a VSCO sampler part, and even build() reads each sample's measured attack). Its timing is proven on the
note data by music/tests/test_c5_pass2.py; its sound, balance and feeling are unheard and unapproved.

Pass 1 (score_v3_C5.py, COMPOSER-C2) is the music. Pass 2 changes only what the delivered frames contradict, plus the
one requested addition. Everything else calls pass 1's (and score_v3_C's) own functions unchanged, and keeps its ride,
anticipation table, headroom fader, trim, level bands and rules.

  1. TWO VOICES AT THE FIRST FIRES (C13, 2880, measured: both glows peak on the shot's first frame). Pass 1 had one
     far horn's CALL echoing off the ranges. Now both fires get the CALL on the same beat: hers near and centred (hn3,
     D4 A4 D5), the rival's far and to the right, where its fire is in the picture (hn_far, the same CALL a fifth
     lower, A3 E4 A4). The same figure in two places, neither leading; pass 1's second and third echoes follow as
     before. The ANSWER motif stays reserved for beacon 2 of the run (the page rule).
  2. THE HOLDOUT RESOLVES ON THE MEASURED CATCH. The last beacon catches 3786-3791 (first visible, full), not 3760.
     Pass 1's plan is kept chord for chord (F, Dm, C, F/A; Gm7, Eb maj7; A7sus4; D) but anchored on the picture: the
     held A7sus4 now spans exactly the measured pause (one kingdom dark: 3722 to the catch), the two darkening chords
     lead into it by pass 1's own 2 + 2 beats, and the D arrives when the holdout's flame is full (all_lit, 3791).
     The far horns' answers move onto the 3rd and 4th measured catches (3523, 3584; pass 1: 3520, 3560).
  3. HARD SILENCE 3848 THROUGH 3999. Pass 1 cut the swell with the breath at 3848, which ducks the dry sound to
     -45 dB and cuts the hall, but its notes ran on to 3848 + 0.25 beats and released for up to 0.5 s under that
     floor (a residue at -45 dB until ~3865), and the Ring's entry at 4000 was anticipated by each quiet-string
     sample's measured attack, capped at 0.52 s (score_v3_C.ANTIC_HI): up to 12 frames into the silence, and
     humanised. Pass 2 ends every note sounding at the shutdown so that its 30 ms release is over by 3848 even at the
     largest humanising shift (the breath stays and still cuts the hall), and starts the Ring's entry ON the cut
     (4000 + 0.02 beats, no anticipation, no humanising), so the strings' bows now bloom AFTER the cut, late by their
     attack (unheard). check() proves that no note sounds anywhere in [3848, 4000): releases, one-shot lengths,
     pre-roll and worst-case anticipation included. It cannot prove the rendered file: verify_c5_render.py does that.
  4. THE TRAP ON ITS MEASURED FRAMES. The one forge sinks 2423 (pass 1: 2400), the others surge 2483 (2440), the low
     forge returns 2546 (2480), and TWO front-runners climb 2588-2625 (one tower, 2560). The ostinato, tremolo, the
     lone horn's refusal and the corrupted calls follow the events as pass 1 wrote them; two gestures would no longer
     fit their windows and are fitted, proportions kept: the returning horn's (1, 1, 2) call into the 2.1 beats before
     the climb, and the violins' climb (D Eb F Ab .. A) onto the climb itself, its A arriving as the front-runners
     reach the Ring (2625) and held into the cut to black.
  5. The Ring's storm (storm_gone) is re-timed to the measured 4181. The refusal's two cues (2120, 2200) lie inside
     the measured drawing of the offering hand and the raised hand and are kept (the cor anglais' V-i still lands on
     2280).
"""
import kit_v3 as K
import score_v3_C as C1
import score_v3_C5 as P1
from dsl import m
from score_v3_C import dyn, hold, line, lines, voiced
from timeline_v3 import BEAT_S

SILENCE_GUARD_BEATS = 0.02         # the Ring's entry starts this far after the cut (16.7 ms > the sampler's 4 ms pre-roll)
CUT_REL_S = 0.03                   # the release of a note cut on the shutdown (the breath's own 30 ms fade-down)
# pass 1's absolute anticipations, with its resolution entries (last_beacon + 0.5) moved onto the measured all_lit
ANTIC_SET5 = {k: v for k, v in P1.ANTIC_SET5.items() if k[0] != ("last_beacon", 0.5)}
ANTIC_SET5.update({("all_lit", pn): 0.05 for pn in ("cb", "vc", "vla", "vln2", "vln1", "hn", "hn2", "hn3", "tbn",
                                                     "tuba")})
# render 1 of pass 2's battery (2026-09-29), in seconds added to one entry's anticipation. A shift moves a note's start,
# never its sample or level, so its arrival moves with it.
#   one_dark vln1: an attack onto a plateau, arriving 2.74 frames EARLY (verify_c5_render.py): its lateness, -114 ms.
#   pulls_ahead vln2: the climb's lower octave, arriving 2.18 frames early: -91 ms (pass 1's shared +0.20 s for the
#     climb left the two octaves 4.9 frames apart).
#   pulls_ahead vln1: a SWELL (-63 to -45 dB across its 0.44 s D5), so its "arrival" depends on the threshold: -51 ms
#     at -8 dB under the peak (analyze_v3's arrivew, +-50 ms) and +114 ms at -6 dB (verify_c5_render.py); the frame
#     already lies between the two, and 10 ms later brings the render's own probe inside its tolerance.
ANTIC_DT5P2 = {("pulls_ahead", "vln1"): -0.010, ("pulls_ahead", "vln2"): -2.18 / 24,
               ("one_dark", "vln1"): -2.74 / 24}
# the dawn's ANSWER: the solo violin (svln, D6) above the octave violins is a BLOOM out of their arrival, not a second
# arrival. Pass 1 chased its crescendo with anticipation (0.23 s, then 0.325 s after render 4's +95 ms) and so moved
# its real attack ever earlier: in render 1 of pass 2 it sounds from -280 ms, on a plateau from -160 ms, while the
# octave violins arrive at +26 ms, and the arrivew probe (8 dB under a peak 0.8 s into a 10 dB crescendo) still read
# +170 ms. Its onset now sits on the beat (the kit's bloom rule, 0.05 s) and it is probed as a bloom.
# the ride (the players play softer; the sampler's layers follow): pass 1's, and C13. Pass 2's Reveal gives the two
# first fires two voices at once, and render 1 of pass 2 measured the section at -3.4 LU under the film's loudest
# (score) and -2.3 (with its effects) against a band of -24..-6: a night, pp, two fires far apart. -5 dB of dynamics
# (render 3: -4 left the score's loudest 3 s at -16.0 LUFS, the band's edge on its own) with the two ignitions
# trimmed in sound_c5_table.py brings the section into its band, the voices' balance unchanged.
RIDE = dict(P1.RIDE, C13=-5.0)
SVLN_ANSWER = ("sunrise", 8)
ANTIC_SET5[(SVLN_ANSWER, "svln")] = 0.05


# ---------------------------------------------------------------------------
# C11 THE TRAP (pass 1's, on the measured frames; two gestures fitted to their windows)
# ---------------------------------------------------------------------------
def trap(S, bm):
    ev = bm.ev
    t0, low, surge, back, ahead, reach, black = (ev("trap"), ev("low_fire"), ev("surge"), ev("flares_back"),
                                                 ev("pulls_ahead"), ev("leaders_reach"), ev("flint_black"))
    cut = black + 0.1
    pat = ["D3", "D3", "D3", "Ab3", "D3", "D3", "Ab3", "D3"]
    for a, b in ((t0, low), (surge, black)):
        t = a
        k = 0
        while t < b - 0.01:
            acc = 1 if k % 4 == 0 else 0
            v = 0.34 + 0.1 * acc + 0.1 * (t - t0) / (black - t0)
            S.P("vc_sp").n(pat[k % 8], t, 0.5, v)
            if k % 2 == 0:
                S.P("cb_sp").n(m(pat[k % 8]) - 12, t, 0.5, v - 0.04)
            t += 0.5
            k += 1
    for a, b in ((t0, low), (surge, cut)):
        voiced(S, [(a, {"vla_trem": ["D4", "Ab4"], "vln_trem": ["D5", "Ab5"]})], b)
    dyn(S, ("vla_trem", "vln_trem"), (t0 - 0.1, 0.22), (low - 0.3, 0.36), (low, 0.05), (surge, 0.3),
        (ahead, 0.44), (black - 0.1, 0.52))
    for pn, root in (("tbn", "D3"), ("tbn2", "D2")):
        line(S, pn, [(root, 1.0), (m(root) + 6, 1.0), (m(root) + 12, 1.9)], t0)
    lines(S, ("hn", "hn2"), [("D4", 1.0), ("Ab4", 1.0), ("D5", 0.9)], t0 + 1.0)
    S.sync.append((t0 * BEAT_S, "THE TRAP: the corrupted call in the trombones (D3)", "tbn", 0.2, "arrive:50"))
    line(S, "hn3", [("A4", 1.0), ("G4", 0.5), ("F4", 0.6)], low)
    S.sync.append((low * BEAT_S, "one forge sinks (measured): a lone horn tries the refusal (A4)", "hn3", 0.25,
                   "arrive:69"))
    hold(S, "cb_q", "D2", low - 0.1, surge + 0.3)
    dyn(S, "cb_q", (low - 0.2, 0.1), (low + 1, 0.12), (surge + 0.3, 0.03))
    for pn, root in (("tbn", "Eb3"), ("tbn2", "Eb2")):
        line(S, pn, [(root, 1.0), (m(root) + 6, 1.0), (m(root) + 12, 1.9)], surge, sync_first=False)
    # PASS 2: the returning forge's horn call, pass 1's (1, 1, 2), fitted into the measured return -> climb window
    k = (ahead - back) / 4.0
    lines(S, ("hn", "hn2", "hn3"), [("D4", 1.0 * k), ("Ab4", 1.0 * k), ("D5", 2.0 * k)], back)
    S.sync.append((back * BEAT_S, "the low forge flares back (measured): its horn joins the race (D4)", "hn3", 0.25,
                   "arrive:62"))
    # PASS 2: the two front-runners climb (measured 2588 -> 2625): the violins' climb D Eb F Ab spans it, and the A
    # arrives as they reach the Ring, held into the cut
    kc = (reach - ahead) / 3.5
    lines(S, ("vln1", "vln2"), [("D5", kc), ("Eb5", kc), ("F5", kc), ("Ab5", 0.5 * kc), ("A5", cut - reach)], ahead,
          octaves=(0, -1))
    voiced(S, [(ahead, {"tbn": ["D3", "Ab3"], "tuba": ["D2"], "hn": ["D4"], "hn2": ["Ab4"]})], cut)
    S.sync.append((ahead * BEAT_S, "the two front-runners climb (measured): the violins climb (D5)", "vln1", 0.25,
                   "arrivew"))
    hold(S, "timp_roll", "D2", ahead + 0.05, cut, legato=True)
    dyn(S, "timp_roll", (ahead, 0.1), (black - 0.1, 0.28))
    dyn(S, ("tbn", "tbn2"), (t0 - 0.05, 0.42), (low - 0.2, 0.46), (low, 0.05), (surge, 0.48), (black - 0.1, 0.54))
    dyn(S, ("hn", "hn2"), (t0 + 0.9, 0.38), (low - 0.2, 0.42), (low, 0.05), (back - 0.1, 0.44), (black - 0.1, 0.52))
    dyn(S, "hn3", (low - 0.1, 0.3), (low + 1.5, 0.3), (surge, 0.24), (back - 0.1, 0.42), (black - 0.1, 0.5))
    dyn(S, "tuba", (ahead - 0.1, 0.32), (black - 0.1, 0.5))
    dyn(S, ("vln1", "vln2"), (ahead - 0.1, 0.32), (black - 0.1, 0.52))


# ---------------------------------------------------------------------------
# C13 THE REVEAL: two voices at the two first fires
# ---------------------------------------------------------------------------
NEAR, FAR = "hn3", "hn_far"          # her fire: near, centre-left; the rival's: far, right (as in the picture)


def first_fires(S, bm):
    rev = bm.ev("reveal")
    for pn, root, v in ((NEAR, "D4", 0.40), (FAR, "A3", 0.46)):
        line(S, pn, [(root, 1.0), (m(root) + 7, 1.0), (m(root) + 12, 2.2)], rev)
        S.P(pn).d((rev - 0.05, v), (rev + 1.5, v * 1.02), (rev + 3.2, v * 0.9), (rev + 4.3, 0.03))
        S.sync.append((rev * BEAT_S, f"THE REVEAL: two first fires, together: the CALL on {root} ({pn})", pn, 0.25,
                       f"arrive:{m(root)}"))
    # pass 1's second and third echoes off the ranges, farther each time
    for k, (pn, v) in enumerate((("hn_farther", 0.44), ("hn_far2", 0.42)), start=1):
        t = rev + 4 * k
        line(S, pn, [("D4", 1.0), ("A4", 1.0), ("D5", 2.2)], t)
        S.P(pn).d((t - 0.05, v), (t + 1.5, v * 1.02), (t + 3.2, v * 0.9), (t + 4.3, 0.03))
        S.sync.append((t * BEAT_S, f"the call echoes off the ranges ({k + 1})", pn, 0.25, "arrive:62"))


# ---------------------------------------------------------------------------
# C15 THE LAST BEACON on the measured map (and C16's silence)
# ---------------------------------------------------------------------------
def last_beacon(S, bm):
    ev = bm.ev
    mp, sus, ham, last, lit, cold = (ev("map"), ev("one_dark"), ev("hammer_alone"), ev("last_beacon"), ev("all_lit"),
                                     ev("forges_cold"))
    dark = sus - 4.0                   # pass 1's darkening, 2 + 2 beats, leads into the measured pause
    off = cold + 0.25
    H = [(mp, {"cb": ["F2"], "vc": ["C3", "F3"], "vla": ["A3", "C4"], "vln2": ["F4"], "vln1": ["A4"]}),
         (mp + 2, {"cb": ["D2"], "vc": ["A2", "F3"], "vla": ["A3", "C4"], "vln2": ["F4"], "vln1": ["A4"]}),
         (mp + 4, {"cb": ["C2"], "vc": ["G2", "E3"], "vla": ["G3", "C4"], "vln2": ["E4"], "vln1": ["G4"]}),
         (mp + 6, {"cb": ["A1"], "vc": ["F2", "C3"], "vla": ["F3", "C4"], "vln2": ["F4"], "vln1": ["A4"]}),
         (dark, {"cb": ["G1"], "vc": ["D2", "F3"], "vla": ["Bb3"], "vln2": ["D4"]}),
         (dark + 2, {"cb": ["Eb2"], "vc": ["Bb2", "G3"], "vla": ["Bb3", "D4"], "vln2": ["G4"]}),
         (sus, {"cb": ["A1"], "vc": ["A2", "E3"], "vla": ["G3", "D4"], "vln2": ["E4"], "vln1": ["A4"]}),
         (lit, {"cb": ["D2"], "vc": ["A2", "D3"], "vla": ["F#3", "A3"], "vln2": ["D4", "F#4"], "vln1": ["A4", "D5"]})]
    voiced(S, H, off, sync_first=True)
    for pn, a in (("cb", 0.36), ("vc", 0.38), ("vla", 0.37), ("vln2", 0.36), ("vln1", 0.36)):
        S.P(pn).d((mp - 0.05, a), (mp + 4, a * 1.02), (dark - 0.3, a * 0.9), (dark + 1, a * 0.62), (sus, a * 0.5),
                  (ham, a * 0.36), (last, a * 0.36), (lit + 0.7, a * 0.75), (cold - 0.1, a * 1.1), (off, a * 1.1))
    S.sync.append((mp * BEAT_S, "THE LAST BEACON: the run's C lands in F on the map (strings)", "vla", 0.2, "bloom"))
    S.sync.append((lit * BEAT_S, "every beacon lit (measured): the held A7sus4 resolves into D (vla G3-F#3)", "vla",
                   0.25, "pitch:54"))
    for pn in ("ob_c", "ob_dbl"):
        K.answer(S, pn, "F5" if pn == "ob_c" else "F4", mp, mode="major", rhythm=(1, .5, .5, 2))
    S.P("ob_c").d((mp - 0.05, 0.3), (mp + 2, 0.32), (mp + 4, 0.1), (mp + 4.3, 0.03))
    S.P("ob_dbl").d((mp - 0.05, 0.26), (mp + 2, 0.28), (mp + 4, 0.1), (mp + 4.3, 0.03))
    S.sync.append((mp * BEAT_S, "the near beacons answer: the oboe (F5)", "ob_c", 0.2, "arrive:77"))
    for pn, top, t, v in (("hn_far", "C5", ev("map_beacon_3"), 0.36), ("hn_farther", "F4", ev("map_beacon_4"), 0.32)):
        K.answer(S, pn, top, t, mode="major", rhythm=(1, .5, .5, 1.8))
        S.P(pn).d((t - 0.05, v), (t + 2, v * 1.02), (t + 3.8, 0.03))
        S.sync.append((t * BEAT_S, f"a far beacon catches (measured) and answers ({pn}, {top})", pn, 0.25,
                       f"arrive:{m(top)}"))
    voiced(S, [(lit, {"hn": ["D4"], "hn2": ["F#4"], "hn3": ["A4"], "tbn": ["A3"], "tuba": ["D2"]})], off)
    dyn(S, ("hn", "hn2", "hn3", "tbn", "tuba"), (lit - 0.1, 0.1), (lit + 1.0, 0.25), (cold - 0.1, 0.4), (off, 0.4))
    hold(S, "timp_roll", "D2", lit + 0.5, off, legato=True)
    dyn(S, "timp_roll", (lit + 0.4, 0.08), (cold - 0.1, 0.3), (off, 0.3))


# ---------------------------------------------------------------------------
# C16: the swell stops dead on the shutdown
# ---------------------------------------------------------------------------
def end_on_the_cut(S, bm, patches=None):
    """every sampler note still sounding at the shutdown (3848) is shortened so that its sound is over BY 3848 at the
    largest humanising shift: a held note gets a 30 ms release, a one-shot a length that ends there (the sampler fades
    its last 60 ms). Pass 1's notes ran on under the breath's -45 dB floor; here the floor has nothing left to hide.
    Returns [(part, beat, old end beat, new end beat)]. Runs after the anticipation tables (it moves ends, not starts)."""
    cold = bm.event("forges_cold")["t"]
    ring = bm.event("ring_unfinished")["t"]
    done = []
    for pn, p in S.parts.items():
        if p.kind != "sampler":
            continue                  # a synth voice's tail is not modelled here: check() flags one, never hides it
        hum = 1.5 * p.humanize_ms / 1000.0
        for n in p.notes:
            if n.pitch is None:
                continue
            lo, hi = sounding(S, pn, n, patches)
            if not (lo < cold < hi) or lo >= ring:
                continue
            t = n.start * BEAT_S
            j = 0.0 if n.sync else hum
            key = n.art or p.inst
            pp = (patches or {}).get(key) or {}
            kind = n.kw.get("kind", pp.get("kind", "sus"))
            old_end = n.start + n.dur
            if kind == "one":
                n.kw["maxlen"] = max(0.02, cold - t - j)
            else:
                n.kw["rel"] = CUT_REL_S
                n.dur = max(0.02, (cold - t - j - CUT_REL_S) / BEAT_S)
            done.append((pn, round(n.start, 4), round(old_end, 4), round(n.start + n.dur, 4)))
    return done


# ---------------------------------------------------------------------------
# C16/C17: the Ring's entry starts ON the cut, never inside the silence
# ---------------------------------------------------------------------------
def pin_after_silence(S, bm):
    """after anticipation: every note that starts on the Ring's cut begins just after it, unanticipated and
    unhumanised, so the silence 3848-3999 holds to its last frame"""
    r0 = bm.ev("ring_unfinished")
    moved = []
    for pn, p in S.parts.items():
        for n in p.notes:
            if n.pitch is not None and abs(n.start - r0) < 1e-6:
                n.start = r0 + SILENCE_GUARD_BEATS
                n.dur = max(0.05, n.dur - SILENCE_GUARD_BEATS)
                n.kw["antic"] = 0.0
                n.sync = True
                moved.append(pn)
    return moved


# ---------------------------------------------------------------------------
def build(bm):
    S = K.Score("C5P2", bm)
    C1.setup(S)
    for fn in (C1.storyteller, C1.mountain, C1.letters_to_fire, C1.fire_alone, C1.forging, C1.race, C1.deep, C1.eye):
        fn(S, bm)
    for nt in S.P("tam").notes:
        nt.kw["rel"] = P1.TAM_BLOOM_S
    for fn in (P1.refusal, trap, P1.flint, first_fires, P1.reveal_pad, C1.run, last_beacon, P1.ring_unfinished,
               P1.deep_abandoned, P1.watch, C1.illumination, C1.plenty, C1.last_pages):
        fn(S, bm)
    end = bm.bars * 4
    for pn in ("hn", "hn2", "hn3", "hn_far", "hn_farther", "hn_far2", "hn_ans", "hn_st", "hn_st2", "ca", "ca_dbl",
               "cl_c", "ob_c", "bsn_c", "svln"):
        K.breathe(S, pn, 0, end, depth=0.1, min_dur=1.4)
    K.ride(S, bm, RIDE)
    for pn in ("ca", "ca_dbl", "cl_c", "cl_dbl", "ob_c", "ob_dbl", "bsn_c", "hn", "hn2", "hn3", "hn_st", "hn_st2",
               "hn_far", "hn_farther", "hn_far2", "hn_ans", "tbn", "tbn2", "tuba", "svln", "svln2", "vla_q",
               "vc_ring", "line_vla", "line_vc", "vla", "vc_trem", "vln1", "vc_q", "cb_q", "vln1_q", "vln2_q", "vc", "cb",
               "vln2"):
        K.anticipate(S, pn, hi=C1.ANTIC_HI.get(pn, 0.25))
    sun = bm.ev("sunrise")
    for pn in ("vln1", "svln", "svln2"):
        for n in S.P(pn).notes:
            if n.legato and sun <= n.start < bm.ev("plenty"):
                n.kw["antic"] = 0.02
    have = {e["id"] for e in bm.events}
    for (eid, pn), a in C1.ANTIC_SET.items():
        e = eid if isinstance(eid, str) else eid[0]
        if e not in have or e in ("map",):
            continue
        b0 = bm.ev(eid) if isinstance(eid, str) else bm.ev(eid[0]) + eid[1]
        for n in S.P(pn).notes:
            if abs(n.start - b0) < 1e-6:
                n.kw["antic"] = a
    for (eid, pn), a in ANTIC_SET5.items():
        b0 = bm.ev(eid) if isinstance(eid, str) else bm.ev(eid[0]) + eid[1]
        for n in S.P(pn).notes:
            if abs(n.start - b0) < 1e-6:
                n.kw["antic"] = a
    for (eid, pn), dt in list(P1.ANTIC_DT5.items()) + list(ANTIC_DT5P2.items()):
        hit = [n for n in S.P(pn).notes if abs(n.start - bm.ev(eid)) < 1e-6 and not n.legato]
        if (eid, pn) in ANTIC_DT5P2 and len(hit) != 1:
            raise ValueError(f"ANTIC_DT5P2 {eid}/{pn}: {len(hit)} attacked notes on the event, not 1")
        for n in hit:
            n.kw["antic"] = max(0.02, n.kw.get("antic", 0.0) + dt)
    # the dawn's ANSWER on the solo violin is probed as the bloom it now is (SVLN_ANSWER, above)
    t_ans = (bm.ev(SVLN_ANSWER[0]) + SVLN_ANSWER[1]) * BEAT_S
    hit = [k for k, x in enumerate(S.sync) if x[2] == "svln" and abs(x[0] - t_ans) < 1e-6]
    if len(hit) != 1:
        raise ValueError(f"the solo violin's ANSWER: {len(hit)} sync entries at {t_ans:.3f} s, not 1")
    x = S.sync[hit[0]]
    S.sync[hit[0]] = (x[0], x[1] + ": a bloom, its onset on the beat", x[2], x[3], "bloom")
    S.pinned = pin_after_silence(S, bm)
    S.cut_at_shutdown = end_on_the_cut(S, bm, _patches())
    for p in S.parts.values():
        p.gain_db += C1.SCORE_TRIM_DB
    for s in bm.sections:
        if s.get("rel"):
            S.levels.append((s["id"], *s["rel"]))
    t = lambda k: bm.event(k)["t"]                       # noqa: E731
    S.rules = [("C's loudest is the slit: everything after the Eye at least 1 LU under the film's loudest",
                t("mirror"), bm.seconds, -1.0),
               ("the dawn is warm, not loud: at least 2 LU under the slit", t("sunrise"), t("plenty"), -2.0),
               ("the book's prologue is quiet: C1-C4 at least 6 LU under the slit", 0.0, t("fire_alone"), -6.0)]
    S.fader = P1.CLIMAX_FADER
    return S


# ---------------------------------------------------------------------------
# the page checks: pass 1's, the measured ignitions, and the hard silence
# ---------------------------------------------------------------------------
IGNITIONS = P1.IGNITIONS + ("reveal", "map_beacon_2", "map_beacon_3", "map_beacon_4", "map_beacon_5", "map_beacon_6",
                            "map_beacon_7")
PRE_S = 0.004                           # sampler.PRE: the pre-roll kept before a sample's detected onset
SYNTH_TAIL_S = 3.0                      # a synth voice's tail, when its own release is not given


def sounding(S, pn, n, patches=None):
    """(first, last) second at which note n of part pn can make sound: its anticipation, legato crossfade, pre-roll
    and humanising before; its release (a held patch) or its whole natural decay (a one-shot) after. patches is
    sampler.P (None: the conservative defaults)"""
    p = S.P(pn)
    t = n.start * BEAT_S
    hum = 0.0 if n.sync else 1.5 * p.humanize_ms / 1000.0
    dur = n.dur * BEAT_S
    if p.kind != "sampler":
        rel = float(n.kw.get("rel", SYNTH_TAIL_S))
        return t - hum, t + dur + max(rel, SYNTH_TAIL_S) + hum
    key = n.art or p.inst
    pp = (patches or {}).get(key) or dict(kind="sus", rel=0.8, antic=0.06, lfade=0.1, maxlen=None)
    antic = float(n.kw.get("antic", pp.get("antic", 0.0)))
    if n.legato:
        antic += 0.5 * float(n.kw.get("lfade", pp.get("lfade", 0.07)))
    lo = t - antic - PRE_S - hum
    kind = n.kw.get("kind", pp.get("kind", "sus"))
    rel = float(n.kw.get("rel", pp.get("rel", 0.35)))
    if kind == "sus":
        hi = t + dur + rel + hum
    elif kind == "pno":
        hi = t + min(float(pp.get("maxlen") or 30.0), dur + rel) + hum
    else:
        hi = t + float(n.kw.get("maxlen", pp.get("maxlen") or 30.0)) + hum
    return lo, hi


def _patches():
    """sampler.P with the v2 patches registered (the quiet strings: vla_q, vc_q, cb_q...), or None if the sampler
    cannot be imported (then sounding() uses conservative defaults and check() says so)"""
    import importlib
    try:
        sampler = importlib.import_module("sampler")
        importlib.import_module("sampler_v2")              # registers the quiet strings etc.
        return sampler.P
    except ImportError:
        return None


def silence_window(bm):
    """(start, end) s of the hard silence: the shutdown (3848) to the Ring's cut (4000). The shutdown's breath
    (3848 - 0.03 s .. + 1.2 s) must exist: it cuts the hall's tail, which no note list can do"""
    cold, ring = bm.event("forges_cold")["t"], bm.event("ring_unfinished")["t"]
    if not any(abs(b0 * BEAT_S - (cold - 0.03)) < 0.01 for b0, _, _ in bm.breath_beats()):
        raise ValueError("no breath at forges_cold: the hall would ring on into the silence")
    return cold, ring


def silence_problems(S, bm, patches=None):
    t_cold, t_ring = silence_window(bm)
    out = []
    for pn, p in S.used().items():
        for n in p.notes:
            if n.pitch is None:
                continue
            lo, hi = sounding(S, pn, n, patches)
            if t_cold - 0.03 <= lo < t_ring:
                out.append(f"STARTS IN THE SILENCE: {pn} at beat {n.start:.3f} (sound from {lo * 24:.1f} f)")
            elif lo < t_ring and hi > t_cold + 1e-9:
                out.append(f"SOUNDS INTO THE SILENCE: {pn} at beat {n.start:.3f} ({lo * 24:.1f}-{hi * 24:.1f} f)")
    return out


def check(S, bm, patches=None):
    out = []
    for k in IGNITIONS:
        b = bm.ev(k)
        for pn in C1.STROKES:
            if pn in S.parts:
                for n in S.P(pn).notes:
                    if abs(n.start - b) < 0.07:
                        out.append(f"STROKE ON {k}: {pn} at beat {n.start:.2f}")
    b2, sun = bm.ev("beacon_2"), bm.ev("sunrise")
    for pn, p in S.parts.items():
        ns = sorted((n for n in p.notes if n.pitch is not None), key=lambda n: n.start)
        ps = [int(round(n.pitch)) for n in ns]
        for i in range(len(ns) - 3):
            iv = tuple(ps[i + j] - ps[i] for j in range(4))
            if iv in ((0, -1, -3, -8), (0, -2, -4, -9)) and ns[i].start < b2 - 1e-6:
                out.append(f"ANSWER BEFORE BEACON 2: {pn} at beat {ns[i].start:.2f}")
            if iv == (0, -2, -5, -9) and ns[i].start < sun - 1e-6:
                out.append(f"HOME BEFORE THE DAWN: {pn} at beat {ns[i].start:.2f}")
    a, b = bm.ev("line_in"), bm.ev("blank")
    for pn, p in S.parts.items():
        for n in p.notes:
            if n.pitch is not None and a - 1e-6 <= n.start < b - 0.3:
                out.append(f"NOTE IN THE LINE'S WINDOW: {pn} at beat {n.start:.2f}")
    if patches is None:
        patches = _patches()
        if patches is None:                                # never pass the silence on guessed releases silently
            out.append("SILENCE CHECK USED DEFAULT RELEASES: sampler not importable")
    out += silence_problems(S, bm, patches)
    out += two_voice_problems(S, bm)
    return out


def two_voice_problems(S, bm):
    """both first fires get a voice: a sounding, attacked note in NEAR and in FAR on the reveal"""
    rev = bm.ev("reveal")
    voices = sorted(pn for pn in (NEAR, FAR) if pn in S.parts and any(
        n.pitch is not None and abs(n.start - rev) < 1e-6 and not n.legato for n in S.P(pn).notes))
    if voices != sorted((NEAR, FAR)):
        return [f"TWO VOICES AT THE FIRST FIRES: only {voices} enter on the reveal"]
    return []


if __name__ == "__main__":
    from timeline_v3 import BarMap
    bm_ = BarMap("C5P2")
    S_ = build(bm_)
    parts = S_.used()
    print(f"C5P2: {len(parts)} parts, {sum(len(p.notes) for p in parts.values())} notes, {len(S_.sync)} sync points; "
          f"pinned after the silence: {S_.pinned}; cut on the shutdown: {len(S_.cut_at_shutdown)} notes")
    w = K.check_notes(parts, bm_.bars * 4) + K.check_rates(parts) + check(S_, bm_)
    print(f"rule checks: {len(w)} warning(s)")
    for x in w[:80]:
        print("  " + x)
