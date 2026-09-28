"""THE LONG DAWN v3 - C5 . THE LAST PAGES, script v5.2: the score (owner: COMPOSER-C2).

    source ~/.venvs/longdawn/env.sh && cd music/src
    python ~/mishamisha/_local_logs/renderq.py -- python render_v3.py C5      # -> out/v3/final_C5.wav + battery
    python score_v3_C5.py                                                     # page checks

v5.2 (the user approved the structure on 28 Sep) rebuilds the second half as a story about coordination. There is
no Ring fall, no heroine arc and no council. Each smith lights a beacon as a conditional promise, and when the LAST
beacon catches, every forge goes cold at once. The film is 5920 frames (74 bars). Its bar map is
music/v3/barmap_C5.json and its cue sheet music/v3/cues_C5.json (made from the v5 timeline; frames marked (est.)
are musical estimates until the picture lands). The effects are SOUND-C's, so this render carries none. The v1
film's score_v3_C.py is untouched: bars 1-26 (C1-C9) are ITS functions, called unchanged.

The user's notes on C's score, and how this answers them:
  (1) TURN when the story turns: every shot of the new half has its own material and orchestration, and nothing
      carries over a mood change (the book's chamber music, the race, black, the night, the map, silence, the
      Ring, the still mine, the watch, the dawn).
  (2) REAL CADENCES: the refusal closes V-i in D minor; the reveal ends on the dominant, the promise's "if",
      and the run answers it on D; the last beacon's held A7sus4 resolves into D; the Deep's descent closes V-i;
      the watch's chorale closes on A into the dawn's D; the hymn and the plagal close stay as they were.
  (3) NOTHING CARTOON-LIGHT: no glass, no glitter, no falling figures, no synthesised colours in the new half.
      The Ring's end is stillness: its motif once, on stopped horns, never finished.
  (4) KEEP WHAT'S GOOD: the first half as it was (plus the climax's headroom fix, a slow premaster fader
      ride), the call echoing off the ranges, the Living Ink Run, the dawn's whole theme, the hymn, and the close.

THE PLAN (bar.beat; C1-C9 as score_v3_C)
  C10 27-29  THE REFUSAL: the book again. The storyteller's pad (low quiet strings, D minor). The old story's Ring
             is held out on the clarinet (D Ab D' C' Ab, pp). The cor anglais turns away from it: A G F E D, over
             Gm/Bb - A - Dm, a real cadence (29 b3). The harp closes the page.
  C11 30-33  THE TRAP: the race, leaving room for SOUND's hammers (sustained tremolo, a low 8th ostinato, brass;
             no metal, no timpani strokes). One fire sinks low (31): the race drops out and one horn tries the
             refusal (A G F). The others surge (31 b3); it is swallowed, and the horn joins the race (32). One tower
             pulls ahead (33): the violins climb to within a semitone of the Ring, then a cut to black (34 b1).
  C12 34-36  FLINT: silence, then the night's low D (ppp) under the strikes and the long breath. The fire catches
             (effects): a warm D major opens after it, with no stroke.
  C13 37-39  THE REVEAL / A PROMISE: the call echoes off the ranges, farther each time (as before), over
             D - G/D - Asus4 - A: the promise's "if", a half cadence.
  C14 40-43  THE BEACON RUN (as before; the ANSWER first heard on beacon 2).
  C15 44-48  THE LAST BEACON: the run's C lands in F on the map; the beacons answer near (oboe) and far (horns).
             One kingdom stays dark (46): the harmony darkens (Gm7, Eb maj7) and thins. The held pause (47-48):
             A7sus4, still, under the lone hammer. The last beacon catches (48 b1, effects): the sus resolves into
             D and the orchestra swells ... and at 3848 every forge goes dark: HARD SILENCE (a breath cuts the hall).
  C16 49-50  THE FORGES GO COLD: silence.
  C17 51-53  THE RING, UNFINISHED: the Ring motif once, slow, on stopped horns (a2), and it never finishes:
             D Ab D' C' ... hanging on the C as the glow drains, over a low D and the Ab fading out. A pause, not
             a victory.
  C18 54-56  THE DEEP, ABANDONED: bassoon and cellos descend at rest (D C Bb A G F E, A, D), a lament bass
             closing V-i in D minor: they left the gold in the ground.
  C19 57-59  THE WATCH: a pianissimo brass chorale, steady (D Bm G Em7 Asus4 A), far horns calling from the peaks.
             It closes on A into the 275 ms breath.
  C20 60-65  THE ILLUMINATION (as before: the bloom on D add9, the whole theme, home on bar 64).
  C21 66-68  THE YEAR OF PLENTY (as before: the hymn).
  C22 69-71  THE LAST PAGES: 5460-5540 is left clean for a spoken line; after it the chorale's Bb/D returns.
  C23 72-74  TITLE: the plagal close into D (74 b1), ringing into silence.
"""
import kit_v3 as K
import score_v3_C as C1
from dsl import m
from score_v3_C import dyn, hold, line, lines, voiced
from timeline_v3 import BEAT_S

TAM_BLOOM_S = 3.2          # the slit's tam-tam bloom ends before the hearth flares (v1: 6 s, into the Mirror)
# the headroom fix (A/B pair 2): a slow premaster fader into the slit, (t_s, dB). The master's gain reduction at
# the slit falls from comp 5.0 + lim 5.6 dB to 2.7 + 0.9 dB
CLIMAX_FADER = [(79.0, 0.0), (80.5, -1.0), (81.6, -2.5), (82.6, -5.0), (83.25, -7.0), (84.6, -6.5), (85.4, -3.0),
                (86.6, -0.5), (87.6, 0.0)]
RIDE = {"C6": -1.5, "C7": -1.0, "C9": -1.0, "C14": -3.0, "C20": -0.5}   # v1's, on the v5 section ids


# ---------------------------------------------------------------------------
# C10 THE REFUSAL
# ---------------------------------------------------------------------------
def refusal(S, bm):
    ev = bm.ev
    r0, offer, turn, trap = ev("refusal"), ev("old_story"), ev("turns_away"), ev("trap")
    cad = r0 + 10
    end = trap - 0.35
    voiced(S, [(r0, {"cb_q": ["D2"], "vc_q": ["A2", "D3"], "vla_q": ["F3"]}),
               (turn, {"cb_q": ["Bb1"], "vc_q": ["F2", "D3"], "vla_q": ["G3"]}),
               (r0 + 8, {"cb_q": ["A1"], "vc_q": ["E2", "C#3"], "vla_q": ["E3", "A3"]}),
               (cad, {"cb_q": ["D2"], "vc_q": ["A2", "D3"], "vla_q": ["F3", "A3"]})], end, sync_first=True)
    dyn(S, ("cb_q", "vc_q", "vla_q"), (r0 - 0.2, 0.03), (r0 + 1.2, 0.15), (turn, 0.18), (r0 + 8, 0.2), (cad, 0.17),
        (end - 0.7, 0.09), (end, 0.02))
    S.sync.append((r0 * BEAT_S, "THE REFUSAL: the book again, the storyteller's pad (D minor)", "vla_q", 0.3, "bloom"))
    # the old story's Ring, held out: the clarinet, pp
    line(S, "cl_c", [("D4", 1.0), ("Ab4", 1.0), ("D5", 1.0), ("C5", 0.5), ("Ab4", 1.0)], offer)
    S.P("cl_c").d((offer - 0.1, 0.2), (offer + 2, 0.24), (offer + 3.5, 0.2), (offer + 4.5, 0.03))
    S.sync.append((offer * BEAT_S, "the old story's Ring held out (clarinet, D4)", "cl_c", 0.2, "arrive:62"))
    # the wise turn away: the cor anglais (clarinet in unison), A G F E ... D: the cadence
    lines(S, ("ca", "ca_dbl"), [("A4", 1.0), ("G4", 0.5), ("F4", 0.5), ("E4", 2.0), ("D4", 1.7)], turn)
    pts = [(turn - 0.1, 0.28), (turn + 1, 0.32), (turn + 2, 0.31), (cad, 0.27), (end - 0.2, 0.12), (end + 0.05, 0.03)]
    dyn(S, "ca", *pts)
    dyn(S, "ca_dbl", *[(t, max(0.02, v - 0.03)) for t, v in pts])
    S.sync.append((turn * BEAT_S, "the wise refuse: the cor anglais turns away (A4)", "ca+ca_dbl", 0.2, "arrive:69"))
    S.sync.append((cad * BEAT_S, "the refusal closes V-i: the cor anglais on D4", "ca+ca_dbl", 0.25, "arrive:62"))
    for p, dt, v in (("D3", 0.02, 0.2), ("A3", 0.14, 0.18), ("D4", 0.26, 0.18), ("F4", 0.38, 0.17)):
        S.P("harp").n(p, cad + dt, 2.0, v)


# ---------------------------------------------------------------------------
# C11 THE TRAP
# ---------------------------------------------------------------------------
def trap(S, bm):
    ev = bm.ev
    t0, low, surge, back, ahead, black = (ev("trap"), ev("low_fire"), ev("surge"), ev("flares_back"),
                                          ev("pulls_ahead"), ev("flint_black"))
    cut = black + 0.1                       # the breath at the black cuts everything dead
    # the forge's drive: a low 8th ostinato on D with the tritone (cellos spiccato), basses on the beat only;
    # nothing metallic and no timpani strokes: the hammers are SOUND's
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
    # the tremolo on the tritone (violas, violins), dropping out where the one fire sinks
    for a, b in ((t0, low), (surge, cut)):
        voiced(S, [(a, {"vla_trem": ["D4", "Ab4"], "vln_trem": ["D5", "Ab5"]})], b)
    dyn(S, ("vla_trem", "vln_trem"), (t0 - 0.1, 0.22), (low - 0.3, 0.36), (low, 0.05), (surge, 0.3),
        (ahead, 0.5), (black - 0.1, 0.62))
    # the corrupted call in the trombones, answered in canon by the horns
    for pn, root in (("tbn", "D3"), ("tbn2", "D2")):
        line(S, pn, [(root, 1.0), (m(root) + 6, 1.0), (m(root) + 12, 1.9)], t0)
    lines(S, ("hn", "hn2"), [("D4", 1.0), ("Ab4", 1.0), ("D5", 0.9)], t0 + 1.0)
    S.sync.append((t0 * BEAT_S, "THE TRAP: the corrupted call in the trombones (D3)", "tbn", 0.2, "arrive:50"))
    # one fire sinks low: the race drops out; one horn tries the refusal, alone (A G F)
    line(S, "hn3", [("A4", 1.0), ("G4", 0.5), ("F4", 0.6)], low)
    S.sync.append((low * BEAT_S, "one fire sinks low: a lone horn tries the refusal (A4)", "hn3", 0.25, "arrive:69"))
    hold(S, "cb_q", "D2", low - 0.1, surge + 0.3)
    dyn(S, "cb_q", (low - 0.2, 0.1), (low + 1, 0.12), (surge + 0.3, 0.03))
    # the others surge: the corrupted call a semitone higher, then the lone horn is swallowed into the race
    for pn, root in (("tbn", "Eb3"), ("tbn2", "Eb2")):
        line(S, pn, [(root, 1.0), (m(root) + 6, 1.0), (m(root) + 12, 1.9)], surge, sync_first=False)
    lines(S, ("hn", "hn2", "hn3"), [("D4", 1.0), ("Ab4", 1.0), ("D5", 2.0)], back)
    S.sync.append((back * BEAT_S, "the low fire flares back: its horn joins the race (D4)", "hn3", 0.25, "arrive:62"))
    # one tower pulls ahead: the violins climb toward the Ring (D Eb F Ab A), the brass hold the tritone under
    lines(S, ("vln1", "vln2"), [("D5", 1.0), ("Eb5", 1.0), ("F5", 1.0), ("Ab5", 0.5), ("A5", 0.6)], ahead,
          octaves=(0, -1))
    voiced(S, [(ahead, {"tbn": ["D3", "Ab3"], "tuba": ["D2"], "hn": ["D4"], "hn2": ["Ab4"]})], cut)
    S.sync.append((ahead * BEAT_S, "one tower pulls ahead: the violins climb (D5)", "vln1", 0.25, "arrivew"))
    hold(S, "timp_roll", "D2", ahead + 0.05, cut, legato=True)
    dyn(S, "timp_roll", (ahead, 0.12), (black - 0.1, 0.4))
    dyn(S, ("tbn", "tbn2"), (t0 - 0.05, 0.42), (low - 0.2, 0.46), (low, 0.05), (surge, 0.5), (black - 0.1, 0.66))
    dyn(S, ("hn", "hn2"), (t0 + 0.9, 0.38), (low - 0.2, 0.42), (low, 0.05), (back - 0.1, 0.46), (black - 0.1, 0.62))
    dyn(S, "hn3", (low - 0.1, 0.3), (low + 1.5, 0.3), (surge, 0.24), (back - 0.1, 0.44), (black - 0.1, 0.6))
    dyn(S, "tuba", (ahead - 0.1, 0.34), (black - 0.1, 0.6))
    dyn(S, ("vln1", "vln2"), (ahead - 0.1, 0.36), (black - 0.1, 0.66))


# ---------------------------------------------------------------------------
# C12 FLINT, C13 THE REVEAL (its pad)
# ---------------------------------------------------------------------------
def flint(S, bm):
    ev = bm.ev
    b, catch, rev = ev("flint_black"), ev("catch"), ev("reveal")
    # black and silence; then the night: a low open fifth, ppp, under the strikes and the long breath
    voiced(S, [(b + 2.0, {"cb_q": ["D2"], "vc_q": ["A2"]})], rev + 0.1)
    dyn(S, ("cb_q", "vc_q"), (b + 1.8, 0.03), (b + 4.5, 0.1), (catch, 0.12), (catch + 1.5, 0.15), (rev, 0.14))
    # the fire catches (the effects own it): a warm D major opens after it, no stroke
    voiced(S, [(catch + 0.6, {"vla_q": ["F#3", "A3"], "vln2_q": ["D4"]})], rev + 0.1)
    dyn(S, ("vla_q", "vln2_q"), (catch + 0.5, 0.03), (catch + 1.6, 0.14), (rev, 0.15))
    S.sync.append(((catch + 0.6) * BEAT_S, "the fire catches: a warm D major opens after it", "vla_q", 0.3, "bloom"))


def reveal_pad(S, bm):
    ev = bm.ev
    rev, run = ev("reveal"), ev("run")
    # under the call's echoes: D - G/D - Asus4 - A: the promise's "if" (a half cadence the run answers on D)
    voiced(S, [(rev, {"cb_q": ["D2"], "vc_q": ["A2", "D3"], "vla_q": ["F#3", "A3"], "vln2_q": ["D4"]}),
               (rev + 4, {"cb_q": ["D2"], "vc_q": ["G2", "D3"], "vla_q": ["G3", "B3"], "vln2_q": ["D4"]}),
               (rev + 8, {"cb_q": ["A1"], "vc_q": ["A2", "D3"], "vla_q": ["E3", "A3"], "vln2_q": ["D4"]}),
               (rev + 10, {"cb_q": ["A1"], "vc_q": ["A2", "C#3"], "vla_q": ["E3", "A3"], "vln2_q": ["C#4"]})],
           run - 0.15)
    dyn(S, ("cb_q", "vc_q", "vla_q", "vln2_q"), (rev + 0.2, 0.15), (rev + 4, 0.17), (rev + 8, 0.2), (rev + 10, 0.22),
        (run - 0.5, 0.14), (run - 0.1, 0.03))


# ---------------------------------------------------------------------------
# C15 THE LAST BEACON (and C16's silence)
# ---------------------------------------------------------------------------
def last_beacon(S, bm):
    ev = bm.ev
    mp, dark, ham, last, cold = ev("map"), ev("one_dark"), ev("hammer_alone"), ev("last_beacon"), ev("forges_cold")
    off = cold + 0.25                        # everything ends in the breath (the hall is cut at 3848)
    H = [(mp, {"cb": ["F2"], "vc": ["C3", "F3"], "vla": ["A3", "C4"], "vln2": ["F4"], "vln1": ["A4"]}),
         (mp + 2, {"cb": ["D2"], "vc": ["A2", "F3"], "vla": ["A3", "C4"], "vln2": ["F4"], "vln1": ["A4"]}),
         (mp + 4, {"cb": ["C2"], "vc": ["G2", "E3"], "vla": ["G3", "C4"], "vln2": ["E4"], "vln1": ["G4"]}),
         (mp + 6, {"cb": ["A1"], "vc": ["F2", "C3"], "vla": ["F3", "C4"], "vln2": ["F4"], "vln1": ["A4"]}),
         (dark, {"cb": ["G1"], "vc": ["D2", "F3"], "vla": ["Bb3"], "vln2": ["D4"]}),
         (dark + 2, {"cb": ["Eb2"], "vc": ["Bb2", "G3"], "vla": ["Bb3", "D4"], "vln2": ["G4"]}),
         (dark + 4, {"cb": ["A1"], "vc": ["A2", "E3"], "vla": ["G3", "D4"], "vln2": ["E4"], "vln1": ["A4"]}),
         (last + 0.5, {"cb": ["D2"], "vc": ["A2", "D3"], "vla": ["F#3", "A3"], "vln2": ["D4", "F#4"],
                       "vln1": ["A4", "D5"]})]
    voiced(S, H, off, sync_first=True)
    for pn, a in (("cb", 0.36), ("vc", 0.38), ("vla", 0.37), ("vln2", 0.36), ("vln1", 0.36)):
        S.P(pn).d((mp - 0.05, a), (mp + 4, a * 1.02), (dark - 0.3, a * 0.9), (dark + 1, a * 0.62),
                  (dark + 4, a * 0.5), (ham, a * 0.36), (last + 0.3, a * 0.36), (last + 1.2, a * 0.8),
                  (cold - 0.1, a * 1.5), (off, a * 1.5))
    S.sync.append((mp * BEAT_S, "THE LAST BEACON: the run's C lands in F on the map (strings)", "vla", 0.2, "bloom"))
    S.sync.append(((last + 0.5) * BEAT_S, "the last beacon caught: the held A7sus4 resolves into D", "vla", 0.25,
                   "bloom"))
    # the beacons answer, near (the oboe, a clarinet under it) and far (horns)
    for pn in ("ob_c", "ob_dbl"):
        K.answer(S, pn, "F5" if pn == "ob_c" else "F4", mp, mode="major", rhythm=(1, .5, .5, 2))
    S.P("ob_c").d((mp - 0.05, 0.3), (mp + 2, 0.32), (mp + 4, 0.1), (mp + 4.3, 0.03))
    S.P("ob_dbl").d((mp - 0.05, 0.26), (mp + 2, 0.28), (mp + 4, 0.1), (mp + 4.3, 0.03))
    S.sync.append((mp * BEAT_S, "the near beacons answer: the oboe (F5)", "ob_c", 0.2, "arrive:77"))
    for pn, top, t, v in (("hn_far", "C5", mp + 4, 0.36), ("hn_farther", "F4", mp + 6, 0.32)):
        K.answer(S, pn, top, t, mode="major", rhythm=(1, .5, .5, 1.8))
        S.P(pn).d((t - 0.05, v), (t + 2, v * 1.02), (t + 3.8, 0.03))
        S.sync.append((t * BEAT_S, f"a far beacon answers ({pn}, {top})", pn, 0.25, f"arrive:{m(top)}"))
    # the swell out of the resolution: horns, trombone and tuba join the D, and the timpani roll grows
    voiced(S, [(last + 0.5, {"hn": ["D4"], "hn2": ["F#4"], "hn3": ["A4"], "tbn": ["A3"], "tuba": ["D2"]})], off)
    dyn(S, ("hn", "hn2", "hn3", "tbn", "tuba"), (last + 0.4, 0.1), (last + 1.5, 0.3), (cold - 0.1, 0.56),
        (off, 0.56))
    hold(S, "timp_roll", "D2", last + 1.0, off, legato=True)
    dyn(S, "timp_roll", (last + 0.9, 0.08), (cold - 0.1, 0.42), (off, 0.42))


# ---------------------------------------------------------------------------
# C17 THE RING, UNFINISHED; C18 THE DEEP, ABANDONED; C19 THE WATCH
# ---------------------------------------------------------------------------
def ring_unfinished(S, bm):
    ev = bm.ev
    r0, gone = ev("ring_unfinished"), ev("storm_gone")
    m0 = r0 + 1.0
    end = r0 + 10.0
    # the Ring motif once, slow, on the stopped horns (a2), as on the mountain page; it never finishes (no final
    # Ab): it hangs on the C while the glow drains
    for pn, v in (("hn_st", 0.36), ("hn_st2", 0.32)):
        line(S, pn, [("D4", 1.5), ("Ab4", 1.5), ("D5", 1.5), ("C5", 4.5)], m0)
        S.P(pn).d((m0 - 0.1, v), (m0 + 4.5, v * 1.05), (m0 + 6.5, v * 0.7), (end, 0.03))
    S.sync.append((m0 * BEAT_S, "THE RING, UNFINISHED: its motif once on stopped horns (D4)", "hn_st", 0.25,
                   "arrivew"))
    # under it a low D, and the tritone's Ab draining out before the storm is gone
    voiced(S, [(r0, {"cb_q": ["D2"], "vla_q": ["D3"]})], end)
    hold(S, "vc_q", "Ab2", r0, gone + 0.5)
    dyn(S, ("cb_q", "vla_q"), (r0 - 0.2, 0.03), (r0 + 1.5, 0.13), (gone, 0.1), (end, 0.02))
    dyn(S, "vc_q", (r0 - 0.2, 0.03), (r0 + 1.5, 0.13), (gone - 1.5, 0.08), (gone + 0.5, 0.02))


def deep_abandoned(S, bm):
    ev = bm.ev
    d0 = ev("deep_still")
    # the Deep's descent, at rest: bassoon and cellos in unison, a lament bass that closes V-i
    notes = [("D4", 2), ("C4", 1), ("Bb3", 1), ("A3", 2), ("G3", 1), ("F3", 1), ("E3", 1), ("A2", 1), ("D3", 1.7)]
    lines(S, ("bsn_c", "line_vc"), notes, d0)
    S.P("bsn_c").d((d0 - 0.1, 0.3), (d0 + 4, 0.3), (d0 + 9, 0.27), (d0 + 10.5, 0.2), (d0 + 11.7, 0.03))
    S.P("line_vc").d((d0 - 0.1, 0.2), (d0 + 4, 0.2), (d0 + 9, 0.18), (d0 + 10.5, 0.13), (d0 + 11.7, 0.03))
    S.sync.append((d0 * BEAT_S, "THE DEEP, ABANDONED: bassoon and cellos, the descent at rest (D4)", "bsn_c", 0.2,
                   "arrivew"))
    S.sync.append(((d0 + 10) * BEAT_S, "the Deep closes V-i: the gold left in the ground (D3)", "bsn_c", 0.25,
                   "arrivew"))
    voiced(S, [(d0, {"vla_q": ["A3"], "vln2_q": ["F4"]}), (d0 + 2, {"vla_q": ["G3"], "vln2_q": ["E4"]}),
               (d0 + 3, {"vla_q": ["F3"], "vln2_q": ["D4"]}), (d0 + 4, {"vla_q": ["E3"], "vln2_q": ["C#4"]}),
               (d0 + 6, {"vla_q": ["D3"], "vln2_q": ["Bb3"]}), (d0 + 7, {"vla_q": ["C3"], "vln2_q": ["A3"]}),
               (d0 + 8, {"vla_q": ["G3"], "vln2_q": ["C#4"]}), (d0 + 9, {"vla_q": ["E3"], "vln2_q": ["C#4"]}),
               (d0 + 10, {"vla_q": ["F3", "A3"], "vln2_q": ["D4"]})], d0 + 11.7)
    dyn(S, ("vla_q", "vln2_q"), (d0 - 0.1, 0.12), (d0 + 4, 0.14), (d0 + 9, 0.13), (d0 + 10.5, 0.1),
        (d0 + 11.7, 0.02))


def watch(S, bm):
    ev = bm.ev
    w0, sun = ev("watch"), ev("sunrise")
    # the beacons burned on: a pianissimo brass chorale, steady, closing on A into the dawn
    seq = [(w0, {"hn": ["F#4"], "hn2": ["D4"], "tbn": ["A3"], "tuba": ["D2"]}),
           (w0 + 2, {"hn": ["F#4"], "hn2": ["D4"], "tbn": ["A3"], "tuba": ["B1"]}),
           (w0 + 4, {"hn": ["G4"], "hn2": ["D4"], "tbn": ["B3"], "tuba": ["G1"]}),
           (w0 + 6, {"hn": ["G4"], "hn2": ["E4"], "tbn": ["B3"], "tuba": ["E2"]}),
           (w0 + 8, {"hn": ["A4"], "hn2": ["D4"], "tbn": ["E3"], "tuba": ["A1"]}),
           (w0 + 10, {"hn": ["A4"], "hn2": ["C#4"], "tbn": ["E3"], "tuba": ["A1"]})]
    voiced(S, seq, sun - 0.25, sync_first=True)
    dyn(S, ("hn", "hn2", "tbn", "tuba"), (w0 - 0.1, 0.14), (w0 + 2, 0.18), (w0 + 6, 0.2), (w0 + 8, 0.22),
        (w0 + 10, 0.2), (sun - 0.5, 0.13), (sun - 0.25, 0.04))
    S.sync.append((w0 * BEAT_S, "THE WATCH: the brass chorale, pp (D)", "hn2", 0.25, "arrive:62"))
    # a far beacon calls from a peak, and a farther one on the dominant
    for pn, root, t, v in (("hn_far", "D4", w0 + 2, 0.3), ("hn_farther", "A3", w0 + 6, 0.27)):
        line(S, pn, [(root, 1.0), (m(root) + 7, 1.0), (m(root) + 12, 2.0)], t)
        S.P(pn).d((t - 0.05, v), (t + 2, v * 1.03), (t + 3.6, v * 0.8), (t + 4.2, 0.03))
        S.sync.append((t * BEAT_S, f"THE WATCH: a far beacon calls ({pn}, {root})", pn, 0.25, f"arrive:{m(root)}"))


# ---------------------------------------------------------------------------
def build(bm):
    S = K.Score("C5", bm)
    C1.setup(S)
    for fn in (C1.storyteller, C1.mountain, C1.letters_to_fire, C1.fire_alone, C1.forging, C1.race, C1.deep, C1.eye):
        fn(S, bm)
    for nt in S.P("tam").notes:
        nt.kw["rel"] = TAM_BLOOM_S
    for fn in (refusal, trap, flint, C1.reveal, reveal_pad, C1.run, last_beacon, ring_unfinished, deep_abandoned,
               watch, C1.illumination, C1.plenty, C1.last_pages):
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
        if e not in have or e in ("map",):         # v1's map entries were tuned on v1's map music
            continue
        b0 = bm.ev(eid) if isinstance(eid, str) else bm.ev(eid[0]) + eid[1]
        for n in S.P(pn).notes:
            if abs(n.start - b0) < 1e-6:
                n.kw["antic"] = a
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
    S.fader = CLIMAX_FADER
    return S


IGNITIONS = ("fire_catches", "catch", "beacon_1", "beacon_2", "beacon_3", "beacon_4", "beacon_5", "beacon_6",
             "beacon_7", "last_beacon", "sunrise")


def check(S, bm):
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
    # the spoken line's window stays clean (5460-5540): nothing may start inside it
    a, b = bm.ev("line_in"), bm.ev("blank")
    for pn, p in S.parts.items():
        for n in p.notes:
            if n.pitch is not None and a - 1e-6 <= n.start < b - 0.3:
                out.append(f"NOTE IN THE LINE'S WINDOW: {pn} at beat {n.start:.2f}")
    return out


if __name__ == "__main__":
    from timeline_v3 import BarMap
    bm_ = BarMap("C5")
    S_ = build(bm_)
    parts = S_.used()
    print(f"C5: {len(parts)} parts, {sum(len(p.notes) for p in parts.values())} notes, {len(S_.sync)} sync points")
    w = K.check_notes(parts, bm_.bars * 4) + K.check_rates(parts) + check(S_, bm_)
    print(f"rule checks: {len(w)} warning(s)")
    for x in w[:80]:
        print("  " + x)
