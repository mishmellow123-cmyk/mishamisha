"""THE LONG DAWN v3 - A . EVERY STEP CLOSER, the score: from sixteenths to footsteps.

Locked to music/v3/barmap_A.json (SHOWRUNNER-REV) through music/v3/cues_A.json (cues_v3.py); both LOCKED.
Written fresh (COMPOSER-A, 27 Sep) to REVISION 1's score rules and the "sound and music" column of A's locked beat
sheet, on the shared kit (kit_v3).  Nothing of the v1/v2 kindling or race is reused (their strokes on the ignition,
32nd rolls and cymbals are what REVISION 1 forbids).

Palette (BIBLE_V3 section 9): glass, taiko, dark strings, piano, low brass; horns for the call and the answer.
No choir, no organ, no cymbal, no stroke on any fire; percussion never faster than sixteenths; every exposed line
doubled or sitting on a pad; the dawn warm, never loud.

The rules it keeps (checked by analyze_v3 and the notes):
  THE THINKING CYCLE's tempo is its pace: far-off sixteenths (false dawn); eighths opening to sixteenths and a
  cycle that shrinks from seven notes to one (the spiral to the point); below notice in the calm fire; sixteenths
  while raced (the towers, the edge); slowing through eighths to quarter notes locked to the bar (towers in the
  light); walking in step (the crossing), its glass warming toward gold; warm inside the dawn.
  Whoever lights a fire gets the CALL (her fire: the violas; the karst: a horn; the desert: the low strings; the
  Run's odd fires: the near horns; every watch-fire: the violas, inside the chord); only an answer earns the ANSWER,
  first heard on the Run's second fire (51 b1) on a far horn; the whole theme (CALL, ANSWER, HOME) only in the blue
  hour, with the film's first real cadence (A7 -> D) on 78 b1.  The effects own every fire: only blooms, swells and
  soft entries on them.  The dawn is the warmest moment, not the loudest.

THE PLAN
  A1 black        a sub drone on D, on a soft pad (basses + cellos), rising out of the silence with the wind
  A2 false dawn   the glass tone from bar 3: A5 and A5 four cents sharp, beating (the false light is unstable); the
                  thinking cycle far off in sixteenths from bar 5, as the stars go out; the drone thins into the glow
  A3 into the     a swell into the white (the high strings take the glass tone's A); from bar 9 the tone opens into
     light        the kindling's cycles in eighths (a glockenspiel on each cycle's head); from 12 b1 sixteenths, and
                  the cycle shrinks 7 -> 5 -> 3 -> 1 notes (the spiral compressing) ...
  A4 the point    ... to one point, D6, with its octave and a high violin D, swelling; one beat of breath (13 b4)
  A5 ignition     the Lydian bloom (Bbmaj9#11/D) out of the breath, breathing once a bar, too evenly; the cycle
                  below notice; the promise (17 b1): one warm line (violas + cellos in unison) rises D4 .. C5 and
                  is cut off on 18 b3, a step short of its octave
  A6 towers       A7(b9), the prize, built tower by tower from the bottom, darkest first (root; the tritone E-Bb
                  with the b9; the diminished seventh closing; the b9 on top; the major third only ever inside); the
                  cycle raced but smothered (sixteenths of the prize circled low and close, A Bb A G A Bb C#, dark
                  and damped); low drums prepare, a tightening pulse (1 -> 1 and 3 -> quarters, timpani on A under
                  it, then a timpani roll; no pickups); from 22 b3.5 the two giants: two trombones on opposite sides,
                  on the prize's tritone (C#, G), and a sub swell on A; 200 ms breath
  A7 the edge     the corrupted call (D Ab D) at four speeds: taiko + spiccato in quarters (24), eighths (26),
                  sixteenths (28: the violas and second violins; cellos and first violins keep eighths), a roll (30:
                  sustained drum and timpani rolls, string tremolo; never 32nds); the spiccato insistent, never
                  skipping (the cellos grind D D Ab D, the violas hammer the tritone Ab Ab G Ab, the second violins
                  press D against Eb), nothing above D5; the subdivisions on the low drum only, a pulse, not a
                  groove; the corrupted call in the low brass (horns in unison with the trombones, low) at each
                  doubling; the gold's sweet ring, a pure fifth struck with the bar's first surge (and its third from
                  26): lovely, a chime above the machine, never a tune; the bellows drawing into every beat; the
                  cycle racing smothered (D Eb D C D Eb Ab)
  A8 the brink    the rolls, a Shepard rise and the brass clusters thickening to A's loudest (33 b1); the rim
                  crumbling (drum and low piano, irregular on a sixteenth grid); the suck (33 b3: the orchestra cut,
                  a falling noise and a sub drop into the 125 ms breath); the IMPACT on the white (timpani, drum and
                  low piano on D, under the effects' roar)
  A9 dead valley  nothing but the impact dying into wind and ash
  A10 the ember   two seconds of true silence; the silence piano between the phrases of T10: D (36 b4), A (37 b3),
                  D (38 b3), and after "None would slow alone" the low fifth, together (39 b3)
  A11 dark adapt. one high string harmonic (A5: the false dawn's pitch, made pure), doubled
  A12 first fire  effects only, over a low D (the basses)
  A13 every ridge from the roar, strings and low brass on Bb, swelling in under it; the watch tone (cellos, D3)
                  starts and never stops; the CALL in the violas; in canon a horn at the karst and the low strings
                  at the desert, each shorter (the chain tightening), the last bringing the bass to D
  A14 beacon run  D pedal; one chord per ignition, rising by thirds: Bb | Dm F Am C Em G Bm (from D minor to the
                  brink of D major); the near horns call on fires 1 3 5 7, the far horns answer on 2 4 6: the
                  ANSWER's first sounding (51 b1, in F), its first MAJOR sounding on fire 6 (over G: its C# the #11)
  A15 watchers    the strings hold; the corrupted call, softened, in the low brass (over D6 its Ab is the fire's
                  #11); a far horn answers with the CALL
  A16 towers in   the violas hold the tritone (Ab3) until the giants open (57 b1): it heals into the fifth (A3); the
      the light   cycle slows: the race's smothered sixteenths (56), clearing into the glass in eighths as the giants
                  open (57-58), quarter notes locked to the bar (59 b1); the
                  smaller towers follow on the beat (58): the high strings enter one per beat
  A17 fire seen   the cycle in quarter notes under the held chord (D sus2); from 61 b1 all of it gathers into D
  A18 crossing    the WALK: pizzicato basses and a soft hand drum on 1 and 3; the harmony moves only at a watch-fire
                  (D | Bb | F | G | Em7); the violas' CALL inside the chord at every watch-fire; the cycle walking
                  in step, its glass warming toward gold; bars 68-69: one viola, the wind and the watch tone
  A19 blue hour   275 ms of silence; a bloom on D add9 with the cycle warm inside; the whole theme: CALL (horns and
                  violas), ANSWER (violins and a far horn), HOME (violins and a horn) arriving on D on 78 b1, the
                  first real cadence (A7 -> D); mf at most
  A20 title       D add9 rings out to silence; the watch tone ends with it; over it (v2b, from B's sunrise) the
                  ANSWER passed outward note by note, horn -> far -> farther -> farthest, each then silent (the
                  violins hand HOME's D to the first); A11's pure A5 returns ppp as the sky pales (PASS_FRAMES)
"""
import numpy as np

import kit_v3 as K
from dsl import m

# ---------------------------------------------------------------------------
# A's seats (v1 seating and fader calibration); the kit's SEATS hold the rest
# ---------------------------------------------------------------------------
_S = dict(bus="strings")
_B = dict(bus="brass")
_PC = dict(bus="perc", humanize_ms=0)
_X = dict(kind="synth", bus="synth", humanize_ms=0)
A_SEATS = {
    "vln1_sp": dict(inst="vln_spic", pan=-0.62, width=0.55, depth=0.25, send=0.2, gain_db=3.4, humanize_ms=6, **_S),
    "vln2_sp": dict(inst="vln_spic", pan=-0.30, width=0.5, depth=0.3, send=0.2, gain_db=2.4, humanize_ms=6, **_S),
    "vla_sp": dict(inst="vla_spic", pan=0.05, width=0.5, depth=0.3, send=0.2, gain_db=2.1, humanize_ms=6, **_S),
    "vc_sp": dict(inst="vc_spic", pan=0.35, width=0.5, depth=0.3, send=0.2, gain_db=1.4, humanize_ms=6, **_S),
    "cb_sp": dict(inst="cb_spic", pan=0.58, width=0.4, depth=0.35, send=0.2, gain_db=1.4, humanize_ms=6, **_S),
    # the two giants: two trombones on opposite sides of the fire (they lead the race's low brass)
    # A20's pass: the farthest horn, toward the horizon ahead (the cut follows her look)
    "hn_farthest": dict(inst="horn", pan=0.22, width=0.2, depth=1.0, send=0.95, gain_db=-4.5, humanize_ms=14, **_B),
    "tbn_l": dict(inst="trombone", pan=-0.55, width=0.4, depth=0.65, send=0.42, gain_db=-2.8, humanize_ms=8, **_B),
    "tbn_r": dict(inst="trombone", pan=0.6, width=0.4, depth=0.65, send=0.42, gain_db=-2.8, humanize_ms=8, **_B),
    "dr_taiko": dict(inst="giant_mallet", pan=-0.1, width=0.9, depth=0.6, send=0.35, gain_db=7.7, **_PC),
    "dr_tenor": dict(inst="tenor_lo", pan=0.2, width=0.8, depth=0.6, send=0.3, gain_db=0.5, **_PC),
    "dr_tenorhi": dict(inst="tenor_hi", pan=-0.25, width=0.8, depth=0.6, send=0.3, gain_db=-0.2, **_PC),
    "dr_bdrum": dict(inst="bdrum2", pan=0.1, width=0.8, depth=0.7, send=0.4, gain_db=7.5, **_PC),
    "imp_bd": dict(inst="bdrum2", pan=0.05, width=0.9, depth=0.7, send=0.45, gain_db=7.5, **_PC),
    "bdroll": dict(inst="bdrum_roll", pan=0.1, width=0.8, depth=0.7, send=0.4, gain_db=-1.0, **_PC),
    "feet": dict(inst="giant_hand", pan=0.05, width=0.6, depth=0.55, send=0.3, gain_db=4.0, **_PC),
    "glock": dict(inst="glock", pan=0.35, width=0.6, depth=0.5, send=0.4, gain_db=9.0, **_PC),
    "piano_lo": dict(inst="piano", pan=0.1, width=0.9, depth=0.45, send=0.45, gain_db=8.3, humanize_ms=0, bus="keys"),
    "glass_far": dict(inst="fmbell", pan=0.0, width=1.0, depth=0.9, send=0.8, gain_db=-12.0,
                      params=dict(ratio=3.5, index=1.6, decay=1.1), **_X),
    "ring": dict(inst="fmbell", pan=0.25, width=1.0, depth=0.5, send=0.5, gain_db=-8.3,
                 params=dict(ratio=1.0, index=1.2, decay=1.8, bright=0.7), **_X),
    "glasstone": dict(inst="harmonic", pan=0.0, width=0.8, depth=0.6, send=0.6, gain_db=-8.0, **_X),
    "cycle": dict(inst="fmwarm", pan=0.0, width=1.0, depth=0.45, send=0.5, gain_db=-7.0,
                  params=dict(ratio=3.5, index=1.6, decay=1.2, lp=7000.0, chorus=3.0), **_X),
    # the race's cycle (v2): the same FM (ratio 3.5), smothered: low index, low-passed, no glass partial, damped
    "cycle_dk": dict(inst="fmwarm", pan=0.0, width=0.8, depth=0.55, send=0.45, gain_db=-13.0,
                     params=dict(ratio=3.5, index=0.9, decay=0.9, lp=1800.0, chorus=2.0, atk=0.003), **_X),
    "bellows": dict(inst="riser", pan=0.0, width=1.0, depth=0.5, send=0.3, gain_db=-9.0, **_X),
    "shepard": dict(inst="shepard", pan=0.0, width=1.0, depth=0.4, send=0.35, gain_db=-5.8, **_X),
    "suck": dict(inst="riser", pan=0.0, width=1.0, depth=0.4, send=0.2, gain_db=-7.0, **_X),
    "subdrop": dict(inst="subdrop", pan=0.0, width=0.0, depth=0.0, send=0.0, gain_db=-6.1, **_X),
}
for _k, _v in A_SEATS.items():
    K.SEATS.setdefault(_k, _v)

# role parts -> seat (a role gets its own part: its own dynamics curve and cache entry)
ROLES = {
    "pad_cb": "cb_q", "pad_vc": "vc_q", "watch": "vc_q", "prom_vc": "vc", "prom_vla": "vla", "call_vla": "vla",
    "call_vlaq": "vla_q", "call_vlaq2": "vla_q", "desert_vc": "vc", "desert_cb": "cb", "harm_str": "svln",
    "tw3": "vln2_q", "tw4": "vln1_q", "theme_vla": "vla", "bloom_vla": "vla",
}

# the conductor's ride (dB on the players' dynamics, per section), from the battery
RIDE = {"A2": -2.3, "A3": -2.3, "A4": -3.5, "A7": -2.4, "A10": -5.0, "A16": -1.0, "A17": -0.8}
# the drum hits share a bus limiter (score-mix domain, before the master), so no single stroke drives the master
GROUPS = {"dr_": dict(ceiling_db=-9.0, release=0.08)}
# the intended colour arc (median spectral centroid of the score stem, Hz): the point blinding, the race dark,
# the blue hour and the title warm (render 2: 1792, 459, 446, 505, 506)
CENTROID = {"A4": (1000, 3000), "A7": (250, 700), "A8": (250, 700), "A19": (300, 700), "A20": (300, 800)}
# measured extra anticipation (s) for soft entries whose fundamental arrives later than the kit's attack table
# predicts (render 1, analysis/v3/final_A): {(part, global beat): seconds}
SYNC_TRIM = {("call_vla", 180.0): 0.15, ("desert_vc", 194.0): 0.06, ("desert_cb", 194.0): 0.06,
             ("hn_far", 216.0): 0.03, ("watch", 180.0): 0.05, ("hn", 198.0): 0.03,
             **{("call_vlaq", b): 0.38 for b in (244.0, 252.0, 264.0, 276.0, 288.0)}}

# the kindling's cycles (kit) and A's own
CYC_IGN = ("D6", "A6", "E6", "Bb5", "F6", "C6", "E6")          # the calm fire: Bbmaj9(#11), below notice
CYC_PRIZE = ("A3", "Bb3", "A3", "G3", "A3", "Bb3", "C#4")       # the towers: the prize circled, low and close
CYC_RACE = ("D4", "Eb4", "D4", "C4", "D4", "Eb4", "Ab4")        # the edge (and 56): D Phrygian, the tritone's jab
DAMP = lambda u: dict(damp=True)                                # the race's cycle is damped: it ticks, never rings
CYC_HEAL = ("D5", "E5", "A5", "D6", "E6", "A5", "E5")          # healed: seven eighths against the bar
SHRINK = (("D6", "A5", "E6", "F5", "C6", "Bb5", "E5"), ("D6", "A5", "E6", "F5", "C6"), ("D6", "A5", "E6"), ("D6",))
LOCK = ("D5", "A5", "D6", "E6")
# A20 (v2b, from B's sunrise; the director, 28 Sep): the ANSWER passed outward, one note per horn, each horn then
# silent; A11's pure A5 returns ppp as the rose sky pales. In FRAMES (20 a beat) so that A-FIX's fire-paling frames
# drop straight in: one frame per note, in order. They must fall after the cadence's horn (>= 6200) and before the
# fade; nothing else moves with them except the violins' hand-off of HOME's D to the first horn.
PASS_FRAMES = (6240, 6290, 6340, 6390)                         # 79 b1, 79 b3.5, 80 b2, 80 b4.5 (current A20)
PASS_NOTES = (("hn", "D4", 0.34), ("hn_far", "C#4", 0.33), ("hn_farther", "B3", 0.32), ("hn_farthest", "F#3", 0.3))
HARM_RETURN_FRAME = 6320                                       # 80 b1                                # locked to the bar (kit LOCK4 on D)


def T(b):
    """seconds of a global beat"""
    return b * 60.0 / 72.0


def setup(S):
    for pn in ("cb", "vc", "vla", "vln2", "vln1", "cb_q", "vc_q", "vla_q", "vln2_q", "vln1_q",
               "hn", "hn2", "hn_far", "hn_farther", "tuba", "harmonic", "sub", "glass", "piano", "cb_pizz",
               "vln_trem", "vla_trem", "vc_trem", "timp", "timp_roll"):
        S.add(pn)
    for pn in A_SEATS:
        S.add(pn)
    for pn, seat in ROLES.items():
        S.add(pn, seat)
    S.parts["harm_str"] = S.add("harm_str", "svln", inst="svln_q", pan=-0.4, depth=0.6, send=0.6, gain_db=-4.0)
    S.add("call_vlaq2", "vla_q", pan=0.18)
    S.add("tw3", "vln2_q", pan=-0.2)
    S.add("tw4", "vln1_q", pan=-0.5)


def sync(S, b, label, part, kind, w=0.2):
    S.sync.append((T(b), label, part, w, kind))


def phrase(S, pn, notes, b0, legato=True, sync_first=True, **kw):
    """notes = [(pitch, beats), ...]; returns the end beat"""
    t = b0
    for i, (p, d) in enumerate(notes):
        S.P(pn).n(p, t, d, legato=(legato and i > 0), sync=(sync_first and i == 0), **kw)
        t += d
    return t


def hold(S, pn, pitches, b0, b1, legato=False, sync=False, **kw):
    for p in pitches:
        S.P(pn).n(p, b0, b1 - b0, legato=legato, sync=sync, **kw)


def cycle(S, pn, pattern, b0, b1, step, v0, v1=None, pan_amp=0.45, heads=None, head_v=0.2, sync_first=True,
          morph=None):
    """the thinking cycle on a grid (the kit's rule: step >= a sixteenth); heads = a part doubling each cycle's
    first note (the glockenspiel); morph(u) -> per-note timbre kw (the glass warming), u = 0..1 over [b0, b1)"""
    assert step >= 0.25 - 1e-9
    v1 = v0 if v1 is None else v1
    t, i = b0, 0
    while t < b1 - 1e-6:
        u = (t - b0) / max(1e-6, b1 - b0)
        kw = morph(u) if morph else {}
        S.P(pn).n(pattern[i % len(pattern)], t, step, v0 + (v1 - v0) * u, pan=pan_amp * np.sin(i * 0.9),
                  sync=(sync_first and i == 0), **kw)
        if heads and i % len(pattern) == 0:
            S.P(heads).n(pattern[0], t, 1.0, head_v)
        t += step
        i += 1
    return t


def warm_morph(u0, u1):
    """the lantern warms from ice-white toward gold: the glass cycle's timbre from the kindling's glass (u = 0)
    to the v2 warm FM (u = 1), between u0 and u1 of a span"""
    def f(u):
        w = min(1.0, max(0.0, u0 + (u1 - u0) * u))
        return dict(ratio=3.5 - 2.5 * w, index=1.6 - 0.9 * w, glass=1.0 - w, lp=7000.0 - 2200.0 * w,
                    decay=1.2 + 1.2 * w, atk=0.0015 + 0.004 * w)
    return f


# ---------------------------------------------------------------------------
def build(bm):
    S = K.Score("A", bm)
    setup(S)
    ev = bm.ev
    opening(S, bm, ev)
    ignition(S, bm, ev)
    towers(S, bm, ev)
    edge(S, bm, ev)
    brink(S, bm, ev)
    ember(S, bm, ev)
    ridges(S, bm, ev)
    run(S, bm, ev)
    watchers(S, bm, ev)
    light(S, bm, ev)
    crossing(S, bm, ev)
    blue_hour(S, bm, ev)
    end = bm.bars * 4
    for pn in ("hn", "hn2", "hn_far", "hn_farther", "hn_farthest", "tbn_l", "tbn_r", "tuba"):
        K.breathe(S, pn, 0, end, depth=0.12, min_dur=1.4)
    K.ride(S, bm, RIDE)
    S.groups = dict(GROUPS)
    # every attacked note of a bowed or blown part starts early by its own sample's measured attack, so that it
    # ARRIVES on the beat; legato changes cross-fade centred on the beat
    for pn, p in S.parts.items():
        if p.kind == "sampler" and p.inst in ("vln", "vla", "vc", "cb", "vln_q", "vla_q", "vc_q", "cb_q", "svln_q",
                                              "horn", "trombone", "tuba", "vln_trem", "vla_trem", "vc_trem"):
            K.anticipate(S, pn, hi=0.45)
            for n in p.notes:
                if n.legato:
                    n.kw["antic"] = n.kw.get("lantic", 0.02)
    for (pn, b), dt in SYNC_TRIM.items():
        for n in S.P(pn).notes:
            if abs(n.start - b) < 1e-6 and not n.legato:
                n.kw["antic"] = n.kw.get("antic", 0.0) + dt
    for s in bm.sections:
        if s.get("rel"):
            S.levels.append((s["id"], *s["rel"]))
        if s["id"] in CENTROID:
            S.centroid.append((s["id"], *CENTROID[s["id"]]))
    t = lambda sid: bm.event(sid)["t"]
    S.rules = [("the brink is A's loudest: the edge at least 1.5 LU under it", t("edge"), t("vortex"), -1.5),
               ("the dawn is warm, not loud: the blue hour at least 3 LU under the brink", t("set_down"),
                t("end_fade"), -3.0),
               ("the first fire is near silence: the strikes to the catch at least 10 LU under the brink",
                t("strike1"), t("catch"), -10.0)]
    return S


# ---------------------------------------------------------------------------
# A1-A4: black, the false dawn, into the light, the point
# ---------------------------------------------------------------------------
def opening(S, bm, ev):
    glow, so, white, letters = ev("glow"), ev("stars_out"), ev("white"), ev("letters")
    spiral, point, br = ev("spiral"), ev("point"), ev("breath_ign")
    # A1-A2: a sub drone on D, on a soft pad; it thins into the glow and is gone in the white
    S.P("sub").n("D1", 0.3, white + 0.7, atk=2.5, rel=2.0)
    S.P("sub").d((0.3, 0.05), (3.5, 0.2), (glow, 0.25), (white - 6, 0.24), (white - 1, 0.1), (white + 0.8, 0.03))
    K.held(S, "pad_cb", "D2", 0.6, white + 0.4, bow=8)
    K.held(S, "pad_vc", "D3", 1.2, white + 0.4, bow=8)
    for pn, lv in (("pad_cb", 0.2), ("pad_vc", 0.15)):
        S.P(pn).d((0.6, 0.03), (4, lv * 0.75), (glow, lv), (so, lv * 0.95), (white - 5, lv * 0.8),
                  (white - 1, lv * 0.35), (white + 0.3, 0.03))
    # the glass tone: A5 and A5 four cents sharp, beating ~2 Hz (the false light is unstable); it swells into the
    # white and opens into the arpeggios
    for p, pan in ((81.0, -0.35), (81.04, 0.35)):
        S.P("glasstone").n(p, glow, letters + 2 - glow, pan=pan, atk=3.0, rel=2.5)
    S.P("glasstone").d((glow, 0.09), (glow + 4, 0.15), (so, 0.18), (white - 4, 0.21), (white, 0.25), (letters, 0.18),
                       (letters + 2, 0.08))
    # the thinking cycle, far off, in sixteenths, as the stars nearest the glow go out
    cycle(S, "glass_far", K.CYC5, so, white, 0.25, 0.08, 0.15, pan_amp=0.6)
    # a swell into the white: the high strings take the glass tone's A and open it to the fifth
    sw = white - 6
    for pn, p in (("vln1_q", "A5"), ("vln2_q", "E5"), ("vla_q", "A4")):
        S.P(pn).n(p, sw, letters + 1.2 - sw)
        S.P(pn).d((sw, 0.04), (white - 2, 0.16), (white - 0.2, 0.23), (white + 1.5, 0.19), (letters, 0.09),
                  (letters + 1.1, 0.03))
    # the glyphs: the tone opens into the kindling's cycles, in eighths (letters drifting in from every side),
    # a glockenspiel on each cycle's head; a low D far under them
    cycle(S, "glass", K.CYC6A, letters, letters + 8, 0.5, 0.16, 0.2, pan_amp=0.55, heads="glock", head_v=0.13)
    cycle(S, "glass", K.CYC7, letters + 8, spiral, 0.5, 0.2, 0.23, pan_amp=0.55, heads="glock", head_v=0.15,
          sync_first=False)
    K.held(S, "pad_cb", "D2", letters + 1, spiral + 2, bow=8)
    K.held(S, "pad_vc", "D3", letters + 1.4, spiral + 2, bow=8)
    for pn, lv in (("pad_cb", 0.14), ("pad_vc", 0.11)):
        S.P(pn).d((letters + 1, 0.03), (letters + 4, lv), (spiral - 1, lv), (spiral + 1.8, 0.03))
    # from 12 b1 sixteenths, and the cycle shrinks 7 -> 5 -> 3 -> 1: the spiral compressing to one point
    t, v = spiral, 0.22
    for pat in SHRINK:
        for k, p in enumerate(pat):
            S.P("glass").n(p, t, 0.25, v, pan=0.5 * np.sin(k * 1.3 + len(pat)))
            t += 0.25
            v += 0.007
    assert abs(t - point) < 1e-6, (t, point)
    # the point: D6 and its octave in sixteenths, a glockenspiel glint, a high violin D, swelling to the breath
    tt, k = point, 0
    while tt < br - 1e-6:
        u = (tt - point) / (br - point)
        S.P("glass").n("D6", tt, 0.25, 0.3 + 0.12 * u, pan=0.15 * np.sin(k * 2.1), sync=(k == 0))
        if k % 2 == 0:
            S.P("glass").n("D7", tt, 0.25, 0.12 + 0.07 * u, pan=-0.15 * np.sin(k * 2.1))
        tt += 0.25
        k += 1
    S.P("glock").n("D6", point, 1.0, 0.2, sync=True)
    S.P("vln1_q").n("D6", point, br - point + 0.05)
    S.P("vln1_q").d((point, 0.05), (br - 0.3, 0.22), (br, 0.22))
    sync(S, point, "the point: the cycle converges on D6", "glass", "hit", 0.1)


# ---------------------------------------------------------------------------
# A5: IGNITION . THE PROMISE
# ---------------------------------------------------------------------------
BLOOM_A = {"cb": ["D2"], "vc": ["Bb2"], "vla": ["F3"], "bloom_vla": ["C4"], "vln2": ["D4", "E4"], "vln1": ["A4", "E5"]}
BLOOM_MID = ("bloom_vla", "vln2")          # the bloom's voices in the promise line's register: they make room for it


def ignition(S, bm, ev):
    ig, prom, cut, gone, tw = ev("ignition"), ev("promise"), ev("promise_cut"), ev("promise_gone"), ev("towers")
    end = cut + 1.6
    # the Lydian bloom out of the breath's silence, breathing once a bar, too evenly (the same breath every bar)
    for pn, ps in BLOOM_A.items():
        mid = pn in BLOOM_MID
        hold(S, pn, ps, ig, (prom + 1.6) if mid else end, sync=True)
        base = 0.3 if pn != "vln1" else 0.27
        pts = [(ig, 0.08), (ig + 1.5, base + 0.06)]
        b = ig + 4
        while b < prom - 1e-6:
            pts += [(b - 0.4, base), (b + 1.6, base + 0.06)]
            b += 4
        if mid:                                            # out of the line's way as it enters
            pts += [(prom - 0.4, base), (prom + 1.5, 0.03)]
        else:                                              # the rest keeps breathing, receded
            pts += [(prom - 0.4, base), (prom + 1.6, base * 0.78), (prom + 3.6, base * 0.72),
                    (cut - 0.4, base * 0.74), (cut + 0.4, base * 0.5), (end - 0.1, 0.03)]
        S.P(pn).d(*pts)
    sync(S, ig, "IGNITION: the bloom out of the breath (score stem)", "score", "bloom", 0.15)
    # the calm fire's filaments, below notice: its cycle in sixteenths, pp, cut off with the promise (v2: it ran
    # on, bright, into the towers); the towers' cycle starts low and dark on 19 b1
    cycle(S, "glass", CYC_IGN, ig + 0.5, cut, 0.25, 0.06, 0.08, pan_amp=0.5, sync_first=False)
    # THE PROMISE: one warm line, violas and cellos in unison, rising; cut off on 18 b3, a step short of its octave
    line = [("D4", 1.5), ("E4", 0.5), ("F4", 1.0), ("A4", 1.0), ("G4", 0.5), ("A4", 0.5), ("C5", 1.0)]
    for pn in ("prom_vc", "prom_vla"):
        e = phrase(S, pn, line, prom)
        assert abs(e - cut) < 1e-6
        S.P(pn).notes[-1].kw["rel"] = 0.1                  # cut off, not released
        S.P(pn).d((prom - 0.1, 0.33), (prom + 1.5, 0.37), (prom + 3.5, 0.41), (prom + 5, 0.45), (cut - 0.2, 0.49),
                  (cut, 0.49))
    sync(S, prom, "THE PROMISE: the warm line enters (cellos)", "prom_vc", "arrive:62")


# ---------------------------------------------------------------------------
# A6: THE TOWERS . TWO GIANTS
# ---------------------------------------------------------------------------
def towers(S, bm, ev):
    tw, giants, br, e0 = ev("towers"), ev("giants"), ev("breath_race"), ev("edge")
    # A7(b9), the prize, built tower by tower from the bottom, darkest first (v2; the user: the race's start read
    # as upbeat): the root; then the tritone E3-Bb3 with the b9 over the root; then the diminished seventh closes
    # (G3, C#4); the b9 on top at the giants' bar. The major third (C#) is only ever inside, never on top.
    entries = [(tw, "cb_q", "A1"), (tw, "vc_q", "A2"), (tw + 4, "vc_q", "E3"), (tw + 4, "vla_q", "Bb3"),
               (tw + 8, "vla_q", "G3"), (tw + 8, "vln2_q", "C#4"), (tw + 12, "vln1_q", "Bb4"), (tw + 12, "vln2_q", "G4")]
    for b, pn, p in entries:
        S.P(pn).n(p, b, br - b + 0.02, sync=(b == tw))
    for pn, lv in (("cb_q", 0.24), ("vc_q", 0.22), ("vla_q", 0.22), ("vln2_q", 0.2), ("vln1_q", 0.2)):
        first = min(b for b, q, _ in entries if q == pn)
        S.P(pn).d((first - 0.05, lv * 0.75), (first + 2, lv * 0.8), (giants, lv), (br - 0.5, lv * 1.3), (br, lv * 1.3))
    sync(S, tw, "THE TOWERS: A7(b9), the prize (basses)", "cb_q", "arrive:33")
    # the cycle, raced but smothered (v2: it was the prize's bright arpeggio, E5-E6 in the glass, and read as
    # excitement): the prize circled low and close in sixteenths (the b9's sigh, the tritone's jab), damped, dark,
    # rising from below notice
    cycle(S, "cycle_dk", CYC_PRIZE, tw, e0, 0.25, 0.06, 0.2, pan_amp=0.35, sync_first=False, morph=DAMP)
    # low drums prepare, a tightening pulse, not a drive: 1 | 1 3 | 1 3 | quarters | quarters and a timpani roll on
    # A, into the breath; the timpani double it softly on the prize's root from bar 21 (no pickups, no accents)
    hits = [tw, tw + 4, tw + 6, tw + 8, tw + 10] + [tw + 12 + k for k in range(8)]
    for k, b in enumerate(hits):
        S.P("dr_bdrum").n(60, b, 1.0, 0.14 + 0.24 * k / (len(hits) - 1), sync=(k == 0))
        if tw + 8 <= b <= giants - 2:
            S.P("timp").n("A1", b, 1.0, 0.13 + 0.03 * (b - tw - 8), maxlen=1.4)
    S.P("timp_roll").n("A1", tw + 16, br - (tw + 16))
    S.P("timp_roll").d((tw + 16, 0.08), (br - 0.2, 0.42))
    # the two giants: two trombones on opposite sides of the fire, on the prize's tritone (C#3, G3), and a sub on A
    for pn, p in (("tbn_l", "C#3"), ("tbn_r", "G3")):
        S.P(pn).n(p, giants, br - giants + 0.02, sync=True)
        S.P(pn).d((giants - 0.05, 0.24), (giants + 1.5, 0.28), (br - 0.2, 0.42))
    sync(S, giants, "the two giants begin to grow: the trombones' tritone", "tbn_l", "arrive:49", 0.25)
    S.P("sub").n("A1", giants, br - giants, atk=1.5, rel=0.1)
    S.P("sub").d((giants, 0.05), (br - 0.3, 0.42))


# ---------------------------------------------------------------------------
# A7: THE EDGE (the corrupted call at four speeds)
# ---------------------------------------------------------------------------
def edge(S, bm, ev):
    e0, e8, e16, roll, vx = ev("edge"), ev("eighths"), ev("sixteenths"), ev("roll"), ev("vortex")
    # the taiko on every beat from the downbeat (accent on 1), and its subdivisions on lighter drums
    b = e0
    while b < vx - 1e-6:
        u = (b - e0) / (vx - e0)
        acc = 0.1 if (b - e0) % 4 == 0 else 0.0
        if b < roll or (b - roll) % 2 == 0:              # bar 30: beats 1 and 3 under the roll
            S.P("dr_taiko").n(60, b, 1.0, min(0.72, 0.38 + 0.24 * u + acc * 0.6), sync=(b == e0))
        b += 1
    # the subdivisions (v2): a tightening pulse, not a groove. Only the low drum, softer than the beat and even:
    # the offbeat eighths a heartbeat's echo (26-27), then every sixteenth (28-29); no high drum, no backbeat
    for k in range(int((e16 - e8) * 2)):
        t = e8 + 0.5 + k                                    # eighths: the offbeats
        if t < e16:
            S.P("dr_tenor").n(60, t, 0.5, 0.26 + 0.06 * k / 16)
    t = e16
    while t < roll - 1e-6:                                  # sixteenths: e / & / a, even, slowly pressing
        if round((t - e16) % 1, 3) > 0:
            S.P("dr_tenor").n(60, t, 0.25, 0.25 + 0.07 * (t - e16) / (roll - e16))
        t += 0.25
    sync(S, e0, "THE EDGE: the taiko on the downbeat", "dr_taiko", "hit", 0.1)
    sync(S, e8, "the rhythm doubles: eighths", "dr_taiko", "hit", 0.1)
    sync(S, e16, "the rhythm doubles: sixteenths", "dr_taiko", "hit", 0.1)
    # the roll (never 32nds): the sustained bass-drum and timpani rolls, crescendo into the brink
    S.P("bdroll").n(60, roll, ev("suck") - roll)
    S.P("bdroll").d((roll, 0.22), (vx, 0.3), (ev("loudest"), 0.5), (ev("suck") - 0.1, 0.52))
    S.P("timp_roll").n("D2", roll, ev("suck") - roll, sync=True)
    S.P("timp_roll").d((roll, 0.25), (vx, 0.32), (ev("loudest"), 0.6), (ev("suck") - 0.1, 0.62))
    # the spiccato ostinato (v2: insistent, never skipping; it was the corrupted call arpeggiated, D Ab D' Ab, up
    # to D6 in the first violins, and bounced): the corrupted call ground into pitches that press. The cellos grind
    # a repeated D that drops to the tritone below (D D Ab D); the violas hammer the tritone itself, pulled down a
    # semitone and back (Ab Ab G Ab: stuck on the note that heals only when the giants open, 57 b1); from bar 28
    # the second violins press the D against its flat second (D D Eb D) and the first violins double the violas'
    # tritone an octave up. Nothing above D5. (Sparse beats dense: at each doubling one new layer takes the faster
    # value; in the sixteenths the violas and second violins run, the cellos and first violins keep eighths.)
    figs = {"vc_sp": ("D3", "D3", "Ab2", "D3"), "vla_sp": ("Ab3", "Ab3", "G3", "Ab3"),
            "vln2_sp": ("D4", "D4", "Eb4", "D4"), "vln1_sp": ("Ab4", "Ab4", "G4", "Ab4")}
    for (b0, b1, step), parts in (((e0, e8, 1.0), ("vc_sp",)), ((e8, e16, 0.5), ("vc_sp", "vla_sp")),
                                  ((e16, roll, 0.5), ("vc_sp", "vln1_sp")), ((e16, roll, 0.25), ("vla_sp", "vln2_sp"))):
        for pn in parts:
            t, k = b0, 0
            while t < b1 - 1e-6:
                u = (t - e0) / (roll - e0)
                acc = 0.08 if k % 4 == 0 else 0.0
                S.P(pn).n(figs[pn][k % 4], t, step, min(0.72, 0.42 + 0.18 * u + acc), sync=(t == e0))
                t += step
                k += 1
    t = e0
    while t < roll - 1e-6:                                  # the basses: D on the beat (eighths from bar 28)
        u = (t - e0) / (roll - e0)
        st = 1.0 if t < e16 else 0.5
        S.P("cb_sp").n("D2", t, st, min(0.8, 0.46 + 0.26 * u))
        t += st
    # the roll in the strings: tremolo on the tritone, into the brink's clusters
    for pn, ps in (("vc_trem", ["D3", "Ab3"]), ("vla_trem", ["D4", "Ab4"]), ("vln_trem", ["Ab4", "D5"])):
        hold(S, pn, ps, roll, ev("suck"), sync=True)
    # the corrupted call in the low brass at each doubling, fuller each time; the tritone held between (v2: the
    # horns join in unison with the trombones, low and covered, not an octave up: a rising horn call up there read
    # as adventure)
    calls = [(e0, ("tbn_l", "tuba"), 0.36), (e8, ("tbn_l", "tbn_r", "tuba"), 0.4),
             (e16, ("tbn_l", "tbn_r", "tuba", "hn", "hn2"), 0.4), (roll, ("tbn_l", "tbn_r", "tuba", "hn", "hn2"), 0.46)]
    reg = {"tbn_l": "D3", "tbn_r": "D3", "tuba": "D2", "hn": "D3", "hn2": "D3"}
    for b0, parts, v in calls:
        for pn in parts:
            r = m(reg[pn])
            phrase(S, pn, [(r, 1.0), (r + 6, 1.0), (r + 12, 2.0)], b0)
            if b0 < roll:                                    # the tritone held until the next call
                S.P(pn).n(r + 6, b0 + 4, 3.95, legato=True)
            S.P(pn).d((b0 - 0.05, v), (b0 + 1, v), (b0 + 3.5, v * 1.08))
            if b0 < roll:
                S.P(pn).d((b0 + 7.8, v * 0.95))
        sync(S, b0, "the corrupted call in the low brass", parts[0], "arrive:50", 0.25)
    # the gold's sweet ring (the gold is lovely: the horror). v2: it was a D-major figure on every beat far up
    # (D7 A6 F#6 A6) with a glockenspiel, a tune, and read as excitement. Now a pure open fifth (D6 + A6), struck
    # with each bar's first surge, and from 26 b1 (the rim crumbling under the gilded ones) with its third too:
    # still lovely, a chime above the machine, never a tune, no major third
    b = e0
    while b < vx - 1e-6:
        u = (b - e0) / (vx - e0)
        beat = round((b - e0) % 4)
        if beat == 0 or (beat == 2 and b >= e8):
            for p, pan in (("D6", -0.25), ("A6", 0.3)):
                S.P("ring").n(p, b, 1.0, 0.14 + 0.08 * u, pan=pan)
        b += 1
    # the forge-stacks breathe like bellows: a low draw of air into every beat
    b = e0 + 1
    while b < vx + 1e-6:
        u = (b - e0) / (vx - e0)
        S.P("bellows").n(60, b - 0.55, 0.55, 0.26 + 0.2 * u, f0=140.0, f1=900.0, curve=2.2)
        b += 1
    # the cycle keeps racing, smothered, under it all (v2: it was the prize's bright arpeggio in the glass): the
    # race's own cell, D Eb D C D Eb Ab (D Phrygian, the tritone's jab), damped; a sub under the rim
    cycle(S, "cycle_dk", CYC_RACE, e0, vx, 0.25, 0.2, 0.26, pan_amp=0.35, sync_first=False, morph=DAMP)
    S.P("sub").n("D1", e0, ev("suck") - e0, atk=0.05, rel=0.15)
    S.P("sub").d((e0, 0.3), (roll, 0.42), (ev("loudest"), 0.5), (ev("suck") - 0.1, 0.5))


# ---------------------------------------------------------------------------
# A8-A9: THE BRINK . OVER THE RIM; THE DEAD VALLEY
# ---------------------------------------------------------------------------
def brink(S, bm, ev):
    vx, rim, crown, over = ev("vortex"), ev("rim_gives"), ev("crown_falls"), ev("over_rim")
    loud, suck, white = ev("loudest"), ev("suck"), ev("white_impact")
    roll = ev("roll")
    # the string roll, from bar 30 to the suck: its dynamics
    for pn in ("vc_trem", "vla_trem", "vln_trem"):
        S.P(pn).d((roll, 0.3), (vx, 0.42), (over, 0.6), (loud, 0.85), (suck - 0.05, 0.85))
    # the brass clusters, thickening (D Ab | + A Eb | + G) to the loudest, cut on the suck
    cl = [("tuba", "D2", vx), ("tbn_l", "D3", vx), ("tbn_r", "Ab3", vx), ("hn2", "A3", rim), ("hn", "Eb4", rim),
          ("hn_far", "G3", over)]
    for pn, p, b0 in cl:
        S.P(pn).n(p, b0, suck - b0, legato=(b0 == vx and pn != "hn_far"), rel=0.12,
                  gain_db=(4.0 if pn == "hn_far" else 0.0))       # the far seat pulled in for the cluster
        S.P(pn).d((b0, 0.64 if b0 == vx else 0.3), (over, 0.7), (loud, 0.9), (suck - 0.05, 0.9))
    # the Shepard rise: vertigo, from the vortex into the breath
    S.P("shepard").n(60, vx, white - 0.15 - vx, 0.6, oct_per_s=0.5, base=55.0, curve=1.6)
    # the crumbling rim: drum and low piano, irregular on a sixteenth grid, tumbling down a chromatic slope
    offs = (0, 0.75, 1.25, 2.0, 2.5, 2.75, 3.5, 4.25, 5.0, 5.25, 6.0, 6.5, 7.25)
    for k, o in enumerate(offs):
        b = rim + o
        v = 0.5 + 0.2 * k / len(offs)
        d = min(0.75, (offs[k + 1] - o) if k + 1 < len(offs) else 0.75)
        S.P("cb_sp").n(m("D2") - k, b, d, v)
        if k % 2 == 0:
            S.P("dr_taiko").n(60, b, 1.0, 0.4 + 0.16 * k / len(offs))
    # the suck: everything cut on 33 b3; a falling noise and a sub drop, into the 125 ms breath
    S.P("suck").n(60, suck, white - 0.15 - suck, 0.85, f0=6000.0, f1=160.0, curve=2.4)
    S.P("subdrop").n(60, suck, white - 0.15 - suck, 0.8, f0=60.0, f1=24.0)
    # the IMPACT on the white is the effects' (the fire's own physics); the score only gives it a low D under it
    # (timpani and bass drum, short), and is then silent until the silence piano
    S.P("timp").n("D2", white, 1.0, 0.62, sync=True, maxlen=3.0)
    S.P("imp_bd").n(60, white, 1.0, 0.56, sync=True, maxlen=3.0)
    sync(S, white, "the IMPACT on the white (timpani under the effects)", "timp", "hit", 0.1)


# ---------------------------------------------------------------------------
# A10-A12: BLACK . THE EMBER; DARK ADAPTATION; THE FIRST FIRE
# ---------------------------------------------------------------------------
def ember(S, bm, ev):
    sil, stars, mw, s1, catch, roar = ev("silence"), ev("stars"), ev("milky_way"), ev("strike1"), ev("catch"), ev("roar")
    # two seconds of true silence from 36 b1 (to f2848); then wind; the silence piano between T10's phrases:
    # D before "Many who raced...", A between the two lines, D after "None would slow alone", then together
    d1, a1, d2, tog = sil + 3.0, sil + 6.0, sil + 10.0, sil + 14.0          # 36 b4, 37 b3, 38 b3, 39 b3
    damp = stars + 3.5
    for p, b, v in (("D4", d1, 0.24), ("A4", a1, 0.21), ("D5", d2, 0.23)):
        S.P("piano").n(p, b, damp - b, v, sync=True)
        sync(S, b, f"the silence piano: {p}", "piano", "hit", 0.1)
    S.P("piano").n("D3", d1 + 0.02, damp - d1, 0.12)
    S.P("piano").n("D2", tog, damp - tog, 0.15, sync=True)
    S.P("piano").n("A2", tog + 0.03, damp - tog, 0.13)
    sync(S, tog, "the silence piano: the low fifth, together", "piano", "hit", 0.1)
    # dark adaptation: one high string harmonic (the false dawn's A, made pure), doubled
    S.P("harmonic").n("A5", stars + 0.5, s1 - 0.5 - stars, atk=2.2, rel=1.8)
    S.P("harmonic").d((stars + 0.5, 0.12), (stars + 4, 0.2), (mw, 0.26), (s1 - 3, 0.2), (s1 - 0.6, 0.05))
    S.P("harm_str").n("A5", stars + 1.0, s1 - 1.2 - stars)
    S.P("harm_str").d((stars + 1.0, 0.04), (mw, 0.09), (s1 - 3, 0.07), (s1 - 1.3, 0.03))
    # the first fire: effects only, over a low D in the basses (which steps to Bb at the roar)
    K.held(S, "cb_q", "D2", s1 - 2, roar + 0.08, bow=8)
    S.P("cb_q").d((s1 - 2, 0.03), (s1, 0.1), (ev("blow"), 0.12), (catch - 0.5, 0.16), (catch, 0.18), (roar - 0.2, 0.2))


# ---------------------------------------------------------------------------
# A13: EVERY RIDGE . FAR PLACES (the watch tone starts)
# ---------------------------------------------------------------------------
def ridges(S, bm, ev):
    roar, kf, df, run0 = ev("roar"), ev("karst_flare"), ev("desert_fire"), ev("run")
    end = bm.bars * 4
    # the watch tone: a low D in the cellos, from the roar until the last chord (re-bowed; the breath before the
    # blue hour is the one silence it keeps)
    K.held(S, "watch", "D3", roar, ev("end_fade") + 1.0, bow=8)
    S.P("watch").d((roar - 0.05, 0.15), (roar + 2, 0.2), (run0, 0.22), (ev("watchers"), 0.2), (ev("lantern"), 0.18),
                   (ev("narrowest"), 0.14), (ev("watchfire_3"), 0.18), (ev("set_down"), 0.2), (ev("cadence"), 0.22),
                   (ev("title") + 2, 0.18), (ev("end_fade") - 0.5, 0.06), (ev("end_fade") + 0.9, 0.02))
    sync(S, roar, "the watch tone starts (cellos, D3)", "watch", "arrive:50", 0.25)
    # strings and low brass on Bb, swelling in under the roar (the effects own it)
    bb = {"cb": ["Bb1"], "vc": ["F3"], "vln2": ["F4", "Bb4"], "vln1": ["D5"]}
    for pn, ps in bb.items():
        hold(S, pn, ps, roar, run0 + 0.08 if pn != "cb" else run0 + 0.08)
        S.P(pn).d((roar, 0.06), (roar + 2.5, 0.2), (kf, 0.22), (run0 - 0.3, 0.24))
    for pn, p in (("tuba", "Bb1"), ("tbn_r", "Bb2"), ("tbn_l", "F3")):
        S.P(pn).n(p, roar + 0.5, run0 - roar - 0.4)
        S.P(pn).d((roar + 0.5, 0.06), (roar + 3, 0.2), (kf, 0.22), (run0 - 0.5, 0.2), (run0, 0.05))
    # the CALL in the violas on the roar (broad), then in canon, tighter at each far place: a horn at the karst,
    # the low strings at the desert (whose D becomes the Run's pedal)
    phrase(S, "call_vla", [("D3", 2), ("A3", 2), ("D4", 4)], roar)
    S.P("call_vla").d((roar - 0.1, 0.36), (roar + 2, 0.4), (roar + 6, 0.42), (roar + 8, 0.24), (roar + 8.5, 0.03))
    sync(S, roar, "the CALL in the violas on the roar", "call_vla", "arrive:50")
    kc = kf + 0.25                                        # 48 b4: the beat after the karst beacon flares
    phrase(S, "hn2", [("D3", 1), ("A3", 1), ("D4", 2)], kc)
    S.P("hn2").d((kc - 0.05, 0.3), (kc + 1, 0.34), (kc + 3, 0.32), (kc + 4.4, 0.04))
    sync(S, kc, "the karst: the CALL on a horn", "hn2", "arrive:50", 0.25)
    for pn, r in (("desert_vc", "D3"), ("desert_cb", "D2")):
        r = m(r)
        phrase(S, pn, [(r, 1), (r + 7, 1), (r + 12, 2)], df)
        S.P(pn).d((df - 0.05, 0.3), (df + 1, 0.33), (df + 2.5, 0.34), (df + 4.2, 0.03))
    sync(S, df, "the desert: the CALL in the low strings", "desert_vc", "arrive:50")


# ---------------------------------------------------------------------------
# A14: THE BEACON RUN (one chord per ignition, rising by thirds; call and answer)
# ---------------------------------------------------------------------------
RUN_CHORDS = [("Bb", (10, 2, 5)), ("Dm", (2, 5, 9)), ("F", (5, 9, 0)), ("Am", (9, 0, 4)), ("C", (0, 4, 7)),
              ("Em", (4, 7, 11)), ("G", (7, 11, 2)), ("Bm", (11, 2, 6))]


def contour_lead(chords, ranges, prev, top0, top1, w=1.5):
    """voice-lead chords (pitch-class tuples, root first) for len(ranges) voices, low to high, with the top voice
    drawn along a straight contour from top0 to top1 (MIDI) and penalised for falling; inner voices move least;
    every voicing keeps the root and the third"""
    import itertools
    out = []
    n = len(chords)
    for i, ch in enumerate(chords):
        tgt = top0 + (top1 - top0) * i / max(1, n - 1)
        pcs = set(c % 12 for c in ch)
        cands = [[p for p in range(lo, hi + 1) if p % 12 in pcs] for lo, hi in ranges]
        best = None
        for combo in itertools.product(*cands):
            if any(b <= a for a, b in zip(combo[:-1], combo[1:])):
                continue
            have = {x % 12 for x in combo}
            miss = len({ch[0] % 12, ch[1] % 12} - have)
            mv = sum(abs(a - b) for a, b in zip(combo[:-1], prev[:-1]))
            cost = 12 * miss + mv + w * abs(combo[-1] - tgt) + (3.0 * (prev[-1] - combo[-1]) if combo[-1] < prev[-1]
                                                                 else 0.0)
            cost += 0.5 * (len(combo) - len(have))
            if best is None or cost < best[0]:
                best = (cost, combo)
        prev = list(best[1])
        out.append(prev)
    return out


def run(S, bm, ev):
    run0, w = ev("run"), ev("watchers")
    fires = [ev(f"run_{k}") for k in range(1, 8)]
    light = ev("backs_lit")
    # the D pedal (basses), under the watch tone
    K.held(S, "cb", "D2", run0, light + 0.06, bow=8)
    S.P("cb").d((run0, 0.24), (fires[-1], 0.3), (w, 0.24), (light - 0.5, 0.18))
    # the upper strings, voice-led, one chord per ignition (legato: the change is not a stroke)
    times = [run0] + fires
    # top line D5 D5 F5 A5 G5 G5 B5 B5: the watch climbing the ranges, a third higher at each fire
    vs = contour_lead([c for _, c in RUN_CHORDS], [(57, 69), (62, 76), (65, 84)], [m("F4"), m("Bb4"), m("D5")], 74, 83)
    for pn, k in (("vla", 0), ("vln2", 1), ("vln1", 2)):
        for i, t in enumerate(times):
            t1 = times[i + 1] if i + 1 < len(times) else light
            S.P(pn).n(vs[i][k], t, t1 - t + 0.08, legato=i > 0, sync=True)
        pts = [(run0, 0.28)]
        for i, t in enumerate(fires):
            lv = 0.28 + 0.08 * i / 6
            pts += [(t - 0.2, lv), (t + 0.6, lv + 0.04)]
        pts += [(w, 0.24), (w + 4, 0.18), (light - 0.3, 0.1)]
        S.P(pn).d(*pts)
    for i, t in enumerate(fires):
        k = next(k for k in (2, 1, 0) if vs[i + 1][k] != vs[i][k])        # the highest voice that moves
        pn = ("vla", "vln2", "vln1")[k]
        sync(S, t, f"run fire {i + 1}: {RUN_CHORDS[i + 1][0]} ({pn})", pn, f"pitch:{vs[i + 1][k]}")
    # the horns: near horns call on fires 1 3 5 7, far horns answer on 2 4 6
    seats = {0: "hn", 2: "hn2", 4: "hn", 6: "hn2", 1: "hn_far", 3: "hn_farther", 5: "hn_far"}
    for i, t in enumerate(fires):
        pn = seats[i]
        if i % 2 == 0:
            K.call(S, pn, "D3", t, rhythm=(1, 1, 2), sync_first=True)
            v = 0.33 + 0.02 * i / 2
            S.P(pn).d((t - 0.05, v * 0.85), (t + 1, v), (t + 3, v * 0.95), (t + 4.3, 0.04))
            sync(S, t, f"run fire {i + 1}: the CALL ({pn})", pn, "arrive:50", 0.25)
        else:
            mode = "major" if i == 5 else "minor"
            K.answer(S, pn, "D4", t, mode=mode, rhythm=(1, .5, .5, 2), sync_first=True)
            v = 0.4 if i == 1 else 0.37
            S.P(pn).d((t - 0.05, v * 0.9), (t + 0.8, v), (t + 3.2, v * 0.9), (t + 4.3, 0.04))
            what = "THE ANSWER, first heard (a far horn)" if i == 1 else \
                ("the first MAJOR answer" if i == 5 else "the answer")
            sync(S, t, f"run fire {i + 1}: {what}", pn, "arrive:62", 0.25)


# ---------------------------------------------------------------------------
# A15: THE WATCHERS
# ---------------------------------------------------------------------------
def watchers(S, bm, ev):
    w, light = ev("watchers"), ev("backs_lit")
    # the corrupted call, softened, in the low brass (legato, p); a far horn answers with the CALL
    for pn, r in (("tbn_l", "D3"), ("tuba", "D2")):
        r = m(r)
        phrase(S, pn, [(r, 1.5), (r + 6, 1.5), (r + 12, 3.0)], w)
        v = 0.36 if pn == "tbn_l" else 0.28
        S.P(pn).d((w - 0.05, v * 0.8), (w + 1.5, v), (w + 4, v * 0.9), (w + 6.3, 0.04))
    sync(S, w, "THE WATCHERS: the corrupted call, softened (trombone)", "tbn_l", "arrive:50", 0.25)
    a = w + 4
    K.call(S, "hn_far", "D3", a, rhythm=(1, 1, 2), sync_first=True)
    S.P("hn_far").d((a - 0.05, 0.32), (a + 1, 0.33), (a + 3, 0.3), (a + 4.4, 0.04))
    sync(S, a, "the far horn answers with the CALL", "hn_far", "arrive:50", 0.25)


# ---------------------------------------------------------------------------
# A16-A17: TOWERS IN THE LIGHT; THE FIRE, SEEN
# ---------------------------------------------------------------------------
def light(S, bm, ev):
    bl, stop, go, others, lock, holds = (ev(x) for x in ("backs_lit", "surges_stop", "giants_open", "others_open",
                                                         "cycle_locks", "rim_holds"))
    seen, heart, lantern = ev("fire_seen"), ev("heart"), ev("lantern")
    # the basses' D; the violas hold the tritone until the giants open: it heals into the fifth
    K.held(S, "cb_q", "D2", bl, lantern + 0.08, bow=8)
    S.P("cb_q").d((bl, 0.12), (bl + 2, 0.18), (heart, 0.2), (lantern - 0.3, 0.2))
    S.P("vla_q").n("Ab3", bl, go - bl + 0.06, sync=True)
    S.P("vla_q").n("A3", go, heart - go + 0.06, legato=True, lantic=0.08)
    S.P("vla_q").n("D4", heart, lantern - heart + 0.06, legato=True)
    S.P("vla_q").d((bl, 0.1), (bl + 2, 0.2), (go - 0.3, 0.2), (go + 1, 0.24), (heart, 0.22), (lantern - 0.3, 0.2))
    sync(S, go, "the giants open: the tritone heals into the fifth (violas Ab3 -> A3)", "vla_q", "pitch:57")
    # the smaller towers follow, one by one, on the beat: the high strings enter (D sus2), then hold
    tw = [(others, "vln2_q", "E4"), (others + 1, "vln1_q", "A4"), (others + 2, "tw3", "D5"), (others + 3, "tw4", "E5")]
    for b, pn, p in tw:
        S.P(pn).n(p, b, heart - b + 0.06)
        S.P(pn).d((b, 0.04), (b + 0.9, 0.16), (lock, 0.2), (seen, 0.22), (heart - 0.2, 0.22))
    # from 61 b1 everything gathers into D: one small, intense heart
    for pn, p in (("vln2_q", "D4"), ("vln1_q", "D5"), ("tw3", "D5"), ("tw4", "D5")):
        S.P(pn).n(p, heart, lantern - heart + 0.06, legato=True)
        S.P(pn).d((heart + 1.5, 0.25), (lantern - 1.0, 0.16), (lantern - 0.1, 0.12))
    # the cycle: sixteenths while the towers still race (56), eighths once the giants open (57-58), quarter notes
    # locked to the bar from 59 b1; from the heart, only D
    # (v2: bar 56 is the race's own smothered cycle, as in the edge, fading as the surges stop; it clears into the
    #  glass as the giants open)
    wm = warm_morph(0.0, 0.0)
    cycle(S, "cycle_dk", CYC_RACE, bl, go, 0.25, 0.4, 0.22, pan_amp=0.35, sync_first=False, morph=DAMP)
    cycle(S, "cycle", CYC_HEAL, go, lock, 0.5, 0.22, 0.22, pan_amp=0.35, morph=wm, sync_first=False)
    cycle(S, "cycle", LOCK, lock, heart, 1.0, 0.24, 0.24, pan_amp=0.25, morph=wm, sync_first=True)
    cycle(S, "cycle", ("D5", "D6"), heart, lantern, 1.0, 0.24, 0.2, pan_amp=0.15, morph=wm, sync_first=False)
    sync(S, lock, "the cycle locks to the bar in quarter notes", "cycle", "hit", 0.1)


# ---------------------------------------------------------------------------
# A18: THE CROSSING
# ---------------------------------------------------------------------------
def crossing(S, bm, ev):
    lan = ev("lantern")
    wf = [ev(f"watchfire_{k}") for k in range(1, 5)]
    narrow, br, down = ev("narrowest"), ev("breath_dawn"), ev("set_down")
    # the WALK: pizzicato basses and a soft hand drum on 1 and 3; the harmony moves only at a watch-fire
    K.walk(S, "cb_pizz", "feet", lan, narrow, [(lan, "D2"), (wf[0], "Bb1"), (wf[1], "F2")], v=0.4, dv=0.28)
    K.walk(S, "cb_pizz", "feet", wf[2], br, [(wf[2], "G2"), (wf[3], "E2")], v=0.4, dv=0.28)
    S.P("feet").d((lan, 0.28))
    for n in S.P("cb_pizz").notes:                      # a footstep, damped: each pizz rings ~1.3 s, not into the next
        n.kw["maxlen"] = 1.3
    sync(S, lan, "THE CROSSING: the walk's first step (the drum)", "feet", "hit", 0.1)
    for k, r in zip((0, 1, 2, 3), ("Bb1", "F2", "G2", "E2")):
        sync(S, wf[k], f"watch-fire {k + 1}: the harmony moves (pizzicato {r})", "cb_pizz", "hit", 0.1)
    # the violins' pad over the violas' call: D sus2 | Bbmaj7 | F6/9 | (the narrowest: nothing) | G add9 | Em7
    pad = [(lan, ["E4", "A4"]), (wf[0], ["F4", "A4"]), (wf[1], ["F4", "C5"]), (narrow, None), (wf[2], ["B4", "D5"]),
           (wf[3], ["G4", "B4"]), (br + 0.02, None)]
    for i, (t, ps) in enumerate(pad[:-1]):
        if not ps:
            continue
        t1 = pad[i + 1][0]
        for pn, p in zip(("vln2_q", "vln1_q"), ps):
            S.P(pn).n(p, t, t1 - t + (0.08 if pad[i + 1][1] else 0.0), legato=(i > 0 and pad[i - 1][1] is not None),
                      sync=(i == 0))
    for pn in ("vln2_q", "vln1_q"):
        S.P(pn).d((lan, 0.16), (wf[0], 0.18), (narrow - 1.0, 0.16), (narrow, 0.04), (wf[2], 0.1), (wf[2] + 2, 0.2),
                  (wf[3], 0.22), (br - 0.2, 0.2))
    # the call inside the chord, in the violas, at the lantern and at every watch-fire; in the narrowest stretch
    # its D4 is held alone (one viola, doubled by a second desk), with the wind and the watch tone
    starts = [lan, wf[0], wf[1], wf[2], wf[3]]
    for i, t in enumerate(starts):
        nxt = starts[i + 1] if i + 1 < len(starts) else br
        rh = (2, 2, nxt - t - 4) if i + 1 < len(starts) else (1, 1, br - t - 2 + 0.02)
        phrase(S, "call_vlaq", [("D3", rh[0]), ("A3", rh[1]), ("D4", rh[2] + (0.08 if i + 1 < len(starts) else 0))],
               t, legato=True, sync_first=True)
        sync(S, t, f"the call inside the chord (violas) at {'the lantern' if i == 0 else f'watch-fire {i}'}",
             "call_vlaq", "arrive:50")
    # each call is a new bow (a legato octave drop arrives half a second late in the quiet violas); its low D3
    # speaks slowly, so it starts early by its measured arrival (SYNC_TRIM)
    S.P("call_vlaq").d((lan, 0.24), (wf[0], 0.26), (narrow, 0.27), (narrow + 4, 0.28), (wf[2] - 0.5, 0.26),
                       (wf[3], 0.28), (br - 0.2, 0.26))
    # the narrowest stretch doubled by a second desk (the exposed line)
    S.P("call_vlaq2").n("D4", narrow, wf[2] - narrow + 0.06, sync=True)
    S.P("call_vlaq2").d((narrow, 0.1), (narrow + 3, 0.2), (wf[2] - 1, 0.18), (wf[2], 0.06))
    # the cycle walking in step (quarter notes), warming from glass toward gold; silent in the narrowest stretch
    cycle(S, "cycle", LOCK, lan, narrow, 1.0, 0.14, 0.13, pan_amp=0.25, morph=warm_morph(0.0, 0.45),
          sync_first=False)
    cycle(S, "cycle", LOCK, wf[2], br, 1.0, 0.13, 0.15, pan_amp=0.25, morph=warm_morph(0.6, 0.9),
          sync_first=False)


# ---------------------------------------------------------------------------
# A19-A20: THE BLUE HOUR; TITLE
# ---------------------------------------------------------------------------
def blue_hour(S, bm, ev):
    down, rose, valley, cad, title, fade = (ev(x) for x in ("set_down", "rose", "valley", "cadence", "title",
                                                              "end_fade"))
    end = bm.bars * 4
    last = fade + 0.9
    # a bloom out of the breath on D add9 (no stroke); the violins' A4 becomes the ANSWER at 76 b1
    call0, ans0, home0 = down + 2, valley, valley + 4                    # 74 b3, 76 b1, 77 b1
    bloom = {"cb": ["D2"], "vc": ["A2"], "vla": ["F#3"], "vln2": ["E4"]}
    for pn, ps in bloom.items():
        hold(S, pn, ps, down, ans0 + 2.06, sync=True)
    S.P("vln1").n("A4", down, ans0 - down + 0.06, sync=True)
    sync(S, down, "the blue hour: a bloom out of the silence (score stem)", "score", "bloom", 0.15)
    # the harmony under the ANSWER and HOME: D (maj7 passing) | Bm7 | G | A | A7 | D, the first real cadence
    prog = [(ans0 + 2, {"cb": "B1", "vc": "F#2", "vla": "D3", "vln2": "A3"}),
            (home0, {"cb": "G1", "vc": "D3", "vla": "B3", "vln2": "G3"}),
            (home0 + 1, {"cb": "A1", "vc": "E3", "vla": "C#4", "vln2": "A3"}),
            (home0 + 2, {"cb": "A1", "vc": "E3", "vla": "C#4", "vln2": "G3"}),
            (cad, {"cb": "D2", "vc": "A2", "vla": "F#3", "vln2": "A3"})]
    for i, (t, ch) in enumerate(prog):
        t1 = prog[i + 1][0] if i + 1 < len(prog) else title
        for pn, p in ch.items():
            prev = (prog[i - 1][1] if i else bloom)[pn]
            prev = prev[0] if isinstance(prev, list) else prev
            S.P(pn).n(p, t, t1 - t + 0.06, legato=True, lantic=(-0.02 if t == cad else 0.02))
    # A20: D add9 on the title's bar, ringing out to silence
    for pn, p in (("cb", "D2"), ("vc", "A2"), ("vla", "F#3"), ("vln2", "E4")):
        S.P(pn).n(p, title, last - title, legato=True)
    lv = {"cb": 0.22, "vc": 0.24, "vla": 0.25, "vln2": 0.24}
    for pn, L in lv.items():
        S.P(pn).d((down, 0.08), (down + 1.4, L * 1.12), (down + 5, L), (rose, L), (valley, L * 0.95),
                  (home0, L * 0.95), (cad, L * 1.1), (cad + 2, L), (title, L * 0.9), (title + 4, L * 0.55),
                  (last - 2, 0.06), (last - 0.2, 0.02))
    # the whole theme, for the first time in A: the CALL on the horns and the violas (74 b3)
    for pn, r in (("hn", "D3"), ("theme_vla", "D3")):
        K.call(S, pn, r, call0, rhythm=(2, 2, 2.3), sync_first=True)
        v = 0.36 if pn == "hn" else 0.3
        S.P(pn).d((call0 - 0.05, v), (call0 + 2, v), (call0 + 4, v * 1.05), (ans0 + 0.2, v * 0.8),
                  (ans0 + 0.6, 0.03))
    sync(S, call0, "the whole theme: the CALL (horn)", "hn", "arrive:50", 0.25)
    # the ANSWER on the violins (the bloom's A4 steps up to D5), with a far horn an octave under
    phrase(S, "vln1", [("D5", 1.0), ("C#5", 0.5), ("B4", 0.5), ("F#4", 2.0)], ans0, legato=True, sync_first=False)
    S.P("vln1").notes[-4].legato = True                  # the bloom's A4 steps up to D5
    K.answer(S, "hn_far", "D4", ans0, mode="major", rhythm=(1, .5, .5, 2), sync_first=True)
    S.P("hn_far").d((ans0 - 0.05, 0.36), (ans0 + 1, 0.4), (ans0 + 3.4, 0.34), (ans0 + 4.3, 0.04))
    sync(S, ans0, "the whole theme: the ANSWER (the violins step to D5)", "vln1", "pitch:74")
    # HOME on the violins and a horn, arriving on D as the cadence falls (78 b1)
    pass_b = [f / 20.0 for f in PASS_FRAMES]
    assert cad + 2 <= pass_b[0] and all(a < b for a, b in zip(pass_b, pass_b[1:])) and pass_b[-1] < fade - 1, pass_b
    phrase(S, "vln1", [("B4", 1), ("A4", 1), ("F#4", 2), ("D4", pass_b[0] + 1.5 - cad)], home0, legato=True,
           sync_first=False)
    S.P("vln1").notes[-4].legato = True
    phrase(S, "hn", [("B3", 1), ("A3", 1), ("F#3", 2), ("D3", 3.0)], home0)
    S.P("hn").d((home0 - 0.05, 0.36), (home0 + 1, 0.38), (cad, 0.4), (cad + 2, 0.32), (cad + 3.2, 0.04))
    S.P("vln1").d((down, 0.07), (down + 1.4, 0.26), (ans0 - 0.3, 0.27), (ans0 + 0.4, 0.42), (home0, 0.43),
                  (cad, 0.44), (cad + 2, 0.34), *sorted([(pass_b[0], 0.26), (pass_b[0] + 1.4, 0.05)]))
    sync(S, home0, "the whole theme: HOME (the horn)", "hn", "arrive:59", 0.25)
    sync(S, cad, "the first real cadence: A7 -> D (the basses)", "cb", "pitch:38")
    sync(S, cad, "the first real cadence: HOME arrives on D4 (violins)", "vln1", "pitch:62")
    # A20: the ANSWER passed outward over the ringing D add9, horn -> far -> farther -> farthest, each then silent
    # (the violins hand HOME's D to the first); the last one fades with the picture
    for i, ((pn, p, v), t) in enumerate(zip(PASS_NOTES, pass_b)):
        t1 = pass_b[i + 1] if i + 1 < len(pass_b) else fade
        S.P(pn).n(p, t, t1 - t + (0.6 if i + 1 < len(pass_b) else 0.9), sync=True)
        tail = [(t1, v * 0.7), (t1 + 0.6, 0.02)] if i + 1 < len(pass_b) else [(fade - 0.5, v * 0.6), (last - 0.3, 0.02)]
        S.P(pn).d((t - 0.05, v * 0.8), (t + 0.8, v), *tail)
        sync(S, t, f"A20: the ANSWER passed, {p} ({pn}), then silent", pn, f"arrive:{m(p)}", 0.25)
    # A11's pure A5 (the false dawn's beating A, made pure) returns, ppp, as the rose sky pales; out with the fade
    hr = HARM_RETURN_FRAME / 20.0
    S.P("harmonic").n("A5", hr, last - 0.1 - hr, atk=2.2, rel=1.6)
    S.P("harmonic").d((hr, 0.03), (hr + 2.5, 0.1), (fade - 0.5, 0.08), (last - 0.3, 0.02))
    S.P("harm_str").n("A5", hr + 0.5, last - 0.3 - hr - 0.5)
    S.P("harm_str").d((hr + 0.5, 0.02), (hr + 3, 0.05), (fade - 0.5, 0.04), (last - 0.4, 0.015))
    # the cycle, warm, inside the bloom; it stops as the chord rings out
    cycle(S, "cycle", LOCK, down, title + 4, 1.0, 0.11, 0.1, pan_amp=0.3, morph=warm_morph(1.0, 1.0),
          sync_first=False)
