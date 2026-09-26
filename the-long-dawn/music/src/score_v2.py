"""THE LONG DAWN v2 - the scores, as code.

    build("AB")  -> score AB (cuts A and B: the first half's sound world, carried on)
    build("C")   -> score C  (cut C: the epic-fantasy tradition)

Bars 1-15 (v2 frames 0-1199) are the v1 first half, called unchanged from score.py
(intro, kindling, race, silence).  From v2 frame 1200 each score writes its own second
half to the v2 sync table (timeline_v2.py / BIBLE_V2 section 3).

Global beats: gb(bar, beat) ; frames: fb(frame) ; 1 beat = 20 frames = 40 000 samples.
Every sync-critical note uses sync=True (never humanised).

THE MATERIAL (both scores)
  the CALL      root - fifth - octave (D4 A4 D5), the Beacon theme's first phrase
  the ANSWER    the theme's second phrase (D5 C5 Bb4 F4 over Bb)
  the THINKING  the KINDLING's 7-note cycles on a 16th grid (cyc6a = D5 E5 A5 D6 E6 A5 F5);
                in AB it returns warm as the world answers (a web of cycles of different
                lengths drifting against each other), and when the emissaries' flames merge
                (2160) every voice LOCKS into one 8-note cycle that fits the bar.

SCORE AB, second half (bars 16-37)
  FIRST BEACON  D pedal in the dark (only the piano's fifths from the SILENCE ringing);
                1318 Bb/D blooms in low strings; 1340 C/D; 1360 the roar: organ + strings +
                low brass open on Bb, the CALL in half notes inside the texture (horns + violas)
  FAR PEAK      Gm; a pulse starts; 1480 the shepherd = the ANSWER (C Bb -> F), deep hit
  BEACON RUN    F major; organ + string 16ths, bass climbing a step per beacon
                (F G A Bb C D E F), one deep hit per beacon, top line rising in 10ths
  MONTAGE       harmony moves by thirds, one chord per ignition: F Bb Gm Eb C F/A
  WORLD ANSWERS Bbmaj9(#11) - the IGNITION chord, now in root position: a web of warm
                cycles; the theme's ANSWER + third phrase inside the texture
  ACCORD        D minor hush (the thinking cycle alone, on piano); 2160 the lock; a long
                build over a D pedal (Dm - Bb/D - C/D, changes on off-beats, never on an oath),
                2320 Bb, 2360 C, hearth flare, BREATH
  DAWN          2400 D major: organ, strings, low brass; the CALL broad (half notes) and the
                answer; the thinking cycle glowing high in D major; bass D B G A G F#
  CODA          the piano (the SILENCE's instrument) asks the call again; the D5 hangs
                through the question; the answer during the hand-over; 2800 warm swell (G/D);
                answering fires = soft swells building D add9; 2880 final chord
"""
import numpy as np

import score as V1
from dsl import Part, m
from timeline_v2 import fb, gb, RUN_BEACONS, MONTAGE_IGN

PARTS = V1.PARTS
P = V1.P

# Breaths: every part, the hall AND the SFX ducked to -45 dB over the window, except the
# exempt parts/events.  (start_beat, end_beat, exempt)   - applied by render_v2.
BREATHS = [
    (fb(640) - 0.24, fb(640), {"revcym"}),     # RACE drums: 200 ms breath (v1 design)
    (fb(1037), fb(1040), {"revcym"}),          # the suck before the IMPACT (v1 design)
    (fb(2400) - 0.33, fb(2400), set()),        # CLIMAX: 275 ms of real silence
]

# mix-time EQ per part: name -> (highpass Hz or None, lowpass Hz or None)
MIX_EQ = {}
# bus groups: parts whose names start with the key are summed (with their own hall) and put
# through a true-peak bus limiter at ceiling_db (score-mix domain, before the master)
BUS_GROUPS = {"v2r_": dict(ceiling_db=-4.0, release=0.08)}


# ---------------------------------------------------------------------------
# parts
# ---------------------------------------------------------------------------
def clone(new, old, **over):
    """A new part seated (and fadered) exactly like a v1 part."""
    o = PARTS[old]
    kw = dict(kind=o.kind, bus=o.bus, pan=o.pan, width=o.width, gain_db=o.gain_db, send=o.send,
              depth=o.depth, humanize_ms=o.humanize_ms, params=dict(o.params))
    inst = over.pop("inst", o.inst)
    kw.update(over)
    PARTS[new] = Part(new, inst, **kw)
    return PARTS[new]


def add(name, inst, **kw):
    PARTS[name] = Part(name, inst, **kw)
    return PARTS[name]


def setup_ab():
    for s in ("vln1", "vln2", "vla", "vc", "cb", "svln"):
        clone("ab_" + s, s)
    PARTS["ab_svln"].gain_db = V1.FADERS["svln"] + 4.0
    clone("ab_vln1b", "vln1", pan=-0.45, gain_db=V1.FADERS["vln1"] - 2.0)      # divisi / second desk
    for s in ("vln2_sp", "vla_sp", "vc_sp", "cb_sp"):
        clone("ab_" + s, s)
    clone("ab_vln1_sp", "vln1_sp")
    clone("ab_vln_tr", "vln_tr")
    clone("ab_vla_tr", "vla_tr", gain_db=-1.0)
    clone("ab_vc_tr", "vc_tr")
    clone("ab_hn1", "hns", pan=-0.42, gain_db=V1.FADERS["hns"] + 2.0)
    clone("ab_hn2", "hns", pan=-0.22, humanize_ms=12, gain_db=V1.FADERS["hns"] + 0.5)
    clone("ab_hn34", "hns2", pan=-0.3, humanize_ms=14, gain_db=V1.FADERS["hns"] + 0.5)   # horns 3-4
    clone("ab_hnfar", "hn_solo", pan=0.62, depth=0.95, send=0.75, gain_db=V1.FADERS["hn_solo"] - 3.0)  # the far horn
    clone("ab_tbn", "tbn")
    clone("ab_tuba", "tuba")
    clone("ab_timp", "timp")
    clone("ab_timp_roll", "timp_roll")
    clone("ab_bdrum", "bdrum", gain_db=V1.FADERS["bdrum"] - 3.0)
    clone("ab_bdroll", "bdrum", inst="bdrum_roll", gain_db=2.0)
    clone("ab_piano", "piano")
    clone("ab_pnolo", "piano", gain_db=-3.0, width=0.9)          # low piano octaves under the hits
    clone("ab_pnoarp", "piano", gain_db=+3.0, width=0.9)         # the locked cycle in the ACCORD
    clone("ab_vcm", "vc", pan=0.28)                               # cellos on the melody
    clone("ab_cl", "cl")
    clone("ab_fl", "fl", gain_db=V1.FADERS["fl"] - 2.0)
    # the organ (sampled, Ivy Audio / VSCO-2-CE): far, in the hall
    add("ab_org", "organ", bus="keys", pan=0.0, width=1.0, depth=0.75, send=0.48, gain_db=-2.0, humanize_ms=4)
    add("ab_orgarp", "organ", bus="keys", pan=0.0, width=1.0, depth=0.7, send=0.42, gain_db=+1.0, humanize_ms=0)
    add("ab_orgq", "organ_q", bus="keys", pan=0.05, width=1.0, depth=0.7, send=0.5, gain_db=+9.0, humanize_ms=4)
    add("ab_orgped", "organ_ped", bus="keys", pan=0.0, width=1.0, depth=0.8, send=0.42, gain_db=-5.0,
        humanize_ms=0)
    add("ab_orgpedq", "organ_pedq", bus="keys", pan=0.0, width=1.0, depth=0.8, send=0.45, gain_db=+1.0,
        humanize_ms=0)
    MIX_EQ["ab_orgped"] = (None, 2400.0)
    # the thinking colour, warm
    X = dict(kind="synth", bus="synth", humanize_ms=0)
    add("ab_warm", "fmwarm", pan=0.3, width=1.0, depth=0.45, send=0.5, gain_db=-5.0,
        params=dict(ratio=1.0, index=0.7, decay=2.4, lp=4800.0), **X)
    add("ab_warm2", "fmwarm", pan=-0.35, width=1.0, depth=0.5, send=0.55, gain_db=-6.0,
        params=dict(ratio=2.0, index=0.35, decay=3.2, lp=2600.0, chorus=6.0), **X)
    add("ab_deep", "deephit", pan=0.0, width=1.0, depth=0.35, send=0.3, gain_db=-10.0, **X)
    add("ab_morph", "fmwarm", pan=0.0, width=1.0, depth=0.45, send=0.48, gain_db=-7.0,
        params=dict(ratio=1.0, index=0.6, decay=1.6, lp=5000.0), **X)
    add("ab_sub", "subpulse", pan=0.0, width=0.0, depth=0.0, send=0.0, gain_db=-2.0, **X)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def N(pn, pitch, start, dur, vel=None, **kw):
    return P(pn).n(pitch, start, dur, vel, **kw)


def hold(pn, pitches, b0, b1, vel=None, **kw):
    for p in pitches:
        P(pn).n(p, b0, b1 - b0, vel, **kw)


def chords(pn, seq, overlap=0.06, legato=True, **kw):
    """seq: [(start_beat, [pitches])...(end_beat, None)] -> held chords, legato between."""
    for i in range(len(seq) - 1):
        t0, ps = seq[i]
        t1 = seq[i + 1][0]
        if ps is None:
            continue
        for p in ps:
            P(pn).n(p, t0, t1 - t0 + overlap, legato=(legato and i > 0), **kw)


def cycle(pn, pitches, b0, b1, step, vels, pans=None, sync_first=False, **kw):
    """repeat a pitch cycle on a grid from b0 to b1 (exclusive)"""
    t, i = b0, 0
    while t < b1 - 1e-6:
        v = vels(i, t) if callable(vels) else vels
        pan = None if pans is None else (pans(i, t) if callable(pans) else pans)
        p = pitches[i % len(pitches)]
        if p is not None:
            P(pn).n(p, t, step, v, pan=pan, sync=(sync_first and i == 0), **kw)
        t += step
        i += 1
    return i


def lin(t, t0, t1, v0, v1):
    u = min(1.0, max(0.0, (t - t0) / (t1 - t0)))
    return v0 + (v1 - v0) * u


# ---------------------------------------------------------------------------
# SCORE AB
# ---------------------------------------------------------------------------
def ab_first_beacon():
    k, c2, roar = fb(1318), fb(1340), fb(1360)
    # --- the dark (1200-1317): only a low D breathing under the piano's dying fifths
    N("ab_cb", "D2", gb(16, 1.5), roar - gb(16, 1.5) + 0.06)
    N("ab_vc", "D3", gb(16, 3.0), roar - gb(16, 3.0) + 0.06)
    N("ab_sub", "D1", gb(16, 2.0), roar - gb(16, 2.0), atk=2.5, rel=0.5)
    P("ab_sub").d((gb(16, 2), 0.06), (gb(17, 1), 0.14), (k, 0.18), (roar - 0.2, 0.3), (roar, 0.1))
    # --- 1318 the kindling catches: Bb/D blooms ; 1340 C/D (sus) ; strings crescendo to the roar
    chords("ab_vc", [(k, ["F3"]), (c2, ["G3"]), (roar, None)], overlap=0.06)
    chords("ab_vla", [(k, ["Bb3"]), (c2, ["C4"]), (roar, None)], overlap=0.06)
    chords("ab_vln2", [(k, ["D4", "F4"]), (c2, ["E4", "G4"]), (roar, None)], overlap=0.06)
    for pn, a in (("ab_cb", 0.0), ("ab_vc", 0.0), ("ab_vla", 0.02), ("ab_vln2", 0.04)):
        P(pn).d((gb(16, 1.5), 0.05), (gb(16, 3.5), 0.13 - a), (gb(17, 1.5), 0.1), (k - 0.1, 0.12),
                (k + 0.6, 0.24 - a), (c2, 0.3), (roar - 0.12, 0.52), (roar, 0.55))
    # --- 1360 THE ROAR: strings + low brass open on Bb - no organ manuals, the 16' pedal as a floor.
    # The CALL sounds inside the string texture (violas, half notes); the shepherd answers at 1480
    # with the theme's answer (C - Bb -> F at the run).  No strokes: the SFX roar owns the fire.
    # harmony: 1360 Bb | 1400 F/A | 1440 Gm (far peak) | 1480 C9sus4 | 1500 C9 | 1520 Dm (the run)
    seq = [(roar, "Bb1", ["Bb2", "F3"], ["F4", "Bb4"], ["D5", "F5"]),
           (fb(1400), "A1", ["A2", "F3"], ["F4", "C5"], ["F5", "A5"]),
           (fb(1440), "G1", ["G2", "D3"], ["G4", "Bb4"], ["D5", "G5"]),
           (fb(1480), "C2", ["C3", "G3"], ["G4", "Bb4"], ["D5", "F5"]),
           (fb(1500), "C2", ["C3", "G3"], ["G4", "Bb4"], ["D5", "E5"])]
    end = fb(1520)
    for i, (t, cb, vc, v2, v1) in enumerate(seq):
        t1 = seq[i + 1][0] if i + 1 < len(seq) else end
        lg = i > 0
        if not (i == 4):
            N("ab_cb", cb, t, (t1 - t) + (0.06 if i < 3 else 1.06), legato=lg)
        for p in vc:
            N("ab_vc", p, t, t1 - t + 0.06, legato=lg)
        for p in v2:
            N("ab_vln2", p, t, t1 - t + 0.06, legato=lg)
        for p in v1:
            N("ab_vln1", p, t, t1 - t + 0.06, legato=lg)
    P("ab_vln1").d((roar, 0.5), (roar + 1, 0.62), (fb(1440), 0.58), (fb(1480), 0.62), (end - 0.1, 0.7))
    for pn, base in (("ab_cb", 0.7), ("ab_vc", 0.7), ("ab_vln2", 0.64)):
        P(pn).d((roar, base), (roar + 1.5, base - 0.04), (fb(1440), base - 0.1), (fb(1480), base - 0.06),
                (end - 0.1, base + 0.08))
    # the call in the violas, mezzo-forte, inside the chord
    P("ab_vla").line("D4:2 A4:2 D5:2 C5:1 Bb4:1", roar)
    N("ab_vla", "F4", fb(1520), 1.04, legato=True)
    P("ab_vla").d((roar, 0.56), (roar + 1.0, 0.62), (fb(1440), 0.64), (fb(1480), 0.66), (end, 0.68),
                  (end + 1, 0.5))
    # the floor: the 16' pedal only
    chords("ab_orgped", [(roar, ["Bb2"]), (fb(1400), ["A2"]), (fb(1440), ["G2"]), (fb(1480), ["C3"]), (end, None)],
           overlap=0.04, legato=False)
    P("ab_orgped").d((roar, 0.42), (roar + 2, 0.5), (fb(1480), 0.5), (end - 0.1, 0.56))
    # low brass: warm sustain under the roar
    chords("ab_tbn", [(roar, ["Bb2", "F3", "D3"]), (fb(1400), ["A2", "F3", "C3"]), (fb(1440), ["G2", "D3", "Bb2"]),
                      (fb(1480), ["C3", "G3", "Bb2"]), (end, None)], overlap=0.05)
    P("ab_tbn").d((roar, 0.38), (roar + 1.5, 0.54), (fb(1440), 0.48), (fb(1480), 0.52), (end - 0.1, 0.6))
    chords("ab_tuba", [(roar, ["Bb1"]), (fb(1400), ["A1"]), (fb(1440), ["G1"]), (fb(1480), ["C2"]), (end, None)],
           overlap=0.05)
    P("ab_tuba").d((roar, 0.34), (roar + 1.5, 0.48), (fb(1480), 0.48), (end - 0.1, 0.56))
    # the pulse begins under the far peak: low strings 8ths, then violas 16ths, a timpani roll
    t = fb(1440)
    k8 = 0
    while t < end - 1e-6:
        prog = (t - fb(1440)) / 4
        bass = "G2" if t < fb(1480) else "C3"
        N("ab_vc_sp", bass, t, 0.5, 0.3 + 0.35 * prog + (0.08 if k8 % 2 == 0 else 0))
        N("ab_cb_sp", "G1" if t < fb(1480) else "C2", t, 0.5, 0.28 + 0.3 * prog)
        t += 0.5
        k8 += 1
    cycle("ab_vla_sp", ["G3", "Bb3", "C4", "Bb3"], fb(1480), end, 0.25,
          lambda i, t: 0.3 + 0.28 * (t - fb(1480)) / 2 + (0.08 if i % 4 == 0 else 0))
    N("ab_timp_roll", "C3", fb(1500), end - fb(1500) - 0.02, rel=0.05)
    P("ab_timp_roll").d((fb(1500), 0.18), (end - 0.05, 0.7))


# ---- THE BEACON RUN ---------------------------------------------------------
# A pedal on D (the film's tonic; the shepherd's C7 lands on it deceptively), and above it the
# harmony moves by thirds, two beats a chord: Dm - F/D - Am/D - C/D (-> F at the desert).  The
# string 16ths follow the chords, so the texture rises by thirds, not by a scale.  Horns call
# near and answer far.  No stroke on any beacon: the SFX whoomps own the beacons.
RUN = [  # frame, violas 16ths, 2nd violins 16ths, sustained 1st violins, horn (part, call root)
    (1520, ["D4", "F4", "A4", "F4"], ["A4", "D5", "F5", "D5"], ["A5"], ("ab_hn1", "D4")),
    (1560, ["F4", "A4", "C5", "A4"], ["C5", "F5", "A5", "F5"], ["A5"], ("ab_hnfar", "F4")),
    (1600, ["A4", "C5", "E5", "C5"], ["E5", "A5", "C6", "A5"], ["A5"], ("ab_hn1", "A3")),
    (1640, ["C5", "E5", "G5", "E5"], ["E5", "G5", "C6", "G5"], ["G5"], ("ab_hnfar", "C4")),
]


def pc_in(names, lo, hi):
    """all pitches of pitch classes `names` between lo and hi (midi)"""
    pcs = {m(n + "4") % 12 for n in names}
    return [p for p in range(m(lo), m(hi) + 1) if p % 12 in pcs]


def ab_run():
    r0, r1 = fb(1520), fb(1680)
    # ---- the pedal: basses arco + the organ's 16' + tuba, trombones on the open fifth
    N("ab_cb", "D2", r0, r1 - r0 + 0.04, legato=True)
    N("ab_orgped", "D2", r0, r1 - r0)
    N("ab_tuba", "D2", r0, r1 - r0, legato=True)
    for p in ("D3", "A3"):
        N("ab_tbn", p, r0, r1 - r0)
    P("ab_cb").d((r0, 0.62), (r1 - 0.2, 0.8))
    P("ab_orgped").d((r0, 0.5), (r1 - 0.2, 0.66))
    P("ab_tuba").d((r0, 0.44), (r1 - 0.2, 0.62))
    P("ab_tbn").d((r0, 0.36), (r1 - 0.2, 0.6))
    # ---- the drive: cellos + basses 8ths on the pedal (spiccato), crescendo
    for k in range(16):
        t = r0 + 0.5 * k
        v = min(1.0, 0.66 + 0.22 * k / 15 + (0.08 if k % 2 == 0 else 0))
        N("ab_vc_sp", "D3", t, 0.5, v, sync=(k == 0))
        N("ab_cb_sp", "D2", t, 0.5, max(0.3, v - 0.06))
    # ---- the harmony by thirds: violas and 2nd violins in 16ths, 1st violins sustained above
    for i, (f, vla, vln2, v1, hn) in enumerate(RUN):
        t = fb(f)
        t1 = fb(RUN[i + 1][0]) if i + 1 < len(RUN) else r1
        k = 0
        tt = t
        while tt < t1 - 1e-6:
            prog = (tt - r0) / (r1 - r0)
            acc = 0.1 if k % 4 == 0 else 0.0
            N("ab_vla_sp", vla[k % 4], tt, 0.25, min(1.0, 0.64 + 0.26 * prog + acc))
            N("ab_vln2_sp", vln2[k % 4], tt, 0.25, min(1.0, 0.6 + 0.26 * prog + acc))
            tt += 0.25
            k += 1
        for p in v1:
            N("ab_vln1", p, t, t1 - t + (0.05 if i + 1 < len(RUN) else 0.0), legato=i > 0)
    P("ab_vln1").d((r0, 0.5), (r1 - 0.3, 0.72))
    # ---- the horns: a call near, answered far - rising by thirds with the harmony (D F A C)
    for i, (f, vla, vln2, v1, (hn, root)) in enumerate(RUN):
        r = m(root)
        t = fb(f)
        for j, (dp, d) in enumerate(zip((0, 7, 12), (0.5, 0.5, 0.95))):
            N(hn, r + dp, t + 0.5 * j, d, legato=j > 0, sync=(j == 0))
    P("ab_hn1").d((r0, 0.68), (r0 + 2, 0.7), (fb(1600), 0.74), (r1, 0.64))
    P("ab_hnfar").d((r0, 0.66), (r1, 0.72))
    # ---- a timpani roll on the pedal grows through the second half of the run (no strokes)
    N("ab_timp_roll", "D2", fb(1600), fb(1677) - fb(1600), rel=0.08)
    P("ab_timp_roll").d((fb(1600), 0.08), (fb(1676), 0.5))


# ---- MONTAGE: harmony moves by thirds, one chord per ignition ----------------
MONT = [  # start frame, bass, chord voicing (low -> high, strings/organ), top (vln1)
    (1680, "F1", ["F2", "C3", "A3", "C4", "F4", "A4"], "C6"),
    (1700, "Bb1", ["Bb2", "F3", "D4", "F4", "Bb4", "C5"], "D6"),
    (1760, "G1", ["G2", "D3", "Bb3", "D4", "G4", "D5"], "D6"),
    (1810, "Eb2", ["Eb2", "Bb2", "G3", "Bb3", "Eb4", "G4"], "D6"),
    (1850, "C2", ["C2", "G2", "E3", "G3", "C4", "G4"], "D6"),
    (1890, "A1", ["A2", "F3", "C4", "F4", "A4", "C5"], "C6"),
]


def ab_montage():
    ends = [fb(f) for f, *_ in MONT[1:]] + [fb(1920)]
    for i, (f, bass, voic, top) in enumerate(MONT):
        t, t1 = fb(f), ends[i]
        lg = i > 0
        N("ab_cb", bass, t, t1 - t + 0.06, legato=True)
        N("ab_vc", voic[0], t, t1 - t + 0.06, legato=True)
        N("ab_vc", voic[1], t, t1 - t + 0.06, legato=lg)
        for p in voic[2:4]:
            N("ab_vla", p, t, t1 - t + 0.06, legato=lg)
        for p in voic[4:]:
            N("ab_vln2", p, t, t1 - t + 0.06, legato=lg)
        N("ab_vln1", top, t, t1 - t + 0.06, legato=True)
        N("ab_vln1b", top, t, t1 - t + 0.06, legato=True)
        N("ab_orgped", m(bass) + 12, t, t1 - t + 0.03, sync=(i > 0))
    # the pulse carries on in 8ths (organ quiet flute + string spiccato), growing into the globe
    for i, (f, bass, voic, top) in enumerate(MONT):
        t0, t1 = fb(f), ends[i]
        tones = sorted(set(m(x) for x in voic[2:]))
        pat = [tones[0], tones[2], tones[1], tones[3] if len(tones) > 3 else tones[2]]
        t = np.ceil(t0 * 2 - 1e-6) / 2
        j = 0
        while t < t1 - 1e-6:
            prog = (t - fb(1680)) / 12
            v = 0.46 + 0.22 * prog + (0.08 if abs(t - round(t)) < 1e-6 else 0)
            N("ab_vla_sp", pat[j % 4], t, 0.5, v)
            N("ab_vln2_sp", m(pat[(j + 2) % 4]) + 12, t, 0.5, v * 0.92)
            N("ab_vc_sp", m(voic[0]), t, 0.5, v * 0.95)
            t += 0.5
            j += 1
    for pn, a, b in (("ab_cb", 0.64, 0.72), ("ab_vc", 0.62, 0.7), ("ab_vla", 0.56, 0.68), ("ab_vln2", 0.54, 0.68),
                     ("ab_vln1", 0.66, 0.76), ("ab_vln1b", 0.6, 0.7)):
        P(pn).d((fb(1680) + 0.3, a), (fb(1700), a - 0.04), (fb(1900), b), (fb(1918), b + 0.04))
    # a slow horn line inside the texture, rising with the world (one chord tone per ignition),
    # falling to A over F/A so that the World Answers' horn (D4) answers it
    hl = [(1700, "F4"), (1760, "G4"), (1810, "Bb4"), (1850, "C5"), (1890, "A4"), (1918, None)]
    for j in range(len(hl) - 1):
        f, p = hl[j]
        N("ab_hn1", p, fb(f), fb(hl[j + 1][0]) - fb(f) + (0.04 if j < len(hl) - 2 else 0.0), legato=j > 0)
    P("ab_hn1").d((fb(1700), 0.34), (fb(1760), 0.4), (fb(1850), 0.48), (fb(1890), 0.44), (fb(1916), 0.36))
    P("ab_orgped").d((fb(1680), 0.5), (fb(1700), 0.42), (fb(1915), 0.52))


# ---- THE WORLD ANSWERS: the thinking fire's cycles, warm, a web --------------
def ab_world():
    w0, w1, w2 = fb(1920), fb(2000), fb(2080)
    # harmony: Bbmaj9(#11) root position (the IGNITION chord, grounded) -> F(add9) -> (Dm at 2080)
    N("ab_cb", "Bb1", w0, w1 - w0 + 0.06, legato=True, sync=True)
    N("ab_cb", "F1", w1, w2 - w1 + 0.06, legato=True)
    chords("ab_vc", [(w0, ["Bb2", "F3"]), (w1, ["F2", "C3"]), (w2, None)])
    chords("ab_vla", [(w0, ["D4", "F4"]), (w1, ["C4", "F4"]), (w2, None)])
    chords("ab_vln2", [(w0, ["C5", "E5"]), (w1, ["C5", "G5"]), (w2, None)])
    chords("ab_vln1", [(w0, ["A5"]), (w1, ["G5"]), (w1 + 2, ["A5"]), (w2, None)])
    for pn, lv0 in (("ab_cb", 0.42), ("ab_vc", 0.42), ("ab_vla", 0.4), ("ab_vln2", 0.38), ("ab_vln1", 0.42)):
        P(pn).d((w0, lv0 + 0.1), (w0 + 1.5, lv0), (w1, lv0 + 0.02), (w2 - 1.0, lv0 - 0.12), (w2, lv0 - 0.16))
    chords("ab_orgpedq", [(w0, ["Bb1"]), (w1, ["F2"]), (w2, None)], legato=False)
    P("ab_orgpedq").d((w0, 0.5), (w2 - 0.5, 0.36))
    # the web: cycles of different lengths and grids drifting against each other
    web_bb = {"p": ["Bb4", "C5", "F5", "A5", "C6", "F5", "D5"],        # piano, 16ths, 7-cycle
              "w": ["D5", "A5", "C6", "E6", "C6"],                     # warm FM, 8ths, 5-cycle
              "q": ["D5", "E5", "A5", "E5"],                           # organ flute, triplet 8ths, 4-cycle
              "w2": ["Bb3", "F4", "C5"]}                               # low warm FM, quarters, 3-cycle
    web_f = {"p": ["F4", "G4", "C5", "F5", "G5", "C5", "A4"],
             "w": ["C5", "E5", "G5", "A5", "G5"],
             "q": ["A4", "C5", "G5", "C5"],
             "w2": ["F3", "C4", "G4"]}

    def web(voice, pn, step, b0, b1, v0, v1, pan_fn, fade_out=None, **kw):
        def run(pitches, t0, t1):
            return cycle(pn, pitches, t0, t1, step,
                         lambda i, t: lin(t, b0, fade_out or b1, v0, v1) * (1.0 if i % len(pitches) == 0 else 0.86),
                         pans=pan_fn, **kw)
        run(web_bb[voice], b0, w1)
        run(web_f[voice], max(b0, w1), b1)
    web("p", "ab_piano", 0.25, w0, w2 - 0.25, 0.38, 0.24, lambda i, t: 0.35 * np.sin(i * 0.7))
    web("w", "ab_warm", 0.5, w0 + 0.75, w2 - 1.0, 0.36, 0.24, lambda i, t: 0.55 * np.sin(i * 1.3 + 1))
    web("q", "ab_orgq", 2 / 3, w0 + 1.5, w2 - 1.5, 0.3, 0.22, None, rel=0.25)
    web("w2", "ab_warm2", 1.0, w0 + 2.5, w2 - 2.0, 0.34, 0.24, lambda i, t: -0.5 + 0.3 * np.sin(i))
    P("ab_orgq").d((w0 + 1.5, 0.36), (w2 - 1.5, 0.26))
    # the thinking fire's own arpeggio, heard first exactly as it was in the KINDLING (glass:
    # FM ratio 3.5, index 1.6) and turning warm note by note as the web spreads (ratio 1, index 0.6)
    cyc = ["Bb5", "C6", "F6", "A6", "C7", "F6", "D6"]
    t, i = w0, 0
    while t < w0 + 6 - 1e-6:
        u = min(1.0, (t - w0) / 4.5) ** 0.8          # 0 = glass .. 1 = warm
        N("ab_morph", cyc[i % 7], t, 0.25, 0.42 - 0.12 * u, sync=(i == 0),
          ratio=3.5 + (1.0 - 3.5) * u, index=1.6 + (0.6 - 1.6) * u, lp=12000 + (4200 - 12000) * u,
          glass=1.0 - u, decay=1.1 + 0.9 * u, pan=0.45 * np.sin(i * 0.9))
        t += 0.25
        i += 1
    # THE ANSWER (the theme's second phrase, over its own Bb) and the third phrase (over F),
    # inside the texture: violas + one horn, mezzo-piano
    P("ab_hn1").line("D4:1 C4:.5 Bb3:.5 F3:2", w0)
    P("ab_hn1").line("F3:.5 G3:.5 A3:1 C4:1 A3:1.2", w1)
    P("ab_hn1").d((w0, 0.42), (w0 + 1, 0.46), (w1, 0.42), (w1 + 2.5, 0.48), (w2, 0.28))
    P("ab_svln").line("D5:1 C5:.5 Bb4:.5 F4:2", w0)
    P("ab_svln").line("F4:.5 G4:.5 A4:1 C5:1 A4:1.2", w1)
    P("ab_svln").d((w0, 0.38), (w0 + 1, 0.44), (w1 + 2.5, 0.46), (w2, 0.26))


# ---- THE ACCORD ---------------------------------------------------------------
LOCK = [0, 2, 7, 12, 14, 7, 3, 2]          # the thinking cycle, locked to 8 (fits the bar)


def lock_pattern(root, third=3):
    r = m(root)
    iv = [0, 2, 7, 12, 14, 7, third, 2]
    return [r + x for x in iv]


def ab_accord():
    a0, lock, s0, s1, bre, dawn = fb(2080), fb(2160), fb(2320), fb(2360), fb(2400) - 0.33, fb(2400)
    # --- 2080 hush: D minor, low strings pp, quiet pedal; the thinking cycle alone on the piano
    N("ab_cb", "D2", a0, s0 - a0 + 0.06, legato=True)
    chords("ab_vc", [(a0, ["D3", "A3"]), (lock, ["D3", "A3"]), (fb(2230), ["D3", "Bb3"]), (fb(2290), ["D3", "G3"]),
                     (s0, None)], overlap=0.06)
    chords("ab_vla", [(a0, ["F4"]), (lock, ["F4"]), (fb(2290), ["G4"]), (s0, None)], overlap=0.06)
    N("ab_orgpedq", "D2", a0, s0 - a0 + 0.03)
    P("ab_orgpedq").d((a0, 0.34), (lock, 0.4), (s0 - 0.1, 0.62))
    cycle("ab_piano", ["D5", "E5", "A5", "D6", "E6", "A5", "F5"], a0, lock, 0.25,
          lambda i, t: 0.2 + 0.04 * (i % 7 == 0))
    # --- 2160 THE LOCK: every voice plays one 8-note cycle that fits the bar, in unison/octaves;
    # the harmony moves only on off-beats (2230, 2290) over the D pedal - never on an oath
    segs = [(lock, fb(2230), "D5", 3), (fb(2230), fb(2290), "Bb4", 4), (fb(2290), s0, "C5", 4),
            (s0, s1, "Bb4", 4), (s1, bre, "C5", 4)]
    t = lock
    i = 0
    while t < bre - 1e-6:
        root, third = None, 3
        for a, b, r, th in segs:
            if a - 1e-6 <= t < b - 1e-6:
                root, third = r, th
        pat = lock_pattern(root, third)
        p = pat[i % 8]
        prog = (t - lock) / (bre - lock)
        step = 0.25
        v = 0.3 + 0.4 * prog
        acc = 0.08 if i % 8 == 0 else 0
        N("ab_pnoarp", p, t, step, min(1.0, v * 0.8 + acc), sync=(i == 0))
        N("ab_warm", p, t, step, min(1.0, 0.16 + 0.2 * prog), pan=0.35 * np.sin(i * 0.5), damp=True)
        N("ab_vla_sp", p - 12, t, step, min(1.0, 0.4 + 0.5 * prog + acc))
        if t >= fb(2190):
            N("ab_vln2_sp", p, t, step, min(1.0, 0.34 + 0.55 * prog + acc))
        if t >= s0:
            N("ab_vln1_sp", p if p + 12 > m("D6") else p + 12, t, step, min(1.0, 0.5 + 0.45 * prog + acc))
        N("ab_org", p - 12, t, step, None, rel=0.1)
        t += step
        i += 1
    P("ab_org").d((lock, 0.2), (s0, 0.4), (bre - 0.05, 0.64))
    # sustained harmony over the D pedal: Dm(add9) - Bb/D (2230) - C/D (2290) ; 2320 Bb ; 2360 C
    chords("ab_vln2", [(lock, ["A4", "D5"]), (fb(2230), ["Bb4", "D5"]), (fb(2290), ["C5", "E5"]),
                       (s0, ["Bb4", "F5"]), (s1, ["C5", "G5"]), (bre, None)], overlap=0.0)
    chords("ab_vla", [(s0, ["D4", "F4"]), (s1, ["E4", "G4"]), (bre, None)], overlap=0.0)
    chords("ab_vc", [(s0, ["Bb2", "F3"]), (s1, ["C3", "G3"]), (bre, None)], overlap=0.0)
    N("ab_cb", "Bb1", s0, s1 - s0 + 0.04, legato=True)
    N("ab_cb", "C2", s1, bre - s1, legato=True)
    chords("ab_orgped", [(s0, ["Bb2"]), (s1, ["C3"]), (bre, None)], overlap=0.0, legato=False)
    chords("ab_tbn", [(s0, ["Bb2", "D3", "F3"]), (s1, ["C3", "E3", "G3"]), (bre, None)], overlap=0.0)
    chords("ab_tuba", [(s0, ["Bb1"]), (s1, ["C2"]), (bre, None)], overlap=0.0)
    # the rising line (violins + horns): A4 . Bb4 . C5 | D5 E5 -> (the dawn)
    for pn in ("ab_vln1", "ab_hn1", "ab_hn2"):
        oc = 0 if pn == "ab_vln1" else -12
        seq = [(lock, "A4"), (fb(2230), "Bb4"), (fb(2290), "C5"), (s0, "D5"), (s1, "E5")]
        for j, (tt, p) in enumerate(seq):
            t1 = seq[j + 1][0] if j + 1 < len(seq) else bre
            N(pn, m(p) + oc, tt, t1 - tt + (0.05 if t1 < bre else 0.0), legato=j > 0)
    N("ab_vln1b", "A5", s1, bre - s1)
    # dynamics: one long crescendo, the ring sweep lifts it, the hearth flare peaks, then the breath
    for pn, lo, hi in (("ab_cb", 0.26, 0.68), ("ab_vc", 0.24, 0.66), ("ab_vla", 0.22, 0.64), ("ab_vln2", 0.2, 0.64),
                       ("ab_vln1", 0.32, 0.72), ("ab_vln1b", 0.54, 0.68), ("ab_hn1", 0.26, 0.68), ("ab_hn2", 0.24, 0.66),
                       ("ab_tbn", 0.42, 0.66), ("ab_tuba", 0.38, 0.64), ("ab_orgped", 0.4, 0.62)):
        P(pn).d((a0 + 0.2, lo * 0.8), (lock, lo), (fb(2240), lo + (hi - lo) * 0.25), (s0, lo + (hi - lo) * 0.55),
                (s1, lo + (hi - lo) * 0.8), (bre - 0.02, hi))
    # the ring sweep (2320-2380): violin tremolo rising through the chord, left to right
    # the ring sweep (2320-2380): a swell - the violins' tremolo on the chord, from nothing into the flare
    chords("ab_vln_tr", [(s0, ["D5", "F5", "Bb5"]), (s1, ["E5", "G5", "C6"]), (bre, None)], overlap=0.0)
    P("ab_vln_tr").d((s0, 0.08), (s1, 0.4), (fb(2380), 0.62), (bre - 0.02, 0.76))
    # timpani roll + bass drum roll: from nothing into the flare, cut by the breath
    N("ab_timp_roll", "C3", fb(2300), bre - fb(2300), rel=0.02)
    P("ab_timp_roll").d((fb(2300), 0.1), (s1, 0.4), (bre - 0.02, 0.82))
    N("ab_bdroll", 60, s1, bre - s1, rel=0.02)
    P("ab_bdroll").d((s1, 0.2), (bre - 0.02, 0.76))


# ---- DAWN ---------------------------------------------------------------------
# Awe from harmony, not melody.  Out of the silence the strings (two sections) bloom on D add9
# with the thinking fire's cycle glowing in it, and hold it for ~80 frames with no tune.  Over the
# D pedal the harmony then turns: E/D (the Lydian light) - Gmaj9/D - back to D add9.  The CALL
# enters late (2480) on one voice, the cellos in unison, mezzo-forte, and comes home on D as the
# chord does.  No stroke, no brass, no organ manual (the 16' pedal as the floor).
DAWN_H = [  # frame, vla, vln2, vln1, vln1b, piano cycle, warm-FM cycle
    (2400, ["E4", "F#4"], ["A4", "D5"], ["E5", "A5"], ["F#5"],
     ["D5", "E5", "A5", "D6", "E6", "A5", "F#5"], ["A5", "D6", "E6", "F#6", "E6"]),
    (2480, ["E4", "G#4"], ["B4", "E5"], ["G#5", "B5"], ["E5"],
     ["E5", "F#5", "B5", "E6", "F#6", "B5", "G#5"], ["B5", "E6", "F#6", "G#6", "F#6"]),
    (2520, ["B3", "F#4"], ["A4", "D5"], ["F#5", "B5"], ["D5"],
     ["D5", "A5", "B5", "D6", "A6", "B5", "G5"], ["B5", "D6", "G6", "A6", "G6"]),
    (2560, ["A3", "F#4"], ["A4", "E5"], ["F#5", "A5"], ["E5"],
     ["D5", "E5", "A5", "D6", "E6", "A5", "F#5"], ["A5", "D6", "E6", "F#6", "E6"]),
]


def ab_dawn():
    d0 = fb(2400)
    dz, end = fb(2624), fb(2680)
    # the floor: basses and the 16' pedal on D; the cellos hold D-A through the bloom, then leave
    # their register to the call
    N("ab_cb", "D2", d0, end - d0, sync=True)
    N("ab_orgped", "D2", d0, end - d0 - 0.5, sync=True)
    for p in ("D3", "A3"):
        N("ab_vc", p, d0, fb(2480) - d0 + 0.05, sync=True)
    P("ab_orgped").d((d0, 0.46), (d0 + 3, 0.5), (fb(2560), 0.42), (dz, 0.28), (end - 0.5, 0.08))
    for i, (f, vla, vln2, vln1, vln1b, pc, wc) in enumerate(DAWN_H):
        t = fb(f)
        t1 = fb(DAWN_H[i + 1][0]) if i + 1 < len(DAWN_H) else end
        lg = i > 0
        for pn, ps in (("ab_vla", vla), ("ab_vln2", vln2), ("ab_vln1", vln1), ("ab_vln1b", vln1b)):
            for p in ps:
                N(pn, p, t, t1 - t + (0.06 if i + 1 < len(DAWN_H) else 0.0), legato=lg, sync=(i == 0))
    # the bloom: from mezzo-piano out of the silence to a full, warm forte in ~1.3 beats, then held
    for pn, top in (("ab_cb", 0.6), ("ab_vc", 0.64), ("ab_vla", 0.66), ("ab_vln2", 0.66), ("ab_vln1", 0.68),
                    ("ab_vln1b", 0.62)):
        P(pn).d((d0, top - 0.24), (d0 + 1.3, top), (fb(2465), top - 0.03), (fb(2482), top - 0.12),
                (fb(2520), top - 0.13), (fb(2560), top - 0.15), (dz, top - 0.28), (end, 0.08))
    # the thinking fire's cycle, glowing in the chord (piano 16ths, warm FM 8ths), from the first frame
    cyc_i, fm_i = 0, 0
    for i, (f, vla, vln2, vln1, vln1b, pc, wc) in enumerate(DAWN_H):
        t = fb(f)
        t1 = fb(DAWN_H[i + 1][0]) if i + 1 < len(DAWN_H) else dz
        tt = t
        while tt < t1 - 1e-6:
            fade = 1.0 if tt < fb(2560) else max(0.3, 1 - (tt - fb(2560)) / (dz - fb(2560)))
            N("ab_piano", pc[cyc_i % 7], tt, 0.25, 0.22 * fade * (1.0 if cyc_i % 7 == 0 else 0.86),
              pan=0.4 * np.sin(cyc_i * 0.9), sync=(cyc_i == 0))
            tt += 0.25
            cyc_i += 1
        tt = t
        while tt < t1 - 1e-6:
            fade = 1.0 if tt < fb(2560) else max(0.3, 1 - (tt - fb(2560)) / (dz - fb(2560)))
            N("ab_warm", wc[fm_i % 5], tt, 0.5, 0.28 * fade, pan=0.5 * np.sin(fm_i * 1.3), sync=(fm_i == 0))
            tt += 0.5
            fm_i += 1
    # THE CALL, late, on one voice: the cellos in unison, mezzo-forte (half notes, home on D)
    P("ab_vcm").line("D3:2 A3:2 D4:4.2", fb(2480))
    P("ab_vcm").d((fb(2480), 0.64), (fb(2490), 0.72), (fb(2520), 0.74), (fb(2560), 0.74), (dz, 0.52),
                  (fb(2650), 0.22))


# ---- CODA ---------------------------------------------------------------------
def ab_coda():
    # the piano asks the call again (the SILENCE's D4 .. A4 .. D5); the D5 hangs through the
    # question; the answer during the hand-over; the low fifth at 2780 (her torch in its hands)
    pno = P("ab_piano")
    pedal = fb(2798)
    pno.n("D4", fb(2652), fb(2780) - fb(2652), 0.24, sync=True)
    pno.n("D3", fb(2652) + 0.02, fb(2780) - fb(2652), 0.12)
    pno.n("A4", fb(2688), fb(2748) - fb(2688), 0.21, sync=True)
    pno.n("D5", fb(2726), fb(2748) - fb(2726) + 0.05, 0.23, sync=True)
    pno.n("C#5", fb(2748), fb(2762) - fb(2748) + 0.05, 0.19, sync=True)
    pno.n("B4", fb(2762), fb(2780) - fb(2762) + 0.05, 0.18, sync=True)
    pno.n("F#4", fb(2780), fb(2890) - fb(2780), 0.2, sync=True)
    pno.n("D2", fb(2780) + 0.03, fb(2900) - fb(2780), 0.15)
    pno.n("A2", fb(2780) + 0.06, fb(2900) - fb(2780), 0.13)
    # strings breathing: a D pedal and a high A, pianissimo, swelling and relaxing
    N("ab_cb", "D2", fb(2656), fb(2800) - fb(2656) + 0.06)
    N("ab_vc", "A2", fb(2670), fb(2800) - fb(2670) + 0.06)
    N("ab_vln1b", "A5", fb(2700), fb(2800) - fb(2700) + 0.06)
    for p, a in (("ab_cb", 0), ("ab_vc", 0.01), ("ab_vln1b", 0.03)):
        P(p).d((fb(2656), 0.1), (fb(2690), 0.3 - a), (fb(2730), 0.2), (fb(2760), 0.32 - a), (fb(2795), 0.34))
    # 2800 the child's torch lights the beacon: a warm swell on G/D (plagal warmth)
    t0 = fb(2800)
    tf = fb(2880)
    endc = fb(2946)
    chords("ab_cb", [(t0, ["D2"]), (endc, None)])
    chords("ab_vc", [(t0, ["D3", "G3"]), (tf, ["D3", "A3"]), (endc, None)])
    chords("ab_vla", [(t0, ["B3", "D4"]), (tf, ["F#4"]), (endc, None)])
    chords("ab_vln2", [(t0, ["B4", "D5"]), (tf, ["A4"]), (endc, None)])
    for p, pk in (("ab_cb", 0.56), ("ab_vc", 0.54), ("ab_vla", 0.52), ("ab_vln2", 0.5)):
        P(p).d((t0, 0.12), (t0 + 1.2, pk), (tf - 0.5, pk - 0.1), (tf + 0.5, pk - 0.06), (endc - 1.0, 0.08),
               (endc, 0.04))
    # the answering fires: soft string swells spreading outward (one family), building D add9 - not bells
    fires = [(2806, "vc", "A3", -0.3, 0.42), (2817, "vla", "F#4", 0.35, 0.42), (2829, "vla", "A4", -0.5, 0.42),
             (2841, "vln2", "E5", 0.55, 0.44), (2853, "vln2", "F#5", -0.7, 0.44), (2865, "vln1", "A5", 0.75, 0.46),
             (2873, "vln1", "D6", -0.8, 0.42)]
    for j, (f, src, p, pan, pk) in enumerate(fires):
        pn = f"ab_fire{j + 1}"
        clone(pn, src, pan=pan, width=0.4)
        tt = fb(f)
        N(pn, p, tt, endc - tt - 0.3)
        # its own swell: from nothing to a warm glow in ~0.6 s, then settling into the chord
        P(pn).d((tt, 0.05), (tt + 0.75, pk), (tf + 1.0, pk * 0.85), (endc - 1.2, 0.06))
    # 2880 the final chord: D add9 - with the piano's low fifth (the SILENCE's "together")
    N("ab_orgpedq", "D2", tf, endc - tf - 0.5)
    P("ab_orgpedq").d((tf, 0.4), (endc - 1.5, 0.12))
    pno.n("E5", tf, fb(2950) - tf, 0.2)
    pno.n("A3", tf + 0.03, fb(2950) - tf, 0.17)


def breathe(pn, t0, t1, depth=0.12, min_dur=1.2):
    """Phrasing: every long note of part `pn` in [t0, t1) gets a messa di voce (it starts a little
    under the line's level, swells to it at ~40 % of the note and relaxes slightly before the next).
    The part's existing curve is kept and modulated, not replaced."""
    from dsl import dyn_at
    o = P(pn)
    notes = sorted((n for n in o.notes if t0 - 1e-6 <= n.start < t1 and n.dur >= min_dur), key=lambda n: n.start)
    if not notes:
        return
    base = list(o.dyn)
    keep = [(b, v) for b, v in base if b < t0 - 1e-6 or b > t1 + 1e-6]
    pts = []
    t = t0
    while t <= t1 + 1e-6:
        g = 1.0
        for n in notes:
            if n.start - 1e-6 <= t < n.start + n.dur:
                u = (t - n.start) / n.dur
                g = (1 - depth) + depth * (np.sin(np.pi * min(1.0, u / 0.8)) if u < 0.8 else 0.35 * (1 - u) / 0.2)
                g = max(1 - depth, g)
        pts.append((t, min(1.0, dyn_at(base, t) * g)))
        t += 0.25
    o.dyn = sorted(keep + pts, key=lambda x: x[0])


def layer(src, new, t0, t1, gain_db=-1.5, pan_shift=0.12, seed_off=7):
    """a second desk/section on the same notes (own round robins, detune and humanising):
    a bigger ensemble, not just more gain"""
    o = PARTS[src]
    q = clone(new, src, pan=max(-1.0, min(1.0, o.pan + pan_shift)), gain_db=o.gain_db + gain_db)
    q.seed = o.seed + seed_off
    for nt in o.notes:
        if t0 - 1e-6 <= nt.start < t1:
            q.n(nt.pitch, nt.start, nt.dur, nt.vel, legato=nt.legato, sync=nt.sync, **dict(nt.kw))
    q.dyn = [(b, v) for b, v in o.dyn if t0 - 4 <= b <= t1 + 4]
    return q


def build_ab():
    setup_ab()
    ab_first_beacon()
    ab_run()
    ab_montage()
    ab_world()
    ab_accord()
    ab_dawn()
    # phrasing on the long melodic notes (the calls, the answers)
    breathe("ab_vla", fb(1360), fb(1480), depth=0.12)
    breathe("ab_hn1", fb(1700), fb(1918), depth=0.12)
    breathe("ab_vcm", fb(2480), fb(2660), depth=0.12)
    breathe("ab_svln", fb(1920), fb(2080), depth=0.14, min_dur=0.9)
    # the dawn's second string section (the call stays on one section of cellos)
    d0, dz = fb(2400), fb(2680)
    for src in ("ab_vln1", "ab_vln1b", "ab_vln2", "ab_vla", "ab_vc", "ab_cb"):
        layer(src, src + "_L2", d0, dz, gain_db=-2.0, pan_shift=0.1 if "vln" in src else -0.1)
    ab_coda()


# ---------------------------------------------------------------------------
def race_entrance_reinforcement():
    """The finished v1 film (edit/mix.py) reinforced the RACE downbeat (640) with the score's own
    instruments and a short lift; that is what the producers approved, so v2 carries it in the
    score itself: the same layers, seated like the v1 parts (v1 score.py untouched)."""
    r0 = fb(640)
    lay = [("v2r_giant", "giant", 60, 1.0, 0.0), ("v2r_bdrum", "bdrum", 60, 1.0, 0.0),
           ("v2r_timp", "timp", "D2", 1.0, -3.0), ("v2r_tenor", "tenor", 60, 1.0, 5.0),
           ("v2r_snare", "snare", 60, 1.0, 6.0), ("v2r_cym", "cym", 60, 1.0, 8.0),
           ("v2r_crash", "crash", 60, 1.0, 2.0)]
    for name, src, pitch, vel, gdb in lay:
        clone(name, src, gain_db=V1.FADERS[src] + gdb)
        N(name, pitch, r0, 2.0, vel, sync=True)
    clone("v2r_taiko", "taiko", gain_db=V1.FADERS["taiko"] - 5.0)
    N("v2r_taiko", 60, r0, 1.0, 1.0, drum="o", sync=True, tune=36.708 / 48)
    N("v2r_taiko", 60, r0, 1.0, 1.0, drum="n", sync=True, tune=73.416 / 92)
    clone("v2r_impact", "impact", gain_db=V1.FADERS["impact"] - 3.0)
    N("v2r_impact", 60, r0, 2.0, 1.0, sync=True, size=0.6, crack=5.0)
    PUSH.append((640, 1.5, 0.0, 0.4, 0.8))


# score-mix lifts right after a downbeat (edit/mix.py's PUSH): (frame, dB above 150 Hz, dB below,
# hold s, release s) - applied by render_v2 to the score mix
PUSH = []


AIR = {"vln": (3.0, 6000.0), "vln_spic": (2.5, 6000.0), "vla": (2.0, 6000.0), "vln_trem": (3.0, 6000.0),
       "svln": (2.0, 6500.0), "organ": (2.0, 5000.0), "organ_q": (2.0, 5000.0), "vla_spic": (1.5, 6000.0),
       "harp": (2.0, 6000.0), "flute": (1.5, 7000.0), "horn": (1.0, 5000.0)}


def add_air(prefix):
    """a gentle air shelf on the second-half strings/organ/harp (they read dark next to the first
    half's glass, cymbals and ticks - measured, see NOTES_v2.md)"""
    for name, p in PARTS.items():
        if name.startswith(prefix) and p.inst in AIR:
            g, f = AIR[p.inst]
            hp, lp = MIX_EQ.get(name, (None, None))[:2]
            MIX_EQ[name] = (hp, lp, g, f)


def build(cut="AB"):
    cut = cut.upper()
    PARTS.clear()
    MIX_EQ.clear()
    V1.setup()
    for k, v in V1.FADERS.items():
        if k in PARTS:
            PARTS[k].gain_db = v
    # the v1 first half, unchanged
    V1.intro()
    V1.kindling()
    V1.race()
    V1.silence()
    V1.glock_fix()
    PUSH.clear()
    race_entrance_reinforcement()
    if cut == "AB":
        build_ab()
        add_air("ab_")
    else:
        import score_v2_c
        score_v2_c.build_c()
        add_air("c_")
    return {k: v for k, v in PARTS.items() if v.notes}


def sync_table(cut):
    """(frame, label, source, window_s, kind) measured by analyze_v2."""
    T = [(0, "wind + drone from silence", "final", 0.3, "start"),
         (320, "KINDLING shimmer", "glass", 0.06, "hit"),
         (480, "IGNITION", "final", 0.05, "hit"),
         (640, "RACE drums enter (after the breath)", "final", 0.05, "hit"),
         (800, "storm swell / crash", "final", 0.05, "hit"),
         (1040, "IMPACT (after the suck)", "final", 0.05, "hit"),
         (1236, "flint strike 1", "sfx", 0.05, "hit"),
         (1262, "flint strike 2", "sfx", 0.05, "hit"),
         (1290, "flint strike 3", "sfx", 0.05, "hit"),
         (1318, "kindling catches (sfx, a soft catch)", "sfx", 0.15, "soft")]
    if cut == "AB":
        T += [(1318, "kindling catches (strings bloom)", "ab_vla", 0.15, "soft"),
              (1360, "beacon ROARS (sfx roar)", "sfx", 0.05, "hit"),
              (1480, "the shepherd's beacon (sfx)", "sfx", 0.05, "hit"),
              (1520, "the near horn's call", "ab_hn1", 0.1, "soft")]
        T += [(f, f"run beacon {i + 1} (sfx whoomp)", "sfx", 0.04, "hit") for i, f in enumerate(RUN_BEACONS)]
        T += [(1560, "the far horn answers", "ab_hnfar", 0.1, "soft")]
        for f, nm in zip(MONTAGE_IGN, ["desert", "ice", "karst", "city", "sea"]):
            T.append((f, f"{nm} ignition (sfx)", "sfx", 0.05, "hit"))
        T += [(1920, "the world answers: the glass arpeggio", "ab_morph", 0.05, "hit"),
              (1920, "the web begins (piano)", "ab_piano", 0.05, "hit"),
              (2160, "flames merge: the lock (piano)", "ab_pnoarp", 0.06, "hit"),
              (2400, "CLIMAX: the bloom out of the silence (final)", "final", 0.05, "hit"),
              (2400, "the thinking cycle glowing (piano)", "ab_piano", 0.05, "hit"),
              (2480, "the call, late, on the cellos", "ab_vcm", 0.1, "soft"),
              (2652, "coda: piano D4", "ab_piano", 0.06, "hit"),
              (2726, "coda: piano D5 (the question hangs)", "ab_piano", 0.06, "hit"),
              (2756, "flint echo 1 (the hand-over)", "sfx", 0.05, "hit"),
              (2768, "flint echo 2", "sfx", 0.05, "hit"),
              (2780, "flint echo 3 + the torch in the child's hands (piano F#4)", "ab_piano", 0.06, "hit"),
              (2800, "beacon catch (sfx)", "sfx", 0.06, "hit"),
              (2880, "final chord (piano)", "ab_piano", 0.06, "hit")]
    else:
        import score_v2_c
        T += score_v2_c.SYNC_C
    return T
