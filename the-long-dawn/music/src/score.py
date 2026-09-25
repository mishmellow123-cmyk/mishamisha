"""THE LONG DAWN - the score, as code.

Global beats: gb(bar, beat); frames: fb(frame) (20 frames = 1 beat).
Every sync-critical hit uses sync=True (never humanised).

Harmonic plan
  INTRO     D pedal: Dm - Bb/D - F/D - C/D, solo flute theme A1-A4
  KINDLING  C/D - Dm(add9) - Bb(add9) -> IGNITION Bbmaj9(#11)/D -> A7(b9)
  RACE      D pedal Dm | Eb/D | Dm | Eb/D, corrupted call D-Ab-D
  GRASP     cluster crescendo on D, glissandi, Shepard -> IMPACT (D + Ab)
  SILENCE   piano D4 .. A4 .. D5, then D2+A2 "together"
  BEACON    D pedal -> Bb/D (kindling) -> Bb/C -> F (the roar), horn call
  BEACONS   ignition-locked harmonic rhythm, bass rising F-A-Bb-C-D-E-F,
            the call in a new voice per beacon (stretto)
  GLOBE     F (choir enters, organ pedal) - Bb/F - C/F
  ACCORD    Dm (hush) ; oaths: Bb/D - C - F/A - G/A ; A (ring sweep) ;
            top line climbs F5 G5 A5 B5 C#6 -> D6
  DAWN      D - Bm - D/F# - G - Gm/Bb  (theme: call, answer, third phrase)
  CODA      flute call (hangs on D5 at the question), answer F#4, swell,
            bells for the answering fires, D add9.
"""
import numpy as np

from dsl import Part, gb, fb, m

PARTS = {}


def part(name, inst, **kw):
    if name not in PARTS:
        PARTS[name] = Part(name, inst, **kw)
    return PARTS[name]


# ---------------------------------------------------------------------------
# instruments / seating
#   pan: -1 L .. +1 R ; depth 0 (close) .. 1 (far) ; send = reverb amount
# ---------------------------------------------------------------------------
def setup():
    S = dict(bus="strings")
    part("vln1", "vln", pan=-0.62, width=0.55, depth=0.25, send=0.26, **S)
    part("vln2", "vln", pan=-0.3, width=0.5, depth=0.3, send=0.27, **S)
    part("vla", "vla", pan=0.05, width=0.5, depth=0.3, send=0.27, **S)
    part("vc", "vc", pan=0.35, width=0.5, depth=0.3, send=0.26, **S)
    part("cb", "cb", pan=0.58, width=0.4, depth=0.35, send=0.25, gain_db=1.5, **S)
    part("svln", "svln", pan=-0.55, width=0.4, depth=0.3, send=0.3, gain_db=-4, **S)
    part("vln1_sp", "vln_spic", pan=-0.62, width=0.55, depth=0.25, send=0.2, humanize_ms=6, **S)
    part("vln2_sp", "vln_spic", pan=-0.3, width=0.5, depth=0.3, send=0.2, humanize_ms=6, **S)
    part("vla_sp", "vla_spic", pan=0.05, width=0.5, depth=0.3, send=0.2, humanize_ms=6, **S)
    part("vc_sp", "vc_spic", pan=0.35, width=0.5, depth=0.3, send=0.2, humanize_ms=6, **S)
    part("cb_sp", "cb_spic", pan=0.58, width=0.4, depth=0.35, send=0.2, humanize_ms=6, gain_db=2, **S)
    part("vln_tr", "vln_trem", pan=-0.45, width=0.7, depth=0.3, send=0.3, **S)
    part("vla_tr", "vla_trem", pan=0.05, width=0.6, depth=0.3, send=0.3, **S)
    part("vc_tr", "vc_trem", pan=0.35, width=0.6, depth=0.3, send=0.3, **S)
    part("cb_pz", "cb_pizz", pan=0.58, width=0.4, depth=0.35, send=0.3, **S)
    part("vc_pz", "vc_pizz", pan=0.35, width=0.5, depth=0.3, send=0.3, **S)
    part("harp", "harp", bus="keys", pan=-0.45, width=0.6, depth=0.45, send=0.4)
    part("gliss", "gliss", kind="synth", bus="strings", pan=0.0, width=1.0, depth=0.35, send=0.35)
    # winds
    part("fl", "flute", bus="winds", pan=-0.08, width=0.35, depth=0.3, send=0.42, humanize_ms=10, gain_db=1.0)
    part("fl2", "flute", bus="winds", pan=-0.15, width=0.4, depth=0.45, send=0.4)
    part("cl", "clarinet", bus="winds", pan=0.12, width=0.35, depth=0.45, send=0.4, humanize_ms=10)
    part("ob", "oboe", bus="winds", pan=0.08, width=0.35, depth=0.45, send=0.4)
    # brass (rear)
    B = dict(bus="brass")
    part("hn_solo", "horn", pan=-0.3, width=0.4, depth=0.55, send=0.45, humanize_ms=10, **B)
    part("hns", "horn", pan=-0.35, width=0.7, depth=0.6, send=0.45, **B)
    part("hns2", "horn", pan=-0.2, width=0.7, depth=0.6, send=0.45, **B)
    part("tpt", "trumpet", pan=0.18, width=0.5, depth=0.6, send=0.4, **B)
    part("tbn", "trombone", pan=0.42, width=0.6, depth=0.65, send=0.42, **B)
    part("tuba", "tuba", pan=0.55, width=0.4, depth=0.65, send=0.4, **B)
    part("hn_st", "horn_stac", pan=-0.35, width=0.7, depth=0.6, send=0.4, humanize_ms=5, **B)
    part("tbn_st", "trombone_stac", pan=0.42, width=0.6, depth=0.65, send=0.4, humanize_ms=5, **B)
    part("tpt_st", "trumpet_stac", pan=0.18, width=0.5, depth=0.6, send=0.38, humanize_ms=5, **B)
    part("tuba_st", "tuba_stac", pan=0.55, width=0.4, depth=0.65, send=0.38, humanize_ms=5, **B)
    # percussion (rear)
    PC = dict(bus="perc", humanize_ms=0)
    part("timp", "timp", pan=-0.05, width=0.6, depth=0.7, send=0.42, **PC)
    part("timp_roll", "timp_roll", pan=-0.05, width=0.6, depth=0.7, send=0.42, **PC)
    part("bdrum", "bdrum", pan=0.1, width=0.8, depth=0.7, send=0.4, **PC)
    part("giant", "giant_mallet", pan=-0.1, width=0.9, depth=0.6, send=0.35, **PC)
    part("tenor", "tenor_lo", pan=0.2, width=0.8, depth=0.6, send=0.3, **PC)
    part("tenor_hi", "tenor_hi", pan=-0.25, width=0.8, depth=0.6, send=0.3, **PC)
    part("snare", "snare", pan=0.15, width=0.6, depth=0.65, send=0.35, **PC)
    part("snare_roll", "snare_roll", pan=0.15, width=0.6, depth=0.65, send=0.35, **PC)
    part("cym", "clash", pan=0.25, width=1.0, depth=0.6, send=0.45, **PC)
    part("crash", "crash", pan=-0.25, width=1.0, depth=0.6, send=0.45, **PC)
    part("swell", "cym_swell_med", pan=0.0, width=1.0, depth=0.6, send=0.45, **PC)
    part("swell_s", "cym_swell_short", pan=0.1, width=1.0, depth=0.6, send=0.45, **PC)
    part("gong", "gong", pan=0.0, width=1.0, depth=0.75, send=0.5, **PC)
    part("glock", "glock", pan=0.35, width=0.6, depth=0.5, send=0.4, **PC)
    part("tubular", "tubular", pan=0.55, width=0.6, depth=0.85, send=0.6, **PC)
    part("triangle", "triangle", pan=0.4, width=0.6, depth=0.6, send=0.45, **PC)
    # keys
    part("piano", "piano", bus="keys", pan=0.0, width=0.8, depth=0.35, send=0.5, humanize_ms=0)
    part("organ", "organ", kind="synth", bus="keys", pan=0.0, width=1.0, depth=0.8, send=0.5,
         params=dict(reg="full"))
    part("organ_ped", "organ", kind="synth", bus="keys", pan=0.0, width=1.0, depth=0.8, send=0.45,
         params=dict(reg="pedal"))
    # choir (synth) - behind the orchestra, wide
    part("choir", "choir", kind="synth", bus="choir", pan=0.0, width=1.0, depth=0.7, send=0.5,
         params=dict(vowel="a", voices=8, spread=0.85, breath=0.05))
    part("choir_oo", "choir", kind="synth", bus="choir", pan=0.0, width=1.0, depth=0.75, send=0.55,
         params=dict(vowel="u", voices=7, spread=0.8, breath=0.07, atk=0.6))
    # synth / fx (score side)
    X = dict(kind="synth", bus="synth")
    part("glass", "fmbell", pan=0.0, width=1.0, depth=0.45, send=0.45, humanize_ms=0,
         params=dict(ratio=3.5, index=1.6, decay=1.1), **X)
    part("celesta", "fmbell", pan=0.3, width=1.0, depth=0.5, send=0.5, humanize_ms=0,
         params=dict(ratio=1.0, index=1.2, decay=1.4, bright=0.7), **X)
    part("taiko", "taiko", pan=0.0, width=1.0, depth=0.55, send=0.3, humanize_ms=0, **X)
    part("tick", "tick", pan=0.0, width=1.0, depth=0.2, send=0.08, humanize_ms=0, **X)
    part("riser", "riser", pan=0.0, width=1.0, depth=0.4, send=0.35, **X)
    part("shepard", "shepard", pan=0.0, width=1.0, depth=0.4, send=0.35, **X)
    part("revcym", "revcym", pan=0.0, width=1.0, depth=0.4, send=0.3, **X)
    part("impact", "impact", pan=0.0, width=1.0, depth=0.3, send=0.35, humanize_ms=0, **X)
    part("subdrop", "subdrop", pan=0.0, width=0.0, depth=0.0, send=0.0, humanize_ms=0, **X)
    part("braam", "braam", pan=0.0, width=1.0, depth=0.45, send=0.4, humanize_ms=0, **X)
    part("sub", "sub", pan=0.0, width=0.0, depth=0.0, send=0.0, **X)
    part("bell", "lowbell", pan=0.0, width=1.0, depth=0.75, send=0.55, humanize_ms=0, **X)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def P(n):
    return PARTS[n]


def hold(pn, pitches, b0, b1, vel=None, **kw):
    for p in pitches:
        P(pn).n(p, b0, b1 - b0, vel, **kw)


def call(pn, root, b0, rhythm=(1, 1, 2), vel=None, last=None, **kw):
    """The call: root - fifth - octave."""
    r = m(root)
    notes = [r, r + 7, r + 12]
    t = b0
    for i, (p, d) in enumerate(zip(notes, rhythm)):
        dd = d if (i < 2 or last is None) else last
        P(pn).n(p, t, dd, vel, legato=(i > 0), **kw)
        t += d
    return t


def rr16(pn, pitches, b0, b1, vels, step=0.25, **kw):
    """Repeating pattern (cycling pitch list) on a fixed grid."""
    t = b0
    i = 0
    while t < b1 - 1e-6:
        p = pitches[i % len(pitches)]
        v = vels[i % len(vels)] if isinstance(vels, (list, tuple)) else vels
        if p is not None:
            P(pn).n(p, t, step, v, **kw)
        t += step
        i += 1


# ---------------------------------------------------------------------------
# 1. INTRO  bars 1-4 (+ flute ends in bar 5)
# ---------------------------------------------------------------------------
def intro():
    # low warm D drone (re-bowed with overlaps), from silence
    cb, vc = P("cb"), P("vc")
    for b0, b1 in [(gb(1, 1), gb(3, 1.3)), (gb(3, 1), gb(5, 1.3)), (gb(5, 1), gb(7, 1.1))]:
        cb.n("D2", b0, b1 - b0)
        vc.n("D2", b0 + 0.1, b1 - b0 - 0.1)
        vc.n("A2", b0 + 0.25, min(b1, gb(6, 3.05)) - b0 - 0.25)
    cb.d((0, 0.05), (gb(1, 3), 0.22), (gb(2, 1), 0.28), (gb(5, 1), 0.3), (gb(6, 1), 0.34),
         (gb(6, 4.5), 0.62), (gb(7, 1), 0.66))
    vc.d((0, 0.04), (gb(1, 3), 0.2), (gb(2, 1), 0.25), (gb(5, 1), 0.28), (gb(6, 1), 0.3),
         (gb(6, 4.5), 0.6), (gb(7, 1), 0.62))
    P("sub").n("D1", gb(1, 1), gb(7, 1) - gb(1, 1), atk=3.0, rel=2.0)
    P("sub").d((0, 0.05), (gb(2, 1), 0.2), (gb(6, 1), 0.22), (gb(7, 1), 0.3), (gb(8, 1), 0.1))

    # soft string pad: Dm - Bb/D - F/D(=Dm7) - C/D
    vla, vln2 = P("vla"), P("vln2")
    vla.n("A3", gb(1, 3), 6.1)
    vla.n("D4", gb(1, 3.2), 5.9)
    vln2.n("F4", gb(2, 1), 8.1)
    vla.n("Bb3", gb(3, 1), 4.1)
    vla.n("D4", gb(3, 1), 4.1, legato=True)
    vla.n("A3", gb(4, 1), 4.1, legato=True)
    vla.n("C4", gb(4, 1), 4.1, legato=True)
    vln2.n("F4", gb(4, 1), 4.1, legato=True)
    vla.n("G3", gb(5, 1), 4.1, legato=True)
    vla.n("C4", gb(5, 1), 4.1, legato=True)
    vln2.n("E4", gb(5, 1), 4.1, legato=True)
    vla.d((gb(1, 3), 0.1), (gb(2, 1), 0.2), (gb(3, 1), 0.24), (gb(4, 3), 0.28), (gb(5, 1), 0.25),
          (gb(6, 1), 0.3))
    vln2.d((gb(2, 1), 0.12), (gb(2, 3), 0.2), (gb(4, 1), 0.24), (gb(5, 1), 0.22), (gb(6, 1), 0.3))

    # harp and a distant bell, now and then
    h = P("harp")
    for i, p in enumerate(["D2", "A2", "D3", "A3"]):
        h.n(p, gb(1, 3) + i * 0.09, 4, 0.3)
    for i, p in enumerate(["Bb1", "F2", "D3"]):
        h.n(p, gb(3, 1) + i * 0.08, 4, 0.26)
    for i, p in enumerate(["F2", "C3", "A3"]):
        h.n(p, gb(4, 3) + i * 0.08, 4, 0.24)
    P("tubular").n("D5", gb(3, 3), 6, 0.2)

    # SOLO FLUTE: theme A1-A4 (bars 2-5)
    fl = P("fl")
    fl.line("D4:1 A4:1 D5:2", gb(2, 1))
    fl.line("D5:1 C5:.5 Bb4:.5 F4:2", gb(3, 1))
    fl.line("F4:.5 G4:.5 A4:1 C5:1 A4:1", gb(4, 1))
    fl.line("G4:1.5 F4:.5 E4:2.2", gb(5, 1))
    fl.d((gb(2, 1), 0.26), (gb(2, 3), 0.38), (gb(2, 4.5), 0.34), (gb(3, 1), 0.36), (gb(3, 3), 0.3),
         (gb(4, 1), 0.3), (gb(4, 3), 0.4), (gb(4, 4), 0.36), (gb(5, 1), 0.34), (gb(5, 3), 0.28),
         (gb(5, 4.8), 0.16))


# ---------------------------------------------------------------------------
# 2. KINDLING  bars 5-8 (frames 320-639); IGNITION at 480 = gb(7,1)
# ---------------------------------------------------------------------------
def kindling():
    g, gl = P("glass"), P("glock")
    # "thinking" arpeggio: 7-note cycles on a 16th grid -> shifting accents
    cyc5 = ["D5", "E5", "G5", "C6", "D6", "G5", "C6"]
    cyc6a = ["D5", "E5", "A5", "D6", "E6", "A5", "F5"]
    t = gb(5, 1)
    i = 0
    vel5 = np.linspace(0.18, 0.3, 16)
    while t < gb(6, 1) - 1e-6:
        g.n(cyc5[i % 7], t, 0.25, float(vel5[i]), pan=0.45 * np.sin(i * 0.9), sync=(i == 0))
        if i % 7 == 0:
            gl.n(cyc5[i % 7], t, 1, 0.22)
        t += 0.25
        i += 1
    i = 0
    while t < gb(6, 3) - 1e-6:
        g.n(cyc6a[i % 7], t, 0.25, 0.32 + 0.02 * i, pan=0.5 * np.sin(i * 0.9 + 1))
        if i % 7 == 0:
            gl.n(cyc6a[i % 7], t, 1, 0.3)
        t += 0.25
        i += 1
    # accelerate: sextuplets then 32nds, converging to one point (D6/E6)
    sx = ["D5", "F5", "Bb5", "C6", "D6", "F6"]
    for k in range(6):
        g.n(sx[k], gb(6, 3) + k / 6, 1 / 6, 0.48 + 0.02 * k, pan=0.55 * np.sin(k * 1.3))
    t32 = ["F5", "Bb5", "D6", "F6", "E6", "D6", "E6", "D6"]
    for k in range(8):
        g.n(t32[k], gb(6, 4) + k / 8, 1 / 8, 0.6 + 0.02 * k, pan=0.35 * np.sin(k * 2.1))
    gl.n("D6", gb(6, 4), 1, 0.4)

    # strings swell bar 6 -> ignition
    vln1, vln2, vla, vc = P("vln1"), P("vln2"), P("vla"), P("vc")
    vln1.n("A4", gb(6, 1), 2.1)
    vln1.n("E5", gb(6, 1), 2.1)
    vln1.n("F5", gb(6, 3), 2.0, legato=True)
    vln1.n("C6", gb(6, 3), 2.0, legato=True)
    vln2.n("F4", gb(6, 1), 2.1, legato=True)
    vln2.n("D5", gb(6, 3), 2.0, legato=True)
    vla.n("A3", gb(6, 1), 2.1, legato=True)
    vla.n("D4", gb(6, 1), 2.1, legato=True)
    vla.n("Bb3", gb(6, 3), 2.0, legato=True)
    vla.n("F4", gb(6, 3), 2.0, legato=True)
    vln1.d((gb(6, 1), 0.2), (gb(6, 3), 0.42), (gb(6, 4.7), 0.72), (gb(7, 1), 0.62))
    vln2.d((gb(6, 1), 0.3), (gb(6, 4.7), 0.7), (gb(7, 1), 0.6))
    vla.d((gb(6, 1), 0.3), (gb(6, 4.7), 0.7), (gb(7, 1), 0.6))

    # riser into the ignition: reverse cymbal + swell + soft noise riser
    P("revcym").n(60, gb(6, 3), 2, 0.55)
    P("swell").n(60, gb(7, 1), 1, 0.5, sync=True)
    P("riser").n(60, gb(6, 1), 4, 0.32, f0=300, f1=7000, curve=3.0)

    # ---- IGNITION (frame 480): deep, bright, awed hit ------------------------
    ig = gb(7, 1)
    P("timp").n("D2", ig, 2, 0.8, sync=True)
    P("bdrum").n(60, ig, 2, 0.78, sync=True)
    P("gong").n(60, ig, 4, 0.5, sync=True)
    P("impact").n(60, ig, 2, 0.55, sync=True, size=0.9, crack=0.3)
    P("crash").n(60, ig, 4, 0.45, sync=True)
    # bloom of glass: a wide Lydian spray
    for k, p in enumerate(["D5", "Bb5", "E6", "A6", "C7", "F6", "E7"]):
        g.n(p, ig + k * 0.06, 2, 0.55 - 0.04 * k, pan=np.sin(k * 1.9) * 0.7, sync=(k == 0))
    gl.n("E6", ig, 2, 0.5, sync=True)
    gl.n("A6", ig + 0.12, 2, 0.35)

    # chord opens: Bbmaj9(#11)/D  -> breathes (bars 7) -> A7(b9) (bar 8)
    cb = P("cb")
    cb.n("D2", gb(7, 1), 4.2)
    vc.n("Bb2", gb(7, 1), 4.2)
    vla.n("F3", gb(7, 1), 4.2)
    vla.n("C4", gb(7, 1), 4.2)
    vln2.n("D4", gb(7, 1), 4.2)
    vln2.n("E4", gb(7, 1), 4.2)
    vln1.n("A4", gb(7, 1), 4.2)
    vln1.n("E5", gb(7, 1), 4.2)
    hn = P("hns")
    hn.n("Bb3", gb(7, 1), 4.1)
    hn.n("D4", gb(7, 1), 4.1)
    hn.d((gb(7, 1), 0.34), (gb(7, 2.5), 0.48), (gb(7, 4), 0.32), (gb(8, 1), 0.4), (gb(8, 4.8), 0.75))
    # breathing: swell-relax
    for pn, base in (("vln1", 0.42), ("vln2", 0.4), ("vla", 0.4), ("vc", 0.42), ("cb", 0.44)):
        P(pn).d((gb(7, 1) + 0.02, base + 0.12), (gb(7, 1.6), base), (gb(7, 2.8), base + 0.14),
                (gb(7, 4.2), base - 0.04), (gb(8, 1), base + 0.02), (gb(8, 4.9), base + 0.3))

    # shimmer continues, breathing, around the chord (bar 7)
    cyc7 = ["D6", "A5", "E6", "F5", "C6", "Bb5", "E5"]
    t = gb(7, 1) + 0.5
    i = 0
    while t < gb(8, 1) - 1e-6:
        br = 0.28 + 0.1 * np.sin((t - gb(7, 1)) / 4 * 2 * np.pi)
        g.n(cyc7[i % 7], t, 0.25, float(br), pan=0.5 * np.sin(i * 0.8 + 2))
        t += 0.25
        i += 1
    # bar 8: the prize - uneasy dominant, towers rising
    cyc8 = ["E5", "Bb5", "C#6", "G5", "E6", "Bb5", "A5"]
    i = 0
    while t < gb(9, 1) - 1e-6:
        g.n(cyc8[i % 7], t, 0.25, 0.3 + 0.022 * i, pan=0.55 * np.sin(i * 0.8 + 3))
        if i % 7 == 0:
            gl.n(cyc8[i % 7], t, 1, 0.3 + 0.02 * i)
        t += 0.25
        i += 1
    cb.n("A1", gb(8, 1), 4.0, legato=True)
    vc.n("A2", gb(8, 1), 4.0, legato=True)
    vla.n("E3", gb(8, 1), 4.0, legato=True)
    vla.n("C#4", gb(8, 1), 4.0, legato=True)
    vln2.n("E4", gb(8, 1), 4.0, legato=True)
    vln2.n("G4", gb(8, 1), 4.0, legato=True)
    vln1.n("Bb4", gb(8, 1), 4.0, legato=True)
    vln1.n("E5", gb(8, 1), 4.0, legato=True)
    hn.n("A3", gb(8, 1), 4.0, legato=True)
    hn.n("C#4", gb(8, 1), 4.0, legato=True)
    # low pulse (towers rising): 8ths crescendo
    for k in range(8):
        t = gb(8, 1) + k * 0.5
        P("cb_sp").n("A1", t, 0.5, 0.35 + 0.06 * k)
        P("vc_sp").n("A2", t, 0.5, 0.32 + 0.06 * k)
    P("timp_roll").n("A2", gb(8, 3), 2.0)
    P("timp_roll").d((gb(8, 3), 0.15), (gb(8, 4.9), 0.75))
    P("revcym").n(60, gb(8, 2), 3, 0.6)
    P("swell_s").n(60, gb(9, 1), 1, 0.6, sync=True)


# ---------------------------------------------------------------------------
# 3. RACE  bars 9-12 (640-959); every beat = tower surge.  4. GRASP bar 13
# ---------------------------------------------------------------------------
def race():
    r0, r1 = gb(9, 1), gb(13, 1)
    taiko, giant, bd, ten, tenh = P("taiko"), P("giant"), P("bdrum"), P("tenor"), P("tenor_hi")
    acc16 = [1.0, 0.42, 0.62, 0.48]
    for beat in range(16):
        t = r0 + beat
        bar_pos = beat % 4
        storm = beat >= 8
        big = 0.95 if bar_pos == 0 else 0.82
        # the surge: odaiko + giant drum + bass drum, locked (sync)
        taiko.n(60, t, 1, big, drum="o", sync=True)
        giant.n(60, t, 1, 0.8 + 0.1 * (bar_pos == 0), sync=True)
        if bar_pos in (0, 2) or storm:
            bd.n(60, t, 1, 0.85 if bar_pos == 0 else 0.7, sync=True)
        # 16ths
        for k in range(4):
            tt = t + k * 0.25
            v = acc16[k] * (0.7 + 0.25 * storm)
            if k > 0:
                taiko.n(60, tt, 0.25, v * 0.8, drum="n")
                ten.n(60, tt, 0.25, v * 0.85)
            taiko.n(60, tt, 0.25, v * (0.55 + 0.2 * storm), drum="s", sync=(k == 0))
            if storm and k in (1, 3):
                tenh.n(60, tt, 0.25, v * 0.8)
        # bar-end fills
        if bar_pos == 3 and beat in (7, 15):
            for k in range(6):
                tenh.n(60, t + k / 6, 1 / 6, 0.5 + 0.08 * k)
                ten.n(60, t + k / 6, 1 / 6, 0.55 + 0.07 * k)
    # timpani on the D pedal
    for beat in range(16):
        t = r0 + beat
        if beat < 8 and beat % 2 == 1:
            continue
        P("timp").n("D2", t, 1, 0.8 if beat % 4 == 0 else 0.66, sync=True)
    # the clock: 16ths, dry, time pressure
    for k in range(64):
        P("tick").n(60, r0 + k * 0.25, 0.25, 0.22 + 0.25 * k / 64 + (0.08 if k % 4 == 0 else 0))
    # crashes: entrance and storm
    P("cym").n(60, r0, 4, 0.7, sync=True)
    P("crash").n(60, gb(11, 1), 4, 0.85, sync=True)
    P("swell").n(60, gb(11, 1), 1, 0.65, sync=True)
    P("gong").n(60, gb(11, 1), 4, 0.55, sync=True)

    # spiccato ostinato
    dm_v1 = ["D5", "F5", "E5", "D5", "A5", "F5", "E5", "D5"]
    eb_v1 = ["D5", "G5", "F5", "Eb5", "Bb5", "G5", "F5", "Eb5"]
    dm_v2 = ["F4", "A4", "D5", "A4"]
    eb_v2 = ["G4", "Bb4", "Eb5", "Bb4"]
    dm_va = ["D4", "D4", "A3", "D4", "D4", "A3", "D4", "A3"]      # 3+3+2
    eb_va = ["D4", "D4", "Bb3", "Eb4", "D4", "Bb3", "Eb4", "Bb3"]
    for bar in range(9, 13):
        dm = bar in (9, 11)
        b = gb(bar, 1)
        vv = [0.78, 0.5, 0.6, 0.52]
        if bar <= 10:
            rr16("vln1_sp", dm_v1 if dm else eb_v1, b, b + 4, [0.72, 0.46, 0.58, 0.5])
        rr16("vln2_sp", dm_v2 if dm else eb_v2, b, b + 4, [0.7, 0.45, 0.56, 0.48])
        acc = [0.8, 0.45, 0.5, 0.75, 0.45, 0.5, 0.75, 0.5]
        rr16("vla_sp", dm_va if dm else eb_va, b, b + 4, [a * (0.9 + 0.1 * (bar > 10)) for a in acc])
        rr16("vc_sp", ["D3", "D3", "D3", "D3"] if dm else ["D3", "D3", "Eb3", "D3"], b, b + 4, vv)
        rr16("cb_sp", ["D2", "D2"], b, b + 4, [0.8, 0.55], step=0.5)
    # storm: violins leave the ostinato for a rising chromatic line (tremolo)
    vt = P("vln_tr")
    chrom = ["D5", "Eb5", "E5", "F5", "F#5", "G5", "Ab5", "A5"]
    for k, p in enumerate(chrom):
        vt.n(p, gb(11, 1) + k, 1.05, legato=k > 0)
        vt.n(m(p) - 12, gb(11, 1) + k, 1.05, legato=k > 0)
    vt.d((gb(11, 1), 0.5), (gb(12, 4.9), 0.95))

    # corrupted call (D - Ab - D) in low brass: the call turned into an alarm
    def ccall(pn, root, b0, v):
        r = m(root)
        P(pn).n(r, b0, 0.95, v)
        P(pn).n(r + 6, b0 + 1, 0.95, v)
        P(pn).n(r + 12, b0 + 2, 1.9, v)
    for pn, root in (("hns", "D3"), ("tbn", "D3"), ("tuba", "D2")):
        ccall(pn, root, gb(9, 1), 0.82)
    # bar 10: Eb/D sforzando-crescendo
    hold("tbn", ["Eb3", "G3", "Bb3"], gb(10, 1), gb(11, 1) - 0.05)
    hold("hns", ["G3", "Bb3", "Eb4"], gb(10, 1), gb(11, 1) - 0.05)
    hold("tuba", ["D2"], gb(10, 1), gb(11, 1) - 0.05)
    for pn in ("tbn", "hns", "tuba"):
        P(pn).d((gb(9, 1), 0.82), (gb(10, 1), 0.82), (gb(10, 1.4), 0.45), (gb(10, 4.8), 0.85),
                (gb(11, 1), 0.9))
    # bar 11: higher, trumpets join
    for pn, root in (("tpt", "D4"), ("hns", "D3"), ("tbn", "D3"), ("tuba", "D2")):
        ccall(pn, root, gb(11, 1), 0.92)
    P("tpt").d((gb(11, 1), 0.92), (gb(12, 1), 0.7), (gb(13, 1), 0.95))
    # bar 12: brass cluster crescendo + trumpet alarm stabs
    hold("tbn", ["D3", "Eb3", "Ab3"], gb(12, 1), gb(13, 1) + 0.05)
    hold("hns", ["G3", "Bb3", "Eb4"], gb(12, 1), gb(13, 1) + 0.05)
    hold("tuba", ["D2"], gb(12, 1), gb(13, 1) + 0.05)
    for pn in ("tbn", "hns", "tuba"):
        P(pn).d((gb(12, 1), 0.5), (gb(12, 4.9), 0.95))
    for k in range(8):
        t = gb(12, 1) + k * 0.5
        P("tpt_st").n("D5", t, 0.5, 0.6 + 0.04 * k)
        P("tpt_st").n("Eb5", t, 0.5, 0.6 + 0.04 * k)
    P("vc_tr").n("D2", gb(12, 1), 4.05)
    P("vc_tr").d((gb(12, 1), 0.5), (gb(13, 1), 0.9))

    # ---- GRASP (bar 13, 960-1039): riser funnel; cut at 1037, IMPACT at 1040
    g0, cut = gb(13, 1), fb(1037)
    glis = P("gliss")
    glis.n("D5", g0, cut - g0, 0.8, to=m("D7"), patch="vln_trem")
    glis.n("A4", g0, cut - g0, 0.75, to=m("A5") + 3, patch="vla_trem")
    glis.n("D3", g0, cut - g0, 0.8, to=m("D4") + 2, patch="vc_trem")
    glis.d((g0, 0.45), (cut, 1.0))
    P("shepard").n(60, g0, cut - g0, 0.75, oct_per_s=0.55, base=50.0)
    P("revcym").n(60, cut - 3.0, 3.0, 0.9)
    P("riser").n(60, gb(12, 3), cut - gb(12, 3), 0.8, f0=150, f1=12000, curve=2.2)
    hold("tbn", ["D3", "Ab3"], g0, cut - 0.02)
    hold("hns", ["A3", "Eb4"], g0, cut - 0.02)
    hold("tpt", ["D5", "Eb5"], g0, cut - 0.02)
    hold("tuba", ["D2"], g0, cut - 0.02)
    for pn in ("tbn", "hns", "tpt", "tuba"):
        P(pn).d((g0, 0.35), (cut - 0.1, 1.0))
    # drums: 8ths -> 16ths -> 32nd roll, crescendo, cut at 1037
    t = g0
    step = 0.5
    while t < cut - 1e-6:
        prog = (t - g0) / (cut - g0)
        step = 0.5 if prog < 0.25 else (0.25 if prog < 0.6 else 0.125)
        v = 0.5 + 0.5 * prog
        taiko.n(60, t, step, v, drum="n")
        ten.n(60, t, step, v * 0.9)
        if prog > 0.5:
            tenh.n(60, t + step / 2, step / 2, v * 0.8)
        t += step
    P("timp_roll").n("D2", g0, cut - g0, rel=0.05)
    P("timp_roll").d((g0, 0.3), (cut - 0.05, 1.0))
    P("snare_roll").n(60, gb(12, 3), cut - gb(12, 3), 0.9, kind="sus", rel=0.04)
    P("snare_roll").d((gb(12, 3), 0.2), (cut, 1.0))

    # ---- IMPACT (frame 1040): huge low hit + sub drop + BRAAM, then silence
    I = fb(1040)
    P("impact").n(60, I, 4, 1.0, sync=True, size=1.4, crack=1.0)
    P("subdrop").n(60, I, 5, 1.0, sync=True, f0=64, f1=22)
    for p in ("D1", "D2", "Ab2", "D3"):
        P("braam").n(p, I, 1.4, 1.0, sync=True, rel=2.6)
    P("bdrum").n(60, I, 4, 1.0, sync=True)
    P("giant").n(60, I, 2, 1.0, sync=True)
    P("gong").n(60, I, 8, 1.0, sync=True)
    P("timp").n("D2", I, 4, 1.0, sync=True)
    P("cym").n(60, I, 6, 1.0, sync=True)
    taiko.n(60, I, 1, 1.0, drum="o", sync=True)
    for pn, ps in (("tbn", ["D2", "Ab2", "D3"]), ("tuba", ["D1"]), ("hns", ["D3", "Ab3"])):
        for p in ps:
            P(pn).n(p, I, 1.3, 1.0, sync=True)
    P("cb").n("D1", I, 1.3, 1.0, sync=True)
    P("vc").n("D2", I, 1.3, 1.0, sync=True)
    for pn in ("tbn", "tuba", "hns", "cb", "vc"):
        P(pn).d((I, 1.0), (I + 1.3, 0.8))


# ---------------------------------------------------------------------------
# 5. SILENCE  bars 14-15: piano fragments of the call
# ---------------------------------------------------------------------------
def silence():
    pn = P("piano")
    pn.n("D4", fb(1070), 7, 0.24)
    pn.n("D3", fb(1070) + 0.02, 7, 0.12)
    pn.n("A4", fb(1106), 6, 0.21)
    pn.n("D5", fb(1144), 12, 0.23)
    # "Together": the low fifth joins the ringing D5
    pn.n("D2", fb(1172), 10, 0.15)
    pn.n("A2", fb(1172) + 0.03, 10, 0.13)


# ---------------------------------------------------------------------------
# 6. FIRST BEACON  bars 16-18 (1200-1439)
# ---------------------------------------------------------------------------
def first_beacon():
    cb, vc, vla, vln2, vln1 = P("cb"), P("vc"), P("vla"), P("vln2"), P("vln1")
    k = fb(1318)       # kindling catches
    roar = fb(1360)    # the beacon roars
    # breathing low pad in the dark
    cb.n("D2", gb(16, 1.4), fb(1340) - gb(16, 1.4) + 0.05)
    vc.n("D3", gb(16, 2.5), fb(1340) - gb(16, 2.5) + 0.05)
    cb.d((gb(16, 1.4), 0.08), (gb(16, 3), 0.17), (gb(17, 1), 0.12), (k, 0.2), (fb(1340), 0.3),
         (roar, 0.55), (gb(18, 3), 0.5), (gb(19, 1), 0.5))
    vc.d((gb(16, 2.5), 0.07), (gb(17, 1), 0.14), (k, 0.2), (fb(1340), 0.34), (roar, 0.52),
         (gb(19, 1), 0.5))
    # 1318: a soft Bb chord blooms over the D pedal (Bb/D)
    vc.n("F3", k, roar - k + 0.05)
    vla.n("Bb3", k, roar - k + 0.05)
    vln2.n("D4", k, roar - k + 0.05)
    vln2.n("F4", k, roar - k + 0.05)
    vln1.n("Bb4", k, roar - k + 0.05)
    for pn in ("vla", "vln2", "vln1"):
        P(pn).d((k, 0.12), (fb(1340), 0.3), (roar - 0.1, 0.46), (roar, 0.52), (gb(18, 3), 0.42),
                (gb(19, 1), 0.46))
    # 1340: bass steps to C (Bb/C = C9sus4)
    cb.n("C2", fb(1340), roar - fb(1340) + 0.05, legato=True)
    vc.n("C3", fb(1340), roar - fb(1340) + 0.05, legato=True)
    # timpani soft roll building into the roar
    P("timp_roll").n("C2", fb(1322), roar - fb(1322) + 0.02, rel=0.1)
    P("timp_roll").d((fb(1322), 0.1), (roar - 0.05, 0.7))
    P("swell_s").n(60, roar, 1, 0.5, sync=True)
    # harp glissando up into the roar (Bb lydian-ish)
    hp = P("harp")
    gl = ["Bb2", "C3", "D3", "F3", "G3", "Bb3", "C4", "D4", "F4", "G4", "Bb4", "C5", "D5", "F5"]
    t0 = roar - 1.0
    for i, p in enumerate(gl):
        hp.n(p, t0 + i * (1.0 / len(gl)), 3, 0.28 + 0.02 * i)
    # 1360: F major. The beacon roars; the solo horn sings the call + answer
    P("timp").n("F2", roar, 2, 0.72, sync=True)
    P("bdrum").n(60, roar, 2, 0.55, sync=True)
    P("crash").n(60, roar, 4, 0.4, sync=True)
    cb.n("F1", roar, 8.1, legato=False)
    vc.n("F2", roar, 8.1)
    vc.n("C3", roar, 8.1)
    vla.n("A3", roar, 4.1)
    vln2.n("A4", roar, 4.1)
    vln2.n("C5", roar, 4.1)
    vln1.n("F5", roar, 4.1)
    vln1.n("A5", roar, 4.1)
    for i, p in enumerate(["F2", "C3", "F3", "A3", "C4", "F4", "A4"]):
        hp.n(p, roar + i * 0.1, 3, 0.35)
    hn = P("hn_solo")
    hn.line("F3:1 C4:1 F4:2", roar)
    hn.line("F4:1 E4:.5 D4:.5 C4:1.2", gb(19, 1))
    hn.d((roar, 0.62), (roar + 1, 0.66), (roar + 2, 0.72), (roar + 3.5, 0.6), (gb(19, 1), 0.64),
         (gb(19, 2), 0.56), (gb(19, 3), 0.5), (gb(19, 4), 0.3))


# ---------------------------------------------------------------------------
# 7. BEACONS montage  bars 19-22 (1440-1759): ignition-locked harmony + stretto
# ---------------------------------------------------------------------------
IGN = [1480, 1540, 1600, 1650, 1690, 1730]


def beacons():
    cb, vc, vla, vln2, vln1 = P("cb"), P("vc"), P("vla"), P("vln2"), P("vln1")
    I = [fb(f) for f in IGN]
    end = gb(23, 1)
    # chord timeline: (start, bass(cb), cello/bass octave, chord tones for ostinato)
    chords = [
        (gb(19, 1), "F1", "F2", ["F4", "A4", "C5", "A4"], ["C4", "F4", "A4", "F4"]),
        (I[0], "A1", "A2", ["F4", "A4", "C5", "A4"], ["C4", "F4", "A4", "F4"]),
        (I[1], "Bb1", "Bb2", ["F4", "Bb4", "D5", "Bb4"], ["D4", "F4", "Bb4", "F4"]),
        (I[2], "C2", "C3", ["G4", "C5", "E5", "C5"], ["E4", "G4", "C5", "G4"]),
        (I[3], "D2", "D3", ["F4", "A4", "D5", "A4"], ["D4", "F4", "A4", "F4"]),
        (I[4], "E2", "E3", ["G4", "C5", "E5", "C5"], ["E4", "G4", "C5", "G4"]),
        (I[5], "F2", "F3", ["A4", "C5", "F5", "C5"], ["F4", "A4", "C5", "A4"]),
    ]
    # hold sustained bass line (cb + vc), rising
    for i, (t, b1, b2, v2pat, vapat) in enumerate(chords):
        t_end = chords[i + 1][0] if i + 1 < len(chords) else end
        if i == 0:
            continue  # bar 19 b1-2 held from the roar (F)
        cb.n(b1, t, t_end - t + 0.05, legato=True)
        if i < 4:
            vc.n(b2, t, t_end - t + 0.05, legato=True)
    cb.d((gb(19, 1), 0.5), (gb(21, 1), 0.6), (gb(22, 4.9), 0.78))
    vc.d((gb(19, 1), 0.5), (gb(21, 1), 0.6), (gb(22, 4.9), 0.78))
    # 8th-note ostinato (vln2 + vla, spiccato), accents on beats
    for i, (t, b1, b2, v2pat, vapat) in enumerate(chords):
        t_end = chords[i + 1][0] if i + 1 < len(chords) else end
        tt = np.ceil(t * 2 - 1e-6) / 2
        k = 0
        while tt < t_end - 1e-6:
            prog = (tt - gb(19, 1)) / 16
            on_beat = abs(tt - round(tt)) < 1e-6
            v = (0.5 + 0.28 * prog) * (1.0 if on_beat else 0.8)
            P("vln2_sp").n(v2pat[k % 4], tt, 0.5, v)
            P("vla_sp").n(vapat[k % 4], tt, 0.5, v * 0.95)
            P("cb_sp").n(b1, tt, 0.5, v * (1.0 if on_beat else 0.7))
            tt += 0.5
            k += 1
    # timpani: pulse on the bass + accent on every ignition
    tp = P("timp")
    for i, (t, b1, b2, *_ ) in enumerate(chords):
        t_end = chords[i + 1][0] if i + 1 < len(chords) else end
        tb = m(b2)
        while tb > 57:
            tb -= 12
        while tb < 38:
            tb += 12
        tt = t
        first = True
        while tt < t_end - 1e-6:
            prog = (tt - gb(19, 1)) / 16
            tp.n(tb, tt, 1, (0.78 if first and i > 0 else 0.42 + 0.3 * prog), sync=(first and i > 0))
            tt += 1.0
            first = False
    for t in I:
        P("bdrum").n(60, t, 2, 0.5, sync=True)
    # snare building from bar 21, roll into the globe
    for k in range(int((I[5] - gb(21, 1)) * 2)):
        t = gb(21, 1) + k * 0.5
        P("snare").n(60, t, 0.5, 0.3 + 0.35 * k / 12)
    P("snare_roll").n(60, I[5], end - I[5] + 0.02, 0.8, kind="sus", rel=0.05)
    P("snare_roll").d((I[5], 0.35), (end, 0.85))
    P("swell").n(60, end, 1, 0.55, sync=True)

    # ---- the calls: a new voice per beacon (stretto) ----------------------
    # 1480 far peak: horns a3 (same call as the first beacon: it is answered)
    call("hns", "F3", I[0], vel=None, last=3.0)
    P("hns").d((I[0], 0.72), (I[0] + 2, 0.78), (I[0] + 4, 0.62), (I[1] + 1, 0.5))
    # 1540 desert: trumpet
    call("tpt", "D4", I[1], last=2.5)
    P("tpt").d((I[1], 0.7), (I[1] + 2, 0.78), (I[1] + 4, 0.6))
    # 1600 ice: flute + glock
    call("fl2", "C5", I[2], last=2.5)
    P("fl2").d((I[2], 0.62), (I[2] + 2, 0.7), (I[2] + 4, 0.5))
    for j, p in enumerate(["C6", "G6", "C7"]):
        P("glock").n(p, I[2] + [0, 1, 2][j], 1, 0.5, sync=(j == 0))
        P("celesta").n(p, I[2] + [0, 1, 2][j], 1, 0.35)
    # 1650 karst jungle: cellos (the call at its original pitch)
    call("vc", "D3", I[3], last=1.6)
    vc.n("F2", I[5], end - I[5] + 0.05)
    vc.n("C3", I[5], end - I[5] + 0.05)
    # 1690 city: trombones
    call("tbn", "C3", I[4], last=2.0)
    P("tbn").d((I[4], 0.72), (I[4] + 2, 0.8), (end, 0.7))
    # 1730 sea: high violins (section + solo violin on top)
    call("vln1", "F5", I[5], last=5.5)
    call("svln", "F5", I[5], last=5.5)
    P("svln").d((I[5], 0.6), (end + 1, 0.8))
    # upper strings: sustained chord support under the stretto
    sup = [(gb(19, 1), ["A4", "C5"], ["F5"]), (I[0], ["A4", "C5"], ["F5"]),
           (I[1], ["Bb4", "D5"], ["F5"]), (I[2], ["C5", "E5"], ["G5"]),
           (I[3], ["D5", "F5"], ["A5"]), (I[4], ["C5", "E5"], ["G5"])]
    for i, (t, v2, v1) in enumerate(sup):
        t_end = sup[i + 1][0] if i + 1 < len(sup) else I[5]
        for p in v1:
            vln1.n(p, t, t_end - t + 0.05, legato=i > 0)
    vln1.d((gb(19, 1), 0.34), (I[2], 0.44), (I[5], 0.55), (I[5] + 1, 0.72), (end, 0.8))
    # harp sparkle on each ignition
    for i, t in enumerate(I):
        root = ["F", "Bb", "C", "D", "C", "F"][i]
        for j, oc in enumerate([3, 4, 5]):
            P("harp").n(f"{root}{oc}", t + j * 0.07, 2, 0.35 + 0.03 * i)


# ---------------------------------------------------------------------------
# 8. THE WORLD ANSWERS  bars 23-24: choir enters, organ pedal, strings
# ---------------------------------------------------------------------------
def globe():
    g0 = gb(23, 1)
    ch = P("choir")
    # F (bar 23) - Bb/F (24.1) - C/F (24.3), top line A5 - Bb5 - C6
    for p in ["F2", "C3", "A3", "F4", "C5", "A5"]:
        ch.n(p, g0, 4.05, sync=True)
    for p in ["F2", "D3", "Bb3", "F4", "D5", "Bb5"]:
        ch.n(p, gb(24, 1), 2.05)
    for p in ["F2", "C3", "G3", "E4", "C5", "C6"]:
        ch.n(p, gb(24, 3), 2.3)
    ch.d((g0 - 0.2, 0.3), (g0 + 0.3, 0.62), (gb(23, 3), 0.66), (gb(24, 3), 0.8), (gb(24, 4.5), 0.72),
         (gb(25, 1), 0.3))
    op = P("organ_ped")
    op.n("F2", g0, 8.1, sync=True)
    op.d((g0, 0.55), (gb(24, 3), 0.7), (gb(25, 1), 0.35))
    org = P("organ")
    for p in ["F3", "A3", "C4"]:
        org.n(p, g0, 4.05)
    for p in ["F3", "Bb3", "D4"]:
        org.n(p, gb(24, 1), 2.05)
    for p in ["E3", "G3", "C4"]:
        org.n(p, gb(24, 3), 2.2)
    org.d((g0, 0.35), (gb(24, 3), 0.45), (gb(25, 1), 0.2))
    # strings sustain
    cb, vc, vla, vln2, vln1 = P("cb"), P("vc"), P("vla"), P("vln2"), P("vln1")
    cb.n("F1", g0, 8.1, legato=True)
    vc.n("F2", g0, 8.1)
    vc.n("C3", g0, 4.05)
    vc.n("D3", gb(24, 1), 2.05, legato=True)
    vc.n("C3", gb(24, 3), 2.1, legato=True)
    vla.n("A3", g0, 4.05)
    vla.n("C4", g0, 4.05)
    vla.n("Bb3", gb(24, 1), 2.05, legato=True)
    vla.n("D4", gb(24, 1), 2.05, legato=True)
    vla.n("G3", gb(24, 3), 2.1, legato=True)
    vla.n("C4", gb(24, 3), 2.1, legato=True)
    vln2.n("F4", g0, 4.05)
    vln2.n("C5", g0, 4.05)
    vln2.n("F4", gb(24, 1), 2.05, legato=True)
    vln2.n("D5", gb(24, 1), 2.05, legato=True)
    vln2.n("E4", gb(24, 3), 2.1, legato=True)
    vln2.n("C5", gb(24, 3), 2.1, legato=True)
    vln1.n("E6", gb(24, 3), 2.1, legato=True)
    P("svln").n("E6", gb(24, 3), 2.1, legato=True)
    vln2.n("A5", g0 + 0.5, 3.55)
    vln2.n("Bb5", gb(24, 1), 2.05, legato=True)
    vln2.n("C6", gb(24, 3), 2.1, legato=True)
    for pn in ("cb", "vc", "vla", "vln2", "vln1"):
        P(pn).d((g0, 0.62), (gb(24, 3), 0.72), (gb(24, 4.6), 0.55), (gb(25, 1), 0.3))
    P("cym").n(60, g0, 4, 0.55, sync=True)
    P("timp").n("F2", g0, 2, 0.75, sync=True)
    P("bdrum").n(60, g0, 2, 0.6, sync=True)
    # the web spreading: tiny calls all over the stereo field
    rng = np.random.default_rng(23)
    roots = ["F5", "C6", "A5", "F6", "C5", "Bb5", "F5", "D6", "C6", "G5", "E6", "C6"]
    for j in range(12):
        t = g0 + 0.5 + j * 0.62 + rng.uniform(-0.1, 0.1)
        r = m(roots[j])
        pan = float(np.clip(rng.uniform(-0.9, 0.9), -1, 1))
        v = 0.22 + 0.1 * rng.uniform()
        for q, d in enumerate([0, 0.25, 0.5]):
            P("celesta").n(r + [0, 7, 12][q], t + d, 0.5, v, pan=pan)
        P("glock").n(r + 12 if r + 12 <= 96 else r, t + 0.5, 1, v * 0.8)
    P("crash").n(60, gb(24, 3), 3, 0.35)


# ---------------------------------------------------------------------------
# 9. THE ACCORD  bars 25-28: hush (Dm), the four oaths, A pedal, ring sweep
# ---------------------------------------------------------------------------
OATHS = [2000, 2040, 2080, 2120]


def accord():
    a0 = gb(25, 1)
    cb, vc, vla, vln2, vln1 = P("cb"), P("vc"), P("vla"), P("vln2"), P("vln1")
    O = [fb(f) for f in OATHS]
    ring0, ring1, dawn = fb(2160), fb(2220), fb(2240)
    # bar 25: hush in D minor; the call quietly in the clarinet
    cb.n("D2", a0, 4.05 + 2.0, legato=False)
    vc.n("D3", a0, 4.05)
    vc.n("A3", a0, 4.05)
    vla.n("F4", a0, 4.05)
    vln2.n("A4", a0, 4.05)
    vln1.n("D5", a0, 4.05)
    for pn in ("cb", "vc", "vla", "vln2", "vln1"):
        P(pn).d((a0, 0.3), (a0 + 0.5, 0.26), (gb(25, 4), 0.32), (O[0], 0.42))
    co = P("choir_oo")
    for p in ["D3", "A3", "D4", "F4", "A4"]:
        co.n(p, a0, 4.1, reg="alto" if m(p) >= 60 else "tenor")
    co.d((a0, 0.25), (gb(25, 3), 0.3), (O[0], 0.35))
    cl = P("cl")
    cl.line("D4:1 A4:1 D5:1.8", a0 + 0.5)
    cl.d((a0, 0.3), (a0 + 2.5, 0.4), (a0 + 3.8, 0.25))
    # heartbeat timpani: quarters (bar 25) -> 8ths (26-27) -> roll (28)
    tp = P("timp")
    for k in range(4):
        tp.n("D2", a0 + k, 1, 0.3 + 0.03 * k)

    # the oaths: chord + timpani + low bell + choir swell, top line climbing
    oath_ch = [
        # bass(cb), vc, vla, vln2, vln1(top), choir, tbn, tuba, bell, timp
        ("D2", ["D3", "Bb3"], ["F4", "D4"], ["Bb4", "D5"], "F5",
         ["D3", "Bb3", "D4", "F4", "Bb4", "F5"], ["F3", "Bb3", "D4"], "D2", "Bb3", "D2"),
        ("C2", ["C3", "G3"], ["E4", "G4"], ["C5", "E5"], "G5",
         ["C3", "G3", "E4", "G4", "C5", "G5"], ["E3", "G3", "C4"], "C2", "C4", "C2"),
        ("A1", ["A2", "F3"], ["C4", "F4"], ["C5", "F5"], "A5",
         ["A2", "C4", "F4", "A4", "C5", "A5"], ["F3", "A3", "C4"], "A1", "F3", "A2"),
        ("A1", ["A2", "G3"], ["D4", "G4"], ["D5", "G5"], "B5",
         ["A2", "D4", "G4", "B4", "D5", "B5"], ["G3", "B3", "D4"], "A1", "G3", "A2"),
    ]
    ch = P("choir")
    for i, (bass, vcs, vlas, v2s, top, chs, tbs, tu, bellp, tim) in enumerate(oath_ch):
        t = O[i]
        t1 = O[i + 1] if i < 3 else ring0
        cb.n(bass, t, t1 - t + 0.05, legato=i > 0)
        for p in vcs:
            vc.n(p, t, t1 - t + 0.05)
        for p in vlas:
            vla.n(p, t, t1 - t + 0.05)
        for p in v2s:
            vln2.n(p, t, t1 - t + 0.05)
        vln1.n(top, t, t1 - t + 0.05, legato=i > 0)
        P("svln").n(top, t, t1 - t + 0.05, legato=i > 0)
        for p in chs:
            ch.n(p, t, t1 - t + 0.05, sync=True)
        for p in tbs:
            P("tbn").n(p, t, t1 - t + 0.05)
        P("tuba").n(tu, t, t1 - t + 0.05)
        P("hns").n(m(top) - 24 if m(top) - 24 >= 50 else m(top) - 12, t, t1 - t + 0.05)
        P("bell").n(bellp, t, 4, 0.55 + 0.08 * i, sync=True)
        tp.n(tim, t, 2, 0.62 + 0.1 * i, sync=True)
        P("bdrum").n(60, t, 2, 0.45 + 0.08 * i, sync=True)
        if i >= 2:
            P("triangle").n(60, t, 2, 0.2 + 0.05 * i)
        # heartbeat 8ths between oaths
        for k in range(1, int((t1 - t) * 2)):
            tp.n(tim, t + k * 0.5, 0.5, 0.3 + 0.08 * i + 0.02 * k)
    # swells at each oath (choir + strings), overall crescendo
    ch.d((O[0] - 0.05, 0.35), (O[0] + 0.15, 0.62), (O[0] + 1.6, 0.46),
         (O[1] + 0.15, 0.68), (O[1] + 1.6, 0.52), (O[2] + 0.15, 0.74), (O[2] + 1.6, 0.6),
         (O[3] + 0.15, 0.8), (O[3] + 1.6, 0.68), (ring0, 0.72), (dawn - 0.1, 0.95))
    for pn, base in (("vln1", 0.5), ("vln2", 0.46), ("vla", 0.46), ("vc", 0.5), ("cb", 0.52),
                     ("svln", 0.4)):
        P(pn).d((O[0], base + 0.1), (O[0] + 1.5, base), (O[1], base + 0.14), (O[1] + 1.5, base + 0.04),
                (O[2], base + 0.18), (O[2] + 1.5, base + 0.1), (O[3], base + 0.24),
                (O[3] + 1.5, base + 0.16), (ring0, base + 0.2), (dawn - 0.1, min(1.0, base + 0.42)))
    for pn in ("tbn", "tuba", "hns"):
        P(pn).d((O[0], 0.42), (O[1], 0.5), (O[2], 0.58), (O[3], 0.66), (ring0, 0.66),
                (dawn - 0.1, 0.92))
    op = P("organ_ped")
    op.n("D2", O[0], O[1] - O[0] + 0.05)
    op.n("C2", O[1], O[2] - O[1] + 0.05)
    op.n("A2", O[2], dawn - O[2] + 0.02)
    op.d((O[0], 0.35), (O[2], 0.45), (ring0, 0.5), (dawn - 0.1, 0.8))

    # bar 28: A major (sus4 -> 3), the "together" ring sweep, roll into the dawn
    cb.n("A1", ring0, dawn - ring0 + 0.02, legato=True)
    vc.n("A2", ring0, dawn - ring0 + 0.02)
    vc.n("E3", ring0, dawn - ring0 + 0.02)
    vla.n("A3", ring0, dawn - ring0 + 0.02)
    vla.n("D4", ring0, 2.0)
    vla.n("C#4", ring0 + 2, dawn - ring0 - 2 + 0.02, legato=True)
    vln2.n("E4", ring0, dawn - ring0 + 0.02)
    vln2.n("A4", ring0, dawn - ring0 + 0.02)
    vln1.n("E5", ring0, dawn - ring0 + 0.02)
    vln1.n("C#6", ring0, dawn - ring0 + 0.02, legato=True)
    P("svln").n("C#6", ring0, dawn - ring0 + 0.02, legato=True)
    for p in ["A2", "E3", "A3", "D4", "E4", "A4", "C#6"]:
        ch.n(p, ring0, dawn - ring0 + 0.05)
    ch.n("D5", ring0, 2.0)
    ch.n("C#5", ring0 + 2, dawn - ring0 - 2 + 0.05)
    for p in ["E3", "A3", "C#4"]:
        P("tbn").n(p, ring0, dawn - ring0 - 0.02)
    P("tuba").n("A1", ring0, dawn - ring0 - 0.02)
    P("hns").n("A3", ring0, dawn - ring0 - 0.02)
    P("hns").n("E4", ring0, dawn - ring0 - 0.02)
    P("tpt").n("A4", ring0 + 1, dawn - ring0 - 1 - 0.02)
    P("tpt").n("E5", ring0 + 1, dawn - ring0 - 1 - 0.02)
    P("tpt").d((ring0 + 1, 0.3), (dawn - 0.1, 0.85))
    P("vln_tr").n("A5", ring0, dawn - ring0 + 0.02)
    P("vln_tr").n("E5", ring0, dawn - ring0 + 0.02)
    P("vln_tr").d((ring0, 0.3), (dawn - 0.05, 0.85))
    # ring sweep: harp glissando rising, panned L->R, + shimmering bells
    sc = ["A2", "B2", "C#3", "E3", "F#3", "A3", "B3", "C#4", "E4", "F#4", "A4", "B4", "C#5", "E5",
          "F#5", "A5", "B5", "C#6", "E6", "F#6", "A6"]
    for i, p in enumerate(sc):
        t = ring0 + (ring1 - ring0) * i / (len(sc) - 1)
        P("harp").n(p, t, 3, 0.36 + 0.015 * i, pan=-0.85 + 1.7 * i / (len(sc) - 1))
        if i % 2 == 0 and m(p) >= 67:
            P("celesta").n(p, t, 1, 0.3, pan=-0.85 + 1.7 * i / (len(sc) - 1))
    for k in range(12):
        t = ring0 + k * 0.25
        P("glass").n(["A6", "E6", "C#7", "E6"][k % 4], t, 0.25, 0.18 + 0.01 * k, pan=0.6 * np.sin(k))
    # timpani roll + snare roll + cymbal swell + reverse cymbal -> 2240
    P("timp_roll").n("A2", ring0, dawn - ring0 - 0.01, rel=0.03)
    P("timp_roll").d((ring0, 0.3), (dawn - 0.05, 0.95))
    P("snare_roll").n(60, gb(28, 2), dawn - gb(28, 2) - 0.01, 0.9, kind="sus", rel=0.03)
    P("snare_roll").d((gb(28, 2), 0.15), (dawn - 0.05, 0.9))
    P("swell").n(60, dawn, 1, 0.8, sync=True)
    P("revcym").n(60, dawn - 2.0, 2.0, 0.7)
    P("riser").n(60, fb(2200), dawn - fb(2200), 0.45, f0=400, f1=10000, curve=2.0)


# ---------------------------------------------------------------------------
# 10. DAWN  bars 29-31 (2240-2479): D MAJOR, the theme, full orchestra
# ---------------------------------------------------------------------------
def dawn():
    d0 = fb(2240)
    b30, b31, b32 = gb(30, 1), gb(31, 1), gb(32, 1)
    # the big downbeat
    P("cym").n(60, d0, 8, 1.0, sync=True)
    P("crash").n(60, d0, 8, 0.9, sync=True)
    P("gong").n(60, d0, 8, 0.7, sync=True)
    P("timp").n("D2", d0, 2, 1.0, sync=True)
    P("timp").n("A2", d0 + 0.02, 2, 0.8)
    P("bdrum").n(60, d0, 4, 0.95, sync=True)
    P("giant").n(60, d0, 2, 0.7, sync=True)
    P("impact").n(60, d0, 2, 0.5, sync=True, size=1.1, crack=0.2)
    # timpani pulse under the theme
    for t, p, v in [(d0 + 2, "A2", 0.55), (d0 + 3, "D2", 0.6), (b30, "D2", 0.75),
                    (b30 + 2, "A2", 0.6), (b31, "G2", 0.75), (b31 + 2, "D2", 0.62)]:
        P("timp").n(p, t, 1, v)

    # melody (the theme): trumpets, violins 8va, horns 8vb, solo violin top
    mel_t = "D4:1 A4:1 D5:2 | D5:1 C#5:.5 B4:.5 F#4:2 | G4:.5 A4:.5 B4:1 D5:1 Bb4:1.1"
    P("tpt").line(mel_t, d0)
    P("tpt").d((d0, 0.95), (b30, 0.86), (b31, 0.8), (b31 + 2, 0.72), (b32, 0.35))
    P("vln1").line("D5:1 A5:1 D6:2 | D6:1 C#6:.5 B5:.5 F#5:2 | G5:.5 A5:.5 B5:1 D6:1 Bb5:1", d0)
    P("vln1").n("A5", b32, 2.0, legato=True)
    P("svln").line("D6:1 A6:1 D7:2 | D7:1 C#7:.5 B6:.5 F#6:2 | G6:.5 A6:.5 B6:1 D7:1 Bb6:1", d0)
    P("svln").d((d0, 0.55), (b31, 0.5), (b32, 0.2))
    P("hns").line("D3:1 A3:1 D4:2 | D4:1 C#4:.5 B3:.5 F#3:2 | G3:.5 A3:.5 B3:1 D4:1 Bb3:1.1", d0)
    P("hns").d((d0, 0.95), (b30, 0.88), (b31, 0.8), (b31 + 2, 0.7), (b32, 0.3))
    P("tpt").n("A4", b32, 1.2, legato=True)
    P("hns").n("A3", b32, 1.4, legato=True)
    # choir: harmony pad (aah), top voice bright
    ch = P("choir")
    chords = [
        (d0, 4, ["D3", "A3", "D4", "F#4", "A4", "D5", "F#5", "A5"]),
        (b30, 2, ["B2", "F#3", "D4", "F#4", "B4", "D5", "F#5"]),
        (b30 + 2, 2, ["F#2", "A3", "D4", "F#4", "A4", "D5", "A5"]),
        (b31, 2, ["G2", "B3", "D4", "G4", "B4", "D5", "G5"]),
        (b31 + 2, 2, ["Bb2", "G3", "D4", "G4", "Bb4", "D5", "G5"]),
        (b32, 2, ["D3", "A3", "D4", "F#4", "A4", "D5"]),
    ]
    for t, d, ps in chords:
        for p in ps:
            ch.n(p, t, d + 0.05, sync=(t == d0))
    ch.d((d0, 1.0), (d0 + 1.5, 0.9), (b30, 0.88), (b31, 0.84), (b31 + 2, 0.72), (b32, 0.3),
         (b32 + 2, 0.12))
    # organ: full, D pedal
    org, op = P("organ"), P("organ_ped")
    for t, d, ps in [(d0, 4, ["D3", "F#3", "A3", "D4"]), (b30, 2, ["D3", "F#3", "B3"]),
                     (b30 + 2, 2, ["D3", "F#3", "A3"]), (b31, 2, ["D3", "G3", "B3"]),
                     (b31 + 2, 2, ["D3", "G3", "Bb3"]), (b32, 2, ["D3", "F#3", "A3"])]:
        for p in ps:
            org.n(p, t, d + 0.03, sync=(t == d0))
    org.d((d0, 0.8), (b31 + 2, 0.6), (b32, 0.25), (b32 + 2, 0.1))
    for t, d, p in [(d0, 4, "D2"), (b30, 2, "B1"), (b30 + 2, 2, "F#2"), (b31, 2, "G2"),
                    (b31 + 2, 2, "Bb1"), (b32, 2.0, "D2")]:
        op.n(p, t, d + 0.03, sync=(t == d0))
    op.d((d0, 0.85), (b31 + 2, 0.7), (b32, 0.3), (b32 + 2, 0.1))
    # strings: bass + harmony
    cb, vc, vla, vln2 = P("cb"), P("vc"), P("vla"), P("vln2")
    bass = [(d0, 4, "D2", ["D3", "A3"]), (b30, 2, "B1", ["B2", "F#3"]), (b30 + 2, 2, "F#1", ["F#2", "A3"]),
            (b31, 2, "G1", ["G2", "D3"]), (b31 + 2, 2, "Bb1", ["Bb2", "G3"]), (b32, 2, "D2", ["D3", "A3"])]
    for i, (t, d, b, vcs) in enumerate(bass):
        cb.n(b, t, d + 0.05, legato=i > 0, sync=(i == 0))
        for p in vcs:
            vc.n(p, t, d + 0.05, legato=i > 0, sync=(i == 0))
    vl = [(d0, 4, ["F#4", "A4"], ["A4", "D5"]), (b30, 2, ["F#4", "B4"], ["B4", "D5"]),
          (b30 + 2, 2, ["F#4", "A4"], ["A4", "D5"]), (b31, 2, ["G4", "B4"], ["B4", "D5"]),
          (b31 + 2, 2, ["G4", "Bb4"], ["Bb4", "D5"]), (b32, 2, ["F#4", "A4"], ["A4", "D5"])]
    for i, (t, d, vas, v2s) in enumerate(vl):
        for p in vas:
            vla.n(p, t, d + 0.05, legato=i > 0, sync=(i == 0))
        for p in v2s:
            vln2.n(p, t, d + 0.05, legato=i > 0, sync=(i == 0))
    for pn, top in (("cb", 0.95), ("vc", 0.95), ("vla", 0.9), ("vln2", 0.9), ("vln1", 0.95)):
        P(pn).d((d0, top), (b30, top - 0.05), (b31, top - 0.08), (b31 + 2, top - 0.2),
                (b32, 0.3), (b32 + 1.0, 0.2), (b32 + 2.0, 0.1))
    # trombones + tuba: warm chords, contrary motion to the bass
    tb, tu = P("tbn"), P("tuba")
    for t, d, ps, tub in [(d0, 4, ["F#3", "A3", "D4"], "D2"), (b30, 2, ["F#3", "B3", "D4"], "B1"),
                          (b30 + 2, 2, ["F#3", "A3", "D4"], "F#1"), (b31, 2, ["G3", "B3", "D4"], "G1"),
                          (b31 + 2, 2, ["G3", "Bb3", "D4"], "Bb1")]:
        for p in ps:
            tb.n(p, t, d + 0.03, sync=(t == d0))
        tu.n(tub, t, d + 0.03, sync=(t == d0))
    tb.d((d0, 0.8), (b31 + 2, 0.66), (b32, 0.3))
    tu.d((d0, 0.8), (b31 + 2, 0.66), (b32, 0.3))
    # harp arpeggios (light), glock doubling of the call
    for i, p in enumerate(["D3", "A3", "D4", "F#4", "A4", "D5", "F#5", "A5", "D6"]):
        P("harp").n(p, d0 + i * 0.08, 3, 0.5)
    for j, p in enumerate(["D6", "A6", "D7"]):
        P("glock").n(p, d0 + [0, 1, 2][j], 1, 0.45, sync=(j == 0))
    P("cym").n(60, b31 + 2, 4, 0.35)
    P("swell_s").n(60, b31 + 2.2, 1, 0.3)


# ---------------------------------------------------------------------------
# 11. CODA  bars 32-35 (2480-2807)
# ---------------------------------------------------------------------------
def coda():
    c0 = fb(2480)
    fl = P("fl")
    # the call; the D5 hangs alone through the question (2500-2570)
    fl.line("D4:1 A4:1 D5:3.0", c0)
    # the answer, during the silent hand-over (2580-2620)
    fl.line("C#5:.5 B4:.5 F#4:2.4", fb(2580))
    fl.d((c0, 0.3), (c0 + 1.5, 0.36), (fb(2530), 0.32), (fb(2575), 0.22), (fb(2580), 0.3),
         (fb(2600), 0.28), (fb(2640), 0.2))
    # soft string pad under the call, thinning to nothing at the question
    vc, vla, cb = P("vc"), P("vla"), P("cb")
    # (bar 32 downbeat chord comes from the DAWN block; it decays here)
    # low pedal ppp returns with the answer
    cb.n("D2", fb(2584), fb(2640) - fb(2584) + 0.05)
    vc.n("A2", fb(2590), fb(2640) - fb(2590) + 0.05)
    cb.d((fb(2584), 0.1), (fb(2640), 0.3))
    vc.d((fb(2590), 0.1), (fb(2640), 0.3))

    # 2640: the child's torch lights the beacon - warm swell (strings + choir pp)
    t0 = fb(2640)
    ch = P("choir_oo")
    swell_ch = [(t0, 2, ["D3", "A3", "D4", "F#4", "A4"]), (t0 + 2, 2, ["D3", "B3", "D4", "G4", "B4"]),
                (fb(2720), 4.35, ["D3", "A3", "E4", "F#4", "A4"])]
    for t, d, ps in swell_ch:
        for p in ps:
            ch.n(p, t, d + 0.05, reg="alto" if m(p) >= 60 else "tenor", vowel="o")
    ch.d((t0 - 0.2, 0.15), (t0 + 1, 0.42), (t0 + 3, 0.4), (fb(2720), 0.38), (fb(2760), 0.3),
         (fb(2800), 0.08))
    cb.n("D2", t0, 4.05, legato=True)
    vc.n("D3", t0, 4.05)
    vc.n("A3", t0, 2.05)
    vc.n("G3", t0 + 2, 2.05, legato=True)
    vla.n("F#4", t0, 2.05)
    vla.n("D4", t0, 2.05)
    vla.n("G4", t0 + 2, 2.05, legato=True)
    vla.n("B3", t0 + 2, 2.05, legato=True)
    P("vln2").n("A4", t0, 2.05)
    P("vln2").n("B4", t0 + 2, 2.05, legato=True)
    P("vln1").n("D5", t0, 4.05)
    P("vln1").n("F#5", t0 + 1, 3.05)
    for pn in ("cb", "vc", "vla", "vln2", "vln1"):
        P(pn).d((t0, 0.2), (t0 + 1.2, 0.42), (t0 + 3, 0.38), (fb(2720), 0.34), (fb(2760), 0.24),
                (fb(2805), 0.05))
    P("timp").n("D2", t0, 2, 0.35, sync=True)
    P("triangle").n(60, t0, 2, 0.3, sync=True)
    # answering fires: one soft bell per fire, spreading outward across the field
    bell_notes = ["A4", "B4", "D5", "E5", "F#5", "A5", "B5", "D6", "E6", "F#6", "A6", "B6"]
    times = [2645, 2651, 2657, 2663, 2669, 2675, 2682, 2689, 2696, 2703, 2710, 2716]
    for i, (p, f) in enumerate(zip(bell_notes, times)):
        side = 1 if i % 2 == 0 else -1
        pan = side * min(0.9, 0.12 + 0.07 * i)
        v = 0.42 - 0.018 * i
        P("celesta").n(p, fb(f), 1, v, pan=pan, sync=True)
        if m(p) >= 67:
            P("glock").n(p, fb(f), 1, v * 0.7, pan=pan)
    # 2720: final chord, D major add9 (D-A-E-F#), ringing out
    tf = fb(2720)
    end_b = fb(2800)
    cb.n("D2", tf, end_b - tf, legato=True)
    vc.n("D3", tf, end_b - tf)
    vc.n("A3", tf, end_b - tf, legato=True)
    vla.n("E4", tf, end_b - tf)
    vla.n("F#4", tf, end_b - tf, legato=True)
    P("vln2").n("A4", tf, end_b - tf, legato=True)
    P("vln1").n("E5", tf, end_b - tf)
    P("vln1").n("F#5", tf, end_b - tf, legato=True)
    for i, p in enumerate(["D2", "A2", "E3", "F#3", "A3", "D4", "E4", "F#4", "A4"]):
        P("harp").n(p, tf + i * 0.11, 5, 0.3)
    P("glass").n("E6", tf + 1.0, 2, 0.12, pan=0.4)
    P("glass").n("F#6", tf + 1.6, 2, 0.1, pan=-0.4)


def build():
    PARTS.clear()
    setup()
    intro()
    kindling()
    race()
    silence()
    first_beacon()
    beacons()
    globe()
    accord()
    dawn()
    coda()
    return {k: v for k, v in PARTS.items() if v.notes}
