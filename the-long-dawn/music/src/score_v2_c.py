"""THE LONG DAWN v2 - SCORE C (the Tolkien cut): the second half in the epic-fantasy
tradition.  Noble, grave, heartfelt.  Original melodies only: the film's own Beacon theme
(the CALL, root-fifth-octave, and its answer) and new material; nothing borrowed.

  FIRST BEACON  the dark; 1318 low strings bloom; 1360 a solo horn cries the CALL (D4 A4 D5)
                over D minor, the strings answer with Bb; timpani
  FAR PEAK      Gm; 1480 a second horn, far away on the right, answers on the dominant
  BEACON RUN    driving strings (16ths grouped 3+3+2), timpani on every beacon, and the CALL
                passed from peak to peak: one horn call per beacon, each on the next root of
                a climbing progression (Dm F Gm Bb C Dm A), each farther away
  MONTAGE       the whole horn section sings the CALL at the desert; then every ignition is
                answered by a horn from another place: Bb Gm Dm Bb C, into ...
  THE MAP       F major, wonder: the theme in the violins (the CALL on F, then the answer over
                Bb with its raised fourth - the IGNITION colour, now bright); harp arpeggios
  THE COUNCIL   hushed D minor: a clarinet intones the CALL; 2160 the Ring in the fire: a
                low-brass chorale (Dm, Bb/D at 2230, Gm/D at 2290 - never on a vow); the build;
                2320 Bb, 2360 A7sus4, 2380 A; the breath
  DAWN          D major over the eastern ranges: the theme soaring in the violins (call, answer,
                and a new rising consequent for the eagles), horns in counterpoint, low brass,
                timpani - no cymbal crash, no victory
  CODA          a Celtic-leaning solo flute (ornamented, over a soft D-A drone): the CALL; the
                D5 hangs through the question; the answer during the hand-over; 2800 warm
                strings; the answering fires = distant horn calls from the hills; D add9
"""
import numpy as np

import score as V1
from dsl import Part, m
from timeline_v2 import fb, gb, RUN_BEACONS, MONTAGE_IGN

PARTS = V1.PARTS
P = V1.P


def clone(new, old, **over):
    o = PARTS[old]
    kw = dict(kind=o.kind, bus=o.bus, pan=o.pan, width=o.width, gain_db=o.gain_db, send=o.send,
              depth=o.depth, humanize_ms=o.humanize_ms, params=dict(o.params))
    inst = over.pop("inst", o.inst)
    kw.update(over)
    PARTS[new] = Part(new, inst, **kw)
    return PARTS[new]


def N(pn, pitch, start, dur, vel=None, **kw):
    # the C3 timpani (VSCO 'Timpani3', keys C3-D3) reaches its attack ~12 ms after the stroke begins:
    # place its strokes 4 ms early so the audible hit sits on the frame like the other drums
    if pn == "c_timp" and pitch is not None and m("C3") <= m(pitch) <= m("D3"):
        start -= 0.005
    return P(pn).n(pitch, start, dur, vel, **kw)


def chords(pn, seq, overlap=0.06, legato=True, **kw):
    for i in range(len(seq) - 1):
        t0, ps = seq[i]
        t1 = seq[i + 1][0]
        if ps is None:
            continue
        for p in ps:
            P(pn).n(p, t0, t1 - t0 + overlap, legato=(legato and i > 0), **kw)


def call(pn, root, b0, rhythm=(1, 1, 2), vel=None, sync=False, legato_first=False, **kw):
    r = m(root)
    t = b0
    for i, (p, d) in enumerate(zip([r, r + 7, r + 12], rhythm)):
        N(pn, p, t, d, vel, legato=(i > 0 or legato_first), sync=(sync and i == 0), **kw)
        t += d
    return t


def setup_c():
    for s in ("vln1", "vln2", "vla", "vc", "cb", "svln"):
        clone("c_" + s, s)
    clone("c_vln1b", "vln1", pan=-0.45, gain_db=V1.FADERS["vln1"] + 0.5)
    PARTS["c_vln1"].gain_db = V1.FADERS["vln1"] + 2.5         # the violins carry C's melodies
    PARTS["c_svln"].gain_db = V1.FADERS["svln"] + 5.0
    for s in ("vln1_sp", "vln2_sp", "vla_sp", "vc_sp", "cb_sp"):
        clone("c_" + s, s)
    clone("c_vln_tr", "vln_tr")
    clone("c_vla_tr", "vla_tr")
    clone("c_vc_tr", "vc_tr")
    # horns: the section (near) and the callers on the peaks (near/far, left/right)
    clone("c_hns", "hns")
    clone("c_hns2", "hns", pan=-0.18, humanize_ms=12, gain_db=V1.FADERS["hns"] - 1.5)
    clone("c_hn34", "hns2", pan=-0.28, humanize_ms=14, gain_db=V1.FADERS["hns"] - 0.5)
    clone("c_hn_solo", "hn_solo", pan=-0.25)
    clone("c_hnL", "hn_solo", pan=-0.6, depth=0.6, send=0.5)
    clone("c_hnR", "hn_solo", pan=0.55, depth=0.62, send=0.5)
    clone("c_hnLf", "hn_solo", pan=-0.8, depth=0.92, send=0.72, gain_db=V1.FADERS["hn_solo"] - 4.0)
    clone("c_hnRf", "hn_solo", pan=0.8, depth=0.92, send=0.72, gain_db=V1.FADERS["hn_solo"] - 4.0)
    clone("c_hnCf", "hn_solo", pan=0.1, depth=1.0, send=0.8, gain_db=V1.FADERS["hn_solo"] - 7.0)
    clone("c_tpt", "tpt")
    clone("c_tbn", "tbn")
    clone("c_tuba", "tuba")
    clone("c_timp", "timp")
    clone("c_timp_roll", "timp_roll")
    clone("c_bdrum", "bdrum", gain_db=V1.FADERS["bdrum"] - 4.0)
    clone("c_swell", "swell", gain_db=V1.FADERS["swell"] - 4.0)
    clone("c_harp", "harp")
    clone("c_fl", "fl")
    clone("c_cl", "cl")
    clone("c_ob", "ob")
    clone("c_piano", "piano")


# ---------------------------------------------------------------------------
def c_first_beacon():
    k, c2, roar = fb(1318), fb(1340), fb(1360)
    far, shep, run = fb(1440), fb(1480), fb(1520)
    # the dark: basses and cellos, a low D, breathing; a distant timpani roll like far thunder
    N("c_cb", "D2", gb(16, 1.5), roar - gb(16, 1.5) + 0.06)
    N("c_vc", "D3", gb(16, 3.0), roar - gb(16, 3.0) + 0.06)
    N("c_timp_roll", "D2", gb(16, 2.0), fb(1300) - gb(16, 2.0), rel=0.6)
    P("c_timp_roll").d((gb(16, 2), 0.05), (gb(16, 4), 0.12), (gb(17, 1), 0.08), (fb(1300), 0.03))
    # 1318 the flame: Bb/D in low strings ; 1340 C/D ; crescendo into the roar
    chords("c_vc", [(k, ["F3"]), (c2, ["G3"]), (roar, None)])
    chords("c_vla", [(k, ["Bb3"]), (c2, ["C4"]), (roar, None)])
    chords("c_vln2", [(k, ["D4", "F4"]), (c2, ["E4", "G4"]), (roar, None)])
    for pn, a in (("c_cb", 0.0), ("c_vc", 0.0), ("c_vla", 0.02), ("c_vln2", 0.04)):
        P(pn).d((gb(16, 1.5), 0.05), (gb(16, 3.5), 0.13 - a), (gb(17, 1.5), 0.1), (k - 0.1, 0.12),
                (k + 0.6, 0.24 - a), (c2, 0.3), (roar - 0.12, 0.55), (roar, 0.6))
    N("c_timp_roll", "A2", c2, roar - c2 - 0.02, rel=0.05)
    P("c_timp_roll").d((c2, 0.1), (roar - 0.05, 0.62))
    # 1360 THE ROAR: timpani + the strings on D minor; a solo horn cries the CALL
    N("c_timp", "D2", roar, 2, 0.78, sync=True)
    N("c_bdrum", 60, roar, 2, 0.5, sync=True)
    H = [(roar, "D2", ["D3", "A3"], ["F4", "A4"], ["D5", "F5"], ["A5"]),
         (fb(1400), "Bb1", ["Bb2", "F3"], ["D4", "F4"], ["Bb4", "D5"], ["F5"]),
         (far, "G1", ["G2", "D3"], ["Bb3", "D4"], ["G4", "Bb4"], ["D5"]),
         (shep, "A1", ["A2", "E3"], ["C#4", "E4"], ["A4", "C#5"], ["E5"]),
         (fb(1500), "A1", ["A2", "E3"], ["C#4", "G4"], ["A4", "C#5"], ["E5"])]
    for i, (t, cb, vc, vla, v2, v1) in enumerate(H):
        t1 = H[i + 1][0] if i + 1 < len(H) else run
        lg = i > 0
        N("c_cb", cb, t, t1 - t + 0.06, legato=lg, sync=(i == 0))
        for p in vc:
            N("c_vc", p, t, t1 - t + 0.06, legato=lg)
        for p in vla:
            N("c_vla", p, t, t1 - t + 0.06, legato=lg)
        for p in v2:
            N("c_vln2", p, t, t1 - t + 0.06, legato=lg)
        for p in v1:
            N("c_vln1", p, t, t1 - t + 0.06, legato=lg)
    for pn, base in (("c_cb", 0.72), ("c_vc", 0.7), ("c_vla", 0.66), ("c_vln2", 0.64), ("c_vln1", 0.6)):
        P(pn).d((roar, base), (roar + 1.0, base - 0.12), (fb(1400), base - 0.1), (far, base - 0.08),
                (shep, base - 0.02), (run - 0.1, base + 0.2))
    # the call: a lone horn, noble, a little far away
    hn = P("c_hn_solo")
    hn.line("D4:1 A4:1 D5:2.6", roar)
    hn.d((roar, 0.66), (roar + 1, 0.72), (roar + 2, 0.8), (roar + 3.5, 0.55), (roar + 4.6, 0.3))
    # low brass under the reveal (Bb, Gm)
    chords("c_tbn", [(fb(1400), ["Bb2", "D3", "F3"]), (far, ["G2", "D3", "Bb3"]), (shep, ["A2", "C#3", "E3"]),
                     (run, None)])
    P("c_tbn").d((fb(1400), 0.38), (far, 0.42), (shep, 0.48), (run - 0.1, 0.66))
    # 1480 the shepherd answers from the far peak (right, farther): the call on A (the dominant)
    N("c_timp", "A2", shep, 2, 0.62, sync=True)
    call("c_hnRf", "A3", shep, (1, 1, 2))
    P("c_hnRf").d((shep, 0.74), (shep + 2, 0.84), (shep + 4, 0.6))
    # the ride begins: cellos 8ths from the far peak, violas 16ths from the shepherd
    t = far
    j = 0
    while t < run - 1e-6:
        prog = (t - far) / 4
        b = "G2" if t < shep else "A2"
        N("c_vc_sp", b, t, 0.5, 0.34 + 0.36 * prog + (0.08 if j % 2 == 0 else 0))
        t += 0.5
        j += 1
    acc = [1, 0, 0, 1, 0, 0, 1, 0]
    t = shep
    j = 0
    while t < run - 1e-6:
        prog = (t - shep) / 2
        N("c_vla_sp", ["A3", "C#4", "E4", "C#4"][j % 4], t, 0.25, 0.34 + 0.3 * prog + 0.12 * acc[j % 8])
        t += 0.25
        j += 1
    N("c_timp_roll", "A2", fb(1500), run - fb(1500) - 0.02, rel=0.05)
    P("c_timp_roll").d((fb(1500), 0.2), (run - 0.05, 0.75))


# ---- THE BEACON RUN: the call passed from peak to peak ----------------------
RUN_C = [  # frame, chord root/bass, chord tones, horn caller, call root, violin top
    (1520, "D2", ["D", "F", "A"], None, None, "F5"),
    (1540, "D2", ["D", "F", "A"], "c_hnL", "D4", "F5"),
    (1560, "F2", ["F", "A", "C"], None, None, "A5"),
    (1580, "G2", ["G", "Bb", "D"], "c_hnR", "G3", "Bb5"),
    (1600, "Bb2", ["Bb", "D", "F"], None, None, "D6"),
    (1620, "C3", ["C", "E", "G"], "c_hnLf", "C4", "E6"),
    (1640, "D3", ["D", "F", "A"], None, None, "D6"),
    (1660, "A2", ["A", "C#", "E"], "c_hnRf", "A3", "E6"),
]


def pcs(names, lo, hi):
    s = {m(n + "4") % 12 for n in names}
    return [p for p in range(m(lo), m(hi) + 1) if p % 12 in s]


def c_run():
    r0, r1 = fb(1520), fb(1680)
    group = [1, 0, 0, 1, 0, 0, 1, 0]        # 16ths grouped 3+3+2
    for i, (f, bass, ch, caller, root, top) in enumerate(RUN_C):
        t = fb(f)
        t1 = fb(RUN_C[i + 1][0]) if i + 1 < len(RUN_C) else r1
        # low strings + low brass: the climbing bass, marcato 8ths
        for k in range(2):
            v = 0.74 + 0.02 * i + (0.08 if k == 0 else 0)
            N("c_vc_sp", bass, t + 0.5 * k, 0.5, min(1.0, v - 0.08), sync=(k == 0))
            N("c_cb_sp", m(bass) - 12, t + 0.5 * k, 0.5, min(1.0, v - 0.08))
        N("c_cb", m(bass) - 12, t, t1 - t + 0.05, legato=i > 0)
        tu = m(bass) - 12 if m(bass) - 12 >= m("F1") else m(bass)
        N("c_tuba", tu, t, t1 - t + 0.04, legato=i > 0)
        tones = [p for p in pcs(ch, "A2", "E4") if p > m(bass)][:3]
        for p in tones:
            N("c_tbn", p, t, t1 - t + 0.04, legato=i > 0)
        # the 16th ostinato (violas + 2nd violins), accents 3+3+2
        up = pcs(ch, "A3", "A5")
        base = [p for p in up if p >= m("D4")][:4]
        pat = [base[0], base[1], base[2], base[1]]
        for q in range(4):
            tt = t + 0.25 * q
            a = group[(i * 4 + q) % 8]
            N("c_vla_sp", pat[q], tt, 0.25, min(1.0, 0.56 + 0.02 * i + 0.14 * a))
            N("c_vln2_sp", pat[q] + 12, tt, 0.25, min(1.0, 0.52 + 0.02 * i + 0.14 * a))
        # the violins' soaring top line
        N("c_vln1", top, t, t1 - t + 0.05, legato=i > 0)
        N("c_vln1b", m(top) - 12, t, t1 - t + 0.05, legato=i > 0)
        # timpani: every beacon (the downbeat too)
        tb = m(bass)
        while tb > m("A2"):
            tb -= 12
        while tb < m("D2"):
            tb += 12
        N("c_timp", tb, t, 1.0, min(1.0, 0.76 + 0.02 * i), sync=True)
        N("c_bdrum", 60, t, 1.0, 0.6 + 0.03 * i, sync=True)
        # THE CALL, on this peak (beacons 1, 3, 5, 7; the far ones between are fire only)
        if caller:
            call(caller, root, t, (0.5, 0.5, 0.9), sync=True)
    for pn, a, b in (("c_cb", 0.58, 0.78), ("c_tuba", 0.46, 0.68), ("c_tbn", 0.42, 0.72), ("c_vln1", 0.54, 0.82),
                     ("c_vln1b", 0.5, 0.78)):
        P(pn).d((r0, a), (r1 - 0.2, b))
    for pn, lv in (("c_hnL", 0.8), ("c_hnR", 0.8), ("c_hnRf", 0.78), ("c_hnLf", 0.78), ("c_hnCf", 0.8)):
        P(pn).d((r0, lv), (r1, lv))
    # the section answers at the desert: all horns, the CALL on D, broad
    for pn in ("c_hns", "c_hns2"):
        call(pn, "D4", r1, (1, 1, 2), sync=(pn == "c_hns"))
        P(pn).d((r1, 0.82), (r1 + 2, 0.86), (r1 + 3.5, 0.6))


# ---- MONTAGE: every ignition answered by a horn from another place ------------
MONT_C = [  # frame, bass, voicing (vc vc vla vla vln2 vln2), top, caller, call root
    (1680, "D2", ["D3", "A3", "D4", "F4", "A4", "D5"], "F5", None, None),
    (1700, "F1", ["F2", "C3", "C4", "F4", "A4", "C5"], "F5", None, None),
    (1760, "C2", ["C3", "G3", "E4", "G4", "C5", "E5"], "G5", "c_hnR", "C4"),
    (1810, "Bb1", ["Bb2", "F3", "D4", "F4", "Bb4", "D5"], "F5", "c_hnLf", "Bb3"),
    (1850, "G1", ["G2", "D3", "Bb3", "D4", "G4", "Bb4"], "G5", "c_hnRf", "G3"),
    (1890, "C2", ["C3", "G3", "E4", "G4", "Bb4", "C5"], "E5", "c_hnCf", "C4"),
]


def c_montage():
    ends = [fb(f) for f, *_ in MONT_C[1:]] + [fb(1920)]
    for i, (f, bass, v, top, caller, root) in enumerate(MONT_C):
        t, t1 = fb(f), ends[i]
        lg = i > 0
        N("c_cb", bass, t, t1 - t + 0.06, legato=True)
        for p in v[:2]:
            N("c_vc", p, t, t1 - t + 0.06, legato=lg)
        for p in v[2:4]:
            N("c_vla", p, t, t1 - t + 0.06, legato=lg)
        for p in v[4:]:
            N("c_vln2", p, t, t1 - t + 0.06, legato=lg)
        N("c_vln1", top, t, t1 - t + 0.06, legato=True)
        if i > 0:
            tb = m(bass)
            while tb < m("D2"):
                tb += 12
            N("c_timp", tb, t, 1.5, 0.66 + 0.02 * i, sync=True)
            if caller:
                span = t1 - t
                call(caller, root, t, (0.5, 0.5, min(1.6, span - 1.05)), sync=True)
    for pn, a, b in (("c_cb", 0.6, 0.66), ("c_vc", 0.56, 0.64), ("c_vla", 0.5, 0.58), ("c_vln2", 0.48, 0.58),
                     ("c_vln1", 0.62, 0.72)):
        P(pn).d((fb(1680) + 0.4, a), (fb(1700), a - 0.04), (fb(1900), b), (fb(1918), b - 0.1))
    # the ride goes on under it, lighter: cellos 8ths + violas 16ths (3+3+2)
    group = [1, 0, 0, 1, 0, 0, 1, 0]
    for i, (f, bass, v, top, caller, root) in enumerate(MONT_C):
        t0, t1 = fb(f), ends[i]
        tones = sorted(set(m(x) for x in v[2:]))
        pat = [tones[0], tones[1], tones[2], tones[1]]
        t = np.ceil(t0 * 4 - 1e-6) / 4
        j = int(round(t * 4))
        while t < t1 - 1e-6:
            prog = (t - fb(1680)) / 12
            a = group[j % 8]
            N("c_vla_sp", pat[j % 4], t, 0.25, 0.4 + 0.18 * prog + 0.12 * a)
            if abs(t * 2 - round(t * 2)) < 1e-6:
                N("c_vc_sp", m(v[0]), t, 0.5, 0.44 + 0.2 * prog + (0.08 if abs(t - round(t)) < 1e-6 else 0))
            t += 0.25
            j += 1
    # low brass: soft sustain; a snare-less build with the timpani into the map
    chords("c_tbn", [(fb(1700), ["A2", "C3", "F3"]), (fb(1760), ["G2", "C3", "E3"]), (fb(1810), ["Bb2", "D3", "F3"]),
                     (fb(1850), ["Bb2", "D3", "G3"]), (fb(1890), ["Bb2", "E3", "G3"]), (fb(1920), None)])
    P("c_tbn").d((fb(1700), 0.34), (fb(1918), 0.5))
    N("c_timp_roll", "C3", fb(1900), fb(1920) - fb(1900) - 0.02, rel=0.05)
    P("c_timp_roll").d((fb(1900), 0.2), (fb(1919), 0.62))


# ---- THE MAP: wonder -------------------------------------------------------------
def c_map():
    w0, w1, w2 = fb(1920), fb(2000), fb(2080)
    N("c_timp", "F2", w0, 2, 0.55, sync=True)
    N("c_cb", "F1", w0, w1 - w0 + 0.06, legato=True, sync=True)
    N("c_cb", "Bb1", w1, w2 - w1 + 0.06, legato=True)
    chords("c_vc", [(w0, ["F2", "C3"]), (w1, ["Bb2", "F3"]), (w2, None)])
    chords("c_vla", [(w0, ["A3", "C4"]), (w1, ["D4", "F4"]), (w2, None)])
    chords("c_vln2", [(w0, ["F4", "A4"]), (w1, ["A4", "C5"]), (w2, None)])
    for pn, lv in (("c_cb", 0.52), ("c_vc", 0.5), ("c_vla", 0.48), ("c_vln2", 0.46)):
        P(pn).d((w0, lv + 0.06), (w0 + 1, lv), (w1, lv + 0.04), (w2 - 1, lv - 0.1), (w2, lv - 0.14))
    # the theme in the violins: the CALL on F, then the answer over Bb with its raised fourth
    for pn in ("c_vln1", "c_vln1b", "c_svln"):
        P(pn).line("F4:1 C5:1 F5:2", w0)
        P(pn).line("F5:1 E5:.5 D5:.5 A4:2.2", w1)
    for nt in P("c_svln").notes:
        if nt.start >= w0 - 1e-6:
            nt.pitch += 12
    P("c_vln1").d((w0, 0.66), (w0 + 2, 0.74), (w1, 0.72), (w1 + 2, 0.62), (w2, 0.34))
    P("c_vln1b").d((w0, 0.62), (w0 + 2, 0.7), (w1, 0.66), (w2, 0.32))
    P("c_svln").d((w0, 0.46), (w0 + 2, 0.54), (w1, 0.52), (w1 + 2, 0.44), (w2, 0.22))
    # horns: soft sustained harmony (a choral colour without a choir)
    chords("c_hns", [(w0, ["C4", "F4"]), (w1, ["D4", "F4"]), (w2, None)], legato=False)
    P("c_hns").d((w0, 0.36), (w0 + 2, 0.42), (w2 - 1, 0.26))
    # harp: arpeggios (F add9, then Bb lydian), gently, not a sweep
    fpat = ["F2", "C3", "F3", "A3", "C4", "G4", "A4", "C5"]
    bpat = ["Bb1", "F2", "Bb2", "D3", "F3", "C4", "D4", "F4"]
    t = w0
    j = 0
    while t < w2 - 1e-6:
        pat = fpat if t < w1 else bpat
        edge = w1 if t < w1 else w2
        N("c_harp", pat[j % 8], t, min(2.0, edge - t + 0.05), 0.27 - 0.03 * ((j % 8) > 4), kind="pno", rel=0.6)
        t += 0.25
        j += 1
    # an oboe answers the violins' answer (the free peoples: one more voice)
    P("c_ob").line("A5:1 G5:1 F5:2", w1 + 2)
    P("c_ob").d((w1 + 2, 0.44), (w1 + 3, 0.5), (w2, 0.24))


# ---- THE COUNCIL ------------------------------------------------------------------
def c_council():
    a0, lock, s0, s1, bre, dawn = fb(2080), fb(2160), fb(2320), fb(2360), fb(2400) - 0.33, fb(2400)
    # hush: D minor, basses + cellos + violas pp; the clarinet intones the CALL
    N("c_cb", "D2", a0, s0 - a0 + 0.06, legato=True)
    chords("c_vc", [(a0, ["D3", "A3"]), (fb(2230), ["D3", "F3"]), (fb(2290), ["D3", "G3"]), (s0, None)])
    chords("c_vla", [(a0, ["F4"]), (fb(2230), ["F4"]), (fb(2290), ["G4"]), (s0, None)])
    P("c_cl").line("D4:1.5 A4:1.5 D5:3", a0 + 0.5)
    P("c_cl").d((a0, 0.32), (a0 + 3, 0.4), (lock + 2, 0.24))
    # 2160 the Ring in the fire: a low-brass chorale, pianissimo; chords move on off-beats
    ch = [(lock, ["D3", "F3", "A3"], "D2"), (fb(2230), ["D3", "F3", "Bb3"], "D2"),
          (fb(2290), ["D3", "G3", "Bb3"], "D2"), (s0, ["D3", "F3", "Bb3"], "Bb1"),
          (s1, ["D3", "E3", "A3"], "A1"), (fb(2380), ["C#3", "E3", "A3"], "A1")]
    for i, (t, ps, tu) in enumerate(ch):
        t1 = ch[i + 1][0] if i + 1 < len(ch) else bre
        for p in ps:
            N("c_tbn", p, t, t1 - t + (0.04 if t1 < bre else 0), legato=i > 0)
        N("c_tuba", tu, t, t1 - t + (0.04 if t1 < bre else 0), legato=i > 0)
    P("c_tbn").d((lock, 0.2), (fb(2240), 0.3), (s0, 0.5), (s1, 0.66), (bre - 0.02, 0.92))
    P("c_tuba").d((lock, 0.2), (s0, 0.5), (bre - 0.02, 0.9))
    # strings: the harmony + the build (tremolo from the vows' end, never on a vow)
    chords("c_vln2", [(lock, ["A4", "D5"]), (fb(2230), ["Bb4", "D5"]), (fb(2290), ["Bb4", "D5"]),
                      (s0, ["Bb4", "F5"]), (s1, ["A4", "D5"]), (fb(2380), ["A4", "C#5"]), (bre, None)], overlap=0.0)
    chords("c_vc", [(s0, ["Bb2", "F3"]), (s1, ["A2", "E3"]), (bre, None)], overlap=0.0)
    chords("c_vla", [(s0, ["D4", "F4"]), (s1, ["D4", "E4"]), (fb(2380), ["C#4", "E4"]), (bre, None)], overlap=0.0)
    N("c_cb", "Bb1", s0, s1 - s0 + 0.04, legato=True)
    N("c_cb", "A1", s1, bre - s1, legato=True)
    for pn, lo, hi in (("c_cb", 0.3, 0.95), ("c_vc", 0.28, 0.92), ("c_vla", 0.26, 0.9), ("c_vln2", 0.24, 0.9)):
        P(pn).d((a0 + 0.3, lo), (lock, lo + 0.02), (fb(2250), lo + 0.12), (s0, lo + (hi - lo) * 0.5),
                (s1, lo + (hi - lo) * 0.8), (bre - 0.02, hi))
    # the violins' line rises slowly over the vows (changes on off-beats), then the ring sweep
    seq = [(lock, "A4"), (fb(2230), "Bb4"), (fb(2290), "D5"), (s0, "F5"), (s1, "E5"), (fb(2380), "E5")]
    for j, (tt, p) in enumerate(seq):
        t1 = seq[j + 1][0] if j + 1 < len(seq) else bre
        N("c_vln1", p, tt, t1 - tt + (0.05 if t1 < bre else 0), legato=j > 0)
    P("c_vln1").d((lock, 0.3), (fb(2250), 0.4), (s0, 0.62), (s1, 0.8), (bre - 0.02, 0.98))
    # the ring sweep 2320-2380: a swell - the violins' second desk holding the chord, from nothing
    chords("c_vln1b", [(s0, ["F5", "Bb5"]), (s1, ["E5", "A5"]), (fb(2380), ["E5", "A5"]), (bre, None)],
           overlap=0.0)
    P("c_vln1b").d((s0, 0.08), (s1, 0.5), (fb(2380), 0.78), (bre - 0.02, 0.9))
    # horns join for the build; the tremolo and the timpani carry the hearth flare into the breath
    chords("c_hns", [(s0, ["D4", "F4"]), (s1, ["D4", "E4"]), (fb(2380), ["C#4", "E4"]), (bre, None)], overlap=0.0)
    P("c_hns").d((s0, 0.5), (bre - 0.02, 0.92))
    chords("c_vla_tr", [(fb(2300), ["Bb3", "D4"]), (s0, ["Bb3", "D4"]), (s1, ["A3", "D4"]), (fb(2380), ["A3", "E4"]),
                        (bre, None)], overlap=0.0)
    P("c_vla_tr").d((fb(2300), 0.1), (bre - 0.02, 0.9))
    N("c_timp_roll", "D2", fb(2300), s0 - fb(2300) + 0.02, rel=0.02)
    N("c_timp_roll", "Bb2", s0, s1 - s0 + 0.02, rel=0.02, legato=True)
    N("c_timp_roll", "A2", s1, bre - s1, rel=0.02, legato=True)
    P("c_timp_roll").d((fb(2300), 0.08), (s1, 0.5), (bre - 0.02, 0.92))
    N("c_swell", 60, bre, 1, 0.55)          # suspended cymbal swell peaking at the breath (no crash)


# ---- DAWN in the east -----------------------------------------------------------------
def c_dawn():
    d0 = fb(2400)
    b32, b33, dz, end = gb(32), gb(33), fb(2624), gb(34, 3)
    # harmony: D | G - D/F# | Em7 - A7sus4 - A | (the coda) ...
    H = [(d0, "D2", ["D3", "A3"], ["F#3", "A3"], ["D4", "F#4"]),
         (d0 + 2, "D2", ["D3", "A3"], ["F#3", "A3"], ["D4", "F#4"]),
         (b32, "B1", ["B2", "F#3"], ["B3", "D4"], ["D4", "F#4"]),
         (b32 + 2, "G1", ["G2", "D3"], ["B3", "D4"], ["D4", "B4"]),
         (b32 + 3, "A1", ["A2", "E3"], ["C#4", "E4"], ["E4", "A4"]),
         (b33, "E1", ["E2", "B2"], ["G3", "B3"], ["D4", "G4"]),
         (b33 + 1, "A1", ["A2", "E3"], ["D4", "E4"], ["G4", "A4"]),
         (b33 + 2, "D2", ["D3", "A3"], ["F#3", "A3"], ["D4", "F#4"])]
    for i, (t, cb, vc, vla, v2) in enumerate(H):
        t1 = H[i + 1][0] if i + 1 < len(H) else end
        N("c_cb", cb, t, t1 - t + 0.06, legato=i > 0, sync=(i == 0))
        for p in vc:
            N("c_vc", p, t, t1 - t + 0.06, legato=i > 0, sync=(i == 0))
        for p in vla:
            N("c_vla", p, t, t1 - t + 0.06, legato=i > 0, sync=(i == 0))
        for p in v2:
            N("c_vln2", p, t, t1 - t + 0.06, legato=i > 0, sync=(i == 0))
    for pn, top in (("c_cb", 0.9), ("c_vc", 0.78), ("c_vla", 0.74), ("c_vln2", 0.78)):
        P(pn).d((d0, top), (b32, top - 0.06), (b33, top - 0.1), (dz, top - 0.3), (end, 0.12))
    N("c_timp_roll", "D2", d0, b32 + 1 - d0, rel=0.4)          # the bloom's floor: a roll, not a stroke
    P("c_timp_roll").d((d0, 0.2), (d0 + 1.2, 0.56), (d0 + 3, 0.5), (b32, 0.36), (b32 + 1, 0.16))
    # THE THEME, soaring (violins in octaves): the call, the answer, and a new rising
    # consequent for the eagles crossing the light
    mel = "D4:1 A4:1 D5:2 | D5:1 C#5:.5 B4:.5 F#4:1 A4:1 | B4:1 A4:1 F#4:1 D4:1.6"
    P("c_vln1").line(mel, d0)
    P("c_vln1b").line(mel, d0)
    P("c_svln").line(mel, d0)
    for pn in ("c_vln1", "c_svln"):          # violins I (and the solo violin) an octave above the seconds
        for nt in P(pn).notes:
            if d0 - 1e-6 <= nt.start < end:
                nt.pitch += 12
    # one long line: the bloom, the call, the answer, and a continuation that settles home - the
    # dynamics only ever relax after the answer (nothing marks the eagles' entrance)
    P("c_vln1").d((d0, 0.66), (d0 + 1.2, 0.9), (b32, 0.88), (b33, 0.8), (b33 + 2, 0.7), (dz, 0.56), (end, 0.2))
    P("c_vln1b").d((d0, 0.62), (d0 + 1.2, 0.86), (b32, 0.84), (b33, 0.76), (b33 + 2, 0.66), (dz, 0.5), (end, 0.2))
    P("c_svln").d((d0, 0.34), (d0 + 1.2, 0.5), (b32, 0.5), (b33, 0.46), (b33 + 2, 0.4), (dz, 0.32), (end, 0.12))
    # horns: noble counterpoint (not the tune)
    for pn in ("c_hns", "c_hns2", "c_hn34"):
        P(pn).line("A3:2 D4:2 | D4:1 E4:1 D4:1 C#4:1 | B3:1 A3:.5 G3:.5 A3:2 | A3:3", d0)
        P(pn).d((d0, 0.9), (b32, 0.86), (b33, 0.84), (dz, 0.54), (end, 0.2))
    # low brass: warm chords, contrary to the bass
    chords("c_tbn", [(d0, ["D3", "F#3", "A3"]), (b32, ["D3", "F#3", "B3"]), (b32 + 2, ["D3", "G3", "B3"]),
                     (b32 + 3, ["C#3", "E3", "A3"]), (b33, ["E3", "G3", "B3"]), (b33 + 1, ["D3", "E3", "A3"]),
                     (b33 + 2, ["D3", "F#3", "A3"]), (end - 1, None)], sync=False)
    chords("c_tuba", [(d0, ["D2"]), (b32, ["B1"]), (b32 + 2, ["G1"]), (b32 + 3, ["A1"]), (b33, ["E2"]),
                      (b33 + 1, ["A1"]), (b33 + 2, ["D2"]), (end - 1, None)])
    for pn in ("c_tbn", "c_tuba"):
        P(pn).d((d0, 0.92), (b32, 0.8), (b33, 0.74), (dz, 0.44), (end - 1, 0.1))
    # the light: violin tremolo high, and the harp's broad arpeggios on the new chords
    chords("c_vln_tr", [(d0, ["A5", "D6"]), (b32, ["F#5", "B5"]), (b32 + 2, ["B5", "D6"]), (b32 + 3, None)])
    P("c_vln_tr").d((d0, 0.2), (d0 + 1, 0.36), (b33, 0.2))
    for t, dur, ps in ((d0, 3.8, ["D2", "A2", "D3", "F#3", "A3", "D4", "F#4", "A4"]),
                       (b32, 1.9, ["B1", "F#2", "B2", "D3", "F#3", "B3", "D4"]),
                       (b33 + 2, 3.0, ["D2", "A2", "D3", "F#3", "A3", "D4"])):
        for j, p in enumerate(ps):
            N("c_harp", p, t + 0.07 * j, dur - 0.07 * j, 0.42, kind="pno", rel=0.8)


# ---- CODA: a Celtic-leaning flute; answering horns from the hills --------------------
def c_coda():
    fl = P("c_fl")
    # the intro's flute, plainly (the film's own voice): the CALL; the D5 hangs through the question;
    # the answer during the hand-over, arriving on F#4 as the torch changes hands.  No ornaments, no drone.
    fl.n("D4", fb(2652), 1.0, sync=True)
    fl.n("A4", fb(2672), 1.8, legato=True)
    fl.n("D5", fb(2708), fb(2746) - fb(2708) + 0.02, legato=True)       # hangs through the question
    fl.n("C#5", fb(2746), fb(2762) - fb(2746) + 0.02, legato=True)
    fl.n("B4", fb(2762), fb(2780) - fb(2762) + 0.02, legato=True)
    fl.n("F#4", fb(2780), 1.7, legato=True)
    fl.d((fb(2652), 0.3), (fb(2672), 0.36), (fb(2708), 0.38), (fb(2730), 0.34), (fb(2746), 0.3), (fb(2780), 0.3),
         (fb(2810), 0.16))
    # 2800: the child's beacon - warm strings (G/D), then the D add9 at 2880
    t0, tf, endc = fb(2800), fb(2880), fb(2946)
    chords("c_cb", [(t0, ["D2"]), (endc, None)])
    chords("c_vc", [(t0, ["D3", "G3"]), (tf, ["D3", "A3"]), (endc, None)])
    chords("c_vla", [(t0, ["B3", "D4"]), (tf, ["F#4"]), (endc, None)])
    chords("c_vln2", [(t0, ["B4", "D5"]), (tf, ["A4", "E5"]), (endc, None)])
    for pn, pk in (("c_cb", 0.38), ("c_vc", 0.36), ("c_vla", 0.34), ("c_vln2", 0.32)):
        P(pn).d((t0, 0.12), (t0 + 1.2, pk), (tf - 0.5, pk - 0.08), (tf + 0.6, pk - 0.04), (endc - 1, 0.06),
                (endc, 0.03))
    # the answering fires: distant horn calls from the hills (pianissimo, far, left and right)
    for f, pn, root in ((2806, "c_hnLf", "D4"), (2822, "c_hnRf", "A3"), (2838, "c_hnCf", "D4"),
                        (2854, "c_hnLf", "A3")):
        call(pn, root, fb(f), (0.5, 0.5, 1.4))
    for pn in ("c_hnLf", "c_hnRf", "c_hnCf"):
        P(pn).d((fb(2800), 0.3), (fb(2870), 0.3), (fb(2880), 0.24))
    # the solo violin rises out of the last call and holds the ninth over the final chord
    sv = P("c_svln")
    sv.n("A5", tf, endc - tf - 0.4, sync=False)
    sv.n("E6", tf + 2, endc - tf - 2.4, legato=True)
    sv.d((tf, 0.12), (tf + 1.5, 0.26), (tf + 2.5, 0.24), (endc - 1, 0.05))
    fl.n("D5", tf + 0.5, endc - tf - 1.0)
    fl.d((tf + 0.5, 0.14), (tf + 2, 0.22), (endc - 1, 0.04))


def build_c():
    import score_v2
    setup_c()
    c_first_beacon()
    c_run()
    c_montage()
    c_map()
    c_council()
    c_dawn()
    # phrasing on the long melodic notes
    score_v2.breathe("c_hn_solo", fb(1360), fb(1470), depth=0.14, min_dur=1.5)
    score_v2.breathe("c_hnRf", fb(1480), fb(1540), depth=0.12, min_dur=1.5)
    for pn in ("c_vln1", "c_vln1b", "c_svln"):
        score_v2.breathe(pn, fb(1920), fb(2080), depth=0.12, min_dur=1.5)
        score_v2.breathe(pn, fb(2400), fb(2660), depth=0.1, min_dur=1.5)
    for pn in ("c_hns", "c_hns2", "c_hn34"):
        score_v2.breathe(pn, fb(1680), fb(1760), depth=0.12, min_dur=1.5)
        score_v2.breathe(pn, fb(2400), fb(2660), depth=0.1, min_dur=1.5)
    score_v2.breathe("c_cl", fb(2080), fb(2200), depth=0.12, min_dur=1.2)
    d0, dz = fb(2400), fb(2660)
    for src in ("c_vln1", "c_vln1b", "c_vln2", "c_vla", "c_vc", "c_cb"):
        score_v2.layer(src, src + "_L2", d0, dz, gain_db=-1.0, pan_shift=0.1 if "vln" in src else -0.1)
    c_coda()


SYNC_C = [(1360, "beacon ROARS (timpani)", "c_timp", 0.05, "hit"),
          (1360, "the call (solo horn)", "c_hn_solo", 0.08, "soft"),
          (1480, "the shepherd answers (far horn)", "c_timp", 0.05, "hit"),
          (1520, "cut to THE BEACON RUN (timp)", "c_timp", 0.05, "hit")] + \
         [(f, f"run beacon {i + 1} (timp)", "c_timp", 0.05, "hit") for i, f in enumerate(RUN_BEACONS)] + \
         [(f, f"{nm} ignition (timp)", "c_timp", 0.05, "hit")
          for f, nm in zip(MONTAGE_IGN, ["desert", "ice", "karst", "city", "sea"])] + \
         [(f, f"run beacon {i + 1} (sfx whoomp)", "sfx", 0.04, "hit") for i, f in enumerate(RUN_BEACONS)] + \
         [(1920, "the map (timp)", "c_timp", 0.05, "hit"),
          (2400, "DAWN in the east (final, after the breath)", "final", 0.05, "hit"),
          (2400, "DAWN: the bloom (violins)", "c_vln1", 0.08, "soft"),
          (2652, "coda: the plain flute (D4)", "c_fl", 0.1, "soft"),
          (2756, "flint echo 1 (the hand-over)", "sfx", 0.05, "hit"),
          (2768, "flint echo 2", "sfx", 0.05, "hit"),
          (2780, "flint echo 3", "sfx", 0.05, "hit"),
          (2800, "beacon catch (sfx)", "sfx", 0.06, "hit")]
