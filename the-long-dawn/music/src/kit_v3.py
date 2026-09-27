"""THE LONG DAWN v3 - the shared MOTIF KIT and the score container.

Every v3 score (A, B, C and the fallback masters) is built from these pieces, on the one grid
(timeline_v3).  Times are global beats; pitches are MIDI numbers or names ('D4').

THE KIT (BIBLE_V3 section 9 + REVISION 1 / review/v3_redteam.md section 5.4)
  CALL            root - fifth - octave, rising (D3 A3 D4 ...).  Whoever lights a fire gets the CALL.
  ANSWER          the theme's falling second phrase: D C# B F# (major) / D C Bb F (minor).  Only an
                  answer earns the ANSWER.
  HOME            the phrase that comes home: B A F# D (major) / Bb A G F G E D (minor, v1 A7-A8).
                  The whole theme (CALL + ANSWER + HOME) is withheld until each cut's dawn.
  CORRUPTED CALL  root - tritone - octave (D Ab D): the race in A.
  RING            the corrupted call as a closed loop (D Ab D' C' Ab | D ...) that never cadences;
                  ring(cadence=True) heals it once: C' -> C#', Ab -> A, and it resolves into D.
  WATCH TONE      a low D held and re-bowed, never stopping (A: from the roar to the last chord).
  THINKING CYCLE  the kindling's glass FM cycles (7 notes on a 16th grid); rule 3: its step is its
                  pace (16ths while raced, quarter notes locked to the bar once carried).
  LYDIAN BLOOM    Bbmaj9(#11)/D, the IGNITION chord, breathing once a bar.
  SILENCE PIANO   D ... A ... D, then the low fifth D2+A2 ("together").
  GROUND          B's chaconne ground: a tetrachord that steps down (or turns and climbs) a degree
                  each variation.
  WALK            A's walking ostinato: pizzicato basses and a soft drum on 1 and 3 (the feet); the
                  harmony moves only when told (a watch-fire passed).
  BREATHS         200 ms before a race entrance, the suck before an IMPACT, 275 ms before a sunrise.

Score rules (red team 5.4), enforced by helpers or checked by analyze_v3:
  sparse beats dense; double every exposed melodic line or sit it on a pad (double(), pad());
  no ensemble figuration faster than 16ths and no percussion faster than 16ths at 72 BPM
  (check_rates()); warm, never loud, dawns (the level map).
"""
from collections import OrderedDict

import numpy as np

from dsl import Part, m, name as nname, dyn_at
from timeline_v3 import gb, fb, tb, bt

# ---------------------------------------------------------------------------
# seating: every part is created from a seat (v1 seating and v1 fader calibration)
# pan -1 L .. +1 R ; depth 0 close .. 1 far ; send = hall
# ---------------------------------------------------------------------------
_S = dict(bus="strings")
_B = dict(bus="brass")
_W = dict(bus="winds")
_K = dict(bus="keys")
_PC = dict(bus="perc", humanize_ms=0)
SEATS = {
    # strings (sections; *_q = the quiet VSCO recordings)
    "vln1": dict(inst="vln", pan=-0.62, width=0.55, depth=0.25, send=0.26, gain_db=2.1, **_S),
    "vln2": dict(inst="vln", pan=-0.30, width=0.50, depth=0.30, send=0.27, gain_db=-1.1, **_S),
    "vla": dict(inst="vla", pan=0.05, width=0.50, depth=0.30, send=0.27, gain_db=-0.2, **_S),
    "vc": dict(inst="vc", pan=0.35, width=0.50, depth=0.30, send=0.26, gain_db=0.3, **_S),
    "cb": dict(inst="cb", pan=0.58, width=0.40, depth=0.35, send=0.25, gain_db=2.0, **_S),
    "vln1_q": dict(inst="vln_q", pan=-0.62, width=0.55, depth=0.3, send=0.3, gain_db=2.1, **_S),
    "vln2_q": dict(inst="vln_q", pan=-0.30, width=0.50, depth=0.32, send=0.3, gain_db=-1.1, **_S),
    "vla_q": dict(inst="vla_q", pan=0.05, width=0.50, depth=0.32, send=0.3, gain_db=-0.2, **_S),
    "vc_q": dict(inst="vc_q", pan=0.35, width=0.50, depth=0.32, send=0.3, gain_db=0.3, **_S),
    "cb_q": dict(inst="cb_q", pan=0.58, width=0.40, depth=0.36, send=0.28, gain_db=2.0, **_S),
    "vln_trem": dict(inst="vln_trem", pan=-0.45, width=0.7, depth=0.3, send=0.3, gain_db=-0.6, **_S),
    "vla_trem": dict(inst="vla_trem", pan=0.05, width=0.6, depth=0.3, send=0.3, gain_db=-1.0, **_S),
    "vc_trem": dict(inst="vc_trem", pan=0.35, width=0.6, depth=0.3, send=0.3, gain_db=5.8, **_S),
    "cb_pizz": dict(inst="cb_pizz", pan=0.58, width=0.4, depth=0.35, send=0.3, humanize_ms=6, **_S),
    "vc_pizz": dict(inst="vc_pizz", pan=0.35, width=0.5, depth=0.3, send=0.3, humanize_ms=6, **_S),
    "svln": dict(inst="svln", pan=-0.40, width=0.4, depth=0.3, send=0.3, gain_db=1.2, humanize_ms=10, **_S),
    # B: HER CELLO.  VSCO-2 CE has no solo cello, so her voice is the quiet cello section, close and narrow
    # (a section cannot sound like a MIDI solo: the red team's "double or pad"), with the solo contrabass in
    # its thumb-position register (D3-D4) inside it as a single player's core (out/v3/tests/voice_*.wav)
    "voice": dict(inst="vc_q", pan=-0.05, width=0.3, depth=0.2, send=0.3, gain_db=0.3, humanize_ms=10, **_S),
    "voice_core": dict(inst="cb", pan=-0.05, width=0.25, depth=0.2, send=0.3, gain_db=-1.0, humanize_ms=8, **_S),
    # brass: near, far, farther (the answers)
    "hn": dict(inst="horn", pan=-0.30, width=0.4, depth=0.55, send=0.45, gain_db=3.4, humanize_ms=10, **_B),
    "hn2": dict(inst="horn", pan=-0.18, width=0.6, depth=0.6, send=0.45, gain_db=-0.8, humanize_ms=12, **_B),
    "hn_far": dict(inst="horn", pan=0.62, width=0.3, depth=0.92, send=0.75, gain_db=0.4, humanize_ms=12, **_B),
    "hn_farther": dict(inst="horn", pan=-0.72, width=0.25, depth=1.0, send=0.86, gain_db=-2.5, humanize_ms=14, **_B),
    "tbn": dict(inst="trombone", pan=0.42, width=0.6, depth=0.65, send=0.42, gain_db=-2.8, **_B),
    "tuba": dict(inst="tuba", pan=0.55, width=0.4, depth=0.65, send=0.4, gain_db=-0.4, **_B),
    # winds (C's book: cor anglais = the oboe's lowest register, doubled/padded; clarinet)
    "ob": dict(inst="oboe", pan=0.08, width=0.35, depth=0.45, send=0.4, humanize_ms=10, **_W),
    "cl": dict(inst="clarinet", pan=0.12, width=0.35, depth=0.45, send=0.4, gain_db=5.2, humanize_ms=10, **_W),
    "fl": dict(inst="flute", pan=-0.08, width=0.35, depth=0.3, send=0.42, gain_db=9.6, humanize_ms=10, **_W),
    "bsn": dict(inst="bassoon", pan=0.2, width=0.35, depth=0.45, send=0.4, humanize_ms=10, **_W),
    # keys / perc
    "piano": dict(inst="piano", pan=0.0, width=0.8, depth=0.35, send=0.5, gain_db=11.3, humanize_ms=0, **_K),
    "harp": dict(inst="harp", pan=-0.45, width=0.6, depth=0.45, send=0.4, gain_db=2.1, **_K),
    "timp": dict(inst="timp", pan=-0.05, width=0.6, depth=0.7, send=0.42, gain_db=6.7, **_PC),
    "timp_roll": dict(inst="timp_roll", pan=-0.05, width=0.6, depth=0.7, send=0.42, gain_db=1.0, **_PC),
    # synth (score side)
    "glass": dict(inst="fmbell", kind="synth", bus="synth", pan=0.0, width=1.0, depth=0.45, send=0.45,
                  gain_db=-10.1, humanize_ms=0, params=dict(ratio=3.5, index=1.6, decay=1.1)),
    "warm": dict(inst="fmwarm", kind="synth", bus="synth", pan=0.3, width=1.0, depth=0.45, send=0.5,
                 gain_db=-5.0, humanize_ms=0, params=dict(ratio=1.0, index=0.7, decay=2.4, lp=4800.0)),
    "sub": dict(inst="subpulse", kind="synth", bus="synth", pan=0.0, width=0.0, depth=0.0, send=0.0,
                gain_db=-4.0, humanize_ms=0),
    "harmonic": dict(inst="harmonic", kind="synth", bus="strings", pan=-0.35, width=0.5, depth=0.5, send=0.55,
                     gain_db=-6.0, humanize_ms=0),
}


class Score:
    """A cut's score: parts + breaths + mix EQ + bus groups + the analysis contract
    (level map, centroid arc, sync table)."""

    def __init__(self, cut, bm):
        self.cut = cut
        self.bm = bm
        self.parts = OrderedDict()
        self.breaths = bm.breath_beats()
        self.eq = {}          # name -> (hp_hz|None, lp_hz|None[, air_db, air_hz])
        self.groups = {}      # name prefix -> dict(ceiling_db, release)
        self.push = []
        self.levels = []      # [(section_id, lo_lufs, hi_lufs)]  short-term LUFS (3 s) after the master
        self.centroid = []    # [(section_id, lo_hz, hi_hz)]     median spectral centroid of the score
        self.sync = []        # [(t_s, label, part, window_s, kind 'hit'|'soft')]

    def add(self, name, seat=None, **over):
        cfg = dict(SEATS[seat or name])
        cfg.update(over)
        inst = cfg.pop("inst")
        self.parts[name] = Part(name, inst, **cfg)
        return self.parts[name]

    def P(self, name):
        if name not in self.parts:
            self.add(name)
        return self.parts[name]

    def used(self):
        return OrderedDict((k, v) for k, v in self.parts.items() if v.notes)


# ---------------------------------------------------------------------------
# modes and motif data
# ---------------------------------------------------------------------------
MODES = {"minor": (0, 2, 3, 5, 7, 8, 10), "dorian": (0, 2, 3, 5, 7, 9, 10), "major": (0, 2, 4, 5, 7, 9, 11),
         "lydian": (0, 2, 4, 6, 7, 9, 11), "mixolydian": (0, 2, 4, 5, 7, 9, 10)}
CALL = (0, 7, 12)
CORRUPT = (0, 6, 12)
ANSWER = {"major": (0, -1, -3, -8), "minor": (0, -2, -4, -9)}          # from the top note (D5 C#5 B4 F#4)
ANSWER_TAIL = -5                                                      # ... A4 (the optional fifth after it)
HOME = {"major": ((9, 7, 4, 0), (1, 1, 1, 2)),                        # B A F# D  (from the root)
        "minor": ((8, 7, 5, 3, 5, 2, 0), (1, .5, .5, 1, 1, 2, 2))}    # Bb A G F G | E D (v1 A7-A8)
RING = (0, 6, 12, 10, 6)                                              # D Ab D' C' Ab -> (D) ...
RING_DUR = (1, 1, 1, .5, .5)
RING_HEALED = (0, 6, 12, 11, 7)                                       # D Ab D' C#' A -> D : cadences once
# the kindling's thinking cycles (v1 score.py) and the v2 lock (8 notes = one bar of 8ths / 2 bars of 4ths)
CYC5 = ("D5", "E5", "G5", "C6", "D6", "G5", "C6")
CYC6A = ("D5", "E5", "A5", "D6", "E6", "A5", "F5")
CYC7 = ("D6", "A5", "E6", "F5", "C6", "Bb5", "E5")
CYC8 = ("E5", "Bb5", "C#6", "G5", "E6", "Bb5", "A5")
LOCK8 = (0, 2, 7, 12, 14, 7, 3, 2)
LOCK4 = (0, 7, 12, 14)
# the IGNITION chord, Bbmaj9(#11)/D, v1 voicing
BLOOM = {"cb": ["D2"], "vc": ["Bb2"], "vla": ["F3", "C4"], "vln2": ["D4", "E4"], "vln1": ["A4", "E5"]}


def degree(root, mode, k):
    """the k-th scale degree (0-based, may be negative or > 6) of `mode` on `root`"""
    r = m(root)
    sc = MODES[mode]
    o, i = divmod(k, 7)
    return r + 12 * o + sc[i]


# ---------------------------------------------------------------------------
# the motifs
# ---------------------------------------------------------------------------
def _line(S, pn, pitches, durs, b0, vel=None, legato=True, sync_first=False, gap=0.0, **kw):
    t = b0
    for i, (p, d) in enumerate(zip(pitches, durs)):
        if p is not None:
            S.P(pn).n(p, t, d - gap, vel, legato=(legato and i > 0 and pitches[i - 1] is not None),
                      sync=(sync_first and i == 0), **kw)
        t += d
    return t


def call(S, pn, root, b0, rhythm=(1, 1, 2), vel=None, last=None, **kw):
    """THE CALL: root - fifth - octave.  `last` overrides the held length of the octave."""
    r = m(root)
    durs = list(rhythm)
    if last is not None:
        durs[-1] = last
    _line(S, pn, [r + x for x in CALL], durs, b0, vel, **kw)
    return b0 + sum(rhythm)


def corrupted_call(S, pn, root, b0, rhythm=(1, 1, 2), vel=None, **kw):
    r = m(root)
    return _line(S, pn, [r + x for x in CORRUPT], rhythm, b0, vel, **kw)


def answer(S, pn, top, b0, mode="major", rhythm=(1, .5, .5, 2), tail=None, vel=None, **kw):
    """THE ANSWER from its top note (D5 C#5 B4 F#4 [A4]).  tail = length of the optional A."""
    t = m(top)
    ps = [t + x for x in ANSWER[mode]]
    ds = list(rhythm)
    if tail:
        ps.append(t + ANSWER_TAIL)
        ds.append(tail)
    return _line(S, pn, ps, ds, b0, vel, **kw)


def home(S, pn, root, b0, mode="major", rhythm=None, vel=None, last=None, **kw):
    """THE HOME phrase from the tonic an octave below its start (B A F# D over root D)."""
    iv, rh = HOME[mode]
    ds = list(rhythm or rh)
    if last is not None:
        ds[-1] = last
    r = m(root)
    return _line(S, pn, [r + x for x in iv], ds, b0, vel, **kw)


def theme(S, pn, root, b0, mode="major", broad=False, vel=None, **kw):
    """the whole theme (only at a cut's dawn): CALL, ANSWER, HOME.  broad = augmented x2"""
    k = 2 if broad else 1
    r = m(root)
    t = call(S, pn, r, b0, rhythm=(k, k, 2 * k), vel=vel, **kw)
    t = answer(S, pn, r + 12, t, mode, rhythm=(k, .5 * k, .5 * k, 2 * k), vel=vel, **kw)
    return home(S, pn, r, t, mode, rhythm=tuple(x * k for x in HOME[mode][1]), vel=vel, **kw)


def ring(S, pn, root, b0, cycles=1, unit=1.0, vel=None, cadence=False, **kw):
    """THE RING (C): the corrupted call closed into a loop; it never cadences unless cadence=True,
    when its last cycle heals (C -> C#, Ab -> A) and resolves into D (returned end includes the D)."""
    r = m(root)
    t = b0
    for c in range(cycles):
        heal = cadence and c == cycles - 1
        iv = RING_HEALED if heal else RING
        t = _line(S, pn, [r + x for x in iv], [d * unit for d in RING_DUR], t, vel, **kw)
    if cadence:
        S.P(pn).n(r, t, 4 * unit, vel, legato=True, **kw)
        t += 4 * unit
    return t


def held(S, pn, pitch, b0, b1, bow=8.0, overlap=0.12, vel=None, sync_first=False, **kw):
    """a held note re-bowed every `bow` beats (legato re-attacks, never a gap): the drone,
    the watch tone, her last note at dusk"""
    t = b0
    first = True
    while t < b1 - 1e-6:
        d = min(bow, b1 - t)
        if b1 - (t + d) < 1.0:              # never leave a stub shorter than a beat
            d = b1 - t
        S.P(pn).n(pitch, t, d + (overlap if t + d < b1 - 1e-6 else 0), vel, legato=not first,
                  sync=(sync_first and first), **kw)
        first = False
        t += d
    return b1


watch_tone = held
drone = held


def thinking(S, pn, pattern, b0, b1, step=0.25, v0=0.25, v1=None, pan_amp=0.45, sync_first=True, accent=0.0,
             **kw):
    """THE THINKING CYCLE: a pitch cycle on a grid (step 0.25 = 16ths while raced; 1.0 = quarter
    notes locked to the bar once carried).  Rule: step >= 0.25 (no figuration faster than 16ths)."""
    assert step >= 0.25 - 1e-9, "no ensemble figuration faster than sixteenths at 72"
    v1 = v0 if v1 is None else v1
    ps = [m(p) for p in pattern]
    t, i = b0, 0
    while t < b1 - 1e-6:
        u = (t - b0) / max(1e-6, b1 - b0)
        v = v0 + (v1 - v0) * u + (accent if i % len(ps) == 0 else 0.0)
        S.P(pn).n(ps[i % len(ps)], t, step, v, pan=pan_amp * np.sin(i * 0.9), sync=(sync_first and i == 0), **kw)
        t += step
        i += 1
    return i


def lock_pattern(root, third=3, four=False):
    r = m(root)
    return [r + x for x in (LOCK4 if four else LOCK8)] if third == 3 else \
        [r + (x if x != 3 else third) for x in (LOCK4 if four else LOCK8)]


def bloom(S, b0, b1, parts=None, level=0.42, breathe=0.1, voicing=None, legato=False, sync=True):
    """THE LYDIAN BLOOM: Bbmaj9(#11)/D in the strings, breathing once a bar.
    parts maps a voicing key (cb, vc, vla, vln2, vln1) to a part name."""
    voicing = voicing or BLOOM
    parts = parts or {k: k for k in voicing}
    for k, ps in voicing.items():
        pn = parts.get(k)
        if not pn:
            continue
        for p in ps:
            S.P(pn).n(p, b0, b1 - b0, legato=legato, sync=sync)
        pts = []
        t = b0
        while t < b1 - 1e-6:
            pts += [(t + 0.02, level + breathe * 0.6), (t + 1.6, level + breathe), (t + 3.6, level)]
            t += 4
        S.P(pn).d(*pts)


def silence_piano(S, pn, b0, at=(0.0, 1.8, 3.7), together=5.1, damp=None, vels=(0.24, 0.21, 0.23),
                  root="D4"):
    """THE SILENCE PIANO: D (with D3 under it) ... A ... D, then the low fifth D2+A2 ("together").
    The notes ring (pedal) until `damp` (global beat)."""
    r = m(root)
    damp = damp if damp is not None else b0 + (together or at[-1]) + 6.0
    for off, p, v in zip(at, (r, r + 7, r + 12), vels):
        S.P(pn).n(p, b0 + off, damp - (b0 + off), v, sync=True)
    S.P(pn).n(r - 12, b0 + at[0] + 0.02, damp - (b0 + at[0]), vels[0] * 0.5)
    if together is not None:
        S.P(pn).n(r - 24, b0 + together, damp - (b0 + together), 0.15)
        S.P(pn).n(r - 17, b0 + together + 0.03, damp - (b0 + together), 0.13)
    return damp


def tetrachord(start, direction=-1, mode="minor", tonic="D"):
    """B's ground: four scale steps from `start` (a note of `mode` on `tonic`), down or up"""
    s = m(start)
    sc = sorted({(m(tonic + "0") + x) % 12 for x in MODES[mode]})
    out = [s]
    p = s
    while len(out) < 4:
        p += direction
        if p % 12 in sc:
            out.append(p)
    return out


def walk(S, pizz, drum, b0, b1, roots, fifth=True, v=0.36, dv=0.22, drum_pitch=60):
    """A's WALK: pizzicato on beats 1 and 3 (root, then fifth below), a soft drum on 1 and 3 (the
    feet).  roots = [(beat, 'D2'), ...]: the harmony moves only at these beats."""
    roots = sorted((b, m(p)) for b, p in roots)
    t = b0
    k = 0
    while t < b1 - 1e-6:
        r = [p for b, p in roots if b <= t + 1e-6]
        r = r[-1] if r else roots[0][1]
        p = r if (k % 2 == 0 or not fifth) else r - 5
        S.P(pizz).n(p, t, 2.0, v * (1.0 if k % 2 == 0 else 0.86))
        if drum:
            S.P(drum).n(drum_pitch, t, 2.0, dv * (1.0 if k % 2 == 0 else 0.8), sync=(k == 0))
        t += 2.0
        k += 1


# ---------------------------------------------------------------------------
# orchestration helpers
# ---------------------------------------------------------------------------
def chords(S, pn, seq, overlap=0.06, legato=True, **kw):
    """seq: [(beat, [pitches]), ..., (end_beat, None)] -> held chords, legato between"""
    for i in range(len(seq) - 1):
        t0, ps = seq[i]
        t1 = seq[i + 1][0]
        if ps is None:
            continue
        for p in ps:
            S.P(pn).n(p, t0, t1 - t0 + overlap, legato=(legato and i > 0 and seq[i - 1][1] is not None), **kw)


def double(S, src, dst, t0, t1, octave=0, vel_scale=1.0, dyn_offset=0.0):
    """double an exposed line (red team 5.4): copy src's notes in [t0, t1) to dst (octave shift),
    with dst's dynamics following src's curve (offset by dyn_offset)"""
    o = S.P(src)
    q = S.P(dst)
    for nt in o.notes:
        if t0 - 1e-6 <= nt.start < t1:
            q.n(nt.pitch + 12 * octave, nt.start, nt.dur, None if nt.vel is None else nt.vel * vel_scale,
                legato=nt.legato, sync=nt.sync, **dict(nt.kw))
    pts = [(b, max(0.02, v + dyn_offset)) for b, v in o.dyn if t0 - 4 <= b <= t1 + 4]
    if pts:
        q.d(*pts)


def pad(S, voices, seq, level, t_end=None, overlap=0.08):
    """sit a line on a pad: voices = [part names, low to high]; seq = [(beat, [pitches low..high]), ...,
    (end, None)] - each chord's k-th pitch goes to voices[k] (extra pitches to the last voice)"""
    for i in range(len(seq) - 1):
        t0, ps = seq[i]
        t1 = seq[i + 1][0]
        if not ps:
            continue
        for k, p in enumerate(ps):
            pn = voices[min(k, len(voices) - 1)]
            S.P(pn).n(p, t0, t1 - t0 + overlap, legato=i > 0)
    for pn in voices:
        if isinstance(level, (list, tuple)):
            S.P(pn).d(*level)
        else:
            S.P(pn).d((seq[0][0], level))


def layer(S, src, new, t0, t1, gain_db=-1.5, pan_shift=0.12, seed_off=7):
    """a second desk/section on the same notes (own round robins, detune, humanising)"""
    o = S.P(src)
    q = S.add(new, None, **dict(inst=o.inst, kind=o.kind, bus=o.bus, pan=float(np.clip(o.pan + pan_shift, -1, 1)),
                                width=o.width, gain_db=o.gain_db + gain_db, send=o.send, depth=o.depth,
                                humanize_ms=o.humanize_ms, params=dict(o.params)))
    q.seed = o.seed + seed_off
    for nt in o.notes:
        if t0 - 1e-6 <= nt.start < t1:
            q.n(nt.pitch, nt.start, nt.dur, nt.vel, legato=nt.legato, sync=nt.sync, **dict(nt.kw))
    q.dyn = [(b, v) for b, v in o.dyn if t0 - 4 <= b <= t1 + 4]
    return q


def breathe(S, pn, t0, t1, depth=0.12, min_dur=1.2):
    """phrasing: each long note in [t0, t1) gets a messa di voce on top of the part's curve"""
    o = S.P(pn)
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


# ---------------------------------------------------------------------------
# rule checks on the page (the analysis battery reads these too)
# ---------------------------------------------------------------------------
def check_notes(parts, end_beat, max_s=22.0):
    """stuck notes (longer than max_s, or past the end), overlapping same-pitch notes in one part
    (not legato re-bows), zero/negative lengths"""
    warn = []
    for pn, p in parts.items():
        ns = sorted((n for n in p.notes if n.pitch is not None), key=lambda n: (n.pitch, n.start))
        for n in ns:
            if n.dur <= 0:
                warn.append(f"ZERO {pn} {nname(n.pitch)} at beat {n.start:.2f}")
            if bt(n.dur) > max_s and p.inst not in ("subpulse",):
                warn.append(f"LONG {pn} {nname(n.pitch)} at beat {n.start:.2f}: {bt(n.dur):.1f} s")
            if n.start + n.dur > end_beat + 0.01:
                warn.append(f"PAST END {pn} {nname(n.pitch)} at beat {n.start:.2f}")
        for a, b in zip(ns[:-1], ns[1:]):
            if a.pitch == b.pitch and b.start < a.start + a.dur - 0.2 and not b.legato:
                warn.append(f"OVERLAP {pn} {nname(a.pitch)} beats {a.start:.2f}+{a.dur:.2f} / {b.start:.2f}")
    return warn


PERC_INST = {"timp", "timp_roll", "bdrum", "giant_mallet", "tenor_lo", "tenor_hi", "snare", "taiko", "deephit",
             "clash", "crash", "suscym", "gong", "triangle"}


def check_rates(parts, max_per_s=8.0, fastest=0.25):
    """no identical sample repeated more than ~8 times a second (same part, same key, 1 s window);
    no ensemble figuration or percussion faster than 16ths (onset spacing < fastest beats)"""
    warn = []
    for pn, p in parts.items():
        starts = {}
        for n in p.notes:
            if n.pitch is None:
                continue
            starts.setdefault(int(round(n.pitch)), []).append(n.start)
        allst = sorted(n.start for n in p.notes if n.pitch is not None)
        for k, st in starts.items():
            st = np.sort(np.array(st))
            if len(st) > max_per_s:
                c = np.searchsorted(st, st + tb(1.0)) - np.arange(len(st))
                if c.max() > max_per_s:
                    warn.append(f"REPEAT {pn} {nname(k)}: {int(c.max())} onsets in 1 s")
        if p.kind == "sampler" or p.inst in PERC_INST:
            d = np.diff(np.unique(np.round(allst, 4)))
            if len(d) and d.min() < fastest - 1e-3 and p.inst not in ("harp", "piano", "glock", "fmbell"):
                warn.append(f"FAST {pn}: onsets {d.min():.3f} beats apart (< 16ths)")
    return warn


def voice_lead(chords, ranges, prev=None, need=None):
    """choose a voicing for each chord (a list of pitch classes, root first) for len(ranges) voices
    (low -> high, each (lo, hi) MIDI), minimising the total movement from the previous voicing; no
    crossings or unisons; every voicing keeps the chord's root and third (need: pcs that must sound)."""
    import itertools
    out = []
    for ch in chords:
        pcs = [c % 12 for c in ch]
        cands = [[p for p in range(lo, hi + 1) if p % 12 in pcs] for lo, hi in ranges]
        must = set(need or pcs[:2])
        best = None
        for combo in itertools.product(*cands):
            if any(b <= a for a, b in zip(combo[:-1], combo[1:])):
                continue
            have = {x % 12 for x in combo}
            miss = len(must - have)
            cost = 12 * miss + (sum(abs(a - b) for a, b in zip(combo, prev)) if prev else
                                sum(abs(x - (lo + hi) / 2) for x, (lo, hi) in zip(combo, ranges)) * 0.3)
            cost += 0.5 * (len(combo) - len(have))              # prefer fewer doublings
            if best is None or cost < best[0]:
                best = (cost, combo)
        prev = list(best[1])
        out.append(prev)
    return out


def ride(S, bm, db, xfade=1.0, skip=()):
    """a conductor's ride: scale every part's dynamics (and explicit velocities) section by section by `db`
    ({section id: dB}); dynamics map to 30*log10(level), so the players play softer (the sampler's velocity
    layers follow), not a fader.  Section joins blend over `xfade` beats."""
    import math
    secs = [(s["b0"], s["b1"], 10 ** (db.get(s["id"], 0.0) / 30.0)) for s in bm.sections]

    def f(b):
        for i, (b0, b1, g) in enumerate(secs):
            if b0 <= b < b1:
                if i > 0 and b < b0 + xfade:
                    u = (b - b0) / xfade
                    return secs[i - 1][2] * (1 - u) + g * u
                return g
        return secs[-1][2]
    for pn, p in S.parts.items():
        if pn in skip:
            continue
        p.dyn = [(b, min(1.0, v * f(b))) for b, v in p.dyn]
        for n in p.notes:
            if n.vel is not None:
                n.vel = min(1.0, n.vel * f(n.start))


_ATT = {}


def attack_time(pkey, pitch, level):
    """seconds from a sample's detected onset to its arrival (-6 dB under its early peak), for the sample the
    sampler would pick for this pitch and level (the nearest velocity layer, first round robin)"""
    import sampler
    import sampler_v2  # noqa: F401  (registers the quiet strings etc.)
    import soundfile as sf
    k = (pkey, int(round(pitch)), round(level, 2))
    if k in _ATT:
        return _ATT[k]
    layers = sampler.pick(pkey, pitch)
    c, regs = min(layers, key=lambda L: abs(L[0] - level * 127))
    r = regs[0]
    x, sr = sf.read(r["path"], dtype="float32", always_2d=True)
    mono = np.abs(x).mean(1)
    mt = sampler.meta(r["path"])
    i0 = int(mt["onset"] * sr)
    w = int(0.005 * sr)
    seg = mono[i0:i0 + int(1.5 * sr)] ** 2
    env = 10 * np.log10(np.convolve(seg, np.ones(w) / w, mode="same") + 1e-14)
    j = int(np.argmax(env > env.max() - 6.0))
    shift = pitch - r["center"]
    t = j / sr * 2 ** (-shift / 12.0)
    _ATT[k] = t
    return t


def anticipate(S, pn, lo=0.04, hi=0.25, frac=1.0):
    """every attacked (non-legato) note of part pn starts early by its sample's own attack time, so that the note
    ARRIVES on the beat (a soft bowed or blown entry blooms into the beat instead of after it)"""
    p = S.P(pn)
    for n in p.notes:
        if n.legato or n.pitch is None or p.kind != "sampler":
            continue
        lvl = n.vel if n.vel is not None else dyn_at(p.dyn, n.start)
        n.kw["antic"] = float(np.clip(frac * attack_time(n.art or p.inst, n.pitch, lvl), lo, hi))
