"""THE LONG DAWN v3 - C . THE LAST PAGES, the score: the epic tradition, heard from inside a book (owner: COMPOSER-C).

Locked to music/v3/barmap_C.json (SHOWRUNNER-REV) through music/v3/cues_C.json; both LOCKED and never edited.  The
beat sheet's C column asks for more effects than the cue sheet's base list; build() adds them IN MEMORY (the pen,
the burn-throughs, the cold metal's tick, the seethe, the cock, the drop, the fire that remains), from
voices_v3_C.py, which also holds C's own synth voices.  H5 CALLS: bars 70-71 are THE FIRE REMAINS (warm, settling,
the ANSWER passed outward, the tritone resolved, into bar 72's 275 ms breath and the sunrise).

Palette (BIBLE_V3 section 9 + REVISION 1): two sound worlds.  The BOOK is chamber music (cor anglais, harp, clarinet,
solo violin, the pen and the page); the FIRE is the orchestra (horns, strings, timpani, a pianissimo brass chorale).
Every burn-through is a change of orchestration, never a hit.  Original melodies only; no choir, no organ, nothing
Shire-like (no whistle, no pipes).

The rules it keeps (checked by build() and analyze_v3):
  whoever lights a fire gets the CALL; only an answer earns the ANSWER, first heard in the Run on beacon 2 (bar 50);
  the whole theme (CALL, ANSWER, HOME) only at the dawn; THE RING (the corrupted call D-Ab-D' C' Ab as a closed loop)
  never cadences until the fire takes it, then resolves into D once (69 b1), C's emotional climax; the effects own
  every fire, the sunrise and the stone (no musical stroke on an ignition); percussion never faster than 16ths;
  exposed lines doubled or on a pad; C's loudest moment is the slit opening onto nothing (26 b1); the dawn warm,
  not loud.

THE PLAN (bar.beat)
  C1  1       black: the hearth, a heavy page (effects only)
  C2  2-4     the storyteller: the cor anglais CALL (D4 A4 D5; clarinet in unison) on a pad of low strings in D
              minor; at the blank sheaf (4 b1) the third gives way to the ninth (D A E: the unwritten chord); the
              harp's open question
  C3  5-7     the riffle, the pen (effects); a harp broken chord a bar (Dm, Bb/D, D half-diminished); on 5 b3, as
              the pen draws the ring, THE RING on stopped horns (a2): two cycles and a half, hanging on its Ab
  C4  8-11    the page darkens: a low D, ppp; 9 b4 the letters glow: harp harmonics; 10 b1 they lift: the glass
              arpeggio under the harp, sparks falling inward to one point; 11 b1 the fire catches (effects); from
              11 b3 (the burn-through) the orchestra's low strings swell in: chamber to orchestra, never a hit
  C5  12-13   THE FIRE, ALONE: the fire's chord (Bbmaj9#11/D) blooms once in divided strings and a harp roll; one
              warm line rises in the violas (cellos below): D E F A C ... D
  C6  14-18   the towers rise: the harmony darkens over a D pedal (Bbmaj7, Gm, Ebmaj7, D half-dim), a quiet 8th
              pulse in the low strings, the glass quickening to 16ths; from 16 b1 the anvil strokes (timpani, low
              bells, the steel of the anvil) on the beat; 17 b3 the inscription: glass and harp harmonics; 18 b1 THE
              RING on open horns
  C7  19-21   the hush: the Ring motif alone, slower (the Ring hangs); the 200 ms breath; from 20 b3 THE RACE:
              timpani like a forge (8ths), taiko on the beat, the corrupted call in trombones and tuba answered in
              canon by the horns, tremolo strings on the tritone, a glass glitter on every surge
  C8  22-24   swept to parchment (the burn): the anvil tick on every beat, 8ths from 23, 16ths from 24; the halls
              stacked downward: bassoon and cellos descend D C Bb Ab G F E Eb; the red glow: a timpani roll and
              the tam-tam's swell begin
  C9  25-26   THE EYE: the Ring in the trombones (tuba and basses under), tremolo strings and horns on the tritone
              swelling with the timpani roll and the tam-tam into 26 b1: the slit opens, C's loudest; the tutti
              stops after a beat and a half and only the tam-tam's bloom and the storm are left: no one is there
  C10 27-29   THE MIRROR: one high held tone (A5, a far harmonic, doubled); harp harmonics on the Ring's D and Ab;
              the drop (28 b3): for one breath the harmonics and a pppp chord turn to D major; then the Ab returns
  C11 30-31   THE GRASP: the riser, tremolo and a timpani roll into the claw; the suck (a reverse swell) as it
              closes (30 b3), then nothing but gold leaking (three glass drops); 31 b1 the crust glows: a high
              tremolo swell; 31 b3 the slip: no impact, one glass tone falls away into silence
  C12 32-34   black: the Ring loop alone, low (clarinet and cellos), slow, hanging on its Ab; harp harmonics; the
              cock far away under the last harmonic (34 b4)
  C13 35-37   THE RING FALLS: the glass tone takes up its fall, turning; the streak (36 b3) a brighter, faster
              fall; gone before the snow (37 b2); the moonlit ink range: a cold open fifth
  C14 38-42   strikes and breath (effects); 39 b3.5 the vision: THE RING in the violas, ppp and sweet, on a D7#11
              halo of quiet strings; cut dead at the fist (40 b2): silence through the long blow and the catch
  C15 43-45   the roar: a lone horn cries the CALL over a low D; the Ring under the flames (bassoon and cellos): it
              should slide off at 44 b3, but the loop holds its D; at 45 b3 she closes her fist and it closes
  C16 46-48   THE REVEAL: the call echoes off the ranges, farther each time (three horns, each deeper in the hall)
  C17 49-52   THE LIVING INK RUN: 16th spiccato in violas and violins, grouped 3+3+2 with the accents OFF the
              beacons (the fires own them); timpani only on the group's 3 and 6; a rising bass (D Bb F G A Bb C)
              and a soaring violin line; the CALL passed peak to peak on beacons 1, 3, 5, 7 (on D, F, A, C, each
              horn farther); THE ANSWER, first heard, on a far horn on beacon 2 (D C Bb F over Bb)
  C18 53-56   THE MAP: wonder in F major: the violins sing the CALL on F, the oboe answers (F E D A), a solo violin
              descant above (doubled), damped harp arpeggios running hill by hill; THE ROAD: a slow pizzicato walk
              (one dot a beat) to the ring of stones, arriving on A (56 b3)
  C19 57-60   THE COUNCIL: the clarinet intones the CALL (violas in unison); a pianissimo brass chorale in half
              notes, from its first chord Bb/D
  C20 61-64   BRING OUT THE RING: the chorale moves between the events (61 b4, 62 b4, ...), never on them; from the
              Ring set on the stone the Ring loop circles beneath (bassoon and cellos), and the chorale darkens into
              its chord (D half-diminished)
  C21 65-67   THE BEARER: the Ring loop rises in the low strings, tremolo, a minor third each cycle (D, F, Ab),
              against the chorale growing from p to f over a bass climbing Bb C Db Eb E F#
  C22 68-69   THE UNMAKING: the white heart: the loop HEALS in the whole orchestra (D Ab D' C#' A over a D pedal,
              the C# and A as the letters flare); 275 ms of silence; 69 b1 D major: the Ring loop cadences into D,
              once, for the first time (C's emotional climax); the seethe of the melt and nothing else
  C23 70-71   THE FIRE REMAINS: D major settles, warm and steady (the torches dip: effects); 71 b1 the burn-through
              to the map: the ANSWER passed outward, horn to farther horn in canon, soft, over G/B - D/A - A; the
              tritone resolved; into the 275 ms breath
  C24 72-77   THE ILLUMINATION: a bloom out of silence on D add9; a timpani roll from the light, pp to mf, no
              stroke; the whole theme at last: the CALL on one horn, then the ANSWER and HOME in octave violins with
              a solo violin above (doubled), horns in counterpoint; home on 76 b1 (B A F# D) over a bass walking
              down G F# E D
  C25 78-80   THE YEAR OF PLENTY: a plain hymn, clarinet over three violas
  C26 81-84   THE HAVENS: an elegy for strings (Bm G D/F# Em7 Bm G A7sus4); the cor anglais carries the CALL twice;
              a far horn answers each coast fire with the ANSWER, farther and softer each time; the sea
  C27 85-87   THE LAST PAGES: the chorale's first chord (Bb/D) returns, pp; the pen writes the last line
  C28 88-90   TITLE: Gm/D, G/D ... 90 b1 the plagal close into D, ringing into silence; the hearth
"""
import numpy as np

import kit_v3 as K
import voices_v3_C as V
from dsl import m
from timeline_v3 import gb, BEAT_S

REV = V.REV


def G(bar, beat=1.0):
    return gb(bar, beat)


def T(bar, beat=1.0):
    return gb(bar, beat) * BEAT_S


# ---------------------------------------------------------------------------
# seats (C's own; the kit's SEATS stay untouched for everyone else)
# ---------------------------------------------------------------------------
_S = dict(bus="strings")
_B = dict(bus="brass")
_W = dict(bus="winds")
_SY = dict(kind="synth", humanize_ms=0)
C_SEATS = {
    # the book: the cor anglais (the oboe's low register, darkened) doubled in unison by a clarinet
    "ca": dict(inst="oboe", pan=0.10, width=0.3, depth=0.3, send=0.36, gain_db=2.0, humanize_ms=10, **_W),
    "ca_dbl": dict(inst="clarinet", pan=0.14, width=0.3, depth=0.32, send=0.36, gain_db=-1.8, humanize_ms=12, **_W),
    "cl_c": dict(inst="clarinet", pan=0.12, width=0.35, depth=0.42, send=0.4, gain_db=5.2, humanize_ms=10, **_W),
    "cl_dbl": dict(inst="vla_q", pan=0.06, width=0.4, depth=0.36, send=0.32, gain_db=-8.0, humanize_ms=10, **_S),
    "ob_c": dict(inst="oboe", pan=0.22, width=0.35, depth=0.45, send=0.42, gain_db=0.0, humanize_ms=10, **_W),
    # the oboe's answer doubled by a clarinet (not a flute: nothing pastoral, nothing Shire-like in C)
    "ob_dbl": dict(inst="clarinet", pan=0.2, width=0.35, depth=0.45, send=0.42, gain_db=1.2, humanize_ms=12, **_W),
    "bsn_c": dict(inst="bassoon", pan=0.18, width=0.35, depth=0.45, send=0.38, gain_db=1.0, humanize_ms=10, **_W),
    # horns: stopped (a2), open, near to farthest
    "hn_st": dict(inst="horn", pan=-0.22, width=0.35, depth=0.42, send=0.36, gain_db=-6.0, humanize_ms=10, **_B),
    "hn3": dict(inst="horn", pan=-0.05, width=0.4, depth=0.58, send=0.46, gain_db=1.0, humanize_ms=12, **_B),
    "hn_ans": dict(inst="horn", pan=0.55, width=0.3, depth=0.85, send=0.68, gain_db=1.5, humanize_ms=12, **_B),
    "hn_far2": dict(inst="horn", pan=0.12, width=0.2, depth=1.0, send=0.92, gain_db=-4.0, humanize_ms=14, **_B),
    "tbn2": dict(inst="trombone", pan=0.52, width=0.5, depth=0.66, send=0.42, gain_db=-3.5, **_B),
    # strings: articulations and the lines
    "vla_sp": dict(inst="vla_spic", pan=0.05, width=0.5, depth=0.32, send=0.26, gain_db=-1.0, humanize_ms=5, **_S),
    "vln2_sp": dict(inst="vln_spic", pan=-0.30, width=0.5, depth=0.32, send=0.26, gain_db=-2.0, humanize_ms=5, **_S),
    "vc_sp": dict(inst="vc_spic", pan=0.35, width=0.5, depth=0.32, send=0.26, gain_db=0.0, humanize_ms=5, **_S),
    "cb_sp": dict(inst="cb_spic", pan=0.58, width=0.4, depth=0.36, send=0.26, gain_db=1.5, humanize_ms=5, **_S),
    "cb_trem": dict(inst="cb_trem", pan=0.58, width=0.4, depth=0.36, send=0.3, gain_db=1.0, **_S),
    "vc_pz": dict(inst="vc_pizz", pan=0.35, width=0.5, depth=0.34, send=0.34, humanize_ms=6, **_S),
    "cb_pz": dict(inst="cb_pizz", pan=0.58, width=0.4, depth=0.36, send=0.34, humanize_ms=6, **_S),
    "svln2": dict(inst="svln_q", pan=-0.34, width=0.4, depth=0.32, send=0.32, gain_db=4.0, humanize_ms=12, **_S),
    "line_vla": dict(inst="vla", pan=0.02, width=0.45, depth=0.3, send=0.3, gain_db=3.0, humanize_ms=8, **_S),
    "line_vc": dict(inst="vc", pan=0.3, width=0.45, depth=0.3, send=0.3, gain_db=-0.5, humanize_ms=8, **_S),
    "vc_ring": dict(inst="vc_q", pan=0.28, width=0.4, depth=0.34, send=0.3, gain_db=0.0, humanize_ms=8, **_S),
    # synth voices (voices_v3_C / synth.py), each carrying REV so an edit of voices_v3_C re-renders them
    "hharm": dict(inst="hharm", bus="keys", pan=-0.42, width=0.5, depth=0.45, send=0.55, gain_db=-16.0,
                  params=dict(decay=2.6, rev=REV), **_SY),
    "anvil": dict(inst="anvil", bus="perc", pan=0.08, width=0.5, depth=0.55, send=0.38, gain_db=-7.0,
                  params=dict(rev=REV), **_SY),
    "tick": dict(inst="anvil", bus="perc", pan=0.22, width=0.3, depth=0.32, send=0.26, gain_db=-5.0,
                 params=dict(rev=REV), **_SY),
    "tam": dict(inst="tamswell", bus="perc", pan=0.0, width=1.0, depth=0.7, send=0.42, gain_db=-1.0,
                params=dict(rev=REV), **_SY),
    "gfall": dict(inst="glassfall", bus="synth", pan=-0.1, width=1.0, depth=0.5, send=0.55, gain_db=-8.0,
                  params=dict(rev=REV), **_SY),
    "gfall2": dict(inst="glassfall", bus="synth", pan=0.2, width=1.0, depth=0.45, send=0.5, gain_db=-10.0,
                   params=dict(rev=REV), **_SY),
    "glitter": dict(inst="fmbell", bus="synth", pan=0.0, width=1.0, depth=0.45, send=0.45, gain_db=-15.0,
                    params=dict(ratio=3.5, index=2.4, decay=0.45, rev=REV), **_SY),
    "taiko": dict(inst="taiko", bus="perc", pan=0.0, width=0.8, depth=0.62, send=0.36, gain_db=-12.0,
                  params=dict(rev=REV), **_SY),
    "lowbell": dict(inst="lowbell", bus="perc", pan=-0.12, width=0.6, depth=0.62, send=0.46, gain_db=-10.0,
                    params=dict(rev=REV), **_SY),
    "riser": dict(inst="riser", bus="synth", pan=0.0, width=1.0, depth=0.5, send=0.45, gain_db=-12.0,
                  params=dict(rev=REV), **_SY),
    "revcym": dict(inst="revcym", bus="perc", pan=0.0, width=1.0, depth=0.6, send=0.5, gain_db=-10.0,
                   params=dict(rev=REV), **_SY),
}
for _k, _v in C_SEATS.items():
    K.SEATS.setdefault("C." + _k, _v)

# dB per section, on the players' dynamics (kit ride): set from the battery (render 1)
RIDE = {"C6": -1.5, "C7": -1.0, "C9": -1.0, "C17": -3.0, "C21": -2.5, "C22": -1.0, "C24": -0.5, "C26": -3.0}
# the whole score against the effects (gain only; the master brings the sum back to -16 LUFS): render 1 had the
# effects 16 LU under the score (B: 8; the fallback, which the cue sheet's level bands were drawn on: 11)
SCORE_TRIM_DB = -4.5
# anticipation caps (s): the quiet sections' slow bow arrives ~0.52 s after it starts, the solo violin ~0.44 s
# (measured per layer, renders 1-2); the attack table over-reads the vibrato samples, so everything else stays at
# the kit's 0.25
ANTIC_HI = {"vla_q": 0.52, "vc_ring": 0.52, "cl_dbl": 0.52, "vc_q": 0.52, "cb_q": 0.52, "vln1_q": 0.52, "vln2_q": 0.52,
            "svln": 0.45, "svln2": 0.5}
# seconds added to one entry's anticipation (the battery's per-layer sync, render 1)
ANTIC_DT = {("torches_down", "vc_trem"): -0.087, ("council", "cl_c"): -0.067}
# (sync probes: "arrive:<midi>" measures a soft entry in its fundamental's band; low violas, the bassoon and a cello's D2
# have almost no fundamental, and a solo violin's vibrato leaves a narrow band, so those entries (each out of its own
# silence) use "arrivew", the same arrival on the part's broadband envelope; a crescendo bloom has no single arrival,
# so it is probed by its onset, "bloom")
# absolute anticipation of one entry (event id, or (event id, beats after it)): a bloom's ONSET sits on the beat
# (a swell has no single arrival); the dawn's first violins measured; a legato change's crossfade starts early by
# this + half its fade
ANTIC_SET = {("fire_alone", "vla"): 0.05, ("fire_alone", "vln1"): 0.05, ("fire_alone", "vln2"): 0.05,
             ("map", "vla"): 0.06, ("map", "vln2"): 0.06, ("map", "vc"): 0.06, ("map", "cb"): 0.06,
             ("ring_set", "bsn_c"): 0.06, ("ring_set", "vc_ring"): 0.3,
             (("sunrise", 8), "vln1"): 0.16, (("sunrise", 8), "svln"): 0.23, (("sunrise", 8), "svln2"): 0.34,
             ("plagal", "vln1_q"): 0.28}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def seat(S, name, base=None, **over):
    """add a part from C's seats (or the kit's), with overrides"""
    if name in S.parts:
        return S.parts[name]
    key = "C." + (base or name)
    if key in K.SEATS:
        return S.add(name, key, **over)
    return S.add(name, base, **over)


def hold(S, pn, pitch, b0, b1, legato=False, sync=False, vel=None, **kw):
    return S.P(pn).n(pitch, b0, b1 - b0, vel, legato=legato, sync=sync, **kw)


def dyn(S, pns, *pts):
    for pn in ([pns] if isinstance(pns, str) else pns):
        S.P(pn).d(*pts)


def voiced(S, seq, t_end, overlap=0.08, sync_first=False, **kw):
    """held chords: seq = [(beat, {part: [pitches]}), ...]; a part re-bows legato when it held a chord just before"""
    for i, (t0, v) in enumerate(seq):
        t1 = seq[i + 1][0] if i + 1 < len(seq) else t_end
        prev = seq[i - 1][1] if i > 0 else {}
        for pn, ps in v.items():
            for p in ps:
                S.P(pn).n(p, t0, t1 - t0 + (overlap if i + 1 < len(seq) else 0.0), legato=bool(prev.get(pn)),
                          sync=(sync_first and i == 0), **kw)


def line(S, pn, notes, b0, vel=None, sync_first=True, legato=True, **kw):
    """notes = [(pitch | None, beats), ...] -> the beat after the last note"""
    t = b0
    prev = None
    for i, (p, d) in enumerate(notes):
        if p is not None:
            S.P(pn).n(p, t, d, vel, legato=(legato and prev is not None), sync=(sync_first and i == 0), **kw)
        prev = p
        t += d
    return t


def lines(S, pns, notes, b0, octaves=None, **kw):
    """the same line in several parts (unison doubling, or at octaves)"""
    end = b0
    for k, pn in enumerate(pns):
        o = 12 * (octaves[k] if octaves else 0)
        end = line(S, pn, [(None if p is None else m(p) + o, d) for p, d in notes], b0, **kw)
    return end


def ring_notes(root, b0, unit=1.0, heal=False):
    r = m(root)
    iv = K.RING_HEALED if heal else K.RING
    out, t = [], b0
    for x, d in zip(iv, K.RING_DUR):
        out.append((r + x, t, d * unit))
        t += d * unit
    return out


def copy_part(S, src, dst, t0, t1):
    """a second desk on the same notes (its own seat, round robins and humanising) with the same dynamics"""
    o, q = S.P(src), S.P(dst)
    for nt in o.notes:
        if t0 - 1e-6 <= nt.start < t1:
            q.n(nt.pitch, nt.start, nt.dur, nt.vel, legato=nt.legato, sync=nt.sync, **dict(nt.kw))
    q.d(*[(b, v) for b, v in o.dyn if t0 - 4 <= b <= t1 + 4])


def put(S, pn, notes, legato=True, sync_first=True, **kw):
    """notes = [(pitch, beat, dur), ...] in time order"""
    for i, (p, t, d) in enumerate(notes):
        S.P(pn).n(p, t, d, legato=(legato and i > 0 and abs(notes[i - 1][1] + notes[i - 1][2] - t) < 0.05),
                  sync=(sync_first and i == 0), **kw)


# ---------------------------------------------------------------------------
# setup
# ---------------------------------------------------------------------------
def setup(S):
    for pn in ("cb", "vc", "vla", "vln2", "vln1", "cb_q", "vc_q", "vla_q", "vln2_q", "vln1_q", "vln_trem", "vla_trem",
               "vc_trem", "svln", "harp", "timp", "timp_roll", "hn", "hn2", "hn_far", "hn_farther", "tbn", "tuba",
               "glass", "harmonic"):
        S.add(pn)
    for pn in C_SEATS:
        seat(S, pn)
    seat(S, "hn_st2", "hn_st", pan=-0.1, gain_db=-8.0)
    seat(S, "harm_str", "svln", inst="svln_q", pan=0.3, depth=0.6, send=0.6, gain_db=-4.0)
    seat(S, "pad_vla2", "vla_q", pan=0.2)
    seat(S, "hymn_vla2", "vla_q", pan=-0.15)
    seat(S, "hymn_vla3", "vla_q", pan=0.28)
    # the cor anglais: the oboe darkened (its reed's upper partials rolled off); the stopped horns: brassy and
    # nasal, the fundamental thinned
    S.P("glass").gain_db = -15.0                   # the fire's glass sits under C's strings
    # C's balance (renders 1-2, per-part probe): the lines over the foundation; the kit's basses sit heavy under
    # C's lighter textures, and the solo violin must be heard above its section
    S.P("cb").gain_db -= 3.0
    S.P("cb_q").gain_db -= 2.0
    S.P("vc").gain_db -= 1.0
    S.P("svln").gain_db += 6.0
    S.eq["ca"] = (None, 3600.0)
    S.eq["hn_st"] = (300.0, None, 6.0, 1500.0)
    S.eq["hn_st2"] = (300.0, None, 6.0, 1500.0)
    S.eq["hharm"] = (180.0, None)


# ---------------------------------------------------------------------------
# PROLOGUE and BOOK ONE (C1-C12)
# ---------------------------------------------------------------------------
def storyteller(S, bm):
    ev = bm.ev
    b, sheaf, rif = ev("book"), ev("blank_sheaf"), ev("riffle")
    # the pad of low strings: D minor; at the blank sheaf the third gives way to the ninth (D A E)
    voiced(S, [(b, {"cb_q": ["D2"], "vc_q": ["A2", "D3"], "vla_q": ["F3"]}),
               (sheaf, {"cb_q": ["D2"], "vc_q": ["A2", "D3"], "vla_q": ["E3"]})], rif + 0.3, sync_first=True)
    dyn(S, ("cb_q", "vc_q", "vla_q"), (b - 0.3, 0.03), (b + 1.3, 0.15), (b + 4, 0.19), (sheaf, 0.17),
        (rif - 1.2, 0.1), (rif + 0.3, 0.02))
    # the storyteller's CALL: cor anglais, the clarinet in unison
    lines(S, ("ca", "ca_dbl"), [("D4", 2.0), ("A4", 1.5), ("D5", 4.3)], b)
    pts = [(b - 0.1, 0.3), (b + 1.4, 0.33), (b + 2.0, 0.31), (b + 3.3, 0.36), (b + 5.0, 0.38), (b + 6.6, 0.3),
           (b + 7.8, 0.04)]
    dyn(S, "ca", *pts)
    dyn(S, "ca_dbl", *[(t, max(0.02, v - 0.03)) for t, v in pts])
    S.sync.append((b * BEAT_S, "the storyteller's CALL: cor anglais (+ clarinet)", "ca+ca_dbl", 0.2, "arrive:62"))
    # the blank sheaf: the harp's open question (D A E ... A)
    for p, t, v in (("D3", sheaf + 0.02, 0.22), ("A3", sheaf + 0.5, 0.2), ("E4", sheaf + 1.0, 0.2),
                    ("A4", sheaf + 2.0, 0.17)):
        S.P("harp").n(p, t, 3.0, v)


def mountain(S, bm):
    ev = bm.ev
    rif, rm = ev("riffle"), ev("ring_motif")
    end = G(8)
    # a harp broken chord a bar as the pen draws: Dm | Bb/D | D half-diminished (no A natural once the Ring sounds)
    for t, ps in ((rif + 0.5, ("D3", "A3", "F4")), (G(6) + 0.5, ("D3", "Bb3", "F4")),
                  (G(7) + 0.5, ("D3", "Ab3", "C4", "F4"))):
        for k, p in enumerate(ps):
            S.P("harp").n(p, t + 0.5 * k, 3.0, 0.19 - 0.01 * k)
    # the low pedal under the drawing
    hold(S, "cb_q", "D2", rm + 0.5, end + 0.2)
    dyn(S, "cb_q", (rm + 0.2, 0.03), (rm + 2, 0.12), (end - 1, 0.1), (end + 0.2, 0.02))
    # THE RING on stopped horns (a2), as the pen draws the ring: two cycles and a half, hanging on its Ab
    ns = ring_notes("D4", rm) + ring_notes("D4", rm + 4)
    ns += [(m("D4"), rm + 8, 1.0), (m("Ab4"), rm + 9, end - (rm + 9) + 0.3)]
    put(S, "hn_st", ns)
    dyn(S, "hn_st", (rm - 0.2, 0.36), (rm + 2, 0.4), (rm + 4, 0.36), (rm + 6, 0.4), (rm + 8, 0.35), (end - 1, 0.3),
        (end + 0.3, 0.04))
    copy_part(S, "hn_st", "hn_st2", rm - 0.1, end + 1)
    S.sync.append((rm * BEAT_S, "THE RING enters on stopped horns (D4)", "hn_st", 0.2, "arrive:62"))


def letters_to_fire(S, bm):
    ev = bm.ev
    pt, glow, lift = ev("page_turn"), ev("letters_glow"), ev("letters_lift")
    catch, burn, alone = ev("fire_catches"), ev("burn_through"), ev("fire_alone")
    # the page almost dark: a low D, ppp, until the burn-through
    voiced(S, [(pt + 0.5, {"vc_q": ["D3"], "cb_q": ["D2"]})], burn + 0.3)
    dyn(S, ("vc_q", "cb_q"), (pt + 0.2, 0.03), (pt + 2.5, 0.1), (glow, 0.1), (burn - 0.5, 0.08), (burn + 0.3, 0.03))
    # the letters glow: harp harmonics (the gold of the Ring's letters in fire)
    for p, t, v in (("A5", glow, 0.34), ("E6", glow + 0.6, 0.3), ("D6", glow + 1.3, 0.3), ("C6", glow + 2.2, 0.27),
                    ("A5", glow + 3.1, 0.25), ("E6", glow + 4.4, 0.22)):
        S.P("hharm").n(p, t, 2.0, v)
    S.sync.append((glow * BEAT_S, "the letters glow: harp harmonics (A5)", "hharm", 0.02, "hit"))
    # the letters lift: the glass arpeggio under the harp, sparks drawn down together into one point (never a
    # spiral): each two-beat fall narrower, then the fire's own cycle once it catches
    sparks = ["E6", "C6", "A5", "F5", "D6", "A5", "E5", "D5"]         # 8ths, lift to the catch
    for k, p in enumerate(sparks):
        S.P("glass").n(p, lift + 0.5 * k, 0.5, None, sync=(k == 0), pan=0.4 * np.sin(k * 0.9))
    S.sync.append((lift * BEAT_S, "the letters lift: the glass arpeggio enters", "glass", 0.02, "hit"))
    K.thinking(S, "glass", K.CYC6A, catch, G(14), step=0.5, v0=0.24, v1=0.2, sync_first=False)
    dyn(S, "glass", (lift - 0.1, 0.26), (catch, 0.3), (alone, 0.24), (G(14), 0.2))
    # the burn-through: the orchestra's low strings swell in under the crackle (chamber to orchestra, never a hit)
    voiced(S, [(burn, {"cb": ["D2"], "vc": ["Bb2"]})], alone + 0.1)
    dyn(S, ("cb", "vc"), (burn - 0.2, 0.02), (burn + 1.0, 0.12), (alone, 0.22))


def fire_alone(S, bm):
    ev = bm.ev
    alone, towers = ev("fire_alone"), ev("towers_rise")
    # the fire's chord blooms once (Bbmaj9#11/D) in divided strings, legato on from the low strings' swell
    for pn, ps in K.BLOOM.items():
        for p in ps:
            S.P(pn).n(p, alone, towers - alone + 0.1, legato=(pn in ("cb", "vc")), sync=(pn == "vla"))
    for pn, top in (("cb", 0.3), ("vc", 0.3), ("vla", 0.3), ("vln2", 0.29), ("vln1", 0.28)):
        S.P(pn).d((alone - 0.05, 0.22 if pn in ("cb", "vc") else 0.12), (alone + 1.6, top), (alone + 4.5, top * 0.9),
                  (towers - 0.5, top * 0.82), (towers + 0.1, top * 0.8))
    S.sync.append((alone * BEAT_S, "THE FIRE, ALONE: the fire's chord blooms (violas: its onset)", "vla", 0.25,
                   "bloom"))
    # a harp roll up the chord with the bloom
    for k, p in enumerate(["D2", "Bb2", "F3", "C4", "D4", "E4", "A4", "E5"]):
        S.P("harp").n(p, alone + 0.02 + 0.14 * k, 4.0, 0.26 - 0.01 * k)
    # one warm line rises (violas, the cellos an octave below): D E F A C ... to D
    notes = [("D4", 1.0), ("E4", 1.0), ("F4", 1.5), ("A4", 0.5), ("C5", 2.5), ("D5", 3.5)]
    lines(S, ("line_vla", "line_vc"), notes, alone + 2.0, octaves=(0, -1))
    for pn, v in (("line_vla", 0.36), ("line_vc", 0.3)):
        S.P(pn).d((alone + 1.9, v * 0.8), (alone + 4, v), (alone + 6.5, v * 1.08), (towers + 1.5, v * 0.95),
                  (towers + 3.5, 0.04))


def forging(S, bm):
    ev = bm.ev
    towers, anv, insc, rises, hush = ev("towers_rise"), ev("anvil"), ev("inscription"), ev("ring_rises"), ev("hush")
    # the towers rise: the harmony darkens over a D pedal
    seq = [(towers, {"cb": ["D2"], "vc": ["D3", "A3"], "vla": ["F3", "Bb3"], "vln2": ["D4"], "vln1": ["A4"]}),
           (towers + 2, {"cb": ["D2"], "vc": ["D3", "G3"], "vla": ["Bb3"], "vln2": ["D4"], "vln1": ["G4"]}),
           (towers + 4, {"cb": ["D2"], "vc": ["D3", "G3"], "vla": ["Bb3"], "vln2": ["Eb4"], "vln1": ["G4"]}),
           (towers + 6, {"cb": ["D2"], "vc": ["D3", "F3"], "vla": ["Ab3"], "vln2": ["C4"], "vln1": ["F4"]})]
    voiced(S, seq, anv + 0.1)
    for pn, a, b in (("cb", 0.28, 0.36), ("vc", 0.28, 0.36), ("vla", 0.26, 0.34), ("vln2", 0.25, 0.33),
                     ("vln1", 0.24, 0.34)):
        S.P(pn).d((towers + 0.1, a), (anv - 0.3, b))
    # a quiet 8th pulse in the low strings (the smiths at work), growing
    t = towers + 2
    k = 0
    while t < anv - 1e-6:
        S.P("vc_sp").n("D3", t, 0.5, None)
        S.P("cb_sp").n("D2", t, 0.5, None)
        t += 0.5
        k += 1
    dyn(S, ("vc_sp", "cb_sp"), (towers + 1.9, 0.16), (anv - 0.5, 0.34))
    # the glass quickens: 16ths from bar 15 (the race of the smiths), darkening (the Ab)
    K.thinking(S, "glass", ["D5", "E5", "A5", "D6", "Eb6", "Ab5", "F5"], towers, towers + 4, step=0.5, v0=0.2,
               v1=0.22, sync_first=False)
    K.thinking(S, "glass", ["D5", "Eb5", "Ab5", "D6", "C6", "Ab5", "F5"], towers + 4, insc, step=0.25, v0=0.2,
               v1=0.27, sync_first=False)
    dyn(S, "glass", (towers, 0.2), (anv, 0.26), (insc, 0.28))
    # the anvil strokes: timpani and low bells on the beat (1 and 3 strong), the anvil's steel on 1 and 3
    t = anv
    while t < rises - 1e-6:
        strong = (round(t - anv) % 2 == 0)
        S.P("timp").n("D2", t, 1.0, 0.5 if strong else 0.3, sync=(t == anv))
        if strong:
            S.P("lowbell").n("D3" if round(t - anv) % 4 == 0 else "Ab2", t, 2.0, 0.42)
            S.P("anvil").n("A5", t, 1.0, 0.4, size=0.85)
        t += 1.0
    S.sync.append((anv * BEAT_S, "the first anvil stroke (timpani)", "timp", 0.02, "hit"))
    # the band beaten: the strings on the Ring's chord (D half-diminished), tremolo rising
    voiced(S, [(anv, {"cb": ["D2"], "vc": ["D3", "F3"], "vla_trem": ["Ab3", "C4"], "vln_trem": ["F4", "Ab4"]}),
               (anv + 4, {"cb": ["D2"], "vc": ["D3", "F3"], "vla_trem": ["Ab3", "D4"], "vln_trem": ["C5", "F5"]})],
           rises + 0.1)
    dyn(S, ("cb", "vc"), (anv, 0.36), (rises, 0.42))
    dyn(S, ("vla_trem", "vln_trem"), (anv - 0.2, 0.08), (anv + 3, 0.2), (insc, 0.3), (rises, 0.36))
    # the inscription burns up out of the metal: harp harmonics and a glass glint
    for p, dt, v in (("D6", 0.0, 0.3), ("Ab5", 0.4, 0.27), ("C6", 0.8, 0.26), ("F6", 1.3, 0.24)):
        S.P("hharm").n(p, insc + dt, 2.0, v)
    # THE RING rises above the towers: the Ring motif on open horns (a2), the strings holding its chord
    voiced(S, [(rises, {"cb": ["D2"], "vc": ["D3"], "vla": ["F3", "Ab3"], "vln2": ["C4"], "vln1": ["F4"]})], hush + 0.2)
    dyn(S, ("cb", "vc", "vla", "vln2", "vln1"), (rises - 0.1, 0.4), (rises + 2, 0.36), (hush - 0.3, 0.2), (hush + 0.2, 0.03))
    put(S, "hn", ring_notes("D4", rises))
    put(S, "hn2", ring_notes("D4", rises))
    for pn, v in (("hn", 0.5), ("hn2", 0.46)):
        S.P(pn).d((rises - 0.1, v), (rises + 2.5, v * 1.04), (hush - 0.1, v * 0.8))
    S.sync.append((rises * BEAT_S, "THE RING rises: the Ring motif on open horns (D4)", "hn+hn2", 0.2, "arrive:62"))


def race(S, bm):
    ev = bm.ev
    hush, go, deep = ev("hush"), ev("race"), ev("deep")
    # the hush: the Ring motif alone, slower (the Ring hangs); the breath cuts its last Ab
    put(S, "hn", ring_notes("D4", hush, unit=1.5))
    put(S, "hn2", ring_notes("D4", hush, unit=1.5))
    for pn, v in (("hn", 0.34), ("hn2", 0.3)):
        S.P(pn).d((hush - 0.05, v), (hush + 3, v * 0.92), (go - 0.3, v * 0.85))
    # THE RACE (from 20 b3, after the 200 ms breath): surges on every beat
    t = go
    while t < deep - 1e-6:
        onb = abs(t - round(t)) < 1e-6
        S.P("timp").n("D2" if onb else ("Ab2" if int(round(t * 2)) % 4 == 3 else "D2"), t, 0.5, 0.52 if onb else 0.3,
                      sync=(t == go))
        t += 0.5
    for k in range(int(deep - go)):
        S.P("taiko").n(60, go + k, 1.0, 0.6 if k % 2 == 0 else 0.45, drum=("o" if k % 2 == 0 else "n"), sync=(k == 0))
        # a glass glitter on each surge (a dyad on the tritone, climbing)
        lo = ("D6", "E6", "F6", "Ab6", "Bb6", "C7")[k % 6]
        for p in (lo, m(lo) + 6):
            S.P("glitter").n(p, go + k + 0.01, 0.5, 0.36 + 0.04 * k)
    S.sync.append((go * BEAT_S, "THE RACE: timpani and taiko after the breath", "timp", 0.02, "hit"))
    # the call bent into the tritone: trombones and tuba; the horns in canon a beat later
    for pn, root in (("tbn", "D3"), ("tuba", "D2")):
        line(S, pn, [(root, 1), (m(root) + 6, 1), (m(root) + 12, 2), (root, 1), (m(root) + 6, 1.0)], go)
    for pn, root in (("hn", "D4"), ("hn2", "D4")):
        line(S, pn, [(root, 1), (m(root) + 6, 1), (m(root) + 12, 2), (root, 1)], go + 1)
    dyn(S, ("tbn", "tuba"), (go - 0.05, 0.56), (deep - 0.5, 0.66), (deep, 0.3))
    dyn(S, ("hn", "hn2"), (go + 0.9, 0.5), (deep - 0.5, 0.6), (deep, 0.3))
    # the strings: tremolo on the tritone, the low strings' spiccato 8ths
    voiced(S, [(go, {"vln_trem": ["D5", "Ab5"], "vla_trem": ["Ab4", "D5"], "cb": ["D2"]})], deep)
    dyn(S, ("vln_trem", "vla_trem"), (go - 0.05, 0.32), (deep - 0.3, 0.48))
    dyn(S, "cb", (go - 0.05, 0.4), (deep - 0.3, 0.5), (deep, 0.1))
    t = go
    while t < deep - 1e-6:
        acc = abs(t - round(t)) < 1e-6
        S.P("vc_sp").n("D3", t, 0.5, 0.5 if acc else 0.36)
        S.P("cb_sp").n("D2", t, 0.5, 0.5 if acc else 0.36)
        t += 0.5


def deep(S, bm):
    ev = bm.ev
    d0, t8, t16, eye = ev("deep"), ev("tick_8"), ev("tick_16"), ev("eye_burn")
    # the anvil tick on every beat, dividing faster (quarters, 8ths, 16ths; never faster)
    t = d0
    k = 0
    while t < eye - 1e-6:
        step = 1.0 if t < t8 - 1e-6 else (0.5 if t < t16 - 1e-6 else 0.25)
        S.P("tick").n(("E6", "D#6")[k % 2], t, step, None, size=0.2, sync=(t in (d0, t8, t16)))
        t += step
        k += 1
    dyn(S, "tick", (d0, 0.34), (t8, 0.38), (t16, 0.42), (eye, 0.5))
    S.sync.append((t8 * BEAT_S, "the Deep's tick divides to 8ths", "tick", 0.02, "hit"))
    S.sync.append((t16 * BEAT_S, "the Deep's tick divides to 16ths", "tick", 0.02, "hit"))
    # the halls stacked downward: bassoon and cellos descend, the basses an octave under
    desc = [("D3", 2), ("C3", 2), ("Bb2", 2), ("Ab2", 2), ("G2", 1), ("F2", 1), ("E2", 1), ("Eb2", 1.1)]
    lines(S, ("bsn_c", "vc_ring", "cb_q"), desc, d0 + 0.05, octaves=(0, 0, -1))
    for pn, a, b in (("bsn_c", 0.26, 0.4), ("vc_ring", 0.24, 0.4), ("cb_q", 0.2, 0.34)):
        S.P(pn).d((d0 - 0.2, a), (t16, (a + b) / 2), (eye - 0.2, b))
    # the red glow grows: a timpani roll from the 16ths
    hold(S, "timp_roll", "D2", t16, eye + 0.05)
    dyn(S, "timp_roll", (t16, 0.1), (eye, 0.3))


def eye(S, bm):
    ev = bm.ev
    e0, slit, mir = ev("eye_burn"), ev("slit_nothing"), ev("mirror")
    cut = slit + 1.5
    # the tam-tam: its wash swells from the Deep's 16ths to the slit, then its bloom (no stroke)
    S.P("tam").n(60, ev("tick_16") + 1.0, slit - ev("tick_16") - 1.0, 0.72, rel=6.0, curve=1.8)
    # THE RING in the trombones (the bass trombone an octave under, the tuba on the pedal)
    put(S, "tbn", ring_notes("D3", e0))
    put(S, "tbn2", ring_notes("D2", e0))
    hold(S, "tbn", "D3", slit, cut, sync=True)                  # the slit: the Ring's D struck again, marcato
    hold(S, "tbn2", "D2", slit, cut)
    hold(S, "tuba", "D2", e0, cut)
    dyn(S, ("tbn", "tbn2"), (e0 - 0.05, 0.3), (slit - 0.3, 0.8), (slit, 0.88), (cut, 0.85))
    dyn(S, "tuba", (e0, 0.4), (slit, 0.8), (cut, 0.8))
    S.sync.append((e0 * BEAT_S, "THE EYE: the Ring motif in the trombones (D3)", "tbn", 0.2, "arrive:50"))
    # the horns and the tremolo strings on the tritone, swelling into the slit
    voiced(S, [(e0, {"hn": ["D4"], "hn2": ["Ab4"], "hn3": ["D5"]}),
               (slit, {"hn": ["D4"], "hn2": ["Ab4"], "hn3": ["D5"]})], cut)
    dyn(S, ("hn", "hn2", "hn3"), (e0 - 0.05, 0.24), (slit - 0.2, 0.78), (slit, 0.86), (cut, 0.84))
    voiced(S, [(e0, {"vln_trem": ["D5", "Ab5"], "vla_trem": ["D4", "Ab4"], "vc_trem": ["D3", "Ab3"],
                     "cb_trem": ["D2"]}),
               (slit, {"vln_trem": ["D5", "Ab5", "D6"], "vla_trem": ["D4", "Ab4"]})], cut)
    dyn(S, ("vln_trem", "vla_trem"), (e0 - 0.05, 0.2), (slit - 0.2, 0.58), (slit, 0.62), (cut, 0.6))
    dyn(S, ("vc_trem", "cb_trem"), (e0 - 0.05, 0.2), (slit - 0.6, 0.5), (slit, 0.3))
    # the weight at the slit on sustained cellos and basses (the tremolo's bow strokes are too peaky to carry it)
    hold(S, "vc", "D3", e0, cut)
    hold(S, "cb", "D2", e0, cut)
    dyn(S, ("vc", "cb"), (e0 - 0.05, 0.28), (slit - 0.2, 0.76), (slit, 0.82), (cut, 0.8))
    hold(S, "timp_roll", "D2", e0 + 0.05, cut, legato=True)
    dyn(S, "timp_roll", (e0, 0.22), (slit, 0.72), (cut, 0.7))
    S.sync.append((slit * BEAT_S, "the slit opens onto nothing: the Ring's D struck (trombones)", "tbn", 0.25,
                   "arrive:50"))


def mirror(S, bm):
    ev = bm.ev
    mir, dr, grasp = ev("mirror"), ev("drop"), ev("claw")
    back = dr + 2.4                         # the golden dawn lasts one breath (to f2248)
    # one high held tone: a far string harmonic, doubled
    S.P("harmonic").n("A5", mir + 0.3, grasp - 0.6 - (mir + 0.3), atk=1.8, rel=1.5)
    S.P("harmonic").d((mir + 0.3, 0.2), (dr, 0.22), (back, 0.23), (grasp - 0.6, 0.18))
    S.P("harm_str").n("A5", mir + 0.8, grasp - 0.8 - (mir + 0.8))
    S.P("harm_str").d((mir + 0.8, 0.06), (dr, 0.09), (grasp - 0.8, 0.05))
    # harp harmonics: the lands burning, reflected (the Ring's D and Ab) ...
    for p, t, v in (("D6", mir + 1.0, 0.24), ("Ab5", mir + 2.5, 0.22), ("D5", mir + 4.0, 0.2),
                    ("Ab5", mir + 5.2, 0.2),
                    # ... the drop: for one breath a golden dawn (D major) ...
                    ("F#5", dr + 0.12, 0.26), ("A5", dr + 0.5, 0.24), ("D6", dr + 0.9, 0.24), ("F#6", dr + 1.5, 0.2),
                    # ... then the ripples carry the fire back over it
                    ("Ab5", back + 0.2, 0.22), ("D5", back + 1.4, 0.2), ("Ab5", back + 2.6, 0.17)):
        S.P("hharm").n(p, t, 2.0, v)
    S.sync.append(((dr + 0.12) * BEAT_S, "the drop: the golden dawn in the water (harp F#5)", "hharm", 0.02, "hit"))
    voiced(S, [(dr + 0.1, {"vla_q": ["D4"], "vln2_q": ["F#4", "A4"], "vln1_q": ["E5"]})], back)
    dyn(S, ("vla_q", "vln2_q", "vln1_q"), (dr, 0.03), (dr + 1.1, 0.13), (back - 0.6, 0.08), (back, 0.02))


def grasp(S, bm):
    ev = bm.ev
    claw, closes, crust, slip, black = ev("claw"), ev("claw_closes"), ev("crust_cracks"), ev("slip"), ev("black")
    # the riser, tremolo and a timpani roll into the claw ...
    S.P("riser").n(60, claw, closes - claw, 0.55, f0=160.0, f1=7000.0, curve=2.2)
    S.P("revcym").n(60, claw + 0.6, closes - (claw + 0.6), 0.6)
    voiced(S, [(claw, {"vc_trem": ["D3", "Ab3"], "vla_trem": ["D4", "Ab4"]})], closes)
    dyn(S, ("vc_trem", "vla_trem"), (claw - 0.05, 0.2), (closes - 0.15, 0.6), (closes, 0.05))
    hold(S, "timp_roll", "D2", claw + 0.1, closes)
    dyn(S, "timp_roll", (claw, 0.14), (closes - 0.1, 0.6), (closes, 0.05))
    # ... the suck as it closes (30 b3); then nothing but gold leaking between the fingers
    for p, t, v in (("Ab5", closes + 0.5, 0.22), ("F5", closes + 1.1, 0.19), ("D5", closes + 1.6, 0.17)):
        S.P("glass").n(p, t, 1.0, v)
    hold(S, "cb_trem", "D2", closes + 0.4, crust + 0.2)
    dyn(S, "cb_trem", (closes + 0.3, 0.06), (crust, 0.14))
    # the crust glows from inside and cracks: a high tremolo swell and the stopped horn's Ab
    voiced(S, [(crust, {"vln_trem": ["Ab5", "D6"], "vla_trem": ["Eb5"], "cb_trem": ["D2"]})], slip)
    dyn(S, ("vln_trem", "vla_trem"), (crust - 0.05, 0.14), (slip - 0.15, 0.55), (slip, 0.05))
    dyn(S, "cb_trem", (crust, 0.16), (slip - 0.2, 0.3), (slip, 0.05))
    hold(S, "hn_st", "Ab4", crust + 0.2, slip)
    dyn(S, "hn_st", (crust + 0.1, 0.3), (slip - 0.2, 0.55), (slip, 0.1))
    # the slip: no impact; one glass tone falls away into silence
    S.P("gfall").n("A6", slip, black + 1.6 - slip, 0.42, drop=26.0, curve=1.3, decay=0.4, sync=True)
    S.sync.append((slip * BEAT_S, "the slip: one glass tone falls away (no impact)", "gfall", 0.02, "hit"))


def old_law(S, bm):
    ev = bm.ev
    black, cock, falls = ev("black"), ev("cock"), ev("ring_falls")
    # the Ring loop alone, low and unresolved: clarinet (chalumeau) and cellos in unison, slow, hanging on the Ab
    t0 = black + 2.0
    ns = ring_notes("D3", t0, unit=1.5)
    ns[-1] = (ns[-1][0], ns[-1][1], 2.6)
    put(S, "cl_c", ns)
    put(S, "vc_ring", ns)
    for pn, v in (("cl_c", 0.22), ("vc_ring", 0.2)):
        S.P(pn).d((t0 - 0.1, v), (t0 + 3, v * 1.05), (t0 + 6, v * 0.95), (t0 + 8.2, 0.03))
    S.sync.append((t0 * BEAT_S, "black: the Ring loop alone, low (clarinet D3)", "cl_c+vc_ring", 0.2, "arrive:50"))
    # harp harmonics; the cock far away under the last
    for p, t, v in (("D5", black + 0.6, 0.2), ("Ab5", black + 4.4, 0.18), ("D5", black + 8.0, 0.18),
                    ("D6", cock - 0.45, 0.2)):
        S.P("hharm").n(p, t, 2.0, v)


# ---------------------------------------------------------------------------
# BOOK TWO (C13-C16): the fall, the find, the fire test, the reveal
# ---------------------------------------------------------------------------
def ring_falls(S, bm):
    ev = bm.ev
    rf, star, snow = ev("ring_falls"), ev("falling_star"), ev("snow")
    S.P("gfall").n("D6", rf, star + 0.3 - rf, 0.36, drop=12.0, curve=1.1, decay=0.9, tumble=1.3)
    S.P("gfall2").n("A6", star, snow - 0.15 - star, 0.4, drop=19.0, curve=1.2, decay=0.7)
    S.sync.append((rf * BEAT_S, "the glass tone takes up its fall", "gfall", 0.02, "hit"))
    S.sync.append((star * BEAT_S, "the streak: a brighter, faster fall", "gfall2", 0.02, "hit"))
    # the moonlit ink range: a cold open fifth, gone before the strikes
    voiced(S, [(star, {"cb_q": ["D2"], "vc_q": ["A2"], "vla_q": ["E3"]})], G(38) - 0.4)
    dyn(S, ("cb_q", "vc_q", "vla_q"), (star - 0.2, 0.03), (star + 1.5, 0.13), (snow + 1, 0.12), (G(38) - 0.5, 0.02))


def the_find(S, bm):
    ev = bm.ev
    vis, fist = ev("vision"), ev("fist")
    # the vision: THE RING in the violas, ppp and sweet (its Ab heard as G#, the #11 of a D7 halo); cut at the fist
    put(S, "vla_q", [(m("D4"), vis, 0.75), (m("G#4"), vis + 0.75, 0.75), (m("D5"), vis + 1.5, fist - vis - 1.5 + 0.02)],
        rel=0.14)
    S.P("vla_q").d((vis - 0.1, 0.16), (vis + 1.5, 0.18), (fist, 0.16))
    voiced(S, [(vis - 0.05, {"vc_q": ["D3"], "vln2_q": ["F#4", "C5"], "vln1_q": ["E5"]})], fist + 0.02, rel=0.14)
    dyn(S, ("vc_q", "vln2_q", "vln1_q"), (vis - 0.3, 0.03), (vis + 0.8, 0.1), (fist, 0.1))
    S.sync.append((vis * BEAT_S, "the vision: THE RING in the violas, ppp (D4)", "vla_q", 0.2, "arrive:62"))


def fire_test(S, bm):
    ev = bm.ev
    roar, steel, tips, fist2, rev = ev("roar"), ev("steel"), ev("steel_tips"), ev("fist_again"), ev("reveal")
    # a lone horn cries the CALL on the roar, over a low D
    line(S, "hn", [("D4", 1.0), ("A4", 1.0), ("D5", 2.6)], roar)
    S.P("hn").d((roar - 0.05, 0.22), (roar + 1, 0.3), (roar + 2.5, 0.42), (roar + 4.6, 0.04))
    S.sync.append((roar * BEAT_S, "the roar: a lone horn cries the CALL (D4)", "hn", 0.2, "arrive:62"))
    voiced(S, [(roar + 1.0, {"cb_q": ["D2"], "vc_q": ["D3"]}), (rev, {"cb_q": ["D2"], "vc_q": ["A2", "D3"],
                                                                 "vla_q": ["A3"]}),
               (rev + 8, {"cb_q": ["D2"], "vc_q": ["A2", "D3"], "vla_q": ["A3"], "vln2_q": ["E4"]})], G(49) + 0.1)
    dyn(S, ("cb_q", "vc_q"), (roar - 0.1, 0.05), (roar + 1.5, 0.18), (rev, 0.16), (G(49), 0.22))
    dyn(S, ("vla_q", "vln2_q"), (rev - 0.1, 0.05), (rev + 2, 0.18), (G(49), 0.24))
    # THE RING under the flames (bassoon + cellos): at the tip it should fall, and the loop holds its D; at the fist
    # it closes
    ns = ring_notes("D3", steel) + [(m("D3"), tips, 1.0), (m("Ab3"), tips + 1, 1.0), (m("D4"), tips + 2, 2.0),
                                    (m("C4"), fist2, 0.5), (m("Ab3"), fist2 + 0.5, 1.6)]
    put(S, "bsn_c", ns)
    put(S, "vc_ring", ns)
    for pn, v in (("bsn_c", 0.3), ("vc_ring", 0.28)):
        S.P(pn).d((steel - 0.1, v), (tips + 2, v * 1.1), (fist2, v), (fist2 + 2.1, 0.03))
    S.sync.append((steel * BEAT_S, "the Ring under the flames (bassoon + cellos, D3)", "bsn_c+vc_ring", 0.2,
                   "arrive:50"))


def reveal(S, bm):
    ev = bm.ev
    rev = ev("reveal")
    # the call echoes off the ranges, farther each time
    for k, (pn, v) in enumerate((("hn_far", 0.46), ("hn_farther", 0.44), ("hn_far2", 0.42))):
        t = rev + 4 * k
        line(S, pn, [("D4", 1.0), ("A4", 1.0), ("D5", 2.2)], t)
        S.P(pn).d((t - 0.05, v), (t + 1.5, v * 1.02), (t + 3.2, v * 0.9), (t + 4.3, 0.03))
        S.sync.append((t * BEAT_S, f"the call echoes off the ranges ({k + 1})", pn, 0.25, "arrive:62"))


# ---------------------------------------------------------------------------
# BOOK THREE (C17-C18): the Living Ink Run, the map, the Road
# ---------------------------------------------------------------------------
RUN_HARM = [  # (beat offset from the run, bass, chord pcs, violin top)
    (0, "D", "D F A", "A4"), (2, "D", "D F A", "A4"), (4, "Bb", "Bb D F", "Bb4"), (6, "F", "F A C", "C5"),
    (8, "G", "G Bb D", "D5"), (10, "A", "A C E", "E5"), (12, "Bb", "Bb D F", "F5"), (14, "C", "C E G", "G5")]
RUN_BASS = {"D": ("D2", "D3"), "Bb": ("Bb1", "Bb2"), "F": ("F1", "F2"), "G": ("G1", "G2"), "A": ("A1", "A2"),
            "C": ("C2", "C3")}
RUN_TIMP = {"D": "D2", "Bb": "Bb2", "F": "F2", "G": "G2", "A": "A2", "C": "C3"}


def _tones(pcs, lo, hi):
    s = {m(x + "4") % 12 for x in pcs.split()}
    return [p for p in range(m(lo), m(hi) + 1) if p % 12 in s]


def run(S, bm):
    ev = bm.ev
    r0, mp = ev("run"), ev("map")
    bea = [ev(f"beacon_{k}") for k in range(1, 8)]
    for i, (off, bass, pcs, top) in enumerate(RUN_HARM):
        t = r0 + off
        t1 = r0 + RUN_HARM[i + 1][0] if i + 1 < len(RUN_HARM) else mp
        cb, vc = RUN_BASS[bass]
        S.P("cb").n(cb, t, t1 - t + 0.06, legato=i > 0, sync=(i == 0))
        S.P("vc").n(vc, t, t1 - t + 0.06, legato=i > 0)
        S.P("vln1").n(top, t, t1 - t + 0.06, legato=i > 0)
        S.P("vln1").n(m(top) - 12, t, t1 - t + 0.06, legato=i > 0)
        # the 16th ostinato, grouped 3+3+2 per two beats; its accents on the group's 3 and 6, never on the
        # beacon itself (the fire owns it); the violas low, the 2nd violins an octave up
        tn = _tones(pcs, "A3", "A4")
        pat = [tn[0], tn[1], tn[2], tn[1], tn[0], tn[2], tn[1], tn[2]]
        for q in range(1, 8):                     # the group's first 16th is the beacon's: a rest
            tt = t + 0.25 * q
            a = 1 if q in (3, 6) else 0
            v = 0.3 + 0.02 * i + 0.16 * a
            S.P("vla_sp").n(pat[q], tt, 0.25, v)
            S.P("vln2_sp").n(pat[q] + 12, tt, 0.25, v - 0.02)
        # the timpani on the 3 and the 6 only (and a light 16th before the 6)
        tp = RUN_TIMP[bass]
        S.P("timp").n(tp, t + 0.75, 0.5, 0.33 + 0.02 * i)
        S.P("timp").n(tp, t + 1.25, 0.25, 0.19 + 0.01 * i)
        S.P("timp").n(tp, t + 1.5, 0.5, 0.36 + 0.02 * i)
    dyn(S, "cb", (r0 - 0.1, 0.31), (mp - 0.3, 0.45))
    dyn(S, "vc", (r0 - 0.1, 0.34), (mp - 0.3, 0.49))
    dyn(S, "vln1", (r0 - 0.1, 0.28), (mp - 0.3, 0.47))
    # THE CALL passed peak to peak (beacons 1, 3, 5, 7: on D, F, A, C), each horn farther
    for k, (pn, root, v) in enumerate((("hn", "D4", 0.58), ("hn2", "F3", 0.6), ("hn_far", "A3", 0.62),
                                       ("hn_farther", "C4", 0.64))):
        t = bea[2 * k]
        line(S, pn, [(root, 0.75), (m(root) + 7, 0.75), (m(root) + 12, 2.3)], t)
        S.P(pn).d((t - 0.05, v), (t + 1.5, v * 1.04), (t + 3.2, v * 0.9), (t + 3.9, 0.04))
        S.sync.append((t * BEAT_S, f"beacon {2 * k + 1}: the CALL on {root[:-1]} ({pn})", pn, 0.25,
                       f"arrive:{m(root)}"))
    # THE ANSWER, first heard: beacon 2, one far horn, over Bb (D C Bb F)
    t = bea[1]
    K.answer(S, "hn_ans", "D4", t, mode="minor", rhythm=(1, .5, .5, 1.8))
    S.P("hn_ans").d((t - 0.05, 0.56), (t + 1.2, 0.58), (t + 3.2, 0.48), (t + 4.2, 0.04))
    S.sync.append((t * BEAT_S, "beacon 2: THE ANSWER, first heard (far horn, D4)", "hn_ans", 0.25, "arrive:62"))


def map_answers(S, bm):
    ev = bm.ev
    mp, road, council = ev("map"), ev("road_arrives"), ev("council")
    # wonder in F major: the strings' chords (F, C/E, Dm, Bb; Gm7, F/A, Bb, C7sus4-C7; A7sus4-A7)
    seq = [(mp, {"cb": ["F1"], "vc": ["F2", "C3"], "vla": ["A3", "C4"], "vln2": ["F4", "A4"]}),
           (mp + 2, {"cb": ["E2"], "vc": ["C3", "G3"], "vla": ["C4"], "vln2": ["E4", "G4"]}),
           (mp + 4, {"cb": ["D2"], "vc": ["D3", "A3"], "vla": ["D4"], "vln2": ["F4", "A4"]}),
           (mp + 6, {"cb": ["Bb1"], "vc": ["F2", "D3"], "vla": ["D4"], "vln2": ["F4", "Bb4"]}),
           (mp + 8, {"cb": ["G1"], "vc": ["D3", "F3"], "vla": ["Bb3"], "vln2": ["D4", "F4"], "vln1": ["Bb4"]}),
           (mp + 10, {"cb": ["A1"], "vc": ["C3", "F3"], "vla": ["A3"], "vln2": ["C4", "F4"], "vln1": ["A4"]}),
           (mp + 11, {"cb": ["Bb1"], "vc": ["D3", "F3"], "vla": ["Bb3"], "vln2": ["D4", "F4"], "vln1": ["Bb4"]}),
           (mp + 12, {"cb": ["C2"], "vc": ["C3", "G3"], "vla": ["Bb3"], "vln2": ["F4", "G4"], "vln1": ["C5"]}),
           (mp + 13, {"cb": ["C2"], "vc": ["C3", "G3"], "vla": ["Bb3"], "vln2": ["E4", "G4"], "vln1": ["C5"]}),
           (road, {"cb": ["A1"], "vc": ["A2", "E3"], "vla": ["A3"], "vln2": ["D4", "G4"], "vln1": ["E5"]}),
           (road + 1, {"cb": ["A1"], "vc": ["A2", "E3"], "vla": ["A3"], "vln2": ["C#4", "G4"], "vln1": ["E5"]})]
    voiced(S, seq, council + 0.1)
    for pn, a, b in (("cb", 0.26, 0.21), ("vc", 0.26, 0.23), ("vla", 0.28, 0.24), ("vln2", 0.28, 0.24),
                     ("vln1", 0.3, 0.24)):
        S.P(pn).d((mp - 0.05, a * 0.5), (mp + 1.6, a), (mp + 8, a * 0.95), (road, b), (council - 0.3, b * 0.6),
                  (council + 0.1, 0.03))
    S.sync.append((mp * BEAT_S, "THE MAP: the strings' F major blooms (violas: its onset)", "vla", 0.25, "bloom"))
    # the violins sing the CALL on F (their own voice, no solo), the oboe answers (F E D A, a clarinet under it)
    line(S, "vln1", [("F4", 1.0), ("C5", 1.0), ("F5", 2.0)], mp + 1, legato=True)
    dyn(S, "vln1", (mp + 0.95, 0.34), (mp + 3, 0.36), (mp + 5, 0.3))
    lines(S, ("ob_c", "ob_dbl"), [("F5", 1.0), ("E5", 0.5), ("D5", 0.5), ("A4", 2.2)], mp + 4)
    for pn, v in (("ob_c", 0.36), ("ob_dbl", 0.2)):
        S.P(pn).d((mp + 3.9, v), (mp + 5, v * 1.04), (mp + 6.5, v * 0.9), (mp + 8.3, 0.03))
    S.sync.append(((mp + 4) * BEAT_S, "the map: the oboe answers (F5)", "ob_c", 0.2, "arrive:77"))
    # the solo violin's descant above (doubled): it floats over the answer, then walks the Road down to the stones
    desc = [("A5", 2.0), ("C6", 2.0), ("Bb5", 2.0), ("A5", 2.0), ("G5", 1.0), ("F5", 1.0), ("E5", 1.0), ("D5", 1.0),
            ("E5", 2.0), ("E5", 2.1)]
    lines(S, ("svln", "svln2"), desc, mp + 4)
    for pn, v in (("svln", 0.3), ("svln2", 0.24)):
        S.P(pn).d((mp + 3.9, v * 0.8), (mp + 6, v), (mp + 10, v * 1.05), (road, v), (council + 0.1, 0.04))
    # the damped harp: arpeggios running hill by hill (8ths, each note stopped short)
    arps = [(mp + 0.5, ("F3", "A3", "C4", "F4", "A4", "C5", "F5")), (mp + 4.5, ("D3", "F3", "A3", "D4", "F4", "A4", "D5")),
            (mp + 8.5, ("G3", "Bb3", "D4", "F4", "G4"))]
    for t, ps in arps:
        for k, p in enumerate(ps):
            S.P("harp").n(p, t + 0.5 * k, 0.5, 0.22 + 0.01 * k, maxlen=0.4)
    # THE ROAD: a slow pizzicato walk, one dot a beat, to the ring of stones (56 b3)
    walk = ["F2", "G2", "A2", "Bb2", "C3", "C3"]
    for k, p in enumerate(walk):
        t = mp + 8 + k
        S.P("vc_pz").n(p, t, 1.0, 0.3 + 0.01 * k)
        S.P("cb_pz").n(m(p) - 12, t, 1.0, 0.28 + 0.01 * k)
    S.P("vc_pz").n("A2", road, 1.0, 0.34, sync=True)
    S.P("cb_pz").n("A1", road, 1.0, 0.32)
    S.sync.append((road * BEAT_S, "the Road reaches the ring of stones (pizz A2)", "vc_pz", 0.02, "hit"))


# ---------------------------------------------------------------------------
# BOOK FOUR (C19-C23): the council, the Ring brought out, the bearer, the unmaking, the fire that remains
# ---------------------------------------------------------------------------
CHORALE = [  # the council's chorale (hn, hn2, tbn, tuba: top to bottom); its first chord Bb/D returns at the end
    (G(57, 3), ["D4", "Bb3", "F3", "D2"]),      # Bb/D
    (G(58, 1), ["D4", "Bb3", "G3", "Bb1"]),     # Gm/Bb
    (G(58, 3), ["D4", "A3", "F3", "F1"]),       # Dm/F
    (G(59, 1), ["C4", "G3", "E3", "C2"]),       # C
    (G(59, 3), ["C4", "A3", "F3", "A1"]),       # F/A
    (G(60, 1), ["D4", "Bb3", "F3", "G1"]),      # Gm7
    (G(60, 3), ["D4", "A3", "E3", "A1"]),       # A sus4
    (G(60, 4), ["C#4", "A3", "E3", "A1"]),      # A
    (G(61, 1), ["D4", "A3", "F3", "D2"]),       # Dm            (C20: moves between the events, never on them)
    (G(61, 4), ["C4", "Ab3", "F3", "D2"]),      # D half-dim    (after the Ring is set on the stone)
    (G(62, 4), ["D4", "Ab3", "F3", "Bb1"]),     # Bb7
    (G(63, 4), ["D4", "C4", "F3", "Ab1"]),      # D half-dim / Ab (the orbit finds the gilded hand at 64 b1)
    (G(64, 4), ["D4", "Ab3", "F3", "Bb1"]),     # Bb7           (C21: against the rising loop, growing)
    (G(65, 3), ["D4", "Ab3", "F3", "C2"]),      # D half-dim / C (her hand on the Ring)
    (G(66, 1), ["Eb4", "B3", "F3", "Db2"]),     # Db9 (the fire rises)
    (G(66, 3), ["Eb4", "B3", "Ab3", "Eb2"]),    # F half-dim / Eb
    (G(67, 1), ["F#4", "D4", "G#3", "E2"]),     # E9
    (G(67, 3), ["Ab4", "D4", "C4", "F#2"]),     # D7b5 / F#
]
CHORALE_PARTS = ("hn", "hn2", "tbn", "tuba")


def council(S, bm):
    ev = bm.ev
    c0, white = ev("council"), ev("white_heart")
    # the clarinet intones the CALL (the violas in unison)
    lines(S, ("cl_c", "cl_dbl"), [("D4", 2.0), ("A4", 2.0), ("D5", 3.7)], c0)
    for pn, v in (("cl_c", 0.34), ("cl_dbl", 0.3)):
        S.P(pn).d((c0 - 0.1, v), (c0 + 2, v * 1.04), (c0 + 4.5, v * 1.08), (c0 + 7.8, 0.04))
    S.sync.append((c0 * BEAT_S, "THE COUNCIL: the clarinet intones the CALL (D4)", "cl_c+cl_dbl", 0.2, "arrive:62"))
    # the brass chorale (pp, half notes) from 57 b3 to the white heart
    for i, (t, ps) in enumerate(CHORALE):
        t1 = CHORALE[i + 1][0] if i + 1 < len(CHORALE) else white
        for pn, p in zip(CHORALE_PARTS, ps):
            S.P(pn).n(p, t, t1 - t + 0.06, legato=i > 0, sync=(i == 0))
    for pn, v in (("hn", 0.2), ("hn2", 0.19), ("tbn", 0.19), ("tuba", 0.18)):
        S.P(pn).d((CHORALE[0][0] - 0.1, v * 0.8), (CHORALE[0][0] + 1.5, v), (ev("ring_set"), v), (ev("gilded_hand"), v),
                  (ev("torches_down"), v * 1.5), (ev("fire_rises"), v * 2.1), (white - 0.4, v * 2.9))


def bring_out(S, bm):
    ev = bm.ev
    rs, tor = ev("ring_set"), ev("torches_down")
    # the Ring loop circles beneath from the moment it is set on the stone (bassoon + cellos)
    ns = []
    for k in range(3):
        ns += ring_notes("D2", rs + 4 * k)
    ns += [(m("D2"), rs + 12, 1.0), (m("Ab2"), rs + 13, tor - (rs + 13) + 0.05)]   # the bearer's loop takes it on
    put(S, "bsn_c", ns)
    put(S, "vc_ring", ns)
    for pn, v in (("bsn_c", 0.24), ("vc_ring", 0.22)):
        S.P(pn).d((rs - 0.1, v), (rs + 8, v * 1.05), (tor, v * 1.1))
    S.sync.append((rs * BEAT_S, "the Ring set on the stone: its loop circles beneath (D2)", "bsn_c+vc_ring", 0.25,
                   "arrivew"))


def bearer(S, bm):
    ev = bm.ev
    tor, white = ev("torches_down"), ev("white_heart")
    # the Ring motif rises in the low strings, tremolo: a minor third each cycle (D, F, Ab)
    ns = []
    for k, r in enumerate(("D3", "F3", "Ab3")):
        ns += ring_notes(r, tor + 4 * k)
    put(S, "vc_trem", ns)
    put(S, "cb_trem", [(p - 12, t, d) for p, t, d in ns])
    dyn(S, ("vc_trem", "cb_trem"), (tor - 0.05, 0.28), (tor + 4, 0.36), (tor + 8, 0.44), (white - 0.3, 0.52))
    S.sync.append((tor * BEAT_S, "the bearer: the Ring rises in the low strings, tremolo (D3)", "vc_trem", 0.2,
                   "arrive:50"))


def unmaking(S, bm):
    ev = bm.ev
    white, flare, unmade, fr = ev("white_heart"), ev("letters_flare"), ev("unmade"), ev("eye_falls")
    # the white heart: the orchestra falls away into the white; only the Ring loop is left (the low strings,
    # tremolo), with the seethe of the melt, and it HEALS (D Ab D' C#' A): the C# and the A as the letters flare;
    # the breath cuts the A
    heal = ring_notes("D3", white, heal=True)
    put(S, "vc_trem", heal)
    put(S, "cb_trem", [(p - 12, t, d) for p, t, d in heal])
    for pn, v in (("vc_trem", 0.28), ("cb_trem", 0.24)):
        S.P(pn).d((white - 0.05, v), (flare, v * 1.2), (unmade - 0.35, v * 1.3))
    # 69 b1: the Ring loop cadences into D, once: D major, full and warm, settling
    DMAJ = {"cb": ["D2"], "vc": ["D3", "A3"], "vla": ["F#3", "D4"], "vln2": ["A4"], "vln1": ["D5", "F#5"],
            "tbn": ["D3"], "tbn2": ["A2"], "tuba": ["D2"], "hn": ["F#4"], "hn2": ["A3"], "hn3": ["D4"]}
    for pn, ps in DMAJ.items():
        for p in ps:
            S.P(pn).n(p, unmade, fr - unmade + 0.08, sync=True)
    for pn, v in (("cb", 0.5), ("vc", 0.52), ("vla", 0.5), ("vln2", 0.5), ("vln1", 0.5), ("tbn", 0.44), ("tbn2", 0.4),
                  ("tuba", 0.44), ("hn", 0.46), ("hn2", 0.44), ("hn3", 0.46)):
        S.P(pn).d((unmade - 0.05, v * 0.85), (unmade + 0.8, v), (unmade + 2.2, v * 0.9), (fr - 0.3, v * 0.62),
                  (fr + 0.1, v * 0.6))
    S.sync.append((unmade * BEAT_S, "the Ring loop cadences into D, once (after the breath)", "score", 0.15, "bloom"))


def fire_remains(S, bm):
    ev = bm.ev
    fr, br = ev("eye_falls"), ev("breath_dawn")
    mapb = fr + 4                                # 71 b1: the burn-through to the map, the roads run outward
    end = br + 0.02
    # bar 70: D major settles, warm and steady, into the quiet strings and one horn
    S.P("hn2").n("A3", fr, mapb - fr, legato=True)
    voiced(S, [(fr, {"cb_q": ["D2"], "vc_q": ["A2", "D3"], "vla_q": ["F#3", "A3"], "vln2_q": ["D4"], "vln1_q": ["F#4"]}),
               (fr + 2, {"cb_q": ["D2"], "vc_q": ["A2", "D3"], "vla_q": ["F#3", "A3"], "vln2_q": ["E4"],
                         "vln1_q": ["F#4"]}),
               # bar 71: G/B - D/A - A: the tritone stays resolved
               (mapb, {"cb_q": ["B1"], "vc_q": ["G2", "D3"], "vla_q": ["G3", "B3"], "vln2_q": ["D4"], "vln1_q": ["G4"]}),
               (mapb + 2, {"cb_q": ["A1"], "vc_q": ["A2", "D3"], "vla_q": ["F#3", "A3"], "vln2_q": ["D4"],
                           "vln1_q": ["F#4"]}),
               (mapb + 3, {"cb_q": ["A1"], "vc_q": ["A2", "E3"], "vla_q": ["G3", "C#4"], "vln2_q": ["E4"],
                           "vln1_q": ["F#4"]})], end, sync_first=True)
    dyn(S, ("cb_q", "vc_q", "vla_q", "vln2_q", "vln1_q"), (fr - 0.1, 0.27), (fr + 1.5, 0.25), (mapb, 0.23),
        (mapb + 2, 0.23), (br - 0.2, 0.26))
    dyn(S, "hn2", (fr - 0.1, 0.26), (fr + 2, 0.24), (mapb - 0.2, 0.16), (mapb + 0.1, 0.03))
    # the ANSWER passed outward: horn to farther horn, in canon, each farther and softer (the breath cuts the last)
    for k, (pn, v) in enumerate((("hn", 0.34), ("hn_far", 0.32), ("hn_farther", 0.3))):
        t = mapb + k
        K.answer(S, pn, "D4", t, mode="major", rhythm=(1, .5, .5, 1.5))
        S.P(pn).d((t - 0.05, v), (t + 0.8, v * 1.03), (t + 2.3, v * 0.9), (t + 3.4, 0.05))
        S.sync.append((t * BEAT_S, f"the fire remains: the ANSWER passed outward ({pn}, D4)", pn, 0.25, "arrive:62"))


# ---------------------------------------------------------------------------
# the DAWN and the EPILOGUE (C24-C28)
# ---------------------------------------------------------------------------
def illumination(S, bm):
    ev = bm.ev
    sun, home, plenty = ev("sunrise"), ev("home"), ev("plenty")
    # the harmony: D add9 (bloom) | G add9/D | D/F# | Bm7 | A/C# G | D/F# | Em7 | A7sus4 A7 | HOME over G D/F# Em9 D
    H = [(sun, {"cb": ["D2"], "vc": ["A2", "D3"], "vla": ["F#3", "A3"], "vln2": ["E4"], "hn2": ["A3"], "hn3": ["F#4"]}),
         (sun + 4, {"cb": ["D2"], "vc": ["G2", "D3"], "vla": ["G3", "B3"], "vln2": ["E4"], "hn2": ["B3"], "hn3": ["G4"]}),
         (sun + 6, {"cb": ["F#2"], "vc": ["A2", "D3"], "vla": ["F#3", "A3"], "vln2": ["D4"], "hn2": ["A3"], "hn3": ["F#4"]}),
         (sun + 8, {"cb": ["B1"], "vc": ["F#2", "D3"], "vla": ["F#3", "B3"], "vln2": ["D4"], "hn2": ["B3"], "hn3": ["F#4"]}),
         (sun + 10, {"cb": ["C#2"], "vc": ["A2", "E3"], "vla": ["E3", "A3"], "vln2": ["C#4"], "hn2": ["A3"], "hn3": ["E4"]}),
         (sun + 11, {"cb": ["G1"], "vc": ["G2", "D3"], "vla": ["D3", "G3"], "vln2": ["B3"], "hn2": ["G3"], "hn3": ["D4"]}),
         (sun + 12, {"cb": ["F#2"], "vc": ["A2", "D3"], "vla": ["D3", "A3"], "vln2": ["D4"], "hn2": ["A3"], "hn3": ["D4"]}),
         (sun + 14, {"cb": ["E2"], "vc": ["G2", "D3"], "vla": ["E3", "B3"], "vln2": ["D4"], "hn2": ["G3"], "hn3": ["E4"]}),
         (sun + 15, {"cb": ["A1"], "vc": ["A2", "E3"], "vla": ["D3", "G3"], "vln2": ["D4"], "hn2": ["A3"], "hn3": ["E4"]}),
         (sun + 15.5, {"cb": ["A1"], "vc": ["A2", "E3"], "vla": ["C#3", "G3"], "vln2": ["E4"], "hn2": ["A3"], "hn3": ["E4"]}),
         (home, {"cb": ["G1"], "vc": ["G2", "D3"], "vla": ["D3", "G3"], "vln2": ["D4"], "hn2": ["G3"], "hn3": ["D4"]}),
         (home + 2, {"cb": ["F#2"], "vc": ["A2", "D3"], "vla": ["D3", "F#3"], "vln2": ["D4"], "hn2": ["A3"], "hn3": ["D4"]}),
         (home + 4, {"cb": ["E2"], "vc": ["G2", "D3"], "vla": ["E3", "B3"], "vln2": ["D4"], "hn2": ["G3"], "hn3": ["B3"]}),
         (home + 6, {"cb": ["D2"], "vc": ["A2", "D3"], "vla": ["F#3", "A3"], "vln2": ["D4"], "hn2": ["A3"], "hn3": ["F#4"]})]
    voiced(S, H, plenty + 0.1, sync_first=True)
    for pn, a in (("cb", 0.3), ("vc", 0.32), ("vla", 0.31), ("vln2", 0.31), ("hn2", 0.28), ("hn3", 0.28)):
        S.P(pn).d((sun - 0.05, a * 0.55), (sun + 1.5, a), (sun + 8, a * 1.12), (sun + 12, a * 1.2), (home, a * 1.15),
                  (home + 6, a * 0.9), (plenty - 0.3, a * 0.6), (plenty + 0.1, 0.03))
    S.sync.append((sun * BEAT_S, "THE ILLUMINATION: the bloom out of the silence (D add9)", "score", 0.15, "bloom"))
    # the timpani roll that begins in the light, pp to mf, no stroke
    hold(S, "timp_roll", "D2", sun + 0.4, plenty - 1.0)
    dyn(S, "timp_roll", (sun + 0.4, 0.06), (sun + 6, 0.2), (sun + 12, 0.36), (home, 0.3), (home + 6, 0.14),
        (plenty - 1.0, 0.04))
    # THE WHOLE THEME at last: the CALL on one horn ...
    c = sun + 2
    line(S, "hn", [("D4", 2.0), ("A4", 2.0), ("D5", 2.2)], c)
    S.P("hn").d((c - 0.05, 0.44), (c + 2, 0.46), (c + 4, 0.5), (c + 6.2, 0.2))
    S.sync.append((c * BEAT_S, "the dawn: the CALL on one horn (D4)", "hn", 0.2, "arrive:62"))
    # ... the ANSWER and HOME in octave violins, a solo violin above (doubled): home on 76 b1
    a = sun + 8
    theme = [("D5", 2.0), ("C#5", 1.0), ("B4", 1.0), ("F#4", 4.0), ("B4", 2.0), ("A4", 2.0), ("F#4", 2.0), ("D4", 2.1)]
    lines(S, ("vln1", "svln", "svln2"), theme, a, octaves=(0, 1, 1))
    for pn, v in (("vln1", 0.5), ("svln", 0.4), ("svln2", 0.33)):
        S.P(pn).d((a - 0.1, v * 0.9), (a + 2, v), (a + 6, v * 1.05), (home, v * 1.02), (home + 6, v * 0.9),
                  (plenty - 0.2, v * 0.6), (plenty + 0.1, 0.03))
    # measured per layer (as B's HOME): the octave violins and the solo violin's lead; its quiet double (6 dB under)
    # plateaus within 1-3 dB of the probe's threshold for 0.3 s, so its arrival is not a stable measure: its onset is
    # aligned to the lead's instead (ANTIC_SET, from a shift simulation on the rendered stem)
    S.sync.append((a * BEAT_S, "the dawn: the ANSWER in the octave violins (D5)", "vln1", 0.2, "arrivew"))
    S.sync.append((a * BEAT_S, "the dawn: the ANSWER, the solo violin above (lead, D6)", "svln", 0.2, "arrivew"))
    S.sync.append((home * BEAT_S, "the line settles HOME (B4, 76 b1)", "vln1", 0.2, "pitch:71"))
    # the horns in counterpoint under the line (hn: after its call)
    cp = [("A4", 2.0), ("A4", 1.0), ("G4", 1.0), ("F#4", 2.0), ("E4", 1.0), ("E4", 1.0), ("D4", 2.0), ("D4", 2.0),
          ("C#4", 2.0), ("D4", 2.1)]
    line(S, "hn", cp, a, sync_first=False)
    S.P("hn").d((a - 0.05, 0.26), (a + 4, 0.3), (home, 0.3), (plenty - 0.3, 0.2))


HYMN = [("F#4", 1), ("A4", 1), ("B4", 1), ("A4", 1), ("D5", 1), ("C#5", 1), ("B4", 1), ("A4", 1),
        ("G4", 1), ("F#4", 1), ("E4", 1), ("D4", 1.1)]


def plenty(S, bm):
    ev = bm.ev
    p0, hv = ev("plenty"), ev("havens")
    # a plain hymn: the clarinet over three violas (alto, tenor, bass), D major
    line(S, "cl_c", HYMN, p0)
    S.P("cl_c").d((p0 - 0.1, 0.3), (p0 + 4, 0.33), (p0 + 8, 0.3), (hv, 0.2))
    S.sync.append((p0 * BEAT_S, "THE YEAR OF PLENTY: the hymn (clarinet F#4)", "cl_c", 0.2, "arrive:66"))
    alto = [("D4", 2), ("D4", 2), ("F#4", 2), ("D4", 2), ("D4", 2), ("C#4", 1), ("A3", 1.1)]
    tenor = [("A3", 2), ("B3", 2), ("B3", 2), ("B3", 2), ("B3", 2), ("A3", 1), ("F#3", 1.1)]
    bass = [("D3", 2), ("D3", 2), ("D3", 2), ("D3", 2), ("E3", 2), ("E3", 1), ("D3", 1.1)]
    line(S, "vla_q", alto, p0)
    line(S, "hymn_vla2", tenor, p0)
    line(S, "hymn_vla3", bass, p0)
    dyn(S, ("vla_q", "hymn_vla2", "hymn_vla3"), (p0 - 0.1, 0.24), (p0 + 4, 0.26), (hv - 0.3, 0.18), (hv + 0.2, 0.03))


def havens(S, bm):
    ev = bm.ev
    hv, ship, blank = ev("havens"), ev("ship_west"), ev("blank")
    coasts = [ev("coast_1"), ev("coast_2"), ev("coast_3")]
    # an elegy for strings: Bm | G | D/F# | Em7 | Bm/D | Gmaj7 | A7sus4 | D sus2 (quiet sections)
    E = [(hv, {"cb_q": ["B1"], "vc_q": ["F#2", "B2"], "vla_q": ["D3", "F#3"], "vln2_q": ["B3"], "vln1_q": ["F#4"]}),
         (hv + 2, {"cb_q": ["G1"], "vc_q": ["D3", "G3"], "vla_q": ["B3"], "vln2_q": ["D4"], "vln1_q": ["G4"]}),
         (hv + 4, {"cb_q": ["F#2"], "vc_q": ["A2", "D3"], "vla_q": ["A3"], "vln2_q": ["D4"], "vln1_q": ["F#4"]}),
         (hv + 6, {"cb_q": ["E2"], "vc_q": ["G2", "D3"], "vla_q": ["B3"], "vln2_q": ["D4"], "vln1_q": ["G4"]}),
         (hv + 8, {"cb_q": ["D2"], "vc_q": ["B2", "F#3"], "vla_q": ["B3"], "vln2_q": ["D4"], "vln1_q": ["F#4"]}),
         (hv + 10, {"cb_q": ["G1"], "vc_q": ["D3", "G3"], "vla_q": ["B3"], "vln2_q": ["D4"], "vln1_q": ["F#4"]}),
         (ship, {"cb_q": ["A1"], "vc_q": ["E3", "A3"], "vla_q": ["D4"], "vln2_q": ["E4"], "vln1_q": ["G4"]}),
         (ship + 2, {"cb_q": ["D2"], "vc_q": ["A2", "D3"], "vla_q": ["A3"], "vln2_q": ["D4"], "vln1_q": ["E4"]})]
    voiced(S, E, blank + 0.1, sync_first=True)
    dyn(S, ("cb_q", "vc_q", "vla_q", "vln2_q", "vln1_q"), (hv - 0.1, 0.2), (hv + 2, 0.26), (hv + 8, 0.27),
        (ship, 0.24), (ship + 2, 0.18), (blank - 0.4, 0.06), (blank + 0.1, 0.02))
    # the cor anglais carries the CALL (twice), the clarinet in unison
    for c, last in ((hv, 3.0), (hv + 8, 4.5)):
        lines(S, ("ca", "ca_dbl"), [("D4", 1.0), ("A4", 1.0), ("D5", last)], c)
        for pn, v in (("ca", 0.32), ("ca_dbl", 0.29)):
            S.P(pn).d((c - 0.1, v), (c + 2, v * 1.06), (c + 2 + last - 0.8, v * 0.8), (c + 2 + last, 0.04))
        S.sync.append((c * BEAT_S, "the Havens: the cor anglais carries the CALL (D4)", "ca+ca_dbl", 0.2, "arrive:62"))
    # a far horn answers each coast fire, farther and softer each time
    for k, (t, pn, v) in enumerate(zip(coasts, ("hn_far", "hn_farther", "hn_far2"), (0.36, 0.31, 0.26))):
        K.answer(S, pn, "D4", t, mode="major", rhythm=(1, .5, .5, 2))
        S.P(pn).d((t - 0.05, v), (t + 0.8, v * 1.02), (t + 3, v * 0.85), (t + 4.3, 0.03))
        S.sync.append((t * BEAT_S, f"coast fire {k + 1}: a far horn answers ({pn})", pn, 0.25, "arrive:62"))


def last_pages(S, bm):
    ev = bm.ev
    blank, title, plagal = ev("blank"), ev("title"), ev("plagal")
    end = bm.bars * 4
    # the chorale's first chord (Bb/D) returns, pianissimo; then Gm/D ... G/D, and the plagal close into D
    seq = [(blank, {"hn": ["D4"], "hn2": ["Bb3"], "tbn": ["F3"], "tuba": ["D2"]}),
           (blank + 6, {"hn": ["D4"], "hn2": ["Bb3"], "tbn": ["G3"], "tuba": ["D2"]}),
           (title + 4, {"hn": ["D4"], "hn2": ["B3"], "tbn": ["G3"], "tuba": ["D2"]}),
           (plagal, {"hn": ["D4"], "hn2": ["A3"], "tbn": ["F#3"], "tuba": ["D2"]})]
    voiced(S, seq, end - 0.4, sync_first=True)
    for pn, v in (("hn", 0.2), ("hn2", 0.19), ("tbn", 0.19), ("tuba", 0.18)):
        S.P(pn).d((blank - 0.1, v * 0.6), (blank + 1.5, v), (title, v * 0.95), (plagal, v * 1.05), (plagal + 1.5, v),
                  (end - 1.2, v * 0.3), (end - 0.4, 0.02))
    S.sync.append((blank * BEAT_S, "the last pages: the chorale's first chord returns (Bb/D)", "hn2", 0.25,
                   "arrive:58"))
    # the strings join for the close: G/D, then D (quiet sections)
    voiced(S, [(title + 4, {"cb_q": ["D2"], "vc_q": ["G2", "D3"], "vla_q": ["B3"], "vln2_q": ["D4"], "vln1_q": ["G4"]}),
               (plagal, {"cb_q": ["D2"], "vc_q": ["A2", "D3"], "vla_q": ["A3"], "vln2_q": ["D4"], "vln1_q": ["F#4"]})],
           end - 0.4)
    dyn(S, ("cb_q", "vc_q", "vla_q", "vln2_q", "vln1_q"), (title + 3.8, 0.04), (title + 5.5, 0.16),
        (plagal - 0.2, 0.19), (plagal + 1.2, 0.22), (end - 2.0, 0.08), (end - 0.4, 0.02))
    S.sync.append((plagal * BEAT_S, "the plagal close into D (violins I F#4)", "vln1_q", 0.2, "pitch:66"))


# ---------------------------------------------------------------------------
# C's effects beyond the locked cue sheet (the beat sheet's C column), added in memory
# ---------------------------------------------------------------------------
def extra_effects(bm):
    ev = lambda k: bm.event(k)["t"]                     # noqa: E731
    fx = []

    def sfx(i, kind, t, gain, pan=None, dist=None, **params):
        fx.append(dict(id=f"C+.{i}", fx=kind, t=float(t), gain_db=gain, pan=pan, dist=dist, exempt=False,
                       params=params, rev=REV))
    # the pen: the drawings and each ink line's first 24 frames
    sfx("pen.mountain", "pen", ev("riffle") + 0.7, -27, dur=2.4, density=0.9)
    sfx("pen.T1", "pen", 400 / 24, -25, dur=1.0)
    sfx("pen.deep", "pen", ev("deep") + 0.4, -28, dur=9.0, density=0.7, pressure=0.7)
    sfx("pen.T7", "pen", 1710 / 24, -25, dur=1.0)
    sfx("pen.T9", "pen", 3640 / 24, -26, dur=1.0)
    sfx("pen.T14", "pen", 6790 / 24, -26, dur=1.0)
    # paper crackle at every burn-through (the hit = the moment the page opens)
    sfx("burn.letters", "burn", ev("burn_through"), -17, dur=3.0, peak=0.5)
    sfx("burn.deep", "burn", ev("deep"), -15, dur=2.2, peak=0.35)
    sfx("burn.eye", "burn", ev("eye_burn"), -13, dur=2.2, peak=0.4)
    sfx("burn.map", "burn", ev("map"), -18, dur=2.6, peak=0.4)
    sfx("burn.remains", "burn", ev("eye_falls") + 4 * BEAT_S, -23, dur=2.0, peak=0.35, size=0.7)
    sfx("burn.title", "burn", 6980 / 24, -27, dur=2.6, peak=0.4, size=0.5)
    # the Mirror's drop; the cock, far and soft, under the harp
    sfx("drop", "drop", ev("drop"), -17)
    sfx("cock", "cock", ev("cock"), -24, dist=None)
    # the tick of cold metal in fire (the Ring on the tip of her steel)
    sfx("coldtick", "coldtick", ev("steel"), -15, dur=ev("fist_again") - ev("steel"), count=8)
    # the seethe of the melt (and nothing else, bars 68-69)
    sfx("seethe", "seethe", ev("white_heart"), -10, dur=7.0, flare=ev("letters_flare") - ev("white_heart"),
        out=ev("unmade") - ev("white_heart"))
    # THE FIRE REMAINS: every torch dips into it; then the roads of small flames run outward
    fr = ev("eye_falls")
    for k, (db, pan) in enumerate(((0.4, -0.35), (0.95, 0.3), (1.5, -0.1), (2.1, 0.45), (2.6, -0.5))):
        sfx(f"dip.{k}", "dip", fr + db, -14 - k * 0.5, pan=pan, size=0.5)
    sfx("roads", "roads", fr + 4 * BEAT_S + 0.25, -22, dur=ev("breath_dawn") - fr - 4 * BEAT_S - 0.3, count=6)
    amb = [dict(id="C+.fire.remains", fx="fire", t0=fr, t1=ev("breath_dawn"), gain_db=-20, fade_in=1.5,
                fade_out=0.3, params=dict(rate=9, level=0.8, breath=1.0), rev=REV),
           # the fire everyone lit rises around her fist, from every torch at once, and roars until the white heart
           dict(id="C+.fire.rises", fx="blaze", t0=ev("fire_rises") - 0.3, t1=ev("white_heart"), gain_db=-17,
                fade_in=0.8, fade_out=0.12, params=dict(grow=8.0), rev=REV),
           # the sea at the Havens, nearer (the cue sheet's bed alone sat 14 LU under the elegy)
           dict(id="C+.sea.near", fx="sea", t0=ev("havens"), t1=ev("blank"), gain_db=-19, fade_in=2.0, fade_out=2.5,
                params={}, rev=REV)]
    return fx, amb


# ---------------------------------------------------------------------------
def build(bm):
    S = K.Score("C", bm)
    setup(S)
    for fn in (storyteller, mountain, letters_to_fire, fire_alone, forging, race, deep, eye, mirror, grasp, old_law,
               ring_falls, the_find, fire_test, reveal, run, map_answers, council, bring_out, bearer, unmaking,
               fire_remains, illumination, plenty, havens, last_pages):
        fn(S, bm)
    # C's own effects, on top of the locked cue sheet (in memory only)
    if not any(e.get("id", "").startswith("C+.") for e in bm.d.get("sfx", [])):
        fx, amb = extra_effects(bm)
        bm.d.setdefault("sfx", []).extend(fx)
        bm.d.setdefault("ambience", []).extend(amb)
    end = bm.bars * 4
    for pn in ("hn", "hn2", "hn3", "hn_far", "hn_farther", "hn_far2", "hn_ans", "ca", "ca_dbl", "cl_c", "ob_c", "svln"):
        K.breathe(S, pn, 0, end, depth=0.1, min_dur=1.4)
    K.ride(S, bm, RIDE)
    # every attacked note starts early by its own sample's measured attack, so that it ARRIVES on the beat
    for pn in ("ca", "ca_dbl", "cl_c", "cl_dbl", "ob_c", "ob_dbl", "bsn_c", "hn", "hn2", "hn3", "hn_st", "hn_st2",
               "hn_far", "hn_farther", "hn_far2", "hn_ans", "tbn", "tbn2", "tuba", "svln", "svln2", "vla_q",
               "vc_ring", "line_vla", "line_vc", "vla", "vc_trem", "vln1", "vc_q", "cb_q", "vln1_q", "vln2_q", "vc", "cb",
               "vln2"):
        K.anticipate(S, pn, hi=ANTIC_HI.get(pn, 0.25))
    # the dawn's line: its legato steps crossfade centred on the beat
    sun = bm.ev("sunrise")
    for pn in ("vln1", "svln", "svln2"):
        for n in S.P(pn).notes:
            if n.legato and sun <= n.start < bm.ev("plenty"):
                n.kw["antic"] = 0.02
    for (eid, pn), dt in ANTIC_DT.items():
        for n in S.P(pn).notes:
            if abs(n.start - bm.ev(eid)) < 1e-6 and not n.legato:
                n.kw["antic"] = max(0.02, n.kw.get("antic", 0.0) + dt)
    for (eid, pn), a in ANTIC_SET.items():
        b0 = bm.ev(eid) if isinstance(eid, str) else bm.ev(eid[0]) + eid[1]
        for n in S.P(pn).notes:
            if abs(n.start - b0) < 1e-6:
                n.kw["antic"] = a
    for p in S.parts.values():
        p.gain_db += SCORE_TRIM_DB
    for s in bm.sections:
        if s.get("rel"):
            S.levels.append((s["id"], *s["rel"]))
    t = lambda k: bm.event(k)["t"]                       # noqa: E731
    S.rules = [("C's loudest is the slit: everything after the Eye at least 1 LU under the film's loudest",
                t("mirror"), bm.seconds, -1.0),
               ("the dawn is warm, not loud: at least 2 LU under the slit", t("sunrise"), t("plenty"), -2.0),
               ("the book's prologue is quiet: C1-C4 at least 6 LU under the slit", 0.0, t("fire_alone"), -6.0)]
    return S


# ---------------------------------------------------------------------------
# the rules on the page (run: python score_v3_C.py)
# ---------------------------------------------------------------------------
IGNITIONS = ("fire_catches", "roar", "beacon_1", "beacon_2", "beacon_3", "beacon_4", "beacon_5", "beacon_6", "beacon_7",
             "catch", "fire_rises", "sunrise", "coast_1", "coast_2", "coast_3", "snow", "ring_set")
STROKES = ("harp", "timp", "vc_pz", "cb_pz", "vc_sp", "cb_sp", "vla_sp", "vln2_sp", "anvil", "tick", "taiko",
           "lowbell", "glitter", "hharm")


def check(S, bm):
    out = []
    ign = [bm.ev(k) for k in IGNITIONS]
    for pn in STROKES:
        if pn not in S.parts:
            continue
        for n in S.P(pn).notes:
            for k, b in zip(IGNITIONS, ign):
                if abs(n.start - b) < 0.07:
                    out.append(f"STROKE ON {k}: {pn} at beat {n.start:.2f}")
    # the ANSWER (D C# B F# / D C Bb F, any transposition) never before beacon 2; HOME never before the dawn
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
    return out


if __name__ == "__main__":
    from timeline_v3 import BarMap
    bm_ = BarMap("C")
    S_ = build(bm_)
    parts = S_.used()
    print(f"C: {len(parts)} parts, {sum(len(p.notes) for p in parts.values())} notes, {len(S_.sync)} sync points, "
          f"{len(bm_.d['sfx'])} effects + {len(bm_.d['ambience'])} beds")
    w = K.check_notes(parts, bm_.bars * 4) + K.check_rates(parts) + check(S_, bm_)
    print(f"rule checks: {len(w)} warning(s)")
    for x in w[:80]:
        print("  " + x)
