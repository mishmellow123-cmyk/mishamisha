"""Score DSL for THE LONG DAWN.

Time is expressed in *global beats* (0-based quarter notes from the top of
the film).  72 BPM, 4/4:  1 beat = 20 video frames = 40 000 samples @ 48 kHz.

    gb(bar, beat)   -> global beat of (1-based) bar/beat, e.g. gb(29, 1) = 112
    fb(frame)       -> global beat of a video frame (frame/20)

A Part is one instrument track (one sampler patch or one synth voice type)
with notes, a dynamics curve (0..1, like CC1/expression) and mix settings.
"""
import math
import re

SR = 48000
BPM = 72
BEAT_S = 60.0 / BPM            # 0.8333 s
BEAT_N = 40000                 # samples per beat
FPS = 24
TOTAL_S = 117.0
TOTAL_N = int(TOTAL_S * SR)    # 5 616 000


def gb(bar, beat=1.0):
    return (bar - 1) * 4 + (beat - 1)


def fb(frame):
    return frame / 20.0


def beat_s(b):
    return b * BEAT_S


PC = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}


def m(name):
    """'D4' -> 62, 'Bb3' -> 58, 'F#5' -> 78. Ints/floats pass through."""
    if isinstance(name, (int, float)):
        return name
    mm = re.fullmatch(r"([A-Ga-g])([#b]*)(-?\d)", name.strip())
    if not mm:
        raise ValueError(name)
    p = PC[mm.group(1).upper()] + mm.group(2).count("#") - mm.group(2).count("b")
    return p + 12 * (int(mm.group(3)) + 1)


NAMES = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]


def name(midi):
    midi = int(round(midi))
    return f"{NAMES[midi % 12]}{midi // 12 - 1}"


# dynamics words -> level (0..1, ~velocity/127)
DYN = {"pppp": 0.08, "ppp": 0.13, "pp": 0.2, "p": 0.3, "mp": 0.42, "mf": 0.55,
       "f": 0.7, "ff": 0.85, "fff": 1.0}


def lv(x):
    return DYN[x] if isinstance(x, str) else float(x)


class Note:
    __slots__ = ("pitch", "start", "dur", "vel", "art", "legato", "sync",
                 "gain_db", "pan", "kw")

    def __init__(self, pitch, start, dur, vel=None, art=None, legato=False,
                 sync=False, gain_db=0.0, pan=None, **kw):
        self.pitch = m(pitch) if pitch is not None else None
        self.start = float(start)
        self.dur = float(dur)
        self.vel = None if vel is None else lv(vel)
        self.art = art
        self.legato = legato
        self.sync = sync
        self.gain_db = gain_db
        self.pan = pan
        self.kw = kw

    def to_dict(self):
        return dict(pitch=self.pitch, start=self.start, dur=self.dur,
                    vel=self.vel, art=self.art, legato=self.legato,
                    sync=self.sync, gain_db=self.gain_db, pan=self.pan,
                    kw=self.kw)


class Part:
    """One track.  kind='sampler' (inst = patch key) or kind='synth'
    (inst = synth function name)."""

    def __init__(self, name, inst, kind="sampler", bus="strings", pan=0.0,
                 width=1.0, gain_db=0.0, send=0.3, depth=0.3, humanize_ms=8.0,
                 seed=None, params=None):
        self.name = name
        self.inst = inst
        self.kind = kind
        self.bus = bus
        self.pan = pan
        self.width = width
        self.gain_db = gain_db
        self.send = send
        self.depth = depth
        self.humanize_ms = humanize_ms
        self.seed = seed if seed is not None else (abs(hash(name)) % 10000)
        self.params = params or {}
        self.notes = []
        self.dyn = []          # [(beat, level)] piecewise, smooth

    # -- note entry -------------------------------------------------------
    def n(self, pitch, start, dur, vel=None, **kw):
        nt = Note(pitch, start, dur, vel, **kw)
        self.notes.append(nt)
        return nt

    def chord(self, pitches, start, dur, vel=None, **kw):
        for p in pitches:
            self.n(p, start, dur, vel, **kw)

    def line(self, text, start, vel=None, legato=True, gap=0.0, **kw):
        """Melody string: 'D4:1 A4:1 D5:2 | r:1 ...'  (durations in beats).
        Returns the beat after the last note."""
        t = start
        toks = [x for x in text.replace("|", " ").split() if x]
        prev_note = False
        for tok in toks:
            p, d = tok.split(":")
            d = eval(d, {}, {})  # allow fractions like 1/3
            if p.lower() != "r":
                v = vel
                if "!" in p:           # accent marker e.g. D5!:1
                    p = p.replace("!", "")
                    v = min(1.0, (lv(vel) if vel is not None else 0.6) + 0.12)
                self.n(p, t, d - gap, v, legato=(legato and prev_note), **kw)
                prev_note = True
            else:
                prev_note = False
            t += d
        return t

    # -- dynamics ------------------------------------------------------------
    def d(self, *pts):
        """d((beat, level), (beat, level), ...) appends dynamic points."""
        for b, l in pts:
            self.dyn.append((float(b), lv(l)))
        self.dyn.sort(key=lambda x: x[0])

    def level_at(self, b):
        return dyn_at(self.dyn, b)

    def to_dict(self):
        return dict(name=self.name, inst=self.inst, kind=self.kind, bus=self.bus,
                    pan=self.pan, width=self.width, gain_db=self.gain_db,
                    send=self.send, depth=self.depth,
                    humanize_ms=self.humanize_ms, seed=self.seed,
                    params=self.params, dyn=self.dyn,
                    notes=[x.to_dict() for x in self.notes])


def smooth(x):
    return x * x * (3 - 2 * x)


def dyn_at(pts, b):
    if not pts:
        return 0.6
    if b <= pts[0][0]:
        return pts[0][1]
    if b >= pts[-1][0]:
        return pts[-1][1]
    for (b0, l0), (b1, l1) in zip(pts[:-1], pts[1:]):
        if b0 <= b <= b1:
            if b1 == b0:
                return l1
            u = (b - b0) / (b1 - b0)
            return l0 + (l1 - l0) * smooth(u)
    return pts[-1][1]


def dyn_array(pts, b0, n, beat_n=BEAT_N):
    """Vectorised dynamics curve sampled every sample from beat b0 for n samples."""
    import numpy as np
    if not pts:
        return np.full(n, 0.6, np.float32)
    bs = b0 + np.arange(n, dtype=np.float64) / beat_n
    xs = np.array([p[0] for p in pts])
    ys = np.array([p[1] for p in pts])
    idx = np.clip(np.searchsorted(xs, bs, side="right") - 1, 0, len(xs) - 1)
    idx2 = np.clip(idx + 1, 0, len(xs) - 1)
    x0, x1 = xs[idx], xs[idx2]
    y0, y1 = ys[idx], ys[idx2]
    span = np.where(x1 > x0, x1 - x0, 1.0)
    u = np.clip((bs - x0) / span, 0, 1)
    u = u * u * (3 - 2 * u)
    out = np.where(x1 > x0, y0 + (y1 - y0) * u, y0)
    out = np.where(bs < xs[0], ys[0], out)
    out = np.where(bs >= xs[-1], ys[-1], out)
    return out.astype(np.float32)


def level_db(level):
    """Loudness mapping of a 0..1 dynamic level (30*log10)."""
    return 30.0 * math.log10(max(level, 0.03))
