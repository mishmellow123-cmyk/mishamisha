"""THE LONG DAWN v3 - C . THE LAST PAGES: the cut's own synth voices and effect designs (owner: COMPOSER-C).

Imported by score_v3_C.py.  Everything here REGISTERS at import time into synth_v3.VOICES (score parts, kind
"synth") and sfx_v3.DESIGNS (the effects stem), so no shared engine file changes and no other cut's cache busts.
REV (a hash of this file) goes into the params of every C synth seat and into every effect the score adds, so an
edit here re-renders exactly the parts and clips that use it (render_v3's own cache keys do not see this file).

Voices (score parts)
  hharm      harp harmonics: the octave flageolet, pure, with the finger's soft touch (C4's glowing letters, the
             Mirror, the black of the old law)
  anvil      a smith's anvil (size 1) down to the Deep's small tick (size 0.2): inharmonic steel modes + the strike
  tamswell   the tam-tam swell into the slit: the VSCO gong's wash reversed into the peak, then its bloom forward
             with the mallet removed (a swell and a bloom, never a stroke)
  glassfall  one glass tone falling away into silence (the slip; the Ring's fall), optional tumble shimmer

Designs (effects stem; the beat sheet's C column beyond the locked cue sheet)
  pen        the pen under each ink line / drawing: nib strokes with the paper's tooth, drifting left to right
  burn       paper crackle at a burn-through (hit = the moment the page opens)
  coldtick   the tick of cold metal in fire: sparse, tiny steel ticks
  seethe     the seethe of the melt: sizzle, bubbling, a low simmer; one swell as the letters flare
  cock       a cock crowing once, far away (for `distance`)
  drop       a single drop into still water, and its small ripple
  dip        a torch dipped into the fire: a soft catch of flame
  roads      small flames running outward along roads, hearths kindling where they arrive
"""
import hashlib
import os

import numpy as np
import soundfile as sf
from scipy import signal

import sfx as FX
import sfx_v3
import synth as SY
import synth_v3
from dsl import SR, BEAT_S, dyn_at

TWOPI = 2 * np.pi
REV = hashlib.sha1(open(os.path.abspath(__file__), "rb").read()).hexdigest()[:10]


def _vel(part, nt):
    return nt["vel"] if nt["vel"] is not None else dyn_at(part["dyn"], nt["start"])


def _i0(nt):
    return int(round(nt["start"] * BEAT_S * SR))


def _bp(lo, hi, order=2):
    return signal.butter(order, [max(20.0, lo) / (SR / 2), min(0.97, hi / (SR / 2))], btype="band", output="sos")


# ---------------------------------------------------------------------------
# voices
# ---------------------------------------------------------------------------
def hharm(part, total_n):
    """harp harmonics: note pitch = the SOUNDING harmonic.  kw decay (s)."""
    rng = np.random.default_rng(part["seed"])
    out = np.zeros((total_n, 2), np.float32)
    dec0 = part["params"].get("decay", 2.6)
    for nt in part["notes"]:
        f0 = SY.hz(nt["pitch"])
        vel = _vel(part, nt)
        dec = nt["kw"].get("decay", dec0) * min(1.0, (660.0 / f0) ** 0.3)
        n = int(min(dec * 4.0 + 0.3, 11.0) * SR)
        t = np.arange(n) / SR
        chans = []
        for side in (-1, 1):
            det = 2 ** (side * rng.uniform(0.6, 2.0) / 1200)
            ph = TWOPI * f0 * det * t + rng.uniform(0, 6.28)
            y = (np.sin(ph) + 0.05 * np.sin(2 * ph + 0.7) * np.exp(-t / (dec * 0.25))
                 + 0.012 * np.sin(3 * ph + 1.3) * np.exp(-t / (dec * 0.12)))
            chans.append(y)
        y = np.stack(chans, 1)
        y *= ((1 - np.exp(-t / 0.0025)) * np.exp(-t / dec))[:, None]
        L = int(0.014 * SR)
        touch = rng.normal(0, 1, L) * np.exp(-np.arange(L) / (0.003 * SR))
        touch = signal.sosfilt(_bp(f0 * 0.6, f0 * 3.2), touch) * 0.1
        y[:L] += touch[:, None]
        SY._place(out, (y * vel * 0.34).astype(np.float32), _i0(nt))
    return out


ANVIL_MODES = ((1.0, 1.0, 1.0), (2.76, 0.55, 0.62), (4.13, 0.34, 0.45), (5.40, 0.28, 0.33), (6.83, 0.17, 0.25),
               (8.93, 0.1, 0.18))


def anvil(part, total_n):
    """kw size: 1 = a smith's anvil ringing under the hammer, 0.2 = the Deep's small dry tick"""
    rng = np.random.default_rng(part["seed"])
    out = np.zeros((total_n, 2), np.float32)
    for nt in part["notes"]:
        f0 = SY.hz(nt["pitch"])
        vel = _vel(part, nt)
        size = float(nt["kw"].get("size", 1.0))
        ring = 0.04 + 0.85 * size ** 1.5
        n = int((ring * 5 + 0.06) * SR)
        t = np.arange(n) / SR
        ys = []
        for side in range(2):
            y = np.zeros(n)
            for r, a, ds in ANVIL_MODES:
                f = f0 * r * (1 + rng.normal(0, 0.003))
                if f > SR * 0.45:
                    continue
                y += a * np.sin(TWOPI * f * t + rng.uniform(0, 6.28)) * np.exp(-t / (ring * ds))
            ys.append(y)
        y = np.stack(ys, 1) * 0.5
        L = int(0.005 * SR)
        click = rng.normal(0, 1, L) * np.exp(-np.arange(L) / (0.0006 * SR))
        click = signal.sosfilt(signal.butter(2, 1800 / (SR / 2), btype="high", output="sos"), click)
        y[:L] += (click * (0.6 + 0.6 * (1 - size)))[:, None]
        y *= (1 - np.exp(-t / 0.0003))[:, None]
        SY._place(out, (y * vel * 0.3).astype(np.float32), _i0(nt))
    return out


_GONG = {}


def _gong(layer=2):
    if layer not in _GONG:
        import sampler
        regs = sorted(sampler.regions("gong"), key=lambda r: r["lovel"])
        x, sr = sf.read(regs[layer]["path"], dtype="float32", always_2d=True)
        if sr != SR:
            from math import gcd
            g = gcd(SR, sr)
            x = signal.resample_poly(x, SR // g, sr // g, axis=0).astype(np.float32)
        if x.shape[1] == 1:
            x = np.repeat(x, 2, 1)
        on = int(np.argmax(np.abs(x).max(1) > np.abs(x).max() * 0.05))
        x = x[on:]
        _GONG[layer] = (x / (np.abs(x).max() + 1e-9)).astype(np.float32)
    return _GONG[layer]


def tamswell(part, total_n):
    """one swell per note: rising from nothing to its peak at the note's END (the reversed wash, shaped by kw
    curve), then the bloom (the forward wash with the mallet faded out) decaying over kw rel seconds"""
    out = np.zeros((total_n, 2), np.float32)
    for nt in part["notes"]:
        g = _gong(int(nt["kw"].get("layer", 2)))
        vel = _vel(part, nt)
        dur_s = nt["dur"] * BEAT_S
        rel = float(nt["kw"].get("rel", 5.0))
        k0 = int(0.18 * SR)
        m = int(dur_s * SR)
        wash = g[k0:k0 + m][::-1].copy()
        if len(wash) < m:
            wash = np.concatenate([np.zeros((m - len(wash), 2), np.float32), wash])
        u = np.linspace(0, 1, m)
        wash *= (u ** nt["kw"].get("curve", 1.6))[:, None]
        r = int(rel * SR)
        bloom = g[k0:k0 + r].copy()
        fi = int(0.05 * SR)
        bloom[:fi] *= np.linspace(0.85, 1, fi)[:, None]
        fo = min(len(bloom), int(1.2 * SR))
        bloom[-fo:] *= np.linspace(1, 0, fo)[:, None] ** 1.5
        y = np.concatenate([wash, bloom])
        SY._place(out, (y * vel * 0.5).astype(np.float32), _i0(nt))
    return out


def glassfall(part, total_n):
    """a glass tone (FM, ratio 3.5) that falls kw drop semitones over the note (kw curve) and dies into silence by
    its end; kw tumble = shimmer rate (Hz) of a slowly turning band catching the light"""
    rng = np.random.default_rng(part["seed"])
    out = np.zeros((total_n, 2), np.float32)
    for nt in part["notes"]:
        f0 = SY.hz(nt["pitch"])
        vel = _vel(part, nt)
        kw = nt["kw"]
        dur_s = nt["dur"] * BEAT_S
        n = int((dur_s + 0.3) * SR)
        t = np.arange(n) / SR
        u = np.clip(t / dur_s, 0, 1)
        f = f0 * 2 ** (-float(kw.get("drop", 24.0)) * u ** kw.get("curve", 1.5) / 12.0)
        chans = []
        for side in (-1, 1):
            ph = TWOPI * np.cumsum(f * 2 ** (side * rng.uniform(1.5, 4.0) / 1200)) / SR
            idx = kw.get("index", 1.1) * (0.35 + 0.65 * np.exp(-t / 0.2))
            car = np.sin(ph + idx * np.sin(3.5 * ph))
            chans.append(car + 0.22 * np.sin(2 * ph + 0.4))
        y = np.stack(chans, 1)
        env = (1 - np.exp(-t / 0.003)) * np.exp(-t / max(0.05, dur_s * kw.get("decay", 0.42)))
        env *= np.clip((dur_s + 0.25 - t) / 0.4, 0, 1)
        tb_ = float(kw.get("tumble", 0.0))
        if tb_ > 0:
            env *= 1 - 0.45 * (0.5 + 0.5 * np.sin(TWOPI * tb_ * t + rng.uniform(0, 6.28)))
        y *= env[:, None]
        SY._place(out, (y * vel * 0.28).astype(np.float32), _i0(nt))
    return out


# ---------------------------------------------------------------------------
# effect designs: (rng, **params) -> (stereo float32, hit index); beds take dur first
# ---------------------------------------------------------------------------
def _st(y):
    y = np.asarray(y, np.float32)
    return np.stack([y, y], 1) if y.ndim == 1 else y


def _drift_pan(mono, p0, p1):
    th = (np.clip(np.linspace(p0, p1, len(mono)), -1, 1) + 1) * np.pi / 4
    return (np.stack([mono * np.cos(th), mono * np.sin(th)], 1) * np.sqrt(2)).astype(np.float32)


def d_pen(rng, dur=1.0, pressure=0.8, density=1.0, pan0=-0.22, pan1=0.22):
    """a pen writing: nib strokes (50-220 ms) with the paper's tooth (a stream of fibre ticks, 2-7.5 kHz) and the
    nib's soft drag, short lifts between strokes; the line drifts left to right"""
    n = int(dur * SR)
    y = np.zeros(n)
    drag = np.zeros(n)
    t = 0.0
    sos_t = _bp(2200, 7500)
    sos_d = _bp(700, 1900)
    while t < dur - 0.04:
        L = rng.uniform(0.05, 0.22)
        i0 = int(t * SR)
        m = min(int(L * SR), n - i0)
        if m < 128:
            break
        tt = np.arange(m) / SR
        rate = rng.uniform(300, 1100)
        imp = (rng.uniform(0, 1, m) < rate / SR) * rng.uniform(0.3, 1.0, m)
        grain = signal.sosfilt(sos_t, imp + 0.18 * rng.normal(0, 1, m))
        e = np.clip(tt / 0.012, 0, 1) * np.clip((L - tt) / 0.025, 0, 1) * (0.75 + 0.25 * np.sin(np.pi * tt / L))
        a = rng.uniform(0.5, 1.0) * pressure
        y[i0:i0 + m] += grain * e * a
        drag[i0:i0 + m] += signal.sosfilt(sos_d, rng.normal(0, 1, m)) * e * a * 0.08
        t += L + rng.uniform(0.02, 0.12) / max(0.3, density)
    mono = y + drag
    mono /= np.abs(mono).max() + 1e-9
    return FX.fade(_drift_pan(mono, pan0, pan1), 0.004, 0.05), 0


def d_burn(rng, dur=2.5, peak=0.45, size=1.0):
    """paper burning through: a fine dense crackle, a few pops, the paper curling, the flame's breath; it swells
    to the moment the page opens (hit) and dies away"""
    n = int(dur * SR)
    t = np.arange(n) / SR
    tp = peak * dur
    env = np.where(t < tp, (t / tp) ** 1.8, np.exp(-(t - tp) / (0.32 * dur)))
    fine = FX.crackle(rng, dur, rate=90 * size, level=0.8, pops=0.4, breath=0.0)[:n]
    pops = FX.crackle(rng, dur, rate=12 * size, level=1.0, pops=0.9, breath=0.0)[:n]
    flame = FX.sos_filter(FX.pink(rng, n), "band", (160, 2200)) * 0.35 * size
    curl = FX.rustle(rng, dur)[:n] * 0.3
    y = (fine * 0.7 + pops * 0.35 + curl) * env[:, None] + flame * (env ** 1.5)[:, None]
    return FX.fade(y.astype(np.float32), 0.01, 0.2), int(tp * SR)


def d_coldtick(rng, dur=6.5, count=8):
    """the tick of cold metal in a fire: sparse tiny steel ticks (the first on the hit)"""
    N = int(dur * SR)
    y = np.zeros((N, 2), np.float32)
    times = [0.0] + sorted(rng.uniform(0.3, dur - 0.3, count - 1))
    L = int(0.07 * SR)
    tt = np.arange(L) / SR
    hp = signal.butter(2, 3000 / (SR / 2), btype="high", output="sos")
    for k, tk in enumerate(times):
        ping = np.zeros(L)
        for f, d, a in zip(rng.uniform(2600, 7800, 3), rng.uniform(0.003, 0.016, 3), (1.0, 0.6, 0.4)):
            ping += a * np.sin(TWOPI * f * tt + rng.uniform(0, 6.28)) * np.exp(-tt / d)
        click = signal.sosfilt(hp, rng.normal(0, 1, L) * np.exp(-tt / 0.0005))
        amp = 1.0 if k == 0 else rng.uniform(0.35, 1.0)
        i = int(tk * SR)
        seg = FX.panner((ping * 0.6 + click * 0.35) * amp, rng.uniform(-0.25, 0.25))
        y[i:i + L] += seg[:N - i]
    return FX.fade(y, 0.0, 0.05), 0


def d_seethe(rng, dur=7.0, flare=2.5, out=3.33):
    """the seethe of the melt: a sizzle that bubbles, a low simmer, a few molten glugs as the band slumps; one
    swell as the letters flare (flare s), settling after they go out (out s)"""
    n = int(dur * SR)
    t = np.arange(n) / SR
    hs = FX.sos_filter(FX.white(rng, n), "band", (1800, 9000))
    bub = np.zeros(n)
    k = 0.0
    while k < dur:
        k += rng.exponential(1 / 38.0)
        i = int(k * SR)
        L = int(rng.uniform(0.004, 0.02) * SR)
        if i + L >= n:
            break
        bub[i:i + L] += np.hanning(L) * rng.uniform(0.3, 1.0)
    sizzle = hs * (0.3 + bub)[:, None]
    lo = FX.sos_filter(FX.brown(rng, n), "low", 260) * (1 + 0.3 * FX.smooth_rand(rng, n, 3.0, 1))
    glug = np.zeros(n)
    for g in rng.uniform(0.2, 2.2, 5):
        i = int(g * SR)
        L = int(0.12 * SR)
        tt = np.arange(L) / SR
        fg = rng.uniform(110, 260) * (1 + 0.6 * tt / 0.12)
        glug[i:i + L] += np.sin(TWOPI * np.cumsum(fg) / SR) * np.exp(-tt / 0.035) * rng.uniform(0.3, 0.8)
    env = np.clip(t / 0.5, 0, 1) * (1 + 0.9 * np.exp(-((t - flare) / 0.22) ** 2))
    env *= np.where(t < out, 1.0, np.exp(-(t - out) / 1.3))
    y = (sizzle * 0.55 + lo * 0.5 + _st(glug) * 0.25) * env[:, None]
    return FX.fade(y.astype(np.float32), 0.05, 0.5), 0


def d_cock(rng, dist=0.85):
    """a cock crowing once, far away: five syllables (cock-a-doodle-doo, the last long, with vibrato and a falling
    tail), a harsh harmonic voice through three formants, then distance"""
    syl = [(0.00, 0.11, 560, 640, 0.8), (0.15, 0.10, 610, 690, 0.7), (0.29, 0.19, 660, 760, 0.9),
           (0.51, 0.10, 720, 650, 0.7), (0.65, 0.95, 790, 600, 1.0)]
    n = int(2.4 * SR)
    y = np.zeros(n)
    for s0, d, fa, fb, a in syl:
        i = int(s0 * SR)
        m = int(d * SR)
        tt = np.arange(m) / SR
        u = tt / d
        if d > 0.5:
            f = fa * (1 + 0.04 * np.sin(np.pi * np.clip(u / 0.7, 0, 1))) * np.where(u < 0.72, 1.0,
                                                                                     (fb / fa) ** ((u - 0.72) / 0.28))
            f *= 1 + 0.012 * np.sin(TWOPI * 6.5 * tt) * np.clip(u / 0.2, 0, 1)
        else:
            f = fa + (fb - fa) * u
        ph = TWOPI * np.cumsum(f * (1 + 0.004 * rng.normal(0, 1, m))) / SR
        v = np.zeros(m)
        for h in range(1, 14):
            v += np.sin(h * ph) / h ** 0.75
        e = np.clip(u / 0.08, 0, 1) * np.clip((1 - u) / 0.15, 0, 1)
        y[i:i + m] += v * e * a
    y += FX.sos_filter(FX.white(rng, n)[:, :1], "band", (900, 4000))[:, 0] * 0.06 * (np.abs(y) > 0.01)
    out = np.zeros(n)
    for fc, bw, g in ((900, 350, 1.0), (1750, 500, 0.8), (2900, 700, 0.45)):
        out += signal.sosfilt(_bp(fc - bw / 2, fc + bw / 2), y) * g
    out /= np.abs(out).max() + 1e-9
    st = FX.panner(out.astype(np.float32), rng.uniform(-0.35, 0.35)).astype(np.float32)
    return sfx_v3.distance(st, dist, rng), 0


def d_drop(rng, ripple=True):
    """a single drop into still water: the tiny splash and the bubble's upward chirp, and a smaller one after"""
    n = int(1.0 * SR)
    t = np.arange(n) / SR
    y = np.zeros(n)
    for t0, a, fs in ((0.0, 1.0, 1.0), (0.23, 0.35, 1.25)) if ripple else ((0.0, 1.0, 1.0),):
        i = int(t0 * SR)
        tt = t[:n - i]
        f = 820 * fs * (1 + 1.25 * (1 - np.exp(-tt / 0.02)))
        ping = np.sin(TWOPI * np.cumsum(f) / SR) * np.exp(-tt / 0.04) * (1 - np.exp(-tt / 0.0008))
        spl = signal.sosfilt(signal.butter(2, 2500 / (SR / 2), btype="high", output="sos"),
                             rng.normal(0, 1, len(tt))) * np.exp(-tt / 0.005) * 0.2
        y[i:] += (ping + spl) * a
    return FX.fade(_st(y / (np.abs(y).max() + 1e-9)), 0.0, 0.1), 0


def d_dip(rng, size=0.6):
    """a torch dipped into the fire: a soft rush of flame catching (hit = the catch) and a brief crackle"""
    n = int(1.7 * SR)
    y = np.zeros((n, 2), np.float32)
    w = FX.whoosh(rng, 0.42, f0=170, f1=1500, peak=0.75, q=0.8) * 0.7
    y[:len(w)] += w
    i0 = int(0.3 * SR)
    m = n - i0
    tt = np.arange(m) / SR
    f = 44 + 30 * np.exp(-tt / 0.05)
    y[i0:] += _st(np.sin(TWOPI * np.cumsum(f) / SR) * np.exp(-tt / 0.16) * 0.45 * size)
    cr = FX.crackle(rng, m / SR, rate=22, level=0.5, pops=0.3, breath=0.25)[:m]
    L = len(cr)
    y[i0:i0 + L] += cr * np.exp(-tt[:L] / 0.5)[:, None] * 0.6
    return FX.fade(y, 0.01, 0.2), i0


def d_roads(rng, dur=3.2, count=6):
    """small flames running outward along roads from the centre (each a moving crackle), and a hearth kindling
    softly where each arrives"""
    n = int(dur * SR)
    y = np.zeros((n, 2), np.float32)
    for k in range(count):
        side = -1.0 if k % 2 == 0 else 1.0
        t0 = rng.uniform(0.0, 0.45)
        L = dur - t0 - rng.uniform(0.35, 0.7)
        i0 = int(t0 * SR)
        cr = FX.crackle(rng, L, rate=28, level=0.6, pops=0.15, breath=0.2).mean(1)
        m = len(cr)
        cr *= np.linspace(1.0, 0.55, m)
        far = side * rng.uniform(0.5, 0.95)
        y[i0:i0 + m] += _drift_pan(cr, 0.0, far)[:n - i0]
        j = i0 + m
        if j < n - int(0.3 * SR):
            w = FX.whoosh(rng, 0.3, f0=250, f1=1400, peak=0.6, q=0.9).mean(1) * 0.35
            pk = FX.crackle(rng, 0.5, rate=35, level=0.5, pops=0.3, breath=0.2).mean(1) * 0.5
            puff = np.zeros(int(0.55 * SR))
            puff[:len(w)] += w
            puff[:len(pk)] += pk[:len(puff)]
            seg = FX.panner(puff.astype(np.float32), far)
            y[j:j + len(seg)] += seg[:n - j]
    return FX.fade(y, 0.05, 0.2), 0


# ---------------------------------------------------------------------------
# registration
# ---------------------------------------------------------------------------
synth_v3.VOICES.update(hharm=hharm, anvil=anvil, tamswell=tamswell, glassfall=glassfall)
sfx_v3.DESIGNS.update(pen=d_pen, burn=d_burn, coldtick=d_coldtick, seethe=d_seethe, cock=d_cock, drop=d_drop,
                      dip=d_dip, roads=d_roads)
