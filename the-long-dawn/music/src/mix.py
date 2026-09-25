"""Mixing & mastering.

* One synthesized hall IR (true stereo, early reflections + frequency-
  dependent exponential tail, RT60 ~3.1 s mid, decorrelated L/R).
* Parts: pan/width (orchestral seating), depth (distance EQ + more send),
  gain, reverb send.
* Master: shared gain envelopes (glue compression + true-peak limiter) are
  computed on the full mix and applied to both stems, so
  score.wav + sfx.wav == mix.wav (to rounding).
"""
import os

import numpy as np
from scipy import signal

from dsl import SR

HERE = os.path.dirname(os.path.abspath(__file__))
MUSIC = os.path.dirname(HERE)
CACHE = os.path.join(MUSIC, "cache")


# ---------------------------------------------------------------------------
# impulse response
# ---------------------------------------------------------------------------
def make_ir(rt_mid=3.1, length=4.6, seed=7, predelay=0.018, er_gain=0.55,
            bright=1.0):
    """Returns 4 IRs (LL, LR, RL, RR) as float32 arrays."""
    rng = np.random.default_rng(seed)
    n = int(length * SR)
    t = np.arange(n) / SR
    # octave bands and their RT60 (hall: long lows, short highs)
    bands = [(20, 125, rt_mid * 1.25), (125, 250, rt_mid * 1.15), (250, 500, rt_mid * 1.05),
             (500, 1000, rt_mid), (1000, 2000, rt_mid * 0.9), (2000, 4000, rt_mid * 0.72),
             (4000, 8000, rt_mid * 0.52 * bright), (8000, 20000, rt_mid * 0.32 * bright)]
    irs = []
    for k in range(4):
        tail = np.zeros(n)
        for lo, hi, rt in bands:
            nz = rng.normal(0, 1, n)
            b, a = signal.butter(2, [lo / (SR / 2), min(hi, SR * 0.45) / (SR / 2)], btype="band")
            nz = signal.lfilter(b, a, nz)
            tail += nz * 10 ** (-3 * t / rt)          # -60 dB at rt
        # late tail builds up
        build = np.clip((t - predelay) / 0.09, 0, 1) ** 1.5
        tail *= build
        # early reflections: sparse taps 8..95 ms, slightly lowpassed
        er = np.zeros(n)
        ntaps = 28
        times = np.sort(rng.uniform(0.006, 0.095, ntaps)) + predelay * 0.5
        same_side = k in (0, 3)
        for i, tm in enumerate(times):
            idx = int(tm * SR)
            amp = (0.9 if same_side else 0.55) * (1 - i / ntaps) ** 0.7 * rng.choice([-1, 1]) * rng.uniform(0.5, 1)
            er[idx] += amp
        b, a = signal.butter(1, 6000 / (SR / 2))
        er = signal.lfilter(b, a, er)
        ir = tail / np.sqrt((tail ** 2).sum()) + er_gain * er / np.sqrt((er ** 2).sum() + 1e-12) * 0.6
        if not same_side:
            ir *= 0.8
        # fade end
        f = int(0.3 * SR)
        ir[-f:] *= np.linspace(1, 0, f)
        irs.append(ir.astype(np.float32))
    norm = np.sqrt(sum((x ** 2).sum() for x in irs) / 2)
    return [x / norm for x in irs]


def get_ir(name="hall", **kw):
    path = os.path.join(CACHE, f"ir_{name}.npy")
    if os.path.exists(path):
        return list(np.load(path))
    irs = make_ir(**kw)
    np.save(path, np.stack(irs))
    return irs


def convolve_stereo(x, irs):
    LL, LR, RL, RR = irs
    n = len(x)
    wl = signal.oaconvolve(x[:, 0], LL)[:n] + signal.oaconvolve(x[:, 1], RL)[:n]
    wr = signal.oaconvolve(x[:, 0], LR)[:n] + signal.oaconvolve(x[:, 1], RR)[:n]
    return np.stack([wl, wr], 1).astype(np.float32)


# ---------------------------------------------------------------------------
# per-part processing
# ---------------------------------------------------------------------------
def pan_width(y, pan, width):
    mid = (y[:, 0] + y[:, 1]) * 0.5
    side = (y[:, 0] - y[:, 1]) * 0.5 * width
    l, r = mid + side, mid - side
    # balance law: centre = unity, hard side = +3 dB on that side
    th = (pan + 1) * np.pi / 4
    gl = np.cos(th) * np.sqrt(2)
    gr = np.sin(th) * np.sqrt(2)
    # spill part of the attenuated channel to the other side (keeps energy)
    if pan > 0:
        r = r + l * (1 - gl) * 0.5
    elif pan < 0:
        l = l + r * (1 - gr) * 0.5
    return np.stack([l * min(gl, 1.25), r * min(gr, 1.25)], 1).astype(np.float32)


def depth_eq(y, depth):
    """Distance: gentle HF loss (air absorption) proportional to depth."""
    if depth <= 0.05:
        return y
    fc = 16000 * (1 - 0.55 * depth)
    b, a = signal.butter(1, fc / (SR / 2))
    return signal.lfilter(b, a, y, axis=0).astype(np.float32)


def highpass(y, fc, order=2):
    b, a = signal.butter(order, fc / (SR / 2), btype="high")
    return signal.lfilter(b, a, y, axis=0).astype(np.float32)


# ---------------------------------------------------------------------------
# master dynamics (shared gain envelopes)
# ---------------------------------------------------------------------------
def _smooth_gain(gr_db, attack, release):
    """One-pole attack/release smoothing of a gain-reduction curve (dB, <=0)."""
    from scipy.signal import lfilter
    # implement with a simple loop in blocks via numba-free approach:
    # downsample to 1 ms control rate, smooth, upsample
    hop = 48
    g = gr_db[: len(gr_db) // hop * hop].reshape(-1, hop).min(1)
    a_att = np.exp(-1 / (attack * 1000))
    a_rel = np.exp(-1 / (release * 1000))
    out = np.empty_like(g)
    s = 0.0
    for i, v in enumerate(g):
        if v < s:
            s = a_att * s + (1 - a_att) * v
        else:
            s = a_rel * s + (1 - a_rel) * v
        out[i] = s
    full = np.repeat(out, hop)
    if len(full) < len(gr_db):
        full = np.pad(full, (0, len(gr_db) - len(full)), mode="edge")
    return full


def glue_comp_gain(x, thresh_db=-20.0, ratio=1.8, attack=0.03, release=0.35, knee=6.0):
    """RMS-detecting bus compressor -> gain curve (linear)."""
    mono = (x ** 2).mean(1)
    w = int(0.02 * SR)
    rms = np.sqrt(np.convolve(mono, np.ones(w) / w, mode="same") + 1e-12)
    lvl = 20 * np.log10(rms)
    over = lvl - thresh_db
    gr = np.where(over <= -knee / 2, 0.0,
                  np.where(over >= knee / 2, over * (1 / ratio - 1),
                           (1 / ratio - 1) * (over + knee / 2) ** 2 / (2 * knee)))
    g = _smooth_gain(gr, attack, release)
    return (10 ** (g / 20)).astype(np.float32)


def true_peak(x, os_factor=4):
    y = signal.resample_poly(x, os_factor, 1, axis=0)
    return float(np.abs(y).max())


def tp_limiter_gain(x, ceiling_db=-1.2, lookahead=0.005, release=0.12):
    """Look-ahead limiter on 4x-oversampled peak envelope -> gain curve."""
    ceil = 10 ** (ceiling_db / 20)
    y = signal.resample_poly(x, 4, 1, axis=0)
    pk = np.abs(y).max(1).reshape(-1, 4).max(1)[: len(x)]
    if len(pk) < len(x):
        pk = np.pad(pk, (0, len(x) - len(pk)))
    need = np.minimum(1.0, ceil / np.maximum(pk, 1e-9))
    la = int(lookahead * SR)
    # min filter over the look-ahead window (so gain is down before the peak)
    from scipy.ndimage import minimum_filter1d
    need = minimum_filter1d(need, size=2 * la + 1, origin=0)
    gdb = 20 * np.log10(need)
    g = _smooth_gain(gdb, 0.0005, release)
    g = np.minimum(g, gdb)  # guarantee
    # final smoothing of the hard min to avoid zipper
    return (10 ** (g / 20)).astype(np.float32)


def tpdf_dither_24(x, rng):
    lsb = 1.0 / (2 ** 23)
    return x + (rng.uniform(-0.5, 0.5, x.shape) + rng.uniform(-0.5, 0.5, x.shape)).astype(np.float32) * lsb
