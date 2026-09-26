"""v2 synthesised voices.  Everything from synth.py (v1) is used unchanged, so the
first half renders exactly as composed; this module only adds:

* fmwarm  - the "thinking fire" FM colour transformed warm: low ratio / low index FM,
            soft mallet attack, long glowing decay, gentle chorus and a lowpass that
            follows the velocity.  (The KINDLING's glass is ratio 3.5 / index 1.6.)
* deephit - a deep, round, pitched hit for the beacons (sub body tuned to the note +
            low skin boom, no crack, no noise tail): weight without "trailer".
* subpulse - a tuned sine sub under the bass that follows the dynamics curve.
"""
import numpy as np
from scipy import signal

import synth as S1
from dsl import SR, BEAT_S, BEAT_N, dyn_array, dyn_at

TWOPI = 2 * np.pi


def _t(n):
    return np.arange(n, dtype=np.float64) / SR


def fmwarm(part, total_n):
    rng = np.random.default_rng(part["seed"])
    prm = part["params"]
    ratio = prm.get("ratio", 1.0)
    index = prm.get("index", 0.55)
    decay = prm.get("decay", 2.6)
    atk = prm.get("atk", 0.006)
    lp = prm.get("lp", 3200.0)
    chorus = prm.get("chorus", 4.0)
    out = np.zeros((total_n, 2), np.float32)
    for nt in part["notes"]:
        f0 = S1.hz(nt["pitch"])
        vel = nt["vel"] if nt["vel"] is not None else dyn_at(part["dyn"], nt["start"])
        # per-note timbre overrides (so a line can morph, e.g. from the KINDLING's glass to warm)
        ratio = nt["kw"].get("ratio", prm.get("ratio", 1.0))
        index = nt["kw"].get("index", prm.get("index", 0.55))
        lp = nt["kw"].get("lp", prm.get("lp", 3200.0))
        atk = nt["kw"].get("atk", prm.get("atk", 0.006))
        glass = nt["kw"].get("glass", 0.0)          # 0..1: amount of the v1 glass partial (octave + shimmer)
        dec = nt["kw"].get("decay", decay) * (1.0 if f0 < 500 else (500 / f0) ** 0.45)
        dur_s = nt["dur"] * BEAT_S
        n = int(min(dec * 4.2 + dur_s * 0.2, 9.0) * SR)
        tt = _t(n)
        chans = []
        for side in (-1, 1):
            det = 2 ** (side * rng.uniform(0.5, 1.0) * chorus / 1200)
            fc = f0 * det
            ie = index * (0.25 + 0.75 * np.exp(-tt / 0.35)) * (0.5 + 0.7 * vel)
            mod = np.sin(TWOPI * fc * ratio * tt + rng.uniform(0, 6.28)) * ie
            car = np.sin(TWOPI * fc * tt + mod)
            # a soft second partial (tine-like body) and a breath of the octave
            body = 0.18 * np.sin(TWOPI * fc * 2.0 * tt) * np.exp(-tt / (dec * 0.25))
            if glass > 0:
                body = body + glass * 0.25 * np.sin(TWOPI * fc * 2.0 * tt + 0.3 * np.sin(TWOPI * fc * 5.02 * tt)
                                                    * np.exp(-tt / 0.05)) * np.exp(-tt / (dec * 0.35))
            env = (1 - np.exp(-tt / atk)) * np.exp(-tt / dec)
            # damp at note end if the note is short (let long notes ring)
            rs = int((dur_s + 0.15) * SR)
            if nt["kw"].get("damp", False) and rs < n:
                env[rs:] *= np.exp(-np.arange(n - rs) / (0.18 * SR))
            chans.append((car + body) * env * vel)
        y = np.stack(chans, 1)
        # velocity-following lowpass: soft notes are darker
        fc_lp = min(SR * 0.45, lp * (0.55 + 0.6 * vel))
        b, a = signal.butter(2, fc_lp / (SR / 2))
        y = signal.lfilter(b, a, y, axis=0).astype(np.float32) * 0.33
        pan = nt["pan"] if nt["pan"] is not None else 0.0
        if pan:
            th = (pan + 1) * np.pi / 4
            y = y * np.array([np.cos(th), np.sin(th)], np.float32) * 1.414
        S1._place(out, y, int(round(nt["start"] * BEAT_S * SR)))
    return out


def deephit(part, total_n):
    rng = np.random.default_rng(part["seed"])
    out = np.zeros((total_n, 2), np.float32)
    for nt in part["notes"]:
        vel = nt["vel"] if nt["vel"] is not None else 0.8
        f_end = S1.hz(nt["pitch"])
        size = nt["kw"].get("size", 1.0)
        n = int(3.5 * size * SR)
        tt = _t(n)
        f = f_end * (1 + 0.9 * np.exp(-tt / 0.045))
        body = np.sin(TWOPI * np.cumsum(f) / SR) * np.exp(-tt / (0.55 * size))
        body = np.tanh(body * 1.5) / np.tanh(1.5)
        h2 = 0.25 * np.sin(TWOPI * np.cumsum(f * 2.01) / SR) * np.exp(-tt / (0.2 * size))
        boom = rng.normal(0, 1, n) * np.exp(-tt / (0.09 * size))
        b, a = signal.butter(2, [45 / (SR / 2), 260 / (SR / 2)], btype="band")
        boom = signal.lfilter(b, a, boom) * 1.8
        skin = rng.normal(0, 1, n) * np.exp(-tt / 0.006)
        b, a = signal.butter(2, [300 / (SR / 2), 1800 / (SR / 2)], btype="band")
        skin = signal.lfilter(b, a, skin) * nt["kw"].get("skin", 0.25)
        y = body * 0.9 + h2 + boom * 0.5 + skin
        a0 = int(0.0015 * SR)
        y[:a0] *= np.linspace(0, 1, a0)
        side = rng.normal(0, 1, n) * np.exp(-tt / 0.25)
        b, a = signal.butter(2, [120 / (SR / 2), 900 / (SR / 2)], btype="band")
        side = signal.lfilter(b, a, side) * 0.12
        st = np.stack([y + side, y - side], 1) * vel * 0.5
        S1._place(out, st.astype(np.float32), int(round(nt["start"] * BEAT_S * SR)))
    return out


def subpulse(part, total_n):
    out = np.zeros((total_n, 2), np.float32)
    for nt in part["notes"]:
        f0 = S1.hz(nt["pitch"])
        dur_s = nt["dur"] * BEAT_S
        rel = nt["kw"].get("rel", 0.6)
        atk = nt["kw"].get("atk", 0.02)
        n = int((dur_s + rel) * SR)
        tt = _t(n)
        y = np.sin(TWOPI * f0 * tt) + 0.08 * np.sin(2 * TWOPI * f0 * tt)
        if part["dyn"]:
            lev = dyn_array(part["dyn"], nt["start"], n, BEAT_N)
        else:
            lev = np.full(n, nt["vel"] if nt["vel"] is not None else 0.5)
        g = 10 ** (30 * np.log10(np.maximum(lev, 0.03)) / 20)
        e = S1._env_adsr(n, atk, rel, rel_start=int(dur_s * SR))
        y = (y * g * e * 0.35).astype(np.float32)
        S1._place(out, np.stack([y, y], 1), int(round(nt["start"] * BEAT_S * SR)))
    return out


VOICES = dict(S1.VOICES)
VOICES.update(fmwarm=fmwarm, deephit=deephit, subpulse=subpulse)


def render_part(part, total_n):
    return VOICES[part["inst"]](part, total_n)
