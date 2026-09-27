"""THE LONG DAWN v3 - MEASURED impulse responses (no synthesized tails), in the engine's 4-IR true-stereo form
(LL, LR, RL, RR, float32 at 48 kHz, energy-normalised exactly like mix.make_ir so a send keeps its level).

    python ir_v3.py            # build + describe every space (cached in music/cache/sound/ir/<name>.npy)

Spaces (all from Freesound, CC0 unless noted; see music/SFX_CREDITS.md):
  forest20   Nox_Sound  "IR_Forest_20m_Stereo" (558823), 96 kHz stereo, balloon at 20 m: T30 ~0.85 s mid.
             The open air: B's summit and A's ridges (a short diffuse space with no walls), C's council ring.
  church     Sadiquecat "I&R Esperaza's church" centre (816203, OKM1) + entrance (816200), 192 kHz mono pair:
             T30 ~2.2-2.6 s mid, highs shorter.  A real stone church, for a score hall (COMPOSER opt-in).
Note: Voxengo's free IR set is NOT measured (its licence.txt: "All impulses in this archive were created with
Impulse Modeler"), so it is not used.

Each measured response is cleaned the same way: aligned 1 ms before its onset; the balloon's direct pop kept but
10 dB down (a send adds a dry copy otherwise); truncated where the decay meets the noise floor (+3 dB) with a
cos^2 fade over the last 25%; resampled to 48 kHz.
"""
import os
import sys

import numpy as np
import soundfile as sf
from scipy import signal

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
MUSIC = os.path.dirname(HERE)
LIB = os.path.join(MUSIC, "cache", "sound")
IR_DIR = os.path.join(LIB, "ir")
SR = 48000
SPACES = {
    "forest20": dict(kind="stereo", src="558823"),
    "church": dict(kind="pair", src=("816203", "816200")),
}


def _fs(sid):
    d = os.path.join(LIB, "fs")
    for f in os.listdir(d):
        if f.startswith(sid + "_orig."):
            return os.path.join(d, f)
    raise FileNotFoundError(f"Freesound {sid}: run  python freesound_v3.py orig {sid}")


def _read48(p):
    x, sr = sf.read(p, dtype="float64", always_2d=True)
    if sr != SR:
        from math import gcd
        g = gcd(SR, sr)
        x = signal.resample_poly(x, SR // g, sr // g, axis=0)
    return x


def clean(x):
    """x [n, c] at 48 kHz -> cleaned response [m, c]"""
    a = np.abs(x).max(1)
    on = int(np.argmax(a > a.max() * 0.1))
    x = x[max(0, on - int(0.001 * SR)):]
    # the direct pop: 10 dB down over the first 2.5 ms, blended back by 4 ms
    k0, k1 = int(0.0025 * SR), int(0.004 * SR)
    g = np.ones(len(x))
    g[:k0] = 10 ** (-10 / 20)
    g[k0:k1] = np.linspace(10 ** (-10 / 20), 1, k1 - k0)
    x = x * g[:, None]
    # truncate at the noise floor
    e = (x ** 2).sum(1)
    w = int(0.02 * SR)
    sm = 10 * np.log10(np.convolve(e, np.ones(w) / w, mode="same") + 1e-20)
    nz = 10 * np.log10(e[-int(0.1 * len(e)):].mean() + 1e-20)
    cross = np.where(sm < nz + 3)[0]
    cross = cross[cross > int(0.1 * SR)]
    end = int(cross[0]) if len(cross) else len(x)
    x = x[:end]
    f = int(0.25 * len(x))
    x[-f:] *= (np.cos(np.linspace(0, np.pi / 2, f)) ** 2)[:, None]
    return x


def build(name):
    sp = SPACES[name]
    if sp["kind"] == "stereo":
        x = clean(_read48(_fs(sp["src"])))
        L, R = x[:, 0], x[:, 1]
        irs = [L, R, R.copy(), L.copy()]                       # mirrored true stereo
    else:
        A = clean(_read48(_fs(sp["src"][0])))[:, 0]
        B = clean(_read48(_fs(sp["src"][1])))[:, 0]
        n = max(len(A), len(B))
        A, B = np.pad(A, (0, n - len(A))), np.pad(B, (0, n - len(B)))
        irs = [A, B, B.copy(), A.copy()]
    norm = np.sqrt(sum((i ** 2).sum() for i in irs) / 2)
    return [(i / norm).astype(np.float32) for i in irs]


def irs(name):
    os.makedirs(IR_DIR, exist_ok=True)
    p = os.path.join(IR_DIR, f"{name}.npy")
    if os.path.exists(p):
        return list(np.load(p))
    out = build(name)
    n = max(len(i) for i in out)
    np.save(p, np.stack([np.pad(i, (0, n - len(i))) for i in out]))
    return out


if __name__ == "__main__":
    for nm in SPACES:
        r = irs(nm)
        e = r[0].astype(np.float64) ** 2
        edc = 10 * np.log10(np.cumsum(e[::-1])[::-1] / e.sum() + 1e-20)
        i5, i35 = np.argmax(edc < -5), np.argmax(edc < -35)
        print(f"{nm:9s} {len(r[0]) / SR:.2f} s   T30 ~{(i35 - i5) / SR * 2:.2f} s   LL/LR corr "
              f"{np.corrcoef(r[0], r[1])[0, 1]:+.2f}")
