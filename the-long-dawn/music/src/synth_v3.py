"""v3 synthesised voices.  Everything in synth.py (v1) and synth_v2.py is available unchanged;
this module adds:

* harmonic - a string harmonic, far off: a near-sine with a slow drift, a breath of bow noise and
             soft upper partials, slow attack; follows the part's dynamics curve (B's "one high
             harmonic" at the far peak; the single held tone of C's Mirror).
"""
import numpy as np
from scipy import signal

import synth as S1
import synth_v2 as S2
from dsl import SR, BEAT_S, BEAT_N, dyn_array

TWOPI = 2 * np.pi


def harmonic(part, total_n):
    rng = np.random.default_rng(part["seed"])
    out = np.zeros((total_n, 2), np.float32)
    for nt in part["notes"]:
        f0 = S1.hz(nt["pitch"])
        dur_s = nt["dur"] * BEAT_S
        atk = nt["kw"].get("atk", 0.9)
        rel = nt["kw"].get("rel", 1.6)
        n = int((dur_s + rel) * SR)
        t = np.arange(n) / SR
        drift = 1.0 + 0.0009 * S1._smooth_noise(rng, n, 0.35) + 0.0006 * np.sin(TWOPI * 4.6 * t) * \
            np.clip(t / 2.5, 0, 1)
        ph = TWOPI * np.cumsum(f0 * drift) / SR
        y = np.sin(ph) + 0.035 * np.sin(2 * ph + 0.4) + 0.012 * np.sin(3 * ph + 1.1)
        bow = rng.normal(0, 1, n)
        b, a = signal.butter(2, [min(0.95, f0 * 0.9 / (SR / 2)), min(0.98, f0 * 1.12 / (SR / 2))], btype="band")
        bow = signal.lfilter(b, a, bow)
        bow *= 0.05 / (np.std(bow) + 1e-9) * (1 + 0.3 * S1._smooth_noise(rng, n, 3.0))
        y = y + bow
        if part["dyn"]:
            lev = dyn_array(part["dyn"], nt["start"], n, BEAT_N)
        else:
            lev = np.full(n, nt["vel"] if nt["vel"] is not None else 0.3)
        g = 10 ** (30 * np.log10(np.maximum(lev, 0.03)) / 20)
        e = S1._env_adsr(n, atk, rel, rel_start=int(dur_s * SR))
        y = (y * g * e * 0.3).astype(np.float32)
        pan = nt["pan"] if nt["pan"] is not None else 0.0
        st = S1._pan(y, pan) if pan else np.stack([y, y], 1)
        S1._place(out, st.astype(np.float32), int(round(nt["start"] * BEAT_S * SR)))
    return out


VOICES = dict(S2.VOICES)
VOICES.update(harmonic=harmonic)


def render_part(part, total_n):
    return VOICES[part["inst"]](part, total_n)
