"""Measure every rendered part stem: level of its loud moments (95th pct of
400 ms RMS while active) and peak, to set mixing faders by role."""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from dsl import SR  # noqa: E402

PARTS_DIR = os.path.join(os.path.dirname(HERE), "cache", "parts")


def stats(name):
    y = np.load(os.path.join(PARTS_DIR, name + ".npy"), mmap_mode="r")
    m = np.asarray(y, dtype=np.float32)
    e = (m ** 2).mean(1)
    w = int(0.4 * SR)
    n = len(e) // w
    r = 10 * np.log10(e[: n * w].reshape(n, w).mean(1) + 1e-14)
    act = r[r > r.max() - 40]
    p95 = np.percentile(act, 95) if len(act) else -120
    pk = 20 * np.log10(np.abs(m).max() + 1e-12)
    return p95, pk


if __name__ == "__main__":
    import score
    parts = score.build()
    rows = []
    for k, p in parts.items():
        if not os.path.exists(os.path.join(PARTS_DIR, k + ".npy")):
            continue
        p95, pk = stats(k)
        rows.append((k, p.bus, p.gain_db, p95, pk))
    for k, bus, g, p95, pk in sorted(rows, key=lambda x: x[1]):
        print(f"{k:12s} {bus:8s} gain {g:+5.1f}  loud(p95) {p95:6.1f} dB  peak {pk:6.1f} dBFS  "
              f"-> with gain {p95 + g:6.1f}")
