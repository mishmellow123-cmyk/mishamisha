"""THE LONG DAWN v3 - who dominates a window?  (COMPOSER-A: a balance tool; reads the part cache, renders nothing)

    python who_v3.py A final_A t0 t1 [t0 t1 ...]      # (fallback_<cut> for a fallback master)

per part: loudness (LUFS, dry, part gain applied, K-weighted) inside each window, from the cached part stems"""
import json
import os
import sys

import numpy as np
import pyloudnorm as pyln

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import render_v3 as R  # noqa: E402
from timeline_v3 import BarMap, SR  # noqa: E402

cut, name = sys.argv[1], sys.argv[2]
wins = [(float(a), float(b)) for a, b in zip(sys.argv[3::2], sys.argv[4::2])]
bm = BarMap(cut)
if name.startswith("fallback"):
    import fallback_v3
    S = fallback_v3.build(bm)
else:
    S = __import__(f"score_v3_{cut}").build(bm)
man = json.load(open(os.path.join(R.CACHE, f"manifest_{name}.json")))
meter = pyln.Meter(SR)
for t0, t1 in wins:
    rows = []
    a, b = int(t0 * SR), int(t1 * SR)
    for pn, p in S.used().items():
        if pn not in man or not os.path.exists(R.stem_path(pn, man[pn]) + ".npy"):
            continue
        off, y = R.load_stem(pn, man[pn])
        s0, s1 = max(a, off), min(b, off + len(y))
        if s1 - s0 < int(0.4 * SR):
            continue
        seg = np.zeros((b - a, 2), np.float64)
        seg[s0 - a:s1 - a] = y[s0 - off:s1 - off] * 10 ** (p.gain_db / 20)
        L = meter.integrated_loudness(seg)
        if np.isfinite(L) and L > -70:
            rows.append((L, pn))
    rows.sort(reverse=True)
    print(f"--- {t0:.2f}-{t1:.2f} s: " + ", ".join(f"{pn} {L:.1f}" for L, pn in rows[:9]))
