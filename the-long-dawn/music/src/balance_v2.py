"""Per-part balance in time windows (fader applied, before pan/reverb): which parts
dominate a section, and in which band.   python balance_v2.py AB 1520 1680 [1680 1920 ...]"""
import json, os, sys
import numpy as np
from scipy import signal
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import render_v2 as R
from timeline_v2 import SR, FPS

def main(cut, wins):
    import score_v2
    parts = score_v2.build(cut)
    man = json.load(open(os.path.join(R.CACHE, f"manifest_{cut}.json")))
    sos_lo = signal.butter(4, 250 / (SR / 2), 'low', output='sos')
    sos_hi = signal.butter(4, 2000 / (SR / 2), 'high', output='sos')
    for f0, f1 in wins:
        a, b = int(f0 / FPS * SR), int(f1 / FPS * SR)
        rows = []
        for name, p in parts.items():
            if name not in man:
                continue
            off, y = R.load_stem(name, man[name])
            s0, s1 = max(a, off), min(b, off + len(y))
            if s1 <= s0:
                continue
            seg = np.asarray(y[s0 - off:s1 - off], np.float64).mean(1) * 10 ** (p.gain_db / 20)
            n = b - a
            e = (seg ** 2).sum() / n
            if e < 1e-12:
                continue
            lo = (signal.sosfilt(sos_lo, seg) ** 2).sum() / n
            hi = (signal.sosfilt(sos_hi, seg) ** 2).sum() / n
            rows.append((10 * np.log10(e), 10 * np.log10(lo + 1e-20), 10 * np.log10(hi + 1e-20), name))
        rows.sort(reverse=True)
        tot = 10 * np.log10(sum(10 ** (r[0] / 10) for r in rows))
        print(f"== {cut} frames {f0}-{f1}: total {tot:.1f} dB")
        for r in rows[:22]:
            print(f"   {r[3]:14s} {r[0]:7.1f} dB   <250Hz {r[1]:7.1f}   >2k {r[2]:7.1f}")

if __name__ == "__main__":
    cut = sys.argv[1].upper()
    v = [int(x) for x in sys.argv[2:]]
    main(cut, list(zip(v[0::2], v[1::2])))
