"""Piano-roll of a v2 second half (bars 16-37), coloured by family: python roll_v2.py AB"""
import os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from timeline_v2 import gb, fb, SECTIONS

FAM = {"strings": "#3b7dd8", "brass": "#d8843b", "winds": "#3bb07d", "keys": "#9b59b6", "perc": "#555555",
       "synth": "#d83b8e", "choir": "#aaaa33"}

def main(cut):
    import sampler_v2  # noqa
    import score_v2
    parts = score_v2.build(cut)
    fig, ax = plt.subplots(figsize=(26, 11))
    for pn, p in parts.items():
        if not (pn.startswith("ab_") or pn.startswith("c_")):
            continue
        col = FAM.get(p.bus, "#888")
        pitched = p.inst not in ("bdrum", "bdrum_roll", "deephit", "cym_swell_med")
        for nt in p.notes:
            if nt.pitch is None:
                continue
            y = nt.pitch if pitched else 22
            lvl = nt.vel if nt.vel is not None else p.level_at(nt.start)
            ax.add_patch(plt.Rectangle((nt.start, y - 0.4), nt.dur, 0.8, color=col, alpha=0.25 + 0.6 * min(1, lvl)))
    for sec, f0, f1 in SECTIONS:
        if f0 >= 1200:
            ax.axvline(fb(f0), color="k", lw=0.8)
            ax.text(fb(f0) + 0.2, 100, sec, fontsize=9)
    for b in range(16, 39):
        ax.axvline(gb(b), color="#ccc", lw=0.4, zorder=0)
    ax.set_xlim(gb(16), fb(2968)); ax.set_ylim(20, 104)
    ax.set_yticks(range(24, 104, 12)); ax.set_yticklabels([f"C{o}" for o in range(1, 8)])
    ax.set_title(f"score {cut}: second half (bars 16-37)  blue strings, orange brass, green winds, purple keys/organ, pink synth, grey perc")
    fig.tight_layout()
    out = os.path.join(os.path.dirname(HERE), "out", "v2", f"analysis_{cut}", "roll.png")
    fig.savefig(out, dpi=55); print(out)

if __name__ == "__main__":
    main(sys.argv[1].upper())
