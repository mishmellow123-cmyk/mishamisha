"""Note audit for a v2 second half (my eyes on the page):

    python notes_v2.py AB [bar_from bar_to]

For every half beat: the bass, the sounding pitches, a chord guess, and flags for
semitone clashes (minor 2nds / major 7ths / minor 9ths between sustained notes that are
not both passing), low clusters (<= 2 semitones below C3) and notes outside a patch's
sampled range.  Writes out/v2/analysis_<cut>/notes.txt.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from dsl import name as nname  # noqa: E402
from timeline_v2 import gb  # noqa: E402

UNPITCHED = {"bdrum", "giant", "tenor", "tenor_hi", "snare", "snare_roll", "cym", "crash", "swell", "swell_s",
             "gong", "triangle", "taiko", "tick", "riser", "shepard", "revcym", "impact", "subdrop", "deephit",
             "bdrum_roll", "clash", "giant_mallet", "tenor_lo", "cym_swell_med", "cym_swell_short"}
PCN = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]
TEMPL = {"": (0, 4, 7), "m": (0, 3, 7), "7": (0, 4, 7, 10), "maj7": (0, 4, 7, 11), "m7": (0, 3, 7, 10),
         "sus4": (0, 5, 7), "sus2": (0, 2, 7), "add9": (0, 2, 4, 7), "m(add9)": (0, 2, 3, 7), "dim": (0, 3, 6),
         "maj9#11": (0, 2, 4, 6, 7, 11), "7sus4": (0, 5, 7, 10), "m9": (0, 2, 3, 7, 10), "maj9": (0, 2, 4, 7, 11),
         "6": (0, 4, 7, 9)}


def chord_guess(pcs, bass):
    best = None
    for r in range(12):
        for nm, iv in TEMPL.items():
            tpl = {(r + x) % 12 for x in iv}
            if not pcs <= tpl:
                continue
            score = len(tpl) - len(pcs) - (0.5 if r == bass % 12 else 0) + (0 if (r in pcs) else 1)
            if best is None or score < best[0]:
                best = (score, f"{PCN[r]}{nm}" + ("" if r == bass % 12 else f"/{PCN[bass % 12]}"))
    return best[1] if best else "?"


def main(cut, b0=16, b1=38):
    import sampler
    import sampler_v2  # noqa: F401
    import score_v2
    parts = score_v2.build(cut)
    out = []
    warn = []
    # range check against the sampler regions (sampler parts only)
    for pn, p in parts.items():
        if p.kind != "sampler" or p.inst in UNPITCHED:
            continue
        for nt in p.notes:
            if nt.start < gb(16):
                continue
            pk = nt.art or p.inst
            rg = sampler.regions(pk)
            lo = min(r["lokey"] for r in rg)
            hi = max(r["hikey"] for r in rg)
            k = nt.pitch + sampler.KEY_OFFSET.get(pk, 0)
            if k < lo - 1 or k > hi + 2:
                warn.append(f"RANGE {pn}: {nname(nt.pitch)} at beat {nt.start:.2f} (bar {int(nt.start // 4) + 1}) "
                            f"outside {nname(lo)}-{nname(hi)}")
    t = gb(b0)
    step = 0.5
    while t < gb(b1) - 1e-6:
        snd = []
        for pn, p in parts.items():
            if p.inst in UNPITCHED or pn.startswith("v2r_") or p.kind == "synth" and p.inst in ("deephit",):
                continue
            for nt in p.notes:
                if nt.pitch is None:
                    continue
                if nt.start <= t + 1e-6 < nt.start + nt.dur - 0.06:
                    long_ = nt.dur >= 0.49
                    snd.append((int(round(nt.pitch)), pn, long_))
        if snd:
            ps = sorted(set(q for q, _, _ in snd))
            bass = ps[0]
            pcs = {q % 12 for q in ps}
            ch = chord_guess(pcs, bass)
            bar = int(t // 4) + 1
            bt = t % 4 + 1
            out.append(f"{t:6.1f} {bar:3d}.{bt:<4.1f} [{ch:14s}] " + " ".join(nname(q) for q in ps))
            longs = sorted(set(q for q, _, lg in snd if lg))
            for i, x in enumerate(longs):
                for y in longs[i + 1:]:
                    iv = (y - x)
                    # a minor 2nd anywhere; a minor 9th / major 7th only low in the texture (upper
                    # major 7ths are the intended colour of the maj7 / lydian chords)
                    if iv == 1 or (iv == 13 and x < 60) or (iv == 11 and x < 55):
                        who = sorted(set(pn for q, pn, _ in snd if q in (x, y)))
                        warn.append(f"CLASH bar {bar}.{bt:.1f}: {nname(x)}-{nname(y)} ({','.join(who)})")
            low = [q for q in ps if q < 48]
            for x, y in zip(low[:-1], low[1:]):
                if y - x <= 2:
                    warn.append(f"LOW CLUSTER bar {bar}.{bt:.1f}: {nname(x)}-{nname(y)}")
        t += step
    AN = os.path.join(os.path.dirname(HERE), "out", "v2", f"analysis_{cut}")
    os.makedirs(AN, exist_ok=True)
    seen = []
    for w in warn:
        if w not in seen:
            seen.append(w)
    open(os.path.join(AN, "notes.txt"), "w").write("\n".join(seen + [""] + out) + "\n")
    print("\n".join(seen))
    print(f"{len(seen)} warnings; half-beat listing in {os.path.join(AN, 'notes.txt')}")


if __name__ == "__main__":
    c = sys.argv[1].upper() if len(sys.argv) > 1 else "AB"
    a = [int(x) for x in sys.argv[2:4]] if len(sys.argv) > 3 else [16, 38]
    main(c, *a)
