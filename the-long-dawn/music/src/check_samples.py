"""Audit every pitched sampler region: FFT f0 of the sustained portion vs.
the SFZ pitch_keycenter (+tune).  Writes analysis/sample_pitch.txt and
cache/pitch_fix.json (per-sample cent corrections used by the sampler when
a sample is measurably out of tune)."""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import sampler as S  # noqa: E402

PITCHED = ["flute", "flute_nv", "flute_stac", "horn", "horn_stac", "trumpet",
           "trumpet_vib", "trumpet_stac", "trombone", "trombone_stac", "tuba",
           "tuba_stac", "vln", "vln_spic", "vln_trem", "vln_pizz", "vla",
           "vla_spic", "vla_trem", "vc", "vc_spic", "vc_trem", "vc_pizz", "cb",
           "cb_spic", "cb_trem", "cb_pizz", "svln", "harp", "piano", "organ",
           "organ_q", "organ_ped", "glock", "timp", "timp_roll", "oboe",
           "clarinet", "bassoon", "tubular"]


def f0_estimate(x, sr, expect):
    n = len(x)
    win = np.hanning(n)
    X = np.abs(np.fft.rfft(x * win, n=1 << 18))
    fr = np.fft.rfftfreq(1 << 18, 1 / sr)
    df = fr[1]

    def mag(f):
        i = f / df
        i0 = np.floor(i).astype(int)
        u = i - i0
        i0 = np.clip(i0, 0, len(X) - 2)
        return X[i0] * (1 - u) + X[i0 + 1] * u

    cands = expect * 2 ** (np.arange(-2600, 2601, 2) / 1200.0)
    cands = cands[(cands > 25) & (cands < 5000)]
    ks = np.arange(1, 9)
    w = 0.9 ** ks
    H = (mag(np.outer(cands, ks)) * w).sum(1)
    Hh = (mag(np.outer(cands, ks + 0.5)) * w).sum(1)
    score = H - 0.6 * Hh
    f = cands[np.argmax(score)]
    # refine with parabolic peak around the fundamental or strongest harmonic
    return f


def main():
    lines = []
    fixes = {}
    bad = 0
    for pk in PITCHED:
        regs = S.regions(pk)
        seen = set()
        for r in regs:
            if r["path"] in seen:
                continue
            seen.add(r["path"])
            x, sr = S.load_raw(r["path"])
            mt = S.meta(r["path"])
            on = int(mt["onset"] * sr)
            dur = len(x) / sr - mt["onset"]
            if dur > 1.6 and S.P[pk]["kind"] == "sus":
                a, b = on + int(0.4 * sr), on + int(1.4 * sr)
            else:
                a, b = on + int(0.03 * sr), on + int(min(0.6, dur) * sr)
            seg = x[a:b].mean(axis=1)
            if len(seg) < 2048:
                continue
            exp_f = 440 * 2 ** ((r["center"] + r["tune"] / 100 - 69) / 12)
            f = f0_estimate(seg, sr, exp_f)
            cents = 1200 * np.log2(f / exp_f)
            oct_err = int(np.round(cents / 1200))
            fine = cents - 1200 * oct_err
            flag = ""
            if abs(fine) > 25 or oct_err != 0:
                flag = "  <-- CHECK"
                bad += 1
            rel = os.path.relpath(r["path"], S.VSCO)
            lines.append(f"{pk:13s} {rel:70s} key {r['center']:3d} exp {exp_f:8.2f} Hz  meas {f:8.2f} Hz  {cents:+7.1f} c{flag}")
            if oct_err == 0 and 12 < abs(fine) < 60 and pk not in ("timp", "timp_roll", "tubular", "glock"):
                fixes[rel] = -round(float(fine), 1)
    S._save_meta()
    out = os.path.join(S.MUSIC, "analysis", "sample_pitch.txt")
    with open(out, "w") as fh:
        fh.write("\n".join(lines) + f"\n\n{bad} regions flagged\n")
    json.dump(fixes, open(os.path.join(S.CACHE, "pitch_fix.json"), "w"), indent=1)
    print(f"{len(lines)} samples checked, {bad} flagged, {len(fixes)} fine-tune fixes")


if __name__ == "__main__":
    main()
