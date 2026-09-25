"""Analysis = my ears.

Writes music/analysis/:
  report.txt        sync table (expected vs measured onsets), LUFS per section,
                    true peak, mono compatibility, DC, clicks
  overview.png      envelope + short-term loudness with sections & sync marks
  sync.png          zoomed envelopes around every sync point
  spec_<sec>.png    log-frequency spectrogram per section
  notes.txt         sounding pitches per half-beat + range / voicing checks
"""
import os
import sys

import numpy as np
import soundfile as sf
from scipy import signal

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pyloudnorm as pyln  # noqa: E402

from dsl import SR, name as nname  # noqa: E402

MUSIC = os.path.dirname(HERE)
OUT = os.path.join(MUSIC, "out")
AN = os.path.join(MUSIC, "analysis")
PARTS_DIR = os.path.join(MUSIC, "cache", "parts")
os.makedirs(AN, exist_ok=True)

SECTIONS = [("intro", 0, 320), ("kindling", 320, 640), ("race", 640, 960), ("grasp", 960, 1040),
            ("silence", 1040, 1200), ("first_beacon", 1200, 1440), ("beacons", 1440, 1760),
            ("globe", 1760, 1920), ("accord", 1920, 2240), ("dawn", 2240, 2480),
            ("coda", 2480, 2808)]

# (frame, label, source, window_s, kind)  source: 'mix', 'sfx' or a part name
SYNC = [
    (0, "wind + drone from silence", "mix", 0.3, "start"),
    (80, "solo flute theme", "fl", 0.12, "soft"),
    (320, "KINDLING shimmer", "glass", 0.06, "hit"),
    (480, "IGNITION", "mix", 0.05, "hit"),
    (640, "RACE drums enter", "mix", 0.05, "hit"),
    (800, "storm swell / crash", "mix", 0.05, "hit"),
    (960, "GRASP riser begins", "gliss", 0.12, "soft"),
    (1040, "IMPACT", "mix", 0.05, "hit"),
    (1070, "piano D4", "piano", 0.06, "hit"),
    (1106, "piano A4", "piano", 0.06, "hit"),
    (1144, "piano D5", "piano", 0.06, "hit"),
    (1236, "flint strike 1", "sfx", 0.05, "hit"),
    (1262, "flint strike 2", "sfx", 0.05, "hit"),
    (1290, "flint strike 3", "sfx", 0.05, "hit"),
    (1318, "kindling catches (strings bloom)", "vla", 0.15, "soft"),
    (1360, "beacon ROARS / horn", "mix", 0.05, "hit"),
    (1480, "ignition 1 (horns)", "timp", 0.05, "hit"),
    (1540, "ignition 2 (trumpet)", "timp", 0.05, "hit"),
    (1600, "ignition 3 (flute+glock)", "timp", 0.05, "hit"),
    (1650, "ignition 4 (cellos)", "timp", 0.05, "hit"),
    (1690, "ignition 5 (trombones)", "timp", 0.05, "hit"),
    (1730, "ignition 6 (high violins)", "timp", 0.05, "hit"),
    (1760, "globe: choir enters", "mix", 0.05, "hit"),
    (2000, "Oath I", "bell", 0.05, "hit"),
    (2040, "Oath II", "bell", 0.05, "hit"),
    (2080, "Oath III", "bell", 0.05, "hit"),
    (2120, "Oath IV", "bell", 0.05, "hit"),
    (2160, "ring sweep start (harp)", "harp", 0.05, "hit"),
    (2220, "ring sweep end (harp)", "harp", 0.05, "hit"),
    (2240, "CLIMAX - sun breaks", "mix", 0.05, "hit"),
    (2480, "coda: solo flute", "fl", 0.12, "soft"),
    (2640, "child's torch lights beacon", "mix", 0.06, "hit"),
    (2645, "answering fire bell 1", "celesta", 0.04, "hit"),
    (2716, "answering fire bell 12", "celesta", 0.04, "hit"),
    (2720, "final chord", "harp", 0.05, "hit"),
]


def load(name):
    if name in ("mix", "score", "sfx"):
        x, _ = sf.read(os.path.join(OUT, name + ".wav"), dtype="float32")
        return x
    return np.load(os.path.join(PARTS_DIR, name + ".npy"), mmap_mode="r")


def env_db(x, t0, t1, hp=120.0, win=0.002, hop=0.0005):
    a, b = max(0, int(t0 * SR)), min(len(x), int(t1 * SR))
    seg = np.asarray(x[a:b], dtype=np.float64)
    if seg.ndim == 2:
        seg = seg.mean(1)
    if hp:
        sos = signal.butter(2, hp / (SR / 2), btype="high", output="sos")
        seg = signal.sosfilt(sos, seg)
    w, h = int(win * SR), int(hop * SR)
    if len(seg) < w + 1:
        return np.array([t0]), np.array([-120.0])
    frames = np.lib.stride_tricks.sliding_window_view(seg ** 2, w)[::h]
    e = 10 * np.log10(frames.mean(1) + 1e-14)
    t = t0 + (np.arange(len(e)) * h + w / 2) / SR
    return t, e


def measure(x, T, W, kind):
    if kind == "start":
        mono = np.abs(np.asarray(x[: int(3 * SR)]).mean(1))
        idx = np.where(mono > 10 ** (-80 / 20))[0]
        return (idx[0] / SR if len(idx) else None), 0.0
    t, e = env_db(x, T - W - 0.05, T + W + 0.05, hp=120.0 if kind == "hit" else 60.0)
    if kind == "hit":
        d = e[4:] - e[:-4]               # dB rise over 2 ms
        tt = t[2:-2]
        m = (tt >= T - W) & (tt <= T + W)
        if not m.any():
            return None, 0
        i = np.argmax(np.where(m, d, -1e9))
        rise = e[min(len(e) - 1, i + 20)] - e[max(0, i - 10)]
        return float(tt[i]), float(rise)
    # soft: first crossing of (peak in window + 0.3 s) - 20 dB
    t2, e2 = env_db(x, T - W, T + 0.6, hp=60.0, win=0.005, hop=0.001)
    pk = e2.max()
    base = np.median(e2[: max(3, int(len(e2) * 0.1))])
    thr = max(pk - 20, base + 10)
    idx = np.where(e2 > thr)[0]
    return (float(t2[idx[0]]) if len(idx) else None), float(pk - base)


def true_peak_db(x):
    y = signal.resample_poly(x, 4, 1, axis=0)
    return 20 * np.log10(np.abs(y).max() + 1e-12)


def short_term(meter_x, t0, t1, win=3.0, hop=0.5):
    meter = pyln.Meter(SR)
    vals = []
    t = t0
    while t + win <= t1 + 1e-6:
        seg = meter_x[int(t * SR): int((t + win) * SR)]
        l = meter.integrated_loudness(seg.astype(np.float64)) if np.abs(seg).max() > 1e-6 else -120
        vals.append((t + win / 2, l if np.isfinite(l) else -120))
        t += hop
    return vals


def clicks(x, thr_db=-30):
    """Find isolated sample-to-sample jumps far above the local HF level."""
    mono = x.mean(1).astype(np.float64)
    d2 = np.abs(np.diff(mono, 2))
    loc = np.convolve(d2, np.ones(480) / 480, mode="same") + 1e-9
    ratio = d2 / loc
    cand = np.where((ratio > 40) & (d2 > 10 ** (thr_db / 20)))[0]
    return cand


def main():
    lines = []
    mix, score, sfxs = load("mix"), load("score"), load("sfx")
    n = len(mix)
    lines.append(f"length: {n} samples = {n / SR:.4f} s  (target 5616000 = 117.000 s)")
    meter = pyln.Meter(SR)
    L = meter.integrated_loudness(mix.astype(np.float64))
    lines.append(f"mix integrated loudness: {L:.2f} LUFS ; true peak {true_peak_db(mix):.2f} dBTP ; "
                 f"sample peak {20 * np.log10(np.abs(mix).max()):.2f} dBFS")
    lines.append(f"score: {meter.integrated_loudness(score.astype(np.float64)):.2f} LUFS, "
                 f"TP {true_peak_db(score):.2f} dBTP ; sfx: "
                 f"{meter.integrated_loudness(sfxs.astype(np.float64)):.2f} LUFS, TP {true_peak_db(sfxs):.2f} dBTP")
    lines.append(f"DC offset L/R: {mix[:, 0].mean():+.2e} / {mix[:, 1].mean():+.2e}")
    lines.append(f"stem sum error max: {np.abs(score + sfxs - mix).max():.2e}")
    end = mix[-int(0.05 * SR):]
    lines.append(f"last 50 ms peak: {20 * np.log10(np.abs(end).max() + 1e-12):.1f} dBFS ; "
                 f"first sample: {mix[0]}")
    ck = clicks(mix)
    lines.append(f"click candidates: {len(ck)}" + (f" at {[round(c / SR, 3) for c in ck[:12]]}" if len(ck) else ""))
    # mono compatibility
    lines.append("")
    lines.append("section            LUFS(int)  ST-max  ST-min   TP(dBTP)  L/R corr  mono-stereo(dB)")
    st_all = short_term(mix, 0, n / SR - 0.01)
    for sec, f0, f1 in SECTIONS:
        a, b = int(f0 / 24 * SR), min(n, int(f1 / 24 * SR))
        seg = mix[a:b].astype(np.float64)
        li = meter.integrated_loudness(seg) if (b - a) > 0.4 * SR else float("nan")
        st = [v for tt, v in st_all if f0 / 24 <= tt < f1 / 24]
        corr = np.corrcoef(seg[:, 0], seg[:, 1])[0, 1] if seg.std() > 0 else 1
        mono = seg.mean(1, keepdims=True).repeat(2, 1)
        lm = meter.integrated_loudness(mono) if (b - a) > 0.4 * SR else float("nan")
        lines.append(f"{sec:14s} {li:10.2f} {max(st) if st else -120:7.1f} {min(st) if st else -120:7.1f} "
                     f"{true_peak_db(seg):9.2f} {corr:9.3f} {lm - li:12.2f}")
    # sync
    lines.append("")
    lines.append("frame  time(s)   measured   offset(ms)  source     event")
    cache = {}
    results = []
    for fr, label, src, W, kind in SYNC:
        if src not in cache:
            try:
                cache[src] = load(src)
            except FileNotFoundError:
                cache[src] = None
        x = cache[src]
        T = fr / 24.0
        if x is None:
            lines.append(f"{fr:5d} {T:8.3f}   (no source {src})")
            continue
        tm, rise = measure(x, T, W, kind)
        off = (tm - T) * 1000 if tm is not None else float("nan")
        ok = "OK" if (tm is not None and abs(off) <= 10) else ("soft" if kind == "soft" else "CHECK")
        results.append((fr, label, src, T, tm, off))
        lines.append(f"{fr:5d} {T:8.3f} {tm if tm is not None else float('nan'):10.4f} {off:+10.1f}  "
                     f"{src:9s}  {label} [{ok}{'' if kind != 'hit' else f', rise {rise:.0f} dB'}]")
    open(os.path.join(AN, "report.txt"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    plots(mix, st_all, results)


def plots(mix, st_all, results):
    # overview
    fig, ax = plt.subplots(2, 1, figsize=(18, 7), sharex=True)
    t, e = env_db(mix, 0, len(mix) / SR, hp=None, win=0.05, hop=0.025)
    ax[0].plot(t, e, lw=0.6, color="#333")
    ax[0].set_ylabel("RMS 50ms (dBFS)")
    ax[0].set_ylim(-80, 0)
    tt = [a for a, b in st_all]
    vv = [b for a, b in st_all]
    ax[1].plot(tt, vv, color="#c44", lw=1.2, label="short-term LUFS (3 s)")
    ax[1].axhline(-16, color="#888", ls=":", lw=0.8)
    ax[1].axhline(-35, color="#88c", ls=":", lw=0.8)
    ax[1].set_ylim(-60, -5)
    ax[1].set_ylabel("LUFS")
    for a in ax:
        for sec, f0, f1 in SECTIONS:
            a.axvline(f0 / 24, color="#48a", lw=0.8, alpha=0.6)
        for fr, label, src, T, tm, off in results:
            a.axvline(T, color="#e90", lw=0.5, alpha=0.5)
    for sec, f0, f1 in SECTIONS:
        ax[0].text(f0 / 24 + 0.3, -5, sec, fontsize=8, color="#248")
    ax[1].set_xlabel("time (s)")
    ax[1].set_xlim(0, 117)
    fig.tight_layout()
    fig.savefig(os.path.join(AN, "overview.png"), dpi=80)
    plt.close(fig)
    # sync zooms
    k = len(results)
    cols = 6
    rows = int(np.ceil(k / cols))
    fig, axs = plt.subplots(rows, cols, figsize=(20, 2.3 * rows))
    for i, (fr, label, src, T, tm, off) in enumerate(results):
        a = axs.flat[i]
        x = load(src) if src not in ("mix",) else mix
        t, e = env_db(x, T - 0.25, T + 0.35, hp=120, win=0.002, hop=0.001)
        a.plot((t - T) * 1000, e, lw=0.7)
        a.axvline(0, color="g", lw=1)
        if tm is not None:
            a.axvline((tm - T) * 1000, color="r", ls="--", lw=0.8)
        a.set_title(f"{fr} {label[:22]} ({off:+.1f} ms)", fontsize=7)
        a.tick_params(labelsize=6)
        a.set_ylim(max(-100, e.max() - 60), e.max() + 3)
    for j in range(i + 1, rows * cols):
        axs.flat[j].axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(AN, "sync.png"), dpi=70)
    plt.close(fig)
    # spectrograms per section
    for sec, f0, f1 in SECTIONS:
        a, b = int(f0 / 24 * SR), min(len(mix), int(f1 / 24 * SR))
        seg = mix[a:b].mean(1)
        f, tt, S = signal.stft(seg, SR, nperseg=4096, noverlap=4096 - 512)
        S = 20 * np.log10(np.abs(S) + 1e-9)
        fig, ax = plt.subplots(figsize=(14, 5))
        sel = (f >= 25) & (f <= 16000)
        ax.pcolormesh(tt + f0 / 24, f[sel], S[sel], vmin=S.max() - 90, vmax=S.max(), shading="auto",
                      cmap="magma")
        ax.set_yscale("log")
        ax.set_ylim(25, 16000)
        ax.set_title(f"{sec}  frames {f0}-{f1}")
        ax.set_xlabel("s")
        for fr, label, src, T, tm, off in results:
            if f0 / 24 <= T < f1 / 24:
                ax.axvline(T, color="c", lw=0.6, alpha=0.7)
        fig.tight_layout()
        fig.savefig(os.path.join(AN, f"spec_{sec}.png"), dpi=70)
        plt.close(fig)


# ---------------------------------------------------------------------------
# note audit
# ---------------------------------------------------------------------------
UNPITCHED = {"bdrum", "giant", "tenor", "tenor_hi", "snare", "snare_roll", "cym", "crash", "swell",
             "swell_s", "gong", "triangle", "taiko", "tick", "riser", "shepard", "revcym", "impact",
             "subdrop"}


def notes_audit():
    import score
    import sampler
    parts = score.build()
    out = []
    # range check vs sampler regions
    for pn, p in parts.items():
        if p.kind != "sampler" or pn in UNPITCHED:
            continue
        regs = sampler.regions(p.inst)
        for nt in p.notes:
            pk = nt.art or p.inst
            rg = sampler.regions(pk)
            lo = min(r["lokey"] for r in rg)
            hi = max(r["hikey"] for r in rg)
            if nt.pitch < lo - 2 or nt.pitch > hi + 3:
                out.append(f"RANGE {pn}: {nnname(nt.pitch)} at beat {nt.start:.2f} outside {lo}-{hi}")
            cands = [r for r in rg if r["lokey"] <= nt.pitch <= r["hikey"]]
            if cands:
                sh = min(abs(nt.pitch - r["center"]) for r in cands)
                if sh > 5:
                    out.append(f"SHIFT {pn}: {nnname(nt.pitch)} shifted {sh} st at beat {nt.start:.2f}")
    # per half-beat sounding pitches (pitched parts)
    grid = np.arange(0, 35 * 4 + 0.5, 0.5)
    out.append("")
    out.append("beat  bar.beat  sounding (low->high)")
    lows = []
    for b in grid:
        ps = []
        for pn, p in parts.items():
            if pn in UNPITCHED or p.inst in ("gliss",):
                continue
            for nt in p.notes:
                if nt.start <= b + 1e-6 < nt.start + nt.dur - 0.05:
                    ps.append((nt.pitch, pn))
        ps = sorted(set(ps))
        names = sorted(set(int(round(q)) for q, _ in ps))
        bar = int(b // 4) + 1
        bt = b % 4 + 1
        out.append(f"{b:6.1f}  {bar:3d}.{bt:<4.1f} " + " ".join(nnname(q) for q in names))
        low = [q for q in names if q < 48]
        for x, y in zip(low[:-1], low[1:]):
            if y - x <= 2:
                lows.append(f"LOW CLUSTER bar {bar}.{bt}: {nnname(x)}-{nnname(y)}")
    # parallel fifths / octaves between tbn/tuba voices at chord changes
    out.append("")
    lowbr = [nt for pn in ("tbn", "tuba") for nt in parts[pn].notes]
    starts = sorted(set(round(nt.start, 3) for nt in lowbr))
    prev = None
    for s0 in starts:
        cur = sorted(int(round(nt.pitch)) for nt in lowbr if abs(nt.start - s0) < 1e-3)
        if prev and len(prev) >= 2 and len(cur) >= 2:
            for i in range(len(prev) - 1):
                for j in range(i + 1, len(prev)):
                    if i < len(cur) and j < len(cur):
                        iv0 = (prev[j] - prev[i]) % 12
                        iv1 = (cur[j] - cur[i]) % 12
                        if iv0 == iv1 and iv0 in (0, 7) and prev[i] != cur[i]:
                            out.append(f"PARALLEL {'5th' if iv0 == 7 else '8ve'} low brass at beat {s0}: "
                                       f"{nnname(prev[i])}-{nnname(prev[j])} -> {nnname(cur[i])}-{nnname(cur[j])}")
        prev = cur
    out.extend(sorted(set(lows)))
    open(os.path.join(AN, "notes.txt"), "w").write("\n".join(out) + "\n")
    print("\n".join(x for x in out if x.startswith(("RANGE", "SHIFT", "PARALLEL", "LOW"))))


def nnname(p):
    return nname(p)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "notes":
        notes_audit()
    else:
        main()
