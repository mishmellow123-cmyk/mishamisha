"""v2 analysis = my ears.   python analyze_v2.py AB   (or C)

Writes music/out/v2/analysis_<cut>/:
  report.txt       format checks, LUFS / true peak (whole + per section), breath depths,
                   the sync table (expected vs measured onsets), stem-sum error, clicks
  overview.png     RMS envelope + short-term loudness, sections and sync marks
  sync.png         zoomed onset envelopes around every sync point
  spec_<sec>.png   log-frequency spectrogram per section
"""
import json
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

from timeline_v2 import SR, FPS, TOTAL_N, SECTIONS  # noqa: E402

MUSIC = os.path.dirname(HERE)
OUT = os.path.join(MUSIC, "out", "v2")
CACHE = os.path.join(MUSIC, "cache", "v2")


def load_wav(name):
    x, sr = sf.read(os.path.join(OUT, name + ".wav"), dtype="float32", always_2d=True)
    assert sr == SR
    return x


def load_part(name, cut):
    man = json.load(open(os.path.join(CACHE, f"manifest_{cut}.json")))
    import render_v2 as R
    off, y = R.load_stem(name, man[name])
    full = np.zeros((TOTAL_N + SR, 2), np.float32)
    e = min(len(full), off + len(y))
    full[off:e] = y[:e - off]
    return full


def env_db(x, t0, t1, hp=120.0, win=0.002, hop=0.0005):
    a, b = max(0, int(t0 * SR)), min(len(x), int(t1 * SR))
    seg = np.asarray(x[a:b], dtype=np.float64)
    if seg.ndim == 2:
        seg = seg.mean(1)
    if hp:
        sos = signal.butter(2, hp / (SR / 2), btype="high", output="sos")
        seg = signal.sosfilt(sos, seg)
    w, h = int(win * SR), max(1, int(hop * SR))
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
        k = 16                                  # 8 ms at 0.5 ms hop
        cs = np.concatenate([[0.0], np.cumsum(e)])
        idx = np.arange(k, len(e) - k)
        after = (cs[idx + k] - cs[idx]) / k
        before = (cs[idx] - cs[idx - k]) / k
        d = after - before
        tt = t[idx]
        m = (tt >= T - W) & (tt <= T + W)
        if not m.any():
            return None, 0
        j = int(np.argmax(np.where(m, d, -1e9)))
        # refine: the steepest 2 ms slope within 4 ms of the sustained-rise point
        i = idx[j]
        lo, hi = max(2, i - 8), min(len(e) - 3, i + 8)
        sl = e[lo + 2:hi + 2] - e[lo - 2:hi - 2]
        i2 = lo + int(np.argmax(sl))
        return float(t[i2]), float(d[j])
    t2, e2 = env_db(x, T - W, T + 0.6, hp=60.0, win=0.005, hop=0.001)
    pk = e2.max()
    base = np.median(e2[: max(3, int(len(e2) * 0.1))])
    thr = max(pk - 20, base + 10)
    idx = np.where(e2 > thr)[0]
    return (float(t2[idx[0]]) if len(idx) else None), float(pk - base)


def true_peak_db(x):
    import render_v2 as R
    return R.true_peak_db(x)


def short_term(x, t0, t1, win=3.0, hop=0.5):
    meter = pyln.Meter(SR)
    vals = []
    t = t0
    while t + win <= t1 + 1e-6:
        seg = x[int(t * SR): int((t + win) * SR)]
        v = meter.integrated_loudness(seg.astype(np.float64)) if np.abs(seg).max() > 1e-6 else -120
        vals.append((t + win / 2, v if np.isfinite(v) else -120))
        t += hop
    return vals


def clicks(x, thr_db=-30):
    mono = x.mean(1).astype(np.float64)
    d2 = np.abs(np.diff(mono, 2))
    loc = np.convolve(d2, np.ones(480) / 480, mode="same") + 1e-9
    return np.where((d2 / loc > 40) & (d2 > 10 ** (thr_db / 20)))[0]


def rms_db(x):
    x = np.asarray(x, np.float64)
    return 10 * np.log10((x ** 2).mean() + 1e-20)


def breath_report(x, breaths, label):
    """depth of every breath: RMS inside the window (after the 30 ms fade, before the 5 ms
    ramp) vs the 300 ms before it."""
    from timeline_v2 import BEAT_N
    lines = []
    ok = True
    for b0, b1, ex in breaths:
        i0, i1 = int(round(b0 * BEAT_N)), int(round(b1 * BEAT_N))
        a, b = i0 + int(0.036 * SR), i1 - int(0.006 * SR)
        inside = rms_db(x[a:b])
        before = rms_db(x[max(0, i0 - int(0.3 * SR)):i0])
        after = rms_db(x[i1:i1 + int(0.2 * SR)])
        depth = before - inside
        good = depth >= 30 or inside < -60
        ok &= good
        lines.append(f"  {label:6s} breath {i0 / SR:8.3f}-{i1 / SR:8.3f}s (frame {i1 / SR * FPS:7.1f})"
                     f"  before {before:6.1f}  inside {inside:6.1f}  after {after:6.1f} dBFS  depth {depth:5.1f} dB"
                     f"  [{'OK' if good else 'MISSING'}]{' exempt: ' + ','.join(sorted(ex)) if ex else ''}")
    return lines, ok


def main(cut):
    import score_v2
    AN = os.path.join(OUT, f"analysis_{cut}")
    os.makedirs(AN, exist_ok=True)
    lines = [f"THE LONG DAWN v2 - score {cut} - analysis", ""]
    fin, sco, sfx = load_wav(f"final_{cut}"), load_wav(f"score_{cut}"), load_wav(f"sfx_{cut}")
    n = len(fin)
    meter = pyln.Meter(SR)
    L = meter.integrated_loudness(fin.astype(np.float64))
    lines.append(f"length: {n} samples = {n / SR:.4f} s (target {TOTAL_N} = {TOTAL_N / SR:.4f} s) "
                 f"[{'OK' if n == TOTAL_N else 'WRONG'}]")
    info = sf.info(os.path.join(OUT, f"final_{cut}.wav"))
    lines.append(f"format: {info.samplerate} Hz, {info.channels} ch, {info.subtype}")
    lines.append(f"final_{cut}: {L:.2f} LUFS integrated ; true peak {true_peak_db(fin):.2f} dBTP ; "
                 f"sample peak {20 * np.log10(np.abs(fin).max()):.2f} dBFS")
    lines.append(f"score_{cut}: {meter.integrated_loudness(sco.astype(np.float64)):.2f} LUFS, TP "
                 f"{true_peak_db(sco):.2f} dBTP ; sfx_{cut}: {meter.integrated_loudness(sfx.astype(np.float64)):.2f} "
                 f"LUFS, TP {true_peak_db(sfx):.2f} dBTP")
    lines.append(f"stem sum error max |score + sfx - final|: {np.abs(sco + sfx - fin).max():.2e}")
    lines.append(f"DC offset L/R: {fin[:, 0].mean():+.2e} / {fin[:, 1].mean():+.2e}")
    lines.append(f"first sample {fin[0]} ; first 10 ms peak {20 * np.log10(np.abs(fin[:480]).max() + 1e-12):.1f} dBFS"
                 f" ; last 50 ms peak {20 * np.log10(np.abs(fin[-int(0.05 * SR):]).max() + 1e-12):.1f} dBFS")
    ck = clicks(sco)
    lines.append(f"click candidates in the score stem: {len(ck)}" +
                 (f" at {[round(c / SR, 3) for c in ck[:12]]}" if len(ck) else ""))
    ck2 = clicks(sfx)
    lines.append(f"sharp transients in the SFX stem (designed flint sparks etc.): {len(ck2)}")
    lines.append("")
    lines.append("BREATHS (every part + the hall + the SFX ducked to -45 dB; the designed exempt part - the")
    lines.append("reversed cymbal rushing into the RACE - keeps playing).  Depth must be >= 30 dB.")
    ok = True
    probe_p = os.path.join(CACHE, f"breath_probe_{cut}.json")
    if os.path.exists(probe_p):
        for pr in json.load(open(probe_p)):
            depth = pr["before_db"] - pr["inside_db"]
            good = depth >= 30
            ok &= good
            lines.append(f"  mix of all non-exempt parts + hall (pre-master)  {pr['start_s']:8.3f}-{pr['end_s']:8.3f}s"
                         f"  before {pr['before_db']:6.1f}  inside {pr['inside_db']:6.1f}  after {pr['after_db']:6.1f} dB"
                         f"  depth {depth:5.1f} dB [{'OK' if good else 'MISSING'}]"
                         f"{'  (exempt: ' + ','.join(pr['exempt']) + ')' if pr['exempt'] else ''}")
    bl, ok1 = breath_report(sco, [b for b in score_v2.BREATHS if not b[2]], "score")
    bl2, ok2 = breath_report(fin, [b for b in score_v2.BREATHS if not b[2]], "final")
    lines += bl + bl2
    lines.append(f"  -> breaths {'ALL PRESENT' if ok and ok1 and ok2 else 'MISSING - CHECK'}")
    lines.append("")
    lines.append("section          frames      LUFS(int)  ST-max  ST-min  TP(dBTP)  L/R corr  mono-stereo(dB)")
    st_all = short_term(fin, 0, n / SR - 0.01)
    for sec, f0, f1 in SECTIONS:
        a, b = int(f0 / FPS * SR), min(n, int(f1 / FPS * SR))
        seg = fin[a:b].astype(np.float64)
        li = meter.integrated_loudness(seg) if (b - a) > 0.4 * SR else float("nan")
        st = [v for tt, v in st_all if f0 / FPS <= tt < f1 / FPS]
        corr = np.corrcoef(seg[:, 0], seg[:, 1])[0, 1] if seg.std() > 0 else 1
        mono = seg.mean(1, keepdims=True).repeat(2, 1)
        lm = meter.integrated_loudness(mono) if (b - a) > 0.4 * SR else float("nan")
        lines.append(f"{sec:15s} {f0:5d}-{f1:<5d} {li:9.2f} {max(st) if st else -120:7.1f} {min(st) if st else -120:7.1f}"
                     f" {true_peak_db(seg):8.2f} {corr:9.3f} {lm - li:12.2f}")
    lines.append("")
    lines.append("SYNC (onset detector: 2 ms dB-slope; OK = within +-10 ms; 'soft' = slow-attack entries)")
    lines.append("frame  time(s)   measured  offset(ms)  source           event")
    cache = {}
    results = []
    for fr, label, src, W, kind in score_v2.sync_table(cut):
        if src not in cache:
            try:
                cache[src] = {"final": fin, "score": sco, "sfx": sfx}.get(src)
                if cache[src] is None:
                    cache[src] = load_part(src, cut)
            except Exception as ex:  # noqa: BLE001
                cache[src] = None
                print("no source", src, ex)
        x = cache[src]
        T = fr / FPS
        if x is None:
            lines.append(f"{fr:5d} {T:8.3f}   (no source {src})")
            continue
        tm, rise = measure(x, T, W, kind)
        off = (tm - T) * 1000 if tm is not None else float("nan")
        ok = "OK" if (tm is not None and abs(off) <= 10) else ("soft" if kind == "soft" else "CHECK")
        results.append((fr, label, src, T, tm, off))
        lines.append(f"{fr:5d} {T:8.3f} {tm if tm is not None else float('nan'):10.4f} {off:+9.1f}   {src:15s}  "
                     f"{label} [{ok}{'' if kind != 'hit' else f', rise {rise:.0f} dB'}]")
    for k in list(cache):
        if k not in ("final", "score", "sfx"):
            cache[k] = None
    open(os.path.join(AN, "report.txt"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    plots(fin, st_all, results, AN, cut)


def plots(mix, st_all, results, AN, cut):
    fig, ax = plt.subplots(2, 1, figsize=(20, 7), sharex=True)
    t, e = env_db(mix, 0, len(mix) / SR, hp=None, win=0.05, hop=0.025)
    ax[0].plot(t, e, lw=0.6, color="#333")
    ax[0].set_ylabel("RMS 50ms (dBFS)")
    ax[0].set_ylim(-80, 0)
    ax[1].plot([a for a, b in st_all], [b for a, b in st_all], color="#c44", lw=1.2)
    ax[1].axhline(-16, color="#888", ls=":", lw=0.8)
    ax[1].set_ylim(-60, -5)
    ax[1].set_ylabel("short-term LUFS (3 s)")
    for a in ax:
        for sec, f0, f1 in SECTIONS:
            a.axvline(f0 / FPS, color="#48a", lw=0.8, alpha=0.6)
        for fr, label, src, T, tm, off in results:
            a.axvline(T, color="#e90", lw=0.5, alpha=0.4)
    for sec, f0, f1 in SECTIONS:
        ax[0].text(f0 / FPS + 0.2, -5, sec, fontsize=7, color="#248")
    ax[1].set_xlabel("time (s)")
    ax[1].set_xlim(0, TOTAL_N / SR)
    fig.suptitle(f"final_{cut}")
    fig.tight_layout()
    fig.savefig(os.path.join(AN, "overview.png"), dpi=70)
    plt.close(fig)
    k = len(results)
    cols = 7
    rows = int(np.ceil(k / cols))
    fig, axs = plt.subplots(rows, cols, figsize=(22, 2.2 * rows))
    for i, (fr, label, src, T, tm, off) in enumerate(results):
        a = axs.flat[i]
        x = mix if src == "final" else None
        if x is None:
            try:
                x = load_wav(f"{src}_{cut}") if src in ("score", "sfx") else load_part(src, cut)
            except Exception:  # noqa: BLE001
                a.axis("off")
                continue
        tt, e = env_db(x, T - 0.25, T + 0.35, hp=120, win=0.002, hop=0.001)
        a.plot((tt - T) * 1000, e, lw=0.7)
        a.axvline(0, color="g", lw=1)
        if tm is not None:
            a.axvline((tm - T) * 1000, color="r", ls="--", lw=0.8)
        a.set_title(f"{fr} {label[:20]} ({off:+.1f})", fontsize=7)
        a.tick_params(labelsize=6)
        a.set_ylim(max(-110, e.max() - 60), e.max() + 3)
        del x
    for j in range(k, rows * cols):
        axs.flat[j].axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(AN, "sync.png"), dpi=60)
    plt.close(fig)
    for sec, f0, f1 in SECTIONS:
        a, b = int(f0 / FPS * SR), min(len(mix), int(f1 / FPS * SR))
        seg = mix[a:b].mean(1)
        f, tt, Sx = signal.stft(seg, SR, nperseg=4096, noverlap=4096 - 512)
        Sx = 20 * np.log10(np.abs(Sx) + 1e-9)
        fig, ax = plt.subplots(figsize=(14, 5))
        sel = (f >= 25) & (f <= 16000)
        ax.pcolormesh(tt + f0 / FPS, f[sel], Sx[sel], vmin=Sx.max() - 90, vmax=Sx.max(), shading="auto",
                      cmap="magma")
        ax.set_yscale("log")
        ax.set_ylim(25, 16000)
        ax.set_title(f"{cut}: {sec}  frames {f0}-{f1}")
        ax.set_xlabel("s")
        for fr, label, src, T, tm, off in results:
            if f0 / FPS <= T < f1 / FPS:
                ax.axvline(T, color="c", lw=0.6, alpha=0.7)
        fig.tight_layout()
        fig.savefig(os.path.join(AN, f"spec_{sec}.png"), dpi=65)
        plt.close(fig)


if __name__ == "__main__":
    main(sys.argv[1].upper() if len(sys.argv) > 1 else "AB")
