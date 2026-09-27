"""THE LONG DAWN v3 - SOUND lane: the ears I don't have.  Measure a recording before it goes near a film.

    python sound_lab_v3.py report <file> [<file> ...]          # format, clipping, hum, bandwidth, floor, bleed
    python sound_lab_v3.py onsets <file> [--thr 18] [--hf 3000]  # transients (strikes, knocks) with shape
    python sound_lab_v3.py spec <file> <png> [--t0 0 --t1 30]    # a log-frequency spectrogram to look at

What each number means (and what fails):
  clip      samples at |x| >= 0.999 or flat-topped runs (>= 3 equal extreme samples): any run = FAIL for a hero
  hum       prominence (dB) of 50/60 Hz and harmonics over their +-6 Hz neighbourhood: > 12 dB = hum
  bw        the frequency above which the long-term spectrum stays 70 dB under its peak: < 15 kHz = a lossy or
            low-rate source (fine for a distant bed, not for a close hero)
  floor     5th percentile of the 50 ms RMS (dBFS): the room / hiss floor; range = median - floor
  tonal     fraction of 40 ms frames whose 80-1000 Hz band is strongly periodic (voice, music, hum, engines)
  chirp     fraction of frames with a narrow spectral peak 1.8-9 kHz standing 18 dB over its band (birds, whistles,
            squeaks): must be ~0 for B's wind before bar 65
  lf        energy below 80 Hz relative to 80 Hz-16 kHz (dB): > -3 dB with bursts = mic buffeting / handling
"""
import os
import sys

import numpy as np
import soundfile as sf
from scipy import signal


def load(path, mono=False, t0=None, t1=None):
    info = sf.info(path)
    sr = info.samplerate
    start = int(t0 * sr) if t0 else 0
    stop = int(t1 * sr) if t1 else None
    x, sr = sf.read(path, dtype="float64", always_2d=True, start=start, stop=stop)
    if mono:
        x = x.mean(1)
    return x, sr, info


def _frames(x, n, hop):
    m = 1 + max(0, (len(x) - n) // hop)
    idx = np.arange(n)[None, :] + hop * np.arange(m)[:, None]
    return x[idx]


def clip_runs(x):
    runs = 0
    worst = 0
    for c in range(x.shape[1]):
        a = np.abs(x[:, c])
        pk = a.max()
        if pk < 0.5:
            continue
        hot = a >= min(0.999, pk * 0.9995)
        d = np.diff(np.concatenate([[0], hot.astype(np.int8), [0]]))
        s, e = np.where(d == 1)[0], np.where(d == -1)[0]
        ln = e - s
        runs += int((ln >= 3).sum()) if pk < 0.999 else int((ln >= 1).sum())
        worst = max(worst, int(ln.max()) if len(ln) else 0)
    return runs, worst


def hum_db(xm, sr):
    f, p = signal.welch(xm, sr, nperseg=min(len(xm), int(sr * 8)), noverlap=0)
    out = {}
    for base in (50, 60):
        proms = []
        for h in (1, 2, 3):
            f0 = base * h
            k = np.argmin(np.abs(f - f0))
            core = p[max(0, k - 2):k + 3].max()
            ring = np.concatenate([p[(f > f0 - 8) & (f < f0 - 3)], p[(f > f0 + 3) & (f < f0 + 8)]])
            proms.append(10 * np.log10(core / (np.median(ring) + 1e-30) + 1e-30) if len(ring) else 0)
        out[base] = max(proms)
    return out


def bandwidth(xm, sr, below=70):
    f, p = signal.welch(xm, sr, nperseg=4096)
    pdb = 10 * np.log10(p + 1e-30)
    ok = np.where(pdb > pdb.max() - below)[0]
    return float(f[ok[-1]]) if len(ok) else 0.0


def rms_track(xm, sr, win=0.05):
    n = int(win * sr)
    fr = _frames(xm, n, n)
    return 20 * np.log10(np.sqrt((fr ** 2).mean(1)) + 1e-12)


def tonal_frac(xm, sr):
    """share of 40 ms frames whose 80-1000 Hz band has a normalised autocorrelation peak > 0.6 (lag 1-12.5 ms)"""
    sos = signal.butter(4, [80, 1000], btype="band", fs=sr, output="sos")
    y = signal.sosfilt(sos, xm)
    n = int(0.04 * sr)
    fr = _frames(y, n, n)
    e = (fr ** 2).sum(1)
    keep = e > np.percentile(e, 30) * 1.0 + 1e-18
    fr = fr[keep]
    if not len(fr):
        return 0.0
    F = np.fft.rfft(fr, 2 * n, axis=1)
    ac = np.fft.irfft(np.abs(F) ** 2, axis=1)[:, :n]
    ac /= ac[:, :1] + 1e-18
    lo, hi = int(sr / 1000), int(sr / 80)
    pk = ac[:, lo:hi].max(1)
    return float((pk > 0.6).mean())


def chirp_frac(xm, sr, band=(1800, 9000), over=15.0, persist=3):
    """share of frames holding a NARROW tonal peak in `band` (birdsong, whistles, squeaks): each frame's log spectrum
    is whitened by its own local envelope (a ~700 Hz running median), a peak must stand `over` dB above it and
    persist `persist` frames (~35 ms) within +-2 bins.  A steep but smooth spectral tilt (wind) scores ~0."""
    from scipy.ndimage import median_filter
    n = 1024 if sr <= 48000 else 2048
    f, t, S = signal.stft(xm, sr, nperseg=n, noverlap=n // 2)
    L = 10 * np.log10(np.abs(S) ** 2 + 1e-20)
    env = median_filter(L, size=(max(3, int(700 / (f[1] - f[0]))), 1), mode="nearest")
    D = L - env
    m = (f >= band[0]) & (f <= band[1])
    Dm = D[m]
    loud = L.max(0) > np.percentile(L.max(0), 20)
    pk_bin = np.argmax(Dm, axis=0)
    pk_val = Dm.max(0)
    cand = (pk_val > over) & loud
    ok = np.zeros_like(cand)
    run = 0
    for j in range(len(cand)):
        if cand[j] and (j == 0 or not cand[j - 1] or abs(int(pk_bin[j]) - int(pk_bin[j - 1])) <= 2):
            run += 1
        else:
            run = 1 if cand[j] else 0
        if run >= persist:
            ok[j - persist + 1:j + 1] = True
    return float(ok.mean())


def lf_ratio(xm, sr):
    f, p = signal.welch(xm, sr, nperseg=8192)
    lo = p[(f > 15) & (f < 80)].sum()
    mid = p[(f >= 80) & (f < 16000)].sum()
    return 10 * np.log10(lo / (mid + 1e-30) + 1e-30)


def report(path):
    x, sr, info = load(path)
    xm = x.mean(1)
    pk = 20 * np.log10(np.abs(x).max() + 1e-12)
    runs, worst = clip_runs(x)
    r = rms_track(xm, sr)
    fl, md, mx = np.percentile(r, 5), np.median(r), r.max()
    h = hum_db(xm, sr)
    corr = float(np.corrcoef(x[:, 0], x[:, 1])[0, 1]) if x.shape[1] == 2 else 1.0
    name = os.path.basename(path)
    print(f"{name[:34]:34s} {sr / 1000:.1f}k {info.subtype:8s} {x.shape[1]}ch {len(x) / sr:6.1f}s  pk {pk:5.1f}  "
          f"clip {runs}{'!' if runs else ''}  hum50 {h[50]:4.1f} hum60 {h[60]:4.1f}  bw {bandwidth(xm, sr) / 1000:4.1f}k  "
          f"floor {fl:5.1f} med {md:5.1f} max {mx:5.1f}  tonal {tonal_frac(xm, sr):.2f}  chirp {chirp_frac(xm, sr):.3f}  "
          f"lf {lf_ratio(xm, sr):+5.1f}  LRcorr {corr:+.2f}")


def onsets(path, thr=18.0, hf=3000.0, t0=None, t1=None, min_gap=0.12):
    """transients: HF-band (>= hf Hz) envelope rising thr dB over the previous 60 ms minimum"""
    x, sr, _ = load(path, t0=t0, t1=t1)
    xm = x.mean(1)
    sos = signal.butter(4, hf, btype="high", fs=sr, output="sos")
    y = signal.sosfilt(sos, xm)
    hop = int(0.001 * sr)
    w = int(0.002 * sr)
    e = np.sqrt(np.convolve(y ** 2, np.ones(w) / w, mode="same")[::hop] + 1e-18)
    edb = 20 * np.log10(e)
    full = 20 * np.log10(np.sqrt(np.convolve(xm ** 2, np.ones(w) / w, mode="same")[::hop] + 1e-18))
    k = 60
    out = []
    base = t0 or 0.0
    i = k
    n_e = len(edb)
    while i < n_e - 40:
        pre = edb[i - k:i - 3].min()
        if edb[i] - pre > thr:
            p = i + int(np.argmax(edb[i:i + 40]))                 # the peak within 40 ms
            d = np.diff(edb[max(k, i - 15):p + 1])
            on = max(k, i - 15) + int(np.argmax(d)) + 1 if len(d) else i   # the steepest rise
            pkv = edb[p]
            fp = full[p:p + 1500]
            below = np.where(fp < full[p] - 30)[0]
            dec = below[0] * 0.001 if len(below) else 1.5
            hb = np.where(edb[p:p + 400] < pkv - 20)[0]
            burst = hb[0] * 0.001 if len(hb) else 0.4
            out.append((base + on * 0.001, pkv, pkv - pre, (p - on) * 0.001, burst, dec, full[p]))
            i = p + int(min_gap * 1000)
        else:
            i += 1
    return out


def spec(path, png, t0=0.0, t1=None, fmax=20000, title=None, n=2048):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    x, sr, _ = load(path, mono=True, t0=t0, t1=t1)
    hop = max(n // 4, int(len(x) / 1800))                       # at most ~1800 columns
    f, t, S = signal.stft(x, sr, nperseg=n, noverlap=max(0, n - hop))
    P = 20 * np.log10(np.abs(S) + 1e-9)
    fig, ax = plt.subplots(2, 1, figsize=(12, 5), gridspec_kw={"height_ratios": [3, 1]}, sharex=True)
    m = f <= fmax
    vmax = np.percentile(P[m], 99.7)
    # log-frequency rows by interpolation (imshow is fast; pcolormesh on 10^7 cells is not)
    fl = np.geomspace(40, min(fmax, sr / 2), 240)
    Pl = np.array([np.interp(fl, f, P[:, j]) for j in range(P.shape[1])]).T
    ax[0].imshow(Pl, origin="lower", aspect="auto", cmap="magma", vmin=vmax - 80, vmax=vmax,
                 extent=[t0 + t[0], t0 + t[-1], 0, len(fl)])
    ticks = [50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000]
    ticks = [k for k in ticks if 40 <= k <= fl[-1]]
    ax[0].set_yticks([np.interp(np.log(k), np.log(fl), np.arange(len(fl))) for k in ticks])
    ax[0].set_yticklabels([f"{k // 1000}k" if k >= 1000 else str(k) for k in ticks])
    ax[0].set_ylabel("Hz")
    r = rms_track(x, sr, 0.02)
    ax[1].plot(t0 + np.arange(len(r)) * 0.02, r, lw=0.6)
    ax[1].set_ylim(max(-100, r.min() - 3), 0)
    ax[1].set_ylabel("dBFS")
    ax[0].set_title(title or os.path.basename(path), fontsize=9)
    fig.tight_layout()
    fig.savefig(png, dpi=70)
    plt.close(fig)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd")
    ap.add_argument("files", nargs="+")
    ap.add_argument("--thr", type=float, default=18.0)
    ap.add_argument("--hf", type=float, default=3000.0)
    ap.add_argument("--t0", type=float, default=None)
    ap.add_argument("--t1", type=float, default=None)
    ap.add_argument("--fmax", type=float, default=20000)
    ap.add_argument("--nfft", type=int, default=2048)
    a = ap.parse_args()
    if a.cmd == "report":
        for p in a.files:
            try:
                report(p)
            except Exception as e:
                print(os.path.basename(p), "ERROR", e)
    elif a.cmd == "onsets":
        for p in a.files:
            print(f"# {os.path.basename(p)}   t  pk(HF dB)  rise  attack  HFburst  decay30  fullpk")
            for (t, pkv, rise, att, burst, dec, fullpk) in onsets(p, a.thr, a.hf, a.t0, a.t1):
                print(f"  {t:8.3f}  {pkv:6.1f}  {rise:5.1f}  {att * 1000:4.0f}ms  {burst * 1000:4.0f}ms  {dec * 1000:5.0f}ms  "
                      f"{fullpk:6.1f}")
    elif a.cmd == "spec":
        spec(a.files[0], a.files[1], a.t0 or 0.0, a.t1, a.fmax, n=a.nfft)


def ir_stats(path):
    """a measured impulse response: onset, peak-to-noise, broadband and octave-band T30 (Schroeder, noise-compensated
    by truncating where the decay meets the noise floor)"""
    x, sr, info = load(path)
    xm = x.mean(1)
    a = np.abs(xm)
    on = int(np.argmax(a > a.max() * 0.1))
    y = xm[on:]
    tail = y[-int(0.1 * len(y)):]
    noise = 10 * np.log10((tail ** 2).mean() + 1e-20)
    peak = 10 * np.log10((y[:int(0.005 * sr)] ** 2).max() + 1e-20)
    out = []
    for lo, hi in [(0, 0), (125, 250), (250, 500), (500, 1000), (1000, 2000), (2000, 4000), (4000, 8000)]:
        z = y if lo == 0 else signal.sosfilt(signal.butter(3, [lo, hi], btype="band", fs=sr, output="sos"), y)
        e = z ** 2
        # truncate at the noise crossing (smoothed energy within 3 dB of the tail's level)
        w = int(0.02 * sr)
        sm = 10 * np.log10(np.convolve(e, np.ones(w) / w, mode="same") + 1e-20)
        nz = 10 * np.log10(e[-int(0.1 * len(e)):].mean() + 1e-20)
        cross = np.where(sm < nz + 3)[0]
        cross = cross[cross > int(0.05 * sr)]
        end = cross[0] if len(cross) else len(e)
        edc = np.cumsum(e[:end][::-1])[::-1]
        edc = 10 * np.log10(edc / edc[0] + 1e-20)
        i5 = np.argmax(edc < -5)
        i35 = np.argmax(edc < -35) if (edc < -35).any() else None
        i25 = np.argmax(edc < -25) if (edc < -25).any() else None
        if i35:
            t = (i35 - i5) / sr * 2
            tag = "T30"
        elif i25:
            t = (i25 - i5) / sr * 3
            tag = "T20"
        else:
            t, tag = float("nan"), "--"
        out.append((lo, hi, t, tag, end / sr))
    print(f"{os.path.basename(path)[:36]:36s} {sr / 1000:.0f}k {x.shape[1]}ch {len(x) / sr:.2f}s onset {on / sr * 1000:.0f}ms  "
          f"peak/noise {peak - noise:.0f} dB")
    print("    " + "  ".join(f"{'all' if lo == 0 else lo}:{t:.2f}s({tag},{end:.1f})" for lo, hi, t, tag, end in out))


if __name__ == "__main__" and sys.argv[1] == "ir":
    for p in sys.argv[2:]:
        ir_stats(p)


def sync_check(path, times, win=0.06, hf=2000.0):
    """for each expected hit time: the peak of the HF (>= hf) envelope and the steepest rise within +-win s"""
    x, sr, _ = load(path, t0=max(0, min(times) - 1.0), t1=max(times) + 1.0)
    base = max(0, min(times) - 1.0)
    xm = x.mean(1)
    y = signal.sosfilt(signal.butter(4, hf, btype="high", fs=sr, output="sos"), xm)
    w = int(0.001 * sr)
    e = np.sqrt(np.convolve(y ** 2, np.ones(w) / w, mode="same") + 1e-18)
    edb = 20 * np.log10(e)
    out = []
    for T in times:
        a, b = int((T - base - win) * sr), int((T - base + win) * sr)
        seg = edb[a:b]
        pk = a + int(np.argmax(seg))
        d = np.diff(edb[max(a, pk - int(0.02 * sr)):pk + 1])
        on = max(a, pk - int(0.02 * sr)) + int(np.argmax(d)) + 1 if len(d) else pk
        out.append((T, (base + on / sr - T) * 1000, (base + pk / sr - T) * 1000, float(edb[pk])))
    return out


if __name__ == "__main__" and sys.argv[1] == "sync":
    ts = [float(v) for v in sys.argv[3].split(",")]
    for T, don, dpk, lv in sync_check(sys.argv[2], ts):
        print(f"  {T:9.3f} s: steepest onset {don:+6.1f} ms, peak {dpk:+6.1f} ms  ({lv:.1f} dB)")
