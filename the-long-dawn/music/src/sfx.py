"""Sound effects stem - everything synthesised (noise shaping, filtered
impulses, resonators).  Each event is a stereo clip that starts on an exact
video frame; the sync point (hit) sits a whole number of frames into the clip
(`lead` frames), so hit_frame = frame + lead.

build_and_render() returns the summed SFX stem (with a small outdoor space)
and writes out/sfx_events/*.wav + out/sfx_events.json.
"""
import json
import os

import numpy as np
import soundfile as sf
from scipy import signal

from dsl import SR

HERE = os.path.dirname(os.path.abspath(__file__))
MUSIC = os.path.dirname(HERE)
OUT = os.path.join(MUSIC, "out")
EV_DIR = os.path.join(OUT, "sfx_events")
FR = SR // 24   # 2000 samples per frame


# ---------------------------------------------------------------------------
# DSP helpers
# ---------------------------------------------------------------------------
def white(rng, n, ch=2):
    return rng.normal(0, 1, (n, ch)).astype(np.float32)


def pink(rng, n, ch=2):
    X = np.fft.rfft(rng.normal(0, 1, (n, ch)), axis=0)
    f = np.fft.rfftfreq(n, 1 / SR)
    f[0] = f[1]
    X /= np.sqrt(f)[:, None]
    y = np.fft.irfft(X, n=n, axis=0)
    return (y / (y.std() + 1e-12)).astype(np.float32)


def brown(rng, n, ch=2):
    X = np.fft.rfft(rng.normal(0, 1, (n, ch)), axis=0)
    f = np.fft.rfftfreq(n, 1 / SR)
    f[0] = f[1]
    X /= f[:, None]
    y = np.fft.irfft(X, n=n, axis=0)
    return (y / (y.std() + 1e-12)).astype(np.float32)


def sos_filter(y, kind, f, order=2):
    if kind == "band":
        f = [max(10, f[0]), min(SR * 0.45, f[1])]
    else:
        f = min(SR * 0.45, max(10, f))
    sos = signal.butter(order, np.array(f) / (SR / 2), btype=kind, output="sos")
    return signal.sosfilt(sos, y, axis=0).astype(np.float32)


def sweep_filter(y, kind, fcurve, q=1.0, blk=512):
    """Time-varying biquad (lowpass / bandpass / highpass), per-block coefficients."""
    n = len(y)
    out = np.zeros_like(y)
    zi = np.zeros((2, y.shape[1]))
    for s in range(0, n, blk):
        f = float(np.clip(fcurve[min(s + blk // 2, n - 1)], 20, SR * 0.45))
        w0 = 2 * np.pi * f / SR
        alpha = np.sin(w0) / (2 * q)
        c = np.cos(w0)
        if kind == "low":
            b = [(1 - c) / 2, 1 - c, (1 - c) / 2]
        elif kind == "high":
            b = [(1 + c) / 2, -(1 + c), (1 + c) / 2]
        else:
            b = [alpha, 0, -alpha]
        a = [1 + alpha, -2 * c, 1 - alpha]
        b = np.array(b) / a[0]
        a = np.array(a) / a[0]
        for ch in range(y.shape[1]):
            seg, zi_ch = signal.lfilter(b, a, y[s:s + blk, ch], zi=zi[:, ch] * 1.0)
            out[s:s + blk, ch] = seg
            zi[:, ch] = zi_ch
    return out


def smooth_rand(rng, n, rate, ch=1):
    k = max(3, int(n / SR * rate) + 3)
    pts = rng.normal(0, 1, (k, ch))
    x = np.stack([np.interp(np.linspace(0, k - 1, n), np.arange(k), pts[:, c]) for c in range(ch)], 1)
    # smooth further (cubic-ish)
    w = max(1, min(int(SR / rate / 4), n // 4))
    if w > 1:
        ker = np.hanning(w * 2 + 1)
        ker /= ker.sum()
        x = np.stack([np.convolve(x[:, c], ker, mode="same")[:n] for c in range(ch)], 1)
    return x.astype(np.float32)


def fade(y, fin=0.01, fout=0.05):
    a, b = int(fin * SR), int(fout * SR)
    if a > 0:
        y[:a] *= np.linspace(0, 1, a)[:, None] ** 1.5
    if b > 0:
        y[-b:] *= np.linspace(1, 0, b)[:, None] ** 1.5
    return y


def env_ar(n, a, r, shape=1.0):
    t = np.arange(n) / SR
    e = np.minimum(1, t / max(a, 1e-4)) ** shape
    e *= np.exp(-np.maximum(0, t - a) / max(r, 1e-4))
    return e.astype(np.float32)


def panner(y, pan):
    th = (np.clip(pan, -1, 1) + 1) * np.pi / 4
    if y.ndim == 1:
        return np.stack([y * np.cos(th), y * np.sin(th)], 1) * np.sqrt(2)
    return np.stack([y[:, 0] * np.cos(th) * np.sqrt(2), y[:, 1] * np.sin(th) * np.sqrt(2)], 1)


def norm_rms(y, db):
    r = np.sqrt((y ** 2).mean()) + 1e-12
    return (y * (10 ** (db / 20) / r)).astype(np.float32)


# ---------------------------------------------------------------------------
# sound designs
# ---------------------------------------------------------------------------
def wind(rng, dur, strength=0.5, howl=0.3, hiss=0.2, cut=700, gust_rate=0.25):
    n = int(dur * SR)
    base = pink(rng, n)
    gust = 0.55 + 0.45 * np.tanh(1.5 * smooth_rand(rng, n, gust_rate, 1))
    gust2 = 0.7 + 0.3 * smooth_rand(rng, n, gust_rate * 3, 2)
    fc = cut * (0.6 + 0.8 * gust[:, 0])
    body = sweep_filter(base, "low", fc, q=0.7) * gust * gust2
    body = sos_filter(body, "high", 60)
    y = body * strength
    if howl > 0:
        hw = white(rng, n)
        fh = 500 + 700 * (0.5 + 0.5 * smooth_rand(rng, n, gust_rate * 0.8, 1)[:, 0]) + 300 * gust[:, 0]
        hw = sweep_filter(hw, "band", fh, q=9.0)
        y += hw * howl * gust ** 2 * 0.9
    if hiss > 0:
        hs = sos_filter(white(rng, n), "band", (3500, 11000))
        g3 = np.clip(gust - 0.45, 0, None) ** 1.5 * 2.0
        y += hs * hiss * g3 * 0.5
    return fade(y.astype(np.float32), 1.0, 1.0)


def crackle(rng, dur, rate=12.0, level=1.0, pops=0.3, breath=0.4, flutter=1.0):
    n = int(dur * SR)
    y = np.zeros((n, 2), np.float32)
    t = 0.0
    while True:
        t += rng.exponential(1.0 / rate)
        if t >= dur - 0.05:
            break
        i = int(t * SR)
        L = int(rng.uniform(0.0008, 0.006) * SR)
        amp = min(3.0, rng.pareto(2.5) * 0.4 + 0.1)
        g = rng.normal(0, 1, L) * np.exp(-np.arange(L) / (L / 4))
        fc = rng.uniform(1500, 7000)
        b, a = signal.butter(2, [fc * 0.6 / (SR / 2), min(0.95, fc * 1.6 / (SR / 2))], btype="band")
        g = signal.lfilter(b, a, g) * amp
        p = rng.uniform(-0.5, 0.5)
        y[i:i + L] += panner(g, p)[: n - i]
        if rng.uniform() < pops * 0.15:
            L2 = int(0.012 * SR)
            g2 = rng.normal(0, 1, L2) * np.exp(-np.arange(L2) / (L2 / 5))
            b, a = signal.butter(2, [250 / (SR / 2), 900 / (SR / 2)], btype="band")
            y[i:i + L2] += panner(signal.lfilter(b, a, g2) * amp * 1.5, p)[: n - i]
    y *= level
    if breath > 0:
        br = brown(rng, n)
        fl = 1 + 0.35 * flutter * smooth_rand(rng, n, 7.0, 1)
        br = sos_filter(br, "low", 380) * fl
        y += br * breath * 0.6
    return fade(y, 0.2, 0.3)


def whoosh(rng, dur, f0=300, f1=4000, peak=0.7, q=1.2, pan0=0.0, pan1=0.0):
    n = int(dur * SR)
    t = np.linspace(0, 1, n)
    fc = f0 * (f1 / f0) ** t
    y = sweep_filter(white(rng, n), "band", fc, q=q)
    e = np.where(t < peak, (t / peak) ** 2.2, np.exp(-(t - peak) / (1 - peak + 1e-3) * 3.5))
    y = y * e[:, None]
    pans = pan0 + (pan1 - pan0) * t
    th = (pans + 1) * np.pi / 4
    y = np.stack([y[:, 0] * np.cos(th), y[:, 1] * np.sin(th)], 1) * np.sqrt(2)
    return fade(y.astype(np.float32), 0.005, 0.03)


def roar(rng, dur, lead, size=1.0, sustain=0.5, bright=1.0):
    """Ignition: pre-whoosh (lead s) -> whoomp + roar -> sustained fire."""
    n = int((lead + dur) * SR)
    t = np.arange(n) / SR
    i0 = int(lead * SR)
    y = np.zeros((n, 2), np.float32)
    if lead > 0:
        w = whoosh(rng, lead + 0.05, f0=250, f1=2500 * bright, peak=0.95, q=0.9)
        y[: len(w)] += w * 0.6
    # whoomp: low thump + downward lowpass sweep on noise
    m = n - i0
    tt = t[:m]
    f = 42 * size ** -0.3 + 70 * np.exp(-tt / 0.05)
    thump = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-tt / (0.35 * size))
    y[i0:] += (thump * 0.9)[:, None]
    nz = brown(rng, m) * 0.5 + white(rng, m) * 0.15
    fc = 250 + 3500 * bright * np.exp(-tt / (0.25 * size))
    burst = sweep_filter(nz, "low", fc, q=0.8)
    env = (1 - np.exp(-tt / 0.012)) * (np.exp(-tt / (0.5 * size)) * (1 - sustain) + sustain)
    fl = 1 + 0.3 * smooth_rand(rng, m, 6.0, 1)
    y[i0:] += burst * (env[:, None] * fl) * 1.6
    # sustained fire crackle
    cr = crackle(rng, m / SR, rate=18 * size, level=0.6, breath=0.0)
    y[i0:] += cr[:m] * (0.3 + 0.7 * (1 - np.exp(-tt / 0.3)))[:, None] * sustain
    tail = int(min(0.6, dur * 0.4) * SR)
    y[-tail:] *= np.linspace(1, 0, tail)[:, None]
    return y.astype(np.float32)


def flint(rng, bright=1.0):
    """Steel on flint: sharp scrape + ringing metal + spark ticks (hit at 0)."""
    n = int(0.9 * SR)
    t = np.arange(n) / SR
    y = np.zeros((n, 2), np.float32)
    scr_len = int(rng.uniform(0.035, 0.06) * SR)
    s = white(rng, scr_len)
    s = sos_filter(s, "band", (2200, 14000))
    e = np.exp(-np.arange(scr_len) / (scr_len / 3.0))
    e[: int(0.0015 * SR)] *= np.linspace(0, 1, int(0.0015 * SR))
    y[:scr_len] += s * e[:, None] * 1.6
    ring = np.zeros(n)
    for fq, d, a in ((3150, 0.05, 0.4), (5270, 0.035, 0.3), (7930, 0.025, 0.2), (9800, 0.02, 0.15)):
        ring += a * np.sin(2 * np.pi * fq * rng.uniform(0.97, 1.03) * t) * np.exp(-t / d)
    y += (ring * 0.5)[:, None]
    # sparks: tiny ticks falling off over ~250 ms
    nt = int(rng.integers(6, 13))
    for k in range(nt):
        tk = abs(rng.normal(0.06, 0.07)) + 0.01
        i = int(tk * SR)
        L = int(0.0008 * SR)
        if i + L >= n:
            continue
        tick = rng.normal(0, 1, L) * np.exp(-np.arange(L) / (L / 3))
        tick = sos_filter(tick[:, None], "high", 5000)[:, 0]
        y[i:i + L] += panner(tick * rng.uniform(0.2, 0.7), rng.uniform(-0.6, 0.6))
    return fade(y * bright, 0.0, 0.1)


def creak(rng, dur, f_res=(180, 420), rate=(25, 90)):
    """Stick-slip creak: modulated click train through resonators."""
    n = int(dur * SR)
    y = np.zeros(n)
    t = 0.0
    rr = smooth_rand(rng, n, 2.0, 1)[:, 0]
    while t < dur:
        i = int(t * SR)
        r = rate[0] + (rate[1] - rate[0]) * (0.5 + 0.5 * np.tanh(rr[min(i, n - 1)]))
        y[i] += rng.uniform(0.5, 1.0)
        t += 1.0 / r * rng.uniform(0.85, 1.15)
    out = np.zeros(n)
    for f in (f_res[0], f_res[1], f_res[1] * 1.9):
        b, a = signal.butter(2, [f * 0.9 / (SR / 2), f * 1.1 / (SR / 2)], btype="band")
        out += signal.lfilter(b, a, y)
    env = np.sin(np.linspace(0, np.pi, n)) ** 0.7
    return (out * env).astype(np.float32)


def rustle(rng, dur):
    n = int(dur * SR)
    y = np.zeros((n, 2), np.float32)
    t = 0.0
    while t < dur:
        L = int(rng.uniform(0.02, 0.12) * SR)
        i = int(t * SR)
        if i + L > n:
            break
        g = white(rng, L) * np.hanning(L)[:, None] * rng.uniform(0.2, 1.0)
        g = sos_filter(g, "band", (rng.uniform(700, 1500), rng.uniform(3000, 6500)))
        y[i:i + L] += g
        t += rng.exponential(0.035)
    env = np.sin(np.linspace(0, np.pi, n)) ** 0.8
    return fade(y * env[:, None], 0.05, 0.1)


def whump(rng, size=1.0):
    n = int(1.6 * SR)
    t = np.arange(n) / SR
    f = 55 + 60 * np.exp(-t / 0.04)
    y = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / 0.25)
    nz = sos_filter(brown(rng, n), "low", 600) * np.exp(-t / 0.2)[:, None] * 0.6
    out = y[:, None] * 0.8 + nz
    return fade(out.astype(np.float32) * size, 0.004, 0.2)


# ---------------------------------------------------------------------------
# the event list
# ---------------------------------------------------------------------------
def events(rng):
    E = []

    def add(name, frame, audio, gain_db, lead=0, pan=None):
        if pan is not None:
            audio = panner(audio, pan) if audio.ndim == 1 else audio
        E.append(dict(name=name, frame=int(frame - lead), hit_frame=int(frame), audio=audio,
                      gain_db=float(gain_db)))

    s = lambda fr: fr / 24.0  # noqa: E731
    # --- ambience beds
    add("wind_hill_intro", 0, wind(rng, s(372), strength=0.55, howl=0.18, hiss=0.08, cut=650), -24)
    add("torch_close_intro", 0, crackle(rng, s(352), rate=10, level=0.7, breath=0.55), -27)
    add("ember_whoosh", 290, whoosh(rng, s(56), f0=200, f1=5500, peak=0.8, q=0.8, pan0=-0.2, pan1=0.2), -24)
    add("ignition_whoomp", 480, roar(rng, 2.2, lead=0.5, size=1.2, sustain=0.1, bright=1.4), -21, lead=12)
    storm = roar(rng, s(240), lead=0, size=2.0, sustain=0.9, bright=0.6)
    tt = np.linspace(0, 1, len(storm))
    storm *= (0.15 + 0.85 * tt ** 1.6)[:, None]
    storm[-int(0.012 * SR):] *= np.linspace(1, 0, int(0.012 * SR))[:, None]   # dead cut at 1040
    add("storm_roar", 800, storm, -20)
    add("summit_wind", 1200, wind(rng, s(252), strength=0.8, howl=0.6, hiss=0.45, cut=900,
                                  gust_rate=0.4), -19)
    for i, f in enumerate([1236, 1262, 1290]):
        add(f"flint_strike_{i + 1}", f, flint(rng, 1.0 + 0.15 * i), -15 - (i == 0) * 1.5,
            pan=-0.12)
    add("kindling_catch", 1318, _kindle(rng), -19, lead=12)
    add("beacon_roar", 1360, roar(rng, s(84), lead=0.5, size=1.5, sustain=0.55, bright=1.0), -16, lead=12)
    # montage ignitions (varied) + location ambiences
    sizes = [1.3, 1.0, 0.9, 1.1, 1.0, 1.2]
    for i, f in enumerate([1480, 1540, 1600, 1650, 1690, 1730]):
        add(f"montage_ignition_{i + 1}", f, roar(rng, 2.4, lead=0.4, size=sizes[i], sustain=0.35,
                                                 bright=0.8 + 0.1 * i), -19, lead=10)
    add("amb_far_peak_wind", 1440, wind(rng, s(80), strength=0.6, howl=0.35, hiss=0.25, cut=800), -25)
    add("amb_desert_wind", 1520, wind(rng, s(60), strength=0.4, howl=0.1, hiss=0.6, cut=500,
                                      gust_rate=0.5), -25)
    ice = wind(rng, s(60), strength=0.45, howl=0.55, hiss=0.3, cut=1100)
    ck = panner(creak(rng, 1.2, (140, 380), (18, 60)), 0.5)
    i0 = int(0.5 * SR)
    ice[i0:i0 + len(ck)] += ck[: len(ice) - i0] * 0.5
    add("amb_ice_wind_creak", 1580, ice, -25)
    add("amb_jungle_night", 1640, _jungle(rng, s(40)), -28)
    add("amb_city_hum", 1680, _city(rng, s(40)), -27)
    add("amb_sea_swell", 1720, _sea(rng, s(48)), -24)
    # accord: torches converging
    conv = np.zeros((int(s(56) * SR), 2), np.float32)
    for k, (st, p) in enumerate([(0, -0.9), (4, 0.8), (9, -0.5), (13, 0.55), (18, -0.2), (22, 0.25)]):
        w = whoosh(rng, s(40 - st), f0=180, f1=1800, peak=0.85, q=0.9, pan0=p, pan1=0.0)
        i0 = int(s(st) * SR)
        conv[i0:i0 + len(w)] += w[: len(conv) - i0] * 0.6
    r = roar(rng, 2.0, lead=0, size=1.1, sustain=0.4)
    i0 = int(s(40) * SR)
    conv[i0:] += r[: len(conv) - i0] * 0.8
    conv = fade(conv, 0.3, 0.4)
    add("torches_merge", 2060, conv, -21, lead=40)
    add("hearth_flare", 2240, whoosh(rng, s(18) + 0.3, f0=300, f1=9000, peak=0.93, q=0.7), -20, lead=18)
    # coda
    add("wind_hill_coda", 2460, wind(rng, s(348), strength=0.5, howl=0.15, hiss=0.08, cut=600), -25)
    add("torch_close_coda", 2480, crackle(rng, s(160), rate=9, level=0.7, breath=0.5), -28)
    add("torch_handover_rustle", 2580, rustle(rng, s(40)), -26)
    add("beacon_catch", 2640, roar(rng, 2.5, lead=0.17, size=0.9, sustain=0.35, bright=0.8), -21, lead=4)
    times = [2645, 2651, 2657, 2663, 2669, 2675, 2682, 2689, 2696, 2703, 2710, 2716]
    for i, f in enumerate(times):
        side = 1 if i % 2 == 0 else -1
        pan = side * min(0.9, 0.15 + 0.07 * i)
        add(f"answering_fire_{i + 1:02d}", f, panner(whump(rng, 1.0)[:, 0], pan),
            -30 - 0.6 * i)
    return E


def _kindle(rng):
    lead = int(12 * FR)
    n = lead + int(2.2 * SR)
    y = np.zeros((n, 2), np.float32)
    cr = crackle(rng, n / SR, rate=35, level=0.8, breath=0.0)
    ramp = np.clip((np.arange(n) - lead * 0.5) / (lead * 0.5), 0, 1) ** 2
    y += cr * ramp[:, None]
    w = whoosh(rng, 0.9, f0=120, f1=900, peak=0.35, q=0.6)
    w0 = lead - int(0.315 * SR)
    y[w0:w0 + len(w)] += w[: n - w0] * 0.9
    b = crackle(rng, 1.8, rate=14, level=0.2, breath=0.6)
    y[lead:lead + len(b)] += b
    return fade(y, 0.05, 0.3)


def _jungle(rng, dur):
    n = int(dur * SR)
    t = np.arange(n) / SR
    y = np.zeros((n, 2), np.float32)
    for k in range(5):
        f = rng.uniform(3800, 7200)
        pr = rng.uniform(18, 45)
        am = (0.5 + 0.5 * np.sign(np.sin(2 * np.pi * pr * t + rng.uniform(0, 6)))) * (0.6 + 0.4 * np.sin(2 * np.pi * rng.uniform(0.2, 0.6) * t))
        ch = np.sin(2 * np.pi * f * t) * am * 0.08
        y += panner(ch, rng.uniform(-0.8, 0.8))
    leaves = sos_filter(pink(rng, n), "band", (800, 5000)) * 0.15 * (0.6 + 0.4 * smooth_rand(rng, n, 0.8, 1))
    return fade(y + leaves, 0.4, 0.4)


def _city(rng, dur):
    n = int(dur * SR)
    t = np.arange(n) / SR
    hum = sum(a * np.sin(2 * np.pi * f * t) for f, a in ((60, 0.3), (120, 0.2), (180, 0.08), (240, 0.04)))
    traffic = sos_filter(brown(rng, n), "low", 450) * (0.6 + 0.4 * smooth_rand(rng, n, 0.3, 1))
    y = traffic * 0.8 + hum[:, None] * 0.25
    return fade(y.astype(np.float32), 0.4, 0.4)


def _sea(rng, dur):
    n = int(dur * SR)
    t = np.arange(n) / SR
    swell = 0.5 + 0.5 * np.sin(2 * np.pi * 0.16 * t - 1.0) ** 2
    wash = sos_filter(pink(rng, n), "low", 1200) * swell[:, None]
    hiss = sos_filter(white(rng, n), "band", (2000, 7000)) * (np.clip(swell - 0.6, 0, None) * 1.5)[:, None] * 0.3
    hull = panner(creak(rng, 1.4, (110, 300), (15, 45)), -0.3)
    y = wash + hiss
    i0 = int(0.6 * SR)
    y[i0:i0 + len(hull)] += hull[: n - i0] * 0.6
    return fade(y.astype(np.float32), 0.3, 0.3)


# ---------------------------------------------------------------------------
def build_and_render(total_n, hall_ir, jobs=1):
    import mix as MX
    rng = np.random.default_rng(2807)
    E = events(rng)
    os.makedirs(EV_DIR, exist_ok=True)
    for f in os.listdir(EV_DIR):
        if f.endswith(".wav"):
            os.remove(os.path.join(EV_DIR, f))
    stem = np.zeros((total_n, 2), np.float32)
    meta = []
    for e in E:
        a = e["audio"].astype(np.float32)
        a = a - a.mean(axis=0, keepdims=True)
        fl = int(0.004 * SR)
        a[:fl] *= np.linspace(0, 1, fl)[:, None]
        a[-fl:] *= np.linspace(1, 0, fl)[:, None]
        # normalise the clip to a reference peak, then place at gain_db
        a = a / (np.abs(a).max() + 1e-9) * 0.5
        g = 10 ** (e["gain_db"] / 20) * 2.0
        i0 = e["frame"] * FR
        seg = a * g
        n = min(len(seg), total_n - i0)
        if i0 < 0:
            seg = seg[-i0:]
            i0 = 0
            n = min(len(seg), total_n)
        stem[i0:i0 + n] += seg[:n]
        fn = f"{e['name']}.wav"
        sf.write(os.path.join(EV_DIR, fn), a, SR, subtype="PCM_24")
        meta.append(dict(name=e["name"], file=f"sfx_events/{fn}", frame=e["frame"],
                         hit_frame=e["hit_frame"], gain_db=round(20 * np.log10(g), 2)))
    # outdoor space: short diffuse slap/tail
    out_ir = MX.get_ir("outdoor", rt_mid=0.9, length=1.4, seed=11, predelay=0.03, er_gain=0.9,
                       bright=1.2)
    wet = MX.convolve_stereo(stem * 0.25, out_ir)
    hall = MX.convolve_stereo(stem * 0.08, hall_ir)
    y = stem + wet + hall
    json.dump(meta, open(os.path.join(OUT, "sfx_events.json"), "w"), indent=1)
    print(f"sfx: {len(meta)} events")
    return y.astype(np.float32), meta
