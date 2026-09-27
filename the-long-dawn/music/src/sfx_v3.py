"""THE LONG DAWN v3 - the effects (the floor of every score, red team 5.4), driven by the bar map.

A bar map may carry two effect lists (see music/v3/README in NOTES_v3.md):
  "ambience": [{"id", "fx", "t0", "t1", "gain_db", "params": {...}, "fade_in", "fade_out",
                "env": [[t_rel_s, db], ...]}]                       beds (wind, fire bed, hearth, sea)
  "sfx":      [{"id", "fx", "t", "gain_db", "params": {...}, "pan", "dist", "exempt"}]
                                                                      point events; `t` = the sync point
Every clip has its own seed (crc32 of its id), so editing the list never changes other clips.
Clip files + an events JSON are written for the editor (out/v3/sfx_<cut>/).

Designs (fx names): wind, spindrift, fire, roar, strike, strikes3, knock_stone, knock_lid, blow, hiss,
birdsong, whump, page, hearth, sea, rustle, creak, gust.
"""
import json
import os
import zlib

import numpy as np
import soundfile as sf
from scipy import signal

import sfx as S1
from timeline_v3 import SR

HERE = os.path.dirname(os.path.abspath(__file__))
MUSIC = os.path.dirname(HERE)


def rng_for(name):
    return np.random.default_rng(zlib.crc32(name.encode()) % (2 ** 31))


def _st(y):
    y = np.asarray(y, np.float32)
    return np.stack([y, y], 1) if y.ndim == 1 else y


def distance(y, dist, rng):
    """dist 0 (close) .. 1 (far): air absorption, less direct sound, a diffuse tail, narrower image"""
    if dist <= 0.01:
        return y
    fc = 9000 * (1 - dist) ** 1.6 + 700
    y = S1.sos_filter(y, "low", fc, order=2)
    n = int(1.6 * SR)
    tail = rng.normal(0, 1, (n, 2)) * np.exp(-np.arange(n) / (0.35 * SR))[:, None]
    tail = S1.sos_filter(tail.astype(np.float32), "low", fc * 0.8)
    tail /= np.sqrt((tail ** 2).sum(0, keepdims=True)) + 1e-9
    wet = np.stack([signal.oaconvolve(y[:, c], tail[:, c])[:len(y)] for c in range(2)], 1)
    out = y * (1.0 - 0.75 * dist) + wet * (0.35 + 0.9 * dist) * np.sqrt((y ** 2).mean() / ((wet ** 2).mean() + 1e-12))
    mid = out.mean(1, keepdims=True)
    return (mid + (out - mid) * (1 - 0.6 * dist)).astype(np.float32)


# ---------------------------------------------------------------------------
# designs: each returns (stereo float32, hit_index)
# ---------------------------------------------------------------------------
def d_wind(rng, dur, strength=0.5, howl=0.25, hiss=0.15, cut=650, gust_rate=0.2):
    return S1.wind(rng, dur, strength=strength, howl=howl, hiss=hiss, cut=cut, gust_rate=gust_rate), 0


def d_spindrift(rng, dur, rate=0.35):
    n = int(dur * SR)
    g = np.clip(S1.smooth_rand(rng, n, rate, 1)[:, 0], 0, None) ** 1.6
    hs = S1.sos_filter(S1.white(rng, n), "band", (3000, 12500))
    grain = 0.75 + 0.25 * np.tanh(3 * S1.smooth_rand(rng, n, 40.0, 2))
    return S1.fade((hs * grain * g[:, None]).astype(np.float32), 0.5, 0.5), 0


def d_gust(rng, dur=3.0):
    y, _ = d_wind(rng, dur, strength=0.8, howl=0.45, hiss=0.3, cut=900, gust_rate=0.5)
    e = np.sin(np.linspace(0, np.pi, len(y))) ** 1.5
    return (y * e[:, None]).astype(np.float32), 0


def d_fire(rng, dur, rate=10.0, level=0.8, pops=0.3, breath=0.5, flutter=1.0):
    return S1.crackle(rng, dur, rate=rate, level=level, pops=pops, breath=breath, flutter=flutter), 0


def d_roar(rng, dur=4.0, lead=0.25, size=1.0, sustain=0.45, bright=1.0):
    return S1.roar(rng, dur, lead=lead, size=size, sustain=sustain, bright=bright), int(lead * SR)


def d_strike(rng, bright=1.0):
    y = np.concatenate([S1.flint(rng, bright), np.zeros((int(0.3 * SR), 2), np.float32)])
    return y, 0


def d_strikes3(rng, spacing=(0.0, 1.25, 2.5), bright=1.0):
    n = int((spacing[-1] + 1.4) * SR)
    y = np.zeros((n, 2), np.float32)
    for k, s in enumerate(spacing):
        f = S1.flint(rng, bright * (0.92 + 0.08 * k))
        i = int(s * SR)
        y[i:i + len(f)] += f[:n - i]
    return y, 0


def _modes(rng, n, freqs, decays, amps):
    t = np.arange(n) / SR
    y = np.zeros(n)
    for f, d, a in zip(freqs, decays, amps):
        y += a * np.sin(2 * np.pi * f * t + rng.uniform(0, 6.28)) * np.exp(-t / d)
    return y


def d_knock_stone(rng, weight=1.0, scrape=0.18, settle=True):
    """a stone set on the cairn: a short grinding slide, the contact (click + stone modes + a low
    thud), sometimes a small settle knock"""
    pre = int(scrape * SR)
    n = pre + int(0.9 * SR)
    y = np.zeros(n)
    if pre > 0:
        s = S1.white(rng, pre)[:, 0]
        b, a = signal.butter(2, [700 / (SR / 2), 4200 / (SR / 2)], btype="band")
        s = signal.lfilter(b, a, s)
        rough = np.abs(S1.smooth_rand(rng, pre, 90.0, 1)[:, 0]) ** 1.5
        s *= rough * np.linspace(0.2, 1.0, pre) ** 1.5 * 0.35
        y[:pre] += s
    m = n - pre
    t = np.arange(m) / SR
    click = rng.normal(0, 1, m) * np.exp(-t / 0.0012)
    b, a = signal.butter(2, 1500 / (SR / 2), btype="high")
    click = signal.lfilter(b, a, click)
    fr = sorted(rng.uniform(650, 3400, 5))
    body = _modes(rng, m, fr, rng.uniform(0.012, 0.035, 5), rng.uniform(0.3, 1.0, 5))
    thud = np.sin(2 * np.pi * rng.uniform(95, 150) * t) * np.exp(-t / 0.045) * 1.2 * weight
    lo = rng.normal(0, 1, m) * np.exp(-t / 0.05)
    b, a = signal.butter(2, 420 / (SR / 2))
    lo = signal.lfilter(b, a, lo) * 1.5 * weight
    y[pre:] += click * 0.9 + body * 0.35 + thud + lo
    if settle:
        k = pre + int(rng.uniform(0.07, 0.13) * SR)
        mm = n - k
        t2 = np.arange(mm) / SR
        body2 = _modes(rng, mm, sorted(rng.uniform(900, 3000, 3)), rng.uniform(0.008, 0.02, 3), [0.3, 0.25, 0.2])
        y[k:] += body2 * 0.35 + np.sin(2 * np.pi * 120 * t2) * np.exp(-t2 / 0.03) * 0.3
    return S1.fade(_st(y), 0.002, 0.1), pre


def d_feed(rng, dur=2.6):
    """she feeds the fire: a branch set down on the burning stack (a dull wooden knock, a little rustle of
    brushwood), then the flames take it (a short flare of crackle and a breath of rising flame)"""
    n = int(dur * SR)
    t = np.arange(n) / SR
    y = np.zeros((n, 2), np.float32)
    pre = int(0.12 * SR)
    ru = S1.rustle(rng, 0.25) * 0.35
    y[:len(ru)] += ru
    m = n - pre
    tt = t[:m]
    wood = _modes(rng, m, sorted(rng.uniform(180, 900, 4)), rng.uniform(0.012, 0.03, 4), rng.uniform(0.4, 1.0, 4))
    thud = np.sin(2 * np.pi * rng.uniform(80, 120) * tt) * np.exp(-tt / 0.05)
    y[pre:] += _st(wood * 0.5 + thud * 0.8)
    fl = S1.crackle(rng, dur - 0.3, rate=30, level=1.0, pops=0.4, breath=0.6)
    k = int(0.3 * SR)
    env = np.clip((t[k:] - 0.3) / 0.35, 0, 1) * np.exp(-np.clip(t[k:] - 0.8, 0, None) / 0.7)
    y[k:k + len(fl)] += fl[: n - k] * env[:len(fl), None][: n - k] * 0.8
    return S1.fade(y, 0.005, 0.2), pre


def d_knock_lid(rng):
    """the clay pot's lid: a hollow ceramic tock with a small scrape"""
    pre = int(0.08 * SR)
    n = pre + int(0.6 * SR)
    y = np.zeros(n)
    s = S1.white(rng, pre)[:, 0]
    b, a = signal.butter(2, [1500 / (SR / 2), 6000 / (SR / 2)], btype="band")
    y[:pre] = signal.lfilter(b, a, s) * np.linspace(0, 1, pre) ** 2 * 0.12
    m = n - pre
    t = np.arange(m) / SR
    body = _modes(rng, m, [rng.uniform(1050, 1250), rng.uniform(1850, 2100), rng.uniform(2800, 3100),
                           rng.uniform(4000, 4500)], [0.05, 0.035, 0.025, 0.015], [1.0, 0.6, 0.35, 0.2])
    hollow = np.sin(2 * np.pi * rng.uniform(230, 290) * t) * np.exp(-t / 0.03) * 0.7
    click = rng.normal(0, 1, m) * np.exp(-t / 0.0008) * 0.6
    y[pre:] += body * 0.5 + hollow + click
    return S1.fade(_st(y), 0.002, 0.08), pre


def d_blow(rng, dur=2.2, strength=1.0, glow=0.5):
    """her breath on the ember: a long, soft exhale, and the ember's faint answer (a few ticks)"""
    n = int(dur * SR)
    t = np.arange(n) / SR
    br = S1.pink(rng, n)
    br = S1.sos_filter(br, "band", (380, 5200))                  # an unvoiced "hhhh", not a rumble
    form = S1.sos_filter(br, "band", (1100, 2400)) * 1.3        # the lips' narrow opening
    e = np.clip(t / 0.18, 0, 1) ** 1.5 * np.clip((dur - t) / 0.45, 0, 1) ** 1.2
    e *= 1 + 0.12 * S1.smooth_rand(rng, n, 5.0, 1)[:, 0]
    y = (br * 0.5 + form) * e[:, None] * strength
    if glow > 0:
        cr = S1.crackle(rng, dur, rate=16, level=0.25 * glow, pops=0.05, breath=0.0)
        y += cr[:n] * np.sin(np.linspace(0, np.pi, n))[:, None]
    return S1.fade(y.astype(np.float32), 0.02, 0.1), 0


def d_hiss(rng, dur=1.8):
    """the ember greying: a thin hiss and the last ticks, thinning out"""
    n = int(dur * SR)
    t = np.arange(n) / SR
    hs = S1.sos_filter(S1.white(rng, n), "band", (2500, 9000)) * np.exp(-t / 0.5)[:, None] * 0.18
    cr = S1.crackle(rng, dur, rate=9, level=0.3, pops=0.0, breath=0.0) * np.exp(-t / 0.6)[:, None]
    return S1.fade((hs + cr[:n]).astype(np.float32), 0.05, 0.2), 0


def _bird_element(rng, kind, f0):
    if kind == "whistle":
        d = rng.uniform(0.08, 0.22)
        n = int(d * SR)
        u = np.linspace(0, 1, n)
        f = f0 * (1 + rng.uniform(-0.12, 0.12) * u + 0.01 * np.sin(2 * np.pi * rng.uniform(8, 14) * u * d))
        env = np.sin(np.pi * u) ** 0.6
    elif kind == "sweep":
        d = rng.uniform(0.04, 0.09)
        n = int(d * SR)
        u = np.linspace(0, 1, n)
        f1 = f0 * rng.choice([0.55, 1.6])
        f = f0 * (f1 / f0) ** (u ** 0.8)
        env = np.sin(np.pi * u) ** 0.5
    elif kind == "trill":
        k = int(rng.integers(5, 10))
        rate = rng.uniform(16, 24)
        d = k / rate
        n = int(d * SR)
        u = np.linspace(0, 1, n)
        ph = (u * k) % 1.0
        f = f0 * (0.85 + 0.3 * ph)
        env = np.sin(np.pi * ph) ** 2 * (0.6 + 0.4 * np.sin(np.pi * u))
    else:  # warble
        d = rng.uniform(0.12, 0.25)
        n = int(d * SR)
        u = np.linspace(0, 1, n)
        f = f0 * (1 + 0.09 * np.sin(2 * np.pi * rng.uniform(28, 45) * u * d))
        env = np.sin(np.pi * u) ** 0.7
    ph = 2 * np.pi * np.cumsum(f) / SR
    y = np.sin(ph) + 0.08 * np.sin(2 * ph) + 0.02 * np.sin(3 * ph)
    y *= env * (1 + 0.1 * rng.normal(0, 1, n) * 0.2)
    return y


def _glide(rng, n, f_pts):
    """a smooth pitch contour through a few control frequencies"""
    xs = np.linspace(0, 1, len(f_pts))
    u = np.linspace(0, 1, n)
    from scipy.interpolate import PchipInterpolator
    return PchipInterpolator(xs, np.log(f_pts))(u)


def d_blackbird(rng, phrases=2, gap=(2.2, 3.6), dist=0.5):
    """the first bird: a blackbird's mellow, fluted phrase (4-7 whistled notes, 1.3-3.2 kHz, gliding),
    ending in a soft high twitter; far and quiet (hit = the first note)"""
    out = []
    t = 0.0
    for p in range(phrases):
        k = int(rng.integers(4, 8))
        base = rng.uniform(1500, 2300)
        el = []
        for i in range(k):
            d = rng.uniform(0.09, 0.32)
            n = int(d * SR)
            f0 = base * 2 ** (rng.normal(0, 0.28))
            pts = [f0 * rng.uniform(0.92, 1.0), f0 * rng.uniform(1.0, 1.12), f0 * rng.uniform(0.85, 1.05)]
            ph = 2 * np.pi * np.cumsum(np.exp(_glide(rng, n, pts)) * (1 + 0.004 * np.sin(np.arange(n) / SR * 2 * np.pi * 28))) / SR
            u = np.linspace(0, 1, n)
            env = np.clip(u / 0.08, 0, 1) ** 1.2 * np.clip((1 - u) / 0.35, 0, 1) ** 1.5
            y = (np.sin(ph) + 0.12 * np.sin(2 * ph + 0.5) + 0.03 * np.sin(3 * ph)) * env * rng.uniform(0.6, 1.0)
            el += [y, np.zeros(int(rng.uniform(0.03, 0.11) * SR))]
        for i in range(int(rng.integers(3, 7))):            # the twitter
            d = rng.uniform(0.02, 0.05)
            n = int(d * SR)
            f0 = rng.uniform(4500, 7000)
            ph = 2 * np.pi * np.cumsum(np.exp(_glide(rng, n, [f0 * 1.15, f0 * 0.8]))) / SR
            u = np.linspace(0, 1, n)
            el += [np.sin(ph) * np.sin(np.pi * u) * rng.uniform(0.15, 0.35), np.zeros(int(rng.uniform(0.015, 0.04) * SR))]
        ph_y = np.concatenate(el)
        out.append((t, ph_y))
        t += len(ph_y) / SR + rng.uniform(*gap)
    n = int((t + 0.6) * SR)
    y = np.zeros(n)
    for t0, ph_y in out:
        i = int(t0 * SR)
        y[i:i + len(ph_y)] += ph_y[: n - i]
    br = S1.sos_filter(S1.white(rng, n), "band", (1200, 6000))[:, 0] * 0.004
    y = y * 0.5 + br * (np.abs(y) > 1e-3)
    y = S1.panner(y, rng.uniform(-0.45, 0.45)).astype(np.float32)
    return distance(y, dist, rng), 0


def d_birdsong(rng, phrases=2, gap=(1.6, 3.2), f_lo=2600, f_hi=6800, dist=0.55):
    """a small songbird at first light: short, varied phrases of whistles, sweeps, trills and warbles,
    irregular, far and quiet (hit = the first element)"""
    parts = []
    t = 0.0
    for p in range(phrases):
        k = int(rng.integers(5, 11))
        els = []
        for i in range(k):
            kind = rng.choice(["whistle", "sweep", "trill", "warble"], p=[0.38, 0.3, 0.17, 0.15])
            f0 = np.exp(rng.uniform(np.log(f_lo), np.log(f_hi)))
            els.append(_bird_element(rng, kind, f0) * rng.uniform(0.45, 1.0))
            els.append(np.zeros(int(rng.uniform(0.02, 0.09) * SR)))
        ph = np.concatenate(els)
        parts.append((t, ph))
        t += len(ph) / SR + rng.uniform(*gap)
    n = int((t + 0.5) * SR)
    y = np.zeros(n)
    for t0, ph in parts:
        i = int(t0 * SR)
        y[i:i + len(ph)] += ph[: n - i]
    y = S1.sos_filter(_st(y * 0.5), "high", 1800)
    y = S1.panner(y[:, 0], rng.uniform(-0.5, 0.5)).astype(np.float32)
    return distance(y, dist, rng), 0


def d_whump(rng, size=1.0, dist=0.6):
    return distance(S1.whump(rng, size), dist, rng), 0


def d_page(rng, dur=0.9):
    """a heavy page turning: a rustle lifting, the air of the leaf, a soft landing"""
    n = int((dur + 0.4) * SR)
    y = np.zeros((n, 2), np.float32)
    ru = S1.rustle(rng, dur * 0.7) * 0.8
    y[:len(ru)] += ru
    w = S1.whoosh(rng, dur * 0.5, f0=250, f1=1200, peak=0.6, q=0.8, pan0=-0.3, pan1=0.3) * 0.5
    i = int(dur * 0.35 * SR)
    y[i:i + len(w)] += w[: n - i]
    k = int(dur * SR)
    m = n - k
    t = np.arange(m) / SR
    land = (np.sin(2 * np.pi * 140 * t) * np.exp(-t / 0.03) * 0.5 + S1.white(rng, m)[:, 0] *
            np.exp(-t / 0.004) * 0.3)
    y[k:] += _st(land)
    return S1.fade(y, 0.01, 0.1), k


def d_hearth(rng, dur):
    y = S1.crackle(rng, dur, rate=8, level=0.7, pops=0.35, breath=0.7, flutter=1.2)
    return y, 0


def d_sea(rng, dur):
    return S1._sea(rng, dur), 0


def d_rustle(rng, dur=0.8):
    return S1.rustle(rng, dur), 0


def d_creak(rng, dur=1.0, lo=180, hi=420):
    return _st(S1.creak(rng, dur, (lo, hi))), 0


DESIGNS = dict(wind=d_wind, spindrift=d_spindrift, gust=d_gust, fire=d_fire, roar=d_roar, strike=d_strike,
               strikes3=d_strikes3, knock_stone=d_knock_stone, feed=d_feed, knock_lid=d_knock_lid, blow=d_blow, hiss=d_hiss,
               birdsong=d_birdsong, blackbird=d_blackbird, whump=d_whump, page=d_page, hearth=d_hearth, sea=d_sea, rustle=d_rustle,
               creak=d_creak)
BEDS = {"wind", "spindrift", "fire", "hearth", "sea"}


# ---------------------------------------------------------------------------
# the stem
# ---------------------------------------------------------------------------
def _norm_peak(a):
    a = np.asarray(a, np.float32)
    a = a - a.mean(axis=0, keepdims=True)
    fl = int(0.004 * SR)
    a[:fl] *= np.linspace(0, 1, fl)[:, None]
    a[-fl:] *= np.linspace(1, 0, fl)[:, None]
    return a / (np.abs(a).max() + 1e-9) * 0.5                    # peak -6 dBFS


def _norm_rms(a, ref_db=-20.0):
    r = np.sqrt((a.astype(np.float64) ** 2).mean()) + 1e-12
    return (a * (10 ** (ref_db / 20) / r)).astype(np.float32)


def build_stem(bm, total_n, irs, breaths_beats, out_dir=None, only=None):
    """-> (stem [total_n, 2], meta list).  Beds are RMS-normalised to -20 dBFS then gain_db;
    point events are peak-normalised to -6 dBFS then gain_db (placed so their hit lands on t)."""
    import render_v3 as R
    wins = R.breath_windows(breaths_beats)
    stem = np.zeros((total_n, 2), np.float32)
    meta = []
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    for bed in bm.d.get("ambience", []):
        if only and bed["id"] not in only:
            continue
        rng = rng_for(bed["id"])
        fi, fo = bed.get("fade_in", 1.5), bed.get("fade_out", 1.5)
        t0, t1 = bed["t0"], bed["t1"]
        dur = t1 - t0
        a, _ = DESIGNS[bed["fx"]](rng, dur, **bed.get("params", {}))
        a = _norm_rms(_st(a)[: int(dur * SR)])
        n = len(a)
        tt = np.arange(n) / SR
        env = np.clip(tt / max(fi, 1e-3), 0, 1) * np.clip((dur - tt) / max(fo, 1e-3), 0, 1)
        env = np.sin(env * np.pi / 2) ** 2
        if bed.get("env"):
            pts = np.array(bed["env"], float)
            env = env * 10 ** (np.interp(tt, pts[:, 0], pts[:, 1]) / 20)
        a = a * env[:, None].astype(np.float32) * np.float32(10 ** (bed.get("gain_db", 0) / 20))
        if bed.get("pan"):
            a = S1.panner(a.mean(1), bed["pan"]).astype(np.float32)
        i0 = int(round(t0 * SR))
        e = min(total_n, i0 + n)
        a = a[: e - i0]
        if wins and not bed.get("exempt"):
            a = a * R.breath_env(len(a), wins, offset=i0)[:, None]
        stem[i0:e] += a
        meta.append(dict(id=bed["id"], fx=bed["fx"], t0=t0, t1=t1, gain_db=bed.get("gain_db", 0), kind="bed"))
    for ev in bm.d.get("sfx", []):
        if only and ev["id"] not in only:
            continue
        rng = rng_for(ev["id"])
        a, hit = DESIGNS[ev["fx"]](rng, **ev.get("params", {}))
        a = _st(a)
        if ev.get("pan") is not None:
            a = S1.panner(a.mean(1), ev["pan"]).astype(np.float32)
        if ev.get("dist"):
            a = distance(a, ev["dist"], rng)
        a = _norm_peak(a)
        if out_dir:
            sf.write(os.path.join(out_dir, f"{ev['id']}.wav"), a, SR, subtype="PCM_24")
        a = a * np.float32(10 ** (ev.get("gain_db", 0) / 20))
        i0 = int(round(ev["t"] * SR)) - hit
        if i0 < 0:
            a, i0 = a[-i0:], 0
        e = min(total_n, i0 + len(a))
        a = a[: e - i0]
        if wins and not ev.get("exempt"):
            a = a * R.breath_env(len(a), wins, offset=i0)[:, None]
        stem[i0:e] += a
        meta.append(dict(id=ev["id"], fx=ev["fx"], t=ev["t"], clip_start_s=round(i0 / SR, 4),
                         hit_offset_s=round(hit / SR, 4), gain_db=ev.get("gain_db", 0),
                         file=f"{ev['id']}.wav" if out_dir else None, exempt=bool(ev.get("exempt"))))
    # the outdoor space (a short diffuse slap off the mountains) + a little of the hall
    import mix as MX
    out_ir = MX.get_ir("outdoor", rt_mid=0.9, length=1.4, seed=11, predelay=0.03, er_gain=0.9, bright=1.2)
    wet = R.convolve_with_breaths(stem * np.float32(0.22), out_ir, wins, total_n)
    hall = R.convolve_with_breaths(stem * np.float32(0.06), irs, wins, total_n)
    y = (stem + wet + hall).astype(np.float32)
    if out_dir:
        json.dump(meta, open(os.path.join(out_dir, "events.json"), "w"), indent=1)
    return y, meta
