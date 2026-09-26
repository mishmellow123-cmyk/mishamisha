"""THE LONG DAWN v2 - sound effects on the v2 timeline (BIBLE_V2 section 3).

Two steps, so the editor can nudge anything without touching code:

  1. make_events(cut): synthesise every clip -> out/v2/sfx_events/<name>.wav (peak -6 dBFS)
     and write out/v2/sfx_events_<cut>.json:
       [{"name", "file", "frame" (clip start, v2 frames), "hit_frame" (its sync point, a whole
         number of frames into the clip), "gain_db" (the gain applied to the clip file in
         sfx_<cut>.wav, before the master - v1's convention: spec level + 6.02 dB for the -6 dBFS
         clip peak), "breath_exempt", "gain_db_master" (after the master's static gain)}]
  2. assemble(cut): read that JSON, place every clip on its frame, apply the score's breaths,
     add the outdoor space + a little of the hall -> the SFX stem.

    python sfx_v2.py --cut AB             # clips + JSON + stem check (no master)
    python sfx_v2.py --cut C --assemble   # rebuild the stem from an edited JSON only

Every event before v2 frame 1520 is the v1 design at the v1 frame; every event after the
Beacon Run is the v1 design moved +160 frames; THE BEACON RUN (1520-1679) and cut C's map
crackle (1920-2087) are new.  Each clip has its own seed (crc32 of its name), so clips do
not change when the list is edited.
"""
import argparse
import json
import os
import sys
import zlib

import numpy as np
import soundfile as sf
from scipy import signal

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import sfx as S1  # noqa: E402  (v1 sound designs)
from timeline_v2 import SR, FR, TOTAL_N  # noqa: E402

MUSIC = os.path.dirname(HERE)
OUT = os.path.join(MUSIC, "out", "v2")
EV_DIR = os.path.join(OUT, "sfx_events")


def s(fr):
    return fr / 24.0


def rng_for(name):
    return np.random.default_rng(zlib.crc32(name.encode()) % (2 ** 31))


# ---------------------------------------------------------------------------
# new designs
# ---------------------------------------------------------------------------
def distance(y, dist, rng):
    """dist 0 (close) .. 1 (far): air absorption (lowpass), less direct sound, more diffuse
    tail (a small in-clip reverb), narrower image."""
    if dist <= 0.01:
        return y
    fc = 9000 * (1 - dist) ** 1.6 + 700
    y = S1.sos_filter(y, "low", fc, order=2)
    n = int(1.6 * SR)
    tail = rng.normal(0, 1, (n, 2)) * np.exp(-np.arange(n) / (0.35 * SR))[:, None]
    tail = S1.sos_filter(tail.astype(np.float32), "low", fc * 0.8)
    tail /= np.sqrt((tail ** 2).sum(0, keepdims=True)) + 1e-9
    wet = np.stack([signal.oaconvolve(y[:, c], tail[:, c])[:len(y)] for c in range(2)], 1)
    dry_g = 1.0 - 0.75 * dist
    out = y * dry_g + wet * (0.35 + 0.9 * dist) * np.sqrt((y ** 2).mean() / ((wet ** 2).mean() + 1e-12))
    mid = out.mean(1, keepdims=True)
    return (mid + (out - mid) * (1 - 0.6 * dist)).astype(np.float32)


def flight_wind(rng, dur):
    """THE BEACON RUN: the rush of flight - fast broadband air, a low buffeting body,
    whistling edges, rising through the run, and short pass-bys on the beats."""
    n = int(dur * SR)
    t = np.arange(n) / SR
    base = S1.pink(rng, n)
    rise = np.clip(t / dur, 0, 1)
    fc = 900 + 3200 * rise ** 1.4 + 700 * S1.smooth_rand(rng, n, 1.5, 1)[:, 0]
    body = S1.sweep_filter(base, "low", fc, q=0.6)
    buffet = 0.75 + 0.25 * np.tanh(2 * S1.smooth_rand(rng, n, 7.0, 2))
    y = body * buffet
    hs = S1.sos_filter(S1.white(rng, n), "band", (2500, 12000))
    y += hs * (0.25 + 0.35 * rise)[:, None] * 0.6
    edge = S1.sweep_filter(S1.white(rng, n), "band", 1400 + 1800 * rise + 300 * np.sin(2 * np.pi * 0.7 * t),
                           q=8.0)
    y += edge * 0.35
    env = (0.55 + 0.45 * rise ** 0.8) * (1 - np.exp(-t / 0.08))
    y = y * env[:, None]
    return S1.fade(y.astype(np.float32), 0.02, 0.012)


def beacon_whoomp(rng, dist=0.0, size=1.0, bright=1.0, lead=3 / 24):
    y = S1.roar(rng, 2.6, lead=lead, size=size, sustain=0.28, bright=bright)
    return distance(y, dist, rng)


def parchment(rng, dur):
    """Cut C's map: old paper under the hand + a small fire tracing lines across it."""
    n = int(dur * SR)
    y = np.zeros((n, 2), np.float32)
    t = 0.0
    while t < dur - 0.2:
        L = int(rng.uniform(0.05, 0.35) * SR)
        i = int(t * SR)
        if i + L >= n:
            break
        g = S1.white(rng, L) * np.hanning(L)[:, None] * rng.uniform(0.15, 0.6)
        g = S1.sos_filter(g, "band", (rng.uniform(1500, 2800), rng.uniform(6000, 11000)))
        # crinkle: sparse ticks inside the rustle
        for _ in range(int(rng.integers(2, 9))):
            k = int(rng.integers(0, max(1, L - 200)))
            g[k:k + 60] += S1.white(rng, 60) * rng.uniform(0.2, 0.7) * np.exp(-np.arange(60) / 12)[:, None]
        y[i:i + L] += S1.panner(g, rng.uniform(-0.5, 0.5))[:L]
        t += rng.exponential(0.45)
    cr = S1.crackle(rng, dur, rate=7, level=0.45, pops=0.1, breath=0.25, flutter=0.6)
    y = y * 0.8 + cr[:n] * 0.7
    env = np.clip(np.arange(n) / (0.8 * SR), 0, 1) * np.clip((n - np.arange(n)) / (0.8 * SR), 0, 1)
    return (y * env[:, None]).astype(np.float32)


def flint_echo(k):
    """The wordless reveal: as the torch changes hands, the SAME strike as flint_strike_<k+1> (same
    seed, same design), remembered - farther, darker, with a little air around it."""
    y = S1.panner(S1.flint(rng_for(f"flint_strike_{k + 1}"), 1.0 + 0.15 * k), -0.12).astype(np.float32)
    y = np.concatenate([y, np.zeros((int(0.8 * SR), 2), np.float32)])
    return distance(y, 0.45, rng_for(f"flint_echo_{k + 1}"))


def torch_field(rng, dur):
    """the plain of torches converging on the stone ring: many distant crackles, swelling."""
    n = int(dur * SR)
    y = np.zeros((n, 2), np.float32)
    for k in range(6):
        c = S1.crackle(rng, dur, rate=rng.uniform(5, 10), level=0.5, pops=0.2, breath=0.3)
        y += S1.panner(c.mean(1), rng.uniform(-0.8, 0.8)) * rng.uniform(0.5, 1.0)
    y = S1.sos_filter(y, "low", 4500)
    env = np.linspace(0.3, 1.0, n) ** 1.5
    return S1.fade((y * env[:, None]).astype(np.float32), 0.6, 0.3)


def ring_sweep(rng, dur):
    """the carved outer ring ignites in a sweep: a travelling fire-hiss from left to right."""
    n = int(dur * SR)
    t = np.linspace(0, 1, n)
    w = S1.whoosh(rng, dur, f0=500, f1=3500, peak=0.55, q=0.8, pan0=-0.85, pan1=0.85)
    cr = S1.crackle(rng, dur, rate=40, level=0.6, pops=0.2, breath=0.0)
    pans = -0.85 + 1.7 * t
    th = (pans + 1) * np.pi / 4
    crm = cr.mean(1)
    crp = np.stack([crm * np.cos(th), crm * np.sin(th)], 1) * np.sqrt(2)
    env = np.sin(np.pi * np.clip(t, 0, 1)) ** 0.6
    return S1.fade((w * 0.8 + crp * env[:, None] * 0.7).astype(np.float32), 0.05, 0.2)


def hearth_flare(rng, rise_s, tail_s=0.05):
    """rising whoosh that peaks exactly at the end of `rise_s` (the start of the breath)."""
    n = int((rise_s + tail_s) * SR)
    t = np.arange(n) / SR
    x = S1.white(rng, n)
    fc = 300 * (9000 / 300) ** np.clip(t / rise_s, 0, 1)
    y = S1.sweep_filter(x, "band", fc, q=0.7)
    env = np.where(t < rise_s, (t / rise_s) ** 2.4, np.exp(-(t - rise_s) / 0.02))
    roarb = S1.sos_filter(S1.brown(rng, n), "low", 500) * (np.clip(t / rise_s, 0, 1) ** 3)[:, None]
    y = y * env[:, None] + roarb * 0.6
    return S1.fade(y.astype(np.float32), 0.01, 0.004)


def dawn_air(rng, dur):
    """cut C's dawn over the eastern ranges: a high, clean mountain air (very soft)."""
    return S1.wind(rng, dur, strength=0.35, howl=0.08, hiss=0.25, cut=1400, gust_rate=0.12)


# ---------------------------------------------------------------------------
# the event list
# ---------------------------------------------------------------------------
def event_specs(cut):
    """[(name, hit_frame, lead_frames, gain_db, exempt, maker)] for a cut ('AB' or 'C')."""
    E = []

    def add(name, hit, maker, gain_db, lead=0, exempt=False, cuts=("AB", "C")):
        if cut in cuts:
            E.append(dict(name=name, hit=int(hit), lead=int(lead), gain_db=float(gain_db), exempt=exempt,
                          maker=maker))

    W, CR, WH, RO = S1.wind, S1.crackle, S1.whoosh, S1.roar
    # --- the v1 first half (frames unchanged)
    add("wind_hill_intro", 0, lambda r: W(r, s(372), strength=0.55, howl=0.18, hiss=0.08, cut=650), -24)
    add("torch_close_intro", 0, lambda r: CR(r, s(352), rate=10, level=0.7, breath=0.55), -27)
    add("ember_whoosh", 290, lambda r: WH(r, s(56), f0=200, f1=5500, peak=0.8, q=0.8, pan0=-0.2, pan1=0.2), -24)
    add("ignition_whoomp", 480, lambda r: RO(r, 2.2, lead=0.5, size=1.2, sustain=0.1, bright=1.4), -21, lead=12)

    def storm(r):
        y = RO(r, s(237), lead=0, size=2.0, sustain=0.9, bright=0.6)
        tt = np.linspace(0, 1, len(y))
        y *= (0.15 + 0.85 * tt ** 1.6)[:, None]
        y[-int(0.012 * SR):] *= np.linspace(1, 0, int(0.012 * SR))[:, None]     # dead cut at 1037
        return y
    add("storm_roar", 800, storm, -20)
    # --- FIRST BEACON (1200-1439) and the shepherd (1440-1519): v1 frames
    add("summit_wind", 1200, lambda r: W(r, s(252), strength=0.8, howl=0.6, hiss=0.45, cut=900, gust_rate=0.4),
        -19)
    for i, f in enumerate([1236, 1262, 1290]):
        add(f"flint_strike_{i + 1}", f, (lambda k: lambda r: S1.panner(S1.flint(r, 1.0 + 0.15 * k), -0.12))(i),
            -15 - (i == 0) * 1.5)
    add("kindling_catch", 1318, lambda r: S1._kindle(r), -19, lead=12)
    add("beacon_roar", 1360, lambda r: RO(r, s(84), lead=0.5, size=1.5, sustain=0.55, bright=1.0), -8, lead=12)
    add("amb_far_peak_wind", 1440, lambda r: W(r, s(80), strength=0.6, howl=0.35, hiss=0.25, cut=800), -25)
    add("shepherd_ignition", 1480, lambda r: RO(r, 2.4, lead=10 / 24, size=1.3, sustain=0.35, bright=0.8), -5,
        lead=10)
    # --- THE BEACON RUN (new, 1520-1679)
    add("run_flight_wind", 1520, lambda r: flight_wind(r, s(160)), -15)
    run = [(1540, 0.05, -0.55, 1.25), (1560, 0.15, 0.5, 1.15), (1580, 0.25, -0.35, 1.1), (1600, 0.35, 0.4, 1.05),
           (1620, 0.45, -0.25, 1.0), (1640, 0.55, 0.3, 0.95), (1660, 0.65, -0.1, 0.95)]
    for i, (f, dist, pan, size) in enumerate(run):
        add(f"run_beacon_{i + 1}", f,
            (lambda d, p, sz: lambda r: S1.panner(beacon_whoomp(r, d, sz, 1.1 - 0.4 * d), p))(dist, pan, size),
            -1.5 - 6.0 * dist, lead=3)
    for i, f in enumerate([1550, 1570, 1590, 1610, 1630, 1650, 1670]):      # small, far ones between
        pan = [0.8, -0.75, 0.65, -0.8, 0.7, -0.6, 0.55][i]
        add(f"run_far_{i + 1}", f, (lambda p: lambda r: S1.panner(beacon_whoomp(r, 0.85, 0.7, 0.6), p))(pan),
            -19, lead=3)
    # --- the montage (v1 +160)
    sizes = [1.0, 0.9, 1.1, 1.0, 1.2]
    for i, f in enumerate([1700, 1760, 1810, 1850, 1890]):
        g = [-7, -5, -6, -5, -6][i]           # (v1's editor had to lift masked ignitions; v2 sets them here)
        add(f"montage_ignition_{i + 2}", f,
            (lambda k: lambda r: RO(r, 2.4, lead=10 / 24, size=sizes[k], sustain=0.35, bright=0.9 + 0.1 * k))(i),
            g, lead=10)
    add("amb_desert_wind", 1680, lambda r: W(r, s(60), strength=0.4, howl=0.1, hiss=0.6, cut=500, gust_rate=0.5),
        -25)

    def ice(r):
        y = W(r, s(60), strength=0.45, howl=0.55, hiss=0.3, cut=1100)
        ck = S1.panner(S1.creak(r, 1.2, (140, 380), (18, 60)), 0.5)
        i0 = int(0.5 * SR)
        y[i0:i0 + len(ck)] += ck[: len(y) - i0] * 0.5
        return y
    add("amb_ice_wind_creak", 1740, ice, -25)
    add("amb_jungle_night", 1800, lambda r: S1._jungle(r, s(40)), -28)
    add("amb_city_hum", 1840, lambda r: S1._city(r, s(40)), -27)
    add("amb_sea_swell", 1880, lambda r: S1._sea(r, s(48)), -24)
    # --- THE WORLD ANSWERS / cut C's map
    add("map_parchment", 1920, lambda r: parchment(r, s(168)), -27, cuts=("C",))
    # --- THE ACCORD (v2: flames merge at 2160, carved ring 2320-2380, hearth flare 2385-2399)
    add("accord_torch_field", 2072, lambda r: torch_field(r, s(100)), -31)

    def merge(r):
        conv = np.zeros((int(s(64) * SR), 2), np.float32)
        for k, (st, p) in enumerate([(0, -0.9), (5, 0.8), (11, -0.5), (16, 0.55), (22, -0.2), (27, 0.25)]):
            w = WH(r, s(40 - st), f0=180, f1=1800, peak=0.85, q=0.9, pan0=p, pan1=0.0)
            i0 = int(s(st) * SR)
            conv[i0:i0 + len(w)] += w[: len(conv) - i0] * 0.6
        rr = RO(r, 2.0, lead=0, size=1.1, sustain=0.4)
        i0 = int(s(40) * SR)
        conv[i0:] += rr[: len(conv) - i0] * 0.8
        return S1.fade(conv, 0.3, 0.4)
    add("torches_merge", 2160, merge, -14, lead=40)
    add("ring_sweep", 2320, lambda r: ring_sweep(r, s(64)), -18)
    add("hearth_flare", 2385, lambda r: hearth_flare(r, s(2393 - 2385) + 0.0), -14)
    # --- DAWN
    add("dawn_air", 2400, lambda r: dawn_air(r, s(250)), -31, cuts=("C",))
    # --- CODA (v1 +160)
    add("wind_hill_coda", 2620, lambda r: W(r, s(348), strength=0.5, howl=0.15, hiss=0.08, cut=600), -25)
    add("torch_close_coda", 2640, lambda r: CR(r, s(160), rate=9, level=0.7, breath=0.5), -28)
    add("torch_handover_rustle", 2740, lambda r: S1.rustle(r, s(40)), -26)
    # as the torch changes hands: a faint echo of her three flint strikes (1236/1262/1290)
    for i, f in enumerate([2756, 2768, 2780]):
        add(f"flint_echo_{i + 1}", f, (lambda k: lambda r: flint_echo(k))(i), [-25.0, -25.0, -23.0][i])
    add("beacon_catch", 2800, lambda r: RO(r, 2.5, lead=4 / 24, size=0.9, sustain=0.35, bright=0.8), -17, lead=4)
    times = [2805, 2811, 2817, 2823, 2829, 2835, 2842, 2849, 2856, 2863, 2870, 2876]
    for i, f in enumerate(times):
        side = 1 if i % 2 == 0 else -1
        pan = side * min(0.9, 0.15 + 0.07 * i)
        add(f"answering_fire_{i + 1:02d}", f,
            (lambda p: lambda r: S1.panner(S1.whump(r, 1.0)[:, 0], p))(pan), -24 - 0.6 * i)
    return E


# ---------------------------------------------------------------------------
def _clip_ready(a):
    a = np.asarray(a, np.float32)
    if a.ndim == 1:
        a = np.stack([a, a], 1)
    a = a - a.mean(axis=0, keepdims=True)
    fl = int(0.004 * SR)
    a[:fl] *= np.linspace(0, 1, fl)[:, None]
    a[-fl:] *= np.linspace(1, 0, fl)[:, None]
    return a / (np.abs(a).max() + 1e-9) * 0.5          # peak -6 dBFS


def make_events(cut):
    os.makedirs(EV_DIR, exist_ok=True)
    meta = []
    for e in event_specs(cut):
        fn = f"{e['name']}.wav"
        path = os.path.join(EV_DIR, fn)
        a = _clip_ready(e["maker"](rng_for(e["name"])))
        sf.write(path, a, SR, subtype="PCM_24")
        meta.append(dict(name=e["name"], file=f"sfx_events/{fn}", frame=e["hit"] - e["lead"], hit_frame=e["hit"],
                         gain_db=round(e["gain_db"] + 6.02, 2), breath_exempt=bool(e["exempt"])))
    json.dump(meta, open(os.path.join(OUT, f"sfx_events_{cut}.json"), "w"), indent=1)
    print(f"sfx {cut}: {len(meta)} events", flush=True)
    return meta


def assemble(cut, total_n, irs, breaths):
    """The SFX stem from sfx_events_<cut>.json (so JSON edits are all it takes)."""
    import mix as MX
    import render_v2 as R
    meta = json.load(open(os.path.join(OUT, f"sfx_events_{cut}.json")))
    wins = R.breath_windows(breaths)
    stem = np.zeros((total_n, 2), np.float32)
    for e in meta:
        a, sr = sf.read(os.path.join(OUT, e["file"]), dtype="float32", always_2d=True)
        assert sr == SR
        i0 = int(e["frame"]) * FR
        a = a * np.float32(10 ** (e["gain_db"] / 20))
        if i0 < 0:
            a, i0 = a[-i0:], 0
        n = min(len(a), total_n - i0)
        if n <= 0:
            continue
        a = a[:n]
        if wins and not e.get("breath_exempt"):
            a = a * R.breath_env(n, wins, offset=i0)[:, None]
        stem[i0:i0 + n] += a
    out_ir = MX.get_ir("outdoor", rt_mid=0.9, length=1.4, seed=11, predelay=0.03, er_gain=0.9, bright=1.2)
    wet = R.convolve_with_breaths(stem * np.float32(0.25), out_ir, wins, total_n)
    hall = R.convolve_with_breaths(stem * np.float32(0.08), irs, wins, total_n)
    return (stem + wet + hall).astype(np.float32), meta


def build_and_render(cut, total_n, irs, breaths):
    make_events(cut)
    return assemble(cut, total_n, irs, breaths)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cut", default="AB")
    ap.add_argument("--assemble", action="store_true", help="only rebuild the stem from the JSON")
    args = ap.parse_args()
    import render_v2 as R
    import mix as MX
    import score_v2
    irs = list(np.load(os.path.join(MUSIC, "cache", "ir_hall.npy")))
    if not args.assemble:
        make_events(args.cut.upper())
    y, _ = assemble(args.cut.upper(), R.RENDER_N, irs, score_v2.BREATHS)
    np.save(os.path.join(MUSIC, "cache", "v2", f"sfx_stem_{args.cut.upper()}.npy"), y)
    print("stem peak", 20 * np.log10(np.abs(y).max() + 1e-12), "dBFS")
