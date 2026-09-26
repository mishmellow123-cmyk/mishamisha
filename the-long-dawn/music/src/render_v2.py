"""THE LONG DAWN v2 - one command re-renders a cut's score, SFX, master and analysis.

    cd music/src && python render_v2.py                 # both scores: AB then C
    python render_v2.py --cut AB                        # one score
    python render_v2.py --cut C --only c_hn1,c_vln1     # force re-render of some parts
    python render_v2.py --cut AB --no-analysis --no-sfx

Outputs (music/out/v2/): score_<cut>.wav, sfx_<cut>.wav, final_<cut>.wav (stem-linked:
score + sfx == final), sfx_events_<cut>.json + sfx_events/*.wav, analysis_<cut>/.

Parts render in at most 2 worker processes and are cached content-addressed in
music/cache/v2/parts/<name>__<key>.npy (+ .json with the sample offset), so the
v1 first-half parts are rendered once and shared by both cuts.

BREATHS (score_v2.BREATHS): every score part, the score's hall reverb AND the SFX
(dry + their outdoor/hall spaces) are ducked to -45 dB over each window, except the
listed exempt parts / events.  The reverb is convolved per segment between breaths
and each segment's tail is cut at the next breath: a real "suck", the old hall does
not come back after the downbeat.  analyze_v2 measures every breath in the output.
"""
import argparse
import hashlib
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
import soundfile as sf

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import mix as MX  # noqa: E402
from timeline_v2 import SR, BEAT_N, TOTAL_N, RENDER_N  # noqa: E402

MUSIC = os.path.dirname(HERE)
CACHE = os.path.join(MUSIC, "cache", "v2")
PARTS_DIR = os.path.join(CACHE, "parts")
OUT = os.path.join(MUSIC, "out", "v2")
os.makedirs(PARTS_DIR, exist_ok=True)
os.makedirs(OUT, exist_ok=True)

BREATH_FLOOR_DB = -45.0
BREATH_PROBES = []      # filled by mix_score: the non-exempt score (dry + hall) around each breath
FADE_DOWN_S = 0.030
FADE_UP_S = 0.005


# ---------------------------------------------------------------------------
# part cache
# ---------------------------------------------------------------------------
def code_hash(kind, extra=()):
    files = (["dsl.py", "sampler.py", "sampler_v2.py", "sfz.py"] if kind == "sampler"
             else ["dsl.py", "synth.py", "synth_v2.py", "sampler.py", "sampler_v2.py"])
    h = hashlib.sha1()
    for f in list(files) + list(extra):
        p = os.path.join(HERE, f)
        if os.path.exists(p):
            h.update(open(p, "rb").read())
    return h.hexdigest()[:12]


_CH = {}


def part_key(pd):
    kind = pd["kind"]
    if kind not in _CH:
        _CH[kind] = code_hash(kind)
    s = json.dumps({k: v for k, v in pd.items() if k not in ("pan", "width", "gain_db", "send", "depth",
                                                             "bus", "_key")},
                   sort_keys=True, default=str)
    return hashlib.sha1((s + _CH[kind] + str(RENDER_N)).encode()).hexdigest()[:16]


def stem_path(name, key):
    return os.path.join(PARTS_DIR, f"{name}__{key}")


def _worker_init():
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    import sampler
    import sampler_v2  # noqa: F401  (registers the v2 patches)
    sampler.RAW_BUDGET = 140e6       # this Mac has 8 GB shared by six departments
    sampler.RS_BUDGET = 220e6


def _render_one(pd):
    t = time.time()
    if pd["kind"] == "sampler":
        import sampler
        import sampler_v2  # noqa: F401
        buf = sampler.render_part(pd, RENDER_N)
    else:
        import synth_v2
        buf = synth_v2.render_part(pd, RENDER_N)
    buf = np.nan_to_num(buf).astype(np.float32)
    act = np.flatnonzero(np.abs(buf).max(1) > 1e-7)
    if len(act):
        a, b = int(act[0]), int(act[-1]) + 1
    else:
        a, b = 0, 1
    p = stem_path(pd["name"], pd["_key"])
    np.save(p + ".tmp.npy", buf[a:b])
    del buf
    os.replace(p + ".tmp.npy", p + ".npy")
    with open(p + ".json", "w") as fh:
        json.dump(dict(offset=a, n=b - a), fh)
    return pd["name"], time.time() - t, b - a


def render_parts(parts, force=False, only=None, jobs=2):
    todo, manifest = [], {}
    for name, p in parts.items():
        pd = p.to_dict()
        pd["_key"] = part_key(pd)
        manifest[name] = pd["_key"]
        cached = os.path.exists(stem_path(name, pd["_key"]) + ".npy") and os.path.exists(
            stem_path(name, pd["_key"]) + ".json")
        if force or (only and name in only) or not cached:
            todo.append(pd)
    weight = {"choir": 6, "taiko": 3, "organ": 3, "fmwarm": 2}
    todo.sort(key=lambda d: -(weight.get(d["inst"], 1) * len(d["notes"])))
    print(f"rendering {len(todo)} / {len(parts)} parts with {jobs} workers", flush=True)
    if todo:
        t0 = time.time()
        if jobs <= 1:
            _worker_init()
            for pd in todo:
                name, dt, n = _render_one(pd)
                print(f"  {name:16s} {dt:6.1f}s  {n / SR:6.1f}s active", flush=True)
        else:
            with Pool(jobs, initializer=_worker_init, maxtasksperchild=4) as pool:
                for name, dt, n in pool.imap_unordered(_render_one, todo):
                    print(f"  {name:16s} {dt:6.1f}s  {n / SR:6.1f}s active", flush=True)
        print(f"parts done in {time.time() - t0:.0f}s", flush=True)
    return manifest


def load_stem(name, key):
    """(offset, float32 array [n, 2]) of a cached part."""
    p = stem_path(name, key)
    meta = json.load(open(p + ".json"))
    return meta["offset"], np.load(p + ".npy", mmap_mode="r")


# ---------------------------------------------------------------------------
# breaths
# ---------------------------------------------------------------------------
def breath_windows(breaths):
    """[(i0, i1, exempt)] in samples, sorted."""
    return sorted((int(round(b0 * BEAT_N)), int(round(b1 * BEAT_N)), set(ex)) for b0, b1, ex in breaths)


def breath_env(n, wins, name=None, offset=0):
    """Gain envelope for samples [offset, offset+n): -45 dB over each window
    (30 ms cos^2 fade down from the window start, 5 ms ramp up ending exactly on the
    downbeat), unless `name` is exempt."""
    D = np.ones(n, np.float32)
    floor = 10 ** (BREATH_FLOOR_DB / 20)
    fd, fu = int(FADE_DOWN_S * SR), int(FADE_UP_S * SR)
    for i0, i1, exempt in wins:
        if name is not None and name in exempt:
            continue
        seg = np.full(i1 - i0, floor, np.float32)
        seg[:fd] = floor + (1 - floor) * np.cos(np.linspace(0, np.pi / 2, fd)) ** 2
        seg[-fu:] = floor + (1 - floor) * np.sin(np.linspace(0, np.pi / 2, fu)) ** 2
        a, b = max(i0, offset), min(i1, offset + n)
        if b > a:
            D[a - offset:b - offset] = np.minimum(D[a - offset:b - offset], seg[a - i0:b - i0])
    return D


def convolve_with_breaths(send, irs, wins, n_out):
    """Reverb that breathes: the send is convolved per segment between breath downbeats;
    each segment's wet output is faded out (30 ms) at the start of the next breath and
    discarded afterwards, so the old hall tail never returns after the suck."""
    wet = np.zeros((n_out, 2), np.float32)
    starts = [0] + [i1 for _, i1, _ in wins]
    cut_at = [i0 for i0, _, _ in wins] + [None]
    fd = int(FADE_DOWN_S * SR)
    floor = 10 ** (BREATH_FLOOR_DB / 20)
    for k, s0 in enumerate(starts):
        s1 = starts[k + 1] if k + 1 < len(starts) else len(send)
        if s1 <= s0:
            continue
        seg = send[s0:s1]
        if not np.any(seg):
            continue
        y = MX.convolve_stereo(np.ascontiguousarray(seg), irs)      # length s1 - s0
        # MX.convolve_stereo truncates to the input length: recompute with the tail
        tail = _tail(seg, irs)
        full = np.concatenate([y, tail], 0)
        c = cut_at[k]
        if c is not None:
            keep = c - s0 + fd
            env = np.ones(len(full), np.float32)
            if c - s0 < len(full):
                a = max(0, c - s0)
                m = min(fd, len(full) - a)
                env[a:a + m] = floor + (1 - floor) * np.cos(np.linspace(0, np.pi / 2, fd))[:m] ** 2
                env[a + m:] = 0.0
            full = full[:max(0, min(len(full), keep))] * env[:max(0, min(len(full), keep))][:, None]
        e = min(n_out, s0 + len(full))
        wet[s0:e] += full[:e - s0]
    return wet


def _tail(seg, irs):
    """The part of the convolution beyond the input length (the reverb tail)."""
    from scipy import signal
    LL, LR, RL, RR = irs
    L = len(LL)
    x = seg[-min(len(seg), L):]
    # tail = conv(full, ir)[len(seg):], which only depends on the last L-1 input samples
    wl = signal.oaconvolve(x[:, 0], LL) + signal.oaconvolve(x[:, 1], RL)
    wr = signal.oaconvolve(x[:, 0], LR) + signal.oaconvolve(x[:, 1], RR)
    return np.stack([wl, wr], 1)[len(x):].astype(np.float32)


# ---------------------------------------------------------------------------
# score mix
# ---------------------------------------------------------------------------
def mix_score(parts, manifest, irs, breaths, eq=None, groups=None):
    eq = eq or {}
    wins = breath_windows(breaths)
    dry = np.zeros((RENDER_N, 2), np.float32)
    send = np.zeros((RENDER_N, 2), np.float32)
    report = []
    pre_s, post_s = int(0.3 * SR), int(0.2 * SR)
    probes = [np.zeros((i1 - i0 + pre_s + post_s, 2), np.float32) for i0, i1, _ in wins]
    groups = groups or {}
    gdry = {g: np.zeros((RENDER_N, 2), np.float32) for g in groups}
    gsend = {g: np.zeros((RENDER_N, 2), np.float32) for g in groups}
    for name, p in parts.items():
        off, y = load_stem(name, manifest[name])
        y = np.array(y, dtype=np.float32)
        grp = next((g for g in groups if name.startswith(g)), None)
        n = len(y)
        if wins:
            y *= breath_env(n, wins, name=name, offset=off)[:, None]
        g = 10 ** (p.gain_db / 20)
        y = MX.pan_width(y * g, p.pan, p.width)
        y = MX.depth_eq(y, p.depth)
        spec = eq.get(name, (None, None))
        hp, lp = spec[0], spec[1]
        if hp:
            y = MX.highpass(y, hp)
        if lp:
            from scipy import signal as _sg
            bb, aa = _sg.butter(2, lp / (SR / 2))
            y = _sg.lfilter(bb, aa, y, axis=0).astype(np.float32)
        if len(spec) > 2 and spec[2]:
            # "air": a parallel high shelf (+g dB above f), phase-coherent enough for a gentle lift
            g_db, f_sh = spec[2], (spec[3] if len(spec) > 3 else 6000.0)
            y = y + MX.highpass(y, f_sh, order=2) * np.float32(10 ** (g_db / 20) - 1)
        e = min(RENDER_N, off + n)
        (gdry[grp] if grp else dry)[off:e] += y[:e - off] * (1.0 - 0.35 * p.depth)
        (gsend[grp] if grp else send)[off:e] += y[:e - off] * p.send
        for k, (i0, i1, ex) in enumerate(wins):
            if name in ex:
                continue
            w0, w1 = i0 - pre_s, i1 + post_s
            a0, a1 = max(w0, off), min(w1, off + n)
            if a1 > a0:
                probes[k][a0 - w0:a1 - w0] += y[a0 - off:a1 - off] * (1.0 - 0.35 * p.depth)
        report.append((name, 20 * np.log10(np.sqrt((y ** 2).mean()) + 1e-12)))
        del y
    wet = convolve_with_breaths(send, irs, wins, RENDER_N)
    del send
    # bus groups (e.g. the race-entrance reinforcement): own hall, then a true-peak bus limiter
    # so their transients sit under the score's peaks instead of making the master pump
    for g, cfg in groups.items():
        gw = convolve_with_breaths(gsend[g], irs, wins, RENDER_N)
        bus = gdry[g] + gw
        del gw
        act = np.flatnonzero(np.abs(bus).max(1) > 1e-7)
        if len(act):
            a0, a1 = max(0, act[0] - SR), min(RENDER_N, act[-1] + SR)
            seg = bus[a0:a1]
            seg *= limiter_gain(seg, cfg.get("ceiling_db", -6.0), release=cfg.get("release", 0.08))[:, None]
            dry[a0:a1] += seg
        del bus
    for k, (i0, i1, ex) in enumerate(wins):
        probes[k] += wet[i0 - pre_s:i1 + post_s]
    BREATH_PROBES.clear()
    for k, (i0, i1, ex) in enumerate(wins):
        x = probes[k].astype(np.float64)
        def r(a0, a1):
            return float(10 * np.log10((x[a0:a1] ** 2).mean() + 1e-20))
        L = i1 - i0
        BREATH_PROBES.append(dict(start_s=i0 / SR, end_s=i1 / SR, exempt=sorted(ex),
                                  before_db=r(0, pre_s), inside_db=r(pre_s + int(0.036 * SR), pre_s + L - int(0.006 * SR)),
                                  after_db=r(pre_s + L, pre_s + L + post_s)))
    out = dry + wet
    del dry, wet
    out = MX.highpass(out, 22.0)
    return out, report


def apply_push(y, pushes, split_hz=150.0):
    """short gain lift just after a downbeat, split at 150 Hz (zero-phase: low = filtfilt, high =
    x - low, so the bands sum back exactly) - edit/mix.py's PUSH, now in the pipeline."""
    from scipy import signal
    sos = signal.butter(2, split_hz / (SR / 2), 'low', output='sos')
    for frame, db_hi, db_lo, hold, rel in pushes:
        T = int(round(frame / 24 * SR))
        a, b = T - int(0.5 * SR), min(len(y), T + int((hold + rel + 1.0) * SR))
        seg = y[a:b].astype(np.float64)
        lo = signal.sosfiltfilt(sos, seg, axis=0)
        hi = seg - lo
        n = b - a
        t = (np.arange(n) - (T - a - int(0.005 * SR))) / SR
        env = np.where(t < 0, 0.0, np.where(t < hold, 1.0, np.cos(np.clip((t - hold) / rel, 0, 1) * np.pi / 2) ** 2))
        g_hi = 10 ** (db_hi * env / 20)
        g_lo = 10 ** (db_lo * env / 20)
        y[a:b] = (hi * g_hi[:, None] + lo * g_lo[:, None]).astype(np.float32)
    return y


# ---------------------------------------------------------------------------
# master (stem-linked: the same gain curve on score and sfx)
# ---------------------------------------------------------------------------
TARGET_LUFS = -16.0
CEIL_DB = -1.3


def lufs(x):
    import pyloudnorm as pyln
    return pyln.Meter(SR).integrated_loudness(np.asarray(x, np.float64))


def tp_peaks(x, chunk=8 * SR, pad=256):
    """per-sample max |x| over 4x oversampling, chunked (low memory)"""
    from scipy import signal
    n = len(x)
    out = np.empty(n, np.float32)
    for a in range(0, n, chunk):
        b = min(n, a + chunk)
        a0, b0 = max(0, a - pad), min(n, b + pad)
        y = signal.resample_poly(np.asarray(x[a0:b0], np.float64), 4, 1, axis=0)
        pk = np.abs(y).max(1).reshape(-1, 4).max(1)
        out[a:b] = pk[a - a0:a - a0 + (b - a)]
    return out


def true_peak_db(x):
    return 20 * np.log10(float(tp_peaks(x).max()) + 1e-12)


def limiter_gain(x, ceiling_db, lookahead=0.006, release=0.15):
    from scipy.ndimage import minimum_filter1d
    need = np.minimum(1.0, 10 ** (ceiling_db / 20) / np.maximum(tp_peaks(x), 1e-9))
    la = int(lookahead * SR)
    need = minimum_filter1d(need, size=2 * la + 1)
    gdb = 20 * np.log10(need)
    g = np.minimum(MX._smooth_gain(gdb, 0.0005, release), gdb)
    return (10 ** (g / 20)).astype(np.float32)


def master(score, sfx, cut, events=None):
    score = MX.highpass(score[:TOTAL_N].astype(np.float32), 8.0, order=1)
    sfx = MX.highpass(sfx[:TOTAL_N].astype(np.float32), 8.0, order=1)
    pre = score + sfx
    G = 10 ** ((TARGET_LUFS - lufs(pre)) / 20)
    for it in range(5):
        x = pre * np.float32(G)
        comp = MX.glue_comp_gain(x, thresh_db=-16.0, ratio=1.5, attack=0.04, release=0.4)
        y = x * comp[:, None]
        lim = limiter_gain(y, CEIL_DB)
        y *= lim[:, None]
        L = lufs(y)
        err = TARGET_LUFS - L
        print(f"  master iter {it}: {L:.2f} LUFS (gain {20 * np.log10(G):+.2f} dB, "
              f"comp GR max {20 * np.log10(comp.min()):.1f} dB, lim GR max {20 * np.log10(lim.min()):.1f} dB)",
              flush=True)
        del x, y
        if abs(err) < 0.08:
            break
        G *= 10 ** (err / 20)
    env = (np.float32(G) * comp * lim).astype(np.float32)[:, None]
    del pre, comp, lim
    s_out = score * env
    x_out = sfx * env
    # from silence at 0, to silence at the end (the film is black by 2965)
    fi, fo = int(0.010 * SR), int(0.35 * SR)
    for y in (s_out, x_out):
        y[:fi] *= np.linspace(0, 1, fi, dtype=np.float32)[:, None]
        y[-fo:] *= (np.cos(np.linspace(0, np.pi / 2, fo)) ** 2).astype(np.float32)[:, None]
    m_out = s_out + x_out
    tp = true_peak_db(m_out)
    if tp > -1.05:
        k = np.float32(10 ** ((-1.1 - tp) / 20))
        s_out *= k
        x_out *= k
        m_out = s_out + x_out
    rng = np.random.default_rng(0)
    paths = {}
    for nm, y in ((f"score_{cut}", s_out), (f"sfx_{cut}", x_out), (f"final_{cut}", m_out)):
        yd = MX.tpdf_dither_24(y, rng)
        assert len(yd) == TOTAL_N
        paths[nm] = os.path.join(OUT, nm + ".wav")
        sf.write(paths[nm], yd, SR, subtype="PCM_24")
    gdb = 20 * np.log10(G)
    if events is not None:
        evp = os.path.join(OUT, f"sfx_events_{cut}.json")
        ev = json.load(open(evp))
        for e in ev:
            e["gain_db_master"] = round(e["gain_db"] + gdb, 2)
        json.dump(ev, open(evp, "w"), indent=1)
    np.save(os.path.join(CACHE, f"master_env_{cut}.npy"), env[:, 0])
    print(f"  wrote {cut}: {lufs(m_out):.2f} LUFS, TP {true_peak_db(m_out):.2f} dBTP, master gain {gdb:+.2f} dB",
          flush=True)
    return paths


# ---------------------------------------------------------------------------
def render_cut(cut, force=False, only=(), jobs=2, do_sfx=True, do_analysis=True):
    import score_v2
    t0 = time.time()
    parts = score_v2.build(cut)
    manifest = render_parts(parts, force=force, only=set(only), jobs=jobs)
    json.dump(manifest, open(os.path.join(CACHE, f"manifest_{cut}.json"), "w"), indent=1)
    irs = MX.make_ir(rt_mid=3.1) if not os.path.exists(os.path.join(MUSIC, "cache", "ir_hall.npy")) \
        else list(np.load(os.path.join(MUSIC, "cache", "ir_hall.npy")))
    if not os.path.exists(os.path.join(MUSIC, "cache", "ir_hall.npy")):
        np.save(os.path.join(MUSIC, "cache", "ir_hall.npy"), np.stack(irs))
    score_mix, rep = mix_score(parts, manifest, irs, score_v2.BREATHS, eq=score_v2.MIX_EQ,
                               groups=getattr(score_v2, "BUS_GROUPS", {}))
    score_mix = apply_push(score_mix, getattr(score_v2, "PUSH", []))
    json.dump(BREATH_PROBES, open(os.path.join(CACHE, f"breath_probe_{cut}.json"), "w"), indent=1)
    print(f"score {cut} mixed ({time.time() - t0:.0f}s)", flush=True)
    if do_sfx:
        import sfx_v2
        sfx_mix, events = sfx_v2.build_and_render(cut, RENDER_N, irs, score_v2.BREATHS)
    else:
        sfx_mix, events = np.zeros_like(score_mix), None
    del irs
    paths = master(score_mix, sfx_mix, cut, events)
    del score_mix, sfx_mix
    print(f"{cut} done in {time.time() - t0:.0f}s", flush=True)
    if do_analysis:
        import analyze_v2
        analyze_v2.main(cut)
    return paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cut", default="all", help="AB, C or all")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--only", default="")
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--no-analysis", action="store_true")
    ap.add_argument("--no-sfx", action="store_true")
    args = ap.parse_args()
    cuts = ["AB", "C"] if args.cut.lower() == "all" else [args.cut.upper()]
    only = [x for x in args.only.split(",") if x]
    for cut in cuts:
        render_cut(cut, force=args.force, only=only, jobs=min(2, args.jobs), do_sfx=not args.no_sfx,
                   do_analysis=not args.no_analysis)


if __name__ == "__main__":
    main()
