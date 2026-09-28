"""THE LONG DAWN v3 - one engine renders any cut's score (and its effects) for its bar map.

    source ~/.venvs/longdawn/env.sh && cd music/src
    python render_v3.py B                    # score_v3_B.build -> score + effects -> master -> analysis
    python render_v3.py B --fallback         # the FALLBACK MASTER of cut B (fallback_v3.build)
    python render_v3.py B --only voice,hn_far    # force some parts to re-render
    python render_v3.py B --no-sfx           # the score alone (mastered alone)
    python render_v3.py T --barmap path.json --fallback   # any bar map

Outputs in music/out/v3/ (48 kHz / 24-bit / stereo, exactly the cut's length, from silence to
silence): <name>_score.wav + <name>_sfx.wav = <name>.wav (stem-linked master, -16 LUFS integrated,
true peak <= -1.2 dBTP), where <name> = final_<cut> or fallback_<cut>; score_<cut>.wav is a copy of
the score stem (the director's review file).  Analysis in music/analysis/v3/<name>/.

Parts are cached content-addressed in music/cache/v3/parts/ and render in ONE process (this Mac
is shared: ~8 GB, heavy swap), with the sampler's caches capped.
"""
import argparse
import hashlib
import json
import os
import shutil
import sys
import time

import numpy as np
import soundfile as sf

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import mix as MX  # noqa: E402
import render_v2 as R2  # noqa: E402  (length-agnostic helpers: breaths, limiter, true peak, LUFS)
from timeline_v3 import SR, BEAT_N, BarMap  # noqa: E402

MUSIC = os.path.dirname(HERE)
CACHE = os.path.join(MUSIC, "cache", "v3")
PARTS_DIR = os.path.join(CACHE, "parts")
OUT = os.path.join(MUSIC, "out", "v3")
AN = os.path.join(MUSIC, "analysis", "v3")
for d in (PARTS_DIR, OUT, AN):
    os.makedirs(d, exist_ok=True)

breath_windows = R2.breath_windows
breath_env = R2.breath_env
convolve_with_breaths = R2.convolve_with_breaths
limiter_gain = R2.limiter_gain
tp_peaks = R2.tp_peaks
true_peak_db = R2.true_peak_db
lufs = R2.lufs
apply_push = R2.apply_push

TARGET_LUFS = -16.0
CEIL_DB = -1.3            # the limiter's ceiling (deliverable: <= -1.2 dBTP)
BREATH_PROBES = []


# ---------------------------------------------------------------------------
# parts
# ---------------------------------------------------------------------------
def code_hash(kind):
    files = (["dsl.py", "sampler.py", "sampler_v2.py", "sfz.py"] if kind == "sampler"
             else ["dsl.py", "synth.py", "synth_v2.py", "synth_v3.py", "sampler.py"])
    h = hashlib.sha1()
    for f in files:
        p = os.path.join(HERE, f)
        if os.path.exists(p):
            h.update(open(p, "rb").read())
    return h.hexdigest()[:12]


_CH = {}


def part_key(pd, render_n):
    kind = pd["kind"]
    if kind not in _CH:
        _CH[kind] = code_hash(kind)
    s = json.dumps({k: v for k, v in pd.items() if k not in ("pan", "width", "gain_db", "send", "depth", "bus",
                                                             "_key")}, sort_keys=True, default=str)
    return hashlib.sha1((s + _CH[kind] + str(render_n)).encode()).hexdigest()[:16]


def stem_path(name, key):
    return os.path.join(PARTS_DIR, f"{name}__{key}")


def _render_one(pd, render_n):
    t = time.time()
    if pd["kind"] == "sampler":
        import sampler
        import sampler_v2  # noqa: F401
        buf = sampler.render_part(pd, render_n)
    else:
        import synth_v3
        buf = synth_v3.render_part(pd, render_n)
    buf = np.nan_to_num(buf).astype(np.float32)
    act = np.flatnonzero(np.abs(buf).max(1) > 1e-7)
    a, b = (int(act[0]), int(act[-1]) + 1) if len(act) else (0, 1)
    p = stem_path(pd["name"], pd["_key"])
    np.save(p + ".tmp.npy", buf[a:b])
    del buf
    os.replace(p + ".tmp.npy", p + ".npy")
    with open(p + ".json", "w") as fh:
        json.dump(dict(offset=a, n=b - a), fh)
    return pd["name"], time.time() - t, b - a


def render_parts(parts, render_n, force=False, only=()):
    import sampler
    import sampler_v2  # noqa: F401
    sampler.RAW_BUDGET = 140e6
    sampler.RS_BUDGET = 220e6
    todo, manifest = [], {}
    for name, p in parts.items():
        pd = p.to_dict()
        pd["_key"] = part_key(pd, render_n)
        manifest[name] = pd["_key"]
        cached = os.path.exists(stem_path(name, pd["_key"]) + ".npy") and os.path.exists(
            stem_path(name, pd["_key"]) + ".json")
        if force or name in only or not cached:
            todo.append(pd)
    print(f"rendering {len(todo)} / {len(parts)} parts", flush=True)
    t0 = time.time()
    for pd in todo:
        name, dt, n = _render_one(pd, render_n)
        print(f"  {name:18s} {dt:6.1f}s  {n / SR:6.1f}s active", flush=True)
    if todo:
        sampler.clear_caches()
        print(f"parts done in {time.time() - t0:.0f}s", flush=True)
    return manifest


def load_stem(name, key):
    p = stem_path(name, key)
    meta = json.load(open(p + ".json"))
    return meta["offset"], np.load(p + ".npy", mmap_mode="r")


# ---------------------------------------------------------------------------
# the score mix (v2's, for any length)
# ---------------------------------------------------------------------------
def mix_score(parts, manifest, irs, breaths, render_n, eq=None, groups=None):
    eq = eq or {}
    groups = groups or {}
    wins = breath_windows(breaths)
    dry = np.zeros((render_n, 2), np.float32)
    send = np.zeros((render_n, 2), np.float32)
    pre_s, post_s = int(0.3 * SR), int(0.2 * SR)
    probes = [np.zeros((i1 - i0 + pre_s + post_s, 2), np.float32) for i0, i1, _ in wins]
    gdry = {g: np.zeros((render_n, 2), np.float32) for g in groups}
    gsend = {g: np.zeros((render_n, 2), np.float32) for g in groups}
    report = []
    for name, p in parts.items():
        off, y = load_stem(name, manifest[name])
        y = np.array(y, dtype=np.float32)
        n = len(y)
        grp = next((g for g in groups if name.startswith(g)), None)
        if wins:
            y *= breath_env(n, wins, name=name, offset=off)[:, None]
        y = MX.pan_width(y * np.float32(10 ** (p.gain_db / 20)), p.pan, p.width)
        y = MX.depth_eq(y, p.depth)
        spec = eq.get(name, (None, None))
        if spec[0]:
            y = MX.highpass(y, spec[0])
        if spec[1]:
            from scipy import signal as _sg
            bb, aa = _sg.butter(2, spec[1] / (SR / 2))
            y = _sg.lfilter(bb, aa, y, axis=0).astype(np.float32)
        if len(spec) > 2 and spec[2]:
            y = y + MX.highpass(y, spec[3] if len(spec) > 3 else 6000.0, order=2) * np.float32(10 ** (spec[2] / 20) - 1)
        e = min(render_n, off + n)
        (gdry[grp] if grp else dry)[off:e] += y[:e - off] * (1.0 - 0.35 * p.depth)
        (gsend[grp] if grp else send)[off:e] += y[:e - off] * p.send
        for k, (i0, i1, ex) in enumerate(wins):
            if name in ex:
                continue
            w0, w1 = i0 - pre_s, i1 + post_s
            a0, a1 = max(w0, off), min(w1, off + n)
            if a1 > a0:
                probes[k][a0 - w0:a1 - w0] += y[a0 - off:a1 - off] * (1.0 - 0.35 * p.depth)
        report.append((name, float(20 * np.log10(np.sqrt((y ** 2).mean()) + 1e-12))))
        del y
    wet = convolve_with_breaths(send, irs, wins, render_n)
    del send
    for g, cfg in groups.items():
        gw = convolve_with_breaths(gsend[g], irs, wins, render_n)
        bus = gdry[g] + gw
        del gw
        act = np.flatnonzero(np.abs(bus).max(1) > 1e-7)
        if len(act):
            a0, a1 = max(0, act[0] - SR), min(render_n, act[-1] + SR)
            seg = bus[a0:a1]
            seg *= limiter_gain(seg, cfg.get("ceiling_db", -6.0), release=cfg.get("release", 0.08))[:, None]
            dry[a0:a1] += seg
        del bus
    BREATH_PROBES.clear()
    for k, (i0, i1, ex) in enumerate(wins):
        x = (probes[k] + wet[i0 - pre_s:i1 + post_s]).astype(np.float64)

        def r(a0, a1):
            return float(10 * np.log10((x[a0:a1] ** 2).mean() + 1e-20))
        L = i1 - i0
        BREATH_PROBES.append(dict(start_s=i0 / SR, end_s=i1 / SR, exempt=sorted(ex), before_db=r(0, pre_s),
                                  inside_db=r(pre_s + int(0.036 * SR), pre_s + L - int(0.006 * SR)),
                                  after_db=r(pre_s + L, pre_s + L + post_s)))
    out = dry + wet
    del dry, wet
    return MX.highpass(out, 22.0), report


# ---------------------------------------------------------------------------
# the master (stem-linked: one gain curve on score and effects)
# ---------------------------------------------------------------------------
def master(score, sfx, total_n, name, target=TARGET_LUFS, ceil_db=CEIL_DB, fade_out=0.35):
    score = MX.highpass(score[:total_n].astype(np.float32), 8.0, order=1)
    sfx = None if sfx is None else MX.highpass(sfx[:total_n].astype(np.float32), 8.0, order=1)
    pre = score if sfx is None else score + sfx
    G = 10 ** ((target - lufs(pre)) / 20)
    for it in range(6):
        x = pre * np.float32(G)
        comp = MX.glue_comp_gain(x, thresh_db=-16.0, ratio=1.5, attack=0.04, release=0.4)
        y = x * comp[:, None]
        lim = limiter_gain(y, ceil_db)
        if sfx is not None:                       # the score stem too must stay under the ceiling on its own
            lim = np.minimum(lim, limiter_gain(score * np.float32(G) * comp[:, None], ceil_db))
        y *= lim[:, None]
        L = lufs(y)
        print(f"  master iter {it}: {L:.2f} LUFS (gain {20 * np.log10(G):+.2f} dB, comp GR max "
              f"{20 * np.log10(comp.min()):.1f} dB, lim GR max {20 * np.log10(lim.min()):.1f} dB)", flush=True)
        del x, y
        if abs(target - L) < 0.08:
            break
        G *= 10 ** ((target - L) / 20)
    env = (np.float32(G) * comp * lim).astype(np.float32)[:, None]
    del comp, lim
    s_out = score * env
    x_out = None if sfx is None else sfx * env
    fi, fo = int(0.010 * SR), int(fade_out * SR)
    for y in (s_out, x_out):
        if y is None:
            continue
        y[:fi] *= np.linspace(0, 1, fi, dtype=np.float32)[:, None]
        y[-fo:] *= (np.cos(np.linspace(0, np.pi / 2, fo)) ** 2).astype(np.float32)[:, None]
    m_out = s_out if x_out is None else s_out + x_out
    tp = max(true_peak_db(m_out), true_peak_db(s_out))
    if tp > -1.2:
        k = np.float32(10 ** ((-1.25 - tp) / 20))
        s_out *= k
        if x_out is not None:
            x_out *= k
        m_out = s_out if x_out is None else s_out + x_out
    rng = np.random.default_rng(0)
    paths = {}
    outs = [(name, m_out)] if x_out is None else [(name + "_score", s_out), (name + "_sfx", x_out), (name, m_out)]
    for nm, y in outs:
        yd = MX.tpdf_dither_24(y, rng)
        assert len(yd) == total_n
        paths[nm] = os.path.join(OUT, nm + ".wav")
        sf.write(paths[nm], yd, SR, subtype="PCM_24")
    np.save(os.path.join(CACHE, f"master_env_{name}.npy"), env[:, 0])
    print(f"  wrote {name}: {lufs(m_out):.2f} LUFS, TP {true_peak_db(m_out):.2f} dBTP, master gain "
          f"{20 * np.log10(G):+.2f} dB", flush=True)
    return paths


def fader_curve(n, pts):
    """a premaster fader ride: [(t_s, dB), ...] breakpoints, cosine-interpolated, 0 dB outside (COMPOSER-C2: C5's
    headroom fix, which takes the master's fast gain reduction at the slit over as a slow ride)"""
    g = np.zeros(n, np.float32)
    for (x0, y0), (x1, y1) in zip(pts[:-1], pts[1:]):
        a, b = int(round(x0 * SR)), min(n, int(round(x1 * SR)))
        if b > a:
            u = np.linspace(0.0, 1.0, b - a, endpoint=False, dtype=np.float32)
            g[a:b] = y0 + (y1 - y0) * (1 - np.cos(np.pi * u)) / 2
    return (10 ** (g / 20)).astype(np.float32)


def hall_ir():
    # OPTION (SOUND lane, agreed with COMPOSER-A 20:44Z): LONGDAWN_HALL=church uses a MEASURED stone church
    # (ir_v3.irs("church"): Freesound CC0 balloon IRs, T30 ~2.2 s mid).  Unset = the synthesized hall below, so
    # B's final and the fallbacks stay reproducible.  It changes the mix only (the parts cache is dry).
    if os.environ.get("LONGDAWN_HALL", "").lower() == "church":
        import ir_v3
        print("  hall: MEASURED church IR (LONGDAWN_HALL=church)", flush=True)
        return ir_v3.irs("church")
    p = os.path.join(MUSIC, "cache", "ir_hall.npy")
    if os.path.exists(p):
        return list(np.load(p))
    irs = MX.make_ir(rt_mid=3.1)
    np.save(p, np.stack(irs))
    return irs


# ---------------------------------------------------------------------------
def render(cut, barmap=None, fallback=False, force=False, only=(), do_sfx=True, do_analysis=True):
    t0 = time.time()
    bm = BarMap(cut, barmap)
    if fallback:
        import fallback_v3
        S = fallback_v3.build(bm)
        name = f"fallback_{bm.cut}"
    else:
        mod = __import__(f"score_v3_{bm.cut}")
        S = mod.build(bm)
        name = f"final_{bm.cut}" if do_sfx else f"score_{bm.cut}"
    parts = S.used()
    manifest = render_parts(parts, bm.render_n, force=force, only=set(only))
    json.dump(manifest, open(os.path.join(CACHE, f"manifest_{name}.json"), "w"), indent=1)
    irs = hall_ir()
    score_mix, rep = mix_score(parts, manifest, irs, S.breaths, bm.render_n, eq=S.eq, groups=S.groups)
    if S.push:
        score_mix = apply_push(score_mix, S.push)
    if getattr(S, "fader", None):             # ADDITIVE (COMPOSER-C2): a slow premaster fader ride, only if a score sets it
        score_mix *= fader_curve(len(score_mix), S.fader)[:, None]
    np.save(os.path.join(CACHE, f"premaster_score_{name}.npy"), score_mix[:bm.n])
    json.dump(BREATH_PROBES, open(os.path.join(CACHE, f"breath_probe_{name}.json"), "w"), indent=1)
    print(f"score mixed ({time.time() - t0:.0f}s)", flush=True)
    sfx_mix = None
    if do_sfx and (bm.d.get("sfx") or bm.d.get("ambience")):
        import sfx_v3
        h = hashlib.sha1(json.dumps([bm.d.get("ambience"), bm.d.get("sfx"), sorted(map(str, S.breaths)),
                                     bm.render_n], sort_keys=True, default=str).encode())
        for f in ("sfx_v3.py", "sfx.py", "mix.py", "render_v2.py"):
            h.update(open(os.path.join(HERE, f), "rb").read())
        sp = os.path.join(CACHE, f"sfxstem_{name}_{h.hexdigest()[:14]}.npy")
        if os.path.exists(sp) and not force:
            sfx_mix = np.load(sp)
            print(f"effects: cached stem ({time.time() - t0:.0f}s)", flush=True)
        else:
            sfx_mix, meta = sfx_v3.build_stem(bm, bm.render_n, irs, S.breaths,
                                              out_dir=os.path.join(OUT, f"sfx_{name}"))
            for old in [f for f in os.listdir(CACHE) if f.startswith(f"sfxstem_{name}_")]:
                os.remove(os.path.join(CACHE, old))
            np.save(sp, sfx_mix)
            print(f"effects: {len(meta)} clips ({time.time() - t0:.0f}s)", flush=True)
    del irs
    paths = master(score_mix, sfx_mix, bm.n, name)
    if sfx_mix is not None and not fallback:
        shutil.copyfile(paths[name + "_score"], os.path.join(OUT, f"score_{bm.cut}.wav"))
    del score_mix, sfx_mix
    print(f"{name} done in {time.time() - t0:.0f}s", flush=True)
    if do_analysis:
        import analyze_v3
        analyze_v3.main(bm, S, name)
    return paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cut")
    ap.add_argument("--barmap", default=None)
    ap.add_argument("--fallback", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--only", default="")
    ap.add_argument("--no-sfx", action="store_true")
    ap.add_argument("--no-analysis", action="store_true")
    a = ap.parse_args()
    render(a.cut, a.barmap, a.fallback, a.force, [x for x in a.only.split(",") if x], not a.no_sfx,
           not a.no_analysis)


if __name__ == "__main__":
    main()
