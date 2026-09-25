"""One command re-renders everything:

    cd music/src && python3 render.py            # parts (cached) + mix + analysis
    python3 render.py --force                    # ignore the part cache
    python3 render.py --only vln1,choir          # re-render some parts
    python3 render.py --no-analysis

Parts are rendered by at most 2 worker processes and cached in
music/cache/parts/<name>.npy keyed by a hash of the part data + engine code.
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

from dsl import SR, TOTAL_N  # noqa: E402
import mix as MX  # noqa: E402

MUSIC = os.path.dirname(HERE)
CACHE = os.path.join(MUSIC, "cache")
PARTS_DIR = os.path.join(CACHE, "parts")
OUT = os.path.join(MUSIC, "out")
os.makedirs(PARTS_DIR, exist_ok=True)
os.makedirs(OUT, exist_ok=True)

RENDER_N = TOTAL_N + int(0.5 * SR)   # small tail margin, trimmed at the end


def code_hash(kind):
    files = ["dsl.py", "sampler.py", "sfz.py"] if kind == "sampler" else ["dsl.py", "synth.py", "sampler.py"]
    h = hashlib.sha1()
    for f in files:
        h.update(open(os.path.join(HERE, f), "rb").read())
    return h.hexdigest()[:12]


def part_key(pd):
    s = json.dumps({k: v for k, v in pd.items() if k not in ("pan", "width", "gain_db", "send",
                                                             "depth", "bus")},
                   sort_keys=True, default=str)
    return hashlib.sha1((s + code_hash(pd["kind"])).encode()).hexdigest()[:16]


def _render_one(pd):
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    t = time.time()
    if pd["kind"] == "sampler":
        import sampler
        buf = sampler.render_part(pd, RENDER_N)
        sampler._save_meta_safe = True
    else:
        import synth
        buf = synth.render_part(pd, RENDER_N)
    buf = np.nan_to_num(buf).astype(np.float32)
    path = os.path.join(PARTS_DIR, pd["name"] + ".npy")
    np.save(path + ".tmp.npy", buf)
    os.replace(path + ".tmp.npy", path)
    with open(os.path.join(PARTS_DIR, pd["name"] + ".key"), "w") as fh:
        fh.write(pd["_key"])
    return pd["name"], time.time() - t, float(np.abs(buf).max())


def render_parts(parts, force=False, only=None, jobs=2):
    todo = []
    for name, p in parts.items():
        pd = p.to_dict() if hasattr(p, "to_dict") else p
        pd["_key"] = part_key(pd)
        kpath = os.path.join(PARTS_DIR, name + ".key")
        cached = (os.path.exists(kpath) and open(kpath).read().strip() == pd["_key"]
                  and os.path.exists(os.path.join(PARTS_DIR, name + ".npy")))
        if force or (only and name in only) or not cached:
            todo.append(pd)
    # heavy parts first for better packing
    weight = {"choir": 5, "taiko": 3, "organ": 3}
    todo.sort(key=lambda d: -(weight.get(d["inst"], 1) * len(d["notes"])))
    print(f"rendering {len(todo)} / {len(parts)} parts with {jobs} workers", flush=True)
    if not todo:
        return
    t0 = time.time()
    with Pool(jobs, maxtasksperchild=6) as pool:
        for name, dt, pk in pool.imap_unordered(_render_one, todo):
            print(f"  {name:14s} {dt:6.1f}s  peak {20 * np.log10(pk + 1e-9):6.1f} dBFS", flush=True)
    print(f"parts done in {time.time() - t0:.0f}s", flush=True)


# ---------------------------------------------------------------------------
BUS_GAIN = dict(strings=0.0, brass=0.0, winds=0.0, perc=0.0, keys=0.0, choir=0.0, synth=0.0)


def load_stem(name):
    return np.load(os.path.join(PARTS_DIR, name + ".npy"), mmap_mode="r")


def mix_score(parts, ir):
    dry = np.zeros((RENDER_N, 2), np.float32)
    send = np.zeros((RENDER_N, 2), np.float32)
    report = []
    for name, p in parts.items():
        y = np.array(load_stem(name), dtype=np.float32)
        g = 10 ** ((p.gain_db + BUS_GAIN.get(p.bus, 0.0)) / 20)
        y = MX.pan_width(y * g, p.pan, p.width)
        y = MX.depth_eq(y, p.depth)
        dry += y * (1.0 - 0.35 * p.depth)       # distant = less direct sound
        send += y * p.send
        report.append((name, 20 * np.log10(np.sqrt((y ** 2).mean()) + 1e-12)))
    wet = MX.convolve_stereo(send, ir)
    out = dry + wet
    out = MX.highpass(out, 22.0)
    return out, report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--only", default="")
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--no-analysis", action="store_true")
    ap.add_argument("--no-sfx", action="store_true")
    args = ap.parse_args()
    import score
    t0 = time.time()
    parts = score.build()
    only = set(x for x in args.only.split(",") if x)
    render_parts(parts, force=args.force, only=only, jobs=min(2, args.jobs))

    ir = MX.get_ir("hall", rt_mid=3.1)
    score_mix, rep = mix_score(parts, ir)
    print(f"score mixed ({time.time() - t0:.0f}s)")

    import sfx
    sfx_mix, events = sfx.build_and_render(RENDER_N, ir, jobs=min(2, args.jobs)) if not args.no_sfx else (
        np.zeros_like(score_mix), [])

    import master
    master.finish(score_mix, sfx_mix, events)
    print(f"all done in {time.time() - t0:.0f}s")
    if not args.no_analysis:
        import analyze
        analyze.main()


if __name__ == "__main__":
    main()
