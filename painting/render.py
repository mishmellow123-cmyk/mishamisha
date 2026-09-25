#!/usr/bin/env python3
"""
Render the still life to a linear-light HDR image (numpy), caching the
landscape and the caustic between runs.

Usage: python3 render.py [scale] [out.npy]      scale 1.0 -> 2400 x 3200
"""
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import landscape as L  # noqa: E402
import props  # noqa: E402
import scene as S  # noqa: E402

SCALE = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
OUT = sys.argv[2] if len(sys.argv) > 2 else "render.npy"
CACHE = os.environ.get("CACHE", os.path.join(os.path.dirname(os.path.abspath(__file__)), ".cache"))
os.makedirs(CACHE, exist_ok=True)
W, H = int(2400 * SCALE), int(3200 * SCALE)
T0 = time.time()


def log(m):
    print(f"[{time.time() - T0:6.1f}s] {m}", flush=True)


CAM = np.array([0.35, 2.75, -9.8])
LOOK = np.array([0.0, 1.30, 0.0])
HFOV = math.radians(36.0)


def cached(name, fn):
    path = os.path.join(CACHE, name + ".npz")
    if os.path.exists(path):
        z = np.load(path, allow_pickle=True)
        return z["a"], tuple(z["meta"])
    a, meta = fn()
    np.savez(path, a=a, meta=np.array(meta, dtype=object))
    return a, meta


def env_hi():
    ppd = 70.0
    ext = (-24.0, 24.0, -17.0, 26.0)
    img, dist = L.paint_env(*ext, ppd)
    return np.concatenate([img, np.isinf(dist)[..., None].astype(np.float32)], -1), (*ext, ppd)


def env_wide():
    ppd = 12.0
    ext = (-90.0, 90.0, -60.0, 72.0)
    img, dist = L.paint_env(*ext, ppd)
    return np.concatenate([img, np.isinf(dist)[..., None].astype(np.float32)], -1), (*ext, ppd)


def caustic():
    img, ext = S.trace_caustic()
    return img, ext


def main():
    hi = cached("env_hi", env_hi)
    log("landscape (near)")
    wide = cached("env_wide", env_wide)
    log("landscape (wide)")
    cau = cached("caustic", caustic)
    log(f"caustic (peak {cau[0].max():.1f})")
    ins_ext = (-2.3, 3.0, -1.6, -0.5)
    ins = props.inscription_height(ins_ext)
    letter = props.letter_texture()
    log("props")
    maps = S.Maps((hi[0], hi[1]), (wide[0], wide[1]), (cau[0], cau[1]), letter, (ins, ins_ext))

    fwd = S.nrm((LOOK - CAM)[None])[0]
    right = S.nrm(np.cross(fwd, np.array([0, 1.0, 0]))[None])[0]
    right = -right if right[0] < 0 else right
    up = np.cross(fwd, right)
    f = (W / 2) / math.tan(HFOV / 2)
    out = np.zeros((H, W, 3), np.float32)
    cat = np.zeros((H, W), np.uint8)       # what each pixel is: 0 world, 1 glass, 2 top, 3 face, 4 column, 5 letter
    skyness = np.zeros((H, W), np.float32)
    elev = np.zeros((H, W), np.float32)
    strip = max(8, int(200000 / W))
    for y0 in range(0, H, strip):
        y1 = min(H, y0 + strip)
        yy, xx = np.mgrid[y0:y1, 0:W].astype(np.float64)
        px = (xx + 0.5 - W / 2) / f
        py = -(yy + 0.5 - H / 2) / f
        d = S.nrm(fwd[None] + px.reshape(-1, 1) * right[None] + py.reshape(-1, 1) * up[None])
        o = np.broadcast_to(CAM, d.shape).copy()
        t_s = S.hit_sphere(o, d)
        glass = np.isfinite(t_s)
        col = np.zeros((len(d), 3))
        if (~glass).any():
            col[~glass] = S.trace_scene(maps, o[~glass], d[~glass])
        if glass.any():
            p = o[glass] + d[glass] * t_s[glass, None]
            col[glass] = S.shade_glass(maps, o[glass], d[glass], p)
        out[y0:y1] = col.reshape(y1 - y0, W, 3)
        tt = S.hit_top(o, d)
        tf = S.hit_front(o, d)
        tc, _ = S.hit_columns(o, d)
        th = S.hit_hanging(o, d)
        tm = np.minimum.reduce([tt, tf, tc, th])
        c = np.zeros(len(d), np.uint8)
        c[(tm == tt) & np.isfinite(tm)] = 2
        c[(tm == tf) & np.isfinite(tm)] = 3
        c[(tm == tc) & np.isfinite(tm)] = 4
        c[(tm == th) & np.isfinite(tm)] = 5
        p = o + d * np.where(np.isfinite(tm), tm, 0)[:, None]
        on_letter = (c == 2) & (p[:, 0] > S.LX0) & (p[:, 0] < S.LX1) & (p[:, 2] < S.LZ_FAR)
        c[on_letter] = 5
        c[glass] = 1
        cat[y0:y1] = c.reshape(y1 - y0, W)
        world = c == 0
        sk = np.zeros(len(d), np.float32)
        if world.any():
            sk[world] = S.sample_env(maps, d[world], channel=3)[:, 0]
        skyness[y0:y1] = sk.reshape(y1 - y0, W)
        elev[y0:y1] = np.degrees(np.arcsin(d[:, 1])).reshape(y1 - y0, W)
        if (y0 // strip) % 10 == 0:
            log(f"rows {y0}-{y1}")
    np.save(OUT, out)
    np.savez_compressed(OUT.replace(".npy", "_aux.npz"), cat=cat, elev=elev, sky=skyness)
    log(f"saved {OUT} {out.shape}")


if __name__ == "__main__":
    main()
