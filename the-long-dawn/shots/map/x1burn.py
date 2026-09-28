"""PAGES-C (28 Sep): the map burn-throughs (C17 -> C18 at 4160, bar 71 at 5600) as screen-space layers, with ONE
explicit t_open and burn edge v2, so they burn like the book's pages. Replaces MAP-L2's road.x1_screen for these two
layers (the farm rendered those in 12-frame chunks and x1_screen's default t_open = (the chunk's first frame + 6) /
24, so the burn restarted every chunk). Only burn/look/numba: no map world is loaded.

Three layers per frame for EDIT's 'burn' transition (edit/assemble.py):
    <out>          glow  (display sRGB, added)
    <out>_matte    keep  (RGB: what is left of the outgoing picture, scorched and charred; 0 in the hole)
    <out>_cover    cover (1 - hole)
    out = O * keep + I * (1 - cover) + glow
(An EDIT that still reads the old 'x1' formula, O * keep + I * (1 - keep) + glow, gets a plain v2 burn without the
dark char band.)

    python3 x1burn.py --center 1130,485 --t-open 4152 --speed 9.4 --frames 4150-4185 --out ../../renders/x1_map_C
    python3 x1burn.py --center 960,402 --t-open 5597 --speed 6.3 --frames 5594-5640 --out ../../renders/x1_map_C71
"""
import argparse
import os
import sys

import numpy as np
from numba import njit, prange

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..', '..', 'lib'))
import look  # noqa: E402
import burn as BURN  # noqa: E402

UNIT = 100.0          # screen px per burn cm (the book's burns are seen at ~100-170 px/cm)


@njit(parallel=True, cache=True)
def _kernel(U, V, glow, keep, cover, bf, t):
    for i in prange(U.shape[0]):
        for j in range(U.shape[1]):
            brown, char, hole, edge, pool, lip, fleck, crk = BURN.field2(bf, U[i, j], V[i, j], t)
            cc = 0.45 + 0.9 * crk + 1.8 * lip
            k0 = (1 - 0.5 * brown) * (1 - char) + 0.06 * cc * char
            k1 = (1 - 0.72 * brown) * (1 - char) + 0.05 * cc * char
            k2 = (1 - 0.88 * brown) * (1 - char) + 0.045 * cc * char
            keep[i, j, 0] = k0 * (1.0 - hole)
            keep[i, j, 1] = k1 * (1.0 - hole)
            keep[i, j, 2] = k2 * (1.0 - hole)
            cover[i, j] = 1.0 - hole
            glow[i, j, 0] = edge * 4.2 + fleck * 2.6 + 0.1 * pool * (1.0 - hole)
            glow[i, j, 1] = edge * 1.25 + fleck * 0.55 + 0.035 * pool * (1.0 - hole)
            glow[i, j, 2] = edge * 0.18 + fleck * 0.06 + 0.006 * pool * (1.0 - hole)


def frames_of(spec):
    out = []
    for part in spec.split(','):
        part = part.strip()
        if '-' in part:
            a, b = part.split('-')
            out += range(int(a), int(b) + 1)
        elif part:
            out.append(int(part))
    return out


def render(center, frames, out, t_open, speed, W=1920, H=804, seed=12):
    cx, cy = center
    bf = BURN.v2(BURN.params((cx / UNIT, cy / UNIT), t_start=t_open / 24.0, speed=speed, p=1.5, amp=0.45, freq=0.35,
                             seed=seed, brown=1.6, char=0.25, edge=0.035, lead=0.3))
    outs = (out, out.rstrip('/') + '_matte', out.rstrip('/') + '_cover')
    for d in outs:
        os.makedirs(d, exist_ok=True)
    ys, xs = np.mgrid[0:H, 0:W]
    U = (xs.astype(np.float64) + 0.5) / UNIT
    V = (ys.astype(np.float64) + 0.5) / UNIT
    for f in frames:
        glow = np.zeros((H, W, 3), np.float32)
        keep = np.ones((H, W, 3), np.float32)
        cover = np.ones((H, W), np.float32)
        _kernel(U, V, glow, keep, cover, bf, f / 24.0)
        rgb = look.finish(glow, exposure=1.0, bloom_strength=0.05, bloom_threshold=1.0, vignette_amount=0.0, lift=0.0)
        look.save_png(look.frame_path(outs[0], f), rgb)
        look.save_png(look.frame_path(outs[1], f), np.clip(keep, 0, 1))
        look.save_png(look.frame_path(outs[2], f), np.repeat(cover[..., None], 3, -1))
        print('x1burn', f, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--center', required=True)
    ap.add_argument('--t-open', type=float, required=True, help='the C frame the burn opens at (one for all)')
    ap.add_argument('--speed', type=float, default=9.4, help='front speed (burn cm / s^1.5)')
    ap.add_argument('--frames', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    cx, cy = (float(v) for v in a.center.split(','))
    render((cx, cy), frames_of(a.frames), a.out, a.t_open, a.speed)


if __name__ == '__main__':
    main()
