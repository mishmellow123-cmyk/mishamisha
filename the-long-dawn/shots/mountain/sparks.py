"""Deterministic spark simulation for the hero beacons -> per-frame streak-quad meshes.

python sparks.py run            -> cache/run_sparks/f_%05d.bin (+ .json meta)

Each spark: born in the fire-basket (burst on ignition + a steady stream), rises on buoyancy
that fades as it cools, drags toward the wind, wanders in a smooth turbulence field, cools from
white-yellow to dull red and dies. Every frame's quads are camera-facing streaks from the
spark's position at the shutter's open to its close (the spark's own motion blur); the camera's
motion blur is added by EEVEE. Width is at least ~1.3 px with brightness scaled to conserve
energy (no strobing sub-pixel dots).
"""
import json
import math
import os
import sys

import numpy as np
from numba import njit

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
CACHE = os.path.join(HERE, 'cache')
FPS = 24.0
WIND = np.array([4.5, 1.0, 0.0])      # m/s toward the east (world +x), a little north
SHUTTER = 0.5 / FPS

# look.blackbody stops
BB = np.array([[0.00, 0.35, 0.02, 0.00], [0.30, 0.90, 0.12, 0.01], [0.55, 1.00, 0.36, 0.04],
               [0.78, 1.00, 0.68, 0.25], [1.00, 1.00, 0.95, 0.85]])


def bb(t):
    t = np.clip(t, 0, 1)
    out = np.zeros(t.shape + (3,))
    for k in range(4):
        a, b = BB[k], BB[k + 1]
        w = np.clip((t - a[0]) / (b[0] - a[0]), 0, 1)[..., None]
        seg = ((t >= a[0]) & (t <= b[0] + 1e-9))[..., None]
        out = np.where(seg, a[1:] * (1 - w) + b[1:] * w, out)
    return out


@njit(cache=True)
def _hash01(i, k, seed):
    a = (i * 73856093) ^ (k * 19349663) ^ (seed * 83492791)
    a = a & 0xFFFFFFFF
    a = ((a ^ 61) ^ (a >> 16)) & 0xFFFFFFFF
    a = (a + (a << 3)) & 0xFFFFFFFF
    a = a ^ (a >> 4)
    a = (a * 0x27D4EB2D) & 0xFFFFFFFF
    a = a ^ (a >> 15)
    return a / 4294967296.0


@njit(cache=True)
def _turb(x, y, z, t, s):
    # smooth divergence-light swirl from a few travelling sinusoids (deterministic)
    ax = (math.sin(0.9 * y + 1.3 * t + s) + 0.6 * math.sin(1.7 * z - 2.1 * t + 2.0 * s)
          + 0.35 * math.sin(3.1 * y + 2.9 * z + 4.0 * t))
    ay = (math.sin(1.1 * z - 1.1 * t + 0.5 * s) + 0.6 * math.sin(1.9 * x + 1.7 * t)
          + 0.35 * math.sin(2.7 * x - 3.3 * z - 3.7 * t + s))
    az = (0.5 * math.sin(1.3 * x + 0.8 * t + s) + 0.3 * math.sin(2.3 * y - 2.6 * t))
    return ax, ay, az


@njit(cache=True)
def simulate(t_ign, base, scale, seed, fr0, fr1, wind, burst, rate, out_pos, out_tail, out_T,
             out_alive):
    """Sequential sim. Particle i is born at time birth(i). Records head/tail position,
    temperature and alive flag at each frame fr in [fr0, fr1]."""
    n = out_pos.shape[1]
    nf = out_pos.shape[0]
    # birth times: burst over the first 0.35 s, then a steady stream with a roar at the start
    births = np.empty(n)
    nb = int(burst)
    for i in range(n):
        if i < nb:
            births[i] = t_ign + 0.35 * _hash01(i, 0, seed) ** 1.5
        else:
            # invert cumulative emission of rate*(1 + 1.5 exp(-(t-t0)/1.5))
            k = i - nb
            # approximate: uniform stream
            births[i] = t_ign + 0.05 + k / rate
    pos = np.zeros((n, 3))
    vel = np.zeros((n, 3))
    life = np.zeros(n)
    born = np.zeros(n, np.bool_)
    dt = 1.0 / 96.0
    t = t_ign
    t_end = fr1 / 24.0 + 0.1
    fi = 0
    sq = math.sqrt(scale)
    while t < t_end:
        # births
        for i in range(n):
            if not born[i] and births[i] <= t:
                born[i] = True
                r = 0.34 * scale * math.sqrt(_hash01(i, 1, seed))
                th = 6.2831853 * _hash01(i, 2, seed)
                pos[i, 0] = base[0] + r * math.cos(th)
                pos[i, 1] = base[1] + r * math.sin(th)
                pos[i, 2] = base[2] + 0.25 * scale * _hash01(i, 3, seed)
                up = (2.5 + 5.0 * _hash01(i, 4, seed)) * sq
                if i < nb:
                    up *= 1.9 + 1.4 * _hash01(i, 9, seed)
                g1 = _hash01(i, 5, seed) - 0.5
                g2 = _hash01(i, 6, seed) - 0.5
                spread = 4.2 if i < nb else 1.4
                vel[i, 0] = spread * g1 * 2.0 * sq
                vel[i, 1] = spread * g2 * 2.0 * sq
                vel[i, 2] = up
                life[i] = 1.1 + 2.6 * _hash01(i, 7, seed) ** 0.8
        # integrate
        for i in range(n):
            if not born[i]:
                continue
            age = t - births[i]
            if age > life[i] + 0.2:
                continue
            T = math.exp(-age / (0.55 * life[i]))
            tx, ty, tz = _turb(pos[i, 0] * 0.35, pos[i, 1] * 0.35, pos[i, 2] * 0.35, t,
                               _hash01(i, 8, seed) * 6.0)
            ax = -1.1 * (vel[i, 0] - wind[0]) + 3.0 * tx
            ay = -1.1 * (vel[i, 1] - wind[1]) + 3.0 * ty
            az = -1.1 * (vel[i, 2] - wind[2]) + 7.5 * T * sq + 2.0 * tz - 1.2
            vel[i, 0] += ax * dt
            vel[i, 1] += ay * dt
            vel[i, 2] += az * dt
            pos[i, 0] += vel[i, 0] * dt
            pos[i, 1] += vel[i, 1] * dt
            pos[i, 2] += vel[i, 2] * dt
        t += dt
        # record at frame times (head at the shutter close, tail at open)
        while fi < nf and (fr0 + fi) / 24.0 <= t + 1e-9:
            tf = (fr0 + fi) / 24.0
            for i in range(n):
                age = tf - births[i]
                if born[i] and age >= 0.0 and age < life[i]:
                    out_alive[fi, i] = True
                    out_pos[fi, i, 0] = pos[i, 0]
                    out_pos[fi, i, 1] = pos[i, 1]
                    out_pos[fi, i, 2] = pos[i, 2]
                    sh = min(SHUTTER, age)
                    out_tail[fi, i, 0] = pos[i, 0] - vel[i, 0] * sh
                    out_tail[fi, i, 1] = pos[i, 1] - vel[i, 1] * sh
                    out_tail[fi, i, 2] = pos[i, 2] - vel[i, 2] * sh
                    fade = min(age / 0.08, 1.0) * min((life[i] - age) / 0.35, 1.0)
                    out_T[fi, i] = math.exp(-age / (0.55 * life[i])) * fade
            fi += 1
    return fi


def quads(head, tail, T, cam, fov_deg, W=1920, size=0.045, gain=38.0):
    """Camera-facing tapered streak quads. Returns (verts [n*4,3], colors [n*4,4])."""
    c = np.asarray(cam, float)
    D = head - tail
    L = np.linalg.norm(D, axis=1)
    V = c[None, :] - head
    dist = np.linalg.norm(V, axis=1)
    V /= dist[:, None]
    # pixel footprint -> minimum width ~1.3 px, energy conserved
    px = dist * 2 * math.tan(math.radians(fov_deg / 2)) / W
    w = np.maximum(size, 1.3 * px)
    energy = (size / w) ** 2
    # direction: the streak, or the camera 'up'-ish if the streak is shorter than its width
    minL = 1.5 * w
    Dn = np.where(L[:, None] > 1e-6, D / np.maximum(L, 1e-6)[:, None], np.array([0, 0, 1.0]))
    short = L < minL
    tail2 = np.where(short[:, None], head - Dn * minL[:, None], tail)
    S = np.cross(Dn, V)
    Sn = np.linalg.norm(S, axis=1)
    S = np.where(Sn[:, None] > 1e-6, S / np.maximum(Sn, 1e-6)[:, None], np.array([1.0, 0, 0]))
    S *= (w / 2)[:, None]
    v = np.stack([head + S, head - S, tail2 - 0.35 * S, tail2 + 0.35 * S], 1).reshape(-1, 3)
    # colour: blackbody of temperature, brightness ~ T^1.6, spread over the streak length
    stretch = np.maximum(np.maximum(L, minL) / w, 1.0)
    I = gain * np.power(np.clip(T, 0, 1), 1.6) * energy / np.sqrt(stretch)
    col = bb(0.35 + 0.65 * T) * I[:, None]
    ch = np.concatenate([col, np.ones((len(col), 1))], 1)
    ct = np.concatenate([col * 0.25, np.ones((len(col), 1))], 1)
    cc = np.stack([ch, ch, ct, ct], 1).reshape(-1, 4)
    return v.astype(np.float32), cc.astype(np.float32)


def run(shot):
    B = json.load(open(os.path.join(CACHE, '%s_beacons.json' % shot)))
    cj = json.load(open(os.path.join(CACHE, '%s_cam.json' % shot)))
    cams = {int(c['frame']): c for c in cj['frames']}
    fr = sorted(cams)
    fr0, fr1 = fr[0], fr[-1]
    outdir = os.path.join(CACHE, '%s_sparks' % shot)
    os.makedirs(outdir, exist_ok=True)
    heroes = [b for b in B if b['name'] in ('S0', 'B3')]
    sims = []
    for k, b in enumerate(heroes):
        sc = b['scale']
        base = np.array(b['pos']) + np.array([0, 0, 1.25 * sc])
        t0 = b['t'] / FPS
        rate = 70.0 * sc
        burst = int(1100 * sc)
        n = burst + int(rate * (fr1 / FPS - t0 + 0.2)) + 10
        nf = fr1 - fr0 + 1
        P = np.zeros((nf, n, 3))
        Tl = np.zeros((nf, n, 3))
        TT = np.zeros((nf, n))
        A = np.zeros((nf, n), np.bool_)
        # simulate from ignition; record only the shot's frames
        simulate(t0, base, sc, 101 + 17 * k, fr0, fr1, WIND, burst, rate, P, Tl, TT, A)
        sims.append((P, Tl, TT, A))
        print(b['name'], 'particles', n, 'max alive', A.sum(1).max(), flush=True)
    meta = {}
    for fi, f in enumerate(range(fr0, fr1 + 1)):
        c = cams[f]
        vs, cs = [], []
        for (P, Tl, TT, A) in sims:
            m = A[fi]
            if m.sum() == 0:
                continue
            v, col = quads(P[fi, m], Tl[fi, m], TT[fi, m], c['loc'], c['hfov'])
            vs.append(v)
            cs.append(col)
        if vs:
            v = np.concatenate(vs)
            col = np.concatenate(cs)
        else:
            v = np.zeros((0, 3), np.float32)
            col = np.zeros((0, 4), np.float32)
        with open(os.path.join(outdir, 'f_%05d.bin' % f), 'wb') as fh:
            fh.write(np.int32(len(v)).tobytes())
            fh.write(v.tobytes())
            fh.write(col.tobytes())
        meta[f] = len(v) // 4
    json.dump(meta, open(os.path.join(outdir, 'meta.json'), 'w'))
    print('frames', len(meta), 'max quads', max(meta.values()))


if __name__ == '__main__':
    run(sys.argv[1] if len(sys.argv) > 1 else 'run')
