"""Deterministic, order-independent particles for ACCORD: embers, torch flames, flame ribbons."""
import math

import numpy as np

import scene as SC
from scene import smooth, smoother, ramp, GOLD, PALE, AMBER, FIRE_HOT, FIRE_MID, FIRE_CORE

# ------------------------------------------------------------------ embers ---
_E = None


def _ember_table():
    global _E
    if _E is not None:
        return _E
    rng = np.random.default_rng(424242)
    spawns = []
    # burst at ignition
    nb = 900
    spawns.append(np.column_stack([SC.IGNITE + rng.uniform(0, 3.0, nb) ** 1.5, np.ones(nb)]))
    # steady stream
    ts = np.arange(SC.IGNITE, 2262.0, 1.0 / 26.0)
    spawns.append(np.column_stack([ts + rng.uniform(0, 1 / 26.0, ts.size), np.zeros(ts.size)]))
    # flare burst
    nf = 1400
    spawns.append(np.column_stack([SC.FLARE_T0 + rng.uniform(0, 1, nf) ** 0.7 * 26.0, 2 * np.ones(nf)]))
    sp = np.concatenate(spawns, 0)
    n = sp.shape[0]
    kind = sp[:, 1]
    E = dict(t0=sp[:, 0], kind=kind)
    E['life'] = np.where(kind == 1, rng.uniform(30, 80, n), rng.uniform(34, 84, n))
    E['r0'] = 0.42 * np.sqrt(rng.random(n))
    E['phi0'] = rng.uniform(0, 2 * np.pi, n)
    E['z0'] = 0.75 + 0.6 * rng.random(n)
    E['v0'] = np.where(kind == 1, rng.uniform(3.0, 8.5, n), np.where(kind == 2, rng.uniform(3.0, 9.0, n),
                                                                      rng.uniform(1.4, 4.4, n)))
    E['vr'] = np.where(kind == 1, rng.uniform(0.3, 1.6, n), rng.uniform(0.1, 0.8, n))
    E['om'] = rng.uniform(0.6, 2.1, n)
    E['b0'] = rng.uniform(0.4, 1.0, n) ** 2 * np.where(kind == 1, 1.6, 1.0)
    E['size'] = rng.uniform(0.004, 0.011, n)
    E['ph'] = rng.uniform(0, 2 * np.pi, n)
    E['wob'] = rng.uniform(0.02, 0.10, n)
    E['hue'] = rng.random(n)
    _E = E
    return E


def ember_pos(E, idx, t):
    """World positions (n,3) of embers idx at time t (frames). age in seconds."""
    a = (t - E['t0'][idx]) / 24.0
    tau = 0.55
    vt = 0.8
    z = E['z0'][idx] + vt * a + (E['v0'][idx] - vt) * tau * (1 - np.exp(-a / tau))
    r = E['r0'][idx] + E['vr'][idx] * a + 0.15 * a * a
    phi = E['phi0'][idx] + E['om'][idx] * a
    w = E['wob'][idx]
    x = r * np.cos(phi) + w * np.sin(7.1 * a + E['ph'][idx])
    y = r * np.sin(phi) + w * np.cos(6.3 * a + 2 * E['ph'][idx])
    return np.stack([x, y, z], -1)


def embers(t, cam_z, shutter=0.5):
    """Streak array for splat_streaks."""
    E = _ember_table()
    age = t - E['t0']
    alive = (age > 0) & (age < E['life'])
    idx = np.nonzero(alive)[0]
    if idx.size == 0:
        return np.zeros((0, 11))
    pa = ember_pos(E, idx, t - shutter * 0.5)
    pb = ember_pos(E, idx, t + shutter * 0.5)
    ok = (pb[:, 2] < cam_z - 0.35) & (pa[:, 2] < cam_z - 0.35)
    idx, pa, pb = idx[ok], pa[ok], pb[ok]
    u = (t - E['t0'][idx]) / E['life'][idx]
    fade = (1 - u) ** 1.4 * np.clip(u * 12, 0, 1)
    fl = 0.7 + 0.3 * np.sin(t * 0.9 + E['ph'][idx] * 5)
    b = 26.0 * E['b0'][idx] * fade * fl
    # colour: hot gold -> deep amber as they cool
    hue = np.clip(E['hue'][idx] * 0.5 + u * 0.7, 0, 1)[:, None]
    col = (1 - hue) * (0.5 * GOLD + 0.5 * PALE) + hue * (0.6 * AMBER + 0.4 * FIRE_MID)
    fl2 = SC.smooth(SC.ramp(t, SC.FLARE_T0, SC.FLARE_T1))
    col = col * (1 - fl2 * 0.5) + fl2 * 0.5 * np.array([1.0, 0.9, 0.75])
    out = np.zeros((idx.size, 11))
    out[:, 0:3] = pa
    out[:, 3:6] = pb
    out[:, 6] = E['size'][idx]
    out[:, 7:10] = col * b[:, None]
    out[:, 10] = 1.0
    return out


# ------------------------------------------------------------ torch flames ---

def _flame_blobs(base, lean, t, seed, I, scale=1.0, gold=0.0):
    """Blobs of a small torch flame rising from base (3,), leaning along lean (3,) (unit-ish)."""
    B = []
    up = np.array([0.0, 0.0, 1.0])
    for j in range(9):
        u = j / 8.0
        h = (0.02 + 0.30 * u) * scale
        sway = 0.035 * u * scale * np.array([math.sin(t * 0.83 + seed + 3 * u), math.cos(t * 0.71 + 2 * seed + 2 * u), 0])
        p = base + up * h + lean * h * 1.1 + sway
        rad = (0.055 * (1 - 0.75 * u) + 0.012) * scale
        fl = 0.85 + 0.15 * math.sin(t * 2.9 + seed * 7 + j)
        heat = 1.0 - u
        c = (heat ** 1.5) * (1 - gold) * FIRE_CORE + (heat ** 0.5) * (1 - heat ** 1.5) * (1 - gold) * FIRE_HOT \
            + (1 - heat) * (1 - gold) * FIRE_MID * 0.8
        c = c + gold * (heat * PALE + (1 - heat) * GOLD)
        B.append([p[0], p[1], p[2], rad, *(c * I * fl * (1.0 - 0.55 * u)), 0.0])
    # halo
    B.append([base[0], base[1], base[2] + 0.12 * scale, 0.28 * scale, *(FIRE_HOT * I * 0.035), 0.0])
    return B


def torch_flames(t):
    B = []
    for i in range(SC.NFIG):
        lit = SC.torch_lit(i, t)
        if lit <= 0.002:
            continue
        tip, mid, tor = SC.torch_tip_world(i, t)
        rp = SC.ribbon_progress(i, t)
        # the flame leans toward the hearth as the ribbon starts
        radial = -tip[:2] / (np.linalg.norm(tip[:2]) + 1e-9)
        lean = np.array([radial[0], radial[1], 0.0]) * (0.4 + 1.6 * rp)
        B += _flame_blobs(tip, lean, t, 1.3 * i, 14.0 * lit, scale=1.0 - 0.3 * rp, gold=0.5 * rp)
    return B


def ribbons(t):
    """Flame ribbons streaming from each torch into the hearth (1986..2000)."""
    B = []
    for i in range(SC.NFIG):
        rp = SC.ribbon_progress(i, t)
        lit = SC.torch_lit(i, t)
        if rp <= 0.0 or lit <= 0.0:
            continue
        n = 34
        for j in range(n):
            u = (j + 0.5) / n * rp
            p = SC.ribbon_point(i, t, u)
            flow = 0.55 + 0.45 * math.sin(2 * math.pi * (u * 5.0 - t * 0.33) + i)
            tipfade = min(1.0, (rp - u) / 0.08 + 0.2)
            rad = 0.05 + 0.03 * math.sin(3 * u + t * 0.5 + i) ** 2
            g = min(1.0, u * 1.3)
            c = (1 - g) * (0.5 * FIRE_HOT + 0.5 * FIRE_CORE) + g * (0.6 * GOLD + 0.4 * PALE)
            I = 9.0 * lit * flow * tipfade
            B.append([p[0], p[1], p[2], rad, *(c * I), 0.0])
    # merge core at the hearth, building as ribbons arrive
    arrived = sum(SC.ribbon_progress(i, t) >= 0.98 for i in range(SC.NFIG))
    core = 0.0
    if t < SC.IGNITE + 3:
        core = (arrived / SC.NFIG) ** 2 * smooth(ramp(t, 1990, SC.IGNITE)) * (1 - smooth(ramp(t, SC.IGNITE, SC.IGNITE + 3)))
    if core > 0:
        p = SC.HEARTH_TARGET
        B.append([p[0], p[1], p[2], 0.10 + 0.20 * core, *((0.5 * GOLD + 0.5 * PALE) * 30 * core), 0.0])
        B.append([p[0], p[1], p[2], 0.45 + 0.4 * core, *(GOLD * 2.0 * core), 0.0])
    return B


def ignition_flash(t):
    """Big soft bloom blob at the moment of merging."""
    if t < SC.IGNITE or t > SC.IGNITE + 10:
        return []
    a = t - SC.IGNITE
    k = math.exp(-a / 2.5)
    p = np.array([0.0, 0.0, 1.3])
    return [[p[0], p[1], p[2], 0.35 + 0.25 * a, *(PALE * 40 * k), 0.0],
            [p[0], p[1], p[2], 1.4 + 0.3 * a, *(GOLD * 2.5 * k), 0.0]]
