"""ACCORD scene: timeline, camera, emissaries, stones, lights, per-frame parameters."""
import math
import os
import sys

import numpy as np
from scipy.interpolate import PchipInterpolator

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'lib'))
import look  # noqa: E402

import textmaps as tm  # noqa: E402
import geom as G  # noqa: E402
import shade as SH  # noqa: E402

W, H = look.W, look.H
FOV_H = 50.0
F_FULL = (W / 2) / math.tan(math.radians(FOV_H / 2))

F0, F1 = 1912, 2247          # delivered range (inclusive)
OATH_HIT = [2000, 2040, 2080, 2120]
WRITE_DUR = 17.0
ROLL_DUR = 20.0
IGNITE = 2000
TOG_T0, TOG_T1 = 2160, 2220
FLARE_T0, FLARE_T1 = 2224, 2240


def lin(h):
    return look.hexrgb(h)


GOLD = lin(look.PALETTE['accord_gold'])
PALE = lin(look.PALETTE['accord_pale'])
AMBER = lin(look.PALETTE['accord_amber'])
MOON = lin(look.PALETTE['moonlight'])
NIGHT_MID = lin(look.PALETTE['night_mid'])
FIRE_HOT = lin(look.PALETTE['fire_hot'])
FIRE_MID = lin(look.PALETTE['fire_mid'])
FIRE_CORE = lin(look.PALETTE['fire_core'])


def clamp01(x):
    return min(max(x, 0.0), 1.0)


def ramp(t, a, b):
    return clamp01((t - a) / (b - a))


def smoother(x):
    x = clamp01(x)
    return x * x * x * (x * (6 * x - 15) + 10)


def smooth(x):
    x = clamp01(x)
    return x * x * (3 - 2 * x)


def ease_out(x, p=3.0):
    x = clamp01(x)
    return 1 - (1 - x) ** p


# ================================================================= camera ===
_H_KEYS = [(1900, 700.0), (1912, 640.0), (1926, 575.0), (1940, 400.0), (1953, 205.0), (1964, 96.0),
           (1974, 44.0), (1983, 24.5), (1991, 17.2), (1998, 13.6), (2003, 11.2), (2008, 9.45),
           (2014, 8.75), (2040, 8.62), (2080, 8.52), (2120, 8.45), (2136, 8.5), (2149, 10.8),
           (2162, 13.3), (2190, 13.05), (2222, 12.8), (2234, 10.6), (2247, 7.6), (2260, 6.0)]
_logh = PchipInterpolator([k[0] for k in _H_KEYS], np.log([k[1] for k in _H_KEYS]))


def cam_h(t):
    return float(np.exp(_logh(t)))


def cam_psi(t):
    """World azimuth (deg) of the screen-up direction."""
    psi = 190.0 - 100.0 * smoother(ramp(t, 1912, 2000))
    for k in range(1, 4):
        psi -= 90.0 * smoother(ramp(t, OATH_HIT[k] - ROLL_DUR, OATH_HIT[k]))
    psi -= 22.0 * smooth(ramp(t, 2140, 2260))
    return psi


def cam_cy(t):
    """Principal point row (full-res px): hearth position on screen (lens shift)."""
    cy = 380.0
    cy += (505.0 - 380.0) * smoother(ramp(t, 1992, 2011))
    cy += (402.0 - 505.0) * smoother(ramp(t, 2138, 2163))
    return cy


def camera(t, scale=1.0):
    h = cam_h(t)
    psi = math.radians(cam_psi(t))
    U = np.array([math.cos(psi), math.sin(psi), 0.0])
    Fw = np.array([0.0, 0.0, -1.0])
    R = np.cross(Fw, U)
    C = np.array([0.0, 0.0, h])
    f = F_FULL * scale
    cx = (W / 2) * scale
    cy = cam_cy(t) * scale
    return np.concatenate([C, R, U, Fw, [f, cx, cy]]).astype(np.float64)


def project(cam, P):
    """World points (...,3) -> screen (...,2) and camera depth (...)."""
    C, R, U, Fw = cam[0:3], cam[3:6], cam[6:9], cam[9:12]
    f, cx, cy = cam[12], cam[13], cam[14]
    v = np.asarray(P, np.float64) - C
    zc = v @ Fw
    xs = cx + f * (v @ R) / zc
    ys = cy - f * (v @ U) / zc
    return np.stack([xs, ys], -1), zc


def exposure(t):
    e = 2.3
    e += (1.55 - 2.3) * smooth(ramp(t, 1975, 1998))
    e += (1.08 - 1.55) * smooth(ramp(t, 1999, 2012))
    return e


# ============================================================= emissaries ===
NFIG = 12
_rng = np.random.default_rng(20250925)
FIG_ANG = np.radians(90.0 + 15.0 + 30.0 * np.arange(NFIG)) + np.radians(_rng.uniform(-2.5, 2.5, NFIG))
FIG_R = 2.93 + _rng.uniform(-0.04, 0.05, NFIG)
FIG_HS = np.array([1.00, 0.95, 1.05, 0.97, 1.02, 0.93, 1.06, 0.99, 0.96, 1.03, 0.98, 1.01])
FIG_WS = np.array([1.00, 0.96, 1.07, 0.95, 1.02, 1.05, 1.00, 0.97, 1.04, 0.99, 0.94, 1.03])
FIG_TYPE = np.array([0, 1, 2, 0, 3, 0, 4, 1, 0, 2, 0, 1])
FIG_SIDE = np.array([-1, -1, 1, -1, -1, 1, -1, -1, -1, 1, -1, -1], np.float64)
_CLOTH = np.array([
    [0.035, 0.040, 0.075],   # indigo
    [0.085, 0.028, 0.026],   # oxblood
    [0.055, 0.058, 0.032],   # olive
    [0.042, 0.042, 0.045],   # charcoal
    [0.080, 0.052, 0.032],   # umber
    [0.045, 0.052, 0.064],   # slate
    [0.028, 0.058, 0.058],   # deep teal
    [0.060, 0.034, 0.055],   # plum
    [0.090, 0.068, 0.036],   # ochre-brown
    [0.066, 0.066, 0.064],   # grey wool
    [0.026, 0.026, 0.030],   # black
    [0.042, 0.052, 0.036],   # moss
], np.float64) * 1.25
# timing of the gesture (frames)
FIG_RAISE = 1977.0 + _rng.uniform(0, 7, NFIG)
FIG_EXT = 1984.0 + _rng.uniform(0, 6, NFIG)
HOLDOUT = 7
FIG_EXT[HOLDOUT] = 1991.5          # the last one hesitates, joins just in time
FIG_RAISE[HOLDOUT] = 1986.0
FIG_WITHDRAW = 2005.0 + _rng.uniform(0, 8, NFIG)
FIG_BREATH_PH = _rng.uniform(0, 2 * np.pi, NFIG)


def _two_bone(S, Hd, L1, L2, pole):
    d = Hd - S
    dist = np.linalg.norm(d)
    dist_c = min(max(dist, 0.08), L1 + L2 - 1e-3)
    u = d / max(dist, 1e-6)
    Hh = S + u * dist_c
    a = (L1 * L1 - L2 * L2 + dist_c * dist_c) / (2 * dist_c)
    hgt = math.sqrt(max(L1 * L1 - a * a, 0.0))
    p = pole - u * (pole @ u)
    p /= np.linalg.norm(p) + 1e-9
    E = S + u * a + p * hgt
    return E, Hh


def _lerp(a, b, t):
    return a + (b - a) * t


def fig_pose(i, t):
    """Local hand position, torch direction (unit), elbow; torch 'lit' amount."""
    side = FIG_SIDE[i]
    hs = FIG_HS[i]
    hand_rest = np.array([0.10, side * 0.31, 1.00 * hs])
    tor_rest = np.array([0.12, side * 0.04, 1.0])
    hand_up = np.array([0.30, side * 0.26, 1.28 * hs])
    tor_up = np.array([0.45, -side * 0.05, 1.0])
    hand_ext = np.array([0.64, side * 0.10, 1.27 * hs])
    tor_ext = np.array([1.0, -side * 0.10, 0.52])
    hand_wd = np.array([0.20, side * 0.29, 1.08 * hs])
    tor_wd = np.array([0.30, side * 0.05, 1.0])
    u_raise = smoother(ramp(t, FIG_RAISE[i], FIG_RAISE[i] + 9))
    u_ext = smoother(ramp(t, FIG_EXT[i], FIG_EXT[i] + 9))
    u_wd = smoother(ramp(t, FIG_WITHDRAW[i], FIG_WITHDRAW[i] + 16))
    hand = _lerp(hand_rest, hand_up, u_raise)
    tor = _lerp(tor_rest, tor_up, u_raise)
    hand = _lerp(hand, hand_ext, u_ext)
    tor = _lerp(tor, tor_ext, u_ext)
    hand = _lerp(hand, hand_wd, u_wd)
    tor = _lerp(tor, tor_wd, u_wd)
    # tiny hand tremor / life
    hand = hand + 0.004 * np.array([math.sin(t * 0.37 + i), math.sin(t * 0.29 + 2 * i), math.sin(t * 0.41 + 3 * i)])
    tor = tor / np.linalg.norm(tor)
    S = np.array([0.0, side * 0.215 * FIG_WS[i], 1.33 * hs])
    pole = np.array([-0.35, side * 1.0, -0.9])
    E, Hh = _two_bone(S, hand, 0.31, 0.30, pole)
    return Hh, tor, E


def torch_lit(i, t):
    """1 = burning torch; fades after the merge."""
    return 1.0 - smooth(ramp(t, IGNITE - 1, IGNITE + 5))


def fig_world(i, P):
    ang = FIG_ANG[i] + math.pi      # facing toward centre
    c, s = math.cos(ang), math.sin(ang)
    x0, y0 = FIG_R[i] * math.cos(FIG_ANG[i]), FIG_R[i] * math.sin(FIG_ANG[i])
    P = np.asarray(P, np.float64)
    wx = x0 + P[..., 0] * c - P[..., 1] * s
    wy = y0 + P[..., 0] * s + P[..., 1] * c
    return np.stack([wx, wy, P[..., 2]], -1)


def torch_tip_world(i, t):
    Hh, tor, E = fig_pose(i, t)
    return fig_world(i, Hh + 0.50 * tor), fig_world(i, Hh + 0.30 * tor), tor


def figures(t):
    Fa = np.zeros((NFIG, G.F_N), np.float64)
    for i in range(NFIG):
        ang = FIG_ANG[i] + math.pi
        x0, y0 = FIG_R[i] * math.cos(FIG_ANG[i]), FIG_R[i] * math.sin(FIG_ANG[i])
        Fa[i, G.F_X], Fa[i, G.F_Y] = x0, y0
        Fa[i, G.F_ANG] = ang
        Fa[i, G.F_C], Fa[i, G.F_S] = math.cos(ang), math.sin(ang)
        Fa[i, G.F_HS], Fa[i, G.F_WS] = FIG_HS[i], FIG_WS[i]
        Fa[i, G.F_TYPE] = FIG_TYPE[i]
        Fa[i, G.F_SEED] = 1.7 * i + 0.3
        Fa[i, G.F_R:G.F_B + 1] = _CLOTH[i]
        Fa[i, G.F_SIDE] = FIG_SIDE[i]
        Hh, tor, E = fig_pose(i, t)
        Fa[i, G.F_HX:G.F_HZ + 1] = Hh
        Fa[i, G.F_TX:G.F_TZ + 1] = tor
        Fa[i, G.F_EX:G.F_EZ + 1] = E
        Fa[i, G.F_BOW] = 0.0
        Fa[i, G.F_TLIT] = 0.9 * (1.0 - smooth(ramp(t, IGNITE + 4, IGNITE + 60)))
        Fa[i, G.F_BREATH] = 0.006 * math.sin(2 * math.pi * t / 96.0 + FIG_BREATH_PH[i])
        # AABB (world) from key local points
        hs, ws = FIG_HS[i], FIG_WS[i]
        pts = [np.array([sx, sy, sz]) for sx in (-0.45 * ws, 0.36 * ws) for sy in (-0.42 * ws, 0.42 * ws)
               for sz in (0.0, 1.78 * hs)]
        pts += [Hh + 0.55 * tor, Hh - 0.14 * tor, E, Hh]
        wp = fig_world(i, np.array(pts))
        m = 0.10
        Fa[i, G.F_BB + 0] = wp[:, 0].min() - m
        Fa[i, G.F_BB + 1] = wp[:, 0].max() + m
        Fa[i, G.F_BB + 2] = wp[:, 1].min() - m
        Fa[i, G.F_BB + 3] = wp[:, 1].max() + m
        Fa[i, G.F_BB + 4] = 0.0
        Fa[i, G.F_BB + 5] = wp[:, 2].max() + m
    return Fa


# ================================================================= stones ===
NSTONE = 19


def stones():
    rng = np.random.default_rng(77)
    S = np.zeros((NSTONE, G.S_N), np.float64)
    base = np.linspace(0, 2 * np.pi, NSTONE, endpoint=False) + 0.09
    for j in range(NSTONE):
        a = base[j] + rng.uniform(-0.05, 0.05)
        r = 10.0 + rng.uniform(-0.35, 0.35)
        x, y = r * math.cos(a), r * math.sin(a)
        yaw = a + math.pi / 2 + rng.uniform(-0.15, 0.15)
        hx = rng.uniform(0.42, 0.70)
        hy = rng.uniform(0.24, 0.38)
        Hh = rng.uniform(2.5, 4.1)
        if j == 5:
            Hh = 1.6    # a broken stump
        S[j, G.S_X], S[j, G.S_Y] = x, y
        S[j, G.S_C], S[j, G.S_S] = math.cos(yaw), math.sin(yaw)
        S[j, G.S_HX], S[j, G.S_HY], S[j, G.S_H] = hx, hy, Hh
        S[j, G.S_TAPER] = rng.uniform(0.1, 0.3)
        S[j, G.S_LX] = rng.uniform(-0.05, 0.05)
        S[j, G.S_LY] = rng.uniform(-0.06, 0.03)
        tnx, tny = rng.uniform(-0.35, 0.35), rng.uniform(-0.25, 0.25)
        S[j, G.S_TNX], S[j, G.S_TNY] = tnx, tny
        S[j, G.S_TOP] = Hh - rng.uniform(0.05, 0.2)
        S[j, G.S_SEED] = rng.uniform(0, 10)
        S[j, G.S_ALB] = rng.uniform(0.2, 0.3)
        c, s = math.cos(yaw), math.sin(yaw)
        ext = max(hx, hy) + 0.25 + 0.1 * Hh
        S[j, G.S_BB + 0] = x - ext - 0.1
        S[j, G.S_BB + 1] = x + ext + 0.1
        S[j, G.S_BB + 2] = y - ext - 0.1
        S[j, G.S_BB + 3] = y + ext + 0.1
        S[j, G.S_BB + 4] = 0.0
        S[j, G.S_BB + 5] = Hh + 0.15
    return S


# ================================================================ lights ===

def fire_scale(t):
    """Visual size of the hearth fire."""
    if t < IGNITE:
        return 0.0
    a = t - IGNITE
    burst = 1.0 + 0.35 * math.exp(-a / 3.0) * (1 - math.exp(-a / 0.8))
    grow = 0.55 + 0.45 * ease_out(ramp(t, IGNITE, IGNITE + 14), 2.0)
    fl = 1.0 + 0.6 * smooth(ramp(t, FLARE_T0, FLARE_T1)) + 0.4 * smooth(ramp(t, FLARE_T1, 2250))
    return burst * grow * fl


def hearth_intensity(t):
    """Scalar intensity of the hearth light (0 before ignition)."""
    if t < IGNITE - 3:
        return 0.0
    pre = 0.35 * smooth(ramp(t, IGNITE - 3, IGNITE))
    if t < IGNITE:
        return pre
    a = t - IGNITE
    base = 0.7 + 0.3 * ease_out(ramp(t, IGNITE, IGNITE + 12), 2.0)
    flash = 1.0 + 1.8 * math.exp(-a / 2.2)
    flick = 1.0 + 0.07 * math.sin(t * 1.37) + 0.05 * math.sin(t * 2.71 + 1.3) + 0.04 * math.sin(t * 0.61)
    flare = math.exp(math.log(55.0) * smooth(ramp(t, FLARE_T0, FLARE_T1)) ** 1.6)
    flare *= 1.0 + 0.5 * ramp(t, FLARE_T1, 2250)
    return base * flash * flick * flare


def torch_lights(t, Fa):
    """Point lights for emissary torches + ribbons. Returns TL (n,6)."""
    L = []
    for i in range(NFIG):
        lit = torch_lit(i, t)
        if lit <= 0.001:
            continue
        tip, _, tor = torch_tip_world(i, t)
        fl = 1.0 + 0.12 * math.sin(t * 1.9 + 3.1 * i) + 0.08 * math.sin(t * 3.3 + i)
        I = 0.95 * lit * fl
        L.append([tip[0], tip[1], tip[2] + 0.15, I * FIRE_HOT[0], I * FIRE_HOT[1], I * FIRE_HOT[2]])
        # ribbon: light travelling toward the hearth
        rp = ribbon_progress(i, t)
        if rp > 0.0:
            P = ribbon_point(i, t, 0.6 * rp)
            I2 = 0.8 * lit * rp
            L.append([P[0], P[1], P[2], I2 * GOLD[0], I2 * GOLD[1], I2 * GOLD[2]])
    if not L:
        return np.zeros((1, 6))
    return np.array(L, np.float64)


# ------------------------------------------------------------- ribbons ---

def ribbon_progress(i, t):
    """0..1 how far the flame ribbon of emissary i has reached toward the hearth."""
    t0 = FIG_EXT[i] + 5.0
    if i == HOLDOUT:
        return smoother(ramp(t, t0, IGNITE - 0.5))
    return smoother(ramp(t, t0, min(t0 + 9.0, IGNITE - 1.0)))


HEARTH_TARGET = np.array([0.0, 0.0, 1.22])


def ribbon_ctrl(i, t):
    tip, _, _ = torch_tip_world(i, t)
    P0 = tip + np.array([0, 0, 0.08])
    P2 = HEARTH_TARGET
    mid = 0.5 * (P0 + P2)
    radial = P0[:2] / (np.linalg.norm(P0[:2]) + 1e-9)
    tang = np.array([-radial[1], radial[0]])
    P1 = np.array([mid[0] + 0.32 * tang[0], mid[1] + 0.32 * tang[1], mid[2] + 0.30])
    return P0, P1, P2


def ribbon_point(i, t, u):
    P0, P1, P2 = ribbon_ctrl(i, t)
    return (1 - u) ** 2 * P0 + 2 * (1 - u) * u * P1 + u * u * P2


# ============================================================== params ===

def oath_progress(k, t):
    return (t - OATH_HIT[k]) / WRITE_DUR if t >= OATH_HIT[k] else -1.0


def params(t, scale=1.0):
    PR = np.zeros(SH.P_NPARAM, np.float64)
    PR[SH.P_T] = t
    fl = 0.04 * math.sin(t * 1.1) + 0.03 * math.sin(t * 2.3 + 1.0)
    PR[SH.P_LX] = 0.03 * math.sin(t * 0.9)
    PR[SH.P_LY] = 0.03 * math.cos(t * 0.7)
    PR[SH.P_LZ] = 2.25 + fl
    PR[SH.P_LRAD] = 0.30 * (1.0 + 0.3 * smooth(ramp(t, FLARE_T0, FLARE_T1)))
    PR[SH.P_LI] = 26.0 * hearth_intensity(t)
    lc = 0.62 * GOLD + 0.38 * PALE
    wh = smooth(ramp(t, FLARE_T0 + 4, FLARE_T1 + 4))
    lc = lc * (1 - wh) + np.array([1.0, 0.93, 0.82]) * wh
    PR[SH.P_LCR:SH.P_LCB + 1] = lc
    el, az = math.radians(40.0), math.radians(222.0)
    PR[SH.P_MDX], PR[SH.P_MDY], PR[SH.P_MDZ] = math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)
    PR[SH.P_MI] = 0.30
    PR[SH.P_MCR:SH.P_MCB + 1] = MOON
    PR[SH.P_SKI] = 1.8
    PR[SH.P_SKR:SH.P_SKB + 1] = NIGHT_MID
    PR[SH.P_WI] = 1.0
    PR[SH.P_WCR:SH.P_WCB + 1] = FIRE_HOT
    for k in range(4):
        PR[SH.P_OP + k] = oath_progress(k, t)
    PR[SH.P_TG] = smooth(ramp(t, TOG_T0, TOG_T1)) * 1.0 if t >= TOG_T0 else 0.0
    PR[SH.P_TGPHI] = math.radians(cam_psi(TOG_T0))
    PR[SH.P_ORNB] = smoother(ramp(t, 2138, 2158))
    PR[SH.P_ORNR] = 0.35 * smooth(ramp(t, IGNITE, IGNITE + 10)) + 3.0 * smooth(ramp(t, FLARE_T0 - 2, FLARE_T1))
    PR[SH.P_ORNT] = 0.8
    PR[SH.P_FL] = smooth(ramp(t, FLARE_T0, FLARE_T1))
    PR[SH.P_COAL] = smooth(ramp(t, IGNITE - 1, IGNITE + 4)) * (1 + 3 * PR[SH.P_FL])
    PR[SH.P_FALLP] = 1.45
    PR[SH.P_FALLD0] = 0.6
    PR[SH.P_EMO] = 1.55 * (1 + 2.0 * PR[SH.P_FL])
    PR[SH.P_EMT] = 1.25 * (1 + 2.0 * PR[SH.P_FL])
    PR[SH.P_EMHOT] = 7.0
    PR[SH.P_SHIM] = 0.16
    PR[SH.P_TEXR] = tm.TEX_R
    PR[SH.P_GNDI] = 0.55
    return PR


def oath_angles():
    a = tm.oath_line_angles()
    O = np.zeros((4, 4), np.float64)
    for k in range(4):
        O[k, 0], O[k, 1] = a[k][0]
        O[k, 2], O[k, 3] = a[k][1]
    return O
