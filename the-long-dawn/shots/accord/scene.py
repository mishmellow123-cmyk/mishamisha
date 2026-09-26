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
TOG_T0, TOG_T1 = 2160, 2220  # the outer ornament band fills with gold (clockwise sweep)
FLARE_T0, FLARE_T1 = 2224, 2240
INNER_T0, INNER_T1 = 2004, 2138   # variant B: the inner carved band kindles flame by flame

# ------------------------------------------------------------------ variant ---
# A = Allegory (the four oaths), B = Legend (wordless: ornament, one continuous turn),
# C = Tolkien (as A, plus the Ring lying in the hearth fire)
VARIANT = os.environ.get('ACCORD_VARIANT', 'A').upper()


def set_variant(v):
    global VARIANT
    VARIANT = v.upper()
    assert VARIANT in ('A', 'B', 'C'), VARIANT


def has_text():
    return VARIANT in ('A', 'C')


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
           (1974, 46.0), (1983, 27.0), (1991, 19.0), (1998, 15.6), (2002, 13.6), (2006, 11.0),
           (2011, 9.2), (2016, 8.6), (2040, 8.5), (2080, 8.42), (2120, 8.38), (2136, 8.45),
           (2149, 12.5), (2162, 17.8), (2190, 18.4), (2220, 18.0), (2233, 14.0), (2247, 9.0), (2260, 7.0)]
# variant B: the same descent, then a slightly higher, centred framing so the whole carved
# mandala turns in frame (no oath to read at the top)
_H_KEYS_B = [k for k in _H_KEYS if k[0] <= 1998] + [
    (2002, 14.0), (2007, 12.4), (2014, 11.2), (2024, 10.8), (2060, 10.55), (2100, 10.4), (2136, 10.45),
    (2149, 13.0), (2162, 17.8), (2190, 18.4), (2220, 18.0), (2233, 14.0), (2247, 9.0), (2260, 7.0)]
# A/C: the rise to the mandala starts 14 frames later than v1, so Oath IV holds full size
_H_KEYS = [(k[0] + 14, k[1]) if 2136 <= k[0] <= 2162 else k for k in _H_KEYS]
_logh = PchipInterpolator([k[0] for k in _H_KEYS], np.log([k[1] for k in _H_KEYS]))
_logh_B = PchipInterpolator([k[0] for k in _H_KEYS_B], np.log([k[1] for k in _H_KEYS_B]))


def cam_h(t):
    if VARIANT == 'B':
        return float(np.exp(_logh_B(t)))
    return float(np.exp(_logh(t)))


def _drift_rate(t):
    """Variant B: the slow drift of the turn (deg/frame) that carries on after the descent."""
    return 0.24 * smooth(ramp(t, 1972, 2006)) * (1.0 - 0.55 * smooth(ramp(t, 2196, 2250)))


_TG = np.arange(1880.0, 2300.0, 0.05)
_DRIFT = np.concatenate([[0.0], np.cumsum([_drift_rate(u) * 0.05 for u in _TG[:-1]])])


def cam_psi(t):
    """World azimuth (deg) of the screen-up direction."""
    psi = 190.0 - 100.0 * smoother(ramp(t, 1912, 2000))
    if VARIANT == 'B':
        # one slow, continuous, eased turn of the mandala (no stepped rolls)
        return psi - float(np.interp(t, _TG, _DRIFT))
    for k in range(1, 4):
        psi -= 90.0 * smoother(ramp(t, OATH_HIT[k] - ROLL_DUR, OATH_HIT[k]))
    psi -= 22.0 * smooth(ramp(t, 2140, 2260))
    return psi


def cam_cy(t):
    """Principal point row (full-res px): hearth position on screen (lens shift)."""
    cy = 340.0
    if VARIANT == 'B':
        cy += (414.0 - 340.0) * smoother(ramp(t, 1994, 2012))
        cy += (402.0 - 414.0) * smoother(ramp(t, 2138, 2163))
        return cy
    cy += (505.0 - 340.0) * smoother(ramp(t, 1994, 2012))
    cy += (402.0 - 505.0) * smoother(ramp(t, 2152, 2177))
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
    e = 5.0
    e += (1.7 - 5.0) * smooth(ramp(t, 1955, 1990))
    e += (1.0 - 1.7) * ease_out(ramp(t, 1998.5, 2004), 2.0)
    return e


# ============================================================= emissaries ===
# Twelve hooded emissaries: dark, muted cloth (near-black, deep earth, one oxblood, one indigo,
# one garnet veil), each silhouette different from above: cowls with liripipes, deep hoods,
# head-wraps, a bare head in a fur mantle, a veiled slighter figure, a pointed capuchin.
NFIG = 12
_rng = np.random.default_rng(20250925)
FIG_ANG = np.radians(90.0 + 15.0 + 30.0 * np.arange(NFIG)) + np.radians(_rng.uniform(-2.5, 2.5, NFIG))
FIG_R = 2.30 + _rng.uniform(-0.03, 0.04, NFIG)
#                  0     1     2     3     4     5     6     7     8     9    10    11
FIG_HS = np.array([1.02, 0.99, 1.00, 1.04, 0.91, 1.03, 0.97, 1.00, 0.96, 1.05, 0.93, 0.99])
FIG_WS = np.array([1.00, 1.04, 0.97, 1.16, 0.86, 0.98, 1.02, 1.00, 1.01, 0.99, 0.90, 1.06])
FIG_TYPE = np.array([0, 1, 2, 3, 4, 5, 1, 1, 2, 0, 4, 5])
FIG_SIDE = np.array([-1, -1, 1, -1, -1, 1, -1, -1, -1, 1, -1, -1], np.float64)
FIG_CAPE = np.array([0.40, 0.36, 0.30, 0.34, 0.33, 0.30, 0.44, 0.40, 0.32, 0.46, 0.34, 0.38])   # mantle drop
FIG_TRAIN = np.array([0.26, 0.18, 0.14, 0.20, 0.22, 0.16, 0.30, 0.20, 0.12, 0.28, 0.18, 0.22])
FIG_BOW = np.radians([8, 5, 3, 6, 11, 7, 4, 2, 6, 9, 10, 4])
FIG_SKIN = np.array([0.30, 0.55, 0.20, 0.45, 0.65, 0.35, 0.50, 0.25, 0.15, 0.60, 0.40, 0.70])
# all cloth in one near-black charcoal-to-umber range: the emissaries differ by weave, sheen and cut,
# not by colour
_CLOTH = np.array([
    [0.0140, 0.0132, 0.0126],   # 0 warm charcoal
    [0.0128, 0.0126, 0.0126],   # 1 charcoal
    [0.0165, 0.0138, 0.0112],   # 2 umber-black
    [0.0135, 0.0130, 0.0126],   # 3 charcoal
    [0.0118, 0.0112, 0.0108],   # 4 black
    [0.0172, 0.0140, 0.0110],   # 5 dark umber
    [0.0124, 0.0124, 0.0128],   # 6 cool charcoal
    [0.0120, 0.0115, 0.0110],   # 7 black
    [0.0150, 0.0132, 0.0115],   # 8 warm charcoal
    [0.0160, 0.0130, 0.0108],   # 9 umber-black
    [0.0116, 0.0114, 0.0116],   # 10 black
    [0.0145, 0.0138, 0.0130],   # 11 grey-black
], np.float64)
# mantles, hoods-thrown-back, wraps and veils: the same range, a step lighter or darker
_CLOTH2 = _CLOTH * np.array([1.18, 0.86, 1.22, 1.25, 0.90, 0.84, 1.15, 1.20, 1.28, 0.88, 1.12, 0.85])[:, None]
FIG_WEAVE = np.array([1.0, 1.6, 0.7, 1.3, 2.0, 0.8, 1.2, 1.5, 0.75, 1.1, 1.8, 0.9])
FIG_SHEENK = np.array([1.2, 0.6, 0.9, 1.4, 1.6, 0.7, 1.1, 0.5, 1.0, 1.3, 1.5, 0.8])
# timing of the gesture (frames)
FIG_RAISE = 1977.0 + _rng.uniform(0, 7, NFIG)
FIG_EXT = 1984.0 + _rng.uniform(0, 6, NFIG)
HOLDOUT = 7
FIG_EXT[HOLDOUT] = 1991.5          # the last one hesitates, joins just in time
FIG_RAISE[HOLDOUT] = 1986.0
FIG_WITHDRAW = 2001.0 + _rng.uniform(0, 4, NFIG)
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
    """Local hand position, torch direction (unit), elbow (z measured from the dais floor)."""
    side = FIG_SIDE[i]
    hs = FIG_HS[i]
    ws = FIG_WS[i]
    hand_rest = np.array([0.12, side * 0.30 * ws, 0.98 * hs])
    tor_rest = np.array([0.12, side * 0.04, 1.0])
    hand_up = np.array([0.30, side * 0.26, 1.24 * hs])
    tor_up = np.array([0.45, -side * 0.05, 1.0])
    hand_ext = np.array([0.60, side * 0.10, 1.16 * hs])
    tor_ext = np.array([1.0, -side * 0.10, 0.52])
    # after the merge the spent torch is held upright before them, like a candle
    hand_wd = np.array([0.215, side * 0.085 * ws, 1.00 * hs])
    tor_wd = np.array([0.10, -side * 0.03, 1.0])
    u_raise = smoother(ramp(t, FIG_RAISE[i], FIG_RAISE[i] + 9))
    u_ext = smoother(ramp(t, FIG_EXT[i], FIG_EXT[i] + 9))
    u_wd = smoother(ramp(t, FIG_WITHDRAW[i], FIG_WITHDRAW[i] + 16))
    hand = _lerp(hand_rest, hand_up, u_raise)
    tor = _lerp(tor_rest, tor_up, u_raise)
    hand = _lerp(hand, hand_ext, u_ext)
    tor = _lerp(tor, tor_ext, u_ext)
    # the torch is drawn back in an arc over the table edge
    arc = np.array([0.04, side * 0.08, 0.10]) * math.sin(math.pi * u_wd)
    hand = _lerp(hand, hand_wd, u_wd) + arc
    tor = _lerp(tor, tor_wd, u_wd)
    # tiny hand tremor / life
    hand = hand + 0.004 * np.array([math.sin(t * 0.37 + i), math.sin(t * 0.29 + 2 * i), math.sin(t * 0.41 + 3 * i)])
    tor = tor / np.linalg.norm(tor)
    S = np.array([0.0, side * 0.200 * ws, 1.295 * hs])
    pole = np.array([-0.35, side * 1.0, -0.9])
    E, Hh = _two_bone(S, hand, 0.31, 0.30, pole)
    return Hh, tor, E


def torch_lit(i, t):
    """1 = burning torch; fades after the merge."""
    return 1.0 - smooth(ramp(t, IGNITE - 1, IGNITE + 5))


def fig_world(i, P):
    """Local figure coordinates (z from the dais floor) -> world."""
    ang = FIG_ANG[i] + math.pi      # facing toward centre
    c, s = math.cos(ang), math.sin(ang)
    x0, y0 = FIG_R[i] * math.cos(FIG_ANG[i]), FIG_R[i] * math.sin(FIG_ANG[i])
    P = np.asarray(P, np.float64)
    wx = x0 + P[..., 0] * c - P[..., 1] * s
    wy = y0 + P[..., 0] * s + P[..., 1] * c
    return np.stack([wx, wy, P[..., 2] + G.DAIS_Z], -1)


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
        Fa[i, G.F_R2:G.F_B2 + 1] = _CLOTH2[i]
        Fa[i, G.F_CAPE] = FIG_CAPE[i] * FIG_HS[i]
        Fa[i, G.F_TRAIN] = FIG_TRAIN[i]
        Fa[i, G.F_SKIN] = FIG_SKIN[i]
        Fa[i, G.F_WEAVE] = FIG_WEAVE[i]
        Fa[i, G.F_SHEENK] = FIG_SHEENK[i]
        Fa[i, G.F_HAND] = smooth(ramp(t, FIG_WITHDRAW[i] + 6, FIG_WITHDRAW[i] + 20))
        Fa[i, G.F_SIDE] = FIG_SIDE[i]
        Hh, tor, E = fig_pose(i, t)
        Fa[i, G.F_HX:G.F_HZ + 1] = Hh
        Fa[i, G.F_TX:G.F_TZ + 1] = tor
        Fa[i, G.F_EX:G.F_EZ + 1] = E
        # heads bow a little further once the fire is lit; a slow, individual breath of motion
        Fa[i, G.F_BOW] = FIG_BOW[i] + math.radians(5.0) * smooth(ramp(t, FIG_WITHDRAW[i], FIG_WITHDRAW[i] + 30)) \
            + math.radians(1.2) * math.sin(2 * math.pi * t / 131.0 + 1.9 * FIG_BREATH_PH[i])
        Fa[i, G.F_TLIT] = 0.25 * (1.0 - smooth(ramp(t, IGNITE + 1, IGNITE + 16)))
        Fa[i, G.F_BREATH] = 0.006 * math.sin(2 * math.pi * t / 96.0 + FIG_BREATH_PH[i])
        # AABB (world) from generous local bounds + the arm/torch
        hs, ws = FIG_HS[i], FIG_WS[i]
        back = -(0.25 + FIG_TRAIN[i]) * ws - 0.06
        pts = [np.array([sx, sy, sz]) for sx in (back, 0.30 * ws) for sy in (-0.37 * ws, 0.37 * ws)
               for sz in (0.0, 1.97 * hs)]
        pts += [Hh + 0.55 * tor, Hh - 0.26 * tor, E, Hh]
        wp = fig_world(i, np.array(pts))
        m = 0.08
        Fa[i, G.F_BB + 0] = wp[:, 0].min() - m
        Fa[i, G.F_BB + 1] = wp[:, 0].max() + m
        Fa[i, G.F_BB + 2] = wp[:, 1].min() - m
        Fa[i, G.F_BB + 3] = wp[:, 1].max() + m
        Fa[i, G.F_BB + 4] = G.DAIS_Z - 0.02
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
        r = 7.6 + rng.uniform(-0.3, 0.3)
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


def flare_mult(t):
    """Multiplier of the hearth's light as it flares to white (1 before 2224)."""
    m = math.exp(math.log(55.0) * smooth(ramp(t, FLARE_T0, FLARE_T1)) ** 1.6)
    return m * (1.0 + 0.5 * ramp(t, FLARE_T1, 2250))


def hearth_intensity(t, with_flare=True):
    """Scalar intensity of the hearth light (0 before ignition)."""
    if t < IGNITE - 3:
        return 0.0
    pre = 0.35 * smooth(ramp(t, IGNITE - 3, IGNITE))
    if t < IGNITE:
        return pre
    a = t - IGNITE
    base = 0.7 + 0.3 * ease_out(ramp(t, IGNITE, IGNITE + 12), 2.0)
    flash = 1.0 + 0.9 * math.exp(-a / 2.0)
    # the light it throws on the room breathes (<= 1 rad/frame, <= ~3.5 %); the flames themselves dance faster
    flick = 1.0 + 0.016 * math.sin(t * 0.61) + 0.012 * math.sin(t * 0.93 + 1.3) + 0.007 * math.sin(t * 0.37 + 0.4)
    return base * flash * flick * (flare_mult(t) if with_flare else 1.0)


def torch_lights(t, Fa):
    """Point lights for emissary torches + ribbons. Returns TL (n,6)."""
    L = []
    for i in range(NFIG):
        lit = torch_lit(i, t)
        if lit <= 0.001:
            continue
        tip, _, tor = torch_tip_world(i, t)
        fl = 1.0 + 0.05 * math.sin(t * 0.9 + 3.1 * i) + 0.03 * math.sin(t * 0.67 + i)
        I = 0.6 * lit * fl
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


HEARTH_TARGET = np.array([0.0, 0.0, 1.0])


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
    fl = 0.035 * math.sin(t * 0.45) + 0.02 * math.sin(t * 0.83 + 1.0)
    PR[SH.P_LX] = 0.03 * math.sin(t * 0.41)
    PR[SH.P_LY] = 0.03 * math.cos(t * 0.33)
    flare = smooth(ramp(t, FLARE_T0, FLARE_T1))
    hi = hearth_intensity(t)
    PR[SH.P_L1Z] = G.TABLE_Z + 0.55 + 0.5 * fl
    PR[SH.P_L1RAD] = 0.28
    PR[SH.P_L1I] = 6.5 * hi
    PR[SH.P_L2Z] = 2.55 + fl + 0.6 * flare
    PR[SH.P_L2RAD] = 0.38 * (1.0 + 0.4 * flare)
    PR[SH.P_L2I] = 5.0 * hi
    PR[SH.P_L3Z] = 3.5 + 0.8 * flare
    PR[SH.P_L3I] = 9.0 * hi
    PR[SH.P_CROWD] = 2.0 * (1.0 - 0.75 * flare)
    lc = 0.45 * GOLD + 0.55 * PALE
    wh = smooth(ramp(t, FLARE_T0 + 4, FLARE_T1 + 4))
    lc = lc * (1 - wh) + np.array([1.0, 0.93, 0.82]) * wh
    PR[SH.P_LCR:SH.P_LCB + 1] = lc
    el, az = math.radians(40.0), math.radians(222.0)
    PR[SH.P_MDX], PR[SH.P_MDY], PR[SH.P_MDZ] = math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)
    PR[SH.P_MI] = 0.12
    PR[SH.P_MCR:SH.P_MCB + 1] = MOON
    PR[SH.P_SKI] = 1.2
    PR[SH.P_SKR:SH.P_SKB + 1] = NIGHT_MID
    PR[SH.P_WI] = 1.0
    PR[SH.P_WCR:SH.P_WCB + 1] = FIRE_HOT
    for k in range(4):
        PR[SH.P_OP + k] = oath_progress(k, t)
    PR[SH.P_TG] = smooth(ramp(t, TOG_T0, TOG_T1)) if t >= TOG_T0 else 0.0
    PR[SH.P_TGPHI] = math.radians(cam_psi(TOG_T0))
    PR[SH.P_ORNB] = smoother(ramp(t, 2138, 2158))
    PR[SH.P_ORNR] = 0.22 * smooth(ramp(t, IGNITE, IGNITE + 10)) + 4.0 * flare
    PR[SH.P_FL] = flare
    PR[SH.P_COAL] = smooth(ramp(t, IGNITE - 1, IGNITE + 4)) * (1 + 3 * flare)
    PR[SH.P_FALLP] = 2.0
    PR[SH.P_FALLD0] = 0.45
    PR[SH.P_GNDI] = 0.5
    PR[SH.P_GNDK] = 2.4
    PR[SH.P_EMO] = 3.2 * (1 + 1.5 * flare)
    PR[SH.P_EMT] = 0.95 * (1 + 1.6 * flare)
    PR[SH.P_EMHOT] = 10.0
    PR[SH.P_SHIM] = 0.16
    PR[SH.P_SHEEN] = 2.2
    PR[SH.P_TEXR] = tm.TEX_R
    # the emissaries stay dark, firelit silhouettes while the hearth flares to white
    PR[SH.P_FIGK] = (1.0 - 0.9 * flare) / flare_mult(t)
    PR[SH.P_FIGRIM] = 0.16 * hearth_intensity(t, False) * (1.0 - 0.5 * flare) + 0.22 * flare
    # variant switches
    PR[SH.P_VAR] = 0.0 if has_text() else 1.0
    if not has_text():
        for k in range(4):
            PR[SH.P_OP + k] = -1.0
        PR[SH.P_INNER] = ramp(t, INNER_T0, INNER_T1) if t >= INNER_T0 else -1.0
        # each station fire kindles as its emissary's torch reaches over it
        for k in range(NFIG):
            PR[SH.P_ST0 + k] = FIG_EXT[k] + 6.0
    else:
        PR[SH.P_INNER] = -1.0
    if VARIANT == 'C':
        PR[SH.P_RING] = 1.0
        # the letters kindle as the fire takes the Ring, then breathe slowly (a heartbeat, not a flicker)
        k = smooth(ramp(t, IGNITE + 1, IGNITE + 16))
        pulse = 0.78 + 0.22 * math.sin(2 * math.pi * (t - IGNITE) / 58.0 - 1.2)
        PR[SH.P_RGLOW] = k * pulse
    return PR


def oath_angles():
    a = tm.oath_line_angles()
    O = np.zeros((4, 4), np.float64)
    for k in range(4):
        O[k, 0], O[k, 1] = a[k][0]
        O[k, 2], O[k, 3] = a[k][1]
    return O
