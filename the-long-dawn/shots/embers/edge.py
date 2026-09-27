"""EMBERS v3, cut A: THE EDGE (A7), THE BRINK (A8), the dead valley's ash (A9), the ember (A10), TOWERS IN THE
LIGHT (A16) and THE FIRE, SEEN (A17), on A's own frames (see a3.py for the schedule and the earlier shots).

THE EDGE. On the first downbeat the ground inside the ring of forges cracks and falls away, plate by plate from the
centre out, into a crater whose floor is molten fire; the towers stand on its rim and lean in, the gilded ones
most. Each surge gilds the nearest (molten gold runs down their fire-facing faces, scene_b._gold_runs) while the
farthest go dark; from bar 26 the rim crumbles under the gilded ones. One steady orbit round the rim, contre-jour.
THE BRINK. The fire swells into a torn vortex that strips embers off the tower tops; the rim gives way (2440); a
gilded tower's crown (never a giant's) breaks off and falls into the pit (2480); the camera tips over the rim and
falls after it toward the fire (2520) until the frame is white (2640).
"""
import math

import numpy as np

import look
from core import Camera, smoothstep, smootherstep, lerp, catmull, rng, vnoise, ease_in, ease_in_out
import scene_b as B
import a3 as A

R_RIM = 15.2          # the lip's mean radius; it runs at the forges' feet (_set_rim), so they stand ON it
DEPTH = 30.0          # the bowl's bottom, below the ground (a deep pit: the fall at the brink is long)
LAKE = 4.5            # the molten lake fills the bowl's bottom this deep
BOWL_P = 2.3          # the wall's profile: steep under the lip, flattening into the lake
FLOOR_Y = B.GROUND - DEPTH + LAKE          # the lake's surface
S_LAKE = (LAKE / DEPTH) ** (1.0 / BOWL_P)  # the lake's radius / the lip's
GILD_W = {2: 1.0, 6: 1.0, 1: 0.85, 4: 0.9, 7: 0.8, 0: 0.15, 3: 0.2, 5: 0.1}   # the nearest (gilded) vs the farthest
FALL_TOWER = 7        # the gilded tower whose crown breaks off (never a giant's)
C_MOLT = np.array([1.0, 0.72, 0.3])
AIR_E = 300.0
CROWN_G = 0.042         # the crown's fall: it passes the lip ~2530 and meets the glare ~2542
LAYER_H = 3.2
WALL_RAMP = lambda d, u: 0.75 * (1.0 - np.exp(-np.maximum(d - 1.0, 0.0) / 6.0)) ** 1.5 + 0.25 * u ** 1.5
_RIM = {}             # the lip's anchors at the forges' feet (set by _set_rim when the crater is built)


def _set_rim(towers):
    """the lip runs just inside each forge's foot (its innermost point at ground level at full height), so every
    tower stands on a spur of the lip, and the rim is torn back into bays between them"""
    aa, rr, ww = [], [], []
    h_of = lambda i: towers.height(i, A.T_WHITE - 1)
    lean0 = towers.lean
    towers.lean = 0.0
    for i in range(8):
        h = h_of(i)
        p = np.concatenate([towers.G[i][k]['p'] for k in range(5) if len(towers.G[i][k]['p'])])
        y0 = towers.TW2.HMAX - h
        p = p[(p[:, 1] >= y0) & (p[:, 1] < y0 + 3.0)]
        w = towers.world(i, p.astype(np.float32), h)
        r = np.hypot(w[:, 0], w[:, 2])
        a = np.arctan2(w[:, 2], w[:, 0])
        da = (a - towers.ang[i] + np.pi) % (2 * np.pi) - np.pi
        aa.append(float(towers.ang[i]))
        rr.append(float(r.min()) - 0.45)
        ww.append(float(np.abs(da).max()))
    towers.lean = lean0
    _RIM.update(a=np.array(aa), r=np.array(rr), w=np.array(ww))


def _wrap(d):
    return (d + np.pi) % (2 * np.pi) - np.pi


def _near_foot(a):
    """1 at a forge's foot, 0 in the bays between them"""
    a = np.asarray(a, np.float64)
    d = _wrap(a[..., None] - _RIM['a'])
    return np.exp(-(d / (_RIM['w'] * 1.15)) ** 4).max(-1)


def rim_r(a):
    """the lip's radius at azimuth a: spurs at the forges' feet, torn bays between them (never a clean circle)"""
    a = np.asarray(a, np.float64)
    torn = (0.035 * np.sin(7 * a + 1.3) + 0.024 * np.sin(12 * a + 0.4) + 0.015 * np.sin(23 * a + 2.1)
            + 0.008 * np.sin(47 * a + 0.7) + 0.004 * np.sin(97 * a + 1.9))
    if not _RIM:
        return R_RIM * (1.0 + torn)
    d = _wrap(a[..., None] - _RIM['a'])
    w = np.exp(25.0 * (np.cos(d) - 1.0))
    base = (w * _RIM['r']).sum(-1) / np.maximum(w.sum(-1), 1e-12)
    near = _near_foot(a)
    bay = 2.4 * (1.0 - near) * (0.75 + 0.25 * np.sin(5 * a + 0.9))
    return (base + bay) * (1.0 + torn * (1.0 - 0.9 * near))


def crater_y(r, a=None):
    rr = R_RIM if a is None else rim_r(a)
    s = np.clip(np.asarray(r, np.float64) / rr, 0.0, 1.0)
    return np.maximum(B.GROUND - DEPTH * (1.0 - s ** BOWL_P), FLOOR_Y)


# ---------------------------------------------------------------- schedule (added to a3.A3Sched)

def _gild(self, i, t):
    if i >= 8 or t < A.T_EDGE - 5:
        return 0.0, (0.35 * self.race(t) if i >= 8 else 0.0)
    w = GILD_W[i]
    # the gold pours on every surge from the first downbeat (scene_b._gold_runs); the skin thickens as it goes on
    base = float(smoothstep(A.T_EDGE, A.T_EDGE + 40, t)) * (0.7 + 0.3 * float(smoothstep(A.T_EDGE, A.T_BRINK, t)))
    if t >= A.T_LIGHT:
        base *= 1.0 - 0.8 * float(smoothstep(A.T_STOP, A.T_HOLDS, t))        # the gilding cools
    g = (w * base * (1.0 + 0.35 * self.beat_pulse(t))) if w >= 0.5 else 0.0
    dark = (1.0 - w) * float(smoothstep(A.T_EDGE + 20, A.T_EDGE + 200, t)) if w < 0.5 else 0.0
    if t >= A.T_LIGHT:
        dark *= 1.0 - float(smoothstep(A.T_OPEN_OTHERS, A.T_HOLDS, t))
    return g, dark


def _tower_lean(self, t):
    if t >= A.T_LIGHT:
        base = 0.07 - 0.03 * float(smoothstep(A.T_STOP, A.T_HOLDS, t))
    else:
        base = 0.07 * float(smoothstep(A.T_EDGE, A.T_EDGE + 400, t)) + 0.02 * float(smoothstep(A.T_BRINK, A.T_WHITE, t))
    lean = np.zeros(18)
    lurch = float(smoothstep(A.T_RIM_GIVES, A.T_RIM_GIVES + 9, t)) if t < A.T_LIGHT else 0.0
    for i in range(8):
        lean[i] = base * (0.35 + 0.9 * GILD_W[i])
        if GILD_W[i] >= 0.5:
            lean[i] += 0.045 * lurch * GILD_W[i]          # the rim gives way under the gilded ones: they lurch in
    return lean


_CROWN = [None]


def _tower_post(self, towers, i, P, tq):
    """the falling crown: FALL_TOWER's top breaks off at 2480 and falls into the pit; after THE BRINK it is gone"""
    if i != FALL_TOWER or tq < A.T_CROWN:
        return P
    if _CROWN[0] is None:
        _CROWN[0] = FallingCrown(towers)
    fc = _CROWN[0]
    cut = fc.cut_y(tq)
    above = P[:, 1] > cut
    if not above.any():
        return P
    P = P.copy()
    if tq >= A.T_LIGHT:
        P[above] = np.array([1e5, 0.0, 1e5], P.dtype)
        return P
    R, pivot, off = fc.state(tq)
    Q = (P[above] - pivot) @ R.T + pivot + off
    Q[Q[:, 1] < FLOOR_Y - 0.5] = np.array([1e5, 0.0, 1e5])       # swallowed by the lake
    P[above] = Q.astype(P.dtype)
    return P


def _tower_post_n(self, towers, i, N, P, tq):
    """the falling crown's normals turn with it (its lit face follows the fire as it tumbles)"""
    if i != FALL_TOWER or tq < A.T_CROWN or tq >= A.T_LIGHT or _CROWN[0] is None:
        return N
    fc = _CROWN[0]
    above = P[:, 1] > fc.cut_y(tq)
    if not above.any():
        return N
    R, _, _ = fc.state(tq)
    N = N.copy()
    N[above] = (N[above] @ R.T).astype(N.dtype)
    return N


def _shutter(self, i, t, pl):
    """window light: shuttered (dim) toward the fire through the race; at the turn the two giants open first, on the
    sides that face each other, then the smaller towers, one by one on the beat (A16)"""
    if t < A.T_LIGHT:
        return 0.04 if i in A.GIANTS else 0.3        # the giants' shutters are shut through the race
    if i in A.GIANTS:
        t0 = A.T_OPEN_GIANTS
    elif i < 8:
        order = [1, 7, 4, 0, 3, 5]
        t0 = A.T_OPEN_OTHERS + BEAT_F * order.index(i) if i in order else A.T_OPEN_OTHERS
    else:
        t0 = A.T_OPEN_OTHERS + BEAT_F * (6 + (i - 8) % 5)
    # the shutters open from the bottom up over half a beat
    y = pl[:, 1]
    k = smoothstep(0.0, 10.0, t - t0 - 0.06 * (y - y.min()))
    return 0.3 + 3.2 * k


def _back_light(self, t):
    """the towers' backs lit warm, for the first time, by the far ridge fires (A16)"""
    if t < A.T_LIGHT:
        return 0.0
    return 0.035 * float(smoothstep(A.T_LIGHT - 5, A.T_LIGHT + 30, t))


def _vortex_shape(self, vx, t, e, P0, P1, rr):
    """THE BRINK's vortex (red team: not a stock spiral): its outer disc lifts toward the tower tops while the throat
    stays on the fire; the eye sits off centre; one arm is much heavier; the ring at its heart tears and flickers"""
    lift = 20.0 * (1.0 - np.exp(-rr / 8.0))
    off = np.exp(-rr / 15.0)
    d = np.stack([3.5 * off, lift, -2.5 * off], 1)
    arm = np.round(vx.arm / (np.pi / 2)).astype(int) % 4
    w_arm = np.array([2.3, 0.65, 1.0, 0.55])[arm]
    ang = np.arctan2(P1[:, 2], P1[:, 0])
    tear = 0.5 + 0.5 * np.sin(7.0 * ang + 0.37 * t) * np.sin(3.0 * ang - 0.23 * t + 1.1)
    heart = np.exp(-rr / 10.0)
    e = e * w_arm * (1.0 - heart + heart * (0.25 + 1.5 * tear ** 2))
    return e, P0 + d, P1 + d


CRATER_ID = 100


def _extra_occluders(self, towers, ctx):
    """THE EDGE: the crater's wall, the plain round it and the ground plates (until they fall) join the occluder,
    so the near wall hides the far one and the ground hides what lies below it (splats are otherwise additive)"""
    t = ctx.t
    if t < A.T_EDGE - 2 or t >= A.T_WHITE:
        return None
    cr = ctx.tl_crater if hasattr(ctx, 'tl_crater') else None
    if cr is None:
        return None
    P, N, Ar = cr.occluders(t)
    return P.astype(np.float64), N.astype(np.float64), Ar.astype(np.float64), np.full(len(P), CRATER_ID, np.int32)


def _dust_k(self, t):
    k = float(smoothstep(A.T_TOWERS, A.T_TOWERS + 80, t))
    if A.T_EDGE <= t < A.T_WHITE:
        k *= 1.0 - 0.7 * float(smoothstep(A.T_EDGE, A.T_EDGE + 30, t))
    return k


def _near_fade(self, ctx, P):
    """(THE EDGE on) embers and sparks close to the lens fade: zref makes them big bright streaks there"""
    if ctx.t < A.T_EDGE:
        return 1.0
    z = (np.asarray(P, np.float64) - ctx.cam.pos) @ ctx.cam.R[2]
    return np.clip(z / 16.0, 0.0, 1.0) ** 2.5


_orig_height = B.Towers.height


def _height(self, i, t):
    if i == FALL_TOWER and A.T_CROWN <= t < A.T_LIGHT:
        t = float(A.T_CROWN)
    return _orig_height(self, i, t)


B.Towers.height = _height
BEAT_F = 20.0
A.A3Sched.near_fade = _near_fade
A.A3Sched.dust_k = _dust_k
A.A3Sched.extra_occluders = _extra_occluders
A.A3Sched.vortex_shape = _vortex_shape
A.A3Sched.shutter = _shutter
A.A3Sched.back_light = _back_light
A.A3Sched.gild = _gild
A.A3Sched.tower_lean = _tower_lean
A.A3Sched.tower_post = _tower_post
A.A3Sched.tower_post_n = _tower_post_n
A.A3Sched.gold_overlay = True


# ---------------------------------------------------------------- the crater

class Crater:
    """THE EDGE's crater of fire: a deep bowl whose lip runs at the forges' feet. The ground inside the ring cracks
    and falls away plate by plate (1840); the wall is ember rock, charcoal under the lip and hotter with depth, torn
    by seams of fire; a lake of molten light churns at the bottom (dark rafts on white gold); heat and embers rise
    out of it. The plain round it is dark crust lit by the fire above, fractured near the lip, and from bar 26 the
    lip crumbles under the gilded towers: chunks of crust break off and tumble into the glow (Chunks)."""

    def __init__(self, tl=None, seed=909):
        if tl is not None and not _RIM:
            _set_rim(tl.towers)
        r = rng(seed)
        # the ground inside the ring: plates that crack and fall away from the centre out
        n = 70000
        a = r.uniform(0, 2 * np.pi, n)
        rr = rim_r(a) * np.sqrt(r.random(n))
        self.g_p = np.stack([rr * np.cos(a), np.full(n, B.GROUND), rr * np.sin(a)], 1)
        ns = 52
        sa = r.uniform(0, 2 * np.pi, ns)
        sr = rim_r(sa) * np.sqrt(r.random(ns))
        seeds = np.stack([sr * np.cos(sa), sr * np.sin(sa)], 1)
        d = ((self.g_p[:, None, [0, 2]] - seeds[None, :, :]) ** 2).sum(-1)
        o = np.argsort(d, 1)
        self.g_cell = o[:, 0]
        d1 = np.sqrt(np.take_along_axis(d, o[:, :1], 1)[:, 0])
        d2 = np.sqrt(np.take_along_axis(d, o[:, 1:2], 1)[:, 0])
        self.g_seam = np.exp(-((d2 - d1) / 0.35) ** 2)                 # the cracks between plates
        self.c_tf = A.T_EDGE + 3.0 + 30.0 * (sr / R_RIM) + r.uniform(0, 6, ns)
        self.c_ax = r.normal(0, 1, (ns, 3))
        self.c_ax /= np.linalg.norm(self.c_ax, axis=1, keepdims=True)
        self.c_spin = r.uniform(0.02, 0.06, ns)
        self.c_ctr = np.stack([seeds[:, 0], np.full(ns, B.GROUND), seeds[:, 1]], 1)
        self.g_E = r.lognormal(0, 0.4, n)
        # the wall, uniform over its surface (importance-sampled along the profile): ember rock, lit from below
        m = 380000
        sg = np.linspace(S_LAKE * 0.97, 1.0, 400)
        dens = sg * np.sqrt(1.0 + (DEPTH * BOWL_P * sg ** (BOWL_P - 1.0) / R_RIM) ** 2)
        cdf = np.concatenate([[0.0], np.cumsum(0.5 * (dens[1:] + dens[:-1]) * np.diff(sg))])
        s = np.interp(r.random(m) * cdf[-1], cdf, sg)
        wa = r.uniform(0, 2 * np.pi, m)
        R = rim_r(wa)
        wr = s * R
        wy = B.GROUND - DEPTH * (1.0 - s ** BOWL_P)
        # knuckles of rock: the wall is not a smooth funnel
        lump = vnoise(np.stack([wr * np.cos(wa), wy, wr * np.sin(wa)], 1) * 0.22, 1.0, (3.1, 0.7, 5.2), 3)[:, 0]
        wr = wr * (1.0 + 0.05 * lump)
        self.w_p = np.stack([wr * np.cos(wa), wy + 0.12 * r.normal(0, 1, m), wr * np.sin(wa)], 1)
        slope = DEPTH * BOWL_P * s ** (BOWL_P - 1.0) / R
        self.w_n = np.stack([-slope * np.cos(wa), np.ones(m), -slope * np.sin(wa)], 1)
        self.w_n /= np.linalg.norm(self.w_n, axis=1, keepdims=True)
        self.w_u = np.clip((B.GROUND - wy) / (B.GROUND - FLOOR_Y), 0.0, 1.0)      # 0 at the lip, 1 at the lake
        cr1 = vnoise(self.w_p * np.array([0.42, 0.75, 0.42]), 1.0, (1.3, 2.2, 0.4), 2)[:, 0]
        cr2 = vnoise(self.w_p * np.array([1.1, 1.6, 1.1]), 1.0, (8.3, 1.2, 4.4), 1)[:, 0]
        self.w_seam = np.exp(-(cr1 / 0.06) ** 2) + 0.45 * np.exp(-(cr2 / 0.05) ** 2)
        self.w_plate = smoothstep(0.02, 0.2, np.abs(cr1))            # the crust plates between the fissures
        # molten veins running down the wall into the lake (in perspective they converge on the bottom: a bowl)
        arc = wa * R_RIM
        vv = vnoise(np.stack([arc * 0.2, wy * 0.035, np.zeros(m)], 1), 1.0, (4.4, 9.1, 0.0), 2)[:, 0]
        self.w_vein = np.exp(-(vv / 0.03) ** 2) * smoothstep(3.0, 9.0, B.GROUND - wy)
        # ledges: the wall steps down in broken terraces, each lip lit from the lake below
        led = (wy + 1.3 * lump) / 3.1
        self.w_ledge = np.exp(-(((led % 1.0) - 0.15) / 0.07) ** 2)
        # strata: the wall steps down in broken terraces that follow the bowl round (arcs in perspective); each step
        # is a dark face over a bright seam where the lake's light catches the underside of the ledge below
        dpt0 = B.GROUND - wy
        lay = (dpt0 + 1.5 * lump + 0.7 * np.sin(3.0 * wa + 1.1) + 0.35 * np.sin(8.0 * wa + 0.3)) / LAYER_H
        fr_ = lay % 1.0
        self.w_strata = 0.12 + 0.45 * smoothstep(0.1, 0.8, fr_) + 1.7 * np.exp(-((fr_ - 0.9) / 0.055) ** 2)
        self.w_E = r.lognormal(0, 0.22, m)
        self.w_a = wa
        self.w_s = s
        # the lake: molten, churning; dark rafts of crust drift on white gold
        k = 90000
        fa = r.uniform(0, 2 * np.pi, k)
        fr_ = rim_r(fa) * S_LAKE * 1.03 * np.sqrt(r.random(k))
        self.f_p = np.stack([fr_ * np.cos(fa), np.full(k, FLOOR_Y) + r.normal(0, 0.08, k), fr_ * np.sin(fa)], 1)
        self.f_E = r.lognormal(0, 0.35, k)
        self.f_r = fr_ / (rim_r(fa) * S_LAKE)
        # the plain the forges stand on: dark crust lit by the fire above; near the lip it is fractured into slabs
        # whose seams glow (a physical edge, not a line)
        g2 = 190000
        pa = r.uniform(0, 2 * np.pi, g2)
        rp = rim_r(pa)
        pr = rp + 70.0 * r.random(g2) ** 2.2
        self.p_p = np.stack([pr * np.cos(pa), np.full(g2, B.GROUND) + r.normal(0, 0.04, g2), pr * np.sin(pa)], 1)
        self.p_d = pr - rp                                              # distance back from the lip
        cr = vnoise(self.p_p * np.array([0.3, 0.0, 0.3]), 1.0, (7.1, 0.0, 2.3), 2)[:, 0]
        self.p_crack = np.exp(-(cr / 0.05) ** 2)
        # slabs along the lip: cells elongated along the rim, 1-2.5 units
        u_ = pa * rp / 1.6
        v_ = self.p_d / 1.1
        cell = vnoise(np.stack([u_, v_ * 1.7, np.zeros(g2)], 1), 1.0, (2.9, 6.1, 0.0), 1)[:, 0]
        self.p_slab = np.exp(-(cell / 0.07) ** 2) * np.exp(-self.p_d / 4.5)
        self.p_E = r.lognormal(0, 0.35, g2)
        # the plain's occluder: uniform in area out to r 95
        go = 110000
        oa = r.uniform(0, 2 * np.pi, go)
        orr = np.sqrt(r.uniform(0, 1, go)) * 95.0
        keep = orr > rim_r(oa)
        self.o_p = np.stack([orr * np.cos(oa), np.full(go, B.GROUND), orr * np.sin(oa)], 1)[keep]
        self.o_A = np.pi * 95.0 ** 2 / go
        # fine debris off the rim under the gilded towers (from bar 26), and a burst at the collapse (2440)
        ndb = 9000
        self.d_tw = r.choice([i for i in range(8) if GILD_W[i] >= 0.5], ndb)
        self.d_t0 = np.where(r.random(ndb) < 0.55, r.uniform(A.T_CRUMBLE, A.T_WHITE, ndb),
                             A.T_RIM_GIVES + r.exponential(14.0, ndb))
        self.d_t0 = np.where((self.d_t0 > A.T_WHITE), A.T_WHITE - r.uniform(1, 30, ndb), self.d_t0)
        self.d_off = r.normal(0, 1, (ndb, 3)) * np.array([1.6, 0.3, 1.6])
        self.d_v = r.normal(0, 1, (ndb, 3)) * np.array([0.05, 0.03, 0.05]) + np.array([0.0, 0.05, 0.0])
        self.d_E = r.lognormal(0, 0.7, ndb)
        self.d_hot = r.random(ndb)
        self.chunks = Chunks(tl)
        print('crater: ground', n, 'wall', m, 'lake', k, 'plain', g2)

    def _ground(self, t):
        P = self.g_p.copy()
        tf = self.c_tf[self.g_cell]
        x = np.maximum(t - tf, 0.0)
        fall = 0.5 * 0.085 * x ** 2
        # each plate tumbles a little about its own axis as it drops
        ang = self.c_spin[self.g_cell] * x
        rel = P - self.c_ctr[self.g_cell]
        ax = self.c_ax[self.g_cell]
        c, s = np.cos(ang)[:, None], np.sin(ang)[:, None]
        rel = rel * c + np.cross(ax, rel) * s + ax * (ax * rel).sum(1, keepdims=True) * (1 - c)
        P = self.c_ctr[self.g_cell] + rel
        P[:, 1] -= fall
        return P, x, fall

    def fire_light(self, P, t):
        """the thinking fire's light on the ground (it hangs over the pit): cosine over distance squared"""
        F = B.crown_centre(t)
        d = F[None, :] - P
        dist = np.linalg.norm(d, axis=1)
        return np.clip(d[:, 1] / dist, 0, 1) / (1.0 + (dist / 14.0) ** 2)

    def emit(self, ctx, lp):
        t = ctx.t
        if t < A.T_EDGE - 2 or (A.T_WHITE <= t < A.T_LIGHT):
            return
        red = B.redness(t)
        open_k = float(smootherstep(A.T_EDGE + 4, A.T_EDGE + 46, t))
        heat = 1.0 + 0.45 * B.beat_pulse(t)
        brink = float(smoothstep(A.T_BRINK, A.T_WHITE, t))
        calm = float(smoothstep(A.T_STOP, A.T_HOLDS, t)) if t >= A.T_LIGHT else 0.0
        # the ground plates: cracks glow before they part, then the plates drop into the glow
        P0, _, _ = self._ground(ctx.t0)
        P1, x, fall = self._ground(ctx.t1)
        pre = smoothstep(A.T_EDGE - 2, A.T_EDGE + 10, t) * (1 - smoothstep(0.0, 6.0, x))
        alive = fall < DEPTH - 2
        e = self.g_E * (0.05 + 1.6 * self.g_seam * pre + 0.25 * smoothstep(0.0, 8.0, x) * smoothstep(DEPTH, 4.0, fall))
        e = e * alive * heat
        col = look.blackbody(np.clip(0.4 + 0.25 * self.g_seam * pre + 0.15 * smoothstep(0, 10, fall), 0, 0.8))
        ctx.fr.splat(P0, P1, 0.05, e, col, ctx.cam0, ctx.cam1, zref=30.0, myid=CRATER_ID)
        # the plain round the crater: dark crust lit by the fire above; slabs along the lip with glowing seams,
        # fine cracks of fire further back
        fl = self.fire_light(self.p_p, t)
        near = np.exp(-self.p_d / 6.0)
        e = self.p_E * (0.004 + 0.4 * fl + (0.3 * self.p_slab + 0.15 * self.p_crack * near) * (0.35 + 0.65 * open_k)) * heat
        col = look.blackbody(np.clip(0.33 + 0.22 * self.p_slab + 0.08 * self.p_crack, 0, 0.8))
        col = col * (1 - 0.35 * red) + B.C_RED * 0.35 * red
        ctx.fr.splat(self.p_p, self.p_p, 0.13, e, col, ctx.cam0, ctx.cam1, zref=30.0, myid=CRATER_ID)
        if open_k <= 0:
            return
        # the wall: charcoal under the lip, hotter with depth, seams of fire, the terraces' lips lit from below
        u = self.w_u
        dpt = B.GROUND - self.w_p[:, 1]
        glow = WALL_RAMP(dpt, u)                        # charcoal under the lip, blazing a few metres down
        seam = self.w_seam
        e = self.w_E * (0.008 + 3.0 * glow * self.w_strata + seam * (0.02 + 1.2 * glow) + 5.0 * self.w_vein * (0.3 + u))
        e = e * open_k * heat * (1.0 + 0.6 * brink) * (1.0 - 0.5 * calm)
        col = look.blackbody(np.clip(0.3 + 0.42 * u ** 0.8 + 0.1 * seam * u, 0, 0.86))
        col = col * (1 - 0.35 * red) + (B.C_RED * 0.6 + B.C_CRIMSON * 0.4) * 0.35 * red
        # the fire, hanging just above the lip, lights the rock under the far lip from the front (cool gold over
        # the lake's orange: the bowl's shape reads)
        F = B.crown_centre(t)
        dv = F[None, :] - self.w_p
        dd = np.linalg.norm(dv, axis=1)
        lam = np.clip((self.w_n * dv).sum(1) / np.maximum(dd, 1e-6), 0, 1)
        fl = 0.9 * lam ** 1.5 / (1.0 + (dd / 12.0) ** 2) * (1.0 - u) * open_k
        col = (col * e[:, None] + look.blackbody(0.6)[None, :] * (fl * self.w_E)[:, None]) / np.maximum(e + fl * self.w_E, 1e-9)[:, None]
        e = e + fl * self.w_E
        ctx.fr.splat(self.w_p, self.w_p, 0.24, e * 1.8, col, ctx.cam0, ctx.cam1, zref=30.0, myid=CRATER_ID)
        # the lake: white gold under drifting dark rafts, churning (the glare the towers lean over)
        q = self.f_p * np.array([0.3, 0.0, 0.3])
        drift = np.array([0.012 * t, 0.0, -0.007 * t])
        w = vnoise(q + drift, 1.0, (0, 0, 0), 2)[:, 0]
        raft = smoothstep(0.08, 0.2, vnoise(self.f_p * np.array([0.55, 0.0, 0.55]) + 1.6 * drift, 1.0, (5.5, 0.0, 1.1), 1)[:, 0])
        edge = smoothstep(0.75, 1.03, self.f_r)
        e = self.f_E * (0.6 + 0.5 * w) * (1.0 - 0.9 * raft) * 11.0 * open_k * heat * (1 + 1.2 * brink) * (1.0 - 0.5 * calm)
        e = e * (1.0 - 0.5 * edge)
        col = look.blackbody(np.clip(0.66 + 0.12 * w - 0.2 * raft, 0, 0.92))
        ctx.fr.splat(self.f_p, self.f_p, 0.11, e, col, ctx.cam0, ctx.cam1, zref=30.0)
        # the air in the pit, lit by the lake: soft blobs rising and flickering, brightest low down; seen over the
        # near lip it is the glare the near rim and the towers stand against
        if not hasattr(self, 'air_p'):
            ra = rng(933)
            na = 1400
            aa = ra.uniform(0, 2 * np.pi, na)
            yy_ = FLOOR_Y + (B.GROUND + 14.0 - FLOOR_Y) * ra.random(na) ** 1.6
            rr_ = rim_r(aa) * np.sqrt(ra.random(na)) * np.clip(0.35 + 0.6 * (yy_ - FLOOR_Y) / (B.GROUND - FLOOR_Y), 0.3, 0.95)
            self.air_p = np.stack([rr_ * np.cos(aa), yy_, rr_ * np.sin(aa)], 1)
            self.air_r = ra.uniform(3.5, 8.0, na)
            self.air_ph = ra.uniform(0, 2 * np.pi, na)
            self.air_v = ra.uniform(0.03, 0.09, na)
        ay = self.air_p[:, 1] + 1.5 * np.sin(self.air_v * t + self.air_ph)
        AP = np.stack([self.air_p[:, 0], ay, self.air_p[:, 2]], 1)
        hu = np.clip((ay - FLOOR_Y) / (B.GROUND + 14.0 - FLOOR_Y), 0, 1)
        fl = 0.75 + 0.25 * np.sin(0.21 * t + self.air_ph)
        ea = AIR_E * np.exp(-3.0 * hu) * smoothstep(0.0, 0.05, hu) * fl * open_k * heat * (1 + 0.8 * brink) * (1 - 0.5 * calm)
        ca = look.blackbody(np.clip(0.62 - 0.14 * hu, 0, 0.9))
        ctx.fr.splat(AP, AP, self.air_r, ea, ca, ctx.cam0, ctx.cam1, profile=1, zref=30.0, rmax=900.0)
        # the pit's light rising out of it: the glowing air over the bowl (the glare the towers stand against)
        H = np.array([[0.0, FLOOR_Y + 3.0, 0.0]])
        ctx.fr.splat(H, H, np.array([11.0]), np.array([5.0e4 * open_k * heat * (1 + brink)]), look.blackbody(0.62)[None, :],
                     ctx.cam0, ctx.cam1, profile=1, rmax=1400.0)
        H2 = np.array([[0.0, B.GROUND + 3.0, 0.0]])
        ctx.fr.splat(H2, H2, np.array([15.0]), np.array([3.0e4 * open_k * heat * (1 + brink)]), look.blackbody(0.55)[None, :],
                     ctx.cam0, ctx.cam1, profile=1, rmax=1400.0)
        # fine debris off the crumbling rim; the chunks
        self._debris(ctx)
        self.chunks.emit(ctx)

    def occluders(self, t):
        """the wall, the plain and the ground plates (until they fall) for the occluder"""
        P, N, Ar = [], [], []
        k = max(1, len(self.w_p) // 170000)
        P.append(self.w_p[::k]); N.append(self.w_n[::k])
        area = 2.4 * np.pi * R_RIM ** 2
        Ar.append(np.full(len(self.w_p[::k]), area / len(self.w_p[::k])))
        P.append(self.o_p); N.append(np.broadcast_to(np.array([0.0, 1.0, 0.0]), (len(self.o_p), 3)))
        Ar.append(np.full(len(self.o_p), self.o_A))
        G, x, fall = self._ground(t)
        keep = fall < 1.0
        if keep.any():
            P.append(G[keep]); N.append(np.broadcast_to(np.array([0.0, 1.0, 0.0]), (int(keep.sum()), 3)))
            Ar.append(np.full(int(keep.sum()), np.pi * R_RIM ** 2 / len(G)))
        cP, cN, cA = self.chunks.occluders(t)
        if len(cP):
            P.append(cP); N.append(cN); Ar.append(cA)
        return np.concatenate(P), np.concatenate(N), np.concatenate(Ar)

    def _debris(self, ctx):
        t = ctx.t
        if t < A.T_CRUMBLE:
            return
        tw = ctx.tl.towers
        base = np.array([tw.base(i) for i in range(8)])
        ang = np.arctan2(base[:, 2], base[:, 0])[self.d_tw]
        src = np.stack([rim_r(ang) * np.cos(ang), np.full(len(ang), B.GROUND), rim_r(ang) * np.sin(ang)], 1) + self.d_off

        def pos(tq):
            a = np.maximum(tq - self.d_t0, 0.0)
            inward = -np.stack([np.cos(ang), np.zeros_like(ang), np.sin(ang)], 1) * (0.08 * a)[:, None]
            return src + self.d_v * a[:, None] + inward + np.array([0.0, -0.045, 0.0]) * (a * a)[:, None], a
        S0, _ = pos(ctx.t0)
        S1, a = pos(ctx.t1)
        on = (a > 0) & (S1[:, 1] > FLOOR_Y)
        if not on.any():
            return
        e = self.d_E * on * (0.4 + 2.2 * self.d_hot ** 3) * smoothstep(0.0, 2.0, a) * 1.6
        col = look.blackbody(np.clip(0.4 + 0.35 * self.d_hot, 0, 0.85))
        ctx.fr.splat(S0[on], S1[on], 0.05, e[on], col[on], ctx.cam0, ctx.cam1, zref=30.0)


class Chunks:
    """the rim crumbling under the gilded towers: slabs of crust break off the lip and tumble into the glow. From
    bar 26 one here and there on the beat; on bar 31 b3 (2440) the lip gives way under the gilded ones at once.
    Each chunk is a rigid block: a charcoal top, raw broken faces that glow, hottest underneath."""

    def __init__(self, tl, seed=931):
        r = rng(seed)
        gild = [i for i in range(8) if GILD_W[i] >= 0.5]
        angs = np.array([2 * np.pi * i / 8 + B.TOWER_ANG0 for i in range(8)])
        items = []
        # the slow crumble: on beats from bar 26, under the gilded towers
        for tb in np.arange(A.T_CRUMBLE, A.T_BRINK, A.BEAT):
            for i in gild:
                if r.random() < 0.35 + 0.4 * (tb - A.T_CRUMBLE) / (A.T_BRINK - A.T_CRUMBLE):
                    items.append((tb + r.uniform(0, 8), angs[i] + r.normal(0, 0.09), r.uniform(0.5, 1.1)))
        # the collapse: the lip gives way under the gilded towers
        for i in gild:
            for j in range(9):
                items.append((A.T_RIM_GIVES + r.exponential(5.0), angs[i] + r.normal(0, 0.12), r.uniform(0.8, 1.9)))
        # and in the brink's storm, more of it keeps going
        for j in range(40):
            i = gild[r.integers(len(gild))]
            items.append((r.uniform(A.T_RIM_GIVES + 10, A.T_WHITE - 10), angs[i] + r.normal(0, 0.14), r.uniform(0.5, 1.4)))
        items.sort()
        self.n = n = len(items)
        self.td = np.array([x[0] for x in items])
        self.a = np.array([x[1] for x in items])
        self.sz = np.array([x[2] for x in items])
        rr = rim_r(self.a)
        inw = -np.stack([np.cos(self.a), np.zeros(n), np.sin(self.a)], 1)
        self.inw = inw
        self.o = np.stack([rr * np.cos(self.a), np.full(n, B.GROUND), rr * np.sin(self.a)], 1) + inw * (0.45 * self.sz)[:, None] \
            + np.array([0.0, -0.35, 0.0]) * self.sz[:, None]
        self.v = inw * r.uniform(0.02, 0.07, n)[:, None] + np.array([0.0, 0.015, 0.0])
        self.ax = r.normal(0, 1, (n, 3))
        self.ax /= np.linalg.norm(self.ax, axis=1, keepdims=True)
        self.spin = r.uniform(0.02, 0.07, n) * np.where(r.random(n) < 0.5, -1, 1)
        # the block's points: an irregular slab (along the rim x, down y, inward z in its own frame)
        m = 260
        self.m = m
        L = np.stack([self.sz * r.uniform(1.1, 1.9, n), self.sz * r.uniform(0.5, 0.9, n), self.sz * r.uniform(0.8, 1.3, n)], 1)
        face = r.integers(0, 6, (n, m))
        q = r.uniform(-0.5, 0.5, (n, m, 3))
        ax_ = face // 2
        sgn = np.where(face % 2 == 0, -0.5, 0.5)
        np.put_along_axis(q, ax_[:, :, None], sgn[:, :, None], 2)
        q = q + r.normal(0, 0.04, (n, m, 3))
        self.q = q * L[:, None, :]
        self.top = (face == 3)                                  # +y: the old crust
        self.under = (face == 2)                                # -y: the hottest broken face
        self.qE = r.lognormal(0, 0.5, (n, m))
        # local frame: x along the rim, y up, z inward
        tan = np.stack([-np.sin(self.a), np.zeros(n), np.cos(self.a)], 1)
        self.F = np.stack([tan, np.tile([0.0, 1.0, 0.0], (n, 1)), inw], 1)      # rows: x, y, z axes in world

    def _pose(self, tq):
        x = np.maximum(tq - self.td, 0.0)
        ang = self.spin * x * (1.0 + 0.02 * x)
        c, s = np.cos(ang)[:, None, None], np.sin(ang)[:, None, None]
        ax = self.ax
        K = np.zeros((self.n, 3, 3))
        K[:, 0, 1], K[:, 0, 2], K[:, 1, 0] = -ax[:, 2], ax[:, 1], ax[:, 2]
        K[:, 1, 2], K[:, 2, 0], K[:, 2, 1] = -ax[:, 0], -ax[:, 1], ax[:, 0]
        R = np.eye(3)[None] + s * K + (1 - c) * (K @ K)
        pos = self.o + self.v * x[:, None] + np.array([0.0, -0.5 * 0.05, 0.0]) * (x * x)[:, None]
        return R, pos, x

    def world(self, tq):
        R, pos, x = self._pose(tq)
        W = np.einsum('nmk,nkj->nmj', self.q, self.F)            # block frame -> world axes
        W = np.einsum('nij,nmj->nmi', R, W) + pos[:, None, :]
        return W, x

    def occluders(self, t):
        if t < A.T_CRUMBLE:
            return np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(0)
        W, x = self.world(t)
        on = (x > 0) & (W[:, :, 1].mean(1) > FLOOR_Y)
        if not on.any():
            return np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(0)
        P = W[on].reshape(-1, 3)
        Nn = np.broadcast_to(np.array([0.0, 1.0, 0.0]), P.shape)
        Ar = np.repeat((self.sz[on] ** 2 * 5.0 / self.m), self.m)
        return P, Nn, Ar

    def emit(self, ctx):
        t = ctx.t
        if t < A.T_CRUMBLE:
            return
        W0, _ = self.world(ctx.t0)
        W1, x = self.world(ctx.t1)
        on = (x > 0) & (W1[:, :, 1].mean(1) > FLOOR_Y - 1.0)
        if not on.any():
            return
        hot = np.where(self.under, 1.0, np.where(self.top, 0.0, 0.55))[on]
        lit = 1.0 - smoothstep(0.0, 2.0, x[on])[:, None] * 0.0
        depth = np.clip((B.GROUND - W1[on][:, :, 1]) / (B.GROUND - FLOOR_Y), 0, 1)
        e = self.qE[on] * (0.02 + 1.7 * hot ** 1.5 * (0.6 + 1.4 * depth)) * lit
        T = 0.36 + 0.25 * hot + 0.15 * depth
        col = look.blackbody(np.clip(T, 0, 0.85).reshape(-1)).reshape(-1, 3)
        ctx.fr.splat(W0[on].reshape(-1, 3), W1[on].reshape(-1, 3), 0.07, e.reshape(-1), col, ctx.cam0, ctx.cam1,
                     zref=30.0, myid=CRATER_ID)


# ---------------------------------------------------------------- the falling crown (A8)

class FallingCrown:
    """a gilded tower's crown breaks off (2480) and falls into the pit as a rigid point cloud, trailing embers"""

    def __init__(self, towers):
        self.tw = towers

    def cut_y(self, t):
        return B.GROUND + self.tw.height(FALL_TOWER, A.T_CROWN) - 9.0

    def centre(self, t):
        """world centre of the crown (before the break: the top of its tower)"""
        tw = self.tw
        top = tw.top(FALL_TOWER, min(t, A.T_CROWN))
        c = top - np.array([0.0, 4.5, 0.0])
        if t < A.T_CROWN:
            return tw.top(FALL_TOWER, t) - np.array([0.0, 4.5, 0.0])
        R, pivot, off = self.state(t)
        return (c - pivot) @ R.T + pivot + off

    def state(self, t):
        """rigid transform applied to the crown's points (world): (R, pivot, offset). It turns about its own centre
        (tipping in toward the pit, then tumbling on) while the centre drifts in over the pit and drops"""
        x = max(t - A.T_CROWN, 0.0)
        tw = self.tw
        b = tw.base(FALL_TOWER)
        inward = -b / max(np.linalg.norm(b[[0, 2]]), 1e-6)
        inward[1] = 0.0
        c0 = tw.top(FALL_TOWER, A.T_CROWN) - np.array([0.0, 4.5, 0.0])
        ang = (0.0016 * x * x + 0.01 * x) if x < 20.0 else (0.84 + 0.074 * (x - 20.0))   # it tips over, then tumbles
        axis = np.cross(np.array([0.0, 1.0, 0.0]), inward)
        axis /= np.linalg.norm(axis)
        c, s = math.cos(ang), math.sin(ang)
        K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
        R = np.eye(3) + s * K + (1 - c) * (K @ K)
        off = inward * min(0.01 * x * x + 0.08 * x, 11.0) + np.array([0.0, -0.5 * CROWN_G * max(x - 3.0, 0.0) ** 2, 0.0])
        return R, c0, off


# ---------------------------------------------------------------- cameras

def lens_fall(cam, fall):
    """a shift lens: the frame drops by `fall` x its half-height while the camera stays level, so the towers stay
    upright (looking down with a wide lens splays them outward: they must read as leaning IN over the pit)"""
    import core
    k = -fall * (core.FULL_H / core.FULL_W) * math.tan(math.radians(cam.hfov) / 2.0)
    R = cam.R.copy()
    R[1] = R[1] - k * R[2]
    cam.R = R
    return cam


ORB_R, ORB_Y = 90.0, 33.0                # the whole stage: the ring of forges round the crater, against its glare
ORB_A0 = A.TH_G2 - 0.11                  # it starts behind the giant where A6 left it
ORB_W = 0.0022                           # rad / frame: one steady orbit (~87 degrees over THE EDGE)


def _orbit_az(t):
    return ORB_A0 + ORB_W * (t - A.T_EDGE)


EDGE_R, EDGE_Y, EDGE_TY, EDGE_HF, EDGE_FALL = 27.0, -8.0, -12.5, 88.0, 0.22
EDGE_AZ0, EDGE_AZ1 = 2.93, 3.25      # rel. ALPHA_C, over THE EDGE: the widest gap (forges 4 and 5 frame it), the
                                     # giants on either side of the fire across the pit
T_CUT_CROWN = A.T_CROWN - 20         # the cut to the crown on bar 31 b4: it stands for a beat, then breaks
T_CUT_FALL = A.T_TIP                 # the cut to the rim on bar 32 b3: the camera tips over after the crown


def _cam(pos, tgt, hf, fall, focus=None, ap=0.03):
    cam = Camera(pos, tgt, hfov=hf, focus=focus if focus else float(np.linalg.norm(tgt - pos)), aperture=ap)
    return lens_fall(cam, fall) if fall else cam


def _shake(t, k):
    """a heavy jolt, not a buzz (fast terms smear the whole frame inside the shutter)"""
    return k * np.array([math.sin(0.83 * t) + 0.35 * math.sin(1.9 * t + 1.0), 0.8 * math.sin(1.13 * t + 0.4),
                         math.cos(0.71 * t) + 0.3 * math.sin(1.7 * t)])


def _cam_edge(tl, t):
    """THE EDGE: one steady orbit just above the rim, outside the ring, looking across the crater through the widest
    gap: the near forges black against the pit's glare, the terraced bowl glowing in the middle band, the fire over
    it, the far towers on the far lip (gilded or dark), the near rim and ground dark in the lower third. A shift lens
    keeps the towers upright. THE BRINK: tilt up with the updraft (the towers converge over it), back down to the
    rim as it gives way (2440)."""
    u = (t - A.T_EDGE) / (A.T_BRINK - A.T_EDGE)
    az = B.ALPHA_C + EDGE_AZ0 + (EDGE_AZ1 - EDGE_AZ0) * u
    r, y, ty, hf, fall = EDGE_R, EDGE_Y, EDGE_TY, EDGE_HF, EDGE_FALL
    # it opens on the fire over the intact ground and follows the ground down as it falls away
    k0 = float(smootherstep(A.T_EDGE + 8, A.T_EDGE + 64, t))
    ty = lerp(1.0, ty, k0)
    fall = lerp(0.0, fall, k0)
    up = float(smootherstep(A.T_BRINK, A.T_BRINK + 18, t)) * (1.0 - float(smootherstep(A.T_RIM_GIVES - 2, A.T_RIM_GIVES + 12, t)))
    ty = ty + 30.0 * up
    fall = fall * (1.0 - up) - 0.15 * up
    hf = hf + 8.0 * up
    pos = np.array([r * math.cos(az), y, r * math.sin(az)])
    if t >= A.T_RIM_GIVES:
        pos = pos + _shake(t, 0.5 * math.exp(-(t - A.T_RIM_GIVES) / 16.0))
    return _cam(pos, np.array([0.0, ty, 0.0]), hf, fall, focus=r)


def _cam_crown(tl, t):
    """THE BRINK's hero (bar 31 b4 - bar 32 b3): a long lens from outside the ring, level with a gilded forge's
    crenellated crown (FALL_TOWER, never a giant), the crown black against the updraft's glare behind it. It
    breaks on bar 32 b1 (the seam of fire across the forge, CrownBreak) and tips over into the pit; the camera
    tilts down after it into the glare."""
    if _CROWN[0] is None:
        _CROWN[0] = FallingCrown(tl.towers)
    fc = _CROWN[0]
    tw = tl.towers
    a7 = float(tw.ang[FALL_TOWER])
    c0 = fc.centre(min(t, A.T_CROWN))
    cc = fc.centre(t)
    u = float(np.clip((t - T_CUT_CROWN) / (T_CUT_FALL - T_CUT_CROWN), 0, 1))
    r = 70.0 - 9.0 * float(smootherstep(0.0, 1.0, u))
    a = a7 + 0.05 + 0.03 * u
    y = c0[1] - 1.0 - 6.0 * float(smoothstep(0.35, 1.0, u))
    pos = np.array([r * math.cos(a), y, r * math.sin(a)])
    hf = 36.0 + 10.0 * float(smoothstep(0.3, 1.0, u))
    tgt = cc + np.array([0.0, 1.0, 0.0])
    if t >= A.T_CROWN:
        pos = pos + _shake(t, 0.12 * math.exp(-(t - A.T_CROWN) / 12.0))
    return _cam(pos, tgt, hf, 0.0, focus=float(np.linalg.norm(cc - pos)), ap=0.02)


def _cam_fall(tl, t):
    """OVER THE RIM (bar 32 b3 - bar 34): at the lip beside the broken forge, looking up at the crown as it drops
    past; the camera tips over the rim after it and falls toward the fire until the frame is white (2640)"""
    if _CROWN[0] is None:
        _CROWN[0] = FallingCrown(tl.towers)
    fc = _CROWN[0]
    tw = tl.towers
    _crater(tl)                                            # (the lip's shape: built before the first camera)
    ag = float(tw.ang[FALL_TOWER]) + 0.39                  # the gap between the broken forge and the next
    rl = float(rim_r(ag))
    p0 = np.array([(rl + 1.6) * math.cos(ag), B.GROUND + 2.6, (rl + 1.6) * math.sin(ag)])
    heart = np.array([0.0, FLOOR_Y + 5.0, 0.0])
    cc = fc.centre(t)
    # 2520-2546: at the lip, following the crown down past it; then the fall: in over the lip and down
    k = float(ease_in(np.clip((t - (A.T_TIP + 22)) / (A.T_WHITE - (A.T_TIP + 22)), 0, 1), 1.8))
    lean = float(smootherstep(A.T_TIP + 8, A.T_TIP + 30, t))
    inward = -p0 * np.array([1.0, 0.0, 1.0]) / max(np.hypot(p0[0], p0[2]), 1e-6)
    pos = p0 + inward * (2.2 * lean) + np.array([0.0, -1.0 * lean, 0.0])
    pos = pos + (heart + np.array([0.0, 9.0, 0.0]) - pos) * (0.9 * k)
    w_crown = 1.0 - float(smoothstep(A.T_TIP + 26, A.T_TIP + 44, t))
    tgt = lerp(heart, cc, w_crown)
    hf = 64.0 + 26.0 * k
    pos = pos + _shake(t, 0.08 + 0.12 * k)
    return _cam(pos, tgt, hf, 0.0, focus=float(np.linalg.norm(tgt - pos)), ap=0.02)


def camera(tl, t):
    if t < A.T_WHITE:
        if t < T_CUT_CROWN:
            return _cam_edge(tl, t)
        if t < T_CUT_FALL:
            return _cam_crown(tl, t)
        return _cam_fall(tl, t)
    if t < A.T_BLACK:
        # the dead valley: the promise's own view, drifting
        u = (t - A.T_WHITE) / (A.T_BLACK - A.T_WHITE)
        pos, tgt, hf = A.cam_a5a6(1400.0)
        pos = pos + np.array([0.0, -1.5 * u, 0.0])
        return Camera(pos, tgt, hfov=hf, focus=float(np.linalg.norm(pos)), aperture=0.02)
    if t < A.T_LIGHT:
        return Camera(np.zeros(3), np.array([0.0, 0.0, -10.0]), hfov=50.0, focus=6.0, aperture=0.03)
    return cam_light(t)


AL = B.ALPHA_C + 0.06


def cam_light(t):
    """A16: among the towers, looking out at the far ridge fires; then turned in to the giants opening toward each
    other past the fire. A17: a walking-pace descent toward the calm fire."""
    out = np.array([math.cos(AL), 0.0, math.sin(AL)])
    if t < A.T_SEEN:
        k = float(smootherstep(A.T_LIGHT + 55, A.T_OPEN_GIANTS + 5, t))
        p_out = out * 21.5 + np.array([0.0, -8.5, 0.0])
        t_out = out * 220.0 + np.array([0.0, 10.0, 0.0])
        p_in = out * 31.0 + np.array([0.0, 1.5, 0.0])
        t_in = np.array([0.0, 4.0, 0.0])
        drift = out * (-1.5 * float(smoothstep(A.T_OPEN_GIANTS, A.T_SEEN, t)))
        pos = lerp(p_out, p_in, k) + drift
        # turn round through the side, not through the tower beside the camera
        side = np.array([-out[2], 0.0, out[0]])
        pos = pos + side * (6.0 * math.sin(math.pi * k))
        tgt = lerp(t_out, t_in, k)
        hf = lerp(64.0, 60.0, k)
        return Camera(pos, tgt, hfov=hf, focus=float(np.linalg.norm(tgt - pos)) if k > 0.5 else 60.0,
                      aperture=0.06)
    u = float(np.clip((t - A.T_SEEN) / (A.T_END - A.T_SEEN), 0, 1))
    w = u * 0.92 + 0.08 * u * u
    p0 = out * 29.5 + np.array([0.0, 1.5, 0.0])
    p1 = out * 8.5 + np.array([0.0, -2.5, 0.0])
    pos = lerp(p0, p1, w) + np.array([0.0, 0.12 * math.sin(0.38 * t), 0.0])     # a walker's step
    tgt = np.array([0.0, 0.8, 0.0])
    return Camera(pos, tgt, hfov=54.0, focus=float(np.linalg.norm(tgt - pos)), aperture=0.07)


# ---------------------------------------------------------------- emit hooks

def _crater(tl):
    return tl._get('crater', lambda: Crater(tl))


def emit_before_towers(tl, ctx, lp, lc, lpw):
    ctx.tl = tl
    _crater(tl).emit(ctx, lp)


def emit_after(tl, ctx, lp, lc, lpw):
    t = ctx.t
    if A.T_EDGE <= t < A.T_WHITE or t >= A.T_LIGHT:
        tl._get('gold_runs', GoldRuns).emit(ctx, tl)
    if A.T_EDGE <= t < A.T_WHITE:
        k = float(smoothstep(A.T_EDGE + 10, A.T_EDGE + 60, t))
        tl._get('pit_embers', PitEmbers).emit(ctx, k * (1.0 + 1.5 * float(smoothstep(A.T_BRINK, A.T_WHITE, t))))
    elif t >= A.T_LIGHT:
        tl._get('pit_embers', PitEmbers).emit(ctx, 0.8 - 0.5 * float(smoothstep(A.T_STOP, A.T_HOLDS, t)))
    if A.T_BRINK <= t < A.T_WHITE:
        tl._get('updraft', Updraft).emit(ctx)
        tl._get('strip_embers', lambda: StripEmbers(tl)).emit(ctx)
    if A.T_CROWN <= t < A.T_WHITE and _CROWN[0] is not None:
        tl._get('crown_trail', CrownTrail).emit(ctx)
    if A.T_CROWN - 12 <= t < A.T_WHITE:
        if _CROWN[0] is None:
            _CROWN[0] = FallingCrown(tl.towers)
        tl._get('crown_break', CrownBreak).emit(ctx, tl)
    if t >= A.T_LIGHT:
        tl._get('ridge_fires', RidgeFires).emit(ctx)
        if t >= A.T_SEEN - 60:
            tl._get('small_lights', lambda: SmallLights(tl)).emit(ctx)


_SHIM_W = np.array([[0.021, 0.19, 0.61], [0.034, -0.12, 1.37], [-0.017, 0.27, 2.9], [0.046, 0.08, 4.1],
                    [-0.029, -0.21, 5.3], [0.011, 0.35, 0.2]])


def _shimmer(ctx, hdr, amp):
    """heat haze over the pit: the air above the bowl ripples, rising (screen-space refraction, masked to the
    bowl's opening and the column of hot air over it)"""
    H, W = hdr.shape[:2]
    cam = ctx.cam
    a = np.linspace(0, 2 * np.pi, 240, endpoint=False)
    R = rim_r(a)
    P = np.stack([R * np.cos(a), np.full(len(a), B.GROUND), R * np.sin(a)], 1)
    u, v, z = cam.project(P, W, H)
    if (z <= 0.5).any():
        return hdr
    import cv2
    m = np.zeros((H, W), np.float32)
    cv2.fillPoly(m, [np.stack([u, v], 1).astype(np.int32)], 1.0)
    acc = m.copy()
    step = max(1, int(0.05 * H))
    for kk in range(1, 9):
        sh = np.zeros_like(m)
        sh[:H - kk * step] = m[kk * step:]
        acc = np.maximum(acc, sh * (0.86 ** kk))
    acc = cv2.GaussianBlur(acc, (0, 0), 0.025 * W)
    if acc.max() <= 1e-3:
        return hdr
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    s = 1920.0 / W
    xs, ys = xx * s, yy * s
    t = ctx.t
    dx = np.zeros((H, W), np.float32)
    dy = np.zeros((H, W), np.float32)
    for kx, ky, ph in _SHIM_W:
        dx += np.sin(kx * xs + ky * ys + 0.31 * t + ph).astype(np.float32)
        dy += np.sin(0.8 * ky * xs - 1.3 * kx * ys + 0.37 * t + 1.7 * ph).astype(np.float32)
    k = (amp / s / len(_SHIM_W) ** 0.5) * acc
    return cv2.remap(hdr, xx + dx * k, yy + dy * k, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def post(tl, ctx, hdr):
    t = ctx.t
    if A.T_EDGE + 30 <= t < A.T_WHITE - 30 or A.T_LIGHT <= t < A.T_SEEN:
        amp = 2.2 + 2.0 * float(smoothstep(A.T_BRINK, A.T_BRINK + 40, t)) if t < A.T_WHITE else 1.2
        hdr = _shimmer(ctx, hdr, amp)
    if A.T_WHITE - 40 <= t < A.T_WHITE:
        # falling into the fire: the frame goes white on the downbeat (the IMPACT)
        k = float(smoothstep(A.T_WHITE - 40, A.T_WHITE - 1, t)) ** 1.6
        hdr = hdr * (1 - 0.3 * k) + np.array([6.0, 5.6, 5.0], np.float32) * (1.2 * k)
    if A.T_WHITE <= t < A.T_DEAD + 6:
        # out of the white, the grey vision
        k = 1.0 - float(smoothstep(A.T_WHITE, A.T_DEAD + 6, t))
        hdr = hdr + np.array([6.0, 5.8, 5.4], np.float32) * (1.2 * k ** 1.5)
    return hdr


class StripEmbers:
    """THE BRINK (H5: fire physics, not a vortex): the fire's updraft strips embers off the tower tops, drags them in
    toward the column and throws them up it"""

    def __init__(self, tl, seed=949):
        r = rng(seed)
        self.tl = tl
        n = 7000
        self.i = r.integers(0, 8, n)
        self.ph = r.random(n)
        self.life = r.uniform(30.0, 70.0, n)
        self.jit = r.normal(0, 1, (n, 3))
        self.E = r.lognormal(0, 0.6, n)
        self.sw = r.uniform(2.0, 3.6, n)

    def pts(self, t):
        tw = self.tl.towers
        age = ((t - A.T_BRINK) / self.life + self.ph) % 1.0 * self.life
        u = age / self.life
        top = np.array([tw.top(i, t) for i in range(8)])[self.i]
        a0 = np.arctan2(top[:, 2], top[:, 0])
        r0 = np.hypot(top[:, 0], top[:, 2])
        # in toward the column (a little twist, never a spiral), then up it, fast
        k_in = smoothstep(0.0, 0.55, u)
        rr = r0 + (Updraft.radius(top[:, 1] + 4.0) * (0.4 + 0.6 * np.abs(self.jit[:, 0]) / 2.5) - r0) * k_in
        a = a0 + 0.25 * (self.sw / 2.8) * k_in
        y = top[:, 1] + 4.0 * k_in + 46.0 * smoothstep(0.45, 1.0, u) ** 1.6
        P = np.stack([rr * np.cos(a), y, rr * np.sin(a)], 1) + self.jit * (0.5 + 0.8 * u)[:, None]
        return P, u, age

    def emit(self, ctx):
        t = ctx.t
        if t < A.T_BRINK or t >= A.T_WHITE:
            return
        g = float(smoothstep(A.T_BRINK, A.T_BRINK + 24, t)) * Updraft.strength(t) ** 0.5
        P0, _, _ = self.pts(ctx.t0)
        P1, u, age = self.pts(ctx.t1)
        ok = age > 0.6
        e = self.E * g * (1 - u) ** 0.6 * smoothstep(0.0, 0.08, u) * 7.0 * ok * (1.0 - 0.7 * float(smoothstep(T_CUT_CROWN - 4, T_CUT_CROWN, t)))
        col = look.blackbody(np.clip(0.5 + 0.25 * smoothstep(0.45, 0.8, u) - 0.15 * smoothstep(0.85, 1.0, u), 0, 0.9))
        ctx.fr.splat(P0[ok], P1[ok], 0.03, e[ok], col[ok], ctx.cam0, ctx.cam1, zref=30.0)



class PitEmbers:
    """embers boiling up out of the lake on the heat, swirling, thinning and reddening as they cool above the lip
    (THE EDGE); in the brink's updraft they are dragged in toward the column and up, faster"""

    def __init__(self, seed=961):
        r = rng(seed)
        n = 10000
        a = r.uniform(0, 2 * np.pi, n)
        s = np.sqrt(r.random(n)) * 1.05
        rl = rim_r(a) * S_LAKE * s
        self.src = np.stack([rl * np.cos(a), np.full(n, FLOOR_Y + 0.3), rl * np.sin(a)], 1)
        self.ph = r.random(n)
        self.life = r.uniform(80.0, 190.0, n)
        self.vy = r.uniform(0.1, 0.3, n)
        self.sw = r.uniform(0, 2 * np.pi, (n, 3))
        self.amp = r.uniform(0.6, 2.4, n)
        self.E = r.lognormal(0, 0.8, n)
        self.T = r.uniform(0.48, 0.74, n)

    def pos(self, tq):
        age = ((tq / self.life + self.ph) % 1.0) * self.life
        u = age / self.life
        brink = float(smoothstep(A.T_BRINK, A.T_BRINK + 30, tq)) if tq < A.T_WHITE else 0.0
        y = self.src[:, 1] + self.vy * age * (1.0 + 0.004 * age) * (1.0 + 1.6 * brink)
        sp = (0.3 + 0.7 * u) * self.amp
        x = self.src[:, 0] * (1.0 + 0.25 * u - 0.7 * brink * u) + sp * np.sin(0.05 * age + self.sw[:, 0])
        z = self.src[:, 2] * (1.0 + 0.25 * u - 0.7 * brink * u) + sp * np.sin(0.043 * age + self.sw[:, 1])
        return np.stack([x, y, z], 1), u, age

    def emit(self, ctx, gain=1.0):
        t = ctx.t
        P0, _, _ = self.pos(ctx.t0)
        P1, u, age = self.pos(ctx.t1)
        ok = age > 0.6
        cool = smoothstep(B.GROUND - 8.0, B.GROUND + 6.0, P1[:, 1])
        e = self.E * gain * (1 - u) ** 1.3 * smoothstep(0.0, 0.06, u) * (1.0 - 0.85 * cool) * 0.9 * ok
        col = look.blackbody(np.clip(self.T - 0.14 * cool, 0, 0.9))
        ctx.fr.splat(P0[ok], P1[ok], 0.02, e[ok], col[ok], ctx.cam0, ctx.cam1, zref=30.0)



class GoldRuns:
    """THE EDGE (director: the incentive has to be visible): on every surge liquid gold pours from the crowns of the
    gilded towers and runs down their fire-facing faces in rivulets, each with a white-hot bead at its head,
    staggered like drips; behind the heads the runs stay molten gold, and between pours they leave a gilt coat.
    Drawn as its own layer on the towers' crust points (crisp, not the crust's grain), after the towers."""
    C_GOLD = np.array([1.0, 0.78, 0.36])
    C_HOT = np.array([1.0, 0.94, 0.78])

    def emit(self, ctx, tl):
        import bisect
        tw = tl.towers
        if tw.cur is None:
            return
        t = ctx.t
        cam = ctx.cam
        cpos = cam.pos
        fwd = cam.R[2]
        fpx = cam.f_px(1920)
        F = B.crown_centre(t)
        beats = B.SCHED.beats
        j = bisect.bisect_right(beats, t) - 1
        age = (t - beats[j]) if j >= 0 else 99.0
        fresh = math.exp(-age / 14.0)
        for i in range(8):
            g, dark = B.SCHED.gild(i, t)
            c = tw.cur[i]
            if g <= 0 or c is None:
                continue
            pt = c['parts'][0]
            if len(pt['idx']) == 0:
                continue
            P, N, pl = pt['P'], pt['N'], pt['pl']
            Lv = F[None, :] - P
            fs = np.clip((N * Lv).sum(1) / np.maximum(np.linalg.norm(Lv, axis=1), 1e-6), 0, 1)
            V = cpos[None, :] - P
            dist = np.linalg.norm(V, axis=1)
            ndv = (N * V).sum(1) / np.maximum(dist, 1e-6)
            sel = (fs > 0.2) & (ndv > 0.03)
            if i in A.GIANTS:
                sel &= P[:, 1] < 12.0
            if not sel.any():
                continue
            P, N, pl, fs, ndv = P[sel], N[sel], pl[sel], fs[sel], ndv[sel]
            z0 = np.zeros(len(pl), pl.dtype)
            n1 = vnoise(np.stack([pl[:, 2] * 0.42, pl[:, 0] * 0.42, z0], 1), 1.0, (7.1, 0.0, 3.3), 2)[:, 0]
            riv = smoothstep(0.15, 0.05, np.abs(n1))                   # broad rivulets (~1 unit), vertical
            n2 = vnoise(np.stack([pl[:, 2] * 0.3, pl[:, 0] * 0.3, z0], 1), 1.0, (2.3, 5.0, 1.7), 1)[:, 0]
            v = 1.1 + 1.3 * smoothstep(-0.4, 0.4, n2)                  # each one's speed (units / frame)
            dtop = 66.0 - pl[:, 1]
            d = v * age - dtop                                          # > 0: the head has passed (poured)
            trail = np.exp(-np.maximum(d, 0.0) / 7.0) * smoothstep(-0.6, 0.2, d)
            head = np.exp(-(d / 0.8) ** 2) * fresh
            Lg = g * fs ** 0.6 * (0.05 + riv * (0.22 + 2.6 * trail * math.exp(-age / 22.0)))
            Lh = g * fs ** 0.6 * riv * head * 6.0
            if i in A.GIANTS:
                k = 1.0 - smoothstep(6.0, 12.0, P[:, 1])
                Lg, Lh = Lg * k, Lh * k
            z = np.maximum((P - cpos[None, :]) @ fwd, 0.3)
            a = tw.G[i][0]['a'][pt['idx'][sel]] / pt['q']
            pa = a * np.clip(ndv, 0.05, 1.0) * (fpx / z) ** 2
            colE = (self.C_GOLD[None, :] * Lg[:, None] + self.C_HOT[None, :] * Lh[:, None]) * pa[:, None]
            E = colE.max(1)
            m = E > 1e-7
            if not m.any():
                continue
            rw = np.sqrt(a / np.pi) * 1.1
            ctx.fr.splat(pt['P0'][sel][m], pt['P1'][sel][m], rw[m], E[m], colE[m] / E[m][:, None], ctx.cam0, ctx.cam1,
                         zref=0.0, myid=i, profile=1)



class CrownBreak:
    """the crown's break (bar 32 b1): a seam of fire opens across the forge half a beat before, then the broken
    faces glow raw (the stump's top and the crown's underside, which turns with it as it falls)"""

    def emit(self, ctx, tl):
        t = ctx.t
        fc = _CROWN[0]
        tw = tl.towers
        if fc is None or tw.cur is None or tw.cur[FALL_TOWER] is None:
            return
        cut = fc.cut_y(t)
        h = tw.height(FALL_TOWER, t)
        pre = float(smoothstep(A.T_CROWN - 12, A.T_CROWN, t))
        x = max(t - A.T_CROWN, 0.0)
        for kind in (0, 4):
            pt = tw.cur[FALL_TOWER]['parts'][kind]
            if len(pt['idx']) == 0:
                continue
            yw = pt['pl'][:, 1] - 66.0 + h + B.GROUND
            dd = yw - cut
            sel = np.abs(dd) < 1.1
            if not sel.any():
                continue
            if t < A.T_CROWN:
                e = pre * np.exp(-(dd[sel] / 0.25) ** 2) * (0.7 + 0.3 * math.sin(1.7 * t)) * 5.0
                T = 0.62
            else:
                e = np.exp(-np.abs(dd[sel]) / 0.45) * (1.8 * math.exp(-x / 18.0) + 0.5 * math.exp(-x / 90.0)) * 4.0
                T = 0.52 + 0.14 * math.exp(-x / 20.0)
            ctx.fr.splat(pt['P0'][sel], pt['P1'][sel], 0.06, e.astype(np.float32), look.blackbody(T), ctx.cam0, ctx.cam1,
                         zref=30.0, myid=FALL_TOWER)


class Updraft:
    """THE BRINK (H5): the fire swells and roars up into ONE column of flame and embers above it: a real fire's
    updraft, fast, turbulent, cooling from white-gold at its root to red embers high up. It roars for ~2 s
    (2410-2465), then subsides into the swollen fire as the rim gives way and the crown falls."""
    T0, T1 = A.T_BRINK, A.T_BRINK + 18          # it builds
    T2, T3 = A.T_BRINK + 70, A.T_TIP + 30       # it subsides (a weaker roar stays under the fall)

    @staticmethod
    def strength(t):
        up = float(smoothstep(Updraft.T0, Updraft.T1, t))
        down = 1.0 - 0.7 * float(smoothstep(Updraft.T2, Updraft.T3, t))
        return up * down * (1.0 - float(smoothstep(A.T_WHITE - 30, A.T_WHITE - 4, t)))

    @staticmethod
    def radius(y):
        """the column's radius at height y (world): a pillar, widening only a little as it climbs"""
        yb = 6.0
        return 2.8 + 0.07 * np.maximum(y - yb, 0.0)

    def __init__(self, seed=959):
        r = rng(seed)
        n = 60000
        self.n = n
        self.ph = r.random(n)
        self.v = r.uniform(0.9, 1.6, n)                     # units / frame: it roars
        self.a = r.uniform(0, 2 * np.pi, n)
        self.rr = r.random(n) ** 1.3                        # a dense, bright core; ragged edges
        self.E = r.lognormal(0, 0.45, n)
        self.rw = r.uniform(0.35, 0.8, n)                   # big soft blobs overlapping: a body of flame, not grains
        m = 14000                                           # embers riding it, faster, sharp
        self.m = m
        self.e_ph = r.random(m)
        self.e_v = r.uniform(1.6, 3.0, m)
        self.e_a = r.uniform(0, 2 * np.pi, m)
        self.e_rr = r.random(m) ** 0.5
        self.e_E = r.lognormal(0, 0.8, m)
        self.e_fl = r.uniform(0, 2 * np.pi, m)
        self.H = 80.0                                       # the column's reach above its root

    def _col(self, t, ph, v, a0, rr):
        C = B.crown_centre(t)
        yb = C[1] + 1.2 * SCHED_fire(t)
        s = (ph + v * (t - A.T_BRINK) / self.H) % 1.0
        y = yb + s * self.H
        R = Updraft.radius(y) * (1.0 + 0.5 * smoothstep(0.0, 0.08, s) * (1.0 - smoothstep(0.08, 0.25, s)))
        a = a0 + 0.012 * (y - yb)
        P = np.stack([C[0] + R * rr * np.cos(a), y, C[2] + R * rr * np.sin(a)], 1)
        # billows: big turbulent eddies carried up with the flow (tongues at the edges, never a smooth cone)
        w = vnoise(P * np.array([0.11, 0.05, 0.11]) + np.array([0.0, -0.05 * t, 0.0]), 1.0, (3.1, 0.7, 5.2), 2)
        amp = (0.8 + 0.1 * (y - yb)) * (0.6 + 0.7 * rr)
        P = P + w * amp[:, None] * np.array([1.0, 0.25, 1.0])
        return P, s, y - yb

    def emit(self, ctx):
        t = ctx.t
        g = Updraft.strength(t)
        if g <= 0.001:
            return
        red = B.redness(t)
        # the flame body
        P0, s0, _ = self._col(ctx.t0, self.ph, self.v, self.a, self.rr)
        P1, s, h = self._col(ctx.t1, self.ph, self.v, self.a, self.rr)
        ok = s >= s0
        # billows travelling up it, with dark gaps between them (tongues, never an even curtain)
        bil = vnoise(P1 * np.array([0.07, 0.035, 0.07]) + np.array([0.0, -0.06 * t, 0.0]), 1.0, (9.1, 2.2, 4.4), 2)[:, 0]
        puff = smoothstep(-0.15, 0.45, bil) ** 1.5
        hot = np.exp(-h / 22.0)
        e = self.E * g * puff * (0.15 + 3.0 * hot ** 1.3) * smoothstep(0.0, 0.03, s) * (1.0 - smoothstep(0.55, 1.0, s)) * ok
        e = e * (1.0 - 0.6 * self.rr) * 16.0
        T = np.clip(0.44 + 0.42 * hot ** 1.2 + 0.08 * (1 - self.rr) * hot, 0.0, 0.95)
        col = look.blackbody(T)
        col = col * (1 - 0.4 * red * (1 - hot))[:, None] + B.C_RED * (0.4 * red * (1 - hot))[:, None]
        m = e > 1e-4
        ctx.fr.splat(P0[m], P1[m], self.rw[m], e[m], col[m], ctx.cam0, ctx.cam1, zref=30.0)
        # the embers torn up with it
        Q0, q0, _ = self._col(ctx.t0, self.e_ph, self.e_v, self.e_a, self.e_rr)
        Q1, q, hq = self._col(ctx.t1, self.e_ph, self.e_v, self.e_a, self.e_rr)
        ok = q >= q0
        fl = 0.6 + 0.4 * np.sin(0.9 * t + self.e_fl)
        e = self.e_E * g * fl * (1.0 - q) ** 0.7 * smoothstep(0.0, 0.04, q) * ok * 8.0
        col = look.blackbody(np.clip(0.52 + 0.2 * np.exp(-hq / 30.0), 0, 0.85))
        m = e > 1e-4
        ctx.fr.splat(Q0[m], Q1[m], 0.03, e[m], col[m], ctx.cam0, ctx.cam1, zref=30.0)
        # its glow: the air round the column's root lit white-gold
        C = B.crown_centre(t) + np.array([0.0, 10.0, 0.0])
        ctx.fr.splat(C[None, :], C[None, :], np.array([11.0]), np.array([1.6e5 * g]), look.blackbody(0.78)[None, :],
                     ctx.cam0, ctx.cam1, profile=1, rmax=1400.0)


def SCHED_fire(t):
    return A.SCHED.fire_scale(t)


class CrownTrail:
    """the falling crown burns: embers shed along its path, hot, falling slower than it (they hang in its wake)"""

    def __init__(self, seed=989):
        r = rng(seed)
        n = 5000
        self.t0 = A.T_CROWN + r.uniform(0.0, 40.0, n)
        self.off = r.normal(0, 1, (n, 3)) * np.array([1.4, 2.2, 1.4])
        self.v = r.normal(0, 1, (n, 3)) * 0.06 + np.array([0.0, 0.02, 0.0])
        self.E = r.lognormal(0, 0.7, n)
        self.T = r.uniform(0.5, 0.8, n)

    def emit(self, ctx):
        t = ctx.t
        fc = _CROWN[0]
        if not hasattr(self, 'grid'):
            ts = np.arange(A.T_CROWN, A.T_WHITE + 1.0, 1.0)
            self.grid_t = ts
            self.grid = np.array([fc.centre(float(x)) for x in ts])
        def at(tq):
            tb = np.minimum(self.t0, tq)
            B0 = np.stack([np.interp(tb, self.grid_t, self.grid[:, k]) for k in range(3)], 1)
            a = np.maximum(tq - self.t0, 0.0)
            return B0 + self.off + self.v * a[:, None] + np.array([0.0, -0.012, 0.0]) * (a * a)[:, None], a
        P0, _ = at(ctx.t0)
        P1, a = at(ctx.t1)
        on = (t > self.t0)
        e = self.E * on * np.exp(-a / 26.0) * 9.0
        m = e > 1e-4
        ctx.fr.splat(P0[m], P1[m], 0.05, e[m], look.blackbody(self.T[m]), ctx.cam0, ctx.cam1, zref=30.0)


class RidgeFires:
    """A16: small fires on the far ridges, irregular (clusters and gaps, never a ring), each a short flame that
    flickers; they are what lights the towers' backs"""

    def __init__(self, seed=929):
        r = rng(seed)
        az = []
        for c in r.uniform(0, 2 * np.pi, 9):
            az += list(c + r.normal(0, 0.12, r.integers(1, 5)))
        az = np.array(az)
        n = len(az)
        rad = r.uniform(140, 330, n)
        self.p = np.stack([rad * np.cos(az), B.GROUND + r.uniform(-2, 22, n), rad * np.sin(az)], 1)
        self.s = r.uniform(0.6, 1.8, n)                      # a 3:1 spread of size and brightness
        self.ph = r.uniform(0, 2 * np.pi, n)
        self.n = n
        m = 40
        self.fl = r.normal(0, 1, (n, m, 3)) * np.array([0.35, 1.0, 0.35])
        self.fl[:, :, 1] = np.abs(self.fl[:, :, 1])

    def emit(self, ctx):
        t = ctx.t
        if t < A.T_LIGHT - 2:
            return
        k = float(smoothstep(A.T_LIGHT - 2, A.T_LIGHT + 20, t))
        f = 0.75 + 0.25 * np.sin(1.3 * t + self.ph) * np.sin(0.37 * t + 2 * self.ph)
        P = (self.p[:, None, :] + self.fl * (self.s * 0.9)[:, None, None]).reshape(-1, 3)
        e = np.repeat(self.s * f * k * 6.0, self.fl.shape[1]) * np.exp(-self.fl[:, :, 1].reshape(-1) / 1.2)
        col = look.blackbody(np.clip(0.5 + 0.12 * np.repeat(f, self.fl.shape[1]), 0, 1))
        ctx.fr.splat(P, P, 0.25, e, col, ctx.cam0, ctx.cam1, zref=0.0)


class SmallLights:
    """A17: small lights come down to the rim, from the towers and from the hills alike, at a walking pace"""

    def __init__(self, tl, seed=939):
        r = rng(seed)
        n = 260
        tw = tl.towers
        src = []
        for j in range(n):
            if r.random() < 0.5:
                i = r.integers(0, 8)
                b = tw.base(i)
                src.append(b + np.array([r.normal(0, 2), 1.0 + r.uniform(0, 3), r.normal(0, 2)]))
            else:
                a = r.uniform(0, 2 * np.pi)
                d = r.uniform(50, 160)
                src.append(np.array([d * math.cos(a), B.GROUND + r.uniform(0, 10), d * math.sin(a)]))
        self.src = np.array(src)
        a = np.arctan2(self.src[:, 2], self.src[:, 0]) + r.normal(0, 0.05, n)
        self.dst = np.stack([(R_RIM + 0.8) * np.cos(a), np.full(n, B.GROUND + 0.8), (R_RIM + 0.8) * np.sin(a)], 1)
        self.t0 = A.T_LIGHT + 220 + r.uniform(0, 140, n)
        self.dur = np.linalg.norm(self.dst - self.src, axis=1) / r.uniform(0.35, 0.6, n)
        self.E = r.lognormal(0, 0.3, n)

    def pos(self, t):
        u = np.clip((t - self.t0) / self.dur, 0, 1)
        return self.src + (self.dst - self.src) * u[:, None], u

    def emit(self, ctx):
        t = ctx.t
        P0, _ = self.pos(ctx.t0)
        P1, u = self.pos(ctx.t1)
        on = t > self.t0
        e = self.E * on * 2.2 * (0.85 + 0.15 * np.sin(0.8 * t + 7 * self.E))
        ctx.fr.splat(P0, P1, 0.06, e, look.blackbody(0.58), ctx.cam0, ctx.cam1, zref=30.0)


class LivingEmber:
    """A10: black; one ember drifts down in the upper middle, above the centred line, and hangs there, flickering
    low. It does not die (E9, extended)."""

    def pos(self, t):
        u = max(t - A.T_BLACK, 0.0)
        x = 0.1 + 0.16 * math.sin(u * 0.03) * math.exp(-u / 120.0) + 0.02 * math.sin(u * 0.011)
        y = 0.64 + 0.7 * math.exp(-u / 38.0) + 0.012 * math.sin(u * 0.05)
        return np.array([x, y, -6.0])

    def life(self, t):
        u = t - A.T_BLACK
        wake = float(smoothstep(0.0, 10.0, u))
        fl = 1 + 0.28 * math.sin(u * 0.9) * math.sin(u * 0.23) + 0.12 * math.sin(u * 2.7 + 1.0)
        dip = 1 - 0.45 * math.exp(-((u - 150.0) / 6.0) ** 2) - 0.35 * math.exp(-((u - 262.0) / 5.0) ** 2)
        return 0.62 * wake * fl * dip

    def emit(self, ctx):
        t = ctx.t
        lv = self.life(t)
        if lv <= 0.001:
            return
        p0 = self.pos(ctx.t0)[None, :]
        p1 = self.pos(ctx.t1)[None, :]
        temp = 0.47 + 0.3 * lv
        ctx.fr.splat(p0, p1, 0.006, np.array([60.0 * lv]), look.blackbody(temp)[None, :], ctx.cam0, ctx.cam1)
        H = np.array([self.pos(t)])
        ctx.fr.splat(H, H, np.array([0.12]), np.array([90.0 * lv]), look.blackbody(temp - 0.1)[None, :],
                     ctx.cam0, ctx.cam1, profile=1)


def emit_black(tl, ctx):
    tl._get('living_ember', LivingEmber).emit(ctx)


class Ash:
    """A9: grey ash falling slowly through the dead valley"""

    def __init__(self, seed=919):
        r = rng(seed)
        n = 26000
        self.p = np.stack([r.uniform(-160, 160, n), r.uniform(-20, 60, n), r.uniform(-160, 160, n)], 1)
        self.v = r.uniform(0.04, 0.12, n)
        self.E = r.lognormal(0, 0.5, n)

    def emit(self, ctx):
        t = ctx.t
        if t < A.T_DEAD - 10 or t >= A.T_BLACK:
            return

        def pts(tq):
            q = self.p.copy()
            q[:, 1] = -20.0 + (q[:, 1] + 20.0 - self.v * (tq - A.T_WHITE)) % 80.0
            return q + vnoise(q * 0.03 + np.array([0.004 * tq, 0, 0]), 0.6, (0, 0, 0), 1) * 3.0
        k = float(smoothstep(A.T_DEAD - 10, A.T_DEAD + 20, t)) * (1 - float(smoothstep(A.T_BLACK - 20, A.T_BLACK - 2, t)))
        e = self.E * 0.35 * k
        ctx.fr.splat(pts(ctx.t0), pts(ctx.t1), 0.05, e, np.array([0.5, 0.5, 0.52]), ctx.cam0, ctx.cam1, zref=30.0)


def emit_ash(tl, ctx):
    tl._get('ash', Ash).emit(ctx)
