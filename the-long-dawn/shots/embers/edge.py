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

R_RIM = 13.6          # the crater's lip (the forges stand on it: their bases are at 16.8-19.2)
WALL = 5.0            # the wall falls over this much radius
DEPTH = 24.0          # the molten floor, below the ground
FLOOR_Y = B.GROUND - DEPTH
GILD_W = {2: 1.0, 6: 1.0, 1: 0.85, 4: 0.9, 7: 0.8, 0: 0.15, 3: 0.2, 5: 0.1}   # the nearest (gilded) vs the farthest
FALL_TOWER = 4        # the gilded tower whose crown breaks off (never a giant's)
C_MOLT = np.array([1.0, 0.72, 0.3])


def crater_y(r):
    return B.GROUND - DEPTH * smoothstep(R_RIM, R_RIM - WALL, r)


# ---------------------------------------------------------------- schedule (added to a3.A3Sched)

def _gild(self, i, t):
    if i >= 8 or t < A.T_EDGE - 5:
        return 0.0, (0.35 * self.race(t) if i >= 8 else 0.0)
    w = GILD_W[i]
    base = float(smoothstep(A.T_EDGE, A.T_EDGE + 320, t))
    if t >= A.T_LIGHT:
        base *= 1.0 - 0.8 * float(smoothstep(A.T_STOP, A.T_HOLDS, t))        # the gilding cools
    g = (w * base * (1.0 + 0.9 * self.beat_pulse(t))) if w >= 0.5 else 0.0
    dark = (1.0 - w) * float(smoothstep(A.T_EDGE + 80, A.T_BRINK, t)) if w < 0.5 else 0.0
    if t >= A.T_LIGHT:
        dark *= 1.0 - float(smoothstep(A.T_OPEN_OTHERS, A.T_HOLDS, t))
    return g, dark


def _tower_lean(self, t):
    if t >= A.T_LIGHT:
        base = 0.07 - 0.03 * float(smoothstep(A.T_STOP, A.T_HOLDS, t))
    else:
        base = 0.07 * float(smoothstep(A.T_EDGE, A.T_EDGE + 400, t)) + 0.02 * float(smoothstep(A.T_BRINK, A.T_WHITE, t))
    lean = np.zeros(18)
    for i in range(8):
        lean[i] = base * (0.35 + 0.9 * GILD_W[i])
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
    P[above] = ((P[above] - pivot) @ R.T + pivot + off).astype(P.dtype)
    return P


def _shutter(self, i, t, pl):
    """window light: shuttered (dim) toward the fire through the race; at the turn the two giants open first, on the
    sides that face each other, then the smaller towers, one by one on the beat (A16)"""
    if t < A.T_LIGHT:
        return 0.3
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


BEAT_F = 20.0
A.A3Sched.vortex_shape = _vortex_shape
A.A3Sched.shutter = _shutter
A.A3Sched.back_light = _back_light
A.A3Sched.gild = _gild
A.A3Sched.tower_lean = _tower_lean
A.A3Sched.tower_post = _tower_post


# ---------------------------------------------------------------- the crater

class Crater:
    def __init__(self, seed=909):
        r = rng(seed)
        # the ground inside the ring: plates that crack and fall away from the centre out
        n = 70000
        rr = R_RIM * np.sqrt(r.random(n))
        a = r.uniform(0, 2 * np.pi, n)
        self.g_p = np.stack([rr * np.cos(a), np.full(n, B.GROUND), rr * np.sin(a)], 1)
        ns = 46
        sr = R_RIM * np.sqrt(r.random(ns))
        sa = r.uniform(0, 2 * np.pi, ns)
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
        # the wall: ember rock with molten veins, lit from below
        m = 110000
        wr = R_RIM - WALL * r.random(m) ** 0.8
        wa = r.uniform(0, 2 * np.pi, m)
        wy = crater_y(wr) + r.normal(0, 0.25, m)
        self.w_p = np.stack([wr * np.cos(wa), wy, wr * np.sin(wa)], 1)
        self.w_n = np.stack([-np.cos(wa), np.full(m, 0.6), -np.sin(wa)], 1)
        self.w_n /= np.linalg.norm(self.w_n, axis=1, keepdims=True)
        vein = vnoise(self.w_p * np.array([0.35, 0.9, 0.35]), 1.0, (1.3, 2.2, 0.4), 2)[:, 0]
        self.w_vein = np.exp(-(vein / 0.07) ** 2)
        self.w_E = r.lognormal(0, 0.45, m)
        self.w_a = wa
        # the floor: molten, churning
        k = 60000
        fr_ = (R_RIM - WALL + 1.5) * np.sqrt(r.random(k))
        fa = r.uniform(0, 2 * np.pi, k)
        self.f_p = np.stack([fr_ * np.cos(fa), np.full(k, FLOOR_Y) + r.normal(0, 0.2, k), fr_ * np.sin(fa)], 1)
        self.f_E = r.lognormal(0, 0.5, k)
        # the rim's lip: a line of heat along the edge
        q = 9000
        la = r.uniform(0, 2 * np.pi, q)
        self.l_p = np.stack([R_RIM * np.cos(la), np.full(q, B.GROUND) + r.normal(0, 0.08, q), R_RIM * np.sin(la)], 1)
        self.l_a = la
        # debris crumbling off the rim under the gilded towers (from bar 26), and the collapse at 2440
        ndb = 9000
        self.d_tw = r.choice([i for i in range(8) if GILD_W[i] >= 0.5], ndb)
        self.d_t0 = np.where(r.random(ndb) < 0.55, r.uniform(A.T_CRUMBLE, A.T_WHITE, ndb),
                             A.T_RIM_GIVES + r.exponential(14.0, ndb))
        self.d_t0 = np.where((self.d_t0 > A.T_WHITE), A.T_WHITE - r.uniform(1, 30, ndb), self.d_t0)
        self.d_off = r.normal(0, 1, (ndb, 3)) * np.array([2.2, 0.4, 2.2])
        self.d_v = r.normal(0, 1, (ndb, 3)) * np.array([0.05, 0.03, 0.05]) + np.array([0.0, 0.05, 0.0])
        self.d_E = r.lognormal(0, 0.7, ndb)
        self.d_hot = r.random(ndb)
        print('crater: ground', n, 'wall', m, 'floor', k)

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

    def emit(self, ctx, lp):
        t = ctx.t
        if t < A.T_EDGE - 2 or (A.T_WHITE <= t < A.T_LIGHT):
            return
        red = B.redness(t)
        open_k = float(smootherstep(A.T_EDGE + 4, A.T_EDGE + 46, t))
        heat = 1.0 + 0.5 * B.beat_pulse(t)
        # the ground plates: cracks glow before they part, then the plates drop into the glow
        P0, _, _ = self._ground(ctx.t0)
        P1, x, fall = self._ground(ctx.t1)
        pre = smoothstep(A.T_EDGE - 2, A.T_EDGE + 10, t) * (1 - smoothstep(0.0, 6.0, x))
        alive = fall < DEPTH - 2
        e = self.g_E * (0.05 + 1.6 * self.g_seam * pre + 0.25 * smoothstep(0.0, 8.0, x) * smoothstep(DEPTH, 4.0, fall))
        e = e * alive * heat
        col = look.blackbody(np.clip(0.4 + 0.25 * self.g_seam * pre + 0.15 * smoothstep(0, 10, fall), 0, 0.8))
        ctx.fr.splat(P0, P1, 0.05, e, col, ctx.cam0, ctx.cam1, zref=30.0)
        if open_k <= 0:
            return
        # the wall: lit from the molten floor below, veins of fire
        up = smoothstep(B.GROUND, FLOOR_Y, self.w_p[:, 1])
        crumble = 0.0
        e = self.w_E * (0.05 + 0.35 * up + 2.2 * self.w_vein * (0.4 + 0.6 * up)) * open_k * heat
        col = look.blackbody(np.clip(0.36 + 0.2 * up + 0.18 * self.w_vein, 0, 0.85))
        col = col * (1 - 0.4 * red) + (B.C_RED * 0.6 + B.C_CRIMSON * 0.4) * 0.4 * red
        ctx.fr.splat(self.w_p, self.w_p, 0.07, e, col, ctx.cam0, ctx.cam1, zref=30.0)
        # the floor: molten fire, churning (the glare the towers lean over)
        w = vnoise(self.f_p * 0.18 + np.array([0.0, 0.0, 0.01 * t]), 1.0, (0, 0, 0), 2)[:, 0]
        e = self.f_E * (0.7 + 0.6 * w) * 2.6 * open_k * heat * (1 + 0.8 * float(smoothstep(A.T_BRINK, A.T_WHITE, t)))
        col = look.blackbody(np.clip(0.6 + 0.12 * w, 0, 0.9))
        ctx.fr.splat(self.f_p, self.f_p, 0.12, e, col, ctx.cam0, ctx.cam1, zref=30.0)
        H = np.array([[0.0, FLOOR_Y + 2.0, 0.0]])
        ctx.fr.splat(H, H, np.array([14.0]), np.array([2600.0 * open_k * heat]), look.blackbody(0.6)[None, :],
                     ctx.cam0, ctx.cam1, profile=1)
        # the lip of the rim
        el = 1.4 * open_k * (0.6 + 0.4 * np.sin(3 * self.l_a + 0.2 * t)) * heat
        ctx.fr.splat(self.l_p, self.l_p, 0.05, el, look.blackbody(0.55), ctx.cam0, ctx.cam1, zref=30.0)
        # debris: the rim crumbling under the gilded towers
        self._debris(ctx)

    def _debris(self, ctx):
        t = ctx.t
        if t < A.T_CRUMBLE:
            return
        tw = ctx.tl.towers
        base = np.array([tw.base(i) for i in range(8)])
        ang = np.arctan2(base[:, 2], base[:, 0])[self.d_tw]
        src = np.stack([R_RIM * np.cos(ang), np.full(len(ang), B.GROUND), R_RIM * np.sin(ang)], 1) + self.d_off

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
        ctx.fr.splat(S0[on], S1[on], 0.06, e[on], col[on], ctx.cam0, ctx.cam1, zref=30.0)


# ---------------------------------------------------------------- the falling crown (A8)

class FallingCrown:
    """a gilded tower's crown breaks off (2480) and falls into the pit as a rigid point cloud, trailing embers"""

    def __init__(self, towers):
        self.tw = towers

    def cut_y(self, t):
        return B.GROUND + self.tw.height(FALL_TOWER, A.T_CROWN) - 9.0

    def state(self, t):
        """rigid transform applied to the crown's points (world): (R, pivot, offset)"""
        x = max(t - A.T_CROWN, 0.0)
        tw = self.tw
        b = tw.base(FALL_TOWER)
        inward = -b / max(np.linalg.norm(b[[0, 2]]), 1e-6)
        inward[1] = 0.0
        pivot = np.array([b[0], self.cut_y(t), b[2]])
        ang = 0.004 * x * x + 0.02 * x                              # it tips over into the pit
        axis = np.cross(np.array([0.0, 1.0, 0.0]), inward)
        axis /= np.linalg.norm(axis)
        c, s = math.cos(ang), math.sin(ang)
        K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
        R = np.eye(3) + s * K + (1 - c) * (K @ K)
        off = inward * (0.05 * x * x * 0.6 + 0.3 * x) + np.array([0.0, -0.5 * 0.07 * max(x - 6.0, 0.0) ** 2, 0.0])
        return R, pivot, off


# ---------------------------------------------------------------- cameras

ORB_R, ORB_Y = 90.0, 33.0                # the whole stage: the ring of forges round the crater, against its glare
ORB_A0 = A.TH_G2 - 0.11                  # it starts behind the giant where A6 left it
ORB_W = 0.0022                           # rad / frame: one steady orbit (~87 degrees over THE EDGE)


def _orbit_az(t):
    return ORB_A0 + ORB_W * (t - A.T_EDGE)


def camera(tl, t):
    if t < A.T_WHITE:
        a = _orbit_az(t)
        k_in = float(smootherstep(A.T_EDGE, A.T_EDGE + 90, t))      # out from behind the giant as the ground falls
        r = lerp(31.0, ORB_R, k_in)
        y = lerp(10.0, ORB_Y, k_in)
        ty = lerp(3.0, -5.0, k_in)
        hf = lerp(56.0, 48.0, k_in)
        if t >= A.T_TIP - 30:
            # over the rim: in, over the lip, pitching down after the falling crown toward the fire
            u = float(ease_in(np.clip((t - (A.T_TIP - 30)) / (A.T_WHITE - (A.T_TIP - 30)), 0, 1), 1.7))
            r = lerp(ORB_R, 2.0, u)
            y = lerp(ORB_Y, -6.0, u ** 1.3)
            ty = lerp(-5.0, -32.0, float(smoothstep(0.0, 0.5, u)))
            hf = lerp(48.0, 74.0, u)
        pos = np.array([r * math.cos(a), y, r * math.sin(a)])
        tgt = np.array([0.0, ty, 0.0])
        if A.T_RIM_GIVES <= t < A.T_WHITE:
            k = math.exp(-(t - A.T_RIM_GIVES) / 18.0) + 0.5 * float(smoothstep(A.T_TIP, A.T_WHITE, t))
            pos = pos + 0.35 * k * np.array([math.sin(3.1 * t), math.sin(4.3 * t), math.cos(2.3 * t)])
        focus = float(np.linalg.norm(B.crown_centre(t) - pos))
        return Camera(pos, tgt, hfov=hf, focus=max(focus, 2.0), aperture=0.1)
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
    return tl._get('crater', Crater)


def emit_before_towers(tl, ctx, lp, lc, lpw):
    ctx.tl = tl
    _crater(tl).emit(ctx, lp)


def emit_after(tl, ctx, lp, lc, lpw):
    t = ctx.t
    if A.T_BRINK <= t < A.T_WHITE:
        tl._get('strip_embers', lambda: StripEmbers(tl)).emit(ctx)
    if t >= A.T_LIGHT:
        tl._get('ridge_fires', RidgeFires).emit(ctx)
        if t >= A.T_SEEN - 60:
            tl._get('small_lights', lambda: SmallLights(tl)).emit(ctx)


def post(tl, ctx, hdr):
    t = ctx.t
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
    """THE BRINK: the vortex strips embers off the tower tops and spirals them in toward the fire"""

    def __init__(self, tl, seed=949):
        r = rng(seed)
        self.tl = tl
        n = 16000
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
        C = B.crown_centre(t)
        rr = r0 * (1 - u) ** 1.3 + 3.0 * u
        a = a0 + self.sw * u ** 1.2
        y = top[:, 1] + 2.0 + (C[1] + 12.0 - top[:, 1]) * u ** 1.6
        P = np.stack([rr * np.cos(a), y, rr * np.sin(a)], 1) + self.jit * (0.6 + 1.6 * u)[:, None]
        return P, u, age

    def emit(self, ctx):
        t = ctx.t
        if t < A.T_BRINK or t >= A.T_WHITE:
            return
        g = float(smoothstep(A.T_BRINK, A.T_BRINK + 60, t))
        P0, _, _ = self.pts(ctx.t0)
        P1, u, age = self.pts(ctx.t1)
        ok = age > 0.6
        e = self.E * g * (1 - u) ** 0.8 * smoothstep(0.0, 0.08, u) * 3.0 * ok
        col = look.blackbody(np.clip(0.5 + 0.3 * u, 0, 0.9))
        ctx.fr.splat(P0[ok], P1[ok], 0.03, e[ok], col[ok], ctx.cam0, ctx.cam1, zref=30.0)


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
