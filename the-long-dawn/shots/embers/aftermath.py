"""EMBERS v3, cut A (EMBERS-A2): A9 THE DEAD VALLEY and A10 BLACK . THE EMBER, on A's own frames.

A9  2640-2800  Out of the white (the IMPACT, bar 34 b1), from bar 34 b2 the fire's second vision: the valley the
               promise drew (a3.Promise, seen from the promise's own camera) in the promise's own ember points,
               but grey and unlit -- cinder under falling ash. The terraces are broken, the orchards are dead
               trunks, the river a dry bed, the roofs fallen in, the windows dark, and where the fire stood there is
               only a cold heap of ash. No light anywhere. It crumbles into the black on the last beat. T9 sits in
               the lower third (2670-2790), so the near ground there stays calm.
A10 2800-3120  Black: two seconds of true silence. Then one ember drifts down out of the dark and hangs, flickering,
               below the centred T10 lines (y 372 / 440, 2880-3060). Twice it nearly goes out; it does not die.
               It ends where A11's star plate carries it on (EMBER_END, px).
"""
import math

import numpy as np

import look
from core import Camera, smoothstep, smootherstep, lerp, rng, vnoise
import scene_b as B
import a3 as A

T_WHITE, T_DEAD, T_BLACK, T_END = A.T_WHITE, A.T_DEAD, A.T_BLACK, A.T_EMBER_END      # 2640 2660 2800 3120
T_CRUMBLE = A.bar(35, 4) - 6        # 2774: the last beat; it is black by 2800

C_ASH = np.array([0.53, 0.54, 0.57])  # cinder grey, a breath cool: nothing in the dead valley is warm
C_ASH_DARK = np.array([0.36, 0.36, 0.38])


# ============================================================================ A9: the dead valley

class DeadValley:
    """the promise's own points, turned to cinder: each kind of form dies its own way"""

    def __init__(self, promise, seed=4040):
        r = rng(seed)
        P, K, E = promise.P, promise.K, promise.E
        out_p, out_e, out_c = [], [], []

        def add(p, e, c):
            out_p.append(np.asarray(p, np.float64))
            out_e.append(np.broadcast_to(np.asarray(e, np.float64), (len(p),)).copy())
            out_c.append(np.broadcast_to(np.asarray(c, np.float64), (len(p), 3)).copy())
        grey = lambda n, lo=0.8, hi=1.15: C_ASH[None, :] * r.uniform(lo, hi, (n, 1))
        # 0 terraces: the contour walls broken into short, sagging runs
        m = K == 0
        p = P[m].copy()
        brk = vnoise(p * np.array([0.09, 0.0, 0.09]), 1.0, (3.1, 0.0, 7.7), 1)[:, 0]
        keep = brk > -0.12
        p = p[keep]
        p[:, 1] -= 0.25 * r.random(len(p))
        add(p, E[m][keep] * 0.95, grey(len(p)))
        # 1 orchards: each tree (16 consecutive points) becomes a dead trunk with a few bare limbs
        m = np.nonzero(K == 1)[0]
        g = m[: len(m) // 16 * 16].reshape(-1, 16)
        cen = P[g].mean(1)
        nt = len(cen)
        base = cen - np.array([0.0, 1.4, 0.0])
        h = r.uniform(1.3, 2.1, nt)
        tr = []
        for k in range(7):                                        # the trunk
            tr.append(base + np.stack([r.normal(0, 0.03, nt), h * (k / 6.0), r.normal(0, 0.03, nt)], 1))
        for k in range(3):                                        # bare limbs, two points each
            ang = r.uniform(0, 2 * np.pi, nt)
            lean = r.uniform(0.25, 0.6, nt)
            y0 = h * r.uniform(0.55, 0.85, nt)
            for s_ in (0.5, 1.0):
                tr.append(base + np.stack([np.cos(ang) * lean * s_, y0 + 0.45 * s_ * lean, np.sin(ang) * lean * s_], 1))
        tp = np.concatenate(tr)
        fallen = r.random(nt) < 0.22                               # some are down: lying on the ground
        ti = np.tile(np.arange(nt), len(tr))
        f_ = fallen[ti]
        rel = tp[f_] - base[ti[f_]]
        tp[f_] = base[ti[f_]] + np.stack([rel[:, 1] * 0.9, 0.05 + 0.1 * r.random(int(f_.sum())), rel[:, 0]], 1)
        add(tp, np.full(len(tp), 1.15), grey(len(tp), 0.7, 1.0))
        # 2 the river: a dry bed; the banks remain, fainter, and the water is gone
        m = K == 2
        p = P[m]
        keep = r.random(len(p)) < 0.55
        add(p[keep] + np.array([0.0, -0.15, 0.0]), E[m][keep] * 0.9, grey(int(keep.sum()), 0.6, 0.9))
        # 3 roofs (160 consecutive points a house): fallen in -- the ridge sags to the walls, half a roof is gone
        m = np.nonzero(K == 3)[0]
        g = m[: len(m) // 160 * 160].reshape(-1, 160)
        for idx in g:
            q = P[idx].copy()
            y0 = q[:, 1].min()
            top = q[:, 1].max()
            v = (q[:, 1] - y0) / max(top - y0, 1e-6)
            c = q.mean(0)
            side = np.sign(((q - c) @ np.array([1.0, 0.0, 0.3])))
            kind = r.random()
            if kind < 0.45:                                       # half the roof gone, the rest slumped
                keep = side > 0
                q[:, 1] -= (top - y0) * 0.55 * v
            elif kind < 0.8:                                      # the ridge broken, sagging in the middle
                keep = r.random(len(q)) < 0.8
                mid = np.exp(-(((q - c) @ np.array([1.0, 0.0, 0.0])) / 1.2) ** 2)
                q[:, 1] -= (top - y0) * 0.8 * v * mid
            else:                                                 # only the walls' stumps
                keep = v < 0.35
            q = q[keep]
            add(q, np.full(len(q), 1.1), grey(len(q), 0.75, 1.1))
            # a little rubble at its feet
            nr = 12
            rub = c + np.stack([r.normal(0, 1.4, nr), np.full(nr, y0 - 1.6 - c[1]) + r.uniform(0, 0.3, nr),
                                r.normal(0, 1.4, nr)], 1)
            add(rub, np.full(nr, 0.8), grey(nr, 0.6, 0.9))
        # 4 the windows: dark (nothing)
        # the cold hearth where the fire stood: a low heap of ash
        n = 900
        a = r.uniform(0, 2 * np.pi, n)
        rr = 3.2 * np.sqrt(r.random(n))
        hh = 1.6 * (1 - rr / 3.2) ** 1.6
        heap = np.stack([rr * np.cos(a), B.GROUND - 1.2 + hh * r.random(n) ** 0.5, rr * np.sin(a)], 1)
        add(heap, np.full(n, 0.9), grey(n, 0.55, 0.95))
        # the ground itself under a skin of ash: a fine, even grey stipple over the valley floor and the hill
        # flanks, thinning with distance (so it is a place, not a point cloud of objects)
        n = 70000
        a = A.RIVER_AZ + r.uniform(-1.75, 1.75, n)
        rr = 8.0 + 420.0 * r.random(n) ** 1.6
        x, z = rr * np.cos(a), rr * np.sin(a)
        gy = A.valley_h(x, z) - 0.1 + 0.15 * r.random(n)
        add(np.stack([x, gy, z], 1), r.lognormal(-0.2, 0.5, n) * 1.2, C_ASH_DARK[None, :] * r.uniform(0.7, 1.2, (n, 1)))
        self.P = np.concatenate(out_p)
        self.E = np.concatenate(out_e)
        self.C = np.concatenate(out_c)
        self.r = np.hypot(self.P[:, 0], self.P[:, 2])
        self.ph = r.uniform(0, 2 * np.pi, len(self.P))
        self.ground = np.zeros(len(self.P), bool)
        self.ground[-70000:] = True
        nh = 220
        a = A.RIVER_AZ + r.uniform(-1.6, 1.6, nh)
        rr = 20.0 + 360.0 * r.random(nh) ** 1.3
        x, z = rr * np.cos(a), rr * np.sin(a)
        self.haze = np.stack([x, A.valley_h(x, z) + r.uniform(1.0, 7.0, nh), z], 1)
        self.haze_r = r.uniform(9.0, 22.0, nh) * (0.5 + rr / 380.0)
        self.haze_e = r.uniform(0.5, 1.5, nh)
        self.fall = r.uniform(3.0, 9.0, len(self.P))
        self.drift = r.normal(0, 1, (len(self.P), 3)) * np.array([0.6, 0.0, 0.6])
        print('dead valley: points', len(self.P))

    def at(self, t):
        """positions at t: still, then crumbling on the last beat (a sag, a slump, a drift of dust)"""
        c = smoothstep(T_CRUMBLE - 10 + 8 * np.sin(self.ph), T_BLACK - 2, t)
        dy = -self.fall * c ** 2
        return self.P + np.stack([self.drift[:, 0] * c, dy, self.drift[:, 2] * c], 1), c

    def emit(self, ctx):
        t = ctx.t
        if t < T_WHITE + 8 or t >= T_BLACK:
            return
        P0, _ = self.at(ctx.t0)
        P1, c = self.at(ctx.t1)
        # the promise inverted: the white is drawn back into the fire's place (A5's light rolled OUT) and behind
        # its front the land is left grey
        cam = ctx.cam
        H, W = ctx.fr.H, ctx.fr.W
        u, v, _ = cam.project(P1, W, H)
        near_k = np.clip(np.linalg.norm(P1 - cam.pos, axis=1) / 70.0, 0.35, 1.0) ** 1.5    # a calm foreground
        uc, vc = hearth_px(cam, W, H)
        d = np.sqrt(((u - uc) / W) ** 2 + ((v - vc) / (0.62 * W)) ** 2)
        R = white_r(t)
        show = smoothstep(0.8 * R, 1.25 * R + 0.02, d) * float(smoothstep(T_WHITE + 4, T_WHITE + 14, t))
        z = np.maximum((P1 - cam.pos) @ cam.R[2], 1.0)
        e = self.E * show * (1.0 - c) ** 1.5 * np.clip(40.0 / z, 0.0, 2.5) * GAIN * np.where(self.ground, near_k, 1.0)
        m = e > 1e-6
        ctx.fr.splat(P0[m], P1[m], 0.1, e[m], self.C[m], ctx.cam0, ctx.cam1, zref=0.0)
        # a low grey haze lying in the valley: ash in the air, catching no light
        hz = float(smoothstep(T_DEAD, T_DEAD + 40, t)) * (1.0 - float(smoothstep(T_CRUMBLE, T_BLACK - 2, t)))
        if hz > 0:
            Hh = self.haze
            zz = np.maximum((Hh - cam.pos) @ cam.R[2], 5.0)
            rpx = self.haze_r * cam.f_px(1920) / zz
            e = HAZE * self.haze_e * np.pi * rpx ** 2 * hz
            ctx.fr.splat(Hh, Hh, self.haze_r, e, np.broadcast_to(C_ASH, (len(Hh), 3)),
                         ctx.cam0, ctx.cam1, profile=1, zref=0.0, rmax=900.0)


HAZE = 0.006
GAIN = 2.4


def white_r(t):
    """the white's radius as a fraction of the frame width: it is drawn back into the fire's place"""
    u = float(np.clip((t - (T_WHITE + 2)) / 40.0, 0.0, 1.0))
    return 1.6 * (1.0 - u) ** 1.8


def hearth_px(cam, W, H):
    """where the fire stood (it hung over the valley in the promise), on screen"""
    u, v, z = cam.project(np.array([[0.0, 0.0, 0.0]]), W, H)
    return float(u[0]), float(v[0])


class Ash:
    """ash falling through the dead valley: flakes that tumble (their grey catches no light, it only flickers as
    they turn), thicker as it goes on; a few pass close to the lens, soft and large"""

    def __init__(self, seed=919):
        r = rng(seed)
        n = 30000
        self.p = np.stack([r.uniform(-170, 170, n), r.uniform(-24, 64, n), r.uniform(-170, 170, n)], 1)
        self.v = r.uniform(0.03, 0.08, n)
        self.E = r.lognormal(0, 0.5, n)
        self.ph = r.uniform(0, 2 * np.pi, n)
        self.fr = r.uniform(0.05, 0.2, n)
        self.T = r.uniform(0.85, 1.1, n)
        # the near ones: a few hundred flakes in the first 14 units of the view
        m = 380
        self.q = r.uniform(-1, 1, (m, 3)) * np.array([7.0, 4.5, 1.0])
        self.qd = r.uniform(2.2, 14.0, m)
        self.qv = r.uniform(0.0012, 0.0035, m)
        self.qph = r.uniform(0, 2 * np.pi, m)
        self.qE = r.lognormal(0, 0.4, m)

    def emit(self, ctx):
        t = ctx.t
        if t < T_WHITE + 6 or t >= T_BLACK:
            return
        k = float(smoothstep(T_WHITE + 6, T_DEAD + 30, t)) * (1.0 + 0.6 * float(smoothstep(T_DEAD + 30, T_BLACK, t)))
        k *= 1.0 - float(smoothstep(T_BLACK - 10, T_BLACK - 1, t))

        def pts(tq):
            q = self.p.copy()
            q[:, 1] = -24.0 + (q[:, 1] + 24.0 - self.v * (tq - T_WHITE)) % 88.0
            return q + vnoise(q * 0.03 + np.array([0.004 * tq, 0, 0]), 0.6, (0, 0, 0), 1) * 3.0
        tum = 0.55 + 0.45 * np.abs(np.sin(self.fr * t + self.ph))             # a flake turning edge-on and back
        cam = ctx.cam
        P1 = pts(ctx.t1)
        z = np.maximum((P1 - cam.pos) @ cam.R[2], 1.0)
        e = self.E * ASH_FAR * k * tum * np.clip(30.0 / z, 0.15, 1.0)
        col = C_ASH[None, :] * self.T[:, None]
        ctx.fr.splat(pts(ctx.t0), P1, 0.06, e, col, ctx.cam0, ctx.cam1, zref=0.0)
        # near flakes, in camera space (they fall through the frame, soft)
        cam0, cam1 = ctx.cam0, ctx.cam1

        def near(cam, tq):
            y = 4.5 - ((4.5 - self.q[:, 1] + self.qv * self.qd * (tq - T_WHITE)) % 9.0)
            x = self.q[:, 0] + 0.35 * np.sin(0.03 * tq + self.qph)
            loc = np.stack([x * self.qd / 7.0, y * self.qd / 7.0, self.qd], 1)
            return cam.pos + loc @ cam.R
        tq_ = 0.3 + 0.7 * np.abs(np.sin(0.09 * t + self.qph)) ** 2
        ctx.fr.splat(near(cam0, ctx.t0), near(cam1, ctx.t1), 0.03, self.qE * ASH_NEAR * k * tq_ * (self.qd / 6.0),
                     np.broadcast_to(C_ASH_DARK, (len(self.qE), 3)), cam0, cam1, zref=0.0)


ASH_FAR = 0.7
ASH_NEAR = 9.0


# ============================================================================ A10: the ember

EMBER_REST = (960.0, 548.0)     # px (1920x804): where it hangs, below the centred T10 lines (y 372 / 440)
EMBER_END = EMBER_REST          # ... and where A11's star plate must carry it on at f3120
T_EMBER_IN = T_BLACK + 30       # 2830: after the true silence has begun, it appears high in the frame
T_EMBER_REST = T_BLACK + 110    # 2910: it has come to rest (before T10b arrives)


def ember_px(t):
    """the ember's screen position (px, full res): it drifts down from above the frame, swaying like something
    much lighter than it is, slows, and hangs; afterwards only a breath of drift"""
    u = np.clip((t - T_EMBER_IN) / (T_EMBER_REST - T_EMBER_IN), 0.0, 1.0)
    fall = 1.0 - (1.0 - u) ** 2.6                              # decelerating
    y = lerp(-40.0, EMBER_REST[1], fall)
    sway = 58.0 * math.sin(2.1 * u * math.pi + 0.4) * (1.0 - u) ** 1.4
    x = EMBER_REST[0] + 34.0 * (1.0 - fall) + sway
    s = max(t - T_EMBER_REST, 0.0)
    w = float(smoothstep(T_EMBER_REST - 30, T_EMBER_REST + 20, t))
    x += w * (2.2 * math.sin(0.021 * s) + 0.9 * math.sin(0.057 * s + 1.0))
    y += w * (1.6 * math.sin(0.017 * s + 0.6) + 0.7 * math.sin(0.043 * s))
    return x, y


def ember_life(t):
    """0..1: its glow. It wakes as it falls into view, breathes, and twice nearly goes out; it does not die"""
    if t < T_EMBER_IN - 6:
        return 0.0
    u = t - T_EMBER_IN
    wake = float(smoothstep(-6.0, 26.0, u))
    fl = 1.0 + 0.2 * math.sin(0.23 * t) * math.sin(0.071 * t + 1.3) + 0.08 * math.sin(1.7 * t + 0.4) \
        + 0.05 * math.sin(3.1 * t)
    dip = 1.0 - 0.8 * math.exp(-((t - 3004.0) / 11.0) ** 2) - 0.55 * math.exp(-((t - 3082.0) / 8.0) ** 2)
    return max(wake * fl * dip, 0.0)


def _px_to_world(cam, x, y, z=10.0):
    f = cam.f_px(1920)
    lx = (x - 959.5) / f * z
    ly = -(y - 401.5) / f * z
    return cam.pos + np.array([lx, ly, z]) @ cam.R


def emit_ember(ctx):
    t = ctx.t
    lv = ember_life(t)
    if lv <= 1e-4:
        return
    cam = ctx.cam
    p0 = _px_to_world(ctx.cam0, *ember_px(ctx.t0))[None, :]
    p1 = _px_to_world(ctx.cam1, *ember_px(ctx.t1))[None, :]
    temp = 0.5 + 0.34 * min(lv, 1.2)
    # the ember itself: a tiny hot bead (a few px), cooler at its skin
    ctx.fr.splat(p0, p1, 0.0045, np.array([70.0 * lv]), look.blackbody(temp)[None, :], ctx.cam0, ctx.cam1, zref=0.0)
    ctx.fr.splat(p0, p1, 0.011, np.array([26.0 * lv]), look.blackbody(temp - 0.14)[None, :], ctx.cam0, ctx.cam1,
                 zref=0.0, profile=1)
    # its glow on the air round it
    H = _px_to_world(cam, *ember_px(t))[None, :]
    ctx.fr.splat(H, H, np.array([0.1]), np.array([110.0 * lv ** 1.3]), look.blackbody(temp - 0.24)[None, :],
                 ctx.cam0, ctx.cam1, profile=1, zref=0.0)


# ============================================================================ the timeline hooks (a3._mine)

def camera(tl, t):
    if t < T_BLACK:
        # the dead valley: the promise's own view, sinking a little as the ash comes down
        u = float(np.clip((t - T_WHITE) / (T_BLACK - T_WHITE), 0, 1))
        pos, tgt, hf = A.cam_a5a6(1400.0)
        d = tgt - pos
        pos = pos + d / np.linalg.norm(d) * (3.5 * u) + np.array([0.0, -1.8 * u, 0.0])
        tgt = tgt + np.array([0.0, -1.0 * u, 0.0])
        return Camera(pos, tgt, hfov=hf, focus=float(np.linalg.norm(tgt - pos)), aperture=0.05)
    return Camera(np.zeros(3), np.array([0.0, 0.0, -10.0]), hfov=50.0, focus=10.0, aperture=0.0)


def render_opts(tl, f):
    if f < T_BLACK:
        return dict(bokeh_pow=0.25, bokeh_cap=1.6, fog_start=260.0, fog_len=320.0, near=0.3)
    return dict(bokeh_pow=0.0, bokeh_cap=1.0, near=0.05)


def emit(tl, ctx):
    t = ctx.t
    if t < T_BLACK:
        tl._get('af_valley', lambda: DeadValley(tl.promise)).emit(ctx)
        tl._get('af_ash', Ash).emit(ctx)
    else:
        emit_ember(ctx)


def post(tl, ctx, hdr):
    t = ctx.t
    if t < T_WHITE + 56:
        # the IMPACT's white is drawn back into the fire's place: a soft glow contracting and dimming, the grey
        # land left behind it (no edge anywhere: light going out, not a wipe)
        H, W = hdr.shape[:2]
        uc, vc = hearth_px(ctx.cam, W, H)
        R = max(white_r(t), 1e-3)
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        d = np.sqrt(((xx - uc) / W) ** 2 + ((yy - vc) / W) ** 2)
        g = np.exp(-(d / R) ** 2)
        # it dims as it contracts: a light going out, never a lamp left standing on the land
        amp = min(1.0, 1.7 * (R / 1.6) ** 0.8) * (1.0 - float(smoothstep(T_WHITE + 30, T_WHITE + 54, t)))
        k = np.clip(g * amp, 0.0, 1.0)[..., None]
        hdr = hdr * (1.0 - 0.95 * k) + np.array([6.0, 5.8, 5.4], np.float32) * (1.3 * k)
    return hdr


def finish_opts(tl, f):
    if f < T_BLACK:
        return dict(exposure=1.0, bloom_strength=0.05 if f > T_DEAD + 16 else 0.15, bloom_threshold=0.8,
                    streak_strength=0.0, vignette_amount=0.3)
    return dict(exposure=1.0, bloom_strength=0.1, bloom_threshold=0.7, streak_strength=0.0, vignette_amount=0.25)
