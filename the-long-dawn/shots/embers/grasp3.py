"""EMBERS-C: E8-C THE GRASP THAT CANNOT HOLD (C11, C frames 2320-2479).

    2320-2358  a claw of embers descends from the upper left, palm down (never a gauntlet: no plates, no metal; the
               crust of the forges, charcoal with fire in its seams), over the Ring hanging alone in the dark
    2360       (bar 30 b3) it closes on the Ring; gold leaks between the fingers
    2400       (bar 31 b1) the crust glows from inside and cracks along the hand's own lines
    2440       (bar 31 b3) the band slips through the fingers and falls away into the black

The claw is a SOLID: scene_c's hand skeleton (tapered capsules, smooth-unioned) sphere-traced per pixel, so its
silhouette is exact and dark; the shading is charcoal lit by the Ring's gold from inside the grip and a faint red
from far below, with scene_c.anat_cracks (tendons, knuckle creases, palm lines, plates) glowing in its seams and
opening as it fails. The Ring is ringsolid's canonical band. Timing and poses are scene_c's v3 grasp (its src
960-1040 plays over C 2320-2480 at half speed).
"""
import math

import cv2
import numpy as np
from numba import njit, prange

import look
from core import Camera, smoothstep, lerp, rng, vnoise
import scene_c as SC
import ringsolid as RS

T0, T1 = 2320, 2480
T_CLOSE, T_CRACK, T_SLIP = 2360, 2400, 2440
GRIP = np.array([0.0, 0.0, 0.0])        # the Ring hangs here (the claw's world is its own: no towers in frame)
RING_W = 2.6                            # the band's width here (outer diameter ~0.2 of the hand's length)


def srct(t):
    """C frame -> scene_c's v3 grasp time (src 960-1040 over C 2320-2480)"""
    return 960.0 + (t - T0) * 0.5


# ================================================================ ray marching ===

@njit(parallel=True, fastmath=True, cache=True)
def _march(CR, CP, f, W, H, x0, x1, y0, y1, HR, HW, HL, A, BA, IL2, RA, DR, GS, GE, KG, KJ, bc, br,
           DEP, NRM, LOC, AL):
    """sphere-trace the hand SDF (hand-local units: world = HW + HL * HR @ local). Per pixel: camera depth DEP (inf
    if no hit), hand-local normal NRM and position LOC, and a soft coverage AL (1 on the surface, fading with the
    ray's closest approach over one pixel's footprint: anti-aliased edges)."""
    for iy in prange(y0, y1):
        for ix in range(x0, x1):
            xc = (ix - (W - 1) * 0.5) / f
            yc = ((H - 1) * 0.5 - iy) / f
            dx = CR[0, 0] * xc + CR[1, 0] * yc + CR[2, 0]
            dy = CR[0, 1] * xc + CR[1, 1] * yc + CR[2, 1]
            dz = CR[0, 2] * xc + CR[1, 2] * yc + CR[2, 2]
            dl = math.sqrt(dx * dx + dy * dy + dz * dz)
            dx /= dl
            dy /= dl
            dz /= dl
            # into hand-local: q = HR^T (p - HW) / HL
            ox = (CP[0] - HW[0]) / HL
            oy = (CP[1] - HW[1]) / HL
            oz = (CP[2] - HW[2]) / HL
            rox = HR[0, 0] * ox + HR[1, 0] * oy + HR[2, 0] * oz
            roy = HR[0, 1] * ox + HR[1, 1] * oy + HR[2, 1] * oz
            roz = HR[0, 2] * ox + HR[1, 2] * oy + HR[2, 2] * oz
            rdx = HR[0, 0] * dx + HR[1, 0] * dy + HR[2, 0] * dz
            rdy = HR[0, 1] * dx + HR[1, 1] * dy + HR[2, 1] * dz
            rdz = HR[0, 2] * dx + HR[1, 2] * dy + HR[2, 2] * dz
            # bounding sphere
            lx = rox - bc[0]
            ly = roy - bc[1]
            lz = roz - bc[2]
            b = lx * rdx + ly * rdy + lz * rdz
            c = lx * lx + ly * ly + lz * lz - br * br
            disc = b * b - c
            if disc <= 0.0:
                continue
            sq = math.sqrt(disc)
            t0 = -b - sq
            t1 = -b + sq
            if t1 <= 0.0:
                continue
            if t0 < 0.0:
                t0 = 0.0
            tt = t0
            dmin = 1e9
            tmin = t0
            hit = False
            fp = 0.0
            for it in range(160):
                px = rox + rdx * tt
                py = roy + rdy * tt
                pz = roz + rdz * tt
                d, gx, gy, gz = SC._sdf(px, py, pz, A, BA, IL2, RA, DR, GS, GE, KG, KJ)
                fp = tt * HL / f / HL * 1.0 + 1e-6            # one pixel's footprint at this depth (local units)
                if d < dmin:
                    dmin = d
                    tmin = tt
                if d < 0.0006 + 0.25 * fp:
                    hit = True
                    NRM[iy, ix, 0] = gx
                    NRM[iy, ix, 1] = gy
                    NRM[iy, ix, 2] = gz
                    LOC[iy, ix, 0] = px
                    LOC[iy, ix, 1] = py
                    LOC[iy, ix, 2] = pz
                    DEP[iy, ix] = tt * HL * (CR[2, 0] * dx + CR[2, 1] * dy + CR[2, 2] * dz)
                    AL[iy, ix] = 1.0
                    break
                tt += d * 0.92
                if tt > t1:
                    break
            if not hit:
                fpm = tmin / f + 1e-6
                a = 1.0 - dmin / (1.2 * fpm)
                if a > 0.0:
                    # a grazing miss: the silhouette's anti-aliased fringe, shaded as the nearest surface point
                    px = rox + rdx * tmin
                    py = roy + rdy * tmin
                    pz = roz + rdz * tmin
                    d, gx, gy, gz = SC._sdf(px, py, pz, A, BA, IL2, RA, DR, GS, GE, KG, KJ)
                    NRM[iy, ix, 0] = gx
                    NRM[iy, ix, 1] = gy
                    NRM[iy, ix, 2] = gz
                    LOC[iy, ix, 0] = px - gx * d
                    LOC[iy, ix, 1] = py - gy * d
                    LOC[iy, ix, 2] = pz - gz * d
                    DEP[iy, ix] = tmin * HL * (CR[2, 0] * dx + CR[2, 1] * dy + CR[2, 2] * dz)
                    AL[iy, ix] = a


# ==================================================================== the claw ===

class Claw:
    def __init__(self):
        self.sk = SC.HandSkel()
        self.A_rest, self.R_rest = self.sk.pose(*SC.REST_POSE)

    def pose(self, t):
        """(A, R, kernel args, HR, HW) at C frame t"""
        ts = srct(t)
        A, R = self.sk.pose(*SC.hand_pose_v3(ts))
        HR, HW = hand_transform(t)
        return A, R, self.sk.kernel_args(A, R), HR, HW

    def owner(self, LOC, A, args):
        """the bone each hand-local point belongs to (nearest capsule)"""
        A_, BA, IL2, RA, DR = args[0], args[1], args[2], args[3], args[4]
        best = np.full(len(LOC), 1e9)
        own = np.zeros(len(LOC), np.int64)
        for j in range(self.sk.M):
            q = LOC - A_[j]
            h = np.clip((q @ BA[j]) * IL2[j], 0, 1)
            c = q - h[:, None] * BA[j][None, :]
            d = np.linalg.norm(c, axis=1) - (RA[j] + DR[j] * h)
            m = d < best
            best[m] = d[m]
            own[m] = j
        return own

    def to_rest(self, LOC, own, A, R):
        """posed hand-local points -> the rest pose (the crack pattern lives there, so it moves with the flesh)"""
        q = np.einsum('nij,ni->nj', R[own], LOC - A[own])            # into each bone's own frame
        return self.A_rest[own] + np.einsum('nij,nj->ni', self.R_rest[own], q)


def hand_transform(t):
    """world placement of the claw (rotation hand-local -> world, wrist position), scene_c's v3 path about GRIP"""
    old = SC.GRIP_OVERRIDE
    SC.GRIP_OVERRIDE = GRIP
    try:
        R, W = SC.hand_transform_v3(srct(t))
    finally:
        SC.GRIP_OVERRIDE = old
    return R, W


FALL_G = 0.13                          # the fall's gravity (units / C frame^2): it drops out of frame in ~1 s


def ring_frame(t):
    """(Rot, centre) of the Ring: hanging, gripped, slipping out between the middle and ring fingers, and then
    falling away into the black (tumbling; out of frame by ~2470)"""
    old = SC.GRIP_OVERRIDE
    SC.GRIP_OVERRIDE = GRIP
    try:
        C, _ = SC.ring_state_v3(min(srct(t), srct(T_SLIP + 6)))
        h = SC._gr_h()
    finally:
        SC.GRIP_OVERRIDE = old
    # a three-quarter view (face-on it reads as a halo): its axis 38 degrees off vertical, toward the lens
    import scene_b as B
    a = math.atan2(h[2], h[0])
    Rm = B._rot_about(a + 0.5, math.radians(38.0))
    if t > T_SLIP:
        x = t - T_SLIP
        tum = 0.07 * x + 0.004 * x * x                                   # it tumbles as it goes
        up = np.array([0.0, 1.0, 0.0])
        ax = np.cross(up, h)
        ax /= np.linalg.norm(ax)
        K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
        Rm = (np.eye(3) + math.sin(tum) * K + (1 - math.cos(tum)) * (K @ K)) @ Rm
    if t > T_SLIP + 6:
        x = t - (T_SLIP + 6)
        C = C + np.array([0.0, -1.0, 0.0]) * (0.3 * x + 0.5 * FALL_G * x * x) + np.array([0.6, 0.0, -0.4]) * (0.2 * x)
    return Rm, C


CAM_D = (126.0, 112.0)                 # distance to the grip: a slow push in
CAM_EL = (24.0, 19.0)                  # elevation (degrees): above the claw, looking down past it into the dark


def camera(t):
    u = float(np.clip((t - T0) / (T1 - T0), 0, 1))
    k = u * u * (3 - 2 * u)
    d = lerp(CAM_D[0], CAM_D[1], k)
    el = math.radians(lerp(CAM_EL[0], CAM_EL[1], k))
    old = SC.GRIP_OVERRIDE
    SC.GRIP_OVERRIDE = GRIP
    try:
        h = SC._gr_h()
    finally:
        SC.GRIP_OVERRIDE = old
    up = np.array([0.0, 1.0, 0.0])
    side = np.cross(up, h)
    pos = GRIP + d * (h * math.cos(el) + up * math.sin(el)) - side * 6.0
    tgt = GRIP + np.array([0.0, 7.0, 0.0]) - side * 4.0
    if t > T_CRACK + 6:
        kk = min((t - (T_CRACK + 6)) / 12.0, 1.0)
        pos = pos + 0.3 * kk * np.array([math.sin(1.7 * t), 0.6 * math.sin(2.3 * t), math.cos(1.3 * t)])
    return Camera(pos, tgt, hfov=52.0, focus=float(np.linalg.norm(GRIP - pos)), aperture=0.0)


# ================================================================== shading ===

def heat_t(t):
    """0..1: how far the crust has failed (it glows from inside from bar 31, blazing by the slip)"""
    return float(smoothstep(T_CRACK - 4, T_SLIP + 10, t))


def shade_claw(claw, t, cam, DEP, NRM, LOC, AL, A, R, args, HR, HW, ring_c):
    """-> (rgb (H, W, 3), alpha (H, W)) of the claw"""
    H, W = DEP.shape
    m = AL > 0.0
    rgb = np.zeros((H, W, 3), np.float32)
    if not m.any():
        return rgb, AL
    L_ = LOC[m]
    Nl = NRM[m]
    own = claw.owner(L_, A, args)
    rest = claw.to_rest(L_, own, A, R)
    sharp, soft, plate, cre = SC.anat_cracks(rest, own, claw.sk, claw.A_rest, claw.R_rest)
    Pw = HW[None, :] + SC.HAND_L * L_ @ HR.T
    Nw = Nl @ HR.T
    Nw /= np.maximum(np.linalg.norm(Nw, axis=1, keepdims=True), 1e-9)
    V = cam.pos[None, :] - Pw
    V /= np.maximum(np.linalg.norm(V, axis=1, keepdims=True), 1e-9)
    ndv = np.clip((Nw * V).sum(1), 0, 1)
    hk = heat_t(t)
    close = float(smoothstep(T_CLOSE - 8, T_CLOSE + 4, t))
    # the Ring's gold: from its place (inside the grip once it closes), strongest on the fingers' inner faces
    Lr = ring_c[None, :] - Pw
    dr = np.linalg.norm(Lr, axis=1)
    Lr /= np.maximum(dr, 1e-9)[:, None]
    lam = np.clip((Nw * Lr).sum(1), 0, 1)
    gold = np.array([1.0, 0.72, 0.3])
    ring_light = 55.0 * lam / (1.0 + (dr / 5.0) ** 2) * (0.35 + 0.65 * close) * (1.0 - 0.8 * float(smoothstep(T_SLIP, T_SLIP + 16, t)))
    # the red of the fires far below, and a rim of it round the silhouette
    below = np.clip(-Nw[:, 1], 0, 1) ** 1.5
    rim = (1.0 - ndv) ** 3
    alb = 0.03 * (0.8 + 0.4 * plate)
    col = (gold[None, :] * (alb * ring_light)[:, None]
           + np.array([0.9, 0.2, 0.06])[None, :] * (0.05 * below + 0.12 * rim)[:, None])
    # the fire in its seams: faint embers while it grips; as it fails the cracks open and blaze while the plates
    # between them stay charcoal (a coal about to break apart, never a lantern)
    crack = sharp ** (2.2 - 1.2 * hk)                                   # thin lines that widen as it fails
    seam = crack * (0.1 + 1.3 * hk) + soft * 0.15 * hk ** 2
    T = np.clip(0.4 + 0.28 * crack + 0.12 * hk, 0, 0.8)
    ember = look.blackbody(T) * (seam * (0.6 + 1.2 * hk))[:, None]
    crease = cre ** 1.5 * (0.05 + 1.4 * hk)
    col = col + ember + look.blackbody(np.full(len(cre), 0.62))[None, 0] * crease[:, None]
    # a dull red through the thin crust at the very edges as it fails
    glowin = hk ** 2 * 0.06 * rim
    col = col + np.array([1.0, 0.3, 0.06])[None, :] * glowin[:, None]
    rgb[m] = col.astype(np.float32)
    return rgb, AL


# ================================================================== the layer ===

class Grasp:
    def __init__(self):
        self.claw = Claw()
        r = rng(4551)
        n = 5000
        self.e_p = np.stack([r.uniform(-60, 60, n), r.uniform(-90, 20, n), r.uniform(-60, 60, n)], 1)
        self.e_v = r.uniform(0.06, 0.2, n)
        self.e_E = r.lognormal(0, 0.6, n)

    def embers(self, ctx):
        """a few embers drifting up out of the dark below (the fires far down): no vortex, no storm eye"""
        def pos(t):
            P = self.e_p.copy()
            P[:, 1] = ((P[:, 1] + self.e_v * (t - T0)) + 90.0) % 110.0 - 90.0
            P = P + vnoise(P * 0.05 + np.array([0, -0.01 * t, 0]), 0.6, (1.0, 2.0, 3.0), 1) * 2.0
            return P
        P0, P1 = pos(ctx.t0), pos(ctx.t1)
        e = self.e_E * 3.0
        col = look.blackbody(np.full(len(e), 0.55))
        ctx.fr.splat(P0, P1, 0.08, e, col, ctx.cam0, ctx.cam1)

    def emit(self, ctx):
        self.embers(ctx)
        # the glow far below: the fires of the race, seen down through the dark
        C = GRIP + np.array([0.0, -140.0, 0.0])
        ctx.fr.splat(C[None, :], C[None, :], np.array([60.0]), np.array([2.5e4]), np.array([[1.0, 0.25, 0.06]]),
                     ctx.cam0, ctx.cam1, profile=1)

    def post(self, ctx, hdr):
        t = ctx.t
        cam = ctx.cam
        H, W = hdr.shape[:2]
        # --- the Ring (three instants across the shutter: it falls fast after the slip)
        acc = None
        for tq in (ctx.t0, 0.5 * (ctx.t0 + ctx.t1), ctx.t1):
            Rot, C = ring_frame(tq)
            st = RS.RingState()
            st.letters = 1.1 * (1.0 - 0.4 * float(smoothstep(T_SLIP, T_SLIP + 30, tq)))
            st.glow = 0.05
            st.glow_col = np.array([1.0, 0.8, 0.4])
            env = RS.Env(above=(0.12, 0.08, 0.035), horizon=(0.42, 0.27, 0.11), below=(0.3, 0.14, 0.05))
            env.point(GRIP + np.array([0.0, -140.0, 0.0]), np.array([6.0, 1.6, 0.4]), 40.0)
            # the claw's own glow (its seams, blazing as it fails) is what the gold mirrors from above
            _, _, _, HRq, HWq = self.claw.pose(tq)
            palm = HWq + SC.HAND_L * HRq @ np.array([0.0, 0.35, 0.05])
            env.point(palm, np.array([7.0, 4.2, 1.6]) * (0.4 + 2.5 * heat_t(tq)), 14.0)
            rgb_r, a_r, d_r = RS.render(Camera(cam.pos, cam.target, hfov=cam.hfov), W, H, Rot, C, RING_W / 1.0, st, env)
            if acc is None:
                acc = [rgb_r, a_r, d_r]
            else:
                acc[0] += rgb_r
                acc[1] += a_r
                acc[2] = np.minimum(acc[2], d_r)
        rgb_r, a_r, d_r = acc[0] / 3.0, acc[1] / 3.0, acc[2]
        Rot, ring_c = ring_frame(t)
        # --- the claw
        A, R, args, HR, HW = self.claw.pose(t)
        DEP = np.full((H, W), np.inf, np.float32)
        NRM = np.zeros((H, W, 3), np.float32)
        LOC = np.zeros((H, W, 3), np.float32)
        AL = np.zeros((H, W), np.float32)
        # screen box of the claw: its bone ends, padded by their radii
        ends = np.concatenate([args[0], args[0] + args[1]])
        Pw = HW[None, :] + SC.HAND_L * ends @ HR.T
        u, v, z = cam.project(Pw, W, H)
        ok = z > 0.5
        if ok.any():
            pad = SC.HAND_L * 0.12 * cam.f_px(W) / max(float(z[ok].min()), 1.0)
            x0 = int(max(u[ok].min() - pad, 0))
            x1 = int(min(u[ok].max() + pad, W))
            y0 = int(max(v[ok].min() - pad, 0))
            y1 = int(min(v[ok].max() + pad, H))
            ctr = 0.5 * (ends.min(0) + ends.max(0))
            rad = float(np.linalg.norm(ends - ctr, axis=1).max() + 0.12)
            if x1 > x0 and y1 > y0:
                _march(np.ascontiguousarray(cam.R, np.float64), np.ascontiguousarray(cam.pos, np.float64),
                       float(cam.f_px(W)), W, H, x0, x1, y0, y1, np.ascontiguousarray(HR, np.float64),
                       np.ascontiguousarray(HW, np.float64), float(SC.HAND_L), *args, ctr.astype(np.float64), rad,
                       DEP, NRM, LOC, AL)
        rgb_h, a_h = shade_claw(self.claw, t, cam, DEP, NRM, LOC, AL, A, R, args, HR, HW, ring_c)
        # --- composite: the nearer of the Ring and the claw per pixel; gold leaks round the grip
        ring_front = d_r < DEP
        a_r_eff = np.where(ring_front, a_r, a_r * (1.0 - a_h))
        a_h_eff = np.where(ring_front, a_h * (1.0 - a_r), a_h)
        out = hdr * (1.0 - np.clip(a_r_eff + a_h_eff, 0, 1))[..., None]
        out = out + rgb_r * np.where(ring_front, 1.0, 1.0 - a_h)[..., None] + rgb_h * np.where(ring_front, 1.0 - a_r, 1.0)[..., None]
        # the leak: the Ring's light through the gaps between the fingers (a glow the claw mostly hides)
        close = float(smoothstep(T_CLOSE - 6, T_CLOSE + 6, t)) * (1.0 - float(smoothstep(T_SLIP, T_SLIP + 10, t)))
        if close > 0:
            uu, vv, zz = cam.project(ring_c[None, :], W, H)
            if zz[0] > 0.5:
                yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
                rpx = RING_W * 2.25 * cam.f_px(W) / zz[0]
                g = np.exp(-((xx - uu[0]) ** 2 + (yy - vv[0]) ** 2) / (2 * (0.75 * rpx) ** 2))
                leak = g * (1.0 - a_h) * close * 1.4
                out = out + leak[..., None] * np.array([1.0, 0.72, 0.3], np.float32)
        return out
