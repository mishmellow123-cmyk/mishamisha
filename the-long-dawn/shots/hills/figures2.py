"""The Elder and the Child, v2 (CODA dept, Sep 2026): cloth-and-body silhouettes for the
merged-silhouette renderer (silhouette.py).

Same pose dictionaries and anchors as characters.elder / characters.child (so intro.py and
coda.py keep every keyframe), but the shapes are drawn as a costume designer would cut them
for a shadow play:

* the Elder's long wool coat hangs from the shoulders and the chest (not from the spine): a
  rounded upper back, the front falling straight from the bust, a flared back hem that lifts
  and folds in the wind (secondary motion), heavy sleeves with a cuff and an elbow crease,
  hands that grip (knuckles, thumb), ankle boots with a heel, a profile with brow, nose, lips
  and a soft jaw, grey hair in a bun with loose wisps;
* the Child's quilted puffer jacket (the baffles scallop the outline), a hood lying on the
  back, puffy sleeves that merge into the body, mittens with thumbs, snow trousers bunching
  over chunky boots, a knitted beanie with a turned-up cuff and a fuzzy pom-pom;
* the scarf is cloth: two bulky loops, a short front end and a long tail that tapers, twists
  (its face turns into and out of the firelight), ripples along its length and ends in a
  fringe of nine yarn tassels. Deep red wool, the same wool as the young woman's (heroine.py).

Everything is built facing LEFT (-x) in card-local metres, origin on the ground between the
feet, then mirrored for facing=+1. Angle conventions as characters.py (dirv(0) = up,
dirv(90) = forward/-x).
"""
import math

import numpy as np

from characters import dirv, backv, rot, ang_of, ik2, lerp, face_poly
from core import fnoise1
from puppet import Group, P_ELL, P_CONE, P_POLY, P_BOX, P_RING

# ------------------------------------------------------------- materials ---
# near-black silhouettes; the rims are the only light on them (sheen <= 0.4 for cloth and skin)
MAT2 = {
    'coat': dict(albedo=(0.0012, 0.0012, 0.0016), bevel=0.05, sheen=0.34, soft=0.05, sky_rim=0.55,
                 tint=(1.0, 0.78, 0.58), aa=1.0, fuzz=0.0007, fuzz_scale=0.004),
    'jacket': dict(albedo=(0.0014, 0.0016, 0.0026), bevel=0.05, sheen=0.34, soft=0.05, sky_rim=0.55,
                   tint=(1.0, 0.80, 0.62), aa=1.0, fuzz=0.0005, fuzz_scale=0.004),
    'skin': dict(albedo=(0.0016, 0.0012, 0.0011), bevel=0.02, sheen=0.30, soft=0.0, sky_rim=0.45,
                 tint=(1.0, 0.70, 0.50), aa=1.0),
    'hair_grey': dict(albedo=(0.004, 0.004, 0.0045), bevel=0.02, sheen=0.75, soft=0.03, sky_rim=0.95,
                      tint=(1.0, 0.92, 0.84), aa=1.25, fuzz=0.0012, fuzz_scale=0.003),
    'hair_wisp': dict(albedo=(0.006, 0.006, 0.0065), bevel=0.01, sheen=0.45, soft=0.0, sky_rim=0.6,
                      tint=(1.0, 0.92, 0.84), aa=1.8, translucent=0.3, cast=0),
    'hair_dark': dict(albedo=(0.0014, 0.0011, 0.0010), bevel=0.015, sheen=0.4, soft=0.02, sky_rim=0.7,
                      tint=(1.0, 0.72, 0.50), aa=1.2, fuzz=0.0010, fuzz_scale=0.003),
    'wool': dict(albedo=(0.0024, 0.0019, 0.0015), bevel=0.03, sheen=0.34, soft=0.05, sky_rim=0.85,
                 tint=(1.0, 0.86, 0.66), aa=1.5, fuzz=0.0014, fuzz_scale=0.0035),
    'pom': dict(albedo=(0.0030, 0.0024, 0.0018), bevel=0.03, sheen=0.36, soft=0.06, sky_rim=1.0,
                tint=(1.0, 0.86, 0.66), aa=2.2, fuzz=0.0030, fuzz_scale=0.004),
    'boot': dict(albedo=(0.0010, 0.0010, 0.0010), bevel=0.03, sheen=0.25, soft=0.0, sky_rim=0.40,
                 tint=(1.0, 0.8, 0.6), aa=1.0),
    'mitt': dict(albedo=(0.0020, 0.0018, 0.0016), bevel=0.02, sheen=0.34, soft=0.04, sky_rim=0.6,
                 tint=(1.0, 0.84, 0.64), aa=1.3, fuzz=0.0008, fuzz_scale=0.003),
    # the scarf: deep red wool (heroine.py M_SCARF albedo), lit by the fire, fuzzy, backlit
    'scarf': dict(albedo=(0.29, 0.022, 0.019), mode=1, bevel=0.022, sheen=0.30, soft=0.06, sky_rim=0.35,
                  tint=(1.0, 0.30, 0.18), diffuse=0.85, translucent=0.30, aa=1.3, tex=1, tex_scale=0.009,
                  tex_amp=0.32, fuzz=0.0008, fuzz_scale=0.005),
    'scarf_tail': dict(albedo=(0.29, 0.022, 0.019), mode=2, bevel=0.02, sheen=0.30, soft=0.05, sky_rim=0.35,
                       tint=(1.0, 0.30, 0.18), diffuse=0.85, translucent=0.40, aa=1.3, tex=2, tex_scale=0.009,
                       tex_amp=0.32, fuzz=0.0008, fuzz_scale=0.005),
    'fringe': dict(albedo=(0.29, 0.022, 0.019), mode=1, bevel=0.004, sheen=0.30, soft=0.0, sky_rim=0.45,
                   tint=(1.0, 0.30, 0.18), diffuse=0.7, translucent=0.55, aa=1.5, cast=0),
    'torch': dict(albedo=(0.004, 0.0028, 0.0018), bevel=0.012, sheen=0.45, soft=0.04, sky_rim=0.4,
                  tint=(1.0, 0.8, 0.6), aa=1.0, cast=0),
    'rag': dict(albedo=(0.006, 0.0040, 0.0025), bevel=0.012, sheen=0.5, soft=0.08, sky_rim=0.4,
                tint=(1.0, 0.75, 0.5), aa=1.2, fuzz=0.0015, fuzz_scale=0.004, cast=0),
}


def G2(name, mat, k=0.01, **kw):
    d = dict(MAT2[mat])
    d.update(kw)
    extra = {}
    for key in ('mode', 'aa', 'tex', 'tex_scale', 'tex_amp', 'fuzz', 'fuzz_scale', 'shadow', 'union', 'cast'):
        if key in d:
            extra[key] = d.pop(key)
    g = Group(name, k=k, **d)
    for key, v in extra.items():
        setattr(g, key, v)
    return g


# --------------------------------------------------------------- helpers ---

def crspline(P, per=8, closed=True, alpha=0.5):
    """Centripetal Catmull-Rom through control points P (n,2) -> dense polyline."""
    P = np.asarray(P, np.float64)
    n = len(P)
    if closed:
        Q = np.concatenate([P[-1:], P, P[:2]], 0)
        segs = n
    else:
        Q = np.concatenate([P[:1] * 2 - P[1:2], P, P[-1:] * 2 - P[-2:-1]], 0)
        segs = n - 1
    out = []
    for s in range(segs):
        p0, p1, p2, p3 = Q[s], Q[s + 1], Q[s + 2], Q[s + 3]
        t0 = 0.0
        t1 = t0 + max(1e-6, np.linalg.norm(p1 - p0)) ** alpha
        t2 = t1 + max(1e-6, np.linalg.norm(p2 - p1)) ** alpha
        t3 = t2 + max(1e-6, np.linalg.norm(p3 - p2)) ** alpha
        for k in range(per):
            t = t1 + (t2 - t1) * k / per
            a1 = (t1 - t) / (t1 - t0) * p0 + (t - t0) / (t1 - t0) * p1
            a2 = (t2 - t) / (t2 - t1) * p1 + (t - t1) / (t2 - t1) * p2
            a3 = (t3 - t) / (t3 - t2) * p2 + (t - t2) / (t3 - t2) * p3
            b1 = (t2 - t) / (t2 - t0) * a1 + (t - t0) / (t2 - t0) * a2
            b2 = (t3 - t) / (t3 - t1) * a2 + (t - t1) / (t3 - t1) * a3
            out.append((t2 - t) / (t2 - t1) * b1 + (t - t1) / (t2 - t1) * b2)
    if not closed:
        out.append(P[-1])
    return np.array(out)


def outline_normals(poly):
    """Outward-ish normals of a closed polyline (assumes counter-clockwise or fixes sign)."""
    P = np.asarray(poly)
    T = np.roll(P, -1, 0) - np.roll(P, 1, 0)
    T /= np.linalg.norm(T, axis=1, keepdims=True) + 1e-12
    N = np.stack([T[:, 1], -T[:, 0]], 1)
    area = 0.5 * np.sum(P[:, 0] * np.roll(P[:, 1], -1) - np.roll(P[:, 0], -1) * P[:, 1])
    if area < 0:
        N = -N
    return N


def quilt(poly, y0, period, amp, mask_fn=None):
    """Scallop an outline: bulge between horizontal stitch lines, pinch at them."""
    P = np.asarray(poly, np.float64).copy()
    N = outline_normals(P)
    ph = np.abs(np.sin(math.pi * (P[:, 1] - y0) / period))
    off = amp * (ph ** 0.6 - 0.62)
    side = np.abs(N[:, 0])                  # only where the outline runs up/down
    if mask_fn is not None:
        side = side * mask_fn(P)
    P += N * (off * side)[:, None]
    return P


def perp(d):
    return np.array([-d[1], d[0]])


def unit(v):
    v = np.asarray(v, np.float64)
    return v / (np.linalg.norm(v) + 1e-12)


def mirror2(groups, cx):
    """Mirror primitives about x=cx (facing right); also per-primitive face normals."""
    for g in groups:
        for row in g.rows:
            typ = int(row[0])
            if typ in (P_ELL, P_RING):
                row[2] = 2 * cx - row[2]
                row[6] = -row[6]
            elif typ == P_CONE:
                row[2] = 2 * cx - row[2]
                row[4] = 2 * cx - row[4]
            elif typ == P_BOX:
                row[2] = 2 * cx - row[2]
                row[6] = -row[6]
        for v in g.verts:
            v[0] = 2 * cx - v[0]
        g.bbs = [(2 * cx - b[2], b[1], 2 * cx - b[0], b[3]) for b in g.bbs]
        pa = getattr(g, 'prim_attr', None)
        if pa is not None:
            pa = np.asarray(pa, np.float64).copy()
            pa[:, 1] = -pa[:, 1]
            if pa.shape[1] > 6:
                pa[:, 4] = 2 * cx - pa[:, 4]
                pa[:, 6] = -pa[:, 6]
            g.prim_attr = pa
    return groups


# ------------------------------------------------------------------ hands ---

def fist(g, W, u_ang, s=1.0, k=0.006):
    """A gripping hand from wrist W, knuckles toward dirv(u_ang): back of hand, the rolled
    fingers (knuckle bumps) and the thumb over them."""
    u = dirv(u_ang)
    v = perp(u)                          # palm side (toward the thumb when facing left)
    W = np.asarray(W, np.float64)
    g.box(W + u * 0.036 * s, 0.036 * s, 0.026 * s, rot=u_ang + 90.0, rnd=0.012 * s, k=k)
    fc = W + u * 0.068 * s + v * 0.004 * s
    g.ellipse(fc, 0.024 * s, 0.031 * s, rot=u_ang + 90.0, k=k)
    for q, o in enumerate((-0.022, -0.007, 0.008, 0.021)):
        g.circle(W + u * (0.083 - 0.004 * abs(o) / 0.02) * s + v * o * s, (0.0105 - 0.0012 * (q == 3)) * s, k=k * 0.8)
    g.cone(W + u * 0.018 * s + v * 0.026 * s, W + u * 0.060 * s + v * 0.034 * s, 0.0115 * s, 0.0092 * s, k=k)


def open_hand(g, W, u_ang, s=1.0, curl=0.35, k=0.005):
    """A hand resting (on a shoulder): palm, four slightly curled fingers, the thumb."""
    u = dirv(u_ang)
    v = perp(u)
    W = np.asarray(W, np.float64)
    g.box(W + u * 0.040 * s, 0.040 * s, 0.028 * s, rot=u_ang + 90.0, rnd=0.012 * s, k=k)
    for q, o in enumerate((-0.020, -0.007, 0.006, 0.018)):
        L1 = (0.040 - 0.004 * abs(q - 1.5)) * s
        a0 = W + u * 0.075 * s + v * o * s
        d1 = dirv(u_ang + curl * 25.0 * (1 + 0.2 * q))
        a1 = a0 + d1 * L1
        d2 = dirv(u_ang + curl * 60.0)
        a2 = a1 + d2 * L1 * 0.7
        g.cone(a0, a1, 0.0085 * s, 0.0075 * s, k=k * 0.6)
        g.cone(a1, a2, 0.0075 * s, 0.0060 * s, k=k * 0.6)
    tb = W + u * 0.020 * s + v * 0.028 * s
    g.cone(tb, tb + dirv(u_ang + 55.0) * 0.045 * s, 0.011 * s, 0.008 * s, k=k)


def mitten(g, W, u_ang, s=1.0, k=0.008):
    u = dirv(u_ang)
    v = perp(u)
    W = np.asarray(W, np.float64)
    g.ellipse(W + u * 0.040 * s, 0.036 * s, 0.028 * s, rot=u_ang + 90.0, k=k)
    g.ellipse(W + u * 0.020 * s, 0.024 * s, 0.026 * s, rot=u_ang + 90.0, k=k)
    g.cone(W + u * 0.020 * s + v * 0.022 * s, W + u * 0.046 * s + v * 0.034 * s, 0.012 * s, 0.010 * s, k=k)


# ------------------------------------------------------------ the scarf ---

def scarf_tail(g_tail, g_fringe, pts, t, width=0.11, thick=0.008, seed=3.0, rip_amp=1.0):
    """The long tail from the verlet chain: a twisting, rippling, tapering ribbon of quads
    (per-segment face normals for the firelight) + nine fringe tassels."""
    pts = np.asarray(pts, np.float64)
    C = crspline(pts, per=3, closed=False)
    n = len(C)
    s_arr = np.linspace(0.0, 1.0, n)
    T = np.gradient(C, axis=0)
    T /= np.linalg.norm(T, axis=1, keepdims=True) + 1e-12
    N2 = np.stack([-T[:, 1], T[:, 0]], 1)
    # secondary motion: ripples travelling down the tail (on top of the chain's flutter)
    rip = (0.016 * np.sin(s_arr * 13.0 - t * 13.0) + 0.007 * np.sin(s_arr * 29.0 - t * 23.0)) * (0.2 + s_arr) * rip_amp
    C = C + N2 * rip[:, None]
    tw = np.array([1.35 * fnoise1(s * 3.1 - t * 1.9, seed, 2) + 0.30 for s in s_arr])
    wfull = width * (0.80 + 0.20 * (1 - s_arr)) * (0.55 + 0.45 * np.minimum(1.0, 3.0 * s_arr + 0.3))
    half = 0.5 * np.maximum(thick, wfull * np.abs(np.cos(tw)))
    Lp = C + N2 * half[:, None]
    Rp = C - N2 * half[:, None]
    attrs = []
    eps = 0.0012
    for i in range(n - 1):
        # hard-edged quads, each overlapping its neighbours along the centreline (no notches
        # at the joints, no half-covered seam lines)
        ti = unit(C[i + 1] - C[i])
        e0 = eps if i > 0 else 0.0
        e1 = eps if i < n - 2 else 0.0
        quad = [Lp[i] - ti * e0, Lp[i + 1] + ti * e1, Rp[i + 1] + ti * e1, Rp[i] - ti * e0]
        g_tail.poly(quad, r=0.0, k=0.0)
        twm = 0.5 * (tw[i] + tw[i + 1])
        tm = unit(T[i] + T[i + 1])
        sn = math.sin(twm)
        cm = 0.5 * (C[i] + C[i + 1])
        hm = 0.5 * (half[i] + half[i + 1])
        wm = 0.5 * (wfull[i] + wfull[i + 1])
        # [flag, face normal xyz, centre xy, tangent xy, apparent half-width, true half-width]
        attrs.append([1.0, sn * tm[1], -sn * tm[0], math.cos(twm), cm[0], cm[1], tm[0], tm[1], hm, 0.5 * wm])
    g_tail.prim_attr = np.array(attrs)
    # fringe: yarn tassels off the end, following the end's width direction
    a, b = C[-2], C[-1]
    tg = unit(b - a)
    nn = perp(tg)
    wend = half[-1]
    for k in range(9):
        off = (k - 4) / 4.0 * wend * 0.92
        root = b + nn * off - tg * 0.004
        L = 0.065 + 0.012 * ((k * 7) % 3)
        wig = 0.012 * fnoise1(t * 3.1 + k * 1.37, 7.0, 2)
        mid = root + tg * L * 0.5 + nn * (off * 0.1 + wig)
        tip = root + tg * L + nn * (off * 0.15 + wig * 2.2)
        g_fringe.cone(root, mid, 0.0034, 0.0027, k=0.003)
        g_fringe.cone(mid, tip, 0.0027, 0.0012, k=0.002)
    return C


def short_end(g, top, t, hw, length=0.17, width=0.085, seed=9.0):
    """The short front end of the scarf hanging down the chest (one smooth panel that lifts and
    sways in the wind; a slight flare and a soft, uneven end)."""
    pts = []
    for q in range(5):
        s = q / 4.0
        sway = hw * 0.5 * s * s + 0.010 * fnoise1(t * 1.3 + s, seed, 2) * s
        pts.append(top + np.array([-0.012 * s + sway, -length * s]))
    pts = np.array(pts)
    T = np.gradient(pts, axis=0)
    T /= np.linalg.norm(T, axis=1, keepdims=True)
    N2 = np.stack([-T[:, 1], T[:, 0]], 1)
    hwid = 0.5 * width * np.array([0.82, 0.95, 1.0, 1.04, 1.06])
    L = pts + N2 * hwid[:, None]
    R = pts - N2 * hwid[:, None]
    end = [L[-1] + T[-1] * 0.004, pts[-1] + T[-1] * (0.010 + 0.004 * fnoise1(t, seed + 2, 1)), R[-1] + T[-1] * 0.006]
    outline = np.concatenate([L, end, R[::-1]], 0)
    g.poly(crspline(outline, per=3, closed=True), r=0.002, k=0.012)
    return pts[-1]


# ------------------------------------------------------------------ ELDER ---

ELDER_FACE2 = [(-0.010, 0.084), (-0.052, 0.074), (-0.071, 0.052), (-0.078, 0.030), (-0.082, 0.016),
               (-0.080, 0.006), (-0.076, -0.001), (-0.080, -0.008), (-0.090, -0.020), (-0.101, -0.033),
               (-0.104, -0.039), (-0.099, -0.044), (-0.088, -0.045), (-0.087, -0.050), (-0.088, -0.055),
               (-0.083, -0.059), (-0.086, -0.063), (-0.084, -0.068), (-0.078, -0.073), (-0.080, -0.081),
               (-0.078, -0.090), (-0.068, -0.099), (-0.052, -0.104), (-0.034, -0.100), (-0.016, -0.090),
               (0.004, -0.060), (0.010, 0.000), (0.004, 0.050)]

ELDER_DEFAULT = dict(x=0.0, hip_y=0.82, lumbar=6.0, thorax=30.0, neck=40.0, head=4.0,
                     n_ua=176.0, n_fa=150.0, n_h=160.0,
                     f_ua=158.0, f_fa=70.0, f_h=40.0, torch_ang=6.0,
                     n_th=182.0, n_sh=178.0, f_th=176.0, f_sh=183.0,
                     breath=0.0, hem_wind=0.05, torch=True, hand_n='fist', hand_f='fist')


def _elder_skeleton(p):
    x0 = p['x']
    P0 = np.array([x0, p['hip_y'] + 0.004 * p['breath']])
    thl, tht, thn, thh = p['lumbar'], p['thorax'] + 1.5 * p['breath'], p['neck'], p['head']
    P1 = P0 + dirv(thl) * 0.20
    P2 = P1 + dirv(tht) * 0.24
    P3 = P2 + dirv(thn) * 0.06
    head_c = P3 + rot(np.array([-0.014, 0.068]), thh)
    return x0, P0, P1, P2, P3, head_c, thl, tht, thn, thh


def elder_anchor_pts(p):
    x0, P0, P1, P2, P3, head_c, thl, tht, thn, thh = _elder_skeleton(p)
    sa = P2 + backv(tht) * 0.062 + dirv(thn) * 0.045
    bun = head_c + rot(np.array([0.078, 0.046]), thh)
    return sa, bun


def elder2(pose, t, scarf_pts=None, facing=-1, wisps=None, anchors_only=False, hem_pts=None):
    """Returns (groups, anchors) - same contract as characters.elder."""
    p = dict(ELDER_DEFAULT)
    p.update(pose)
    x0, P0, P1, P2, P3, head_c, thl, tht, thn, thh = _elder_skeleton(p)
    if anchors_only:
        sa, bun = elder_anchor_pts(p)
        if facing == 1:
            sa = np.array([2 * x0 - sa[0], sa[1]])
            bun = np.array([2 * x0 - bun[0], bun[1]])
        return None, dict(scarf_anchor=sa, bun=bun)
    F = lambda th: -backv(th)
    B = lambda th: backv(th)
    hw = p['hem_wind']
    body = G2('elder_coat', 'coat', k=0.02)
    farg = G2('elder_far', 'coat', k=0.02)
    near = G2('elder_near', 'coat', k=0.02)
    skin = G2('elder_skin', 'skin', k=0.006)
    handg = G2('elder_hands', 'skin', k=0.005, cast=0)
    hair = G2('elder_hair', 'hair_grey', k=0.010)
    wisp = G2('elder_wisps', 'hair_wisp', k=0.003)
    wrap = G2('elder_scarfwrap', 'scarf', k=0.02)
    roll2 = G2('elder_scarfroll2', 'scarf', k=0.02)
    knot = G2('elder_scarfknot', 'scarf', k=0.012)
    tail = G2('elder_scarftail', 'scarf_tail', k=0.004)
    fringe = G2('elder_fringe', 'fringe', k=0.003)
    legs = G2('elder_legs', 'boot', k=0.012)
    torch_g = G2('torch', 'torch', k=0.004)
    rag = G2('torch_rag', 'rag', k=0.006)
    # ---- legs + ankle boots (only the calves show below the hem)
    for (dx, th, sh, sc) in [(0.030, p['f_th'], p['f_sh'], 0.96), (-0.030, p['n_th'], p['n_sh'], 1.0)]:
        hp = P0 + np.array([dx, -0.03])
        K = hp + dirv(th) * 0.40
        A = K + dirv(sh) * 0.36
        A[1] = max(A[1], 0.075)
        legs.cone(K, A, 0.044 * sc, 0.029 * sc)
        ax, ay = A
        boot = [(ax + 0.036, 0.004), (ax + 0.040, 0.030), (ax + 0.034, ay + 0.030), (ax + 0.004, ay + 0.052),
                (ax - 0.030, ay + 0.030), (ax - 0.066, 0.052), (ax - 0.110, 0.038), (ax - 0.128, 0.020),
                (ax - 0.124, 0.002), (ax - 0.060, 0.0), (ax + 0.010, 0.0)]
        legs.poly(crspline(boot, per=4), r=0.004, k=0.012)
    # ---- the coat: hangs from the bust and the rounded back; a soft waist; the wind presses
    # the front against the knees and lifts/folds the back hem (secondary motion)
    fl = 0.010 * fnoise1(t * 2.3, 11.0, 2)
    fl2 = 0.008 * fnoise1(t * 3.1, 13.0, 2)
    flap = 0.016 * fnoise1(t * 1.7, 17.0, 2) + 0.6 * hw
    throat = P2 + F(tht) * 0.044 + dirv(tht) * 0.014
    up_chest = lerp(P1, P2, 0.80) + F(tht) * 0.086
    bust = lerp(P1, P2, 0.50) + F(tht) * 0.104
    belly = P1 + F(thl) * 0.104 + np.array([0.0, -0.045])
    xk = belly[0] + 0.020 + 0.45 * hw
    front = [throat, up_chest, bust,
             lerp(bust, belly, 0.5) + np.array([0.010, 0.0]),
             belly,
             np.array([belly[0] + 0.008 + 0.15 * hw, P0[1] - 0.05]),
             np.array([xk, 0.58 + fl2]),
             np.array([xk + 0.004 + 0.10 * hw, 0.40 + fl2]),
             np.array([xk - 0.006 + 0.12 * hw, 0.255 + fl2])]
    hx0 = xk - 0.006 + 0.12 * hw
    hx1 = x0 + 0.190 + 0.9 * hw + flap
    hem = []
    nf = 5
    for q in range(1, nf):
        s_ = q / nf
        xx = hx0 + (hx1 - hx0) * s_
        yy = 0.246 - 0.014 * s_ + (0.009 if q % 2 else -0.005) + 0.006 * fnoise1(t * 2.0 + q * 1.7, 19.0, 2) + fl * s_
        hem.append(np.array([xx, yy]))
    back_hem = np.array([hx1, 0.230 + fl + 0.25 * flap])
    seat = P0 + B(thl) * 0.128 + np.array([0.0, -0.090])
    back = [back_hem,
            np.array([x0 + 0.165 + 0.6 * hw + 0.6 * flap, 0.44]),
            np.array([seat[0] + 0.018 + 0.2 * hw + 0.2 * flap, 0.60]),
            seat,
            P1 + B(thl) * 0.102,
            lerp(P1, P2, 0.55) + B(tht) * 0.124,
            P2 + B(tht) * 0.094 - dirv(tht) * 0.018,
            P2 + B(tht) * 0.052 + dirv(tht) * 0.028]
    ctrl = front + hem + back
    coat = crspline(ctrl, per=7)
    body.poly(coat, r=0.004)
    # collar standing round the neck (under the wrap)
    body.ellipse(P2 + dirv(thn) * 0.03, 0.056, 0.048, rot=thn)
    body.ellipse(head_c + rot(np.array([0.006, -0.012]), thh), 0.072, 0.082, rot=thh)   # skull mass
    S = P2 - dirv(tht) * 0.062 + B(tht) * 0.006
    body.ellipse(S + B(tht) * 0.012, 0.064, 0.056, rot=tht)
    # ---- head: face profile, grey hair drawn back into a bun, flyaways
    fp = face_poly(ELDER_FACE2, head_c, thh)
    skin.poly(crspline(fp, per=3), r=0.0015, k=0.004)
    skin.ellipse(head_c + rot(np.array([-0.028, -0.034]), thh), 0.046, 0.052, rot=thh, k=0.012)
    hair.ellipse(head_c + rot(np.array([0.014, 0.012]), thh), 0.078, 0.080, rot=thh)
    bun_c = head_c + rot(np.array([0.078, 0.046]), thh)
    hair.ellipse(bun_c, 0.038, 0.035, rot=thh - 30)
    hair.ellipse(bun_c + rot(np.array([0.012, 0.016]), thh), 0.024, 0.020, rot=thh - 30)
    hair.ellipse(bun_c + rot(np.array([0.020, -0.012]), thh), 0.020, 0.018, rot=thh - 30, k=0.012)
    if wisps is not None:
        for q, wp in enumerate(wisps):
            # a fine flyaway strand: smoothed chain, curling more toward its tip
            sp = crspline(np.asarray(wp, np.float64), per=4, closed=False)
            m = len(sp)
            ss_ = np.linspace(0.0, 1.0, m)
            tg = np.gradient(sp, axis=0)
            tg /= np.linalg.norm(tg, axis=1, keepdims=True) + 1e-12
            nrm_ = np.stack([-tg[:, 1], tg[:, 0]], 1)
            curl = 0.0045 * ss_ ** 1.5 * np.sin(ss_ * 9.0 + t * 2.3 + q * 2.1)
            sp = sp + nrm_ * curl[:, None]
            wisp.chain(sp[: max(3, int(m * 0.8))], np.linspace(0.0009, 0.0002, max(3, int(m * 0.8))), k=0.001)
    for q, (ax_, ay_, ang, L) in enumerate(((-0.044, 0.068, 70.0, 0.026), (0.052, -0.024, -125.0, 0.024),
                                           (0.070, 0.034, -100.0, 0.030))):
        a0 = head_c + rot(np.array([ax_, ay_]), thh)
        wv = 8.0 * fnoise1(t * 1.4 + q, 23.0 + q, 2)
        a1 = a0 + dirv(ang + wv) * L * 0.5
        a2 = a1 + dirv(ang + 22.0 + wv * 1.5) * L * 0.5
        wisp.cone(a0, a1, 0.0010, 0.0008, k=0.001)
        wisp.cone(a1, a2, 0.0008, 0.0004, k=0.001)
    # ---- arms: heavy coat sleeves (elbow crease, cuff), IK as v1
    def tgt(key):
        v = np.asarray(p[key], np.float64).copy()
        if facing == 1:
            v[0] = 2 * x0 - v[0]
        return v
    if p.get('f_tgt') is not None:
        p['f_ua'], p['f_fa'] = ik2(S, tgt('f_tgt'), 0.26, 0.225, bend=p.get('f_bend', -1.0))
    if p.get('n_tgt') is not None:
        p['n_ua'], p['n_fa'] = ik2(S, tgt('n_tgt'), 0.26, 0.225, bend=p.get('n_bend', -1.0))

    def sleeve(g, ua, fa):
        E = S + dirv(ua) * 0.26
        W = E + dirv(fa) * 0.225
        g.cone(S, E, 0.056, 0.047)
        g.cone(E, W - dirv(fa) * 0.02, 0.047, 0.043)
        # elbow: fabric bunches on the inside of the bend, the point of the elbow shows outside
        bend = ((fa - ua + 180.0) % 360.0) - 180.0
        side = 1.0 if bend > 0 else -1.0
        mid_in = E + perp(unit(dirv(ua) + dirv(fa))) * (-side) * 0.036
        g.ellipse(mid_in, 0.020, 0.012, rot=fa + 90.0, k=0.012)
        g.circle(E + perp(unit(dirv(ua) + dirv(fa))) * side * 0.012, 0.046, k=0.02)
        cuffc = W - dirv(fa) * 0.018
        g.ellipse(cuffc, 0.049, 0.024, rot=fa, k=0.01)
        return E, W

    Ef, Wf = sleeve(farg, p['f_ua'], p['f_fa'])
    En, Wn = sleeve(near, p['n_ua'], p['n_fa'])
    Hf = Wf + dirv(p['f_h']) * 0.045
    Hn = Wn + dirv(p['n_h']) * 0.045
    torch_top = None
    if p['torch']:
        tdir = dirv(p['torch_ang'])
        tb = Hf - tdir * 0.30
        tt = Hf + tdir * 0.16
        torch_g.cone(tb, tt, 0.0125, 0.0165)
        hg = tt + tdir * 0.055
        rag.cone(tt - tdir * 0.004, hg + tdir * 0.02, 0.023, 0.028, k=0.008)
        nn_ = perp(tdir)
        for q in range(4):
            c = tt + tdir * (0.010 + 0.018 * q)
            rag.ellipse(c + nn_ * (0.002 * ((q % 2) * 2 - 1)), 0.0245 + 0.0015 * q + 0.001 * (q % 2), 0.008, rot=p['torch_ang'], k=0.010)
        torch_top = hg + tdir * 0.035
    wf_ = Wf - dirv(p['f_fa']) * 0.004
    if p.get('hand_f', 'fist') == 'open':
        open_hand(handg, wf_, p['f_h'], 1.0)
    else:
        fist(handg, wf_, p['f_h'], 1.0)
    wn_ = Wn - dirv(p['n_fa']) * 0.004
    if p.get('hand_n', 'fist') == 'open':
        open_hand(handg, wn_, p['n_h'], 1.0)
    else:
        fist(handg, wn_, p['n_h'], 1.0)
    # ---- scarf: two wool rolls round the neck (each shaded as its own roll, so the crease
    # between them shows), the crossing at the throat, the short front end
    wrap.ellipse(P2 + dirv(thn) * 0.016 + B(thn) * 0.004, 0.084, 0.034, rot=thn - 4)
    wrap.ellipse(P2 + dirv(thn) * 0.016 + F(thn) * 0.050, 0.036, 0.036, rot=thn, k=0.02)
    roll2.ellipse(P2 + dirv(thn) * 0.054 + F(thn) * 0.006, 0.070, 0.030, rot=thn + 8)
    roll2.ellipse(P2 + dirv(thn) * 0.050 + F(thn) * 0.052, 0.028, 0.030, rot=thn + 20, k=0.016)
    end_top = P2 + F(tht) * 0.064 + dirv(tht) * 0.006
    short_end(knot, end_top, t, hw, length=0.15, width=0.072)
    knot.ellipse(end_top + np.array([0.0, -0.004]), 0.030, 0.022, rot=tht, k=0.012)
    sa, _ = elder_anchor_pts(p)
    if scarf_pts is not None:
        scarf_tail(tail, fringe, scarf_pts, t)
    groups = [tail, fringe, farg, torch_g, rag, legs, body, hair, wisp, skin, knot, wrap, roll2, near, handg]
    anchors = dict(torch_top=torch_top, scarf_anchor=sa, hand_near=Hn, hand_far=Hf,
                   head_center=head_c, face_front=head_c + rot(np.array([-0.10, -0.03]), thh),
                   neck=P2, chest=lerp(P1, P2, 0.6), bun=bun_c)
    if facing == 1:
        mirror2(groups, x0)
        for k, v in anchors.items():
            if v is not None:
                anchors[k] = np.array([2 * x0 - v[0], v[1]])
    return groups, anchors


# ------------------------------------------------------------------ CHILD ---

CHILD_FACE2 = [(-0.020, 0.072), (-0.066, 0.053), (-0.086, 0.023), (-0.090, 0.005), (-0.087, -0.006),
               (-0.091, -0.013), (-0.096, -0.020), (-0.095, -0.025), (-0.089, -0.028), (-0.090, -0.035),
               (-0.086, -0.040), (-0.088, -0.045), (-0.083, -0.051), (-0.082, -0.058), (-0.072, -0.070),
               (-0.052, -0.078), (-0.026, -0.078), (-0.006, -0.056), (0.004, 0.000), (-0.002, 0.050)]

CHILD_DEFAULT = dict(x=0.0, hip_y=0.50, lean=0.0, head=0.0, head_yaw=90.0,
                     n_ua=180.0, n_fa=175.0, f_ua=180.0, f_fa=178.0,
                     n_th=172.0, n_sh=183.0, f_th=190.0, f_sh=178.0, breath=0.0, hat_pom=(0.0, 0.0),
                     mitt_n=None, mitt_f=None)


def child2(pose, t, facing=-1):
    p = dict(CHILD_DEFAULT)
    p.update(pose)
    x0 = p['x']
    F = lambda th: -backv(th)
    B = lambda th: backv(th)
    body = G2('child_jacket', 'jacket', k=0.02)
    farg = G2('child_far', 'jacket', k=0.016)
    near = G2('child_near', 'jacket', k=0.016)
    skin = G2('child_skin', 'skin', k=0.005)
    hairg = G2('child_hair', 'hair_dark', k=0.006)
    hat = G2('child_hat', 'wool', k=0.012)
    pomg = G2('child_pom', 'pom', k=0.012)
    mitts = G2('child_mitts', 'mitt', k=0.008, cast=0)
    legs = G2('child_legs', 'boot', k=0.016)
    lean = p['lean']
    P0 = np.array([x0, p['hip_y']])
    P1 = P0 + dirv(lean) * 0.13
    P2 = P1 + dirv(lean * 1.2) * 0.15 + np.array([0, 0.004 * p['breath']])
    thh = p['head']
    neck_top = P2 + dirv(lean * 0.5 + thh * 0.35) * 0.035
    head_c = neck_top + rot(np.array([-0.004, 0.082]), thh)
    yaw = math.radians(p['head_yaw'])
    depth = math.sin(yaw)
    # ---- snow trousers + chunky boots
    for (dx, th, sh, sc) in [(0.028, p['f_th'], p['f_sh'], 0.97), (-0.028, p['n_th'], p['n_sh'], 1.0)]:
        hp = P0 + np.array([dx, -0.04])
        K = hp + dirv(th) * 0.22
        A = K + dirv(sh) * 0.20
        A[1] = max(A[1], 0.075)
        legs.cone(hp, K, 0.058 * sc, 0.050 * sc)
        legs.cone(K, A + np.array([0, 0.03]), 0.050 * sc, 0.047 * sc)
        legs.ellipse(A + np.array([0.0, 0.045]), 0.052 * sc, 0.030 * sc, k=0.012)       # bunching over the boot
        ax, ay = A
        boot = [(ax + 0.046, 0.006), (ax + 0.050, 0.040), (ax + 0.044, ay + 0.035), (ax + 0.000, ay + 0.050),
                (ax - 0.040, ay + 0.030), (ax - 0.070, 0.062), (ax - 0.098, 0.046), (ax - 0.108, 0.022),
                (ax - 0.100, 0.002), (ax - 0.030, -0.002), (ax + 0.020, -0.002)]
        legs.poly(crspline(boot, per=4), r=0.006, k=0.012)
    # ---- quilted puffer jacket
    ctrl = [P2 + F(lean) * 0.052 + np.array([0.0, 0.012]),
            lerp(P1, P2, 0.62) + F(lean) * 0.124,
            P1 + F(lean) * 0.132,
            P0 + F(lean) * 0.126 + np.array([0.0, -0.030]),
            P0 + F(lean) * 0.098 + np.array([0.0, -0.078]),
            P0 + np.array([0.0, -0.088]),
            P0 + B(lean) * 0.100 + np.array([0.0, -0.078]),
            P0 + B(lean) * 0.126 + np.array([0.0, -0.030]),
            P1 + B(lean) * 0.128,
            lerp(P1, P2, 0.62) + B(lean) * 0.118,
            P2 + B(lean) * 0.092 + np.array([0.0, 0.016]),       # the hood lying on the back
            P2 + B(lean) * 0.078 + np.array([0.0, 0.058]),
            P2 + B(lean) * 0.036 + np.array([0.0, 0.050])]
    jac = crspline(ctrl, per=7)
    jac = quilt(jac, P0[1] - 0.088, 0.066, 0.013,
                mask_fn=lambda Q: ((Q[:, 1] > P0[1] - 0.07) & (Q[:, 1] < P2[1] - 0.01)).astype(np.float64))
    body.poly(jac, r=0.006)
    body.cone(P2, neck_top, 0.046, 0.040)
    body.ellipse(head_c, 0.086, 0.090, rot=thh)
    S = P2 + np.array([0.0, -0.046])
    body.ellipse(S, 0.070, 0.054)
    # ---- face (profile collapses as the head turns toward camera) + hair at the nape
    if abs(depth) > 0.05:
        prof = np.array(CHILD_FACE2)
        if depth < 0:
            prof[:, 0] = -prof[:, 0]
        fp = face_poly(prof, head_c, thh if depth > 0 else -thh, depth=abs(depth))
        skin.poly(crspline(fp, per=3), r=0.0015, k=0.004)
    skin.ellipse(head_c + rot(np.array([-0.040 * depth, -0.030]), thh), 0.050, 0.052, rot=thh, k=0.012)
    nape = head_c + rot(np.array([0.064 * depth, -0.034]), thh)
    hairg.ellipse(nape, 0.020, 0.013, rot=thh + 30 * depth, k=0.008)
    for q in range(2):
        a0 = nape + rot(np.array([0.004 * q * depth, -0.008]), thh)
        a1 = a0 + dirv(-150.0 * np.sign(depth + 1e-6) + 20 * q + 5 * fnoise1(t + q, 3.0 + q)) * 0.012
        hairg.cone(a0, a1, 0.0030, 0.0010, k=0.003)
    # ---- knitted beanie: slouchy crown, turned-up cuff, fuzzy pom-pom on a spring
    hat.ellipse(head_c + rot(np.array([0.012 * depth, 0.046]), thh), 0.095, 0.080, rot=thh + 4 * depth)
    hat.ellipse(head_c + rot(np.array([0.030 * depth, 0.086]), thh), 0.060, 0.050, rot=thh + 12 * depth, k=0.03)
    hat.box(head_c + rot(np.array([0.006 * depth, 0.008]), thh), 0.098, 0.024, rot=thh - 6 * depth, rnd=0.011)
    pom_base = head_c + rot(np.array([0.012 * depth, 0.122]), thh)
    pom = pom_base + np.asarray(p['hat_pom'])
    hat.cone(pom_base + rot(np.array([0, -0.022]), thh), pom, 0.014, 0.012, k=0.012)
    pomg.circle(pom, 0.033, k=0.01)
    for q in range(11):
        ang = q * 2 * math.pi / 11 + 0.3 + 0.2 * fnoise1(t * 0.7 + q, 5.0)
        rr = 0.029 + 0.004 * ((q * 5) % 3) / 2
        pomg.circle(pom + rr * np.array([math.cos(ang), math.sin(ang)]), 0.010 + 0.002 * (q % 2), k=0.010)

    # ---- puffy sleeves (baffled), elastic cuffs, mittens with thumbs
    def tgt(key):
        v = np.asarray(p[key], np.float64).copy()
        if facing == 1:
            v[0] = 2 * x0 - v[0]
        return v
    if p.get('f_tgt') is not None:
        p['f_ua'], p['f_fa'] = ik2(S, tgt('f_tgt'), 0.155, 0.14 + 0.03, bend=p.get('f_bend', -1.0))
    if p.get('n_tgt') is not None:
        p['n_ua'], p['n_fa'] = ik2(S, tgt('n_tgt'), 0.155, 0.14 + 0.03, bend=p.get('n_bend', -1.0))

    def sleeve(g, ua, fa):
        E = S + dirv(ua) * 0.155
        W = E + dirv(fa) * 0.14
        segs = [(S, E, 0.050, 0.043, 3), (E, W, 0.043, 0.037, 3)]
        for (a, b, ra, rb, m) in segs:
            for q in range(m):
                s0, s1 = q / m, (q + 1) / m
                pa = a + (b - a) * s0
                pb = a + (b - a) * s1
                r0 = ra + (rb - ra) * s0
                r1 = ra + (rb - ra) * s1
                g.cone(pa, pb, r0 * 0.93, r1 * 0.93, k=0.012)
                g.circle(0.5 * (pa + pb), 0.5 * (r0 + r1) * 1.04, k=0.012)
        g.ellipse(W - dirv(fa) * 0.008, 0.030, 0.014, rot=fa, k=0.008)     # elastic cuff
        return E, W, W + dirv(fa) * 0.032

    Ef, Wf, Hf = sleeve(farg, p['f_ua'], p['f_fa'])
    En, Wn, Hn = sleeve(near, p['n_ua'], p['n_fa'])
    mf = p['mitt_f'] if p.get('mitt_f') is not None else p['f_fa']
    mn = p['mitt_n'] if p.get('mitt_n') is not None else p['n_fa']
    mitten(mitts, Wf, mf, 0.95)
    mitten(mitts, Wn, mn, 0.95)
    groups = [farg, legs, body, hairg, skin, hat, pomg, near, mitts]
    anchors = dict(hand_near=Hn, hand_far=Hf, head_center=head_c, pom=pom, pom_base=pom_base,
                   face_front=head_c + rot(np.array([-0.10 * depth, -0.02]), thh), chest=lerp(P1, P2, 0.6),
                   shoulder=S + np.array([0.0, 0.04]))
    if facing == 1:
        mirror2(groups, x0)
        for k, v in anchors.items():
            anchors[k] = np.array([2 * x0 - v[0], v[1]])
    return groups, anchors


# ------------------------------------------------------------------ TORCH ---

def torch_alone2(base, top_dir, length=0.46, t=0.0):
    """The torch in the child's hands: shaft + bound rag head. Returns ([groups], flame base)."""
    d = unit(top_dir)
    tb = np.asarray(base, np.float64)
    tt = tb + d * length
    g = G2('torch', 'torch', k=0.004)
    g.cone(tb, tt, 0.0135, 0.0170)
    rag = G2('torch_rag', 'rag', k=0.006)
    hg = tt + d * 0.06
    rag.cone(tt - d * 0.004, hg + d * 0.022, 0.024, 0.029, k=0.008)
    nn_ = perp(d)
    ang = ang_of(d)
    for q in range(4):
        c = tt + d * (0.010 + 0.019 * q)
        rag.ellipse(c + nn_ * (0.002 * ((q % 2) * 2 - 1)), 0.0255 + 0.0015 * q + 0.001 * (q % 2), 0.008, rot=ang, k=0.010)
    return [g, rag], hg + d * 0.05


def translate(groups, dx, dy):
    """Shift every primitive of the groups by (dx, dy) metres (to share one card)."""
    for g in groups:
        for row in g.rows:
            typ = int(row[0])
            if typ in (P_ELL, P_RING, P_BOX):
                row[2] += dx
                row[3] += dy
            elif typ == P_CONE:
                row[2] += dx
                row[3] += dy
                row[4] += dx
                row[5] += dy
        for v in g.verts:
            v[0] += dx
            v[1] += dy
        g.bbs = [(b[0] + dx, b[1] + dy, b[2] + dx, b[3] + dy) for b in g.bbs]
        pa = getattr(g, 'prim_attr', None)
        if pa is not None and np.asarray(pa).shape[1] > 6:
            pa = np.asarray(pa, np.float64).copy()
            pa[:, 4] += dx
            pa[:, 5] += dy
            g.prim_attr = pa
    return groups
