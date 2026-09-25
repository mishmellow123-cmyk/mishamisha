"""Character rigs -> puppet Groups. All built facing LEFT (-x) in local metres with the
origin on the ground between the feet, then optionally mirrored.

Angle convention: a bone direction angle theta (degrees) is measured from +y (up),
positive rotating toward the facing direction (-x). dirv(0) = up, dirv(90) = forward,
dirv(180) = down, dirv(-90) = backward. Head angle: positive = chin down.
"""
import math

import numpy as np

from core import fnoise1
from puppet import Group, P_ELL, P_CONE, P_POLY, P_BOX, P_RING

# ---------------------------------------------------------------- albedos ---
COAT_ELDER = np.array([0.022, 0.020, 0.024])
COAT_CHILD = np.array([0.020, 0.028, 0.045])
PARKA = np.array([0.026, 0.024, 0.022])
SCARF = np.array([0.46, 0.028, 0.026])
SKIN = np.array([0.30, 0.17, 0.115])
SKIN_CHILD = np.array([0.33, 0.19, 0.13])
HAIR_GREY = np.array([0.20, 0.20, 0.21])
HAIR_DARK = np.array([0.030, 0.020, 0.016])
WOOL_HAT = np.array([0.36, 0.28, 0.17])
BOOT = np.array([0.015, 0.012, 0.011])
STONE = np.array([0.085, 0.080, 0.072])
IRON = np.array([0.030, 0.027, 0.025])
WOOD = np.array([0.070, 0.046, 0.030])
TORCH_WOOD = np.array([0.13, 0.075, 0.040])


def dirv(th):
    t = math.radians(th)
    return np.array([-math.sin(t), math.cos(t)])


def backv(th):
    """unit normal pointing backward (+x side) of a bone at angle th"""
    t = math.radians(th)
    return np.array([math.cos(t), math.sin(t)])


def rot(v, th):
    """rotate local vector by head/body angle th (positive tips top forward/-x)."""
    t = math.radians(th)
    c, s = math.cos(t), math.sin(t)
    v = np.asarray(v, np.float64)
    return np.stack([v[..., 0] * c - v[..., 1] * s, v[..., 0] * s + v[..., 1] * c], -1)


def lerp(a, b, t):
    return np.asarray(a) * (1 - t) + np.asarray(b) * t


def mirror_groups(groups, cx=0.0):
    """Mirror all primitives about x = cx (turns a left-facing rig to face right)."""
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
    return groups


def face_poly(profile, center, head_th, scale=1.0, depth=1.0):
    """Profile polygon points (head-local, facing -x) -> local coords. depth scales the
    forward protrusion (for head turns toward camera)."""
    p = np.array(profile, np.float64) * scale
    p[:, 0] *= depth
    return rot(p, head_th) + center


ELDER_FACE = [(-0.012, 0.080), (-0.058, 0.066), (-0.075, 0.042), (-0.079, 0.020),
              (-0.083, 0.008), (-0.078, -0.002), (-0.084, -0.010), (-0.104, -0.034),
              (-0.100, -0.041), (-0.086, -0.043), (-0.088, -0.052), (-0.083, -0.057),
              (-0.086, -0.063), (-0.080, -0.072), (-0.081, -0.084), (-0.070, -0.097),
              (-0.046, -0.104), (-0.018, -0.090), (0.0, -0.03), (0.0, 0.05)]

CHILD_FACE = [(-0.015, 0.070), (-0.066, 0.050), (-0.084, 0.022), (-0.089, 0.002),
              (-0.087, -0.008), (-0.096, -0.020), (-0.099, -0.027), (-0.090, -0.033),
              (-0.089, -0.040), (-0.092, -0.046), (-0.086, -0.052), (-0.087, -0.057),
              (-0.076, -0.071), (-0.055, -0.083), (-0.028, -0.086), (0.0, -0.06), (0.0, 0.05)]

YW_FACE = [(-0.012, 0.082), (-0.060, 0.062), (-0.074, 0.038), (-0.076, 0.016),
           (-0.080, 0.006), (-0.076, -0.003), (-0.082, -0.012), (-0.098, -0.036),
           (-0.093, -0.042), (-0.082, -0.044), (-0.086, -0.052), (-0.081, -0.057),
           (-0.085, -0.062), (-0.078, -0.068), (-0.078, -0.078), (-0.070, -0.091),
           (-0.052, -0.097), (-0.018, -0.084), (0.0, -0.03), (0.0, 0.055)]


def _limb(g, root, angles, lengths, radii):
    pts = [np.asarray(root, np.float64)]
    for a, L in zip(angles, lengths):
        pts.append(pts[-1] + dirv(a) * L)
    g.chain(pts, radii)
    return pts


def ribbon(g, pts, width, t, seed=0.0, fringe=True, taper_end=0.8):
    """Scarf tail: twisting ribbon along chain points (apparent width varies)."""
    pts = np.asarray(pts, np.float64)
    n = len(pts)
    radii = []
    for i in range(n):
        s = i / max(1, n - 1)
        tw = fnoise1(s * 3.2 - t * 1.7, seed, 2)
        w = width * (0.42 + 0.58 * abs(math.cos(1.4 * tw + 0.6)))
        w *= 1.0 - (1 - taper_end) * s
        radii.append(0.5 * w)
    g.chain(pts, radii)
    if fringe and n >= 2:
        d = pts[-1] - pts[-2]
        d /= (np.linalg.norm(d) + 1e-9)
        nrm = np.array([-d[1], d[0]])
        wend = radii[-1]
        for k in range(5):
            off = (k - 2) / 2.0 * wend * 0.85
            wig = 0.012 * fnoise1(t * 3 + k * 1.3, seed + 5, 1)
            a = pts[-1] + nrm * off
            b = a + d * (0.055 + 0.01 * (k % 2)) + nrm * (off * 0.25 + wig)
            g.cone(a, b, 0.0045, 0.0025, k=0.004)


# ------------------------------------------------------------------ ELDER ---

ELDER_DEFAULT = dict(x=0.0, hip_y=0.84, lumbar=5.0, thorax=27.0, neck=50.0, head=6.0,
                     # near arm (holds the child's hand), far arm (holds torch)
                     n_ua=172.0, n_fa=150.0, n_h=150.0,
                     f_ua=150.0, f_fa=62.0, f_h=30.0,
                     n_th=182.0, n_sh=178.0, f_th=176.0, f_sh=183.0,
                     breath=0.0, hem_wind=0.05, torch=True)


def elder(pose, t, scarf_pts=None, facing=-1, wisps=None):
    """Returns (groups, anchors). anchors: dict of useful world-ish local points
    (torch_top, scarf_anchor, hand_near, head_center, face_front)."""
    p = dict(ELDER_DEFAULT)
    p.update(pose)
    x0 = p['x']
    P0 = np.array([x0, p['hip_y'] + 0.004 * p['breath']])
    thl, tht, thn, thh = p['lumbar'], p['thorax'] + 1.5 * p['breath'], p['neck'], p['head']
    P1 = P0 + dirv(thl) * 0.21
    P2 = P1 + dirv(tht) * 0.25
    P3 = P2 + dirv(thn) * 0.085
    head_c = P3 + rot(np.array([-0.006, 0.072]), thh)
    body = Group('elder_body', COAT_ELDER, k=0.03, bevel=0.06, sheen=0.9, sky_rim=0.5)
    farg = Group('elder_far', COAT_ELDER, k=0.02, bevel=0.05, sheen=0.9, sky_rim=0.4)
    near = Group('elder_near', COAT_ELDER, k=0.02, bevel=0.05, sheen=0.9, sky_rim=0.4)
    skin = Group('elder_skin', SKIN, k=0.008, bevel=0.03, sheen=0.5, sky_rim=0.2)
    hair = Group('elder_hair', HAIR_GREY, k=0.01, bevel=0.02, sheen=1.2, sky_rim=0.5)
    wrap = Group('elder_scarfwrap', SCARF, k=0.02, bevel=0.04, sheen=0.7, sky_rim=0.3)
    tail = Group('elder_scarftail', SCARF, k=0.012, bevel=0.03, sheen=0.7, sky_rim=0.3, translucent=0.35)
    legs = Group('elder_legs', BOOT, k=0.015, bevel=0.03, sheen=0.6)
    torch_g = Group('torch', TORCH_WOOD, k=0.004, bevel=0.012, sheen=0.5, sky_rim=0.2)

    # ---- legs (visible below hem)
    hipn = P0 + np.array([-0.035, -0.02])
    hipf = P0 + np.array([0.035, -0.02])
    for (hp, th, sh, sc) in [(hipf, p['f_th'], p['f_sh'], 0.95), (hipn, p['n_th'], p['n_sh'], 1.0)]:
        K = hp + dirv(th) * 0.40
        A = K + dirv(sh) * 0.40
        legs.cone(K, A, 0.050 * sc, 0.036 * sc)
        # shoe
        toe = A + np.array([-0.125, -0.030])
        legs.cone(A + np.array([0.02, -0.03]), toe, 0.036, 0.026, k=0.02)
    # ---- coat
    b = lambda th: backv(th)
    f = lambda th: -backv(th)
    hw = p['hem_wind']
    fl = 0.012 * fnoise1(t * 2.3, 11.0, 2)
    coat = [
        P2 + f(tht) * 0.050 + dirv(tht) * 0.005,         # throat
        lerp(P1, P2, 0.55) + f(tht) * 0.118,              # chest
        P1 + f(thl) * 0.108,                              # belly
        P0 + f(thl) * 0.112 + np.array([0.0, -0.08]),     # front hip
        np.array([x0 - 0.175 + hw * 0.4, 0.27]),          # hem front
        np.array([x0 - 0.06 + hw * 0.6, 0.245 + fl]),
        np.array([x0 + 0.08 + hw * 0.8, 0.25 - fl]),
        np.array([x0 + 0.215 + hw, 0.235 + fl]),          # hem back (flared)
        P0 + b(thl) * 0.125 + np.array([0.0, -0.10]),     # seat
        P1 + b(thl) * 0.108,                              # mid back
        lerp(P1, P2, 0.5) + b(tht) * 0.118,               # stoop hump
        P2 + b(tht) * 0.070 + dirv(tht) * 0.01,           # back of collar
    ]
    body.poly(coat, r=0.012)
    # collar / neck
    body.cone(P2, P3, 0.052, 0.042)
    # head
    body.ellipse(head_c, 0.080, 0.094, rot=thh)
    # shoulders (rounded)
    S = P2 + dirv(tht + 180) * 0.055 + b(tht) * 0.005
    body.ellipse(S, 0.075, 0.06, rot=tht)
    # ---- face (skin) + hair
    fp = face_poly(ELDER_FACE, head_c, thh)
    skin.poly(fp, r=0.002, k=0.006)
    skin.ellipse(head_c + rot(np.array([-0.05, -0.02]), thh), 0.035, 0.05, rot=thh)
    hair.ellipse(head_c + rot(np.array([0.068, 0.040]), thh), 0.043, 0.040)
    hair.ellipse(head_c + rot(np.array([0.020, 0.070]), thh), 0.070, 0.032, rot=thh - 12)
    if wisps is not None:
        for wp in wisps:
            hair.chain(wp, np.linspace(0.006, 0.0015, len(wp)), k=0.004)
    # ---- arms
    def arm(g, ua, fa, hh, sleeve=1.0):
        E = S + dirv(ua) * 0.27
        W = E + dirv(fa) * 0.235
        g.cone(S, E, 0.058 * sleeve, 0.048 * sleeve)
        g.cone(E, W, 0.048 * sleeve, 0.042 * sleeve)
        g.ellipse(W + dirv(fa) * 0.005, 0.047, 0.036, rot=fa)   # cuff
        H = W + dirv(hh) * 0.055
        return E, W, H
    Ef, Wf, Hf = arm(farg, p['f_ua'], p['f_fa'], p['f_h'])
    En, Wn, Hn = arm(near, p['n_ua'], p['n_fa'], p['n_h'])
    skin.ellipse(Hf, 0.030, 0.042, rot=p['f_h'])
    skin.ellipse(Hn, 0.028, 0.043, rot=p['n_h'])
    # ---- torch in the far hand
    torch_top = None
    if p['torch']:
        tdir = dirv(p.get('torch_ang', p['f_h'] - 30.0))
        tb = Hf - tdir * 0.33
        tt = Hf + tdir * 0.20
        torch_g.cone(tb, tt, 0.015, 0.020)
        head_g = tt + tdir * 0.06
        torch_g.cone(tt, head_g + tdir * 0.03, 0.028, 0.033, k=0.01)
        torch_top = head_g + tdir * 0.05
        # fingers around the handle
        skin.ellipse(Hf + tdir * 0.01, 0.034, 0.028, rot=p['f_h'] + 60)
    # ---- scarf wrap (thick at neck) + front drape
    wrap.ellipse(P2 + dirv(thn) * 0.03, 0.082, 0.058, rot=thn - 90)
    wrap.ellipse(P2 + dirv(thn) * 0.075 + f(thn) * 0.005, 0.066, 0.045, rot=thn - 95)
    dr0 = P2 + f(tht) * 0.07
    dr1 = dr0 + np.array([-0.02, -0.17]) + np.array([hw * 0.3, 0])
    wrap.cone(dr0, dr1, 0.040, 0.032)
    scarf_anchor = P2 + b(tht) * 0.07 + dirv(thn) * 0.045
    if scarf_pts is not None:
        ribbon(tail, scarf_pts, 0.19, t, seed=3.0)
    groups = [tail, farg, torch_g, legs, body, hair, skin, wrap, near]
    anchors = dict(torch_top=torch_top, scarf_anchor=scarf_anchor, hand_near=Hn, hand_far=Hf,
                   head_center=head_c, face_front=head_c + rot(np.array([-0.10, -0.03]), thh),
                   neck=P2, chest=lerp(P1, P2, 0.6))
    if facing == 1:
        mirror_groups(groups, x0)
        for k, v in anchors.items():
            if v is not None:
                anchors[k] = np.array([2 * x0 - v[0], v[1]])
    return groups, anchors


# ------------------------------------------------------------------ CHILD ---

CHILD_DEFAULT = dict(x=0.0, hip_y=0.54, lean=0.0, head=0.0, head_yaw=90.0, body_face=-1,
                     n_ua=180.0, n_fa=175.0, f_ua=180.0, f_fa=178.0,
                     n_th=180.0, n_sh=180.0, f_th=178.0, f_sh=180.0, breath=0.0, hat_pom=(0.0, 0.0))


def child(pose, t, facing=-1):
    """head_yaw: +90 profile facing the body direction, -90 looking back over the
    shoulder, 0 facing camera (profile features collapse)."""
    p = dict(CHILD_DEFAULT)
    p.update(pose)
    x0 = p['x']
    body = Group('child_body', COAT_CHILD, k=0.025, bevel=0.05, sheen=0.9, sky_rim=0.5)
    farg = Group('child_far', COAT_CHILD, k=0.02, bevel=0.04, sheen=0.9, sky_rim=0.4)
    near = Group('child_near', COAT_CHILD, k=0.02, bevel=0.04, sheen=0.9, sky_rim=0.4)
    skin = Group('child_skin', SKIN_CHILD, k=0.006, bevel=0.03, sheen=0.5, sky_rim=0.2)
    hat = Group('child_hat', WOOL_HAT, k=0.012, bevel=0.03, sheen=1.4, sky_rim=0.6)
    legs = Group('child_legs', BOOT, k=0.015, bevel=0.03, sheen=0.6)
    lean = p['lean']
    P0 = np.array([x0, p['hip_y']])
    P1 = P0 + dirv(lean) * 0.16
    P2 = P1 + dirv(lean * 1.2) * 0.17 + np.array([0, 0.004 * p['breath']])     # neck base
    neck_top = P2 + dirv(lean * 0.5 + p['head'] * 0.3) * 0.045
    thh = p['head']
    head_c = neck_top + rot(np.array([-0.004, 0.083]), thh)
    yaw = math.radians(p['head_yaw'])
    depth = math.sin(yaw)            # +1 profile forward, -1 profile backward
    # legs
    for (dx, th, sh) in [(0.03, p['f_th'], p['f_sh']), (-0.03, p['n_th'], p['n_sh'])]:
        hp = P0 + np.array([dx, -0.03])
        K = hp + dirv(th) * 0.25
        A = K + dirv(sh) * 0.24
        legs.cone(hp, K, 0.052, 0.045)
        legs.cone(K, A, 0.045, 0.038)
        legs.ellipse(A + np.array([-0.035, -0.012]), 0.075, 0.042)
    # puffy jacket
    b = lambda th: backv(th)
    f = lambda th: -backv(th)
    coat = [P2 + f(lean) * 0.045, lerp(P1, P2, 0.5) + f(lean) * 0.105, P1 + f(lean) * 0.115,
            P0 + f(lean) * 0.108 + np.array([0, -0.02]), P0 + np.array([-0.08, -0.07]),
            P0 + np.array([0.09, -0.07]), P0 + b(lean) * 0.108 + np.array([0, -0.02]),
            P1 + b(lean) * 0.112, lerp(P1, P2, 0.5) + b(lean) * 0.105, P2 + b(lean) * 0.055]
    body.poly(coat, r=0.028)
    body.cone(P2, neck_top, 0.045, 0.040)
    body.ellipse(head_c, 0.086, 0.090, rot=thh)
    S = P2 + np.array([0.0, -0.04])
    body.ellipse(S, 0.07, 0.05)
    # face (profile collapses with yaw)
    if abs(depth) > 0.05:
        prof = np.array(CHILD_FACE)
        if depth < 0:
            prof[:, 0] = -prof[:, 0]
        fp = face_poly(prof, head_c, thh * (1 if depth > 0 else -1), depth=abs(depth))
        skin.poly(fp, r=0.002, k=0.006)
    skin.ellipse(head_c + rot(np.array([-0.045 * depth, -0.025]), thh), 0.045, 0.05, rot=thh)
    # knitted hat: dome + folded cuff + pom-pom (on a sprung offset)
    hat_rot = thh
    hat.ellipse(head_c + rot(np.array([0.010 * depth, 0.046]), hat_rot), 0.095, 0.074, rot=hat_rot)
    hat.box(head_c + rot(np.array([0.008 * depth, 0.006]), hat_rot), 0.096, 0.021, rot=hat_rot - 7 * depth, rnd=0.008)
    pom_base = head_c + rot(np.array([0.004 * depth, 0.118]), hat_rot)
    pom = pom_base + np.asarray(p['hat_pom'])
    hat.cone(pom_base + rot(np.array([0, -0.02]), hat_rot), pom, 0.012, 0.012, k=0.01)
    hat.ellipse(pom, 0.040, 0.038, k=0.012)
    # arms
    def arm(g, ua, fa):
        E = S + dirv(ua) * 0.17
        W = E + dirv(fa) * 0.15
        g.cone(S, E, 0.043, 0.038)
        g.cone(E, W, 0.038, 0.034)
        return E, W, W + dirv(fa) * 0.035
    Ef, Wf, Hf = arm(farg, p['f_ua'], p['f_fa'])
    En, Wn, Hn = arm(near, p['n_ua'], p['n_fa'])
    skin.ellipse(Hf, 0.026, 0.030)
    skin.ellipse(Hn, 0.026, 0.030)
    groups = [farg, legs, body, skin, hat, near]
    anchors = dict(hand_near=Hn, hand_far=Hf, head_center=head_c, pom=pom,
                   face_front=head_c + rot(np.array([-0.10 * depth, -0.02]), thh), chest=lerp(P1, P2, 0.6))
    if facing == 1:
        mirror_groups(groups, x0)
        for k, v in anchors.items():
            anchors[k] = np.array([2 * x0 - v[0], v[1]])
    return groups, anchors


# ------------------------------------------------------------ YOUNG WOMAN ---

YW_DEFAULT = dict(x=0.0, hip_y=0.93, lumbar=2.0, thorax=6.0, neck=12.0, head=4.0,
                  n_ua=176.0, n_fa=178.0, n_h=180.0, f_ua=178.0, f_fa=180.0, f_h=180.0,
                  n_th=182.0, n_sh=178.0, f_th=176.0, f_sh=182.0, breath=0.0, hem_wind=0.06,
                  kneel=0.0)


def young_woman(pose, t, scarf_pts=None, hair_pts=None, facing=-1, props=True):
    p = dict(YW_DEFAULT)
    p.update(pose)
    x0 = p['x']
    P0 = np.array([x0, p['hip_y']])
    thl, tht, thn, thh = p['lumbar'], p['thorax'] + 1.2 * p['breath'], p['neck'], p['head']
    P1 = P0 + dirv(thl) * 0.23
    P2 = P1 + dirv(tht) * 0.27
    P3 = P2 + dirv(thn) * 0.09
    head_c = P3 + rot(np.array([-0.008, 0.074]), thh)
    body = Group('yw_body', PARKA, k=0.03, bevel=0.06, sheen=0.9, sky_rim=0.5)
    farg = Group('yw_far', PARKA, k=0.02, bevel=0.05, sheen=0.9, sky_rim=0.4)
    near = Group('yw_near', PARKA, k=0.02, bevel=0.05, sheen=0.9, sky_rim=0.4)
    skin = Group('yw_skin', SKIN, k=0.006, bevel=0.03, sheen=0.6, sky_rim=0.2)
    hair = Group('yw_hair', HAIR_DARK, k=0.012, bevel=0.02, sheen=1.3, sky_rim=0.5, translucent=0.2)
    wrap = Group('yw_scarfwrap', SCARF, k=0.02, bevel=0.04, sheen=0.7, sky_rim=0.3)
    tail = Group('yw_scarftail', SCARF, k=0.012, bevel=0.03, sheen=0.7, sky_rim=0.3, translucent=0.35)
    legs = Group('yw_legs', BOOT, k=0.02, bevel=0.035, sheen=0.7)
    b = lambda th: backv(th)
    f = lambda th: -backv(th)
    # legs: thigh 0.45, shin 0.44
    hipn = P0 + np.array([-0.03, -0.03])
    hipf = P0 + np.array([0.03, -0.03])
    feet = []
    for (hp, th, sh, sc) in [(hipf, p['f_th'], p['f_sh'], 0.95), (hipn, p['n_th'], p['n_sh'], 1.0)]:
        K = hp + dirv(th) * 0.45
        A = K + dirv(sh) * 0.44
        legs.cone(hp, K, 0.068 * sc, 0.052 * sc)
        legs.cone(K, A, 0.052 * sc, 0.040 * sc)
        # boot: along ground if foot flat, else follows shin
        fdir = dirv(sh + 90) if abs(sh - 180) < 45 else dirv(sh)
        toe = A + fdir * 0.14 + np.array([0, -0.03])
        legs.cone(A + np.array([0.015, -0.02]), toe, 0.045, 0.034, k=0.02)
        feet.append((K, A))
    # parka (to mid thigh), hood bunched at back
    hw = p['hem_wind']
    fl = 0.015 * fnoise1(t * 2.7, 21.0, 2)
    hem_y = P0[1] - 0.30
    coat = [
        P2 + f(tht) * 0.055 + dirv(tht) * 0.01,
        lerp(P1, P2, 0.6) + f(tht) * 0.125,
        P1 + f(thl) * 0.112,
        P0 + f(thl) * 0.118 + np.array([0, -0.10]),
        P0 + f(thl) * 0.12 + dirv(thl + 180) * 0.30 + np.array([hw * 0.3, 0]),
        P0 + dirv(thl + 180) * 0.31 + np.array([hw * 0.6, fl]),
        P0 + b(thl) * 0.14 + dirv(thl + 180) * 0.29 + np.array([hw, -fl]),
        P0 + b(thl) * 0.13 + np.array([0, -0.06]),
        P1 + b(thl) * 0.112,
        lerp(P1, P2, 0.55) + b(tht) * 0.118,
        P2 + b(tht) * 0.075 + dirv(tht) * 0.01,
    ]
    body.poly(coat, r=0.014)
    body.cone(P2, P3, 0.050, 0.040)
    body.ellipse(head_c, 0.077, 0.093, rot=thh)
    S = P2 + dirv(tht + 180) * 0.055
    body.ellipse(S, 0.075, 0.062, rot=tht)
    # hood bunched behind the neck
    body.ellipse(P2 + b(tht) * 0.075 + dirv(tht) * 0.02, 0.085, 0.07, rot=tht + 20)
    fp = face_poly(YW_FACE, head_c, thh)
    skin.poly(fp, r=0.002, k=0.006)
    skin.ellipse(head_c + rot(np.array([-0.045, -0.02]), thh), 0.035, 0.05, rot=thh)
    # hair: cap + blown strands
    hair.ellipse(head_c + rot(np.array([0.012, 0.030]), thh), 0.083, 0.078, rot=thh)
    if hair_pts is not None:
        for hp_ in hair_pts:
            hair.chain(hp_, np.linspace(0.028, 0.004, len(hp_)), k=0.02)
    # arms
    def arm(g, ua, fa, hh):
        E = S + dirv(ua) * 0.28
        W = E + dirv(fa) * 0.25
        g.cone(S, E, 0.060, 0.050)
        g.cone(E, W, 0.050, 0.043)
        g.ellipse(W + dirv(fa) * 0.004, 0.048, 0.038, rot=fa)
        H = W + dirv(hh) * 0.055
        return E, W, H
    Ef, Wf, Hf = arm(farg, p['f_ua'], p['f_fa'], p['f_h'])
    En, Wn, Hn = arm(near, p['n_ua'], p['n_fa'], p['n_h'])
    # hands with fingers (bare, for the flint close-up)
    for (H, hh, g) in [(Hf, p['f_h'], skin), (Hn, p['n_h'], skin)]:
        g.ellipse(H, 0.032, 0.046, rot=hh)
        d = dirv(hh)
        nrm = np.array([-d[1], d[0]])
        for k in range(4):
            base = H + d * 0.025 + nrm * (k - 1.5) * 0.014
            g.cone(base, base + d * (0.045 - 0.006 * abs(k - 1.5)) - nrm * 0.004, 0.0085, 0.0065, k=0.006)
        g.cone(H - nrm * 0.03 + d * 0.0, H - nrm * 0.045 + d * 0.035, 0.010, 0.008, k=0.008)  # thumb
    # wrap + drape
    wrap.ellipse(P2 + dirv(thn) * 0.03, 0.084, 0.060, rot=thn - 90)
    wrap.ellipse(P2 + dirv(thn) * 0.078 + f(thn) * 0.004, 0.068, 0.047, rot=thn - 95)
    dr0 = P2 + f(tht) * 0.075
    wrap.cone(dr0, dr0 + np.array([-0.02 + hw * 0.3, -0.18]), 0.040, 0.032)
    scarf_anchor = P2 + b(tht) * 0.075 + dirv(thn) * 0.045
    if scarf_pts is not None:
        ribbon(tail, scarf_pts, 0.19, t, seed=3.0)
    groups = [tail, farg, legs, body, hair, skin, wrap, near]
    anchors = dict(scarf_anchor=scarf_anchor, hand_near=Hn, hand_far=Hf, head_center=head_c,
                   mouth=head_c + rot(np.array([-0.092, -0.052]), thh), neck=P2,
                   hair_root=head_c + rot(np.array([0.05, 0.03]), thh), knee_near=feet[1][0],
                   chest=lerp(P1, P2, 0.6))
    if facing == 1:
        mirror_groups(groups, x0)
        for k, v in anchors.items():
            anchors[k] = np.array([2 * x0 - v[0], v[1]])
    return groups, anchors


# ------------------------------------------------------------------ CAIRN ---

class Cairn:
    """Waist-high stone cairn + iron fire basket + stacked wood. Local origin = base centre."""

    def __init__(self, seed=7, height=0.98, base_hw=0.50, top_hw=0.36):
        rng = np.random.default_rng(seed)
        self.height = height
        self.stones = []
        y = 0.0
        row = 0
        while y < height - 0.04:
            sh = rng.uniform(0.085, 0.13)
            hwid = base_hw + (top_hw - base_hw) * (y / height)
            x = -hwid + rng.uniform(-0.03, 0.03) + (0.05 if row % 2 else 0)
            while x < hwid - 0.02:
                sw = rng.uniform(0.11, 0.21)
                cx = x + sw / 2
                edge = abs(cx) / hwid
                self.stones.append((cx, y + sh / 2 + rng.uniform(-0.01, 0.01),
                                    sw / 2 * 1.08, sh / 2 * (1.12 - 0.15 * edge), rng.uniform(-9, 9)))
                x += sw * rng.uniform(0.92, 1.0)
            y += sh * 0.9
            row += 1
        self.top = y
        # basket geometry
        self.bk_bot = self.top + 0.02
        self.bk_top = self.bk_bot + 0.34
        self.bk_rb = 0.20
        self.bk_rt = 0.33
        n = 9
        self.bars = [(-1 + 2 * i / (n - 1)) for i in range(n)]
        self.logs = []
        for i in range(8):
            a = -1 + 2 * (i + 0.5) / 8 + rng.uniform(-0.08, 0.08)
            x_b = a * 0.16
            x_t = a * 0.26 + rng.uniform(-0.06, 0.06)
            y_t = self.bk_top + rng.uniform(0.05, 0.20)
            self.logs.append((x_b, self.bk_bot + 0.04, x_t, y_t, rng.uniform(0.028, 0.045)))
        for i in range(3):
            y_ = self.bk_bot + 0.10 + i * 0.08
            self.logs.append((-0.25 + rng.uniform(-0.03, 0.03), y_, 0.25 + rng.uniform(-0.03, 0.03),
                              y_ + rng.uniform(-0.05, 0.05), 0.035))

    def groups(self, x=0.0, snow=0.0, burn=0.0):
        st = Group('cairn', STONE, k=0.012, bevel=0.035, sheen=0.35, sky_rim=0.45, per_prim=True)
        for (cx, cy, rx, ry, r) in self.stones:
            st.ellipse((x + cx, cy), rx, ry, rot=r, k=0.012)
        # basket stand
        iron = Group('basket', IRON, k=0.006, bevel=0.012, sheen=0.8, sky_rim=0.4)
        iron.box((x, self.top + 0.005), 0.17, 0.02, rnd=0.006)
        yb, yt = self.bk_bot, self.bk_top
        for a in self.bars:
            iron.cone((x + a * self.bk_rb, yb), (x + a * self.bk_rt, yt), 0.0085, 0.0085, k=0.004)
        iron.box((x, yb), self.bk_rb + 0.01, 0.012, rnd=0.005)
        iron.box((x, yb + 0.14), (self.bk_rb + self.bk_rt) / 2 + 0.01, 0.009, rnd=0.004)
        iron.box((x, yt), self.bk_rt + 0.015, 0.013, rnd=0.005)
        # finials on the rim
        for a in (-1, -0.5, 0, 0.5, 1):
            iron.cone((x + a * self.bk_rt, yt), (x + a * self.bk_rt * 1.04, yt + 0.045), 0.007, 0.004, k=0.003)
        wood = Group('wood', WOOD, k=0.004, bevel=0.02, sheen=0.4, sky_rim=0.3, per_prim=True,
                     emissive=None)
        for (xb, yb_, xt, yt_, r) in self.logs:
            wood.cone((x + xb, yb_), (x + xt, yt_), r, r * 0.85, k=0.004)
        return [wood, st, iron]

    def fire_base(self, x=0.0):
        return np.array([x, self.bk_bot + 0.10])


# ------------------------------------------------------------------ TORCH ---

def torch_alone(base, top_dir, length=0.55, t=0.0):
    """A torch held by someone else (e.g. the child's hands)."""
    g = Group('torch', TORCH_WOOD, k=0.004, bevel=0.012, sheen=0.5, sky_rim=0.2)
    d = np.asarray(top_dir, np.float64)
    d = d / np.linalg.norm(d)
    tb = np.asarray(base, np.float64)
    tt = tb + d * length
    g.cone(tb, tt, 0.015, 0.020)
    hg = tt + d * 0.06
    g.cone(tt, hg + d * 0.03, 0.028, 0.033, k=0.01)
    return g, hg + d * 0.05
