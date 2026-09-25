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
COAT_ELDER = np.array([0.010, 0.009, 0.011])
COAT_CHILD = np.array([0.010, 0.012, 0.020])
PARKA = np.array([0.012, 0.011, 0.010])
SCARF = np.array([0.30, 0.018, 0.016])
SKIN = np.array([0.24, 0.13, 0.085])
SKIN_CHILD = np.array([0.26, 0.145, 0.095])
HAIR_GREY = np.array([0.12, 0.12, 0.125])
HAIR_DARK = np.array([0.014, 0.010, 0.008])
WOOL_HAT = np.array([0.20, 0.15, 0.09])
BOOT = np.array([0.008, 0.007, 0.007])
STONE = np.array([0.060, 0.056, 0.050])
IRON = np.array([0.020, 0.018, 0.017])
WOOD = np.array([0.045, 0.030, 0.020])
TORCH_WOOD = np.array([0.09, 0.052, 0.028])


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

ELDER_DEFAULT = dict(x=0.0, hip_y=0.82, lumbar=6.0, thorax=30.0, neck=40.0, head=4.0,
                     # near arm (holds the child's hand), far arm (holds the torch)
                     n_ua=176.0, n_fa=150.0, n_h=160.0,
                     f_ua=158.0, f_fa=70.0, f_h=40.0, torch_ang=6.0,
                     n_th=182.0, n_sh=178.0, f_th=176.0, f_sh=183.0,
                     breath=0.0, hem_wind=0.05, torch=True)


def hand_fist(g, H, d_ang, size=1.0, k=0.006):
    """Curled hand: palm + knuckle bumps + thumb, oriented along dirv(d_ang)."""
    d = dirv(d_ang)
    nrm = np.array([-d[1], d[0]])
    g.ellipse(H, 0.030 * size, 0.040 * size, rot=d_ang, k=k)
    for q in range(4):
        base = H + d * 0.028 * size + nrm * (q - 1.5) * 0.013 * size
        g.ellipse(base - nrm * 0.006 * size, 0.0105 * size, 0.012 * size, rot=d_ang, k=k)
    g.cone(H - nrm * 0.024 * size, H - nrm * 0.030 * size + d * 0.028 * size, 0.010 * size, 0.008 * size, k=k)


def elder(pose, t, scarf_pts=None, facing=-1, wisps=None):
    """Returns (groups, anchors)."""
    p = dict(ELDER_DEFAULT)
    p.update(pose)
    x0 = p['x']
    P0 = np.array([x0, p['hip_y'] + 0.004 * p['breath']])
    thl, tht, thn, thh = p['lumbar'], p['thorax'] + 1.5 * p['breath'], p['neck'], p['head']
    P1 = P0 + dirv(thl) * 0.20
    P2 = P1 + dirv(tht) * 0.24
    P3 = P2 + dirv(thn) * 0.06
    head_c = P3 + rot(np.array([-0.014, 0.068]), thh)
    body = Group('elder_body', COAT_ELDER, k=0.03, bevel=0.045, sheen=1.3, sky_rim=0.30)
    farg = Group('elder_far', COAT_ELDER, k=0.02, bevel=0.04, sheen=1.3, sky_rim=0.25)
    near = Group('elder_near', COAT_ELDER, k=0.02, bevel=0.04, sheen=1.3, sky_rim=0.25)
    skin = Group('elder_skin', SKIN, k=0.006, bevel=0.025, sheen=0.9, sky_rim=0.10)
    hair = Group('elder_hair', HAIR_GREY, k=0.010, bevel=0.02, sheen=1.6, sky_rim=0.45)
    wrap = Group('elder_scarfwrap', SCARF, k=0.02, bevel=0.035, sheen=0.9, sky_rim=0.15)
    tail = Group('elder_scarftail', SCARF, k=0.010, bevel=0.015, sheen=0.9, sky_rim=0.15, translucent=0.5)
    legs = Group('elder_legs', BOOT, k=0.015, bevel=0.03, sheen=0.8)
    torch_g = Group('torch', TORCH_WOOD, k=0.004, bevel=0.010, sheen=0.6, sky_rim=0.1)
    b = lambda th: backv(th)
    f = lambda th: -backv(th)
    # ---- legs (visible below the hem)
    for (hp, th, sh, sc) in [(P0 + np.array([0.035, -0.02]), p['f_th'], p['f_sh'], 0.95),
                             (P0 + np.array([-0.035, -0.02]), p['n_th'], p['n_sh'], 1.0)]:
        K = hp + dirv(th) * 0.39
        A = K + dirv(sh) * 0.39
        legs.cone(K, A, 0.046 * sc, 0.033 * sc)
        legs.cone(A + np.array([0.025, -0.028]), A + np.array([-0.11, -0.034]), 0.030, 0.022, k=0.02)
    # ---- coat: collar, hunched back, A-line to mid calf
    hw = p['hem_wind']
    fl = 0.012 * fnoise1(t * 2.3, 11.0, 2)
    fl2 = 0.010 * fnoise1(t * 3.1, 13.0, 2)
    coat = [
        P2 + f(tht) * 0.046 + dirv(tht) * 0.012,          # throat
        lerp(P1, P2, 0.55) + f(tht) * 0.100,               # chest
        P1 + f(thl) * 0.100,                               # belly
        P0 + f(thl) * 0.108 + np.array([0.0, -0.10]),      # front hip
        np.array([x0 - 0.185 + hw * 0.35, 0.235]),          # hem front
        np.array([x0 - 0.06 + hw * 0.6, 0.215 + fl]),
        np.array([x0 + 0.09 + hw * 0.85, 0.22 + fl2]),
        np.array([x0 + 0.225 + hw * 1.1, 0.215 + fl]),       # hem back (flared)
        P0 + b(thl) * 0.125 + np.array([0.0, -0.12]),       # seat
        P1 + b(thl) * 0.108,                                # mid back
        lerp(P1, P2, 0.45) + b(tht) * 0.122,                # hump
        P2 + b(tht) * 0.085 + dirv(tht) * 0.02,             # high back
    ]
    body.poly(coat, r=0.012)
    body.cone(P2 + dirv(tht) * 0.01, P3, 0.050, 0.040)     # neck (mostly hidden by scarf)
    body.ellipse(head_c, 0.078, 0.090, rot=thh)
    S = P2 + dirv(tht + 180) * 0.065
    body.ellipse(S + b(tht) * 0.01, 0.070, 0.058, rot=tht)
    # ---- hair (grey cap + bun), face
    hair.ellipse(head_c + rot(np.array([0.010, 0.018]), thh), 0.080, 0.086, rot=thh)
    hair.ellipse(head_c + rot(np.array([0.070, 0.048]), thh), 0.036, 0.034)
    if wisps is not None:
        for wp in wisps:
            hair.chain(wp, np.linspace(0.004, 0.0012, len(wp)), k=0.003)
    fp = face_poly(ELDER_FACE, head_c, thh)
    skin.poly(fp, r=0.002, k=0.006)
    skin.ellipse(head_c + rot(np.array([-0.048, -0.030]), thh), 0.036, 0.048, rot=thh)
    # ---- arms
    def arm(g, ua, fa):
        E = S + dirv(ua) * 0.26
        W = E + dirv(fa) * 0.225
        g.cone(S, E, 0.054, 0.046)
        g.cone(E, W, 0.046, 0.040)
        g.ellipse(W, 0.046, 0.034, rot=fa)   # cuff
        return E, W
    Ef, Wf = arm(farg, p['f_ua'], p['f_fa'])
    En, Wn = arm(near, p['n_ua'], p['n_fa'])
    Hf = Wf + dirv(p['f_h']) * 0.045
    Hn = Wn + dirv(p['n_h']) * 0.045
    # ---- torch in the far hand
    torch_top = None
    if p['torch']:
        tdir = dirv(p['torch_ang'])
        tb = Hf - tdir * 0.30
        tt = Hf + tdir * 0.16
        torch_g.cone(tb, tt, 0.013, 0.017)
        hg = tt + tdir * 0.055
        torch_g.cone(tt, hg + tdir * 0.02, 0.024, 0.029, k=0.008)
        torch_top = hg + tdir * 0.035
    hand_fist(skin, Hf, p['f_h'])
    hand_fist(skin, Hn, p['n_h'])
    # ---- scarf: bulky wrap at the neck + front drape
    wrap.ellipse(P2 + dirv(thn) * 0.025 + b(thn) * 0.005, 0.080, 0.056, rot=thn - 90)
    wrap.ellipse(P2 + dirv(thn) * 0.065, 0.064, 0.044, rot=thn - 95)
    dr0 = P2 + f(tht) * 0.060
    wrap.cone(dr0, dr0 + np.array([-0.015 + hw * 0.3, -0.16]), 0.036, 0.028)
    scarf_anchor = P2 + b(tht) * 0.070 + dirv(thn) * 0.04
    if scarf_pts is not None:
        ribbon(tail, scarf_pts, 0.18, t, seed=3.0)
    groups = [tail, farg, torch_g, legs, body, hair, skin, wrap, near]
    anchors = dict(torch_top=torch_top, scarf_anchor=scarf_anchor, hand_near=Hn, hand_far=Hf,
                   head_center=head_c, face_front=head_c + rot(np.array([-0.10, -0.03]), thh),
                   neck=P2, chest=lerp(P1, P2, 0.6), bun=head_c + rot(np.array([0.070, 0.048]), thh))
    if facing == 1:
        mirror_groups(groups, x0)
        for k, v in anchors.items():
            if v is not None:
                anchors[k] = np.array([2 * x0 - v[0], v[1]])
    return groups, anchors


# ------------------------------------------------------------------ CHILD ---

CHILD_DEFAULT = dict(x=0.0, hip_y=0.50, lean=0.0, head=0.0, head_yaw=90.0,
                     n_ua=180.0, n_fa=175.0, f_ua=180.0, f_fa=178.0,
                     n_th=180.0, n_sh=180.0, f_th=178.0, f_sh=180.0, breath=0.0, hat_pom=(0.0, 0.0))


def child(pose, t, facing=-1):
    """head_yaw: +90 profile in the body's facing direction, -90 looking back over the
    shoulder, 0 = facing camera (profile features collapse)."""
    p = dict(CHILD_DEFAULT)
    p.update(pose)
    x0 = p['x']
    body = Group('child_body', COAT_CHILD, k=0.03, bevel=0.045, sheen=1.3, sky_rim=0.30)
    farg = Group('child_far', COAT_CHILD, k=0.02, bevel=0.035, sheen=1.3, sky_rim=0.25)
    near = Group('child_near', COAT_CHILD, k=0.02, bevel=0.035, sheen=1.3, sky_rim=0.25)
    skin = Group('child_skin', SKIN_CHILD, k=0.006, bevel=0.025, sheen=0.9, sky_rim=0.1)
    hat = Group('child_hat', WOOL_HAT, k=0.012, bevel=0.03, sheen=1.8, sky_rim=0.45)
    legs = Group('child_legs', BOOT, k=0.02, bevel=0.03, sheen=0.8)
    lean = p['lean']
    P0 = np.array([x0, p['hip_y']])
    P1 = P0 + dirv(lean) * 0.13
    P2 = P1 + dirv(lean * 1.2) * 0.15 + np.array([0, 0.004 * p['breath']])     # neck base
    thh = p['head']
    neck_top = P2 + dirv(lean * 0.5 + thh * 0.35) * 0.035
    head_c = neck_top + rot(np.array([-0.004, 0.082]), thh)
    yaw = math.radians(p['head_yaw'])
    depth = math.sin(yaw)
    # legs
    for (dx, th, sh, sc) in [(0.03, p['f_th'], p['f_sh'], 0.96), (-0.03, p['n_th'], p['n_sh'], 1.0)]:
        hp = P0 + np.array([dx, -0.03])
        K = hp + dirv(th) * 0.23
        A = K + dirv(sh) * 0.215
        legs.cone(hp, K, 0.056 * sc, 0.047 * sc)
        legs.cone(K, A, 0.047 * sc, 0.040 * sc)
        legs.ellipse(A + np.array([-0.030, -0.018]), 0.068, 0.040)
    # puffy jacket (short, round)
    b = lambda th: backv(th)
    f = lambda th: -backv(th)
    coat = [P2 + f(lean) * 0.050, lerp(P1, P2, 0.5) + f(lean) * 0.118, P1 + f(lean) * 0.125,
            P0 + f(lean) * 0.118 + np.array([0, -0.035]), P0 + np.array([-0.07, -0.085]),
            P0 + np.array([0.08, -0.085]), P0 + b(lean) * 0.118 + np.array([0, -0.035]),
            P1 + b(lean) * 0.122, lerp(P1, P2, 0.5) + b(lean) * 0.112, P2 + b(lean) * 0.060]
    body.poly(coat, r=0.034)
    body.cone(P2, neck_top, 0.050, 0.042)
    body.ellipse(head_c, 0.088, 0.092, rot=thh)
    S = P2 + np.array([0.0, -0.045])
    body.ellipse(S, 0.072, 0.052)
    # face (profile collapses with yaw)
    if abs(depth) > 0.05:
        prof = np.array(CHILD_FACE)
        if depth < 0:
            prof[:, 0] = -prof[:, 0]
        fp = face_poly(prof, head_c, thh, depth=abs(depth))
        skin.poly(fp, r=0.002, k=0.006)
    skin.ellipse(head_c + rot(np.array([-0.045 * depth, -0.028]), thh), 0.046, 0.050, rot=thh)
    # knitted hat: dome + folded cuff + pom-pom on a sprung offset
    hat.ellipse(head_c + rot(np.array([0.010 * depth, 0.044]), thh), 0.096, 0.076, rot=thh)
    hat.box(head_c + rot(np.array([0.008 * depth, 0.006]), thh), 0.097, 0.022, rot=thh - 6 * depth, rnd=0.010)
    pom_base = head_c + rot(np.array([0.004 * depth, 0.116]), thh)
    pom = pom_base + np.asarray(p['hat_pom'])
    hat.cone(pom_base + rot(np.array([0, -0.02]), thh), pom, 0.013, 0.013, k=0.012)
    # fuzzy pom-pom: core + a ring of little lobes
    hat.ellipse(pom, 0.036, 0.034, k=0.01)
    for q in range(9):
        ang = q * 2 * math.pi / 9 + 0.3
        hat.ellipse(pom + 0.030 * np.array([math.cos(ang), math.sin(ang)]), 0.011, 0.011, k=0.012)
    # arms
    def arm(g, ua, fa):
        E = S + dirv(ua) * 0.155
        W = E + dirv(fa) * 0.14
        g.cone(S, E, 0.045, 0.040)
        g.cone(E, W, 0.040, 0.036)
        return E, W, W + dirv(fa) * 0.03
    Ef, Wf, Hf = arm(farg, p['f_ua'], p['f_fa'])
    En, Wn, Hn = arm(near, p['n_ua'], p['n_fa'])
    skin.ellipse(Hf, 0.025, 0.029, rot=p['f_fa'])
    skin.ellipse(Hn, 0.025, 0.029, rot=p['n_fa'])
    groups = [farg, legs, body, skin, hat, near]
    anchors = dict(hand_near=Hn, hand_far=Hf, head_center=head_c, pom=pom, pom_base=pom_base,
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
                  flint=False)


def young_woman(pose, t, scarf_pts=None, hair_pts=None, facing=-1):
    p = dict(YW_DEFAULT)
    p.update(pose)
    x0 = p['x']
    P0 = np.array([x0, p['hip_y']])
    thl, tht, thn, thh = p['lumbar'], p['thorax'] + 1.2 * p['breath'], p['neck'], p['head']
    P1 = P0 + dirv(thl) * 0.23
    P2 = P1 + dirv(tht) * 0.27
    P3 = P2 + dirv(thn) * 0.085
    head_c = P3 + rot(np.array([-0.010, 0.072]), thh)
    body = Group('yw_body', PARKA, k=0.03, bevel=0.045, sheen=1.3, sky_rim=0.30)
    farg = Group('yw_far', PARKA, k=0.02, bevel=0.04, sheen=1.3, sky_rim=0.25)
    near = Group('yw_near', PARKA, k=0.02, bevel=0.04, sheen=1.3, sky_rim=0.25)
    skin = Group('yw_skin', SKIN, k=0.006, bevel=0.025, sheen=0.9, sky_rim=0.1)
    hair = Group('yw_hair', HAIR_DARK, k=0.010, bevel=0.015, sheen=1.6, sky_rim=0.4, translucent=0.25)
    wrap = Group('yw_scarfwrap', SCARF, k=0.02, bevel=0.035, sheen=0.9, sky_rim=0.15)
    tail = Group('yw_scarftail', SCARF, k=0.010, bevel=0.015, sheen=0.9, sky_rim=0.15, translucent=0.5)
    legs = Group('yw_legs', BOOT, k=0.02, bevel=0.035, sheen=0.9)
    tools = Group('yw_tools', IRON, k=0.003, bevel=0.008, sheen=1.2, sky_rim=0.1, per_prim=True)
    b = lambda th: backv(th)
    f = lambda th: -backv(th)
    feet = []
    for (hp, th, sh, sc) in [(P0 + np.array([0.03, -0.03]), p['f_th'], p['f_sh'], 0.95),
                             (P0 + np.array([-0.03, -0.03]), p['n_th'], p['n_sh'], 1.0)]:
        K = hp + dirv(th) * 0.45
        A = K + dirv(sh) * 0.44
        legs.cone(hp, K, 0.066 * sc, 0.050 * sc)
        legs.cone(K, A, 0.050 * sc, 0.040 * sc)
        fdir = dirv(sh - 90) if abs(sh - 180) < 50 else dirv(sh)
        legs.cone(A + np.array([0.012, -0.02]), A + fdir * 0.135 + np.array([0, -0.03]), 0.043, 0.032, k=0.02)
        feet.append((K, A))
    hw = p['hem_wind']
    fl = 0.015 * fnoise1(t * 2.7, 21.0, 2)
    coat = [
        P2 + f(tht) * 0.052 + dirv(tht) * 0.012,
        lerp(P1, P2, 0.6) + f(tht) * 0.118,
        P1 + f(thl) * 0.108,
        P0 + f(thl) * 0.114 + np.array([0, -0.10]),
        P0 + f(thl) * 0.125 + dirv(thl + 180) * 0.30 + np.array([hw * 0.3, 0]),
        P0 + dirv(thl + 180) * 0.32 + np.array([hw * 0.6, fl]),
        P0 + b(thl) * 0.14 + dirv(thl + 180) * 0.29 + np.array([hw, -fl]),
        P0 + b(thl) * 0.13 + np.array([0, -0.06]),
        P1 + b(thl) * 0.108,
        lerp(P1, P2, 0.55) + b(tht) * 0.115,
        P2 + b(tht) * 0.080 + dirv(tht) * 0.015,
    ]
    body.poly(coat, r=0.016)
    body.cone(P2, P3, 0.048, 0.039)
    body.ellipse(head_c, 0.076, 0.092, rot=thh)
    S = P2 + dirv(tht + 180) * 0.06
    body.ellipse(S, 0.074, 0.060, rot=tht)
    body.ellipse(P2 + b(tht) * 0.080 + dirv(tht) * 0.015, 0.085, 0.066, rot=tht + 20)   # hood
    fp = face_poly(YW_FACE, head_c, thh)
    skin.poly(fp, r=0.002, k=0.006)
    skin.ellipse(head_c + rot(np.array([-0.045, -0.022]), thh), 0.034, 0.05, rot=thh)
    # hair: cap over the skull + blown strands merging at the root
    hair.ellipse(head_c + rot(np.array([0.012, 0.028]), thh), 0.082, 0.080, rot=thh)
    hair.ellipse(head_c + rot(np.array([0.055, -0.01]), thh), 0.05, 0.07, rot=thh + 20)
    if hair_pts is not None:
        for q, hp_ in enumerate(hair_pts):
            n = len(hp_)
            r0 = 0.020 if q % 3 == 0 else 0.013
            hair.chain(hp_, np.linspace(r0, 0.0025, n), k=0.018)
    def arm(g, ua, fa):
        E = S + dirv(ua) * 0.28
        W = E + dirv(fa) * 0.245
        g.cone(S, E, 0.058, 0.049)
        g.cone(E, W, 0.049, 0.042)
        g.ellipse(W, 0.047, 0.036, rot=fa)
        return E, W
    Ef, Wf = arm(farg, p['f_ua'], p['f_fa'])
    En, Wn = arm(near, p['n_ua'], p['n_fa'])
    Hf = Wf + dirv(p['f_h']) * 0.045
    Hn = Wn + dirv(p['n_h']) * 0.045
    hand_fist(skin, Hf, p['f_h'], 1.0)
    hand_fist(skin, Hn, p['n_h'], 1.0)
    flint_pt = None
    if p['flint']:
        # flint in the far hand (angular dark stone), C-shaped steel in the near hand
        d = dirv(p['f_h'])
        nrm = np.array([-d[1], d[0]])
        c = Hf + d * 0.045
        tools.poly([c - nrm * 0.03, c + d * 0.035 - nrm * 0.01, c + d * 0.03 + nrm * 0.025,
                    c + nrm * 0.028 - d * 0.01], r=0.002)
        d2 = dirv(p['n_h'])
        c2 = Hn + d2 * 0.02
        tools.ring(c2, 0.045, 0.022, 0.009, rot=p['n_h'] - 90)
        flint_pt = c + d * 0.03
    # scarf
    wrap.ellipse(P2 + dirv(thn) * 0.03, 0.082, 0.058, rot=thn - 90)
    wrap.ellipse(P2 + dirv(thn) * 0.075 + f(thn) * 0.004, 0.066, 0.045, rot=thn - 95)
    dr0 = P2 + f(tht) * 0.07
    wrap.cone(dr0, dr0 + np.array([-0.02 + hw * 0.3, -0.17]), 0.038, 0.030)
    scarf_anchor = P2 + b(tht) * 0.075 + dirv(thn) * 0.045
    if scarf_pts is not None:
        ribbon(tail, scarf_pts, 0.18, t, seed=3.0)
    groups = [tail, farg, legs, body, hair, skin, wrap, near, tools]
    anchors = dict(scarf_anchor=scarf_anchor, hand_near=Hn, hand_far=Hf, head_center=head_c,
                   mouth=head_c + rot(np.array([-0.092, -0.054]), thh), neck=P2,
                   hair_root=head_c + rot(np.array([0.055, 0.020]), thh), knee_near=feet[1][0],
                   chest=lerp(P1, P2, 0.6), flint=flint_pt)
    if facing == 1:
        mirror_groups(groups, x0)
        for k, v in anchors.items():
            if v is not None:
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
        st = Group('cairn', STONE, k=0.006, bevel=0.03, sheen=0.5, sky_rim=0.45, per_prim=True)
        st2 = Group('cairn2', STONE * 0.6, k=0.006, bevel=0.03, sheen=0.5, sky_rim=0.45, per_prim=True)
        for q, (cx, cy, rx, ry, r) in enumerate(self.stones):
            g = st if (q * 7) % 3 else st2
            g.box((x + cx, cy), rx * 0.86, ry * 0.72, rot=r * 0.6, rnd=min(rx, ry) * 0.30, k=0.006)
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
        return [wood, st, st2, iron]

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
