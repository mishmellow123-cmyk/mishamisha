"""Her thin leather glove for the Ring close-ups (Blender side, numpy-free).

A low-poly subdivision cage (palm tube + four finger tubes + thumb tube, ~330 verts) deformed by an armature
(dual-quaternion skinning) and then subdivided, so every pose is one smooth, closed leather surface with real
knuckles and creases. No per-frame meshes: poses are keyed bone rotations, so Cycles gets true deformation
motion blur, and a cloud job needs nothing but bpy.

Hand-local frame for a RIGHT hand (mirror=True gives her LEFT hand): wrist at the origin, fingers toward +x,
the back of the hand +z (the palm faces -z), the thumb toward +y. Metres. A woman's hand in a thin glove:
palm 8 cm across the knuckles, middle finger 9.5 cm from its knuckle joint.

  G = Glove('gl', leather, sleeve_mat, mirror=True)
  G.key(frame, pose)         # pose: see POSES / pose_mix; angles in degrees, + = flexion toward the palm
  G.rig.matrix_world = M     # place and key the hand like any object (G.key_place(frame, M))
"""
import math

import bpy
from mathutils import Matrix, Quaternion, Vector

FING = ('index', 'middle', 'ring', 'little')
# centre y, knuckle x, (prox, mid, dist) lengths, half width, half height at the base, splay (deg, + toward the thumb)
# (v2, MONTAGE-3D-5: fingers slimmed ~16 % and tapered to the tip; the old ones read as fat sausages)
FD = {'index': (0.0300, 0.0885, (0.0385, 0.0235, 0.0212), 0.0091, 0.0083, 3.0),
      'middle': (0.0100, 0.0925, (0.0425, 0.0262, 0.0220), 0.0093, 0.0085, 0.0),
      'ring': (-0.0100, 0.0895, (0.0398, 0.0252, 0.0212), 0.0089, 0.0081, -3.5),
      'little': (-0.0290, 0.0805, (0.0315, 0.0197, 0.0193), 0.0079, 0.0072, -8.0)}
COLS = (0.0400, 0.0300, 0.0200, 0.0100, 0.0000, -0.0100, -0.0200, -0.0290, -0.0375)   # knuckle-line columns
# thumb chain in the rest pose: CMC (inside the palm), MCP, IP, tip
TH = (Vector((0.022, 0.021, -0.010)), Vector((0.050, 0.057, -0.019)), Vector((0.0725, 0.0705, -0.0225)),
      Vector((0.0960, 0.0785, -0.0245)))
CORNER = 0.88                      # pull on the cage corners: a rounded box after subdivision

# the finger-ring pattern (v across toward the thumb side, w toward the back of the hand), matching the
# order of each finger's base ring on the palm grid: T(2i), T(2i+1), T(2i+2), M(2i+2), B(2i+2), B(2i+1), B(2i), M(2i)
PAT = ((1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1), (1, 0))


def _ring(c, ev, ew, hw, hh, pad=0.0):
    out = []
    for v, w in PAT:
        k = CORNER if (v and w) else 1.0
        ww = w * hh * k
        if w < 0:
            ww -= pad                  # a fuller pad on the palm side
        out.append(c + ev * (v * hw * k) + ew * ww)
    return out


def _palm_ring(x, a, t, b, xs=None, zt=None, zb=None, ys=None):
    """20 points: T0..T8 (y from +a to -a), S8, B8..B0, S0. xs/zt/zb/ys override per column."""
    ys = ys or [a * (1.0 - 2.0 * j / 8.0) for j in range(9)]
    T, B = [], []
    for j in range(9):
        y = ys[j]
        e = min(1.0, abs(y) / (a * 1.02))
        if j in (0, 8):
            y *= 0.95
        top = zt[j] if zt else t * max(0.52, (1.0 - e ** 3) ** (1.0 / 3.0))
        bot = zb[j] if zb else b * max(0.52, (1.0 - e ** 3) ** (1.0 / 3.0))
        xx = xs[j] if xs else x
        T.append(Vector((xx, y, top)))
        B.append(Vector((xx, y, bot)))
    xm8 = xs[8] if xs else x
    xm0 = xs[0] if xs else x
    S8 = Vector((xm8, -abs(ys[8]) * 1.04 if ys else -a * 1.04, 0.22 * (t + b)))
    S0 = Vector((xm0, abs(ys[0]) * 1.04 if ys else a * 1.04, 0.22 * (t + b)))
    return T + [S8] + B[::-1] + [S0]


class Cage:
    """Vertices, quads and per-vertex bone weights of the rest-pose glove."""

    def __init__(self):
        self.V, self.F, self.W, self.seam = [], [], [], []
        self.bones = {}                 # name -> (head, tail, roll_up, parent)

    def add(self, p, weights, seam=0.0):
        self.V.append(Vector(p))
        self.W.append(weights)
        self.seam.append(seam)
        return len(self.V) - 1

    def quad_strip(self, A, B):
        n = len(A)
        for i in range(n):
            self.F.append((A[i], A[(i + 1) % n], B[(i + 1) % n], B[i]))


def build_cage():
    C = Cage()
    H = {'hand': 1.0}
    # ---- cuff, wrist and palm rings (x, half width, top, bottom)
    rows = [(-0.072, 0.0330, 0.0205, -0.0240), (-0.046, 0.0300, 0.0188, -0.0222), (-0.019, 0.0282, 0.0178, -0.0222),
            (0.004, 0.0322, 0.0184, -0.0262), (0.026, 0.0370, 0.0180, -0.0240), (0.048, 0.0402, 0.0168, -0.0212),
            (0.068, 0.0414, 0.0155, -0.0178)]
    rings = []
    for (x, a, t, b) in rows:
        pts = _palm_ring(x, a, t, b)
        for q in pts[10:19]:                          # the palm side (B8..B0): thenar and hypothenar pads, a hollow
            y = q.y
            q.z -= 0.0055 * math.exp(-((y - 0.024) / 0.012) ** 2) * max(0.0, 1.0 - max(0.0, x - 0.0) / 0.058) * (x > -0.01)
            q.z -= 0.0032 * math.exp(-((y + 0.027) / 0.010) ** 2) * max(0.0, 1.0 - max(0.0, x - 0.005) / 0.07) * (x > -0.01)
            q.z += 0.0022 * math.exp(-((y - 0.004) / 0.014) ** 2) * math.exp(-((x - 0.050) / 0.018) ** 2)
        rings.append([C.add(p, H) for p in pts])
    # ---- the knuckle line (front ring): columns follow each finger's knuckle
    fx = {f: FD[f][1] for f in FING}
    fhh = {f: FD[f][4] for f in FING}
    order = ('index', 'middle', 'ring', 'little')
    xs, zt, zb = [], [], []
    for j in range(9):
        if j % 2 == 1:
            f = order[j // 2]
            xs.append(fx[f])
            zt.append(fhh[f] + 0.0017)
            zb.append(-fhh[f] * 1.08)
        elif j in (0, 8):
            f = order[0] if j == 0 else order[3]
            xs.append(fx[f] - 0.004)
            zt.append(fhh[f] * 0.62)
            zb.append(-fhh[f] * 0.66)
        else:
            f1, f2 = order[j // 2 - 1], order[j // 2]
            xs.append(0.5 * (fx[f1] + fx[f2]) - 0.0015)
            zt.append(0.84 * 0.5 * (fhh[f1] + fhh[f2]))
            zb.append(-0.86 * 0.5 * (fhh[f1] + fhh[f2]))
    front = _palm_ring(0.09, 0.04, 0.0, 0.0, xs=xs, zt=zt, zb=zb, ys=list(COLS))
    front_i = [C.add(p, H) for p in front]
    rings.append(front_i)
    for k in range(len(rings) - 1):
        A, B_ = rings[k], rings[k + 1]
        for i in range(20):
            i2 = (i + 1) % 20
            if 4 <= k <= 5 and i in (18, 19):         # the thumb's opening (rings 4..6 on the +y side)
                continue
            C.F.append((A[i], A[i2], B_[i2], B_[i]))
    # webbing vertices between the fingers (front-grid middle row, columns 2, 4, 6)
    web = {}
    for j in (2, 4, 6):
        f1, f2 = order[j // 2 - 1], order[j // 2]
        web[j] = C.add((0.5 * (fx[f1] + fx[f2]) + 0.0115, COLS[j], -0.0042), H, seam=1.0)
    Tf = front_i[0:9]
    Bf = front_i[10:19][::-1]                          # B0..B8
    M = {0: front_i[19], 8: front_i[9], 2: web[2], 4: web[4], 6: web[6]}
    # ---- fingers
    for i, f in enumerate(order):
        yc, xk, L, hw0, hh0, spl = FD[f]
        s = math.radians(spl)
        u = Vector((math.cos(s), math.sin(s), 0.0))
        ew = Vector((0.0, 0.0, 1.0))
        ev = ew.cross(u)
        mcp = Vector((xk - 0.0060, yc, -0.0006))
        pip, dip = mcp + u * L[0], mcp + u * (L[0] + L[1])
        tip = mcp + u * sum(L)
        pre = f[:1]
        C.bones[pre + '1'] = (mcp, pip, ew, 'hand')
        C.bones[pre + '2'] = (pip, dip, ew, pre + '1')
        C.bones[pre + '3'] = (dip, tip, ew, pre + '2')
        base = [Tf[2 * i], Tf[2 * i + 1], Tf[2 * i + 2], M[2 * i + 2], Bf[2 * i + 2], Bf[2 * i + 1], Bf[2 * i], M[2 * i]]
        b1, b2, b3 = pre + '1', pre + '2', pre + '3'
        Lt = sum(L)
        spec = [(0.0125, 1.00, 1.00, {'hand': 0.25, b1: 0.75}),
                (L[0] - 0.0045, 0.965, 0.955, {b1: 0.8, b2: 0.2}),
                (L[0] + 0.0045, 0.935, 0.925, {b1: 0.2, b2: 0.8}),
                (L[0] + L[1] - 0.0038, 0.865, 0.85, {b2: 0.8, b3: 0.2}),
                (L[0] + L[1] + 0.0038, 0.845, 0.83, {b2: 0.2, b3: 0.8}),
                (Lt - 0.0080, 0.79, 0.77, {b3: 1.0}),
                (Lt - 0.0030, 0.65, 0.62, {b3: 1.0})]
        prev = base
        for d, kw, kh, wts in spec:
            c = mcp + u * d + ew * (-0.0008 * d / Lt)
            pad = 0.0009 if d > 0.02 else 0.0
            pts = _ring(c, ev, ew, hw0 * kw, hh0 * kh, pad)
            cur = [C.add(p, wts, seam=(1.0 if k in (3, 7) else 0.0)) for k, p in enumerate(pts)]
            C.quad_strip(prev, cur)
            prev = cur
        cap = C.add(mcp + u * (Lt + 0.0010) + ew * (-0.0018), {b3: 1.0})
        for k in (0, 2, 4, 6):
            C.F.append((prev[k], prev[k + 1], prev[(k + 2) % 8], cap))
    # ---- thumb: base ring = the opening on the palm's +y side (rings 4..6), then a tube along the chain
    r4, r5, r6 = rings[4], rings[5], rings[6]
    base = [r4[0], r5[0], r6[0], r6[19], r6[18], r5[18], r4[18], r4[19]]
    # pattern (a toward +x, b toward the back) for that ring: T0_4 (-1,+1), T0_5 (0,+1), T0_6 (+1,+1), S0_6 (+1,0), ...
    TPAT = ((-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0))
    P0, P1, P2, P3 = TH
    C.bones['t1'] = (P0, P1, None, 'hand')
    C.bones['t2'] = (P1, P2, None, 't1')
    C.bones['t3'] = (P2, P3, None, 't2')
    nail_base, nail_tip = Vector((0.0, 0.0, 1.0)), Vector((-0.25, 0.62, 0.74)).normalized()

    def tframe(u, k):
        nb = nail_base.lerp(nail_tip, k).normalized()
        eb = (nb - u * nb.dot(u)).normalized()
        ea = u.cross(eb)
        return ea, eb

    u01, u12, u23 = (P1 - P0).normalized(), (P2 - P1).normalized(), (P3 - P2).normalized()
    tspec = [(Vector((0.0465, 0.0500, -0.0075)), u01, 0.15, 0.0190, 0.0152, {'hand': 0.3, 't1': 0.7}),
             (P1 - u01 * 0.0048, u01, 0.35, 0.0112, 0.0098, {'t1': 0.8, 't2': 0.2}),
             (P1 + u12 * 0.0048, u12, 0.45, 0.0107, 0.0094, {'t1': 0.2, 't2': 0.8}),
             (P2 - u12 * 0.0040, u12, 0.7, 0.0101, 0.0087, {'t2': 0.8, 't3': 0.2}),
             (P2 + u23 * 0.0040, u23, 0.8, 0.0097, 0.0082, {'t2': 0.2, 't3': 0.8}),
             (P3 - u23 * 0.0085, u23, 0.95, 0.0086, 0.0072, {'t3': 1.0}),
             (P3 - u23 * 0.0031, u23, 1.0, 0.0068, 0.0057, {'t3': 1.0})]
    prev = base
    for c, u, kf, hw, hh, wts in tspec:
        ea, eb = tframe(u, kf)
        pts = []
        for a_, b_ in TPAT:
            kc = CORNER if (a_ and b_) else 1.0
            bb = b_ * hh * kc - (0.0008 if b_ < 0 else 0.0)
            pts.append(c + ea * (a_ * hw * kc) + eb * bb)
        cur = [C.add(p, wts, seam=(1.0 if k in (1, 5) else 0.0)) for k, p in enumerate(pts)]
        C.quad_strip(prev, cur)
        prev = cur
    ea, eb = tframe(u23, 1.0)
    cap = C.add(P3 - u23 * 0.0006 - eb * 0.0018, {'t3': 1.0})
    for k in (0, 2, 4, 6):
        C.F.append((prev[k], prev[k + 1], prev[(k + 2) % 8], cap))
    C.bones['hand'] = (Vector((-0.03, 0.0, 0.0)), Vector((0.06, 0.0, 0.0)), Vector((0.0, 0.0, 1.0)), None)
    C.thumb_up = {'t1': tframe(u01, 0.15)[1], 't2': tframe(u12, 0.45)[1], 't3': tframe(u23, 0.9)[1]}
    return C


def leather_material(new_material, base=(0.050, 0.030, 0.018), rough=0.46, detail=True, thread=None):
    """Thin, supple leather: dark brown, a soft waxy sheen, a pebble grain you can see, the palm's working creases
    and (detail=True) what makes it read as a GLOVE at a glance: bunched wrinkles over every finger joint on the
    back, a flexion crease at each joint on the palm side, the knuckles burnished lighter by wear, the three
    stitched points on the back of the hand and stitch dashes along the finger side seams (thread: its colour).
    Everything lives in rest-pose coordinates (the 'rest' attribute), so it rides the leather as the hand moves."""
    m, nb = new_material('leather')
    R = nb.attr('rest').outputs['Vector']
    grain = nb.voronoi(R, scale=2600.0, feature='SMOOTH_F1')
    fine = nb.noise(R, scale=900.0, detail=4.0, rough=0.6)
    mottle = nb.noise(R, scale=45.0, detail=3.0, rough=0.5)
    seam = nb.attr('seam').outputs['Fac']
    ridge = nb.sstep(0.72, 0.93, seam)
    rx, ry, rz = nb.sep(R)
    palm = nb.sstep(-0.004, -0.010, rz)                     # the palm side only
    wob = nb.noise(R, scale=160.0, detail=2.0).outputs['Fac']
    x_ = nb.add(rx, nb.mul(nb.sub(wob, 0.5), 0.004))

    def line(x0, slope, curve, ymin, ymax, w):
        # a crease along x = x0 + slope*y + curve*y^2 over ymin..ymax
        xc = nb.add(nb.add(x0, nb.mul(ry, slope)), nb.mul(nb.mul(ry, ry), curve))
        d = nb.math('ABSOLUTE', nb.sub(x_, xc))
        span = nb.mul(nb.sstep(ymin - 0.004, ymin + 0.004, ry), nb.sstep(ymax + 0.004, ymax - 0.004, ry))
        return nb.mul(nb.sstep(w, 0.0, d), span)

    cr = line(0.071, -0.10, -6.0, -0.040, 0.020, 0.0024)                  # distal palm crease
    cr = nb.mx(cr, line(0.058, 0.22, 4.0, -0.030, 0.034, 0.0024))         # proximal palm crease
    cr = nb.mx(cr, line(0.030, 0.9, 0.0, 0.002, 0.034, 0.0022))           # round the thumb's mound
    cr = nb.mx(cr, line(0.012, 0.0, 0.0, -0.028, 0.028, 0.0026))          # the wrist
    cr = nb.mul(cr, palm)
    dk = nb.value(0.0).outputs[0]                    # joint wrinkles (dark grooves)
    pk = nb.value(0.0).outputs[0]                    # palm-side flexion creases
    wear = nb.value(0.0).outputs[0]                  # burnished knuckles
    stitch = nb.value(0.0).outputs[0]
    if detail:
        wob2 = nb.noise(R, scale=520.0, detail=2.0).outputs['Fac']
        dors = nb.sstep(0.0004, 0.0040, rz)
        palmf = nb.sstep(-0.0030, -0.0065, rz)
        for f in FING:
            yc, xk, L, hw0, hh0, spl = FD[f]
            sp_ = math.radians(spl)
            cs_, sn_ = math.cos(sp_), math.sin(sp_)
            dx = nb.sub(rx, xk - 0.0060)
            dy = nb.sub(ry, yc)
            a = nb.add(nb.mul(dx, cs_), nb.mul(dy, sn_))                  # along the finger from its knuckle joint
            v = nb.math('ABSOLUTE', nb.sub(nb.mul(dy, cs_), nb.mul(dx, sn_)))
            lat = nb.sstep(hw0 * 1.05, hw0 * 0.50, v)
            aw = nb.add(a, nb.mul(nb.sub(wob2, 0.5), 0.0012))              # the lines wander a little
            for aj, nl, spc, k in ((L[0], 3, 0.0015, 1.0), (L[0] + L[1], 2, 0.0013, 0.8), (0.0035, 2, 0.0018, 0.45)):
                d = nb.sub(aw, aj)
                ad = nb.math('ABSOLUTE', d)
                win = nb.sstep(spc * (0.5 * nl + 0.35), spc * 0.5 * nl * 0.55, ad)
                per = nb.math('ABSOLUTE', nb.math('SINE', nb.mul(d, math.pi / spc)))
                grooves = nb.mul(nb.sstep(0.42, 0.0, per), nb.mul(nb.mul(win, lat), dors))
                dk = nb.mx(dk, nb.mul(grooves, k))
                flex = nb.mul(nb.sstep(0.0011, 0.0, ad), nb.mul(lat, palmf))  # one deep flexion crease
                pk = nb.mx(pk, flex)
            for aj, wk in ((0.0, 1.0), (L[0], 0.7)):                      # the knuckles, burnished by wear
                q_ = nb.div(nb.sub(a, aj), 0.0065)
                blob = nb.exp(nb.mul(nb.mul(q_, q_), -1.0))
                wear = nb.mx(wear, nb.mul(nb.mul(blob, nb.mul(lat, dors)), wk))
            pk = nb.mx(pk, nb.mul(nb.mul(nb.sstep(0.0012, 0.0, nb.math('ABSOLUTE', nb.sub(aw, 0.012))), lat), palmf))
        # the three stitched points on the back of the hand, from the knuckle gaps toward the wrist
        back = nb.sstep(0.0085, 0.0125, rz)
        xr = nb.mul(nb.sstep(0.024, 0.030, rx), nb.sstep(0.074, 0.068, rx))
        ph = nb.math('FRACT', nb.div(rx, 0.0023))
        dash = nb.mul(nb.sstep(0.06, 0.16, ph), nb.sstep(0.74, 0.64, ph))
        yw = nb.add(ry, nb.mul(nb.sub(wob, 0.5), 0.0006))
        pts = nb.value(0.0).outputs[0]
        for y0 in (0.0205, 0.0, -0.0195):
            for dy_ in (-0.0017, 0.0017):
                pts = nb.mx(pts, nb.sstep(0.00042, 0.00010, nb.math('ABSOLUTE', nb.sub(yw, y0 + dy_))))
        stitch = nb.mul(nb.mul(pts, dash), nb.mul(xr, back))
        # stitch dashes along the finger side seams (the fourchettes)
        side = nb.mul(nb.sstep(0.80, 0.97, seam), nb.sstep(0.050, 0.070, rx))
        stitch = nb.mx(stitch, nb.mul(side, dash))
    rnd = nb.sstep(0.022, 0.0, nb.math('ABSOLUTE', nb.sub(nb.noise(R, scale=120.0, detail=2.0).outputs['Fac'], 0.5)))
    rnd = nb.mul(rnd, 0.5)                                                # a web of fine creases: worn, not new
    pebble = nb.noise(R, scale=420.0, detail=3.0, rough=0.55).outputs['Fac']
    tone = nb.mul(nb.madd(mottle.outputs['Fac'], 0.45, 0.78), nb.madd(ridge, -0.35, 1.0))
    tone = nb.mul(tone, nb.madd(cr, -0.16, 1.0))
    tone = nb.mul(tone, nb.madd(nb.mx(dk, pk), -0.45, 1.0))
    tone = nb.mul(tone, nb.madd(rnd, -0.25, 1.0))
    tone = nb.mul(tone, nb.madd(wear, 0.85, 1.0))
    col = nb.colscale(base, tone)
    col = nb.mixcol(nb.mul(wear, 0.35), col, (base[0] * 2.2, base[1] * 2.0, base[2] * 1.9))
    th = thread or (min(1.0, base[0] * 2.6), min(1.0, base[1] * 2.4), min(1.0, base[2] * 2.0))
    col = nb.mixcol(nb.mul(stitch, 0.85), col, th)
    rgh = nb.madd(fine.outputs['Fac'], 0.16, rough - 0.08)
    rgh = nb.sub(rgh, nb.mul(wear, 0.10))
    rgh = nb.add(rgh, nb.mul(nb.mx(dk, pk), 0.10))
    bs = nb.principled(Base_Color=col, Roughness=rgh)
    bs.inputs['Specular IOR Level'].default_value = 0.55
    bs.inputs['Coat Weight'].default_value = 0.18
    bs.inputs['Coat Roughness'].default_value = 0.34
    h = nb.add(nb.mul(grain.outputs['Distance'], 0.8), nb.mul(fine.outputs['Fac'], 0.3))
    h = nb.add(h, nb.mul(pebble, 0.9))
    h = nb.add(h, nb.mul(ridge, 2.2))
    h = nb.sub(h, nb.mul(cr, 1.1))
    h = nb.sub(h, nb.mul(dk, 1.8))
    h = nb.sub(h, nb.mul(pk, 1.5))
    h = nb.sub(h, nb.mul(rnd, 0.8))
    h = nb.add(h, nb.mul(stitch, 1.6))
    nb.link(nb.bump(h, 0.35, 0.00020), bs.inputs['Normal'])
    nb.output(surface=bs)
    return m


def wool_material(new_material, base=(0.030, 0.026, 0.024)):
    """Her coat sleeve: heavy dark wool, fuzzy sheen, no pattern."""
    m, nb = new_material('coatwool')
    P = nb.texco().outputs['Object']
    fuzz = nb.noise(P, scale=1800.0, detail=6.0, rough=0.8)
    weave = nb.noise(nb.vmath('MULTIPLY', P, (1.0, 4.0, 4.0)), scale=260.0, detail=2.0)
    col = nb.colscale(base, nb.madd(fuzz.outputs['Fac'], 0.5, 0.75))
    bs = nb.principled(Base_Color=col, Roughness=0.93)
    bs.inputs['Sheen Weight'].default_value = 0.8
    bs.inputs['Sheen Roughness'].default_value = 0.5
    bs.inputs['Sheen Tint'].default_value = (0.9, 0.85, 0.8, 1.0)
    nb.link(nb.bump(nb.add(fuzz.outputs['Fac'], nb.mul(weave.outputs['Fac'], 0.6)), 0.6, 0.0008), bs.inputs['Normal'])
    nb.output(surface=bs)
    return m


def _sleeve_mesh(name, mat, length=0.26, r_cuff=0.046, r_far=0.058, n_a=28, n_l=14, seed=3.0):
    """A loose wool sleeve over the glove's cuff, from the wrist back along -x, with a few soft folds."""
    V, F = [], []
    for j in range(n_l):
        t = j / (n_l - 1)
        x = -0.030 - t * length
        r = r_cuff + (r_far - r_cuff) * t
        for i in range(n_a):
            a = 2 * math.pi * i / n_a
            fold = 0.0035 * math.sin(3 * a + 1.3 + 4.0 * t + seed) + 0.0022 * math.sin(5 * a + 0.4 + 7.0 * t)
            sag = -0.004 * max(0.0, -math.sin(a)) * t
            rr = r + fold
            V.append((x, rr * math.cos(a), rr * math.sin(a) * 0.86 + sag))
    for j in range(n_l - 1):
        for i in range(n_a):
            i2 = (i + 1) % n_a
            F.append((j * n_a + i, j * n_a + i2, (j + 1) * n_a + i2, (j + 1) * n_a + i))
    me = bpy.data.meshes.new(name)
    me.from_pydata(V, [], F)
    me.update()
    me.shade_smooth()
    me.materials.append(mat)
    ob = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(ob)
    sd = ob.modifiers.new('sub', 'SUBSURF')
    sd.levels, sd.render_levels = 1, 2
    so = ob.modifiers.new('thick', 'SOLIDIFY')
    so.thickness = 0.004
    return ob


class Glove:
    """One gloved hand: rig (armature object; key its transform to move the hand), mesh (the glove), sleeve."""

    def __init__(self, name, leather, sleeve_mat=None, mirror=False, sub=3, sleeve=True):
        C = build_cage()
        self.cage = C
        self.mirror = mirror
        sc = bpy.context.scene
        arm = bpy.data.armatures.new(name + '_arm')
        rig = bpy.data.objects.new(name + '_rig', arm)
        sc.collection.objects.link(rig)
        bpy.context.view_layer.objects.active = rig
        rig.select_set(True)
        bpy.ops.object.mode_set(mode='EDIT')
        eb = {}
        order = ['hand'] + [k for k in C.bones if k != 'hand']
        for bn in order:
            h, t, up, par = C.bones[bn]
            b = arm.edit_bones.new(bn)
            b.head, b.tail = h, t
            if up is None:
                up = C.thumb_up[bn]
            b.align_roll(up)
            if par:
                b.parent = eb[par]
                b.use_connect = False
            eb[bn] = b
        bpy.ops.object.mode_set(mode='OBJECT')
        rig.select_set(False)
        me = bpy.data.meshes.new(name)
        me.from_pydata([tuple(v) for v in C.V], [], C.F)
        me.update()
        me.shade_smooth()
        me.materials.append(leather)
        at = me.attributes.new('seam', 'FLOAT', 'POINT')
        at.data.foreach_set('value', C.seam)
        rest = me.attributes.new('rest', 'FLOAT_VECTOR', 'POINT')
        rest.data.foreach_set('vector', [c for v in C.V for c in v])
        ob = bpy.data.objects.new(name, me)
        sc.collection.objects.link(ob)
        groups = {}
        for vi, wts in enumerate(C.W):
            for bn, w in wts.items():
                if bn not in groups:
                    groups[bn] = ob.vertex_groups.new(name=bn)
                groups[bn].add([vi], w, 'REPLACE')
        ob.parent = rig
        md = ob.modifiers.new('arm', 'ARMATURE')
        md.object = rig
        md.use_deform_preserve_volume = True
        sd = ob.modifiers.new('sub', 'SUBSURF')
        sd.levels, sd.render_levels = 2, sub
        self.rig, self.mesh = rig, ob
        if mirror:                                   # her left hand: mirror the whole rig across y (normals fixed)
            rig.scale = (1.0, -1.0, 1.0)
        self.sleeve = None
        if sleeve and sleeve_mat is not None:
            sl = _sleeve_mesh(name + '_sleeve', sleeve_mat)
            sl.parent = rig                          # the 'hand' bone never moves inside the rig: object parenting
            self.sleeve = sl
        for pb in rig.pose.bones:
            pb.rotation_mode = 'QUATERNION'

    # ------------------------------------------------------------------ posing
    def _q(self, bn, flex, side=0.0, twist=0.0):
        """Local rotation: flex about the bone's X (toward the palm = +flex), side about Z, twist about Y."""
        q = Quaternion((1.0, 0.0, 0.0), math.radians(-flex))
        if side:
            q = Quaternion((0.0, 0.0, 1.0), math.radians(side)) @ q
        if twist:
            q = q @ Quaternion((0.0, 1.0, 0.0), math.radians(twist))
        return q

    def set_pose(self, pose):
        pb = self.rig.pose.bones
        curl = pose.get('curl', {})
        spread = pose.get('spread', 0.0)
        for f in FING:
            a = curl.get(f, (0.0, 0.0, 0.0))
            sgn = {'index': 1.0, 'middle': 0.3, 'ring': -0.5, 'little': -1.0}[f]
            pb[f[:1] + '1'].rotation_quaternion = self._q(f[:1] + '1', a[0], side=sgn * spread)
            pb[f[:1] + '2'].rotation_quaternion = self._q(f[:1] + '2', a[1])
            pb[f[:1] + '3'].rotation_quaternion = self._q(f[:1] + '3', a[2])
        th = pose.get('thumb', (0.0, 0.0, 0.0, 0.0))     # (flex toward the palm, swing toward the index, mcp, ip)
        pb['t1'].rotation_quaternion = self._q('t1', th[0], side=-th[1])
        pb['t2'].rotation_quaternion = self._q('t2', th[2])
        pb['t3'].rotation_quaternion = self._q('t3', th[3])
        pb['hand'].rotation_quaternion = Quaternion()

    def key(self, frame, pose):
        self.set_pose(pose)
        for pb in self.rig.pose.bones:
            pb.keyframe_insert('rotation_quaternion', frame=frame)

    def key_place(self, frame, M):
        """Key the hand's placement (a 4x4 world matrix of the RIGHT-hand frame; mirroring is kept)."""
        loc, rot, _ = M.decompose()
        self.rig.location = loc
        self.rig.rotation_mode = 'QUATERNION'
        self.rig.rotation_quaternion = rot
        self.rig.keyframe_insert('location', frame=frame)
        self.rig.keyframe_insert('rotation_quaternion', frame=frame)


def pose_mix(a, b, t):
    """Blend two poses (dicts) by t (0 = a, 1 = b)."""
    out = {'curl': {}, 'spread': a.get('spread', 0.0) + (b.get('spread', 0.0) - a.get('spread', 0.0)) * t}
    for f in FING:
        pa = a.get('curl', {}).get(f, (0.0, 0.0, 0.0))
        pbv = b.get('curl', {}).get(f, (0.0, 0.0, 0.0))
        out['curl'][f] = tuple(x + (y - x) * t for x, y in zip(pa, pbv))
    ta, tb = a.get('thumb', (0, 0, 0, 0)), b.get('thumb', (0, 0, 0, 0))
    out['thumb'] = tuple(x + (y - x) * t for x, y in zip(ta, tb))
    return out


POSES = {
    'flat': {'curl': {f: (0.0, 0.0, 0.0) for f in FING}, 'thumb': (0.0, 0.0, 0.0, 0.0), 'spread': 0.0},
    'relaxed': {'curl': {'index': (12, 18, 10), 'middle': (16, 24, 12), 'ring': (20, 28, 14), 'little': (24, 32, 16)},
                'thumb': (8, 6, 10, 8), 'spread': 3.0},
    'reach': {'curl': {'index': (8, 14, 8), 'middle': (10, 18, 10), 'ring': (13, 22, 12), 'little': (16, 26, 14)},
              'thumb': (-6, 2, 6, 4), 'spread': 6.0},
    'fist': {'curl': {'index': (92, 108, 74), 'middle': (95, 110, 74), 'ring': (97, 110, 74), 'little': (99, 108, 72)},
             'thumb': (34, 40, 44, 38), 'spread': -3.0},
    'cup': {'curl': {'index': (22, 30, 16), 'middle': (24, 32, 16), 'ring': (26, 34, 18), 'little': (30, 36, 18)},
            'thumb': (10, 14, 12, 10), 'spread': 2.0},
}
