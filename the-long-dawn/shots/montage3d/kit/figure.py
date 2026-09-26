"""MONTAGE-3D figure kit (Blender side): silhouette people carved by firelight.

A figure = rigid tapered-capsule body segments (keyed transforms: robust, motion-blur friendly, joints
hidden inside clothes) + parametric cloth meshes (hooded robe, hood, wide sleeves, streaming scarf,
hoodie) rebuilt per frame and stored as shape keys, so the wind animates them deterministically.

Figure-local space: origin between the feet on the ground, +Y forward (the way the figure faces),
+Z up, +X the figure's right. `root` = world matrix of that space (position + yaw).
Poses are dicts of a few angles (radians) interpolated with easing between keyed frames.
"""
import math

import bmesh
import bpy
from mathutils import Matrix, Quaternion, Vector, noise

from . import core as C
from .nodes import new_material


# ------------------------------------------------------------- materials ---

def cloth_material(name, base=(0.022, 0.021, 0.024), sheen=0.9, sheen_tint=(1.0, 0.9, 0.8), rough=0.85,
                   weave=True):
    m, nb = new_material(name)
    tc = nb.texco()
    col = base
    if weave:
        n = nb.noise(tc.outputs['Object'], scale=35.0, detail=3.0, rough=0.6)
        col = nb.colscale(base, nb.madd(n.outputs['Fac'], 0.5, 0.75))
    bs = nb.principled(Base_Color=col, Roughness=rough)
    if weave:
        # soft folds and creases so grazing firelight breaks up on the cloth (no smooth 'clay')
        fo = nb.noise(tc.outputs['Object'], scale=9.0, detail=4.0, rough=0.55, dist=1.2)
        nb.link(nb.bump(fo.outputs['Fac'], strength=0.4, distance=0.02), bs.inputs['Normal'])
    bs.inputs['Specular IOR Level'].default_value = 0.3
    bs.inputs['Sheen Weight'].default_value = sheen
    bs.inputs['Sheen Tint'].default_value = tuple(sheen_tint) + (1.0,)
    bs.inputs['Sheen Roughness'].default_value = 0.35
    nb.output(surface=C.fogged(nb, bs))
    m.use_backface_culling = False
    return m


# ------------------------------------------------------------ primitives ---

def capsule_data(name, r0, r1, L, rx=1.0, ry=1.0, n=18, m=8, cap=6):
    """Tapered capsule along +Z from 0 to L (radius r0 at 0, r1 at L), elliptical section (rx, ry)."""
    V, F = [], []
    rings = []
    for i in range(cap, 0, -1):                      # bottom cap
        a = (math.pi / 2) * i / cap
        rings.append((-r0 * math.sin(a), r0 * math.cos(a)))
    for j in range(m + 1):
        v = j / m
        rings.append((v * L, r0 + (r1 - r0) * v))
    for i in range(1, cap + 1):                      # top cap
        a = (math.pi / 2) * i / cap
        rings.append((L + r1 * math.sin(a), r1 * math.cos(a)))
    for z, r in rings:
        for k in range(n):
            a = 2 * math.pi * k / n
            V.append((math.cos(a) * r * rx, math.sin(a) * r * ry, z))
    nr = len(rings)
    for j in range(nr - 1):
        for k in range(n):
            k2 = (k + 1) % n
            F.append((j * n + k, j * n + k2, (j + 1) * n + k2, (j + 1) * n + k))
    me = bpy.data.meshes.new(name)
    me.from_pydata(V, [], F)
    me.update()
    me.shade_smooth()
    return me


def ellipsoid_data(name, rx, ry, rz, n=20, m=14):
    V, F = [], []
    for j in range(m + 1):
        th = math.pi * j / m
        for k in range(n):
            ph = 2 * math.pi * k / n
            V.append((rx * math.sin(th) * math.cos(ph), ry * math.sin(th) * math.sin(ph), rz * math.cos(th)))
    for j in range(m):
        for k in range(n):
            k2 = (k + 1) % n
            F.append((j * n + k, j * n + k2, (j + 1) * n + k2, (j + 1) * n + k))
    me = bpy.data.meshes.new(name)
    me.from_pydata(V, [], F)
    me.validate()
    me.update()
    me.shade_smooth()
    return me


def loft_data(name, L, prof, n=22, ext0=0.0, ext1=0.0, cap=3):
    """Lofted limb/torso along +Z: prof = [(t, rx, ry, cx, cy)], t in [0,1] over [-ext0, L+ext1].
    Rounded caps (shrinking rings) at both ends. Section axes: local X, Y (see seg_quat)."""
    Lt = L + ext0 + ext1
    rings = [(-ext0 + t * Lt, rx, ry, cx, cy) for (t, rx, ry, cx, cy) in prof]
    z0, rx0, ry0, cx0, cy0 = rings[0]
    z1, rx1, ry1, cx1, cy1 = rings[-1]
    pre = []
    for i in range(cap, 0, -1):
        a = (math.pi / 2) * i / (cap + 1)
        k = math.cos(a)
        pre.append((z0 - math.sin(a) * min(rx0, ry0) * 0.6, rx0 * k, ry0 * k, cx0, cy0))
    post = []
    for i in range(1, cap + 1):
        a = (math.pi / 2) * i / (cap + 1)
        k = math.cos(a)
        post.append((z1 + math.sin(a) * min(rx1, ry1) * 0.6, rx1 * k, ry1 * k, cx1, cy1))
    rings = pre + rings + post
    V, F = [], []
    for (z, rx, ry, cx, cy) in rings:
        for k in range(n):
            a = 2 * math.pi * k / n
            V.append((cx + math.cos(a) * rx, cy + math.sin(a) * ry, z))
    nr = len(rings)
    for j in range(nr - 1):
        for k in range(n):
            k2 = (k + 1) % n
            F.append((j * n + k, j * n + k2, (j + 1) * n + k2, (j + 1) * n + k))
    # poles
    zb = rings[0][0] - 0.004
    ze = rings[-1][0] + 0.004
    V.append((rings[0][3], rings[0][4], zb))
    V.append((rings[-1][3], rings[-1][4], ze))
    ib, ie = len(V) - 2, len(V) - 1
    for k in range(n):
        k2 = (k + 1) % n
        F.append((k2, k, ib))
        F.append(((nr - 1) * n + k, (nr - 1) * n + k2, ie))
    me = bpy.data.meshes.new(name)
    me.from_pydata(V, [], F)
    me.validate()
    me.update()
    me.shade_smooth()
    return me


def seg_quat(d, ref):
    """Orientation with local Z along d and local X = ref x Z (so local Y ~ ref)."""
    z = Vector(d).normalized()
    r = Vector(ref)
    x = r.cross(z)
    if x.length < 1e-4:
        x = Vector((0, 0, 1)).cross(z) if abs(z.z) < 0.9 else Vector((0, 1, 0)).cross(z)
    x.normalize()
    y = z.cross(x)
    M = Matrix((x, y, z)).transposed()
    return M.to_quaternion()


# ------------------------------------------------------------------- pose ---

def ease(u):
    u = min(max(u, 0.0), 1.0)
    return u * u * u * (u * (u * 6 - 15) + 10)


def interp_pose(frame, keys):
    """keys: [(frame, {param: value})]; missing params carry over. Smootherstep between keys."""
    names = set()
    for _, d in keys:
        names.update(d.keys())
    out = {}
    for nm in names:
        kf = [(f, d[nm]) for f, d in keys if nm in d]
        if frame <= kf[0][0]:
            out[nm] = kf[0][1]
            continue
        if frame >= kf[-1][0]:
            out[nm] = kf[-1][1]
            continue
        for (f0, v0), (f1, v1) in zip(kf[:-1], kf[1:]):
            if f0 <= frame <= f1:
                u = ease((frame - f0) / (f1 - f0)) if f1 > f0 else 1.0
                out[nm] = v0 + (v1 - v0) * u
                break
    return out


def rotx(a):
    return Matrix.Rotation(a, 3, 'X')


def roty(a):
    return Matrix.Rotation(a, 3, 'Y')


def rotz(a):
    return Matrix.Rotation(a, 3, 'Z')


DEFAULT = dict(
    lean=0.0, side_lean=0.0, twist=0.0, head_yaw=0.0, head_pitch=0.0, crouch=0.0,
    # arms: shoulder swing forward (flex), out to the side (abd), elbow bend, forearm roll
    r_flex=0.1, r_abd=0.12, r_elbow=0.25, l_flex=0.05, l_abd=0.1, l_elbow=0.2,
    # legs: hip flex, knee bend; stance width
    r_hip=0.0, r_knee=0.05, l_hip=0.0, l_knee=0.05, stance=0.11, step=0.0,
    breath=0.0, shoulder_up=0.0, torch_pitch=0.0,
)


def fk(p, dims=None):
    """Joint positions (figure-local) + a few frames from pose params p."""
    d = dict(DEFAULT)
    d.update(p)
    p = d
    D = dict(hip_h=0.93, spine=0.22, chest=0.2, neck=0.17, head=0.12, sh_w=0.185, upper=0.29, fore=0.27,
             hand=0.09, thigh=0.44, shin=0.43, foot=0.2)
    if dims:
        D.update(dims)
    J = {}
    crouch = p['crouch']
    pelvis = Vector((0.0, 0.02 * crouch, D['hip_h'] * (1 - 0.14 * crouch)))
    Rtor = rotz(-p['twist'] * 0.5) @ rotx(-p['lean']) @ roty(p['side_lean'])
    J['pelvis'] = pelvis
    J['spine'] = pelvis + Rtor @ Vector((0, 0, D['spine']))
    Rch = rotz(-p['twist'] * 0.5) @ Rtor @ rotx(-0.3 * p['lean'])
    J['chest'] = J['spine'] + Rch @ Vector((0, 0.01 + 0.004 * p['breath'], D['chest']))
    J['neck'] = J['chest'] + Rch @ Vector((0, 0.015, D['neck'] * 0.55))
    Rh = Rch @ rotz(-p['head_yaw']) @ rotx(p['head_pitch'])
    J['head'] = J['neck'] + Rh @ Vector((0, 0.02, D['head']))
    J['R_head'] = Rh
    J['R_chest'] = Rch
    J['R_pelvis'] = Rtor
    for side, sg in (('r', 1.0), ('l', -1.0)):
        sh = J['chest'] + Rch @ Vector((sg * D['sh_w'], -0.01, 0.02 + 0.03 * p['shoulder_up'] * (sg > 0)))
        J[side + '_sh'] = sh
        flex, abd, elb = p[side + '_flex'], p[side + '_abd'], p[side + '_elbow']
        # upper arm hangs down (-Z), swings forward by flex (about X), out by abd (about Y)
        Ru = Rch @ rotx(flex) @ roty(sg * abd)
        el = sh + Ru @ Vector((0, 0, -D['upper']))
        Rf = Ru @ rotx(elb)
        wr = el + Rf @ Vector((0, 0, -D['fore']))
        hd = wr + Rf @ Vector((0, 0, -D['hand']))
        J[side + '_el'], J[side + '_wr'], J[side + '_hand'] = el, wr, hd
        J['R_' + side + '_fore'] = Rf
        hp = pelvis + Rtor @ Vector((sg * 0.1, 0, -0.03))
        J[side + '_hip'] = hp
        hflex = p[side + '_hip'] + (0.35 * crouch)
        kn_b = p[side + '_knee'] + 0.7 * crouch
        st = p['stance'] * sg
        Rt = rotx(hflex) @ roty(-sg * 0.02) if True else None
        base_off = Vector((st - sg * 0.1, 0.0, 0.0))
        kn = hp + base_off * 0.5 + Rt @ Vector((0, 0, -D['thigh']))
        Rs = Rt @ rotx(-kn_b)
        an = kn + base_off * 0.5 + Rs @ Vector((0, 0, -D['shin']))
        # keep feet on the ground (stance: feet flat)
        an.z = max(an.z, 0.075)
        J[side + '_knee'], J[side + '_ankle'] = kn, an
        J[side + '_toe'] = an + Vector((0.02 * sg, D['foot'], -0.045))
    return J


def ik_arm(J, side, hand_target, pole=(0.35, -0.3, -1.0), L1=0.29, L2=0.27, Lh=0.09):
    """Two-bone IK: place shoulder->elbow->wrist->hand so the hand reaches `hand_target` (the hand
    continues along the forearm). Elbow bends toward `pole` (figure-local). Updates J in place."""
    S = J[side + '_sh']
    tgt = Vector(hand_target)
    d = tgt - S
    fdir = d.normalized()
    wt = tgt - fdir * Lh
    dw = wt - S
    dist = min(max(dw.length, 0.05), L1 + L2 - 1e-3)
    dn = dw.normalized()
    ca = (L1 * L1 + dist * dist - L2 * L2) / (2 * L1 * dist)
    ca = min(max(ca, -1.0), 1.0)
    sa = math.sqrt(max(0.0, 1 - ca * ca))
    pv = Vector(pole)
    pv = (pv - dn * pv.dot(dn))
    if pv.length < 1e-6:
        pv = Vector((0, 0, -1)) - dn * dn.z
    pv.normalize()
    el = S + dn * (L1 * ca) + pv * (L1 * sa)
    wr = S + dn * dist
    hd = wr + (wr - el).normalized() * Lh
    J[side + '_el'], J[side + '_wr'], J[side + '_hand'] = el, wr, hd
    fd = (wr - el).normalized()
    J['R_' + side + '_fore'] = fd.to_track_quat('-Z', 'Y').to_matrix()
    return J


# ---------------------------------------------------------------- figure ---

class Figure:
    """Rigid body segments with keyed transforms. `parts` maps a segment name to
    (joint_a, joint_b, r0, r1, rx, ry) for capsules, or ('ELL', joint, rx, ry, rz, offset) for
    ellipsoids (oriented by the chest/head frame)."""

    def __init__(self, name, parts, mat, root, J0, mat_skin=None):
        self.name = name
        self.parts = parts
        self.root = root
        self.objs = {}
        for pn, spec in parts.items():
            if spec[0] == 'ELL':
                me = ellipsoid_data(f'{name}_{pn}', spec[2], spec[3], spec[4])
            elif spec[0] == 'LOFT':
                _, a, b, prof, ext0, ext1, ref = spec
                me = loft_data(f'{name}_{pn}', max((J0[b] - J0[a]).length, 1e-3), prof, ext0=ext0, ext1=ext1)
            else:
                a, b, r0, r1, rx, ry = spec
                A = J0[a] if isinstance(a, str) else Vector(a)
                B = J0[b] if isinstance(b, str) else Vector(b)
                me = capsule_data(f'{name}_{pn}', r0, r1, max((B - A).length, 1e-3), rx, ry)
            ob = bpy.data.objects.new(f'{name}_{pn}', me)
            C.link_obj(ob)
            ob.rotation_mode = 'QUATERNION'
            m = mat_skin if (mat_skin is not None and pn in ('head', 'r_hand', 'l_hand')) else mat
            me.materials.append(m)
            self.objs[pn] = ob

    def key(self, frame, J):
        W = self.root
        Rw = W.to_3x3()
        for pn, spec in self.parts.items():
            ob = self.objs[pn]
            if spec[0] == 'LOFT':
                _, a, b, prof, ext0, ext1, ref = spec
                A, B = J[a], J[b]
                if isinstance(ref, str):
                    rv = J[ref] @ Vector((0, 1, 0))
                else:
                    rv = J.get('R_pelvis', Matrix.Identity(3)) @ Vector(ref)
                q = seg_quat(B - A, rv)
                ob.location = W @ A
                ob.rotation_quaternion = Rw.to_quaternion() @ q
                ob.scale = (1, 1, 1)
            elif spec[0] == 'ELL':
                _, jn, rx, ry, rz, off, frame_name = spec
                R = J.get(frame_name, Matrix.Identity(3))
                pos = W @ (J[jn] + R @ Vector(off))
                q = (Rw @ R).to_quaternion()
                ob.location = pos
                ob.rotation_quaternion = q
                ob.scale = (1, 1, 1)
            else:
                a, b, r0, r1, rx, ry = spec
                A = J[a] if isinstance(a, str) else Vector(a)
                B = J[b] if isinstance(b, str) else Vector(b)
                d = B - A
                q = d.normalized().to_track_quat('Z', 'Y')
                ob.location = W @ A
                ob.rotation_quaternion = (Rw.to_quaternion() @ q)
                ob.scale = (1, 1, 1)
            ob.keyframe_insert('location', frame=frame)
            ob.keyframe_insert('rotation_quaternion', frame=frame)
            ob.keyframe_insert('scale', frame=frame)


# ----------------------------------------------------------------- cloth ---

class ClothSeq:
    """A mesh with fixed topology whose vertices are regenerated per frame (shape key per frame)."""

    def __init__(self, name, faces, mat, nverts):
        me = bpy.data.meshes.new(name)
        me.from_pydata([(0.0, 0.0, 0.0)] * nverts, [], faces)
        me.update()
        me.shade_smooth()
        me.materials.append(mat)
        self.ob = C.link_obj(bpy.data.objects.new(name, me))
        self.ob.shape_key_add(name='Basis', from_mix=False)
        self.frames = []
        self.n = nverts

    def add(self, frame, verts):
        kb = self.ob.shape_key_add(name=f'f{frame}', from_mix=False)
        kb.data.foreach_set('co', [c for v in verts for c in v])
        self.frames.append((frame, kb))
        if len(self.frames) == 1:
            # the basis = first frame (so the bounding box / undeformed mesh is sensible)
            self.ob.data.shape_keys.key_blocks['Basis'].data.foreach_set('co', [c for v in verts for c in v])
            self.ob.data.vertices.foreach_set('co', [c for v in verts for c in v])

    def finalize(self):
        fr = [f for f, _ in self.frames]
        for i, (f, kb) in enumerate(self.frames):
            pts = {f: 1.0}
            if i > 0:
                pts[fr[i - 1]] = 0.0
            if i < len(fr) - 1:
                pts[fr[i + 1]] = 0.0
            for ff, v in sorted(pts.items()):
                kb.value = v
                kb.keyframe_insert('value', frame=ff)
        # (the rest of the timeline is irrelevant: the shot only renders its own frames)


def ring_faces(nu, nv, wrap=True):
    return C.grid_faces(nu, nv, wrap_u=wrap)


def flap(s, t, freq=1.6, k=2.4, phase=0.0):
    """Travelling flag wave along a free cloth edge (s = 0 at the attachment, 1 at the free end)."""
    return math.sin(2 * math.pi * (freq * t - k * s) + phase)


def robe_verts(J, t, wind_local, nu=44, nv=26, top_z=None, hem_z=0.03, flare=(0.31, 0.27), gust=1.0,
               seed=0.0, folds=11, chest_r=(0.215, 0.14), front_open=0.0):
    """Hooded robe body: rings from the shoulders to the hem around the body axis, with folds,
    a flared hem and wind (push + travelling ripples + noise). wind_local: figure-local (x, y) m/s-ish."""
    ch = J['chest']
    pel = J['pelvis']
    top_z = ch.z + 0.08 if top_z is None else top_z
    wx, wy = wind_local
    wl = math.hypot(wx, wy) + 1e-9
    wd = Vector((wx / wl, wy / wl, 0))
    V = []
    for j in range(nv):
        v = j / (nv - 1)                       # 0 top .. 1 hem
        z = top_z + (hem_z - top_z) * v
        # body axis x/y follows the chest near the top and the pelvis lower down
        cx = ch.x + (pel.x - ch.x) * min(v * 2.2, 1.0)
        cy = ch.y + (pel.y - ch.y) * min(v * 2.2, 1.0)
        rx = chest_r[0] + (flare[0] - chest_r[0]) * (v ** 1.25)
        ry = chest_r[1] + (flare[1] - chest_r[1]) * (v ** 1.1)
        for i in range(nu):
            a = 2 * math.pi * i / nu
            ca, sa = math.cos(a), math.sin(a)
            fold = (0.004 + 0.03 * v ** 1.3) * math.sin(folds * a + 1.3 * v + seed) \
                + 0.012 * v * noise.noise(Vector((ca * 2 + seed, sa * 2, v * 3 + 0.2 * t)))
            x = cx + ca * (rx + fold)
            y = cy + sa * (ry + fold)
            # wind: push downwind (more at the hem and on the leeward side), travelling ripples
            side = max(0.0, ca * wd.x + sa * wd.y)          # 1 on the downwind side
            push = gust * wl * (0.045 * v ** 1.6) * (0.6 + 0.8 * side)
            rip = gust * wl * 0.022 * v ** 1.4 * flap(v, t, 1.7, 1.6, a * 2 + seed)
            nn = gust * wl * 0.02 * v * noise.noise(Vector((x * 2.5 - wx * t * 0.9, y * 2.5 - wy * t * 0.9, z * 2 + seed)))
            x += wd.x * (push + nn) + ca * rip
            y += wd.y * (push + nn) + sa * rip
            zz = z + (0.03 * v * v * side * gust * wl * 0.3)
            V.append((x, y, zz))
    return V, ring_faces(nu, nv)


def hood_verts(J, t, wind_local, nu=30, nv=16, r=0.155, depth=0.19, drape=0.24, peak=0.05, gust=1.0,
               seed=1.0, open_ang=0.95):
    """Deep hood around the head: a shell open at the face (front, +Y of the head frame), draping onto
    the shoulders at the back, a soft peak at the crown; the rim flutters."""
    Rh = J['R_head']
    H = J['head']
    wx, wy = wind_local
    wl = math.hypot(wx, wy) + 1e-9
    V = []
    for j in range(nv):
        v = j / (nv - 1)                      # 0 = face rim ... 1 = back/bottom drape
        for i in range(nu):
            u = i / (nu - 1)                   # around the face opening: 0 = left chin, 1 = right chin
            # angle around the head from the left side over the crown to the right side
            ph = -math.pi / 2 + math.pi * u
            # from the face rim (front) backwards
            th = open_ang + (math.pi - open_ang) * v
            # base sphere direction in head frame: forward = +Y
            dx = math.sin(ph) * math.sin(th)
            dz = math.cos(ph) * math.sin(th)
            dy = math.cos(th)
            rr = r * (1.0 + 0.1 * v)
            p = Vector((dx * rr, dy * rr - 0.02, dz * rr + 0.01))
            # peak at the crown-back
            crown = math.exp(-((ph) / 0.6) ** 2) * math.exp(-((th - 1.9) / 0.5) ** 2)
            p += Vector((0, -0.4, 0.6)).normalized() * peak * crown
            # drape: the lower part of the back falls onto the shoulders
            low = max(0.0, -math.cos(ph) * 0.0 + (1 - abs(math.sin(ph)) * 0.0))
            fall = max(0.0, (abs(ph) - 0.9) / (math.pi / 2 - 0.9)) * v
            p += Vector((math.sin(ph) * 0.05 * fall, -0.05 * v, -drape * fall))
            # rim flutter
            fl = gust * wl * 0.012 * (1 - v) * flap(u, t, 2.2, 1.5, seed) + gust * wl * 0.006 * noise.noise(
                Vector((u * 3 + t * 1.3, v * 2, seed)))
            p += Vector((0, fl, fl * 0.4))
            V.append(tuple(H + Rh @ p))
    return V, C.grid_faces(nu, nv, wrap_u=False)


def sleeve_verts(J, side, t, wind_local, nu=16, nv=12, r0=0.075, r1=0.17, droop=0.14, gust=1.0, seed=2.0):
    """Wide robe sleeve from shoulder to wrist; its lower edge droops under gravity and flaps."""
    sh, el, wr = J[side + '_sh'], J[side + '_el'], J[side + '_wr']
    wx, wy = wind_local
    wl = math.hypot(wx, wy) + 1e-9
    wd = Vector((wx / wl, wy / wl, 0))
    V = []
    for j in range(nv):
        v = j / (nv - 1)
        if v < 0.5:
            c = sh.lerp(el, v * 2)
        else:
            c = el.lerp(wr, (v - 0.5) * 2 + 0.05)
        ax = ((el - sh) if v < 0.5 else (wr - el)).normalized()
        # a frame around the arm axis
        ref = Vector((0, 0, 1)) if abs(ax.z) < 0.95 else Vector((0, 1, 0))
        n1 = ax.cross(ref).normalized()
        n2 = ax.cross(n1).normalized()
        rr = r0 + (r1 - r0) * v ** 1.4
        for i in range(nu):
            a = 2 * math.pi * i / nu
            off = n1 * math.cos(a) * rr + n2 * math.sin(a) * rr
            # gravity droop: points below the arm hang lower, more toward the wrist
            below = max(0.0, -off.normalized().z)
            off += Vector((0, 0, -droop * v ** 1.5 * below))
            fl = gust * wl * v * (0.02 * flap(v, t, 1.9, 1.3, a + seed) + 0.015 * noise.noise(Vector((a, v * 3 + t, seed))))
            off += wd * (fl + gust * wl * 0.03 * v * below)
            V.append(tuple(c + off))
    return V, ring_faces(nu, nv)


def ribbon_verts(anchor, t, wind_local, L=0.9, w0=0.12, w1=0.05, n=18, sag=0.25, amp=0.09, freq=2.0,
                 k=1.6, gust=1.0, seed=3.0, up=(0, 0, 1), side_dir=None):
    """A scarf / cloth tail streaming downwind from `anchor` (figure-local)."""
    wx, wy = wind_local
    wl = math.hypot(wx, wy) + 1e-9
    wd = Vector((wx / wl, wy / wl, 0.0))
    side = Vector(side_dir) if side_dir else wd.cross(Vector(up)).normalized()
    V = []
    for i in range(n):
        s = i / (n - 1)
        lift = min(1.0, wl / 5.0)
        p = Vector(anchor) + wd * (L * s * (0.35 + 0.65 * lift)) + Vector((0, 0, -sag * s * s * (1.2 - lift)))
        wave = amp * s * flap(s, t, freq, k, seed) * gust
        tw = 0.6 * flap(s, t, freq * 0.8, k * 0.7, seed + 1.0)
        p += Vector((0, 0, wave)) + side * (0.3 * wave)
        nn = noise.noise(Vector((s * 2 - t * 1.2, seed, t * 0.5))) * 0.04 * s
        p += Vector((0, 0, nn))
        w = w0 + (w1 - w0) * s
        dvec = (Vector((0, 0, 1)) * math.cos(tw) + side * math.sin(tw)) * (w / 2)
        V.append(tuple(p - dvec))
        V.append(tuple(p + dvec))
    F = [(2 * i, 2 * i + 1, 2 * i + 3, 2 * i + 2) for i in range(n - 1)]
    return V, F


class Cloths:
    """Collect per-frame cloth pieces; each piece keeps its topology across frames."""

    def __init__(self, name, mat):
        self.name = name
        self.mat = mat
        self.seqs = {}

    def add(self, frame, piece, VF, root):
        V, F = VF
        if piece not in self.seqs:
            self.seqs[piece] = ClothSeq(f'{self.name}_{piece}', F, self.mat, len(V))
        W = root
        self.seqs[piece].add(frame, [tuple(W @ Vector(v)) for v in V])

    def finalize(self):
        for s in self.seqs.values():
            s.finalize()


# ------------------------------------------------------------------ torch ---

def torch_obj(name, length=0.62, r=0.017, head=0.1, head_r=0.032):
    """Wooden torch with a rag head; origin at the grip, pointing +Z."""
    from .fire import tube, merge
    parts = [tube([(0, 0, -0.12), (0, 0, length - head)], r, 6)]
    parts.append(tube([(0, 0, length - head), (0, 0, length - head * 0.5), (0, 0, length)], head_r, 8,
                      radii=[head_r * 0.9, head_r * 1.1, head_r * 0.8]))
    V, F = merge(parts)
    m, nb = new_material(name + '_m')
    bs = nb.principled(Base_Color=(0.018, 0.013, 0.01), Roughness=0.95)
    bs.inputs['Specular IOR Level'].default_value = 0.2
    x, y, z = nb.sep(nb.texco().outputs['Object'])
    hot = nb.sstep(length - head - 0.02, length - head * 0.3, z)
    em = nb.emission((1.0, 0.35, 0.08), nb.mul(hot, 2.0))
    nb.output(surface=C.fogged(nb, nb.addshader(bs, em)))
    ob = C.mesh_obj(name, V, F, mat=m)
    ob.rotation_mode = 'QUATERNION'
    return ob


# ------------------------------------------------------ SDF figure meshes ---

def load_bin(path):
    """Mesh written by figures.write_mesh: (verts [(x,y,z)], quads [(a,b,c,d)], mat [float])."""
    import array
    import struct
    with open(path, 'rb') as f:
        nv, nq = struct.unpack('<ii', f.read(8))
        Va = array.array('f')
        Va.frombytes(f.read(nv * 12))
        Qa = array.array('i')
        Qa.frombytes(f.read(nq * 16))
        Ma = array.array('f')
        Ma.frombytes(f.read(nv * 4))
    verts = [(Va[i], Va[i + 1], Va[i + 2]) for i in range(0, 3 * nv, 3)]
    quads = [(Qa[i], Qa[i + 1], Qa[i + 2], Qa[i + 3]) if Qa[i + 3] >= 0 else (Qa[i], Qa[i + 1], Qa[i + 2])
             for i in range(0, 4 * nq, 4)]
    return verts, quads, Ma


def mesh_from_bin(name, path, mat):
    verts, quads, Ma = load_bin(path)
    me = bpy.data.meshes.new(name)
    me.from_pydata(verts, [], quads)
    me.update()
    me.shade_smooth()
    a = me.attributes.new('mat', 'FLOAT', 'POINT')
    a.data.foreach_set('value', Ma)
    me.materials.append(mat)
    return me


def figure_material(name, zones, sheen=0.35, sheen_tint=(1.0, 0.9, 0.8), rough=0.9, folds=0.35):
    """One material for an SDF figure; zones = [albedo per 'mat' id 0..3] blended by the attribute."""
    m, nb = new_material(name)
    a = nb.attr('mat').outputs['Fac']
    col = zones[0]
    for i in range(1, len(zones)):
        col = nb.mixcol(nb.sstep(i - 0.5, i - 0.45, a), col, zones[i])
    tc = nb.texco()
    n = nb.noise(tc.outputs['Object'], scale=35.0, detail=3.0, rough=0.6)
    col = nb.colscale(col, nb.madd(n.outputs['Fac'], 0.4, 0.8))
    bs = nb.principled(Base_Color=col, Roughness=rough)
    bs.inputs['Specular IOR Level'].default_value = 0.3
    bs.inputs['Sheen Weight'].default_value = sheen
    bs.inputs['Sheen Tint'].default_value = tuple(sheen_tint) + (1.0,)
    bs.inputs['Sheen Roughness'].default_value = 0.35
    if folds:
        fo = nb.noise(tc.outputs['Object'], scale=16.0, detail=4.0, rough=0.55, dist=0.8)
        nb.link(nb.bump(fo.outputs['Fac'], strength=folds, distance=0.01), bs.inputs['Normal'])
    nb.output(surface=C.fogged(nb, bs))
    return m


class MeshSeq:
    """An object whose mesh is swapped per rendered frame (files <dir>/<name>_<frame>.bin)."""
    registry = []

    def __init__(self, name, mesh_dir, mat, root, first_frame):
        self.name, self.dir, self.mat = name, mesh_dir, mat
        me = mesh_from_bin(f'{name}_{first_frame}', self.path(first_frame), mat)
        self.ob = C.link_obj(bpy.data.objects.new(name, me))
        self.ob.matrix_world = root
        MeshSeq.registry.append(self)

    def path(self, f):
        import os
        return os.path.join(self.dir, f'{self.name}_{f:05d}.bin')

    def set_frame(self, f):
        old = self.ob.data
        self.ob.data = mesh_from_bin(f'{self.name}_{f}', self.path(f), self.mat)
        if old.users == 0:
            bpy.data.meshes.remove(old)


def swap_all(frame):
    for s in MeshSeq.registry:
        s.set_frame(frame)
