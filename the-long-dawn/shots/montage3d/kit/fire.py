"""MONTAGE-3D fire kit (Blender side): the beacon (dry-stone cairn + iron fire-basket + stacked split
wood), the city drum, the flame card (HDR sprite sequence from the venv, camera-facing, additive),
flickering fire lights, a rising smoke plume and a local glow haze (volumes lit only by the fire).

Timing curves (ignition envelope, flicker) come from the venv (the montage toolkit) and are passed in
as per-frame tables so every beacon in the film shares them exactly.
"""
import math
import random

import bmesh
import bpy
from mathutils import Matrix, Vector, noise

from . import core as C
from .nodes import NB, new_material


# ------------------------------------------------------------- mesh helpers ---

def _bm_to_lists(bm, M=None, base=0, vlist=None, flist=None):
    vlist = [] if vlist is None else vlist
    flist = [] if flist is None else flist
    idx = {}
    for v in bm.verts:
        idx[v.index] = len(vlist)
        p = v.co if M is None else M @ v.co
        vlist.append((p.x, p.y, p.z))
    for f in bm.faces:
        flist.append(tuple(idx[v.index] for v in f.verts))
    return vlist, flist


def tube(points, radius, sides=6, cap=True, radii=None):
    """Tube mesh along a polyline (parallel-transport frames). Returns (verts, faces)."""
    pts = [Vector(p) for p in points]
    n = len(pts)
    tang = []
    for i in range(n):
        a = pts[max(i - 1, 0)]
        b = pts[min(i + 1, n - 1)]
        tang.append((b - a).normalized())
    ref = Vector((0, 0, 1)) if abs(tang[0].z) < 0.9 else Vector((1, 0, 0))
    nrm = tang[0].cross(ref).normalized()
    verts, faces = [], []
    for i in range(n):
        if i > 0:
            # parallel transport
            b = tang[i - 1].cross(tang[i])
            if b.length > 1e-6:
                ang = tang[i - 1].angle(tang[i])
                nrm = Matrix.Rotation(ang, 3, b.normalized()) @ nrm
        bnm = tang[i].cross(nrm).normalized()
        r = radius if radii is None else radii[i]
        for k in range(sides):
            a = 2 * math.pi * k / sides
            p = pts[i] + (nrm * math.cos(a) + bnm * math.sin(a)) * r
            verts.append((p.x, p.y, p.z))
    for i in range(n - 1):
        for k in range(sides):
            k2 = (k + 1) % sides
            faces.append((i * sides + k, i * sides + k2, (i + 1) * sides + k2, (i + 1) * sides + k))
    if cap:
        faces.append(tuple(range(sides))[::-1])
        faces.append(tuple((n - 1) * sides + k for k in range(sides)))
    return verts, faces


def merge(parts):
    V, F = [], []
    for v, f in parts:
        o = len(V)
        V.extend(v)
        F.extend(tuple(i + o for i in fc) for fc in f)
    return V, F


def stone_mesh(rng, w, d, h, seed, sub=2, chip=0.16, round_=0.45):
    """Flattened, chipped dry-stone slab centred at the origin (x=length, y=depth, z=height)."""
    bm = bmesh.new()
    bmesh.ops.create_cube(bm, size=2.0)
    bmesh.ops.subdivide_edges(bm, edges=bm.edges[:], cuts=sub, use_grid_fill=True)
    bm.verts.index_update()
    off = Vector((seed * 3.17, seed * 1.71, seed * 7.31))
    m = min(w, d, h)
    for v in bm.verts:
        c = v.co.copy()
        nrm = c.normalized()
        p = c.lerp(nrm * 1.25, round_)
        p = Vector((p.x * w / 2, p.y * d / 2, p.z * h / 2))
        q = c * 1.3 + off
        disp = noise.noise(q) * chip * m + noise.noise(q * 2.9 + off) * chip * 0.4 * m
        # flat-ish beds: displace less vertically on the top/bottom faces
        dn = Vector((nrm.x, nrm.y, nrm.z * 0.35))
        p += dn * disp
        v.co = p
    return bm


# -------------------------------------------------------------- materials ---

def stone_material(name='stone', base=(0.16, 0.15, 0.14), var=0.5, rough=0.92):
    m, nb = new_material(name)
    tc = nb.texco()
    a = nb.attr('srand')
    n1 = nb.noise(tc.outputs['Object'], scale=9.0, detail=6.0, rough=0.6)
    n2 = nb.noise(tc.outputs['Object'], scale=45.0, detail=4.0, rough=0.7)
    tone = nb.madd(a.outputs['Fac'], var, 1.0 - var * 0.5)
    tone = nb.mul(tone, nb.madd(n1.outputs['Fac'], 0.5, 0.75))
    col = nb.mixcol(a.outputs['Fac'], base, (base[0] * 1.15, base[1] * 1.08, base[2] * 0.95))
    col = nb.colscale(col, tone)
    bump = nb.bump(nb.add(n2.outputs['Fac'], nb.mul(n1.outputs['Fac'], 0.5)), strength=0.35, distance=0.01)
    bs = nb.principled(Base_Color=col, Roughness=rough)
    bs.inputs['Specular IOR Level'].default_value = 0.35
    nb.link(bump, bs.inputs['Normal'])
    nb.output(surface=C.fogged(nb, bs))
    return m


def iron_material(name='iron', glow_key=None):
    m, nb = new_material(name)
    tc = nb.texco()
    n1 = nb.noise(tc.outputs['Object'], scale=30.0, detail=5.0, rough=0.6)
    col = nb.mixcol(n1.outputs['Fac'], (0.018, 0.016, 0.015), (0.05, 0.03, 0.02))
    bs = nb.principled(Base_Color=col, Roughness=nb.madd(n1.outputs['Fac'], 0.3, 0.45), Metallic=0.75)
    nb.output(surface=C.fogged(nb, bs))
    return m


def wood_material(name='wood', ember_frames=None):
    """Split firewood; after ignition, glowing ember cracks (emission keyed by `ember_frames`)."""
    m, nb = new_material(name)
    tc = nb.texco()
    obj = tc.outputs['Object']
    grain = nb.wave(obj, scale=6.0, dist=6.0, detail=3.0, direction='X')
    n1 = nb.noise(obj, scale=8.0, detail=5.0, rough=0.6)
    col = nb.mixcol(nb.mul(grain.outputs['Fac'], n1.outputs['Fac']), (0.10, 0.065, 0.04), (0.22, 0.15, 0.09))
    bs = nb.principled(Base_Color=col, Roughness=0.85)
    # ember cracks: thin bright veins where a voronoi edge meets high noise
    vor = nb.voronoi(obj, scale=22.0, feature='DISTANCE_TO_EDGE')
    crack = nb.sstep(0.05, 0.0, vor.outputs['Distance'])
    heat = nb.sstep(0.35, 0.75, nb.noise(obj, scale=4.0, detail=3.0).outputs['Fac'])
    glow = nb.mul(nb.add(nb.mul(crack, 1.0), nb.mul(heat, 0.25)), heat)
    ev = nb.value(0.0, 'ember')
    if ember_frames:
        for f, v in ember_frames:
            C.key_socket(ev.outputs[0], f, v)
    em = nb.emission((1.0, 0.33, 0.07), nb.mul(glow, ev))
    add = nb.addshader(bs, em)
    nb.output(surface=C.fogged(nb, add))
    return m


# ------------------------------------------------------------------ beacon ---

def beacon(name, base, seed=1, height=1.0, r_base=0.55, r_top=0.40, basket_h=0.42, basket_r=(0.2, 0.46),
           bars=9, logs=True, ember_frames=None, stone_base=(0.16, 0.15, 0.14), yaw=0.0):
    """Waist-high dry-stone cairn topped with an iron fire-basket of stacked split wood.
    Returns dict(objs, fire_base (world Vector), rb (flame base half-width))."""
    rng = random.Random(seed)
    base = Vector(base)
    R = Matrix.Rotation(yaw, 4, 'Z')
    T = Matrix.Translation(base) @ R
    V, F, SR = [], [], []
    z = 0.0
    k = 0
    while z < height - 0.04:
        u = z / height
        r = r_base + (r_top - r_base) * u
        ch = min(0.07 + 0.08 * rng.random(), height - z)
        circ = 2 * math.pi * r
        ang = rng.random() * 2 * math.pi
        acc = 0.0
        while acc < circ - 0.05:
            L = min(0.16 + 0.26 * rng.random() ** 1.3, circ - acc)
            dep = 0.18 + 0.2 * rng.random()
            a = ang + (acc + L / 2) / r
            hh = ch * (0.82 + 0.3 * rng.random())
            sd = rng.random() * 100
            bm = stone_mesh(rng, L * 0.97, dep, hh * 0.96, sd)
            M = (T @ Matrix.Translation(Vector((math.cos(a) * (r - dep / 2 + 0.01), math.sin(a) * (r - dep / 2 + 0.01),
                                                 z + hh / 2 + (rng.random() - 0.5) * 0.012)))
                 @ Matrix.Rotation(a + math.pi / 2 + (rng.random() - 0.5) * 0.12, 4, 'Z')
                 @ Matrix.Rotation((rng.random() - 0.5) * 0.1, 4, 'X')
                 @ Matrix.Rotation((rng.random() - 0.5) * 0.08, 4, 'Y'))
            n0 = len(V)
            _bm_to_lists(bm, M, vlist=V, flist=F)
            SR.extend([rng.random()] * (len(V) - n0))
            bm.free()
            acc += L + 0.006 + 0.01 * rng.random()
        z += ch
        k += 1
    top = z
    # capstones: two or three big flat slabs on top
    for i in range(3):
        a = rng.random() * 6.28
        bm = stone_mesh(rng, 0.42 + 0.1 * rng.random(), 0.3 + 0.1 * rng.random(), 0.06, rng.random() * 100)
        M = T @ Matrix.Translation(Vector((math.cos(a) * 0.12, math.sin(a) * 0.12, top + 0.02 + 0.01 * i))) @ \
            Matrix.Rotation(rng.random() * 6.28, 4, 'Z')
        n0 = len(V)
        _bm_to_lists(bm, M, vlist=V, flist=F)
        SR.extend([rng.random()] * (len(V) - n0))
        bm.free()
    top += 0.06
    # dark core filling the inside (only seen through chinks)
    cv, cf = tube([(0, 0, 0.02), (0, 0, top - 0.05)], 1.0, sides=16, radii=[r_base - 0.12, r_top - 0.12])
    cv = [tuple(T @ Vector(p)) for p in cv]
    stones = C.mesh_obj(name + '_stones', V, F, mat=stone_material(name + '_stone', stone_base))
    a = stones.data.attributes.new('srand', 'FLOAT', 'POINT')
    a.data.foreach_set('value', SR)
    core = C.mesh_obj(name + '_core', cv, cf, mat=C.surface_mat(name + '_corem', (0.01, 0.01, 0.01), 1.0))
    objs = [stones, core]
    # --- iron basket
    r0, r1 = basket_r
    yb0 = top - 0.02
    yb1 = top + basket_h
    parts = []
    for i in range(bars):
        a = 2 * math.pi * (i + 0.5) / bars
        ca, sa = math.cos(a), math.sin(a)
        pts = []
        for s in range(7):
            v = s / 6
            rr = r0 + (r1 - r0) * (v ** 0.8)
            pts.append((ca * rr, sa * rr, yb0 + v * basket_h))
        # a small outward curl at the top of each bar
        pts.append((ca * (r1 + 0.03), sa * (r1 + 0.03), yb1 + 0.035))
        parts.append(tube(pts, 0.011, 5, cap=True))
    for (rr, zz, th) in ((r1, yb1, 0.014), (r0, yb0 + 0.01, 0.014), (r0 + (r1 - r0) * 0.55, yb0 + basket_h * 0.5, 0.009)):
        ring = [(math.cos(2 * math.pi * s / 48) * rr, math.sin(2 * math.pi * s / 48) * rr, zz) for s in range(49)]
        parts.append(tube(ring, th, 5, cap=False))
    # legs into the cairn
    for i in range(3):
        a = 2 * math.pi * i / 3 + 0.3
        parts.append(tube([(math.cos(a) * r0, math.sin(a) * r0, yb0), (math.cos(a) * r0 * 0.8, math.sin(a) * r0 * 0.8, yb0 - 0.12)],
                          0.013, 5))
    bv, bf = merge(parts)
    bv = [tuple(T @ Vector(p)) for p in bv]
    basket = C.mesh_obj(name + '_basket', bv, bf, mat=iron_material(name + '_iron'))
    objs.append(basket)
    fire_base = T @ Vector((0, 0, yb0 + 0.08))
    if logs:
        parts = []
        layers = 4
        for Ly in range(layers):
            zz = yb0 + 0.05 + Ly * 0.075
            rr = r0 + (r1 - r0) * ((zz - yb0) / basket_h) ** 0.8
            a0 = (math.pi / 2) * (Ly % 2) + rng.random() * 0.4
            nl = 3
            for j in range(nl):
                off = (j - (nl - 1) / 2) * rr * 0.55
                L = 2 * rr * (0.95 + 0.25 * rng.random())
                d = Vector((math.cos(a0), math.sin(a0), 0))
                o = Vector((-math.sin(a0), math.cos(a0), 0)) * off
                p0 = o - d * L / 2 + Vector((0, 0, zz + (rng.random() - 0.5) * 0.02))
                p1 = o + d * L / 2 + Vector((0, 0, zz + (rng.random() - 0.5) * 0.03))
                rad = 0.03 + 0.017 * rng.random()
                tv, tf = tube([p0, p0.lerp(p1, 0.5), p1], rad, 7, cap=True)
                parts.append((tv, tf))
        lv, lf = merge(parts)
        lv = [tuple(T @ Vector(p)) for p in lv]
        wood = C.mesh_obj(name + '_wood', lv, lf, mat=wood_material(name + '_woodm', ember_frames))
        objs.append(wood)
    return dict(objs=objs, fire_base=fire_base, rb=r1 * 0.82, top=T @ Vector((0, 0, yb1)))


def drum(name, base, h=0.88, r=0.29, yaw=0.0, ember_frames=None, holes=True):
    """Steel 55-gallon drum, rusted, air holes punched near the bottom that glow once lit."""
    T = Matrix.Translation(Vector(base)) @ Matrix.Rotation(yaw, 4, 'Z')
    n = 48
    rings = [0.0, 0.02, 0.2, 0.215, 0.23, 0.45, 0.465, 0.48, 0.7, 0.715, 0.73, 0.98, 1.0]
    V, F = [], []
    for j, v in enumerate(rings):
        rib = 0.012 if v in (0.215, 0.465, 0.715) else 0.0
        rr = r + rib - (0.006 if v in (0.0, 1.0) else 0.0)
        for i in range(n):
            a = 2 * math.pi * i / n
            dent = 0.006 * noise.noise(Vector((math.cos(a) * 2, math.sin(a) * 2, v * 3)))
            V.append((math.cos(a) * (rr + dent), math.sin(a) * (rr + dent), v * h))
    for j in range(len(rings) - 1):
        for i in range(n):
            i2 = (i + 1) % n
            F.append((j * n + i, j * n + i2, (j + 1) * n + i2, (j + 1) * n + i))
    # rolled rim (inner wall to show thickness) + bottom
    o = len(V)
    for i in range(n):
        a = 2 * math.pi * i / n
        V.append((math.cos(a) * (r - 0.012), math.sin(a) * (r - 0.012), h * 0.995))
    for i in range(n):
        a = 2 * math.pi * i / n
        V.append((math.cos(a) * (r - 0.012), math.sin(a) * (r - 0.012), h * 0.55))
    top_j = len(rings) - 1
    for i in range(n):
        i2 = (i + 1) % n
        F.append((top_j * n + i, top_j * n + i2, o + i2, o + i))
        F.append((o + i, o + i2, o + n + i2, o + n + i))
    F.append(tuple(range(n))[::-1])
    V = [tuple(T @ Vector(p)) for p in V]
    m, nb = new_material(name + '_drumm')
    tc = nb.texco()
    obj = tc.outputs['Object']
    rust = nb.noise(obj, scale=6.0, detail=8.0, rough=0.65, dist=0.4)
    rust2 = nb.noise(obj, scale=40.0, detail=4.0, rough=0.6)
    rm = nb.sstep(0.42, 0.62, nb.add(rust.outputs['Fac'], nb.mul(nb.sub(rust2.outputs['Fac'], 0.5), 0.3)))
    col = nb.mixcol(rm, (0.03, 0.035, 0.04), (0.12, 0.05, 0.022))
    bs = nb.principled(Base_Color=col, Roughness=nb.madd(rm, 0.35, 0.5), Metallic=nb.sub(0.85, nb.mul(rm, 0.7)))
    shader = bs
    if holes:
        # punched air holes in a band near the bottom: emissive once lit (keyed)
        x, y, zz = nb.sep(obj)
        ang = nb.math('ARCTAN2', y, x)
        hu = nb.math('FRACT', nb.mul(ang, 12 / (2 * math.pi)))
        hv = nb.div(nb.sub(zz, 0.12 * h), 0.05)
        dd = nb.add(nb.pw(nb.mul(nb.sub(hu, 0.5), 2.2), 2.0), nb.pw(hv, 2.0))
        hole = nb.sstep(0.55, 0.3, dd)
        ev = nb.value(0.0, 'ember')
        if ember_frames:
            for f, val in ember_frames:
                C.key_socket(ev.outputs[0], f, val)
        em = nb.emission((1.0, 0.45, 0.12), nb.mul(nb.mul(hole, ev), 6.0))
        dark = nb.principled(Base_Color=(0.0, 0.0, 0.0), Roughness=1.0)
        shader = nb.addshader(nb.mixshader(hole, bs, dark), em)
    nb.output(surface=C.fogged(nb, shader))
    ob = C.mesh_obj(name, V, F, mat=m)
    # a few burning pallet slats poking out of the top
    parts = []
    rng = random.Random(7)
    for i in range(5):
        a = rng.random() * 6.28
        p0 = Vector((math.cos(a) * r * 0.2, math.sin(a) * r * 0.2, h * 0.6))
        p1 = p0 + Vector((math.cos(a + 0.6) * 0.18, math.sin(a + 0.6) * 0.18, 0.42 + 0.1 * rng.random()))
        tv, tf = tube([p0, p1], 0.02, 4, cap=True)
        parts.append((tv, tf))
    lv, lf = merge(parts)
    lv = [tuple(T @ Vector(p)) for p in lv]
    wood = C.mesh_obj(name + '_wood', lv, lf, mat=wood_material(name + '_woodm', ember_frames))
    return dict(objs=[ob, wood], fire_base=T @ Vector((0, 0, h + 0.03)), rb=r * 0.85, top=T @ Vector((0, 0, h)))


# ------------------------------------------------------------ flame card ---

def flame_card(name, base, card, seq_first_file, cam, gain=1.0, fog=True):
    """Camera-facing (vertical, locked-Z) additive card showing the flame sprite sequence.
    card = (x0, x1, z0, z1) metres relative to the flame base."""
    x0, x1, z0, z1 = card
    V = [(x0, 0, z0), (x1, 0, z0), (x1, 0, z1), (x0, 0, z1)]
    ob = C.mesh_obj(name, V, [(0, 1, 2, 3)], smooth=False)
    me = ob.data
    uv = me.uv_layers.new(name='UVMap')
    for li, (u, v) in enumerate([(0, 0), (1, 0), (1, 1), (0, 1)]):
        uv.data[li].uv = (u, v)
    ob.location = base
    img = bpy.data.images.load(seq_first_file, check_existing=False)
    img.source = 'SEQUENCE'
    img.colorspace_settings.name = 'Linear Rec.709'
    m, nb = new_material(name + '_m')
    tex = nb.n('ShaderNodeTexImage')
    tex.image = img
    tex.interpolation = 'Cubic'
    tex.extension = 'CLIP'
    iu = tex.image_user
    iu.frame_start = 1
    iu.frame_duration = 100000
    iu.frame_offset = 0
    iu.use_auto_refresh = True
    nb.link(nb.texco().outputs['UV'], tex.inputs['Vector'])
    col = tex.outputs['Color']
    if fog:
        col = nb.colscale(col, C.fog_T(nb))
    em = nb.emission(col, gain)
    tr = nb.n('ShaderNodeBsdfTransparent')
    nb.output(surface=nb.addshader(tr, em))
    m.surface_render_method = 'BLENDED'
    m.use_backface_culling = False
    try:
        m.use_transparency_overlap = True
    except Exception:
        pass
    me.materials.append(m)
    ob.visible_shadow = False
    try:
        ob.visible_volume_scatter = False
    except Exception:
        pass
    con = ob.constraints.new('LOCKED_TRACK')
    con.target = cam
    con.track_axis = 'TRACK_NEGATIVE_Y'
    con.lock_axis = 'LOCK_Z'
    return ob


# ----------------------------------------------------------- fire lights ---

def fire_lights(name, fire_base, Hf, table, power=600.0, radius=0.3, cutoff=None, spread=(0.22, 0.7),
                split=(0.55, 0.45), shadow=True, volume=1.0):
    """Two point lights up the flame axis. table: [(frame, light_multiplier)] (envelope x flicker)."""
    L = []
    for i, (hk, w) in enumerate(zip(spread, split)):
        ob = C.point(f'{name}_L{i}', Vector(fire_base) + Vector((0, 0, hk * Hf)), C.FIRE_LIGHT, 0.0,
                     radius=radius, shadow=shadow, volume=volume, cutoff=cutoff)
        try:
            ob.data.use_shadow_jitter = True
        except Exception:
            pass
        for f, v in table:
            C.key(ob.data, 'energy', f, power * w * v)
        L.append(ob)
    return L


# ------------------------------------------------------------------ volumes ---

def time_value(nb, name='time'):
    """Value node that equals scene time in seconds (linear keys, linear extrapolation)."""
    v = nb.value(0.0, name)
    C.key_socket(v.outputs[0], 0, 0.0)
    C.key_socket(v.outputs[0], 240, 10.0)
    ad = nb.nt.animation_data
    if ad and ad.action:
        for fc in _fcurves(ad.action):
            fc.extrapolation = 'LINEAR'
    return v.outputs[0]


def _fcurves(action):
    try:
        return list(action.fcurves)
    except Exception:
        out = []
        for layer in action.layers:
            for strip in layer.strips:
                for bag in strip.channelbags:
                    out.extend(bag.fcurves)
        return out


def smoke_plume(name, fire_base, t_ign, height=7.0, r0=0.25, spread=0.28, rise=1.3, wind=(0.6, 0.2),
                dens=3.0, albedo=0.35, seed=0.0, t_build=1.2, top_fade=0.75):
    """Billowing smoke column rising from the fire, bent downwind. Volume lit by the fire lights."""
    wx, wy = wind
    wlen = math.hypot(wx, wy) + 1e-9
    yaw = math.atan2(wy, wx)
    ext = height * 0.9
    V = [(x, y, z) for z in (0.0, height) for y in (-ext * 0.5, ext * 0.5) for x in (-1.2, ext * 1.1)]
    F = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
    ob = C.mesh_obj(name, V, F, smooth=False)
    ob.location = fire_base
    ob.rotation_euler = (0, 0, yaw)
    m, nb = new_material(name + '_m')
    t = time_value(nb)
    tc = nb.texco()
    x, y, z = nb.sep(tc.outputs['Object'])
    zz = nb.mx(z, 0.0)
    xc = nb.mul(nb.pw(zz, 1.45), 0.22 * wlen)                   # downwind bend
    rz = nb.madd(zz, spread, r0)
    q2 = nb.div(nb.add(nb.pw(nb.sub(x, xc), 2.0), nb.pw(y, 2.0)), nb.pw(rz, 2.0))
    shape = nb.exp(nb.mul(q2, -1.6))
    # advected noise: coordinates move with the rising gas
    rise_t = nb.mul(t, rise)
    pc = nb.comb(nb.sub(x, nb.mul(t, 0.25 * wlen)), y, nb.sub(z, rise_t))
    n1 = nb.noise(pc, scale=0.9, detail=5.0, rough=0.62, dist=0.35, dims='4D', w=nb.madd(t, 0.15, seed))
    n2 = nb.noise(pc, scale=2.6, detail=3.0, rough=0.55, dims='4D', w=nb.madd(t, 0.3, seed + 3.0))
    billow = nb.clamp01(nb.madd(nb.add(n1.outputs['Fac'], nb.mul(nb.sub(n2.outputs['Fac'], 0.5), 0.5)), 2.6, -1.05))
    # the column front rises from the fire after ignition; fade at the top of the domain
    age = nb.sub(t, t_ign)
    front = nb.sstep(0.0, -1.0, nb.sub(z, nb.madd(age, rise, 0.4)))
    build = nb.sstep(0.05, t_build, age)
    base_fade = nb.sstep(0.35, 0.9, z)
    topf = nb.sstep(height, height * top_fade, z)
    d = nb.mul(nb.mul(nb.mul(nb.mul(nb.mul(shape, billow), front), build), base_fade), topf)
    pv = nb.n('ShaderNodeVolumePrincipled')
    nb.set(pv.inputs['Density'], nb.mul(d, dens))
    pv.inputs['Color'].default_value = (albedo, albedo * 0.97, albedo * 0.95, 1)
    pv.inputs['Absorption Color'].default_value = (0.05, 0.05, 0.05, 1)
    pv.inputs['Anisotropy'].default_value = 0.2
    nb.output(volume=pv)
    ob.data.materials.append(m)
    ob.visible_shadow = False
    return ob


def glow_haze(name, center, radius, density=0.02, aniso=0.35, color=(1, 1, 1), flat=1.0, noise_amt=0.0):
    """Soft sphere of thin air/mist around a fire so its light scatters (in-scatter halo, god rays)."""
    bm = bmesh.new()
    bmesh.ops.create_icosphere(bm, subdivisions=2, radius=radius)
    V, F = _bm_to_lists(bm)
    bm.free()
    V = [(x, y, z * flat) for x, y, z in V]
    ob = C.mesh_obj(name, V, F, smooth=False)
    ob.location = center
    m, nb = new_material(name + '_m')
    tc = nb.texco()
    o = tc.outputs['Object']
    x, y, z = nb.sep(o)
    r = nb.length(nb.comb(x, y, nb.div(z, flat)))
    fall = nb.sstep(radius, radius * 0.35, r)
    dd = fall
    if noise_amt:
        t = time_value(nb)
        nz = nb.noise(nb.vadd(o, nb.comb(nb.mul(t, 0.4), 0.0, nb.mul(t, 0.1))), scale=1.5 / radius * 3, detail=3.0)
        dd = nb.mul(dd, nb.madd(nz.outputs['Fac'], noise_amt * 2, 1.0 - noise_amt))
    pv = nb.n('ShaderNodeVolumePrincipled')
    nb.set(pv.inputs['Density'], nb.mul(dd, density))
    pv.inputs['Color'].default_value = tuple(color) + (1.0,)
    pv.inputs['Absorption Color'].default_value = (0.0, 0.0, 0.0, 1)
    pv.inputs['Anisotropy'].default_value = aniso
    nb.output(volume=pv)
    ob.data.materials.append(m)
    ob.visible_shadow = False
    return ob
