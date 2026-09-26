"""The beacon: dry-stone cairn + iron fire-basket + stacked wood, procedural flame cards,
fire light, glow. Everything deterministic (seeded).

Ignition envelope = the montage department's (catch one frame before, whoomp with ~1.4x
overshoot ~5 frames after the hit, settle to a roar), so every beacon in the film ignites alike.
"""
import math
import random

import bmesh
import bpy
from mathutils import Matrix, Vector

from nodes import Tree
import scene as SC

FIRE_LIGHT = (1.0, 0.50, 0.17)
WIND_DIR = (0.976, 0.217)          # unit vector the wind blows toward (east, a little north)


# ------------------------------------------------------------------ envelope ---
def ignite_env(u):
    """u = frames since ignition -> (size, intensity, light) multipliers."""
    if u < -1.5:
        return 0.0, 0.0, 0.0
    if u < 0.0:
        k = (u + 1.5) / 1.5
        return 0.25 * k, 0.9 * k, 0.3 * k
    rise = 1.0 - math.exp(-u / 1.4)
    over = 0.42 * math.exp(-((u - 5.0) / 4.0) ** 2) + 0.08 * math.exp(-((u - 13.0) / 4.0) ** 2)
    size = 0.5 + 0.5 * rise + over
    inten = 1.0 + 0.9 * math.exp(-u / 4.0)
    light = 1.0 + 1.2 * math.exp(-u / 5.0)
    return size, inten, light


def flicker(t, seed=0.0, amt=1.0):
    s = seed * 1.7
    v = (0.07 * math.sin(2 * math.pi * 1.9 * t + s) + 0.05 * math.sin(2 * math.pi * 3.7 * t + 2.1 + s)
         + 0.035 * math.sin(2 * math.pi * 7.3 * t + 0.7 * s) + 0.02 * math.sin(2 * math.pi * 11.1 * t + s))
    return 1.0 + amt * v


# ------------------------------------------------------------------ materials ---
def stone_material(look):
    if 'Stone' in bpy.data.materials:
        return bpy.data.materials['Stone']
    m = bpy.data.materials.new('Stone')
    m.use_nodes = True
    nt = m.node_tree
    nt.nodes.clear()
    T = Tree(nt)
    out = T.n('ShaderNodeOutputMaterial')
    geo = T.n('ShaderNodeNewGeometry')
    P = geo.outputs['Position']
    n1 = T.noise(P, 6.0, 6.0, 0.6)
    n2 = T.noise(P, 1.3, 3.0, 0.5)
    col = T.mixc(n2, (0.055, 0.050, 0.046, 1), (0.13, 0.12, 0.105, 1))
    # frost/snow caught on the top faces of the stones
    nz = T.sep(geo.outputs['Normal'])[2]
    frost = T.mul(T.smooth(0.55, 0.9, nz), T.smooth(0.45, 0.7, n1))
    col = T.mixc(frost, col, (0.7, 0.74, 0.8, 1))
    bump = T.n('ShaderNodeBump', Strength=0.6, Distance=0.02)
    T.link(n1, bump.inputs['Height'])
    T.link(geo.outputs['Normal'], bump.inputs['Normal'])
    bsdf = T.n('ShaderNodeBsdfPrincipled', Roughness=0.8)
    T.link(col, bsdf.inputs['Base Color'])
    T.link(bump.outputs['Normal'], bsdf.inputs['Normal'])
    ndl = T.vmath('DOT_PRODUCT', bump.outputs['Normal'], tuple(look.moon_dir))
    E = T.mul(T.math('MAXIMUM', ndl, 0.0), look.moon_E)
    em = T.n('ShaderNodeEmission', Color=T.vmath('MULTIPLY', col, T.vmath('SCALE', tuple(look.moon_col), scale=E)))
    add = T.n('ShaderNodeAddShader')
    T.link(bsdf.outputs[0], add.inputs[0])
    T.link(em.outputs[0], add.inputs[1])
    T.link(SC.use_fog(T, add.outputs[0]), out.inputs['Surface'])
    return m


def iron_material(look):
    if 'Iron' in bpy.data.materials:
        return bpy.data.materials['Iron']
    m = bpy.data.materials.new('Iron')
    m.use_nodes = True
    nt = m.node_tree
    nt.nodes.clear()
    T = Tree(nt)
    out = T.n('ShaderNodeOutputMaterial')
    bsdf = T.n('ShaderNodeBsdfPrincipled', Roughness=0.45, Metallic=0.9)
    bsdf.inputs['Base Color'].default_value = (0.05, 0.045, 0.04, 1)
    # hot iron glows dull red near the flame once lit
    heat = T.attr('heat', 'OBJECT')
    geo = T.n('ShaderNodeNewGeometry')
    Pz = T.sep(T.n('ShaderNodeTexCoord').outputs['Object'])[2]
    glow = T.mul(heat, T.smooth(0.9, 0.2, Pz))
    em = T.n('ShaderNodeEmission', Color=(1.0, 0.18, 0.03, 1))
    T.link(T.mul(glow, 2.5), em.inputs['Strength'])
    add = T.n('ShaderNodeAddShader')
    T.link(bsdf.outputs[0], add.inputs[0])
    T.link(em.outputs[0], add.inputs[1])
    T.link(SC.use_fog(T, add.outputs[0]), out.inputs['Surface'])
    return m


def wood_material(look):
    if 'Wood' in bpy.data.materials:
        return bpy.data.materials['Wood']
    m = bpy.data.materials.new('Wood')
    m.use_nodes = True
    nt = m.node_tree
    nt.nodes.clear()
    T = Tree(nt)
    out = T.n('ShaderNodeOutputMaterial')
    geo = T.n('ShaderNodeNewGeometry')
    P = geo.outputs['Position']
    heat = T.attr('heat', 'OBJECT')
    tm = T.attr('ld_time', 'VIEW_LAYER')
    bsdf = T.n('ShaderNodeBsdfPrincipled', Roughness=0.85)
    grain = T.noise(T.vmath('MULTIPLY', P, (40.0, 40.0, 4.0)), 1.0, 3.0, 0.5)
    col = T.mixc(grain, (0.06, 0.04, 0.025, 1), (0.14, 0.09, 0.05, 1))
    col = T.mixc(heat, col, (0.02, 0.018, 0.017, 1))          # charred once burning
    T.link(col, bsdf.inputs['Base Color'])
    # glowing embers in the cracks of the charred wood
    cr = T.noise(P, 14.0, 4.0, 0.6, dims='4D', w=T.mul(tm, 0.8))
    emb = T.mul(T.smooth(0.55, 0.72, cr), heat)
    em = T.n('ShaderNodeEmission', Color=(1.0, 0.28, 0.04, 1))
    T.link(T.mul(emb, 25.0), em.inputs['Strength'])
    add = T.n('ShaderNodeAddShader')
    T.link(bsdf.outputs[0], add.inputs[0])
    T.link(em.outputs[0], add.inputs[1])
    T.link(SC.use_fog(T, add.outputs[0]), out.inputs['Surface'])
    return m


def flame_material():
    """Camera-facing card flame. UV x across [-1, 1], y up [0, 1]. Object props: seed, fint,
    rise (structure speed, card-heights / s)."""
    if 'Flame' in bpy.data.materials:
        return bpy.data.materials['Flame']
    m = bpy.data.materials.new('Flame')
    m.use_nodes = True
    m.surface_render_method = 'BLENDED'
    m.use_backface_culling = False
    nt = m.node_tree
    nt.nodes.clear()
    T = Tree(nt)
    out = T.n('ShaderNodeOutputMaterial')
    uv = T.n('ShaderNodeUVMap').outputs['UV']
    u, v, _ = T.sep(uv)
    X = T.mul(T.sub(u, 0.5), 2.0)
    Y = v
    tm = T.attr('ld_time', 'VIEW_LAYER')
    seed = T.attr('seed', 'OBJECT')
    fint = T.attr('fint', 'OBJECT')
    rise = T.attr('rise', 'OBJECT')
    t = T.add(tm, seed)
    Yc = T.math('MAXIMUM', Y, 0.0)
    # big slow sway + mid-scale billow, both advected upward and growing with height
    n1 = T.noise(T.comb(T.mul(X, 0.7), T.sub(T.mul(Y, 0.9), T.mul(t, T.mul(rise, 0.8))), seed),
                 1.0, 2.0, 0.5, dims='4D', w=T.mul(t, 0.35))
    n2 = T.noise(T.comb(T.mul(X, 1.9), T.sub(T.mul(Y, 2.2), T.mul(t, T.mul(rise, 1.6))),
                        T.add(seed, 5.0)), 1.0, 3.0, 0.55, dims='4D', w=T.mul(t, 0.9))
    Xw = T.add(X, T.mul(T.sub(n1, 0.5), T.mul(T.math('POWER', Yc, 1.1), 1.3)))
    Xw = T.add(Xw, T.mul(T.sub(n2, 0.5), T.mul(Yc, 0.5)))
    Yw = T.add(Y, T.mul(T.sub(n2, 0.5), 0.10))
    om = T.math('MAXIMUM', T.sub(1.0, Yw), 0.0)
    bw = T.mul(T.math('POWER', om, 0.55), 0.86)                    # envelope half-width
    inside = T.sub(bw, T.math('ABSOLUTE', Xw))
    env = T.smooth(-0.02, 0.22, inside)
    # tongues: noise stretched vertically, rising fast; the threshold climbs with height so
    # the body splits into licks and the tips tear off
    tg = T.noise(T.comb(T.mul(Xw, 3.4), T.sub(T.mul(Y, 1.1), T.mul(t, T.mul(rise, 2.3))),
                        T.add(seed, 9.0)), 1.0, 3.0, 0.55, dims='4D', w=T.mul(t, 1.3))
    thr = T.add(0.30, T.mul(T.math('POWER', Yc, 0.8), 0.42))
    tongue = T.smooth(T.sub(thr, 0.07), T.add(thr, 0.07), T.add(tg, T.mul(T.sub(1.0, Yc), 0.25)))
    dens = T.mul(T.mul(env, tongue), T.smooth(-0.02, 0.06, Y))
    core = T.math('MAXIMUM', T.div(inside, T.math('MAXIMUM', bw, 0.02)), 0.0)
    temp = T.mul(dens, T.mul(T.add(0.45, T.mul(core, 0.55)), T.sub(1.0, T.mul(Yc, 0.62))))
    ramp = T.n('ShaderNodeValToRGB')
    cr = ramp.color_ramp
    cr.interpolation = 'LINEAR'
    stops = [(0.00, (0.35, 0.02, 0.0)), (0.30, (0.90, 0.12, 0.01)), (0.55, (1.0, 0.36, 0.04)),
             (0.78, (1.0, 0.68, 0.25)), (1.00, (1.0, 0.95, 0.85))]
    cr.elements[0].position = stops[0][0]
    cr.elements[0].color = (*stops[0][1], 1)
    cr.elements[1].position = stops[-1][0]
    cr.elements[1].color = (*stops[-1][1], 1)
    for pos, c in stops[1:-1]:
        e = cr.elements.new(pos)
        e.color = (*c, 1)
    T.link(temp, ramp.inputs['Fac'])
    I = T.mul(T.math('POWER', temp, 1.6), fint)
    em = T.n('ShaderNodeEmission')
    T.link(ramp.outputs['Color'], em.inputs['Color'])
    T.link(I, em.inputs['Strength'])
    tr = T.n('ShaderNodeBsdfTransparent')
    add = T.n('ShaderNodeAddShader')
    T.link(em.outputs[0], add.inputs[0])
    T.link(tr.outputs[0], add.inputs[1])
    T.link(add.outputs[0], out.inputs['Surface'])
    return m


def sprite_material():
    """Distance-compensated fire point for far beacons: a gaussian core + soft halo (additive).
    Object props: sint (peak), shalo (halo gain)."""
    if 'FireSprite' in bpy.data.materials:
        return bpy.data.materials['FireSprite']
    m = bpy.data.materials.new('FireSprite')
    m.use_nodes = True
    m.surface_render_method = 'BLENDED'
    nt = m.node_tree
    nt.nodes.clear()
    T = Tree(nt)
    out = T.n('ShaderNodeOutputMaterial')
    uv = T.n('ShaderNodeUVMap').outputs['UV']
    u, v, _ = T.sep(uv)
    x = T.mul(T.sub(u, 0.5), 2.0)
    y = T.mul(T.sub(v, 0.5), 2.0)
    # slightly taller than wide: a flame seen far away. Card ~24 px: core ~2 px, halo ~8 px
    r2 = T.add(T.mul(x, x), T.mul(T.mul(y, y), 0.55))
    core = T.math('EXPONENT', T.mul(r2, -80.0))
    halo = T.add(T.math('EXPONENT', T.mul(r2, -7.0)), T.mul(T.math('EXPONENT', T.mul(r2, -1.8)), 0.25))
    # window to exactly zero at the card's edge (no visible square)
    rr = T.add(T.mul(x, x), T.mul(y, y))
    win = T.math('POWER', T.math('MAXIMUM', T.sub(1.0, rr), 0.0), 2.0)
    halo = T.mul(halo, win)
    core = T.mul(core, win)
    sint = T.attr('sint', 'OBJECT')
    shalo = T.attr('shalo', 'OBJECT')
    col = T.vmath('ADD', T.vmath('SCALE', (1.0, 0.78, 0.45), scale=T.mul(core, sint)),
                  T.vmath('SCALE', (1.0, 0.40, 0.10), scale=T.mul(halo, T.mul(sint, shalo))))
    em = T.n('ShaderNodeEmission', Color=col, Strength=1.0)
    tr = T.n('ShaderNodeBsdfTransparent')
    add = T.n('ShaderNodeAddShader')
    T.link(em.outputs[0], add.inputs[0])
    T.link(tr.outputs[0], add.inputs[1])
    T.link(add.outputs[0], out.inputs['Surface'])
    return m


def glow_material():
    """Soft additive halo card (the lit air/smoke around a fire). Object props: gint."""
    if 'Glow' in bpy.data.materials:
        return bpy.data.materials['Glow']
    m = bpy.data.materials.new('Glow')
    m.use_nodes = True
    m.surface_render_method = 'BLENDED'
    nt = m.node_tree
    nt.nodes.clear()
    T = Tree(nt)
    out = T.n('ShaderNodeOutputMaterial')
    uv = T.n('ShaderNodeUVMap').outputs['UV']
    u, v, _ = T.sep(uv)
    x = T.mul(T.sub(u, 0.5), 2.0)
    y = T.mul(T.sub(v, 0.5), 2.0)
    r2 = T.add(T.mul(x, x), T.mul(y, y))
    g = T.add(T.mul(T.math('EXPONENT', T.mul(r2, -9.0)), 0.8), T.mul(T.math('EXPONENT', T.mul(r2, -2.5)), 0.2))
    gint = T.attr('gint', 'OBJECT')
    em = T.n('ShaderNodeEmission', Color=(1.0, 0.42, 0.12, 1))
    T.link(T.mul(g, gint), em.inputs['Strength'])
    tr = T.n('ShaderNodeBsdfTransparent')
    add = T.n('ShaderNodeAddShader')
    T.link(em.outputs[0], add.inputs[0])
    T.link(tr.outputs[0], add.inputs[1])
    T.link(add.outputs[0], out.inputs['Surface'])
    return m


def smoke_material(look):
    """Camera-facing plume card (UV x [-1,1], y [0,1]): billowing, leaning downwind, lit warm
    from the fire beneath, cool and faint above. Object props: seed, lean, sfire, sdens."""
    if 'Smoke' in bpy.data.materials:
        return bpy.data.materials['Smoke']
    m = bpy.data.materials.new('Smoke')
    m.use_nodes = True
    m.surface_render_method = 'BLENDED'
    m.use_backface_culling = False
    nt = m.node_tree
    nt.nodes.clear()
    T = Tree(nt)
    out = T.n('ShaderNodeOutputMaterial')
    uv = T.n('ShaderNodeUVMap').outputs['UV']
    u, v, _ = T.sep(uv)
    X = T.mul(T.sub(u, 0.5), 2.0)
    Y = v
    tm = T.attr('ld_time', 'VIEW_LAYER')
    seed = T.attr('seed', 'OBJECT')
    lean = T.attr('lean', 'OBJECT')
    sfire = T.attr('sfire', 'OBJECT')
    sdens = T.attr('sdens', 'OBJECT')
    t = T.add(tm, seed)
    xc = T.mul(lean, T.math('POWER', Y, 1.4))
    w = T.add(0.10, T.mul(Y, 0.42))
    n1 = T.noise(T.comb(T.mul(X, 1.6), T.sub(T.mul(Y, 2.4), T.mul(t, 0.22)), seed), 1.0, 4.0, 0.6,
                 dims='4D', w=T.mul(t, 0.12))
    n2 = T.noise(T.comb(T.mul(X, 4.5), T.sub(T.mul(Y, 6.0), T.mul(t, 0.55)), T.add(seed, 3.0)),
                 1.0, 3.0, 0.55, dims='4D', w=T.mul(t, 0.3))
    xd = T.add(T.sub(X, xc), T.mul(T.sub(n1, 0.5), T.mul(Y, 0.6)))
    prof = T.math('EXPONENT', T.mul(T.mul(T.div(xd, w), T.div(xd, w)), -1.0))
    bill = T.smooth(0.30, 0.72, T.add(T.add(T.mul(n1, 0.7), T.mul(n2, 0.35)), T.mul(T.sub(1.0, Y), 0.22)))
    dens = T.mul(T.mul(prof, bill), T.mul(T.smooth(0.0, 0.07, Y), T.sub(1.0, T.smooth(0.55, 1.0, Y))))
    alpha = T.math('MINIMUM', T.mul(dens, sdens), 0.95)
    # light: fire from beneath (strong low down), a little moon/sky above
    firelit = T.mul(T.math('EXPONENT', T.mul(Y, -6.0)), sfire)
    col = T.vmath('ADD', T.vmath('SCALE', (1.0, 0.42, 0.13), scale=firelit),
                  T.vmath('SCALE', tuple(look.moon_col), scale=0.028))
    col = T.vmath('ADD', col, (0.006, 0.0055, 0.005))
    em = T.n('ShaderNodeEmission', Color=col, Strength=1.0)
    tr = T.n('ShaderNodeBsdfTransparent')
    mix = T.n('ShaderNodeMixShader')
    T.link(alpha, mix.inputs[0])
    T.link(tr.outputs[0], mix.inputs[1])
    T.link(em.outputs[0], mix.inputs[2])
    T.link(SC.use_fog(T, mix.outputs[0]), out.inputs['Surface'])
    return m


# ------------------------------------------------------------------- geometry ---
def _mesh_obj(name, bm):
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    ob = bpy.data.objects.new(name, me)
    return ob


def cairn_mesh(seed, scale=1.0):
    """A squarish, slightly tapered dry-stone pillar of flat stones (~1.0 m), top at z=top."""
    rng = random.Random(seed)
    bm = bmesh.new()
    H = 1.0 * scale
    z = 0.0
    course = 0
    while z < H - 0.02 * scale:
        ch = rng.uniform(0.085, 0.13) * scale
        frac = z / H
        half = (0.36 - 0.07 * frac) * scale         # half-width of the pillar
        # four faces, stones laid along each face, offset per course
        for side in range(4):
            ang = side * math.pi / 2
            ox = 0.5 * rng.random() if course % 2 else 0.0
            x = -half + ox * 0.18 * scale
            while x < half - 0.03 * scale:
                ln = rng.uniform(0.16, 0.30) * scale
                ln = min(ln, half - x)
                dep = rng.uniform(0.16, 0.24) * scale
                cx = x + ln / 2
                cy = half - dep / 2 + rng.uniform(-0.012, 0.012) * scale
                r = bmesh.ops.create_cube(bm, size=1.0)
                vs = r['verts']
                sx, sy, sz = ln * 0.96, dep, ch * rng.uniform(0.86, 0.98)
                for vv in vs:
                    vv.co = Vector((vv.co.x * sx, vv.co.y * sy, vv.co.z * sz))
                    vv.co += Vector((rng.uniform(-0.12, 0.12) * sx, rng.uniform(-0.1, 0.1) * sy,
                                     rng.uniform(-0.18, 0.18) * sz))
                rot = Matrix.Rotation(ang + rng.uniform(-0.05, 0.05), 3, 'Z')
                tilt = Matrix.Rotation(rng.uniform(-0.05, 0.05), 3, 'X')
                for vv in vs:
                    vv.co = rot @ (tilt @ vv.co + Vector((cx, cy, z + sz / 2)))
                x += ln
        # hearting: a core block so no light leaks through
        r = bmesh.ops.create_cube(bm, size=1.0)
        for vv in r['verts']:
            vv.co = Vector((vv.co.x * half * 1.5, vv.co.y * half * 1.5, vv.co.z * ch + z + ch / 2))
        z += ch
        course += 1
    # cap slab
    r = bmesh.ops.create_cube(bm, size=1.0)
    for vv in r['verts']:
        vv.co = Vector((vv.co.x * 0.62 * scale, vv.co.y * 0.6 * scale, vv.co.z * 0.07 * scale + z + 0.035 * scale))
        vv.co += Vector((rng.uniform(-0.02, 0.02), rng.uniform(-0.02, 0.02), rng.uniform(-0.01, 0.01))) * scale
    top = z + 0.07 * scale
    return bm, top


def basket_mesh(scale=1.0, bars=10):
    """Flared iron cage: bottom ring r0, top ring r1, bars bowed outward. Origin at its base."""
    bm = bmesh.new()
    r0, r1, h, th = 0.27 * scale, 0.47 * scale, 0.48 * scale, 0.017 * scale

    def tube(p0, p1, rad, seg=6):
        d = p1 - p0
        L = d.length
        r = bmesh.ops.create_cone(bm, cap_ends=True, cap_tris=False, segments=seg, radius1=rad,
                                  radius2=rad, depth=L)
        q = d.to_track_quat('Z', 'Y')
        M = Matrix.Translation((p0 + p1) / 2) @ q.to_matrix().to_4x4()
        bmesh.ops.transform(bm, matrix=M, verts=r['verts'])

    def ring(z, rad, n=28):
        pts = [Vector((rad * math.cos(2 * math.pi * i / n), rad * math.sin(2 * math.pi * i / n), z))
               for i in range(n)]
        for i in range(n):
            tube(pts[i], pts[(i + 1) % n], th)

    ring(0.0, r0)
    ring(h, r1)
    ring(h * 0.55, r0 + (r1 - r0) * 0.62)
    for b in range(bars):
        a = 2 * math.pi * b / bars
        prev = None
        for k in range(5):
            s = k / 4
            rad = r0 + (r1 - r0) * (s ** 0.8) + 0.02 * scale * math.sin(math.pi * s)
            p = Vector((rad * math.cos(a), rad * math.sin(a), h * s))
            if prev is not None:
                tube(prev, p, th)
            prev = p
        # short legs down onto the cap
    for b in range(4):
        a = 2 * math.pi * b / 4 + 0.3
        tube(Vector((r0 * math.cos(a), r0 * math.sin(a), 0.0)),
             Vector((r0 * 0.8 * math.cos(a), r0 * 0.8 * math.sin(a), -0.06 * scale)), th * 1.3)
    return bm, h


def wood_mesh(seed, scale=1.0):
    rng = random.Random(seed + 7)
    bm = bmesh.new()
    z = 0.05 * scale
    for layer in range(4):
        ang = (layer % 2) * math.pi / 2 + rng.uniform(-0.2, 0.2)
        for k in range(3):
            off = (k - 1) * 0.13 * scale + rng.uniform(-0.02, 0.02) * scale
            L = rng.uniform(0.5, 0.7) * scale * (1.0 + 0.15 * layer)
            rad = rng.uniform(0.035, 0.05) * scale
            r = bmesh.ops.create_cone(bm, cap_ends=True, segments=7, radius1=rad, radius2=rad * 0.9, depth=L)
            M = (Matrix.Rotation(ang, 4, 'Z') @ Matrix.Translation((off, 0, z))
                 @ Matrix.Rotation(math.pi / 2, 4, 'X') @ Matrix.Rotation(rng.uniform(-0.1, 0.1), 4, 'Y'))
            bmesh.ops.transform(bm, matrix=M, verts=r['verts'])
        z += 0.085 * scale
    # a few upright kindling sticks
    for k in range(5):
        a = rng.uniform(0, 2 * math.pi)
        rr = rng.uniform(0.0, 0.12) * scale
        L = rng.uniform(0.35, 0.55) * scale
        r = bmesh.ops.create_cone(bm, cap_ends=True, segments=5, radius1=0.02 * scale, radius2=0.012 * scale, depth=L)
        M = (Matrix.Translation((rr * math.cos(a), rr * math.sin(a), 0.12 * scale + L / 2))
             @ Matrix.Rotation(rng.uniform(-0.35, 0.35), 4, 'X') @ Matrix.Rotation(rng.uniform(-0.35, 0.35), 4, 'Y'))
        bmesh.ops.transform(bm, matrix=M, verts=r['verts'])
    return bm


def card_mesh(name, w, h, centered=False):
    bm = bmesh.new()
    y0 = -h / 2 if centered else 0.0
    vs = [bm.verts.new((-w / 2, 0, y0)), bm.verts.new((w / 2, 0, y0)),
          bm.verts.new((w / 2, 0, y0 + h)), bm.verts.new((-w / 2, 0, y0 + h))]
    f = bm.faces.new(vs)
    uvl = bm.loops.layers.uv.new('UVMap')
    for loop, uvc in zip(f.loops, ((0, 0), (1, 0), (1, 1), (0, 1))):
        loop[uvl].uv = uvc
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    return me


# --------------------------------------------------------------------- beacon ---
class Beacon:
    def __init__(self, look, name, pos, t_ign, scale, cam, detail=True, shadows=False,
                 light=True, seed=0, far_boost=1.0, sprite=False, smoke=False):
        self.name = name
        self.pos = Vector(pos)
        self.t_ign = t_ign
        self.scale = scale
        self.seed = seed
        self.objs = []
        coll = bpy.context.scene.collection
        self.flames = []
        top = 1.07 * scale
        if detail:
            bm, top = cairn_mesh(seed, scale)
            c = _mesh_obj(name + '_cairn', bm)
            c.data.materials.append(stone_material(look))
            c.location = self.pos + Vector((0, 0, -0.12 * scale))
            c.rotation_euler = (0, 0, random.Random(seed).uniform(0, 6.28))
            coll.objects.link(c)
            top = top - 0.12 * scale
            bm, bh = basket_mesh(scale)
            b = _mesh_obj(name + '_basket', bm)
            b.data.materials.append(iron_material(look))
            b.location = self.pos + Vector((0, 0, top + 0.06 * scale))
            coll.objects.link(b)
            b['heat'] = 0.0
            wm = wood_mesh(seed, scale)
            wd = _mesh_obj(name + '_wood', wm)
            wd.data.materials.append(wood_material(look))
            wd.location = b.location.copy()
            coll.objects.link(wd)
            wd['heat'] = 0.0
            self.hot = [b, wd]
            base_z = top + 0.14 * scale
        else:
            self.hot = []
            base_z = top
        self.fire_base = self.pos + Vector((0, 0, base_z))
        # flame cards: a cluster, each faces the camera about Z
        fm = flame_material()
        rng = random.Random(seed * 31 + 5)
        n_cards = 5 if detail else 0
        for k in range(n_cards):
            wk = rng.uniform(1.4, 1.8) * scale * far_boost if k else 1.9 * scale * far_boost
            hk = rng.uniform(2.8, 3.6) * scale * far_boost if k else 4.4 * scale * far_boost
            me = card_mesh(name + '_flame%d' % k, wk, hk)
            fo = bpy.data.objects.new(name + '_flame%d' % k, me)
            fo.data.materials.append(fm)
            off = Vector((rng.uniform(-0.22, 0.22), rng.uniform(-0.22, 0.22), 0)) * scale if k else Vector()
            fo.location = self.fire_base + off - Vector((0, 0, 0.18 * scale))
            fo['seed'] = rng.uniform(0, 100)
            fo['fint'] = 0.0
            fo['rise'] = 1.1 / (hk / scale / far_boost)
            con = fo.constraints.new('LOCKED_TRACK')
            con.target = cam
            con.track_axis = 'TRACK_NEGATIVE_Y'
            con.lock_axis = 'LOCK_Z'
            coll.objects.link(fo)
            self.flames.append((fo, hk))
        # glow card
        gm = glow_material()
        gs = 7.0 * scale * far_boost
        me = card_mesh(name + '_glow', gs, gs, centered=True)
        go = bpy.data.objects.new(name + '_glow', me)
        go.data.materials.append(gm)
        go.location = self.fire_base + Vector((0, 0, 1.2 * scale))
        go['gint'] = 0.0
        con = go.constraints.new('TRACK_TO')
        con.target = cam
        con.track_axis = 'TRACK_NEGATIVE_Y'
        con.up_axis = 'UP_Z'
        coll.objects.link(go)
        self.glow = go
        # smoke plume card
        self.smoke = None
        if smoke:
            sw, sh = 12.0 * scale, 48.0 * scale
            me = card_mesh(name + '_smoke', sw, sh)
            so = bpy.data.objects.new(name + '_smoke', me)
            so.data.materials.append(smoke_material(look))
            so.location = self.fire_base + Vector((0, 0, 0.6 * scale))
            so['seed'] = random.Random(seed * 7 + 1).uniform(0, 50)
            so['lean'] = 0.0
            so['sfire'] = 0.0
            so['sdens'] = 0.0
            con = so.constraints.new('LOCKED_TRACK')
            con.target = cam
            con.track_axis = 'TRACK_NEGATIVE_Y'
            con.lock_axis = 'LOCK_Z'
            coll.objects.link(so)
            self.smoke = so
        # distance-compensated sprite (keeps far fires a crisp, blooming point)
        self.sprite = None
        if sprite:
            me = card_mesh(name + '_sprite', 1.0, 1.0, centered=True)
            so = bpy.data.objects.new(name + '_sprite', me)
            so.data.materials.append(sprite_material())
            so.location = self.fire_base + Vector((0, 0, 1.0 * scale))
            so['sint'] = 0.0
            so['shalo'] = 0.035
            con = so.constraints.new('TRACK_TO')
            con.target = cam
            con.track_axis = 'TRACK_NEGATIVE_Y'
            con.up_axis = 'UP_Z'
            coll.objects.link(so)
            self.sprite = so
        # light
        self.light = None
        if light:
            ld = bpy.data.lights.new(name + '_light', 'POINT')
            ld.color = FIRE_LIGHT
            ld.shadow_soft_size = 0.35 * scale
            ld.use_shadow = shadows
            ld.energy = 0.0
            lo = bpy.data.objects.new(name + '_light', ld)
            lo.location = self.fire_base + Vector((0, 0, 0.7 * scale))
            coll.objects.link(lo)
            self.light = lo

    def animate(self, frames, fps=24.0, power=420.0, fint=7.0, glow=0.10, cam_path=None,
                px_deg=1920.0 / 62.0, sprite_px=24.0, sprite_peak=70.0, cam_right=None):
        """Keyframe ignition + flicker for all frames. cam_path: {frame: (x, y, z)} for the
        distance-compensated sprite."""
        for f in frames:
            u = f - self.t_ign
            size, inten, lightm = ignite_env(u)
            t = f / fps
            fl = flicker(t, self.seed)
            on = 1.0 if size > 0 else 0.0
            for fo, hk in self.flames:
                fo.scale = (1.0 + 0.12 * (size - 1.0), 1.0, max(size, 0.001))
                fo.keyframe_insert('scale', frame=f)
                fo['fint'] = fint * inten * on * (0.9 + 0.1 * fl)
                fo.keyframe_insert('["fint"]', frame=f)
            self.glow['gint'] = glow * lightm * fl * on
            self.glow.keyframe_insert('["gint"]', frame=f)
            for h in self.hot:
                h['heat'] = min(max(u / 12.0, 0.0), 1.0)
                h.keyframe_insert('["heat"]', frame=f)
            if self.light is not None:
                self.light.data.energy = power * self.scale ** 2 * lightm * fl * on
                self.light.data.keyframe_insert('energy', frame=f)
            if self.smoke is not None:
                grow = min(max(u / 30.0, 0.0), 1.0)
                self.smoke['sdens'] = 0.9 * grow ** 0.7
                self.smoke['sfire'] = 0.55 * lightm * fl * on
                if cam_right is not None and f in cam_right:
                    r = cam_right[f]
                    self.smoke['lean'] = 0.55 * (WIND_DIR[0] * r[0] + WIND_DIR[1] * r[1])
                self.smoke.keyframe_insert('["sdens"]', frame=f)
                self.smoke.keyframe_insert('["sfire"]', frame=f)
                self.smoke.keyframe_insert('["lean"]', frame=f)
            if self.sprite is not None and cam_path is not None and f in cam_path:
                c = cam_path[f]
                d = math.dist(c, tuple(self.sprite.location))
                # world size giving ~sprite_px pixels, never smaller than the real fire
                ang = math.radians(sprite_px / px_deg)
                sz = max(d * ang, 8.0 * self.scale) * (0.7 + 0.3 * size)
                self.sprite.scale = (sz, sz, sz)
                self.sprite.keyframe_insert('scale', frame=f)
                # keep the integrated brightness of a real fire when it is resolved
                real = 2.2 * self.scale
                k = min(1.0, (real / (d * ang)) ** 2) if d * ang > 0 else 1.0
                self.sprite['sint'] = sprite_peak * inten * fl * on * max(k, 0.0) ** 0.0
                self.sprite.keyframe_insert('["sint"]', frame=f)
