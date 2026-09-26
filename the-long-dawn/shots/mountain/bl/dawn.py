"""DAWN_C: the morning after, in the same world. Sky, sun-lit materials, sunlit smoke, eagles.

The sun is emission-coded like the moon at night, but animated: the scene property 'sun_el'
(radians, keyframed per frame from cache/dawn_sun.json) is compared with each vertex's baked
horizon angle toward the sun ('sunhor'), so light floods down the peaks and across the cloud sea
exactly as the sun clears the eastern ridge.
"""
import math

import bpy
from mathutils import Vector

from nodes import Tree
import scene as SC


def lin(h):
    return SC.srgb(h)


class DawnLook:
    def __init__(self, sun_az):
        self.sun_az = sun_az
        self.sun_col = (1.0, 0.74, 0.46)          # low sun: dawn gold
        self.sun_E = 3.2                           # sunlight irradiance scale (emission-coded)
        self.zenith = tuple(0.55 * c for c in lin('#1B3766'))
        self.mid = tuple(0.8 * c for c in lin('#5B7FB4'))
        self.hor_cool = tuple(0.8 * c for c in lin('#8C97B8'))
        self.hor_warm = tuple(1.0 * c for c in lin('#F4B684'))
        self.gold = lin('#FFD9A0')
        self.sky_fill = 0.30
        self.fill_col = (0.62, 0.72, 1.0)
        self.sun_disc = math.radians(0.42)
        self.sun_I = 900.0
        self.mist_d0 = 1.8e-3
        self.mist_H = 55.0
        self.haze_d0 = 2.6e-5
        self.haze_H = 1800.0


def _sun_dir(T):
    """World sun direction from scene props (sun_az constant, sun_el animated)."""
    el = T.attr('sun_el', 'VIEW_LAYER')
    az = T.attr('sun_az', 'VIEW_LAYER')
    ce = T.math('COSINE', el)
    x = T.mul(T.math('SINE', az), ce)
    y = T.mul(T.math('COSINE', az), ce)
    z = T.math('SINE', el)
    return T.comb(x, y, z), el


def _sun_strength(T, el):
    """The sun gains strength as it climbs out of the horizon murk (0 at -0.4 deg, full at 4)."""
    return T.smooth(math.radians(-0.4), math.radians(4.0), el)


# ------------------------------------------------------------------------ fog ---
def dawn_fog_group(look):
    ng = bpy.data.node_groups.new('AerialFog', 'ShaderNodeTree')
    ng.interface.new_socket('Shader', in_out='INPUT', socket_type='NodeSocketShader')
    ng.interface.new_socket('Shader', in_out='OUTPUT', socket_type='NodeSocketShader')
    ng.interface.new_socket('Boost', in_out='INPUT', socket_type='NodeSocketFloat')
    T = Tree(ng)
    gi = T.n('NodeGroupInput')
    go = T.n('NodeGroupOutput')
    geo = T.n('ShaderNodeNewGeometry')
    cam = T.n('ShaderNodeCameraData')
    P = geo.outputs['Position']
    I = geo.outputs['Incoming']
    L = cam.outputs['View Distance']
    pz = T.sep(P)[2]
    iz = T.sep(I)[2]
    cz = T.madd(iz, L, pz)
    boost = T.add(1.0, gi.outputs['Boost'])
    tm = T.mul(SC._layer_tau(T, L, pz, cz, look.mist_d0, look.mist_H), boost)
    th = T.mul(SC._layer_tau(T, L, pz, cz, look.haze_d0, look.haze_H), boost)
    Tm = T.math('EXPONENT', T.mul(tm, -1.0))
    Th = T.math('EXPONENT', T.mul(th, -1.0))
    sd, el = _sun_dir(T)
    ss = _sun_strength(T, el)
    vd = T.vmath('SCALE', I, scale=-1.0)
    cth = T.vmath('DOT_PRODUCT', vd, sd)
    g = 0.78
    hg = T.div(1 - g * g, T.math('POWER', T.sub(1 + g * g, T.mul(2 * g, cth)), 1.5))
    # the haze: blue-grey away from the sun, blazing gold toward it
    base = T.mixc(T.smooth(-0.2, 0.9, cth), (*look.hor_cool, 1), (*look.hor_warm, 1))
    hcol = T.vmath('ADD', T.vmath('SCALE', base, scale=0.55),
                   T.vmath('SCALE', look.gold, scale=T.mul(T.mul(hg, 0.08), ss)))
    mcol = T.vmath('ADD', T.vmath('SCALE', base, scale=0.5),
                   T.vmath('SCALE', look.sun_col, scale=T.mul(T.mul(hg, 0.10), ss)))
    em_m = T.n('ShaderNodeEmission', Color=mcol, Strength=1.0)
    em_h = T.n('ShaderNodeEmission', Color=hcol, Strength=1.0)
    mix1 = T.n('ShaderNodeMixShader')
    T.link(T.sub(1.0, Tm), mix1.inputs[0])
    T.link(gi.outputs['Shader'], mix1.inputs[1])
    T.link(em_m.outputs[0], mix1.inputs[2])
    mix2 = T.n('ShaderNodeMixShader')
    T.link(T.sub(1.0, Th), mix2.inputs[0])
    T.link(mix1.outputs[0], mix2.inputs[1])
    T.link(em_h.outputs[0], mix2.inputs[2])
    T.link(mix2.outputs[0], go.inputs['Shader'])
    return ng


# ------------------------------------------------------------------------ sky ---
def build_dawn_world(look):
    w = bpy.data.worlds.new('DawnSky')
    bpy.context.scene.world = w
    w.use_nodes = True
    nt = w.node_tree
    nt.nodes.clear()
    T = Tree(nt)
    out = T.n('ShaderNodeOutputWorld')
    tc = T.n('ShaderNodeTexCoord')
    D = T.vmath('NORMALIZE', tc.outputs['Generated'])
    e = T.sep(D)[2]
    sd, el = _sun_dir(T)
    ss = _sun_strength(T, el)
    cth = T.vmath('DOT_PRODUCT', D, sd)
    # azimuthal closeness to the sun (horizontal)
    dh = T.vmath('NORMALIZE', T.vmath('MULTIPLY', D, (1.0, 1.0, 0.0)))
    sh = T.vmath('NORMALIZE', T.vmath('MULTIPLY', sd, (1.0, 1.0, 0.0)))
    caz = T.vmath('DOT_PRODUCT', dh, sh)
    hor = T.mixc(T.smooth(-0.3, 1.0, caz), (*look.hor_cool, 1), (*look.hor_warm, 1))
    c1 = T.mixc(T.smooth(-0.01, 0.20, e), hor, (*look.mid, 1))
    sky = T.mixc(T.smooth(0.15, 0.9, e), c1, (*look.zenith, 1))
    # brighten the whole sky a little as the sun comes up
    sky = T.vmath('SCALE', sky, scale=T.add(0.75, T.mul(ss, 0.35)))
    # aureole around the sun (Mie), both for camera rays and lighting
    au = T.add(T.mul(T.math('EXPONENT', T.mul(T.sub(cth, 1.0), 60.0)), 2.2),
               T.mul(T.math('EXPONENT', T.mul(T.sub(cth, 1.0), 7.0)), 0.30))
    au = T.mul(au, T.add(0.25, T.mul(ss, 0.75)))
    sky = T.vmath('ADD', sky, T.vmath('SCALE', look.gold, scale=au))
    # the sun disc (camera rays only)
    cosr = math.cos(look.sun_disc)
    disc = T.smooth(cosr - 0.0000012, cosr + 0.0000012, cth)
    sun = T.vmath('SCALE', (1.0, 0.93, 0.82), scale=T.mul(disc, look.sun_I))
    cam_sky = T.vmath('ADD', sky, sun)
    lp = T.n('ShaderNodeLightPath')
    amb = T.vmath('SCALE', sky, scale=0.8)
    sel = T.mixc(lp.outputs['Is Camera Ray'], amb, cam_sky)
    bg = T.n('ShaderNodeBackground', Color=sel, Strength=1.0)
    T.link(bg.outputs[0], out.inputs['Surface'])
    return w


# ---------------------------------------------------------------- materials ---
def dawn_land_material(look):
    m = bpy.data.materials.new('LandDawn')
    m.use_nodes = True
    nt = m.node_tree
    nt.nodes.clear()
    T = Tree(nt)
    out = T.n('ShaderNodeOutputMaterial')
    geo = T.n('ShaderNodeNewGeometry')
    P = geo.outputs['Position']
    N = geo.outputs['Normal']
    snow_a = T.attr('snow')
    ao = T.attr('ao')
    sunhor = T.attr('sunhor')
    Pz = T.sep(P)[2]
    Nz = T.sep(N)[2]
    Tc = T.vmath('NORMALIZE', T.vmath('CROSS_PRODUCT', N, (0.0, 0.0, 1.0)))
    u = T.vmath('DOT_PRODUCT', P, Tc)
    fl_c = T.comb(T.div(u, 9.0), T.div(Pz, 70.0), 0.0)
    flute = T.noise(fl_c, 1.0, 2.0, 0.55)
    lw = T.noise(T.vmath('SCALE', P, scale=1.0 / 90.0), 1.0, 2.0, 0.5)
    ledge = T.noise(T.comb(T.div(u, 160.0), T.add(T.div(Pz, 7.0), T.mul(lw, 6.0)), 3.0),
                    1.0, 2.0, 0.5)
    r1 = T.noise(P, 0.6, 3.0, 0.55)
    rock_h = T.add(T.mul(r1, 0.35), T.add(T.mul(flute, 0.35), T.mul(ledge, 0.35)))
    bump_r = T.n('ShaderNodeBump', Strength=0.45, Distance=1.0)
    T.link(rock_h, bump_r.inputs['Height'])
    T.link(N, bump_r.inputs['Normal'])
    sn = T.noise(T.vmath('MULTIPLY', P, (0.08, 0.08, 0.2)), 1.0, 4.0, 0.5)
    sas = T.noise(T.vmath('MULTIPLY', P, (0.9, 0.25, 0.9)), 1.0, 3.0, 0.5)
    sn = T.add(T.mul(sn, 0.7), T.mul(sas, 0.3))
    bump_s = T.n('ShaderNodeBump', Strength=0.22, Distance=0.8)
    T.link(sn, bump_s.inputs['Height'])
    T.link(N, bump_s.inputs['Normal'])
    edge = T.noise(T.vmath('SCALE', P, scale=1.0 / 7.0), 1.0, 4.0, 0.6)
    edge2 = T.noise(T.vmath('SCALE', P, scale=1.0 / 45.0), 1.0, 3.0, 0.55)
    s = T.add(snow_a, T.mul(T.sub(edge, 0.5), 0.22))
    s = T.add(s, T.mul(T.sub(edge2, 0.5), 0.28))
    steepness = T.smooth(0.85, 0.5, Nz)
    s = T.add(s, T.mul(T.mul(T.smooth(0.54, 0.64, ledge), steepness), 0.55))
    snow = T.smooth(0.40, 0.56, s)
    strata_c = T.smooth(0.35, 0.65, ledge)
    rock_c = T.mixc(strata_c, (0.070, 0.066, 0.062, 1), (0.13, 0.12, 0.11, 1))
    rock_c = T.mixc(T.smooth(0.55, 0.75, r1), rock_c, (0.045, 0.044, 0.046, 1))
    snow_c = T.mixc(sn, (0.78, 0.81, 0.87, 1), (0.88, 0.90, 0.94, 1))
    alb = T.mixc(snow, rock_c, snow_c)
    nrm = T.vmath('NORMALIZE', T.mixc(snow, bump_r.outputs['Normal'], bump_s.outputs['Normal']))
    sd, el = _sun_dir(T)
    ss = _sun_strength(T, el)
    # lit where the sun has cleared this vertex's own eastern horizon (soft: the disc's size)
    lit = T.smooth(T.sub(sunhor, math.radians(0.30)), T.add(sunhor, math.radians(0.30)), el)
    ndl = T.vmath('DOT_PRODUCT', nrm, sd)
    wrap = T.mixf(snow, 0.0, 0.10)
    lam = T.math('MAXIMUM', T.div(T.add(ndl, wrap), T.add(1.0, wrap)), 0.0)
    E = T.mul(T.mul(T.mul(lam, lit), ss), look.sun_E)
    sunlit = T.vmath('MULTIPLY', alb, T.vmath('SCALE', look.sun_col, scale=E))
    up = T.add(T.mul(Nz, 0.5), 0.5)
    fillE = T.mul(T.mul(T.add(T.mul(up, 0.6), 0.4), T.add(T.mul(ao, 0.7), 0.3)), look.sky_fill)
    sunlit = T.vmath('ADD', sunlit, T.vmath('MULTIPLY', alb, T.vmath('SCALE', look.fill_col, scale=fillE)))
    em = T.n('ShaderNodeEmission', Color=sunlit, Strength=1.0)
    bsdf = T.n('ShaderNodeBsdfPrincipled', Roughness=0.62)
    T.link(T.vmath('SCALE', alb, scale=T.add(T.mul(ao, 0.8), 0.2)), bsdf.inputs['Base Color'])
    T.link(nrm, bsdf.inputs['Normal'])
    bsdf.inputs['Specular IOR Level'].default_value = 0.25
    add = T.n('ShaderNodeAddShader')
    T.link(bsdf.outputs[0], add.inputs[0])
    T.link(em.outputs[0], add.inputs[1])
    T.link(SC.use_fog(T, add.outputs[0]), out.inputs['Surface'])
    return m


def dawn_cloud_material(look):
    m = bpy.data.materials.new('CloudDawn')
    m.use_nodes = True
    m.surface_render_method = 'DITHERED'
    nt = m.node_tree
    nt.nodes.clear()
    T = Tree(nt)
    out = T.n('ShaderNodeOutputMaterial')
    geo = T.n('ShaderNodeNewGeometry')
    P = geo.outputs['Position']
    N = geo.outputs['Normal']
    I = geo.outputs['Incoming']
    clear = T.attr('clear')
    sunhor = T.attr('sunhor')
    hol = T.attr('hollow')
    tm = T.attr('ld_time', 'VIEW_LAYER')
    drift = T.vmath('ADD', P, T.comb(T.mul(tm, -3.0), T.mul(tm, -1.2), 0.0))
    d1 = T.noise(T.vmath('SCALE', drift, scale=1.0 / 70.0), 1.0, 4.0, 0.55, dims='4D',
                 w=T.mul(tm, 0.03))
    d2 = T.noise(T.vmath('SCALE', drift, scale=1.0 / 18.0), 1.0, 2.0, 0.5)
    hb = T.add(d1, T.mul(d2, 0.12))
    bump = T.n('ShaderNodeBump', Strength=0.22, Distance=6.0)
    T.link(hb, bump.inputs['Height'])
    T.link(N, bump.inputs['Normal'])
    nrm = bump.outputs['Normal']
    sd, el = _sun_dir(T)
    ss = _sun_strength(T, el)
    ndl = T.vmath('DOT_PRODUCT', nrm, sd)
    wrap = T.math('MAXIMUM', T.div(T.add(ndl, 0.6), 1.6), 0.0)
    vd = T.vmath('SCALE', I, scale=-1.0)
    cth = T.vmath('DOT_PRODUCT', vd, sd)
    g = 0.62
    hg = T.div(1 - g * g, T.math('POWER', T.sub(1 + g * g, T.mul(2 * g, cth)), 1.5))
    lit = T.smooth(T.sub(sunhor, math.radians(0.35)), T.add(sunhor, math.radians(0.35)), el)
    cav = T.mul(T.maprange(hb, 0.3, 0.7, 0.8, 1.0), T.maprange(hol, -0.2, 1.0, 1.0, 0.6))
    lightf = T.mul(T.mul(T.mul(T.add(T.mul(wrap, 0.7), T.mul(hg, 0.30)), lit), ss), look.sun_E * 0.8)
    lightf = T.mul(lightf, cav)
    alb = (0.80, 0.82, 0.88, 1)
    col = T.vmath('SCALE', look.sun_col, scale=lightf)
    # the sky's blue fill on the deck (brighter at dawn than at night)
    col = T.vmath('ADD', col, T.vmath('SCALE', look.fill_col, scale=T.mul(cav, look.sky_fill * 0.9)))
    em = T.n('ShaderNodeEmission', Color=T.vmath('MULTIPLY', col, alb), Strength=1.0)
    dif = T.n('ShaderNodeBsdfDiffuse', Color=alb)
    T.link(nrm, dif.inputs['Normal'])
    add = T.n('ShaderNodeAddShader')
    T.link(dif.outputs[0], add.inputs[0])
    T.link(em.outputs[0], add.inputs[1])
    fogged = SC.use_fog(T, add.outputs[0])
    wisp = T.noise(T.vmath('SCALE', drift, scale=1.0 / 25.0), 1.0, 4.0, 0.6)
    a = T.smooth(0.0, 45.0, T.add(clear, T.mul(T.sub(wisp, 0.5), 50.0)))
    tr = T.n('ShaderNodeBsdfTransparent')
    mix = T.n('ShaderNodeMixShader')
    T.link(a, mix.inputs[0])
    T.link(tr.outputs[0], mix.inputs[1])
    T.link(fogged, mix.inputs[2])
    T.link(mix.outputs[0], out.inputs['Surface'])
    return m


def dawn_smoke_material(look):
    """Morning smoke from the night's beacons: thin, drifting, sunlit gold on the sun side."""
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
    sdens = T.attr('sdens', 'OBJECT')
    t = T.add(tm, seed)
    xc = T.mul(lean, T.math('POWER', Y, 1.3))
    w = T.add(0.07, T.mul(Y, 0.30))
    n1 = T.noise(T.comb(T.mul(X, 1.4), T.sub(T.mul(Y, 2.0), T.mul(t, 0.12)), seed), 1.0, 4.0, 0.6,
                 dims='4D', w=T.mul(t, 0.08))
    n2 = T.noise(T.comb(T.mul(X, 4.0), T.sub(T.mul(Y, 5.0), T.mul(t, 0.3)), T.add(seed, 3.0)),
                 1.0, 3.0, 0.55, dims='4D', w=T.mul(t, 0.2))
    xd = T.add(T.sub(X, xc), T.mul(T.sub(n1, 0.5), T.mul(Y, 0.7)))
    prof = T.math('EXPONENT', T.mul(T.mul(T.div(xd, w), T.div(xd, w)), -1.0))
    bill = T.smooth(0.34, 0.72, T.add(T.add(T.mul(n1, 0.7), T.mul(n2, 0.35)), T.mul(T.sub(1.0, Y), 0.15)))
    dens = T.mul(T.mul(prof, bill), T.mul(T.smooth(0.0, 0.05, Y), T.sub(1.0, T.smooth(0.45, 1.0, Y))))
    alpha = T.math('MINIMUM', T.mul(dens, sdens), 0.9)
    geo = T.n('ShaderNodeNewGeometry')
    I = geo.outputs['Incoming']
    sd, el = _sun_dir(T)
    ss = _sun_strength(T, el)
    cth = T.vmath('DOT_PRODUCT', T.vmath('SCALE', I, scale=-1.0), sd)
    g = 0.7
    hg = T.div(1 - g * g, T.math('POWER', T.sub(1 + g * g, T.mul(2 * g, cth)), 1.5))
    lit = T.mul(T.add(0.35, T.mul(hg, 0.25)), T.mul(ss, 0.9))
    col = T.vmath('ADD', T.vmath('SCALE', look.sun_col, scale=lit),
                  T.vmath('SCALE', look.fill_col, scale=0.10))
    col = T.vmath('MULTIPLY', col, (0.55, 0.52, 0.5))
    em = T.n('ShaderNodeEmission', Color=col, Strength=1.0)
    tr = T.n('ShaderNodeBsdfTransparent')
    mix = T.n('ShaderNodeMixShader')
    T.link(alpha, mix.inputs[0])
    T.link(tr.outputs[0], mix.inputs[1])
    T.link(em.outputs[0], mix.inputs[2])
    T.link(SC.use_fog(T, mix.outputs[0]), out.inputs['Surface'])
    return m


def eagle_material(look):
    m = bpy.data.materials.new('Eagle')
    m.use_nodes = True
    nt = m.node_tree
    nt.nodes.clear()
    T = Tree(nt)
    out = T.n('ShaderNodeOutputMaterial')
    bsdf = T.n('ShaderNodeBsdfPrincipled', Roughness=0.85)
    bsdf.inputs['Base Color'].default_value = (0.020, 0.016, 0.013, 1)
    # backlit feather edges glow a little with the sun
    lw = T.n('ShaderNodeLayerWeight', Blend=0.35)
    geo = T.n('ShaderNodeNewGeometry')
    sd, el = _sun_dir(T)
    cth = T.vmath('DOT_PRODUCT', T.vmath('SCALE', geo.outputs['Incoming'], scale=-1.0), sd)
    back = T.smooth(0.2, 0.95, cth)
    rim = T.mul(T.mul(T.math('POWER', lw.outputs['Facing'], 3.0), back), 0.9)
    em = T.n('ShaderNodeEmission', Color=(1.0, 0.72, 0.42, 1))
    T.link(rim, em.inputs['Strength'])
    add = T.n('ShaderNodeAddShader')
    T.link(bsdf.outputs[0], add.inputs[0])
    T.link(em.outputs[0], add.inputs[1])
    T.link(SC.use_fog(T, add.outputs[0]), out.inputs['Surface'])
    return m


# ------------------------------------------------------------------- eagles ---
def eagle_rest(span):
    """Rest-pose eagle: vertices (x right, y forward, z up) in metres, faces, and per-vertex
    wing weights (0 body, >0 inner wing, >1 outer wing; sign = side)."""
    S = span
    V, F, Wt = [], [], []

    def v(x, y, z, w=0.0):
        V.append((x * S, y * S, z * S))
        Wt.append(w)
        return len(V) - 1

    # body: a spindle of rings
    rings = [(-0.14, 0.018), (-0.09, 0.034), (-0.03, 0.045), (0.03, 0.046), (0.08, 0.038),
             (0.12, 0.028), (0.15, 0.020)]
    nseg = 8
    idx = []
    for y, r in rings:
        ring = []
        for k in range(nseg):
            a = 2 * math.pi * k / nseg
            ring.append(v(r * math.cos(a), y, 0.8 * r * math.sin(a)))
        idx.append(ring)
    for i in range(len(idx) - 1):
        for k in range(nseg):
            a, b = idx[i][k], idx[i][(k + 1) % nseg]
            c, d = idx[i + 1][(k + 1) % nseg], idx[i + 1][k]
            F.append((a, b, c, d))
    # head + hooked beak
    hc = v(0.0, 0.185, 0.004)
    last = idx[-1]
    for k in range(nseg):
        F.append((last[k], last[(k + 1) % nseg], hc))
    bk = v(0.0, 0.205, -0.006)
    F.append((hc, idx[-1][0], bk))
    # tail: a fan
    t0 = v(-0.028, -0.13, 0.0)
    t1 = v(0.028, -0.13, 0.0)
    t2 = v(0.065, -0.245, 0.0)
    t3 = v(0.0, -0.26, 0.0)
    t4 = v(-0.065, -0.245, 0.0)
    F.append((t0, t1, t2, t3))
    F.append((t0, t3, t4))
    # wings: broad inner wing to the wrist, then 6 fingered primaries
    for side in (1, -1):
        s = side
        inner = [v(s * 0.035, 0.035, 0.005, s * 0.2), v(s * 0.16, 0.055, 0.004, s * 0.6),
                 v(s * 0.29, 0.060, 0.0, s * 1.0)]
        trail = [v(s * 0.035, -0.07, 0.0, s * 0.2), v(s * 0.16, -0.095, 0.0, s * 0.6),
                 v(s * 0.29, -0.080, 0.0, s * 1.0)]
        for i in range(2):
            a, b, c, d = inner[i], inner[i + 1], trail[i + 1], trail[i]
            F.append((a, b, c, d) if s > 0 else (d, c, b, a))
        # hand: 6 primaries fanning from the wrist
        wx0, wy_top, wy_bot = 0.29, 0.060, -0.080
        n = 6
        prev_top = inner[2]
        for k in range(n):
            f0 = k / n
            f1 = (k + 1) / n
            ya = wy_top + (wy_bot - wy_top) * (f0 + 0.04)
            yb = wy_top + (wy_bot - wy_top) * (f1 - 0.04)
            ang = math.radians(8 - 11 * (k + 0.5))        # fan
            L = 0.18 + 0.05 * math.sin(math.pi * (k + 0.5) / n)
            ca, sa = math.cos(ang), math.sin(ang)
            r0 = v(s * wx0, ya, 0.0, s * 1.2)
            r1 = v(s * wx0, yb, 0.0, s * 1.2)
            tip_w = 0.010
            tx = wx0 + L * ca
            ty = (ya + yb) / 2 + L * sa
            p1 = v(s * (tx - 0.02), ty + tip_w, 0.0, s * 2.0)
            p2 = v(s * tx, ty - tip_w * 0.3, 0.0, s * 2.0)
            q = (r0, p1, p2, r1)
            F.append(q if s > 0 else q[::-1])
    return V, F, Wt


def eagle_pose(rest, Wt, t, period, phase, glide, span):
    """Flap: inner wing rotates about the body axis, the hand lags and swings further.
    Returns posed vertex list (local)."""
    cyc = (t / period + phase) % 1.0
    flapping = 1.0 - glide * (0.5 + 0.5 * math.sin(2 * math.pi * (t / (period * 3.3) + phase)))
    a_in = math.radians(26) * math.sin(2 * math.pi * cyc) * flapping + math.radians(6)
    a_out = math.radians(38) * math.sin(2 * math.pi * cyc - 0.9) * flapping + math.radians(4)
    fold = 0.10 * max(0.0, math.sin(2 * math.pi * cyc + 1.2)) * flapping   # hand tucks on upstroke
    out = []
    for (x, y, z), w in zip(rest, Wt):
        aw = abs(w)
        if aw == 0.0:
            out.append((x, y, z))
            continue
        s = 1.0 if w > 0 else -1.0
        # inner rotation about the shoulder (x = 0.035 S), fraction by weight
        xs = 0.035 * span * s
        k_in = min(aw, 1.0)
        ang = a_in * k_in
        dx = x - xs
        x1 = xs + dx * math.cos(ang)
        z1 = z + abs(dx) * math.sin(ang)
        if aw > 1.0:
            # the hand: rotate further about the wrist
            xw = 0.29 * span * s
            wx = xs + (xw - xs) * math.cos(a_in)
            wz = abs(xw - xs) * math.sin(a_in)
            dxh = x - xw
            ang2 = a_in + a_out * min(aw - 1.0, 1.0)
            L = abs(dxh) * (1.0 - fold)
            x1 = wx + s * L * math.cos(ang2)
            z1 = wz + L * math.sin(ang2)
            y = y - fold * 0.3 * span * (aw - 1.0)
        out.append((x1, y, z1))
    return out
