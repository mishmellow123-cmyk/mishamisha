"""Builds the MOUNTAIN world in Blender (EEVEE Next): sky, land, cloud sea, fog, beacons.

Lighting model
* The moon (night) / the sun (dawn) is EMISSION-CODED in every material, using per-vertex
  visibility baked in the venv (long terrain shadows are exact at any distance and cost nothing).
* The world probe gives the sky's ambient light; fires are real point lights (shadowed near).
* Aerial perspective is computed per pixel in each material (exact exponential height-fog
  integral along the view ray) so it anti-aliases with the geometry.
"""
import math
import os
import sys

import bpy
from mathutils import Vector

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nodes import Tree


def srgb(h):
    h = h.lstrip('#')
    c = [int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4)]
    return tuple(x / 12.92 if x <= 0.04045 else ((x + 0.055) / 1.055) ** 2.4 for x in c)


def rgba(c, a=1.0):
    return (c[0], c[1], c[2], a)


def dir_from(az_deg, el_deg):
    az, el = math.radians(az_deg), math.radians(el_deg)
    return Vector((math.sin(az) * math.cos(el), math.cos(az) * math.cos(el), math.sin(el)))


class Look:
    """Per-shot look parameters (night run / dawn)."""

    def __init__(self, **kw):
        self.moon_dir = dir_from(-30, 16)
        self.moon_col = (0.62, 0.76, 1.0)
        self.moon_E = 0.55                   # moonlight irradiance scale (emission-coded)
        self.moon_disc = 0.0085              # angular radius (rad), ~2x real
        self.moon_I = 30.0
        self.zenith = srgb('#070B1C')
        self.mid = srgb('#131D3B')
        self.horizon = srgb('#27335E')
        self.sky_gain = 1.0
        self.amb = 1.0                       # world lighting gain
        self.mist_d0 = 2.2e-3                # cloud-top mist extinction at z = 0 (1/m)
        self.mist_H = 45.0                   # its scale height (m)
        self.haze_d0 = 3.2e-5                # regional haze at z = 0
        self.haze_H = 1600.0
        self.fog_col = srgb('#27335E')       # haze in-scatter (matches the sky's horizon)
        self.haze_gain = 1.25
        self.mist_gain = 0.42                # mist in-scatter ~ moonlit cloud x this
        self.cloud_gain = 0.66
        self.fog_moon = 0.10                 # forward-scatter glow toward the moon
        self.star_I = 1.0
        self.sky_fill = 0.045                # dim sky + cloud-bounce fill on the land
        self.fill_col = (0.55, 0.68, 1.0)
        self.sun = False
        for k, v in kw.items():
            setattr(self, k, v)


# ---------------------------------------------------------------------- fog ---
def _layer_tau(T, L, pz, cz, d0, H):
    """Optical depth of sigma(z) = d0 exp(-z/H) along a ray of length L from height cz to pz."""
    dz = T.sub(pz, cz)
    x = T.div(dz, H)
    # f(x) = (1 - exp(-x)) / x, stable near 0 (|x| floored at 1e-4 with its sign kept)
    sg = T.math('SIGN', x)
    sg = T.add(sg, T.sub(1.0, T.math('ABSOLUTE', sg)))          # 0 -> +1
    xs = T.mul(T.math('MAXIMUM', T.math('ABSOLUTE', x), 1e-4), sg)
    f = T.div(T.sub(1.0, T.math('EXPONENT', T.mul(xs, -1.0))), xs)
    ec = T.math('EXPONENT', T.div(cz, -H))
    return T.mul(T.mul(T.mul(d0, L), ec), f)


def fog_group(look):
    """Node group: Shader -> fogged Shader. Two exponential layers integrated exactly along the
    camera ray: a dense mist hugging the cloud top (in-scatter = moonlit cloud) and a regional
    haze (in-scatter = sky horizon colour + forward scatter toward the moon)."""
    ng = bpy.data.node_groups.new('AerialFog', 'ShaderNodeTreeGroup' if False else 'ShaderNodeTree')
    ng.interface.new_socket('Shader', in_out='INPUT', socket_type='NodeSocketShader')
    ng.interface.new_socket('Shader', in_out='OUTPUT', socket_type='NodeSocketShader')
    ng.interface.new_socket('Boost', in_out='INPUT', socket_type='NodeSocketFloat')
    T = Tree(ng)
    gi = T.n('NodeGroupInput')
    go = T.n('NodeGroupOutput')
    geo = T.n('ShaderNodeNewGeometry')
    cam = T.n('ShaderNodeCameraData')
    P = geo.outputs['Position']
    I = geo.outputs['Incoming']            # unit vector toward the camera
    L = cam.outputs['View Distance']
    pz = T.sep(P)[2]
    iz = T.sep(I)[2]
    cz = T.madd(iz, L, pz)                 # camera z
    boost = T.add(1.0, gi.outputs['Boost'])
    tm = T.mul(_layer_tau(T, L, pz, cz, look.mist_d0, look.mist_H), boost)
    th = T.mul(_layer_tau(T, L, pz, cz, look.haze_d0, look.haze_H), boost)
    Tm = T.math('EXPONENT', T.mul(tm, -1.0))
    Th = T.math('EXPONENT', T.mul(th, -1.0))
    vd = T.vmath('SCALE', I, scale=-1.0)                       # camera -> point
    cth = T.vmath('DOT_PRODUCT', vd, tuple(look.moon_dir))
    g = 0.72
    den = T.math('POWER', T.sub(1 + g * g, T.mul(2 * g, cth)), 1.5)
    hg = T.div((1 - g * g), den)
    # mist: lit like the cloud deck (wrap + forward scatter)
    mcol = T.vmath('SCALE', tuple(look.moon_col),
                   scale=T.mul(T.add(0.55, T.mul(hg, 0.12)), look.moon_E * look.mist_gain))
    mcol = T.vmath('ADD', mcol, T.vmath('SCALE', tuple(look.fog_col), scale=0.6))
    hcol = T.vmath('ADD', T.vmath('SCALE', tuple(look.fog_col), scale=look.haze_gain),
                   T.vmath('SCALE', tuple(look.moon_col), scale=T.mul(hg, look.fog_moon)))
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


def use_fog(T, shader_socket, boost=0.0):
    g = T.nt.nodes.new('ShaderNodeGroup')
    g.node_tree = bpy.data.node_groups['AerialFog']
    T.link(shader_socket, g.inputs['Shader'])
    g.inputs['Boost'].default_value = boost
    return g.outputs['Shader']


# ---------------------------------------------------------------------- sky ---
def build_world(look, name='Sky'):
    w = bpy.data.worlds.new(name)
    bpy.context.scene.world = w
    w.use_nodes = True
    nt = w.node_tree
    nt.nodes.clear()
    T = Tree(nt)
    out = T.n('ShaderNodeOutputWorld')
    tc = T.n('ShaderNodeTexCoord')
    D = T.vmath('NORMALIZE', tc.outputs['Generated'])
    e = T.sep(D)[2]
    # gradient
    c1 = T.mixc(T.smooth(-0.02, 0.22, e), rgba(look.horizon), rgba(look.mid))
    c2 = T.mixc(T.smooth(0.18, 0.95, e), c1, rgba(look.zenith))
    cth = T.vmath('DOT_PRODUCT', D, tuple(look.moon_dir))
    # broad moon-side brightening of the lower sky + halo
    broad = T.mul(T.math('EXPONENT', T.mul(T.sub(cth, 1.0), 2.2)), T.sub(1.0, T.smooth(0.0, 0.6, e)))
    halo = T.math('EXPONENT', T.mul(T.sub(cth, 1.0), 900.0))
    halo2 = T.math('EXPONENT', T.mul(T.sub(cth, 1.0), 90.0))
    glowf = T.add(T.add(T.mul(broad, 0.030), T.mul(halo, 0.9)), T.mul(halo2, 0.08))
    sky = T.vmath('ADD', c2, T.vmath('SCALE', tuple(look.moon_col), scale=glowf))
    sky = T.vmath('SCALE', sky, scale=look.sky_gain)
    # ---- stars (camera rays only): three voronoi layers
    ext = T.smooth(0.0, 0.22, e)                               # horizon extinction
    # milky way band
    mw_n = Vector((0.55, -0.25, 0.80)).normalized()
    b = T.vmath('DOT_PRODUCT', D, tuple(mw_n))
    band = T.math('EXPONENT', T.mul(T.mul(b, b), -28.0))
    mwn = T.noise(T.vmath('SCALE', D, scale=6.0), 1.0, 6.0, 0.62)
    mwn2 = T.noise(T.vmath('SCALE', D, scale=22.0), 1.0, 4.0, 0.6)
    lanes = T.smooth(0.42, 0.62, mwn2)                         # dark dust lanes
    mw = T.mul(T.mul(band, T.smooth(0.38, 0.72, mwn)), T.sub(1.0, T.mul(lanes, 0.6)))
    stars_total = None
    for k, (scale, dens, bright, psf) in enumerate(((900.0, 0.010, 10.0, 0.16),
                                                    (380.0, 0.012, 30.0, 0.12),
                                                    (150.0, 0.010, 110.0, 0.10))):
        vor = T.n('ShaderNodeTexVoronoi', Scale=scale)
        vor.voronoi_dimensions = '3D'
        T.link(D, vor.inputs['Vector'])
        col = T.sep(vor.outputs['Color'])
        dist = vor.outputs['Distance']
        thr = T.add(dens, T.mul(mw, dens * 3.0)) if k == 0 else dens
        on = T.math('LESS_THAN', col[0], thr)
        mag = T.math('POWER', col[1], 6.0)
        psfv = T.math('EXPONENT', T.mul(T.mul(T.div(dist, psf), T.div(dist, psf)), -1.0))
        s = T.mul(T.mul(on, psfv), T.mul(mag, bright * look.star_I))
        # tint: warm or cool by the cell's third random
        tint = T.mixc(col[2], (1.0, 0.85, 0.7, 1), (0.75, 0.85, 1.0, 1))
        sc = T.vmath('SCALE', tint, scale=s)
        stars_total = sc if stars_total is None else T.vmath('ADD', stars_total, sc)
    stars_total = T.vmath('SCALE', stars_total, scale=ext)
    mwc = T.vmath('SCALE', (0.55, 0.62, 0.8), scale=T.mul(mw, 0.010 * look.star_I))
    # moon disc
    cosr = math.cos(look.moon_disc)
    disc = T.smooth(cosr - 0.0000009, cosr + 0.0000009, cth)
    mnz = T.noise(T.vmath('SCALE', D, scale=900.0), 1.0, 3.0, 0.55)
    maria = T.maprange(mnz, 0.35, 0.65, 0.78, 1.0)
    rel = T.div(T.math('ARCCOSINE', T.math('MINIMUM', cth, 1.0)), look.moon_disc)
    limb = T.math('POWER', T.math('MAXIMUM', T.sub(1.0, T.mul(rel, rel)), 0.0), 0.25)
    moon = T.vmath('SCALE', (1.0, 0.98, 0.93), scale=T.mul(T.mul(disc, T.mul(maria, limb)), look.moon_I))
    night_extra = T.vmath('ADD', T.vmath('ADD', stars_total, mwc), moon)
    cam_sky = T.vmath('ADD', sky, T.vmath('SCALE', night_extra, scale=0.0 if look.sun else 1.0))
    # lighting (non-camera rays): smooth, dim
    amb = T.vmath('SCALE', c2, scale=look.amb)
    lp = T.n('ShaderNodeLightPath')
    sel = T.mixc(lp.outputs['Is Camera Ray'], amb, cam_sky)
    bg = T.n('ShaderNodeBackground', Color=sel, Strength=1.0)
    T.link(bg.outputs[0], out.inputs['Surface'])
    return w


# --------------------------------------------------------------------- land ---
def land_material(look, name='Land'):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    nt.nodes.clear()
    T = Tree(nt)
    out = T.n('ShaderNodeOutputMaterial')
    geo = T.n('ShaderNodeNewGeometry')
    P = geo.outputs['Position']
    N = geo.outputs['Normal']
    moonv = T.attr('moon')
    snow_a = T.attr('snow')
    ao = T.attr('ao')
    Pz = T.sep(P)[2]
    Nz = T.sep(N)[2]
    # fall-line frame: contour direction Tc = N x up; u runs along the contour
    Tc = T.vmath('NORMALIZE', T.vmath('CROSS_PRODUCT', N, (0.0, 0.0, 1.0)))
    u = T.vmath('DOT_PRODUCT', P, Tc)
    # couloirs / flutes: noise that varies fast along the contour, slowly down the fall line
    fl_c = T.comb(T.div(u, 9.0), T.div(Pz, 70.0), 0.0)
    flute = T.noise(fl_c, 1.0, 2.0, 0.55)
    fl_c2 = T.comb(T.div(u, 2.2), T.div(Pz, 16.0), 7.0)
    flute2 = T.noise(fl_c2, 1.0, 3.0, 0.5)
    # horizontal rock strata / ledges (warped)
    lw = T.noise(T.vmath('SCALE', P, scale=1.0 / 90.0), 1.0, 2.0, 0.5)
    ledge = T.noise(T.comb(T.div(u, 160.0), T.add(T.div(Pz, 7.0), T.mul(lw, 6.0)), 3.0),
                    1.0, 2.0, 0.5)
    # --- rock micro-relief for the bump (subtle)
    r1 = T.noise(P, 0.6, 3.0, 0.55)
    rock_h = T.add(T.mul(r1, 0.35), T.add(T.mul(flute, 0.35), T.mul(ledge, 0.35)))
    bump_r = T.n('ShaderNodeBump', Strength=0.45, Distance=1.0)
    T.link(rock_h, bump_r.inputs['Height'])
    T.link(N, bump_r.inputs['Normal'])
    sn = T.noise(T.vmath('MULTIPLY', P, (0.08, 0.08, 0.2)), 1.0, 4.0, 0.5)
    sas = T.noise(T.vmath('MULTIPLY', P, (0.9, 0.25, 0.9)), 1.0, 3.0, 0.5)     # sastrugi
    sn = T.add(T.mul(sn, 0.7), T.mul(sas, 0.3))
    bump_s = T.n('ShaderNodeBump', Strength=0.22, Distance=0.8)
    T.link(sn, bump_s.inputs['Height'])
    T.link(N, bump_s.inputs['Normal'])
    # --- snow cover: the baked structure (slope + gully concavity) decides; the shader only
    # adds ledge lines on steep ground and a soft, organic breakup of the edges
    edge = T.noise(T.vmath('SCALE', P, scale=1.0 / 7.0), 1.0, 4.0, 0.6)
    edge2 = T.noise(T.vmath('SCALE', P, scale=1.0 / 45.0), 1.0, 3.0, 0.55)
    s = T.add(snow_a, T.mul(T.sub(edge, 0.5), 0.22))
    s = T.add(s, T.mul(T.sub(edge2, 0.5), 0.28))
    steepness = T.smooth(0.85, 0.5, Nz)
    s = T.add(s, T.mul(T.mul(T.smooth(0.54, 0.64, ledge), steepness), 0.55))
    snow = T.smooth(0.40, 0.56, s)
    # albedo
    strata_c = T.smooth(0.35, 0.65, ledge)
    rock_c = T.mixc(strata_c, (0.070, 0.066, 0.062, 1), (0.13, 0.12, 0.11, 1))
    rock_c = T.mixc(T.smooth(0.55, 0.75, r1), rock_c, (0.035, 0.034, 0.036, 1))
    snow_c = T.mixc(sn, (0.76, 0.80, 0.87, 1), (0.86, 0.89, 0.94, 1))
    alb = T.mixc(snow, rock_c, snow_c)
    nrm = T.vmath('NORMALIZE', T.mixc(snow, bump_r.outputs['Normal'], bump_s.outputs['Normal']))
    # --- moonlight (emission-coded): lambert with a little wrap on snow
    ndl = T.vmath('DOT_PRODUCT', nrm, tuple(look.moon_dir))
    wrap = T.mixf(snow, 0.0, 0.12)
    lam = T.math('MAXIMUM', T.div(T.add(ndl, wrap), T.add(1.0, wrap)), 0.0)
    E = T.mul(T.mul(lam, moonv), look.moon_E)
    moonlit = T.vmath('MULTIPLY', alb, T.vmath('SCALE', tuple(look.moon_col), scale=E))
    # sky fill: the whole moonlit sky + cloud sea bounce light the shadowed snow a dim blue
    up = T.add(T.mul(Nz, 0.5), 0.5)
    fillE = T.mul(T.mul(T.add(T.mul(up, 0.6), 0.4), T.add(T.mul(ao, 0.7), 0.3)), look.sky_fill)
    moonlit = T.vmath('ADD', moonlit, T.vmath('MULTIPLY', alb, T.vmath('SCALE', tuple(look.fill_col), scale=fillE)))
    em = T.n('ShaderNodeEmission', Color=moonlit, Strength=1.0)
    bsdf = T.n('ShaderNodeBsdfPrincipled', Roughness=0.62)
    T.link(T.vmath('SCALE', alb, scale=T.add(T.mul(ao, 0.8), 0.2)), bsdf.inputs['Base Color'])
    T.link(nrm, bsdf.inputs['Normal'])
    bsdf.inputs['Specular IOR Level'].default_value = 0.25
    add = T.n('ShaderNodeAddShader')
    T.link(bsdf.outputs[0], add.inputs[0])
    T.link(em.outputs[0], add.inputs[1])
    T.link(use_fog(T, add.outputs[0]), out.inputs['Surface'])
    return m


# -------------------------------------------------------------------- cloud ---
def cloud_material(look, name='Cloud'):
    m = bpy.data.materials.new(name)
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
    moonv = T.attr('moon')
    clear = T.attr('clear')
    tm = T.attr('ld_time', 'VIEW_LAYER')
    # slow drift + boiling detail
    drift = T.vmath('ADD', P, T.comb(T.mul(tm, -3.0), T.mul(tm, -1.2), 0.0))
    d1 = T.noise(T.vmath('SCALE', drift, scale=1.0 / 70.0), 1.0, 4.0, 0.55, dims='4D',
                 w=T.mul(tm, 0.03))
    d2 = T.noise(T.vmath('SCALE', drift, scale=1.0 / 18.0), 1.0, 2.0, 0.5)
    hb = T.add(d1, T.mul(d2, 0.12))
    bump = T.n('ShaderNodeBump', Strength=0.22, Distance=6.0)
    T.link(hb, bump.inputs['Height'])
    T.link(N, bump.inputs['Normal'])
    nrm = bump.outputs['Normal']
    ndl = T.vmath('DOT_PRODUCT', nrm, tuple(look.moon_dir))
    wrap = T.math('MAXIMUM', T.div(T.add(ndl, 0.55), 1.55), 0.0)
    vd = T.vmath('SCALE', I, scale=-1.0)
    cth = T.vmath('DOT_PRODUCT', vd, tuple(look.moon_dir))
    g = 0.55
    hg = T.div(1 - g * g, T.math('POWER', T.sub(1 + g * g, T.mul(2 * g, cth)), 1.5))
    lightf = T.mul(T.mul(T.add(T.mul(wrap, 0.8), T.mul(hg, 0.22)), moonv), look.moon_E * look.cloud_gain)
    # hollows of the deck are darker (baked concavity) + fine bump shading
    hol = T.attr('hollow')
    cav = T.mul(T.maprange(hb, 0.3, 0.7, 0.8, 1.0), T.maprange(hol, -0.2, 1.0, 1.0, 0.55))
    lightf = T.mul(lightf, cav)
    alb = (0.74, 0.80, 0.90, 1)
    col = T.vmath('SCALE', tuple(look.moon_col), scale=lightf)
    em = T.n('ShaderNodeEmission', Color=T.vmath('MULTIPLY', col, alb), Strength=1.0)
    dif = T.n('ShaderNodeBsdfDiffuse', Color=alb)
    T.link(nrm, dif.inputs['Normal'])
    add = T.n('ShaderNodeAddShader')
    T.link(dif.outputs[0], add.inputs[0])
    T.link(em.outputs[0], add.inputs[1])
    fogged = use_fog(T, add.outputs[0])
    # thin, wispy where the land rises just beneath the cloud top
    wisp = T.noise(T.vmath('SCALE', drift, scale=1.0 / 25.0), 1.0, 4.0, 0.6)
    a = T.smooth(0.0, 45.0, T.add(clear, T.mul(T.sub(wisp, 0.5), 50.0)))
    tr = T.n('ShaderNodeBsdfTransparent')
    mix = T.n('ShaderNodeMixShader')
    T.link(a, mix.inputs[0])
    T.link(tr.outputs[0], mix.inputs[1])
    T.link(fogged, mix.inputs[2])
    T.link(mix.outputs[0], out.inputs['Surface'])
    return m


# ------------------------------------------------------------------- render ---
def render_settings(scale=1.0, samples=32, motion_blur=True):
    sc = bpy.context.scene
    sc.render.engine = 'BLENDER_EEVEE_NEXT'
    sc.render.resolution_x = 1920
    sc.render.resolution_y = 804
    sc.render.resolution_percentage = int(round(scale * 100))
    sc.render.fps = 24
    ee = sc.eevee
    ee.taa_render_samples = samples
    ee.use_shadows = True
    ee.shadow_pool_size = '128'
    ee.use_raytracing = False
    ee.use_volumetric_shadows = False
    ee.clamp_surface_indirect = 10.0
    sc.render.use_motion_blur = motion_blur
    sc.render.motion_blur_shutter = 0.5
    ee.motion_blur_steps = 1
    sc.view_settings.view_transform = 'Standard'
    sc.view_settings.look = 'None'
    sc.view_settings.exposure = 0.0
    sc.view_settings.gamma = 1.0
    sc.display_settings.display_device = 'sRGB'
    sc.render.image_settings.file_format = 'OPEN_EXR'
    sc.render.image_settings.color_depth = '16'
    sc.render.image_settings.color_mode = 'RGB'
    sc.render.image_settings.exr_codec = 'ZIP'
    sc.render.film_transparent = False
    sc.render.filter_size = 1.5
