"""MONTAGE-3D core (runs inside Blender 4.5, no numpy): scene/render setup, camera rig + export for
post, keyframe helpers, mesh helpers, GN instancing, shader-fog atmosphere, procedural night sky.

World: metres, Z up. Cameras look along +Y at yaw 0; +yaw turns right (toward +X), +pitch looks up.
"""
import json
import math
import os

import bpy
from mathutils import Matrix, Quaternion, Vector

from .nodes import NB, new_material

FPS = 24


# ------------------------------------------------------------------ colour ---

def s2l(c):
    c = max(0.0, min(1.0, c))
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def hexlin(h, gain=1.0):
    h = h.lstrip('#')
    return tuple(s2l(int(h[i:i + 2], 16) / 255.0) * gain for i in (0, 2, 4))


def lerp3(a, b, t):
    return tuple(x + (y - x) * t for x, y in zip(a, b))


FIRE_LIGHT = (1.0, 0.50, 0.17)          # same as the montage toolkit (linear)
MOON = hexlin('#9DB4D9')


# ------------------------------------------------------------------- scene ---

def reset():
    for coll in (bpy.data.objects, bpy.data.meshes, bpy.data.materials, bpy.data.lights, bpy.data.cameras,
                 bpy.data.node_groups, bpy.data.images, bpy.data.curves, bpy.data.metaballs,
                 bpy.data.armatures, bpy.data.worlds, bpy.data.textures):
        for d in list(coll):
            try:
                coll.remove(d)
            except Exception:
                pass
    for c in list(bpy.data.collections):
        bpy.data.collections.remove(c)
    prefs = bpy.context.preferences.edit
    prefs.keyframe_new_interpolation_type = 'LINEAR'
    prefs.keyframe_new_handle_type = 'VECTOR'
    return bpy.context.scene


def setup_render(scale=1.0, samples=64, vol=None, mblur=False, shutter=0.5, filter_px=1.5, raytrace=True,
                 shadow_pool='512', light_threshold=0.001, exr_half=False):
    sc = bpy.context.scene
    r = sc.render
    r.engine = 'BLENDER_EEVEE_NEXT'
    r.resolution_x = 1920
    r.resolution_y = 804
    r.resolution_percentage = max(1, int(round(scale * 100)))
    r.fps = FPS
    r.filter_size = filter_px
    r.use_compositing = False
    r.use_sequencer = False
    r.film_transparent = False
    im = r.image_settings
    im.file_format = 'OPEN_EXR_MULTILAYER'
    im.color_depth = '16' if exr_half else '32'
    im.exr_codec = 'ZIP'
    sc.view_settings.view_transform = 'Standard'
    sc.view_settings.look = 'None'
    sc.view_settings.exposure = 0.0
    sc.view_settings.gamma = 1.0
    ee = sc.eevee
    ee.taa_render_samples = samples
    ee.use_shadows = True
    ee.shadow_pool_size = shadow_pool
    ee.shadow_ray_count = 2
    ee.shadow_step_count = 8
    ee.light_threshold = light_threshold
    ee.use_raytracing = raytrace
    if raytrace:
        ee.ray_tracing_method = 'SCREEN'
        ro = ee.ray_tracing_options
        ro.resolution_scale = '2'
        ro.trace_max_roughness = 0.6
        ee.use_fast_gi = True
        ee.fast_gi_method = 'GLOBAL_ILLUMINATION'
    if vol:
        ee.use_volume_custom_range = True
        ee.volumetric_start = vol.get('start', 0.5)
        ee.volumetric_end = vol.get('end', 100.0)
        ee.volumetric_tile_size = str(vol.get('tile', 8))
        ee.volumetric_samples = vol.get('samples', 64)
        ee.volumetric_sample_distribution = vol.get('dist', 0.8)
        ee.use_volumetric_shadows = vol.get('shadows', True)
        ee.volumetric_shadow_samples = vol.get('shadow_samples', 16)
        ee.volumetric_light_clamp = 0.0
    r.use_motion_blur = mblur
    if mblur:
        r.motion_blur_shutter = shutter
        ee.motion_blur_steps = 1
        ee.motion_blur_max = 48
    vl = sc.view_layers[0]
    vl.use_pass_z = True
    return sc


# ------------------------------------------------------------------ camera ---

def look_quat(yaw_deg, pitch_deg, roll_deg=0.0):
    """Camera orientation: yaw 0 looks along +Y, +yaw turns toward +X, +pitch looks up."""
    y = math.radians(yaw_deg)
    p = math.radians(pitch_deg)
    fwd = Vector((math.sin(y) * math.cos(p), math.cos(y) * math.cos(p), math.sin(p)))
    q = fwd.to_track_quat('-Z', 'Y')
    if roll_deg:
        q = Quaternion(fwd, math.radians(roll_deg)) @ q
    return q


def make_camera(name='CAM', hfov=50.0, clip=(0.1, 20000.0)):
    cd = bpy.data.cameras.new(name)
    cd.sensor_fit = 'HORIZONTAL'
    cd.sensor_width = 36.0
    cd.lens = 18.0 / math.tan(math.radians(hfov) / 2)
    cd.clip_start, cd.clip_end = clip
    ob = bpy.data.objects.new(name, cd)
    bpy.context.scene.collection.objects.link(ob)
    ob.rotation_mode = 'QUATERNION'
    bpy.context.scene.camera = ob
    return ob


def key_camera(cam, frame, pos, yaw, pitch, roll=0.0, shift=(0.0, 0.0), lens=None):
    cam.location = Vector(pos)
    cam.rotation_quaternion = look_quat(yaw, pitch, roll)
    cam.keyframe_insert('location', frame=frame)
    cam.keyframe_insert('rotation_quaternion', frame=frame)
    cam.data.shift_x, cam.data.shift_y = shift
    cam.data.keyframe_insert('shift_x', frame=frame)
    cam.data.keyframe_insert('shift_y', frame=frame)
    if lens is not None:
        cam.data.lens = lens
        cam.data.keyframe_insert('lens', frame=frame)


def camera_record(cam):
    """Everything post needs to project world points exactly as Blender does (current frame)."""
    sc = bpy.context.scene
    M = cam.matrix_world
    cd = cam.data
    return dict(M=[list(r) for r in M], lens=cd.lens, sensor=cd.sensor_width, shift_x=cd.shift_x,
                shift_y=cd.shift_y, W=sc.render.resolution_x * sc.render.resolution_percentage // 100,
                H=sc.render.resolution_y * sc.render.resolution_percentage // 100)


# --------------------------------------------------------------- keyframes ---

def key(target, prop, frame, value, index=-1):
    if index >= 0:
        getattr(target, prop)[index] = value
    else:
        setattr(target, prop, value)
    target.keyframe_insert(prop, frame=frame, index=index)


def key_socket(sock, frame, value):
    sock.default_value = value
    sock.keyframe_insert('default_value', frame=frame)


# ------------------------------------------------------------------ meshes ---

def link_obj(ob, coll=None):
    (coll or bpy.context.scene.collection).objects.link(ob)
    return ob


def mesh_obj(name, verts, faces, mat=None, smooth=True, coll=None, edges=()):
    me = bpy.data.meshes.new(name)
    me.from_pydata(verts, list(edges), faces)
    me.validate(clean_customdata=False)
    me.update()
    if smooth and len(me.polygons):
        me.shade_smooth()
    if mat is not None:
        mats = mat if isinstance(mat, (list, tuple)) else [mat]
        for m in mats:
            me.materials.append(m)
    ob = bpy.data.objects.new(name, me)
    return link_obj(ob, coll)


def grid_faces(nu, nv, wrap_u=False):
    """Quad indices for an nu x nv vertex grid (row-major, u fastest)."""
    faces = []
    cols = nu if wrap_u else nu - 1
    for j in range(nv - 1):
        for i in range(cols):
            i2 = (i + 1) % nu
            a = j * nu + i
            faces.append((a, j * nu + i2, (j + 1) * nu + i2, (j + 1) * nu + i))
    return faces


def collection(name, hide=False):
    c = bpy.data.collections.new(name)
    bpy.context.scene.collection.children.link(c)
    if hide:
        # keep instancer sources out of the render: exclude from the view layer
        lc = bpy.context.view_layer.layer_collection.children[name]
        lc.exclude = True
    return c


def instancer(name, points, coll_src, rot=None, scl=None, var=None, mat=None):
    """Instance the objects of collection `coll_src` on `points` [(x,y,z)] via Geometry Nodes.
    rot: [(rx,ry,rz)] euler radians, scl: [(sx,sy,sz)], var: [int] which child to use."""
    n = len(points)
    me = bpy.data.meshes.new(name + '_pts')
    me.vertices.add(n)
    me.vertices.foreach_set('co', [c for p in points for c in p])
    if rot is not None:
        a = me.attributes.new('rot', 'FLOAT_VECTOR', 'POINT')
        a.data.foreach_set('vector', [c for p in rot for c in p])
    if scl is not None:
        a = me.attributes.new('scl', 'FLOAT_VECTOR', 'POINT')
        a.data.foreach_set('vector', [c for p in scl for c in p])
    if var is not None:
        a = me.attributes.new('var', 'INT', 'POINT')
        a.data.foreach_set('value', list(var))
    me.update()
    ob = link_obj(bpy.data.objects.new(name, me))
    ng = bpy.data.node_groups.new(name + '_GN', 'GeometryNodeTree')
    ng.interface.new_socket('Geometry', in_out='INPUT', socket_type='NodeSocketGeometry')
    ng.interface.new_socket('Geometry', in_out='OUTPUT', socket_type='NodeSocketGeometry')
    nt = ng.nodes
    gi = nt.new('NodeGroupInput')
    go = nt.new('NodeGroupOutput')
    iop = nt.new('GeometryNodeInstanceOnPoints')
    ci = nt.new('GeometryNodeCollectionInfo')
    ci.inputs['Collection'].default_value = coll_src
    ci.inputs['Separate Children'].default_value = True
    ci.inputs['Reset Children'].default_value = True
    ci.transform_space = 'ORIGINAL'
    L = ng.links
    L.new(gi.outputs[0], iop.inputs['Points'])
    L.new(ci.outputs[0], iop.inputs['Instance'])
    iop.inputs['Pick Instance'].default_value = True
    if var is not None:
        na = nt.new('GeometryNodeInputNamedAttribute')
        na.data_type = 'INT'
        na.inputs['Name'].default_value = 'var'
        L.new(na.outputs['Attribute'], iop.inputs['Instance Index'])
    if rot is not None:
        na = nt.new('GeometryNodeInputNamedAttribute')
        na.data_type = 'FLOAT_VECTOR'
        na.inputs['Name'].default_value = 'rot'
        L.new(na.outputs['Attribute'], iop.inputs['Rotation'])
    if scl is not None:
        na = nt.new('GeometryNodeInputNamedAttribute')
        na.data_type = 'FLOAT_VECTOR'
        na.inputs['Name'].default_value = 'scl'
        L.new(na.outputs['Attribute'], iop.inputs['Scale'])
    L.new(iop.outputs['Instances'], go.inputs[0])
    mod = ob.modifiers.new('inst', 'NODES')
    mod.node_group = ng
    return ob


# ---------------------------------------------------------------- lights ---

def sun(name, direction, color, strength, angle_deg=0.5, volume=0.0, shadow=True):
    """direction: vector pointing TOWARD the light (e.g. the moon)."""
    ld = bpy.data.lights.new(name, 'SUN')
    ld.color = color
    ld.energy = strength
    ld.angle = math.radians(angle_deg)
    ld.use_shadow = shadow
    ld.volume_factor = volume
    ob = link_obj(bpy.data.objects.new(name, ld))
    ob.rotation_mode = 'QUATERNION'
    ob.rotation_quaternion = (-Vector(direction)).normalized().to_track_quat('-Z', 'Y')
    return ob


def point(name, pos, color=FIRE_LIGHT, power=100.0, radius=0.2, shadow=True, volume=1.0, spec=1.0,
          cutoff=None):
    ld = bpy.data.lights.new(name, 'POINT')
    ld.color = color
    ld.energy = power
    ld.shadow_soft_size = radius
    ld.use_shadow = shadow
    ld.volume_factor = volume
    ld.specular_factor = spec
    if cutoff:
        ld.use_custom_distance = True
        ld.cutoff_distance = cutoff
    ob = link_obj(bpy.data.objects.new(name, ld))
    ob.location = pos
    return ob


# ------------------------------------------------------------ atmosphere ---

def atmos_group(p):
    """Shader-fog node group 'ATMOS' (Shader in -> fogged Shader out, plus transmittance T).

    Density rho(z) = a * exp(-(z - h0)/Hs) + u, integrated exactly along the camera->point segment.
    In-scatter colour = amb * (1 + fwd * max(dot(view, moon),0)^pw) + glow * <exp(-(z-h0)/Hg)>_rho
    (the second term: light from below, e.g. a city's street glow, density-weighted along the ray).
    Optional: h0 modulated by 2D noise at the shaded point (mist-sea tops)."""
    ng = bpy.data.node_groups.new('ATMOS', 'ShaderNodeTree')
    ng.interface.new_socket('Shader', in_out='INPUT', socket_type='NodeSocketShader')
    ng.interface.new_socket('Shader', in_out='OUTPUT', socket_type='NodeSocketShader')
    ng.interface.new_socket('T', in_out='OUTPUT', socket_type='NodeSocketFloat')
    nb = NB(ng)
    gi = nb.n('NodeGroupInput')
    go = nb.n('NodeGroupOutput')
    g = nb.geo()
    cd = nb.camdata()
    d = cd.outputs['View Distance']
    px, py, pz = nb.sep(g.outputs['Position'])
    ix, iy, iz = nb.sep(g.outputs['Incoming'])
    zc = nb.madd(iz, d, pz)                                  # camera height
    h0 = p.get('h0', 0.0)
    if p.get('h0_noise'):
        amp, sc_ = p['h0_noise']
        nz = nb.noise(g.outputs['Position'], scale=1.0 / sc_, detail=3.0, rough=0.55)
        h0 = nb.madd(nb.sub(nz.outputs['Fac'], 0.5), 2.0 * amp, h0)

    def exp_tau(Hs):
        # a * d * exp(-(zc-h0)/Hs) * (1 - exp(-x))/x,  x = (zp - zc)/Hs
        ec = nb.exp(nb.mn(nb.div(nb.sub(h0, zc), Hs), 30.0))
        x = nb.math('MAXIMUM', nb.math('MINIMUM', nb.div(nb.sub(pz, zc), Hs), 30.0), -30.0)
        xs = nb.add(x, 1e-5)
        gg = nb.div(nb.sub(1.0, nb.exp(nb.mul(xs, -1.0))), xs)
        return nb.mul(nb.mul(nb.mul(ec, gg), d), p.get('a', 0.0))

    Hs = p.get('Hs', 100.0)
    tau_e = exp_tau(Hs)
    tau = nb.madd(d, p.get('u', 0.0), tau_e)
    T = nb.exp(nb.mul(tau, -1.0))
    # forward scatter toward the moon
    md = Vector(p.get('moon_dir', (0, 1, 0.3))).normalized()
    cosm = nb.mul(nb.dot(g.outputs['Incoming'], tuple(md)), -1.0)
    ph = nb.madd(nb.pw(nb.mx(cosm, 0.0), p.get('fwd_pow', 8.0)), p.get('fwd', 0.0), 1.0)
    amb = p.get('amb', (0.02, 0.025, 0.04))
    col = nb.colscale(amb, ph)
    if p.get('glow'):
        Hg = p.get('Hg', 60.0)
        Hc = 1.0 / (1.0 / Hs + 1.0 / Hg)
        tau_g = exp_tau(Hc)
        R = nb.div(tau_g, nb.mx(tau, 1e-6))
        col = nb.vadd(col, nb.colscale(p['glow'], R))
    em = nb.emission(col, 1.0)
    fac = nb.sub(1.0, T)
    ms = nb.mixshader(fac, gi.outputs[0], em)
    nb.link(ms, go.inputs[0])
    nb.link(T, go.inputs[1])
    return ng


def fogged(nb, shader, group=None):
    """Wrap a surface shader with the scene's ATMOS group (if present)."""
    group = group or bpy.data.node_groups.get('ATMOS')
    if group is None:
        return shader
    gn = nb.n('ShaderNodeGroup')
    gn.node_tree = group
    nb.link(shader if not isinstance(shader, bpy.types.Node) else shader.outputs[0], gn.inputs[0])
    return gn.outputs[0]


def fog_T(nb, group=None):
    group = group or bpy.data.node_groups.get('ATMOS')
    if group is None:
        return 1.0
    gn = nb.n('ShaderNodeGroup')
    gn.node_tree = group
    return gn.outputs[1]


def surface_mat(name, base, rough=0.8, spec=0.5, bump=None, sheen=0.0, sheen_tint=(1, 1, 1), extra=None,
                fog=True):
    """Simple fogged Principled material. base: colour tuple or socket."""
    m, nb = new_material(name)
    kw = dict(Base_Color=base, Roughness=rough)
    bs = nb.principled(**kw)
    bs.inputs['Specular IOR Level'].default_value = spec
    if sheen:
        bs.inputs['Sheen Weight'].default_value = sheen
        bs.inputs['Sheen Tint'].default_value = tuple(sheen_tint) + (1.0,)
        bs.inputs['Sheen Roughness'].default_value = 0.4
    if bump is not None:
        nb.link(bump, bs.inputs['Normal'])
    if extra:
        extra(nb, bs)
    out = fogged(nb, bs) if fog else bs
    nb.output(surface=out)
    return m


# --------------------------------------------------------------------- sky ---

def sky_world(p):
    """Procedural night sky: palette gradient, moon disk + halo, horizon glow, stars, Milky Way.
    p keys: zenith, horizon, below, moon_dir, moon_r_deg, moon_I, halo(I,w), halo2(I,w),
    hglow(col, k), stars(dict), milky(dict), light_gain (world lighting multiplier)."""
    w = bpy.data.worlds.new('SKY')
    bpy.context.scene.world = w
    w.use_nodes = True
    nb = NB(w.node_tree, clear=True)
    tc = nb.texco()
    D = nb.vmath('NORMALIZE', tc.outputs['Generated'])
    dx, dy, dz = nb.sep(D)
    zen = p.get('zenith', hexlin('#070B1C'))
    hor = p.get('horizon', hexlin('#27335E'))
    below = p.get('below', tuple(c * 0.6 for c in hor))
    u = nb.pw(nb.sstep(0.0, 0.75, dz), 0.6)
    col = nb.mixcol(u, hor, zen)
    if p.get('hglow'):
        gc, k = p['hglow']
        gl = nb.exp(nb.mul(nb.mx(dz, 0.0), -k))
        col = nb.vadd(col, nb.colscale(gc, gl))
    bl = nb.sstep(0.0, -0.02, dz)
    col = nb.mixcol(bl, col, below)
    md = Vector(p.get('moon_dir', (0.3, 1.0, 0.4))).normalized()
    cosm = nb.dot(D, tuple(md))
    ang = nb.math('ARCCOSINE', nb.math('MINIMUM', cosm, 1.0))
    mc = p.get('moon_col', MOON)
    hI, hw = p.get('halo', (0.04, 0.2))
    h2I, h2w = p.get('halo2', (0.015, 0.8))
    halo = nb.add(nb.mul(nb.exp(nb.div(ang, -hw)), hI), nb.mul(nb.exp(nb.mul(nb.pw(nb.div(ang, h2w), 2.0), -1.0)), h2I))
    col = nb.vadd(col, nb.colscale(mc, halo))
    sky_only = col                      # for world lighting (no disk / stars)
    if p.get('moon_I', 0) > 0:
        R = math.radians(p.get('moon_r_deg', 0.6))
        disk = nb.sstep(R * 1.04, R * 0.96, ang)
        # a little limb darkening + maria texture
        mt_ = nb.noise(nb.vscale(D, 900.0), scale=1.0, detail=4.0, rough=0.6)
        tex = nb.madd(mt_.outputs['Fac'], 0.45, 0.72)
        col = nb.vadd(col, nb.colscale((1.0, 0.97, 0.92), nb.mul(nb.mul(disk, tex), p['moon_I'])))
    if p.get('milky'):
        col = nb.vadd(col, _milky_way(nb, D, p['milky']))
    if p.get('stars'):
        col = nb.vadd(col, _stars(nb, D, dz, p['stars']))
    # world lighting (what surfaces see) vs camera rays: keep the lighting smooth (no stars)
    lp = nb.lightpath()
    lg = p.get('light_gain', 1.0)
    bg_cam = nb.n('ShaderNodeBackground', inputs={'Color': col, 'Strength': 1.0})
    bg_light = nb.n('ShaderNodeBackground', inputs={'Color': sky_only, 'Strength': lg})
    ms = nb.mixshader(lp.outputs['Is Camera Ray'], bg_light, bg_cam)
    o = nb.n('ShaderNodeOutputWorld')
    nb.link(ms, o.inputs['Surface'])
    try:
        w.sun_threshold = 1000.0          # never extract a 'sun' from the moon disk
    except Exception:
        pass
    return w


def _stars(nb, D, dz, s):
    """Voronoi star field on the direction sphere; several layers, steep brightness distribution."""
    acc = None
    for li, (scale, dens, gain, rad) in enumerate(s.get('layers', [(260.0, 0.35, 1.0, 0.09),
                                                                    (700.0, 0.25, 0.35, 0.07),
                                                                    (1600.0, 0.3, 0.12, 0.06)])):
        v = nb.voronoi(nb.vadd(D, (li * 17.3, li * 5.1, li * 9.7)), scale=scale, feature='F1', rand=1.0)
        dist = v.outputs['Distance']
        rnd = nb.sep(v.outputs['Color'])
        keep = nb.math('LESS_THAN', rnd[0], dens)
        br = nb.pw(nb.div(rnd[1], dens * 0 + 1.0), 6.0)
        br = nb.madd(br, 12.0, 0.25)
        core = nb.exp(nb.mul(nb.pw(nb.div(dist, rad), 2.0), -1.0))
        tint = nb.mixcol(rnd[2], (1.0, 0.86, 0.72), (0.78, 0.86, 1.0))
        st = nb.colscale(tint, nb.mul(nb.mul(nb.mul(core, keep), br), gain))
        acc = st if acc is None else nb.vadd(acc, st)
    ext = nb.sstep(s.get('ext0', 0.0), s.get('ext1', 0.12), dz)
    band = s.get('band')
    if band:
        # more (and brighter) faint stars along the galactic band
        G, wdt, boost = band
        gb = nb.exp(nb.mul(nb.pw(nb.div(nb.dot(D, tuple(Vector(G).normalized())), wdt), 2.0), -1.0))
        ext = nb.mul(ext, nb.madd(gb, boost, 1.0))
    return nb.colscale(acc, nb.mul(ext, s.get('gain', 1.0)))


def _milky_way(nb, D, m):
    """Milky Way band: great circle with pole G, brighter toward the core direction C; clumpy clouds
    and dark dust lanes along the mid-plane."""
    G = Vector(m['pole']).normalized()
    C = Vector(m['core']).normalized()
    w = m.get('width', 0.16)
    b = nb.dot(D, tuple(G))                               # sin(galactic latitude)
    lon_c = nb.dot(D, tuple(C))                           # cos-ish of angle to core
    core = nb.pw(nb.mul(nb.madd(lon_c, 0.5, 0.5), 1.0), m.get('core_pow', 3.0))
    width = nb.madd(core, w * 0.8, w)
    band = nb.exp(nb.mul(nb.pw(nb.div(b, width), 2.0), -1.0))
    clouds = nb.noise(nb.vscale(D, 1.0), scale=m.get('cloud_scale', 7.0), detail=8.0, rough=0.62, dist=0.3)
    cl = nb.pw(nb.clamp01(nb.madd(clouds.outputs['Fac'], 1.6, -0.35)), 1.6)
    fine = nb.noise(nb.vscale(D, 1.0), scale=m.get('fine_scale', 40.0), detail=6.0, rough=0.6)
    cl = nb.mul(cl, nb.madd(fine.outputs['Fac'], 0.8, 0.6))
    # dust lanes: dark filaments hugging the mid-plane
    dn = nb.noise(nb.vscale(D, 1.0), scale=m.get('dust_scale', 11.0), detail=7.0, rough=0.6, dist=0.6)
    lane = nb.sstep(0.46, 0.62, dn.outputs['Fac'])
    lane_w = nb.exp(nb.mul(nb.pw(nb.div(b, nb.mul(width, 0.45)), 2.0), -1.0))
    dust = nb.sub(1.0, nb.mul(nb.mul(lane, lane_w), m.get('dust', 0.85)))
    I = nb.mul(nb.mul(nb.mul(band, cl), dust), nb.madd(core, m.get('core_gain', 2.5), 0.35))
    col = nb.mixcol(nb.clamp01(nb.mul(core, 1.2)), m.get('col_arm', (0.55, 0.62, 0.85)),
                    m.get('col_core', (1.0, 0.86, 0.7)))
    ext = nb.sstep(0.0, 0.25, nb.sep(D)[2])
    return nb.colscale(col, nb.mul(nb.mul(I, ext), m.get('gain', 0.02)))


# ------------------------------------------------------------------- export ---

def write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(obj, f)
