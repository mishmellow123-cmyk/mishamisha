"""CITY (src 1680-1719, ignition 1690; rooftops answer 1694-1716).

One idea: this is our world. A young person at a rooftop parapet lowers a torch into a steel drum on
the beat; across a real city at night -- thousands of lit windows, water tanks, street glow in the
haze -- other rooftops answer, one after another, toward the horizon.

Camera (as the original): slow push 1.3 m, yaw -1 deg held to the hit then easing to +1.2 deg (toward
the answers), small kick on the hit. Shared spec (venv + Blender); build() runs inside Blender.
"""
import math

START, END, IGN = 1680, 1719, 1690
SAMPLES = 96
FPS = 24.0
HFOV = 40.0
STREET = -52.0                     # street level relative to our roof deck
PARAPET_Y = 11.9                   # inner face of the front parapet
CAM0 = (-0.9, -3.1, 1.82)
PITCH0 = -0.4
DRUM = (0.78, 11.5, 0.0)
FEET = (0.12, 10.85, 0.0)
FIG_YAW = 14.0                      # degrees, turned slightly right toward the drum
GRID_ANG = 28.0
DOWNTOWN = (-726.0, 2911.0)         # centre of the tower cluster (bearing -14 deg, 3 km): left third
MIDTOWN = (1608.0, 5260.0)          # a smaller, hazier cluster far right (bearing +17 deg, 5.5 km)

# answering fires: (frame, target distance m, screen-bearing deg from view axis)
ANSWERS = [(1694, 95, 13.0), (1697, 170, 9.5), (1700, 260, 16.5), (1702, 380, 11.0), (1705, 560, 17.5),
           (1708, 800, 12.5), (1710, 1150, 15.5), (1713, 1600, 10.0), (1716, 2300, 14.0)]
ANSWER_Y = [None, None, None, 474, 462, 452, 443, 432, 421]     # target rows for the far answers

RELEASE = 1690.5                    # the torch is let go into the drum on the hit
FINISH = dict(exposure=1.0, bloom_strength=0.075, bloom_threshold=0.8, streak_strength=0.0, vignette_amount=0.25)


def ftime(f):
    return f / FPS


# ======================================================================= venv side ===

def flame_specs():
    import fireparts as FP
    specs = [FP.FlameSpec('drum', Hf=1.5, Rb=0.27, seed=12, I=26.0, tongues=6, lean=0.10, ppm=300,
                          t_ign=ftime(IGN)),
             FP.FlameSpec('torch', Hf=0.30, Rb=0.045, seed=29, I=16.0, tongues=3, lean=0.03, ppm=700, env=False)]
    for i, (fr, dist, brg) in enumerate(ANSWERS[:3]):
        specs.append(FP.FlameSpec(f'ans{i}', Hf=1.0, Rb=0.25, seed=40 + i, I=22.0, tongues=4, lean=0.08,
                                  ppm=60 if i == 0 else 40, t_ign=ftime(fr)))
    return specs


def timing(frames):
    import fireparts as FP
    out = dict(drum=[], torch=[], ember=[], ans={})
    for f in frames:
        t = ftime(f)
        s, i, l = FP.ignite_env(t, ftime(IGN))
        out['drum'].append((f, l * FP.flicker(t, 31)))
        out['torch'].append((f, FP.flicker(t, 5)))
        out['ember'].append((f, 0.0 if f < IGN else min(1.0, (f - IGN + 1) / 10.0) * (0.8 + 0.2 * FP.flicker(t, 3))))
    for k, (fr, dist, brg) in enumerate(ANSWERS):
        tab = []
        for f in frames:
            t = ftime(f)
            s, i, l = FP.ignite_env(t, ftime(fr))
            tab.append((f, l * FP.flicker(t, fr)))
        out['ans'][str(k)] = tab
    return out


KID_KEYS = [
    (START, dict(lean=0.1, twist=0.1, head_yaw=0.18, head_pitch=-0.25, r_flex=0.55, r_abd=0.2, r_elbow=0.95,
                 l_flex=0.32, l_abd=0.16, l_elbow=1.25, r_hip=0.1, r_knee=0.3, l_hip=-0.02, l_knee=0.03,
                 stance=0.075, side_lean=0.03, breath=0.0)),
    (1684, dict(lean=0.12, twist=0.14, head_yaw=0.28, head_pitch=-0.35, r_flex=0.7, r_abd=0.25, r_elbow=0.9)),
    (1689.5, dict(lean=0.22, twist=0.3, head_yaw=0.35, head_pitch=-0.45, r_flex=1.05, r_abd=0.4, r_elbow=0.35,
                  l_flex=0.25, l_abd=0.2, l_elbow=1.15, breath=0.5)),
    (1692, dict(lean=0.02, twist=0.18, head_yaw=0.05, head_pitch=-0.15, r_flex=0.85, r_abd=0.4, r_elbow=0.8,
                l_flex=0.3, l_abd=0.17, l_elbow=1.2, breath=1.0)),
    (1697, dict(lean=-0.02, twist=0.1, head_yaw=0.1, head_pitch=0.0, r_flex=0.35, r_abd=0.2, r_elbow=0.9,
                l_flex=0.33, l_abd=0.15, l_elbow=1.28, breath=0.4)),
    (1706, dict(lean=0.0, twist=0.05, head_yaw=0.45, head_pitch=0.08, r_flex=0.25, r_abd=0.18, r_elbow=0.85)),
    (END, dict(lean=0.0, twist=0.02, head_yaw=0.62, head_pitch=0.1, r_flex=0.22, r_abd=0.16, r_elbow=0.8, breath=0.8)),
]


def _to_local(P):
    import numpy as np
    a = math.radians(FIG_YAW)
    d = np.asarray(P, float) - np.asarray(FEET, float)
    c, s_ = math.cos(a), math.sin(a)
    return np.array([c * d[0] - s_ * d[1], s_ * d[0] + c * d[1], d[2]])


def kid_frame(f):
    """Pose (joints) + torch grip/direction (figure-local) for frame f."""
    import numpy as np
    import figures as FG
    p = FG.interp_pose(f, KID_KEYS)
    p['breath'] = p.get('breath', 0.0) + 0.3 * math.sin(2 * math.pi * f / 72.0)
    J = FG.fk(p)
    mouth = _to_local(np.array(DRUM) + np.array([0, 0, 0.95 - 0.12]))
    u_in = FG.ease((f - 1683.5) / 6.0) * (1 - FG.ease((f - 1690.5) / 5.0))
    hold = np.array([0.12, 0.5, 1.0])
    hold /= np.linalg.norm(hold)
    thr = mouth - J['r_sh']
    thr /= np.linalg.norm(thr)
    thr = thr + np.array([0, 0, -0.35])
    thr /= np.linalg.norm(thr)
    tdir = hold * (1 - u_in) + thr * u_in
    tdir /= np.linalg.norm(tdir)
    if u_in > 1e-4:
        FG.ik_arm(J, 'r', J['r_hand'] * (1 - u_in) + (mouth - thr * 0.6) * u_in, pole=(0.6, -0.2, -0.8))
    if f >= RELEASE:
        return J, mouth - thr * 0.6, thr
    # left hand stays in the hoodie's front pocket (elbow out and back)
    pocket = J['pelvis'] + J['R_pelvis'] @ np.array([-0.045, 0.125, 0.12])
    FG.ik_arm(J, 'l', pocket, pole=(-0.9, -0.6, -0.25))
    return J, J['r_hand'].copy(), tdir


def prep(frames, cache):
    """Per-frame SDF meshes of the young person + torch transforms (figure-local)."""
    import os
    import figures as FG
    d = os.path.join(cache, 'figure')
    os.makedirs(d, exist_ok=True)
    torch = {}
    for f in frames:
        J, grip, tdir = kid_frame(f)
        torch[str(f)] = [list(map(float, grip)), list(map(float, tdir))]
        path = os.path.join(d, f'kid_{f:05d}.bin')
        if os.path.exists(path) and not os.environ.get('MT3D_REFIG'):
            continue
        Vv, Q, M = FG.mesh_sdf(FG.hoodie_body(J, f / FPS), h=0.0105, disp=(0.003, 8.0, 21.0, 0.0))
        FG.write_mesh(path, Vv, Q, M)
    return dict(kid_dir=d, kid_torch=torch)


_SPARKS = None


def post(frame, hdr, depth, cam, scene):
    """Sparks from the drum, heat shimmer, distant answering fires (glows) -- all depth-tested."""
    import numpy as np
    import fireparts as FP
    global _SPARKS
    t = ftime(frame)
    fb = scene['drum_fire_base']
    if frame >= IGN:
        if _SPARKS is None:
            _SPARKS = FP.ZSparks(83, (fb[0], fb[1], fb[2] + 0.2), ftime(IGN), ftime(END) + 0.1, burst=110, rate=28,
                                 ember_rate=8, wind=(0.7, 0.3, 0.0), radius=0.2, I=26.0, burst_speed=(2.5, 6.5),
                                 speed=(1.5, 4.0), spread=0.38, buoy=4.0)
        s, i, l = FP.ignite_env(t, ftime(IGN))
        FP.shimmer(hdr, cam, fb, 1.5 * s, 0.27, t, amp_px=1.1 * cam.W / 1920)
        _SPARKS.render(hdr, depth, cam, t)
    for k, a in enumerate(scene['answers']):
        fr = a['frame']
        if frame < fr - 1:
            continue
        s, i, l = FP.ignite_env(t, ftime(fr))
        P = a['pos']
        sx, sy, z = cam.project(np.array(P))
        e = (1.0 + 0.8 * math.exp(-(frame - fr) / 3.0)) * FP.flicker(t, fr) * min(max(s, 0) / 0.5, 1.0)
        ppm = cam.fpx / max(z, 1.0)
        sW = cam.W / 1920.0
        if k >= 3:                     # far answers: hot glow points (near ones are 3-D sprite cards)
            core_px = (max(0.45 * ppm, 0.6) + 0.5) * sW
            FP.glow(hdr, depth, cam, np.array(P) + np.array([0, 0, 0.5]), 70.0 * e * sW * sW, core_px, zbias=8.0)
            FP.glow(hdr, depth, cam, np.array(P) + np.array([0, 0, 0.6]), 30.0 * e * sW * sW, 2.6 * core_px, zbias=8.0)
        # air glow around every answering fire (haze in-scatter)
        FP.halo(hdr, depth, cam, np.array(P) + np.array([0, 0, 1.0]), max(14.0, 3.5 * ppm) * sW,
                0.03 * e * min(1.0, 500.0 / max(z, 1.0)) ** 0.4, zbias=-1e9)
    return hdr


# ==================================================================== Blender side ===

def build(job):
    import random

    import bpy
    from mathutils import Matrix, Vector

    from kit import core as C
    from kit import fire as FK
    from kit import figure as FG
    from kit.nodes import new_material

    T = job['timing']
    opts = job.get('opts', {})
    C.setup_render(scale=job['scale'], samples=job['samples'],
                   vol=dict(start=3.0, end=60.0, tile=4 if job.get('final') else 8, samples=64, dist=0.6),
                   mblur=bool(opts.get('mblur', True)), shutter=0.5)
    rng = random.Random(505)

    # ------------------------------------------------------------- atmosphere / sky
    hz = C.hexlin('#2A3558')
    C.atmos_group(dict(a=1.0 / 650.0, Hs=80.0, h0=STREET, u=1.0 / 40000.0, amb=(0.020, 0.025, 0.040),
                       glow=(0.15, 0.14, 0.13), Hg=40.0, moon_dir=(-0.5, 0.6, 0.45), fwd=0.3, fwd_pow=6.0))
    C.sky_world(dict(zenith=C.hexlin('#070B1C'), horizon=hz, below=(0.03, 0.032, 0.04),
                     hglow=((0.085, 0.085, 0.095), 6.0), moon_dir=(-0.5, 0.6, 0.45), moon_I=0.0,
                     halo=(0.01, 0.3), halo2=(0.006, 1.0),
                     stars=dict(gain=0.35, ext0=0.05, ext1=0.3,
                                layers=[(260.0, 0.18, 1.0, 0.08), (700.0, 0.12, 0.3, 0.07)]),
                     light_gain=0.8))
    moon = C.sun('moon', (-0.5, 0.6, 0.45), C.MOON, 0.10, angle_deg=1.0, volume=0.0)

    # ------------------------------------------------------------------ camera
    cam = C.make_camera('CAM', hfov=HFOV, clip=(0.1, 15000.0))

    def ease_push(u):
        u = min(max(u, 0.0), 1.0)
        return u * 0.7 + 0.3 * u * u * (3 - 2 * u)

    def smoother(u):
        u = min(max(u, 0.0), 1.0)
        return u * u * u * (u * (u * 6 - 15) + 10)

    def kick(t, t0, amp=1.0, freq=5.5, decay=5.5):
        if t < t0:
            return 0.0
        u = t - t0
        return amp * math.exp(-decay * u) * math.sin(2 * math.pi * freq * u)

    # shallow depth of field on the figure: the far city's windows soften into small bokeh
    dof = cam.data.dof
    dof.use_dof = True
    dof.aperture_fstop = 1.8
    dof.aperture_blades = 7
    dof.aperture_ratio = 1.0
    sc_ = bpy.context.scene
    sc_.eevee.use_bokeh_jittered = True
    sc_.eevee.bokeh_max_size = 60.0
    for f in range(START - 1, END + 2):
        u = (f - START) / (END - START)
        push = 1.3 * ease_push(u)
        yaw = -1.0 if f <= IGN else -1.0 + 2.2 * smoother((f - IGN) / (END - IGN))
        k = kick(ftime(f), ftime(IGN))
        C.key_camera(cam, f, (CAM0[0], CAM0[1] + push, CAM0[2] + 0.01 * k), yaw + 0.05 * k, PITCH0 + 0.07 * k)
        dof.focus_distance = math.hypot(FEET[0] - CAM0[0], FEET[1] - CAM0[1] - push)
        dof.keyframe_insert('focus_distance', frame=f)

    # --------------------------------------------------------------- our roof
    deck_m, nb = new_material('deck')
    tc = nb.texco()
    P = nb.geo().outputs['Position']
    n1 = nb.noise(P, scale=0.35, detail=6.0, rough=0.6)
    n2 = nb.noise(P, scale=6.0, detail=5.0, rough=0.7)
    px_, py_, pz_ = nb.sep(P)
    pv_ = nb.add(n1.outputs['Fac'], nb.mul(nb.sub(n2.outputs['Fac'], 0.5), 0.05))
    puddle = nb.sstep(0.6, 0.64, pv_)
    base = nb.mixcol(n2.outputs['Fac'], (0.028, 0.028, 0.03), (0.05, 0.048, 0.047))
    base = nb.mixcol(puddle, base, (0.012, 0.012, 0.014))
    rough = nb.mixf(puddle, nb.madd(n2.outputs['Fac'], 0.25, 0.62), 0.04)
    grit = nb.bump(n2.outputs['Fac'], strength=nb.mixf(puddle, 0.25, 0.0), distance=0.01)
    bs = nb.principled(Base_Color=base, Roughness=rough)
    bs.inputs['Specular IOR Level'].default_value = 0.5
    nb.link(grit, bs.inputs['Normal'])
    nb.output(surface=C.fogged(nb, bs))
    C.mesh_obj('deck', [(-16, -12, 0), (16, -12, 0), (16, PARAPET_Y, 0), (-16, PARAPET_Y, 0)], [(0, 1, 2, 3)],
               mat=deck_m, smooth=False)

    conc = C.surface_mat('parapet', (0.06, 0.058, 0.056), rough=0.9,
                         extra=lambda nb, bs: nb.link(nb.bump(nb.noise(nb.texco().outputs['Object'], scale=20.0,
                                                                        detail=6.0).outputs['Fac'], 0.3, 0.01),
                                                          bs.inputs['Normal']))
    coping = C.surface_mat('coping', (0.09, 0.09, 0.095), rough=0.45, spec=0.6)
    # front parapet (thick wall) + coping cap; side parapet on the left running back
    def box(name, x0, x1, y0, y1, z0, z1, mat):
        V = [(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0), (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)]
        F = [(0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7), (4, 5, 6, 7), (0, 3, 2, 1)]
        return C.mesh_obj(name, V, F, mat=mat, smooth=False)
    box('parapet_f', -16, 16, PARAPET_Y, PARAPET_Y + 0.38, 0.0, 1.02, conc)
    box('coping_f', -16.05, 16.05, PARAPET_Y - 0.04, PARAPET_Y + 0.42, 1.02, 1.08, coping)
    box('parapet_l', -16.4, -16.0, -12, PARAPET_Y + 0.38, 0.0, 1.02, conc)
    # our building's front wall below the parapet (seen past the coping at grazing angles)
    wallm = C.surface_mat('wall', (0.05, 0.045, 0.042), rough=0.9)
    box('front_wall', -16, 16, PARAPET_Y + 0.1, PARAPET_Y + 0.38, STREET, 0.0, wallm)

    metal = C.surface_mat('hvac', (0.07, 0.075, 0.08), rough=0.5, spec=0.6)
    box('hvac1', -9.5, -6.8, 4.0, 6.2, 0.0, 1.55, metal)
    box('hvac1_top', -9.4, -6.9, 4.1, 6.1, 1.55, 1.62, metal)
    box('hvac2', 5.6, 7.2, 7.4, 9.9, 0.0, 1.1, metal)
    box('bulkhead', 9.5, 14.5, 1.5, 6.0, 0.0, 3.2, wallm)
    box('bulk_cap', 9.4, 14.6, 1.4, 6.1, 3.2, 3.3, coping)

    # ------------------------------------------------------------ water tanks
    tank_col = C.collection('TANKS_SRC', hide=True)
    tank_ob = _water_tank(C, FK, new_material, 'tankA', tank_col)
    _water_tank(C, FK, new_material, 'tankB', tank_col, dia=3.0, h=3.4, leg=3.2, staves=True)
    # our own tank, big, on the left (framing silhouette)
    own = []
    for (x, y, z, s, yaw_) in ((-12.5, 9.0, 0.0, 1.0, 0.3),):
        own.append((x, y, z))
    C.instancer('own_tank', own, tank_col, rot=[(0, 0, 0.3)], scl=[(1.0, 1.0, 1.0)], var=[0])

    # ------------------------------------------------------------------- city
    info_city = _city(C, FK, new_material, rng, tank_col)

    # --------------------------------------------------------------- the drum
    ember = [(f, v) for f, v in T['ember']]
    D = FK.drum('drum', DRUM, yaw=0.4, ember_frames=[(f, v * 1.2) for f, v in ember])
    fire_base = D['fire_base']
    card = T['sprites']['drum']['card']
    FK.flame_card('drum_flame', fire_base, card, T['sprites']['drum']['first'], cam)
    FK.fire_lights('drum', fire_base, 1.5, T['drum'], power=360.0, radius=0.3, cutoff=120.0)
    FK.smoke_plume('drum_smoke', fire_base + Vector((0, 0, 0.5)), ftime(IGN), height=9.0, r0=0.2, spread=0.26,
                   rise=1.4, wind=(0.8, 0.35), dens=2.2, albedo=0.3)
    FK.glow_haze('drum_haze', fire_base + Vector((0, 0, 0.9)), 5.5, density=0.004, aniso=0.3, flat=0.75)

    # ------------------------------------------------------------ the figure
    fig_info = _young_person(C, FG, FK, cam, T, fire_base)

    # ------------------------------------------------------ answering fires
    answers = []
    for k, (fr, dist, brg) in enumerate(ANSWERS):
        roof = info_city['answer_roofs'][k]
        pos = Vector(roof)
        answers.append(dict(frame=fr, pos=[pos.x, pos.y, pos.z]))
        tab = T['ans'][str(k)]
        pw = 1400.0 * (1.0 + dist / 400.0)
        FK.fire_lights(f'ans{k}', pos, 1.0, tab, power=pw, radius=0.4, cutoff=max(80.0, dist * 0.15),
                       spread=(0.4,), split=(1.0,), shadow=k < 3, volume=0.0)
        if k < 3:
            s = T['sprites'][f'ans{k}']
            FK.flame_card(f'ans{k}_flame', pos, s['card'], s['first'], cam)
            # a small beacon drum on that roof
            FK.drum(f'ans{k}_drum', pos - Vector((0, 0, 0.8)), yaw=0.2 * k, holes=False)
    if opts.get('figcam'):
        # debug: close inspection camera behind/left of the figure
        cam.animation_data_clear()
        cam.data.animation_data_clear()
        ox, oy, oz, yw, pt = opts['figcam']
        for f in range(START - 1, END + 2):
            C.key_camera(cam, f, (FEET[0] + ox, FEET[1] + oy, oz), yw, pt)
    # someone across the street answers: a small silhouette beside the first answering fire
    import os
    a0 = answers[0]['pos']
    pmat = FG.figure_material('answerer', [(0.011, 0.011, 0.013), (0.009, 0.010, 0.015), (0.02, 0.013, 0.010),
                                           (0.03, 0.03, 0.032)], sheen=0.35, folds=0.0)
    me = FG.mesh_from_bin('answerer', os.path.join(T['kid_dir'], f'kid_{END:05d}.bin'), pmat)
    per = C.link_obj(bpy.data.objects.new('answerer', me))
    per.matrix_world = Matrix.Translation(Vector((a0[0] - 0.95, a0[1] + 0.15, a0[2] - 0.85))) @ \
        Matrix.Rotation(math.radians(-70.0), 4, 'Z')
    info = dict(drum_fire_base=list(fire_base), answers=answers, **fig_info)
    return info


def _water_tank(C, FK, new_material, name, coll, dia=4.0, h=4.3, leg=4.2, staves=True):
    """Classic wooden rooftop water tank on a steel stand (instancing source, base at origin)."""
    import bpy
    from mathutils import Vector
    r = dia / 2
    parts_wood = []
    n = 40
    V, F = [], []
    rings = [0.0, h]
    for j, z in enumerate(rings):
        for i in range(n):
            a = 2 * math.pi * i / n
            rr = r * (1.0 - 0.03 * (z / h))
            V.append((math.cos(a) * rr, math.sin(a) * rr, leg + z))
    for i in range(n):
        i2 = (i + 1) % n
        F.append((i, i2, n + i2, n + i))
    F.append(tuple(range(n))[::-1])
    # conical roof
    o = len(V)
    for i in range(n):
        a = 2 * math.pi * i / n
        V.append((math.cos(a) * (r + 0.12), math.sin(a) * (r + 0.12), leg + h - 0.05))
    V.append((0, 0, leg + h + r * 0.55))
    for i in range(n):
        i2 = (i + 1) % n
        F.append((o + i, o + i2, o + n))
    for i in range(n):
        i2 = (i + 1) % n
        F.append((n + i, n + i2, o + i2, o + i))
    wm, nb = new_material(name + '_wood')
    P = nb.texco().outputs['Object']
    x, y, z = nb.sep(P)
    ang = nb.math('ARCTAN2', y, x)
    st = nb.math('FRACT', nb.mul(ang, 60 / (2 * math.pi)))
    groove = nb.sstep(0.0, 0.08, nb.math('MINIMUM', st, nb.sub(1.0, st)))
    hoops = nb.sstep(0.035, 0.0, nb.math('ABSOLUTE', nb.sub(nb.math('FRACT', nb.div(z, 0.55)), 0.5)))
    grain = nb.noise(nb.comb(ang, 0.0, nb.mul(z, 0.3)), scale=8.0, detail=6.0)
    col = nb.mixcol(grain.outputs['Fac'], (0.035, 0.028, 0.022), (0.075, 0.06, 0.045))
    col = nb.mixcol(hoops, col, (0.02, 0.02, 0.022))
    bs = nb.principled(Base_Color=col, Roughness=0.85)
    nb.link(nb.bump(nb.mul(groove, nb.sub(1.0, hoops)), 0.4, 0.02), bs.inputs['Normal'])
    nb.output(surface=C.fogged(nb, bs))
    nwood = len(F)
    # steel stand: legs + cross bracing + platform beams
    steel = C.surface_mat(name + '_steel', (0.03, 0.03, 0.032), rough=0.6, spec=0.5)
    parts = []
    nl = 6
    for i in range(nl):
        a = 2 * math.pi * i / nl + 0.26
        x0, y0 = math.cos(a) * r * 0.82, math.sin(a) * r * 0.82
        x1, y1 = math.cos(a) * r * 0.72, math.sin(a) * r * 0.72
        parts.append(FK.tube([(x0, y0, 0.0), (x1, y1, leg)], 0.06, 4))
        a2 = 2 * math.pi * (i + 1) / nl + 0.26
        xa, ya = math.cos(a2) * r * 0.82, math.sin(a2) * r * 0.82
        xb, yb = math.cos(a2) * r * 0.72, math.sin(a2) * r * 0.72
        parts.append(FK.tube([(x0, y0, 0.3), (xb, yb, leg * 0.52)], 0.018, 4))
        parts.append(FK.tube([(xa, ya, 0.3), (x1, y1, leg * 0.52)], 0.018, 4))
        parts.append(FK.tube([((x0 + x1) / 2, (y0 + y1) / 2, leg * 0.52), ((xa + xb) / 2, (ya + yb) / 2, leg * 0.52)], 0.025, 4))
    for i in range(5):
        yy = -r + (i + 0.5) * dia / 5
        w = math.sqrt(max(r * r - yy * yy, 0.01))
        parts.append(FK.tube([(-w, yy, leg - 0.08), (w, yy, leg - 0.08)], 0.07, 4))
    # ladder
    parts.append(FK.tube([(r * 0.9, -0.25, 0.0), (r + 0.05, -0.25, leg + h)], 0.02, 4))
    parts.append(FK.tube([(r * 0.9, 0.25, 0.0), (r + 0.05, 0.25, leg + h)], 0.02, 4))
    for k in range(int((leg + h) / 0.35)):
        z = 0.2 + k * 0.35
        xx = r * 0.9 + (r + 0.05 - r * 0.9) * z / (leg + h)
        parts.append(FK.tube([(xx, -0.25, z), (xx, 0.25, z)], 0.012, 3))
    pv, pf = FK.merge(parts)
    o = len(V)
    V = V + pv
    F = F + [tuple(i + o for i in fc) for fc in pf]
    tank = C.mesh_obj(name, V, F, mat=[wm, steel], coll=coll)
    for poly in tank.data.polygons:
        poly.material_index = 1 if poly.vertices[0] >= o else 0
    tank.data.update()
    return tank


def _city(C, FK, new_material, rng, tank_col):
    """Grid city rotated GRID_ANG to the view. Pass 1 plans lots (footprint, height, type); the lots
    under the answering fires get roofs just above the sight line over our parapet; pass 2 emits one
    merged mesh of building boxes (setbacks, cornices, bulkheads, spires), instanced water tanks, a
    street-glow ground. Returns the answer-fire roof positions."""
    ga = math.radians(GRID_ANG)
    ca, sa = math.cos(ga), math.sin(ga)

    def to_world(u, v):             # grid (u along avenues, v across) -> world xy
        return (u * ca - v * sa, u * sa + v * ca)

    def to_grid(x, y):
        return (x * ca + y * sa, -x * sa + y * ca)

    def sight(d):                   # height of the sight line over our coping at distance d
        return CAM0[2] - (CAM0[2] - 1.12) * d / (PARAPET_Y - CAM0[1])

    BU, BV = 250.0, 84.0             # block pitch (u: along avenues, v: along streets)
    AVE, STR = 30.0, 19.0            # street widths
    cam_xy = (CAM0[0], CAM0[1])
    dx0, dy0 = DOWNTOWN
    # answer targets (world xy)
    targets = []
    for fr, dist, brg in ANSWERS:
        b = math.radians(brg)
        targets.append((cam_xy[0] + dist * math.sin(b), cam_xy[1] + dist * math.cos(b)))
    # ------------------------------------------------------------------ pass 1: lots
    lots = []
    for iu in range(-14, 16):
        for iv in range(-4, 80):
            u0 = iu * BU + AVE / 2
            u1 = (iu + 1) * BU - AVE / 2
            v0 = iv * BV + STR / 2
            v1 = (iv + 1) * BV - STR / 2
            cu, cv = (u0 + u1) / 2, (v0 + v1) / 2
            wx, wy = to_world(cu, cv)
            dx, dy = wx - cam_xy[0], wy - cam_xy[1]
            dist = math.hypot(dx, dy)
            if dist > 7000:
                continue
            brg = math.degrees(math.atan2(dx, dy))
            if dy < -20 or abs(brg) > 36 + 60 * math.exp(-dist / 150.0):
                continue
            for row in (0, 1):
                va = v0 if row == 0 else (v0 + v1) / 2 + 0.5
                vb = (v0 + v1) / 2 - 0.5 if row == 0 else v1
                u = u0
                while u < u1 - 6:
                    lw = min(12 + 30 * rng.random() ** 1.3, u1 - u)
                    if dist > 3500:
                        lw = min(lw * 1.8, u1 - u)
                    if u1 - (u + lw) < 8:
                        lw = u1 - u
                    lx0, lx1 = u + 0.4, u + lw - 0.4
                    u += lw
                    mx, my = to_world((lx0 + lx1) / 2, (va + vb) / 2)
                    ddx, ddy = mx - cam_xy[0], my - cam_xy[1]
                    d = math.hypot(ddx, ddy)
                    if (ddy < 14.0 and abs(mx) < 40) or d < 25:
                        continue
                    dd = math.hypot(mx - dx0, my - dy0)
                    down = math.exp(-(dd / 420.0) ** 2)
                    mid2 = math.exp(-(math.hypot(mx - MIDTOWN[0], my - MIDTOWN[1]) / 320.0) ** 2)
                    r = rng.random()
                    if down > 0.08 and r < 0.42 * down:
                        H = 85 + 360 * rng.random() ** 2.2 * (0.3 + 0.7 * down)
                        bt = 2 if rng.random() < 0.65 else 1
                    elif mid2 > 0.1 and r < 0.6 * mid2:
                        H = 90 + 150 * rng.random() * mid2
                        bt = 2 if rng.random() < 0.5 else 1
                    elif r < 0.035:
                        H = 50 + 45 * rng.random()
                        bt = 1 if rng.random() < 0.4 else 0
                    else:
                        H = 11 + 28 * rng.random() ** 1.4
                        bt = 0
                    if d < 150:
                        H = max(12.0, sight(d) - 6.0 - 6.0 * rng.random() - STREET)   # hidden
                        bt = 0
                    elif d < 650 and down < 0.2:
                        # roofs around the sight line over our parapet: most hide, some peek (tanks)
                        H = max(14.0, sight(d) - 7.5 + 9.0 * rng.random() - STREET)
                        bt = 0
                    lots.append(dict(u0=lx0, u1=lx1, v0=va, v1=vb, H=H, bt=bt, d=d, mx=mx, my=my,
                                     bid=rng.random(), r1=rng.random(), r2=rng.random(), r3=rng.random()))
    # ------------------------------------------------- answers: pick + raise their lots
    ans = []
    used = set()
    for k, (tx, ty) in enumerate(targets):
        tu, tv = to_grid(tx, ty)
        best = None
        for i, L in enumerate(lots):
            if i in used or L['bt'] != 0 or (L['u1'] - L['u0']) < 9:
                continue
            cu = min(max(tu, L['u0'] + 3), L['u1'] - 3)
            cv = min(max(tv, L['v0'] + 3), L['v1'] - 3)
            e = math.hypot(cu - tu, cv - tv)
            if best is None or e < best[0]:
                best = (e, i, cu, cv)
        e, i, cu, cv = best
        used.add(i)
        L = lots[i]
        wx, wy = to_world(cu, cv)
        d = math.hypot(wx - cam_xy[0], wy - cam_xy[1])
        L['answer'] = True
        L['tank'] = False
        yt = ANSWER_Y[k]
        if yt is None:
            # keep the building itself below our parapet line; the fire stands on a stair bulkhead
            # that just peeks over it (a small dark box, not a wall that hides the next answers)
            L['H'] = min(L['H'], sight(d) - 2.2 - STREET)
            top = sight(d) + 1.3 + 0.4 * rng.random()
            L['perch'] = (cu, cv, STREET + L['H'], top)
            ans.append((wx, wy, top + 0.85))
        else:
            # choose the roof height so the fire lands on row yt (the chain climbs toward the horizon)
            fpx = 960.0 / math.tan(math.radians(HFOV / 2))
            ang = math.radians(PITCH0) - math.atan((yt - 402.0) / fpx)
            z_fire = CAM0[2] + d * math.tan(ang)
            L['H'] = max(12.0, z_fire - 0.85 - STREET)
            L['bt'] = 0
            ans.append((wx, wy, STREET + L['H'] + 0.85))
    # ------------------------------------------------------------------ pass 2: emit
    V, F, BID, BTYPE, BTOP = [], [], [], [], []
    tanks, tank_rot, tank_var = [], [], []
    cur_top = [0.0]

    def add_box(x0, x1, y0, y1, z0, z1, bid, bt):
        w = [to_world(u, v) for u, v in ((x0, y0), (x1, y0), (x1, y1), (x0, y1))]
        o = len(V)
        for (x, y) in w:
            V.append((x, y, z0))
        for (x, y) in w:
            V.append((x, y, z1))
        for fc in ((0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7), (4, 5, 6, 7)):
            F.append(tuple(o + i for i in fc))
        BID.extend([bid] * 8)
        BTYPE.extend([bt] * 8)
        BTOP.extend([cur_top[0]] * 8)

    fig_brg = (3.0, 13.0)
    for L in lots:
        lx0, lx1, va, vb, H, bt, d, bid = L['u0'], L['u1'], L['v0'], L['v1'], L['H'], L['bt'], L['d'], L['bid']
        z0, z1 = STREET, STREET + H
        cur_top[0] = z1
        fw, fd = lx1 - lx0, vb - va
        brg = math.degrees(math.atan2(L['mx'] - cam_xy[0], L['my'] - cam_xy[1]))
        if L.get('perch'):
            pu, pv, pz0, pz1 = L['perch']
            add_box(pu - 1.5, pu + 1.5, pv - 1.4, pv + 1.4, pz0, pz1, bid, 3)
            add_box(pu - 1.62, pu + 1.62, pv - 1.52, pv + 1.52, pz1 - 0.05, pz1 + 0.12, bid, 3)
        if bt == 0:
            add_box(lx0, lx1, va, vb, z0, z1, bid, bt)
            if d < 2500:
                add_box(lx0 - 0.35, lx1 + 0.35, va - 0.35, vb + 0.35, z1 - 0.9, z1 - 0.5, bid, 3)
            near_fig = d < 700 and fig_brg[0] < brg < fig_brg[1]
            if d < 2600 and L['r1'] < 0.42 and fw > 10 and not near_fig and L.get('tank', True):
                tu = lx0 + fw * (0.25 + 0.5 * L['r2'])
                tv = va + fd * (0.3 + 0.4 * L['r3'])
                tx, ty = to_world(tu, tv)
                tanks.append((tx, ty, z1))
                tank_rot.append((0, 0, L['r2'] * 6.28))
                tank_var.append(0 if L['r3'] < 0.5 else 1)
            if L['r2'] < 0.5 and d < 1800 and not L.get('answer'):
                bu = lx0 + fw * (0.2 + 0.5 * L['r3'])
                bv = va + fd * (0.2 + 0.5 * L['r1'])
                add_box(bu, bu + 4.5, bv, bv + 4.0, z1, z1 + 3.2, bid, 3)
        else:
            tiers = 1 + int(L['r1'] * 3)
            zc = z0
            inset = 0.0
            hh = H
            for ti in range(tiers):
                th = hh * 0.62 if ti < tiers - 1 else z1 - zc
                add_box(lx0 + inset, lx1 - inset, va + inset, vb - inset, zc, zc + th, bid, bt)
                zc += th
                inset += min(fw, fd) * (0.1 + 0.08 * L['r2'])
                hh -= th
                if lx1 - inset - (lx0 + inset) < 6:
                    break
            if L['r3'] < 0.35:
                cx_, cy_ = (lx0 + lx1) / 2, (va + vb) / 2
                add_box(cx_ - 0.8, cx_ + 0.8, cy_ - 0.8, cy_ + 0.8, zc, zc + 18 + 40 * L['r2'], bid, 3)
    ci = C.mesh_obj('city', V, F, mat=_facade_material(C, new_material), smooth=False)
    me = ci.data
    a = me.attributes.new('bid', 'FLOAT', 'POINT')
    a.data.foreach_set('value', BID)
    a = me.attributes.new('btype', 'FLOAT', 'POINT')
    a.data.foreach_set('value', BTYPE)
    a = me.attributes.new('btop', 'FLOAT', 'POINT')
    a.data.foreach_set('value', BTOP)
    print('city: %d lots, %d boxes, %d tanks' % (len(lots), len(F) // 5, len(tanks)))
    if tanks:
        C.instancer('city_tanks', tanks, tank_col, rot=tank_rot, scl=[(1, 1, 1)] * len(tanks), var=tank_var)
    # street-glow ground (only the far avenues are ever seen; it mainly lights the haze look)
    gm, nb = new_material('streets')
    Pg = nb.geo().outputs['Position']
    x, y, z = nb.sep(Pg)
    gu = nb.add(nb.mul(x, ca), nb.mul(y, sa))
    gv = nb.add(nb.mul(x, -sa), nb.mul(y, ca))
    fu = nb.math('FRACT', nb.div(gu, BU))
    du = nb.mul(nb.math('MINIMUM', fu, nb.sub(1.0, fu)), BU)
    fv = nb.math('FRACT', nb.div(gv, BV))
    dv = nb.mul(nb.math('MINIMUM', fv, nb.sub(1.0, fv)), BV)
    st = nb.add(nb.sstep(AVE / 2, AVE / 2 - 3, du), nb.sstep(STR / 2, STR / 2 - 2, dv))
    em = nb.emission((0.9, 0.85, 0.78), nb.mul(nb.clamp01(st), 1.6))
    nb.output(surface=C.fogged(nb, em))
    R = 9000
    C.mesh_obj('streets', [(-R, -200, STREET), (R, -200, STREET), (R, R, STREET), (-R, R, STREET)], [(0, 1, 2, 3)],
               mat=gm, smooth=False)
    return dict(answer_roofs=ans)


def _facade_material(C, new_material):
    """Procedural facades: masonry (pre-war) or glass/office towers; per-window hash -> lit state,
    colour temperature, brightness; unlit glass reflects the sky; street light uplights the base."""
    m, nb = new_material('facade')
    g = nb.geo()
    P = g.outputs['Position']
    N = g.outputs['Normal']
    nx, ny, nz = nb.sep(N)
    px, py, pz = nb.sep(P)
    bid = nb.attr('bid').outputs['Fac']
    bt = nb.attr('btype').outputs['Fac']
    # facade coordinates: u along the wall, v = height above the street
    u = nb.add(nb.mul(px, nb.mul(ny, -1.0)), nb.mul(py, nx))
    v = nb.sub(pz, STREET)
    is_glass = nb.math('GREATER_THAN', bt, 0.5)
    is_tower = nb.math('GREATER_THAN', bt, 1.5)
    is_trim = nb.math('GREATER_THAN', bt, 2.5)
    bw = nb.madd(nb.math('FRACT', nb.mul(bid, 7.13)), 1.4, 1.5)
    bw = nb.mixf(is_glass, bw, nb.madd(nb.math('FRACT', nb.mul(bid, 3.1)), 1.0, 1.5))
    fh = nb.madd(nb.math('FRACT', nb.mul(bid, 3.71)), 0.7, 3.3)
    uu = nb.div(u, bw)
    vv = nb.div(v, fh)
    cu = nb.math('FLOOR', uu)
    cv = nb.math('FLOOR', vv)
    fu = nb.sub(uu, cu)
    fv = nb.sub(vv, cv)
    # window aperture (masonry: punched windows; glass: nearly full-bay)
    wu = nb.mixf(is_glass, 0.28, 0.46)
    wv0 = nb.mixf(is_glass, 0.22, 0.12)
    wv1 = nb.mixf(is_glass, 0.84, 0.94)
    ax = nb.sstep(wu, nb.sub(wu, 0.04), nb.math('ABSOLUTE', nb.sub(fu, 0.5)))
    ay = nb.mul(nb.sstep(wv0, nb.add(wv0, 0.03), fv), nb.sstep(wv1, nb.sub(wv1, 0.03), fv))
    win = nb.mul(ax, ay)
    wall = nb.math('LESS_THAN', nb.math('ABSOLUTE', nz), 0.5)
    win = nb.mul(nb.mul(win, wall), nb.sub(1.0, is_trim))
    # per-window random numbers
    wn = nb.white(nb.comb(cu, cv, nb.mul(bid, 997.0)), dims='3D')
    r1 = nb.sep(wn.outputs['Color'])
    # floor bands: offices light whole floors
    fb = nb.white(nb.comb(cv, nb.mul(bid, 131.0), 0.5), dims='3D').outputs['Value']
    litp = nb.madd(nb.math('FRACT', nb.mul(bid, 17.3)), 0.35, 0.15)
    band_on = nb.math('GREATER_THAN', fb, nb.madd(nb.math('FRACT', nb.mul(bid, 29.1)), 0.35, 0.4))
    litp_glass = nb.madd(band_on, 0.82, 0.05)
    litp = nb.mixf(is_tower, litp, litp_glass)
    lit = nb.math('LESS_THAN', r1[0], litp)
    # colour temperature: mostly cool/neutral, some gently warm (kept desaturated)
    tcol = nb.mixcol(nb.sstep(0.55, 0.95, r1[1]), (0.72, 0.84, 1.0), (1.0, 0.93, 0.80))
    tcol = nb.mixcol(nb.math('GREATER_THAN', r1[1], 0.93), tcol, (1.0, 0.82, 0.62))
    br = nb.madd(nb.pw(r1[2], 2.5), 0.9, 0.12)
    br = nb.mul(br, nb.mixf(is_tower, 1.0, 0.7))
    # curtains / blinds: gradient + dim rooms
    curtain = nb.madd(nb.sstep(0.3, 0.9, fv), 0.5, 0.5)
    em_str = nb.mul(nb.mul(nb.mul(nb.mul(win, lit), br), curtain), nb.mixf(is_glass, 0.6, 0.95))
    dist = nb.camdata().outputs['View Distance']
    em_str = nb.mul(em_str, nb.madd(nb.math('MINIMUM', nb.div(dist, 1400.0), 2.0), 1.0, 1.0))
    em = nb.emission(tcol, em_str)
    top = nb.attr('btop').outputs['Fac']
    below_top = nb.sub(top, pz)
    crown = nb.mul(nb.mul(nb.sstep(5.0, 1.5, below_top), is_glass), nb.sub(1.0, is_trim))
    crown = nb.mul(crown, nb.math('GREATER_THAN', nb.math('FRACT', nb.mul(bid, 11.7)), 0.45))
    em = nb.addshader(em, nb.emission((0.78, 0.86, 1.0), nb.mul(nb.mul(crown, wall), 0.1)))
    # facade albedo: masonry warm-grey/brick-ish (dark) or glass panels
    fac_col = nb.mixcol(nb.math('FRACT', nb.mul(bid, 5.3)), (0.05, 0.043, 0.038), (0.1, 0.085, 0.07))
    fac_col = nb.mixcol(is_glass, fac_col, (0.018, 0.022, 0.028))
    rough = nb.mixf(win, nb.mixf(is_glass, 0.85, 0.3), 0.06)
    bs = nb.principled(Base_Color=nb.mixcol(win, fac_col, (0.005, 0.006, 0.008)), Roughness=rough)
    bs.inputs['Specular IOR Level'].default_value = 0.5
    # uplight from the street (lower floors catch street light) + faint roof edge
    up = nb.exp(nb.div(v, -14.0))
    upl = nb.emission(nb.colscale(fac_col, 1.0), nb.mul(nb.mul(up, 6.0), nb.sub(1.0, win)))
    sh = nb.addshader(nb.addshader(bs, em), upl)
    nb.output(surface=C.fogged(nb, sh))
    return m


def _young_person(C, FG, FK, cam, T, fire_base):
    """The hooded young person: per-frame SDF body meshes (venv) swapped in at render time; the torch,
    its flame card and light are keyed from the venv's grip/direction table."""
    from mathutils import Matrix, Vector
    root = Matrix.Translation(Vector(FEET)) @ Matrix.Rotation(math.radians(-FIG_YAW), 4, 'Z')
    mat = FG.figure_material('kid', [(0.011, 0.011, 0.013), (0.009, 0.010, 0.015), (0.02, 0.013, 0.010),
                                     (0.03, 0.03, 0.032)], sheen=0.35)
    FG.MeshSeq('kid', T['kid_dir'], mat, root, START - 1)
    torch = FG.torch_obj('torch')
    heads = []
    for f in range(START - 1, END + 2):
        grip, tdir = T['kid_torch'][str(f)]
        g = root @ Vector(grip)
        d = (root.to_3x3() @ Vector(tdir)).normalized()
        torch.location = g
        torch.rotation_quaternion = d.to_track_quat('Z', 'Y')
        torch.keyframe_insert('location', frame=f)
        torch.keyframe_insert('rotation_quaternion', frame=f)
        heads.append((f, g + d * 0.6))
    tf = FK.flame_card('torch_flame', heads[0][1], T['sprites']['torch']['card'], T['sprites']['torch']['first'], cam)
    tl = C.point('torch_L', heads[0][1], C.FIRE_LIGHT, 0.0, radius=0.06, cutoff=40.0)
    tl_tab = {f: v for f, v in T['torch']}
    for f, P in heads:
        tf.location = P + Vector((0, 0, -0.02))
        tf.keyframe_insert('location', frame=f)
        tf.scale = (1, 1, 1) if f < RELEASE + 1 else (0.001, 0.001, 0.001)
        tf.keyframe_insert('scale', frame=f)
        tl.location = P + Vector((0, 0, 0.3))
        tl.keyframe_insert('location', frame=f)
        C.key(tl.data, 'energy', f, 35.0 * tl_tab.get(f, 1.0) * (1.0 if f < IGN else 0.0))
    return dict(torch_head=[[f, list(P)] for f, P in heads])


def per_frame(job, f):
    from kit import figure as FG
    FG.swap_all(f)
