"""Opt-in cut-D returns to A2: the race, two promises, watch, and true dawn.

The accepted route delegates to falsedawn.render without changing its globals.
Variant frames are absolute cut-D frames. Musical phase is (D frame + beat_offset);
80 frames per 4/4 bar at 24 fps / 72 BPM. Camera is fixed at A500's pose.
"""
from dataclasses import dataclass
from functools import lru_cache
import math
from types import SimpleNamespace

import numpy as np
from numba import njit, prange

import falsedawn as FD
import nighta as NA
import dawn as DA

PI, WD, SK = FD.PI, FD.WD, FD.SK
FINISH = FD.FINISH
RANGES = {'brink': (4080, 4239), 'twofires': (4560, 5039),
          'watch': (7040, 7359), 'truedawn': (7360, 7839)}
LENGTHS = {name: end-start+1 for name, (start, end) in RANGES.items()}
FPS, BPM, BAR, BEAT = 24, 72, 80, 20
# D bar n begins at (n-1)*80. Both racing returns surge every quarter note.
BAR_MAP = {name: ((0, 1),) for name in RANGES}
# Final native basket-root centres, measured on A2's existing visible summits.
# The crowns shot matches these anchors; flame centroids move with the fire.
CROWN_LEFT_BASE = (733.4482365330008, 463.0645838537898)
CROWN_RIGHT_BASE = (1386.8711906600877, 455.13424081989217)
_STARS = None
SIGNAL_HAZE_EXPONENT = 0.35


@dataclass(frozen=True)
class Config:
    beat_offset: int = 0
    pair_frame: int = 4320
    cascade_frame: int = 4760
    camera_frame: int = 420

    def __post_init__(self):
        for key in ('beat_offset', 'pair_frame', 'cascade_frame', 'camera_frame'):
            value = getattr(self, key)
            if not isinstance(value, (int, np.integer)):
                raise ValueError(f'{key} must be an integer frame')
        if self.camera_frame != 420:
            raise ValueError('The crown match fixes the variant camera at A500 (420)')
        if not 0 <= self.pair_frame < self.cascade_frame:
            raise ValueError('pair_frame must precede cascade_frame')


def ease(u):
    u = min(max(float(u), 0.0), 1.0)
    return u * u * (3.0 - 2.0 * u)


def beat_clock(frame, variant='brink', offset=0):
    """Surge peaks on absolute D beats; offset is added, not subtracted."""
    if variant not in BAR_MAP:
        raise ValueError(f'Unknown glow variant: {variant}')
    subdivision = 1
    for bar, divisions in BAR_MAP[variant]:
        if frame >= bar * BAR:
            subdivision = divisions
    phase = ((frame + offset) % (BEAT / subdivision)) / (BEAT / subdivision)
    return ((1.0 + math.cos(2.0 * math.pi * phase)) * 0.5) ** 4


def state(frame, variant, config=None):
    c = config or Config()
    if variant not in LENGTHS:
        raise ValueError(f'Unknown glow variant: {variant}')
    if not RANGES[variant][0] <= frame <= RANGES[variant][1]:
        raise ValueError(f'Frame outside {variant}: {frame}')
    pulse = beat_clock(frame, variant, c.beat_offset)
    breath = 0.5 + 0.5 * math.cos(2 * math.pi * (frame + c.beat_offset) / BAR)
    settle = float(variant in ('watch', 'truedawn'))
    dawn = ease((frame-RANGES['truedawn'][0]) / (LENGTHS['truedawn']-1)) if variant == 'truedawn' else 0.0
    return dict(pulse=pulse, settle=settle, dawn=dawn,
                glow=(0.16 + 0.84 * pulse) * (1 - settle) + (0.14 + 0.018 * breath) * settle,
                underglow=(0.25 + 8.0 * pulse) * (1 - settle) + (0.16 + 0.014 * breath) * settle)


def ignition_frames(count, config=None):
    c = config or Config()
    if count < 2:
        raise ValueError('The first promise requires two beacons')
    # Keep catch events on the same bar clock when the owner supplies an offset.
    first = c.cascade_frame + (-(c.cascade_frame + c.beat_offset)) % (2 * BEAT)
    return np.array([c.pair_frame, c.pair_frame] + [first + (k - 2) * 2 * BEAT for k in range(2, count)], dtype=int)


def fires_table(variant, config=None):
    from glowvars_sites import sites, watch_sites
    c = config or Config()
    place = watch_sites if variant in ('watch', 'truedawn') else sites
    xyz = place()
    ign = ignition_frames(len(xyz), c)
    if variant in ('watch', 'truedawn'):
        # The chain persists; the remaining ridges are alight by D's EVERY RIDGE.
        ign[7:] = 5040
    # These are the existing great pyres, with the last near watch-fire smaller.
    sizes = np.full(len(xyz), 2.7)
    sizes[6] = 1.7
    return np.column_stack([xyz, ign, sizes, np.arange(len(xyz)) + 810])


def curved(pos, camera_pos):
    p = np.asarray(pos, dtype=float).copy()
    p[1] -= ((p[0] - camera_pos[0]) ** 2 + (p[2] - camera_pos[2]) ** 2) / (2 * WD.R_EARTH)
    return p


def fire_height_px(row, camera):
    """Nominal native flame height; preserve canonical pyre shape and motion.

    The first pair carries the match cut. Subsequent chain fires grow nearer;
    additional watch fires follow distance. This scales their screen image,
    not the physical size that drives smoke or the cleared rock platform.
    """
    index = int(row[5]) - 810
    if index < 2:
        return 60.0
    if index < 7:
        # Preserve accepted sizes after deferring the low caption-band ridge.
        return (20.0, 24.0, 32.0, 36.0, 40.0)[index-2]
    distance = np.linalg.norm(row[:3] - camera.pos)
    return 12.0 + 14.0*(1.0-ease((distance-3500.0)/18000.0))


def terrain_lights(fires, frame, camera, fogp):
    """Existing inverse-square light, with a roughly ten-native-pixel pool.

    Flame magnification must not multiply the shared kit's physical size (and
    therefore its smoke and light intensity). Pool extent and intensity are
    controlled separately; the existing terrain shader supplies snow, normals,
    rock colour and fog. Inverse-square tails are not a hard screen mask.
    """
    rows = []
    native_f = camera.f * 1920.0 / camera.W
    for fire in fires:
        pos = curved(fire[:3], camera.pos)
        depth = camera.project(pos)[2]
        if depth <= 0 or frame < fire[3]:
            continue
        _, _, light = NA._ign(frame, fire[3])
        radius = (8.0 + fire_height_px(fire, camera)*0.10)*depth/native_f
        height, soft = 0.30*radius, 0.40*radius
        trans = NA.fire_trans(camera.pos, fire[:3], fogp)
        strength = 2.50 / max(trans, 0.12)**0.65
        intensity = strength*(height*height+soft*soft)*light*NA.F.flicker(frame/FPS, int(fire[5])+3)
        rows.append([pos[0], pos[1]+height, pos[2], *NA.F.FIRE_LIGHT, intensity, soft])
    return np.asarray(rows, dtype=float).reshape(-1, 8)


@lru_cache(maxsize=32)
def smoke_for(row):
    """The night kit's smoke recipe, keyed by its complete construction state.

    nighta._smoke's shared key omits ignition time; reusing it would give a
    dawn rendered after twofires the twofires smoke start. Keep this cache local.
    """
    x, y, z, ignition, size, seed = row
    height = NA.fire_dims(size)[0]
    big = size > 1.8
    return NA.F.Smoke(100 + int(seed), np.array([x, y + .65*height, z]), ignition/FD.FPS + .2,
                      9999., rate=3. if big else 2.5, wind=(1.6, 0., .4),
                      rise=2.2 if big else 1.5, life=6., r0=.9 if big else .4,
                      growth=1.1 if big else .6, dens=.45)


def pair_camera(source, pos, nominal, pxs):
    """Project the fire kit's local metres about the pinned basket root.

    Local +x is screen right, +y is up. Smoke and sparks therefore retain
    canonical metre-scale simulation while their screen scale matches the
    enlarged flame. Depth stays at the real summit for terrain occlusion.
    """
    bx, by, depth = source.project(pos)
    ppm = nominal*pxs/NA.fire_dims(2.7)[0]

    def project(points):
        points = np.asarray(points, dtype=float)
        return bx + points[..., 0]*ppm, by - points[..., 1]*ppm, depth + points[..., 2]

    return SimpleNamespace(f=ppm*depth, W=source.W, project=project)


@lru_cache(maxsize=2)
def pair_tongues(seed):
    """Separate upper tongues across the basket instead of one tapered crown."""
    table = NA.F.tongue_table(7, int(seed)*7+3, spread=1.05)
    table[:, 1] = [.74, .93, .80, 1.0, .82, .95, .76]
    table[:, 2] *= .95
    table.setflags(write=False)
    return table


@lru_cache(maxsize=8)
def pair_effects(seed, ignition):
    """Existing smoke and particle solvers, with a shared rightward wind."""
    hf = NA.fire_dims(2.7)[0]
    start, end = ignition/FPS, (RANGES['truedawn'][1]+1)/FPS + 5.0
    smoke = NA.F.Smoke(100+int(seed), np.array([0., .70*hf, 0.]), start+.2, end,
                       rate=7., rise=2.0, wind=(.9, 0., 0.), life=5.0,
                       r0=.24, growth=.32, dens=.45, jitter=.12)
    sparks = NA.F.Sparks(500+int(seed), np.array([0., .45*hf, 0.]), start, end,
                         burst=0, rate=7., ember_rate=2., speed=(1.7, 4.0),
                         spread=.22, life=(.6, 1.4), ember_life=(1.5, 2.8),
                         wind=(.9, 0., 0.), buoy=6.5, updraft_h=hf,
                         drag=1.6, radius=.30, I=12., turb=.35, curl=.35)
    sparks.T0 *= .80
    return smoke, sparks


@njit(cache=True)
def basket_pass(img, depth, bx, by, z, pxs, width, energy, zbias):
    """Small tapered iron cage at the pinned root; flame shows between bars."""
    x0, x1 = max(0, int(bx-width*.65)), min(img.shape[1], int(bx+width*.65)+1)
    y0, y1 = max(0, int(by-7*pxs)), min(img.shape[0], int(by+pxs)+1)
    for y in range(y0, y1):
        h = (by-y-.5)/pxs
        if h < 0 or h > 6:
            continue
        half = width*(.30+.20*h/6)
        for x in range(x0, x1):
            if depth[y, x] < z-zbias:
                continue
            u = (x+.5-bx)/half
            if abs(u) > 1:
                continue
            rim = h > 5.0 or h < .8
            bar = min(abs(u+.68), abs(u), abs(u-.68)) < .075 or abs(u) > .90
            if not rim and not bar:
                continue
            hot = .075*energy if rim else .025*energy
            img[y, x, 0] = .009+hot
            img[y, x, 1] = .010+hot*.42
            img[y, x, 2] = .012+hot*.10


def draw_fire(fr, physical, frame, fogp):
    projected = physical.copy()
    projected[:3] = curved(physical[:3], fr.src.pos)
    pos, ignition, size, seed = projected[:3], projected[3], projected[4], projected[5]
    tr = NA.fire_trans(fr.src.pos, physical[:3], fogp)
    depth = fr.src.project(pos)[2]
    # Art direction for unresolved signals: retain distance ordering but give
    # the plain-orange beacon the legibility of beacons.py's distant hot point.
    # The atmosphere retains A2's parameters; terrain_lights adds local spill.
    distant = ease((depth - 4000.) / 4000.)
    tr = tr ** (1.0 - distant*(1.0 - SIGNAL_HAZE_EXPONENT))
    is_pair = int(seed) in (810, 811)
    if 0 < depth < 4000 and not is_pair:
        sz, _, light = NA._ign(frame, ignition)
        height = NA.fire_dims(size)[0]*sz
        fl = NA.F.flicker(frame/FD.FPS, int(seed)+11, 1.3)
        smoke_for(tuple(projected)).render(
            fr.img, fr.zb, fr.src, frame/FD.FPS, pos + np.array([0., .8*height, 0.]),
            (2.5 if size > 1.8 else .8)*size*light*fl, albedo=.25,
            amb=(.004, .005, .009), zbias=NA.fire_depth_bias(depth))
    sz, intensity, light = NA._ign(frame, ignition)
    if sz <= 0 or depth <= 0:
        return
    pxs = PI.src_scale(fr)
    nominal = fire_height_px(physical, fr.src)
    hf, rb = NA.fire_dims(size)[:2]
    flicker = NA.F.flicker(frame/FPS, int(seed)+11, 1.3)
    energy = light*flicker*tr
    if is_pair:
        fx_camera = pair_camera(fr.src, pos, nominal, pxs)
        smoke, sparks = pair_effects(int(seed), int(ignition))
        smoke.render(fr.img, fr.zb, fx_camera, frame/FPS,
                     np.array([0., .65*hf, 0.]), 4.0*energy,
                     amb=(.007, .008, .011), albedo=.22,
                     zbias=NA.fire_depth_bias(depth), light_falloff=1.8)
    # fire2 reads only .f and .project: magnify the canonical seven tongues
    # about the real, depth-tested base without changing their noise speed.
    flame_camera = SimpleNamespace(f=nominal*pxs*depth/hf, project=fr.src.project)
    NA.F2.flame(fr.img, fr.zb, flame_camera, pos, hf*sz, rb*sz, frame/FPS,
                seed=int(seed)*7+3, I=(60.0 if is_pair else 30.0)*intensity*tr, lean=.25*sz,
                zbias=NA.fire_depth_bias(depth), tongues=7,
                TG=pair_tongues(int(seed)) if is_pair else None)
    x, y, _ = fr.src.project(pos)
    NA.F2.halo(fr.img, fr.zb, x, y-.35*nominal*pxs*sz, .22*nominal*pxs,
               .09*energy, z=depth, zbias=NA.fire_depth_bias(depth))
    NA.F2.halo(fr.img, fr.zb, x, y-.25*nominal*pxs*sz, .62*nominal*pxs,
               .009*energy, z=depth, zbias=NA.fire_depth_bias(depth), col=NA.F2.AURA_COL)
    if is_pair:
        basket_pass(fr.img, fr.zb, x, y, depth, pxs, 2.2*rb*nominal*pxs/hf,
                    energy, NA.fire_depth_bias(depth))
        sparks.render(fr.img, fr.zb, fx_camera, frame/FPS, gain=tr,
                      zbias=NA.fire_depth_bias(depth), max_len_px=6., res_scale=pxs)


def underglow_rows(s):
    """A13/A14's actual patch catalogue and cloud-glow shader, on our clock."""
    table = NA.ug_table()
    rows = np.zeros((len(table), 8))
    rows[:, :3] = table[:, :3]
    col = np.array([1.0, 0.10, 0.015]) * (1 - s['settle']) + np.array([1.0, 0.48, 0.16]) * s['settle']
    rows[:, 3:6] = table[:, 3, None] * col * s['underglow']
    rows[:, 6:] = [900.0, 0.35]
    return rows


def dawn_parameters(k, cam):
    """Reuse dawn.dawn_sky; the sun clears this view's own terrain skyline."""
    a = math.radians(FD.GLOW_AZ)
    r = np.geomspace(200.0, 90000.0, 900)
    x, z = cam.pos[0] + np.sin(a) * r, cam.pos[2] + np.cos(a) * r
    h = np.zeros(len(r))
    WD.heights(x, z, 20.0, FD.terrain_rows(), h)
    horizon = np.max(np.arctan2(np.maximum(h, WD.CLOUD_Y) - r*r/(2*WD.R_EARTH) - cam.pos[1], r))
    el = horizon + math.radians(-6.3 + 7.0 * k)
    direction = np.array([math.cos(el) * math.sin(a), math.sin(el), math.cos(el) * math.cos(a)])
    sd = np.zeros(21)
    sd[:3] = direction
    sd[3:6] = [1.0, 0.52, 0.29]
    sd[6:8] = [90.0, math.radians(0.27)]
    sd[8:11] = FD.lin('#17285F') * (0.65 + 0.25 * k)
    sd[11:14] = FD.lin('#E39A86') * (0.08 + 0.80 * k)
    sd[14:17] = FD.lin('#4E5B8E') * (0.42 + 0.25 * k)
    sd[17:21] = [0.05 + 0.12*k, 0.01, 0.03 + 0.02*k, 0.55]
    return sd


@njit(parallel=True, cache=True)
def dawn_pass(img, dist, trans, C, SD, mix):
    """Dawn luminance veils the existing stars; no threshold or star-count cut."""
    for j in prange(img.shape[0]):
        for i in range(img.shape[1]):
            if dist[j, i] < 1e8:
                continue
            xo, vy = i + 0.5 - C[8], C[9] - (j + 0.5)
            dxh, dzh = C[3]*C[7] + C[5]*xo, C[4]*C[7] + C[6]*xo
            nn = math.sqrt(dxh*dxh + dzh*dzh + vy*vy)
            r, g, b = DA.dawn_sky(dxh/nn, vy/nn, dzh/nn, SD, 1.0/C[7])
            img[j, i, 0] = img[j, i, 0]*(1-mix) + r*mix
            img[j, i, 1] = img[j, i, 1]*(1-mix) + g*mix
            img[j, i, 2] = img[j, i, 2]*(1-mix) + b*mix
            trans[j, i] *= math.exp(-mix*(0.2126*r + 0.7152*g + 0.0722*b)/0.002)


def lighting(frame, variant, c, cam):
    s = state(frame, variant, c)
    lt, gp = FD.light(c.camera_frame, 'arc')
    lt = tuple(q.copy() for q in lt)
    gp = FD.sky_parameters(gp, 'clear_high_deck')
    gp[2] = s['glow']
    gp[3:6] = gp[3:6]*(1-s['settle']) + np.array([1.0, 0.65, 0.38])*s['settle']
    sd = None
    if s['dawn'] > 0:
        k = s['dawn']
        sd = dawn_parameters(k, cam)
        Lk, amb, S, fogp, Q = lt
        # The blue-hour palette, with the key kept on the glow's -30° azimuth.
        el = math.radians(14.0)
        az = math.radians(FD.GLOW_AZ)
        # Interpolate from the watch's exact lighting. At k=1 the approved
        # Round-1 rose dawn and sun position are retained.
        direction = np.array([math.cos(el)*math.sin(az), math.sin(el), math.cos(el)*math.cos(az)])
        Lk[:3] = Lk[:3]*(1-k) + direction*k
        Lk[:3] /= np.linalg.norm(Lk[:3])
        Lk[3:] = Lk[3:]*(1-k) + (FD.lin('#F0B4A0')*(0.55+0.45*k) + FD.lin('#8C9AD0')*.45*(1-k))*k
        amb[:] = amb*(1-k) + FD.lin('#5B6FAA')*(0.10+0.36*k)*k
        Q[0] = Q[0]*(1-k) + (0.05 + 0.20*k)*k
        Q[7] = Q[7]*(1-k) + 0.04*k
        Q[13] = Q[13]*(1-k) + 2.8*k
        fogp[5:8] = fogp[5:8]*(1-k) + FD.lin('#9989A8')*(0.08+0.16*k)*k
    return lt, gp, sd, s


def render(frame, variant='accepted', scale=1.0, ss=1.5, config=None):
    """Linear HDR. Omitted variant is the accepted clear-high-deck A2 path.

    Accepted frames retain A2 shot numbering (A500 = 420); opted-in variants
    use absolute cut-D numbering. No state in falsedawn is patched.
    """
    global _STARS
    if variant == 'accepted':
        return FD.render(frame, design='arc', scale=scale, ss=ss, sky_candidate='clear_high_deck')
    c = config or Config()
    state(frame, variant, c)  # reject before terrain work
    if not 0 < scale <= 1 or not 0 < ss <= 2 or min(round(804*scale), round(round(804*scale)*ss)) < 1:
        raise ValueError('scale must be in (0, 1] and ss in (0, 2]')
    cam = FD.camera(c.camera_frame, round(1920*scale), round(804*scale))
    fr = PI.Frame(cam, ss)
    lt, gp, sd, s = lighting(frame, variant, c, cam)
    fires = np.zeros((0, 6)) if variant == 'brink' else fires_table(variant, c)
    scam, cr = fr.src, FD.terrain_rows()
    C = scam.params()
    # Hold the cloud time at A500. This preserves the matching cloud shapes;
    # flicker, sky extinction and every fire advance on the shot clock.
    p = np.array([scam.pos[0], scam.pos[2], c.camera_frame/FD.FPS, 0.0])
    D = np.zeros((scam.H, scam.W))
    WD.march(p, cr, C, .2, 90000., .0035, .35, FD.terrain_hmax(), 9, D)
    if len(fires):
        from glowvars_sites import seat_on_raster
        fires = fires.copy()
        fires[:, :3] = seat_on_raster(scam, fires[:, :3], D)
    fr.img = np.zeros((scam.H, scam.W, 3), np.float32)
    fr.zb = np.zeros(D.shape, np.float32)
    fr.dist = np.zeros(D.shape, np.float32)
    # Both lights and platform y are compared with curved surface y in
    # world.shade. Exclude the ignition envelope's pre-catch pickup.
    projected_fires = fires.copy()
    for row in projected_fires:
        row[:3] = curved(row[:3], scam.pos)
    lit = fires[fires[:, 3] <= frame]
    WD.shade(C, D, p, cr, lt[2], terrain_lights(lit, frame, scam, lt[3]), lt[0], lt[4], lt[1], lt[3],
             fr.img, fr.zb, fr.dist, NA.pl_rows(projected_fires))
    WD.cloud_glow(C, D, p, cr, underglow_rows(s), lt[3], fr.img)
    # Keep variant caches out of falsedawn: a variant rendered first must not
    # seed the accepted shot's single-position skyline cache.
    skl0, skld, skl = NA.skyline(scam.pos, cr)
    haze = gp.copy()
    haze[11] = 0.0
    FD.glow_haze(fr.img, fr.dist, C, haze, skl0, skld, skl, .25, 30000.)
    trans = np.zeros(D.shape, np.float32)
    FD.sky_pass(fr.img, fr.dist, trans, C, gp, FD.milky_way(), skl0, skld, skl, scam.pos[1],
                (c.camera_frame+frame)/FD.FPS, np.zeros((0, 6)))
    if sd is not None:
        dawn_pass(fr.img, fr.dist, trans, C, sd, s['dawn'])
    if _STARS is None:
        _STARS = SK.make_stars(16000, 101, lum_scale=6.0)
    SK.splat_stars(fr.img, scam, _STARS, trans, t=(c.camera_frame+frame)/FD.FPS,
                   gain=ss*ss, scale=PI.src_scale(fr))
    for fire in sorted(fires, key=lambda f: -np.linalg.norm(f[:3]-scam.pos)):
        if frame < fire[3]:
            continue
        # Curvature belongs in projection exactly once; fog sees the physical site.
        draw_fire(fr, fire, frame, lt[3])
    return PI.to_target(fr)[0]
