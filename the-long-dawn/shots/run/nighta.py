"""A's night kit: one moonlit night for A13 R2-A THE REVEAL, A14 R3 THE BEACON RUN and A15 R16 THE WATCHERS
(RUN-A3 wrote it for RUN-A-L and itself; A18's crossing may adopt the under-glow table). Every A-night shot draws
these pieces from here, so the glow, the red under the cloud and the fires are literally the same in every shot.

  * LIGHT: moonlit, `world.night_light()` (s1's moon, az -59 el 21), as H1's take. A2's moonless glow key does not
    come back after the vision.
  * THE COLD GLOW: FALSE DAWN's arc (`falsedawn.glow_at` with `glow_params('arc', I)`) at az -30, el_s -3.2, with no
    deck, added over the moonlit sky; A2's star kill; a faint share of it in the far haze. I(f) = I0 * (1 + 0.07 sin(2
    pi f / 80)): A2's own breath, once a bar, phase-locked to the bar line (f = A cut frame).
  * THE RED UNDER THE CLOUD: one world-fixed patch table (seed UG_SEED) for `world.cloud_glow`, in crossing's
    orange-red (pure crimson over blue moonlit cloud reads magenta), pulsing on every beat (onsets on the cut frames
    with f % 20 == 0): a 2-frame rise, an exp(-dt / 9 f) decay, a floor of 0.55, so it throbs and never blinks off.
  * FIRES: beacons.py's recipe (the fire2 flame, two air-glow halos, a hot point and an orange aura once it is too far
    to resolve, smoke when near, a warm LT pool and a cleared platform on the terrain), one ignition envelope
    (mt.fire.ignite_env), and H5 sparks (short, orange, falling and curving) for a near catch.
    A fire row is (x, y, z, f_ign, size, seed): (x, y, z) the base on the ground (world), f_ign the A cut frame it
    catches, size 1.0 = a watch-fire on a cairn (a 2.4 m flame), about 2.7 = a great pyre.

Everything takes A CUT FRAMES (f), never seconds; t = f / 24 inside. Nothing here edits world.py: the world module is
passed in (`world` itself, or HILLS' generated `world_keep`, for H1's summit).

    import nighta as NA
    GP = NA.glow_gp(f); skl = NA.skyline(cam_pos, CR, WD)          # once per shot (or per camera)
    NA.glow_pass(img, dist, kill, C, GP, *skl, NA.HAZE_K, NA.HAZE_D)  # after world.shade, before the stars
    WD.cloud_glow(C, D, P, CR, NA.ug_rows(f), fogp, img)            # the red under the cloud
    LT = NA.lt_rows(FIRES, f); PL = NA.pl_rows(FIRES)               # into world.shade
    NA.fires_layer(img, zb, scam, FIRES, f, pxs)                    # the flames, after the stars
"""
import math
import os
import sys

import numpy as np
from numba import njit, prange

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import world as WD          # noqa: E402  (only for its constants and height queries; shading uses the caller's module)
import fire2 as F2          # noqa: E402
import fire_near_a as FN    # noqa: E402
from falsedawn import glow_at as _glow_at, glow_params as _fd_glow_params   # noqa: E402  (RUN-A-L's; read-only)
from mt import fire as F    # noqa: E402

FPS = 24.0
BEAT = 20
BAR = 80

# ------------------------------------------------------------------ the light: one moonlit night ---
# world.shade knobs by key: 'qN' -> Q[N], 'fN' -> fogp[N]. TERRAIN = the crossing's fixes (A14/A15: anti-streak snow
# noise, snow held on steeper ground, more aerial depth). CLOUD = the shared moonlit cloud sea (RUN-A-L tunes it on
# A14's frames; A2 and A13 use the same): q5 shadow floor | q8 trough dark | q9 albedo | q13 ambient gain | q15
# self-shadow | f2, f3 cloud-top mist density and height scale. Starts at world.night_light's own values.
# Look-dev override from a farm job: NIGHT_CLOUD="q8=0.75,q5=0.2,f2=3e-4" (any qN / fN key).
TERRAIN = {'q18': 1.0, 'q3': 0.30, 'q4': 0.54, 'f0': 1.0e-4}
_L0 = WD.night_light()
CLOUD = {'q5': float(_L0[4][5]), 'q8': float(_L0[4][8]), 'q9': float(_L0[4][9]), 'q13': float(_L0[4][13]),
         'q15': float(_L0[4][15]), 'f2': float(_L0[3][2]), 'f3': float(_L0[3][3])}


def cloud_values():
    """CLOUD with the NIGHT_CLOUD env override applied."""
    out = dict(CLOUD)
    for part in os.environ.get('NIGHT_CLOUD', '').split(','):
        if '=' in part:
            k, v = part.split('=', 1)
            out[k.strip()] = float(v)
    return out


def apply_knobs(Q, fogp, knobs, u=1.0):
    """Q, fogp (copies) moved a fraction u of the way to the knob values."""
    Q = np.array(Q, np.float64, copy=True)
    fogp = np.array(fogp, np.float64, copy=True)
    for k, v in knobs.items():
        arr = Q if k[0] == 'q' else fogp
        i = int(k[1:])
        arr[i] = arr[i] + (float(v) - arr[i]) * u
    return Q, fogp


def night_light():
    """(Lk, amb, S, fogp, Q) of A's moonlit night: world.night_light + TERRAIN + CLOUD (+ NIGHT_CLOUD)."""
    Lk, amb, S, fogp, Q = WD.night_light()
    Q, fogp = apply_knobs(Q, fogp, TERRAIN)
    Q, fogp = apply_knobs(Q, fogp, cloud_values())
    return Lk, amb, S, fogp, Q


# ------------------------------------------------------------------ the cold glow ---
I0 = 0.14                     # the director's one knob for the glow in A13-A15
GLOW_AZ = -30.0
EL_S = -3.2
HAZE_K = 0.35                 # the glow's share in the far haze over terrain and cloud (0 = sky only)
HAZE_D = 30000.0              # metres: how fast that share grows with distance


def glow_I(f):
    return I0 * (1.0 + 0.07 * math.sin(2.0 * math.pi * f / BAR))


def glow_gp(f, shadow_rays=True):
    """falsedawn's arc at this frame's intensity, with no deck (GP[16] = 0) and A2's star kill (GP[17])."""
    GP = _fd_glow_params('arc', glow_I(f))
    GP[0] = math.radians(GLOW_AZ)
    GP[1] = math.radians(EL_S)
    GP[16] = 0.0
    if not shadow_rays:
        GP[11] = 0.0
    return GP


_SKL = {}


def skyline(cam_pos, CR, wmod=WD, fp=20.0):
    """(SKL0, SKLD, SKL): the skyline's elevation (rad) every 0.1 deg over GLOW_AZ +-90 deg seen from cam_pos, for the
    glow's shadow rays (the far notches cast them). Cached per camera position to the nearest 50 m."""
    key = (id(wmod), CR.shape[0], tuple(np.round(np.asarray(cam_pos, np.float64) / 50.0).astype(int)))
    if key not in _SKL:
        azs = np.radians(np.arange(GLOW_AZ - 90.0, GLOW_AZ + 90.0001, 0.1))
        rs = np.geomspace(300.0, 90000.0, 420)
        A, Rr = np.meshgrid(azs, rs, indexing='ij')
        X = cam_pos[0] + np.sin(A) * Rr
        Z = cam_pos[2] + np.cos(A) * Rr
        h = np.zeros(X.size)
        wmod.heights(X.ravel().copy(), Z.ravel().copy(), fp, CR, h)
        h = h.reshape(X.shape)
        hc = np.maximum(h, WD.CLOUD_Y)
        el = np.arctan2(hc - Rr * Rr / (2 * WD.R_EARTH) - cam_pos[1], Rr)
        from scipy.ndimage import gaussian_filter1d
        _SKL[key] = (float(azs[0]), 0.1 * math.pi / 180.0, gaussian_filter1d(el.max(axis=1), 5.0))
    return _SKL[key]


@njit(parallel=True, fastmath=True, cache=True)
def glow_pass(img, dist, kill, C, GP, SKL0, SKLD, SKL, haze_k, haze_d):
    """Add the cold glow to a shaded frame in a lens-shift camera (C = SrcCam/mt.cam params): over sky pixels in
    full, over terrain and cloud as a share of the far haze. kill <- the star transmittance (A2's star kill) on sky
    pixels, 0 elsewhere."""
    H, W = img.shape[0], img.shape[1]
    f, cx, cyy = C[7], C[8], C[9]
    # the haze sees the smooth arch only: under the horizon every pixel shares el = 0, so the shadow rays (GP[11])
    # would run straight down each column as hard vertical slabs over the far terrain (RUN-A-L's find in A2)
    GPh = GP.copy()
    GPh[11] = 0.0
    for j in prange(H):
        for i in range(W):
            xo = i + 0.5 - cx
            dxh = C[3] * f + C[5] * xo
            dzh = C[4] * f + C[6] * xo
            vy = cyy - (j + 0.5)
            nn = math.sqrt(dxh * dxh + dzh * dzh + vy * vy)
            dx, dy, dz = dxh / nn, vy / nn, dzh / nn
            d = dist[j, i]
            if d >= 1e8:
                G = _glow_at(dx, dy, dz, GP, SKL0, SKLD, SKL)
                img[j, i, 0] += G * GP[3]
                img[j, i, 1] += G * GP[4]
                img[j, i, 2] += G * GP[5]
                kill[j, i] = math.exp(-G / GP[17])
            else:
                kill[j, i] = 0.0
                hz = haze_k * (1.0 - math.exp(-d / haze_d))
                if hz > 1e-4:
                    # the haze sees the glow at the horizon's elevation (the arch's foot), not the ground's
                    G = _glow_at(dx, max(dy, 0.0), dz, GPh, SKL0, SKLD, SKL)
                    img[j, i, 0] += G * GP[3] * hz
                    img[j, i, 1] += G * GP[4] * hz
                    img[j, i, 2] += G * GP[5] * hz


# ------------------------------------------------------------------ the red under the cloud ---
UG_SEED = 4613
UG_COL = np.array([1.0, 0.34, 0.07])          # crossing's orange-red
UG_GAIN = 1.4                                 # crossing's k
UG_FLOOR = 0.55
UG_RISE = 2.0                                 # frames
UG_DECAY = 9.0                                # frames
# where the race burns under the deck: the cloud sea north of the Run, toward the glow (world x, z box, metres)
UG_BOX = (-62000.0, 22000.0, -8000.0, 96000.0)
UG_N = 150
_UG = None


def ug_table():
    """World-fixed patches: (x, z, radius, gain, beat delay (frames), noise phase). Uniform over UG_BOX, thinned away
    from the glow's half of the sky as seen from the Run's middle (so every A view toward az -30 has a few within
    14-48 km), dropped where a range stands over the patch centre (the deck is not there)."""
    global _UG
    if _UG is None:
        rng = np.random.default_rng(UG_SEED)
        x0, x1, z0, z1 = UG_BOX
        rows = []
        ga = math.radians(GLOW_AZ)
        centre = np.array([-2700.0, 16000.0])            # between the A2 ridge and her summit
        tries = 0
        while len(rows) < UG_N and tries < 20000:
            tries += 1
            x = rng.uniform(x0, x1)
            z = rng.uniform(z0, z1)
            a = math.atan2(x - centre[0], z - centre[1])
            da = math.atan2(math.sin(a - ga), math.cos(a - ga))
            if rng.random() > 0.35 + 0.65 * math.exp(-(da / math.radians(70.0)) ** 2):
                continue
            if WD.ground(x, z, np.zeros((0, WD.NCR)), fp=40.0) > WD.CLOUD_Y - 20.0:
                continue
            rows.append([x, z, rng.uniform(1400.0, 3400.0), rng.uniform(0.6, 1.0), rng.uniform(0.0, 1.5),
                         rng.uniform(0.0, 6.28)])
        _UG = np.array(rows)
    return _UG


def ug_pulse(f, delay=0.0):
    """The race's beat: 0..1 above the floor. Onsets on cut frames f % 20 == 0 (+ a sub-frame delay per patch)."""
    u = (f - delay) % BEAT
    if u < UG_RISE:
        e = u / UG_RISE
        # the tail of the previous beat still decays under the rise
        prev = math.exp(-(u + BEAT - UG_RISE) / UG_DECAY)
        return max(e, prev)
    return math.exp(-(u - UG_RISE) / UG_DECAY)


def ug_rows(f, gain=1.0):
    """UG rows for world.cloud_glow at A cut frame f: (x, z, radius, r, g, b, noise scale, trough bias)."""
    T = ug_table()
    out = np.zeros((T.shape[0], 8))
    col = UG_COL * UG_GAIN * gain
    for k in range(T.shape[0]):
        x, z, rad, g, dly, ph = T[k]
        I = g * (UG_FLOOR + (1.0 - UG_FLOOR) * ug_pulse(f, dly))
        out[k] = [x, z, rad, col[0] * I, col[1] * I, col[2] * I, 900.0, 0.35]
    return out


# ------------------------------------------------------------------ fires ---
def _ign(f, f_ign):
    return F.ignite_env(f / FPS, f_ign / FPS)


POOL_I = 6.0                  # the warm pool a fire throws on the snow (world.shade LT intensity at size 1): a few metres
                              # round a fire are warm, beyond that the moon holds (14 lit a whole ledge like clay)


def fire_dims(size):
    """(flame height m, base half-width m, halo peak, point energy, aura peak, LT intensity, LT radius, platform r):
    size 1 = a watch-fire on a cairn (beacons' cairn), size 2.7 = a great pyre (beacons' pyre), smooth between."""
    s = max(size, 0.05)
    k = min(max((s - 1.0) / 1.7, 0.0), 1.0)
    return (2.4 * s, 0.42 * s ** 0.85, 0.45 + 0.55 * k, 3.5 + 3.0 * k, 0.08 + 0.05 * k, POOL_I * s ** 1.6,
            0.5 + 0.3 * k, 2.8 + 1.7 * k)


def lt_rows(fires, f):
    """LT rows (x, y, z, r, g, b, I, r0) for world.shade: the warm pool each lit fire throws on the snow."""
    LT = []
    for x, y, z, f_ign, size, seed in np.asarray(fires, np.float64).reshape(-1, 6):
        sz, inten, light = _ign(f, f_ign)
        if light <= 0:
            continue
        Hf, Rb, pk, pe, ak, I, r0, _ = fire_dims(size)
        fl = F.flicker(f / FPS, int(seed) + 3)
        LT.append([x, y + 0.55 * Hf, z, F.FIRE_LIGHT[0], F.FIRE_LIGHT[1], F.FIRE_LIGHT[2], I * light * fl, r0])
    return np.array(LT, np.float64).reshape(-1, 8)


def pl_rows(fires, max_dist=None, cam_pos=None):
    """PL rows (x, y, z, radius) for world.shade: a cleared rock platform under every fire."""
    out = []
    for x, y, z, f_ign, size, seed in np.asarray(fires, np.float64).reshape(-1, 6):
        if max_dist is not None and cam_pos is not None and math.hypot(x - cam_pos[0], z - cam_pos[2]) > max_dist:
            continue
        out.append([x, y, z, fire_dims(size)[7]])
    return np.array(out, np.float64).reshape(-1, 4)


_SMOKE = {}


def _smoke(pos, f_ign, size, seed):
    key = (round(pos[0], 1), round(pos[2], 1), int(seed))
    if key not in _SMOKE:
        Hf = fire_dims(size)[0]
        big = size > 1.8
        _SMOKE[key] = F.Smoke(100 + int(seed), np.array([pos[0], pos[1] + 0.65 * Hf, pos[2]]),
                              f_ign / FPS + 0.2, 9999.0, rate=3.0 if big else 2.5, wind=(1.6, 0.0, 0.4),
                              rise=2.2 if big else 1.5, life=6.0, r0=0.9 if big else 0.4,
                              growth=1.1 if big else 0.6, dens=0.45)
    return _SMOKE[key]


def fire_depth_bias(z, near_hearth=False):
    """Metres of source-depth tolerance; close A14/A15 stones and cowls must occlude flame."""
    if near_hearth and z < 60.0:
        return max(0.025, min(0.08, 0.001 * z))
    return max(3.0, 0.004 * z)


def fire_layer(img, zb, scam, pos, f_ign, f, size=1.0, seed=0, pxs=1.0, smoke=True, trans=1.0,
               near_hearth=False, base_offset=None):
    """Draw one fire in a lens-shift SOURCE camera (verticals vertical; scam.project, scam.f, scam.pos), depth-tested
    against zb (view depth, as world.shade writes it). pos = the base on the ground (world); f_ign, f = A cut frames;
    pxs = pixel scale vs a 1920-wide frame; trans = the air's transmittance to the fire (fire_trans). Beacons.py's
    recipe."""
    t = f / FPS
    sz, inten, light = _ign(f, f_ign)
    if sz <= 0:
        return
    Hf0, Rb, pk, pe, ak, _, _, _ = fire_dims(size)
    offset = 0.45 * Hf0 if base_offset is None else float(base_offset)
    volume = near_hearth and base_offset is not None
    base = np.array([pos[0], pos[1] + offset, pos[2]])
    sx, sy, z = scam.project(base)
    if z <= 1.0:
        return
    Hh, Ww = img.shape[0], img.shape[1]
    if not volume and (sx < -300 or sx > Ww + 300 or sy < -300 or sy > Hh + 300):
        return
    ppm = scam.f / z
    fl = F.flicker(t, int(seed) + 11, 1.3)
    zbias = fire_depth_bias(z, near_hearth)
    Hf = Hf0 * sz
    if smoke and z < 4000.0:
        # The revised local hearth gets moonlit smoke; all other shots retain their recipe.
        smoke_amb = (0.024, 0.032, 0.051) if near_hearth and z < 60.0 else (0.004, 0.005, 0.009)
        _smoke(pos, f_ign, size, seed).render(img, zb, scam, t, base + np.array([0, 0.8 * Hf, 0]),
                                              (2.5 if size > 1.8 else 0.8) * size * light * fl, albedo=0.25,
                                              amb=smoke_amb, zbias=zbias)
    e = light * fl * trans
    aura = .15 if volume else 1.0
    # the air glow round a fire, capped on screen: near a fire (tens of metres) a 7 m / 30 m world-sized halo would
    # wash the whole frame warm through everything behind it; beacons at their usual distances are unchanged
    sig1 = min(max(7.0 * ppm, 2.2 * pxs), 90.0 * pxs)
    sig2 = min(max(30.0 * ppm, 9.0 * pxs), 260.0 * pxs)
    F2.halo(img, zb, sx, sy - 0.4 * Hf * ppm, sig1, aura * 0.018 * pk * e, z=z, zbias=zbias)
    F2.halo(img, zb, sx, sy - 0.4 * Hf * ppm, sig2, aura * 0.0016 * pk * e, z=z, zbias=zbias)
    if volume:
        # Roots overlap the fuel; opaque cinders clip them into separate gaps
        # instead of all tongues starting on a visible horizontal plane.
        fuel_origin = base - np.array([0., .06 * sz, 0.])
        FN.draw(img, zb, scam, fuel_origin, .85 * Hf, 1.2 * Rb * sz, t, seed=int(seed) * 7 + 3,
                I=12.0 * inten, lean=(.16 * sz, .04 * sz), trans=trans)
    else:
        F2.flame(img, zb, scam, base, Hf, Rb * sz, t, seed=int(seed) * 7 + 3,
                 I=(24.0 + 6.0 * min(size - 1.0, 1.7) / 1.7) * inten * trans,
                 lean=0.25 * sz, zbias=zbias, tongues=7 if size > 1.8 else 5)
    if Hf * ppm < 3.0 * pxs:
        # too far to resolve: a warm hot point with a soft orange aura (a fire, not a lamp)
        F2.glow(img, zb, sx, sy - 0.3 * Hf * ppm, 0.95 * pxs, pe * e * pxs * pxs, z=z, zbias=zbias,
                col=np.array([1.0, 0.62, 0.26]))
        F2.halo(img, zb, sx, sy - 0.3 * Hf * ppm, 3.4 * pxs, ak * e, z=z, zbias=zbias, col=F2.AURA_COL)
        F2.halo(img, zb, sx, sy - 0.3 * Hf * ppm, 11.0 * pxs, ak * 0.09 * e, z=z, zbias=zbias, col=F2.AURA_COL)


TRANS_K = 0.8                 # a fire's light survives the haze a little better than the terrain's reflected light


def fire_trans(cam_pos, pos, fogp, wmod=WD, k=TRANS_K):
    """exp(-k tau): the air between the eye and a fire, with world.shade's two fog terms (the height fog and the
    cloud-top mist) and the earth's curvature."""
    dx = pos[0] - cam_pos[0]
    dz = pos[2] - cam_pos[2]
    yw = pos[1] - (dx * dx + dz * dz) / (2.0 * WD.R_EARTH)
    dist = math.sqrt(dx * dx + dz * dz + (yw - cam_pos[1]) ** 2)
    tau = wmod.height_fog_tau(dist, cam_pos[1], yw, fogp[0], fogp[1])
    tau += wmod.height_fog_tau(dist, cam_pos[1] - WD.CLOUD_Y, yw - WD.CLOUD_Y, fogp[2], fogp[3])
    return math.exp(-k * tau)


def fires_layer(img, zb, scam, fires, f, pxs=1.0, smoke=True, fogp=None, wmod=WD,
                near_hearth=False, base_offsets=None):
    """Every fire of a (N, 6) table, far to near; with fogp, each is dimmed by the air in front of it."""
    FR = np.asarray(fires, np.float64).reshape(-1, 6)
    if FR.shape[0] == 0:
        return
    cp = np.asarray(scam.pos, np.float64)
    d = np.linalg.norm(FR[:, :3] - cp[None], axis=1)
    for k in np.argsort(-d):
        x, y, z, f_ign, size, seed = FR[k]
        tr = 1.0 if fogp is None else fire_trans(cp, (x, y, z), fogp, wmod)
        offset = None if base_offsets is None else base_offsets.get(int(seed))
        fire_layer(img, zb, scam, (x, y, z), f_ign, f, size, seed, pxs, smoke, tr,
                   near_hearth=near_hearth, base_offset=offset)


_SPARKS = {}


def ignition_sparks(pos, f_ign, size=1.0, seed=0):
    """H5 sparks for a near catch: a short burst, born orange (never white), arcing up and falling as they curve down
    the wind. Render with .render(img, zb, scam, f / 24, ...) (mt.fire.Sparks), only when the fire is resolved."""
    key = (round(pos[0], 1), round(pos[2], 1), int(seed))
    if key not in _SPARKS:
        Hf = fire_dims(size)[0]
        sp = F.Sparks(500 + int(seed), np.array([pos[0], pos[1] + 0.3 * Hf, pos[2]]), f_ign / FPS,
                      f_ign / FPS + 1.2, burst=int(60 * size), rate=18.0 * size, ember_rate=4.0 * size,
                      speed=(1.5, 3.5), burst_speed=(2.5, 5.5), spread=0.55, life=(0.35, 0.9), wind=(1.6, 0.0, 0.4),
                      buoy=1.2, updraft_h=0.8 * Hf, drag=1.6, radius=0.35 * size, I=9.0, turb=1.2, curl=0.9)
        sp.T0 *= 0.80                     # born orange-yellow, not white
        _SPARKS[key] = sp
    return _SPARKS[key]
