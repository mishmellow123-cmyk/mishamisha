"""Opt-in A14/A15 hearth flame: an art-directed, world-space emitting volume.

This is procedural look development, not a fluid simulation. Fuel locations
are stationary; continuous advected noise bends and breaks the rising tongues.
The caller owns ignition, smoke, fuel geometry, light pools and finishing.
"""
import math
from functools import lru_cache

import numpy as np
from numba import njit, prange

from mt.noise import fbm3, smoothstep


@lru_cache(maxsize=32)
def _fuel(seed):
    """Irregular fuel patches, not an evenly spaced ring of upright jets.

    Columns: X/Rb, Z/Rb, height/Hf, width/Rb, noise phase, seat height/Rb,
    density, patch angle, width aspect, sideways drift/Rb. Seats are stationary
    relief within the fuel bed; they do not follow the animated flame envelope.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(6):
        x, z = np.clip(rng.normal(0., .30, 2), -.58, .58)
        rows.append([x, z, rng.uniform(.46, 1.15), rng.uniform(.30, .55),
                     rng.uniform(-8., 8.), rng.uniform(.015, .24), rng.uniform(.42, 1.),
                     rng.uniform(-math.pi, math.pi), rng.uniform(.38, .72), rng.uniform(-.18, .18)])
    return np.asarray(rows, np.float64)


def _tongues(seed, t):
    rows = _fuel(seed).copy()
    # Slow height variation is sampled once per draw. There is no phase wrap,
    # birth/death counter or time-dependent random seed.
    for k in range(len(rows)):
        rows[k, 2] *= 1. + .18 * fbm3(rows[k, 4], t * .43, 1.7, 2., seed + 73)
        rows[k, 6] *= 1. + .28 * fbm3(rows[k, 4] + 3.1, t * .91, 4.3, 2., seed + 89)
    return rows


@njit(cache=True, fastmath=True)
def field(x, y, z, Hf, Rb, t, seed, lean_x, lean_z, tongues):
    """Return dimensionless density and colour temperature at a local point."""
    if y <= 0. or y >= 1.65 * Hf + .24 * Rb:
        return 0., 0.
    rise = 1.1 + .85 * math.sqrt(Hf)
    advected_y = (y - rise * t) / Rb
    # Two world-space noise fields give irregular lateral meander. Their
    # contribution fades at the fuel, so the source does not swim over stone.
    a = fbm3(x / Rb * .85 + 2.3, advected_y * .44, z / Rb * .85, 2., seed)
    b = fbm3(x / Rb * .85, advected_y * .44 + 4.7, z / Rb * .85 + 7.1, 2., seed + 19)
    density, heat, weight = 0., 0., 0.
    for k in range(tongues.shape[0]):
        row = tongues[k]
        v = (y - row[5] * Rb) / Hf
        if v <= 0.:
            continue
        height = row[2]
        s = v / height
        if s >= 1.:
            continue
        bend = v ** 1.45
        root_release = smoothstep(0., .15, v)
        curl = .55 * v + .16 * root_release
        ca, sa = math.cos(row[7]), math.sin(row[7])
        drift = Rb * row[9] * smoothstep(0., .35, v)
        cx = Rb * row[0] + lean_x * bend + Rb * curl * a - sa * drift
        cz = Rb * row[1] + lean_z * bend + Rb * curl * b + ca * drift
        # Elongated patches curl into thin sheets with unequal starts. Width
        # and density break low down instead of preserving bright candle roots.
        width = Rb * row[3] * (1. - s) ** .48 * (.75 + .35 * smoothstep(0., .18, v))
        dx, dz = x - cx, z - cz
        across, along = dx * ca + dz * sa, -dx * sa + dz * ca
        radial2 = across * across + along * along / (row[8] * row[8])
        if radial2 >= (width * 1.10) ** 2 + 1e-12:
            continue
        n = fbm3(x / Rb * 3.1 + row[4], advected_y * 1.65,
                 z / Rb * 3.1, 2., seed + 41)
        width *= .80 + .30 * smoothstep(-.20, .25, n)
        q = radial2 / (width * width + 1e-12)
        if q >= 1.:
            continue
        profile = 1. - q
        # Advected necks can close completely above the root, leaving soft
        # detached wisps. Smooth gates avoid hard births and frame jumps.
        broken = smoothstep(-.13, .20, n)
        gate = 1. - smoothstep(.012, .12, v) * (1. - broken)
        d = row[6] * profile * profile * gate * smoothstep(0., .045, v)
        d *= 1. - smoothstep(.80, 1., s)
        hot = smoothstep(.86, .99, profile) * math.exp(-((v - .09) / .10) ** 2)
        hot *= smoothstep(.08, .32, n)
        temp = .29 + .40 * profile * (1. - .55 * min(v, 1.)) + .29 * hot
        density = max(density, d)
        # Continuous mixing keeps a temperature seam from appearing when two
        # overlapping tongues exchange which has the greater density.
        heat += d * temp
        weight += d
    return density, heat / weight if weight > 0. else 0.


@njit(inline='always', fastmath=True)
def _colour(temperature):
    # Most of the volume stays orange/yellow. The pale end requires the small
    # root-core mask in field(), rather than simply integrating a thick ray.
    yellow = smoothstep(.30, .78, temperature)
    hot = smoothstep(.83, .98, temperature)
    return 1., .22 + .45 * yellow + .24 * hot, .018 + .105 * yellow + .54 * hot


@njit(cache=True, fastmath=True)
def _interval(ox, oy, oz, dx, dy, dz, low, high):
    near, far = 0., 1e20
    for axis in range(3):
        origin = ox if axis == 0 else (oy if axis == 1 else oz)
        direction = dx if axis == 0 else (dy if axis == 1 else dz)
        if abs(direction) < 1e-12:
            if origin < low[axis] or origin > high[axis]:
                return 1., 0.
        else:
            a = (low[axis] - origin) / direction
            b = (high[axis] - origin) / direction
            near = max(near, min(a, b))
            far = min(far, max(a, b))
    return near, far


@njit(parallel=True, fastmath=True, cache=True)
def _render(img, zb, C, low, high, rect, Hf, Rb, t, seed, I, lean_x, lean_z, trans, strength, tongues):
    for j in prange(rect[1], rect[3]):
        for i in range(rect[0], rect[2]):
            u, v = i + .5 - C[8], C[9] - j - .5
            dx, dy, dz = C[3] * C[7] + C[5] * u, v, C[4] * C[7] + C[6] * u
            length = math.sqrt(dx * dx + dy * dy + dz * dz)
            dx, dy, dz = dx / length, dy / length, dz / length
            forward = dx * C[3] + dz * C[4]
            if forward <= 0.:
                continue
            near, far = _interval(C[0], C[1], C[2], dx, dy, dz, low, high)
            # zb is horizontal-forward metres, whereas the integral is along
            # a unit 3D ray. Occluders can truncate the volume at any depth.
            far = min(far, float(zb[j, i]) / forward)
            near = max(near, .02)
            if far <= near:
                continue
            step = max(Rb / 14., near / C[7] * .45)
            count = min(96, max(1, int(math.ceil((far - near) / step))))
            ds = (far - near) / count
            throughput, er, eg, eb = 1., 0., 0., 0.
            for k in range(count):
                distance = near + (k + .5) * ds
                x, y, z = C[0] + dx * distance, C[1] + dy * distance, C[2] + dz * distance
                density, temperature = field(x, y, z, Hf, Rb, t, seed, lean_x, lean_z, tongues)
                if density <= 0.:
                    continue
                alpha = 1. - math.exp(-1.65 * density * strength * ds / Rb)
                r, g, b = _colour(temperature)
                source = I * .16 * (.25 + 1.35 * temperature * temperature)
                contribution = throughput * alpha * source
                er += contribution * r
                eg += contribution * g
                eb += contribution * b
                throughput *= 1. - alpha
                if throughput < .004:
                    break
            attenuation = 1. - trans * (1. - throughput)
            img[j, i, 0] = img[j, i, 0] * attenuation + trans * er
            img[j, i, 1] = img[j, i, 1] * attenuation + trans * eg
            img[j, i, 2] = img[j, i, 2] * attenuation + trans * eb


def _bounds(Hf, Rb, lean):
    # Explicit safety envelope: the field also fades out before these limits.
    # Gradient noise is bounded conservatively by 2 here; normalized fBm cannot
    # exceed that bound. 2.3 covers v**1.45 for v <= 1.65.
    # Fuel offset <= .58 Rb; curl <= (.55*1.65+.16)*2 Rb;
    # drift <= .18 Rb; greatest sheet width <= .55*1.10*1.10 Rb.
    padding = 3.7 * Rb
    low = np.array([min(0., 2.3 * lean[0]) - padding, 0., min(0., 2.3 * lean[1]) - padding])
    high = np.array([max(0., 2.3 * lean[0]) + padding, 1.65 * Hf + .24 * Rb,
                     max(0., 2.3 * lean[1]) + padding])
    return low, high


def _rect(C, low, high, width, height):
    corners = np.array([[x, y, z] for x in (low[0], high[0])
                        for y in (low[1], high[1]) for z in (low[2], high[2])])
    delta = corners - C[:3]
    depth = delta[:, 0] * C[3] + delta[:, 2] * C[4]
    if depth.max() <= .02:
        return np.zeros(4, np.int64)
    if depth.min() <= .02:
        return np.array([0, 0, width, height], np.int64)
    sx = C[8] + C[7] * (delta[:, 0] * C[5] + delta[:, 2] * C[6]) / depth
    sy = C[9] - C[7] * delta[:, 1] / depth
    return np.array([max(0, math.floor(sx.min()) - 1), max(0, math.floor(sy.min()) - 1),
                     min(width, math.ceil(sx.max()) + 1), min(height, math.ceil(sy.max()) + 1)], np.int64)


def draw(img, zb, scam, base_world, Hf, Rb, t, seed=0, I=24., lean=0., trans=1., strength=1.):
    """Composite into linear RGB through a lens-shift source camera.

    base_world is the fuel's flame seat; Hf/Rb and lean are metres, t seconds.
    lean is world (X, Z) displacement at nominal Hf (a scalar means world +X).
    I scales emission; strength scales optical density; trans is atmospheric
    transmission, applied once to the composite contribution. I=24 matches the
    caller's input range, not the shared fire2 renderer's resulting brightness.
    Opaque horizontal-forward zb is read but never changed. No depth bias,
    halo, smoke, automatic distant fallback or post-processing is added here.
    """
    if img.ndim != 3 or img.shape[2] != 3 or zb.shape != img.shape[:2]:
        raise ValueError('near fire requires matching RGB and source-depth buffers')
    if not np.issubdtype(img.dtype, np.floating):
        raise ValueError('near fire requires floating linear RGB')
    base = np.asarray(base_world, np.float64)
    wind = np.array([lean, 0.], np.float64) if np.isscalar(lean) else np.asarray(lean, np.float64)
    if base.shape != (3,) or wind.shape != (2,) or not np.isfinite(base).all() or not np.isfinite(wind).all():
        raise ValueError('near fire requires finite xyz base and XZ lean')
    if not np.isfinite([Hf, Rb, t, I, trans, strength]).all() or not 0. <= trans <= 1.:
        raise ValueError('near fire parameters must be finite; trans must be in [0, 1]')
    if Hf <= 0. or Rb <= 0. or I <= 0. or strength <= 0. or trans == 0.:
        return
    C = np.asarray(scam.params(), np.float64).copy()
    if C.shape != (12,) or not np.isfinite(C).all() or C[7] <= 0.:
        raise ValueError('near fire requires valid SrcCam parameters')
    C[:3] -= base
    low, high = _bounds(float(Hf), float(Rb), wind)
    h, w = zb.shape
    rect = _rect(C, low, high, w, h)
    if rect[2] <= rect[0] or rect[3] <= rect[1]:
        return
    _render(img, zb, C, low, high, rect, float(Hf), float(Rb), float(t), int(seed), float(I),
            float(wind[0]), float(wind[1]), float(trans), float(strength), _tongues(int(seed), float(t)))
