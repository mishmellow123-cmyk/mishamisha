"""A15's dry-stone hearth, with its own geometry and source-camera tracer.

Build Hearth(world_base, ground=height_callback) once. Call draw(img, zb, scam,
hearth, LT, light) after terrain, before figures and flames. Buffers are linear
RGB and horizontal-forward depth, as in world.shade. No flame or figure lives
here. fire_base_offset is the topmost central stone plus 3 mm, above world_base.y.
"""
import math

import numpy as np
from numba import njit, prange


NPLANES = 12
NPITS = 4


def _fire_seat(stones, planes, pits, scale):
    """Trace downward at the fire axis without compiling a rendering kernel.

    The central cap guarantees a hit. On sloped ground, neighbouring rubble
    can project above it; measure the whole union rather than only that cap.
    These vectorized distances use the same plane bevel and pit subtraction
    as stone_distance. Each is conservative, so downward steps cannot skip
    a higher stone and find an unrelated lower surface.
    """
    y = float(np.max(stones[:, 1] + stones[:, 3]))
    k = stones[:, 9]
    lateral = -planes[:, :, 0] * stones[:, 0, None] - planes[:, :, 2] * stones[:, 2, None]
    for _ in range(512):
        values = lateral + planes[:, :, 1] * (y - stones[:, 1, None]) - planes[:, :, 3]
        distance = np.full(len(stones), -1e20)
        for n in range(NPLANES):
            blend = np.maximum(k - np.abs(distance - values[:, n]), 0.) / k
            distance = np.maximum(distance, values[:, n]) + blend * blend * k * .25
        pit_distance = pits[:, :, 3] - np.sqrt(pits[:, :, 0] ** 2 + (y - pits[:, :, 1]) ** 2 + pits[:, :, 2] ** 2)
        distance = np.maximum(distance, np.max(np.where(pits[:, :, 3] > 0., pit_distance, -1e20), axis=1))
        nearest = float(distance.min())
        if nearest < 1e-8 * scale:
            return y + .003 * scale
        y -= .9 * nearest
    raise ValueError('could not locate the hearth fire seat on its central stones')


class Hearth:
    """Seeded, stationary stones in coordinates relative to a world-space base.

    Ground callback: ground(world_x, world_z) -> world_y. A broken course is
    embedded in the ground; smaller rubble fills its centre. ground_band can
    restrict additions to the fuel bed's level, leaving uphill natural rock
    exposed instead of growing an artificial staircase along a steep ledge.
    """
    def __init__(self, base, seed=7070, scale=1.0, ground=None, ground_band=None):
        self.base = np.array(base, dtype=np.float64, copy=True)
        if self.base.shape != (3,) or not np.isfinite(self.base).all():
            raise ValueError('hearth base must be a finite world-space xyz')
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError('hearth scale must be positive and finite')
        self.scale = float(scale)
        if ground_band is not None:
            lo, hi = ground_band
            if not (math.isfinite(lo) and math.isfinite(hi) and lo <= 0 <= hi):
                raise ValueError('ground band must be finite and include the fuel-bed level')
        rng = np.random.default_rng(seed)
        rows, planes, pits = [], [], []

        def terrain(x, z):
            value = self.base[1] if ground is None else float(ground(self.base[0] + x, self.base[2] + z))
            if not math.isfinite(value):
                raise ValueError('hearth ground callback returned a nonfinite height')
            return value - self.base[1]

        def stone(x, z, lift, hx, hy, hz, yaw, kind=0, flat=False):
            if kind == 1:
                lift = min(lift, .5 * hy)  # rubble intersects the ground even without a neighbour beneath it
            x, z, hx, hy, hz, lift = np.array([x, z, hx, hy, hz, lift]) * scale
            gy = terrain(x, z)
            cy = gy + lift
            ax, az = (0., 0.) if flat else rng.uniform(-0.17, 0.17, 2)
            sy, cyaw = math.sin(yaw), math.cos(yaw)
            sx, cx, sz, cz = math.sin(ax), math.cos(ax), math.sin(az), math.cos(az)
            rotation = np.array([[cyaw, 0., sy], [0., 1., 0.], [-sy, 0., cyaw]]) \
                @ np.array([[1., 0., 0.], [0., cx, -sx], [0., sx, cx]]) \
                @ np.array([[cz, -sz, 0.], [sz, cz, 0.], [0., 0., 1.]])
            pp = np.zeros((NPLANES, 4))
            for k in range(6):
                a = 2 * math.pi * k / 6 + rng.uniform(-0.10, 0.10)
                n = np.array([math.cos(a), 0., math.sin(a)])
                support = (n[0] ** 2 / hx ** 2 + n[2] ** 2 / hz ** 2) ** -0.5
                pp[k, :3] = rotation @ n
                pp[k, 3] = support * rng.uniform(0.90, 1.04)
            pp[6, :3], pp[6, 3] = rotation @ np.array([0., 1., 0.]), hy
            pp[7, :3], pp[7, 3] = rotation @ np.array([0., -1., 0.]), hy
            # Oblique fracture faces remove corners rather than inflating a box.
            for k in range(4):
                a = rng.uniform(0, 2 * math.pi)
                side = np.array([math.cos(a), 0., math.sin(a)])
                support = (side[0] ** 2 / hx ** 2 + side[2] ** 2 / hz ** 2) ** -0.5
                n = .76 * side + np.array([0., .65 * (1 if k % 2 else -1), 0.])
                length = np.linalg.norm(n)
                pp[8 + k, :3] = rotation @ (n / length)
                pp[8 + k, 3] = (.76 * support + .65 * hy - .07 * min(hx, hz)) / length
            centre = np.array([x, cy, z])
            pit = np.zeros((NPITS, 4))
            for k in range(NPITS):
                radius = rng.uniform(.009, .020) * scale
                local = np.array([rng.uniform(-.48, .48) * hx,
                                  hy + radius * .78, rng.uniform(-.48, .48) * hz])
                pit[k, :3] = centre + rotation @ local
                pit[k, 3] = radius
            tone = rng.uniform(.055, .115) if kind == 0 else rng.uniform(.023, .060)
            rgb = tone * rng.uniform([.91, .94, .94], [1.08, 1.05, 1.07])
            bevel = rng.uniform(.004, .009) * scale
            # Six lateral normal angles have gaps <= pi/3+.2; this radius is
            # deliberately wider than their convex support, including tilt.
            bound = 2.0 * max(hx, hz) + hy
            rows.append(np.r_[centre, bound, rgb, gy, float(kind), bevel, rng.integers(1, 100000)])
            planes.append(pp)
            pits.append(pit)
            return cy, hy

        for course, count in ((0, 10),):
            for k in range(count):
                a = 2 * math.pi * (k + .47 * course) / count + rng.uniform(-.08, .08)
                radius = (.47 - .025 * course) * rng.uniform(.94, 1.06)
                hy = rng.uniform(.048, .074)
                lift = hy * .62 if course == 0 else .105 + hy * .73
                stone(math.cos(a) * radius, math.sin(a) * radius, lift,
                      rng.uniform(.15, .205), hy, rng.uniform(.105, .15),
                      math.pi / 2 - a + rng.uniform(-.2, .2))
        # One flat central cap gives the fire an explicit geometric seat. Its
        # centre avoids the chipped perimeter; pits are removed there below.
        fuel_cap = len(rows)
        stone(0., 0., .035, .18, .080, .16, .3, kind=1, flat=True)
        pits[-1][:, 3] = 0.0
        for _ in range(12):
            a = rng.uniform(0., 2 * math.pi)
            r = rng.uniform(.07, .32)
            stone(r * math.cos(a), r * math.sin(a), rng.uniform(.015, .045),
                  rng.uniform(.065, .115), rng.uniform(.025, .050), rng.uniform(.045, .090),
                  rng.uniform(0., math.pi), kind=1)
        for _ in range(8):
            a = rng.uniform(0., 2 * math.pi)
            r = rng.uniform(.72, 1.03)
            hy = rng.uniform(.025, .055)
            stone(r * math.cos(a), r * math.sin(a), hy * .20,
                  rng.uniform(.070, .125), hy, rng.uniform(.045, .090), rng.uniform(0., math.pi), kind=2)
        self.stones = np.ascontiguousarray(rows, np.float64)
        self.planes = np.ascontiguousarray(planes, np.float64)
        self.pits = np.ascontiguousarray(pits, np.float64)
        if ground_band is not None:
            keep = ((self.stones[:, 7] >= lo * scale) & (self.stones[:, 7] <= hi * scale))
            # Keep the central fuel cap. Other stones never change identity
            # when neighbours are culled: filtering follows all RNG draws.
            keep[fuel_cap] = True
            self.stones = np.ascontiguousarray(self.stones[keep])
            self.planes = np.ascontiguousarray(self.planes[keep])
            self.pits = np.ascontiguousarray(self.pits[keep])
        self.fire_base_offset = _fire_seat(self.stones, self.planes, self.pits, self.scale)
        low = (self.stones[:, :3] - self.stones[:, 3, None]).min(0)
        high = (self.stones[:, :3] + self.stones[:, 3, None]).max(0)
        self.bound_centre = .5 * (low + high)
        self.bound_radius = float(np.linalg.norm(.5 * (high - low)))


@njit(inline='always', fastmath=True)
def _smooth_max(a, b, k):
    h = max(k - abs(a - b), 0.0) / k
    return max(a, b) + h * h * k * .25


@njit(fastmath=True, cache=True)
def stone_distance(x, y, z, index, stones, planes, pits):
    row = stones[index]
    px, py, pz = x - row[0], y - row[1], z - row[2]
    d = -1e20
    for k in range(NPLANES):
        p = planes[index, k]
        d = _smooth_max(d, px * p[0] + py * p[1] + pz * p[2] - p[3], row[9])
    # Removing shallow spheres yields actual pits, not an albedo-only stipple.
    for k in range(NPITS):
        p = pits[index, k]
        if p[3] > 0.0:
            dx, dy, dz = x - p[0], y - p[1], z - p[2]
            d = max(d, p[3] - math.sqrt(dx * dx + dy * dy + dz * dz))
    return d


@njit(fastmath=True, cache=True)
def scene_distance(x, y, z, stones, planes, pits, ignore=-1):
    nearest, which = 1e20, -1
    for k in range(stones.shape[0]):
        if k == ignore:
            continue
        dx, dy, dz = x - stones[k, 0], y - stones[k, 1], z - stones[k, 2]
        lower = math.sqrt(dx * dx + dy * dy + dz * dz) - stones[k, 3]
        if lower > nearest:
            continue
        d = stone_distance(x, y, z, k, stones, planes, pits)
        if d < nearest:
            nearest, which = d, k
    return nearest, which


@njit(fastmath=True, cache=True)
def visibility(x, y, z, lx, ly, lz, distance, stones, planes, ignore=-1):
    """Visibility through stone convex hulls; millimetre pits/bevels are omitted.

    A hit stone is excluded for its own convex-surface direct light. Joint AO
    and pit normals handle local crevices; no positive depth bias is applied.
    """
    for k in range(stones.shape[0]):
        if k == ignore:
            continue
        dx, dy, dz = x - stones[k, 0], y - stones[k, 1], z - stones[k, 2]
        b = dx * lx + dy * ly + dz * lz
        if b * b - (dx * dx + dy * dy + dz * dz - stones[k, 3] ** 2) < 0.0:
            continue
        lo, hi = .003, distance
        for n in range(NPLANES):
            p = planes[k, n]
            den = p[0] * lx + p[1] * ly + p[2] * lz
            gap = p[3] - (p[0] * dx + p[1] * dy + p[2] * dz)
            if abs(den) < 1e-10:
                if gap < 0:
                    hi = -1.
                    break
            elif den > 0:
                hi = min(hi, gap / den)
            else:
                lo = max(lo, gap / den)
        if hi >= lo:
            return 0.0
    return 1.0


@njit(inline='always', fastmath=True)
def _hash(i, j, k, seed):
    h = ((i * 73856093) ^ (j * 19349663) ^ (k * 83492791) ^ seed) & 0xFFFFFFFF
    h = ((h ^ (h >> 16)) * 0x45d9f3b) & 0xFFFFFFFF
    h = ((h ^ (h >> 16)) * 0x45d9f3b) & 0xFFFFFFFF
    return ((h ^ (h >> 16)) & 0xFFFFFFFF) / 4294967295.0


@njit(fastmath=True, cache=True)
def _noise(x, y, z, seed):
    ix, iy, iz = math.floor(x), math.floor(y), math.floor(z)
    u, v, w = x - ix, y - iy, z - iz
    u, v, w = u * u * (3. - 2. * u), v * v * (3. - 2. * v), w * w * (3. - 2. * w)
    value = 0.
    for k in range(2):
        for j in range(2):
            for i in range(2):
                weight = (u if i else 1. - u) * (v if j else 1. - v) * (w if k else 1. - w)
                value += weight * _hash(ix + i, iy + j, iz + k, seed)
    return 2. * value - 1.


@njit(inline='always', fastmath=True)
def _fog_tau(distance, y0, y1, a, b):
    dy = y1 - y0
    e0 = math.exp(-b * y0)
    if abs(dy) < 1e-3:
        return a * distance * e0
    return a * distance * (e0 - math.exp(-b * y1)) / (b * dy)


@njit(fastmath=True, cache=True)
def _shade(x, y, z, nx, ny, nz, fp, hit, stones, planes, pits, lights, moon, ambient, scale):
    row = stones[hit]
    # Joint probes exclude the hit stone: its conservative plane distance is
    # not an ambient-occlusion measurement of itself.
    occ = 0.0
    for h, weight in ((.025, .50), (.075, .32), (.18, .18)):
        h *= scale
        d, _ = scene_distance(x + nx * h, y + ny * h, z + nz * h, stones, planes, pits, hit)
        occ += weight * max(0.0, 1.0 - max(d, 0.0) / h)
    ground = max(y - row[7], 0.) / scale
    ao = max(.18, (1.0 - .78 * occ) * (.48 + .52 * min(ground / .12, 1.)))
    seed = int(row[10])
    grain = _noise(x * 7. / scale, y * 7. / scale, z * 7. / scale, seed)
    fine = max(0.0, 1.0 - fp / (.018 * scale))
    flecks = _noise(x * 160. / scale, y * 160. / scale, z * 160. / scale, seed + 7)
    mot = .86 + .18 * grain + .15 * fine * max(0., flecks)
    radial = math.sqrt(x * x + z * z) / scale
    soot = .55 * max(0.0, 1.0 - radial / .58) * max(0.0, min(1.0, (y / scale + .08) / .22))
    if row[8] == 1.:
        soot = max(soot, .38)
    ar, ag, ab = row[4] * mot * (1. - soot), row[5] * mot * (1. - soot), row[6] * mot * (1. - soot)
    sky = (.35 + .65 * max(ny, 0.)) * ao
    cr, cg, cb = ar * ambient[0] * sky, ag * ambient[1] * sky, ab * ambient[2] * sky
    ndm = max(nx * moon[0] + ny * moon[1] + nz * moon[2], 0.)
    mv = visibility(x + nx * .004 * scale, y + ny * .004 * scale, z + nz * .004 * scale,
                    moon[0], moon[1], moon[2], 5.0 * scale, stones, planes, hit)
    key = moon[6] * (ndm * mv + .045 * (1.0 - max(ny, 0.)) * ao)
    cr += ar * moon[3] * key
    cg += ag * moon[4] * key
    cb += ab * moon[5] * key
    for k in range(lights.shape[0]):
        lx, ly, lz = lights[k, 0] - x, lights[k, 1] - y, lights[k, 2] - z
        d2 = lx * lx + ly * ly + lz * lz
        distance = math.sqrt(d2) + 1e-12
        lx, ly, lz = lx / distance, ly / distance, lz / distance
        ndl = max(nx * lx + ny * ly + nz * lz, 0.)
        vis = visibility(x + nx * .004 * scale, y + ny * .004 * scale, z + nz * .004 * scale,
                         lx, ly, lz, distance, stones, planes, hit)
        energy = lights[k, 6] / (d2 + lights[k, 7] ** 2)
        # A small, occluded bounce from the rubble opens the joints. It does
        # not emit light by itself and vanishes with the supplied fire energy.
        bounce = .055 * ao * (.45 + .55 * max(-ny, 0.))
        diffuse = energy * (ndl * vis + bounce)
        cr += ar * lights[k, 3] * diffuse
        cg += ag * lights[k, 4] * diffuse
        cb += ab * lights[k, 5] * diffuse
    return cr, cg, cb


@njit(parallel=True, fastmath=True, cache=True)
def _render(img, zb, C, base_y, stones, planes, pits, bc, br, lights, moon, ambient, fogp, scale, rect):
    fx, fz, rx, rz, f = C[3], C[4], C[5], C[6], C[7]
    for j in prange(rect[1], rect[3]):
        for i in range(rect[0], rect[2]):
            u, v = i + .5 - C[8], C[9] - (j + .5)
            dx, dy, dz = fx * f + rx * u, v, fz * f + rz * u
            norm = math.sqrt(dx * dx + dy * dy + dz * dz)
            dx, dy, dz = dx / norm, dy / norm, dz / norm
            forward = dx * fx + dz * fz
            if forward <= 0:
                continue
            lx, ly, lz = C[0] - bc[0], C[1] - bc[1], C[2] - bc[2]
            b = lx * dx + ly * dy + lz * dz
            disc = b * b - (lx * lx + ly * ly + lz * lz - br * br)
            if disc < 0:
                continue
            end = min(-b + math.sqrt(disc), float(zb[j, i]) / forward)
            t = max(.02, -b - math.sqrt(disc))
            hit = -1
            for _ in range(160):
                if t > end:
                    break
                x, y, z = C[0] + dx * t, C[1] + dy * t, C[2] + dz * t
                distance, which = scene_distance(x, y, z, stones, planes, pits)
                eps = max(.0004 * scale, min(.18 * t / f, .003 * scale))
                if distance < eps:
                    hit = which
                    break
                t += max(distance * .82, eps * .45)
            if hit < 0 or t * forward >= zb[j, i]:
                continue
            e = max(.0007 * scale, min(.35 * t / f, .006 * scale))
            nx = stone_distance(x + e, y, z, hit, stones, planes, pits) - stone_distance(x - e, y, z, hit, stones, planes, pits)
            ny = stone_distance(x, y + e, z, hit, stones, planes, pits) - stone_distance(x, y - e, z, hit, stones, planes, pits)
            nz = stone_distance(x, y, z + e, hit, stones, planes, pits) - stone_distance(x, y, z - e, hit, stones, planes, pits)
            length = math.sqrt(nx * nx + ny * ny + nz * nz) + 1e-12
            nx, ny, nz = nx / length, ny / length, nz / length
            # Filtered mineral roughness perturbs normals at resolvable scales;
            # the chipped silhouette and pits remain actual geometry.
            bump = .0012 * scale * max(0., 1. - t / (f * .018 * scale))
            if bump > 0.:
                seed = int(stones[hit, 10])
                freq = 65. / scale
                bx = (_noise((x + e) * freq, y * freq, z * freq, seed)
                      - _noise((x - e) * freq, y * freq, z * freq, seed)) / (2 * e)
                by = (_noise(x * freq, (y + e) * freq, z * freq, seed)
                      - _noise(x * freq, (y - e) * freq, z * freq, seed)) / (2 * e)
                bz = (_noise(x * freq, y * freq, (z + e) * freq, seed)
                      - _noise(x * freq, y * freq, (z - e) * freq, seed)) / (2 * e)
                dot = nx * bx + ny * by + nz * bz
                nx, ny, nz = nx + bump * (bx - dot * nx), ny + bump * (by - dot * ny), nz + bump * (bz - dot * nz)
                length = math.sqrt(nx * nx + ny * ny + nz * nz)
                nx, ny, nz = nx / length, ny / length, nz / length
            cr, cg, cb = _shade(x, y, z, nx, ny, nz, t / f, hit, stones, planes, pits, lights, moon, ambient, scale)
            tau = _fog_tau(t, C[1] + base_y, y + base_y, fogp[0], fogp[1])
            tau += _fog_tau(t, C[1] + base_y + 650., y + base_y + 650., fogp[2], fogp[3])
            tr = math.exp(-tau)
            img[j, i, 0] = cr * tr + fogp[5] * (1. - tr)
            img[j, i, 1] = cg * tr + fogp[6] * (1. - tr)
            img[j, i, 2] = cb * tr + fogp[7] * (1. - tr)
            zb[j, i] = t * forward


@njit(parallel=True, fastmath=True, cache=True)
def _ground_shadow(img, zb, C, stones, planes, pits, lights, moon, ambient, scale, rect):
    """Local irradiance-ratio correction on the already shaded terrain.

    This is an approximation (the source pass has no separate light AOV). It
    preserves colour rather than painting a black disc, and never changes depth.
    Call before drawing people so their pixels cannot enter this ground pass.
    """
    for j in prange(rect[1], rect[3]):
        for i in range(rect[0], rect[2]):
            depth = float(zb[j, i])
            if depth <= .02 or depth >= 1e8:
                continue
            u, v = (i + .5 - C[8]) / C[7], (C[9] - j - .5) / C[7]
            x, y, z = C[0] + depth * (C[3] + C[5] * u), C[1] + depth * v, C[2] + depth * (C[4] + C[6] * u)
            if x * x + z * z > (2.6 * scale) ** 2 or abs(y) > 2.0 * scale:
                continue
            nearest, _ = scene_distance(x, y + .005 * scale, z, stones, planes, pits)
            contact = 1. - .35 * math.exp(-max(nearest, 0.) / (.045 * scale))
            # Upward terrain normal is a stable approximation for the narrow
            # cleared hearth patch. Larger slopes remain the world's surface.
            before = np.empty(3)
            after = np.empty(3)
            for c in range(3):
                before[c] = ambient[c] * .8 + moon[c + 3] * moon[6] * max(moon[1], 0.)
                after[c] = before[c]
            for k in range(lights.shape[0]):
                lx, ly, lz = lights[k, 0] - x, lights[k, 1] - y, lights[k, 2] - z
                d2 = lx * lx + ly * ly + lz * lz
                length = math.sqrt(d2) + 1e-12
                energy = lights[k, 6] * max(ly / length, 0.) / (d2 + lights[k, 7] ** 2)
                vis = visibility(x, y + .012 * scale, z, lx / length, ly / length, lz / length,
                                 length, stones, planes)
                for c in range(3):
                    coloured = energy * lights[k, 3 + c]
                    before[c] += coloured
                    after[c] += coloured * (.10 + .90 * vis)
            for c in range(3):
                ratio = min(1., max(.10, after[c] / before[c])) if before[c] > 1e-12 else 1.
                img[j, i, c] *= contact * ratio


def _rect(C, centre, radius, width, height):
    delta = centre - C[:3]
    depth = delta[0] * C[3] + delta[2] * C[4]
    if depth + radius <= .02:
        return np.array([0, 0, 0, 0], np.int64)
    if depth <= radius:
        return np.array([0, 0, width, height], np.int64)
    sx = C[8] + C[7] * (delta[0] * C[5] + delta[2] * C[6]) / depth
    sy = C[9] - C[7] * delta[1] / depth
    # Include perspective expansion when the sphere centre is off-axis.
    lateral = max(abs(delta[0] * C[5] + delta[2] * C[6]), abs(delta[1]))
    pixels = C[7] * radius * (1. + lateral / depth) / (depth - radius) + 2.
    return np.array([max(0, int(math.floor(sx - pixels))), max(0, int(math.floor(sy - pixels))),
                     min(width, int(math.ceil(sx + pixels + 1))), min(height, int(math.ceil(sy + pixels + 1)))], np.int64)


def draw(img, zb, scam, hearth, LT, light, ground_shadow=True):
    """Draw in place using SrcCam, Nx8 point lights and the existing light tuple.

    img: floating linear RGB; zb: horizontal-forward depth, both source-sized.
    No multi-metre depth bias: foreground terrain/figures occlude the stones.
    """
    if img.ndim != 3 or img.shape[2] != 3 or zb.shape != img.shape[:2]:
        raise ValueError('hearth requires matching source RGB and depth buffers')
    C = np.asarray(scam.params(), np.float64).copy()
    C[:3] -= hearth.base
    lights = np.asarray(LT, np.float64).reshape(-1, 8).copy()
    if len(lights):
        lights[:, :3] -= hearth.base
        # Distant fires contribute negligible local stone lighting; retaining
        # every visible beacon would multiply every joint-shadow query.
        lights = np.ascontiguousarray(lights[np.linalg.norm(lights[:, :3], axis=1) < 40.0 * hearth.scale])
    lk, amb, _sky, fogp, q = light
    moon = np.r_[np.asarray(lk[:6], np.float64), q[0]]
    moon[:3] /= max(np.linalg.norm(moon[:3]), 1e-12)
    ambient = np.asarray(amb, np.float64)
    fogp = np.asarray(fogp, np.float64)
    h, w = zb.shape
    if ground_shadow:
        rect = _rect(C, np.array([0., .5 * hearth.scale, 0.]), 3.3 * hearth.scale, w, h)
        _ground_shadow(img, zb, C, hearth.stones, hearth.planes, hearth.pits,
                       lights, moon, ambient, hearth.scale, rect)
    rect = _rect(C, hearth.bound_centre, hearth.bound_radius, w, h)
    _render(img, zb, C, float(hearth.base[1]), hearth.stones, hearth.planes, hearth.pits,
            hearth.bound_centre, hearth.bound_radius, lights, moon, ambient, fogp, hearth.scale, rect)
