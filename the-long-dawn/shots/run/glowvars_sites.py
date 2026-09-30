"""Fixed beacon layout on the accepted A500 terrain and camera.

The approved pair uses measured native-raster seats, not fine-grid summit tips.
Its original summit predecessors and the other sites remain catalogue-audited.
All returned coordinates are physical; curvature belongs to the view transform.
"""
from functools import lru_cache
import math
from pathlib import Path

import numpy as np

import falsedawn as FD
import world as WD

CAMERA_FRAME = 420
CHAIN_COUNT = 7
PAIR_NATIVE_BASES = np.array([
    [733.4482365330008, 463.0645838537898],
    [1386.8711906600877, 455.13424081989217],
])
# Measured physical ground points supporting those exact native source pixels.
# They sit on the accepted rendered crest; neither is claimed as a fine-grid
# local maximum. Do not rerun a screen-position selector for these anchors.
PAIR_ANCHORS = np.array([
    [-5893.961767525668, 345.47371458581017, 7824.677509257563],
    [-16897.70185712461, 1302.8076234419345, 43064.91201378168],
])
_APPROVED_SUMMITS = np.array([
    [-5896.6637047077365, 371.46285979763894, 7839.818396682447],
    [-16934.100236666865, 1358.4573458559541, 43142.43784220825],
    [-2017.0522668541182, -101.97868048769328, 7836.023101270959],
    [-4689.503862996358, -96.5110506234139, 5608.926926114098],
    [-4138.53543291798, -184.0651412605207, 4647.616459287168],
    [-2767.993000893333, 25.699266118119795, 4387.56631178565],
    [-1127.7837073149942, 48.19270033196926, 3908.01041367354],
    [-1028.8411594986264, 69.73155349359831, 3836.855738219373],
    [-7779.884261469668, 282.33600204443144, 6093.388543594818],
    [-991.8949040311154, -108.91458381981624, 2765.466105703173],
    [-10369.277121672421, 1791.0729764614737, 45722.262523843405],
    [-6828.06511064123, 618.7911097231386, 6963.679623654493],
    [-4138.483343938829, -158.91344425246075, 8638.483343938828],
    [-6287.709969623496, 143.58698801519793, 6356.3012076420855],
    [-5247.5, -204.62105492721616, 27488.75],
    [-13752.5, 130.65978660334804, 23541.875],
    [-3309.566563462609, -499.11179583934404, 8733.563467498609],
    [-5879.373881921259, -463.87356175495097, 9763.204382838583],
    [-10902.5, 129.84991410338193, 23693.75],
    [-6857.326919341524, 42.219580800051254, 5932.499795601497],
    [-9290.0, 203.3915184310663, 34242.5],
    [-13844.58873224265, 1533.9473394221884, 44065.66950099331],
    [-3170.3698145543526, -210.80278215126032, 7927.662262837294],
    [-7091.057575090326, 440.6986019596974, 6605.908766705717],
    [-9786.875, -350.5628053648484, 24655.625],
    [-6508.826929629968, 447.2664758583112, 7156.489976436293],
    [-5548.9605591060745, 273.0674358090123, 8211.396593724066],
    [-7929.619357986923, -162.50000863333116, 12913.079571991282],
    [-8159.375, -30.958086497790646, 23778.125],
    [-7539.995662963681, 376.83518941653784, 6379.417168766378],
    [-7452.5, 202.2595043157528, 33608.75],
    [-6294.890217898636, 438.05826169941975, 7570.281270332647],
    [-6783.125, -284.4364707868474, 26540.0],
    [-8476.25, -314.2289172501339, 18646.25],
    [-15438.819520405013, 1879.3548246759683, 43080.83118913913],
    [-12080.0, 175.47377290104828, 18946.25],
    [-14935.625, 60.117802693227986, 29180.0],
    [-4117.514821075298, 17.255831392357322, 7676.646905232937],
    [-10797.5, -147.148311435398, 35661.875],
    [-7512.3342282335225, 171.9447339455631, 6172.069820600289],
])
# The old fourth tail site seats at native y645.21, inside the caption band.
# Keep it in the watch, already lit, after the seven-site ignition chain.
_WATCH_ORDER = np.array([0, 1, 2, 3, 5, 6, 7, 4, *range(8, 40)])
for _constant in (PAIR_NATIVE_BASES, PAIR_ANCHORS, _APPROVED_SUMMITS, _WATCH_ORDER):
    _constant.setflags(write=False)


def curved(points, camera):
    """Return camera-relative curved coordinates, leaving the input untouched."""
    out = np.asarray(points, np.float64).copy()
    dx = out[..., 0] - camera.pos[0]
    dz = out[..., 2] - camera.pos[2]
    out[..., 1] -= (dx * dx + dz * dz) / (2.0 * WD.R_EARTH)
    return out


def _visible(point, camera, terrain, camera_frame, lift=2.4):
    target = curved(point, camera) + np.array([0.0, lift, 0.0])
    delta = target - camera.pos
    distance = float(np.linalg.norm(delta))
    world_params = np.array([camera.pos[0], camera.pos[2], camera_frame / FD.FPS, 0.0])
    # Log-only samples leave kilometre gaps next to a far target and can miss
    # the wall's intervening crests. Bound the world-space gap along the ray.
    fractions = np.unique(np.r_[np.geomspace(max(0.5 / distance, 1e-5), 0.998, 80),
                                np.linspace(0.0, 0.998, max(32, int(math.ceil(distance / 20.0))))])
    for fraction in fractions:
        q = camera.pos + delta * fraction
        if WD.hfun(q[0], q[2], max(0.15, fraction * distance / 1500.0), world_params, terrain) > q[1]:
            return False
    return True


@lru_cache(maxsize=4)
def _catalogue(camera_frame):
    from beacons import _raycast

    camera = FD.camera(camera_frame)
    terrain = FD.terrain_rows()
    source = np.load(Path(__file__).with_name('summits.npy'))[:, :3]
    seeds = list(source)
    for row in terrain:
        if -1.5 < row[12] < 0.0:
            seeds.extend((row[[0, 2, 1]], row[[3, 5, 4]]))
        elif row[12] >= 0.0:
            seeds.append(row[[0, 2, 1]])
    # The catalogue's coarse world grid can miss the foreground's small crests.
    # Screen rays add only points already present in the accepted terrain.
    for x in np.linspace(0.08, 0.92, 15):
        for y in np.linspace(0.65, 0.96, 5):
            hit, distance = _raycast(camera, x * camera.W, y * camera.H, terrain, camera_frame / FD.FPS)
            if hit is not None:
                ground = WD.ground(hit[0], hit[2], terrain)
                if ground > WD.CLOUD_Y + 35.0:
                    seeds.append(WD.summit_near(hit[0], hit[2], terrain,
                                               rad=max(60.0, min(600.0, 0.07 * distance)), n=17))
    seeds = np.asarray(seeds)
    sx, sy, depth = camera.project(curved(seeds, camera))
    distance = np.linalg.norm(seeds - camera.pos, axis=1)
    keep = ((depth > 0) & (sx > -0.05 * camera.W) & (sx < 1.05 * camera.W)
            & (sy > 0.15 * camera.H) & (sy < 1.05 * camera.H)
            & (distance > 50.0) & (distance < 75000.0)
            & (seeds[:, 1] > WD.CLOUD_Y + 35.0))
    visible = []
    for seed, dist in zip(seeds[keep], distance[keep]):
        point = WD.summit_near(seed[0], seed[2], terrain,
                               rad=max(12.0, min(120.0, 0.0075 * dist)), n=9)
        # A bounded climb can finish on a rising flank. Require a local crest.
        ring = [WD.ground(point[0] + 3.0 * math.cos(a), point[2] + 3.0 * math.sin(a), terrain)
                for a in np.linspace(0.0, 2.0 * math.pi, 8, endpoint=False)]
        if max(ring) > point[1] + 0.25:
            continue
        if any(np.linalg.norm(point[[0, 2]] - row[:3][[0, 2]]) < 45.0 for row in visible):
            continue
        x, y, z = camera.project(curved(point, camera))
        if z <= 0 or not (0.045 < x / camera.W < 0.955 and 0.20 < y / camera.H < 0.93):
            continue
        if not _visible(point, camera, terrain, camera_frame):
            continue
        dist = float(np.linalg.norm(point - camera.pos))
        visible.append(np.r_[point, x / camera.W, y / camera.H, dist])
    required = len(_APPROVED_SUMMITS)
    if len(visible) < required:
        raise RuntimeError(f'Only {len(visible)} verified visible false-dawn summits; need the {required} approved watch sites')
    result = np.asarray(visible)
    result.setflags(write=False)
    return result


def _selection():
    """Audit the frozen summit predecessors; never choose new positions."""
    catalogue = _catalogue(CAMERA_FRAME)
    picked = []
    for index, point in enumerate(_APPROVED_SUMMITS):
        errors = np.linalg.norm(catalogue[:, :3] - point, axis=1)
        chosen = int(np.argmin(errors))
        if errors[chosen] > 1e-6:
            raise RuntimeError(f'Approved summit {index} no longer belongs to the accepted terrain catalogue')
        picked.append(chosen)
    return catalogue, picked


def _production_points():
    points = _APPROVED_SUMMITS[_WATCH_ORDER].copy()
    points[:2] = PAIR_ANCHORS
    return points


def sites():
    """Fixed physical pair and five approaching ridge sites, at camera 420."""
    _selection()
    return _production_points()[:CHAIN_COUNT].copy()


def validate_chain_caption_band(points):
    """Reject new chain fire bases in the lower 20% of the native picture."""
    camera = FD.camera(CAMERA_FRAME)
    points = np.asarray(points, np.float64).reshape(-1, 3)
    _, y, _ = camera.project(curved(points, camera))
    if np.any(y >= 0.8 * camera.H):
        raise RuntimeError('Rendered chain ignition intrudes into the lower 20% caption band')


def _validate_anchor(source, point, D, terrain):
    """Keep an approved physical anchor; validate exact native raster support.

    Coarser renders retain the same ground point. Their larger pixel footprint
    may miss the fine crest itself, so require nearby raster support instead.
    """
    if abs(WD.ground(point[0], point[2], terrain) - point[1]) > 1e-6:
        raise RuntimeError('Fixed beacon anchor no longer lies on physical ground')
    x, y, depth = source.project(curved(point, source))
    if not (0 <= x < source.W and 0 <= y < source.H and depth > 0):
        raise RuntimeError('Fixed beacon anchor is outside the source view')
    expected_camera = FD.camera(CAMERA_FRAME)
    if not np.allclose(source.pos, expected_camera.pos, rtol=0, atol=1e-9):
        raise RuntimeError('Fixed beacon anchors require the accepted camera 420')
    distance = float(np.linalg.norm(point[[0, 2]] - source.pos[[0, 2]]))
    scale = source.f / (1.5 * expected_camera.f)
    if math.isclose(scale, 1., rel_tol=0., abs_tol=1e-9):
        if abs(float(D[int(y), int(x)]) - distance) > .01:
            raise RuntimeError('Fixed beacon anchor lost its approved native raster support')
    else:
        row, col = int(y), int(x)
        nearby = D[max(0, row-3):min(source.H, row+4), max(0, col-3):min(source.W, col+4)]
        if not np.any(np.abs(nearby-distance) < max(150., .015*distance)):
            raise RuntimeError('Fixed beacon anchor lacks nearby support in the coarse raster')


def seat_on_raster(source, xyz, D):
    """Return physical fire seats on the already-marched visible ridge surface.

    Fine-grid summit tips can sit above the accepted marcher's sampled crest.
    Reconstruct nearby hit points from D, then check their physical ground
    heights project back onto those source pixels. This changes no terrain.
    Bounds scale with the source focal length; missing support is an error.
    """
    terrain = FD.terrain_rows()
    C = source.params()
    if D.shape != (source.H, source.W):
        raise ValueError('D must match the source camera dimensions')
    pixel_scale = source.f / (1.5 * FD.camera(420).f)
    half_width = max(2, int(round(12.0 * pixel_scale)))
    max_down = max(4, int(round(45.0 * pixel_scale)))
    result = []
    production = _production_points()[:CHAIN_COUNT]
    for index, point in enumerate(np.asarray(xyz, np.float64).reshape(-1, 3)):
        matched = np.flatnonzero(np.max(np.abs(production - point), axis=1) < 1e-8)
        if len(matched) and matched[0] < 2:
            _validate_anchor(source, point, D, terrain)
            result.append(point.copy())
            continue
        x, y, _ = source.project(curved(point, source))
        left = max(0, int(x) - half_width)
        right = min(source.W, int(x) + half_width + 1)
        horizontal_distance = float(np.linalg.norm(point[[0, 2]] - source.pos[[0, 2]]))
        radius = max(150.0, 0.015 * horizontal_distance)
        choices = []
        for col in range(left, right):
            near = np.flatnonzero((np.abs(D[:, col] - horizontal_distance) < radius)
                                  & (np.arange(source.H) >= math.floor(y) - 1)
                                  & (np.arange(source.H) <= y + max_down))
            for row in near[:3]:
                xo = col + 0.5 - C[8]
                dx = C[3] * C[7] + C[5] * xo
                dz = C[4] * C[7] + C[6] * xo
                norm = math.hypot(dx, dz)
                px = C[0] + dx / norm * D[row, col]
                pz = C[2] + dz / norm * D[row, col]
                py = WD.ground(px, pz, terrain)
                candidate = np.array([px, py, pz])
                _, projected_y, _ = source.project(curved(candidate, source))
                if row <= projected_y < row + 1.0:
                    choices.append((row + 0.10 * abs(col + 0.5 - x), candidate))
        if not choices:
            raise RuntimeError(f'No rendered ridge surface supports beacon {index} within the seating bounds')
        seat = min(choices, key=lambda choice: choice[0])[1]
        if len(matched) and matched[0] >= 2:
            validate_chain_caption_band([seat])
        result.append(seat)
    return np.asarray(result, np.float64).reshape(-1, 3)


def watch_sites():
    """The fixed chain followed by the remaining approved watch sites."""
    _selection()
    return _production_points()


def audit():
    """JSON-compatible site provenance and measured projection for review."""
    catalogue, picked = _selection()
    camera = FD.camera(CAMERA_FRAME)
    points = _production_points()
    x, y, _ = camera.project(curved(points, camera))
    distance = np.linalg.norm(points-camera.pos, axis=1)
    rows = [dict(xyz=p.tolist(), screen_xy=[float(px/camera.W), float(py/camera.H)],
                 distance=float(d)) for p, px, py, d in zip(points, x, y, distance)]
    return dict(camera_frame=CAMERA_FRAME, terrain='falsedawn.terrain_rows()',
                sources=['shots/run/summits.npy', 'falsedawn.terrain_rows ridge endpoints and crag centres',
                         'beacons._raycast across the accepted foreground'],
                verified_visible_candidates=len(catalogue),
                pair_native_bases=PAIR_NATIVE_BASES.tolist(), pair_anchors=PAIR_ANCHORS.tolist(),
                pair_provenance='Approved native raster seats on physical ground; original catalogue summit predecessors retained for audit',
                summit_predecessors=catalogue[np.asarray(picked)[_WATCH_ORDER], :3].tolist(),
                chain_site_count=CHAIN_COUNT,
                watch_site_count=len(points), sites=rows[:CHAIN_COUNT], watch_sites=rows)
