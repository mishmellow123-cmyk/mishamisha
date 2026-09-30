"""Default-off RUN-A4 studies; accepted crossing.py remains untouched.

render_cut() uses absolute A frames. It returns the original linear HDR image;
the caller applies crossing.PI.look.finish(..., **crossing.FINISH) once.
Importing this module loads no renderer, NumPy, OpenCV, or Numba and sets no
thread/cache options. Render calls belong in an isolated, single-render process:
candidate hooks temporarily replace renderer functions and restore them on exit.

The rope study changes geometry only. It retains the original lighting, width,
opacity, segment-depth test and rasterizer. Snow clearance is verified at every
vertex and interior chord probes spaced at most 2 cm apart in the terrain X/Z
plane (the count is chosen from the original 3D chord before lifting). This
is a sampled terrain contract, not a proof about arbitrarily narrow unsampled
terrain peaks. Endpoints never move; an attachment below the requested snow
clearance fails visibly instead of silently detaching the rope.

Studies only. The A crossing finals remain held pending the director's decision.
"""
from dataclasses import dataclass
import hashlib
import importlib
import math
from pathlib import Path
import sys
from threading import RLock


CUT0 = 4880
CUT_END = 5839
BASELINE_COMMIT = '76aec6c60d59a72388fca44c88984f9817ecb8bc'
BASELINE_SHA256 = '67c2ee812fd4f6e30436c7ec53dbcc1b21806ee70a0b18125d7045a348238ac3'
ROPE_OPTIONS = ('accepted', 'snow_clearance', 'snow_decal')
TERRAIN_OPTIONS = ('accepted', 'round_cap')
_RENDER_LOCK = RLock()


@dataclass(frozen=True)
class RopeSolution:
    points: tuple
    probe_count: int
    max_lift: float
    minimum_sampled_clearance: float
    original_minimum_sampled_clearance: float


def sag_points(a, b, count=14):
    """Original crossing.py:1245-1248 parabola, expressed without NumPy."""
    a, b = tuple(map(float, a)), tuple(map(float, b))
    if len(a) != 3 or len(b) != 3 or count < 3:
        raise ValueError('two 3D attachments and at least three points required')
    if not all(math.isfinite(v) for p in (a, b) for v in p):
        raise ValueError('attachments must be finite')
    d = math.dist(a, b)
    sag = min((0.10 + 0.04 * math.sin(d * 3.1)) * d, 0.55)
    points = []
    for i in range(count):
        u = i / (count - 1)
        p = [a[j] + (b[j] - a[j]) * u for j in range(3)]
        p[1] -= 4.0 * sag * u * (1.0 - u)
        points.append(tuple(p))
    points[0], points[-1] = a, b
    return tuple(points)


def clear_rope(points, ground_many, *, clearance=0.03, probe_spacing=0.02,
               max_probes=16384):
    """Lift vertices until their actual connecting chords clear sampled snow.

    All terrain samples are gathered in one ground_many call. Constraints are
    linear in the two neighbouring heights; raising either free vertex cannot
    invalidate an already satisfied constraint. Boundary vertices stay fixed.
    X/Z and point count stay unchanged. Terrain callbacks must depend on X/Z,
    as crossing.ground_many does, rather than the input point's Y.
    """
    pts = [list(map(float, p)) for p in points]
    if len(pts) < 3 or any(len(p) != 3 for p in pts):
        raise ValueError('at least three 3D rope points required')
    if not all(math.isfinite(v) for p in pts for v in p):
        raise ValueError('rope points must be finite')
    if not math.isfinite(clearance) or clearance < 0:
        raise ValueError('clearance must be finite and nonnegative')
    if not math.isfinite(probe_spacing) or probe_spacing <= 0:
        raise ValueError('probe_spacing must be finite and positive')
    original = tuple(tuple(p) for p in pts)
    probes = [(i, 0.0, tuple(p)) for i, p in enumerate(pts)]
    for i, (a, b) in enumerate(zip(pts[:-1], pts[1:])):
        n = max(1, math.ceil(math.dist(a, b) / probe_spacing))
        if len(probes) + n - 1 > max_probes:
            raise ValueError('rope probe budget exceeded')
        for k in range(1, n):
            u = k / n
            probes.append((i, u, tuple(a[j] + (b[j] - a[j]) * u for j in range(3))))
    heights = tuple(map(float, ground_many([p for _, _, p in probes])))
    if len(heights) != len(probes) or not all(math.isfinite(h) for h in heights):
        raise ValueError('ground_many returned invalid heights')
    for i in (0, len(pts) - 1):
        if pts[i][1] < heights[i] + clearance - 1e-10:
            raise ValueError('fixed attachment lies below requested snow clearance')
    # First satisfy vertices. The first len(pts) entries are their exact samples.
    for i in range(1, len(pts) - 1):
        pts[i][1] = max(pts[i][1], heights[i] + clearance)
    for (i, u, _), h in zip(probes[len(pts):], heights[len(pts):]):
        required = h + clearance
        current = (1.0 - u) * pts[i][1] + u * pts[i + 1][1]
        delta = max(0.0, required - current)
        if i == 0:
            pts[i + 1][1] += delta / u
        elif i + 1 == len(pts) - 1:
            pts[i][1] += delta / (1.0 - u)
        else:
            pts[i][1] += delta
            pts[i + 1][1] += delta
    def minimum(values):
        return min(((1.0 - u) * values[i][1] + u * values[i + 1][1]
                    if u else values[i][1]) - h
                   for (i, u, _), h in zip(probes, heights))
    achieved = minimum(pts)
    if achieved < clearance - 1e-9:
        raise RuntimeError('sampled chord clearance contract failed')
    return RopeSolution(tuple(tuple(p) for p in pts), len(probes),
                        max(p[1] - old[1] for p, old in zip(pts, original)),
                        achieved, minimum(original))


def draw_clear_rope(renderer, img, zb, scam, waists, lights, md, mf, diagnostics=None):
    """Original rope pass with only its world-space point construction replaced."""
    np = renderer.np
    W, L = np.array(waists), np.array(lights)
    for a, b in zip(W[:-1], W[1:]):
        solved = clear_rope(sag_points(a, b), renderer.ground_many)
        if diagnostics is not None:
            diagnostics.append(solved)
        pts = np.array(solved.points)
        sx, sy, z = scam.project(pts)
        if (z < 0.3).any():
            continue
        cols = np.zeros((len(pts), 3))
        for li in L:
            l2 = np.sum((pts - li[None, :3]) ** 2, axis=1)
            cols += li[3:6][None] * (li[6] / (l2 + li[7] ** 2))[:, None] * 0.6
        cols += np.array([0.42, 0.52, 0.72])[None] * 0.25 * mf
        cols *= np.array([0.16, 0.12, 0.08])[None] * 0.35
        for m in range(len(pts) - 1):
            zz = 0.5 * (z[m] + z[m + 1])
            wpx = max(0.012 * scam.f / zz, 0.5)
            alpha = min(0.012 * scam.f / zz / 0.5, 1.0) ** 1.5
            c = 0.5 * (cols[m] + cols[m + 1]) * alpha
            renderer._rope_seg(img, zb, sx[m], sy[m], sx[m + 1], sy[m + 1],
                               zz, wpx, c, alpha * 0.85)


# snow_decal (owner night, 29 Sep): this study replaced the original rasterizer's sub-segment MEAN depth and 5 cm
# margin with per-pixel depth and a terrain margin. Its rippled-depth unit test exercises the rope-cutting case.
# The lower-left dashes in farm A5043 were subsequently traced to ridge 3's unequal-width endcap (round_cap_rows
# below); they were initially attributed to the rope. Here each
# pixel is tested at its own interpolated depth, and where the front surface is the terrain (the depth captured just
# before the figures are drawn) the margin grows with distance, so the rope lies ON the snow as a line; where a
# figure is in front the original 5 cm margin still hides it. Geometry, width, colour and opacity are snow_clearance's.
DECAL_MARGIN = (0.06, 0.025)            # metres: terrain margin = a + b * depth


def _rope_seg_decal(np, img, zb, zt, x0, y0, z0, x1, y1, z1, w, c, alpha):
    H, W = img.shape[:2]
    xmin, xmax = max(int(min(x0, x1) - w - 2), 0), min(int(max(x0, x1) + w + 2), W)
    ymin, ymax = max(int(min(y0, y1) - w - 2), 0), min(int(max(y0, y1) + w + 2), H)
    if xmax <= xmin or ymax <= ymin:
        return
    yy, xx = np.mgrid[ymin:ymax, xmin:xmax]
    dx, dy = x1 - x0, y1 - y0
    L = math.sqrt(dx * dx + dy * dy) + 1e-9
    px, py = xx + 0.5 - x0, yy + 0.5 - y0
    u = np.clip((px * dx + py * dy) / (L * L), 0.0, 1.0)
    ex, ey = px - dx * u, py - dy * u
    cov = np.clip(0.5 * w + 0.5 - np.sqrt(ex * ex + ey * ey), 0.0, 1.0) * alpha
    zr = z0 + (z1 - z0) * u
    zp = zb[ymin:ymax, xmin:xmax]
    figure = zp < zt[ymin:ymax, xmin:xmax] - 0.02
    margin = np.where(figure, 0.05, DECAL_MARGIN[0] + DECAL_MARGIN[1] * zr)
    cov = np.where(zp >= zr - margin, cov, 0.0).astype(np.float32)
    a = max(alpha, 1e-3)
    sub = img[ymin:ymax, xmin:xmax]
    sub[:] = sub * (1 - cov[..., None]) + np.asarray(c, np.float32)[None, None, :] * (cov / a)[..., None]


def draw_decal_rope(renderer, state, img, zb, scam, waists, lights, md, mf):
    """snow_clearance's rope, rasterized as a decal on the terrain (see DECAL_MARGIN)."""
    np = renderer.np
    zt = state.get('zt', zb)
    W, L = np.array(waists), np.array(lights)
    for a, b in zip(W[:-1], W[1:]):
        pts = np.array(clear_rope(sag_points(a, b), renderer.ground_many).points)
        sx, sy, z = scam.project(pts)
        if (z < 0.3).any():
            continue
        cols = np.zeros((len(pts), 3))
        for li in L:
            l2 = np.sum((pts - li[None, :3]) ** 2, axis=1)
            cols += li[3:6][None] * (li[6] / (l2 + li[7] ** 2))[:, None] * 0.6
        cols += np.array([0.42, 0.52, 0.72])[None] * 0.25 * mf
        cols *= np.array([0.16, 0.12, 0.08])[None] * 0.35
        for m in range(len(pts) - 1):
            zz = 0.5 * (z[m] + z[m + 1])
            wpx = max(0.012 * scam.f / zz, 0.5)
            alpha = min(0.012 * scam.f / zz / 0.5, 1.0) ** 1.5
            c = 0.5 * (cols[m] + cols[m + 1]) * alpha
            _rope_seg_decal(np, img, zb, zt, float(sx[m]), float(sy[m]), float(z[m]), float(sx[m + 1]),
                            float(sy[m + 1]), float(z[m + 1]), wpx, c, alpha * 0.85)


def load_renderer():
    """Explicit render-time import; fail rather than regenerate a missing fire cache."""
    here = Path(__file__).resolve().parent
    source = here / 'crossing.py'
    if hashlib.sha256(source.read_bytes()).hexdigest() != BASELINE_SHA256:
        raise RuntimeError('crossing.py differs from the pinned study baseline')
    if not (here / 'crossing_fires.npy').is_file():
        raise FileNotFoundError('crossing_fires.npy missing; regeneration is not authorized')
    sys.path.insert(0, str(here))
    try:
        renderer = importlib.import_module('crossing')
    finally:
        sys.path.remove(str(here))
    if Path(renderer.__file__).resolve() != source:
        raise RuntimeError('crossing module came from another worktree')
    return renderer


def round_cap_rows(renderer, rows):
    """Copy the terrain table and opt in only the measured offending ridge.

    A5043: CR0[3]'s endcap projects to (100.52, 784.92) and (200.52, 755.56).
    Across 20 micrometres its unequal flank widths jump the combined ground by
    0.016730 and 0.009227 m (dash_terrain_seam.json, scalar height/projection
    probes). world.ridge column 16 joins those widths continuously on the cap.
    Match the complete pinned row so another scene cannot silently opt in.
    """
    np = renderer.np
    expected = renderer.CR0[3]
    if (rows.ndim != 2 or len(rows) <= 3 or rows.shape[1] <= 16
            or expected[12] != -1.0 or expected[16] != 0.0
            or not np.array_equal(rows[3], expected)):
        raise ValueError('round_cap requires the original crossing ridge 3')
    result = rows.copy()
    result[3, 16] = 1.0
    return result


def _round_cap_pass(renderer, original, table_index):
    def call(*args):
        args = list(args)
        args[table_index] = round_cap_rows(renderer, args[table_index])
        return original(*args)
    return call


def render_cut(frame, *, rope='accepted', rock='accepted', terrain='accepted', scale=0.5, ss=1.5,
               variant='main', trail=True, renderer=None, diagnostics=None,
               rock_modifier=None):
    """Return linear HDR at absolute A frame; options act independently.

    rock_modifier(scene, renderer, cfg) is an explicit study integration hook.
    Until a rock study is supplied, non-accepted rock options fail. It receives
    the freshly built scene and must return the replacement scene. This function
    does not finish/write the image, reset caches, or adjust thread settings.

    round_cap supplies a private terrain table to the march, shade (including
    shadows), and cloud-glow passes. Camera, path, lights, and figure construction
    keep the accepted terrain and caches; this repairs the drawn cap without
    restaging the crossing. Rendered foot contact still needs farm review.
    """
    if isinstance(frame, bool) or int(frame) != frame or not CUT0 <= frame <= CUT_END:
        raise ValueError('absolute A frame must be an integer in 4880..5839')
    if rope not in ROPE_OPTIONS:
        raise ValueError('unknown rope option')
    if terrain not in TERRAIN_OPTIONS:
        raise ValueError('unknown terrain option')
    if rock != 'accepted' and rock_modifier is None:
        raise ValueError('rock candidate requires an explicit rock_modifier')
    if rock == 'accepted' and rock_modifier is not None:
        raise ValueError('rock_modifier requires an explicit non-accepted rock option')
    cr = load_renderer() if renderer is None else renderer
    kwargs = dict(scale=scale, ss=ss, variant=variant, trail=trail)
    with _RENDER_LOCK:
        if rope == 'accepted' and rock == 'accepted' and terrain == 'accepted':
            return cr.render(int(frame) - CUT0, **kwargs)
        original_rope, original_scene = cr.draw_rope, cr.build_scene
        original_figures = getattr(cr, 'render_figures', None)     # only the decal replaces it
        terrain_hooks = []
        try:
            if terrain == 'round_cap':
                # Patch only draw passes. In particular, ground_many/heights
                # must not change cached placement or move a camera between calls.
                for name, index in (('march', 1), ('shade', 3), ('cloud_glow', 3)):
                    original = getattr(cr.WD, name)
                    terrain_hooks.append((name, original))
                    setattr(cr.WD, name, _round_cap_pass(cr, original, index))
            if rope == 'snow_clearance':
                def candidate_rope(*args):
                    return draw_clear_rope(cr, *args, diagnostics=diagnostics)
                cr.draw_rope = candidate_rope
            elif rope == 'snow_decal':
                state = {}
                def figures(fr, *args, **kw):
                    state['zt'] = fr.zb.copy()          # the terrain's depth, before any figure is drawn
                    return original_figures(fr, *args, **kw)
                def candidate_rope(*args):
                    return draw_decal_rope(cr, state, *args)
                cr.render_figures, cr.draw_rope = figures, candidate_rope
            if rock != 'accepted':
                def candidate_scene(t, cfg):
                    result = original_scene(t, cfg)
                    return (rock_modifier(result[0], cr, cfg), *result[1:])
                cr.build_scene = candidate_scene
            return cr.render(int(frame) - CUT0, **kwargs)
        finally:
            for name, original in terrain_hooks:
                setattr(cr.WD, name, original)
            cr.draw_rope, cr.build_scene = original_rope, original_scene
            if rope == 'snow_decal':
                cr.render_figures = original_figures
