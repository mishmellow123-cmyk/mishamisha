"""Default-off RUN-A4 studies; accepted crossing.py remains untouched.

render_cut() uses absolute A frames. It returns the original linear HDR image;
the caller applies crossing.PI.look.finish(..., **crossing.FINISH) once.
Importing this module loads no renderer, NumPy, OpenCV, or Numba and sets no
thread/cache options. Render calls belong in an isolated, single-render process:
candidate hooks temporarily replace two globals of the original renderer.

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
ROPE_OPTIONS = ('accepted', 'snow_clearance')
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


def render_cut(frame, *, rope='accepted', rock='accepted', scale=0.5, ss=1.5,
               variant='main', trail=True, renderer=None, diagnostics=None,
               rock_modifier=None):
    """Return linear HDR at absolute A frame; options act independently.

    rock_modifier(scene, renderer, cfg) is an explicit study integration hook.
    Until a rock study is supplied, non-accepted rock options fail. It receives
    the freshly built scene and must return the replacement scene. This function
    does not finish/write the image, reset caches, or adjust thread settings.
    """
    if isinstance(frame, bool) or int(frame) != frame or not CUT0 <= frame <= CUT_END:
        raise ValueError('absolute A frame must be an integer in 4880..5839')
    if rope not in ROPE_OPTIONS:
        raise ValueError('unknown rope option')
    if rock != 'accepted' and rock_modifier is None:
        raise ValueError('rock candidate requires an explicit rock_modifier')
    if rock == 'accepted' and rock_modifier is not None:
        raise ValueError('rock_modifier requires an explicit non-accepted rock option')
    cr = load_renderer() if renderer is None else renderer
    kwargs = dict(scale=scale, ss=ss, variant=variant, trail=trail)
    with _RENDER_LOCK:
        if rope == 'accepted' and rock == 'accepted':
            return cr.render(int(frame) - CUT0, **kwargs)
        original_rope, original_scene = cr.draw_rope, cr.build_scene
        try:
            if rope == 'snow_clearance':
                def candidate_rope(*args):
                    return draw_clear_rope(cr, *args, diagnostics=diagnostics)
                cr.draw_rope = candidate_rope
            if rock != 'accepted':
                def candidate_scene(t, cfg):
                    result = original_scene(t, cfg)
                    return (rock_modifier(result[0], cr, cfg), *result[1:])
                cr.build_scene = candidate_scene
            return cr.render(int(frame) - CUT0, **kwargs)
        finally:
            cr.draw_rope, cr.build_scene = original_rope, original_scene
