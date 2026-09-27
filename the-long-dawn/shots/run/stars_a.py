"""X2 DARK ADAPTATION (cut A, A11): A cut frames 3120-3359 (bars 40-42) -> renders/stars_A/ (EDIT-v3's T('stars')).

Out of the black the stars come back the way an eye adapts to the dark: a few of the brightest on bar 40 b1, then
more and fainter, then the Milky Way by bar 42 b1. EMBERS' ember from A10 carries straight on without a jump. It
rests at (960, 548) = (0.5 W, 0.68 H) with EMBERS' core and halo (fitted to their A10 frame 3119) and flickers on
EMBERS' own clock. It never dies, and it is one light among the stars.
The night is H1's moonlit night (A12-A15), so the camera looks away from the moon (az -59, el 21), toward az ~110.
There the sky is darkest and the band of the Milky Way comes down to the ranges. The ranges along the bottom emerge
last and faintly, moonlit from behind the camera. There is no fire and no glow, because the reveal belongs to A13.

  farm: python3 the-long-dawn/cloud/farm.py the-long-dawn/cloud/jobs/stars_a.json --test 4
  local: python3 shots/run/stars_a.py --frames 3130,3200,3280,3359 --scale 0.5 --out tests
"""
import argparse
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ.setdefault('NUMBA_NUM_THREADS', '1')

import rcam as RC           # noqa: E402
import world as WD          # noqa: E402
import pipe as PI           # noqa: E402
import falsedawn as FD      # noqa: E402
from mt import sky as SK    # noqa: E402

FR0, NFR = 3120, 240                      # A11: bars 40-42
T_BLACK = 2800                            # EMBERS' A10 clock (edge.LivingEmber) starts on A cut 2800
lin = PI.CM.lin

# the camera: A2's ridge, turned away from the moon (az -59) and the glow (az -30), tilted up into the stars
CAM = np.array([60.0, 172.0, 480.0])
YAW = float(os.environ.get('X2_YAW', '110.0'))
PITCH = float(os.environ.get('X2_PITCH', '12.0'))
HFOV = float(os.environ.get('X2_HFOV', '60.0'))


def camera(W=1920, H=804):
    return RC.RCam(CAM, YAW, PITCH, 0.0, HFOV, W, H)


# ------------------------------------------------------------------ adaptation (u = frames since bar 40 b1) ---
def smooth(a, b, x):
    t = min(max((x - a) / (b - a), 0.0), 1.0)
    return t * t * (3 - 2 * t)


def n_visible(u, n_in):
    """How many of the frame's stars the eye has found: 3 on bar 40 b1, then exponentially more, all by bar 42 b1."""
    s = min(max(u / 160.0, 0.0), 1.0) ** 0.85
    return 3.0 * (n_in / 3.0) ** s


def sky_k(u):
    return smooth(8.0, 175.0, u)            # the night's own glow (airglow) creeps up out of black


def mw_k(u):
    return smooth(80.0, 160.0, u)           # the Milky Way last, fully in by bar 42 b1


def ter_k(u):
    return 0.55 * smooth(70.0, 230.0, u)    # the snow of the skyline, barely


# ------------------------------------------------------------------ the ember (EMBERS edge.LivingEmber, cut A) ---
# Position: EMBERS' measured centroid at 3119, eased to the director's rest point; look fitted to that frame through the same finish
# (their black-section finish = this shot's FINISH): an HDR core 8.08 (sigma 1.0 px) + a halo 0.26 exp(-r / 10.0 px),
# blackbody(temp) and blackbody(temp - 0.1), at their life lv = 0.84. Scaled per frame by EMBERS' flicker.
EMB_X0, EMB_Y0 = 958.6, 547.5                   # EMBERS' final A10 frame 3119 (measured centroid, embers_A3)
EMB_X1, EMB_Y1 = 960.0, 548.0                   # the director's rest point, reached over two bars (1.4 px: unseen)
LV_FIT = 0.84


def ember_pos(t):
    u = min(max((t - FR0) / 160.0, 0.0), 1.0)
    e = u * u * (3 - 2 * u)
    return EMB_X0 + (EMB_X1 - EMB_X0) * e, EMB_Y0 + (EMB_Y1 - EMB_Y0) * e


def ember_life(t):
    """EMBERS' LivingEmber.life on A's cut clock (u = t - 2800); past A10 the wake is full and the dips are over."""
    u = t - T_BLACK
    fl = 1 + 0.28 * math.sin(u * 0.9) * math.sin(u * 0.23) + 0.12 * math.sin(u * 2.7 + 1.0)
    return 0.62 * fl


def add_ember(img, t):
    H, W = img.shape[:2]
    s = W / 1920.0
    ex, ey = ember_pos(t)
    ex, ey = ex * s, ey * s                      # continuous pixel coords (pixel i spans [i, i+1))
    lv = ember_life(t)
    k = lv / LV_FIT
    temp = 0.47 + 0.3 * lv
    c_core = PI.look.blackbody(temp)
    c_halo = PI.look.blackbody(temp - 0.1)
    r = int(70 * s) + 4
    x0, y0 = int(ex) - r, int(ey) - r
    yy, xx = np.mgrid[y0:y0 + 2 * r + 1, x0:x0 + 2 * r + 1].astype(np.float32)
    r2 = (xx + 0.5 - ex) ** 2 + (yy + 0.5 - ey) ** 2
    sc = max(1.0 * s, 0.5)
    core = 8.08 * k * np.exp(-r2 / (2 * sc * sc))
    halo = 0.26 * k * np.exp(-np.sqrt(r2) / (10.0 * s))
    patch = core[..., None] * c_core[None, None, :] + halo[..., None] * c_halo[None, None, :]
    ya, yb, xa, xb = max(y0, 0), min(y0 + 2 * r + 1, H), max(x0, 0), min(x0 + 2 * r + 1, W)
    if ya < yb and xa < xb:
        img[ya:yb, xa:xb] += patch[ya - y0:yb - y0, xa - x0:xb - x0]
    return img


# ------------------------------------------------------------------ the static parts (once per process) ---
_ST = {}


def static(scale, ss):
    key = (scale, ss)
    if key in _ST:
        return _ST[key]
    W, H = int(round(1920 * scale)), int(round(804 * scale))
    fr = PI.Frame(camera(W, H), ss)
    _, GP = FD.light(0, 'arc')                  # A2's sky model with the glow off (swell 0)
    PI.render_terrain(fr, 0, FD.terrain_rows(), WD.night_light(), np.zeros((0, 8)))   # H1's moonlit night
    scam = fr.src
    stars = FD._STARS if FD._STARS is not None else SK.make_stars(16000, 101, lum_scale=6.0)
    # the frame's own stars, ranked by brightness (the eye finds the brightest first)
    d = stars['dir']
    x = d[:, 0] * scam.right[0] + d[:, 2] * scam.right[2]
    z = d[:, 0] * scam.fwd[0] + d[:, 2] * scam.fwd[2]
    y = d[:, 1]
    ok = z > 0.05
    sx = np.full(len(y), -1e9)
    sy = np.full(len(y), -1e9)
    sx[ok] = scam.cx + scam.f * x[ok] / z[ok]
    sy[ok] = scam.cy + scam.shift - scam.f * y[ok] / z[ok]
    Hs, Ws = fr.dist.shape
    inside = ok & (sx >= 0) & (sx < Ws) & (sy >= 0) & (sy < Hs)
    ix = np.clip(sx.astype(int), 0, Ws - 1)
    iy = np.clip(sy.astype(int), 0, Hs - 1)
    inside &= fr.dist[iy, ix] > 1e8              # in the sky, not behind the skyline
    sub = {k: v[inside] for k, v in stars.items()}
    order = np.argsort(-sub['lum'])
    rank = np.empty(len(order))
    rank[order] = np.arange(len(order))
    st = dict(fr=fr, terr=fr.img.copy(), dist=fr.dist.copy(), zb=fr.zb.copy(), GP=GP, stars=sub, rank=rank,
              skl=FD.skyline(scam.pos))
    _ST[key] = st
    return st


def render(frame, scale=1.0, ss=1.5):
    """Linear HDR image (H, W, 3) for A cut frame `frame`."""
    st = static(scale, ss)
    u = frame - FR0
    fr = st['fr']
    fr.img = st['terr'] * ter_k(u)
    fr.dist = st['dist']
    fr.zb = st['zb']
    GP = st['GP'].copy()
    GP[2] = 0.0                                  # no glow at all: it is behind us
    GP[16] = 0.0                                 # a clear sky, no deck
    GP[20:26] *= sky_k(u)
    GP[26] = 0.09 * mw_k(u)
    scam = fr.src
    C = scam.params()
    skl0, skld, skl = st['skl']
    trans = np.zeros(fr.dist.shape, np.float32)
    FD.sky_pass(fr.img, fr.dist, trans, C, GP, FD.milky_way(), skl0, skld, skl, scam.pos[1], frame / 24.0,
                np.zeros((0, 6)))
    stars = dict(st['stars'])
    n_in = max(len(st['rank']), 4)
    nv = n_visible(u, n_in)
    # each star fades in over the span of the next ~35% more stars, so none pops
    g = np.clip((nv - st['rank']) / (0.35 * nv + 1.0), 0.0, 1.0)
    g = g * g * (3 - 2 * g)
    stars['lum'] = stars['lum'] * g
    ss_ = fr.ss
    SK.splat_stars(fr.img, scam, stars, trans, t=frame / 24.0, gain=ss_ * ss_, scale=PI.src_scale(fr))
    img, zb, di = PI.to_target(fr)
    return add_ember(img, frame)


FINISH = dict(exposure=1.0, bloom_strength=0.1, bloom_threshold=0.7, streak_strength=0.0, vignette_amount=0.25)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--frames', default=None)
    ap.add_argument('--range', default=None)
    ap.add_argument('--scale', type=float, default=1.0)
    ap.add_argument('--ss', type=float, default=1.5)
    ap.add_argument('--out', default=None)
    ap.add_argument('--threads', type=int, default=2)
    ap.add_argument('--procs', type=int, default=1)
    ap.add_argument('--skip', action='store_true')
    a = ap.parse_args()
    base = os.path.join(PI.CM.ROOT, 'renders', 'stars_A')
    out = base if a.out is None else os.path.join(base, a.out)
    if a.range:
        s, e = a.range.split('-')
        frames = list(range(int(s), int(e) + 1))
    else:
        frames = [int(x) for x in (a.frames or '3280').split(',')]
    bad = [f for f in frames if not FR0 <= f < FR0 + NFR]
    if bad:
        raise SystemExit(f'frames outside A11 ({FR0}-{FR0 + NFR - 1}): {bad[:5]}')
    os.makedirs(out, exist_ok=True)
    look = PI.look
    todo = [f for f in frames if not (a.skip and os.path.exists(look.frame_path(out, f)))]
    if a.procs <= 1:
        _work((todo, a.scale, a.ss, out, a.threads))
    else:
        import multiprocessing as mp
        chunks = [todo[i::a.procs] for i in range(a.procs)]
        with mp.get_context('spawn').Pool(a.procs) as pool:
            pool.map(_work, [(c, a.scale, a.ss, out, 1) for c in chunks])


def _work(args):
    frames, scale, ss, od, threads = args
    import numba                        # NUMBA_NUM_THREADS is fixed once numba runs (the farm sets it per node)
    numba.set_num_threads(max(1, min(threads, numba.config.NUMBA_NUM_THREADS)))
    look = PI.look
    for f in frames:
        t0 = time.time()
        img = look.finish(render(f, scale, ss), **FINISH)
        look.save_png(look.frame_path(od, f), img)
        print(f'X2 frame {f} {time.time() - t0:.1f}s', flush=True)


if __name__ == '__main__':
    main()
