"""B1 . DUSK (R8) on B's own landform (bworld): the range at sunset from high above the cloud sea.

The sun sets behind the camera; we look NNE over B's old, broad massifs toward her summit. The locked frame is marched
once (bworld.build: G-buffer + each pixel's clearance toward the sun's azimuth) and every frame is a relight at the
sun's elevation e(f): each point keeps the sun only while its grazing ray clears the cloud sea and the ranges (with
the Earth's curvature), and gets it through that grazing path (air mass; Rayleigh + a little aerosol), so the shadow
line climbs every face as the sun sinks and the lowest tops go dark first. e(f) is solved from the bar map: six
tops lose their last light on the downbeats of bars 2-7 (f 80..480) and her summit's last red point goes out on bar
8 b3 (f 600). H5 fixes: rose on snow (not orange rock), a soft-topped Earth's shadow under a broad Belt of Venus (no
seam), snow that follows the form (no fall-line streaks), no triangle anywhere.

  python shots/run/dusk.py --frames 0,240,480,560,600 --scale 0.25       # tests -> renders/run_b_tests/dusk_t
  python shots/run/dusk.py --build --scale 1.0 --ss 1.5                   # cache the G-buffer (cloud setup step)
  python shots/run/dusk.py --range 0-639 --scale 1.0 --ss 1.5 --out renders/dusk_B --skip
"""
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import bworld as BW         # noqa: E402
import rcam as RC           # noqa: E402
import pipe as PI           # noqa: E402

CM = PI.CM
look = PI.look
lin = CM.lin
CR = BW.CR_B
TOP = np.array([BW.TX, BW.TOP_Y, BW.TZ])
CACHE = os.path.join(CM.ROOT, 'renders', 'run_b_tests', 'cache')
TESTS = os.path.join(CM.ROOT, 'renders', 'run_b_tests')

# ------------------------------------------------------------------ camera ---
CAM_AZ_FROM_TOP = 212.0      # the camera sits SSW of her summit, looking NNE
CAM_DIST = 12000.0
CAM_Y = 650.0
HFOV = 38.0
LOOK_OFF = -5.0              # her summit a little right of centre
PITCH = -1.2
SUN_AZ = 205.0               # sets behind the camera, a little to its left

F0, F1 = 0, 640
SYNC_PEAKS = [80, 160, 240, 320, 400, 480]
SYNC_LAST = 600


def camera(W, H):
    a = math.radians(CAM_AZ_FROM_TOP)
    pos = np.array([BW.TX + CAM_DIST * math.sin(a), CAM_Y, BW.TZ + CAM_DIST * math.cos(a)])
    d = TOP - pos
    yaw = math.degrees(math.atan2(d[0], d[2])) + LOOK_OFF
    return RC.RCam(pos, yaw, PITCH, 0.0, HFOV, W, H)


def sun_dir(el_deg, az=SUN_AZ):
    return BW.sun_vec(el_deg, az)


# ------------------------------------------------------------------ light ---

def params(e, pix_ang, u=0.0):
    """LP / SKY / amb / fog at sun elevation e (deg); u = 0..1 through the shot (the Earth's shadow's climb)."""
    L = sun_dir(e)
    anti = -L.copy()
    anti[1] = 0.0
    anti /= np.linalg.norm(anti)
    dep = max(-e, 0.0)
    SK = np.zeros(24)
    SK[0:3] = anti
    SK[3] = math.radians(-1.6 + 6.0 * u ** 1.2)             # the Earth's shadow climbs through the shot (art-directed)
    SK[4:7] = lin('#1D2C63') * 0.50
    SK[7:10] = lin('#8792BF') * 0.44
    SK[10:13] = lin('#56628F') * 0.36
    SK[13:16] = lin('#EBA3B8') * 0.26 * (1.0 - 0.5 * u)
    SK[16] = math.radians(6.5)                               # a broad Belt of Venus...
    SK[17] = 1.0
    SK[18] = math.radians(5.0)                               # ...over a soft-topped shadow: never a seam
    LP = np.zeros(BW.NLP)
    LP[0:3] = L
    LP[3] = 5.0
    LP[4:7] = [0.020, 0.050, 0.080]                          # the grazing path: rose, not orange
    LP[7] = 0.018
    LP[8] = 0.0011
    LP[9] = 0.92
    LP[10] = 2.4
    LP[11] = 0.030 * (1.0 - u)                               # the belt's faint rose on the west faces, fading
    LP[12] = 0.30
    LP[13:16] = sun_dir(6.0)
    LP[16:19] = lin('#E9A0A0') * 0.028
    LP[19] = 0.0
    LP[20] = pix_ang
    LP[30] = 0.55                                            # her last point stays a vivid red to the end...
    LP[31] = 0.00012                                         # ...and goes out crisply on bar 8 b3
    LP[32] = -1
    LP[33] = -1
    LP[44] = 1.0
    amb = lin('#5A6CA8') * 0.30
    # the far air takes the horizon sky's own colour, so each farther range fades a step further into the sky
    fogc = (lin('#8792BF') * 0.44 * (1.0 - 0.45 * u) + lin('#56628F') * 0.30 * 0.45 * u) * 0.95
    fogp = np.zeros(16)
    fogp[:8] = [1.15e-4, 1 / 3600.0, 4.0e-5, 1 / 160.0, 0.0, fogc[0], fogc[1], fogc[2]]
    fogp[8] = 8.0
    fogp[9:12] = fogc
    fogp[12:15] = L
    return LP, SK, amb, fogp


class DuskShot:
    def __init__(self, scale=0.25, ss=1.5, cache=True):
        self.scale, self.ss = scale, ss
        W, H = int(round(1920 * scale)), int(round(804 * scale))
        self.tc = camera(W, H)
        self.fr = PI.Frame(self.tc, ss)
        scam = self.fr.src
        self.P = np.array([scam.pos[0], scam.pos[2], 0.0, 0.0])
        p = os.path.join(CACHE, f'bdusk_G_{BW.VERSION}_{scale:.3f}_{ss:.2f}.npy')
        if cache and os.path.exists(p):
            self.G = np.load(p)
        else:
            self.G = BW.build(scam, self.P, CR, SUN_AZ, nsteps=88, tmax=160000.0)
            if cache:
                os.makedirs(CACHE, exist_ok=True)
                tmp = p + f'.{os.getpid()}.tmp.npy'
                np.save(tmp, self.G)
                os.replace(tmp, p)
        self.clamp_to_her()
        self.schedule()

    def clamp_to_her(self, margin=0.05):
        """Art direction: her summit holds the last red point. Every lit pixel off her summit (more than 160 m from
        her top, or 80 m below it) loses its light at least `margin` deg of sun elevation before her tip does."""
        G = self.G
        rock = G[..., BW.G_FLAG] == 1.0
        dxz = np.hypot(G[..., BW.G_X] - BW.TX, G[..., BW.G_Z] - BW.TZ)
        hers = rock & (dxz < 160.0) & (G[..., BW.G_Y] > BW.TOP_Y - 80.0)
        c0 = G[..., BW.G_C0]
        c_her = float(c0[hers].max()) if hers.any() else float(c0[rock].max())
        e_her = math.degrees(math.asin(np.clip(-c_her - 0.0011, -0.2, 0.2)))
        cap = -math.sin(math.radians(e_her + margin)) - 0.0011
        G[..., BW.G_C0] = np.where(rock & ~hers, np.minimum(c0, cap), c0)
        G[..., BW.G_TROUGH] = np.where(hers, 2.0, G[..., BW.G_TROUGH])        # tag her summit (hero)
        # the glint: the knoll's top (within 26 m of her top, on the lit side) goes out together with its last pixel
        top_ = hers & (dxz < 26.0) & (G[..., BW.G_Y] > BW.TOP_Y - 9.0)
        if top_.any():
            G[..., BW.G_C0] = np.where(top_, max(float(c0[hers].max()), float(c0[top_].max())), G[..., BW.G_C0])
        self.her_mask = hers

    def dark_el(self):
        """Per pixel: the sun elevation (deg) at which the pixel's last direct light goes (land only)."""
        c0 = self.G[..., BW.G_C0]
        rock = self.G[..., BW.G_FLAG] == 1.0
        pen = np.where(self.her_mask, 0.00035, 0.0011)
        s = np.clip(-c0 - pen, -0.2, 0.2)
        e = np.degrees(np.arcsin(s))
        return np.where(rock, e, -9.0)

    def schedule(self):
        """Pick the six bar-map tops and solve e(f)."""
        from scipy.ndimage import minimum_filter
        from scipy.interpolate import PchipInterpolator
        E = self.dark_el()
        scam = self.fr.src
        hx, hy, hz = scam.project(TOP + np.array([0.0, 2.0, 0.0]))
        y0, x0 = int(hy), int(hx)
        self.e_her = float(E[self.her_mask].min()) if self.her_mask.any() else float(E[E > -8].min())
        k = max(int(round(18 * self.scale * self.ss)), 3)
        mn = minimum_filter(np.where(E > -8, E, 9.0), size=k)
        tips = np.argwhere((E == mn) & (E > -8) & (E < 6.0))
        cand = []
        for j, i in tips:
            if abs(i - x0) < 3 * k and abs(j - y0) < 3 * k:
                continue
            area = int(((E[max(j - k, 0):j + k, max(i - k, 0):i + k] < E[j, i] + 0.12)
                        & (E[max(j - k, 0):j + k, max(i - k, 0):i + k] > -8)).sum())
            cand.append((float(E[j, i]), int(i), int(j), area))
        cand = [c for c in cand if c[0] > self.e_her + 0.08 and c[3] >= 4]
        cand.sort(key=lambda c: -c[0])
        e_hi = cand[0][0] if cand else self.e_her + 0.8
        targets = np.linspace(min(e_hi, self.e_her + 1.1), self.e_her + 0.10, 6)
        chosen = []
        for tg in targets:
            best = None
            for c in cand:
                if any(abs(c[1] - q[1]) < 2 * k and abs(c[2] - q[2]) < 2 * k for q in chosen):
                    continue
                if chosen and c[0] >= chosen[-1][0] - 0.02:
                    continue
                score = abs(c[0] - tg) * 8.0 - min(c[3], 400) / 400.0
                if best is None or score < best[0]:
                    best = (score, c)
            if best is not None:
                chosen.append(best[1])
        self.chosen = chosen
        # from the sixth top on, only her summit holds light: everything else is capped to go dark with it
        if chosen:
            e6 = chosen[-1][0]
            cap = -math.sin(math.radians(e6)) - 0.0011
            G = self.G
            rock = G[..., BW.G_FLAG] == 1.0
            G[..., BW.G_C0] = np.where(rock & ~self.her_mask, np.minimum(G[..., BW.G_C0], cap), G[..., BW.G_C0])
        # her last point is half gone exactly on bar 8 b3 (its last pixel's clearance crosses zero there)
        if self.her_mask.any():
            c_top = float(self.G[..., BW.G_C0][self.her_mask].max())
            self.e_out = math.degrees(math.asin(min(max(-c_top, -0.2), 0.2)))
        else:                                   # (tiny test scales only: her summit covers no pixel)
            self.e_out = self.e_her
        fs = [F0] + SYNC_PEAKS[:len(chosen)] + [SYNC_LAST, F1 - 1]
        es = [chosen[0][0] + 1.6 if chosen else self.e_out + 2.4] + [c[0] for c in chosen] + \
            [self.e_out, self.e_out - 0.05]
        self.e_of = PchipInterpolator(np.array(fs, float), np.array(es, float))

    def render(self, frame):
        e = float(self.e_of(frame))
        scam = self.fr.src
        LP, SK, amb, fogp = params(e, 1.0 / scam.f, min(max(frame / float(SYNC_LAST), 0.0), 1.2))
        out = np.zeros((scam.H, scam.W, 3), np.float32)
        BW.shade(self.G, LP, SK, amb, fogp, float(scam.pos[1]), out)
        e_low = self.chosen[0][0] if self.chosen else self.e_her + 1.0
        out = wisps(out, self.G, scam.pos, e, e_low, 1.0 / scam.f)
        fr = self.fr
        fr.img = out
        dist = self.G[..., BW.G_DIST]
        fr.zb = dist
        fr.dist = dist
        img, _, _ = PI.to_target(fr)
        return img


from numba import njit, prange   # noqa: E402
from mt.noise import fbm2 as _fbm2   # noqa: E402


@njit(parallel=True, fastmath=True, cache=True)
def _fbm_arr(u, v, out):
    for i in prange(u.shape[0]):
        out[i] = _fbm2(u[i], v[i], 4.0, 311)


def wisps(img, G, cam_pos, e, e_low, pix):
    """A few thin wisps of cloud just above the cloud sea: streaks drawn out along the wind, sky-lit lavender, and
    rose while the sun still reaches their height (they go blue with the lowest tops). Vectorised over the G-buffer."""
    dy = G[..., BW.G_DY]
    dist = G[..., BW.G_DIST]
    yw = BW.CLOUD_Y + 200.0
    with np.errstate(divide='ignore', invalid='ignore'):
        t = (yw - cam_pos[1]) / dy
    ok = (dy < -1e-4) & (t > 0) & (t < dist) & (t < 90000.0)
    if not ok.any():
        return img
    t = np.where(ok, t, 0.0)
    # the curved Earth drops the plane with distance: one correction step
    t2 = (yw - cam_pos[1] - t * t / (2 * BW.R_EARTH)) / np.where(ok, dy, -1.0)
    t = np.where(ok, np.minimum(t2, dist), 0.0)
    x = cam_pos[0] + G[..., BW.G_DX] * t
    z = cam_pos[2] + G[..., BW.G_DZ] * t
    jj, ii = np.nonzero(ok)
    xs, zs = x[jj, ii], z[jj, ii]
    ca, sa = math.cos(math.radians(30.0)), math.sin(math.radians(30.0))
    u = (xs * ca + zs * sa) / 3200.0
    v = (-xs * sa + zs * ca) / 700.0
    n = np.zeros(len(u))
    _fbm_arr(u.astype(np.float64), v.astype(np.float64), n)
    d = np.clip((n - 0.18) / 0.35, 0.0, 1.0)
    d = d * d * (3 - 2 * d) * np.clip(1.0 - t[jj, ii] / 90000.0, 0.0, 1.0) ** 0.5
    lit = min(max((e - (e_low - 0.35)) / 0.5, 0.0), 1.0)
    col = lin('#6E78A8') * 0.40 * (1.0 - lit) + (lin('#F2A6B4') * 0.55 * lit + lin('#6E78A8') * 0.30 * lit)
    a = 0.32 * d
    img[jj, ii, :] = img[jj, ii, :] * (1.0 - a[:, None]) + col[None, :] * a[:, None]
    return img


FINISH = dict(exposure=1.05, bloom_strength=0.05, bloom_threshold=1.2, streak_strength=0.0, vignette_amount=0.22)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--build', action='store_true', help='only march + cache the G-buffer')
    ap.add_argument('--range', default=None, help='frames a-b (inclusive), 0-639')
    ap.add_argument('--frames', default=None, help='frames, comma list')
    ap.add_argument('--scale', type=float, default=0.25)
    ap.add_argument('--ss', type=float, default=1.5)
    ap.add_argument('--out', default='dusk_t')
    ap.add_argument('--skip', action='store_true')
    a = ap.parse_args()
    t0 = time.time()
    shot = DuskShot(a.scale, a.ss)
    print(f'G-buffer {time.time() - t0:.1f}s; her last point at e={shot.e_her:.3f}; tops '
          + ', '.join(f'{c[0]:.3f}@({c[1]},{c[2]})' for c in shot.chosen), flush=True)
    if a.build:
        return
    out = os.path.join(CM.ROOT, a.out) if a.out.startswith('renders/') else os.path.join(TESTS, a.out)
    os.makedirs(out, exist_ok=True)
    if a.range:
        s0, s1 = a.range.split('-')
        frames = list(range(int(s0), int(s1) + 1))
    else:
        frames = [int(x) for x in (a.frames or '0,240,480,600').split(',')]
    if a.skip:
        frames = [f for f in frames if not os.path.exists(look.find_frame(out, f))]   # find_frame never returns None
    for f in frames:
        t1 = time.time()
        img = look.finish(shot.render(f), **FINISH)
        look.save_png(look.frame_path(out, f), img)
        print(f'frame {f} e={float(shot.e_of(f)):+.3f} {time.time() - t1:.2f}s', flush=True)


if __name__ == '__main__':
    main()
