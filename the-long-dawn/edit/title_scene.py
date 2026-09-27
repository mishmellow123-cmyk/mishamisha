"""THE LONG DAWN v3: the scenes of the X3 ember titles (A20, B14): their clocks, spark sources and stand-in skies.

Light on purpose (numpy + cv2 only): the edit (assemble.Ctx) and the title renderer (ember_title_v3) both use it.

Spark sources are the fires IN THE PLATE: warm bright points found in the sky plate's first frame (A: RUN-A's
watch-fires in the blue-hour valley; B: her beacon and the last far fires at dawn). Until a plate lands, the
stand-in sky below is shown (an EDIT proxy) and the stand-in fires are drawn in it, so every spark has a source.
"""
import os

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
W, H = 1920, 804
SPAN = {'A': (6240, 6480), 'B': (5200, 5440)}
# v3 frame -> the engine's own clock (v2 frames: gather from 2807, letters land 2853-2886, hold, loosen from 2927,
# dark by 2961): a monotone curve through these points (A: kindles, holds, crumbles by the end of its text window
# 6440, before the fade to black from 6456; B: kindles and holds, never loosens: it fades into the light instead)
CLOCK = {'A': [(6240, 2811.0), (6300, 2853.0), (6330, 2880.0), (6400, 2924.0), (6420, 2936.0), (6456, 2958.0),
               (6480, 2966.0)],
         'B': [(5200, 2811.0), (5260, 2853.0), (5290, 2880.0), (5440, 2924.0)]}
FADE = {'B': (5340, 5400)}            # B: the title fades into the light over the last 60 f of its window
TITLE_Y = 360                          # titles.py v3: the A and B titles sit in the sky (Cinzel 92, 500, 0.28)


def clock(cut, f):
    from scipy.interpolate import PchipInterpolator
    x, y = zip(*CLOCK[cut])
    return float(PchipInterpolator(x, y, extrapolate=True)(f))


def fade(cut, f):
    if cut not in FADE:
        return 1.0
    a, b = FADE[cut]
    t = min(max((f - a) / (b - a), 0.0), 1.0)
    return 1.0 - t * t * (3 - 2 * t)


# ------------------------------------------------------------------------------ the stand-in fires ---
def _ridges(seed=81):
    """A: three knife-edge ranges, far to near, as screen-space profiles y(x): ridged, serrated, no clean peaks."""
    rng = np.random.default_rng(seed)
    xs = np.arange(W, dtype=np.float64)
    out = []
    for base, amp, freq in ((515.0, 105.0, 3.0), (598.0, 135.0, 4.0), (705.0, 150.0, 3.2)):
        r, wsum = np.zeros(W), 0.0
        for o in range(6):                                 # linear knots give hard kinks; ridging makes crests
            n = int(freq * 2 ** o) + 2
            v = np.interp(xs, np.linspace(-60, W + 60, n), rng.uniform(0, 1, n))
            r += (1 - np.abs(2 * v - 1)) ** 2 * 0.52 ** o
            wsum += 0.52 ** o
        r = (r / wsum - (r / wsum).min()) / np.ptp(r / wsum)
        out.append(base - amp * r ** 1.3)
    return out


RIDGES_A = _ridges()


def standin_fires(cut):
    """[(x, y, flame height px)] of the stand-in fires (screen space, full res)."""
    if cut == 'A':                                         # watch-fires on the shoulders of the near range
        near = RIDGES_A[2]
        xs = np.linspace(170, 1750, 11) + np.random.default_rng(7).uniform(-40, 40, 11)
        return [(float(x), float(near[int(x)] - 3.0), 7.0) for x in xs]
    # B: on dusk_B's first frame (the plate used as B14's stand-in): her beacon on the great massif, far beacons
    return [(1200.0, 437.0, 9.0), (225.0, 497.0, 5.0), (560.0, 612.0, 5.0), (905.0, 688.0, 6.0),
            (1330.0, 698.0, 6.0), (1655.0, 680.0, 5.0), (1802.0, 470.0, 4.0)]


def detect_fires(img_srgb, y_min=0.45, max_n=24):
    """Warm bright points in a plate (float sRGB HxWx3, full res) below y_min*H: [(x, y, flame px)]."""
    r, g, b = img_srgb[..., 0], img_srgb[..., 1], img_srgb[..., 2]
    m = ((r > 0.55) & (r - b > 0.25) & (g > 0.18)).astype(np.uint8)
    m[:int(y_min * img_srgb.shape[0])] = 0
    n, lab, st, cen = cv2.connectedComponentsWithStats(m, 8)
    out = []
    for k in range(1, n):
        area = st[k, cv2.CC_STAT_AREA]
        if area >= 3:
            out.append((float(cen[k][0]), float(st[k, cv2.CC_STAT_TOP] + st[k, cv2.CC_STAT_HEIGHT]),
                        float(min(24.0, max(4.0, 1.3 * st[k, cv2.CC_STAT_HEIGHT]))), area))
    out.sort(key=lambda t: -t[3])
    return [t[:3] for t in out[:max_n]]


# ------------------------------------------------------------------------------ the stand-in skies ---
def _lin(x):
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def _srgb(x):
    x = np.clip(x, 0, None)
    return np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(x, 1 / 2.4) - 0.055)


_B_PLATE = {}


def standin_sky(cut, f, w, h):
    """The EDIT proxy under the title until the plate lands (float sRGB hxwx3 at w x h)."""
    u = (f - SPAN[cut][0]) / float(SPAN[cut][1] - SPAN[cut][0])
    s = w / float(W)
    if cut == 'B':                                         # dusk_B's first frame, the light growing into dawn
        if 'img' not in _B_PLATE:
            p = os.path.join(ROOT, 'renders', 'dusk_B', 'f_00000.jpg')
            im = cv2.imread(p)
            _B_PLATE['img'] = None if im is None else _lin(im[..., ::-1].astype(np.float32) / 255.0)
        base = _B_PLATE['img']
        if base is None:
            base = np.zeros((H, W, 3), np.float32) + np.array([0.05, 0.04, 0.08], np.float32)
        lin = cv2.resize(base, (w, h), interpolation=cv2.INTER_AREA)
        grow = u * u * (3 - 2 * u)
        lin = lin * (1.0 + 1.6 * grow) * np.array([1.0 + 0.10 * grow, 1.0 + 0.04 * grow, 1.0 - 0.06 * grow], np.float32)
        lin = lin + (0.22 * grow ** 2) * np.array([1.0, 0.90, 0.76], np.float32)     # into the light
        pale = 1.0 - 0.85 * grow                           # the beacons pale as the sun comes
        return _srgb(_glows(lin, standin_fires('B'), s, f, pale)).astype(np.float32)
    # A: the blue hour over knife-edge ranges; the east pales to rose while the title plays
    yy = np.linspace(0, 1, h, dtype=np.float32)[:, None, None]
    xx = np.linspace(0, 1, w, dtype=np.float32)[None, :, None]
    top = np.array([0.012, 0.022, 0.070], np.float32) * (1 - u) + np.array([0.030, 0.050, 0.120], np.float32) * u
    hor = np.array([0.20, 0.100, 0.125], np.float32) * (1 - u) + np.array([0.46, 0.25, 0.23], np.float32) * u
    east = 0.55 + 0.45 * xx                                # the rose is warmest to the east (right)
    t = np.clip(yy / 0.62, 0, 1) ** 1.7
    sky = top * (1 - t) + hor * east * t
    lin = np.broadcast_to(sky, (h, w, 3)).copy()
    haze = np.array([0.075, 0.060, 0.090], np.float32) * (1 + 0.8 * u)
    ys = np.arange(h, dtype=np.float32)[:, None]
    for k, (prof, tone) in enumerate(zip(RIDGES_A, (0.55, 0.32, 0.12))):
        py = cv2.resize(prof[None, :].astype(np.float32), (w, 1), interpolation=cv2.INTER_AREA)[0] * s
        a = np.clip((ys - py[None, :]) / max(0.8, 1.2 * s) + 0.5, 0, 1)[..., None]       # soft 1-px edge
        lin = lin * (1 - a) + (haze * tone + sky[-1] * 0.10 * tone) * a
    rng = np.random.default_rng(int(f))
    lin = lin + rng.normal(0, 0.0012, lin.shape).astype(np.float32)                # no banding in the gradient
    return _srgb(_glows(lin, standin_fires('A'), s, f, 1.0)).astype(np.float32)


def _glows(lin, fires, s, f, gain):
    h, w = lin.shape[:2]
    for k, (x, y, hp) in enumerate(fires):
        x, y, r = x * s, y * s, max(1.0, hp * s)
        fl = 0.85 + 0.15 * np.sin(0.9 * f + 2.1 * k) * np.sin(0.37 * f + k)
        x0, x1, y0, y1 = int(max(0, x - 8 * r)), int(min(w, x + 8 * r + 1)), int(max(0, y - 8 * r)), int(min(h, y + 3 * r))
        if x1 <= x0 or y1 <= y0:
            continue
        gy, gx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
        d2 = ((gx - x) / (0.45 * r)) ** 2 + ((gy - (y - 0.45 * r)) / (0.8 * r)) ** 2          # a small upright flame
        core = np.exp(-d2) * 2.2
        halo = np.exp(-(((gx - x) ** 2 + (gy - y + 0.3 * r) ** 2) / (2 * (2.6 * r) ** 2))) * 0.10
        lin[y0:y1, x0:x1] += ((core + halo) * fl * gain)[..., None] * np.array([1.0, 0.42, 0.10], np.float32)
    return lin
