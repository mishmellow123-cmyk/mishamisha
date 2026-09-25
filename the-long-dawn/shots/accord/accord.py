"""THE LONG DAWN — ACCORD shot renderer (global frames 1912–2247).

Usage:
  python3 shots/accord/accord.py still 2030 --scale 0.5          # test still -> renders/accord/tests/
  python3 shots/accord/accord.py range 1912 2247 [--scale 1] [--step 1] [--worker k/n]
  python3 shots/accord/accord.py finish                           # preview.mp4 + contact.png
"""
import argparse
import math
import os
import sys
import time

os.environ.setdefault('NUMBA_NUM_THREADS', '2')
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def _invalidate_numba_cache():
    """numba's on-disk cache does not track cross-module dependencies: wipe it whenever
    any source file in this folder changes."""
    import glob
    import hashlib
    hsh = hashlib.sha1()
    for p in sorted(glob.glob(os.path.join(HERE, '*.py'))):
        with open(p, 'rb') as fh:
            hsh.update(fh.read())
    stamp = os.path.join(HERE, '__pycache__', 'src.sha1')
    os.makedirs(os.path.dirname(stamp), exist_ok=True)
    old = open(stamp).read() if os.path.exists(stamp) else ''
    if old != hsh.hexdigest():
        for f in glob.glob(os.path.join(HERE, '__pycache__', '*.nb[ic]')):
            try:
                os.remove(f)
            except OSError:
                pass
        with open(stamp, 'w') as fh:
            fh.write(hsh.hexdigest())


_invalidate_numba_cache()

import numpy as np  # noqa: E402

import scene as SC  # noqa: E402
from scene import look  # noqa: E402
import textmaps as tm  # noqa: E402
import shade as SH  # noqa: E402
import nbcore  # noqa: E402
import cv2  # noqa: E402
import fire as FI  # noqa: E402
import particles as PT  # noqa: E402

OUT = os.path.join(SC.ROOT, 'renders', 'accord')
TESTS = os.path.join(OUT, 'tests')

_RES = {}


def resources():
    if not _RES:
        flat, offs, sizes, tl = tm.load()
        _RES['tex'] = (flat, offs.astype(np.int64), sizes.astype(np.int64))
        _RES['oang'] = SC.oath_angles()
        _RES['stones'] = SC.stones()
        _RES['noise3'] = nbcore.make_noise3(64, 7)
        try:
            import plain
            _RES['plain'] = plain.load()
        except Exception as e:  # noqa
            print('plain not available:', e)
            z = np.zeros((4, 4), np.float32)
            _RES['plain'] = dict(pgc=z, pgf=z, pgc_x0=-500.0, pgc_cell=250.0, pgf_x0=-40.0, pgf_cell=20.0)
    return _RES


def edge_mask(oid, depth):
    m = np.zeros(oid.shape, bool)
    d = oid[:, 1:] != oid[:, :-1]
    m[:, 1:] |= d
    m[:, :-1] |= d
    d = oid[1:, :] != oid[:-1, :]
    m[1:, :] |= d
    m[:-1, :] |= d
    # figures/stones: always AA (fine silhouettes & folds)
    m |= (oid >= 10)
    return m


def render_surfaces(t, scale, aa=True):
    R = resources()
    Wd, Hd = int(round(SC.W * scale)), int(round(SC.H * scale))
    cam = SC.camera(t, scale)
    PR = SC.params(t, scale)
    Fa = SC.figures(t)
    S = R['stones']
    TL = SC.torch_lights(t, Fa)
    PR[SH.P_NT] = TL.shape[0] if TL[0, 3] + TL[0, 4] + TL[0, 5] > 0 else 0
    pl = R['plain']
    PR[SH.P_PGC_X0], PR[SH.P_PGC_CELL] = pl['pgc_x0'], pl['pgc_cell']
    PR[SH.P_PGF_X0], PR[SH.P_PGF_CELL] = pl['pgf_x0'], pl['pgf_cell']
    igc, igf, ig_par = walker_irradiance(t)
    PR[SH.P_IGC_X0], PR[SH.P_IGC_CELL], PR[SH.P_IGF_X0], PR[SH.P_IGF_CELL] = ig_par
    flat, offs, sizes = R['tex']
    rgb = np.zeros((Hd, Wd, 3), np.float32)
    depth = np.zeros((Hd, Wd), np.float32)
    oid = np.zeros((Hd, Wd), np.int32)
    dummy = np.zeros((1, 1), np.bool_)
    SH.render_surfaces(Wd, Hd, cam, PR, TL, Fa, Fa.shape[0], S, S.shape[0], flat, offs, sizes,
                       pl['pgc'], pl['pgf'], igc, igf, R['oang'], rgb, depth, oid, False, dummy, 1)
    if aa:
        m = edge_mask(oid, depth)
        SH.render_surfaces(Wd, Hd, cam, PR, TL, Fa, Fa.shape[0], S, S.shape[0], flat, offs, sizes,
                           pl['pgc'], pl['pgf'], igc, igf, R['oang'], rgb, depth, oid, True, m, 4)
    return rgb, depth, oid, cam, PR, Fa


def walker_irradiance(t):
    try:
        import plain
        return plain.irradiance(t)
    except Exception:
        z = np.zeros((4, 4), np.float32)
        return z, z, (-500.0, 250.0, -40.0, 20.0)


def fire_params(t):
    FP = np.zeros(FI.FP_N, np.float64)
    FP[FI.FP_T] = t
    hi = SC.hearth_intensity(t)
    FP[FI.FP_I] = 24.0 * hi
    FP[FI.FP_SCALE] = SC.fire_scale(t)
    FP[FI.FP_H] = 2.3
    FP[FI.FP_SWIRL] = 1.8
    FP[FI.FP_Z0] = 0.47
    FP[FI.FP_WHITE] = SC.smooth(SC.ramp(t, SC.FLARE_T0 + 3, SC.FLARE_T1 + 2))
    FP[FI.FP_R0] = 0.40
    FP[FI.FP_SPREAD] = 0.85
    FP[FI.FP_RISE] = 3.2
    return FP


def heat_haze(img, t, cam, scale):
    """Screen-space shimmer of whatever is seen through the hot air above the hearth."""
    if SC.fire_scale(t) <= 0.0:
        return img
    Hd, Wd = img.shape[:2]
    (cx, cy), zc = SC.project(cam, np.array([0.0, 0.0, 1.3]))
    rad = cam[12] * 1.25 * SC.fire_scale(t) / zc
    rng = np.random.default_rng(int(t * 7) % 100000)
    k = 24
    ph = t * 0.35
    yy, xx = np.mgrid[0:Hd, 0:Wd].astype(np.float32)
    rr = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / max(rad, 1.0)
    fall = np.clip(1.0 - rr, 0, 1) ** 1.5
    if fall.max() <= 0:
        return img
    # smooth animated noise from sines (cheap, coherent)
    amp = 2.2 * scale
    dx = amp * fall * (np.sin(xx * 0.045 / scale + ph * 3.1) * np.cos(yy * 0.037 / scale - ph * 2.3)
                       + 0.5 * np.sin((xx + yy) * 0.09 / scale + ph * 4.7))
    dy = amp * fall * (np.cos(xx * 0.041 / scale - ph * 2.7) * np.sin(yy * 0.052 / scale + ph * 3.3)
                       + 0.5 * np.cos((xx - yy) * 0.08 / scale - ph * 5.1))
    return cv2.remap(img, (xx + dx).astype(np.float32), (yy + dy).astype(np.float32), cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REFLECT)


def text_band(t, scale):
    """Screen band (y0,y1,amount) kept calm while edit text T12 is up (1922-1998)."""
    amt = 0.55 * SC.smooth(SC.ramp(t, 1914, 1922)) * (1 - SC.smooth(SC.ramp(t, 1996, 2001)))
    return np.array([560.0 * scale, 700.0 * scale, amt])


def render_frame(t, scale=1.0, aa=True, mb=True):
    R = resources()
    rgb, depth, oid, cam, PR, Fa = render_surfaces(t, scale, aa)
    Hd, Wd = depth.shape
    band = text_band(t, scale)
    # subtle grad on the lower band while T12 is up
    if band[2] > 0:
        yy = np.arange(Hd, dtype=np.float32)[:, None, None]
        g = 1 - 0.18 * band[2] / 0.55 * np.clip((yy - band[0] + 40 * scale) / (40 * scale), 0, 1) * \
            np.clip((band[1] + 40 * scale - yy) / (40 * scale), 0, 1)
        rgb *= g.astype(np.float32)
    rgb = heat_haze(rgb, t, cam, scale)
    FP = fire_params(t)
    FI.fire_volume(Wd, Hd, cam, FP, R['noise3'], depth, rgb, 28)
    blobs = PT.torch_flames(t) + PT.ribbons(t) + PT.ignition_flash(t)
    if blobs:
        B = np.array(blobs, np.float64)
        FI.splat_blobs(rgb, depth, cam, B, B.shape[0], 0.3, band)
    walkers_draw(rgb, depth, cam, t, scale, band)
    if mb:
        camA = SC.camera(t - 0.2, scale)
        camB = SC.camera(t + 0.2, scale)
        out = np.empty_like(rgb)
        FI.motion_blur(rgb, out, depth, cam, camA, camB, 20, 70.0 * scale)
        rgb = out
    E = PT.embers(t, cam[2])
    if E.shape[0]:
        zf = cam[2] - 0.62
        FI.splat_streaks(rgb, depth, cam, E, E.shape[0], 0.028, zf, band, 45.0 * scale)
    return rgb, dict(depth=depth, oid=oid, cam=cam, PR=PR, Fa=Fa)


def walkers_draw(rgb, depth, cam, t, scale, band):
    try:
        import plain
    except Exception:
        return
    plain.draw(rgb, depth, cam, t, scale, band)


def finish(hdr, t):
    return look.finish(hdr, exposure=SC.exposure(t), bloom_strength=0.07, bloom_threshold=0.9,
                       vignette_amount=0.22)


def still(t, scale, tag=''):
    t0 = time.time()
    hdr, info = render_frame(t, scale)
    t1 = time.time()
    img = finish(hdr, t)
    os.makedirs(TESTS, exist_ok=True)
    p = os.path.join(TESTS, f'still_{int(t)}{tag}_s{scale}.png')
    look.save_png(p, img)
    print(f'{p}  render {t1 - t0:.1f}s', flush=True)
    return p


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd')
    ap.add_argument('a', nargs='?', type=float)
    ap.add_argument('b', nargs='?', type=float)
    ap.add_argument('--scale', type=float, default=0.5)
    ap.add_argument('--tag', default='')
    args = ap.parse_args()
    if args.cmd == 'still':
        still(args.a, args.scale, args.tag)
