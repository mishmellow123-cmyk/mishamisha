"""EXR -> delivered PNG through the project finish. python post_run.py <shot> [--watch]
Reads shots/mountain/cache/<shot>_exr/f_%05d.exr, writes renders/<out>/f_%05d.png."""
import glob
import os
import sys
import time

os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR', '1')
import math

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'lib'))
import look

cv2.setNumThreads(2)


def sun_glare(hdr, cap=40.0):
    """Clamp the sun disc and add a synthetic glare scaled by its visible energy: a soft halo,
    an 8-ray diffraction starburst (long/short alternating, like the globe's sunrise) and a
    faint anamorphic streak. Returns the new HDR."""
    lum = hdr.max(axis=2)
    m = lum > cap
    if not m.any():
        return hdr
    ex = np.clip(lum - cap, 0, None)
    E = float(ex.sum())
    ys, xs = np.nonzero(m)
    w = ex[m]
    cy = float((ys * w).sum() / w.sum())
    cx = float((xs * w).sum() / w.sum())
    out = hdr.copy()
    sc = np.minimum(1.0, cap / np.maximum(lum, 1e-6))[..., None]
    out = out * sc
    h, wd = lum.shape
    Y, X = np.mgrid[0:h, 0:wd].astype(np.float32)
    dx = X - cx
    dy = Y - cy
    r = np.sqrt(dx * dx + dy * dy) + 1e-3
    s = math.sqrt(E) / 60.0                       # glare size ~ sqrt(visible energy)
    g = np.zeros((h, wd), np.float32)
    halo = (np.exp(-(r / (22 * s + 2)) ** 2) * 3.0 + np.exp(-r / (140 * s + 5)) * 0.35)
    g += halo
    phi = np.arctan2(dy, dx)
    for k in range(8):
        th = math.radians(22.5 + 45.0 * k)
        L = (520 if k % 2 == 0 else 260) * s + 10
        d = phi - th
        d = (d + math.pi) % (2 * math.pi) - math.pi
        perp = r * np.sin(np.abs(d))
        along = np.cos(d) > 0
        g += along * np.exp(-(perp / 1.3) ** 2) * np.exp(-r / L) * 1.2
    streak = np.exp(-(dy / 1.6) ** 2) * np.exp(-np.abs(dx) / (700 * s + 20)) * 0.25
    g += streak
    tint = np.array([1.0, 0.86, 0.66], np.float32)
    return out + (g * min(E / 4000.0, 6.0))[..., None] * tint


FINS = {'run': dict(exposure=1.0, bloom_strength=0.08, bloom_threshold=0.8, vignette_amount=0.2),
        'dawn': dict(exposure=1.0, bloom_strength=0.08, bloom_threshold=0.8, vignette_amount=0.2)}


def process(hdr, shot):
    if shot == 'dawn':
        hdr = sun_glare(hdr)
    return look.finish(hdr, **FINS[shot])


if __name__ == '__main__':
    shot = sys.argv[1]
    OUT = {'run': 'run', 'dawn': 'dawn_C'}[shot]
    FIN = {'run': dict(exposure=1.0, bloom_strength=0.08, bloom_threshold=0.8, vignette_amount=0.2),
           'dawn': dict(exposure=1.0, bloom_strength=0.08, bloom_threshold=0.8, vignette_amount=0.2)}[shot]
    src = os.path.join(HERE, 'cache', '%s_exr' % shot)
    dst = os.path.join(ROOT, 'renders', OUT)
    os.makedirs(dst, exist_ok=True)
    watch = '--watch' in sys.argv
    done = set()
    while True:
        todo = [p for p in sorted(glob.glob(os.path.join(src, 'f_*.exr'))) if p not in done]
        for p in todo:
            # skip files still being written (size stable for 2 s)
            s0 = os.path.getsize(p)
            time.sleep(0.5 if not watch else 2.0)
            if os.path.getsize(p) != s0 or s0 == 0:
                continue
            im = cv2.imread(p, cv2.IMREAD_UNCHANGED)
            if im is None:
                continue
            hdr = im[..., :3][..., ::-1].astype(np.float32)
            if shot == 'dawn':
                hdr = sun_glare(hdr)
            out = look.finish(hdr, **FIN)
            f = int(os.path.basename(p)[2:7])
            look.save_png(look.frame_path(dst, f), out)
            done.add(p)
            print('post', f, flush=True)
        if not watch:
            break
        if os.path.exists(os.path.join(src, 'DONE')) and not [p for p in glob.glob(os.path.join(src, 'f_*.exr')) if p not in done]:
            break
        time.sleep(5)
