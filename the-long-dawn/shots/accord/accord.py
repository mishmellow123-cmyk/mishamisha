"""THE LONG DAWN — ACCORD shot renderer (src frames 1912–2247; v2 timeline = src + 160).

Three variants (BIBLE_V2.md):  A = Allegory (the four oaths carved and lit),
                               B = Legend (wordless: carved ornament, one continuous turn),
                               C = Tolkien (as A, plus the Ring lying in the hearth fire).
Usage:
  python shots/accord/accord.py still 2030 --variant B [--scale 0.5]     # test still -> renders/accord_B/tests/
  python shots/accord/accord.py still 2030 --variant C --window 700,300,520,300   # full-res crop (look-dev)
  python shots/accord/accord.py range 1912 2247 --variant A --worker 0/2  # delivery frames, FULL resolution
  python shots/accord/accord.py finish --variant A                        # preview.mp4 + contact.png

`range` renders at full resolution by default. A reduced --scale never writes into the delivery
folder (it goes to renders/accord_<V>/draft/): in v1 the half-res default, bilinearly upscaled,
was what made local renders of the inscriptions look soft.
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
import ring as RG  # noqa: E402

_RES = {}
NOBAND = np.zeros(3)          # v2: no screen band is calmed for text (the T12 dimming is gone)


def out_dir(variant=None):
    return os.path.join(SC.ROOT, 'renders', f'accord_{(variant or SC.VARIANT)}')


def resources():
    if not _RES:
        flat, offs, sizes = tm.load(tm.kind_for(SC.VARIANT))
        _RES['tex'] = (flat, offs.astype(np.int64), sizes.astype(np.int64))
        _RES['oang'] = SC.oath_angles()
        _RES['stones'] = SC.stones()
        _RES['noise3'] = nbcore.make_noise3(64, 7)
        _RES['ring'] = RG.load()
        try:
            import plain
            _RES['plain'] = plain.load()
        except Exception as e:  # noqa
            print('plain not available:', e)
            z = np.zeros((4, 4), np.float32)
            _RES['plain'] = dict(pgc=z, pgf=z, pgc_x0=-500.0, pgc_cell=250.0, pgf_x0=-40.0, pgf_cell=20.0)
    return _RES


def edge_mask(oid, depth, rgb=None):
    """Pixels that get 4 extra samples: object-id edges, plus - on the emissaries, stones and the
    Ring - pixels whose (log) brightness differs sharply from a neighbour (crease lines, rims)."""
    m = np.zeros(oid.shape, bool)
    d = oid[:, 1:] != oid[:, :-1]
    m[:, 1:] |= d
    m[:, :-1] |= d
    d = oid[1:, :] != oid[:-1, :]
    m[1:, :] |= d
    m[:-1, :] |= d
    obj = (oid >= 10) | (oid == 3)
    if rgb is None:
        return m | obj
    L = np.log(rgb.max(axis=2) + 1e-3)
    c = np.zeros(oid.shape, bool)
    dx = np.abs(L[:, 1:] - L[:, :-1]) > 0.22
    c[:, 1:] |= dx
    c[:, :-1] |= dx
    dy = np.abs(L[1:, :] - L[:-1, :]) > 0.22
    c[1:, :] |= dy
    c[:-1, :] |= dy
    m |= c & obj
    m |= (oid == 3)
    return m


def camera(t, scale, window=None):
    cam = SC.camera(t, scale)
    if window is not None:
        cam = cam.copy()
        cam[13] -= window[0] * scale
        cam[14] -= window[1] * scale
    return cam


def frame_size(scale, window=None):
    if window is not None:
        return int(round(window[2] * scale)), int(round(window[3] * scale))
    return int(round(SC.W * scale)), int(round(SC.H * scale))


def render_surfaces(t, scale, aa=True, window=None):
    R = resources()
    Wd, Hd = frame_size(scale, window)
    cam = camera(t, scale, window)
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
    RP = RG.ring_params(t)
    rtex, rsz = R['ring']
    rgb = np.zeros((Hd, Wd, 3), np.float32)
    depth = np.zeros((Hd, Wd), np.float32)
    oid = np.zeros((Hd, Wd), np.int32)
    dummy = np.zeros((1, 1), np.bool_)
    SH.render_surfaces(Wd, Hd, cam, PR, TL, Fa, Fa.shape[0], S, S.shape[0], flat, offs, sizes,
                       pl['pgc'], pl['pgf'], igc, igf, R['oang'], RP, rtex, rsz, rgb, depth, oid, False, dummy, 1)
    if aa:
        m = edge_mask(oid, depth, rgb)
        SH.render_surfaces(Wd, Hd, cam, PR, TL, Fa, Fa.shape[0], S, S.shape[0], flat, offs, sizes,
                           pl['pgc'], pl['pgf'], igc, igf, R['oang'], RP, rtex, rsz, rgb, depth, oid, True, m, 4)
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
    # variant C: the flames rise AROUND the Ring, so it lies visible in a bed of coals
    # (the white flare swallows it at the end)
    FP[FI.FP_HOLLOW] = RG.fire_hollow(t) if SC.VARIANT == 'C' else 0.0
    return FP


def heat_haze(img, t, cam, scale, window=None):
    """Screen-space shimmer of whatever is seen through the hot air above the hearth."""
    if SC.fire_scale(t) <= 0.0:
        return img
    Hd, Wd = img.shape[:2]
    (cx, cy), zc = SC.project(cam, np.array([0.0, 0.0, 1.3]))
    rad = cam[12] * 1.25 * SC.fire_scale(t) / zc
    ph = t * 0.35
    yy, xx = np.mgrid[0:Hd, 0:Wd].astype(np.float32)
    rr = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / max(rad, 1.0)
    fall = np.clip(1.0 - rr, 0, 1) ** 1.5
    if fall.max() <= 0:
        return img
    # smooth animated noise from sines (cheap, coherent); in full-frame pixel coordinates
    ox = window[0] * scale if window is not None else 0.0
    oy = window[1] * scale if window is not None else 0.0
    X = xx + ox
    Y = yy + oy
    amp = 2.2 * scale
    dx = amp * fall * (np.sin(X * 0.045 / scale + ph * 3.1) * np.cos(Y * 0.037 / scale - ph * 2.3)
                       + 0.5 * np.sin((X + Y) * 0.09 / scale + ph * 4.7))
    dy = amp * fall * (np.cos(X * 0.041 / scale - ph * 2.7) * np.sin(Y * 0.052 / scale + ph * 3.3)
                       + 0.5 * np.cos((X - Y) * 0.08 / scale - ph * 5.1))
    return cv2.remap(img, (xx + dx).astype(np.float32), (yy + dy).astype(np.float32), cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REFLECT)


def dof_amount(t):
    """Depth of field while the camera is low over the table (2000-2140): the carved table is in
    focus, the emissaries close to the lens at the frame edges go soft."""
    return SC.smooth(SC.ramp(t, 2000, 2013)) * (1.0 - SC.smooth(SC.ramp(t, 2138, 2156)))


def near_dof(rgb, depth, cam, t, scale, window=None):
    """Layered near-field defocus: circles of confusion from the depth buffer (focus on the table top),
    heavier toward the frame edges; each layer is blurred with its own alpha and laid over a
    background estimate, so the blur of a near figure spills over the sharp table behind it."""
    amt = dof_amount(t)
    if amt <= 0.0:
        return rgb
    Hd, Wd = depth.shape
    ox = window[0] * scale if window is not None else 0.0
    oy = window[1] * scale if window is not None else 0.0
    yy, xx = np.mgrid[0:Hd, 0:Wd].astype(np.float32)
    f, cx, cy = cam[12], cam[13], cam[14]
    sx = (xx + 0.5 - cx) / f
    sy = -(yy + 0.5 - cy) / f
    dz = cam[11] + sx * cam[5] + sy * cam[8]
    ln = np.sqrt((cam[9] + sx * cam[3] + sy * cam[6]) ** 2 + (cam[10] + sx * cam[4] + sy * cam[7]) ** 2 + dz ** 2)
    dz = dz / ln
    zf = (cam[2] - 0.62) / np.maximum(-dz, 1e-3)                  # distance to the table-top plane
    coc = np.clip(zf / np.maximum(depth, 1e-3) - 1.0, 0.0, None) * (46.0 * scale * amt)
    ex = (xx + ox) / (SC.W * scale) * 2.0 - 1.0
    ey = (yy + oy) / (SC.H * scale) * 2.0 - 1.0
    edge = np.clip(np.sqrt(ex * ex * 0.8 + ey * ey * 1.2) / 1.2, 0.0, 1.0)
    coc *= 0.55 + 0.95 * edge ** 2
    coc = np.minimum(coc, 14.0 * scale)
    sig = np.array([1.6, 3.6, 7.5, 13.0]) * scale                # layer blur radii (px)
    # weights of each pixel's coc across the layers (0 below the first)
    lv = np.interp(coc, [0.0, *sig], [0.0, 1.0, 2.0, 3.0, 4.0])
    out_c = np.zeros_like(rgb)
    out_a = np.zeros(depth.shape, np.float32)
    for k in range(4):
        w = np.clip(1.0 - np.abs(lv - (k + 1)), 0.0, 1.0).astype(np.float32)
        if k == 3:
            w = np.clip(lv - 3.0, 0.0, 1.0).astype(np.float32)
        if w.max() <= 0.0:
            continue
        out_c += cv2.GaussianBlur(rgb * w[..., None], (0, 0), float(sig[k]))
        out_a += cv2.GaussianBlur(w, (0, 0), float(sig[k]))
    a_ = np.clip(out_a, 0.0, 1.0)
    # background behind the defocused layer: the sharp image where nothing is defocused, a normalised
    # blur of the sharp surroundings where the near layer sits
    near = np.clip(lv, 0.0, 1.0).astype(np.float32)
    keep = 1.0 - near
    bgn = cv2.GaussianBlur(rgb * keep[..., None], (0, 0), float(sig[2]))
    bgd = cv2.GaussianBlur(keep, (0, 0), float(sig[2]))[..., None]
    bg = rgb * keep[..., None] + (bgn / np.maximum(bgd, 1e-3)) * near[..., None]
    return (out_c + bg * (1.0 - a_[..., None])).astype(np.float32)


def render_frame(t, scale=1.0, aa=True, mb=True, window=None):
    R = resources()
    rgb, depth, oid, cam, PR, Fa = render_surfaces(t, scale, aa, window)
    Hd, Wd = depth.shape
    rgb = heat_haze(rgb, t, cam, scale, window)
    FP = fire_params(t)
    if FP[FI.FP_SCALE] > 0.0:
        # the fire in its own buffer, softened by a sub-pixel blur that dissolves the ray-march jitter
        fb = np.zeros_like(rgb)
        FI.fire_volume(Wd, Hd, cam, FP, R['noise3'], depth, fb, 28)
        rgb += cv2.GaussianBlur(fb, (0, 0), 0.85 * max(scale, 0.5))
    blobs = PT.torch_flames(t) + PT.ribbons(t) + PT.ignition_flash(t)
    if blobs:
        B = np.array(blobs, np.float64)
        FI.splat_blobs(rgb, depth, cam, B, B.shape[0], 0.3, NOBAND)
    walkers_draw(rgb, depth, cam, t, scale, NOBAND)
    FT = PT.frieze_tongues(t)
    if FT.shape[0]:
        FI.splat_streaks(rgb, depth, cam, FT, FT.shape[0], 0.0, 10.0, NOBAND, 1.0)
    rgb = near_dof(rgb, depth, cam, t, scale, window)
    if mb:
        camA = camera(t - 0.2, scale, window)
        camB = camera(t + 0.2, scale, window)
        out = np.empty_like(rgb)
        FI.motion_blur(rgb, out, depth, cam, camA, camB, 20, 70.0 * scale)
        rgb = out
    E = PT.embers(t, cam[2])
    if E.shape[0]:
        zf = cam[2] - 0.62
        FI.splat_streaks(rgb, depth, cam, E, E.shape[0], 0.028, zf, NOBAND, 45.0 * scale)
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


def still(t, scale, tag='', window=None):
    t0 = time.time()
    hdr, info = render_frame(t, scale, window=window)
    t1 = time.time()
    img = finish(hdr, t)
    tests = os.path.join(out_dir(), 'tests')
    os.makedirs(tests, exist_ok=True)
    w = '' if window is None else '_w' + '-'.join(str(int(v)) for v in window)
    p = os.path.join(tests, f'still_{int(t)}{tag}{w}_s{scale}.png')
    look.save_png(p, img)
    print(f'{p}  render {t1 - t0:.1f}s', flush=True)
    return p


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['still', 'range', 'finish'])
    ap.add_argument('a', nargs='?', type=float)
    ap.add_argument('b', nargs='?', type=float)
    ap.add_argument('--variant', default=os.environ.get('ACCORD_VARIANT', 'A'))
    ap.add_argument('--scale', type=float, default=None, help='still: 0.5 by default; range: 1.0 (delivery)')
    ap.add_argument('--window', default=None, help='x0,y0,w,h in full-res pixels (still only)')
    ap.add_argument('--tag', default='')
    ap.add_argument('--worker', default='0/1')
    ap.add_argument('--force', action='store_true')
    args = ap.parse_args()
    SC.set_variant(args.variant)
    OUT = out_dir()
    if args.cmd == 'still':
        win = tuple(float(v) for v in args.window.split(',')) if args.window else None
        still(args.a, args.scale if args.scale is not None else (1.0 if win else 0.5), args.tag, win)
    elif args.cmd == 'range':
        cv2.setNumThreads(1)
        k, n = [int(v) for v in args.worker.split('/')]
        a, b = int(args.a), int(args.b)
        scale = args.scale if args.scale is not None else 1.0
        dest = OUT if scale == 1.0 else os.path.join(OUT, 'draft')
        os.makedirs(dest, exist_ok=True)
        for fr in range(a + k, b + 1, n):
            p = look.frame_path(dest, fr)
            if os.path.exists(p) and not args.force:
                continue
            t0 = time.time()
            hdr, info = render_frame(float(fr), scale)
            img = finish(hdr, float(fr))
            if scale != 1.0:
                img = cv2.resize(img, (SC.W, SC.H), interpolation=cv2.INTER_LINEAR)
            tmp = p[:-4] + '.tmp.png'
            look.save_png(tmp, img)
            os.replace(tmp, p)
            print(f'frame {fr} {time.time() - t0:.1f}s', flush=True)
    elif args.cmd == 'finish':
        look.preview_mp4(OUT, os.path.join(OUT, 'preview.mp4'), SC.F0, SC.F1 + 1)
        frames = [1912, 1930, 1950, 1970, 1985, 1994, 1998, 2001, 2006, 2014, 2030, 2044,
                  2060, 2084, 2110, 2124, 2150, 2170, 2190, 2210, 2226, 2234, 2240, 2247]
        look.contact_sheet(OUT, frames, os.path.join(OUT, 'contact.png'), cols=4, thumb_w=480)
