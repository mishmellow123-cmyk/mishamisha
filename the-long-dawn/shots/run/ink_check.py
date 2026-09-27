"""The boiling check for C's ink pass (R12): are the strokes anchored to the terrain?

For consecutive frames N, N+1 every inked world point of N+1 (from its AOV cache) is projected into frame N's camera;
where it was visible in N, its ink coverage then and now are compared. Anchored strokes give a small residual (only
LOD fades, resampling, occlusion edges); strokes dealt afresh each frame ("boiling") give a residual about as large
as comparing the two frames pixel for pixel. Reports both, per frame pair, and writes a residual heat map.

  python ink_check.py --shot scroll --range 140-163 --scale 0.5
"""
import argparse
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import render_ink as RI     # noqa: E402

OUT = RI.OUT
SURF = 'land'          # land | cloud | all (the cloud sea itself evolves, so it is judged apart)


def load_ink(shot, frame, tag=''):
    p = os.path.join(OUT, shot + tag + '_ink', f'f_{frame:05d}.png')
    return cv2.imread(p, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0


def project(cam, X):
    d = X - cam['pos'][None, None]
    x = d @ cam['right']
    y = d @ cam['up']
    z = d @ cam['fwd']
    zz = np.maximum(z, 1e-6)
    return cam['cx'] + cam['f'] * x / zz - 0.5, cam['cy'] - cam['f'] * y / zz - 0.5, z


def pair(shot, f0, f1, scale, tag='', ctrl=None):
    R0 = RI.load_cache(RI.cache_path(shot, f0, scale))
    R1 = RI.load_cache(RI.cache_path(shot, f1, scale))
    i0, i1 = load_ink(shot, f0, tag), load_ink(shot, f1, tag)
    A1 = R1['A']
    X = np.stack([A1[..., 2], A1[..., 3], A1[..., 4]], -1).astype(np.float64)
    ok = (A1[..., 0] < 1.5) if SURF == 'land' else ((A1[..., 0] > 1.5) & (A1[..., 0] < 2.5)) if SURF == 'cloud' \
        else (A1[..., 0] < 2.5)
    mx, my, z = project(R0['cam'], X)
    # visible in N: its distance from N's eye matches N's depth there (1 %)
    d_should = np.linalg.norm(X - R0['cam']['pos'][None, None], axis=-1)
    d_n = cv2.remap(R0['A'][..., 1], mx.astype(np.float32), my.astype(np.float32), cv2.INTER_NEAREST,
                    borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)
    vis = ok & (z > 0) & (np.abs(d_n - d_should) < 0.01 * d_should) & (mx > 1) & (my > 1) & \
        (mx < i0.shape[1] - 2) & (my < i0.shape[0] - 2)
    back = cv2.remap(i0, mx.astype(np.float32), my.astype(np.float32), cv2.INTER_LINEAR)
    inked = vis & ((i1 > 0.05) | (back > 0.05))
    # compare at the stroke scale: a sub-pixel resampling of a 1-2 px line is not a moved stroke
    bl = lambda im: cv2.GaussianBlur(im, (0, 0), 1.2 * i1.shape[1] / 1920.0)
    i1b, backb, i0b = bl(i1), bl(back), bl(i0)
    ra = np.abs(i1b - backb)
    rs = np.abs(i1b - i0b)
    out = [float(ra[inked].mean()), float(rs[inked].mean()), float(inked.mean()), ra * inked]
    if ctrl is not None:
        # the boiling control: the same frame N with every stroke re-dealt, reprojected the same way
        backc = bl(cv2.remap(ctrl, mx.astype(np.float32), my.astype(np.float32), cv2.INTER_LINEAR))
        out.append(float(np.abs(i1b - backc)[inked].mean()))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--shot', default='scroll')
    ap.add_argument('--range', default='140-163')
    ap.add_argument('--scale', type=float, default=0.5)
    ap.add_argument('--tag', default='')
    ap.add_argument('--surf', default='land')
    a = ap.parse_args()
    global SURF
    SURF = a.surf
    s, e = [int(v) for v in a.range.split('-')]
    rows = []
    heat = None
    for f in range(s, e):
        cp = os.path.join(OUT, a.shot + '_reseed_ink', f'f_{f:05d}.png')
        ctrl = cv2.imread(cp, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0 if os.path.exists(cp) else None
        res = pair(a.shot, f, f + 1, a.scale, a.tag, ctrl)
        ra, rs, frac, hm = res[:4]
        rc = res[4] if len(res) > 4 else float('nan')
        rows.append((f, ra, rs, rc))
        print(f'{f}->{f + 1}: anchored {ra:.4f} | re-dealt (boiling) {rc:.4f} | screen-fixed {rs:.4f} | inked {frac:.2f}',
              flush=True)
        heat = hm if heat is None else np.maximum(heat, hm)
    r = np.array([[x[1], x[2], x[3]] for x in rows])
    rc = np.nanmean(r[:, 2]) if np.isfinite(r[:, 2]).any() else float('nan')
    print(f'MEAN anchored {r[:, 0].mean():.4f} | re-dealt {rc:.4f} | screen-fixed {r[:, 1].mean():.4f} | '
          f'anchored/re-dealt {r[:, 0].mean() / rc:.3f}')
    hm = (np.clip(heat / 0.5, 0, 1) * 255).astype(np.uint8)
    cv2.imwrite(os.path.join(OUT, f'{a.shot}{a.tag}_boil_{a.surf}_{s}-{e}.png'), cv2.applyColorMap(hm, cv2.COLORMAP_INFERNO))


if __name__ == '__main__':
    main()
