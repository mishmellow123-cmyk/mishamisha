"""Look-dev sheet for C's drawn fires: inkpass.flame_glyph at the sizes the shots use (a far beacon a few px tall up to
her fire filling the reveal's first frames), drawn at 2x on the page colour and downsampled like a production frame.
Frame k = time k * 0.5 s (the sway). -> renders/runC_flames/f_%05d.png (1920x804). Run from shots/run/:

    python3 flame_sheet.py 0,1,2
"""
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import inkpass as IP        # noqa: E402

OUT = os.path.normpath(os.path.join(HERE, '..', '..', 'renders', 'runC_flames'))
SIZES = (22, 40, 70, 120, 220, 420, 760)            # flame height, page px at 2x (her fire at reveal f0 is ~500-700)


def sheet(k):
    kpx = 2.0
    H, W = 1608, 3840
    pp = np.zeros((H, W, 2), np.float32)
    IP.paper(H, W, kpx, pp)
    tt = np.clip(pp[..., 0], 0, 1)[..., None]
    rgb = IP.s2l(IP.PAPER)[None, None] * (1 - 0.55 * tt) + IP.s2l(IP.PAPER_DARK)[None, None] * (0.55 * tt)
    rgb = rgb * (1.0 + 0.06 * pp[..., 1])[..., None]
    t = 0.5 * k
    ROWS = ((520.0, ((22, 3), (40, 3), (70, 3), (120, 3))), (1585.0, ((220, 2), (420, 1), (760, 1))))
    j = 0
    for sy, row in ROWS:
      x = 70.0
      for hp, reps in row:
        j += 1
        for rep in range(reps):
            sx = x + 0.9 * hp
            x0, x1 = int(max(sx - 0.9 * hp - 4, 0)), int(min(sx + 1.2 * hp + 5, W))
            y0, y1 = int(max(sy - 2.3 * hp - 4, 0)), int(min(sy + 0.15 * hp + 5, H))
            yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float64)
            near = min(max((hp / kpx - 20.0) / 120.0, 0.0), 1.0)
            pw = kpx * (0.75 + 0.55 * near)
            n = 3 if hp < 22.0 * kpx else (5 if hp < 70.0 * kpx else 7)
            seed = (11 + 5 * j + 3 * rep) * 7 + 3
            grain = np.clip(pp[y0:y1, x0:x1, 1] * 0.5 + 0.5, 0, 1)
            a, col, ln = IP.flame_glyph(xx + 0.5 - sx, sy - (yy + 0.5), hp, t, seed, n, kpx, pw, grain)
            sub = rgb[y0:y1, x0:x1]
            sub = sub * (1 - a[..., None]) + col * a[..., None]
            sub = sub * (1 - 0.88 * ln[..., None]) + IP.s2l(IP.INK)[None, None] * (0.88 * ln[..., None])
            rgb[y0:y1, x0:x1] = sub
            x = sx + 1.3 * hp + 40.0
    img = IP.look.linear_to_srgb(np.clip(rgb, 0, 1)).astype(np.float32)
    img = cv2.resize(img, (1920, 804), interpolation=cv2.INTER_AREA)
    os.makedirs(OUT, exist_ok=True)
    IP.look.save_png(IP.look.frame_path(OUT, k), img)
    print('flame sheet', k, flush=True)


if __name__ == '__main__':
    for k in [int(v) for v in (sys.argv[1] if len(sys.argv) > 1 else '0,1,2').split(',')]:
        sheet(k)
