"""Quick terrain-only composition stills:  python quick.py 1520,1541,1580 [scale] [ss] [tag]"""
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pipe as PI          # noqa: E402
import world as WD         # noqa: E402
import run as RN           # noqa: E402
from mt import sky as SK   # noqa: E402

look = PI.look


def main():
    frames = [int(f) for f in sys.argv[1].split(',')]
    scale = float(sys.argv[2]) if len(sys.argv) > 2 else 0.3
    ss = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
    tag = sys.argv[4] if len(sys.argv) > 4 else 'q'
    W, H = int(round(1920 * scale)), int(round(804 * scale))
    stars = SK.make_stars(14000, 101, lum_scale=7.0)
    outdir = os.path.join(PI.CM.ROOT, 'renders', 'run_v2', 'tests', tag)
    os.makedirs(outdir, exist_ok=True)
    for f in frames:
        t0 = time.time()
        tcam = RN.camera(f, W, H)
        fr = PI.Frame(tcam, ss)
        PI.render_terrain(fr, f, RN.CR, WD.night_light(), np.zeros((0, 8)))
        mask = (fr.dist > 1e8).astype(np.float32)
        SK.splat_stars(fr.img, fr.src, stars, mask, t=PI.ftime(f), gain=ss * ss, scale=PI.src_scale(fr))
        img, zb, di = PI.to_target(fr)
        out = look.finish(img, exposure=1.0, bloom_strength=0.07, bloom_threshold=0.8, vignette_amount=0.25)
        look.save_png(look.frame_path(outdir, f), out)
        print(f, '%.1fs' % (time.time() - t0), 'src', fr.src.W, fr.src.H, flush=True)


if __name__ == '__main__':
    main()
