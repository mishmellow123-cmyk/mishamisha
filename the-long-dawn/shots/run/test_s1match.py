"""Match check: render the run's world from s1's camera (terrain + sky + stars only) next to s1's frame."""
import math
import os
import sys
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pipe as PI          # noqa: E402
import world as WD         # noqa: E402
import rcam as RC          # noqa: E402
from mt import sky as SK   # noqa: E402

S1 = WD.S1
look = PI.look


def main(frame=1500, scale=0.5, ss=1.5):
    W, H = int(1920 * scale), int(804 * scale)
    c = S1.camera(frame, W, H)
    tcam = RC.RCam(c.pos, math.degrees(c.yaw), math.degrees(c.tilt), 0.0, 36.0, W, H)
    fr = PI.Frame(tcam, ss)
    CR = np.zeros((0, WD.NCR))
    t0 = time.time()
    PI.render_terrain(fr, frame, CR, WD.night_light(), np.zeros((0, 8)))
    t1 = time.time()
    scam = fr.src
    stars = SK.make_stars(14000, 101, lum_scale=7.0)
    sc = PI.src_scale(fr)
    mask = (fr.dist > 1e8).astype(np.float32)
    SK.splat_stars(fr.img, scam, stars, mask, t=PI.ftime(frame), gain=ss * ss, scale=sc)
    img, zb, di = PI.to_target(fr)
    print('terrain %.1fs total %.1fs src %dx%d' % (t1 - t0, time.time() - t0, scam.W, scam.H))
    out = look.finish(img, **S1.FINISH)
    os.makedirs(os.path.join(PI.CM.ROOT, 'renders', 'run_v2', 'tests'), exist_ok=True)
    p = os.path.join(PI.CM.ROOT, 'renders', 'run_v2', 'tests', f's1match_{frame}.png')
    look.save_png(p, out)
    ref = cv2.imread(look.frame_path(os.path.join(PI.CM.ROOT, 'renders', 'montage'), frame))
    ref = cv2.resize(ref, (W, H), interpolation=cv2.INTER_AREA)
    mine = cv2.imread(p)
    cv2.imwrite(os.path.join(PI.CM.ROOT, 'renders', 'run_v2', 'tests', f's1match_{frame}_cmp.png'),
                np.vstack([ref, mine]))
    print(p)


if __name__ == '__main__':
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 1500)
