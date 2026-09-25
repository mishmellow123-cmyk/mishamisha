"""Fire look-dev: render the hearth region only (surface + fire volume), crop, for several frames."""
import os, sys, time
os.environ.setdefault('NUMBA_NUM_THREADS', '2')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, cv2
import accord as A, scene as SC, fire as FI
from scene import look

def fire_only(t, scale=1.0, crop=(560, 0, 800, 804), I=None, extra=None):
    R = A.resources()
    rgb, depth, oid, cam, PR, Fa = A.render_surfaces(t, scale, aa=False)
    FP = A.fire_params(t)
    if I is not None: FP[FI.FP_I] = I
    if extra:
        for k, v in extra.items(): FP[k] = v
    t0 = time.time()
    FI.fire_volume(rgb.shape[1], rgb.shape[0], cam, FP, R['noise3'], depth, rgb, 44)
    dt = time.time() - t0
    img = look.finish(rgb, exposure=SC.exposure(t), bloom_strength=0.07, bloom_threshold=0.9, vignette_amount=0.0)
    x0, y0, w, h = [int(v * scale) for v in crop]
    return img[y0:y0 + h, x0:x0 + w], dt

if __name__ == '__main__':
    frames = [float(a) for a in sys.argv[1].split(',')]
    I = float(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2] != '-' else None
    tiles = []
    for t in frames:
        im, dt = fire_only(t, 1.0, I=I)
        print(t, 'fire', round(dt, 2), 's', 'max', im.max())
        tiles.append(im)
    out = np.hstack(tiles)
    look.save_png('renders/accord/tests/fire_test.png', out)
