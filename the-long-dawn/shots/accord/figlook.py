"""Look-dev: frame emissaries from above with the real lighting of frame t.
usage: python figlook.py <variant> <t> <out.png> [fig indices] [height]"""
import os, sys, math
os.environ.setdefault('NUMBA_NUM_THREADS', '2')
sys.path.insert(0, os.path.expanduser('~/mishamisha/the-long-dawn/shots/accord'))
import numpy as np, cv2
import accord as A, scene as SC
from scene import look

variant, t, out = sys.argv[1], float(sys.argv[2]), sys.argv[3]
figs = [int(v) for v in sys.argv[4].split(',')] if len(sys.argv) > 4 else list(range(12))
hcam = float(sys.argv[5]) if len(sys.argv) > 5 else 9.0
SC.set_variant(variant)
orig = SC.camera

def cam_over(i):
    def camera(tt, scale=1.0):
        c = orig(tt, scale).copy()
        # camera straight above the hearth at hcam, rotated so figure i is at the top, then
        # lens-shift so the figure sits at the tile centre
        ang = SC.FIG_ANG[i]
        U = np.array([math.cos(ang), math.sin(ang), 0.0])
        Fw = np.array([0.0, 0.0, -1.0])
        R = np.cross(Fw, U)
        c[0:3] = [0, 0, hcam]
        c[3:6] = R
        c[6:9] = U
        f = SC.F_FULL * 1.15 * scale
        c[12] = f
        # project figure (head ~1.5 m) -> shift so it lands mid-tile
        zc = hcam - 1.2
        ys = f * SC.FIG_R[i] / zc
        c[13] = 200 * scale
        c[14] = 200 * scale + ys
        return c
    return camera

tiles = []
for i in figs:
    SC.camera = cam_over(i)
    A.SC.camera = SC.camera
    hdr, info = A.render_frame(t, 1.0, mb=False, window=(0, 0, 400, 400))
    img = A.finish(hdr, t)
    im8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)[..., ::-1].copy()
    cv2.putText(im8, f'{i} type {SC.FIG_TYPE[i]}', (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
    tiles.append(im8)
cols = min(6, len(tiles))
while len(tiles) % cols:
    tiles.append(np.zeros_like(tiles[0]))
rows = [np.hstack(tiles[k:k + cols]) for k in range(0, len(tiles), cols)]
cv2.imwrite(out, np.vstack(rows))
print('wrote', out)
