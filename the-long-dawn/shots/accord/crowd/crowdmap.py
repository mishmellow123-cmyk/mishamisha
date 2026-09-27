"""Top-down debug map of the crowd (no rendering): python crowd/crowdmap.py out.jpg 4480,4600,4700,5580,5620,5660 [--R 60]"""
import argparse, math, os, sys
import numpy as np, cv2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import crowd3 as CR
import scene3 as SC
import geom3 as G

ap = argparse.ArgumentParser()
ap.add_argument('out'); ap.add_argument('frames'); ap.add_argument('--R', type=float, default=60.0); ap.add_argument('--px', type=int, default=700)
a = ap.parse_args()
W = CR.world(); S = W['S']
tiles = []
for f in [float(v) for v in a.frames.split(',')]:
    n = a.px; R = a.R; k = n / (2 * R)
    im = np.full((n, n, 3), 18, np.uint8)
    def P(x, y): return (int(n / 2 + x * k), int(n / 2 - y * k))
    for rt in W['roads']['routes']:
        pts = np.array([P(x, y) for x, y in rt['pts'][::2]], np.int32)
        cv2.polylines(im, [pts], False, (60, 60, 60), 1, cv2.LINE_AA)
    for j in range(S.shape[0]):
        cv2.circle(im, P(S[j, G.S_X], S[j, G.S_Y]), max(2, int(0.5 * k)), (150, 150, 170), -1)
    cs = CR.state(f)
    cam = SC.camera(f, 1.0)
    # the camera footprint
    C, Rv, U, Fw = cam[0:3], cam[3:6], cam[6:9], cam[9:12]; fl, cx, cy = cam[12], cam[13], cam[14]
    fp = []
    for (x, y) in [(0, 0), (1920, 0), (1920, 804), (0, 804)]:
        d = Fw + (x - cx) / fl * Rv - (y - cy) / fl * U; d /= np.linalg.norm(d)
        tt = -C[2] / d[2] if d[2] < 0 else 400.0
        q = C + d * min(tt, 400.0); fp.append(P(q[0], q[1]))
    cv2.polylines(im, [np.array(fp, np.int32)], True, (0, 120, 255), 1, cv2.LINE_AA)
    x, y, lit = cs['x'], cs['y'], cs['lit']
    CF = cs['CF']
    walking = CF[:, G.F_HEM] > 0.01
    for i in range(len(x)):
        col = (40, 170, 255) if lit[i] > 0.5 else (90, 60, 50)
        if walking[i]: col = (60, 230, 255) if lit[i] > 0.5 else (160, 110, 90)
        cv2.circle(im, P(x[i], y[i]), max(1, int(0.25 * k)), col, -1)
        a_ = CF[i, G.F_ANG]
        if k > 6: cv2.line(im, P(x[i], y[i]), P(x[i] + 0.4 * math.cos(a_), y[i] + 0.4 * math.sin(a_)), (255, 255, 255), 1)
    # the council + her
    Fa = SC.figures(f)
    for i in range(SC.NFIG):
        col = (40, 40, 220) if i == SC.HER else (200, 200, 60)
        cv2.circle(im, P(Fa[i, G.F_X], Fa[i, G.F_Y]), max(2, int(0.25 * k)), col, -1)
    hw = CR.her_walkin(f)
    if hw is not None:
        cv2.circle(im, P(*hw['pos']), max(3, int(0.3 * k)), (0, 0, 255), 2)
    cv2.putText(im, f'{int(f)}  n={len(x)} lit={lit.mean():.2f}', (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    tiles.append(im)
cols = 3
while len(tiles) % cols: tiles.append(np.zeros_like(tiles[0]))
rows = [np.concatenate(tiles[i:i + cols], 1) for i in range(0, len(tiles), cols)]
cv2.imwrite(a.out, np.concatenate(rows, 0), [cv2.IMWRITE_JPEG_QUALITY, 88])
print(a.out)
