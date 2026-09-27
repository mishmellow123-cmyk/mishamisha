"""MONTAGE-MELT: preview the melt's geometry and letters without Blender (numpy software raster through the shot's
real camera). Flat-ish metal shading, the script of fire in red where it is awake, the stone as a dark plane.

  source ~/.venvs/longdawn/env.sh; cd the-long-dawn/shots/montage3d
  python tests/t_meltprev.py 5381,5410,5420,5433,5440,5443,5446,5449,5452,5456,5464,5480 out.jpg [scale]
"""
import math
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import meltc as M  # noqa: E402


def _q2m(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def _look(pos, tgt):
    f = np.array(tgt, float) - np.array(pos, float)
    f /= np.linalg.norm(f)
    r = np.cross(f, [0.0, 0.0, 1.0])
    r /= np.linalg.norm(r)
    u = np.cross(r, f)
    return r, u, f


def render(f, W=960, H=402, topo=None, tex=None):
    V = M.melt_verts(f)
    c, q = M.ring_state(f)
    R = _q2m(q)
    V = (V - np.array([0.0, 0.0, M.HW])) @ R.T + np.array(c)
    pos, tgt, _ = M.cam_state(f)
    r, u, fw = _look(pos, tgt)
    rel = V - np.array(pos)
    x, y, z = rel @ r, rel @ u, rel @ fw
    fpx = 100.0 / 36.0 * W
    sx = W / 2 + fpx * x / z
    sy = H / 2 - fpx * y / z
    img = np.zeros((H, W, 3), np.float32)
    # the stone: z = 0 plane, shaded by distance
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    d = np.stack([(xx - W / 2) / fpx, -(yy - H / 2) / fpx, np.ones_like(xx)], -1)
    dw = d[..., 0:1] * r + d[..., 1:2] * u + d[..., 2:3] * fw
    t = -pos[2] / np.minimum(dw[..., 2], -1e-6)
    hit = (dw[..., 2] < 0) & (t > 0)
    img[hit] = (0.05, 0.045, 0.04)
    Q = topo['quads']
    zq = z[Q].mean(axis=1)
    order = np.argsort(-zq)
    P0, P1, P3 = V[Q[:, 0]], V[Q[:, 1]], V[Q[:, 3]]
    N = np.cross(P1 - P0, P3 - P0)
    N /= np.linalg.norm(N, axis=1, keepdims=True) + 1e-12
    view = V[Q].mean(axis=1) - np.array(pos)
    view /= np.linalg.norm(view, axis=1, keepdims=True)
    L = np.array([0.0, 0.6, 0.8])
    L /= np.linalg.norm(L)
    lam = np.clip(N @ L, 0, 1)
    refl = view - 2 * (view * N).sum(1, keepdims=True) * N
    spec = np.clip(refl @ np.array([0.0, 0.7, 0.7]) / 0.99, 0, 1) ** 30
    uvc = topo['quv'].mean(axis=1)
    outer = topo['outer'][Q].mean(axis=1)
    inner = topo['inner'][Q].mean(axis=1)
    th_, tw_ = tex[0].shape
    tu = ((uvc[:, 0] % 1.0) * tw_).astype(int) % tw_
    tv = ((1.0 - uvc[:, 1]) * th_).astype(int).clip(0, th_ - 1)
    ti = ((1.0 - (uvc[:, 0] % 1.0)) * tex[1].shape[1]).astype(int) % tex[1].shape[1]
    lt = tex[0][tv, tu] * np.clip((outer - 0.25) / 0.35, 0, 1) + tex[1][tv, ti] * np.clip((inner - 0.25) / 0.35, 0, 1)
    Lk = M.letters(f)
    base = np.array([0.28, 0.18, 0.05])
    col = base[None] * (0.25 + 0.75 * lam[:, None]) + spec[:, None] * np.array([1.0, 0.85, 0.5])
    col = col + (Lk * lt)[:, None] * np.array([1.0, 0.22, 0.03])
    col = col + M.hot(f) * np.array([1.0, 0.35, 0.05])[None] * 0.5
    pts = np.stack([sx, sy], -1)
    for k in order:
        poly = np.round(pts[Q[k]] * 4).astype(np.int32)
        cv2.fillConvexPoly(img, poly, tuple(float(c) for c in col[k][::-1]), lineType=cv2.LINE_AA, shift=2)
    out = np.clip(img, 0, 1) ** (1 / 2.2)
    out = (out * 255).astype(np.uint8)
    cv2.putText(out, str(f), (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def main():
    frames = [int(float(x)) for x in sys.argv[1].split(',')]
    out = sys.argv[2]
    sc = float(sys.argv[3]) if len(sys.argv) > 3 else 0.5
    W, H = int(1920 * sc), int(804 * sc)
    T = M.topology()
    tex = []
    for nm in ('inscription_outer.png', 'inscription_inner.png'):
        im = cv2.imread(os.path.join(M.ADIR, nm), cv2.IMREAD_UNCHANGED).astype(np.float32) / 65535.0
        tex.append(cv2.resize(im, (im.shape[1] // 4, im.shape[0] // 4), interpolation=cv2.INTER_AREA))
    tiles = [render(f, W, H, T, tex) for f in frames]
    cols = 3 if len(tiles) > 4 else len(tiles)
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)]
    cv2.imwrite(out, np.vstack(rows))
    print('wrote', out)


if __name__ == '__main__':
    main()
