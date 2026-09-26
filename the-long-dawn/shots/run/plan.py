"""Top-down plan of the Run: hillshade of the set, camera path (ticks every 5 frames), look direction,
clearance above the terrain, and beacons with their ignition frames. Writes renders/run_v2/tests/plan.png"""
import math
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import world as WD      # noqa: E402
import run as RN        # noqa: E402


def main(x0=-450.0, z0=-50.0, size=1000.0, n=1000, out="plan.png", beacons=None):
    dx = size / n
    H = np.zeros((n, n))
    WD.height_grid(x0, z0, dx, n, n, RN.CR, 63.0, H)
    gy, gx = np.gradient(H, dx)
    L = WD.MOON_DIR
    nrm = np.stack([-gx, np.ones_like(H), -gy], -1)
    nrm /= np.linalg.norm(nrm, axis=-1, keepdims=True)
    sh = np.clip(nrm @ L, 0, 1)
    img = np.stack([sh] * 3, -1) * 0.85 + 0.1
    img[H < WD.CLOUD_Y] = (0.35, 0.22, 0.12)
    img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    for lev in range(-600, 100, 50):
        m = (H[:-1, :] < lev) != (H[1:, :] < lev)
        img[:-1][m] = (0, 140, 200) if lev % 100 == 0 else (0, 90, 130)

    def px(x, z):
        return int((x - x0) / dx), int((z - z0) / dx)

    fr = np.arange(RN.START, RN.END + 1)
    P = np.array([RN.path(f) for f in fr])
    for k in range(len(fr) - 1):
        cv2.line(img, px(P[k, 0], P[k, 2]), px(P[k + 1, 0], P[k + 1, 2]), (0, 0, 255), 1)
    rows = []
    for k, f in enumerate(fr):
        x, y, z = P[k]
        g = WD.ground(x, z, RN.CR, 0.2)
        yaw, pitch = RN.look(f)
        v = np.linalg.norm(RN.path(f + 0.5) - RN.path(f - 0.5)) * 24
        rows.append((f, x, y, z, g, y - g, v, yaw, pitch, RN.bank(f)))
        if f % 5 == 0:
            c = px(x, z)
            cv2.circle(img, c, 2, (0, 0, 255), -1)
            a = math.radians(yaw)
            e = px(x + 60 * math.sin(a), z + 60 * math.cos(a))
            cv2.line(img, c, e, (0, 255, 0), 1)
    if beacons is not None:
        for b in beacons:
            c = px(b['pos'][0], b['pos'][2])
            cv2.circle(img, c, 4, (0, 200, 255), -1)
    img = img[::-1].copy()
    for f in range(RN.START, RN.END + 1, 20):
        k = f - RN.START
        c = px(P[k, 0], P[k, 2])
        cv2.putText(img, str(f), (c[0] + 5, n - c[1]), 0, 0.4, (255, 255, 255), 1)
    p = os.path.join(WD.CM.ROOT, 'renders', 'run_v2', 'tests', out)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    cv2.imwrite(p, img)
    print(p)
    print(' frame      x      y      z  ground  clear  speed    yaw  pitch  roll')
    for r in rows:
        if r[0] % 5 == 0 or r[5] < 12:
            print('%6d %6.0f %6.1f %6.0f %7.1f %6.1f %6.1f %6.1f %6.1f %5.1f' % r)
    print('pyre', RN.PYRE_XZ, 'crags top', [round(t, 1) for t in RN.CR[:, 2]])
    for f in range(1560, 1606, 2):
        c = RN.camera(f)
        sx, sy, z = c.project(RN.PYRE_XZ + np.array([0, 2.5, 0]))
        print('  pyre @%d: sx %.0f sy %.0f dist %.0f' % (f, sx, sy, np.linalg.norm(RN.PYRE_XZ - c.pos)))


if __name__ == '__main__':
    main()
