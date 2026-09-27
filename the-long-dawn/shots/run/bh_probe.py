"""RUN-A2 tool: where is the valley under the cloud sea east of the blue-hour shoulder? (look-dev only)

  python bh_probe.py --frames 1,2,3        -> renders/bluehour_A/probe/f_00001..3
  f1: top-down height map, forward = east (the crossing's walking direction), right = south; the shoulder at the
      left edge's middle; grid every 2 km; bands: valley floor < -1500 green, -1500..-1000 olive, -1000..-800 brown,
      the cloud band grey-blue, islands above the cloud white.
  f2: the blue hour's view with the cloud sea removed (a hole row over everything; valley floor shading on).
  f3: the same view with the cloud sea, for reference.
"""
import argparse
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ.setdefault('NUMBA_NUM_THREADS', '8')

import numpy as np          # noqa: E402
import cv2                  # noqa: E402

import world as WD          # noqa: E402
import rcam as RC           # noqa: E402
import pipe as PI           # noqa: E402
import crossing as X        # noqa: E402

SH_U, SH_V = 330.0, 14.0
OUT = os.path.join(PI.CM.ROOT, 'renders', 'bluehour_A', 'probe')


def topdown():
    W, H = 1920, 804
    mpp = 15.625                                    # 30 km across
    fw = np.arange(W) * mpp
    lat = (np.arange(H) - H / 2) * mpp
    FF, LL = np.meshgrid(fw, lat)
    c = X.uv(SH_U, SH_V)
    xs = c[0] + X.E3[0] * FF + X.S3[0] * LL
    zs = c[2] + X.E3[2] * FF + X.S3[2] * LL
    h = np.zeros(xs.size)
    WD.heights(xs.ravel().copy(), zs.ravel().copy(), 10.0, X.CR, h)
    h = h.reshape(H, W)
    img = np.zeros((H, W, 3), np.float32)
    bands = [(-1e9, -1500, (0.20, 0.45, 0.20)), (-1500, -1000, (0.35, 0.40, 0.18)), (-1000, -800, (0.40, 0.30, 0.18)),
             (-800, -500, (0.45, 0.50, 0.62)), (-500, 1e9, (0.92, 0.92, 0.95))]
    for lo, hi, col in bands:
        m = (h >= lo) & (h < hi)
        img[m] = col
    gy, gx = np.gradient(h, mpp)
    shade = np.clip(0.75 - 0.35 * (gx * 0.7 - gy * 0.7), 0.3, 1.2)
    img *= shade[..., None]
    img = (np.clip(img, 0, 1) * 255).astype(np.uint8)[..., ::-1].copy()
    for k in range(0, 31, 2):
        x = int(k * 1000 / mpp)
        cv2.line(img, (x, 0), (x, H - 1), (60, 60, 60), 1)
        cv2.putText(img, f'{k}km', (x + 3, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
    for k in range(-6, 7, 2):
        y = int(H / 2 + k * 1000 / mpp)
        cv2.line(img, (0, y), (W - 1, y), (60, 60, 60), 1)
        cv2.putText(img, f'{k:+d}km S' if k else '0', (3, y - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
    # view directions (yaw relative to east; negative = left = north = up on this map)
    for dyaw, col in ((-18.0, (0, 0, 255)), (0.0, (0, 140, 255)), (12.0, (0, 200, 0)), (24.0, (255, 0, 0))):
        a = math.radians(dyaw)
        x1 = int(30000 * math.cos(a) / mpp)
        y1 = int(H / 2 + 30000 * math.sin(a) / mpp)
        cv2.line(img, (0, H // 2), (x1, y1), col, 1)
        cv2.putText(img, f'{dyaw:+.0f}', (min(x1, W - 60) - 40, min(max(y1, 20), H - 8)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, col, 1)
    return img[..., ::-1].astype(np.float32) / 255.0


def view(frame, cloud):
    t = 0.0
    W, H = 1920, 804
    base = X.uv(SH_U - 2.0, SH_V)
    base[1] = WD.ground(base[0], base[2], X.CR) + 1.6
    cam = RC.RCam(base, X.YAW_E + 4.0, -7.0, 0.0, 64.0, W, H)
    fr = PI.Frame(cam, 1.0)
    scam = fr.src
    C = scam.params()
    CR = X.CR if cloud else np.vstack([WD.hole_row(base[0], base[2], 90000.0, 6000.0, edge=0.01)[None, :], X.CR])
    Pp = np.array([scam.pos[0], scam.pos[2], t, 0.0])
    D = np.zeros((scam.H, scam.W))
    WD.march(Pp, CR, C, 0.2, 90000.0, 0.0035, 0.35, 700.0, 9, D)
    Lk, amb, S, fogp, Q = WD.night_light()
    e, a = math.radians(14.0), math.radians(X.YAW_E)
    Lk = np.r_[np.array([math.cos(e) * math.sin(a), math.sin(e), math.cos(e) * math.cos(a)]), PI.CM.lin('#F0B4A0')]
    amb = PI.CM.lin('#5B6FAA') * 0.45
    Q = Q.copy()
    Q[0] = 0.15
    Q[19] = -1250.0
    fogc = PI.CM.lin('#707CAA') * 0.28
    fogp = np.array([3.0e-5, 1 / 1500.0, 2.2e-4, 1 / 140.0, 2.5, fogc[0], fogc[1], fogc[2], 150.0])
    fr.img = np.zeros((scam.H, scam.W, 3), np.float32)
    fr.zb = np.zeros((scam.H, scam.W), np.float32)
    fr.dist = np.zeros((scam.H, scam.W), np.float32)
    WD.shade(C, D, Pp, CR, S, np.zeros((0, 8)), Lk, Q, amb, fogp, fr.img, fr.zb, fr.dist, np.zeros((0, 4)))
    img, zb, di = PI.to_target(fr)
    img = PI.look.finish(img, exposure=2.2, bloom_strength=0.0, streak_strength=0.0, vignette_amount=0.0)
    # distance labels along the image's central column and a few others
    out = (np.clip(img, 0, 1) * 255).astype(np.uint8)[..., ::-1].copy()
    for xf in (0.2, 0.4, 0.5, 0.6, 0.8):
        x = int(W * xf)
        for y in range(40, H, 60):
            d = di[y, x]
            if d < 1e8:
                cv2.putText(out, f'{d / 1000:.1f}', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
    cv2.putText(out, f'yaw {X.YAW_E + 4.0:.0f} (east = {X.YAW_E:.0f}) pitch -7 hfov 64  cloud={cloud}', (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
    return out[..., ::-1].astype(np.float32) / 255.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--frames', default='1,2,3')
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    X.wfs()
    for f in [int(v) for v in a.frames.split(',')]:
        img = topdown() if f == 1 else view(f, cloud=(f == 3))
        PI.look.save_png(PI.look.frame_path(OUT, f), img)
        print('frame', f, flush=True)


if __name__ == '__main__':
    main()
