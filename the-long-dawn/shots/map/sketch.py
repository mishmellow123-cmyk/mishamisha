"""MAP-L2's world sketches: review stills of the INVENTED world, written as frames so the farm can render them.

    python sketch.py --frames 1-6 --out renders/map_sketch

  1  the whole sheet, design colours (relief tint, rain, rivers, lakes, coasts); the ring, her beacon, the Run's
     seven and the Road marked; the camera's footprint at the open (4160), the widest (~4390) and the end (4479)
  2  the camera's widest view (v2 2040 ~ C 4390) in design colours, through the shot's own camera
  3  the whole sheet inked (the bake at low resolution): what the hand draws, everywhere
  4  the widest view inked, through the camera (no fires, no light): the geography as the viewer will see it
  5  the opening (4160) inked: her range, her beacon and the Run's seven
  6  the end (4479) inked: the ring of stones on the High Moor
"""
import argparse
import math
import os
import sys
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..', '..', 'lib'))
import look  # noqa: E402
import geo  # noqa: E402
import terra  # noqa: E402

W_, H_ = 1920, 804


def design_rgb(X0, Y1, ppd, W, H):
    """Design colours on a map-space pixel grid (sRGB float)."""
    E = geo.on_grid('E', X0, Y1, ppd, W, H)
    P = geo.on_grid('rain', X0, Y1, ppd, W, H)
    land = geo.on_grid(terra.world()['land'].astype(np.float32), X0, Y1, ppd, W, H) > 0.5
    lake = geo.on_grid(terra.world()['lake'].astype(np.float32), X0, Y1, ppd, W, H) > 0.5
    sea = np.array([0.56, 0.63, 0.67])
    wet = np.array([0.50, 0.63, 0.40])
    dry = np.array([0.91, 0.81, 0.58])
    hi = np.array([0.50, 0.41, 0.33])
    snow = np.array([0.93, 0.91, 0.89])
    r = np.clip((P - 0.2) / 1.2, 0, 1)[..., None]
    base = dry * (1 - r) + wet * r
    e = E[..., None]
    a = np.clip((e - 2.2) / 3.8, 0, 1)
    b = np.clip((e - 6.0) / 3.0, 0, 1)
    col = base * (1 - a) + hi * a
    col = col * (1 - b) + snow * b
    gy, gx = np.gradient(cv2.GaussianBlur(E, (0, 0), max(ppd * 0.12, 0.6)), 1.0 / ppd)
    sh = np.clip(1.0 - (gx * 0.6 - gy * 0.6) * 0.06, 0.55, 1.35)
    col = col * sh[..., None]
    img = np.where(land[..., None], col, sea)
    img = np.where(lake[..., None], np.array([0.46, 0.56, 0.63]), img)
    return np.clip(img, 0, 1).astype(np.float32)


def overlay(im8, to_px, lw=1.0, road=None, chain=None, rivers=True, coasts=True, marks=True, feet=()):
    """Rivers, coasts and markers drawn on a uint8 BGR image through a map->pixel function."""
    if rivers:
        for P_, fl, _ in terra.world()['rivers']:
            w = np.clip(np.sqrt(fl / terra.RIVER_T) * 0.55 * lw, 1, 5)
            q = np.round(to_px(P_) * 16).astype(np.int32)
            for k in range(len(q) - 1):
                cv2.line(im8, tuple(q[k]), tuple(q[k + 1]), (150, 95, 45), int(round(w[k])), cv2.LINE_AA, shift=4)
    if coasts:
        for xy, hole in terra.world()['coast']:
            q = np.round(to_px(xy) * 16).astype(np.int32)
            cv2.polylines(im8, [q], True, (40, 40, 40), max(1, int(lw)), cv2.LINE_AA, shift=4)
        for xy in terra.world()['lakes']:
            q = np.round(to_px(xy) * 16).astype(np.int32)
            cv2.polylines(im8, [q], True, (90, 70, 50), 1, cv2.LINE_AA, shift=4)
    for F, col in feet:
        q = np.round(to_px(F) * 16).astype(np.int32)
        cv2.polylines(im8, [q], True, col, 2, cv2.LINE_AA, shift=4)
    if road is not None:
        q = np.round(to_px(road) * 16).astype(np.int32)
        cv2.polylines(im8, [q], False, (20, 20, 150), max(2, int(2 * lw)), cv2.LINE_AA, shift=4)
    if chain is not None:
        for c in chain:
            p = to_px(np.array([c]))[0]
            cv2.circle(im8, (int(p[0]), int(p[1])), max(3, int(3 * lw)), (0, 120, 255), -1, cv2.LINE_AA)
    if marks:
        p = to_px(np.array([terra.RING]))[0]
        cv2.circle(im8, (int(p[0]), int(p[1])), max(6, int(7 * lw)), (0, 0, 210), 2, cv2.LINE_AA)
        p = to_px(np.array([terra.BEACON]))[0]
        cv2.drawMarker(im8, (int(p[0]), int(p[1])), (0, 90, 255), cv2.MARKER_TRIANGLE_UP, max(12, int(16 * lw)), 2)


def footprint(cam, n=24):
    """The map-space outline of a camera's frame."""
    t = np.linspace(0, 1, n)
    W, H = cam.W, cam.H
    xs = np.concatenate([t * W, np.full(n, W), (1 - t) * W, np.zeros(n)])
    ys = np.concatenate([np.zeros(n), t * H, np.full(n, H), (1 - t) * H])
    X, Y = cam.screen_to_map(xs, ys)
    return np.stack([X, Y], 1)


def through(cam, X0, Y1, ppd, img):
    """Warp a map-space image (pixel (i, j) centre at X0 + (j+.5)/ppd, Y1 - (i+.5)/ppd) through the camera."""
    A = np.array([[ppd, 0, -X0 * ppd - 0.5], [0, -ppd, Y1 * ppd - 0.5], [0, 0, 1.0]])
    T = A @ cam.Hinv
    return cv2.warpPerspective(img, T, (cam.W, cam.H), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=(0.09, 0.06, 0.04))


def cams():
    import road
    R = terra.RING
    return {'open': road.camera_at(1920.0, R), 'wide': road.camera_at(2040.0, R), 'end': road.camera_at(2087.0, R)}


def the_road():
    import road
    import relay
    try:
        r = relay.Relay()
        a = r.P[0]
        chain = r.P[r.chain[1:]]
    except Exception as e:                      # before the glyphs exist, the design points
        print('relay not available:', e)
        a = np.array(terra.BEACON)
        chain = np.array(relay.CHAIN)
    return road.road_path(a, np.array(terra.RING)), chain


def frame(k, out):
    t0 = time.time()
    if k == 1:
        ppd = H_ / (geo.MAP_Y1 - geo.MAP_Y0 + 4.0)
        W = int(round(360 * ppd))
        X0, Y1 = -180.0, geo.MAP_Y1 + 2.0
        img = design_rgb(X0, Y1, ppd, W, H_)
        im8 = (img[..., ::-1] * 255).astype(np.uint8).copy()
        cm = cams()
        rd, ch = the_road()

        def to_px(P):
            P = np.asarray(P, np.float64)
            return np.stack([(P[:, 0] - X0) * ppd - 0.5, (Y1 - P[:, 1]) * ppd - 0.5], 1)
        overlay(im8, to_px, 1.0, road=rd, chain=ch,
                feet=[(footprint(cm['open']), (0, 200, 255)), (footprint(cm['wide']), (255, 255, 255)),
                      (footprint(cm['end']), (0, 0, 255))])
        full = np.full((H_, W_, 3), 20, np.uint8)
        x = (W_ - W) // 2
        full[:, x:x + W] = im8
        img = full[..., ::-1].astype(np.float32) / 255.0
    elif k == 2:
        cam = cams()['wide']
        ppd = 24.0
        X0, Y1, W, H = -70.0, 72.0, int(150 * ppd), int(110 * ppd)
        tex = design_rgb(X0, Y1, ppd, W, H)
        im8 = (tex[..., ::-1] * 255).astype(np.uint8).copy()
        rd, ch = the_road()

        def to_px(P):
            P = np.asarray(P, np.float64)
            return np.stack([(P[:, 0] - X0) * ppd - 0.5, (Y1 - P[:, 1]) * ppd - 0.5], 1)
        overlay(im8, to_px, 2.0, road=rd, chain=ch)
        img = through(cam, X0, Y1, ppd, im8[..., ::-1].astype(np.float32) / 255.0)
    elif k == 3:
        import bake
        ppd = H_ / (geo.MAP_Y1 - geo.MAP_Y0 + 4.0)
        W = int(round(360 * ppd))
        r = bake.bake_region(-180.0, geo.MAP_Y1 + 2.0, W, H_, ppd)
        full = np.full((H_, W_, 3), 0.01, np.float32)
        x = (W_ - W) // 2
        full[:, x:x + W] = bake.l2s(r['rgb'])
        img = full
    else:
        import bake
        cam = cams()[{4: 'wide', 5: 'open', 6: 'end'}[k]]
        F = footprint(cam)
        ppd = {4: 22.0, 5: 40.0, 6: 90.0}[k]
        X0, X1 = F[:, 0].min() - 1.0, F[:, 0].max() + 1.0
        Y0, Y1 = F[:, 1].min() - 1.0, F[:, 1].max() + 1.0
        W, H = int((X1 - X0) * ppd), int((Y1 - Y0) * ppd)
        print('bake region', round(X0, 1), round(Y1, 1), W, H, ppd, flush=True)
        r = bake.bake_region(X0, Y1, W, H, ppd)
        img = through(cam, X0, Y1, ppd, bake.l2s(r['rgb']).astype(np.float32))
    look.save_png(look.frame_path(out, k), np.clip(img, 0, 1))
    print('sketch', k, round(time.time() - t0, 1), 's', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--frames', default='1-6')
    ap.add_argument('--out', default=os.path.join(geo.ROOT, 'renders', 'map_sketch'))
    a = ap.parse_args()
    fr = []
    for part in a.frames.split(','):
        if '-' in part:
            x, y = part.split('-')
            fr += range(int(x), int(y) + 1)
        elif part:
            fr.append(int(part))
    out = a.out if os.path.isabs(a.out) else os.path.join(geo.ROOT, a.out)
    os.makedirs(out, exist_ok=True)
    for k in fr:
        frame(k, out)


if __name__ == '__main__':
    main()
