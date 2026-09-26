"""Top-down diagnostic map of the land function: hillshade, cloud-covered area tinted blue,
contours every 500 m, designed points marked. python landmap.py out.png [x0 y0 size px]"""
import math
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mtn import land
import world
if os.environ.get('XLP'):
    import json as _j
    world.LAND_PARAMS = np.array(_j.loads(os.environ['XLP']))

out = sys.argv[1]
x0, y0, size = (float(v) for v in (sys.argv[2:5] if len(sys.argv) > 4 else (-20000, -5000, 40000)))
px = int(sys.argv[5]) if len(sys.argv) > 5 else 800
dx = size / px
g = land.raster(px, px, x0, y0, dx, *world.land_args()).astype(np.float64)
print('h range', g.min(), g.max(), 'frac above 0:', (g > 0).mean(), 'above 500', (g > 500).mean())
gy, gx = np.gradient(g, dx)
n = np.stack([-gx, -gy, np.ones_like(g)], -1)
n /= np.linalg.norm(n, axis=-1, keepdims=True)
az, el = math.radians(300), math.radians(35)
L = np.array([math.cos(el) * math.sin(az), math.cos(el) * math.cos(az), math.sin(el)])
s = np.clip(n @ L, 0, 1)
img = np.stack([s, s, s], -1) * 0.85 + 0.15 * ((g - g.min()) / (g.max() - g.min()))[..., None]
cloud = g < 0
img[cloud] = img[cloud] * 0.35 + np.array([0.45, 0.30, 0.15]) * 0.65   # BGR-ish tint later flipped
img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
# contours
for lev in range(500, 4000, 500):
    m = (g > lev).astype(np.uint8)
    cs, _ = cv2.findContours(m, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(img, cs, -1, (60, 200, 255) if lev % 1000 == 0 else (40, 120, 160), 1)
img = img[::-1].copy()     # north up
for name, (x, y) in world.MARKS.items():
    i = int((x - x0) / dx)
    j = px - 1 - int((y - y0) / dx)
    if 0 <= i < px and 0 <= j < px:
        cv2.circle(img, (i, j), 4, (0, 0, 255), -1)
        cv2.putText(img, name, (i + 5, j - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
cv2.imwrite(out, img)
