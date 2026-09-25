#!/usr/bin/env python3
"""
From light to paint.

The ray tracer knows only light. This turns its image into an oil painting on
a poplar panel, the way one would age a real one in reverse:

  tone        a filmic curve, lifted so that nothing is ever quite black
              (oil paint isn't), warm in the lights, cool in the half-tones
  palette     ultramarine deepened in the sky, greens turned toward the
              olive and umber that copper greens become with time
  paint       a generalised Kuwahara filter breaks the render's photographic
              smoothness into small facets of paint
  brushwork   strokes follow the forms (along the isophotes); the sky is
              laid in with long horizontal strokes
  impasto     the lights are thicker, and catch a little light of their own
  craquelure  a net of fine cracks, running mostly across the grain of a
              vertically grained panel
  varnish     old, yellowed, a little darker toward the edges

Usage: python3 oil.py hdr.npy out.png
"""
import sys
import time

import numpy as np
from PIL import Image
from scipy import ndimage
from scipy.spatial import cKDTree

IN, OUT = sys.argv[1], sys.argv[2]
W, H = 2400, 3200
T0 = time.time()


def log(m):
    print(f"[{time.time() - T0:6.1f}s] {m}", flush=True)


def smoothstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def resize(a, w, h, method=Image.LANCZOS):
    if a.ndim == 2:
        return np.asarray(Image.fromarray(a.astype(np.float32), "F").resize((w, h), method))
    return np.stack([resize(a[..., c], w, h, method) for c in range(a.shape[2])], -1)


rng = np.random.default_rng(1504)

# ---------------------------------------------------------------------------
# tone
# ---------------------------------------------------------------------------
hdr = np.load(IN).astype(np.float32)
aux = np.load(IN.replace(".npy", "_aux.npz"))
x = hdr * 0.56
# a painted glow around what is brightest: the glints, the gathered sun
hl = np.clip((x @ np.array([0.2126, 0.7152, 0.0722], np.float32)) - 1.1, 0, 40)
small = ndimage.zoom(hl, 0.25, order=1)
glow = 0.10 * ndimage.gaussian_filter(small, 3) + 0.05 * ndimage.gaussian_filter(small, 12)
glow = ndimage.zoom(glow, (x.shape[0] / small.shape[0], x.shape[1] / small.shape[1]), order=1)[:x.shape[0], :x.shape[1]]
x = x + glow[..., None] * np.array([1.0, 0.85, 0.62], np.float32)
del hl, small, glow
# the brightest lights (the gathered sun, the glints) are sacred: no filter touches them
hot = smoothstep(1.0, 2.2, x @ np.array([0.2126, 0.7152, 0.0722], np.float32))
x = (x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14)
x = np.clip(x, 0, 1)
disp = np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(x, 1 / 2.4) - 0.055)
img = resize(disp, W, H)
hot = np.clip(resize(hot, W, H, Image.BILINEAR), 0, 1)
hot = np.clip(ndimage.maximum_filter(hot, 3) * 1.2, 0, 1)
cat = np.asarray(Image.fromarray(aux["cat"]).resize((W, H), Image.NEAREST))
elev = resize(aux["elev"], W, H, Image.BILINEAR)
del hdr, x, disp
img = np.clip(img, 0, 1)
log("tone")

# ---------------------------------------------------------------------------
# palette
# ---------------------------------------------------------------------------
sky = np.clip(resize(aux["sky"], W, H, Image.BILINEAR), 0, 1) * (cat == 0)
lum = img @ np.array([0.2126, 0.7152, 0.0722], np.float32)
# the sky: a painter's gradient, ultramarine overhead to a luminous horizon;
# the clouds keep their own light and shade on top of it
e = np.clip(elev, 0, 40)
top = np.array([0.14, 0.26, 0.58], np.float32)
mid = np.array([0.44, 0.57, 0.80], np.float32)
hor = np.array([0.90, 0.84, 0.70], np.float32)
a1 = smoothstep(0.5, 9.0, e)[..., None]
a2 = smoothstep(8.0, 26.0, e)[..., None]
grad = hor * (1 - a1) + mid * a1
grad = grad * (1 - a2) + top * a2
smooth_l = ndimage.gaussian_filter(lum * sky, 40) / (ndimage.gaussian_filter(sky, 40) + 1e-4)
cloud = np.clip(lum - smooth_l, -0.3, 0.4)[..., None]
skycol = grad * (1 + 2.0 * cloud) + 0.55 * np.clip(cloud, 0, None) * np.array([1.0, 0.95, 0.85], np.float32)
img = img * (1 - 0.85 * sky[..., None]) + skycol * (0.85 * sky[..., None])
# greens toward olive: pull the green channel down a little where it leads,
# then give the valley back some of the depth the air took from it
g_lead = np.clip(img[..., 1] - 0.5 * (img[..., 0] + img[..., 2]), 0, 1)
img[..., 1] -= 0.30 * g_lead
img[..., 0] += 0.10 * g_lead
land = ((cat == 0) & (sky < 0.5)).astype(np.float32)
land = ndimage.gaussian_filter(land, 1.0)[..., None]
lum = img @ np.array([0.2126, 0.7152, 0.0722], np.float32)
sat = img + (img - lum[..., None]) * 0.25
img = img * (1 - land) + np.clip(sat, 0, 1) * land
# the stone a touch warmer
stone = ((cat >= 2) & (cat <= 4)).astype(np.float32)[..., None]
img = img * (1 + stone * np.array([0.025, 0.008, -0.025], np.float32))
# warm lights, cool half-tones, lifted umber shadows
lum = img @ np.array([0.2126, 0.7152, 0.0722], np.float32)
warm = (smoothstep(0.45, 0.95, lum) * (1 - sky))[..., None]
img = img * (1 + warm * np.array([0.035, 0.005, -0.05], np.float32))
cool = (smoothstep(0.15, 0.4, lum) * smoothstep(0.65, 0.4, lum))[..., None]
img = img * (1 + cool * np.array([-0.02, 0.0, 0.025], np.float32))
umber = np.array([0.085, 0.062, 0.040], np.float32)
img = umber + (1 - umber) * img
img = np.clip(img, 0, 1)
log("palette")

# ---------------------------------------------------------------------------
# paint: a generalised Kuwahara filter (8 smooth sectors)
# ---------------------------------------------------------------------------


def kuwahara(im, radius=4, q=6.0):
    r = radius
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1].astype(np.float32)
    ang = np.arctan2(yy, xx)
    rad = np.hypot(xx, yy)
    g = np.exp(-(rad / (0.55 * r)) ** 2) * (rad <= r + 0.5)
    num = np.zeros_like(im)
    den = np.zeros(im.shape[:2], np.float32)
    sq = im * im
    for k in range(8):
        c = -np.pi + (k + 0.5) * np.pi / 4
        dlt = np.angle(np.exp(1j * (ang - c)))
        wsec = np.clip(np.cos(dlt * 2.0), 0, 1) ** 2
        wsec[r, r] = 1.0
        K = (g * wsec).astype(np.float32)
        K /= K.sum()
        m = np.stack([ndimage.correlate(im[..., ch], K, mode="reflect") for ch in range(3)], -1)
        v = np.stack([ndimage.correlate(sq[..., ch], K, mode="reflect") for ch in range(3)], -1) - m * m
        s = np.sqrt(np.maximum(v.sum(-1), 0))
        wk = (1 + 60.0 * s) ** (-q)
        num += m * wk[..., None]
        den += wk
    return num / den[..., None]


kw = kuwahara(img, 4)
# glass and lettering keep more of their crispness
keep = np.where((cat == 1) | (cat == 5), 0.55, 0.15)
keep = ndimage.gaussian_filter(keep.astype(np.float32), 2.0)
keep = np.maximum(keep, hot)[..., None]
img = kw * (1 - keep) + img * keep
del kw
log("paint")

# ---------------------------------------------------------------------------
# brushwork: strokes along the forms
# ---------------------------------------------------------------------------
lum = img @ np.array([0.2126, 0.7152, 0.0722], np.float32)
ls = ndimage.gaussian_filter(lum, 1.5)
gy, gx = np.gradient(ls)
jxx = ndimage.gaussian_filter(gx * gx, 7)
jyy = ndimage.gaussian_filter(gy * gy, 7)
jxy = ndimage.gaussian_filter(gx * gy, 7)
theta = 0.5 * np.arctan2(2 * jxy, jxx - jyy)          # across the forms
coh = np.sqrt((jxx - jyy) ** 2 + 4 * jxy ** 2) / (jxx + jyy + 1e-7)
# stroke direction: along the isophotes; the sky with long horizontals
sx, sy = -np.sin(theta), np.cos(theta)
flat = np.clip(np.clip(1 - coh * 1.5, 0, 1) * 0.6 + 0.4 * sky, 0, 1)
sx = sx * (1 - flat) + 1.0 * flat
sy = sy * (1 - flat)
n = np.hypot(sx, sy) + 1e-6
sx, sy = sx / n, sy / n
noise = ndimage.gaussian_filter(rng.standard_normal((H, W)).astype(np.float32), 0.8)
yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
acc = noise.copy()
wsum = np.ones_like(noise)
for sgn in (1, -1):
    px, py = xx.copy(), yy.copy()
    for k in range(1, 10):
        vx = ndimage.map_coordinates(sx, [py, px], order=1, mode="nearest")
        vy = ndimage.map_coordinates(sy, [py, px], order=1, mode="nearest")
        # the field is a line field: keep going the way we were going
        px += sgn * vx
        py += sgn * vy
        w_ = 1 - k / 10
        acc += w_ * ndimage.map_coordinates(noise, [py, px], order=1, mode="reflect")
        wsum += w_
lic = acc / wsum
lic = (lic - lic.mean()) / (lic.std() + 1e-6)
del acc, wsum, px, py
amp = np.select([cat == 0, cat == 1, cat == 5, cat >= 2], [0.010, 0.004, 0.004, 0.009], 0.008)
amp = amp + 0.004 * sky
img = img * (1 + (amp * (1 - hot) * lic)[..., None])
log("brushwork")

# ---------------------------------------------------------------------------
# impasto: the lights stand up from the panel
# ---------------------------------------------------------------------------
lum = img @ np.array([0.2126, 0.7152, 0.0722], np.float32)
height = smoothstep(0.86, 1.0, lum) * (0.85 + 0.15 * np.tanh(lic))
height = ndimage.gaussian_filter(height, 2.0)
hy, hx = np.gradient(height)
relief = -(hx * -0.6 + hy * -0.8) * 1.5
img = img + (np.clip(relief, -0.02, 0.03))[..., None] * np.array([1.0, 0.97, 0.9], np.float32)
log("impasto")

# ---------------------------------------------------------------------------
# craquelure
# ---------------------------------------------------------------------------


def crack_net(spacing, aniso, warp, seed, width):
    r = np.random.default_rng(seed)
    gxs = np.arange(-spacing, W + spacing, spacing)
    gys = np.arange(-spacing, H + spacing, spacing / aniso)
    GX, GY = np.meshgrid(gxs, gys)
    pts = np.stack([GX.ravel() + r.uniform(-0.45, 0.45, GX.size) * spacing,
                    GY.ravel() + r.uniform(-0.45, 0.45, GX.size) * spacing / aniso], 1)
    wx = warp * ndimage.gaussian_filter(r.standard_normal((H // 4 + 1, W // 4 + 1)), 3)
    wy = warp * ndimage.gaussian_filter(r.standard_normal((H // 4 + 1, W // 4 + 1)), 3)
    wx = ndimage.zoom(wx, 4, order=1)[:H, :W] * 25
    wy = ndimage.zoom(wy, 4, order=1)[:H, :W] * 25
    q = np.stack([(xx + wx).ravel(), ((yy + wy) * 1.0).ravel()], 1)
    tree = cKDTree(pts * np.array([1.0, aniso]))
    d, _ = tree.query(q * np.array([1.0, aniso]), k=2, workers=-1)
    gap = (d[:, 1] - d[:, 0]).reshape(H, W)
    return smoothstep(width, 0.0, gap)


fine = crack_net(24.0, 1.7, 1.0, 11, 0.75)
fine *= np.clip(0.55 + 0.9 * ndimage.gaussian_filter(rng.standard_normal((H, W)), 6) * 4, 0, 1)
coarse = crack_net(120.0, 2.2, 2.0, 12, 0.95)
cracks = np.clip(fine * 0.55 + coarse * 0.9, 0, 1)
lum = img @ np.array([0.2126, 0.7152, 0.0722], np.float32)
dark_in = cracks * smoothstep(0.25, 0.75, lum) * 0.16 * (1 - hot)
light_in = cracks * smoothstep(0.25, 0.08, lum) * 0.05
img = img * (1 - dark_in[..., None]) + light_in[..., None] * np.array([0.55, 0.48, 0.36], np.float32)
# each island of paint cups a little at its edges
dist = ndimage.distance_transform_edt(cracks < 0.5)
cy_, cx_ = np.gradient(np.minimum(dist, 6.0))
img = img * (1 + 0.006 * (-(cx_ * -0.6 + cy_ * -0.8)))[..., None]
del fine, coarse, dist
log("craquelure")

# ---------------------------------------------------------------------------
# varnish, and time
# ---------------------------------------------------------------------------
old = np.array([1.0, 0.955, 0.84], np.float32)
img = img * (1 - 0.35 + 0.35 * old)
img = img * 0.975 + 0.025 * np.array([0.40, 0.29, 0.14], np.float32)
edge = np.minimum.reduce([xx / W, 1 - xx / W, yy / H, 1 - yy / H])
grime = ndimage.gaussian_filter(rng.standard_normal((H // 8, W // 8)), 6)
grime = ndimage.zoom(grime / grime.std(), 8, order=1)[:H, :W]
dark = 0.13 * (1 - smoothstep(0.0, 0.14, edge)) + 0.018 * grime
img = img * (1 - np.clip(dark, -0.03, 0.3))[..., None]
# the panel's grain shows faintly through the thinnest paint
grain = ndimage.gaussian_filter(rng.standard_normal((H, W // 6)), (0.6, 0.3))
grain = ndimage.zoom(grain, (1, 6), order=1)[:H, :W]
img = img * (1 + 0.008 * grain / grain.std())[..., None]
img += (rng.standard_normal((H, W, 1)) * 0.006).astype(np.float32)
out = (np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8)
Image.fromarray(out).save(OUT, optimize=True)
log(f"saved {OUT}")
