"""
From rendered light to the final picture.

Each frame is linear-light EXR from Cycles. The grade adds what a lens and
film would: a soft bloom around everything bright, a filmic tone curve,
blacks lifted to the deepest blue rather than nothing, a gentle vignette, a
trace of chromatic aberration, and fine animated grain.

Titles are drawn here too, over black.
"""
import os

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

OUT_W, OUT_H = 1920, 816
GRAIN = float(os.environ.get("GRAIN", "0.005"))
FONT = "/usr/share/fonts/truetype/freefont/FreeSerifItalic.ttf"
FONT_UPRIGHT = "/usr/share/fonts/truetype/freefont/FreeSerif.ttf"


def read_exr(path):
    a = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if a is None:
        raise IOError(path)
    return np.ascontiguousarray(a[..., :3][..., ::-1]).astype(np.float32)


def bloom(img):
    """Energy-conserving-ish multi-scale glow of the bright parts."""
    h, w = img.shape[:2]
    bright = np.maximum(img - 0.45, 0.0)
    out = np.zeros_like(img)
    for scale, weight in ((2, 0.20), (4, 0.16), (8, 0.12), (16, 0.08)):
        small = cv2.resize(bright, (w // scale, h // scale), interpolation=cv2.INTER_AREA)
        small = cv2.GaussianBlur(small, (0, 0), 2.0)
        out += weight * cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    return img + out


def aces(x):
    x = np.maximum(x, 0.0)
    return np.clip((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), 0.0, 1.0)


def to_srgb(x):
    return np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(np.clip(x, 0, 1), 1 / 2.4) - 0.055)


_VIG = {}


def vignette(h, w):
    key = (h, w)
    if key not in _VIG:
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        r = np.hypot((xx - w / 2) / (w / 2), (yy - h / 2) / (h / 2) * 0.62)
        _VIG[key] = (1.0 - 0.28 * np.clip(r - 0.35, 0, None) ** 1.6)[..., None]
    return _VIG[key]


def chromatic(img, amount=0.9):
    """Shift red outward and blue inward, a fraction of a pixel at the edges."""
    h, w = img.shape[:2]
    out = img.copy()
    for ch, k in ((0, amount), (2, -amount)):
        s = 1.0 + k / (w / 2)
        M = np.float32([[s, 0, (1 - s) * w / 2], [0, s, (1 - s) * h / 2]])
        out[..., ch] = cv2.warpAffine(img[..., ch], M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return out


def grade(lin, frame_index, exposure=1.0, fade=1.0):
    """Linear EXR (any size) -> uint8 sRGB at the output size."""
    x = bloom(lin * exposure)
    x = aces(x)
    x = to_srgb(x)
    x = cv2.resize(x, (OUT_W, OUT_H), interpolation=cv2.INTER_LANCZOS4)
    x = chromatic(x)
    x = x * vignette(OUT_H, OUT_W)
    # blacks lift to the deepest blue, highlights keep a little warmth
    lum = x @ np.array([0.2126, 0.7152, 0.0722], np.float32)
    x = x + (1 - lum[..., None]) * np.array([0.006, 0.008, 0.016], np.float32)
    x = x * fade
    rng = np.random.default_rng(10007 + frame_index)
    g = rng.standard_normal((OUT_H // 2, OUT_W // 2)).astype(np.float32)
    g = cv2.resize(g, (OUT_W, OUT_H), interpolation=cv2.INTER_LINEAR)
    x = x + (GRAIN * (0.4 + 0.6 * np.sqrt(np.clip(lum, 0, 1))) * g)[..., None]
    return (np.clip(x, 0, 1) * 255 + 0.5).astype(np.uint8)


def title_frame(lines, alpha, frame_index):
    """Text over black. lines: list of (text, size, font, y_center, opacity)."""
    im = Image.new("RGB", (OUT_W, OUT_H), (2, 3, 6))
    d = ImageDraw.Draw(im)
    for text, size, font_path, yc, op in lines:
        f = ImageFont.truetype(font_path, size)
        tw = d.textlength(text, font=f)
        a = max(0.0, min(1.0, alpha * op))
        col = tuple(int(c * a + b * (1 - a)) for c, b in zip((236, 226, 208), (2, 3, 6)))
        d.text(((OUT_W - tw) / 2, yc - size * 0.62), text, font=f, fill=col)
    x = np.asarray(im).astype(np.float32) / 255.0
    rng = np.random.default_rng(20011 + frame_index)
    g = cv2.resize(rng.standard_normal((OUT_H // 2, OUT_W // 2)).astype(np.float32), (OUT_W, OUT_H))
    x = x + 0.010 * g[..., None]
    return (np.clip(x, 0, 1) * 255 + 0.5).astype(np.uint8)
