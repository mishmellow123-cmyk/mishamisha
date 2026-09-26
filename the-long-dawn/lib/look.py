"""Shared image pipeline for THE LONG DAWN.

Every shot renderer works in LINEAR-LIGHT float32 HDR (values may exceed 1.0,
fire cores should be 4-30). Call `finish()` once per frame to get a
display-referred sRGB image in [0,1], then `save_png()`.

Do NOT add film grain, letterbox bars, or titles; the edit adds those so the
whole film shares one finish.
"""
import os
import subprocess

import cv2
import numpy as np

W, H = 1920, 804          # active 2.39:1 picture area (letterbox added in edit)
FPS = 24
BPM = 72
BEAT = 20                 # frames per beat (60/72 s * 24 fps)
BAR = 80                  # frames per 4/4 bar

cv2.setNumThreads(2)


def bar_frame(bar, beat=1, sub=0.0):
    """Global frame of `bar` (1-based) and `beat` (1-based), plus fractional beats."""
    return int(round((bar - 1) * BAR + (beat - 1 + sub) * BEAT))


# ---------------------------------------------------------------- colour ---

def hexrgb(h):
    """'#FFB347' -> linear-light float32 RGB."""
    h = h.lstrip('#')
    c = np.array([int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4)], np.float32)
    return srgb_to_linear(c)


def srgb_to_linear(x):
    x = np.asarray(x, np.float32)
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4).astype(np.float32)


def linear_to_srgb(x):
    x = np.clip(np.asarray(x, np.float32), 0.0, 1.0)
    return np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(x, 1 / 2.4) - 0.055).astype(np.float32)


def blackbody(t):
    """Fire colour ramp. t in [0,1] (0=dark ember red, 1=white-hot). Returns linear RGB
    with unit-ish luminance; multiply by intensity yourself."""
    t = np.clip(np.asarray(t, np.float32), 0, 1)[..., None]
    stops = np.array([
        [0.00, 0.35, 0.02, 0.00],
        [0.30, 0.90, 0.12, 0.01],
        [0.55, 1.00, 0.36, 0.04],
        [0.78, 1.00, 0.68, 0.25],
        [1.00, 1.00, 0.95, 0.85],
    ], np.float32)
    out = np.zeros(t.shape[:-1] + (3,), np.float32)
    for i in range(len(stops) - 1):
        a, b = stops[i], stops[i + 1]
        w = np.clip((t[..., 0] - a[0]) / (b[0] - a[0]), 0, 1)[..., None]
        seg = (t[..., 0] >= a[0]) & (t[..., 0] <= b[0] + 1e-6)
        out = np.where(seg[..., None], a[1:] * (1 - w) + b[1:] * w, out)
    return out


# Palette (sRGB hex; convert with hexrgb()). Keep every shot inside this world.
PALETTE = {
    'night_zenith': '#070B1C', 'night_mid': '#131D3B', 'night_horizon': '#27335E',
    'dusk_violet': '#4A3F6E', 'dusk_amber': '#C9824F', 'dusk_rose': '#9C5B6E',
    'moonlight': '#9DB4D9', 'snow_moonlit': '#8FA3C4', 'haze_blue': '#3B4F78',
    'silhouette': '#05060B',
    'fire_core': '#FFF4DC', 'fire_hot': '#FFC56B', 'fire_mid': '#FF8A2A', 'fire_deep': '#C2410C',
    'ember': '#FF5A1F', 'smoke': '#2A2220',
    'mind_core': '#F2FAFF', 'mind_ice': '#A8DBFF', 'mind_gold': '#FFD98A',   # the new fire
    'race_red': '#FF3B1F', 'race_crimson': '#8E0F12', 'race_black': '#1A0404',
    'accord_gold': '#FFD27A', 'accord_amber': '#E8A33D', 'accord_pale': '#FFF1C9',
    'dawn_gold': '#FFD9A0', 'dawn_peach': '#F4A67A', 'dawn_sky': '#7FA7D6',
    'scarf_red': '#9E1B1B', 'stone': '#6B655C',
}


# ------------------------------------------------------------ tone + glow ---

_ACES_IN = np.array([[0.59719, 0.35458, 0.04823],
                     [0.07600, 0.90834, 0.01566],
                     [0.02840, 0.13383, 0.83777]], np.float32)
_ACES_OUT = np.array([[1.60475, -0.53108, -0.07367],
                      [-0.10208, 1.10813, -0.00605],
                      [-0.00327, -0.07276, 1.07602]], np.float32)


def tonemap(hdr, exposure=1.0):
    """ACES fitted (Hill). Bright fire desaturates gracefully toward white."""
    x = hdr.astype(np.float32) * exposure
    x = x @ _ACES_IN.T
    a = x * (x + 0.0245786) - 0.000090537
    b = x * (0.983729 * x + 0.4329510) + 0.238081
    x = (a / b) @ _ACES_OUT.T
    return np.clip(x, 0, 1)


def _blur_pyr(img, levels):
    """Returns list of progressively downsampled+blurred copies (for glow)."""
    out = []
    cur = img
    for _ in range(levels):
        cur = cv2.pyrDown(cur)
        out.append(cur)
    return out


def bloom(hdr, strength=0.08, threshold=0.8, levels=6, tint=None):
    """Multi-scale glow. strength ~0.03 subtle .. 0.25 heavy. Operates in linear HDR."""
    lum = hdr.max(axis=2, keepdims=True)
    k = np.clip((lum - threshold) / np.maximum(lum, 1e-6), 0, 1)
    bright = (hdr * k).astype(np.float32)
    h, w = hdr.shape[:2]
    acc = np.zeros_like(hdr)
    pyr = _blur_pyr(bright, levels)
    for i, p in enumerate(pyr):
        p = cv2.GaussianBlur(p, (0, 0), 1.5)
        up = cv2.resize(p, (w, h), interpolation=cv2.INTER_LINEAR)
        acc += up * (0.6 + 0.25 * i)          # wider levels weigh a little more
    acc /= levels
    if tint is not None:
        acc *= np.asarray(tint, np.float32)
    return hdr + strength * acc * levels * 0.5


def streak(hdr, strength=0.04, threshold=2.0, length=0.35, tint=(1.0, 0.75, 0.55)):
    """Horizontal anamorphic flare streaks from very bright sources. length = fraction of width."""
    lum = hdr.max(axis=2, keepdims=True)
    bright = np.where(lum > threshold, hdr, 0).astype(np.float32)
    h, w = hdr.shape[:2]
    small = cv2.resize(bright, (w // 4, h // 4), interpolation=cv2.INTER_AREA)
    klen = max(3, int(w // 4 * length)) | 1
    kernel = np.exp(-np.abs(np.linspace(-4, 4, klen)))[None, :].astype(np.float32)
    kernel /= kernel.sum()
    s = cv2.filter2D(small, -1, kernel)
    s = cv2.resize(s, (w, h), interpolation=cv2.INTER_LINEAR)
    return hdr + strength * s * np.asarray(tint, np.float32) * 6.0


def vignette(img, amount=0.25, roundness=1.0):
    h, w = img.shape[:2]
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    x = (x - w / 2) / (w / 2)
    y = (y - h / 2) / (h / 2) * roundness * (h / w) * 2.39 / 1.0 * 0.5
    r2 = x * x + y * y
    return img * (1.0 - amount * np.clip(r2, 0, 1.5))[..., None]


def finish(hdr, exposure=1.0, bloom_strength=0.08, bloom_threshold=0.8,
           streak_strength=0.0, vignette_amount=0.2, lift=0.004):
    """Standard shot finish: bloom (+ optional streak) -> vignette -> tonemap -> sRGB.
    Returns display-referred sRGB float32 in [0,1]."""
    x = hdr.astype(np.float32)
    if bloom_strength > 0:
        x = bloom(x, bloom_strength, bloom_threshold)
    if streak_strength > 0:
        x = streak(x, streak_strength)
    if vignette_amount > 0:
        x = vignette(x, vignette_amount)
    x = tonemap(x, exposure)
    x = x + lift * (1 - x)           # blacks never quite crush to zero (film base)
    return linear_to_srgb(x)


# -------------------------------------------------------------------- io ---

def save_png(path, srgb01):
    """Save display-referred [0,1] RGB as 8-bit PNG with TPDF dither (kills banding)."""
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    rng = np.random.default_rng()
    d = (rng.random(srgb01.shape, np.float32) - rng.random(srgb01.shape, np.float32)) / 255.0
    out = np.clip(np.round((srgb01 + d) * 255.0), 0, 255).astype(np.uint8)
    # atomic: write a temp file then rename, so no reader ever sees a half-written frame
    tmp = f'{path}.{os.getpid()}.tmp.png'
    cv2.imwrite(tmp, out[..., ::-1], [cv2.IMWRITE_PNG_COMPRESSION, 3])
    os.replace(tmp, path)


def frame_path(shot_dir, frame):
    """Global frame numbering: renders/<shot>/f_%05d.png"""
    return os.path.join(shot_dir, f'f_{frame:05d}.png')


def preview_mp4(shot_dir, out_mp4, start, end, scale=960):
    """Quick H.264 preview of global frames [start, end) for review."""
    subprocess.run([
        'ffmpeg', '-y', '-loglevel', 'error', '-framerate', str(FPS), '-start_number', str(start),
        '-i', os.path.join(shot_dir, 'f_%05d.png'), '-frames:v', str(end - start),
        '-vf', f'scale={scale}:-2', '-c:v', 'libx264', '-crf', '20', '-pix_fmt', 'yuv420p', out_mp4
    ], check=True)


def contact_sheet(shot_dir, frames, out_png, cols=4, thumb_w=480):
    """Grid of thumbnails of the given global frames, labelled with frame numbers."""
    thumbs = []
    for f in frames:
        img = cv2.imread(frame_path(shot_dir, f))
        if img is None:
            img = np.zeros((H, W, 3), np.uint8)
        t = cv2.resize(img, (thumb_w, int(thumb_w * H / W)), interpolation=cv2.INTER_AREA)
        cv2.putText(t, str(f), (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        thumbs.append(t)
    while len(thumbs) % cols:
        thumbs.append(np.zeros_like(thumbs[0]))
    rows = [np.hstack(thumbs[i:i + cols]) for i in range(0, len(thumbs), cols)]
    cv2.imwrite(out_png, np.vstack(rows))
