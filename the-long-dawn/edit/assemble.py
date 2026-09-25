"""Assemble THE LONG DAWN.

EDL (global frames) -> transitions in linear light -> titles -> one unified
finish (grade, grain) -> 2.39:1 letterbox in 1920x1080 -> ffmpeg with the mix.

    python3 edit/assemble.py                 # full master
    python3 edit/assemble.py --draft         # fast half-res check
    python3 edit/assemble.py --range 1200 1500 --draft
    python3 edit/assemble.py --stills 150,480,1400,2250,2700
"""
import argparse
import os
import subprocess
import sys
from multiprocessing import Pool

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'lib'))
sys.path.insert(0, os.path.join(ROOT, 'edit'))
import look      # noqa: E402
import titles    # noqa: E402

cv2.setNumThreads(1)
W, H = look.W, look.H
OUT_W, OUT_H = 1920, 1080
TOTAL = 2808                       # 117.0 s
BAR_TOP = (OUT_H - H) // 2         # 138
LINES = titles.story_lines()


def smooth(x):
    x = float(np.clip(x, 0.0, 1.0))
    return x * x * (3 - 2 * x)


def load(dept, f):
    p = look.frame_path(os.path.join(ROOT, 'renders', dept), f)
    img = cv2.imread(p, cv2.IMREAD_COLOR) if os.path.exists(p) else None
    if img is None:
        ph = np.full((H, W, 3), 38, np.uint8)
        cv2.putText(ph, f'MISSING {dept} {f}', (60, H // 2), cv2.FONT_HERSHEY_SIMPLEX, 2.0,
                    (90, 90, 90), 3, cv2.LINE_AA)
        img = ph
    if img.shape[:2] != (H, W):
        img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
    return img[..., ::-1].astype(np.float32) / 255.0


def lin(x):
    return look.srgb_to_linear(x)


def mix(a, b, t):
    """Dissolve in linear light (light fades like light)."""
    return look.linear_to_srgb(lin(a) * (1 - t) + lin(b) * t)


def exposure(img, stops):
    return look.linear_to_srgb(lin(img) * (2.0 ** stops))


# --------------------------------------------------------------- the cut ---

def picture(f):
    if f < 300:
        img = load('hills', f)
    elif f < 340:                                  # push into the torch -> the fire's visions
        img = mix(load('hills', f), load('embers', f), smooth((f - 300) / 40))
    elif f < 1200:
        img = load('embers', f)
    elif f < 1440:
        img = load('hills', f)
    elif f < 1760:
        img = load('montage', f)
    elif f < 1912:
        img = load('globe', f)
    elif f < 1928:                                 # globe -> the plain of torches
        img = mix(load('globe', f), load('accord', f), smooth((f - 1912) / 16))
    elif f < 2240:
        img = load('accord', f)
        if f >= 2228:                              # hearth flare builds to white
            img = exposure(img, 2.2 * smooth((f - 2228) / 12))
    elif f < 2464:
        img = load('globe', f)
        if f < 2254:                               # ...and the sun is born out of it
            img = exposure(img, 2.2 * (1 - smooth((f - 2240) / 14)))
    elif f < 2496:                                 # dawn over the world -> back to the hill
        img = mix(load('globe', f), load('hills', f), smooth((f - 2464) / 32))
    else:
        img = load('hills', f)

    fade_in = smooth((f - 8) / 56)                 # from black
    fade_out = smooth((2805 - f) / 50)             # to black
    k = fade_in * fade_out
    if k < 1:
        img = look.linear_to_srgb(lin(img) * k)
    return img


# ------------------------------------------------------------ finishing ---

_GRAIN = None


def grain_bank(n=24, seed=7):
    global _GRAIN
    if _GRAIN is None:
        rng = np.random.default_rng(seed)
        bank = []
        for _ in range(n):
            g = rng.standard_normal((H, W, 1)).astype(np.float32)
            g = cv2.GaussianBlur(g, (0, 0), 0.65)[..., None]
            c = rng.standard_normal((H // 2, W // 2, 3)).astype(np.float32)
            c = cv2.resize(cv2.GaussianBlur(c, (0, 0), 0.8), (W, H))
            bank.append((g * 1.0 + c * 0.25).astype(np.float16))
        _GRAIN = bank
    return _GRAIN


def finish(img, f, grain=0.022):
    # gentle unified grade: cool the deepest shadows a hair, keep highlights warm
    lum = img.mean(axis=2, keepdims=True)
    shadow = np.clip(1 - lum * 4, 0, 1)
    img = img + shadow * np.array([-0.002, 0.0, 0.006], np.float32)
    # film grain, strongest in the mid-tones, softer in deep black and white
    if grain > 0:
        g = grain_bank()[f % 24].astype(np.float32)
        if (f // 24) % 2:
            g = g[::-1]
        amp = grain * (0.35 + 2.6 * lum * (1 - lum))
        img = img + g * amp
    return np.clip(img, 0, 1)


def frame(f, draft=False, grain=0.022):
    img = picture(f)
    img = titles.composite(img, LINES, f)
    img = finish(img, f, 0.0 if draft else grain)
    out = np.zeros((OUT_H, OUT_W, 3), np.float32)
    out[BAR_TOP:BAR_TOP + H] = img
    out8 = (out * 255 + 0.5).astype(np.uint8)
    if draft:
        out8 = cv2.resize(out8, (OUT_W // 2, OUT_H // 2), interpolation=cv2.INTER_AREA)
    return out8


def _job(args):
    f, draft, grain = args
    return frame(f, draft, grain).tobytes()


def render(out_path, start, end, draft=False, audio=None, workers=2, grain=0.022,
           crf=17, maxrate='6000k', preset='slow'):
    w, h = (OUT_W // 2, OUT_H // 2) if draft else (OUT_W, OUT_H)
    cmd = ['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
           '-s', f'{w}x{h}', '-framerate', str(look.FPS), '-i', '-']
    if audio and os.path.exists(audio):
        cmd += ['-ss', f'{start / look.FPS:.6f}', '-t', f'{(end - start) / look.FPS:.6f}', '-i', audio]
    cmd += ['-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-colorspace', 'bt709',
            '-color_primaries', 'bt709', '-color_trc', 'bt709']
    if draft:
        cmd += ['-preset', 'veryfast', '-crf', '22']
    else:
        cmd += ['-preset', preset, '-crf', str(crf), '-maxrate', maxrate,
                '-bufsize', str(int(maxrate.rstrip('k')) * 2) + 'k', '-tune', 'grain',
                '-profile:v', 'high', '-level', '4.1']
    if audio and os.path.exists(audio):
        cmd += ['-c:a', 'aac', '-b:a', '256k', '-ar', '48000', '-shortest']
    cmd += ['-movflags', '+faststart', out_path]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    jobs = [(f, draft, grain) for f in range(start, end)]
    with Pool(workers) as pool:
        for i, buf in enumerate(pool.imap(_job, jobs, chunksize=4)):
            proc.stdin.write(buf)
            if i % 120 == 0:
                print(f'  frame {start + i}/{end}', flush=True)
    proc.stdin.close()
    proc.wait()
    print('wrote', out_path)


def stills(frames, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    for f in frames:
        img = frame(f)
        cv2.imwrite(os.path.join(out_dir, f'still_{f:05d}.jpg'), img[..., ::-1],
                    [cv2.IMWRITE_JPEG_QUALITY, 93])


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--draft', action='store_true')
    ap.add_argument('--range', nargs=2, type=int, default=[0, TOTAL])
    ap.add_argument('--out', default=None)
    ap.add_argument('--audio', default=os.path.join(ROOT, 'music', 'out', 'mix.wav'))
    ap.add_argument('--workers', type=int, default=2)
    ap.add_argument('--grain', type=float, default=0.022)
    ap.add_argument('--stills', default=None)
    a = ap.parse_args()
    if a.stills:
        stills([int(x) for x in a.stills.split(',')], os.path.join(ROOT, 'out', 'stills'))
        sys.exit()
    out = a.out or os.path.join(ROOT, 'out', 'draft.mp4' if a.draft else 'the_long_dawn.mp4')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    render(out, a.range[0], a.range[1], a.draft, a.audio, a.workers, a.grain)
