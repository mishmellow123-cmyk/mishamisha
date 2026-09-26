"""Assemble THE LONG DAWN (v2: three cuts from one picture — see BIBLE_V2.md).

EDL (v2 frames) -> per-cut layered frame lookup -> transitions in linear light -> titles
-> one unified finish (grade, grain) -> 2.39:1 letterbox in 1920x1080 -> ffmpeg with the mix.

    python3 edit/assemble.py --cut A                 # full master of cut A (Allegory)
    python3 edit/assemble.py --cut B --draft         # fast half-res check of cut B (Legend)
    python3 edit/assemble.py --cut C --range 1500 1700 --draft
    python3 edit/assemble.py --cut C --stills 480,1600,2000,2410

Frame lookup for department X in cut K: renders/X_K -> renders/X_v2 -> renders/X. Existing
departments keep their ORIGINAL (src) frame numbers; the EDL maps v2 frames onto them. The v1
edit is kept as edit/assemble_v1.py.
"""
import argparse
import os
os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR', '1')   # the ember title layer is EXR
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
import ember_title  # noqa: E402  (THE LONG DAWN formed from embers, v2 2800-2966)

cv2.setNumThreads(1)
W, H = look.W, look.H
OUT_W, OUT_H = 1920, 1080
TOTAL = 2968                       # v2: 123.667 s (the Beacon Run adds 160 frames at 1520)
SHIFT = 160                        # everything after the Beacon Run sits 160 frames later than v1
BAR_TOP = (OUT_H - H) // 2         # 138
CUT = os.environ.get('LONGDAWN_CUT', 'A').upper()
LINES = titles.story_lines(CUT)


def set_cut(cut):
    global CUT, LINES
    CUT = cut.upper()
    LINES = titles.story_lines(CUT)


def smooth(x):
    x = float(np.clip(x, 0.0, 1.0))
    return x * x * (3 - 2 * x)


def _path(dept, f):
    for d in (f'{dept}_{CUT}', f'{dept}_v2', dept):
        p = look.frame_path(os.path.join(ROOT, 'renders', d), f)
        if os.path.exists(p):
            return p
    return None


STRICT = os.environ.get('LONGDAWN_STRICT', '1') == '1'   # masters: a missing/corrupt frame is an error


def load(dept, f):
    p = _path(dept, f)
    img = cv2.imread(p, cv2.IMREAD_COLOR) if p else None
    if img is None and STRICT:
        raise FileNotFoundError(f'frame missing or unreadable: {dept}_{CUT} src {f} ({p})')
    if img is None:
        ph = np.full((H, W, 3), 38, np.uint8)
        cv2.putText(ph, f'MISSING {dept}_{CUT} {f}', (60, H // 2), cv2.FONT_HERSHEY_SIMPLEX, 2.0,
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


_YY, _XX = np.mgrid[0:H, 0:W].astype(np.float32)
_HEARTH_D = np.sqrt((_XX - W / 2) ** 2 + (_YY - H / 2) ** 2)   # the hearth sits at frame centre at the flare


def hearth_burst(img, t):
    """Light bursting outward from the hearth: a white core whose edge races to the frame corners;
    reaches full white at t=1 (v2 2399) instead of bleaching the whole picture uniformly."""
    t = float(np.clip(t, 0.0, 1.0))
    radius = 120.0 + 1500.0 * t ** 1.6
    edge = np.clip((radius - _HEARTH_D) / (60.0 + 260.0 * t), 0.0, 1.0)
    a = np.clip(edge * (0.35 + 0.65 * t) + t ** 4, 0.0, 1.0)[..., None]
    return look.linear_to_srgb(lin(img) * (1 - a) + a)


# --------------------------------------------------------------- the cut ---

def world(f):
    """THE WORLD ANSWERS (v2 1920-2087): the globe (src = f-160), or cut C's map (v2 numbering)."""
    return load('map', f) if CUT == 'C' else load('globe', f - SHIFT)


def dawn(f):
    """DAWN (v2 2400-2655): sunrise over Earth (src = f-160), or cut C's dawn in the east
    (falls back to the orbital sunrise until cut C's own dawn is rendered)."""
    if CUT == 'C' and _path('dawn', f):
        return load('dawn', f)
    return load('globe', f - SHIFT)


def picture(f):
    if f < 300:
        img = load('hills', f)
    elif f < 340:                                  # push into the torch -> the fire's visions
        img = mix(load('hills', f), load('embers', f), smooth((f - 300) / 40))
    elif f < 1200:
        img = load('embers', f)
    elif f < 1440:                                 # FIRST BEACON
        img = load('hills', f)
    elif f < 1520:                                 # the shepherd answers
        img = load('montage', f)
    elif f < 1680:                                 # THE BEACON RUN (new, v2 numbering)
        img = load('run', f)
    elif f < 1920:                                 # desert, ice, karst, city, sea
        img = load('montage', f - SHIFT)
    elif f < 2072:                                 # THE WORLD ANSWERS
        img = world(f)
    elif f < 2088:                                 # -> the plain of torches
        img = mix(world(f), load('accord', f - SHIFT), smooth((f - 2072) / 16))
    elif f < 2400:                                 # THE ACCORD
        img = load('accord', f - SHIFT)
        if f >= 2384:                              # the hearth's light bursts outward to white
            img = hearth_burst(img, (f - 2384) / 15.0)
    elif f < 2624:                                 # DAWN
        img = dawn(f)
        if f < 2418:                               # held white through the hit, then the sun is born out of it
            w = 1.0 - smooth((f - 2402) / 16)
            img = look.linear_to_srgb(lin(img) * (1 - w) + w)
    elif f < 2656:                                 # dawn -> back to the hill
        img = mix(dawn(f), load('hills', f - SHIFT), smooth((f - 2624) / 32))
    else:                                          # CODA
        img = load('hills', f - SHIFT)

    fade_in = smooth((f - (76 if CUT == 'A' else 8)) / 56)   # from black (cut A holds black for its opening card)
    fade_out = smooth((TOTAL - 3 - f) / 50)        # to black
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
    img = ember_title.composite(img, f)          # additive linear light, soft-clipped
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
        cmd += ['-af', 'apad', '-c:a', 'aac', '-b:a', '256k', '-ar', '48000', '-shortest']   # pad: picture may outrun the 117 s mix
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
    ap.add_argument('--cut', default=CUT, choices=['A', 'B', 'C', 'a', 'b', 'c'])
    ap.add_argument('--draft', action='store_true')
    ap.add_argument('--range', nargs=2, type=int, default=[0, TOTAL])
    ap.add_argument('--out', default=None)
    ap.add_argument('--audio', default=None, help='default: music/out/v2/final_AB.wav (A, B) or final_C.wav (C)')
    ap.add_argument('--workers', type=int, default=2)
    ap.add_argument('--grain', type=float, default=0.022)
    ap.add_argument('--stills', default=None)
    a = ap.parse_args()
    set_cut(a.cut)
    os.environ['LONGDAWN_CUT'] = CUT               # worker processes re-import this module
    if a.audio is None:
        a.audio = os.path.join(ROOT, 'music', 'out', 'v2', 'final_C.wav' if CUT == 'C' else 'final_AB.wav')
    if a.stills:
        stills([int(x) for x in a.stills.split(',')], os.path.join(ROOT, 'out', 'stills', CUT))
        sys.exit()
    out = a.out or os.path.join(ROOT, 'out', f'draft_{CUT}.mp4' if a.draft else f'the_long_dawn_{CUT}.mp4')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    render(out, a.range[0], a.range[1], a.draft, a.audio, a.workers, a.grain)
