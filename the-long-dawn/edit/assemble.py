"""Assemble THE LONG DAWN v3: three films on one bar grid (BIBLE_V3 "REVISION 1 · LOCKED BEAT SHEETS" as amended
by the DIRECTOR'S H5 CALLS; bar maps music/v3/barmap_{A,B,C}.json; EDLs in edit/edl_v3.py).

    python3 edit/assemble.py --cut A --animatic               # -> _local_logs/animatic/A_animatic.mp4 (960x402, CRF 23)
    python3 edit/assemble.py --cut A --animatic --variant codedtowers    # A's alternate master (_alt_codedtowers)
    python3 edit/assemble.py --cut C --stills 1200,2600       # half-res review stills -> edit/cache/stills/
    python3 edit/assemble.py --coverage [--notes]             # coverage of all three cuts (and write it into NOTES_v3)
    python3 edit/assemble.py --edl                            # edit/edl/edl_{A,B,C}.json for the departments

Picture: per shot, the first take that covers every frame (else the take covering most; its gaps are slates).
Folders: renders/<stem>_<CUT>/ -> _v3 -> _v2 -> base (per take mode, see edl_v3.py). Anything missing is a SLATE:
a dark frame with the shot code, name and one line, at the right duration, so the animatic is always complete.
Frames are streamed to ffmpeg (no intermediate image sequences). Audio: the COMPOSER's master for the cut if it
exists (music/NOTES_v3.md MASTERS table, then music/out/v3/final_<cut>.wav), else the fallback master
(fallback_<cut>.wav), else silence with a soft click on every bar (accented on section starts) for sync checks.
The v2 assembler is kept as edit/assemble_v2.py.
"""
import argparse
import json
import math
import os
import re
import subprocess
import sys
import time
from multiprocessing import Pool

os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR', '1')          # the ember title layers are half-float EXR
import inspect

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'lib'))
sys.path.insert(0, os.path.join(ROOT, 'edit'))
import edl_v3 as EDL  # noqa: E402
import look  # noqa: E402
import title_scene as TS  # noqa: E402
import titles  # noqa: E402

cv2.setNumThreads(1)
RENDERS = os.path.join(ROOT, 'renders')
CACHE = os.path.join(ROOT, 'edit', 'cache')
ANIMATIC_DIR = os.path.expanduser('~/mishamisha/_local_logs/animatic')
NOTES = os.path.join(ROOT, 'edit', 'NOTES_v3.md')
FPS = 24
FULL_W, FULL_H = 1920, 804
ALT = '_alt_codedtowers'
FONTS = os.path.join(ROOT, 'assets', 'fonts')


def smooth(x):
    x = float(np.clip(x, 0.0, 1.0))
    return x * x * (3 - 2 * x)


def bar_beat(f):
    return f // 80 + 1, (f % 80) / 20.0 + 1.0


# ------------------------------------------------------------------------------------------- frame lookup
_INDEX = {}


def index(d):
    """{src frame: path} of a render folder (png preferred over jpg, as look.find_frame)."""
    if d not in _INDEX:
        out = {}
        if os.path.isdir(d):
            for n in os.listdir(d):
                m = re.match(r'f_(\d+)\.(png|jpg)$', n)
                if m:
                    k = int(m.group(1))
                    if k not in out or m.group(2) == 'png':
                        out[k] = os.path.join(d, n)
        _INDEX[d] = out
    return _INDEX[d]


def chain(take, cut, variant=None):
    st, mode = take['stem'], take['mode']
    if mode == 'exact':
        dirs = [st]
    elif mode == 'v3':
        dirs = [f'{st}_{cut}', f'{st}_v3']
    else:
        dirs = [f'{st}_{cut}', f'{st}_v3', f'{st}_v2', st]
    if variant == 'codedtowers':
        dirs = [x for d in dirs for x in (d + ALT, d)]
    return [os.path.join(RENDERS, d) for d in dirs]


_VIDN = {}


def video_frames(take):
    p = os.path.join(ROOT, take['video'])
    if p not in _VIDN:
        n = 0
        if os.path.exists(p):
            c = cv2.VideoCapture(p)
            n = int(c.get(cv2.CAP_PROP_FRAME_COUNT))
            c.release()
        _VIDN[p] = n
    return _VIDN[p]


def locate(take, cut, variant, f):
    """(path or ('video', path, index), from_alt) for cut frame f in this take, or (None, False)."""
    src = f + take['off']
    if take['mode'] == 'video':
        n = video_frames(take)
        return ((('video', os.path.join(ROOT, take['video']), src), False) if 0 <= src < n else (None, False))
    if take.get('add') and not index(os.path.join(RENDERS, take['add'])).get(src):
        return None, False                                    # an additive layer must be there as well
    for d in chain(take, cut, variant):
        p = index(d).get(src)
        if p:
            return p, d.endswith(ALT)
    return None, False


def locate_under(spec, f):
    """Metadata-only (path, folder stem) matching Ctx.under's first-present source and held-frame rules.

    Keep the renderer unchanged to preserve existing segment hashes. The parity tests exercise its actual
    reads against this lookup, including decode failure (which never tries the next folder).
    """
    if not spec:
        return None, None
    how, stem, *rest = spec
    g = rest[0] if (how == 'hold' and rest) else f
    for d in (stem, stem + '_half'):
        p = index(os.path.join(RENDERS, d)).get(g)
        if p:
            return p, d
    return None, None


def provisional_sources(take, cut, variant, f):
    """Declared provisional sources selected for a present frame, including its optional book under-layer.

    Like coverage, this uses source indexes, not decoded pixels. Missing matte/under layers keep their
    existing compositor behavior; a selected under-layer counts only when the matte path is present.
    """
    p, _ = locate(take, cut, variant, f)
    if not p:
        return ()
    sources = [] if EDL.is_final_take(take) else [take['stem']]
    if take.get('matte') and index(os.path.join(RENDERS, take['matte'])).get(f + take['off']):
        up, stem = locate_under(take.get('under'), f)
        if up and not EDL.UNDER_FINAL_ELIGIBILITY.get(stem, True):
            sources.append('under:' + stem)
    return tuple(sources)


def complete(take, cut, variant, a, b):
    got = set()
    for d in chain(take, cut, variant):
        got |= set(index(d))
    return all(k in got for k in range(a, b + 1))


def plan_shot(shot, cut, variant):
    """Pick the shot's take: the first covering every frame, else the one covering most (its gaps are slated)."""
    if shot['kind'] == 'black':
        return dict(kind='black', take=None, have=shot['f1'] - shot['f0'], alt=0)
    best = None
    for take in shot['takes']:
        if take.get('need') and not complete(take, cut, variant, *take['need']):
            continue
        have, alt = 0, 0
        for f in range(shot['f0'], shot['f1']):
            p, a = locate(take, cut, variant, f)
            if p:
                have += 1
                alt += a
        if have == shot['f1'] - shot['f0']:
            return dict(kind='take', take=take, have=have, alt=alt)
        if have and (best is None or have > best['have']):
            best = dict(kind='take', take=take, have=have, alt=alt)
    if best:
        return best
    if shot['kind'] == 'x2':
        return dict(kind='x2', take=None, have=shot['f1'] - shot['f0'], alt=0)
    if shot['kind'] == 'title':                               # X3: the stand-in sky until the plate lands
        return dict(kind='titlesky', take=None, have=shot['f1'] - shot['f0'], alt=0)
    return dict(kind='slate', take=None, have=0, alt=0)


# ---------------------------------------------------------------------------------------------- loading
class Ctx:
    """Per-process state for one cut."""

    def __init__(self, cut, variant=None, scale=0.5, clean=False):
        self.cut, self.variant, self.scale, self.clean = cut, variant, scale, clean
        self.W, self.H = int(round(FULL_W * scale)), int(round(FULL_H * scale))
        self.shots = EDL.EDL[cut]
        self.plans = [plan_shot(s, cut, variant) for s in self.shots]
        self.lines = titles.lines_v3(cut, scale)
        self.lines_nt = [ln for ln in self.lines if ln.kind != 'title']     # when the ember title plays
        self._ember_on = False
        self._slates = {}
        self._vid = {}
        self._x2 = None
        self.sec_starts = {s['f0'] for s in self.shots}

    def shot_at(self, f):
        for i, s in enumerate(self.shots):
            if s['f0'] <= f < s['f1']:
                return i, s
        raise IndexError(f)

    # -------------------------------------------------------------------------------- source frames
    def read(self, ref, crop=None, gray=False):
        if isinstance(ref, tuple):                                # ('video', path, index)
            _, path, i = ref
            v = self._vid.get(path)
            if v is None:
                v = self._vid[path] = [cv2.VideoCapture(path), -1]
            if i != v[1] + 1:
                v[0].set(cv2.CAP_PROP_POS_FRAMES, i)
            ok, img = v[0].read()
            v[1] = i
            if not ok:
                return None
        else:
            flag = cv2.IMREAD_GRAYSCALE if gray else cv2.IMREAD_COLOR
            if crop is None and ref.endswith('.jpg') and not gray:
                img = cv2.imread(ref, cv2.IMREAD_REDUCED_COLOR_2 if self.scale <= 0.5 else cv2.IMREAD_COLOR)
                if img is not None and img.shape[1] < self.W:       # a half-res jpg: read it whole instead
                    img = cv2.imread(ref, cv2.IMREAD_COLOR)
            else:
                img = cv2.imread(ref, flag)
            if img is None:
                return None
        if crop is not None:
            h, w = img.shape[:2]
            x0, y0, cw, ch = crop
            img = img[int(y0 * h):int((y0 + ch) * h), int(x0 * w):int((x0 + cw) * w)]
        if img.shape[1] != self.W or img.shape[0] != self.H:
            up = img.shape[1] < self.W
            img = cv2.resize(img, (self.W, self.H), interpolation=cv2.INTER_CUBIC if up else cv2.INTER_AREA)
        if gray:
            if img.ndim == 3:
                img = img[..., 0]
            return img.astype(np.float32) / 255.0
        return img[..., ::-1].astype(np.float32) / 255.0

    def take_frame(self, take, f):
        p, _ = locate(take, self.cut, self.variant, f)
        if not p:
            return None
        img = self.read(p, take.get('crop'))
        if img is None:
            return None
        if take.get('matte'):                                     # MAP's book layer: rgb + (1 - matte) * under
            mp = index(os.path.join(RENDERS, take['matte'])).get(f + take['off'])
            if mp:
                m = self.read(mp, gray=True)
                under = self.under(take.get('under'), f)
                if m is not None and under is not None:
                    img = np.clip(img + (1 - m)[..., None] * under, 0, 1)
        if take.get('add'):                                       # e.g. EMBERS' fire added over MAP's page
            ap = index(os.path.join(RENDERS, take['add'])).get(f + take['off'])
            layer = self.read(ap) if ap else None
            if layer is not None:
                img = np.clip(img + layer, 0, 1)
        if take.get('grade'):
            img = grade(img, take['grade'])
        return img

    # Source selection is mirrored by locate_under for metadata; test_picture_readiness pins parity.
    def under(self, spec, f):
        if not spec:
            return None
        how, stem, *rest = spec
        g = rest[0] if (how == 'hold' and rest) else f
        for d in (stem, stem + '_half'):
            p = index(os.path.join(RENDERS, d)).get(g)
            if p:
                return self.read(p)
        return None

    # ------------------------------------------------------------------------------------------ slates
    def slate(self, i, status):
        key = (i, status)
        if key not in self._slates:
            self._slates[key] = make_slate(self.shots[i], self.cut, status, self.W, self.H)
        return self._slates[key].astype(np.float32) / 255.0

    def x2(self, f, shot):
        """X2 DARK ADAPTATION, an EDIT proxy until RUN-A's star plate lands: stars return brightest first from
        bar 40 b1, then many, then the Milky Way by bar 42 b1; the ember stays one warm light among them."""
        if self._x2 is None:
            rng = np.random.default_rng(40)
            n = 2600
            x = rng.uniform(0, self.W, n)
            y = rng.uniform(0, self.H * 0.97, n)
            lum = np.minimum(0.02 * np.exp(-np.log(rng.random(n) + 1e-9)) ** 1.35, 1.6)
            order = np.argsort(-lum)
            rank = np.empty(n)
            rank[order] = np.arange(n) / n                        # 0 = brightest
            temp = rng.random(n)
            col = np.stack([0.85 + 0.3 * temp, 0.93 + 0.05 * temp, 1.12 - 0.35 * temp], 1)
            yy, xx = np.mgrid[0:self.H, 0:self.W].astype(np.float32)
            d = (yy - (0.95 * self.H - 0.55 * xx * self.H / self.W)) / self.H   # a diagonal band
            band = np.exp(-(d / 0.16) ** 2)
            cl = titles._noise((self.H // 4, self.W // 4), 7, 2.0)
            cl = cv2.resize(cl, (self.W, self.H))
            mw = (band * (0.35 + 0.65 * cl) * 0.10)[..., None] * np.array([0.78, 0.82, 1.0], np.float32)
            self._x2 = dict(x=x, y=y, lum=lum, rank=rank, col=col, mw=mw.astype(np.float32))
        X = self._x2
        t0 = shot['f0']
        vis = smooth((f - t0) / 160.0) ** 1.6                    # fraction of the catalogue back
        a = np.clip((vis - X['rank']) / 0.04, 0, 1) * np.clip((f - t0) / 12.0, 0, 1)
        acc = np.zeros((self.H, self.W, 3), np.float32)
        on = a > 0
        ix = np.clip(np.round(X['x'][on]).astype(int), 0, self.W - 1)
        iy = np.clip(np.round(X['y'][on]).astype(int), 0, self.H - 1)
        np.add.at(acc, (iy, ix), (X['col'][on] * (X['lum'][on] * a[on])[:, None]).astype(np.float32))
        acc = cv2.GaussianBlur(acc, (0, 0), 0.55)
        acc += X['mw'] * smooth((f - (t0 + 80)) / 80.0)
        # the ember: one small warm light, breathing
        # director ~19:25Z: at (960, 548) = (0.5 W, 0.68 H), where EMBERS' A10 ember rests at the cut
        ex, ey = 960 * self.scale, 548 * self.scale
        fl = 0.8 + 0.2 * np.sin(f * 0.9) * np.sin(f * 0.37 + 1)
        yy, xx = np.ogrid[0:self.H, 0:self.W]
        r2 = (xx - ex) ** 2 + (yy - ey) ** 2
        s = self.scale
        glow = (np.exp(-r2 / (2 * (1.4 * s + 0.6) ** 2)) * 1.4 + np.exp(-r2 / (2 * (7 * s) ** 2)) * 0.18) * fl
        acc += glow[..., None].astype(np.float32) * np.array([1.0, 0.45, 0.12], np.float32)
        return np.clip(acc, 0, 1)

    # --------------------------------------------------------------------------------------- the frame
    def picture(self, f):
        i, shot = self.shot_at(f)
        plan = self.plans[i]
        src = None
        if plan['kind'] == 'black':
            img = np.zeros((self.H, self.W, 3), np.float32)
            status = 'EDIT'
        elif plan['kind'] == 'x2':
            img = self.x2(f, shot)
            status = 'EDIT proxy'
        elif plan['kind'] == 'titlesky':
            img = TS.standin_sky(self.cut, f, self.W, self.H)
            status = 'EDIT proxy (stand-in sky)'
        elif plan['kind'] == 'take':
            img = self.take_frame(plan['take'], f)
            if img is None and shot['kind'] == 'title':
                img = TS.standin_sky(self.cut, f, self.W, self.H)
                status = 'EDIT proxy (stand-in sky)'
            elif img is None:
                img = self.slate(i, 'gap')
                status = 'SLATE (frame not rendered)'
            else:
                src = plan['take']
                status = src['stem'] + (' +ALT' if plan['alt'] else '')
        else:
            img = self.slate(i, 'slate')
            status = 'SLATE'
        is_slate = status.startswith('SLATE')
        if is_slate:
            draw_slate_clock(img, f, shot)
        self._ember_on = False
        if shot['kind'] == 'title':
            img, self._ember_on = self.ember(img, f)
            status += ' + ember title' if self._ember_on else ''
        if self.cut == 'A' and f >= 6456:                         # A: fade to black from bar 81 b3.8
            img = img * (1 - smooth((f - 6456) / 23.0))
        return img, shot, status, src

    def ember(self, img, f):
        """X3: add the ember title layer (renders/title_<cut>, linear light) the way the v2 title was composited:
        linear_to_srgb(soft_clip(srgb_to_linear(picture) + layer)). (img, True) when the layer frame exists."""
        p = os.path.join(RENDERS, f'title_{self.cut}', f'f_{f:05d}.exr')
        lay = cv2.imread(p, cv2.IMREAD_UNCHANGED) if os.path.exists(p) else None
        if lay is None:
            return img, False
        lay = lay[..., 2::-1].astype(np.float32)
        if lay.shape[1] != self.W or lay.shape[0] != self.H:
            lay = cv2.resize(lay, (self.W, self.H), interpolation=cv2.INTER_AREA)
        x = np.maximum(look.srgb_to_linear(np.clip(img, 0, 1)) + lay, 0.0)
        x = np.where(x <= 0.8, x, 0.8 + 0.2 * (1.0 - np.exp(-(x - 0.8) / 0.2)))           # soft_clip, knee 0.8
        return look.linear_to_srgb(x).astype(np.float32), True

    def frame(self, f):
        img, shot, status, src = self.picture(f)
        img = np.ascontiguousarray(img, np.float32)
        titles.composite_v3(img, self.lines_nt if self._ember_on else self.lines, f)
        out = (np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8)
        if not self.clean:
            burn_in(out, f, shot, status, src, self.scale)
        return out


# ------------------------------------------------------------------------------------------------ grades
def grade(img, g):
    """Per-cut grades of the shared H1 take (BIBLE §8 H1: A night, B cold moonlight, C parchment); fire stays warm."""
    warm = np.clip((img[..., 0] - img[..., 2]) * 3.0 - 0.15, 0, 1)[..., None]
    if g == 'B':
        # B's moonlit night: the H5 re-key lifted the sky, which the hands-and-tinder crop magnified into flat blue
        # shapes between the basket's bars; all but the fire goes to deep night blue. Warm hues AND white-hot cores
        # are kept (the old cool multiply turned the roar's white core blue).
        lum = (img @ np.array([0.2126, 0.7152, 0.0722], np.float32))[..., None]
        hot = np.clip((lum - 0.45) / 0.35, 0, 1)
        keep = np.maximum(warm, hot * hot * (3 - 2 * hot))
        night = lum * np.array([0.60, 0.74, 1.08], np.float32) * 0.80 + img * 0.12
        return np.clip(img * keep + night * (1 - keep), 0, 1)
    if g == 'C':
        lum = (img @ np.array([0.2126, 0.7152, 0.0722], np.float32))[..., None]
        sep = lum * np.array([1.08, 0.92, 0.70], np.float32) * 1.05 + np.array([0.030, 0.024, 0.016], np.float32)
        return np.clip(img * warm + (0.35 * img + 0.65 * sep) * (1 - warm), 0, 1)
    return img


# ------------------------------------------------------------------------------------------------ slates
BG = (13, 14, 19)


def _pil_font(name, size):
    return ImageFont.truetype(os.path.join(FONTS, name), max(6, int(size)), layout_engine=ImageFont.Layout.RAQM)


def _fit(draw, text, name, size, max_w):
    f = _pil_font(name, size)
    while draw.textlength(text, font=f) > max_w and size > 8:
        size *= 0.94
        f = _pil_font(name, size)
    return f


def make_slate(shot, cut, status, W, H):
    """A dark frame: shot code, name, one line; the lower third and the centre stay clear for the text."""
    s = W / 1920.0
    im = Image.new('RGB', (W, H), BG)
    d = ImageDraw.Draw(im)
    cx = W / 2
    b0, _ = bar_beat(shot['f0'])
    b1, _ = bar_beat(shot['f1'] - 1)
    top = f"{shot['sec']}  ·  {shot['code']}  ·  {shot['owner']}"
    f1 = _fit(d, top, 'Cinzel.ttf', 30 * s, W * 0.9)
    d.text((cx, 0.085 * H), top, font=f1, fill=(120, 124, 136), anchor='mm')
    f2 = _fit(d, shot['name'], 'Cinzel.ttf', 64 * s, W * 0.9)
    d.text((cx, 0.165 * H), shot['name'], font=f2, fill=(222, 222, 228), anchor='mm')
    f3 = _fit(d, shot['desc'], 'CormorantGaramond-Italic.ttf', 38 * s, W * 0.92)
    d.text((cx, 0.245 * H), shot['desc'], font=f3, fill=(170, 172, 182), anchor='mm')
    what = 'SLATE · not yet rendered' if status == 'slate' else 'SLATE · this frame not yet rendered'
    meta = (f"{what}   ·   bars {b0}–{b1}   ·   f {shot['f0']}–{shot['f1']}   ·   "
            f"{(shot['f1'] - shot['f0']) / FPS:.2f} s")
    f4 = _fit(d, meta, 'CormorantGaramond.ttf', 28 * s, W * 0.9)
    d.text((cx, 0.31 * H), meta, font=f4, fill=(118, 106, 88), anchor='mm')
    return np.asarray(im, np.uint8).copy()


def draw_slate_clock(img, f, shot):
    """Per-frame slate motion: a thin progress line with beat ticks, and a beat light (downbeats brighter)."""
    H, W = img.shape[:2]
    y = int(H * 0.965)
    x0, x1 = int(W * 0.05), int(W * 0.95)
    n = shot['f1'] - shot['f0']
    u = (f - shot['f0'] + 1) / n
    img[y:y + 1, x0:x1] = 0.16
    img[y - 1:y + 2, x0:x0 + int((x1 - x0) * u)] = np.array([0.62, 0.52, 0.36], np.float32)
    for k in range(0, n + 1, 20):
        xx = x0 + int((x1 - x0) * k / n)
        if x0 <= xx < x1:
            h = 5 if (shot['f0'] + k) % 80 == 0 else 2
            img[y - h:y, xx:xx + 1] = 0.35
    ph = (f % 20) / 20.0
    down = (f % 80) < 20
    lvl = max(0.0, 1.0 - ph * 2.2) * (1.0 if down else 0.45)
    cy, cxp, r = int(H * 0.91), int(W * 0.95), max(2, int(H * 0.012))
    cv2.circle(img, (cxp, cy), r, (0.2 + 0.75 * lvl, 0.18 + 0.55 * lvl, 0.14 + 0.3 * lvl), -1, cv2.LINE_AA)


def burn_in(out, f, shot, status, src, scale):
    H, W = out.shape[:2]
    bar, beat = bar_beat(f)
    fs = 0.34 * scale / 0.5
    left = f"{shot['sec']} {shot['name']}  |  {status}"
    if src is not None:
        left += f"  {f + src['off']}"
    left = left.replace('·', '/').replace('–', '-').encode('ascii', 'replace').decode()
    right = f"bar {bar} b{beat:.2f}   f {f}   {f / FPS:6.2f}s"
    out[:int(18 * scale / 0.5)] = (out[:int(18 * scale / 0.5)] * 0.5).astype(np.uint8)   # keeps the labels legible
    for txt, x in ((left, 6), (right, W - 6 - int(cv2.getTextSize(right, cv2.FONT_HERSHEY_SIMPLEX, fs, 1)[0][0]))):
        cv2.putText(out, txt, (x + 1, 13), cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(out, txt, (x, 12), cv2.FONT_HERSHEY_SIMPLEX, fs, (150, 150, 150), 1, cv2.LINE_AA)


# ------------------------------------------------------------------------------------------------- audio
def masters_table():
    """{('B', 'score'): path, ...} from the COMPOSER's music/NOTES_v3.md MASTERS table."""
    p = os.path.join(ROOT, 'music', 'NOTES_v3.md')
    out = {}
    if not os.path.exists(p):
        return out
    for line in open(p):
        # e.g. "| B score (THE VIGIL) | `music/out/v3/final_B.wav` (stems ...) | ..." -> the first .wav in cell 2
        m = re.match(r'\|\s*([ABC])\s+(score|fallback|master)\b[^|]*\|([^|]*)\|', line)
        w = re.search(r'([\w./-]+\.wav)', m.group(3)) if m else None
        if w:
            path = w.group(1)
            for cand in (path, os.path.join(ROOT, path), os.path.join(ROOT, 'music', path),
                         os.path.join(ROOT, 'music', 'out', 'v3', path)):
                if os.path.isfile(cand):
                    out[(m.group(1), 'fallback' if m.group(2) == 'fallback' else 'score')] = os.path.abspath(cand)
                    break
    return out


# Director-adopted sound masters: they take a cut's sound ahead of the MASTERS table (EDIT wires them; the table
# stays the music lanes'). B (director, 27 Sep ~23:40Z): SOUND's sound_B.wav = final_B's pre-master score + real
# recorded effects through COMPOSER's master chain (battery: level map 14/14, sync 25/25, dawn rule, -16.05 LUFS,
# TP -1.30 dBTP). Outside deliver._code_hash(): a change re-muxes, it never re-encodes picture.
ADOPTED_AUDIO = {'B': ('music/out/v3/sound_B.wav', 'SOUND master'),
                 # C: SOUND-C's sound_C.wav (director, 28 Sep ~01:10Z) fits the RETIRED 7,200-frame cut, so it is no
                 # longer C's master (EDIT-C5, 29 Sep). C5's (owner, under the director's standing delegation, 29 Sep):
                 # score pass 2 on the measured picture + the C5 effects table (SOUND-C's recordings, the first half's
                 # included) through the master chain: 5,920 frames, -16.06 LUFS, TP -1.30 dBTP; verify_c5_render
                 # 24/24, render sync 49/49, level map 23/23 with its effects, rules 3/3, notes 0, clicks 0.
                 # Alternative: pass 1 (music/out/v3/final_C5.wav, score only, pass 1's estimated timings).
                 'C': ('music/out/v3/sound_C5P2.wav', 'SOUND master (C5 pass 2)'),
                 # A (owner, 29 Sep, under the standing delegation): score PASS 2 (cut id AP2) on the restaged
                 # crossing: the walk enters at the measured set-off 5180 instead of 4880 (the line stands until then),
                 # everything else note-for-note pass 1; A's effects with strike 1's pre-cut scrape muted (syncA).
                 # -16.07 LUFS, TP -1.30 dBTP, 12,960,000 samples; verify_ap2_render 0 failures.
                 # Alternative: the 28 Sep master music/out/v3/sound_A.wav (director, ~02:20Z; COMPOSER-A2's v2 score +
                 # SOUND-C's real effects, sync 59/59), whose walk starts at 4880.
                 'A': ('music/out/v3/sound_AP2.wav', 'SOUND master (A pass 2)')}
# Every C sound file on record (sound_C, final_C, fallback_C, the MASTERS table's C rows) was made for the 7,200-frame
# cut: C takes a file only when its length is the cut's, so a stale file is never picked by its name (EDIT-C5).
LENGTH_GUARD = {'C'}


def adopted_audio(cut):
    """(path, label) of the cut's adopted sound master if the file exists, else (None, None)."""
    rel, label = ADOPTED_AUDIO.get(cut, (None, None))
    p = os.path.join(ROOT, rel) if rel else None
    return (p, label) if p and os.path.isfile(p) else (None, None)


def audio_fits(path, cut):
    """True when a guarded cut's wav is exactly the cut's length (unguarded cuts: always True)."""
    if cut not in LENGTH_GUARD:
        return True
    try:
        import soundfile as sf
        return abs(sf.info(path).duration - EDL.TOTAL[cut] / FPS) < 0.01
    except Exception:
        return False


def _audio_candidates(cut):
    tab = masters_table()
    v3 = os.path.join(ROOT, 'music', 'out', 'v3')
    return (adopted_audio(cut), (tab.get((cut, 'score')), 'COMPOSER master'),
            (os.path.join(v3, f'final_{cut}.wav'), 'COMPOSER master'), (tab.get((cut, 'fallback')), 'FALLBACK master'),
            (os.path.join(v3, f'fallback_{cut}.wav'), 'FALLBACK master'))


def audio_choice(cut):
    """(path, label, refused): the first candidate that exists and, for a guarded cut (LENGTH_GUARD), is exactly the
    cut's length, or (None, None, refused); refused lists the (path, label) that exist but were refused. The ONE walk:
    the masters (resolve_audio), their labels, the C5 gate and refresh_watch.sh's signature all read it, so no reader
    can take a file another would refuse."""
    refused = []
    for path, label in _audio_candidates(cut):
        if path and os.path.isfile(path):
            if audio_fits(path, cut):
                return path, label, refused
            refused.append((path, label))
    return None, None, refused


def audio_signature(cut):
    """refresh_watch.sh's audio term: the file a master would carry now, with its mtime, or 'click'. A refused file
    never enters it, so re-rendering a stale file cannot stand in for the cut's sound or trigger a refresh, and a file
    that starts to fit changes the term, which is the refresh that should happen. No spaces: the watcher splits its
    signature on them."""
    path, _, _ = audio_choice(cut)
    return f'{os.path.basename(path)}@{int(os.path.getmtime(path))}' if path else 'click'


def resolve_audio(cut):
    """(wav path, label): the adopted master, the composer's master, else the fallback master, else a click. A guarded
    cut (LENGTH_GUARD) refuses, loudly, any file that is not exactly its length."""
    path, label, refused = audio_choice(cut)
    for p, _ in refused:
        print(f'  AUDIO REFUSED for {cut}: {os.path.relpath(p, ROOT)} is not {EDL.TOTAL[cut]} frames long '
              f'(a sound master for another cut)', flush=True)
    if path:
        return path, f'{label} ({os.path.relpath(path, ROOT)}, {time.strftime("%d %b %H:%M", time.localtime(os.path.getmtime(path)))} local)'
    return click_track(cut), 'CLICK track (soft click each bar, accented on section starts)'


def snapshot_audio(path, cut, tries=36):
    """COMPOSER re-renders masters in place, and ffmpeg reads its input for the whole encode, so encode from a
    snapshot: wait (up to 6 min) until the file is the cut's exact length and unchanged for 5 s, copy it to
    edit/cache/, and check it did not change during the copy. The caller deletes the snapshot after the encode."""
    import shutil
    import soundfile as sf
    dst = os.path.join(CACHE, f'audio_{cut}_{os.getpid()}.wav')          # per process: builds may overlap
    want = EDL.TOTAL[cut] / FPS
    for _ in range(tries):
        try:
            st = os.stat(path)
            ok = abs(sf.info(path).duration - want) < 0.01 and time.time() - st.st_mtime > 5
            if ok:
                shutil.copyfile(path, dst + '.part')
                st2 = os.stat(path)
                if (st2.st_mtime, st2.st_size) == (st.st_mtime, st.st_size):
                    os.replace(dst + '.part', dst)
                    return dst
        except (OSError, RuntimeError):
            pass
        time.sleep(10)
    print(f'  warning: {path} never settled; encoding from it directly', flush=True)
    return path


def click_track(cut):
    """Silence with a soft click on every bar line (brighter on section starts), 48 kHz mono 16-bit."""
    os.makedirs(CACHE, exist_ok=True)
    sr = 48000
    n = EDL.TOTAL[cut] * sr // FPS
    y = np.zeros(n, np.float32)
    starts = {s['f0'] for s in EDL.EDL[cut]}
    with open(EDL.barmap_path(cut, os.path.join(ROOT, 'music', 'v3'))) as fh:
        bm = json.load(fh)
    sec_starts = {s['f0'] for s in bm['sections']}
    t = np.arange(int(0.03 * sr)) / sr
    for f in range(0, EDL.TOTAL[cut], 80):
        big = f in sec_starts
        amp = 10 ** ((-19 if big else -27) / 20)
        hz = 1568.0 if big else 1046.5
        burst = amp * np.sin(2 * np.pi * hz * t) * np.exp(-t / 0.006)
        i = f * sr // FPS
        y[i:i + len(burst)] += burst[:max(0, n - i)]
    del starts
    p = os.path.join(CACHE, f'click_{cut}.wav')
    import soundfile as sf
    sf.write(p, y, sr, subtype='PCM_16')
    return p


# ------------------------------------------------------------------------------------------------ render
_CTX = None


def _init(cut, variant, scale, clean, finish=None):
    global _CTX
    _CTX = Ctx(cut, variant, scale, clean)
    if finish:                                                # FINISH (finish/stage.py): the masters' film finish
        _CTX.picture = _finishing(_CTX, None if finish is True else finish)
    if EDL.TRANS.get(cut):                                    # EDIT transitions (EDL.TRANS), after the finish
        _CTX.picture = _transitions(_CTX, finish)


def _finishing(ctx, look=None):
    """FINISH: the last picture stage (finish/stage.py; lane FINISH). It runs after the take frame (crop, per-cut
    grade, book matte, add layers) and before titles and burn-ins. Slates, black and EDIT proxies pass untouched.
    Grain is seeded per (cut, cut frame)."""
    sys.path.insert(0, os.path.join(ROOT, 'finish'))
    import stage
    fin = stage.Finisher(look)
    picture = Ctx.picture.__get__(ctx)

    def finished(f):
        img, shot, status, src = picture(f)
        if src is not None and not status.startswith('SLATE'):
            img = fin(img, src, ctx.cut, f)
        return img, shot, status, src
    return finished


def transition_at(cut, f):
    """The EDIT transition window (EDL.TRANS) holding cut frame f, else None."""
    for t in EDL.TRANS.get(cut, ()):
        if t.get('ready', True) and t['f0'] <= f < t['f1']:
            return t
    return None


def transition_layers(t, f):
    """{'glow': path, 'keep': path} of a window's layer frames at f; None if one is missing (then: a hard cut)."""
    out = {}
    for k in ('glow', 'keep', 'cover'):
        if t.get(k):
            p = index(os.path.join(RENDERS, t[k])).get(f)
            if not p:
                return None
            out[k] = p
    return out


def _to_lin(x):
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def _to_srgb(x):
    x = np.maximum(x, 0)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * x ** (1 / 2.4) - 0.055)


import afix_comp as AFIX  # noqa: E402  (lane A-FIX's transition kinds, dispatched by _transitions)


def _swell(o, i, d, f, t, k):
    """EDIT kind 'swell' (A4 -> A5, user 28 Sep): the outgoing point's glow grows (ease-in) into the incoming's first
    frame d (the ignition disc), drifting to its centre; a soft cross-blend over t['blend'] joins the two sides; after
    the cut the disc shrinks and fades toward the flame so it condenses into it over t['f1'] - t['cut'] frames.
    Everything added is ROUND: a bloom with the disc's own azimuthal-mean profile (lifted off its black, soft limit R),
    and the incoming halo beyond r 140 px is replaced by its azimuthal mean (the render's halo is a rounded square)
    from before the cut until it has faded; the flame inside r 140 is untouched."""
    H, W = d.shape[:2]
    cut, f0, f1 = t['cut'], t['f0'], t['f1']
    ys, xs = np.mgrid[0:H, 0:W].astype(np.float32)
    dx, dy = t['disc'][0] * k, t['disc'][1] * k
    rd = np.hypot(xs - dx, ys - dy)
    ri = rd.astype(np.int32).ravel()
    cnt = np.maximum(np.bincount(ri), 1)

    def profile(img):                                         # per-channel azimuthal mean about the disc centre
        return np.stack([np.bincount(ri, img[..., c].ravel()) / cnt for c in range(3)], 1).astype(np.float32)

    def radial(prof, r):
        idx = np.arange(len(prof), dtype=np.float32)
        return np.stack([np.interp(r, idx, prof[:, c], right=0.0) for c in range(3)], -1).astype(np.float32)

    pd = np.clip(profile(d) - float(np.median(d)), 0, None)
    pd *= np.clip(1.2 - np.arange(len(pd), dtype=np.float32) / (t['R'] * k), 0, 1)[:, None]

    def disc(scale, cx, cy):                                  # the round bloom, scaled about (cx, cy)
        return radial(pd, np.hypot(xs - cx, ys - cy) / max(scale, 1e-3))

    u = min(1.0, max(0.0, (f - f0) / (cut - f0)))
    e = u ** 1.2                                                # before the cut: ease-in growth, visible from ~1/4
    cx = (t['p0'][0] + (t['c1'][0] - t['p0'][0]) * e) * k
    cy = (t['p0'][1] + (t['c1'][1] - t['p0'][1]) * e) * k
    a_img = np.maximum(o, u ** 0.5 * disc(t['s0'] + (1 - t['s0']) * e, cx, cy))
    v = min(1.0, max(0.0, (f - cut) / (f1 - cut)))
    e2 = v * v * (3 - 2 * v)                                    # after the cut: the disc condenses into the flame
    hm = (np.clip((rd - 140 * k) / (40 * k), 0, 1) * (1 - e2))[..., None]
    i2 = i * (1 - hm) + radial(profile(i), rd) * hm             # the incoming halo made round
    fx = (t['disc'][0] + (t['flame'][0] - t['disc'][0]) * e2) * k
    fy = (t['disc'][1] + (t['flame'][1] - t['disc'][1]) * e2) * k
    b_img = np.maximum(i2, (1 - e2) * disc(1 - (1 - t['s1']) * e2, fx, fy))
    b0, b1 = t['blend']
    w = min(1.0, max(0.0, (f - b0 + 0.5) / (b1 - b0)))
    w = w * w * (3 - 2 * w)
    return a_img * (1 - w) + b_img * w


# The transition kinds (EDL.TRANS), one function each, so a change to one kind re-keys only its windows
# (deliver.transition_sources keys a window on _transitions' core + its kind's function; A-FIX's on afix_comp).
# Two-sided kinds take (o, i, f, t, ctx, lay, first): first() returns the incoming's first frame, finished.
def _tk_burn(o, i, f, t, ctx, lay, first):
    return o * ctx.read(lay['keep']) + i * (1 - ctx.read(lay['cover'], gray=True)[..., None]) + ctx.read(lay['glow'])


def _tk_x1(o, i, f, t, ctx, lay, first):
    keep = ctx.read(lay['keep'], gray=True)[..., None]
    return o * keep + i * (1 - keep) + ctx.read(lay['glow'])


def _tk_dissolve(o, i, f, t, ctx, lay, first):
    a = (f - t['f0'] + 0.5) / (t['f1'] - t['f0'])
    a = a * a * (3 - 2 * a)
    return _to_srgb(_to_lin(o) * (1 - a) + _to_lin(i) * a)


def _tk_swell(o, i, f, t, ctx, lay, first):
    return _swell(o, i, first(), f, t, ctx.W / 1920.0)


def page_turn(o, i, p, tilt=8.0, radius=0.11):
    """The outgoing page o turns over, right to left, onto the incoming i at progress p (0: o exactly; 1: i exactly).

    Screen space (it bridges two cameras; the book's own leaves are book_c's 3D renders). A cylinder of radius R rolls
    along a fold tilted `tilt` degrees, so the lower corner leads as a lifting hand would. Left of the fold the page lies
    flat (o). The near half of the cylinder shows o's front bending away, foreshortened by the geometry and shaded for
    a hearth low on the left; the far half and the leaf laid back over the flat page show the paper's back: the
    incoming's own paper colour (its brightest 40%) under the light the leaf carried on its front, with its fibres and
    ink showing through, mirrored by the geometry itself. The incoming lies bare right of the curl under the soft
    shadow the lifted leaf casts right and down (shifted, never wrapped round the frame)."""
    H, W = o.shape[:2]
    R = max(2.0, radius * W)
    a = math.radians(tilt)
    ca, sa = math.cos(a), math.sin(a)
    ys, xs = np.mgrid[0:H, 0:W].astype(np.float32)
    u = xs * ca + (ys - H / 2) * sa                                  # across the fold; the fold is u == uc
    lo, hi = float(u.min()), float(u.max())
    uc = hi + 0.02 * R + ((lo - 1.2 * R) - (hi + 0.02 * R)) * float(p)   # enters past the right edge, leaves past left

    def where(us):
        """Source coordinates of the page point with across-coordinate us on each pixel's own line, and coverage."""
        d = us - u
        sx, sy = xs + d * ca, ys + d * sa
        cov = np.clip(np.minimum(np.minimum(sx, W - 1 - sx), np.minimum(sy, H - 1 - sy)) + 0.5, 0.0, 1.0)
        return sx, sy, cov

    def at(img, sx, sy):
        return cv2.remap(img, sx, sy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    lum = i @ np.array([0.2126, 0.7152, 0.0722], np.float32)
    paper = np.median(i[lum >= np.quantile(lum, 0.6)], axis=0).astype(np.float32)
    o_lum = o @ np.array([0.2126, 0.7152, 0.0722], np.float32)
    light = cv2.GaussianBlur(o_lum, (0, 0), 0.05 * W)                 # the page's own light, carried by the leaf
    light = np.clip(light / max(1e-4, float(light.mean())), 0.75, 1.15).astype(np.float32)
    grain = (o - cv2.GaussianBlur(o, (0, 0), 1.5)).astype(np.float32)   # fibres and ink, seen through from behind

    def back(sx, sy, shade):                                         # the leaf's back: paper, lit where it is
        return np.clip(paper * (shade * at(light, sx, sy))[..., None] + 0.3 * at(grain, sx, sy), 0.0, 1.0)

    out = np.where((u < uc)[..., None], o, i)
    d = u - uc
    on_cyl = (d >= 0) & (d <= R)
    th1 = np.arcsin(np.clip(d / R, 0.0, 1.0))                        # near half: 0 (flat) .. pi/2 (the ridge)
    sx1, sy1, cov1 = where(uc + R * th1)
    shade1 = np.clip(0.35 + 0.65 * (0.5 * np.sin(th1) + np.cos(th1)), 0.3, 1.1)
    m1 = (on_cyl * cov1)[..., None]
    th2 = np.pi - th1                                                # far half: pi/2 .. pi (upside down)
    sx2, sy2, cov2 = where(uc + R * th2)
    shade2 = np.clip(0.35 + 0.65 * (-0.5 * np.sin(th2) - np.cos(th2)), 0.3, 1.1)
    m2 = (on_cyl * cov2)[..., None]
    sx3, sy3, cov3 = where(2.0 * uc + np.pi * R - u)                 # the leaf laid back over the flat page
    m3 = ((u < uc) * cov3)[..., None]
    leaf = np.clip(m1[..., 0] + m2[..., 0] + m3[..., 0], 0.0, 1.0)
    if leaf.any():                                                   # its shadow, cast right and a little down
        k = max(1, int(round(0.3 * R)))
        moved = np.zeros_like(leaf)                                  # shifted, never wrapped round the frame
        moved[k // 3:, k:] = leaf[:H - k // 3, :W - k]
        sh = cv2.GaussianBlur(moved, (0, 0), 0.35 * R)
        out = out * (1.0 - 0.45 * np.clip(sh - leaf, 0.0, 1.0))[..., None]
    out = out * (1 - m1) + np.clip(at(o, sx1, sy1) * shade1[..., None], 0.0, 1.0) * m1
    out = out * (1 - m2) + back(sx2, sy2, shade2) * m2
    out = out * (1 - m3) + back(sx3, sy3, np.full_like(u, 0.97)) * m3
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def _tk_page_turn(o, i, f, t, ctx, lay, first):
    n = t['f1'] - t['f0']
    p = min(1.0, max(0.0, (f - t['f0']) / max(1, n - 1)))
    return page_turn(o, i, p * p * (3 - 2 * p), t.get('tilt', 8.0), t.get('radius', 0.11))


# One-shot kinds (no cut) take (img, src, f, t, fin, cut) and return the finished frame, or None for the plain one.
def _tk_grade(img, src, f, t, fin, cut):
    img = np.clip(img * np.asarray(t['gain'], np.float32), 0, 1)
    return fin(img, src, cut, f) if fin else img


def _tk_finish_ramp(img, src, f, t, fin, cut):
    if fin is None:
        return None
    a = (f - t['f0']) / (t['f1'] - t['f0'])
    a = a * a * (3 - 2 * a)
    return fin.ink(img, cut, f) * (1 - a) + fin.film(img, cut, f) * a


FILM_BASE = 0.004        # lib/look.py's film-base lift in linear light: a rendered black is 14/255 in sRGB, EDIT's is 0


def _tk_floor(img, src, f, t, fin, cut):
    """REVIEW (29 Sep): where a rendered (film-base) black meets a true black at a join the whole screen stepped by
    14/255 (C 80 and 1040, A 80 and 2800; all four reviewers). Over the window the film base goes from k0 to k1 of
    its removal (1 = taken out, a true black; 0 = as rendered), eased, so the floor moves imperceptibly instead."""
    img = fin(img, src, cut, f) if fin else img
    a = min(1.0, max(0.0, (f - t['f0']) / max(1, t['f1'] - 1 - t['f0'])))
    k = t['k0'] + (t['k1'] - t['k0']) * a * a * (3 - 2 * a)
    if k <= 0:
        return img
    lin = look.srgb_to_linear(np.clip(img, 0, 1))
    lin = np.maximum(lin - k * FILM_BASE * (1 - lin) / (1 - FILM_BASE), 0.0)
    return look.linear_to_srgb(lin).astype(np.float32)


def dawn_sweep(img, f, t):
    """C20's finished ink fills with daylight from the sun's side (left); no geometry or added light source.

    night_rgb scales sRGB luma into C19's measured neutral balance. The broad front restores the delivered
    luminance together with its colour. A small neutral prelight precedes it (R20's dark-side slice otherwise
    measured 2.91:1 at full size); it adds no highlight beyond the delivered luma. Absolute start/done frames let the
    incoming dissolve plate and the following one-shot window share the same grade. At done the input is returned
    exactly, preserving C20's last picture and C21's existing ink-to-film hand-off.
    """
    if f >= t['done']:
        return img
    lum = (img @ np.array([0.2126, 0.7152, 0.0722], np.float32))[..., None]
    p = np.clip((f - t['start']) / (t['done'] - t['start']), 0.0, 1.0)
    p = p * p * (3 - 2 * p)
    balance = np.asarray(t['night_rgb'], np.float32)
    night = lum * (balance + (1 - balance) * t['prelight'] * p)
    width = t['width']                                        # full soft-front width, fraction of picture width
    front = -width / 2 + (1 + width) * p
    x = np.linspace(0.0, 1.0, img.shape[1], dtype=np.float32)[None, :, None]
    a = np.clip((front - x) / width + 0.5, 0.0, 1.0)
    a = a * a * (3 - 2 * a)
    return (night * (1 - a) + img * a).astype(np.float32)


def _tk_dawn_sweep(img, src, f, t, fin, cut):
    return dawn_sweep(fin(img, src, cut, f) if fin else img, f, t)


def _tk_dawn_dissolve(o, i, f, t, ctx, lay, first):
    # Both sides are already finished. Neutralise only the incoming dawn, so C19's fires keep their own fade.
    i = dawn_sweep(i, f, t)
    a = (f - t['f0'] + 0.5) / (t['f1'] - t['f0'])
    a = a * a * (3 - 2 * a)
    return _to_srgb(_to_lin(o) * (1 - a) + _to_lin(i) * a)


TKINDS_PAIR = dict(burn=_tk_burn, x1=_tk_x1, dissolve=_tk_dissolve, swell=_tk_swell, page_turn=_tk_page_turn)
TKINDS_PAIR['dawn_dissolve'] = _tk_dawn_dissolve
TKINDS_SHOT = dict(grade=_tk_grade, finish_ramp=_tk_finish_ramp, floor=_tk_floor, dawn_sweep=_tk_dawn_sweep)
KIND_HELPERS = dict(swell=_swell, page_turn=page_turn, dawn_dissolve=dawn_sweep, dawn_sweep=dawn_sweep)


def transition_code(kind):
    """The source that decides a window of this kind (for deliver's segment keys)."""
    if kind in AFIX.KINDS:
        return inspect.getsource(AFIX)
    fn = TKINDS_PAIR.get(kind) or TKINDS_SHOT.get(kind)
    src = inspect.getsource(fn) if fn else ''
    return src + (inspect.getsource(KIND_HELPERS[kind]) if kind in KIND_HELPERS else '')


def _transitions(ctx, finish=None):
    """EDIT transitions (EDL.TRANS), installed by _init after the finish. Inside a window the frame joins the outgoing
    shot (its frames before t['cut'], then its last frame held) and the incoming shot (its first frame held until
    t['cut'], then its frames), each finished on its own (ink or film look; grain seeded per cut frame, so a held
    frame's grain still renews), by the window's kind (TKINDS_PAIR, A-FIX's afix_comp.KINDS); one-shot kinds
    (TKINDS_SHOT) work on a single shot. No Ctx method changes (deliver._code_hash); deliver.segment_key adds a
    window's spec, this core and its kind's code only to the segments it touches. Slates on either side, or a
    missing layer frame, leave the plain cut."""
    raw = Ctx.picture.__get__(ctx)
    outer = ctx.picture
    fin = None
    if finish:
        sys.path.insert(0, os.path.join(ROOT, 'finish'))
        import stage
        fin = stage.Finisher(None if finish is True else finish)

    def side(g, f):
        img, shot, status, src = raw(g)
        if status.startswith('SLATE'):
            return None, None
        return (fin(img, src, ctx.cut, f) if (fin and src is not None) else img), (shot, status, src)

    def pic(f):
        t = transition_at(ctx.cut, f)
        lay = transition_layers(t, f) if t else None
        if lay is None:
            return outer(f)
        kind = t['kind']
        if kind in TKINDS_SHOT:
            img, shot, status, src = raw(f)
            if src is None or status.startswith('SLATE'):
                return outer(f)
            img = TKINDS_SHOT[kind](img, src, f, t, fin, ctx.cut)
            if img is None:
                return outer(f)
            return np.clip(img, 0, 1).astype(np.float32), shot, f'{status} + {kind}', src
        (o, mo), (i, mi) = side(min(f, t['cut'] - 1), f), side(max(f, t['cut']), f)
        if o is None or i is None:
            return outer(f)
        shot, status, src = mo if f < t['cut'] else mi
        if kind in AFIX.KINDS:                                # lane A-FIX's comps (edit/afix_comp.py)
            img = AFIX.apply(kind, o, i, f, t)
        else:
            img = TKINDS_PAIR[kind](o, i, f, t, ctx, lay, lambda: side(t['cut'], f)[0])
        return np.clip(img, 0, 1).astype(np.float32), shot, f'{status} + {kind}', src
    return pic


def _job(f):
    return _CTX.frame(f).tobytes()


def render(cut, out_path, variant=None, scale=0.5, clean=False, workers=3, rng=None, crf=23, audio=None):
    ctx = Ctx(cut, variant, scale, clean)
    start, end = rng or (0, EDL.TOTAL[cut])
    if audio is None:
        audio, alabel = resolve_audio(cut)
    else:
        alabel = f'given ({audio})'
    snap = None
    if audio and not os.path.basename(audio).startswith('click_'):
        snap = audio = snapshot_audio(audio, cut)
        if not snap.startswith(CACHE):
            snap = None
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    tmp = out_path + '.part.mp4'
    cmd = ['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{ctx.W}x{ctx.H}',
           '-framerate', str(FPS), '-i', '-']
    if audio:
        cmd += ['-ss', f'{start / FPS:.6f}', '-t', f'{(end - start) / FPS:.6f}', '-i', audio]
    cmd += ['-map', '0:v'] + (['-map', '1:a', '-c:a', 'aac', '-b:a', '160k', '-af', 'apad'] if audio else [])
    cmd += ['-c:v', 'libx264', '-preset', 'medium', '-crf', str(crf), '-pix_fmt', 'yuv420p',
            '-colorspace', 'bt709', '-color_primaries', 'bt709', '-color_trc', 'bt709',
            '-t', f'{(end - start) / FPS:.6f}', '-movflags', '+faststart', tmp]
    print(f'{cut}: {end - start} frames -> {out_path}  [{alabel}]', flush=True)
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    t0 = time.time()
    with Pool(workers, initializer=_init, initargs=(cut, variant, scale, clean)) as pool:
        for i, buf in enumerate(pool.imap(_job, range(start, end), chunksize=16)):
            proc.stdin.write(buf)
            if i % 480 == 0:
                print(f'  {cut} frame {start + i}/{end}  {time.time() - t0:5.0f}s', flush=True)
    proc.stdin.close()
    rc = proc.wait()
    if snap:
        os.remove(snap)
    if rc != 0:
        raise SystemExit(f'ffmpeg failed for {out_path}')
    os.replace(tmp, out_path)
    print(f'wrote {out_path} ({os.path.getsize(out_path) / 1e6:.1f} MB, {time.time() - t0:.0f}s)', flush=True)
    return alabel


def stills(cut, frames, variant=None, scale=0.5, clean=False):
    ctx = Ctx(cut, variant, scale, clean)
    d = os.path.join(CACHE, 'stills')
    os.makedirs(d, exist_ok=True)
    out = []
    for f in frames:
        p = os.path.join(d, f'{cut}_{f:05d}.jpg')
        cv2.imwrite(p, ctx.frame(f)[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 90])
        out.append(p)
    return out


# ---------------------------------------------------------------------------------------------- coverage
def coverage(cut, variant=None):
    rows = []
    for shot in EDL.EDL[cut]:
        plan = plan_shot(shot, cut, variant)
        n = shot['f1'] - shot['f0']
        take = plan['take']
        if plan['kind'] == 'black':
            status, src = 'EDIT (black)', 'edit'
        elif plan['kind'] == 'x2':
            status, src = 'EDIT proxy', 'edit star field (RUN-A plate pending)'
        elif plan['kind'] == 'titlesky':
            lay = os.path.join(RENDERS, f'title_{cut}')
            status = 'EDIT proxy'
            src = (f"stand-in sky (plate pending: {', '.join(os.path.relpath(chain(t, cut, None)[0], ROOT) for t in shot['takes'][:2])})"
                   f" + ember title {'renders/title_' + cut if os.path.isdir(lay) else '(not rendered yet)'}")
        elif plan['kind'] == 'slate':
            status, src = 'SLATE', 'expects ' + ', '.join(
                (os.path.relpath(chain(t, cut, None)[0], ROOT) + (f" + renders/{t['add']}" if t.get('add') else ''))
                if t['mode'] != 'video' else t['video'] for t in shot['takes'][:2])
        else:
            full = plan['have'] == n
            status = 'RENDERED' if full else f"PARTIAL {plan['have']}/{n}"
            if plan['alt']:
                status = ('VARIANT' if full else status + ' +VARIANT') + f" ({plan['alt']} alt frames)"
            if take['mode'] == 'video':
                srcdir = take['video']
            else:
                dirs = [d for d in chain(take, cut, variant) if index(d)]
                srcdir = ', '.join(os.path.relpath(d, RENDERS) for d in dirs[:2])
            src = f"{srcdir} {shot['f0'] + take['off']}-{shot['f1'] - 1 + take['off']}"
            if take.get('note'):
                src += f" ({take['note']})"
        b0, _ = bar_beat(shot['f0'])
        b1, _ = bar_beat(shot['f1'] - 1)
        rows.append(dict(sec=shot['sec'], code=shot['code'], name=shot['name'], f0=shot['f0'], f1=shot['f1'],
                         bars=f'{b0}-{b1}', status=status, src=src, owner=shot['owner']))
    return rows


def coverage_md(variant=None):
    lines = []
    for cut in 'ABC':
        rows = coverage(cut, None)
        nf = sum(r['f1'] - r['f0'] for r in rows if r['status'].startswith(('RENDERED', 'EDIT', 'VARIANT')))
        lines.append(f"**{cut}** ({EDL.TOTAL[cut]} f): {nf} f picture ({100 * nf / EDL.TOTAL[cut]:.0f}%), "
                     f"audio: {resolve_audio_label(cut)}")
        lines.append('')
        lines.append('| sec | shot | frames | bars | status | source / expected folder |')
        lines.append('|---|---|---|---|---|---|')
        for r in rows:
            lines.append(f"| {r['sec']} | {r['code']} {r['name']} | {r['f0']}-{r['f1']} | {r['bars']} | "
                         f"{r['status']} | {r['src']} |")
        if cut == 'A':
            alt = [r for r in coverage('A', 'codedtowers') if 'VARIANT' in r['status']]
            lines.append('')
            lines.append('A with `--variant codedtowers`: ' + (
                ', '.join(f"{r['sec']} {r['status']}" for r in alt) if alt else
                'no `_alt_codedtowers` frames on the A timeline yet (embers_A3_alt_codedtowers is empty), '
                'so the ALT master is identical to MAIN.'))
        lines.append('')
    return '\n'.join(lines)


def resolve_audio_label(cut):
    path, label, _ = audio_choice(cut)
    if path:
        return f'{label} `{os.path.relpath(path, ROOT)}`'
    return 'click track (no score or fallback yet' + (' of the right length)' if cut in LENGTH_GUARD else ')')


def write_notes_coverage(md):
    """Replace the block between the COVERAGE markers in edit/NOTES_v3.md."""
    a, b = '<!-- COVERAGE:BEGIN -->', '<!-- COVERAGE:END -->'
    s = open(NOTES).read() if os.path.exists(NOTES) else ''
    stamp = time.strftime('%d %b %H:%MZ', time.gmtime())
    block = f'{a}\n_Generated by `edit/assemble.py --coverage --notes` at {stamp}._\n\n{md}\n{b}'
    if a in s and b in s:
        s = s[:s.index(a)] + block + s[s.index(b) + len(b):]
    else:
        s = s + '\n\n## COVERAGE\n' + block + '\n'
    open(NOTES, 'w').write(s)


def export_edl():
    d = os.path.join(ROOT, 'edit', 'edl')
    os.makedirs(d, exist_ok=True)
    for cut in 'ABC':
        with open(os.path.join(d, f'edl_{cut}.json'), 'w') as fh:
            json.dump(edl_doc(cut), fh, indent=1)
    return d


def edl_doc(cut):
    """The exported EDL of a cut, exactly as edit/edl/edl_<cut>.json holds it (the readiness check compares them)."""
    shots = []
    for s in EDL.EDL[cut]:
        takes = [dict(t, folders=['renders/' + os.path.relpath(x, RENDERS) for x in chain(t, cut)]   # never a
                      if t['mode'] != 'video' else [t['video']]) for t in s['takes']]     # machine path
        b0, _ = bar_beat(s['f0'])
        shots.append(dict(s, takes=takes, bars=[b0, bar_beat(s['f1'] - 1)[0]]))
    doc = dict(cut=cut, fps=FPS, frames=EDL.TOTAL[cut], shots=shots, text=titles.text_table(cut),
               transitions=EDL.TRANS.get(cut, []),
               lookup='per take: v3 = <stem>_<cut>, <stem>_v3; layered = + <stem>_v2, <stem>; '
                      'exact = <stem>; src = cut frame + off; --variant codedtowers tries <dir>'
                      '_alt_codedtowers first')
    if cut == 'C':
        doc.update(EDL.c5_export_extra())
    return json.loads(json.dumps(doc))                        # the JSON form (tuples as lists), as written


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--cut', default='A', type=str.upper, choices=['A', 'B', 'C'])
    ap.add_argument('--animatic', action='store_true')
    ap.add_argument('--variant', default=None, choices=[None, 'codedtowers'])
    ap.add_argument('--clean', action='store_true', help='no burn-ins (slates stay)')
    ap.add_argument('--out', default=None)
    ap.add_argument('--range', nargs=2, type=int, default=None)
    ap.add_argument('--workers', type=int, default=3)
    ap.add_argument('--crf', type=int, default=23)
    ap.add_argument('--audio', default=None)
    ap.add_argument('--stills', default=None)
    ap.add_argument('--coverage', action='store_true')
    ap.add_argument('--notes', action='store_true', help='with --coverage: write the table into edit/NOTES_v3.md')
    ap.add_argument('--edl', action='store_true')
    a = ap.parse_args()
    EDL.check(os.path.join(ROOT, 'music', 'v3'))
    if a.edl:
        print('wrote', export_edl())
    if a.coverage:
        md = coverage_md()
        print(md)
        if a.notes:
            write_notes_coverage(md)
    if a.stills:
        for p in stills(a.cut, [int(x) for x in a.stills.split(',')], a.variant, clean=a.clean):
            print(p)
    if a.animatic:
        name = f"{a.cut}_animatic{'_' + a.variant if a.variant else ''}{'_clean' if a.clean else ''}.mp4"
        render(a.cut, a.out or os.path.join(ANIMATIC_DIR, name), a.variant, 0.5, a.clean, a.workers,
               tuple(a.range) if a.range else None, a.crf, a.audio)
