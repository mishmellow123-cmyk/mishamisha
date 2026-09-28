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
import os
import re
import subprocess
import sys
import time
from multiprocessing import Pool

os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR', '1')          # the ember title layers are half-float EXR
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
                 # C (director, 28 Sep ~01:10Z): SOUND-C's sound_C.wav, every effect re-synced to the measured picture
                 # (music/sound/picture_sync_C.json); passes the battery
                 'C': ('music/out/v3/sound_C.wav', 'SOUND master'),
                 # A (director, 28 Sep ~02:20Z): COMPOSER-A2's v2 score + SOUND-C's real effects re-synced to the
                 # measured picture (incl. A-FIX's h1_A fire); sync 59/59. Re-renders are picked up by name.
                 'A': ('music/out/v3/sound_A.wav', 'SOUND master')}


def adopted_audio(cut):
    """(path, label) of the cut's adopted sound master if the file exists, else (None, None)."""
    rel, label = ADOPTED_AUDIO.get(cut, (None, None))
    p = os.path.join(ROOT, rel) if rel else None
    return (p, label) if p and os.path.isfile(p) else (None, None)


def resolve_audio(cut):
    """(wav path, label): the adopted master, the composer's master, else the fallback master, else a click."""
    tab = masters_table()
    v3 = os.path.join(ROOT, 'music', 'out', 'v3')
    ad, ad_label = adopted_audio(cut)
    for key, path, label in ((None, ad, ad_label),
                             ((cut, 'score'), tab.get((cut, 'score')), 'COMPOSER master'),
                             (None, os.path.join(v3, f'final_{cut}.wav'), 'COMPOSER master'),
                             ((cut, 'fallback'), tab.get((cut, 'fallback')), 'FALLBACK master'),
                             (None, os.path.join(v3, f'fallback_{cut}.wav'), 'FALLBACK master')):
        if path and os.path.isfile(path):
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
    bm = json.load(open(os.path.join(ROOT, 'music', 'v3', f'barmap_{cut}.json')))
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


def _transitions(ctx, finish=None):
    """EDIT transitions (EDL.TRANS), installed by _init after the finish. Inside a window the frame joins the outgoing
    shot (its frames before t['cut'], then its last frame held) and the incoming shot (its first frame held until
    t['cut'], then its frames), each finished on its own (ink or film look; grain seeded per cut frame, so a held
    frame's grain still renews), by the window's matte (x1: O * keep + I * (1 - keep) + glow) or a linear-light
    dissolve. No Ctx method changes (deliver._code_hash); deliver.segment_key adds a window's spec and frames only
    to the segments it touches. Slates on either side, or a missing layer frame, leave the plain cut."""
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
        if t['kind'] == 'grade':                              # one shot: a per-channel gain before the finish
            img, shot, status, src = raw(f)
            if src is None or status.startswith('SLATE'):
                return outer(f)
            img = np.clip(img * np.asarray(t['gain'], np.float32), 0, 1)
            return (fin(img, src, ctx.cut, f) if fin else img), shot, f'{status} + grade', src
        if t['kind'] == 'finish_ramp':                        # one shot, no cut: ink look -> film look
            img, shot, status, src = raw(f)
            if fin is None or src is None or status.startswith('SLATE'):
                return outer(f)
            a = (f - t['f0']) / (t['f1'] - t['f0'])
            a = a * a * (3 - 2 * a)
            img = fin.ink(img, ctx.cut, f) * (1 - a) + fin.film(img, ctx.cut, f) * a
            return np.clip(img, 0, 1).astype(np.float32), shot, f'{status} + finish_ramp', src
        (o, mo), (i, mi) = side(min(f, t['cut'] - 1), f), side(max(f, t['cut']), f)
        if o is None or i is None:
            return outer(f)
        shot, status, src = mo if f < t['cut'] else mi
        if t['kind'] in AFIX.KINDS:                           # lane A-FIX's comps (edit/afix_comp.py)
            img = AFIX.apply(t['kind'], o, i, f, t)
        elif t['kind'] == 'burn':
            img = (o * ctx.read(lay['keep']) + i * (1 - ctx.read(lay['cover'], gray=True)[..., None])
                   + ctx.read(lay['glow']))
        elif t['kind'] == 'x1':
            keep = ctx.read(lay['keep'], gray=True)[..., None]
            img = o * keep + i * (1 - keep) + ctx.read(lay['glow'])
        else:                                                 # dissolve
            a = (f - t['f0'] + 0.5) / (t['f1'] - t['f0'])
            a = a * a * (3 - 2 * a)
            img = _to_srgb(_to_lin(o) * (1 - a) + _to_lin(i) * a)
        return np.clip(img, 0, 1).astype(np.float32), shot, f'{status} + {t["kind"]}', src
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
    tab = masters_table()
    v3 = os.path.join(ROOT, 'music', 'out', 'v3')
    for path, label in (adopted_audio(cut), (tab.get((cut, 'score')), 'COMPOSER master'), (os.path.join(v3, f'final_{cut}.wav'),
                        'COMPOSER master'), (tab.get((cut, 'fallback')), 'FALLBACK master'),
                        (os.path.join(v3, f'fallback_{cut}.wav'), 'FALLBACK master')):
        if path and os.path.isfile(path):
            return f'{label} `{os.path.relpath(path, ROOT)}`'
    return 'click track (no score or fallback yet)'


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
        shots = []
        for s in EDL.EDL[cut]:
            takes = [dict(t, folders=[os.path.relpath(x, ROOT) for x in chain(t, cut)] if t['mode'] != 'video'
                          else [t['video']]) for t in s['takes']]
            b0, _ = bar_beat(s['f0'])
            shots.append(dict(s, takes=takes, bars=[b0, bar_beat(s['f1'] - 1)[0]]))
        txt = titles.text_table(cut)
        json.dump(dict(cut=cut, fps=FPS, frames=EDL.TOTAL[cut], shots=shots, text=txt, transitions=EDL.TRANS.get(cut, []),
                       lookup='per take: v3 = <stem>_<cut>, <stem>_v3; layered = + <stem>_v2, <stem>; '
                              'exact = <stem>; src = cut frame + off; --variant codedtowers tries <dir>'
                              '_alt_codedtowers first'),
                  open(os.path.join(d, f'edl_{cut}.json'), 'w'), indent=1)
    return d


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
