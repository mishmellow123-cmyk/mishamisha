"""BURN-C (28 Sep): REAL FILMED burn elements composited onto C's rendered pages (the user: "the paper burning
should look far more realistic"; the footage route first).

The element: a sheet of black paper burning in front of a green backdrop (Pexels, PureRaw 8828893, 4K 25 fps; see
assets/burn_footage/LICENSES.md). Keyed per frame into
    hole   what is gone (the green shows): a RUNNING UNION over the clip (a burn never un-burns), soft at the edge
    flame  the real flames and the glowing rim, as light (linear), the backdrop subtracted, colour rebuilt where the
           green clipped it
    dist   the distance from the hole's edge into the paper (for the char band and the scorch, whose colour ramp is
           measured from a lit sheet burning: Pixabay 189188)
and laid onto the page by the page's own geometry: every screen pixel is ray-cast (the camera of book_c.cam_letters,
pure numpy, no numba) onto the top leaf, and the leaf's world (x, y) indexes the footage (its birth point pinned to
the fire point Fw, footage-up = the far side of the page). Retimed so the footage's birth lands on the render's
(C 841).

    python3 ftburn.py geo                     # check: Fw on screen per frame
    python3 ftburn.py plate                   # the backdrop plate (once; assets/burn_footage/_plate_8828893.npz)
    python3 ftburn.py c5 --frames 841-1039 --out DIR [--layers]
Heavy? No: numpy + cv2 on one frame at a time. SAFE MODE: run it through renderq.
"""
import argparse
import math
import os
import subprocess
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TLD = os.path.abspath(os.path.join(HERE, '..', '..'))
FF = os.path.expanduser('~/.venvs/longdawn/lib/python3.12/site-packages/imageio_ffmpeg/binaries/'
                        'ffmpeg-macos-aarch64-v7.1')
FOOT = os.path.join(TLD, 'assets', 'burn_footage')
FPS = 24
W, H = 1920, 804


def smooth(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3 - 2 * x)


def to_lin(c):
    c = np.asarray(c, np.float32)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4).astype(np.float32)


def to_srgb(c):
    c = np.clip(c, 0.0, None).astype(np.float32)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * np.power(c, 1 / 2.4) - 0.055).clip(0, 1).astype(np.float32)


# ============================================================ the book's geometry (a numpy copy of book.py) ===

class Leaf:
    """book.Book's top leaf, right side (TR thick), without numba: page (u, v) cm -> world, and z(x)."""

    def __init__(self, PW=20.0, PH=29.0, TL=0.66, TR=2.74, bt=0.4):
        self.PW, self.PH, self.TL, self.TR, self.bt = PW, PH, TL, TR, bt
        self.zg = bt + 0.5 * min(TL, TR) + 0.3
        self.xr = 2.4 + 0.6 * max(TL, TR)
        self.dome = 0.35
        self.a_tab = np.arange(0.0, PW + 2.0, 0.0025)
        z = self.top(self.a_tab)
        self.uR = np.concatenate([[0], np.cumsum(np.hypot(np.diff(self.a_tab), np.diff(z)))])

    def top(self, a):
        a = np.asarray(a, np.float64)
        zt = self.bt + self.TR
        q = 1.0 - a / self.xr
        r = np.sqrt(np.clip(1.0 - q * q, 0.0, None))
        r = r * (0.82 + 0.18 * r)
        inner = self.zg + (zt + self.dome - self.zg) * r
        s = (a - self.xr) / (0.42 * self.PW)
        outer = zt + self.dome * np.exp(-s * s)
        return np.where(a < self.xr, inner, outer)

    def page_to_world(self, u, v):
        a = np.interp(u, self.uR, self.a_tab)
        return np.array([a, 0.5 * self.PH - v, float(self.top(a))])


def keyed(keys, t):
    ts = [k[0] for k in keys]
    if t <= ts[0]:
        return np.asarray(keys[0][1], np.float64)
    if t >= ts[-1]:
        return np.asarray(keys[-1][1], np.float64)
    i = max(j for j in range(len(ts) - 1) if ts[j] <= t)
    u = float(smooth((t - ts[i]) / (ts[i + 1] - ts[i])))
    return np.asarray(keys[i][1], np.float64) * (1 - u) + np.asarray(keys[i + 1][1], np.float64) * u


class Cam:
    """book.Cam: pinhole at pos looking at target, hfov degrees, z up."""

    def __init__(self, pos, target, hfov, W, H):
        self.pos = np.asarray(pos, np.float64)
        f = np.asarray(target, np.float64) - self.pos
        f /= np.linalg.norm(f)
        r = np.cross(f, np.array([0.0, 0.0, 1.0]))
        r /= np.linalg.norm(r)
        u = np.cross(r, f)
        self.f, self.r, self.u, self.W, self.H = f, r, u, W, H
        self.F = (W / 2.0) / math.tan(math.radians(hfov) / 2.0)

    def project(self, P):
        d = np.asarray(P, np.float64) - self.pos
        xc, yc, zc = d @ self.r, d @ self.u, d @ self.f
        zc = np.maximum(zc, 1e-6)
        return np.stack([self.W / 2.0 + self.F * xc / zc, self.H / 2.0 - self.F * yc / zc], -1)

    def rays(self):
        ys, xs = np.mgrid[0:self.H, 0:self.W].astype(np.float64)
        xc = (xs + 0.5 - self.W / 2.0) / self.F
        yc = -(ys + 0.5 - self.H / 2.0) / self.F
        return (self.f[None, None] + xc[..., None] * self.r[None, None] + yc[..., None] * self.u[None, None])


LEAF = Leaf()
FW = LEAF.page_to_world(10.0, 14.2)            # book_c.letters: Kindling F = (10.0, 14.2) on the right page
PLATE = LEAF.page_to_world(9.8, 9.4)


def cam_letters(f):
    """book_c.cam_letters at C frame f (shot seconds from 560)."""
    t = (f - 560) / FPS
    keys_t = [(0.0, PLATE + [0.3, 0.2, 0]), (0.6, PLATE + [0.3, 0.2, 0]), (2.8, FW + [0.0, 1.0, 0]),
              (20.0, FW + [0.0, 0.2, 0])]
    keys_p = [(0.0, PLATE + [-0.2, -18.4, 16.9]), (0.6, PLATE + [-0.2, -18.4, 16.9]), (2.8, FW + [-0.5, -17.0, 17.5]),
              (20.0, FW + [-0.3, -11.5, 12.0])]
    return Cam(keyed(keys_p, t), keyed(keys_t, t), 38.0, W, H)


def screen_to_leaf(cam, iters=3):
    """Every screen pixel ray-cast onto the top leaf (z = top(x)); world (x, y, z) per pixel."""
    D = cam.rays()
    z = np.full(D.shape[:2], FW[2])
    for _ in range(iters):
        s = (z - cam.pos[2]) / np.minimum(D[..., 2], -1e-6)
        X = cam.pos[0] + s * D[..., 0]
        z = LEAF.top(np.clip(X, 0.0, None))
    s = (z - cam.pos[2]) / np.minimum(D[..., 2], -1e-6)
    return np.stack([cam.pos[0] + s * D[..., 0], cam.pos[1] + s * D[..., 1], z], -1)


def cmd_geo(a):
    for f in (800, 840, 841, 860, 900, 960, 1000, 1039):
        c = cam_letters(f)
        print(f, 'Fw ->', np.round(c.project(FW), 1), 'dist %.2f' % np.linalg.norm(FW - c.pos))




# ================================================================== the footage ===

CLIP = os.path.join(FOOT, 'pexels_8828893_4k.mp4')
CW, CH, CFPS = 3840, 2160, 25.0


class Reader:
    """Sequential frames of a clip (optionally cropped: x0, y0, w, h in clip px), as uint8 RGB, one at a time."""

    def __init__(self, path, crop=None, start=0):
        self.crop = crop
        vf = []
        if crop:
            x0, y0, w, h = crop
            vf.append('crop=%d:%d:%d:%d' % (w, h, x0, y0))
            self.w, self.h = w, h
        else:
            self.w, self.h = CW, CH
        args = [FF, '-loglevel', 'error']
        if start:
            args += ['-ss', '%.4f' % (start / CFPS)]
        args += ['-i', path]
        if vf:
            args += ['-vf', ','.join(vf)]
        args += ['-f', 'rawvideo', '-pix_fmt', 'rgb24', '-']
        self.p = subprocess.Popen(args, stdout=subprocess.PIPE, bufsize=self.w * self.h * 3)
        self.n = start

    def read(self):
        b = self.p.stdout.read(self.w * self.h * 3)
        if len(b) < self.w * self.h * 3:
            return None
        self.n += 1
        return np.frombuffer(b, np.uint8).reshape(self.h, self.w, 3)

    def close(self):
        self.p.stdout.close()
        self.p.kill()


def cmd_probe(a):
    """The birth point (first hole pixels) and colour stats: the backdrop, the flames over black paper."""
    r = Reader(CLIP, crop=(1920, 1080, 1280, 1000))
    acc = None
    for k in range(70):
        im = r.read().astype(np.float32) / 255
        hole = (im[..., 1] - im[..., 0] > 0.15) & (im[..., 1] > 0.3)
        acc = hole if acc is None else (acc | hole)
        if 18 <= k <= 30 or k in (40, 50, 60):
            ys, xs = np.nonzero(acc)
            fl = (im[..., 0] > 0.35) & (~acc)
            st = ''
            if fl.sum() > 50:
                px = im[fl]
                o = np.argsort(px[:, 0])
                q = px[o[len(o) // 2]], px[o[int(len(o) * 0.9)]], px[o[-1]]
                st = ' flame over black (median/p90/max R): ' + ' '.join('(%.2f %.2f %.2f)' % tuple(v) for v in q)
            if len(xs):
                print(k, 'n=%d c=(%.0f, %.0f)' % (len(xs), 1920 + xs.mean(), 1080 + ys.mean()), st)
    r.close()
    r = Reader(CLIP, start=820)
    im = r.read().astype(np.float32) / 255
    r.close()
    hole = (im[..., 1] - im[..., 0] > 0.15) & (im[..., 1] > 0.3)
    for (x0, y0) in ((0.1, 0.1), (0.5, 0.1), (0.9, 0.1), (0.5, 0.5), (0.9, 0.5), (0.5, 0.9), (0.9, 0.9)):
        x, y = int(x0 * CW), int(y0 * CH)
        blk = im[y - 20:y + 20, x - 20:x + 20]
        hb = hole[y - 20:y + 20, x - 20:x + 20]
        if hb.mean() > 0.9:
            print('backdrop at', (x0, y0), np.round(blk.reshape(-1, 3).mean(0), 3))


# ============================================================ key, char, warp, comp ===

BIRTH = (2545.0, 1425.0)       # the clip's first hole pixels (4K px, clip frame 18 = 0.72 s)
BIRTH_N = 18
ROI = (1600, 560, 1800, 1200)  # the part of the clip C5's screen ever sees (x0, y0, w, h at 4K)
K_CM = 0.0106                  # page cm per clip px (4K): the clip's sheet is ~41 cm wide on the book's page
F_BIRTH = 841                  # the C frame the render's hole is born at (book_C_matte), and SOUND-C's sync

# the char band, measured off a lit sheet burning (Pixabay 189188: keep = pixel / clean paper, linear, by distance
# from the hole) and scaled to the book's thinner page (the band ~1 cm): (cm, keep_r, keep_g, keep_b)
CHAR = np.array([
    (0.00, 0.05, 0.04, 0.035),
    (0.03, 0.10, 0.08, 0.07),
    (0.09, 0.20, 0.16, 0.15),
    (0.17, 0.33, 0.25, 0.23),
    (0.27, 0.52, 0.36, 0.30),
    (0.37, 0.68, 0.48, 0.39),
    (0.48, 0.86, 0.73, 0.61),
    (0.62, 1.00, 1.00, 1.00)], np.float32)


def char_keep(dcm):
    out = np.empty(dcm.shape + (3,), np.float32)
    for c in range(3):
        out[..., c] = np.interp(dcm, CHAR[:, 0], CHAR[:, c + 1])
    return out


def backdrop_fit(im):
    """A smooth (quadratic) fit of the green backdrop over the ROI, per channel (sRGB), from the keyed pixels."""
    h, w = im.shape[:2]
    g = im[::8, ::8]
    m = (g[..., 1] - g[..., 0] > 0.3) & (g[..., 1] > 0.6)
    ys, xs = np.nonzero(m)
    X = xs / (w / 8.0) - 0.5
    Y = ys / (h / 8.0) - 0.5
    A = np.stack([np.ones_like(X), X, Y, X * X, X * Y, Y * Y], -1)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    xx = xx / w - 0.5
    yy = yy / h - 0.5
    B = np.stack([np.ones_like(xx), xx, yy, xx * xx, xx * yy, yy * yy], -1)
    out = np.empty((h, w, 3), np.float32)
    for c in range(3):
        coef, *_ = np.linalg.lstsq(A, g[ys, xs, c], rcond=None)
        out[..., c] = B @ coef.astype(np.float32)
    return out


class FootageBurn:
    """The clip keyed frame by frame, in order (a running union for the hole)."""

    def __init__(self, clip=CLIP, roi=ROI):
        self.roi = roi
        x0, y0, w, h = roi
        # the backdrop from a late frame (the sheet is almost gone at 33 s)
        r = Reader(clip, crop=roi, start=825)
        late = r.read().astype(np.float32) / 255
        r.close()
        self.plate_srgb = backdrop_fit(late)
        self.plate_lin = to_lin(self.plate_srgb)
        self.reader = Reader(clip, crop=roi)
        self.state = np.zeros((h, w), np.float32)
        self.hist = []                 # the union a few clip frames back (the front's speed)
        self.cur = None
        self.n = -1
        rng = np.random.default_rng(7)
        nz = rng.standard_normal((h // 24 + 2, w // 24 + 2)).astype(np.float32)
        nz = cv2.GaussianBlur(cv2.resize(nz, (w, h), interpolation=cv2.INTER_CUBIC), (0, 0), 6)
        nz2 = cv2.GaussianBlur(cv2.resize(rng.standard_normal((h // 6 + 2, w // 6 + 2)).astype(np.float32), (w, h),
                                          interpolation=cv2.INTER_CUBIC), (0, 0), 2)
        self.noise = (nz / (nz.std() + 1e-6) * 0.8 + nz2 / (nz2.std() + 1e-6) * 0.2).astype(np.float32)

    def advance(self, n):
        """Read clip frames up to index n, folding each into the hole union; keep the last one."""
        while self.n < n:
            im = self.reader.read()
            if im is None:
                break
            self.n += 1
            im = im.astype(np.float32) / 255
            r, g, b = im[..., 0], im[..., 1], im[..., 2]
            # the backdrop, also where a flame in front washes it pale: bright in blue and never redder than green
            # (the sheet is black; a flame over the sheet is orange to yellow-white, always R >= G)
            a = smooth((b - 0.35) / 0.2) * smooth((g - r + 0.03) / 0.08)
            # a flame thick enough in front of the backdrop washes it warm-white (R > G): white (all channels high)
            # next to the hole is that wash, never flame over the black sheet (orange there, blue low)
            near = cv2.dilate((self.state > 0.5).astype(np.uint8), np.ones((41, 41), np.uint8)).astype(np.float32)
            a = np.maximum(a, smooth((im.min(-1) - 0.62) / 0.15) * near)
            a = cv2.morphologyEx(a.astype(np.float32), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
            a = cv2.GaussianBlur(a, (0, 0), 0.8)
            np.maximum(self.state, a, out=self.state)
            self.hist.append(self.state.copy())
            if len(self.hist) > 5:
                self.hist.pop(0)
            self.cur = im
        return self.layers()

    def layers(self):
        im = self.cur
        lin = to_lin(im)
        hole = self.state
        # the flames and the glowing rim: light over what is behind (backdrop where the hole is, black paper else)
        hb = cv2.GaussianBlur(hole, (0, 0), 3.0)
        ex = np.clip(lin - hb[..., None] * self.plate_lin - 0.006, 0, None)
        # over the black sheet the flame is the pixel itself (despilled: a flame is never greener or bluer than red)
        ex[..., 1] = np.minimum(ex[..., 1], ex[..., 0])
        ex[..., 2] = np.minimum(ex[..., 2], ex[..., 1])
        # over the hole the camera's highlight roll-off makes backdrop + flame far from additive (the flame washes
        # the green pale and reads 3x brighter than over the sheet), so the flame there is taken from how far the
        # pixel has left the backdrop's green (0 = clean backdrop, 1 = no green left), at the brightness the same
        # flames have over the black sheet, and only the white cores go hotter
        gr = self.plate_srgb[..., 1] - self.plate_srgb[..., 0]
        fl = np.clip(1.0 - (im[..., 1] - im[..., 0]) / np.maximum(gr, 0.1), 0, 1)
        core = np.clip((im.min(-1) - 0.86) / 0.12, 0, 1)
        I = fl ** 2 * (0.3 * core)
        ramp = np.stack([I, 0.75 * I ** 1.6, 0.35 * I ** 3.0], -1)
        wgt = smooth(hb * 1.6)[..., None]
        # the sheet-side flame and rim, and their glow carried a little way over the edge into the hole (so the
        # edge never shows as a step in the fire's light)
        # (warm light only: the sheet's torn fibres catch the backdrop as white specks, which are not fire)
        warm = np.clip((ex[..., 0] - 1.3 * ex[..., 2]) / np.maximum(ex[..., 0], 1e-4), 0, 1)[..., None]
        side = cv2.GaussianBlur(ex * warm, (0, 0), 2.0) * (1 - wgt)    # (soften the clip's h264 blocks)
        ext = cv2.GaussianBlur(cv2.dilate(side, np.ones((5, 5), np.uint8)), (0, 0), 3)
        # (the over-hole cores in `ramp` came out as h264-blocky blobs: the tongues (a) are the fire over the hole now)
        flame = side + (0.0 * ramp + 0.4 * ext) * wgt
        # distance from the hole's edge into the paper (cm), for the char band
        paper = (hole < 0.5).astype(np.uint8)
        d = cv2.distanceTransform(paper, cv2.DIST_L2, 5).astype(np.float32) * K_CM
        return hole, flame.astype(np.float32), d

    def speed(self):
        """How fast the front moves (0..1): the band the union swept over the last 4 clip frames, spread to the front."""
        old = self.hist[0] if self.hist else self.state
        return np.clip(cv2.GaussianBlur(self.state - old, (0, 0), 10) / 0.22, 0, 1).astype(np.float32)


# ======================================================= (a) flame tongues along the front ===

# real flames on black: the first seconds of two PureRaw clips, before the hole shows the green (flames rising off
# the sheet's burning lower edge, the frame's bottom = their base); (file, first frame, last frame + 1)
FLAME_BANK = [('pexels_8828898_1080.mp4', 24, 96), ('pexels_8828892_1080.mp4', 36, 84)]
BW, BH = 640, 360
SU = 3.5                        # strip px per clip px of the page (a tongue ~30-50 screen px wide)
H_MIN, H_MAX = 8.0, 125.0       # tongue height on screen (px): slow edge -> fast edge


class FlameBank:
    """Short flame clips at 640x360 (uint8, ~55 MB each), played in loops with a 10-frame crossfade."""

    def __init__(self):
        self.clips = []
        for name, a, b in FLAME_BANK:
            p = subprocess.Popen([FF, '-loglevel', 'error', '-ss', '%.4f' % (a / CFPS), '-i', os.path.join(FOOT, name),
                                  '-frames:v', str(b - a), '-vf', 'scale=%d:%d' % (BW, BH), '-f', 'rawvideo',
                                  '-pix_fmt', 'rgb24', '-'], stdout=subprocess.PIPE)
            raw = p.stdout.read()
            p.wait()
            arr = np.frombuffer(raw, np.uint8).reshape(-1, BH, BW, 3).copy()
            g = arr[..., 1].astype(np.int16) - arr[..., 0].astype(np.int16)
            arr[g > 10] = 0                                    # any green (the hole starting): not flame
            self.clips.append(arr)
        self.lut = to_lin(np.arange(256, dtype=np.float32) / 255.0)

    def frame(self, ci, k):
        arr = self.clips[ci]
        n = len(arr)
        K = 10
        L = n - K
        i = int(k) % L
        a = self.lut[arr[i]]
        if i < K:
            w = (i + 1.0) / (K + 1.0)
            a = a * w + self.lut[arr[L + i]] * (1 - w)
        return np.clip(a - 0.012, 0, None)      # the clip's black level (else its noise stacks into faint streaks)


def flame_tongues(hs, sp, fx, fy, bank, f):
    """Real flame tongues stood on the hole's edge, rising screen-up: each edge pixel carries one column of a flame
    strip (tile = one clip at its own phase), the column chosen by a page-fixed coordinate (so a tongue stays put
    on the paper as the edge moves), its height and brightness set by how fast the front moves there."""
    out = np.zeros((H * W, 3), np.float32)
    cs, _ = cv2.findContours((hs > 0.5).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    cs = [c[:, 0, :] for c in cs if len(c) > 12]
    if not cs:
        return out.reshape(H, W, 3)
    # the edge's direction at each point (a vertical run would stack its columns into one tall streak: there the
    # tongues stay short, hugging the edge)
    # each tongue's base is the edge's height smoothed along the contour (+-12 px), so neighbouring columns share a
    # baseline and a flame stays whole (independent 1-px columns on a slanted edge shear it into slivers); steep
    # runs get no tongues (only the rim), there the columns would stack into streaks
    xs_, ys_, flat_ = [], [], []
    box = np.ones(25) / 25.0
    for c in cs:
        d = np.roll(c, -4, 0) - np.roll(c, 4, 0)
        ysm = np.convolve(np.concatenate([c[-12:, 1], c[:, 1], c[:12, 1]]).astype(np.float64), box, 'valid')
        xs_.append(c[:, 0])
        ys_.append(np.round(ysm).astype(np.int64))
        flat_.append(np.abs(d[:, 0]) / (np.hypot(d[:, 0], d[:, 1]) + 1e-6))
    x, y, flat = np.concatenate(xs_), np.concatenate(ys_), np.concatenate(flat_)
    ok = (x > 2) & (x < W - 3) & (y > 2) & (y < H - 3) & (flat > 0.45)
    x, y, flat = x[ok], y[ok], flat[ok]
    if len(x) == 0:
        return out.reshape(H, W, 3)
    u = (fx[y, x] + 0.7 * fy[y, x]) * SU
    spd = sp[y, x]
    wob = 1.0 + 0.25 * np.sin(u * 0.013 + 1.3) * np.sin(u * 0.0071 + f * 0.05)
    lie = np.clip((flat - 0.45) / 0.35, 0.0, 1.0) ** 0.7
    h = (H_MIN + (H_MAX - H_MIN) * spd ** 0.7) * wob * lie
    amp = (0.2 + 0.8 * spd ** 0.5) * (0.55 + 0.45 * lie)
    # overlapping tiles (stride BW - M), crossfaded over M strip px, so no tongue is cut at a tile's seam
    M = 120.0
    stride = BW - M
    t0 = np.floor(u / stride).astype(np.int64)
    c0 = u - t0 * stride
    w0 = np.clip(c0 / M, 0, 1)
    tiles = [(t0, c0, np.sqrt(w0)), (t0 - 1, c0 + stride, np.sqrt(1 - w0))]
    kmax = int(np.ceil(h.max())) + 1
    ks = np.arange(kmax, dtype=np.float32)[None, :]
    for tt, cc, ww in tiles:
        use = ww > 0.02
        for t in np.unique(tt[use]):
            m = use & (tt == t)
            ci = int(t) % len(bank.clips)
            n = len(bank.clips[ci])
            phase = (int(t) * 7919) % n
            S = bank.frame(ci, (f - F_BIRTH) * CFPS / FPS + phase)
            hh = h[m][:, None]
            valid = ks < hh
            yy = (y[m][:, None] - ks).astype(np.int64)
            valid &= yy >= 0
            srow = np.clip(((BH - 1) * (1.0 - ks / hh)).astype(np.int64), 0, BH - 1)
            scol = np.broadcast_to(np.clip(cc[m], 0, BW - 1).astype(np.int64)[:, None], srow.shape)
            g = (amp[m] * ww[m])[:, None]
            vals = S[srow[valid], scol[valid]] * np.broadcast_to(g, srow.shape)[valid][:, None]
            idx = yy[valid] * W + np.broadcast_to(x[m][:, None], yy.shape)[valid]
            for c in range(3):
                np.maximum.at(out[:, c], idx, vals[:, c])
    out = out.reshape(H, W, 3)
    return cv2.GaussianBlur(out, (0, 0), 1.0)


def tau_of(f, speed=1.0):
    """The clip frame index for C frame f: the clip's birth on the render's, then (for now) real time."""
    return BIRTH_N + (f - F_BIRTH) * CFPS / FPS * speed


def maps_for(f):
    """Screen -> ROI pixel maps (float32) for C frame f, and the world points (for the plate)."""
    cam = cam_letters(f)
    P = screen_to_leaf(cam)
    fx = BIRTH[0] + (P[..., 0] - FW[0]) / K_CM - ROI[0]
    fy = BIRTH[1] - (P[..., 1] - FW[1]) / K_CM - ROI[1]
    return fx.astype(np.float32), fy.astype(np.float32), P


def read_rgb(path):
    im = cv2.imread(path, cv2.IMREAD_COLOR)
    return (im[..., ::-1].astype(np.float32) / 255)


def frame_path(folder, f):
    for ext in ('.png', '.jpg'):
        p = os.path.join(folder, 'f_%05d%s' % (f, ext))
        if os.path.exists(p):
            return p
    return None


def cmd_c5(a):
    frames = []
    for part in a.frames.split(','):
        lo, _, hi = part.partition('-')
        frames += list(range(int(lo), int(hi or lo) + 1))
    frames = sorted(set(frames))
    os.makedirs(a.out, exist_ok=True)
    R = os.path.join(TLD, 'renders')
    # the page before the burn (C 840), reprojected onto each frame's camera (the leaf is near-planar)
    f_plate = F_BIRTH - 1
    plate = read_rgb(frame_path(os.path.join(R, 'book_C'), f_plate))
    cam_p = cam_letters(f_plate)
    FBn = FootageBurn()
    bank = FlameBank() if a.tongues else None
    for f in frames:
        hole, flame, dist = FBn.advance(int(round(tau_of(f, a.speed))))
        fx, fy, P = maps_for(f)
        pp = cam_p.project(P).astype(np.float32)
        O = to_lin(cv2.remap(plate, np.ascontiguousarray(pp[..., 0]), np.ascontiguousarray(pp[..., 1]),
                             cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE))
        hs = cv2.remap(hole, fx, fy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        fl = cv2.remap(flame, fx, fy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        ds = cv2.remap(dist, fx, fy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=99.0)
        nz = cv2.remap(FBn.noise, fx, fy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        keep = char_keep(ds * (1.0 + 0.3 * np.tanh(nz)))
        if bank is not None:
            sp = cv2.remap(FBn.speed(), fx, fy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            tg = flame_tongues(hs, sp, fx, fy, bank, f) * a.tgain
            tg = tg + cv2.GaussianBlur(tg, (0, 0), 10) * 0.35                # their halo
            fl = fl + tg
        # the flames light the paper round them (a soft pool that flickers with them)
        lum = fl @ np.array([0.5, 0.35, 0.15], np.float32)
        small = cv2.resize(lum, (W // 4, H // 4), interpolation=cv2.INTER_AREA)
        pool = cv2.resize(cv2.GaussianBlur(small, (0, 0), 18), (W, H), interpolation=cv2.INTER_LINEAR)
        pool = pool[..., None] * np.array([1.0, 0.62, 0.3], np.float32) * a.pool
        cover = 1.0 - hs
        page = O * keep * (1.0 + pool)
        glow = fl * a.gain * np.array(a.grade, np.float32)
        out = page * cover[..., None] + glow
        rgb = to_srgb(out)
        if a.e15:
            e = frame_path(os.path.join(R, 'embers_C3_e15'), f)
            if e:
                rgb = np.clip(rgb + read_rgb(e), 0, 1)
        cv2.imwrite(os.path.join(a.out, 'f_%05d.jpg' % f), (rgb[..., ::-1] * 255 + 0.5).astype(np.uint8),
                    [cv2.IMWRITE_JPEG_QUALITY, 92])
        if a.layers:
            for sub, img in (('_keep', keep * cover[..., None]), ('_cover', np.repeat(cover[..., None], 3, -1)),
                             ('_glow', to_srgb(glow))):
                d = a.out.rstrip('/') + sub
                os.makedirs(d, exist_ok=True)
                cv2.imwrite(os.path.join(d, 'f_%05d.jpg' % f), (np.clip(img, 0, 1)[..., ::-1] * 255 + 0.5).astype(np.uint8),
                            [cv2.IMWRITE_JPEG_QUALITY, 92])
        print('ftburn c5', f, 'clip', FBn.n, 'hole %.3f' % hs.mean(), flush=True)
    FBn.reader.close()


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd')
    ap.add_argument('--frames', default='841-1039')
    ap.add_argument('--out', default='')
    ap.add_argument('--speed', type=float, default=1.0)
    ap.add_argument('--gain', type=float, default=1.6)
    ap.add_argument('--pool', type=float, default=1.2)
    ap.add_argument('--grade', type=lambda s: [float(v) for v in s.split(',')], default=[1.0, 0.8, 0.55])
    ap.add_argument('--e15', type=int, default=1)
    ap.add_argument('--layers', action='store_true')
    ap.add_argument('--tongues', type=int, default=1, help='(a) real flame tongues along the front')
    ap.add_argument('--tgain', type=float, default=1.0)
    a = ap.parse_args()
    {'geo': cmd_geo, 'probe': cmd_probe, 'c5': cmd_c5}[a.cmd](a)
