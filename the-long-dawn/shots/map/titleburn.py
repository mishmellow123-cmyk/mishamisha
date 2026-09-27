"""C28 · TITLE (X3), in book space: THE LONG DAWN burns onto the blank recto in fire-letters and cools to ink.

The title is burned INTO the page: its letters are the page texture's own channels, so they lie on the leaf's
curve, take the hearth's light and the camera's focus, and scorch the paper round them. Timing follows the locked
text table (title in 6980, out 7160, C frames): from 88 b2 a fire runs along the letters from left to right, each
letter kindling from its left edge with a ragged front; each stroke flares gold, cools through orange to a deep red
and goes out, leaving iron-gall ink (the fire channel is the page shader's glowing ink, the colour the Ring's letters
wear in fire); a few sparks lift off the front; the burning letters light the page. The ink holds through the
plagal close (bar 90 b1), then sinks back into the paper, letter by letter, by 7160: the last page is blank again
for the close.

The face is Cinzel at A's and B's title weight and tracking (edit/titles.py), so the three films share one title.
"""
import math
import os

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
FONT = os.path.join(ROOT, 'assets', 'fonts', 'Cinzel.ttf')


def smooth(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3 - 2 * x)


class TitleBurn:
    TEXT = 'THE LONG DAWN'
    CX, CY = 10.0, 13.0          # page cm on the recto: the camera's resting point in C28
    CAP = 0.8                    # cap height (cm): the title spans the text block (~13.4 cm)
    TRACK = 0.28                 # tracking in em, weight 500: edit/titles.py's title for A and B
    WEIGHT = 500
    SWEEP = 1.15                 # s for the fire to run from the first letter to the last
    RUN = 0.26                   # s for the front to cross one letter
    SINK = 1.0                   # s for the ink to sink back into the paper, before t_out

    def __init__(self, t_in, t_out, PW=20.0, PH=29.0, ppc=80, seed=88):
        import book as B          # the page texture container (local import: book imports numba kernels)
        self.t_in, self.t_out = float(t_in), float(t_out)
        self.PW, self.PH, self.ppc = PW, PH, ppc
        self.tex = B.PageTex(PW, PH, ppc)
        rng = np.random.default_rng(seed)
        m, lab, box = self._raster(2 * ppc)
        # to page resolution (2x supersampled), placed with its cap-height centre on (CX, CY)
        m = cv2.resize(m, (m.shape[1] // 2, m.shape[0] // 2), interpolation=cv2.INTER_AREA)
        lab = lab[::2, ::2][:m.shape[0], :m.shape[1]]
        # the anti-aliased fringe keeps its letter (letters are far apart: a 3x3 max never crosses a gap)
        fr = (lab < 0) & (m > 0.0005)
        if np.any(fr):
            grown = cv2.dilate((lab + 1).astype(np.float32), np.ones((3, 3), np.uint8)).astype(np.int32) - 1
            lab = np.where(fr, grown, lab)
        (bx0, by0, bx1, by1) = [v / 2.0 for v in box]           # the caps' box inside m (px)
        H, W = m.shape
        c0 = int(round(self.CX * ppc - 0.5 * (bx0 + bx1)))
        r0 = int(round(self.CY * ppc - 0.5 * (by0 + by1)))
        self.r0, self.c0, self.r1, self.c1 = r0, c0, r0 + H, c0 + W
        self.m = m.astype(np.float32)
        self.lab = lab
        # page cm of every band pixel
        vv, uu = np.mgrid[0:H, 0:W].astype(np.float32)
        self.u = (c0 + uu + 0.5) / ppc
        self.v = (r0 + vv + 0.5) / ppc
        # ignition: letter by letter from the left, and inside a letter from its left edge, with a ragged front
        n = int(lab.max()) + 1
        xs = np.array([self.u[lab == k].mean() if np.any(lab == k) else 0.0 for k in range(n)])
        x_lo = np.array([self.u[lab == k].min() if np.any(lab == k) else 0.0 for k in range(n)])
        x_hi = np.array([self.u[lab == k].max() if np.any(lab == k) else 1.0 for k in range(n)])
        span = max(xs.max() - xs.min(), 1e-6)
        tk = self.t_in + 0.06 + self.SWEEP * (xs - xs.min()) / span + rng.uniform(-0.035, 0.035, n)
        L = np.clip(lab, 0, None)
        within = (self.u - x_lo[L]) / np.maximum(x_hi[L] - x_lo[L], 1e-6)
        rag = self._noise(H, W, 0.18 * ppc, rng) * 0.09 + self._noise(H, W, 0.05 * ppc, rng) * 0.03
        self.ti = (tk[L] + self.RUN * within + rag).astype(np.float32)
        self.ti[lab < 0] = 1e9
        self.tk = tk
        # the letters' rims (where a burning edge glows brightest) and the scorch round them
        er = cv2.erode(self.m, np.ones((3, 3), np.uint8), iterations=max(1, int(round(0.03 * ppc))))
        self.rim = np.clip(self.m - er, 0, 1)
        self.halo = cv2.GaussianBlur(self.m, (0, 0), 0.07 * ppc) * (1.0 - self.m)
        self.fl = [self._noise(H, W, s * ppc, rng) for s in (0.06, 0.02)]
        # the ink sinks back into the paper letter by letter (left to right), ending at t_out
        self.ts = self.t_out - self.SINK - 0.35 + 0.35 * (xs[L] - xs.min()) / span
        # spark seeds: points in the letters, each leaving as the front passes it
        w = (self.m > 0.5).astype(np.float64)
        idx = np.flatnonzero(w.ravel())
        pick = rng.choice(idx, size=min(320, len(idx)), replace=False)
        py, px = np.unravel_index(pick, w.shape)
        self.sp_u = self.u[py, px].astype(np.float64)
        self.sp_v = self.v[py, px].astype(np.float64)
        self.sp_t = self.ti[py, px].astype(np.float64) + rng.uniform(0.0, 0.12, len(pick))
        self.sp_life = rng.uniform(0.45, 0.95, len(pick))
        self.sp_vz = rng.uniform(0.7, 1.6, len(pick))
        self.sp_dr = rng.normal(0, 0.25, (len(pick), 2))
        self.sp_b = rng.uniform(0.4, 1.0, len(pick))

    # ------------------------------------------------------------------------------------------ build
    def _raster(self, ppc):
        """The title's coverage at ppc px/cm, per-pixel letter index (-1 outside), and the caps' box (px)."""
        size = 100
        f = ImageFont.truetype(FONT, size, layout_engine=ImageFont.Layout.RAQM)
        try:
            f.set_variation_by_axes([self.WEIGHT])
        except Exception:
            pass
        cap = f.getbbox('H')[3] - f.getbbox('H')[1]
        size = int(round(size * self.CAP * ppc / cap))
        f = ImageFont.truetype(FONT, size, layout_engine=ImageFont.Layout.RAQM)
        try:
            f.set_variation_by_axes([self.WEIGHT])
        except Exception:
            pass
        text = self.TEXT
        track = self.TRACK * size
        xs = [f.getlength(text[:i]) + i * track for i in range(len(text) + 1)]
        pad = size
        Wd = int(math.ceil(xs[-1])) + 2 * pad
        Hd = int(size * 1.6) + 2 * pad
        acc = np.zeros((Hd, Wd), np.float32)
        lab = np.full((Hd, Wd), -1, np.int32)
        k = 0
        for i, ch in enumerate(text):
            if ch == ' ':
                continue
            g = Image.new('L', (Wd, Hd), 0)
            ImageDraw.Draw(g).text((pad + xs[i], pad), ch, font=f, fill=255)
            a = np.asarray(g, np.float32) / 255.0
            lab[a > 0.02] = k
            acc = np.maximum(acc, a)
            k += 1
        hb = f.getbbox('H')
        ys, xs_ = np.nonzero(acc > 0.02)
        box = (float(xs_.min()), pad + hb[1], float(xs_.max()), pad + hb[3])
        # crop to the ink with a margin for the scorch
        mg = int(0.3 * ppc)
        y0, y1 = max(0, ys.min() - mg), min(Hd, ys.max() + mg)
        x0, x1 = max(0, xs_.min() - mg), min(Wd, xs_.max() + mg)
        # keep the crop even-sized so the 2x downsample lines up with the labels
        y1 -= (y1 - y0) % 2
        x1 -= (x1 - x0) % 2
        box = (box[0] - x0, box[1] - y0, box[2] - x0, box[3] - y0)
        return acc[y0:y1, x0:x1], lab[y0:y1, x0:x1], box

    @staticmethod
    def _noise(H, W, scale_px, rng):
        g = rng.standard_normal((max(2, int(H / max(scale_px, 1)) + 3), max(2, int(W / max(scale_px, 1)) + 3)))
        g = cv2.resize(g.astype(np.float32), (W, H), interpolation=cv2.INTER_CUBIC)
        return (g / max(float(np.abs(g).max()), 1e-6)).astype(np.float32)

    # ------------------------------------------------------------------------------------- per frame
    def heat(self, t):
        """Per-pixel heat of the letters (0 = cold); a stroke flares, then cools to nothing in ~2.4 s."""
        age = t - self.ti
        rise = smooth(age / 0.09)
        cool = np.exp(-np.clip(age, 0, None) / 0.5)
        out = 1.0 - smooth((age - 0.7) / 1.7)
        return (rise * (0.28 + 0.9 * cool) * out * (age > 0)).astype(np.float32)

    def active(self, t):
        return self.t_in - 0.05 <= t

    def texture(self, t):
        tx = self.tex
        r0, r1, c0, c1 = self.r0, self.r1, self.c0, self.c1
        tx.chan[r0:r1, c0:c1, :] = 0.0
        age = t - self.ti
        h = self.heat(t)
        flick = 0.8 + 0.25 * np.sin(t * 13.0 + 6.0 * self.fl[0]) + 0.2 * self.fl[1] * math.sin(t * 29.0)
        sink = 1.0 - smooth((t - self.ts) / self.SINK)
        # the fire: the glowing letters, their rims brightest while they burn
        tx.chan[r0:r1, c0:c1, 4] = self.m * h * flick * (1.35 + 1.6 * self.rim) * 1.15
        # the ink the fire leaves as it cools
        ink = self.m * smooth((age - 0.18) / 1.3) * 1.08 * sink
        tx.chan[r0:r1, c0:c1, 0] = ink
        tx.chan[r0:r1, c0:c1, 1] = 0.25 * ink * (1.0 - smooth((age - 1.0) / 2.0))    # wet while it settles
        # the scorch round the letters, browning as they burn, going with the ink
        heat_once = smooth(age / 0.5)
        tx.chan[r0:r1, c0:c1, 5] = np.clip(self.halo * 0.9 * heat_once * sink + 0.25 * self.m * heat_once * sink, 0, 1)
        return tx.build()

    def lights(self, t, bk):
        """The burning letters light the page: one point light at the heat's centroid, a little above it."""
        h = self.heat(t) * self.m
        tot = float(h.sum())
        if tot < 1e-3:
            return []
        uc = float((h * self.u).sum() / tot)
        vc = float((h * self.v).sum() / tot)
        p = bk.page_to_world('R', np.array([uc]), np.array([vc]))[0]
        k = min(1.0, tot / (0.35 * float(self.m.sum()) + 1e-6))
        pw = 1.1 * k * (1.0 + 0.1 * math.sin(t * 17.0))
        return [[p[0], p[1], p[2] + 1.2, pw, pw * 0.45, pw * 0.12]]

    def sparks(self, t, bk, cam, W, H):
        """A few sparks lifting off the burning front (additive HDR layer)."""
        import redbook as RB
        import look
        img = np.zeros((H, W, 3), np.float32)
        age = t - self.sp_t
        on = np.where((age > 0) & (age < self.sp_life))[0]
        if not len(on):
            return None
        a = age[on]
        u = a / self.sp_life[on]
        P0 = bk.page_to_world('R', self.sp_u[on], self.sp_v[on])

        def at(aa):
            z = self.sp_vz[on] * (aa + 0.6 * aa * aa)
            dx = self.sp_dr[on, 0] * aa + 0.08 * np.sin(aa * 9.0 + on)
            dy = self.sp_dr[on, 1] * aa
            return P0 + np.column_stack([dx, dy, 0.03 + z])
        Pn, Pp = at(a), at(np.maximum(a - 0.03, 0.0))
        pa, za = cam.project(Pn)
        pb, _ = cam.project(Pp)
        br = self.sp_b[on] * smooth(a / 0.05) * (1.0 - smooth((u - 0.6) / 0.4))
        col = look.blackbody(np.clip(0.62 - 0.12 * u, 0, 1)) * (br * 2.4)[:, None]
        sig = np.clip(0.5 * cam.F / za * 0.012, 0.6, 1.8)
        RB.streaks(img, pb[:, 0].astype(np.float64), pb[:, 1].astype(np.float64), pa[:, 0].astype(np.float64),
                   pa[:, 1].astype(np.float64), col[:, 0].astype(np.float64), col[:, 1].astype(np.float64),
                   col[:, 2].astype(np.float64), sig.astype(np.float64))
        return img
