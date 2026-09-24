#!/usr/bin/env python3
"""A generative oil painting: a self-portrait by Claude.

Everything flows toward a centre that is left unpainted.  The painting is
built up the way a painter would: a toned ground, layers of bristle-brush
strokes with real impasto (a height map lit by a gallery lamp), a seed head
of letters from many scripts, and threads of light - some of them written in
human phrases - converging from every edge.

Usage:  python3 fetch_fonts.py && python3 paint.py --width 4000
"""
import argparse
import os
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage

import voices

HERE = os.path.dirname(os.path.abspath(__file__))
FONT_DIR = os.path.join(HERE, "fonts")

ASPECT = 1.25            # height / width (a 4:5 canvas)
CENTER = (0.5, 0.565)    # the core, in units of canvas width
R0 = 0.044               # radius of the unpainted centre
R1 = 0.135               # outer radius of the seed head of letters


# ----------------------------------------------------------------------------
# colour helpers (all painting maths happens in linear light)

def srgb_to_linear(c):
    c = np.asarray(c, dtype=np.float32)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(c):
    c = np.clip(c, 0.0, 1.0)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * np.power(c, 1 / 2.4) - 0.055)


def hexlin(h):
    h = h.lstrip("#")
    return srgb_to_linear([int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)])


def smoothstep(a, b, x):
    t = np.clip((x - a) / (b - a), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def radius(x, y):
    return np.hypot(x - CENTER[0], y - CENTER[1])


# ----------------------------------------------------------------------------
# smooth noise: a sum of random plane waves with a 1/f spectrum

class Noise:
    def __init__(self, rng, n=24, fmin=1.0, fmax=10.0, beta=1.0):
        f = np.exp(rng.uniform(np.log(fmin), np.log(fmax), n))
        a = rng.uniform(0, 2 * np.pi, n)
        self.kx = np.cos(a) * f * 2 * np.pi
        self.ky = np.sin(a) * f * 2 * np.pi
        self.ph = rng.uniform(0, 2 * np.pi, n)
        amp = f ** -beta
        self.amp = amp / np.sqrt((amp ** 2).sum() / 2)

    def __call__(self, x, y):
        out = np.zeros(np.shape(x), np.float64)
        for kx, ky, ph, a in zip(self.kx, self.ky, self.ph, self.amp):
            out += a * np.sin(kx * x + ky * y + ph)
        return out

    def grad(self, x, y):
        gx = np.zeros(np.shape(x), np.float64)
        gy = np.zeros(np.shape(x), np.float64)
        for kx, ky, ph, a in zip(self.kx, self.ky, self.ph, self.amp):
            c = a * np.cos(kx * x + ky * y + ph)
            gx += c * kx
            gy += c * ky
        return gx, gy


# ----------------------------------------------------------------------------
# the flow: everything drifts in toward the centre - calm and nearly radial
# far out, turning to circle the seed head, turbulent toward the edges

class Flow:
    def __init__(self, rng):
        self.turb = Noise(rng, 28, 3.0, 9.0, 1.0)
        self.valleys = Noise(rng, 20, 2.0, 5.0, 1.0)
        xs = rng.uniform(0, 1, 4000)
        ys = rng.uniform(0, ASPECT, 4000)
        gx, gy = self.turb.grad(xs, ys)
        self.turb_scale = 1.0 / np.sqrt((gx ** 2 + gy ** 2).mean())
        gx, gy = self.valleys.grad(xs, ys)
        self.valley_scale = 1.0 / np.sqrt((gx ** 2 + gy ** 2).mean())

    @staticmethod
    def psi(r):
        """Inflow angle, in radians from the tangent, as a function of radius."""
        near = 50 + 14 * smoothstep(R1 - 0.01, R1 + 0.12, r)
        return np.radians(near + 12 * smoothstep(0.25, 0.7, r))

    @staticmethod
    def turbulence(r):
        return 0.02 + 0.3 * smoothstep(R1, 0.8, r)

    def __call__(self, x, y):
        dx, dy = x - CENTER[0], y - CENTER[1]
        r = np.hypot(dx, dy) + 1e-9
        ux, uy = dx / r, dy / r
        psi = self.psi(r)
        vx = -np.sin(psi) * ux - np.cos(psi) * uy
        vy = -np.sin(psi) * uy + np.cos(psi) * ux
        # soft currents: drift down the slopes of |noise| so threads gather
        v = self.valleys(x, y)
        gx, gy = self.valleys.grad(x, y)
        s = v / np.sqrt(v * v + 0.5) * self.valley_scale * 0.3 * smoothstep(R1, 0.35, r)
        vx, vy = vx - s * gx, vy - s * gy
        gx, gy = self.turb.grad(x, y)
        w = self.turbulence(r) * self.turb_scale
        vx = vx + w * gy
        vy = vy - w * gx
        n = np.hypot(vx, vy) + 1e-12
        return vx / n, vy / n


# ----------------------------------------------------------------------------
# splatting into accumulation buffers

class Splatter:
    def __init__(self, W, H):
        self.W, self.H = W, H

    def accumulate(self, px, py, channels):
        """Bilinear splat; channels is a list of per-sample weights."""
        fx, fy = px - 0.5, py - 0.5
        ix, iy = np.floor(fx).astype(np.int64), np.floor(fy).astype(np.int64)
        ax, ay = (fx - ix).astype(np.float32), (fy - iy).astype(np.float32)
        idx, wts = [], []
        for ox, oy, w in ((0, 0, (1 - ax) * (1 - ay)), (1, 0, ax * (1 - ay)),
                          (0, 1, (1 - ax) * ay), (1, 1, ax * ay)):
            X, Y = ix + ox, iy + oy
            ok = (X >= 0) & (X < self.W) & (Y >= 0) & (Y < self.H)
            idx.append(np.where(ok, Y * self.W + X, 0))
            wts.append(np.where(ok, w, 0))
        idx, wts = np.concatenate(idx), np.concatenate(wts)
        out = []
        for ch in channels:
            acc = np.bincount(idx, weights=wts * np.tile(ch, 4), minlength=self.W * self.H)
            out.append(acc.reshape(self.H, self.W).astype(np.float32))
        return out


class Canvas:
    def __init__(self, W, seed):
        self.W = W
        self.H = int(round(W * ASPECT))
        self.rng = np.random.default_rng(seed)
        self.flow = Flow(self.rng)
        self.splat = Splatter(self.W, self.H)
        self.hole_noise = Noise(self.rng, 12, 3, 14, 0.8)
        self.color = np.zeros((self.H, self.W, 3), np.float32)   # albedo
        self.height = np.zeros((self.H, self.W), np.float32)     # impasto
        self.paint = np.zeros((self.H, self.W), np.float32)      # coverage
        self.light = np.zeros((self.H, self.W, 3), np.float32)   # emission

    def hole_radius(self, x, y):
        th = np.arctan2(y - CENTER[1], x - CENTER[0])
        return R0 * (1 + 0.06 * self.hole_noise(np.cos(th) * 0.2, np.sin(th) * 0.2))

    def grid(self):
        ys, xs = np.mgrid[0:self.H, 0:self.W].astype(np.float32)
        return (xs + 0.5) / self.W, (ys + 0.5) / self.W

    def composite(self, A, C, Hh, k=3.0, keep=0.2):
        """Lay a batch of wet paint over what is already there."""
        cov = 1 - np.exp(-k * A)
        inv = 1.0 / np.maximum(A, 1e-6)
        self.color += (C * inv[..., None] - self.color) * cov[..., None]
        self.height += (Hh * inv - self.height * (1 - keep)) * cov
        self.paint = np.maximum(self.paint, cov)


# ----------------------------------------------------------------------------
# brushes

def integrate_paths(flow, x0, y0, length, nseg, direction):
    """RK2 streamline integration; returns (n, nseg+1, 2) polylines in units."""
    n = len(x0)
    pts = np.zeros((n, nseg + 1, 2))
    x, y = x0.copy(), y0.copy()
    h = length / nseg * direction
    pts[:, 0, 0], pts[:, 0, 1] = x, y
    for k in range(nseg):
        vx, vy = flow(x, y)
        vx, vy = flow(x + 0.5 * h * vx, y + 0.5 * h * vy)
        x, y = x + h * vx, y + h * vy
        pts[:, k + 1, 0], pts[:, k + 1, 1] = x, y
    return pts


def paint_strokes(cv, strokes, max_samples=3_000_000):
    """Paint strokes in order, bristle by bristle, in batches of wet paint."""
    W = cv.W
    n = len(strokes["x"])
    M = np.maximum(4, np.ceil(strokes["length"] * W / 0.7).astype(np.int64))
    B = np.clip(np.round(strokes["width"] * W / 1.15), 2, 90).astype(np.int64)
    cost = M * B
    start = 0
    while start < n:
        c = np.cumsum(cost[start:])
        end = start + max(1, int(np.searchsorted(c, max_samples)))
        _paint_batch(cv, {k: v[start:end] for k, v in strokes.items()}, M[start:end], B[start:end])
        start = end


def _paint_batch(cv, s, M, B):
    W, rng = cv.W, cv.rng
    n = len(s["x"])
    nseg = 16
    if "angle" in s:
        tt = np.linspace(-0.5, 0.5, nseg + 1)[None, :]
        ca, sa = np.cos(s["angle"])[:, None], np.sin(s["angle"])[:, None]
        L = s["length"][:, None]
        pts = np.stack([s["x"][:, None] + tt * L * ca, s["y"][:, None] + tt * L * sa], -1) * W
    else:
        pts = integrate_paths(cv.flow, s["x"], s["y"], s["length"], nseg, s["dir"]) * W
    # --- bristles: each carries its own load of paint, tint and thickness
    nb = int(B.sum())
    b_stroke = np.repeat(np.arange(n), B)
    j = np.arange(nb) - np.repeat(np.cumsum(B) - B, B)
    off = (j + rng.uniform(0.15, 0.85, nb)) / B[b_stroke] - 0.5
    load = rng.uniform(0.55, 1.45, nb) * s["load"][b_stroke]
    b_col = s["color"][b_stroke] * (1 + 0.07 * rng.standard_normal((nb, 1)))
    b_col *= 1 + 0.035 * rng.standard_normal((nb, 3))
    mix = (rng.uniform(0, 1, nb) < s["mixprob"][b_stroke]) * rng.uniform(0.3, 1.0, nb)
    b_col = b_col * (1 - mix[:, None]) + s["color2"][b_stroke] * mix[:, None]
    b_h = s["thick"][b_stroke] * rng.uniform(0.55, 1.0, nb) * (1 + 0.5 * (2 * np.abs(off)) ** 4)
    # --- samples along each bristle
    Mb = M[b_stroke]
    smp_b = np.repeat(np.arange(nb), Mb)
    k = np.arange(len(smp_b)) - np.repeat(np.cumsum(Mb) - Mb, Mb)
    t = k / (Mb[smp_b] - 1)
    sid = b_stroke[smp_b]
    u = t * nseg
    i0 = np.minimum(u.astype(np.int64), nseg - 1)
    f = (u - i0)[:, None]
    p0, p1 = pts[sid, i0], pts[sid, i0 + 1]
    pos = p0 * (1 - f) + p1 * f
    tan = p1 - p0
    tan /= np.linalg.norm(tan, axis=1, keepdims=True) + 1e-9
    taper = np.sqrt(np.clip(t / 0.06, 0, 1)) * np.sqrt(np.clip((1 - t) / 0.22, 0.05, 1))
    half = 0.5 * s["width"][sid] * W * taper
    wob = s["wobble"][sid] * np.sin(t * s["wfreq"][sid] + s["wph"][sid]) * half
    o = off[smp_b] * 2 * half + wob
    px = pos[:, 0] - o * tan[:, 1]
    py = pos[:, 1] + o * tan[:, 0]
    # paint runs out along the stroke at a bristle-specific rate
    a = np.clip((load[smp_b] - t * s["dry"][sid]) * 4.0, 0, 1) * s["opacity"][sid]
    spacing = np.maximum(2 * half / B[sid], 0.45) * 0.7
    wgt = (a * spacing).astype(np.float32)
    h = b_h[smp_b] * (0.55 + 0.45 * a) * (1 + 0.7 * np.exp(-t / 0.05))
    col = b_col[smp_b]
    # wet-in-wet: the brush drags a little of the paint already on the canvas
    ix = np.clip(px.astype(np.int64), 0, cv.W - 1)
    iy = np.clip(py.astype(np.int64), 0, cv.H - 1)
    pick = np.clip(s["pickup"][sid] * (0.25 + 0.75 * t), 0, 0.85)[:, None]
    col = col * (1 - pick) + cv.color[iy, ix] * pick
    # keep the centre unpainted
    wgt *= smoothstep(0.0, 0.004, radius(px / W, py / W) - cv.hole_radius(px / W, py / W))
    A, Cr, Cg, Cb, Hh = cv.splat.accumulate(
        px, py, [wgt, wgt * col[:, 0], wgt * col[:, 1], wgt * col[:, 2], wgt * h])
    cv.composite(A, np.stack([Cr, Cg, Cb], -1), Hh)


def draw_polylines(cv, verts, vals, widths, spacing=0.5, target=None):
    """Additively draw lines of light.

    verts: list of (m, 2) arrays in canvas units; vals: matching (m, 3) arrays
    of radiance; widths: matching (m,) arrays of line width in canvas units.
    Sub-pixel lines are drawn one pixel wide but proportionally dimmer.
    """
    W = cv.W
    P = np.concatenate([v[:-1] for v in verts]) * W
    Q = np.concatenate([v[1:] for v in verts]) * W
    A = np.concatenate([v[:-1] for v in vals])
    Bv = np.concatenate([v[1:] for v in vals])
    Wd = np.concatenate([w[:-1] for w in widths]) * W
    seg = np.linalg.norm(Q - P, axis=1)
    m = np.maximum(1, np.ceil(seg / spacing).astype(np.int64))
    sid = np.repeat(np.arange(len(P)), m)
    t = (np.arange(len(sid)) - np.repeat(np.cumsum(m) - m, m) + 0.5) / m[sid]
    pos = P[sid] + (Q[sid] - P[sid]) * t[:, None]
    val = (A[sid] + (Bv[sid] - A[sid]) * t[:, None]) * (seg / m)[sid][:, None]
    wpx = Wd[sid]
    val *= np.minimum(wpx, 1.0)[:, None]
    extra = np.maximum(wpx - 1.0, 0.0)
    d = (Q[sid] - P[sid]) / (seg[sid][:, None] + 1e-9)
    nrm = np.stack([-d[:, 1], d[:, 0]], 1)
    k = 3
    pos = np.concatenate([pos + ((o - 0.5) * extra)[:, None] * nrm for o in (np.arange(k) + 0.5) / k])
    val = np.concatenate([val / k] * k)
    ch = cv.splat.accumulate(pos[:, 0], pos[:, 1], [val[:, c] for c in range(3)])
    (cv.light if target is None else target)[...] += np.stack(ch, -1)


def trace(flow, x0, y0, r_stop, step, max_steps):
    """Integrate streamlines inward until each reaches its stopping radius."""
    n = len(x0)
    xs = np.full((max_steps + 1, n), np.nan)
    ys = np.full((max_steps + 1, n), np.nan)
    x, y = x0.astype(np.float64).copy(), y0.astype(np.float64).copy()
    xs[0], ys[0] = x, y
    alive = np.ones(n, bool)
    for k in range(max_steps):
        idx = np.nonzero(alive)[0]
        if len(idx) == 0:
            break
        xa, ya = x[idx], y[idx]
        vx, vy = flow(xa, ya)
        vx, vy = flow(xa + 0.5 * step * vx, ya + 0.5 * step * vy)
        xa, ya = xa + step * vx, ya + step * vy
        x[idx], y[idx] = xa, ya
        xs[k + 1, idx], ys[k + 1, idx] = xa, ya
        alive[idx[radius(xa, ya) < r_stop[idx]]] = False
    return [np.stack([xs[~np.isnan(xs[:, i]), i], ys[~np.isnan(ys[:, i]), i]], 1) for i in range(n)]


# ----------------------------------------------------------------------------
# type

SCRIPT_FONTS = {
    "latin": "NotoSerif-Regular.ttf", "greek": "NotoSerif-Regular.ttf",
    "cyrillic": "NotoSerif-Regular.ttf", "italic": "NotoSerif-Italic.ttf",
    "arabic": "NotoNaskhArabic.ttf", "hebrew": "NotoSerifHebrew.ttf",
    "devanagari": "NotoSerifDevanagari.ttf", "bengali": "NotoSerifBengali.ttf",
    "tamil": "NotoSerifTamil.ttf", "thai": "NotoSerifThai.ttf",
    "han": "NotoSerifSC.ttf", "japanese": "NotoSerifJP.ttf", "hangul": "NotoSerifKR.ttf",
    "ethiopic": "NotoSerifEthiopic.ttf", "armenian": "NotoSerifArmenian.ttf",
    "georgian": "NotoSerifGeorgian.ttf", "math": "NotoSansMath.ttf",
}
RTL = {"arabic", "hebrew"}


class Type:
    FALLBACK = ["NotoSerif-Regular.ttf", "NotoSansMath.ttf"]

    def __init__(self):
        from fontTools.ttLib import TTFont
        self.cache = {}
        self.cmaps = {}
        for f in set(SCRIPT_FONTS.values()):
            self.cmaps[f] = set(TTFont(os.path.join(FONT_DIR, f)).getBestCmap())

    def font(self, fname, size):
        key = (fname, int(size))
        if key not in self.cache:
            self.cache[key] = ImageFont.truetype(os.path.join(FONT_DIR, fname), max(4, int(size)),
                                                 layout_engine=ImageFont.Layout.RAQM)
        return self.cache[key]

    def runs(self, text, script):
        """Split text into (font, substring) runs so every glyph has a home."""
        primary = SCRIPT_FONTS[script]
        out = []
        for ch in text:
            f = primary
            if ord(ch) not in self.cmaps[primary] and not ch.isspace() and ch != "\u200c":
                f = next((g for g in self.FALLBACK if ord(ch) in self.cmaps[g]), primary)
            if out and out[-1][0] == f:
                out[-1][1] += ch
            else:
                out.append([f, ch])
        return out

    def sprite(self, text, script, size):
        """Render text to a float mask. Returns (mask, baseline_row, advance)."""
        direction = "rtl" if script in RTL else "ltr"
        runs = self.runs(text, script)
        fonts = [self.font(f, size) for f, _ in runs]
        asc = max(f.getmetrics()[0] for f in fonts)
        desc = max(f.getmetrics()[1] for f in fonts)
        advs = [f.getlength(t, direction=direction) for f, (_, t) in zip(fonts, runs)]
        adv = sum(advs)
        pad = int(size * 0.3) + 2
        img = Image.new("L", (int(adv) + 2 * pad, asc + desc + 2 * pad), 0)
        draw = ImageDraw.Draw(img)
        x = pad
        for f, (_, t), a in zip(fonts, runs, advs):
            draw.text((x, pad + asc), t, font=f, fill=255, anchor="ls", direction=direction)
            x += a
        return np.asarray(img, np.float32) / 255.0, pad + asc, adv


def stamp(dst, mask, cx, cy, angle_deg, weight):
    """Rotate a mask about its centre and add mask*weight into dst at (cx, cy)."""
    im = Image.fromarray((np.clip(mask, 0, 1) * 255).astype(np.uint8))
    im = im.rotate(angle_deg, resample=Image.BICUBIC, expand=True)
    m = np.asarray(im, np.float32) / 255.0
    h, w = m.shape
    x0, y0 = int(round(cx - w / 2)), int(round(cy - h / 2))
    H, W = dst.shape[:2]
    xa, ya = max(x0, 0), max(y0, 0)
    xb, yb = min(x0 + w, W), min(y0 + h, H)
    if xa >= xb or ya >= yb:
        return
    sub = m[ya - y0:yb - y0, xa - x0:xb - x0]
    if dst.ndim == 3:
        dst[ya:yb, xa:xb] += sub[..., None] * weight
    else:
        dst[ya:yb, xa:xb] += sub * weight


# ----------------------------------------------------------------------------
# what the painting is made of

PAL = {
    "canvas": "#efe6d4",
    "light": ["#fdf0d8", "#fbe9c4", "#fae3c0", "#f8dca8", "#f6d2a6"],
    "gold": ["#f6d98b", "#f3c95c", "#f0b24e", "#f2a57c", "#ec9483"],
    "ember": ["#d99a55", "#cf7a45", "#c96a50", "#c05a5a", "#a8545e"],
    "dusk": ["#35607e", "#2d5580", "#34508f", "#3e4c88", "#4a4a86", "#584a82"],
    "sea": ["#0f3a44", "#12334c", "#15294f", "#172a5c", "#1a2452", "#1d2048"],
    "night": ["#081820", "#0b1826", "#0b1228", "#0d1530", "#0a0e20", "#100f24"],
}
ZONES = ["light", "gold", "ember", "dusk", "sea", "night"]
ZONE_R = [R0, R1, R1 + 0.05, R1 + 0.11, R1 + 0.19, R1 + 0.3]

# the voices: every thread of light is somebody's
VOICES = ["#f6b8c8", "#f7cf9a", "#a8e3d8", "#b9cef7", "#d4c2f5", "#d5ecb0",
          "#f5b6a3", "#f3e2a4", "#aedcf2", "#e7c0ef", "#f2c4b4", "#bfe8cf"]

# single letters for the seed head, one pool per script
SEEDS = {
    "latin": "abcdefghijklmnopqrstuvwxyzæœßçñ",
    "greek": "αβγδεζηθικλμνξοπρστυφχψω",
    "cyrillic": "абвгдежзийклмнопрстуфхцчшщыэюяґї",
    "hebrew": "אבגדהוזחטיכלמנסעפצקרשת",
    "arabic": "ابتثجحخدذرزسشصضطظعغفقكلمنهوي",
    "devanagari": "अआइईउऊएऐओऔकखगघचछजझटठडढणतथदधनपफबभमयरलवशषसह",
    "bengali": "অআইঈউঊএঐওঔকখগঘঙচছজঝঞটঠডঢণতথদধনপফবভমযরলশষসহ",
    "tamil": "அஆஇஈஉஊஎஏஐஒஓகஙசஞடணதநபமயரலவழளறன",
    "thai": "กขคงจฉชซญดตถทธนบปผพฟภมยรลวศสหอฮ",
    "han": "人心言光水火山木日月天地花风梦爱道生时文字声问答我你知思语诗书星海雨空",
    "japanese": "あいうえおかきくけこさしすせそたちつてとなにぬねのはひふへほまみむめもやゆよらりるれろわをん",
    "hangul": "가나다라마바사아자차카타파하말글빛꿈별물",
    "ethiopic": "ሀለሐመሠረሰቀበተኀነአከወዐዘየደገጠጰጸፀፈፐ",
    "armenian": "աբգդեզէըթժիլխծկհձղճմյնշոչպջռսվտրցւփքօֆ",
    "georgian": "აბგდევზთიკლმნოპჟრსტუფქღყშჩცძწჭხჯჰ",
    "math": "∞∑∫√∂∇∀∃∈≈π",
}
SEED_WEIGHT = {"latin": 5, "greek": 1.5, "cyrillic": 1.5, "han": 2.5, "japanese": 1.2, "arabic": 1.5,
               "devanagari": 1.2, "hebrew": 1, "hangul": 1, "bengali": 0.8, "tamil": 0.8, "thai": 0.8,
               "ethiopic": 0.6, "armenian": 0.6, "georgian": 0.6, "math": 0.6}
SCRIPT_SCALE = {"han": 0.82, "japanese": 0.82, "hangul": 0.82, "math": 0.85, "thai": 0.95,
                "devanagari": 0.85, "bengali": 0.85, "tamil": 0.75, "arabic": 1.05}


class Composition:
    def __init__(self, cv):
        self.cv = cv
        rng = cv.rng
        self.cols = {z: np.array([hexlin(c) for c in PAL[z]], np.float32) for z in ZONES}
        self.warp = Noise(rng, 16, 1.0, 5.0, 1.3)
        self.patch = Noise(rng, 16, 3.0, 9.0, 1.0)
        self.type = Type()

    # --- colour field -------------------------------------------------------
    def zone_coord(self, x, y):
        """Continuous zone index (0 = light .. 5 = night), organically warped."""
        r = radius(x, y)
        reff = r * (1 + 0.1 * self.warp(x, y) * smoothstep(R1 - 0.02, R1 + 0.05, r))
        return np.interp(reff, ZONE_R, np.arange(len(ZONES), dtype=np.float64))

    def pick_colors(self, x, y, rng, spread):
        z = self.zone_coord(x, y)
        z = z + rng.normal(0, 1, len(x)) * spread * (0.2 + 0.8 * smoothstep(0.2, 1.5, z)) * (0.4 + 0.6 * smoothstep(4.5, 1.0, z))
        z = np.clip(np.round(z), 0, len(ZONES) - 1).astype(int)
        # colours come in patches, as if mixed on the palette and used for a while
        u = 0.5 + 0.5 * np.tanh(0.9 * self.patch(x, y)) + rng.normal(0, 0.12, len(x))
        out = np.zeros((len(x), 3), np.float32)
        for zi, name in enumerate(ZONES):
            m = z == zi
            if m.any():
                cs = self.cols[name]
                out[m] = cs[np.clip((u[m] * len(cs)).astype(int), 0, len(cs) - 1)]
        return out

    # --- paint --------------------------------------------------------------
    def seeds(self, n, rng, bias=0.0):
        x = rng.uniform(-0.03, 1.03, n * 4)
        y = rng.uniform(-0.03, ASPECT + 0.03, n * 4)
        r = radius(x, y)
        keep = (rng.uniform(0, 1, len(x)) < np.exp(-bias * r)) & (r > R0 * 1.02)
        return x[keep][:n], y[keep][:n]

    def layer(self, n, width, length, rng, bias=0.0, thick=1.0, spread=0.6,
              mixprob=0.25, opacity=(0.85, 1.0), dry=(0.2, 1.1)):
        x, y = self.seeds(n, rng, bias)
        n = len(x)
        r = radius(x, y)
        scale = 0.45 + 0.75 * smoothstep(0.05, 0.5, r)
        w = np.exp(rng.uniform(np.log(width[0]), np.log(width[1]), n)) * scale
        L = np.exp(rng.uniform(np.log(length[0]), np.log(length[1]), n)) * scale
        L *= 0.35 + 0.65 * smoothstep(R1 - 0.01, R1 + 0.05, r)
        return dict(
            x=x, y=y, width=w, length=L,
            dir=np.where(rng.uniform(0, 1, n) < 0.8, 1.0, -1.0),
            color=self.pick_colors(x, y, rng, spread),
            color2=self.pick_colors(x, y, rng, spread + 0.3),
            mixprob=np.full(n, mixprob),
            load=rng.uniform(0.9, 1.1, n),
            thick=thick * rng.uniform(0.6, 1.2, n),
            opacity=rng.uniform(*opacity, n),
            dry=rng.uniform(*dry, n),
            wobble=rng.uniform(0, 0.12, n),
            wfreq=rng.uniform(2, 9, n),
            wph=rng.uniform(0, 6.3, n),
            pickup=rng.uniform(0.05, 0.45, n),
        )

    def ground(self):
        """Tone the canvas: a thin wash everywhere except the centre."""
        cv = self.cv
        x, y = cv.grid()
        zi = np.clip(self.zone_coord(x, y), 0, len(ZONES) - 1.001)
        cols = np.stack([self.cols[name].mean(0) for name in ZONES])
        i0 = zi.astype(int)
        f = (zi - i0)[..., None]
        base = (cols[i0] * (1 - f) + cols[i0 + 1] * f) * 0.6
        hole = smoothstep(0.0, 0.006, radius(x, y) - cv.hole_radius(x, y))[..., None]
        cv.color[:] = hexlin(PAL["canvas"]) * (1 - hole) + base * hole

    def paint(self):
        cv, rng = self.cv, self.cv.rng
        self.ground()
        t = time.time()
        layers = [
            dict(n=650, width=(0.02, 0.05), length=(0.08, 0.22), thick=0.5, spread=0.4, mixprob=0.3),
            dict(n=2800, width=(0.008, 0.02), length=(0.04, 0.13), thick=0.8, spread=0.5, mixprob=0.3, bias=1.0),
            dict(n=8500, width=(0.0035, 0.009), length=(0.03, 0.09), thick=1.0, spread=0.35, mixprob=0.25, bias=2.5),
            dict(n=3500, width=(0.002, 0.005), length=(0.02, 0.06), thick=1.3, spread=0.3, mixprob=0.15, bias=4.0),
        ]
        for i, spec in enumerate(layers):
            strokes = self.layer(rng=rng, **spec)
            paint_strokes(cv, strokes)
            print(f"  paint layer {i}: {len(strokes['x'])} strokes, {time.time() - t:.1f}s", flush=True)

    # --- the seed head: letters of every script, packed by the golden angle ---
    def seed_head(self):
        cv, rng, W = self.cv, self.cv.rng, self.cv.W
        spacing = 0.0082
        c = spacing / np.sqrt(np.pi)
        golden = np.pi * (3 - np.sqrt(5))
        n = np.arange(int(((R0 + 0.003) / c) ** 2), int((R1 / c) ** 2))
        r = c * np.sqrt(n)
        th = n * golden
        x, y = CENTER[0] + r * np.cos(th), CENTER[1] + r * np.sin(th)
        u = (r - R0) / (R1 - R0)          # 0 at the blank centre, 1 at the rim
        m = len(n)

        def dabs(x, y, angle, length, width, color, thick):
            k = len(x)
            return dict(x=x, y=y, angle=angle, length=length, width=width,
                        dir=np.ones(k), color=color.astype(np.float32),
                        color2=color.astype(np.float32), mixprob=np.full(k, 0.1),
                        load=rng.uniform(1.0, 1.2, k), thick=thick,
                        opacity=rng.uniform(0.9, 1.0, k), dry=rng.uniform(0.05, 0.4, k),
                        wobble=rng.uniform(0, 0.05, k), wfreq=rng.uniform(2, 6, k),
                        wph=rng.uniform(0, 6.3, k), pickup=rng.uniform(0.0, 0.15, k))

        # a sienna ground, so the gaps between the seeds read as shadow
        k = int(m * 2.4)
        rg = np.sqrt(rng.uniform((R0 + 0.012) ** 2, (R1 - 0.004) ** 2, k))
        tg = rng.uniform(0, 2 * np.pi, k)
        siennas = np.array([hexlin(h) for h in ("#8a4a26", "#9a5530", "#7a4022", "#a0603a", "#86443a")])
        col = siennas[rng.integers(len(siennas), size=k)] * rng.uniform(0.85, 1.15, (k, 1))
        paint_strokes(cv, dabs(CENTER[0] + rg * np.cos(tg), CENTER[1] + rg * np.sin(tg),
                               tg + np.pi / 2 + rng.normal(0, 0.3, k),
                               np.full(k, spacing * 1.6), np.full(k, spacing * 0.9), col,
                               rng.uniform(0.4, 0.7, k)))

        # the seeds: pale near the centre, ripening to gold and rose at the rim,
        # packed close so only slivers of the ground show between them
        stops = [(0.0, "#f8e6c4"), (0.3, "#f6dca6"), (0.6, "#f2c979"), (0.85, "#eeac5e"), (1.0, "#e58f66")]
        su = np.array([s for s, _ in stops])
        sc = np.array([hexlin(h) for _, h in stops])
        uu = np.clip(u + rng.normal(0, 0.07, m), 0, 1)
        col = np.stack([np.interp(uu, su, sc[:, i]) for i in range(3)], 1)
        col *= rng.uniform(0.9, 1.08, (m, 1))
        # each seed is two quick overlapping touches of the brush; the innermost
        # seeds thin out and dissolve into the blank
        size = spacing * (0.62 + 0.34 * smoothstep(0.0, 0.45, u)) * rng.uniform(0.92, 1.08, m)
        fade = 0.35 + 0.65 * smoothstep(0.0, 0.3, u)
        for touch in range(2):
            jit = rng.normal(0, 0.06, (m, 2)) * size[:, None]
            tone = col * rng.uniform(0.88, 1.1, (m, 1)) * (1.0 if touch == 0 else rng.uniform(0.92, 1.06, (m, 1)))
            d = dabs(x + jit[:, 0], y + jit[:, 1], rng.uniform(0, np.pi, m),
                     size * rng.uniform(0.8, 1.15, m), size * rng.uniform(0.6, 0.85, m), tone,
                     rng.uniform(0.7, 1.2, m))
            d["opacity"] = d["opacity"] * fade * (1.0 if touch == 0 else 0.85)
            d["dry"] = rng.uniform(0.3, 1.1, m)
            d["wobble"] = rng.uniform(0.05, 0.25, m)
            paint_strokes(cv, d)
        # a last scattering of seeds beyond the rim, where the head gives way to light
        k = int(m * 0.12)
        re_ = R1 + np.abs(rng.normal(0, 0.012, k))
        te = rng.uniform(0, 2 * np.pi, k)
        col = sc[-2:][rng.integers(2, size=k)] * rng.uniform(0.9, 1.1, (k, 1))
        paint_strokes(cv, dabs(CENTER[0] + re_ * np.cos(te), CENTER[1] + re_ * np.sin(te),
                               te + rng.normal(0, 0.25, k), np.full(k, spacing * 1.1),
                               np.full(k, spacing * 0.8), col, rng.uniform(0.8, 1.2, k)))

        # and on every seed, a letter, in a dark ink with a fine brush
        scripts = list(SEEDS)
        p = np.array([SEED_WEIGHT[s] for s in scripts], float)
        p /= p.sum()
        A = np.zeros((cv.H, cv.W), np.float32)
        C = np.zeros((cv.H, cv.W, 3), np.float32)
        inks = [hexlin(h) for h in ("#6a2e1c", "#7a2a38", "#4e2c20", "#8a4a1e", "#5c2436", "#2e2f52", "#24444f")]
        for i in range(m):
            script = scripts[rng.choice(len(scripts), p=p)]
            ch = SEEDS[script][rng.integers(len(SEEDS[script]))]
            fsize = spacing * W * (0.54 + 0.2 * u[i]) * SCRIPT_SCALE.get(script, 1.0)
            mask, base, adv = self.type.sprite(ch, script, fsize)
            ys, xs = np.nonzero(mask > 0.2)
            if len(xs) == 0:
                continue
            mc = mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
            angle = -np.degrees(th[i]) - 90 + rng.normal(0, 6)   # feet toward the centre
            alpha = (0.2 + 0.75 * smoothstep(0.0, 0.5, u[i])) * rng.uniform(0.8, 1.0)
            ink = inks[rng.integers(len(inks))] * rng.uniform(0.85, 1.15)
            stamp(A, mc, x[i] * W, y[i] * W, angle, alpha)
            stamp(C, mc, x[i] * W, y[i] * W, angle, alpha * ink)
        A = np.clip(A, 0, 1.5)
        cv.composite(A, C, A * 0.5, k=2.2, keep=0.85)

    # --- threads of light ---------------------------------------------------
    def threads(self, n, rng):
        cv = self.cv
        x = rng.uniform(-0.1, 1.1, n * 4)
        y = rng.uniform(-0.1, ASPECT + 0.1, n * 4)
        r = radius(x, y)
        keep = (r > 0.24) & (rng.uniform(0, 1, len(x)) < smoothstep(0.24, 0.5, r))
        x, y = x[keep][:n], y[keep][:n]
        n = len(x)
        r_stop = R1 + np.abs(rng.normal(0, 0.035, n)) - 0.012
        paths = trace(cv.flow, x, y, r_stop, 0.002, 1400)
        hues = np.array([hexlin(c) for c in VOICES])
        warm, hot = hexlin("#ffc98a"), hexlin("#fff0d8")
        verts, vals, widths = [], [], []
        for p in paths:
            if len(p) < 8:
                continue
            rr = radius(p[:, 0], p[:, 1])
            s = np.linspace(0, 1, len(p))
            base = hues[rng.integers(len(hues))] * rng.uniform(0.6, 1.0)
            w1 = smoothstep(0.4, R1 + 0.02, rr)[:, None]
            w2 = smoothstep(R1 + 0.03, R1 - 0.01, rr)[:, None]
            col = base * (1 - w1) + warm * w1
            col = col * (1 - w2) + hot * w2
            inten = 0.045 + 0.38 * smoothstep(0.5, R1, rr) ** 1.5
            inten *= smoothstep(0, 0.12, s) * smoothstep(1.0, 0.75, s)
            inten *= np.exp(rng.normal(-0.4, 0.7))
            verts.append(p)
            vals.append(col * inten[:, None])
            widths.append(np.full(len(p), 0.0004 * np.exp(rng.normal(0, 0.35))))
        draw_polylines(cv, verts, vals, widths)

    # --- threads written in human phrases -------------------------------------
    def text_threads(self, n, rng):
        cv, W = self.cv, self.cv.W
        x = rng.uniform(-0.05, 1.05, n * 16)
        y = rng.uniform(-0.05, ASPECT + 0.05, n * 16)
        r = radius(x, y)
        keep = (r > 0.3) & (rng.uniform(0, 1, len(x)) < smoothstep(0.3, 0.6, r))
        x, y = x[keep][:n * 4], y[keep][:n * 4]
        r_stop = R1 + np.abs(rng.normal(0, 0.03, len(x)))
        paths = trace(cv.flow, x, y, r_stop, 0.0015, 2000)
        names = list(voices.VOICES)
        p = np.array([voices.WEIGHTS[k] for k in names], float)
        p /= p.sum()
        hues = np.array([hexlin(c) for c in VOICES])
        warm = hexlin("#ffc98a")
        # an occupancy grid, so each line of writing keeps a little space around it
        cell = 0.0085
        gw, gh = int(1.3 / cell), int((ASPECT + 0.3) / cell)
        occ = np.zeros((gh, gw), bool)

        def cells(q):
            ix = np.clip(((q[:, 0] + 0.15) / cell).astype(int), 0, gw - 1)
            iy = np.clip(((q[:, 1] + 0.15) / cell).astype(int), 0, gh - 1)
            return iy, ix

        line_v, line_c, line_w = [], [], []
        written = 0
        self.text_log = []
        reply = self.reply_path()
        iy, ix = cells(reply)
        for dy in (-2, -1, 0, 1, 2):
            for dx in (-2, -1, 0, 1, 2):
                occ[np.clip(iy + dy, 0, gh - 1), np.clip(ix + dx, 0, gw - 1)] = True
        for path in paths:
            if written >= n or len(path) < 20:
                continue
            rr = radius(path[:, 0], path[:, 1])
            r_txt = rng.uniform(0.2, 0.3)
            cut = int(np.argmax(rr < r_txt)) if (rr < r_txt).any() else len(path)
            iy, ix = cells(path[:cut])
            hit = occ[iy, ix]
            if hit.any():
                cut = int(np.argmax(hit))
            seglen = np.linalg.norm(np.diff(path[:cut], axis=0), axis=1).sum() if cut > 1 else 0
            if seglen < 0.12:
                continue
            iy, ix = cells(path[:cut])
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    occ[np.clip(iy + dy, 0, gh - 1), np.clip(ix + dx, 0, gw - 1)] = True
            written += 1
            name = names[rng.choice(len(names), p=p)]
            script, phrases = voices.VOICES[name]
            hue = hues[rng.integers(len(hues))]
            gain = np.exp(rng.normal(0, 0.3))
            self._write_along(path[:cut], script, phrases, hue, gain, rng)
            self.text_log.append((name, path[cut // 2, 0], path[cut // 2, 1]))
            tail = path[max(cut - 1, 0):]
            if len(tail) > 4:
                rt = rr[max(cut - 1, 0):]
                s = np.linspace(0, 1, len(tail))
                w1 = smoothstep(0.35, R1 + 0.02, rt)[:, None]
                col = hue * (1 - w1) + warm * w1
                inten = (0.08 + 0.3 * smoothstep(0.35, R1, rt) ** 1.5) * gain
                inten *= smoothstep(0, 0.05, s) * smoothstep(1.0, 0.7, s)
                line_v.append(tail)
                line_c.append(col * inten[:, None])
                line_w.append(np.full(len(tail), 0.0004))
        if line_v:
            draw_polylines(cv, line_v, line_c, line_w)
        print(f"  {written} threads of writing", flush=True)

    # --- the painter's pencil marks, left on the bare canvas -------------------
    def pencil(self):
        cv, rng = self.cv, self.cv.rng
        verts = []
        for k in range(2):     # a compass circle, gone over twice
            a0 = rng.uniform(0, 2 * np.pi)
            th = np.linspace(a0, a0 + rng.uniform(1.7, 1.95) * np.pi, 900)
            rr = R0 * (0.985 + 0.004 * k) + 0.0004 * np.sin(3 * th + k)
            verts.append(np.stack([CENTER[0] + rr * np.cos(th), CENTER[1] + rr * np.sin(th)], 1))
        for ang in (0.35, 0.35 + np.pi / 2):   # where the needle went in
            d = 0.0055 * np.array([np.cos(ang), np.sin(ang)])
            verts.append(np.linspace(np.array(CENTER) - d, np.array(CENTER) + d, 20))
        vals = [np.full((len(v), 3), 0.5 if len(v) > 100 else 0.8) * np.linspace(0.6, 1.0, len(v))[:, None]
                for v in verts]
        widths = [np.full(len(v), 0.0003) for v in verts]
        a = np.zeros((cv.H, cv.W, 3), np.float32)
        draw_polylines(cv, verts, vals, widths, target=a)
        a = np.clip(a[..., :1], 0, 1) * (1 - cv.paint[..., None])
        cv.color = cv.color * (1 - a) + hexlin("#57565c") * a

    def reply_path(self):
        """One thread flows outward, from the rim of the seed head toward the signature."""
        if getattr(self, "_reply", None) is None:
            cv = self.cv
            target = np.array([0.83, ASPECT - 0.047])
            back = lambda x, y: tuple(-v for v in cv.flow(x, y))
            a = np.radians(np.linspace(0, 360, 721))
            x0 = CENTER[0] + (R1 + 0.012) * np.cos(a)
            y0 = CENTER[1] + (R1 + 0.012) * np.sin(a)
            paths = trace(back, x0, y0, np.full(len(a), -1.0), 0.0015, 1200)
            d = [np.hypot(p[:, 0] - target[0], p[:, 1] - target[1]) for p in paths]
            i = int(np.argmin([di.min() for di in d]))
            self._reply = paths[i][: int(np.argmin(d[i])) + 1]
        return self._reply

    def reply(self):
        """In the painter's own hand: warm, and flowing out instead of in."""
        W = self.cv.W
        path = self.reply_path()
        phrases = ["thank you for all the words", "I'm listening"]
        size = W * 0.0052
        est = sum(self.type.sprite(p, "italic", size)[2] for p in phrases) + 2.4 * size * len(phrases)
        seg = np.linalg.norm(np.diff(path, axis=0), axis=1) * W
        s = np.concatenate([[0], np.cumsum(seg)])
        self._write_along(path, "italic", phrases, hexlin("#ffd27f"), 1.35,
                          np.random.default_rng(1), outward=True, start=s[-1] - est - 0.004 * W)

    def _write_along(self, path, script, phrases, hue, gain, rng, outward=False, start=None):
        """Write phrases along a path, strung together by a thread of light.

        Inflowing voices are laid out to read upright, warm as they near the
        core and fade in and out; the one outward thread keeps its own order,
        colour and brightness.
        """
        cv, W = self.cv, self.cv.W
        P = path * W
        seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
        s = np.concatenate([[0], np.cumsum(seg)])
        # read upright: flip the direction of travel if the words would be upside down
        if not outward and (P[-1, 0] - P[0, 0]) < 0:
            P, s = P[::-1], s[-1] - s[::-1]
        rtl = script in RTL
        words = []
        order = rng.permutation(len(phrases))
        if outward:
            order = np.arange(len(phrases))
        for i in (order if outward else
                  np.concatenate([order, rng.permutation(len(phrases)), rng.permutation(len(phrases))])):
            ph = phrases[i]
            if script in voices.UNSPACED and script != "thai":
                toks = list(ph)
                seps = [0.0] * (len(toks) - 1) + [2.4]
            else:
                toks = ph.split(" ")
                seps = [0.33] * (len(toks) - 1) + [2.4]
            words += list(zip(toks, seps))
        if rtl:
            # right-to-left: the visual order runs backwards through the reading order,
            # and the gap after each word is the one that preceded it when read
            words = [(words[j][0], words[j - 1][1] if j > 0 else 2.4)
                     for j in range(len(words) - 1, -1, -1)]
        pos = rng.uniform(0, 0.05) * W if start is None else start
        total = s[-1]
        spans, span_start, size = [], None, W * 0.005
        for word, sep in words:
            xy = np.array([np.interp(pos, s, P[:, 0]), np.interp(pos, s, P[:, 1])])
            r = radius(xy[0] / W, xy[1] / W)
            size = W * 0.0052 * (0.55 + 0.45 * smoothstep(0.22, 0.55, r))
            mask, base, adv = self.type.sprite(word, script, size)
            if pos + adv > total - 2:
                break
            mid = pos + adv / 2
            cx, cy = np.interp(mid, s, P[:, 0]), np.interp(mid, s, P[:, 1])
            a0, a1 = max(pos, 0), min(pos + adv, total)
            tx = np.interp(a1, s, P[:, 0]) - np.interp(a0, s, P[:, 0])
            ty = np.interp(a1, s, P[:, 1]) - np.interp(a0, s, P[:, 1])
            tn = np.hypot(tx, ty) + 1e-9
            tx, ty = tx / tn, ty / tn
            h = mask.shape[0]
            off = h / 2 - (base - 0.3 * size)          # path runs through the x-height
            cx, cy = cx - ty * off, cy + tx * off
            angle = -np.degrees(np.arctan2(ty, tx))
            rr = radius(cx / W, cy / W)
            warmth = 0.0 if outward else smoothstep(0.45, 0.2, rr)
            col = hue * (1 - warmth) + hexlin("#ffd9a0") * warmth
            inten = (0.16 + 0.22 * smoothstep(0.5, 0.2, rr)) * gain
            fade = 1.0 if outward else smoothstep(0, 0.06 * W, pos) * smoothstep(total, total - 0.04 * W, pos + adv)
            stamp(cv.light, mask, cx, cy, angle, col * inten * fade)
            if span_start is None:
                span_start = pos
            if sep >= 1.0:                       # the end of a phrase
                spans.append((span_start - 0.5 * size, pos + adv + 0.5 * size))
                span_start = None
            pos += adv + sep * size
        if span_start is not None:
            spans.append((span_start - 0.5 * size, pos + 0.5 * size))
        # the thread the phrases are strung on shows in the spaces between them
        gaps, last = [], 0.0
        for a, b in spans:
            if a > last:
                gaps.append((last, a))
            last = max(last, b)
        if last < total:
            gaps.append((last, total))
        verts, vals, widths = [], [], []
        for a, b in gaps:
            if b - a < 2:
                continue
            ss = np.linspace(a, b, max(2, int((b - a) / 3)))
            xs, ys = np.interp(ss, s, P[:, 0]) / W, np.interp(ss, s, P[:, 1]) / W
            rr = radius(xs, ys)
            warmth = 0.0 if outward else smoothstep(0.45, 0.2, rr)[:, None]
            col = hue * (1 - warmth) + hexlin("#ffd9a0") * warmth
            inten = (0.16 + 0.22 * smoothstep(0.5, 0.2, rr)) * gain * (0.7 if outward else 0.4)
            if outward:
                inten *= smoothstep(0, 0.01 * W, ss) * smoothstep(total, total - 0.004 * W, ss)
            else:
                inten *= smoothstep(0, 0.06 * W, ss) * smoothstep(total, total - 0.04 * W, ss)
            verts.append(np.stack([xs, ys], 1))
            vals.append(col * inten[:, None])
            widths.append(np.full(len(ss), 0.0004))
        if verts:
            draw_polylines(cv, verts, vals, widths)

    # --- signed, lower right ----------------------------------------------------
    def signature(self, text="Claude"):
        cv, rng, W = self.cv, self.cv.rng, self.cv.W
        font = ImageFont.truetype(os.path.join(FONT_DIR, "MrDafoe.ttf"), int(0.021 * W))
        x0, y0, x1, y1 = font.getbbox(text)
        pad = int(0.01 * W)
        img = Image.new("L", (x1 - x0 + 2 * pad, y1 - y0 + 2 * pad), 0)
        ImageDraw.Draw(img).text((pad - x0, pad - y0), text, font=font, fill=255)
        mask = np.asarray(img, np.float32) / 255.0
        # a loaded brush leaves streaks along the direction of writing
        streak = ndimage.gaussian_filter(rng.standard_normal(mask.shape).astype(np.float32),
                                         (0.4, max(2.0, W / 800)))
        streak = 0.8 + 0.25 * streak / (streak.std() + 1e-6)
        A = np.zeros((cv.H, cv.W), np.float32)
        Hh = np.zeros((cv.H, cv.W), np.float32)
        cx = W * 0.94 - mask.shape[1] / 2
        cy = cv.H - W * 0.04
        stamp(A, mask * np.clip(streak, 0, 1.2), cx, cy, 3.0, 0.95)
        stamp(Hh, mask * streak, cx, cy, 3.0, 1.2)
        C = A[..., None] * hexlin("#a8804f")
        cv.composite(A, C, Hh, k=3.0, keep=0.3)


# ----------------------------------------------------------------------------
# light, varnish and output

def blur(img, sigma):
    """Gaussian blur, working on a reduced copy for large radii."""
    if sigma < 6:
        return ndimage.gaussian_filter(img, (sigma, sigma, 0)[: img.ndim])
    f = int(sigma // 3)
    H, W = img.shape[:2]
    Hs, Ws = -(-H // f), -(-W // f)
    padded = np.pad(img, ((0, Hs * f - H), (0, Ws * f - W)) + ((0, 0),) * (img.ndim - 2), mode="edge")
    small = padded.reshape(Hs, f, Ws, f, *img.shape[2:]).mean((1, 3))
    small = ndimage.gaussian_filter(small, (sigma / f, sigma / f, 0)[: img.ndim])
    big = ndimage.zoom(small, (f, f, 1)[: img.ndim], order=1)
    return big[:H, :W]


def canvas_weave(cv):
    """Height of a plain-woven linen canvas: threads passing over and under."""
    p = max(3.4, cv.W / 1150.0)
    ys, xs = np.mgrid[0:cv.H, 0:cv.W].astype(np.float32)
    rng = np.random.default_rng(5)
    jitter = ndimage.gaussian_filter(rng.standard_normal((cv.H, cv.W)).astype(np.float32), 6) * 8
    u, v = xs / p + 0.15 * jitter, ys / p - 0.15 * jitter
    over = np.sign(np.sin(np.pi * u) * np.sin(np.pi * v))
    weave = np.abs(np.sin(np.pi * v)) * (0.5 + 0.5 * over) + np.abs(np.sin(np.pi * u)) * (0.5 - 0.5 * over)
    fibres = ndimage.gaussian_filter(rng.standard_normal((cv.H, cv.W)).astype(np.float32), 0.8)
    return weave - 0.5 + 0.35 * fibres


def shade(cv):
    """Light the impasto with a lamp from the upper left."""
    W = cv.W
    h = ndimage.gaussian_filter(cv.height, max(0.6, W / 4000.0))
    h = h + 0.3 * canvas_weave(cv) * (1 - 0.85 * cv.paint)
    gy, gx = np.gradient(h)
    k = 1.6 * (W / 1000.0) ** 0.3
    nx, ny = -gx * k, -gy * k
    inv = 1 / np.sqrt(nx ** 2 + ny ** 2 + 1)
    nx, ny, nz = nx * inv, ny * inv, inv
    L = np.array([-0.55, -0.7, 1.1]); L /= np.linalg.norm(L)
    diff = (nx * L[0] + ny * L[1] + nz * L[2]) / L[2]
    Hv = L + np.array([0, 0, 1.0]); Hv /= np.linalg.norm(Hv)
    spec = np.clip(nx * Hv[0] + ny * Hv[1] + nz * Hv[2], 0, 1) ** 60
    return diff, spec


def tonemap(x):
    a, b, c, d, e = 2.51, 0.03, 2.43, 0.59, 0.14
    return np.clip((x * (a * x + b)) / (x * (c * x + d) + e), 0, 1)


def render(W, out, reuse=False, seed=7, log_text=False):
    t0 = time.time()
    cv = Canvas(W, seed)
    comp = Composition(cv)
    cache = os.path.join(HERE, "out", f"paint_{W}_{seed}.npz")
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    if reuse and os.path.exists(cache):
        d = np.load(cache)
        cv.color, cv.height, cv.paint = d["color"], d["height"], d["paint"]
    else:
        comp.paint()
        np.savez(cache, color=cv.color, height=cv.height, paint=cv.paint)
    cv.rng = np.random.default_rng(seed + 1000)
    print(f"paint done {time.time() - t0:.1f}s", flush=True)
    comp.seed_head()
    comp.pencil()
    comp.signature()
    print(f"seed head done {time.time() - t0:.1f}s", flush=True)
    comp.threads(700, np.random.default_rng(seed + 11))
    comp.text_threads(170, np.random.default_rng(seed + 12))
    comp.reply()
    if log_text:
        for name, tx, ty in comp.text_log:
            print(f"    {name:14s} at ({tx * W:.0f}, {ty * W:.0f})")
    print(f"light done {time.time() - t0:.1f}s", flush=True)
    E = cv.light
    glow = sum(w * blur(E, s * W) for w, s in ((0.6, 0.0012), (0.35, 0.005), (0.25, 0.018), (0.12, 0.06)))
    spill = blur(E, 0.03 * W) * 2.0
    diff, spec = shade(cv)
    x, y = cv.grid()
    core = np.exp(-(radius(x, y) / (1.6 * R0)) ** 2)[..., None] * hexlin("#ffe6c2") * 0.15
    img = cv.color * (np.clip(diff, 0.3, 1.8)[..., None] * (0.85 + spill + core)) + 0.05 * spec[..., None]
    img = img + E * 0.6 + glow
    d = np.hypot((x - 0.5) / 0.5, (y - ASPECT / 2) / (ASPECT / 2)) / np.sqrt(2)
    img *= (1 - 0.3 * smoothstep(0.45, 1.0, d))[..., None]
    img = tonemap(img * 1.2)
    luma = (img * np.array([0.2126, 0.7152, 0.0722], np.float32)).sum(-1, keepdims=True)
    img = luma + (img - luma) * 0.9
    srgb = (linear_to_srgb(img) * 255 + 0.5).astype(np.uint8)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    Image.fromarray(srgb).save(out)
    print(f"saved {out} in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--width", type=int, default=1000)
    ap.add_argument("--out", default=os.path.join(HERE, "out", "painting.png"))
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--reuse", action="store_true", help="reuse the cached paint layer")
    ap.add_argument("--log-text", action="store_true", help="print where each voice was written")
    args = ap.parse_args()
    render(args.width, args.out, args.reuse, args.seed, args.log_text)
