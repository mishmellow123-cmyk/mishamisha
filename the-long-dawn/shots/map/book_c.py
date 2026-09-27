"""C · THE LAST PAGES: the book shots on the LOCKED bar map (`music/v3/barmap_C.json`), numbered on C's own timeline.

    C2      THE RED BOOK        80-320    the drift over the last written leaves; from bar 4 b1 the blank sheaf
    C3      THE MOUNTAIN       320-560    the leaves riffle back; the pen draws the fire, the ring (bar 5 b3), the mountain
    C4+C5   LETTERS TO FIRE    560-1040   the turn, the hearth sinks, glow 9b4, lift 10b1, fire 11b1, burn 11b3, the fire alone
    C8+C9   THE DEEP           1680-1992  swept to parchment; the pen follows the vein; the red glow burns through (C9)
    C25-28  THE LAST PAGES     6160-7200  plenty; the Havens (coast fires 81b3 82b3 83b3, west 84b1); blank; the next; title

    python book_c.py frames --frames 80-319 --out renders/book_C        # delivery (matte to renders/book_C_matte)
    python book_c.py frames --frames 700,760,840 --scale 0.5 --out DIR  # tests
    python book_c.py export --out renders/book_C                        # x1_letters.json for EMBERS (spark seeds)

Every frame is a pure function of its C frame number. The layer is premultiplied colour plus a matte: EDIT comps
out = rgb + (1 - matte) * next shot. C4-C5 (560-1040) is PAGE ONLY (director, agreed with EMBERS-C): the letters
glow and each stroke flares and goes out on its locked timing, the page is lit by the fire (point lights), the page
burns and opens; the sparks, the ember and the flame are EMBERS-C's (renders/embers_C3_e15, added by EDIT; seeds
from `Kindling` / `export`). `--with-fire` draws MAP's own fire for tests. Text lines are EDIT's (T1 is written in
the Mountain's caption band, T7 on the Deep, T14 on the first healed blank page, the title on the last).
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
from numba import njit, prange

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..', '..', 'lib'))
import look  # noqa: E402
import pen  # noqa: E402
import pages as PG  # noqa: E402
import book as B  # noqa: E402
import burn as BURN  # noqa: E402
import redbook as RB  # noqa: E402
from pen import INK, Strokes  # noqa: E402

ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
FPS = 24
SHOTS = {                     # name: (first frame, end frame) on C's timeline, half-open
    'red_book': (80, 320),
    'mountain': (320, 560),
    'letters': (560, 1040),
    'deep': (1680, 1992),
    'ring_melt': (5360, 5520),       # the Ring close-up FALLBACK for C22 (ringpage.py): the ink ring on the page
    'last_pages': (6160, 7200),
}
smooth = RB.smooth
lerp = RB.lerp
keyed = RB.keyed
HEARTH = (1.0, 0.66, 0.36)


def ramp(t, a, b):
    return smooth((t - a) / (b - a))


# ======================================================= the burn, in world ===

@njit(cache=True, parallel=True)
def burn_post(hdr, alpha, G, bf, t, PH):
    """Burn every pixel of the layer (page, leather, table alike) with a field evaluated at its world position
    (u = x, v = PH/2 - y), so a hole crosses the whole frame: browning and char darken, the edge glows, the hole
    empties the matte."""
    H, W = alpha.shape
    for i in prange(H):
        for j in range(W):
            if G[i, j, 0] == 0:
                u = 0.0
                v = -40.0
            else:
                u = G[i, j, 2]
                v = 0.5 * PH - G[i, j, 3]
            brown, char, hole, edge, pool = BURN.field(bf, u, v, t)
            k0 = (1 - 0.5 * brown) * (1 - 0.95 * char)
            k1 = (1 - 0.72 * brown) * (1 - 0.96 * char)
            k2 = (1 - 0.88 * brown) * (1 - 0.97 * char)
            r = hdr[i, j, 0] * k0 * (1.0 + 0.6 * pool)
            g = hdr[i, j, 1] * k1 * (1.0 + 0.3 * pool)
            b = hdr[i, j, 2] * k2 * (1.0 + 0.1 * pool)
            cov = alpha[i, j] * (1.0 - hole)
            er = edge * 4.2
            eg = edge * 1.25
            eb = edge * 0.18
            hdr[i, j, 0] = r * (1.0 - hole) + er
            hdr[i, j, 1] = g * (1.0 - hole) + eg
            hdr[i, j, 2] = b * (1.0 - hole) + eb
            alpha[i, j] = cov


# ================================================================ the fire ===

class Kindling:
    """C4-C5 (locked; director's X1 notes). On bar 9 b4 the dense leaf's script glows AS LETTERS, lit like the
    Ring's letters in fire (a deep orange-red core, never white) and legible as writing. From bar 10 b1 a peel runs
    in from the rim: word by word, letter by letter, each stroke flares and peels off the page as the sparks that
    were its ink, and every word's sparks go as one stream, drawn down toward the page's heart like sparks in a
    hearth's draught, curving and quickening, with short trails (never a field of dots, never a spiral). They
    gather at the heart; on bar 11 b1 one flame catches there, gold and calm, and on bar 11 b3 it burns the page
    open."""

    T_GLOW, T_LIFT, T_FIRE, T_BURN = 140 / 24.0, 160 / 24.0, 240 / 24.0, 280 / 24.0   # shot seconds from 560
    PEEL = 1.75                   # the peel front: rim (bar 10 b1) to the heart
    TRAIL = 0.035                 # a spark's short trail (s)
    GATHER = 4.2                  # how strongly the draught gathers sparks into streams

    def __init__(self, book, words, F, S=None, side='R', seed=3, step=0.14):
        self.book, self.words, self.F = book, words, np.asarray(F, np.float64)
        rng = np.random.default_rng(seed)
        n = len(words)
        cxy = np.array([[0.5 * (w[2] + w[3]), w[4] - 0.08] for w in words])
        dist = np.hypot(cxy[:, 0] - F[0], cxy[:, 1] - F[1])
        dn = dist / max(dist.max(), 1e-6)
        # the glow ripples out from the heart in a third of a second (bar 9 b4)
        self.tg = self.T_GLOW + 0.3 * dn + rng.uniform(0, 0.06, n)
        # the peel: from the rim in toward the heart; each word from its far end to its near end
        self.tl = self.T_LIFT + self.PEEL * (1.0 - dn) ** 1.1 + rng.uniform(0, 0.12, n)
        self.dur = 0.22 + 0.1 * np.array([w[3] - w[2] for w in words])
        Fw = book.page_to_world(side, np.array([F[0]]), np.array([F[1]]))[0]
        self.Fw = Fw
        n_st = max(w[1] for w in words) if words else 0
        self.ts = np.full(n_st, 1e9)                # each stroke's peel time
        pts, t0s, wis = [], [], []
        for wi, (a, b, xl, xr, base, li) in enumerate(words):
            d = np.array([F[0] - cxy[wi, 0], F[1] - cxy[wi, 1]])
            d /= max(np.hypot(*d), 1e-6)
            for k in range(a, b):
                P = S.P[k] if S is not None else np.array([[xl, base], [xr, base]])
                c = P.mean(0)
                # position along the word from its far end (0) to its near end (1)
                w_ = (xr - xl) + 1e-6
                fr_ = np.clip(((c[0] - xl) / w_ - 0.5) * np.sign(d[0] if abs(d[0]) > 0.2 else 1.0) + 0.5, 0, 1)
                self.ts[k] = self.tl[wi] + self.dur[wi] * fr_
                L = float(np.sum(np.hypot(*np.diff(P, axis=0).T))) if len(P) > 1 else 0.0
                m = int(np.clip(round(L / step), 1, 4))
                if len(P) > 1:
                    s_ = np.concatenate([[0], np.cumsum(np.hypot(*np.diff(P, axis=0).T))])
                    q = np.sort(rng.uniform(0, max(s_[-1], 1e-6), m))
                    pp = np.column_stack([np.interp(q, s_, P[:, 0]), np.interp(q, s_, P[:, 1])])
                else:
                    pp = P[:1].repeat(m, 0)
                for j in range(m):
                    pts.append(pp[j])
                    t0s.append(self.ts[k] + 0.05 * j / max(m, 1) + rng.uniform(0, 0.03))
                    wis.append(wi)
        pts = np.array(pts)
        self.wi = np.array(wis, int)
        self.t0 = np.array(t0s)
        self.side = side
        self.P0 = book.page_to_world(side, pts[:, 0], pts[:, 1]) + np.array([0, 0, 0.02])
        # the draught: every spark runs down a gently uneven slope toward the heart (the gradient of distance plus a
        # smooth noise, turned only sideways): a flow with no turning in it, so no spiral, whose streamlines gather
        # into streams that merge like rivulets as they near the heart, curving, and quicken as they come
        self.pts = pts
        self._fly(rng, seed)
        self.heat = rng.uniform(0.6, 1.0, len(self.t0))

    def _fly(self, rng, seed, h=0.05, ds=0.06, nmax=420):
        from scipy.ndimage import map_coordinates
        from noise import fbm_grid
        F = self.F
        u0, v0, u1, v1 = -1.0, -1.0, 21.0, 30.0
        Wg, Hg = int((u1 - u0) / h), int((v1 - v0) / h)
        Nf = fbm_grid(u0, -v0, h, h, Hg, Wg, 1.0 / 4.6, int(seed) + 17, 2, 2.0, 0.45).astype(np.float64)
        gv, gu = np.gradient(Nf, h)
        U = u0 + (np.arange(Wg) + 0.5) * h
        V = v0 + (np.arange(Hg) + 0.5) * h
        du = U[None, :] - F[0]
        dv = V[:, None] - F[1]
        r = np.maximum(np.hypot(du, dv), 1e-6)
        ru, rv = du / r, dv / r
        tu, tv = -rv, ru
        side = self.GATHER * (gu * tu + gv * tv) * np.clip(r / 1.2, 0, 1)
        fu = -ru - side * tu
        fv = -rv - side * tv
        n = np.maximum(np.hypot(fu, fv), 1e-9)
        fu, fv = fu / n, fv / n

        def flow(P):
            cj = (P[:, 0] - u0) / h - 0.5
            ci = (P[:, 1] - v0) / h - 0.5
            return (map_coordinates(fu, [ci, cj], order=1, mode='nearest'),
                    map_coordinates(fv, [ci, cj], order=1, mode='nearest'))
        N = len(self.pts)
        P = self.pts.copy()
        lift = rng.uniform(0.3, 0.65, N)
        tr = np.zeros((N, nmax, 3), np.float32)
        tm = np.zeros((N, nmax), np.float32)
        sacc = np.zeros(N)
        tacc = np.zeros(N)
        done = np.zeros(N, bool)
        for k in range(nmax):
            rr = np.hypot(P[:, 0] - F[0], P[:, 1] - F[1])
            z = 0.02 + lift * RB.smooth_arr(sacc / 0.35) * np.clip(rr / 2.2, 0.12, 1.0) ** 0.7
            tr[:, k, 0], tr[:, k, 1], tr[:, k, 2] = P[:, 0], P[:, 1], z
            tm[:, k] = tacc
            done |= rr < 0.12
            a_, b_ = flow(P)
            ha_ = 0.5 * ds
            a2, b2 = flow(P + ha_ * np.column_stack([a_, b_]))              # midpoint step
            step = np.where(done, 0.0, ds)
            P = P + step[:, None] * np.column_stack([a2, b2])
            speed = 2.2 + 9.0 / (rr + 1.5)                                    # cm/s: quicker near the heart
            tacc = tacc + step / speed
            sacc = sacc + step
        T = tm[:, -1].astype(np.float64)
        avail = np.maximum(self.T_FIRE - 0.12 - self.t0, 0.4)
        sc = np.minimum(1.0, avail / np.maximum(T, 1e-6))
        self.tm = (tm * sc[:, None]).astype(np.float32)
        self.d = T * sc
        # page -> world, once
        W3 = self.book.page_to_world(self.side, tr[..., 0].ravel().astype(np.float64), tr[..., 1].ravel().astype(np.float64))
        W3 = W3.reshape(N, nmax, 3)
        W3[..., 2] += tr[..., 2]
        self.tr = W3.astype(np.float32)

    def gains(self, t, nstrokes):
        gi = np.ones(nstrokes)
        gf = np.zeros(nstrokes)
        gg = np.zeros(nstrokes)
        for wi, (a, b, xl, xr, base, li) in enumerate(self.words):
            tg = self.tg[wi]
            if t < tg:
                continue
            e = smooth((t - tg) / 0.35)
            fl = 1.0 + 0.12 * math.sin(t * 17.0 + wi * 1.7) + 0.07 * math.sin(t * 31.0 + wi)
            ts = self.ts[a:b]
            out = RB.smooth_arr((t - ts) / 0.1)
            flash = np.exp(-((t - ts) / 0.05) ** 2)
            gf[a:b] = (0.8 * e * fl + 0.9 * flash) * (1.0 - out)
            gi[a:b] = (1.0 - 0.45 * e) * (1.0 - RB.smooth_arr((t - ts + 0.02) / 0.1))
            gg[a:b] = 0.42 * RB.smooth_arr((t - ts) / 0.3)
        return gi, gf, gg

    def pos(self, trel, idx):
        """World positions of sparks idx at times trel since their lift (along their recorded flights)."""
        tm = self.tm[idx]
        k = np.clip((tm <= trel[:, None]).sum(1) - 1, 0, tm.shape[1] - 2)
        r = np.arange(len(idx))
        ta, tb = tm[r, k], tm[r, k + 1]
        w = np.clip((trel - ta) / np.maximum(tb - ta, 1e-6), 0, 1)[:, None]
        return self.tr[idx, k] * (1 - w) + self.tr[idx, k + 1] * w

    def sparks(self, t):
        trel = t - self.t0
        live = np.where((trel > 0) & (trel < self.d))[0]
        if not len(live):
            z = np.zeros((0, 3))
            return z, z, np.zeros(0), np.zeros(0)
        tl = trel[live]
        Pn = self.pos(tl, live)
        Pp = self.pos(np.maximum(tl - self.TRAIL, 0.0), live)
        dl = Pp - Pn                                        # the trail stays short even where they race
        L = np.linalg.norm(dl, axis=1)
        Pp = Pn + dl * np.minimum(1.0, 0.28 / np.maximum(L, 1e-9))[:, None]
        u = tl / self.d[live]
        br = self.heat[live] * np.clip(tl / 0.05, 0, 1) * (1.0 - RB.smooth_arr((u - 0.88) / 0.12))
        temp = 0.56 + 0.14 * u
        return Pn.astype(np.float64), Pp.astype(np.float64), br, temp

    def ember(self, t):
        """The glowing point gathering at the heart before the fire catches (0..1)."""
        arr = np.sum(t > self.t0 + 0.9 * self.d) / max(len(self.t0), 1)
        return min(1.0, 1.3 * arr) * (1.0 - ramp(t, self.T_FIRE, self.T_FIRE + 0.6))

    def flame_h(self, t):
        k = ramp(t, self.T_FIRE, self.T_FIRE + 1.4)
        calm = ramp(t, self.T_BURN + 2.0, self.T_BURN + 5.0)
        return (0.3 + 3.0 * k) * (1.0 - 0.1 * calm) if t > self.T_FIRE else 0.0


def fire_layer(cam, K, t, W, H):
    """Sparks, the gathering ember, and the flame as an additive HDR layer; plus the flame's light for the page."""
    img = np.zeros((H, W, 3), np.float32)
    Pn, Pp, br, temp = K.sparks(t)
    if len(Pn):
        a, za = cam.project(Pn)
        b, zb = cam.project(Pp)
        col = look.blackbody(np.clip(temp, 0, 1)) * (br * 2.9)[:, None]      # embers, orange, never white
        sig = np.clip(0.55 * cam.F / za * 0.012, 0.6, 2.2)
        RB.streaks(img, b[:, 0].astype(np.float64), b[:, 1].astype(np.float64), a[:, 0].astype(np.float64),
                   a[:, 1].astype(np.float64), col[:, 0].astype(np.float64), col[:, 1].astype(np.float64),
                   col[:, 2].astype(np.float64), sig.astype(np.float64))
    base = K.Fw + np.array([0, 0, 0.02])
    (bx, by), zb_ = cam.project(base)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    xl = []
    e = K.ember(t)
    if e > 0.01:
        r = 2.0 + 5.0 * e
        g = np.exp(-(((xx - bx) ** 2 + (yy - by) ** 2) / (2 * r * r)))
        img += g[..., None] * np.array([3.0, 1.2, 0.3], np.float32) * e
        xl.append([base[0], base[1], base[2] + 0.4, 0.5 * e, 0.22 * e, 0.06 * e])
    h = K.flame_h(t)
    if h > 0.02:
        top = base + np.array([0, 0, h])
        (tx, ty), _ = cam.project(top)
        hp = math.hypot(tx - bx, ty - by)
        ux, uy = (tx - bx) / max(hp, 1e-6), (ty - by) / max(hp, 1e-6)
        fr = t * FPS
        calm = ramp(t, K.T_BURN + 2.0, K.T_BURN + 5.0)
        # one natural flame, gold and calm (no side tongues): a slow sway, its licks soft
        RB.FIRE.flame(img, float(bx), float(by), float(hp), float(ux), float(uy), float(fr * (0.9 - 0.35 * calm)), 7,
                      float(1.55), float(1.3), -0.22, 0.55 - 0.2 * calm)
        rr = max(hp * 0.16, 2.0)
        bed = np.exp(-(((xx - bx) / (rr * 1.6)) ** 2 + ((yy - by) / (rr * 0.55)) ** 2))
        img += bed[..., None] * np.array([2.2, 0.55, 0.08], np.float32)
        pw = 1.6 * h * (1.0 + (0.12 * math.sin(t * 17.0) + 0.08 * math.sin(t * 29.0)) * (1 - 0.6 * calm))
        xl.append([base[0], base[1], base[2] + 0.6 * h, pw, pw * 0.5, pw * 0.16])
    return img, (np.array(xl) if xl else None)


def fire_lights(K, t):
    """The fire's light on the page only: the gathering ember and the flame as point lights (the same values
    fire_layer returns), for the PAGE-ONLY C4-C5 layer (director, 27 Sep, agreed with EMBERS-C: EMBERS renders the
    sparks, the ember and the flame into renders/embers_C3_e15; EDIT adds them over book_C)."""
    base = K.Fw + np.array([0, 0, 0.02])
    xl = []
    e = K.ember(t)
    if e > 0.01:
        xl.append([base[0], base[1], base[2] + 0.4, 0.5 * e, 0.22 * e, 0.06 * e])
    h = K.flame_h(t)
    if h > 0.02:
        calm = ramp(t, K.T_BURN + 2.0, K.T_BURN + 5.0)
        pw = 1.6 * h * (1.0 + (0.12 * math.sin(t * 17.0) + 0.08 * math.sin(t * 29.0)) * (1 - 0.6 * calm))
        xl.append([base[0], base[1], base[2] + 0.6 * h, pw, pw * 0.5, pw * 0.16])
    return np.array(xl) if xl else None


# ================================================================== pages ===

def dense_leaf(seed=61):
    """The leaf dense with the tale's script (C4): 38 lines, the words recorded for the kindling."""
    S = Strokes()
    h = pen.Hand(seed=seed, xh=0.21)
    h.write_block(S, 3.2, 3.0, 13.8, 38, 0.61)
    return S, list(h.words)


class Book3:
    """The book shots of C, sharing pages and textures."""

    def __init__(self, W=1920, H=804, with_fire=False):
        self.W, self.H = W, H
        self.books = {}
        self.cache = {}
        self.with_fire = with_fire      # C4-C5: MAP's own sparks and flame (tests only; the delivery is PAGE ONLY)

    def book(self, TL, TR, seed=3):
        key = (round(TL, 3), round(TR, 3), seed)
        if key not in self.books:
            self.books[key] = B.Book(TL=TL, TR=TR, seed=seed)
        return self.books[key]

    def once(self, key, fn):
        if key not in self.cache:
            self.cache[key] = fn()
        return self.cache[key]

    # ---- shared textures
    def tex_last_leaves(self):
        def mk():
            L, R, yl = RB.leaves_last()
            return RB.Page(L, 70).texture(1e9), RB.Page(R, 70).texture(1e9), yl
        return self.once('last', mk)

    def tex_text(self, seed, ppc=45):
        def mk():
            S = Strokes()
            PG.text_page(S, seed, lines=38, box=(3.0, 2.9, 16.8, 26.4))
            return RB.Page(S, ppc).texture(1e9)
        return self.once(('text', seed, ppc), mk)

    def mountain(self):
        def mk():
            mt = PG.Mountain(11)
            R = mt.build('ink', 1.1, 8.6, sched=PG.Mountain.SCHED_C3)
            h = pen.Hand(seed=18, xh=0.2)
            h.write_block(R, 3.2, 15.05, 13.8, 18, 0.62)
            # its letters kindle faintly in the drawn fire and fade: a small arc of the script round the ring
            fl = Strokes()
            hh = pen.Hand(seed=19, xh=0.055, nib=0.2, thin=0.04)
            fx, fy = mt.ring
            k0 = len(fl)
            x = fx - 0.55
            for w in range(3):
                x = hh.write_word(fl, hh.word(3), x, fy + 0.52 - 0.06 * abs(w - 1)) + 0.05
            for q in range(k0, len(fl)):
                P = fl.P[q].copy()
                # bend the line into an arc under the ring
                a = (P[:, 0] - fx) / 0.9
                fl.P[q] = np.column_stack([fx + 0.62 * np.sin(a), fy + 0.1 + 0.46 * np.cos(a) - (fy + 0.52 - P[:, 1])])
            return mt, RB.Page(R, 110), fl.pack()
        return self.once('mountain', mk)

    # ---- the shots
    def frame(self, f):
        for name, (a, b) in SHOTS.items():
            if a <= f < b:
                return getattr(self, 'shot_' + name)((f - a) / FPS, f)
        return None

    def light(self, pos, power, t, seed=3, amt=0.09, col=HEARTH, radius=9.0):
        fl = B.flicker(t, seed, amt)
        L = B.Light(np.asarray(pos, np.float64) + np.array([1.5 * math.sin(t * 1.3), 0.8 * math.sin(t * 0.9),
                                                             0.5 * math.sin(t * 2.1)]), col, power * fl, radius)
        return L

    def finish_layer(self, bk, cam, L, tL, tR, t, fire=None, xl=None, leaf=None, burnL=None, burnR=None, post=None,
                     dof=None):
        hdr, alpha, G = B.render(bk, cam, L, tL, tR, t, burnL=burnL, burnR=burnR, xlights=xl, leaf=leaf)
        if post is not None:
            burn_post(hdr, alpha, G, post, float(t), float(bk.PH))
        k = cam.dof_k if dof is None else dof
        hdr = B.dof(hdr, G[..., 1].astype(np.float32), cam, k * self.W / 1920.0)
        if fire is not None:
            hdr = hdr + fire
        return hdr, alpha

    # C2 THE RED BOOK (80-320): the drift over the last written leaves; from bar 4 b1, the blank sheaf
    def shot_red_book(self, t, f):
        bk = self.book(2.6, 1.4)
        tL, tR, yl = self.tex_last_leaves()
        cam = self.cam_red_book(bk, t)
        L = self.light((-55, 55, 38), 2.2, t)
        # the room's warm bounce from the right, so the sheaf's leaf-edges read at the end
        bounce = 2200.0 * ramp(t, 5.0, 9.0)
        xl = np.array([[70.0, -35.0, 30.0, bounce, bounce * 0.62, bounce * 0.36]])
        return self.finish_layer(bk, cam, L, tL, tR, t, xl=xl)

    def cam_red_book(self, bk, t):
        ship = bk.page_to_world('R', np.array([10.1]), np.array([18.5]))[0]
        keys_t = [(0.0, (-10.0, 3.0, 2.0)), (3.8, (3.0, 1.0, 2.0)), (6.67, ship + [0.5, 0.5, 0]),
                  (10.0, (19.0, -6.5, 1.3))]
        keys_p = [(0.0, (-12.0, -17.0, 13.0)), (3.8, (-1.0, -19.5, 15.5)), (6.67, ship + [-1.5, -17.0, 15.0]),
                  (10.0, (29.0, -19.0, 9.0))]
        cam = B.Cam(RB.catrom(keys_p, t), RB.catrom(keys_t, t), 30.0 + 4.0 * smooth(t / 10.0), self.W, self.H)
        cam.dof_k = 4.5 - 1.5 * smooth(t / 10.0)
        return cam

    # C3 THE MOUNTAIN (320-560): the riffle back, then the pen
    RIFFLE = [(0.06 + 0.075 * k, 0.42) for k in range(12)]

    def shot_mountain(self, t, f):
        mt, pM, flk = self.mountain()
        land = sum(smooth((t - a) / d) for a, d in self.RIFFLE) / len(self.RIFFLE)
        TL = 2.6 + (0.6 - 2.6) * land
        TR = 1.4 + (2.8 - 1.4) * land
        bk = self.book(TL, TR)
        tText = self.tex_text(41)
        tM = pM.texture(t)
        # its letters kindle faintly in the drawn fire, and fade
        g = 0.45 * ramp(t, 1.9, 2.4) * (1.0 - ramp(t, 3.4, 4.6)) * (1 + 0.2 * math.sin(t * 23))
        if g > 0.005:
            C, _ = pen.raster(flk, 1e9, pM.ppc, tM.H, tM.W, INK, gain=np.full(len(flk['T0']), g))
            pM.tex.chan[..., 4] = C
            tM = pM.tex.build()
        L = self.light((-50, 48, 36), 2.2, t)
        in_riffle = t < 1.45
        subs = (-1.0 / 72, 0.0, 1.0 / 72) if in_riffle else (0.0,)
        acc = None
        for dt in subs:
            ts = t + dt
            phis = [1.0 - smooth((ts - a) / d) for a, d in self.RIFFLE]
            leaf = (phis, tM, tText) if in_riffle else None
            tR = tText if (in_riffle and land < 0.999) else tM
            cam = self.cam_mountain(bk, ts)
            hdr, alpha = self.finish_layer(bk, cam, L, tText, tR, ts, leaf=leaf)
            acc = (hdr, alpha) if acc is None else (acc[0] + hdr, acc[1] + alpha)
        n = len(subs)
        return acc[0] / n, acc[1] / n

    def cam_mountain(self, bk, t):
        plate = bk.page_to_world('R', np.array([9.8]), np.array([9.4]))[0]
        keys_t = [(0.0, (19.0, -6.5, 1.3)), (1.0, (6.0, 1.0, 2.0)), (2.1, plate), (10.0, plate + [0.3, 0.2, 0])]
        keys_p = [(0.0, (29.0, -19.0, 9.0)), (1.0, (9.0, -32.0, 30.0)), (2.1, plate + [-0.6, -19.8, 18.2]),
                  (10.0, plate + [-0.2, -18.4, 16.9])]
        cam = B.Cam(keyed(keys_p, t), keyed(keys_t, t), 36.0 + 2.0 * ramp(t, 0.8, 2.1), self.W, self.H)
        cam.dof_k = 2.0
        return cam

    # C4 + C5 LETTERS TO FIRE (560-1040)
    def letters(self):
        def mk():
            bk = self.book(0.66, 2.74, seed=5)
            S, words = dense_leaf()
            F = (10.0, 14.2)
            K = Kindling(bk, words, F, S=S)
            return bk, RB.Page(S, 100), K
        return self.once('letters', mk)

    def blur(self, fn, t, n=3, shutter=1.0 / 48):
        """Average n renders across the shutter (a turning leaf moves a long way in one frame)."""
        acc = None
        for k in range(n):
            ts = t + shutter * ((k + 0.5) / n - 0.5)
            h, a = fn(ts)
            acc = (h, a) if acc is None else (acc[0] + h, acc[1] + a)
        return acc[0] / n, acc[1] / n

    def shot_letters(self, t, f):
        if 0.0 < t < 1.4:
            return self.blur(self.letters_at, t, n=5)
        return self.letters_at(t)

    def letters_at(self, t):
        mt, pM, flk = self.mountain()
        bk, pD, K = self.letters()
        tM = pM.texture(1e9)
        tBack = self.tex_text(43)           # the Mountain leaf's back
        tD = pD.texture(1e9, K.gains(t, len(pD.S)))
        phi = smooth(t / 1.4)
        leaf = (phi, tM, tBack) if 0 < phi < 1 else None
        tL = self.tex_text(41) if phi <= 0 else tBack
        tR = tD if phi > 0 else tM
        cam = self.cam_letters(bk, K, t)
        # the hearth sinks low until the page is almost dark; then the letters and the fire light it
        power = 2.2 - 1.9 * ramp(t, 0.5, 5.0)
        L = self.light((-50, 48, 36), power, t, amt=0.12)
        # PAGE ONLY (contract with EMBERS-C): the page lit by the fire, not the fire; stroke timings and the camera
        # are EMBERS-C's seeds and must not change without telling them
        if self.with_fire:
            fire, xl = fire_layer(cam, K, t, self.W, self.H)
        else:
            fire, xl = None, fire_lights(K, t)
        post = None
        if t > K.T_FIRE:
            Fu, Fv = float(K.Fw[0]), float(0.5 * bk.PH - K.Fw[1])
            post = BURN.hold_params((Fu, Fv), K.T_BURN, 6.6, 1.2, amp=0.42, freq=0.35, seed=4, brown=2.8, char=0.34,
                                    edge=0.04, lead=1.5)
        return self.finish_layer(bk, cam, L, tL, tR, t, fire=fire, xl=xl, leaf=leaf, post=post)

    def cam_letters(self, bk, K, t):
        plate = bk.page_to_world('R', np.array([9.8]), np.array([9.4]))[0]
        F = K.Fw
        keys_t = [(0.0, plate + [0.3, 0.2, 0]), (0.6, plate + [0.3, 0.2, 0]), (2.8, F + [0.0, 1.0, 0]),
                  (20.0, F + [0.0, 0.2, 0])]
        keys_p = [(0.0, plate + [-0.2, -18.4, 16.9]), (0.6, plate + [-0.2, -18.4, 16.9]), (2.8, F + [-0.5, -17.0, 17.5]),
                  (20.0, F + [-0.3, -11.5, 12.0])]
        cam = B.Cam(keyed(keys_p, t), keyed(keys_t, t), 38.0, self.W, self.H)
        cam.dof_k = 2.5
        return cam

    # C22 FALLBACK (5360-5520): THE UNMAKING as the ink ring on the book's page (ringpage.py)
    def ring_page(self):
        def mk():
            import ringpage
            return self.book(1.5, 1.5, seed=13), ringpage.RingPage(), ringpage
        return self.once('ring_page', mk)

    def shot_ring_melt(self, t, f):
        if 0.2 < t < 0.62:                              # the fall: a real shutter, or it strobes
            return self.blur(lambda ts: self.ring_melt_at(ts), t, n=9)
        return self.ring_melt_at(t)

    def ring_melt_at(self, t):
        bk, rp, RP = self.ring_page()
        white = smooth((t - 5.4) / 1.2) ** 1.5
        tR = rp.texture(t, white=white)
        tL = self.tex_text(47)
        cam = self.cam_ring(bk, t)
        L = self.light((-50, 48, 36), 0.4, t, amt=0.1)
        fx, fy = RP.RingPage.FIRE
        fw = bk.page_to_world('R', np.array([fx]), np.array([fy - 1.6]))[0]
        fl = B.flicker(t * 1.6, 5, 0.14) * (1.0 + 9.0 * white) * (1.0 - 0.25 * smooth((t - 3.3) / 1.0) * (1 - white))
        xl = np.array([[fw[0], fw[1], fw[2] + 2.6, 1.9 * fl, 0.9 * fl, 0.24 * fl]])
        hdr, alpha = self.finish_layer(bk, cam, L, tL, tR, t, xl=xl)
        # the hearth flares to white: bar 70 opens in white
        hdr = hdr * (1.0 + 5.0 * white) + (white ** 2) * 7.0
        return hdr, alpha

    def cam_ring(self, bk, t, still=None):
        tgt = bk.page_to_world('R', np.array([10.2]), np.array([16.9]))[0]
        off = np.array([-0.5, -17.5, 20.0]) * (1.0 - 0.08 * smooth(t / 6.6))
        if still == 'find':
            tgt = bk.page_to_world('R', np.array([10.2]), np.array([18.0]))[0]
            off = np.array([0.8, -14.0, 16.5])
        cam = B.Cam(tgt + off, tgt, 36.0, self.W, self.H)
        cam.dof_k = 2.2
        return cam

    def ring_still(self, name):
        """The find (the Ring lying on the page, its letters dark, a glint) and the fire test (in the drawn fire,
        its letters awake, unmarked): the stills for their fallbacks."""
        bk, rp, RP = self.ring_page()
        if name == 'find':
            tR = rp.texture(1.2, find=True, pristine=True)
        else:
            tR = rp.texture(1.2, pristine=True, awake=0.9)      # in the fire, letters awake, unmarked
        tL = self.tex_text(47)
        cam = self.cam_ring(bk, 0.0, still=name)
        L = self.light((-50, 48, 36), 0.9 if name == 'find' else 0.55, 0.0, amt=0.1)
        fx, fy = RP.RingPage.FIRE
        fw = bk.page_to_world('R', np.array([fx]), np.array([fy - 1.6]))[0]
        xl = None if name == 'find' else np.array([[fw[0], fw[1], fw[2] + 2.6, 2.6, 1.25, 0.35]])
        return self.finish_layer(bk, cam, L, tL, tR, 0.0, xl=xl)

    # C8 THE DEEP (1680-1920) + the burn into THE EYE (1920-1992)
    def deep(self):
        def mk():
            bk = self.book(1.6, 2.2, seed=9)
            dp = PG.Deep()
            spans = np.array([2.7, 1.7, 1.67, 1.13, 1.1, 0.7])
            acc = np.concatenate([[0], np.cumsum(spans)]) / spans.sum()
            R = dp.build('ink', 0.6, 9.6, pace=lambda k: (acc[k], acc[k + 1]))
            return bk, dp, RB.Page(R, 110)
        return self.once('deep', mk)

    def shot_deep(self, t, f):
        bk, dp, pDp = self.deep()
        tDp = pDp.texture(t)
        tL = self.tex_text(31, 40)
        cam = self.cam_deep(bk, dp, t)
        L = self.light((-50, 48, 36), 2.1, t)
        gx, gy = dp.glow
        gw = bk.page_to_world('R', np.array([gx]), np.array([gy]))[0]
        gi = 6.5 * ramp(t, 6.8, 10.0) * (1.0 + 0.15 * math.sin(t * 13.0) + 0.1 * math.sin(t * 31.0))
        xl = np.array([[gw[0], gw[1], gw[2] + 0.35, 1.6 * gi, 0.2 * gi, 0.025 * gi]]) if gi > 0 else None
        if t < 1.8:
            # an ember edge sweeps the race away down the frame, and leaves parchment behind it (X1 reversed)
            post = BURN.sweep_params((0.0, 1.0), 0.0, 1.25, span=(-1.5, 12.5), seed=31)
        else:
            # the paper browns and smokes round the glow; on bar 25 b1 the glow burns through (into the storm)
            Gu, Gv = float(gw[0]), float(0.5 * bk.PH - gw[1] - 0.25)
            post = BURN.params((Gu, Gv), t_start=10.0, speed=3.2, p=1.9, amp=0.45, freq=0.35, seed=9, brown=3.6,
                               char=0.3, edge=0.035, lead=3.0)
        return self.finish_layer(bk, cam, L, tL, tDp, t, xl=xl, post=post)

    def cam_deep(self, bk, dp, t):
        # the camera follows the pen down: each level is framed while it is drawn
        ts_ = [0.0, 2.0, 3.3, 5.0, 6.67, 7.8, 8.9, 10.0, 13.0]
        vs_ = [4.2, 5.4, 7.8, 11.6, 15.6, 19.2, 22.6, 24.0, 24.4]
        q = np.linspace(-0.6, 0.6, 13)
        w = np.exp(-(q / 0.35) ** 2)
        v = float(np.sum(w * np.interp(t + q, ts_, vs_)) / np.sum(w))
        xv = float(np.interp(v + 1.0, dp.vein[:, 1], dp.vein[:, 0]))
        xv = 0.55 * xv + 0.45 * 9.6
        tgt = bk.page_to_world('R', np.array([xv]), np.array([v]))[0]
        cam = B.Cam(tgt + np.array([-0.4, -13.5, 13.0]), tgt, 40.0, self.W, self.H)
        cam.dof_k = 2.5
        return cam

    # C25-C28 THE LAST PAGES (6160-7200)
    TURNS = ((10.0, 1.4), (23.333, 1.35), (25.0, 1.35))     # the Havens (81 b1); blank (85 b1); the next (85 b3)
    COAST = (11.667, 15.0, 18.333)                           # 81 b3, 82 b3, 83 b3
    STERN = 19.3
    WEST = 20.0                                              # 84 b1: the ship slides west, off the page
    HEAL = (21.5, 26.0)                                      # healed by the first blank page (T14 at 6790)

    def last_pages(self):
        def mk():
            ep = RB.Epilogue(self.W, self.H)
            ep.fire_t = [self.COAST[0], self.COAST[1], self.COAST[2], 18.9 + 0.4]
            return ep
        return self.once('last_pages', mk)

    def havens_tex(self, ep, t):
        tx = B.PageTex(PG.PW, PG.PH, ep.hv_ppc)
        dx = 2.6 * ramp(t, 10.8, self.WEST) + 11.0 * ramp(t, self.WEST, self.WEST + 3.3) ** 1.4
        ship, stern = ep.hv.ship_strokes(dx)
        C, _ = pen.raster(ship.pack(), 1e9, ep.hv_ppc, tx.H, tx.W, INK)
        occ = ep.hv.ship_mask(dx, ep.hv_ppc, tx.H, tx.W)
        tx.chan[..., 0] = np.maximum(ep.hv_ink * (1.0 - occ), C)
        fs = Strokes()
        fs.extend(ep.hv_fire)
        fs.add(np.array([[stern[0], stern[1] + 0.02], [stern[0] + 0.004, stern[1] - 0.05]]), np.array([0.05, 0.03]),
               np.array([1.0, 1.0]))
        ft = list(self.COAST) + [self.STERN + 0.4, self.STERN]       # three coast fires, the lamp, the stern light
        g = np.array([smooth((t - a) / 0.5) * (0.85 + 0.15 * math.sin(t * (17 + 3 * i) + i)) for i, a in enumerate(ft)])
        F, _ = pen.raster(fs.pack(), 1e9, ep.hv_ppc, tx.H, tx.W, INK, gain=g)
        tx.chan[..., 4] = F
        return tx.build(), g

    def shot_last_pages(self, t, f):
        if any(a < t < a + d for a, d in self.TURNS):
            return self.blur(self.last_pages_at, t, n=5)
        return self.last_pages_at(t)

    def last_pages_at(self, t):
        ep = self.last_pages()
        bk = ep.book
        k, phi = None, 0.0
        for q, (a, d) in enumerate(self.TURNS):
            if a <= t < a + d:
                k, phi = q, smooth((t - a) / d)
        leaf = None
        xl = []
        if t < self.TURNS[0][0]:
            tL, tR = ep.tex_text, ep.pPlenty.texture(t)
        else:
            hvt, g = self.havens_tex(ep, t)
            tR = ep.tex_blank
            if k == 0:
                if ep.tex_plenty_done is None:
                    ep.tex_plenty_done = ep.pPlenty.texture(1e9)
                tL = ep.tex_text
                leaf = (phi, ep.tex_plenty_done, hvt)
            else:
                tL = hvt if t < self.TURNS[1][0] + self.TURNS[1][1] else ep.tex_blank
                if k in (1, 2):
                    leaf = (phi, ep.tex_blank, ep.tex_blank)
                if tL is hvt:
                    for i, (fx, fy, hh) in enumerate(ep.hv.fires):
                        if g[i] > 0.01:
                            wp = bk.page_to_world('L', np.array([fx]), np.array([fy - 0.3]))[0]
                            xl.append([wp[0], wp[1], wp[2] + 0.6, 0.35 * g[i], 0.12 * g[i], 0.03 * g[i]])
        burnL = BURN.edge_params('L', 0.9, self.HEAL[1], seed=21, start=self.HEAL[0])
        burnR = BURN.edge_params('R', 0.9, self.HEAL[1], seed=22, start=self.HEAL[0])
        cam = self.cam_last_pages(bk, t)
        L = self.light((-55, 50, 30), 1.9, t, seed=5, amt=0.12, col=(1.0, 0.52, 0.24), radius=10)
        return self.finish_layer(bk, cam, L, tL, tR, t, xl=(np.array(xl) if xl else None), leaf=leaf, burnL=burnL,
                                 burnR=burnR)

    def cam_last_pages(self, bk, t):
        pl = bk.page_to_world('R', np.array([10.5]), np.array([11.0]))[0]
        hv = bk.page_to_world('L', np.array([10.0]), np.array([10.5]))[0]
        sp = np.array([0.0, 0.0, bk.zg])
        bl = bk.page_to_world('R', np.array([10.0]), np.array([13.0]))[0]
        keys_t = [(0.0, pl + [1.2, 0, 0]), (9.6, pl + [1.0, 0, 0]), (12.2, hv + [0.5, -1.0, 0]), (22.8, hv + [0.2, -1.5, 0]),
                  (24.2, sp + [0, -0.5, 0]), (31.5, sp + [2.5, -0.5, 0]), (35.0, bl), (43.33, bl + [0, 0.3, 0])]
        keys_o = [(0.0, (-2.0, -30.0, 28.0)), (9.6, (-1.8, -29.0, 27.0)), (12.2, (1.0, -31.0, 29.0)),
                  (22.8, (0.8, -30.0, 28.0)), (24.2, (0.0, -52.0, 46.0)), (31.5, (0.5, -50.0, 44.0)),
                  (35.0, (0.0, -30.0, 28.0)), (43.33, (0.0, -28.0, 26.0))]
        cam = B.Cam(keyed(keys_t, t) + keyed(keys_o, t), keyed(keys_t, t), 38.0, self.W, self.H)
        cam.dof_k = 2.0
        return cam


# ================================================================== export ===

def export_letters(out, W=1920, H=804):
    """x1_letters.json for EMBERS: the page's letters (screen boxes at the glow), every spark's seed (screen
    position and frame at the lift) and the heart's screen track, so E15's particles can take over exactly."""
    b3 = Book3(W, H)
    bk, pD, K = b3.letters()
    a0 = SHOTS['letters'][0]
    rec = dict(frame_glow=a0 + int(round(K.T_GLOW * FPS)), frame_lift=a0 + int(round(K.T_LIFT * FPS)),
               frame_fire=a0 + int(round(K.T_FIRE * FPS)), frame_burn=a0 + int(round(K.T_BURN * FPS)), W=W, H=H)
    cam = b3.cam_letters(bk, K, K.T_LIFT)
    sxy, _ = cam.project(K.P0)
    rec['sparks'] = [dict(x=round(float(x), 2), y=round(float(y), 2), frame=round(a0 + float(t0) * FPS, 2),
                          arrive=round(a0 + float(t0 + d) * FPS, 2))
                     for (x, y), t0, d in zip(sxy, K.t0, K.d)]
    track = []
    for f in range(SHOTS['letters'][0], SHOTS['letters'][1]):
        t = (f - a0) / FPS
        c = b3.cam_letters(bk, K, t)
        (x, y), z = c.project(K.Fw)
        top, _ = c.project(K.Fw + np.array([0, 0, max(K.flame_h(t), 1e-3)]))
        track.append(dict(frame=f, x=round(float(x), 2), y=round(float(y), 2),
                          flame_px=round(float(math.hypot(top[0] - x, top[1] - y)), 2)))
    rec['heart'] = track
    os.makedirs(out, exist_ok=True)
    p = os.path.join(out, 'x1_letters.json')
    json.dump(rec, open(p, 'w'))
    print(p, len(rec['sparks']), 'sparks')


def frames_of(spec):
    out = []
    for part in spec.split(','):
        part = part.strip()
        if '-' in part:
            a, b = part.split('-')
            out += range(int(a), int(b) + 1)
        elif part:
            out.append(int(part))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('what', choices=['frames', 'export'])
    ap.add_argument('--frames', default='')
    ap.add_argument('--out', default=os.path.join(ROOT, 'renders', 'book_C'))
    ap.add_argument('--scale', type=float, default=1.0)
    ap.add_argument('--with-fire', action='store_true', help="C4-C5 tests: add MAP's own sparks and flame")
    a = ap.parse_args()
    W, H = int(1920 * a.scale), int(804 * a.scale)
    if a.what == 'export':
        export_letters(a.out, W, H)
        return
    b3 = Book3(W, H, with_fire=a.with_fire)
    matte = a.out.rstrip('/') + '_matte'
    for f in frames_of(a.frames):
        t0 = time.time()
        r = b3.frame(f)
        if r is None:
            continue
        hdr, alpha = r
        rgb = look.finish(hdr, exposure=1.15, bloom_strength=0.06, bloom_threshold=1.2, vignette_amount=0.32)
        look.save_png(look.frame_path(a.out, f), rgb)
        look.save_png(look.frame_path(matte, f), np.repeat(np.clip(alpha, 0, 1)[..., None], 3, -1))
        print(f, '%.1fs' % (time.time() - t0), flush=True)


if __name__ == '__main__':
    main()
