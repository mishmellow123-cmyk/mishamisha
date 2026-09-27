"""EMBERS-C: E15 LETTERS TO FIRE (C4-C5, C frames 700-1039): the sparks and the fire over MAP-L's page.

    python e15.py --frames FRAMES [--scale 0.5] [--out DIR]   (FRAMES as render.py: "700-1039", "720,760", "700-1039:4")

Output renders/embers_C3_e15 (C numbering): additive light on black. EDIT: out = book_rgb + (1 - matte) * black + e15
(the director's contract, 27 Sep). MAP-L's book_C is the page only: the letters glowing AS letters, each stroke's
flare-and-go-out, the page lit by the fire, the burn and the hole.

Everything that must line up with the page comes from MAP's own book engine (shots/map/book_c.py, imported once and
cached): its camera for every frame, every stroke's spark seeds (page position, lift time: the instant the stroke
goes out), the page's heart, and its flame-height track (its page lighting follows that track, so this flame does).

    bar 9 b4  (700)  the letters glow (MAP's page)
    bar 10 b1 (720)  the peel from the rim: each stroke's sparks pop off the page and are drawn down by a draught
                     toward the heart (a sink with curl-noise eddies: curving, quickening, never a spiral; each word's
                     sparks travel as one thin stream; some go out on the way); short trails
    ~765-797         they gather: the ember at the heart glows up
    bar 11 b1 (800)  ONE natural flame catches (cflame.Flame, gold and calm) and climbs on MAP's flame track
    bar 11 b3 (840)  MAP burns the page open; the flame burns on in the hole
    C5 880-1039      the fire alone in the black; 880-975 a few letters of the tale are legible in its flames, then
                     burn away; 1039 hands over to E5-C (c3.py opens on this flame, from this camera, at 1040)
"""
import argparse
import hashlib
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
MAPDIR = os.path.join(ROOT, 'shots', 'map')
sys.path.insert(0, os.path.join(ROOT, 'lib'))
sys.path.insert(0, HERE)

import look  # noqa: E402
from core import Frame, Camera, smoothstep, window, rng, snoise, vnoise  # noqa: E402
import cflame as CF  # noqa: E402

OUT = os.path.join(ROOT, 'renders', 'embers_C3_e15')
F0, F1 = 700, 1040                        # the layer's frames
A0 = 560                                  # MAP's shot 'letters' starts here (its t = (f - 560) / 24)
FPS = 24.0
T_FIRE, T_BURN = 800.0, 840.0
TEXT = [(700, 716), (920, 1030)]          # C-T2's tail and C-T5a: calm the lower third (EDIT, 27 Sep)
FLAME_CM = 3.3                            # MAP's flame: (0.3 + 3.0 k) cm, x0.9 once it calms


def to_e(P):
    """MAP world (x, y, z up; cm) -> embers world (x, y up, z)"""
    P = np.asarray(P, np.float64)
    return np.stack([P[..., 0], P[..., 2], -P[..., 1]], -1)


# ================================================================ MAP's data ===

def _key():
    h = hashlib.sha1()
    for f in ('book_c.py', 'book.py', 'pen.py', 'redbook.py', 'pages.py', 'burn.py'):
        p = os.path.join(MAPDIR, f)
        if os.path.exists(p):
            h.update(open(p, 'rb').read())
    h.update(b'e15-v1')
    return h.hexdigest()[:16]


def map_data():
    """camera track, seeds, heart and flame track from MAP's book engine (cached on disk by its sources)"""
    d = os.path.join(OUT, 'cache')
    f = os.path.join(d, 'mapdata_%s.npz' % _key())
    if os.path.exists(f):
        z = np.load(f)
        return {k: z[k] for k in z.files}
    sys.path.insert(0, MAPDIR)
    import book_c
    b3 = book_c.Book3(1920, 804)
    bk, pD, K = b3.letters()
    ts = np.arange(A0, F1 + 1.001, 0.25)
    cp, ct = [], []
    for fq in ts:
        c = b3.cam_letters(bk, K, (fq - A0) / FPS)
        cp.append(c.pos)
        ct.append(c.target)
    fl = np.array([K.flame_h((fq - A0) / FPS) for fq in ts])
    em = np.array([K.ember((fq - A0) / FPS) for fq in ts])
    ug = np.linspace(-2.0, 24.0, 2601)
    Wg = bk.page_to_world('R', ug, np.zeros_like(ug))
    out = dict(ts=ts, cam_pos=np.array(cp), cam_tgt=np.array(ct), hfov=np.array(38.0), flame_h=fl, ember=em,
               ug=ug, ux=Wg[:, 0], uz=Wg[:, 2], PH=np.array(float(bk.PH)),
               seeds=np.asarray(K.pts, np.float64), lift=np.asarray(K.t0, np.float64) * FPS + A0,
               word=np.asarray(K.wi, np.int64), heart_uv=np.asarray(K.F, np.float64),
               heart_w=np.asarray(K.Fw, np.float64), heat=np.asarray(K.heat, np.float64))
    os.makedirs(d, exist_ok=True)
    tmp = f + '.%d.npz' % os.getpid()
    np.savez_compressed(tmp, **out)
    os.replace(tmp, f)
    return out


class Page:
    """page cm (u from the right page's left edge, v down from its head) -> MAP world, via MAP's own table"""

    def __init__(self, D):
        self.ug, self.ux, self.uz, self.PH = D['ug'], D['ux'], D['uz'], float(D['PH'])

    def world(self, u, v, h=0.0):
        x = np.interp(u, self.ug, self.ux)
        z = np.interp(u, self.ug, self.uz) + h
        y = 0.5 * self.PH - v
        return np.stack([x, y, z], -1)


# ================================================================= the draught ===

class Draught:
    """every spark's flight, built once: it pops off the page and is drawn DOWN INTO THE HEART along the draught's
    channels, like sparks in a hearth's draught: never a radial implosion, never a spiral. The draught has three
    curving channels (asymmetric: from the head of the page on the left, from the right margin, from the foot on the
    left); the cheapest way to the heart runs down them, so every word's sparks travel as ONE thin stream that
    curves into the nearest channel, joins it like a tributary and quickens toward the heart (a geodesic cost field,
    Dijkstra on a 1 mm grid; paths by descent). Near the heart the height it rose to goes, so the streams sink into
    the point. Paths are stored in page space (u, v) against each spark's own clock and timed so every spark
    reaches the heart by ~797 (stragglers staggered, not all at once)."""

    DT = 1.0 / 96.0
    T0 = (9.5, 11.2)            # the trunk: every channel joins here and is drawn DOWN into the heart (one side)
    CH = [  # channels: page-cm control points to the trunk (the heart is appended to each)
        [(3.6, 3.2), (5.2, 5.8), (7.2, 8.8), T0],
        [(17.4, 8.6), (14.8, 8.9), (12.0, 9.9), T0],
        [(4.4, 24.8), (4.3, 19.8), (5.3, 15.0), (7.2, 11.9), T0],
        [T0],
    ]

    def __init__(self, D, seed=1515, nmax=360):
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import dijkstra
        r = rng(seed)
        S = D['seeds']
        n = len(S)
        F = np.asarray(D['heart_uv'], np.float64)
        self.n = n
        self.lift = D['lift']
        wd = D['word']
        nw = int(wd.max()) + 1
        wr = rng(seed + 1)
        w_h = wr.uniform(0.25, 0.7, nw)
        self.hmax = w_h[wd] * r.uniform(0.8, 1.25, n)
        # --- the cost field: cheap along the channels (soft 0.4 cm), 1 elsewhere
        du = 0.1
        ug = np.arange(0.0, 22.0 + 1e-6, du)
        vg = np.arange(0.0, 29.0 + 1e-6, du)
        UU, VV = np.meshgrid(ug, vg)
        dmin = np.full(UU.shape, 1e9)
        for ch in self.CH:
            pts = np.array(list(ch) + [tuple(F)], np.float64)
            # smooth the polyline (Catmull-like resample)
            q = np.linspace(0, len(pts) - 1, 200)
            cx = np.interp(q, np.arange(len(pts)), pts[:, 0])
            cy = np.interp(q, np.arange(len(pts)), pts[:, 1])
            for x_, y_ in zip(cx, cy):
                dmin = np.minimum(dmin, (UU - x_) ** 2 + (VV - y_) ** 2)
        dmin = np.sqrt(dmin)
        cost = 1.0 - 0.72 * np.exp(-(dmin / 0.4) ** 2)
        # the heart is entered only down the trunk: a soft wall round it, open along the trunk (so the foot of the
        # page swings round into the trunk instead of fanning straight in)
        tx0, ty0 = self.T0
        vx, vy = F[0] - tx0, F[1] - ty0
        L2 = vx * vx + vy * vy
        k_ = np.clip(((UU - tx0) * vx + (VV - ty0) * vy) / L2, 0.0, 1.0)
        dtr = np.hypot(UU - (tx0 + k_ * vx), VV - (ty0 + k_ * vy))
        rh = np.hypot(UU - F[0], VV - F[1])
        cost = cost + 5.0 * np.exp(-((rh - 1.6) / 0.9) ** 2) * (1.0 - np.exp(-(dtr / 0.55) ** 2))
        H_, W_ = cost.shape
        idx = np.arange(H_ * W_).reshape(H_, W_)
        rows, cols, vals = [], [], []
        for dy, dx in ((0, 1), (1, 0), (1, 1), (1, -1)):
            a = idx[max(0, -dy):H_ - max(0, dy), max(0, -dx):W_ - max(0, dx)]
            b = idx[max(0, dy):H_ - max(0, -dy) if dy <= 0 else H_, max(0, dx):W_ - max(0, -dx) if dx <= 0 else W_]
            b = idx[max(0, dy):max(0, dy) + a.shape[0], max(0, dx):max(0, dx) + a.shape[1]]
            ca = cost.ravel()[a.ravel()]
            cb = cost.ravel()[b.ravel()]
            L = du * math.hypot(dx, dy)
            w = 0.5 * (ca + cb) * L
            rows += [a.ravel(), b.ravel()]
            cols += [b.ravel(), a.ravel()]
            vals += [w, w]
        G = coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(H_ * W_, H_ * W_)).tocsr()
        src = int(round(F[1] / du)) * W_ + int(round(F[0] / du))
        T = dijkstra(G, indices=src).reshape(H_, W_)
        gy, gx = np.gradient(T, du)
        self.T = T

        def grad(P):
            fu = np.clip(P[:, 0] / du, 0, W_ - 1.001)
            fv = np.clip(P[:, 1] / du, 0, H_ - 1.001)
            i0 = fu.astype(int)
            j0 = fv.astype(int)
            a = fu - i0
            b = fv - j0
            def bil(Z):
                return (Z[j0, i0] * (1 - a) * (1 - b) + Z[j0, i0 + 1] * a * (1 - b) + Z[j0 + 1, i0] * (1 - a) * b
                        + Z[j0 + 1, i0 + 1] * a * b)
            return np.stack([bil(gx), bil(gy)], 1)
        # --- one path per word, from its centroid, by descent
        cen = np.zeros((nw, 2))
        for w in range(nw):
            m = wd == w
            cen[w] = S[m].mean(0) if m.any() else F
        paths = []
        P = cen.copy()
        steps = 900
        trk = np.zeros((nw, steps, 2))
        for k in range(steps):
            trk[:, k] = P
            g = grad(P)
            gn = np.maximum(np.linalg.norm(g, axis=1), 1e-9)
            d = np.hypot(P[:, 0] - F[0], P[:, 1] - F[1])
            st = np.minimum(0.05, d)
            P = P - g / gn[:, None] * st[:, None]
        # smooth each track and resample by arc length
        ns_ = nmax
        Wp = np.zeros((nw, ns_, 2))
        Ws = np.zeros((nw, ns_))
        for w in range(nw):
            tr = trk[w]
            k = np.convolve(np.ones(9) / 9.0, np.ones(1), 'full')
            sm = np.stack([np.convolve(np.pad(tr[:, j], 4, mode='edge'), np.ones(9) / 9.0, 'valid') for j in range(2)], 1)
            sm[0] = tr[0]
            seg = np.hypot(*np.diff(sm, axis=0).T)
            cs = np.concatenate([[0], np.cumsum(seg)])
            Ltot = max(cs[-1], 1e-3)
            q = np.linspace(0, Ltot, ns_)
            Wp[w] = np.stack([np.interp(q, cs, sm[:, 0]), np.interp(q, cs, sm[:, 1])], 1)
            Wp[w, -1] = F
            Ws[w] = q
        # --- every spark: its word's stream, entered from its own stroke within the first 1.2 cm, a small offset
        # across the stream that closes toward the heart, a flutter of eddies; the clock quickens toward the heart
        off = r.normal(0, 0.16, n)
        spd = r.uniform(0.9, 1.1, n)
        pos = np.zeros((n, ns_, 2), np.float32)
        tt = np.zeros((n, ns_), np.float32)
        for w in range(nw):
            m = np.nonzero(wd == w)[0]
            if not len(m):
                continue
            path = Wp[w]
            q = Ws[w]
            tan = np.gradient(path, axis=0)
            tan /= np.maximum(np.linalg.norm(tan, axis=1), 1e-9)[:, None]
            nrm = np.stack([-tan[:, 1], tan[:, 0]], 1)
            d = np.hypot(path[:, 0] - F[0], path[:, 1] - F[1])
            speed = 1.6 + 7.0 / (d + 0.9)                       # cm / s: it quickens as the draught narrows
            dt = np.concatenate([[0], np.diff(q) / (0.5 * (speed[1:] + speed[:-1]))])
            tq = np.cumsum(dt)
            join = 1.0 - smoothstep(0.0, 1.2, q)
            close = np.clip(d / 3.0, 0.0, 1.0)
            for j in m:
                pp = path + (S[j] - cen[w])[None, :] * join[:, None] + nrm * (off[j] * close)[:, None]
                pos[j] = pp
                tt[j] = tq / spd[j]
        # eddies: a slowly varying flutter across the stream (divergence-free noise), fading at the heart
        Q = np.stack([pos[..., 0].ravel() * 0.35, pos[..., 1].ravel() * 0.35, np.repeat(wd * 0.37, ns_)], 1)
        fl = snoise(Q, 1.0, (0.0, 0.0, 0.0), 2).reshape(n, ns_)
        dd = np.hypot(pos[..., 0] - F[0], pos[..., 1] - F[1])
        amp = 0.22 * np.clip((dd - 0.3) / 3.0, 0.0, 1.0)
        for w in range(nw):
            m = np.nonzero(wd == w)[0]
            if not len(m):
                continue
            tan = np.gradient(Wp[w], axis=0)
            tan /= np.maximum(np.linalg.norm(tan, axis=1), 1e-9)[:, None]
            nrm = np.stack([-tan[:, 1], tan[:, 0]], 1)
            pos[m] += (nrm[None, :, :] * (fl[m] * amp[m])[..., None]).astype(np.float32)
        self.pos = pos
        self.tt = tt
        T_ = tt[:, -1].astype(np.float64)
        # arrive by ~797 (a hair before the catch): squeeze the slow ones; the rest keep their own pace, so they
        # arrive staggered (the stream keeps coming until the catch)
        avail = np.maximum((T_FIRE - 3.0 - self.lift) / FPS, 0.35)
        right = cen[wd, 0] > 11.5                    # (the right margin's streams arrive first: no pair of wings at the end)
        avail = np.where(right, np.maximum(avail - 0.55, 0.3), avail)
        self.sc = np.minimum(1.0, avail / np.maximum(T_, 1e-6))
        self.dur = T_ * self.sc
        self.E = r.lognormal(0, 0.45, n)
        self.flk_f = r.uniform(5.0, 13.0, n)
        self.flk_p = r.uniform(0, 2 * np.pi, n)
        die = r.random(n) < 0.24
        self.die_at = np.where(die, r.uniform(0.25, 0.85, n), 2.0)
        cd = np.hypot(cen[wd, 0] - F[0], cen[wd, 1] - F[1])
        self.die_at = np.where(cd < 2.4, r.uniform(0.08, 0.2, n), self.die_at)
        self.col_t = r.uniform(0.58, 0.72, n)
        self.sz = r.uniform(0.8, 1.3, n)
        self.F = F

    def at(self, t_frames, idx):
        """page (u, v, h) of sparks idx at C frame t (clamped to their flights) and their progress 0..1"""
        s = (t_frames - self.lift[idx]) / FPS / self.sc[idx]
        tm = self.tt[idx]
        k = np.clip((tm <= s[:, None]).sum(1) - 1, 0, tm.shape[1] - 2)
        rI = np.arange(len(idx))
        ta, tb = tm[rI, k], tm[rI, k + 1]
        w = np.clip((s - ta) / np.maximum(tb - ta, 1e-6), 0, 1)[:, None]
        uv = self.pos[idx, k] * (1 - w) + self.pos[idx, k + 1] * w
        prog = np.clip(s * self.sc[idx] / np.maximum(self.dur[idx], 1e-6), 0, 1)
        rr = np.hypot(uv[:, 0] - self.F[0], uv[:, 1] - self.F[1])
        pop = smoothstep(0.0, 0.25, s)
        h = 0.02 + self.hmax[idx] * pop * np.clip(rr / 2.6, 0.0, 1.0) ** 0.8
        return uv, h, prog


# ================================================================== the layer ===

class E15:
    def __init__(self):
        self.D = map_data()
        self.page = Page(self.D)
        self.dr = Draught(self.D)
        self.tip_sparks = CF.Sparks()
        self.heart = to_e(self.D['heart_w'])
        self._glyphs = None

    # ------------------------------------------------------------- camera
    def camera(self, t):
        D = self.D
        pos = np.stack([np.interp(t, D['ts'], D['cam_pos'][:, j]) for j in range(3)])
        tgt = np.stack([np.interp(t, D['ts'], D['cam_tgt'][:, j]) for j in range(3)])
        pe, te = to_e(pos), to_e(tgt)
        return Camera(pe, te, hfov=float(D['hfov']), focus=float(np.linalg.norm(self.heart - pe)), aperture=0.02)

    def flame_h(self, t):
        return float(np.interp(t, self.D['ts'], self.D['flame_h']))

    def ember(self, t):
        return float(np.interp(t, self.D['ts'], self.D['ember']))

    # ------------------------------------------------------------- sparks
    def sparks(self, ctx):
        t = ctx.t
        dr = self.dr
        live = (t >= dr.lift - 0.5) & (t < dr.lift + dr.dur * FPS + 0.5)
        if not live.any():
            return
        idx = np.nonzero(live)[0]
        uv0, h0, _ = dr.at(np.full(len(idx), ctx.t0), idx)
        uv1, h1, prog = dr.at(np.full(len(idx), ctx.t1), idx)
        P0 = to_e(self.page.world(uv0[:, 0], uv0[:, 1], h0))
        P1 = to_e(self.page.world(uv1[:, 0], uv1[:, 1], h1))
        age = (t - dr.lift[idx]) / FPS
        born = smoothstep(0.0, 0.05, age)
        flare = np.exp(-age / 0.07)                                   # the stroke's own flash as it lets go
        fl = 1.0 + 0.3 * np.sin(dr.flk_f[idx] * age * 2 * np.pi + dr.flk_p[idx])
        dying = 1.0 - smoothstep(dr.die_at[idx] - 0.12, dr.die_at[idx], prog)
        near = 1.0 + 0.8 * smoothstep(0.75, 1.0, prog)
        arrive = 1.0 - smoothstep(0.93, 1.0, prog)
        e = dr.E[idx] * born * fl * dying * near * arrive * (1.0 + 2.5 * flare) * 11.0
        T = np.clip(dr.col_t[idx] + 0.12 * flare + 0.08 * smoothstep(0.6, 1.0, prog) - 0.2 * (1.0 - dying), 0.35, 0.95)
        col = look.blackbody(T)
        rw = 0.008 * dr.sz[idx]
        ctx.fr.splat(P0, P1, rw, e, col, ctx.cam0, ctx.cam1)
        # short trails: where each was a moment ago
        for lag, g in ((0.9, 0.5), (1.9, 0.22)):
            uvA, hA, _ = dr.at(np.full(len(idx), ctx.t0 - lag), idx)
            PA = to_e(self.page.world(uvA[:, 0], uvA[:, 1], hA))
            ok = (ctx.t0 - lag) >= dr.lift[idx]
            ctx.fr.splat(PA[ok], P0[ok], rw[ok] * 0.8, (e * g)[ok], col[ok], ctx.cam0, ctx.cam1)

    # --------------------------------------------------------- the ember
    def ember_glow(self, ctx):
        t = ctx.t
        e = self.ember(t)
        if e <= 0.005:
            return
        C = self.heart + np.array([0.0, 0.04, 0.0])
        H = np.array([C, C, C])
        fl = 1.0 + 0.12 * math.sin(1.9 * t) + 0.07 * math.sin(4.3 * t + 1.0)
        rr = np.array([0.03 + 0.06 * e, 0.16 + 0.2 * e, 0.6])
        ee = np.array([60.0, 150.0, 200.0]) * e * fl
        col = np.array([[1.0, 0.82, 0.5], [1.0, 0.55, 0.18], [1.0, 0.4, 0.1]])
        ctx.fr.splat(H, H, rr, ee, col, ctx.cam0, ctx.cam1, profile=1)

    # ------------------------------------------------------ the letters in the flame
    def glyphs(self):
        """a few letters of the tale (MAP's book hand), as polylines in cm, centred"""
        if self._glyphs is None:
            sys.path.insert(0, MAPDIR)
            import pen
            h = pen.Hand(seed=77, xh=0.22)
            S = pen.Strokes()
            x = 0.0
            words = [h.word(3)]
            lines = []
            for li, w in enumerate(words):
                S2 = pen.Strokes()
                x2 = h.write_word(S2, w, 0.0, 0.0)
                for si, P in enumerate(S2.P):
                    lines.append((P - np.array([0.5 * x2, 0.0]), 1000 * li + si))
            pts, lid = [], []
            for P, li in lines:
                seg = np.hypot(*np.diff(P, axis=0).T)
                L = float(seg.sum())
                m = max(int(L / 0.004), 2)
                s_ = np.concatenate([[0], np.cumsum(seg)])
                q = np.linspace(0, L, m)
                pts.append(np.column_stack([np.interp(q, s_, P[:, 0]), np.interp(q, s_, P[:, 1])]))
                lid.append(np.full(m, li))
            self._glyphs = (np.concatenate(pts), np.concatenate(lid))
        return self._glyphs

    def letters_in_flame(self, ctx):
        """(the old bright splat letters: replaced by letters_mask, drawn into the flame itself)"""
        return

    def letters_mask(self, H, W, root, tip, t, scale):
        """C5: a few letters of the tale, still legible in the flame for a moment (882-972): one short word of the
        book hand, dark as ink against the gold body at 40-55 % of the flame's height, with a thin hot rim; it
        rises slowly with the licks and burns away from below. Returns (mask, rim) (H, W) or None."""
        import cv2
        k = float(smoothstep(882, 898, t)) * (1.0 - float(smoothstep(950, 972, t)))
        if k <= 0:
            return None
        G, lid = self.glyphs()
        x0, y0 = root[0] * scale, root[1] * scale
        dx, dy = (tip[0] - root[0]) * scale, (tip[1] - root[1]) * scale
        Hp = math.hypot(dx, dy)
        if Hp < 20:
            return None
        ax, ay = dx / Hp, dy / Hp
        px, py = -ay, ax
        gx = G[:, 0] - 0.5 * (G[:, 0].max() + G[:, 0].min())
        gy = -G[:, 1]
        gy = gy - 0.5 * (gy.max() + gy.min())
        wide = max(float(gx.max() - gx.min()), 1e-6)
        sc_ = 0.19 / wide                                  # the word sits inside the body (19 % of the flame's height) ...
        yc = 0.37 + 0.0016 * (t - 882.0)                   # it rises slowly with the licks
        fx = gx * sc_
        fy = yc + gy * sc_ * 2.4                           # ... and the heat draws its letters up tall (legible)
        X = x0 + (ax * fy + px * fx) * Hp
        Y = y0 + (ay * fy + py * fx) * Hp
        burn = smoothstep(944 + 60 * (fy - yc + 0.1), 962 + 60 * (fy - yc + 0.1), t)   # from below
        m = np.zeros((H, W), np.uint8)
        th = max(2, int(round(0.017 * Hp)))
        seg_ok = (1.0 - burn) > 0.5
        pts = np.stack([X, Y], 1)
        for a_ in range(len(pts) - 1):
            if lid[a_] != lid[a_ + 1] or not (seg_ok[a_] and seg_ok[a_ + 1]):
                continue
            if abs(pts[a_ + 1, 0] - pts[a_, 0]) + abs(pts[a_ + 1, 1] - pts[a_, 1]) > 0.05 * Hp:
                continue                                    # (a pen lift)
            cv2.line(m, (int(round(pts[a_, 0] * 4)), int(round(pts[a_, 1] * 4))),
                     (int(round(pts[a_ + 1, 0] * 4)), int(round(pts[a_ + 1, 1] * 4))), 255, th, cv2.LINE_AA, 2)
        mf = cv2.GaussianBlur(m.astype(np.float32) / 255.0, (0, 0), 0.5)
        rim = np.clip(cv2.GaussianBlur(mf, (0, 0), 1.2 + 0.004 * Hp) - mf, 0, 1)
        return mf * k, rim * k

    # ------------------------------------------------------------- frame
    def render(self, f, scale=1.0):
        fr = Frame(scale)

        class Ctx:
            pass
        ctx = Ctx()
        ctx.f, ctx.t, ctx.t0, ctx.t1, ctx.fr, ctx.scale = f, float(f), f - 0.25, f + 0.25, fr, scale
        ctx.cam0, ctx.cam1, ctx.cam = self.camera(ctx.t0), self.camera(ctx.t1), self.camera(float(f))
        k = 0.0
        for a, b in TEXT:
            k = max(k, float(window(f, a, b, 10, 10)))
        fr.set(focus=ctx.cam.focus, aperture=ctx.cam.aperture, band=(560, 700, 0.62 * k), bokeh_pow=0.2,
               bokeh_cap=1.5, near=0.05)
        self.sparks(ctx)
        self.ember_glow(ctx)
        h = self.flame_h(float(f))
        if h > 0.01:
            self.tip_sparks.emit(ctx, self.heart, h, amount=float(smoothstep(T_FIRE + 20, T_FIRE + 60, f)))
        self.letters_in_flame(ctx)
        hdr = fr.resolve()
        hdr = np.nan_to_num(hdr, nan=0.0, posinf=0.0, neginf=0.0)
        if h > 0.01:
            # the flame itself (cflame.draw; c3 draws the same flame from 1040): on MAP's flame track
            import c3
            bright = c3.fire_bright(float(f)) * (1.0 + 0.35 * math.exp(-max(f - T_FIRE, 0.0) / 8.0))
            cam = ctx.cam
            u, v, _ = cam.project(np.stack([self.heart, self.heart + np.array([0.0, h, 0.0])]), 1920, 804)
            fl = np.zeros_like(hdr)
            CF.draw(fl, (u[0], v[0]), (u[1], v[1]), float(f), bright=bright, calm=1.0, scale=scale)
            lm = self.letters_mask(fl.shape[0], fl.shape[1], (u[0], v[0]), (u[1], v[1]), float(f), scale)
            if lm is not None:
                mk, rim = lm
                lum = fl.max(axis=2)
                body = np.clip((lum / max(float(lum.max()), 1e-6) - 0.12) / 0.3, 0.0, 1.0)   # never ink on the halo
                fl = fl * (1.0 - 0.74 * mk * body)[..., None]
                fl = fl + (np.array([1.0, 0.5, 0.12], np.float32)[None, None, :] * (1.4 * rim * body * lum)[..., None])
            if k > 0:
                fl = fl * band_rows(fl.shape[0], scale, 0.62 * k)[:, None, None]
            hdr = hdr + fl
        return look.finish(hdr, exposure=1.0, bloom_strength=0.14, bloom_threshold=0.7, vignette_amount=0.0,
                           lift=0.0)


def band_rows(H, scale, k):
    """the text band's calm (render.band_k's profile: y 560-700 full-res, soft 45 px edges), per row"""
    y = np.arange(H, dtype=np.float32) / scale
    a = np.clip((y - (560 - 45)) / 90.0, 0, 1)
    a = a * a * (3 - 2 * a)
    b = np.clip((y - (700 - 45)) / 90.0, 0, 1)
    b = b * b * (3 - 2 * b)
    return (1.0 - k * a * (1.0 - b)).astype(np.float32)


def parse_frames(s):
    out = []
    for part in s.split(','):
        step = 1
        if ':' in part:
            part, st = part.split(':')
            step = int(st)
        if '-' in part:
            a, b = part.split('-')
            out += list(range(int(a), int(b) + 1, step))
        else:
            out.append(int(part))
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('frames', nargs='?')
    ap.add_argument('--frames', dest='frames_opt', default=None, help='same as the positional (farm.py rewrites it)')
    ap.add_argument('--scale', type=float, default=1.0)
    ap.add_argument('--out', default=OUT)
    a = ap.parse_args()
    a.frames = a.frames_opt or a.frames
    import variant
    variant.set_cut('C3')
    L = E15()
    os.makedirs(a.out, exist_ok=True)
    for f in parse_frames(a.frames):
        t0 = time.time()
        img = L.render(f, a.scale)
        look.save_png(look.frame_path(a.out, f), img)
        print(f'frame {f} scale {a.scale} {time.time() - t0:.2f}s', flush=True)
