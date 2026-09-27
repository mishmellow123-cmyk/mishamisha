"""EMBERS-C: E15 LETTERS TO FIRE (C4-C5, C frames 700-1039): the sparks and the fire over MAP-L's page.

    python e15.py FRAMES [--scale 0.5] [--out DIR]          (FRAMES as render.py: "700-1039", "720,760", "700-1039:4")

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
    """every spark's flight, integrated once: it pops off the page, then a draught draws it down toward the heart:
    a sink whose pull quickens near the heart, plus eddies (the curl of a slowly varying noise: they meander but
    never gather into a vortex); the height it rose to fades as it nears the heart. Paths are stored in page space
    (u, v, h) against each spark's own clock, then timed so that every spark reaches the heart by ~797."""

    DT = 1.0 / 96.0

    def __init__(self, D, seed=1515, nmax=520):
        r = rng(seed)
        S = D['seeds']
        n = len(S)
        F = D['heart_uv']
        self.n = n
        self.lift = D['lift']
        wd = D['word']
        nw = int(wd.max()) + 1
        wr = rng(seed + 1)
        # each word travels as one stream: its sparks share an eddy phase and a height
        w_phase = wr.uniform(0, 50, nw)
        w_h = wr.uniform(0.12, 0.45, nw)
        self.hmax = w_h[wd] * r.uniform(0.8, 1.25, n)
        P = S.copy()
        tph = w_phase[wd] + 0.05 * self.lift
        pos = np.zeros((n, nmax, 2), np.float32)
        tt = np.zeros((n, nmax), np.float32)
        acc = np.zeros(n)
        done = np.zeros(n, bool)
        # the pop: a little kick off the page, outward from the heart and at random
        d0 = S - F
        r0 = np.maximum(np.hypot(d0[:, 0], d0[:, 1]), 1e-6)
        kick = (d0 / r0[:, None]) * 0.25 + r.normal(0, 0.35, (n, 2))
        h_eps = 0.03
        for k in range(nmax):
            pos[:, k] = P
            tt[:, k] = acc
            dx = P - F
            rr = np.hypot(dx[:, 0], dx[:, 1])
            done |= rr < 0.1
            rhat = -dx / np.maximum(rr, 1e-6)[:, None]
            # eddies: the curl of a noise stream-function (divergence-free)
            Q = np.stack([P[:, 0] * 0.33, P[:, 1] * 0.33, tph * 0.02], 1)
            e = 0.02
            a1 = snoise(Q + np.array([e, 0, 0]), 1.0, (0, 0, 0), 2)
            a2 = snoise(Q - np.array([e, 0, 0]), 1.0, (0, 0, 0), 2)
            b1 = snoise(Q + np.array([0, e, 0]), 1.0, (0, 0, 0), 2)
            b2 = snoise(Q - np.array([0, e, 0]), 1.0, (0, 0, 0), 2)
            dpsi_du = (a1 - a2) / (2 * e) * 0.33
            dpsi_dv = (b1 - b2) / (2 * e) * 0.33
            curl = np.stack([dpsi_dv, -dpsi_du], 1)
            amp = 2.6 * np.clip((rr - 0.4) / 5.0, 0.0, 1.0)
            speed = 2.4 + 16.0 / (rr + 1.4)                          # cm / s: quicker near the heart
            v = rhat * speed[:, None] + curl * amp[:, None]
            if k < 18:                                               # the pop fades into the draught
                v = v + kick * (1.0 - k / 18.0) * 3.0
            v = np.where(done[:, None], 0.0, v)
            P = P + v * self.DT
            acc = acc + np.where(done, 0.0, self.DT)
        self.pos = pos
        self.tt = tt
        T = tt[:, -1].astype(np.float64)
        # arrive by ~797 (a hair before the catch): squeeze the slow ones
        avail = np.maximum((T_FIRE - 3.0 - self.lift) / FPS, 0.35)
        self.sc = np.minimum(1.0, avail / np.maximum(T, 1e-6))
        self.dur = T * self.sc                                       # seconds, lift -> heart
        self.E = r.lognormal(0, 0.45, n)
        self.flk_f = r.uniform(5.0, 13.0, n)
        self.flk_p = r.uniform(0, 2 * np.pi, n)
        die = r.random(n) < 0.24
        self.die_at = np.where(die, r.uniform(0.25, 0.85, n), 2.0)   # some go out on the way
        self.col_t = r.uniform(0.58, 0.72, n)
        self.sz = r.uniform(0.8, 1.3, n)
        self.F = F

    def at(self, t_frames, idx):
        """page (u, v, h) of sparks idx at C frame t (clamped to their flights) and their progress 0..1"""
        s = (t_frames - self.lift[idx]) / FPS / self.sc[idx]               # own clock (path time)
        tm = self.tt[idx]
        k = np.clip((tm <= s[:, None]).sum(1) - 1, 0, tm.shape[1] - 2)
        rI = np.arange(len(idx))
        ta, tb = tm[rI, k], tm[rI, k + 1]
        w = np.clip((s - ta) / np.maximum(tb - ta, 1e-6), 0, 1)[:, None]
        uv = self.pos[idx, k] * (1 - w) + self.pos[idx, k + 1] * w
        prog = np.clip(s * self.sc[idx] / np.maximum(self.dur[idx], 1e-6), 0, 1)
        rr = np.hypot(uv[:, 0] - self.F[0], uv[:, 1] - self.F[1])
        pop = smoothstep(0.0, 0.2, s)
        h = 0.02 + self.hmax[idx] * pop * np.clip(rr / 2.2, 0.0, 1.0) ** 0.8
        return uv, h, prog


# ================================================================== the layer ===

class E15:
    def __init__(self):
        self.D = map_data()
        self.page = Page(self.D)
        self.dr = Draught(self.D)
        self.flame = CF.Flame()
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
        e = dr.E[idx] * born * fl * dying * near * arrive * (1.0 + 2.5 * flare) * 0.022
        T = np.clip(dr.col_t[idx] + 0.12 * flare + 0.08 * smoothstep(0.6, 1.0, prog) - 0.2 * (1.0 - dying), 0.35, 0.95)
        col = look.blackbody(T)
        rw = 0.006 * dr.sz[idx]
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
        ee = np.array([2.2, 1.2, 0.5]) * e * fl * 0.35
        col = np.array([[1.0, 0.82, 0.5], [1.0, 0.55, 0.18], [1.0, 0.4, 0.1]])
        ctx.fr.splat(H, H, rr, ee, col, ctx.cam0, ctx.cam1, profile=1)

    # ------------------------------------------------------ the letters in the flame
    def glyphs(self):
        """a few letters of the tale (MAP's book hand), as polylines in cm, centred"""
        if self._glyphs is None:
            sys.path.insert(0, MAPDIR)
            import pen
            h = pen.Hand(seed=77, xh=0.34)
            S = pen.Strokes()
            x = 0.0
            words = [h.word(3), h.word(2)]
            lines = []
            for li, w in enumerate(words):
                S2 = pen.Strokes()
                x2 = h.write_word(S2, w, 0.0, 0.0)
                for P in S2.P:
                    lines.append((P - np.array([0.5 * x2, 0.0]), li))
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
        t = ctx.t
        k = float(smoothstep(882, 896, t)) * (1.0 - float(smoothstep(944, 972, t)))
        if k <= 0:
            return
        G, lid = self.glyphs()
        cam = ctx.cam
        right, up = cam.R[0], np.array([0.0, 1.0, 0.0])
        right = right - (right @ up) * up
        right /= np.linalg.norm(right)
        sc_ = self.flame_h(t) / FLAME_CM

        def place(tq):
            rise = 0.012 * (tq - 882.0)
            y0 = np.where(lid == 0, 1.45, 0.95) * sc_ + rise                # two short lines, one above the other
            x = G[:, 0]
            y = -G[:, 1] + y0
            wob = vnoise(np.column_stack([x, y, np.full(len(x), 0.02 * tq)]), 1.2, (2.0, 5.0, 1.0), 1)
            return self.heart[None, :] + right[None, :] * (x + 0.03 * wob[:, 0])[:, None] + up[None, :] * (y + 0.02 * wob[:, 1])[:, None]
        P0 = place(ctx.t0)
        P1 = place(ctx.t1)
        # they burn away from below as the flame takes them
        yl = -G[:, 1]
        burn = smoothstep(944 + 10 * (yl + 0.3), 960 + 10 * (yl + 0.3), t)
        e = k * (1.0 - burn) * 0.0065 * (1.0 + 0.15 * math.sin(2.3 * t))
        col = np.array([1.0, 0.86, 0.55])
        ctx.fr.splat(P0, P1, 0.012, np.full(len(G), e), col, ctx.cam0, ctx.cam1)

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
            import c3
            bright = c3.fire_bright(float(f))
            self.flame.emit(ctx, self.heart, FLAME_CM, bright=bright, height=h / FLAME_CM, calm=1.0,
                            sparks=float(smoothstep(T_FIRE + 20, T_FIRE + 60, f)))
        self.letters_in_flame(ctx)
        hdr = fr.resolve()
        hdr = np.nan_to_num(hdr, nan=0.0, posinf=0.0, neginf=0.0)
        return look.finish(hdr, exposure=1.0, bloom_strength=0.14, bloom_threshold=0.7, vignette_amount=0.0,
                           lift=0.0)


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
    ap.add_argument('frames')
    ap.add_argument('--scale', type=float, default=1.0)
    ap.add_argument('--out', default=OUT)
    a = ap.parse_args()
    import variant
    variant.set_cut('C3')
    L = E15()
    os.makedirs(a.out, exist_ok=True)
    for f in parse_frames(a.frames):
        t0 = time.time()
        img = L.render(f, a.scale)
        look.save_png(look.frame_path(a.out, f), img)
        print(f'frame {f} scale {a.scale} {time.time() - t0:.2f}s', flush=True)
