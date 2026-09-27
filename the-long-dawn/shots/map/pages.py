"""THE INK PAGES (cut C, P2) and the Red Book's written leaves (P1), drawn stroke by stroke.

Each page is built as pen.Strokes in page cm (20 x 29, y down), in the order an illustrator works (frame,
the big contour, the dark accents, hatching from the top down, the sky, the ground), with draw windows in
shot seconds, so any frame of the drawing-on is a pure function of time. Every page has two states: PENCIL
(the underdrawing: construction lines, sketchy repeated contours, loose shading) and INK (iron-gall pen,
hatching that follows the form and the light, gold leaf only where the story puts it).

The drawings' own light comes from the upper left, like the hearth that lights the page.
"""
import math

import numpy as np

import pen
from pen import INK, PENCIL, GILT, Strokes, hand, hatch, line, stipple, catmull, resample, arclen

PW, PH = 20.0, 29.0


def _poly_inside(P):
    """Vectorised point-in-polygon for a closed polyline P (N,2)."""
    P = np.asarray(P, np.float64)
    x0, y0 = P[:, 0], P[:, 1]
    x1, y1 = np.roll(x0, -1), np.roll(y0, -1)

    def f(x, y):
        x = np.asarray(x)[..., None]
        y = np.asarray(y)[..., None]
        c = ((y0 > y) != (y1 > y)) & (x < (x1 - x0) * (y - y0) / (y1 - y0 + 1e-12) + x0)
        return (np.sum(c, -1) % 2) == 1
    return f


def billows(c, w, rng, grow=(0.5, 1.05), step=0.4, jitter=0.4, res=0.01):
    """Smoke as a union of puffs along a centreline c (N,2) with half-width w (N,): returns the circles
    [(x, y, r)] in drawing order (back to front) and the outer outline(s) as polylines (page cm)."""
    import cv2
    s = arclen(c)
    circles = []
    q = 0.0
    while q < s[-1]:
        i = min(np.searchsorted(s, q), len(c) - 1)
        nrm = pen.normals(c)[i]
        for k in range(2):
            r = w[i] * rng.uniform(*grow) * (0.62 if k else 1.0)
            off = nrm * w[i] * rng.uniform(-jitter, jitter) + (nrm * w[i] * 0.55 * (1 if rng.random() < 0.5 else -1) if k else 0)
            circles.append((c[i, 0] + off[0], c[i, 1] + off[1], r))
        q += step * w[i] * rng.uniform(0.7, 1.2) + 0.08
    xs = [x for x, y, r in circles]
    ys = [y for x, y, r in circles]
    rs = [r for x, y, r in circles]
    x0, y0 = min(np.subtract(xs, rs)) - 0.1, min(np.subtract(ys, rs)) - 0.1
    x1, y1 = max(np.add(xs, rs)) + 0.1, max(np.add(ys, rs)) + 0.1
    H, W = int((y1 - y0) / res) + 1, int((x1 - x0) / res) + 1
    m = np.zeros((H, W), np.uint8)
    for x, y, r in circles:
        cv2.circle(m, (int((x - x0) / res), int((y - y0) / res)), int(r / res), 255, -1, lineType=cv2.LINE_8)
    cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    outl = []
    for cc in cs:
        P = cc[:, 0, :].astype(np.float64) * res + [x0, y0]
        if len(P) > 20:
            outl.append(resample(np.vstack([P, P[:1]]), 0.02))
    return circles, outl


def ridgeline(xa, xb, ybase, hmax, rng, n=7, rough=0.55):
    """A jagged far range: midpoint displacement between a few peaks (page cm, y up is negative)."""
    k = rng.integers(3, 5)
    xs = np.linspace(xa, xb, k + 2)
    ys = np.concatenate([[ybase], ybase - hmax * rng.uniform(0.35, 1.0, k), [ybase]])
    P = np.column_stack([xs, ys])
    amp = hmax * 0.5
    for _ in range(n):
        Q = [P[0]]
        for a, b in zip(P[:-1], P[1:]):
            m = 0.5 * (a + b)
            m[1] += rng.normal(0, amp)
            m[1] = min(m[1], ybase)
            Q += [m, b]
        P = np.array(Q)
        amp *= rough
    return P


def frame_rules(S, box, seed, t0=0.0, t1=0.0, layer=INK):
    """A plate's double rule, drawn with a ruler but by hand: slightly uneven weight, corners overshoot."""
    x0, y0, x1, y1 = box
    rng = np.random.default_rng(seed)
    k0 = len(S)
    for inset, w in ((0.0, 0.034), (0.16, 0.012)):
        a, b, c, d = x0 + inset, y0 + inset, x1 - inset, y1 - inset
        for (p, q) in (((a - 0.05, b), (c + 0.05, b)), ((c, b - 0.05), (c, d + 0.05)),
                       ((c + 0.05, d), (a - 0.05, d)), ((a, d + 0.05), (a, b - 0.05))):
            pts = np.array([p, q])
            pp, rr, dd = hand(pts, w, int(rng.integers(1 << 30)), slow=(6.0, 0.006), fast=(0.4, 0.0015),
                              taper=(0.02, 0.02), dens=0.95, pool=0.15, thin_end=0.6, press=0.08)
            S.add(pp, rr, dd, layer=layer)
    if t1 > t0:
        _window(S, k0, t0, t1, overlap=0.2)


def _window(S, k0, t0, t1, overlap=0.0, order=None):
    """Retime strokes k0.. of S into [t0, t1]."""
    idx = list(range(k0, len(S))) if order is None else [k0 + i for i in order]
    if not idx:
        return
    L = np.array([arclen(S.P[i])[-1] for i in idx]) + 0.05
    span = (t1 - t0) / (L.sum() * (1.0 - overlap) + L[-1] * overlap)
    t = t0
    for k, i in enumerate(idx):
        d = L[k] * span
        S.T0[i], S.T1[i] = t, t + d
        t += d * (1.0 - overlap)


# ============================================================ THE MOUNTAIN ===

class Mountain:
    """A mountain with a fire in its throat, and in the fire a small ring: the only gold on the page.

    Plate box on the recto (page cm); the mountain is a concave volcanic cone lit from the upper left,
    ridged and gullied; smoke rolls out of the throat and away to the right; an engraver's sky of ruled
    lines; a plain with a road that climbs to a door in the flank."""

    def __init__(self, seed=11):
        self.seed = seed
        self.box = (2.3, 2.6, 17.3, 12.2)            # the plate
        self.caption = (2.3, 12.8, 17.3, 14.2)       # left clear for T1 (EDIT's ink write-on)
        bx0, by0, bx1, by1 = self.box
        self.Xc = bx0 + 8.3
        self.Yb = by0 + 7.55                          # base of the cone
        self.Ys = by0 + 2.25                          # summit (crater rim)
        self.Wb = 4.9                                 # half-width at the base
        self.Wt = 0.62                                # half-width of the rim
        self.Yh = by0 + 6.85                          # horizon
        self.ring = (self.Xc + 0.02, self.Ys - 0.42)  # the ring, in the fire
        rng = np.random.default_rng(seed)
        self.rphase = rng.uniform(0, 6.28, 6)
        self.light = np.array([-0.62, -0.55, 0.56])   # page space: from the upper left (y down), toward us
        self.light /= np.linalg.norm(self.light)

    # --- the cone
    def half_width(self, h, phi=0.0):
        """Half-width at height fraction h (0 base .. 1 rim), with ridges by azimuth phi."""
        h = np.clip(h, 0.0, 1.0)
        base = self.Wt + (self.Wb - self.Wt) * (1.0 - h) ** 1.55
        # a shoulder on the left low down, the right flank a little steeper
        sh = 0.3 * np.exp(-((h - 0.28) / 0.12) ** 2) * (np.asarray(phi) < 0)
        rid = 0.035 * np.sin(5.0 * phi + self.rphase[0]) + 0.02 * np.sin(11.0 * phi + self.rphase[1])
        return base * (1.0 + rid * (0.3 + h)) + sh * (1.0 - h) - 0.18 * (np.asarray(phi) > 0) * h * (1 - h)

    def surf(self, h, phi):
        """Page point of the surface at height h, azimuth phi (0 faces us, +-pi/2 the silhouettes)."""
        w = self.half_width(h, phi)
        x = self.Xc + w * np.sin(phi)
        y = self.Yb - (self.Yb - self.Ys) * h - 0.22 * w * (1.0 - np.cos(phi)) * 0.35
        return x, y

    def tone(self, h, phi):
        """Darkness 0..1 from the light (upper left): the shadow side darkens toward its silhouette, the
        corrugation of ridges and gullies swings the tone, and the foot of the cone is a little darker."""
        h = np.clip(h, 0, 1)
        phi = np.asarray(phi, np.float64)
        side = np.clip(np.sin(phi) * 0.5 + 0.5, 0, 1)             # 0 lit silhouette .. 1 shadow silhouette
        cor = np.sin(7.0 * phi + 3.0 * h + self.rphase[2]) * (0.3 + 0.3 * h)
        t = 0.05 + 0.95 * side ** 1.4 + 0.22 * cor * (0.4 + side) + 0.12 * (1 - h) ** 3
        return np.clip(t, 0, 1)

    def _tone_old(self, h, phi):
        slope = 0.5 + 0.9 * h                         # steeper toward the top (radians-ish)
        nx = np.sin(phi) * np.cos(slope) * 0.0 + np.sin(phi) * np.sin(slope)
        nz = np.cos(phi) * np.sin(slope)
        ny = -np.cos(slope)                           # up (page y is down)
        # ridges: normals swing with the corrugation
        cor = np.sin(9.0 * phi + 2.0 * h + self.rphase[2]) * (0.25 + 0.35 * h)
        nx = nx + 0.35 * cor * np.cos(phi)
        nn = np.sqrt(nx * nx + ny * ny + nz * nz)
        lam = (nx * self.light[0] + ny * self.light[1] + nz * self.light[2]) / nn
        t = 1.0 - np.clip(lam, 0, 1)
        return np.clip(0.08 + 1.05 * (t - 0.25), 0, 1)

    def silhouette(self, n=220):
        hs = np.linspace(0, 1, n)
        L = np.array([self.surf(h, -np.pi / 2) for h in hs])
        R = np.array([self.surf(h, np.pi / 2) for h in hs])
        return L, R

    def inside(self, x, y):
        """Inside the cone's silhouette (below the rim)."""
        x = np.asarray(x, np.float64)
        y = np.asarray(y, np.float64)
        h = np.clip((self.Yb - y) / (self.Yb - self.Ys), 0, 1)
        side = np.where(x < self.Xc, -np.pi / 2, np.pi / 2)
        w = self.half_width(h, side)
        return (np.abs(x - self.Xc) < w) & (y <= self.Yb + 0.05) & (y >= self.Ys - 0.02)

    def plume(self):
        """Centreline and half-width of the smoke, out of the throat, up, and away to the right."""
        bx0, by0, bx1, by1 = self.box
        c = catmull([(self.Xc + 0.05, self.Ys - 0.9), (self.Xc + 0.25, self.Ys - 1.5), (self.Xc + 1.1, by0 + 1.3),
                     (self.Xc + 3.0, by0 + 1.05), (self.Xc + 5.2, by0 + 1.2), (bx1 - 0.6, by0 + 1.45)], 12)
        c = resample(c, 0.05)
        t = arclen(c) / arclen(c)[-1]
        w = (0.28 + 0.7 * t ** 0.7) * (1.0 - 0.45 * np.clip((t - 0.75) / 0.25, 0, 1))
        return c, w

    def build(self, mode='ink', t0=0.0, t1=8.0):
        """Strokes for the plate, drawn on across [t0, t1] (mode 'ink' or 'pencil')."""
        rng = np.random.default_rng(self.seed)
        S = Strokes()
        lay = INK if mode == 'ink' else PENCIL
        pencil = mode == 'pencil'
        bx0, by0, bx1, by1 = self.box
        T = lambda a, b: (t0 + (t1 - t0) * a, t0 + (t1 - t0) * b)
        W = (lambda w: w * 1.15) if pencil else (lambda w: w)
        dens = 0.55 if pencil else 0.95
        fx, fy = self.ring
        c_pl, w_pl = self.plume()
        circles, outl = billows(c_pl, w_pl, np.random.default_rng(self.seed + 5))
        cx_ = np.array([q[0] for q in circles])
        cy_ = np.array([q[1] for q in circles])
        cr_ = np.array([q[2] for q in circles])

        def in_smoke(x, y, pad=0.0):
            x = np.asarray(x, np.float64)[..., None]
            y = np.asarray(y, np.float64)[..., None]
            return np.any((x - cx_) ** 2 + (y - cy_) ** 2 < (cr_ + pad) ** 2, -1)

        # 1. the frame
        k = len(S)
        if pencil:
            for (p, q) in (((bx0 - 0.3, by0), (bx1 + 0.3, by0)), ((bx1, by0 - 0.3), (bx1, by1 + 0.3)),
                           ((bx1 + 0.3, by1), (bx0 - 0.3, by1)), ((bx0, by1 + 0.3), (bx0, by0 - 0.3))):
                pp, rr, dd = hand(np.array([p, q]), 0.02, int(rng.integers(1 << 30)), slow=(5.0, 0.01),
                                  dens=0.35, thin_end=0.5)
                S.add(pp, rr, dd, layer=lay)
        else:
            frame_rules(S, self.box, self.seed + 1, layer=lay)
        _window(S, k, *T(0.0, 0.07), overlap=0.3)

        # pencil construction: the axis, the horizon, the base and rim ellipses, the smoke's puffs
        if pencil:
            k = len(S)
            cons = [np.array([[self.Xc, self.Ys - 2.4], [self.Xc, self.Yb + 0.6]]),
                    np.array([[bx0 + 0.2, self.Yh], [bx1 - 0.2, self.Yh]])]
            for (cy, rx, ry) in ((self.Yb, self.Wb, 0.5), (self.Ys, self.Wt, 0.16)):
                a = np.linspace(0, 2 * np.pi, 80)
                cons.append(np.column_stack([self.Xc + rx * np.cos(a), cy + ry * np.sin(a)]))
            for (x, y, r) in circles[::3]:
                a = np.linspace(0, 2 * np.pi, 40)
                cons.append(np.column_stack([x + r * np.cos(a), y + r * np.sin(a)]))
            for q in cons:
                pp, rr, dd = hand(q, 0.012, int(rng.integers(1 << 30)), slow=(3.0, 0.012), dens=0.22,
                                  thin_end=0.6, taper=(0.1, 0.1))
                S.add(pp, rr, dd, layer=lay)
            _window(S, k, *T(0.07, 0.15), overlap=0.3)

        # 2. the silhouette: bold on the shadow side, broken on the lit side
        k = len(S)
        Lsil, Rsil = self.silhouette()
        passes = 3 if pencil else 1
        for ps in range(passes):
            jit = (lambda P: P + rng.normal(0, 0.01, P.shape)) if pencil else (lambda P: P)
            line(S, jit(Rsil[::-1]), W(0.04 if not pencil else 0.02), self.seed + 10 + ps, dens=dens,
                 layer=lay, lift=(1.2, 2.8), taper=(0.04, 0.2))
            n = len(Lsil)
            cuts = sorted(rng.choice(np.arange(20, n - 20), 3, replace=False))
            prev = 0
            for c_ in list(cuts) + [n]:
                seg = Lsil[prev:c_ - (5 if c_ < n else 0)]
                if len(seg) > 4:
                    line(S, jit(seg[::-1]), W(0.022 if not pencil else 0.018), self.seed + 20 + c_, dens=dens * 0.9,
                         layer=lay, lift=(1.0, 2.0), taper=(0.05, 0.15))
                prev = c_
        rim = np.array([[self.Xc - self.Wt, self.Ys], [self.Xc - 0.35, self.Ys + 0.1],
                        [self.Xc - 0.1, self.Ys + 0.17], [self.Xc + 0.2, self.Ys + 0.14], [self.Xc + self.Wt, self.Ys]])
        line(S, rim, W(0.03), self.seed + 31, dens=dens, layer=lay, lift=(3, 4))
        _window(S, k, *T(0.1 if not pencil else 0.15, 0.24), overlap=0.1)

        # 3. the fire in the throat: tall tongues of flame; the ring among them
        k = len(S)
        nt = 7
        for m in range(nt):
            q = (m - (nt - 1) / 2) / ((nt - 1) / 2)
            hgt = (0.7 + 0.35 * rng.random()) * (1.25 - 0.5 * abs(q))
            base = np.array([fx + 0.48 * q + rng.normal(0, 0.03), self.Ys + 0.1])
            lean = 0.25 * q + rng.normal(0, 0.08)
            wdt = 0.16 + 0.06 * rng.random()
            sgn = 1 if rng.random() < 0.5 else -1
            # an S-curved tongue: fat low belly, a pinched neck, a flicked tip
            L_ = [base + [-wdt, 0], base + [-wdt * 1.2 + lean * 0.2, -0.3 * hgt],
                  base + [-wdt * 0.4 + lean * 0.55 - sgn * 0.05, -0.62 * hgt],
                  base + [lean + sgn * 0.12, -1.0 * hgt]]
            R_ = [base + [lean + sgn * 0.12, -1.0 * hgt], base + [wdt * 0.25 + lean * 0.6 + sgn * 0.08, -0.6 * hgt],
                  base + [wdt * 1.05 + lean * 0.2, -0.28 * hgt], base + [wdt, 0]]
            line(S, np.array(L_), W(0.016), self.seed + 40 + m, dens=dens, layer=lay, lift=(4, 6), smooth=10,
                 taper=(0.05, 0.25))
            line(S, np.array(R_), W(0.014), self.seed + 45 + m, dens=dens, layer=lay, lift=(4, 6), smooth=10,
                 taper=(0.2, 0.05))
        if not pencil:
            for m in range(9):
                ex = fx + rng.normal(0, 0.55)
                ey = self.Ys - 1.1 - rng.random() * 0.9
                if in_smoke(ex, ey):
                    continue
                q = np.array([[ex, ey], [ex + 0.03, ey - 0.05]])
                pp, rr, dd = hand(q, 0.018, int(rng.integers(1 << 30)), dens=0.9, thin_end=0.6, taper=(0.01, 0.02))
                S.add(pp, rr, dd, layer=lay)
        # rays of light from the throat, broken, into the dark sky
        for m in range(17):
            a = -np.pi / 2 + (m - 8) * 0.17 + rng.normal(0, 0.03)
            r0, r1 = 1.25 + 0.2 * rng.random(), 1.7 + 0.45 * rng.random()
            p0 = np.array([fx + r0 * math.cos(a), fy + r0 * math.sin(a)])
            p1 = np.array([fx + r1 * math.cos(a), fy + r1 * math.sin(a)])
            if in_smoke(p1[0], p1[1], 0.1) or p1[1] < by0 + 0.2:
                continue
            pp, rr, dd = hand(np.array([p0, p1]), W(0.01), int(rng.integers(1 << 30)), taper=(0.04, 0.15),
                              dens=dens * 0.75, thin_end=0.15)
            S.add(pp, rr, dd, layer=lay)
        _window(S, k, *T(0.24, 0.34), overlap=0.3)

        # the ring: gold leaf, outlined in fine ink
        k = len(S)
        a = np.linspace(0, 2 * np.pi, 90)
        rx, ry = 0.17, 0.13
        ringp = np.column_stack([fx + rx * np.cos(a), fy + ry * np.sin(a)])
        if pencil:
            pp, rr, dd = hand(ringp, 0.014, self.seed + 60, slow=(1.0, 0.004), dens=0.5, thin_end=0.8,
                              taper=(0.0, 0.0))
            S.add(pp, rr, dd, layer=lay)
        else:
            S.add(ringp, np.full(len(ringp), 0.03), np.full(len(ringp), 1.0), layer=GILT)
            for sc in (1.0 + 0.03 / 0.15 * 1.25, 1.0 - 0.03 / 0.15 * 1.25):
                q = np.column_stack([fx + rx * sc * np.cos(a), fy + ry * sc * np.sin(a)])
                pp, rr, dd = hand(q, 0.006, self.seed + 61, slow=(1.0, 0.002), dens=0.8, thin_end=0.9,
                                  taper=(0, 0), fast=(0.2, 0.0008))
                S.add(pp, rr, dd, layer=INK)
        _window(S, k, *T(0.34, 0.4))

        # 4. hatching on the cone
        k = len(S)
        if pencil:
            def ins(x, y):
                return self.inside(x, y)

            def ton(x, y):
                h = np.clip((self.Yb - y) / (self.Yb - self.Ys), 0, 1)
                w = self.half_width(h)
                phi = np.arcsin(np.clip((x - self.Xc) / np.maximum(w, 1e-3), -0.999, 0.999))
                return self.tone(h, phi)
            box = (self.Xc - self.Wb - 0.6, self.Ys, self.Xc + self.Wb + 0.6, self.Yb + 0.1)
            hatch(S, ins, ton, box, 62, 0.085, 0.5, 0.016, self.seed + 70, dens=0.4, seg=(0.4, 1.4), layer=lay,
                  wob=0.012, jitter=0.6)
            hatch(S, ins, ton, box, 62, 0.085, 0.78, 0.016, self.seed + 71, dens=0.4, seg=(0.4, 1.2), layer=lay,
                  wob=0.012, jitter=0.6)
        else:
            # fall lines: three levels, thinned as they converge toward the rim, drawn in short pulls
            nl = 120
            for li, sb in enumerate(np.linspace(-0.98, 0.98, nl)):
                lvl = 0 if li % 4 == 0 else (1 if li % 2 == 0 else 2)
                hmax = (0.95, 0.72, 0.5)[lvl] + rng.uniform(-0.05, 0.03)
                phi0 = math.asin(sb)
                hs = np.linspace(0.0, hmax, 200)
                phi = phi0 + 0.05 * np.sin(hs * 5 + li * 0.7) * (1 - hs)
                xs, ys = self.surf(hs, phi)
                tn = self.tone(hs, phi)
                thr = (0.34, 0.5, 0.66)[lvl] + 0.04 * math.sin(li * 2.3)
                keep = tn > thr
                if lvl == 0:
                    keep |= (hs < 0.22 + 0.1 * math.sin(li * 1.3)) & (hs > 0.02)
                dm = np.diff(np.concatenate([[0], keep.astype(np.int8), [0]]))
                for a_, b_ in zip(np.nonzero(dm == 1)[0], np.nonzero(dm == -1)[0] - 1):
                    if b_ - a_ < 6:
                        continue
                    pts = np.column_stack([xs[a_:b_ + 1], ys[a_:b_ + 1]])[::-1]
                    wdt = 0.008 + 0.016 * float(np.mean(tn[a_:b_ + 1])) ** 2
                    line(S, pts, wdt, int(rng.integers(1 << 30)), dens=0.92, layer=lay, lift=(0.7, 1.8), smooth=0,
                         slow=(2.5, 0.004), fast=(0.3, 0.001), taper=(0.04, 0.25), pool=0.2, thin_end=0.12)
            # ridges: spurs running down from the rim; each casts a hatched shadow on its right
            for gi, ph in enumerate((-0.78, -0.3, 0.2, 0.62, 1.0)):
                hs = np.linspace(0.04, 0.9 - 0.12 * abs(ph), 120)
                phi = ph + 0.06 * np.sin(hs * 6 + gi) + 0.04 * hs
                xs, ys = self.surf(hs, phi)
                pts = np.column_stack([xs, ys])[::-1]
                line(S, pts, 0.014 + 0.006 * (ph > 0), self.seed + 90 + gi, dens=0.95, layer=lay, lift=(1.0, 2.2),
                     smooth=0, taper=(0.06, 0.35), thin_end=0.15)
                for off in (0.035, 0.07, 0.105):
                    if rng.random() < 0.25:
                        continue
                    hs2 = np.linspace(0.06, hs[-1] * rng.uniform(0.55, 0.95), 80)
                    phi2 = ph + off + 0.06 * np.sin(hs2 * 6 + gi) + 0.04 * hs2
                    xs2, ys2 = self.surf(hs2, phi2)
                    line(S, np.column_stack([xs2, ys2])[::-1], 0.011, int(rng.integers(1 << 30)), dens=0.9, layer=lay,
                         lift=(0.6, 1.4), smooth=0, taper=(0.05, 0.3), thin_end=0.12)
            # cross-contours in the deepest shadow
            for hh in np.arange(0.03, 0.75, 0.03):
                phis = np.linspace(0.3, 1.52, 90)
                xs, ys = self.surf(np.full_like(phis, hh), phis)
                tn = self.tone(np.full_like(phis, hh), phis)
                keep = tn > 0.86 + 0.03 * math.sin(hh * 40)
                dm = np.diff(np.concatenate([[0], keep.astype(np.int8), [0]]))
                for a_, b_ in zip(np.nonzero(dm == 1)[0], np.nonzero(dm == -1)[0] - 1):
                    if b_ - a_ < 6:
                        continue
                    pts = np.column_stack([xs[a_:b_ + 1], ys[a_:b_ + 1]])
                    pp, rr, dd = hand(pts, 0.008, int(rng.integers(1 << 30)), slow=(2.0, 0.004),
                                      taper=(0.06, 0.1), dens=0.85, thin_end=0.2)
                    S.add(pp, rr, dd, layer=lay)
        idx = list(range(k, len(S)))
        idx.sort(key=lambda i: S.P[i][:, 1].min())
        _window(S, k, *T(0.4, 0.7), overlap=0.93, order=[i - k for i in idx])

        # 5. the smoke: puffs outlined, the billows inside drawn where one rolls over another, the undersides hatched
        k = len(S)
        for oi, o in enumerate(outl):
            line(S, o, W(0.02), self.seed + 110 + oi, dens=dens, layer=lay, lift=(0.8, 1.8), taper=(0.04, 0.12), smooth=0)
        if not pencil:
            for ci, (x, y, r) in enumerate(circles):
                a = np.linspace(-0.2, 2 * np.pi - 0.2, 120)
                px_ = x + r * np.cos(a)
                py_ = y + r * np.sin(a)
                # the visible part of this puff's rim: over puffs behind it, not under puffs in front
                front = np.zeros(len(a), bool)
                behind = np.zeros(len(a), bool)
                for cj, (x2, y2, r2) in enumerate(circles):
                    if cj == ci:
                        continue
                    ins_ = (px_ - x2) ** 2 + (py_ - y2) ** 2 < (r2 * 0.98) ** 2
                    if cj > ci:
                        front |= ins_
                    else:
                        behind |= ins_
                vis = behind & ~front
                # only the lower and right parts of a puff show a rim line (the light is from the upper left)
                vis &= (np.sin(a) > -0.35)
                dm = np.diff(np.concatenate([[0], vis.astype(np.int8), [0]]))
                for a_, b_ in zip(np.nonzero(dm == 1)[0], np.nonzero(dm == -1)[0] - 1):
                    if b_ - a_ < 8:
                        continue
                    pts = np.column_stack([px_[a_:b_ + 1], py_[a_:b_ + 1]])
                    pp, rd, dd = hand(pts, 0.013, int(rng.integers(1 << 30)), taper=(0.05, 0.2), dens=0.9, thin_end=0.2)
                    S.add(pp, rd, dd, layer=lay)
                # shading: short curved strokes inside the lower right of the puff
                for q in range(int(6 + 22 * r)):
                    rr_ = r * rng.uniform(0.5, 0.93)
                    a0 = rng.uniform(-0.2, 1.8)
                    aa = np.linspace(a0, a0 + rng.uniform(0.25, 0.5), 8)
                    pts = np.column_stack([x + rr_ * np.cos(aa), y + rr_ * np.sin(aa)])
                    if in_smoke(pts[:, 0], pts[:, 1]).all():
                        # not where a puff in front covers it
                        ok = True
                        for cj in range(ci + 1, len(circles)):
                            x2, y2, r2 = circles[cj]
                            if np.any((pts[:, 0] - x2) ** 2 + (pts[:, 1] - y2) ** 2 < r2 ** 2):
                                ok = False
                                break
                        if ok:
                            pp, rd, dd = hand(pts, 0.0075, int(rng.integers(1 << 30)), taper=(0.03, 0.06), dens=0.8,
                                              thin_end=0.25)
                            S.add(pp, rd, dd, layer=lay)
        _window(S, k, *T(0.58, 0.78), overlap=0.6)

        # 6. the sky: ruled lines, denser toward the top, stopping short of the mountain, the smoke and the glow
        k = len(S)
        if not pencil:
            def clear(x, y):
                x = np.asarray(x)
                y = np.asarray(y)
                ok = ~self.inside(x, y + 0.13) & ~self.inside(x - 0.1, y) & ~self.inside(x + 0.1, y)
                ok &= ~in_smoke(x, y, 0.1)
                ang = np.arctan2(y - fy, x - fx)
                rglow = 0.95 + 0.25 * np.sin(ang * 5 + 1.0) + 0.12 * np.sin(ang * 11)
                ok &= np.hypot(x - fx, y - fy) > rglow
                return ok

            def sky_tone(x, y):
                t_ = np.clip(1.0 - (y - by0) / (self.Yh - by0), 0, 1) ** 0.8
                # the fire lights the sky round the throat: lines thin out toward it
                g = np.exp(-np.hypot(x - fx, (y - fy) * 1.3) / 1.6)
                return t_ * (1.0 - 0.85 * g)

            for li, yy in enumerate(np.arange(by0 + 0.22, self.Yh - 0.05, 0.062)):
                lvl = 0 if li % 4 == 0 else (1 if li % 2 == 0 else 2)
                thr = (0.0, 0.22, 0.45)[lvl]
                xs = np.arange(bx0 + 0.28, bx1 - 0.28, 0.02)
                ys = np.full_like(xs, yy) + 0.006 * np.sin(xs * 1.3 + li)
                tn = sky_tone(xs, ys)
                keep = clear(xs, ys) & (tn > thr)
                dm = np.diff(np.concatenate([[0], keep.astype(np.int8), [0]]))
                for a_, b_ in zip(np.nonzero(dm == 1)[0], np.nonzero(dm == -1)[0] - 1):
                    if b_ - a_ < 4:
                        continue
                    pts = np.column_stack([xs[a_:b_ + 1], ys[a_:b_ + 1]])
                    # a ruled line drawn in two or three pulls; it swells where the sky is darkest
                    s0 = len(S)
                    line(S, pts, 0.008, int(rng.integers(1 << 30)), dens=0.88, layer=lay, lift=(3.0, 6.0), smooth=0,
                         slow=(5.0, 0.002), fast=(0.4, 0.0008), taper=(0.02, 0.04), thin_end=0.5, press=0.08)
                    for q in range(s0, len(S)):
                        sw = 1.0 + 1.5 * sky_tone(S.P[q][:, 0], S.P[q][:, 1]) ** 2
                        S.R[q] = S.R[q] * sw
            # the darkest band at the top: lines laid between the lines
            for li, yy in enumerate(np.arange(by0 + 0.25, by0 + 0.25 + 0.3 * (self.Yh - by0), 0.062)):
                xs = np.arange(bx0 + 0.28, bx1 - 0.28, 0.02)
                ys = np.full_like(xs, yy) + 0.006 * np.sin(xs * 1.7 + li)
                tn = sky_tone(xs, ys)
                keep = clear(xs, ys) & (tn > 0.72 + 0.05 * math.sin(li * 1.9))
                dm = np.diff(np.concatenate([[0], keep.astype(np.int8), [0]]))
                for a_, b_ in zip(np.nonzero(dm == 1)[0], np.nonzero(dm == -1)[0] - 1):
                    if b_ - a_ < 4:
                        continue
                    pts = np.column_stack([xs[a_:b_ + 1], ys[a_:b_ + 1]])
                    line(S, pts, 0.009, int(rng.integers(1 << 30)), dens=0.88, layer=lay, lift=(2.0, 5.0), smooth=0,
                         slow=(5.0, 0.002), fast=(0.4, 0.0008), taper=(0.03, 0.05), thin_end=0.4, press=0.08)
            for m in range(9):
                sx = bx0 + 0.8 + rng.random() * 5.0
                sy = by0 + 0.45 + rng.random() * 1.9
                if not clear(np.array([sx]), np.array([sy]))[0]:
                    continue
                for a in (0.0, np.pi / 2):
                    r_ = 0.06 + 0.04 * rng.random()
                    q = np.array([[sx - r_ * math.cos(a), sy - r_ * math.sin(a)], [sx + r_ * math.cos(a), sy + r_ * math.sin(a)]])
                    pp, rd, dd = hand(q, 0.01, int(rng.integers(1 << 30)), dens=0.9, thin_end=0.3, taper=(0.02, 0.02))
                    S.add(pp, rd, dd, layer=lay)
        else:
            for yy in np.arange(by0 + 0.4, by0 + 2.0, 0.35):
                q = np.array([[bx0 + 0.4, yy], [bx0 + 4.5 + rng.random() * 2, yy + 0.02]])
                pp, rd, dd = hand(q, 0.016, int(rng.integers(1 << 30)), slow=(3.0, 0.015), dens=0.3, thin_end=0.4)
                S.add(pp, rd, dd, layer=lay)
        _window(S, k, *T(0.72, 0.9), overlap=0.8)

        # 7. the far ranges, the plain, the road and the door, rocks in the foreground
        k = len(S)
        for side, (xa, xb) in enumerate(((bx0 + 0.25, self.Xc - self.Wb + 0.3), (self.Xc + self.Wb - 0.3, bx1 - 0.25))):
            P = ridgeline(xa, xb, self.Yh + 0.02, 0.5 if side == 0 else 0.42, np.random.default_rng(self.seed + 130 + side), n=6, rough=0.45)
            keep = ~self.inside(P[:, 0], P[:, 1] + 0.05)
            dm = np.diff(np.concatenate([[0], keep.astype(np.int8), [0]]))
            for a_, b_ in zip(np.nonzero(dm == 1)[0], np.nonzero(dm == -1)[0] - 1):
                if b_ - a_ < 3:
                    continue
                seg = P[a_:b_ + 1]
                line(S, seg, W(0.014), int(rng.integers(1 << 30)), dens=dens * 0.9, layer=lay, lift=(1.5, 3.0),
                     smooth=0, taper=(0.05, 0.1))
                if not pencil:
                    # shade the faces that turn from the light: short strokes down the fall line
                    for q in range(len(seg) - 1):
                        a, b = seg[q], seg[q + 1]
                        if b[1] <= a[1]:
                            continue          # rising: lit face
                        for f in np.linspace(0, 1, max(1, int(np.hypot(*(b - a)) / 0.045))):
                            p0 = a + (b - a) * f
                            ln = min(0.08 + 0.35 * rng.random(), self.Yh - p0[1])
                            if ln > 0.03:
                                pp, rd, dd = hand(np.array([p0 + [0, 0.02], p0 + [-0.03, ln]]), 0.0075,
                                                  int(rng.integers(1 << 30)), dens=0.8, thin_end=0.3)
                                S.add(pp, rd, dd, layer=lay)
        xs = np.arange(bx0 + 0.25, bx1 - 0.25, 0.02)
        keep = ~self.inside(xs, np.full_like(xs, self.Yh))
        dm = np.diff(np.concatenate([[0], keep.astype(np.int8), [0]]))
        for a_, b_ in zip(np.nonzero(dm == 1)[0], np.nonzero(dm == -1)[0] - 1):
            if b_ - a_ > 5:
                line(S, np.column_stack([xs[a_:b_], np.full(b_ - a_, self.Yh)]), W(0.011), int(rng.integers(1 << 30)),
                     dens=dens * 0.8, layer=lay, lift=(1.0, 3.0), smooth=0)
        if not pencil:
            yy = self.Yh + 0.12
            while yy < by1 - 0.15:
                gap_ = 0.07 + 0.22 * (yy - self.Yh) / (by1 - self.Yh)
                xx = bx0 + 0.3 + rng.random() * 0.4
                while xx < bx1 - 0.3:
                    ln = 0.1 + 0.35 * rng.random() * (0.5 + (yy - self.Yh) / (by1 - self.Yh))
                    if not self.inside(np.array([xx]), np.array([yy]))[0] and rng.random() < 0.5:
                        q = np.array([[xx, yy], [xx + ln, yy + rng.normal(0, 0.01)]])
                        pp, rd, dd = hand(q, 0.0085, int(rng.integers(1 << 30)), dens=0.8, thin_end=0.25,
                                          taper=(0.03, 0.08))
                        S.add(pp, rd, dd, layer=lay)
                    xx += ln + rng.uniform(0.2, 0.9)
                yy += gap_ + rng.uniform(0, 0.05)
        door = self.surf(0.36, -0.52)
        road = catmull([(bx0 + 1.2, by1 - 0.1), (bx0 + 2.8, by1 - 1.0), (self.Xc - 3.3, self.Yb + 0.3),
                        (self.Xc - 2.7, self.Yb - 0.25), (door[0] - 0.3, door[1] + 0.35), (door[0], door[1] + 0.12)], 14)
        road = resample(road, 0.1)
        for q in road:
            pp, rd, dd = hand(np.array([q, q + [0.022, 0.0]]), W(0.02), int(rng.integers(1 << 30)), dens=dens,
                              thin_end=0.8, taper=(0, 0))
            S.add(pp, rd, dd, layer=lay)
        dx_, dy_ = door
        arch = np.array([[dx_ - 0.13, dy_ + 0.12], [dx_ - 0.13, dy_ - 0.06], [dx_, dy_ - 0.19], [dx_ + 0.13, dy_ - 0.06],
                         [dx_ + 0.13, dy_ + 0.12]])
        line(S, arch, W(0.016), self.seed + 150, dens=dens, layer=lay, lift=(4, 5))
        if not pencil:
            hatch(S, _poly_inside(catmull(arch, 6)), lambda x, y: np.ones_like(x), (dx_ - 0.15, dy_ - 0.22, dx_ + 0.15, dy_ + 0.14),
                  90, 0.022, 0.0, 0.01, self.seed + 151, dens=0.95, seg=(0.3, 0.4))
        for (cx, cy, r_) in ((bx0 + 1.1, by1 - 0.55, 0.55), (bx0 + 2.25, by1 - 0.33, 0.3), (bx1 - 1.6, by1 - 0.5, 0.45)):
            a = np.linspace(np.pi, 2 * np.pi, 30)
            rk = np.column_stack([cx + r_ * 1.4 * np.cos(a) + 0.05 * np.sin(a * 5), cy + r_ * 0.8 * np.sin(a) * (1 + 0.1 * np.cos(a * 3))])
            rk = np.vstack([rk, [[cx + r_ * 1.45, cy + 0.02]]])
            line(S, rk, W(0.022), int(rng.integers(1 << 30)), dens=dens, layer=lay, lift=(1.5, 3.0))
            if not pencil:
                poly = np.vstack([rk, [[cx - r_ * 1.4, cy + 0.02]]])
                hatch(S, _poly_inside(poly), lambda x, y, cx=cx, r_=r_: (x - cx) / (r_ * 1.4) * 0.5 + 0.55,
                      (cx - r_ * 1.5, cy - r_, cx + r_ * 1.5, cy + 0.05), -60, 0.045, 0.45, 0.009,
                      int(rng.integers(1 << 30)), dens=0.85, seg=(0.2, 0.6))
        _window(S, k, *T(0.84, 1.0), overlap=0.85)
        return S

    def gilt_glint_time(self, t0, t1):
        return t0 + 0.42 * (t1 - t0)


# ============================================================== THE DEEP ===

class Deep:
    """Pillared halls stacked downward, a bright vein followed deeper and deeper, a red glow waking at the
    bottom of the page. The whole text block is the drawing: a section through the roots of a mountain, each
    hall a dark carved void with lit pillars and pointed vaults, larger the deeper; the camera descends it."""

    def __init__(self, seed=23):
        self.seed = seed
        self.box = (2.3, 2.3, 17.3, 26.6)
        bx0, by0, bx1, by1 = self.box
        rng = np.random.default_rng(seed)
        ys = np.linspace(by0 + 1.5, by1 - 0.9, 14)
        xs = np.array([6.2, 7.4, 8.6, 9.1, 8.2, 7.2, 7.9, 9.8, 11.0, 10.6, 9.4, 8.8, 9.6, 10.0])
        xs = xs + rng.normal(0, 0.15, len(xs))
        self.vein = resample(catmull(np.column_stack([xs, ys]), 16), 0.03)
        self.halls = []
        tops = [by0 + 2.4, by0 + 6.1, by0 + 10.0, by0 + 14.2, by0 + 18.6]
        for k, yt in enumerate(tops):
            hgt = 2.0 + 0.5 * k                       # crown to floor
            xv = float(np.interp(yt + 0.5 * hgt, self.vein[:, 1], self.vein[:, 0]))
            wid = 3.6 + 2.2 * k
            x0 = xv - 0.5 * wid + rng.uniform(-0.6, 0.6)
            x0 = min(max(x0, bx0 + 0.4), bx1 - 0.4 - wid)
            n = 2 + k                                 # free-standing pillars
            self.halls.append(dict(x0=x0, x1=x0 + wid, yc=yt, ys=yt + 0.34 * hgt, y1=yt + hgt, n=n, k=k,
                                   pw=0.16 + 0.035 * k))
        self.glow = (float(self.vein[-1, 0]), by1 - 0.55)

    # ---- hall geometry
    def supports(self, h):
        return np.linspace(h['x0'], h['x1'], h['n'] + 2)

    def arch_y(self, h, x):
        """The underside of the vault above x (pointed arches springing from the supports)."""
        x = np.asarray(x, np.float64)
        sp = self.supports(h)
        i = np.clip(np.searchsorted(sp, x) - 1, 0, len(sp) - 2)
        a, b = sp[i], sp[i + 1]
        half = 0.5 * (b - a)
        u = np.clip(np.abs(x - 0.5 * (a + b)) / half, 0, 1)
        rise = h['ys'] - h['yc']
        # a pointed (two-centred) arch
        return h['ys'] - rise * np.sqrt(np.clip(1 - u ** 1.6, 0, 1))

    def pillar_mask(self, h, x, y):
        x = np.asarray(x, np.float64)
        y = np.asarray(y, np.float64)
        sp = self.supports(h)[1:-1]
        pw = h['pw']
        m = np.zeros(np.broadcast(x, y).shape, bool)
        for px in sp:
            # a shaft that flares into its capital and stands on a plinth
            fl = np.clip((h['ys'] + 0.35 - y) / 0.35, 0, 1) ** 2
            wdt = 0.5 * pw * (1 + 1.6 * fl)
            base = np.where(y > h['y1'] - 0.12, 0.5 * pw * 1.45, 0)
            m |= (np.abs(x - px) < np.maximum(wdt, base)) & (y > self.arch_y(h, x) - 0.01) & (y < h['y1'])
        return m

    def void(self, h, x, y):
        x = np.asarray(x, np.float64)
        y = np.asarray(y, np.float64)
        return (x > h['x0']) & (x < h['x1']) & (y < h['y1']) & (y > self.arch_y(h, x)) & ~self.pillar_mask(h, x, y)

    def any_hall(self, x, y, pad=0.0):
        x = np.asarray(x, np.float64)
        y = np.asarray(y, np.float64)
        m = np.zeros(np.broadcast(x, y).shape, bool)
        for h in self.halls:
            m |= (x > h['x0'] - pad) & (x < h['x1'] + pad) & (y < h['y1'] + pad) & (y > self.arch_y(h, x) - pad)
        return m

    def surface(self, x):
        bx0, by0, bx1, by1 = self.box
        x = np.asarray(x, np.float64)
        return by0 + 1.25 - 0.55 * np.exp(-((x - (bx0 + 11.5)) / 2.2) ** 2) - 0.3 * np.exp(-((x - (bx0 + 4.0)) / 1.6) ** 2) \
            + 0.05 * np.sin(x * 3.1)

    def rock_inside(self, x, y):
        x = np.asarray(x)
        y = np.asarray(y)
        bx0, by0, bx1, by1 = self.box
        ok = (x > bx0 + 0.18) & (x < bx1 - 0.18) & (y > self.surface(x) + 0.05) & (y < by1 - 0.18)
        return ok & ~self.any_hall(x, y, 0.07)

    def depth_tone(self, y):
        bx0, by0, bx1, by1 = self.box
        return np.clip((np.asarray(y) - by0) / (by1 - by0), 0, 1)

    def build(self, mode='ink', t0=0.0, t1=8.0, pace=None):
        """pace(k) -> (a, b) fractions of [t0, t1] for level k (default: each level drawn in less time than the
        one above: 'dividing faster and faster')."""
        rng = np.random.default_rng(self.seed)
        S = Strokes()
        pencil = mode == 'pencil'
        lay = INK if not pencil else PENCIL
        bx0, by0, bx1, by1 = self.box
        dens = 0.55 if pencil else 0.95
        W = (lambda w: w * 1.1) if pencil else (lambda w: w)
        T = lambda a, b: (t0 + (t1 - t0) * a, t0 + (t1 - t0) * b)
        if pace is None:
            spans = [0.26, 0.2, 0.16, 0.13, 0.11, 0.09]
            acc = np.concatenate([[0], np.cumsum(spans)])
            acc = acc / acc[-1]
            pace = lambda k: (acc[k], acc[k + 1])

        # the frame and the surface: the mountainside above, a few far peaks, the gate where the vein comes out
        k = len(S)
        frame_rules(S, self.box, self.seed + 1, layer=lay)
        xs = np.linspace(bx0 + 0.18, bx1 - 0.18, 220)
        sy = self.surface(xs)
        line(S, np.column_stack([xs, sy]), W(0.03), self.seed + 5, dens=dens, layer=lay, lift=(2.0, 4.0), smooth=0)
        for m, (cx, hh) in enumerate(((bx0 + 11.5, 0.7), (bx0 + 9.8, 0.45), (bx0 + 13.3, 0.5))):
            base_y = float(np.interp(cx, xs, sy))
            top = max(base_y - hh, by0 + 0.25)
            pk = np.array([[cx - 0.9 * (base_y - top), base_y + 0.02], [cx - 0.06, top], [cx + 0.05, top + 0.01],
                           [cx + 0.85 * (base_y - top), base_y + 0.02]])
            line(S, pk, W(0.016), self.seed + 10 + m, dens=dens, layer=lay, lift=(3, 4), smooth=0)
            if not pencil:
                for q in np.linspace(0.12, 0.9, 6):
                    p0 = pk[2] + (pk[3] - pk[2]) * q
                    pp, rd, dd = hand(np.array([p0, p0 + np.array([-0.04, 0.6 * (base_y - p0[1])])]), 0.008,
                                      int(rng.integers(1 << 30)), dens=0.8, thin_end=0.25)
                    S.add(pp, rd, dd, layer=lay)
        gx = float(self.vein[0, 0])
        gy = float(np.interp(gx, xs, sy))
        gate = np.array([[gx - 0.28, gy + 0.02], [gx - 0.28, gy - 0.22], [gx, gy - 0.42], [gx + 0.28, gy - 0.22],
                         [gx + 0.28, gy + 0.02]])
        line(S, gate, W(0.018), self.seed + 20, dens=dens, layer=lay, lift=(4, 5))
        if not pencil:
            hatch(S, _poly_inside(catmull(gate, 6)), lambda x, y: np.ones_like(x), (gx - 0.3, gy - 0.45, gx + 0.3, gy + 0.03),
                  90, 0.028, 0.0, 0.011, self.seed + 21, dens=0.95, seg=(0.3, 0.5))
        a0_, b0_ = pace(0)
        _window(S, k, *T(a0_, a0_ + 0.4 * (b0_ - a0_)), overlap=0.3)

        for hk, h in enumerate(self.halls):
            a_, b_ = pace(hk)
            k = len(S)
            x0, x1, yc, ys_, y1, n, pw = h['x0'], h['x1'], h['yc'], h['ys'], h['y1'], h['n'], h['pw']
            sp = self.supports(h)
            # walls and floor
            line(S, np.array([[x0 - 0.05, y1], [x1 + 0.05, y1]]), W(0.028), self.seed + 100 * hk + 1, dens=dens,
                 layer=lay, lift=(2, 5), smooth=0)
            for xw in (x0, x1):
                line(S, np.array([[xw, y1], [xw, float(self.arch_y(h, np.array([xw + (0.01 if xw == x0 else -0.01)]))[0])]]),
                     W(0.026), self.seed + 100 * hk + 2 + int(xw == x1), dens=dens, layer=lay, lift=(3, 5), smooth=0)
            # the vault: one pointed arch per bay
            for q in range(len(sp) - 1):
                xx = np.linspace(sp[q] + 0.01, sp[q + 1] - 0.01, 60)
                line(S, np.column_stack([xx, self.arch_y(h, xx)]), W(0.022), self.seed + 100 * hk + 5 + q, dens=dens,
                     layer=lay, lift=(3, 5), smooth=0)
            # pillars: flared capitals, shafts, plinths; shaded on the right (the light is from the left)
            for p, px in enumerate(sp[1:-1]):
                yy = np.linspace(float(self.arch_y(h, np.array([px - 0.5 * pw * 2.6]))[0]), y1 - 0.12, 60)
                fl = np.clip((ys_ + 0.35 - yy) / 0.35, 0, 1) ** 2
                wdt = 0.5 * pw * (1 + 1.6 * fl)
                line(S, np.column_stack([px - wdt, yy]), W(0.016), self.seed + 100 * hk + 20 + p, dens=dens, layer=lay,
                     lift=(4, 8), smooth=0)
                line(S, np.column_stack([px + wdt, yy]), W(0.02), self.seed + 100 * hk + 40 + p, dens=dens, layer=lay,
                     lift=(4, 8), smooth=0)
                plinth = np.array([[px - 0.72 * pw, y1], [px - 0.72 * pw, y1 - 0.12], [px + 0.72 * pw, y1 - 0.12],
                                   [px + 0.72 * pw, y1]])
                line(S, plinth, W(0.016), self.seed + 100 * hk + 60 + p, dens=dens, layer=lay, lift=(5, 6), smooth=0)
                if not pencil:
                    for q in np.linspace(0.15, 0.9, 4):
                        xq = px + q * 0.5 * pw
                        pp, rd, dd = hand(np.array([[xq, ys_ + 0.3], [xq, y1 - 0.15]]), 0.007 + 0.004 * q,
                                          int(rng.integers(1 << 30)), dens=0.8, thin_end=0.3, taper=(0.1, 0.1))
                        S.add(pp, rd, dd, layer=lay)
            # the darkness of the hall behind the pillars; lighter bands where the far rows of pillars stand
            if not pencil:
                def dark(x, y, h=h, sp=sp):
                    x = np.asarray(x, np.float64)
                    t_ = 0.7 + 0.08 * h['k'] + 0.2 * np.clip((np.asarray(y) - h['ys']) / (h['y1'] - h['ys']), 0, 1)
                    # the far colonnade: a lighter band midway between front pillars, fainter each row back
                    mids = 0.5 * (sp[:-1] + sp[1:])
                    d = np.min(np.abs(x[..., None] - mids), -1)
                    t_ = t_ - 0.4 * np.exp(-(d / (0.5 * h['pw'])) ** 2)
                    return t_
                box = (x0, yc - 0.1, x1, y1)
                hatch(S, lambda x, y, h=h: self.void(h, x, y), dark, box, 90, 0.04, 0.2, 0.0095,
                      self.seed + 100 * hk + 70, dens=0.92, seg=(0.5, 1.5), wob=0.003, overshoot=0.0)
                hatch(S, lambda x, y, h=h: self.void(h, x, y), dark, box, 0, 0.045, 0.45, 0.009,
                      self.seed + 100 * hk + 71, dens=0.88, seg=(0.5, 1.8), wob=0.003, overshoot=0.0)
                hatch(S, lambda x, y, h=h: self.void(h, x, y), dark, box, 45, 0.05, 0.68, 0.0085,
                      self.seed + 100 * hk + 72, dens=0.85, seg=(0.3, 1.0), wob=0.003, overshoot=0.0)
                hatch(S, lambda x, y, h=h: self.void(h, x, y), dark, box, -45, 0.055, 0.84, 0.0085,
                      self.seed + 100 * hk + 73, dens=0.85, seg=(0.3, 1.0), wob=0.003, overshoot=0.0)
                xv = float(np.interp(y1 - 0.2, self.vein[:, 1], self.vein[:, 0]))
                for f_ in range(1 + hk // 2):
                    fxp = xv + (f_ - hk / 4.0) * 0.6 + rng.uniform(-0.1, 0.1)
                    if fxp < x0 + 0.3 or fxp > x1 - 0.3:
                        continue
                    if np.any(np.abs(sp - fxp) < pw * 1.5):
                        fxp += 1.6 * pw
                    self._miner(S, fxp, y1 - 0.01, 0.26 + 0.02 * hk, rng, lay)
            # a stair down to the next hall
            if hk + 1 < len(self.halls):
                nh = self.halls[hk + 1]
                left = hk % 2 == 1
                sx0 = (x0 + 0.25) if left else (x1 - 0.25)
                sx1 = (nh['x0'] + 0.3) if left else (nh['x1'] - 0.3)
                ya, yb = y1, float(self.arch_y(nh, np.array([sx1]))[0])
                steps = []
                nstep = 10
                xs_ = np.linspace(sx0, sx1, nstep + 1)
                yy_ = np.linspace(ya, yb, nstep + 1)
                for q in range(nstep):
                    steps += [[xs_[q], yy_[q]], [xs_[q + 1], yy_[q]], [xs_[q + 1], yy_[q + 1]]]
                line(S, np.array(steps), W(0.013), self.seed + 100 * hk + 80, dens=dens, layer=lay, lift=(8, 9), smooth=0)
            _window(S, k, *T(a_ + 0.3 * (b_ - a_), b_), overlap=0.9)

        # the rock: bedding planes, broken; darker and cross-hatched toward the deep
        k = len(S)
        if not pencil:
            def rtone(x, y):
                return 0.15 + 0.95 * self.depth_tone(y) ** 1.2 + 0.1 * np.sin(np.asarray(x) * 1.3 + np.asarray(y) * 0.5)

            def strata(u, v):
                return 0.22 * np.sin(u * 0.45 + v * 0.7) + 0.1 * np.sin(u * 1.2 + v * 1.9)
            hatch(S, self.rock_inside, rtone, self.box, -6, 0.1, 0.2, 0.009, self.seed + 900, dens=0.88,
                  seg=(1.0, 3.2), wob=0.004, curve=strata, jitter=0.4, gap=0.18)
            hatch(S, self.rock_inside, rtone, self.box, 38, 0.085, 0.55, 0.0085, self.seed + 901, dens=0.82,
                  seg=(0.3, 1.0), wob=0.004, jitter=0.6, gap=0.15)
            hatch(S, self.rock_inside, rtone, self.box, -48, 0.08, 0.78, 0.0085, self.seed + 902, dens=0.82,
                  seg=(0.3, 0.9), wob=0.004, jitter=0.6, gap=0.15)
            hatch(S, self.rock_inside, rtone, self.box, 84, 0.075, 0.95, 0.0085, self.seed + 903, dens=0.82,
                  seg=(0.3, 0.9), wob=0.004, jitter=0.6, gap=0.15)
            # a few cracks and joints in the rock
            for m in range(26):
                cx = rng.uniform(bx0 + 0.5, bx1 - 0.5)
                cy = rng.uniform(by0 + 2.0, by1 - 0.8)
                if not self.rock_inside(np.array([cx]), np.array([cy]))[0]:
                    continue
                a = rng.uniform(0, np.pi)
                pts = [np.array([cx, cy])]
                for q in range(4):
                    a += rng.normal(0, 0.5)
                    pts.append(pts[-1] + 0.25 * np.array([math.cos(a), math.sin(a)]))
                pts = np.array(pts)
                if self.rock_inside(pts[:, 0], pts[:, 1]).all():
                    pp, rd, dd = hand(pts, 0.014, int(rng.integers(1 << 30)), dens=0.9, thin_end=0.2, taper=(0.05, 0.2))
                    S.add(pp, rd, dd, layer=lay)
        else:
            def rtone(x, y):
                return 0.25 + 0.8 * self.depth_tone(y)
            hatch(S, self.rock_inside, rtone, self.box, 55, 0.16, 0.45, 0.018, self.seed + 900, dens=0.35,
                  seg=(0.4, 1.4), wob=0.012, jitter=0.8)
        idx = list(range(k, len(S)))
        ymid = np.array([S.P[i][:, 1].mean() for i in idx])
        edges = [h['y1'] for h in self.halls] + [by1 + 1]
        groups = np.searchsorted(edges, ymid)
        for g in range(len(edges)):
            sel = [idx[q] - k for q in range(len(idx)) if groups[q] == g]
            if not sel:
                continue
            a_, b_ = pace(min(g, len(self.halls)))
            sel.sort(key=lambda q: S.P[k + q][:, 1].mean())
            _window_sel(S, k, sel, *T(a_ + 0.1 * (b_ - a_), b_), overlap=0.97)

        # the vein: gold leaf, a bright thread with small branches, followed down level by level
        for k_h in range(len(self.halls) + 1):
            ya = by0 + 1.2 if k_h == 0 else self.halls[k_h - 1]['y1']
            yb = self.halls[k_h]['y1'] if k_h < len(self.halls) else self.glow[1] - 0.1
            m = (self.vein[:, 1] >= ya) & (self.vein[:, 1] <= yb)
            seg = self.vein[m]
            if len(seg) < 2:
                continue
            a_, b_ = pace(min(k_h, len(self.halls)))
            k = len(S)
            s = arclen(seg)
            wdt = 0.035 + 0.012 * k_h
            if pencil:
                pp, rd, dd = hand(seg, 0.02, self.seed + 700 + k_h, dens=0.5, taper=(0.1, 0.1))
                S.add(pp, rd, dd, layer=lay)
            else:
                pw_ = wdt * (0.7 + 0.5 * np.abs(np.sin(s * 3.1 + k_h)))
                S.add(seg, 0.5 * pw_, np.ones(len(seg)), layer=GILT)
                nrm = pen.normals(seg)
                for side in (-1, 1):
                    q = seg + nrm * (side * (0.5 * pw_ + 0.012))[:, None]
                    pp, rd, dd = hand(q, 0.006, self.seed + 710 + k_h * 3 + side, dens=0.7, thin_end=0.6, taper=(0.05, 0.05))
                    S.add(pp, rd, dd, layer=INK)
                for b in range(3 + k_h):
                    i = int(rng.integers(5, len(seg) - 5))
                    d = nrm[i] * (1 if rng.random() < 0.5 else -1)
                    br = np.array([seg[i], seg[i] + d * 0.3 + rng.normal(0, 0.08, 2), seg[i] + d * 0.55 + rng.normal(0, 0.14, 2)])
                    br = resample(catmull(br, 8), 0.02)
                    S.add(br, np.linspace(0.5 * wdt * 0.6, 0.004, len(br)), np.ones(len(br)), layer=GILT)
            _window(S, k, *T(a_, a_ + 0.5 * (b_ - a_)), overlap=0.2)

        # the bottom: a chasm, black, where the glow will wake
        k = len(S)
        gx, gy = self.glow
        L_ = np.array([[gx - 1.9, by1 - 0.18], [gx - 1.3, gy + 0.05], [gx - 0.95, gy - 0.3], [gx - 0.55, gy - 0.22],
                       [gx - 0.3, gy - 0.55], [gx - 0.05, gy - 0.4]])
        R_ = np.array([[gx - 0.05, gy - 0.4], [gx + 0.2, gy - 0.6], [gx + 0.45, gy - 0.25], [gx + 0.9, gy - 0.32],
                       [gx + 1.2, gy + 0.02], [gx + 1.8, by1 - 0.18]])
        for q, P_ in enumerate((L_, R_)):
            line(S, P_, W(0.028), self.seed + 990 + q, dens=dens, layer=lay, lift=(2, 4), smooth=0)
        if not pencil:
            chp = np.vstack([L_, R_[1:], [[gx - 1.9, by1 - 0.18]]])
            hatch(S, _poly_inside(chp), lambda x, y: np.ones_like(x), (gx - 2.0, gy - 0.7, gx + 1.9, by1 - 0.15),
                  80, 0.03, 0.0, 0.013, self.seed + 991, dens=0.95, seg=(0.3, 0.9))
            hatch(S, _poly_inside(chp), lambda x, y: np.ones_like(x), (gx - 2.0, gy - 0.7, gx + 1.9, by1 - 0.15),
                  -20, 0.035, 0.0, 0.012, self.seed + 992, dens=0.92, seg=(0.3, 0.9))
        a_, b_ = pace(len(self.halls))
        _window(S, k, *T(a_, b_), overlap=0.5)
        return S

    def _miner(self, S, x, yfoot, h, rng, lay):
        """A small bent figure with a pick raised: a silhouette of a few strokes (no face)."""
        lean = 0.25
        hip = np.array([x, yfoot - 0.45 * h])
        head = hip + np.array([lean * h * 0.6, -0.48 * h])
        feet = [np.array([x - 0.14 * h, yfoot]), np.array([x + 0.12 * h, yfoot])]
        for f in feet:
            pp, rd, dd = hand(np.array([hip, f]), 0.045 * h + 0.004, int(rng.integers(1 << 30)), dens=0.95, thin_end=0.6)
            S.add(pp, rd, dd, layer=lay)
        pp, rd, dd = hand(np.array([hip, hip + (head - hip) * 0.8]), 0.12 * h, int(rng.integers(1 << 30)), dens=0.95,
                          thin_end=0.7)
        S.add(pp, rd, dd, layer=lay)
        S.add(np.array([head, head + [0.001, 0]]), np.array([0.055 * h, 0.055 * h]), np.array([0.95, 0.95]), layer=lay)
        sh = hip + (head - hip) * 0.7
        hand_ = sh + np.array([0.3 * h, -0.35 * h])
        pp, rd, dd = hand(np.array([sh, hand_]), 0.035 * h + 0.003, int(rng.integers(1 << 30)), dens=0.95, thin_end=0.6)
        S.add(pp, rd, dd, layer=lay)
        pick = np.array([hand_ + [-0.05 * h, -0.1 * h], hand_ + [0.25 * h, 0.12 * h]])
        pp, rd, dd = hand(pick, 0.03 * h + 0.003, int(rng.integers(1 << 30)), dens=0.95, thin_end=0.4)
        S.add(pp, rd, dd, layer=lay)


def _window_sel(S, k0, sel, t0, t1, overlap=0.0):
    idx = [k0 + i for i in sel]
    L = np.array([arclen(S.P[i])[-1] for i in idx]) + 0.05
    span = (t1 - t0) / (L.sum() * (1.0 - overlap) + L[-1] * overlap)
    t = t0
    for k, i in enumerate(idx):
        d = L[k] * span
        S.T0[i], S.T1[i] = t, t + d
        t += d * (1.0 - overlap)


# ======================================================= the written leaves ===

def text_page(S, seed, lines=36, lead=0.62, xh=0.2, box=(2.4, 2.9, 17.2, 26.4), last_frac=None, skip=None,
              t0=0.0, t1=0.0, first_line=0, layer=INK, dens=0.95):
    """A page of the flowing hand in the text block; returns the line records."""
    h = pen.Hand(seed=seed, xh=xh, dens=dens, layer=layer)
    x0, y0, x1, y1 = box
    n = min(lines, int((y1 - y0) / lead) + 1)
    return h.write_block(S, x0, y0 + first_line * lead, x1 - x0, n, lead, t0=t0, t1=t1, last_frac=last_frac,
                         skip=skip)


def to_tex(S, t, ppc, tex, dry=0.9, gains=None):
    """Rasterise strokes at time t into a book.PageTex's channels (ink, wet, gilt, pencil)."""
    pk = S.pack()
    H, W = tex.H, tex.W
    C, Wt = pen.raster(pk, t, ppc, H, W, INK, dry=dry, gain=gains)
    tex.chan[..., 0] = np.maximum(tex.chan[..., 0], C)
    tex.chan[..., 1] = np.maximum(tex.chan[..., 1], Wt)
    C, _ = pen.raster(pk, t, ppc, H, W, GILT, gain=gains)
    tex.chan[..., 2] = np.maximum(tex.chan[..., 2], np.clip(C, 0, 1))
    C, _ = pen.raster(pk, t, ppc, H, W, PENCIL, gain=gains)
    tex.chan[..., 3] = np.maximum(tex.chan[..., 3], C)
    return tex


# ======================================================= THE YEAR OF PLENTY ===

class Plenty:
    """A young silver-barked tree in golden flower in an open field; beyond it a patchwork of harvest fields,
    hedgerows, an orchard, stooks of corn and a few cottages with smoke from every chimney. No round doors: the
    tree in the field is the reference. The flowers are gold leaf and catch the hearth."""

    def __init__(self, seed=41):
        self.seed = seed
        self.box = (2.3, 3.0, 17.3, 19.0)
        bx0, by0, bx1, by1 = self.box
        self.Yh = by0 + 7.0            # the far skyline
        self.tx = bx0 + 6.3            # the tree's foot
        self.ty = by0 + 12.6
        self.th = 8.2                  # its height

    def hills(self, x, k):
        """The k-th ridge of rolling hills (0 = far) as a y for each x."""
        bx0, by0, bx1, by1 = self.box
        x = np.asarray(x, np.float64)
        base = self.Yh + 0.9 * k + 0.45 * k * k
        return base - (0.55 - 0.1 * k) * np.sin(x * (0.55 - 0.08 * k) + 1.3 * k) - 0.25 * np.sin(x * 1.4 + 3.0 * k)

    def build(self, mode='ink', t0=0.0, t1=8.0):
        rng = np.random.default_rng(self.seed)
        S = Strokes()
        pencil = mode == 'pencil'
        lay = INK if not pencil else PENCIL
        bx0, by0, bx1, by1 = self.box
        dens = 0.55 if pencil else 0.95
        W = (lambda w: w * 1.1) if pencil else (lambda w: w)
        T = lambda a, b: (t0 + (t1 - t0) * a, t0 + (t1 - t0) * b)
        k = len(S)
        frame_rules(S, self.box, self.seed + 1, layer=lay)
        _window(S, k, *T(0.0, 0.05), overlap=0.3)

        # the tree: a slender trunk rising into a vase of limbs; the crown a cloud of leaf-clusters
        k = len(S)
        tx, ty, th = self.tx, self.ty, self.th
        trunk_top = np.array([tx + 0.15, ty - 0.45 * th])
        limbs = []
        for m in range(7):
            a = -np.pi / 2 + (m - 3) * 0.26 + rng.normal(0, 0.05)
            L_ = th * (0.36 + 0.12 * rng.random()) * (1.0 - 0.12 * abs(m - 3))
            p0 = trunk_top + np.array([0.04 * (m - 3), 0.25 * abs(m - 3) * 0.2])
            p1 = p0 + 0.5 * L_ * np.array([math.cos(a) * 0.7, math.sin(a)])
            p2 = p0 + L_ * np.array([math.cos(a) * 1.25, math.sin(a) * 0.95])
            limbs.append(catmull([p0, p1, p2], 10))
        # the trunk: two edges, a slight flare at the foot, smooth silver bark (sparse strokes on the shade side)
        hs = np.linspace(0, 1, 60)
        cx_ = tx + 0.15 * hs + 0.04 * np.sin(hs * 5)
        wdt = 0.2 * (1 - 0.45 * hs) + 0.18 * np.exp(-hs / 0.06)
        yy = ty - 0.45 * th * hs
        line(S, np.column_stack([cx_ - wdt, yy]), W(0.02), self.seed + 5, dens=dens, layer=lay, lift=(3, 6), smooth=0)
        line(S, np.column_stack([cx_ + wdt, yy]), W(0.026), self.seed + 6, dens=dens, layer=lay, lift=(3, 6), smooth=0)
        if not pencil:
            for q in np.linspace(0.25, 0.9, 5):
                xs_ = cx_ + wdt * q
                pp, rd, dd = hand(np.column_stack([xs_, yy])[3:-2], 0.007, int(rng.integers(1 << 30)), dens=0.7,
                                  thin_end=0.3, taper=(0.3, 0.3))
                S.add(pp, rd, dd, layer=lay)
            for m in range(6):              # the faint rings of young bark
                hq = rng.uniform(0.1, 0.9)
                i = int(hq * 59)
                q = np.array([[cx_[i] - wdt[i] * 0.6, yy[i]], [cx_[i] + wdt[i] * 0.9, yy[i] + 0.03]])
                pp, rd, dd = hand(q, 0.006, int(rng.integers(1 << 30)), dens=0.6, thin_end=0.4)
                S.add(pp, rd, dd, layer=lay)
        for li, lb in enumerate(limbs):
            line(S, lb, W(0.02 - 0.0015 * abs(li - 3)), self.seed + 10 + li, dens=dens, layer=lay, lift=(3, 5), smooth=0,
                 taper=(0.05, 0.4))
        _window(S, k, *T(0.05, 0.25), overlap=0.2)

        # the crown: leaf clusters (scalloped), shaded on the lower right; golden flowers among them
        k = len(S)
        crown_c = np.array([tx + 0.3, ty - 0.74 * th])
        clusters = []
        for m in range(46):
            a = rng.uniform(0, 2 * np.pi)
            r_ = math.sqrt(rng.random())
            c = crown_c + np.array([2.4 * r_ * math.cos(a), 1.75 * r_ * math.sin(a)])
            clusters.append((c[0], c[1], rng.uniform(0.33, 0.55)))
        clusters.sort(key=lambda q: q[1])
        cxs = np.array([q[0] for q in clusters])
        cys = np.array([q[1] for q in clusters])
        crs = np.array([q[2] for q in clusters])
        flowers = []
        for ci, (x, y, r) in enumerate(clusters):
            a = np.linspace(0, 2 * np.pi, 90)
            sc = 1.0 + 0.1 * np.abs(np.sin(a * 4.5 + ci))           # scalloped edge: leaves
            px_ = x + r * sc * np.cos(a)
            py_ = y + r * sc * np.sin(a) * 0.85
            vis = np.ones(len(a), bool)
            for cj in range(ci + 1, len(clusters)):
                vis &= (px_ - cxs[cj]) ** 2 + (py_ - cys[cj]) ** 2 > (crs[cj] * 0.97) ** 2
            dm = np.diff(np.concatenate([[0], vis.astype(np.int8), [0]]))
            for a_, b_ in zip(np.nonzero(dm == 1)[0], np.nonzero(dm == -1)[0] - 1):
                if b_ - a_ < 6:
                    continue
                pp, rd, dd = hand(np.column_stack([px_[a_:b_ + 1], py_[a_:b_ + 1]]), W(0.014), int(rng.integers(1 << 30)),
                                  dens=dens, taper=(0.04, 0.1), thin_end=0.3)
                S.add(pp, rd, dd, layer=lay)
            if not pencil:
                # leaves: short curved ticks on the shaded lower right of each cluster
                for q in range(int(10 * r / 0.4)):
                    aa = rng.uniform(-0.4, 1.9)
                    rr_ = r * rng.uniform(0.35, 0.85)
                    p0 = np.array([x + rr_ * math.cos(aa), y + 0.85 * rr_ * math.sin(aa)])
                    if any((p0[0] - cxs[cj]) ** 2 + (p0[1] - cys[cj]) ** 2 < crs[cj] ** 2 for cj in range(ci + 1, len(clusters))):
                        continue
                    p1 = p0 + 0.09 * np.array([math.cos(aa + 1.9), math.sin(aa + 1.9)])
                    pp, rd, dd = hand(np.array([p0, (p0 + p1) / 2 + [0.01, -0.01], p1]), 0.008, int(rng.integers(1 << 30)),
                                      dens=0.85, thin_end=0.25)
                    S.add(pp, rd, dd, layer=lay)
                # flowers on the lit upper left of the cluster
                for q in range(int(rng.integers(2, 5))):
                    aa = rng.uniform(2.6, 5.0)
                    rr_ = r * rng.uniform(0.2, 0.75)
                    fx_ = x + rr_ * math.cos(aa)
                    fy_ = y + 0.85 * rr_ * math.sin(aa)
                    if any((fx_ - cxs[cj]) ** 2 + (fy_ - cys[cj]) ** 2 < crs[cj] ** 2 for cj in range(ci + 1, len(clusters))):
                        continue
                    flowers.append((fx_, fy_))
        _window(S, k, *T(0.25, 0.55), overlap=0.85)

        # the far country: hills, a patchwork of fields, hedgerows, an orchard, stooks, cottages with smoke
        k = len(S)
        xs = np.linspace(bx0 + 0.25, bx1 - 0.25, 300)
        for kk in range(3):
            ys_ = self.hills(xs, kk)
            keep = ~(((xs - crown_c[0]) / 2.7) ** 2 + ((ys_ - crown_c[1]) / 2.0) ** 2 < 1) & \
                ~((np.abs(xs - tx) < 0.35) & (ys_ > crown_c[1]))
            dm = np.diff(np.concatenate([[0], keep.astype(np.int8), [0]]))
            for a_, b_ in zip(np.nonzero(dm == 1)[0], np.nonzero(dm == -1)[0] - 1):
                if b_ - a_ > 5:
                    line(S, np.column_stack([xs[a_:b_ + 1], ys_[a_:b_ + 1]]), W(0.016 + 0.004 * kk), int(rng.integers(1 << 30)),
                         dens=dens * (0.8 + 0.07 * kk), layer=lay, lift=(1.5, 4.0), smooth=0)
        if not pencil:
            # fields between the ridges: each a patch of plough-lines at its own angle, some left as stubble dots
            def free(x, y):
                x = np.asarray(x)
                y = np.asarray(y)
                ok = ~((((x - crown_c[0]) / 2.75) ** 2 + ((y - crown_c[1]) / 2.05) ** 2) < 1)
                ok &= ~((np.abs(x - tx - 0.1) < 0.42) & (y > crown_c[1]) & (y < ty + 0.1))
                return ok
            for kk in range(2):
                y_lo = lambda x, kk=kk: self.hills(x, kk)
                y_hi = lambda x, kk=kk: self.hills(x, kk + 1)
                xcuts = np.sort(rng.uniform(bx0 + 0.3, bx1 - 0.3, 5 + 2 * kk))
                xcuts = np.concatenate([[bx0 + 0.25], xcuts, [bx1 - 0.25]])
                for fi in range(len(xcuts) - 1):
                    xa, xb = xcuts[fi], xcuts[fi + 1]
                    ang = rng.choice([0, 12, -14, 25, -30, 8]) + rng.normal(0, 3)
                    sp = rng.uniform(0.06, 0.1)

                    def inside(x, y, xa=xa, xb=xb, y_lo=y_lo, y_hi=y_hi):
                        x = np.asarray(x)
                        y = np.asarray(y)
                        return (x > xa + 0.05) & (x < xb - 0.05) & (y > y_lo(x) + 0.06) & (y < y_hi(x) - 0.06) & free(x, y)
                    if rng.random() < 0.72:
                        hatch(S, inside, lambda x, y: np.ones_like(x), (xa, self.Yh - 1.0, xb, self.hills(xb, 2) + 0.5),
                              ang, sp, 0.0, 0.0075, int(rng.integers(1 << 30)), dens=0.8, seg=(0.5, 2.0), wob=0.003)
                    else:
                        stipple(S, inside, lambda x, y: np.full_like(x, 0.55), (xa, self.Yh - 1.0, xb, self.hills(xb, 2) + 0.5),
                                260, 0.012, int(rng.integers(1 << 30)))
                    # the hedge along the field's edge: a row of small round bushes
                    hx = np.arange(xa + 0.1, xb - 0.1, 0.16)
                    for x_ in hx:
                        y_ = float(self.hills(np.array([x_]), kk + 1)[0]) - 0.04
                        if not free(np.array([x_]), np.array([y_]))[0] or rng.random() < 0.2:
                            continue
                        a = np.linspace(np.pi * 0.95, np.pi * 2.05, 14)
                        r_ = 0.07 + 0.03 * rng.random()
                        q = np.column_stack([x_ + r_ * np.cos(a), y_ + r_ * np.sin(a)])
                        pp, rd, dd = hand(q, 0.009, int(rng.integers(1 << 30)), dens=0.9, thin_end=0.4, taper=(0.02, 0.02))
                        S.add(pp, rd, dd, layer=lay)
            # stooks of corn in the near field (little tents of sheaves), in rows
            for row in range(3):
                yb_ = self.hills(np.array([bx0 + 10.0]), 2)[0] + 0.55 + 0.45 * row
                for x_ in np.arange(bx0 + 8.6 + 0.2 * row, bx1 - 0.8, 0.62 + 0.1 * row):
                    if not free(np.array([x_]), np.array([yb_]))[0]:
                        continue
                    hh = 0.26 + 0.05 * row
                    q = np.array([[x_ - 0.1 - 0.02 * row, yb_], [x_, yb_ - hh], [x_ + 0.11 + 0.02 * row, yb_]])
                    line(S, q, 0.012, int(rng.integers(1 << 30)), dens=0.95, layer=lay, lift=(3, 4), smooth=0)
                    for f in np.linspace(0.25, 0.75, 3):
                        p0 = q[1] + (q[2] - q[1]) * f
                        pp, rd, dd = hand(np.array([q[1] + [0.005, 0.02], p0]), 0.006, int(rng.integers(1 << 30)), dens=0.8,
                                          thin_end=0.4)
                        S.add(pp, rd, dd, layer=lay)
            # the orchard on the far slope: rows of small round trees
            for row in range(3):
                for x_ in np.arange(bx0 + 0.8 + 0.2 * row, bx0 + 3.6, 0.34):
                    y_ = float(self.hills(np.array([x_]), 1)[0]) + 0.25 + 0.3 * row
                    a = np.linspace(0, 2 * np.pi, 20)
                    r_ = 0.1 + 0.015 * row
                    q = np.column_stack([x_ + r_ * np.cos(a), y_ - r_ + r_ * np.sin(a)])
                    pp, rd, dd = hand(q, 0.009, int(rng.integers(1 << 30)), dens=0.9, thin_end=0.7, taper=(0, 0))
                    S.add(pp, rd, dd, layer=lay)
                    pp, rd, dd = hand(np.array([[x_, y_], [x_, y_ + 0.08]]), 0.01, int(rng.integers(1 << 30)), dens=0.9)
                    S.add(pp, rd, dd, layer=lay)
            # cottages: gabled roofs, small windows, a chimney each with a thread of smoke rising straight up
            for (cx, kk, w_) in ((bx0 + 10.4, 0, 0.55), (bx0 + 11.3, 0, 0.45), (bx0 + 12.6, 1, 0.6), (bx0 + 13.7, 0, 0.5),
                                 (bx0 + 1.2, 0, 0.45)):
                gy_ = float(self.hills(np.array([cx]), kk)[0]) + 0.28
                h_ = 0.3 * w_ / 0.5
                body = np.array([[cx - w_ / 2, gy_], [cx - w_ / 2, gy_ - h_], [cx + w_ / 2, gy_ - h_], [cx + w_ / 2, gy_]])
                roof = np.array([[cx - w_ / 2 - 0.06, gy_ - h_], [cx - 0.05, gy_ - h_ - 0.3 * w_ / 0.5],
                                 [cx + w_ / 2 + 0.06, gy_ - h_]])
                line(S, body, 0.012, int(rng.integers(1 << 30)), dens=0.95, layer=lay, lift=(4, 5), smooth=0)
                line(S, roof, 0.014, int(rng.integers(1 << 30)), dens=0.95, layer=lay, lift=(4, 5), smooth=0)
                hatch(S, _poly_inside(roof), lambda x, y: np.ones_like(x), (cx - w_, gy_ - h_ - 0.4, cx + w_, gy_ - h_),
                      70, 0.03, 0.0, 0.007, int(rng.integers(1 << 30)), dens=0.85, seg=(0.2, 0.5))
                chx = cx + 0.22 * w_
                chy = gy_ - h_ - 0.14 * w_ / 0.5
                line(S, np.array([[chx, chy + 0.04], [chx, chy - 0.12], [chx + 0.06, chy - 0.12], [chx + 0.06, chy + 0.01]]),
                     0.01, int(rng.integers(1 << 30)), dens=0.95, layer=lay, lift=(3, 4), smooth=0)
                sm = np.array([[chx + 0.03, chy - 0.16 - 0.35 * q] for q in range(7)])
                sm[:, 0] += 0.05 * np.sin(np.arange(7) * 1.3)
                pp, rd, dd = hand(catmull(sm, 6), 0.007, int(rng.integers(1 << 30)), dens=0.75, thin_end=0.1, taper=(0.02, 0.8))
                S.add(pp, rd, dd, layer=lay)
                for wx in (cx - 0.12 * w_ / 0.5, cx + 0.1 * w_ / 0.5):
                    q = np.array([[wx, gy_ - 0.55 * h_], [wx + 0.001, gy_ - 0.54 * h_]])
                    S.add(q, np.array([0.02, 0.02]), np.array([0.9, 0.9]), layer=lay)
        _window(S, k, *T(0.45, 0.8), overlap=0.9)

        # the near field: long grass, a few wild flowers; the tree's shadow falling to the right
        k = len(S)
        if not pencil:
            yg = self.hills(np.array([bx0 + 7.0]), 2)[0] + 0.3
            y_ = yg
            while y_ < by1 - 0.12:
                x_ = bx0 + 0.3 + rng.random() * 0.3
                while x_ < bx1 - 0.3:
                    hh = (0.05 + 0.3 * (y_ - yg) / (by1 - yg)) * rng.uniform(0.6, 1.3)
                    near_tree = abs(x_ - tx) < 0.35 and abs(y_ - ty) < 0.3
                    if not near_tree and rng.random() < 0.6:
                        lean = rng.normal(0.08, 0.05)
                        q = np.array([[x_, y_], [x_ + lean * 0.5, y_ - hh * 0.6], [x_ + lean, y_ - hh]])
                        pp, rd, dd = hand(q, 0.008, int(rng.integers(1 << 30)), dens=0.85, thin_end=0.15, taper=(0.01, 0.6))
                        S.add(pp, rd, dd, layer=lay)
                    x_ += rng.uniform(0.05, 0.22) * (1.0 + 1.5 * (y_ - yg) / (by1 - yg))
                y_ += 0.12 + 0.14 * (y_ - yg) / (by1 - yg)
            # the shadow of the tree on the grass (to the right, the light is from the left)
            sh = np.array([[tx + 0.1, ty + 0.02], [tx + 1.8, ty + 0.25], [tx + 3.2, ty + 0.3], [tx + 3.4, ty + 0.45],
                           [tx + 1.6, ty + 0.5], [tx - 0.1, ty + 0.2]])
            hatch(S, _poly_inside(sh), lambda x, y: np.ones_like(x), (tx - 0.2, ty - 0.1, tx + 3.6, ty + 0.6), 0, 0.04,
                  0.0, 0.008, self.seed + 900, dens=0.8, seg=(0.3, 1.2))
            # birds, far off
            for m in range(5):
                bx_ = bx0 + 9.5 + rng.random() * 4.5
                by_ = by0 + 1.2 + rng.random() * 2.2
                s_ = 0.08 + 0.05 * rng.random()
                q = np.array([[bx_ - s_, by_ - 0.3 * s_], [bx_ - 0.4 * s_, by_ - 0.45 * s_], [bx_, by_],
                              [bx_ + 0.4 * s_, by_ - 0.45 * s_], [bx_ + s_, by_ - 0.3 * s_]])
                pp, rd, dd = hand(catmull(q, 4), 0.008, int(rng.integers(1 << 30)), dens=0.9, thin_end=0.3)
                S.add(pp, rd, dd, layer=lay)
        _window(S, k, *T(0.7, 0.92), overlap=0.9)

        # the gold: the flowers, laid last, one by one
        k = len(S)
        for (fx_, fy_) in flowers:
            if pencil:
                a = np.linspace(0, 2 * np.pi, 12)
                q = np.column_stack([fx_ + 0.035 * np.cos(a), fy_ + 0.035 * np.sin(a)])
                pp, rd, dd = hand(q, 0.008, int(rng.integers(1 << 30)), dens=0.4, thin_end=0.8, taper=(0, 0))
                S.add(pp, rd, dd, layer=lay)
                continue
            for p_ in range(5):
                a = p_ * 1.2566 + rng.uniform(0, 0.5)
                q = np.array([[fx_, fy_], [fx_ + 0.05 * math.cos(a), fy_ + 0.05 * math.sin(a)]])
                S.add(q, np.array([0.02, 0.016]), np.array([1.0, 1.0]), layer=GILT)
        _window(S, k, *T(0.9, 1.0), overlap=0.7)
        return S
