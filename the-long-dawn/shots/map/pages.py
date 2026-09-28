"""THE INK PAGES (cut C, P2) and the Red Book's written leaves (P1), drawn stroke by stroke.

Each page is built as pen.Strokes in page cm (20 x 29, y down), in the order an illustrator works (frame,
the big contour, the dark accents, hatching from the top down, the sky, the ground), with draw windows in
shot seconds, so any frame of the drawing-on is a pure function of time. Every page has two states: PENCIL
(the underdrawing: construction lines, sketchy repeated contours, loose shading) and INK (iron-gall pen,
hatching that follows the form and the light, gold leaf only where the story puts it).

The drawings' own light comes from the upper left, like the hearth that lights the page.
"""
import math

import cv2
import numpy as np

import pen
from pen import INK, PENCIL, GILT, Strokes, hand, hatch, line, stipple, catmull, resample, arclen

PW, PH = 20.0, 29.0


def smooth_(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3 - 2 * x)


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
            r = w[i] * (grow[0] + (grow[1] - grow[0]) * rng.random() ** 1.8) * (0.62 if k else 1.0)
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
        """Half-width at height fraction h (0 base .. 1 rim), with ridges by azimuth phi. MAP-L: "a great cone on
        a huge base": a broad base whose outer flank climbs to a shelf, then the steep concave cone; the two sides
        differ (the left shelf higher and wider), and the flanks are uneven, never a clean triangle."""
        h = np.clip(h, 0.0, 1.0)
        phi = np.asarray(phi, np.float64)
        right = (phi > 0).astype(np.float64)
        hb = 0.31 - 0.07 * right                     # the shelf
        Wm = self.Wb * (0.52 - 0.05 * right)         # the cone's half-width at the shelf
        T = 0.85 - 0.25 * right                      # the shelf's depth
        hc = np.clip((h - hb) / (1.0 - hb), 0, 1)
        cone = self.Wt + (Wm - self.Wt) * (1.0 - hc) ** 1.3
        a = np.clip(h / (0.85 * hb), 0, 1)
        outer = (self.Wb - Wm - T) * (1.0 - a) ** 0.9
        terr = T * (1.0 - smooth_((h - 0.85 * hb) / (0.22 * hb)))
        base = cone + outer + terr
        une = 0.07 * np.sin(h * 8.0 + self.rphase[3] + 2.1 * right) + 0.04 * np.sin(h * 21.0 + self.rphase[4] + right)
        rid = 0.035 * np.sin(5.0 * phi + self.rphase[0]) + 0.02 * np.sin(11.0 * phi + self.rphase[1])
        return base * (1.0 + rid * (0.3 + h)) + une * (0.35 + 0.65 * (1.0 - h))

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
        c = catmull([(self.Xc + 0.05, self.Ys - 0.8), (self.Xc + 0.12, self.Ys - 1.3), (self.Xc + 0.5, by0 + 1.4),
                     (self.Xc + 1.7, by0 + 1.05), (self.Xc + 3.5, by0 + 1.1), (self.Xc + 5.4, by0 + 1.35), (bx1 - 0.7, by0 + 1.55)], 12)
        c = resample(c, 0.05)
        t = arclen(c) / arclen(c)[-1]
        w = (0.24 + 0.6 * t ** 0.6) * (1.0 - 0.45 * np.clip((t - 0.78) / 0.22, 0, 1))
        return c, w

    SCHED = dict(frame=(0.0, 0.07), cons=(0.07, 0.15), sil=(0.1, 0.24), fire=(0.24, 0.34), ring=(0.34, 0.4),
                 hatch=(0.4, 0.7), smoke=(0.58, 0.78), sky=(0.72, 0.9), ground=(0.84, 1.0))
    # C3 (locked): the pen begins after the riffle; the fire and the ring come first (the ring on bar 5 b3)
    SCHED_C3 = dict(frame=(0.0, 0.03), cons=(0.0, 0.03), fire=(0.02, 0.07), ring=(0.07, 0.085), sil=(0.09, 0.22),
                    hatch=(0.2, 0.55), smoke=(0.45, 0.68), sky=(0.6, 0.85), ground=(0.78, 1.0))

    def build(self, mode='ink', t0=0.0, t1=8.0, sched=None):
        """Strokes for the plate, drawn on across [t0, t1] (mode 'ink' or 'pencil'); sched maps each section
        (frame, cons, sil, fire, ring, hatch, smoke, sky, ground) to its window as fractions of [t0, t1]."""
        rng = np.random.default_rng(self.seed)
        S = Strokes()
        lay = INK if mode == 'ink' else PENCIL
        pencil = mode == 'pencil'
        bx0, by0, bx1, by1 = self.box
        sched_ = dict(self.SCHED)
        sched_.update(sched or {})
        T = lambda a, b: (t0 + (t1 - t0) * a, t0 + (t1 - t0) * b)
        TS = lambda key: T(*sched_[key])
        W = (lambda w: w * 1.15) if pencil else (lambda w: w)
        dens = 0.55 if pencil else 0.95
        fx, fy = self.ring
        c_pl, w_pl = self.plume()
        circles, outl = billows(c_pl, w_pl, np.random.default_rng(self.seed + 5), grow=(0.38, 1.08), step=0.45, jitter=0.45)
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
        _window(S, k, *TS('frame'), overlap=0.3)

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
            _window(S, k, *TS('cons'), overlap=0.3)

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
        _window(S, k, *TS('sil'), overlap=0.1)

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
            line(S, np.array(L_), W(0.02), self.seed + 40 + m, dens=dens, layer=lay, lift=(4, 6), smooth=10,
                 taper=(0.05, 0.25))
            line(S, np.array(R_), W(0.017), self.seed + 45 + m, dens=dens, layer=lay, lift=(4, 6), smooth=10,
                 taper=(0.2, 0.05))
            if not pencil:
                # PAGES-C: a tongue within the tongue, as an engraver draws fire (the outlines alone read as hair)
                I_ = [base + [-wdt * 0.45, -0.04 * hgt], base + [-wdt * 0.55 + lean * 0.25, -0.34 * hgt],
                      base + [-wdt * 0.1 + lean * 0.5 + sgn * 0.03, -0.58 * hgt], base + [lean * 0.8 + sgn * 0.07, -0.8 * hgt]]
                line(S, np.array(I_), W(0.011), self.seed + 50 + m, dens=dens * 0.9, layer=lay, lift=(4, 6), smooth=10,
                     taper=(0.1, 0.4))
                J_ = [base + [wdt * 0.5, -0.03 * hgt], base + [wdt * 0.55 + lean * 0.3, -0.3 * hgt],
                      base + [wdt * 0.15 + lean * 0.55 - sgn * 0.03, -0.5 * hgt]]
                line(S, np.array(J_), W(0.009), self.seed + 55 + m, dens=dens * 0.85, layer=lay, lift=(4, 6), smooth=10,
                     taper=(0.1, 0.4))
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
        _window(S, k, *TS('fire'), overlap=0.3)

        # the ring: gold leaf, outlined in fine ink
        k = len(S)
        a = np.linspace(0, 2 * np.pi, 120)
        rx, ry = 0.25, 0.185                        # PAGES-C: bigger and bolder (it read as a 20 px outline)
        ringp = np.column_stack([fx + rx * np.cos(a), fy + ry * np.sin(a)])
        if pencil:
            pp, rr, dd = hand(ringp, 0.014, self.seed + 60, slow=(1.0, 0.004), dens=0.5, thin_end=0.8,
                              taper=(0.0, 0.0))
            S.add(pp, rr, dd, layer=lay)
        else:
            # the band: gold leaf, a little heavier on the near side as a ring seen from above is
            S.add(ringp, 0.042 + 0.012 * np.clip(np.sin(a), 0, 1), np.full(len(ringp), 1.0), layer=GILT)
            for sc in (1.22, 0.76):
                q = np.column_stack([fx + rx * sc * np.cos(a), fy + ry * sc * np.sin(a)])
                pp, rr, dd = hand(q, 0.006, self.seed + 61, slow=(1.0, 0.002), dens=0.8, thin_end=0.9,
                                  taper=(0, 0), fast=(0.2, 0.0008))
                S.add(pp, rr, dd, layer=INK)
        _window(S, k, *TS('ring'))

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
        _window(S, k, *TS('hatch'), overlap=0.93, order=[i - k for i in idx])

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
                for q in range(int(8 + 34 * r)):
                    rr_ = r * rng.uniform(0.45, 0.93)
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
        _window(S, k, *TS('smoke'), overlap=0.6)

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
        _window(S, k, *TS('sky'), overlap=0.8)

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
        door = self.surf(0.36, -0.52)          # (H5: no dotted route to it: it read as a treasure map)
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
        _window(S, k, *TS('ground'), overlap=0.85)
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

        # the seam (H5): a gold vein, thick and tapered, pinching and swelling as real veins do, branching into
        # stringers; gilt, outlined in fine ink; it thickens the deeper they follow it
        vs = arclen(self.vein)
        VL = vs[-1]
        grow = 0.1 + 0.2 * (vs / VL) ** 1.2                                   # cm: thicker with depth
        swell = 0.55 + 0.45 * np.abs(np.sin(vs * 1.9 + 0.7)) + 0.12 * np.sin(vs * 5.3)
        ends = np.clip(vs / 0.9, 0, 1) ** 0.6 * np.clip((VL - vs) / 1.2, 0, 1) ** 0.5   # tapered where it starts and ends
        vein_w = grow * swell * ends
        vn = pen.normals(self.vein)
        for k_h in range(len(self.halls) + 1):
            ya = by0 + 1.2 if k_h == 0 else self.halls[k_h - 1]['y1']
            yb = self.halls[k_h]['y1'] if k_h < len(self.halls) else self.glow[1] - 0.1
            m = (self.vein[:, 1] >= ya) & (self.vein[:, 1] <= yb)
            if m.sum() < 2:
                continue
            seg = self.vein[m]
            w_ = vein_w[m]
            nr = vn[m]
            a_, b_ = pace(min(k_h, len(self.halls)))
            k = len(S)
            if pencil:
                for side in (-1, 1):
                    pp, rd, dd = hand(seg + nr * (side * 0.5 * w_)[:, None], 0.014, self.seed + 700 + 2 * k_h + side,
                                      dens=0.45, taper=(0.1, 0.1))
                    S.add(pp, rd, dd, layer=lay)
            else:
                # the gilt body: a band of overlapping pulls across its width, the edges wandering
                for q in np.linspace(-0.42, 0.42, 5):
                    wob = 0.08 * np.sin(np.arange(len(seg)) * 0.37 + q * 9.0)
                    S.add(seg + nr * ((q + wob) * w_)[:, None], 0.3 * w_ + 0.004, np.ones(len(seg)), layer=GILT)
                for side in (-1, 1):
                    edge = seg + nr * (side * (0.5 * w_ + 0.01 + 0.012 * np.sin(np.arange(len(seg)) * 0.5 + side)))[:, None]
                    pp, rd, dd = hand(edge, 0.008, self.seed + 710 + k_h * 3 + side, dens=0.85, thin_end=0.5,
                                      taper=(0.08, 0.08))
                    S.add(pp, rd, dd, layer=INK)
                # stringers: branches leaving at acute angles, tapering to nothing, some forking again
                for b in range(3 + 2 * k_h):
                    i = int(rng.integers(3, max(len(seg) - 3, 4)))
                    sgn = 1 if rng.random() < 0.5 else -1
                    t_ = np.array([seg[min(i + 1, len(seg) - 1)] - seg[max(i - 1, 0)]])[0]
                    t_ = t_ / (np.linalg.norm(t_) + 1e-9)
                    d = 0.55 * t_ + sgn * 0.85 * nr[i]
                    Lb = rng.uniform(0.5, 1.4) * (0.7 + 0.15 * k_h)
                    pts = [seg[i]]
                    for q in range(5):
                        d = d + rng.normal(0, 0.25, 2)
                        d /= np.linalg.norm(d)
                        pts.append(pts[-1] + d * Lb / 5)
                    br = resample(catmull(np.array(pts), 6), 0.02)
                    wb = np.linspace(0.42 * w_[i], 0.004, len(br))
                    S.add(br, 0.5 * wb, np.ones(len(br)), layer=GILT)
                    if w_[i] > 0.12:
                        for side in (-1, 1):
                            bn = pen.normals(br)
                            pp, rd, dd = hand(br + bn * (side * (0.5 * wb + 0.008))[:, None], 0.006,
                                              int(rng.integers(1 << 30)), dens=0.75, thin_end=0.2, taper=(0.02, 0.3))
                            S.add(pp, rd, dd, layer=INK)
                    if rng.random() < 0.45 and len(br) > 8:
                        j = int(len(br) * rng.uniform(0.3, 0.6))
                        d2 = d + sgn * 0.6 * np.array([-d[1], d[0]])
                        d2 /= np.linalg.norm(d2)
                        tw = np.array([br[j], br[j] + d2 * Lb * 0.35 + rng.normal(0, 0.05, 2)])
                        tw = resample(tw, 0.02)
                        S.add(tw, 0.5 * np.linspace(wb[j], 0.003, len(tw)), np.ones(len(tw)), layer=GILT)
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
        self.tx = bx0 + 6.0            # the tree's foot
        self.ty = by0 + 12.9
        self.th = 10.5                 # its height

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

        # the tree (MAP-L redraw): a young mallorn, silver-barked, with long leaves and golden flowers. A trunk
        # with a flared foot forks into a vase of limbs; the limbs branch to drooping twigs; the twigs carry long
        # pointed leaves, drawn back to front (each hides what is behind it), the shaded ones ribbed and hatched;
        # racemes of golden flowers hang from the twig ends, thickest on the lit side. Sky shows between the limbs.
        k = len(S)
        tx, ty, th = self.tx, self.ty, self.th
        cc = np.array([tx + 0.25, ty - 0.64 * th])
        R = np.array([0.31 * th, 0.27 * th])
        fork = np.array([tx + 0.1, ty - 0.3 * th])
        segs, tips = [], []

        def rot(d, a):
            ca, sa = math.cos(a), math.sin(a)
            return np.array([ca * d[0] - sa * d[1], sa * d[0] + ca * d[1]])

        def ovo(p):
            q = (p - cc) / R
            return q[0] ** 2 + q[1] ** 2 * (1.0 if q[1] < 0 else 1.45)

        def grow(p, d, Ln, w, depth):
            n = 9
            pts = [p.copy()]
            q, dd = p.copy(), d / np.linalg.norm(d)
            bend = rng.normal(0, 0.32) / n
            inside0 = ovo(p) < 1.0
            for i in range(n):
                ang = bend
                if depth == 0:                       # twigs droop toward their ends
                    ang += 0.07 * np.sign(dd[0] + 1e-6) * (i / n)
                if depth == 3:                       # main limbs: out from the leader, then turning up
                    ang += -0.09 * np.sign(dd[0] + 1e-6) * (1.0 - i / n)
                dd = rot(dd, ang)
                if depth <= 1:
                    dd = dd + np.array([0.0, 0.035 * (i / n)])
                    dd = dd / np.linalg.norm(dd)
                q2 = q + dd * Ln / n
                if depth <= 2 and ovo(q2) > 0.92 and (inside0 or i > 2):    # branches end inside the crown
                    break
                q = q2
                pts.append(q.copy())
            P = np.array(pts)
            if len(P) < 3:
                return
            n = len(P) - 1
            segs.append((P, w))
            if depth == 0:
                tips.append((P, dd))
                return
            if depth == 1:
                tips.append((P[n // 2:], dd))
            sg = rng.choice([-1, 1])
            if depth >= 2:
                kids = [(0.45, sg * rng.uniform(0.5, 0.85)), (0.75, -sg * rng.uniform(0.4, 0.7)), (1.0, rng.normal(0, 0.15))]
            else:
                kids = [(0.6, sg * rng.uniform(0.5, 0.8)), (1.0, rng.normal(0, 0.15))]
            for frac, a in kids:
                i = int(round(frac * n))
                grow(P[i], rot(dd if frac >= 1.0 else (P[min(i + 1, n)] - P[max(i - 1, 0)]), a), Ln * rng.uniform(0.55, 0.7),
                     w * 0.62, depth - 1)
        for li_, a in enumerate((-0.56, -0.19, 0.17, 0.52)):
            # each limb leaves the leader at its own height (never all from one point), first outward, then up
            org = fork + np.array([0.02 * a, -(0.0, 0.55, 0.3, 0.85)[li_]])
            d0 = rot(np.array([0.0, -1.0]), 1.35 * a + rng.normal(0, 0.06))
            # the limb's length: 60% of the way from the fork to where its line leaves the crown's ovoid (the fork
            # sits below the crown, so march out along the ray and take the far exit)
            ts_ = np.linspace(0.0, 9.0, 361)
            ins = np.array([ovo(org + d0 * tt) < 1.0 for tt in ts_])
            exit_ = ts_[np.nonzero(ins)[0][-1]] if ins.any() else 0.5 * R[1]
            grow(org, d0, 0.62 * exit_, 0.068 - 0.006 * li_, 3)
        # the leader: the trunk carried up through the limbs into the crown
        lead = catmull([fork + np.array([0.0, 0.1]), fork + np.array([0.05, -0.6]), fork + np.array([-0.04, -1.2]),
                        cc + np.array([0.08, 0.35 * R[1]])], 8)
        segs.append((lead, 0.07))

        # the leaves: sprays of long pointed leaves, each spray on its own short twig, filling the crown's ovoid
        # (irregular edge, flatter below) except for a few windows of sky where the limbs show
        wins = [cc + np.array([-0.42, 0.42]) * R, cc + np.array([0.38, 0.28]) * R, cc + np.array([0.02, 0.62]) * R,
                cc + np.array([-0.15, -0.2]) * R]
        wr = [0.42, 0.36, 0.34, 0.26]
        clumps = []
        tries = 0
        while len(clumps) < 78 and tries < 6000:
            tries += 1
            ang_ = rng.uniform(0, 2 * np.pi)
            rr_ = math.sqrt(rng.random())
            edge = 1.0 + 0.1 * math.sin(3 * ang_ + 1.0) + 0.06 * math.sin(7 * ang_)
            p_ = cc + R * edge * np.array([rr_ * math.cos(ang_), rr_ * math.sin(ang_)])
            if p_[1] > cc[1] + 0.62 * R[1] * (1 - 0.5 * abs(math.cos(ang_))):
                continue
            if any(np.hypot(*(p_ - w_)) < r_ for w_, r_ in zip(wins, wr)):
                continue
            if any(np.hypot(*(p_ - q_[0])) < 0.34 for q_ in clumps):
                continue
            out = (p_ - cc) / R
            out = out / max(np.linalg.norm(out), 1e-6)
            dirc = out * 0.8 + np.array([0.0, 0.55]) + rng.normal(0, 0.25, 2)     # outward, drooping
            dirc = dirc / np.linalg.norm(dirc)
            clumps.append((p_, dirc, rng.random() + 0.35 * float(out[1])))
        leaves = []
        twigs = []
        for (p_, dirc, z) in clumps:
            base = p_ - dirc * 0.22
            twigs.append((np.array([base, p_ + dirc * 0.12]), z))
            nl = int(rng.integers(8, 14))
            for j in range(nl):
                f_ = rng.uniform(0.0, 1.0)
                bp = base + dirc * (0.34 * f_)
                ang = (1 if j % 2 else -1) * rng.uniform(0.35, 1.25) * (1.0 - 0.5 * f_)
                ld = rot(dirc, ang) + np.array([0.0, rng.uniform(0.1, 0.4)])
                ld = ld / np.linalg.norm(ld)
                ln = rng.uniform(0.26, 0.42)
                leaves.append((bp, ld, ln, ln * rng.uniform(0.17, 0.23), z + rng.uniform(-0.05, 0.05)))
        leaves.sort(key=lambda q: q[4])                                  # back to front
        nL = len(leaves)
        LC = np.array([q[0] + q[1] * 0.5 * q[2] for q in leaves])     # leaf centres
        LD = np.array([q[1] for q in leaves])
        LA = np.array([0.5 * q[2] for q in leaves])
        LB = np.array([0.5 * q[3] for q in leaves])

        def in_leaf(x, y, idx, grow_=1.0):
            dx = x[:, None] - LC[idx, 0][None]
            dy = y[:, None] - LC[idx, 1][None]
            a_ = dx * LD[idx, 0][None] + dy * LD[idx, 1][None]
            b_ = -dx * LD[idx, 1][None] + dy * LD[idx, 0][None]
            return ((a_ / (LA[idx][None] * grow_)) ** 2 + (b_ / (LB[idx][None] * grow_)) ** 2) < 1.0

        # a coverage map of the foliage for everything drawn behind it (limbs, hills, fields)
        gx0, gy0 = cc[0] - R[0] - 1.0, cc[1] - R[1] - 1.2
        gres = 25.0
        gw_, gh_ = int((2 * R[0] + 2.0) * gres), int((2 * R[1] + 2.6) * gres)
        fmask = np.zeros((gh_, gw_), np.uint8)
        for (bp, ld, ln, lw, z) in leaves:
            c_ = (bp + ld * 0.5 * ln - [gx0, gy0]) * gres
            ang = math.degrees(math.atan2(ld[1], ld[0]))
            cv2.ellipse(fmask, (int(c_[0] * 16), int(c_[1] * 16)), (int(0.5 * ln * gres * 16), int(0.5 * lw * gres * 16)),
                        ang, 0, 360, 1, -1, cv2.LINE_8, 4)
        fdist = cv2.distanceTransform((1 - fmask).astype(np.uint8), cv2.DIST_L2, 3) / gres

        def leafy(x, y, pad=0.0):
            x = np.asarray(x, np.float64)
            y = np.asarray(y, np.float64)
            ix = np.clip(((x - gx0) * gres).astype(int), 0, gw_ - 1)
            iy = np.clip(((y - gy0) * gres).astype(int), 0, gh_ - 1)
            inb = (x >= gx0) & (x < gx0 + gw_ / gres) & (y >= gy0) & (y < gy0 + gh_ / gres)
            return inb & (fdist[iy, ix] <= pad + 0.02)
        self._leafy = leafy

        # trunk
        hs = np.linspace(0, 1, 70)
        cx_ = tx + 0.1 * hs ** 1.5 + 0.03 * np.sin(hs * 4)
        wdt = 0.19 * (1 - 0.45 * hs) + 0.22 * np.exp(-hs / 0.05)
        yy = ty - 0.3 * th * hs
        line(S, np.column_stack([cx_ - wdt, yy]), W(0.022), self.seed + 5, dens=dens, layer=lay, lift=(3, 6), smooth=0)
        line(S, np.column_stack([cx_ + wdt, yy]), W(0.028), self.seed + 6, dens=dens, layer=lay, lift=(3, 6), smooth=0)
        if not pencil:
            for q in np.linspace(0.3, 0.92, 5):
                pp, rd, dd = hand(np.column_stack([cx_ + wdt * q, yy])[4:-3], 0.0065, int(rng.integers(1 << 30)), dens=0.7,
                                  thin_end=0.3, taper=(0.3, 0.3))
                S.add(pp, rd, dd, layer=lay)
            for m in range(7):
                i = int(rng.uniform(0.08, 0.95) * 69)
                q = np.array([[cx_[i] - wdt[i] * 0.5, yy[i]], [cx_[i] + wdt[i] * 0.95, yy[i] + 0.02]])
                pp, rd, dd = hand(q, 0.0055, int(rng.integers(1 << 30)), dens=0.55, thin_end=0.4)
                S.add(pp, rd, dd, layer=lay)
        for (tw, z) in twigs:
            segs.append((pen.resample(tw, 0.03) if len(tw) > 1 else tw, 0.012))
        # limbs, as double lines where thick, single where thin, hidden behind foliage
        for (cpts, w) in segs:
            pts = resample(cpts, 0.02)
            vis = ~leafy(pts[:, 0], pts[:, 1], 0.02)
            dm = np.diff(np.concatenate([[0], vis.astype(np.int8), [0]]))
            for a_, b_ in zip(np.nonzero(dm == 1)[0], np.nonzero(dm == -1)[0] - 1):
                if b_ - a_ < 2:
                    continue
                seg = pts[a_:b_ + 1]
                if w > 0.045:
                    nrm = pen.normals(seg)
                    for sd in (-1, 1):
                        pp, rd, dd = hand(seg + nrm * (sd * w * 0.5), W(0.014 if sd < 0 else 0.018), int(rng.integers(1 << 30)),
                                          dens=dens, thin_end=0.5, taper=(0.02, 0.05))
                        S.add(pp, rd, dd, layer=lay)
                else:
                    pp, rd, dd = hand(seg, W(max(0.008, w * 0.35)), int(rng.integers(1 << 30)), dens=dens, thin_end=0.4,
                                      taper=(0.02, 0.1))
                    S.add(pp, rd, dd, layer=lay)
        _window(S, k, *T(0.05, 0.25), overlap=0.2)

        # the crown: the leaves back to front, each hidden where a nearer leaf covers it; the shaded ones (lower
        # right, and deep in the crown) get a midrib and a stroke or two of hatching
        k = len(S)
        flowers = []
        light = np.array([-0.62, -0.78])
        for li_, (bp, ld, ln, lw, z) in enumerate(leaves):
            s = np.linspace(0, 1, 16)
            hw = lw * 0.5 * np.sin(np.pi * s) ** 0.75 * (1.0 - 0.25 * s)
            nrm = np.array([-ld[1], ld[0]])
            side1 = bp[None] + ld[None] * (s * ln)[:, None] + nrm[None] * hw[:, None]
            side2 = bp[None] + ld[None] * (s * ln)[:, None] - nrm[None] * hw[:, None]
            outline = np.vstack([side1, side2[::-1][1:]])
            front = np.arange(li_ + 1, nL)
            vis = ~np.any(in_leaf(outline[:, 0], outline[:, 1], front, 1.02), axis=1) if len(front) else np.ones(len(outline), bool)
            dm = np.diff(np.concatenate([[0], vis.astype(np.int8), [0]]))
            for a_, b_ in zip(np.nonzero(dm == 1)[0], np.nonzero(dm == -1)[0] - 1):
                if b_ - a_ < 2:
                    continue
                pp, rd, dd = hand(outline[a_:b_ + 1], W(0.0062), int(rng.integers(1 << 30)), dens=dens, thin_end=0.5,
                                  taper=(0.01, 0.02))
                S.add(pp, rd, dd, layer=lay)
            if pencil:
                continue
            q = (bp + ld * 0.5 * ln - cc) / R
            shade = float(np.dot(q, -light)) * 0.6 + 0.35 * (z < 0.45) + rng.normal(0, 0.15)
            if shade > 0.25:
                rib = bp[None] + ld[None] * np.array([[0.08], [0.85]]) * ln
                vr = ~np.any(in_leaf(rib[:, 0], rib[:, 1], front, 1.0), axis=1) if len(front) else np.ones(2, bool)
                if vr.all():
                    pp, rd, dd = hand(rib, 0.0045, int(rng.integers(1 << 30)), dens=0.85, thin_end=0.3)
                    S.add(pp, rd, dd, layer=lay)
                for hq in range(1 + int(shade > 0.6)):
                    f_ = 0.3 + 0.3 * hq
                    h0 = bp + ld * f_ * ln - nrm * 0.3 * lw
                    h1 = h0 + ld * 0.22 * ln + nrm * 0.1 * lw
                    hp = np.array([h0, h1])
                    vh = ~np.any(in_leaf(hp[:, 0], hp[:, 1], front, 1.0), axis=1) if len(front) else np.ones(2, bool)
                    if vh.all():
                        pp, rd, dd = hand(hp, 0.0042, int(rng.integers(1 << 30)), dens=0.8, thin_end=0.3)
                        S.add(pp, rd, dd, layer=lay)
        # the golden flowers: racemes hanging from the sprays, most on the lit side, a few single florets
        for (p_, dirc, z) in clumps:
            q = (p_ - cc) / R
            lit = float(np.dot(q, light))
            if rng.random() > 0.55 + 0.4 * np.clip(lit + 0.35, 0, 1):
                continue
            n_ = int(rng.integers(8, 13))
            sg_ = np.sign(dirc[0] + 1e-6)
            st = p_ + dirc * 0.1
            hang = catmull([st, st + np.array([0.05 * sg_, 0.17]), st + np.array([0.08 * sg_, rng.uniform(0.32, 0.46)])], 6)
            if not pencil:
                pp, rd, dd_ = hand(hang, 0.004, int(rng.integers(1 << 30)), dens=0.8, thin_end=0.4)
                S.add(pp, rd, dd_, layer=lay)
            sh_ = np.concatenate([[0], np.cumsum(np.hypot(*np.diff(hang, axis=0).T))])
            for m in range(n_):
                sm_ = (m + 0.5) / n_ * sh_[-1]
                fx_, fy_ = np.interp(sm_, sh_, hang[:, 0]), np.interp(sm_, sh_, hang[:, 1])
                sd = 1 if m % 2 else -1
                flowers.append((fx_ + sd * 0.04, fy_, 0.068 - 0.03 * m / n_))
        for (bp, ld, ln, lw, z) in leaves[::7]:
            q = (bp - cc) / R
            if float(np.dot(q, light)) > 0.1 and z > 0.5:
                flowers.append((bp[0], bp[1], 0.032))
        _window(S, k, *T(0.25, 0.55), overlap=0.85)


        # the far country: hills, a patchwork of fields, hedgerows, an orchard, stooks, cottages with smoke
        k = len(S)
        xs = np.linspace(bx0 + 0.25, bx1 - 0.25, 300)
        for kk in range(3):
            ys_ = self.hills(xs, kk)
            keep = ~leafy(xs, ys_, 0.06) & ~((np.abs(xs - tx - 0.1) < 0.4) & (ys_ > fork[1] - 0.2))
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
                ok = ~leafy(x, y, 0.08)
                ok &= ~((np.abs(x - tx - 0.1) < 0.45) & (y > fork[1] - 0.2) & (y < ty + 0.1))
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

                    sl = rng.uniform(-0.6, 0.6)

                    def inside(x, y, xa=xa, xb=xb, y_lo=y_lo, y_hi=y_hi, sl=sl):
                        x = np.asarray(x)
                        y = np.asarray(y)
                        xs_ = x + sl * (y - y_lo(x))
                        return (xs_ > xa + 0.05) & (xs_ < xb - 0.05) & (y > y_lo(x) + 0.06) & (y < y_hi(x) - 0.06) & free(x, y)
                    if rng.random() < 0.86:
                        hatch(S, inside, lambda x, y: np.ones_like(x), (xa, self.Yh - 1.0, xb, self.hills(xb, 2) + 0.5),
                              ang, sp, 0.0, 0.0075, int(rng.integers(1 << 30)), dens=0.8, seg=(0.5, 2.0), wob=0.003)
                    else:
                        stipple(S, inside, lambda x, y: np.full_like(x, 0.55), (xa, self.Yh - 1.0, xb, self.hills(xb, 2) + 0.5),
                                150, 0.011, int(rng.integers(1 << 30)))
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
            # stooks of corn in the near field: sheaves leaning together, tied at the neck, in loose rows
            for row in range(3):
                yb_ = self.hills(np.array([bx0 + 10.0]), 2)[0] + 0.5 + 0.55 * row
                sc_ = 0.8 + 0.25 * row
                x_ = bx0 + 8.9 + 0.3 * row + rng.uniform(0, 0.3)
                while x_ < bx1 - 0.9:
                    if free(np.array([x_]), np.array([yb_]))[0] and rng.random() < 0.8:
                        hh = 0.3 * sc_ * rng.uniform(0.9, 1.1)
                        top = np.array([x_ + rng.normal(0, 0.02), yb_ - hh])
                        for m in range(5):
                            fx_ = x_ + (m - 2) * 0.055 * sc_
                            q = np.array([[fx_ + (m - 2) * 0.02 * sc_, yb_ + rng.normal(0, 0.01)], top + [(m - 2) * 0.008, 0]])
                            pp, rd, dd = hand(q, 0.006 + 0.001 * sc_, int(rng.integers(1 << 30)), dens=0.85, thin_end=0.4)
                            S.add(pp, rd, dd, layer=lay)
                        # the ears at the top, the tie, the shadow on the grass
                        for m in range(4):
                            a = -np.pi / 2 + (m - 1.5) * 0.45
                            q = np.array([top, top + 0.07 * sc_ * np.array([math.cos(a), math.sin(a)])])
                            pp, rd, dd = hand(q, 0.006, int(rng.integers(1 << 30)), dens=0.85, thin_end=0.3)
                            S.add(pp, rd, dd, layer=lay)
                        q = np.array([[x_ - 0.07 * sc_, yb_ - 0.62 * hh], [x_ + 0.07 * sc_, yb_ - 0.6 * hh]])
                        pp, rd, dd = hand(q, 0.01, int(rng.integers(1 << 30)), dens=0.9, thin_end=0.6)
                        S.add(pp, rd, dd, layer=lay)
                        q = np.array([[x_ + 0.05, yb_ + 0.01], [x_ + 0.3 * sc_, yb_ + 0.03]])
                        pp, rd, dd = hand(q, 0.012, int(rng.integers(1 << 30)), dens=0.7, thin_end=0.3)
                        S.add(pp, rd, dd, layer=lay)
                    x_ += (0.55 + 0.25 * row) * rng.uniform(0.8, 1.3)
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
            # the smoke of many hearths, rising straight in the still air from beyond the hills (no houses drawn)
            for m, (cx, kk) in enumerate(((bx0 + 9.6, 0), (bx0 + 10.9, 0), (bx0 + 12.4, 1), (bx0 + 13.8, 0), (bx0 + 1.4, 0),
                                          (bx0 + 3.2, 1), (bx0 + 11.7, 0), (bx0 + 14.4, 1))):
                gy_ = float(self.hills(np.array([cx]), kk)[0]) - 0.02
                if not free(np.array([cx]), np.array([gy_]))[0]:
                    continue
                n_ = 8 + m % 3
                sm = np.array([[cx + 0.06 * math.sin(q * 1.2 + m), gy_ - 0.3 * q] for q in range(n_)])
                sm[:, 0] += np.linspace(0, 0.25 + 0.1 * (m % 2), n_) ** 1.5
                pp, rd, dd = hand(catmull(sm, 6), 0.0075, int(rng.integers(1 << 30)), dens=0.72, thin_end=0.1,
                                  taper=(0.05, 0.9))
                S.add(pp, rd, dd, layer=lay)
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
        for (fx_, fy_, fr_) in flowers:
            if pencil:
                a = np.linspace(0, 2 * np.pi, 12)
                q = np.column_stack([fx_ + fr_ * np.cos(a), fy_ + fr_ * np.sin(a)])
                pp, rd, dd = hand(q, 0.008, int(rng.integers(1 << 30)), dens=0.4, thin_end=0.8, taper=(0, 0))
                S.add(pp, rd, dd, layer=lay)
                continue
            # a floret: a small gilt boss with four short petals, a little irregular
            S.add(np.array([[fx_, fy_], [fx_ + 0.002, fy_ + 0.001]]), np.array([fr_ * 0.62, fr_ * 0.62]), np.array([1.0, 1.0]),
                  layer=GILT)
            for p_ in range(4):
                a = p_ * 1.5708 + rng.uniform(0, 0.6)
                q = np.array([[fx_, fy_], [fx_ + fr_ * math.cos(a), fy_ + fr_ * math.sin(a)]])
                S.add(q, np.array([fr_ * 0.4, fr_ * 0.3]), np.array([1.0, 1.0]), layer=GILT)
        _window(S, k, *T(0.9, 1.0), overlap=0.7)
        return S


# ============================================================== THE HAVENS ===

class Havens:
    """A harbour at dusk on a verso: the grey ship slips out west (to the left, off the page's outer edge);
    flame glyphs stand along the coast to kindle one by one in farewell; at the stern a small light. In a
    roundel, the one drawn detail: her bound hand raising the small light."""

    def __init__(self, seed=53):
        self.seed = seed
        self.box = (3.0, 3.0, 17.0, 14.2)
        bx0, by0, bx1, by1 = self.box
        self.Yh = by0 + 6.2                       # horizon
        self.ship = (bx0 + 7.2, self.Yh + 2.3)    # the ship's keel centre at rest (page cm)
        self.roundel = (bx0 + 2.7, by1 + 3.35, 2.5)   # below the plate, in the page's lower left (PAGES-C: 1.85 ->
                                                       # 2.5: at 1.85 her bound hand read as a fist with a ball)
        rng = np.random.default_rng(seed)
        # the coast: headlands on the right (east), the haven among them; fires on the heights
        self.fires = []

    def coast(self, x):
        """The top of the eastern land (hills rising from the harbour mouth)."""
        bx0, by0, bx1, by1 = self.box
        x = np.asarray(x, np.float64)
        u = np.clip((x - (bx0 + 8.6)) / 5.4, 0, 1)
        return self.Yh + 0.9 - 2.4 * u ** 0.7 - 0.35 * np.sin(x * 2.3) * u - 0.25 * np.sin(x * 5.1 + 1) * u

    def shore_x(self, y):
        """The land's western shoreline, running down and to the right from the harbour mouth."""
        bx0, by0, bx1, by1 = self.box
        y = np.asarray(y, np.float64)
        return bx0 + 8.55 + 1.9 * np.clip(y - (self.Yh + 0.9), 0, None) + 0.12 * np.sin(y * 7.0)

    def land(self, x, y):
        x = np.asarray(x, np.float64)
        y = np.asarray(y, np.float64)
        return (y > self.coast(x)) & (x > self.shore_x(y))

    def ship_parts(self, dx=0.0):
        """The swan-ship's shapes (page cm), shifted west by dx: hull (a swan's body, the stern its tail), the
        prow a swan's neck and head, the sail on its yard."""
        x, y = self.ship
        x -= dx
        s = 1.3
        hull = catmull([(x + 1.52 * s, y - 0.72 * s), (x + 1.38 * s, y - 0.36 * s), (x + 0.9 * s, y - 0.02 * s),
                        (x, y + 0.08 * s), (x - 0.95 * s, y + 0.02 * s), (x - 1.38 * s, y - 0.3 * s)], 10)
        gun = catmull([(x + 1.52 * s, y - 0.72 * s), (x + 0.6 * s, y - 0.5 * s), (x - 0.5 * s, y - 0.46 * s),
                       (x - 1.38 * s, y - 0.3 * s)], 10)
        # the neck rises from the bow in an S and bends its head forward, the beak pointing west
        neck_f = catmull([(x - 1.38 * s, y - 0.3 * s), (x - 1.56 * s, y - 0.62 * s), (x - 1.6 * s, y - 1.0 * s),
                          (x - 1.46 * s, y - 1.34 * s), (x - 1.5 * s, y - 1.6 * s), (x - 1.7 * s, y - 1.7 * s),
                          (x - 1.86 * s, y - 1.64 * s)], 10)
        neck_b = catmull([(x - 1.2 * s, y - 0.42 * s), (x - 1.38 * s, y - 0.72 * s), (x - 1.44 * s, y - 1.02 * s),
                          (x - 1.32 * s, y - 1.36 * s), (x - 1.36 * s, y - 1.68 * s), (x - 1.56 * s, y - 1.8 * s),
                          (x - 1.74 * s, y - 1.76 * s), (x - 1.86 * s, y - 1.64 * s)], 10)
        yl, yr, ytop, ybot = x - 0.62 * s, x + 0.78 * s, y - 2.35 * s, y - 0.95 * s
        sail = np.array([[yl, ytop], [yl - 0.16 * s, 0.5 * (ytop + ybot)], [yl - 0.05 * s, ybot], [x + 0.1 * s, ybot + 0.12 * s],
                         [yr - 0.02 * s, ybot + 0.03 * s], [yr - 0.08 * s, 0.5 * (ytop + ybot)], [yr, ytop - 0.02 * s]])
        return dict(x=x, y=y, s=s, hull=hull, gun=gun, neck_f=neck_f, neck_b=neck_b, sail=sail,
                    yard=(yl, yr, ytop, ybot))

    def ship_mask(self, dx, ppc, H, W):
        """Coverage of the ship (hull, neck, sail), to erase the water and sky drawn behind it."""
        import cv2
        sp = self.ship_parts(dx)
        body = np.vstack([sp['hull'], sp['gun'][::-1]])
        neck = np.vstack([sp['neck_f'], sp['neck_b'][::-1]])
        m = np.zeros((H, W), np.uint8)
        for P in (body, neck, sp['sail']):
            cv2.fillPoly(m, [np.round(P * ppc * 16).astype(np.int32)], 255, lineType=cv2.LINE_AA, shift=4)
        m = cv2.GaussianBlur(m.astype(np.float32) / 255.0, (0, 0), 0.8)
        return np.clip(m * 1.05, 0, 1)

    def ship_strokes(self, dx=0.0, mode='ink'):
        """The swan-ship at the plate's own line weights (outline 0.024, detail 0.012, hatching 0.0075), and the
        small figure with her light at the stern, shifted west by dx cm. Returns (Strokes, stern light position)."""
        S = Strokes()
        lay = INK if mode == 'ink' else PENCIL
        rng = np.random.default_rng(self.seed + 7)
        sp = self.ship_parts(dx)
        x, y, s = sp['x'], sp['y'], sp['s']
        line(S, sp['hull'], 0.024, self.seed + 1, layer=lay, lift=(8, 9), smooth=0)
        line(S, sp['gun'], 0.016, self.seed + 2, layer=lay, lift=(8, 9), smooth=0)
        line(S, sp['neck_f'], 0.02, self.seed + 3, layer=lay, lift=(8, 9), smooth=0)
        line(S, sp['neck_b'], 0.016, self.seed + 4, layer=lay, lift=(8, 9), smooth=0)
        hx, hy = x - 1.62 * s, y - 1.72 * s
        S.add(np.array([[hx, hy], [hx + 0.001, hy]]), np.array([0.012, 0.012]), np.array([0.9, 0.9]), layer=lay)  # the eye
        if mode == 'ink':
            # the swan's wing carved along the side: long feathers sweeping back from the bow
            for q in range(6):
                f0 = np.array([x - (1.05 - 0.08 * q) * s, y - (0.42 - 0.035 * q) * s])
                f1 = np.array([x + (0.25 + 0.2 * q) * s, y - (0.32 - 0.05 * q) * s])
                fm = 0.5 * (f0 + f1) + np.array([0.0, 0.06 * s])
                pp, rr, dd = hand(catmull([f0, fm, f1], 8), 0.012, int(rng.integers(1 << 30)), thin_end=0.25,
                                  taper=(0.05, 0.3))
                S.add(pp, rr, dd, layer=lay)
            body = np.vstack([sp['hull'], sp['gun'][::-1]])
            hatch(S, _poly_inside(body), lambda px, py: np.clip((py - (y - 0.6 * s)) / (0.7 * s), 0, 1) + 0.25,
                  (x - 1.5 * s, y - 0.8 * s, x + 1.6 * s, y + 0.15 * s), 8, 0.035, 0.45, 0.0075, self.seed + 5,
                  dens=0.85, seg=(0.4, 1.2))
            neck = np.vstack([sp['neck_f'], sp['neck_b'][::-1]])
            hatch(S, _poly_inside(neck), lambda px, py: (px - (x - 1.62 * s)) / (0.4 * s) + 0.5,
                  (x - 1.9 * s, y - 1.85 * s, x - 1.15 * s, y - 0.3 * s), 70, 0.03, 0.55, 0.0075, self.seed + 6,
                  dens=0.8, seg=(0.2, 0.6))
        mast = np.array([[x + 0.05 * s, y - 0.45 * s], [x + 0.08 * s, y - 2.6 * s]])
        line(S, mast, 0.02, self.seed + 8, layer=lay)
        yl, yr, ytop, ybot = sp['yard']
        line(S, np.array([[yl - 0.08 * s, ytop + 0.02 * s], [yr + 0.08 * s, ytop - 0.02 * s]]), 0.016, self.seed + 9,
             layer=lay, lift=(8, 9), smooth=0)
        lead = catmull([(yl, ytop), (yl - 0.16 * s, 0.5 * (ytop + ybot)), (yl - 0.05 * s, ybot)], 10)
        trail = catmull([(yr, ytop - 0.02 * s), (yr - 0.08 * s, 0.5 * (ytop + ybot)), (yr - 0.02 * s, ybot + 0.03 * s)], 10)
        foot = catmull([(yl - 0.05 * s, ybot), (x + 0.1 * s, ybot + 0.12 * s), (yr - 0.02 * s, ybot + 0.03 * s)], 10)
        for q_, P_ in enumerate((lead, trail, foot)):
            line(S, P_, 0.014, self.seed + 10 + q_, layer=lay, lift=(8, 9), smooth=0)
        if mode == 'ink':
            for q in np.linspace(0.12, 0.88, 6):
                xq = yl + (yr - yl) * q
                seam = catmull([(xq, ytop), (xq - 0.1 * s * math.sin(q * math.pi), 0.5 * (ytop + ybot)),
                                (xq, ybot + 0.12 * s * math.sin(q * math.pi))], 8)
                pp, rr, dd = hand(seam, 0.0075, int(rng.integers(1 << 30)), thin_end=0.4, dens=0.8)
                S.add(pp, rr, dd, layer=lay)
            sail_poly = np.vstack([lead, foot[1:], trail[::-1][1:]])
            hatch(S, _poly_inside(sail_poly), lambda px, py: (px - yl) / (yr - yl), (yl - 0.2 * s, ytop, yr, ybot + 0.15 * s),
                  75, 0.04, 0.55, 0.0075, self.seed + 13, dens=0.75, seg=(0.2, 0.7))
        # the small figure at the stern (a silhouette), one arm raised with the light
        fx, fy = x + 1.2 * s, y - 0.66 * s
        body = np.array([[fx, fy], [fx + 0.02, fy - 0.32]])
        S.add(resample(body, 0.02), np.linspace(0.05, 0.035, len(resample(body, 0.02))), 0.95, layer=lay)
        S.add(np.array([[fx + 0.02, fy - 0.38], [fx + 0.021, fy - 0.38]]), np.array([0.035, 0.035]), np.array([0.95, 0.95]),
              layer=lay)
        arm = np.array([[fx + 0.02, fy - 0.28], [fx + 0.09, fy - 0.45], [fx + 0.1, fy - 0.56]])
        pp, rr, dd = hand(arm, 0.012, self.seed + 14, thin_end=0.6)
        S.add(pp, rr, dd, layer=lay)
        return S, (fx + 0.1, fy - 0.62)

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
        _window(S, k, *T(0.0, 0.05))
        # the coast: headlands on the right, cliffs hatched, a quay and a small tower at the haven
        k = len(S)
        xs = np.linspace(bx0 + 8.3, bx1 - 0.2, 200)
        cy = self.coast(xs)
        line(S, np.column_stack([xs, cy]), W(0.024), self.seed + 10, dens=dens, layer=lay, lift=(2, 4), smooth=0)
        # the shoreline, and the land shaded with strokes down its slopes, darker toward the water
        ys_ = np.linspace(self.Yh + 0.9, by1 - 0.2, 80)
        sx_ = self.shore_x(ys_)
        m_ = sx_ < bx1 - 0.2
        line(S, np.column_stack([sx_[m_], ys_[m_]]), W(0.02), self.seed + 15, dens=dens, layer=lay, lift=(2, 4), smooth=0)
        if not pencil:
            hatch(S, lambda x, y: self.land(x, y), lambda x, y: 0.35 + 0.6 * np.clip((np.asarray(y) - self.coast(x)) / 1.5, 0, 1),
                  (bx0 + 8.5, self.Yh - 2.0, bx1 - 0.2, by1 - 0.15), 84, 0.05, 0.3, 0.0075, self.seed + 16, dens=0.85,
                  seg=(0.2, 0.7), jitter=0.5, gap=0.12)
            hatch(S, lambda x, y: self.land(x, y), lambda x, y: 0.35 + 0.6 * np.clip((np.asarray(y) - self.coast(x)) / 1.5, 0, 1),
                  (bx0 + 8.5, self.Yh - 2.0, bx1 - 0.2, by1 - 0.15), 30, 0.06, 0.7, 0.0075, self.seed + 17, dens=0.8,
                  seg=(0.2, 0.6), jitter=0.5, gap=0.12)
        # the far western headland, low and faint
        xw = np.linspace(bx0 + 0.25, bx0 + 3.6, 60)
        yw = self.Yh - 0.25 * np.exp(-((xw - (bx0 + 1.6)) / 1.1) ** 2) - 0.05 * np.sin(xw * 4)
        line(S, np.column_stack([xw, yw]), W(0.012), self.seed + 11, dens=dens * 0.7, layer=lay, lift=(2, 4), smooth=0)
        # the quay and the tower at the haven's inner shore
        qx = bx0 + 9.2
        qy = float(self.coast(np.array([qx]))[0])
        quay = np.array([[qx - 1.4, qy + 0.35], [qx - 1.4, qy + 0.18], [qx + 0.2, qy + 0.18]])
        line(S, quay, W(0.016), self.seed + 12, dens=dens, layer=lay, lift=(4, 5), smooth=0)
        # a slender round watch-tower (MAP-L: never an obelisk): a little batter, a corbelled parapet, merlons
        tw = np.array([[qx + 0.35, qy + 0.05], [qx + 0.39, qy - 1.12], [qx + 0.35, qy - 1.18], [qx + 0.35, qy - 1.34],
                       [qx + 0.43, qy - 1.34], [qx + 0.43, qy - 1.27], [qx + 0.51, qy - 1.27], [qx + 0.51, qy - 1.34],
                       [qx + 0.6, qy - 1.34], [qx + 0.6, qy - 1.27], [qx + 0.68, qy - 1.27], [qx + 0.68, qy - 1.34],
                       [qx + 0.76, qy - 1.34], [qx + 0.76, qy - 1.18], [qx + 0.72, qy - 1.12], [qx + 0.75, qy + 0.05]])
        line(S, tw, W(0.016), self.seed + 13, dens=dens, layer=lay, lift=(5, 6), smooth=0)
        line(S, np.array([[qx + 0.36, qy - 1.18], [qx + 0.75, qy - 1.18]]), W(0.011), self.seed + 15, dens=dens,
             layer=lay, lift=(5, 6), smooth=0)
        line(S, np.array([[qx + 0.39, qy - 1.12], [qx + 0.72, qy - 1.12]]), W(0.009), self.seed + 16, dens=dens,
             layer=lay, lift=(5, 6), smooth=0)
        line(S, np.array([[qx + 0.52, qy - 0.92], [qx + 0.52, qy - 0.8]]), W(0.02), self.seed + 17, dens=dens,
             layer=lay, lift=(5, 6), smooth=0)
        if not pencil:
            hatch(S, _poly_inside(tw), lambda x, y: (x - (qx + 0.35)) / 0.4, (qx + 0.3, qy - 1.4, qx + 0.8, qy + 0.1), 90,
                  0.035, 0.45, 0.007, self.seed + 14, dens=0.85, seg=(0.3, 1.2))
        _window(S, k, *T(0.05, 0.3), overlap=0.6)
        # the flame glyphs on the heights (ink now; they kindle as fire in the shot)
        k = len(S)
        self.fires = []
        for m, fxp in enumerate(np.linspace(bx0 + 10.3, bx1 - 1.0, 3)):
            fy_ = float(self.coast(np.array([fxp]))[0]) - 0.02
            hh = 0.34 + 0.1 * rng.random()
            fl = [(fxp - 0.1, fy_), (fxp - 0.12, fy_ - 0.35 * hh), (fxp - 0.02, fy_ - 0.7 * hh), (fxp + 0.03, fy_ - hh),
                  (fxp + 0.1, fy_ - 0.55 * hh), (fxp + 0.1, fy_)]
            line(S, np.array(fl), W(0.012), self.seed + 20 + m, dens=dens, layer=lay, lift=(4, 5), smooth=8)
            self.fires.append((fxp, fy_, hh))
        _window(S, k, *T(0.3, 0.4), overlap=0.3)
        # the sea: engraved water, the lines closer toward the horizon, broken where light falls on it
        k = len(S)
        if not pencil:
            yy = self.Yh + 0.06
            li = 0
            while yy < by1 - 0.1:
                gap = 0.05 + 0.1 * ((yy - self.Yh) / (by1 - self.Yh)) ** 1.3
                xs = np.arange(bx0 + 0.25, bx1 - 0.25, 0.02)
                ys = yy + 0.012 * np.sin(xs * (9 - 3 * (yy - self.Yh) / (by1 - self.Yh)) + li * 1.7)
                keep = ~self.land(xs, ys - 0.04) & ~self.land(xs - 0.06, ys)
                dm = np.diff(np.concatenate([[0], keep.astype(np.int8), [0]]))
                for a_, b_ in zip(np.nonzero(dm == 1)[0], np.nonzero(dm == -1)[0] - 1):
                    if b_ - a_ < 3:
                        continue
                    pts = np.column_stack([xs[a_:b_ + 1], ys[a_:b_ + 1]])
                    line(S, pts, 0.008 + 0.004 * (yy - self.Yh) / (by1 - self.Yh), int(rng.integers(1 << 30)), dens=0.85,
                         layer=lay, lift=(0.4, 1.6), smooth=0, taper=(0.03, 0.05), thin_end=0.3)
                yy += gap
                li += 1
        # the dusk sky: lines darker toward the top, the afterglow low in the west
        if not pencil:
            for li, yy in enumerate(np.arange(by0 + 0.22, self.Yh - 0.1, 0.062)):
                tone = (1.0 - (yy - by0) / (self.Yh - by0)) ** 0.9
                xs = np.arange(bx0 + 0.25, bx1 - 0.25, 0.02)
                glow = np.exp(-np.hypot((xs - (bx0 + 2.0)) / 5.0, (yy - self.Yh) / 2.2))
                t_ = tone * (1.0 - 0.8 * glow)
                lvl = 0 if li % 4 == 0 else (1 if li % 2 == 0 else 2)
                keep = (t_ > (0.02, 0.25, 0.5)[lvl]) & (yy < self.coast(xs) - 0.1)
                dm = np.diff(np.concatenate([[0], keep.astype(np.int8), [0]]))
                for a_, b_ in zip(np.nonzero(dm == 1)[0], np.nonzero(dm == -1)[0] - 1):
                    if b_ - a_ < 4:
                        continue
                    pts = np.column_stack([xs[a_:b_ + 1], np.full(b_ - a_ + 1, yy)])
                    s0 = len(S)
                    line(S, pts, 0.008, int(rng.integers(1 << 30)), dens=0.88, layer=lay, lift=(3, 6), smooth=0,
                         slow=(5.0, 0.002), fast=(0.4, 0.0008), taper=(0.02, 0.04), thin_end=0.5)
                    for q in range(s0, len(S)):
                        S.R[q] = S.R[q] * (1.0 + 1.2 * tone ** 2)
            # the evening star
            ex, ey = bx0 + 3.1, by0 + 1.6
            for a in (0.0, np.pi / 2, np.pi / 4, -np.pi / 4):
                r_ = 0.14 if a in (0.0, np.pi / 2) else 0.07
                q = np.array([[ex - r_ * math.cos(a), ey - r_ * math.sin(a)], [ex + r_ * math.cos(a), ey + r_ * math.sin(a)]])
                pp, rd, dd = hand(q, 0.01, int(rng.integers(1 << 30)), dens=0.9, thin_end=0.2)
                S.add(pp, rd, dd, layer=lay)
        _window(S, k, *T(0.4, 0.75), overlap=0.9)
        # the roundel: her bound hand raising the small light
        k = len(S)
        self._roundel(S, rng, lay, pencil)
        _window(S, k, *T(0.75, 1.0), overlap=0.7)
        return S

    def _roundel(self, S, rng, lay, pencil):
        """THE one drawn detail (REV 1; MAP-L redraw): her bound hand, close, raising the small light. The hand is
        wrapped in linen to the knuckles (never bare): bands wound across the back of the hand and the wrist, a loose
        end trailing in the wind off the sea; it grips the stem of a small clay lamp held up against the dusk, whose
        flame is the colour (it kindles in the shot); the dark cloak sleeve falls away below. Back-of-hand view."""
        cx, cy, R = self.roundel
        a = np.linspace(0, 2 * np.pi, 200)
        for rr, w in ((R, 0.03), (R - 0.12, 0.012)):
            q = np.column_stack([cx + rr * np.cos(a), cy + rr * np.sin(a)])
            line(S, q, w, int(rng.integers(1 << 30)), layer=lay, lift=(4, 7), smooth=0)
        Ri = R - 0.16
        U = lambda P: np.column_stack([cx + np.asarray(P, np.float64)[:, 0] * R, cy - np.asarray(P, np.float64)[:, 1] * R])
        inside_r = lambda x, y: np.hypot(np.asarray(x) - cx, np.asarray(y) - cy) < Ri
        # the shapes (roundel units, y up)
        fist = U(catmull([(0.02, -0.04), (-0.05, 0.08), (-0.06, 0.2), (-0.02, 0.3), (0.05, 0.36), (0.1, 0.35), (0.14, 0.38),
                          (0.19, 0.365), (0.23, 0.385), (0.28, 0.36), (0.33, 0.33), (0.37, 0.22), (0.36, 0.08),
                          (0.32, -0.04)], 6))
        wrist = U(catmull([(0.03, -0.02), (0.04, -0.16), (0.06, -0.3), (0.32, -0.28), (0.32, -0.14), (0.32, -0.02)], 6))
        thumb = U(catmull([(-0.045, 0.13), (-0.015, 0.2), (0.05, 0.255), (0.125, 0.285), (0.135, 0.255), (0.06, 0.215), (0.005, 0.13)], 6))
        sleeve = U(catmull([(0.05, -0.29), (-0.06, -0.56), (-0.2, -1.05), (0.75, -1.05), (0.56, -0.6), (0.34, -0.27),
                            (0.2, -0.31)], 6))
        fist_in = _poly_inside(np.vstack([fist, fist[:1]]))
        wrist_in = _poly_inside(np.vstack([wrist, wrist[:1]]))
        thumb_in = _poly_inside(np.vstack([thumb, thumb[:1]]))
        sleeve_in = _poly_inside(np.vstack([sleeve, sleeve[:1]]))
        hand_in = lambda x, y: fist_in(x, y) | wrist_in(x, y) | thumb_in(x, y)
        # the lamp: a stem through the fist, a small round-bellied clay lamp on it, its spout to the right
        stem_l = U([(0.14, -0.12), (0.14, 0.52)])
        stem_r = U([(0.185, -0.12), (0.185, 0.52)])
        lamp = U(catmull([(0.0, 0.56), (0.02, 0.62), (0.12, 0.655), (0.24, 0.64), (0.33, 0.62), (0.4, 0.635), (0.41, 0.61),
                          (0.3, 0.56), (0.2, 0.53), (0.06, 0.53), (0.0, 0.56)], 6))
        lamp_in = _poly_inside(np.vstack([lamp, lamp[:1]]))
        stem_in = lambda x, y: (np.abs(np.asarray(x) - (cx + 0.1625 * R)) < 0.0225 * R) & \
            (np.asarray(y) < cy - (-0.12) * R) & (np.asarray(y) > cy - 0.54 * R)
        solid = lambda x, y: hand_in(x, y) | sleeve_in(x, y) | lamp_in(x, y) | stem_in(x, y)

        def draw(P, w, clip=None):
            keep = inside_r(P[:, 0], P[:, 1])
            if clip is not None:
                keep &= ~clip(P[:, 0], P[:, 1])
            dm = np.diff(np.concatenate([[0], keep.astype(np.int8), [0]]))
            for a_, b_ in zip(np.nonzero(dm == 1)[0], np.nonzero(dm == -1)[0] - 1):
                if b_ - a_ > 1:
                    line(S, P[a_:b_ + 1], w, int(rng.integers(1 << 30)), layer=lay, lift=(6, 9), smooth=0)
        # outlines: the sleeve behind the wrist, the stem behind the fist, the thumb over the fist
        draw(sleeve, 0.016, clip=hand_in)
        draw(fist, 0.015, clip=thumb_in)
        draw(wrist, 0.014, clip=lambda x, y: fist_in(x, y) | sleeve_in(x, y))
        draw(thumb, 0.014)
        for st in (stem_l, stem_r):
            draw(pen.resample(st, 0.02), 0.011, clip=lambda x, y: hand_in(x, y) | lamp_in(x, y))
        draw(lamp, 0.013)
        # the knuckles: soft marks where the wrapped fingers fold
        for kx in (0.08, 0.17, 0.26):
            draw(U(catmull([(kx - 0.02, 0.33), (kx, 0.31), (kx + 0.03, 0.325)], 4)), 0.008)
        if not pencil:
            box = (cx - R, cy - R, cx + R, cy + R)
            dark_in = lambda x, y: sleeve_in(x, y) & inside_r(x, y) & ~hand_in(x, y)
            hatch(S, dark_in, lambda x, y: np.ones_like(x), box, 70, 0.028, 0.0, 0.011, int(rng.integers(1 << 30)),
                  dens=0.95, seg=(0.3, 1.0), wob=0.003)
            hatch(S, dark_in, lambda x, y: np.ones_like(x), box, 5, 0.034, 0.0, 0.01, int(rng.integers(1 << 30)),
                  dens=0.9, seg=(0.3, 1.0), wob=0.003)
            # the linen: bands wound on the slant across the back of the hand and the wrist, each band's lower edge
            # drawn, a little shading under each edge on the shadow (right) side
            for m, off in enumerate(np.arange(-0.46, 0.5, 0.085)):
                xs_ = np.linspace(-0.1, 0.42, 60)
                band = U(np.column_stack([xs_, off + 0.42 * xs_ + 0.012 * np.sin(xs_ * 14 + m)]))
                inb = hand_in(band[:, 0], band[:, 1]) & ~thumb_in(band[:, 0], band[:, 1])
                dm = np.diff(np.concatenate([[0], inb.astype(np.int8), [0]]))
                for a_, b_ in zip(np.nonzero(dm == 1)[0], np.nonzero(dm == -1)[0] - 1):
                    if b_ - a_ > 3:
                        pp, rd, dd = hand(band[a_:b_ + 1], 0.0065, int(rng.integers(1 << 30)), dens=0.9, thin_end=0.5,
                                          taper=(0.02, 0.05))
                        S.add(pp, rd, dd, layer=lay)
                        seg = band[a_:b_ + 1]
                        for j in range(len(seg) // 2, len(seg), 3):
                            p0 = seg[j]
                            p1 = p0 + np.array([0.012, 0.05])
                            if hand_in(np.array([p1[0]]), np.array([p1[1]]))[0]:
                                pp, rd, dd = hand(np.array([p0, p1]), 0.0045, int(rng.integers(1 << 30)), dens=0.8,
                                                  thin_end=0.3)
                                S.add(pp, rd, dd, layer=lay)
            # the thumb's wrap, crossing the other way
            for off in (0.06, 0.14):
                xs_ = np.linspace(-0.08, 0.16, 20)
                band = U(np.column_stack([xs_, off + 0.9 * (xs_ + 0.05) - 0.06]))
                inb = thumb_in(band[:, 0], band[:, 1])
                if inb.sum() > 3:
                    pp, rd, dd = hand(band[inb], 0.006, int(rng.integers(1 << 30)), dens=0.9, thin_end=0.5)
                    S.add(pp, rd, dd, layer=lay)
            # the loose end of the linen, trailing from the wrist in the wind off the sea
            tail = U(catmull([(0.31, -0.2), (0.42, -0.24), (0.52, -0.2), (0.6, -0.26), (0.68, -0.22)], 8))
            tail2 = U(catmull([(0.31, -0.25), (0.42, -0.3), (0.51, -0.26), (0.6, -0.32), (0.66, -0.29)], 8))
            draw(tail, 0.01, clip=None)
            draw(tail2, 0.009, clip=None)
            # the dusk behind: sky lines, darker upward; the sea's line low down, and its swell below it
            sky_in = lambda x, y: inside_r(x, y) & ~solid(x, y)
            for q, yy in enumerate(np.arange(cy - Ri, cy + 0.42 * R, 0.07)):
                tone = (1.0 - (yy - (cy - Ri)) / (1.42 * R)) ** 1.2
                if q % 2 and tone < 0.5:
                    continue
                xs = np.arange(cx - R, cx + R, 0.02)
                lf = (cx + 0.39 * R, cy - 0.68 * R)
                keep = sky_in(xs, np.full_like(xs, yy)) & (np.hypot(xs - lf[0], yy - lf[1]) > 0.28 * R)
                dm = np.diff(np.concatenate([[0], keep.astype(np.int8), [0]]))
                for a_, b_ in zip(np.nonzero(dm == 1)[0], np.nonzero(dm == -1)[0] - 1):
                    if b_ - a_ > 3:
                        pp, rd, dd = hand(np.column_stack([xs[a_:b_ + 1], np.full(b_ - a_ + 1, yy)]), 0.008 * (1 + tone),
                                          int(rng.integers(1 << 30)), dens=0.85, thin_end=0.5, taper=(0.02, 0.02))
                        S.add(pp, rd, dd, layer=lay)
            for q, yy in enumerate(np.arange(cy + 0.46 * R, cy + Ri, 0.1)):
                xs = np.arange(cx - R, cx + R, 0.02)
                wv = yy + 0.012 * np.sin(xs * 9.0 + q * 1.7)
                keep = sky_in(xs, wv)
                dm = np.diff(np.concatenate([[0], keep.astype(np.int8), [0]]))
                for a_, b_ in zip(np.nonzero(dm == 1)[0], np.nonzero(dm == -1)[0] - 1):
                    if b_ - a_ > 3:
                        pp, rd, dd = hand(np.column_stack([xs[a_:b_ + 1], wv[a_:b_ + 1]]), 0.007, int(rng.integers(1 << 30)),
                                          dens=0.8, thin_end=0.5, taper=(0.02, 0.02))
                        S.add(pp, rd, dd, layer=lay)
        # the lamp's flame, drawn lightly (it burns in colour in the shot)
        fl = U(catmull([(0.39, 0.64), (0.37, 0.7), (0.395, 0.79), (0.42, 0.7), (0.405, 0.64)], 6))
        line(S, fl, 0.008, int(rng.integers(1 << 30)), layer=lay, lift=(6, 7), smooth=0)
        self.lamp_flame = (cx + 0.395 * R, cy - 0.68 * R)
