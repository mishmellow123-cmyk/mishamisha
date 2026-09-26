"""Seven procedural towers of embers (one per kingdom), each architecturally distinct.

Every tower is built in local coordinates with its TOP at y = HMAX; the body
extends downward to y = 0 so a tower can keep rising out of the ground
("surge") without ever running out of building. Returns dict with
    p (N,3)   local points
    n (N,3)   outward normals (lines use the radial direction)
    kind (N,) 0 surface, 1 edge/line (brighter), 2 window light
"""
import numpy as np

HMAX = 66.0
DENS_L = 1.5     # line density multiplier
DENS_S = 2.4     # surface density multiplier


def _radial(p):
    n = np.zeros_like(p)
    n[:, 0], n[:, 2] = p[:, 0], p[:, 2]
    l = np.linalg.norm(n, axis=1, keepdims=True)
    return n / np.maximum(l, 1e-6)


class B:
    def __init__(self, seed):
        self.r = np.random.default_rng(seed)
        self.P, self.N, self.K = [], [], []

    def add(self, p, kind, n=None):
        p = np.asarray(p, np.float64).reshape(-1, 3)
        if n is None:
            n = _radial(p)
        self.P.append(p)
        self.N.append(n)
        self.K.append(np.full(len(p), kind, np.int8))

    def line(self, a, b, dens=40.0, kind=1, jit=0.02):
        dens *= DENS_L
        a, b = np.asarray(a, float), np.asarray(b, float)
        L = np.linalg.norm(b - a)
        n = max(2, int(L * dens))
        t = self.r.random(n)
        p = a + (b - a) * t[:, None] + self.r.normal(0, jit, (n, 3))
        self.add(p, kind)

    def hloop(self, pts_xz, y, dens=40.0, kind=1, jit=0.02):
        """closed horizontal polyline through xz points at height y"""
        pts = np.asarray(pts_xz, float)
        for i in range(len(pts)):
            a = np.array([pts[i][0], y, pts[i][1]])
            b = np.array([pts[(i + 1) % len(pts)][0], y, pts[(i + 1) % len(pts)][1]])
            self.line(a, b, dens, kind, jit)

    def circle(self, r, y, dens=40.0, kind=1, jit=0.02):
        n = max(8, int(2 * np.pi * r * dens))
        a = self.r.random(n) * 2 * np.pi
        p = np.stack([r * np.cos(a), np.full(n, y), r * np.sin(a)], 1) + self.r.normal(0, jit, (n, 3))
        self.add(p, kind)

    def revolve(self, rfn, y0, y1, dens=6.0, kind=0):
        """random points on surface of revolution r(y)"""
        dens *= DENS_S
        ys = np.linspace(y0, y1, 64)
        area = np.trapezoid(2 * np.pi * np.maximum(rfn(ys), 0.05), ys)
        n = int(area * dens)
        y = self.r.uniform(y0, y1, n)
        rr = rfn(y)
        keep = self.r.random(n) < rr / max(rr.max(), 1e-6)
        y, rr = y[keep], rr[keep]
        a = self.r.random(len(y)) * 2 * np.pi
        self.add(np.stack([rr * np.cos(a), y, rr * np.sin(a)], 1), kind)

    def square_faces(self, hwfn, y0, y1, dens=6.0, kind=0, twist=None):
        dens *= DENS_S
        ys = np.linspace(y0, y1, 64)
        area = np.trapezoid(8 * hwfn(ys), ys)
        n = int(area * dens)
        y = self.r.uniform(y0, y1, n)
        hw = hwfn(y)
        side = self.r.integers(0, 4, n)
        s = self.r.uniform(-1, 1, n) * hw
        x = np.where(side == 0, hw, np.where(side == 1, -hw, s))
        z = np.where(side == 2, hw, np.where(side == 3, -hw, s))
        nrm = np.zeros((n, 3))
        nrm[side == 0, 0] = 1
        nrm[side == 1, 0] = -1
        nrm[side == 2, 2] = 1
        nrm[side == 3, 2] = -1
        p = np.stack([x, y, z], 1)
        if twist is not None:
            a = twist(y)
            c, sn = np.cos(a), np.sin(a)
            p = np.stack([c * x - sn * z, y, sn * x + c * z], 1)
            nrm = np.stack([c * nrm[:, 0] - sn * nrm[:, 2], nrm[:, 1], sn * nrm[:, 0] + c * nrm[:, 2]], 1)
        self.add(p, kind, nrm)

    def windows_rev(self, rfn, y0, y1, dy, nphi, frac=0.4, out=0.03):
        frac = min(1.0, frac * 1.5)
        ys = np.arange(y0, y1, dy)
        for y in ys:
            r = rfn(np.array([y]))[0] + out
            for k in range(nphi):
                if self.r.random() < frac:
                    a = 2 * np.pi * (k + 0.5) / nphi
                    self.add(np.array([[r * np.cos(a), y, r * np.sin(a)]]), 2)

    def windows_square(self, hwfn, y0, y1, dy, nper, frac=0.4, twist=None):
        frac = min(1.0, frac * 1.5)
        ys = np.arange(y0, y1, dy)
        for y in ys:
            hw = hwfn(np.array([y]))[0] + 0.03
            for side in range(4):
                for k in range(nper):
                    if self.r.random() < frac:
                        s = (-1 + 2 * (k + 0.5) / nper) * hw * 0.85
                        x, z = [(hw, s), (-hw, s), (s, hw), (s, -hw)][side]
                        if twist is not None:
                            a = twist(np.array([y]))[0]
                            x, z = np.cos(a) * x - np.sin(a) * z, np.sin(a) * x + np.cos(a) * z
                        self.add(np.array([[x, y, z]]), 2)

    def done(self):
        return dict(p=np.concatenate(self.P), n=np.concatenate(self.N), kind=np.concatenate(self.K))


def needle_spire(seed=0):
    b = B(seed)
    top = HMAX
    body = top - 12.0

    def rf(y):
        return 1.7 - 0.5 * np.clip(y / body, 0, 1)
    for k in range(8):
        a = 2 * np.pi * k / 8
        r0, r1 = rf(np.array([0.0]))[0], rf(np.array([body]))[0]
        b.line([r0 * np.cos(a), 0, r0 * np.sin(a)], [r1 * np.cos(a), body, r1 * np.sin(a)], 30)
    for y in np.arange(1.0, body, 3.0):
        b.circle(rf(np.array([y]))[0], y, 26)
    b.revolve(rf, 0, body, 3.0)
    # crown: cone then needle
    b.revolve(lambda y: np.interp(y, [body, body + 5.0], [1.2, 0.28]), body, body + 5.0, 12.0, 1)
    b.line([0, body + 4.5, 0], [0, top, 0], 60, 1, 0.01)
    for k in range(4):
        a = 2 * np.pi * k / 4 + 0.4
        b.line([1.2 * np.cos(a), body, 1.2 * np.sin(a)], [0.1 * np.cos(a), body + 7.0, 0.1 * np.sin(a)], 40)
    b.windows_rev(rf, 1.5, body - 1, 1.4, 16, 0.35)
    return b.done()


def ziggurat(seed=1):
    b = B(seed)
    top = HMAX
    th = 2.2
    shrine = 1.8
    y = top - shrine
    b.square_faces(lambda yy: np.full_like(yy, 0.8), y, top, 14.0)
    b.hloop([(0.8, 0.8), (-0.8, 0.8), (-0.8, -0.8), (0.8, -0.8)], top, 40)
    k = 0
    while y > 0:
        hw = 1.3 + 0.75 * min(k, 7)
        y0 = max(y - th, 0.0)
        sq = [(hw, hw), (-hw, hw), (-hw, -hw), (hw, -hw)]
        b.hloop(sq, y, 34)
        b.hloop(sq, y0 + 0.05, 20)
        for (x, z) in sq:
            b.line([x, y0, z], [x, y, z], 30)
        b.square_faces(lambda yy, hw=hw: np.full_like(yy, hw), y0, y, 3.5)
        # step top (the terrace ring between this and the tier above)
        if k > 0:
            hi = 1.3 + 0.75 * min(k - 1, 7)
            if hw > hi + 0.05:
                n = int((4 * hw * hw - 4 * hi * hi) * 3)
                x = b.r.uniform(-hw, hw, n)
                z = b.r.uniform(-hw, hw, n)
                m = (np.abs(x) > hi) | (np.abs(z) > hi)
                b.add(np.stack([x[m], np.full(m.sum(), y), z[m]], 1), 0,
                      np.tile([0, 1.0, 0], (m.sum(), 1)))
        # a central stair
        b.line([hw, y0, -0.3], [hw, y, -0.3], 10, 1)
        b.windows_square(lambda yy, hw=hw: np.full_like(yy, hw), y0 + 0.9, y - 0.3, 1.1, max(2, int(hw * 1.3)), 0.3)
        y = y0
        k += 1
    return b.done()


def lattice_mast(seed=2):
    b = B(seed)
    top = HMAX
    mast = top - 5.0

    def hw(y):
        return 0.45 + 2.3 * (1 - np.clip(y / mast, 0, 1)) ** 1.3
    legs = [0.0, 2 * np.pi / 3, 4 * np.pi / 3]
    ys = np.arange(0.0, mast + 0.01, 2.2)

    def leg(a, y):
        r = hw(np.array([y]))[0]
        return np.array([r * np.cos(a), y, r * np.sin(a)])
    for a in legs:
        b.line(leg(a, 0), leg(a, mast), 36, 1, 0.01)
    for i in range(len(ys) - 1):
        y0, y1 = ys[i], ys[i + 1]
        for j in range(3):
            a0, a1 = legs[j], legs[(j + 1) % 3]
            b.line(leg(a0, y0), leg(a1, y1), 22, 1, 0.01)
            b.line(leg(a1, y0), leg(a0, y1), 22, 1, 0.01)
            b.line(leg(a0, y1), leg(a1, y1), 16, 1, 0.01)
    b.line([0, mast, 0], [0, top, 0], 50, 1, 0.005)
    for y in (mast - 3.0, mast + 1.5):
        b.circle(0.9, y, 30)
    for y in np.arange(4.0, mast, 8.0):
        b.add(leg(legs[0], y)[None, :], 2)
    b.add(np.array([[0, top, 0]]), 2)
    return b.done()


def ringed_cylinder(seed=3):
    b = B(seed)
    top = HMAX
    body = top - 3.0
    R = 2.0

    def rf(y):
        return np.full_like(y, R)
    b.revolve(rf, 0, body, 3.0)
    for y in np.arange(1.3, body, 2.6):
        b.circle(R + 0.95, y, 26)
        b.circle(R + 0.95, y - 0.18, 12)
        n = 60
        a = b.r.random(n) * 2 * np.pi
        rr = b.r.uniform(R, R + 0.95, n)
        b.add(np.stack([rr * np.cos(a), np.full(n, y), rr * np.sin(a)], 1), 0)
    b.revolve(lambda y: np.sqrt(np.maximum(R * R - (y - body) ** 2, 0.0)) + 0.01, body, body + R * 0.999, 10.0, 1)
    b.line([0, body + R, 0], [0, top + 1.5, 0], 50, 1, 0.005)
    b.windows_rev(rf, 0.6, body - 0.5, 0.65, 18, 0.33)
    return b.done()


def twisted(seed=4):
    b = B(seed)
    top = HMAX
    body = top - 4.0

    def hw(y):
        return 1.9 - 0.7 * np.clip(y / body, 0, 1)

    def tw(y):
        return 1.3 * np.pi * np.clip(y / body, 0, 1)
    for c in range(4):
        ys = np.linspace(0, body, 600)
        a0 = [np.pi / 4, 3 * np.pi / 4, 5 * np.pi / 4, 7 * np.pi / 4][c]
        r = hw(ys) * np.sqrt(2)
        a = a0 + tw(ys)
        pts = np.stack([r * np.cos(a), ys, r * np.sin(a)], 1)
        for i in range(len(pts) - 1):
            pass
        k = int(body * 40)
        idx = b.r.integers(0, len(pts) - 1, k)
        u = b.r.random(k)[:, None]
        b.add(pts[idx] * (1 - u) + pts[idx + 1] * u + b.r.normal(0, 0.015, (k, 3)), 1)
    b.square_faces(hw, 0, body, 3.5, 0, tw)
    for y in np.arange(0.75, body, 1.5):
        h = hw(np.array([y]))[0]
        a = tw(np.array([y]))[0]
        c, s = np.cos(a), np.sin(a)
        sq = [(c * x - s * z, s * x + c * z) for (x, z) in [(h, h), (-h, h), (-h, -h), (h, -h)]]
        b.hloop(sq, y, 12, 0)
    b.windows_square(hw, 1.0, body - 0.5, 0.75, 3, 0.35, tw)
    # crown: pyramid
    h = hw(np.array([body]))[0]
    a = tw(np.array([body]))[0]
    for c0 in range(4):
        aa = a + np.pi / 4 + c0 * np.pi / 2
        b.line([h * np.sqrt(2) * np.cos(aa), body, h * np.sqrt(2) * np.sin(aa)], [0, top, 0], 40)
    return b.done()


def pagoda(seed=5):
    b = B(seed)
    top = HMAX
    tier = 3.4
    y = top - 4.5
    b.line([0, y, 0], [0, top, 0], 50, 1, 0.005)
    for yy in np.linspace(y + 0.6, top - 0.6, 6):
        b.circle(0.22, yy, 40)
    k = 0
    while y > 0:
        body_hw = 1.4 + 0.08 * min(k, 6)
        roof_hw = body_hw + 1.6
        y0 = max(y - tier, 0.0)
        # roof: flared skirt from body top outward and down, corners upturned
        n = 900
        s = b.r.random(n)
        side = b.r.integers(0, 4, n)
        t = b.r.uniform(-1, 1, n)
        hwv = body_hw + (roof_hw - body_hw) * s
        droop = -0.9 * s + 0.75 * s ** 3 * np.abs(t) ** 4
        yy = y - 0.1 + droop
        x = np.where(side == 0, hwv, np.where(side == 1, -hwv, t * hwv))
        z = np.where(side == 2, hwv, np.where(side == 3, -hwv, t * hwv))
        b.add(np.stack([x, yy, z], 1), 0)
        # roof eave edge (dense) with upturned corners
        m = 700
        t = b.r.uniform(-1, 1, m)
        side = b.r.integers(0, 4, m)
        ye = y - 0.1 - 0.9 + 0.75 * np.abs(t) ** 4
        x = np.where(side == 0, roof_hw, np.where(side == 1, -roof_hw, t * roof_hw))
        z = np.where(side == 2, roof_hw, np.where(side == 3, -roof_hw, t * roof_hw))
        b.add(np.stack([x, ye, z], 1), 1)
        # body
        if y - 0.9 > y0 + 0.05:
            b.square_faces(lambda q, h=body_hw: np.full_like(q, h), y0, y - 0.9, 3.5)
            for (px, pz) in [(body_hw, body_hw), (-body_hw, body_hw), (-body_hw, -body_hw), (body_hw, -body_hw)]:
                b.line([px, y0, pz], [px, y - 0.9, pz], 24)
            b.windows_square(lambda q, h=body_hw: np.full_like(q, h), y0 + 0.6, y - 1.2, 0.9, 3, 0.45)
        y = y0
        k += 1
    return b.done()


def pod_tower(seed=6):
    b = B(seed)
    top = HMAX
    pod_y = top - 11.0
    R = 3.1
    shaft = 0.9
    for k in range(6):
        a = 2 * np.pi * k / 6
        b.line([shaft * np.cos(a), 0, shaft * np.sin(a)], [shaft * np.cos(a), pod_y - R + 0.3, shaft * np.sin(a)], 28)
    b.revolve(lambda y: np.full_like(y, shaft), 0, pod_y - R, 2.5)
    for y in np.arange(2.0, pod_y - R, 6.0):
        b.circle(shaft + 0.02, y, 30)
    # the pod: latitude rings + surface
    for lat in np.linspace(-0.85, 0.85, 9):
        b.circle(R * np.cos(np.arcsin(lat)), pod_y + R * lat, 30)
    b.revolve(lambda y: np.sqrt(np.maximum(R * R - (y - pod_y) ** 2, 0)) + 0.01, pod_y - R * 0.999, pod_y + R * 0.999, 9.0)
    r2 = 1.35
    y2 = pod_y + R + 2.4
    b.revolve(lambda y: np.sqrt(np.maximum(r2 * r2 - (y - y2) ** 2, 0)) + 0.01, y2 - r2 * 0.999, y2 + r2 * 0.999, 12.0)
    b.line([0, pod_y + R, 0], [0, y2 - r2, 0], 40, 1)
    b.line([0, y2 + r2, 0], [0, top + 2.0, 0], 50, 1, 0.005)
    # window band around the pod equator
    for y in (pod_y - 0.5, pod_y + 0.3, pod_y + 1.1):
        rr = np.sqrt(max(R * R - (y - pod_y) ** 2, 0)) + 0.03
        for k in range(28):
            if b.r.random() < 0.6:
                a = 2 * np.pi * k / 28
                b.add(np.array([[rr * np.cos(a), y, rr * np.sin(a)]]), 2)
    return b.done()


def blade(seed=7):
    """A tall curved sail/blade: lens-shaped section that narrows and leans to a sharp tip."""
    b = B(seed)
    top = HMAX
    n = 60

    def sect(y):
        u = np.clip(y / top, 0, 1)
        L = 3.4 * (1 - u ** 1.6) + 0.05          # half-length of the lens section
        th = 0.55 * (1 - u) + 0.04                 # half-thickness
        lean = 2.2 * u ** 2.2                      # tip leans
        return L, th, lean
    ys = np.linspace(0, top, 700)
    for side in (-1, 1):
        # leading / trailing edges (dense lines)
        L, th, lean = sect(ys)
        x = side * L + lean
        pts = np.stack([x, ys, np.zeros_like(ys)], 1)
        k = int(top * 45 * DENS_L)
        idx = b.r.integers(0, len(pts) - 1, k)
        uu = b.r.random(k)[:, None]
        b.add(pts[idx] * (1 - uu) + pts[idx + 1] * uu + b.r.normal(0, 0.015, (k, 3)), 1)
    # faces (both sides of the lens)
    m = int(top * 7 * 2 * 2.4 * DENS_S)
    y = b.r.uniform(0, top, m)
    L, th, lean = sect(y)
    s = b.r.uniform(-1, 1, m)
    side = np.where(b.r.random(m) < 0.5, -1.0, 1.0)
    x = s * L + lean
    z = side * th * np.sqrt(np.maximum(1 - s * s, 0))
    nrm = np.stack([np.zeros(m), np.zeros(m), side], 1)
    b.add(np.stack([x, y, z], 1), 0, nrm)
    # floor lines across the faces
    for yy in np.arange(1.0, top - 2.0, 1.6):
        L, th, lean = sect(np.array([yy]))
        s = b.r.uniform(-1, 1, 70)
        side = np.where(b.r.random(70) < 0.5, -1.0, 1.0)
        x = s * L[0] + lean[0]
        z = side * th[0] * np.sqrt(np.maximum(1 - s * s, 0))
        b.add(np.stack([x, np.full(70, yy), z], 1), 0, np.stack([np.zeros(70), np.zeros(70), side], 1))
    # windows
    for yy in np.arange(1.2, top - 4.0, 0.8):
        L, th, lean = sect(np.array([yy]))
        for k in range(8):
            if b.r.random() < 0.5:
                s = -0.85 + 1.7 * (k + 0.5) / 8
                sd = 1.0 if b.r.random() < 0.5 else -1.0
                b.add(np.array([[s * L[0] + lean[0], yy, sd * (th[0] * np.sqrt(1 - s * s) + 0.03)]]), 2,
                      np.array([[0, 0, sd]]))
    return b.done()


BUILDERS = [needle_spire, ziggurat, lattice_mast, ringed_cylinder, twisted, pagoda, pod_tower, blade]
NAMES = ['needle spire', 'ziggurat', 'lattice mast', 'ringed cylinder', 'twisted prism', 'pagoda', 'pod tower',
         'blade']


def build_all():
    return [f(i) for i, f in enumerate(BUILDERS)]


if __name__ == '__main__':
    for nm, t in zip(NAMES, build_all()):
        print(nm, len(t['p']), 'lines', int((t['kind'] == 1).sum()), 'windows', int((t['kind'] == 2).sum()),
              'ymin %.1f ymax %.1f' % (t['p'][:, 1].min(), t['p'][:, 1].max()))
