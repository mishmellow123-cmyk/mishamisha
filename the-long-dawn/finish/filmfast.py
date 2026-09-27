"""THE LONG DAWN · FINISH, the delivery path: filmfinish.Look fused into two numba kernels (same maths, one pass
over the pixels each), for the edit's per-frame stage. Validate against the reference with `python3 filmfast.py`.

    import filmfast
    fin = filmfast.Fast(filmfinish.make('500T_2383_fire'))
    out = fin(img_srgb01_float32, cut, frame)                 # ~0.2-0.4 s a frame at 1920x804

License: GPL-3.0-or-later (it follows spektrafilm's published halation and grain models; see edit/CREDITS.md).
"""
import os
import sys

import cv2
import numpy as np
import numba as nb

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import filmfinish as FF  # noqa: E402

TAB = 4096


@nb.njit(inline='always', fastmath=True)
def _tet(lut, r, g, b):
    n = lut.shape[0] - 1
    r = min(max(r, 0.0), 1.0) * n
    g = min(max(g, 0.0), 1.0) * n
    b = min(max(b, 0.0), 1.0) * n
    i = min(int(r), n - 1)
    j = min(int(g), n - 1)
    k = min(int(b), n - 1)
    fr, fg, fb = r - i, g - j, b - k
    o0 = 0.0
    o1 = 0.0
    o2 = 0.0
    for c in range(3):
        c000 = lut[i, j, k, c]
        c111 = lut[i + 1, j + 1, k + 1, c]
        if fr > fg:
            if fg > fb:
                v = (1 - fr) * c000 + (fr - fg) * lut[i + 1, j, k, c] + (fg - fb) * lut[i + 1, j + 1, k, c] + fb * c111
            elif fr > fb:
                v = (1 - fr) * c000 + (fr - fb) * lut[i + 1, j, k, c] + (fb - fg) * lut[i + 1, j, k + 1, c] + fg * c111
            else:
                v = (1 - fb) * c000 + (fb - fr) * lut[i, j, k + 1, c] + (fr - fg) * lut[i + 1, j, k + 1, c] + fg * c111
        else:
            if fb > fg:
                v = (1 - fb) * c000 + (fb - fg) * lut[i, j, k + 1, c] + (fg - fr) * lut[i, j + 1, k + 1, c] + fr * c111
            elif fb > fr:
                v = (1 - fg) * c000 + (fg - fb) * lut[i, j + 1, k, c] + (fb - fr) * lut[i, j + 1, k + 1, c] + fr * c111
            else:
                v = (1 - fg) * c000 + (fg - fr) * lut[i, j + 1, k, c] + (fr - fb) * lut[i + 1, j + 1, k, c] + fb * c111
        if c == 0:
            o0 = v
        elif c == 1:
            o1 = v
        else:
            o2 = v
    return o0, o1, o2


NT = 8192
_DEC = np.array([s / 12.92 if s <= 0.04045 else ((s + 0.055) / 1.055) ** 2.4
                 for s in np.linspace(0, 1, NT + 1)], np.float32)
_ENC = np.array([12.92 * x if x <= 0.0031308 else 1.055 * x ** (1 / 2.4) - 0.055
                 for x in np.linspace(0, 1, NT + 1)], np.float32)


@nb.njit(inline='always', fastmath=True)
def _tab(t, x):
    x = min(max(x, 0.0), 1.0) * NT
    i = min(int(x), NT - 1)
    u = x - i
    return t[i] * (1 - u) + t[i + 1] * u


@nb.njit(inline='always', fastmath=True)
def _dec(s):
    return _tab(_DEC, s)


@nb.njit(inline='always', fastmath=True)
def _enc(x):
    return _tab(_ENC, x)


@nb.njit(inline='always', fastmath=True)
def _rrt_inv(t):
    A, B, C, D, E = 0.0245786, 0.000090537, 0.983729, 0.4329510, 0.238081
    v0 = (-A + np.sqrt(A * A + 4 * B)) / 2.0
    if t < 0:
        slope0 = (2 * v0 + A) / (C * v0 * v0 + D * v0 + E)
        return v0 + t / slope0
    tc = min(t, 1.0 / C - 1e-4)
    q = 1.0 - tc * C
    p = A - tc * D
    r = B + tc * E
    v = (-p + np.sqrt(p * p + 4.0 * q * r)) / (2.0 * q)
    return min(v, 32.0)


@nb.njit(inline='always', fastmath=True)
def _cct(x):
    if x <= 0.0078125:
        return 10.5402377416545 * x + 0.0729055341958355
    return (np.log2(max(x, 1e-10)) + 9.72) / 17.52


@nb.njit(inline='always', fastmath=True)
def _hsv(r, g, b):
    v = max(r, max(g, b))
    m = min(r, min(g, b))
    d = v - m
    s = d / v if v > 0 else 0.0
    if d <= 0:
        h = 0.0
    elif v == r:
        h = 60.0 * (g - b) / d
    elif v == g:
        h = 120.0 + 60.0 * (b - r) / d
    else:
        h = 240.0 + 60.0 * (r - g) / d
    if h < 0:
        h += 360.0
    return h, s, v


@nb.njit(inline='always', fastmath=True)
def _rgb(h, s, v):
    h = h % 360.0
    c = v * s
    hp = h / 60.0
    x = c * (1 - abs(hp % 2 - 1))
    if hp < 1:
        r, g, b = c, x, 0.0
    elif hp < 2:
        r, g, b = x, c, 0.0
    elif hp < 3:
        r, g, b = 0.0, c, x
    elif hp < 4:
        r, g, b = 0.0, x, c
    elif hp < 5:
        r, g, b = x, 0.0, c
    else:
        r, g, b = c, 0.0, x
    m = v - c
    return r + m, g + m, b + m


@nb.njit(inline='always', fastmath=True)
def _hal(e, E, a, s):
    """log10((1 - s) * E + s * a) with E = 10**e: e + log10(1 + s * (a / E - 1)); log1p's first terms when tiny."""
    xx = s * (a / max(E, 1e-30) - 1.0)
    if abs(xx) < 0.004:
        return e + xx * 0.4342944819 * (1.0 - 0.5 * xx)
    return e + np.log10(max(1.0 + xx, 1e-10))


@nb.njit(cache=True, fastmath=True)
def _kernel_a(src, mo, ma, m7, gain, lift, l1, le_min, le_span, le, hsc):
    h, w = src.shape[0], src.shape[1]
    for y in range(h):
        for x in range(w):
            x0 = (_dec(src[y, x, 0]) - lift) / (1 - lift)
            x1 = (_dec(src[y, x, 1]) - lift) / (1 - lift)
            x2 = (_dec(src[y, x, 2]) - lift) / (1 - lift)
            v0 = _rrt_inv(mo[0, 0] * x0 + mo[0, 1] * x1 + mo[0, 2] * x2)
            v1 = _rrt_inv(mo[1, 0] * x0 + mo[1, 1] * x1 + mo[1, 2] * x2)
            v2 = _rrt_inv(mo[2, 0] * x0 + mo[2, 1] * x1 + mo[2, 2] * x2)
            a0 = gain * (ma[0, 0] * v0 + ma[0, 1] * v1 + ma[0, 2] * v2)
            a1 = gain * (ma[1, 0] * v0 + ma[1, 1] * v1 + ma[1, 2] * v2)
            a2 = gain * (ma[2, 0] * v0 + ma[2, 1] * v1 + ma[2, 2] * v2)
            c0, c1, c2 = _tet(l1, _cct(a0), _cct(a1), _cct(a2))
            le[y, x, 0] = c0 * le_span + le_min
            le[y, x, 1] = c1 * le_span + le_min
            le[y, x, 2] = c2 * le_span + le_min
            r = max(m7[0, 0] * v0 + m7[0, 1] * v1 + m7[0, 2] * v2, 0.0)
            g = max(m7[1, 0] * v0 + m7[1, 1] * v1 + m7[1, 2] * v2, 0.0)
            b = max(m7[2, 0] * v0 + m7[2, 1] * v1 + m7[2, 2] * v2, 0.0)
            hh, ss, vv = _hsv(r, g, b)
            hsc[y, x] = hh


@nb.njit(cache=True, fastmath=True)
def _kernel_b(src, le, ee, acc, hal, noise, gamt, gtab, l2, l3, le_min, le_span, dmin, dspan, tabs,
              fire_on, restore, max_shift, red_floor, hsc, lift, mix, out):
    h, w = src.shape[0], src.shape[1]
    nt = tabs.shape[1] - 1
    ng = gtab.shape[1] - 1
    for y in range(h):
        for x in range(w):
            e0 = le[y, x, 0]
            e1 = le[y, x, 1]
            e2 = le[y, x, 2]
            if hal[0] > 0:
                e0 = _hal(e0, ee[y, x, 0], acc[y, x, 0], hal[0])
            if hal[1] > 0:
                e1 = _hal(e1, ee[y, x, 1], acc[y, x, 1], hal[1])
            if hal[2] > 0:
                e2 = _hal(e2, ee[y, x, 2], acc[y, x, 2], hal[2])
            d0, d1, d2 = _tet(l2, (e0 - le_min) / le_span, (e1 - le_min) / le_span, (e2 - le_min) / le_span)
            dd = (d0 * dspan[0] + dmin[0], d1 * dspan[1] + dmin[1], d2 * dspan[2] + dmin[2])
            q0 = 0.0
            q1 = 0.0
            q2 = 0.0
            for c in range(3):
                D = dd[c]
                if gamt > 0:
                    t = min(max((D - dmin[c]) / dspan[c], 0.0), 1.0) * ng
                    i = min(int(t), ng - 1)
                    u = t - i
                    D = D + gamt * (gtab[c, i] * (1 - u) + gtab[c, i + 1] * u) * noise[y, x, c]
                q = min(max((D - dmin[c]) / dspan[c], 0.0), 1.0)
                if c == 0:
                    q0 = q
                elif c == 1:
                    q1 = q
                else:
                    q2 = q
            f0, f1, f2 = _tet(l3, q0, q1, q2)
            # print grey balance (per-channel 1-D tables over display [0, 1])
            fs = (f0, f1, f2)
            fb0 = 0.0
            fb1 = 0.0
            fb2 = 0.0
            for c in range(3):
                t = min(max(fs[c], 0.0), 1.0) * nt
                i = min(int(t), nt - 1)
                u = t - i
                val = tabs[c, i] * (1 - u) + tabs[c, i + 1] * u
                if c == 0:
                    fb0 = val
                elif c == 1:
                    fb1 = val
                else:
                    fb2 = val
            s0, s1, s2 = src[y, x, 0], src[y, x, 1], src[y, x, 2]
            fl = (fb0, fb1, fb2)
            sl = (s0, s1, s2)
            o0 = 0.0
            o1 = 0.0
            o2 = 0.0
            for c in range(3):
                f = lift + (1.0 - lift) * _dec(fl[c])
                a = _dec(sl[c])
                v = _enc(a + mix * (f - a))
                if c == 0:
                    o0 = v
                elif c == 1:
                    o1 = v
                else:
                    o2 = v
            # the fire rule, on the final colour
            if fire_on and s0 >= s1 and s0 >= s2 and s0 > 0.08 and (s0 - min(s1, s2)) > 0.2 * s0:
                hs, ss, vs = _hsv(s0, s1, s2)
                if (hs < 60.0 or hs > 345.0) and ss > 0.2 and vs > 0.08:
                    hf, sf, vf = _hsv(o0, o1, o2)
                    hsw = hs - 360.0 if hs > 180.0 else hs
                    hfw = hf - 360.0 if hf > 180.0 else hf
                    hh = hfw
                    if restore > 0:
                        hc = hsc[y, x]
                        hcw = hc - 360.0 if hc > 180.0 else hc
                        target = max(hcw, min(red_floor, hfw))
                        if hfw > target:
                            hh = hfw + restore * (target - hfw)
                    hh = min(hh, hsw + max_shift)
                    if abs(hh - hfw) > 0.05:
                        wgt = min(max((ss - 0.2) / 0.2, 0.0), 1.0) * min(max((vs - 0.08) / 0.1, 0.0), 1.0)
                        r2, g2, b2 = _rgb(hh + 360.0 if hh < 0 else hh, sf, vf)
                        o0 = o0 * (1 - wgt) + r2 * wgt
                        o1 = o1 * (1 - wgt) + g2 * wgt
                        o2 = o2 * (1 - wgt) + b2 * wgt
            out[y, x, 0] = min(max(o0, 0.0), 1.0)
            out[y, x, 1] = min(max(o1, 0.0), 1.0)
            out[y, x, 2] = min(max(o2, 0.0), 1.0)


def blur_acc(E, sig, bounces=3, decay=0.5):
    """sum_k w_k * G(sig*(k+1)) * E, at half resolution (sig >= 5 px, so the result is the same to 1e-3)."""
    h, w = E.shape
    wts = np.array([decay ** k for k in range(bounces)], np.float32)
    wts /= wts.sum()
    small = cv2.resize(E, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
    acc = np.zeros_like(small)
    for k in range(bounces):
        acc += wts[k] * cv2.GaussianBlur(small, (0, 0), sig * (k + 1) / 2.0)
    return cv2.resize(acc, (w, h), interpolation=cv2.INTER_LINEAR)


def grain_noise(seed, h, w, rho=0.7, blur_px=0.65):
    """The three layers' unit grain fields (blurred by the dye clouds): the same draw as filmfinish.grain()."""
    rng = np.random.Generator(np.random.PCG64(seed))
    shared = rng.standard_normal((h, w), dtype=np.float32)
    own = rng.standard_normal((h, w, 3), dtype=np.float32)
    k = np.float32(np.sqrt(max(1.0 - rho * rho, 0.0)))
    g = np.empty((h, w, 3), np.float32)
    for c in range(3):
        gc = rho * shared + k * own[..., c]
        g[..., c] = cv2.GaussianBlur(gc, (0, 0), blur_px) if blur_px > 0.4 else gc
    return g


class Fast:
    """The delivery path of a filmfinish.Look (grain_only looks fall back to the reference, which is cheap)."""

    def __init__(self, look, hal_strength=(0.05, 0.015, 0.0), particle_um2=0.2, scale=(1.6, 1.6, 3.2),
                 uniformity=(0.97, 0.99, 0.97)):
        self.look = look
        if look.grain_only:
            return
        b = look.bundle
        self.l1, self.l2, self.l3 = (np.ascontiguousarray(x, np.float32) for x in (b.l1, b.l2, b.l3))
        self.le_min, self.le_span = float(b.le_min), float(b.le_max - b.le_min)
        self.dmin = b.d_min.astype(np.float64)
        self.dspan = (b.d_max - b.d_min).astype(np.float64)
        self.dmax = b.d_max.astype(np.float64)
        self.mo = FF._OUT_INV.astype(np.float64)
        self.ma = (FF._709_TO_AP1 @ FF._CURVE_TO_709.astype(np.float64))
        self.m7 = FF._CURVE_TO_709.astype(np.float64)
        self.hal = np.array([s * look.hal_amt for s in hal_strength], np.float64)
        self.sig = 65.0 / FF.PIX_UM
        area = FF.PIX_UM * FF.PIX_UM
        gn = [area / (particle_um2 * s) for s in scale]
        dens = self.dmin[:, None] + self.dspan[:, None] * np.linspace(0, 1, 2049)[None, :]
        self.gtab = np.empty((3, 2049), np.float64)
        for c in range(3):                                  # grain sd(D): filmfinish.grain's statistics
            d = np.maximum(dens[c] + 0.03, 0.0)
            p = np.clip(d / self.dmax[c], 0.0, 1.0)
            neff = gn[c] * (0.5 + 0.5 * np.clip(p * 2.0, 0, 1))
            self.gtab[c] = np.sqrt(np.maximum(d * self.dmax[c] * (1.0 - p * uniformity[c]), 0.0) / neff)
        xs = np.linspace(0, 1, TAB + 1)
        if look.curves is not None:
            self.tabs = np.stack([np.interp(xs, *look.curves[c]) for c in range(3)]).astype(np.float64)
        else:
            self.tabs = np.stack([xs] * 3).astype(np.float64)

    def __call__(self, srgb01, cut='A', frame=0):
        L = self.look
        if L.grain_only:
            return L(srgb01, cut, frame)
        src = np.ascontiguousarray(np.clip(srgb01, 0, 1), np.float32)
        h, w = src.shape[:2]
        le = np.empty((h, w, 3), np.float32)
        hsc = np.empty((h, w), np.float32)
        _kernel_a(src, self.mo, self.ma, self.m7, float(L.gain), float(FF.LIFT), self.l1, self.le_min,
                  self.le_span, le, hsc)
        acc = np.zeros((h, w, 3), np.float32)
        ee = np.zeros((h, w, 3), np.float32)
        for c in range(3):
            if self.hal[c] > 0:
                ee[..., c] = np.power(np.float32(10.0), le[..., c])
                acc[..., c] = blur_acc(ee[..., c], self.sig)
        seed = FF.frame_seed(cut, frame)
        noise = grain_noise(seed, h, w, rho=L.grain_rho) if L.grain_amt > 0 else np.zeros((1, 1, 3), np.float32)
        out = np.empty_like(src)
        fire_on = bool(L.fire_guard or L.fire_restore > 0)
        _kernel_b(src, le, ee, acc, self.hal, noise, float(L.grain_amt), self.gtab, self.l2, self.l3,
                  self.le_min, self.le_span, self.dmin, self.dspan, self.tabs, fire_on,
                  float(L.fire_restore), 2.0, 22.0, hsc, float(L.black_lift), float(L.mix), out)
        return out


if __name__ == '__main__':                                          # validate against the reference path
    import time
    os.environ.setdefault('NUMBA_CACHE_DIR', os.path.join(HERE, '.numba'))
    p = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(HERE), 'renders', 'h1_v3h5',
                                                           'f_01480.jpg')
    img = cv2.imread(p)[..., ::-1].astype(np.float32) / 255
    for name in ('500T_2383_fire', '250D_2383_fire'):
        ref = FF.make(name)
        fast = Fast(ref)
        fast(img[:8, :8], 'A', 1)                                     # compile
        t = time.time()
        a = fast(img, 'A', 7)
        tf = time.time() - t
        t = time.time()
        r = ref(img, 'A', 7)
        tr = time.time() - t
        d = np.abs(a - r) * 255
        print(f'{name}: fast {tf:.2f} s, reference {tr:.2f} s; |fast-ref| in 8-bit codes: mean {d.mean():.3f}, '
              f'p99.9 {np.percentile(d, 99.9):.2f}, max {d.max():.2f}')
