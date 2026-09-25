"""HILLS core: camera, easing/keyframes, numba noise, splatting, compositing.

World units are metres. x right, y up, z forward (left-handed, like the screen).
All radiometry is linear-light HDR float32; `look.finish()` makes the display image.
"""
import math
import os
import sys

import numba as nb
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if os.path.join(ROOT, 'lib') not in sys.path:
    sys.path.insert(0, os.path.join(ROOT, 'lib'))
import look  # noqa: E402

FULL_W, FULL_H = look.W, look.H
RENDER_DIR = os.path.join(ROOT, 'renders', 'hills')
CACHE_DIR = os.path.join(ROOT, 'shots', 'hills', 'cache')


def C(name, gain=1.0):
    """Palette colour (linear) times gain."""
    return look.hexrgb(look.PALETTE[name]) * np.float32(gain)


def rgb(h, gain=1.0):
    return look.hexrgb(h) * np.float32(gain)


# ------------------------------------------------------------------ easing ---

def clamp(x, a=0.0, b=1.0):
    return a if x < a else (b if x > b else x)


def smoothstep(e0, e1, x):
    t = clamp((x - e0) / (e1 - e0))
    return t * t * (3 - 2 * t)


def smootherstep(e0, e1, x):
    t = clamp((x - e0) / (e1 - e0))
    return t * t * t * (t * (t * 6 - 15) + 10)


def ease_in(t, p=2.0):
    return clamp(t) ** p


def ease_out(t, p=2.0):
    return 1 - (1 - clamp(t)) ** p


def ease_io(t, p=2.0):
    t = clamp(t)
    return 0.5 * (2 * t) ** p if t < 0.5 else 1 - 0.5 * (2 - 2 * t) ** p


EASES = {
    'lin': lambda t: clamp(t),
    'io': lambda t: ease_io(t, 2.0),
    'io3': lambda t: ease_io(t, 3.0),
    'in': lambda t: ease_in(t, 2.0),
    'in3': lambda t: ease_in(t, 3.0),
    'out': lambda t: ease_out(t, 2.0),
    'out3': lambda t: ease_out(t, 3.0),
    'smooth': lambda t: smootherstep(0, 1, t),
    'hold': lambda t: 0.0,
}


def track(f, keys, default_ease='smooth'):
    """Keyframe track. keys = [(frame, value), ...] or (frame, value, ease) where the
    ease applies to the segment ENDING at that key. Values may be floats or arrays."""
    if f <= keys[0][0]:
        return np.asarray(keys[0][1], np.float64) if hasattr(keys[0][1], '__len__') else keys[0][1]
    for i in range(1, len(keys)):
        k0, k1 = keys[i - 1], keys[i]
        if f <= k1[0]:
            e = k1[2] if len(k1) > 2 else default_ease
            t = EASES[e]((f - k0[0]) / max(1e-9, (k1[0] - k0[0])))
            v0, v1 = np.asarray(k0[1], np.float64), np.asarray(k1[1], np.float64)
            v = v0 + (v1 - v0) * t
            return v if v.ndim else float(v)
    last = keys[-1][1]
    return np.asarray(last, np.float64) if hasattr(last, '__len__') else last


def fnoise1(t, seed=0.0, octaves=3):
    """Smooth 1-D noise in [-1,1]-ish (python scalar), for idle motion / flicker."""
    v = 0.0
    a = 1.0
    norm = 0.0
    fr = 1.0
    for o in range(octaves):
        v += a * _pn1(t * fr + seed * 17.13 + o * 31.7)
        norm += a
        a *= 0.5
        fr *= 2.03
    return v / norm


def _pn1(x):
    i = math.floor(x)
    fx = x - i
    g0 = _h1(i) * 2 - 1
    g1 = _h1(i + 1) * 2 - 1
    u = fx * fx * fx * (fx * (fx * 6 - 15) + 10)
    return 2.0 * (g0 * fx * (1 - u) + g1 * (fx - 1) * u)


def _h1(i):
    n = (int(i) * 374761393 + 668265263) & 0xFFFFFFFF
    n = ((n ^ (n >> 13)) * 1274126177) & 0xFFFFFFFF
    return ((n ^ (n >> 16)) & 0xFFFFFF) / float(0xFFFFFF)


# ------------------------------------------------------------------ camera ---

class Camera:
    """Pinhole camera. f is in FULL-RES pixels (1920 wide); `scale` renders smaller."""

    def __init__(self, pos, yaw=0.0, pitch=0.0, roll=0.0, hfov=None, f=None, scale=1.0,
                 shift_x=0.0, shift_y=0.0):
        self.pos = np.asarray(pos, np.float64)
        self.yaw, self.pitch, self.roll = float(yaw), float(pitch), float(roll)
        if f is None:
            f = (FULL_W / 2) / math.tan(math.radians(hfov) / 2)
        self.f_full = float(f)
        self.scale = float(scale)
        self.W = int(round(FULL_W * scale))
        self.H = int(round(FULL_H * scale))
        self.f = self.f_full * scale
        self.cx = self.W / 2 + shift_x * scale
        self.cy = self.H / 2 + shift_y * scale
        cy_, sy_ = math.cos(self.yaw), math.sin(self.yaw)
        cp, sp = math.cos(self.pitch), math.sin(self.pitch)
        fwd = np.array([sy_ * cp, sp, cy_ * cp])
        right = np.cross([0.0, 1.0, 0.0], fwd)
        right /= np.linalg.norm(right)
        up = np.cross(fwd, right)
        if self.roll:
            cr, sr = math.cos(self.roll), math.sin(self.roll)
            right, up = right * cr + up * sr, -right * sr + up * cr
        self.R = np.stack([right, up, fwd], axis=1)  # world = R @ cam

    def look_at(self, target):
        d = np.asarray(target, np.float64) - self.pos
        yaw = math.atan2(d[0], d[2])
        pitch = math.atan2(d[1], math.hypot(d[0], d[2]))
        return yaw, pitch

    @property
    def hfov(self):
        return 2 * math.degrees(math.atan(self.W / 2 / self.f))

    def params(self):
        """Packed float64 array for numba kernels:
        [px,py,pz, R00..R22 (row major), f, cx, cy, W, H]"""
        return np.concatenate([self.pos, self.R.reshape(-1),
                               [self.f, self.cx, self.cy, self.W, self.H]]).astype(np.float64)

    def project(self, P):
        """P (...,3) world -> (sx, sy, depth) in render pixels."""
        P = np.asarray(P, np.float64)
        q = (P - self.pos) @ self.R          # cam coords (x right, y up, z fwd)
        z = q[..., 2]
        zs = np.where(np.abs(z) < 1e-9, 1e-9, z)
        sx = self.cx + self.f * q[..., 0] / zs
        sy = self.cy - self.f * q[..., 1] / zs
        return sx, sy, z

    def ppm(self, depth):
        """render pixels per metre at a given depth"""
        return self.f / max(depth, 1e-6)


@nb.njit(cache=True, fastmath=True, inline='always')
def cam_ray(cam, px, py):
    """Unit world ray through render pixel centre (px, py) (floats, pixel units)."""
    x = (px - cam[13]) / cam[12]
    y = -(py - cam[14]) / cam[12]
    dx = cam[3] * x + cam[4] * y + cam[5]
    dy = cam[6] * x + cam[7] * y + cam[8]
    dz = cam[9] * x + cam[10] * y + cam[11]
    n = 1.0 / math.sqrt(dx * dx + dy * dy + dz * dz)
    return dx * n, dy * n, dz * n


# ------------------------------------------------------------------- noise ---

_perm = np.random.RandomState(20260925).permutation(256)
PERM = np.concatenate([_perm, _perm]).astype(np.int64)


@nb.njit(cache=True, fastmath=True, inline='always')
def _fade(t):
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


@nb.njit(cache=True, fastmath=True, inline='always')
def _g3(h, x, y, z):
    h = h & 15
    u = x if h < 8 else y
    if h < 4:
        v = y
    elif h == 12 or h == 14:
        v = x
    else:
        v = z
    return (u if (h & 1) == 0 else -u) + (v if (h & 2) == 0 else -v)


@nb.njit(cache=True, fastmath=True)
def perlin3(x, y, z):
    fx0 = math.floor(x)
    fy0 = math.floor(y)
    fz0 = math.floor(z)
    X = int(fx0) & 255
    Y = int(fy0) & 255
    Z = int(fz0) & 255
    x -= fx0
    y -= fy0
    z -= fz0
    u = _fade(x)
    v = _fade(y)
    w = _fade(z)
    P = PERM
    A = P[X] + Y
    AA = P[A] + Z
    AB = P[A + 1] + Z
    B = P[X + 1] + Y
    BA = P[B] + Z
    BB = P[B + 1] + Z
    x1 = _g3(P[AA], x, y, z) + u * (_g3(P[BA], x - 1, y, z) - _g3(P[AA], x, y, z))
    x2 = _g3(P[AB], x, y - 1, z) + u * (_g3(P[BB], x - 1, y - 1, z) - _g3(P[AB], x, y - 1, z))
    y1 = x1 + v * (x2 - x1)
    x3 = _g3(P[AA + 1], x, y, z - 1) + u * (_g3(P[BA + 1], x - 1, y, z - 1) - _g3(P[AA + 1], x, y, z - 1))
    x4 = _g3(P[AB + 1], x, y - 1, z - 1) + u * (_g3(P[BB + 1], x - 1, y - 1, z - 1) - _g3(P[AB + 1], x, y - 1, z - 1))
    y2 = x3 + v * (x4 - x3)
    return y1 + w * (y2 - y1)


@nb.njit(cache=True, fastmath=True, inline='always')
def _g2(h, x, y):
    h = h & 7
    if h == 0:
        return x + y
    elif h == 1:
        return -x + y
    elif h == 2:
        return x - y
    elif h == 3:
        return -x - y
    elif h == 4:
        return 1.4142 * x
    elif h == 5:
        return -1.4142 * x
    elif h == 6:
        return 1.4142 * y
    return -1.4142 * y


@nb.njit(cache=True, fastmath=True)
def perlin2(x, y):
    fx0 = math.floor(x)
    fy0 = math.floor(y)
    X = int(fx0) & 255
    Y = int(fy0) & 255
    x -= fx0
    y -= fy0
    u = _fade(x)
    v = _fade(y)
    P = PERM
    A = P[X] + Y
    B = P[X + 1] + Y
    a = _g2(P[A], x, y)
    b = _g2(P[B], x - 1, y)
    c = _g2(P[A + 1], x, y - 1)
    d = _g2(P[B + 1], x - 1, y - 1)
    return 0.9 * (a + u * (b - a) + v * ((c + u * (d - c)) - (a + u * (b - a))))


@nb.njit(cache=True, fastmath=True)
def fbm2(x, y, octaves, lac, gain):
    s = 0.0
    a = 1.0
    n = 0.0
    for o in range(octaves):
        s += a * perlin2(x, y)
        n += a
        a *= gain
        x = x * lac + 17.3
        y = y * lac - 9.1
    return s / n


@nb.njit(cache=True, fastmath=True)
def fbm3(x, y, z, octaves, lac, gain):
    s = 0.0
    a = 1.0
    n = 0.0
    for o in range(octaves):
        s += a * perlin3(x, y, z)
        n += a
        a *= gain
        x = x * lac + 17.3
        y = y * lac - 9.1
        z = z * lac + 3.7
    return s / n


@nb.njit(cache=True, fastmath=True)
def hash11(n):
    """int -> float in [0,1)"""
    n = (n * 374761393 + 668265263) & 0xFFFFFFFF
    n = ((n ^ (n >> 13)) * 1274126177) & 0xFFFFFFFF
    return ((n ^ (n >> 16)) & 0xFFFFFF) / 16777216.0


def fbm1_np(x, octaves=6, lac=2.0, gain=0.5, seed=0):
    """Vectorised 1-D fbm for ridge profiles (numpy array input)."""
    x = np.asarray(x, np.float64)
    out = np.zeros_like(x)
    _fbm1_kernel(x.ravel(), out.ravel(), octaves, lac, gain, float(seed))
    return out


@nb.njit(cache=True, fastmath=True)
def _fbm1_kernel(x, out, octaves, lac, gain, seed):
    for i in range(x.shape[0]):
        out[i] = fbm2(x[i], seed * 7.31 + 0.5, octaves, lac, gain)


# --------------------------------------------------------------- splatting ---

@nb.njit(cache=True, fastmath=True)
def splat_gauss(img, x, y, sigma, r, g, b):
    """Add an energy-normalised gaussian (sum of weights == 1) of colour (r,g,b)."""
    H = img.shape[0]
    W = img.shape[1]
    if sigma < 0.3:
        sigma = 0.3
    rad = int(math.ceil(sigma * 3.0))
    ix = int(math.floor(x))
    iy = int(math.floor(y))
    if ix + rad < 0 or iy + rad < 0 or ix - rad >= W or iy - rad >= H:
        return
    inv = 1.0 / (2.0 * sigma * sigma)
    tot = 0.0
    for j in range(iy - rad, iy + rad + 1):
        dy = j + 0.5 - y
        for i in range(ix - rad, ix + rad + 1):
            dx = i + 0.5 - x
            tot += math.exp(-(dx * dx + dy * dy) * inv)
    if tot <= 0:
        return
    k = 1.0 / tot
    for j in range(max(0, iy - rad), min(H, iy + rad + 1)):
        dy = j + 0.5 - y
        for i in range(max(0, ix - rad), min(W, ix + rad + 1)):
            dx = i + 0.5 - x
            w = math.exp(-(dx * dx + dy * dy) * inv) * k
            img[j, i, 0] += r * w
            img[j, i, 1] += g * w
            img[j, i, 2] += b * w


@nb.njit(cache=True, fastmath=True)
def splat_disc(img, x, y, radius, r, g, b, soft):
    """Energy-normalised soft disc (bokeh)."""
    H = img.shape[0]
    W = img.shape[1]
    if radius < 0.7:
        splat_gauss(img, x, y, 0.5 + 0.3 * radius, r, g, b)
        return
    rad = int(math.ceil(radius + soft + 1))
    ix = int(math.floor(x))
    iy = int(math.floor(y))
    if ix + rad < 0 or iy + rad < 0 or ix - rad >= W or iy - rad >= H:
        return
    area = math.pi * radius * radius
    k = 1.0 / area
    for j in range(max(0, iy - rad), min(H, iy + rad + 1)):
        dy = j + 0.5 - y
        for i in range(max(0, ix - rad), min(W, ix + rad + 1)):
            dx = i + 0.5 - x
            d = math.sqrt(dx * dx + dy * dy)
            w = radius - d
            w = 0.5 + w / (2.0 * soft)
            if w <= 0:
                continue
            if w > 1:
                w = 1.0
            # slightly brighter rim (lens bokeh)
            rim = 1.0 + 0.35 * (d / radius) ** 4 if d < radius else 1.0
            w *= k * rim
            img[j, i, 0] += r * w
            img[j, i, 1] += g * w
            img[j, i, 2] += b * w


@nb.njit(cache=True, fastmath=True)
def splat_streak(img, x0, y0, x1, y1, sigma, r, g, b):
    """Motion-blurred point: energy spread uniformly along the segment."""
    L = math.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)
    n = int(L / max(0.35, sigma * 0.6)) + 1
    if n > 400:
        n = 400
    inv = 1.0 / n
    for s in range(n):
        t = (s + 0.5) * inv
        splat_gauss(img, x0 + (x1 - x0) * t, y0 + (y1 - y0) * t, sigma, r * inv, g * inv, b * inv)


@nb.njit(cache=True, fastmath=True)
def splat_particles(img, xs0, ys0, xs1, ys1, sig, cols):
    """Batch of streaks: arrays of start/end positions, sigma, colour (N,3) energy."""
    for i in range(xs0.shape[0]):
        if sig[i] > 2.5:
            # defocused: bokeh disc (motion smear approximated by 2 discs)
            mx = 0.5 * (xs0[i] + xs1[i])
            my = 0.5 * (ys0[i] + ys1[i])
            L = math.sqrt((xs1[i] - xs0[i]) ** 2 + (ys1[i] - ys0[i]) ** 2)
            if L > sig[i] * 0.5:
                n = int(L / (sig[i] * 0.5)) + 1
                if n > 12:
                    n = 12
                for s in range(n):
                    t = (s + 0.5) / n
                    splat_disc(img, xs0[i] + (xs1[i] - xs0[i]) * t, ys0[i] + (ys1[i] - ys0[i]) * t,
                               sig[i], cols[i, 0] / n, cols[i, 1] / n, cols[i, 2] / n, 1.0)
            else:
                splat_disc(img, mx, my, sig[i], cols[i, 0], cols[i, 1], cols[i, 2], 1.0)
        else:
            splat_streak(img, xs0[i], ys0[i], xs1[i], ys1[i], sig[i], cols[i, 0], cols[i, 1], cols[i, 2])


# ------------------------------------------------------------- compositing ---

def over(dst, rgb_premul, alpha):
    """dst = src + dst*(1-a); alpha HxW (or HxWx1)."""
    if alpha.ndim == 2:
        alpha = alpha[..., None]
    dst *= (1.0 - alpha)
    dst += rgb_premul
    return dst


def over_region(dst, y0, x0, rgb_premul, alpha):
    h, w = alpha.shape[:2]
    H, W = dst.shape[:2]
    ya, xa = max(0, y0), max(0, x0)
    yb, xb = min(H, y0 + h), min(W, x0 + w)
    if yb <= ya or xb <= xa:
        return dst
    sub = dst[ya:yb, xa:xb]
    a = alpha[ya - y0:yb - y0, xa - x0:xb - x0]
    if a.ndim == 2:
        a = a[..., None]
    sub *= (1.0 - a)
    sub += rgb_premul[ya - y0:yb - y0, xa - x0:xb - x0]
    return dst


def ensure_dirs():
    os.makedirs(RENDER_DIR, exist_ok=True)
    os.makedirs(os.path.join(RENDER_DIR, 'tests'), exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)
