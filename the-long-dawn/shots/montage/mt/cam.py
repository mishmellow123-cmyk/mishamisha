"""Pinhole camera with yaw + lens-shift tilt (verticals stay vertical), plus animation helpers.

World: metres, y up. yaw=0 looks toward +z, right = +x.
Screen: sx = cx + f * x_cam / z_cam,  sy = cy - f * y_cam / z_cam + shift
(shift = f * tan(tilt): positive tilt looks up without keystoning).
"""
import math

import numpy as np


class Cam:
    def __init__(self, pos, yaw_deg=0.0, tilt_deg=0.0, hfov_deg=50.0, W=1920, H=804, roll_deg=0.0):
        self.pos = np.asarray(pos, np.float64)
        self.yaw = math.radians(yaw_deg)
        self.tilt = math.radians(tilt_deg)
        self.hfov = math.radians(hfov_deg)
        self.W = int(W)
        self.H = int(H)
        self.f = 0.5 * self.W / math.tan(0.5 * self.hfov)
        self.cx = 0.5 * self.W
        self.cy = 0.5 * self.H
        self.shift = self.f * math.tan(self.tilt)
        self.fwd = np.array([math.sin(self.yaw), 0.0, math.cos(self.yaw)])
        self.right = np.array([math.cos(self.yaw), 0.0, -math.sin(self.yaw)])
        self.roll = math.radians(roll_deg)

    def scaled(self, s):
        """Same camera at a different render resolution."""
        c = Cam(self.pos, math.degrees(self.yaw), math.degrees(self.tilt), math.degrees(self.hfov),
                round(self.W * s), round(self.H * s), math.degrees(self.roll))
        return c

    @property
    def cyy(self):
        """Effective vertical centre used by kernels (cy + shift)."""
        return self.cy + self.shift

    def params(self):
        """Flat float64 array for numba kernels."""
        return np.array([self.pos[0], self.pos[1], self.pos[2],
                         self.fwd[0], self.fwd[2], self.right[0], self.right[2],
                         self.f, self.cx, self.cy + self.shift, float(self.W), float(self.H)], np.float64)

    def to_cam(self, P):
        d = np.asarray(P, np.float64) - self.pos
        x = d[..., 0] * self.right[0] + d[..., 2] * self.right[2]
        z = d[..., 0] * self.fwd[0] + d[..., 2] * self.fwd[2]
        y = d[..., 1]
        return x, y, z

    def project(self, P):
        """World point(s) -> (sx, sy, depth_along_view). depth<=0 means behind camera."""
        x, y, z = self.to_cam(P)
        zz = np.maximum(z, 1e-6)
        sx = self.cx + self.f * x / zz
        sy = self.cy + self.shift - self.f * y / zz
        return sx, sy, z

    def unproject(self, sx, sy, depth):
        """Screen point at given view depth -> world point."""
        x = (sx - self.cx) * depth / self.f
        y = (self.cy + self.shift - sy) * depth / self.f
        return self.pos + self.right * x + self.fwd * depth + np.array([0.0, y, 0.0])

    def ppm(self, depth):
        """Pixels per metre at a view depth."""
        return self.f / max(depth, 1e-6)

    def ray_dir(self, sx, sy):
        d = self.fwd * self.f + self.right * (sx - self.cx) + np.array([0.0, self.cy + self.shift - sy, 0.0])
        return d / np.linalg.norm(d)


# ------------------------------------------------------------------ easing ---

def clamp01(x):
    return min(max(x, 0.0), 1.0)


def smooth(x):
    x = clamp01(x)
    return x * x * (3 - 2 * x)


def smoother(x):
    x = clamp01(x)
    return x * x * x * (x * (x * 6 - 15) + 10)


def ease_out(x, p=3.0):
    x = clamp01(x)
    return 1 - (1 - x) ** p


def ease_in(x, p=2.0):
    x = clamp01(x)
    return x ** p


def lin(a, b, x):
    return a + (b - a) * x


def ramp(t, t0, t1):
    """0 before t0, 1 after t1, smoothstep between."""
    if t1 <= t0:
        return 1.0 if t >= t0 else 0.0
    return smooth((t - t0) / (t1 - t0))


def keys(t, kf, ease=smoother):
    """Piecewise interpolation through keyframes [(t, value), ...] (values: float or arrays)."""
    if t <= kf[0][0]:
        return np.asarray(kf[0][1], np.float64) if not np.isscalar(kf[0][1]) else kf[0][1]
    for (t0, v0), (t1, v1) in zip(kf[:-1], kf[1:]):
        if t <= t1:
            u = ease((t - t0) / (t1 - t0)) if t1 > t0 else 1.0
            if np.isscalar(v0):
                return v0 + (v1 - v0) * u
            return np.asarray(v0, np.float64) + (np.asarray(v1, np.float64) - np.asarray(v0, np.float64)) * u
    v = kf[-1][1]
    return np.asarray(v, np.float64) if not np.isscalar(v) else v


def kick(t, t0, amp=1.0, freq=7.0, decay=5.0):
    """Damped oscillation starting at t0 (seconds) -- for ignition camera energy."""
    if t < t0:
        return 0.0
    u = t - t0
    return amp * math.exp(-decay * u) * math.sin(2 * math.pi * freq * u)


def pulse(t, t0, attack=0.04, decay=0.35):
    """Fast attack, exponential decay envelope (0..1) starting at t0 (seconds)."""
    if t < t0 - attack:
        return 0.0
    if t < t0:
        return smooth((t - (t0 - attack)) / attack)
    return math.exp(-(t - t0) / decay)
