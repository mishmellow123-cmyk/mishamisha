"""Glider camera: yaw + true pitch + roll, rendered through the column-coherent marcher.

The heightfield marcher needs every screen column to be a vertical plane through the eye (no roll,
tilt as lens shift). A rotation of a pinhole camera about its centre is an exact homography of the
image, so we render a lens-shift SOURCE camera (same eye, same yaw, window chosen to cover the target
frustum) and warp it to the rolled/pitched TARGET camera. World-anchored 2D effects (flames, smoke,
cairns, stars) are drawn in source space, where world verticals are screen verticals, so they roll
with the horizon for free. Then depth-based vector motion blur (McGuire-style tile/neighbour max
reconstruction) in target space, and 3D sparks streaked through the moving camera.
"""
import math

import cv2
import numpy as np
from numba import njit, prange


def _basis(yaw, pitch, roll):
    cy, sy = math.cos(yaw), math.sin(yaw)
    fwd0 = np.array([sy, 0.0, cy])
    right0 = np.array([cy, 0.0, -sy])
    up0 = np.array([0.0, 1.0, 0.0])
    cp, sp = math.cos(pitch), math.sin(pitch)
    fwd = fwd0 * cp + up0 * sp
    up1 = up0 * cp - fwd0 * sp
    cr, sr = math.cos(roll), math.sin(roll)
    right = right0 * cr - up1 * sr          # +roll = bank right (right side of frame dips)
    up = up1 * cr + right0 * sr
    return fwd, right, up


class RCam:
    """Target camera. Angles in degrees; pitch + looks up; roll + banks right."""

    def __init__(self, pos, yaw, pitch, roll, hfov, W, H):
        self.pos = np.asarray(pos, np.float64)
        self.yaw_d, self.pitch_d, self.roll_d, self.hfov_d = yaw, pitch, roll, hfov
        self.W, self.H = int(W), int(H)
        self.f = 0.5 * self.W / math.tan(math.radians(hfov) * 0.5)
        self.cx = 0.5 * self.W
        self.cy = 0.5 * self.H
        self.fwd, self.right, self.up = _basis(math.radians(yaw), math.radians(pitch), math.radians(roll))

    def scaled(self, s):
        return RCam(self.pos, self.yaw_d, self.pitch_d, self.roll_d, self.hfov_d, round(self.W * s),
                    round(self.H * s))

    def project(self, P):
        d = np.asarray(P, np.float64) - self.pos
        x = d @ self.right
        y = d @ self.up
        z = d @ self.fwd
        zz = np.maximum(z, 1e-6)
        return self.cx + self.f * x / zz, self.cy - self.f * y / zz, z

    def project_dir(self, D):
        D = np.asarray(D, np.float64)
        x = D @ self.right
        y = D @ self.up
        z = D @ self.fwd
        zz = np.maximum(z, 1e-9)
        return self.cx + self.f * x / zz, self.cy - self.f * y / zz, z

    def ray(self, u, v):
        """World ray directions (unnormalised) for pixel coordinates u, v (arrays)."""
        return (self.fwd[None] * self.f + self.right[None] * (u - self.cx)[..., None]
                + self.up[None] * (self.cy - v)[..., None])

    def up_screen(self, P, h=1.0):
        """Screen-space angle (radians, 0 = straight up) of the world vertical at P."""
        a = self.project(P)
        b = self.project(np.asarray(P) + np.array([0.0, h, 0.0]))
        return math.atan2(b[0] - a[0], -(b[1] - a[1]))


class SrcCam:
    """Lens-shift camera (horizontal fwd, verticals vertical) with an arbitrary window; duck-types
    montage's Cam for mt.fire / mt.figure / mt.sky."""

    def __init__(self, pos, yaw, f, a0, a1, b0, b1):
        self.pos = np.asarray(pos, np.float64)
        self.yaw = yaw
        self.f = float(f)
        self.fwd = np.array([math.sin(yaw), 0.0, math.cos(yaw)])
        self.right = np.array([math.cos(yaw), 0.0, -math.sin(yaw)])
        self.W = int(math.ceil(a1 - a0))
        self.H = int(math.ceil(b1 - b0))
        self.cx = -a0
        self.cy = 0.5 * self.H
        self.shift = b1 - self.cy
        self.tilt = 0.0

    @property
    def cyy(self):
        return self.cy + self.shift

    def params(self):
        return np.array([self.pos[0], self.pos[1], self.pos[2], self.fwd[0], self.fwd[2], self.right[0],
                         self.right[2], self.f, self.cx, self.cy + self.shift, float(self.W), float(self.H)],
                        np.float64)

    def to_cam(self, P):
        d = np.asarray(P, np.float64) - self.pos
        x = d[..., 0] * self.right[0] + d[..., 2] * self.right[2]
        z = d[..., 0] * self.fwd[0] + d[..., 2] * self.fwd[2]
        return x, d[..., 1], z

    def project(self, P):
        x, y, z = self.to_cam(P)
        zz = np.maximum(z, 1e-6)
        return self.cx + self.f * x / zz, self.cy + self.shift - self.f * y / zz, z

    def ppm(self, depth):
        return self.f / max(depth, 1e-6)


def source_for(tcam, margin=6.0):
    """Source lens-shift camera covering the target frustum (same eye, same yaw, same focal)."""
    W, H = tcam.W, tcam.H
    n = 48
    us = np.r_[np.linspace(0, W, n), np.full(n, W), np.linspace(W, 0, n), np.zeros(n)]
    vs = np.r_[np.zeros(n), np.linspace(0, H, n), np.full(n, H), np.linspace(H, 0, n)]
    d = tcam.ray(us, vs)
    yaw = math.radians(tcam.yaw_d)
    fwd = np.array([math.sin(yaw), 0.0, math.cos(yaw)])
    right = np.array([math.cos(yaw), 0.0, -math.sin(yaw)])
    c = d @ fwd
    if np.any(c <= 1e-3):
        raise ValueError('target frustum not coverable by a lens-shift source')
    a = tcam.f * (d @ right) / c
    b = tcam.f * d[:, 1] / c
    return SrcCam(tcam.pos, yaw, tcam.f, a.min() - margin, a.max() + margin, b.min() - margin,
                  b.max() + margin)


def warp_maps(tcam, scam):
    """cv2.remap maps: for every target pixel, where to sample the source image."""
    v, u = np.mgrid[0:tcam.H, 0:tcam.W].astype(np.float64)
    d = tcam.ray(u + 0.5, v + 0.5)
    c = d @ scam.fwd
    a = scam.f * (d @ scam.right) / c
    b = scam.f * d[..., 1] / c
    mx = (scam.cx + a - 0.5).astype(np.float32)
    my = (scam.cyy - b - 0.5).astype(np.float32)
    return mx, my


def warp(img, mx, my, interp=cv2.INTER_LINEAR):
    return cv2.remap(img, mx, my, interp, borderMode=cv2.BORDER_REPLICATE)


# ----------------------------------------------------------- motion blur ---

def velocity(camfn, frame, tcam, dist, shutter=0.5):
    """Per-pixel screen motion (px over the open shutter) of a static world seen by the camera path.
    camfn(frame) -> RCam at the same resolution. dist: Euclidean distance per target pixel (>=1e8 sky)."""
    H, W = dist.shape
    v, u = np.mgrid[0:H, 0:W].astype(np.float64)
    d = tcam.ray(u + 0.5, v + 0.5)
    d /= np.linalg.norm(d, axis=-1, keepdims=True)
    c0 = camfn(frame - 0.5 * shutter)
    c1 = camfn(frame + 0.5 * shutter)
    sky = dist >= 1e8
    P = tcam.pos[None, None] + d * np.where(sky, 1.0, dist)[..., None]
    x0, y0, z0 = c0.project(P)
    x1, y1, z1 = c1.project(P)
    if sky.any():
        xs0, ys0, _ = c0.project_dir(d[sky])
        xs1, ys1, _ = c1.project_dir(d[sky])
        x0[sky], y0[sky], x1[sky], y1[sky] = xs0, ys0, xs1, ys1
    V = np.stack([x1 - x0, y1 - y0], -1).astype(np.float32)
    bad = (z0 <= 0.05) | (z1 <= 0.05)
    V[bad & ~sky] = 0.0
    return V


@njit(parallel=True, fastmath=True, cache=True)
def _tile_max(V, T, out):
    H, W = V.shape[0], V.shape[1]
    th, tw = out.shape[0], out.shape[1]
    for ty in prange(th):
        for tx in range(tw):
            best = 0.0
            bx = 0.0
            by = 0.0
            for y in range(ty * T, min(H, ty * T + T)):
                for x in range(tx * T, min(W, tx * T + T)):
                    m = V[y, x, 0] * V[y, x, 0] + V[y, x, 1] * V[y, x, 1]
                    if m > best:
                        best = m
                        bx = V[y, x, 0]
                        by = V[y, x, 1]
            out[ty, tx, 0] = bx
            out[ty, tx, 1] = by


@njit(parallel=True, fastmath=True, cache=True)
def _neighbor_max(TM, out):
    th, tw = TM.shape[0], TM.shape[1]
    for ty in prange(th):
        for tx in range(tw):
            best = -1.0
            bx = 0.0
            by = 0.0
            for yy in range(max(0, ty - 1), min(th, ty + 2)):
                for xx in range(max(0, tx - 1), min(tw, tx + 2)):
                    m = TM[yy, xx, 0] ** 2 + TM[yy, xx, 1] ** 2
                    if m > best:
                        best = m
                        bx = TM[yy, xx, 0]
                        by = TM[yy, xx, 1]
            out[ty, tx, 0] = bx
            out[ty, tx, 1] = by


@njit(inline='always', fastmath=True)
def _cone(r, v):
    return min(max(1.0 - r / max(v, 1e-6), 0.0), 1.0)


@njit(inline='always', fastmath=True)
def _cyl(r, v):
    x = (r - 0.95 * v) / max(0.1 * v, 1e-6)
    x = min(max(x, 0.0), 1.0)
    return 1.0 - x * x * (3.0 - 2.0 * x)


@njit(inline='always', fastmath=True)
def _softz(za, zb, ext):
    # 1 when a is in front of b
    return min(max(1.0 + (zb - za) / ext, 0.0), 1.0)


@njit(parallel=True, fastmath=True, cache=True)
def _reconstruct(img, V, Z, NM, T, nsamp, out):
    """McGuire et al. 2012 reconstruction filter. V = full motion over the shutter (px); blur radius = |V|/2."""
    H, W = img.shape[0], img.shape[1]
    for y in prange(H):
        for x in range(W):
            nvx = NM[y // T, x // T, 0] * 0.5
            nvy = NM[y // T, x // T, 1] * 0.5
            nl = math.sqrt(nvx * nvx + nvy * nvy)
            if nl < 0.5:
                out[y, x, 0] = img[y, x, 0]
                out[y, x, 1] = img[y, x, 1]
                out[y, x, 2] = img[y, x, 2]
                continue
            vx = V[y, x, 0] * 0.5
            vy = V[y, x, 1] * 0.5
            vl = max(math.sqrt(vx * vx + vy * vy), 0.5)
            zx = Z[y, x]
            wsum = 1.0 / vl
            cr = img[y, x, 0] * wsum
            cg = img[y, x, 1] * wsum
            cb = img[y, x, 2] * wsum
            # deterministic per-pixel jitter (interleaved gradient noise)
            jit = (52.9829189 * ((0.06711056 * x + 0.00583715 * y) % 1.0)) % 1.0 - 0.5
            for s in range(nsamp):
                if s == (nsamp - 1) // 2:
                    continue
                tt = -1.0 + 2.0 * (s + 0.5 + jit * 0.95) / nsamp
                sx = int(x + 0.5 + nvx * tt)
                sy = int(y + 0.5 + nvy * tt)
                if sx < 0 or sy < 0 or sx >= W or sy >= H:
                    continue
                r = abs(tt) * nl
                zy = Z[sy, sx]
                uvx = V[sy, sx, 0] * 0.5
                uvy = V[sy, sx, 1] * 0.5
                ul = max(math.sqrt(uvx * uvx + uvy * uvy), 0.5)
                ext = 0.02 * min(zx, zy) + 0.5
                f = _softz(zy, zx, ext)          # sample in front of centre
                b = _softz(zx, zy, ext)          # sample behind centre
                w = f * _cone(r, ul) + b * _cone(r, vl) + _cyl(r, ul) * _cyl(r, vl) * 2.0
                cr += img[sy, sx, 0] * w
                cg += img[sy, sx, 1] * w
                cb += img[sy, sx, 2] * w
                wsum += w
            out[y, x, 0] = cr / wsum
            out[y, x, 1] = cg / wsum
            out[y, x, 2] = cb / wsum


def motion_blur(img, V, Z, max_r=48, nsamp=19):
    """img HDR (H,W,3) float32, V (H,W,2) full shutter motion px, Z (H,W) depth (sky large)."""
    T = int(max_r)
    Vc = V.copy()
    m = np.linalg.norm(Vc, axis=-1, keepdims=True)
    lim = 2.0 * max_r
    Vc = np.where(m > lim, Vc * (lim / np.maximum(m, 1e-9)), Vc).astype(np.float32)
    H, W = img.shape[:2]
    th, tw = (H + T - 1) // T, (W + T - 1) // T
    TM = np.zeros((th, tw, 2), np.float32)
    _tile_max(Vc, T, TM)
    NM = np.zeros_like(TM)
    _neighbor_max(TM, NM)
    out = np.empty_like(img)
    _reconstruct(img.astype(np.float32), Vc, np.minimum(Z, 1e7).astype(np.float32), NM, T, int(nsamp), out)
    return out
