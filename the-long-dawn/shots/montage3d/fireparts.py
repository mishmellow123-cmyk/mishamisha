"""Venv-side fire pieces shared by the MONTAGE-3D shots.

* Timing is the montage toolkit's own (`mt.fire.ignite_env`, `flicker`) so the ignition envelope is
  identical to every other beacon in the film.
* Flame sprites: the toolkit's tongue bonfire (`mt.fire._bonfire`) rendered orthographically into an
  HDR image per frame; Blender maps them onto a camera-facing additive card inside the 3-D scene.
* Sparks: the toolkit's deterministic `Sparks` (integrated from birth each frame, motion-blurred
  streaks), projected through the exact Blender camera and depth-tested against the EXR depth.
"""
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
os.environ.setdefault('NUMBA_CACHE_DIR', os.path.join(HERE, 'cache', 'numba'))
sys.path.insert(0, os.path.join(ROOT, 'lib'))
sys.path.insert(0, os.path.join(ROOT, 'shots', 'montage'))     # read-only: the shared montage toolkit

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from mt import fire as MF  # noqa: E402

FPS = 24.0
ignite_env = MF.ignite_env
flicker = MF.flicker
FIRE_LIGHT = MF.FIRE_LIGHT


def ftime(frame):
    return frame / FPS


# ------------------------------------------------------------------ sprites ---

class FlameSpec:
    """A flame element: base size + envelope. The sprite covers a fixed card (metres) so the card
    never has to change size; the flame grows inside it."""

    def __init__(self, name, Hf, Rb, seed, I=26.0, tongues=5, warp=1.0, lean=0.0, ppm=200.0,
                 max_size=1.5, t_ign=None, env=True, lean_fn=None):
        self.name, self.Hf, self.Rb, self.seed, self.I = name, Hf, Rb, seed, I
        self.tongues, self.warp, self.lean, self.ppm = tongues, warp, lean, ppm
        self.max_size, self.t_ign, self.env = max_size, t_ign, env
        self.lean_fn = lean_fn
        Hm = Hf * max_size
        self.half_w = Rb * 1.5 + abs(lean) * max_size * 1.3 + 0.25 * Hm + 0.05
        self.top = 1.9 * Hm + 0.05
        self.pad = 0.12 * Hm + 0.03
        self.Wpx = int(math.ceil(2 * self.half_w * ppm)) | 1
        self.Hpx = int(math.ceil((self.top + self.pad) * ppm))
        self.TG = MF.tongue_table(tongues, seed)

    def card(self):
        """Card geometry in metres relative to the flame base: (x0, x1, z0, z1)."""
        return (-self.Wpx / 2 / self.ppm, self.Wpx / 2 / self.ppm, -self.pad, self.Hpx / self.ppm - self.pad)

    def params(self, frame):
        t = ftime(frame)
        if self.env and self.t_ign is not None:
            size, inten, light = ignite_env(t, self.t_ign)
        else:
            size, inten, light = 1.0, 1.0, 1.0
        lean = self.lean * size if self.lean_fn is None else self.lean_fn(frame, size)
        return t, self.Hf * size, inten, light, lean

    def render(self, frame):
        t, Hf, inten, light, lean = self.params(frame)
        img = np.zeros((self.Hpx, self.Wpx, 3), np.float32)
        if Hf <= 0.01 or inten <= 0:
            return img
        depth = np.full((self.Hpx, self.Wpx), 1e9, np.float32)
        bx = self.Wpx / 2.0
        by = self.Hpx - self.pad * self.ppm
        MF._bonfire(img, depth, float(bx), float(by), float(self.ppm), 1.0, float(Hf), float(self.Rb),
                    float(lean), float(t), float(self.seed), float(self.I * inten), 0, self.Wpx, 0, self.Hpx,
                    0.0, self.TG, float(self.warp), 0)
        return img


def write_sprites(spec, frames, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    for f in frames:
        p = os.path.join(out_dir, f'{spec.name}_{f:05d}.exr')
        img = spec.render(f)
        cv2.imwrite(p, img[..., ::-1].astype(np.float32))
    return out_dir


# --------------------------------------------------------------- projection ---

class BCam:
    """Projection matching a Blender camera record (see kit.core.camera_record)."""

    def __init__(self, rec, W=None, H=None):
        self.M = np.array(rec['M'], np.float64)
        self.Minv = np.linalg.inv(self.M)
        self.W = W or rec['W']
        self.H = H or rec['H']
        self.fpx = rec['lens'] / rec['sensor'] * self.W
        self.cx = self.W / 2.0 - rec['shift_x'] * self.W
        self.cy = self.H / 2.0 + rec['shift_y'] * self.W
        self.pos = self.M[:3, 3]

    def project(self, P):
        P = np.asarray(P, np.float64)
        q = P @ self.Minv[:3, :3].T + self.Minv[:3, 3]
        z = -q[..., 2]
        zz = np.maximum(z, 1e-6)
        sx = self.cx + self.fpx * q[..., 0] / zz
        sy = self.cy - self.fpx * q[..., 1] / zz
        return sx, sy, z

    def ppm(self, depth):
        return self.fpx / max(depth, 1e-6)


# ------------------------------------------------------------------- sparks ---

class ZSparks:
    """montage toolkit Sparks in a Z-up world (the integrator is y-up: swap y/z in and out)."""

    def __init__(self, seed, origin, t_ign, t_end, wind=(0.8, 0.0, 0.0), **kw):
        o = (origin[0], origin[2], origin[1])
        w = (wind[0], wind[2], wind[1])
        self.S = MF.Sparks(seed, np.array(o, np.float64), t_ign, t_end, wind=w, **kw)

    def render(self, img, depth, cam, t, gain=1.0, shutter=1.0 / 48.0, zbias=0.3, max_len_px=260.0, K=5):
        S = self.S
        pts, alive, age = S.state(t, shutter, K)
        if not alive.any():
            return
        idx = np.nonzero(alive)[0]
        P = pts[idx][..., [0, 2, 1]]
        sx, sy, z = cam.project(P)
        ok = (z > 0.2).all(axis=1)
        idx, sx, sy, z = idx[ok], sx[ok], sy[ok], z[ok]
        if len(idx) == 0:
            return
        a = np.clip(age[idx] / S.life[idx], 0, 1)
        T = S.T0[idx] * (1.0 - 0.72 * a ** 0.9)
        tw = 0.7 + 0.3 * np.sin(S.tw[idx] * (t - S.ts[idx]) + S.ph[idx, 0])
        fade_in = np.clip((t - S.ts[idx]) / 0.04, 0, 1)
        fade_out = np.clip((1.0 - a) / 0.2, 0, 1)
        inten = S.I * gain * S.lum[idx] * (T / 0.9) ** 4.0 * tw * fade_in * fade_out
        zc = z.mean(axis=1)
        s = cam.W / 1920.0
        rad = np.maximum(S.wid[idx] * s, 0.3)
        col = MF.bb_vec(T)
        MF._draw_streaks(img, depth, sx.astype(np.float64), sy.astype(np.float64), zc.astype(np.float64),
                         rad.astype(np.float64), col.astype(np.float64), inten.astype(np.float64), float(zbias),
                         float(max_len_px * s))


def shimmer(img, cam, base, Hf, Rb, t, amp_px=1.2, seed=0.0):
    """Heat haze above a flame (screen-space displacement of what lies behind/above it)."""
    sx, sy, z = cam.project(np.asarray(base, np.float64))
    if z <= 0.1:
        return
    ppm = cam.fpx / z
    H, W = img.shape[:2]
    x0 = int(max(0, sx - 1.6 * Rb * ppm - 0.3 * Hf * ppm))
    x1 = int(min(W, sx + 1.6 * Rb * ppm + 0.3 * Hf * ppm))
    y0 = int(max(0, sy - 3.0 * Hf * ppm))
    y1 = int(min(H, sy - 0.2 * Hf * ppm))
    if x1 - x0 < 4 or y1 - y0 < 4:
        return
    yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
    u = (xx - sx) / (Rb * ppm * 1.6 + 1e-6)
    v = (sy - yy) / (Hf * ppm)
    fade = np.clip(1 - np.abs(u), 0, 1) ** 1.5 * np.clip((v - 0.2) / 0.5, 0, 1) * np.clip((3.0 - v) / 1.2, 0, 1)
    k = 1.0 / max(0.06 * ppm, 1e-3)
    dx = np.sin(xx * 0.9 * k + (yy + t * 2.2 * ppm) * 1.7 * k + seed) * 0.6 + \
        np.sin((yy + t * 2.9 * ppm) * 2.9 * k + t * 3.0) * 0.4
    dy = np.sin((yy + t * 2.4 * ppm) * 2.1 * k + xx * 0.7 * k + 1.3 + seed) * 0.5
    mx = (xx + dx * amp_px * fade).astype(np.float32)
    my = (yy + dy * amp_px * fade).astype(np.float32)
    img[y0:y1, x0:x1] = cv2.remap(img, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def glow(img, depth, cam, P, energy, rad_px, col=None, zbias=2.0):
    """Tiny distant fire as a Gaussian splat of total energy (depth-tested)."""
    sx, sy, z = cam.project(np.asarray(P, np.float64))
    if z <= 0.1:
        return
    MF.glow(img, depth, sx, sy, rad_px, energy, z=z, zbias=zbias, col=col)


def halo(img, depth, cam, P, sig_px, peak, col=None, zbias=0.0, squash=1.0):
    sx, sy, z = cam.project(np.asarray(P, np.float64))
    if z <= 0.1:
        return
    MF.halo(img, depth, sx, sy, sig_px, peak, z=z, zbias=zbias, col=col, squash=squash)
