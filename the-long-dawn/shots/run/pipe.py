"""Frame pipeline shared by the Run and Dawn C: source render -> 2D fx in source space -> warp to the
target camera -> downsample -> motion blur -> 3D particles -> finish."""
import math
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import world as WD          # noqa: E402
import rcam as RC           # noqa: E402

CM = WD.CM
look = CM.look
FPS = 24.0


def ftime(frame):
    return frame / FPS


class Frame:
    """Everything one frame needs: target cam (supersampled), source cam, buffers."""

    def __init__(self, tcam_out, ss):
        self.t_out = tcam_out                    # target camera at output resolution
        self.ss = ss
        self.t_ss = tcam_out.scaled(ss)          # target camera at supersampled resolution
        self.src = RC.source_for(self.t_ss)
        self.img = None
        self.zb = None
        self.dist = None


def render_terrain(fr, frame, CR, light, LT, dmax=90000.0, PL=None):
    scam = fr.src
    C = scam.params()
    P = np.array([scam.pos[0], scam.pos[2], ftime(frame), 0.0])
    D = np.zeros((scam.H, scam.W))
    WD.march(P, CR, C, 0.2, dmax, 0.0035, 0.35, 700.0, 9, D)
    Lk, amb, S, fogp, Q = light
    out = np.zeros((scam.H, scam.W, 3), np.float32)
    zb = np.zeros((scam.H, scam.W), np.float32)
    di = np.zeros((scam.H, scam.W), np.float32)
    LT = np.asarray(LT, np.float64).reshape(-1, 8)
    PL = np.zeros((0, 4)) if PL is None else np.asarray(PL, np.float64).reshape(-1, 4)
    WD.shade(C, D, P, CR, S, LT, Lk, Q, amb, fogp, out, zb, di, PL)
    fr.img, fr.zb, fr.dist = out, zb, di
    return fr


def to_target(fr):
    """Warp the source buffers to the (supersampled) target camera, then downsample to output res.
    Returns img (H,W,3), zbuf (conservative min, H,W), dist (H,W, nearest)."""
    mx, my = RC.warp_maps(fr.t_ss, fr.src)
    img = RC.warp(fr.img, mx, my, cv2.INTER_LINEAR)
    zb = RC.warp(fr.zb, mx, my, cv2.INTER_NEAREST)
    di = RC.warp(fr.dist, mx, my, cv2.INTER_NEAREST)
    W, H = fr.t_out.W, fr.t_out.H
    img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
    k = max(1, int(round(fr.ss)))
    zmin = cv2.erode(zb, np.ones((k + 1, k + 1), np.uint8))
    zb = cv2.resize(zmin, (W, H), interpolation=cv2.INTER_NEAREST)
    di = cv2.resize(cv2.erode(di, np.ones((k + 1, k + 1), np.uint8)), (W, H), interpolation=cv2.INTER_NEAREST)
    return img, zb, di


def src_scale(fr):
    """Pixel scale of the source image relative to a 1920-wide frame at the target's focal length."""
    return fr.src.f / (0.5 * 1920 / math.tan(math.radians(fr.t_out.hfov_d) * 0.5))
