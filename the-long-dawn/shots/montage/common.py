"""Shared driver utilities for the MONTAGE shots."""
import math
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'lib'))
sys.path.insert(0, HERE)

import look  # noqa: E402

FPS = 24.0
OUT_DIR = os.path.join(ROOT, 'renders', 'montage')


def ftime(frame):
    """Global time in seconds for a (possibly fractional) global frame."""
    return frame / FPS


def downsample(img, W, H):
    return cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)


def depth_down(dep, W, H):
    """Conservative (nearest) depth for compositing: min over each footprint."""
    h, w = dep.shape
    if (w, h) == (W, H):
        return dep.astype(np.float32)
    k = max(1, int(round(w / W)))
    d = cv2.erode(dep.astype(np.float32), np.ones((k + 1, k + 1), np.uint8))
    return cv2.resize(d, (W, H), interpolation=cv2.INTER_NEAREST)


def lin(hexcol):
    return look.hexrgb(hexcol).astype(np.float64)
