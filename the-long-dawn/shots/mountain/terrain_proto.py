"""Prototype: tune stream-power parameters at low res and look at a hillshade."""
import math
import os
import sys
import time

import cv2
import numpy as np
from numba import njit, prange

sys.path.insert(0, os.path.dirname(__file__))
from mtn import erosion as E
from mtn.noise import fbm2, ridged2, smoothstep

OUT = sys.argv[1] if len(sys.argv) > 1 else '.'


@njit(parallel=True, cache=True)
def uplift_field(nx, ny, x0, y0, dx, seed):
    U = np.empty((ny, nx))
    for j in prange(ny):
        for i in range(nx):
            x = x0 + i * dx
            y = y0 + j * dx
            # warped low-frequency "range" field
            wx = fbm2(x / 9000.0 + 3.3, y / 9000.0 - 1.1, 3.0, seed + 1)
            wy = fbm2(x / 9000.0 - 7.1, y / 9000.0 + 2.9, 3.0, seed + 2)
            r = ridged2(x / 16000.0 + 0.4 * wx, y / 16000.0 + 0.4 * wy, 3.0, seed + 3, 2.0, 0.5, 1.5)
            m = fbm2(x / 30000.0, y / 30000.0, 2.0, seed + 4)
            u = 0.25 + 0.75 * smoothstep(0.25, 0.85, r) * (0.7 + 0.5 * m)
            u *= 0.85 + 0.3 * fbm2(x / 2500.0, y / 2500.0, 3.0, seed + 5)
            U[j, i] = max(u, 0.05)
    return U


@njit(parallel=True, cache=True)
def init_noise(nx, ny, x0, y0, dx, seed, amp):
    h = np.empty((ny, nx))
    for j in prange(ny):
        for i in range(nx):
            x = x0 + i * dx
            y = y0 + j * dx
            h[j, i] = amp * fbm2(x / 3000.0, y / 3000.0, 6.0, seed + 11)
    return h


def hillshade(h, dx, az=315, alt=35):
    gy, gx = np.gradient(h, dx)
    n = np.stack([-gx, -gy, np.ones_like(h)], -1)
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    a, e = math.radians(az), math.radians(alt)
    L = np.array([math.cos(e) * math.sin(a), math.cos(e) * math.cos(a), math.sin(e)])
    s = np.clip(n @ L, 0, 1)
    return s


if __name__ == '__main__':
    N = int(os.environ.get('N', 1024))
    dx = float(os.environ.get('DX', 40.0))
    iters = int(os.environ.get('IT', 150))
    Kdt = float(os.environ.get('KDT', 0.75))
    Udt = float(os.environ.get('UDT', 15.0))
    t0 = time.time()
    x0 = -N * dx / 2
    y0 = -N * dx / 4
    U = uplift_field(N, N, x0, y0, dx, 7)
    h = init_noise(N, N, x0, y0, dx, 7, 60.0)
    print('fields', time.time() - t0)
    t0 = time.time()
    A = E.stream_power(h, U * Udt, Kdt, 0.5, 1.0, dx, iters, 5, None)
    print('sp', time.time() - t0, h.min(), h.max())
    t0 = time.time()
    E.thermal(h, dx, math.radians(55), 20)
    print('thermal', time.time() - t0)
    np.save(os.path.join(OUT, 'proto_h.npy'), h.astype(np.float32))
    s = hillshade(h, dx)
    hn = (h - h.min()) / (h.max() - h.min())
    img = np.clip(0.75 * s + 0.25 * hn, 0, 1)
    img = (img[::-1] * 255).astype(np.uint8)   # north up
    cv2.imwrite(os.path.join(OUT, 'proto_hs.png'), cv2.resize(img, (1024, 1024)))
    la = np.log10(A)
    cv2.imwrite(os.path.join(OUT, 'proto_U.png'), (cv2.resize(U / U.max(), (512, 512))[::-1] * 255).astype(np.uint8))
