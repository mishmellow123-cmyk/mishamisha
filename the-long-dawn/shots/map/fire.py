"""Screen-space fire primitives (numba): glowing lines, points, and small living flames.

All intensities are linear HDR radiance; colours are linear RGB. Sizes are in screen pixels.
"""
import math

import numpy as np
from numba import njit

from noise import gnoise


@njit(cache=True)
def splat_lines(img, xs, ys, cols, sig, starts, mul):
    """Polylines with a Gaussian cross-profile. cols (N,3) = peak radiance at each vertex,
    sig (N,) = profile sigma in px (clamped >= 0.5 with energy kept), mul (N,) extra gain."""
    H, W = img.shape[0], img.shape[1]
    for s in range(len(starts) - 1):
        for k in range(starts[s], starts[s + 1] - 1):
            ax, ay, bx, by = xs[k], ys[k], xs[k + 1], ys[k + 1]
            sa = sig[k]
            sb = sig[k + 1]
            ga = mul[k]
            gb = mul[k + 1]
            if sa < 0.5:
                ga *= sa / 0.5
                sa = 0.5
            if sb < 0.5:
                gb *= sb / 0.5
                sb = 0.5
            if ga <= 0.0 and gb <= 0.0:
                continue
            smax = max(sa, sb)
            r = 3.0 * smax + 1.0
            x0 = int(math.floor(min(ax, bx) - r))
            x1 = int(math.ceil(max(ax, bx) + r))
            y0 = int(math.floor(min(ay, by) - r))
            y1 = int(math.ceil(max(ay, by) + r))
            if x1 < 0 or y1 < 0 or x0 >= W or y0 >= H:
                continue
            if x1 - x0 > 4000 or y1 - y0 > 4000:
                continue
            x0 = max(x0, 0)
            y0 = max(y0, 0)
            x1 = min(x1, W - 1)
            y1 = min(y1, H - 1)
            dx = bx - ax
            dy = by - ay
            L2 = dx * dx + dy * dy
            for i in range(y0, y1 + 1):
                py = i + 0.5 - ay
                for j in range(x0, x1 + 1):
                    px = j + 0.5 - ax
                    if L2 > 1e-9:
                        t = (px * dx + py * dy) / L2
                        if t < 0.0:
                            t = 0.0
                        elif t > 1.0:
                            t = 1.0
                    else:
                        t = 0.0
                    qx = px - t * dx
                    qy = py - t * dy
                    d2 = qx * qx + qy * qy
                    sg = sa + (sb - sa) * t
                    e = d2 / (2.0 * sg * sg)
                    if e > 9.0:
                        continue
                    w = math.exp(-e) * (ga + (gb - ga) * t)
                    # joints: segments overlap at vertices; take the max, not the sum
                    for c in range(3):
                        v = w * (cols[k, c] + (cols[k + 1, c] - cols[k, c]) * t)
                        if v > img[i, j, c]:
                            img[i, j, c] = v


@njit(cache=True)
def add_lines(img, xs, ys, cols, sig, starts, mul):
    """Like splat_lines, but additive over separate polylines (max within one polyline)."""
    H, W = img.shape[0], img.shape[1]
    tmp = np.zeros((H, W, 3), np.float32)
    for s in range(len(starts) - 1):
        a = starts[s]
        b = starts[s + 1]
        if b - a < 2:
            continue
        # bbox of this polyline
        xmn = 1e9
        xmx = -1e9
        ymn = 1e9
        ymx = -1e9
        smx = 0.5
        for k in range(a, b):
            xmn = min(xmn, xs[k])
            xmx = max(xmx, xs[k])
            ymn = min(ymn, ys[k])
            ymx = max(ymx, ys[k])
            smx = max(smx, sig[k])
        r = 3.0 * smx + 2.0
        x0 = max(int(math.floor(xmn - r)), 0)
        x1 = min(int(math.ceil(xmx + r)), W - 1)
        y0 = max(int(math.floor(ymn - r)), 0)
        y1 = min(int(math.ceil(ymx + r)), H - 1)
        if x1 < x0 or y1 < y0:
            continue
        st = np.zeros(2, np.int64)
        st[1] = b - a
        sub = tmp[y0:y1 + 1, x0:x1 + 1]
        sub[:] = 0.0
        splat_lines(sub, xs[a:b] - x0, ys[a:b] - y0, cols[a:b], sig[a:b], st, mul[a:b])
        img[y0:y1 + 1, x0:x1 + 1] += sub


@njit(cache=True)
def splat_points(img, xs, ys, cols, sig):
    """Gaussian dots; cols = PEAK radiance (sigma clamped >= 0.5 with energy kept)."""
    H, W = img.shape[0], img.shape[1]
    for k in range(len(xs)):
        s = sig[k]
        g = 1.0
        if s < 0.5:
            g = (s / 0.5) ** 2
            s = 0.5
        r = 3.0 * s + 1.0
        x0 = max(int(math.floor(xs[k] - r)), 0)
        x1 = min(int(math.ceil(xs[k] + r)), W - 1)
        y0 = max(int(math.floor(ys[k] - r)), 0)
        y1 = min(int(math.ceil(ys[k] + r)), H - 1)
        for i in range(y0, y1 + 1):
            dy = i + 0.5 - ys[k]
            for j in range(x0, x1 + 1):
                dx = j + 0.5 - xs[k]
                e = (dx * dx + dy * dy) / (2.0 * s * s)
                if e > 9.0:
                    continue
                w = math.exp(-e) * g
                img[i, j, 0] += w * cols[k, 0]
                img[i, j, 1] += w * cols[k, 1]
                img[i, j, 2] += w * cols[k, 2]


@njit(cache=True)
def _ramp(T):
    """Flame colour for temperature T in [0, 1] (linear RGB, max channel 1)."""
    if T < 0.0:
        T = 0.0
    if T < 0.25:
        a = T / 0.25
        return 0.55 + 0.45 * a, 0.06 + 0.12 * a, 0.01 + 0.01 * a
    if T < 0.5:
        a = (T - 0.25) / 0.25
        return 1.0, 0.18 + 0.24 * a, 0.02 + 0.05 * a
    if T < 0.78:
        a = (T - 0.5) / 0.28
        return 1.0, 0.42 + 0.28 * a, 0.07 + 0.16 * a
    a = min((T - 0.78) / 0.22, 1.0)
    return 1.0, 0.7 + 0.18 * a, 0.23 + 0.3 * a


@njit(cache=True)
def flame(img, bx, by, h, ux, uy, t, seed, gain, wide):
    """A small living flame standing at screen (bx, by): height h px, up direction (ux, uy) (unit,
    screen coords), time t (frames), brightness gain; wide scales the girth. The body is a warped
    teardrop whose upper part breaks into licking tongues; colour runs from a yellow heart through
    orange to a red, fraying tip."""
    H, W = img.shape[0], img.shape[1]
    if h < 0.3:
        return
    wmax = 0.2 * h * wide
    r = 1.3 * h + wmax + 3.0
    x0 = max(int(math.floor(bx - r)), 0)
    x1 = min(int(math.ceil(bx + r)), W - 1)
    y0 = max(int(math.floor(by - r)), 0)
    y1 = min(int(math.ceil(by + r)), H - 1)
    rx = -uy
    ry = ux
    ph = (seed % 997) * 0.37
    sway = 0.12 * math.sin(t * 0.21 + ph) + 0.06 * math.sin(t * 0.57 + 2.0 * ph)
    stretch = 1.0 + 0.12 * math.sin(t * 0.33 + 1.3 * ph) + 0.08 * gnoise(t * 0.17, ph, 71)
    hh = h * stretch
    aa = max(0.55, 0.03 * h)
    for i in range(y0, y1 + 1):
        for j in range(x0, x1 + 1):
            dx = j + 0.5 - bx
            dy = i + 0.5 - by
            v = (dx * ux + dy * uy) / hh
            if v < -0.14 or v > 1.3:
                continue
            u = (dx * rx + dy * ry) / max(wmax, 0.5)          # across, in half-widths
            vv = min(max(v, 0.0), 1.2)
            # warp: the body leans and licks, more toward the tip
            wu = u - (sway * vv * vv * hh / max(wmax, 0.5)) \
                - 0.55 * vv * vv * gnoise(vv * 2.2 - t * 0.33, ph, 73) \
                - 0.25 * vv * gnoise(vv * 5.0 - t * 0.6, ph + 3.0, 74)
            if v < 0.0:
                rad = math.sqrt(max(1.0 - (v / 0.14) ** 2, 0.0)) * 0.66
            else:
                rad = (0.66 + 1.0 * vv) * max(1.0 - vv, 0.0) ** 0.95
            # the upper third frays into tongues
            n1 = gnoise(wu * 1.7 + ph, vv * 3.4 - t * 0.55, 77)
            n2 = gnoise(wu * 3.6 - ph, vv * 6.5 - t * 0.9, 78)
            rad *= 1.0 + (0.7 * n1 + 0.35 * n2) * vv
            tip = vv - 0.55 - 0.45 * n1 - 0.2 * n2
            d = (rad - abs(wu)) * max(wmax, 0.5) / aa
            if d <= -2.5:
                continue
            cov = 1.0 / (1.0 + math.exp(-2.0 * d))
            if tip > 0.0:
                cov *= max(0.0, 1.0 - tip * 3.2)
            if cov <= 0.002:
                continue
            core = max(0.0, 1.0 - abs(wu) / max(rad, 1e-3))
            # a yellow heart low in the body; orange flanks; a red, fraying top
            hc = min(max(core / 0.75, 0.0), 1.0)
            heart = hc * hc * (3.0 - 2.0 * hc) * math.exp(-((vv - 0.3) / 0.34) ** 2)
            T = 0.16 + 0.66 * heart + 0.2 * core * (1.0 - vv) - 0.2 * vv
            if v < 0.0:
                T *= 1.0 + 1.5 * v          # the root of the flame is a little cooler
            r_, g_, b_ = _ramp(T)
            I = gain * cov * (0.3 + 1.9 * T * T * T)
            img[i, j, 0] += I * r_
            img[i, j, 1] += I * g_
            img[i, j, 2] += I * b_
