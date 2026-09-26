"""Landscape evolution on a heightfield (numba).

* `priority_flood`  : depression filling with an epsilon gradient (Barnes et al. 2014)
* `stream_power`    : FastScape-style implicit stream-power incision (Braun & Willett 2013),
                      dh/dt = U - K A^m S  (n = 1), with steepest-descent (D8) routing
* `thermal`         : talus relaxation (limits slopes to a critical angle)
* `droplets`        : particle hydraulic erosion (fine gullies)

Grids are float64 [ny, nx], cell size dx (metres). Border cells are base level.
"""
import math

import numpy as np
from numba import njit

_DI = np.array([-1, -1, -1, 0, 0, 1, 1, 1], np.int64)
_DJ = np.array([-1, 0, 1, -1, 1, -1, 0, 1], np.int64)
_DD = np.array([math.sqrt(2), 1, math.sqrt(2), 1, 1, math.sqrt(2), 1, math.sqrt(2)])


# ------------------------------------------------------------------ heap ---
@njit(inline='always', cache=True)
def _hpush(hk, hv, n, k, v):
    i = n
    hk[i] = k
    hv[i] = v
    while i > 0:
        p = (i - 1) >> 1
        if hk[p] <= hk[i]:
            break
        hk[p], hk[i] = hk[i], hk[p]
        hv[p], hv[i] = hv[i], hv[p]
        i = p
    return n + 1


@njit(inline='always', cache=True)
def _hpop(hk, hv, n):
    k = hk[0]
    v = hv[0]
    n -= 1
    hk[0] = hk[n]
    hv[0] = hv[n]
    i = 0
    while True:
        l = 2 * i + 1
        r = l + 1
        m = i
        if l < n and hk[l] < hk[m]:
            m = l
        if r < n and hk[r] < hk[m]:
            m = r
        if m == i:
            break
        hk[m], hk[i] = hk[i], hk[m]
        hv[m], hv[i] = hv[i], hv[m]
        i = m
    return k, v, n


@njit(cache=True)
def priority_flood(h, eps=1e-3):
    """Fill depressions in place so every cell drains to the border."""
    ny, nx = h.shape
    N = ny * nx
    closed = np.zeros(N, np.uint8)
    hk = np.empty(N, np.float64)
    hv = np.empty(N, np.int64)
    pit = np.empty(N, np.int64)
    n = 0
    for j in range(ny):
        for i in range(nx):
            if j == 0 or i == 0 or j == ny - 1 or i == nx - 1:
                c = j * nx + i
                closed[c] = 1
                n = _hpush(hk, hv, n, h[j, i], c)
    p0 = 0
    p1 = 0
    while n > 0 or p1 > p0:
        if p1 > p0:
            c = pit[p0]
            p0 += 1
            if p0 == p1:
                p0 = 0
                p1 = 0
        else:
            _, c, n = _hpop(hk, hv, n)
        cj = c // nx
        ci = c - cj * nx
        hc = h[cj, ci]
        for k in range(8):
            jj = cj + _DI[k]
            ii = ci + _DJ[k]
            if jj < 0 or ii < 0 or jj >= ny or ii >= nx:
                continue
            q = jj * nx + ii
            if closed[q]:
                continue
            closed[q] = 1
            if h[jj, ii] <= hc + eps * _DD[k]:
                h[jj, ii] = hc + eps * _DD[k]
                pit[p1] = q
                p1 += 1
            else:
                n = _hpush(hk, hv, n, h[jj, ii], q)


@njit(cache=True)
def _receivers(h, dx, rec, dist):
    ny, nx = h.shape
    for j in range(ny):
        for i in range(nx):
            c = j * nx + i
            rec[c] = c
            dist[c] = dx
            if j == 0 or i == 0 or j == ny - 1 or i == nx - 1:
                continue
            best = 0.0
            for k in range(8):
                jj = j + _DI[k]
                ii = i + _DJ[k]
                d = _DD[k] * dx
                s = (h[j, i] - h[jj, ii]) / d
                if s > best:
                    best = s
                    rec[c] = jj * nx + ii
                    dist[c] = d


@njit(cache=True)
def _stack(rec, N):
    ndon = np.zeros(N, np.int64)
    for c in range(N):
        if rec[c] != c:
            ndon[rec[c]] += 1
    off = np.zeros(N + 1, np.int64)
    for c in range(N):
        off[c + 1] = off[c] + ndon[c]
    fill = off[:-1].copy()
    don = np.empty(max(off[N], 1), np.int64)
    for c in range(N):
        r = rec[c]
        if r != c:
            don[fill[r]] = c
            fill[r] += 1
    stack = np.empty(N, np.int64)
    work = np.empty(N, np.int64)
    ns = 0
    for c in range(N):
        if rec[c] == c:
            w = 0
            work[w] = c
            w += 1
            while w > 0:
                w -= 1
                q = work[w]
                stack[ns] = q
                ns += 1
                for t in range(off[q], off[q + 1]):
                    work[w] = don[t]
                    w += 1
    return stack


@njit(cache=True)
def _area(rec, stack, cellA, A):
    N = rec.shape[0]
    for c in range(N):
        A[c] = cellA
    for t in range(N - 1, -1, -1):
        c = stack[t]
        r = rec[c]
        if r != c:
            A[r] += A[c]


@njit(cache=True)
def stream_power(h, U, K, m, dt, dx, iters, flood_every=5, Kmap=None):
    """Run `iters` implicit stream-power steps in place. U, K may be arrays or scalars
    via Kmap (array) / K (scalar)."""
    ny, nx = h.shape
    N = ny * nx
    rec = np.empty(N, np.int64)
    dist = np.empty(N, np.float64)
    A = np.empty(N, np.float64)
    hf = h.ravel()
    Uf = U.ravel()
    for it in range(iters):
        # uplift (interior)
        for j in range(1, ny - 1):
            for i in range(1, nx - 1):
                h[j, i] += Uf[j * nx + i] * dt
        if it % flood_every == 0:
            priority_flood(h, 1e-4 * dx)
        _receivers(h, dx, rec, dist)
        stack = _stack(rec, N)
        _area(rec, stack, dx * dx, A)
        for t in range(N):
            c = stack[t]
            r = rec[c]
            if r != c:
                kk = K if Kmap is None else Kmap.ravel()[c]
                f = kk * dt * A[c] ** m / dist[c]
                hf[c] = (hf[c] + f * hf[r]) / (1.0 + f)
    return A.reshape(ny, nx)


@njit(cache=True)
def drainage_area(h, dx):
    ny, nx = h.shape
    N = ny * nx
    rec = np.empty(N, np.int64)
    dist = np.empty(N, np.float64)
    A = np.empty(N, np.float64)
    g = h.copy()
    priority_flood(g, 1e-4 * dx)
    _receivers(g, dx, rec, dist)
    stack = _stack(rec, N)
    _area(rec, stack, dx * dx, A)
    return A.reshape(ny, nx)


@njit(cache=True)
def thermal(h, dx, talus, iters, rate=0.25):
    """Move material downhill where slope exceeds tan(talus)."""
    ny, nx = h.shape
    tmax = math.tan(talus)
    d = np.zeros_like(h)
    for it in range(iters):
        d[:, :] = 0.0
        for j in range(1, ny - 1):
            for i in range(1, nx - 1):
                hc = h[j, i]
                tot = 0.0
                mx = 0.0
                for k in range(8):
                    dd = _DD[k] * dx
                    ex = hc - h[j + _DI[k], i + _DJ[k]] - tmax * dd
                    if ex > 0.0:
                        tot += ex
                        if ex > mx:
                            mx = ex
                if tot <= 0.0:
                    continue
                move = rate * mx
                for k in range(8):
                    dd = _DD[k] * dx
                    ex = hc - h[j + _DI[k], i + _DJ[k]] - tmax * dd
                    if ex > 0.0:
                        q = move * ex / tot
                        d[j + _DI[k], i + _DJ[k]] += q
                        d[j, i] -= q
        for j in range(ny):
            for i in range(nx):
                h[j, i] += d[j, i]


# ------------------------------------------------------------- droplets ---
@njit(inline='always', cache=True)
def _hgrad(h, x, y):
    ny, nx = h.shape
    i = int(x)
    j = int(y)
    u = x - i
    v = y - j
    a = h[j, i]
    b = h[j, i + 1]
    c = h[j + 1, i]
    d = h[j + 1, i + 1]
    gx = (b - a) * (1 - v) + (d - c) * v
    gy = (c - a) * (1 - u) + (d - b) * u
    hh = a * (1 - u) * (1 - v) + b * u * (1 - v) + c * (1 - u) * v + d * u * v
    return hh, gx, gy


@njit(cache=True)
def droplets(h, n, seed, radius=3, inertia=0.05, cap=4.0, mincap=0.01, erode=0.3,
             deposit=0.3, evap=0.01, grav=4.0, steps=80, mask=None):
    """Hans Beyer style particle erosion. Heights in cell units expected to be ~metres/dx
    scaled by caller; returns nothing (in place)."""
    ny, nx = h.shape
    # brush
    R = radius
    bw = np.zeros((2 * R + 1, 2 * R + 1))
    s = 0.0
    for a in range(-R, R + 1):
        for b in range(-R, R + 1):
            w = max(0.0, R - math.sqrt(a * a + b * b))
            bw[a + R, b + R] = w
            s += w
    bw /= s
    np.random.seed(seed)
    for it in range(n):
        x = np.random.random() * (nx - 2 * R - 2) + R + 1
        y = np.random.random() * (ny - 2 * R - 2) + R + 1
        if mask is not None:
            if np.random.random() > mask[int(y), int(x)]:
                continue
        dxv = 0.0
        dyv = 0.0
        sp = 1.0
        wat = 1.0
        sed = 0.0
        for st in range(steps):
            i = int(x)
            j = int(y)
            u = x - i
            v = y - j
            hh, gx, gy = _hgrad(h, x, y)
            dxv = dxv * inertia - gx * (1 - inertia)
            dyv = dyv * inertia - gy * (1 - inertia)
            L = math.sqrt(dxv * dxv + dyv * dyv)
            if L < 1e-12:
                break
            dxv /= L
            dyv /= L
            x += dxv
            y += dyv
            if x < R + 1 or y < R + 1 or x >= nx - R - 2 or y >= ny - R - 2:
                break
            hn, _, _ = _hgrad(h, x, y)
            dh = hn - hh
            c = max(-dh * sp * wat * cap, mincap)
            if sed > c or dh > 0:
                dep = min(dh, sed) if dh > 0 else (sed - c) * deposit
                sed -= dep
                h[j, i] += dep * (1 - u) * (1 - v)
                h[j, i + 1] += dep * u * (1 - v)
                h[j + 1, i] += dep * (1 - u) * v
                h[j + 1, i + 1] += dep * u * v
            else:
                er = min((c - sed) * erode, -dh)
                for a in range(-R, R + 1):
                    for b in range(-R, R + 1):
                        w = bw[a + R, b + R]
                        if w > 0:
                            h[j + a, i + b] -= er * w
                sed += er
            sp = math.sqrt(max(sp * sp - dh * grav, 0.0))
            wat *= (1 - evap)


# -------------------------------------------------------------- sampling ---
@njit(inline='always', cache=True)
def _cub(p0, p1, p2, p3, t):
    return p1 + 0.5 * t * (p2 - p0 + t * (2 * p0 - 5 * p1 + 4 * p2 - p3 + t * (3 * (p1 - p2) + p3 - p0)))


@njit(cache=True)
def bicubic(g, x, y):
    """Catmull-Rom sample of grid g at fractional cell coords (x = column, y = row), clamped."""
    ny, nx = g.shape
    x = min(max(x, 0.0), nx - 1.001)
    y = min(max(y, 0.0), ny - 1.001)
    i = int(x)
    j = int(y)
    u = x - i
    v = y - j
    rows = np.empty(4)
    for a in range(4):
        jj = min(max(j - 1 + a, 0), ny - 1)
        p = np.empty(4)
        for b in range(4):
            ii = min(max(i - 1 + b, 0), nx - 1)
            p[b] = g[jj, ii]
        rows[a] = _cub(p[0], p[1], p[2], p[3], u)
    return _cub(rows[0], rows[1], rows[2], rows[3], v)


@njit(cache=True)
def bilinear(g, x, y):
    ny, nx = g.shape
    x = min(max(x, 0.0), nx - 1.001)
    y = min(max(y, 0.0), ny - 1.001)
    i = int(x)
    j = int(y)
    u = x - i
    v = y - j
    return (g[j, i] * (1 - u) * (1 - v) + g[j, i + 1] * u * (1 - v) + g[j + 1, i] * (1 - u) * v
            + g[j + 1, i + 1] * u * v)
