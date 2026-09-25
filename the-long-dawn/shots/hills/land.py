"""Layered ridge 'cards' at real distances: analytic AA edges, aerial perspective,
valley mist, beacon light on the hills, depth buffer for occlusion."""
import math

import numba as nb
import numpy as np

from core import cam_ray, fbm1_np, fbm2
from sky import sky_rad


class Ridge:
    def __init__(self, z, x0, x1, heights, albedo, fog_mul=1.0, rim=0.0, mist=1.0, tex=0.0, name='',
                 fog_el=0.035, mist_scale=0.0, seed=0.0, snow=0.0):
        self.z = float(z)
        self.x0 = float(x0)
        self.x1 = float(x1)
        self.h = np.asarray(heights, np.float64)
        self.dx = (self.x1 - self.x0) / (len(self.h) - 1)
        self.albedo = np.asarray(albedo, np.float64)
        self.fog_mul = fog_mul
        self.rim = rim
        self.mist = mist
        self.tex = tex
        self.name = name
        self.fog_el = fog_el
        self.mist_scale = mist_scale
        self.seed = seed
        self.snow = snow
        self.snow = snow

    def height(self, X):
        return np.interp(X, np.linspace(self.x0, self.x1, len(self.h)), self.h)

    def peaks(self, xmin, xmax, min_sep):
        """Local maxima positions (for placing beacons)."""
        xs = np.linspace(self.x0, self.x1, len(self.h))
        h = self.h
        idx = np.nonzero((h[1:-1] > h[:-2]) & (h[1:-1] >= h[2:]))[0] + 1
        idx = [i for i in idx if xmin <= xs[i] <= xmax]
        idx.sort(key=lambda i: -h[i])
        chosen = []
        for i in idx:
            if all(abs(xs[i] - xs[j]) > min_sep for j in chosen):
                chosen.append(i)
        return sorted([(xs[i], h[i]) for i in chosen])


def make_profile(x0, x1, n, base, amp, scale, seed, octaves=6, ridged=0.0, gain=0.5,
                 trees=None, rng=None):
    """1-D terrain profile. ridged in [0,1] mixes in sharp crests."""
    X = np.linspace(x0, x1, n)
    u = X / scale
    a = fbm1_np(u, octaves, 2.0, gain, seed)
    if ridged > 0:
        r = 1.0 - np.abs(fbm1_np(u * 0.7 + 11.3, octaves, 2.0, gain, seed + 5))
        r = r * r
        a = a * (1 - ridged) + (r - 0.45) * ridged * 1.6
    h = base + amp * a
    if trees is not None:
        # trees = (density per metre, height_min, height_max, width_frac, kind)
        dens, hmin, hmax, wfrac, kind, patch_scale = trees
        rng = rng or np.random.default_rng(seed + 99)
        L = x1 - x0
        cnt = int(L * dens)
        tx = rng.uniform(x0, x1, cnt)
        # forest patches
        patch = fbm1_np(tx / patch_scale + 3.3, 3, 2.0, 0.5, seed + 13)
        tx = tx[patch > 0.05]
        th = rng.uniform(hmin, hmax, len(tx))
        dxs = (x1 - x0) / (n - 1)
        ground = h.copy()
        for x, hh in zip(tx, th):
            w = hh * wfrac
            i0 = max(0, int((x - w - x0) / dxs))
            i1 = min(n - 1, int((x + w - x0) / dxs) + 1)
            if i1 <= i0:
                continue
            base_h = np.interp(x, X, ground)
            xx = X[i0:i1 + 1]
            r = np.abs(xx - x) / w
            if kind == 'conifer':
                # narrow spire with a hint of tiers
                prof = np.clip(1 - r, 0, 1) ** 1.25
                prof *= 1.0 + 0.10 * np.sin((xx - x) / w * 9.0 + x) * (r > 0.2)
            else:  # broadleaf: rounded crown
                prof = np.sqrt(np.clip(1 - r * r, 0, 1)) * 0.8 + 0.2 * np.clip(1 - r, 0, 1)
            h[i0:i1 + 1] = np.maximum(h[i0:i1 + 1], base_h - 0.15 * hh + hh * prof)
    return h


def pack_ridges(ridges):
    """Sort far->near and pack into arrays for the kernel."""
    rs = sorted(ridges, key=lambda r: -r.z)
    meta = np.zeros((len(rs), 16), np.float64)
    hs = []
    off = 0
    for i, r in enumerate(rs):
        meta[i, 0] = r.z
        meta[i, 1] = r.x0
        meta[i, 2] = r.dx
        meta[i, 3] = len(r.h)
        meta[i, 4] = off
        meta[i, 5:8] = r.albedo
        meta[i, 8] = r.fog_mul
        meta[i, 9] = r.rim
        meta[i, 10] = r.mist
        meta[i, 11] = r.tex
        meta[i, 12] = r.fog_el
        meta[i, 13] = r.mist_scale
        meta[i, 14] = r.seed
        meta[i, 15] = r.snow
        meta[i, 15] = r.snow
        hs.append(r.h)
        off += len(r.h)
    return rs, meta, np.concatenate(hs)


@nb.njit(cache=True, fastmath=True)
def render_ridges(out_rgb, out_a, depth, cam, meta, hs, sp, fog, lights, light_off, ambient, t):
    """Front-to-back composite of ridge cards.
    fog = [sigma0, sigma1, mist_y, mist_h, mist_gain, fog_lift_el, crest_rim_m, amb_top, tex_scale]
    lights: (N, 8) = x, y, z, radius, r, g, b, layer_index ; light_off unused (all lights scanned
    per layer by index).
    out_rgb premultiplied, out_a alpha, depth = distance of first >50% cover."""
    H = out_rgb.shape[0]
    W = out_rgb.shape[1]
    nl = meta.shape[0]
    nlights = lights.shape[0]
    for j in range(H):
        for i in range(W):
            dx, dy, dz = cam_ray(cam, i + 0.5, j + 0.5)
            accr = 0.0
            accg = 0.0
            accb = 0.0
            A = 0.0
            dep = 1e9
            hx = dx
            hz = dz
            hn = math.sqrt(hx * hx + hz * hz) + 1e-9
            last_el = -1.0
            fr = 0.0
            fg = 0.0
            fb = 0.0
            for li in range(nl - 1, -1, -1):      # nearest first
                # fog colour: sky near the horizon along this azimuth (per-layer elevation)
                el_f = meta[li, 12]
                if el_f != last_el:
                    fr, fg, fb = sky_rad(hx / hn * math.cos(el_f), math.sin(el_f), hz / hn * math.cos(el_f), sp)
                    last_el = el_f
                Z = meta[li, 0]
                if dz <= 1e-6:
                    continue
                tt = (Z - cam[2]) / dz
                if tt <= 0:
                    continue
                X = cam[0] + tt * dx
                Y = cam[1] + tt * dy
                u = (X - meta[li, 1]) / meta[li, 2]
                n = int(meta[li, 3])
                if u < 0 or u >= n - 1:
                    continue
                k = int(u)
                fu = u - k
                off = int(meta[li, 4])
                h0 = hs[off + k]
                h1 = hs[off + k + 1]
                h = h0 + (h1 - h0) * fu
                slope = (h1 - h0) / meta[li, 2]
                pix = tt / cam[12]                   # metres per pixel at this depth
                cov = 0.5 + (h - Y) / (pix * math.sqrt(1.0 + slope * slope))
                if cov <= 0:
                    continue
                if cov > 1:
                    cov = 1.0
                below = max(0.0, h - Y)              # metres below the crest
                # base shading: albedo * sky ambient (brighter near crest, slight texture)
                amb = ambient * (0.55 + 0.45 * math.exp(-below / (fog[7] + 1e-6)))
                if meta[li, 11] > 0:
                    tx = fbm2(X * fog[8] / (1.0 + 0.002 * tt), Y * fog[8] * 2.0 / (1.0 + 0.002 * tt), 4, 2.0, 0.55)
                    amb *= 1.0 + meta[li, 11] * tx
                cr = meta[li, 5] * amb
                cg = meta[li, 6] * amb
                cb = meta[li, 7] * amb
                if meta[li, 15] > 0.0:
                    # moonlit snow face: lit where the smoothed ridge rises toward +x
                    # (flank faces the moon on the left), couloir streaks down the fall line
                    Dd = 45.0 * meta[li, 2]
                    ul = (X - Dd - meta[li, 1]) / meta[li, 2]
                    ur = (X + Dd - meta[li, 1]) / meta[li, 2]
                    kl = min(max(int(ul), 0), n - 1)
                    kr = min(max(int(ur), 0), n - 1)
                    sl = (hs[off + kr] - hs[off + kl]) / (2.0 * Dd)
                    lit = 0.35 + 1.6 * sl * fog[12]
                    if lit < 0.0:
                        lit = 0.0
                    if lit > 1.0:
                        lit = 1.0
                    sc = 1.0 / (0.004 * tt + 0.3)
                    st = fbm2(X * sc * 0.9 + meta[li, 14], (Y + 0.35 * below) * sc * 0.22, 4, 2.0, 0.55)
                    snowv = 0.55 + 0.9 * st
                    if snowv < 0.08:
                        snowv = 0.08
                    if snowv > 1.0:
                        snowv = 1.0
                    lit *= 0.35 + 0.65 * snowv
                    lit *= 0.55 + 0.45 * math.exp(-below / (0.5 * pix * 60.0 + 0.4 * (tt * 0.02)))
                    sn = meta[li, 15]
                    cr += sn * (fog[9] * lit + 0.10 * fog[9] * snowv)
                    cg += sn * (fog[10] * lit + 0.10 * fog[10] * snowv)
                    cb += sn * (fog[11] * lit + 0.10 * fog[11] * snowv)
                # crest rim (sky light grazing the crest)
                rim = meta[li, 9] * math.exp(-below / (fog[6] * pix))
                cr += rim * fr
                cg += rim * fg
                cb += rim * fb
                # fire light on this layer
                for q in range(nlights):
                    if int(lights[q, 7]) != li:
                        continue
                    ddx = X - lights[q, 0]
                    ddy = Y - lights[q, 1]
                    rr = lights[q, 3]
                    d2 = (ddx * ddx + ddy * ddy) / (rr * rr)
                    if d2 < 9.0:
                        w = math.exp(-d2) * (1.0 if ddy <= 0 else math.exp(-ddy / rr * 1.5))
                        cr += lights[q, 4] * w
                        cg += lights[q, 5] * w
                        cb += lights[q, 6] * w
                # aerial perspective + valley mist
                me = math.exp(-max(0.0, Y - fog[2]) / fog[3])
                if meta[li, 13] > 0:
                    mn = fbm2(X * meta[li, 13] + meta[li, 14], 0.37 * meta[li, 14] + tt * 0.0001 + t * 0.002, 3, 2.0, 0.5)
                    me *= max(0.0, 0.55 + 1.5 * mn)
                sig = fog[0] + fog[1] * me
                fa = 1.0 - math.exp(-tt * sig * meta[li, 8])
                mistb = 1.0 + fog[4] * meta[li, 10] * me
                cr = cr * (1 - fa) + fr * mistb * fa
                cg = cg * (1 - fa) + fg * mistb * fa
                cb = cb * (1 - fa) + fb * mistb * fa
                w = (1.0 - A) * cov
                accr += w * cr
                accg += w * cg
                accb += w * cb
                A += w
                if A > 0.5 and dep > 1e8:
                    dep = tt
                if A > 0.999:
                    break
            out_rgb[j, i, 0] = accr
            out_rgb[j, i, 1] = accg
            out_rgb[j, i, 2] = accb
            out_a[j, i] = A
            depth[j, i] = dep


@nb.njit(cache=True, fastmath=True)
def splat_occluded(img, depth, x, y, z, sigma, r, g, b):
    """Energy-normalised gaussian, suppressed where depth buffer is nearer than z."""
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
    for jj in range(iy - rad, iy + rad + 1):
        ddy = jj + 0.5 - y
        for ii in range(ix - rad, ix + rad + 1):
            ddx = ii + 0.5 - x
            tot += math.exp(-(ddx * ddx + ddy * ddy) * inv)
    k = 1.0 / tot
    for jj in range(max(0, iy - rad), min(H, iy + rad + 1)):
        ddy = jj + 0.5 - y
        for ii in range(max(0, ix - rad), min(W, ix + rad + 1)):
            if depth[jj, ii] < z * 0.985:
                continue
            ddx = ii + 0.5 - x
            w = math.exp(-(ddx * ddx + ddy * ddy) * inv) * k
            img[jj, ii, 0] += r * w
            img[jj, ii, 1] += g * w
            img[jj, ii, 2] += b * w
