"""FALSE DAWN (cut A, R1). 480 frames: A cut frames 80-559 (renders/falsedawn_A/, EDIT-v3's convention).

High on a ridge at midnight, moonless: the Milky Way above, the cloud sea below. Beyond the far ranges a cold
white glow swells under the horizon, at the azimuth where the true morning will come (the world's sunrise
azimuth, -30 deg), silvers the undersides of a high cloud deck and puts out the stars near it. A slow push.
It must read as morning come too early, never as a city glow (amber, a dome, lit from above) or a moonrise (one
bright point about to break). Three designs of the glow itself, all sharing the deck, the star extinction and
the terrain lit from below the horizon:
  arc   a wide dawn arch with shadow rays fanning up from the far peaks (a sun-like source below the horizon)
  cone  a tall leaning pyramid of pale light (the zodiacal light, the astronomer's own "false dawn")
  veil  no visible source: only the deck lit from beneath across the sky, and a thin white line on the skyline

  python falsedawn.py --frames 500 --scale 0.5 --out stills          # the peak (cut 500 = shot 420), a test still
  python falsedawn.py --range 80-559 --procs 4 --skip                # finals, A cut frames -> renders/falsedawn_A/
"""
import argparse
import math
import os
import sys
import time

import numpy as np
from numba import njit, prange

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ.setdefault('NUMBA_NUM_THREADS', '1')

import world as WD          # noqa: E402
import rcam as RC           # noqa: E402
import pipe as PI           # noqa: E402
from mt import sky as SK    # noqa: E402
from mt.noise import fbm2   # noqa: E402

FPS = 24.0
NFR = 480
lin = PI.CM.lin
R_EARTH = WD.R_EARTH
GLOW_AZ = -30.0
EL_S = -3.2                                   # the source's depth under the horizon (deg)
CAM0 = np.array([60.0, 168.0, 480.0])
YAW = -33.0
PUSH = 34.0                                   # metres forward over the shot
DESIGNS = ('arc', 'cone', 'veil')


# ------------------------------------------------------------------ A's landform: one great knife-edge range ---
# The far islands are a field of uniform needles; one big range stands across the view 5-6 km out, a sawtooth
# of knife-edge crests and faceted spires, and the false dawn rises behind it (its notches cast the rays).
# A-only: these rows are added to this shot's terrain table, never to the shared Run world (R3 in cut A should add
# FD_RANGE too if its last frames look this way).
RANGE_YAW = -45.0                                # left of the glow (az -30): the glow rises behind its right end
RANGE_D = 9500.0
RANGE_KNOTS = [(-1900.0, 60.0), (-1600.0, 250.0), (-1380.0, 205.0), (-1150.0, 330.0), (-950.0, 285.0),
               (-700.0, 420.0), (-520.0, 360.0), (-300.0, 470.0), (-120.0, 400.0), (80.0, 445.0), (300.0, 350.0),
               (520.0, 395.0), (760.0, 300.0), (1000.0, 340.0), (1250.0, 230.0), (1500.0, 270.0), (1800.0, 80.0)]


def build_range():
    ya = math.radians(RANGE_YAW)
    dirv = np.array([math.sin(ya), math.cos(ya)])
    perp = np.array([math.cos(ya), -math.sin(ya)])
    c = CAM0[[0, 2]] + dirv * RANGE_D
    rows = []
    pts = []
    rng = np.random.default_rng(88)
    for o, y in RANGE_KNOTS:
        q = c + perp * o + dirv * rng.uniform(-150.0, 150.0)
        pts.append((q[0], y, q[1]))
    for k in range(len(pts) - 1):
        a, b = pts[k], pts[k + 1]
        # flanks a little under 60 deg: steeper, the heightfield shows vertical stripes (a terrain tell)
        rows.append(WD.ridge_row(a, b, wl=44.0, wr=40.0, seed=101 + k, k=12.0, detail=0.20, slope=1.7))
        rows[-1][14] = 0.95                # a safe early-out bound (world.ridge, v3 flag)
    hi = [k for k in range(1, len(pts) - 1) if pts[k][1] > max(pts[k - 1][1], pts[k + 1][1])]
    main = max(hi, key=lambda k: pts[k][1])
    for k in hi:
        if k != main and rng.random() < 0.35:
            continue                       # not every high point grows a spire: no crown of equal teeth
        big = k == main
        # faceted spires of unequal height, width and lean; one dominant, asymmetric horn is the focal landform
        rows.append(WD.crag_row(pts[k][0] + rng.uniform(-60.0, 60.0), pts[k][2] + rng.uniform(-60.0, 60.0),
                                pts[k][1] + (150.0 if big else rng.uniform(15.0, 70.0)),
                                L=(150.0 if big else rng.uniform(25.0, 60.0)), s_hi=(2.2 if big else rng.uniform(2.4, 3.6)),
                                s_lo=1.7, aniso=(2.3 if big else rng.uniform(1.2, 2.2)),
                                ang=ya + math.pi / 2 + rng.uniform(-0.6, 0.6), seed=140 + k, k=(30.0 if big else 14.0),
                                detail=0.36, shelf=0.0, nf=(4 if big else 3)))
        rows[-1][14] = 0.95                # a safe early-out bound for big detailed crags (world.crag, v3 flag)
        # crag_row's cutoff radius assumes an isotropic falloff: along an elongated crag's long axis the flank is
        # still ~200 m above the cloud where it is cut, which stands as a wall (drawn as vertical stripes). A
        # 2.5x radius lets every flank reach the cloud first.
        rows[-1][13] *= 2.5
    return np.array(rows), np.array(pts)


FD_RANGE, RANGE_PTS = build_range()

# The great far wall (H5, director 27 Sep ~19:10Z: "the uniform needle horizon: give it big landforms and aerial
# depth"). A second knife-edge range stands right of the glow, 46 km out, beyond her summit (32 km at az -10), so
# her fires in A14 stand in front of it. Its crests (world y 1300-2300 m) rise well over the uniform far needles,
# pale with haze. The sierra at 9.5 km is dark, the needles are between, and the wall is palest: three veils
# instead of one sawtooth. It is A-only and lives in this shot's terrain table (terrain_rows), which X2, A14 and
# A15 share.
WALL_YAW = -15.0                                 # centre azimuth from CAM0 (the glow is at -30)
WALL_D = 46000.0
WALL_HALF = 8600.0                               # half its length (m): about +-10.5 deg from CAM0
WALL_N = 49                                      # crest knots (~360 m apart): a fractal skyline, not straight slopes
# (v2, 20:00Z) crags removed: at 46 km a steep faceted horn reads as a flat-topped tower (battlements)


def _wall_profile(rng):
    """Crest heights along the wall (world y): three unequal massifs on a low base, then midpoint displacement
    (roughness 0.55) so the skyline is broken at every scale; no perfect triangle, no flat run."""
    o = np.linspace(-WALL_HALF, WALL_HALF, WALL_N)
    base = 650.0 + 1150.0 * np.exp(-((o + 3600.0) / 1900.0) ** 2) + 1550.0 * np.exp(-((o - 700.0) / 1500.0) ** 2) \
        + 950.0 * np.exp(-((o - 4700.0) / 1300.0) ** 2)
    base *= 0.72 + 0.28 * np.clip(1.0 - (np.abs(o) / WALL_HALF) ** 3, 0.0, 1.0)      # the ends fall into the cloud
    n = WALL_N - 1
    d = np.zeros(WALL_N)
    step, amp = n, 420.0
    while step > 1:
        half = step // 2
        for i in range(half, n, step):
            d[i] = 0.5 * (d[i - half] + d[i + half]) + rng.normal(0.0, amp)
        step, amp = half, amp * 0.55
    return o, base + d


def build_wall():
    ya = math.radians(WALL_YAW)
    dirv = np.array([math.sin(ya), math.cos(ya)])
    perp = np.array([math.cos(ya), -math.sin(ya)])
    c = CAM0[[0, 2]] + dirv * WALL_D
    rng = np.random.default_rng(311)
    o, hy = _wall_profile(rng)
    # the crest wanders in depth (low frequency), so its faces turn to the light at different angles
    dep = 700.0 * np.sin(o / 2300.0 + 0.7) + 380.0 * np.sin(o / 870.0 + 2.1)
    pts = [(c[0] + perp[0] * oo + dirv[0] * dd, h, c[1] + perp[1] * oo + dirv[1] * dd) for oo, h, dd in zip(o, hy, dep)]
    rows = []
    for k in range(len(pts) - 1):
        a, b = pts[k], pts[k + 1]
        w = rng.uniform(300.0, 380.0)
        _ = rng.uniform(0.8, 1.1)                # (kept: the rng stream, so every other draw is unchanged)
        # wl == wr: world.ridge picks the flank width by side, so unequal widths make the height jump along each
        # segment's extension line beyond its end cap, which stands as a sheer cliff (it read as a building)
        rows.append(WD.ridge_row(a, b, wl=w, wr=w, seed=331 + k, k=60.0, detail=0.22,
                                 slope=rng.uniform(1.10, 1.45)))
        rows[-1][14] = 0.95
        rows[-1][13] = 4200.0                    # the flanks reach the cloud sea (~1.3 km out) before the cut
    return np.array(rows)


FD_WALL = build_wall()
_CRF = None


def terrain_rows():
    global _CRF
    if _CRF is None:
        import run as RN
        _CRF = np.vstack([RN.CR, FD_RANGE, FD_WALL])
    return _CRF


def terrain_hmax(CR=None):
    """Conservative world-height ceiling for A's terrain, before Earth curvature.

    The far wall's current ridge endpoints reach 2332.665 m; the old 700 m marcher
    cutoff can stop an upward ray before it reaches them. Bound the primitives,
    including their displacement and smooth unions, rather than sampling peaks.

    Source contract: mt.noise's unit gradients and convex interpolation give
    abs(gnoise2), abs(fbm2) <= 2 and 0 <= ridged2 <= 1 (including fractional LOD).
    In world.ridge/crag, ribs therefore add at most (1 + .35) * .55 = .7425
    times their amplitude; strata add at most their period times their blend.
    world.smax adds at most k/4 at each union. hfun's curvature and the optional
    cloud holes/valley carves only lower height. Keep these envelopes in step
    with those functions if their terrain/noise recipes change.
    """
    CR = terrain_rows() if CR is None else CR
    # S1.h_near: base <= .55*.9*1.3 + .05*2 + .2; each of its five
    # boulders updates h <= max(h, 1.6*bh + .3*h), leaving h < 3 m.
    # h_cloud <= CLOUD_Y + 150*2 + 55 + 10*2 + 7 (world's extra billows).
    # S1.h_far: prom,r <= 1, plus the nonnegative beacon Gaussian.
    ceiling = max(3.0, WD.CLOUD_Y + 382.0, 300.0 + max(WD.S1.HB + 450.0, 0.0))
    for row in CR:
        kind = row[12]
        if kind < -2.5:                       # track/carve/hole rows add no height
            continue
        if kind < -1.5:
            # near_range's final soft cap <= lip + 2000; d <= reach.
            top = row[6] - row[10] + 2000.0 + max(-row[7] * row[11], 0.0)
        elif kind < 0.0:
            if min(row[6], row[7]) <= 0.0 or min(row[10], row[11]) < 0.0:
                raise ValueError('A terrain ceiling requires positive ridge widths and nonnegative detail/slope')
            top = max(row[2], row[5]) + 0.5 * 5.5 + 0.7425 * row[10] * 90.0 + 0.5 * 2.0
        else:
            if row[3] <= 0.0 or min(row[4], row[5], row[10], row[11]) < 0.0:
                raise ValueError('A terrain ceiling requires positive crag L and nonnegative slopes/detail/shelf')
            # _drop(m) >= 0 for nonnegative slopes. Strata tw <= .22*1.5,
            # period .22*L; ribs amplitude <= detail*2*L. Shelf noise has
            # absolute bound .55*2 + .35*.55, including a signed shelf blend.
            top = row[2] + 0.33 * 0.22 * row[3] + 0.7425 * row[10] * 2.0 * row[3]
            if row[11] > 0.0:
                top += 1.2925 * abs(1.0 - row[15])
        ceiling = max(ceiling, top) + max(row[9], 0.0) * 0.25
    return float(ceiling)


def camera(frame, W=1920, H=804):
    u = min(max(frame / (NFR - 1), 0.0), 1.0)
    e = u * u * (3 - 2 * u) * 0.35 + 0.65 * u
    yaw = math.radians(YAW)
    fwd = np.array([math.sin(yaw), 0.0, math.cos(yaw)])
    pos = CAM0 + fwd * PUSH * e + np.array([0.0, 4.0 * e, 0.0])
    return RC.RCam(pos, YAW, 3.2 - 0.6 * e, 0.0, 52.0 - 4.0 * e, W, H)


def swell(frame):
    """The glow's growth: nothing, then a slow swell, breathing once a bar (80 frames), too evenly."""
    # locked to A2 (cut 80-560): the glow shows on bar 3 b1 (shot 80), is bright enough to put out the nearest
    # stars by bar 5 b1 (shot 240), and peaks for the push into the white at bar 8 b1 (A3); a breath once a bar
    u = min(max((frame - 80.0) / 360.0, 0.0), 1.0)
    return (u * u * (3 - 2 * u)) ** 1.1 * (1.0 + 0.07 * math.sin(2 * math.pi * frame / 80.0))


# ------------------------------------------------------------------ skyline (for the shadow rays) ---
_SKL = None


def skyline(cam_pos):
    """Skyline elevation (rad) every 0.1 deg over GLOW_AZ +-90 deg, seen from cam_pos."""
    global _SKL
    if _SKL is None:
        azs = np.radians(np.arange(GLOW_AZ - 90.0, GLOW_AZ + 90.0001, 0.1))
        rs = np.geomspace(300.0, 90000.0, 420)
        A, Rr = np.meshgrid(azs, rs, indexing='ij')
        X = cam_pos[0] + np.sin(A) * Rr
        Z = cam_pos[2] + np.cos(A) * Rr
        h = np.zeros(X.size)
        WD.heights(X.ravel().copy(), Z.ravel().copy(), 20.0, terrain_rows(), h)
        h = h.reshape(X.shape)
        hc = np.maximum(h, WD.CLOUD_Y)
        el = np.arctan2(hc - Rr * Rr / (2 * R_EARTH) - cam_pos[1], Rr)
        from scipy.ndimage import gaussian_filter1d
        _SKL = (float(azs[0]), 0.1 * math.pi / 180.0, gaussian_filter1d(el.max(axis=1), 5.0))
    return _SKL


# ------------------------------------------------------------------ the sky ---
# GP: 0 az_s | 1 el_s | 2 I | 3-5 glow rgb | 6 arch az width | 7 arch height | 8 core I | 9 core az w | 10 core h
#     11 rays | 12 cone tilt | 13 cone base width | 14 cone height | 15 design (0 arc, 1 cone, 2 veil) | 16 deck gain
#     17 star kill | 18 deck coverage | 19 deck altitude | 20-22 zenith | 23-25 horizon | 26 MW I


@njit(inline='always', fastmath=True)
def glow_at(dx, dy, dz, GP, SKL0, SKLD, SKL):
    el = math.asin(min(max(dy, -1.0), 1.0))
    az = math.atan2(dx, dz)
    daz = az - GP[0]
    if daz > math.pi:
        daz -= 2 * math.pi
    if daz < -math.pi:
        daz += 2 * math.pi
    I = GP[2]
    d = GP[15]
    g = 0.0
    e = max(el, 0.0)
    if d < 0.5:
        # the arch: wide and low, flattening away from the source; a whiter core just over the source
        hh = GP[7] * (1.0 - 0.45 * min((daz / GP[6]) ** 2, 1.0))
        g = I * math.exp(-e / max(hh, 1e-3)) * math.exp(-(daz / GP[6]) ** 2)
        g += I * GP[8] * math.exp(-e / GP[10]) * math.exp(-(daz / GP[9]) ** 2)
        # shadow rays: the ray from the sub-horizon source through this point crosses the skyline at az_x;
        # where a far peak stands above the general skyline there, the ray is in shadow
        if GP[11] > 0.0 and el > -0.01:
            dl = el - GP[1]
            if dl > 1e-4:
                ex = 0.004 - GP[1]
                azx = GP[0] + daz * ex / dl
                k = int((azx - SKL0) / SKLD)
                if k > 2 and k < SKL.shape[0] - 3:
                    pk = SKL[k]
                    loc = 0.0
                    for q in range(-30, 31, 6):
                        kk = min(max(k + q, 0), SKL.shape[0] - 1)
                        loc += SKL[kk]
                    loc /= 11.0
                    blk = min(max((pk - loc - 0.0006) / 0.0030, 0.0), 1.0)
                    fade = math.exp(-dl / 0.22) * math.exp(-(daz / 0.45) ** 2)   # rays only near the source
                    g *= 1.0 - GP[11] * blk * fade
    elif d < 1.5:
        # a leaning pyramid of pale light (the zodiacal cone)
        ct, st = math.cos(GP[12]), math.sin(GP[12])
        v = daz * ct + e * st
        u = -daz * st + e * ct
        vv = max(v, 0.0)
        w = GP[13] * max(1.0 - vv / GP[14], 0.08)
        g = I * math.exp(-(u / w) ** 2) * math.exp(-vv / (0.45 * GP[14])) * math.exp(-max(-v, 0.0) / 0.03)
        g += I * GP[8] * math.exp(-e / GP[10]) * math.exp(-(daz / GP[9]) ** 2)
    else:
        # the veil: only a thin line on the skyline
        g = I * GP[8] * math.exp(-e / GP[10]) * math.exp(-(daz / GP[9]) ** 2)
    return g


@njit(inline='always', fastmath=True)
def lens_at(az, el, LN, k):
    """Lenticular k (LN row: az, el, half width, half height (rad), tilt, seed): density and how far up its
    body the point is (0 = underside, 1 = top). A lens: a flat, bright underside and a domed top that thin to
    fine tips, with faint laminar banding."""
    daz = az - LN[k, 0]
    if daz > math.pi:
        daz -= 2 * math.pi
    if daz < -math.pi:
        daz += 2 * math.pi
    u = daz / LN[k, 2]
    if abs(u) >= 1.0:
        return 0.0, 0.0
    v = (el - LN[k, 1] - LN[k, 4] * daz) / LN[k, 3]
    q = 1.0 - u * u
    vb = -0.55 * q ** 0.9
    vt = 1.0 * q ** 0.6
    if v <= vb - 0.12 or v >= vt + 0.12:
        return 0.0, 0.0
    e = 0.09
    d = min(max((v - vb) / e + 0.5, 0.0), 1.0) * min(max((vt - v) / e + 0.5, 0.0), 1.0)
    band = 0.84 + 0.16 * math.sin(v * 14.0 + 1.5 * fbm2(u * 3.0 + LN[k, 5], v * 0.7, 3.0, 181))
    d *= band * min(q * 6.0, 1.0)
    return d, min(max((v - vb) / max(vt - vb, 1e-3), 0.0), 1.0)


@njit(parallel=True, fastmath=True, cache=True)
def sky_pass(img, dist, trans, C, GP, MW, SKL0, SKLD, SKL, cam_y, t, LN):
    """Overwrite sky pixels: night gradient + Milky Way (dimmed by the glow) + the glow + a high altocumulus deck
    lit from beneath by the sub-horizon source. trans <- the deck's transmittance (star mask)."""
    H, W = img.shape[0], img.shape[1]
    f, cx, cyy = C[7], C[8], C[9]
    pix = 1.0 / f
    for j in prange(H):
        for i in range(W):
            if dist[j, i] < 1e8:
                trans[j, i] = 0.0
                continue
            xo = i + 0.5 - cx
            dxh = C[3] * f + C[5] * xo
            dzh = C[4] * f + C[6] * xo
            vy = cyy - (j + 0.5)
            nn = math.sqrt(dxh * dxh + dzh * dzh + vy * vy)
            dx, dy, dz = dxh / nn, vy / nn, dzh / nn
            el = math.asin(min(max(dy, -1.0), 1.0))
            u = min(max(el / 0.9, 0.0), 1.0) ** 0.55
            r = GP[23] * (1 - u) + GP[20] * u
            g_ = GP[24] * (1 - u) + GP[21] * u
            b = GP[25] * (1 - u) + GP[22] * u
            G = glow_at(dx, dy, dz, GP, SKL0, SKLD, SKL)
            kill = math.exp(-G / GP[17])
            mr, mg, mb = SK.milky_way(dx, dy, dz, MW, GP[26])
            r += mr * kill + G * GP[3]
            g_ += mg * kill + G * GP[4]
            b += mb * kill + G * GP[5]
            tr = 1.0
            if dy > 0.004 and GP[16] > 0.0:
                Hc = GP[19]
                tt = math.sqrt(2.0 * R_EARTH * Hc + (R_EARTH * dy) ** 2) - R_EARTH * dy
                hx = dx / math.sqrt(dx * dx + dz * dz)
                hz = dz / math.sqrt(dx * dx + dz * dz)
                x = C[0] + hx * tt
                z = C[2] + hz * tt
                fp = tt * pix / max(dy + 0.02, 0.02)
                o1 = min(max(math.log2(max(2600.0 / max(fp * 2.0, 1e-3), 1.0)), 1.0), 6.0)
                o2 = min(max(math.log2(max(700.0 / max(fp * 2.0, 1e-3), 1.0)), 0.0), 4.0)
                # a mackerel sky: distinct cloudlets in rows across the wind, banks with clear gaps between them;
                # crisp edges (no smudges), so the glow can silver each cloudlet's rim
                ua = 0.80 * x + 0.60 * z
                va = -0.60 * x + 0.80 * z
                n1 = fbm2(ua / 3800.0 + 0.00002 * t, va / 1600.0, o1, 171)
                n2 = fbm2(x / 700.0, z / 700.0, o2, 172) if o2 > 0.5 else 0.0
                cov = GP[18] + 0.95 * fbm2(x / 30000.0 + 3.0, z / 30000.0, 3.0, 173)
                dens = min(max((n1 + 0.45 * n2 + cov + 0.02) / 0.22, 0.0), 1.0)
                dens = dens * dens * (3.0 - 2.0 * dens)
                # far off the deck thins into streaks, and fades out toward the horizon
                dens *= min(max((dy - 0.03) / 0.12, 0.0), 1.0)
                if dens > 0.0:
                    tau = 2.6 * dens
                    a = 1.0 - math.exp(-tau)
                    # light from the source: strongest for cloud in its direction and nearer to it (low, far)
                    az = math.atan2(dx, dz)
                    daz = az - GP[0]
                    if daz > math.pi:
                        daz -= 2 * math.pi
                    if daz < -math.pi:
                        daz += 2 * math.pi
                    wz = 0.85 if GP[15] < 0.5 else (0.45 if GP[15] < 1.5 else 1.3)
                    # grazing light from below the horizon: cloud high overhead gets little of it
                    Lc = GP[2] * GP[16] * math.exp(-(daz / wz) ** 2) * (0.10 + 0.90 * math.exp(-el / 0.13))
                    cosg = math.cos(daz) * math.cos(el)
                    ph = 0.25 + 2.2 * max(cosg, 0.0) ** 6
                    # the whole underside glows softly; the thin edges glow brighter (forward scatter)
                    silver = a * math.exp(-0.9 * tau) * ph * 1.5 + 0.55 * a
                    cr = GP[3] * Lc * silver + a * 0.0045
                    cg = GP[4] * Lc * silver + a * 0.0055
                    cb = GP[5] * Lc * silver + a * 0.0090
                    tr = 1.0 - a
                    r = r * tr + cr
                    g_ = g_ * tr + cg
                    b = b * tr + cb
            # the lenticular stack over the great range: lit from beneath by the glow, dark-topped
            az = math.atan2(dx, dz)
            for k in range(LN.shape[0]):
                dl, hv = lens_at(az, el, LN, k)
                if dl <= 0.0:
                    continue
                al = min(dl * 0.92, 0.95)
                daz = az - GP[0]
                if daz > math.pi:
                    daz -= 2 * math.pi
                if daz < -math.pi:
                    daz += 2 * math.pi
                Ls = GP[2] * math.exp(-(daz / 0.9) ** 2) * 1.25
                under = (1.0 - hv) ** 2.2
                rim = math.exp(-hv / 0.06)
                lr = GP[3] * Ls * (0.12 + 0.95 * under + 0.9 * rim) + 0.004
                lg = GP[4] * Ls * (0.12 + 0.95 * under + 0.9 * rim) + 0.005
                lb = GP[5] * Ls * (0.12 + 0.95 * under + 0.9 * rim) + 0.009
                r = r * (1.0 - al) + lr * al
                g_ = g_ * (1.0 - al) + lg * al
                b = b * (1.0 - al) + lb * al
                tr *= 1.0 - al
            trans[j, i] = tr * kill
            img[j, i, 0] = r
            img[j, i, 1] = g_
            img[j, i, 2] = b


def glow_params(design, I):
    GP = np.zeros(32)
    GP[0] = math.radians(GLOW_AZ)
    GP[1] = math.radians(EL_S)
    GP[2] = I
    col = np.array([0.72, 0.88, 1.00])          # the mind palette: ice-white, a breath of cyan
    GP[3:6] = col
    GP[6] = math.radians(44.0)
    GP[7] = math.radians(5.0)
    GP[8] = 0.7
    GP[9] = math.radians(30.0)
    GP[10] = math.radians(2.6)
    GP[11] = 0.40
    GP[12] = math.radians(62.0)
    GP[13] = math.radians(13.0)
    GP[14] = math.radians(42.0)
    GP[15] = DESIGNS.index(design)
    GP[16] = {'arc': 0.95, 'cone': 0.30, 'veil': 0.75}[design]
    GP[17] = 0.032
    GP[18] = {'arc': -0.30, 'cone': -0.34, 'veil': -0.14}[design]
    GP[19] = 5500.0
    GP[20:23] = lin('#030822') * 1.0
    GP[23:26] = lin('#101A40') * 0.8
    GP[26] = 0.09
    if design == 'veil':
        GP[8] = 1.0
        GP[9] = math.radians(34.0)
        GP[10] = math.radians(0.9)
    return GP


MOON = os.environ.get('FD_MOON', '1') == '1'
MOON_I = float(os.environ.get('FD_MOON_I', '0.28'))     # a thin moon: the Run's night is 0.55


@njit(parallel=True, fastmath=True, cache=True)
def glow_haze(img, dist, C, GP, SKL0, SKLD, SKL, haze_k, haze_d):
    """The glow in the far haze over terrain and cloud (sky pixels are sky_pass's): a share that grows with distance,
    seen at the horizon's elevation (the arch's foot). Used with the moon key, whose light cannot carry the glow."""
    H, W = img.shape[0], img.shape[1]
    f, cx, cyy = C[7], C[8], C[9]
    for j in prange(H):
        for i in range(W):
            d = dist[j, i]
            if d >= 1e8:
                continue
            hz = haze_k * (1.0 - math.exp(-d / haze_d))
            if hz < 1e-4:
                continue
            xo = i + 0.5 - cx
            dxh = C[3] * f + C[5] * xo
            dzh = C[4] * f + C[6] * xo
            vy = cyy - (j + 0.5)
            nn = math.sqrt(dxh * dxh + dzh * dzh + vy * vy)
            G = glow_at(dxh / nn, max(vy / nn, 0.0), dzh / nn, GP, SKL0, SKLD, SKL)
            img[j, i, 0] += G * GP[3] * hz
            img[j, i, 1] += G * GP[4] * hz
            img[j, i, 2] += G * GP[5] * hz


def milky_way():
    # the band rises from the left horizon (the bulge low on the left) and arches over the top of the frame
    a = np.array([math.sin(math.radians(-62.0)) * math.cos(math.radians(-4.0)), math.sin(math.radians(-4.0)),
                  math.cos(math.radians(-62.0)) * math.cos(math.radians(-4.0))])
    b = np.array([math.sin(math.radians(-38.0)) * math.cos(math.radians(42.0)), math.sin(math.radians(42.0)),
                  math.cos(math.radians(-38.0)) * math.cos(math.radians(42.0))])
    n = np.cross(a, b)
    n /= np.linalg.norm(n)
    c = a + 0.12 * (b - a)
    c /= np.linalg.norm(c)
    return np.array([n[0], n[1], n[2], c[0], c[1], c[2], 0.20, 31.0])


def light(frame, design):
    I = 0.46 * swell(frame)
    GP = glow_params(design, I)
    lk_el = math.radians(1.6)
    a = math.radians(GLOW_AZ)
    Lk = np.r_[np.array([math.cos(lk_el) * math.sin(a), math.sin(lk_el), math.cos(lk_el) * math.cos(a)]), GP[3:6]]
    if MOON:
        # (H5, director: "the cloud sea ... texture and moonlit tops") a thin moon, out of frame high on the left (the
        # A night's moon, az -59 el 21), keys the snow and the cloud-sea billows from the first frame; the glow keeps
        # the sky and lights the far haze (glow_haze), so it still reads as a light below the horizon
        Lk = np.r_[WD.MOON_DIR, lin('#9DB4D9')]
    amb = lin('#1A2440') * 0.10
    S = SK.sky_params(zenith='#04071A', horizon='#141C3C', moon_dir=(0.0, -1.0, 0.0), halo_I=0.0, halo2_I=0.0,
                      horizon_glow=0.0)
    fogc = lin('#232A3A') * 0.22
    # (H5) aerial depth: the near islands dark, the sierra half veiled, the far wall and the needles pale; a thicker
    # cloud-top mist so the peaks stand IN the cloud sea (soft feet), never on a lake shore
    fogp = np.array([1.0e-4, 1 / 1500.0, 3.6e-4, 1 / 140.0, 6.0 * min(I / 0.46, 1.0), fogc[0], fogc[1], fogc[2]])
    if MOON:
        fogp[4] = 0.0                    # no in-scatter toward the moon: the haze glows toward the glow (glow_haze)
    Q = np.zeros(24)
    Q[0] = {'arc': 0.40, 'cone': 0.20, 'veil': 0.25}[design] * I / 0.46
    if MOON:
        Q[0] = MOON_I
    Q[1] = 3.0
    Q[2] = 10.0
    Q[3] = 0.40
    Q[4] = 0.60
    Q[5] = 0.25
    Q[7] = 0.25
    Q[9] = 0.9
    Q[10] = 1.0
    Q[11] = 12.0
    Q[13] = 3.2                      # (H5) the cloud sea is brighter than the rock under the night sky: cloud, not water
    Q[8] = 0.72                      # deeper troughs between the billows: relief, not ripples
    Q[16] = 0.05
    Q[18] = 1.0                      # (H5) anti-streak snow noise (world.shade flag)
    Q[3] = 0.30                      # snow holds on steeper ground (no combed stripes)
    Q[4] = 0.54
    return (Lk, amb, S, fogp, Q), GP


def lenses():
    """Three lenticulars stacked over the range's highest spire (angles from the camera at frame 0)."""
    top = RANGE_PTS[int(np.argmax(RANGE_PTS[:, 1]))]
    d = top - CAM0
    az = math.atan2(d[0], d[2])
    return np.array([[az + math.radians(1.0), math.radians(5.2), math.radians(8.5), math.radians(0.95), 0.02, 3.0],
                     [az - math.radians(0.6), math.radians(7.3), math.radians(6.2), math.radians(0.78), -0.02, 7.0],
                     [az + math.radians(0.5), math.radians(9.0), math.radians(4.0), math.radians(0.55), 0.03, 11.0]])


_STARS = None


def sky_parameters(GP, candidate='accepted'):
    """Default-off A2 study; the accepted parameter array is never mutated.

    At cut 212, isolating GP[16] removes the dark horizontal cloudlets while
    GP[26] alone carries the pale diagonal Milky Way (sky_probe.py). Clear the
    high deck for this candidate; keep the cloud sea, glow and galactic dust.
    Do not smooth the finished plate: it would also smear stars and ridge edges.
    """
    if candidate == 'accepted':
        return GP
    if candidate != 'clear_high_deck':
        raise ValueError(f'Unknown false-dawn sky candidate: {candidate}')
    clear = GP.copy()
    clear[16] = 0.0
    return clear


def render(frame, design='arc', scale=1.0, ss=1.5, sky_candidate='accepted'):
    global _STARS
    import run as RN
    W, H = int(round(1920 * scale)), int(round(804 * scale))
    tcam = camera(frame, W, H)
    fr = PI.Frame(tcam, ss)
    lt, GP = light(frame, design)
    GP = sky_parameters(GP, sky_candidate)
    PI.render_terrain(fr, frame, terrain_rows(), lt, np.zeros((0, 8)), hmax=terrain_hmax())
    scam = fr.src
    C = scam.params()
    skl0, skld, skl = skyline(scam.pos)
    if MOON:
        GPh = GP.copy()
        GPh[11] = 0.0                    # no shadow rays in the haze: under the horizon they fall as vertical slabs
        glow_haze(fr.img, fr.dist, C, GPh, skl0, skld, skl, 0.25, 30000.0)
    trans = np.zeros(fr.dist.shape, np.float32)
    # no lenticular stack: at night, lit from beneath, stacked lenses read as a fleet of saucers (a real-life
    # "UFO cloud"); the deck carries the structure instead
    sky_pass(fr.img, fr.dist, trans, C, GP, milky_way(), skl0, skld, skl, scam.pos[1], frame / FPS,
             np.zeros((0, 6)))
    if _STARS is None:
        _STARS = SK.make_stars(16000, 101, lum_scale=6.0)
    SK.splat_stars(fr.img, scam, _STARS, trans, t=frame / FPS, gain=ss * ss, scale=PI.src_scale(fr))
    img, zb, di = PI.to_target(fr)
    return img


FINISH = dict(exposure=1.05, bloom_strength=0.06, bloom_threshold=0.6, streak_strength=0.0, vignette_amount=0.25)


CUT0 = 80                                        # A2 starts on A's cut frame 80 (bar 2): files use cut frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--frames', default=None)
    ap.add_argument('--range', default=None)
    ap.add_argument('--design', default='arc')          # the H5 call: ARC only (cone, veil kept for the record)
    ap.add_argument('--scale', type=float, default=1.0)
    ap.add_argument('--ss', type=float, default=1.5)
    ap.add_argument('--out', default=None)
    ap.add_argument('--threads', type=int, default=4)
    ap.add_argument('--procs', type=int, default=1)
    ap.add_argument('--skip', action='store_true')
    ap.add_argument('--numbering', choices=('cut', 'shot'), default='cut',
                    help="cut (default, EDIT-v3's delivery convention): frames are A's cut frames 80-559; "
                         "shot: 0-479 (tests)")
    a = ap.parse_args()
    off = CUT0 if a.numbering == 'cut' else 0
    base = os.path.join(PI.CM.ROOT, 'renders', 'falsedawn_A')
    out = base if a.out is None else os.path.join(base, a.out)
    if a.range:
        s, e = a.range.split('-')
        frames = list(range(int(s), int(e) + 1))
    else:
        frames = [int(x) for x in (a.frames or str(off + 420)).split(',')]
    bad = [f for f in frames if not off <= f < off + NFR]
    if bad:
        raise SystemExit(f'frames outside the shot ({off}-{off + NFR - 1} in {a.numbering} numbering): {bad[:5]}')
    designs = DESIGNS if a.design == 'all' else (a.design,)
    look = PI.look
    for d in designs:
        od = out if len(designs) == 1 and a.out is None else os.path.join(out, d) if len(designs) > 1 else out
        os.makedirs(od, exist_ok=True)
        todo = [f for f in frames if not (a.skip and os.path.exists(look.frame_path(od, f)))]
        if a.procs <= 1:
            _work((todo, d, a.scale, a.ss, od, a.threads, off))
        else:
            import multiprocessing as mp
            chunks = [todo[i::a.procs] for i in range(a.procs)]
            with mp.get_context('spawn').Pool(a.procs) as pool:
                pool.map(_work, [(c, d, a.scale, a.ss, od, 1, off) for c in chunks])


def _work(args):
    frames, d, scale, ss, od, threads, off = args
    import numba                        # NUMBA_NUM_THREADS is fixed once numba runs (the farm sets it per node)
    numba.set_num_threads(max(1, min(threads, numba.config.NUMBA_NUM_THREADS)))
    look = PI.look
    for f in frames:
        t0 = time.time()
        img = look.finish(render(f - off, d, scale, ss), **FINISH)
        look.save_png(look.frame_path(od, f), img)
        print(f'{d} frame {f} {time.time() - t0:.1f}s', flush=True)


if __name__ == '__main__':
    main()
