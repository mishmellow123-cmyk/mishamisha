"""FALSE DAWN (cut A, R1). 480 frames, shot-local numbering 0-479 (renders/falsedawn_A/).

High on a ridge at midnight, moonless: the Milky Way above, the cloud sea below. Beyond the far ranges a cold
white glow swells under the horizon, at the azimuth where the true morning will come (the world's sunrise
azimuth, -30 deg), silvers the undersides of a high cloud deck and puts out the stars near it. A slow push.
It must read as morning come too early, never as a city glow (amber, a dome, lit from above) or a moonrise (one
bright point about to break). Three designs of the glow itself, all sharing the deck, the star extinction and
the terrain lit from below the horizon:
  arc   a wide dawn arch with shadow rays fanning up from the far peaks (a sun-like source below the horizon)
  cone  a tall leaning pyramid of pale light (the zodiacal light, the astronomer's own "false dawn")
  veil  no visible source: only the deck lit from beneath across the sky, and a thin white line on the skyline

  python falsedawn.py --design arc --frames 420 --scale 0.5 --out stills
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


def camera(frame, W=1920, H=804):
    u = min(max(frame / (NFR - 1), 0.0), 1.0)
    e = u * u * (3 - 2 * u) * 0.35 + 0.65 * u
    yaw = math.radians(YAW)
    fwd = np.array([math.sin(yaw), 0.0, math.cos(yaw)])
    pos = CAM0 + fwd * PUSH * e + np.array([0.0, 4.0 * e, 0.0])
    return RC.RCam(pos, YAW, 3.2 - 0.6 * e, 0.0, 52.0 - 4.0 * e, W, H)


def swell(frame):
    """The glow's growth: nothing, then a slow swell, breathing once a bar (80 frames), too evenly."""
    u = min(max((frame - 30.0) / 400.0, 0.0), 1.0)
    return (u * u * (3 - 2 * u)) ** 1.3 * (1.0 + 0.07 * math.sin(2 * math.pi * frame / 80.0))


# ------------------------------------------------------------------ skyline (for the shadow rays) ---
_SKL = None


def skyline(cam_pos):
    """Skyline elevation (rad) every 0.1 deg over GLOW_AZ +-90 deg, seen from cam_pos."""
    global _SKL
    if _SKL is None:
        import run as RN
        azs = np.radians(np.arange(GLOW_AZ - 90.0, GLOW_AZ + 90.0001, 0.1))
        rs = np.geomspace(300.0, 90000.0, 420)
        A, Rr = np.meshgrid(azs, rs, indexing='ij')
        X = cam_pos[0] + np.sin(A) * Rr
        Z = cam_pos[2] + np.cos(A) * Rr
        h = np.zeros(X.size)
        WD.heights(X.ravel().copy(), Z.ravel().copy(), 20.0, RN.CR, h)
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
                    fade = math.exp(-dl / 0.22)
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


@njit(parallel=True, fastmath=True, cache=True)
def sky_pass(img, dist, trans, C, GP, MW, SKL0, SKLD, SKL, cam_y, t):
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
                n1 = fbm2(x / 2600.0 + 0.00002 * t, z / 2600.0, o1, 171)
                n2 = fbm2(x / 700.0, z / 700.0, o2, 172) if o2 > 0.5 else 0.0
                cov = GP[18] + 0.60 * fbm2(x / 42000.0 + 3.0, z / 42000.0, 3.0, 173)
                dens = min(max((n1 + 0.4 * n2 + cov + 0.05) / 0.75, 0.0), 1.0) ** 1.5
                # far off the deck thins into streaks, and fades out toward the horizon
                dens *= min(max((dy - 0.03) / 0.12, 0.0), 1.0)
                if dens > 0.0:
                    tau = 1.7 * dens
                    a = 1.0 - math.exp(-tau)
                    # light from the source: strongest for cloud in its direction and nearer to it (low, far)
                    az = math.atan2(dx, dz)
                    daz = az - GP[0]
                    if daz > math.pi:
                        daz -= 2 * math.pi
                    if daz < -math.pi:
                        daz += 2 * math.pi
                    wz = 0.85 if GP[15] < 0.5 else (0.45 if GP[15] < 1.5 else 1.3)
                    Lc = GP[2] * GP[16] * math.exp(-(daz / wz) ** 2) * (0.35 + 0.65 * math.exp(-el / 0.35))
                    cosg = math.cos(daz) * math.cos(el)
                    ph = 0.25 + 2.2 * max(cosg, 0.0) ** 6
                    silver = a * math.exp(-0.9 * tau) * ph * 1.8 + 0.30 * a
                    cr = GP[3] * Lc * silver + a * 0.0045
                    cg = GP[4] * Lc * silver + a * 0.0055
                    cb = GP[5] * Lc * silver + a * 0.0090
                    tr = 1.0 - a
                    r = r * tr + cr
                    g_ = g_ * tr + cg
                    b = b * tr + cb
            trans[j, i] = tr * kill
            img[j, i, 0] = r
            img[j, i, 1] = g_
            img[j, i, 2] = b


def glow_params(design, I):
    GP = np.zeros(32)
    GP[0] = math.radians(GLOW_AZ)
    GP[1] = math.radians(EL_S)
    GP[2] = I
    col = np.array([0.80, 0.90, 1.00])          # the mind palette: ice-white, a breath of cyan
    GP[3:6] = col
    GP[6] = math.radians(44.0)
    GP[7] = math.radians(6.0)
    GP[8] = 0.7
    GP[9] = math.radians(30.0)
    GP[10] = math.radians(2.6)
    GP[11] = 0.40
    GP[12] = math.radians(62.0)
    GP[13] = math.radians(13.0)
    GP[14] = math.radians(42.0)
    GP[15] = DESIGNS.index(design)
    GP[16] = {'arc': 0.40, 'cone': 0.30, 'veil': 0.75}[design]
    GP[17] = 0.032
    GP[18] = {'arc': -0.30, 'cone': -0.34, 'veil': -0.14}[design]
    GP[19] = 5500.0
    GP[20:23] = lin('#04071A') * 0.9
    GP[23:26] = lin('#141C3C') * 0.8
    GP[26] = 0.09
    if design == 'veil':
        GP[8] = 1.0
        GP[9] = math.radians(34.0)
        GP[10] = math.radians(0.9)
    return GP


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
    amb = lin('#1A2440') * 0.07
    S = SK.sky_params(zenith='#04071A', horizon='#141C3C', moon_dir=(0.0, -1.0, 0.0), halo_I=0.0, halo2_I=0.0,
                      horizon_glow=0.0)
    fogc = lin('#232A3A') * 0.22
    fogp = np.array([5.0e-5, 1 / 1500.0, 2.2e-4, 1 / 140.0, 6.0 * min(I / 0.46, 1.0), fogc[0], fogc[1], fogc[2]])
    Q = np.zeros(24)
    Q[0] = {'arc': 0.40, 'cone': 0.20, 'veil': 0.25}[design] * I / 0.46
    Q[1] = 3.0
    Q[2] = 10.0
    Q[3] = 0.40
    Q[4] = 0.60
    Q[5] = 0.25
    Q[7] = 0.25
    Q[8] = 0.55
    Q[9] = 0.9
    Q[10] = 1.0
    Q[11] = 12.0
    Q[13] = 1.5
    Q[16] = 0.05
    return (Lk, amb, S, fogp, Q), GP


_STARS = None


def render(frame, design='arc', scale=1.0, ss=1.5):
    global _STARS
    import run as RN
    W, H = int(round(1920 * scale)), int(round(804 * scale))
    tcam = camera(frame, W, H)
    fr = PI.Frame(tcam, ss)
    lt, GP = light(frame, design)
    PI.render_terrain(fr, frame, RN.CR, lt, np.zeros((0, 8)))
    scam = fr.src
    C = scam.params()
    skl0, skld, skl = skyline(scam.pos)
    trans = np.zeros(fr.dist.shape, np.float32)
    sky_pass(fr.img, fr.dist, trans, C, GP, milky_way(), skl0, skld, skl, scam.pos[1], frame / FPS)
    if _STARS is None:
        _STARS = SK.make_stars(16000, 101, lum_scale=6.0)
    SK.splat_stars(fr.img, scam, _STARS, trans, t=frame / FPS, gain=ss * ss, scale=PI.src_scale(fr))
    img, zb, di = PI.to_target(fr)
    return img


FINISH = dict(exposure=1.05, bloom_strength=0.06, bloom_threshold=0.6, streak_strength=0.0, vignette_amount=0.25)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--frames', default='420')
    ap.add_argument('--range', default=None)
    ap.add_argument('--design', default='arc')
    ap.add_argument('--scale', type=float, default=1.0)
    ap.add_argument('--ss', type=float, default=1.5)
    ap.add_argument('--out', default=None)
    ap.add_argument('--threads', type=int, default=4)
    ap.add_argument('--procs', type=int, default=1)
    ap.add_argument('--skip', action='store_true')
    a = ap.parse_args()
    base = os.path.join(PI.CM.ROOT, 'renders', 'falsedawn_A')
    out = base if a.out is None else os.path.join(base, a.out)
    if a.range:
        s, e = a.range.split('-')
        frames = list(range(int(s), int(e) + 1))
    else:
        frames = [int(x) for x in a.frames.split(',')]
    designs = DESIGNS if a.design == 'all' else (a.design,)
    look = PI.look
    for d in designs:
        od = out if len(designs) == 1 and a.out is None else os.path.join(out, d) if len(designs) > 1 else out
        os.makedirs(od, exist_ok=True)
        todo = [f for f in frames if not (a.skip and os.path.exists(look.frame_path(od, f)))]
        if a.procs <= 1:
            _work((todo, d, a.scale, a.ss, od, a.threads))
        else:
            import multiprocessing as mp
            chunks = [todo[i::a.procs] for i in range(a.procs)]
            with mp.get_context('spawn').Pool(a.procs) as pool:
                pool.map(_work, [(c, d, a.scale, a.ss, od, 1) for c in chunks])


def _work(args):
    frames, d, scale, ss, od, threads = args
    os.environ['NUMBA_NUM_THREADS'] = str(threads)
    import numba
    numba.set_num_threads(threads)
    look = PI.look
    for f in frames:
        t0 = time.time()
        img = look.finish(render(f, d, scale, ss), **FINISH)
        look.save_png(look.frame_path(od, f), img)
        print(f'{d} frame {f} {time.time() - t0:.1f}s', flush=True)


if __name__ == '__main__':
    main()
