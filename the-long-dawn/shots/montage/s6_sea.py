"""Shot 6 (1720-1767) SEA.

One idea: a boat in the moon-path raises fire, and the whole coast answers.
Dark swells, the moon low over the sea laying a glitter path; a small boat sits in the path, a
sailor in oilskins at the bow strikes and raises a burning hand flare on the hit at 1730; then along
the far coast a chain of beacons ignites one after another, left -> right toward frame centre
(running into the 1760-1767 handle and on into the globe). Camera floats on the swell.

v2 (director's review: "the flare reads as a tiny white sparkler next to a big bright moon"):
* the flare is the hero light: a dense tongue flame (white-gold head -> orange body -> red tips)
  rendered 3x supersampled, a white-hot burning head, a multi-scale HDR glow + air halo, its own
  punchy ignition envelope (catch on 1729, whoomp + light flash on the 1730 hit) and a sputter;
* it lights the scene: the sailor (face, arm, oilskin speculars), bow, gunwale, stays, mast and a
  furled sail on the boom, plus a warm pool on the wave faces around the boat; its specular
  streak on the water is several times stronger, with a broad soft sheen under it;
* a dense smoke plume born at the flare head (it tracks the moving hand) drifting downwind, lit
  orange from within; sparks and slow embers born at the flare; burning dross dripping to the sea;
* coast beacons: the whole chain is on screen (the first used to be off-frame and the last two sat
  inside the flare's glow), each with a hot core, an ignition pop, a small glow and a short
  glittering reflection column on the water;
* the moon disk and halo are tamed (the moon path keeps its own glitter intensity).
"""
import math

import cv2
import numpy as np
from numba import njit, prange

import common as CM
from mt import sky as SK, fire as F, figure as FG, props as PR, people as PP
from mt.figure import Drawing
from mt.cam import Cam, keys, smoother, kick
from mt.noise import fbm2, gnoise2, smoothstep

START, END, IGN = 1720, 1767, 1730
HFOV = 45.0
F_FULL = 960.0 / math.tan(math.radians(HFOV / 2))
HORIZON_Y0 = 352.0
YAW0 = -0.5
MOON_AZ, MOON_EL = 8.5, 6.2
MOON_DIR = np.array([math.sin(math.radians(MOON_AZ)) * math.cos(math.radians(MOON_EL)), math.sin(math.radians(MOON_EL)),
                     math.cos(math.radians(MOON_AZ)) * math.cos(math.radians(MOON_EL))])
BOAT = np.array([6.3, 0.0, 72.0])
COAST_D = 7000.0

GOLD = np.array([1.0, 0.62, 0.26])      # hot glow around the flame head
EMBER = np.array([1.0, 0.42, 0.12])     # deeper orange for the wide air glow

# wave spectrum: wavelength, amplitude, direction (rad), phase
_rng = np.random.default_rng(606)
_L = np.array([46.0, 27.0, 15.0, 8.7, 5.1, 3.0, 1.8, 1.1, 0.7])
WAVES = np.stack([_L, 0.0075 * _L ** 1.05, 0.25 + _rng.normal(size=len(_L)) * 0.5, _rng.random(len(_L)) * 6.28], 1)

# materials: the shared set + canvas (furled sail) + painted hull planking
MATS = np.vstack([FG.MATS,
                  [[0.100, 0.090, 0.076, 1.1, 0.1, 8.0, 0.07, 0, 0, 0, 0],      # 13 canvas
                   [0.042, 0.034, 0.028, 0.9, 0.5, 12.0, 0.22, 0, 0, 0, 0]]])   # 14 painted hull
MATS[3, 4:6] = (0.5, 20.0)   # oilskin: a tighter, weaker sheen (the flare is close; flat puppet faces)


@njit(inline='always', fastmath=True)
def wave_slope(x, z, t, fp, Wv):
    sx = 0.0
    sz = 0.0
    var = 0.0
    for i in range(Wv.shape[0]):
        lam = Wv[i, 0]
        k = 6.2832 / lam
        a = Wv[i, 1]
        ca = math.cos(Wv[i, 2])
        sa = math.sin(Wv[i, 2])
        om = math.sqrt(9.81 * k)
        w = 1.0 - smoothstep(0.12, 0.5, fp / lam)
        ph = k * (ca * x + sa * z) - om * t + Wv[i, 3]
        c = math.cos(ph) * a * k
        sx += w * c * ca
        sz += w * c * sa
        var += (1.0 - w) * 0.5 * (a * k) ** 2
    return sx, sz, var


@njit(fastmath=True)
def swell_h(x, z, t, Wv):
    h = 0.0
    for i in range(3):
        lam = Wv[i, 0]
        k = 6.2832 / lam
        om = math.sqrt(9.81 * k)
        h += Wv[i, 1] * math.sin(k * (math.cos(Wv[i, 2]) * x + math.sin(Wv[i, 2]) * z) - om * t + Wv[i, 3])
    return h


@njit(inline='always', fastmath=True)
def coast_elev(az):
    """Angular height (rad) of the far coast above the sea horizon at azimuth az (rad)."""
    if az > 0.075:
        return -1.0
    base = 0.010 + 0.016 * max(fbm2(az * 7.0 + 3.0, 0.5, 5.0, 91), -0.3) + 0.006 * math.sin(az * 23.0)
    fade = smoothstep(0.075, 0.035, az)
    return max(base, 0.002) * fade


@njit(parallel=True, fastmath=True)
def shade(C, S, Wv, t, LT, FL, out, dep, skym):
    """Sky / far coast / sea. LT rows: specular fire streaks [x,y,z, r,g,b, I, lobe width, -].
    FL: the flare's warm pool on the wave faces [x,y,z, I, r,g,b, cutoff radius^2].
    S[25], S[26]: moon-path glitter intensity and lobe radius (independent of the tamed disk)."""
    H, W = out.shape[0], out.shape[1]
    fx, fz, rx, rz = C[3], C[4], C[5], C[6]
    f, cx, cyy = C[7], C[8], C[9]
    ox, oy, oz = C[0], C[1], C[2]
    pix = 1.0 / f
    for j in prange(H):
        for i in range(W):
            vx = (i + 0.5 - cx)
            vy = (cyy - (j + 0.5))
            dx = fx * f + rx * vx
            dz = fz * f + rz * vx
            dy = vy
            n = math.sqrt(dx * dx + dy * dy + dz * dz)
            dx /= n
            dy /= n
            dz /= n
            az = math.atan2(dx, dz)
            # sea horizon dips slightly below 0 because the camera is above the water
            dip = -math.sqrt(2.0 * max(oy, 0.1) / 6.371e6)
            if dy > dip:
                ce = coast_elev(az)
                if ce > 0.0 and dy < dip + ce:
                    # far coast: hazy dark land, faint moonlit rim on the ridge
                    rimv = math.exp(-((dip + ce - dy) / (2.0 * pix)))
                    out[j, i, 0] = 0.0085 + 0.004 * rimv
                    out[j, i, 1] = 0.0115 + 0.005 * rimv
                    out[j, i, 2] = 0.022 + 0.008 * rimv
                    dep[j, i] = 7000.0
                    skym[j, i] = 0.0
                    continue
                r, g, b = SK.sky_base(dx, dy, dz, S)
                mr, mg, mb = SK.moon_disk(dx, dy, dz, S, pix)
                out[j, i, 0] = r + mr
                out[j, i, 1] = g + mg
                out[j, i, 2] = b + mb
                dep[j, i] = 1e9
                skym[j, i] = 1.0
                continue
            skym[j, i] = 0.0
            tt = -oy / dy
            px = ox + dx * tt
            pz = oz + dz * tt
            fp = tt * pix / max(-dy, 0.02)
            sx, sz, var = wave_slope(px, pz, t, fp, Wv)
            nx = -sx
            ny = 1.0
            nz = -sz
            nn = math.sqrt(nx * nx + ny * ny + nz * nz)
            nx /= nn
            ny /= nn
            nz /= nn
            dn = dx * nx + dy * ny + dz * nz
            rx_ = dx - 2.0 * dn * nx
            ry_ = max(dy - 2.0 * dn * ny, 0.002)
            rz_ = dz - 2.0 * dn * nz
            rn = math.sqrt(rx_ * rx_ + ry_ * ry_ + rz_ * rz_)
            rx_ /= rn
            ry_ /= rn
            rz_ /= rn
            fres = 0.02 + 0.98 * (1.0 - max(-dn, 0.0)) ** 5
            r, g, b = SK.sky_base(rx_, ry_, rz_, S)
            # moon glitter: disk lobe widened by the unresolved slope variance
            cm = rx_ * S[6] + ry_ * S[7] + rz_ * S[8]
            th = math.acos(min(max(cm, -1.0), 1.0))
            R0 = S[26]
            tw2 = R0 * R0 + 4.0 * var + 1e-6
            gl = S[25] * 7.0 * (R0 * R0 / tw2) * math.exp(-th * th / tw2)
            r += gl * S[9]
            g += gl * S[10]
            b += gl * S[11]
            cr = r * fres + 0.0012
            cg = g * fres + 0.0022
            cb = b * fres + 0.0032
            # fires: specular streaks
            for li in range(LT.shape[0]):
                lx = LT[li, 0] - px
                ly = LT[li, 1]
                lz = LT[li, 2] - pz
                ll = math.sqrt(lx * lx + ly * ly + lz * lz)
                lx /= ll
                ly /= ll
                lz /= ll
                c2 = rx_ * lx + ry_ * ly + rz_ * lz
                th2 = math.acos(min(max(c2, -1.0), 1.0))
                w2 = LT[li, 7] * LT[li, 7] + 4.0 * var + 1e-6
                E = LT[li, 6] * (LT[li, 7] * LT[li, 7] / w2) * math.exp(-th2 * th2 / w2) * fres * 25.0
                cr += E * LT[li, 3]
                cg += E * LT[li, 4]
                cb += E * LT[li, 5]
            # the flare's warm pool: wave faces turned toward it catch its light (foam / subsurface)
            if FL[3] > 0.0:
                lx = FL[0] - px
                ly = FL[1]
                lz = FL[2] - pz
                d2 = lx * lx + ly * ly + lz * lz
                if d2 < FL[7]:
                    il = 1.0 / math.sqrt(d2)
                    ndl = (nx * lx + ny * ly + nz * lz) * il
                    if ndl > 0.0:
                        q = 1.0 - d2 / FL[7]
                        E = FL[3] * ndl / (d2 + 1.0) * q * q
                        cr += E * FL[4]
                        cg += E * FL[5]
                        cb += E * FL[6]
            dist = tt
            tr = math.exp(-dist / 9000.0)
            out[j, i, 0] = cr * tr + 0.012 * (1 - tr)
            out[j, i, 1] = cg * tr + 0.018 * (1 - tr)
            out[j, i, 2] = cb * tr + 0.040 * (1 - tr)
            dep[j, i] = dist


@njit(fastmath=True, cache=True)
def _refl_col(img, depth, sx, yh, L, sig, I, cr, cg, cb, t, seed, sc):
    """Short reflection column of a distant fire on the sea: starts at the horizon row yh, decays
    over L px, broken into twinkling wave-crest glints that wobble sideways."""
    H, W = img.shape[0], img.shape[1]
    ya = max(int(math.floor(yh)), 0)
    yb = min(int(math.ceil(yh + 4.0 * L)), H)
    for py in range(ya, yb):
        dd = py + 0.5 - yh
        if dd < 0.0:
            continue
        fall = math.exp(-dd / L)
        q = dd / sc
        g1 = gnoise2(q * 0.55 - t * 5.0, seed * 1.37, 11)
        g2 = gnoise2(q * 1.3 + t * 3.0, seed * 0.71 + 5.0, 12)
        gl = max(0.3 + 0.9 * g1 + 0.45 * g2, 0.0)
        gl = gl * gl
        xo = sx + 0.8 * sc * gnoise2(q * 0.35 + t * 2.0, seed + 3.3, 13)
        x0 = max(int(xo - 4.0 * sig), 0)
        x1 = min(int(xo + 4.0 * sig) + 1, W)
        for px in range(x0, x1):
            if depth[py, px] < 300.0:
                continue
            u = (px + 0.5 - xo) / sig
            w = I * fall * gl * math.exp(-0.5 * u * u)
            img[py, px, 0] += cr * w
            img[py, px, 1] += cg * w
            img[py, px, 2] += cb * w


def camera(frame, W=1920, H=804):
    t = CM.ftime(frame)
    tilt0 = math.degrees(math.atan((HORIZON_Y0 - 402.0) / F_FULL))
    push = keys(frame, [(START, 0.0), (END, 2.5)], ease=lambda u: u)
    bob = 0.20 * math.sin(2 * math.pi * t / 5.3) + 0.06 * math.sin(2 * math.pi * t / 2.1 + 1.0)
    tl = 0.30 * math.sin(2 * math.pi * t / 4.4 + 1.0)
    k = kick(t, CM.ftime(IGN), amp=1.4, freq=5.0, decay=5.0)
    return Cam((0.0, 2.3 + bob, push), YAW0 + 0.05 * k, tilt0 + tl + 0.06 * k, HFOV, W, H)


def boat_drawing():
    d = Drawing()
    d.new_group()
    # hull: sheer line rising to the bow (left); sits ~1.1 m above the water
    d.trap((0.3, -0.15), (0.3, 1.05), 3.6, 4.2, rnd=0.05, mat=14)
    d.tri((-3.9, 1.05), (-4.9, 1.6), (-3.2, 0.2), rnd=0.02, k=0.1, mat=14)
    d.capsule((-4.8, 1.55), (3.9, 1.15), 0.05, 0.05, k=0.03, mat=5)          # gunwale
    d.new_group()
    d.trap((1.8, 1.0), (1.8, 2.9), 0.95, 0.85, rnd=0.04, mat=0)               # wheelhouse
    d.trap((1.8, 2.85), (1.8, 3.05), 1.1, 1.05, rnd=0.02, mat=0)
    d.new_group()
    d.capsule((-0.6, 1.0), (-0.6, 6.4), 0.06, 0.04, mat=4)                    # mast
    d.capsule((-0.6, 6.3), (-4.7, 1.6), 0.008, 0.008, mat=5)                  # forestay
    d.capsule((-0.6, 6.3), (3.8, 1.3), 0.008, 0.008, mat=5)                   # backstay
    d.capsule((-0.6, 2.2), (2.6, 2.4), 0.04, 0.035, mat=4)                    # boom
    # mainsail furled along the boom: a lumpy canvas bundle with a couple of ties
    d.new_group()
    d.chain([(-0.5, 2.33), (0.3, 2.40), (1.1, 2.43), (1.9, 2.47), (2.45, 2.47)], 0.13, 0.07, k=0.08, mat=13,
            fuzz=0.025, ff=6.0)
    d.capsule((-0.55, 2.25), (-0.55, 3.1), 0.07, 0.035, k=0.08, mat=13, fuzz=0.012, ff=9.0)   # luff up the mast
    return d


def mirror_drawing(d, y0):
    """Copy of a Drawing mirrored about the horizontal line y = y0 (local metres)."""
    m = Drawing()
    m.rows = [r.copy() for r in d.rows]
    m.group = d.group
    for r in m.rows:
        typ = int(r[0])
        if typ == FG.ELLIPSE:
            r[2] = 2 * y0 - r[2]
            r[5] = -r[5]
        elif typ == FG.TRI:
            r[2], r[4], r[6] = 2 * y0 - r[2], 2 * y0 - r[4], 2 * y0 - r[6]
        else:
            r[2], r[4] = 2 * y0 - r[2], 2 * y0 - r[4]
    return m


def boat_reflection(img, cam, boat, bpos, lights, famb, t, scale, y_water=0.15):
    """Dark, wave-broken mirror image of the (firelit) hull under the waterline -- seats the boat in
    the water instead of on a bright band of sky reflection."""
    mb = mirror_drawing(boat, y_water)
    ml = []
    for L in lights:
        L = dict(L)
        if 'pos' in L:
            p = np.array(L['pos'], np.float64)
            p[1] = 2 * (bpos[1] + y_water) - p[1]
            L['pos'] = p
        else:
            L['dir'] = np.array(L['dir'], np.float64) * np.array([1, -1, 1])
        ml.append(L)
    H, W = img.shape[:2]
    xmin, xmax, ymin, ymax = mb.bounds()
    sx, sy, z = cam.project(bpos)
    ppm = cam.f / z
    x0 = int(max(0, sx + xmin * ppm - 6))
    x1 = int(min(W, sx + xmax * ppm + 6))
    ywl = sy - y_water * ppm
    y0 = int(max(0, math.floor(ywl)))
    y1 = int(min(H, sy - ymin * ppm + 6))
    if x1 - x0 < 4 or y1 - y0 < 4:
        return
    reg = (slice(y0, y1), slice(x0, x1))
    A = np.zeros_like(img[reg])
    B = np.ones_like(img[reg])
    dz = np.full(A.shape[:2], 1e9, np.float32)
    shift = (x0, y0)

    class _Sub:          # camera shim for rendering into the sub-rectangle
        f = cam.f
        right, fwd = cam.right, cam.fwd

        @staticmethod
        def project(P):
            a, b, c = cam.project(P)
            return a - shift[0], b - shift[1], c
    for buf in (A, B):
        FG.render(buf, dz, _Sub, mb, bpos, ml, amb=famb, t=t, mats=MATS, write_depth=False)
    cov = np.clip(1.0 - (B - A)[..., 0], 0.0, 1.0)
    # wave distortion: rows wobble sideways, more with distance below the waterline
    yy, xx = np.mgrid[0:y1 - y0, 0:x1 - x0].astype(np.float32)
    dd = (yy + y0 - ywl) / max(scale, 1e-3)                 # px below the waterline at 1920
    amp = (0.6 + 0.06 * dd) * scale
    wob = (np.sin(yy / scale * 0.9 + t * 5.0) * 0.6 + np.sin(yy / scale * 2.3 - t * 7.0 + xx / scale * 0.05) * 0.4)
    mx = (xx + amp * wob).astype(np.float32)
    my = yy.astype(np.float32)
    A = cv2.remap(A, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    cov = cv2.remap(cov, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    # broken into bands by the swell, and fading out with depth below the waterline
    band = 0.75 + 0.25 * np.sin(yy / scale * 0.55 + t * 3.0 + np.sin(xx / scale * 0.02))
    k = (0.82 * np.exp(-np.maximum(dd, 0.0) / 55.0) * band)[..., None]
    img[reg] = img[reg] * (1.0 - k * cov[..., None]) + k * A


# ------------------------------------------------------------------ the sailor + flare pose ---

ARM_K = [(START, 1.0), (1729, 1.0), (1731, 0.8), (1738, -0.25), (END, -0.3)]
LEAN_K = [(START, 0.1), (1729, 0.12), (1738, -0.05), (END, -0.05)]
SAILOR_OFF = np.array([-3.0, 1.1])      # feet relative to boat centre (screen frame)


def boat_state(frame):
    t = CM.ftime(frame)
    bh = swell_h(BOAT[0], BOAT[2], t, WAVES) * 0.9
    roll = 0.05 * math.sin(2 * math.pi * t / 4.1 + 0.6)
    return BOAT + np.array([0.0, bh - 0.15, 0.0]), roll


def sailor_pose(frame, cam):
    """Sailor drawing (mirrored: faces left, toward the coast) standing on the rolling deck, and the
    world positions of the flare head, his head and hand."""
    bpos, roll = boat_state(frame)
    arm = keys(frame, ARM_K, ease=smoother)
    lean = keys(frame, LEAN_K, ease=smoother)
    sail, pts = PP.torch_person('sailor', arm=arm, lean=lean)
    c, s_ = math.cos(roll), math.sin(roll)
    off = SAILOR_OFF
    offr = np.array([c * off[0] - s_ * off[1], s_ * off[0] + c * off[1]])
    sail.rotate(-roll)                  # rolls with the deck, about his feet (mirrored -> -roll)
    feet = PR.local_to_world(cam, bpos, offr[0], offr[1], 0.0)

    def w(p):
        q = np.array([c * (-p[0]) - s_ * p[1], s_ * (-p[0]) + c * p[1]])
        return PR.local_to_world(cam, feet, q[0], q[1], 0.0)
    return dict(bpos=bpos, roll=roll, sail=sail, feet=feet, flare=w(pts['torch_head']), head=w(pts['head']),
                hand=w(pts['hand']))


_FT = None


def flare_track():
    """Flare-head world position every 1/4 frame (particles and smoke are born where the hand was)."""
    global _FT
    if _FT is None:
        cam = camera(START)
        fr = np.arange(START - 4, END + 4.001, 0.25)
        P = np.array([sailor_pose(f, cam)['flare'] for f in fr])
        _FT = (fr, P)
    return _FT


def flare_at(times):
    fr, P = flare_track()
    f = np.asarray(times, np.float64) * CM.FPS
    return np.stack([np.interp(f, fr, P[:, k]) for k in range(3)], -1)


def flare_env(frame):
    """Hand-flare ignition (size, intensity, light): a bright catch on 1729, then on the 1730 hit the
    whoomp -- already bigger than its steady burn, overshooting to ~1.45x two frames later -- with a
    light flash (3.6x) that settles over ~10 frames."""
    u = frame - IGN
    if u < -1.5:
        return 0.0, 0.0, 0.0
    if u < -0.5:
        k = min(max((u + 1.5) / 0.5, 0.0), 1.0)
        return 0.22 * k, 1.3 * k, 0.35 * k
    rise = 1.0 - math.exp(-(u + 0.5) / 0.7)
    over = 0.5 * math.exp(-((u - 1.5) / 3.0) ** 2)
    size = 0.35 + 0.65 * rise + over
    inten = 1.0 + 1.1 * math.exp(-max(u, 0.0) / 2.5)
    light = 1.0 + 1.7 * math.exp(-max(u, 0.0) / 3.0)
    return size, inten, light


def flare_flicker(t):
    """Marine flares sputter: the fire's organic flicker plus a faster, smaller jitter."""
    return F.flicker(t, 41) * (1.0 + 0.06 * math.sin(2 * math.pi * 9.3 * t + 0.4)
                               + 0.04 * math.sin(2 * math.pi * 14.9 * t + 1.3))


# ------------------------------------------------------------------ flame (supersampled) ---

class _PatchCam:
    """Camera shim: projects into a supersampled local patch (origin x0,y0 in frame pixels)."""

    def __init__(self, cam, s, x0, y0):
        self.c, self.s, self.x0, self.y0 = cam, s, x0, y0
        self.f = cam.f * s

    def project(self, P):
        sx, sy, z = self.c.project(P)
        return (sx - self.x0) * self.s, (sy - self.y0) * self.s, z


_TG = F.tongue_table(6, 14, spread=0.6)


def flare_flame(img, cam, base, Hf, Rb, t, Ib, lean, ss=3, gamma=1.8):
    """Dense tongue flame, ss x ss supersampled in a local patch. The mt.fire bonfire kernel gives the
    shape and temperature field T (its debug mode); emission is re-mapped here with a flatter curve
    than the kernel's T^4.2 so the body sits in the saturated orange range (~Ib) with a gold-white
    root (~5 Ib) and visible red tips, instead of blown-white-or-invisible."""
    sx, sy, z = cam.project(base)
    if z <= 0.1 or Hf <= 0.01:
        return
    ppm = cam.f / z
    H, W = img.shape[:2]
    hw = (Rb * 1.5 + abs(lean) * 1.3 + 0.25 * Hf) * ppm + 3
    x0 = int(math.floor(sx - hw))
    x1 = int(math.ceil(sx + hw))
    y0 = int(math.floor(sy - Hf * 1.9 * ppm - 3))
    y1 = int(math.ceil(sy + 0.1 * Hf * ppm + 3))
    xa, xb, ya, yb = max(x0, 0), min(x1, W), max(y0, 0), min(y1, H)
    if xb <= xa or yb <= ya:
        return
    patch = np.zeros(((y1 - y0) * ss, (x1 - x0) * ss, 3), np.float32)
    pdep = np.full(patch.shape[:2], 1e9, np.float32)
    F.flame(patch, pdep, _PatchCam(cam, ss, x0, y0), base, Hf, Rb, t, seed=14, I=1.0, lean=lean, zbias=0.0,
            TG=_TG, min_px=1.0, debug=1)
    T = np.clip(patch[..., 0], 0.0, 1.0)
    # pinch the bonfire's wide flat base so the flame streams out of the flare head (teardrop root)
    pp = ppm * ss
    yy, xx = np.mgrid[0:patch.shape[0], 0:patch.shape[1]].astype(np.float32)
    X = (xx + 0.5 - (sx - x0) * ss) / pp
    v = np.maximum(((sy - y0) * ss - (yy + 0.5)) / pp / Hf, 0.0)
    wa = Rb * (0.28 + 3.0 * v)
    pinch = np.clip((wa - np.abs(X - lean * v * v)) / (0.35 * wa), 0.0, 1.0)
    T *= np.where(v < 0.25, pinch, 1.0)
    m = T > 0
    E = np.zeros_like(T)
    E[m] = Ib * (T[m] / 0.55) ** gamma
    col = F.bb_vec(T[m].astype(np.float64)).astype(np.float32)
    em = np.zeros_like(patch)
    em[m] = col * E[m][:, None]
    small = cv2.resize(em, (x1 - x0, y1 - y0), interpolation=cv2.INTER_AREA)
    img[ya:yb, xa:xb] += small[ya - y0:yb - y0, xa - x0:xb - x0]


# ------------------------------------------------------------------ particles + smoke ---

class Drips:
    """Burning dross dripping off the flare head: ballistic, cooling orange -> red, motion-blurred."""

    def __init__(self, seed, t0, t1, rate):
        rng = np.random.default_rng(seed)
        n = int((t1 - t0) * rate) + 1
        self.ts = t0 + np.sort(rng.random(n)) * (t1 - t0)
        self.p0 = flare_at(self.ts) + rng.normal(size=(n, 3)) * 0.025
        self.v0 = np.stack([0.25 + rng.normal(size=n) * 0.3, rng.uniform(-0.3, 0.9, n), rng.normal(size=n) * 0.25], 1)
        self.life = rng.uniform(0.45, 0.95, n)
        self.lum = 0.35 + 0.65 * rng.random(n) ** 2
        self.T0 = rng.uniform(0.72, 0.86, n)

    def render(self, img, depth, cam, t, gain, shutter=1.0 / 48.0, K=4):
        ok = (t + 0.5 * shutter - self.ts > 0) & (t - 0.5 * shutter - self.ts < self.life)
        idx = np.nonzero(ok)[0]
        if len(idx) == 0:
            return
        s = np.linspace(-0.5, 0.5, K) * shutter
        A = np.maximum((t + s)[None, :] - self.ts[idx, None], 0.0)
        P = self.p0[idx, None, :] + self.v0[idx, None, :] * A[..., None]
        P[..., 0] += 0.4 * A ** 2                           # wind drag
        P[..., 1] -= 4.9 * A ** 2
        sx, sy, z = cam.project(P)
        a = np.clip((t - self.ts[idx]) / self.life[idx], 0, 1)
        T = self.T0[idx] * (1.0 - 0.45 * a)
        inten = gain * self.lum[idx] * (T / 0.8) ** 4 * np.clip((1.0 - a) / 0.3, 0, 1) * np.clip(
            (t - self.ts[idx]) / 0.03, 0, 1)
        rad = np.full(len(idx), max(0.5 * cam.W / 1920.0, 0.3))
        F._draw_streaks(img, depth, sx.astype(np.float64), sy.astype(np.float64),
                        z.mean(axis=1).astype(np.float64), rad, F.bb_vec(T).astype(np.float64),
                        inten.astype(np.float64), 0.5, 200.0 * cam.W / 1920.0)


class FlareSmoke:
    """mt.fire.Smoke, but every puff is born at the flare head where it was at that instant."""

    def __init__(self, seed, t_start, t_end, rate=5.0, rise=1.5, wind=(0.9, 0.0, 0.0), life=5.0, r0=0.3,
                 growth=0.42, dens=0.8, jitter=0.12, lift=0.35):
        rng = np.random.default_rng(seed)
        n = int(max(1, (t_end - t_start + 0.5) * rate))
        self.ts = t_start + np.arange(n) / rate + rng.random(n) / rate * 0.8
        self.o = flare_at(self.ts) + np.array([0.0, lift, 0.0])
        self.off = rng.normal(size=(n, 3)) * jitter
        self.off[:, 1] = np.abs(self.off[:, 1]) * 0.5
        self.rise = rise * (0.7 + 0.6 * rng.random(n))
        self.wind = np.asarray(wind, np.float64)
        self.life = life * (0.7 + 0.6 * rng.random(n))
        self.r0 = r0 * (0.7 + 0.6 * rng.random(n))
        self.growth = growth
        self.dens = dens * (0.5 + 0.9 * rng.random(n))
        self.seed = rng.integers(0, 10000, n)
        self.ph = rng.random(n) * 6.28
        self.spin = rng.normal(size=n) * 0.4

    def render(self, img, depth, cam, t, fire_pos, fire_I, amb, albedo=0.35, zbias=0.3, light_falloff=1.2):
        age = t - self.ts
        ok = (age > 0) & (age < self.life)
        if not ok.any():
            return
        idx = np.nonzero(ok)[0]
        a = age[idx]
        up = self.rise[idx] * 1.6 * (1 - np.exp(-a / 1.6))
        pos = self.o[idx] + self.off[idx] * (1 + 0.8 * a[:, None])
        pos[:, 1] += up + 0.3 * a
        shear = 0.3 + 0.7 * np.clip(up / 3.0, 0, 1)
        pos[:, 0] += self.wind[0] * a * shear + 0.15 * np.sin(1.3 * a + self.ph[idx])
        pos[:, 2] += self.wind[2] * a * shear
        rad = self.r0[idx] + self.growth * a ** 0.85
        d = self.dens[idx] * np.clip(a / 0.25, 0, 1) * (1 - a / self.life[idx]) ** 1.6
        sx, sy, z = cam.project(pos)
        good = z > 0.3
        if not good.any():
            return
        sel = np.nonzero(good)[0][np.argsort(-z[good])]
        ppm = cam.f / z[sel]
        F._smoke(img, depth, sx[sel].astype(np.float64), sy[sel].astype(np.float64), z[sel].astype(np.float64),
                 (rad[sel] * ppm).astype(np.float64), d[sel].astype(np.float64), pos[sel].astype(np.float64),
                 a[sel].astype(np.float64), self.seed[idx][sel].astype(np.float64),
                 self.spin[idx][sel].astype(np.float64), np.asarray(fire_pos, np.float64), float(fire_I),
                 np.asarray(amb, np.float64), float(albedo), float(zbias), float(light_falloff),
                 rad[sel].astype(np.float64))


_ST = None
_SM = None
_SP = None
_DR = None


def _stars():
    global _ST
    if _ST is None:
        _ST = SK.make_stars(12000, 606, lum_scale=5.0)
    return _ST


def _smoke():
    global _SM
    if _SM is None:
        _SM = FlareSmoke(91, CM.ftime(IGN) + 0.02, CM.ftime(END) + 0.1, rate=14.0, wind=(3.2, 0, 0.3), rise=0.8,
                         life=3.0, r0=0.14, growth=0.75, dens=0.65, jitter=0.08)
    return _SM


def _sparks():
    global _SP
    if _SP is None:
        t0 = CM.ftime(IGN)
        ref = flare_at([CM.ftime(1740)])[0]
        sp = F.Sparks(93, ref, t0, CM.ftime(END) + 0.1, burst=170, rate=46, ember_rate=10, speed=(0.8, 2.6),
                      burst_speed=(2.5, 7.5), spread=0.6, life=(0.25, 0.85), ember_life=(1.4, 2.8),
                      wind=(2.8, 0.0, 0.3), buoy=4.0, updraft_h=1.6, drag=1.6, radius=0.06, I=55.0, turb=1.4,
                      curl=1.2)
        rng = np.random.default_rng(94)
        sp.p0 = flare_at(sp.ts) + np.array([0.0, 0.1, 0.0]) + rng.normal(size=(sp.n, 3)) * 0.035
        # warmer than a sparkler: yellow-orange heads cooling to red
        sp.T0 = np.where(sp.kind == 1, sp.T0 * 1.1, rng.uniform(0.78, 0.95, sp.n))
        _SP = sp
    return _SP


def _drips():
    global _DR
    if _DR is None:
        _DR = Drips(95, CM.ftime(IGN) + 0.15, CM.ftime(END) + 0.1, rate=9.0)
    return _DR


# ------------------------------------------------------------------ coast beacons ---

def _az_of_x(x):
    """Azimuth whose ridge-top fire lands at screen column x (1920 frame, reference camera)."""
    return math.atan((x - 960.0) / F_FULL) + math.radians(YAW0)


# nine ignite left -> right toward frame centre on a steady 3-frame ripple (1735-1759), all on
# screen and clear of the flare's glow; three more fill the gaps in the handle.
COAST_FIRES = [(_az_of_x(x), fr) for x, fr in
               [(62, 1735), (172, 1738), (286, 1741), (398, 1744), (508, 1747), (612, 1750), (712, 1753),
                (808, 1756), (898, 1759), (118, 1762), (344, 1764), (560, 1766)]]


def coast_fires(frame, cam, t):
    out = []
    for i, (az, fr) in enumerate(COAST_FIRES):
        if frame >= fr - 1:
            s2, i2, l2 = F.ignite_env(t, CM.ftime(fr))
            el = coast_elev(az) - math.sqrt(2.0 * 2.3 / 6.371e6) + 0.0008
            d = np.array([math.sin(az) * math.cos(el), math.sin(el), math.cos(az) * math.cos(el)])
            P = cam.pos * np.array([1, 0, 1]) + d * COAST_D + np.array([0, 2.3, 0])
            e = i2 * F.flicker(t, fr) * min(s2 / 0.5, 1.0)
            out.append((P, e, l2, az, i))
    return out


def draw_coast_fires(img, depth, cam, cf, t, scale):
    dip = -math.sqrt(2.0 * max(cam.pos[1], 0.1) / 6.371e6)
    for P, e, l2, az, i in cf:
        sx, sy, z = cam.project(P)
        pop = 1.0 + 1.6 * max(l2 - 1.0, 0.0)          # ignition flash
        rc = max(0.85 * scale, 0.6)
        # hot core + a tiny flame above it (vertical teardrop), then a small glow and a haze halo
        F.glow(img, depth, sx, sy, rc, 48.0 * e * pop * scale * scale, z=z - 10, zbias=0, col=GOLD)
        F.glow(img, depth, sx, sy - 1.3 * scale, rc, 22.0 * e * scale * scale, z=z - 10, zbias=0,
               col=np.array([1.0, 0.5, 0.15]))
        F.halo(img, depth, sx, sy, 4.5 * scale, 0.30 * e * pop, z=z - 10, zbias=0, col=EMBER)
        F.halo(img, depth, sx, sy - 2 * scale, 20 * scale, 0.028 * e * pop, z=z - 10, zbias=0, col=EMBER)
        # short glittering reflection column on the sea below it
        a_rel = az - cam.yaw
        yh = cam.cy + cam.shift - cam.f * math.tan(dip) / max(math.cos(a_rel), 0.5)
        _refl_col(img, depth, float(sx), float(yh), 9.0 * scale, max(0.9 * scale, 0.6), 1.3 * e * pop,
                  1.0, 0.52, 0.18, float(t), float(i * 7.1 + 3.0), float(scale))


# ------------------------------------------------------------------ render ---

def render(frame, scale=0.5, ss=1.25):
    W, H = int(round(1920 * scale)), int(round(804 * scale))
    t = CM.ftime(frame)
    cam = camera(frame, W, H)
    cami = cam.scaled(ss)
    size, inten, light = flare_env(frame)
    fl = flare_flicker(t)
    P = sailor_pose(frame, cam)
    bpos, roll, feet, flare_w = P['bpos'], P['roll'], P['feet'], P['flare']
    lit = light > 0
    flare_L = light * fl                               # light multiplier (1 = steady burn)
    cf = coast_fires(frame, cam, t)
    # specular streaks of the flare on the water: a tight glitter streak + a broad soft sheen
    LT = []
    FL = np.zeros(8)
    if lit:
        LT.append([flare_w[0], flare_w[1] + 0.2, flare_w[2], *F.FIRE_LIGHT, 2.8 * flare_L, 0.012, 0])
        LT.append([flare_w[0], flare_w[1] + 0.2, flare_w[2], *F.FIRE_LIGHT, 0.10 * flare_L, 0.05, 0])
        FL[:] = [flare_w[0], flare_w[1], flare_w[2], 2.6 * flare_L, *F.FIRE_LIGHT, 22.0 ** 2]
    LT = np.array(LT, np.float64).reshape(-1, 9) if LT else np.zeros((0, 9))
    # moon: disk + halo tamed so the fire is the brightest thing; the path keeps its own glitter
    S = SK.sky_params(zenith='#070B1C', horizon='#2A3866', moon_dir=MOON_DIR, moon_radius_deg=0.44, moon_I=2.8,
                      halo_I=0.032, halo_w=0.08, halo2_I=0.022, halo2_w=0.6, horizon_glow=0.3)
    S[25] = 8.0
    S[26] = math.radians(0.55)
    out = np.zeros((cami.H, cami.W, 3), np.float32)
    dep = np.zeros((cami.H, cami.W), np.float32)
    skm = np.zeros((cami.H, cami.W), np.float32)
    shade(cami.params(), S, WAVES, t, LT, FL, out, dep, skm)
    img = CM.downsample(out, W, H)
    depth = CM.depth_down(dep, W, H)
    skymask = CM.downsample(skm, W, H)
    SK.splat_stars(img, cam, _stars(), skymask, t=t, gain=0.8, scale=scale)
    draw_coast_fires(img, depth, cam, cf, t, scale)
    # boat + sailor, lit by the moon and (once lit) the flare -- the light is nudged toward the camera
    # so the faces of the flat puppets catch it, not just their edges
    moon_col = CM.lin('#9DB4D9')
    moon = dict(dir=-MOON_DIR * np.array([1, -1, 1]), col=moon_col, I=0.12)
    lights_b, lights_s = [moon], [moon]
    if lit:
        # flash on the hit, but the close sailor must not blow out to beige; the boat is farther from the
        # flare, so it gets a stronger (and more frontal) light to read the spill on bow, mast and sail
        fl_b = (1.0 + 0.45 * (light - 1.0)) * fl if light >= 1.0 else light * fl
        fl_s = (1.0 + 0.25 * (light - 1.0)) * fl if light >= 1.0 else light * fl
        Lp = flare_w + np.array([0, 0.12, 0])
        lights_b = [moon, dict(pos=Lp - cam.fwd * 0.9, col=F.FIRE_LIGHT, I=52.0 * fl_b, r0=0.5)]
        lights_s = [moon, dict(pos=Lp - cam.fwd * 0.45, col=F.FIRE_LIGHT, I=19.0 * fl_s, r0=0.35)]
    famb = np.array([0.008, 0.010, 0.018])
    boat = boat_drawing().rotate(roll)
    boat_reflection(img, cam, boat, bpos, lights_b, famb, t, scale)
    FG.render(img, depth, cam, boat, bpos, lights_b, amb=famb, t=t, mats=MATS)
    FG.render(img, depth, cam, P['sail'], feet, lights_s, amb=famb, t=t, seed=31, flipx=True, mats=MATS)
    if lit:
        _smoke().render(img, depth, cam, t, flare_w + np.array([0, 0.25, 0]), 5.5 * flare_L,
                        amb=(0.010, 0.013, 0.026), albedo=0.30, light_falloff=1.4)
        Hf = 1.2 * size
        flare_flame(img, cam, flare_w, Hf, 0.24, t, 2.2 * inten, lean=0.3 * min(size, 1.2))
        sx, sy, z = cam.project(flare_w + np.array([0, 0.06, 0]))
        # white-hot burning head
        F.glow(img, depth, sx, sy, max(1.5 * scale, 0.6), 700.0 * inten * min(size * 2.5, 1.0) * scale * scale,
               z=0, zbias=0, col=np.array([1.0, 0.86, 0.62]))
        # glow in the air / lens around it (drawn over everything in front: z=0)
        gx, gy, gz = cam.project(flare_w + np.array([0, 0.3 * Hf, 0]))
        gs = 0.55 + 0.45 * min(size, 1.3)
        F.halo(img, depth, gx, gy, 8.0 * scale * gs, 0.8 * flare_L, z=0, zbias=0, col=GOLD)
        F.halo(img, depth, gx, gy, 60.0 * scale * gs, 0.25 * flare_L, z=0, zbias=0, col=EMBER)
        F.halo(img, depth, gx, gy, 190.0 * scale, 0.028 * flare_L, z=0, zbias=0, col=EMBER)
        _sparks().render(img, depth, cam, t)
        _drips().render(img, depth, cam, t, 45.0)
    return img


FINISH = dict(exposure=1.0, bloom_strength=0.09, bloom_threshold=1.0, streak_strength=0.03, vignette_amount=0.25)
