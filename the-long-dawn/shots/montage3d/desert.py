"""DESERT (src 1520-1579, ignition 1540).

One idea: a lone robe on a knife-edge crest under the whole galaxy. The moon rakes a field of transverse
dunes from behind-right, so every crest splits into a lit windward face and a slip face in shadow. A
line of footprints climbs the S-curved crest from the foreground to the summit, where a robed, hooded
figure has shielded a torch from the wind all the way up; on the beat it thrusts it into the stone
beacon, the fire flares and streams downwind, lighting the sand, the robe and the sand blowing off the
crest. The Milky Way's core sits low over the dunes. Camera (as the original): slow truck right, a
small kick on the hit.

Terrain (venv, numpy): every dune is its own crest-conforming mesh (rows along the crest, columns
across it, one column exactly ON the crest) split into a windward and a slip-face object, so each
crest is a true knife edge (no heightfield aliasing, no lumpy mounds). A radial ground grid carries
the interdune floor; dune skirts dip under it so surfaces meet in clean intersection lines.
"""
import json
import math
import os

START, END, IGN = 1520, 1579, 1540
SAMPLES = 96
FPS = 24.0
HFOV = 42.0
EYE = 1.62                              # camera height over the sand
CAM_X0, CAM_X1 = 0.35, 1.35             # truck right
PITCH0, YAW0 = 5.2, 3.0
# the wind blows toward -x (over each crest into its slip face), a little +y
WIND = (-0.985, 0.17)
MOON_AZ, MOON_EL, MOON_I = 118.0, 24.0, 1.35
# hero crest: (y, x, H) control points (PCHIP). The summit plateau at y~32 carries the beacon.
HERO_CREST = [(-80.0, 9.0, 4.4), (-40.0, 3.6, 5.0), (-15.0, 0.9, 5.6), (0.0, -1.0, 6.0), (8.0, -2.3, 6.5),
              (15.0, -2.1, 7.4), (22.0, -0.2, 8.6), (28.0, 2.2, 9.6), (32.0, 3.8, 10.0), (37.0, 5.7, 9.8),
              (45.0, 8.5, 9.1), (60.0, 13.6, 7.7), (90.0, 21.0, 6.2), (140.0, 30.0, 5.2), (250.0, 43.0, 4.6),
              (500.0, 62.0, 4.2), (1000.0, 88.0, 3.8)]
SUMMIT_Y = 32.0
PLATEAU_R = 3.2
# far dunes: (x offset at y=0, H0, sinuosity amp, wavelength, phase, seed)
FAR_DUNES = [(-52.0, 6.5, 6.0, 170.0, 0.4, 1), (-118.0, 9.0, 9.0, 230.0, 2.1, 2), (-196.0, 12.5, 12.0, 300.0, 4.0, 3),
             (-286.0, 8.5, 10.0, 260.0, 1.2, 4), (-392.0, 14.0, 16.0, 380.0, 5.1, 5), (-520.0, 10.5, 14.0, 330.0, 3.3, 6),
             (-680.0, 16.0, 20.0, 450.0, 0.9, 7), (-880.0, 12.0, 18.0, 420.0, 2.7, 8), (-1120.0, 18.0, 26.0, 520.0, 4.4, 9),
             (78.0, 6.0, 5.0, 150.0, 3.9, 10), (150.0, 8.0, 8.0, 210.0, 1.7, 11), (236.0, 10.0, 11.0, 280.0, 5.6, 12),
             (340.0, 12.5, 14.0, 340.0, 0.2, 13), (470.0, 11.0, 16.0, 400.0, 2.9, 14)]
SLOPE_SLIP = 0.64                       # angle of repose ~32.6 deg
LW_K, LW_P = 3.8, 1.5                   # windward length = LW_K * H, profile power
FIG_KEYS_T0 = 1531.0
FINISH = dict(exposure=1.0, bloom_strength=0.075, bloom_threshold=0.8, streak_strength=0.0, vignette_amount=0.25)


def ftime(f):
    return f / FPS


def moon_dir(az=None, el=None):
    a, e = math.radians(MOON_AZ if az is None else az), math.radians(MOON_EL if el is None else el)
    return (math.sin(a) * math.cos(e), math.cos(a) * math.cos(e), math.sin(e))


def sky_dir(az, el):
    a, e = math.radians(az), math.radians(el)
    return (math.sin(a) * math.cos(e), math.cos(a) * math.cos(e), math.sin(e))


# ======================================================================= venv side ===

def flame_specs():
    import fireparts as FP

    def lean_fn(frame, size):              # the flame streams downwind (screen left), gusting
        t = ftime(frame)
        g = 0.8 + 0.2 * math.sin(t * 2.3) * math.sin(t * 1.1 + 1.0)
        return -0.5 * size * g

    return [FP.FlameSpec('beacon', Hf=2.0, Rb=0.33, seed=11, I=26.0, tongues=5, lean=-0.5, ppm=160,
                         t_ign=ftime(IGN), lean_fn=lean_fn),
            FP.FlameSpec('torch', Hf=0.3, Rb=0.045, seed=17, I=16.0, tongues=3, lean=-0.12, ppm=500, env=False)]


def timing(frames):
    import fireparts as FP
    out = dict(beacon=[], torch=[], ember=[])
    for f in frames:
        t = ftime(f)
        s, i, l = FP.ignite_env(t, ftime(IGN))
        out['beacon'].append((f, l * FP.flicker(t, 13)))
        out['torch'].append((f, FP.flicker(t, 7)))
        out['ember'].append((f, 0.0 if f < IGN else min(1.0, (f - IGN + 1) / 10.0) * (0.8 + 0.2 * FP.flicker(t, 3))))
    return out


# ---------------------------------------------------------------- terrain ---

def _pchip(pts):
    import numpy as np
    from scipy.interpolate import PchipInterpolator
    P = np.array(pts, np.float64)
    fx = PchipInterpolator(P[:, 0], P[:, 1], extrapolate=True)
    fh = PchipInterpolator(P[:, 0], P[:, 2], extrapolate=True)
    return fx, fx.derivative(), fh


def base_z(x, y):
    """The dune field's floor: broad undulation + earth curvature (seen from the origin)."""
    import numpy as np
    return (2.2 * np.sin(x / 170.0 + 0.7) * np.sin(y / 230.0 + 1.3) + 1.4 * np.sin((x + 0.6 * y) / 410.0 + 2.0)
            - (x * x + y * y) / (2.0 * 6.371e6))


def profile(wp, H):
    """Dune cross-section: windward (wp > 0) H*(1 - wp/Lw)^p, slip face at the angle of repose with a
    soft apron. Returns height above the floor."""
    import numpy as np
    wind = H * np.clip(1.0 - wp / (LW_K * H), 0.0, 1.0) ** LW_P
    slip = H + wp * SLOPE_SLIP
    slip = np.where(slip < 0.8, 0.8 * np.exp(np.minimum(slip - 0.8, 0.0) / 0.8), slip)
    return np.where(wp >= 0.0, wind, slip)


def _hero_fns():
    return _pchip(HERO_CREST)


def hero_height(x, y, fns=None):
    """Terrain height of the hero dune (with the summit plateau) at points (x, y) -- no floor."""
    import numpy as np
    fx, dfx, fh = fns or _hero_fns()
    xc, dx, H = fx(y), dfx(y), fh(y)
    wp = (x - xc) / np.sqrt(1.0 + dx * dx)
    h = profile(wp, H)
    return _plateau(x, y, h, fns)


def _plateau(x, y, h, fns=None):
    """A small rounded top at the summit (a knife edge can't carry a cairn and a person)."""
    import numpy as np
    fx, dfx, fh = fns or _hero_fns()
    sx, sh = float(fx(SUMMIT_Y)), float(fh(SUMMIT_Y))
    r = np.hypot(x - sx, (y - SUMMIT_Y) * 0.8)
    u = np.clip((PLATEAU_R * 1.9 - r) / (PLATEAU_R * 0.9), 0.0, 1.0)
    u = u * u * (3 - 2 * u)
    dome = sh - 0.22 - 0.035 * r * r
    return h + u * (np.minimum(dome, h + 0.25) - h) * (h > sh - 2.5)


def ground_at(x, y):
    """Final terrain height (max of floor and every dune) -- for placing things on the sand."""
    import numpy as np
    x = np.asarray(x, np.float64)
    y = np.asarray(y, np.float64)
    h = hero_height(x, y)
    for d in FAR_DUNES:
        fx, dfx, fh = _far_fns(d)
        xc, dx, H = fx(y), dfx(y), fh(y)
        h = np.maximum(h, profile((x - xc) / np.sqrt(1 + dx * dx), H))
    return base_z(x, y) + h


def _far_fns(d):
    """Analytic crest for a far dune: x_c(y), dx_c/dy, H(y) (tapering at the ends)."""
    import numpy as np
    x0, H0, A, lam, ph, seed = d
    k = 2 * math.pi / lam

    def fx(y):
        y = np.asarray(y, np.float64)
        return x0 + 0.13 * y + A * np.sin(k * y + ph) + 0.35 * A * np.sin(2.3 * k * y + 2 * ph + seed)

    def dfx(y):
        y = np.asarray(y, np.float64)
        return 0.13 + A * k * np.cos(k * y + ph) + 0.35 * A * 2.3 * k * np.cos(2.3 * k * y + 2 * ph + seed)

    def fh(y):
        y = np.asarray(y, np.float64)
        v = H0 * (1.0 + 0.22 * np.sin(y / (0.7 * lam) + seed) + 0.1 * np.sin(y / (0.23 * lam) + 2 * seed))
        return v * np.clip((y + 400.0) / 250.0, 0.05, 1.0)

    return fx, dfx, fh


def _rows(y0, y1, dy_of):
    ys = [y0]
    while ys[-1] < y1:
        ys.append(ys[-1] + dy_of(ys[-1]))
    return ys


def _cols(first, ratio, extent):
    c = [0.0]
    step = first
    while c[-1] < extent:
        c.append(c[-1] + step)
        step *= ratio
    return c


def dune_meshes(name, fns, y0, y1, Hmax, cam=(0.0, 0.0), near_dy=0.12, kdy=0.012, first=0.02, ratio=1.12,
                plateau=False):
    """Two crest-conforming meshes (windward, slip) for one dune. Vertex channels: u (arc length along the
    crest), wp (signed distance from the crest, + windward). Skirts dip under the floor at the feet."""
    import numpy as np
    fx, dfx, fh = fns

    def dy_of(y):
        xc = float(fx(y))
        d = math.hypot(xc - cam[0], y - cam[1])
        return float(np.clip(kdy * d, near_dy, 40.0))

    ys = np.array(_rows(y0, y1, dy_of))
    xc, dx, H = fx(ys), dfx(ys), fh(ys)
    ca = 1.0 / np.sqrt(1.0 + dx * dx)
    seg = np.hypot(np.diff(xc), np.diff(ys))
    u = np.concatenate([[0.0], np.cumsum(seg)])
    out = {}
    for side, sgn, ext in (('wind', 1.0, LW_K * Hmax * 1.05 + 2.0), ('slip', -1.0, Hmax / SLOPE_SLIP + 6.0)):
        w = sgn * np.array(_cols(first, ratio, ext))
        W, _ = np.meshgrid(w, ys)
        X = xc[:, None] + W
        Y = np.repeat(ys[:, None], len(w), 1)
        WP = W * ca[:, None]
        h = profile(WP, H[:, None])
        if plateau:
            h = _plateau(X, Y, h, fns)
        skirt = 0.45 * (1.0 - np.clip(h / 0.9, 0.0, 1.0)) ** 2
        Z = base_z(X, Y) + h - skirt
        V = np.stack([X, Y, Z], -1).reshape(-1, 3).astype(np.float32)
        U = np.repeat(u[:, None], len(w), 1)
        CH = np.stack([U.reshape(-1), WP.reshape(-1)], 1).astype(np.float32)
        nr, nc = X.shape
        i = np.arange(nc - 1)
        Q = []
        for j in range(nr - 1):
            a, b = j * nc, (j + 1) * nc
            q = np.stack([a + i, a + i + 1, b + i + 1, b + i], 1)
            Q.append(q if sgn > 0 else q[:, ::-1])
        out[side] = (V, np.vstack(Q).astype(np.int32), CH)
    return out


def ground_mesh(cam=(0.0, 0.0), r0=1.5, r1=45000.0, ratio=1.025, az=(-75.0, 75.0), daz=0.4):
    import numpy as np
    rs = [r0]
    while rs[-1] < r1:
        rs.append(rs[-1] * ratio)
    rs = np.array(rs)
    azs = np.radians(np.arange(az[0], az[1] + 1e-6, daz))
    R, A = np.meshgrid(rs, azs)
    X = cam[0] + R * np.sin(A)
    Y = cam[1] + R * np.cos(A)
    Z = base_z(X, Y)
    V = np.stack([X, Y, Z], -1).reshape(-1, 3).astype(np.float32)
    nr, nc = X.shape
    i = np.arange(nc - 1)
    Q = np.vstack([np.stack([j * nc + i, (j + 1) * nc + i, (j + 1) * nc + i + 1, j * nc + i + 1], 1)
                   for j in range(nr - 1)]).astype(np.int32)
    CH = np.zeros((len(V), 2), np.float32)
    return V, Q, CH


def write_mesh(path, V, Q, CH):
    import struct

    import numpy as np
    with open(path, 'wb') as f:
        f.write(struct.pack('<iii', len(V), len(Q), CH.shape[1]))
        f.write(np.ascontiguousarray(V, np.float32).tobytes())
        f.write(np.ascontiguousarray(Q, np.int32).tobytes())
        f.write(np.ascontiguousarray(CH, np.float32).tobytes())


# ------------------------------------------------------------------ figure ---

# poses (figures.fk parameters). She has carried the torch up the crest shielding it from the wind with
# her left hand; lowers it into the kindling on the beat, lets go, steps back from the flare with the
# forearm up against the heat and the blown sparks, then stands and watches, robe whipping.
FIG_KEYS = [
    (START, dict(lean=0.08, twist=0.05, head_pitch=-0.3, head_yaw=0.05, r_flex=0.55, r_abd=0.12, r_elbow=0.75,
                 l_flex=0.75, l_abd=0.05, l_elbow=1.65, r_hip=0.08, r_knee=0.12, l_hip=-0.04, l_knee=0.08,
                 stance=0.11, side_lean=0.03)),
    (1530, dict(lean=0.1, head_pitch=-0.35)),
    (1539.5, dict(lean=0.3, twist=0.12, head_pitch=-0.5, crouch=0.12, l_flex=0.45, l_elbow=1.2, r_hip=0.18,
                  r_knee=0.3)),
    (1544, dict(lean=-0.12, twist=-0.05, head_pitch=0.05, head_yaw=-0.3, crouch=0.05, r_flex=0.35, r_abd=0.3,
                r_elbow=0.55, l_flex=1.3, l_abd=0.35, l_elbow=1.95, r_hip=0.02, r_knee=0.12, l_hip=-0.08, l_knee=0.1)),
    (1552, dict(lean=-0.05, head_pitch=0.12, head_yaw=-0.15, l_flex=1.1, l_elbow=1.85)),
    (1566, dict(lean=0.0, twist=0.0, head_pitch=0.08, head_yaw=0.05, crouch=0.0, r_flex=0.18, r_abd=0.14,
                r_elbow=0.4, l_flex=0.35, l_abd=0.12, l_elbow=0.9)),
    (END, dict(lean=0.02, head_pitch=0.05, head_yaw=0.12, r_flex=0.15, l_flex=0.3, l_elbow=0.8)),
]
RELEASE = 1540.5


def robe_body(J, t, wind_l, gust):
    """A long hooded desert robe (mat 0), a wrapped scarf whose tail streams downwind (mat 1), skin
    (hands, mat 2). wind_l: unit wind in figure-local xy. All one smooth SDF body."""
    import numpy as np
    import figures as FG
    V, cone, ell = FG.V, FG.cone, FG.ell
    P = []
    Rp, Rc, Rh = J['R_pelvis'], J['R_chest'], J['R_head']
    pel, sp, ch, nk, hd = J['pelvis'], J['spine'], J['chest'], J['neck'], J['head']
    wx, wy = wind_l
    W = np.array([wx, wy, 0.0])

    def flap(z, ph):
        # the hem whips: a travelling wave downwind, stronger toward the hem and in gusts
        a = gust * (0.45 + 0.55 * max(0.0, 1.0 - z / 0.9))
        return a * (0.035 * math.sin(2 * math.pi * 2.3 * t + ph - 5.0 * z) + 0.015 * math.sin(2 * math.pi * 5.1 * t + 2 * ph))

    # torso: chest / belly / hips (robe over the body, loose)
    P.append(ell(ch + Rc @ V(0, 0.01, -0.02), (0.185, 0.14, 0.16), Rc, k=0.08, mat=0))
    P.append(ell(sp + Rc @ V(0, 0.01, 0.0), (0.18, 0.14, 0.15), Rc, k=0.09, mat=0))
    P.append(ell(pel + Rp @ V(0, 0.0, 0.02), (0.2, 0.155, 0.14), Rp, k=0.1, mat=0))
    # skirt of the robe: sections between the legs, flaring to the hem, blown downwind (pressed flat on
    # the windward side, bellying out on the lee side)
    kn = 0.5 * (J['l_knee'] + J['r_knee'])
    an = 0.5 * (J['l_ankle'] + J['r_ankle'])
    spread = np.linalg.norm(J['l_ankle'][:2] - J['r_ankle'][:2])
    for z_u, (rx, ry, rz) in ((0.72, (0.21, 0.165, 0.13)), (0.52, (0.225, 0.18, 0.13)), (0.33, (0.245, 0.195, 0.12)),
                              (0.16, (0.265, 0.21, 0.1)), (0.06, (0.285, 0.225, 0.06))):
        # interpolate the section centre between pelvis, knees and ankles by height
        if z_u > kn[2]:
            u = (z_u - kn[2]) / max(pel[2] - kn[2], 1e-3)
            c = kn + (pel - kn) * min(max(u, 0.0), 1.0)
        else:
            u = (z_u - an[2]) / max(kn[2] - an[2], 1e-3)
            c = an + (kn - an) * min(max(u, 0.0), 1.0)
        c = np.array([c[0], c[1], z_u])
        lower = max(0.0, 1.0 - z_u / 0.8)
        blow = gust * lower * (0.06 + 0.03 * math.sin(2 * math.pi * 1.7 * t + 3.0 * z_u))
        c = c + W * (blow + flap(z_u, 0.0))
        rxx = rx + 0.25 * spread * lower
        P.append(ell(c, (rxx, ry, rz), None, k=0.12, mat=0))
    # sleeves (wide, a little bell at the cuff), hands
    for s in ('l', 'r'):
        P.append(cone(J[s + '_sh'], J[s + '_el'], 0.072, 0.07, k=0.06, mat=0))
        cuff = J[s + '_wr'] + (J[s + '_el'] - J[s + '_wr']) * 0.05 + W * 0.015 * gust
        P.append(cone(J[s + '_el'], cuff, 0.07, 0.085, k=0.03, mat=0))
        P.append(cone(J[s + '_wr'] + (J[s + '_hand'] - J[s + '_wr']) * 0.2, J[s + '_hand'], 0.034, 0.03, k=0.015,
                      mat=2))
    # shoulders / yoke
    P.append(cone(J['l_sh'] + Rc @ V(0.02, 0, 0.0), J['r_sh'] + Rc @ V(-0.02, 0, 0.0), 0.078, 0.078, k=0.08, mat=0))
    # wrapped scarf round the neck (mat 1) + its tail streaming downwind from the back of the neck
    P.append(ell(nk + Rc @ V(0, 0.0, -0.01), (0.11, 0.1, 0.075), Rc, k=0.04, mat=1))
    a0 = nk + Rc @ V(0.03, -0.07, -0.03)
    prev = a0
    down = np.array([0.0, 0.0, -1.0])
    for i in range(1, 9):
        s_ = i / 8.0
        L = 0.085
        wob = gust * (0.16 * s_) * math.sin(2 * math.pi * 2.6 * t - 4.2 * s_) + 0.04 * s_ * math.sin(
            2 * math.pi * 6.3 * t - 7.0 * s_)
        perp = np.array([-wy, wx, 0.0])
        d = W * (0.75 + 0.2 * gust) + down * (0.45 - 0.35 * gust) + perp * wob + np.array([0, 0, 0.35 * wob])
        d /= np.linalg.norm(d)
        p = prev + d * L
        P.append(cone(prev, p, 0.05 * (1 - 0.45 * s_) + 0.012, 0.05 * (1 - 0.45 * (s_ + 0.125)) + 0.012, k=0.02,
                      mat=1))
        prev = p
    # head + hood: a deep hood (dome + peak + drape to the shoulders), face opening carved
    P.append(cone(ch + Rc @ V(0, 0, 0.02), nk + Rc @ V(0, 0.01, 0.04), 0.06, 0.055, k=0.03, mat=2))
    P.append(ell(hd, (0.076, 0.092, 0.105), Rh, k=0.02, mat=2))
    P.append(ell(hd + Rh @ V(0, -0.02, 0.02), (0.112, 0.128, 0.132), Rh, k=0.02, mat=0))
    P.append(ell(hd + Rh @ V(0, -0.07, 0.08) + W * 0.015 * gust, (0.06, 0.07, 0.07), Rh, k=0.05, mat=0))
    P.append(cone(hd + Rh @ V(0, -0.06, -0.03), nk + Rc @ V(0, -0.085, -0.06), 0.095, 0.105, k=0.07, mat=0))
    P.append(ell(hd + Rh @ V(0, 0.118, -0.012), (0.066, 0.05, 0.088), Rh, k=0.02, mat=2, op=1))
    return P


def fig_frame(f, wind_l):
    """Pose (joints), gust, torch grip/direction (figure-local) for frame f."""
    import numpy as np
    import figures as FG
    p = FG.interp_pose(f, FIG_KEYS)
    p['breath'] = 0.3 * math.sin(2 * math.pi * f / 80.0)
    J = FG.fk(p)
    t = ftime(f)
    gust = 0.75 + 0.25 * math.sin(t * 2.1) * math.sin(t * 0.9 + 1.3)
    return J, gust, t


def _figure_frames(frames, cache, beacon, root, yaw, basket_top):
    """SDF robe meshes (figure-local) + torch grip/dir + the step back, per frame."""
    import numpy as np
    import figures as FG
    d = os.path.join(cache, 'figure')
    os.makedirs(d, exist_ok=True)
    a = math.radians(yaw)
    ca, sa = math.cos(a), math.sin(a)

    def to_local(P):
        q = np.asarray(P, float) - root
        return np.array([ca * q[0] + sa * q[1], -sa * q[0] + ca * q[1], q[2]])

    wl = np.array([ca * WIND[0] + sa * WIND[1], -sa * WIND[0] + ca * WIND[1]])
    wl /= np.linalg.norm(wl)
    basket = to_local(np.array(basket_top) + np.array([0, 0, -0.12]))
    torch, step = {}, {}
    for f in frames:
        J, gust, t = fig_frame(f, wl)
        back = 0.42 * FG.ease((f - 1540.5) / 5.0)
        u_in = FG.ease((f - 1531.0) / 8.5) * (1 - FG.ease((f - 1540.5) / 3.0))
        hold = np.array([0.18, 0.75, 0.62])
        hold /= np.linalg.norm(hold)
        thr = basket - J['r_sh']
        thr /= np.linalg.norm(thr)
        thr = thr + np.array([0, 0, -0.25])
        thr /= np.linalg.norm(thr)
        tdir = hold * (1 - u_in) + thr * u_in
        tdir /= np.linalg.norm(tdir)
        grip_t = J['r_hand'] * (1 - u_in) + (basket - thr * 0.5) * u_in
        if u_in > 1e-4:
            FG.ik_arm(J, 'r', grip_t, pole=(0.5, -0.3, -0.8))
        if f >= RELEASE:
            grip, tdir = basket - thr * 0.5 + np.array([0, back, 0]), thr        # left in the basket
        else:
            grip = J['r_hand'].copy()
        # left hand: cupped round the torch head against the wind until the thrust, then up to the face
        head = grip + tdir * 0.62
        cup = (1 - FG.ease((f - 1533.0) / 5.0))
        if cup > 1e-3:
            tgt = head + np.array([-wl[0], -wl[1], 0.0]) * 0.1 + np.array([0, 0, 0.02])
            FG.ik_arm(J, 'l', J['l_hand'] * (1 - cup) + tgt * cup, pole=(-0.6, -0.3, -0.8))
        sh = FG.ease((f - 1540.0) / 3.0) * (1 - FG.ease((f - 1552.0) / 14.0))
        if sh > 1e-3:
            face = J['head'] + J['R_head'] @ np.array([0.04, 0.17, 0.0])
            FG.ik_arm(J, 'l', J['l_hand'] * (1 - sh) + face * sh, pole=(-0.6, -0.2, -0.9))
        torch[str(f)] = [list(map(float, grip)), list(map(float, tdir))]
        step[str(f)] = -back
        path = os.path.join(d, f'robe_{f:05d}.bin')
        if not os.path.exists(path) or os.environ.get('MT3D_REFIG'):
            Vv, Q, M = FG.mesh_sdf(robe_body(J, t, wl, gust), h=0.011, disp=(0.004, 7.0, 19.0, 3.0 * t))
            FG.write_mesh(path, Vv, Q, M)
    return dict(fig_dir=d, fig_torch=torch, fig_step=step)


def prep(frames, cache):
    import numpy as np
    geo = os.path.join(cache, 'geo')
    os.makedirs(geo, exist_ok=True)
    redo = bool(os.environ.get('MT3D_REDUNE'))
    info = dict(geo=geo, meshes=[])
    fns = _hero_fns()
    cam = (CAM_X0, 0.0)
    nv = 0

    def add(name, parts):
        nonlocal nv
        for side, (V, Q, CH) in parts.items():
            p = os.path.join(geo, f'{name}_{side}.bin')
            write_mesh(p, V, Q, CH)
            nv += len(V)
            info['meshes'].append(dict(name=f'{name}_{side}', path=p, kind=side, hero=name == 'hero'))

    stamp = os.path.join(geo, 'meshes.json')
    if redo or not os.path.exists(stamp):
        add('hero', dune_meshes('hero', fns, -60.0, 1000.0, 10.0, cam=cam, near_dy=0.08, kdy=0.01, first=0.015,
                                ratio=1.1, plateau=True))
        for k, d in enumerate(FAR_DUNES):
            f2 = _far_fns(d)
            add(f'far{k}', dune_meshes(f'far{k}', f2, -350.0, 4500.0, d[1] * 1.35, cam=cam, near_dy=0.5, kdy=0.014,
                                       first=0.05, ratio=1.15))
        V, Q, CH = ground_mesh(cam)
        p = os.path.join(geo, 'ground.bin')
        write_mesh(p, V, Q, CH)
        nv += len(V)
        info['meshes'].append(dict(name='ground', path=p, kind='ground', hero=False))
        json.dump(info['meshes'], open(stamp, 'w'))
        print(f'desert geometry: {nv / 1e6:.2f} M verts', flush=True)
    else:
        info['meshes'] = json.load(open(stamp))
    # camera heights over the sand along the truck
    xs = np.linspace(CAM_X0 - 0.2, CAM_X1 + 0.2, 13)
    info['cam_z'] = [[float(x), float(ground_at(x, 0.0)) + EYE] for x in xs]
    # the summit: beacon on the plateau just camera-side of the crest; the figure camera-left of it
    sx, sh = float(fns[0](SUMMIT_Y)), float(fns[2](SUMMIT_Y))
    b_xy = np.array([sx + 0.55, SUMMIT_Y + 0.25])
    info['beacon'] = [float(b_xy[0]), float(b_xy[1]), float(ground_at(b_xy[0], b_xy[1]))]
    fig_xy = b_xy + np.array([-0.95, -0.72])
    root = np.array([fig_xy[0], fig_xy[1], float(ground_at(fig_xy[0], fig_xy[1]))])
    info['figure_root'] = list(map(float, root))
    yaw = math.degrees(math.atan2(-(b_xy[0] - fig_xy[0]), b_xy[1] - fig_xy[1]))
    info['figure_yaw'] = yaw
    info['summit'] = [sx, SUMMIT_Y, float(ground_at(sx, SUMMIT_Y))]
    # arc length along the hero crest (mesh rows start at y=-60): footprints from 6 m behind the camera
    # to 1.2 m short of the figure
    yy = np.linspace(-60.0, 60.0, 4801)
    xx = fns[0](yy)
    uu = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(xx), np.diff(yy)))])
    info['path_u'] = [float(np.interp(-6.0, yy, uu)), float(np.interp(fig_xy[1] - 1.2, yy, uu))]
    basket_top = [info['beacon'][0], info['beacon'][1], info['beacon'][2] + 1.36]
    info.update(_figure_frames(frames, cache, np.array(info['beacon']), root, yaw, basket_top))
    # footprints: along the crest's windward shoulder from behind the camera to the figure
    return dict(desert=info)


_SPARKS = None


def post(frame, hdr, depth, cam, scene):
    import fireparts as FP
    global _SPARKS
    t = ftime(frame)
    fb = scene['fire_base']
    if frame >= IGN:
        if _SPARKS is None:
            _SPARKS = FP.ZSparks(53, (fb[0], fb[1], fb[2] + 0.35), ftime(IGN), ftime(END) + 0.1, burst=90, rate=26,
                                 ember_rate=9, wind=(2.6 * WIND[0], 2.6 * WIND[1], 0.25), radius=0.22, I=24.0,
                                 burst_speed=(2.0, 5.5), speed=(1.2, 3.5), spread=0.4, buoy=3.5)
        s, i, l = FP.ignite_env(t, ftime(IGN))
        FP.shimmer(hdr, cam, fb, 2.0 * s, 0.33, t, amp_px=1.0 * cam.W / 1920)
        _SPARKS.render(hdr, depth, cam, t)
    return hdr


# ==================================================================== Blender side ===

def _mesh_ch(name, path, mat):
    """desert write_mesh binary -> mesh; channel 0/1 -> UV map 'crest' (u along the crest, wp across)."""
    import array
    import struct

    import bpy
    with open(path, 'rb') as f:
        nv, nq, nch = struct.unpack('<iii', f.read(12))
        Va = array.array('f')
        Va.frombytes(f.read(nv * 12))
        Qa = array.array('i')
        Qa.frombytes(f.read(nq * 16))
        Ca = array.array('f')
        Ca.frombytes(f.read(nv * nch * 4))
    me = bpy.data.meshes.new(name)
    me.vertices.add(nv)
    me.vertices.foreach_set('co', Va)
    me.loops.add(nq * 4)
    me.loops.foreach_set('vertex_index', Qa)
    me.polygons.add(nq)
    me.polygons.foreach_set('loop_start', array.array('i', range(0, nq * 4, 4)))
    me.update(calc_edges=True)
    me.shade_smooth()
    if nch >= 2:
        uv = me.uv_layers.new(name='crest')
        buf = array.array('f', [0.0]) * (nq * 8)
        for li in range(nq * 4):
            vi = Qa[li]
            buf[2 * li] = Ca[vi * nch]
            buf[2 * li + 1] = Ca[vi * nch + 1]
        uv.data.foreach_set('uv', buf)
    me.materials.append(mat)
    return me


def build(job):
    import bpy
    from mathutils import Matrix, Vector

    from kit import core as C
    from kit import fire as FK
    from kit import figure as FGk
    from kit.nodes import new_material

    T = job['timing']
    D = T['desert']
    opts = job.get('opts', {})
    C.setup_render(scale=job['scale'], samples=job['samples'],
                   vol=dict(start=1.0, end=160.0, tile=4 if job.get('final') else 8, samples=64, dist=0.7,
                            shadow_samples=16),
                   mblur=bool(opts.get('mblur', True)), shutter=0.5, shadow_pool='256')
    md = moon_dir(opts.get('moon_az'), opts.get('moon_el'))
    # ------------------------------------------------------------------ atmosphere / sky
    hz = C.hexlin('#27335E')
    C.atmos_group(dict(a=0.0, Hs=100.0, h0=0.0, u=1.0 / 14000.0, amb=tuple(c * 1.05 for c in hz),
                       moon_dir=md, fwd=0.8, fwd_pow=3.0))
    core = sky_dir(-14.0, 2.5)         # the galactic core, low over the dunes left of the summit
    G = Vector(core).cross(Vector(sky_dir(38.0, 62.0))).normalized()
    C.sky_world(dict(zenith=C.hexlin('#070B1C'), horizon=hz, below=tuple(c * 0.8 for c in hz), moon_dir=md, moon_I=0.0,
                     halo=(0.02, 0.35), halo2=(0.012, 1.1), hglow=((0.05, 0.05, 0.07), 7.0),
                     stars=dict(gain=0.5, ext0=0.02, ext1=0.25, band=(tuple(G), 0.2, 2.5)),
                     milky=dict(pole=tuple(G), core=core, width=0.15, core_pow=4.0, gain=opts.get('mw_gain', 0.05),
                                core_gain=2.8, dust=0.85),
                     light_gain=0.9))
    C.sun('moon', md, C.MOON, MOON_I * opts.get('moon_gain', 1.0), angle_deg=0.6, volume=0.5)
    # ------------------------------------------------------------------ camera
    cam = C.make_camera('CAM', hfov=HFOV, clip=(0.1, 60000.0))
    czt = D['cam_z']

    def cam_z(x):
        for (x0, z0), (x1, z1) in zip(czt[:-1], czt[1:]):
            if x0 <= x <= x1:
                return z0 + (z1 - z0) * (x - x0) / (x1 - x0)
        return czt[-1][1]

    def ease_x(u):
        u = min(max(u, 0.0), 1.0)
        return u * 0.65 + 0.35 * u * u * (3 - 2 * u)

    def kick(t, t0, amp=1.0, freq=5.0, decay=5.0):
        if t < t0:
            return 0.0
        u = t - t0
        return amp * math.exp(-decay * u) * math.sin(2 * math.pi * freq * u)

    crop = opts.get('crop')
    if crop:
        cam.data.lens = 18.0 / (math.tan(math.radians(HFOV) / 2) * crop[2] / 1920.0)
    for f in range(START - 1, END + 2):
        u = (f - START) / (END - START)
        x = CAM_X0 + (CAM_X1 - CAM_X0) * ease_x(u)
        k = kick(ftime(f), ftime(IGN))
        shift = ((crop[0] - 960.0) / crop[2], -(crop[1] - 402.0) / crop[2]) if crop else (0.0, 0.0)
        C.key_camera(cam, f, (x, 0.0, cam_z(x) + 0.004 * k), YAW0 + 0.04 * k, PITCH0 + 0.05 * k, shift=shift)
    # ------------------------------------------------------------------ sand
    mats = _sand_materials(C, new_material, D)
    for m in D['meshes']:
        kind = m['kind']
        mat = mats['hero_wind'] if (m['hero'] and kind == 'wind') else mats[kind]
        me = _mesh_ch(m['name'], m['path'], mat)
        C.link_obj(bpy.data.objects.new(m['name'], me))
    # ------------------------------------------------------------------ beacon
    bx, by, bz = D['beacon']
    B = FK.beacon('beacon', (bx, by, bz - 0.12), seed=23, height=1.0, ember_frames=T['ember'], yaw=math.radians(35),
                  stone_base=(0.2, 0.17, 0.14))
    fb = B['fire_base']
    FK.flame_card('beacon_flame', fb, T['sprites']['beacon']['card'], T['sprites']['beacon']['first'], cam)
    FK.fire_lights('beacon', fb, 2.0, T['beacon'], power=opts.get('fire_power', 900.0), radius=0.4, cutoff=120.0)
    FK.smoke_plume('beacon_smoke', fb + Vector((0, 0, 0.8)), ftime(IGN), height=12.0, r0=0.3, spread=0.3, rise=1.3,
                   wind=(2.6 * WIND[0], 2.6 * WIND[1]), dens=1.2, albedo=0.3)
    FK.glow_haze('beacon_haze', fb + Vector((0, 0, 0.9)), 7.0, density=0.006, aniso=0.35, flat=0.7)
    # ------------------------------------------------------------------ blowing sand
    _streamers(C, FK, new_material, D, opts)
    # ------------------------------------------------------------------ figure
    rx_, ry_, rz_ = D['figure_root']
    yaw = D['figure_yaw']
    fmat = FGk.figure_material('robe', [(0.05, 0.034, 0.024), (0.09, 0.03, 0.022), (0.05, 0.032, 0.024),
                                        (0.03, 0.03, 0.03)], sheen=0.5, sheen_tint=(1.0, 0.85, 0.7), folds=0.5)
    fwd = Vector((-math.sin(math.radians(yaw)), math.cos(math.radians(yaw)), 0.0))
    seq = FGk.MeshSeq('robe', D['fig_dir'], fmat, Matrix.Identity(4), START - 1)
    torch = FGk.torch_obj('torch')
    heads = []
    for f in range(START - 1, END + 2):
        root = Matrix.Translation(Vector((rx_, ry_, rz_)) + fwd * D['fig_step'][str(f)]) @ \
            Matrix.Rotation(math.radians(yaw), 4, 'Z')
        seq.ob.matrix_world = root
        seq.ob.keyframe_insert('location', frame=f)
        seq.ob.keyframe_insert('rotation_euler', frame=f)
        grip, tdir = D['fig_torch'][str(f)]
        g = root @ Vector(grip)
        dvec = (root.to_3x3() @ Vector(tdir)).normalized()
        torch.location = g
        torch.rotation_quaternion = dvec.to_track_quat('Z', 'Y')
        torch.keyframe_insert('location', frame=f)
        torch.keyframe_insert('rotation_quaternion', frame=f)
        heads.append((f, g + dvec * 0.6))
    tf = FK.flame_card('torch_flame', heads[0][1], T['sprites']['torch']['card'], T['sprites']['torch']['first'], cam)
    tl = C.point('torch_L', heads[0][1], C.FIRE_LIGHT, 0.0, radius=0.05, cutoff=40.0)
    tt = {f: v for f, v in T['torch']}
    for f, P in heads:
        tf.location = P
        tf.keyframe_insert('location', frame=f)
        tf.scale = (1, 1, 1) if f < IGN + 1 else (0.001, 0.001, 0.001)
        tf.keyframe_insert('scale', frame=f)
        tl.location = P + Vector((0.05, 0, 0.2))
        tl.keyframe_insert('location', frame=f)
        C.key(tl.data, 'energy', f, 30.0 * tt.get(f, 1.0) * (1.0 if f < IGN else 0.0))
    return dict(fire_base=list(fb))


def per_frame(job, f):
    from kit import figure as FG
    FG.swap_all(f)


def _sand_materials(C, new_material, D):
    """Moonlit sand. Windward faces + floor: wind ripples (asymmetric, sinuous, merging; faded by pixel
    footprint so they never alias). Slip faces: smooth avalanche sand with faint grainflow tongues, no
    ripples. The hero's windward face also carries the footprints up the crest and the trampled sand
    round the beacon."""
    out = {}
    alb = (0.52, 0.43, 0.33)
    for kind in ('wind', 'hero_wind', 'slip', 'ground'):
        m, nb = new_material('sand_' + kind)
        P = nb.geo().outputs['Position']
        px, py, pz = nb.sep(P)
        dist = nb.camdata().outputs['View Distance']
        # tone: broad patches (coarser / finer sand, dust)
        n1 = nb.noise(P, scale=0.02, detail=3.0, rough=0.5)
        n2 = nb.noise(P, scale=0.4, detail=4.0, rough=0.55)
        tone = nb.mul(nb.madd(n1.outputs['Fac'], 0.3, 0.85), nb.madd(n2.outputs['Fac'], 0.12, 0.94))
        col = nb.colscale(alb, tone)
        height = None
        if kind in ('wind', 'hero_wind', 'ground'):
            # ripples: crests perpendicular to the wind, ~11 cm apart, sinuous, with defects; asymmetric
            # profile (gentle stoss, steep lee); fade out where a ripple spans < ~3 px
            wx, wy = WIND
            along = nb.add(nb.mul(px, -wx), nb.mul(py, -wy))
            warp = nb.noise(P, scale=0.35, detail=2.0, rough=0.5)
            ph = nb.div(nb.add(along, nb.mul(nb.sub(warp.outputs['Fac'], 0.5), 0.9)), 0.11)
            fr = nb.math('FRACT', ph)
            prof = nb.mn(nb.div(fr, 0.72), nb.div(nb.sub(1.0, fr), 0.28))
            amp_n = nb.noise(P, scale=0.08, detail=2.0)
            amp = nb.mul(nb.madd(amp_n.outputs['Fac'], 1.0, 0.25), 0.012)
            pxsize = nb.mul(dist, 2.0 * math.tan(math.radians(HFOV / 2)) / 1920.0)
            fade = nb.sstep(0.035, 0.012, pxsize)
            height = nb.mul(nb.mul(prof, amp), fade)
            if kind == 'hero_wind':
                # the knife edge stays smooth: ripples die out in the last ~0.4 m below the crest
                uv = nb.n('ShaderNodeUVMap')
                uv.uv_map = 'crest'
                cu, cw, _ = nb.sep(uv.outputs['UV'])
                height = nb.mul(height, nb.sstep(0.05, 0.45, cw))
                fp = _footprints(nb, cu, cw, D['path_u'])
                height = nb.add(height, fp)
                # trampled sand round the beacon
                bx, by, _bz = D['beacon']
                rr = nb.length(nb.comb(nb.sub(px, bx - 0.4), nb.sub(py, by - 0.3), 0.0))
                tr = nb.noise(P, scale=3.5, detail=3.0, rough=0.6)
                tram = nb.mul(nb.sstep(2.6, 1.0, rr), nb.mul(nb.sub(tr.outputs['Fac'], 0.5), 0.05))
                height = nb.add(nb.mul(height, nb.sstep(0.8, 2.4, rr)), tram)
        else:
            uv = nb.n('ShaderNodeUVMap')
            uv.uv_map = 'crest'
            cu, cw, _ = nb.sep(uv.outputs['UV'])
            # grainflow tongues: shallow lobes running down the slip face from the brink
            g = nb.noise(nb.comb(nb.div(cu, 2.2), nb.div(cw, 9.0), 0.0), scale=1.0, detail=3.0, rough=0.5)
            height = nb.mul(nb.sub(g.outputs['Fac'], 0.5), 0.05)
            col = nb.colscale(col, 1.04)
        bs = nb.principled(Base_Color=col, Roughness=0.93)
        bs.inputs['Specular IOR Level'].default_value = 0.25
        bs.inputs['Sheen Weight'].default_value = 0.35
        bs.inputs['Sheen Tint'].default_value = (1.0, 0.95, 0.85, 1.0)
        if height is not None:
            nb.link(nb.bump(height, 1.0, 1.0), bs.inputs['Normal'])
        nb.output(surface=C.fogged(nb, bs))
        out[kind] = m
    return out


def _footprints(nb, cu, cw, path_u):
    """Bump height of a line of footprints in UV (u along the crest, w across), from behind the camera
    to the summit: alternating left/right prints 0.36 m apart, a path wandering 0.3-0.8 m below the
    brink; each print a sloped-wall pit with a kicked-out rim, the toe dug deeper (climbing)."""
    STEP = 0.36
    u0, u1 = path_u                     # arc length of the path along the hero crest (behind camera .. figure)
    k = nb.math('FLOOR', nb.div(nb.sub(cu, u0), STEP))
    uc = nb.madd(k, STEP, u0 + STEP * 0.5)
    side = nb.sub(nb.mul(nb.math('MODULO', k, 2.0), 2.0), 1.0)
    path = nb.madd(nb.math('SINE', nb.mul(uc, 0.21)), 0.16, 0.52)
    jit = nb.white(nb.comb(k, 0.0, 0.0), dims='3D')
    wc = nb.add(nb.add(path, nb.mul(side, 0.095)), nb.mul(nb.sub(nb.sep(jit.outputs['Color'])[0], 0.5), 0.05))
    du = nb.sub(cu, nb.add(uc, nb.mul(nb.sub(nb.sep(jit.outputs['Color'])[1], 0.5), 0.06)))
    dw = nb.sub(cw, wc)
    e = nb.length(nb.comb(nb.div(du, 0.15), nb.div(dw, 0.065), 0.0))
    pit = nb.sstep(1.0, 0.55, e)
    rim = nb.mul(nb.sstep(1.45, 1.05, e), nb.sstep(0.85, 1.05, e))
    toe = nb.madd(nb.div(du, 0.15), 0.35, 1.0)
    inside = nb.mul(nb.sstep(u0, u0 + 0.5, cu), nb.sstep(u1, u1 - 0.5, cu))
    return nb.mul(nb.add(nb.mul(nb.mul(pit, toe), -0.045), nb.mul(rim, 0.012)), inside)


def _streamers(C, FK, new_material, D, opts):
    """Sand blowing off the crest near the summit: a thin sheet leaving the brink and streaming downwind
    over the slip face, torn into wisps (advected noise), fading as it spreads. Lit by moon and fire."""
    import bpy
    sx, sy, sh = D['summit']
    V = [(x, y, z) for z in (-1.5, 3.5) for y in (-16.0, 16.0) for x in (-9.0, 1.0)]
    F = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
    ob = C.mesh_obj('streamers', V, F, smooth=False)
    ob.location = (sx, sy - 6.0, sh)
    m, nb = new_material('streamers_m')
    t = FK.time_value(nb)
    P = nb.geo().outputs['Position']
    px, py, pz = nb.sep(P)
    # distance downwind of the crest (the crest runs ~ along y here, tilted)
    xc = nb.madd(nb.sub(py, sy), 0.36, sx)
    down = nb.mul(nb.sub(xc, px), 1.0)                      # metres downwind (toward -x) of the brink
    zc = nb.madd(nb.pw(nb.sub(py, sy), 2.0), -0.012, sh)    # crest height falls away from the summit
    above = nb.sub(pz, nb.sub(zc, nb.mul(nb.mx(down, 0.0), 0.35)))
    # sheet: rises a little as it leaves the brink, thickens and thins out downwind
    thick = nb.madd(nb.mx(down, 0.0), 0.12, 0.12)
    sheet = nb.exp(nb.mul(nb.pw(nb.div(nb.sub(above, nb.mul(nb.mx(down, 0.0), 0.08)), thick), 2.0), -1.0))
    reach = nb.mul(nb.sstep(-0.3, 0.4, down), nb.exp(nb.mul(nb.mx(down, 0.0), -0.28)))
    ends = nb.sstep(16.0, 8.0, nb.math('ABSOLUTE', nb.sub(py, sy)))
    adv = nb.comb(nb.madd(t, 3.2, px), nb.madd(t, -0.5, py), pz)
    n1 = nb.noise(nb.comb(nb.mul(nb.sep(adv)[0], 0.5), nb.mul(nb.sep(adv)[1], 1.5), nb.mul(pz, 1.5)), scale=1.4,
                  detail=4.0, rough=0.6, dims='4D', w=nb.mul(t, 0.7))
    wisp = nb.pw(nb.clamp01(nb.madd(n1.outputs['Fac'], 2.6, -1.05)), 1.4)
    dens = nb.mul(nb.mul(nb.mul(nb.mul(sheet, reach), ends), wisp), opts.get('streamer_dens', 0.35))
    pv = nb.n('ShaderNodeVolumePrincipled')
    nb.set(pv.inputs['Density'], dens)
    pv.inputs['Color'].default_value = (0.75, 0.66, 0.55, 1.0)
    pv.inputs['Absorption Color'].default_value = (0.0, 0.0, 0.0, 1.0)
    pv.inputs['Anisotropy'].default_value = 0.55
    nb.output(volume=pv)
    ob.data.materials.append(m)
    ob.visible_shadow = False
    return ob
