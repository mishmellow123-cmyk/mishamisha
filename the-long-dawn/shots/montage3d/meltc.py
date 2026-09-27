"""C22 THE UNMAKING: THE MELT (lane MONTAGE-MELT). C frames 5360-5519, Cycles (the farm's h100-1 GPUs).

  farm look-dev:  python3 the-long-dawn/cloud/farm.py the-long-dawn/cloud/jobs/meltC.json --test 4
  local (tiny):   MT3D_ENGINE=CYCLES python render.py meltc --frames 5420 --scale 0.25 --samples 16 --out t_m0
  local motion:   MELT_PREVIEW=1 python render.py meltc --range 5360-5519 --step 4 --scale 0.25 --out t_mp
                  (Workbench: the band's motion and melt only, seconds per frame)

The beat (BIBLE_V3 C22 + the director's H5 calls; sync = music/v3/barmap_C.json):
* 5360 the white heart of the fire everyone lit. Her fingers are forced open (ACCORD-3's P2 handle, top-down); here the
  Ring falls in at the top of frame and strikes the hearth stone (5363): a crisp drop, one hop, a coin's rattle, still
  by 5381. Its letters are awake (the fire has had it in her fist since 5200): the script of fire, deep orange-red.
* 5388-5433 it softens and slumps: the wall sags and leans, the circle goes out of true, the far side (toward the
  fire's heart) sags most, and the letters ride the metal and stay letters. 5420 (68 b4) the letters flare once.
* 5433-5440 the breath: nothing moves. 5440 (69 b1) the letters go out and, on the cadence, the closed loop breaks at
  the back: its two ends run forward round the band and gather into one bead at the front (5440-5452).
* 5452-5476 the bead: molten gold, trembling, a mirror for the fire. 5476-5519 the hearth flares to (near) full white.
Never: a doubled rim (the band never spreads into a pool round itself), a ring seen flat from above (a plughole), white
letter-blobs, haze. Dark and crisp; the only whiteout is the flare at the end.

The band is the canonical Ring (== ringc.py: R_in 9.4 mm, 2.3 mm thick, 5.2 mm wide, superellipse 2.8) with the
script of fire (assets/ring/inscription_{outer,inner}.png, mapping in inscription.json). ringc.py and glove.py belong
to MONTAGE-3D-4: this file reads only the Ring asset and the shared kit.

Geometry: ONE mesh for the whole shot, an open tube round the band (N_S+1 columns over s = -1..1, both ends at the
back where it will break, flat-shaded end fans hidden inside the metal until then) x N_A rows round the section.
The venv (numpy) computes every frame's vertex positions (`melt_verts`) -> cache/meltc/data/melt_%05d.bin; Blender
loads them as shape keys keyed 0-1-0 (linear), so Cycles gets true deformation blur and the UVs (the letters) ride
the metal. Deliveries: renders/ring_C (RGB) and renders/ring_C_mask (the 'ringmask' AOV: the metal's coverage, for
ACCORD-3's AC3 composite).
"""
import array
import math
import os

FPS = 24.0
SAMPLES = 192
PREP_ONLY_FRAMES = True
START, END = 5360, 5519
HERE = os.path.dirname(os.path.abspath(__file__))
ADIR = os.path.abspath(os.path.join(HERE, '..', '..', 'assets', 'ring'))
PREVIEW = bool(os.environ.get('MELT_PREVIEW'))

# ---- the canonical band (== ringc.py)
R_IN, THICK, WIDTH, SQ = 0.0094, 0.0023, 0.0052, 2.8
R_MID = R_IN + 0.5 * THICK
R_OUT = R_IN + THICK
HW = 0.5 * WIDTH
N_S, N_A = 480, 48

# ---- the beat (C frames)
B = dict(drop=5360.0, hit=5363.25, flat=5365.0, land=5366.8, still=5381.0, soft0=5388.0, flare=5420.0,
         neck0=5427.0, breath=5433.0, out=5440.0, merge=5451.0, bead=5458.0, white0=5476.0)

# the band's FRONT (the arc that faces the lens once it lies still): the long seven-letter word of the script
# (outer strip u 0.86-0.96) runs across it; the break is diametrically behind, toward the fire's heart
U_FRONT = 0.880
TH_F = 2.0 * math.pi * U_FRONT
YAW = -0.5 * math.pi - TH_F                  # object yaw that turns the band's front toward the lens (world -y)
# the melt's own axis sits 22 deg to the right of the lens axis: the break (behind), the two runs and the bead are
# never mirror-symmetric in frame (two equal lobes read as a face or a moustache)
DELTA = math.radians(22.0)
TH_C = TH_F + DELTA
TH_B = TH_C + math.pi

# ---- the bead (a drop of gold: ~11 mm across, a little flattened on the stone; the rest went into the fire)
BEAD_A, BEAD_C, BEAD_R = 0.0060, 0.0027, 0.0074   # horizontal semi-axis, half-height, centre's distance from the axis
SIG_MAX = 0.85                                    # how far the band slumps before the breath (1 = to a sausage)

_F = dict(exposure=0.62, bloom_strength=0.022, bloom_threshold=2.2, streak_strength=0.0, vignette_amount=0.30,
          lift=0.0015)
FINISH = _F


# ======================================================================= both sides (pure python) ===

def _cl(x, a=0.0, b=1.0):
    return min(max(x, a), b)


def _ss(u):
    u = _cl(u)
    return u * u * (3.0 - 2.0 * u)


def _sm(u):
    u = _cl(u)
    return u * u * u * (u * (6.0 * u - 15.0) + 10.0)


def _eout(u, p=3.0):
    u = _cl(u)
    return 1.0 - (1.0 - u) ** p


def letters(f):
    """Emission of the script of fire (1 = awake). Awake from the first frame; the fire works it brighter; one flare
    on 68 b4 (5420); held through the breath; out on 69 b1 (gone by 5441.5)."""
    base = 1.0 + 0.28 * _ss((f - 5381.0) / 38.0)
    tw = 1.0 + 0.045 * math.sin(f * 1.73) * math.sin(f * 0.61 + 1.0)
    if f < B['hit'] + 3.0:
        tw *= 1.0 - 0.35 * _ss((f - B['hit']) / 0.8) * (1.0 - _ss((f - B['hit'] - 1.0) / 2.0))  # the strike jars it
    if B['breath'] <= f:
        tw = 1.0
    fl = 1.55 * math.exp(-max(0.0, f - (B['flare'] + 0.8)) / 5.5) * _ss((f - (B['flare'] - 1.6)) / 2.4)
    return (base * tw + fl) * (1.0 - _ss((f - (B['out'] - 0.6)) / 2.1))


def flare_col(f):
    """0 = the deep script colour, 1 = the flare's brighter orange (never white)."""
    return _cl(math.exp(-max(0.0, f - (B['flare'] + 0.8)) / 4.5) * _ss((f - (B['flare'] - 1.6)) / 2.4))


def hot(f):
    """The metal's own incandescence (blackbody strength): barely there while it lies; a dull red as it slumps
    (mostly at the back, see the shader); molten orange once it runs; up with the white flare."""
    v = 0.012 + 0.10 * _ss((f - B['soft0']) / (B['breath'] - B['soft0']))
    v += 0.16 * _ss((f - B['out'] - 1.0) / 11.0) + 0.05 * _ss((f - B['merge']) / 20.0)
    v += 1.6 * _ss((f - B['white0']) / (END - B['white0'])) ** 1.6
    return v


def hot_temp(f):
    return 1180.0 + 180.0 * _ss((f - B['soft0']) / 50.0) + 420.0 * _ss((f - B['out']) / 12.0) + \
        300.0 * _ss((f - B['white0']) / (END - B['white0']))


def liquid(f):
    """0 = solid polished gold, 1 = liquid (a mirror; the polish marks gone)."""
    return _ss((f - (B['out'] - 1.5)) / 5.0)


def fire(f):
    """The fire's light (lamps, flame cards): the white heart, alive; a lull in the breath; it takes the Ring on the
    cadence; the flare to white from 5476."""
    fl = 1.0 + 0.07 * math.sin(f * 0.93 + 0.4) + 0.05 * math.sin(f * 2.31 + 1.1) + 0.03 * math.sin(f * 4.7)
    lull = _ss((f - (B['breath'] - 3.0)) / 3.0) * (1.0 - _ss((f - B['out']) / 2.0))
    fl = fl * (1.0 - lull) + 0.93 * lull
    swell = 0.16 * _ss((f - B['out']) / 3.0) * math.exp(-max(0.0, f - B['out'] - 3.0) / 10.0)
    w = _ss((f - B['white0']) / (END - B['white0']))
    return (fl + swell) * (1.0 + 11.0 * w ** 1.8)


def white(f):
    return _ss((f - B['white0']) / (END - B['white0']))


# ---- quaternions (w, x, y, z) for the drop, pure python
def _qa(ax, ang):
    s = math.sin(0.5 * ang)
    n = math.sqrt(sum(c * c for c in ax)) or 1.0
    return (math.cos(0.5 * ang), ax[0] / n * s, ax[1] / n * s, ax[2] / n * s)


def _qm(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (aw * bw - ax * bx - ay * by - az * bz, aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx, aw * bz + ax * by - ay * bx + az * bw)


def _qv(q, v):
    w, x, y, z = q
    p = (0.0, v[0], v[1], v[2])
    r = _qm(_qm(q, p), (w, -x, -y, -z))
    return r[1:]


DROP_H = 0.030            # the band's lowest point at 5360, m above the stone (her palm is just above frame)
TILT0 = math.radians(24.0)
GAM0 = math.radians(28.0)     # the tilt's axis at the strike (the edge that hits first: front left)
SPIN0 = math.radians(22.0)    # yaw still to come at the strike (it skates round a little as it settles)
SKID = (0.0021, -0.0012)      # where it strikes, relative to where it comes to rest (m)


def _yaw_at(f):
    if f <= B['hit']:
        return YAW + SPIN0 + math.radians(9.0) * (B['hit'] - f) / (B['hit'] - B['drop'])
    return YAW + SPIN0 * math.exp(-(f - B['hit']) / 4.2) * (1.0 - _ss((f - B['still'] + 3.0) / 3.0))


def _skid_at(f):
    if f <= B['hit']:
        return SKID
    k = math.exp(-(f - B['hit']) / 3.0) * (1.0 - _ss((f - B['still'] + 4.0) / 4.0))
    return (SKID[0] * k, SKID[1] * k)


def _contact_z(alpha):
    """Centre height of a band resting on its edge, tilted by alpha."""
    return R_OUT * math.sin(abs(alpha)) + HW * math.cos(alpha)


def ring_state(f):
    """(centre xyz, quaternion) of the band at (fractional) frame f; the centre = mid-width on the axis."""
    yaw = _yaw_at(f)
    sx, sy = _skid_at(f)
    if f <= B['hit']:                                               # the fall: accelerating, tipped, turning
        t_all = (B['hit'] - B['drop']) / FPS
        t = (f - B['drop']) / FPS
        g = 2.0 * DROP_H / (t_all * t_all)
        alpha = TILT0 + math.radians(7.0) * (B['hit'] - f) / (B['hit'] - B['drop'])
        zc = _contact_z(TILT0) + DROP_H - 0.5 * g * t * t
        gam = GAM0 - math.radians(6.0) * (B['hit'] - f) / (B['hit'] - B['drop'])
    elif f <= B['flat']:                                            # the strike: it slaps down flat on its edge
        u = (f - B['hit']) / (B['flat'] - B['hit'])
        alpha = TILT0 * (1.0 - _ss(u) ** 0.8)
        zc = _contact_z(alpha)
        gam = GAM0
    elif f <= B['land']:                                            # one small hop, landing tipped the other way
        u = (f - B['flat']) / (B['land'] - B['flat'])
        alpha = math.radians(4.5) * math.sin(math.pi * u * 0.5)
        zc = _contact_z(alpha) + 0.0011 * 4.0 * u * (1.0 - u)
        gam = GAM0 + math.pi
    elif f < B['still']:                                            # the rattle: an Euler's disk winding down
        t = f - B['land']
        a0 = math.radians(4.5)
        tau = 4.3
        alpha = a0 * math.exp(-t / tau) * (1.0 - _ss((f - B['still'] + 2.5) / 2.5))
        # precession speeds up as the tilt dies: gamma = gam_land + integral of Omega dt
        om0 = 2.0 * math.pi * 1.9 / FPS                             # rad per frame at a0
        gam = GAM0 + math.pi + om0 * 2.0 * tau * (math.exp(t / (2.0 * tau)) - 1.0)
        zc = _contact_z(alpha)
    else:
        return (0.0, 0.0, HW), _qa((0.0, 0.0, 1.0), YAW)
    q = _qm(_qa((math.cos(gam), math.sin(gam), 0.0), alpha), _qa((0.0, 0.0, 1.0), yaw))
    return (sx, sy, zc), q


def cam_state(f):
    """Camera: a low macro across the stone, 12 deg up, easing in; the band's front in focus, then the bead."""
    el = math.radians(8.0)
    d = 0.170 - 0.022 * _sm((f - 5364.0) / 72.0) - 0.004 * _ss((f - 5446.0) / 60.0)
    drift = 0.0012 * _ss((f - START) / (END - START))
    tgt = (drift, -0.0030, 0.0050 + 0.0034 * (1.0 - _sm((f - 5361.0) / 14.0)))
    pos = (tgt[0] - 0.0009 * _ss((f - START) / 160.0), tgt[1] - d * math.cos(el), tgt[2] + d * math.sin(el))
    front = (0.0, -R_OUT, 0.0030)
    rb = BEAD_R + BEAD_A * 0.92
    bead = (rb * math.sin(DELTA), -rb * math.cos(DELTA), 0.0022)
    k = _ss((f - B['out'] - 2.0) / 12.0)
    foc = tuple(front[i] + (bead[i] - front[i]) * k for i in range(3))
    return pos, tgt, foc


# ======================================================================= venv side (numpy) ===

def flame_specs():
    """The hearth's flames (the film's bonfire sprites), rooted in the burning kindling 7-17 cm behind the band:
    at this distance their tongues, not their roots, fill the upper frame."""
    if PREVIEW:
        return []
    import fireparts as FP
    kw = dict(I=24.0, env=False)
    return [FP.FlameSpec('kA', Hf=0.075, Rb=0.022, seed=43, tongues=5, lean=0.002, ppm=3400, **kw),
            FP.FlameSpec('kB', Hf=0.105, Rb=0.030, seed=71, tongues=6, lean=-0.003, ppm=3000, **kw),
            FP.FlameSpec('kC', Hf=0.140, Rb=0.042, seed=19, tongues=7, lean=0.003, ppm=2600, **kw),
            FP.FlameSpec('kD', Hf=0.050, Rb=0.014, seed=5, tongues=4, lean=-0.001, ppm=4000, **kw)]


def timing(frames):
    return {}


def _sgnpow(x, p):
    import numpy as np
    return np.sign(x) * np.abs(x) ** p


def _smax(a, b, k):
    """Smooth maximum (polynomial, width k)."""
    import numpy as np
    h = np.clip(0.5 + 0.5 * (a - b) / k, 0.0, 1.0)
    return b + (a - b) * h + k * h * (1.0 - h)


def _grid():
    import numpy as np
    s = np.linspace(-1.0, 1.0, N_S + 1)
    a = 2.0 * np.pi * np.arange(N_A) / N_A
    return np.meshgrid(s, a, indexing='ij')             # (N_S+1, N_A) each


def _sigma_g(f):
    return SIG_MAX * _ss((f - B['soft0']) / (B['breath'] - B['soft0'])) ** 1.25


def _section(S, A, f, sig_g=None):
    """The slumped section of every vertex at frame f (frozen at the breath): returns (rho, zeta, Rc, th, wfar)
    = offset from the centreline (radial, up from the stone), the centreline radius, its angle, the far weight."""
    import numpy as np
    sg = _sigma_g(min(f, B['breath'])) if sig_g is None else sig_g
    th = TH_C + np.pi * S
    wfar = 0.5 * (1.0 + np.cos(th - TH_B))
    sig = sg * (0.42 + 1.05 * wfar)
    n = SQ + (2.15 - SQ) * np.clip(1.5 * sig, 0.0, 1.0)
    ca, sa = np.cos(A), np.sin(A)
    cx = _sgnpow(ca, 2.0 / n)
    cz = _sgnpow(sa, 2.0 / n)
    b = 1.0 - 0.55 * sig                                         # the bottom flattens onto the stone
    cz = np.where(cz < 0.0, cz * b, cz)
    cx = cx * (1.0 + 0.24 * sig * np.clip(-sa, 0.0, 1.0))         # ... and spreads
    H = WIDTH * (1.0 - 0.40 * sig)
    T = THICK * (1.0 + 0.62 * sig)
    rho = 0.5 * T * cx
    zeta = H * (cz + b) / (1.0 + b)
    lam = sig * (0.20 + 0.24 * wfar)                              # the wall leans out as it sags (front: toward us)
    cl, sl = np.cos(lam), np.sin(lam)
    rho, zeta = rho * cl + zeta * sl, -rho * sl + zeta * cl
    zeta = zeta - zeta.min(axis=1, keepdims=True)                 # it still sits on the stone
    # the molten skin: a slow crawl of ripples, growing with the slump (tiny: the letters must stay letters)
    t = f / FPS
    rip = (np.sin(9.0 * th + 3.0 * A + 5.1 * t) * 0.6 + np.sin(14.0 * th - 2.0 * A - 7.3 * t + 1.3) * 0.4)
    amp = 0.000045 * np.clip(sig / 0.5, 0.0, 1.0) * (0.3 + wfar)
    rho = rho + amp * rip * ca
    zeta = zeta + amp * rip * np.clip(sa, 0.0, 1.0)
    lobe = (0.075 * np.sin(2.0 * (th - TH_B) + 0.7) + 0.050 * np.sin(3.0 * (th - TH_B) + 2.1) +
            0.028 * np.sin(5.0 * (th - TH_B) + 0.3))
    Rc = R_MID * (1.0 + sg * lobe * (0.30 + 0.70 * wfar) - 0.05 * sg * wfar)
    return rho, zeta, Rc, th, wfar


def _place(rho, zeta, Rc, th):
    import numpy as np
    r = Rc + rho
    return np.stack([r * np.cos(th), r * np.sin(th), zeta], axis=-1)


def _neck(S, f):
    """Section scale near the back as the metal drains from it before it breaks (1 = none)."""
    import numpy as np
    nu = 0.90 * _ss((f - B['neck0']) / (B['out'] - B['neck0'])) ** 1.5
    g = np.exp(-((1.0 - np.abs(S)) / 0.075) ** 2)
    return 1.0 - nu * g


def _retract_phi(f):
    """Half-angle of what is left of the arc (from the front) while the ends run forward."""
    u = _cl((f - B['out']) / (B['merge'] - B['out']))
    r = 1.0 - (1.0 - u) ** 1.55
    r = r * _ss(u / 0.12) ** 0.5 if u < 0.12 else r                 # it takes a frame to start
    return math.pi - (math.pi - 0.55) * r


def melt_verts(f):
    """All vertex positions (band-local: z up, the stone at z = 0, the band's centre on the z axis) at frame f,
    as float32 (N, 3) + the two end-fan centres. Before the break: the slump (+ the neck). After: the two halves run
    forward along the arc and gather into bulbs, the bulbs meet at the front and pull into one bead that trembles."""
    import numpy as np
    S, A = _grid()
    if f <= B['out']:
        rho, zeta, Rc, th, _ = _section(S, A, f)
        if f > B['neck0']:
            k = _neck(S, f)
            rho, zeta = rho * k, zeta * k
        P = _place(rho, zeta, Rc, th)
    else:
        P = _after_break(S, A, f)
    ends = np.stack([P[0].mean(axis=0), P[-1].mean(axis=0)])
    return np.concatenate([P.reshape(-1, 3), ends]).astype(np.float32)


def _ssn(u):
    import numpy as np
    u = np.clip(u, 0.0, 1.0)
    return u * u * (3.0 - 2.0 * u)


def _trap(y, x):
    return float(((y[1:] + y[:-1]) * 0.5 * (x[1:] - x[:-1])).sum())


# the tube's section at the break, nominal (half-width across the band, height): what the bulb scales refer to
W0, H0 = 0.0016, 0.0040
H_MAX = 0.0050                     # a liquid bulb of gold on stone is squat: never taller than this


def _tube_h(phi):
    """Height of the slumped band (at the breath) at arc angle phi from the melt's front."""
    wfar = 0.5 * (1.0 - math.cos(phi))
    return WIDTH * (1.0 - 0.40 * SIG_MAX * (0.42 + 1.05 * wfar))


def _gauss1d(y, sig_n):
    import numpy as np
    if sig_n < 0.5:
        return y
    r = int(3 * sig_n) + 1
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sig_n) ** 2)
    k /= k.sum()
    yp = np.concatenate([np.full(r, y[0]), y, np.full(r, y[-1])])
    return np.convolve(yp, k, mode='valid')


def _profile(Phi, Rc, v_half, kt, n=600):
    """One half of the running band, from the melt's front (phi = 0) to its end (phi = Phi): the tube (section scale
    kt) swelling into a squat end bulb (round in plan, radius a; never taller than the band it runs along), its tip
    at Phi. Returns phi and the horizontal / vertical section scales (kh, kv), the bulb sized so the volume is v_half,
    smoothed along the arc (no crease where the tube meets the bulb)."""
    import numpy as np
    phi = np.linspace(0.0, Phi, n)
    area0 = 2.0 * W0 * H0 * 0.80
    hmax = min(H_MAX, 1.0 * _tube_h(Phi))
    a = 0.0018
    for _ in range(40):
        a = min(a, 0.98 * Rc * Phi)                               # a bulb never outgrows what is left of the arc
        h = min(hmax, 1.6 * a)
        beta = min(a / Rc, 0.999 * Phi)
        pc = Phi - beta
        d = np.clip(Rc * (phi - pc) / a, -1.0, 1.0)
        prof = np.sqrt(np.clip(1.0 - d * d, 0.0, 1.0))
        khb = (a / W0) * prof
        kvb = (h / H0) * prof
        cut = _ssn((pc - phi) / max(0.8 * a / Rc, 1e-6) + 0.5)     # the tube runs into the bulb and ends in it
        kvt = min(kt, 1.04)                                          # the thickening tube spreads, never rises
        kh = _smax(kt * kt / kvt * cut, khb, 0.35)
        kv = _smax(kvt * cut, kvb, 0.35)
        beyond = phi > pc
        kh = np.where(beyond, khb, kh)
        kv = np.where(beyond, kvb, kv)
        vol = _trap(kh * kv * area0 * Rc, phi)
        a *= (v_half / max(vol, 1e-15)) ** 0.4
    bulb = _ssn((phi - (pc - 0.6 * beta)) / max(0.6 * beta, 1e-6))   # 1 inside the bulb: its sections are round
    return phi, kh, kv, bulb


def _after_break(S, A, f):
    """After the break: each half runs forward round the band (what is left of the arc shrinks to the melt's front)
    and gathers into a squat bulb at its end; the vertices sit by volume (the material keeps its place in the metal).
    The sections keep the shape they had at the breath (the letters' metal), rounding off as they swell."""
    import numpy as np
    rho0, zeta0, Rc0, th0, _ = _section(S, A, B['out'])          # the neck's metal is already in the bulb
    zc0 = 0.5 * zeta0.max(axis=1, keepdims=True)
    Phi = _retract_phi(f)
    u = _cl((f - B['out']) / (B['merge'] - B['out']))
    keep = 1.0 - 0.41 * _ss((f - B['out']) / 14.0)                # some of the gold goes into the fire
    Rc_u = R_MID * (1.0 - 0.14 * u)                               # the running metal draws in a little
    area0 = 2.0 * W0 * H0 * 0.80
    v_half = keep * area0 * R_MID * math.pi
    kt = 1.0 + 0.55 * u
    phi, kh_p, kv_p, bulb_p = _profile(Phi, Rc_u, v_half, kt)
    dv = kh_p * kv_p
    cum = np.concatenate([[0.0], np.cumsum(0.5 * (dv[1:] + dv[:-1]) * np.diff(phi))])
    cum /= cum[-1]
    m = np.abs(S)
    ph = np.interp(m, cum, phi)
    kh = np.interp(ph, phi, kh_p)
    kv = np.interp(ph, phi, kv_p)
    bulb = np.interp(ph, phi, bulb_p)
    sgn = np.where(S >= 0.0, 1.0, -1.0)
    th = TH_C + sgn * ph
    rho = rho0 * kh
    zeta = zc0 * kv + (zeta0 - zc0) * kv
    ca, sa = np.cos(A), np.sin(A)
    rnd = np.maximum(_ssn((np.maximum(kh, kv) - 1.0) / 1.0), bulb)   # where it swells it rounds into a drop
    rho_e = W0 * kh * np.sign(ca) * np.abs(ca) ** 0.85
    zeta_e = 0.5 * H0 * kv * (1.0 + np.sign(sa) * np.abs(sa) ** 0.85)
    rho = rho + (rho_e - rho) * rnd
    zeta = zeta + (zeta_e - zeta) * rnd
    zeta = _smax(zeta - zeta.min(axis=1, keepdims=True), np.full_like(zeta, 0.00010), 0.0003)
    Rc = Rc0 + (Rc_u - Rc0) * _ss(u / 0.35)                        # no pop at the break: the lobes relax
    P = _place(rho, zeta, Rc, th)
    # the two bulbs meet at the melt's front and pull into one bead (a sessile drop), overshoot and tremble
    w = _ss((f - (B['merge'] - 3.0)) / 4.0)                      # touching drops bridge at once: no saddle
    if w > 0.0:
        P = P + (_bead(S, A, f) - P) * w
    return P


def _xi_of(c):
    """Axial position on an ellipsoid (-1..1) holding the volume fraction c (0..1) between it and the equator."""
    import numpy as np
    x = np.linspace(0.0, 1.0, 2001)
    F = 0.5 * (3.0 * x - x ** 3)
    return np.interp(c, F, x)


def _bead(S, A, f):
    import numpy as np
    t = max(0.0, f - (B['merge'] - 3.0))
    wob = 0.30 * math.exp(-t / 6.5) * math.cos(2.0 * math.pi * t / 7.2)   # born long (two drops just met)
    wob2 = 0.05 * math.exp(-t / 9.0) * math.sin(2.0 * math.pi * t / 4.3 + 0.8)
    at = BEAD_A * (1.0 + wob)
    an = BEAD_A * (1.0 - 0.5 * wob + wob2)
    cz = BEAD_C * (1.0 - 0.45 * wob - wob2)
    tf = np.array([-math.sin(TH_C), math.cos(TH_C), 0.0])        # along the band at the melt's front (S > 0 side)
    nf = np.array([math.cos(TH_C), math.sin(TH_C), 0.0])         # outward there (toward the lens, 22 deg right)
    xi = np.sign(S) * _xi_of(np.abs(S))                           # the metal keeps its share of the volume
    ro = np.sqrt(np.clip(1.0 - xi * xi, 0.0, 1.0))
    ca, sa = np.cos(A), np.sin(A)
    se = 2.0 / 2.3                                                # a hair fuller than an ellipsoid (a drop)
    cs = np.sign(ca) * np.abs(ca) ** se
    ss = np.sign(sa) * np.abs(sa) ** se
    C = BEAD_R * nf
    z0 = 0.82 * cz
    X = C[None, None, :] + (at * xi)[..., None] * tf + (an * ro * cs)[..., None] * nf
    Z = z0 + cz * ro * ss
    Z = _smax(Z, np.full_like(Z, 0.00012), 0.00035)               # the flat foot on the stone
    X[..., 2] = Z
    rip = 0.00004 * math.exp(-t / 14.0) * np.sin(7.0 * A + 5.0 * S + 0.9 * t)
    X = X + rip[..., None] * np.stack([ca * nf[0], ca * nf[1], sa], axis=-1)
    return X


def topology():
    """Faces (quads round the tube + a fan at each end), loop UVs ('strip': u = th / 2pi over the loop, continuous
    through the front; v = 1 at the band's top edge) and the per-vertex 'outer'/'inner' facing (from the rest
    section), for Blender. The last two vertices are the end-fan centres."""
    import numpy as np
    S, A = _grid()
    nv = (N_S + 1) * N_A
    th = TH_C + np.pi * S
    idx = np.arange(nv).reshape(N_S + 1, N_A)
    j = np.arange(N_S)[:, None]
    i = np.arange(N_A)[None, :]
    i2 = (i + 1) % N_A
    quads = np.stack([idx[j, i], idx[j + 1, i], idx[j + 1, i2], idx[j, i2]], axis=-1).reshape(-1, 4)
    ai = 2.0 * np.pi * np.arange(N_A + 1) / N_A                    # the row angle with the wrap row = 2 pi
    vrow = 0.5 * _sgnpow(np.sin(ai), 2.0 / SQ) + 0.5
    ucol = th[:, 0] / (2.0 * np.pi)
    ii = np.broadcast_to(i, (N_S, N_A))
    jj = np.broadcast_to(j, (N_S, N_A))
    corners = [(jj, ii), (jj + 1, ii), (jj + 1, ii + 1), (jj, ii + 1)]
    quv = np.stack([np.stack([ucol[cj], vrow[ci]], axis=-1) for cj, ci in corners], axis=2).reshape(-1, 4, 2)
    fans, fuv = [], []
    for k, col in enumerate((0, N_S)):
        cen = nv + k
        for r in range(N_A):
            r2 = (r + 1) % N_A
            tri = (cen, idx[col, r], idx[col, r2]) if k == 0 else (cen, idx[col, r2], idx[col, r])   # outward
            fans.append(tri)
            fuv.append([(ucol[col], 0.5)] * 3)
    outer = np.maximum(np.cos(A), 0.0).reshape(-1)
    inner = np.maximum(-np.cos(A), 0.0).reshape(-1)
    sco = S.reshape(-1)
    outer = np.concatenate([outer, [0.0, 0.0]])
    inner = np.concatenate([inner, [0.0, 0.0]])
    sco = np.concatenate([sco, [-1.0, 1.0]])
    return dict(quads=quads.astype(np.int32), quv=quv.astype(np.float32), fans=np.array(fans, np.int32),
                fuv=np.array(fuv, np.float32), outer=outer.astype(np.float32), inner=inner.astype(np.float32),
                s=sco.astype(np.float32), nv=nv + 2)


def _write_f32(path, arr):
    import numpy as np
    tmp = path + '.tmp'
    np.ascontiguousarray(arr, np.float32).tofile(tmp)
    os.replace(tmp, path)


def prep(frames, cache):
    import json

    import numpy as np
    d = os.path.join(cache, 'data')
    os.makedirs(d, exist_ok=True)
    redo = bool(os.environ.get('MELT_REMESH'))
    tp = os.path.join(d, 'topo.json')
    if redo or not os.path.exists(tp):
        T = topology()
        for k in ('quads', 'fans'):
            T[k].astype(np.int32).tofile(os.path.join(d, f'{k}.i32'))
        for k in ('quv', 'fuv', 'outer', 'inner', 's'):
            _write_f32(os.path.join(d, f'{k}.f32'), T[k])
        with open(tp + '.tmp', 'w') as fh:
            json.dump(dict(nv=int(T['nv']), nq=int(len(T['quads'])), nf=int(len(T['fans']))), fh)
        os.replace(tp + '.tmp', tp)
    bp = os.path.join(d, 'basis.f32')
    if redo or not os.path.exists(bp):
        _write_f32(bp, melt_verts(START))
    need = [f for f in frames if f >= B['soft0'] - 1]
    for f in need:
        p = os.path.join(d, f'melt_{f:05d}.f32')
        if redo or not os.path.exists(p):
            _write_f32(p, melt_verts(f))
    return dict(melt=dict(data=d, tex=os.path.join(ADIR, 'inscription_outer.png'),
                          tex_in=os.path.join(ADIR, 'inscription_inner.png'),
                          keyed=[f for f in need]))


def post(f, hdr, depth, cam, scene):
    """The hearth's flare to white (5476 -> near full white at 5519) and a breath of heat haze behind the band."""
    import numpy as np
    if not PREVIEW:
        hdr = _haze(f, hdr, depth)
    w = white(f)
    if w <= 0.0:
        return hdr
    return hdr * (1.0 + 5.0 * w * w) + 26.0 * w ** 3.2


def _haze(f, hdr, depth):
    """Screen-space heat shimmer over what lies well behind the band (never the band itself: depth-gated)."""
    import cv2
    import numpy as np
    H, W = hdr.shape[:2]
    if depth is None:
        return hdr
    t = f / FPS
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    s = W / 1920.0
    gate = np.clip((depth - 0.19) / 0.03, 0.0, 1.0).astype(np.float32)   # the band sits at ~0.15-0.17 m
    gate = cv2.GaussianBlur(gate, (0, 0), 6.0 * s)
    amp = 2.2 * s * gate
    dx = np.sin(yy * 0.045 / s + t * 11.0) * 0.6 + np.sin(xx * 0.021 / s + yy * 0.031 / s + t * 7.0) * 0.4
    dy = np.sin(yy * 0.037 / s + xx * 0.013 / s + t * 9.0 + 1.3)
    mx = (xx + dx * amp).astype(np.float32)
    my = (yy + dy * amp * 0.7 + 1.1 * s * gate).astype(np.float32)
    return cv2.remap(hdr.astype(np.float32), mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def post_aux(f, ch, out_dir, f_out):
    """ACCORD-3's AC3 composite: the metal's coverage (the 'ringmask' AOV) -> <out_dir>_mask/f_%05d.png (8-bit)."""
    key = next((k for k in ch if 'ringmask' in k), None)
    if key is None:
        return
    import cv2
    import numpy as np
    d = out_dir.rstrip('/') + '_mask'
    os.makedirs(d, exist_ok=True)
    m8 = (np.clip(ch[key], 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    p = os.path.join(d, f'f_{f_out:05d}.png')
    cv2.imwrite(p + '.tmp.png', m8)
    os.replace(p + '.tmp.png', p)


# ======================================================================= Blender side ===

def _read(path, typ):
    a = array.array(typ)
    with open(path, 'rb') as fh:
        a.frombytes(fh.read())
    return a


def _keyv(nb, name, fn, frames):
    from kit import core as C
    v = nb.value(0.0, name)
    for f in frames:
        C.key_socket(v.outputs[0], f, fn(f))
    return v.outputs[0]


def _aov(nb, name, value):
    nd = nb.n('ShaderNodeOutputAOV')
    try:
        nd.aov_name = name
    except Exception:
        nd.name = name
    nb.set(nd.inputs['Value'], value)
    return nd


def _frames_span(T):
    fr = sorted(set(T.get('_frames', [])))
    lo = (fr[0] if fr else START) - 2
    hi = (fr[-1] if fr else END) + 2
    return list(range(max(lo, START - 2), min(hi, END + 2) + 1))


def gold_melt(new_material, R, fr):
    """Polished yellow gold that melts. Letters: the script of fire through the canonical masks on the outer and
    inner faces, deep orange-red (never white), keyed. The metal's own glow: a blackbody, keyed, weighted to the back
    (toward the fire's heart) until it runs, then all of it. Liquid: the polish marks go, a mirror."""
    import bpy
    m, nb = new_material('gold_melt')
    P = nb.texco().outputs['Object']
    uvn = nb.n('ShaderNodeUVMap')
    uvn.uv_map = 'strip'
    u_, v_, _ = nb.sep(uvn.outputs['UV'])
    liq = _keyv(nb, 'liquid', liquid, fr)
    polish = nb.noise(P, scale=160.0, detail=2.0, rough=0.5)
    rgh = nb.madd(polish.outputs['Fac'], 0.045, 0.035)
    rgh = nb.mixf(liq, rgh, 0.018)
    bs = nb.principled(Base_Color=(1.0, 0.71, 0.29), Metallic=1.0, Roughness=rgh)
    bs.inputs['Specular Tint'].default_value = (1.0, 0.93, 0.78, 1.0)
    shader = bs
    # the script of fire
    cov = None
    for key, path, flip in (('outer', R['tex'], False), ('inner', R['tex_in'], True)):
        img = bpy.data.images.load(path, check_existing=True)
        img.colorspace_settings.name = 'Non-Color'
        tx = nb.n('ShaderNodeTexImage')
        tx.image = img
        tx.interpolation = 'Cubic'
        tx.extension = 'REPEAT'
        nb.link(nb.comb(nb.sub(1.0, u_) if flip else u_, v_, 0.0), tx.inputs['Vector'])
        c = nb.mul(tx.outputs['Color'], nb.sstep(0.25, 0.6, nb.attr(key).outputs['Fac']))
        cov = c if cov is None else nb.add(cov, c)
    col = nb.mixcol(_keyv(nb, 'flarecol', flare_col, fr), (1.0, 0.19, 0.022), (1.0, 0.40, 0.06))
    lk = _keyv(nb, 'letters', letters, fr)
    shader = nb.addshader(shader, nb.emission(col, nb.mul(nb.mul(cov, lk), 1.9)))
    # the metal's own heat: weighted to the back while solid (strip u -> angle round the band), even once liquid
    ang = nb.mul(u_, 2.0 * math.pi)
    back = nb.madd(nb.math('COSINE', nb.sub(ang, TH_B)), 0.5, 0.5)
    wgt = nb.mixf(liq, nb.madd(nb.pw(back, 2.0), 0.9, 0.1), 1.0)
    bb = nb.n('ShaderNodeBlackbody')
    nb.link(_keyv(nb, 'temp', hot_temp, fr), bb.inputs['Temperature'])
    # a metal glows where it faces you and mirrors at grazing angles (emissivity = 1 - reflectance)
    lw = nb.n('ShaderNodeLayerWeight')
    lw.inputs['Blend'].default_value = 0.5
    face = nb.pw(nb.sub(1.0, lw.outputs['Facing']), 1.6)
    hk = nb.mul(nb.mul(_keyv(nb, 'hot', hot, fr), wgt), face)
    shader = nb.addshader(shader, nb.emission(bb.outputs['Color'], hk))
    nb.output(surface=shader)
    _aov(nb, 'ringmask', 1.0)
    return m


def _band(C, mat, R, T, fr):
    """The band: one mesh (open tube + end fans), its UVs and facing attributes, shape keys for the melt frames."""
    import json

    import bpy
    d = R['data']
    with open(os.path.join(d, 'topo.json')) as fh:
        tp = json.load(fh)
    nv, nq, nf = tp['nv'], tp['nq'], tp['nf']
    quads = _read(os.path.join(d, 'quads.i32'), 'i')
    fans = _read(os.path.join(d, 'fans.i32'), 'i')
    me = bpy.data.meshes.new('ring')
    me.vertices.add(nv)
    me.vertices.foreach_set('co', _read(os.path.join(d, 'basis.f32'), 'f'))
    me.loops.add(nq * 4 + nf * 3)
    me.loops.foreach_set('vertex_index', quads + fans)
    me.polygons.add(nq + nf)
    me.polygons.foreach_set('loop_start', array.array('i', list(range(0, nq * 4, 4)) +
                                                        list(range(nq * 4, nq * 4 + nf * 3, 3))))
    me.update(calc_edges=True)
    me.polygons.foreach_set('use_smooth', [True] * nq + [False] * nf)          # the end fans stay flat
    uv = me.uv_layers.new(name='strip')
    uv.data.foreach_set('uv', _read(os.path.join(d, 'quv.f32'), 'f') + _read(os.path.join(d, 'fuv.f32'), 'f'))
    for nm in ('outer', 'inner', 's'):
        at = me.attributes.new(nm, 'FLOAT', 'POINT')
        at.data.foreach_set('value', _read(os.path.join(d, f'{nm}.f32'), 'f'))
    me.materials.append(mat)
    me.update()
    ob = bpy.data.objects.new('ring', me)
    C.link_obj(ob)
    ob.pass_index = 1
    keyed = [f for f in R['keyed'] if fr[0] <= f <= fr[-1]]
    if keyed:
        ob.shape_key_add(name='Basis', from_mix=False)
        for f in keyed:
            p = os.path.join(d, f'melt_{f:05d}.f32')
            if not os.path.exists(p):
                continue
            sk = ob.shape_key_add(name=f'm{f}', from_mix=False)
            sk.data.foreach_set('co', _read(p, 'f'))
            for ff, v in ((f - 1, 0.0), (f, 1.0), (f + 1, 0.0)):
                sk.value = v
                sk.keyframe_insert('value', frame=ff)
        ad = ob.data.shape_keys.animation_data
        if ad and ad.action:
            for fc in _fcurves_of(ad.action):
                for kp in fc.keyframe_points:
                    kp.interpolation = 'LINEAR'
        print(f'melt: {len(keyed)} shape keys {keyed[0]}-{keyed[-1]}', flush=True)
    # the drop: key the transform at quarter frames through the strike and the rattle (crisp motion blur)
    ob.rotation_mode = 'QUATERNION'
    fq = []
    x = float(fr[0])
    while x <= fr[-1] + 1e-6:
        fq.append(round(x, 3))
        x += 0.25 if x < B['still'] + 1.0 else 1.0
    for f in fq:
        c, q = ring_state(f)
        off = _qv(q, (0.0, 0.0, HW))
        ob.location = (c[0] - off[0], c[1] - off[1], c[2] - off[2])
        ob.rotation_quaternion = q
        ob.keyframe_insert('location', frame=f)
        ob.keyframe_insert('rotation_quaternion', frame=f)
    return ob


def _fcurves_of(action):
    try:
        return list(action.fcurves)
    except Exception:
        pass
    out = []
    for layer in getattr(action, 'layers', []):
        for strip in layer.strips:
            for bag in getattr(strip, 'channelbags', []):
                out.extend(bag.fcurves)
    return out


def _stone(C, new_material):
    """The council's hearth stone: fire-blackened dark stone, a hair uneven, soot, a dust of pale ash; dead flat
    under the band. Macro: grains and ash flecks are sub-millimetre. Never glowing cracks (a lava floor)."""
    import bpy
    from mathutils import Vector, noise
    half, n = 0.34, 300
    V, F = [], []
    for j in range(n + 1):
        for i in range(n + 1):
            x = -half + 2 * half * i / n
            y = -0.20 + (0.20 + half) * j / n
            r2 = x * x + y * y
            z = 0.0007 * noise.noise(Vector((x * 22.0, y * 22.0, 0.3))) + 0.00025 * noise.noise(
                Vector((x * 90.0, y * 90.0, 1.7)))
            z *= _cl((r2 - 0.0004) / 0.0025)                        # dead flat where the band lies
            V.append((x, y, z))
    for j in range(n):
        for i in range(n):
            F.append((j * (n + 1) + i, j * (n + 1) + i + 1, (j + 1) * (n + 1) + i + 1, (j + 1) * (n + 1) + i))
    m, nb = new_material('hearthstone')
    P = nb.texco().outputs['Object']
    grain = nb.voronoi(P, scale=1400.0, feature='F1')
    gr, _, _ = nb.sep(grain.outputs['Color'])
    soot = nb.noise(P, scale=55.0, detail=5.0, rough=0.6)
    ash = nb.sstep(0.56, 0.78, nb.noise(P, scale=70.0, detail=7.0, rough=0.75).outputs['Fac'])
    fleck = nb.sstep(0.93, 0.97, nb.noise(P, scale=900.0, detail=2.0, rough=0.5).outputs['Fac'])
    col = nb.mixcol(nb.mul(nb.sstep(0.80, 0.95, gr), 0.5), (0.026, 0.024, 0.022), (0.045, 0.041, 0.038))
    col = nb.mixcol(nb.mul(nb.sstep(0.30, 0.72, soot.outputs['Fac']), 0.9), col, (0.006, 0.005, 0.0045))
    col = nb.mixcol(nb.mul(ash, 0.55), col, (0.11, 0.105, 0.10))
    col = nb.mixcol(nb.mul(fleck, 0.15), col, (0.16, 0.15, 0.14))
    bs = nb.principled(Base_Color=col, Roughness=nb.madd(soot.outputs['Fac'], 0.08, 0.88))
    bs.inputs['Specular IOR Level'].default_value = 0.10
    h = nb.add(nb.mul(grain.outputs['Distance'], 0.6), nb.mul(ash, 0.4))
    nb.link(nb.bump(h, 0.35, 0.00018), bs.inputs['Normal'])
    nb.output(surface=bs)
    ob = C.mesh_obj('stone', V, F, mat=m)
    return ob


def _coals(C, new_material, fr):
    """A few fallen embers of the torches on the stone, at the sides and behind (never in front of the band): small,
    ash-grey skins cracked with heat; out of focus they are soft glows, never a glowing floor."""
    import random

    import bpy
    from mathutils import Vector, noise
    rng = random.Random(23)
    m, nb = new_material('coal')
    P = nb.texco().outputs['Object']
    crack = nb.voronoi(P, scale=2600.0, feature='DISTANCE_TO_EDGE')
    glow = nb.sub(1.0, nb.sstep(0.0, 0.06, crack.outputs['Distance']))
    skin = nb.noise(P, scale=900.0, detail=3.0)
    bb = nb.n('ShaderNodeBlackbody')
    bb.inputs['Temperature'].default_value = 1350.0
    gk = _keyv(nb, 'coalk', fire, fr)
    em = nb.emission(bb.outputs['Color'], nb.mul(nb.mul(nb.add(nb.mul(glow, 0.8), nb.mul(skin.outputs['Fac'], 0.05)),
                                                        gk), 1.2))
    bs = nb.principled(Base_Color=(0.05, 0.045, 0.04), Roughness=0.9)
    nb.output(surface=nb.addshader(bs, em))
    spots = [(-0.034, 0.018), (-0.046, 0.052), (0.038, 0.012), (0.052, 0.046), (-0.018, 0.070), (0.020, 0.080),
             (0.064, 0.020), (-0.066, 0.030), (0.006, 0.060)]
    for k, (x, y) in enumerate(spots):
        r = rng.uniform(0.0022, 0.0045)
        V, F = _blob(r, rng.uniform(0.0, 50.0), squash=rng.uniform(0.45, 0.7))
        ob = C.mesh_obj(f'coal{k}', V, F, mat=m)
        ob.location = (x, y, 0.35 * r)
        ob.rotation_euler = (0.0, 0.0, rng.uniform(0.0, 6.28))


def _blob(r, seed, squash=0.6, nu=18, nv=10):
    from mathutils import Vector, noise
    V, F = [], []
    for j in range(1, nv):
        ph = math.pi * j / nv
        for i in range(nu):
            t = 2 * math.pi * i / nu
            d = Vector((math.sin(ph) * math.cos(t), math.sin(ph) * math.sin(t), math.cos(ph)))
            k = 1.0 + 0.28 * noise.noise(d * 1.7 + Vector((seed, 0.0, 0.0)))
            V.append((r * k * d.x, r * k * d.y, r * k * squash * d.z))
    top, bot = len(V), len(V) + 1
    V += [(0.0, 0.0, r * squash), (0.0, 0.0, -r * squash)]
    for j in range(nv - 2):
        for i in range(nu):
            i2 = (i + 1) % nu
            F.append((j * nu + i, j * nu + i2, (j + 1) * nu + i2, (j + 1) * nu + i))
    for i in range(nu):
        F.append((top, (i + 1) % nu, i))
        F.append((bot, (nv - 2) * nu + i, (nv - 2) * nu + (i + 1) % nu))
    return V, F


def _embers(C, new_material, fr):
    """Sparks and ash in the updraft: a few from the strike (5363), then the fire's own, rising fast (at this
    scale they cross the frame in a frame or two: streaks), some near the lens (soft discs)."""
    import random

    import bpy
    rng = random.Random(61)
    m, nb = new_material('ember')
    oi = nb.objinfo()
    bb = nb.n('ShaderNodeBlackbody')
    nb.link(nb.madd(oi.outputs['Random'], 500.0, 1500.0), bb.inputs['Temperature'])
    k = _keyv(nb, 'emberk', fire, fr)
    nb.output(surface=nb.emission(bb.outputs['Color'], nb.mul(nb.madd(oi.outputs['Random'], 30.0, 22.0), k)))
    V, F = _blob(1.0, 0.0, squash=1.0, nu=8, nv=5)
    me = bpy.data.meshes.new('ember')
    me.from_pydata(V, [], F)
    me.materials.append(m)
    parts = []
    # the strike: ash motes puffed up and two sparks kicked out
    for i in range(12):
        a = rng.uniform(0, 2 * math.pi)
        parts.append(dict(t0=B['hit'] + rng.uniform(0.0, 1.0), life=rng.uniform(7, 14), r=rng.uniform(0.00012, 0.00022),
                          p0=(0.0021 + 0.012 * math.cos(a), -0.0012 + 0.012 * math.sin(a), 0.0004),
                          v=(0.03 * math.cos(a), 0.03 * math.sin(a), rng.uniform(0.05, 0.12)), up=0.9, hotk=0.05))
    for i in range(3):
        a = rng.uniform(-0.6, 0.6) + (math.pi if i == 1 else 0.0)
        parts.append(dict(t0=B['hit'] + 0.2, life=rng.uniform(3, 5), r=0.00008,
                          p0=(0.0021 + 0.011 * math.cos(a), -0.0012 + 0.011 * math.sin(a), 0.001),
                          v=(0.45 * math.cos(a), 0.45 * math.sin(a), rng.uniform(0.25, 0.4)), up=-3.0, hotk=1.0))
    # the fire's own sparks, all through the shot (fewer in the breath)
    t = START - 4.0
    while t < END:
        t += rng.expovariate(1.0 / 2.6)
        if B['breath'] - 2 < t < B['out'] + 2 and rng.random() < 0.7:
            continue
        x = rng.uniform(-0.06, 0.06)
        y = rng.choice([rng.uniform(0.012, 0.09), rng.uniform(0.012, 0.09), rng.uniform(-0.08, -0.03)])
        parts.append(dict(t0=t, life=rng.uniform(4, 10), r=rng.uniform(0.00008, 0.00022),
                          p0=(x, y, rng.uniform(0.0, 0.004)),
                          v=(rng.uniform(-0.08, 0.08), rng.uniform(-0.04, 0.04), rng.uniform(0.25, 0.7)), up=0.6,
                          hotk=1.0))
    n = 0
    for p in parts:
        f0, f1 = int(math.floor(p['t0'])), int(math.ceil(p['t0'] + p['life']))
        if f1 < fr[0] or f0 > fr[-1]:
            continue
        ob = bpy.data.objects.new(f'ember{n}', me)
        C.link_obj(ob)
        n += 1
        for f in range(max(f0 - 1, fr[0]), min(f1 + 1, fr[-1]) + 1):
            tt = max(0.0, (f - p['t0']) / FPS)
            x = p['p0'][0] + p['v'][0] * tt + 0.01 * math.sin(tt * 9.0 + n)
            y = p['p0'][1] + p['v'][1] * tt
            z = p['p0'][2] + p['v'][2] * tt + 0.5 * p['up'] * tt * tt
            live = 0.0 if (f < p['t0'] or f > p['t0'] + p['life']) else 1.0
            fade = live * (1.0 - _ss((f - p['t0'] - 0.75 * p['life']) / (0.25 * p['life'])))
            s = p['r'] * max(fade, 0.0) + 1e-7
            ob.location = (x, y, max(z, s))
            ob.scale = (s, s, s)
            ob.keyframe_insert('location', frame=f)
            ob.keyframe_insert('scale', frame=f)
    print(f'embers: {n} in this job', flush=True)


def _flame_mat(new_material, name, W, H, lam, seed, gain, fr, rise=1.0, glossy_only=False):
    """A sheet of flame tongues W x H metres (u across, v up). A noise field along the sheet (swaying more as it
    rises, drifting, flickering) is cut by a threshold that climbs with height, so every tongue is wide at its root
    and tapers to a torn point, at its own irregular spacing and height; wisps are torn off the edges; the roots
    glow in patches (the white heart), never a line. Blackbody colour: yellow-white at the roots, orange at the
    tips. Additive (emission + transparent)."""
    m, nb = new_material(name)
    uvn = nb.n('ShaderNodeUVMap')
    uvn.uv_map = 'UVMap'
    u_, v_, _ = nb.sep(uvn.outputs['UV'])
    t = _keyv(nb, name + '_t', lambda f: f / FPS, fr)
    x = nb.mul(nb.sub(u_, 0.5), W)
    z = nb.mul(v_, H)
    zn = nb.clamp01(nb.div(z, H))
    k1 = 1.6 / lam
    n1 = nb.noise(nb.comb(nb.mul(x, k1), nb.sub(nb.mul(z, k1 * 0.7), nb.mul(t, rise * k1 * 0.09)), seed),
                  scale=1.0, detail=3.0, rough=0.5, dims='4D', w=nb.mul(t, 0.7))
    sway = nb.mul(nb.mul(nb.sub(n1.outputs['Fac'], 0.5), 1.4 * lam), nb.pw(zn, 0.8))
    xs = nb.add(x, sway)
    # the tongue field: irregular columns (1-D noise across, slowly changing), sharpened
    c = nb.noise(nb.comb(nb.div(xs, lam), nb.mul(t, 0.35), seed + 3.0), scale=1.0, detail=2.0, rough=0.45,
                 dims='3D')
    cf = nb.sstep(0.28, 0.78, c.outputs['Fac'])
    # each tongue's reach flickers
    nh = nb.noise(nb.comb(nb.div(xs, 1.7 * lam), 0.0, seed + 5.0), scale=1.0, detail=1.0, dims='4D',
                  w=nb.mul(t, 1.4))
    reach = nb.madd(nh.outputs['Fac'], 0.9, 0.35)
    # threshold climbs with height: wide roots, pointed tips
    thr = nb.div(zn, reach)
    tongue = nb.sstep(nb.mul(thr, 0.92), nb.add(nb.mul(thr, 0.92), 0.22), cf)
    k2 = 6.0 / lam
    n2 = nb.noise(nb.comb(nb.mul(x, k2), nb.sub(nb.mul(z, k2 * 0.55), nb.mul(t, rise * k2 * 0.14)), seed + 7.0),
                  scale=1.0, detail=3.0, rough=0.6, dims='4D', w=nb.mul(t, 1.6))
    wisp = nb.sstep(0.28, 0.70, n2.outputs['Fac'])
    dens = nb.mul(tongue, nb.madd(wisp, 0.55, 0.45))
    roots = nb.mul(nb.sub(1.0, nb.sstep(0.0, 0.14, zn)), nb.sstep(0.35, 0.75, n1.outputs['Fac']))
    dens = nb.mx(dens, nb.mul(roots, 0.8))
    dens = nb.mul(dens, nb.sstep(-0.004, 0.002, z))
    heat = nb.clamp01(nb.sub(1.0, nb.mul(zn, 1.25)))
    bb = nb.n('ShaderNodeBlackbody')
    nb.link(nb.madd(heat, 950.0, 1250.0), bb.inputs['Temperature'])
    hotk = nb.madd(heat, 0.8, 0.2)
    k = _keyv(nb, name + '_k', fire, fr)
    em = nb.emission(bb.outputs['Color'], nb.mul(nb.mul(nb.mul(dens, hotk), k), gain))
    tr = nb.n('ShaderNodeBsdfTransparent')
    nb.output(surface=nb.addshader(tr, em))
    try:
        m.emission_sampling = 'NONE'
    except Exception:
        pass
    return m


def _flame_sheet(C, new_material, name, center, W, H, lam, seed, gain, fr, rise=1.0, yaw=0.0):
    import bpy
    x0, y0, z0 = center
    cy, sy = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
    P = []
    for (dx, dz) in ((-0.5 * W, -0.006), (0.5 * W, -0.006), (0.5 * W, H), (-0.5 * W, H)):
        P.append((x0 + dx * cy, y0 + dx * sy, z0 + dz))
    m = _flame_mat(new_material, name, W, H + 0.006, lam, seed, gain, fr, rise)
    ob = C.mesh_obj(name, P, [(0, 1, 2, 3)], mat=m, smooth=False)
    uv = ob.data.uv_layers.new(name='UVMap')
    for li, (uu, vv) in enumerate([(0, 0), (1, 0), (1, 1), (0, 1)]):
        uv.data[li].uv = (uu, vv)
    for attr in ('visible_diffuse', 'visible_shadow', 'visible_volume_scatter', 'visible_transmission'):
        try:
            setattr(ob, attr, False)
        except Exception:
            pass
    return ob


def _stick(r, length, seed, nu=14, nl=24):
    """A charred stick along +x (radius r, length), a hair crooked and knotty, closed ends."""
    from mathutils import Vector, noise
    V, F = [], []
    for j in range(nl + 1):
        x = -0.5 * length + length * j / nl
        bend = 0.08 * r * noise.noise(Vector((x * 30.0, seed, 0.0)))
        for i in range(nu):
            a = 2 * math.pi * i / nu
            k = 1.0 + 0.12 * noise.noise(Vector((x * 90.0, a * 1.3, seed))) - 0.1 * max(0.0, math.cos(a - 1.2)) ** 3
            V.append((x, bend + r * k * math.cos(a), r * k * math.sin(a)))
    for j in range(nl):
        for i in range(nu):
            i2 = (i + 1) % nu
            F.append((j * nu + i, j * nu + i2, (j + 1) * nu + i2, (j + 1) * nu + i))
    c0, c1 = len(V), len(V) + 1
    V += [(-0.5 * length, 0.0, 0.0), (0.5 * length, 0.0, 0.0)]
    for i in range(nu):
        i2 = (i + 1) % nu
        F.append((c0, i2, i))
        F.append((c1, nl * nu + i, nl * nu + i2))
    return V, F


def _char_mat(new_material, fr):
    """Burning kindling: black char in squares and cracks, the cracks and the underside glowing, pale ash at the
    edges of the cracks."""
    m, nb = new_material('char')
    P = nb.texco().outputs['Object']
    vo = nb.voronoi(nb.mul(P, (1.0, 2.2, 2.2)), scale=520.0, feature='DISTANCE_TO_EDGE')
    crack = nb.sub(1.0, nb.sstep(0.0, 0.07, vo.outputs['Distance']))
    under = nb.sstep(0.2, -0.6, nb.sep(nb.geo().outputs['Normal'])[2])
    hotn = nb.noise(P, scale=160.0, detail=3.0)
    glow = nb.mul(nb.mx(crack, nb.mul(under, 0.6)), nb.sstep(0.35, 0.7, hotn.outputs['Fac']))
    bb = nb.n('ShaderNodeBlackbody')
    bb.inputs['Temperature'].default_value = 1400.0
    em = nb.emission(bb.outputs['Color'], nb.mul(nb.mul(glow, _keyv(nb, 'chark', fire, fr)), 4.0))
    ash = nb.mul(nb.sstep(0.0, 0.10, vo.outputs['Distance']), nb.sub(1.0, nb.sstep(0.10, 0.2, vo.outputs['Distance'])))
    col = nb.mixcol(nb.mul(ash, 0.5), (0.018, 0.016, 0.015), (0.16, 0.15, 0.14))
    bs = nb.principled(Base_Color=col, Roughness=0.92)
    bs.inputs['Specular IOR Level'].default_value = 0.2
    nb.link(nb.bump(vo.outputs['Distance'], 0.5, 0.0006), bs.inputs['Normal'])
    nb.output(surface=nb.addshader(bs, em))
    return m


def _flames(C, new_material, T, cam, fr):
    """The white heart: the hearth's kindling burning in a loose arc round the flat stone, 7-17 cm behind the band,
    its flames (the film's bonfire tongues) rising and filling the upper frame; the gold mirrors them. The cards light
    nothing diffusely (the lamps do that)."""
    import random

    import bpy
    from kit import fire as FK
    from mathutils import Vector
    m = _char_mat(new_material, fr)
    rng = random.Random(7)
    sticks = [(-0.045, 0.085, 0.0065, 0.11, 18.0), (0.035, 0.095, 0.0075, 0.12, -24.0),
              (0.000, 0.125, 0.0090, 0.14, 6.0), (-0.090, 0.110, 0.0070, 0.10, 52.0),
              (0.085, 0.120, 0.0070, 0.11, -48.0), (-0.020, 0.160, 0.010, 0.15, -12.0),
              (0.060, 0.170, 0.009, 0.13, 20.0), (0.012, 0.072, 0.0045, 0.06, 70.0)]
    for k, (x, y, r, L, yaw) in enumerate(sticks):
        V, F = _stick(r, L, 3.0 + k * 1.7)
        ob = C.mesh_obj(f'stick{k}', V, F, mat=m)
        ob.location = (x, y, r * 0.85 + (0.004 if k in (5, 6) else 0.0))
        ob.rotation_euler = (0.0, math.radians(rng.uniform(-3, 3)), math.radians(yaw))
    cards = [('kD', (0.012, 0.074, 0.004), 0.20), ('kA', (-0.042, 0.088, 0.006), 0.22),
             ('kA', (0.036, 0.098, 0.007), 0.22), ('kB', (-0.004, 0.128, 0.008), 0.20),
             ('kB', (-0.088, 0.114, 0.006), 0.16), ('kB', (0.084, 0.124, 0.006), 0.16),
             ('kC', (-0.020, 0.165, 0.012), 0.15), ('kC', (0.060, 0.172, 0.011), 0.14)]
    for i, (name, pos, gain) in enumerate(cards):
        sp = T['sprites'][name]
        fc = FK.flame_card(f'{name}_{i}', Vector(pos), sp['card'], sp['first'], cam, gain=gain, fog=False)
        for nd in fc.data.materials[0].node_tree.nodes:
            if nd.type == 'EMISSION':
                base = nd.inputs['Strength'].default_value
                for f in fr:
                    C.key_socket(nd.inputs['Strength'], f, base * fire(f))
        try:
            fc.data.materials[0].emission_sampling = 'NONE'
        except Exception:
            pass
        for attr in ('visible_diffuse', 'visible_shadow', 'visible_volume_scatter', 'visible_transmission'):
            try:
                setattr(fc, attr, False)
            except Exception:
                pass


def _mirrors(C, new_material, fr):
    """What the gold mirrors (reflections only; the lens and the stone never see them): the fire's canopy overhead,
    brightest over the heart behind the band, fading toward the lens; and a dim warm bounce behind the lens, so the
    band's front stays dark gold (the letters read) but still reads as gold."""
    m, nb = new_material('canopy')
    P = nb.texco().outputs['Generated']
    gx, gy, _ = nb.sep(P)
    k = nb.mul(nb.sstep(0.05, 0.85, gy), nb.sub(1.0, nb.mul(nb.pw(nb.math('ABSOLUTE', nb.sub(gx, 0.5)), 2.0), 2.4)))
    n = nb.noise(nb.comb(nb.mul(gx, 5.0), nb.mul(gy, 3.0), _keyv(nb, 'cant', lambda f: f * 0.09, fr)), scale=1.0,
                 detail=3.0, dims='3D')
    k = nb.mul(k, nb.madd(n.outputs['Fac'], 1.2, 0.4))
    col = nb.mixcol(nb.sstep(0.3, 1.0, gy), (1.0, 0.50, 0.17), (1.0, 0.74, 0.42))
    nb.output(surface=nb.emission(col, nb.mul(nb.mul(nb.clamp01(k), _keyv(nb, 'canopyk', fire, fr)), 1.6)))
    cp = C.mesh_obj('canopy', [(-0.22, -0.06, 0.09), (0.22, -0.06, 0.09), (0.22, 0.24, 0.13), (-0.22, 0.24, 0.13)],
                    [(0, 3, 2, 1)], mat=m, smooth=False)
    def bounce(f):
        return fire(f) * (0.10 + 0.55 * _ss((f - B['out'] - 2.0) / 10.0))
    m2 = _flame_mat(new_material, 'bounce', 0.60, 0.22, 0.06, 41.0, 1.0, fr, 1.0)
    for nd in m2.node_tree.nodes:                                 # re-key its strength: bounce(f), not fire(f)
        if nd.type == 'VALUE' and nd.name == 'bounce_k':
            for f in fr:
                nd.outputs[0].default_value = bounce(f)
                nd.outputs[0].keyframe_insert('default_value', frame=f)
    bo = C.mesh_obj('bounce', [(-0.30, -0.36, -0.02), (0.30, -0.36, -0.02), (0.30, -0.36, 0.20),
                               (-0.30, -0.36, 0.20)], [(0, 1, 2, 3)], mat=m2, smooth=False)
    uvb = bo.data.uv_layers.new(name='UVMap')
    for li, (uu, vv) in enumerate([(0, 0), (1, 0), (1, 1), (0, 1)]):
        uvb.data[li].uv = (uu, vv)
    for ob in (cp, bo):
        for attr in ('visible_camera', 'visible_diffuse', 'visible_shadow', 'visible_transmission',
                     'visible_volume_scatter'):
            try:
                setattr(ob, attr, False)
            except Exception:
                pass
    try:
        m.emission_sampling = 'NONE'
    except Exception:
        pass


def _lights(C, fr):
    """The fire's light on the band: the white heart behind and above (the rims, the top edges, the inner face),
    a bed of embers low at each side (a grazing warmth on the stone); nothing from the lens side, so the band's
    front stays dark gold and the letters read."""
    import bpy
    from mathutils import Euler
    L = []

    def area(name, pos, rot, size, col, w):
        ld = bpy.data.lights.new(name, 'AREA')
        ld.shape = 'DISK'
        ld.size = size
        ld.color = col
        ob = C.link_obj(bpy.data.objects.new(name, ld))
        ob.location = pos
        ob.rotation_euler = Euler(tuple(math.radians(a) for a in rot))
        for attr in ('visible_camera', 'visible_glossy'):
            try:
                setattr(ob, attr, False)
            except Exception:
                pass
        L.append((ld, w))
        return ob

    area('heart', (0.0, 0.075, 0.055), (-128.0, 0.0, 0.0), 0.10, (1.0, 0.62, 0.32), 0.85)
    area('crown', (0.0, 0.020, 0.110), (0.0, 0.0, 0.0), 0.12, (1.0, 0.58, 0.28), 0.22)
    area('bedL', (-0.075, 0.020, 0.008), (0.0, -84.0, 0.0), 0.05, (1.0, 0.40, 0.10), 0.12)
    area('bedR', (0.075, 0.030, 0.008), (0.0, 84.0, 0.0), 0.05, (1.0, 0.40, 0.10), 0.12)
    for f in fr:
        k = fire(f)
        for ld, w in L:
            ld.energy = w * k
            ld.keyframe_insert('energy', frame=f)


def _camera(C, fr):
    import bpy
    from mathutils import Vector
    cd = bpy.data.cameras.new('CAM')
    cd.sensor_fit = 'HORIZONTAL'
    cd.sensor_width = 36.0
    cd.lens = 100.0
    cd.clip_start, cd.clip_end = 0.004, 5.0
    ob = C.link_obj(bpy.data.objects.new('CAM', cd))
    ob.rotation_mode = 'QUATERNION'
    bpy.context.scene.camera = ob
    cd.dof.use_dof = not PREVIEW
    cd.dof.aperture_fstop = 20.0
    cd.dof.aperture_blades = 7
    cd.dof.aperture_rotation = math.radians(12.0)
    for f in fr:
        pos, tgt, foc = cam_state(f)
        ob.location = pos
        ob.rotation_quaternion = (Vector(tgt) - Vector(pos)).to_track_quat('-Z', 'Y')
        ob.keyframe_insert('location', frame=f)
        ob.keyframe_insert('rotation_quaternion', frame=f)
        cd.dof.focus_distance = (Vector(foc) - Vector(pos)).length
        cd.dof.keyframe_insert('focus_distance', frame=f)
    return ob


def build(job):
    import bpy

    from kit import core as C
    from kit.nodes import new_material
    T = job['timing']
    R = T['melt']
    T = dict(T, _frames=list(job['frames']))
    fr = _frames_span(T)
    sc = C.setup_render(scale=job['scale'], samples=job['samples'], mblur=True, shutter=0.35, light_threshold=1e-6,
                        shadow_pool='256')
    if PREVIEW:
        sc.render.engine = 'BLENDER_WORKBENCH'
        sh = sc.display.shading
        sh.light = 'STUDIO'
        sh.color_type = 'OBJECT'
        sc.render.use_motion_blur = False
    else:
        C.use_cycles(sc, job['samples'])
        sc.cycles.light_sampling_threshold = 0.0
        sc.cycles.use_light_tree = True
        sc.cycles.glossy_bounces = 6
        sc.cycles.max_bounces = 10
        sc.cycles.sample_clamp_indirect = 6.0
        sc.render.use_motion_blur = True
        sc.render.motion_blur_shutter = 0.35
        try:
            sc.render.motion_blur_position = 'CENTER'
        except Exception:
            pass
    vl = sc.view_layers[0]
    try:
        a = vl.aovs.add()
        a.name = 'ringmask'
        a.type = 'VALUE'
    except Exception as e:
        print('AOV unavailable:', e, flush=True)
    w = bpy.data.worlds.new('W')
    sc.world = w
    w.use_nodes = True
    bg = w.node_tree.nodes['Background']
    bg.inputs[0].default_value = (0.020, 0.008, 0.004, 1.0)
    bg.inputs[1].default_value = 1.0
    for f in fr:
        C.key_socket(bg.inputs[1], f, 0.8 * fire(f))
    cam = _camera(C, fr)
    g = gold_melt(new_material, R, fr)
    _band(C, g, R, T, fr)
    _stone(C, new_material)
    if not PREVIEW:
        _embers(C, new_material, fr)
        _mirrors(C, new_material, fr)
        _flames(C, new_material, T, cam, fr)
        _lights(C, fr)
    else:
        for ob in bpy.data.objects:
            if ob.type == 'MESH':
                ob.color = (1.0, 0.72, 0.28, 1.0) if ob.name == 'ring' else (0.12, 0.11, 0.10, 1.0)
    return {}


def per_frame(job, f):
    return None
