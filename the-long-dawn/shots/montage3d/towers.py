"""KARST v3 (H5, LANDSCAPES NOT LANDMARKS): irregular weathered rock towers in a mist sea. No pines, and
no fluted, vertical-sided sandstone columns (the Zhangjiajie/Huangshan read).

A tower is karstgen's r(theta, z) field with its vertical fluting turned almost off and its section lumpier,
reshaped here in numpy (no numba recompile):
  * a height profile: an irregular taper, a waist or two, a bulge, so no two towers share a silhouette;
  * BEDS: the rock is a stack of layers of different hardness. Soft beds are recessed and hard beds stand
    out, with bevelled edges and a notch at each bedding plane, so the walls read as weathered strata
    with ledges and overhangs, not as columns;
  * a lean (a few degrees, each its own way) and a wandering axis;
  * a broken crown: sub-blocks (karstgen lobes) end at different heights (a stepped, jagged top) with a
    shallow, rubbly cap.
The styles are 'stack' (massive, gently tapering), 'spire' (slender, leaning, waisted, a stepped top),
'split' (two summits with a deep cleft) and 'slab' (a wide, thin fin).
"""
import math

import numpy as np

STYLES = ('stack', 'spire', 'split', 'slab')


def style_for(seed, R0, top):
    """A style from the tower's own proportions and seed (the hero is set by hand)."""
    rng = np.random.default_rng(int(abs(seed) * 977) % (2 ** 31))
    ratio = top / max(R0, 1e-3)
    if ratio > 6.5:
        return 'spire' if rng.random() < 0.75 else 'split'
    if ratio < 3.0:
        return 'stack' if rng.random() < 0.6 else 'slab'
    return str(rng.choice(['stack', 'split', 'spire', 'slab'], p=[0.35, 0.3, 0.2, 0.15]))


def _lobes(style, R0, zb, zt, rng, front=None):
    """Sub-blocks (theta_c, half_width, radius, ztop, taper, edge_sharp) for karstgen._pillar_r."""
    H = zt - zb
    if style == 'split':
        n = 5
    elif style == 'slab':
        n = 4
    else:
        n = int(rng.integers(4, 7))
    th0 = rng.uniform(0, 2 * math.pi)
    gaps = rng.uniform(0.6, 1.4, n)
    ths = th0 + np.cumsum(gaps) / gaps.sum() * 2 * math.pi
    if front is not None:                                # the highest sub-block faces this way (the hero's beacon)
        ths = ths - ths[0] + front
    L = []
    for k in range(n):
        hw = (2 * math.pi / n) * rng.uniform(0.30, 0.52)
        rad = rng.uniform(0.70, 1.10)
        if style == 'slab':
            # two opposite long faces: lobes 0 and 2 wide and strong, 1 and 3 short and shallow
            rad = 1.12 if k % 2 == 0 else 0.55
            hw = (2 * math.pi / n) * (0.62 if k % 2 == 0 else 0.3)
        if k == 0:
            ztop = zt
        elif k == 1 and style != 'split':
            ztop = zt - rng.uniform(0.0, 0.05) * H
        elif style == 'split' and k == 2:
            ztop = zt - rng.uniform(0.06, 0.14) * H        # the second summit, across the cleft
        else:
            ztop = zt - rng.uniform(0.07, 0.45) * H * (0.6 if style == 'stack' else 1.0)
        taper = rng.uniform(0.25, 0.55) if style in ('spire', 'split') else rng.uniform(0.08, 0.3)
        L.append((ths[k], hw, rad, ztop, taper, rng.uniform(2.2, 4.0)))
    if style == 'split':
        # open the cleft between the two summits: the lobes between them stop low
        L[1] = (L[1][0], L[1][1] * 0.8, L[1][2] * 0.9, zt - rng.uniform(0.3, 0.5) * H, L[1][4], L[1][5])
    return np.array(L, np.float64)


def _profile(style, u, th, rng, taper=None):
    """Radius multiplier over (height u 0..1, angle th): an irregular taper plus waists and bulges that are
    ONE-SIDED (an undercut on one face, a buttress on another), so the two sides of a silhouette never mirror
    each other (a symmetric neck reads as a bottle or a chess piece)."""
    if taper is not None:
        rng.uniform(0.0, 1.0)                            # keep the random stream the same
    elif style == 'stack':
        taper = rng.uniform(0.28, 0.45)
    elif style == 'slab':
        taper = rng.uniform(0.15, 0.3)
    else:
        taper = rng.uniform(0.45, 0.62)
    U = u[:, None]
    p = 1.0 - taper * U ** rng.uniform(0.8, 1.6) + 0.0 * th[None, :]
    for sgn, n in ((-1.0, int(rng.integers(1, 4))), (1.0, int(rng.integers(1, 3)))):
        for _ in range(n):
            uc = rng.uniform(0.12, 0.85)
            amp = rng.uniform(0.08, 0.2)
            wd = rng.uniform(0.03, 0.1)
            face = rng.uniform(0, 2 * math.pi)
            spread = rng.uniform(0.9, 2.0)
            side = np.clip(0.5 + 0.5 * np.cos(th - face), 0, 1) ** spread
            p = p + sgn * amp * np.exp(-((U - uc) / wd) ** 2) * side[None, :]
    return p


def _beds(z, R0, rng, bed_len):
    """Strata: per-layer recess/protrusion (as a fraction of the radius) with bevelled edges and a notch at
    each bedding plane. Returns (radius multiplier, notch depth in metres) per z."""
    jit = 0.35 * bed_len * np.sin(z / (3.7 * bed_len) + rng.uniform(0, 6.28))
    zz = (z + jit) / bed_len
    k = np.floor(zz)
    f = zz - k
    # layer hardness from a hash of its index (hard beds stand out, soft beds recede)
    h = np.sin(k * 12.9898 + rng.uniform(0, 100)) * 43758.5453
    h = h - np.floor(h)
    h2 = np.sin((k + 1) * 12.9898 + rng.uniform(0, 100)) * 43758.5453
    off = 0.10 * (h - 0.55)
    # bevel toward the next layer over the top 18% of a bed
    t = np.clip((f - 0.82) / 0.18, 0.0, 1.0)
    t = t * t * (3 - 2 * t)
    nxt = 0.10 * ((h2 - np.floor(h2)) - 0.55)
    mult = 1.0 + off * (1 - t) + nxt * t
    notch = np.exp(-(np.minimum(f, 1 - f) / 0.035) ** 2) * (0.25 + 0.5 * h)
    return mult, notch * min(0.9, 0.035 * R0)


def tower_mesh(cx, cy, R0, zb, zt, seed, nt, dz, z_lo, z_hi=None, dcap=None, style=None, lean=None, cap=0.16,
               rock=1.0, bed_len=None, keep=None, shoulder=None, bites=True, taper=None):
    """A tower meshed only on the visible window [z_lo, z_hi] (nt around, dz between rows). Returns (V float32
    Nx3, Q int32 Mx4 with -1 for triangles, veg float32 N: a little moss affinity in the bedding notches)."""
    import karstgen as KG
    rng = np.random.default_rng(int(abs(seed) * 1000) % (2 ** 31))
    style = style or style_for(seed, R0, zt - zb)
    H = zt - zb
    BL = _lobes(style, R0, zb, zt, rng, front=None if keep is None else keep[0])
    bed_len = bed_len or float(rng.uniform(3.2, 7.5)) * (R0 / 10.0) ** 0.3
    prm = np.array([R0, zb, zt, 0.14, 0.08, 97.0, 0.004, 0.66], np.float64)  # lumpy section, almost no fluting
    th = np.linspace(0, 2 * math.pi, nt, endpoint=False)
    closed = z_hi is None or z_hi >= zt
    top = zt if closed else z_hi
    nz = max(8, int(math.ceil((top - z_lo) / dz)) + 1)
    z = np.linspace(z_lo, top, nz)
    TH, Z = np.meshgrid(th, z)
    R = np.empty_like(TH)
    VEG = np.empty_like(TH)
    u = np.clip((z - zb) / H, 0.0, 1.0)
    prof = _profile(style, u, th, rng, taper)
    if taper is not None:                                # a broad summit: the sub-blocks barely narrow either
        BL[:, 4] = np.minimum(BL[:, 4], 0.4 * taper + 0.02)
    KG._pillar_r(TH, Z, prm, BL, float(seed % 97), R, VEG)
    bm, notch = _beds(z, R0, rng, bed_len)
    # the beds wander a little round the tower (a bed is never a perfect ring)
    wav = 1.0 + 0.025 * np.sin(TH * 2 + rng.uniform(0, 6.28)) * np.sin(Z / (2.3 * bed_len) + rng.uniform(0, 6.28))
    R = R * prof * bm[:, None] * wav - notch[:, None]
    # weathered shoulders: the top rounds off (a cut-flat top reads as a stump or a column)
    shoulder = rng.uniform(0.2, 0.45) if shoulder is None else shoulder
    R = R * (1.0 - shoulder * np.clip((u - 0.8) / 0.2, 0.0, 1.0) ** 2)[:, None]
    # a broken crown: wedges bitten out of the top edge, so the summit is stepped and jagged, never a flat cap
    bite = np.zeros_like(th)
    for _ in range(int(rng.integers(2, 5))):
        c0, wdt = rng.uniform(0, 2 * math.pi), rng.uniform(0.25, 0.9)
        d = np.abs(np.arctan2(np.sin(th - c0), np.cos(th - c0))) / wdt
        bite = np.maximum(bite, np.clip(1.0 - d, 0.0, 1.0) ** 0.6 * rng.uniform(0.35, 0.8) * (1.0 if bites else 0.0))
    if keep is not None:                                 # (angle, half-width): leave this side of the crown whole
        dk = np.abs(np.arctan2(np.sin(th - keep[0]), np.cos(th - keep[0]))) / keep[1]
        bite = bite * np.clip(dk - 0.6, 0.0, 1.0)
    top_u = np.clip((u - 0.8) / 0.2, 0.0, 1.0) ** 1.3
    R = R * (1.0 - top_u[:, None] * bite[None, :])
    R = np.maximum(R, 0.1 * R0)
    if lean is None:
        a = rng.uniform(0, 2 * math.pi)
        mag = {'spire': rng.uniform(0.04, 0.10), 'split': rng.uniform(0.02, 0.07)}.get(style, rng.uniform(0.0, 0.04))
        lean = (mag * math.cos(a) * H, mag * math.sin(a) * H)
    uu = ((Z - zb) / H) ** 1.5
    ox = cx + lean[0] * uu + 0.07 * R0 * np.sin(Z / (0.45 * H) + seed) + 0.03 * R0 * np.sin(Z / 11.0 + 3 * seed)
    oy = cy + lean[1] * uu + 0.07 * R0 * np.cos(Z / (0.38 * H) + 2 * seed) + 0.03 * R0 * np.cos(Z / 9.0 + seed)
    V = [np.stack([ox + R * np.cos(TH), oy + R * np.sin(TH), Z], -1).reshape(-1, 3)]
    veg = [np.clip(notch[:, None] / max(0.035 * R0, 1e-3) * 0.5 + 0.0 * TH, 0, 0.45).reshape(-1)]
    nc = 0
    if closed:
        rt_top = R[-1]
        ctop = zt + cap * R0
        nc = max(6, int(math.ceil(float(rt_top.mean()) * math.pi / 2 / (dcap or dz))))
        rngc = np.random.default_rng(int(abs(seed) * 313) % (2 ** 31))
        waves = [(rngc.uniform(0, 2 * math.pi), 2 * math.pi / rngc.uniform(1.2, 6.0), rngc.uniform(0, 2 * math.pi),
                  rngc.uniform(0.4, 1.0)) for _ in range(9)]
        wsum = sum(w[3] for w in waves)
        for c in range(1, nc + 1):
            a = c / nc * math.pi / 2
            rr = rt_top * math.cos(a)
            zz = zt + (ctop - zt) * math.sin(a)
            px = ox[-1] + rr * np.cos(th)
            py = oy[-1] + rr * np.sin(th)
            rocky = sum(A * np.sin(k * (math.cos(d) * px + math.sin(d) * py) + ph) for d, k, ph, A in waves) / wsum
            # broken rubble on the crown, strongest just inside the rim
            V.append(np.stack([px, py, zz + rock * (0.55 + 0.45 * math.sin(2 * a)) * rocky * math.sin(a) ** 0.5], -1))
            veg.append(np.full(nt, 0.15))
        V.append(np.array([[ox[-1, 0], oy[-1, 0], ctop + 0.02 * R0]]))
        veg.append(np.array([0.15]))
    V = np.vstack(V).astype(np.float32)
    veg = np.concatenate(veg).astype(np.float32)
    rows = nz + nc
    i = np.arange(nt)
    i2 = (i + 1) % nt
    Q = [np.stack([j * nt + i, j * nt + i2, (j + 1) * nt + i2, (j + 1) * nt + i], 1) for j in range(rows - 1)]
    if closed:
        a = (rows - 1) * nt
        Q.append(np.stack([a + i, a + i2, np.full(nt, rows * nt), np.full(nt, -1)], 1))
    return V, np.vstack(Q).astype(np.int32), veg
