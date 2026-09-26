"""Bake the map: parchment + ink, in map space, into a mip pyramid of tiles.

    python bake.py preview world 6            # quick look at the whole sheet (6 px/deg)
    python bake.py preview region X0 Ytop W H ppd
    python bake.py pyramid                    # full bake: level 0 (40 px/deg) in tiles + mips

A region is baked by bake_region(X0, Ytop, W, H, ppd) -> dict of float32 layers:
  'rgb'   linear-light albedo of the sheet (paper, washes, ink) incl. the table beyond its edge
  'h'     paper relief (cockle, folds, edge curl), map-degree units, for the renderer's lighting
Everything is a function of map position, so tiles at any resolution agree.
"""
import math
import os
import sys
import time

import cv2
import numpy as np

import geo
import ink
import noise as nz
import features as ft
import sheet

cv2.setNumThreads(2)

LEVEL0_PPD = 40.0
TILE = 1536
PAD = 160                        # px of context around a tile (ripples, glyphs, blur)

# paper and ink colours (sRGB 0..255 -> linear on use)
PAPER = np.array([226, 204, 160], np.float32)
PAPER_DARK = np.array([196, 160, 108], np.float32)
PAPER_STAIN = np.array([170, 120, 70], np.float32)
INK = np.array([44, 30, 22], np.float32)
INK_LIGHT = np.array([92, 62, 40], np.float32)
TABLE = np.array([22, 14, 10], np.float32)


def s2l(c):
    c = np.asarray(c, np.float32) / 255.0
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4).astype(np.float32)


def l2s(c):
    c = np.clip(c, 0, 1)
    return np.where(c <= 0.0031308, 12.92 * c, 1.055 * np.power(c, 1 / 2.4) - 0.055)


_FEAT = None
_COAST = None
_LAKES = None
_RIVERS = None
_GLY = None


def feats():
    global _FEAT, _COAST, _LAKES, _RIVERS
    if _FEAT is None:
        _FEAT = ft.build()
        _COAST = ft.coast_lines()
        _LAKES = ft.lake_lines()
        _RIVERS = ft.river_lines()
    return _FEAT, _COAST, _LAKES, _RIVERS


# ------------------------------------------------------- glyph geometry ---

def glyph_bank():
    """All glyphs pre-generated (fill polygons + strokes), painter-ordered, cached."""
    global _GLY
    if _GLY is not None:
        return _GLY
    path = os.path.join(geo.CACHE, 'glyphbank.npz')
    if os.path.exists(path):
        d = np.load(path)
        _GLY = {k: d[k] for k in d.files}
        return _GLY
    F, _, _, _ = feats()
    G = F['glyphs']
    fillP, fillO = [], [0]
    shP, shO = [], [0]
    sP, sR, sD, sO = [], [], [], [0]
    gS = [0]
    bb = []
    for g in G:
        fill, st, shade = ft.make_glyph(g)
        fillP.append(fill.astype(np.float32))
        fillO.append(fillO[-1] + len(fill))
        if shade is None:
            shade = np.zeros((0, 2))
        shP.append(shade.astype(np.float32))
        shO.append(shO[-1] + len(shade))
        P, R, D, so = st.arrays()
        for k in range(len(so) - 1):
            sP.append(P[so[k]:so[k + 1]].astype(np.float32))
            sR.append(R[so[k]:so[k + 1]].astype(np.float32))
            sD.append(D[so[k]:so[k + 1]].astype(np.float32))
            sO.append(sO[-1] + so[k + 1] - so[k])
        gS.append(len(sO) - 1)
        allp = np.concatenate([fill, P])
        m = float(R.max()) if len(R) else 0.0
        bb.append((allp[:, 0].min() - m, allp[:, 0].max() + m, allp[:, 1].min() - m, allp[:, 1].max() + m))
    _GLY = dict(fillP=np.concatenate(fillP), fillO=np.array(fillO, np.int64),
                shP=np.concatenate(shP), shO=np.array(shO, np.int64),
                sP=np.concatenate(sP), sR=np.concatenate(sR), sD=np.concatenate(sD),
                sO=np.array(sO, np.int64), gS=np.array(gS, np.int64), bb=np.array(bb, np.float32),
                kind=G[:, 1].astype(np.int8))
    np.savez(path, **_GLY)
    return _GLY


def draw_glyphs(A, X0, Y1, ppd, Wsh=None):
    """Painter's-order glyphs: each erases the ink (and shadow wash) under its silhouette, then
    lays its shadow wash and draws its strokes."""
    gb = glyph_bank()
    Hh, Ww = A.shape
    X1 = X0 + Ww / ppd
    Y0 = Y1 - Hh / ppd
    bb = gb['bb']
    sel = np.where((bb[:, 1] > X0) & (bb[:, 0] < X1) & (bb[:, 3] > Y0) & (bb[:, 2] < Y1))[0]
    fillP, fillO = gb['fillP'], gb['fillO']
    shP, shO = gb['shP'], gb['shO']
    sP, sR, sD, sO, gS = gb['sP'], gb['sR'], gb['sD'], gb['sO'], gb['gS']
    kinds = gb['kind']
    for gi in sel:
        fp = fillP[fillO[gi]:fillO[gi + 1]].astype(np.float64)
        px = (fp[:, 0] - X0) * ppd - 0.5
        py = (Y1 - fp[:, 1]) * ppd - 0.5
        x0 = max(int(np.floor(px.min())) - 1, 0)
        y0 = max(int(np.floor(py.min())) - 1, 0)
        x1 = min(int(np.ceil(px.max())) + 2, Ww)
        y1 = min(int(np.ceil(py.max())) + 2, Hh)
        if x1 <= x0 or y1 <= y0:
            continue
        m = np.zeros((y1 - y0, x1 - x0), np.uint8)
        pts = np.stack([px - x0, py - y0], 1)
        cv2.fillPoly(m, [np.round(pts * 16).astype(np.int32)], 255, cv2.LINE_AA, shift=4)
        cov = m.astype(np.float32) / 255.0
        A[y0:y1, x0:x1] *= 1.0 - cov
        if Wsh is not None:
            Wsh[y0:y1, x0:x1] *= 1.0 - cov
            if shO[gi + 1] > shO[gi]:
                sp = shP[shO[gi]:shO[gi + 1]].astype(np.float64)
                m[:] = 0
                pts = np.stack([(sp[:, 0] - X0) * ppd - 0.5 - x0, (Y1 - sp[:, 1]) * ppd - 0.5 - y0], 1)
                cv2.fillPoly(m, [np.round(pts * 16).astype(np.int32)], 255, cv2.LINE_AA, shift=4)
                k = 0.42 if kinds[gi] == 0 else 0.3
                Wsh[y0:y1, x0:x1] = np.maximum(Wsh[y0:y1, x0:x1], m.astype(np.float32) / 255.0 * k)
        a, b = sO[gS[gi]], sO[gS[gi + 1]]
        if b > a:
            P = sP[a:b].astype(np.float64)
            st = (sO[gS[gi]:gS[gi + 1] + 1] - a).astype(np.int64)
            ink._draw(A, (P[:, 0] - X0) * ppd - 0.5, (Y1 - P[:, 1]) * ppd - 0.5,
                      sR[a:b].astype(np.float64) * ppd, sD[a:b].astype(np.float64), st)


# ------------------------------------------------------------ the sheet ---

def grid(X0, Y1, W, H, ppd):
    X = X0 + (np.arange(W) + 0.5) / ppd
    Y = Y1 - (np.arange(H) + 0.5) / ppd
    return X, Y


def _ring_area(P):
    x, y = P[:, 0], P[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


_AREAS = None


def land_mask(X0, Y1, W, H, ppd, min_area=0.0):
    """Anti-aliased land (1) / water (0), lakes cut out. min_area drops small islands."""
    global _AREAS
    _, coast, lakes, _ = feats()
    if _AREAS is None:
        _AREAS = [(_ring_area(P) if not hole else 0.0) for P, hole in coast]
    m = np.zeros((H, W), np.uint8)
    X1 = X0 + W / ppd
    Y0 = Y1 - H / ppd
    for (P, hole), ar in zip(coast, _AREAS):
        if not hole and ar < min_area:
            continue
        if P[:, 0].max() < X0 - 1 or P[:, 0].min() > X1 + 1 or P[:, 1].max() < Y0 - 1 or P[:, 1].min() > Y1 + 1:
            continue
        pts = np.stack([(P[:, 0] - X0) * ppd - 0.5, (Y1 - P[:, 1]) * ppd - 0.5], 1)
        cv2.fillPoly(m, [np.round(pts * 16).astype(np.int32)], 0 if hole else 255, cv2.LINE_AA, shift=4)
    for P in lakes:
        if P[:, 0].max() < X0 - 1 or P[:, 0].min() > X1 + 1 or P[:, 1].max() < Y0 - 1 or P[:, 1].min() > Y1 + 1:
            continue
        pts = np.stack([(P[:, 0] - X0) * ppd - 0.5, (Y1 - P[:, 1]) * ppd - 0.5], 1)
        cv2.fillPoly(m, [np.round(pts * 16).astype(np.int32)], 0, cv2.LINE_AA, shift=4)
    return m.astype(np.float32) / 255.0


def _dist(land, ppd):
    sea = (land < 0.5).astype(np.uint8)
    return cv2.distanceTransform(sea, cv2.DIST_L2, 5).astype(np.float32) / ppd


def sea_marks(land, land_big, X0, Y1, ppd):
    """Coastal ink wash + water-lining (ripples) from the distance to land (map degrees)."""
    H, W = land.shape
    d = _dist(land, ppd)
    db = _dist(land_big, ppd)
    sig = max(0.10 * ppd, 0.6)
    dbs = cv2.GaussianBlur(db, (0, 0), sig)
    dbs = np.where(db < 0.05, db, dbs)
    wash = 0.6 * np.exp(-dbs / 0.5) + 0.3 * np.exp(-dbs / 2.8)
    wash *= (1.0 - land_big)
    brk = nz.fbm_grid(X0, Y1, 1.0 / ppd, 1.0 / ppd, H, W, 0.7, 505, 3, 2.0, 0.5)
    rip = np.zeros((H, W), np.float32)
    for k, (dk, r, dens) in enumerate(((0.2, 0.020, 0.85), (0.46, 0.017, 0.72), (0.8, 0.015, 0.6), (1.22, 0.013, 0.45))):
        dd = d if k == 0 else dbs
        rp = r * ppd
        c = np.clip(0.5 + (max(rp, 0.5) - np.abs(dd - dk) * ppd), 0, 1) * min(1.0, rp / 0.5)
        gate = 1.0 if k == 0 else np.clip((brk + 0.25 - 0.12 * k) / 0.15, 0, 1)
        rip = np.maximum(rip, c * dens * gate)
    rip *= (1.0 - land)
    return wash.astype(np.float32), rip.astype(np.float32)


def fill_ink(A, polys, X0, Y1, ppd, dens=0.95):
    if not polys:
        return
    H, W = A.shape
    m = np.zeros((H, W), np.uint8)
    for P in polys:
        pts = np.stack([(P[:, 0] - X0) * ppd - 0.5, (Y1 - P[:, 1]) * ppd - 0.5], 1)
        cv2.fillPoly(m, [np.round(pts * 16).astype(np.int32)], 255, cv2.LINE_AA, shift=4)
    np.maximum(A, m.astype(np.float32) / 255.0 * dens, out=A)


def erase(A, P, X0, Y1, ppd):
    H, W = A.shape
    m = np.zeros((H, W), np.uint8)
    pts = np.stack([(P[:, 0] - X0) * ppd - 0.5, (Y1 - P[:, 1]) * ppd - 0.5], 1)
    cv2.fillPoly(m, [np.round(pts * 16).astype(np.int32)], 255, cv2.LINE_AA, shift=4)
    A *= 1.0 - m.astype(np.float32) / 255.0


_FURN = None


def furniture():
    global _FURN
    if _FURN is None:
        _FURN = (sheet.border(), sheet.rose(), sheet.stain_list(), sheet.spot_list())
    return _FURN


def neat_mask(X0, Y1, W, H, ppd):
    X, Y = grid(X0, Y1, W, H, ppd)
    mx = np.clip(0.5 + (np.minimum(X - geo.MAP_X0, geo.MAP_X1 - X)) * ppd, 0, 1)
    my = np.clip(0.5 + (np.minimum(Y - geo.MAP_Y0, geo.MAP_Y1 - Y)) * ppd, 0, 1)
    return (my[:, None] * mx[None, :]).astype(np.float32)


def bake_region(X0, Y1, W, H, ppd, pad=None):
    """Bake the sheet for pixels [0,W)x[0,H) whose top-left corner is map (X0, Y1)."""
    if pad is None:
        pad = int(min(PAD, max(24, 2.6 * ppd)))
    Xp, Yp = X0 - pad / ppd, Y1 + pad / ppd
    Wp, Hp = W + 2 * pad, H + 2 * pad
    F, coast, lakes, rivers = feats()
    (bst, bfills), rose_items, stains, spots = furniture()
    land = land_mask(Xp, Yp, Wp, Hp, ppd)
    land_big = land_mask(Xp, Yp, Wp, Hp, ppd, min_area=0.6)
    wash, rip = sea_marks(land, land_big, Xp, Yp, ppd)
    A = np.zeros((Hp, Wp), np.float32)
    # coasts: pressure varies slowly around each ring
    for P, hole in coast:
        if P[:, 0].max() < Xp - 1 or P[:, 0].min() > Xp + Wp / ppd + 1:
            continue
        if P[:, 1].max() < Yp - Hp / ppd - 1 or P[:, 1].min() > Yp + 1:
            continue
        Pc = np.concatenate([P, P[:1]])
        s = ink.arclen(Pc)
        pr = 0.85 + 0.25 * np.sin(s * 2.3 + P[0, 0]) + 0.12 * np.sin(s * 7.1 + P[0, 1])
        ink._draw(A, (Pc[:, 0] - Xp) * ppd - 0.5, (Yp - Pc[:, 1]) * ppd - 0.5,
                  0.042 * pr * ppd, np.full(len(Pc), 0.97), np.array([0, len(Pc)], np.int64))
    for P in lakes:
        ink._draw(A, (P[:, 0] - Xp) * ppd - 0.5, (Yp - P[:, 1]) * ppd - 0.5,
                  np.full(len(P), 0.024 * ppd), np.full(len(P), 0.93), np.array([0, len(P)], np.int64))
    for P, R, name in rivers:
        ink._draw(A, (P[:, 0] - Xp) * ppd - 0.5, (Yp - P[:, 1]) * ppd - 0.5,
                  R * ppd, np.full(len(P), 0.9), np.array([0, len(P)], np.int64))
    dots = F['dots']
    rng = np.random.default_rng(7)
    rad = 0.016 + 0.012 * rng.random(len(dots))
    ink.dots(A, dots, rad, np.full(len(dots), 0.8), Xp, Yp, ppd)
    Wsh = np.zeros((Hp, Wp), np.float32)
    draw_glyphs(A, Xp, Yp, ppd, Wsh)
    A = np.maximum(A, rip)
    # the rose sits on the sea: its paper disc erases ripples and wash beneath
    for disc, solids, st in rose_items:
        erase(A, disc, Xp, Yp, ppd)
        fill_ink(A, solids, Xp, Yp, ppd, 0.93)
        ink.draw(A, st, Xp, Yp, ppd)
    cx, cy = sheet.ROSE_C
    Xg, Yg = grid(Xp, Yp, Wp, Hp, ppd)
    rr = np.sqrt((Xg[None, :] - cx) ** 2 + (Yg[:, None] - cy) ** 2)
    wash *= np.clip((rr - sheet.ROSE_R * 1.02) * 2.0, 0, 1)
    # everything geographic stays inside the neatline
    nm = neat_mask(Xp, Yp, Wp, Hp, ppd)
    A *= nm
    wash *= nm
    fill_ink(A, bfills, Xp, Yp, ppd, 0.92)
    ink.draw(A, bst, Xp, Yp, ppd)
    # ---- the parchment
    Xe, Ye = Xp + Wp / ppd, Yp - Hp / ppd
    sm = (stains[:, 0] + 1.7 * stains[:, 2] > Xp) & (stains[:, 0] - 1.7 * stains[:, 2] < Xe) & \
         (stains[:, 1] + 1.7 * stains[:, 2] > Ye) & (stains[:, 1] - 1.7 * stains[:, 2] < Yp)
    pm = (spots[:, 0] + 3 * spots[:, 2] > Xp) & (spots[:, 0] - 3 * spots[:, 2] < Xe) & \
         (spots[:, 1] + 3 * spots[:, 2] > Ye) & (spots[:, 1] - 3 * spots[:, 2] < Yp)
    ch, hgt = sheet.parchment(Xp, Yp, Hp, Wp, float(ppd), np.ascontiguousarray(stains[sm]),
                              np.ascontiguousarray(spots[pm]))
    t, stn, ed, fw, gr = ch[..., 0], ch[..., 1], ch[..., 2], ch[..., 3], ch[..., 4]
    t = np.clip(t, 0, 1)
    base = s2l(PAPER)[None, None, :] * (1 - 0.6 * t[..., None]) + s2l(PAPER_DARK)[None, None, :] * (0.6 * t[..., None])
    base *= (1.0 + 0.075 * gr)[..., None]
    stn = np.clip(stn, 0, 1.2)
    base = base * (1 - 0.55 * stn[..., None]) + s2l(PAPER_STAIN)[None, None, :] * (0.55 * stn[..., None])
    # aged edge: browner toward the rim, a darker line right at it
    age = np.clip(1 - ed / 5.0, 0, 1) ** 1.6
    base = base * (1 - 0.45 * age[..., None]) + s2l(np.array([150, 105, 60]))[None, None, :] * (0.45 * age[..., None])
    base *= (1 - 0.35 * np.exp(-np.maximum(ed, 0) / 0.18))[..., None]
    # folds: dirt in the crease, a lighter worn band beside it
    base *= (1 - 0.18 * np.exp(-((fw - 1.0) / 0.02) ** 2) * 0 - 0.12 * fw ** 8)[..., None]
    base *= (1 + 0.02 * (fw - fw ** 8))[..., None]
    # ink: pen pressure and wear (folds, grain)
    inkvar = nz.fbm_grid(Xp, Yp, 1.0 / ppd, 1.0 / ppd, Hp, Wp, 0.08, 606, 3, 2.0, 0.5)
    # ragged ink edges: the paper's tooth takes more or less ink at the edge of a stroke
    Ar = np.clip(A + 0.22 * gr * (A * (1 - A)) * 4.0, 0, 1)
    Ae = Ar * np.clip(0.9 + 0.35 * inkvar, 0.6, 1.0) * (1 - 0.35 * fw)
    sea = (1.0 - land_big) * nm
    wtot = np.clip(0.16 * sea + 0.55 * wash, 0, 1)
    wash_col = base * np.array([0.62, 0.6, 0.56], np.float32)[None, None, :]
    rgb = base * (1 - wtot[..., None]) + wash_col * wtot[..., None]
    # sepia shadow wash on the east faces of peaks and hills (soft-edged, like a brush)
    Wsh = cv2.GaussianBlur(Wsh, (0, 0), max(0.035 * ppd, 0.5)) * nm * np.clip(0.8 + 0.5 * inkvar, 0.4, 1.1)
    shade_col = s2l(np.array([150, 112, 72], np.float32))
    rgb = rgb * (1 - 0.5 * Wsh[..., None]) + (rgb * shade_col / s2l(PAPER))[..., :] * (0.5 * Wsh[..., None])
    inkc = s2l(INK)[None, None, :] * (1 - 0.35 * (1 - A[..., None])) + s2l(INK_LIGHT)[None, None, :] * (0.35 * (1 - A[..., None]))
    rgb = rgb * (1 - Ae[..., None]) + inkc * Ae[..., None]
    # beyond the edge: the table
    inside = np.clip(0.5 + ed * ppd, 0, 1)[..., None]
    wood = nz.fibre_grid(Xp, Yp, 1.0 / ppd, 1.0 / ppd, Hp, Wp, 1.5, 808, 0.06)
    table = s2l(TABLE)[None, None, :] * (0.8 + 0.5 * wood[..., None])
    rgb = rgb * inside + table * (1 - inside)
    rgb = rgb[pad:pad + H, pad:pad + W]
    return dict(rgb=rgb.astype(np.float32), h=hgt[pad:pad + H, pad:pad + W], A=A[pad:pad + H, pad:pad + W],
                inside=inside[pad:pad + H, pad:pad + W, 0])


# ------------------------------------------------------------ pyramid ---

def tex_dims(ppd):
    W = int(round((geo.TEX_X1 - geo.TEX_X0) * ppd))
    H = int(round((geo.TEX_Y1 - geo.TEX_Y0) * ppd))
    return W, H


def level_path(tag, L):
    return os.path.join(geo.CACHE, f'tex_{tag}_L{L}.npy')


def _bake_tiles(args):
    tag, ppd, tiles, W, H = args
    mm = np.lib.format.open_memmap(level_path(tag, 0), mode='r+')
    for (ty, tx) in tiles:
        x0, y0 = tx * TILE, ty * TILE
        w, h = min(TILE, W - x0), min(TILE, H - y0)
        t0 = time.time()
        r = bake_region(geo.TEX_X0 + x0 / ppd, geo.TEX_Y1 - y0 / ppd, w, h, ppd)
        srgb = l2s(r['rgb'])
        rng = np.random.default_rng(ty * 1000 + tx)
        srgb = srgb + (rng.random(srgb.shape, np.float32) - rng.random(srgb.shape, np.float32)) / 255.0
        mm[y0:y0 + h, x0:x0 + w] = np.clip(np.round(srgb * 255), 0, 255).astype(np.uint8)
        mm.flush()
        print(f'  tile {ty},{tx} {w}x{h} {time.time() - t0:.1f}s', flush=True)
    return len(tiles)


def downsample(tag, L):
    """Level L+1 from level L, box-filtered in linear light, in strips."""
    src = np.load(level_path(tag, L), mmap_mode='r')
    H, W = src.shape[:2]
    H2, W2 = H // 2, W // 2
    dst = np.lib.format.open_memmap(level_path(tag, L + 1), mode='w+', dtype=np.uint8, shape=(H2, W2, 3))
    lut = s2l(np.arange(256, dtype=np.float32))
    strip = 1024
    for y in range(0, 2 * H2, strip):
        blk = lut[np.asarray(src[y:min(y + strip, 2 * H2), :2 * W2])]
        small = cv2.resize(blk, (W2, blk.shape[0] // 2), interpolation=cv2.INTER_AREA)
        dst[y // 2:y // 2 + small.shape[0]] = np.clip(np.round(l2s(small) * 255), 0, 255).astype(np.uint8)
    dst.flush()
    return H2, W2


def relief(tag='h', ppd=4.0):
    """Low-res paper relief (cockle, folds, edge curl) for the whole texture."""
    W, H = tex_dims(ppd)
    (bst, bf), ri, st, sp = furniture()
    _, h = sheet.parchment(geo.TEX_X0, geo.TEX_Y1, H, W, ppd, st[:0], sp[:0])
    np.save(os.path.join(geo.CACHE, f'relief_{int(ppd)}.npy'), h.astype(np.float32))


def pyramid(tag, ppd, levels=6, workers=2):
    from multiprocessing import Pool
    W, H = tex_dims(ppd)
    print('level 0', W, H, f'{ppd} px/deg')
    feats()
    glyph_bank()
    mm = np.lib.format.open_memmap(level_path(tag, 0), mode='w+', dtype=np.uint8, shape=(H, W, 3))
    del mm
    tiles = [(ty, tx) for ty in range((H + TILE - 1) // TILE) for tx in range((W + TILE - 1) // TILE)]
    jobs = [(tag, ppd, tiles[k::workers], W, H) for k in range(workers)]
    t0 = time.time()
    with Pool(workers) as pool:
        pool.map(_bake_tiles, jobs)
    print('level 0 baked in', round(time.time() - t0), 's')
    for L in range(levels - 1):
        print('level', L + 1, downsample(tag, L))
    relief()


def preview(X0, Y1, W, H, ppd, out):
    t0 = time.time()
    r = bake_region(X0, Y1, W, H, ppd)
    print('baked', W, H, 'at', ppd, 'px/deg in', round(time.time() - t0, 1), 's')
    cv2.imwrite(out, (l2s(r['rgb']) * 255).astype(np.uint8)[..., ::-1])


if __name__ == '__main__':
    S = '/private/tmp/claude-501/-Users-mishasalahshoor/d71369b9-7742-4da4-abf2-0e5d5e46099b/scratchpad/'
    if sys.argv[1] == 'pyramid':
        tag = sys.argv[2] if len(sys.argv) > 2 else 'full'
        ppd = float(sys.argv[3]) if len(sys.argv) > 3 else LEVEL0_PPD
        pyramid(tag, ppd)
    elif sys.argv[1] == 'preview':
        if sys.argv[2] == 'world':
            ppd = float(sys.argv[3])
            W = int(376 * ppd)
            H = int((geo.MAP_Y1 - geo.MAP_Y0 + 16) * ppd)
            preview(-188.0, geo.MAP_Y1 + 8, W, H, ppd, S + f'world_{int(ppd)}.png')
        else:
            X0, Y1, W, H, ppd = [float(v) for v in sys.argv[3:8]]
            preview(X0, Y1, int(W), int(H), ppd, S + sys.argv[8] if len(sys.argv) > 8 else S + 'region.png')
