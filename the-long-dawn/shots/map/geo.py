"""Geography for the MAP: projection, land rings, lakes, and coarse fields (elevation, ruggedness,
forest, desert) derived from public-domain data.

Map coordinates (X, Y) are in "map degrees": X = longitude relative to the central meridian CM
(wrapped to [-180, 180)), Y = yproj(latitude), a gentle cylindrical projection between plate
carree and Miller (continents look familiar, the poles are not blown up).

Sources: Natural Earth 50m land (public domain); the three.js Earth normal map in assets/textures
(integrated into an elevation proxy); NASA Blue Marble (public domain, via the GLOBE cache) for
forest / desert / lake classification only.
"""
import json
import os

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
CACHE = os.path.join(ROOT, 'renders', 'map_C', 'cache')
os.makedirs(CACHE, exist_ok=True)

CM = 11.0                      # central meridian: the map's seam runs down the Bering Strait
A_COEF = 0.08                  # y = phi (1 + a phi^2)
LAT_TOP, LAT_BOT = 85.0, -60.0


def yproj(lat):
    ph = np.radians(np.asarray(lat, np.float64))
    return np.degrees(ph * (1.0 + A_COEF * ph * ph))


_YT = np.linspace(-90.0, 90.0, 18001)
_YV = yproj(_YT)


def ilat(Y):
    return np.interp(np.asarray(Y, np.float64), _YV, _YT)


def lon2x(lon):
    return (np.asarray(lon, np.float64) - CM + 180.0) % 360.0 - 180.0


def x2lon(X):
    return (np.asarray(X, np.float64) + CM + 180.0) % 360.0 - 180.0


def ll2map(lat, lon):
    return np.stack([lon2x(lon), yproj(lat)], -1)


MAP_X0, MAP_X1 = -180.0, 180.0
MAP_Y0, MAP_Y1 = float(yproj(LAT_BOT)), float(yproj(LAT_TOP))
# the neatline is the map frame; the border band sits outside it, then a parchment margin
BORDER_W = 2.6
PARCH_X0, PARCH_X1 = MAP_X0 - 7.5, MAP_X1 + 7.5
PARCH_Y0, PARCH_Y1 = MAP_Y0 - 7.5, MAP_Y1 + 7.5
# the texture covers a little table around the parchment
TEX_X0, TEX_X1 = -194.0, 194.0
TEX_Y0, TEX_Y1 = MAP_Y0 - 14.0, MAP_Y1 + 14.0


# ------------------------------------------------------------------ land ---

def _rings(path):
    g = json.load(open(path))
    out = []
    for f in g['features']:
        geom = f['geometry']
        polys = [geom['coordinates']] if geom['type'] == 'Polygon' else geom['coordinates']
        for p in polys:
            for k, r in enumerate(p):
                out.append((np.asarray(r, np.float64), k > 0))
    return out


_LAND = None


def land_rings():
    """[(lonlat (N,2), is_hole)] from Natural Earth 50m land."""
    global _LAND
    if _LAND is None:
        _LAND = _rings(os.path.join(ROOT, 'assets', 'data', 'ne_50m_land.geojson'))
    return _LAND


def to_map_ring(ll):
    """lon/lat ring -> map XY, unwrapped (continuous in X; may extend past +-180)."""
    x = lon2x(ll[:, 0])
    x = np.degrees(np.unwrap(np.radians(x)))
    y = yproj(ll[:, 1])
    return np.stack([x, y], 1)


def wrap_copies(xy):
    """Copies of a map polyline shifted by 0/+-360 so it can be clipped by the map frame."""
    out = [xy]
    if xy[:, 0].min() < -180.0:
        out.append(xy + [360.0, 0.0])
    if xy[:, 0].max() > 180.0:
        out.append(xy - [360.0, 0.0])
    return out


# ------------------------------------------------------ equirect fields ---

EQ_W, EQ_H = 2048, 1024


def _land_mask_eq(W, H):
    m = np.zeros((H, W), np.uint8)
    for ll, hole in land_rings():
        pts = np.stack([(ll[:, 0] + 180.0) / 360.0 * W, (90.0 - ll[:, 1]) / 180.0 * H], 1)
        cv2.fillPoly(m, [np.round(pts * 16).astype(np.int32)], 0 if hole else 255, cv2.LINE_AA, shift=4)
    return m


_FIELDS = None


def fields():
    """Equirect 2048x1024 fields: land (0..1), elev (relative, ~0 at sea), rug (slope), forest,
    desert (0..1). Cached (disk and memory)."""
    global _FIELDS
    if _FIELDS is not None:
        return _FIELDS
    path = os.path.join(CACHE, 'fields.npz')
    if os.path.exists(path):
        d = np.load(path)
        _FIELDS = {k: d[k].astype(np.float32) for k in d.files}
        return _FIELDS
    W, H = EQ_W, EQ_H
    land = _land_mask_eq(W, H).astype(np.float32) / 255.0
    # elevation proxy: Frankot-Chellappa integration of the relief normal map
    nm = cv2.imread(os.path.join(ROOT, 'assets', 'textures', 'earth_normal_2048.jpg')).astype(np.float32)[..., ::-1] / 255.0 * 2 - 1
    nz = np.maximum(nm[..., 2], 0.2)
    p = -nm[..., 0] / nz
    q = -nm[..., 1] / nz
    P = np.concatenate([p, p[::-1]], 0)
    Q = np.concatenate([q, -q[::-1]], 0)
    wy = np.fft.fftfreq(2 * H)[:, None] * 2 * np.pi
    wx = np.fft.fftfreq(W)[None, :] * 2 * np.pi
    den = wx ** 2 + wy ** 2
    den[0, 0] = 1
    Z = (-1j * wx * np.fft.fft2(P) - 1j * wy * np.fft.fft2(Q)) / den
    Z[0, 0] = 0
    z = np.real(np.fft.ifft2(Z))[:H].astype(np.float32)
    sea = cv2.erode((land < 0.5).astype(np.uint8), np.ones((5, 5), np.uint8)).astype(np.float32)
    num = cv2.GaussianBlur(z * sea, (0, 0), 40, borderType=cv2.BORDER_REFLECT)
    dd = cv2.GaussianBlur(sea, (0, 0), 40, borderType=cv2.BORDER_REFLECT)
    elev = (z - num / np.maximum(dd, 1e-3)) * land
    rug = np.sqrt(p * p + q * q).astype(np.float32)
    rug = cv2.GaussianBlur(rug, (0, 0), 1.2) * land
    # biomes from NASA Blue Marble (classification only; no imagery is shown)
    bm = cv2.imread(os.path.join(ROOT, 'renders', 'globe', 'cache', 'earth-blue-marble.jpg'))
    bm = cv2.resize(bm, (W, H), interpolation=cv2.INTER_AREA)[..., ::-1].astype(np.float32)
    R, G, B = bm[..., 0], bm[..., 1], bm[..., 2]
    tot = R + G + B
    green = G - 0.5 * (R + B)
    dark = np.clip((150.0 - tot) / 90.0, 0, 1)
    forest = np.clip((green - 4.0) / 10.0, 0, 1) * dark * land
    forest = cv2.GaussianBlur(forest, (0, 0), 1.0)
    sandy = np.clip((R - B - 45.0) / 40.0, 0, 1) * np.clip((tot - 330.0) / 120.0, 0, 1)
    desert = cv2.GaussianBlur(sandy * land, (0, 0), 1.5)
    np.savez_compressed(path, land=land.astype(np.float16), elev=elev.astype(np.float16),
                        rug=rug.astype(np.float16), forest=forest.astype(np.float16),
                        desert=desert.astype(np.float16))
    return fields()


def sample_eq(field, lat, lon):
    """Bilinear sample of an equirect field at lat/lon arrays."""
    H, W = field.shape
    x = (np.asarray(lon) + 180.0) / 360.0 * W - 0.5
    y = (90.0 - np.asarray(lat)) / 180.0 * H - 0.5
    return cv2.remap(np.asarray(field, np.float32), x.astype(np.float32).reshape(-1, 1),
                     y.astype(np.float32).reshape(-1, 1), cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_WRAP).reshape(np.shape(lat))


# ----------------------------------------------------------------- lakes ---

def lakes():
    """Large lakes as lon/lat rings, traced from the Blue Marble (dark water inside land)."""
    path = os.path.join(CACHE, 'lakes.npz')
    if os.path.exists(path):
        d = np.load(path)
        pts, off = d['pts'], d['off']
        return [pts[off[i]:off[i + 1]] for i in range(len(off) - 1)]
    bm = cv2.imread(os.path.join(ROOT, 'renders', 'globe', 'cache', 'earth-blue-marble.jpg'))[..., ::-1].astype(np.float32)
    H, W = bm.shape[:2]
    land = _land_mask_eq(W, H)
    lake = ((bm.sum(2) < 40) & (land > 200)).astype(np.uint8)
    lake = cv2.morphologyEx(lake, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    # smooth the blobs before tracing (supersample x4)
    big = cv2.resize(lake * 255, (W * 2, H * 2), interpolation=cv2.INTER_LINEAR)
    big = cv2.GaussianBlur(big, (0, 0), 1.6)
    n, lab, st, cen = cv2.connectedComponentsWithStats((big > 127).astype(np.uint8), 8)
    rings = []
    for i in range(1, n):
        if st[i, 4] < 4 * 26:
            continue
        # skip high-arctic ice-shelf artefacts
        la = 90.0 - cen[i][1] / (2 * H) * 180.0
        if la > 75:
            continue
        m = (lab == i).astype(np.uint8)
        cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for c in cs:
            c = c[:, 0, :].astype(np.float64)
            if len(c) < 8:
                continue
            c = c[::2]
            lon = (c[:, 0] + 0.5) / (2 * W) * 360.0 - 180.0
            lat = 90.0 - (c[:, 1] + 0.5) / (2 * H) * 180.0
            rings.append(np.stack([lon, lat], 1))
    off = np.cumsum([0] + [len(r) for r in rings])
    np.savez_compressed(path, pts=np.concatenate(rings), off=off)
    return rings


if __name__ == '__main__':
    f = fields()
    print({k: (v.shape, float(v.min()), float(v.max())) for k, v in f.items()})
    L = lakes()
    print('lakes', len(L))
    print('map Y range', MAP_Y0, MAP_Y1)
