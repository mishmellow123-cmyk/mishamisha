"""Geography for the MAP (MAP-L2): the sheet's frame, and the INVENTED world's coasts, lakes and fields.

The map is no longer our Earth. Every coast, range, river and lake comes from `terra.py`, an invented continent in
the same map-degree frame as before, so the camera, the bake and the relay did not have to change. This module
keeps only the sheet's frame (the neatline, the parchment and the texture's extent) and the gentle cylindrical
graticule the border band is ruled in. It reads no real-world data.

Map coordinates (X, Y) are "map degrees": X east in [-180, 180), Y north in [MAP_Y0, MAP_Y1].
"""
import os

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
# the invented world's caches (the real-Earth caches of v2/rev 3 lived in renders/map_C/cache and are retired)
CACHE = os.path.join(ROOT, 'renders', 'map_C', 'cache_w')
os.makedirs(CACHE, exist_ok=True)

# the sheet's graticule (the border band is ruled in it): a gentle cylindrical projection
CM = 11.0
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


MAP_X0, MAP_X1 = -180.0, 180.0
MAP_Y0, MAP_Y1 = float(yproj(LAT_BOT)), float(yproj(LAT_TOP))
# the neatline is the map frame; the border band sits outside it, then a parchment margin
BORDER_W = 2.6
PARCH_X0, PARCH_X1 = MAP_X0 - 7.5, MAP_X1 + 7.5
PARCH_Y0, PARCH_Y1 = MAP_Y0 - 7.5, MAP_Y1 + 7.5
# the texture covers a little table around the parchment
TEX_X0, TEX_X1 = -194.0, 194.0
TEX_Y0, TEX_Y1 = MAP_Y0 - 14.0, MAP_Y1 + 14.0


# ------------------------------------------------------ the invented world ---

def _terra():
    import terra
    return terra


def land_rings():
    """[(XY (N,2) map degrees, is_hole)]: the invented world's coasts."""
    return _terra().world()['coast']


def to_map_ring(xy):
    """(compatibility) rings are already in map coordinates."""
    return np.asarray(xy, np.float64)


def wrap_copies(xy):
    """Copies of a map polyline shifted by 0/+-360 so it can be clipped by the map frame."""
    out = [xy]
    if xy[:, 0].min() < -180.0:
        out.append(xy + [360.0, 0.0])
    if xy[:, 0].max() > 180.0:
        out.append(xy - [360.0, 0.0])
    return out


def lakes():
    """Lake outlines (map XY)."""
    return _terra().world()['lakes']


def rivers():
    """[(XY (N,2) source -> mouth, flow (N,), reaches the sea)]."""
    return _terra().world()['rivers']


def fields():
    """The invented world's rasters on terra's grid (terra.D map degrees per cell, row 0 at the sheet's top):
    land (bool), E (elevation units), Er (the ranges' part), rug (slope), rain, acc (flow), lake (bool)."""
    return _terra().world()


def sample(field, X, Y):
    """Bilinear sample of one of the world's rasters (by name or array) at map points X, Y (any shape)."""
    t = _terra()
    A = t.world()[field] if isinstance(field, str) else field
    A = np.asarray(A, np.float32)
    X = np.asarray(X, np.float64)
    Y = np.asarray(Y, np.float64)
    u = ((X - MAP_X0) / t.D - 0.5).astype(np.float32)
    v = ((MAP_Y1 - Y) / t.D - 0.5).astype(np.float32)
    out = cv2.remap(A, u.reshape(-1, 1), v.reshape(-1, 1), cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return out.reshape(X.shape)


def on_grid(field, X0, Y1, ppd, W, H):
    """One of the world's rasters resampled onto a pixel grid (pixel (i, j) centre at X0 + (j+.5)/ppd,
    Y1 - (i+.5)/ppd)."""
    t = _terra()
    A = t.world()[field] if isinstance(field, str) else field
    A = np.asarray(A, np.float32)
    xs = X0 + (np.arange(W) + 0.5) / ppd
    ys = Y1 - (np.arange(H) + 0.5) / ppd
    u = ((xs - MAP_X0) / t.D - 0.5).astype(np.float32)
    v = ((MAP_Y1 - ys) / t.D - 0.5).astype(np.float32)
    mx, my = np.meshgrid(u, v)
    interp = cv2.INTER_AREA if ppd < 1.0 / t.D else cv2.INTER_LINEAR
    if interp == cv2.INTER_AREA:
        # pre-blur to the target footprint, then sample
        k = (1.0 / t.D) / ppd
        A = cv2.GaussianBlur(A, (0, 0), 0.45 * k)
    return cv2.remap(A, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


if __name__ == '__main__':
    w = fields()
    print({k: (getattr(v, 'shape', len(v))) for k, v in w.items()})
    print('map Y range', MAP_Y0, MAP_Y1)
