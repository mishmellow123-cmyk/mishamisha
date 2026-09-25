"""GLOBE data prep. Builds every cache the renderer needs in renders/globe/cache/.

    python3 shots/globe/prep.py          # idempotent; rebuilds only what is missing

Sources
  assets/data/ne_50m_land.geojson            crisp land/sea mask (rasterised, anti-aliased)
  assets/data/ne_10m_populated_places_simple city light cores (population weighted)
  assets/textures/earth_clouds_1024.png      cloud coverage (low-frequency; detail is procedural)
  assets/textures/earth_normal_2048.jpg      relief normals
  assets/textures/earth_lights_2048.png      DMSP city lights (rural speckle guide)
  NASA Blue Marble NG 4k   (public domain)   land colour        } via npm package three-globe@2.45.2
  NASA Black Marble 2012 4k (public domain)  VIIRS night lights } example/img/, fetched with `npm pack`
"""
import json
import os
import subprocess
import sys
import tarfile

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'lib'))
import look  # noqa: E402

CACHE = os.path.join(ROOT, 'renders', 'globe', 'cache')
ASSETS = os.path.join(ROOT, 'assets')
MW, MH = 8192, 4096          # mask / albedo resolution


def cpath(name):
    return os.path.join(CACHE, name)


# ------------------------------------------------------------------ NASA ---

def fetch_nasa():
    need = [f for f in ('earth-blue-marble.jpg', 'earth-night.jpg') if not os.path.exists(cpath(f))]
    if not need:
        return
    tmp = cpath('npm')
    os.makedirs(tmp, exist_ok=True)
    tgz = os.path.join(tmp, 'three-globe-2.45.2.tgz')
    if not os.path.exists(tgz):
        subprocess.run(['npm', 'pack', 'three-globe@2.45.2', '--silent'], cwd=tmp, check=True)
    with tarfile.open(tgz) as tf:
        for f in need:
            data = tf.extractfile('package/example/img/' + f).read()
            with open(cpath(f), 'wb') as fh:
                fh.write(data)
    print('fetched', need)


# ------------------------------------------------------------- land mask ---

def _rings(geom):
    polys = [geom['coordinates']] if geom['type'] == 'Polygon' else geom['coordinates']
    for p in polys:
        yield p[0], p[1:]


def build_land_mask():
    out = cpath('land_mask_8k.png')
    if os.path.exists(out):
        return
    ss = 2
    W, H = MW * ss, MH * ss
    big = np.zeros((H, W), np.uint8)
    d = json.load(open(os.path.join(ASSETS, 'data', 'ne_50m_land.geojson')))

    def px(ring):
        a = np.asarray(ring, np.float64)
        x = (a[:, 0] + 180.0) / 360.0 * W
        y = (90.0 - a[:, 1]) / 180.0 * H
        return np.round(np.stack([x, y], 1) * 16).astype(np.int32)   # shift=4 sub-pixel

    holes = []
    for ft in d['features']:
        for ext, hs in _rings(ft['geometry']):
            cv2.fillPoly(big, [px(ext)], 255, lineType=cv2.LINE_AA, shift=4)
            holes += hs
    for h in holes:
        cv2.fillPoly(big, [px(h)], 0, lineType=cv2.LINE_AA, shift=4)
    small = cv2.resize(big, (MW, MH), interpolation=cv2.INTER_AREA)
    cv2.imwrite(out, small)
    print('land mask', small.shape, (small > 127).mean())


# ---------------------------------------------------------------- albedo ---

def push_pull(col, w, levels=9):
    """Fill colour into pixels with low weight from their neighbourhood (normalised pyramid)."""
    num = [col * w[..., None]]
    den = [w.copy()]
    for _ in range(levels):
        num.append(cv2.pyrDown(num[-1]))
        den.append(cv2.pyrDown(den[-1]))
    acc = num[-1] / np.maximum(den[-1][..., None], 1e-6)
    for i in range(levels - 1, -1, -1):
        h, w_ = num[i].shape[:2]
        up = cv2.resize(acc, (w_, h), interpolation=cv2.INTER_LINEAR)
        a = np.clip(den[i] * 4.0, 0, 1)[..., None]
        acc = (num[i] / np.maximum(den[i][..., None], 1e-6)) * a + up * (1 - a)
    return acc


def build_albedo():
    out = cpath('albedo_8k.png')
    if os.path.exists(out):
        return
    mask = cv2.imread(cpath('land_mask_8k.png'), cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
    bm = cv2.imread(cpath('earth-blue-marble.jpg'))[..., ::-1].astype(np.float32) / 255.0
    bm = look.srgb_to_linear(bm)
    bmu = cv2.resize(bm, (MW, MH), interpolation=cv2.INTER_CUBIC)
    bmu = np.clip(bmu, 0, 1)
    # land colour: trust BM only well inside the crisp mask, fill the coast from the interior
    conf = cv2.erode((mask > 0.99).astype(np.uint8), np.ones((7, 7), np.uint8)).astype(np.float32)
    land = push_pull(bmu, conf)
    a = cv2.GaussianBlur(conf, (0, 0), 1.5)[..., None]
    land = bmu * a + land * (1 - a)
    # ocean: deep navy with a faint memory of the shelves; keep sea ice from BM
    Y = bmu @ np.array([0.2126, 0.7152, 0.0722], np.float32)
    deep = np.array([0.0055, 0.0125, 0.030], np.float32)
    shelf = np.array([0.012, 0.030, 0.045], np.float32)
    s = np.clip((Y - 0.035) / 0.08, 0, 1)[..., None] * 0.6
    ocean = deep * (1 - s) + shelf * s
    ice = np.clip((Y - 0.25) / 0.25, 0, 1)[..., None]
    ocean = ocean * (1 - ice) + bmu * ice
    m = mask[..., None]
    alb = land * m + ocean * (1 - m)
    srgb = look.linear_to_srgb(alb)
    cv2.imwrite(out, np.round(srgb[..., ::-1] * 255).astype(np.uint8))
    print('albedo done')


# ---------------------------------------------------------------- lights ---

def lights_weight():
    """Linear night-light radiance proxy on a 4096x2048 equirect grid."""
    n = cv2.imread(cpath('earth-night.jpg')).astype(np.float32)[..., ::-1]
    r, b = n[..., 0], n[..., 2]
    L = np.clip(r - 0.30 * b - 3.0, 0, None) / 255.0
    L = L ** 2.0
    old = cv2.imread(os.path.join(ASSETS, 'textures', 'earth_lights_2048.png')).astype(np.float32)[..., ::-1]
    # DMSP: warm white lights over a blue-grey base
    lo = np.clip(np.minimum(old[..., 0], old[..., 1]) - 0.55 * old[..., 2] - 6.0, 0, None) / 255.0
    lo = cv2.resize(lo ** 2.0, (4096, 2048), interpolation=cv2.INTER_LINEAR)
    Wt = L / max(np.percentile(L[L > 0], 99.5), 1e-6) + 0.35 * lo / max(np.percentile(lo[lo > 0], 99.5), 1e-6)
    mask = cv2.imread(cpath('land_mask_8k.png'), cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
    m4 = cv2.resize(mask, (4096, 2048), interpolation=cv2.INTER_AREA)
    m4 = cv2.dilate(m4, np.ones((3, 3), np.uint8))
    return (Wt * m4).astype(np.float32)


def lonlat_to_vec(lon, lat):
    lon = np.radians(lon)
    lat = np.radians(lat)
    return np.stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)], -1).astype(np.float32)


def build_lights():
    out = cpath('lights_points.npz')
    if os.path.exists(out):
        return
    rng = np.random.default_rng(7)
    Wt = lights_weight()
    np.save(cpath('lights_w4k.npy'), Wt.astype(np.float16))
    H, W = Wt.shape
    lat_c = 90.0 - (np.arange(H) + 0.5) / H * 180.0
    area = np.cos(np.radians(lat_c))[:, None]
    p = (Wt * area).ravel().astype(np.float64)
    total = p.sum()
    N = 1_400_000
    cdf = np.cumsum(p)
    cdf /= cdf[-1]
    idx = np.searchsorted(cdf, rng.random(N))
    yy, xx = np.divmod(idx, W)
    lon = (xx + rng.random(N)) / W * 360.0 - 180.0
    lat = 90.0 - (yy + rng.random(N)) / H * 180.0
    e = rng.lognormal(0, 0.7, N).astype(np.float32)
    e /= e.mean()
    tint = rng.random(N).astype(np.float32)          # 0 = sodium, 1 = pale white
    # city cores from populated places
    d = json.load(open(os.path.join(ASSETS, 'data', 'ne_10m_populated_places_simple.geojson')))
    clon, clat, cpop = [], [], []
    for ft in d['features']:
        pr = ft['properties']
        pop = max(pr.get('pop_max') or 0, pr.get('pop_min') or 0)
        if pop >= 15000:
            clon.append(pr['longitude'])
            clat.append(pr['latitude'])
            cpop.append(pop)
    clon, clat, cpop = map(np.asarray, (clon, clat, cpop))
    cl_lon, cl_lat, cl_e, cl_t = [], [], [], []
    for lo_, la_, pp in zip(clon, clat, cpop):
        n = int(8 + np.sqrt(pp) / 2.5)
        sig_km = 2.0 + 0.012 * np.sqrt(pp)
        # organic sprawl: a soft core, arms along a few radial roads, scattered suburbs
        r = np.abs(rng.normal(0, sig_km, n)) * (0.3 + 1.2 * rng.random(n))
        arms = rng.integers(3, 7)
        base = rng.random() * 2 * np.pi
        onarm = rng.random(n) < 0.25
        ang = np.where(onarm,
                       base + (rng.integers(0, arms, n) * 2 * np.pi / arms) + rng.normal(0, 0.25, n) + 0.8 * r / (3 * sig_km),
                       rng.random(n) * 2 * np.pi)
        r = np.where(onarm, r * 1.8, r)
        dx = r * np.cos(ang) / (111.2 * max(np.cos(np.radians(la_)), 0.15))
        dy = r * np.sin(ang) / 111.2
        cl_lon.append(lo_ + dx)
        cl_lat.append(la_ + dy)
        ee = 0.35 + np.exp(-r / (sig_km * 0.8))
        ee = ee * rng.lognormal(0, 0.6, n)
        ee = ee / ee.sum() * (pp ** 0.80)
        cl_e.append(ee)
        cl_t.append(rng.random(n))
    cl_lon = np.concatenate(cl_lon)
    cl_lat = np.concatenate(cl_lat)
    cl_e = np.concatenate(cl_e)
    cl_t = np.concatenate(cl_t)
    # balance: city cores carry ~30% of the texture energy
    cl_e = cl_e / cl_e.sum() * (0.22 * N)
    lon = np.concatenate([lon, cl_lon])
    lat = np.concatenate([lat, cl_lat])
    e = np.concatenate([e, cl_e.astype(np.float32)])
    tint = np.concatenate([tint, cl_t.astype(np.float32)])
    P = lonlat_to_vec(lon, lat)
    np.savez_compressed(out, P=P, e=e.astype(np.float32), tint=tint.astype(np.float32))
    print('lights points', len(e), 'city pts', len(cl_e), 'weight total', total)


# ------------------------------------------------------------ small maps ---

def build_maps():
    out = cpath('maps.npz')
    if os.path.exists(out):
        return
    c = cv2.imread(os.path.join(ASSETS, 'textures', 'earth_clouds_1024.png'), cv2.IMREAD_UNCHANGED)
    cov = c[..., 3].astype(np.float32) / 255.0
    # the map's poles are smeared; fade them out
    lat = 90.0 - (np.arange(cov.shape[0]) + 0.5) / cov.shape[0] * 180.0
    cov *= np.clip((78.0 - np.abs(lat)) / 10.0, 0, 1)[:, None]
    nm = cv2.imread(os.path.join(ASSETS, 'textures', 'earth_normal_2048.jpg')).astype(np.float32)[..., ::-1] / 255.0
    nm = nm * 2.0 - 1.0
    np.savez_compressed(out, clouds=cov.astype(np.float32), normal=nm.astype(np.float16))
    print('maps done')


def main():
    os.makedirs(CACHE, exist_ok=True)
    fetch_nasa()
    build_land_mask()
    build_albedo()
    build_lights()
    build_maps()


if __name__ == '__main__':
    main()
