"""The festival hill (INTRO + CODA): ridges, sky presets, distant beacon placement."""
import math

import numpy as np

from core import C, rgb, fbm1_np
from land import Ridge, make_profile, pack_ridges
from sky import Sky, dir_from_az_el

R_EARTH = 6371000.0

# (z, base, amp, scale, ridged, trees)
LAYERS = [
    (620.0, -92.0, 22.0, 260.0, 0.0, (0.05, 9.0, 16.0, 0.45, 'broad', 180.0)),
    (1250.0, -112.0, 34.0, 420.0, 0.0, (0.035, 10.0, 18.0, 0.4, 'broad', 300.0)),
    (2300.0, -118.0, 55.0, 650.0, 0.15, (0.02, 12.0, 20.0, 0.30, 'conifer', 500.0)),
    (3900.0, -96.0, 90.0, 950.0, 0.25, (0.012, 14.0, 22.0, 0.28, 'conifer', 800.0)),
    (6600.0, -60.0, 140.0, 1500.0, 0.35, None),
    (11000.0, 30.0, 260.0, 2300.0, 0.45, None),
    (18500.0, 330.0, 480.0, 3300.0, 0.6, None),
    (30000.0, 1150.0, 850.0, 5200.0, 0.75, None),
    (52000.0, 2500.0, 1400.0, 8200.0, 0.85, None),
]


def build_ridges(seed=3):
    ridges = []
    for i, (z, base, amp, scale, ridged, trees) in enumerate(LAYERS):
        x0, x1 = -1.0 * z - 400, 1.0 * z + 400
        n = int(min(16000, max(3000, (x1 - x0) / (z / 3500.0))))
        drop = z * z / (2 * R_EARTH)
        h = make_profile(x0, x1, n, base - drop, amp, scale, seed * 31 + i * 7, octaves=7,
                         ridged=ridged, trees=trees)
        near = 1.0 - i / (len(LAYERS) - 1)
        albedo = np.array([0.018, 0.020, 0.036]) * (0.7 + 0.6 * near)
        ridges.append(Ridge(z, x0, x1, h, albedo, fog_mul=1.0, rim=0.10 + 0.05 * near,
                            mist=1.0, tex=0.35 if i < 4 else 0.15, name=f'L{i}'))
    return ridges


def crest_ridge(seed=5):
    """The foreground hilltop the figures stand on (depth 14 m)."""
    z = 14.0
    x0, x1 = -24.0, 24.0
    n = 9600
    X = np.linspace(x0, x1, n)
    h = 0.30 - 0.0032 * (X - 3.2) ** 2 - 0.0009 * np.maximum(0, -(X - 3.2)) ** 2.4 * 0.0
    h += 0.05 * fbm1_np(X / 1.3 + 4.0, 5, 2.0, 0.5, seed)
    h += 0.015 * fbm1_np(X / 0.21 + 1.0, 3, 2.0, 0.5, seed + 1)
    albedo = np.array([0.010, 0.010, 0.014])
    return Ridge(z, x0, x1, h, albedo, fog_mul=0.0, rim=0.05, mist=0.0, tex=0.0, name='crest')


FOG = np.array([1.0 / 26000.0, 1.0 / 2600.0, -175.0, 55.0, 0.55, math.radians(2.0), 0.9,
                25.0, 1.0 / 40.0], np.float64)


def sky_intro():
    return Sky(
        sun=(-16.0, -6.0),
        zenith=C('night_zenith', 1.6),
        horizon=C('night_horizon', 1.25),
        amber=C('dusk_amber', 1.35),
        rose=C('dusk_rose', 0.55),
        violet=C('dusk_violet', 0.42),
        base_fall=0.20, amber_fall=0.032, rose_fall=0.085, violet_fall=0.28, az_pow=2.2,
        moon=(-27.0, 10.5), moon_radius_deg=0.75, moon_gain=2.2, moon_aureole=0.010,
        earthshine=0.045, city_gain=0.9,
        mw=0.0, star_gain=0.9, star_thresh=10.0, n_stars=14000,
        ring=dict(lat=57.0, radius=3.4, view_az=236.0, shadow_theta=-40.0, shadow_width=14.0,
                  shadow_floor=0.05, nodes=28, node_gain=2.0, color=np.array([1.0, 0.92, 0.80])),
        seed=1)


def sky_coda():
    return Sky(
        sun=(-10.0, -13.0),
        zenith=C('night_zenith', 1.05),
        horizon=C('night_horizon', 0.85),
        amber=C('dusk_amber', 0.35),
        rose=C('dusk_rose', 0.22),
        violet=C('dusk_violet', 0.30),
        base_fall=0.18, amber_fall=0.03, rose_fall=0.07, violet_fall=0.25, az_pow=2.0,
        moon=(-25.0, 8.0), moon_radius_deg=0.75, moon_gain=2.4, moon_aureole=0.012,
        earthshine=0.06, city_gain=1.4,
        mw=0.9, mw_pole=dir_from_az_el(-150.0, 25.0), star_gain=1.3, star_thresh=5.0, n_stars=16000,
        ring=dict(lat=57.0, radius=3.4, view_az=236.0, shadow_theta=-40.0, shadow_width=14.0,
                  shadow_floor=0.04, nodes=28, node_gain=2.2, color=np.array([1.0, 0.92, 0.80])),
        seed=1)
