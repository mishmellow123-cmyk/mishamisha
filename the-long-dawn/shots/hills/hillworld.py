"""The festival hill (INTRO + CODA): ridges, crest, sky presets, beacon placement.

World: camera origin near (0,0,0); +z looks roughly SSW (azimuth VIEW_AZ) over a
valley; the sun set to the right (WSW). The hilltop crest where the Elder, the
Child and the cairn stand is 14 m away at the right third of the frame.
"""
import math

import numpy as np

from core import fbm1_np
from land import Ridge, make_profile
from sky import Sky, dir_from_az_el

R_EARTH = 6371000.0
VIEW_AZ = 197.0
CREST_Z = 14.0
CREST_TOP_X = 3.3          # x of the hilltop (figures stand around here)

# per layer: z, base, amp, feature scale, ridged, trees, albedo gain, fog_el(deg), mist, mist_scale, rim
LAYERS = [
    # a dark forested ridge close below us
    dict(z=480.0, base=-58.0, amp=16.0, scale=210.0, ridged=0.0,
         trees=(0.10, 9.0, 19.0, 0.32, 'conifer', 140.0), alb=1.0, fog_el=7.0, mist=0.7, ms=1 / 300., rim=0.08),
    # rounded hills with scattered broadleaf trees
    dict(z=1150.0, base=-86.0, amp=30.0, scale=380.0, ridged=0.0,
         trees=(0.03, 10.0, 17.0, 0.55, 'broad', 260.0), alb=0.85, fog_el=6.0, mist=1.0, ms=1 / 700., rim=0.10),
    # smooth long hill
    dict(z=2300.0, base=-96.0, amp=48.0, scale=820.0, ridged=0.0,
         trees=(0.012, 12.0, 20.0, 0.32, 'conifer', 420.0), alb=0.7, fog_el=4.5, mist=1.0, ms=1 / 1100., rim=0.12),
    # ridge with rocky knolls
    dict(z=4300.0, base=-80.0, amp=95.0, scale=900.0, ridged=0.35,
         trees=None, alb=0.6, fog_el=3.5, mist=0.9, ms=1 / 1800., rim=0.12),
    # rolling
    dict(z=8200.0, base=-30.0, amp=150.0, scale=2100.0, ridged=0.15,
         trees=None, alb=0.5, fog_el=2.5, mist=0.8, ms=1 / 3000., rim=0.10),
    # long gentle ridge
    dict(z=15500.0, base=120.0, amp=260.0, scale=3600.0, ridged=0.3,
         trees=None, alb=0.4, fog_el=1.8, mist=0.6, ms=1 / 5000., rim=0.08),
    # distant jagged range
    dict(z=31000.0, base=820.0, amp=780.0, scale=4200.0, ridged=0.9,
         trees=None, alb=0.3, fog_el=1.2, mist=0.3, ms=0.0, rim=0.06),
    # farthest peaks
    dict(z=56000.0, base=1650.0, amp=1300.0, scale=7000.0, ridged=0.95,
         trees=None, alb=0.25, fog_el=0.9, mist=0.2, ms=0.0, rim=0.05),
]


def build_ridges(seed=3):
    ridges = []
    for i, L in enumerate(LAYERS):
        z = L['z']
        x0, x1 = -1.05 * z - 300, 1.05 * z + 300
        n = int(min(20000, max(3000, (x1 - x0) / (z / 3600.0))))
        drop = z * z / (2 * R_EARTH)
        h = make_profile(x0, x1, n, L['base'] - drop, L['amp'], L['scale'], seed * 31 + i * 7,
                         octaves=8, ridged=L['ridged'], trees=L['trees'])
        albedo = np.array([0.020, 0.024, 0.046]) * L['alb']
        ridges.append(Ridge(z, x0, x1, h, albedo, fog_mul=1.0, rim=L['rim'], mist=L['mist'],
                            tex=0.3 if i < 3 else 0.12, name=f'L{i}',
                            fog_el=math.radians(L['fog_el']), mist_scale=L['ms'], seed=11.3 * i + 2.0))
    return ridges


def crest_profile(X, seed=5):
    """Hilltop the figures stand on, as seen from the camera (height above camera eye)."""
    X = np.asarray(X, np.float64)
    d = X - CREST_TOP_X
    top = 0.34
    left = top - 0.050 * d ** 2 - 0.006 * np.maximum(0, -d) ** 3
    right = top - 0.016 * d ** 2
    h = np.where(d < 0, left, right)
    h += 0.045 * fbm1_np(X / 1.1 + 4.0, 5, 2.0, 0.5, seed)
    h += 0.012 * fbm1_np(X / 0.17 + 1.0, 3, 2.0, 0.5, seed + 1)
    return h


def crest_ridge(seed=5):
    z = CREST_Z
    x0, x1 = -20.0, 24.0
    n = 11000
    X = np.linspace(x0, x1, n)
    h = crest_profile(X, seed)
    # a few rocks poking out of the turf
    for xr, w, hh in [(-1.4, 0.35, 0.16), (-0.9, 0.2, 0.09), (6.3, 0.45, 0.13), (8.2, 0.3, 0.08),
                      (0.6, 0.18, 0.07)]:
        r = np.abs(X - xr) / w
        bump = np.sqrt(np.clip(1 - r ** 2, 0, 1)) * hh * (1 + 0.15 * np.sin(X * 40 + xr))
        base = np.interp(xr, X, h)
        h = np.where(r < 1, np.maximum(h, base - 0.05 + bump), h)
    albedo = np.array([0.010, 0.010, 0.015])
    return Ridge(z, x0, x1, h, albedo, fog_mul=0.0, rim=0.0, mist=0.0, tex=0.0, name='crest',
                 fog_el=math.radians(6.0), mist_scale=0.0)


def crest_height(x):
    return float(crest_profile(np.array([x]))[0])


FOG = np.array([1.0 / 15000.0, 1.0 / 2200.0, -150.0, 45.0, 0.65, math.radians(2.0), 0.9,
                30.0, 1.0 / 40.0], np.float64)

RING = dict(lat=57.0, radius=2.6, view_az=VIEW_AZ, shadow_theta=25.0, shadow_width=22.0,
            shadow_floor=0.03, nodes=30, node_gain=1.6, color=np.array([1.0, 0.90, 0.78]))


def sky_intro():
    return Sky(
        sun=(27.0, -6.0),
        zenith=np.array([0.0016, 0.0030, 0.030]),
        horizon=np.array([0.022, 0.040, 0.125]),
        amber=np.array([1.25, 0.52, 0.14]),
        rose=np.array([0.26, 0.085, 0.12]),
        violet=np.array([0.050, 0.036, 0.100]),
        base_fall=0.17, amber_fall=0.022, rose_fall=0.060, violet_fall=0.22, az_pow=2.6,
        moon=(-17.0, 6.8), moon_radius_deg=0.9, moon_gain=3.4, moon_aureole=0.006,
        earthshine=0.045, city_gain=1.1, moon_phase=128.0, moon_pa=-22.0,
        mw=0.0, star_gain=1.0, star_thresh=2.2, n_stars=14000,
        ring=dict(RING), seed=1)


def sky_coda():
    return Sky(
        sun=(25.0, -13.0),
        zenith=np.array([0.0008, 0.0016, 0.019]),
        horizon=np.array([0.014, 0.026, 0.085]),
        amber=np.array([0.34, 0.13, 0.04]),
        rose=np.array([0.10, 0.035, 0.06]),
        violet=np.array([0.030, 0.024, 0.075]),
        base_fall=0.16, amber_fall=0.020, rose_fall=0.055, violet_fall=0.20, az_pow=2.4,
        moon=(-17.0, 5.6), moon_radius_deg=0.9, moon_gain=3.6, moon_aureole=0.008,
        earthshine=0.06, city_gain=1.6, moon_phase=128.0, moon_pa=-22.0,
        mw=0.9, mw_pole=dir_from_az_el(-120.0, 20.0), star_gain=1.35, star_thresh=1.4, n_stars=18000,
        ring=dict(RING, shadow_floor=0.025, node_gain=1.9), seed=1)
