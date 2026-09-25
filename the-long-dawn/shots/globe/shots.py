"""Shot definitions for GLOBE: THE WORLD ANSWERS (1752-1935) and DAWN (2232-2495).

render(shot, frame, scale) -> display-referred sRGB float image (H, W, 3).
All timing in global frames (24 fps, 20 frames per beat).
"""
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import globe as G  # noqa: E402
from globe import Atmosphere, Camera, World, normalize, vec_ll  # noqa: E402
from web import Web, ll2v  # noqa: E402

import look  # noqa: E402  (path set by globe)

_WORLD = None
_WEB = None
_ATMO = {}


def world():
    global _WORLD
    if _WORLD is None:
        _WORLD = World()
    return _WORLD


def web():
    global _WEB
    if _WEB is None:
        _WEB = Web(seed=5)
    return _WEB


def atmo(key, **kw):
    if key not in _ATMO:
        _ATMO[key] = Atmosphere(**kw)
    return _ATMO[key]


# ------------------------------------------------------------- easing ---

def smooth(x):
    x = min(1.0, max(0.0, x))
    return x * x * (3 - 2 * x)


def smoother(x):
    x = min(1.0, max(0.0, x))
    return x * x * x * (x * (x * 6 - 15) + 10)


def lerp(a, b, t):
    return a + (b - a) * t


def ramp(t, t0, t1):
    return smooth((t - t0) / (t1 - t0))


def keyed(t, keys):
    """Piecewise interpolation through [(frame, value), ...] with smooth (C1) segments."""
    if t <= keys[0][0]:
        return keys[0][1]
    for (t0, v0), (t1, v1) in zip(keys[:-1], keys[1:]):
        if t <= t1:
            return lerp(v0, v1, smooth((t - t0) / (t1 - t0)))
    return keys[-1][1]


def spline(t, keys):
    """Catmull-Rom through keyframes (frame, value) with eased time; smoother than keyed()."""
    ts = [k[0] for k in keys]
    vs = [k[1] for k in keys]
    if t <= ts[0]:
        return vs[0]
    if t >= ts[-1]:
        return vs[-1]
    i = max(j for j in range(len(ts) - 1) if ts[j] <= t)
    u = (t - ts[i]) / (ts[i + 1] - ts[i])
    p0 = vs[max(i - 1, 0)]
    p1 = vs[i]
    p2 = vs[i + 1]
    p3 = vs[min(i + 2, len(vs) - 1)]
    u2, u3 = u * u, u * u * u
    return 0.5 * ((2 * p1) + (-p0 + p2) * u + (2 * p0 - 5 * p1 + 4 * p2 - p3) * u2 + (-p0 + 3 * p1 - 3 * p2 + p3) * u3)


def cam_rig(lat, lon, alt_km, heading, limb_y, hfov, W, H, roll=0.0):
    """Orbit camera looking along `heading`, pitched so the limb straight ahead sits at
    screen height limb_y (fraction of H from the top)."""
    d = 1.0 + alt_km / G.R_KM
    dip = math.degrees(math.acos(1.0 / d))
    f = (W / 2.0) / math.tan(math.radians(hfov) / 2.0)
    pitch = dip + math.degrees(math.atan((H / 2.0 - limb_y * H) / f))
    return Camera.orbit(lat, lon, alt_km, heading, pitch, hfov, W, H, roll)


# ================================================== THE WORLD ANSWERS ===

class Answers:
    name = 'answers'
    first, last = 1752, 1935
    moon = normalize(vec_ll(28, 42))
    sun = vec_ll(-8, -105)                 # far side: the whole view is night

    def camera(self, t, W, H):
        # slow, heavy pull-back and rise; drift west to take in the web's reach
        s = smoother((t - 1740.0) / (1950.0 - 1740.0))
        lat = lerp(11.0, 2.0, s)
        lon = lerp(86.0, 78.0, s)
        alt = math.exp(lerp(math.log(2300.0), math.log(8200.0), s))
        heading = lerp(3.0, -6.0, s)
        limb = lerp(0.16, 0.18, s)
        fov = lerp(52.0, 56.0, s)
        return cam_rig(lat, lon, alt, heading, limb, fov, W, H)

    def render_hdr(self, t, scale=1.0):
        W, H = int(round(1920 * scale)), int(round(804 * scale))
        wd = world()
        at = atmo('answers', X=2.0, mie_scale=3.0, airglow=2.5, rim=3.0, airglow_h_km=135,
                  airglow_warm=0.08, airglow_w_km=1.6)
        cam = self.camera(t, W, H)
        Emoon = np.array([0.337, 0.456, 0.69]) * 0.8
        Esun = np.array([1.0, 0.97, 0.92]) * 20.0
        img, cov, tv = G.render_planet(wd, cam, at, self.sun, Esun, self.sun, 0.0, self.moon, Emoon)
        img += G.render_lights(wd, cam, at, self.sun, gain=0.5e-7)
        img += G.render_stars(wd, cam, at, gain=0.9)
        web().draw(img, cam, t, gain=1.0, scale=scale)
        return img, cam

    def render(self, t, scale=1.0):
        img, cam = self.render_hdr(t, scale)
        return G.finish_frame(img, exposure=1.4, bloom_strength=0.085, bloom_threshold=0.8,
                              vignette_amount=0.22)


SHOTS = {'answers': Answers()}
