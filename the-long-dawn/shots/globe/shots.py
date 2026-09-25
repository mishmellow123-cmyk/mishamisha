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
from web import Hearths, Web, ll2v  # noqa: E402

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



# ================================================================ DAWN ===

_HEARTHS = None


def hearths():
    global _HEARTHS
    if _HEARTHS is None:
        _HEARTHS = Hearths(web())
    return _HEARTHS


def rig_frame(lat, lon, heading):
    """Local horizontal heading direction and up vector at (lat, lon)."""
    e, n, u = G.enu(lat, lon)
    hd = math.radians(heading)
    return math.sin(hd) * e + math.cos(hd) * n, u


def seg_frac(h, r):
    """Visible fraction of a disk of radius r whose centre is h above a straight horizon."""
    if h >= r:
        return 1.0
    if h <= -r:
        return 0.0
    return (r * r * math.acos(-h / r) + h * math.sqrt(r * r - h * h)) / (math.pi * r * r)


class Dawn:
    name = 'dawn'
    first, last = 2232, 2495
    moon = normalize(vec_ll(-10, -20))
    SUN_R = 0.2665

    def params(self, t):
        s = smoother((t - 2232.0) / (2495.0 - 2232.0))
        p = dict(
            lat=lerp(-2.0, -4.5, s), lon=lerp(12.0, 9.0, s),
            alt=math.exp(lerp(math.log(2400.0), math.log(4300.0), s)),
            heading=lerp(80.0, 77.0, s), limb=lerp(0.43, 0.40, s), fov=lerp(58.0, 61.0, s),
        )
        # sun elevation above the limb (deg): crest at 2240 exactly
        p['sun_el'] = spline(t, [(2200, -1.6), (2232, -0.75), (2240, -self.SUN_R + 0.02), (2250, -0.02),
                                 (2266, 0.32), (2300, 0.75), (2400, 1.7), (2495, 2.4)])
        # the lighting sun leads the disk so the terminator can visibly race toward us
        p['lead'] = spline(t, [(2200, 0.0), (2240, 0.0), (2252, 1.5), (2270, 6.0), (2300, 13.0),
                               (2350, 21.0), (2420, 28.0), (2495, 34.0)])
        p['sun_az'] = -1.5
        return p

    def camera(self, t, W, H):
        p = self.params(t)
        return cam_rig(p['lat'], p['lon'], p['alt'], p['heading'], p['limb'], p['fov'], W, H), p

    def suns(self, p):
        d = 1.0 + p['alt'] / G.R_KM
        dip = math.degrees(math.acos(1.0 / d))
        hz, up = rig_frame(p['lat'], p['lon'], p['heading'])
        az = math.radians(p['sun_az'])
        side = np.cross(up, hz)
        hz2 = math.cos(az) * hz + math.sin(az) * side

        def at(el):
            a = math.radians(dip - el)
            return normalize(math.cos(a) * hz2 - math.sin(a) * up)
        return at(p['sun_el']), at(p['sun_el'] + p['lead'])

    def calm(self, t, H):
        # text windows 2270-2345 and 2352-2440: keep y 560..700 (of 804) calm
        k = max(ramp(t, 2262, 2272) * (1 - ramp(t, 2343, 2350)), ramp(t, 2350, 2356) * (1 - ramp(t, 2438, 2446)))
        if k <= 0:
            return None
        y0, y1 = 545.0 / 804 * H, 715.0 / 804 * H
        soft = 30.0 / 804 * H

        def f(y):
            a = np.clip((y - (y0 - soft)) / soft, 0, 1) * np.clip(((y1 + soft) - y) / soft, 0, 1)
            return 1.0 - 0.5 * k * a
        return f

    def render_hdr(self, t, scale=1.0):
        W, H = int(round(1920 * scale)), int(round(804 * scale))
        wd = world()
        at = atmo('dawn', X=2.6, mie_scale=4.0, g=0.82, airglow=1.5, rim=2.5, airglow_h_km=135,
                  airglow_warm=0.08, airglow_w_km=1.6)
        cam, p = self.camera(t, W, H)
        Sd, Sl = self.suns(p)
        Emoon = np.array([0.337, 0.456, 0.69]) * 0.6
        Esun = np.array([1.0, 0.91, 0.76]) * 17.0
        cp = World.default_cp(at.X)
        cp[15] = math.radians(-16.0)        # this morning's weather: the African cloud mass sits
        cp[16] = math.radians(-2.0)         # where the terminator sweeps
        cp[17] = 0.55                       # broken fair-weather cloud everywhere
        img, cov, tv = G.render_planet(wd, cam, at, Sl, Esun, Sd, 40.0, self.moon, Emoon, cp=cp)
        img += G.render_lights(wd, cam, at, Sl, gain=0.5e-7, cp=cp)
        star_k = 1.0 - ramp(t, 2236, 2262) * 0.85
        img += G.render_stars(wd, cam, at, gain=0.9 * star_k)
        calm = self.calm(t, H)
        web().draw(img, cam, 2600.0, t_anim=t, gain=0.9, scale=scale, calm=calm, sun=Sl, sparks=False,
                   ground_only=True)
        hearths().draw(img, cam, t, sun=Sl, gain=1.0, calm=calm, scale=scale)
        # the sun through the lens
        sp, z = cam.project(cam.pos[None, :] + Sd[None, :] * 50.0)
        sx, sy = sp[0]
        frac = seg_frac(p['sun_el'], self.SUN_R)
        burst = math.exp(-max(t - 2240.0, 0.0) / 5.0) if t >= 2240 else 0.0
        pre = ramp(t, 2226.0, 2240.0)
        core = 7.0 * frac ** 0.6 + 1.0 * pre * (1.0 - frac)
        settle = 1.0 - 0.45 * ramp(t, 2262.0, 2300.0)
        spikes = 3.6 * frac ** 0.5 * (1.0 + 1.0 * burst) * settle
        slen = 1.0 - 0.4 * ramp(t, 2255.0, 2290.0)
        mask = (470.0, 560.0, 0.85 * max(ramp(t, 2255, 2268), 0.0))
        glare = G.sun_glare(W, H, sx, sy, core, spikes, flash=5.0 * burst * min(1.0, frac * 6.0 + 0.3),
                            spike_len=slen, spike_mask_y=mask)
        return img, glare, cam, p, (sx, sy, frac, burst)

    def exposure(self, t):
        return spline(t, [(2200, 1.3), (2238, 1.3), (2244, 1.15), (2270, 0.9), (2495, 0.85)])

    def render(self, t, scale=1.0):
        img, glare, cam, p, sun = self.render_hdr(t, scale)
        sb = sun[3]
        return G.finish_frame(img, sun_layer=glare, exposure=self.exposure(t), bloom_strength=0.07,
                              bloom_threshold=0.9, streak_strength=0.035 + 0.06 * sb, streak_threshold=3.0,
                              streak_length=0.5, vignette_amount=0.22)


SHOTS = {'answers': Answers(), 'dawn': Dawn()}
