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
from web import Hearths, Web, ll2v  # noqa: E402  (v1 web; kept for reference)
from fires import FireNet  # noqa: E402  (v2: fire, not fibre)

import look  # noqa: E402  (path set by globe)
import lens as L  # noqa: E402

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


_FNET = None


def fnet():
    global _FNET
    if _FNET is None:
        _FNET = FireNet(seed=int(os.environ.get('LONGDAWN_FIRE_SEED', '3')))
    return _FNET


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
        planet = img * cov[..., None]
        # v2: the cities' lights a little lower, so the fires are the protagonists
        img += G.render_lights(wd, cam, at, self.sun, gain=0.3e-7)
        img += G.render_stars(wd, cam, at, gain=0.9)
        fnet().draw(img, cam, t, gain=1.0, scale=scale, planet=planet)
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
    """DAWN v2: no web, no hearths. The answering fires burn across the night side and pale as the
    terminator reaches them (the sun takes over from the beacons). The sun is a clean glare with a
    restrained anamorphic streak."""
    name = 'dawn'
    first, last = 2232, 2495
    moon = normalize(vec_ll(-10, -20))
    SUN_R = 0.2665
    # camera framings (start, end) -- see NOTES.md; 'v1' is the delivered v1 framing
    CAMS = {
        'v1': dict(lat=(-2.0, -4.5), lon=(12.0, 9.0), alt=(2400.0, 4300.0), heading=(80.0, 77.0),
                   limb=(0.43, 0.40), fov=(58.0, 61.0), az=-1.5, clear=(6.0, 50.0)),
        'aden': dict(lat=(5.0, 3.0), lon=(14.0, 11.5), alt=(2700.0, 5200.0), heading=(72.0, 70.0),
                     limb=(0.43, 0.40), fov=(58.0, 61.0), az=-1.0, clear=(12.0, 52.0)),
        'three': dict(lat=(13.0, 11.0), lon=(10.0, 7.0), alt=(3400.0, 6200.0), heading=(84.0, 81.0),
                      limb=(0.43, 0.40), fov=(58.0, 61.0), az=-2.0, clear=(14.0, 52.0)),
    }

    def cam_keys(self):
        return self.CAMS[os.environ.get('LONGDAWN_DAWN_CAM', 'aden')]

    def params(self, t):
        s = smoother((t - 2232.0) / (2495.0 - 2232.0))
        k = self.cam_keys()
        p = dict(
            lat=lerp(*k['lat'], s), lon=lerp(*k['lon'], s),
            alt=math.exp(lerp(math.log(k['alt'][0]), math.log(k['alt'][1]), s)),
            heading=lerp(*k['heading'], s), limb=lerp(*k['limb'], s), fov=lerp(*k['fov'], s),
        )
        # sun elevation above the limb (deg): crest at 2240 exactly
        p['sun_el'] = spline(t, [(2200, -1.6), (2232, -0.72), (2239, -self.SUN_R - 0.01), (2240, -0.19),
                                 (2250, 0.0), (2266, 0.32), (2300, 0.75), (2400, 1.7), (2495, 2.4)])
        # the lighting sun leads the disk so the terminator can visibly race toward us
        p['lead'] = spline(t, [(2200, 0.0), (2240, 0.0), (2252, 1.5), (2270, 6.0), (2300, 12.0),
                               (2350, 17.5), (2420, 22.0), (2495, 26.0)])
        p['sun_az'] = k['az']
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

    @staticmethod
    def calm(t, H, nocalm=None):
        """Cut A's line "In time, the race was over." (v2 2512-2600 = src 2352-2440): keep the band
        y 545..715 (of 804) calm -- the fires there at half strength. None for the wordless cut."""
        if nocalm is None:
            nocalm = os.environ.get('LONGDAWN_NOCALM') == '1'
        if nocalm:
            return None
        k = ramp(t, 2343, 2353) * (1 - ramp(t, 2439, 2449))
        if k <= 0:
            return None
        y0, y1 = 545.0 / 804 * H, 715.0 / 804 * H
        soft = 30.0 / 804 * H

        def f(y):
            a = np.clip((y - (y0 - soft)) / soft, 0, 1) * np.clip(((y1 + soft) - y) / soft, 0, 1)
            return 1.0 - 0.5 * k * a
        return f

    def render_base(self, t, scale=1.0):
        """Everything that does not depend on the text calm: planet, city lights, stars, the sun."""
        W, H = int(round(1920 * scale)), int(round(804 * scale))
        wd = world()
        at = atmo('dawn', X=2.6, mie_scale=4.0, g=0.82, airglow=1.5, rim=2.5, airglow_h_km=135,
                  airglow_warm=0.08, airglow_w_km=1.6)
        cam, p = self.camera(t, W, H)
        Sd, Sl = self.suns(p)
        Emoon = np.array([0.337, 0.456, 0.69]) * 0.6
        Esun = np.array([1.0, 0.94, 0.84]) * 17.0
        cp = World.default_cp(at.X)
        cp[15] = math.radians(-16.0)        # this morning's weather: the African cloud mass sits
        cp[16] = math.radians(-2.0)         # where the terminator sweeps
        cp[23] = 0.012                      # cloud-top relief in the raking light
        clear = normalize(vec_ll(*self.cam_keys()['clear']))   # a clearer sky near the sunrise
        cp[18:21] = clear
        cp[21] = math.cos(math.radians(17.0))
        cp[22] = 0.65
        img, cov, tv = G.render_planet(wd, cam, at, Sl, Esun, Sd, 40.0, self.moon, Emoon, cp=cp)
        planet = img * cov[..., None]
        img += G.render_lights(wd, cam, at, Sl, gain=0.4e-7, cp=cp)
        star_k = 1.0 - ramp(t, 2236, 2262) * 0.85
        img += G.render_stars(wd, cam, at, gain=0.9 * star_k)
        # the sun through the lens: a clean disc glare, a brief round flash as it breaks the limb,
        # and a thin, restrained anamorphic line
        sp, z = cam.project(cam.pos[None, :] + Sd[None, :] * 50.0)
        sx, sy = sp[0]
        frac = seg_frac(p['sun_el'], self.SUN_R)
        burst = math.exp(-max(t - 2240.0, 0.0) / 5.0) if t >= 2240 else 0.0
        pre = ramp(t, 2226.0, 2240.0)
        core = 7.0 * frac ** 0.6 + 2.5 * pre * (1.0 - frac)
        streak = (0.4 + 0.8 * burst) * frac ** 0.5 * (1.0 - 0.35 * ramp(t, 2262.0, 2320.0))
        glare = L.sun_glare_clean(W, H, sx, sy, core, flash=5.0 * burst, streak=streak)
        return dict(img=img, planet=planet, glare=glare, cam=cam, p=p, Sl=Sl, sun=(sx, sy, frac, burst))

    def fires(self, base, t, scale, calm):
        img = base['img'].copy()
        fnet().draw(img, base['cam'], 2700.0, t_anim=t, embers=False, sun=base['Sl'], day_keep=0.0,
                    gain=1.1, scale=scale, calm=calm, planet=base['planet'], light=1.0)
        # v1's village hearths blooming into the dark are gone (LONGDAWN_HEARTHS=1 brings them back)
        if os.environ.get('LONGDAWN_HEARTHS') == '1':
            hearths().draw(img, base['cam'], t, sun=base['Sl'], gain=1.0, calm=calm, scale=scale)
        return img

    def exposure(self, t):
        return spline(t, [(2200, 1.3), (2238, 1.3), (2244, 1.15), (2270, 0.9), (2495, 0.85)])

    def finish(self, img, base, t):
        return G.finish_frame(img, sun_layer=base['glare'], exposure=self.exposure(t), bloom_strength=0.07,
                              bloom_threshold=0.9, streak_strength=0.0, vignette_amount=0.22)

    def render(self, t, scale=1.0, nocalm=None):
        base = self.render_base(t, scale)
        H = base['img'].shape[0]
        img = self.fires(base, t, scale, self.calm(t, H, nocalm))
        return self.finish(img, base, t)

    def render_both(self, t, scale=1.0):
        """(cut A with the text calm, cut B without) from one render of the planet."""
        base = self.render_base(t, scale)
        H = base['img'].shape[0]
        ca = self.calm(t, H, nocalm=False)
        a = self.finish(self.fires(base, t, scale, ca), base, t)
        b = a if ca is None else self.finish(self.fires(base, t, scale, None), base, t)
        return a, b


SHOTS = {'answers': Answers(), 'dawn': Dawn()}
