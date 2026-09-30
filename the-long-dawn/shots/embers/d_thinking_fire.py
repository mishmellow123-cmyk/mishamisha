"""Explicit cut-D placement of A's actual thinking-fire emitter and sparks.

The point flow, filaments, rim palette and seeded motes are scene_b's originals.
Only the screen placement, size and continuing D clock are new. A scoped
schedule is restored even if drawing fails; importing this module changes none
of the accepted A/C renderer globals.
"""
from contextlib import contextmanager
from functools import lru_cache
import math
import threading
from types import SimpleNamespace

import numpy as np

import a3
import scene_b as B
from core import Camera, Frame

_LOCK = threading.RLock()
D_ORIGIN, A_ORIGIN = 1680., 1200.


class _Schedule(a3.A3Sched):
    end = float('inf')

    def fire_centre(self, t): return np.zeros(3)
    def fire_scale(self, t): return 1.
    def power(self, t): return 1.
    def redness(self, t): return 0.
    def beat_pulse(self, t): return 0.


@contextmanager
def source_schedule():
    with _LOCK:
        old = B.SCHED, B.IGN, B.BEATS
        B.SCHED, B.IGN, B.BEATS = _Schedule(), a3.T_IGN, []
        try:
            yield
        finally:
            B.SCHED, B.IGN, B.BEATS = old


@lru_cache(maxsize=1)
def _emitters():
    with source_schedule():
        return B.MindFire(), B.FireSparks()


def source_clock(frame):
    """Continuous absolute clock, including the D2079/D2080 shot boundary."""
    return A_ORIGIN + float(frame) - D_ORIGIN


class _ScreenCamera(Camera):
    def __init__(self, root, tip):
        root, tip = np.asarray(root, float), np.asarray(tip, float)
        delta = tip - root
        height = float(np.linalg.norm(delta))
        focal = height * 40. / 5.2
        super().__init__((0., 0., 40.), (0., 0., 0.),
                         hfov=math.degrees(2. * math.atan(960. / focal)),
                         roll=math.atan2(delta[0], -delta[1]))
        # Source orb origin lies one nominal radius above the flame's root.
        self.principal = root + delta * .25

    def params(self, width, height):
        p = super().params(width, height)
        p[13:15] = self.principal * (width / 1920.)
        return p


class _EnergyFrame:
    def __init__(self, frame, gain):
        self.frame, self.gain = frame, gain

    def splat(self, p0, p1, radius, energy, color, cam0, cam1, **kwargs):
        self.frame.splat(p0, p1, radius, np.asarray(energy) * self.gain,
                         color, cam0, cam1, **kwargs)


def draw(hdr, root_px, tip_px, frame, bright=1., scale=1., vis=None):
    """Add live A fire to an HDR image; positions use native 1920×804 pixels.

    Adjacent shots share this function and absolute frame clock. Geometry and
    finish remain the caller's responsibility. The apparent height is nominal;
    the original A emitter breathes and its sparks extend beyond the tongue.
    """
    height = float(np.linalg.norm(np.asarray(tip_px) - np.asarray(root_px)))
    if height < 1.5 or bright <= 0:
        return hdr
    if hdr.shape != (round(804 * scale), round(1920 * scale), 3):
        raise ValueError('Thinking fire requires the cut-D native aspect ratio')
    cam = _ScreenCamera(root_px, tip_px)
    fr = Frame(scale)
    fr.set(focus=40., aperture=0., bokeh_pow=.25, bokeh_cap=1.6,
           fog_start=260., fog_len=320., near=.3)
    proxy = _EnergyFrame(fr, float(bright) * (height / 220.) ** 2)
    time = source_clock(frame)
    ctx = SimpleNamespace(t=time, t0=time - .25, t1=time + .25,
                          cam=cam, cam0=cam, cam1=cam, fr=proxy, scale=scale)
    with source_schedule():
        fire, sparks = _emitters()
        fire.emit(ctx)
        sparks.emit(ctx)
    layer = fr.resolve()
    if vis is not None:
        layer *= np.asarray(vis)[..., None]
    hdr += layer
    return hdr
