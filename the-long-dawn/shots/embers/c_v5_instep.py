"""Default-off C16/C17 study: working forges share a pace; the Ring keeps giving.

Accepted frames delegate to the adopted lead24 driver. No edit/caption changes.
The scoped substitutions below are needed by the inherited global-based renderer;
all are restored even if a frame fails. Farm writes remain cand_render's concern.
"""
import argparse
from contextlib import contextmanager
import json
import math
import threading

import numpy as np

import c_v5 as base
import c_v5_cold_leadin as lead

VARIANTS = ('accepted', 'in-step')
START, SETTLED = 3848, 3872
WORK_LEVEL = .85
PULSE_AMPLITUDE = .10  # inherited tower surge is 1 + .9 * beat_pulse: +9% peak (owner, 30 Sep: raised from .04, +3.6%, judged too faint to read as a shared pulse; unseen in motion until the farm frames)
RING_LEVEL = .80
SMOKE_LEVEL = .12      # fraction of the accepted warm chimney emission
DROP_PERIOD = 104.     # accepted flight cycle is 52 frames
DROP_GAIN = .55
BARMAP = base.ROOT / 'music/v3/barmap_C5P2.json'
_LOCK = threading.Lock()
_RACE_LEVEL = base.forge_level
_RING_WARMTH = base.ring_warmth


def beat_frames(path=BARMAP):
    """Read the score's metric grid, not its irregular picture-sync events.

    Same frame/beat conversion as music/src/timeline_v3.py:38-40,70-90;
    sound_c5_table.py:103-105 reads this exact C5P2 map.
    """
    data = json.loads(path.read_text())
    if data['cut'] != 'C5P2' or data['fps'] != 24 or data['frames'] != 5920:
        raise ValueError('Expected the 24 fps, 5920-frame C5P2 score map')
    if not math.isfinite(data['bpm']) or data['bpm'] <= 0:
        raise ValueError('Invalid score tempo')
    step = data['fps'] * 60. / data['bpm']
    if not math.isfinite(step) or step <= 0:
        raise ValueError('Invalid score beat spacing')
    return tuple(np.arange(0., data['frames'], step)), step


class Schedule(base.Schedule):
    def __init__(self):
        self.beats, self.beat_step = beat_frames()

    def power(self, t):
        return 1. if t >= START else super().power(t)

    def beat_pulse(self, t):
        # Raised cosine: a full beat of breath, peak exactly on the score beat.
        # Ease it in AFTER convergence, avoiding a jump at the settling boundary.
        if t <= SETTLED:
            return 0.
        phase = (t - self.beats[0]) / self.beat_step
        return PULSE_AMPLITUDE * .5 * (1. + math.cos(2. * math.pi * phase)) * base.ease(SETTLED, SETTLED + self.beat_step, t)

    def redness(self, t):
        return .38 * ring_warmth(t)


def forge_level(i, t):
    if t < START:
        return _RACE_LEVEL(i, t)
    u = base.ease(START, SETTLED, t)
    return (1. - u) * _RACE_LEVEL(i, START - 1.) + u * WORK_LEVEL


def ring_warmth(t):
    if t < START:
        return _RING_WARMTH(t)
    return 1. - (1. - RING_LEVEL) * base.ease(4000, 4024, t)


def flame_clock(t):
    # Integral of a smooth 1 -> .25 speed change: continuous position and speed.
    x = max(0., min(24., t - START))
    u = x / 24.
    return START + x - .75 * 24. * (u ** 3 - .5 * u ** 4) + .25 * max(0., t - SETTLED)


class WorkingSmokeFrame:
    def __init__(self, frame, t, gain=None):
        self.frame, self.t, self.gain = frame, t, gain

    def __getattr__(self, name):
        return getattr(self.frame, name)

    def splat(self, P0, P1, radius, energy, colour, *args, **kwargs):
        gain = (1. - (1. - SMOKE_LEVEL) * base.ease(START, SETTLED, self.t)
                if self.gain is None else self.gain)
        # No shutdown blue-grey conversion or cooling-to-zero.
        return self.frame.splat(P0, P1, radius, energy * gain, colour, *args, **kwargs)


@contextmanager
def working_world(schedule, towers):
    if not _LOCK.acquire(blocking=False):
        raise RuntimeError('In-step renderer is not reentrant')
    old = base.SCHED, base.forge_level, base.ring_warmth, towers.dly
    try:
        base.SCHED, base.forge_level, base.ring_warmth = schedule, forge_level, ring_warmth
        # B.Towers.emit reads t - dly[i]. Geometry/heat-wave BEATS stay empty.
        towers.dly = np.zeros_like(towers.dly)
        with base.world():
            yield
    finally:
        base.SCHED, base.forge_level, base.ring_warmth, towers.dly = old
        _LOCK.release()


def window_targets(towers, schedule, camera, t, ceiling=None):
    """One real WIN centroid per tower; crown throats are valid WIN fallbacks."""
    points, keys = [], []
    ceiling = base.RING_C[1] - 3. if ceiling is None else ceiling
    for i in range(towers.k_all):
        g = towers.G[i][3]
        choices = []
        for key in np.unique(g['key']):
            mask = g['key'] == key
            p = g['p'][mask].mean(axis=0, keepdims=True)
            n = g['n'][mask].mean(axis=0, keepdims=True)
            world = towers.world(i, p, towers.height(i, t))
            world = schedule.tower_post(towers, i, world, t)[0]
            normal = towers.nworld(i, n)[0]
            target = world + normal * .12
            if target[1] <= base.B.GROUND + 1. or target[1] >= ceiling:
                continue
            sight = camera.pos - world
            facing = float(np.dot(normal, sight) / max(np.linalg.norm(sight), 1e-9))
            incoming = base.RING_C - world
            ringward = float(np.dot(normal, incoming) / max(np.linalg.norm(incoming), 1e-9))
            # Receive from the Ring-facing wall, so the flight does not cross
            # the tower to reach its camera-facing outer wall. Throat fallback.
            score = (ringward > .05, abs(normal[1]) < .5, facing > .05, world[1], ringward)
            choices.append((score, target, int(key)))
        if not choices:
            raise ValueError(f'No above-ground gold-receiving window for tower {i}')
        _, point, key = max(choices, key=lambda row: row[0])
        points.append(point)
        keys.append(key)
    return np.array(points), tuple(keys)


class Scene(lead.Scene):
    def __init__(self, variant='accepted'):
        if variant not in VARIANTS:
            raise ValueError(f'Unknown in-step study: {variant}')
        # lead24 is the accepted C3816..3999 material, including its first shutter.
        super().__init__('lead24')
        self.variant = variant
        self.schedule = Schedule() if variant == 'in-step' else None

    def ring_frame(self, t):
        return super().ring_frame(min(t, 4000.) if self.variant == 'in-step' and t >= START else t)

    def _shared_drops(self):
        if hasattr(self, 'shared_targets'):
            return
        n = self.towers.k_all
        # Exactly two drops per tower per cycle, evenly staggered across towers.
        self.shared_tower = np.tile(np.arange(n), 2)
        self.shared_phase = (np.arange(2 * n) / (2 * n))
        self.shared_theta = 2. * np.pi * (np.arange(2 * n) / (2 * n))
        if base.c3.OR.enabled():
            self.shared_theta = base.c3.OR.arc_theta(4000., self.shared_theta / (2. * np.pi))
        rot, centre, size = self.ring_frame(4000.)
        rim, _ = base.RS.local_points(self.shared_theta, np.zeros(2 * n))
        ceiling = float((centre + size * rim @ rot.T)[:, 1].min()) - .25
        self.shared_targets, self.shared_window_keys = window_targets(
            self.towers, self.schedule, self.camera(4000.), 4000., ceiling=ceiling)

    def shared_positions(self, t):
        self._shared_drops()
        u = (self.shared_phase + (t - 4000.) / DROP_PERIOD) % 1.
        rot, centre, size = self.ring_frame(t)
        p, _ = base.RS.local_points(self.shared_theta, np.zeros(len(u)))
        start = centre + size * p @ rot.T
        end = self.shared_targets[self.shared_tower]
        if np.any(end[:, 1] >= start[:, 1]):
            raise ValueError('Gold must fall down to every receiving window')
        q = start + (end - start) * u[:, None]
        q[:, 1] = start[:, 1] + (end[:, 1] - start[:, 1]) * u * u
        return q, u

    def drops(self, ctx):
        if self.variant != 'in-step' or ctx.t < 4000:
            return super().drops(ctx)
        mix = base.ease(4000, 4024, ctx.t)
        if mix < 1.:
            original = ctx.fr
            try:
                ctx.fr = WorkingSmokeFrame(original, ctx.t, gain=1. - mix)
                super().drops(ctx)
            finally:
                ctx.fr = original
        if mix <= 0.:
            return
        p0, u0 = self.shared_positions(ctx.t0)
        p1, u1 = self.shared_positions(ctx.t1)
        ok = (u1 >= u0) & (u1 > .015) & (u1 < .995)
        z = np.maximum((p1 - ctx.cam.pos) @ ctx.cam.R[2], 1.)
        energy = mix * DROP_GAIN * .016 * (ctx.cam.f_px(1920) / z) ** 2
        ctx.fr.splat(p0[ok], p1[ok], .065, energy[ok], [1., .9, .62], ctx.cam0, ctx.cam1, zref=0.)
        # A small steady pool at each selected opening identifies the recipient;
        # no impact flare and no revival of the facade-wide office-window grid.
        target = self.shared_targets
        z = np.maximum((target - ctx.cam.pos) @ ctx.cam.R[2], 1.)
        ctx.fr.splat(target, target, .18, mix * .004 * (ctx.cam.f_px(1920) / z) ** 2,
                     [1., .72, .30], ctx.cam0, ctx.cam1, zref=0., profile=1)

    def frame(self, f, scale=.5):
        if self.variant != 'in-step' or f < START:
            # Do not change any original rendering arithmetic before the beacon.
            if lead.FIRST <= f < 3840:
                with lead._extended_cold():
                    return base.Scene.frame(self, f, scale)
            return base.Scene.frame(self, f, scale)
        name = base.shot_at(f)
        a, b = base.SHOTS[name]
        ctx = argparse.Namespace(t=float(f), f=f, t0=max(f - .25, a),
                                 t1=min(f + .25, b - .02), scale=scale)
        ctx.cam, ctx.cam0, ctx.cam1 = self.camera(ctx.t), self.camera(ctx.t0), self.camera(ctx.t1)
        ctx.fr = base.Frame(scale)
        ctx.fr.set(focus=ctx.cam.focus, aperture=ctx.cam.aperture, bokeh_pow=.25, bokeh_cap=1.6,
                   fog_start=90., fog_len=120., near=.3)
        with working_world(self.schedule, self.towers):
            self.towers.prepare(ctx)
            ring = self.ring_layer(ctx)
            self.towers.emit(ctx, base.RING_C, np.array([1., .62, .26]), 70. * ring_warmth(f))
            original = ctx.fr
            try:
                ctx.fr = WorkingSmokeFrame(original, f)
                self.smoke.emit(ctx, base.RING_C, np.array([1., .62, .26]), 70. * ring_warmth(f))
            finally:
                ctx.fr = original
            self.drops(ctx)
            hdr = ctx.fr.resolve() + ring + self.storm(ctx)
            breath = 1. + .9 * self.schedule.beat_pulse(f)
            for i in np.argsort([(self.towers.top(j, f) - ctx.cam.pos) @ ctx.cam.R[2]
                                for j in range(self.towers.k_all)])[::-1]:
                k = forge_level(int(i), f)
                root = self.towers.top(int(i), f) - np.array([0., .4, 0.])
                tip = root + np.array([0., 4.2 * k, 0.])
                u, v, z = ctx.cam.project(np.stack([root, tip]), 1920, 804)
                vis = base.c3.occ_vis(ctx.fr, float(z[0] - .5), ctx.fr.H, ctx.fr.W)
                base.CF.draw(hdr, (u[0], v[0]), (u[1], v[1]), flame_clock(f) + 31. * i,
                             bright=k * .85 * breath, calm=.8, vis=vis, scale=scale)
        if not np.isfinite(hdr).all():
            raise ValueError(f'Non-finite in-step HDR at C{f}')
        return base.look.finish(hdr, exposure=1., bloom_strength=.09, bloom_threshold=1.1,
                                streak_strength=0., vignette_amount=.28)
