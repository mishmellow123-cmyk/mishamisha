"""Cut D's industrial forges. Explicit entrypoint; C renderers remain unchanged.

All public scene methods take absolute D frames. Timing and C offsets belong to
openring_d; the renderer owns staging. Crown forge flames have no emitter here.
The separate vision lane owns the two promise beacons.
"""
import argparse
from contextlib import contextmanager
from functools import lru_cache
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import sys
import threading
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path[:0] = [str(ROOT / 'lib'), str(HERE)]

import cv2
import numpy as np
import look
import variant
import scene_b as B
import c3
import c_v5 as V5
import c_v5_instep as STEP
import c_eye as EYE
import ringsolid as RS
import openring_d as D
import d_glare
from d_lamplight import ReadingLights, GOLD_WHITE
from d_race_embers import ForgeEmbers
from d_output import output_dir, publish
from core import Camera, Frame, rng

LEADERS = V5.LEADERS
HOLDOUT = V5.LOW_FORGE
RING_C = np.array([0., B.GROUND + 60., 0.])
LAMP_RADIUS, LAMP_BOTTOM, LAMP_TOP = .24, -.26, .65
FORGING_START = D.SHOTS['forging'].d_start
RACE_START = D.SHOTS['race'].d_start
INSTEP_START = D.SHOTS['instep'].d_start
STUDY_START = D.SHOTS['unfinished'].d_start
STUDY_CLOSE = STUDY_START + 160.
_LOCK = threading.Lock()
cv2.setNumThreads(0)


def ease(a, b, t):
    u = np.clip((np.asarray(t) - a) / (b - a), 0., 1.)
    return u * u * (3. - 2. * u)


def caption_backdrop_gain(shot, t, native_y):
    """Dim the working city beneath cutd's fixed lower thirds, before titles.

    D06 occupies y589..717 and D15 y626..681 at native size. The broad soft
    grade never touches the band's layer or central fire. It is fully present
    before each caption enters and clears after it leaves, before the cut.
    """
    if shot not in ('forging', 'unfinished'):
        return np.ones_like(np.asarray(native_y), dtype=float)
    a, b = D.shot_range(shot)
    amount = float(ease(a, a + 24., t) * (1. - ease(b - 24., b, t)))
    zone = ease(510., 584., native_y) * (1. - ease(730., 804., native_y))
    return 1. - .90 * amount * zone


@lru_cache(maxsize=10000)
def race_progress(t):
    return float(np.clip((120. - D.gap_deg(t)) / 40., 0., 1.))


def forge_level(i, t):
    """Inherited v5 power events, translated exactly; work continues at the pause."""
    if t < D.SHOTS['trap'].d_start:
        return 1. + .6 * race_progress(t)
    if t < D.PAUSE_START:
        c = D.legacy_frame('trap', t) if t < D.SHOTS['trap'].d_end else 2639.
        value = V5.forge_level(i, c)
        # Adopted front_smoke_near's complete loss of the low forge's fire.
        if i == HOLDOUT:
            value = 1. - float(ease(2420., 2438., c)) + 1.30 * float(ease(2540., 2552., c))
        return value
    u = float(ease(D.PAUSE_START, D.SETTLED, t))
    return (1. - u) * forge_level(i, D.PAUSE_START - 1.) + u * STEP.WORK_LEVEL


def stroke_frames(shot):
    a, b = D.shot_range(shot)
    if shot in ('forging', 'race'):
        # The adjoining shots share a clock: the first race impact launches sparks
        # twelve frames before the cut, during the end of forging.
        first, last = D.SHOTS['forging'].d_start, D.SHOTS['race'].d_end
        return tuple(start for start, _, _ in D.GAP_SCHEDULE if first <= start < last)
    if shot == 'holdout':
        return (D.HOLDOUT_STRIKE,)
    return tuple(range(a, b, 80 if shot in ('instep', 'unfinished') else 20))


@lru_cache(maxsize=1)
def race_work_beats():
    """Work on every beat, then twice per beat in the final bar.

    The final impact is under the Deep: its approaching grains remain in D2719.
    Work strokes do not all close the band; D owns that separate angular score.
    """
    a, b = D.shot_range('race')
    return tuple(range(a, b - 80, 20)) + tuple(range(b - 80, b + 1, 10))


@lru_cache(maxsize=65536)
def race_lift(i, t):
    """Cumulative smooth surges, retaining the two giants' height advantage."""
    return (.50 if i in LEADERS else .75) * sum(
        float(ease(hit, hit + 6., t)) for hit in race_work_beats())


def spark_reach(i, t):
    """Fraction of forge-to-end distance attained; work sparks retreat at pause."""
    if t >= D.PAUSE_START:
        return .22
    if i in LEADERS:
        return 1.
    if i == HOLDOUT and D.HOLDOUT_STRIKE <= t < D.SHOTS['holdout'].d_end:
        return 1.
    # Closure strokes advance followers; race work beats also raise their
    # frontier when the angular score holds. Trap adds its own inherited surges.
    trap_start, trap_end = D.shot_range('trap')
    trap_surges = max(0, min((trap_end - trap_start) // 20, int((t - trap_start) // 20) + 1))
    work = sum(hit < t for hit in race_work_beats()) if RACE_START < t <= D.SHOTS['race'].d_end else 0
    return .30 + .46 * float(np.clip((120. - D.gap_deg(t)) / 65., 0., 1.)) + .005 * trap_surges + .006 * work


class Schedule(c3.C3Sched):
    """C3's window/seam treatment, with D event clocks and shared bar breathing."""
    ign = 1000.
    tower_rise = FORGING_START
    end = D.SHOTS['unfinished'].d_end + 1.
    smoke_gain = .42
    race_restage = False  # Vision inherits this schedule for its frozen world.

    def __init__(self, shot):
        self.shot = shot
        a, b = D.shot_range(shot)
        self.tower_rise = FORGING_START if shot == 'forging' else a - 120.
        self.end = b + 1.
        self.beats = list(stroke_frames(shot))
        self.pulses = self.beats

    def ashlar_k(self, t): return 0.
    def masonry(self, i, pl, t): return None
    def redness(self, t): return .18 + .22 * (1. - float(ease(D.PAUSE_START, D.COOL_END, t)))
    def tower_lean(self, t): return 0.
    def tower_rise_glow(self, t): self._t = t; return .14
    def tower_grow(self, t): return 1.12
    def tower_extra(self, towers, i, t): return 0.
    def race(self, t): return race_progress(t)
    def power(self, t): return 1.

    def beat_pulse(self, t):
        if t >= D.SETTLED:
            # D treatment: one shared breath per 80-frame bar.
            return .10 * .5 * (1. + math.cos(2. * math.pi * (t - INSTEP_START) / 80.)) * float(ease(D.SETTLED, D.SETTLED + 80., t))
        return max((math.exp(-(t - b) / 4.) for b in self.pulses if 0 <= t - b < 18), default=0.)

    def shutter(self, i, t, pl):
        return .42 + .38 * self.beat_pulse(t)

    def tower_post(self, towers, i, points, t):
        q = points.copy()
        if i not in LEADERS:
            return q
        c = min(D.legacy_frame('trap', t), 2639.)
        k = .14 * float(ease(2580., 2630., c))
        base = towers.base(i)
        s = np.clip((points[:, 1] - base[1]) / max(towers.height(i, t), 1.), 0., 1.)
        inward = -np.array([base[0], 0., base[2]]) / np.linalg.norm(base[[0, 2]])
        return q + (k * towers.height(i, t) * s * s)[:, None] * inward


@contextmanager
def world(schedule):
    if not _LOCK.acquire(blocking=False):
        raise RuntimeError('D forge renderer is not reentrant')
    old = variant.CUT, B.SCHED, B.IGN, B.BEATS
    variant.set_cut('C3')
    B.SCHED, B.IGN, B.BEATS = schedule, schedule.ign, schedule.beats
    try:
        yield
    finally:
        variant.CUT, B.SCHED, B.IGN, B.BEATS = old
        _LOCK.release()


class Forges(B.Towers):
    ZF = 30.  # D's wider staging needs denser masonry, not isolated dotted rims.

    def __init__(self, schedule):
        self.schedule = schedule
        super().__init__()
        c3.layout_towers(self)
        self.t_rise += D.SHOTS['forging'].offset
        self.t_rise[list(LEADERS)] = D.SHOTS['forging'].d_start
        self.rest = self.h_rise + self.J.sum(axis=1)
        self.rest[list(LEADERS)] = max(self.rest) + 8.
        self.design = np.arange(self.k_all) % self.k
        for i in range(self.k, self.k_all):
            self.G[i] = self.G[self.design[i]]
        self.G[LEADERS[0]] = self.G[LEADERS[1]]
        self.design[LEADERS[0]] = self.design[LEADERS[1]]
        # The holdout is visible in front of the far ring, with the same facade.
        self.ang[HOLDOUT] = c3.AZ0 + .36
        self.rad[HOLDOUT] = 34.
        self.rot[HOLDOUT] = -self.ang[HOLDOUT] + np.pi
        self.dly[:] = 0.

    def height(self, i, t):
        rise = float(ease(self.t_rise[i], self.t_rise[i] + 80., t))
        h = self.rest[i] * rise + (4. if i in LEADERS else 2.) * race_progress(t)
        c = min(D.legacy_frame('trap', t), 2639.)
        if i == HOLDOUT:
            # D adds a visible settling to the inherited power-loss event.
            h -= 4. * float(ease(2420., 2438., c)) * (1. - float(ease(2540., 2552., c)))
            h += 7. * float(ease(2540., 2578., c))
        else:
            h += 6. * float(ease(2480., 2538., c))
            if i in LEADERS:
                h += (55. - h) * float(ease(2580., 2636., c))
        return h + (race_lift(i, t) if self.schedule.race_restage else 0.)

    def top(self, i, t):
        p = super().top(i, t)
        return self.schedule.tower_post(self, i, p[None, :], t)[0]

    def _splat(self, ctx, i, pt, colE, a, geo, z, fpx, rw):
        level = forge_level(i, ctx.t)
        # Masonry remains weakly visible when one forge loses its working light.
        ambient = np.array([.005, .0045, .004])[None, :] if pt is self.cur[i]['parts'][0] else 0.
        super()._splat(ctx, i, pt, colE * level + ambient, a, geo, z, fpx, rw)


def ring_state(t):
    state = RS.RingState()
    state.end_caps = state.leading_glyph = state.worked_caps = True
    state.heat, state.write = D.heat(t), D.write(t)
    state.letters, state.glow, state.hammer = .62, .08, .22
    return state


class Scene:
    def __init__(self, shot):
        if not c3.OR.enabled():
            raise ValueError('Cut D forges require LD_OPEN_RING=1')
        self.shot = shot
        D.shot_range(shot)
        self.schedule = Schedule(shot)
        if shot == 'race':
            # Explicit scene opt-in: d_forges inherits Schedule/Forges for
            # vision's frozen world and must retain its accepted architecture.
            self.schedule.race_restage = True
            self.schedule.beats = sorted(set(self.schedule.beats) | set(race_work_beats()))
            self.schedule.pulses = self.schedule.beats
        with world(self.schedule):
            self.towers = Forges(self.schedule)
            self.smoke = EYE._all_smoke(self.towers)
            self.surface_embers = ForgeEmbers(self.towers) if shot == 'race' else None
        self._windows = None
        r = rng(903030)
        self.spark_jitter = r.normal(0., 1., (self.towers.k_all, 64, 3))
        self.spark_delay = r.uniform(0., 2.5, (self.towers.k_all, 64))
        # The first grain contacts on the scored frame (the insert at D5496).
        # Later race packets spread these delays over ten frames; other shots
        # retain the original 2.5-frame dispersion.
        self.spark_delay[:, 0] = 0.
        # Every shot uses one fixed camera-derived orientation, shared late.
        # The race camera now moves. Its opening is the old shared reference,
        # so placement stays identical to the already-rendering forging.
        reference = RACE_START if shot in ('forging', 'race') else INSTEP_START + 80.
        base_rot = c3._closed_ring_frame(1580.)[0]
        self.rotation = D.placement(base_rot, self.camera(reference).pos, RING_C, reference)

    def camera(self, t):
        if self.shot in ('forging', 'race'):
            u = float(ease(FORGING_START, RACE_START, t))
            radius, y, target, hfov = 150. - 71. * u, 32. + 16. * u, 27. + 27. * u, 62. - 10. * u
            az = c3.AZ0 + .06 + .08 * u
            if self.shot == 'race':
                reveal = float(ease(RACE_START, RACE_START + 40., t))
                push = float(ease(RACE_START + 40., D.SHOTS['race'].d_end - 1., t))
                radius += (102. - 34. * push - radius) * reveal
                y += (32. - y) * reveal
                target += (46. + 2. * push - target) * reveal
                hfov += (62. + 2. * push - hfov) * reveal
                az -= (.14 * push) * reveal
        elif self.shot == 'holdout':
            # Put the holdout's throat directly beneath the end it first reaches.
            radius, y, target, hfov, az = 80., 36., 50., 48., c3.AZ0 + .36
        elif self.shot == 'trap':
            # Keep the low forge visible through its full four-unit settlement.
            radius, y, target, hfov, az = 85., 24., 49., 48., c3.AZ0 + .20
        elif self.shot == 'unfinished':
            # See every lamp leave its own tower before moving in to study the
            # letters; the close framing alone hid the nearest departures.
            u = float(ease(STUDY_START, STUDY_CLOSE, t))
            radius, y, target, hfov, az = 145. - 65. * u, 32. + 17. * u, 30. + 24. * u, 62. - 10. * u, c3.AZ0 + .14
        else:
            u = float(ease(INSTEP_START, STUDY_CLOSE, t))
            radius, y, target, hfov, az = 88. - 8. * u, 45. + 4. * u, 50. + 4. * u, 52., c3.AZ0 + .14
        pos = np.array([radius * math.cos(az), B.GROUND + y, radius * math.sin(az)])
        return Camera(pos, [0., B.GROUND + target, 0.], hfov=hfov,
                      focus=np.linalg.norm(RING_C - pos), aperture=.012)

    def ring_frame(self, t):
        centre = RING_C.copy()
        if self.shot in ('forging', 'race'):
            centre[1] -= 8. * (1. - float(ease(FORGING_START + 160., RACE_START, t)))
        return self.rotation, centre, 2.2

    def ends(self, t):
        frames = np.asarray(t)
        centre = np.broadcast_to(RING_C, frames.shape + (3,)).copy()
        if self.shot in ('forging', 'race'):
            centre[..., 1] -= 8. * (1. - ease(FORGING_START + 160., RACE_START, frames))
        th = np.stack(D.theta_range(frames), axis=-1)
        p = np.stack([RS.R_MID * np.cos(th), np.zeros_like(th), RS.R_MID * np.sin(th)], axis=-1)
        return centre[..., None, :] + 2.2 * p @ self.rotation.T

    def lamp_positions(self, t):
        n = self.towers.k_all
        ids = np.arange(n)
        rotation, centre, size = self.ring_frame(t)
        theta = D.arc_theta(t, (ids + .5) / n)
        # Hang outside the extant band, leaving the opening unobstructed.
        local = np.column_stack([(RS.R_OUT + .65) * np.cos(theta), np.full(n, .75),
                                 (RS.R_OUT + .65) * np.sin(theta)])
        target = centre + size * local @ rotation.T
        source = np.array([self.towers.top(i, STUDY_START) for i in ids])
        u = ease(STUDY_START + ids * 2., STUDY_START + 80. + ids * 2., t)
        return source * (1. - u[:, None]) + target * u[:, None], u

    def ring_layer(self, ctx):
        rotation, centre, size = self.ring_frame(ctx.t)
        shading_frame = self.ring_shading_frame(ctx.t)
        state = ring_state(shading_frame)
        env = RS.Env(above=(.18, .19, .21), horizon=(.42, .35, .23), below=(.10, .045, .015))
        env.lobe([-.7, .6, .4], [.85, .88, .92], 5.)
        for i in range(self.towers.k_all):
            env.point(self.towers.top(i, shading_frame), np.array([4., 1.7, .5]) * forge_level(i, shading_frame), 2.)
        if self.shot == 'unfinished':
            lamps, amount = self.lamp_positions(ctx.t)
            state.letters, state.exposure = .055, .65
            state.reading_light = ReadingLights(lamps, powers=5. * amount)
            for p, k in zip(lamps, amount):
                env.point(p, GOLD_WHITE * 2. * k, .4)
        handoff = float(ease(FORGING_START, FORGING_START + 10., ctx.t)) if self.shot == 'forging' else 1.
        if handoff > 0.:
            rgb, alpha, depth = RS.render(ctx.cam, ctx.fr.W, ctx.fr.H, rotation, centre, size,
                                         state, env, th_range=D.theta_range(ctx.t))
        if handoff < 1.:
            # Local import avoids inscription's intentional dependency on our
            # frozen camera. Reuse its material, including the narrow hot ends,
            # instead of maintaining a second approximation of the cut frame.
            import d_inscription as incoming
            entry_rgb, entry_alpha, entry_depth = RS.render(
                ctx.cam, ctx.fr.W, ctx.fr.H, rotation, centre, size,
                incoming.ring_state(incoming.END - 1), incoming.environment(incoming.END - 1),
                th_range=incoming.theta_range(incoming.END - 1), ss=2, nt=1000, npp=64)
            if handoff == 0.:
                rgb, alpha, depth = entry_rgb, entry_alpha, entry_depth
            else:
                rgb = entry_rgb * (1. - handoff) + rgb * handoff
                alpha = entry_alpha * (1. - handoff) + alpha * handoff
                depth = np.minimum(entry_depth, depth)
        before = RS.merge_occluder(ctx.fr, alpha, depth)
        return rgb * RS.visibility(before, depth, ctx.fr.H, ctx.fr.W)[..., None]

    def spark_paths(self, t):
        """Actual rendered trajectories with tower identity and frontier reach."""
        return self._spark_samples(t)[:3]

    def _spark_samples(self, t):
        """Particle keys also identify packets across shutter-boundary changes."""
        points, energy, owners, keys = [], [], [], []
        count = self.spark_jitter.shape[1]
        for packet, hit in enumerate(self.schedule.beats):
            stream = self.shot == 'race' and hit > RACE_START
            if stream and t <= RACE_START:
                continue
            flight = 18. if stream else 12.
            spread = 4. if stream else 1.
            if not hit - flight - 3. <= t < hit + 2.5 * spread + 1.5:
                continue
            ends = self.ends(hit)
            for i in range(self.towers.k_all):
                if self.shot == 'holdout' and i != HOLDOUT:
                    continue
                base = self.towers.top(i, hit - 12.) - [0., .5, 0.]
                side = LEADERS.index(i) if i in LEADERS else np.argmin(np.linalg.norm(ends - base, axis=1))
                # Each particle meets the end at its own arrival time, including
                # movement during the stroke; its launch stays at its own birth.
                delay = self.spark_delay[i] * spread
                end = self.ends(hit + delay)[:, side]
                base = np.array([self.towers.top(i, hit - flight + d) for d in delay]) - [0., .5, 0.]
                reach = spark_reach(i, hit)
                u = (t - hit + flight - delay) / flight
                alive = (u >= -1e-10) & (u <= 1. + 1e-10)
                u = np.clip(u, 0., 1.)
                travel = u if reach == 1. else np.sin(np.pi * u)
                q = base + (end - base) * (reach * travel[:, None])
                q += self.spark_jitter[i] * (.60 * np.sin(np.pi * u))[:, None]
                e = alive * (.8 + .2 * np.sin(np.pi * u)) * forge_level(i, hit)
                points.append(q)
                energy.append(e)
                owners.extend([i] * len(q))
                keys.extend((packet * self.towers.k_all + i) * count + np.arange(len(q)))
        if not points:
            return np.empty((0, 3)), np.empty(0), np.empty(0, dtype=int), np.empty(0, dtype=int)
        return np.concatenate(points), np.concatenate(energy), np.array(owners), np.array(keys)

    def race_handoff(self, t):
        return float(ease(RACE_START, RACE_START + 10., t)) if self.shot == 'race' else 1.

    def ring_shading_frame(self, t):
        amount = self.race_handoff(t)
        return (RACE_START - 1.) * (1. - amount) + t * amount if amount < 1. else t

    def sparks(self, ctx):
        # Incandescent grains shed heat rapidly: show a short luminous streak,
        # not the entire mechanical-shutter path of a persistent bright rod.
        amount = self.race_handoff(ctx.t)
        def draw(p0, p1, energy, gain):
            if not len(p1) or gain <= 0.: return
            z = np.maximum((p1 - ctx.cam.pos) @ ctx.cam.R[2], 1.)
            ctx.fr.splat(p0, p1, .014, energy * .012 * (ctx.cam.f_px(1920) / z) ** 2 * gain,
                         [1., .72, .3], ctx.cam0, ctx.cam1, zref=0.)
        if amount < 1.:
            # Continue the outgoing centred shutter. Dead future packets must
            # not erase the live packet's streak when their row counts differ.
            before, _, _, keys0 = self._spark_samples(ctx.t - .06)
            after, energy, _, keys1 = self._spark_samples(ctx.t + .06)
            p0 = after.copy()
            rows = np.searchsorted(keys0, keys1)
            valid = rows < len(keys0)
            valid[valid] &= keys0[rows[valid]] == keys1[valid]
            p0[valid] = before[rows[valid]]
            draw(p0, after, energy, 1. - amount)
        if amount > 0.:
            p0, _, ids0 = self.spark_paths(max(ctx.t - .06, ctx.t0))
            p1, energy, ids1 = self.spark_paths(min(ctx.t + .06, ctx.t1))
            if not np.array_equal(ids0, ids1): p0 = p1
            draw(p0, p1, energy, amount)

    def window_sources(self):
        if self._windows is None:
            rows = []
            for i, g in enumerate(self.towers.G):
                win = g[3]
                for key in np.unique(win['key']):
                    m = win['key'] == key
                    p, n = win['p'][m].mean(0), win['n'][m].mean(0)
                    if abs(n[1]) < .5:
                        rows.append((i, int(key), p, n))
            self._windows = rows
        return self._windows

    def drops(self, ctx):
        # Stable window IDs make the recipients real openings in the facade.
        rows = self.window_sources()
        near = set(c3.GoldRain._near_forges(self.towers))
        if self.shot in ('trap', 'holdout'): near.add(HOLDOUT)
        shared = self.shot in ('instep', 'unfinished')
        positions, prev, energies = [], [], []
        rotation, centre, size = self.ring_frame(ctx.t)
        for i, key, local, normal in rows:
            if not shared and i not in near: continue
            target = self.towers.world(i, local[None, :], self.towers.height(i, ctx.t))
            target = self.schedule.tower_post(self.towers, i, target, ctx.t)[0]
            nw = self.towers.nworld(i, normal[None, :])[0]
            if target[1] < B.GROUND + 6. or target[1] > centre[1] - size * RS.R_OUT - 1.: continue
            if np.dot(nw, centre - target) <= 0.: continue
            phase = ((key * .61803398875 + i * .137) % 1.)
            period = 104. if shared else 64.
            u = (phase + ctx.t / period) % 1.
            v = (phase + ctx.t0 / period) % 1.
            if u < .02 or u > .99 or v > u: continue
            birth = ctx.t - u * period
            theta = D.arc_theta(birth, phase)
            birth_rotation, birth_centre, birth_size = self.ring_frame(birth)
            p, _ = RS.local_points(np.array([theta]), np.array([0.]))
            origin = birth_centre + birth_size * p[0] @ birth_rotation.T
            end = target + nw * .12
            def at(x):
                q = origin + (end - origin) * x
                q[1] = origin[1] + (end[1] - origin[1]) * x * x
                return q
            positions.append(at(u)); prev.append(at(v))
            value = .30 if shared else .40 * min(forge_level(i, ctx.t), 2.)
            # The lost recipient's allocation goes to the neighboring forges.
            if self.shot == 'trap' and i != HOLDOUT:
                value *= 1. + (1. - min(forge_level(HOLDOUT, ctx.t), 1.)) / max(len(near) - 1, 1)
            energies.append(value)
        if positions:
            points = np.array(positions)
            z = np.maximum((points - ctx.cam.pos) @ ctx.cam.R[2], 1.)
            ctx.fr.splat(np.array(prev), points, .045, np.array(energies) * .016 * (ctx.cam.f_px(1920) / z) ** 2,
                         [1., .87, .49], ctx.cam0, ctx.cam1, zref=0.)

    def lamps(self, ctx):
        if self.shot != 'unfinished': return
        points, u = self.lamp_positions(ctx.t)
        before, _ = self.lamp_positions(ctx.t0)
        z = np.maximum((points - ctx.cam.pos) @ ctx.cam.R[2], 1.)
        active = (u > 0.).astype(float)
        projected = active * (ctx.cam.f_px(1920) / z) ** 2
        # At this distance the lamp is its light: a warm white centre with a
        # low-energy halo. Tiny luminous cage outlines read as extra glyphs.
        ctx.fr.splat(before, points, .075, projected * .050,
                     GOLD_WHITE, ctx.cam0, ctx.cam1, zref=0.)
        ctx.fr.splat(before, points, .34, projected * .014,
                     GOLD_WHITE, ctx.cam0, ctx.cam1, profile=1, zref=0.)

    def glare_layer(self, ctx):
        score = D.glare(ctx.t)
        pulse = self.schedule.beat_pulse(ctx.t)
        amount = self.race_handoff(ctx.t)
        if amount < 1.:
            incoming = D.glare(RACE_START - 1.)
            score = D.Glare(*(a * (1. - amount) + b * amount for a, b in zip(incoming, score)))
            pulse = self.schedule.beat_pulse(RACE_START - 1.) * (1. - amount) + pulse * amount
        if self.shot == 'forging':
            score = score._replace(gain=score.gain * float(ease(FORGING_START, FORGING_START + 10., ctx.t)))
        if score.gain <= 0.:
            return np.zeros((ctx.fr.H, ctx.fr.W, 3), np.float32)
        ends = self.ends(ctx.t)
        rotation, centre, size = self.ring_frame(ctx.t)
        caps = np.concatenate([RS.cap_points(theta, 24)[0][1]
                               for theta in D.theta_range(ctx.t)])
        caps = centre + size * caps @ rotation.T
        depth = float(np.min((ends - ctx.cam.pos) @ ctx.cam.R[2]))
        visibility = c3.occ_vis(ctx.fr, depth - .65, ctx.fr.H, ctx.fr.W)
        return d_glare.render(ctx.cam, ctx.fr.W, ctx.fr.H, ends, caps, score,
                              pulse, visibility)

    @lru_cache(maxsize=8)
    def race_fire_world(self):
        """Lift the last forging anchors onto the vertical plane at the axis.

        The screen-space forging ease is already settled at D2399. Intersect
        its two pixel rays from the D2400 camera with this fixed world plane;
        subsequent cameras project the same root and tip, without screen locks.
        """
        pixels = np.array(self.central_fire_anchors(RACE_START - 1.))
        cam = self.camera(RACE_START)
        xy = (pixels - np.array([959.5, 401.5])) / cam.f_px(1920)
        rays = cam.R[2] + xy[:, :1] * cam.R[0] - xy[:, 1:] * cam.R[1]
        normal = cam.pos * np.array([1., 0., 1.])
        distance = -np.dot(cam.pos - c3.FIRE_ROOT, normal) / (rays @ normal)
        return cam.pos + rays * distance[:, None]

    def central_fire_anchors(self, t):
        """Continue the source at the cut, then keep it below the rising band.

        Only this fire's screen staging changes. The locked camera otherwise
        sends the ground source below picture by D2240. Over the first 80 frames
        the same live source rises above the caption area as towers surround it.
        Coordinates are native pixels, as required by d_thinking_fire.draw.
        """
        if self.shot == 'race' and t >= RACE_START:
            x, y, _ = self.camera(t).project(self.race_fire_world(), 1920, 804)
            return np.array([x[0], y[0]]), np.array([x[1], y[1]])
        points = np.array([c3.FIRE_ROOT, c3.FIRE_ROOT + [0., c3.HF, 0.]])
        x, y, _ = self.camera(FORGING_START).project(points, 1920, 804)
        amount = float(ease(FORGING_START, FORGING_START + 80., t))
        root = np.array([x[0], y[0] + (540. - y[0]) * amount])
        height = (y[0] - y[1]) * (1. + .25 * amount)
        return root, root - np.array([0., height])

    def draw_central_fire(self, hdr, ctx):
        if self.shot not in ('forging', 'race'): return
        import d_thinking_fire as thinking_fire
        root, tip = self.central_fire_anchors(ctx.t)
        world_root = self.race_fire_world()[0] if self.shot == 'race' else c3.FIRE_ROOT
        depth = float((world_root - ctx.cam.pos) @ ctx.cam.R[2])
        visibility = c3.occ_vis(ctx.fr, depth, ctx.fr.H, ctx.fr.W)
        thinking_fire.draw(hdr, root, tip, ctx.t, bright=1.1, vis=visibility, scale=ctx.scale)

    def frame(self, f, scale=1.):
        a, b = D.shot_range(self.shot)
        if not a <= f < b: raise ValueError(f'D{f} is outside {self.shot} [{a},{b})')
        ctx = argparse.Namespace(t=float(f), f=f, t0=max(f - .25, a), t1=min(f + .25, b - .02), scale=scale)
        ctx.cam, ctx.cam0, ctx.cam1 = self.camera(f), self.camera(ctx.t0), self.camera(ctx.t1)
        ctx.fr = Frame(scale)
        ctx.fr.set(focus=ctx.cam.focus, aperture=ctx.cam.aperture, bokeh_pow=.25, bokeh_cap=1.6,
                   fog_start=100., fog_len=150., near=.3)
        with world(self.schedule):
            self.towers.prepare(ctx)
            ring = self.ring_layer(ctx)
            self.towers.emit(ctx, RING_C, np.array([1., .65, .31]), 90.)
            self.smoke.emit(ctx, RING_C, np.array([1., .60, .28]), 55.)
            if self.shot == 'race':
                self.surface_embers.emit_bounded(ctx, self.ends(ctx.t0), self.ends(ctx.t1),
                                                 float(ease(RACE_START, RACE_START + 20., f)))
            self.sparks(ctx)
            self.drops(ctx)
            self.lamps(ctx)
            city = ctx.fr.resolve()
            grade = caption_backdrop_gain(self.shot, f, np.arange(ctx.fr.H) / scale)
            city *= grade[:, None, None]
            hdr = city + ring + self.glare_layer(ctx)
            self.draw_central_fire(hdr, ctx)
        if not np.isfinite(hdr).all(): raise ValueError(f'Non-finite D{f}')
        return look.finish(hdr, exposure=1.05, bloom_strength=.08, bloom_threshold=1.1,
                           streak_strength=0., vignette_amount=.20)


def frames(spec):
    result = []
    for part in spec.split(','):
        span, _, stride = part.partition(':')
        step = int(stride or 1)
        if step <= 0: raise ValueError('Frame step must be positive')
        if '-' in span:
            a, b = map(int, span.split('-'))
            if b < a: raise ValueError('Reversed frame range')
            result.extend(range(a, b + 1, step))
        else: result.append(int(span))
    if len(set(result)) != len(result): raise ValueError('Duplicate frames')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('shot', choices=tuple(D.SHOTS))
    parser.add_argument('--range', required=True)
    parser.add_argument('--scale', type=float, default=1.)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--format', choices=('png', 'jpg'), default='jpg')
    args = parser.parse_args()
    if not 0. < args.scale <= 1.: parser.error('Scale must lie in (0,1]')
    requested = frames(args.range)
    a, b = D.shot_range(args.shot)
    if any(not a <= f < b for f in requested): parser.error('Frames outside selected D shot')
    destination = output_dir(args.shot, args.out, ROOT)
    paths = V5.output_paths(destination, requested, args.format)
    scene = Scene(args.shot)
    destination.mkdir(parents=True, exist_ok=True)
    for frame, path in zip(requested, paths):
        started = time.perf_counter()
        im = scene.frame(frame, args.scale)
        elapsed = time.perf_counter() - started
        publish(path, im)
        print(json.dumps(dict(frame=frame, shot=args.shot, seconds=elapsed,
                              peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == 'darwin' else 1024),
                              shape=list(im.shape), sha256=hashlib.sha256(im.tobytes()).hexdigest())), flush=True)


if __name__ == '__main__':
    main()
