"""Default-off, test-resolution studies of C5's failed unilateral withdrawal.

Accepted source is c_v5.py, left untouched. ``accepted`` delegates to its Scene;
``pilot30`` retains a .30 pilot in the withdrawing forge; ``pilot30_gap`` also
completes the other towers' first rise at C2504 instead of C2538. The director's
later ``front_smoke`` and ``front_smoke_near`` studies bring that existing forge
into the foreground, let its crown go fully dark, and give its withdrawal a
rising neutral smoke cue. Every alternative remains an unadopted study.

Public API mirrors the base Scene's frame arguments::

    scene = Scene(variant='pilot30')
    rgb = scene.frame(2479, scale=.5)  # reuse the scene for a sequential batch
    rgb = render(2479, scale=.5, variant='accepted')  # one frame, fresh scene

Use one process per variant, with frames rendered sequentially. c_v5 and its
world already use process globals: candidate_world restores these two extra
substitutions in finally, but cannot make unrelated render threads safe.
Construction and each frame run inside that scope. Use ``scene.world(frame)``
for candidate geometry diagnostics: outside it, all object state is accepted.
No file output, render launch, cache policy or final-selection decision is made
by this module. The caller must use new candidate directories and resource gates.
"""
from contextlib import contextmanager
import math
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import c_v5 as C

PILOT_VARIANTS = ('pilot30', 'pilot30_gap')
STAGED_VARIANTS = ('front_smoke', 'front_smoke_near')
VARIANTS = ('accepted', *PILOT_VARIANTS, *STAGED_VARIANTS)
# Radius, offset from the existing world's camera azimuth, rest-height offset.
# The nearer alternative is lower so its returned crown remains below the
# paired leaders. Neither alternative moves a tower during the shot.
STAGING = {'front_smoke': (48., .23, 0.),
           'front_smoke_near': (56., .20, -4.)}
_ACCEPTED_FORGE_LEVEL = C.forge_level
_ACCEPTED_HEIGHT = C.Forges.height


def _variant(name):
    if name not in VARIANTS:
        raise ValueError(f'Unknown Trap variant {name!r}; choose from {VARIANTS}')
    return name


def forge_level(i, t, variant='accepted'):
    """Alter only the low forge during withdrawal/return; preserve other paths."""
    _variant(variant)
    # Delegate endpoints and all non-Trap times to the actual accepted function.
    # This preserves its operation order, including the cold-shot recursion.
    if variant == 'accepted' or i != C.LOW_FORGE or not 2420. < t < 2552.:
        return _ACCEPTED_FORGE_LEVEL(i, t)
    if variant in STAGED_VARIANTS:
        return (1. - C.ease(2420, 2438, t)) + 1.30 * C.ease(2540, 2552, t)
    return (1. - .70 * C.ease(2420, 2438, t)) + 1.00 * C.ease(2540, 2552, t)


def forge_height(towers, i, t, variant='accepted'):
    """B alone accelerates competitors' initial rise; all final heights agree."""
    _variant(variant)
    if variant != 'pilot30_gap' or i == C.LOW_FORGE or not 2480. < t < 2538.:
        return _ACCEPTED_HEIGHT(towers, i, t)
    # This interval ends before paired-leader growth starts at C2580. No new
    # leader identity, camera, crown attachment or end-state expression is used.
    return towers.rest[i] + 6. * C.ease(2480, 2504, t)


@contextmanager
def candidate_world(variant='accepted'):
    """Scope only the two candidate schedules; restore even if rendering fails."""
    _variant(variant)
    if variant == 'accepted':
        yield
        return
    before = C.forge_level, C.Forges.height
    try:
        C.forge_level = lambda i, t: forge_level(i, t, variant)
        C.Forges.height = _ACCEPTED_HEIGHT
        if variant == 'pilot30_gap':
            C.Forges.height = lambda towers, i, t: forge_height(towers, i, t, variant)
        yield
    finally:
        C.forge_level, C.Forges.height = before


def withdrawal_smoke_gain(t):
    """A finite cue through withdrawal/surge, gone before paired growth."""
    return C.ease(2420, 2438, t) * (1. - C.ease(2540, 2578, t))


def smoke_packets(towers, t):
    """Pure world-space parcel state; no RNG, meshes, renderer or JIT call."""
    births = np.arange(2420., 2540., 2.)
    age = t - births
    alive = (age >= 0.) & (age < 90.)
    births, age = births[alive], age[alive]
    if not len(age):
        return births, np.empty((0, 3)), age, age
    roots = np.array([towers.top(C.LOW_FORGE, b) for b in births])
    phase = .37 * births
    drift = np.stack((.35 * np.sin(.07 * age + phase) + .005 * age,
                      .35 + .115 * age,
                      .24 * np.cos(.055 * age + phase)), axis=1)
    radius = .35 + .026 * age
    entering = np.clip(age / 4., 0., 1.)
    density = .032 * entering * (1. - age / 90.) ** 1.4 * withdrawal_smoke_gain(t)
    return births, roots + drift, radius, density


class WithdrawalSmoke:
    """Use the world's existing smoke splat profile for a brief neutral plume."""
    def __init__(self, original, towers):
        self.original, self.towers = original, towers

    def emit(self, ctx, *light):
        self.original.emit(ctx, *light)
        if withdrawal_smoke_gain(ctx.t) <= 0:
            return
        births, positions, radius, density = smoke_packets(self.towers, ctx.t)
        if not len(births):
            return
        # The shutter centres may straddle a birth/death. Use the same current
        # packet identities at each shutter end so particles cannot mismatch.
        def shutter_positions(t):
            age = np.maximum(t - births, 0.)
            roots = np.array([self.towers.top(C.LOW_FORGE, b) for b in births])
            phase = .37 * births
            return roots + np.stack((.35 * np.sin(.07 * age + phase) + .005 * age,
                                     .35 + .115 * age,
                                     .24 * np.cos(.055 * age + phase)), axis=1)
        z = np.maximum((positions - ctx.cam.pos) @ ctx.cam.R[2], 1.)
        projected_radius = radius * ctx.cam.f_px(1920) / z
        energy = density * np.pi * projected_radius ** 2
        active = energy > 1e-8
        if active.any():
            ctx.fr.splat(shutter_positions(ctx.t0)[active], shutter_positions(ctx.t1)[active],
                         radius[active], energy[active], np.array([.80, .85, .90]),
                         ctx.cam0, ctx.cam1, profile=1, zref=0., rmax=100.)


@contextmanager
def staged_scene(scene, variant):
    """Change only this scene's foreground forge; always restore object state."""
    if variant not in STAGED_VARIANTS:
        yield
        return
    towers = scene.towers
    previous = towers.rad, towers.ang, towers.rot, towers.rest, scene.smoke
    previous_smoke_heights = scene.smoke.H
    radius, offset, lower = STAGING[variant]
    try:
        towers.rad, towers.ang, towers.rest = towers.rad.copy(), towers.ang.copy(), towers.rest.copy()
        towers.rot = list(towers.rot)
        towers.rad[C.LOW_FORGE] = radius
        towers.ang[C.LOW_FORGE] = C.c3.AZ0 + offset
        towers.rot[C.LOW_FORGE] = -towers.ang[C.LOW_FORGE] + math.pi
        towers.rest[C.LOW_FORGE] += lower
        # Existing smoke caches crown heights at birth. A lowered foreground
        # forge must lower that row too, or its old smoke floats off the crown.
        scene.smoke.H = previous_smoke_heights.copy()
        scene.smoke.H[C.LOW_FORGE] += lower
        scene.smoke = WithdrawalSmoke(scene.smoke, towers)
        yield
    finally:
        towers.rad, towers.ang, towers.rot, towers.rest, scene.smoke = previous
        scene.smoke.H = previous_smoke_heights


class Scene:
    """Reuse the untouched base scene while scoping a candidate to each call."""
    def __init__(self, variant='accepted'):
        self.variant = _variant(variant)
        if self.variant == 'accepted':
            self._scene = C.Scene()
        else:
            with candidate_world(self.variant):
                self._scene = C.Scene()

    def frame(self, f, scale=.5):
        if self.variant == 'accepted':
            return self._scene.frame(f, scale)
        with self.world(f):
            return self._scene.frame(f, scale)

    @contextmanager
    def world(self, f):
        # Geometry and smoke changes apply to Trap only. The same wrapper can
        # still reproduce Cold/Unfinished on accepted paths, without restaging.
        if self.variant in STAGED_VARIANTS:
            start, end = C.SHOTS['trap']
            if not start <= f < end:
                yield
                return
        with candidate_world(self.variant):
            with staged_scene(self._scene, self.variant):
                yield

    def __getattr__(self, name):
        return getattr(self._scene, name)


def render(f, scale=.5, variant='accepted'):
    """Return the base renderer's RGB array; construct a fresh scene each call."""
    return Scene(variant=variant).frame(f, scale)
