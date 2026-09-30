"""Explicit cut-D forge world; imported C3/C5 renderers retain their state.

This helper never draws crown flames. The crowns shot owns its two orange
beacons. Other callers supply their camera, Ring and per-tower bend.
"""
from contextlib import contextmanager
import hashlib
import math
from pathlib import Path
import pickle
from types import SimpleNamespace

import numpy as np

import c_v5 as V5
import c_d as CD
import c3
import c_eye
import scene_b as B
import variant
from core import Frame, rng

LEADERS = CD.LEADERS
RING_C = CD.RING_C.copy()
AZIMUTH = c3.AZ0 + .082
ORANGE = np.array([1., .145, .012], np.float32)
GOLD = np.array([1., .77, .31], np.float32)


def _geometry_readonly(module, giants, alt):
    """Reuse the legacy cache without creating a file under renders/."""
    key = hashlib.sha1(Path(module.__file__).read_bytes()
                       + repr((tuple(giants), bool(alt))).encode()).hexdigest()[:16]
    cached = Path(B.__file__).resolve().parents[2] / 'renders' / 'embers_A3' / 'cache' / f'towers_{key}.pkl'
    if cached.is_file():
        with cached.open('rb') as stream:
            return pickle.load(stream)
    return module.build_all(giants, alt=alt), module.build_skyline()


def bend_points(towers, i, points, t, amount):
    base = towers.base(i)
    height = max(towers.height(i, t), 1.)
    inward = -base * np.array([1., 0., 1.])
    inward /= max(np.linalg.norm(inward), 1e-9)
    u = np.clip((points[:, 1] - base[1]) / height, 0., 1.3)
    displacement = amount * height * u * u
    out = points.copy()
    out += displacement[:, None] * inward
    out[:, 1] -= .5 * amount * displacement * u
    return out


class Schedule(CD.Schedule):
    """The industrial D shader, with an explicit frozen architectural state."""
    def __init__(self, phase='race', world_frame=2959.):
        super().__init__(phase)
        self.end = 9200.
        self.world_frame = float(world_frame)
        self.lean_amount = None
        self.red = None
        self.beacon_power = 0.

    def amount(self, i):
        if self.lean_amount is None:
            return 0.
        if isinstance(self.lean_amount, dict):
            return self.lean_amount.get(i, 0.)
        if np.ndim(self.lean_amount):
            return self.lean_amount[i]
        return float(self.lean_amount)

    def redness(self, t):
        return super().redness(t) if self.red is None else self.red
    def beacon_flux(self, i, t):
        return self.beacon_power*(.90+.07*math.sin(.17*t+1.3*i)
                                  +.03*math.sin(.47*t+2.1*i))
    def tower_post(self, towers, i, points, t):
        if self.lean_amount is None:
            return super().tower_post(towers, i, points, self.world_frame)
        return bend_points(towers, i, points, t, self.amount(i))
    def race(self, t): return super().race(self.world_frame)


class Forges(CD.Forges):
    def __init__(self, schedule):
        super().__init__(schedule)
        self._frozen_height = np.array([CD.Forges.height(self, i, schedule.world_frame)
                                        for i in range(self.k_all)])

    def height(self, i, t):
        if hasattr(self, '_frozen_height'):
            return float(self._frozen_height[i])
        return super().height(i, self.schedule.world_frame)

    def receiving_window(self, i, t):
        """Use an existing inward-facing vertical window from the forge mesh."""
        windows = self.G[i][3]
        points = self.world(i, windows['p'], self.height(i, t))
        normals = self.nworld(i, windows['n'])
        inward = -self.base(i)*[1., 0., 1.]
        inward /= max(np.linalg.norm(inward), 1e-9)
        wanted_height = self.top(i, t)[1]-4.2
        eligible = (normals @ inward > .45) & (np.abs(normals[:, 1]) < .3)
        if not eligible.any():
            raise ValueError(f'Forge {i} has no inward-facing receiving window')
        indices = np.flatnonzero(eligible)
        score = np.abs(points[indices, 1]-wanted_height)
        point = points[indices[np.argmin(score)]]
        return self.schedule.tower_post(self, i, point[None], t)[0]

    def _splat(self, ctx, i, pt, colE, a, geo, z, fpx, rw):
        # Apply the other leader's beacon as a real directional surface light;
        # window/self-emission continues to use the gold forge palette.
        if pt is self.cur[i]['parts'][0]:
            view = ctx.cam.pos - pt['P']
            visible = np.clip((view * pt['N']).sum(1) / np.maximum(np.linalg.norm(view, axis=1), 1e-8), 0., 1.)
            if i in LEADERS and self.schedule.beacon_power > 0.:
                other = LEADERS[1] if i == LEADERS[0] else LEADERS[0]
                source = self.top(other, ctx.t) + np.array([0., 2., 0.])
                ray = source - pt['P']
                distance = np.linalg.norm(ray, axis=1)
                lambert = np.maximum((ray * pt['N']).sum(1) / np.maximum(distance, 1e-8), 0.)
                brightness = .14 * self.schedule.beacon_flux(other,ctx.t) * lambert * visible / (1. + (distance / 35.) ** 2)
                # Its own broad brazier lights the crenellations and upper
                # storey from above; the opposite one still lights the face.
                source = self.top(i, ctx.t) + np.array([0., 2.0, 0.])
                # A broad fire is an area source. Sample its facing edge, so
                # an emitter at the shaft's centre does not leave every outer
                # vertical wall facing away from its own flame.
                radial = pt['P']-source
                radial[:,1] = 0.
                radial /= np.maximum(np.linalg.norm(radial,axis=1,keepdims=True),1e-8)
                source = source + 2.6*radial
                ray = source - pt['P']
                distance = np.linalg.norm(ray, axis=1)
                own_lambert = np.maximum((ray * pt['N']).sum(1) / np.maximum(distance, 1e-8), 0.)
                upper = np.clip(1.-(source[:,1]-pt['P'][:,1])/12., 0., 1.)
                brightness += .48*self.schedule.beacon_flux(i,ctx.t)*own_lambert*upper*visible/(1.+(distance/7.)**2)
                colE = colE + brightness[:, None] * ORANGE
        super()._splat(ctx, i, pt, colE, a, geo, z, fpx, rw)


class SmokeSources:
    """The inherited plume samples base + crown height, so offset that source.

    The source frame follows the current bend; the plume's existing age/noise
    motion remains untouched. Its height table uses the same fixed D heights.
    This proxy is used only by smoke, never by tower geometry.
    """
    def __init__(self, towers):
        self._tw = towers
        self.k = towers.k_all
        self.t = 0.

    def __getattr__(self, name):
        return getattr(self._tw, name)

    def base(self, i):
        displacement = self._tw.top(i, self.t) - B.Towers.top(self._tw, i, self.t)
        return self._tw.base(i) + displacement


def make_context(f, scale, camera, shutter=.25):
    ctx = SimpleNamespace(t=float(f), f=int(f), t0=float(f)-shutter,
                          t1=float(f)+shutter, scale=scale)
    ctx.cam = camera(float(f)) if callable(camera) else camera
    ctx.cam0 = camera(ctx.t0) if callable(camera) else camera
    ctx.cam1 = camera(ctx.t1) if callable(camera) else camera
    ctx.fr = Frame(scale)
    ctx.fr.set(focus=ctx.cam.focus, aperture=ctx.cam.aperture, bokeh_pow=.25,
               bokeh_cap=1.6, fog_start=110., fog_len=140., near=.3)
    return ctx


class Scene:
    def __init__(self, phase='race', world_frame=None):
        if world_frame is None:
            world_frame = 4079. if phase == 'trap' else 2959.
        self.schedule = Schedule(phase, world_frame)
        with self.world():
            old_cache = B._tower_cache
            try:
                B._tower_cache = _geometry_readonly
                self.towers = Forges(self.schedule)
                self.smoke = c_eye._all_smoke(self.towers)
                self.smoke.tw = SmokeSources(self.towers)
            finally:
                B._tower_cache = old_cache
        random = rng(19346)
        self.spark_velocity = random.normal(0., 1., (90, 3)) * [.15, .12, .15]
        self.spark_velocity[:, 1] += .055
        self.spark_energy = random.uniform(.5, 1.5, 90)
        self.drop_fraction = random.uniform(.04, .96, 52)
        self.drop_phase = random.uniform(0., 1., 52)
        self.drop_leader = random.integers(0, 2, 52)

    @contextmanager
    def world(self, lean=None, red=None, beacon_power=0.):
        old = variant.CUT, B.SCHED, B.IGN, B.BEATS
        settings = self.schedule.lean_amount, self.schedule.red, self.schedule.beacon_power
        try:
            variant.set_cut('C3')
            B.SCHED, B.IGN, B.BEATS = self.schedule, self.schedule.ign, self.schedule.beats
            self.schedule.lean_amount, self.schedule.red, self.schedule.beacon_power = lean, red, beacon_power
            yield self
        finally:
            variant.CUT, B.SCHED, B.IGN, B.BEATS = old
            self.schedule.lean_amount, self.schedule.red, self.schedule.beacon_power = settings

    def prepare(self, ctx):
        self.towers.prepare(ctx)

    def emit(self, ctx, light_pos=RING_C, light_col=GOLD, light_pow=70.):
        self.towers.emit(ctx, light_pos, np.asarray(light_col), light_pow)
        self.smoke.tw.t = ctx.t
        self.smoke.emit(ctx, light_pos, np.asarray(light_col), light_pow)

    def hammer_sparks(self, ctx, endpoints):
        """Only the two leaders send work to the cut ends, on alternating beats."""
        endpoints = np.asarray(endpoints)
        for k, i in enumerate(LEADERS):
            age = (ctx.t - 10. * k) % 20.
            end = endpoints[k]
            # The supplied centreline point can lie behind the thick cut face.
            # Lift spatter onto its camera-visible side and spread the first
            # shutter sample; ninety coincident points read as no impact.
            toward_camera = ctx.cam.pos-end
            toward_camera /= max(np.linalg.norm(toward_camera), 1e-9)
            impact = end + 1.25*toward_camera
            depth = max(float((impact-ctx.cam.pos) @ ctx.cam.R[2]), 1.)
            projected = (ctx.cam.f_px(1920)/depth)**2
            if age < 9.:
                p0 = impact + self.spark_velocity * (.65+max(age-.25, 0.))
                p1 = impact + self.spark_velocity * (.65+age+.25)
                p0[:, 1] -= .010 * max(age-.25, 0.) ** 2
                p1[:, 1] -= .010 * (age+.25) ** 2
                energy = self.spark_energy * .026*projected * (1.-age/9.) ** 1.5
                ctx.fr.splat(p0, p1, .016, energy, GOLD, ctx.cam0, ctx.cam1, zref=0.)
                if age < 3.:
                    flash = 3.*np.pi*.18**2*projected*math.exp(-age/1.2)
                    ctx.fr.splat(impact[None], impact[None], .18, np.array([flash]),
                                 [1., .90, .58], ctx.cam0, ctx.cam1, profile=1, zref=0.)
                    root = self.towers.top(i, ctx.t)+[0., -.8, 0.]
                    arriving = impact + .085*(root-impact)
                    ctx.fr.splat(arriving[None], impact[None], .025,
                                 np.array([.09*projected*math.exp(-age/1.2)]), GOLD,
                                 ctx.cam0, ctx.cam1, zref=0.)
            # Eight frames of rising hammer spatter connect the giant to its end.
            if age > 12.:
                root = self.towers.top(i, ctx.t) + np.array([0., -.8, 0.])
                u = np.clip((age-12.)/8., 0., 1.)
                q0 = root + (end-root) * max(u-.045, 0.)
                q1 = root + (end-root) * u
                spread = self.spark_velocity[:18] * .65 * (1.-u)
                ctx.fr.splat(q0+spread, q1+spread, .024, np.full(18, 5.), GOLD,
                             ctx.cam0, ctx.cam1)

    def gold(self, ctx, ring_pose, theta_range):
        """Gold falls from extant metal to the two giants' inward windows."""
        import ringsolid as RS
        rot, centre, size = ring_pose
        a, b = theta_range
        p, _ = RS.local_points(a + self.drop_fraction*(b-a), np.zeros(len(self.drop_phase)))
        origin = centre + size * p @ rot.T
        windows = np.array([self.towers.receiving_window(i, ctx.t) for i in LEADERS])
        target = windows[self.drop_leader]
        def positions(t):
            u = (self.drop_phase + t/48.) % 1.
            q = origin + (target-origin)*u[:, None]
            q[:, 1] = origin[:, 1] + (target[:, 1]-origin[:, 1])*u*u
            return q, u
        p0, u0 = positions(ctx.t0)
        p1, u1 = positions(ctx.t1)
        ok = (u1 >= u0) & (u1 > .01) & (u1 < .97)
        depth = np.maximum((p1-ctx.cam.pos) @ ctx.cam.R[2], 1.)
        energy = .018*(ctx.cam.f_px(1920)/depth)**2
        ctx.fr.splat(p0[ok], p1[ok], .055, energy[ok], GOLD, ctx.cam0, ctx.cam1, zref=0.)
