"""C v5: THE TRAP, THE FORGES GO COLD, THE RING UNFINISHED.

Standalone entrypoint, absolute C frames. Existing render.py/C3/A paths are untouched.
Reuses C3's layout, C Eye's medieval forge designs/ashlar, cflame, ringsolid and
c2's storm. Captions, the page burn, audio and EDIT transitions belong to EDIT.
"""
import argparse
from contextlib import contextmanager
import json
import math
import os
from pathlib import Path
import resource
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path[:0] = [str(ROOT / 'lib'), str(HERE)]
_IMPORT_STARTED = time.perf_counter()

import cv2
import numpy as np
import look
import variant
import scene_b as B
import c3
import c_eye as EYE
import c2
import cflame as CF
import ringsolid as RS
from core import Camera, Frame, rng

cv2.setNumThreads(0)
IMPORT_SECONDS = time.perf_counter() - _IMPORT_STARTED
SHOTS = {'trap': (2320, 2640), 'cold': (3840, 4000), 'unfinished': (4000, 4240)}
LOW_FORGE = 8
LEADERS = (4, 5)
OFF = 3848
CLEAN = 4216
RING_C = np.array([0., B.GROUND + 55., 0.])


def ease(a, b, t):
    u = max(0., min(1., (float(t) - a) / (b - a)))
    return u * u * (3. - 2. * u)


def shot_at(t):
    for name, (a, b) in SHOTS.items():
        if a <= t < b:
            return name
    raise ValueError(f'C frame {t} is outside the three v5 ember shots')


def forge_level(i, t):
    if t >= OFF:
        return 0.
    if t >= 3840:
        return forge_level(i, 2639.)
    if i == LOW_FORGE:
        return (1. - .94 * ease(2420, 2438, t)) + 1.24 * ease(2540, 2552, t)
    return 1. + .65 * ease(2480, 2492, t) + (.45 * ease(2580, 2598, t) if i in LEADERS else 0.)


def ring_warmth(t):
    return 1. - ease(4000, 4184, t)


def storm_amount(t):
    return 1. - ease(4016, CLEAN, t)


def gold_amount(t):
    return 1. - ease(4000, 4048, t)


class Schedule(EYE.EyeSched):
    end = 4241.
    tower_rise = 2200.
    beats = []
    pulses = []
    smoke_gain = .7

    def redness(self, t): return .38 * ring_warmth(t)
    def beat_pulse(self, t): return 0.
    def tower_lean(self, t): return 0.
    def tower_post(self, towers, i, P, t):
        # Both rivals incline toward the Ring using the same growth recipe.
        b = towers.base(i)
        q = P.copy()
        s = np.clip((P[:, 1] - b[1]) / max(towers.height(i, t), 1.), 0., 1.)
        k = .14 * ease(2580, 2630, min(t, 2639.)) if i in LEADERS else .015
        inward = -np.array([b[0], 0., b[2]]) / np.linalg.norm(b[[0, 2]])
        q += (k * towers.height(i, t) * s * s)[:, None] * inward
        return q
    def tower_rise_glow(self, t): self._t = t; return .08
    def tower_grow(self, t): return 1.25
    def race(self, t): return .75
    def power(self, t): return float(t < OFF)


SCHED = Schedule()


@contextmanager
def world():
    """Existing modules use process globals; restore them even on a failed render."""
    old = variant.CUT, B.SCHED, B.IGN, B.BEATS
    variant.set_cut('C3')
    B.SCHED, B.IGN, B.BEATS = SCHED, 2200., []
    try:
        yield
    finally:
        variant.CUT, B.SCHED, B.IGN, B.BEATS = old


class Forges(B.Towers):
    def __init__(self):
        super().__init__()
        EYE._medieval(c3.layout_towers(self))
        self.rest = self.h_rise + self.J.sum(axis=1)
        # Abstract rivals: the same existing forge form and starting height,
        # with no separate colour, insignia or architectural identity.
        left, right = LEADERS
        self.G[left] = self.G[right]
        self.design[left] = self.design[right]
        self.rest[list(LEADERS)] = max(self.rest[left], self.rest[right])

    def height(self, i, t):
        t = min(t, 2639.)
        h = self.rest[i]
        if i == LOW_FORGE:
            return h + 7. * ease(2540, 2578, t)
        h += 6. * ease(2480, 2538, t)
        if i in LEADERS:
            h += (55. - h) * ease(2580, 2636, t)
        return h

    def top(self, i, t):
        # Flames, smoke, Ring lights and falling gold use the actual crown,
        # including the same inward displacement applied to its geometry.
        p = super().top(i, t)
        return SCHED.tower_post(self, i, p[None, :], t)[0]

    def _splat(self, ctx, i, pt, colE, a, geo, z, fpx, rw):
        k = forge_level(i, ctx.t)
        # Emission goes out on the same frame everywhere. Cold masonry remains
        # legible under the Ring/sky; it does not retain red self-illumination.
        cold = np.zeros_like(colE)
        if pt is self.cur[i]['parts'][0]:
            V = ctx.cam.pos - pt['P']
            V /= np.maximum(np.linalg.norm(V, axis=1, keepdims=True), 1e-9)
            face = np.clip((V * pt['N']).sum(1), 0., 1.)
            cold = np.array([.012, .013, .014])[None, :] * (.2 + .8 * face)[:, None]
        col = colE * k + cold * (1. - min(k, 1.))
        super()._splat(ctx, i, pt, col, a, geo, z, fpx, rw)


class SmokeFrame:
    def __init__(self, frame, t): self.frame, self.t = frame, t
    def __getattr__(self, n): return getattr(self.frame, n)
    def splat(self, P0, P1, radius, energy, colour, *args, **kw):
        if self.t >= OFF:
            colour = np.asarray(colour)
            neutral = np.sum(colour * np.array([.2126, .7152, .0722]), axis=-1, keepdims=True)
            colour = neutral * np.array([.8, .86, 1.])
        energy = energy * (1. - ease(4060, CLEAN, self.t))
        return self.frame.splat(P0, P1, radius, energy, colour, *args, **kw)


class Scene:
    def __init__(self):
        with world():
            self.towers = Forges()
            self.smoke = EYE._all_smoke(self.towers)
        r = rng(51126)
        self.drop_phase = r.uniform(0, 1, 75)
        self.drop_th = r.uniform(0, 2 * np.pi, 75)
        self.drop_tower = r.integers(0, self.towers.k_all, 75)

    def camera(self, t):
        if t < 3000:
            u = ease(2320, 2640, t)
            r, y, ty, hf = 110. - 5. * u, 38., 44., 54.
            az = c3.AZ0 + .06 + .022 * u
        else:
            # A continuous quiet view across the shot16/17 boundary; ease to a hold.
            u = ease(3840, CLEAN, t)
            r, y, ty, hf = 105. - 18. * u, 38. + 7. * u, 44. + 5. * u, 54.
            az = c3.AZ0 + .082
        pos = np.array([r * math.cos(az), B.GROUND + y, r * math.sin(az)])
        return Camera(pos, [0., B.GROUND + ty, 0.], hfov=hf,
                      focus=np.linalg.norm(RING_C - pos), aperture=.02)

    def ring_frame(self, t):
        # Canonical C3 orientation; freeze the slow spin during the final clean hold.
        clock = 1580. + .12 * ((t - 2320) if t < 3000 else 320. + min(t, CLEAN) - 3840.)
        if c3.OR.enabled():
            reference = 2480. if t < 3000 else 4000.
            ref_clock = 1580. + .12 * ((reference - 2320.) if t < 3000 else 320. + reference - 3840.)
            ref_rot, _, size = c3._closed_ring_frame(ref_clock)
            rot = c3.OR.placement(ref_rot, self.camera(reference).pos, RING_C, reference)
            return rot, RING_C, size
        rot, _, size = c3.ring_frame(clock)
        return rot, RING_C, size

    def ring_layer(self, ctx):
        warm = ring_warmth(ctx.t)
        st = RS.RingState()
        st.letters, st.glow = .48 * warm, .24 * warm
        st.heat = lambda th: np.full_like(th, .31 * warm)
        e = RS.Env(above=(.12, .13, .15), horizon=(.35, .30, .23), below=(.07, .045, .025))
        e.lobe([-.7, .6, .4], [.7, .72, .76], 5.)
        for i in range(self.towers.k_all):
            e.point(self.towers.top(i, ctx.t), np.array([4., 1.7, .5]) * forge_level(i, ctx.t), 2.)
        rot, centre, size = self.ring_frame(ctx.t)
        if c3.OR.enabled():
            st.end_caps = st.leading_glyph = True
            st.heat = c3.OR.heat(ctx.t)
            st.letters, st.glow, st.hammer = .6, .09, .18
            st.write = c3.ring_write(ctx.t)
            rgb, a, d = RS.render(ctx.cam, ctx.fr.W, ctx.fr.H, rot, centre, size, st, e,
                                  th_range=c3.OR.theta_range(ctx.t))
        else:
            rgb, a, d = RS.render(ctx.cam, ctx.fr.W, ctx.fr.H, rot, centre, size, st, e)
            grey = np.sum(rgb * np.array([.2126, .7152, .0722]), axis=2, keepdims=True)
            rgb = rgb * warm + grey * (1. - warm)
        before = RS.merge_occluder(ctx.fr, a, d)
        return rgb * RS.visibility(before, d, ctx.fr.H, ctx.fr.W)[..., None]

    def drops(self, ctx):
        gain = gold_amount(ctx.t)
        if gain <= 0: return
        # Existing C3's gold-bead palette, deterministic ballistic fall from
        # the same canonical band to the forge crowns; never rising fountain arcs.
        def positions(t):
            u = (self.drop_phase + (t - 2320.) / 52.) % 1.
            rot, centre, size = self.ring_frame(t)
            theta = self.drop_th
            if c3.OR.enabled():
                # Each drop keeps its birth origin while the leading edge advances.
                theta = c3.OR.arc_theta(t - 52. * u, self.drop_th / (2. * np.pi))
            p, _ = RS.local_points(theta, np.zeros(len(u)))
            start = centre + size * p @ rot.T
            end = np.array([self.towers.top(int(i), t) for i in self.drop_tower])
            q = start + (end - start) * u[:, None]
            q[:, 1] = start[:, 1] + (end[:, 1] - start[:, 1]) * u * u
            return q, u
        p0, u0 = positions(ctx.t0)
        p1, u1 = positions(ctx.t1)
        ok = (u1 >= u0) & (u1 > .015) & (u1 < .97)
        z = np.maximum((p1 - ctx.cam.pos) @ ctx.cam.R[2], 1.)
        energy = gain * .016 * (ctx.cam.f_px(1920) / z) ** 2
        ctx.fr.splat(p0[ok], p1[ok], .065, energy[ok], [1., .9, .62], ctx.cam0, ctx.cam1, zref=0.)

    def storm(self, ctx):
        H, W = ctx.fr.H, ctx.fr.W
        sky = np.zeros((H, W, 3), np.float32)
        amount = storm_amount(ctx.t)
        if amount <= 0: return sky
        for dz, seed, sigma in [(-2.2, 11., 2.4), (.8, 23., 2.2)]:
            centre = EYE.EYE_C + EYE.N_EYE * (dz * EYE.EYE_R)
            X, Y, ok = c2.plane_coords(ctx.cam, W, H, centre, EYE.EX, EYE.EY, EYE.N_EYE, EYE.EYE_R)
            # Same C9 turbulence, without forming an iris or eye-shaped hole.
            pr = np.array([seed, .42, 1.03, .55, 0., .97, .16, 1.25, 3.6, .02, .012, 0., 1.])
            lp = np.array([1.35, 1.35, 1.25, .3, sigma, .06, -dz, .7, .06, .07, 0.])
            rgb, alpha = np.empty_like(sky), np.empty((H, W), np.float32)
            clock = 1920. + (ctx.t - 2320. if ctx.t < 3000 else 320. + ctx.t - 3840.) * .23
            c2.storm_layer(X, Y, ok, clock, pr, lp, rgb, alpha)
            sky = sky * (1. - alpha[..., None] * amount) + rgb * amount
        if ctx.fr.occ is not None:
            A = ctx.fr.occ[1][0]
            visible = 1. - np.clip(cv2.resize(A, (W, H)), 0., 1.)
            sky *= visible[..., None]
        return sky

    def frame(self, f, scale=.5):
        name = shot_at(f)
        a, b = SHOTS[name]
        # Do not integrate the previous hot state into the shutdown downbeat.
        lo = OFF if f == OFF else a
        ctx = argparse.Namespace(t=float(f), f=f, t0=max(f - .25, lo), t1=min(f + .25, b - .02), scale=scale)
        ctx.cam, ctx.cam0, ctx.cam1 = self.camera(ctx.t), self.camera(ctx.t0), self.camera(ctx.t1)
        ctx.fr = Frame(scale)
        ctx.fr.set(focus=ctx.cam.focus, aperture=ctx.cam.aperture, bokeh_pow=.25, bokeh_cap=1.6,
                   fog_start=90., fog_len=120., near=.3)
        with world():
            self.towers.prepare(ctx)
            ring = self.ring_layer(ctx)
            self.towers.emit(ctx, RING_C, np.array([1., .62, .26]), 70. * ring_warmth(f))
            original = ctx.fr
            ctx.fr = SmokeFrame(original, f)
            self.smoke.emit(ctx, RING_C, np.array([1., .62, .26]), 70. * ring_warmth(f))
            ctx.fr = original
            self.drops(ctx)
            hdr = ctx.fr.resolve() + ring + self.storm(ctx)
            # The already-established gold flame, now visibly owned by each forge.
            if f < OFF:
                for i in np.argsort([(self.towers.top(j, f) - ctx.cam.pos) @ ctx.cam.R[2]
                                    for j in range(self.towers.k_all)])[::-1]:
                    k = forge_level(int(i), f)
                    base = self.towers.top(int(i), f) - np.array([0., .4, 0.])
                    top = base + np.array([0., 4.2 * max(.12, k), 0.])
                    u, v, z = ctx.cam.project(np.stack([base, top]), 1920, 804)
                    vis = c3.occ_vis(ctx.fr, float(z[0] - .5), ctx.fr.H, ctx.fr.W)
                    CF.draw(hdr, (u[0], v[0]), (u[1], v[1]), f + 31. * i, bright=k * .85,
                            calm=.8, vis=vis, scale=scale)
        if not np.isfinite(hdr).all(): raise ValueError(f'Non-finite HDR at C{f}')
        return look.finish(hdr, exposure=1., bloom_strength=.09, bloom_threshold=1.1,
                           streak_strength=0., vignette_amount=.28)


def frames(spec):
    out = []
    for item in spec.split(','):
        span, _, stride = item.partition(':')
        step = int(stride or 1)
        if step < 1: raise ValueError('frame step must be positive')
        if '-' in span:
            a, b = map(int, span.split('-'))
            if b < a: raise ValueError('reversed frame range')
            out.extend(range(a, b + 1, step))
        else: out.append(int(span))
    if len(set(out)) != len(out): raise ValueError('duplicate requested frames')
    for f in out: shot_at(f)
    return out


def output_paths(out, requested, fmt):
    paths = [out / f'f_{f:05d}.{fmt}' for f in requested]
    present = [p.name for p in paths if p.exists()]
    if present:
        raise FileExistsError('Refusing to overwrite existing frames: ' + ', '.join(present[:8]))
    return paths


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('frames', nargs='?', help='inclusive absolute C frame range (alias for --frames)')
    ap.add_argument('--frames', dest='frame_spec', help='rewritable frame range for cloud/farm.py')
    ap.add_argument('--scale', type=float, default=1.)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--format', choices=['png', 'jpg'], default='png')
    args = ap.parse_args()
    if bool(args.frames) == bool(args.frame_spec): ap.error('provide one frame range, positional or --frames')
    requested = frames(args.frame_spec or args.frames)
    if not args.out.is_absolute(): ap.error('--out must be absolute')
    if not 0 < args.scale <= 1: ap.error('--scale must be in (0,1]')
    output_paths(args.out, requested, args.format)
    args.out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    scene = Scene()
    print(json.dumps({'import_seconds': IMPORT_SECONDS, 'initialization_seconds':time.perf_counter() - started,
                      'opencv_threads':cv2.getNumThreads()}), flush=True)
    for f in requested:
        output_paths(args.out, [f], args.format)
        start = time.perf_counter()
        rgb = scene.frame(f, args.scale)
        p = args.out / f'f_{f:05d}.{args.format}'
        if args.format == 'png': look.save_png(str(p), rgb)
        else:
            from PIL import Image
            Image.fromarray(np.rint(np.clip(rgb, 0, 1) * 255).astype(np.uint8)).save(p, quality=95, subsampling=0)
        print(json.dumps({'frame':f, 'seconds':time.perf_counter()-start,
                          'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == 'darwin' else 1024),
                          'output':str(p)}), flush=True)


if __name__ == '__main__': main()
