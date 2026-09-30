"""In-step contracts and deliberately broken controls; no images, geometry or JIT.

Each contract below is exercised against the candidate, then against a mutation
that reintroduces the relevant failure. A green mutant is therefore a test failure.
The frame checks execute the real compositor with tiny substitute render sinks.
"""
from contextlib import contextmanager
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import c_v5_instep as C


class InStep(unittest.TestCase):
    def scene(self, variant='in-step'):
        with patch.object(C.base.Scene, '__init__', return_value=None):
            return C.Scene(variant)

    def rejects(self, oracle, mutant):
        """Actually execute a broken implementation and require oracle failure."""
        with self.assertRaises(AssertionError):
            oracle(mutant)

    def test_exact_accepted_delegation_and_prebeacon_arithmetic(self):
        original = C.base.SHOTS
        calls = []
        sentinel = object()

        def original_frame(scene, f, scale):
            calls.append((f, scale, C.base.SHOTS['cold']))
            return sentinel

        def oracle(scene):
            calls.clear()
            for f in (3816, 3840, 3847):
                self.assertIs(scene.frame(f, .5), sentinel)
            self.assertEqual(calls, [(3816, .5, (3816, 4000)),
                                    (3840, .5, (3840, 4000)),
                                    (3847, .5, (3840, 4000))])
            self.assertIs(C.base.SHOTS, original)

        with patch.object(C.base.Scene, 'frame', original_frame):
            oracle(self.scene())
            oracle(self.scene('accepted'))
            oracle(self.scene(C.Scene.__init__.__defaults__[0]))
            broken = SimpleNamespace(frame=lambda f, scale: object())
            self.rejects(oracle, broken)
            accepted = self.scene('accepted')
            for f in (3848, 3900, 4120, 4239):
                self.assertIs(accepted.frame(f, .5), sentinel)
                self.assertEqual(calls[-1], (f, .5, (3840, 4000)))
        self.assertIs(C.base.SHOTS, original)

    def test_unknown_variant_rejected_before_any_geometry(self):
        def oracle(constructor):
            with patch.object(C.base.Scene, '__init__') as build:
                with self.assertRaises(ValueError):
                    constructor('accidentally-enabled')
                build.assert_not_called()
        oracle(C.Scene)
        self.rejects(oracle, lambda variant: None)

    def test_every_forge_converges_smoothly_from_its_own_race_level(self):
        def oracle(level):
            old = np.array([C._RACE_LEVEL(i, 3847.) for i in range(28)])
            self.assertGreater(len(np.unique(old)), 1)
            for i in range(28):
                for f in (3816, 3840, 3847):
                    self.assertEqual(level(i, f), C._RACE_LEVEL(i, f))
                self.assertEqual(level(i, 3848), old[i])
                self.assertAlmostEqual(level(i, 3860), (old[i] + .85) / 2)
                self.assertEqual(level(i, 3872), .85)
                values = np.array([level(i, t) for t in np.arange(3848, 3872.01, .25)])
                self.assertTrue(np.all(values > 0.))
                self.assertTrue(np.all(np.diff(values) <= 0.))
                # The ease has zero slope at both ends, unlike a linear fade.
                self.assertLess(abs(level(i, 3848.001) - old[i]), 1e-7)
                self.assertLess(abs(level(i, 3871.999) - .85), 1e-7)
                for f in (3900, 4000, 4216, 4239):
                    self.assertEqual(level(i, f), .85)
        oracle(C.forge_level)
        self.rejects(oracle, C._RACE_LEVEL)  # Original shutdown.
        self.rejects(oracle, lambda i, t: C.forge_level(0, t))  # Loses leaders' own level.
        self.rejects(oracle, lambda i, t: C._RACE_LEVEL(i, t) if t < 3848 else .85)

    def test_map_grid_and_full_beat_breath_use_the_score_phase(self):
        data = json.loads(C.BARMAP.read_text())
        spacing = data['fps'] * 60. / data['bpm']

        def grid_oracle(reader):
            beats, step = reader()
            self.assertEqual(step, spacing)
            np.testing.assert_array_equal(beats, np.arange(0., data['frames'], spacing))
        grid_oracle(C.beat_frames)
        self.rejects(grid_oracle, lambda: (np.arange(1., data['frames'], spacing), spacing))
        self.rejects(grid_oracle, lambda: (np.arange(0., data['frames'], 24.), 24.))

        def pulse_oracle(pulse):
            for t in (3816, 3847, 3848, 3864, 3872):
                self.assertEqual(pulse(t), 0.)
            # Whole cycles after the onset ramp: 3920 and 3940 are score beats.
            self.assertAlmostEqual(pulse(3920), .10)
            self.assertAlmostEqual(pulse(3920 + spacing / 4), .05)
            self.assertAlmostEqual(pulse(3920 + spacing / 2), 0.)
            self.assertAlmostEqual(pulse(3920 + spacing), .10)
            values = np.array([pulse(t) for t in np.arange(3872., 4240., .25)])
            self.assertTrue(np.all((values >= 0.) & (values <= .10)))
            self.assertLess(np.max(abs(np.diff(values))), .005)   # the raised cosine at .10: max step .10*pi/20*.25 = .0039
            self.assertLess(abs(pulse(3920 - .001) - pulse(3920 + .001)), 1e-10)
            self.assertLess(pulse(3872.001), 1e-8)
        schedule = C.Schedule()
        pulse_oracle(schedule.beat_pulse)
        self.rejects(pulse_oracle, lambda t: 0.)
        self.rejects(pulse_oracle, lambda t: schedule.beat_pulse(t + 2.))
        self.rejects(pulse_oracle, lambda t: .10 if t % spacing < 1 else 0.)
        self.rejects(pulse_oracle, lambda t: 8. * schedule.beat_pulse(t))

    def test_map_identity_and_positive_spacing_are_validated(self):
        data = json.loads(C.BARMAP.read_text())

        def oracle(reader):
            for key, wrong in (('cut', 'A'), ('fps', 30), ('frames', 5921),
                               ('bpm', -72), ('bpm', 0), ('bpm', float('inf'))):
                source = SimpleNamespace(read_text=lambda: json.dumps({**data, key: wrong}))
                with self.assertRaises(ValueError):
                    reader(source)
        oracle(C.beat_frames)
        self.rejects(oracle, lambda source: (tuple(range(296)), 20.))

    def test_power_stays_on_and_flame_motion_slows_without_a_clock_jump(self):
        def power_oracle(power):
            for t in (3816, 3847, 3848, 3872, 4000, 4216, 4239):
                self.assertEqual(power(t), 1.)
        power_oracle(C.Schedule().power)
        self.rejects(power_oracle, C.base.Schedule().power)

        def clock_oracle(clock):
            self.assertEqual(clock(3848), 3848.)
            self.assertAlmostEqual(clock(3848.001) - clock(3848), .001, places=8)
            self.assertAlmostEqual(clock(3872.001) - clock(3872), .00025, places=8)
            self.assertEqual(clock(4217) - clock(4216), .25)
            self.assertTrue(np.all(np.diff([clock(t) for t in range(3848, 4240)]) > 0))
        clock_oracle(C.flame_clock)
        self.rejects(clock_oracle, lambda t: t)
        self.rejects(clock_oracle, lambda t: .25 * t)

    def test_scope_synchronizes_real_hook_and_restores_on_success_and_failure(self):
        schedule = C.Schedule()
        old = (C.base.SCHED, C.base.forge_level, C.base.ring_warmth,
               C.base.B.SCHED, C.base.B.IGN, C.base.B.BEATS, C.base.variant.CUT)
        towers = SimpleNamespace(dly=np.array([.3, 1.4, 2.1]))
        original_delay = towers.dly

        def oracle(scope):
            for fail in (False, True):
                try:
                    with scope(schedule, towers):
                        self.assertIs(C.base.SCHED, schedule)
                        self.assertIs(C.base.B.SCHED, schedule)
                        self.assertIs(C.base.forge_level, C.forge_level)
                        self.assertIs(C.base.ring_warmth, C.ring_warmth)
                        np.testing.assert_array_equal(towers.dly, [0., 0., 0.])
                        self.assertEqual(C.base.B.BEATS, [])
                        pulses = [C.base.B.beat_pulse(3920 - delay) for delay in towers.dly]
                        np.testing.assert_array_equal(pulses, [.10] * 3)
                        if fail:
                            raise RuntimeError('render failed')
                except RuntimeError as error:
                    self.assertEqual(str(error), 'render failed')
                self.assertIs(towers.dly, original_delay)
                restored = (C.base.SCHED, C.base.forge_level, C.base.ring_warmth,
                            C.base.B.SCHED, C.base.B.IGN, C.base.B.BEATS, C.base.variant.CUT)
                for before, after in zip(old, restored):
                    self.assertIs(before, after)
                self.assertFalse(C._LOCK.locked())
        oracle(C.working_world)

        @contextmanager
        def unsynchronized(schedule, towers):
            with C.working_world(schedule, towers):
                towers.dly = original_delay
                yield
        self.rejects(oracle, unsynchronized)
        self.assertIs(towers.dly, original_delay)
        self.assertFalse(C._LOCK.locked())

        @contextmanager
        def leaked_delay(schedule, towers):
            with C.working_world(schedule, towers):
                yield
            towers.dly = np.zeros_like(original_delay)
        try:
            self.rejects(oracle, leaked_delay)
        finally:
            towers.dly = original_delay

    def test_chimney_smoke_keeps_warm_colour_and_faint_emission_through_end(self):
        colour = np.array([[1., .4, .1]])
        energy = np.array([2.])

        def oracle(proxy):
            received = []
            frame = SimpleNamespace(splat=lambda *args, **kwargs: received.append(args))
            for t, gain in ((3848, 1.), (3860, .56), (3872, .12), (4216, .12), (4239, .12)):
                proxy(frame, t).splat(None, None, .5, energy, colour)
                self.assertAlmostEqual(received[-1][3][0], 2. * gain)
                np.testing.assert_array_equal(received[-1][4], colour)
        oracle(C.WorkingSmokeFrame)
        self.rejects(oracle, C.base.SmokeFrame)
        self.rejects(oracle, lambda frame, t: C.WorkingSmokeFrame(frame, t, gain=0.))
        self.rejects(oracle, lambda frame, t: C.WorkingSmokeFrame(frame, t, gain=1.))

    def test_ring_settles_to_gold_and_freezes_its_geometry(self):
        def warmth_oracle(warmth):
            self.assertEqual(warmth(3848), 1.)
            self.assertEqual(warmth(4000), 1.)
            self.assertAlmostEqual(warmth(4012), .9)
            for t in (4024, 4120, 4184, 4216, 4239):
                self.assertAlmostEqual(warmth(t), .8)
        warmth_oracle(C.ring_warmth)
        self.rejects(warmth_oracle, C._RING_WARMTH)
        self.rejects(warmth_oracle, lambda t: 0.)

        scene = self.scene()
        def geometry_oracle(ring_frame):
            for t in (3816, 3847, 3900, 3999):
                expected = C.base.Scene.ring_frame(scene, t)
                for actual, wanted in zip(ring_frame(t), expected):
                    np.testing.assert_array_equal(actual, wanted)
            fixed = ring_frame(4000.)
            for t in (4001, 4120, 4216, 4239):
                for actual, wanted in zip(ring_frame(t), fixed):
                    np.testing.assert_array_equal(actual, wanted)
        geometry_oracle(scene.ring_frame)
        self.rejects(geometry_oracle, lambda t: C.base.Scene.ring_frame(scene, t))

        # Exercise the inherited Ring compositor under the substituted world.
        scene.towers = SimpleNamespace(k_all=0, dly=np.array([]))
        states = []
        def render(cam, w, h, rot, centre, size, state, env):
            states.append(state)
            return np.ones((2, 2, 3)) * [1., .5, .1], np.ones((2, 2)), np.ones((2, 2))
        ctx = SimpleNamespace(t=4216., fr=SimpleNamespace(W=2, H=2), cam=scene.camera(4216.))
        def gold_oracle(scope):
            with scope(scene.schedule, scene.towers):
                rgb = scene.ring_layer(ctx)
            self.assertTrue(np.all(rgb[..., 0] > rgb[..., 1]))
            self.assertTrue(np.all(rgb[..., 1] > rgb[..., 2]))
            self.assertAlmostEqual(states[-1].glow, .24 * .8)
            np.testing.assert_allclose(states[-1].heat(np.array([0., 1.])), .31 * .8)
        with patch.object(C.base.RS, 'render', render), \
             patch.object(C.base.RS, 'merge_occluder', return_value=None), \
             patch.object(C.base.RS, 'visibility', return_value=np.ones((2, 2))):
            gold_oracle(C.working_world)
            @contextmanager
            def old_cooling(schedule, towers):
                with C.working_world(schedule, towers):
                    with patch.object(C.base, 'ring_warmth', C._RING_WARMTH):
                        yield
            self.rejects(gold_oracle, old_cooling)

    def test_storm_uses_original_thinning_and_is_absent_at_4216(self):
        def oracle(storm):
            self.assertIs(storm, C.base.Scene.storm)
            self.assertEqual(C.base.storm_amount(4000), 1.)
            self.assertGreater(C.base.storm_amount(4120), 0.)
            self.assertEqual(C.base.storm_amount(4216), 0.)
            scene = self.scene()
            ctx = SimpleNamespace(t=4216., fr=SimpleNamespace(H=2, W=3))
            np.testing.assert_array_equal(storm(scene, ctx), np.zeros((2, 3, 3)))
        oracle(C.Scene.storm)
        self.rejects(oracle, lambda scene, ctx: np.ones((2, 3, 3)))

    def towers_with_windows(self, count=5):
        # Tiny actual WIN-format samples; no construction of renderer geometry.
        geometry = []
        for i in range(count):
            points = np.array([[i, C.base.B.GROUND + 12., 0.],
                               [i, C.base.B.GROUND + 12., 2.],
                               [i, C.base.B.GROUND + 9., 1.]])
            windows = {'key': np.array([10 + i, 10 + i, 30 + i]),
                       'p': points, 'n': np.tile([0., 0., 1.], (3, 1))}
            geometry.append([None, None, None, windows])
        return SimpleNamespace(k_all=count, G=geometry, height=lambda i, t: 30.,
                               world=lambda i, p, h: p.copy(), nworld=lambda i, n: n.copy())

    def test_recipients_are_real_windows_for_every_tower_and_no_crown_shortcut(self):
        towers = self.towers_with_windows()
        camera = SimpleNamespace(pos=np.array([0., C.base.B.GROUND + 25., 20.]))
        schedule = SimpleNamespace(tower_post=lambda tw, i, p, t: p + [2., 0., 0.])

        def oracle(targeter):
            points, keys = targeter(towers, schedule, camera, 4000.)
            self.assertEqual(points.shape, (5, 3))
            self.assertEqual(keys, tuple(range(10, 15)))
            for i, key in enumerate(keys):
                group = towers.G[i][3]
                selected = group['key'] == key
                centroid = group['p'][selected].mean(axis=0)
                np.testing.assert_array_equal(points[i], centroid + [2., 0., .12])
        oracle(C.window_targets)
        self.rejects(oracle, lambda *args: (np.tile([0., 35., 0.], (5, 1)), tuple(range(10, 15))))
        self.rejects(oracle, lambda *args: (np.zeros((2, 3)), (10, 11)))
        towers.G[2][3]['p'][:, 1] = C.base.B.GROUND
        with self.assertRaisesRegex(ValueError, 'tower 2'):
            C.window_targets(towers, schedule, camera, 4000.)

    def test_window_ceiling_includes_normal_offset_and_prefers_ring_facing_wall(self):
        towers = self.towers_with_windows(count=1)
        schedule = SimpleNamespace(tower_post=lambda tw, i, p, t: p)
        camera = SimpleNamespace(pos=np.array([40., C.base.B.GROUND + 25., 0.]))
        group = {'key': np.array([10, 30]),
                 'p': np.array([[20., C.base.B.GROUND + 12., 0.],
                                [20., C.base.B.GROUND + 9., 0.]]),
                 'n': np.array([[1., 0., 0.], [-1., 0., 0.]])}
        towers.G[0][3] = group
        def facing_oracle(targeter):
            points, keys = targeter(towers, schedule, camera, 4000.)
            self.assertEqual(keys, (30,))
            np.testing.assert_array_equal(points[0], group['p'][1] + [-.12, 0., 0.])
        facing_oracle(C.window_targets)
        self.rejects(facing_oracle, lambda *a: (group['p'][:1] + [.12, 0., 0.], (10,)))

        # An upward-facing centroid below the ceiling is rejected when its
        # actual normal-offset recipient would be above the launch ceiling.
        group['p'][0, 1] = C.base.B.GROUND + 10.96
        group['n'][0] = [0., 1., 0.]
        def ceiling_oracle(targeter):
            points, keys = targeter(towers, schedule, camera, 4000.,
                                   ceiling=C.base.B.GROUND + 11.)
            self.assertEqual(keys, (30,))
            self.assertLess(points[0, 1], C.base.B.GROUND + 11.)
        ceiling_oracle(C.window_targets)
        self.rejects(ceiling_oracle, lambda *a, **kw: (group['p'][:1] + [0., .12, 0.], (10,)))

        scene = self.scene()
        scene.towers = SimpleNamespace(k_all=5)
        def launch_oracle(initialize):
            with patch.object(C, 'window_targets', return_value=(np.zeros((5, 3)), tuple(range(5)))) as targeter:
                initialize()
            self.assertEqual(targeter.call_count, 1)
            rot, centre, size = scene.ring_frame(4000.)
            rim, _ = C.base.RS.local_points(scene.shared_theta, np.zeros(10))
            minimum_launch = (centre + size * rim @ rot.T)[:, 1].min()
            self.assertAlmostEqual(targeter.call_args.kwargs.get('ceiling', np.inf), minimum_launch - .25)
        launch_oracle(scene._shared_drops)
        def missing_ceiling():
            C.window_targets(scene.towers, scene.schedule, scene.camera(4000.), 4000.)
        self.rejects(launch_oracle, missing_ceiling)

    def drop_scene(self):
        scene = self.scene()
        scene.towers = SimpleNamespace(k_all=5)
        targets = np.array([[i * 2., C.base.B.GROUND + 10. + i, 0.] for i in range(5)])
        with patch.object(C, 'window_targets', return_value=(targets, tuple(range(5)))):
            scene._shared_drops()
        return scene

    def test_shared_gold_is_even_slower_descending_and_reaches_every_window(self):
        scene = self.drop_scene()
        def allocation_oracle(tower_ids):
            np.testing.assert_array_equal(np.bincount(tower_ids, minlength=5), [2] * 5)
        allocation_oracle(scene.shared_tower)
        self.rejects(allocation_oracle, np.zeros(10, dtype=int))
        self.rejects(allocation_oracle, np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 0]))
        np.testing.assert_allclose(np.diff(scene.shared_phase), .1)

        def motion_oracle(positions):
            p0, u0 = positions(4100.)
            p1, u1 = positions(4101.)
            no_wrap = u1 >= u0
            np.testing.assert_allclose((u1 - u0)[no_wrap], 1. / 104.)
            self.assertTrue(np.all(p1[no_wrap, 1] < p0[no_wrap, 1]))
            cycle, phase = positions(4204.)
            np.testing.assert_allclose(cycle, p0, atol=1e-10)
            np.testing.assert_allclose(phase, u0, atol=1e-12)
            # Intercept phase immediately before each wrap to observe endpoints.
            for index, initial in enumerate(scene.shared_phase):
                t = 4000. + (1. - initial - 1e-8) * 104.
                points, phase = positions(t)
                np.testing.assert_allclose(points[index], scene.shared_targets[scene.shared_tower[index]], atol=1e-5)
        motion_oracle(scene.shared_positions)
        with patch.object(C, 'DROP_PERIOD', 52.):
            self.rejects(motion_oracle, scene.shared_positions)
        def rising(t):
            points, phase = scene.shared_positions(t)
            points[:, 1] *= -1
            return points, phase
        self.rejects(motion_oracle, rising)

    def test_real_drop_dispatch_continues_after_old_cutoff_and_pools_at_each_window(self):
        scene = self.drop_scene()
        camera = SimpleNamespace(pos=np.array([0., 0., -80.]), R=np.eye(3), f_px=lambda w: 100.)
        received = []
        frame = SimpleNamespace(splat=lambda *a, **kw: received.append((a, kw)))
        ctx = SimpleNamespace(t=4216., t0=4215.75, t1=4216.25, fr=frame,
                              cam=camera, cam0=camera, cam1=camera)

        def oracle(dispatch):
            received.clear()
            dispatch(ctx)
            self.assertEqual(len(received), 2)
            falling, pool = received
            self.assertGreater(len(falling[0][0]), 0)
            self.assertTrue(np.all(falling[0][3] > 0.))
            self.assertTrue(np.all(falling[0][1][:, 1] < falling[0][0][:, 1]))
            np.testing.assert_array_equal(pool[0][0], scene.shared_targets)
            np.testing.assert_array_equal(pool[0][1], scene.shared_targets)
            self.assertTrue(np.all(pool[0][3] > 0.))
            np.testing.assert_array_equal(pool[0][4], [1., .72, .30])
            self.assertIs(ctx.fr, frame)
        oracle(scene.drops)
        self.rejects(oracle, lambda ctx: C.base.Scene.drops(scene, ctx))
        with patch.object(C, 'DROP_GAIN', 0.):
            self.rejects(oracle, scene.drops)

        original_calls = []
        with patch.object(C.base.Scene, 'drops', side_effect=lambda ctx: original_calls.append(ctx.fr)):
            for variant, t in (('accepted', 4120.), ('in-step', 3999.), ('in-step', 4000.)):
                scene.variant, ctx.t = variant, t
                received.clear()
                scene.drops(ctx)
                self.assertEqual(received, [])
                self.assertIs(ctx.fr, frame)
        self.assertEqual(len(original_calls), 3)

    def test_drop_handoff_crossfades_and_restores_frame_on_original_drop_failure(self):
        scene = self.drop_scene()
        camera = SimpleNamespace(pos=np.array([0., 0., -80.]), R=np.eye(3), f_px=lambda w: 100.)
        received = []
        frame = SimpleNamespace(splat=lambda *a, **kw: received.append(a))
        ctx = SimpleNamespace(t=4012., t0=4011.75, t1=4012.25, fr=frame,
                              cam=camera, cam0=camera, cam1=camera)
        def inherited(_scene, ctx):
            ctx.fr.splat(None, None, .065, np.array([2.]), [1., .9, .62])
        def oracle(dispatch):
            received.clear()
            dispatch(ctx)
            self.assertEqual(len(received), 3)
            np.testing.assert_array_equal(received[0][3], [1.])
            self.assertTrue(np.all(received[1][3] > 0.))
            self.assertTrue(np.all(received[2][3] > 0.))
            self.assertIs(ctx.fr, frame)
        with patch.object(C.base.Scene, 'drops', inherited):
            oracle(scene.drops)
            with patch.object(C, 'WorkingSmokeFrame', side_effect=lambda frame, *a, **kw: frame):
                self.rejects(oracle, scene.drops)
        def restore_oracle(dispatch):
            try:
                with self.assertRaisesRegex(RuntimeError, 'original drop failure'):
                    dispatch(ctx)
                self.assertIs(ctx.fr, frame)
            finally:
                ctx.fr = frame
        with patch.object(C.base.Scene, 'drops', side_effect=RuntimeError('original drop failure')):
            restore_oracle(scene.drops)
            def leaked(ctx):
                ctx.fr = C.WorkingSmokeFrame(ctx.fr, ctx.t, gain=.5)
                raise RuntimeError('original drop failure')
            self.rejects(restore_oracle, leaked)

    def test_actual_frame_keeps_all_flames_and_the_shared_breath_in_dispatch(self):
        scene = self.scene()
        original_delay = np.array([.3, 1.1, 2.2, 3.3])
        seen = []
        def emit(ctx, *args):
            seen.append((C.base.B.SCHED.power(ctx.t), scene.towers.dly.copy()))
        scene.towers = SimpleNamespace(k_all=4, dly=original_delay, prepare=lambda ctx: None,
            top=lambda i, t: np.array([i * 4., 20., 0.]), emit=emit)
        scene.smoke = SimpleNamespace(emit=lambda *args: None)
        scene.ring_layer = lambda ctx: np.zeros((4, 8, 3))
        scene.storm = lambda ctx: np.zeros((4, 8, 3))
        scene.drops = lambda ctx: None
        class Frame:
            W, H, occ = 8, 4, None
            def __init__(self, scale): pass
            def set(self, **kwargs): pass
            def resolve(self): return np.zeros((4, 8, 3))

        def oracle(dispatch):
            for t in (3848, 3864, 3872, 3920, 4120, 4216, 4239):
                with patch.object(C.base.CF, 'draw') as draw:
                    dispatch(t, .5)
                self.assertEqual(draw.call_count, 4)
                levels = sorted(call.kwargs['bright'] for call in draw.call_args_list)
                self.assertGreater(min(levels), 0.)
                if t >= 3872:
                    expected = .85 * .85 * (1. + .9 * scene.schedule.beat_pulse(t))
                    np.testing.assert_allclose(levels, [expected] * 4)
                for call in draw.call_args_list:
                    self.assertEqual(call.kwargs['scale'], .5)
                self.assertEqual(seen[-1][0], 1.)
                np.testing.assert_array_equal(seen[-1][1], np.zeros(4))
                self.assertIs(scene.towers.dly, original_delay)
        with patch.object(C.base, 'Frame', Frame), \
             patch.object(C.base.c3, 'occ_vis', return_value=1.), \
             patch.object(C.base.look, 'finish', side_effect=lambda hdr, **kwargs: hdr):
            oracle(scene.frame)
            self.rejects(oracle, lambda f, scale: C.base.Scene.frame(scene, f, scale))
            with patch.object(C, 'forge_level', return_value=0.):
                self.rejects(oracle, scene.frame)
        self.assertIs(scene.towers.dly, original_delay)
        self.assertFalse(C._LOCK.locked())


if __name__ == '__main__':
    unittest.main()
