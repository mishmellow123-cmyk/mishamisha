"""Candidate schedule/routing controls, with no frame rendering or JIT work."""
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(HERE), str(HERE.parents[1] / 'lib')]
import c_v5 as C
import c_v5_trap_candidates as T


class Schedules(unittest.TestCase):
    def setUp(self):
        self.towers = C.Forges.__new__(C.Forges)
        self.towers.rest = np.linspace(26., 39., 18)
        self.towers.rest[list(C.LEADERS)] = 33.

    def test_default_values_are_exactly_accepted_across_all_three_shots(self):
        for a, b in C.SHOTS.values():
            for f in range(a, b):
                for i in range(18):
                    self.assertEqual(T.forge_level(i, f), C.forge_level(i, f))
                    self.assertEqual(T.forge_height(self.towers, i, f), self.towers.height(i, f))

    def test_pilot_is_visible_in_schedule_but_not_a_restored_full_fire(self):
        low = C.LOW_FORGE
        for variant in T.PILOT_VARIANTS:
            self.assertLess(C.forge_level(low, 2479), .1)
            self.assertAlmostEqual(T.forge_level(low, 2479, variant), .30)
            self.assertLess(T.forge_level(low, 2479, variant), T.forge_level(0, 2479, variant))
            down = [T.forge_level(low, f, variant) for f in range(2420, 2439)]
            up = [T.forge_level(low, f, variant) for f in range(2540, 2553)]
            self.assertTrue(np.all(np.diff(down) < 0))
            self.assertTrue(np.all(np.diff(up) > 0))
            self.assertAlmostEqual(up[-1], 1.30)
            for f in (2420., 2429.25, 2438., 2479., 2540., 2546.25, 2552.):
                expected = (1. - .70 * C.ease(2420, 2438, f)) + 1.00 * C.ease(2540, 2552, f)
                self.assertEqual(T.forge_level(low, f, variant), expected)

    def test_only_low_forge_levels_change_and_cold_contract_remains_exact(self):
        times = (2319., 2320., 2419.75, 2420., 2429.25, 2479., 2546.25,
                 2552., 2580., 2639.75, 2640., 3840., 3847.75, 3848., 4216., 4239.)
        for variant in T.PILOT_VARIANTS:
            for f in times:
                for i in range(18):
                    if i != C.LOW_FORGE or not 2420. < f < 2552.:
                        self.assertEqual(T.forge_level(i, f, variant), C.forge_level(i, f))
                if f >= C.OFF:
                    self.assertTrue(all(T.forge_level(i, f, variant) == 0. for i in range(18)))

    def test_gap_is_a_discriminating_change_with_unchanged_end_states(self):
        for i in range(18):
            for f in (2320., 2479.75, 2480., 2538., 2580., 2608.25, 2639., 3848., 4239.):
                self.assertEqual(T.forge_height(self.towers, i, f, 'pilot30_gap'), self.towers.height(i, f))
            for f in np.arange(2480., 2538., .25):
                self.assertEqual(T.forge_height(self.towers, i, f, 'pilot30'), self.towers.height(i, f))
                if i == C.LOW_FORGE:
                    self.assertEqual(T.forge_height(self.towers, i, f, 'pilot30_gap'), self.towers.height(i, f))
            if i != C.LOW_FORGE:
                self.assertGreater(T.forge_height(self.towers, i, 2518., 'pilot30_gap'),
                                   T.forge_height(self.towers, i, 2518., 'pilot30'))
                for f in (2504., 2518., 2537.75):
                    self.assertEqual(T.forge_height(self.towers, i, f, 'pilot30_gap'), self.towers.rest[i] + 6.)

    def test_two_leaders_finish_together_and_surpass_all_others(self):
        for variant in T.VARIANTS:
            for f in np.arange(2580., 2640., .25):
                a, b = C.LEADERS
                self.assertEqual(T.forge_height(self.towers, a, f, variant), T.forge_height(self.towers, b, f, variant))
                self.assertEqual(T.forge_level(a, f, variant), T.forge_level(b, f, variant))
            for i in C.LEADERS:
                self.assertEqual(T.forge_height(self.towers, i, 2639., variant), 55.)
                self.assertGreater(T.forge_height(self.towers, i, 2639., variant),
                                   max(T.forge_height(self.towers, j, 2639., variant)
                                       for j in range(18) if j not in C.LEADERS))


class Routing(unittest.TestCase):
    def setUp(self):
        self.before = C.forge_level, C.Forges.height

    def tearDown(self):
        self.assertIs(C.forge_level, self.before[0])
        self.assertIs(C.Forges.height, self.before[1])

    def test_accepted_default_calls_original_scene_without_substitution(self):
        result = object()
        def construct():
            self.assertIs(C.forge_level, self.before[0])
            self.assertIs(C.Forges.height, self.before[1])
            return SimpleNamespace(frame=base_frame)
        def base_frame(f, scale):
            self.assertEqual((f, scale), (2479, .5))
            self.assertIs(C.forge_level, self.before[0])
            self.assertIs(C.Forges.height, self.before[1])
            return result
        with patch.object(C, 'Scene', side_effect=construct) as constructor:
            self.assertIs(T.render(2479), result)
            constructor.assert_called_once_with()

    def test_candidate_active_for_construction_and_each_reused_frame_only(self):
        events = []
        tw = SimpleNamespace(rest=np.full(18, 32.))
        def capture():
            return C.forge_level(C.LOW_FORGE, 2479), C.Forges.height(tw, 0, 2518)
        def construct():
            events.append(('construct', capture()))
            return SimpleNamespace(frame=frame, marker='original geometry')
        def frame(f, scale):
            events.append((f, capture()))
            return capture()
        with patch.object(C, 'Scene', side_effect=construct):
            scene = T.Scene('pilot30_gap')
            self.assertIs(C.forge_level, self.before[0])
            self.assertEqual(scene.marker, 'original geometry')
            for f in (2479, 2518):
                level, height = scene.frame(f)
                self.assertAlmostEqual(level, .30)
                self.assertEqual(height, 38.)
                self.assertIs(C.forge_level, self.before[0])
                self.assertIs(C.Forges.height, self.before[1])
        self.assertEqual([e[0] for e in events], ['construct', 2479, 2518])
        self.assertNotEqual(events[0][1], (self.before[0](C.LOW_FORGE, 2479), self.before[1](tw, 0, 2518)))

    def test_exception_restores_both_overrides_after_failed_construction_or_frame(self):
        for variant in T.PILOT_VARIANTS:
            with patch.object(C, 'Scene', side_effect=RuntimeError('construction probe')):
                with self.assertRaisesRegex(RuntimeError, 'construction probe'):
                    T.Scene(variant)
            self.assertIs(C.forge_level, self.before[0])
            self.assertIs(C.Forges.height, self.before[1])
            def fail(f, scale):
                self.assertAlmostEqual(C.forge_level(C.LOW_FORGE, 2479), .30)
                raise RuntimeError('frame probe')
            with patch.object(C, 'Scene', return_value=SimpleNamespace(frame=fail)):
                scene = T.Scene(variant)
                with self.assertRaisesRegex(RuntimeError, 'frame probe'):
                    scene.frame(2479)
            self.assertIs(C.forge_level, self.before[0])
            self.assertIs(C.Forges.height, self.before[1])

    def test_context_restores_prior_bindings_not_just_import_time_defaults(self):
        sentinel = lambda i, t: 123.
        with patch.object(C, 'forge_level', sentinel):
            with self.assertRaises(RuntimeError):
                with T.candidate_world('pilot30_gap'):
                    self.assertAlmostEqual(C.forge_level(C.LOW_FORGE, 2479), .30)
                    raise RuntimeError('context probe')
            self.assertIs(C.forge_level, sentinel)

    def test_repeated_variants_do_not_contaminate_later_accepted_frame(self):
        def frame(f, scale):
            return C.forge_level(C.LOW_FORGE, f)
        with patch.object(C, 'Scene', return_value=SimpleNamespace(frame=frame)):
            baseline = T.render(2479)
            for variant in T.PILOT_VARIANTS:
                self.assertNotEqual(T.render(2479, variant=variant), baseline)
                self.assertEqual(T.render(2479), baseline)
                for f in (2399, 2639, 3847, 3848, 4239):
                    self.assertEqual(T.render(f, variant=variant), self.before[0](C.LOW_FORGE, f))

    def test_invalid_variant_fails_before_scene_construction(self):
        with patch.object(C, 'Scene') as constructor:
            with self.assertRaisesRegex(ValueError, 'Unknown Trap variant'):
                T.render(2479, variant='typo')
            constructor.assert_not_called()
        with self.assertRaises(ValueError):
            with T.candidate_world('typo'):
                self.fail('invalid scope entered')


class ForegroundStudies(unittest.TestCase):
    def setUp(self):
        # Build only schedules and coordinates, never tower meshes or a Scene.
        self.tw = C.Forges.__new__(C.Forges)
        self.tw.k, self.tw.k_all = 8, 18
        self.tw.G = [object() for _ in range(18)]
        C.EYE._medieval(C.c3.layout_towers(self.tw))
        self.tw.rest = self.tw.h_rise + self.tw.J.sum(axis=1)
        a, b = C.LEADERS
        self.tw.G[a] = self.tw.G[b]
        self.tw.design[a] = self.tw.design[b]
        self.tw.rest[[a, b]] = max(self.tw.rest[a], self.tw.rest[b])
        self.smoke = SimpleNamespace(H=np.repeat(self.tw.rest[:, None], 8, axis=1), emit=lambda *a: None)
        self.base_scene = SimpleNamespace(towers=self.tw, smoke=self.smoke)
        self.defaults = C.forge_level, C.Forges.height

    def tearDown(self):
        self.assertIs(C.forge_level, self.defaults[0])
        self.assertIs(C.Forges.height, self.defaults[1])

    def test_new_levels_go_dark_and_return_without_changing_any_other_forge(self):
        for variant in T.STAGED_VARIANTS:
            for f in (2438., 2479., 2492., 2539.):
                self.assertEqual(T.forge_level(C.LOW_FORGE, f, variant), 0.)
                self.assertGreater(C.forge_level(C.LOW_FORGE, f), 0.)
            for f in (2399., 2420., 2552., 2639., 3847., 3848., 4239.):
                self.assertEqual(T.forge_level(C.LOW_FORGE, f, variant), C.forge_level(C.LOW_FORGE, f))
            for f in range(2320, 2640):
                for i in range(18):
                    if i != C.LOW_FORGE:
                        self.assertEqual(T.forge_level(i, f, variant), C.forge_level(i, f))

    def test_staging_changes_only_one_forge_and_restores_exact_object_bindings(self):
        before = self.tw.rad, self.tw.ang, self.tw.rot, self.tw.rest, self.base_scene.smoke, self.smoke.H
        forms, designs = self.tw.G.copy(), self.tw.design.copy()
        for variant in T.STAGED_VARIANTS:
            with T.staged_scene(self.base_scene, variant):
                rad, angle, lower = T.STAGING[variant]
                self.assertEqual(self.tw.rad[8], rad)
                self.assertEqual(self.tw.ang[8], C.c3.AZ0 + angle)
                self.assertEqual(self.tw.rest[8], before[3][8] + lower)
                for old, new in zip(before[:4], (self.tw.rad, self.tw.ang, self.tw.rot, self.tw.rest)):
                    np.testing.assert_array_equal(np.delete(new, 8), np.delete(old, 8))
                self.assertIsInstance(self.base_scene.smoke, T.WithdrawalSmoke)
                np.testing.assert_array_equal(self.smoke.H[8], before[5][8] + lower)
                np.testing.assert_array_equal(np.delete(self.smoke.H, 8, axis=0), np.delete(before[5], 8, axis=0))
                self.assertEqual(self.tw.G, forms)
                np.testing.assert_array_equal(self.tw.design, designs)
            after = self.tw.rad, self.tw.ang, self.tw.rot, self.tw.rest, self.base_scene.smoke, self.smoke.H
            for old, new in zip(before, after):
                self.assertIs(new, old)

    def test_projection_makes_initial_flame_largest_but_final_pair_stays_higher(self):
        camera_scene = C.Scene.__new__(C.Scene)
        def tips(f):
            roots = np.array([self.tw.top(i, f) - np.array([0., .4, 0.]) for i in range(18)])
            ends = roots + np.array([[0., 4.2 * max(.12, C.forge_level(i, f)), 0.] for i in range(18)])
            cam = camera_scene.camera(f)
            u, v, z = cam.project(roots, 960, 402)
            tu, tv, _ = cam.project(ends, 960, 402)
            return np.hypot(tu-u, tv-v), tv, z
        for variant in T.STAGED_VARIANTS:
            with T.candidate_world(variant), T.staged_scene(self.base_scene, variant):
                axis, _, depth = tips(2399)
                self.assertGreater(axis[8], np.delete(axis, 8).max())
                self.assertLess(depth[8], np.delete(depth, 8).min())
                _, final_tip, _ = tips(2639)
                self.assertLess(max(final_tip[list(C.LEADERS)]), final_tip[8])
                for leader in C.LEADERS:
                    self.assertEqual(self.tw.height(leader, 2639), 55.)
                    self.assertGreater(self.tw.height(leader, 2639), self.tw.height(8, 2639))

    def test_plume_packets_rise_deterministically_from_the_staged_crown(self):
        with T.staged_scene(self.base_scene, 'front_smoke'):
            b0, p0, radius, density = T.smoke_packets(self.tw, 2479.)
            b1, p1, _, _ = T.smoke_packets(self.tw, 2480.)
            repeated = T.smoke_packets(self.tw, 2479.)
            np.testing.assert_array_equal(repeated[1], p0)
            shared, i0, i1 = np.intersect1d(b0, b1, return_indices=True)
            self.assertGreater(len(shared), 20)
            np.testing.assert_allclose(p1[i1, 1] - p0[i0, 1], .115, atol=1e-12)
            self.assertTrue(np.all(p0[:, 1] > self.tw.top(8, 2479)[1]))
            self.assertTrue(np.isfinite(p0).all())
            self.assertTrue(np.all(radius > 0))
            self.assertTrue(np.any(density > 0))
        for f in (2399., 2420., 2578., 2580., 2639., 3848.):
            self.assertEqual(T.withdrawal_smoke_gain(f), 0.)
        for f in (2438., 2479., 2492., 2539.):
            self.assertEqual(T.withdrawal_smoke_gain(f), 1.)

    def test_plume_uses_actual_smoke_dispatch_and_is_absent_before_paired_rise(self):
        emitted, original = [], []
        self.smoke.emit = lambda *args: original.append(args)
        cam = SimpleNamespace(pos=np.array([0., 0., 100.]), R=np.diag([1., 1., -1.]), f_px=lambda width: 1000.)
        for f in (2399., 2440., 2492., 2578., 2639.):
            ctx = SimpleNamespace(t=f, t0=f-.25, t1=f+.25, cam=cam, cam0=cam, cam1=cam,
                                  fr=SimpleNamespace(splat=lambda *args, **kwargs: emitted.append((args, kwargs))))
            emitted.clear()
            with T.staged_scene(self.base_scene, 'front_smoke'):
                self.base_scene.smoke.emit(ctx, None, None, None)
            self.assertEqual(len(emitted), int(f in (2440., 2492.)))
            if emitted:
                args, kwargs = emitted[0]
                self.assertEqual(args[0].shape, args[1].shape)
                self.assertTrue(np.all(args[1][:, 1] > args[0][:, 1]))
                self.assertTrue(np.all(args[3] > 0.))
                self.assertEqual(kwargs['profile'], 1)
        self.assertEqual(len(original), 5)

    def test_failed_frame_restores_staging_smoke_and_globals_then_non_trap_is_accepted(self):
        before = self.tw.rad, self.tw.ang, self.tw.rot, self.tw.rest, self.base_scene.smoke, self.smoke.H
        for variant in T.STAGED_VARIANTS:
            def fail(f, scale):
                self.assertIsInstance(self.base_scene.smoke, T.WithdrawalSmoke)
                self.assertEqual(C.forge_level(8, f), 0.)
                raise RuntimeError('staged frame probe')
            self.base_scene.frame = fail
            with patch.object(C, 'Scene', return_value=self.base_scene):
                scene = T.Scene(variant)
                with self.assertRaisesRegex(RuntimeError, 'staged frame probe'):
                    scene.frame(2479)
                after = self.tw.rad, self.tw.ang, self.tw.rot, self.tw.rest, self.base_scene.smoke, self.smoke.H
                for old, new in zip(before, after):
                    self.assertIs(new, old)
                self.base_scene.frame = lambda f, scale: (self.tw.rad, self.tw.rest, self.base_scene.smoke, C.forge_level(8, f))
                for f in (3840., 3848., 4239.):
                    rad, rest, smoke, level = scene.frame(f)
                    self.assertIs(rad, before[0]); self.assertIs(rest, before[3]); self.assertIs(smoke, before[4])
                    self.assertEqual(level, self.defaults[0](8, f))


if __name__ == '__main__':
    unittest.main()
