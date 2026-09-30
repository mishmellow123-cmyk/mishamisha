"""Opt-in glow variants: musical, spatial and accepted-route regressions.

Real sky kernels run on tiny tiles; the accepted-route comparison substitutes
terrain. This suite does not establish full native-frame equality or look-dev.
"""
import hashlib
from contextlib import ExitStack
import math
from pathlib import Path
import sys
import unittest
from unittest import mock

import numpy as np

RUN = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RUN))
import falsedawn as FD
import glowvars as GV
import glowvars_sites as GS


# Measured from git HEAD:./shots/run/falsedawn.py before this opt-in addition.
ACCEPTED_SOURCE_SHA256 = 'b95c8e42e41aeb87cf333d49b09d5d40e3d3afc784f51666da6c4d48c92f5d21'
D_RANGES = {'brink': (4080, 4239), 'twofires': (4560, 5039),
            'watch': (7040, 7359), 'truedawn': (7360, 7839)}
CROWN_BASES = np.array([[733.4482365330008, 463.0645838537898],
                        [1386.8711906600877, 455.13424081989217]])


def assert_same_bytes(actual, expected):
    assert actual.dtype == expected.dtype
    assert actual.shape == expected.shape
    assert actual.tobytes() == expected.tobytes()


def assert_musical_peaks(clock, offset):
    # 24 fps * 60 seconds / 72 quarter notes = 20 frames per quarter note.
    # D's ridge shots keep the racing quarter-note pulse throughout each shot.
    for variant in ('brink', 'twofires'):
        start, end = D_RANGES[variant]
        expected = [f for f in range(start, end + 1) if (f + offset) % 20 == 0]
        actual = [f for f in range(start, end + 1)
                  if math.isclose(clock(f, variant, offset), 1., abs_tol=1e-12)]
        assert actual == expected
        for frame in expected:
            assert clock(frame + 1, variant, offset) < clock(frame, variant, offset)


def assert_racing_does_not_settle(state):
    opening = np.array([[state(f, 'twofires')[key] for key in ('glow', 'underglow')]
                        for f in range(4560, 4580)])
    whole_shot = np.array([[state(f, 'twofires')[key] for key in ('glow', 'underglow')]
                           for f in range(4560, 5040)])
    assert np.all(np.ptp(opening, axis=0) > 0.)
    np.testing.assert_allclose(whole_shot, np.tile(opening, (24, 1)), rtol=0., atol=1e-14)


def assert_lighting_bridge(watch, dawn):
    watch_lights, watch_sky, _, _ = watch
    dawn_lights, dawn_sky, _, dawn_state = dawn
    for before, after in zip(watch_lights, dawn_lights):
        np.testing.assert_array_equal(after, before)
    # Only the scalar glow breath advances on the intervening frame.
    np.testing.assert_array_equal(np.delete(dawn_sky, 2), np.delete(watch_sky, 2))
    assert dawn_state['dawn'] == 0.


def assert_nominal_flame_height(call, source_scale, expected):
    camera, pos, physical_height = call.args[2:5]
    depth = camera.project(pos)[2]
    apparent_height = camera.f * physical_height / depth / source_scale
    np.testing.assert_allclose(apparent_height, expected, rtol=0., atol=1e-10)


def assert_crown_projection(points, camera):
    x, y, depth = camera.project(GS.curved(points[:2], camera))
    assert np.all(depth > 0.)
    np.testing.assert_allclose(np.column_stack([x, y]), CROWN_BASES, atol=1e-6, rtol=0.)


def assert_approaching_ignitions(fires, camera_pos, offset):
    ignition = fires[:, 3]
    assert ignition[0] == ignition[1]
    assert ignition[2] > ignition[1]
    np.testing.assert_array_equal(np.diff(ignition[2:]), 40.)
    np.testing.assert_array_equal((ignition[2:] + offset) % 40, 0.)
    physical_distance = np.linalg.norm(fires[:, :3] - camera_pos, axis=1)
    ordered = np.r_[min(physical_distance[:2]), physical_distance[2:]]
    assert np.all(np.diff(ordered) < 0)


def assert_curved_fire_rows(actual, physical, camera_pos):
    expected = physical.copy()
    squared_horizontal = np.sum((physical[:, [0, 2]] - camera_pos[[0, 2]]) ** 2, axis=1)
    expected[:, 1] -= squared_horizontal / (2 * GV.WD.R_EARTH)
    np.testing.assert_array_equal(actual, expected)


def assert_caught_fires_only(fires, frame):
    assert np.all(fires[:, 3] <= frame)


def assert_ignition_cache_isolation(cache, row):
    changed_row = (*row[:3], 5040., *row[4:])
    first_smoke = cache(row)
    changed_smoke = cache(changed_row)
    assert changed_smoke is not first_smoke
    assert cache(row) is first_smoke
    assert cache(changed_row) is changed_smoke


def assert_pair_effect_cache_isolation(cache):
    first = cache(810, 4320)
    other_seed = cache(811, 4320)
    other_ignition = cache(810, 4400)
    for index in (0, 1):
        assert first[index] is not other_seed[index]
        assert first[index] is not other_ignition[index]
    assert cache(810, 4320) is first
    assert cache(810, 4400) is other_ignition


def assert_basket_opening(image, original):
    np.testing.assert_array_equal(image[22, 18], original[22, 18])
    assert np.any(image[22, 17] != original[22, 17])
    assert np.any(image[19, 20] != original[19, 20])


def assert_separated_pair_tongues(table, canonical):
    assert table.shape == (7, 6)
    assert np.isfinite(table).all()
    assert np.all(np.diff(table[:, 0]) > 0.)
    assert np.ptp(table[:, 0]) > np.ptp(canonical[:, 0])
    assert np.all((table[:, 2] > 0.) & (table[:, 2] < canonical[:, 2]))
    assert np.all(table[[0, 6], 1] > canonical[[0, 6], 1])
    for peak in (1, 3, 5):
        assert table[peak, 1] > max(table[peak - 1, 1], table[peak + 1, 1])
    # Changes to the outline must retain the kit's independent cycle and sway.
    np.testing.assert_array_equal(table[:, 3:], canonical[:, 3:])


class MarchReached(Exception):
    pass


class GlowVariantTests(unittest.TestCase):
    def test_accepted_source_pin_and_its_negative_control(self):
        source = Path(FD.__file__).read_bytes()
        self.assertEqual(hashlib.sha256(source).hexdigest(), ACCEPTED_SOURCE_SHA256)
        with self.assertRaises(AssertionError):
            self.assertEqual(hashlib.sha256(source + b'\n# changed\n').hexdigest(),
                             ACCEPTED_SOURCE_SHA256)

    def test_default_and_explicit_accepted_route_preserve_returned_bytes(self):
        expected = np.arange(60, dtype=np.float32).reshape(4, 5, 3)
        for kwargs in ({}, {'variant': 'accepted'}):
            with self.subTest(kwargs=kwargs), mock.patch.object(FD, 'render', return_value=expected) as render:
                actual = GV.render(420, scale=.25, ss=1., **kwargs)
                render.assert_called_once_with(420, design='arc', scale=.25, ss=1.,
                                               sky_candidate='clear_high_deck')
                self.assertIs(actual, expected)
                assert_same_bytes(actual, expected)
        changed = expected.copy()
        changed[0, 0, 0] = np.nextafter(changed[0, 0, 0], np.float32(1.))
        with self.assertRaises(AssertionError):
            assert_same_bytes(changed, expected)

    def test_accepted_route_with_real_tiny_sky_and_stars(self):
        def terrain(frame, *args, **kwargs):
            shape = (frame.src.H, frame.src.W)
            frame.img = np.zeros((*shape, 3), np.float32)
            frame.zb = np.full(shape, 1e9, np.float32)
            frame.dist = np.full(shape, 1e9, np.float32)
            frame.dist[-2:] = 100.
            frame.img[-2:] = [.03, .02, .01]

        # Native kernels and source focal length at a small image size; the
        # march and skyline are fixtures, not evidence about a full frame.
        with mock.patch.object(FD.PI, 'render_terrain', side_effect=terrain), \
             mock.patch.object(FD, 'skyline', return_value=(0., 1., np.zeros(3))):
            expected = FD.render(420, design='arc', scale=.02, ss=1.,
                                 sky_candidate='clear_high_deck')
            actual = GV.render(420, scale=.02, ss=1.)
        assert_same_bytes(actual, expected)
        self.assertTrue(np.isfinite(actual).all())
        self.assertGreater(float(actual.max()), 0.)

    def test_pulses_follow_the_bar_map_and_additive_owner_offset(self):
        for offset in (0, 7, -13, 87):
            assert_musical_peaks(GV.beat_clock, offset)
        with self.assertRaises(AssertionError):
            assert_musical_peaks(lambda frame, variant, offset:
                                 GV.beat_clock(frame, variant, -offset), 7)
        with self.assertRaises(AssertionError):
            assert_musical_peaks(lambda frame, variant, offset:
                                 ((1 + math.cos(2 * math.pi * (frame + offset) / 10)) / 2) ** 4, 0)

    def test_variants_require_their_absolute_D_frame_ranges(self):
        self.assertEqual(GV.RANGES, D_RANGES)
        self.assertEqual(GV.LENGTHS, {key: end - start + 1 for key, (start, end) in D_RANGES.items()})
        for variant, (start, end) in D_RANGES.items():
            for frame in (start, end):
                self.assertTrue(np.isfinite(GV.state(frame, variant)['glow']))
            for frame in (0, start - 1, end + 1):
                with self.subTest(variant=variant, frame=frame), self.assertRaises(ValueError):
                    GV.state(frame, variant)

    def test_twofires_races_through_its_last_frame(self):
        assert_racing_does_not_settle(GV.state)
        self.assertTrue(all(GV.state(f, 'twofires')['settle'] == 0.
                            for f in range(4560, 5040)))

        def premature_settlement(frame, variant):
            state = GV.state(frame, variant).copy()
            if frame >= 4960:
                state['glow'] = .15
                state['underglow'] = .17
            return state

        with self.assertRaises(AssertionError):
            assert_racing_does_not_settle(premature_settlement)

        def middle_bar_settlement(frame, variant):
            state = GV.state(frame, variant).copy()
            if 4640 <= frame < 4720:
                state['glow'] = .15
                state['underglow'] = .17
            return state

        with self.assertRaises(AssertionError):
            assert_racing_does_not_settle(middle_bar_settlement)

    def test_watch_has_nonzero_warm_bar_breath_preserved_into_dawn(self):
        config = GV.Config(beat_offset=7)
        frames = range(7040, 7120)
        glow = np.array([GV.state(f, 'watch', config)['glow'] for f in frames])
        underglow = np.array([GV.state(f, 'watch', config)['underglow'] for f in frames])
        repeated = np.array([GV.state(f + 80, 'watch', config)['glow'] for f in frames])
        np.testing.assert_allclose(glow, repeated, atol=1e-14, rtol=0.)
        self.assertTrue(np.all(glow > 0.) and np.all(underglow > 0.))
        self.assertGreater(float(np.ptp(glow)), 0.)
        self.assertLess(float(np.ptp(glow)), .1 * (
            GV.state(4080, 'brink')['glow'] - GV.state(4090, 'brink')['glow']))
        peaks = [i for i in range(80) if glow[i] > glow[(i-1) % 80]
                 and glow[i] > glow[(i+1) % 80]]
        self.assertEqual(peaks, [73])
        warm = GV.underglow_rows(GV.state(7320, 'watch', config))[:, 3:6]
        self.assertTrue(np.all(warm[:, 0] > warm[:, 1]))
        self.assertTrue(np.all(warm[:, 1] > warm[:, 2]))
        self.assertTrue(np.all(warm > 0.))
        dawn_glow = np.array([GV.state(f + 320, 'truedawn', config)['glow'] for f in frames])
        dawn_under = np.array([GV.state(f + 320, 'truedawn', config)['underglow'] for f in frames])
        np.testing.assert_allclose(glow, dawn_glow, atol=1e-14, rtol=0.)
        np.testing.assert_allclose(underglow, dawn_under, atol=1e-14, rtol=0.)

    def test_watch_to_dawn_has_no_lighting_jump_and_dawn_progress_is_smooth(self):
        camera = FD.camera(420)
        with mock.patch.object(GV.WD, 'heights'):
            watch = GV.lighting(7359, 'watch', GV.Config(), camera)
            dawn = GV.lighting(7360, 'truedawn', GV.Config(), camera)
        assert_lighting_bridge(watch, dawn)
        progress = np.array([GV.state(frame, 'truedawn')['dawn'] for frame in range(7360, 7840)])
        self.assertEqual(progress[0], 0.)
        self.assertEqual(progress[-1], 1.)
        steps = np.diff(progress)
        self.assertTrue(np.all(steps > 0.))
        self.assertLess(steps[0], steps[len(steps) // 2])
        self.assertLess(steps[-1], steps[len(steps) // 2])
        changed_lights = tuple(array.copy() for array in dawn[0])
        changed_lights[0][3] += .01
        with self.assertRaises(AssertionError):
            assert_lighting_bridge(watch, (changed_lights, *dawn[1:]))

    def test_dawn_composite_has_no_minimum_brightness_jump_after_the_cut(self):
        with ExitStack() as stack:
            for obj, key in ((GV.WD, 'march'), (GV.WD, 'shade'), (GV.WD, 'cloud_glow'),
                             (FD, 'glow_haze'), (FD, 'sky_pass'), (GV.SK, 'splat_stars')):
                stack.enter_context(mock.patch.object(obj, key))
            stack.enter_context(mock.patch.object(GV.NA, 'skyline', return_value=(0., 1., np.zeros(3))))
            stack.enter_context(mock.patch.object(FD, 'milky_way', return_value=np.zeros((1, 1))))
            stack.enter_context(mock.patch.object(GV, 'underglow_rows', return_value=np.zeros((0, 8))))
            stack.enter_context(mock.patch.object(GV, 'fires_table', return_value=np.zeros((0, 6))))
            stack.enter_context(mock.patch.object(GV, '_STARS', np.zeros((1, 3))))
            stack.enter_context(mock.patch.object(GV, 'dawn_parameters', return_value=np.ones(21)))
            stack.enter_context(mock.patch.object(GV.PI, 'to_target', side_effect=lambda f: (f.img, f.zb, f.dist)))
            blend = stack.enter_context(mock.patch.object(GV, 'dawn_pass'))
            for frame, variant in ((7359, 'watch'), (7360, 'truedawn'),
                                   (7361, 'truedawn'), (7839, 'truedawn')):
                blend.reset_mock()
                GV.render(frame, variant, scale=.025, ss=1.)
                progress = GV.state(frame, variant)['dawn']
                if progress > 0.:
                    blend.assert_called_once()
                if blend.called:
                    self.assertEqual(blend.call_args.args[-1], progress)
            # The old 20% minimum would jump on D7361 despite a correct D7360.
            progress = GV.state(7361, 'truedawn')['dawn']
            with self.assertRaises(AssertionError):
                self.assertAlmostEqual(.2 + .8*progress, progress)

    def test_camera_and_cloud_clock_are_locked_to_the_accepted_pose(self):
        terrain = FD.terrain_rows()
        before = terrain.copy()
        config = GV.Config(camera_frame=420)
        expected = FD.PI.Frame(FD.camera(420, 48, 20), 1.).src.params()
        for variant, frame in [(key, frame) for key, ends in D_RANGES.items() for frame in ends]:
            with self.subTest(variant=variant, frame=frame), \
                 mock.patch.object(GV, 'fires_table', return_value=np.zeros((0, 6))), \
                 mock.patch.object(GV, 'dawn_parameters', return_value=np.zeros(21)), \
                 mock.patch.object(GV.WD, 'march', side_effect=MarchReached) as march:
                with self.assertRaises(MarchReached):
                    GV.render(frame, variant, scale=.025, ss=1., config=config)
                params, actual_terrain, actual_camera = march.call_args.args[:3]
                self.assertIs(actual_terrain, terrain)
                np.testing.assert_array_equal(actual_camera, expected)
                self.assertEqual(params[2], 420 / FD.FPS)
        np.testing.assert_array_equal(terrain, before)

    def test_zero_pixel_scale_or_supersampling_fails_before_geometry(self):
        for scale, ss in ((1., 1e-10), (.02, 1e-10), (.0001, 1.)):
            with self.subTest(scale=scale, ss=ss), \
                 mock.patch.object(FD, 'camera', side_effect=AssertionError('geometry reached')) as camera:
                with self.assertRaises(ValueError):
                    GV.render(4080, 'brink', scale=scale, ss=ss)
                camera.assert_not_called()
        # A small nonzero image must get through the guard.
        with mock.patch.object(FD, 'camera', side_effect=MarchReached):
            with self.assertRaises(MarchReached):
                GV.render(4080, 'brink', scale=.002, ss=1.)

    def test_variant_camera_and_pair_are_not_retargetable(self):
        with self.assertRaisesRegex(ValueError, 'camera'):
            GV.Config(camera_frame=419)
        with self.assertRaises(TypeError):
            GV.Config(pair_x=(.2, .8))

    def test_smoke_cache_includes_ignition_and_reuses_exact_construction_state(self):
        row = (123., 45., 678., 4960., 1.7, 817.)
        GV.smoke_for.cache_clear()
        self.addCleanup(GV.smoke_for.cache_clear)
        with mock.patch.object(GV.NA.F, 'Smoke', side_effect=lambda *args, **kwargs: object()) as constructor:
            assert_ignition_cache_isolation(GV.smoke_for, row)
            self.assertEqual(constructor.call_count, 2)
            first, changed = constructor.call_args_list
            self.assertAlmostEqual(first.args[2], 4960. / 24. + .2)
            self.assertAlmostEqual(changed.args[2], 5040. / 24. + .2)
            self.assertAlmostEqual(changed.args[2] - first.args[2], 80. / 24.)
            expected_base = np.array(row[:3])
            expected_base[1] += .65 * GV.NA.fire_dims(row[4])[0]
            np.testing.assert_array_equal(first.args[1], expected_base)
            np.testing.assert_array_equal(changed.args[1], expected_base)
            for index, delta in ((0, .01), (1, .01), (2, .01), (4, .1), (5, 1.)):
                changed = list(row)
                changed[index] += delta
                self.assertIsNot(GV.smoke_for(tuple(changed)), GV.smoke_for(row))

            former_cache = {}

            def without_ignition_key(key_row):
                key = (round(key_row[0], 1), round(key_row[2], 1), int(key_row[5]))
                if key not in former_cache:
                    former_cache[key] = GV.smoke_for(key_row)
                return former_cache[key]

            with self.assertRaises(AssertionError):
                assert_ignition_cache_isolation(without_ignition_key, row)

    def test_lighting_and_underglow_do_not_mutate_shared_arrays(self):
        lights, sky = FD.light(420, 'arc')
        light_before = tuple(a.copy() for a in lights)
        sky_before = sky.copy()
        patches = GV.NA.ug_table()
        patches_before = patches.copy()
        terrain = FD.terrain_rows()
        terrain_before = terrain.copy()
        camera = FD.camera(420)
        with mock.patch.object(FD, 'light', return_value=(lights, sky)), \
             mock.patch.object(GV.NA, 'ug_table', return_value=patches), \
             mock.patch.object(GV.WD, 'heights'):
            for variant in GV.LENGTHS:
                actual, gp, _, state = GV.lighting(D_RANGES[variant][1], variant, GV.Config(), camera)
                GV.underglow_rows(state)
                self.assertFalse(np.shares_memory(gp, sky))
                for result, original in zip(actual, lights):
                    self.assertFalse(np.shares_memory(result, original))
        for actual, original in zip(lights, light_before):
            np.testing.assert_array_equal(actual, original)
        np.testing.assert_array_equal(sky, sky_before)
        np.testing.assert_array_equal(patches, patches_before)
        np.testing.assert_array_equal(terrain, terrain_before)

    def test_complete_variant_routing_keeps_accepted_caches_and_curves_both_light_and_platform(self):
        config = GV.Config(camera_frame=420)
        camera = FD.camera(config.camera_frame)
        physical = np.array([
            [camera.pos[0] - 6000., 700., camera.pos[2] + 10000., 4320., 2.7, 810.],
            [camera.pos[0] + 6000., 800., camera.pos[2] + 10000., 4320., 2.7, 811.],
            [camera.pos[0] + 1000., 300., camera.pos[2] + 4000., 4760., 1.7, 812.],
        ])
        before = physical.copy()
        stars = np.array([[.1, .2, .3]])
        for frame, lit_count in ((4560, 2), (4759, 2), (4760, 3), (5039, 3)):
            with self.subTest(frame=frame), ExitStack() as stack:
                for obj, key in ((GV.WD, 'cloud_glow'),
                                 (FD, 'glow_haze'), (FD, 'sky_pass'), (GV.SK, 'splat_stars')):
                    stack.enter_context(mock.patch.object(obj, key))
                march = stack.enter_context(mock.patch.object(GV.WD, 'march',
                                                              side_effect=lambda *args: args[-1].fill(123.)))
                seats = stack.enter_context(mock.patch.object(GS, 'seat_on_raster',
                                                              side_effect=lambda source, xyz, depth: xyz.copy()))
                shade = stack.enter_context(mock.patch.object(GV.WD, 'shade'))
                stack.enter_context(mock.patch.object(FD, '_SKL', None))
                stack.enter_context(mock.patch.object(FD, '_STARS', None))
                stack.enter_context(mock.patch.object(GV, '_STARS', None))
                stack.enter_context(mock.patch.object(FD, 'skyline', side_effect=AssertionError('accepted cache route')))
                stack.enter_context(mock.patch.object(GV.NA, 'skyline', return_value=(0., 1., np.zeros(3))))
                stack.enter_context(mock.patch.object(FD, 'milky_way', return_value=np.zeros((1, 1))))
                stack.enter_context(mock.patch.object(GV, 'underglow_rows', return_value=np.zeros((0, 8))))
                stack.enter_context(mock.patch.object(GV, 'fires_table', return_value=physical))
                stack.enter_context(mock.patch.object(GV, 'smoke_for'))
                stack.enter_context(mock.patch.object(GV, 'pair_effects', return_value=(mock.Mock(), mock.Mock())))
                stack.enter_context(mock.patch.object(GV, 'basket_pass'))
                stack.enter_context(mock.patch.object(GV.SK, 'make_stars', return_value=stars))
                stack.enter_context(mock.patch.object(GV.PI, 'to_target', side_effect=lambda f: (f.img, f.zb, f.dist)))
                lights = stack.enter_context(mock.patch.object(GV, 'terrain_lights', wraps=GV.terrain_lights))
                platforms = stack.enter_context(mock.patch.object(GV.NA, 'pl_rows', wraps=GV.NA.pl_rows))
                fog = stack.enter_context(mock.patch.object(GV.NA, 'fire_trans', return_value=.5))
                flames = stack.enter_context(mock.patch.object(GV.NA.F2, 'flame'))
                stack.enter_context(mock.patch.object(GV.NA.F2, 'halo'))
                route = mock.Mock()
                for name, call in (('march', march), ('seat', seats), ('lights', lights),
                                   ('platforms', platforms), ('shade', shade)):
                    route.attach_mock(call, name)
                GV.render(frame, 'twofires', scale=.025, ss=1., config=config)
                self.assertEqual([call[0] for call in route.mock_calls],
                                 ['march', 'seat', 'lights', 'platforms', 'shade'])
                self.assertIs(seats.call_args.args[2], march.call_args.args[-1])
                np.testing.assert_array_equal(seats.call_args.args[2], 123.)
                lit = lights.call_args.args[0]
                self.assertEqual(len(lit), lit_count)
                assert_caught_fires_only(lit, frame)
                np.testing.assert_array_equal(lit, physical[physical[:, 3] <= frame])
                assert_curved_fire_rows(platforms.call_args.args[0], physical, camera.pos)
                self.assertIsNone(FD._SKL)
                self.assertIsNone(FD._STARS)
                self.assertIs(GV._STARS, stars)
                self.assertEqual(flames.call_count, lit_count)
                for call in fog.call_args_list:
                    self.assertTrue(any(np.array_equal(call.args[1], row[:3]) for row in physical))
                for call in flames.call_args_list:
                    fire = physical[physical[:, 5] * 7 + 3 == call.kwargs['seed']][0]
                    expected_position = fire[:3].copy()
                    expected_position[1] -= np.sum((fire[[0, 2]] - camera.pos[[0, 2]]) ** 2) / (2 * GV.WD.R_EARTH)
                    np.testing.assert_array_equal(call.args[3], expected_position)
                    self.assertEqual(call.args[6], frame / 24.)
                    self.assertIs(call.args[2].project.__self__, lights.call_args.args[2])
                    size = GV.NA._ign(frame, fire[3])[0]
                    self.assertEqual(call.args[4], GV.NA.fire_dims(fire[4])[0] * size)
        np.testing.assert_array_equal(physical, before)
        with self.assertRaises(AssertionError):
            assert_caught_fires_only(physical, 4759)
        with self.assertRaises(AssertionError):
            assert_curved_fire_rows(physical, physical, camera.pos)

    def test_dawn_rises_on_the_existing_glow_azimuth(self):
        camera = FD.camera(420)
        with mock.patch.object(GV.WD, 'heights'):
            early, late = (GV.dawn_parameters(k, camera) for k in (0., 1.))
        for parameters in (early, late):
            self.assertAlmostEqual(math.degrees(math.atan2(parameters[0], parameters[2])),
                                   FD.GLOW_AZ, places=12)
        self.assertGreater(late[1], early[1])

    def test_real_dawn_kernel_extinguishes_stars_continuously_and_preserves_terrain(self):
        source = FD.PI.Frame(FD.camera(420), 1.5).src
        camera = source.params()
        camera[8] -= 1200.
        camera[9] -= 200.
        with mock.patch.object(GV.WD, 'heights'):
            parameters = GV.dawn_parameters(.5, source)
        shape = (4, 5)
        distance = np.full(shape, 1e9, np.float32)
        distance[-1] = 100.
        original = np.full((*shape, 3), .01, np.float32)
        transmission = np.full(shape, .75, np.float32)
        results = []
        for mix in (0., .2, .21, 1.):
            image, trans = original.copy(), transmission.copy()
            GV.dawn_pass(image, distance, trans, camera, parameters, mix)
            assert_same_bytes(image[-1], original[-1])
            assert_same_bytes(trans[-1], transmission[-1])
            self.assertTrue(np.isfinite(image).all())
            results.append((image, trans))
        assert_same_bytes(results[0][0], original)
        assert_same_bytes(results[0][1], transmission)
        for (_, earlier), (_, later) in zip(results, results[1:]):
            self.assertTrue(np.all(later[:-1] < earlier[:-1]))
            self.assertTrue(np.all(later[:-1] > 0.))
        # Adjacent blend phases change transmission without a binary star cut.
        self.assertLess(float(np.max(results[1][1] - results[2][1])),
                        float(np.max(results[0][1] - results[1][1])))


class RasterSeatTests(unittest.TestCase):
    def scene(self):
        # A nine-column source camera over a synthetic, flat y=0 surface.
        # Its central pixel at (4.5, 6.5) intersects that curved plane at 400 m.
        distance = 400.
        height = 10. - distance**2 / (2 * GV.WD.R_EARTH)
        source = FD.RC.SrcCam([0., height, 0.], 0., 20., -4.5, 4.5, -6., 6.)
        depth = np.zeros((source.H, source.W))
        depth[6, 4] = distance
        floating_tip = np.array([[0., 40., distance]])
        return source, floating_tip, depth

    def test_seat_lands_on_supported_pixel_and_physical_ground_without_mutation(self):
        source, points, depth = self.scene()
        before_points, before_depth = points.copy(), depth.copy()
        with mock.patch.object(GS.WD, 'ground', return_value=0.):
            seated = GS.seat_on_raster(source, points, depth)
        np.testing.assert_allclose(seated, [[0., 0., 400.]], rtol=0., atol=1e-12)
        x, y, _ = source.project(GS.curved(seated, source))
        np.testing.assert_allclose([x[0], y[0]], [4.5, 6.5], rtol=0., atol=1e-12)
        self.assertEqual(depth[math.floor(y[0]), math.floor(x[0])],
                         np.linalg.norm(seated[0, [0, 2]] - source.pos[[0, 2]]))
        np.testing.assert_array_equal(points, before_points)
        np.testing.assert_array_equal(depth, before_depth)
        self.assertFalse(np.shares_memory(points, seated))
        # The same candidate has no valid seat if its one raster hit is lost.
        with mock.patch.object(GS.WD, 'ground', return_value=0.):
            with self.assertRaisesRegex(RuntimeError, 'supports beacon'):
                GS.seat_on_raster(source, points, np.zeros_like(depth))

    def test_missing_support_wrong_dimensions_and_wrong_ground_fail(self):
        source, points, depth = self.scene()
        with self.assertRaisesRegex(ValueError, 'dimensions'):
            GS.seat_on_raster(source, points, depth[:-1])
        with self.assertRaisesRegex(RuntimeError, 'supports beacon'):
            GS.seat_on_raster(source, points, np.full_like(depth, 1e9))
        # A nearby depth alone is insufficient: the ground must reproject to
        # its hit pixel, rather than float back at the catalogue's old tip.
        with mock.patch.object(GS.WD, 'ground', return_value=40.):
            with self.assertRaisesRegex(RuntimeError, 'supports beacon'):
                GS.seat_on_raster(source, points, depth)


class FireProjectionTests(unittest.TestCase):
    def fire(self, camera, distance, seed=810., ignition=4320., size=2.7):
        azimuth = math.radians(FD.YAW)
        position = camera.pos + np.array([math.sin(azimuth)*distance, 0., math.cos(azimuth)*distance])
        return np.r_[position, ignition, size, seed]

    def test_pair_projection_stays_legible_without_inflating_physical_flames(self):
        frame = FD.PI.Frame(FD.camera(420), 1.5)
        frame.img = np.zeros((1, 1, 3), np.float32)
        frame.zb = np.zeros((1, 1), np.float32)
        source_scale = FD.PI.src_scale(frame)

        def draw(row):
            with mock.patch.object(GV.NA, 'fire_trans', return_value=.5), \
                 mock.patch.object(GV.NA, '_ign', return_value=(1., 1., 1.)), \
                 mock.patch.object(GV, 'pair_effects', return_value=(mock.Mock(), mock.Mock())), \
                 mock.patch.object(GV, 'basket_pass'), \
                 mock.patch.object(GV.NA.F2, 'flame') as flame, \
                 mock.patch.object(GV.NA.F2, 'halo'):
                GV.draw_fire(frame, row, 4560, np.zeros(8))
                return flame.call_args

        for distance, seed in ((10000., 810.), (45000., 811.)):
            row = self.fire(frame.src, distance, seed)
            before = row.copy()
            call = draw(row)
            assert_nominal_flame_height(call, source_scale, 60.)
            self.assertEqual(call.args[4], GV.NA.fire_dims(2.7)[0])
            self.assertEqual(call.args[5], GV.NA.fire_dims(2.7)[1])
            self.assertIs(call.args[2].project.__self__, frame.src)
            self.assertIs(call.kwargs['TG'], GV.pair_tongues(int(seed)))
            np.testing.assert_array_equal(row, before)
        self.assertIsNone(draw(self.fire(frame.src, 10000., 812.)).kwargs['TG'])
        with mock.patch.object(GV, 'fire_height_px', return_value=2.7):
            with self.assertRaises(AssertionError):
                assert_nominal_flame_height(draw(self.fire(frame.src, 45000.)), source_scale, 60.)

        chain = [GV.fire_height_px(self.fire(frame.src, 10000., seed), frame.src)
                 for seed in range(812, 817)]
        np.testing.assert_array_equal(chain, [20., 24., 32., 36., 40.])
        self.assertTrue(all(20. <= height <= 40. for height in chain))
        self.assertTrue(np.all(np.diff(chain) > 0.))
        watch = [GV.fire_height_px(self.fire(frame.src, distance, 818.), frame.src)
                 for distance in (2000., 5000., 20000., 40000.)]
        self.assertTrue(all(12. <= height <= 26. for height in watch))
        self.assertTrue(np.all(np.diff(watch) < 0.))

    def test_ground_light_pools_are_curved_bounded_and_exclude_precatch_light(self):
        camera = FD.camera(420)
        fires = np.vstack([self.fire(camera, 10000.), self.fire(camera, 45000., 811.),
                           self.fire(camera, 3000., 812., ignition=5040.)])
        before = fires.copy()
        with mock.patch.object(GV.NA, 'fire_trans', return_value=.5), \
             mock.patch.object(GV.NA.F, 'flicker', return_value=1.):
            lights = GV.terrain_lights(fires, 5039, camera, np.zeros(8))
            inflated = fires.copy()
            inflated[:, 4] *= 1000.
            np.testing.assert_array_equal(GV.terrain_lights(inflated, 5039, camera, np.zeros(8)), lights)
            early = GV.terrain_lights(fires, 4319, camera, np.zeros(8))
        self.assertEqual(early.shape, (0, 8))
        # The reused kit does emit a pre-catch pickup at -1 frame; the new
        # caller's time gate is therefore necessary, rather than vacuous.
        self.assertGreater(len(GV.NA.lt_rows(fires, 4319)), 0)
        self.assertEqual(lights.shape, (2, 8))
        self.assertTrue(np.isfinite(lights).all())
        for light, fire in zip(lights, fires[:2]):
            curved = GV.curved(fire[:3], camera.pos)
            np.testing.assert_array_equal(light[[0, 2]], curved[[0, 2]])
            depth = camera.project(curved)[2]
            height_pixels = (light[1] - curved[1]) * camera.f / depth
            soft_pixels = light[7] * camera.f / depth
            self.assertTrue(0. < height_pixels < 8.)
            self.assertTrue(3. < soft_pixels < 6.)
            self.assertGreater(light[6], 0.)
        np.testing.assert_array_equal(fires, before)


class PairEffectTests(unittest.TestCase):
    def test_pair_tongues_are_separated_cached_and_keep_canonical_motion(self):
        GV.pair_tongues.cache_clear()
        self.addCleanup(GV.pair_tongues.cache_clear)
        for seed in (810, 811):
            canonical = GV.NA.F.tongue_table(7, seed * 7 + 3)
            before = canonical.copy()
            table = GV.pair_tongues(seed)
            assert_separated_pair_tongues(table, canonical)
            self.assertIs(GV.pair_tongues(seed), table)
            self.assertFalse(table.flags.writeable)
            np.testing.assert_array_equal(canonical, before)
            for column, replacement in ((0, canonical[:, 0]), (1, np.ones(7)),
                                         (2, canonical[:, 2]), (2, np.zeros(7))):
                regressed = table.copy()
                regressed[:, column] = replacement
                with self.subTest(seed=seed, column=column), self.assertRaises(AssertionError):
                    assert_separated_pair_tongues(regressed, canonical)
        self.assertIsNot(GV.pair_tongues(810), GV.pair_tongues(811))

    def test_local_effect_camera_keeps_the_root_and_positive_x_moves_right(self):
        source = FD.PI.Frame(FD.camera(420), 1.5).src
        pos = source.pos + source.fwd * 10000.
        local = GV.pair_camera(source, pos, 60., 1.5)
        root = np.array(source.project(pos))
        projected = np.array(local.project(np.array([[0., 0., 0.], [1., 0., 0.],
                                                    [0., 1., 0.], [0., 0., 2.]]))).T
        np.testing.assert_array_equal(projected[0], root)
        self.assertGreater(projected[1, 0], root[0])
        self.assertLess(projected[2, 1], root[1])
        self.assertEqual(projected[3, 2], root[2] + 2.)
        tip = local.project([0., GV.NA.fire_dims(2.7)[0], 0.])
        self.assertAlmostEqual(root[1] - tip[1], 90.)
        with self.assertRaises(AssertionError):
            np.testing.assert_array_equal(np.array(source.project([0., 0., 0.])), root)

    def test_pair_effects_reuse_seed_and_ignition_and_share_rightward_wind(self):
        GV.pair_effects.cache_clear()
        self.addCleanup(GV.pair_effects.cache_clear)
        with mock.patch.object(GV.NA.F, 'Smoke', side_effect=lambda *a, **kw: mock.Mock()) as smoke, \
             mock.patch.object(GV.NA.F, 'Sparks',
                               side_effect=lambda *a, **kw: mock.Mock(T0=np.array([1200., 1800.]))) as sparks:
            assert_pair_effect_cache_isolation(GV.pair_effects)
            first = GV.pair_effects(810, 4320)
            self.assertEqual(smoke.call_count, 3)
            self.assertEqual(sparks.call_count, 3)
            for constructor in (smoke, sparks):
                for call in constructor.call_args_list:
                    self.assertGreater(call.kwargs['wind'][0], 0.)
                    np.testing.assert_array_equal(call.kwargs['wind'][1:], [0., 0.])
                    self.assertGreater(call.kwargs['rate'], 0.)
            self.assertEqual(smoke.call_args_list[0].args[2], 4320/24 + .2)
            self.assertEqual(sparks.call_args_list[0].args[2], 4320/24)
            self.assertTrue(np.all((first[1].T0 > 0.) & (first[1].T0 < [1200., 1800.])))
            by_seed_only = {}

            def wrong_cache(seed, ignition):
                if seed not in by_seed_only:
                    by_seed_only[seed] = GV.pair_effects(seed, ignition)
                return by_seed_only[seed]

            with self.assertRaises(AssertionError):
                assert_pair_effect_cache_isolation(wrong_cache)

    def test_real_basket_kernel_has_openings_and_obeys_scene_depth(self):
        original = np.full((40, 40, 3), .3, np.float32)
        depth = np.full((40, 40), 1e9, np.float32)
        before_depth = depth.copy()
        visible = original.copy()
        GV.basket_pass(visible, depth, 20., 25., 1000., 1., 10., 1., .1)
        changed = np.any(visible != original, axis=2)
        self.assertGreater(np.count_nonzero(changed), 0)
        assert_basket_opening(visible, original)
        solid = visible.copy()
        solid[19:25, 15:25] = visible[19, 20]
        with self.assertRaises(AssertionError):
            assert_basket_opening(solid, original)
        self.assertFalse(np.any(changed[25:]))
        self.assertTrue(np.isfinite(visible).all())
        self.assertTrue(np.all(visible[changed] < original[changed]))
        self.assertTrue(np.all(visible[changed, 0] > visible[changed, 1]))
        hidden = original.copy()
        GV.basket_pass(hidden, np.full_like(depth, 100.), 20., 25., 1000., 1., 10., 1., .1)
        np.testing.assert_array_equal(hidden, original)
        np.testing.assert_array_equal(depth, before_depth)

    def test_pair_draw_routes_smoke_and_sparks_through_one_anchored_camera(self):
        frame = FD.PI.Frame(FD.camera(420), 1.5)
        frame.img = np.zeros((1, 1, 3), np.float32)
        frame.zb = np.zeros((1, 1), np.float32)
        fire = np.r_[frame.src.pos + frame.src.fwd*10000., 4320., 2.7, 810.]
        smoke, sparks, order = mock.Mock(), mock.Mock(), mock.Mock()
        with mock.patch.object(GV, 'pair_effects', return_value=(smoke, sparks)) as effects, \
             mock.patch.object(GV.NA, 'fire_trans', return_value=.5), \
             mock.patch.object(GV.NA.F2, 'flame') as flame, \
             mock.patch.object(GV.NA.F2, 'halo') as halo, \
             mock.patch.object(GV, 'basket_pass') as basket:
            for name, call in (('smoke', smoke.render), ('flame', flame), ('halo', halo),
                               ('basket', basket), ('sparks', sparks.render)):
                order.attach_mock(call, name)
            GV.draw_fire(frame, fire, 4560, np.zeros(8))
        effects.assert_called_once_with(810, 4320)
        self.assertIs(flame.call_args.kwargs['TG'], GV.pair_tongues(810))
        self.assertEqual(flame.call_args.kwargs['tongues'], 7)
        self.assertEqual([call[0] for call in order.mock_calls],
                         ['smoke', 'flame', 'halo', 'halo', 'basket', 'sparks'])
        effect_camera = smoke.render.call_args.args[2]
        self.assertIs(sparks.render.call_args.args[2], effect_camera)
        np.testing.assert_array_equal(np.array(effect_camera.project([0., 0., 0.])),
                                      np.array(frame.src.project(GV.curved(fire[:3], frame.src.pos))))
        self.assertEqual(smoke.render.call_args.args[3], 4560/24)
        self.assertEqual(sparks.render.call_args.args[3], 4560/24)
        self.assertGreater(smoke.render.call_args.args[5], 0.)
        self.assertGreater(sparks.render.call_args.kwargs['gain'], 0.)


class GlowSiteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.camera = FD.camera(420)
        cls.points = GV.fires_table('twofires')[:, :3].copy()

    def test_real_sites_are_grounded_and_approach_on_two_beat_catches(self):
        terrain = FD.terrain_rows()
        before = terrain.copy()
        for point in self.points:
            self.assertAlmostEqual(point[1], GV.WD.ground(point[0], point[2], terrain), places=8)
        for offset in (0, 7, -13):
            config = GV.Config(beat_offset=offset)
            fires = GV.fires_table('twofires', config)
            np.testing.assert_array_equal(fires[:, :3], self.points)
            assert_approaching_ignitions(fires, self.camera.pos, offset)
        np.testing.assert_array_equal(GV.fires_table('twofires')[:, 3],
                                      [4320., 4320., 4760., 4800., 4840., 4880., 4920.])
        self.assertEqual(len(self.points), 7)
        self.assertEqual(np.count_nonzero(GV.fires_table('twofires')[:, 3] <= 4560), 2)
        np.testing.assert_array_equal(terrain, before)

    def test_staggered_pair_and_receding_chain_are_rejected(self):
        fires = GV.fires_table('twofires')
        staggered = fires.copy()
        staggered[1, 3] += 1
        with self.assertRaises(AssertionError):
            assert_approaching_ignitions(staggered, self.camera.pos, 0)
        reversed_tail = fires.copy()
        reversed_tail[2:, :3] = fires[:1:-1, :3]
        with self.assertRaises(AssertionError):
            assert_approaching_ignitions(reversed_tail, self.camera.pos, 0)

    def test_curvature_is_copy_only_and_uses_camera_relative_distance(self):
        before = self.points.copy()
        curved = GS.curved(self.points, self.camera)
        expected_drop = np.sum((self.points[:, [0, 2]] - self.camera.pos[[0, 2]]) ** 2,
                               axis=1) / (2 * GV.WD.R_EARTH)
        np.testing.assert_allclose(self.points[:, 1] - curved[:, 1], expected_drop, atol=1e-12)
        np.testing.assert_array_equal(curved[:, [0, 2]], self.points[:, [0, 2]])
        np.testing.assert_array_equal(self.points, before)
        self.assertFalse(np.shares_memory(curved, self.points))
        for point in self.points:
            np.testing.assert_array_equal(GV.curved(point, self.camera.pos), GS.curved(point, self.camera))
        default_sites = GS.sites()
        returned = GS.sites()
        returned[0] = 0.
        np.testing.assert_array_equal(GS.sites(), default_sites)

    def test_frozen_layout_fails_if_its_catalogue_predecessor_moves(self):
        catalogue, selected = GS._selection()
        before = catalogue.copy()
        changed = catalogue.copy()
        changed[selected[0], 0] += 1.
        with mock.patch.object(GS, '_catalogue', return_value=changed):
            with self.assertRaisesRegex(RuntimeError, 'no longer belongs'):
                GS.sites()
        np.testing.assert_array_equal(catalogue, before)

    def test_fixed_crown_bases_project_to_the_native_match_cut_positions(self):
        assert_crown_projection(self.points, self.camera)
        np.testing.assert_array_equal(np.array([GV.CROWN_LEFT_BASE, GV.CROWN_RIGHT_BASE]), CROWN_BASES)
        np.testing.assert_array_equal(GS.PAIR_NATIVE_BASES, CROWN_BASES)
        np.testing.assert_array_equal(GS.PAIR_ANCHORS, self.points[:2])
        self.assertFalse(GS.PAIR_ANCHORS.flags.writeable)
        self.assertFalse(GS.PAIR_NATIVE_BASES.flags.writeable)
        displaced = self.points.copy()
        displaced[0, 0] += 25.
        with self.assertRaises(AssertionError):
            assert_crown_projection(displaced, self.camera)

    def test_sparse_raster_seating_preserves_crowns_and_keeps_new_fires_above_captions(self):
        # Measured native seats from the placement audit's site_support.json.
        # Sparse D exercises inverse projection and real ground, but does not
        # reproduce a native march or independently prove these hits visible.
        seats = np.array([
            [-5893.961767525668, 345.47371458581017, 7824.677509257563],
            [-16897.70185712461, 1302.8076234419345, 43064.91201378168],
            [-2016.6510169513372, -128.21857507335358, 7824.359521043991],
            [-4713.231578180752, -99.36926433671692, 5608.094712384385],
            [-2758.647755778466, 9.641783917682233, 4385.74148229244],
            [-1127.8148235207993, 39.12886087826655, 3900.2399474579233],
            [-1027.5300496076838, 61.11387261299615, 3831.819250829244],
            [-4130.959259382069, -210.37954813209558, 4637.067777304031],
        ])
        source = FD.PI.Frame(self.camera, 1.5).src
        depth = np.full((source.H, source.W), 1e9)
        x, y, _ = source.project(GS.curved(seats, source))
        distance = np.linalg.norm(seats[:, [0, 2]] - source.pos[[0, 2]], axis=1)
        depth[y.astype(int), x.astype(int)] = distance
        original_depth = depth.copy()
        watch_only = GS.watch_sites()[7]
        inputs = np.vstack([self.points, watch_only])
        before = inputs.copy()
        with mock.patch.object(GS, 'validate_chain_caption_band',
                               wraps=GS.validate_chain_caption_band) as caption:
            seated = GS.seat_on_raster(source, inputs, depth)
        self.assertEqual(caption.call_count, 5)
        np.testing.assert_allclose(seated, seats, rtol=0., atol=1e-6)
        np.testing.assert_array_equal(seated[:2], GS.PAIR_ANCHORS)
        assert_crown_projection(seated, self.camera)
        native_y = self.camera.project(GS.curved(seated, self.camera))[1]
        self.assertTrue(np.all(native_y[:7] < .8 * self.camera.H))
        self.assertGreater(native_y[7], .8 * self.camera.H)
        np.testing.assert_array_equal(depth, original_depth)
        np.testing.assert_array_equal(inputs, before)
        with self.assertRaisesRegex(RuntimeError, 'caption band'):
            GS.validate_chain_caption_band([seated[7]])
        # Putting that same ridge back in the igniting chain must fail after
        # seating; merely checking its old summit tip would miss the intrusion.
        wrong_chain = np.vstack([self.points[:6], watch_only])
        with mock.patch.object(GS, '_production_points', return_value=wrong_chain):
            with self.assertRaisesRegex(RuntimeError, 'caption band'):
                GS.seat_on_raster(source, [watch_only], depth)
        for index in (0, 1):
            corrupted = depth.copy()
            corrupted[int(y[index]), int(x[index])] = 1e9
            with self.assertRaisesRegex(RuntimeError, 'native raster support'):
                GS.seat_on_raster(source, self.points[:2], corrupted)

    def test_watch_retains_chain_and_adds_unique_distributed_sites(self):
        watch_fires = GV.fires_table('watch')
        dawn_fires = GV.fires_table('truedawn')
        watch = watch_fires[:, :3]
        np.testing.assert_array_equal(watch[:len(self.points)], self.points)
        self.assertEqual(len(watch), 40)
        np.testing.assert_array_equal(watch_fires, dawn_fires)
        np.testing.assert_array_equal(watch_fires[:len(self.points)], GV.fires_table('twofires'))
        np.testing.assert_array_equal(watch_fires[len(self.points):, 3], 5040.)
        self.assertEqual(len(GV.NA.lt_rows(watch_fires, 7040)), 40)
        self.assertEqual(len(np.unique(watch, axis=0)), len(watch))
        x, y, _ = self.camera.project(GS.curved(watch, self.camera))
        screen = np.column_stack([x / self.camera.W, y / self.camera.H])
        for index in range(len(self.points), len(watch)):
            self.assertGreaterEqual(float(np.linalg.norm(screen[:index] - screen[index], axis=1).min()), .04)


if __name__ == '__main__':
    unittest.main()
