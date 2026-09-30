"""Race work stays continuous, scored, and short of the band except at its giants.

Exercise real trajectories and schedules with the deterministic tower layout;
only mesh construction is replaced. Every new oracle has a broken control.
"""
from copy import copy
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from test_c_d import C, context, scene_for, tower_init
import d_forges as VF
import d_race_embers as RE


WORK_BEATS = tuple(range(2400, 2640, 20)) + tuple(range(2640, 2721, 10))


class RaceContracts(unittest.TestCase):
    def rejects(self, oracle, *args):
        with self.assertRaises(AssertionError):
            oracle(*args)

    def test_every_forge_has_live_sparks_throughout_the_race_and_followers_miss(self):
        scene = scene_for('race')
        followers = set(range(scene.towers.k_all)) - set(C.LEADERS)
        # The complete end face fits inside this section bounding rectangle.
        # Its circumscribed sphere also excludes accidental off-centre contact;
        # testing only equality with the end centre would miss that failure.
        cap_radius = 2.2 * np.hypot(C.RS.WIDTH * .5, C.RS.THICK * .5)

        def oracle(paths):
            for frame in range(2400, 2720):
                points, energy, owners = paths(frame)
                live = energy > 0.
                self.assertEqual(set(owners[live]), set(range(18)), f'D{frame}')
                follower = live & np.isin(owners, tuple(followers))
                distance = np.linalg.norm(points[follower, None, :] -
                                          scene.ends(frame)[None, :, :], axis=2).min(1)
                self.assertGreater(float(distance.min()), cap_radius + .014,
                                   f'a follower touches an end at D{frame}')

        oracle(scene.spark_paths)
        original = scene.spark_paths

        def missing_forge(frame):
            points, energy, owners = original(frame)
            return points, energy * (owners != C.HOLDOUT), owners
        self.rejects(oracle, missing_forge)

        def illegal_contact(frame):
            points, energy, owners = original(frame)
            points[(owners == C.HOLDOUT) & (energy > 0.)] = scene.ends(frame)[0]
            return points, energy, owners
        self.rejects(oracle, illegal_contact)

        # Restoring the old forty-frame work spacing leaves dead intervals.
        with patch.object(scene.schedule, 'beats', [b for b in scene.schedule.beats
                                                    if b <= 2400 or b % 40 == 0]):
            self.rejects(oracle, scene.spark_paths)

    def test_both_giants_contact_their_own_moving_end_on_each_scored_impact(self):
        scene = scene_for('race')

        def oracle():
            for hit in WORK_BEATS[:-1]:
                points, energy, owners = scene.spark_paths(hit)
                ends = scene.ends(hit)
                for side, owner in enumerate(C.LEADERS):
                    live = (owners == owner) & (energy > 0.)
                    self.assertTrue(live.any())
                    distance = np.linalg.norm(points[live] - ends[side], axis=1).min()
                    self.assertLess(float(distance), 1e-7, f'giant {owner} at D{hit}')
        oracle()
        with patch.object(C, 'spark_reach', return_value=.8):
            self.rejects(oracle)
        original = scene.spark_paths

        def wrong_end(frame):
            points, energy, owners = original(frame)
            points[owners == C.LEADERS[1]] = scene.ends(frame)[0]
            return points, energy, owners
        with patch.object(scene, 'spark_paths', side_effect=wrong_end):
            self.rejects(oracle)

    def test_work_clock_hits_every_twenty_frames_then_doubles_toward_the_deep(self):
        def oracle():
            schedule = scene_for('race').schedule
            beats = tuple(b for b in schedule.beats if b >= 2400)
            self.assertEqual(beats, WORK_BEATS)
            self.assertEqual(tuple(schedule.pulses), tuple(schedule.beats))
            with C.world(schedule):
                self.assertEqual(tuple(C.B.BEATS), tuple(schedule.beats))
                for hit in WORK_BEATS:
                    self.assertEqual(C.B.beat_pulse(hit), 1.)
                    self.assertLess(C.B.beat_pulse(hit + 8.), 1.)
                    self.assertEqual(schedule.shutter(0, hit, np.zeros((1, 3))), .8)
        oracle()
        with patch.object(C, 'race_work_beats', return_value=tuple(range(2400, 2721, 40))):
            self.rejects(oracle)

    def test_restaging_is_opt_in_and_vision_keeps_its_original_frozen_architecture(self):
        # Round4's public closure clock: vision inherits it independently of
        # the additional work beats now owned by the explicit race scene.
        original_beats = (2240., 2260., 2280., 2300., 2320., 2340.,
                          2400., 2440., 2480., 2520., 2560., 2600., 2620.,
                          2640., 2660., 2670., 2680., 2690., 2700., 2710.)

        def original_oracle(schedule):
            self.assertEqual(tuple(schedule.beats), original_beats)
            self.assertEqual(tuple(schedule.pulses), original_beats)
            with patch.object(C, 'race_lift', side_effect=AssertionError('restaging leaked into shared world')), \
                 patch.object(C.B.Towers, '__init__', tower_init):
                frozen = isinstance(schedule, VF.Schedule)
                towers = VF.Forges(schedule) if frozen else C.Forges(schedule)
                # At D2959 all original rises have finished, the race gap is
                # already55degrees, and none of the later trap events has
                # begun: accepted heights are rest+2, or rest+4 for the giants.
                expected = towers.rest + np.where(np.isin(np.arange(18), C.LEADERS), 4., 2.)
                for frame in ((2959., 2960., 4079., 9199.) if frozen else (2959.,)):
                    actual = np.array([towers.height(i, frame) for i in range(18)])
                    np.testing.assert_array_equal(actual, expected)
        original_oracle(C.Schedule('race'))
        vision = VF.Schedule('race', 2959.)
        original_oracle(vision)
        with patch.object(vision, 'race_restage', True):
            self.rejects(original_oracle, vision)
        with patch.object(vision, 'beats', list(WORK_BEATS)):
            self.rejects(original_oracle, vision)

        race = scene_for('race')
        def opted_in_oracle():
            with patch.object(C, 'race_lift', return_value=123.) as lift:
                race.towers.height(0, 2560.)
                lift.assert_called_once_with(0, 2560.)
        opted_in_oracle()
        with patch.object(race.schedule, 'race_restage', False):
            self.rejects(opted_in_oracle)

    def test_work_impacts_drive_facade_heat_and_gap_glare(self):
        scene = scene_for('race')

        def oracle():
            with C.world(scene.schedule):
                for hit in (2420, 2460, 2540, 2650, 2690, 2710):
                    # Six frames after impact, the inherited heat wave crosses
                    # height43: -2 + 7.5 * 6, independent of the gap's holds.
                    wave = scene.towers.heat_wave(0, hit + 6., np.array([43.]))
                    self.assertGreater(float(wave[0]), 0.)
                    with patch.object(C.d_glare, 'render', return_value=None) as glare:
                        scene.glare_layer(context(scene, hit, .02))
                        glare.assert_called_once()
                        self.assertEqual(glare.call_args.args[6], 1.)
        oracle()
        with patch.object(scene.schedule, 'pulses', []):
            self.rejects(oracle)
        with patch.object(scene.schedule, 'beats', []):
            self.rejects(oracle)

    def test_nonclosing_beats_still_lift_towers_and_advance_followers(self):
        scene = scene_for('race')

        def oracle():
            for hit in (2420, 2460, 2500, 2540, 2580):
                self.assertEqual(C.D.gap_deg(hit), C.D.gap_deg(hit + 6.))
                for owner in range(18):
                    self.assertGreater(scene.towers.height(owner, hit + 6.),
                                       scene.towers.height(owner, hit))
                    if owner not in C.LEADERS:
                        self.assertGreater(C.spark_reach(owner, hit + 6.),
                                           C.spark_reach(owner, hit - 1.))
                        self.assertLess(C.spark_reach(owner, hit + 6.), 1.)
        oracle()
        with patch.object(C, 'race_lift', return_value=0.):
            self.rejects(oracle)
        original = C.spark_reach
        with patch.object(C, 'spark_reach', side_effect=lambda i, t: original(i, 2400.)):
            self.rejects(oracle)

    def test_camera_moves_after_an_exact_forging_pose_handoff(self):
        forging, race = scene_for('forging'), scene_for('race')

        def opening_oracle(camera, pose):
            expected = forging.camera(2400.)
            for field in ('pos', 'target', 'R', 'hfov', 'focus', 'aperture'):
                np.testing.assert_array_equal(getattr(camera, field), getattr(expected, field))
            for actual, expected in zip(pose, forging.ring_frame(2400.)):
                np.testing.assert_array_equal(actual, expected)
        opening_oracle(race.camera(2400.), race.ring_frame(2400.))
        shifted = copy(race.camera(2400.))
        shifted.pos = shifted.pos + [0., .1, 0.]
        self.rejects(opening_oracle, shifted, race.ring_frame(2400.))
        rotation, centre, size = race.ring_frame(2400.)
        self.rejects(opening_oracle, race.camera(2400.), (rotation + .01, centre, size))

        def motion_oracle(camera):
            samples = [camera(f) for f in (2400., 2440., 2560., 2719.)]
            for before, after in zip(samples, samples[1:]):
                self.assertFalse(np.array_equal(before.pos, after.pos))
            # After the initial reframing, the camera closes on the forge field.
            self.assertGreater(np.linalg.norm(samples[1].pos[[0, 2]]),
                               np.linalg.norm(samples[-1].pos[[0, 2]]))
        motion_oracle(race.camera)
        self.rejects(motion_oracle, lambda frame: race.camera(2400.))

        def field_oracle(camera):
            # The close retains a broad field while foreground columns can
            # leave picture. The guard requires sixteen including both giants.
            # This measures framing, not depth occlusion.
            for frame in (2440., 2560., 2719.):
                tops = np.array([race.towers.top(i, frame) for i in range(18)])
                x, y, z = camera(frame).project(tops, 1920, 804)
                visible = (x >= 0.) & (x < 1920.) & (y >= 0.) & (y < 804.) & (z > 0.)
                ids = set(np.flatnonzero(visible))
                self.assertGreaterEqual(len(ids), 16, f'D{frame}')
                self.assertTrue(set(C.LEADERS).issubset(ids), f'D{frame}')
                self.assertTrue(ids.intersection(range(8, 18)), f'D{frame}')
                # The measured v2 field spans1124px at D2464 and1658px
                # at D2719; keep it wider than half the native picture.
                self.assertGreater(float(np.ptp(x[visible])), 960., f'D{frame}')
                if frame == 2719.:
                    self.assertTrue(np.all(y[list(C.LEADERS)] < 402.))
        field_oracle(race.camera)
        self.rejects(field_oracle, lambda frame: forging.camera(2400.))

        def distorted_camera(frame, narrow=False):
            camera = race.camera(frame)
            def project(points, width, height):
                x, y, z = camera.project(points, width, height)
                if narrow:
                    x = 959.5 + .1 * (x - 959.5)
                elif frame == 2719.:
                    y = 430. + .1 * y
                return x, y, z
            return SimpleNamespace(project=project)
        self.rejects(field_oracle, lambda frame: distorted_camera(frame, True))
        self.rejects(field_oracle, distorted_camera)

    def test_surface_embers_construct_the_original_emitter_for_all_eighteen_forges(self):
        towers = SimpleNamespace(k=8, k_all=18)

        def oracle(factory):
            with patch.object(RE.B.TowerEmbers, '__init__', return_value=None) as construct:
                factory(towers)
                construct.assert_called_once()
                view = construct.call_args.args[0]
                self.assertEqual(view.k, towers.k_all)
                self.assertIs(view._tw, towers)
                self.assertEqual(construct.call_args.kwargs['per'], 2000)
        oracle(RE.ForgeEmbers)
        with patch.object(RE.EYE, '_AllTowers', side_effect=lambda tw: SimpleNamespace(k=8, _tw=tw)):
            self.rejects(oracle, RE.ForgeEmbers)

    def test_surface_ember_sink_removes_contact_at_both_shutter_positions(self):
        emitter = object.__new__(RE.ForgeEmbers)
        ends0 = np.array([[-10., 0., 0.], [10., 0., 0.]])
        ends1 = ends0 + [0., 8., 0.]
        # One safe grain per forge, plus grains violating each of the four
        # endpoint/time combinations and one lying exactly on the boundary.
        safe0 = np.column_stack([np.arange(18), np.full(18, 30.), np.zeros(18)])
        safe1 = safe0 + [0., .1, 0.]
        far = np.array([0., 30., 0.])
        p0 = np.concatenate([safe0, [ends0[0], far, ends0[1], far, ends0[0] + [3., 0., 0.]]])
        p1 = np.concatenate([safe1, [far, ends1[0], far, ends1[1], far]])
        energy = np.arange(1., len(p0) + 1.)
        colour = np.array([1., .8, .5])
        target = SimpleNamespace(splat=Mock())
        ctx = SimpleNamespace(fr=target)

        def original_emit(self, proxy):
            proxy.fr.splat(p0, p1, .006, energy, colour, 'camera0', 'camera1', zref=28.)

        def oracle(invoke):
            target.splat.reset_mock()
            with patch.object(RE.B.TowerEmbers, 'emit', original_emit):
                invoke(ctx, ends0, ends1, .4)
            target.splat.assert_called_once()
            args, kwargs = target.splat.call_args
            np.testing.assert_array_equal(args[0], p0)
            np.testing.assert_array_equal(args[1], p1)
            self.assertEqual(args[2], .006)
            np.testing.assert_array_equal(args[3][:18], energy[:18] * .4)
            np.testing.assert_array_equal(args[3][18:], 0.)
            np.testing.assert_array_equal(args[4], colour)
            self.assertEqual(args[5:], ('camera0', 'camera1'))
            self.assertEqual(kwargs, {'zref': 28.})
            self.assertIs(ctx.fr, target)
            np.testing.assert_array_equal(energy, np.arange(1., len(p0) + 1.))
        oracle(emitter.emit_bounded)
        self.rejects(oracle, lambda ctx, e0, e1, gain: original_emit(emitter, ctx))
        self.rejects(oracle, lambda ctx, e0, e1, gain: emitter.emit_bounded(ctx, e0, e0, gain))
        self.rejects(oracle, lambda ctx, e0, e1, gain: emitter.emit_bounded(ctx, e1, e1, gain))

        def dark_oracle(invoke):
            with patch.object(RE.B.TowerEmbers, 'emit') as source:
                for gain in (0., -1.):
                    invoke(ctx, ends0, ends1, gain)
                source.assert_not_called()
        dark_oracle(emitter.emit_bounded)
        self.rejects(dark_oracle, lambda ctx, e0, e1, gain: emitter.emit_bounded(ctx, e0, e1, 1.))

    def test_surface_ember_sweeps_cannot_cross_a_moving_end_between_shutter_samples(self):
        emitter = object.__new__(RE.ForgeEmbers)
        ends0 = np.array([[-10., 0., 0.], [10., 0., 0.]])
        ends1 = ends0 + [0., 8., 0.]
        # Relative to the moving left end these travel from x=-6 to x=+6.
        # Every shutter endpoint is outside the exclusion sphere; the first
        # two cross it, and the third is a real near miss that must survive.
        offset = np.array([[-6., 0., 0.], [-6., 2.9, 0.], [-6., 3.1, 0.]])
        p0 = ends0[0] + offset
        p1 = ends1[0] + offset + [12., 0., 0.]
        energy = np.array([1., 2., 3.])
        target = SimpleNamespace(splat=Mock())
        ctx = SimpleNamespace(fr=target)

        def source(self, proxy):
            proxy.fr.splat(p0, p1, .006, energy, [1., .8, .5])

        def oracle(invoke):
            target.splat.reset_mock()
            with patch.object(RE.B.TowerEmbers, 'emit', source):
                invoke(ctx, ends0, ends1, .5)
            target.splat.assert_called_once()
            np.testing.assert_array_equal(target.splat.call_args.args[3], [0., 0., 1.5])
        oracle(emitter.emit_bounded)

        def endpoints_only(ctx, e0, e1, gain):
            d0 = np.linalg.norm(p0[:, None] - e0[None], axis=2).min(1)
            d1 = np.linalg.norm(p1[:, None] - e1[None], axis=2).min(1)
            keep = np.minimum(d0, d1) > 3.
            ctx.fr.splat(p0, p1, .006, energy * gain * keep, [1., .8, .5])
        self.rejects(oracle, endpoints_only)


if __name__ == '__main__':
    unittest.main()
