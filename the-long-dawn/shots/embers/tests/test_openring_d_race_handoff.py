"""Continue the forging source and ease race optics without retiming the score.

The real compositor methods run against recording sinks. Negative controls
remove the source, restore the optical step, and restore collapsed spark rows.
"""
from unittest.mock import patch
import unittest

import numpy as np

from test_c_d import C, context, scene_for
import d_thinking_fire as F


def frame_context(scene, frame):
    ctx = context(scene, frame, .05)
    start, end = C.D.shot_range(scene.shot)
    ctx.t0, ctx.t1 = max(frame - .25, start), min(frame + .25, end - .02)
    ctx.cam0, ctx.cam1 = scene.camera(ctx.t0), scene.camera(ctx.t1)
    return ctx


def original_sparks(scene, ctx):
    """Round5's renderer footprint, independent of the new transition helper."""
    p0, _, ids0 = scene.spark_paths(max(ctx.t - .06, ctx.t0))
    p1, energy, ids1 = scene.spark_paths(min(ctx.t + .06, ctx.t1))
    if not len(p1):
        return
    if not np.array_equal(ids0, ids1):
        p0 = p1
    z = np.maximum((p1 - ctx.cam.pos) @ ctx.cam.R[2], 1.)
    ctx.fr.splat(p0, p1, .014, energy * .012 * (ctx.cam.f_px(1920) / z) ** 2,
                 [1., .72, .3], ctx.cam0, ctx.cam1, zref=0.)


class RaceHandoff(unittest.TestCase):
    def rejects(self, oracle, *args):
        with self.assertRaises(AssertionError):
            oracle(*args)

    def test_same_thinking_fire_continues_at_the_cut_and_on_the_absolute_clock(self):
        forging, race = scene_for('forging'), scene_for('race')
        incoming = np.array(forging.central_fire_anchors(2399.))

        def oracle(draw):
            with patch.object(F, 'draw') as source, patch.object(C.V5.CF, 'draw') as orange, \
                 patch.object(C.c3, 'occ_vis', return_value=np.ones((40, 96))) as visibility:
                for frame in (2400., 2401., 2405., 2410., 2440., 2560., 2719.):
                    source.reset_mock()
                    ctx = frame_context(race, frame)
                    draw(ctx.fr.resolve(), ctx)
                    source.assert_called_once()
                    args, kwargs = source.call_args
                    self.assertEqual(args[3], frame)
                    self.assertEqual(kwargs['bright'], 1.1)
                    self.assertEqual(kwargs['scale'], ctx.scale)
                    np.testing.assert_array_equal(kwargs['vis'], 1.)
                    np.testing.assert_array_equal(args[1:3], race.central_fire_anchors(frame))
                    if frame == 2400.:
                        np.testing.assert_allclose(args[1:3], incoming, rtol=0., atol=1e-9)
                    depth = float((race.race_fire_world()[0] - ctx.cam.pos) @ ctx.cam.R[2])
                    self.assertEqual(visibility.call_args.args[1], depth)
                orange.assert_not_called()
            self.assertEqual(F.source_clock(2400.) - F.source_clock(2399.), 1.)
        oracle(race.draw_central_fire)
        self.rejects(oracle, lambda hdr, ctx: None)
        self.rejects(oracle, lambda hdr, ctx: F.draw(hdr, incoming[0], incoming[1], 2399., bright=1.1))
        world = race.race_fire_world().copy()
        with patch.object(race, 'race_fire_world', return_value=world + [0., .5, 0.]):
            self.rejects(oracle, race.draw_central_fire)

    def test_fire_is_one_fixed_world_source_as_the_existing_camera_moves(self):
        race = scene_for('race')
        world = race.race_fire_world().copy()
        normal = race.camera(2400.).pos * [1., 0., 1.]
        np.testing.assert_allclose((world - C.c3.FIRE_ROOT) @ normal, 0., atol=1e-10)

        def oracle(anchors):
            positions = []
            for frame in (2400., 2410., 2440., 2560., 2719.):
                x, y, _ = race.camera(frame).project(world, 1920, 804)
                expected = np.column_stack([x, y])
                np.testing.assert_allclose(anchors(frame), expected, rtol=0., atol=1e-10)
                np.testing.assert_array_equal(race.race_fire_world(), world)
                positions.append(expected)
            self.assertFalse(np.array_equal(positions[0], positions[-1]))
        oracle(race.central_fire_anchors)
        locked = race.central_fire_anchors(2400.)
        self.rejects(oracle, lambda frame: locked)

    def test_ring_shading_eases_for_ten_frames_while_geometry_uses_the_current_score(self):
        race = scene_for('race')

        def raster(*args, **kwargs):
            shape = (args[2], args[1])
            return np.zeros(shape + (3,), np.float32), np.zeros(shape), np.full(shape, np.inf)

        def oracle():
            for frame, amount, shading in ((2400., 0., 2399.), (2405., .5, 2402.), (2410., 1., 2410.),
                                           (2560., 1., 2560.), (2719., 1., 2719.)):
                self.assertEqual(race.race_handoff(frame), amount)
                self.assertEqual(race.ring_shading_frame(frame), shading)
                with patch.object(C, 'ring_state', wraps=C.ring_state) as state, \
                     patch.object(C.RS, 'render', side_effect=raster) as render, \
                     patch.object(C.RS, 'merge_occluder', return_value=None), \
                     patch.object(C.RS, 'visibility', return_value=np.ones((40, 96))):
                    race.ring_layer(frame_context(race, frame))
                    state.assert_called_once_with(shading)
                    args, kwargs = render.call_args
                    for actual, expected in zip(args[3:6], race.ring_frame(frame)):
                        np.testing.assert_array_equal(actual, expected)
                    np.testing.assert_array_equal(kwargs['th_range'], C.D.theta_range(frame))
                    for owner, point in enumerate(args[7].points):
                        np.testing.assert_array_equal(point[0], race.towers.top(owner, shading))
                        np.testing.assert_array_equal(point[1], np.array([4., 1.7, .5]) * C.forge_level(owner, shading))
        oracle()
        with patch.object(race, 'race_handoff', return_value=1.):
            self.rejects(oracle)
        for shot in ('forging', 'trap', 'holdout', 'instep', 'unfinished'):
            scene = scene_for(shot)
            for frame in (2400., 2405., 2410.):
                self.assertEqual(scene.race_handoff(frame), 1.)
                self.assertEqual(scene.ring_shading_frame(frame), frame)

    def test_gap_glare_starts_at_forging_strength_and_rejoins_the_race_by_2410(self):
        race = scene_for('race')
        incoming_score = C.D.glare(2399.)
        incoming_pulse = race.schedule.beat_pulse(2399.)

        def oracle():
            for frame, amount in ((2400., 0.), (2405., .5), (2410., 1.), (2420., 1.), (2719., 1.)):
                with patch.object(C.d_glare, 'render', return_value=None) as render:
                    race.glare_layer(frame_context(race, frame))
                    args = render.call_args.args
                    expected = np.array(incoming_score) * (1. - amount) + np.array(C.D.glare(frame)) * amount
                    np.testing.assert_array_equal(args[5], expected)
                    self.assertEqual(args[6], incoming_pulse * (1. - amount) + race.schedule.beat_pulse(frame) * amount)
                    np.testing.assert_array_equal(args[3], race.ends(frame))
                    if frame == 2400.:
                        self.assertEqual(args[6], 0.)
        oracle()
        with patch.object(race, 'race_handoff', return_value=1.):
            self.rejects(oracle)

    def test_incoming_spark_streaks_survive_packet_birth_and_removal(self):
        forging, race = scene_for('forging'), scene_for('race')

        def oracle(draw):
            for frame in (2400., 2404.):
                ctx = frame_context(race, frame)
                draw(ctx)
                lengths = []
                for args, _ in ctx.fr.calls:
                    live = np.asarray(args[3]) > 0.
                    lengths.extend(np.linalg.norm(args[1][live] - args[0][live], axis=1))
                self.assertTrue(lengths)
                self.assertGreater(max(lengths), 1e-3, f'all live streaks collapsed at D{frame}')
                if frame == 2400.:
                    args = ctx.fr.calls[0][0]
                    live = np.asarray(args[3]) > 0.
                    expected0, _, _ = forging.spark_paths(frame - .06)
                    expected1, energy, _ = forging.spark_paths(frame + .06)
                    expected_live = energy > 0.
                    np.testing.assert_array_equal(args[0][live], expected0[expected_live])
                    np.testing.assert_array_equal(args[1][live], expected1[expected_live])
        oracle(race.sparks)
        self.rejects(oracle, lambda ctx: original_sparks(race, ctx))
        with patch.object(race, 'race_handoff', return_value=1.):
            self.rejects(oracle, race.sparks)

    def test_spark_footprints_crossfade_then_match_round5_exactly(self):
        race = scene_for('race')

        def oracle(draw):
            ctx = frame_context(race, 2405.)
            draw(ctx)
            self.assertEqual(len(ctx.fr.calls), 2)
            reference = frame_context(race, 2405.)
            original_sparks(race, reference)
            expected_energy = reference.fr.calls[0][0][3]
            for args, _ in ctx.fr.calls:
                np.testing.assert_array_equal(args[3], expected_energy * .5)
            for frame in (2410., 2420., 2560., 2719.):
                actual, expected = frame_context(race, frame), frame_context(race, frame)
                draw(actual)
                original_sparks(race, expected)
                self.assertEqual(len(actual.fr.calls), len(expected.fr.calls))
                for (got, got_kw), (want, want_kw) in zip(actual.fr.calls, expected.fr.calls):
                    for left, right in zip(got[:5], want[:5]):
                        np.testing.assert_array_equal(left, right)
                    for left, right in zip(got[5:], want[5:]):
                        np.testing.assert_array_equal(left.params(96, 40), right.params(96, 40))
                    self.assertEqual(got_kw, want_kw)
        oracle(race.sparks)
        self.rejects(oracle, lambda ctx: original_sparks(race, ctx))
        handoff = race.race_handoff
        with patch.object(race, 'race_handoff', side_effect=lambda t: handoff(min(t, 2409.))):
            self.rejects(oracle, race.sparks)


if __name__ == '__main__':
    unittest.main()
