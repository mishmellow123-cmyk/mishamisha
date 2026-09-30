"""D-scene contracts exercised without tower meshes, rasterization, or JIT.

The actual scene compositor, trajectories, lamp light wiring and screen-space
glare run through recording sinks. Each visual rule has a broken control.
"""
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(HERE), str(HERE.parents[1] / 'lib')]
import c_d as C


class RecordingFrame:
    def __init__(self, scale=.2):
        self.W, self.H = round(1920 * scale), round(804 * scale)
        self.occ, self.calls = None, []

    def set(self, **kwargs):
        self.options = kwargs

    def splat(self, *args, **kwargs):
        self.calls.append((args, kwargs))

    def resolve(self):
        return np.zeros((self.H, self.W, 3), np.float32)


def tower_init(towers):
    """Only deterministic inputs to the real C3 layout and D height methods."""
    towers.k, towers.k_all = 8, 18
    towers.G = [
        {3: {'p': np.array([[1., 40., 0.]]), 'n': np.array([[1., 0., 0.]]),
             'key': np.array([0])}}
        for _ in range(towers.k_all)
    ]
    towers.source_forge_forms = towers.G[:towers.k]


def scene_for(shot):
    with patch.dict(os.environ, {'LD_OPEN_RING': '1'}), \
         patch.object(C.B.Towers, '__init__', tower_init), \
         patch.object(C.EYE, '_all_smoke', return_value=SimpleNamespace(emit=Mock())):
        return C.Scene(shot)


def context(scene, frame, scale=.2):
    camera = scene.camera(frame)
    return SimpleNamespace(t=float(frame), t0=frame - .25, t1=frame + .25,
                           scale=scale, cam=camera, cam0=camera, cam1=camera,
                           fr=RecordingFrame(scale))


class CutDContracts(unittest.TestCase):
    def rejects(self, oracle, *args):
        with self.assertRaises(AssertionError):
            oracle(*args)

    def test_explicit_d_entrypoint_requires_flag_before_constructing_geometry(self):
        with patch.dict(os.environ, {'LD_OPEN_RING': '0'}), \
             patch.object(C, 'Forges') as geometry:
            with self.assertRaisesRegex(ValueError, 'LD_OPEN_RING=1'):
                C.Scene('race')
            geometry.assert_not_called()

    def test_no_crown_flame_call_from_any_actual_frame_pipeline(self):
        import d_thinking_fire
        def oracle(inject_flame=False):
            with patch.object(C.V5.CF, 'draw') as draw, \
                 patch.object(d_thinking_fire, 'draw') as thinking_draw, \
                 patch.object(C, 'Frame', RecordingFrame), \
                 patch.object(C.look, 'finish', side_effect=lambda hdr, **kw: hdr):
                for shot in C.D.SHOTS:
                    scene = scene_for(shot)
                    scene.towers.prepare = Mock()
                    scene.towers.emit = Mock()
                    scene.ring_layer = lambda ctx: ctx.fr.resolve()
                    scene.drops = lambda ctx: None
                    frame = C.D.shot_range(shot)[0] + 24
                    draw.reset_mock()
                    thinking_draw.reset_mock()
                    result = scene.frame(frame, .05)
                    if inject_flame:
                        C.V5.CF.draw(None, (0., 0.), (0., 0.), frame)
                    self.assertEqual(draw.call_count, 0)
                    self.assertEqual(thinking_draw.call_count, int(shot == 'forging'))
                    if shot == 'forging':
                        root, tip = scene.central_fire_anchors(frame)
                        np.testing.assert_allclose(thinking_draw.call_args.args[1], root)
                        np.testing.assert_allclose(thinking_draw.call_args.args[2], tip)
                    self.assertEqual(result.shape, (40, 96, 3))
                    scene.towers.emit.assert_called_once()
                    scene.smoke.emit.assert_called_once()
        oracle()
        self.rejects(oracle, True)

    def test_industrial_facades_keep_windows_seams_and_joints(self):
        def oracle(schedule):
            for frame in [shot.d_start + 20. for shot in C.D.SHOTS.values()]:
                self.assertEqual(schedule.ashlar_k(frame), 0.)
                self.assertIsNone(schedule.masonry(0, np.zeros((2, 3)), frame))
                schedule.tower_rise_glow(frame)
                self.assertGreater(schedule.seam_k(0), 0.)
                self.assertGreater(schedule.joint_k(0), 0.)
                self.assertGreater(schedule.shutter(0, frame, np.zeros((2, 3))), .1)
                self.assertFalse(hasattr(schedule, 'line_mod'))
        for shot in C.D.SHOTS:
            oracle(C.Schedule(shot))
        broken = C.Schedule('trap')
        broken.ashlar_k = lambda t: 1.
        self.rejects(oracle, broken)
        broken = C.Schedule('trap')
        broken.shutter = lambda *args: 0.
        self.rejects(oracle, broken)

    def test_far_ring_uses_the_same_eight_forge_forms(self):
        towers = scene_for('trap').towers
        def oracle(forms):
            self.assertEqual(len(forms), towers.k_all)
            for form in forms:
                self.assertTrue(any(form is original for original in towers.source_forge_forms))
        oracle(towers.G)
        wrong = towers.G.copy()
        wrong[C.HOLDOUT] = object()
        self.rejects(oracle, wrong)

    def test_source_events_retain_measured_timing_and_pair_are_tallest(self):
        scene = scene_for('trap')
        towers = scene.towers

        def oracle(level, height):
            # Power intervals come from c_v5, translated by D's explicit table.
            for i in range(towers.k_all):
                if i == C.HOLDOUT:
                    continue
                for c in (2320., 2479., 2480., 2486., 2492., 2579., 2580., 2589., 2598., 2639.):
                    d = c + C.D.SHOTS['trap'].offset
                    self.assertEqual(level(i, d), C.V5.forge_level(i, c))
            sink_start, sink_end = C.D.EVENTS['sink']
            returning, returned = C.D.EVENTS['return']
            surge_start, surge_end = C.D.EVENTS['surge_height']
            self.assertEqual(level(C.HOLDOUT, sink_start), 1.)
            self.assertEqual(level(C.HOLDOUT, sink_end), 0.)
            self.assertEqual(level(C.HOLDOUT, returning - 1.), 0.)
            self.assertEqual(level(C.HOLDOUT, returning), 0.)
            self.assertLess(abs(level(C.HOLDOUT, returning + .001)
                                - level(C.HOLDOUT, returning - .001)), 1e-6)
            self.assertGreater(level(C.HOLDOUT, returned), 1.)
            self.assertEqual(height(0, surge_start - 1.), height(0, surge_start))
            self.assertAlmostEqual(height(0, surge_end) - height(0, surge_start), 6.)
            self.assertLess(height(C.HOLDOUT, sink_end), height(C.HOLDOUT, sink_start))
            self.assertAlmostEqual(height(C.HOLDOUT, C.D.EVENTS['return_height'][1])
                                   - height(C.HOLDOUT, sink_start - 1.), 7.)
            probes = [shot.d_end - 1. for shot in C.D.SHOTS.values()]
            probes += [C.D.SHOTS['trap'].d_start, surge_end, C.D.HOLDOUT_STRIKE]
            for frame in probes:
                heights = np.array([height(i, frame) for i in range(towers.k_all)])
                self.assertEqual(set(np.argsort(heights)[-2:]), set(C.LEADERS))
            for frame in (C.D.SETTLED, C.D.COOL_END, C.D.SHOTS['unfinished'].d_start + 240.):
                for i in range(towers.k_all):
                    self.assertEqual(level(i, frame), C.STEP.WORK_LEVEL)

        oracle(C.forge_level, towers.height)
        self.rejects(oracle, lambda i, t: C.forge_level(i, t - 1.), towers.height)
        self.rejects(oracle, C.forge_level,
                     lambda i, t: towers.height(i, t) + (100. if i == C.HOLDOUT else 0.))
        def old_pilot_jump(i, t):
            if i == C.HOLDOUT and C.D.EVENTS['return'][0] <= t <= C.D.EVENTS['return'][1]:
                return C.V5.forge_level(i, C.D.legacy_frame('trap', t))
            return C.forge_level(i, t)
        self.rejects(oracle, old_pilot_jump, towers.height)

    def test_rendered_giant_and_holdout_sparks_contact_the_current_cut_end(self):
        def oracle(scene, hit, owners):
            for owner in owners:
                for particle in (0, 7, 31, 63):
                    arrival = hit + float(scene.spark_delay[owner, particle])
                    points, energy, ids = scene.spark_paths(arrival)
                    row = np.flatnonzero(ids == owner)[particle]
                    self.assertGreater(energy[row], 0.)
                    ends = scene.ends(arrival)
                    if owner in C.LEADERS:
                        distance = np.linalg.norm(ends[C.LEADERS.index(owner)] - points[row])
                    else:
                        distance = np.linalg.norm(ends - points[row], axis=1).min()
                    self.assertLess(distance, 1e-5,
                                    f'tower {owner} misses moving end at D{arrival}: {distance}')

        race = scene_for('race')
        hit = C.D.SHOTS['race'].d_start + 120.
        oracle(race, hit, C.LEADERS)
        holdout = scene_for('holdout')
        oracle(holdout, C.D.HOLDOUT_STRIKE, (C.HOLDOUT,))
        with patch.object(C, 'spark_reach', return_value=.8):
            self.rejects(oracle, race, hit, C.LEADERS)
        original = race.spark_paths
        def both_to_one_end(t):
            points, energy, owners = original(t)
            points[owners == C.LEADERS[1]] = race.ends(t)[0]
            return points, energy, owners
        with patch.object(race, 'spark_paths', side_effect=both_to_one_end):
            self.rejects(oracle, race, hit, C.LEADERS)

    def test_non_giant_trajectories_fall_short_and_holdout_has_no_early_contact(self):
        def shortfall_oracle(scene, hit):
            measured = {i: np.inf for i in range(scene.towers.k_all) if i not in C.LEADERS}
            for frame in np.arange(hit - 12., hit + 3., .25):
                points, energy, owners = scene.spark_paths(frame)
                if not len(points):
                    continue
                distances = np.linalg.norm(points[:, None, :] - scene.ends(frame)[None, :, :], axis=2).min(1)
                for owner in measured:
                    live = (owners == owner) & (energy > 0.)
                    if live.any():
                        measured[owner] = min(measured[owner], float(distances[live].min()))
            self.assertTrue(all(np.isfinite(list(measured.values()))))
            self.assertGreater(min(measured.values()), .2)

        race = scene_for('race')
        hit = C.D.SHOTS['race'].d_start + 120.
        shortfall_oracle(race, hit)
        with patch.object(C, 'spark_reach', return_value=1.):
            self.rejects(shortfall_oracle, race, hit)
        holdout = scene_for('holdout')
        for frame in np.arange(C.D.SHOTS['holdout'].d_start, C.D.HOLDOUT_STRIKE, .25):
            points, energy, owners = holdout.spark_paths(frame)
            live = energy > 0.
            self.assertTrue(np.all(owners[live] == C.HOLDOUT))
            if live.any():
                distances = np.linalg.norm(points[live, None, :] - holdout.ends(frame)[None, :, :], axis=2).min(1)
                self.assertGreater(distances.min(), 1e-4)

    def test_forging_race_cut_preserves_the_next_strikes_live_approach(self):
        forging, race = scene_for('forging'), scene_for('race')
        cut = C.D.SHOTS['race'].d_start
        self.assertEqual(C.D.SHOTS['forging'].d_end, cut)
        def oracle():
            for frame in (cut - 1., cut - .25, cut):
                before = forging.spark_paths(frame)
                after = race.spark_paths(frame)
                self.assertGreater(np.count_nonzero(before[1] > 0.), 0)
                for left, right in zip(before, after):
                    np.testing.assert_array_equal(left, right)
                for giant in C.LEADERS:
                    self.assertTrue(np.any((before[2] == giant) & (before[1] > 0.)))
        oracle()
        forging.schedule.beats = [hit for hit in forging.schedule.beats if hit < cut]
        self.rejects(oracle)

    def test_glare_pixels_widen_during_race_and_clear_at_pause(self):
        scene = scene_for('race')
        race_start, race_end = C.D.shot_range('race')
        reference = race_start + 120.
        fixed_ends = scene.ends(reference)
        fixed_camera = scene.camera(reference)
        fixed_angles = C.D.theta_range(reference)

        def footprint(frame):
            ctx = context(scene, frame)
            ctx.cam = fixed_camera
            with patch.object(scene, 'ends', return_value=fixed_ends), \
                 patch.object(C.D, 'theta_range', return_value=fixed_angles), \
                 patch.object(scene.schedule, 'beat_pulse', return_value=1.):
                rgb = scene.glare_layer(ctx)
            self.assertTrue(np.isfinite(rgb).all())
            return int(np.count_nonzero(rgb[..., 0] > .02))

        def oracle():
            racing = [footprint(f) for f in (race_start, reference, race_end - 1.)]
            cooling = [footprint(f) for f in (C.D.PAUSE_START, C.D.SETTLED,
                                             C.D.COOL_END - 12., C.D.COOL_END)]
            self.assertTrue(all(a < b for a, b in zip(racing, racing[1:])), racing)
            self.assertTrue(all(a > b for a, b in zip(cooling, cooling[1:])), cooling)
            self.assertEqual(cooling[-1], 0)
            self.assertEqual(footprint(C.D.SHOTS['unfinished'].d_end - 1.), 0)

        oracle()
        with patch.object(C.D, 'glare', return_value=C.D.Glare(32., 1.)):
            self.rejects(oracle)

    def test_gold_detaches_from_birth_pose_while_forging_ring_rises(self):
        scene = scene_for('forging')
        target = np.array([10., 10., 0.])
        normal = np.array([-1., 0., 0.])
        scene.window_sources = lambda: [(0, 0, np.zeros(3), normal)]
        scene.towers.world = lambda *args: target[None, :].copy()
        scene.towers.nworld = lambda *args: normal[None, :].copy()
        actual_pose = scene.ring_frame
        a, b = C.D.shot_range('forging')
        # Choose one complete 64-frame flight while the band rises, independent
        # of how the edit renumbers the shot or its modulo-64 drop phases.
        flight_birth = 64. * np.ceil((a + b) / 128.)
        self.assertLess(flight_birth + 24., b)

        def capture(frame, current_pose=False):
            ctx = context(scene, frame)
            pose = (lambda time: actual_pose(frame)) if current_pose else actual_pose
            with patch.object(C.c3.GoldRain, '_near_forges', return_value=[0]), \
                 patch.object(scene, 'ring_frame', side_effect=pose):
                scene.drops(ctx)
            self.assertEqual(len(ctx.fr.calls), 1)
            return ctx.fr.calls[0][0][1][0]

        def oracle(capture_position):
            recovered = []
            for frame in (flight_birth + 2., flight_birth + 14., flight_birth + 24.):
                point = capture_position(frame)
                u = (frame / 64.) % 1.
                end = target + normal * .12
                origin = (point - end * u) / (1. - u)
                origin[1] = (point[1] - end[1] * u * u) / (1. - u * u)
                recovered.append(origin)
                birth = frame - 64. * u
                rotation, centre, size = actual_pose(birth)
                p, _ = C.RS.local_points(np.array([C.D.arc_theta(birth, 0.)]), np.array([0.]))
                np.testing.assert_allclose(origin, centre + size * p[0] @ rotation.T, atol=1e-10)
            np.testing.assert_allclose(recovered, np.broadcast_to(recovered[0], (3, 3)), atol=1e-10)

        oracle(capture)
        self.rejects(oracle, lambda frame: capture(frame, current_pose=True))

    def test_each_lamp_rises_from_a_tower_then_its_visible_position_lights_ring(self):
        scene = scene_for('unfinished')
        n = scene.towers.k_all
        first, last = C.D.shot_range('unfinished')
        origin, start = scene.lamp_positions(first)
        destination, settled = scene.lamp_positions(first + 240.)
        np.testing.assert_array_equal(start, np.zeros(n))
        np.testing.assert_array_equal(settled, np.ones(n))
        np.testing.assert_allclose(origin, [scene.towers.top(i, first) for i in range(n)])
        self.assertTrue(np.all(destination[:, 1] > origin[:, 1]),
                        'Each lamp must rise from its own tower to its hanging position')
        for frame in (first + 20., first + 60., first + 100.):
            positions, amount = scene.lamp_positions(frame)
            np.testing.assert_allclose(positions,
                                       origin * (1. - amount[:, None]) + destination * amount[:, None])
        np.testing.assert_allclose(scene.lamp_positions(last - 1.)[0], destination)

        seen = []
        ctx = context(scene, first + 90., .025)
        def raster(camera, w, h, rot, centre, size, state, env, **kwargs):
            seen.extend(env.points)
            return np.zeros((h, w, 3)), np.ones((h, w)), np.ones((h, w))
        with patch.object(C.RS, 'render', raster), \
             patch.object(C.RS, 'merge_occluder', return_value=None), \
             patch.object(C.RS, 'visibility', return_value=np.ones((ctx.fr.H, ctx.fr.W))):
            scene.ring_layer(ctx)
        scene.lamps(ctx)
        light_positions = np.array([row[0] for row in seen[-n:]])
        visible_positions = ctx.fr.calls[0][0][1]

        def oracle(lights, visible):
            self.assertEqual(len(lights), n)
            self.assertEqual(len(visible), n)
            np.testing.assert_array_equal(lights, visible)
        oracle(light_positions, visible_positions)
        self.rejects(oracle, light_positions + np.array([0., 1., 0.]), visible_positions)
        self.rejects(oracle, light_positions[:-1], visible_positions)

    def test_failed_world_restores_c_globals_and_releases_lock(self):
        before = C.variant.CUT, C.B.SCHED, C.B.IGN, C.B.BEATS
        functions = C.V5.forge_level, C.V5.ring_warmth, C.c3.ring_frame
        with self.assertRaisesRegex(RuntimeError, 'deliberate failure'):
            with C.world(C.Schedule('trap')):
                self.assertEqual(C.variant.CUT, 'C3')
                raise RuntimeError('deliberate failure')
        self.assertEqual((C.variant.CUT, C.B.SCHED, C.B.IGN, C.B.BEATS), before)
        self.assertEqual((C.V5.forge_level, C.V5.ring_warmth, C.c3.ring_frame), functions)
        with C.world(C.Schedule('race')):
            self.assertEqual(C.B.SCHED.shot, 'race')

    def test_band_and_active_lamps_keep_native_frame_margins_for_every_d_frame(self):
        psi = np.linspace(0., 2. * np.pi, 17)
        fractions = np.linspace(0., 1., 97)
        def margin_oracle(scene, frames, camera_override=None):
            worst = (np.inf, None, None, None)
            for frame in frames:
                rotation, centre, size = scene.ring_frame(frame)
                th0, th1 = C.D.theta_range(frame)
                theta, sections = np.meshgrid(th0 + fractions * (th1 - th0), psi)
                local, _ = C.RS.local_points(theta.ravel(), sections.ravel())
                points = centre + size * local @ rotation.T
                owners = np.full(len(points), -1, dtype=int)
                if scene.shot == 'forging' and frame == C.D.SHOTS['forging'].d_start:
                    # The first view establishes the existing band over the
                    # central fire before the camera follows the ring upward.
                    fire = np.array([C.c3.FIRE_ROOT, C.c3.FIRE_ROOT + [0., C.c3.HF, 0.]])
                    points = np.concatenate([points, fire])
                    owners = np.concatenate([owners, np.full(2, -2)])
                if scene.shot == 'unfinished':
                    lamps, active = scene.lamp_positions(frame)
                    lamp_ids = np.flatnonzero(active > 0.)
                    lamps = lamps[active > 0.]
                    if len(lamps):
                        # Enclose the published cage/handle bounds, including
                        # the depth direction under the actual camera angle.
                        cage = np.concatenate([lamps + [x, y, z]
                                               for x in (-C.LAMP_RADIUS, C.LAMP_RADIUS)
                                               for y in (C.LAMP_BOTTOM, C.LAMP_TOP)
                                               for z in (-C.LAMP_RADIUS, C.LAMP_RADIUS)])
                        points = np.concatenate([points, cage])
                        owners = np.concatenate([owners, np.tile(lamp_ids, 8)])
                camera = scene.camera(frame) if camera_override is None else camera_override(frame)
                x, y, z = camera.project(points, 1920, 804)
                self.assertGreater(z.min(), 0.)
                edges = np.stack([x, 1919. - x, y, 803. - y])
                edge, point = np.unravel_index(np.argmin(edges), edges.shape)
                margin = edges[edge, point]
                if margin < worst[0]:
                    worst = (margin, frame, ('left', 'right', 'top', 'bottom')[edge], owners[point])
            self.assertGreaterEqual(worst[0], 12.,
                                    f'{scene.shot} margin {worst[0]:.3f}px at D{worst[1]} '
                                    f'{worst[2]} edge, lamp {worst[3]} (-1=band, -2=ground fire)')

        for shot in C.D.SHOTS:
            with self.subTest(shot=shot):
                margin_oracle(scene_for(shot), range(*C.D.shot_range(shot)))
        # The first native holdout framing aimed six units below the corrected
        # target. That observed crop must continue to fail this same oracle.
        scene = scene_for('holdout')
        def old_target(frame):
            angle = C.c3.AZ0 + .16
            position = np.array([75. * np.cos(angle), C.B.GROUND + 48., 75. * np.sin(angle)])
            return C.Camera(position, [0., C.B.GROUND + 48., 0.], hfov=46.,
                            focus=np.linalg.norm(C.RING_C - position), aperture=.012)
        self.rejects(margin_oracle, scene, range(*C.D.shot_range('holdout')), old_target)

    def test_withdrawing_and_holdout_crown_stay_visible_through_their_shots(self):
        # ID8 is the round forge-chimney form. This box conservatively encloses
        # its cap and crenellations so a visible centre cannot hide a cropped rim.
        corners = np.array([[x, y, z] for x in (-1.8, 1.8) for y in (-.8, .3)
                            for z in (-1.8, 1.8)])
        def oracle(scene, camera_override=None):
            worst = (np.inf, None)
            for frame in range(*C.D.shot_range(scene.shot)):
                points = scene.towers.top(C.HOLDOUT, frame) + corners
                camera = scene.camera(frame) if camera_override is None else camera_override
                x, y, z = camera.project(points, 1920, 804)
                self.assertGreater(z.min(), 0.)
                margin = min(x.min(), 1919. - x.max(), y.min(), 803. - y.max())
                if margin < worst[0]:
                    worst = (margin, frame)
            self.assertGreaterEqual(worst[0], 20.,
                                    f'{scene.shot} holdout crown margin {worst[0]:.3f}px at D{worst[1]}')

        for shot, old in (('trap', (88., 45., 50., 52., .14)),
                          ('holdout', (75., 48., 54., 46., .16))):
            scene = scene_for(shot)
            oracle(scene)
            radius, y, target, hfov, az_offset = old
            angle = C.c3.AZ0 + az_offset
            position = np.array([radius * np.cos(angle), C.B.GROUND + y, radius * np.sin(angle)])
            old_camera = C.Camera(position, [0., C.B.GROUND + target, 0.], hfov=hfov,
                                  focus=np.linalg.norm(C.RING_C - position), aperture=.012)
            self.rejects(oracle, scene, old_camera)

    def test_cli_keeps_d_frame_identity_and_rejects_frames_outside_shot(self):
        first, stop = C.D.shot_range('unfinished')
        last = stop - 1
        self.assertEqual(C.frames(f'{first}-{first + 4}:2,{last}'), [first, first + 2, first + 4, last])
        for spec in (f'{first},{first}', f'{first + 4}-{first}', f'{first}-{first + 4}:0'):
            with self.assertRaises(ValueError):
                C.frames(spec)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve() / 'repo'
            root.mkdir()
            out = root.parent / 'frames'
            scene = SimpleNamespace(frame=Mock(return_value=np.zeros((2, 3, 3), np.float32)))
            argv = ['c_d.py', 'unfinished', '--range', f'{first},{last}', '--out', str(out), '--format', 'png']
            with patch.object(sys, 'argv', argv), patch.object(C, 'Scene', return_value=scene), \
                 patch.object(C, 'ROOT', root), patch('builtins.print'):
                C.main()
            self.assertEqual(sorted(p.name for p in out.iterdir()), [f'f_{first:05d}.png', f'f_{last:05d}.png'])
            self.assertEqual([call.args[0] for call in scene.frame.call_args_list], [first, last])
            argv[3] = str(first - 1)
            with patch.object(sys, 'argv', argv), patch.object(C, 'Scene') as constructor, \
                 patch.object(sys, 'stderr'):
                with self.assertRaises(SystemExit):
                    C.main()
                constructor.assert_not_called()


if __name__ == '__main__':
    unittest.main()
