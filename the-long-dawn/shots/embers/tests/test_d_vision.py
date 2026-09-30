"""Cut-D conditional closure and independently projected match-cut geometry."""
from copy import copy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(HERE), str(HERE.parents[1] / 'lib')]
import d_vision as D
from core import Camera


class ConditionalVision(unittest.TestCase):
    def assert_schedule(self, state):
        for frame in range(2960, 3520):
            data = state(frame)
            if frame < 3200 or frame >= 3440:
                self.assertEqual(data['gap'], 55., frame)
                self.assertFalse(data['closed'], frame)
                self.assertEqual(data['eye'], 0., frame)
                self.assertEqual(data['opening'], 0., frame)
            else:
                self.assertEqual(data['closed'], frame >= 3232, frame)
                self.assertGreaterEqual(data['gap'], 0., frame)
                if frame >= 3232:
                    self.assertEqual(data['gap'], 0., frame)
                if data['eye'] > 0:
                    self.assertTrue(data['closed'], frame)
        self.assertEqual(state(3200)['gap'], 55.)
        self.assertEqual(state(3439)['eye'], 1.)
        self.assertEqual(state(3439)['opening'], 1.)
        gaps = [state(f)['gap'] for f in range(3200, 3440)]
        self.assertTrue(np.all(np.diff(gaps) <= 0.))

    def test_closure_is_confined_to_vision_and_reopens_on_the_cut(self):
        self.assert_schedule(D.state)
        mutations = ((3199, 'closed', True), (3440, 'gap', 0.),
                     (3440, 'eye', 1.), (3440, 'opening', 1.),
                     (3231, 'closed', True), (3232, 'gap', 1.),
                     (3200, 'gap', 54.), (3439, 'eye', .9),
                     (3439, 'opening', .9), (3228, 'gap', -1.),
                     (3201, 'eye', .5), (3210, 'gap', 55.))
        for target, key, value in mutations:
            def wrong(frame):
                data = D.state(frame).copy()
                if frame == target:
                    data[key] = value
                return data
            with self.subTest(frame=target, key=key), self.assertRaises(AssertionError):
                self.assert_schedule(wrong)

    def assert_lean_order(self, state):
        for frame in range(3440, 3520):
            self.assertEqual(state(frame)['lean_leaders'], 0.)
            self.assertEqual(state(frame)['lean_others'], 0.)
        leader_base = state(3272)['lean_leaders']
        others_base = state(3272)['lean_others']
        first_leader = next(f for f in range(3273, 3321)
                            if state(f)['lean_leaders'] > leader_base)
        first_other = next(f for f in range(3273, 3321)
                           if state(f)['lean_others'] > others_base)
        self.assertLess(first_leader, first_other)
        self.assertGreater(state(3320)['lean_leaders'], leader_base)
        self.assertGreater(state(3320)['lean_others'], others_base)

    def test_leaders_bend_first_and_gap_towers_are_upright(self):
        self.assert_lean_order(D.state)
        def simultaneous(frame):
            data = D.state(frame).copy()
            data['lean_others'] = data['lean_leaders']
            return data
        def still_bent(frame):
            data = D.state(frame).copy()
            if frame >= 3440:
                data['lean_leaders'] = .1
            return data
        for wrong in (simultaneous, still_bent):
            with self.assertRaises(AssertionError):
                self.assert_lean_order(wrong)

    def test_state_rejects_frames_owned_by_other_shots(self):
        def oracle(state):
            for frame in (2959, 3520, 4240):
                with self.assertRaises(ValueError):
                    state(frame)
        oracle(D.state)
        with self.assertRaises(AssertionError):
            oracle(lambda frame: D.state(2960))


class MissingLetters(unittest.TestCase):
    @staticmethod
    def premature_incomplete_write(frame):
        """The pre-fix formula: leaks before closure and stops short at 2π."""
        def written(theta):
            half = np.deg2rad(55.) / 2.
            missing = (theta < half) | (theta > 2. * np.pi - half)
            passed = D.smoothstep(-.035, .055, 2. * np.pi * D.state(frame)['seam'] - theta)
            return np.where(missing, passed, 1.)
        return written

    @staticmethod
    def written(frame):
        """Capture the state supplied to real shading without rasterizing."""
        captured = []
        def capture(cam, width, height, rotation, centre, scale, state, env, **kwargs):
            captured.append(state)
            return (np.zeros((height, width, 3), np.float32),
                    np.zeros((height, width), np.float32),
                    np.full((height, width), np.inf, np.float32))
        with patch.object(D.RS, 'render', side_effect=capture), \
             patch.object(D, 'inscription', return_value=object()):
            D.ring_layer(frame, None, 2, 2)
        return captured[0].write

    def test_new_metal_has_no_written_letters_before_the_ends_touch(self):
        def oracle(written):
            for frame in (3228., 3231.):
                lo, hi = D.theta_range(frame)
                # Both samples are on extant metal within the original gap.
                np.testing.assert_allclose(written(frame)(np.array([lo + .0001, hi - .0001])),
                                           0., atol=0., rtol=0.)
        oracle(self.written)
        with self.assertRaises(AssertionError):
            oracle(lambda frame: lambda theta: np.ones_like(theta))
        with self.assertRaises(AssertionError):
            oracle(self.premature_incomplete_write)

    def test_old_letters_remain_lit_while_missing_letters_follow_the_weld(self):
        def oracle(written):
            old_half = np.deg2rad(55.) / 2.
            old = np.linspace(old_half + .01, 2. * np.pi - old_half - .01, 101)
            for frame in (2960, 3231, 3240, 3264, 3439, 3440):
                np.testing.assert_allclose(written(frame)(old), 1., atol=0., rtol=0.)
            # At this time the weld has crossed the first half of the missing
            # interval; its other half is still ahead of the travelling front.
            np.testing.assert_allclose(written(3240)(np.array([.12, 2. * np.pi - .12])),
                                       [1., 0.], atol=0., rtol=0.)
        oracle(self.written)
        for value in (0., 1.):
            with self.assertRaises(AssertionError):
                oracle(lambda frame: lambda theta: np.full_like(theta, value))

    def test_completed_weld_leaves_the_entire_missing_interval_written(self):
        half = np.deg2rad(55.) / 2.
        missing = np.r_[np.linspace(0., half, 129),
                        np.linspace(2. * np.pi - half, 2. * np.pi, 129)]
        def oracle(written):
            for frame in (3264, 3296, 3439):
                np.testing.assert_allclose(written(frame)(missing), 1., atol=0., rtol=0.)
        oracle(self.written)
        with self.assertRaises(AssertionError):
            oracle(lambda frame: lambda theta: np.where(theta > 2. * np.pi - .01, .3, 1.))
        with self.assertRaises(AssertionError):
            oracle(self.premature_incomplete_write)


class CameraMotion(unittest.TestCase):
    def assert_no_held_frames(self, camera_at):
        for frame in range(2961, 3520):
            if frame in (3200, 3440):
                continue  # Shot cuts do not constrain camera continuity.
            before, after = camera_at(frame - 1), camera_at(frame)
            movement = max(float(np.max(np.abs(after.pos - before.pos))),
                           float(np.max(np.abs(after.R - before.R))))
            self.assertGreater(movement, 1e-12, f'Camera held at D{frame - 1}–{frame}')

    @staticmethod
    def old_camera(frame):
        """Negative control: hold before the Eye push and the gap pullback."""
        if frame >= 3440:
            pull = float(D.smootherstep(3456., 3519., frame))
            close = D.gap_camera()
            wide = D._camera(135., D.RING_C - 14. * D.UP)
            pos = close.pos * (1 - pull) + wide.pos * pull
            target = close.target * (1 - pull) + wide.target * pull
            return Camera(pos, target, hfov=46.)
        if frame >= 3200:
            q = float(D.smootherstep(3296., 3439., frame))
            return D._camera(135. * (1 - q) + D.match_geometry()['eye_distance'] * q,
                             D.RING_C - 14. * (1 - q) * D.UP)
        return D.camera(frame)

    def test_every_adjacent_frame_moves_until_the_later_crowns_shot(self):
        self.assert_no_held_frames(D.camera)
        for first, last in ((3200, 3440), (3440, 3520)):
            def held(frame):
                return self.old_camera(frame) if first <= frame < last else D.camera(frame)
            with self.subTest(first=first), self.assertRaises(AssertionError):
                self.assert_no_held_frames(held)


class WholeBandFraming(unittest.TestCase):
    def assert_whole_band_visible(self, camera_at, frames):
        for frame in frames:
            theta, psi = np.meshgrid(np.linspace(*D.theta_range(frame), 129),
                                     np.linspace(-np.pi, np.pi, 65))
            p, _ = D.RS.local_points(theta.ravel(), psi.ravel())
            world = D.RING_C + D.RING_SIZE * p @ D.ring_rotation(frame).T
            x, y, z = camera_at(frame).project(world, 1920, 804)
            self.assertGreater(float(z.min()), 0., frame)
            self.assertGreaterEqual(min(float(x.min()), 1919. - float(x.max()),
                                        float(y.min()), 803. - float(y.max())), 0., frame)

    def test_entire_physical_band_fits_every_brink_and_vision_frame(self):
        self.assert_whole_band_visible(D.camera, range(2960, 3440))
        def cropped(frame):
            cam = D.camera(frame)
            return Camera(cam.pos, cam.target, hfov=4.)
        with self.assertRaises(AssertionError):
            self.assert_whole_band_visible(cropped, [2960, 3439])
        def behind_lens(frame):
            cam = D.camera(frame)
            return Camera(cam.pos, 2. * cam.pos - cam.target, hfov=cam.hfov)
        with self.assertRaises(AssertionError):
            self.assert_whole_band_visible(behind_lens, [2960])


class CircularInscriptionCoverage(unittest.TestCase):
    def assert_all_source_ink_mapped(self, circular):
        for face, sign in (('outer', -1.), ('inner', 1.)):
            strip = circular.source.lv[face][0]
            ink_columns = np.flatnonzero(np.any(strip > 1e-6, axis=0))
            import d_inscription
            anchor = d_inscription.TH0 / (2. * np.pi)
            ink_angles = anchor + (sign * (ink_columns + .5) / strip.shape[1] - anchor) % 1.
            captured = []
            def source_sample(face, u, v, footprint):
                captured.append(np.asarray(u))
                return np.zeros_like(u)
            # Exercise the wrapper's real mapping at both sides of its wrap.
            # The final source glyph extends beyond the opening shot's cut.
            with patch.object(circular.source, 'sample', side_effect=source_sample):
                circular.sample(face, sign * np.array([0., 1. - 1e-10]),
                                np.array([.5, .5]), np.ones(2))
            covered = anchor + (sign * captured[0] - anchor) % 1.
            self.assertLessEqual(float(covered.min()), float(ink_angles.min()), face)
            self.assertGreaterEqual(float(covered.max()), float(ink_angles.max()), face)

    def test_complete_ring_maps_both_halves_of_last_glyph_and_all_existing_ink(self):
        import d_inscription
        circular = D.CircularInscription()
        self.assert_all_source_ink_mapped(circular)
        old = copy(circular)
        old.offset = d_inscription.TH0 / (2. * np.pi)
        old.extent = d_inscription.EXTENT / (2. * np.pi)
        with self.assertRaises(AssertionError):
            self.assert_all_source_ink_mapped(old)
        clipped_start = copy(circular)
        clipped_start.offset += .03
        clipped_start.extent -= .03
        with self.assertRaises(AssertionError):
            self.assert_all_source_ink_mapped(clipped_start)


def projected_slit_bounds(camera, width_scale=1.):
    """Independent polygonal slit outline before antialiasing and bloom."""
    match = D.match_geometry()
    hh = match['half_height']
    yy = np.linspace(-hh, hh, 32769)
    xx = match['half_width'] * width_scale * np.clip((hh - np.abs(yy)) / (.12 * hh), 0., 1.)
    normal = D.ring_rotation(3439)[:, 1]
    ex = np.cross(D.UP, normal)
    ex /= np.linalg.norm(ex)
    local = np.concatenate((xx[:, None] * ex + yy[:, None] * D.UP,
                            -xx[:, None] * ex + yy[:, None] * D.UP))
    world = D.RING_C + D.RS.R_IN * D.RING_SIZE * local
    u, v, _ = camera.project(world, 1920, 804)
    return np.array([u.min(), v.min(), u.max(), v.max()])


def projected_gap_bounds(camera):
    """Project physical end profiles at D2560, independent of aperture_bounds."""
    boxes = []
    for theta in D.theta_range(3440):
        psi = np.linspace(-np.pi, np.pi, 4097)
        p, _ = D.RS.local_points(np.full_like(psi, theta), psi)
        world = D.RING_C + D.RING_SIZE * p @ D.ring_rotation(3440).T
        u, v, _ = camera.project(world, 1920, 804)
        boxes.append(np.array([u.min(), v.min(), u.max(), v.max()]))
    top, bottom = sorted(boxes, key=lambda box: box[1])
    return np.array([max(top[0], bottom[0]), top[3],
                     min(top[2], bottom[2]), bottom[1]])


class SlitGapMatch(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.eye_cam = D.camera(3439.)
        cls.gap_cam = D.camera(3440.)
        cls.slit = projected_slit_bounds(cls.eye_cam)
        cls.gap = projected_gap_bounds(cls.gap_cam)

    def assert_match(self, slit, gap):
        np.testing.assert_allclose(slit, gap, atol=.03, rtol=0.)

    def test_actual_projected_contour_matches_actual_cut_faces(self):
        self.assert_match(self.slit, self.gap)
        offset = D.RIGHT * .25
        moved = Camera(self.gap_cam.pos, self.gap_cam.target + offset,
                       hfov=self.gap_cam.hfov)
        with self.assertRaises(AssertionError):
            self.assert_match(self.slit, projected_gap_bounds(moved))
        with self.assertRaises(AssertionError):
            self.assert_match(projected_slit_bounds(self.eye_cam, width_scale=1.1), self.gap)

    def test_metadata_describes_centered_vertical_aperture(self):
        def oracle(bounds):
            np.testing.assert_allclose(bounds, self.gap, atol=.03, rtol=0.)
            np.testing.assert_allclose((bounds[:2] + bounds[2:]) / 2.,
                                       [959.5, 401.5], atol=.03, rtol=0.)
            width, height = bounds[2:] - bounds[:2]
            self.assertGreater(height, width)
            self.assertGreater(width, 0.)
        oracle(np.asarray(D.match_geometry()['bounds']))
        for wrong in (self.gap + [5., 0., 5., 0.],
                      self.gap[[1, 0, 3, 2]], self.gap[[2, 1, 0, 3]]):
            with self.assertRaises(AssertionError):
                oracle(wrong)



class BrinkGlare(unittest.TestCase):
    def test_shared_glare_pulses_at_the_gap_and_is_absent_from_eye_and_gap_shots(self):
        def oracle(glare):
            camera = D.camera(3080.)
            high = glare(3080., camera, 480, 201)
            low = glare(3089., camera, 480, 201)
            self.assertGreater(float(high.max()), 1.5 * float(low.max()))
            xx, yy, _ = camera.project(D.endpoints(3080.), 480, 201)
            peak = np.unravel_index(np.argmax(high[..., 0]), high.shape[:2])
            self.assertLess(abs(float(peak[1]) - float(xx.mean())), 2.)
            self.assertLess(abs(float(peak[0]) - float(yy.mean())), 2.)
            for f in (3200., 3320., 3439., 3440., 3519.):
                np.testing.assert_array_equal(glare(f, camera, 480, 201), 0.)
        oracle(D.glare_layer)
        with self.assertRaises(AssertionError):
            oracle(lambda f, cam, w, h: D.glare_layer(3080., cam, w, h))
        def shifted(f, cam, w, h):
            return np.roll(D.glare_layer(f, cam, w, h), 12, axis=1)
        with self.assertRaises(AssertionError):
            oracle(shifted)
        def leaking(f, cam, w, h):
            return D.glare_layer(f, cam, w, h) + (.1 if f >= 3200 else 0.)
        with self.assertRaises(AssertionError):
            oracle(leaking)


class RoundTwoContracts(unittest.TestCase):
    def test_v2_ranges_and_finished_eye_reading_interval(self):
        def oracle(shots, state):
            self.assertEqual(shots, {'brink': (2960, 3200), 'vision': (3200, 3440),
                                     'gap': (3440, 3520)})
            for frame in range(3296, 3344):
                self.assertEqual(state(frame)['eye'], 1., frame)
                self.assertEqual(state(frame)['opening'], 1., frame)
                self.assertEqual(state(frame)['seam'], 1., frame)
                self.assertEqual(state(frame)['push'], 0., frame)
            self.assertGreater(state(3345)['push'], 0.)
        oracle(D.SHOTS, D.state)
        with self.assertRaises(AssertionError):
            oracle({'brink': (2240, 2400), 'vision': (2400, 2560), 'gap': (2560, 2640)}, D.state)
        def premature(frame):
            value = D.state(frame).copy()
            value['push'] = float(D.smootherstep(3296., 3439., frame))
            return value
        with self.assertRaises(AssertionError):
            oracle(D.SHOTS, premature)
        def unfinished(frame):
            value = D.state(frame).copy()
            value['eye'] = .9
            return value
        with self.assertRaises(AssertionError):
            oracle(D.SHOTS, unfinished)

    def test_eye_uses_every_written_atlas_item_without_organic_shader(self):
        def oracle(source, bank):
            import ast
            tree = ast.parse(source)
            imports = [node.name for node in ast.walk(tree) if isinstance(node, ast.alias)]
            self.assertNotIn('c_eye', imports)
            self.assertNotIn('eye3', imports)
            calls = [node.func.attr for node in ast.walk(tree)
                     if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)]
            self.assertNotIn('_eye', calls)
            self.assertEqual(bank['indices'], tuple(D.inscription().source.indices))
            atlas = D.glyphs3.load_atlas()
            self.assertEqual(set(bank['categories']), set(atlas['cats']))
            self.assertEqual(bank['categories'], tuple(atlas['cats'][int(atlas['cat'][i])]
                                                      for i in bank['indices']))
            self.assertEqual(bank['texts'], tuple(atlas['text'][i] for i in bank['indices']))
            self.assertEqual(len(bank['masks']), len(bank['indices']))
            self.assertTrue(all(np.any(mask > 0.) for mask in bank['masks']))
        source = Path(D.__file__).read_text()
        bank = D.glyph_bank()
        oracle(source, bank)
        with self.assertRaises(AssertionError):
            oracle(source + '\nimport c_eye\nc_eye._eye()\n', bank)
        repeated_category_removed = dict(bank)
        repeated_category_removed['categories'] = bank['categories'][:-1]
        with self.assertRaises(AssertionError):
            oracle(source, repeated_category_removed)
        for key in ('indices', 'categories', 'masks'):
            broken = dict(bank)
            broken[key] = bank[key][1:]
            with self.assertRaises(AssertionError):
                oracle(source, broken)

    def test_writing_moves_inward_and_has_only_ice_white_emission(self):
        def oracle(streams, eye_layer):
            xy0, cat0, _ = streams(3301.)
            xy1, cat1, _ = streams(3302.)
            np.testing.assert_array_equal(cat0, cat1)
            moved = np.linalg.norm(xy1 - xy0, axis=1)
            self.assertTrue(np.all(moved > 0.))
            radius0, radius1 = np.linalg.norm(xy0, axis=1), np.linalg.norm(xy1, axis=1)
            # One wrap per stream cycle is expected; the rest move inward.
            self.assertGreater(float(np.mean(radius1 < radius0)), .8)
            rgb, alpha, _ = eye_layer(3330., D.camera(3330.), 480, 201)
            self.assertGreater(float(rgb.max()), 0.)
            active = rgb[..., 2] > .01
            np.testing.assert_allclose(rgb[active, 0] / rgb[active, 2], D.ICE[0], atol=1e-6)
            np.testing.assert_allclose(rgb[active, 1] / rgb[active, 2], D.ICE[1], atol=1e-6)
            self.assertGreater(float(alpha.max()), .9)
        oracle(D.glyph_streams, D.eye_layer)
        with self.assertRaises(AssertionError):
            oracle(lambda f: D.glyph_streams(3301.), D.eye_layer)
        def red_eye(*args):
            rgb, alpha, void = D.eye_layer(*args)
            rgb[..., 0] = rgb[..., 2] * 2.
            return rgb, alpha, void
        with self.assertRaises(AssertionError):
            oracle(D.glyph_streams, red_eye)

    def test_band_dimensions_and_start_pose_are_shared_with_race(self):
        def oracle(centre, size, rotation):
            import c_d
            np.testing.assert_array_equal(centre, c_d.RING_C)
            self.assertEqual(size, 2.2)
            # Material-gap bisector re-expression preserves the same pose.
            np.testing.assert_allclose(rotation[:, 1], D.race_rotation()[:, 1], atol=1e-12)
            self.assertAlmostEqual(float(np.linalg.det(rotation)), 1., places=12)
        oracle(D.RING_C, D.RING_SIZE, D.ring_rotation(2960))
        for centre, size, rotation in ((D.RING_C - [0., 5., 0.], D.RING_SIZE, D.ring_rotation(2960)),
                                       (D.RING_C, 3.5, D.ring_rotation(2960)),
                                       (D.RING_C, D.RING_SIZE, D.ROT)):
            with self.assertRaises(AssertionError):
                oracle(centre, size, rotation)

if __name__ == '__main__':
    unittest.main()
