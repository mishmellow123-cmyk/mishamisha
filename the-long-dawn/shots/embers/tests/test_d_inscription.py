"""D inscription's v2 timing, live fire, atlas coverage and isolated opt-in."""
from pathlib import Path
import sys
import unittest

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / 'shots' / 'embers'), str(ROOT / 'lib')]
import d_inscription as D
import ringsolid as R
from core import Camera


# Delivered openring.forging-opening.v1 contract; explicit source pin.
# SHA-256 c3be8d09aea492447c1f45744f5988c26a323d48bb2bf14cb2587a2be8b17533
CONTRACT = {'frame': 2080,
 'native_size': [1920, 804],
 'camera': {'position': [-12.896215704688116, 18.0, 149.44459716061385],
            'target': [0.0, 13.0, 0.0],
            'horizontal_fov_degrees': 62.0,
            'focus': 152.5909564816998,
            'aperture': 0.012},
 'band': {'centre': [0.0, 38.0, 0.0],
          'rotation_matrix': [[-0.6872480717197039, -0.5744632015400474, 0.44461457240390556],
                              [-0.6655996975676453, 0.7431448254773944, -0.06864991597979037],
                              [-0.29097616829460116, -0.343114847307038, -0.8930873815266119]],
          'rotation_convention': 'row-vector local point @ rotation_matrix.T; world = centre + '
                                 'scale * rotated_point',
          'scale': 2.2,
          'gap_degrees': 120.0,
          'theta_range_radians': [-0.5999999999999996, 3.588790204786391]},
 'central_fire': {'position': [0.0, -14.0, 0.0], 'tip': [0.0, -7.5, 0.0]}}

class Inscription(unittest.TestCase):
    def rejects(self, oracle, value):
        with self.assertRaises(AssertionError):
            oracle(value)

    def test_arc_stops_at_120_degrees_and_never_closes(self):
        def oracle(fn):
            values = np.array([fn(f) for f in range(D.START, D.END)])
            self.assertTrue(np.all(np.diff(values) <= 1e-10))
            self.assertTrue(np.all(values >= 120. - 1e-10))
            np.testing.assert_allclose(values[D.STOP - D.START:], 120., atol=1e-10)
            self.assertGreater(values[D.STOP - D.START - 1], 120.)
        oracle(D.gap_deg)
        self.rejects(oracle, lambda f: max(0., D.gap_deg(f) - max(f - D.STOP, 0)))

    def test_every_atlas_category_contributes_real_ink_before_the_cut(self):
        texture = D.inscription()
        atlas = D.glyphs3.load_atlas()
        def oracle(indices):
            self.assertEqual(set(atlas['cat'][indices[:-1]]), set(range(len(atlas['cats']))))
            self.assertEqual(len(indices[:-1]), len(atlas['cats']))
            self.assertTrue(np.all(atlas['cnt'][indices] > 0))
            self.assertEqual(atlas['text'][indices[-1]], '光')
        oracle(texture.indices)
        changed = texture.indices.copy()
        changed[3] = changed[0]
        self.rejects(oracle, changed)
        self.assertEqual(texture.categories, tuple(atlas['cats']))

    def test_last_glyph_is_bisected_at_ink_mass_median(self):
        tex = D.inscription()
        def oracle(fraction):
            self.assertGreater(fraction, .40)
            self.assertLess(fraction, .60)
        oracle(tex.final_ink_fraction)
        self.rejects(oracle, 1.)
        # The stop lies inside actual outer-face ink, not merely its bounding box.
        u = np.full(100, (-D.theta_range(D.STOP)[1] / (2 * np.pi)) % 1.)
        v = np.linspace(.30, .70, 100)
        ink = tex.sample('outer', u, v, np.ones(100))
        self.assertGreater(ink.max(), .1)

    def test_front_contrast_uses_identical_material_uvs_at_every_front(self):
        tex = D.inscription()
        def oracle(provider):
            for frame in (1784, 1860, 1930, D.STOP, D.END - 1):
                front = D.theta_range(frame)[1]
                th = np.linspace(front - .12, front, 101)
                for face, sign in (('outer', -1.), ('inner', 1.)):
                    midpoint, _ = provider(front).leading_glyph(face)
                    sampled_uv = (midpoint + sign * (th - front) / (2 * np.pi)) % 1.
                    material_uv = (sign * th / (2 * np.pi)) % 1.
                    np.testing.assert_allclose(sampled_uv, material_uv, rtol=0., atol=1e-14)
        oracle(lambda front: D.FrontInscription(tex, front))
        self.rejects(oracle, lambda front: tex)

    def test_stopped_cut_faces_tonemap_to_near_white(self):
        state = D.ring_state(D.STOP)
        camera = D.camera(D.STOP)
        def oracle(gain):
            state.cap_heat_gain = gain
            for theta, side in zip(D.theta_range(D.STOP), (-1., 1.)):
                color = R.shade_cap(np.array([0.]), np.array([0.]), theta, side,
                                    camera, *D.ring_frame(D.STOP), state, D.environment(D.STOP))
                display = D.look.linear_to_srgb(D.look.tonemap(color, exposure=.79))
                self.assertGreater(float(display.min()), .90)
                self.assertLess(float(display.max() - display.min()), .07)
        oracle(state.cap_heat_gain)
        self.rejects(oracle, 2.4)

    def test_exact_delivered_handoff_contract(self):
        def oracle(contract):
            self.assertEqual(D.END, contract['frame'])
            self.assertEqual(contract['native_size'], [1920, 804])
            cam = D.camera(D.END - 1)
            camera = contract['camera']
            for field, key in (('pos', 'position'), ('target', 'target'), ('hfov', 'horizontal_fov_degrees'),
                               ('focus', 'focus'), ('aperture', 'aperture')):
                np.testing.assert_array_equal(getattr(cam, field), camera[key])
            rotation, centre, size = D.ring_frame(D.END - 1)
            band = contract['band']
            np.testing.assert_array_equal(rotation, band['rotation_matrix'])
            np.testing.assert_array_equal(centre, band['centre'])
            self.assertEqual(size, band['scale'])
            np.testing.assert_array_equal(D.theta_range(D.END - 1), band['theta_range_radians'])
            self.assertAlmostEqual(D.gap_deg(D.END - 1), band['gap_degrees'])
            fire = contract['central_fire']
            np.testing.assert_array_equal(D.FORGE.c3.FIRE_ROOT, fire['position'])
            np.testing.assert_array_equal(D.FORGE.c3.FIRE_ROOT + [0., D.FORGE.c3.HF, 0.], fire['tip'])
        oracle(CONTRACT)
        import copy
        for group, field in (('camera', 'position'), ('band', 'centre'), ('central_fire', 'position')):
            bad = copy.deepcopy(CONTRACT)
            bad[group][field][0] += 1.
            self.rejects(oracle, bad)

    def test_handoff_camera_pose_and_gap_match_adjoining_forging(self):
        scene = D.handoff_scene()
        expected_cam = scene.camera(D.END)
        expected_pose = scene.ring_frame(D.END)
        def oracle(camera_fn, pose_fn):
            for frame in range(D.PULL_END, D.END):
                actual = camera_fn(frame)
                for field in ('pos', 'target', 'R', 'hfov', 'focus', 'aperture'):
                    np.testing.assert_array_equal(getattr(actual, field), getattr(expected_cam, field))
                for part, expected in zip(pose_fn(frame), expected_pose):
                    np.testing.assert_array_equal(part, expected)
                np.testing.assert_allclose(D.theta_range(frame), D.FORGE.D.theta_range(D.END), atol=1e-14)
        oracle(D.camera, D.ring_frame)
        def shifted(frame):
            rot, centre, size = D.ring_frame(frame)
            return rot, centre + [0., 1., 0.], size
        self.rejects(lambda fn: oracle(D.camera, fn), shifted)
        self.assertEqual(D.theta_range(D.STOP), D.theta_range(D.STOP + 8))
        def motion_oracle(fn):
            self.assertFalse(np.array_equal(fn(D.STOP).pos, fn(D.STOP + 8).pos))
        motion_oracle(D.camera)
        self.rejects(motion_oracle, lambda frame: D.camera(D.STOP))

    def test_preroll_is_live_reproducible_fire_at_the_drawn_ring(self):
        first = D.render_frame(D.START, scale=.25, save=False, verbose=False)
        repeat = D.render_frame(D.START, scale=.25, save=False, verbose=False)
        later = D.render_frame(D.START + 1, scale=.25, save=False, verbose=False)
        np.testing.assert_array_equal(first, repeat)
        def live_oracle(image):
            self.assertFalse(np.array_equal(first, image))
        live_oracle(later)
        self.rejects(live_oracle, repeat)
        def position_oracle(image):
            mask = (image.max(2) > .4).astype(np.uint8)
            _, _, stats, centres = cv2.connectedComponentsWithStats(mask)
            body = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            self.assertLess(np.linalg.norm(centres[body] / .25 - D.PAGE_RING), 10.)
            self.assertGreater(stats[body, cv2.CC_STAT_WIDTH] / .25, 45.)
            self.assertLess(stats[body, cv2.CC_STAT_WIDTH] / .25, 75.)
        position_oracle(first)
        self.rejects(position_oracle, np.roll(first, 12, axis=1))

    def test_preroll_closes_and_centres_before_the_forming_cut(self):
        start_root, start_tip = D.preroll_fire(D.START)
        def oracle(fn):
            root, tip = fn(D.FORM_START - 1)
            height = np.linalg.norm(tip - root)
            self.assertGreater(height, 2.5 * np.linalg.norm(start_tip - start_root))
            np.testing.assert_allclose(root + (tip - root) * .402, [960., 402.], atol=1e-9)
            for actual, expected in zip(D.front_fire(D.FORM_START), (root, tip)):
                np.testing.assert_array_equal(actual, expected)
        oracle(D.preroll_fire)
        self.rejects(oracle, lambda frame: (start_root, start_tip))

    def test_thinking_fire_uses_a_emitters_and_restores_schedule(self):
        B, F = D.FIRE.B, D.FIRE
        before = B.SCHED, B.IGN, B.BEATS
        fire, sparks = F._emitters()
        self.assertIsInstance(fire, B.MindFire)
        self.assertIsInstance(sparks, B.FireSparks)
        with self.assertRaises(RuntimeError):
            with F.source_schedule():
                self.assertEqual(B._mind_cols(), (B.C_CORE, B.C_MIND_ICE, B.C_MIND_EDGE))
                raise RuntimeError('forced failed draw')
        self.assertIs(B.SCHED, before[0])
        self.assertEqual(B.IGN, before[1])
        self.assertIs(B.BEATS, before[2])
        def clock_oracle(fn):
            self.assertEqual(fn(D.END) - fn(D.END - 1), 1.)
        clock_oracle(F.source_clock)
        self.rejects(clock_oracle, lambda frame: F.source_clock(min(frame, D.END - 1)))

    def test_custom_texture_does_not_mutate_legacy_texture(self):
        cam = Camera((4., 3., 7.), (0., 0., 0.), hfov=47.)
        state = R.RingState()
        state.glow, state.letters = .11, .8
        args = (cam, 128, 80, np.eye(3), np.zeros(3), 1., state, R.Env())
        def draw():
            return R.render(*args, ss=1, nt=64, npp=20)[0]
        original = draw()
        old_texture = R.inscription()
        state.inscription = D.inscription()
        custom = draw()
        self.assertIs(R.inscription(), old_texture)
        self.assertFalse(np.array_equal(original, custom))
        state.inscription = None
        np.testing.assert_array_equal(draw(), original)
        # Negative control: retaining the new texture makes the legacy pin fail.
        def oracle(img):
            np.testing.assert_array_equal(img, original)
        self.rejects(oracle, custom)

    def test_thinking_fire_adds_light_without_cutting_dark_lines(self):
        base = np.full((100, 240, 3), 10., np.float32)
        actual = base.copy()
        D.FIRE.draw(actual, (998., 175.), (998., 25.), D.START, scale=.125)
        def oracle(image):
            self.assertTrue(np.all(image >= base))
            self.assertTrue(np.any(image > base))
        oracle(actual)
        overwritten = actual.copy()
        overwritten[20, 30] = [3.4, 1.8, .34]
        self.rejects(oracle, overwritten)

    def test_cap_heat_gain_changes_only_hot_section_emission(self):
        cam = Camera((5., 2., -2.), (0., 0., 0.))
        state = R.RingState()
        self.assertEqual(state.cap_heat_gain, 1.)
        state.heat = lambda theta: np.full_like(theta, .9)
        r, y = np.linspace(-.15, .15, 40), np.linspace(-.3, .3, 40)
        cap_args = (r, y, .1, -1., cam, np.eye(3), np.zeros(3), 1., state, R.Env())
        th, ps = np.linspace(.1, 1., 40), np.zeros(40)
        band_args = (th, ps, cam, 180., 1., np.eye(3), np.zeros(3), 1., state, R.Env())
        base_cap = R.shade_cap(*cap_args)
        base_band = R.shade(*band_args)
        state.cap_heat_gain = 2.4
        hot_cap = R.shade_cap(*cap_args)
        def oracle(result):
            self.assertTrue(np.all(result >= base_cap))
            self.assertGreater(float(np.max(result - base_cap)), .1)
        oracle(hot_cap)
        self.rejects(oracle, base_cap)
        np.testing.assert_array_equal(R.shade(*band_args), base_band)
        state.heat = None
        cold_boosted = R.shade_cap(*cap_args)
        state.cap_heat_gain = 1.
        np.testing.assert_array_equal(R.shade_cap(*cap_args), cold_boosted)

    def test_explicit_output_and_frame_domain(self):
        with self.assertRaises(ValueError):
            D.render_frame(D.START - 1, save=False)
        with self.assertRaises(ValueError):
            D.render_frame(D.END, save=False)
        with self.assertRaises(ValueError):
            D.render_frame(D.START, scale=0., save=False)
        with self.assertRaises(ValueError):
            D._output_dir(None)
        with self.assertRaises(ValueError):
            D._output_dir(ROOT / 'renders' / 'embers_A3')
        if (ROOT / 'renders').is_symlink():
            with self.assertRaises(ValueError):
                D._output_dir(ROOT / 'renders' / 'embers_D_inscription')
        else:
            self.assertEqual(D._output_dir(ROOT / 'renders' / 'embers_D_inscription'),
                             (ROOT / 'renders' / 'embers_D_inscription').resolve())


if __name__ == '__main__':
    unittest.main()
