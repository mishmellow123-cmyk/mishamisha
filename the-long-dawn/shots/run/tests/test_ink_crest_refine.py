"""Real C13 crest rays, retaining the marcher's complete vertical history."""
import os
from pathlib import Path
import sys
import unittest
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import render_ink as RI
import reveal_pair_c_v5 as RP


FLAG = 'LD_INK_CREST_REFINE'


class MarchReached(Exception):
    pass


class InkCrestRefineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.shot = RP.make_shot()

    def march_args(self, frame, enabled):
        # Capture the real entrypoint at delivered resolution before shading.
        with mock.patch.dict(os.environ, {FLAG: enabled}):
            with mock.patch.object(RI.WD, 'march', side_effect=MarchReached) as march:
                with self.assertRaises(MarchReached):
                    RI.render_aov(self.shot, RP.local_frame(frame), scale=1., ss=2.)
        self.assertEqual(march.call_count, 1)
        args = list(march.call_args.args)
        args[9] = None  # Release the full-frame allocation; tests march a strip.
        return args

    def crest_strip(self, frame, enabled):
        args = self.march_args(frame, enabled)
        target = self.shot.cam(RP.local_frame(frame)).scaled(2.)
        source = RI.RC.source_for(target)
        # Measured C3000..3002 fault: target (1123,920) at ss=2 jumps
        # between the ~4.5 km crest and the ~11.4 km ridge behind it.
        x = np.arange(1116, 1136, dtype=float)
        direction = target.ray(x + .5, np.full(x.shape, 920.5))
        forward = direction @ source.fwd
        # Match rcam.warp_maps and INTER_NEAREST, including float32 maps.
        mx = (source.cx + source.f * (direction @ source.right) / forward - .5).astype(np.float32)
        my = (source.cyy - source.f * direction[:, 1] / forward - .5).astype(np.float32)
        cols, rows = np.rint(mx).astype(int), np.rint(my).astype(int)
        left, right = int(cols.min()), int(cols.max()) + 1
        args[2] = args[2].copy()
        args[2][8] -= left
        args[2][10] = right - left
        # Cropping vertically would reset da/ha and hide the actual failure.
        args[9] = np.zeros((source.H, right - left))
        return args, rows, cols - left

    def assert_continuous_near_crest(self, depths):
        self.assertTrue(np.all((depths > 4000.) & (depths < 5000.)), depths)
        self.assertLess(float(np.max(np.abs(np.diff(depths, axis=1)))), 2.)
        self.assertLess(float(np.max(np.abs(np.diff(depths, axis=0)))), 1.)

    def assert_aov_bytes_equal(self, actual, expected, omitted=()):
        self.assertEqual(actual.keys(), expected.keys())
        for key in actual:
            if key in omitted:
                continue
            if isinstance(actual[key], np.ndarray):
                self.assertEqual(actual[key].dtype, expected[key].dtype, key)
                self.assertEqual(actual[key].shape, expected[key].shape, key)
                self.assertEqual(actual[key].tobytes(), expected[key].tobytes(), key)
            elif key == 'cam':
                self.assert_aov_bytes_equal(actual[key], expected[key])
            else:
                self.assertEqual(actual[key], expected[key], key)

    def test_refinement_recovers_crest_across_actual_flicker_frames(self):
        old, fixed, reference = [], [], []
        for frame in (3000, 3001, 3002):
            args, rows, cols = self.crest_strip(frame, '1')
            # A still finer independent march checks convergence on the near
            # crest; no cached AOV or invented depth fixture enters the test.
            for output, knobs in ((fixed, None), (old, (.0035, .35)),
                                  (reference, (.00001, .025))):
                sample = list(args)
                if knobs is not None:
                    sample[5:7] = knobs
                sample[9] = np.zeros_like(args[9])
                RI.WD.march(*sample)
                output.append(sample[9][rows, cols])
        old, fixed, reference = map(np.asarray, (old, fixed, reference))
        self.assert_continuous_near_crest(fixed)
        np.testing.assert_allclose(fixed, reference, rtol=0., atol=.05)
        # Negative control: this assertion must reject the real legacy rays.
        with self.assertRaises(AssertionError):
            self.assert_continuous_near_crest(old)
        # Pin the reported temporal failure, not just a spatial difference.
        point = 1123 - 1116
        np.testing.assert_array_equal(old[:, point] > 10000., [True, False, True])
        self.assertGreater(float(np.max(np.abs(np.diff(old[:, point])))), 6000.)

    def test_only_opt_in_changes_the_actual_march_arguments(self):
        old = self.march_args(3001, '0')
        fixed = self.march_args(3001, '1')
        self.assertEqual(old[3:9], [.2, 90000., .0035, .35, 900., 9])
        self.assertEqual(fixed[5:7], [.0001, .1])
        for index in (0, 1, 2):
            np.testing.assert_array_equal(fixed[index], old[index])
        self.assertEqual(fixed[3:5], old[3:5])
        self.assertEqual(fixed[7:9], old[7:9])

    def test_default_aov_is_byte_identical_to_literal_legacy_march(self):
        original_march = RI.WD.march

        def legacy(P, CR, C, d0, dmax, rel, kgap, hmax, nbis, out_d):
            return original_march(P, CR, C, .2, 90000., .0035, .35, 900., 9, out_d)

        # Exercise all AOV generation and camera warps at a small resolution.
        # The literal wrapper retains the exact pre-change march settings.
        with mock.patch.dict(os.environ, {FLAG: '0'}):
            with mock.patch.object(RI.WD, 'march', side_effect=legacy) as march:
                expected = RI.render_aov(self.shot, RP.local_frame(3001), scale=.025, ss=1.)
        self.assertEqual(march.call_count, 1)
        for value in (None, '0'):
            with self.subTest(flag=value), mock.patch.dict(os.environ):
                if value is None:
                    os.environ.pop(FLAG, None)
                else:
                    os.environ[FLAG] = value
                actual = RI.render_aov(self.shot, RP.local_frame(3001), scale=.025, ss=1.)
            self.assert_aov_bytes_equal(actual, expected)

    def test_omitting_motion_vectors_skips_velocity_and_preserves_other_aov_bytes(self):
        for flag in ('0', '1'):
            with self.subTest(flag=flag), mock.patch.dict(os.environ, {FLAG: flag}):
                args = (self.shot, RP.local_frame(3001), .025, 1.)
                expected = RI.render_aov(*args)
                self.assertIsInstance(expected['V'], np.ndarray)
                self.assertEqual(expected['V'].shape, expected['A'].shape[:2] + (2,))
                with mock.patch.object(RI.RC, 'velocity', side_effect=RuntimeError('velocity requested')) as velocity:
                    # Negative control: enabled velocity really reaches the
                    # guarded allocation; returning V=None alone is insufficient.
                    with self.assertRaisesRegex(RuntimeError, 'velocity requested'):
                        RI.render_aov(*args, motion_vectors=True)
                    velocity.assert_called_once()
                    velocity.reset_mock()
                    actual = RI.render_aov(*args, motion_vectors=False)
                    velocity.assert_not_called()
                self.assertIsNone(actual['V'])
                self.assert_aov_bytes_equal(actual, expected, omitted=('V',))

    def test_paired_driver_omits_velocity_only_with_fix_and_preserves_composed_bytes(self):
        import inkpass as IP

        for flag in (None, '0', '1'):
            with self.subTest(flag=flag), mock.patch.dict(os.environ):
                if flag is None:
                    os.environ.pop(FLAG, None)
                else:
                    os.environ[FLAG] = flag
                # Real AOV + ink composition using the old velocity-producing
                # call establishes the image reference for each march setting.
                reference = RI.render_aov(self.shot, RP.local_frame(3001), .025, 1.)
                reference['kpx'] = .025
                expected, _ = IP.compose(reference, B=self.shot.B, CR=self.shot.CR)
                with mock.patch.object(RI, 'render_aov', wraps=RI.render_aov) as render:
                    actual = RP.render(3001, self.shot, scale=.025, ss=1.)
                kwargs = {'motion_vectors': False} if flag == '1' else {}
                render.assert_called_once_with(self.shot, RP.local_frame(3001), .025, 1., **kwargs)
                self.assertEqual(actual.shape, expected.shape)
                self.assertEqual(actual.dtype, expected.dtype)
                self.assertEqual(actual.tobytes(), expected.tobytes())

    @staticmethod
    def synthetic_crest():
        # Explicit test geometry: two land planes separated by a depth step.
        # These arrays are fixtures, not measured production-frame data.
        A = np.zeros((48, 128, 5), np.float32)
        A[:, :64, 1] = 1000.
        A[:, 64:, 1] = 2000.
        return A

    def test_hatch_filter_preserves_nonland_and_pixels_beyond_crest_band(self):
        import inkpass as IP

        A = self.synthetic_crest()
        A[:8, :, 0] = 2.  # Cloud.
        A[-8:, :, 0] = 3.  # Sky.
        hat = np.random.default_rng(29).random((*A.shape[:2], 2)).astype(np.float32)
        for kpx in (1., 2.):
            with self.subTest(kpx=kpx):
                actual = IP._crest_hatch_aa(A, hat, kpx)
                self.assertEqual(actual.shape, hat.shape)
                self.assertEqual(actual.dtype, hat.dtype)
                self.assertTrue(np.isfinite(actual).all())
                np.testing.assert_array_equal(actual[A[..., 0] >= 1.5], hat[A[..., 0] >= 1.5])
                # The near-side edge is x=63. The radius is in page pixels,
                # so both source scales must leave these columns untouched.
                far_columns = np.abs(np.arange(128) - 63) >= 16 * kpx
                np.testing.assert_array_equal(actual[:, far_columns], hat[:, far_columns])
        # A 4% depth change does not activate the >5% crest detector.
        A[:, 64:, 1] = 1040.
        np.testing.assert_array_equal(IP._crest_hatch_aa(A, hat, 1.), hat)

    def test_hatch_filter_preserves_constants_and_separates_depth_surfaces(self):
        import cv2
        import inkpass as IP

        A = self.synthetic_crest()
        constant = np.broadcast_to(np.array([.25, .75], np.float32), (48, 128, 2)).copy()
        np.testing.assert_array_equal(IP._crest_hatch_aa(A, constant, 1.), constant)
        two_surfaces = constant.copy()
        two_surfaces[:, :64] = [1., .25]
        two_surfaces[:, 64:] = [0., .75]
        actual = IP._crest_hatch_aa(A, two_surfaces, 1.)
        np.testing.assert_array_equal(actual, two_surfaces)
        # Negative control: an ordinary Gaussian crosses this exact boundary.
        crossing = cv2.GaussianBlur(two_surfaces, (0, 0), .7)
        with self.assertRaises(AssertionError):
            np.testing.assert_array_equal(crossing[:, 62:66], two_surfaces[:, 62:66])

    def test_hatch_filter_reduces_subpixel_jitter_in_synthetic_crest_fixture(self):
        import inkpass as IP

        A = self.synthetic_crest()
        yy, xx = np.mgrid[:48, :128]
        # A half-pixel translation of explicit near-Nyquist hatch signals.
        # This measures the filter response, not the film's grain target.
        frames = [np.stack((.5 + .45 * np.cos(2 * np.pi * .45 * (xx - phase)),
                            .5 + .45 * np.cos(2 * np.pi * .45 * (yy - phase))), axis=-1).astype(np.float32)
                  for phase in (0., .5)]
        filtered = [IP._crest_hatch_aa(A, frame, 1.) for frame in frames]
        patch = np.s_[12:36, 56:63]
        old_difference = float(np.abs(frames[1][patch] - frames[0][patch]).mean())
        new_difference = float(np.abs(filtered[1][patch] - filtered[0][patch]).mean())
        self.assertGreater(old_difference, 0.)

        def assert_jitter_reduced(value):
            self.assertLess(value, .5 * old_difference)

        assert_jitter_reduced(new_difference)
        # The actual unfiltered fixture must fail the same improvement gate.
        with self.assertRaises(AssertionError):
            assert_jitter_reduced(old_difference)

    def test_compose_hatch_filter_requires_opt_in_and_default_bytes_match_bypass(self):
        import inkpass as IP

        with mock.patch.dict(os.environ, {FLAG: '0'}):
            aov = RI.render_aov(self.shot, RP.local_frame(3001), .025, 1.)
        aov['kpx'] = .025
        # Bypass only the new operation to reconstruct the pre-filter compose.
        with mock.patch.dict(os.environ, {FLAG: '1'}):
            with mock.patch.object(IP, '_crest_hatch_aa', side_effect=lambda A, hat, kpx: hat) as helper:
                expected, _ = IP.compose(aov, B=self.shot.B, CR=self.shot.CR)
            helper.assert_called_once()
        for flag in (None, '0'):
            with self.subTest(flag=flag), mock.patch.dict(os.environ):
                if flag is None:
                    os.environ.pop(FLAG, None)
                else:
                    os.environ[FLAG] = flag
                with mock.patch.object(IP, '_crest_hatch_aa', side_effect=RuntimeError('crest filter reached')) as helper:
                    actual, _ = IP.compose(aov, B=self.shot.B, CR=self.shot.CR)
                    helper.assert_not_called()
                self.assertEqual(actual.dtype, expected.dtype)
                self.assertEqual(actual.shape, expected.shape)
                self.assertEqual(actual.tobytes(), expected.tobytes())
        # Negative control: enabling the option reaches the guarded operation.
        with mock.patch.dict(os.environ, {FLAG: '1'}):
            with mock.patch.object(IP, '_crest_hatch_aa', side_effect=RuntimeError('crest filter reached')) as helper:
                with self.assertRaisesRegex(RuntimeError, 'crest filter reached'):
                    IP.compose(aov, B=self.shot.B, CR=self.shot.CR)
                helper.assert_called_once()


if __name__ == '__main__':
    unittest.main()
