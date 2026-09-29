"""C3 held-word camera contracts; geometry and tiny glyph crops only."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import book_c_t1_candidates as C
import t1_camera_projection as P


def camera_state(camera):
    return np.r_[camera.pos, camera.target, camera.hfov]


class HeldCameraContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        C.cv2.setNumThreads(0)
        cls.geometry = P.InscriptionGeometry()
        cls.book = cls.geometry.book
        cls.original = C.make_renderer('current-words', scale=1)
        cls.held = C.make_renderer('current-words-held', scale=1)

    def assert_camera_equal(self, t):
        a = self.original.cam_mountain(self.book, t)
        b = self.held.cam_mountain(self.book, t)
        np.testing.assert_allclose(camera_state(a), camera_state(b), rtol=0, atol=1e-9)
        for name in ('f', 'u', 'r', 'F', 'focus', 'fstop', 'dof_k'):
            if getattr(a, name) is None:
                self.assertIsNone(getattr(b, name))
            else:
                np.testing.assert_allclose(getattr(a, name), getattr(b, name), rtol=0, atol=1e-9)

    def assert_margin(self, renderer, frame, minimum):
        camera = renderer.cam_mountain(self.book, (frame-320)/24.0)
        points = self.geometry.screen(camera, self.geometry.bounds)
        # Use last pixel row 803, stricter than continuous image boundary 804.
        # Two extra pixels cover more than the observed calibration residual;
        # this is a projection reserve, not a rendered darkness guarantee.
        self.assertLessEqual(points[:, 1].max()+2., P.HEIGHT-1-minimum, msg=f'C{frame}')
        self.assertGreaterEqual(points[:, 1].min()-2., 0., msg=f'C{frame} top')
        self.assertGreaterEqual(points[:, 0].min()-2., 0., msg=f'C{frame} left')
        self.assertLessEqual(points[:, 0].max()+2., P.WIDTH-1, msg=f'C{frame} right')

    def test_original_projection_reproduces_supplied_unclipped_rows(self):
        for frame, observed in P.REFERENCE_ROWS.items():
            camera = self.original.cam_mountain(self.book, (frame-320)/24.0)
            projected = self.geometry.screen(camera, self.geometry.glyphs)[:, 1].max()
            self.assertLessEqual(abs(projected-observed), 2., msg=f'C{frame}')
        camera = self.original.cam_mountain(self.book, (515-320)/24.0)
        self.assertGreater(self.geometry.screen(camera, self.geometry.glyphs)[:, 1].max(), 803.)

    def test_camera_is_original_before_arrival_and_from_557_including_9_9(self):
        times = list(np.linspace(-1., C.HELD_START, 80))
        times += [(f-320)/24.0 for f in range(557, 561)]
        times += list(np.linspace(9.9, 12., 100))
        for t in times:
            self.assert_camera_equal(t)

    def test_every_frame_has_40px_through_527_and_is_inside_through_539(self):
        for frame in range(400, 540):
            self.assert_margin(self.held, frame, 40. if frame <= 527 else 0.)

    def test_only_opt_in_mountain_orientation_changes(self):
        held_line = C.make_line('current-words-held')
        current_line = C.make_line('current-words')
        for name in ('a', 'noise', 'dens', 'xx', 'yy'):
            np.testing.assert_array_equal(getattr(held_line, name), getattr(current_line, name))
        for name in ('c0', 'r0', 'f_in', 'f_out', 'ppc', 'text'):
            self.assertEqual(getattr(held_line, name), getattr(current_line, name))
        for name in ('frame', 'shot_mountain', 'mountain', 'cam_letters', 'light', 'finish_layer',
                     'shot_deep', 'shot_last_pages', 'tex_text', 'ink_line'):
            self.assertIs(getattr(C.HeldWordsBook3, name), getattr(C.CandidateBook3, name))
        for frame in range(320, 561):
            t = (frame-320)/24.0
            a = self.original.cam_mountain(self.book, t)
            b = self.held.cam_mountain(self.book, t)
            np.testing.assert_array_equal(a.pos, b.pos)
            np.testing.assert_allclose(a.r, b.r, rtol=0, atol=1e-14)
            self.assertEqual((a.hfov, a.focus, a.fstop, a.dof_k), (b.hfov, b.focus, b.fstop, b.dof_k))
        self.assertIs(C.CandidateBook3.cam_mountain, C.BC.Book3.cam_mountain)
        self.assertGreater(np.linalg.norm(self.original.cam_mountain(self.book, 6.).target -
                                         self.held.cam_mountain(self.book, 6.).target), .1)
        self.assertEqual(self.original.books, {})
        self.assertEqual(self.held.books, {})

    def test_velocity_is_continuous_at_every_camera_and_envelope_join(self):
        # One-sided vector derivatives detect direction changes that a speed
        # magnitude alone misses. Include original key/field-of-view joins.
        joins = (0., .8, 1., C.HELD_START, C.HELD_SETTLE, C.HELD_RELEASE, C.HELD_END, 10.)
        for renderer in (self.original, self.held):
            for t in joins:
                for dt in (1e-5, 1e-6):
                    a = camera_state(renderer.cam_mountain(self.book, t-dt))
                    b = camera_state(renderer.cam_mountain(self.book, t))
                    c = camera_state(renderer.cam_mountain(self.book, t+dt))
                    np.testing.assert_allclose((b-a)/dt, (c-b)/dt, rtol=0, atol=2e-3,
                                               err_msg=f'{type(renderer).__name__} t={t}')
        for t in (C.HELD_START, C.HELD_SETTLE, C.HELD_RELEASE, C.HELD_END):
            dt = 1e-4
            # Correction itself has zero first derivative at all four joins.
            self.assertLess(abs((C.held_tilt(t+dt)-C.held_tilt(t-dt))/(2*dt)), 1e-8)

    def test_equality_guard_rejects_a_late_return(self):
        with patch.object(C, 'HELD_END', 10.1):
            with self.assertRaises(AssertionError):
                self.assert_camera_equal((557-320)/24.0)

    def test_margin_guard_rejects_the_uncorrected_camera(self):
        with self.assertRaises(AssertionError):
            self.assert_margin(self.original, 527, 40.)
        with patch.object(C, 'HELD_MAX_TILT', 0.):
            with self.assertRaises(AssertionError):
                self.assert_margin(self.held, 527, 40.)

    def test_report_has_all_160_frames_and_explicit_absent_ink(self):
        rows, summary = P.audit()
        self.assertEqual([r['frame'] for r in rows], list(range(400, 560)))
        self.assertIsNone(rows[0]['held_active_ink_bottom'])
        self.assertTrue(all(r['held_active_ink_bottom'] is None for r in rows if r['frame'] >= 540))
        self.assertTrue(all(r['held_active_ink_bottom'] is not None for r in rows if 400 < r['frame'] < 540))
        self.assertLessEqual(summary['max_absolute_residual_px'], 2.)
        self.assertEqual(summary['speed_difference_max'], 0.)
        self.assertGreaterEqual(summary['minimum_margin_through_527'], 42.)


if __name__ == '__main__':
    unittest.main()
