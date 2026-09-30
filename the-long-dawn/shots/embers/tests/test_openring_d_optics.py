"""D optical rules exercised on projected geometry, with the old fog as a control."""
import unittest
from unittest.mock import patch

import numpy as np

from test_c_d import C, context, scene_for


class GapOptics(unittest.TestCase):
    def test_far_band_survives_every_hot_shot(self):
        def oracle(old_fog=False):
            for shot, frame in [('forging', 2240), ('race', 2719), ('trap', 3900),
                                ('holdout', 5496), ('instep', 5680)]:
                scene = scene_for(shot)
                ctx = context(scene, frame, .5)
                rgb = scene.glare_layer(ctx)
                a, b = C.D.theta_range(frame)
                # Central third of extant arc, remote from both cut faces.
                th = np.linspace(a + (b-a)/3, b - (b-a)/3, 33)
                p, _ = C.RS.local_points(th, np.zeros_like(th))
                rot, centre, size = scene.ring_frame(frame)
                px, py, _ = ctx.cam.project(centre + size * p @ rot.T, ctx.fr.W, ctx.fr.H)
                if old_fog:
                    ex, ey, _ = ctx.cam.project(scene.ends(frame), ctx.fr.W, ctx.fr.H)
                    yy, xx = np.ogrid[:ctx.fr.H, :ctx.fr.W]
                    for x, y in zip(ex, ey):
                        rgb += (2.2 * np.exp(-.5 * (((xx-x)/32.)**2 + ((yy-y)/24.)**2)))[..., None]
                values = rgb[np.rint(py).astype(int), np.rint(px).astype(int), 0]
                self.assertLess(values.max(), .025, (shot, values.max()))
        oracle()
        with self.assertRaises(AssertionError):
            oracle(old_fog=True)

    def test_core_centred_on_gap_and_pulsed_by_rendered_stroke_clock(self):
        scene = scene_for('race')
        hit = C.D.SHOTS['race'].d_start + 310
        ctx = context(scene, hit, .5)
        x, y, _ = ctx.cam.project(scene.ends(hit), ctx.fr.W, ctx.fr.H)
        cx, cy = float(x.mean()), float(y.mean())

        def oracle(frozen_pulse=None):
            with patch.object(scene.schedule, 'beat_pulse', return_value=1. if frozen_pulse is None else frozen_pulse):
                bright = scene.glare_layer(ctx)
            with patch.object(scene.schedule, 'beat_pulse', return_value=0. if frozen_pulse is None else frozen_pulse):
                quiet = scene.glare_layer(ctx)
            peak_y, peak_x = np.unravel_index(bright[..., 0].argmax(), bright.shape[:2])
            self.assertLess(np.hypot(peak_x-cx, peak_y-cy), 1.5)
            self.assertGreater(bright.max(), quiet.max() * 2.)
            for ex, ey in zip(x, y):
                self.assertGreater(bright[round(ey), round(ex), 0], .10)
            # A faint horizontal streak, subordinate to the compact hot core.
            horizontal = bright[round(cy), round(cx+100), 0]
            vertical = bright[round(cy+100), round(cx), 0]
            self.assertGreater(horizontal, vertical)
            self.assertLess(horizontal, bright.max() * .01)
        oracle()
        with self.assertRaises(AssertionError):
            oracle(frozen_pulse=1.)
        self.assertEqual(scene.schedule.beat_pulse(hit), 1.)
        self.assertLess(scene.schedule.beat_pulse(hit+9), .11)


class StudyOptics(unittest.TestCase):
    def test_caption_grade_respects_handoff_and_recovers_at_cuts(self):
        y = np.arange(804.)
        def oracle(grade):
            for shot in ('forging', 'unfinished'):
                a, b = C.D.shot_range(shot)
                np.testing.assert_array_equal(grade(shot, a, y), np.ones(804))
                np.testing.assert_array_equal(grade(shot, b, y), np.ones(804))
                shaded = grade(shot, a + 160, y)
                self.assertTrue(np.all(shaded[589:718] <= .11))
                np.testing.assert_array_equal(shaded[:500], np.ones(500))
                self.assertGreaterEqual(shaded.min(), .099)
                self.assertLess(np.abs(np.diff(shaded)).max(), .02)
            np.testing.assert_array_equal(grade('race', 2560, y), np.ones(804))
        oracle(C.caption_backdrop_gain)
        with self.assertRaises(AssertionError):
            oracle(lambda shot,t,y: np.ones_like(y))
        with self.assertRaises(AssertionError):
            oracle(lambda shot,t,y: np.full_like(y,.1))

    def test_visible_points_and_surface_lights_share_position_and_gold_white(self):
        scene = scene_for('unfinished')
        ctx = context(scene, C.STUDY_CLOSE, .05)
        states = []
        def raster(cam, w, h, rotation, centre, scale, state, env, **kw):
            states.append(state)
            return np.zeros((h,w,3)), np.ones((h,w)), np.ones((h,w))
        with patch.object(C.RS, 'render', side_effect=raster), \
             patch.object(C.RS, 'merge_occluder', return_value=None), \
             patch.object(C.RS, 'visibility', return_value=np.ones((ctx.fr.H,ctx.fr.W))):
            scene.ring_layer(ctx)
        scene.lamps(ctx)
        self.assertEqual(len(ctx.fr.calls), 2, 'One core and Gaussian halo; no glyph-like cage')
        def halo_oracle(calls):
            self.assertEqual(calls[1][1].get('profile'), 1)
        halo_oracle(ctx.fr.calls)
        bad = list(ctx.fr.calls)
        bad[1] = (bad[1][0], dict(bad[1][1], profile=0))
        with self.assertRaises(AssertionError):
            halo_oracle(bad)
        state = states[0]
        self.assertLess(state.letters, .1)
        def oracle(positions, color):
            for args, _ in ctx.fr.calls:
                np.testing.assert_array_equal(args[1], positions)
                np.testing.assert_array_equal(args[4], color)
            self.assertGreater(color[2] / color[0], .6)
        oracle(state.reading_light.positions, C.GOLD_WHITE)
        with self.assertRaises(AssertionError):
            oracle(state.reading_light.positions + [1.,0.,0.], C.GOLD_WHITE)
        with self.assertRaises(AssertionError):
            oracle(state.reading_light.positions, np.array([1.,.7,.26]))


if __name__ == '__main__':
    unittest.main()
