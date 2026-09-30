"""D v2 timeline, visible margin and glare score, with deliberately broken controls."""
import inspect
import math
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import openring as C
import openring_d as D


class DOffsets(unittest.TestCase):
    EXPECTED = {
        'forging': (2080, 2400, 1040, 1040),
        'race': (2400, 2720, 1440, 960),
        'trap': (3760, 4080, 2320, 1440),
        'holdout': (5480, 5520, 2600, 2880),
        'instep': (5680, 5840, 3840, 1840),
        'unfinished': (6080, 6400, 4000, 2080),
    }

    def assert_offsets(self, table, convert, ranges):
        self.assertEqual(set(table), set(self.EXPECTED))
        for name, (a, b, c, offset) in self.EXPECTED.items():
            self.assertEqual(ranges(name), (a, b))
            self.assertEqual(table[name].offset, offset)
            np.testing.assert_array_equal(convert(name, np.array([a, a + 1, b - 1, b])),
                                          [c, c + 1, c + b - a - 1, c + b - a])
            self.assertEqual(len(range(*ranges(name))), b - a)

    def test_offsets_are_explicit_and_extended_clocks_do_not_clamp(self):
        self.assert_offsets(D.SHOTS, D.legacy_frame, D.shot_range)
        with self.assertRaises(AssertionError):
            self.assert_offsets(D.SHOTS,
                                lambda shot, f: np.minimum(D.legacy_frame(shot, f), 1679.),
                                D.shot_range)
        wrong = dict(D.SHOTS, trap=D.Shot(3760, 4080, 2340))
        with patch.object(D, 'SHOTS', wrong):
            with self.assertRaises(AssertionError):
                self.assert_offsets(D.SHOTS, D.legacy_frame, D.shot_range)
        with self.assertRaises(AssertionError):
            self.assert_offsets(D.SHOTS, D.legacy_frame,
                                lambda shot: (D.shot_range(shot)[0], D.shot_range(shot)[1] - 1))

    def assert_events(self, events):
        self.assertEqual(events, {
            'sink': (3860, 3878),
            'surge': (3920, 3932),
            'surge_height': (3920, 3978),
            'return': (3980, 3992),
            'return_height': (3980, 4018),
            'leaders': (4020, 4038),
            'leaders_height': (4020, 4076),
            'leaders_lean': (4020, 4070),
            'instep_ease': (5688, 5712),
            'ends_cool': (5688, 5752),
        })

    def test_events_use_the_source_clock_and_distinguish_settling_from_cooling(self):
        self.assert_events(D.EVENTS)
        def metadata_oracle(api):
            self.assertEqual((api.PAUSE_START, api.SETTLED, api.COOL_END), (5688, 5712, 5752))
            self.assertFalse(api.HOLDOUT_PROVISIONAL)
            self.assertEqual(api.HOLDOUT_STRIKE, 5496)

        metadata_oracle(D)
        for name in D.EVENTS:
            a, b = D.EVENTS[name]
            with self.subTest(event=name), self.assertRaises(AssertionError):
                self.assert_events(dict(D.EVENTS, **{name: (a + 20, b + 20)}))
        for field, wrong in (('SETTLED', D.COOL_END), ('HOLDOUT_PROVISIONAL', True),
                             ('HOLDOUT_STRIKE', D.HOLDOUT_STRIKE - 20)):
            with patch.object(D, field, wrong), self.assertRaises(AssertionError):
                metadata_oracle(D)


class DGap(unittest.TestCase):
    def assert_milestones(self, gap):
        for f, expected in ((2080, 120), (2239, 120), (2345, 108), (2400, 108),
                            (2715, 80), (2720, 80), (2925, 55), (2960, 55),
                            (3760, 55), (4079, 55), (5480, 55), (5496, 55),
                            (5501, 50), (5519, 50), (5680, 50), (5840, 50),
                            (6080, 50), (6399, 50), (1e8, 50)):
            self.assertAlmostEqual(gap(f), expected, msg=str(f))

    def assert_holds(self, gap):
        for (_, end, target), (start, _, _) in zip(D.GAP_SCHEDULE, D.GAP_SCHEDULE[1:]):
            np.testing.assert_allclose(gap(np.linspace(end, start, 17)), target,
                                       atol=1e-10, rtol=0.)
        np.testing.assert_allclose(gap(np.arange(5501., 9200., .5)), 50., atol=1e-10, rtol=0.)

    def assert_monotone(self, gap):
        values = gap(np.arange(2080., 6400.01, .25))
        self.assertTrue(np.all(np.diff(values) <= 1e-10))
        self.assertGreaterEqual(float(values.min()), 50. - 1e-10)
        self.assertLessEqual(float(values.max()), 120. + 1e-10)

    def test_margin_milestones_monotone_strokes_and_exact_holds(self):
        self.assert_milestones(D.gap_deg)
        self.assert_monotone(D.gap_deg)
        self.assert_holds(D.gap_deg)
        with self.assertRaises(AssertionError):
            self.assert_milestones(C.gap_deg)
        wrong = list(D.GAP_SCHEDULE)
        a, b, _ = wrong[7]
        wrong[7] = a, b, 109.
        with patch.object(D, 'GAP_SCHEDULE', tuple(wrong)):
            with self.assertRaises(AssertionError):
                self.assert_monotone(D.gap_deg)
        with patch.object(D, 'GAP_SCHEDULE', D.GAP_SCHEDULE + ((6300., 6305., 49.),)):
            with self.assertRaises(AssertionError):
                self.assert_holds(D.gap_deg)
            with self.assertRaises(AssertionError):
                self.assert_monotone(D.gap_deg)
        frames = [2080.] + [end for _, end, _ in D.GAP_SCHEDULE]
        gaps = [120.] + [gap for _, _, gap in D.GAP_SCHEDULE]
        with self.assertRaises(AssertionError):
            self.assert_holds(lambda f: np.interp(f, frames, gaps))

    def assert_rate(self, rate):
        for start, end, _ in D.GAP_SCHEDULE:
            self.assertGreater(rate((start + end) / 2.), 0.)
            self.assertEqual(rate(start), 0.)
            self.assertEqual(rate(end), 0.)
            self.assertEqual(rate(end + 1), 0.)
            f = start + .37 * (end - start)
            numerical = -(D.gap_deg(f + 1e-4) - D.gap_deg(f - 1e-4)) / 2e-4
            self.assertAlmostEqual(rate(f), numerical, places=6)

    def test_rate_is_gap_derivative_and_race_strokes_accelerate(self):
        self.assert_rate(D.advance_rate)
        for wrong in (lambda f: 0., lambda f: D.advance_rate(f) + .01):
            with self.assertRaises(AssertionError):
                self.assert_rate(wrong)
        starts = np.array([start for start, _, _ in D.GAP_SCHEDULE if 2400 <= start < 2720])
        spacings = np.diff(starts)
        oracle = lambda x: self.assertTrue(np.all(np.diff(x) <= 0.) and x[0] > x[-1])
        oracle(spacings)
        with self.assertRaises(AssertionError):
            oracle(spacings[::-1])
        with self.assertRaises(AssertionError):
            oracle(np.full_like(spacings, 20.))

    def assert_arc_and_stop(self, theta_range, arc_theta, write):
        frames = np.array([2080., 2242., 2622., 3760., 5498., 5752., 6399.])[:, None]
        a, b = theta_range(frames)
        np.testing.assert_allclose((a + b) / 2., D.GAP_CENTRE + np.pi)
        np.testing.assert_allclose(b - a, np.deg2rad(360. - D.gap_deg(frames)))
        fractions = np.linspace(0., 1., 31)[None, :]
        theta = arc_theta(frames, fractions)
        self.assertTrue(np.all(theta > a))
        self.assertTrue(np.all(theta < b))
        for f in frames[:, 0]:
            a, b = theta_range(f)
            th = np.array([a + .1, b - .1, b, b + .01, a + 2. * np.pi - .01])
            np.testing.assert_allclose(write(f)(th), [1., 1., .5, 0., 0.], atol=1e-10)

    def test_pose_bisector_arc_births_and_inscription_stop_preserve_contract(self):
        self.assert_arc_and_stop(D.theta_range, D.arc_theta, D.write)
        with self.assertRaises(AssertionError):
            self.assert_arc_and_stop(C.theta_range, D.arc_theta, D.write)
        with self.assertRaises(AssertionError):
            self.assert_arc_and_stop(D.theta_range, lambda f, p: 2. * np.pi * p, D.write)
        with self.assertRaises(AssertionError):
            self.assert_arc_and_stop(D.theta_range, D.arc_theta, lambda f: np.ones_like)


class DGlareHeat(unittest.TestCase):
    def assert_race_glare(self, glare):
        samples = np.array([glare(f) for f in np.linspace(2400., 2720., 41)])
        self.assertTrue(np.all(np.diff(samples, axis=0) >= 0.))
        self.assertTrue(np.all(samples[-1] > samples[0] * 3.))
        self.assertGreater(samples[0, 0], 0.)
        self.assertGreater(samples[0, 1], 0.)

    def assert_glare_clears(self, glare):
        samples = np.array([glare(f) for f in np.linspace(D.PAUSE_START, D.COOL_END, 65)])
        self.assertTrue(np.all(np.diff(samples, axis=0) <= 0.))
        self.assertTrue(np.all(samples[0] > 0.))
        np.testing.assert_array_equal(samples[-1], [0., 0.])
        np.testing.assert_array_equal(glare(6399.), [0., 0.])

    def test_glare_widens_in_race_and_has_no_pause_residue(self):
        self.assert_race_glare(D.glare)
        self.assert_glare_clears(D.glare)
        for wrong in (lambda f: D.Glare(16., .5), lambda f: D.glare(5120. - f)):
            with self.assertRaises(AssertionError):
                self.assert_race_glare(wrong)
        for wrong in (lambda f: D.Glare(64., 2.2),
                      lambda f: np.asarray(D.glare(f)) + .01,
                      lambda f: D.glare(D.PAUSE_START + D.COOL_END - f)):
            with self.assertRaises(AssertionError):
                self.assert_glare_clears(wrong)

    def assert_ends_hot_body_cool(self, heat):
        for f in (2400., 2560., 2719., 3760., 5498., D.PAUSE_START):
            a, b = D.theta_range(f)
            values = heat(f)(np.array([a, b, (a + b) / 2.]))
            self.assertGreaterEqual(values[0], .88 - 1e-12)
            self.assertAlmostEqual(values[0], values[1])
            self.assertLess(values[2], .001)

    def assert_cooling(self, heat):
        values = np.array([heat(f)(D.theta_range(f)[0]) for f in range(D.PAUSE_START, 6400)])
        self.assertGreater(values[0], .99)
        self.assertTrue(np.all(np.diff(values) <= 1e-12))
        self.assertAlmostEqual(values[-1], .13)
        self.assertAlmostEqual(heat(D.COOL_END)(D.theta_range(D.COOL_END)[0]), .13)

    def test_white_heat_stays_on_both_ends_then_cools_to_gold(self):
        self.assert_ends_hot_body_cool(D.heat)
        self.assert_cooling(D.heat)
        for wrong in (lambda f: lambda theta: np.ones_like(theta),
                      lambda f: lambda theta: np.zeros_like(theta)):
            with self.assertRaises(AssertionError):
                self.assert_ends_hot_body_cool(wrong)
            with self.assertRaises(AssertionError):
                self.assert_cooling(wrong)
        with self.assertRaises(AssertionError):
            self.assert_cooling(lambda f: D.heat(D.PAUSE_START))


class Compatibility(unittest.TestCase):
    def assert_api(self, api):
        for name in ('gap_deg', 'advance_rate', 'theta_range', 'arc_theta', 'write', 'heat', 'placement'):
            self.assertEqual(inspect.signature(getattr(api, name)), inspect.signature(getattr(C, name)))
        self.assertIs(api.placement, C.placement)
        self.assertEqual(api.GAP_CENTRE, C.GAP_CENTRE)

    def test_existing_placement_is_reused_and_public_signatures_stay_stable(self):
        self.assert_api(D)
        with patch.object(D, 'placement', lambda rotation: rotation):
            with self.assertRaises(AssertionError):
                self.assert_api(D)
        with patch.object(D, 'GAP_CENTRE', D.GAP_CENTRE + math.pi):
            with self.assertRaises(AssertionError):
                self.assert_api(D)


if __name__ == '__main__':
    unittest.main()
