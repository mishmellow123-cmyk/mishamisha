"""A19's arrival light and the score's four watch-fire withdrawals.

These tests use a synthetic dark-blue plate at 960x402. They test what the comp
emits; the delivered crossing/vista eye trace and caption remain image audits.
No delivered frame is read; all compositing uses synthetic half-size plates.
"""
from copy import deepcopy
from pathlib import Path
import sys
import unittest

import numpy as np

EDIT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EDIT))
import afix_comp as AF  # noqa: E402

SCALE = 0.5
SHAPE = (402, 960, 3)
LUMA = np.array([0.2126, 0.7152, 0.0722], np.float32)
PLATE = np.broadcast_to(np.float32([0.025, 0.035, 0.050]), SHAPE).copy()


def render(frame, config):
    return AF.watchfires(PLATE, PLATE, frame, config)


def patch(image, point, radius=24):
    """A square around a known emitter; point/radius are master pixels."""
    cx, cy = (np.asarray(point) * SCALE).round().astype(int)
    r = int(round(radius * SCALE))
    return image[cy - r:cy + r + 1, cx - r:cx + r + 1]


def emitted_metrics(image, point):
    pixels = patch(image, point)
    y = pixels @ LUMA
    # A bright, warm footprint; the cool synthetic plate is excluded by both
    # conditions. These are acceptance criteria, not claimed film measurements.
    warm = (y > 0.45) & (pixels[..., 0] - pixels[..., 2] > 0.10)
    return float(y.max()), int(warm.sum())


def assert_arrival_contract(config):
    """Shared positive/legacy-negative assertion over every arrival frame."""
    lantern = tuple(config['lantern'])
    other_fires = [(x, y) for x, y, _, _ in config['fires'] if (x, y) != lantern]
    assert other_fires, 'the dominance check must contain competing watch-fires'
    for frame in range(5840, 5901):
        image = render(frame, config)
        peak, area = emitted_metrics(image, lantern)
        others = [emitted_metrics(image, p) for p in other_fires]
        assert peak > 0.90, f'{frame}: lantern core must stay bright, got {peak}'
        assert peak > max(p for p, _ in others) + 0.05, (
            f'{frame}: lantern must dominate every other fire, got {peak}, {others}')
        assert area > 2 * max(a for _, a in others), (
            f'{frame}: lantern must have a distinct warm footprint, got {area}, {others}')


def linear_luma(image):
    rgb = np.where(image <= 0.04045, image / 12.92, ((image + 0.055) / 1.055) ** 2.4)
    return rgb @ LUMA


class ArrivalTests(unittest.TestCase):
    def test_lantern_is_the_brightest_broad_warm_point_through_the_arrival(self):
        assert_arrival_contract(AF.WATCHFIRES)

    def test_legacy_configuration_is_rejected_by_the_same_arrival_assertions(self):
        legacy = deepcopy(AF.WATCHFIRES)
        for key in ('arrival', 'fire_scale', 'figure_scale'):
            legacy.pop(key, None)
        with self.assertRaisesRegex(AssertionError, 'lantern'):
            assert_arrival_contract(legacy)

    def test_bloom_settles_by_5876_without_moving_or_extinguishing_the_core(self):
        config = AF.WATCHFIRES
        cx, cy = np.asarray(config['lantern']) * SCALE
        yy, xx = np.mgrid[:SHAPE[0], :SHAPE[1]]
        distance = np.hypot(xx - cx, yy - cy) / SCALE
        # Sample above the ground pool and beyond the luminous body. Its mean
        # must contract, while the body's brightest pixel stays at the lantern.
        annulus = (distance >= 30) & (distance <= 80) & (yy < cy - 12 * SCALE)
        core = (np.abs(xx - cx) <= 12 * SCALE) & (np.abs(yy - cy) <= 12 * SCALE)
        means, positions = {}, []
        for frame in (5840, 5858, 5876, 5888, 5900):
            image = render(frame, config)
            y = image @ LUMA
            means[frame] = float(y[annulus].mean())
            at = np.unravel_index(int(np.argmax(np.where(core, y, -1))), y.shape)
            positions.append(np.array([at[1], at[0]], np.float64))
            self.assertGreater(float(y[at]), 0.90, frame)
            self.assertLessEqual(float(np.linalg.norm(positions[-1] - [cx, cy])), 1.5, frame)
        self.assertGreater(means[5840] - means[5876], 0.03)
        self.assertGreater(means[5840], means[5858])
        self.assertGreater(means[5858], means[5876])
        for frame in (5888, 5900):
            self.assertAlmostEqual(means[frame], means[5876], delta=0.001, msg=str(frame))
        for point in positions[1:]:
            self.assertLessEqual(float(np.linalg.norm(point - positions[0])), 1.0)

    def test_composition_is_finite_bounded_deterministic_and_preserves_its_input(self):
        before = PLATE.copy()
        for frame in (5840, 5876, 5900, 6240, 6390):
            first = render(frame, AF.WATCHFIRES)
            self.assertEqual(first.shape, SHAPE)
            self.assertEqual(first.dtype, np.float32)
            self.assertTrue(np.isfinite(first).all(), frame)
            self.assertGreaterEqual(float(first.min()), 0.0)
            self.assertLessEqual(float(first.max()), 1.0)
            np.testing.assert_array_equal(first, render(frame, AF.WATCHFIRES))
        np.testing.assert_array_equal(PLATE, before)


class FinalFadeTests(unittest.TestCase):
    def test_final_black_contains_no_surviving_lantern_fires_or_figures(self):
        config = deepcopy(AF.WATCHFIRES)
        config['destreak'] = None
        black = np.zeros(SHAPE, np.float32)
        np.testing.assert_array_equal(AF.watchfires(black, black, 6479, config), black)

    def test_final_frame_preserves_an_already_faded_nonblack_plate(self):
        config = deepcopy(AF.WATCHFIRES)
        config['destreak'] = None
        plate = PLATE * np.float32(0.35)
        output = AF.watchfires(plate, plate, 6479, config)
        np.testing.assert_allclose(output, plate, rtol=0, atol=5e-7)

    def test_mid_fade_attenuates_only_the_composites_delta_from_the_plate(self):
        config = deepcopy(AF.WATCHFIRES)
        config['destreak'] = None
        postponed = deepcopy(config)
        postponed['arrival']['fade'] = (6500, 6523)
        # This plate has already been halved upstream. It is bright enough that
        # a seated cloak removes light; both negative and positive deltas must
        # withdraw while the plate itself remains unchanged.
        plate = np.broadcast_to(np.float32([0.30, 0.35, 0.40]) * 0.5, SHAPE).copy()
        frame = 6467
        output = AF.watchfires(plate, plate, frame, config)
        full = AF.watchfires(plate, plate, frame, postponed)
        base = linear_luma(plate)
        delta = linear_luma(full) - base
        # Ignore clipped emitter cores: the unclipped source intensity cannot
        # be recovered from those pixels in the postponed output.
        usable = (full < 0.98).all(axis=-1)
        self.assertGreater(float(delta[usable].max()), 0.20)
        self.assertLess(float(delta[usable].min()), -0.002)
        u = (frame - 6456) / 23.0
        remaining = 1.0 - (3.0 * u * u - 2.0 * u * u * u)
        expected = base + delta * remaining
        np.testing.assert_allclose(linear_luma(output)[usable], expected[usable], rtol=0, atol=1e-6)


class WatchFireTimingTests(unittest.TestCase):
    def test_horn_frames_still_complete_each_paling_while_lantern_stays_lit(self):
        # These coordinates and score frames are the A19/A20 continuity contract,
        # not values read back from the code under test. A shifted timing fails.
        schedule = ((430, 752, 6240), (1330, 452, 6290),
                    (180, 470, 6340), (1560, 350, 6390))
        for x, y, pale in schedule:
            actual = deepcopy(AF.WATCHFIRES)
            unpaled = deepcopy(actual)
            unpaled['fires'] = [(fx, fy, size, 99999 if (fx, fy) == (x, y) else end)
                                for fx, fy, size, end in actual['fires']]
            absent = deepcopy(actual)
            absent['fires'] = [fire for fire in actual['fires'] if fire[:2] != (x, y)]
            for frame, limits in ((pale - 20, (0.999, 1.001)),
                                  (pale - 10, (0.35, 0.80)),
                                  (pale, (0.10, 0.20)),
                                  (pale + 10, (0.10, 0.20))):
                image = render(frame, actual)
                base = linear_luma(patch(render(frame, absent), (x, y)))
                full = linear_luma(patch(render(frame, unpaled), (x, y))) - base
                current = linear_luma(patch(image, (x, y))) - base
                self.assertGreater(float(full.sum()), 0.0, (x, y, frame))
                fraction = float(current.sum() / full.sum())
                self.assertGreaterEqual(fraction, limits[0], (x, y, frame, fraction))
                self.assertLessEqual(fraction, limits[1], (x, y, frame, fraction))
                self.assertGreater(emitted_metrics(image, actual['lantern'])[0], 0.90, frame)


if __name__ == '__main__':
    unittest.main()
