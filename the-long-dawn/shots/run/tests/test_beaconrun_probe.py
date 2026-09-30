"""Probe colour and component contracts using synthetic arrays, not film data.

The probe module imports no renderer or assembler until main() runs. These
tests call only its two array helpers; no delivered frame or LUT is loaded.
"""
from pathlib import Path
import sys
from unittest import mock

import numpy as np
import pytest

RUN = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RUN))
import probe_beaconrun as probe


@pytest.mark.parametrize('kind,frame', [('beaconrun', 3960), ('watchers', 4240)])
def test_finished_bytes_normalizes_rgb_and_preserves_finisher_context(kind, frame):
    # Synthetic byte colours include black, white, weak and saturated channels.
    source = np.array([[[0, 64, 255], [1, 128, 254]]], dtype=np.uint8)
    before = source.copy()
    finisher = mock.Mock(side_effect=lambda image, *args: image)
    result = probe.finished_bytes(source, finisher, kind, 'linked-fires', frame)

    finisher.assert_called_once()
    image, take, cut, actual_frame = finisher.call_args.args
    expected = np.array([[[0., 64 / 255., 1.], [1 / 255., 128 / 255., 254 / 255.]]], np.float32)
    assert image.dtype == np.float32
    np.testing.assert_array_equal(image, expected)
    assert take == {'stem': f'cand_{kind}_linked-fires'}
    assert cut == 'A' and actual_frame == frame
    assert result.dtype == np.uint8
    np.testing.assert_array_equal(result, before)
    np.testing.assert_array_equal(source, before)


def test_finished_bytes_clips_and_rounds_finisher_output_to_master_bytes():
    source = np.zeros((1, 3, 3), dtype=np.uint8)
    # Synthetic out-of-gamut values and half-byte boundaries distinguish the
    # master's clip, multiply, +.5, cast from wrapping or truncation.
    film_output = np.array([[[-.05, 0., .49 / 255.],
                             [.5 / 255., 1.5 / 255., .5],
                             [1., 1.02, 255.]]], dtype=np.float32)
    before = film_output.copy()
    result = probe.finished_bytes(source, lambda *args: film_output,
                                  'beaconrun', 'linked-fires', 3960)
    assert result.dtype == np.uint8
    np.testing.assert_array_equal(result, np.array([[[0, 0, 0], [1, 2, 128], [255, 255, 255]]], np.uint8))
    np.testing.assert_array_equal(film_output, before)


def test_warm_thresholds_use_strict_rgb_boundaries():
    # Pure fixtures at and around each inequality; none is a film measurement.
    pixels = np.array([[[110, 80, 40], [111, 80, 40],
                        [125, 100, 50], [126, 100, 50],
                        [200, 120, 100], [200, 121, 100],
                        [200, 6, 5], [200, 7, 5],
                        [255, 205, 100], [255, 200, 166],
                        [255, 200, 167], [240, 240, 240], [40, 100, 200]]], np.uint8)
    before = pixels.copy()
    warm, _ = probe.warm_components(pixels)
    # 1.2 * 5 rounds to exactly 6 in float32, so that pair tests the strict
    # equality boundary; 1.2 * 100 is slightly above 120 in float32.
    expected = np.array([[0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0]], np.uint8)
    assert warm.dtype == np.uint8
    np.testing.assert_array_equal(warm, expected)
    np.testing.assert_array_equal(pixels, before)


def test_warm_components_keep_eight_connected_identity_and_area_floor():
    # Synthetic diagonal triplet, four-pixel L, pair and singleton. The last
    # two remain in the mask but do not qualify as reported components.
    pixels = np.zeros((9, 12, 3), np.uint8)
    diagonal = [(1, 1), (2, 2), (3, 3)]
    corner = [(1, 7), (1, 8), (2, 7), (3, 7)]
    small = [(6, 1), (6, 2), (6, 10)]
    expected_mask = np.zeros(pixels.shape[:2], np.uint8)
    for y, x in diagonal + corner + small:
        pixels[y, x] = [200, 100, 40]
        expected_mask[y, x] = 1
    warm, components = probe.warm_components(pixels)

    np.testing.assert_array_equal(warm, expected_mask)
    components = sorted(components, key=lambda item: tuple(item['box']))
    assert components == [
        {'area': 3, 'box': [1, 1, 3, 3], 'centre': [2., 2.]},
        {'area': 4, 'box': [7, 1, 2, 3], 'centre': [7.25, 1.75]},
    ]
