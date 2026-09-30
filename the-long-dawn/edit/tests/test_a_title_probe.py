"""Synthetic registration controls; no delivered frames or renderer are loaded."""
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

EDIT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(EDIT), str(EDIT / 'tools')]
import a_title_probe as TP


def test_registration_follows_shifted_strokes_instead_of_assuming_source_position():
    alpha = np.zeros((40, 70), np.float32)
    alpha[9:32, 10:14] = 1
    alpha[9:13, 10:29] = 1
    alpha[22:26, 10:25] = 1
    alpha[12:33, 42:47] = 1
    alpha[29:33, 42:60] = 1
    original, shifted = (22, 18), (81, 63)
    for position in (original, shifted):
        x, y = position
        layer = np.zeros((130, 190), np.float32)
        layer[y:y + 40, x:x + 70] = alpha
        match = TP.register_template(layer, alpha)
        assert (match['x0'], match['y0']) == position
        assert match['score'] > 0.99
        assert match['provenance_verified'] is False
    # Negative control: retaining the original source coordinates must fail
    # the same registration requirement once delivered strokes move.
    with pytest.raises(AssertionError):
        assert (match['x0'], match['y0']) == original


def test_empty_layer_cannot_return_an_arbitrary_registration():
    alpha = np.zeros((20, 30), np.float32)
    alpha[5:15, 10:15] = 1
    with pytest.raises(ValueError, match='No sharp stroke variation'):
        TP.register_template(np.zeros((80, 100), np.float32), alpha)


def test_backdrop_restores_ember_on_picture_failure():
    original = lambda image, frame: (image, True)

    def fail(frame):
        raise RuntimeError('decode failed')

    ctx = SimpleNamespace(ember=original, _ember_on=True, picture=fail)
    with pytest.raises(RuntimeError, match='decode failed'):
        TP.without_title(ctx, 6360, (0, 0, 10, 10))
    assert ctx.ember is original
    assert ctx._ember_on is True


@pytest.mark.parametrize('ratio', [1.2, 3.5, 12.0])
def test_unvalidated_template_values_never_become_title_contrast_verdicts(ratio):
    # Values on either side of the caption thresholds need identical unusable
    # status. Their arithmetic remains available only within the diagnostic.
    values = dict(overall=ratio, worst=ratio, slices=[ratio] * 8,
                  limiting_slice=0, slice_counts=[[30, 60]] * 8)
    measurement = TP.conditional_measurement(values)

    def assert_conditional(result):
        assert result['contrast_usable'] is False
        assert result['mask_validation'] == 'pending independent validation'
        assert result['values'] == values
        assert 'flags' not in result
        assert 'worst' not in result
        assert 'overall' not in result

    assert_conditional(measurement)
    for wrong in (dict(measurement, contrast_usable=True),
                  dict(measurement, flags=['LOW-CONTRAST']),
                  dict(measurement, flags=[]), dict(measurement, worst=ratio)):
        with pytest.raises(AssertionError):
            assert_conditional(wrong)


def test_contact_label_cannot_be_read_as_a_title_contrast_result():
    def assert_unusable(label):
        assert 'A 6360' in label
        assert 'template diagnostic' in label
        assert 'contrast unusable' in label
        assert ':1' not in label

    assert_unusable(TP.sample_label(6360))
    with pytest.raises(AssertionError):
        assert_unusable('A 6360  sampled worst 1.200:1')
