"""Sweep entrance contracts on synthetic pixels; no scene or delivered-frame loads."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from unittest import mock

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import sweep_entry_candidates as C


def pixels():
    race = np.full((4, 7, 3), [.03, .07, .11], np.float32)
    burn = np.full_like(race, [.91, .63, .24])
    return race, burn


def test_default_and_tail_preserve_the_exact_accepted_object():
    race, burn = pixels()
    for frame in range(1670, 1721):
        assert C.composite(burn, race, frame) is burn
        assert C.composite(burn, race, frame, 'accepted') is burn
        if not 1680 <= frame < 1688:
            assert C.composite(burn, race, frame, 'soft-entry') is burn


def test_onset_is_the_held_race_and_synthetic_light_enters_monotonically():
    race, burn = pixels()
    assert C.composite(burn, race, 1680, 'soft-entry') is race
    previous = race
    increments = []
    light_increments = []

    def red_light(image):
        red = float(image[0, 0, 0])
        return red / 12.92 if red <= .04045 else ((red + .055) / 1.055) ** 2.4

    for frame in range(1681, 1689):
        result = C.composite(burn, race, frame, 'soft-entry')
        assert result.dtype == np.float32
        assert np.isfinite(result).all()
        assert (result >= previous).all()
        assert (result <= burn).all()
        increments.append(float(np.max(result - previous)))
        light_increments.append(red_light(result) - red_light(previous))
        previous = result
    assert max(increments) < float(np.max(burn - race)) / 2
    assert light_increments[0] < light_increments[1]
    assert light_increments[-1] < light_increments[-2]


def test_midpoint_mixes_light_without_mutating_inputs():
    race, burn = pixels()
    originals = race.copy(), burn.copy()
    # Independent closed-form half-light mix, with the red toe handled below,
    # verifies that this is a light blend of the pre-finish source images.
    expected = 1.055 * (.5 * ((race + .055) / 1.055) ** 2.4
                        + .5 * ((burn + .055) / 1.055) ** 2.4) ** (1 / 2.4) - .055
    # Red in race lies below the toe; calculate that operand by the toe rule.
    red = .5 * race[..., 0] / 12.92 + .5 * ((burn[..., 0] + .055) / 1.055) ** 2.4
    expected[..., 0] = 1.055 * red ** (1 / 2.4) - .055
    out = C.composite(burn, race, 1684, 'soft-entry')
    np.testing.assert_allclose(out, expected, atol=2e-7, rtol=0)
    np.testing.assert_array_equal(race, originals[0])
    np.testing.assert_array_equal(burn, originals[1])


def test_invalid_candidate_inputs_and_frame_coordinates_fail_closed():
    race, burn = pixels()
    with pytest.raises(ValueError, match='unknown'):
        C.composite(burn, race, 1681, 'typo')
    with pytest.raises(ValueError, match='integer'):
        C.entry_opacity(1681.5)
    for invalid in (burn.astype(np.float64), burn * np.nan, burn + 2, burn - 2):
        with pytest.raises(ValueError, match='float32'):
            C.composite(invalid, race, 1681, 'soft-entry')
    with pytest.raises(ValueError, match='same shape'):
        C.composite(burn, race[:2], 1681, 'soft-entry')
    for frame in (1679, 1688, 1680.5):
        with pytest.raises(ValueError, match='C1680'):
            C.render_original(frame)
    for scale in (0, 2, float('nan')):
        with pytest.raises(ValueError, match='scale'):
            C.make_renderer(scale=scale)


def test_renderer_routes_keep_opaque_bake_and_only_candidate_reads_race():
    race, burn = pixels()
    inputs = {'race': {'file': 'held.jpg'}, '1683': {'file': 'accepted.jpg'}}

    def read(path, scale):
        assert scale == .5
        return race if path.name == 'held.jpg' else burn

    with mock.patch.object(C, 'validate_inputs', return_value={'inputs': inputs}), \
            mock.patch.object(C, '_read', side_effect=read) as reader:
        original = C.render_original(1683, .5)
        for renderer in (C.make_renderer(scale=.5), C.make_renderer('accepted', .5)):
            result = renderer(1683)
            assert result['rgb'] is original['rgb']
            np.testing.assert_array_equal(result['alpha'], np.ones((4, 7), np.float32))
        assert all(call.args[0].name == 'accepted.jpg' for call in reader.call_args_list)
        candidate = C.make_renderer('soft-entry', .5)(1683)
        assert any(call.args[0].name == 'held.jpg' for call in reader.call_args_list)
        assert not np.array_equal(candidate['rgb'], burn)
        np.testing.assert_array_equal(candidate['alpha'], np.ones((4, 7), np.float32))


def test_source_manifest_rejects_changed_bytes_missing_frames_and_path_substitution():
    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp)
        rows = {}
        for key, filename, source in [('race', 'race_01679.jpg', 'renders/embers_C3/f_01679.jpg')] + [
                (str(f), f'sweep_{f:05d}.jpg', f'renders/book_C_ft/f_{f:05d}.jpg')
                for f in range(1680, 1688)]:
            data = key.encode()  # Bytes only: this test does not decode images.
            (folder / filename).write_bytes(data)
            rows[key] = dict(file=filename, source=source, bytes=len(data),
                             sha256=hashlib.sha256(data).hexdigest())
        manifest = dict(schema=1, inputs=rows)
        path = folder / 'source.json'
        path.write_text(json.dumps(manifest))
        assert C.validate_inputs(folder) == manifest
        (folder / 'race_01679.jpg').write_bytes(b'bad!')
        with pytest.raises(ValueError, match='hash mismatch'):
            C.validate_inputs(folder)
        (folder / 'race_01679.jpg').write_bytes(b'race')
        rows['1680']['file'] = '../elsewhere.jpg'
        path.write_text(json.dumps(manifest))
        with pytest.raises(ValueError, match='path mismatch'):
            C.validate_inputs(folder)
        rows.pop('1680')
        path.write_text(json.dumps(manifest))
        with pytest.raises(ValueError, match='frame set'):
            C.validate_inputs(folder)


def test_reader_checks_native_shape_and_uses_half_decode():
    native = np.empty((804, 1920, 3), np.uint8)
    native[:] = [17, 31, 73]
    half = native[::2, ::2].copy()
    for scale, image, flag in ((1, native, C.cv2.IMREAD_COLOR), (.5, half, C.cv2.IMREAD_REDUCED_COLOR_2)):
        with mock.patch.object(C.cv2, 'imread', return_value=image) as reader:
            out = C._read(Path('synthetic.jpg'), scale)
            reader.assert_called_once_with('synthetic.jpg', flag)
            assert out.shape == image.shape
            np.testing.assert_allclose(out[0, 0], np.array([73, 31, 17]) / 255, atol=1e-7)
    for image in (None, np.zeros((40, 96, 3), np.uint8)):
        with mock.patch.object(C.cv2, 'imread', return_value=image):
            with pytest.raises(ValueError, match='native'):
                C._read(Path('synthetic.jpg'), .5)
