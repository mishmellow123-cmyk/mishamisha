"""The ridge fire opens its held source onto the map using the existing C/D burn grammar."""
from copy import deepcopy
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import assemble as AS
import edl_v3 as EDL


def ridge_burn():
    return next(t for t in EDL.TRANS['D'] if t.get('cut') == 5180)


def assert_ridge_contract(t):
    assert (t['f0'], t['f1'], t['cut']) == (5180, 5220, 5180)
    assert (t['kind'], t['center'], t['t_open'], t['speed'], t['seed']) == (
        'ring_burn', (1136., 623.), 5182., 9.4, 12)
    assert AS.transition_source_frames(t, 5180) == (5179, 5180)
    assert AS.transition_source_frames(t, 5219) == (5179, 5219)
    shots = {s['code']: s for s in EDL.D}
    assert EDL.source_frame(shots['D20']['takes'][0], 5179) == 3799
    assert EDL.source_frame(shots['D21a']['takes'][0], 5180) == 3440
    assert EDL.source_frame(shots['D21a']['takes'][0], 5219) == 3479
    assert AS.transition_layers(t, 5200) == {}


def test_ridge_uses_measured_fire_and_exact_held_source_clock():
    assert_ridge_contract(ridge_burn())
    for key, value in [('center', (998., 92.)), ('cut', 5181), ('t_open', 5180.), ('f1', 5219)]:
        bad = dict(ridge_burn(), **{key: value})
        with pytest.raises(AssertionError):
            assert_ridge_contract(bad)


def test_ridge_window_does_not_change_the_drawn_ring_burn():
    t = deepcopy(next(t for t in EDL.TRANS['D'] if t.get('cut') == 1680))
    t.pop('note')
    assert t == dict(f0=1680, f1=1760, cut=1680, kind='ring_burn', center=(998., 92.),
                     t_open=1682., speed=2.8, seed=12)
    assert ridge_burn()['f0'] >= t['f1']
    with pytest.raises(AssertionError):
        assert ridge_burn()['center'] == t['center']
