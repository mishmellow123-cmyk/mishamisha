"""A4/A5 continuity contracts on synthetic plates; no delivered images are read.

Negative controls exercise the same acceptance assertions with the removed
treatment, so a green test demonstrates that its detector can see the defect.
"""
from contextlib import contextmanager
from copy import deepcopy
import inspect
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest

EDIT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EDIT))
import afix_comp as AF
import assemble as AS
import edl_v3 as EDL
import ignition_comp as IGN


def window(kind):
    return deepcopy(next(t for t in EDL.TRANS['A'] if t['kind'] == kind and 960 <= t['f0'] < 1440))


def plate(value=.2, shape=(80, 192, 3)):
    return np.broadcast_to(np.asarray(value, np.float32), shape).copy()


def assert_retime_contract(source, t):
    assert source(960, t) == 960
    assert source(1016, t) == t['source_end']
    assert source(1025, t) == t['source_end']
    assert source(1008, t) < t['source_end'], 'the arrival must continue into the breath'
    values = np.asarray([source(f, t) for f in np.arange(960, 1016.001, .125)])
    assert np.all(np.diff(values) >= 0), 'the collapse must never reverse'
    assert abs((values[1] - values[0]) / .125 - 1) < .01, 'enter at native speed'
    assert (values[-1] - values[-2]) / .125 < .01, 'land at zero speed'


def test_collapse_arrives_smoothly_at_the_held_point_and_rejects_linear_retime():
    t = window('collapse')
    assert (t['source_end'], t['land']) == (1012, 1016)
    assert_retime_contract(IGN.collapse_source, t)

    def old_linear(f, config):
        return min(config['source_end'], 960 + (f - 960) * (config['source_end'] - 960) / 56)

    with pytest.raises(AssertionError, match='native speed'):
        assert_retime_contract(old_linear, t)


def test_collapse_first_frame_is_bit_identical_and_early_shutter_control_fails():
    t = window('collapse')

    def read(f):
        return plate(.1 if f % 2 == 0 else .9)

    def assert_native(config):
        np.testing.assert_array_equal(IGN.collapse(960, config, read), read(960))

    assert_native(t)
    early = dict(t, f0=959)
    with pytest.raises(AssertionError):
        assert_native(early)


def test_collapse_shutter_reduces_alternating_flicker_and_bypass_is_detected():
    t = window('collapse')

    def variation(config):
        means = [float(IGN.collapse(f, config, lambda g: plate(.1 if g % 2 else .9)).mean())
                 for f in range(968, 1001)]
        return float(np.abs(np.diff(means)).mean())

    unfiltered = variation(dict(t, shutter=0))
    assert unfiltered > .05, 'negative stimulus must visibly alternate'

    def assert_shutter(config):
        assert variation(config) < .4 * unfiltered, 'shutter must suppress alternating source frames'

    assert_shutter(t)
    with pytest.raises(AssertionError, match='suppress'):
        assert_shutter(dict(t, shutter=0))


def test_shutter_engages_during_first_three_frames_and_eight_frame_entry_fails():
    t = window('collapse')

    def read(frame):
        return plate(.1 if frame % 2 else .9)

    def assert_early_onset():
        means = [float(IGN.collapse(frame, t, read).mean()) for frame in range(960, 964)]
        assert np.abs(np.diff(means)).mean() < .4, \
            'shutter must suppress flicker during frames961–963'

    assert_early_onset()
    original_ease = IGN.ease

    def old_entry(a, b, frame):
        return original_ease(a, a + 8 if (a, b) == (t['f0'], t['f0'] + 3) else b, frame)

    with mock.patch.object(IGN, 'ease', old_entry):
        with pytest.raises(AssertionError, match='frames961–963'):
            assert_early_onset()


def test_collapse_hold_reads_only_landed_source_without_residual_ghosts():
    t = window('collapse')

    def assert_hold(config):
        for f in (1016, 1020, 1025):
            reads = []

            def read(g):
                reads.append(g)
                return plate(.3 if g == t['source_end'] else .8)

            out = IGN.collapse(f, config, read)
            assert reads == [t['source_end']], 'held point must not sample moving glyphs'
            np.testing.assert_allclose(out, plate(.3), atol=1e-7, rtol=0)

    assert_hold(t)
    with pytest.raises(AssertionError, match='moving glyphs'):
        assert_hold(dict(t, land=1030))


def disc_fixture():
    t = window('swell')
    k = .1
    yy, xx = np.mgrid[:80, :192]
    dx, dy = np.asarray(t['disc']) * k
    value = .05 + .92 * np.exp(-((xx - dx) ** 2 + (yy - dy) ** 2) / (2 * 11 ** 2))
    disc = np.repeat(value[..., None], 3, axis=2).astype(np.float32)
    return t, k, plate(.05), disc


def assert_disc_handover(render, t, k, incoming, disc):
    out = render(incoming, incoming, disc, 1048, t, k)
    assert float((out - incoming).max()) > .2, 'disc must remain legible after frame1048'
    np.testing.assert_array_equal(render(incoming, incoming, disc, 1072, t, k), incoming)


def test_disc_persists_into_flame_handover_and_old_eight_frame_release_fails():
    t, k, incoming, disc = disc_fixture()
    assert_disc_handover(IGN.swell, t, k, incoming, disc)
    old = dict(t, f1=1048)
    with pytest.raises(AssertionError, match='remain legible'):
        assert_disc_handover(AS._swell, old, k, incoming, disc)


def test_handover_tracks_central_flame_and_rejects_off_axis_bright_objects():
    t, k, incoming, disc = disc_fixture()
    yy, _ = np.mgrid[:80, :192]

    def halo_centroid(fy, distractor=False, follow=True):
        src = incoming.copy()
        src[fy, 96] = 1
        if distractor:
            src[68:76, 160:175] = 1
        config = t if follow else dict(t, cut=1056)  # v=0: carried disc is still anchored at ignition
        out = IGN.swell(incoming, src, disc, 1056, config, k)
        residual = np.maximum(IGN.linear(out) - IGN.linear(incoming), 0).mean(2)
        residual[fy, 96] = 0
        residual[68:76, 160:175] = 0
        return float((residual * yy).sum() / residual.sum())

    def assert_follow(follow):
        assert halo_centroid(60, follow=follow) - halo_centroid(30, follow=follow) > 10, \
            'the carried light must follow the flame body'

    assert_follow(True)
    with pytest.raises(AssertionError, match='follow the flame'):
        assert_follow(False)
    assert abs(halo_centroid(60, distractor=True) - halo_centroid(60)) < .05


def test_breathing_swell_requires_opt_in_and_preserves_legacy_dispatch():
    t, k, incoming, disc = disc_fixture()
    ctx = SimpleNamespace(W=192)
    old = dict(t)
    old.pop('breathing')
    for f in (1026, 1033, 1040, 1048):
        actual = AS._tk_swell(incoming, incoming, f, old, ctx, {}, lambda: disc)
        expected = AS._swell(incoming, incoming, disc, f, old, k)
        np.testing.assert_array_equal(actual, expected)
    assert not np.array_equal(AS._tk_swell(incoming, incoming, 1048, t, ctx, {}, lambda: disc),
                              AS._tk_swell(incoming, incoming, 1048, old, ctx, {}, lambda: disc))


@pytest.mark.parametrize('frame,key', [(1280, 'enter'), (1439, 'land')])
def test_promise_matches_live_plate_at_both_boundaries_and_old_window_fails(frame, key):
    t = window('vision')
    incoming, outgoing = plate((.18, .21, .26)), plate((.06, .07, .08))

    def assert_boundary(config):
        np.testing.assert_allclose(AF.vision(outgoing, incoming, frame, config), incoming,
                                   atol=2e-7, rtol=0, err_msg='vision must meet the live plate')

    assert_boundary(t)
    t.pop(key)
    with pytest.raises(AssertionError, match='meet the live plate'):
        assert_boundary(t)


def test_promise_body_and_other_vision_keep_the_original_treatment():
    t = window('vision')
    old = deepcopy(t)
    old.pop('enter')
    old.pop('land')
    incoming, outgoing = plate((.18, .21, .26)), plate((.06, .07, .08))
    for f in (1296, 1330, 1380, 1412):
        np.testing.assert_array_equal(AF.vision(outgoing, incoming, f, t),
                                      AF.vision(outgoing, incoming, f, old))
    dead = next(t for t in EDL.TRANS['A'] if t['kind'] == 'vision' and t['f0'] == 2640)
    assert 'enter' not in dead and 'land' not in dead


def test_windows_do_not_overlap_and_overlap_negative_control_is_seen():
    def assert_no_overlap(windows):
        windows = sorted(windows, key=lambda t: t['f0'])
        assert all(a['f1'] <= b['f0'] for a, b in zip(windows, windows[1:])), 'transition windows overlap'

    assert_no_overlap(EDL.TRANS['A'])
    with pytest.raises(AssertionError, match='overlap'):
        assert_no_overlap(list(EDL.TRANS['A']) + [dict(f0=1020, f1=1030)])


@contextmanager
def synthetic_dispatch(missing=()):
    ctx = AS.Ctx.__new__(AS.Ctx)
    ctx.cut, ctx.W, ctx.H = 'A', 192, 80
    reads, finishes = [], []

    def raw(_ctx, f):
        reads.append(f)
        if f in missing:
            return plate(.8), {'sec': 'synthetic'}, 'SLATE (frame not rendered)', None
        return plate(.1 + (f % 100) / 200), {'sec': 'synthetic'}, 'synthetic', {'stem': str(f)}

    class Finisher:
        def __call__(self, img, src, cut, f):
            finishes.append((int(src['stem']), cut, f))
            return img + np.float32((f % 7) / 1000)

    fin = Finisher()

    def outer(f):
        img, shot, status, src = raw(ctx, f)
        return fin(img, src, ctx.cut, f), shot, status, src

    ctx.picture = outer
    with mock.patch.object(AS.Ctx, 'picture', raw), \
            mock.patch.dict(sys.modules, stage=SimpleNamespace(Finisher=lambda look: fin)):
        yield AS._transitions(ctx, True), reads, finishes, outer


def test_dispatcher_reads_arbitrary_source_frames_and_finishes_on_output_clock():
    def sample(f, t, read):
        return .5 * read(982) + .5 * read(983)

    with mock.patch.object(IGN, 'collapse', sample), synthetic_dispatch() as (picture, reads, finishes, _):
        out, _, status, _ = picture(1008)
        assert reads[-2:] == [982, 983]
        assert finishes[-2:] == [(982, 'A', 1008), (983, 'A', 1008)]
        assert all(seed == 1008 for _, _, seed in finishes)
        np.testing.assert_allclose(out, plate(.5125), atol=1e-7, rtol=0)
        assert status.endswith(' + collapse')

    # The prechange callback accepted no source-frame argument. Reintroducing
    # that adapter must fail before it can pretend the first plate is every tap.
    def first_only(o, i, f, t, ctx, lay, first):
        return IGN.collapse(f, t, lambda: first())

    with mock.patch.object(IGN, 'collapse', sample), \
            mock.patch.dict(AS.TKINDS_PAIR, collapse=first_only), \
            synthetic_dispatch() as (picture, _, _, _):
        with pytest.raises(TypeError):
            picture(1008)


def test_dispatcher_outside_owned_windows_is_bit_identical_to_outer():
    with synthetic_dispatch() as (picture, reads, finishes, outer):
        for f in (959, 1073, 1279, 1440, 1441, 1839):
            expected = outer(f)[0]
            reads.clear()
            finishes.clear()
            np.testing.assert_array_equal(picture(f)[0], expected)
            assert reads == [f] and finishes == [(f, 'A', f)]

    # An accidentally extended collapse window would hold A4 over the A6 cut.
    extended = [dict(t, f1=1441) if t['kind'] == 'collapse' else t for t in EDL.TRANS['A']]
    with mock.patch.dict(EDL.TRANS, A=extended), synthetic_dispatch() as (picture, _, _, outer):
        with pytest.raises(AssertionError):
            np.testing.assert_array_equal(picture(1440)[0], outer(1440)[0])


@pytest.mark.parametrize('frame', [990, 1020])
def test_missing_shutter_tap_falls_back_to_live_plate_on_both_sides(frame):
    t = window('collapse')
    absent = int(IGN.collapse_source(frame, t))
    assert absent != frame, 'the missing tap must differ from the available live frame'
    with synthetic_dispatch(missing={absent}) as (picture, reads, finishes, outer):
        np.testing.assert_array_equal(picture(frame)[0], outer(frame)[0])
        assert absent in reads
        assert absent not in [source for source, _, _ in finishes], 'a slate must never pass through the finisher'

    def no_fallback(o, i, f, config, ctx, lay, read):
        return IGN.collapse(f, config, read)

    with mock.patch.dict(AS.TKINDS_PAIR, collapse=no_fallback), \
            synthetic_dispatch(missing={absent}) as (picture, _, _, _):
        with pytest.raises(TypeError):
            picture(frame)


def test_new_helpers_participate_in_both_transition_cache_keys():
    original = inspect.getsource
    before = {kind: AS.transition_code(kind) for kind in ('collapse', 'swell', 'dissolve')}
    helper_source = original(IGN)
    for kind in ('collapse', 'swell'):
        assert helper_source in before[kind]
    with mock.patch.object(AS.inspect, 'getsource',
                           side_effect=lambda obj: original(obj) + ('\n# changed helper\n' if obj is IGN else '')):
        after = {kind: AS.transition_code(kind) for kind in before}
    assert before['collapse'] != after['collapse']
    assert before['swell'] != after['swell']
    assert before['dissolve'] == after['dissolve']

    def assert_dependency(code):
        assert helper_source in code, 'ignition helper source must affect the cache key'

    with pytest.raises(AssertionError, match='cache key'):
        assert_dependency(original(AS._tk_swell) + original(AS._swell))
