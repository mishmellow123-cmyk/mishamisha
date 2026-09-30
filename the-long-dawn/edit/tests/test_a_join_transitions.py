"""A2's two joins: synthetic finished plates, no delivered frames or LUTs.

The acceptance helpers also run against the old configurations. These limits
describe the synthetic stimuli; measured film values belong in the join audit.
"""
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest

EDIT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EDIT))
import assemble as AS
import afix_comp as AF
import deliver as D
import edl_v3 as EDL


def plate(rgb):
    return np.broadcast_to(np.asarray(rgb, np.float32), (27, 64, 3)).copy()


@contextmanager
def synthetic_picture(cut='A'):
    """Exercise the dispatcher, with recorded source reads and finish calls."""
    ctx = AS.Ctx.__new__(AS.Ctx)
    ctx.cut, ctx.W, ctx.H = cut, 64, 27
    reads, finishes = [], []
    dark = plate((0.15, 0.20, 0.27))
    bright = plate((0.955, 0.955, 0.970))

    def raw(_ctx, frame):
        reads.append(frame)
        sec = 'black' if frame < 80 else ('dawn' if frame < 560 else 'white')
        img = np.zeros_like(dark) if sec == 'black' else (dark if sec == 'dawn' else bright).copy()
        src = None if sec == 'black' else {'stem': sec}
        return img, {'sec': sec}, sec, src

    class Finisher:
        def __call__(self, img, src, name, frame):
            finishes.append((src['stem'], name, frame))
            return img * np.float32(0.9) + np.float32(0.04)

    fin = Finisher()

    def outer(frame):
        img, shot, status, src = raw(ctx, frame)
        return (fin(img, src, cut, frame) if src else img), shot, status, src

    ctx.picture = outer
    with mock.patch.object(AS.Ctx, 'picture', raw), \
            mock.patch.dict(sys.modules, stage=SimpleNamespace(Finisher=lambda look: fin)):
        yield AS._transitions(ctx, True), reads, finishes


def assert_opening_contract(picture):
    """A constant nonblack sky emerges from black without a boundary step."""
    black = picture(79)[0]
    images = [picture(frame)[0] for frame in range(80, 129)]
    np.testing.assert_array_equal(black, np.zeros_like(black))
    assert float(images[0].max()) < 1 / 255, 'opening must start within one code value of black'
    assert float(images[-1].min()) > 0.1, 'the stimulus must contain a visible incoming sky'
    for previous, current in zip(images, images[1:]):
        assert float((current - previous).min()) >= -1e-7, 'constant sky must not dim during its opening'
        assert float(np.abs(current - previous).max()) < 0.04, 'opening contains a large one-frame step'
    assert float(np.abs(images[-1] - images[-2]).max()) < 1 / 255, 'opening exit must reach the live sky'


def bloom_window():
    return deepcopy(next(t for t in EDL.TRANS['A'] if t['kind'] == 'bloom' and t['cut'] == 560))


def assert_bloom_landing_contract(config):
    """The outgoing white lands on the incoming plate before the hard cut."""
    outgoing, incoming = plate((0.15, 0.20, 0.27)), plate((0.90, 0.92, 0.94))
    images = [AF.bloom(outgoing, incoming, frame, config) for frame in range(552, 561)]
    np.testing.assert_allclose(images[-2], incoming, rtol=0, atol=2e-7,
                               err_msg='bloom must land on incoming white at 559')
    np.testing.assert_array_equal(images[-1], incoming)
    steps = [float(np.abs(b - a).max()) for a, b in zip(images, images[1:])]
    assert max(steps[-3:]) < 0.025, 'bloom leaves a large step at the end of its landing'
    assert steps[-1] < 2e-7, 'the 559|560 boundary must not add another white-level step'
    for image in images:
        assert image.dtype == np.float32 and np.isfinite(image).all()
        assert float(image.min()) >= 0 and float(image.max()) <= 1


def assert_no_overlaps(windows):
    ordered = sorted(windows, key=lambda t: t['f0'])
    for left, right in zip(ordered, ordered[1:]):
        assert left['f1'] <= right['f0'], f"overlap: {left['kind']} / {right['kind']}"


def test_opening_real_window_and_prechange_floor_control():
    with synthetic_picture() as (picture, _, _):
        assert_opening_contract(picture)
    old = [dict(f0=80, f1=128, kind='floor', k0=1.0, k1=0.0)
           if t['f0'] == 80 else t for t in EDL.TRANS['A']]
    with mock.patch.dict(EDL.TRANS, A=old), synthetic_picture() as (picture, _, _):
        with pytest.raises(AssertionError, match='opening must start'):
            assert_opening_contract(picture)


def test_opening_dispatch_holds_black_and_finishes_the_live_sky_once():
    with synthetic_picture() as (picture, reads, finishes):
        for frame in (80, 104, 127):
            reads.clear()
            finishes.clear()
            _, shot, status, _ = picture(frame)
            assert reads == [79, frame]
            assert finishes == [('dawn', 'A', frame)]
            assert shot['sec'] == 'dawn' and status.endswith(' + dissolve')
        reads.clear()
        finishes.clear()
        picture(128)
        assert reads == [128] and finishes == [('dawn', 'A', 128)]


def test_opening_change_does_not_install_a_dissolve_in_b_or_c():
    assert AS.transition_at('B', 80) is None
    assert AS.transition_at('C', 80)['kind'] == 'floor'
    for cut in ('B', 'C'):
        with synthetic_picture(cut) as (picture, _, _):
            assert float(picture(80)[0].min()) > 0.1


def test_bloom_landing_and_prechange_nonlanding_control():
    config = bloom_window()
    assert_bloom_landing_contract(config)
    config.pop('land', None)
    with pytest.raises(AssertionError, match='bloom must land'):
        assert_bloom_landing_contract(config)


def test_landing_preserves_the_earlier_bloom_and_requires_explicit_opt_in():
    config = bloom_window()
    assert tuple(config['land']) == (552, 559)
    legacy = deepcopy(config)
    legacy.pop('land')
    outgoing, incoming = plate((0.15, 0.20, 0.27)), plate((0.90, 0.92, 0.94))
    before = outgoing.copy()
    for frame in range(536, 553):
        np.testing.assert_array_equal(AF.bloom(outgoing, incoming, frame, config),
                                      AF.bloom(outgoing, incoming, frame, legacy))
    # Without an opt-in the bloom depends on the outgoing plate until the cut.
    # An unconditional incoming blend would fail even if A2's endpoint passed.
    alternate = plate((0.3, 0.4, 0.5))
    for frame in (552, 555, 559):
        np.testing.assert_array_equal(AF.bloom(outgoing, incoming, frame, legacy),
                                      AF.bloom(outgoing, alternate, frame, legacy))
    np.testing.assert_array_equal(outgoing, before)
    np.testing.assert_array_equal(AF.bloom(outgoing, alternate, 560, legacy), alternate)


def test_bloom_dispatch_lands_on_the_finished_incoming_plate_without_double_finish():
    with synthetic_picture() as (picture, reads, finishes):
        at559 = picture(559)[0]
        assert reads == [559, 560]
        assert finishes == [('dawn', 'A', 559), ('white', 'A', 559)]
        reads.clear()
        finishes.clear()
        at560 = picture(560)[0]
        assert reads == [560] and finishes == [('white', 'A', 560)]
        np.testing.assert_allclose(at559, at560, rtol=0, atol=2e-7)


def test_a_windows_do_not_mask_one_another_and_overlap_control_is_rejected():
    assert_no_overlaps(EDL.TRANS['A'])
    overlap = list(EDL.TRANS['A']) + [dict(f0=550, f1=562, cut=560, kind='dissolve')]
    with pytest.raises(AssertionError, match='overlap'):
        assert_no_overlaps(overlap)


def test_window_config_changes_rekey_a2_and_preserve_other_sampled_segments():
    # This checks configuration dependencies. Changing afix_comp's source has
    # its existing wider cache effect because transition_code hashes that module.
    code = {k: 'synthetic-code' for k in ('frame', 'text', 'slate', 'x2', 'title')}
    shots = [(i, s) for i, s in enumerate(EDL.EDL['A']) if s['sec'] in ('A1', 'A2', 'A3', 'A14', 'A19')]

    def plan(shot, cut, variant):
        return dict(kind='black' if shot['kind'] == 'black' else 'take',
                    take=shot['takes'][0] if shot['takes'] else None,
                    have=shot['f1'] - shot['f0'], alt=False)

    def keys():
        with mock.patch.object(D, '_TRANS_CODE', []):
            return {s['sec']: D.segment_key('A', None, D.PROFILES['master'], i, s,
                                            plan(s, 'A', None), code, []) for i, s in shots}

    with mock.patch.object(AS, 'plan_shot', plan), \
            mock.patch.object(D, 'frame_sources', side_effect=lambda *args: []), \
            mock.patch.object(AS, 'transition_layers', return_value={}):
        current = keys()
        old = deepcopy(EDL.TRANS['A'])
        for index, window in enumerate(old):
            if window['f0'] == 80:
                old[index] = dict(f0=80, f1=128, kind='floor', k0=1.0, k1=0.0)
            if window['kind'] == 'bloom':
                window.pop('land', None)
        with mock.patch.dict(EDL.TRANS, A=old):
            previous = keys()
        assert {sec for sec in current if current[sec] != previous[sec]} == {'A2'}
