"""A6/A7 gain-only polish (lane polishedge, second pass) on synthetic plates; no film assets are read.

Each contract is also run against the behaviour it replaces (the plain frame, a hard exposure step, the rejected
first pass's temporal pixel average, an incomplete dependency list), so a green test proves the check can fail.
Film measurements live in the lane's review, not in these synthetic thresholds.
"""
from contextlib import contextmanager
import hashlib
import math
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest

EDIT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EDIT))
import assemble as AS  # noqa: E402
import deliver as D  # noqa: E402
import edge_polish as EP  # noqa: E402

H, W = 40, 96                      # the synthetic statistics decode (the real one is the plate at a quarter size)


def srgb(level, shape=(H, W)):
    return np.full(shape + (3,), level, np.float32)


def bgr(img):
    return (np.clip(img, 0, 1)[..., ::-1] * 255 + 0.5).astype(np.uint8)


def provider(frames):
    """plate(g) over a dict {frame: sRGB float image}; None where absent (a slate or a missing file)."""
    cache = {}

    def plate(g):
        if g not in frames:
            return None
        if g not in cache:
            cache[g] = EP.Plate(bgr(frames[g]))
        return cache[g]
    return plate


def grey(img):
    return float((np.clip(img, 0, 1) @ np.array([0.299, 0.587, 0.114], np.float32)).mean() * 255)


def render(frames, f, spec, memo=None):
    g = EP.gain(f, spec, provider(frames), memo)
    return frames[f] if g is None else np.clip(EP.apply(frames[f], g), 0, 1)


# ------------------------------------------------------------------------------------------ exposure ramps
def ramp_spec(cut, n_out, n_in, split):
    return dict(EP.WINDOW, ramps=((cut, n_out, n_in, split),), surges=(), chroma=0.0)


def ramp_contract(compose, cut=1840, n_out=0, n_in=10, split=1.0, dark=0.10, bright=0.55):
    spec = ramp_spec(cut, n_out, n_in, split)
    frames = {g: srgb(dark if g < cut else bright) for g in range(cut - 12, cut + 12)}
    greys = {g: grey(compose(frames, g, spec)) for g in range(cut - 10, cut + 12)}
    step = math.log(greys[cut] / greys[cut - 1])
    gap = math.log(grey(frames[cut]) / grey(frames[cut - 1]))
    assert gap > 0.8, 'fixture must contain a large exposure step'
    assert abs(step) < 0.2 * gap, 'the cut keeps its exposure shock'
    after = [greys[g] for g in range(cut, cut + n_in + 1)]
    assert all(b >= a - 1e-6 for a, b in zip(after, after[1:])), 'the incoming ramp is not monotonic'
    before = [greys[g] for g in range(cut - n_out - 1, cut)]
    assert all(b >= a - 1e-6 for a, b in zip(before, before[1:])), 'the outgoing ramp is not monotonic'
    steps = [abs(math.log(greys[g] / greys[g - 1])) for g in range(cut - n_out, cut + n_in + 1) if g != cut]
    assert max(steps) < 0.35 * gap, 'one ramp frame carries most of the exposure change'
    # outside the ramp: the plain frames, exactly
    for g in (cut - n_out - 1, cut + n_in):
        assert np.array_equal(compose(frames, g, spec), frames[g])
    return greys


@pytest.mark.parametrize('n_out,n_in,split', [(0, 10, 1.0), (0, 6, 1.0), (6, 8, 0.5), (8, 0, 0.0)])
def test_exposure_ramp_matches_the_cut_and_rejects_the_plain_hard_step(n_out, n_in, split):
    greys = ramp_contract(render, n_out=n_out, n_in=n_in, split=split)
    if split == 1.0:
        assert greys[1839] == pytest.approx(grey(srgb(0.10)), abs=1e-3), "'in' must leave the outgoing tail alone"
    if split == 0.0:
        assert greys[1840] == pytest.approx(grey(srgb(0.55)), abs=1e-3), "'out' must leave the incoming alone"
    with pytest.raises(AssertionError, match='exposure shock'):
        ramp_contract(lambda frames, g, spec: frames[g], n_out=n_out, n_in=n_in, split=split)


def core_contract(out, inc):
    assert grey(out) < 0.8 * grey(inc), 'the fixture frame was not graded down'
    assert out[:10, :10].min() > 0.95, 'the white core was greyed like a display dimmer'
    assert np.array_equal(out[-10:, -10:], inc[-10:, -10:]), 'black moved'


def test_exposure_ramp_keeps_highlights_white_and_blacks_black():
    """Scene-linear gain through the tone path: an incoming frame graded down (the film's gaps are about 2x in grey)
    keeps its clipped core white and its black black; a display-space dimmer of the same mean greys the core."""
    spec = ramp_spec(1840, 0, 10, 1.0)
    inc = srgb(0.5)
    inc[:10, :10] = 1.0
    inc[-10:, -10:] = 0.0
    frames = {1839: srgb(0.3), 1840: inc}
    out = render(frames, 1840, spec)
    core_contract(out, inc)
    lo, hi = 0.0, 1.0                                     # the negative control: a display-linear dimmer, same grey
    dec = EP._ff()._srgb_decode(inc)
    for _ in range(40):
        k = 0.5 * (lo + hi)
        lo, hi = (k, hi) if grey(EP._ff()._srgb_encode(dec * k)) < grey(out) else (lo, k)
    with pytest.raises(AssertionError, match='greyed like a display dimmer'):
        core_contract(EP._ff()._srgb_encode(dec * lo), inc)


# ------------------------------------------------------------------------------------------------ surges
SURGE = dict(EP.WINDOW, ramps=(), surges=((1920, 2400),))


def flare_frames(beat=2000):
    """A dim crater with one tower strip that flares 3x for ONE frame (scene_b's leap-frogging surge), and sparks
    that differ in every frame."""
    rng = np.random.default_rng(7)
    frames = {}
    for g in range(1960, 2040):
        img = srgb(0.18)
        img[:, 70:84] = 0.30 if g != beat else 0.75
        img[rng.integers(0, H, 12), rng.integers(0, 50, 12)] = 1.0      # sparks, away from the tower
        frames[g] = img
    return frames


def strip_light(img):
    return float(img[:, 72:82].mean())


def surge_contract(compose, beat=2000):
    frames = flare_frames(beat)
    light = {g: strip_light(compose(frames, g, SURGE)) for g in range(beat - 6, beat + 14)}
    base = strip_light(frames[beat - 6])
    assert max(light, key=light.get) == beat, 'the surge must still peak on its own frame'
    assert light[beat - 4] == pytest.approx(base, rel=0.02), 'the attack starts more than three frames early'
    assert base * 1.02 < light[beat - 3] < light[beat - 2] < light[beat - 1] < light[beat], \
        'the surge needs a three-frame attack'
    assert all(light[beat + k] > light[beat + k + 1] for k in range(8)), 'the release is shorter than 8 frames'
    assert light[beat + 1] > base * 1.2, 'the surge drops back in one frame'
    rise = light[beat] - light[beat - 1]
    assert rise < 0.5 * (strip_light(frames[beat]) - base), 'the one-frame pop is kept'
    return light


def test_surge_gets_attack_and_release_and_rejects_the_one_frame_pop():
    surge_contract(render)
    with pytest.raises(AssertionError, match='three-frame attack'):
        surge_contract(lambda frames, g, spec: frames[g])


def rejected_temporal_average(frames, g, spec):
    """The first pass's approach, for the negative control: a weighted mix of neighbouring plates."""
    w = {-2: .1, -1: .2, 0: .4, 1: .2, 2: .1}
    return sum(frames[g + k] * np.float32(v) for k, v in w.items())


def current_frame_only_contract(compose):
    """The output is the current plate times a smooth gain: a texture that exists only in the neighbours (a
    checkerboard) cannot appear, and the current frame's own fine texture survives at full contrast."""
    yy, xx = np.indices((H, W))
    check = ((xx // 2 + yy // 2) % 2).astype(np.float32)
    stripes = (xx % 3 == 0).astype(np.float32)
    frames = {g: srgb(0.2) for g in range(1980, 2020)}
    for g in frames:
        frames[g] = frames[g] + (0.25 * check if g != 2000 else 0.25 * stripes)[..., None]
    frames[2001] = frames[2001] * np.float32(2.5)                        # a surge right after the current frame
    out = compose(frames, 2000, SURGE)
    assert abs(float(np.corrcoef(stripes.ravel(), check.ravel())[0, 1])) < 0.02, 'fixture textures must differ'
    ghost = abs(float(np.corrcoef(out[..., 1].ravel(), check.ravel())[0, 1]))
    assert ghost < 0.05, "a neighbouring frame's texture reached the picture"
    kept = float(np.corrcoef(out[..., 1].ravel(), stripes.ravel())[0, 1])
    assert kept > 0.9, "the current frame's own texture was lost"


def test_output_uses_current_pixels_only_and_rejects_the_temporal_average():
    current_frame_only_contract(render)
    with pytest.raises(AssertionError, match="neighbouring frame's texture"):
        current_frame_only_contract(rejected_temporal_average)


def camera_contract():
    spec = EP.WINDOW
    for f in range(spec['f0'], spec['f1']):
        lo, hi = EP.camera(f, spec)
        ramp_cuts = {c for c, _, _ in EP.ramps_at(f, spec)}
        allowed = set(range(lo, hi))
        for c in ramp_cuts:                                 # a ramp reads its two anchor frames (and their surges)
            allowed |= {c - 1, c} | set(EP.fill_frames(c - 1, spec)) | set(EP.fill_frames(c, spec))
        for g in EP.source_frames(f, spec):
            assert spec['f0'] <= g < spec['f1'], f'{f} reads {g}, outside the lane'
            assert g in allowed, f'{f} reads {g} across a camera change'


def test_surge_reads_stay_inside_their_camera_and_the_lane():
    camera_contract()
    # negative control: a surge window that ignores the 1920 camera change reads the insert from the crater
    with mock.patch.object(EP, 'surge_interval', lambda f, spec: (1860, 2400) if f >= 1860 else None):
        with pytest.raises(AssertionError, match='across a camera change'):
            camera_contract()


# -------------------------------------------------------------------------------------------- identity
def test_plain_frames_outside_the_ramps_and_surges_and_at_the_lane_ends():
    spec = EP.WINDOW
    plain = provider({g: srgb(0.2 + 0.001 * (g % 7)) for g in range(1400, 2440)})
    for f in (1400, 1439, 1440, 1441, 1470, 1486, 1766, 1800, 1839, 1866, 1900, 1913, 2399, 2400, 2439):
        assert EP.gain(f, spec, plain) is None, f
    assert EP.gain(1760, spec, plain) is not None and EP.gain(2000, spec, plain) is not None
    assert EP.taper(1486, spec) == 0 and EP.taper(1496, spec) == 1, 'a mid-shot interval must fade in'
    assert EP.taper(1840, spec) == 1 and EP.taper(1920, spec) == 1, 'an interval opening on a cut starts at once'
    assert EP.taper(2399, spec) == 0 and EP.taper(2389, spec) == 1 and EP.taper(1759, spec) == 1
    surging = provider({g: srgb(0.35 if g == 1488 else 0.2) for g in range(1470, 1510)})   # a beat at 1488
    assert EP.gain(1486, spec, surging) is None, 'the fade-in must hold the field at 0 on the interval\'s first frame'
    with mock.patch.object(EP, 'taper', lambda f, spec: 1.0):      # negative control: no fade-in, a field at once
        assert EP.gain(1486, spec, surging) is not None
    img = np.random.default_rng(3).random((H, W, 3)).astype(np.float32)
    assert np.array_equal(EP.apply(img, (np.ones(3), np.zeros((EP.GY, EP.GX)))), img), \
        'a unit gain must return the plate bit for bit'


def test_ramps_and_surges_stay_inside_the_lane_and_do_not_overlap():
    spec = EP.WINDOW
    assert (spec['f0'], spec['f1']) == (1441, 2400)
    ramp_frames = []
    for cut, n_out, n_in, split in spec['ramps']:
        assert cut in spec['cameras'] and 0 <= split <= 1
        assert (n_out > 0) == (split < 1) and (n_in > 0) == (split > 0)
        ramp_frames += list(range(cut - n_out, cut + n_in))
    assert len(ramp_frames) == len(set(ramp_frames))
    assert min(ramp_frames) >= 1441 and max(ramp_frames) <= 2399
    for a, b in spec['surges']:
        assert spec['f0'] <= a < b <= spec['f1']
        assert EP.camera(a, spec) == EP.camera(b - 1, spec)


# ------------------------------------------------------------------------------------ assembler dispatch
@contextmanager
def synthetic_picture(cut='A', missing=(), finishing=True):
    """The real dispatcher (_transitions, one-shot kind) with synthetic reads, finish and captions."""
    ctx = AS.Ctx.__new__(AS.Ctx)
    ctx.cut, ctx.W, ctx.H, ctx.clean, ctx.variant = cut, W, H, True, None
    ctx.lines = ctx.lines_nt = []
    ctx._ember_on = False
    events = []

    def level(g):
        return 0.30 if (g % 20 == 0 and g >= 1920) else 0.15 + 0.1 * (g >= 1840)

    def raw(_ctx, frame):
        events.append(('read', frame))
        state = 'SLATE missing' if frame in missing else 'synthetic'
        return srgb(level(frame)), {'sec': 'synthetic-shot'}, state, (None if frame in missing else {'stem': 's'})

    def finish(image, source, name, frame):
        events.append(('finish', frame))
        return image * np.float32(.8) + np.float32(.03)

    def outer(frame):
        image, shot, state, source = raw(ctx, frame)
        return (finish(image, source, cut, frame) if finishing and source else image), shot, state, source

    def captions(image, lines, frame):
        events.append(('captions', frame))
        image[0, 0] = (1, 0, 1)

    plates = {}

    def plate(c, variant, g):
        if g in missing:
            return None
        if g not in plates:
            plates[g] = EP.Plate(bgr(srgb(level(g))))
        return plates[g]

    ctx.picture = outer
    with mock.patch.object(AS.Ctx, 'picture', raw), \
            mock.patch.dict(AS.EDL.TRANS, A=[EP.WINDOW], B=[], C=[]), \
            mock.patch.dict(sys.modules, stage=SimpleNamespace(Finisher=lambda look: finish)), \
            mock.patch.object(AS, '_edge_plate', plate), mock.patch.object(AS, '_EDGE_MEMO', {}), \
            mock.patch.object(AS.titles, 'composite_v3', side_effect=captions):
        ctx.picture = AS._transitions(ctx, True if finishing else None)
        yield ctx, events, outer


@pytest.mark.parametrize('finishing', [False, True])
def test_dispatch_grades_before_the_finish_and_captions_last(finishing):
    with synthetic_picture(finishing=finishing) as (ctx, events, outer):
        image = ctx.frame(1840)                                   # the incoming ramp's first frame
        assert events[-1] == ('captions', 1840)
        assert events.count(('finish', 1840)) == int(finishing)
        assert [e for e in events if e[0] == 'read'] == [('read', 1840)], 'the picture read another frame'
        np.testing.assert_array_equal(image[0, 0], (255, 0, 255))
        events.clear()
        plain = outer(1840)[0]
        assert grey(image[1:] / 255.0) < grey(plain[1:]) - 5, 'the ramp did not grade the incoming frame'


def test_dispatch_plays_the_plain_frame_outside_the_ramps_and_other_films():
    for f in (1441, 1800, 1900, 2399, 2400):
        with synthetic_picture() as (ctx, events, outer):
            current = ctx.picture(f)
            legacy = outer(f)
            np.testing.assert_array_equal(current[0], legacy[0])
            assert current[1:] == legacy[1:]
    for cut in ('B', 'C'):
        assert all(t['kind'] != 'edge_polish' for t in AS.EDL.TRANS[cut])


@pytest.mark.parametrize('missing', [1839, 1840, 2002])
def test_missing_statistics_or_slate_play_the_plain_frame(missing):
    frame = 2000 if missing == 2002 else 1840
    with synthetic_picture(missing=(missing,)) as (ctx, events, outer):
        actual = ctx.picture(frame)
        assert 'edge_polish' not in actual[2]
        expected = outer(frame)
        np.testing.assert_array_equal(actual[0], expected[0])


# -------------------------------------------------------------------------------------------- delivery
def dependency_contract(frame=2000):
    deps = set(EP.source_frames(frame, EP.WINDOW))
    universe = range(frame - 20, frame + 10)
    versions = {g: 0 for g in universe}
    seen = []

    def sources(cut, variant, plan, g):
        seen.append(g)
        return [f'synthetic:{g}:v{versions[g]}']

    def key():
        return hashlib.sha256(repr(D.transition_sources('A', None, EP.WINDOW, frame)).encode()).hexdigest()

    with mock.patch.object(D, '_TRANS_CODE', []), \
            mock.patch.object(AS, 'plan_shot', return_value={}), \
            mock.patch.object(D, 'frame_sources', side_effect=sources):
        before = key()
        assert set(seen) == deps, 'the delivery key omits a frame whose statistics decide the gain'
        for g in deps:
            versions[g] = 1
            assert key() != before, f'a changed source {g} did not re-key the segment'
            versions[g] = 0
        unused = next(g for g in universe if g not in deps)
        versions[unused] = 1
        assert key() == before, 'an unused neighbour changed the key'


def test_delivery_keys_every_statistics_frame_and_rejects_current_frame_only():
    assert len(EP.source_frames(2000, EP.WINDOW)) >= 14
    dependency_contract()
    with mock.patch.object(AS, 'transition_source_frames', lambda t, f: (f,)):
        with pytest.raises(AssertionError, match='omits a frame'):
            dependency_contract()
    code = AS.transition_code('edge_polish')
    assert 'def gain(' in code and 'def apply(' in code and 'def _edge_plate(' in code


def test_real_window_is_registered_once_and_exclusive():
    matches = [t for t in AS.EDL.TRANS['A'] if t['kind'] == 'edge_polish']
    assert matches == [EP.WINDOW]
    windows = sorted(AS.EDL.TRANS['A'], key=lambda t: t['f0'])
    assert all(a['f1'] <= b['f0'] for a, b in zip(windows, windows[1:]))
    for frame in (1440, 2400):
        window = AS.transition_at('A', frame)
        assert window is None or window['kind'] != 'edge_polish'
    assert AS.TKINDS_SHOT['edge_polish'] is AS._tk_edge_polish
