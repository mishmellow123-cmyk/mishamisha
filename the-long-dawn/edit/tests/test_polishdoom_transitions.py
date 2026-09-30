"""A8/A9 polish contracts (polishdoom) on small synthetic plates; no delivered frames are read.

The owner's standard for this range: no temporal pixel averaging and no cross-dissolve bridges between camera views;
hard cuts stay hard on their beats; beat surges and cut exposures are shaped by a gain field on each frame's own
pixels; the white is full on the impact frame; the valley comes out of that white smoothly. Every acceptance check
below is also run against a rejected configuration (a first-pass bridge, an entry blend, the old white entry, a
window cut that plays the clipped tail) and must fail there. Film measurements live in the lane audit, not in these
synthetic thresholds.
"""
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest import mock

import cv2
import numpy as np
import pytest

EDIT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EDIT))
import afix_comp as AF
import assemble as AS
import deliver as D
import edl_v3 as EDL

BEATS_RIM = (2400, 2420, 2440, 2480)
BEATS_FALL = (2540, 2560, 2580, 2600, 2620)
CUTS = (2460, 2520)


def plate(rgb, shape=(27, 64)):
    return np.broadcast_to(np.asarray(rgb, np.float32), shape + (3,)).copy()


def detailed_plate(shift=0.0):
    yy, xx = np.mgrid[0:27, 0:64].astype(np.float32)
    body = 0.23 + 0.18 * ((xx / 63 + shift) % 1.0) + 0.08 * yy / 26
    return body[..., None] * np.float32([1.0, 0.82, 0.65])


def window(kind, start=None):
    return deepcopy(next(t for t in EDL.TRANS['A'] if t['kind'] == kind
                         and (start is None or t['f0'] == start)))


def legacy_iceheart(o, i, f, t):
    """Frozen pre-polish colour path (A-FIX E3), without any exposure table."""
    img = i if f >= t['cut'] else o
    w = AF._ss(t['ramp'][0], t['ramp'][1], f)
    if w <= 0.0:
        return img
    L = AF._lin(img)
    Y = (0.2126 * L[..., 0] + 0.7152 * L[..., 1] + 0.0722 * L[..., 2])[..., None]
    ice = np.float32([0.86, 0.94, 1.0]) * Y * t.get('gain', 1.35)
    gold = np.float32([1.0, 0.78, 0.45]) * Y
    tgt = ice * 0.85 + gold * 0.15
    tgt = 1.0 - (1.0 - tgt) * (1.0 - t.get('lift', 0.25) * w)
    return np.clip(AF._srgb(L * (1.0 - w) + tgt * w), 0.0, 1.0)


def legacy_vision(o, i, f, t):
    """Frozen pre-doom vision path at the owner's base b429b7b: A5's foreground flame and THE PROMISE's entry/landing
    blends from the ignition lane (enter/land), no veil."""
    H, W = i.shape[:2]
    k = W / AF.REF_W
    a, rim, (cx, cy), s = AF._window(H, W, f, t, k)
    fm = AF._flame_key(i, f, t, k) if t.get('flame_key') else None
    src = i
    if t.get('shift_y'):
        src = cv2.warpAffine(i, np.float32([[1, 0, 0], [0, 1, t['shift_y'] * k]]), (W, H),
                             flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    if fm is not None:
        src = np.clip(AF._srgb(AF._fill(AF._lin(src), cv2.dilate(fm, np.ones((5, 5), np.uint8)), k)), 0.0, 1.0)
    vis = AF._shimmer(src, f, t.get('shimmer', 1.6) * s, k)
    vis_l = AF._lin(vis) * np.float32(t.get('inside_tint', (1.0, 1.0, 1.0))) * t.get('inside_gain', 1.0)
    gl = t.get('inside_glow', 0.35)
    if gl > 0.0:
        vis_l = vis_l + gl * cv2.GaussianBlur(vis_l, (0, 0), 9.0 * k)
    if t.get('outside', 'o') == 'o':
        out_l = AF._lin(o)
    else:
        g = AF._key(t['glow_keys'], f).astype(np.float32)
        xx, yy = AF._grid(H, W)
        d = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / (t.get('glow_r', 900.0) * k)
        out_l = (g[None, None, :] * (t.get('glow_floor', 0.55) + (1 - t.get('glow_floor', 0.55))
                                     * np.exp(-d * d))[..., None]).astype(np.float32)
    rim_c = np.float32(t.get('rim', (1.0, 0.72, 0.38))) * t.get('rim_gain', 0.35)
    out = out_l * (1.0 - a[..., None]) + vis_l * a[..., None] + rim[..., None] * rim_c
    if fm is not None:
        out = out * (1.0 - fm[..., None]) + AF._lin(i) * fm[..., None]
    if t.get('enter'):
        w = AF._ss(*t['enter'], f)
        out = AF._lin(i) * (1 - w) + out * w
    if t.get('land'):
        w = AF._ss(*t['land'], f)
        out = out * (1 - w) + AF._lin(i) * w
    return out, (cx, cy)


def first_pass_white_entry(o, i, f, t):
    """The rejected first-pass A9 entry: a smoothstep reveal of the vision through white, in linear light."""
    out, (cx, cy) = legacy_vision(o, i, f, t)
    H, W = i.shape[:2]
    p = AF._ss(2640, 2676, f)
    xx, yy = AF._grid(H, W)
    radius = np.sqrt(((xx - cx) / W) ** 2 + ((yy - cy) / W) ** 2)
    reveal = np.clip(p * 1.65 - radius * 0.65, 0.0, 1.0)
    reveal = reveal * reveal * (3.0 - 2.0 * reveal)
    if f >= 2676:
        reveal[:] = 1.0
    return np.clip(AF._srgb(1.0 + (out - 1.0) * reveal[..., None]), 0.0, 1.0)


@contextmanager
def synthetic_picture(source, cut='A', windows=None):
    """Record source coordinates and finish invocations without constructing Ctx."""
    ctx = AS.Ctx.__new__(AS.Ctx)
    ctx.cut, ctx.W, ctx.H = cut, 64, 27
    reads, finishes = [], []

    def raw(_ctx, frame):
        reads.append(frame)
        return source(frame).copy(), {'sec': str(frame)}, 'synthetic', {'stem': str(frame)}

    class Finisher:
        def __call__(self, img, src, name, frame):
            finishes.append((int(src['stem']), name, frame))
            return img * np.float32(0.91) + np.float32(0.025)

    fin = Finisher()

    def outer(frame):
        img, shot, status, src = raw(ctx, frame)
        return fin(img, src, cut, frame), shot, status, src

    ctx.picture = outer
    patches = [mock.patch.object(AS.Ctx, 'picture', raw),
               mock.patch.dict(sys.modules, stage=SimpleNamespace(Finisher=lambda look: fin))]
    if windows is not None:
        patches.append(mock.patch.dict(EDL.TRANS, A=windows))
    for p in patches:
        p.start()
    try:
        yield AS._transitions(ctx, True), reads, finishes, fin
    finally:
        for p in reversed(patches):
            p.stop()


def distinct_source(f):
    """Every frame its own colour and texture, so any pixel borrowed from another frame is detectable."""
    rng = np.random.default_rng(f)
    return np.clip(detailed_plate(shift=(f % 17) / 17.0) * np.float32(0.6 + 0.4 * rng.random(3)), 0, 1)


# ------------------------------------------------------------------------------------------ own pixels only
def assert_own_pixels(picture, frames):
    """Each output frame must equal its own finished plate times a gain field: no other frame's pixels."""
    for f in frames:
        own = distinct_source(f) * np.float32(0.91) + np.float32(0.025)
        t = AS.transition_at('A', f)
        if t['kind'] == 'iceheart':
            own = legacy_iceheart(own, own, f, t)
        g = t.get('exposure', {}).get(f)
        expected = own if g is None else AF.gain_field(own, g[0], g[1])
        np.testing.assert_allclose(picture(f)[0], expected, rtol=0, atol=2e-6,
                                   err_msg=f'A{f} must be its own plate times a gain field (no borrowed pixels)')


BRINK_FRAMES = sorted({f for b in BEATS_RIM + BEATS_FALL + CUTS + (2500,) for f in range(b - 1, b + 12)} - {2635})


def test_every_brink_frame_is_its_own_plate_times_a_gain_field():
    frames = [f for f in BRINK_FRAMES if 2400 <= f < 2635]
    with synthetic_picture(distinct_source) as (picture, _, _, _):
        assert_own_pixels(picture, frames)
    # Negative control: the first pass's four-frame bridges (a held previous frame dissolved into the live ones).
    bridged = deepcopy(EDL.TRANS['A'])
    bridged = [t for t in bridged if t['kind'] != 'exposure']
    bridged += [dict(f0=b, f1=b + 4, cut=b, kind='dissolve') for b in (2400, 2420, 2440, 2460, 2480, 2500)]
    with synthetic_picture(distinct_source, windows=bridged) as (picture, _, _, _):
        with pytest.raises(AssertionError, match='borrowed pixels'):
            assert_own_pixels(picture, [2401, 2461])


def test_fall_ignores_the_held_outgoing_plate_and_rejects_the_entry_blend():
    config = window('iceheart', 2520)
    o, i = detailed_plate(0.3), detailed_plate(0.7)
    for f in (2520, 2540, 2541, 2602, 2634):
        np.testing.assert_array_equal(AF.iceheart(o, i, f, config), AF.iceheart(plate((0.9, 0.1, 0.1)), i, f, config))

    def first_pass_entry(o, i, f, t):
        a = AF._ss(2540, 2544, f)                  # the rejected blend of the held previous beat into the live one
        blend = AF._srgb(AF._lin(o) * (1 - a) + AF._lin(i) * a)
        return legacy_iceheart(blend, blend, f, t)

    with pytest.raises(AssertionError):
        np.testing.assert_array_equal(first_pass_entry(o, i, 2541, config),
                                      first_pass_entry(plate((0.9, 0.1, 0.1)), i, 2541, config))


# ------------------------------------------------------------------------------------------ the gain field
def test_gain_field_is_exact_identity_at_unity_and_protects_the_fire_cores():
    img = np.concatenate([plate((0.98, 0.97, 0.95), (27, 32)), plate((0.22, 0.14, 0.08), (27, 32))], axis=1)
    assert AF.gain_field(img, 1.0, 1) is img
    dim = AF.gain_field(img, 0.5, 1)
    core, shadow = AF._lin(img[:, :4]).mean(), AF._lin(img[:, -4:]).mean()
    assert abs(AF._lin(dim[:, :4]).mean() / core - 1.0) < 0.01, 'a protected gain must leave the cores'
    assert abs(AF._lin(dim[:, -4:]).mean() / shadow - 0.5) < 0.01, 'the shadows take the full gain'
    flat = AF.gain_field(img, 0.5, 0)
    np.testing.assert_allclose(AF._lin(flat), AF._lin(img) * 0.5, rtol=1e-5, atol=1e-6)
    with mock.patch.object(AF, 'gain_field', side_effect=lambda im, g, p=0.0: AF._srgb(AF._lin(im) * g)):
        broken = AF.gain_field(img, 0.5, 1)
    with pytest.raises(AssertionError):
        assert abs(AF._lin(broken[:, :4]).mean() / core - 1.0) < 0.01


def assert_beat_shapes(table):
    for b in BEATS_RIM + BEATS_FALL:
        assert table.get(b, (1.0, 1))[0] < 1.0, f'A{b}: the beat frame must be held under its one-frame pop'
    for b in BEATS_FALL:
        g = [table.get(f, (1.0, 1))[0] for f in range(b, b + 11)]
        assert g[0] < g[1] < 1.0 < g[2], f'A{b}: three-frame attack onto the pulse peak'
        assert all(x >= y for x, y in zip(g[2:], g[3:])) and g[10] == 1.0, f'A{b}: release eases to unity in 8'


def test_beats_rise_over_three_frames_and_the_fall_releases_over_eight():
    table = {**window('exposure')['exposure'], **window('iceheart', 2520)['exposure']}
    assert_beat_shapes(table)
    with pytest.raises(AssertionError, match='held under'):
        assert_beat_shapes({})


def expected_cut(cut, after):
    """The production window's own treatment of the incoming plate at a cut (captured before any patch)."""
    t = next(t for t in EDL.TRANS['A'] if t['f0'] <= cut < t['f1'])
    own = after * np.float32(0.91) + np.float32(0.025)
    if t['kind'] == 'iceheart':
        own = legacy_iceheart(own, own, cut, t)
    g = t['exposure'][cut]
    assert g[0] != 1.0, 'the cut must carry an exposure match'
    return [AF.gain_field(own, g[0], g[1])]


def assert_cut_hard(picture, cut, expected):
    np.testing.assert_allclose(picture(cut)[0], expected, rtol=0, atol=2e-6,
                               err_msg=f'A{cut} must cut hard: only the incoming view, exposure-matched')


@pytest.mark.parametrize('cut', CUTS)
def test_camera_cuts_stay_hard_on_their_beat_and_reject_a_bridge(cut):
    red, blue = plate((0.8, 0.1, 0.05)), plate((0.05, 0.2, 0.7))
    source = lambda f: red if f < cut else blue
    expected = expected_cut(cut, blue)[0]
    with synthetic_picture(source) as (picture, _, _, _):
        assert_cut_hard(picture, cut, expected)
    # Negative control: the first pass's bridge, a four-frame dissolve from the held outgoing view on the cut.
    bridged = [t for t in deepcopy(EDL.TRANS['A']) if not t['f0'] <= cut < t['f1']]
    bridged.append(dict(f0=cut, f1=cut + 4, cut=cut, kind='dissolve'))
    with synthetic_picture(source, windows=bridged) as (picture, _, _, _):
        with pytest.raises(AssertionError, match='cut hard'):
            assert_cut_hard(picture, cut, expected)


# ------------------------------------------------------------------------------------------ the white impact
def assert_white_arrival(picture):
    images = [picture(f)[0] for f in range(2635, 2641)]
    means = [float(x.mean()) for x in images]
    assert float(images[0].std()) > 0.03, 'the flood must begin on a detailed exposure'
    assert max(means[:-1]) < 0.99, 'full white must wait for the impact frame'
    np.testing.assert_allclose(images[-1], 1.0, rtol=0, atol=2e-7,
                               err_msg='the impact frame must be full white, including its corners')
    assert np.all(np.diff(means) > 0), 'the flood must keep rising into the impact'


def clipping_fall(f):
    """Like the delivered fall: the plate floods on its own from 2636 (mean luma 200 -> 227 -> 253 -> 255 at
    2636-2639, finished frames) and is clipped white from 2638."""
    own = {2635: 0.0, 2636: 0.05, 2637: 0.35}
    if f >= 2638:
        return plate((1, 1, 1))
    w = own.get(f, 0.0)
    return AF._srgb(AF._lin(detailed_plate()) * (1 - w) + w)


def test_fall_reaches_white_on_the_impact_frame_and_rejects_the_clipped_tail():
    source = clipping_fall
    with synthetic_picture(source) as (picture, _, _, _):
        assert_white_arrival(picture)
    with synthetic_picture(source, windows=impact_cut_at(2640)) as (picture, _, _, _):
        with pytest.raises(AssertionError, match='full white must wait'):
            assert_white_arrival(picture)


def impact_cut_at(cut):
    """A's windows with the impact window's cut moved (2640: the first pass's coordinates without its hold, where o
    plays the source's own clipped tail on 2638-2639)."""
    windows = deepcopy(EDL.TRANS['A'])
    next(t for t in windows if t['kind'] == 'impact_white')['cut'] = cut
    return windows


def assert_held_source_contract(picture, reads, finishes):
    for frame in range(2635, 2640):
        reads.clear()
        finishes.clear()
        picture(frame)
        held, incoming = min(frame, 2637), max(frame, 2638)
        assert reads == [held, incoming], 'the impact holds 2637, its last detailed exposure, after 2637'
        assert finishes == [(held, 'A', frame), (incoming, 'A', frame)], 'each side must be finished exactly once'


def test_impact_source_coordinates_and_finish_once_have_a_hold_negative_control():
    # the hold is the assembler's own o/i convention at the window's cut (2638); no assembler change is involved
    with synthetic_picture(lambda f: detailed_plate()) as (picture, reads, finishes, _):
        assert_held_source_contract(picture, reads, finishes)
    with synthetic_picture(lambda f: detailed_plate(), windows=impact_cut_at(2640)) as (picture, reads, finishes, _):
        with pytest.raises(AssertionError, match='holds 2637'):
            assert_held_source_contract(picture, reads, finishes)


def assert_cache_sees_held_source(config):
    version = {2637: 'before'}

    def sources(cut, variant, plan, frame):
        return [f'source:{frame}:{version.get(frame, "unchanged")}']

    with mock.patch.object(AS, 'plan_shot', return_value={}), \
            mock.patch.object(D, 'frame_sources', side_effect=sources), \
            mock.patch.object(D, '_TRANS_CODE', [{'': 'core', 'impact_white': 'kind'}]):
        before = D.transition_sources('A', None, config, 2639)
        version[2637] = 'changed'
        after = D.transition_sources('A', None, config, 2639)
    assert before != after, 'changing the held exposure must invalidate delivery cache sources'
    assert 'source:2637:changed' in after


def test_delivery_dependencies_follow_the_held_exposure_and_reject_old_coordinates():
    # deliver.transition_sources is the delivered code, unchanged: it keys both sides at the window's cut
    assert_cache_sees_held_source(window('impact_white'))
    with pytest.raises(AssertionError, match='invalidate delivery cache'):
        assert_cache_sees_held_source(next(t for t in impact_cut_at(2640) if t['kind'] == 'impact_white'))


@pytest.mark.parametrize('kind', ['impact_white', 'exposure'])
def test_afix_wrapped_kinds_key_on_wrapper_and_compositor(kind):
    config = window(kind)
    original = AS.inspect.getsource
    version = {'wrapper': 'wrapper-v1', 'comp': 'comp-v1'}

    def source(obj):
        if obj is AS.AFIX_WRAPPED[kind]:
            return version['wrapper']
        if obj is AS.AFIX:
            return version['comp']
        return original(obj)

    def key():
        with mock.patch.object(D, '_TRANS_CODE', []):
            return D.transition_sources('A', None, config, config['f0'])[0]

    def check():
        before = key()
        version['wrapper'] = version['wrapper'] + '+'
        wrapped = key()
        assert wrapped != before, 'changing the executed wrapper must invalidate the cache'
        version['comp'] = version['comp'] + '+'
        assert key() != wrapped, 'changing the compositor must invalidate the cache'

    with mock.patch.object(AS.inspect, 'getsource', side_effect=source), \
            mock.patch.object(AS, 'plan_shot', return_value={}), \
            mock.patch.object(D, 'frame_sources', return_value=[]):
        check()
        with mock.patch.object(AS, 'transition_code', side_effect=lambda k: AS.inspect.getsource(AS.AFIX)):
            with pytest.raises(AssertionError, match='executed wrapper'):
                check()


# ------------------------------------------------------------------------------------------ out of the white
def valley_stimulus():
    o = plate((1, 1, 1), (60, 144))
    yy, xx = np.mgrid[0:60, 0:144].astype(np.float32)
    i = (0.05 + 0.04 * ((xx // 6 + yy // 6) % 2))[..., None] * np.float32([0.9, 0.95, 1.0])
    return o, i.astype(np.float32)


def assert_valley_emerges(render, config):
    o, i = valley_stimulus()
    frames = range(2640, 2690)
    images = [render(o, i, f, config) for f in frames]
    np.testing.assert_allclose(images[0], 1.0, rtol=0, atol=2e-7,
                               err_msg='the valley must open on the impact white over the entire frame')
    means = np.array([float(x.mean()) for x in images]) * 255
    steps = -np.diff(means)
    assert means[1] < means[0], 'the white must start lifting on the frame after the impact'
    assert np.all(steps >= -1e-3), 'the valley must emerge without exposure reversals'
    assert steps.max() < 2.0 * steps[:44].mean(), 'no single frame may carry the emergence (a switched exposure)'
    live, _ = legacy_vision(o, i, 2684, config)
    np.testing.assert_array_equal(images[44], np.clip(AF._srgb(live), 0.0, 1.0),
                                  err_msg='at 2684 the veil must be gone: the live vision, bit for bit')


def test_valley_emerges_from_white_without_a_spike_and_rejects_the_first_pass_entry():
    config = window('vision', 2640)
    q = [k[1] for k in config['veil']['q']]
    assert q[0] == 0.0 and q[-1] == 1.0 and all(b >= a for a, b in zip(q, q[1:]))
    assert_valley_emerges(AF.vision, config)
    with pytest.raises(AssertionError, match='switched exposure'):
        assert_valley_emerges(first_pass_white_entry, config)
    no_veil = deepcopy(config)
    no_veil.pop('veil')
    with pytest.raises(AssertionError, match='impact white'):
        assert_valley_emerges(AF.vision, no_veil)


def test_vision_veil_is_an_explicit_opt_in_and_a5_is_bit_identical():
    o, i = detailed_plate(), np.flip(detailed_plate(), axis=1).copy()
    a5 = window('vision', 1280)
    assert 'veil' not in a5
    for frame in (1280, 1298, 1340, 1400, 1439):
        out, _ = legacy_vision(o, i, frame, a5)
        np.testing.assert_array_equal(AF.vision(o, i, frame, a5), np.clip(AF._srgb(out), 0.0, 1.0))
    accidental = dict(a5, veil=dict(delay=0.4, gamma=2.4, q=[(1280, 0.0), (1324, 1.0)]))
    out, _ = legacy_vision(o, i, 1300, a5)
    with pytest.raises(AssertionError):
        np.testing.assert_array_equal(AF.vision(o, i, 1300, accidental), np.clip(AF._srgb(out), 0.0, 1.0))


def test_iceheart_without_an_exposure_entry_is_bit_identical_to_a_fix():
    o, i = detailed_plate(), np.flip(detailed_plate(), axis=0).copy()
    config = window('iceheart', 2520)
    plain = deepcopy(config)
    plain.pop('exposure')
    for frame in (2519, 2520, 2528, 2560, 2600, 2639):
        np.testing.assert_array_equal(AF.iceheart(o, i, frame, plain), legacy_iceheart(o, i, frame, plain))
    for frame in (2530, 2550, 2570, 2590, 2610, 2630):
        assert frame not in config['exposure']
        np.testing.assert_array_equal(AF.iceheart(o, i, frame, config), legacy_iceheart(o, i, frame, config))
    with pytest.raises(AssertionError):
        np.testing.assert_array_equal(AF.iceheart(o, i, 2600, config), legacy_iceheart(o, i, 2600, plain))


def test_polish_windows_stay_inside_2400_3360():
    polished = ('exposure', 'veil', 'flood', 'push')
    for t in EDL.TRANS['A']:
        if any(k in t for k in polished):
            assert 2400 <= t['f0'] and t['f1'] <= 3360, t['note']
            for f in t.get('exposure', {}):
                assert t['f0'] <= f < t['f1'], (t['note'], f)
    rogue = deepcopy(EDL.TRANS['A']) + [dict(f0=2300, f1=2320, kind='exposure', exposure={2310: (0.9, 1)},
                                             note='rogue')]
    with pytest.raises(AssertionError):
        for t in rogue:
            if any(k in t for k in polished):
                assert 2400 <= t['f0'] and t['f1'] <= 3360, t['note']
