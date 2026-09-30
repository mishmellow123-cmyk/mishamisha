"""Closed-band pins and the physical opening of the opt-in last-leaf Ring.

Legacy hashes were measured from the unmodified ringpage implementation (68484e9).
These small page-channel rasters exercise actual ink/gilt output without tracing a book scene.
"""
import hashlib
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import ringpage as R


P = np.array([5., 4., 3.3, .4, 1.32, .46, 0., 1.3, 0., .12, .61, 0., 1., 1., 0.])
FILL_HASH = 'abb3630ae86869435e4b85f8d4dc2559b4c1de1f60fc5d1fc85cd3d0171baf85'
DRAWING_HASH = '844fa1ea291e857b1ee0fa1e981da00b393dff86807446f51472307ad521d0ce'
PAGE_HASHES = {
    .9: '090d86b3a885637157f81521864791c040ac8052c438093672ff5d0164cceb23',
    3.5: 'ea99ae97d077a2b8ce8c7532d94afc60a5cca7e021c3a22beefcd8a9d42be624',
    5.9: 'b12993ddb1e49dd93a18c078a89b46dfc1582cccc4cf606cfe794a91ee9cf8a1',
}


def digest(a):
    return hashlib.sha256(a.tobytes()).hexdigest()


def drawing(gap=None, outline=False, ppc=20.):
    ch = np.zeros((int(8*ppc), int(10*ppc), 6), np.float32)
    kwargs = {} if gap is None else {'gap_degrees': gap}
    R.draw_ring(ch, 0., 0., ppc, P, R.strip('outer'), R.strip('inner'), 0., 1., 0., **kwargs)
    if outline:
        S = R.Strokes()
        R.ring_outline(S, P, 77, **kwargs)
        ink, _ = R.pen.raster(S.pack(), 1e9, ppc, ch.shape[0], ch.shape[1], R.pen.INK)
        np.maximum(ch[..., 0], ink, out=ch[..., 0])
        ch[..., 2] *= 1.-.9*np.clip(ink, 0., 1.)
    return ch


def assert_open_gap(ch, ppc):
    # A window in the near wall, below the far inner wall's foot. It must expose bare parchment.
    assert np.count_nonzero(ch[int(5.7*ppc):int(6.3*ppc), int(4.8*ppc):int(5.2*ppc)]) == 0


def test_original_fill_and_finished_drawing_are_byte_identical():
    assert digest(drawing()) == FILL_HASH
    assert digest(drawing(outline=True)) == DRAWING_HASH
    for outline in (False, True):
        np.testing.assert_array_equal(drawing(outline=outline), drawing(0., outline=outline))


def test_original_animated_page_is_pinned_through_ring_heat_and_bead():
    default = R.RingPage(ppc=8)
    explicit_off = R.RingPage(ppc=8, open_band=False, gap_degrees=60.)
    zero = R.RingPage(ppc=8, open_band=True, gap_degrees=0.)
    for t, expected in PAGE_HASHES.items():
        a = default.texture(t).chan.copy()
        assert digest(a) == expected
        np.testing.assert_array_equal(a, explicit_off.texture(t).chan)
        np.testing.assert_array_equal(a, zero.texture(t).chan)


def test_opening_faces_viewer_and_cut_metal_has_ink_and_gilt():
    ppc = 45.
    for gap in (15., 30., 60.):
        ch = drawing(gap, outline=True, ppc=ppc)
        assert_open_gap(ch, ppc)
        for a in (np.pi/2-np.deg2rad(gap)/2, np.pi/2+np.deg2rad(gap)/2):
            corners = np.array([R._corner(P, part, a, s)
                                for part, s in ((0, 0.), (3, 0.), (3, 1.), (0, 1.))])
            x, y = corners.mean(axis=0)
            assert ch[int(y*ppc), int(x*ppc), 2] > .2
            for x, y in corners:
                window = ch[int(y*ppc)-2:int(y*ppc)+3, int(x*ppc)-2:int(x*ppc)+3]
                assert np.max(window[..., 0]) > .08
        # Back wall still carries gilt through the hole; removing the gap cannot erase it.
        assert np.max(ch[int(3.3*ppc):int(4.1*ppc), int(4.8*ppc):int(5.2*ppc), 2]) > .5
        assert np.isfinite(ch).all()
        assert not np.any(ch[..., [1, 3, 4, 5]])


def test_gap_angle_removes_more_near_band_as_it_widens():
    a, b = drawing(15.), drawing(60.)
    # Compare the frontmost foot; cut faces and the far inner wall cannot contribute here.
    near = np.s_[124:130, :, 2]
    assert np.count_nonzero(b[near]) < np.count_nonzero(a[near])


def test_leading_end_bisects_real_canonical_ink_and_has_two_unaltered_halves():
    source = R.strip('outer')
    saved = source.copy()
    u = R._half_glyph_u(source)
    col = int(u*source.shape[1])
    assert np.max(source[:, col]) > .1
    assert np.max(source[:, col-2:col]) > .1
    assert np.max(source[:, col+1:col+3]) > .1
    drawing(30.)
    np.testing.assert_array_equal(source, saved)
    # The no-ink negative control cannot masquerade as the half-formed canonical letter.
    empty = np.zeros_like(source)
    with pytest.raises(AssertionError):
        assert np.max(empty[:, int(R._half_glyph_u(empty)*empty.shape[1])]) > .1


def test_static_leaf_is_cached_and_bare_below_drawing():
    page = R.RingLeaf(ppc=10)
    tex = page.texture()
    assert page.gap_degrees == 30.
    assert page.texture(10.) is tex
    assert np.count_nonzero(tex.chan[110:]) == 0
    assert np.count_nonzero(tex.chan[..., 2]) > 0
    assert not np.any(tex.chan[..., [1, 3, 4, 5, 6]])
    assert tex.data.dtype == np.float32
    assert tex.meta.shape == (7, 3)


@pytest.mark.parametrize('gap', [-1., 180., 181., np.inf, np.nan])
def test_invalid_gaps_are_rejected_before_drawing(gap):
    for call in (lambda: R.validate_gap(gap), lambda: drawing(gap),
                 lambda: R.ring_outline(R.Strokes(), P, 77, gap_degrees=gap),
                 lambda: R.RingLeaf(ppc=10, gap_degrees=gap)):
        with pytest.raises(ValueError):
            call()


def test_negative_controls_prove_legacy_and_gap_guards_fail():
    # Applying open geometry to an accepted closed shot must trip the legacy hash pin.
    with pytest.raises(AssertionError):
        assert digest(drawing(30.)) == FILL_HASH
    # Conversely, silently ignoring the requested gap must trip the exposed-parchment guard.
    with pytest.raises(AssertionError):
        assert_open_gap(drawing(0., outline=True), 20.)
    closed = R.RingLeaf(ppc=10, gap_degrees=0.).texture().chan
    opened = R.RingLeaf(ppc=10).texture().chan
    with pytest.raises(AssertionError):
        np.testing.assert_array_equal(closed, opened)


def test_first_round_leaf_remains_byte_identical_with_plate_available():
    assert digest(R.RingLeaf(ppc=10).texture().chan) == 'd7da247fda3117b910cad6927905211831b61bbf3f22c435039e25b5d87e70f2'


def assert_plate_proportions(params):
    # Measure the lip's projected cardinal points, independently of the wall height.
    cardinal = np.array([R._corner(params, 3, a, 0.) for a in (0., np.pi/2, np.pi, 3*np.pi/2)])
    width = np.ptp(cardinal[:, 0])
    height = np.ptp(cardinal[:, 1])
    assert width == pytest.approx(8.)
    assert height/width >= .6
    assert height/width == pytest.approx(.75)


def test_plate_has_eight_centimetre_ring_and_legible_on_page_ellipse():
    plate = R.RingPlate(ppc=10)
    assert_plate_proportions(plate.params)
    assert plate.params[0] == R.PG.PW/2
    assert plate.params[1]+plate.params[2]*plate.params[3]+plate.params[4] < R.PG.PH/2
    assert plate.gap_degrees == 30.
    ch = plate.texture().chan
    assert np.count_nonzero(ch[118:123, 98:102]) == 0
    # The earlier thin/small ellipse must fail the geometry guard even though wall height adds pixels.
    with pytest.raises(AssertionError):
        assert_plate_proportions(R.RingLeaf(ppc=10).params)
    too_flat = plate.params.copy(); too_flat[3] = .4
    with pytest.raises(AssertionError):
        assert_plate_proportions(too_flat)


def test_plate_uses_existing_double_rules_and_book_hand_with_blank_lower_leaf():
    with patch.object(R.PG, 'frame_rules', wraps=R.PG.frame_rules) as rules, \
            patch.object(R.pen, 'Hand', wraps=R.pen.Hand) as hand:
        plate = R.RingPlate(ppc=10)
    rules.assert_called_once()
    assert rules.call_args.args[1] == R.RingPlate.BOX
    hand.assert_called_once()
    ch = plate.texture().chan
    assert np.count_nonzero(ch[25:45, :, 0]) > 0
    assert np.count_nonzero(ch[46:49, 32:168, 0]) > 0
    assert np.count_nonzero(ch[int(R.RingPlate.BLANK_FROM*10):]) == 0
    assert np.count_nonzero(ch[..., [1, 3, 4, 5, 6]]) == 0
    spoiled = ch.copy(); spoiled[200, 100, 0] = 1.
    with pytest.raises(AssertionError):
        assert np.count_nonzero(spoiled[int(R.RingPlate.BLANK_FROM*10):]) == 0


def test_plate_gold_is_on_rim_and_local_shoulder_not_inner_wall_or_cut_faces():
    plate = R.RingPlate(ppc=20)
    ch, ppc = plate.texture().chan, plate.ppc
    P = plate.params
    # Uppermost rolled rim receives leaf; the middle of the inner wall remains ink on paper.
    assert ch[int((P[1]-P[3]*(P[2]-P[5]/2))*ppc), int(P[0]*ppc), 2] > .4
    assert np.max(ch[int(6.*ppc):int(6.7*ppc), int(9.*ppc):int(11.*ppc), 0]) > .5
    assert np.count_nonzero(ch[int(6.*ppc):int(6.7*ppc), int(9.*ppc):int(11.*ppc), 2]) == 0
    for th in (np.pi/2-np.pi/12, np.pi/2+np.pi/12):
        corners = np.array([R._corner(P, part, th, s)
                            for part, s in ((0, 0.), (3, 0.), (3, 1.), (0, 1.))])
        x, y = corners.mean(axis=0)
        assert ch[int(y*ppc), int(x*ppc), 2] == 0.
    # A full-metal version is a negative control for the ink-led surface treatment.
    metal = np.zeros_like(ch)
    R._draw_open_ring(metal, 0., 0., float(ppc), P, R.strip('outer'), R.strip('inner'), 0., 1., 0., 30.)
    with pytest.raises(AssertionError):
        assert np.count_nonzero(metal[int(6.*ppc):int(6.7*ppc), int(9.*ppc):int(11.*ppc), 2]) == 0


@pytest.mark.parametrize('kwargs', [{'aspect': .4}, {'aspect': np.nan}, {'radius': -1.},
                                  {'ppc': 0.}, {'center': (10., 15.)}])
def test_plate_rejects_geometry_that_cannot_meet_its_composition(kwargs):
    with pytest.raises(ValueError):
        R.RingPlate(**dict({'ppc': 10}, **kwargs))
