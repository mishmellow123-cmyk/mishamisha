"""D-v2 dispatch, retained title, neighbour lights and last-leaf composition."""
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import book_d_v2 as D
import book_d as D1
import lastleaf_v2 as L


@pytest.mark.parametrize('shot,count', [('oldfire', 240), ('lastleaf', 320), ('title', 240)])
def test_v2_ranges_and_old_numbering_rejection(shot, count):
    first, end = D.RANGES[shot]
    assert end-first == count
    assert D.phase(shot, first) == 0. and D.phase(shot, end-1) == 1.
    for bad in (D1.RANGES[shot][0], 0, first-1, end, float(first), True):
        with pytest.raises(ValueError):
            D.phase(shot, bad)


def test_title_is_exactly_the_retained_r3_renderer_with_only_a_frame_offset():
    old = D1.DSpread('title', 96, 40, 12)
    new = D.RetainedTitle(96, 40, 12)
    for frame in (8880, 8940, 9119):
        for actual, expected in zip(new.frame(frame), old.frame(frame-2000)):
            np.testing.assert_array_equal(actual, expected)
    with pytest.raises(ValueError):
        new.frame(6880)


def test_lamp_endpoints_use_the_actual_neighbour_light_states():
    first, last = L.neighbour_lights()
    # Evaluated independently from accepted C6399 and C5440 source states;
    # the native reference receipt also records their image patch statistics.
    np.testing.assert_allclose(first.pos, [-54.44436780698207, 50.35679050063251,
                                         30.440660018411766], atol=1e-12)
    np.testing.assert_allclose(first.col, [2.234591064045053, 1.1619873533034275,
                                         .5363018553708127], atol=1e-12)
    np.testing.assert_allclose(last.pos, [-55., 50., 30.], atol=1e-12)
    np.testing.assert_allclose(last.col, [1.8394974142303668, .9565386553997908,
                                        .44147937941528803], atol=1e-12)
    for q, original, gain in ((0., first, L.START_GAIN), (1., last, L.END_GAIN)):
        light = L.bridge_light(q)
        np.testing.assert_allclose(light.pos, original.pos, atol=1e-13)
        np.testing.assert_allclose(light.col/gain, original.col, atol=1e-13)
        assert light.radius == 10.
    assert first.col[0] > last.col[0]
    assert np.all(L.START_GAIN > 0.) and np.all(L.END_GAIN > 0.)
    # The v1 rose dawn has neither of these source positions/powers.
    with pytest.raises(AssertionError):
        np.testing.assert_allclose(L.bridge_light(1.).pos, D1.L.PLATE_DAWN_POS)


def test_camera_visits_ring_break_and_blank_page_with_a_calm_caption_band():
    scene = L.LastLeafV2(ppc=12)
    book, start = scene.layout_at(8320)
    ring = book.page_to_world('L', np.array([6., 14.]), np.array([8.3, 8.3]))
    pixels, _ = start.project(ring)
    assert np.ptp(pixels[:, 0]) > 500.
    mesh = L.A.geometry(book)
    for frame in range(8320, 8640):
        bk, camera = scene.layout_at(frame)
        tip, _ = camera.project(L.A.pen_tip(bk, mesh)[None, :])
        assert 100 < tip[0, 0] < camera.W-100
        assert 50 < tip[0, 1] < camera.H*.65
        pen_pixels, _ = camera.project(mesh[0])
        # The pen and its writing stop never occupy the lower caption band.
        assert pen_pixels[:, 1].max() < camera.H*.8
    _, middle = scene.layout_at(8480)
    _, end = scene.layout_at(8639)
    assert start.target[0] < middle.target[0] < end.target[0]
    assert end.target[0] > 5.
    travel = np.array([scene.layout_at(frame)[1].target[0] for frame in range(8320, 8640)])
    assert np.all(np.diff(travel) >= -1e-12)
    # A quadratic through these three compositions reverses late in the shot.
    points = [scene.layout_at(frame)[1].target for frame in (8320, 8480, 8639)]
    bad = np.array([L.bezier(*points, q)[0] for q in np.linspace(0., 1., 320)])
    with pytest.raises(AssertionError):
        assert np.all(np.diff(bad) >= -1e-12)
    # The earlier full-spread camera does not provide this traverse.
    with pytest.raises(AssertionError):
        assert D1.DSpread().layout_at(6639)[1].target[0] > 5.


def test_cli_rejects_v1_before_instantiating_any_renderer(tmp_path):
    for shot, frames in (('oldfire', '4800-4959'), ('lastleaf', '6320'), ('title', '6880')):
        with pytest.raises(SystemExit) as error:
            D.main(['--shot', shot, '--frames', frames, '--out', str(tmp_path/'bad')])
        assert error.value.code == 2
    assert not list(tmp_path.iterdir())


def test_new_leaf_keeps_r2_renderer_output_unchanged():
    before = D1.L.make_renderer(True, scale=.05, ppc=12, plate=True).frame(120)
    D.make_renderer('lastleaf', scale=.05, ppc=12).frame(8480)
    after = D1.L.make_renderer(True, scale=.05, ppc=12, plate=True).frame(120)
    for first, last in zip(before, after):
        np.testing.assert_array_equal(first, last)
