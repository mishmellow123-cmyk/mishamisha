"""Last-leaf opt-in contracts, including a small real render of the off path."""
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import lastleaf as L
import book_c_v5_pen_candidates as C


def test_flag_off_is_pixel_exact_to_accepted_pen():
    accepted = C.CandidatePagesV5('soft_spine_metal', 96, 40, 12).frame(5560)
    omitted = L.make_renderer(scale=.05, ppc=12).frame(5560)
    explicit = L.make_renderer(open_leaf=False, scale=.05, ppc=12).frame(5560)
    for ref, a, b in zip(accepted, omitted, explicit):
        np.testing.assert_array_equal(ref, a)
        np.testing.assert_array_equal(ref, b)
    # A real enabled frame must fail the same equality gate.
    enabled = L.make_renderer(open_leaf=True, scale=.05, ppc=12).frame(120)
    with pytest.raises(AssertionError):
        np.testing.assert_array_equal(accepted[0], enabled[0])


def test_leaf_is_written_only_at_head_and_recto_stays_blank():
    renderer = L.make_renderer(True, scale=.05, ppc=12)
    left = renderer.leaf()
    assert left.chan[..., 0].max() > .3
    assert left.chan[..., 2].max() > .3
    np.testing.assert_array_equal(left.chan[int(12 * 12):], 0.)
    np.testing.assert_array_equal(renderer.blank.chan, 0.)
    np.testing.assert_array_equal(left.chan[..., 4], 0.)
    assert not renderer.with_title


def test_resting_spread_camera_and_pen_keep_accepted_geometry():
    renderer = L.make_renderer(True, scale=.05, ppc=12)
    first, cam = renderer.layout(0.)
    last, end_cam = renderer.layout(239 / 24.)
    assert first is last
    np.testing.assert_array_equal(cam.target, [.6, .2, 2.])
    np.testing.assert_allclose(cam.pos, [.1, -38.8, 41.], atol=1e-14, rtol=0.)
    np.testing.assert_array_equal(cam.target, end_cam.target)
    assert first.TL == 2.8 and first.TR == 1.2 and first.seed == 13


def test_dawn_and_hearth_envelopes_and_light_direction():
    values = np.array([L.light_state(t) for t in np.linspace(0., 10., 241)])
    assert np.all(np.diff(values[:, 0]) <= 0.)
    assert np.all(np.diff(values[:, 1]) >= 0.)
    assert values[-1, 0] > 0.
    assert L.DAWN_POS[0] < 0. and L.HEARTH_POS[0] < 0.
    assert L.DAWN_POS[2] / abs(L.DAWN_POS[0]) < L.HEARTH_POS[2] / abs(L.HEARTH_POS[0])


def test_title_reuses_recto_texture_and_leaves_ring_unchanged():
    renderer = L.make_renderer(True, scale=.05, ppc=16, title=True)
    ring = renderer.leaf().chan.copy()
    title = renderer.title()
    np.testing.assert_array_equal(title.texture(0).chan, 0.)
    assert title.texture(4.5).chan[..., 4].max() > 0.
    cold = title.texture(9.9).chan
    assert cold[..., 0].max() > .9
    np.testing.assert_array_equal(cold[..., 4], 0.)
    np.testing.assert_array_equal(renderer.leaf().chan, ring)


def test_invalid_frames_and_unrequested_title_fail():
    renderer = L.make_renderer(True, scale=.05, ppc=12)
    for f in (-1, 240, 1.5, True):
        with pytest.raises(ValueError):
            renderer.frame(f)
    with pytest.raises(ValueError):
        L.make_renderer(title=True)
    for gap in (-1, 180, np.nan):
        with pytest.raises(ValueError):
            L.make_renderer(True, gap_degrees=gap)


def test_shared_renders_and_aliases_cannot_be_written(tmp_path, monkeypatch):
    # Use an owned artificial lane tree; this also runs on farms without the
    # workstation's renders symlink. Negative controls cover RGB and matte.
    repo = tmp_path / 'lane'
    (repo / 'shots/map').mkdir(parents=True)
    shared = tmp_path / 'owner'
    shared.mkdir()
    (repo / 'renders').symlink_to(shared)
    monkeypatch.setattr(L, '__file__', str(repo / 'shots/map/lastleaf.py'))
    alias = tmp_path / 'alias'
    alias.symlink_to(shared)
    for target in (repo / 'renders/new', alias / 'new'):
        with pytest.raises(ValueError):
            L.output_dir(target)
    output = tmp_path / 'study'
    output.with_name('study_matte').symlink_to(shared)
    with pytest.raises(ValueError):
        L.output_dir(output)
    assert L.output_dir(tmp_path / 'safe') == tmp_path / 'safe'


def test_plate_option_preserves_earlier_defaults_and_frames_the_writing():
    old = L.make_renderer(True, scale=.1, ppc=12)
    new = L.make_renderer(True, scale=.1, ppc=12, plate=True)
    assert old.plate is False and new.plate is True
    with pytest.raises(ValueError):
        L.make_renderer(plate=True)
    book, camera = new.layout(239 / 24.)
    points = book.page_to_world('L', np.array([3.3, 16.7, 3.3, 16.7]),
                                np.array([2.55, 2.55, 13.3, 13.3]))
    def assert_framed(cam):
        pixels, _ = cam.project(points)
        assert (pixels[:, 0] > 0).all() and (pixels[:, 0] < cam.W).all()
        assert (pixels[:, 1] > 0).all() and (pixels[:, 1] < cam.H).all()
    assert_framed(camera)
    # The accepted insert crops the first writing line; it cannot pass this guard.
    with pytest.raises(AssertionError):
        assert_framed(old.layout(239 / 24.)[1])
    np.testing.assert_array_equal(new.leaf().chan[int(13.5 * 12):], 0.)
    np.testing.assert_array_equal(new.blank.chan, 0.)


def test_front_left_dawn_reaches_recto_without_erasing_fold_shadow():
    scene = L.make_renderer(True, scale=.1, ppc=12, plate=True)
    book, camera = scene.layout(239 / 24.)
    def lit_fraction(position):
        light = L.B.Light(position, power=1., radius=2.4)
        _, _, geometry = L.B.render(book, camera, light, scene.blank, scene.blank, 0.)
        # Body of recto, away from its gutter and outer edge.
        region = ((geometry[..., 0] == L.B.M_PAGE_R) &
                  (geometry[..., 8] > 4.) & (geometry[..., 8] < 16.) &
                  (geometry[..., 9] > 6.) & (geometry[..., 9] < 23.))
        assert region.sum() > 100
        return np.mean(geometry[..., 11][region] > .5)
    assert lit_fraction(L.PLATE_DAWN_POS) > .8
    # R1's back-left, almost horizontal source must fail the same coverage check.
    with pytest.raises(AssertionError):
        assert lit_fraction(L.DAWN_POS) > .8
