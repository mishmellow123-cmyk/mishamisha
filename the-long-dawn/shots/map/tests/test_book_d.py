"""Cut D uses D-numbered frames and preserves the approved earlier entry point."""
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import book_d as D
import lastleaf as L


@pytest.mark.parametrize('shot,count', [('oldfire', 160), ('lastleaf', 320), ('title', 240)])
def test_d_ranges_include_both_endpoints_and_reject_local_frames(shot, count):
    start, end = D.RANGES[shot]
    assert end-start == count
    assert D.phase(shot, start) == 0.
    assert D.phase(shot, end-1) == 1.
    for invalid in (0, count-1, start-1, end, float(start), True):
        with pytest.raises(ValueError):
            D.phase(shot, invalid)


def test_camera_move_spans_all_320_frames_and_matches_round2_endpoints():
    scene = D.make_renderer('lastleaf', scale=.05, ppc=10)
    old = L.make_renderer(True, scale=.05, ppc=10, plate=True)
    for frame, old_time in ((6320, 0.), (6639, 239/24)):
        new_book, new_camera = scene.layout_at(frame)
        old_book, old_camera = old.layout(old_time)
        np.testing.assert_array_equal(new_camera.pos, old_camera.pos)
        np.testing.assert_array_equal(new_camera.target, old_camera.target)
        np.testing.assert_array_equal(new_book.params, old_book.params)
    at_240, at_end = (scene.layout_at(f)[1] for f in (6560, 6639))
    assert np.linalg.norm(at_240.pos-at_end.pos) > .05
    with pytest.raises(AssertionError):
        np.testing.assert_array_equal(at_240.pos, at_end.pos)
    assert scene.gap_degrees == 50.
    assert old.gap_degrees == 30.
    np.testing.assert_array_equal(scene.blank.chan, 0.)
    np.testing.assert_array_equal(scene.leaf().chan[135:], 0.)


def test_dawn_is_present_on_entry_and_keeps_rising_after_frame240():
    values = np.array([D.dawn_state(D.phase('lastleaf', f)) for f in range(6320, 6640)])
    assert values[0, 1] > 0 and values[-1, 0] > 0
    assert np.all(np.diff(values[:, 0]) < 0)
    assert np.all(np.diff(values[:, 1]) > 0)
    assert values[-1, 1] > values[240, 1]+.3
    # Reusing the old ten-second clock plateaus too early and fails this gate.
    old = np.array([L.plate_light_state(f/24) for f in range(320)])
    with pytest.raises(AssertionError):
        assert old[-1, 1] > old[240, 1]+.3


def assert_slow_travel(positions):
    assert np.linalg.norm(positions[-1]-positions[0]) > 8.
    assert np.linalg.norm(np.diff(positions, axis=0), axis=1).max() < .08


def test_gilt_reflection_travels_without_one_frame_jumps():
    sources = np.array([D.gilt_reflection(D.phase('lastleaf', f)) for f in range(6320, 6640)])
    assert_slow_travel(sources[:, :3])
    assert np.all(np.diff(sources[:, 3:6], axis=0) > 0.)
    with pytest.raises(AssertionError):
        assert_slow_travel(np.repeat(sources[160:161, :3], 320, axis=0))
    jumped = sources[:, :3].copy()
    jumped[160, 0] += 10.
    with pytest.raises(AssertionError):
        assert_slow_travel(jumped)


def test_title_ignites_after20_frames_and_holds_dark_ink_to_the_cut():
    scene = D.make_renderer('title', scale=.05, ppc=16)
    title = scene.title()
    np.testing.assert_array_equal(title.texture(0.).chan, 0.)
    np.testing.assert_array_equal(title.texture(19/24).chan, 0.)
    assert title.texture(60/24).chan[..., 4].max() > .2
    end = title.texture(239/24).chan
    assert end[..., 0].max() > .9
    np.testing.assert_array_equal(end[..., 4], 0.)
    # A delayed R2 title does not ignite in the corresponding early X3 window.
    with pytest.raises(AssertionError):
        assert L.LastLeaf(96, 40, 16, title=True, plate=True).title().texture(60/24).chan[..., 4].max() > .2


def test_dedicated_title_framing_keeps_the_actual_letter_mask_large_and_inside_frame():
    scene = D.make_renderer('title', ppc=16)
    title = scene.title()
    rows, cols = np.nonzero(title.m > .5)
    u = (np.array([cols.min(), cols.max(), cols.min(), cols.max()])+title.c0+.5)/16
    v = (np.array([rows.min(), rows.min(), rows.max(), rows.max()])+title.r0+.5)/16
    def assert_legible(book, camera):
        pixels, _ = camera.project(book.page_to_world('R', u, v))
        assert np.ptp(pixels[:, 0]) > 850.
        assert pixels[:, 0].min() > 50. and pixels[:, 0].max() < camera.W-50.
        assert pixels[:, 1].min() > 50. and pixels[:, 1].max() < camera.H-50.
    for frame in (6940, 7040, 7119):
        assert_legible(*scene.layout_at(frame))
    # The whole-spread R2 insert is materially smaller than the X3 framing.
    with pytest.raises(AssertionError):
        assert_legible(*L.LastLeaf(1920, 804, 16, title=True, plate=True).layout(135/24))


def test_explicit_d_module_does_not_change_round2_rendering():
    before = L.make_renderer(True, scale=.05, ppc=12, plate=True).frame(120)
    enabled = D.make_renderer('lastleaf', scale=.05, ppc=12).frame(6480)
    after = L.make_renderer(True, scale=.05, ppc=12, plate=True).frame(120)
    for first, last in zip(before, after):
        np.testing.assert_array_equal(first, last)
    with pytest.raises(AssertionError):
        np.testing.assert_array_equal(before[0], enabled[0])


def test_cli_rejects_wrong_frame_numbering_before_rendering(tmp_path):
    for shot, frames in (('lastleaf', '0-319'), ('oldfire', '4960'), ('title', '6880-7120')):
        with pytest.raises(SystemExit) as exc:
            D.main(['--shot', shot, '--frames', frames, '--out', str(tmp_path/'frames')])
        assert exc.value.code == 2
    for kwargs in ({'scale': 0.}, {'scale': 2.}, {'ppc': 0}, {'ppc': True}):
        with pytest.raises(ValueError):
            D.make_renderer('lastleaf', **kwargs)


def test_oldfire_cli_dispatch_writes_d_numbered_rgb_and_matte(tmp_path, capsys):
    import json
    from PIL import Image
    out = tmp_path/'oldfire'
    D.main(['--shot', 'oldfire', '--frames', '4818', '--scale', '.05',
            '--ppc', '12', '--out', str(out)])
    receipt = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert receipt['shot'] == 'oldfire' and receipt['frame'] == 4818
    for directory in (out, tmp_path/'oldfire_matte'):
        assert [p.name for p in directory.iterdir()] == ['f_04818.png']
        with Image.open(directory/'f_04818.png') as image:
            assert image.size == (96, 40)
            assert np.asarray(image).max() > 0
