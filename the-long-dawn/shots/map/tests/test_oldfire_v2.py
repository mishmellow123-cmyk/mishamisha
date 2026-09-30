"""Close-camera readability, timing, unchanged sources and cropped texture safety."""
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import oldfire as R3
import oldfire_v2 as V


def ring_points(t):
    state = V.RP.RingPage.state(V.unmaking_time(t))[0]
    if state[11] >= .85:
        angle = np.linspace(0., 2*np.pi, 100)
        points = np.column_stack((state[0]+state[2]*np.cos(angle),
            state[1]+.5*state[4]+(state[3]*state[2]+.5*state[4])*np.sin(angle)))
    else:
        strokes = V.pen.Strokes()
        V.RP.ring_outline(strokes, state, 77)
        points = np.concatenate(strokes.P)
    return R3.mapped_point(points)


def project(scene, frame, points):
    book, camera = scene.layout(frame/24.)
    return camera.project(book.page_to_world('R', points[:, 0], points[:, 1]))[0]


def test_new_absolute_interval_leaves_r3_interval_untouched():
    scene = V.OldFireV2(96, 40, ppc=20)
    assert (V.START, V.END) == (5840, 6080)
    assert (R3.START, R3.END) == (4800, 4960)
    assert V.local_time(5840) == 0 and V.local_time(np.int64(6079)) == 239/24.
    for frame in (4800, 5839, 6080, 5840., True):
        with pytest.raises(ValueError, match='absolute integer'):
            scene.frame(frame)
    assert not scene.cache


def test_close_camera_makes_ring_and_bead_legible_and_keeps_them_inside_frame():
    scene = V.OldFireV2()
    for frame in (0, 20, 60, 100, 140):
        pixels = project(scene, frame, ring_points(frame/24.))
        assert pixels[:, 0].min() > 40 and pixels[:, 0].max() < 1880
        assert pixels[:, 1].min() > 40 and pixels[:, 1].max() < 764
        if frame in (0, 20, 60):
            assert 350 < np.ptp(pixels[:, 0]) < 550
        if frame == 140:
            assert np.ptp(pixels[:, 0]) > 90
    # The R3 wide camera is an actual negative control for the new scale gate.
    wide = R3.OldFire()
    book, camera = wide.layout(0.)
    points = ring_points(0.)
    pixels = camera.project(book.page_to_world('R', points[:, 0], points[:, 1]))[0]
    assert np.ptp(pixels[:, 0]) < 120


def test_close_covers_canonical_flare_extinction_and_bead_before_pull():
    state0 = V.RP.RingPage.state(V.unmaking_time(0.))
    flare = V.RP.RingPage.state(V.unmaking_time(60/24.))
    assert state0[1] == 0. and flare[1] > 1.
    assert V.CLOSE_END == 140 < V.CAPTION_START < V.PULL_END
    def require_bead_before_pull(clock):
        final = V.RP.RingPage.state(clock(V.CLOSE_END/V.FPS))
        assert final[0][11] == 1. and final[0][13] == 0.
        assert final[1] == 0. and final[2] == 0.
    require_bead_before_pull(V.unmaking_time)
    # Actual duration regression: spread the same clock over all 239 local
    # frame intervals. The camera would leave before the bead was complete.
    stretched = lambda t: V.unmaking_time(t*V.CLOSE_END/(V.END-V.START-1))
    with pytest.raises(AssertionError):
        require_bead_before_pull(stretched)


def test_full_caption_visible_while_writing_then_held_on_whole_page():
    scene = V.OldFireV2()
    caption = scene.caption()
    assert caption.text == R3.CAPTION
    assert caption.f_in == V.START+V.CAPTION_START
    assert caption.f_out >= V.END+12
    x0, x1 = caption.c0/caption.ppc, (caption.c0+caption.a.shape[1])/caption.ppc
    y0, y1 = caption.r0/caption.ppc, (caption.r0+caption.a.shape[0])/caption.ppc
    xx, yy = np.meshgrid(np.linspace(x0, x1, 12), [y0, y1])
    points = np.column_stack((xx.ravel(), yy.ravel()))
    for frame in (V.CAPTION_START, V.CAPTION_START+24, 239):
        pixels = project(scene, frame, points)
        assert pixels[:, 0].min() > 24 and pixels[:, 0].max() < 1896
        assert pixels[:, 1].min() > 24 and pixels[:, 1].max() < 780
    # Revealing the same caption at the start of the pull would hide it.
    early = project(scene, V.CLOSE_END, points)
    assert early[:, 1].max() > 804
    mountain = V.PG.Mountain(11)
    x0, y0, x1, y1 = mountain.box
    xx, yy = np.meshgrid(np.linspace(x0, x1, 12), np.linspace(y0, y1, 12))
    pixels = project(scene, 239, np.column_stack((xx.ravel(), yy.ravel())))
    assert pixels[:, 0].min() > 24 and pixels[:, 0].max() < 1896
    assert pixels[:, 1].min() > 24 and pixels[:, 1].max() < 780


@pytest.fixture(scope='module')
def texture_pair():
    full = V.B.PageTex(V.PG.PW, V.PG.PH, 64)
    yy, xx = np.mgrid[:full.H, :full.W]
    full.chan[..., 0] = (.15 + .0002*yy + .0001*xx).astype(np.float32)
    full.build()
    strip = V.DetailStrip(64)
    strip.chan[:] = full.chan[strip.row0:strip.row0+strip.H]
    strip.build()
    return full, strip


def test_cropped_mips_sample_the_identical_absolute_coordinates(texture_pair):
    full, strip = texture_pair
    for u in (3., 10.62, 17.):
        for v in (1.75, 2.05, 4.53, 5.95, 6.25):
            for footprint in (.002, .012, .04, .10, .30, 10.):
                expected = np.zeros(V.B.NCH)
                V.B.tex(full.data, full.meta, 64., u, v, footprint, expected)
                np.testing.assert_array_equal(strip.sample(u, v, footprint), expected)


def test_strip_offset_and_safety_guard_have_real_negative_controls(texture_pair):
    full, strip = texture_pair
    correct = strip.sample(10.62, 4.53, .02)
    original = strip.meta.copy()
    strip.meta[:, 0] += strip.meta[:, 2]*V.B.NCH
    try:
        assert not np.array_equal(strip.sample(10.62, 4.53, .02), correct)
    finally:
        strip.meta[:] = original
    for u, v in ((-1., 4.), (10., .5), (10., 8.)):
        with pytest.raises(ValueError, match='safe interval'):
            strip.sample(u, v, .02)
    with pytest.raises(ValueError, match='multiple of 64'):
        V.DetailStrip(110)
    # Reproduce the original too-narrow preview crop. Its coarsest filtered
    # samples reflect omitted rows instead of reading the actual page.
    short = V.DetailStrip(64)
    short.Y0, short.Y1, short.row0, short.H = 1., 7., 64, 384
    short.chan = full.chan[64:448].copy()
    short.build()
    assert not np.array_equal(short.sample(10.62, 6.25, 10.),
                              strip.sample(10.62, 6.25, 10.))


def test_actual_mountain_geometry_and_r3_outputs_are_preserved():
    old = R3.OldFire(96, 40, ppc=20)
    before = old.texture(4800).chan.copy()
    scene = V.OldFireV2(96, 40, ppc=20)
    strip = scene.detail(5840)
    _, strokes = R3.mountain_strokes()
    reference = R3.RB.Page(strokes, strip.ppc).texture(1e9).chan
    expected = reference[strip.row0:strip.row0+strip.H]
    # A substantial part of the Mountain around its dynamic throat: identical
    # strokes, raster density and absolute coordinate origin.
    left = np.s_[:, :int(8.5*strip.ppc), :]
    right = np.s_[:, int(12.5*strip.ppc):, :]
    np.testing.assert_array_equal(strip.chan[left], expected[left])
    np.testing.assert_array_equal(strip.chan[right], expected[right])
    assert expected[..., 0].sum() > 0
    scene.texture(6079)
    np.testing.assert_array_equal(before, old.texture(4800).chan)
    assert scene.ring_for(scene.detail_ppc).gap_degrees == 0.
    scene.ring_for(scene.detail_ppc).gap_degrees = 30.
    with pytest.raises(ValueError, match='closed Ring'):
        scene.detail(5840)


def test_ember_hot_centers_are_separate_and_other_drawing_is_unchanged():
    original = R3.MountainRing(ppc=54.4)
    corrected = V.MountainRingV2(ppc=54.4)
    class CoalsOnly(V.MountainRingV2):
        _fire = R3.MountainRing._fire
    coals_only = CoalsOnly(ppc=54.4)
    x0, y0, x1, y1 = original.WIN
    ox, oy = int(x0*original.ppc)/original.ppc, int(y0*original.ppc)/original.ppc
    shape = (int(y1*original.ppc)-int(y0*original.ppc),
             int(x1*original.ppc)-int(x0*original.ppc), V.B.NCH)
    baseline, fixed = np.zeros(shape, np.float32), np.zeros(shape, np.float32)
    original._embers(baseline, ox, oy, 5.4, 1.)
    corrected._embers(fixed, ox, oy, 5.4, 1.)
    def require_separate_centers(channels):
        hot = (channels[..., 4] > .1).astype(np.uint8)
        count, labels, stats, centers = V.cv2.connectedComponentsWithStats(hot)
        assert count-1 == len(corrected.embers)
        assert stats[1:, V.cv2.CC_STAT_WIDTH].max()/corrected.ppc < .30
    require_separate_centers(fixed)
    with pytest.raises(AssertionError):
        require_separate_centers(baseline)
    # The actual canonical Ring, tongue and bead paths remain unchanged;
    # changes are confined to the source coal-bed band, far from patch edges.
    for t in (.2, 2.65, 5.4):
        old_texture = original.texture(t).chan.copy()
        new_texture = corrected.texture(t).chan
        low, high = int(18.1*original.ppc), int(19.15*original.ppc)
        np.testing.assert_array_equal(old_texture[:low], new_texture[:low])
        np.testing.assert_array_equal(old_texture[high:], new_texture[high:])
        coal_texture = coals_only.texture(t).chan
        if t == 2.65:
            # At the flare, the still-wide Ring/front flames cover this bed.
            np.testing.assert_array_equal(old_texture, coal_texture)
        else:
            assert not np.array_equal(old_texture[low:high], coal_texture[low:high])
        assert not np.array_equal(old_texture[low:high], new_texture[low:high])
        for actual, expected in zip(corrected.state(t), original.state(t)):
            np.testing.assert_array_equal(actual, expected)


def test_flame_feet_are_curved_without_changing_original_upper_body():
    original = R3.MountainRing(ppc=54.4)
    corrected = V.MountainRingV2(ppc=54.4)
    bx, by, h, width, lean = original.tongues[2]
    ox, oy = bx-2., by-5.
    shape = (int(6.*original.ppc), int(4.*original.ppc), V.B.NCH)
    before, after = np.zeros(shape, np.float32), np.zeros(shape, np.float32)
    one = [(bx, by, h, width, lean)]
    original._fire(before, ox, oy, 5.4, one, 1., 11)
    corrected._fire(after, ox, oy, 5.4, one, 1., 11)
    base_row = int((by-oy)*original.ppc)
    np.testing.assert_array_equal(before[:base_row], after[:base_row])
    def require_curved_foot(channels):
        mask = channels[..., 4] > .03
        x0 = int((bx-.6*width-ox)*original.ppc)
        x1 = int((bx+.6*width-ox)*original.ppc)
        bottom = mask.shape[0]-1-np.argmax(mask[::-1, x0:x1], axis=0)
        assert np.ptp(bottom) > 4
        assert bottom.max() > base_row+5
    require_curved_foot(after)
    with pytest.raises(AssertionError):
        require_curved_foot(before)
