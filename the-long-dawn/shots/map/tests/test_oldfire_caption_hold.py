"""Caption reading time without moving the adopted fall or letter flare."""
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import oldfire_v2 as V
import book_d_v2 as D


def r4_clock(t):
    return float(np.interp(t, [0., .55, .90, 2.5, 3.6, 140/24., 239/24.],
                          [.2, .2, .55, 2.65, 3.5, 5.4, 6.3]))


def test_clock_preserves_every_pre_change_frame_and_motion_blur_sample():
    for frame in range(82):
        t = frame/24.
        offsets = ((np.arange(9)+.5)/9./48.-1./96. if .55 < t < .97 else (0.,))
        for dt in offsets:
            assert V.unmaking_time(t+dt, True) == r4_clock(t+dt)
    assert V.HOLD_FIRST_CHANGED == 5922
    assert V.unmaking_time(82/24., True) != r4_clock(82/24.)
    for frame in range(240):
        assert V.unmaking_time(frame/24.) == r4_clock(frame/24.)


def require_sound_anchors(clock):
    states = [V.RP.RingPage.state(clock(f/24.)) for f in range(240)]
    fall = next(f for f, state in enumerate(states) if state[0][1] != states[0][0][1])
    flare = [f for f, state in enumerate(states) if state[1] > 0.]
    assert fall+V.START == 5854
    assert flare == list(range(5897-V.START, 5922-V.START))


def test_moved_fall_or_flare_fails_the_independent_sound_anchors():
    clock = lambda t: V.unmaking_time(t, True)
    require_sound_anchors(clock)
    for boundary in (20/24., 56/24.):
        # Perturb the fall alone, then the flare interval alone.
        moved = lambda t: clock(t+1/24. if (t < boundary if boundary < 1. else t >= boundary) else t)
        with pytest.raises(AssertionError):
            require_sound_anchors(moved)


def require_caption_hold(line):
    complete = line.f_in+24
    assert complete <= 5983
    assert V.END-complete >= 96
    assert line.f_out >= V.END+12


def test_late_complete_sentence_fails_and_accepted_write_style_is_exact():
    scene = V.OldFireV2(ppc=40, caption_hold=True)
    line = scene.caption()
    original = V.OldFireV2(ppc=40).caption()
    require_caption_hold(line)
    assert line.f_in == 5959 and line.f_in+24 == 5983
    for name in ('a', 'xx', 'yy', 'noise', 'dens'):
        np.testing.assert_array_equal(getattr(line, name), getattr(original, name))
    shape = (line.r0+line.a.shape[0]+1, line.c0+line.a.shape[1]+1, V.B.NCH)
    for offset in (0, 1, 12, 23, 24, 30, 96):
        actual, reference = np.zeros(shape, np.float32), np.zeros(shape, np.float32)
        line.apply(actual, line.f_in+offset)
        original.apply(reference, original.f_in+offset)
        # Late reference frames are outside R4's hold; style pin covers reveal/drying.
        if offset <= 30:
            np.testing.assert_array_equal(actual, reference)
    complete = np.zeros(shape, np.float32)
    line.apply(complete, 5983)
    region = complete[line.r0:line.r0+line.a.shape[0], line.c0:line.c0+line.a.shape[1], 0]
    assert np.all(region >= line.a*line.dens-1e-7)
    line.f_in += 1
    with pytest.raises(AssertionError):
        require_caption_hold(line)
    with pytest.raises(AssertionError):
        require_caption_hold(original)


def test_caption_enters_before_reveal_and_bead_stays_large_before_pull():
    scene = V.OldFireV2(caption_hold=True)
    line = scene.caption()
    x0, x1 = line.c0/line.ppc, (line.c0+line.a.shape[1])/line.ppc
    y0, y1 = line.r0/line.ppc, (line.r0+line.a.shape[0])/line.ppc
    xx, yy = np.meshgrid(np.linspace(x0, x1, 32), [y0, y1])
    for frame in range(5958, 6080):
        book, camera = scene.layout((frame-V.START)/24.)
        p, _ = camera.project(book.page_to_world('R', xx.ravel(), yy.ravel()))
        assert p[:, 0].min() > 24 and p[:, 0].max() < 1896
        assert p[:, 1].min() > 24 and p[:, 1].max() < 780
    angle = np.linspace(0., 2*np.pi, 100)
    for local in range(91, 100):
        state, glow, engraving, _ = V.RP.RingPage.state(V.unmaking_time(local/24., True))
        assert state[11] == 1. and glow == 0. and engraving == 0.
        points = np.column_stack((state[0]+state[2]*np.cos(angle),
            state[1]+.5*state[4]+(state[3]*state[2]+.5*state[4])*np.sin(angle)))
        uv = V.R3.mapped_point(points)
        book, camera = scene.layout(local/24.)
        p, _ = camera.project(book.page_to_world('R', uv[:, 0], uv[:, 1]))
        assert np.ptp(p[:, 0]) > 119.


def test_both_camera_motions_have_zero_endpoint_velocity():
    scene = V.OldFireV2(caption_hold=True)
    def values(frame):
        cam = scene.layout(frame/24.)[1]
        return cam.target, cam.pos-cam.target
    eps = 1e-5
    for index, finish in ((0, 99+24/1.18), (1, 123.)):
        for frame in (99., finish):
            derivative = (values(frame+eps)[index]-values(frame-eps)[index])/(2*eps)
            assert np.linalg.norm(derivative) < 1e-5


def test_opt_in_does_not_change_the_r4_frames_or_other_shots():
    old = V.OldFireV2(96, 40, ppc=20)
    new = V.OldFireV2(96, 40, ppc=20, caption_hold=True)
    for frame in (5840, 5854, 5897, 5921):
        for a, b in zip(old.frame(frame), new.frame(frame)):
            np.testing.assert_array_equal(a, b)
    for shot in ('lastleaf', 'title'):
        with pytest.raises(ValueError, match='only defined for oldfire'):
            D.make_renderer(shot, caption_hold=True)
