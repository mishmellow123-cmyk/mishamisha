"""D21 composition, closed-Ring motion and isolated page-writing contracts."""
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import oldfire as O


def test_absolute_frame_interval_and_early_rejection():
    scene = O.OldFire(96, 40, ppc=20)
    assert O.local_time(4800) == 0.
    assert O.local_time(np.int64(4959)) == 159 / 24.
    for frame in (0, 4799, 4960, 4800., True, '4800'):
        with pytest.raises(ValueError, match='absolute integer'):
            scene.frame(frame)
    assert scene.cache == {}


def test_mountain_illustration_preserved_except_three_ring_strokes():
    original = O.PG.Mountain(11).build('ink', 1.1, 8.6, sched=O.PG.Mountain.SCHED_C3)
    tagged = O.PG.Mountain(11).build('ink', 1.1, 8.6,
                                    sched=dict(O.PG.Mountain.SCHED_C3, ring=(-2., -1.)))
    retained = [i for i, t in enumerate(tagged.T0) if t >= 0.]
    expected = O.pen.Strokes()
    for field in ('P', 'R', 'D', 'T0', 'T1', 'L', 'G', 'N'):
        setattr(expected, field, [getattr(original, field)[i] for i in retained])
    O.pen.Hand(seed=18, xh=.2).write_block(expected, 3.2, 15.05, 13.8, 18, .62)
    _, actual = O.mountain_strokes()
    assert len(original) - len(retained) == 3
    for field, array in expected.pack().items():
        np.testing.assert_array_equal(actual.pack()[field], array)


def test_changed_component_schedule_fails_instead_of_erasing_picture(monkeypatch):
    original = O.PG.Mountain.build
    def broken(self, mode, t0, t1, sched):
        return original(self, mode, t0, t1, sched=O.PG.Mountain.SCHED_C3)
    monkeypatch.setattr(O.PG.Mountain, 'build', broken)
    with pytest.raises(RuntimeError, match='stroke contract'):
        O.mountain_strokes()


def test_closed_canonical_ring_falls_flares_and_becomes_bead():
    states = [O.RP.RingPage.state(O.unmaking_time(t)) for t in (0., .5, .9, 3., 6.625)]
    np.testing.assert_array_equal(states[0][0], states[1][0])
    assert states[0][0][1] < states[2][0][1]
    assert states[2][0][1] == O.RP.RingPage.BED[1]
    assert states[2][1] == 0. and states[3][1] > 0.
    assert states[-1][0][11] >= .85
    assert states[-1][0][13] == 0. and states[-1][2] == 0.
    scene = O.OldFire(96, 40, ppc=20)
    ring = scene.ring_page()
    assert ring.gap_degrees == 0.
    row, col, patch = scene.fire_patch(0.)
    assert row > 0 and col > 0 and patch[..., 2].max() > .5
    ring.gap_degrees = 30.
    with pytest.raises(ValueError, match='closed Ring'):
        scene.fire_patch(0.)


def test_exact_caption_writes_on_page_and_holds_without_mutating_c_timing():
    original = dict(O.T1.IL.LINES)
    scene = O.OldFire(96, 40, ppc=20)
    line = scene.caption()
    assert line.text == 'In the old story, the Ring is unmade in the fire that forged it.'
    def caption(frame):
        channels = np.zeros((580, 400, 7), np.float32)
        line.apply(channels, frame)
        return channels
    assert not caption(4800).any()
    assert caption(4812)[..., 0].sum() < caption(4831)[..., 0].sum()
    assert caption(4831)[..., 0].sum() > 0
    np.testing.assert_array_equal(caption(4831), caption(4959))
    assert O.T1.IL.LINES == original


def test_camera_contains_full_mountain_falling_ring_and_caption():
    scene = O.OldFire()
    line = scene.caption()
    mountain = O.PG.Mountain(11)
    bounds = [mountain.box, (line.c0 / line.ppc, line.r0 / line.ppc,
                           (line.c0 + line.a.shape[1]) / line.ppc,
                           (line.r0 + line.a.shape[0]) / line.ppc)]
    ring_corners = O.mapped_point(np.asarray(O.RP.RingPage.WIN).reshape(2, 2))
    bounds.append((*ring_corners[0], *ring_corners[1]))
    for t in (0., 159 / 24.):
        book, camera = scene.layout(t)
        for x0, y0, x1, y1 in bounds:
            xs, ys = np.meshgrid(np.linspace(x0, x1, 9), np.linspace(y0, y1, 9))
            points = book.page_to_world('R', xs.ravel(), ys.ravel())
            pixels, depth = camera.project(points)
            assert (depth > 0).all()
            assert (pixels[:, 0] > 24).all() and (pixels[:, 0] < camera.W - 24).all()
            assert (pixels[:, 1] > 24).all() and (pixels[:, 1] < camera.H - 24).all()
    # Negative control: copying the accepted close-up would clip the falling
    # Ring's upper window. C's camera is valid for its own shot, not this one.
    original = O.BC.Book3().cam_mountain(book, 5.)
    pixels, _ = original.project(book.page_to_world('R', ring_corners[:, 0], ring_corners[:, 1]))
    assert pixels[:, 1].min() < 24


def test_patch_material_covers_old_ink_but_preserves_uncovered_illustration():
    receiver = np.zeros((2, 3, 7), np.float32)
    receiver[..., 0] = .8
    patch = np.zeros_like(receiver)
    patch[0, 0, 2] = .9
    patch[0, 1, 0] = 1.
    patch[1, 0, 4] = .5
    naive_max = np.maximum(receiver, patch)
    O.overlay_patch(receiver, patch)
    assert receiver[0, 0, 0] == 0. and receiver[1, 0, 0] == 0.
    assert naive_max[0, 0, 0] > 0. and naive_max[1, 0, 0] > 0.
    np.testing.assert_array_equal(receiver[0, 0], patch[0, 0])
    assert receiver[0, 1, 0] == 1. and receiver[1, 2, 0] == np.float32(.8)


def test_texture_is_frame_deterministic_and_retains_mountain_outside_patch():
    scene = O.OldFire(96, 40, ppc=20)
    before = scene.texture(4800).chan.copy()
    during = scene.texture(4872).chan.copy()
    after = scene.texture(4959).chan.copy()
    np.testing.assert_array_equal(before, scene.texture(4800).chan)
    assert not np.array_equal(before, during)
    assert not np.array_equal(during, after)
    _, base = scene.mountain_page()
    # A rectangle covering the drawn cone, away from the fire and caption.
    cone = np.s_[int(6*scene.ppc):int(12.2*scene.ppc),
                 int(2.3*scene.ppc):int(17.3*scene.ppc)]
    assert base[cone][..., 0].sum() > 0.
    np.testing.assert_array_equal(before[cone], base[cone])
    np.testing.assert_array_equal(after[cone], base[cone])


def test_late_bead_contour_preserves_state_and_earlier_textures():
    # Compare actual textures, including all earlier flames, against the
    # unchanged source renderer. An early fire reduction fails this equality.
    original = O.RP.RingPage(ppc=18.7)
    mountain = O.MountainRing(ppc=18.7)
    for t in (.2, .4, 2.65, 4.8):
        assert O.foreground_growth(t) == 1.
        np.testing.assert_array_equal(mountain.texture(t).chan, original.texture(t).chan)
        for actual, expected in zip(mountain.state(t), original.state(t)):
            np.testing.assert_array_equal(actual, expected)
    t = O.unmaking_time(159 / 24.)
    state = original.state(t)[0]
    assert state[11] >= .85
    class NoContour(O.MountainRing):
        def _bead_contour(self, win, ox, oy, t):
            pass
    baseline = NoContour(ppc=18.7).texture(t).chan.copy()
    exposed = mountain.texture(t).chan.copy()
    cx, cy = state[0], state[1] + .5 * state[4]
    rx, ry = state[2], state[3] * state[2] + .5 * state[4]
    yy, xx = np.mgrid[:exposed.shape[0], :exposed.shape[1]]
    distance = np.sqrt((((xx + .5) / mountain.ppc - cx) / rx)**2 +
                       (((yy + .5) / mountain.ppc - cy) / ry)**2)
    rim = np.abs(distance - 1.) < .06
    assert rim.any()
    def require_drawn_edge(candidate):
        # A minimum material-contrast change at the actual ellipse perimeter;
        # native image inspection remains the human-legibility gate.
        assert (candidate[..., 0][rim] - baseline[..., 0][rim]).mean() > .15
        assert candidate[..., 4][rim].mean() < baseline[..., 4][rim].mean()
    require_drawn_edge(exposed)
    with pytest.raises(AssertionError):
        require_drawn_edge(baseline)
    # This is an edge, not a dark plate over the fire or an enlarged bead.
    away = (distance < .55) | (distance > 1.35)
    np.testing.assert_array_equal(exposed[away], baseline[away])
    for actual, expected in zip(mountain.state(t), original.state(t)):
        np.testing.assert_array_equal(actual, expected)
