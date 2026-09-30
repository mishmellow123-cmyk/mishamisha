"""Candidate isolation and A14/A15 continuity, without shot or terrain renders.

The camera functions run from their live source in real module dictionaries,
so scoped binding changes affect their globals exactly as in production.
Only terrain heights and the picture passes are substituted. Run with
NUMBA_DISABLE_JIT=1 while the single picture-render process owns the machine.
"""
import json
import math
import os
import sys
import types
from unittest import mock

import numpy as np
import pytest

from test_night_hearth import RUN, functions
import beaconrun_candidates as candidate
import rcam as RC


def shot_modules(monkeypatch):
    """Extract actual shot cameras without importing their terrain globals."""
    B, W = types.ModuleType('beaconrun_a'), types.ModuleType('watchers_a')
    accepted = np.load(RUN / 'beaconrun_a_chain.npy')
    night = functions('nighta.py', (), ('GLOW_AZ',), {})
    B.__dict__.update(np=np, math=math, os=os, RC=RC, CHAIN=accepted, FIRES=accepted)
    functions('beaconrun_a.py', ('end_cam', '_ease', '_cam_env', 'camera'),
              ('FR0', 'T_SETTLE', 'BACK0', 'YAW0', '_PLATE', 'END_CAM'), B.__dict__)
    W.__dict__.update(np=np, math=math, os=os, RC=RC,
                     NA=types.SimpleNamespace(**night),
                     ground=lambda x, z: .004 * x - .006 * z)
    functions('watchers_a.py', ('br', '_dir', '_plate_axes', 'seventh_spot',
              'chain', 'seventh_fire', 'plate', 'default_cam', 'camera', 'hearth_lighting'),
              ('FR0', 'UP', 'LOOK_AZ', 'PLATE_YAW', 'WATCHER_BEARING', 'FIRE_GAP',
               'EYE', 'BACK', 'HFOV', 'PITCH', 'PUSH'), W.__dict__)
    B.NA = types.SimpleNamespace(fires_layer=mock.Mock(),
                                _SMOKE={901: object()}, _SPARKS={905: object()})
    B.WD = types.SimpleNamespace(shade=mock.Mock())
    B.draw_flares = mock.Mock()
    B.FINISH = {'exposure': .73}
    B.PI = types.SimpleNamespace(look=types.SimpleNamespace(finish=mock.Mock(side_effect=lambda x, **kw: x)))
    W.finish = mock.Mock(side_effect=lambda x: x)
    for key in ('A14_CAM', 'A14_LABELS', 'A14_NOUG', 'NIGHT_CLOUD',
                'W15_P7', 'W15_CAM', 'W15_OWN_CAM'):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setitem(sys.modules, 'beaconrun_a', B)
    monkeypatch.setitem(sys.modules, 'watchers_a', W)
    monkeypatch.setattr(candidate, 'modules', lambda: (B, W))
    return B, W


@pytest.mark.parametrize('kind,frame', [('beaconrun', 3960), ('watchers', 4240)])
def test_default_delegates_accepted_bytes_without_entering_candidate(monkeypatch, kind, frame):
    B, W = shot_modules(monkeypatch)
    # Distinctive bytes expose an accidental scale, grade, copy or extra pass.
    accepted = np.arange(36, dtype=np.uint8).reshape(3, 4, 3)
    original = accepted.copy()
    B.render = mock.Mock(return_value=(accepted, object(), object()))
    W.render = mock.Mock(return_value=accepted)
    enter = mock.Mock(side_effect=AssertionError('default path entered candidate'))
    monkeypatch.setattr(candidate, 'linked_scene', enter)

    result = candidate.render(frame, kind=kind, scale=.5, ss=1.25)

    assert result is accepted
    np.testing.assert_array_equal(result, original)
    enter.assert_not_called()
    selected, unused = (B.render, W.render) if kind == 'beaconrun' else (W.render, B.render)
    selected.assert_called_once_with(frame, .5, 1.25)
    unused.assert_not_called()
    if kind == 'beaconrun':
        B.PI.look.finish.assert_called_once_with(accepted, **B.FINISH)
        W.finish.assert_not_called()
    else:
        W.finish.assert_called_once_with(accepted)
        B.PI.look.finish.assert_not_called()


@pytest.mark.parametrize('kind,frame', [('beaconrun', 4239), ('watchers', 4240)])
@pytest.mark.parametrize('cached', [False, True])
def test_failed_render_restores_shared_bindings_and_cached_plate(monkeypatch, kind, frame, cached):
    B, W = shot_modules(monkeypatch)
    if cached:
        B.end_cam()
    entered = []

    def fail(*args):
        entered.append(True)
        # A false-clean rollback test must not pass because activation was skipped.
        assert B.CHAIN is not accepted
        assert B.FIRES is B.CHAIN
        assert B.camera is not bindings[0]['camera']
        assert W.camera is not bindings[1]['camera']
        B.NA._SMOKE[999] = object()
        B.NA._SPARKS[999] = object()
        raise RuntimeError('intentional picture failure')

    B.render = W.render = fail
    objects = (B, W, B.NA, B.WD)
    bindings = [vars(obj).copy() for obj in objects]
    cache_contents = (B.NA._SMOKE.copy(), B.NA._SPARKS.copy())
    accepted = B.CHAIN
    accepted_bytes = accepted.tobytes()
    with pytest.raises(RuntimeError, match='intentional picture failure'):
        candidate.render(frame, kind=kind, candidate=candidate.OPTION, scale=.5)
    assert entered == [True]
    for obj, before in zip(objects, bindings):
        assert set(vars(obj)) == set(before)
        for key, value in before.items():
            assert getattr(obj, key) is value, key
    assert accepted.tobytes() == accepted_bytes
    assert B.NA._SMOKE == cache_contents[0]
    assert B.NA._SPARKS == cache_contents[1]


@pytest.mark.parametrize('width,height', [(960, 402), (1920, 804)])
def test_candidate_camera_and_projected_anchors_join_at_4239_4240(monkeypatch, width, height):
    B, W = shot_modules(monkeypatch)
    initial = (B.camera, W.camera, B.CHAIN, B._PLATE)
    with candidate.linked_scene() as chain:
        last, first = B.camera(4239, width, height), W.camera(4240, width, height)
        for key in ('pos', 'fwd', 'right', 'up', 'f', 'cx', 'cy'):
            np.testing.assert_allclose(getattr(last, key), getattr(first, key), rtol=0., atol=1e-12)
        assert (last.W, last.H) == (first.W, first.H) == (width, height)
        watcher = W.seventh_spot(chain[6])[0]
        points = np.vstack([chain[:, :3], watcher, watcher + [0., 1.75, 0.]])
        np.testing.assert_allclose(last.project(points), first.project(points), rtol=0., atol=1e-9)
    for actual, original in zip((B.camera, W.camera, B.CHAIN, B._PLATE), initial):
        assert actual is original


def test_chain_copies_input_keeps_hearth_and_lands_on_score():
    accepted = np.load(RUN / 'beaconrun_a_chain.npy')
    before = accepted.copy()
    chain = candidate.candidate_chain(accepted)
    score = json.loads((RUN.parents[1] / 'music/v3/barmap_A.json').read_text())
    sync = {point['id']: point['f'] for point in score['sync']}
    expected = np.array([sync[f'run_{k}'] for k in range(1, 8)])
    assert not np.shares_memory(chain, accepted)
    np.testing.assert_array_equal(accepted, before)
    np.testing.assert_array_equal(chain[6, :3], before[6, :3])
    np.testing.assert_array_equal(chain[:, 5], before[:, 5])
    assert 3920 <= chain[0, 3] <= 3944
    np.testing.assert_array_equal(chain[1:, 3], expected[1:])
    # The early first flame still receives its flare on the first score accent.
    np.testing.assert_array_equal(candidate.SYNC, expected)


def test_pool_falls_smoothly_without_a_visible_hard_cutoff():
    # World metres across the near hearth and surrounding ledge. Fine paired
    # samples catch a cutoff even when its endpoint misses the coarse grid.
    distances = np.linspace(0., 12., 12001)
    weights = candidate.pool_weight(distances)
    assert weights.shape == distances.shape
    assert np.isfinite(weights).all()
    assert weights[0] == pytest.approx(1.)
    assert np.all((weights > 0.) & (weights <= 1.))
    assert np.all(np.diff(weights) <= 0.)
    assert weights[-1] < .001
    close = candidate.pool_weight(distances + 1e-6)
    assert np.max(np.abs(close - weights)) < 1e-5
    # A step between samples still makes one adjacent difference conspicuous.
    assert np.max(np.abs(np.diff(weights))) < .005


@pytest.mark.parametrize('include_hearth', [False, True])
def test_patched_shader_tapers_only_hearth_light_at_actual_world_hits(monkeypatch, include_hearth):
    B, W = shot_modules(monkeypatch)
    p7 = B.CHAIN[6, :3].copy()
    # Off-axis source pixels exercise all three world coordinates; the oracle
    # uses camera basis vectors rather than the shader's packed-parameter math.
    cam = RC.SrcCam(p7 + [0., 1., -10.], 0., 5., -2.5, 2.5, -1.5, 1.5)
    j, i = np.mgrid[:cam.H, :cam.W]
    rays = (cam.fwd + cam.right * ((i + .5 - cam.cx) / cam.f)[..., None]
            + np.array([0., 1., 0.]) * ((cam.cyy - j - .5) / cam.f)[..., None])
    hits = cam.pos + 10. * rays
    forward_depth = np.full((cam.H, cam.W), 10.)
    ray_distance = np.linalg.norm(hits - cam.pos, axis=-1)
    world_distance = np.linalg.norm(hits - p7, axis=-1)
    moon = np.arange(cam.H * cam.W * 3).reshape(cam.H, cam.W, 3) / 1000. + .1
    remote_rgb, hearth_rgb = np.array([.04, .07, .03]), np.array([.8, .2, .01])
    lights = np.array([[*p7, 1., .4, .1, 2., .3],
                       [p7[0] + 50., p7[1], p7[2], .2, .4, 1., 3., .3]])
    if not include_hearth:
        lights = lights[1:].copy()
    before = lights.copy()
    calls = []

    def shade(C, D, P, CR, S, LT, Lk, Q, amb, fogp, out, zb, dist, PL):
        calls.append(LT.copy())
        near = np.isclose(LT[:, 0], p7[0])
        out[:] = moon + remote_rgb + (hearth_rgb if near.any() else 0.)
        zb[:] = forward_depth
        dist[:] = ray_distance

    B.WD.shade = shade
    weight = mock.Mock(wraps=candidate.pool_weight)
    monkeypatch.setattr(candidate, 'pool_weight', weight)
    out = np.full((cam.H, cam.W, 3), np.nan)
    zb, dist = np.empty_like(forward_depth), np.empty_like(ray_distance)
    with candidate.linked_scene():
        B.WD.shade(cam.params(), None, None, None, None, lights, None, None,
                   None, None, out, zb, dist, np.zeros((0, 4)))
    expected = moon + remote_rgb
    if include_hearth:
        expected = expected + hearth_rgb * weight._mock_wraps(world_distance)[..., None]
        assert len(calls) == 2
        np.testing.assert_array_equal(calls[0], before)
        np.testing.assert_array_equal(calls[1], before[1:])
        weight.assert_called_once()
        np.testing.assert_allclose(weight.call_args.args[0], world_distance, rtol=0., atol=1e-12)
    else:
        assert len(calls) == 1
        weight.assert_not_called()
    np.testing.assert_allclose(out, expected, rtol=0., atol=1e-12)
    np.testing.assert_array_equal(zb, forward_depth)
    np.testing.assert_array_equal(dist, ray_distance)
    np.testing.assert_array_equal(lights, before)


def test_patched_fire_layer_sorts_custom_and_accepted_emitters_together(monkeypatch):
    B, W = shot_modules(monkeypatch)
    import fire2 as F2

    draws, accepted_calls = [], []
    img, zb = object(), object()
    fog = np.zeros(8)
    offsets = {906: .12}
    camera = types.SimpleNamespace(pos=np.zeros(3), f=10.,
                                   project=lambda p: (4., 4., float(p[2])))

    def accepted(image, depth, cam, rows, f, pxs, **kw):
        assert image is img and depth is zb and cam is camera
        assert f == 4240 and pxs == .5
        assert kw['fogp'] is fog and kw['base_offsets'] is offsets
        assert kw['near_hearth'] is True
        accepted_calls.append(rows.copy())
        draws.extend(('accepted', int(row[5])) for row in rows)

    def custom(image, depth, cam, base, height, radius, t, **kw):
        assert image is img and depth is zb and cam is camera
        draws.append(('custom', (kw['seed'] - 3) // 7))

    B.NA.fires_layer = accepted
    B.NA.fire_trans = lambda *a: 1.
    B.NA.fire_depth_bias = lambda *a: .1
    monkeypatch.setattr(F2, 'flame', custom)
    monkeypatch.setattr(F2, 'halo', mock.Mock())
    with candidate.linked_scene() as chain:
        rows = chain[[6, 0, 2, 6, 6]].copy()
        # Accepted entries interleave the custom pair. Group-wise sorting in
        # either order must fail even if each group is internally sorted.
        rows[:, :3] = [[0., 0., 7.], [0., 0., 31.], [0., 0., 19.],
                       [0., 0., 3.], [0., 0., 25.]]
        rows[3:, 5] = [777., 778.]
        before = rows.copy()
        B.NA.fires_layer(img, zb, camera, rows, 4240, .5,
                         fogp=fog, near_hearth=True, base_offsets=offsets)
    assert draws == [('custom', 900), ('accepted', 778), ('custom', 902),
                     ('accepted', 906), ('accepted', 777)]
    np.testing.assert_array_equal(np.vstack(accepted_calls), before[[4, 0, 3]])
    np.testing.assert_array_equal(rows, before)


@pytest.mark.parametrize('frame', [4239, 4240])
def test_seventh_scene_consumes_relay_heading_across_the_join(monkeypatch, frame):
    B, W = shot_modules(monkeypatch)
    from mt.noise import smoothstep

    W.smoothstep = smoothstep
    W._figure = mock.Mock()
    functions('watchers_a.py', ('lighter_pose', 'seventh_scene'), (), W.__dict__)
    accepted_heading = W.LOOK_AZ
    scene = object()
    with candidate.linked_scene() as chain:
        W.seventh_scene(scene, frame, chain[6])
        W._figure.assert_called_once()
        call = W._figure.call_args
        assert call.args[0] is scene
        assert call.args[3] == frame
        # Inspect the vectors actually passed into figure construction. A
        # changed LOOK_AZ that the scene never consumes cannot pass this check.
        toward_relay = chain[0, :3] - call.args[1]
        toward_relay[1] = 0.
        toward_relay /= np.linalg.norm(toward_relay)
        for direction in (call.args[2], call.kwargs['stance_w']):
            assert np.linalg.norm(direction) == pytest.approx(1.)
            assert direction[1] == pytest.approx(0.)
            assert float(np.dot(direction, toward_relay)) > .999
    assert W.LOOK_AZ == accepted_heading
