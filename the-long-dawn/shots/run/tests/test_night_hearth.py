"""Small night/hearth contracts; no world initialization or full shot renders.

Run with NUMBA_DISABLE_JIT=1 and one thread while a render owns the machine.
Functions are extracted from the live source to avoid importing shot globals;
fire2 and sdfppl are real, including their pixel/primitive implementations.
"""
import ast
import math
from pathlib import Path
import sys
import types
import unittest
from unittest import mock

import cv2
import numpy as np

cv2.setNumThreads(0)
RUN = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RUN))
sys.path.insert(0, str(RUN.parent / 'montage'))
import fire2 as F2
import fire_near_a as FN
import sdfppl as SP
from mt import fire as F
from mt.noise import smoothstep


def functions(filename, names, constants, scope):
    """Compile actual function bodies and their selected source constants."""
    path = RUN / filename
    tree = ast.parse(path.read_text())
    nodes = []
    found = set()
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            nodes.append(node)
            found.add(node.name)
        elif isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in constants
                                                 for target in node.targets for t in ast.walk(target)):
            nodes.append(node)
    if found != set(names):
        raise AssertionError(f'{filename}: missing functions {set(names) - found}')
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), scope)
    return scope


def night_scope(fire2=F2, near_fire=FN):
    scope = dict(np=np, math=math, F=F, F2=fire2, FN=near_fire, WD=object())
    # A distinctive synthetic sky recipe makes accidental unrelated changes visible.
    scope['_fd_glow_params'] = lambda design, intensity: np.arange(27, dtype=np.float64) + intensity
    return functions('nighta.py', ('glow_I', 'glow_gp', '_ign', 'fire_dims', 'fire_depth_bias',
                                  'fire_layer', 'fires_layer'),
                     ('FPS', 'BAR', 'I0', 'GLOW_AZ', 'EL_S', 'POOL_I'), scope)


class Camera:
    def __init__(self, depth=16., width=32, height=32, focal=160., position=(0., 1., 0.)):
        self.pos = np.array(position, dtype=np.float64)
        self.f = focal
        self.W, self.H = width, height
        self.depth = depth

    def project(self, point):
        x, y, z = np.asarray(point) - self.pos
        return self.W / 2 + self.f * x / z, self.H / 2 - self.f * y / z, z

    def params(self):
        return np.array([*self.pos, 0., 1., 1., 0., self.f, self.W / 2, self.H / 2, self.W, self.H])


class NightHearthTests(unittest.TestCase):
    def test_depth_opt_in_preserves_foreground_occlusion_and_far_policy(self):
        bias = night_scope()['fire_depth_bias']
        # A half-metre-separated cowl/stone must occlude a close flame; the old
        # tolerance deliberately passes it. No particular new epsilon is pinned.
        self.assertLess(bias(16., True), .5)
        self.assertGreater(bias(16.), .5)
        for z in (1., 16., 59., 60., 400., 2000., 18000.):
            self.assertEqual(bias(z), max(3., .004 * z))
        for z in (60., 400., 2000., 18000.):
            self.assertEqual(bias(z, True), bias(z))

    def test_actual_resolved_flame_is_blocked_by_near_foreground(self):
        scope = night_scope()
        camera = Camera()
        initial = np.full((32, 32, 3), .02, np.float32)

        def draw(depth, **kwargs):
            image = initial.copy()
            zb = np.full((32, 32), depth, np.float32)
            # Isolate the real resolved flame from broad glow; no smoke simulation.
            with mock.patch.object(F2, 'halo'), mock.patch.object(F2, 'glow'), \
                    mock.patch.object(F2, '_bonfire', wraps=F2._bonfire) as resolved:
                scope['fire_layer'](image, zb, camera, (0., 0., 16.), 0., 24.,
                                    size=.6, smoke=False, **kwargs)
                resolved.assert_called_once()
            np.testing.assert_array_equal(zb, np.full_like(zb, depth))
            return image

        visible = draw(1e9, near_hearth=True)
        old_tolerance = draw(15.5)
        hidden = draw(15.5, near_hearth=True)
        self.assertGreater(float(np.max(visible - initial)), 0.)
        self.assertGreater(float(np.max(old_tolerance - initial)), 0.)
        np.testing.assert_array_equal(hidden, initial)

    def capture_fire(self, depth=16., **kwargs):
        emitters = types.SimpleNamespace(flame=mock.Mock(), halo=mock.Mock(), glow=mock.Mock(),
                                        AURA_COL=F2.AURA_COL)
        scope = night_scope(emitters, near_fire=types.SimpleNamespace(draw=mock.Mock()))
        smoke = mock.Mock()
        scope['_smoke'] = mock.Mock(return_value=smoke)
        scope['fire_layer'](np.zeros((32, 32, 3), np.float32), np.full((32, 32), 1e9),
                            Camera(depth), (0., 0., depth), 0., 24., size=.6, **kwargs)
        return scope, emitters, smoke

    def test_volume_dispatch_requires_explicit_hearth_identity_at_every_distance(self):
        # Distance must not switch renderers as the A14 camera crosses 60 m.
        # A zero fuel-seat offset is a real explicit override, not missing data.
        for depth in (16., 59.9, 60., 60.1, 200., 1000.):
            for near in (False, True):
                for offset in (None, 0., .12):
                    with self.subTest(depth=depth, near=near, offset=offset):
                        scope, emitters, _ = self.capture_fire(depth, smoke=False,
                                                              near_hearth=near, base_offset=offset)
                        volume = scope['FN'].draw
                        enabled = near and offset is not None
                        self.assertEqual(volume.call_count, int(enabled))
                        self.assertEqual(emitters.flame.call_count, int(not enabled))
                        if enabled:
                            origin = volume.call_args.args[3]
                            np.testing.assert_array_equal(origin[[0, 2]], [0., depth])
                            # The measured fuel seat remains the caller's anchor;
                            # a shallow inset lets opaque cinders clip the roots.
                            self.assertGreater(offset - origin[1], 0.)
                            self.assertLessEqual(offset - origin[1], .10)

    def test_volume_origin_translates_with_explicit_fuel_seat(self):
        ground, _, _ = self.capture_fire(smoke=False, near_hearth=True, base_offset=0.)
        raised, _, _ = self.capture_fire(smoke=False, near_hearth=True, base_offset=.12)
        delta = raised['FN'].draw.call_args.args[3] - ground['FN'].draw.call_args.args[3]
        np.testing.assert_allclose(delta, [0., .12, 0.], rtol=0., atol=1e-12)

    def test_volume_receives_atmosphere_once_without_pre_attenuated_emission(self):
        a, _, _ = self.capture_fire(smoke=False, near_hearth=True, base_offset=.12, trans=1.)
        b, _, _ = self.capture_fire(smoke=False, near_hearth=True, base_offset=.12, trans=.25)
        full, attenuated = a['FN'].draw.call_args, b['FN'].draw.call_args
        self.assertGreater(full.kwargs['I'], 0.)
        self.assertEqual(full.kwargs['I'], attenuated.kwargs['I'])
        self.assertEqual(full.kwargs['trans'], 1.)
        self.assertEqual(attenuated.kwargs['trans'], .25)

    def test_volume_roi_owns_culling_when_fuel_base_is_outside_frame(self):
        emitters = types.SimpleNamespace(flame=mock.Mock(), halo=mock.Mock(), glow=mock.Mock(),
                                        AURA_COL=F2.AURA_COL)
        volume = types.SimpleNamespace(draw=mock.Mock())
        scope = night_scope(emitters, volume)
        camera = Camera(focal=400.)
        # A close, tightly cropped view sees the upper flame although the base
        # falls beyond the legacy billboard's 300-pixel centre guard.
        self.assertGreater(camera.project([0., .12, 1.1])[1], camera.H + 300)
        scope['fire_layer'](np.zeros((32, 32, 3), np.float32), np.full((32, 32), 1e9),
                            camera, (0., 0., 1.1), 0., 24., size=.6, smoke=False,
                            near_hearth=True, base_offset=.12)
        volume.draw.assert_called_once()
        emitters.flame.assert_not_called()

    def test_actual_opted_volume_obeys_foreground_and_single_transmission(self):
        scope = night_scope()
        camera = Camera(width=7, height=7, focal=35., position=(0., .58, -4.))

        def draw(depth=1e9, trans=1.):
            image = np.zeros((7, 7, 3), np.float32)
            zb = np.full((7, 7), depth, np.float32)
            # Isolate the volume from its separate aura and smoke. This exercises
            # the real nighta dispatch and real ray integral, with no aesthetic test.
            with mock.patch.object(F2, 'halo'), mock.patch.object(F2, 'glow'), \
                    mock.patch.object(F2, 'flame') as billboard, mock.patch.object(FN, 'draw', wraps=FN.draw) as volume:
                scope['fire_layer'](image, zb, camera, (0., 0., 0.), 0., 24., size=.6, seed=314,
                                    smoke=False, near_hearth=True, base_offset=.12, trans=trans)
                volume.assert_called_once()
                billboard.assert_not_called()
            np.testing.assert_array_equal(zb, np.full_like(zb, depth))
            return image

        full = draw()
        self.assertTrue(np.isfinite(full).all())
        self.assertGreater(float(full.max()), 0.)
        np.testing.assert_array_equal(draw(depth=2.), np.zeros_like(full))
        np.testing.assert_allclose(draw(trans=.25), full * .25, rtol=1e-6, atol=1e-7)

    def test_default_and_explicit_off_keep_all_emitter_arguments(self):
        for depth in (16., 1000.):
            _, baseline, smoke0 = self.capture_fire(depth)
            _, explicit, smoke1 = self.capture_fire(depth, near_hearth=False, base_offset=None)
            for name in ('flame', 'halo', 'glow'):
                a, b = getattr(baseline, name).call_args_list, getattr(explicit, name).call_args_list
                self.assertEqual(len(a), len(b))
                for ca, cb in zip(a, b):
                    for aa, bb in zip(ca.args[3:], cb.args[3:]):
                        np.testing.assert_array_equal(aa, bb)
                    for key in ca.kwargs:
                        np.testing.assert_array_equal(ca.kwargs[key], cb.kwargs[key])
            self.assertEqual(smoke0.render.call_args.kwargs, smoke1.render.call_args.kwargs)

    def test_opt_in_smoke_and_flame_use_same_close_depth_policy(self):
        scope, emitters, smoke = self.capture_fire(16., near_hearth=True)
        expected = scope['fire_depth_bias'](16., True)
        self.assertEqual(emitters.flame.call_args.kwargs['zbias'], expected)
        self.assertEqual(smoke.render.call_args.kwargs['zbias'], expected)
        self.assertTrue(all(c.kwargs['zbias'] == expected for c in emitters.halo.call_args_list))
        _, _, original_smoke = self.capture_fire(16.)
        # The unchanged A13/default smoke recipe is an explicit compatibility
        # contract; the brighter local-hearth ambient must not become its default.
        self.assertEqual(original_smoke.render.call_args.kwargs['amb'], (.004, .005, .009))
        self.assertTrue(np.all(np.array(smoke.render.call_args.kwargs['amb']) >
                               original_smoke.render.call_args.kwargs['amb']))
        _, distant, distant_smoke = self.capture_fire(1000., near_hearth=True)
        _, original, original_distant_smoke = self.capture_fire(1000.)
        self.assertEqual(distant.flame.call_args.kwargs, original.flame.call_args.kwargs)
        self.assertEqual(distant_smoke.render.call_args.kwargs, original_distant_smoke.render.call_args.kwargs)

    def test_base_offset_moves_origin_without_resizing_flame(self):
        scope, original, _ = self.capture_fire(smoke=False)
        _, lowered, _ = self.capture_fire(smoke=False, base_offset=.12)
        _, at_ground, _ = self.capture_fire(smoke=False, base_offset=0.)
        base = original.flame.call_args.args[3]
        self.assertAlmostEqual(base[1], .45 * scope['fire_dims'](.6)[0])
        np.testing.assert_array_equal(lowered.flame.call_args.args[3], [0., .12, 16.])
        np.testing.assert_array_equal(at_ground.flame.call_args.args[3], [0., 0., 16.])
        self.assertEqual(original.flame.call_args.args[4:6], lowered.flame.call_args.args[4:6])

    def test_offsets_follow_seed_after_depth_sort_and_default_is_opt_out(self):
        scope = night_scope()
        calls = mock.Mock()
        scope['fire_layer'] = calls
        fires = np.array([[0., 0., 16., 0., .6, 906.], [0., 0., 200., 0., 1., 900.]])
        before = fires.copy()
        scope['fires_layer'](None, None, Camera(), fires, 24., near_hearth=True, base_offsets={906: .12})
        self.assertEqual([c.args[7] for c in calls.call_args_list], [900., 906.])
        self.assertEqual([c.kwargs['base_offset'] for c in calls.call_args_list], [None, .12])
        self.assertTrue(all(c.kwargs['near_hearth'] for c in calls.call_args_list))
        np.testing.assert_array_equal(fires, before)
        calls.reset_mock()
        scope['fires_layer'](None, None, Camera(), fires, 24.)
        self.assertTrue(all(c.kwargs == {'near_hearth': False, 'base_offset': None}
                            for c in calls.call_args_list))

    def test_shadow_ray_option_changes_only_shadow_component(self):
        glow = night_scope()['glow_gp']
        for frame in (3920., 4200., 4240.):
            original, on, off = glow(frame), glow(frame, shadow_rays=True), glow(frame, shadow_rays=False)
            np.testing.assert_array_equal(original, on)
            self.assertGreater(original[11], 0.)
            self.assertEqual(off[11], 0.)
            np.testing.assert_array_equal(np.delete(off, 11), np.delete(original, 11))
            np.testing.assert_array_equal(glow(frame), original)

    def test_hearth_lighting_changes_only_seventh_pool_and_platform_without_mutation(self):
        apply = functions('watchers_a.py', ('hearth_lighting',), (), dict(np=np))['hearth_lighting']
        seventh = np.array([100., 4., -50., 4200., .6, 906.])
        # Permuted rows and different light/platform heights: identity is world
        # x/z, not row position or emitter y (the light sits above the ground).
        lights = np.array([[104., 4.5, -50., 1., .4, .1, 9., .3],
                           [100., 4.8, -50., 1., .4, .1, 8., .3],
                           [100., 4.8, -54., 1., .4, .1, 7., .3]])
        platforms = np.array([[100., 4., -54., 3.], [104., 4., -50., 3.], [100., 4., -50., 3.]])
        originals = [a.copy() for a in (lights, platforms, seventh)]
        result_lights, result_platforms = apply(lights, platforms, seventh)
        for actual, original in zip((lights, platforms, seventh), originals):
            np.testing.assert_array_equal(actual, original)
        self.assertFalse(np.shares_memory(result_lights, lights))
        self.assertFalse(np.shares_memory(result_platforms, platforms))
        self.assertGreater(result_lights[1, 6], 0.)
        self.assertLess(result_lights[1, 6], lights[1, 6])
        self.assertGreater(result_platforms[2, 3], 0.)
        self.assertLess(result_platforms[2, 3], platforms[2, 3])
        result_lights[1, 6] = lights[1, 6]
        result_platforms[2, 3] = platforms[2, 3]
        np.testing.assert_array_equal(result_lights, lights)
        np.testing.assert_array_equal(result_platforms, platforms)

    def test_hearth_lighting_accepts_unlit_and_unmatched_tables(self):
        apply = functions('watchers_a.py', ('hearth_lighting',), (), dict(np=np))['hearth_lighting']
        seventh = np.array([100., 4., -50., 4200., .6, 906.])
        for lights, platforms in ((np.zeros((0, 8)), np.zeros((0, 4))),
                                  (np.zeros((0, 8)), np.array([[100., 4., -50., 3.]])),
                                  (np.array([[2., 3., 4., 1., .4, .1, 7., .3]]),
                                   np.array([[2., 3., 4., 3.]]))):
            l, p = apply(lights, platforms, seventh)
            self.assertEqual(l.shape, lights.shape)
            self.assertEqual(p.shape, platforms.shape)
            np.testing.assert_array_equal(l, lights)
            if len(lights):
                np.testing.assert_array_equal(p, platforms)

    def test_actual_watchers_render_selects_only_seventh_base_and_shadow_opt_out(self):
        """Execute the actual orchestration with one-pixel synthetic pass outputs."""
        import mt
        cam = types.SimpleNamespace(pos=np.zeros(3), W=1, H=1, params=lambda: np.zeros(12))
        frame = types.SimpleNamespace(src=cam)
        fires = np.array([[float(k), 0., 10., 4200., .6, 900. + k] for k in range(7)])
        light = (np.ones(6), np.ones(3), None, np.zeros(8), np.zeros(24))
        br = types.SimpleNamespace(CR=np.zeros((0, 16)), fires_table=lambda: fires.copy(),
                                   light=lambda f: light, stars=lambda: None)
        wd = types.SimpleNamespace(march=mock.Mock(), shade=mock.Mock(), cloud_glow=mock.Mock())
        lights = np.array([[6., .8, 10., 1., .4, .1, 8., .3], [0., .8, 10., 1., .4, .1, 7., .3]])
        platforms = np.array([[0., 0., 10., 3.], [6., 0., 10., 3.]])
        originals = lights.copy(), platforms.copy()
        na = types.SimpleNamespace(lt_rows=lambda *a: lights, pl_rows=lambda *a, **k: platforms,
            ug_rows=lambda f: None, skyline=lambda *a: (0., 0., None), glow_pass=mock.Mock(),
            glow_gp=mock.Mock(return_value=np.zeros(27)), HAZE_K=0., HAZE_D=1., fires_layer=mock.Mock())
        hearth = mock.Mock(return_value=types.SimpleNamespace(fire_base_offset=.123))
        scope = functions('watchers_a.py', ('render', 'hearth_lighting'), ('FPS',),
            dict(np=np, br=lambda: br, camera=lambda *a: cam, chain=lambda: fires.copy(),
                 extra_fires=lambda: np.zeros((0, 6)), WD=wd, NA=na,
                 FD=types.SimpleNamespace(terrain_hmax=lambda cr: 3000.),
                 PI=types.SimpleNamespace(Frame=lambda *a: frame, src_scale=lambda fr: 1.,
                                          to_target=lambda fr: (fr.img, fr.zb, fr.dist)),
                 draw_figures=mock.Mock(), seventh_hearth=hearth))
        sky = types.SimpleNamespace(splat_stars=mock.Mock())
        with mock.patch.object(mt, 'sky', sky, create=True), mock.patch.dict(sys.modules, {'mt.sky': sky}):
            image = scope['render'](4240., scale=.01)
        self.assertEqual(image.shape, (1, 1, 3))
        na.glow_gp.assert_called_once_with(4240., shadow_rays=False)
        self.assertTrue(na.fires_layer.call_args.kwargs['near_hearth'])
        self.assertEqual(na.fires_layer.call_args.kwargs['base_offsets'], {906: .123})
        np.testing.assert_array_equal(hearth.call_args.args[0], fires[6])
        # The actual render must pass the local treatment into both terrain and
        # foreground shading; testing the helper alone cannot catch a missed call.
        rendered_lights, rendered_platforms = wd.shade.call_args.args[5], wd.shade.call_args.args[13]
        self.assertLess(rendered_lights[0, 6], lights[0, 6])
        self.assertLess(rendered_platforms[1, 3], platforms[1, 3])
        np.testing.assert_array_equal(rendered_lights[1], lights[1])
        np.testing.assert_array_equal(rendered_platforms[0], platforms[0])
        np.testing.assert_array_equal(scope['draw_figures'].call_args.args[4], rendered_lights)
        np.testing.assert_array_equal(lights, originals[0])
        np.testing.assert_array_equal(platforms, originals[1])


class WatcherMotionTests(unittest.TestCase):
    def setUp(self):
        self.calls = []

        def traveller(*args, **kwargs):
            self.calls.append((args, kwargs))
            return SP.traveller(*args, **kwargs)

        kit = types.SimpleNamespace(traveller=traveller)
        self.scope = functions('watchers_a.py', ('_dir', '_plate_axes', 'seventh_spot', 'lighter_spot',
            'lighter_pose', '_figure', 'lighter_scene', 'seventh_scene'),
            ('FPS', 'UP', 'WIND', 'LOOK_AZ', 'PLATE_YAW', 'WATCHER_BEARING', 'FIRE_GAP'),
            dict(np=np, math=math, SP=kit, NA=types.SimpleNamespace(GLOW_AZ=-30.), smoothstep=smoothstep,
                 ground=lambda x, z: 0., hearth_scene=lambda *args: None))
        self.fire = np.array([0., 0., 0., 4200., .6, 906.])

    def figure(self, frame, seventh=True):
        self.calls.clear()
        scene = SP.Scene()
        if seventh:
            self.scope['seventh_scene'](scene, frame, self.fire)
        else:
            self.scope['lighter_scene'](scene, frame, self.fire, -.7, 7005)
        self.assertEqual(len(self.calls), 1)
        args, kwargs = self.calls[0]
        primitives, objects = scene.arrays()
        self.assertTrue(np.isfinite(primitives).all())
        self.assertTrue(np.isfinite(objects).all())
        return np.array([args[3], args[4]]), np.asarray(kwargs['carry']), primitives, kwargs

    def test_feet_remain_planted_through_entire_rise_and_turn(self):
        for seventh in (False, True):
            feet0, _, _, _ = self.figure(4198., seventh)
            for frame in range(4199, 4242):
                with self.subTest(seventh=seventh, frame=frame):
                    feet, _, _, _ = self.figure(float(frame), seventh)
                    np.testing.assert_allclose(feet, feet0, rtol=0., atol=1e-12)

    def test_hands_are_continuous_at_old_switch_and_pose_endpoints(self):
        # Subframe probes catch discontinuous branch changes, without assigning an
        # artistic maximum speed. Include the old bad interval and every new endpoint.
        times = set(np.arange(4198., 4241.01, .25)) | {4207., 4208.}
        pose = self.scope['lighter_pose']
        for level in (.2,):
            lo, hi = 4200., 4240.
            for _ in range(50):
                mid = (lo + hi) / 2
                if pose(mid, 4200.)[1] > level:
                    lo = mid
                else:
                    hi = mid
            times.add((lo + hi) / 2)
        for seventh in (False, True):
            for frame in sorted(times):
                with self.subTest(seventh=seventh, frame=frame):
                    _, before, p0, _ = self.figure(frame - 1e-5, seventh)
                    _, after, p1, _ = self.figure(frame + 1e-5, seventh)
                    self.assertEqual(before.shape, (2, 3))
                    self.assertEqual(p0.shape, p1.shape)
                    self.assertLess(float(np.max(np.abs(after - before))), 1e-4)
                    self.assertLess(float(np.max(np.abs(p1 - p0))), 1e-3)

    def test_hand_targets_are_real_glove_geometry_and_release_reaches_rest(self):
        for frame in (4199., 4207., 4208., 4222., 4238., 4241.):
            _, hands, primitives, _ = self.figure(frame)
            gloves = primitives[(primitives[:, 0] == 0.) &
                                np.all(primitives[:, 1:4] == primitives[:, 4:7], axis=1), 1:4]
            for hand in hands:
                self.assertTrue(np.any(np.linalg.norm(gloves - hand, axis=1) < 1e-12))
        # Once reach=0, the explicit targets must coincide with the unchanged
        # traveller's hanging-hand geometry, not leave a residual pointing arm.
        self.figure(4241.)
        args, kwargs = self.calls[0]
        scene = SP.Scene()
        resting = dict(kwargs, carry=None)
        SP.traveller(scene, *args[1:], **resting)
        natural, _ = scene.arrays()
        _, hands, _, _ = self.figure(4241.)
        natural_gloves = natural[(natural[:, 0] == 0.) &
                                np.all(natural[:, 1:4] == natural[:, 4:7], axis=1), 1:4]
        for hand in hands:
            self.assertTrue(np.any(np.linalg.norm(natural_gloves - hand, axis=1) < 1e-12))


if __name__ == '__main__':
    unittest.main()
