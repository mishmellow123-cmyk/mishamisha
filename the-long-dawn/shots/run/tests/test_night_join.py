"""Synthetic A14/A15 integration contracts; no terrain initialization or renders.

Run with NUMBA_DISABLE_JIT=1 and one thread. Actual shot function bodies and
camera mathematics are used; render passes and terrain sampling are substitutes.
"""
import math
import os
import sys
import types
import unittest
from unittest import mock

import numpy as np

from test_night_hearth import RUN, functions
import rcam as RC


class NightJoinTests(unittest.TestCase):
    def beacon_render(self):
        """Execute the live A14 render body with one-pixel pass outputs."""
        cam = types.SimpleNamespace(pos=np.zeros(3), W=1, H=1, params=lambda: np.zeros(12))
        frame = types.SimpleNamespace(src=cam)
        fires = np.array([[float(k), 0., 10., 3960. + 40 * k, .6, 900. + k] for k in range(7)])
        light = (np.ones(6), np.ones(3), None, np.zeros(8), np.zeros(24))
        wd = types.SimpleNamespace(march=mock.Mock(), shade=mock.Mock(), cloud_glow=mock.Mock())
        lights = np.array([[6., .8, 10., 1., .4, .1, 8., .3], [0., .8, 10., 1., .4, .1, 7., .3]])
        platforms = np.array([[0., 0., 10., 3.], [6., 0., 10., 3.]])
        originals = lights.copy(), platforms.copy(), fires.copy()
        na = types.SimpleNamespace(lt_rows=lambda *a: lights, pl_rows=lambda *a, **k: platforms,
            ug_rows=lambda f: None, skyline=lambda *a: (0., 0., None), glow_pass=mock.Mock(),
            glow_gp=mock.Mock(return_value=np.zeros(27)), HAZE_K=0., HAZE_D=1.,
            fires_layer=mock.Mock(), ignition_sparks=mock.Mock(return_value=types.SimpleNamespace(render=mock.Mock())))
        hearth = mock.Mock(return_value=types.SimpleNamespace(fire_base_offset=.123))
        lighting = functions('watchers_a.py', ('hearth_lighting',), (), dict(np=np))['hearth_lighting']
        wa = types.SimpleNamespace(hearth_lighting=lighting, seventh_hearth=hearth, draw_figures=mock.Mock())
        scope = functions('beaconrun_a.py', ('render',), ('FPS', 'LT_MAX'),
            dict(np=np, os=os, camera=lambda *a: cam, fires_table=lambda: fires.copy(), CHAIN=fires,
                 light=lambda f: light, CR=np.zeros((0, 16)), WD=wd, NA=na,
                 FD=types.SimpleNamespace(terrain_hmax=lambda cr: 3000.),
                 PI=types.SimpleNamespace(Frame=lambda *a: frame, src_scale=lambda fr: 1.,
                                          to_target=lambda fr: (fr.img, fr.zb, fr.dist)),
                 SK=types.SimpleNamespace(splat_stars=mock.Mock()), stars=lambda: None,
                 draw_flares=mock.Mock()))
        with mock.patch.dict(sys.modules, {'watchers_a': wa}):
            result = scope['render'](4239., scale=.01, mblur=False)
        self.assertEqual(result[0].shape, (1, 1, 3))
        self.assertIs(result[1], cam)
        for actual, before in zip((lights, platforms, fires), originals):
            np.testing.assert_array_equal(actual, before)
        return types.SimpleNamespace(na=na, wa=wa, wd=wd, hearth=hearth, fires=fires,
                                     lights=lights, platforms=platforms)

    def test_beacon_render_uses_shared_seventh_lighting_for_terrain_and_figure(self):
        actual = self.beacon_render()
        lights = actual.wd.shade.call_args.args[5]
        platforms = actual.wd.shade.call_args.args[13]
        self.assertGreater(lights[0, 6], 0.)
        self.assertLess(lights[0, 6], actual.lights[0, 6])
        self.assertGreater(platforms[1, 3], 0.)
        self.assertLess(platforms[1, 3], actual.platforms[1, 3])
        np.testing.assert_array_equal(lights[1], actual.lights[1])
        np.testing.assert_array_equal(platforms[0], actual.platforms[0])
        np.testing.assert_array_equal(actual.wa.draw_figures.call_args.args[4], lights)

    def test_beacon_render_passes_seventh_fuel_seat_and_disables_glow_shadow_rays(self):
        actual = self.beacon_render()
        actual.na.glow_gp.assert_called_once_with(4239., shadow_rays=False)
        self.assertTrue(actual.na.fires_layer.call_args.kwargs.get('near_hearth'))
        self.assertEqual(actual.na.fires_layer.call_args.kwargs.get('base_offsets'), {906: .123})
        actual.hearth.assert_called_once()
        np.testing.assert_array_equal(actual.hearth.call_args.args[0], actual.fires[6])

    def test_actual_camera_pose_and_projection_match_across_4239_4240(self):
        # Only terrain height is synthetic: the shot camera functions, plate,
        # source constants, committed chain and RCam projection are actual code.
        chain = np.load(RUN / 'beaconrun_a_chain.npy')
        night = functions('nighta.py', (), ('GLOW_AZ',), {})
        br = functions('beaconrun_a.py', ('end_cam', '_ease', '_cam_env', 'camera'),
            ('FR0', 'T_SETTLE', 'BACK0', 'YAW0', '_PLATE', 'END_CAM'),
            dict(np=np, math=math, os=os, RC=RC, CHAIN=chain, FIRES=chain))
        wa = functions('watchers_a.py', ('br', '_dir', '_plate_axes', 'seventh_spot', 'chain',
            'seventh_fire', 'plate', 'default_cam', 'camera'),
            ('FR0', 'UP', 'LOOK_AZ', 'PLATE_YAW', 'WATCHER_BEARING', 'FIRE_GAP',
             'EYE', 'BACK', 'HFOV', 'PITCH', 'PUSH'),
            dict(np=np, math=math, os=os, RC=RC, NA=types.SimpleNamespace(**night),
                 ground=lambda x, z: .004 * x - .006 * z))
        modules = {'watchers_a': types.SimpleNamespace(**wa), 'beaconrun_a': types.SimpleNamespace(**br)}
        # Ambient look-development overrides must not turn a default-camera
        # regression into an unrelated local configuration failure.
        with mock.patch.dict(os.environ, {'A14_CAM': '', 'W15_CAM': '', 'W15_OWN_CAM': '', 'W15_P7': ''}), \
                mock.patch.dict(sys.modules, modules):
            for width, height in ((960, 402), (1920, 804)):
                with self.subTest(width=width):
                    last = br['camera'](4239., width, height)
                    first = wa['camera'](4240., width, height)
                    for attr in ('pos', 'fwd', 'right', 'up', 'f', 'cx', 'cy'):
                        np.testing.assert_allclose(getattr(last, attr), getattr(first, attr), rtol=0., atol=1e-12)
                    self.assertEqual((last.W, last.H), (first.W, first.H))
                    # The seven actual fire anchors and the watcher's head span
                    # near and distant projected features at the shared cut.
                    spot = wa['seventh_spot'](chain[6])[0]
                    points = np.vstack([chain[:, :3], spot + [0., 1.75, 0.]])
                    np.testing.assert_allclose(last.project(points), first.project(points), rtol=0., atol=1e-9)


if __name__ == '__main__':
    unittest.main()
