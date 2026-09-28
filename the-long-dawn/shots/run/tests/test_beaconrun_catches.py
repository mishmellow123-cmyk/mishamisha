"""A14 catch and opaque-depth contracts, without terrain initialization or shot renders."""
import math
import os
import sys
import types
import unittest
from unittest import mock

import cv2
import numpy as np

from test_night_hearth import RUN, functions, night_scope
import fire2 as F2
import rcam as RC

cv2.setNumThreads(0)


def ray_distance(camera, forward_depth):
    """Independent geometric oracle using actual source-camera basis vectors."""
    C = camera.params()
    j, i = np.mgrid[:camera.H, :camera.W]
    u, v = i + .5 - C[8], C[9] - j - .5
    rays = np.stack([C[3] * C[7] + C[5] * u, v, C[4] * C[7] + C[6] * u], axis=-1)
    forward = rays[..., 0] * C[3] + rays[..., 2] * C[4]
    return forward_depth * np.linalg.norm(rays, axis=-1) / forward


class BeaconCatchTests(unittest.TestCase):
    def test_loaded_chain_enlarges_only_three_glide_fires_and_preserves_disk_input(self):
        original = np.load(RUN / 'beaconrun_a_chain.npy')
        scope = functions('beaconrun_a.py', ('glide_chain', 'fires_table'), ('CHAIN', 'FIRES'),
            dict(np=np, os=os, HERE=str(RUN), LIT=np.zeros((0, 6)), extra=lambda: np.zeros((0, 6))))
        result = scope['glide_chain'](original)
        expected = original.copy()
        expected[[0, 2, 4], 4] *= 4. / 3.
        np.testing.assert_array_equal(result, expected)
        self.assertFalse(np.shares_memory(result, original))
        np.testing.assert_array_equal(original, np.load(RUN / 'beaconrun_a_chain.npy'))
        # Pin the actual loaded table and A15's FIRES alias, not just an unused helper.
        np.testing.assert_array_equal(scope['CHAIN'], expected)
        np.testing.assert_array_equal(scope['FIRES'], expected)
        np.testing.assert_array_equal(scope['fires_table'](), expected)

    def flare_scope(self, transmission=1.):
        night = night_scope()
        fires = np.array([[0., 0., 8., 1000., 1., float(k)] for k in range(7)])
        fires[6, 3:5] = [0., .6]
        scope = functions('beaconrun_a.py', ('flare_env', 'draw_flares'), ('FLARE_RISE',),
            dict(np=np, math=math, CHAIN=fires, WD=object(),
                 NA=types.SimpleNamespace(fire_dims=night['fire_dims'], fire_depth_bias=night['fire_depth_bias'],
                                          fire_trans=lambda *a: transmission)))
        return scope

    def capture_flares(self, frame=2., transmission=1., seat=.12, pxs=1.):
        scope = self.flare_scope(transmission)
        camera = RC.SrcCam(np.array([0., .65, 0.]), 0., 48. * pxs,
                           -16. * pxs, 16. * pxs, -16. * pxs, 16. * pxs)
        hearth = types.SimpleNamespace(seventh_hearth=lambda p: types.SimpleNamespace(fire_base_offset=seat))
        with mock.patch.dict(sys.modules, {'watchers_a': hearth}), mock.patch.object(F2, 'halo') as halo:
            scope['draw_flares'](None, None, camera, frame, pxs, np.zeros(8))
        return scope, camera, halo.call_args_list

    def test_catch_envelope_retains_original_timing_and_warm_core_is_compact(self):
        scope, _, calls = self.capture_flares()
        for frame in np.arange(-1., 61.01, .5):
            old = 0. if frame < 0. or frame >= 60. else min(frame / 2., 1.) * math.exp(-max(frame - 2., 0.) / 9.)
            self.assertEqual(scope['flare_env'](frame), old)
        self.assertEqual(len(calls), 3)
        aura, broad, core = calls
        self.assertLess(core.args[4], min(aura.args[4], broad.args[4]))
        self.assertGreater(core.args[5], max(aura.args[5], broad.args[5]))
        color = core.kwargs['col']
        self.assertGreater(color[0], color[1])
        self.assertGreater(color[1], color[2])
        self.assertGreater(color[2], 0.)
        for frame in (-1., 0., 60., 61.):
            self.assertEqual(len(self.capture_flares(frame=frame)[2]), 0)

    def test_catch_uses_one_transmission_factor_and_resolution_scaled_radius(self):
        _, _, full = self.capture_flares()
        _, _, dim = self.capture_flares(transmission=.25)
        _, _, half = self.capture_flares(pxs=.5)
        for a, b, c in zip(full, dim, half):
            self.assertEqual(b.args[5], a.args[5] * .25)
            self.assertEqual(c.args[4], a.args[4] * .5)
            self.assertEqual(c.args[5], a.args[5])

    def test_seventh_catch_tracks_measured_fuel_seat(self):
        scope, camera, low = self.capture_flares(seat=.12)
        _, _, high = self.capture_flares(seat=.32)
        Hf = scope['NA'].fire_dims(.6)[0]
        for a, b in zip(low, high):
            self.assertEqual(a.args[2], b.args[2])
            self.assertAlmostEqual(a.args[3] - b.args[3], camera.f * .20 / 8.)
            world_y = camera.pos[1] + (camera.cyy - a.args[3]) * 8. / camera.f
            self.assertGreater(world_y, .12)
            self.assertLess(world_y - .12, .25 * Hf)

    def test_real_catch_pixels_are_occluded_by_close_foreground(self):
        scope = self.flare_scope()
        camera = RC.SrcCam(np.array([0., .65, 0.]), 0., 48., -16., 16., -16., 16.)
        hearth = types.SimpleNamespace(seventh_hearth=lambda p: types.SimpleNamespace(fire_base_offset=.12))

        def draw(depth):
            image = np.zeros((32, 32, 3), np.float32)
            zb = np.full((32, 32), depth, np.float32)
            with mock.patch.dict(sys.modules, {'watchers_a': hearth}):
                scope['draw_flares'](image, zb, camera, 2., 1., np.zeros(8))
            np.testing.assert_array_equal(zb, np.full_like(zb, depth))
            return image

        self.assertGreater(float(draw(1e9).max()), 0.)
        # The old 3 m flare tolerance painted this half-metre foreground surface.
        np.testing.assert_array_equal(draw(7.5), np.zeros((32, 32, 3), np.float32))

    def test_worker_selects_sequential_opencv_after_import(self):
        numba = types.SimpleNamespace(set_num_threads=mock.Mock(), config=types.SimpleNamespace(NUMBA_NUM_THREADS=2))
        cv = types.SimpleNamespace(setNumThreads=mock.Mock())
        work = functions('beaconrun_a.py', ('_work',), (),
                         dict(PI=types.SimpleNamespace(look=object())))['_work']
        with mock.patch.dict(sys.modules, {'numba': numba, 'cv2': cv}):
            work(([], .5, 1.5, 'unused', 2))
        cv.setNumThreads.assert_called_once_with(0)


class OpaqueMotionDepthTests(unittest.TestCase):
    def test_source_conversion_preserves_unchanged_terrain_sky_and_depth_buffer(self):
        camera = RC.SrcCam(np.array([1., 2., 3.]), math.radians(31.), 4., -2., 1., -1., 1.)
        original_zb = np.array([[20., 20., 20.], [1e9, 20., 1e9]], np.float32)
        zb = original_zb.copy()
        zb[0, 0], zb[1, 0] = 4., 5.
        zb[0, 1], zb[1, 1] = 21., np.nan  # no valid nearer hit
        before = zb.copy()
        original_dist = ray_distance(camera, original_zb).astype(np.float32)
        original_dist[1, 2] = 1e9
        frame = types.SimpleNamespace(src=camera, zb=zb, dist=original_dist.copy())
        apply = functions('beaconrun_a.py', ('sync_opaque_distance',), (), dict(np=np))['sync_opaque_distance']
        apply(frame, original_zb)
        changed = np.zeros_like(zb, bool)
        changed[0, 0] = changed[1, 0] = True
        np.testing.assert_allclose(frame.dist[changed], ray_distance(camera, zb)[changed], rtol=1e-7)
        np.testing.assert_array_equal(frame.dist[~changed], original_dist[~changed])
        np.testing.assert_array_equal(frame.zb, before)
        self.assertTrue(np.all(frame.dist[changed] > zb[changed]))

    def test_actual_render_sends_nearer_opaque_distance_through_warp_to_velocity(self):
        W, H, f = 4, 2, 3990.

        def camera(frame, width=W, height=H):
            return RC.RCam(np.array([.2 * (frame - f), 1., 0.]), 11., -4., 0., 45., width, height)

        frames, snapshots = [], []
        target = functions('pipe.py', ('to_target',), (), dict(np=np, cv2=cv2, RC=RC))['to_target']

        def make_frame(cam, ss):
            fr = types.SimpleNamespace(t_out=cam, t_ss=cam.scaled(ss), ss=ss, src=RC.source_for(cam.scaled(ss)))
            frames.append(fr)
            return fr

        def shade(*args):
            fr = frames[-1]
            args[10][:] = .1
            args[11][:] = 100.
            args[12][:] = ray_distance(fr.src, 100.)
            fr.terrain_dist = fr.dist.copy()

        def figures(img, zb, scam, *a, **kw):
            # Two opaque source pixels stand in for hearth and figure writes.
            zb[scam.H // 2, scam.W // 2] = 5.
            zb[scam.H // 2, scam.W // 2 + 1] = 7.

        def to_target(fr):
            snapshots.append(fr.dist.copy())
            return target(fr)

        fires = np.array([[0., 0., 8., 10000., .6, float(k)] for k in range(7)])
        wa = types.SimpleNamespace(draw_figures=figures, hearth_lighting=lambda l, p, f: (l, p),
                                   seventh_hearth=lambda p: types.SimpleNamespace(fire_base_offset=.12))
        na = types.SimpleNamespace(lt_rows=lambda *a: np.zeros((0, 8)), pl_rows=lambda *a, **k: np.zeros((0, 4)),
             ug_rows=lambda f: None, skyline=lambda *a: (0., 0., None), glow_pass=mock.Mock(),
             glow_gp=lambda *a, **k: np.zeros(27), HAZE_K=0., HAZE_D=1., fires_layer=mock.Mock())
        velocity = mock.Mock(wraps=RC.velocity)
        blur = mock.Mock(side_effect=lambda img, *a, **k: img)
        scope = functions('beaconrun_a.py', ('render', 'sync_opaque_distance', '_cam_env'), ('FPS', 'LT_MAX', 'SHUTTER'),
            dict(np=np, os=os, camera=camera, CHAIN=fires, fires_table=lambda: fires.copy(), CR=np.zeros((0, 16)),
                 light=lambda f: (np.ones(6), np.ones(3), None, np.zeros(8), np.zeros(24)), NA=na,
                 WD=types.SimpleNamespace(march=mock.Mock(), shade=shade, cloud_glow=mock.Mock()),
                 FD=types.SimpleNamespace(terrain_hmax=lambda cr: 3000.),
                 PI=types.SimpleNamespace(Frame=make_frame, src_scale=lambda fr: 1., to_target=to_target),
                 SK=types.SimpleNamespace(splat_stars=mock.Mock()), stars=lambda: None, draw_flares=mock.Mock(),
                 RC=types.SimpleNamespace(velocity=velocity, motion_blur=blur)))
        with mock.patch.dict(sys.modules, {'watchers_a': wa}), mock.patch.dict(os.environ, {'A14_CAM': ''}):
            image, cam, distances = scope['render'](f, scale=W / 1920., ss=1., mblur=True)
        fr = frames[0]
        changed = fr.zb < 100.
        self.assertEqual(int(changed.sum()), 2)
        np.testing.assert_allclose(snapshots[0][changed], ray_distance(fr.src, fr.zb)[changed], rtol=1e-7)
        np.testing.assert_array_equal(snapshots[0][~changed], fr.terrain_dist[~changed])
        velocity.assert_called_once()
        np.testing.assert_array_equal(velocity.call_args.args[3], distances)
        expected = RC.velocity(lambda ff: camera(ff, W, H), f, cam, distances, shutter=scope['SHUTTER'])
        np.testing.assert_array_equal(blur.call_args.args[1], expected)
        self.assertTrue(np.isfinite(expected).all())
        self.assertGreater(float(np.abs(expected).max()), 0.)
        self.assertTrue(np.any(distances < 20.))
        j, i = np.mgrid[:H, :W]
        forward = distances * cam.f / np.linalg.norm(cam.ray(i + .5, j + .5), axis=-1)
        np.testing.assert_allclose(blur.call_args.args[2], forward, rtol=1e-7)
        self.assertEqual(image.shape, (H, W, 3))


if __name__ == '__main__':
    unittest.main()
