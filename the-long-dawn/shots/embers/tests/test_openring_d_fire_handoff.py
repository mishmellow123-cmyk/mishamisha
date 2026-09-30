"""The inscription-to-forging cut keeps its live source and eases its heat.

Record actual compositor inputs without loading tower meshes. Broken controls
substitute the old emitter, move/resize the source, and restore the glare jump.
"""
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(HERE), str(HERE.parents[1] / 'lib')]
import c_d as C
import d_inscription as I
import d_thinking_fire as F


class RecordingFrame:
    def __init__(self, scale=.05):
        self.W, self.H = round(1920 * scale), round(804 * scale)
        self.occ = None

    def set(self, **kwargs):
        pass

    def resolve(self):
        return np.zeros((self.H, self.W, 3), np.float32)


def scene_for(shot='forging'):
    scene = object.__new__(C.Scene)
    scene.shot = shot
    scene.schedule = C.Schedule(shot)
    reference = C.RACE_START + 120. if shot in ('forging', 'race') else C.INSTEP_START + 80.
    scene.rotation = C.D.placement(C.c3._closed_ring_frame(1580.)[0],
                                  scene.camera(reference).pos, C.RING_C, reference)
    scene.towers = SimpleNamespace(k_all=0, prepare=Mock(), emit=Mock())
    scene.smoke = SimpleNamespace(emit=Mock())
    scene.surface_embers = SimpleNamespace(emit_bounded=Mock())
    return scene


def context(scene, frame, scale=.05):
    camera = scene.camera(frame)
    return SimpleNamespace(t=float(frame), t0=frame - .25, t1=frame + .25,
                           scale=scale, cam=camera, cam0=camera, cam1=camera,
                           fr=RecordingFrame(scale))


class ThinkingFireHandoff(unittest.TestCase):
    def rejects(self, oracle, *args):
        with self.assertRaises(AssertionError):
            oracle(*args)

    def test_actual_draw_reuses_incoming_emitter_position_size_and_energy(self):
        scene = scene_for()
        ctx = context(scene, I.END)
        hdr = ctx.fr.resolve()
        with patch.object(F, 'draw') as incoming_draw:
            I.draw_fire(hdr.copy(), I.END - 1, ctx.scale)
            incoming = [call for call in incoming_draw.call_args_list
                        if call.kwargs.get('bright', 1.) > 0.]
        self.assertEqual(len(incoming), 1)
        source = incoming[0]

        def oracle(draw):
            with patch.object(F, 'draw') as thinking, patch.object(C.V5.CF, 'draw') as old, \
                 patch.object(C.c3, 'occ_vis', return_value=np.ones(hdr.shape[:2])):
                draw(hdr.copy(), ctx)
                thinking.assert_called_once()
                old.assert_not_called()
                call = thinking.call_args
                for index in (1, 2):
                    np.testing.assert_array_equal(call.args[index], source.args[index])
                self.assertEqual(call.args[3], I.END)
                self.assertEqual(call.kwargs['bright'], source.kwargs['bright'])
                self.assertEqual(call.kwargs['scale'], source.kwargs['scale'])
                np.testing.assert_array_equal(call.kwargs['vis'], 1.)
        oracle(scene.draw_central_fire)
        self.rejects(oracle, lambda hdr, ctx: C.V5.CF.draw(
            hdr, source.args[1], source.args[2], ctx.t, bright=1.1, scale=ctx.scale))
        anchors = scene.central_fire_anchors
        for offset, resize in ((np.array([9., 0.]), 1.), (np.zeros(2), 1.1)):
            def broken(frame):
                root, tip = anchors(frame)
                return root + offset, root + offset + (tip - root) * resize
            with patch.object(scene, 'central_fire_anchors', side_effect=broken):
                self.rejects(oracle, scene.draw_central_fire)

    def test_source_clock_continues_across_cut(self):
        def oracle(clock):
            times = [clock(f) for f in range(I.END - 1, I.END + 12)]
            np.testing.assert_array_equal(np.diff(times), 1.)
            self.assertEqual(times[0], F.A_ORIGIN + I.END - 1 - F.D_ORIGIN)
        oracle(F.source_clock)
        self.rejects(oracle, lambda frame: F.source_clock(min(frame, I.END - 1)))

    def test_fire_stays_in_picture_as_towers_rise(self):
        scene = scene_for()

        def oracle(anchors):
            for frame in (2080, 2160, 2240, 2399):
                root, tip = anchors(frame)
                points = np.array([root, tip])
                self.assertTrue(np.all(points >= [12., 12.]))
                self.assertTrue(np.all(points <= [1907., 791.]))
                self.assertGreater(np.linalg.norm(tip - root), 60.)
                if frame >= 2160:
                    self.assertLess(root[1], 589.)  # D06's fixed caption top.
        oracle(scene.central_fire_anchors)

        def old_ground_projection(frame):
            points = np.array([C.c3.FIRE_ROOT, C.c3.FIRE_ROOT + [0., C.c3.HF, 0.]])
            x, y, _ = scene.camera(frame).project(points, 1920, 804)
            return np.column_stack([x, y])
        self.rejects(oracle, old_ground_projection)

    def test_actual_pipeline_emits_central_source_only_in_forging(self):
        def oracle(extra_fire=False):
            with patch.object(C, 'Frame', RecordingFrame), \
                 patch.object(C.look, 'finish', side_effect=lambda hdr, **kw: hdr), \
                 patch.object(F, 'draw') as thinking, patch.object(C.V5.CF, 'draw') as old:
                for shot in C.D.SHOTS:
                    scene = scene_for(shot)
                    scene.ring_layer = lambda ctx: ctx.fr.resolve()
                    scene.glare_layer = lambda ctx: ctx.fr.resolve()
                    scene.sparks = scene.drops = scene.lamps = lambda ctx: None
                    thinking.reset_mock()
                    frame = C.D.SHOTS[shot].d_start
                    scene.frame(frame, .05)
                    if extra_fire:
                        F.draw(None, (0., 0.), (0., 10.), frame)
                    self.assertEqual(thinking.call_count, int(shot == 'forging'))
                    old.assert_not_called()
        oracle()
        self.rejects(oracle, True)

    def test_ring_material_starts_at_inscription_and_blends_for_ten_frames(self):
        scene = scene_for()
        records = []

        def raster(*args, **kwargs):
            state = args[6]
            records.append((args, kwargs))
            value = 1. if state.inscription is not None else 3.
            shape = (args[2], args[1])
            return (np.full(shape + (3,), value, np.float32),
                    np.ones(shape, np.float32), np.full(shape, 50., np.float32))

        with patch.object(C.RS, 'render', side_effect=raster), \
             patch.object(C.RS, 'merge_occluder', return_value=None), \
             patch.object(C.RS, 'visibility', side_effect=lambda before, depth, h, w: np.ones((h, w))), \
             patch.object(I, 'inscription', return_value=SimpleNamespace(lv={})):  # Texture never rasterized here.
            for frame, value, count in ((2080, 1., 1), (2085, 2., 2), (2090, 3., 1)):
                records.clear()
                actual = scene.ring_layer(context(scene, frame))
                np.testing.assert_array_equal(actual, value)
                self.assertEqual(len(records), count)
                for args, kwargs in records:
                    state, env = args[6:8]
                    incoming = state.inscription is not None
                    expected = I.ring_state(I.END - 1) if incoming else C.ring_state(frame)
                    for field in ('worked_caps', 'cap_heat_gain', 'letters', 'glow', 'hammer'):
                        self.assertEqual(getattr(state, field), getattr(expected, field))
                    theta = np.linspace(*I.theta_range(I.END - 1), 201)
                    np.testing.assert_array_equal(state.heat(theta), expected.heat(theta))
                    if incoming:
                        self.assertIsInstance(state.inscription, I.FrontInscription)
                        self.assertIsInstance(env, I.ForgeEnvironment)
                        self.assertEqual((kwargs['ss'], kwargs['nt'], kwargs['npp']), (2, 1000, 64))
                        np.testing.assert_array_equal(kwargs['th_range'], I.theta_range(I.END - 1))

            def opening_oracle():
                np.testing.assert_array_equal(scene.ring_layer(context(scene, 2080)), 1.)
            opening_oracle()
            with patch.object(I, 'ring_state', return_value=C.ring_state(2080)):
                self.rejects(opening_oracle)

    def test_glare_is_absent_at_cut_and_reaches_original_gain_by_2090(self):
        scene = scene_for()

        def raster(cam, width, height, ends, caps, score, pulse, visibility):
            return np.full((height, width, 3), score.gain, np.float32)

        def oracle():
            with patch.object(C.d_glare, 'render', side_effect=raster):
                for frame, fraction in ((2080, 0.), (2085, .5), (2090, 1.)):
                    image = scene.glare_layer(context(scene, frame))
                    np.testing.assert_allclose(image, C.D.glare(frame).gain * fraction)
        oracle()
        with patch.object(C, 'ease', return_value=1.):
            self.rejects(oracle)


if __name__ == '__main__':
    unittest.main()
