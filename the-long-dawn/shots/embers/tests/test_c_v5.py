"""Lightweight schedule and actual compositor-routing tests; no renderer/JIT."""
import sys
from pathlib import Path
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(HERE), str(HERE.parents[1] / 'lib')]
import c_v5 as C


class Timing(unittest.TestCase):
    def test_absolute_shot_ranges_reject_unchanged_first_half(self):
        self.assertEqual(len(C.frames('2320-2639,3840-4239')), 720)
        for f in (0, 2079, 2319, 2640, 3839, 4240):
            with self.assertRaises(ValueError): C.shot_at(f)

    def test_one_forge_sinks_before_others_respond(self):
        levels = np.array([C.forge_level(i, 2460) for i in range(18)])
        self.assertLess(levels[C.LOW_FORGE], .1)
        self.assertTrue(np.all(np.delete(levels, C.LOW_FORGE) == 1.))
        self.assertGreater(C.forge_level(0, 2510), 1.5)
        self.assertLess(C.forge_level(C.LOW_FORGE, 2510), .1)
        self.assertGreater(C.forge_level(C.LOW_FORGE, 2560), 1.)

    def test_shutdown_is_simultaneous_and_does_not_dim_ring(self):
        for i in range(18):
            self.assertGreater(C.forge_level(i, 3847.999), 0.)
            self.assertEqual(C.forge_level(i, 3848), 0.)
            self.assertEqual(C.forge_level(i, 4239), 0.)
        self.assertEqual(C.ring_warmth(3999), 1.)

    def test_ring_cools_monotonically_and_last_second_is_clean(self):
        warmth = [C.ring_warmth(f) for f in range(4000, 4240)]
        self.assertTrue(np.all(np.diff(warmth) <= 0))
        for f in range(4216, 4240):
            self.assertEqual(C.ring_warmth(f), 0.)
            self.assertEqual(C.storm_amount(f), 0.)
            self.assertEqual(C.gold_amount(f), 0.)

    def test_response_height_does_not_start_early_or_sink_after_shutdown(self):
        tw = C.Forges.__new__(C.Forges)
        tw.rest = np.linspace(26., 39., 18)
        self.assertEqual(tw.height(0, 2479), tw.height(0, 2420))
        self.assertGreater(tw.height(0, 2539), tw.height(0, 2479))
        self.assertEqual(tw.height(C.LOW_FORGE, 2539), tw.rest[C.LOW_FORGE])
        self.assertGreater(tw.height(C.LOW_FORGE, 2579), tw.rest[C.LOW_FORGE])
        for leader in C.LEADERS:
            self.assertGreater(tw.height(leader, 2639), max(tw.height(i, 2639) for i in range(18) if i not in C.LEADERS))
        for i in range(18):
            self.assertEqual(tw.height(i, 2639), tw.height(i, 4239))

    def test_rivals_share_existing_form_height_and_intensity_without_shared_edits(self):
        forms = [object() for _ in range(18)]
        def initialize(tw):
            tw.G = forms.copy()
            tw.h_rise = np.linspace(26., 39., 18)
            tw.J = np.zeros((18, 2))
            tw.design = np.arange(18)
        with patch.object(C.B.Towers, '__init__', initialize), patch.object(C.c3, 'layout_towers', side_effect=lambda tw:tw), patch.object(C.EYE, '_medieval', side_effect=lambda tw:tw):
            tw = C.Forges()
        a, b = C.LEADERS
        self.assertIs(tw.G[a], forms[b])
        self.assertIs(tw.G[b], forms[b])
        self.assertIsNot(forms[a], forms[b])
        self.assertEqual(tw.design[a], tw.design[b])
        for f in np.arange(2580., 2640., .25):
            self.assertEqual(tw.height(a, f), tw.height(b, f))
            self.assertEqual(C.forge_level(a, f), C.forge_level(b, f))
        for i in range(18):
            if i not in C.LEADERS:
                self.assertIs(tw.G[i], forms[i])

    def test_flame_and_smoke_crown_follows_actual_bent_geometry(self):
        tw = C.Forges.__new__(C.Forges)
        tw.rest = np.full(18, 32.)
        tw.ang = np.linspace(0., 2*np.pi, 18, endpoint=False)
        tw.rad = np.full(18, 30.)
        for f in (2580., 2610., 2639., 3848., 4216.):
            for i in C.LEADERS:
                axis_top = C.B.Towers.top(tw, i, f)
                expected = C.SCHED.tower_post(tw, i, axis_top[None, :], f)[0]
                np.testing.assert_array_equal(tw.top(i, f), expected)
                if f >= 2610:
                    self.assertLess(np.linalg.norm(tw.top(i, f)[[0,2]]), np.linalg.norm(axis_top[[0,2]]))

    def test_world_restores_shared_defaults_even_on_exception(self):
        before = C.variant.CUT, C.B.SCHED, C.B.IGN, C.B.BEATS
        with self.assertRaises(RuntimeError):
            with C.world():
                self.assertEqual(C.variant.CUT, 'C3')
                raise RuntimeError('test')
        self.assertEqual((C.variant.CUT, C.B.SCHED, C.B.IGN, C.B.BEATS), before)

    def test_camera_is_continuous_at_cold_ring_join_and_holds_clean(self):
        scene = C.Scene.__new__(C.Scene)
        a, b = scene.camera(3999.999), scene.camera(4000.001)
        self.assertLess(np.linalg.norm(a.pos-b.pos), .001)
        for f in range(4216, 4240):
            np.testing.assert_array_equal(scene.camera(f).pos, scene.camera(4216).pos)
            np.testing.assert_array_equal(scene.ring_frame(f)[0], scene.ring_frame(4216)[0])

    def test_entire_canonical_band_stays_inside_frame_with_margin(self):
        scene = C.Scene.__new__(C.Scene)
        th, ps = np.meshgrid(np.linspace(0,2*np.pi,80),np.linspace(0,2*np.pi,12))
        points,_ = C.RS.local_points(th.ravel(),ps.ravel())
        for a,b in C.SHOTS.values():
            for f in range(a,b,4):
                rot,centre,size=scene.ring_frame(f)
                u,v,z=scene.camera(f).project(centre+size*points@rot.T,960,402)
                self.assertGreater(z.min(),0.)
                self.assertGreater(u.min(),24.)
                self.assertLess(u.max(),936.)
                self.assertGreater(v.min(),24.)
                self.assertLess(v.max(),378.)

    def test_jobs_cover_only_assigned_frames_without_duplicates(self):
        import json, shlex
        for shot,(a,b) in C.SHOTS.items():
            job=json.loads((C.ROOT/'cloud'/'jobs'/f'embers_C5_{shot}.json').read_text())
            actual=[]
            for command in job['render']:
                tokens=shlex.split(command)
                i=tokens.index('--frames')
                actual += C.frames(tokens[i+1])
                self.assertIn('--format',tokens)
                self.assertEqual(tokens[tokens.index('--format')+1],'png')
            # Execute only the farm's regex assignments, never import its daemon/CLI.
            import ast, re
            parsed=ast.parse((C.ROOT/'cloud'/'farm.py').read_text())
            assignments=[node for node in parsed.body if isinstance(node,ast.Assign)
                         and any(isinstance(t,ast.Name) and t.id in ('RX','RX_STEP') for t in node.targets)]
            assignments += [node for node in parsed.body if isinstance(node,(ast.FunctionDef,ast.ClassDef))
                            and node.name in ('frames_of','LaneCmd')]
            scope={'re':re}
            exec(compile(ast.Module(body=assignments,type_ignores=[]),'farm_RX','exec'),scope)
            for command in job['render']:
                matches=[(kind,m) for kind,rx in scope['RX'] for m in rx.finditer(command)]
                self.assertEqual(len(matches),1)
                self.assertEqual(matches[0][0],'frames')
                lane=scope['LaneCmd'](command)
                self.assertEqual(lane.seq,C.frames(matches[0][1].group(2)))
                pick=[lane.seq[len(lane.seq)//2]]
                rewritten=scope['LaneCmd'](lane.render(pick))
                self.assertEqual(rewritten.seq,pick)
            self.assertEqual(sorted(actual),list(range(a,b)))
            self.assertEqual(job['out_dir'],f'renders/embers_C5_{shot}')

    def test_output_preflight_rejects_existing_and_duplicate_frames(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            out=Path(d)
            self.assertEqual(C.output_paths(out,[2360],'jpg'),[out/'f_02360.jpg'])
            (out/'f_02360.jpg').write_bytes(b'preserve me')
            with self.assertRaises(FileExistsError): C.output_paths(out,[2360],'jpg')
            self.assertEqual((out/'f_02360.jpg').read_bytes(),b'preserve me')
        with self.assertRaises(ValueError): C.frames('2360,2360')

    def test_cold_ring_is_grey_with_same_solid_coverage(self):
        scene = C.Scene.__new__(C.Scene)
        scene.towers = SimpleNamespace(k_all=0)
        colour = np.ones((2, 2, 3), np.float32) * [1., .5, .1]
        ctx = SimpleNamespace(t=4230., fr=SimpleNamespace(W=2, H=2), cam=scene.camera(4230.))
        captured = []
        def render(cam,W,H,rot,centre,size,state,env):
            captured.append(state)
            return colour.copy(), np.ones((2,2)), np.ones((2,2))
        with patch.object(C.RS, 'render', render), patch.object(C.RS, 'merge_occluder', return_value=None), patch.object(C.RS, 'visibility', return_value=np.ones((2,2))):
            rgb = scene.ring_layer(ctx)
        np.testing.assert_array_equal(rgb[...,0], rgb[...,1])
        np.testing.assert_array_equal(rgb[...,1], rgb[...,2])
        self.assertEqual(captured[0].letters, 0.)
        self.assertEqual(captured[0].glow, 0.)
        self.assertTrue(np.all(captured[0].heat(np.array([0.,1.])) == 0))

    def test_actual_frame_dispatch_turns_every_flame_off_on_downbeat(self):
        scene = C.Scene.__new__(C.Scene)
        contexts = []
        scene.towers = SimpleNamespace(k_all=3, prepare=lambda ctx: contexts.append(ctx),
            top=lambda i,t: np.array([i*4., 20., 0.]), emit=lambda *args: None)
        scene.smoke = SimpleNamespace(emit=lambda *args: None)
        scene.ring_layer = lambda ctx: np.zeros((4,8,3), np.float32)
        scene.storm = lambda ctx: np.zeros((4,8,3), np.float32)
        scene.drops = lambda ctx: None
        class Frame:
            W,H,occ = 8,4,None
            def __init__(self, scale): pass
            def set(self, **kw): pass
            def resolve(self): return np.zeros((4,8,3), np.float32)
        with patch.object(C, 'Frame', Frame), patch.object(C.CF, 'draw') as draw, patch.object(C.c3, 'occ_vis', return_value=1.), patch.object(C.look, 'finish', side_effect=lambda hdr,**kw:hdr):
            scene.frame(3847)
            self.assertEqual(draw.call_count, 3)
            draw.reset_mock()
            scene.frame(3848)
            self.assertEqual(draw.call_count, 0)
            self.assertEqual(contexts[-1].t0, 3848.)

    def test_actual_tower_emission_override_kills_hot_edges_but_keeps_stone(self):
        tw = C.Forges.__new__(C.Forges)
        stone = {'P': np.array([[0.,0.,0.]]), 'N': np.array([[0.,0.,1.]])}
        edge = {}
        tw.cur = [{'parts': {0: stone}}]
        ctx = SimpleNamespace(t=3848,cam=SimpleNamespace(pos=np.array([0.,0.,1.])))
        calls=[]
        with patch.object(C.B.Towers, '_splat', side_effect=lambda *a:calls.append(a)):
            tw._splat(ctx,0,edge,np.array([[2.,1.,.2]]),None,None,None,None,None)
            tw._splat(ctx,0,stone,np.array([[2.,1.,.2]]),None,None,None,None,None)
        self.assertTrue(np.all(calls[0][3] == 0.))
        self.assertTrue(np.all(calls[1][3] > 0.))
        self.assertGreater(calls[1][3][0,2],calls[1][3][0,0])

    def test_cold_smoke_keeps_energy_and_has_no_orange_emission(self):
        calls=[]
        frame=SimpleNamespace(splat=lambda *a,**k:calls.append(a))
        proxy=C.SmokeFrame(frame,3900)
        proxy.splat(None,None,None,np.array([2.]),np.array([[1.,.4,.1]]))
        self.assertGreater(calls[0][3][0],0.)
        self.assertGreater(calls[0][4][0,2],calls[0][4][0,0])
        calls.clear()
        C.SmokeFrame(frame,4216).splat(None,None,None,np.array([2.]),np.array([[1.,.4,.1]]))
        self.assertEqual(calls[0][3][0],0.)


if __name__ == '__main__': unittest.main()
