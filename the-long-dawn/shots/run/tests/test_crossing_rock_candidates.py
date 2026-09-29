"""Rock contracts without importing crossing, terrain initialization or Numba."""
import ast
import copy
import math
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
BEFORE_IMPORT = set(sys.modules)
import crossing_rock_candidates as C
ADDED_MODULES = set(sys.modules)-BEFORE_IMPORT


def definitions(path, names, namespace):
    tree = ast.parse(path.read_text())
    picked = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names]
    if {node.name for node in picked} != set(names):
        raise AssertionError('Accepted source definitions moved')
    for node in picked:
        node.decorator_list = []
    exec(compile(ast.Module(body=picked, type_ignores=[]), str(path), 'exec'), namespace)


NS = dict(np=np, math=math, NP=20, NO=16)
definitions(HERE.parents[0]/'montage/mt/noise.py', ('ihash', 'hash3i', 'fade', '_grad3', 'gnoise3'), NS)
definitions(HERE/'sdfppl.py', ('Scene', '_sd_box', '_smin', '_prim', '_map'), NS)
Scene = NS['Scene']
TREE = ast.parse((HERE/'crossing.py').read_text())
BUILD = next(node for node in TREE.body if isinstance(node, ast.FunctionDef) and node.name == 'build_scene')
# Execute the actual accepted rock block, not a second copy in the test.
ROCK = next(node for node in BUILD.body if isinstance(node, ast.If) and node.body
            and isinstance(node.body[0], ast.Assign) and isinstance(node.body[0].targets[0], ast.Tuple)
            and [x.id for x in node.body[0].targets[0].elts] == ['p0', 'w0'])


def crossing_fixture():
    event = dict(s_edge=0., ka=0, kb=1, t0=27., t_go=31.8)
    def at(s):
        q = np.asarray(s)
        return np.stack([np.zeros_like(q), np.zeros_like(q), q], -1), np.broadcast_to([0., 0., 1.], q.shape+(3,)).copy()
    crossing = SimpleNamespace(STEP_H=.42, step_event=lambda cfg:event, at=at,
        ground_many=lambda p:np.zeros(np.asarray(p).reshape(-1, 3).shape[0]), SP=SimpleNamespace(_map=NS['_map']))
    return crossing


def accepted_scene(crossing):
    scene = Scene()
    scene.begin();scene.box([-5., 0., 0.], [.2, .3, .4], mat=6);scene.end();scene.O[-1][14] = -1.
    namespace = dict(np=np, math=math, sc=scene, ev=crossing.step_event({}), at=crossing.at,
                     ground_many=crossing.ground_many, UP=np.array([0., 1., 0.]), STEP_H=.42)
    exec(compile(ast.Module(body=[ROCK], type_ignores=[]), 'accepted_crossing_rock_block', 'exec'), namespace)
    scene.begin();scene.box([5., 0., 0.], [.1, .2, .3], mat=0);scene.end()
    return scene


class RockCandidates(unittest.TestCase):
    def test_import_has_no_crossing_or_numba_side_effects(self):
        for name in ('crossing', 'sdfppl', 'numba'):
            self.assertNotIn(name, ADDED_MODULES)

    def test_default_returns_identical_scene_without_reading_crossing(self):
        original = object()
        self.assertIs(C.apply_to_scene(original, None, None), original)
        self.assertIs(C.apply_to_scene(original, None, None, 'accepted'), original)

    def test_exact_source_ownership_and_unrelated_rows_preserved(self):
        crossing = crossing_fixture()
        original = accepted_scene(crossing)
        P, O = original.arrays()
        result = C.apply_to_scene(original, crossing, {}, 'low_shoulders')
        self.assertIsNot(result, original)
        actualP, actualO = result.arrays()
        np.testing.assert_array_equal(original.arrays()[0], P)
        np.testing.assert_array_equal(original.arrays()[1], O)
        self.assertEqual(result.crossing_rock_study['object_index'], 1)
        np.testing.assert_array_equal(actualP[[0, 4]], P[[0, 4]])
        np.testing.assert_array_equal(actualO[[0, 2]], O[[0, 2]])
        self.assertFalse(np.array_equal(actualP[1:4], P[1:4]))
        self.assertTrue((actualP[1:4, 9] == 6).all())
        self.assertEqual(actualO[1, 14], -1.)
        self.assertTrue(np.isfinite(actualP).all() and np.isfinite(actualO).all())
        self.assertEqual(crossing.STEP_H, .42)
        self.assertFalse(result.crossing_rock_study['boot_contact_proved'])

    def test_mismatch_or_ambiguous_owned_object_fails_without_mutating(self):
        crossing = crossing_fixture()
        changed = accepted_scene(crossing)
        changed.P[1][11] += .001
        before = copy.deepcopy(changed.P)
        with self.assertRaises(ValueError): C.apply_to_scene(changed, crossing, {}, 'low_shoulders')
        np.testing.assert_array_equal(changed.P, before)
        duplicate = accepted_scene(crossing)
        duplicate.O.append(list(duplicate.O[1]))
        with self.assertRaises(ValueError): C.apply_to_scene(duplicate, crossing, {}, 'low_shoulders')
        crossing.STEP_H = .43
        with self.assertRaises(ValueError): C.apply_to_scene(accepted_scene(crossing), crossing, {}, 'low_shoulders')

    def test_no_event_and_bad_option_contracts(self):
        scene, diagnostics = Scene(), {}
        crossing = crossing_fixture();crossing.step_event=lambda cfg:None
        self.assertIs(C.apply_to_scene(scene, crossing, {}, 'low_shoulders', diagnostics), scene)
        self.assertFalse(diagnostics['applied'])
        with self.assertRaises(ValueError): C.apply_to_scene(scene, crossing, {}, 'other')

    def test_flat_tread_and_lower_shoulders_use_actual_sdf_math(self):
        crossing = crossing_fixture()
        result = C.apply_to_scene(accepted_scene(crossing), crossing, {}, 'low_shoulders')
        support = result.crossing_rock_study['support']
        self.assertAlmostEqual(support['core_top_y'], .42)
        self.assertTrue(all(margin > .04 for margin in support['shoulder_upper_below_tread_m']))
        core = np.asarray(result.P[1:4])
        # The kernel's true scalar SDF source (decorators removed), including
        # its real hash noise. These are synthetic flat-terrain coordinates.
        for x in (-.18, 0., .18):
            for z in (.2, .65, 1.1):
                top = C._surface_y(NS['_map'], core, x, z, -1., 1.)
                self.assertAlmostEqual(top, .42, places=6)
        self.assertTrue((core[:, 10] == 0).all())
        self.assertEqual(core[0, 5], 0.)
        self.assertEqual(core[0, 14], 0.)

    def test_support_probe_reports_gaps_and_empty_denominator_honestly(self):
        crossing = crossing_fixture()
        result = C.apply_to_scene(accepted_scene(crossing), crossing, {}, 'low_shoulders')
        crossing.SPEED = .45
        crossing.line_s = lambda t,cfg:np.array([0., 0., .55, .75])
        crossing.walker_cut = lambda k:dict(h=1.)
        crossing.stride_of = lambda i:.72
        crossing._hh = lambda i,j:0.
        crossing.foot_cycle = lambda arc,stride,phase,delta:(arc, 0.)
        crossing.smoothstep = lambda a,b,x:float(np.clip((x-a)/(b-a), 0, 1))
        crossing._facing = lambda w,theta:w
        lift = [0.]
        def plant(figs, **kwargs):
            s=figs[0][0]
            return [(np.array([0., 1., s]), np.array([.1, .495+lift[0], s]),
                     np.array([-.1, .495+lift[0], s]), np.array([0., 0., 1.]))]
        crossing.plant = plant
        report = C.probe_support(crossing, {}, result, [5584])
        self.assertEqual(report['status'], 'sampled_geometry_pass')
        self.assertEqual(report['plateau_stance_samples'], 4)
        self.assertAlmostEqual(report['max_abs_plateau_stance_gap_m'], .008, places=6)
        self.assertFalse(report['boot_contact_proved'])
        lift[0] = .12
        report = C.probe_support(crossing, {}, result, [5584])
        self.assertEqual(report['status'], 'failed')
        self.assertEqual(report['plateau_stance_failures'], 4)
        crossing.foot_cycle=lambda *a:(3., 0.)
        report = C.probe_support(crossing, {}, result, [5584])
        self.assertEqual(report['status'], 'no_plateau_stance_samples')
        self.assertIsNone(report['max_abs_plateau_stance_gap_m'])


if __name__ == '__main__':
    unittest.main()
