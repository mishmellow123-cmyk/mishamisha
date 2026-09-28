"""Compiled noise boundaries and real shader expressions; no full-frame render."""
import ast
import copy
import inspect
import json
from pathlib import Path
import sys
import textwrap
import types
import unittest

import numpy as np
from numba import njit

RUN = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RUN))
sys.path.insert(0, str(RUN.parent / 'montage'))
from mt import noise as NOISE

SHARED_NOISE = NOISE.gnoise3
SHARED_OPTIONS = dict(SHARED_NOISE.targetoptions)
import sdfppl as SP

SHADER_SEEDS = (11, 12, 13, 14, 15, 21, 22, 23, 31, 32, 33, 34, 35, 41, 42)


@njit(fastmath=True)
def original_shader_noise(x, y, z, seed):
    # All original SP callers are fastmath=True; direct dispatch of the shared
    # function alone would not necessarily have that compilation context.
    return SHARED_NOISE(x, y, z, seed)


@njit(fastmath=False)
def strict_caller(x, y, z, seed):
    return SP.gnoise3(x, y, z, seed)


@njit(fastmath=True)
def biased_noise(x, y, z, seed):
    """Deliberate numerical mutation for the helper-comparison negative control."""
    return SHARED_NOISE(x, y, z, seed) + 1e-6


def points():
    ordinary = [(0., 0., 0.), (.125, .25, .5), (-.125, -.25, -.5),
                (-1004.137, 235.731, -508.913), (61252.3, -91437.1, 125006.7)]
    # Put each coordinate on, just below, and just above a lattice boundary;
    # leave the other coordinates fractional so these are not all zero noise.
    for axis in range(3):
        for cell in (-2048., -2., -1., 0., 1., 2., 2048.):
            for value in (np.nextafter(cell, -np.inf), cell, np.nextafter(cell, np.inf)):
                xyz = [.371, -.819, .213]
                xyz[axis] = value
                ordinary.append(tuple(xyz))
    ordinary.extend(np.random.default_rng(20260928).uniform(-150000., 150000., (32, 3)))
    return np.asarray(ordinary, dtype=np.float64)


def original_helper(dispatcher, noise=SHARED_NOISE):
    """Compile the actual helper body against its former shared noise binding."""
    function = dispatcher.py_func
    globals_ = dict(function.__globals__)
    globals_['gnoise3'] = noise
    clone = types.FunctionType(function.__code__, globals_, function.__name__, function.__defaults__)
    return njit(fastmath=True)(clone)


def render_noise_calls(noise):
    """Compile real cloth/stone sample coordinates and seeds, without the tracer.

    Extracting the calls makes recipe edits part of this check rather than
    maintaining a second, potentially stale transcription of those expressions.
    The full renderer and its lighting/normal arithmetic need separate image tests.
    """
    tree = ast.parse(textwrap.dedent(inspect.getsource(SP.render.py_func)))
    calls = [copy.deepcopy(node) for node in ast.walk(tree)
             if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
             and node.func.id == 'gnoise3']
    function = ast.parse('def sample(qx, qy, qz):\n    pass\n').body[0]
    function.body = [ast.Return(value=ast.Tuple(elts=calls, ctx=ast.Load()))]
    module = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
    scope = {'gnoise3': noise}
    exec(compile(module, '<actual SP.render noise calls>', 'exec'), scope)
    return njit(fastmath=True)(scope['sample']), calls


class SDFNoiseDispatchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.xyz = points()

    def assert_samples(self, label, actual, expected, maxulp=0):
        actual = np.asarray(actual)
        expected = np.asarray(expected)
        self.assertTrue(np.isfinite(actual).all(), label)
        self.assertTrue(np.isfinite(expected).all(), label)
        print(json.dumps({'comparison': label, 'values': actual.size,
                          'allowed_float64_ulps': maxulp,
                          'unequal_values': int(np.count_nonzero(actual != expected)),
                          'max_abs_difference': float(np.max(np.abs(actual - expected))),
                          'differences': [dict(index=list(index), actual=float(actual[index]),
                                               expected=float(expected[index]),
                                               expected_ulps=float(abs(actual[index] - expected[index]) /
                                                                   abs(np.spacing(expected[index]))))
                                          for index in zip(*np.nonzero(actual != expected))]},
                         default=int), flush=True)
        if maxulp:
            np.testing.assert_array_max_ulp(actual, expected, maxulp=maxulp)
        else:
            np.testing.assert_array_equal(actual, expected, err_msg=label)

    def test_local_policy_is_explicit_and_shared_dispatcher_unchanged(self):
        self.assertIs(NOISE.gnoise3, SHARED_NOISE)
        self.assertIs(SP._shared_gnoise3, SHARED_NOISE)
        self.assertIsNot(SP.gnoise3, SHARED_NOISE)
        self.assertEqual(NOISE.gnoise3.targetoptions, SHARED_OPTIONS)
        self.assertEqual(SHARED_NOISE.targetoptions['inline'], 'always')
        self.assertEqual(SP.gnoise3.targetoptions['inline'], 'never')
        self.assertIs(SP.gnoise3.targetoptions.get('fastmath'), True)
        self.assertIs(SP.render.targetoptions.get('fastmath'), True)
        self.assertIs(SP.render.targetoptions.get('parallel'), True)

    def test_every_actual_shader_seed_is_covered(self):
        tree = ast.parse(Path(SP.__file__).read_text())
        calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                 and isinstance(node.func, ast.Name) and node.func.id == 'gnoise3']
        self.assertTrue(calls)
        self.assertTrue(all(isinstance(node.args[3], ast.Constant) for node in calls))
        self.assertEqual({node.args[3].value for node in calls}, set(SHADER_SEEDS))

    def test_scalar_boundary_coordinates_and_all_material_seeds(self):
        actual, expected = [], []
        for seed in SHADER_SEEDS:
            for x, y, z in self.xyz:
                actual.append(SP.gnoise3(x, y, z, seed))
                expected.append(original_shader_noise(x, y, z, seed))
        self.assert_samples('scalar shader-context noise', actual, expected)

    def test_strict_caller_cannot_change_explicit_shader_math_policy(self):
        actual, expected = [], []
        for seed in SHADER_SEEDS:
            for x, y, z in self.xyz[::7]:
                actual.append(strict_caller(x, y, z, seed))
                expected.append(original_shader_noise(x, y, z, seed))
        self.assert_samples('strict caller of SP-local noise', actual, expected)

    def test_rough_box_uses_unchanged_actual_sdf_helper(self):
        original = original_helper(SP._prim)
        actual, expected, displaced = [], [], []
        scene = SP.Scene()
        scene.begin()
        scene.box((-.17, .08, -.11), (.21, .12, .18), yaw=.31, pitch=-.17,
                  rnd=.025, mat=6, rough=.017, rough_f=13.)
        scene.end()
        row = scene.arrays()[0][0]
        sample_points = np.vstack((self.xyz[:12], self.xyz[-32:] * .000002))
        for xyz in sample_points:
            value = SP._prim(*xyz, row)
            actual.append(value)
            expected.append(original(*xyz, row))
            # The rough branch halves its signed distance. This control shows
            # the selected points actually exercise nonzero displacement.
            displaced.append(value - .5 * SP._sd_box(*xyz, row))
        self.assertGreater(np.max(np.abs(displaced)), 0.)
        # Measured on Numba 0.62.1/macOS arm64, the call boundary changes one of
        # these 44 composed distances by one float64 ULP (1.735e-18 m).
        # This fixture fence is not permission for a changed trace/hit mask.
        self.assert_samples('rough-box actual _prim', actual, expected, maxulp=1)

    def test_horn_uses_unchanged_actual_helper(self):
        original = original_helper(SP._horn)
        actual, expected = [], []
        # Near and away from the flame centre, including negative world positions.
        for centre in ((0., .15, 0.), (-1004., 237., -508.)):
            for offset in self.xyz[-32:] * .000001:
                xyz = np.asarray(centre) + offset
                args = (*xyz, *centre)
                actual.append(SP._horn(*args))
                expected.append(original(*args))
        self.assertGreater(np.ptp(expected), 0.)
        # The raw samples remain exact, but this composed, fastmath helper
        # differs by up to three float64 ULPs (1.666e-16 here). Pin that observed
        # limit, emit every raw difference, and leave frame equivalence to the
        # separate renderer comparison; do not turn this into a broad allclose.
        self.assert_samples('horn actual _horn', actual, expected, maxulp=3)

    def test_actual_cloth_and_stone_noise_sample_expressions(self):
        candidate, calls = render_noise_calls(SP.gnoise3)
        original, _ = render_noise_calls(SHARED_NOISE)
        self.assertEqual({call.args[3].value for call in calls}, set(range(11, 16)) | set(range(31, 36)))
        actual = [candidate(*xyz) for xyz in self.xyz]
        expected = [original(*xyz) for xyz in self.xyz]
        self.assert_samples('actual render cloth/stone noise calls', actual, expected)

    def test_helper_ulp_budgets_reject_a_changed_noise_function(self):
        # The helper budgets must reject an actual noise change, even one far
        # below an 8-bit display code. These probes do not mutate global modules.
        horn_changed = original_helper(SP._horn, noise=biased_noise)
        args = (.071, -.113, .039, 0., .15, 0.)
        with self.assertRaises(AssertionError):
            np.testing.assert_array_max_ulp(SP._horn(*args), horn_changed(*args), maxulp=3)
        prim_changed = original_helper(SP._prim, noise=biased_noise)
        row = np.zeros(SP.NP)
        row[0], row[9] = 1., 6.
        row[11:14] = (.21, .12, .18)
        row[14:16] = (.017, 13.)
        args = (.07, -.11, .04, row)
        with self.assertRaises(AssertionError):
            np.testing.assert_array_max_ulp(SP._prim(*args), prim_changed(*args), maxulp=1)


if __name__ == '__main__':
    unittest.main()
