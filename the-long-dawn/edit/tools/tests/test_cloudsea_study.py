"""Lightweight study-driver tests; no renderer import or JIT."""
from contextlib import redirect_stderr
import importlib.util
import hashlib
import io
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock

import cv2
import numpy as np

SOURCE = Path(__file__).resolve().parents[1] / 'cloudsea_study.py'
SPEC = importlib.util.spec_from_file_location('cloudsea_study', SOURCE)
STUDY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(STUDY)


class StudyTests(unittest.TestCase):
    def test_cli_is_off_by_default_and_uses_cut_numbers(self):
        args = STUDY.parse_args(['--shot', 'A2', '--frames', '380,500', '--out', '/tmp/relief-study-test'])
        self.assertEqual(args.relief, 0.0)
        self.assertEqual(args.frames, [380, 500])
        self.assertEqual(args.scale, 0.5)
        self.assertEqual(args.threads, 1)

    def test_cli_rejects_ambiguous_or_invalid_inputs(self):
        base = ['--shot', 'A13', '--frames', '3720', '--out', '/tmp/relief-study-test']
        for override in (['--relief', 'nan'], ['--relief', '-1'], ['--relief', '1.1'],
                         ['--out', 'relative'], ['--frames', '1596'], ['--frames', '3720,3720'],
                         ['--frames', ''], ['--threads', '5'], ['--scale', 'inf']):
            with self.subTest(override=override), redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    STUDY.parse_args(base + override)

    def test_all_entrypoints_receive_copied_runtime_flag_and_restore(self):
        received = []

        def march(p, other):
            received.append(('march', p.copy(), other))

        def shade(a, b, P):
            received.append(('shade', P.copy(), a))

        def glow(a, b, P):
            received.append(('cloud_glow', P.copy(), a))

        world = types.SimpleNamespace(march=march, shade=shade, cloud_glow=glow)
        p = np.array([1., 2., 3., 0.])
        marker = object()
        with STUDY.relief_inputs(world, 0.75) as calls:
            world.march(p, marker)
            world.shade(marker, None, p)
            world.cloud_glow(marker, None, P=p)
        np.testing.assert_array_equal(p, [1., 2., 3., 0.])
        for name, got, unchanged in received:
            np.testing.assert_array_equal(got, [1., 2., 3., 0.75])
            self.assertIs(unchanged, marker)
            self.assertEqual(calls[name], [[1., 2., 3., 0.75]])
        self.assertIs(world.march, march)
        self.assertIs(world.shade, shade)
        self.assertIs(world.cloud_glow, glow)

    def test_entrypoints_restore_after_failure(self):
        def fail(*args):
            raise RuntimeError('injected renderer failure')

        world = types.SimpleNamespace(march=fail, shade=fail, cloud_glow=fail)
        with self.assertRaisesRegex(RuntimeError, 'injected renderer failure'):
            with STUDY.relief_inputs(world, 1.0):
                world.march(np.zeros(4))
        self.assertTrue(all(getattr(world, name) is fail for name in ('march', 'shade', 'cloud_glow')))

    def test_depth_hash_records_completed_march_without_changing_parameter_log(self):
        def march(P, CR, C, d0, dmax, rel, kgap, hmax, nbis, out_d):
            out_d[:] = [[3.0, 7.0], [11.0, 1e30]]  # synthetic completed depth buffer

        world = types.SimpleNamespace(march=march, shade=lambda *a: None, cloud_glow=lambda *a: None)
        depth = np.zeros((2, 2))
        initial_hash = hashlib.sha256(depth.tobytes()).hexdigest()
        records = []
        with STUDY.relief_inputs(world, 1.0, records) as calls:
            world.march(np.zeros(4), None, None, 0., 0., 0., 0., 0., 0, out_d=depth)
        self.assertEqual(calls, {'march': [[0., 0., 0., 1.]], 'shade': [], 'cloud_glow': []})
        self.assertEqual(records, [{'shape': [2, 2], 'dtype': 'float64',
                                    'sha256': hashlib.sha256(depth.tobytes()).hexdigest()}])
        self.assertNotEqual(records[0]['sha256'], initial_hash)
        np.testing.assert_array_equal(depth, [[3., 7.], [11., 1e30]])

    def test_adapter_a2_maps_frame_and_finishes_once(self):
        hdr, finished, world = object(), object(), object()
        module = types.SimpleNamespace(CUT0=80, _SKL=object(), FINISH={'exposure': 1.05},
            render=mock.Mock(return_value=hdr),
            PI=types.SimpleNamespace(WD=world, look=types.SimpleNamespace(finish=mock.Mock(return_value=finished))))
        with mock.patch.dict(sys.modules, falsedawn=module), mock.patch.object(sys, 'path', sys.path.copy()):
            actual, render, _ = STUDY.shot_adapter('A2')
            self.assertIs(actual, world)
            self.assertIs(render(380, 0.5), finished)
        module.render.assert_called_once_with(300, 'arc', 0.5, 1.5)
        module.PI.look.finish.assert_called_once_with(hdr, exposure=1.05)
        self.assertIsNone(module._SKL)

    def test_adapter_a13_uses_generated_world_and_returns_full_finished_render(self):
        finished, world = object(), object()
        table = np.array([[100., 10., 200., 3700., 1., 7.]])  # synthetic x,y,z,ignition,size,seed
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'reveal_a_fires.npy'
            np.save(path, table)
            module = types.SimpleNamespace(_orig_s1_world=lambda: {'WD': world}, FIRES_NPY=str(path),
                NA=types.SimpleNamespace(_SKL={'old': 'cache'}), render=mock.Mock(return_value=finished))
            with mock.patch.dict(sys.modules, reveal_a=module), mock.patch.object(sys, 'path', sys.path.copy()):
                with mock.patch.object(STUDY, 'RUN', Path(directory)):
                    actual, render, _ = STUDY.shot_adapter('A13')
                    self.assertIs(actual, world)
                    self.assertIs(render(3720, 0.5), finished)
        module.render.assert_called_once_with(3720, 0.5)
        self.assertEqual(module.NA._SKL, {})
        np.testing.assert_array_equal(module._FIRES, table)

    def test_a13_missing_fire_table_fails_before_renderer_setup(self):
        module = types.SimpleNamespace(_orig_s1_world=mock.Mock(), render=mock.Mock())
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch.dict(sys.modules, reveal_a=module), mock.patch.object(sys, 'path', sys.path.copy()):
                with mock.patch.object(STUDY, 'RUN', Path(directory)):
                    with self.assertRaisesRegex(FileNotFoundError, 'requires the existing fire table'):
                        STUDY.shot_adapter('A13')
        module._orig_s1_world.assert_not_called()
        module.render.assert_not_called()

    def test_a13_fire_table_must_have_usable_real_finite_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'fires.npy'
            for table in (np.zeros((0, 6)), np.zeros((2, 5)), np.full((1, 6), np.nan),
                          np.full((1, 6), np.inf), np.zeros((1, 6), complex), np.full((1, 6), 'text')):
                with self.subTest(shape=table.shape, dtype=table.dtype):
                    np.save(path, table)
                    with self.assertRaisesRegex(ValueError, 'nonempty, finite, real'):
                        STUDY.validate_a13_fires(path)

    def test_a13_source_record_includes_fire_table_hash_only_for_a13(self):
        world = types.SimpleNamespace(__file__=str(STUDY.RUN / 'world.py'))
        with mock.patch.object(STUDY.subprocess, 'run', return_value=types.SimpleNamespace(stdout='fixture-revision\n')):
            a13 = STUDY.source_record(world, 'A13')['source_sha256']
            a2 = STUDY.source_record(world, 'A2')['source_sha256']
        role = 'shots/run/reveal_a_fires.npy'
        self.assertEqual(a13[role], hashlib.sha256((STUDY.RUN / 'reveal_a_fires.npy').read_bytes()).hexdigest())
        self.assertNotIn(role, a2)

    def test_png_is_deterministic_and_decodes_to_expected_rgb(self):
        rgb = np.array([[[0., 0.5, 1.], [0.1, 0.2, 0.3]], [[1., 0., 0.], [0., 1., 0.]]], np.float32)
        with tempfile.TemporaryDirectory() as directory:
            a, b = Path(directory) / 'a.png', Path(directory) / 'b.png'
            expected = STUDY.save_png(a, rgb)
            STUDY.save_png(b, rgb)
            self.assertEqual(a.read_bytes(), b.read_bytes())
            np.testing.assert_array_equal(cv2.imread(str(a))[..., ::-1], expected)
            np.testing.assert_array_equal(expected[1, 0], [255, 0, 0])

    def test_failed_encode_preserves_destination(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'old.png'
            path.write_bytes(b'prior destination')
            with mock.patch.object(cv2, 'imencode', return_value=(False, None)):
                with self.assertRaisesRegex(OSError, 'PNG encoding failed'):
                    STUDY.save_png(path, np.zeros((2, 2, 3)))
            self.assertEqual(path.read_bytes(), b'prior destination')
            self.assertEqual(list(Path(directory).iterdir()), [path])


if __name__ == '__main__':
    unittest.main()
