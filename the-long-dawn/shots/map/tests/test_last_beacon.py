"""The shot's editorial contract; no world bake or renderer construction."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

SOURCE = Path(__file__).resolve().parents[1] / 'last_beacon.py'
SPEC = importlib.util.spec_from_file_location('last_beacon', SOURCE)
shot = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(shot)


class LastBeaconContract(unittest.TestCase):
    def test_exact_shot_and_holds(self):
        self.assertEqual(shot.F1-shot.F0, 400)
        self.assertEqual(shot.ONE_DARK[1]-shot.ONE_DARK[0], 60)
        self.assertEqual(shot.CAPTION_HOLD[1]-shot.CAPTION_HOLD[0], 48)
        for f in range(*shot.ONE_DARK):
            catch = shot.catch_at(f)
            self.assertEqual(catch[shot.LAST], 0)
            np.testing.assert_array_equal(np.delete(catch, shot.LAST), 1)
        self.assertEqual(shot.catch_at(shot.ONE_DARK[0]-1).min(), 0)
        self.assertLess(np.sort(shot.catch_at(shot.ONE_DARK[0]-1))[1], 1)
        for f in range(*shot.CAPTION_HOLD):
            np.testing.assert_array_equal(shot.catch_at(f), 1)

    def test_last_catches_only_after_silence(self):
        self.assertEqual(shot.IGNITION[shot.LAST], shot.ONE_DARK[1])
        self.assertEqual(shot.IGNITION[shot.LAST]+shot.CATCH, shot.CAPTION_HOLD[0])
        values = [shot.catch_at(f)[shot.LAST] for f in range(3784, 3793)]
        self.assertTrue(all(a < b for a, b in zip(values, values[1:])))

    def test_eight_exclusive_kingdoms_and_irregular_order(self):
        p = shot.BEACONS
        np.testing.assert_array_equal(shot.kingdom_at(p[:, 0], p[:, 1]), np.arange(8))
        order = np.argsort(shot.IGNITION)
        distance = np.linalg.norm(p[order]-p[0], axis=1)
        self.assertTrue(np.any(np.diff(distance) > 0))
        self.assertTrue(np.any(np.diff(distance) < 0))
        self.assertTrue(np.any(np.diff(p[order, 0]) > 0))
        self.assertTrue(np.any(np.diff(p[order, 0]) < 0))

    def test_all_beacons_visible_and_camera_settled_for_holds(self):
        points = np.column_stack([shot.BEACONS, np.zeros(8)])
        for f in range(shot.F0, shot.F1):
            cam = shot.camera_at(f)
            uv, z = cam.project(points)
            self.assertTrue(np.all(z > 0))
            self.assertTrue(np.all((uv[:, 0] > 80) & (uv[:, 0] < 1840)))
            self.assertTrue(np.all((uv[:, 1] > 60) & (uv[:, 1] < 740)))
        np.testing.assert_array_equal(shot.camera_at(3724).Hm, shot.camera_at(3839).Hm)

    def test_atlas_covers_every_frame(self):
        s = shot.atlas_spec()
        for f in range(shot.F0, shot.F1):
            x, y = shot.camera_at(f).screen_to_map(np.array([0, 1920, 1920, 0]),
                                                 np.array([0, 0, 804, 804]))
            self.assertGreater(x.min(), s['x0']+1)
            self.assertLess(x.max(), s['x0']+s['width']/s['ppd']-1)
            self.assertGreater(y.min(), s['y1']-s['height']/s['ppd']+1)
            self.assertLess(y.max(), s['y1']-1)

    def test_only_c_frame_numbers_and_source_pinned_atlas(self):
        self.assertEqual(shot.parse_frames('3440-3442,3839'), [3440, 3441, 3442, 3839])
        for bad in ('0', '3439', '3840', '3442-3440'):
            with self.assertRaises(ValueError):
                shot.parse_frames(bad)
        manifest = shot.source_manifest()
        self.assertEqual(manifest['world'], 5)
        self.assertTrue(all(len(v) == 64 for v in manifest['sources'].values()))

    def test_completed_cache_rejects_missing_corrupt_or_wrong_source_data(self):
        manifest = dict(atlas=dict(width=96, height=96), sources={'terrain': 'a'})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path/'manifest.json').write_text(json.dumps(manifest))
            with self.assertRaisesRegex(RuntimeError, 'Incomplete or corrupt'):
                shot.validate_atlas(path, manifest)
            for L in range(6):
                np.save(path/f'L{L}.npy', np.zeros((96//2**L, 96//2**L, 3), np.uint8))
            np.save(path/'relief.npy', np.zeros((8, 8), np.float32))
            shot.validate_atlas(path, manifest)
            np.save(path/'L2.npy', np.zeros((24, 24, 3), np.float32))
            with self.assertRaisesRegex(RuntimeError, 'unexpected shape or dtype'):
                shot.validate_atlas(path, manifest)
            with self.assertRaisesRegex(RuntimeError, 'source manifest mismatch'):
                shot.validate_atlas(path, dict(manifest, sources={'terrain': 'b'}))


if __name__ == '__main__':
    unittest.main()
