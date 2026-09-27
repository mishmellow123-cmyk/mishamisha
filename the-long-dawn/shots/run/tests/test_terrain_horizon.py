"""A-world horizon cutoff: real marcher on a narrow A15 strip, no shading/render.

Run: python -B -m unittest discover -s the-long-dawn/shots/run/tests -p 'test_*.py'
"""
import os
from pathlib import Path
import sys
import unittest
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import beaconrun_a as BR
import falsedawn as FD
import pipe as PI
import stars_a as ST
import watchers_a as WA
import world as WD


class MarchReached(Exception):
    pass


class TerrainHorizonTests(unittest.TestCase):
    def setUp(self):
        self.enterContext(mock.patch.dict(os.environ, A14_CAM='', W15_P7='', W15_CAM='', W15_OWN_CAM=''))
        self.enterContext(mock.patch.object(BR, '_PLATE', None))
        self.enterContext(mock.patch.dict(ST._ST, clear=True))

    def march_args(self, render):
        # Stop at the actual entrypoint's march call, before terrain JIT/shading.
        with mock.patch.object(WD, 'march', side_effect=MarchReached) as march:
            with self.assertRaises(MarchReached):
                render()
        self.assertEqual(march.call_count, 1)
        return list(march.call_args.args)

    def test_a15_farwall_survives_foreground_column_history(self):
        args = self.march_args(lambda: WA.render(4360, scale=0.5, ss=1.5))
        camera = WA.camera(4360, 960, 402)
        source = PI.Frame(camera, 1.5).src
        # Native output pixel (724,120) lies beyond the old vertical sky cut.
        direction = camera.ray(np.array([724.5]), np.array([120.5]))[0]
        forward = direction @ source.fwd
        column = int(source.cx + source.f * (direction @ source.right) / forward)
        row = int(source.cyy - source.f * direction[1] / forward)
        # Keep every source row: march reuses each preceding lower pixel's hit.
        # Cropping vertically would remove the foreground-to-farwall transition.
        args[2] = args[2].copy()
        args[2][8] -= column - 12
        args[2][10] = 25

        def depths(hmax):
            sample = list(args)
            sample[7] = hmax
            sample[9] = np.zeros((source.H, 25))
            WD.march(*sample)
            return sample[9]

        old = depths(700.0)                    # negative control: actual old cutoff
        fixed = depths(args[7])               # bound supplied by the real A15 render
        generous = depths(3000.0)             # independently rendered diagnostic
        self.assertGreaterEqual(old[row, 12], 1e29)
        self.assertGreater(fixed[row, 12], 45000.0)
        self.assertLess(fixed[row, 12], 50000.0)
        self.assertGreater(np.count_nonzero((old >= 1e29) & (fixed < 1e29)), 0)
        np.testing.assert_array_equal(fixed[old < 1e29], old[old < 1e29])
        np.testing.assert_array_equal(fixed, generous)

    def test_all_a_world_entrypoints_supply_the_scene_bound(self):
        renders = {
            'A2': lambda: FD.render(420, scale=0.05),
            'A11': lambda: ST.static(0.05, 1.5),
            'A14': lambda: BR.render(4200, scale=0.05),
            'A15': lambda: WA.render(4360, scale=0.05),
        }
        for shot, render in renders.items():
            with self.subTest(shot=shot):
                args = self.march_args(render)
                self.assertEqual(args[7], FD.terrain_hmax(args[1]))
                self.assertGreater(args[7], np.max(FD.FD_WALL[:, [2, 5]]))

    def test_pipeline_default_remains_700_for_other_worlds(self):
        frame = PI.Frame(WA.camera(4360, 96, 40), 1.0)
        args = self.march_args(lambda: PI.render_terrain(frame, 0, np.zeros((0, WD.NCR)), None, []))
        self.assertEqual(args[7], 700.0)

    def test_ceiling_tracks_changed_crest_heights_without_resampling(self):
        rows = FD.terrain_rows().copy()
        before = FD.terrain_hmax(rows)
        wall = len(rows) - len(FD.FD_WALL)
        rows[wall, [2, 5]] += 10000.0
        after = FD.terrain_hmax(rows)
        self.assertGreater(after, before)
        self.assertGreater(after, max(rows[wall, 2], rows[wall, 5]))


if __name__ == '__main__':
    unittest.main()
