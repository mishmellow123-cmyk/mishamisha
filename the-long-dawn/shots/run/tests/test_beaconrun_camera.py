"""A14 camera boundary: real projection/motion-blur code, no terrain or picture renders.

Run: python -B -m unittest discover -s the-long-dawn/shots/run/tests -p 'test_*.py'
"""
import math
import os
from pathlib import Path
import sys
import unittest
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import beaconrun_a as BR
import rcam as RC


def original_camera(f, W=1920, H=804):
    """The valid-time camera formula from 07d5dca, retained to pin the accepted motion."""
    pos1, yaw1, pitch1, hfov1 = BR.end_cam()
    fw = np.array([math.sin(math.radians(yaw1)), 0.0, math.cos(math.radians(yaw1))])
    pos0 = pos1 - fw * BR.BACK0 + np.array([0.0, BR.UP0, 0.0])
    e = BR._ease((f - BR.FR0) / float(BR.T_SETTLE - BR.FR0))
    yaw = BR.YAW0 + (yaw1 - BR.YAW0) * BR._ease(((f - BR.FR0) / float(BR.T_SETTLE - BR.FR0)) ** 1.2)
    eh = BR._ease(((f - BR.FR0) / float(BR.T_SETTLE - BR.FR0)) ** 1.15)
    pos = pos0 + (pos1 - pos0) * np.array([e, eh, e])
    pitch = BR.PITCH0 + (pitch1 - BR.PITCH0) * BR._ease(((f - BR.FR0) / float(BR.T_SETTLE - BR.FR0)) ** 1.3)
    hfov = BR.HFOV0 + (hfov1 - BR.HFOV0) * e
    return RC.RCam(pos, yaw, pitch, 0.0, hfov, W, H)


class BeaconRunCameraTests(unittest.TestCase):
    def setUp(self):
        self.enterContext(mock.patch.dict(os.environ, A14_CAM='', W15_P7='', W15_CAM='', W15_OWN_CAM=''))
        self.enterContext(mock.patch.object(BR, '_PLATE', None))

    def assert_same_camera(self, actual, expected):
        np.testing.assert_array_equal(actual.pos, expected.pos)
        np.testing.assert_array_equal(actual.fwd, expected.fwd)
        np.testing.assert_array_equal(actual.up, expected.up)
        np.testing.assert_array_equal(actual.right, expected.right)
        self.assertEqual((actual.yaw_d, actual.pitch_d, actual.hfov_d, actual.W, actual.H),
                         (expected.yaw_d, expected.pitch_d, expected.hfov_d, expected.W, expected.H))

    def test_pre_start_camera_holds_first_pose(self):
        first = BR.camera(BR.FR0)
        self.assert_same_camera(BR.camera(BR.FR0 - BR.SHUTTER / 2), first)

    def test_first_frame_velocity_samples_are_finite(self):
        # Exactly render()'s callback/shutter contract, with a tiny controlled depth plane.
        W, H = 2, 1
        velocity = RC.velocity(lambda ff: BR.camera(ff, W, H), BR.FR0, BR.camera(BR.FR0, W, H),
                               np.full((H, W), 1000.0), shutter=BR.SHUTTER)
        self.assertEqual(velocity.shape, (H, W, 2))
        self.assertTrue(np.isfinite(velocity).all())

    def test_all_in_range_poses_and_shutter_samples_are_unchanged(self):
        for f in range(BR.FR0, BR.FR0 + BR.NFR):
            for offset in (-BR.SHUTTER / 2, 0, BR.SHUTTER / 2):
                sample = f + offset
                if sample < BR.FR0:
                    continue
                with self.subTest(frame=sample):
                    self.assert_same_camera(BR.camera(sample), original_camera(sample))


if __name__ == '__main__':
    unittest.main()
