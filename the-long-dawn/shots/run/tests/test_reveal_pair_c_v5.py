"""Actual shared-world checks for the paired-reveal amendment."""
from pathlib import Path
import sys
import unittest
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import reveal_pair_c_v5 as pair
import render_ink as RI


class PairedReveal(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.shot=pair.make_shot()

    def test_cut_has240_frames_and_uses_only_original_wide_camera_poses(self):
        self.assertEqual([pair.local_frame(f) for f in range(2880,3120)],list(range(240)))
        self.assertEqual(pair.camera_frame(0),160.)
        self.assertEqual(pair.camera_frame(239),239.)
        for f in (2879,3120):
            with self.assertRaises(ValueError):pair.local_frame(f)

    def test_both_actual_fire_envelopes_catch_at_the_same_frame(self):
        B=self.shot.B
        self.assertEqual(B.shape,(2,7))
        np.testing.assert_array_equal(B[:,3],[0.,0.])
        for f in (-2.,0.,1.,5.,23.,239.):
            np.testing.assert_array_equal(RI.BC.env(f,B[0,3]),RI.BC.env(f,B[1,3]))
        self.assertEqual(RI.BC.env(-2.,0.),(0.,0.,0.))
        self.assertGreater(RI.BC.env(0.,0.)[0],0.)

    def test_near_original_is_unchanged_and_far_sits_on_catalogue_peak(self):
        original=RI.Shot('reveal')
        self.assertEqual(original.B[0,3],-400.)
        np.testing.assert_array_equal(self.shot.B[0,:3],original.B[0,:3])
        cat=np.load(Path(pair.__file__).with_name('summits.npy'))
        far=self.shot.B[1,:3]
        self.assertLess(np.linalg.norm(far[[0,2]]-cat[pair.FAR_SUMMIT,[0,2]]),64.)
        self.assertGreater(np.linalg.norm(far-self.shot.B[0,:3]),5000.)

    def test_sites_are_separate_inside_frame_and_visible_in_real_terrain(self):
        sites=self.shot.B[:,:3]
        for f in (0,40,80,120,160,200,239):
            cam=pair.camera(f)
            u,v,z=cam.project(sites+np.array([0.,1.6,0.]))
            self.assertTrue(np.all(z>0.))
            self.assertTrue(np.all((u>192)&(u<1728)&(v>80)&(v<764)))
            self.assertGreater(abs(u[0]-u[1]),1920*.17)
            for site in sites:
                self.assertTrue(RI.BC._visible(cam,site+np.array([0.,2.5,0.]),self.shot.CR,f/24.))


if __name__=='__main__':unittest.main()
