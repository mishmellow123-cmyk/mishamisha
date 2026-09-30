"""A2's opt-in high-deck removal; no terrain shading or full frames rendered."""
from pathlib import Path
import sys
import unittest
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'shots/run'))
import falsedawn as FD


class SkyReached(Exception):
    pass


class FalseDawnSkyTests(unittest.TestCase):
    def test_parameter_isolation_and_accepted_default(self):
        _, params = FD.light(132, 'arc')
        before = params.copy()
        self.assertIs(FD.sky_parameters(params), params)
        self.assertIs(FD.sky_parameters(params, 'accepted'), params)
        actual = FD.sky_parameters(params, 'clear_high_deck')
        expected = params.copy()
        expected[16] = 0.
        self.assertGreater(before[16], 0.)
        self.assertFalse(np.shares_memory(actual, params))
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(params, before)
        with self.assertRaisesRegex(ValueError, 'Unknown false-dawn sky candidate'):
            FD.sky_parameters(params, 'typo')

    def test_renderer_routes_sky_without_changing_terrain_light(self):
        original_light, original_gp = FD.light(132, 'arc')
        def terrain(fr, *args, **kwargs):
            fr.dist = np.full((fr.src.H, fr.src.W), 1.e9, np.float32)
            fr.img = np.zeros((*fr.dist.shape, 3), np.float32)
        for candidate in (None, 'accepted', 'clear_high_deck'):
            kwargs = {} if candidate is None else {'sky_candidate': candidate}
            with self.subTest(candidate=candidate), \
                 mock.patch.object(FD.PI, 'render_terrain', side_effect=terrain) as ground, \
                 mock.patch.object(FD, 'skyline', return_value=(0.,1.,np.zeros(3))), \
                 mock.patch.object(FD, 'glow_haze'), \
                 mock.patch.object(FD, 'sky_pass', side_effect=SkyReached) as sky:
                with self.assertRaises(SkyReached):
                    FD.render(132, scale=.1, **kwargs)
                for actual, expected in zip(ground.call_args.args[3], original_light):
                    np.testing.assert_array_equal(actual, expected)
                actual_gp = sky.call_args.args[4]
                expected_gp = original_gp.copy()
                if candidate == 'clear_high_deck':
                    expected_gp[16] = 0.
                np.testing.assert_array_equal(actual_gp, expected_gp)
        with mock.patch.object(FD.PI, 'render_terrain') as terrain:
            with self.assertRaises(ValueError):
                FD.render(132, scale=.1, sky_candidate='typo')
            terrain.assert_not_called()

    def test_actual_sky_kernel_clears_deck_and_preserves_terrain(self):
        # Real native camera focal length, bounded source tile containing the
        # upper-left cloud bank at A212. No terrain marcher or whole frame.
        source = FD.PI.Frame(FD.camera(132), 1.5).src
        camera = source.params()
        camera[8] -= 20
        camera[9] -= 250
        shape = (180, 360)
        dist = np.full(shape, 1.e9, np.float32)
        dist[-5:, :] = 100.
        _, params = FD.light(132, 'arc')
        def draw(gp):
            img = np.full((*shape, 3), .123, np.float32)
            trans = np.zeros(shape, np.float32)
            FD.sky_pass(img, dist, trans, camera, gp, FD.milky_way(),
                        0., 1., np.zeros(3), source.pos[1], 132/24., np.zeros((0,6)))
            return img, trans
        before, bt = draw(params)
        after, at = draw(FD.sky_parameters(params, 'clear_high_deck'))
        manual = params.copy()
        manual[16] = 0.
        expected, et = draw(manual)
        np.testing.assert_array_equal(after, expected)
        np.testing.assert_array_equal(at, et)
        self.assertTrue(np.isfinite(after).all())
        self.assertTrue(np.all((at >= 0.) & (at <= 1.)))
        self.assertGreater(np.max(np.abs(before-after)), .001)
        self.assertGreater(np.max(at-bt), .1)
        np.testing.assert_array_equal(after[-5:], before[-5:])
        np.testing.assert_array_equal(after[-5:], np.full((5,360,3),.123,np.float32))
        self.assertTrue(np.all(at[-5:] == 0.))
        # Repeat deterministically: no per-frame random patch over the deck.
        repeated, rt = draw(FD.sky_parameters(params, 'clear_high_deck'))
        np.testing.assert_array_equal(after, repeated)
        np.testing.assert_array_equal(at, rt)

    def test_opening_cut_and_floor_ramp_stay_locked(self):
        sys.path.insert(0, str(ROOT/'edit'))
        import edl_v3 as EDL
        first, second = EDL.EDL['A'][:2]
        self.assertEqual((first['f0'],first['f1'],first['kind']), (0,80,'black'))
        self.assertEqual((second['f0'],second['f1']), (80,560))
        self.assertEqual(second['takes'][0]['stem'], 'falsedawn')
        floor = [t for t in EDL.TRANS['A'] if t['kind']=='floor' and t['f0']==80]
        self.assertEqual(len(floor),1)
        self.assertEqual({k:floor[0][k] for k in ('f0','f1','kind','k0','k1')},
                         dict(f0=80,f1=128,kind='floor',k0=1.,k1=0.))


if __name__ == '__main__':
    unittest.main()
