"""Schedule/routing checks without constructing renderer geometry or JIT."""
import sys
from pathlib import Path
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import c_v5_cold_leadin as C


class ColdLeadIn(unittest.TestCase):
    def scene(self, variant):
        with patch.object(C.base.Scene, '__init__', return_value=None):
            return C.Scene(variant)

    def test_accepted_rejects_preexisting_map_frames(self):
        scene = self.scene('accepted')
        for f in (3815, 3816, 3839):
            with self.assertRaises(ValueError):
                scene.frame(f)

    def test_extension_is_scoped_and_bounds_first_shutter(self):
        original = C.base.SHOTS
        observed = []
        def intercept(scene, frame, scale):
            name = C.base.shot_at(frame)
            observed.append((frame, name, C.base.SHOTS[name], scale))
            return object()
        scene = self.scene('lead24')
        with patch.object(C.base.Scene, 'frame', intercept):
            for f in (3816, 3839):
                scene.frame(f, .5)
        self.assertEqual(observed, [(3816, 'cold', (3816, 4000), .5),
                                    (3839, 'cold', (3816, 4000), .5)])
        self.assertIs(C.base.SHOTS, original)
        with self.assertRaises(ValueError):
            C.base.shot_at(3839)

    def test_exception_restores_and_releases(self):
        original = C.base.SHOTS
        with patch.object(C.base.Scene, 'frame', side_effect=RuntimeError('probe')):
            with self.assertRaises(RuntimeError):
                self.scene('lead24').frame(3824)
        self.assertIs(C.base.SHOTS, original)
        self.assertFalse(C._LOCK.locked())

    def test_accepted_domain_never_changes_for_either_variant(self):
        original = C.base.SHOTS
        def intercept(scene, frame, scale):
            self.assertIs(C.base.SHOTS, original)
            return frame, scale
        with patch.object(C.base.Scene, 'frame', intercept):
            for variant in C.VARIANTS:
                scene = self.scene(variant)
                for f in (2320, 2639, 3840, 3847, 3848, 3999, 4000, 4239):
                    self.assertEqual(scene.frame(f, .5), (f, .5))

    def test_leadin_levels_match_original_lit_state_and_shutdown(self):
        # Checks the actual inherited schedule, not a new approximation.
        for i in range(28):
            for f in (3816, 3824, 3839, 3840, 3847.999):
                self.assertEqual(C.base.forge_level(i, f), C.base.forge_level(i, 3840))
                self.assertGreater(C.base.forge_level(i, f), 0)
            self.assertEqual(C.base.forge_level(i, 3848), 0)

    def test_invalid_variant_fails_before_geometry(self):
        with patch.object(C.base.Scene, '__init__') as build:
            with self.assertRaises(ValueError):
                C.Scene('unknown')
            build.assert_not_called()


if __name__ == '__main__':
    unittest.main()
