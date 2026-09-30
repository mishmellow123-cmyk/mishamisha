"""Pin inherited hot capped output when both D-only hooks are disabled."""
import hashlib
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / 'shots/embers'), str(ROOT / 'lib')]
import ringsolid as R
from core import Camera


class DefaultCapCompatibility(unittest.TestCase):
    def test_hot_caps_and_leading_glyph_match_pre_d_bytes(self):
        # Measured against 2d0c9bc2dcafdde1db21fe333e41cacd0fdab9ec's renderer
        # in the prescribed validation environment (NumPy 2.3.5/OpenCV 5.0.0).
        # This extends the existing uncapped/full-ring pins to hot solid caps.
        expected = ('0c04e85d35f1f3a3ef2ad0ed6a53448c1002c1c930772d6d3689856796e70f79',
                    'ffcc756d2c8dcf2b98c6565e74e3df536c0ef5a2e4adb2fe773ec9c2da482e8c',
                    '0c2cd256505b5293f4de312007a3e927e4ffbb1fd82513416d7dc7976ca2866d')
        cam = Camera((5., 2., -2.), (0., 0., 0.), hfov=60.)
        st = R.RingState()
        st.end_caps = st.leading_glyph = True
        st.glow, st.letters = .11, .8
        st.heat = lambda th: .75 + .25 * np.sin(th) ** 2
        result = R.render(cam, 128, 80, np.eye(3), np.zeros(3), 1., st, R.Env(),
                          ss=1, nt=64, npp=20, th_range=(.1, 5.4))
        def oracle(arrays):
            self.assertEqual(tuple(hashlib.sha256(a.tobytes()).hexdigest() for a in arrays), expected)
        oracle(result)
        changed = [a.copy() for a in result]
        changed[0][0, 0, 0] = np.nextafter(changed[0][0, 0, 0], np.float32(1.))
        with self.assertRaises(AssertionError):
            oracle(changed)
