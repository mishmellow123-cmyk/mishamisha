"""LD_SURGE_EASE (A7 THE EDGE re-render): the eased tower surge and the accepted one, without images, geometry or JIT.

scene_b imports the renderer's modules, so its surge function and constants are read out of the source by AST and run
alone; ease_out_back comes from core the same way. Each contract is also run against what it replaces, which must fail.
"""
import ast
import math
from pathlib import Path
import unittest

import numpy as np

HERE = Path(__file__).resolve().parents[1]
SCENE_B = (HERE / 'scene_b.py').read_text()
CORE = (HERE / 'core.py').read_text()


def _load(src, names):
    """the named top-level functions/assignments of a module source, executed in one namespace"""
    tree = ast.parse(src)
    keep = [n for n in tree.body
            if (isinstance(n, ast.FunctionDef) and n.name in names)
            or (isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id in names for t in n.targets))
            or (isinstance(n, ast.Assign) and any(isinstance(t, ast.Tuple) and any(e.id in names for e in t.elts)
                                                  for t in n.targets))]
    ns = dict(math=math, np=np)
    exec(compile(ast.Module(body=keep, type_ignores=[]), 'extract', 'exec'), ns)
    return ns


EASE = _load(SCENE_B, {'surge_ease', 'SURGE_PEAK', 'SURGE_OVER', 'SURGE_SETTLE'})
PULSE = _load(SCENE_B, {'pulse_attack', 'tower_pulse', 'PULSE_ATTACK', 'BEATS'})
CORE_NS = _load(CORE, {'clamp01', 'ease_out_back'})


def accepted(d):
    """the accepted per-beat share: ease_out_back((t - beat - delay) / 5, 1.6), zero before the onset"""
    x = d / 5.0
    return float(CORE_NS['ease_out_back'](x, 1.6)) if x > 0 else 0.0


def eased(d):
    return EASE['surge_ease'](d)


def grid(f, lo=-2.0, hi=12.0, step=1 / 64):
    n = int(round((hi - lo) / step))
    return [(lo + k * step, f(lo + k * step)) for k in range(n + 1)]


def first_frame(f):
    return f(1.0)


def worst_step(f, span):
    """the largest rise of the share over any interval of `span` frames (a one-frame step or a shutter's sweep)"""
    pts = grid(f)
    step = pts[1][0] - pts[0][0]
    k = int(round(span / step))
    return max(pts[i + k][1] - pts[i][1] for i in range(len(pts) - k))


class SurgeEase(unittest.TestCase):
    def check_soft_attack(self, f):
        self.assertLessEqual(first_frame(f), 0.25)
        self.assertLessEqual(worst_step(f, 1.0), 0.45)
        self.assertLessEqual(worst_step(f, 0.5), 0.23)

    def check_same_jump_and_timing(self, f):
        pts = grid(f)
        self.assertEqual(f(0.0), 0.0)
        self.assertEqual(f(-1.0), 0.0)
        for d in (7.0, 8.0, 20.0, 400.0):
            self.assertAlmostEqual(f(d), 1.0, places=12)       # the whole jump J, held exactly
        peak_d, peak = max(pts, key=lambda p: p[1])
        base_d = max(grid(accepted), key=lambda p: p[1])[0]
        self.assertLessEqual(abs(peak_d - base_d), 1.0)       # the highest frame is the same one or the next
        self.assertTrue(1.0 < peak <= 1.10)                   # a small overshoot, never more than the accepted's

    def check_continuous(self, f):
        pts = grid(f)
        self.assertLess(max(abs(b[1] - a[1]) for a, b in zip(pts, pts[1:])), 0.02)

    def test_eased_surge(self):
        self.check_soft_attack(eased)
        self.check_same_jump_and_timing(eased)
        self.check_continuous(eased)
        self.assertLess(eased(0.01) / 0.01, 0.05)             # zero slope at the onset (a straight ramp: 0.29)

    def test_accepted_surge_is_the_lurch_being_replaced(self):
        self.assertGreater(first_frame(accepted), 0.6)        # 69% of the jump in the first frame
        with self.assertRaises(AssertionError):
            self.check_soft_attack(accepted)

    def test_a_linear_ramp_fails_the_onset(self):
        ramp = lambda d: 0.0 if d <= 0 else min(1.0, d / 3.5)
        self.assertGreater(ramp(0.01) / 0.01, 0.05)
        with self.assertRaises(AssertionError):
            self.check_same_jump_and_timing(ramp)             # no overshoot: the surge would lose its bounce

    def test_off_is_the_accepted_expression(self):
        """the flag is read once at import, off by default, and off takes the accepted expression verbatim"""
        self.assertIn("SURGE_EASE = os.environ.get('LD_SURGE_EASE', '0') == '1'", SCENE_B)
        self.assertIn("(surge_ease(t - tb - self.dly[i]) if SURGE_EASE else float(ease_out_back(x, 1.6)))",
                      SCENE_B)
        self.assertEqual(SCENE_B.count('surge_ease('), 2)    # its definition and the one call in Towers.height


def accepted_pulse(x):
    """a3.A3Sched.beat_pulse relative to a beat: 1 on the onset frame, exp(-x/3) for 0 <= x < 20"""
    return math.exp(-x / 3.0) if 0 <= x < 20 else 0.0


class TowerPulse(unittest.TestCase):
    """LD_TOWER_PULSE_EASE: a tower's flash and heat band rise over PULSE_ATTACK frames, then decay as accepted"""
    tb = 640.0                                                # the first of scene_b's own BEATS

    def eased(self, x):
        return PULSE['tower_pulse'](self.tb + x)

    def check_soft_onset(self, f):
        self.assertEqual(f(-0.5), 0.0)
        self.assertLessEqual(f(1.0), 0.5)                     # at most half the flash one frame in (accepted: all)
        self.assertLess(f(0.01), 0.01)

    def test_eased_flash(self):
        A = PULSE['PULSE_ATTACK']
        self.check_soft_onset(self.eased)
        self.assertAlmostEqual(self.eased(A), 1.0, places=12)          # the same brightness, A frames on
        for x in (3.0, 5.0, 9.5, 19.0):
            self.assertAlmostEqual(self.eased(x), math.exp(-(x - A) / 3.0), places=12)   # the accepted decay
        self.assertEqual(PULSE['pulse_attack'](0.0), 0.0)
        self.assertEqual(PULSE['pulse_attack'](A), 1.0)
        self.assertAlmostEqual(PULSE['pulse_attack'](A / 2), 0.5, places=12)

    def test_accepted_flash_is_the_one_frame_step(self):
        self.assertEqual(accepted_pulse(0.0), 1.0)
        with self.assertRaises(AssertionError):
            self.check_soft_onset(accepted_pulse)

    def test_off_is_the_accepted_expression(self):
        self.assertIn("TOWER_PULSE_EASE = os.environ.get('LD_TOWER_PULSE_EASE', '0') == '1'", SCENE_B)
        self.assertIn("bp = tower_pulse(t - self.dly[i]) if TOWER_PULSE_EASE else beat_pulse(t - self.dly[i])",
                      SCENE_B)
        self.assertIn("w += (pulse_attack(x) if TOWER_PULSE_EASE else 1.0) * math.exp(-x / 11.0)", SCENE_B)
        self.assertEqual(SCENE_B.count('tower_pulse('), 2)   # its definition and the one call in the tower light


if __name__ == '__main__':
    unittest.main()
