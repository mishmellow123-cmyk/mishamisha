"""C20's dawn: synthetic pixels and source metadata only; no delivered frames or LUTs are read."""
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

EDIT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EDIT))
import assemble as AS
import deliver as D
import edl_v3 as EDL


def plate(w=101):
    return np.broadcast_to(np.array([0.78, 0.42, 0.12], np.float32), (3, w, 3)).copy()


def changed_helper(img, f, t):
    """A distinct helper source for the cache-dependency test; never used to render a picture."""
    return img * 0.5


class DawnSweepTests(unittest.TestCase):
    def test_opening_has_the_measured_neutral_balance_and_preserves_detail(self):
        img = plate() * np.linspace(0.3, 1.0, 101, dtype=np.float32)[None, :, None]
        img[..., 2] *= np.linspace(1.0, 0.3, 101, dtype=np.float32)[None, :]
        tint = np.asarray(EDL.C_DAWN['night_rgb'], np.float32)
        for f in (4712, 4720, EDL.C_DAWN['start']):
            out = AS.dawn_sweep(img, f, EDL.C_DAWN)
            # Different input hues must converge on the measured night balance, without flattening the drawing.
            np.testing.assert_allclose(out[..., 0] / out[..., 1], tint[0] / tint[1], atol=2e-7)
            np.testing.assert_allclose(out[..., 2] / out[..., 1], tint[2] / tint[1], atol=2e-7)
            self.assertLess(float(out.mean()), float(img.mean()))
            self.assertGreater(float(out[0, -1, 1]), float(out[0, 0, 1]))
            self.assertEqual(out.dtype, np.float32)

    def test_done_and_the_final_hold_are_exactly_the_delivered_picture(self):
        img = plate()
        before = img.copy()
        for f in (EDL.C_DAWN['done'], 5199, 5200):
            np.testing.assert_array_equal(AS.dawn_sweep(img, f, EDL.C_DAWN), img)
        AS.dawn_sweep(img, 4944, EDL.C_DAWN)
        np.testing.assert_array_equal(img, before)

    def test_colour_front_advances_left_to_right_and_never_retreats(self):
        for w in (41, 101):
            img = plate(w)
            # This chroma axis is orthogonal to both the night tint and neutral white, so diffuse prelight
            # contributes nothing. Its projection therefore measures restored colour independently of luma.
            axis = np.cross(np.asarray(EDL.C_DAWN['night_rgb'], np.float32), np.ones(3, np.float32))
            axis /= np.linalg.norm(axis)
            frames = np.linspace(EDL.C_DAWN['start'], EDL.C_DAWN['done'], 17).astype(int)
            weights = []
            for f in frames:
                out = AS.dawn_sweep(img, int(f), EDL.C_DAWN)
                # Recover the colour mix from a constant plate; this tests the visible result, not the mask formula.
                weights.append((out[0] @ axis) / (img[0] @ axis))
                self.assertTrue(np.isfinite(out).all())
            weights = np.asarray(weights)
            self.assertGreaterEqual(float(weights.min()), -1e-6)
            self.assertLessEqual(float(weights.max()), 1.0 + 1e-6)
            self.assertTrue(np.all(np.diff(weights, axis=0) >= -1e-6), 'colour retreats in time')
            self.assertTrue(np.all(np.diff(weights, axis=1) <= 1e-6), 'colour arrives from the right')
            self.assertGreater(float(weights[8, 0] - weights[8, -1]), 0.9)
            self.assertAlmostEqual(float(weights[8, w // 2]), 0.5, places=6)

    def test_shot_finishes_once_before_the_grade_and_also_supports_no_finish(self):
        img, src = plate(), {'stem': 'runC_illum'}
        finished = img * np.float32(0.75) + np.float32(0.1)
        fin = mock.Mock(return_value=finished)
        for f in (4944, EDL.C_DAWN['done']):
            fin.reset_mock()
            out = AS._tk_dawn_sweep(img, src, f, EDL.C_DAWN, fin, 'C')
            fin.assert_called_once()
            self.assertIs(fin.call_args.args[0], img)
            self.assertEqual(fin.call_args.args[1:], (src, 'C', f))
            np.testing.assert_array_equal(out, AS.dawn_sweep(finished, f, EDL.C_DAWN))
        np.testing.assert_array_equal(AS._tk_dawn_sweep(img, src, 4944, EDL.C_DAWN, None, 'C'),
                                      AS.dawn_sweep(img, 4944, EDL.C_DAWN))

    def test_dispatcher_joins_windows_without_double_finish_or_changing_plenty(self):
        ctx = AS.Ctx.__new__(AS.Ctx)
        ctx.cut = 'C'
        raw_reads, finishes = [], []
        incoming = plate(7)
        outgoing = np.full_like(incoming, [0.34, 0.35, 0.33])

        def raw(_ctx, f):
            raw_reads.append(f)
            sec = 'C19' if f < 4720 else ('C20' if f < 5200 else 'C21')
            return (outgoing if sec == 'C19' else incoming).copy(), {'sec': sec}, sec, {'stem': sec}

        class Finisher:
            def __call__(self, img, src, cut, f):
                finishes.append(('source', src['stem'], f))
                return img + np.float32(0.03)

            def ink(self, img, cut, f):
                finishes.append(('ink', f))
                return img + np.float32(0.03)

            def film(self, img, cut, f):
                finishes.append(('film', f))
                return img * np.float32(0.8)

        fin = Finisher()

        def outer(f):
            img, shot, status, src = raw(ctx, f)
            return fin(img, src, 'C', f), shot, status, src

        ctx.picture = outer
        pair = next(t for t in EDL.TRANS['C'] if t['kind'] == 'dawn_dissolve')
        expected_kinds = {4711: None, 4712: 'dawn_dissolve', 4720: 'dawn_dissolve',
                          4727: 'dawn_dissolve', 4728: 'dawn_sweep', 5160: 'dawn_sweep',
                          5161: None, 5199: None, 5200: 'finish_ramp'}
        fake_stage = SimpleNamespace(Finisher=lambda look: fin)
        with mock.patch.object(AS.Ctx, 'picture', raw), mock.patch.dict(sys.modules, stage=fake_stage):
            picture = AS._transitions(ctx, True)
            for f, kind in expected_kinds.items():
                with self.subTest(frame=f):
                    raw_reads.clear()
                    finishes.clear()
                    tw = AS.transition_at('C', f)
                    self.assertEqual(tw['kind'] if tw else None, kind)
                    out, _, status, _ = picture(f)
                    if kind == 'dawn_dissolve':
                        self.assertEqual(raw_reads, [min(f, 4719), max(f, 4720)])
                        self.assertEqual(finishes, [('source', 'C19', f), ('source', 'C20', f)])
                        # Existing dissolve, with a neutral incoming plate, is the expected composition.
                        lit = incoming + np.float32(0.03)
                        night = (lit @ np.array([0.2126, 0.7152, 0.0722], np.float32))[..., None]
                        night = night * np.asarray(EDL.C_DAWN['night_rgb'], np.float32)
                        expected = AS._tk_dissolve(outgoing + np.float32(0.03), night, f, pair, None, {}, None)
                    elif f == 5200:
                        self.assertEqual(finishes, [('ink', f), ('film', f)])
                        expected = incoming + np.float32(0.03)
                    else:
                        sec = 'C19' if f < 4720 else 'C20'
                        self.assertEqual(finishes, [('source', sec, f)])
                        expected = (outgoing if sec == 'C19' else incoming) + np.float32(0.03)
                        if f == 4728:
                            expected = AS.dawn_sweep(expected, f, EDL.C_DAWN)
                    np.testing.assert_allclose(out, expected, atol=1e-7)
                    if kind:
                        self.assertIn(' + ' + kind, status)

    def test_both_windows_key_the_shared_grade_and_leave_other_segments_unchanged(self):
        code = {k: 'synthetic-code' for k in ('frame', 'text', 'slate', 'x2', 'title')}
        shots = [(i, s) for i, s in enumerate(EDL.EDL['C']) if s['sec'] in ('C19', 'C20', 'C21')]

        def plan(s, cut, variant):
            return dict(kind='take', take=s['takes'][0], have=s['f1'] - s['f0'], alt=False)

        def keys():
            with mock.patch.object(D, '_TRANS_CODE', []):
                return {s['sec']: D.segment_key('C', None, D.PROFILES['master'], i, s,
                                                plan(s, 'C', None), code, []) for i, s in shots}

        def assert_only_dawn_changed(before, after):
            self.assertEqual({sec for sec in before if before[sec] != after[sec]}, {'C19', 'C20'})

        # segment_key extends its source list in place; each read must own a fresh list, like frame_sources does.
        with mock.patch.object(AS, 'plan_shot', plan), mock.patch.object(D, 'frame_sources', side_effect=lambda *a: []), \
                mock.patch.object(AS, 'transition_layers', return_value={}):
            current = keys()
            old = [dict(t, kind='dissolve') if t['kind'] == 'dawn_dissolve' else t
                   for t in EDL.TRANS['C'] if t['kind'] != 'dawn_sweep']
            with mock.patch.dict(EDL.TRANS, C=old):
                assert_only_dawn_changed(keys(), current)
            changed = [dict(t, width=t['width'] + 0.1) if t['kind'].startswith('dawn_') else t
                       for t in EDL.TRANS['C']]
            with mock.patch.dict(EDL.TRANS, C=changed):
                assert_only_dawn_changed(current, keys())
            for kind in ('dawn_dissolve', 'dawn_sweep'):
                self.assertIn('def dawn_sweep(', AS.transition_code(kind))
            with mock.patch.dict(AS.KIND_HELPERS, dawn_dissolve=changed_helper, dawn_sweep=changed_helper):
                assert_only_dawn_changed(current, keys())


if __name__ == '__main__':
    unittest.main()
