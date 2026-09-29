"""C5 edit tools: the page_turn transition kind, the explicit local asset map, and C's sound-length guard.
Synthetic images and temporary folders only.

Run: python -B -m unittest discover -s the-long-dawn/edit/tests -p 'test_*.py'
"""
import contextlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

EDIT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EDIT))
sys.path.insert(0, str(EDIT / 'tools'))
import assemble as AS  # noqa: E402
import c5_assets as CA  # noqa: E402
import edl_v3 as EDL  # noqa: E402


def pages(h=80, w=192):
    """An outgoing 'page' (a warm ramp with ink strokes) and an incoming one (cooler, different)."""
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    o = np.stack([0.75 + 0.1 * x / w, 0.6 + 0.05 * y / h, np.full_like(x, 0.4)], -1)
    o[::9, :] *= 0.5                                                       # ink lines
    i = np.stack([np.full_like(x, 0.55), 0.5 + 0.1 * x / w, 0.45 + 0.05 * y / h], -1)
    return o.astype(np.float32), i.astype(np.float32)


class PageTurnTests(unittest.TestCase):
    def test_endpoints_are_exact_and_values_stay_in_range(self):
        o, i = pages()
        np.testing.assert_array_equal(AS.page_turn(o, i, 0.0), o)
        np.testing.assert_array_equal(AS.page_turn(o, i, 1.0), i)
        for p in np.linspace(0, 1, 23):
            out = AS.page_turn(o, i, float(p))
            self.assertEqual(out.dtype, np.float32)
            self.assertTrue(np.isfinite(out).all() and out.min() >= 0 and out.max() <= 1)

    def test_the_incoming_is_revealed_monotonically_from_the_right(self):
        o, i = pages()
        prev = -1.0
        for p in np.linspace(0, 1, 23):
            out = AS.page_turn(o, i, float(p))
            bare = float((np.abs(out - i).max(-1) < 1e-6).mean())          # pixels showing the incoming untouched
            self.assertGreaterEqual(bare + 1e-9, prev)
            prev = bare
        early = AS.page_turn(o, i, 0.3)
        self.assertLess(np.abs(early[:, -8:] - i[:, -8:]).mean(), np.abs(early[:, :8] - i[:, :8]).mean())

    def test_the_shadow_never_wraps_round_to_the_far_edge(self):
        """The first implementation rolled the shadow with np.roll: a leaf at the right edge darkened the left edge."""
        o, i = pages()
        for p in (0.12, 0.2, 0.28):                                          # the curl near the right edge only
            out = AS.page_turn(o, i, p)
            np.testing.assert_array_equal(out[:, :24], o[:, :24])

    def test_window_dispatch_is_a_registered_pair_kind_with_its_helper_keyed(self):
        self.assertIn('page_turn', AS.TKINDS_PAIR)
        self.assertIn('def page_turn', AS.transition_code('page_turn'))
        t = next(w for w in EDL.TRANS['C'] if w['kind'] == 'page_turn')
        self.assertEqual((t['f0'], t['cut'], t['f1']), (5430, 5440, 5452))
        o, i = pages()
        ctx = mock.Mock(W=o.shape[1])
        np.testing.assert_array_equal(AS._tk_page_turn(o, i, t['f0'], t, ctx, {}, None), o)
        np.testing.assert_array_equal(AS._tk_page_turn(o, i, t['f1'] - 1, t, ctx, {}, None), i)


class AssetMapTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.out = self.root / 'out'
        self.renders = self.root / 'renders'
        self.map = self.root / 'map.json'
        self.enterContext(mock.patch.object(CA, 'RENDERS', str(self.renders)))
        self.enterContext(mock.patch.object(CA, 'MANIFEST', str(self.root / 'manifest.json')))
        self.enterContext(mock.patch.dict(os.environ, {'LD_C5_ASSET_MAP': str(self.map)}))

    def deliver(self, stem, extra=(), missing=()):
        a, b = CA.DELIVERED[stem]
        d = self.out / stem
        d.mkdir(parents=True)
        for f in list(range(a, b + 1)) + list(extra):
            if f not in missing:
                (d / f'f_{f:05d}.jpg').write_bytes(b'x')
        return d

    def write_map(self, stems):
        self.map.write_text(json.dumps(dict(version=1, stems={s: str(self.out / s) for s in stems})))

    def test_exact_delivery_links_and_nothing_else_is_discovered(self):
        self.deliver('embers_C5_cold')
        self.deliver('embers_C5_trap')
        (self.out / 'embers_C5_cold_v2').mkdir()                              # a lookalike: must never be used
        self.write_map(['embers_C5_cold'])
        self.assertEqual(CA.run('link'), 0)
        self.assertTrue((self.renders / 'embers_C5_cold').is_symlink())
        self.assertEqual(os.path.realpath(self.renders / 'embers_C5_cold'), os.path.realpath(self.out / 'embers_C5_cold'))
        self.assertFalse((self.renders / 'embers_C5_trap').exists())        # delivered but unmapped: not linked
        self.assertEqual(sorted(p.name for p in self.renders.iterdir()), ['embers_C5_cold'])
        self.assertEqual(CA.run('link'), 0)                                   # idempotent

    def test_a_missing_or_stray_frame_refuses_the_link(self):
        self.deliver('embers_C5_cold', missing=(3900,))
        self.deliver('embers_C5_unfinished', extra=(4240,))
        self.write_map(['embers_C5_cold', 'embers_C5_unfinished'])
        self.assertEqual(CA.run('link'), 1)
        self.assertFalse((self.renders / 'embers_C5_cold').exists())
        self.assertFalse((self.renders / 'embers_C5_unfinished').exists())

    def test_existing_folder_or_other_link_is_never_replaced(self):
        self.deliver('embers_C5_cold')
        self.renders.mkdir()
        (self.renders / 'embers_C5_cold').mkdir()                             # a real production folder
        self.write_map(['embers_C5_cold'])
        self.assertEqual(CA.run('link'), 1)
        self.assertFalse((self.renders / 'embers_C5_cold').is_symlink())

    def test_receipts_beside_run_frames_are_allowed_other_files_are_not(self):
        d = self.deliver('runC_watch_v5')
        (d / 'receipt_04480_04719.json').write_text('{}')
        self.write_map(['runC_watch_v5'])
        self.assertEqual(CA.run('check'), 0)
        (d / 'notes.txt').write_text('x')
        self.assertEqual(CA.run('check'), 1)

    def test_a_moved_cold_cut_owes_the_lead_in_and_still_links_the_delivery(self):
        cut = [r for r in EDL.EDL['C'] if r['sec'] != 'C15']
        j = next(k for k, r in enumerate(cut) if r['f0'] == 3840)
        self.enterContext(mock.patch.dict(EDL.EDL, {'C': cut[:j] + EDL.last_beacon_rows(3816) + cut[j:]}))
        self.deliver('embers_C5_cold')                                        # PR11's 3840-3999: no lead-in
        self.write_map(['embers_C5_cold'])
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.assertEqual(CA.run('link'), 0)
        self.assertIn('OWED embers_C5_cold: the EDL reads 3816-3839 (24 frames) beyond the recorded delivery '
                      '3840-3999', out.getvalue())
        self.assertTrue((self.renders / 'embers_C5_cold').is_symlink())
        self.assertEqual(CA.split_need('embers_C5_cold', set(range(3816, 4000))),
                         (set(range(3840, 4000)), set(range(3816, 3840))))

    def test_needed_frames_come_from_the_edl(self):
        need = CA.needed_frames('C')
        self.assertEqual(need['runC_illum'], set(range(2398, 2878)))
        self.assertEqual(need['book_C5_pen_matte'], set(range(5440, 5680)))
        self.assertEqual(need['embers_C3'] & {1679}, {1679})                  # C8's held race
        self.assertNotIn('ring_C', need)                                      # C12 is an open decision


class SoundGuardTests(unittest.TestCase):
    def test_c_refuses_a_sound_file_of_another_length_and_a_keeps_its_behaviour(self):
        from types import SimpleNamespace
        lengths = {'sound_C.wav': 7200 / 24, 'sound_C5.wav': 5920 / 24}          # seconds, from the WAV header
        with mock.patch('soundfile.info', lambda p: SimpleNamespace(duration=lengths[os.path.basename(p)])):
            self.assertFalse(AS.audio_fits('/x/sound_C.wav', 'C'))              # the 7,200-frame cut's master
            self.assertTrue(AS.audio_fits('/x/sound_C5.wav', 'C'))
            self.assertTrue(AS.audio_fits('/x/sound_C.wav', 'A'))               # A and B: unguarded, unchanged
            with mock.patch.object(AS, '_audio_candidates', lambda cut: (('/x/sound_C.wav', 'SOUND master'),)), \
                    mock.patch.object(AS.os.path, 'isfile', lambda p: True), \
                    mock.patch.object(AS, 'click_track', lambda cut: 'click.wav'):
                path, label = AS.resolve_audio('C')
        self.assertEqual(path, 'click.wav')
        self.assertTrue(label.startswith('CLICK track'))
        self.assertNotIn('C', AS.ADOPTED_AUDIO)


if __name__ == '__main__':
    unittest.main()
