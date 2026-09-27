"""Picture-source readiness contracts; synthetic frame indexes, no rendering or live paths.

Run: python -B -m unittest discover -s the-long-dawn/edit/tests -p 'test_*.py'
"""
from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

EDIT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EDIT))
import assemble as AS
import deliver as D
import edl_v3 as EDL
import h9_kit as H9


class PictureReadinessTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.index = {}
        self.enterContext(mock.patch.object(AS, '_INDEX', self.index))
        self.enterContext(mock.patch.object(AS, 'RENDERS', str(self.root / 'renders')))
        self.c6 = next(s for s in EDL.EDL['C'] if s['sec'] == 'C6')
        self.c9 = next(s for s in EDL.EDL['C'] if s['sec'] == 'C9' and s['f0'] == 1920)

    def add_folder(self, stem, frames):
        folder = os.path.join(AS.RENDERS, stem)
        for f in frames:
            self.index.setdefault(folder, {})[f] = os.path.join(folder, f'f_{f:05d}.jpg')

    def trace_frame(self, take, f, failed_folders=()):
        """Run the real compositor on tiny controlled arrays, retaining its source-read order."""
        ctx = AS.Ctx.__new__(AS.Ctx)
        ctx.cut, ctx.variant = 'C', None
        reads = []
        def read(ref, crop=None, gray=False):
            folder = Path(ref).parent.name
            reads.append((folder, int(Path(ref).stem[2:])))
            if folder in failed_folders:
                return None
            if gray:
                return np.zeros((1, 1), np.float32)
            return np.full((1, 1, 3), 0.5 if folder == 'embers_C3_half' else 0.0, np.float32)
        ctx.read = read
        return ctx.take_frame(take, f), reads

    def add_take(self, take, frames):
        """Populate lookup metadata only; no synthetic frame is passed off as a real render."""
        folder = AS.chain(take, 'C', None)[0]
        for f in frames:
            src = f + take['off']
            self.index.setdefault(folder, {})[src] = os.path.join(folder, f'f_{src:05d}.jpg')
            if take.get('add'):
                add = os.path.join(AS.RENDERS, take['add'])
                self.index.setdefault(add, {})[src] = os.path.join(add, f'f_{src:05d}.jpg')
        if take.get('need'):
            for src in range(take['need'][0], take['need'][1] + 1):
                self.index.setdefault(folder, {})[src] = os.path.join(folder, f'f_{src:05d}.jpg')

    def full_c(self, provisional_c6=False):
        for shot in EDL.EDL['C']:
            if shot['kind'] == 'black':
                continue
            take = shot['takes'][1 if provisional_c6 and shot is self.c6 else 0]
            self.add_take(take, range(shot['f0'], shot['f1']))

    def qc(self, *, build=None, audio='COMPOSER master'):
        """Exercise real QC/readiness logic; unrelated media measurements use controlled signals."""
        total = EDL.TOTAL['C']
        # Tiny constant signal view with the correct sample count; media measurement helpers are isolated below.
        samples = np.broadcast_to(np.float32(0.1), (total * D.SR // D.FPS, 2))
        mov = self.root / 'C_master.mov'
        mov.write_bytes(b'synthetic-media-fixture')
        checks = {
            'ff_info': ('h264, yuv420p, 1920x804, 24 fps, bt709', '48000 Hz, stereo'),
            'count_video_frames': total,
            'decode_audio': samples,
            'true_peak_db': -2.0,
            'picture_stats': (np.full(total, 0.5), np.empty((0, 0)), total),
            'flashes': (0, 0),
        }
        with mock.patch.multiple(D, **{k: mock.Mock(return_value=v) for k, v in checks.items()}), \
                mock.patch.dict(sys.modules, {'pyloudnorm': None}):
            # Missing loudness support intentionally yields its existing WARN, independent of picture readiness.
            return D.qc('C', None, str(mov), None, audio,
                        build or dict(segments=len(EDL.EDL['C']), encoded=0, frames_encoded=0, seconds=0))

    def notified(self, report):
        """Run the actual watcher's completion reader, without sourcing its polling/rendering script."""
        (self.root / 'C_master_QC.json').write_text(json.dumps(report))
        script = (EDIT / 'refresh_watch.sh').read_text()
        reader = script[script.index('complete() {'):script.index('\nstep() {')]
        # Its Python executable is pinned to the test environment, not the user's shell Python.
        command = 'python3() { "$TEST_PYTHON" "$@"; }\n' + reader + '\ncomplete\n'
        result = subprocess.run(['bash', '-c', command], capture_output=True, text=True, check=True,
                                env=dict(os.environ, DELIV=str(self.root), TEST_PYTHON=sys.executable,
                                         PYTHONDONTWRITEBYTECODE='1'))
        return result.stdout.splitlines()

    def watch_signature(self):
        """Execute the watcher's actual metadata signature with this test's isolated frame index."""
        script = (EDIT / 'refresh_watch.sh').read_text()
        code = script.split("python3 - <<'EOF'\n", 1)[1].split('\nEOF\n', 1)[0]
        out = io.StringIO()
        with redirect_stdout(out):
            exec(compile(code, str(EDIT / 'refresh_watch.sh'), 'exec'), {})
        return out.getvalue()

    def test_explicit_provisional_families_only_and_approved_reuse(self):
        provisional = {t['stem'] for shots in EDL.EDL.values() for s in shots for t in s['takes']
                       if not EDL.is_final_take(t)}
        self.assertEqual(provisional, {'h1_v3', 'montage', 'run_b_tests/dusk_motion',
                                       'x1_letters_C_test', 'embers_C3_half'})
        self.assertTrue(EDL.is_final_take(EDL.T('accepted_v2', mode='layered', note='approved fire test')))
        self.assertTrue(EDL.is_final_take({'stem': 'legacy_approved_take'}))
        self.assertFalse(EDL.is_final_take(None))
        self.assertTrue(EDL.check(str(EDIT.parent / 'music' / 'v3')))

    def test_full_provisional_c6_stays_visible_but_cannot_notify_complete(self):
        self.full_c(provisional_c6=True)
        plan = AS.plan_shot(self.c6, 'C', None)
        self.assertEqual(plan['take']['stem'], 'embers_C3_half')
        self.assertEqual(plan['have'], 400)  # existing fallback selection is preserved
        self.assertEqual(H9.Film('C').status(self.c6['f0']), ('STAND_IN', False))
        self.assertEqual(D.status_frames('C', None)[0::3], (0, 0))  # no slates or EDIT proxies
        txt, report = self.qc()
        self.assertFalse(report['complete'])
        self.assertEqual(report['provisional_frames'], 400)
        self.assertIn('[WARN] provisional picture sources', txt)
        self.assertEqual(self.notified(report), [])

    def test_preferred_final_restores_completion_and_planned_black_is_valid(self):
        self.full_c(provisional_c6=True)
        self.add_take(self.c6['takes'][0], range(self.c6['f0'], self.c6['f1']))
        self.assertEqual(AS.plan_shot(self.c6, 'C', None)['take']['stem'], 'embers_C3')
        film = H9.Film('C')
        self.assertEqual(film.status(self.c6['f0']), ('RENDERED', False))
        self.assertEqual(film.status(0), ('BLACK', False))
        self.assertEqual(film.status(2480), ('BLACK', False))
        _, report = self.qc()
        self.assertTrue(report['complete'])
        self.assertEqual(report['provisional_frames'], 0)
        self.assertEqual(self.notified(report), ['C_master'])

    def test_partial_provisional_counts_only_present_frames(self):
        self.full_c(provisional_c6=True)
        folder = AS.chain(self.c6['takes'][1], 'C', None)[0]
        del self.index[folder][self.c6['f0']]
        self.assertEqual(D.provisional_frames('C', None)[0], 399)
        self.assertEqual(D.status_frames('C', None)[0], 1)

    def test_c9_provisional_under_composes_but_cannot_notify_complete(self):
        self.full_c()
        frames = range(self.c9['f0'], self.c9['f1'])
        self.add_folder('book_C_matte', frames)
        self.add_folder('embers_C3_half', frames)
        take = AS.plan_shot(self.c9, 'C', None)['take']
        self.assertEqual(take['stem'], 'book_C')
        picture, reads = self.trace_frame(take, 1920)
        self.assertEqual(reads, [('book_C', 1920), ('book_C_matte', 1920), ('embers_C3_half', 1920)])
        np.testing.assert_array_equal(picture, np.full((1, 1, 3), 0.5, np.float32))
        _, report = self.qc()
        self.assertFalse(report['complete'])
        self.assertEqual(report['provisional_frames'], 72)
        self.assertIn('under:embers_C3_half', report['provisional_sources'][0])
        self.assertEqual(H9.Film('C').status(1920), ('STAND_IN', False))
        self.assertEqual(self.notified(report), [])

    def test_c9_final_under_arrival_restores_readiness_and_rekeys_picture(self):
        self.full_c()
        frames = range(self.c9['f0'], self.c9['f1'])
        self.add_folder('book_C_matte', frames)
        self.add_folder('embers_C3_half', frames)
        plan = AS.plan_shot(self.c9, 'C', None)
        i = EDL.EDL['C'].index(self.c9)
        code, table = D._code_hash(), AS.titles.text_table('C')
        def key():
            return D.segment_key('C', None, D.PROFILES['master'], i, self.c9, plan, code, table)
        provisional_key, provisional_sig = key(), self.watch_signature()
        self.add_folder('embers_C3', range(1920, 1956))
        self.assertEqual(D.provisional_frames('C', None)[0], 36)
        self.assertEqual(H9.Film('C').status(1920), ('RENDERED', False))
        self.assertEqual(H9.Film('C').status(1956), ('STAND_IN', False))
        self.add_folder('embers_C3', range(1956, 1992))
        picture, reads = self.trace_frame(plan['take'], 1920)
        self.assertEqual(reads[-1], ('embers_C3', 1920))
        np.testing.assert_array_equal(picture, np.zeros((1, 1, 3), np.float32))
        _, report = self.qc()
        self.assertTrue(report['complete'])
        self.assertEqual(report['provisional_frames'], 0)
        self.assertEqual(self.notified(report), ['C_master'])
        self.assertNotEqual(key(), provisional_key)  # changed picture must re-encode
        self.assertNotEqual(self.watch_signature(), provisional_sig)

    def test_under_metadata_change_refreshes_qc_without_reencoding_picture(self):
        self.full_c()
        frames = range(self.c9['f0'], self.c9['f1'])
        self.add_folder('book_C_matte', frames)
        self.add_folder('embers_C3_half', frames)
        plan = AS.plan_shot(self.c9, 'C', None)
        i = EDL.EDL['C'].index(self.c9)
        code, table = D._code_hash(), AS.titles.text_table('C')
        def key():
            return D.segment_key('C', None, D.PROFILES['master'], i, self.c9, plan, D._code_hash(), table)
        before_signature, before_key = self.watch_signature(), key()
        with mock.patch.dict(EDL.UNDER_FINAL_ELIGIBILITY, embers_C3_half=True):
            self.assertEqual(AS.provisional_sources(plan['take'], 'C', None, 1920), ())
            self.assertNotEqual(self.watch_signature(), before_signature)
            self.assertEqual(D._code_hash(), code)
            self.assertEqual(key(), before_key)

    def test_c8_held_under_uses_1679_for_every_composited_frame(self):
        shot = next(s for s in EDL.EDL['C'] if s['sec'] == 'C8')
        take = shot['takes'][0]
        self.add_take(take, range(1680, 1920))
        self.add_folder('book_C_matte', range(1680, 1920))
        self.add_folder('embers_C3_half', [1679, 1680, 1919])
        for f in (1680, 1919):
            with self.subTest(frame=f):
                picture, reads = self.trace_frame(take, f)
                path, stem = AS.locate_under(take['under'], f)
                self.assertEqual(reads[-1], (stem, 1679))
                self.assertEqual(Path(path).name, 'f_01679.jpg')
                np.testing.assert_array_equal(picture, np.full((1, 1, 3), 0.5, np.float32))
                self.assertEqual(AS.provisional_sources(take, 'C', None, f), ('under:embers_C3_half',))
        self.assertIn('C8 P2 (240 f, under:embers_C3_half)', D.provisional_frames('C', None)[1])
        self.add_folder('embers_C3', [1679])
        self.assertEqual(AS.provisional_sources(take, 'C', None, 1919), ())
        self.assertEqual(self.trace_frame(take, 1919)[1][-1], ('embers_C3', 1679))

    def test_optional_layers_and_first_present_decode_failure_keep_existing_behavior(self):
        take = self.c9['takes'][0]
        self.add_take(take, [1920])
        self.add_folder('embers_C3_half', [1920])
        # The under source is unused without a matte path.
        self.assertEqual(self.trace_frame(take, 1920)[1], [('book_C', 1920)])
        self.assertEqual(AS.provisional_sources(take, 'C', None, 1920), ())
        self.add_folder('book_C_matte', [1920])
        half = self.index.pop(os.path.join(AS.RENDERS, 'embers_C3_half'))
        self.assertEqual(AS.locate_under(take['under'], 1920), (None, None))
        self.assertEqual(self.trace_frame(take, 1920)[1], [('book_C', 1920), ('book_C_matte', 1920)])
        self.assertEqual(AS.provisional_sources(take, 'C', None, 1920), ())
        self.index[os.path.join(AS.RENDERS, 'embers_C3_half')] = half
        self.add_folder('embers_C3', [1920])
        picture, reads = self.trace_frame(take, 1920, failed_folders=('embers_C3',))
        path, stem = AS.locate_under(take['under'], 1920)
        self.assertEqual(reads[-1], (stem, 1920))
        self.assertEqual(stem, 'embers_C3')
        self.assertEqual(Path(path).parent.name, stem)
        self.assertNotIn(('embers_C3_half', 1920), reads)
        np.testing.assert_array_equal(picture, np.zeros((1, 1, 3), np.float32))
        self.assertEqual(AS.provisional_sources(take, 'C', None, 1920), ())

    def test_watcher_tracks_under_folders_when_no_primary_take_uses_them(self):
        self.add_take(self.c9['takes'][0], range(1920, 1992))
        for stem in ('embers_C3', 'embers_C3_half'):
            (Path(AS.RENDERS) / stem).mkdir(parents=True)
        with mock.patch.dict(EDL.EDL, {'A': [], 'B': [], 'C': [self.c9]}):
            for stem in ('embers_C3', 'embers_C3_half'):
                with self.subTest(folder=stem):
                    before = self.watch_signature()
                    folder = Path(AS.RENDERS) / stem
                    st = folder.stat()
                    os.utime(folder, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000_000))
                    self.assertNotEqual(self.watch_signature(), before)

    def test_eligibility_change_refreshes_qc_without_reencoding_picture(self):
        self.full_c()
        shot = self.c6
        i = EDL.EDL['C'].index(shot)
        plan = AS.plan_shot(shot, 'C', None)
        code, table = D._code_hash(), AS.titles.text_table('C')
        def key():
            return D.segment_key('C', None, D.PROFILES['master'], i, shot, plan, code, table)
        before_signature, before_key = self.watch_signature(), key()
        with mock.patch.dict(plan['take'], final_eligible=False):
            self.assertNotEqual(self.watch_signature(), before_signature)
            self.assertEqual(key(), before_key)

    def test_picture_completeness_does_not_silently_add_audio_or_finish_policy(self):
        self.full_c()
        build = dict(segments=len(EDL.EDL['C']), encoded=0, frames_encoded=0, seconds=0,
                     finish='synthetic-look', finish_backlog_segments=1, finish_backlog_frames=400)
        txt, report = self.qc(build=build, audio='CLICK track')
        self.assertTrue(report['complete'])
        self.assertEqual(report['result'], 'WARN')
        self.assertIn('[WARN] finish:', txt)
        self.assertIn('sync click track', txt)


if __name__ == '__main__':
    unittest.main()
