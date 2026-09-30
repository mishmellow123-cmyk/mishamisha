"""Caption backing edits must invalidate only the affected C shot's cache.

Real EDL/text tables, synthetic source/code identities; no frame is read or drawn.
"""
from pathlib import Path
import sys
import unittest
from unittest import mock

EDIT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EDIT))
import deliver as D  # noqa: E402


class CaptionBackingCacheTests(unittest.TestCase):
    def setUp(self):
        # Each frame owns a fresh source list; segment_key appends title/transition inputs in place.
        self.enterContext(mock.patch.object(D, 'frame_sources', side_effect=lambda *args: []))
        self.enterContext(mock.patch.object(D, '_stat', return_value='synthetic-source-stat'))
        self.enterContext(mock.patch.object(D.AS, 'transition_at', return_value=None))
        self.code = {k: 'synthetic-code' for k in ('frame', 'text', 'slate', 'x2', 'title')}

    def keys(self):
        out = {}
        for cut in ('A', 'B', 'C'):
            table = D.titles.text_table(cut)
            for index, shot in enumerate(D.EDL.EDL[cut]):
                take = shot['takes'][0] if shot['takes'] else None
                plan = dict(kind='take' if take else shot['kind'], take=take,
                            have=shot['f1'] - shot['f0'], alt=False)
                out[(cut, index)] = D.segment_key(cut, None, D.PROFILES['master'], index, shot,
                                                plan, self.code, table)
        return out

    def assert_backing_rekeys_only_caption_shot(self, ident, before_value, after_value):
        rows = [dict(row) for row in D.titles.C5_TEXT]
        row = next(row for row in rows if row['id'] == ident)
        expected = {('C', index) for index, shot in enumerate(D.EDL.EDL['C'])
                    if row['f_in'] < shot['f1'] and row['f_out'] > shot['f0']}
        self.assertEqual(len(expected), 1, 'The fixture must own exactly one affected C shot')
        with mock.patch.object(D.titles, 'C5_TEXT', rows):
            if before_value is None:
                row.pop('backing', None)
            else:
                row['backing'] = before_value
            before = self.keys()
            row['backing'] = after_value
            after = self.keys()
        changed = {key for key in before if before[key] != after[key]}
        self.assertEqual(changed, expected,
                         'Only the backed caption shot may change; other C shots and every A/B shot must retain keys')

    def test_enabling_backing_rekeys_only_its_c_caption_shot(self):
        self.assert_backing_rekeys_only_caption_shot('R20', None, 0.6)

    def test_changing_existing_strength_rekeys_only_its_c_caption_shot(self):
        self.assert_backing_rekeys_only_caption_shot('R22', 0.2, 0.7)


if __name__ == '__main__':
    unittest.main()
