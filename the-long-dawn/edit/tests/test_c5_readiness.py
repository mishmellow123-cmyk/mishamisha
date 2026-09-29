"""C5 readiness gate (edit/c5_readiness.py): a synthetic all-present C5 goes green, and every defect class turns it red
in the mode it should. Synthetic frame indexes only: no image is read and no live path is used.

Run: python -B -m unittest discover -s the-long-dawn/edit/tests -p 'test_*.py'
"""
import copy
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

EDIT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EDIT))
sys.path.insert(0, str(EDIT / 'tools'))
import assemble as AS  # noqa: E402
import c5_assets as CA  # noqa: E402
import c5_readiness as RD  # noqa: E402
import edl_v3 as EDL  # noqa: E402
import titles  # noqa: E402

V52_T1 = 'In the old story, a Dark Lord forges a Ring to rule the world.'


class C5ReadinessTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.index = {}
        self.enterContext(mock.patch.object(AS, '_INDEX', self.index))
        self.enterContext(mock.patch.object(AS, 'RENDERS', str(self.root / 'renders')))
        self.enterContext(mock.patch.object(RD, '_size', lambda p: 1))   # synthetic paths: present and non-empty
        self.enterContext(mock.patch.object(RD, '_matte_holes', lambda paths: []))   # synthetic mattes: opaque

    # --------------------------------------------------------------------------------------------- helpers
    def put(self, stem, frames):
        folder = os.path.join(AS.RENDERS, stem)
        for f in frames:
            self.index.setdefault(folder, {})[f] = os.path.join(folder, f'f_{f:05d}.jpg')

    def drop(self, stem, frame):
        del self.index[os.path.join(AS.RENDERS, stem)][frame]

    def green(self, flint='A', cold_cut=None, set_cold_cut=True):
        """C5 with every source present, a Flint candidate realised, the designed-but-unbuilt transitions left out,
        T1's pixels re-rendered with the v5.2 words, and edl_C.json regenerated: the full gate must pass. cold_cut
        rebuilds C15 as edl_v3 would with COLD_CUT set there (and sets it, unless set_cold_cut is False)."""
        cut = [r for r in EDL.EDL['C'] if r['sec'] != 'C12']
        i = next(k for k, r in enumerate(cut) if r['f0'] == 2880)
        cut = cut[:i] + EDL.flint_rows(flint) + cut[i:]
        if cold_cut is not None:
            cut = [r for r in cut if r['sec'] != 'C15']
            j = next(k for k, r in enumerate(cut) if r['f0'] == 3840)
            cut = cut[:j] + EDL.last_beacon_rows(cold_cut) + cut[j:]
            if set_cold_cut:
                self.enterContext(mock.patch.object(EDL, 'COLD_CUT', cold_cut))
        self.enterContext(mock.patch.dict(EDL.EDL, {'C': cut}))
        self.enterContext(mock.patch.object(EDL, 'FLINT_CHOICE', flint))      # as setting it in edl_v3.py would
        self.enterContext(mock.patch.dict(EDL.TRANS, {'C': [t for t in EDL.TRANS['C'] if t.get('ready', True)]}))
        baked = [dict(b, line=V52_T1) if b['src'] == (400, 540) else b for b in titles.BAKED_TEXT]
        self.enterContext(mock.patch.object(titles, 'BAKED_TEXT', baked))
        for s in cut:
            if not s['takes']:
                continue
            t = s['takes'][0]
            src = [f + t['off'] for f in range(s['f0'], s['f1'])]
            self.put(t['stem'], src)
            for layer in (t.get('matte'), t.get('add')):
                if layer:
                    self.put(layer, src)
            if t.get('under'):
                how, stem, *rest = t['under']
                self.put(stem, [rest[0]] if how == 'hold' else range(s['f0'], s['f1']))
        for t in EDL.TRANS['C']:                  # a built transition's layer frames are delivered too (29 Sep: the burns)
            if t.get('ready', True):
                for k in ('glow', 'keep', 'cover'):
                    if t.get(k):
                        self.put(t[k], range(t['f0'], t['f1']))
        self.write_json()

    def write_json(self):
        p = self.root / 'edl_C.json'
        p.write_text(json.dumps(AS.edl_doc('C'), indent=1))
        self.enterContext(mock.patch.object(RD, 'JSON_PATH', str(p)))

    def run_gate(self, partial=False):
        rep = RD.run(partial)
        return rep, {(i['level'], i['what']) for i in rep.items}, ' | '.join(i['msg'] for i in rep.items
                                                                            if i['level'] in ('FAIL', 'GAP'))

    def assertGreen(self, rep, text):
        self.assertFalse(rep.failed(), text)
        self.assertEqual((rep.count('FAIL'), rep.count('GAP')), (0, 0), text)

    # ----------------------------------------------------------------------------------------------- tests
    def test_green_control_full_c5_passes_the_full_gate(self):
        self.green()
        rep, _, text = self.run_gate()
        self.assertGreen(rep, text)
        present = sum(b - a for a, b, _, _, st in rep.runs if st in ('present', 'black'))
        self.assertEqual(present, 5920)

    def test_committed_state_fails_full_and_passes_partial(self):
        """The branch as committed: no source here, Flint chosen (29 Sep), T1 re-rendered with the current words and
        both burns built (29 Sep): FULL fails only on this machine's missing sources."""
        rep, keys, text = self.run_gate(partial=False)
        self.assertTrue(rep.failed())
        self.assertEqual(rep.count('FAIL'), 0, text)
        self.assertIsNotNone(EDL.FLINT_CHOICE)
        self.assertNotIn(('GAP', 'C12 FLINT 2640-2879: decision_required'), keys)
        self.assertTrue(any(k[1].startswith('C12 FLINT ') for k in keys if k[0] == 'GAP'), text)   # realised rows, no source here
        # T1 plays the adopted re-render with R02's current words (29 Sep): no re-render is owed any more
        self.assertFalse(any('RE-RENDER' in i['msg'] for i in rep.items if i['level'] == 'GAP'))
        self.assertEqual(sum(i['what'].startswith('transition burn') for i in rep.items if i['level'] == 'GAP'), 2)
        rep_p, _, _ = self.run_gate(partial=True)
        self.assertFalse(rep_p.failed())

    def test_missing_source_frame_fails_full_only(self):
        self.green()
        self.drop('cand_trap_front_smoke_near', 2500)
        rep, keys, text = self.run_gate()
        self.assertTrue(rep.failed())
        self.assertIn('1 of 320 frames not resolvable from cand_trap_front_smoke_near 2320-2639', text)
        self.assertFalse(self.run_gate(partial=True)[0].failed())

    def test_missing_matte_or_under_is_a_missing_source(self):
        self.green()
        self.drop('cand_pen_soft_spine_metal_matte', 5500)
        self.drop('embers_C3', 1950)                             # the Eye's live storm under the filmed burn
        rep, _, text = self.run_gate()
        self.assertIn('(missing: cand_pen_soft_spine_metal_matte)', text)
        self.assertIn('(missing: under:embers_C3)', text)

    def test_provisional_under_is_not_final(self):
        self.green()
        self.drop('embers_C3', 1950)
        self.put('embers_C3_half', [1950])                       # Ctx.under's silent _half fallback
        rep, _, text = self.run_gate()
        self.assertTrue(rep.failed())
        self.assertIn('1 use provisional sources', text)

    def test_open_flint_decision_fails_full_only(self):
        self.green()
        self.enterContext(mock.patch.dict(EDL.EDL, {'C': _with_flint(_ORIGINAL_C, None)}))   # C12 back to the decision row
        self.enterContext(mock.patch.object(EDL, 'FLINT_CHOICE', None))
        self.write_json()
        rep, keys, text = self.run_gate()
        self.assertEqual([k for k in keys if k[0] in ('FAIL', 'GAP')],
                         [('GAP', 'C12 FLINT 2640-2879: decision_required')], text)
        self.assertTrue(rep.failed())
        self.assertFalse(self.run_gate(partial=True)[0].failed())

    def test_flint_rows_that_disagree_with_the_choice_fail_both_modes(self):
        self.green()
        self.enterContext(mock.patch.object(EDL, 'FLINT_CHOICE', None))      # rows realised, choice not recorded
        rep, _, text = self.run_gate(partial=True)
        self.assertTrue(rep.failed())
        self.assertIn('are not the single decision row', text)

    def test_flint_candidate_reading_the_retired_find_fails_both_modes(self):
        self.green()
        bad = dict(EDL.FLINT_CANDIDATES, X=dict(pieces=((2640, 2880, 3120),), note='uniform +480'))
        self.enterContext(mock.patch.object(EDL, 'FLINT_CANDIDATES', bad))
        rep, keys, text = self.run_gate(partial=True)
        self.assertTrue(rep.failed())
        self.assertIn('RETIRED find/vision', text)

    def test_gap_and_overlap_fail_both_modes(self):
        self.green()
        rows = copy.deepcopy(EDL.EDL['C'])
        k = next(i for i, r in enumerate(rows) if r['sec'] == 'C16')
        rows[k]['f0'] += 8                                       # 3840-3847 now belong to no row
        with mock.patch.dict(EDL.EDL, {'C': rows}):
            rep, keys, text = self.run_gate(partial=True)
        self.assertTrue(rep.failed())
        self.assertIn('gaps at 3840-3847', text)
        rows = copy.deepcopy(EDL.EDL['C'])
        rows[k]['f0'] -= 8                                       # 3832-3839 now belong to two rows
        with mock.patch.dict(EDL.EDL, {'C': rows}):
            rep, keys, text = self.run_gate(partial=True)
        self.assertIn('overlaps at 3832-3839', text)

    # ------------------------------------------------ the movable map -> Cold cut (edl_v3.COLD_CUT, one number)
    def test_committed_cold_cut_is_the_adopted_lead_in(self):
        """29 Sep: COLD_CUT 3816 plays Codex's lead24 (cand_cold_lead24) after the adopted map; R15 crosses that cut on
        purpose (titles.C5_TEXT R15 across=3816), which the gate reports as INFO, never as a WARN."""
        self.assertEqual(EDL.COLD_CUT, 3816)
        self.assertEqual([(r['f0'], r['f1'], r['takes'][0]['stem']) for r in _ORIGINAL_C if r['sec'] in ('C15', 'C16')],
                         [(3440, 3816, 'cand_map_beacon-falloff'), (3816, 3840, 'cand_cold_lead24'),
                          (3840, 4000, 'embers_C5_cold')])
        self.assertEqual(RD.hard_cuts().get(3816), ('cand_map_beacon-falloff', 'cand_cold_lead24'))
        items = RD.run(True).items
        self.assertFalse([i for i in items if 'runs across the hard cut' in i['msg']])
        self.assertTrue([i for i in items if i['level'] == 'INFO' and 'runs across the cut at 3816 on purpose' in i['msg']])

    def test_a_staged_line_must_rejoin_and_each_part_stay_in_its_rows_shots(self):
        """R15 in parts (REVIEW 29 Sep): the parts must rejoin to the verbatim line, not overlap, and each lie inside one
        of row 15's shots (C15, C16); breaking any of the three FAILs by name."""
        self.green()
        rep, _, text = self.run_gate()
        self.assertGreen(rep, text)
        base = [dict(r) for r in titles.C5_TEXT]
        r15 = next(r for r in base if r['id'] == 'R15')
        for label, parts, needle in (
                ('words', (dict(r15['parts'][0]), dict(r15['parts'][1], line='every forge went dark.')),
                 'do not rejoin to the verbatim line'),
                ('overlap', (dict(r15['parts'][0]), dict(r15['parts'][1], f_in=3830)), 'its parts overlap in time'),
                ('outside', (dict(r15['parts'][0]), dict(r15['parts'][1], f_out=4010)),
                 "R15b 3850-4009 is not inside one of its row's shots")):
            with self.subTest(label):
                rows = [dict(r, parts=parts) if r['id'] == 'R15' else r for r in base]
                with mock.patch.object(titles, 'C5_TEXT', rows):
                    rep, _, text = self.run_gate()
                fails = ' '.join(i['msg'] for i in rep.items if i['level'] == 'FAIL')
                self.assertIn(needle, fails)

    def test_cold_cut_moves_in_one_place_and_names_what_follows(self):
        """COLD_CUT 3800 plus a delivery covering its lead-in is green; R15's marker names 3816, so crossing 3800
        is WARNed by name (a crossing nobody reviewed). Since the review (29 Sep) R15 is staged in parts and the part
        that crosses is named: R15a, the catch."""
        self.green(cold_cut=3800)
        self.enterContext(mock.patch.dict(CA.DELIVERED, {'cand_cold_lead24': (3800, 3999)}))   # a longer delivery
        rep, _, text = self.run_gate()
        self.assertGreen(rep, text)
        self.assertEqual([(r['f0'], r['f1'], r['takes'][0]['stem'], r['takes'][0]['off']) for r in RD.rows_of('C15')],
                         [(3440, 3800, 'cand_map_beacon-falloff', 0), (3800, 3840, 'cand_cold_lead24', 0)])
        warns = [i['msg'] for i in rep.items if i['level'] == 'WARN']
        self.assertIn('R15a 3776-3839 runs across the hard cut at 3800 (cand_map_beacon-falloff -> cand_cold_lead24): '
                      'its words change picture mid-line; end it by 3800, or keep it across on purpose', warns)
        self.assertEqual(CA.needed_frames()['cand_cold_lead24'], set(range(3800, 3840)))

    def test_moved_cold_cut_before_its_delivery_is_owed_not_broken(self):
        self.green(cold_cut=3800)
        for f in range(3800, 3816):
            self.drop('cand_cold_lead24', f)                      # the delivered lead-in starts at 3816
        rep, keys, text = self.run_gate()
        self.assertTrue(rep.failed())
        self.assertEqual(rep.count('FAIL'), 0, text)
        self.assertIn('16 of 40 frames not resolvable from cand_cold_lead24 3800-3839', text)
        self.assertIn('the EDL reads 3800-3815 (16 frames) beyond the recorded delivery 3816-3999', text)
        self.assertFalse(self.run_gate(partial=True)[0].failed())

    def test_cold_cut_outside_the_hold_or_unrecorded_fails_both_modes(self):
        self.green(cold_cut=3780)                                 # would cut away the last kingdom's catch
        rep, _, text = self.run_gate(partial=True)
        self.assertTrue(rep.failed())
        self.assertIn('edl_v3.COLD_CUT 3780 is outside 3792-3840', text)

    def test_c15_rows_for_a_cut_the_constant_does_not_name_fail(self):
        self.green(cold_cut=3800, set_cold_cut=False)             # rows edited by hand, COLD_CUT still 3816
        rep, _, text = self.run_gate(partial=True)
        self.assertTrue(rep.failed())
        self.assertIn('C15: rows [(3440, 3800), (3800, 3840)], expected [(3440, 3816), (3816, 3840)] '
                      '(edl_v3.COLD_CUT = 3816)', text)

    def test_transition_hole_from_an_opaque_page_matte_fails_both_modes(self):
        self.green()
        wins = [t for t in EDL.TRANS['C'] if t['kind'] != 'page_turn'] + [
            dict(f0=5430, f1=5452, cut=5440, kind='burn', glow='x1_pen', keep='book_C5_pen_matte', cover='x1_pen_c')]
        with mock.patch.dict(EDL.TRANS, {'C': wins}):
            self.write_json()
            rep, _, text = self.run_gate(partial=True)
        self.assertTrue(rep.failed())
        self.assertIn("burn 5430-5451: takes a layer from ['book_C5_pen_matte']", text)

    def test_stale_json_fails_both_modes(self):
        self.green()
        doc = AS.edl_doc('C')
        doc['shots'][0]['desc'] = 'hand-edited'
        p = self.root / 'stale.json'
        p.write_text(json.dumps(doc))
        with mock.patch.object(RD, 'JSON_PATH', str(p)):
            rep, keys, text = self.run_gate(partial=True)
        self.assertTrue(rep.failed())
        self.assertIn(('FAIL', 'edl_C.json'), keys)

    def test_wrong_offset_and_superseded_burn_fail_the_shot_map(self):
        self.green()
        rows = copy.deepcopy(EDL.EDL['C'])
        k = next(i for i, r in enumerate(rows) if r['sec'] == 'C20')
        rows[k]['takes'][0]['off'] = 2398 - 5680                 # the 7,200-frame cut's offset, unconverted
        with mock.patch.dict(EDL.EDL, {'C': rows}):
            self.assertIn("('runC_illum', -3282, 'exact')", self.run_gate(partial=True)[2])
        rows = copy.deepcopy(EDL.EDL['C'])
        k = next(i for i, r in enumerate(rows) if r['sec'] == 'C5')
        rows[k]['takes'] = [EDL.book_e15()]                      # the old burn back inside the filmed range
        with mock.patch.dict(EDL.EDL, {'C': rows}):
            self.assertIn('superseded book_C burn can play inside the filmed range', self.run_gate(partial=True)[2])

    def test_double_composited_sweep_fails(self):
        self.green()
        rows = copy.deepcopy(EDL.EDL['C'])
        k = next(i for i, r in enumerate(rows) if r['sec'] == 'C8' and r['f0'] == 1680)
        rows[k]['takes'][0]['under'] = ('hold', 'embers_C3', 1679)
        with mock.patch.dict(EDL.EDL, {'C': rows}):
            rep, _, text = self.run_gate(partial=True)
        self.assertTrue(rep.failed())
        self.assertIn('is not BURN_NOTES', text)

    def test_caption_not_verbatim_or_outside_its_shot_fails_both_modes(self):
        self.green()
        rows = copy.deepcopy(titles.C5_TEXT)
        rows[[r['id'] for r in rows].index('R13')]['line'] = 'So the two furthest ahead lit the first beacons together.'
        with mock.patch.object(titles, 'C5_TEXT', rows):
            self.write_json()
            rep, _, text = self.run_gate(partial=True)
        self.assertTrue(rep.failed())
        self.assertIn('is not verbatim', text)
        rows = copy.deepcopy(titles.C5_TEXT)
        rows[[r['id'] for r in rows].index('R17')]['f_out'] = 4260    # runs into the Deep
        with mock.patch.object(titles, 'C5_TEXT', rows):
            self.write_json()
            self.assertIn("R17 4040-4259 is not inside one of its row's shots", self.run_gate(partial=True)[2])

    def test_a_line_break_that_changes_the_words_fails_both_modes(self):
        self.green()
        rows = copy.deepcopy(titles.C5_TEXT)
        k = [r['id'] for r in rows].index('R18')
        self.assertEqual(rows[k]['lines'], ('They left the gold', 'in the ground.'))     # as committed: verbatim
        rows[k]['lines'] = ('They left the gold', 'in the ground')                     # the full stop lost
        with mock.patch.object(titles, 'C5_TEXT', rows):
            self.write_json()
            rep, _, text = self.run_gate(partial=True)
        self.assertTrue(rep.failed())
        self.assertIn('R18: its line breaks', text)

    def test_a_caption_whose_glyphs_leave_the_picture_fails_both_modes(self):
        self.green()
        rows = copy.deepcopy(titles.C5_TEXT)
        rows[[r['id'] for r in rows].index('R13')]['x'] = 300                # a 1,187-px line centred at x 300
        with mock.patch.object(titles, 'C5_TEXT', rows):
            self.write_json()
            rep, _, text = self.run_gate(partial=True)
        self.assertTrue(rep.failed())
        self.assertIn('R13: its glyphs span x -', text)

    def test_in_picture_over_unbaked_pen_frames_fails_both_modes(self):
        """The old T14 suppression: flag the ending line in_picture over the new, text-free Pen frames and it would
        silently vanish."""
        self.green()
        rows = copy.deepcopy(titles.C5_TEXT)
        rows[[r['id'] for r in rows].index('R22')]['set'] = 'in_picture'
        with mock.patch.object(titles, 'C5_TEXT', rows):
            self.write_json()
            rep, _, text = self.run_gate(partial=True)
        self.assertTrue(rep.failed())
        self.assertIn('bakes no words', text)
        self.assertIn('silently vanish', text)

    def test_old_baked_words_need_a_rerender_and_edit_text_on_them_is_double(self):
        """The accepted mountain page (edl_v3.ALTERNATIVES['C3']: book_C with the old T1 words) played again."""
        self.green()
        rows = copy.deepcopy(EDL.EDL['C'])
        k = next(i for i, r in enumerate(rows) if r['sec'] == 'C3')
        rows[k]['takes'] = [EDL.book()]
        self.enterContext(mock.patch.dict(EDL.EDL, {'C': rows}))
        self.put('book_C', range(320, 560))
        self.put('book_C_matte', range(320, 560))
        self.enterContext(mock.patch.object(titles, 'BAKED_TEXT', [dict(b) for b in _ORIGINAL_BAKED]))
        self.write_json()                                         # edl_C.json carries the registry too
        rep, _, text = self.run_gate()
        self.assertTrue(rep.failed())
        self.assertIn('RE-RENDER book_C 400-539', text)
        self.assertFalse(self.run_gate(partial=True)[0].failed())
        rows = copy.deepcopy(titles.C5_TEXT)
        rows[0]['set'] = 'ink'                                    # EDIT's line over the old baked words
        with mock.patch.object(titles, 'C5_TEXT', rows):
            self.write_json()
            self.assertIn('double text', self.run_gate()[2])

    def test_baked_words_no_line_owns_fail(self):
        self.green()
        rows = copy.deepcopy(EDL.EDL['C'])
        k = next(i for i, r in enumerate(rows) if r['sec'] == 'C23')
        rows[k]['takes'][0]['off'] = 6720 - 5680 + 110            # the title slot reading the old C27 verso's T14
        with mock.patch.dict(EDL.EDL, {'C': rows}):
            self.assertIn("that no in_picture line owns", self.run_gate(partial=True)[2])

    def test_designed_transition_fails_full_only_and_bad_windows_fail_both(self):
        self.green()
        # the Refusal burn as it stood before its layers were built (29 Sep): designed, not ready
        designed = [dict(t, ready=False) if t['f0'] == 2310 else t for t in _ORIGINAL_TRANS]
        self.enterContext(mock.patch.dict(EDL.TRANS, {'C': designed}))
        self.write_json()
        rep, keys, _ = self.run_gate()
        self.assertIn(('GAP', 'transition burn 2310-2345'), keys)
        self.assertFalse(self.run_gate(partial=True)[0].failed())
        wins = list(_ORIGINAL_TRANS) + [dict(f0=4600, f1=4610, cut=4605, kind='dissolve')]   # not a shot boundary
        with mock.patch.dict(EDL.TRANS, {'C': wins}):
            self.write_json()
            self.assertIn('cut 4605 is not a boundary', self.run_gate(partial=True)[2])

    def test_delivered_folder_must_be_exact(self):
        self.green()
        d = self.root / 'renders' / 'embers_C5_cold'
        d.mkdir(parents=True)
        for f in list(range(3840, 4000)) + [4000]:               # one stray frame beyond the delivery
            (d / f'f_{f:05d}.jpg').write_bytes(b'x')
        rep, keys, text = self.run_gate(partial=True)
        self.assertTrue(rep.failed())
        self.assertIn('1 frames outside the delivered 3840-3999', text)

    def test_held_under_fails_once_its_matte_has_a_hole(self):
        self.green()
        self.assertGreen(*self.run_gate()[::2])
        self.enterContext(mock.patch.object(RD, '_matte_holes', lambda paths: [f for f in paths if f == 1910]))
        rep, keys, text = self.run_gate()
        self.assertTrue(rep.failed())
        self.assertIn(('FAIL', 'held under'), keys)
        self.assertIn('holds embers_C3 1920, but book_C_ft_matte lets it through at 1910-1910', text)

    def test_held_under_without_its_matte_frames_is_a_gap(self):
        self.green()
        self.drop('book_C_ft_matte', 1912)
        rep, keys, text = self.run_gate()
        self.assertIn(('GAP', 'held under 1905-1919'), keys)
        self.assertIn('cannot confirm the held storm is hidden', text)

    def test_glow_row_holds_what_ft_held_names_and_nothing_else_passes(self):
        glow = next(r for r in EDL.EDL['C'] if (r['f0'], r['f1']) == (1905, 1920))
        self.assertEqual(tuple(glow['takes'][0]['under']), RD.FT_HELD[(1905, 1920)])
        self.green()
        live = [dict(r, takes=[dict(glow['takes'][0], under=['same', 'embers_C3'])]) if r is glow else r
                for r in EDL.EDL['C']]
        self.enterContext(mock.patch.dict(EDL.EDL, {'C': live}))
        rep, _, text = self.run_gate()
        self.assertTrue(rep.failed())
        self.assertIn("C8 1905-1920: book_C_ft comp", text)

    def test_matte_holes_reads_real_pixels(self):
        import cv2
        import numpy as np
        opaque, pinhole, broken = (str(self.root / n) for n in ('o.png', 'p.png', 'b.png'))
        cv2.imwrite(opaque, np.full((8, 8), 255, np.uint8))
        img = np.full((8, 8), 255, np.uint8)
        img[3, 5] = 254                                          # one pixel lets 1/255 of the under-layer through
        cv2.imwrite(pinhole, img)
        Path(broken).write_bytes(b'not an image')
        self.assertEqual(REAL_MATTE_HOLES({1: opaque, 2: pinhole, 3: broken}), [2, 3])

    def test_script_parser_reads_the_handover_table(self):
        rows = RD.script_v52()
        self.assertEqual(sorted(rows), list(range(1, 24)))
        self.assertIsNone(rows[12])
        self.assertEqual(rows[22], 'The last pages were left for us.')
        self.assertEqual(rows[23], 'THE LONG DAWN')


_ORIGINAL_C = list(EDL.EDL['C'])


def _with_flint(rows, choice):
    """rows with C12 rebuilt as edl_v3 builds it for `choice` (None: the single decision row). _ORIGINAL_C carries
    whatever edl_v3.FLINT_CHOICE was at import, so a test about the open decision must not rely on it."""
    out = [r for r in rows if r['sec'] != 'C12']
    i = next(k for k, r in enumerate(out) if r['f0'] == 2880)
    return out[:i] + EDL.flint_rows(choice) + out[i:]
REAL_MATTE_HOLES = RD._matte_holes                          # captured before setUp mocks it
_ORIGINAL_BAKED = [dict(b) for b in titles.BAKED_TEXT]
_ORIGINAL_TRANS = list(EDL.TRANS['C'])

if __name__ == '__main__':
    unittest.main()
