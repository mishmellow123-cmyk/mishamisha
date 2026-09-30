"""The 115-bar D edit's music metadata: inherited evidence and provisional new picture timing."""
import copy
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'edit'))
sys.path.insert(0, str(ROOT / 'music/src'))
import edl_v3 as EDL
import titles
from timeline_v3 import BarMap, validate


def load(name):
    return json.loads((ROOT / f'music/v3/{name}.json').read_text())


def by_id(rows):
    return {row['id']: row for row in rows}


def test_d_own_sections_tile_115_bars_and_match_the_edit():
    disk = load('barmap_D')
    assert (disk['frames'], disk['bars'], disk['fps'], disk['bpm']) == (9200, 115, 24, 72)
    assert [(s['id'], s['bar_start'], s['bar_end'], s['f0'], s['f1']) for s in disk['sections']] == [
        ('D0', 1, 18, 0, 1440), ('D1', 19, 30, 1440, 2400), ('D2', 31, 44, 2400, 3520),
        ('D3', 45, 51, 3520, 4080), ('D4', 52, 73, 4080, 5840),
        ('D5', 74, 88, 5840, 7040), ('D6', 89, 92, 7040, 7360), ('D7', 93, 115, 7360, 9200)]
    assert EDL.TOTAL['D'] == disk['frames']
    for section in disk['sections']:
        rows = [r for r in EDL.D if r['sec'] == section['id']]
        assert (rows[0]['f0'], rows[-1]['f1']) == (section['f0'], section['f1'])
    assert validate(BarMap('D'), verbose=False) == ([], [])


def test_all_34_picture_entries_and_internal_treatment_events_are_named():
    events = by_id(load('barmap_D')['sync'])
    starts = [0, 80, 560, 960, 1040, 1440, 1680, 1760, 2080, 2400, 2720, 2960, 3200, 3440,
              3520, 3760, 4080, 4240, 4560, 5040, 5180, 5680, 5840, 6080, 6400, 6640,
              7040, 7360, 7840, 8080, 8320, 8640, 8880, 9120]
    entries = sorted((e['shot'], e['f']) for e in events.values() if e.get('picture_name'))
    assert entries == list(enumerate(starts, 1))
    required = {'inscription_stops_midletter', 'inscription_pullback', 'forging_towers_rise',
                'forging_hammer_strokes', 'forging_gold_pays_striker', 'race_beat_surges',
                'race_glare_blooms', 'race_giants_ahead', 'deep_ember_sweep', 'deep_pen_descends',
                'deep_red_glow', 'vision_ends_touch', 'vision_white_seam', 'vision_missing_letters',
                'vision_ice_eye', 'vision_giants_bend', 'vision_all_towers_bend', 'vision_slit_push',
                'trap_forge_slows', 'trap_gold_passes', 'trap_forge_overtaken', 'trap_forge_rejoins',
                'ridge_answers', 'holdout_gap_notch', 'in_step_shared_work', 'in_step_ends_cool',
                'in_step_glare_clears', 'old_fire_ring_falls', 'unfinished_lamps_rise',
                'unfinished_gold_all_windows', 'watch_bar_breath', 'dawn_rose', 'dawn_sunlight',
                'holdout_first_contact', 'last_leaf_line_breaks'}
    assert required <= set(events)


def assert_provenance(row):
    assert row.get('source'), 'every sync needs source provenance'
    assert row['bar'] == row['f'] // 80 + 1
    assert row['beat'] == 1 + (row['f'] % 80) / 20
    assert row['t'] == round(row['f'] / 24, 3)
    if row['timing_status'] == 'estimated':
        assert row['picture_frame'] is None, 'estimated timing cannot claim a measured D frame'
        assert '(est.' in row['what'], 'estimated timing must label the visible event description'
        assert 'D_TREATMENT_v2.md' in row['source'] and 'provisional' in row['source']
        assert row['timing_window'][0] <= row['f'] < row['timing_window'][1]


def test_every_event_has_source_and_new_internal_timings_stay_explicitly_estimated():
    disk = load('barmap_D')
    events = disk['sync']
    assert len({e['id'] for e in events}) == len(events)
    assert all(0 <= e['f'] < disk['frames'] for e in events)
    for row in events:
        assert_provenance(row)
    event = by_id(events)['crown_beacons']
    assert (event['f'], event['bar'], event['beat'], event['timing_status']) == (4320, 55, 1.0, 'estimated')
    for key, value, message in [('source', '', 'source provenance'), ('picture_frame', 4320, 'measured D frame'),
                                ('what', 'both beacons kindle', 'visible event description')]:
        bad = dict(event, **{key: value})
        with pytest.raises(AssertionError, match=message):
            assert_provenance(bad)


def test_cue_annotations_preserve_provenance_through_the_existing_loader():
    disk = by_id(load('barmap_D')['sync'])
    bm = BarMap('D')
    assert bm.d['status'] == 'treatment; new picture pending + cues'
    for event in bm.events:
        source = disk[event['id']]
        assert event['source'] == source['source']
        assert event['timing_status'] == source['timing_status']
        assert event['picture_frame'] is None
        if source['timing_status'] == 'estimated':
            assert event['lock'] == 'soft'
        if 'measured_ref' in source:
            assert event['measured_ref'] == source['measured_ref']


def assert_inherited_event(event):
    assert event['source_file'] in ('music/v3/barmap_C5P2.json', 'music/v3/barmap_AP2.json')
    source = by_id(json.loads((ROOT / event['source_file']).read_text())['sync'])[event['source_event']]
    assert event['source_clock'] == ('C5 edit' if 'C5P2' in event['source_file'] else 'A edit')
    assert event['source_frame'] == source['f']
    assert event['f'] == source['f'] + event['source_to_d'], 'inherited frame must use its declared remap'
    assert event['timing_status'] == 'inherited_' + source['source']
    assert event['source_original'] == source['source']
    assert event['source_what'] == source['what']
    assert 'not remeasured' in event['source']
    if 'measured_ref' in source:
        assert event['measured_ref'] == source['measured_ref']
        assert (ROOT / event['measurement_file']).is_file()


def test_inherited_measurements_name_actual_c_sources_and_remap_exactly():
    events = by_id(load('barmap_D')['sync'])
    for row in events.values():
        if 'source_event' in row:
            assert_inherited_event(row)
    catch = events['last_beacon']
    assert (catch['f'], catch['bar'], catch['source_frame'], catch['measured_ref']) == (
        5600, 71, 3786, 'map.last_catch.first')
    row = next(r for r in EDL.D if r['f0'] <= catch['f'] < r['f1'])
    assert row['takes'][0]['stem'] == 'cand_map_beacon-falloff'
    assert EDL.source_frame(row['takes'][0], catch['f']) == catch['source_frame']
    with pytest.raises(AssertionError, match='declared remap'):
        assert_inherited_event(dict(catch, f=5526))  # stale pre-insert offset +1740
    assert events['all_lit']['f'] == 5605
    assert events['refusal_offering_hand']['f'] == 3560
    assert events['refusal_raised_hand']['f'] == 3640
    assert events['refusal_burn_fastest']['f'] == 3771
    title = events['title_first_light']
    assert title['source_frame'] == 5702
    assert title['render_source'] == {'stem': 'book_C', 'frame': 6982}
    title_row = next(r for r in EDL.D if r['f0'] <= title['f'] < r['f1'])
    assert EDL.source_frame(title_row['takes'][0], title['f']) == title['render_source']['frame']


def test_map_holds_insert_and_caption_catch_order_match_the_committed_edit():
    events = by_id(load('barmap_D')['sync'])
    assert [events[k]['f'] for k in ('holdout_insert', 'map_dark_hold', 'map_resumes', 'last_beacon',
                                   'map_all_lit_hold', 'in_step')] == [5480, 5520, 5554, 5600, 5630, 5680]
    assert events['holdout_first_contact']['f'] == 5496
    assert events['holdout_first_contact']['timing_status'] == 'estimated'
    assert events['hammer_alone']['f'] == 5556
    assert events['hammer_alone']['timing_status'] == 'inherited_derived'
    assert '94 frames' in events['hammer_alone']['why']
    text = by_id(load('barmap_D')['text'])
    assert (text['D13']['f_in'], text['D13']['f_out']) == (5612, 5816)
    assert events['last_beacon']['f'] + 12 == text['D13']['f_in']
    assert text['D13']['f_in'] < events['in_step']['f'] < text['D13']['f_out']


def test_d_captions_are_the_live_table_and_retired_story_text_is_absent():
    disk = load('barmap_D')
    expected = copy.deepcopy(disk['text'])
    assert by_id(expected)['D14']['set'] == 'lower'  # authored fallback is still available to a missing plate
    by_id(expected)['D14']['set'] = 'in_picture'  # owner's complete D23 adoption supplies the same words
    live = json.loads(json.dumps(titles.text_table('D')))
    assert expected == live
    for ident, field, value in [('D14', 'set', 'lower'), ('D13', 'set', 'in_picture'),
                                ('D14', 'f_in', 5865), ('D14', 'line', 'Changed words.')]:
        wrong = copy.deepcopy(expected)
        by_id(wrong)[ident][field] = value
        with pytest.raises(AssertionError):
            assert wrong == live  # no extra placement, timing or wording difference is permitted
    assert len(disk['text']) == 21
    assert by_id(disk['text'])['D07']['f_in'] == 2772
    assert (by_id(disk['text'])['D02']['f_in'], by_id(disk['text'])['D02']['f_out']) == (684, 932)
    for filename in ('barmap_D', 'cues_D'):
        text = json.dumps(load(filename)).lower()
        for retired in ('scaffold', 'the eye onto nothing', 'wise refused the ring', 'forges go cold',
                        'flint_black', 'cold stone', 'promise to stop,',
                        'none knew how to master', 'each beacon was a promise to go no closer',
                        'ours hung there, unfinished', 'trusted us with the last pages'):
            assert retired not in text


def test_opening_preserves_ap2_pass_one_labels_without_claiming_new_measurements():
    expected = [e for e in load('barmap_AP2')['sync'] if e['f'] < 1440]
    events = by_id(load('barmap_D')['sync'])
    for source in expected:
        row = events['ap2_' + source['id']]
        assert_inherited_event(row)
        assert row['f'] == source['f']
        assert row['source_original'] == 'pass 1'
        assert row['timing_status'] == 'inherited_pass 1'
        assert 'measured_ref' not in row and 'measurement_file' not in row
    bad = dict(events['ap2_ignition'], timing_status='inherited_measured')
    with pytest.raises(AssertionError):
        assert_inherited_event(bad)


def test_crossing_source_selection_preserves_ap2_measurement_scope():
    events = by_id(load('barmap_D')['sync'])
    expected = [e for e in load('barmap_AP2')['sync'] if 5180 <= e['f'] < 5580]
    for source in expected:
        row = events['crossing_' + source['id']]
        assert_inherited_event(row)
        assert row['source_to_d'] == 1460
        if source['source'] == 'measured':
            assert row['measurement_file'] == 'music/v3/events_AP2_measured.json'
            assert 'camera-projection' in row['measurement_scope']
            assert 'pixels do not alone resolve an onset frame' in row['measurement_scope']
        shot = next(s for s in EDL.D if s['f0'] <= row['f'] < s['f1'])
        assert EDL.source_frame(shot['takes'][0], row['f']) == source['f']
    assert events['crossing_walk_setoff']['f'] == 6640
    assert events['crossing_walk_full']['f'] == 6736
    assert events['crossing_sky_wheels']['timing_status'] == 'inherited_pass 1'
    assert 'no spatial match claimed' in events['crossing_lantern_handover']['what']
    assert 'crossing_lantern_match' not in events
    with pytest.raises(AssertionError, match='declared remap'):
        assert_inherited_event(dict(events['crossing_walk_setoff'], f=6420))


def assert_burn_receipt(receipt):
    burn = receipt['burn']
    assert burn['sample_grid'] == [480, 201]
    assert burn['native_frame'] == [1920, 804]
    assert [r['frame'] for r in burn['frames']] == list(range(1680, 1760))
    assert burn['first_open_on_sample_grid'] == next(
        r['frame'] for r in burn['frames'] if r['hole_fraction'] > 0), 'cover-grid onset must match samples'
    assert burn['first_fully_gone_on_sample_grid'] == next(
        r['frame'] for r in burn['frames'] if r['page_remaining_max'] == 0), 'completion must match samples'
    assert all(r['page_remaining_max'] == 0 for r in burn['frames']
               if r['frame'] >= burn['first_fully_gone_on_sample_grid'])


def test_burn_receipt_distinguishes_parameter_cover_samples_and_future_plate():
    receipt = load('events_D_measured')
    assert_burn_receipt(receipt)
    assert len(receipt['source_sha256']) == len(receipt['collector_sha256']) == 64
    assert receipt['source_basename'] == 'D_burn_coverage_all80.json'
    transition = next(t for t in EDL.TRANS['D'] if t['kind'] == 'ring_burn')
    for key, value in receipt['parameters'].items():
        assert value == json.loads(json.dumps(transition[key]))
    assert receipt['parameters']['t_open'] == 1682
    events = by_id(load('barmap_D')['sync'])
    for ident, ref, frame in [('drawn_ring_burn_opens', 'first_open_on_sample_grid', 1683),
                              ('drawn_ring_burn_page_gone', 'first_fully_gone_on_sample_grid', 1748)]:
        row = events[ident]
        assert row['f'] == row['mask_frame'] == receipt['burn'][ref] == frame
        assert row['picture_frame'] is None
        assert row['timing_status'] == 'measured_D_cover'
        assert row['measurement_file'] == 'music/v3/events_D_measured.json'
        assert row['measured_ref'] == 'burn.' + ref
        assert '480×201' in row['measurement_scope'] and 'not native-resolution' in row['measurement_scope']
    assert events['drawn_ring_burn_t_open']['timing_status'] == 'renderer_parameter'
    assert events['inscription_fire_centered']['timing_status'] == 'estimated'
    for key, wrong, message in [('first_open_on_sample_grid', 1682, 'onset'),
                                ('first_fully_gone_on_sample_grid', 1750, 'completion')]:
        bad = copy.deepcopy(receipt)
        bad['burn'][key] = wrong
        with pytest.raises(AssertionError, match=message):
            assert_burn_receipt(bad)


def test_pending_score_only_copies_attributed_opening_recipe_values():
    cues = load('cues_D')
    assert cues['score_status'] == 'pending'
    assert cues['audio'] == 'STAND-IN silence'
    assert set(cues['sections']) == {f'D{k}' for k in range(8)}
    for section in cues['sections'].values():
        assert 'rel' not in section and 'level' not in section
    for key in ('extra_events', 'sfx'):
        assert cues[key] == []
    ap2 = load('cues_AP2')
    for kind in ('breaths', 'ambience'):
        assert len(cues[kind]) == 1
        row = cues[kind][0]
        expected = copy.deepcopy(ap2[kind][row['source_index']])
        for field in ('t0', 't1', 't'):
            if isinstance(expected.get(field), dict) and 'at' in expected[field]:
                expected[field]['at'] = 'ap2_' + expected[field]['at']
        assert row['source_file'] == 'music/v3/cues_AP2.json'
        assert 'unchanged opening reference' in row['source']
        assert {k: v for k, v in row.items() if k not in ('source', 'source_file', 'source_index')} == expected
    for sid, reference in cues['inherited_opening_sections'].items():
        assert {k: v for k, v in reference.items() if k != 'source'} == ap2['sections'][sid]
    for name in ('barmap_D', 'cues_D', 'events_D_measured'):
        raw = (ROOT / f'music/v3/{name}.json').read_text()
        assert '/Users/' not in raw and '/home/' not in raw
