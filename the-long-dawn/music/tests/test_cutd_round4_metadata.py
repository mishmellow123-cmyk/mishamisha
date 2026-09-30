"""Round 4 transition receipts keep measured cover, source clocks and renderer parameters distinct."""
from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'edit'))
import edl_v3 as EDL


def load(name):
    return json.loads((ROOT / f'music/v3/{name}.json').read_text())


def assert_ridge_receipt(burn):
    assert burn['sample_grid'] == [480, 201]
    assert burn['native_frame'] == [1920, 804]
    assert burn['thresholds'] == {'open_cover_below': .999, 'remaining_cover_above': .001}
    assert [r['frame'] for r in burn['frames']] == list(range(5180, 5220)), 'every cover frame must be sampled'
    assert burn['first_open_on_sample_grid'] == next(r['frame'] for r in burn['frames'] if r['open_pixels']), 'onset must match samples'
    assert burn['first_fully_gone_on_sample_grid'] == next(r['frame'] for r in burn['frames'] if not r['remaining_cover_pixels']), 'completion must match samples'
    assert all(r['remaining_cover_pixels'] == 0 for r in burn['frames'] if r['frame'] >= burn['first_fully_gone_on_sample_grid'])


def test_ridge_all_frame_measurement_and_native_origin_match_the_active_transition():
    burn = load('events_D_round4_measured')['ridge_burn']
    assert_ridge_receipt(burn)
    transition = next(t for t in EDL.TRANS['D'] if t.get('cut') == 5180)
    assert burn['parameters'] == json.loads(json.dumps(transition))
    assert (burn['parameters']['t_open'], burn['first_open_on_sample_grid'],
            burn['first_fully_gone_on_sample_grid'], burn['remains_gone_through']) == (5182, 5183, 5212, 5219)
    assert burn['origin']['source'] == 'reveal_A/f_03799.jpg'
    assert burn['origin']['bbox'] == [1126, 600, 19, 39]
    assert burn['origin']['chosen_origin'] == burn['parameters']['center'] == [1136, 623]
    assert 'width, height' in burn['origin']['bbox_format']
    for obj in (burn, burn['origin']):
        assert len(obj['source_sha256']) == len(obj['collector_sha256']) == 64
    assert 'not native-resolution or finished-picture timing' in burn['limitations']
    for key, wrong, message in [('first_open_on_sample_grid', 5182, 'onset'),
                                ('first_fully_gone_on_sample_grid', 5211, 'completion')]:
        bad = deepcopy(burn)
        bad[key] = wrong
        with pytest.raises(AssertionError, match=message):
            assert_ridge_receipt(bad)
    bad = deepcopy(burn)
    bad['frames'].pop()
    with pytest.raises(AssertionError, match='every cover frame'):
        assert_ridge_receipt(bad)


def test_ridge_event_annotations_share_the_receipt_without_promoting_cover_to_picture():
    events = {r['id']: r for r in load('barmap_D')['sync']}
    cues = load('cues_D')['events']
    for ident, frame, ref in [('last_kingdom_burn', 5183, 'first_open_on_sample_grid'),
                              ('last_kingdom_burn_ridge_gone', 5212, 'first_fully_gone_on_sample_grid')]:
        row = events[ident]
        assert row['f'] == row['mask_frame'] == frame
        assert row['timing_status'] == 'measured_D_cover'
        assert row['picture_frame'] is None
        assert row['timing_window'] == [5180, 5220]
        assert row['measurement_file'] == 'music/v3/events_D_round4_measured.json'
        assert row['measured_ref'] == 'ridge_burn.' + ref
        assert '480×201' in row['measurement_scope']
        assert 'not native-resolution' in row['measurement_scope']
        assert row['source'] == cues[ident]['source']
        assert row['mask_frame'] == cues[ident]['mask_frame']
    parameter = events['last_kingdom_burn_t_open']
    assert parameter['f'] == 5182 and parameter['timing_status'] == 'renderer_parameter'
    assert 'mask_frame' not in parameter
    assert cues[parameter['id']]['kind'] == 'parameter'
    raw = (ROOT / 'music/v3/events_D_round4_measured.json').read_text()
    assert '/Users/' not in raw and '/home/' not in raw


def assert_deep_source_receipt(source):
    assert source['sample_grid'] == [1920, 804]
    assert [r['frame'] for r in source['frames']] == list(range(2945, 2981)), 'all 36 selected masks required'
    for row in source['frames']:
        assert row['source_frame'] == round(1905 + (row['frame'] - 2945) * 86 / 35), 'source remap must match'
        assert len(row['rgb_sha256']) == len(row['cover_sha256']) == 64
    first = next(r for r in source['frames'] if r['cover_mean'] < 1)
    assert source['first_open_on_selected_source_masks'] == first['frame'], 'source-mask onset must match'
    assert source['first_open_source_frame'] == first['source_frame']
    assert source['first_fully_gone_on_selected_source_masks'] is None, 'raw source never becomes fully clear'
    assert all(r['cover_max'] > 0 for r in source['frames'])


def test_deep_resampling_preserves_raw_measurements_separately_from_tail_removal():
    source = load('events_D_round4_measured')['deep_exit_source']
    assert_deep_source_receipt(source)
    assert (source['first_open_on_selected_source_masks'], source['first_open_source_frame']) == (2953, 1925)
    assert source['visible_glow_onset'] is None
    assert source['frames'][-1]['cover_mean'] == 0.009283720795532142
    assert source['frames'][-1]['cover_max'] == 255
    for mutation, message in [(lambda s: s.update(first_open_on_selected_source_masks=2945), 'source-mask onset'),
                              (lambda s: s['frames'][8].update(source_frame=1924), 'source remap'),
                              (lambda s: s.update(first_fully_gone_on_selected_source_masks=2980), 'raw source'),
                              (lambda s: s['frames'].pop(), 'all 36')]:
        bad = deepcopy(source)
        mutation(bad)
        with pytest.raises(AssertionError, match=message):
            assert_deep_source_receipt(bad)
    events = {r['id']: r for r in load('barmap_D')['sync']}
    row = events['brink_burnthrough']
    assert (row['f'], row['source_frame'], row['timing_window']) == (2953, 1925, [2945, 2981])
    assert row['timing_status'] == 'measured_source_cover' and row['picture_frame'] is None
    assert row['measurement_file'] == 'music/v3/events_D_round4_measured.json'
    assert row['measured_ref'] == 'deep_exit_source.first_open_on_selected_source_masks'
    assert load('cues_D')['events'][row['id']]['source'] == row['source']


def test_native_deep_tail_receipt_measures_same_frame_identity_before_finish():
    receipt = load('events_D_round4_measured')['deep_exit_tail']
    transition = next(t for t in EDL.TRANS['D'] if t['kind'] == 'deep_reveal')
    assert receipt['tail_clear'] == list(transition['tail_clear']) == [2976, 2980]
    assert receipt['native_frame'] == [1920, 804]
    assert [r['frame'] for r in receipt['frames']] == list(range(2976, 2982))
    rows = receipt['frames'][:-1]
    assert [r['weight'] for r in rows] == [1., .84375, .5, .15625, 0.]
    assert all(r['original_cover_max'] == 1. for r in rows)
    assert rows[0]['changed_pixels'] == 0
    assert rows[-1]['effective_cover_max'] == rows[-1]['after_vs_incoming_max'] == 0.
    assert all(r['effective_cover_max'] > 0 for r in rows[:-1])
    assert rows[-1]['frame'] == receipt['clear_frame_before_finish'] == 2980
    assert receipt['first_modified_frame'] == 2977
    event = next(r for r in load('barmap_D')['sync'] if r['id'] == 'brink_burn_clear')
    assert event['f'] == 2980 and event['timing_status'] == 'measured_D_composite'
    assert event['measured_ref'] == 'deep_exit_tail.clear_frame_before_finish'
    assert 'before finish' in event['measurement_scope']
    # Removing only the cover leaves the original RGB remnant, unlike the observed paired envelope.
    assert rows[-1]['before_vs_incoming_max'] > 0
