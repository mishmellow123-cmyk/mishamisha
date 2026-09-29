"""Delivered pixel evidence cannot silently turn a failed control into a musical move."""
import copy
import json

import pytest

import barmap_c5p2 as B
import measure_c5_events as M
from measure_c5_analysis import onset_event, ramp_event, control


def test_empty_control_is_not_negative_evidence():
    with pytest.raises(ValueError, match='incomplete control'):
        control([], 'signal', (1, 4), 10)


def test_constant_ramp_cannot_manufacture_an_event():
    assert M.ramp([0, 0, 0], 560) == (None, None, None, None)
    rows = [dict(f=f, signal=5) for f in range(560, 600)]
    e = ramp_event('letters', rows, 'test', 'signal', (560, 599), (560, 569),
                   (590, 599), 'test', 'test', [0, 0, 1, 1])
    assert e['status'] == 'unresolved' and e['frames']['first'] is None


def test_real_signal_rejected_when_absent_window_also_fires():
    rows = [dict(f=f, signal=20 if f in (561, 580) else 0) for f in range(560, 600)]
    e = onset_event('letters', rows, 'test', 'signal', (575, 599), (560, 569), 10,
                    'test', 'test', [0, 0, 1, 1])
    assert e['negative_controls'][0]['triggers'] == 1
    assert e['frames']['first'] is None and e['status'] == 'unresolved'


def test_barmap_rejects_an_accepted_event_with_failed_control(monkeypatch):
    load = B._load
    table = copy.deepcopy(load(B.MEASURED))
    event = next(e for e in table['events'] if e.get('sync_id') == 'letters_glow')
    event['negative_controls'][0]['triggers'] = 1
    monkeypatch.setattr(B, '_load', lambda p: table if p == B.MEASURED else load(p))
    with pytest.raises(ValueError, match='failed negative control'):
        B.build()


def test_every_unmoved_cue_has_a_reason_and_every_new_move_has_clean_controls():
    bm, _ = B.build()
    table = B._load(B.MEASURED)
    events = {e['id']: e for e in table['events']}
    for sync in bm['sync']:
        if sync['source'] == 'pass 1':
            assert sync.get('why'), sync['id']
        elif sync['source'] == 'measured':
            eid = sync['measured_ref'].rsplit('.', 1)[0]
            event = events[eid]
            if event['shot'] in M.EXTRA_SHOTS:
                assert event['negative_controls']
                assert all(c['triggers'] == 0 and c['frames_examined'] > 0 for c in event['negative_controls'])
    assert table['not_on_this_mac'] == []


def test_helper_edit_invalidates_provenance():
    table = copy.deepcopy(B._load(B.MEASURED))
    table['generator_dependencies']['measure_c5_analysis.py'] = '0' * 64
    assert not M.is_current(table)
