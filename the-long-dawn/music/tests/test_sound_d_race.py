"""Round5 cadence and acoustic separation, with deliberately synthetic landmarks."""
from copy import deepcopy
from pathlib import Path

import pytest
import sound_d_binding as B
import sound_d_race as R


def rows():
    plan = B.build()
    return plan['new_events'], plan['new_beds']


def test_renderer_clock_is_pinned_and_changed_clock_rejected(tmp_path):
    assert R.work_clock() == tuple(range(2400, 2640, 20)) + tuple(range(2640, 2721, 10))
    p = tmp_path / R.CLOCK_FILE
    p.parent.mkdir(parents=True)
    p.write_text((R.ROOT / R.CLOCK_FILE).read_text().replace('b + 1, 10', 'b + 1, 20'))
    with pytest.raises(ValueError, match='work clock changed'):
        R.work_clock(tmp_path)


def test_round5_cannot_use_unremeasured_giants():
    events, beds = rows()
    with pytest.raises(ValueError, match='freshly validated'):
        R.apply(events, beds, {'series': {}})
    with pytest.raises(ValueError, match='unknown race'):
        B.build(race_revision='uninspected')


def test_work_clock_fallback_is_estimated_and_accelerates_without_hidden_tick():
    events, beds = rows()
    other_before = deepcopy([r for r in events + beds if not r['request_id'].startswith('D.new.race.')])
    result = R.apply(events, beds, {'series': {'D.new.race.giants': {}}})
    field = [r for r in events if r['request_id'] == 'D.new.race.anvils']
    near = [r for r in events if r['request_id'] == 'D.new.race.giants']
    assert [r['hit_f'] for r in field] == list(R.work_clock()[:-1])
    assert all(r['source'] == 'est.' and r['picture_frame'] is None for r in field)
    assert all(r['stop_f'] == 2720 for r in field + near)
    assert result['field_pulses'] == 20 and result['work_clock']['omitted_frame'] == 2720
    assert all(r['dist'] > near[0]['dist'] and r['level_trim_db'] <= near[0]['level_trim_db']-10 for r in field)
    assert all(len(r['recipe_overrides']['layers']) == 4 for r in field)
    assert all(r['texture_provenance']['layer_count'] == 4 for r in field)
    assert other_before == [r for r in events + beds if not r['request_id'].startswith('D.new.race.')]
    fire = next(r for r in beds if r['request_id'] == 'D.new.race.fire')
    assert fire['f1'] == 2720
    assert all(a[0] < b[0] for a, b in zip(fire['env_f'], fire['env_f'][1:]))
    assert all(p['envelope_offsets_f'] == [0, 2, 6] and p['gain_source'].startswith('Authored')
               for p in fire['envelope_provenance'])


def test_measured_work_responses_survive_design_and_drive_authored_surges():
    events, beds = rows()
    field = [r for r in events if r['request_id'] == 'D.new.race.anvils']
    for row in field:
        row.update(hit_f=row['hit_f'] + 1, source='measured', picture_frame=row['hit_f'] + 1,
                   measurement_scope='native_delivered_plate', measured_ref='synthetic fixture only',
                   measurement={'claim': 'onset', 'predicate': 'synthetic field response'})
    observed = [(r['hit_f'], deepcopy(r['measurement'])) for r in field]
    R.apply(events, beds, {'series': {'D.new.race.giants': {}, 'D.new.race.anvils': {}}})
    assert [(r['hit_f'], r['measurement']) for r in field] == observed
    fire = next(r for r in beds if r['request_id'] == 'D.new.race.fire')
    assert [p['frame'] for p in fire['envelope_provenance']] == [2400] + [r['hit_f'] for r in field]
    assert all(p['source'] == 'measured' and p['measurement_scope'] == 'native_delivered_plate'
               for p in fire['envelope_provenance'][1:])
    assert fire['envelope_provenance'][0]['source'] == 'est.'


def test_round5_rejects_an_unadopted_race_even_before_loading_evidence():
    import json
    edl = json.loads((R.ROOT / "edit/edl/edl_D.json").read_text())
    next(s for s in edl["shots"] if s["code"] == "D10")["takes"][0]["stem"] = "synthetic_wrong_take"
    with pytest.raises(ValueError, match="adopted native race"):
        B.build(edl=edl, race_revision="round5")
