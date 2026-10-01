"""Opt-in sound treatment for the locked D picture; timing lives in the overlay."""
from copy import deepcopy


def apply(events, beds, overlay):
    """Continue the visible fire and accompany the retimed ink, without a page turn."""
    if 'D.new.page.to_blank' not in overlay['suppressions']:
        raise ValueError('locked picture requires an evidence-based page-turn suppression')
    if any(r['request_id'] == 'D.new.page.to_blank' for r in events):
        raise ValueError('obsolete page turn survived its suppression')
    forge = next(r for r in beds if r['request_id'] == 'D.new.forging.fire')
    ink = next(r for r in events if r['request_id'] == 'D.new.oldfire.quill')
    if ink.get('source') != 'measured':
        raise ValueError('locked picture requires a measured ink onset')
    for row in (forge, ink):
        if row['request_id'] not in overlay['bounds'] or 'end_provenance' not in row:
            raise ValueError('locked picture requires a measured end bound: ' + row['request_id'])
    if not 2720 < forge['f1'] <= 2758:
        raise ValueError('thinking-fire end must be measured inside the Deep entry')
    if not 5840 <= ink['hit_f'] < ink['stop_f'] <= 6080:
        raise ValueError('retimed ink accompaniment must remain inside D23')
    # One longer, identically seeded source has no fade-out/restart at2400.
    # Levels follow authored acoustic perspective, not image brightness.
    forge.update(fade_out_s=.35,
        env_f=[[forge['f0'], 0], [2400, 0], [2440, -8], [2719, -10], [forge['f1'], -18]],
        envelope_provenance=dict(source='authored gain envelope',
            reason='Thinking fire persists while the race camera reveals the field; it recedes under work and fades with the page occlusion.',
            end=deepcopy(forge['end_provenance']), levels_are_pixel_measurements=False),
        continuity_provenance=dict(previous_end_exclusive=2400, new_end_exclusive=forge['f1'],
            source_seed=forge['id'], source_streams=1, restart_at_2400=False,
            scope='Authored continuous source treatment; measured endpoint scope remains on end_provenance.'),
        picture_revision='locked')
    ink.setdefault('recipe_overrides', {}).update(pre=0., post=(ink['stop_f']-ink['hit_f'])/24,
                                                 fi=.03, fo=2/24)
    ink.update(timing_note='Authored quiet quill-texture accompaniment to a measured wet-ink reveal. No visible pen or physical pen contact is claimed.',
               picture_revision='locked')
    for row in events:
        if row['id'] == 'D.new.race.anvils.work.2400':
            row['timing_note'] = ('Authored opening work texture under the continuing thinking fire. '
                                  'The material/spark handoff is gradual; no new visible work onset is claimed at2400.')
    return dict(revision='locked', thinking_fire_end_exclusive=forge['f1'],
                caption_interval=[ink['hit_f'], ink['stop_f']],
                page_turn='suppressed by selected dissolve evidence',
                gains_and_fades='authored; measured start/end points remain in the evidence table')
