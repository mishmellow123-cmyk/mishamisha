"""Opt-in effects bridges for the adopted R8 edit; gains are authored."""
from copy import deepcopy

TRANSITION_REQUEST = dict(id='D.new.unfinished.burn', beat=24,
                          design='ring_burn', kind='event')
WINDOWS = {4080: (4068, 4092, 'dissolve'), 4240: (4234, 4246, 'dissolve'),
           4560: (4554, 4566, 'dissolve'), 5840: (5828, 5852, 'dissolve'),
           6080: (6080, 6120, 'ring_burn')}


def adopted_windows(edl):
    result = {}
    for cut, expected in WINDOWS.items():
        rows = [t for t in edl['transitions'] if t.get('cut') == cut]
        if len(rows) != 1 or tuple(rows[0].get(k) for k in ('f0', 'f1', 'kind')) != expected:
            raise ValueError('effects bridges require adopted R8 window: ' + str(cut))
        result[cut] = deepcopy(rows[0])
    burn = result[6080]
    if (burn.get('center'), burn.get('t_open'), burn.get('speed'), burn.get('seed')) != ([1013., 187.], 6082., 10., 12):
        raise ValueError('new burn construction changed; remeasure before bridging')
    if any(t.get('cut') == 6640 for t in edl['transitions']):
        raise ValueError('crossing bridge requires the adopted straight cut')
    return result


def apply(events, beds, reuse, overlay, edl):
    windows = adopted_windows(edl)
    all_rows = events + beds + reuse['EXTRA_EVENTS'] + reuse['BED_CROPS']
    by_id = {r['id']: r for r in all_rows}
    changes = []

    def author(row, cut, reason, **updates):
        before = {k: deepcopy(row.get(k)) for k in updates}
        row.update(updates)
        note = dict(cut=cut, authored=True, reason=reason,
                    picture_window=windows.get(cut), previous=before,
                    scope='Authored acoustic bridge; no new measured contact or image-to-level equivalence.')
        row.setdefault('bridge_provenance', []).append(note)
        changes.append(dict(id=row['id'], **deepcopy(note)))

    author(by_id['D.06.C5.riffle'], 1440,
           'Soften the interior riffle rise after the protected minute; its inherited1446 marker stays.',
           post_gain_f=[[1440, -9], [1446, -9], [1450, -5], [1458, -5], [1470, -3], [1482, 0]])

    # Extend D's own copy of the donor bed. Neither donor source nor seed is edited.
    def extend_donor(rid, cut, new_end, fade_frames, reason):
        row = by_id[rid]
        delta = new_end-row['f1']
        original = deepcopy(row['donor_row'])
        original['f1'] += delta
        original['fade_out'] = fade_frames/24
        author(row, cut, reason, f1=new_end,
               donor_crop=[row['donor_crop'][0], row['donor_crop'][1]+delta], donor_row=original)

    extend_donor('D.15.C5.hearth.refusal', 3760, 3786, 36,
                 'Carry the quiet page hearth through the existing burn instead of ending at its midpoint.')
    author(by_id['D.new.trap.fire'], 3760,
           'Let the forge enter with the inherited burn reveal, reaching its bed level at the window end.',
           f0=3756, fade_in_s=30/24)
    for row in events:
        if row['request_id'] == 'D.new.trap.giants' and row['hit_f'] == 3760:
            author(row, 3760, 'The native opening contact is initially covered by the page; gain follows an authored reveal curve.',
                   post_gain_f=[[3750, -24], [3760, -18], [3770, -6], [3780, 0]])
    for rid in ('D.new.trap.fire', 'D.new.trap.surges'):
        author(by_id[rid], 4080, 'Carry the outgoing texture through the dissolve.',
               f1=4092, fade_out_s=24/24)

    author(by_id['D.new.glow.wind'], 4080, 'Bring in ridge wind through the adopted dissolve.',
           f0=4068, fade_in_s=24/24)
    # Both crops sample one seeded source at the same absolute sound clock.
    # Their sin-squared fades sum smoothly; the incoming target is3dB lower.
    author(by_id['D.new.glow.wind'], 4560, 'Continue the same wind source through the fire match.',
           f1=4566, fade_out_s=12/24, source_span=[4068, 5180], source_seed='D.new.glow.wind')
    author(by_id['D.new.ridges.wind'], 4560, 'Incoming crop of the same continuous wind, with its existing lower level.',
           f0=4554, fade_in_s=12/24, source_span=[4068, 5180], source_seed='D.new.glow.wind')
    author(by_id['D.new.crowns.fire'], 4240, 'Ease the working-fire bed through the glow-to-Ring match; kindles stay4320.',
           f0=4234, fade_in_s=12/24)
    author(by_id['D.new.crowns.fire'], 4560, 'Keep a short fire tail across the paired-fire dissolve.',
           f1=4566, fade_out_s=12/24)
    author(by_id['D.new.instep.fire'], 5840, 'Let the working fire lose scale across the book dissolve.',
           f1=5852, fade_out_s=24/24)
    author(by_id['D.new.oldfire.hearth'], 5840, 'Low hearth enters under the page as the forge recedes.',
           f0=5828, fade_in_s=24/24)

    burn = by_id[TRANSITION_REQUEST['id']]
    if burn.get('source') != 'measured' or 'end_provenance' not in burn:
        raise ValueError('new burn requires measured opening and clear-frame evidence')
    start, end = burn['hit_f'], burn['stop_f']
    if not 6080 <= start < end <= 6120:
        raise ValueError('new burn evidence lies outside the adopted transition')
    author(burn, 6080, 'Restrained paper texture starts at the measured opening and ends at measured clearance.',
           level_trim_db=-10, recipe_overrides=dict(pre=0., post=(end-start)/24,
                                                  fi=.008, fo=.15, send=0., no_breath=True))
    author(by_id['D.new.oldfire.hearth'], 6080, 'The held page keeps its hearth until the burn clears it.',
           f1=end, fade_out_s=(end-start)/24)
    author(by_id['D.new.lamps.forge'], 6080, 'The incoming forge grows under the measured aperture; level is authored.',
           f0=start, fade_in_s=(end-start)/24)

    extend_donor('D.25.C5.hearth.deep', 6640, 6652, 24,
                 'A short outgoing audio tail softens the adopted straight picture cut.')
    for rid in ('D.26.A.wind.crossing', 'D.26.A.watchfire_1'):
        row = by_id[rid]
        author(row, 6640, 'Read the preceding donor handle on the same A-to-D clock; quiet audio pre-lap before the straight cut.',
               f0=6628, donor_crop=[row['donor_crop'][0]-12, row['donor_crop'][1]],
               post_gain_f=[[6628, -120], [6634, -12], [6640, -6], [6652, 0]])
    author(by_id['D.new.crossing.creak.1'], 6640,
           'Restrain the estimated lantern-motion accent at the incoming straight cut.',
           post_gain_f=[[6636, -9], [6640, -6], [6652, 0]])
    return dict(revision='r8', picture_windows=windows, changes=changes,
                unaffected_joins=[2080, 3520, 5480, 5520],
                note='No discrete anchor moved; the new measured burn is additive. Every gain/fade remains authored.')
