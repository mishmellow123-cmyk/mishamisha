"""Round 5 race sound design, applied after the picture evidence gate.

Work-clock parameters describe the renderer. Only the validated overlay can
promote their audible accompaniment to a measured picture response.
"""
import ast
from copy import deepcopy
import hashlib
from pathlib import Path

import sound_d_designs as D

ROOT = Path(__file__).resolve().parents[2]
CLOCK_FILE = 'shots/embers/c_d.py'


def work_clock(root=ROOT):
    """Fail closed if the inspected renderer clock changes; import no renderer."""
    source = (Path(root) / CLOCK_FILE).read_text()
    function = next((n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef)
                     and n.name == 'race_work_beats'), None)
    if function is None:
        raise ValueError('race work clock changed: function is missing')
    body = [n for n in function.body if not (isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant)
                                            and isinstance(n.value.value, str))]
    expected = ast.parse("a, b = D.shot_range('race')\nreturn tuple(range(a, b - 80, 20)) + tuple(range(b - 80, b + 1, 10))").body
    if ast.dump(ast.Module(body=body, type_ignores=[])) != ast.dump(ast.Module(body=expected, type_ignores=[])):
        raise ValueError('race work clock changed: remeasure and revise sound design')
    return tuple(range(2400, 2640, 20)) + tuple(range(2640, 2721, 10))


def apply(events, beds, overlay):
    """Modify caller-owned bound rows; keep all observation and donor histories."""
    ticks = work_clock()
    if 'D.new.race.giants' not in overlay['series']:
        raise ValueError('round5 requires a freshly validated native race giant series')
    visible = ticks[:-1]
    clock = dict(source=CLOCK_FILE + ':race_work_beats',
                 source_sha256=hashlib.sha256((ROOT / CLOCK_FILE).read_bytes()).hexdigest(),
                 parameter_frames=list(ticks), visible_plate_interval=[2400, 2720],
                 omitted_frame=2720,
                 omission_reason='The Deep holds race D2719; the source-clock tick2720 has no delivered race frame.')
    giant_rows = [r for r in events if r['request_id'] == 'D.new.race.giants']
    field_rows = [r for r in events if r['request_id'] == 'D.new.race.anvils']
    if 'D.new.race.anvils' not in overlay['series']:
        events[:] = [r for r in events if r['request_id'] != 'D.new.race.anvils']
        old = field_rows
        field_rows = []
        for frame in visible:
            row = deepcopy(min(old, key=lambda r: abs(r['hit_f'] - frame)))
            row.update(id=f'D.new.race.anvils.work.{frame}', hit_f=frame,
                       offset_f=frame-row['hook_frame'], source='est.', picture_frame=None,
                       measured_ref=None, spacing_basis='Renderer work clock; visible field response unmeasured',
                       timing_note='Authored field-work accompaniment to the inspected renderer clock; no measured hammer contact.')
            row.pop('spacing_f', None)
            field_rows.append(row)
        events.extend(field_rows)
    elif not any(r['hit_f'] == 2400 for r in field_rows):
        # The measured field is dimmer at the cut than forging2399; do not
        # manufacture a visible ignition. This opening texture is authored.
        row = dict(id='D.new.race.anvils.work.2400', request_id='D.new.race.anvils',
                   beat=10, design='dull_hammer', source='est.', hit_f=2400,
                   hook='race_beat_surges', hook_frame=2400, offset_f=0,
                   hook_timing_status='renderer_work_clock', hook_source=clock['source'],
                   picture_frame=None, measured_ref=None, pan=0., dist=0., level_trim_db=0.,
                   spacing_basis='Authored work texture at the race edit boundary',
                   timing_note='Opening field onset is unmeasured; native field brightness decreases across2399 to2400. The sound establishes already-live work.')
        field_rows.insert(0, row)
        events.append(row)
    for row in giant_rows + field_rows:
        row['race_work_clock'] = deepcopy(clock)
        row['stop_f'] = 2720
        row['design_revision'] = 'round5'
    for row in giant_rows:
        row.update(level_trim_db=-4, dist=.08,
                   recipe_overrides=dict(post=.48, fo=.16, lp=6500, send=.015, no_breath=True),
                   distance_basis='Authored near perspective for the two giants at the moving Ring ends.',
                   level_trim_basis='Authored restrained near-contact target, 4 dB below the inherited anvil target.',
                   race_role='near giant contact')
    base = D.build('dull_hammer')
    for i, row in enumerate(field_rows):
        # Small pitch/delay spreads are authored sound texture, not four more
        # measured contacts. Match the whole cluster to one low event target.
        layers = [dict(base, dt=delay, stretch=stretch, g=gain, post=.30, fo=.13,
                       hp=230, lp=1800, width=.8)
                  for delay, stretch, gain in ((0, .96, 0), (.032, 1.04, -2),
                                               (.073, 1.01, -3), (.119, .98, -5))]
        row.update(level_trim_db=-14, dist=.72, pan=(-.35, .3, -.18, .38)[i % 4],
                   recipe_overrides=dict(layers=layers, send=.025, no_breath=True),
                   distance_basis='Authored distant field perspective; image brightness does not measure acoustic distance.',
                   pan_basis='Authored restrained alternating field spread; no individual forge position asserted.',
                   level_trim_basis='Authored cluster target, 14 dB below the inherited anvil target, after matching the summed cluster.',
                   race_role='distant field work',
                   texture_provenance=dict(layer_delays_s=[0, .032, .073, .119],
                       layer_count=4, basis='Authored density, not a count of visible contacts'))
    fire = next(r for r in beds if r['request_id'] == 'D.new.race.fire')
    env, provenance = [], []
    for row in field_rows:
        frame = row['hit_f']
        for off, gain in ((0, -7), (2, 0), (6, -7)):
            if frame + off < 2720:
                env.append([frame + off, gain])
        provenance.append(dict(frame=frame, source=row['source'],
            measurement_scope=row.get('measurement_scope'), measured_ref=row.get('measured_ref'),
            picture_frame=row.get('picture_frame'),
            envelope_offsets_f=[0, 2, 6], gain_db=[-7, 0, -7],
            gain_source='Authored six-frame forge surge; the source renderer uses six-frame tower rises.'))
    env.append([2720, -14])
    fire.update(env_f=env, envelope_provenance=provenance, fade_in_s=.08, fade_out_s=.25,
                level_trim_db=-6, race_work_clock=deepcopy(clock), design_revision='round5',
                recipe_overrides=dict(lp=4200, width=.85, send=0, no_breath=True),
                race_role='continuous forge field with short work surges',
                level_trim_basis='Authored restrained bed target with six-frame pulses; no new fire ignition asserted.')
    return dict(revision='round5', work_clock=clock, giant_contacts=len(giant_rows),
                field_pulses=len(field_rows), final_bar_start=2640,
                sound_stop_exclusive=2720,
                scope='Measured row claims retain their exact pixel predicates; spatial treatment and surge envelopes are authored.')
