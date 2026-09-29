"""Render the review from generated tables; no hand-entered measurement values."""
import collections
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
load = lambda p: json.loads(p.read_text())
table = load(ROOT/'music/v3/events_C5_measured.json')
bm = load(ROOT/'music/v3/barmap_C5P2.json')
old = {s['id']: s for s in load(ROOT/'music/v3/barmap_C5.json')['sync']}
events = {e['id']: e for e in table['events']}
sound = {e['id']: e for e in load(ROOT/'music/sound/c5_sound_events.json')['events']}
counts = collections.Counter(s['source'] for s in bm['sync'])
clean = lambda s: str(s).replace('|', '/').replace('\n', ' ')
lines = ['# C5 delivered-picture synchronization audit', '',
 'Continued the interrupted lane in its existing worktree. Existing contact sheets and candidate detectors were inspected; every shot was re-read from delivered frames. No audio was rendered. All frame numbers below are absolute C5 cut frames at 24 fps.', '',
 '## Result and limits', '',
 f'{len(bm["sync"])} sync ids: '+', '.join(f'{n} {k}' for k,n in sorted(counts.items()))+'.', '',
 'The added sections use `assemble.Ctx.picture` at 960×402, before caption overlay; source selection uses the exported EDL and EDIT’s `plan_shot`/`locate`/`index`/`chain` implementation. Composite layers are checked for presence, and each source file is hashed. The previously measured nine shots retain their native/half-resolution detector conventions. No frames were written under renders/.', '',
 'FLINT has two measured spark flashes and no black frame. Catch and blowing onset remain unresolved: the candidate detectors also trigger on glowing tinder or the strike flash. A visible flame develops later, but these measurements do not establish its first frame; no substitute onset is invented. The seven run catches agree with pass 1. The sun is already partly above the ridge at the illumination cut, so its onset is left-censored. PLENTY is a continuous camera transition; title lettering first becomes detectable after its section entry.', '',
 'Thresholded onsets are operational pixel measurements, not claims of the earliest perceptible photon. The page catch is scoped after the flying glyph disappears; earlier glyph sparks also satisfy the same brightness mask. Run fastest/full remain null because camera-registered glow novelty does not measure flame-size growth. Medium confidence reflects these scene-specific masks and downsampling. No motion playback or listening judgment is claimed.', '',
 '## Every sync id', '',
 'Delta is the selected C5P2 cue minus pass 1. “moves the music” flags every absolute delta ≥2 frames; the score’s actual use of a cue is separately checked below. Added pass-2 ids have no pass-1 frame or delta. A retained cue’s frame is not a measured frame.', '',
 '| Sync id | Pass 1 | Measured frame(s) / evidence | C5P2 | Delta | Method / retention reason | Confidence |',
 '|---|---:|---|---:|---|---|---|']
for s in sorted(bm['sync'],key=lambda s:(old.get(s['id'],s)['f'],s['id'])):
    ref=s.get('measured_ref')
    if ref:
        eid,field=ref.rsplit('.',1)
        e=events[eid]
        frames=clean(json.dumps(e['frames'],separators=(',',':')))
        method=clean(eid+': '+e['method']+'; selected '+field)
        confidence=e['confidence']
        if s['id']=='leaders_reach':method+='; rounded mean of left/right full frames (inherited placement)'
        if s['id']=='pulls_ahead':method+='; earlier of left/right first frames'
    elif s['source']=='verified':
        eid=s['verified_by'].split()[0];e=events[eid]
        frames=clean(json.dumps(e['frames'],separators=(',',':')))
        method=clean(e['method']+'; '+s['why']);confidence=e['confidence']
    else:
        frames='—';method=clean(s.get('why',s.get('derived_from','')))
        confidence='not measured' if s['source']=='pass 1' else 'musical derivation'
    f0=old.get(s['id'],{}).get('f')
    delta=None if f0 is None else s['f']-f0
    dt='new id' if delta is None else f'{delta:+d}'+(' **moves the music**' if abs(delta)>=2 else '')
    lines.append(f"| {s['id']} | {f0 if f0 is not None else '—'} | {frames} | {s['f']} | {dt} | {method} | {confidence} |")
lines += ['', '## FLINT cross-check against SOUND’s derived mapping', '',
 '| Event | SOUND derived | Pixel first | Delta | Result |','|---|---:|---:|---:|---|']
for k,eid in [('strike1','flint.strike1'),('strike3','flint.strike3'),('blow','flint.blow_test'),('catch','flint.catch_test')]:
    f=sound['C5.flint.'+k]['hit_f'];e=events[eid];m=e['frames'].get('first')
    lines.append(f"| {k} | {f} | {m if m is not None else 'unresolved'} | {m-f if m is not None else '—'} | {clean(e.get('note','Agrees at the measured spark onset; peak is recorded separately.'))} |")
e=events['flint.black_test']
lines += ['', f"Black detector: {len(e['frames']['found'])} black frames / {e['evidence']['frames_examined']} examined; minimum frame-maximum luma {e['evidence']['minimum_frame_max_luma']:.6f}, threshold ≤2. SOUND’s references to a cut ‘to black’ disagree with the delivered picture. Unresolved blow/catch are unverified, not measured disagreements.", '',
 '## Negative controls', '',
 'Each row applies the stated detector threshold to a delivered interval without the target event. Counts include the denominator beside every zero. Failed controls are retained. A clean interval validates only that interval and threshold, not universal specificity. Ramp controls use the event’s fixed baseline/top, never a fresh fit to the control. Inherited argmax/peak controls are identified explicitly; these narrower checks must not be confused with a general semantic-event classifier.', '',
 '| Event / detector | Negative interval | Triggers / frames examined | Threshold / direction | Interpretation |',
 '|---|---|---|---|---|']
for e in table['events']:
    for c in e.get('negative_controls',[]) + e.get('diagnostic_controls',[]):
        lines.append(f"| {e['id']} / {c['detector']} | {c['first']}–{c['last']} | {c['triggers']} / {c['frames_examined']} | {clean(c.get('threshold'))}; direction {c.get('direction',1)} | {'TRIGGERS; see interpretation. ' if c['triggers'] else ''}{clean(c['selection_reason'])} |")
lines += ['', 'The illumination and PLENTY records are inspected coverage notes, with no accepted event onset; they are not successful detectors and have no claimed negative-control pass. The anvil impulse candidate at C1262 was visually rejected (adjacent stills show a rotating Ring). Exact numeric traces are in the `*_trace.json` files; accepted events carry their deciding samples and fixed thresholds in `events_C5_measured.json`.', '',
 'The inherited low-forge sink requires a subsequent <10% extinction crossing before assigning its preceding 90% onset; the 90% threshold alone also trips on baseline flicker (retained as a diagnostic control). WATCH produces two occlusion candidates across its full shot, so that tracker cannot establish ignition semantics; the opening interval supplies a separate clean control. Neither observation is hidden by a zero from a shorter interval.', '', '## Verification and close-out', '']
verify=HERE/'verification.json'
if verify.exists():
    v=load(verify)
    lines += [f'- {clean(item)}' for item in v['checks']]
    lines += ['',v['limitations'],'',v['commit']]
else:
    lines += ['Verification is pending; this draft is not a completion claim.']
lines += ['', '## Owner follow-up', '',
 'Review the moved cues against picture and sound before a new mix. FLINT catch/blow still need a reliable visible boundary or an explicit musical placement; the retained SOUND frames remain derived. SOUND’s existing “not on this Mac” verification text and black-cut descriptions are stale; this lane refreshes its source hashes for snapshot consistency but does not redesign effects. No EDL/source-selection edits, frame renders, audio renders, pushes, or merges were performed.', '']
(HERE/'REPORT.md').write_text('\n'.join(lines))
print('wrote REPORT.md;',len(bm['sync']),'sync rows')
