"""Read delivered D book plates; never render, rewrite or adopt a picture.

The flare detector measures the Ring's lettering in a settled-camera interval.
It does not measure the earlier fall, a source shader parameter, or finished
film pixels. Run the CLI through onepy. No media is retained except a small
labelled evidence sheet in the explicitly supplied output directory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import resource
import sys
import time

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
INPUTS = ('edit/edl/edl_D.json', 'music/v3/barmap_D.json', 'music/v3/cues_D.json',
          'music/v3/events_D_measured.json', 'music/v3/events_D_round4_measured.json')
ROIS = {'letters': (877, 446, 1006, 497), 'metal_control': (1008, 440, 1030, 465),
        'flame_control': (888, 527, 936, 560)}
FIRST, STOP = 5880, 5925
ROUND5_FLARE_STOP = 5922
CAPTION_ROIS = {'first_ink': (410, 700, 600, 755), 'terminal_period': (1472, 699, 1490, 716)}
INK_THRESHOLDS = {'strict': (110, 80, 55), 'primary': (120, 85, 60), 'loose': (130, 90, 65)}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def native_frame(root, stem, frame):
    """Decode the actual delivered file and reject a concurrent replacement."""
    relative = Path('renders') / stem / f'f_{frame:05d}.jpg'
    path = Path(root) / relative
    before = sha(path)
    with Image.open(path) as im:
        if im.size != (1920, 804):
            raise ValueError(f'wrong native dimensions: {relative}')
        rgb = np.array(im.convert('RGB'))
    if sha(path) != before:
        raise ValueError(f'frame changed during native read: {relative}')
    return rgb, dict(frame=frame, source_frame=frame, source_stem=stem,
                     path=relative.as_posix(), sha256=before)


def dark_count(rgb, thresholds):
    value = np.asarray(rgb)
    if value.ndim != 3 or value.shape[-1] != 3 or not np.isfinite(value).all():
        raise ValueError('finite RGB pixels required')
    return int(np.all(value < np.asarray(thresholds), axis=-1).sum())


def persistent_entry(rows, first, stop, key, *, floor):
    """A measured entry needs a complete negative prefix and positive hold."""
    if [r.get('frame') for r in rows] != list(range(first, stop)):
        raise ValueError('entry search missing, duplicate or disordered frames')
    hits = [r['frame'] for r in rows if r[key] >= floor]
    if (not hits or hits[0] < first + 3 or hits[0] + 2 >= stop
            or hits != list(range(hits[0], stop))):
        raise ValueError('entry lacks a clean prefix or sustained observed hold')
    return hits[0]


def counts(rgb):
    """Orange entry and subsequent bright yellow glyph pixels, in RGB bytes."""
    rgb = np.asarray(rgb, dtype=np.float32)
    if rgb.ndim != 3 or rgb.shape[2] != 3 or not np.isfinite(rgb).all():
        raise ValueError('finite RGB image required')
    r, g = rgb[..., 0], rgb[..., 1]
    return dict(red=int(((r > 180) & (r-1.4*g > 40)).sum()),
                light=int(((r > 240) & (g > 170)).sum()))


def detect_flare(rows, first=FIRST, stop=STOP, *, red_floor=50, light_floor=100):
    """Require a complete search, dark prefix, localized orange entry and glow.

    A bright fire alone is not a lettering flare. The adjacent metal control
    must stay below its equal-area normalized red threshold; the next two
    frames must show yellow lettering. Missing prior frames fail closed.
    """
    if [r.get('frame') for r in rows] != list(range(first, stop)):
        raise ValueError('missing, duplicate or disordered search frames')
    found = [i for i, r in enumerate(rows) if r['letters']['red'] >= red_floor]
    if not found:
        return None
    index = found[0]
    if index < 5 or index+2 >= len(rows):
        raise ValueError('flare lacks a dark prefix or confirmed following frames')
    if any(r['letters']['light'] >= light_floor for r in rows[:index]):
        raise ValueError('lettering was already bright before the proposed onset')
    letter_area = (ROIS['letters'][2]-ROIS['letters'][0])*(ROIS['letters'][3]-ROIS['letters'][1])
    control_area = (ROIS['metal_control'][2]-ROIS['metal_control'][0])*(ROIS['metal_control'][3]-ROIS['metal_control'][1])
    if any(r['metal_control']['red']/control_area >= red_floor/letter_area for r in rows):
        raise ValueError('adjacent non-letter metal trips the detector')
    if any(r['letters']['light'] < light_floor for r in rows[index+1:index+3]):
        raise ValueError('orange change is not followed by the letter glow')
    return rows[index]['frame']


def measure_oldfire(root=ROOT):
    rows, evidence = [], []
    for frame in range(FIRST, STOP):
        relative = Path('renders/book_D_oldfire')/f'f_{frame:05d}.jpg'
        path = Path(root)/relative
        with Image.open(path) as im:
            if im.size != (1920, 804):
                raise ValueError(f'wrong native dimensions: {relative}')
            rgb = np.asarray(im.convert('RGB'))
        row = dict(frame=frame)
        for name, (x0, y0, x1, y1) in ROIS.items():
            row[name] = counts(rgb[y0:y1, x0:x1])
        rows.append(row)
        evidence.append(dict(frame=frame, path=relative.as_posix(), sha256=sha(path)))
    frame = detect_flare(rows)
    if frame is None:
        raise ValueError('no admissible letter flare in the declared search window')
    robust = {str(n): detect_flare(rows, red_floor=n) for n in (50, 100, 200)}
    if set(robust.values()) != {frame}:
        raise ValueError('red threshold changes the measured onset')
    return dict(frame=frame, rows=rows, frames=evidence, thresholds=robust,
                scope='native_delivered_plate', rois=ROIS,
                limits='First letter-glow onset in the complete settled-Ring search [5880,5925); '
                       'not the earlier Ring fall, flame flicker or completed D edit.')


def overlay(result, root=ROOT):
    method = ('Native RGB settled-Ring ROI: first frame with >=50 orange glyph pixels '
              '(R>180, R-1.4G>40), all prior frames unlit by R>240,G>170; '
              'next two frames have >=100 bright glyph pixels. Adjacent metal fails '
              'the same area-normalized orange gate. Red counts50/100/200 agree. '
              'Before/on/after crops emitted for separate visual inspection; all45 frames decoded.')
    event = dict(landmark='old_fire_letters_flare', frame=result['frame'], status='measured',
                 source='Delivered book_D_oldfire RGB frame scan; measure_sound_d_books.py',
                 measured_ref='oldfire_measurements.json:frame', measurement_scope='native_delivered_plate',
                 editorial_reference='old_fire_ring_falls',
                 measurement=dict(method=method, claim='onset', predicate=method,
                     frame_size=[1920,804], frames=result['frames'],
                     coverage=dict(first=FIRST, last=STOP-1, complete=True, expected=STOP-FIRST,
                                   observed=len(result['frames']), missing=[]),
                     rois=ROIS, limitations=result['limits']))
    return dict(schema='long-dawn/d-effects-measurements/1', cut='D', fps=24, frames=9200,
                input_sha256={p:sha(Path(root)/p) for p in INPUTS}, hooks={}, series={},
                events={'D.new.oldfire.flare':event}, suppressions={})


def caption_onset(trace, first=6011, stop=6020):
    if [r.get('frame') for r in trace] != list(range(first,stop)):
        raise ValueError('caption search missing, duplicate or disordered frames')
    found=[r['frame'] for r in trace if r['dark_pixels']>=20]
    if not found or found[0] <= first+2 or found != list(range(found[0],stop)):
        raise ValueError('caption lacks dark-free prefix and sustained ink reveal')
    return found[0]


def book_revision_overlay(root=ROOT):
    """A caption onset and source-supported retirement of static-pen cues.

    The caption is an ink reveal without a visible pen. Its accompanying quill
    remains an authored sound choice. The last-leaf retirement cites source
    behavior as well as observed stills; sparse stills alone prove no absence.
    """
    root = Path(root)
    doc = dict(schema='long-dawn/d-effects-measurements/1', cut='D', fps=24, frames=9200,
               input_sha256={p:sha(root/p) for p in INPUTS}, hooks={}, series={}, events={}, suppressions={})
    frames, trace = [], []
    for f in range(6011, 6020):
        p=Path('renders/book_D_oldfire')/f'f_{f:05d}.jpg'
        with Image.open(root/p) as im:
            if im.size != (1920,804):
                raise ValueError('caption measurement requires native plates')
            rgb=np.asarray(im.convert('RGB'))[700:800,340:940]
        n=int(((rgb[...,0]<120)&(rgb[...,1]<85)&(rgb[...,2]<60)).sum())
        trace.append(dict(frame=f, dark_pixels=n))
        frames.append(dict(frame=f,path=p.as_posix(),sha256=sha(root/p)))
    onset=caption_onset(trace)
    measurement=dict(method='Native dark-ink count in caption ROI[340,700,940,800] after the diagram border has left that ROI; all9 frames6011:6019 decoded. Boundary crops visually inspected separately.',
        claim='onset',predicate='First >=20 pixels with R<120,G<85,B<60, following dark-free prefix; all later search frames positive.',
        frame_size=[1920,804],frames=frames,coverage=dict(first=6011,last=6019,complete=True,missing=[]),
        trace=trace,limitations='Visible caption ink reveal, not a visible pen contact; search window6011:6019. Earlier6010 diagram-border pixels trigger the same dark predicate; they are excluded by the visual caption ROI search interval, not called ink.')
    doc['events']['D.new.oldfire.quill']=dict(landmark='old_fire_caption_first_ink',frame=onset,status='measured',
        source='Delivered book_D_oldfire caption ROI RGB scan',measured_ref='book_revisions_overlay.json:events.D.new.oldfire.quill',
        measurement_scope='native_delivered_plate',editorial_reference='old_fire',measurement=measurement)
    stills=[]
    for f in (8320,8440,8480):
        p=Path('renders/book_lastleaf_open')/f'f_{f:05d}.jpg'
        with Image.open(root/p) as im:
            if im.size != (1920,804):
                raise ValueError('last-leaf observation requires native plates')
            im.load()
        stills.append(dict(frame=f,path=p.as_posix(),sha256=sha(root/p)))
    code={p:sha(root/p) for p in ('shots/map/lastleaf_v2.py','shots/map/lastleaf_v2_assets.py')}
    observed=dict(method='Three native delivered plates inspected with source review: LastLeafV2.leaf caches an already-rasterized broken line (pen.raster time1e9), frame caches a stationary pen mesh; only camera/lighting change.',
        claim='static_state',frame_size=[1920,804],frames=stills,
        coverage=dict(first=8320,last=8480,complete=False,missing=[f for f in range(8320,8481) if f not in (8320,8440,8480)]),
        source_sha256=code,limitations='Sparse stills establish visible resting pen; source establishes no write-on or pen-settle action. Not a full-shot optical-motion measurement.')
    for request in ('leaf.quill','leaf.stop','leaf.settle'):
        doc['suppressions']['D.new.'+request]=dict(reason='Delivered v2 plate begins with interrupted ink and resting pen already present; no on-screen writing, stop or pen-settle action exists in this implementation.',
            source='Native delivered plate inspection plus LastLeafV2.frame and lastleaf_v2_assets.leaf source',measurement=observed)
    return doc


def measure_round5(root=ROOT, *, baseline):
    """Re-read the complete swapped old-fire plate, with a preserved baseline."""
    root = Path(root)
    baseline = json.loads(Path(baseline).read_text())
    frames, flare_rows, caption_rows, comparisons = [], [], [], []
    for frame in range(5840, 6080):
        rgb, proof = native_frame(root, 'book_D_oldfire', frame)
        frames.append(proof)
        previous = root / 'renders/book_D_oldfire_r4' / f'f_{frame:05d}.jpg'
        old_hash = sha(previous)
        comparisons.append(dict(frame=frame, previous_sha256=old_hash, current_sha256=proof['sha256'],
                                byte_identical=old_hash == proof['sha256']))
        if FIRST <= frame < STOP:
            row = dict(frame=frame)
            for name, (x0,y0,x1,y1) in ROIS.items():
                row[name] = counts(rgb[y0:y1,x0:x1])
            flare_rows.append(row)
        if frame >= 5954:
            row = dict(frame=frame)
            for name, (x0,y0,x1,y1) in CAPTION_ROIS.items():
                for label, thresholds in INK_THRESHOLDS.items():
                    row[name+'_'+label] = dark_count(rgb[y0:y1,x0:x1], thresholds)
            caption_rows.append(row)
    inherited_controls = {}
    for threshold in (50,100,200):
        try:
            inherited_controls[str(threshold)] = dict(frame=detect_flare(flare_rows, red_floor=threshold), accepted=True)
        except ValueError as exc:
            inherited_controls[str(threshold)] = dict(frame=None, accepted=False, reason=str(exc))
    settled = [r for r in flare_rows if r['frame'] < ROUND5_FLARE_STOP]
    flare_controls = {str(n):detect_flare(settled,stop=ROUND5_FLARE_STOP,red_floor=n) for n in (50,100,200)}
    if None in flare_controls.values() or len(set(flare_controls.values())) != 1:
        raise ValueError('flare threshold controls disagree')
    onset_rows = [r for r in caption_rows if r['frame'] >= 5956]
    terminal_rows = [r for r in caption_rows if r['frame'] >= 5963]
    onset_controls = {label:persistent_entry(onset_rows,5956,6080,'first_ink_'+label,floor=20)
                      for label in INK_THRESHOLDS}
    completion_controls = {label:persistent_entry(terminal_rows,5963,6080,'terminal_period_'+label,floor=5)
                           for label in INK_THRESHOLDS}
    if len(set(onset_controls.values())) != 1 or len(set(completion_controls.values())) != 1:
        raise ValueError('caption thresholds disagree; do not promote a single completion frame')
    onset, completion = onset_controls['primary'], completion_controls['primary']
    if completion <= onset:
        raise ValueError('caption completion must follow its observed first ink')
    source_hashes = {p:sha(root/p) for p in ('shots/map/oldfire_v2.py','shots/map/oldfire.py')}
    # These source pins support only the already-reviewed static leaf decision.
    # Changed code is recorded as needing review; this collector never replaces
    # an old suppression's hash simply because a new version exists.
    leaf = []
    for request, decision in baseline.get('suppressions', {}).items():
        if not request.startswith('D.new.leaf.'):
            continue
        proof = decision['measurement']
        code = [dict(path=p, previous_sha256=h, current_sha256=sha(root/p), unchanged=h == sha(root/p))
                for p,h in proof.get('source_sha256', {}).items()]
        images = []
        for prior in proof['frames']:
            p = Path(prior['path'])
            _, current = native_frame(root,p.parts[1],prior.get('source_frame',prior['frame']))
            images.append(dict(path=prior['path'], previous_sha256=prior['sha256'],
                               current_sha256=current['sha256'], unchanged=prior['sha256'] == current['sha256']))
        leaf.append(dict(request_id=request, sources=code, frames=images,
                         needs_source_revalidation=not all(r['unchanged'] for r in code),
                         needs_frame_revalidation=not all(r['unchanged'] for r in images),
                         action='Retain the prior decision only when its existing code and native frame hashes still agree.'))
    old_events = baseline['events']
    changed = [r['frame'] for r in comparisons if not r['byte_identical']]
    return dict(scope='native_delivered_plate', coverage=dict(first=5840,last=6079,complete=True,missing=[]),
        frames=frames, flare=dict(frame=flare_controls['50'],thresholds=flare_controls,trace=flare_rows,
            search=dict(first=FIRST,last=ROUND5_FLARE_STOP-1),inherited_window_controls=inherited_controls,
            limits='All45 former search frames were remeasured. The revised melt after5921 enters the adjacent-metal ROI; retain those rejected controls. The onset search uses the settled interval5880:5921, before the changed melt, without weakening its pixel predicates.'),
        caption_onset=dict(frame=onset,thresholds=onset_controls,coverage=dict(first=5956,last=6079,complete=True,missing=[])),
        caption_completion=dict(frame=completion,thresholds=completion_controls,
            coverage=dict(first=5963,last=6079,complete=True,missing=[]),
            claim='First terminal punctuation after the last t is visible and remains present through the final frame; no pen-contact or ink-drying completion claim.'),
        caption_trace=caption_rows, caption_rois=CAPTION_ROIS, ink_thresholds=INK_THRESHOLDS,
        source_sha256=source_hashes, previous_comparison=comparisons,
        first_changed_frame=changed[0] if changed else None, changed_frames=changed,
        retained_leaf_revalidation=leaf,
        movements={k:dict(previous_frame=old_events[k]['frame'],frame=f,delta_f=f-old_events[k]['frame'])
                   for k,f in (('D.new.oldfire.flare',flare_controls['50']),('D.new.oldfire.quill',onset))},
        limits='All240 adopted native plates decoded. Old comparison files were hashed, not decoded. Caption onset search begins after the moving diagram border leaves its ROI; earlier border positives are retained in the trace. These are source-plate observations, not assembled film or audio-arrival measurements. No separate fall, extinction or bead sound row exists in the inherited effects binding.')


def round5_overlay(result, root=ROOT):
    doc = dict(schema='long-dawn/d-effects-measurements/1',cut='D',fps=24,frames=9200,
               input_sha256={p:sha(Path(root)/p) for p in INPUTS}, hooks={},events={},series={},suppressions={},bounds={})
    def proof(first, last, method, claim, predicate):
        return dict(method=method,claim=claim,predicate=predicate,frame_size=[1920,804],
                    frames=[r for r in result['frames'] if first <= r['frame'] <= last],
                    coverage=dict(first=first,last=last,complete=True,missing=[]),
                    source_sha256=result['source_sha256'],limitations=result['limits'])
    flare_method='Same native settled-Ring orange-glyph/next-two-bright-frames detector as Phase3, re-run on all42 current settled frames5880:5921 with red-count50/100/200 controls. The former45-frame window is remeasured and its later melt contamination retained separately.'
    flare = dict(landmark='old_fire_letters_flare',frame=result['flare']['frame'],status='measured',
        source='Current delivered book_D_oldfire native RGB scan',measured_ref='oldfire_round5_measurements.json:flare.frame',
        measurement_scope='native_delivered_plate',editorial_reference='old_fire_ring_falls',
        measurement=proof(FIRST,ROUND5_FLARE_STOP-1,flare_method,'onset','First >=50 pixels with R>180 and R-1.4G>40; following two frames >=100 pixels R>240,G>170.'))
    flare['measurement'].update(rois=ROIS,threshold_controls=result['flare']['thresholds'])
    doc['events']['D.new.oldfire.flare'] = flare
    ink = dict(landmark='old_fire_caption_first_ink',frame=result['caption_onset']['frame'],status='measured',
        source='Current delivered book_D_oldfire native caption ROI scan',measured_ref='oldfire_round5_measurements.json:caption_onset.frame',
        measurement_scope='native_delivered_plate',editorial_reference='old_fire',
        measurement=proof(5956,6079,'First-ink ROI after the moving diagram border exits; complete124-frame native trace and strict/primary/loose RGB controls.',
                          'onset','First >=20 dark pixels in ROI[410,700,600,755], primary R<120,G<85,B<60; every subsequent frame through6079 remains positive.'))
    ink['measurement'].update(roi=CAPTION_ROIS['first_ink'],threshold_controls=result['caption_onset']['thresholds'],
        limitations='The caption is an ink reveal with no visible pen. Its quill is authored accompaniment. D5954/5955 border pixels trip this predicate and are explicitly excluded by the inspected search window; four clear preceding frames5956:5959 establish the measured entry.')
    doc['events']['D.new.oldfire.quill'] = ink
    end = dict(frame=result['caption_completion']['frame'],offset_f=1,status='measured',
        source='Current delivered book_D_oldfire terminal punctuation ROI and full observed hold',
        measured_ref='oldfire_round5_measurements.json:caption_completion.frame',measurement_scope='native_delivered_plate',
        measurement=proof(5963,6079,'Final punctuation appears after the last t; complete117-frame terminal ROI trace, three RGB thresholds and before/on/after plus final-frame crops.',
                          'completion','First >=5 dark pixels in ROI[1472,699,1490,716], primary R<120,G<85,B<60; every subsequent frame through6079 remains positive.'))
    end['measurement'].update(roi=CAPTION_ROIS['terminal_period'],threshold_controls=result['caption_completion']['thresholds'],
        limitations='Completion means the final caption punctuation is visible. It does not measure ink drying, a pen stop, camera completion or an audible cutoff. The exclusive sound bound is an authored +1-frame convention.')
    doc['bounds']['D.new.oldfire.quill'] = dict(end=end)
    return doc


def round5_sheets(out, result, root=ROOT):
    """Native crops for independent visual checking; no picture alterations."""
    groups = [('flare_boundary',(result['flare']['frame']-1,result['flare']['frame'],result['flare']['frame']+1),(720,350,1200,590)),
              ('caption_first_ink',(5955,5956,5959,result['caption_onset']['frame'],5961),(390,685,690,765)),
              ('caption_complete',(5981,result['caption_completion']['frame']-1,result['caption_completion']['frame'],5984,6079),(1390,675,1510,745))]
    for name, selected, box in groups:
        width,height = box[2]-box[0],box[3]-box[1]
        sheet=Image.new('RGB',(width*len(selected),height+24),'#181818');draw=ImageDraw.Draw(sheet)
        for i,frame in enumerate(selected):
            rgb,_=native_frame(root,'book_D_oldfire',frame)
            sheet.paste(Image.fromarray(rgb).crop(box),(i*width,24))
            draw.text((i*width+4,5),f'D{frame}',fill='white')
        sheet.save(Path(out)/(name+'.png'))


def run(out, root=ROOT, *, revision='phase3', baseline=None):
    start=time.monotonic()
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    if revision == 'round5':
        if baseline is None:
            raise ValueError('Round5 requires the preceding overlay as --baseline')
        result=measure_round5(root,baseline=baseline)
        round5_sheets(out,result,root)
        result.update(seconds=time.monotonic()-start,
                      peak_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)*(1 if sys.platform=='darwin' else 1024))
        (out/'oldfire_round5_measurements.json').write_text(json.dumps(result,indent=2)+'\n')
        (out/'books_overlay.json').write_text(json.dumps(round5_overlay(result,root),indent=2)+'\n')
        print(json.dumps({k:result[k] for k in ('movements','caption_completion','first_changed_frame','seconds','peak_rss_bytes')}))
        return result
    result=measure_oldfire(root)
    sheet=Image.new('RGB',(480*3,270),(20,20,20)); draw=ImageDraw.Draw(sheet)
    for i,frame in enumerate(range(result['frame']-1,result['frame']+2)):
        with Image.open(Path(root)/'renders/book_D_oldfire'/f'f_{frame:05d}.jpg') as im:
            sheet.paste(im.convert('RGB').crop((720,350,1200,590)),(480*i,0))
        draw.text((480*i+6,245),str(frame),fill='white')
    sheet.save(out/'oldfire_flare_boundary.jpg',quality=96)
    result.update(seconds=time.monotonic()-start,
                  peak_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)*(1 if sys.platform=='darwin' else 1024))
    (out/'oldfire_measurements.json').write_text(json.dumps(result,indent=2)+'\n')
    (out/'oldfire_overlay.json').write_text(json.dumps(overlay(result,root),indent=2)+'\n')
    (out/'book_revisions_overlay.json').write_text(json.dumps(book_revision_overlay(root),indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('frame','thresholds','seconds','peak_rss_bytes','limits')}))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--root',type=Path,default=ROOT)
    parser.add_argument('--revision',choices=('phase3','round5'),default='phase3')
    parser.add_argument('--baseline',type=Path,help='Preceding sound measurement overlay for explicit old/new provenance')
    args=parser.parse_args();run(args.out,args.root,revision=args.revision,baseline=args.baseline)
