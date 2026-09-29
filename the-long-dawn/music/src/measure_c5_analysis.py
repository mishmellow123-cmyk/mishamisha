"""Event decisions from the streaming delivered-picture statistics.

Controls use the event's fixed thresholds, never a new normalization fitted to the
control. Failed controls are retained as evidence and cannot produce a sync move.
"""
import numpy as np


def control(rows, field, interval, threshold, direction=1, why=''):
    selected=[r for r in rows if interval[0] <= r['f'] <= interval[1]]
    if len(selected) != interval[1]-interval[0]+1:
        raise ValueError(f'incomplete control {interval}')
    values=[float(r[field])*direction for r in selected]
    hits=[r['f'] for r,v in zip(selected,values) if v >= threshold]
    return dict(detector=field,first=interval[0],last=interval[1],frames_examined=len(selected),
                threshold=threshold,direction=direction,trigger_frames=hits,
                triggers=len(hits),maximum=max(values),selection_reason=why)


def measured(key, eid, what, frames, method, region, evidence, confidence='medium', sync=None, controls=()):
    from measure_c5_events import ev
    e=ev(f'{key}.{eid}',key,what,frames,method,dict(box=region,space='960x402 composed RGB'),
         evidence,confidence)
    e['negative_controls']=list(controls)
    if sync:
        e['sync_id']=sync
        e['sync_field']='first'
    return e


def ramp_event(key, rows, eid, field, search, before, after, what, method, region, sync=None,
               direction=1, min_span=1):
    from measure_c5_events import ramp
    select=lambda w: np.array([float(r[field])*direction for r in rows if w[0]<=r['f']<=w[1]])
    base,top=float(np.median(select(before))),float(np.median(select(after)))
    s=select(search)
    ctrl=control(rows,field,before,base+.1*(top-base),direction,
                 'inspected pre-event interval; same fixed 10% presence threshold as the event')
    good=top-base>=min_span and ctrl['triggers']==0
    first,full,fastest,half=ramp(s,search[0],base=base,top=top) if good else (None,)*4
    e=measured(key,eid,what,dict(first=first,half=half,full=full),
               method+f'; baseline median C{before[0]}-{before[1]}, end median C{after[0]}-{after[1]}; '
               f'first/half/full = 10/50/90%; presence requires span >= {min_span} and no baseline trigger',region,
               dict(field=field,search=list(search),frames_examined=len(s),base=base,top=top,
                    values={str(r['f']):r[field] for r in rows if search[0]<=r['f']<=search[1]}),
               sync=sync,controls=[ctrl])
    if not good or first is None or full is None:
        e.update(status='unresolved',confidence='low',note='Presence/control gate failed; no measured synchronization frame.')
    return e


def onset_event(key, rows, eid, field, search, absent, threshold, what, method, region, sync=None):
    positive=[r for r in rows if search[0]<=r['f']<=search[1]]
    c=control(rows,field,absent,threshold,why='inspected interval without this event')
    hits=[r for r in positive if r[field]>=threshold]
    first=hits[0]['f'] if hits and c['triggers']==0 else None
    e=measured(key,eid,what,dict(first=first),method+f'; first = first sample >= {threshold}',region,
               dict(field=field,search=list(search),frames_examined=len(positive),threshold=threshold,
                    values={str(r['f']):r[field] for r in positive}),sync=sync,controls=[c])
    if first is None:
        e.update(status='unresolved',confidence='low',note='No accepted onset: presence or negative-control gate failed.')
    return e


def add_onset(e, rows, field, search, absent, threshold, what, why):
    """A breach (a page burning through, a slit parting) is visible frames before its ramp reaches 10% of the size it
    grows to, and the music answers the breach, not the growth: onset = the first frame in `search` whose value
    reaches a FIXED threshold set just above the no-event level, checked against an interval without the event.
    The ramp's first/half/full stay as measured; the onset becomes the sync field only when its control is clean."""
    if e.get('status') == 'unresolved':
        return e
    c=control(rows,field,absent,threshold,why=why)
    hits=[r['f'] for r in rows if search[0]<=r['f']<=search[1] and r[field]>=threshold]
    e['frames']['onset']=hits[0] if hits and c['triggers']==0 else None
    e['negative_controls'].append(c)
    e['method']+=f'; onset = first frame C{search[0]}-{search[1]} with {what} >= {threshold} (fixed; see its control)'
    e['evidence']['onset']=dict(field=field,search=list(search),threshold=threshold,
                                values={str(r['f']):r[field] for r in rows if search[0]<=r['f']<=search[0]+40})
    if e['frames']['onset'] is not None:
        e['sync_field']='onset'
    return e


def analyze(key, rows):
    events=[]
    if key=='opening':
        events.append(onset_event(key,rows,'riffle','delta',(320,359),(380,399),10,
            'rapid page riffle begins','mean absolute luma change from previous frame; threshold separates '
            'the rapid leaf sweep from the inspected slow camera drift', [0,0,960,402],sync='riffle'))
    elif key=='letters':
        events.append(onset_event(key,rows,'page_turn','delta',(560,599),(650,679),10,
            'rapid sweep to the script leaf','mean absolute luma change from previous frame; '
            'the event is the rapid sweep, not an inferred physical hand contact',[0,0,960,402],sync='page_turn'))
        events.append(ramp_event(key,rows,'glow','glow',(690,729),(680,699),(716,729),
            'the script glows','count R>90, R-G>20 and luma minus Gaussian(sigma=3)>8',
            [0,0,960,402],sync='letters_glow',min_span=100))
        events.append(ramp_event(key,rows,'lift','lift',(730,765),(730,734),(756,765),
            'glowing letters leave their original positions','count the glow mask outside a 3-pixel dilation '
            'of the C730 glyph mask; camera drift within that dilation is tolerated; measures displaced area, '
            'not a physical lift height',[0,0,960,402],sync='letters_lift',min_span=100))
        fire=onset_event(key,rows,'catch','fire',(797,840),(797,800),100,
            'fire catches at the page heart','count luma>150 and R>180 in the central fire ROI',
            [430,180,530,340],sync='fire_catches')
        fire['note']='Scoped to the catch after the last flying glyph disappears at C797. Earlier glyph sparks also pass this brightness threshold; it is not a general flame classifier.'
        if fire['frames']['first'] is not None:
            f=fire['frames']['first'];local=[r for r in rows if f<=r['f']<=840]
            reference=float(np.median([r['fire'] for r in rows if 818<=r['f']<=830]))
            fire['frames'].update(fastest=max(local[1:],key=lambda r:r['fire']-rows[r['f']-rows[0]['f']-1]['fire'])['f'],
                                 full=next(r['f'] for r in local if r['fire']>=.9*reference))
            fire['method']+='; fastest = largest core-area increase through C840; full = 90% of median core area C818-830 (the catch flash)'
            fire['evidence']['area_reference']=reference
        events.append(fire)
        events.append(add_onset(ramp_event(key,rows,'burn','hole',(820,960),(820,839),(940,959),
            'the page burns open onto black','pixels with luma<10 in the page interior; composed '
            'book_C_ft plus embers_C3_e15, never the matte alone',[250,60,730,390],sync='burn_through',min_span=100),
            rows,'hole',(820,960),(560,839),1,'any see-through (luma<10) pixel',
            'the whole page before its burn: no see-through pixel in 280 frames'))
    elif key=='forge':
        events.append(ramp_event(key,rows,'towers','towers',(1040,1120),(1040,1059),(1110,1120),
            'forge towers become visible around the central fire','count luma>55 and R-B>25 outside '
            'x380-589 (exclude the lone central flame); apparent emergence in the moving camera',
            [0,0,960,402],sync='towers_rise',min_span=100))
        events.append(onset_event(key,rows,'inscription','inscription',(1310,1330),(1310,1319),120,
            'the first inscription glyph glows on the Ring','count luma-Gaussian(sigma=3)>3, luma<235, '
            'R>160 in the inspected first-glyph ROI; reject the saturated reflection',
            [430,52,455,82],sync='inscription'))
        events.append(ramp_event(key,rows,'ring_rises','body_y',(1378,1450),(1378,1384),(1440,1450),
            'the Ring climbs within the picture','centroid y of the largest connected gold body '
            '(R-B>25,R>60,area>=30) in x380-629; track the body rather than its specular highlights; '
            'apparent screen-space rise during the camera pullback',[380,0,630,402],sync='ring_rises',
            direction=-1,min_span=20))
        from scipy.ndimage import median_filter
        d=np.array([r['delta'] for r in rows]);threshold=3*median_filter(d,size=15,mode='nearest')+.5
        hits=[r['f'] for r,t in zip(rows,threshold) if r['delta']>t]
        controls=[dict(detector='anvil impulse',first=1041,last=1199,frames_examined=159,
                       trigger_frames=[f for f in hits if 1041<=f<=1199],
                       triggers=sum(1041<=f<=1199 for f in hits),
                       selection_reason='the opening tower emergence, with no pictured anvil contact',
                       threshold='dY > 3 * 15-frame running median + 0.5')]
        events.append(measured(key,'anvil_test','search for a pictured anvil impulse',dict(found=hits),
            'mean absolute luma difference; impulse >3x 15-frame running median+0.5; '
            'a candidate would still require a pictured strike, not a camera move',[0,0,960,402],
            dict(frames_examined=len(rows)-1,maximum=float(d.max())),controls=controls))
    elif key=='eye':
        for a,b in zip(rows,rows[1:]):             # the scorch lands as a sudden darkening of the glowing spot
            b['scorch_drop']=a['scorch']-b['scorch']
        rows[0]['scorch_drop']=0.
        events.append(add_onset(ramp_event(key,rows,'burn','hole',(1905,1970),(1905,1920),(1955,1970),
            'the mine page opens into the storm','count luma<30 in the page interior of '
            'EDIT RGB+(1-matte)*embers_C3; the backing storm is not black',[300,50,650,380],sync='eye_burn',min_span=100),
            rows,'scorch_drop',(1905,1940),(1681,1922),10,
            'a one-frame drop of the mean luma in the burn region (x380-579, y220-369)',
            'the Deep and the glow before the burn: its tick-lit growth never darkens the region this fast'))
        events.append(add_onset(ramp_event(key,rows,'slit','slit',(1992,2079),(1992,2000),(2060,2079),
            'the Eye slit opens onto darkness','count luma<25 in the iris central strip; '
            'exclude the earlier page tear by starting after the composited burn',[470,80,490,280],
            sync='slit_nothing',min_span=100),
            rows,'slit90',(1992,2079),(1992,2000),2000,'pixels with luma<90 in the iris central strip (the lens)',
            'the resolved Eye with its slit closed to a line'))
    elif key=='flint':
        for eid,search,absent in [('strike1',(2640,2669),(2640,2649)),('strike3',(2670,2720),(2670,2697))]:
            e=onset_event(key,rows,eid,'flash',search,absent,1000,
                          f'FLINT {eid}: the spark flash begins','count luma>150 and R>180 around '
                          'the flint contact, excluding the tinder bed',[350,40,650,230])
            e['frames']['peak']=max([r for r in rows if search[0]<=r['f']<=search[1]],key=lambda r:r['flash'])['f']
            events.append(e)
        black=[r['f'] for r in rows if r['black']]
        events.append(measured(key,'black_test','search for FLINT black frames',dict(found=black),
            'black = maximum composed-frame luma <=2 (8-bit units)',[0,0,960,402],
            dict(frames_examined=len(rows),minimum_frame_max_luma=min(r['max_luma'] for r in rows)),
            'high',controls=[control(rows,'max_luma',(2640,2649),-2,-1,
                                     'inspected hands and blue sky at the opening')]))
        # Failed controls remain visible; no sync can be inferred from these metrics.
        for eid,field,threshold,absent in [('catch_test','flame',100,(2828,2835)),
                                          ('blow_test','plume',20,(2698,2703))]:
            e=onset_event(key,rows,eid,field,(2721,2879),absent,threshold,
                          'unresolved flame catch' if eid=='catch_test' else 'unresolved blowing onset',
                          'candidate gold-pixel area above the tinder' if field=='flame' else 'candidate smoke/plume mean luma',
                          [430,100,600,240] if field=='flame' else [490,180,630,280])
            e['frames']={'first':None,'fastest':None,'full':None}
            e.update(status='unresolved',confidence='low',note=(
                'The candidate also fires on glowing tinder/smoke before the distinct flame. No measured catch frame; '
                'retain pass 1 pending a reliable flame-shape boundary.' if eid=='catch_test' else
                'The candidate also fires on the strike flash when nobody is visibly blowing. The mouth is outside '
                'this framing; no unique visible onset was isolated. Do not infer it from smoke brightness.'))
            events.append(e)
    elif key=='run':
        events.extend(analyze_run(rows))
    elif key=='illumination':
        events.append(measured(key,'sun_already_visible','the sun is already partly above the ridge at the incoming cut',
            dict(visible_from=4720,onset=None),
            'all 480 composed frames scanned; inspected C4720-4840 crops show the disc already present at C4720; '
            'a within-shot argmax cannot locate an onset that precedes the delivered interval',
            [200,90,340,190],dict(frames_examined=len(rows),sky_mean_min=min(r['sky'] for r in rows),
                                  sky_mean_max=max(r['sky'] for r in rows)), 'medium'))
    elif key=='plenty':
        events.append(measured(key,'continuous_page','the landscape becomes a book page with a tree drawing',{},
            'all 240 composed frames scanned; mean frame difference and local ink counts are non-specific '
            'during camera motion; the barmap cue is the section/hymn entry',[0,0,960,402],
            dict(frames_examined=len(rows),largest_change_frame=max(rows,key=lambda r:r['delta'])['f']), 'low'))
    elif key=='title':
        e=onset_event(key,rows,'first_light','glow',(5680,5759),(5680,5700),3,
            'the first title lettering glows on the blank page','count R>150 and '
            'luma-Gaussian(sigma=3)>12 in the title line; excludes the book gutter',[240,170,800,320],sync='title')
        e['note']='Onset threshold only; later glow and ink phases are not one monotone ramp, so no full-title frame is inferred from their maximum.'
        events.append(e)
    return events


def analyze_run(rows):
    # The threshold is spatial: ordinary flicker stays inside the dilated previous flame.
    # Registration is measured from the landscape, rather than assumed from the EDL.
    samples=[]
    for r in rows:
        n=max([b['area'] for b in r['novel']]+[0])
        samples.append(dict(f=r['f'],novel_area=n))
    births=[r for r in samples if r['novel_area']>=50]
    controls=[control(samples,'novel_area',(3121,3159),50,why='opening landscape with no burning beacon'),
              control(samples,'novel_area',(3208,3231),50,why='camera and already-burning beacons; no new catch')]
    events=[]
    if len(births)!=7 or any(c['triggers'] for c in controls):
        raise ValueError('beacon detector did not establish exactly seven spatially distinct catches with clean controls')
    for k,birth in enumerate(births,1):
        f=birth['f'];r=rows[f-rows[0]['f']]
        blob=max(r['novel'],key=lambda b:b['area'])
        e=measured('run',f'beacon_{k}',f'Beacon Run catch {k}',dict(first=f,fastest=None,full=None),
            'warm pixels R-B>80,R>170; register preceding mask using phase correlation of landscape rows220-389; '
            'warp then dilate by 3px; a new connected warm component >=50px is a catch. '
            'First is the arrival of new warm area. Fastest/full are null: registered glow novelty is not flame-size growth.',
            [blob['x0'],blob['y0'],blob['x1'],blob['y1']],
            dict(frames_examined=len(rows)-1,novel_area=blob['area'],camera_dx=r['dx'],camera_dy=r['dy'],
                 registration_response=r['registration_response'],
                 neighborhood={str(s['f']):s['novel_area'] for s in samples if f-2<=s['f']<=f+3}),
            sync=f'beacon_{k}',controls=controls)
        events.append(e)
    return events


# Reasons to preserve musical cues; these are not measured synchronization claims.
PASS1_REASONS = {
 'hearth': 'Sound cue over EDIT black; no visible hearth event.',
 'page_turn_0': 'C0-79 are EDIT black; no visible page turn at C40.',
 'book': 'Section/music entry at C80; the book fades in after this boundary.',
 'blank_sheaf': 'Camera drift across illustrated pages; no isolated blank-sheaf arrival established.',
 'ring_motif': 'Musical motif entry over a mountain illustration; no separate small Ring stroke isolated.',
 'fire_alone': 'Section/chord entry during the continuing page burn; no separate ignition at C880.',
 'anvil': 'No pictured hammer contact identified. Impulse C1262 shows a rotating Ring in adjacent stills, not a strike.',
 'hush': 'Musical hush at the section cut; no discrete picture onset measured.',
 'breath_race': 'Musical breath, not a picture event.',
 'race': 'Musical acceleration; continuous tower/camera motion, no accepted discrete onset.',
 'deep': 'Section boundary within a continuing transition onto parchment.',
 'tick_8': 'Musical subdivision, not a picture event.',
 'tick_16': 'Musical subdivision, not a picture event.',
 'mirror': 'Section boundary/legacy alias for refusal.',
 'refusal': 'Section cut into the Refusal page; no new within-shot event.',
 'trap': 'Section cut; inter-shot burn is not supplied by Ctx.picture.',
 'flint_black': 'No black in 240 examined FLINT frames (maximum-luma threshold <=2); retain musical cut at 2640 without claiming pictured black.',
 'catch': 'Flame-area detector fails its pre-catch control on glowing tinder/smoke; onset, fastest and full unresolved. SOUND 2836 remains derived.',
 'promise': 'Caption timing, excluded from picture-only measurements.',
 'run': 'Section cut into Beacon Run; separate catches are measured.',
 'map': 'Section cut to map; separate catches already measured.',
 'smoke': 'Smoke continues from shutdown; no discrete onset.',
 'ring_unfinished': 'Section cut; separate gold-drain and storm-thinning ramps measured.',
 'deep_still': 'Section cut to held mine drawing; no discrete within-shot event.',
 'watch': 'Section cut into already-burning beacons; no catch in the shot.',
 'sunrise': 'Sun already partly above ridge at C4720; no onset within delivered 480-frame interval. Preserve section entry.',
 'home': 'Musical resolution, not a discrete sunrise event.',
 'plenty': 'Section/hymn entry; camera pulls from landscape to illustrated book page continuously.',
 'havens': 'Section cut; incoming page-turn transition absent from delivered PEN frames.',
 'line_in': 'Optional spoken-line placement, not a picture event.',
 'blank': 'Musical return after spoken line; existing PEN frames have no discrete ink event.',
 'plagal': 'Musical cadence, not a picture event.',
}
