"""Measure adopted D transitions at native resolution, before master finish.

The incoming/outgoing black-source counterfactuals distinguish visible source
contribution from the burn's added light. They do not measure physical flame
energy. Use the owner production slot; all output is external and explicit.
"""
from copy import deepcopy
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import tempfile
import time

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
INPUTS = ('edit/edl/edl_D.json','music/v3/barmap_D.json','music/v3/cues_D.json',
          'music/v3/events_D_measured.json','music/v3/events_D_round4_measured.json')
EVENT = 'D.new.unfinished.burn'
SIZE = (1920,804)
CUTS = (4080,4240,4560,5840,6080,6640)
CAPTION_ROI = (390,675,1510,765)
LEVELS = (1,4,8)


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1<<20),b''): h.update(block)
    return h.hexdigest()


def pixels(image):
    x=np.asarray(image)
    if x.shape!=(804,1920,3) or not np.isfinite(x).all():
        raise ValueError('requires finite native RGB picture')
    return (np.clip(x,0.,1.)*255.+.5).astype(np.uint8)


def counts(actual, counter):
    if actual.dtype!=np.uint8 or counter.dtype!=np.uint8 or actual.shape!=counter.shape:
        raise ValueError('equal uint8 picture arrays required')
    delta=np.max(np.abs(actual.astype(np.int16)-counter.astype(np.int16)),axis=-1)
    return {str(t):int(np.count_nonzero(delta>=t)) for t in LEVELS}


def counterfactuals(outgoing, incoming, frame, spec, layer_builder):
    """Identical selected burn field for all three source interventions."""
    if frame == spec['f0']:
        return outgoing.copy(), outgoing.copy(), np.zeros_like(outgoing)
    glow,keep,cover=layer_builder(frame,spec,outgoing.shape[1],outgoing.shape[0])
    old=outgoing*keep
    new=incoming*(1.-cover[...,None])
    return old+new+glow, old+glow, new+glow


def boundaries(rows, field, first=6080, stop=6120):
    if [r['frame'] for r in rows]!=list(range(first,stop)):
        raise ValueError('transition boundary needs full ordered unique coverage')
    result={}
    for t in LEVELS:
        positive=[r['frame'] for r in rows if r[field][str(t)]>0]
        onset=min(positive) if positive and min(positive)>first else None
        clear=max(positive)+1 if positive and max(positive)+1<stop else None
        result[str(t)]=dict(first_positive=onset,last_positive=max(positive) if positive else None,
                           first_trailing_zero=clear)
    return result


def selected(edl, first, stop):
    shots=[s for s in edl['shots'] if s['f0']<stop and s['f1']>first]
    trans=[t for t in edl['transitions'] if t['f0']<stop and t['f1']>first]
    # A transition's held source may lie just outside its active interval.
    for t in trans:
        for s in edl['shots']:
            if (s['f0']<=t['cut']-1<s['f1'] or s['f0']<=t['cut']<s['f1']) and s not in shots:
                shots.append(s)
    return deepcopy(dict(shots=shots,transitions=trans))


def overlay(result):
    onsets={v['first_positive'] for v in result['burn']['incoming_contribution'].values()}
    clears={v['first_trailing_zero'] for v in result['burn']['outgoing_contribution'].values()}
    if len(onsets)!=1 or None in onsets or len(clears)!=1 or None in clears:
        raise ValueError('sensitivity controls disagree; do not promote one burn frame')
    start,end=next(iter(onsets)),next(iter(clears))
    def point(frame,claim):
        images=[r for r in result['retained_frames'] if frame-1<=r['frame']<=frame+1]
        if [r['frame'] for r in images]!=list(range(frame-1,frame+2)):
            raise ValueError('burn landmark needs inspected before/at/after native bracket')
        proof=dict(method=result['method'],claim=claim,frame_size=list(SIZE),frames=deepcopy(images),
            coverage=dict(first=frame-1,last=frame+1,complete=True,missing=[]),
            predicate=('First incoming-picture contribution exceeds each 1/4/8 RGB-code threshold after a zero prefix.'
                       if claim=='onset' else 'Outgoing-picture contribution is zero at every 1/4/8 RGB-code threshold through D6119 after positive preceding frame.'),
            controls=deepcopy(result['burn']),dependencies=deepcopy(result['dependencies']),
            edl_dependencies=deepcopy(result['edl_dependencies']),
            limitations='Native pre-finish composite; displayed source visibility only. Master-finished visibility, physical fire energy, caption legibility and audible level are not measured.')
        return dict(frame=frame,status='measured',source='Adopted native pre-finish D6080 ring burn',
            measured_ref='transition_measurements.json#burn',measurement_scope='native_pre_finish_composite',measurement=proof)
    onset=point(start,'onset'); onset['landmark']='unfinished_first_visible_hole'
    onset['editorial_reference']='ring_unfinished'
    endp=point(end,'completion');endp['offset_f']=0
    return dict(schema='long-dawn/d-effects-measurements/1',cut='D',fps=24,frames=9200,
        input_sha256=deepcopy(result['input_sha256']),hooks={},series={},suppressions={},
        events={EVENT:onset},bounds={EVENT:{'end':endp}})


def scan(out, local_renders):
    started=time.perf_counter()
    out=Path(out).resolve(); local_renders=Path(local_renders).resolve()
    if out.is_relative_to(ROOT.resolve()) or out.is_relative_to(local_renders):
        raise ValueError('output must be outside repository and source assets')
    out.mkdir(parents=True,exist_ok=True);(out/'composites').mkdir(exist_ok=True)
    inputs={name:digest(ROOT/name) for name in INPUTS}
    edl=json.loads((ROOT/INPUTS[0]).read_text())
    runtime_sha=digest(ROOT/'edit/edl_v3.py')
    codepaths=['edit/assemble.py','edit/ring_burn.py','shots/map/x1burn.py','shots/map/burn.py',
               'shots/map/noise.py','lib/look.py','music/src/measure_sound_d_transitions.py']
    deps={p:dict(root='repo',path=p,sha256=digest(ROOT/p)) for p in codepaths}
    retained=[]; burn_rows=[]; windows=[]
    os.environ['CUTD_LOCAL_RENDERS']=str(local_renders)
    with tempfile.TemporaryDirectory(prefix='soundd-phase6-transition-') as scratch:
        os.environ['NUMBA_CACHE_DIR']=str(Path(scratch)/'numba')
        os.environ.setdefault('NUMBA_NUM_THREADS','2')
        sys.path.insert(0,str(ROOT/'edit'))
        import assemble as AS
        AS.CACHE=str(Path(scratch)/'assembler');AS._INDEX.clear()
        if AS.edl_doc('D')!=edl: raise ValueError('assembler/export EDL mismatch')
        AS._init('D',None,1.,True,False);ctx=AS._CTX
        original_read=ctx.read
        def read(ref,*args,**kwargs):
            if not isinstance(ref,str): raise ValueError('unexpected video source')
            path=Path(ref)
            try: name=path.relative_to(ROOT).as_posix()
            except ValueError: raise ValueError('unexpected non-repository transition source')
            if name not in deps:
                with Image.open(path) as image:
                    if image.size!=SIZE: raise ValueError('source would be silently resized')
                deps[name]=dict(root='repo',path=name,sha256=digest(path))
            return original_read(ref,*args,**kwargs)
        ctx.read=read
        sheet=Image.new('RGB',(5*384,8*183),'#111111');draw=ImageDraw.Draw(sheet)
        for cut in CUTS:
            ts=[t for t in edl['transitions'] if t.get('cut')==cut]
            if cut==6640:
                if ts: raise ValueError('D6640 must remain straight cut')
                first,stop=cut-1,cut+2
            else:
                if len(ts)!=1: raise ValueError('expected exactly one adopted window')
                first,stop=ts[0]['f0'],ts[0]['f1']
            metrics=[]
            keep={first,cut-1,cut,stop-1}
            if cut==6080: keep.update({6082,6083,6084,6093,6094,6095,6108,6109,6110})
            for f in range(first,stop):
                actual,_,status,source=ctx.picture(f)
                if status.startswith('SLATE') or source is None: raise ValueError('assembler returned fallback')
                rgb=pixels(actual)
                row=dict(frame=f,pixel_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),status=status)
                if cut==6080:
                    t=ts[0]
                    outgoing=AS.Ctx.picture(ctx,t['cut']-1)[0]
                    incoming=AS.Ctx.picture(ctx,f)[0]
                    direct,no_incoming,no_outgoing=counterfactuals(outgoing,incoming,f,t,AS.RB.layers)
                    direct_rgb=pixels(direct)
                    if not np.array_equal(rgb,direct_rgb): raise ValueError('native burn construction mismatch')
                    row['incoming_contribution']=counts(rgb,pixels(no_incoming))
                    row['outgoing_contribution']=counts(rgb,pixels(no_outgoing))
                    x0,y0,x1,y1=CAPTION_ROI
                    row['caption_change']=counts(rgb[y0:y1,x0:x1],pixels(outgoing)[y0:y1,x0:x1])
                    burn_rows.append(row)
                    x,y=((f-first)%5)*384,((f-first)//5)*183
                    sheet.paste(Image.fromarray(rgb).resize((384,161),Image.Resampling.LANCZOS),(x,y+22))
                    draw.text((x+4,y+4),f'D{f} pre-finish',fill='white')
                metrics.append(row)
                if f in keep:
                    relative=f'composites/f_{f:05d}.png'
                    Image.fromarray(rgb).save(out/relative)
                    retained.append(dict(frame=f,root='soundd_deep_composites',path=relative,sha256=digest(out/relative)))
            windows.append(dict(cut=cut,half_open=[first,stop],coverage_complete=True,frames=metrics,
                edl_dependencies=selected(edl,first,stop),
                interpretation='Observed assembled screen states; transition gain and any sound bridge remain authored.'))
        sheet.save(out/'burn_contact_sheet.jpg',quality=90)
    for row in deps.values():
        if digest(ROOT/row['path'])!=row['sha256']: raise ValueError('dependency changed during scan')
    if inputs!={name:digest(ROOT/name) for name in INPUTS} or runtime_sha!=digest(ROOT/'edit/edl_v3.py'):
        raise ValueError('editorial inputs changed during scan')
    result=dict(schema='long-dawn/d-transition-native-measurements/1',scope='native_pre_finish_composite',
        frame_size=list(SIZE),input_sha256=inputs,runtime_edl_sha256=runtime_sha,
        method='All 40 native D6080:6119 assembler composites before master finish; exact 8-bit display RGB differences to the identical burn with incoming or outgoing source blacked out. Thresholds 1/4/8 code values. All six adopted join windows also decoded without slates.',
        windows=windows,retained_frames=retained,dependencies=list(deps.values()),
        edl_dependencies=selected(edl,6080,6120),caption_roi=list(CAPTION_ROI),
        caption_scope='First visible 8-bit RGB change in the stated caption rectangle; this does not identify first mathematical scorch coefficient or a physical ink-destruction onset.',
        burn={key:boundaries(burn_rows,key) for key in ('incoming_contribution','outgoing_contribution','caption_change')},
        seconds=time.perf_counter()-started,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    (out/'transition_measurements.json').write_text(json.dumps(result,indent=2)+'\n')
    (out/'burn_overlay.json').write_text(json.dumps(overlay(result),indent=2)+'\n')
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--local-renders',type=Path,required=True)
    a=p.parse_args();r=scan(a.out,a.local_renders)
    print(json.dumps({k:r[k] for k in ('burn','seconds','peak_rss_bytes')}),flush=True)


if __name__=='__main__':main()
