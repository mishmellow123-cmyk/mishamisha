"""Delivered C5 additions: EDIT's source resolver/compositor, streaming pixel statistics.

All coordinates are in the composed 960x402 picture, before EDIT captions. No renderer
schedule or score estimate supplies an event frame. Scalar traces can be saved for review.
"""
import hashlib
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EXTRA_SHOTS = {
    'opening': ('EDIT:book', 0, 559, 'C1-C3'),
    'letters': ('EDIT:letters+fire', 560, 1039, 'C4-C5'),
    'forge': ('EDIT:forge', 1040, 1679, 'C6-C7'),
    'eye': ('EDIT:deep+eye', 1680, 2079, 'C8-C9'),
    'flint': ('EDIT:flint', 2640, 2879, 'C12'),
    'run': ('EDIT:beacon-run', 3120, 3439, 'C14'),
    'illumination': ('EDIT:illumination', 4720, 5199, 'C20'),
    'plenty': ('EDIT:plenty', 5200, 5439, 'C21'),
    'title': ('EDIT:title', 5680, 5919, 'C23'),
}


class Picture:
    def __init__(self, root):
        sys.path.insert(0, os.path.join(ROOT, 'edit'))
        import assemble
        self.A = assemble
        assemble.RENDERS = root
        self.ctx = assemble.Ctx('C', scale=.5, clean=True)
        edl = json.load(open(os.path.join(ROOT, 'edit', 'edl', 'edl_C.json')))
        # Use the exported EDL, including every piece boundary; Ctx supplies the composition.
        self.ctx.shots = edl['shots']
        self.ctx.plans = [assemble.plan_shot(s, 'C', None) for s in self.ctx.shots]
        self.root = root
        self.hashes = {}

    def sources(self, f):
        i, shot = self.ctx.shot_at(f)
        p = self.ctx.plans[i]
        if p['kind'] == 'black':
            return [], {'cut': f, 'kind': 'EDIT black'}
        if p['kind'] != 'take' or p['have'] != shot['f1']-shot['f0']:
            raise FileNotFoundError(f"C{f}: incomplete shot {shot['name']}; no fallback/slate measurements")
        t = p['take']
        if t['mode'] == 'video':
            raise ValueError('video fallback is not delivered-frame evidence')
        path, _ = self.A.locate(t, 'C', None, f)
        paths = [path]
        for k in ('matte', 'add'):
            if t.get(k):
                p0 = self.A.index(os.path.join(self.root, t[k])).get(f + t['off'])
                if not p0:
                    raise FileNotFoundError(f"C{f}: missing {k} {t[k]}")
                paths.append(p0)
        if t.get('under'):
            p0, _ = self.A.locate_under(t['under'], f)
            if not p0:
                raise FileNotFoundError(f'C{f}: missing under-layer')
            paths.append(p0)
        exr = os.path.join(self.root, 'title_C', f'f_{f:05d}.exr')
        if shot['kind'] == 'title' and os.path.isfile(exr):
            paths.append(exr)
        return paths, dict(cut=f, source=f+t['off'], stem=t['stem'],
                           layers=[os.path.relpath(p0, self.root) for p0 in paths])

    def frame(self, f):
        self.sources(f)  # Fail on a missing layer even when EDIT would silently omit it.
        x, _, status, _ = self.ctx.picture(f)
        if status.startswith('SLATE'):
            raise ValueError(f'C{f}: slate')
        return x * 255

    def identity(self, key):
        stem, a, b, section = EXTRA_SHOTS[key]
        h = hashlib.sha256()
        pieces = []
        for shot, plan in zip(self.ctx.shots, self.ctx.plans):
            if shot['f0'] <= b and shot['f1'] > a:
                pieces.append(dict(first=max(a, shot['f0']), last=min(b, shot['f1']-1),
                                   kind=plan['kind'], take=plan['take']))
        for f in range(a,b+1):
            paths, row = self.sources(f)
            for p in paths:
                if p not in self.hashes:
                    sh = hashlib.sha256()
                    with open(p,'rb') as fh:
                        for chunk in iter(lambda: fh.read(1024*1024), b''):
                            sh.update(chunk)
                    self.hashes[p] = sh.hexdigest()
            row['sha256'] = [self.hashes[p] for p in paths]
            h.update((json.dumps(row,sort_keys=True)+'\n').encode())
        return dict(stem=stem,first=a,last=b,section=section,frame_set_sha256=h.hexdigest(),
                    pieces=pieces, identity_method='sha256 of cut-frame/source/layer names and file hashes; includes EDIT black',
                    composition='assemble.Ctx.picture at scale 0.5; before caption overlay',
                    edl_sha256=hashlib.sha256(open(os.path.join(ROOT,'edit/edl/edl_C.json'),'rb').read()).hexdigest())


def scan(picture, key):
    import cv2
    from measure_c5_events import _guard, luma, blobs
    _, a, b, _ = EXTRA_SHOTS[key]
    rows = []
    prev = None
    prev_warm = None
    template = None
    if key == 'letters':
        x = picture.frame(730)
        y = luma(x)
        template = ((y-cv2.GaussianBlur(y,(0,0),3)>8)&(x[...,0]>90)&(x[...,0]-x[...,1]>20)).astype('uint8')
        template = cv2.dilate(template,np.ones((7,7),'uint8')) > 0
    for f in range(a,b+1):
        x = picture.frame(f)
        y = luma(x)
        R,G,B=x.transpose(2,0,1)
        warm=(R-B>80)&(R>170)
        hot=(y>150)&(R>180)
        hp=y-cv2.GaussianBlur(y,(0,0),3)
        row=dict(f=f,mean=float(y.mean()),delta=0. if prev is None else float(np.abs(y-prev).mean()))
        if key == 'opening':
            row.update(page_mean=float(y[80:320,200:760].mean()),ink=int((hp[40:350,200:800]<-8).sum()))
        elif key == 'letters':
            glyph=(hp>8)&(R>90)&(R-G>20)
            row.update(glow=int(glyph.sum()),lift=int((glyph&~template).sum()),
                       fire=int(hot[180:340,430:530].sum()),hole=int((y[60:390,250:730]<10).sum()),
                       page=float(y[50:350,180:780].mean()))
        elif key == 'forge':
            towers=(y>55)&(R-B>25)
            towers[:,380:590]=False
            ring=blobs(hot[:280,380:590],10)
            ring=sorted(ring,key=lambda q:-q['area'])
            row.update(towers=int(towers.sum()),hot=int(hot.sum()),
                       inscription=int(((hp>3)&(y<235)&(R>160))[52:82,430:455].sum()),
                       ring_y=ring[0]['cy'] if ring else None,
                       ring_area=ring[0]['area'] if ring else 0)
            body=blobs(((R-B>25)&(R>60))[:,380:630],30)
            body=sorted(body,key=lambda q:-q['area'])
            row['body_y']=body[0]['cy'] if body else None
        elif key == 'eye':
            row.update(hole=int((y[50:380,300:650]<30).sum()),
                       slit=int((y[80:280,470:490]<25).sum()),ink=int((hp<-8).sum()))
        elif key == 'flint':
            row.update(flash=int(hot[40:230,350:650].sum()),
                       core=int(hot[170:350,400:650].sum()),
                       flame=int(((R>150)&(R-G>35)&(G-B>20))[100:240,430:600].sum()),
                       black=bool(y.max()<=2), max_luma=float(y.max()),
                       plume=float(y[180:280,490:630].mean()))
        elif key == 'run':
            # Join a flame crown and its glow, which the ink outline otherwise splits.
            m=cv2.morphologyEx(warm.astype('uint8'),cv2.MORPH_CLOSE,np.ones((9,5),'uint8'))
            row['blobs']=blobs(m,20)
            if prev is not None:
                (dx,dy),response=cv2.phaseCorrelate(prev[220:390].copy(),y[220:390].copy())
                aligned=cv2.warpAffine(prev_warm,np.float32([[1,0,dx],[0,1,dy]]),(960,402))
                old=cv2.dilate(aligned,np.ones((7,7),'uint8'))>0
                row.update(novel=blobs(warm&~old,5),dx=dx,dy=dy,registration_response=response)
            else:
                row.update(novel=[],dx=None,dy=None,registration_response=None)
            prev_warm=warm.astype('uint8')
        elif key == 'illumination':
            residual=y-cv2.GaussianBlur(y,(0,0),12)
            row.update(sun=blobs((residual[100:190,200:400]>3).astype('uint8'),15),
                       sky=float(y[40:160,100:850].mean()))
        elif key == 'plenty':
            row.update(ink=int((hp[80:350,450:850]<-8).sum()))
        elif key == 'title':
            row.update(glow=int(((hp>12)&(R>150))[170:320,240:800].sum()),
                       ink=int((hp[170:320,240:800]<-8).sum()))
        rows.append(row)
        prev=y
        _guard()
        if f%100==0:
            print(f'{key}: C{f}',flush=True)
    return rows


if __name__ == '__main__':
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument('--shots',default=','.join(EXTRA_SHOTS))
    a=ap.parse_args()
    p=Picture(os.path.join(ROOT,'renders'))
    for key in a.shots.split(','):
        rows=scan(p,key)
        target=os.path.join(ROOT,'review/codex-events-c5',key+'_trace.json')
        with open(target,'w') as fh: json.dump(rows,fh,separators=(',',':'))
        print('wrote',key,flush=True)
