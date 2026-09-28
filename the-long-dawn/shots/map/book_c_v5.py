"""C v5.2 opt-in ink-page group, absolute C frames; no production dispatch changes.

python shots/map/book_c_v5.py --shot refusal --frames 2080-2319 --out /absolute/path
Outputs opaque book RGB and a matching matte. Captions belong to EDIT.
"""
import argparse
import json
import os
from pathlib import Path
import resource
import sys
import time
import cv2
import numpy as np
import book_c as BC
import book as B
import redbook as RB
import pen
import v5_inkpages as P
import v5_pen

cv2.setNumThreads(0)  # look import configures OpenCV; apply the per-process limit afterward.
SHOTS={'refusal':(2080,2320),'deep_abandoned':(4240,4480),'pen':(5440,5680)}


class PagesV5(BC.Book3):
    def __init__(self,shot,W=1920,H=804,ppc=110):
        if shot not in SHOTS:raise ValueError('unknown v5 page shot')
        super().__init__(W,H)
        self.shot=shot;self.ppc=ppc
        self.blank=B.blank_tex(20.,29.)

    def frame(self,f):
        a,b=SHOTS[self.shot]
        if not a<=f<b:raise ValueError(f'{self.shot} accepts C frames {a}–{b-1}; got {f}')
        self._post=None
        return getattr(self,'shot_'+self.shot)((f-a)/24.,f)

    def shot_refusal(self,t,f):
        bk=self.book(.6,2.8)
        page=self.once('refusal',lambda: RB.Page(P.refusal(),self.ppc))
        target=bk.page_to_world('R',np.array([9.4]),np.array([13.5]))[0]
        off=np.array([-.7,-25.0,36.0])*(1-.035*BC.smooth(t/10))
        cam=B.Cam(target+off,target,38.,self.W,self.H);cam.dof_k=1.5
        # The returning hearth briefly flares; no emblematic gold fire here.
        power=2.2*(1+.22*np.exp(-((t-.30)/.36)**2))
        L=self.light((-50,48,36),power,t,amt=.10)
        return self.finish_layer(bk,cam,L,self.tex_text(41,40),page.texture(t),t)

    def abandoned(self):
        dp=P.AbandonedDeep()
        S=dp.build('ink',.6,9.6);S.extend(dp.additions())
        return dp,RB.Page(S,self.ppc).texture(1e9)

    def shot_deep_abandoned(self,t,f):
        bk=self.book(1.6,2.2,seed=9)
        dp,texture=self.once('abandoned',self.abandoned)
        target=bk.page_to_world('R',np.array([9.6]),np.array([14.2]))[0]
        cam=B.Cam(target+np.array([-.4,-50.,36.]),target,38.,self.W,self.H);cam.dof_k=1.2
        L=self.light((-50,48,36),2.1,t,amt=.09)
        # Retain actual gilt reflection; no emissive vein, red glow or burn.
        return self.finish_layer(bk,cam,L,self.tex_text(31,40),texture,t)

    def shot_pen(self,t,f):
        bk=self.book(2.8,1.2,seed=13)
        target=np.array([.6,.2,2.0])
        cam=B.Cam(target+np.array([-.5,-39.,39.])*(1-.025*BC.smooth(t/10)),target,38.,self.W,self.H)
        cam.dof_k=2.
        L=self.light((-55,50,30),1.9,t,seed=5,amt=.14,col=(1.,.52,.24),radius=10)
        hdr,alpha,G=B.render(bk,cam,L,self.blank,self.blank,t,
            fill=((.35,-.5,.8),(.010,.012,.016)))
        mesh=self.once('physical_pen',lambda:v5_pen.geometry(bk))
        hdr,depth=v5_pen.composite(hdr,G,bk,cam,L,mesh)
        return B.dof(hdr,depth.astype(np.float32),cam,cam.dof_k*self.W/1920),alpha


def parse_args(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--shot',choices=SHOTS,required=True)
    ap.add_argument('--frames',required=True)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--scale',type=float,default=1.)
    ap.add_argument('--format',choices=['png','jpg'],default='png')
    ap.add_argument('--skip',action='store_true')
    a=ap.parse_args(argv)
    if not a.out.is_absolute():ap.error('--out must be absolute')
    if not np.isfinite(a.scale) or a.scale<=0:ap.error('--scale must be positive and finite')
    try:fs=BC.frames_of(a.frames)
    except ValueError:ap.error('invalid frame range')
    lo,hi=SHOTS[a.shot]
    if not fs or any(f<lo or f>=hi for f in fs):ap.error(f'frames must fall in {lo}–{hi-1}')
    a.frame_list=sorted(set(fs))
    return a


def save(path,rgb,fmt):
    if fmt=='png':BC.look.save_png(path,rgb);return
    from PIL import Image
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.tmp')
    try:
        Image.fromarray(np.rint(np.clip(rgb,0,1)*255).astype(np.uint8)).save(temp,format='JPEG',quality=95,subsampling=0)
        os.replace(temp,path)
    finally:
        if temp.exists():temp.unlink()


def main(argv=None):
    a=parse_args(argv)
    renderer=PagesV5(a.shot,int(1920*a.scale),int(804*a.scale))
    matte=Path(str(a.out).rstrip('/')+'_matte')
    for f in a.frame_list:
        path=a.out/f'f_{f:05d}.{a.format}';mp=matte/path.name
        if a.skip and path.exists() and mp.exists():continue
        start=time.perf_counter()
        hdr,alpha=renderer.frame(f)
        rgb=BC.look.finish(hdr,exposure=1.15,bloom_strength=.06,bloom_threshold=1.2,vignette_amount=.32)
        save(path,rgb,a.format);save(mp,np.repeat(np.clip(alpha,0,1)[...,None],3,-1),a.format)
        print(json.dumps(dict(frame=f,shot=a.shot,seconds=time.perf_counter()-start,
             peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024),
             opencv_threads=cv2.getNumThreads())),flush=True)


if __name__=='__main__':main()
