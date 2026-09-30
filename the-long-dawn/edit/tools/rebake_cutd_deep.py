"""Build explicit local D Deep assets; run heavy modes through the shared render slot.

No output defaults to the shared renders tree. Clean plates and footage-derived coefficients stay local;
the runtime compositor reads them through CUTD_LOCAL_RENDERS and holds its explicitly adopted race plate.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import shutil
import sys
import time

os.environ.setdefault('NUMBA_NUM_THREADS','2')
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'edit'),str(ROOT/'shots/map'),str(ROOT/'lib')]
import cv2
import numpy as np
cv2.setNumThreads(1)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path,rgb):
    path.parent.mkdir(parents=True,exist_ok=True)
    pixels=(np.clip(rgb,0.,1.)*255+.5).astype(np.uint8)
    if pixels.ndim==3:
        pixels=pixels[...,::-1]
    if not cv2.imwrite(str(path),pixels,[cv2.IMWRITE_PNG_COMPRESSION,3]):
        raise OSError(str(path))


def frames(spec):
    out=[]
    for part in spec.split(','):
        a,sep,b=part.partition('-')
        out.extend(range(int(a),int(b)+1) if sep else [int(a)])
    return sorted(set(out))


def validate_output(out):
    if out.resolve().is_relative_to((ROOT/'renders').resolve()):
        raise ValueError('Deep assets must stay outside the shared renders tree')


def clean(args):
    import book_c
    import look
    renderer=book_c.Book3(1920,804)
    renderer.no_burn=True
    rows=[]
    for f in frames(args.frames or '1680-1717'):
        if not 1680 <= f <1718:
            raise ValueError('Clean sweep source frames must be C1680–1717')
        start=time.perf_counter()
        hdr,alpha=renderer.frame(f)
        rgb=look.finish(hdr,exposure=1.15,bloom_strength=.06,bloom_threshold=1.2,vignette_amount=.32)
        path=args.out/'cutd_deep_clean'/f'f_{f+1040:05d}.png'
        save(path,rgb)
        row=dict(source_frame=f,frame=f+1040,seconds=time.perf_counter()-start,
                 peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                 alpha_min=float(np.min(alpha)),bytes=path.stat().st_size,sha256=digest(path))
        rows.append(row)
        print(json.dumps(row),flush=True)
    return rows


def plates(args):
    rows=[]
    for source,stem,frame in ((args.race,'cutd_deep_race',2719),(args.brink,'cutd_deep_brink',2960)):
        if source is None:
            raise ValueError('plates needs explicit --race and --brink paths')
        image=cv2.imread(str(source),cv2.IMREAD_COLOR)
        if image is None or image.shape!=(804,1920,3):
            raise ValueError(f'Expected native1920x804 source: {source}')
        target=args.out/stem/f'f_{frame:05d}.png'
        target.parent.mkdir(parents=True,exist_ok=True)
        save(target,image[...,::-1].astype(np.float32)/255.)
        rows.append(dict(stem=stem,frame=frame,source=str(source),source_sha256=digest(source),
                         output_sha256=digest(target),status='STAND-IN'))
    return rows


def exit_layers(args):
    import ftburn as ft
    rows=[]
    for f in range(2945,2981):
        source=int(round(1905+(f-2945)*86/35))
        rgb=Path(ft.frame_path(str(ROOT/'renders/book_C_ft'),source))
        matte=Path(ft.frame_path(str(ROOT/'renders/book_C_ft_matte'),source))
        if not rgb.is_file() or not matte.is_file():
            raise FileNotFoundError(f'Missing original filmed exit source{source}')
        image=cv2.imread(str(rgb),cv2.IMREAD_COLOR)
        cover=cv2.imread(str(matte),cv2.IMREAD_GRAYSCALE)
        save(args.out/'cutd_deep_exit'/f'f_{f:05d}.png',image[...,::-1].astype(np.float32)/255.)
        save(args.out/'cutd_deep_exit_matte'/f'f_{f:05d}.png',cover.astype(np.float32)/255.)
        rows.append(dict(frame=f,source_frame=source,rgb_sha256=digest(rgb),cover_sha256=digest(matte),
                         cover_mean=float(np.mean(cover)/255.),cover_max=int(np.max(cover))))
    return rows


def coefficients(args):
    import ftburn as ft
    if args.footage is None:
        raise ValueError('coeff needs an explicit --footage directory with the original clips')
    ft.FOOT=str(args.footage)
    ft.FF=shutil.which('ffmpeg') or ft.FF
    sh=dict(ft.SHOTS['sweep'])
    sequence=frames(args.frames or '1680-1717')
    allframes=list(range(1680,1718))
    taus,birth=ft.retime(sh,allframes)
    footage=ft.FootageBurn(str(args.footage/sh['clip']),sh['roi'],sh['plate_frame'],sh.get('scale'))
    bank=ft.FlameBank()
    rows=[]
    try:
        for f in sequence:
            start=time.perf_counter()
            plate=args.out/'cutd_deep_clean'/f'f_{f+1040:05d}.png'
            page=cv2.imread(str(plate),cv2.IMREAD_COLOR)
            if page is None:
                raise FileNotFoundError(f'Render the exact clean page first: {plate.name}')
            page=ft.to_lin(page[...,::-1].astype(np.float32)/255.)
            hole,flame,dist=footage.advance(taus[f])
            fx,fy=ft.screen_maps(sh,f)
            hs=cv2.remap(hole,fx,fy,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=0)
            fl=cv2.remap(flame,fx,fy,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=0)
            ds=cv2.remap(dist,fx,fy,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=99.)
            nz=cv2.remap(footage.noise,fx,fy,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT)
            fl*=float(np.clip((f-birth+4)/4.,0.,1.))
            speed=cv2.remap(footage.speed(),fx,fy,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=0)
            tongues=ft.flame_tongues(hs,speed,fx,fy,bank,f)
            fl+=tongues+cv2.GaussianBlur(tongues,(0,0),10)*.35
            lum=fl@np.float32([.5,.35,.15])
            small=cv2.resize(lum,(480,201),interpolation=cv2.INTER_AREA)
            pool=cv2.resize(cv2.GaussianBlur(small,(0,0),18),(1920,804),interpolation=cv2.INTER_LINEAR)
            pool=pool[...,None]*np.float32([1.,.62,.3])*1.2
            gain=ft.char_keep(ds*(1.+.3*np.tanh(nz)))*(1.+pool)*(1.-hs[...,None])
            base=page*(1.+.5*pool)*hs[...,None]+fl*1.6*np.float32([1.,.8,.55])
            target=args.out/'cutd_deep_sweep_coeff'/f'f_{f+1040:05d}.npz'
            target.parent.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(target,gain=gain.astype(np.float16),base=base.astype(np.float16))
            row=dict(frame=f+1040,source_frame=f,clip_frame=taus[f],seconds=time.perf_counter()-start,
                     bytes=target.stat().st_size,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                     clean_sha256=digest(plate),output_sha256=digest(target),
                     gain_float16_error=float(np.max(np.abs(gain-gain.astype(np.float16).astype(np.float32)))),
                     base_float16_error=float(np.max(np.abs(base-base.astype(np.float16).astype(np.float32)))))
            rows.append(row)
            print(json.dumps(row),flush=True)
    finally:
        footage.reader.close()
    return rows


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=('clean','coeff','exit','plates'))
    p.add_argument('--out',required=True,type=Path)
    p.add_argument('--frames',default='')
    p.add_argument('--footage',type=Path)
    p.add_argument('--race',type=Path)
    p.add_argument('--brink',type=Path)
    p.add_argument('--report',required=True,type=Path)
    args=p.parse_args()
    validate_output(args.out)
    args.out.mkdir(parents=True,exist_ok=True)
    (args.out/'.gitignore').write_text('*.png\n*.jpg\n*.npz\n')
    rows={'clean':clean,'coeff':coefficients,'exit':exit_layers,'plates':plates}[args.mode](args)
    report=dict(mode=args.mode,rows=rows,code_sha256=digest(__file__))
    if args.mode=='coeff':
        report['footage']={p.name:digest(p) for p in [args.footage/'pexels_8828892_1080.mp4',
                                                   args.footage/'pexels_8828898_1080.mp4']}
        report['ftburn_sha256']=digest(ROOT/'shots/map/ftburn.py')
    args.report.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    main()
