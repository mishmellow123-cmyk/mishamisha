"""Build explicit local D Deep assets; run heavy modes through the shared render slot.

No output defaults to the shared renders tree. Clean plates and footage-derived coefficients stay local;
the runtime compositor reads them through CUTD_LOCAL_RENDERS and holds its explicitly adopted race plate.
"""
import argparse
from datetime import datetime
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
NATIVE_SHAPE=(804,1920,3)


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


def refresh_gate(args):
    """Read-only race gate; the contract's pre-export float hash differs from the delivered PNG hash."""
    import assemble as AS
    import cutd_adoption as adoption
    if any(getattr(args,key,None) is None for key in ('reuse_from','contract','owner_state','race_after')):
        raise ValueError('refresh-race requires --reuse-from, --contract, --owner-state and --race-after')
    threshold=datetime.fromisoformat(args.race_after)
    if threshold.tzinfo is None:
        raise ValueError('--race-after must include its UTC offset')
    if args.out.resolve().is_relative_to(args.reuse_from.resolve()):
        raise ValueError('Refresh output must be separate from the immutable reused asset root')
    contract=json.loads(args.contract.read_text())
    if (contract.get('schema'),contract.get('frame'),contract.get('farm_plate'),contract.get('native_dimensions')) != (
            'openring.race-ending.r5',2719,'renders/embers_D_race/f_02719.jpg',[1920,804]):
        raise ValueError('Expected the R5 native race-ending contract for farm D2719')
    native=args.contract.parent/contract['native_plate']
    if native.parent.resolve()!=args.contract.parent.resolve():
        raise ValueError('Native race contract path mismatch')
    native_image=cv2.imread(str(native),cv2.IMREAD_COLOR)
    if native_image is None or native_image.shape!=NATIVE_SHAPE:
        raise ValueError('Contract native plate must decode at1920x804')
    native_pixels=hashlib.sha256(native_image[...,::-1].tobytes()).hexdigest()
    receipt_path=args.contract.parent/'delivery_consistency.json'
    receipt=json.loads(receipt_path.read_text())
    files=receipt.get('artifact_sha256',{})
    # The native producer writes quantized8-bit PNG but hashes its pre-export float render separately.
    if (receipt.get('pass_all') is not True or files.get(native.name)!=digest(native)
            or files.get(args.contract.name)!=digest(args.contract)):
        raise ValueError('Native race/contract file hashes do not match the delivery receipt')
    edl_path=ROOT/'edit/edl_v3.py'
    expected=json.loads(args.owner_state.read_text())['owner_edl_sha256']
    if digest(edl_path)!=expected:
        raise ValueError('Owner EDL differs from the initial-state hash; no refresh was performed')
    inventory=next(row for row in adoption.availability(ROOT/'renders') if row['row']=='D10')
    farm=ROOT/contract['farm_plate']
    reasons=[]
    if (inventory['needed'],inventory['available'],inventory['missing'],inventory['complete'])!=(320,320,0,True):
        reasons.append(f"race delivery incomplete: {inventory['available']}/320")
    mtime_ns=farm.stat().st_mtime_ns if farm.is_file() else None
    if mtime_ns is None or mtime_ns <= int(threshold.timestamp()*1_000_000_000):
        reasons.append('farm D2719 is missing or not newer than the required refresh timestamp')
    if not reasons:
        selected,_=AS.locate(AS.EDL.D_DEEP_RACE_SOURCE,'D',None,2719)
        held,_=AS.locate_under(AS.EDL.D_DEEP_RACE_UNDER,2720)
        if not selected or not held or Path(selected).resolve()!=farm.resolve() or Path(held).resolve()!=farm.resolve():
            reasons.append('owner adoption does not select the required farm JPEG for both D10 and Deep')
    farm_dimensions=None
    if not reasons:
        # Ctx.read resizes inputs, so only a raw decode can establish the farm delivery's native dimensions.
        farm_image=cv2.imread(str(farm),cv2.IMREAD_COLOR)
        if farm_image is None or farm_image.shape!=NATIVE_SHAPE:
            raise ValueError('Farm D2719 must decode natively at1920x804 before any Deep assets are copied')
        farm_dimensions=[farm_image.shape[1],farm_image.shape[0]]
    return dict(status='pending' if reasons else 'ready',reasons=reasons,inventory=inventory,
                race_after=args.race_after,farm_path=str(farm),farm_mtime_ns=mtime_ns,
                farm_sha256=digest(farm) if not reasons else None,farm_dimensions=farm_dimensions,
                native_path=str(native),native_pixel_sha256=native_pixels,native_file_sha256=digest(native),
                declared_render_buffer_sha256=contract['render_sha256'],
                contract_sha256=digest(args.contract),delivery_receipt_sha256=digest(receipt_path),
                owner_edl_sha256=expected)


def reuse_assets(source,out):
    """Copy exact independent bytes; never mutate existing assets or follow an output link elsewhere."""
    inventory=[]
    for stem,first,last,suffix in (('cutd_deep_clean',2720,2757,'png'),
                                   ('cutd_deep_sweep_coeff',2720,2757,'npz'),
                                   ('cutd_deep_exit',2945,2980,'png'),
                                   ('cutd_deep_exit_matte',2945,2980,'png')):
        for frame in range(first,last+1):
            relative=Path(stem)/f'f_{frame:05d}.{suffix}'
            original,target=source/relative,out/relative
            if not original.is_file():
                raise FileNotFoundError(f'Reused Deep asset missing: {relative}')
            if not target.resolve().is_relative_to(out.resolve()):
                raise ValueError('A refresh destination resolves outside its local asset root')
            fingerprint=digest(original)
            if target.exists() and digest(target)!=fingerprint:
                raise ValueError(f'Refresh refuses to overwrite a different existing asset: {relative}')
            inventory.append(dict(file=str(relative),sha256=fingerprint,bytes=original.stat().st_size))
    for row in inventory:
        target=out/row['file']
        target.parent.mkdir(parents=True,exist_ok=True)
        if not target.exists():
            shutil.copy2(source/row['file'],target)
        if digest(target)!=row['sha256']:
            raise ValueError(f"Copied Deep asset failed its byte check: {row['file']}")
    return inventory


def refresh_runtime(out,reused,farm):
    """Compose all38 native entry frames and compare the actual glow output across the root switch."""
    import assemble as AS
    prior=os.environ.get('CUTD_LOCAL_RENDERS')
    glow_frames=(2945,2960,2980)
    def context(folder):
        os.environ['CUTD_LOCAL_RENDERS']=str(folder)
        AS._INDEX.clear()
        AS._init('D',None,1.,True,True)
        return AS._CTX
    def glow_hash(ctx,frame):
        image,_,status,source=ctx.picture(frame)
        if source is None or 'deep_reveal' not in status or status.startswith('SLATE'):
            raise ValueError(f'D{frame} did not produce the actual Deep glow transition')
        return hashlib.sha256(image.tobytes()).hexdigest()
    try:
        ctx=context(reused)
        before={f:glow_hash(ctx,f) for f in glow_frames}
        del ctx
        ctx=context(out)
        for code in ('D11a','D11b'):
            i=next(i for i,row in enumerate(ctx.shots) if row['code']==code)
            if ctx.plans[i]['take']!=AS.EDL.D_DEEP_ENTRY:
                raise ValueError(f'{code} did not select the independent Deep entry')
        race=ctx.read(str(farm))
        if race is None or race.shape!=NATIVE_SHAPE:
            raise ValueError('Fresh farm race must decode at native1920x804')
        if not np.array_equal(ctx.take_frame(AS.EDL.D_DEEP_ENTRY,2720),race):
            raise ValueError('Deep first frame does not exactly reproduce the fresh held race')
        rows=[]
        for frame in range(2720,2758):
            start=time.perf_counter()
            picture,_,status,source=ctx.picture(frame)
            if source!=AS.EDL.D_DEEP_ENTRY or status.startswith('SLATE'):
                raise ValueError(f'D{frame} failed to compose the adopted Deep entry')
            image=(np.clip(picture,0.,1.)*255.+.5).astype(np.uint8)
            if image.shape!=NATIVE_SHAPE:
                raise ValueError('Deep refresh did not produce native1920x804 output')
            rows.append(dict(frame=frame,pixel_sha256=hashlib.sha256(image.tobytes()).hexdigest(),
                             seconds=time.perf_counter()-start,
                             peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
            if frame in (2720,2724,2740,2757):
                save(out.parent/f'D_deep_refresh_{frame}.png',image.astype(np.float32)/255.)
        glow=[]
        for frame in glow_frames:
            after=glow_hash(ctx,frame)
            if after!=before[frame]:
                raise ValueError(f'Local root refresh changed glow frame D{frame}')
            glow.append(dict(frame=frame,before_sha256=before[frame],after_sha256=after))
        return dict(frames=rows,glow_identity=glow,first_frame_equals_farm=True)
    finally:
        if prior is None:
            os.environ.pop('CUTD_LOCAL_RENDERS',None)
        else:
            os.environ['CUTD_LOCAL_RENDERS']=prior
        AS._INDEX.clear()


def refresh_race(args):
    started=time.perf_counter()
    def measured(report,ran):
        return dict(report,total_seconds=time.perf_counter()-started,
                    peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                    native_runtime_ran=ran)
    gate=refresh_gate(args)
    if gate['status']=='pending':
        return measured(dict(gate,runtime=None),False)
    assets=reuse_assets(args.reuse_from,args.out)
    (args.out/'.gitignore').write_text('*.png\n*.jpg\n*.npz\n')
    result=refresh_runtime(args.out,args.reuse_from,Path(gate['farm_path']))
    if digest(ROOT/'edit/edl_v3.py')!=gate['owner_edl_sha256']:
        raise ValueError('Owner EDL changed during the refresh')
    if digest(gate['farm_path'])!=gate['farm_sha256']:
        raise ValueError('Farm D2719 changed during the refresh')
    return measured(dict(gate,status='complete',reused_assets=assets,runtime=result),True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=('clean','coeff','exit','plates','refresh-race'))
    p.add_argument('--out',required=True,type=Path)
    p.add_argument('--frames',default='')
    p.add_argument('--footage',type=Path)
    p.add_argument('--race',type=Path)
    p.add_argument('--brink',type=Path)
    p.add_argument('--report',required=True,type=Path)
    p.add_argument('--reuse-from',type=Path,help='immutable previous Deep local-render root')
    p.add_argument('--contract',type=Path,help='R5 race-ending contract JSON beside its native plate')
    p.add_argument('--owner-state',type=Path,help='initial-state JSON pinning owner_edl_sha256')
    p.add_argument('--race-after',help='strict farm D2719 modification-time floor, ISO8601 with UTC offset')
    args=p.parse_args()
    validate_output(args.out)
    if args.mode=='refresh-race':
        report=refresh_race(args)
        report['code_sha256']=digest(__file__)
        args.report.write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps({key:report[key] for key in ('status','reasons','inventory')}),flush=True)
        return 2 if report['status']=='pending' else 0
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
    sys.exit(main())
