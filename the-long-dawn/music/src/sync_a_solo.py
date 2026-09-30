"""Reconcile owner-supplied pre-guard NPZ solos with a delivered SFX stem.

This is measurement, not an audio renderer: it writes no audio/cache/output files.
The source master applies guard -> causal 8 Hz highpass -> linked master envelope ->
fades -> optional constant true-peak gain -> dither. A raw per-channel quotient
would fit any stereo signal and is deliberately never accepted as provenance.
Only a complete, numerically reconstructed stem can return an attributable bank.

Audio is read in one-second chunks; only scalar gain curves span the whole film.
The NPZ loader holds one cue at a time. A temporary mmap holds the decomposition
sum and is removed before return. All production invocation must use onepy.
"""
from __future__ import annotations
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import tempfile

import numpy as np
from scipy import signal
from scipy.ndimage import minimum_filter1d

SR = 48000
BLOCK = SR
# PCM24 output: dither <=1 LSB, conversion <=1 LSB; float32 guard/HP/
# envelope and independently summed solos need additional arithmetic allowance.
# This ceiling is fixed before looking at a target, not fitted to its residual.
RECONSTRUCTION_ATOL = 2e-6
DECOMPOSITION_ATOL = 1e-6


def file_state(path):
    s=Path(path).stat()
    return s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns


def file_hash(path):
    before=file_state(path)
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(262144), b''):
            h.update(chunk)
    if file_state(path)!=before:
        raise ValueError(f'input changed while hashing: {path}')
    return h.hexdigest()


def read_audio(source, start, stop):
    x=source.read(start,stop)
    if x.shape!=(stop-start,2) or not np.isfinite(x).all():
        raise ValueError('delivered stem contains missing or non-finite samples')
    return x


def _array(path, shape=None):
    x = np.load(path, mmap_mode='r', allow_pickle=False)
    if x.dtype != np.float32 or (shape is not None and x.shape != shape):
        raise ValueError(f'invalid float32 array shape: {path}')
    return x


def _load_solo(folder, item, total):
    with np.load(folder / (item['id'] + '.npz'), allow_pickle=False) as z:
        start, data = int(z['start']), z['data']
    if (data.dtype != np.float32 or data.ndim != 2 or data.shape[1] != 2 or
            start != item['start'] or start + len(data) != item['stop'] or
            start < 0 or start + len(data) > total or not np.isfinite(data).all()):
        raise ValueError(f'invalid solo extent/data: {item["id"]}')
    return start, data


def decomposition(folder, index, full):
    """Load each compressed NPZ once; sum on disk rather than in a film array."""
    with tempfile.TemporaryDirectory(prefix='syncA_solo_sum_') as td:
        acc = np.lib.format.open_memmap(Path(td) / 'sum.npy', mode='w+',
                                       dtype=np.float32, shape=full.shape)
        acc[:] = 0
        hashes = {}
        for item in index['items']:
            start, data = _load_solo(folder, item, len(full))
            hashes[item['id']] = file_hash(folder / (item['id'] + '.npz'))
            for a in range(0, len(data), BLOCK):
                b = min(a + BLOCK, len(data))
                acc[start+a:start+b] += data[a:b]
            del data
        maximum, ss, count = 0., 0., 0
        for a in range(0, len(full), BLOCK):
            d = np.asarray(full[a:a+BLOCK], dtype=np.float64) - acc[a:a+BLOCK]
            if not np.isfinite(d).all():
                raise ValueError('nonfinite decomposition data')
            maximum = max(maximum, float(np.max(np.abs(d))))
            ss += float(np.sum(d*d)); count += d.size
        del acc
    return dict(ok=maximum <= DECOMPOSITION_ATOL, maximum_error=maximum,
                rms_error=float(np.sqrt(ss/count)), tolerance=DECOMPOSITION_ATOL,
                samples_checked=len(full), cue_sha256=hashes)


def integrated_loudness(score, effects):
    """Same pyloudnorm K-weighting/gates as render_v2.lufs, streamed exactly.

    Window indices duplicate pyloudnorm's floating-point index calculation,
    including occasional 1-sample boundaries rather than assuming ideal 4800 hops.
    """
    import pyloudnorm as pyln
    meter = pyln.Meter(SR)
    stages = list(meter._filters.values())
    states = [np.zeros((max(len(s.a), len(s.b))-1, 2)) for s in stages]
    n = len(score); tg = meter.block_size; step = 1 - meter.overlap
    num = int(np.round(((n/SR - tg)/(tg*step)))+1)
    lo = np.array([int(tg*(j*step)*SR) for j in range(num)])
    hi = np.array([int(tg*(j*step+1)*SR) for j in range(num)])
    energies = np.zeros((num, 2))
    for a in range(0, n, BLOCK):
        b = min(a+BLOCK, n)
        # The actual peak-guard caller adds its two float32 arrays before lufs.
        y = (score[a:b] + effects[a:b]).astype(np.float64)
        if not np.isfinite(y).all():raise ValueError('non-finite premaster loudness input')
        for i, stage in enumerate(stages):
            y, states[i] = signal.lfilter(stage.b, stage.a, y, axis=0, zi=states[i])
            y *= stage.passband_gain
        c = np.vstack([np.zeros((1,2)), np.cumsum(y*y, axis=0)])
        use = np.flatnonzero((lo < b) & (hi > a))
        energies[use] += c[np.minimum(hi[use],b)-a] - c[np.maximum(lo[use],a)-a]
    z = energies/(tg*SR)
    with np.errstate(divide='ignore'):
        levels = -.691 + 10*np.log10(z.sum(axis=1))
        relative = -.691 + 10*np.log10(z[levels >= -70].mean(axis=0).sum()) - 10
        result = -.691 + 10*np.log10(z[(levels > -70)&(levels > relative)].mean(axis=0).sum())
    if not np.isfinite(result):
        raise ValueError('cannot measure finite premaster loudness')
    return float(result)


def guard_curve(score, full, loudness):
    """Replay sound_v3.peak_guard's scalar control equation, without synthesizing audio."""
    gain = 10**((-16-loudness)/20) * 10**(1.2/20)
    lim = 10**(-2.3/20)/gain
    gr = np.zeros(len(full), np.float32)
    over_count = 0
    for a in range(0, len(full), BLOCK):
        b = min(a+BLOCK,len(full)); s, x = score[a:b], full[a:b]
        sa = np.abs(s).max(axis=1); total = np.abs(s+x).max(axis=1)
        over = (total > lim)&(sa < lim); over_count += int(over.sum())
        need = np.ones(b-a, np.float32)
        need[over] = np.clip((lim*.97-sa[over])/(np.abs(x).max(axis=1)[over]+1e-12), 10**(-10/20), 1)
        gr[a:b] = 20*np.log10(need)
    gr = minimum_filter1d(gr, size=int(.010*SR)+1)
    coeff = np.exp(-1/(.12*SR)); state = np.zeros(1)
    curve = np.empty(len(full), np.float32)
    for a in range(0, len(full), BLOCK):
        b = min(a+BLOCK,len(full))
        sm, state = signal.lfilter([1-coeff],[1,-coeff],gr[a:b],zi=state)
        curve[a:b] = np.power(10., np.minimum(gr[a:b],sm)/20).astype(np.float32)
    return curve, dict(premaster_lufs=loudness, guard_gain=gain,
                       guarded_samples=over_count, minimum_guard_gain=float(curve.min()))


def finish_chunk(filtered, envelope, start, total, safety=1.):
    y = (filtered.astype(np.float32)*envelope[:,None]).astype(np.float32)
    idx = np.arange(start,start+len(y)); fi, fo = int(.010*SR), int(.35*SR)
    first = idx < fi
    if first.any():
        y[first] *= np.linspace(0,1,fi,dtype=np.float32)[idx[first],None]
    last = idx >= total-fo
    if last.any():
        y[last] *= (np.cos(np.linspace(0,np.pi/2,fo))**2).astype(np.float32)[idx[last]-(total-fo),None]
    if safety != 1:
        y *= np.float32(safety)
    return y


def candidate_chunks(full, guard, env, safety=1.):
    b,a = signal.butter(1,8/(SR/2),btype='high'); state = np.zeros((1,2))
    for start in range(0,len(full),BLOCK):
        stop = min(start+BLOCK,len(full))
        raw = (full[start:stop]*guard[start:stop,None]).astype(np.float32)
        if not np.isfinite(raw).all() or not np.isfinite(env[start:stop]).all():
            raise ValueError('non-finite premaster or master gain')
        hp,state = signal.lfilter(b,a,raw,axis=0,zi=state)
        out=finish_chunk(hp,env[start:stop],start,len(full),safety)
        if not np.isfinite(out).all():raise ValueError('non-finite master-chain output')
        yield start,out


def shared_gain_metrics(x,y):
    """Stereo-vector fit plus genuinely held-out opposite-channel prediction."""
    x,y = np.asarray(x,np.float64),np.asarray(y,np.float64)
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('non-finite shared-gain input')
    den = (x*x).sum(axis=1)
    g = np.divide((x*y).sum(axis=1),den,out=np.zeros(len(x)),where=den>1e-20)
    residual = y-x*g[:,None]
    cross = {}
    for source,target in [(0,1),(1,0)]:
        use = ((np.abs(x[:,source]) >= 1e-3)&(np.abs(x[:,target]) >= 1e-3)&
               (np.abs(x[:,source]) >= .25*np.abs(x[:,target]))&
               (np.abs(x[:,target]) >= .25*np.abs(x[:,source])))
        error = y[use,target] - y[use,source]/x[use,source]*x[use,target]
        cross[f'{source}_to_{target}'] = dict(samples=int(use.sum()),
                    maximum_error=float(np.max(np.abs(error))) if len(error) else None,
                    rms_error=float(np.sqrt(np.mean(error*error))) if len(error) else None)
    conditioned = den >= 2e-6
    gains = g[conditioned]
    good_pairs = conditioned[1:]&conditioned[:-1]
    delta = np.diff(g)[good_pairs]
    return dict(vector_maximum_error=float(np.max(np.abs(residual))),
                vector_rms_error=float(np.sqrt(np.mean(residual*residual))),
                conditioned_samples=int(conditioned.sum()),
                gain_min=float(gains.min()) if len(gains) else None,
                gain_max=float(gains.max()) if len(gains) else None,
                gain_median=float(np.median(gains)) if len(gains) else None,
                maximum_adjacent_gain_change=float(np.max(np.abs(delta))) if len(delta) else None,
                cross_channel=cross)


class SoloAudio:
    samplerate, channels, subtype = SR, 2, 'FLOAT'

    def __init__(self, bank, item):
        self.frames = bank.n
        self.start, raw = _load_solo(bank.folder,item,bank.n)
        self.start = max(0,self.start)
        # One second of HP tail is < exp(-2*pi*8) of its initial state.
        stop = min(bank.n,self.start+len(raw)+SR)
        self.data = np.zeros((stop-self.start,2),np.float32)
        b,a = signal.butter(1,8/(SR/2),btype='high'); state = np.zeros((1,2))
        for i in range(0,len(self.data),BLOCK):
            j=min(i+BLOCK,len(self.data)); start=self.start+i
            source=np.zeros((j-i,2),np.float32)
            count=max(0,min(j,len(raw))-i)
            if count: source[:count]=raw[i:i+count]
            source *= bank.guard[start:start+j-i,None]
            hp,state=signal.lfilter(b,a,source,axis=0,zi=state)
            self.data[i:j]=finish_chunk(hp,bank.env[start:start+j-i],start,bank.n,bank.safety)
        if not np.isfinite(self.data).all():raise ValueError('non-finite processed cue')

    def read(self,start,stop):
        if start < 0 or stop < start or stop > self.frames:
            raise ValueError('invalid cue read extent')
        out=np.zeros((stop-start,2),np.float64)
        a,b=max(start,self.start),min(stop,self.start+len(self.data))
        if b>a:out[a-start:b-start]=self.data[a-self.start:b-self.start]
        return out


class SoloBank:
    def __init__(self, folder, index, guard, env, safety):
        self.folder, self.guard, self.env, self.safety = folder,guard,env,safety
        self.n=index['n']; self.items={v['id']:v for v in index['items']}
        self.input_states={}

    @contextmanager
    def open(self,eid):
        for path,state in self.input_states.items():
            if file_state(path)!=state:
                raise ValueError(f'reconciled input changed: {path}')
        source=SoloAudio(self,self.items[eid])
        for path,state in self.input_states.items():
            if file_state(path)!=state:
                raise ValueError(f'reconciled input changed while opening cue: {path}')
        try: yield source
        finally: del source

    def close(self):
        self.guard=None; self.env=None


def processed_decomposition(bank, sfx):
    """Independently sum the processed cue adapters and compare with the actual stem."""
    with tempfile.TemporaryDirectory(prefix='syncA_solo_postsum_') as td:
        acc=np.lib.format.open_memmap(Path(td)/'sum.npy',mode='w+',dtype=np.float32,shape=(bank.n,2))
        acc[:]=0
        for eid in bank.items:
            with bank.open(eid) as cue:
                for i in range(0,len(cue.data),BLOCK):
                    j=min(i+BLOCK,len(cue.data));a=cue.start+i
                    acc[a:a+j-i]+=cue.data[i:j]
        maximum=ss=0.;count=0
        for a in range(0,bank.n,BLOCK):
            b=min(a+BLOCK,bank.n);d=read_audio(sfx,a,b)-acc[a:b]
            maximum=max(maximum,float(np.abs(d).max()));ss+=float(np.sum(d*d));count+=d.size
        del acc
    return dict(ok=maximum<=RECONSTRUCTION_ATOL,maximum_error=maximum,
                rms_error=float(np.sqrt(ss/count)),tolerance=RECONSTRUCTION_ATOL,
                samples_checked=bank.n,cues_checked=len(bank.items))


def reconcile(solo_dir, sfx, expected_ids, *, diagnostic_bank=False):
    """Return (report, bank); failures return no bank unless diagnostic_bank=True.

    That explicit research-only option never changes check['ok']; the bank then
    carries attribution_validated=False and must not certify a named onset.
    Identity/format/decomposition failures never return a bank, even diagnostically.
    """
    check=dict(check='pre_guard_solo_reconstruction',ok=False)
    try:
        folder=Path(solo_dir); index_state=file_state(folder/'index.json')
        index=json.loads((folder/'index.json').read_text())
        if file_state(folder/'index.json')!=index_state:
            raise ValueError('index changed while reading')
        ids=[v['id'] for v in index['items']]
        if len(ids)!=len(set(ids)) or set(ids)!=set(expected_ids) or len(expected_ids)!=len(set(expected_ids)):
            raise ValueError('solo ids must match every delivered metadata row exactly once')
        n=index['n']
        if index['sr']!=SR or sfx.frames!=n or sfx.channels!=2 or sfx.samplerate!=SR:
            raise ValueError('solo/master format or duration mismatch')
        input_paths=[folder/'index.json',folder/'full_pre_guard.npy']+[folder/(eid+'.npz') for eid in ids]
        input_states={p:file_state(p) for p in input_paths}
        if input_states[folder/'index.json']!=index_state:raise ValueError('index changed')
        full=_array(folder/'full_pre_guard.npy',(n,2))
        check['decomposition']=decomposition(folder,index,full)
        if not check['decomposition']['ok']:raise ValueError('solo sum differs from full_pre_guard')
        # source.path is music/out/v3/sound_{cut}_sfx.wav (possibly via symlink).
        stem=Path(sfx.path); tag=stem.stem.removesuffix('_sfx').removeprefix('sound_')
        cache=stem.parent.parent.parent/'cache'/'v3'
        envpath=cache/f'master_env_sound_{tag}.npy'
        scorepath=cache/f'premaster_score_final_{tag}.npy'
        input_states.update({p:file_state(p) for p in [envpath,scorepath,stem]})
        env=_array(envpath,(n,)); score=_array(scorepath,(n,2))
        check['sha256']={str(p):file_hash(p) for p in [folder/'index.json',folder/'full_pre_guard.npy',envpath,scorepath,stem]}
        loudness=integrated_loudness(score,full)
        guard,check['guard']=guard_curve(score,full,loudness)
        # Only one extra global scalar is permitted: source master's true-peak
        # safety trim. Estimate jointly on both channels, validate every sample.
        xy=xx=0.; local_gains=[]
        for a,x in candidate_chunks(full,guard,env):
            y=read_audio(sfx,a,a+len(x)); ex=float(np.sum(x.astype(np.float64)**2)); ey=float(np.sum(x*y))
            xx+=ex; xy+=ey
            if ex > len(x)*2*1e-8: local_gains.append(ey/ex)
        safety=float(np.median(local_gains)) if local_gains else float('nan')
        if abs(safety-1.) <= 1e-6: safety=1.
        global_gain=xy/xx if xx else float('nan')
        check['global_ls_gain']=float(global_gain) if np.isfinite(global_gain) else None
        check['safety_gain_method']='median per-second joint stereo LS; full-sample validation follows; within1e-6 of unity snapped to unity'
        check['safety_gain']=safety if np.isfinite(safety) else None
        if not np.isfinite(safety) or not 0<safety<=1.00001:
            raise ValueError('fitted safety trim is incompatible with source master')
        rows=[];maximum=ss=0.;count=0
        for a,x in candidate_chunks(full,guard,env,safety):
            y=read_audio(sfx,a,a+len(x));d=y-x
            maximum=max(maximum,float(np.abs(d).max()));ss+=float(np.sum(d*d));count+=d.size
            rows.append(dict(start_sample=a,stop_sample=a+len(x),frame=a/2000,
                 maximum_error=float(np.abs(d).max()),rms_error=float(np.sqrt(np.mean(d*d))),
                 frame_maximum_errors=[float(np.abs(d[i:i+2000]).max()) for i in range(0,len(d),2000)],
                 **shared_gain_metrics(x,y)))
        check.update(maximum_error=maximum,rms_error=float(np.sqrt(ss/count)),
                     tolerance=RECONSTRUCTION_ATOL,samples_checked=n,per_second=rows,
                     ok=maximum<=RECONSTRUCTION_ATOL)
        if any(file_state(p)!=state for p,state in input_states.items()):
            raise ValueError('an input changed during reconciliation')
        bank=SoloBank(folder,index,guard,env,safety)
        bank.input_states=input_states
        check['processed_solo_sum']=processed_decomposition(bank,sfx)
        if any(file_state(p)!=state for p,state in input_states.items()):
            raise ValueError('an input changed during processed-solo verification')
        check['ok']=check['ok'] and check['processed_solo_sum']['ok']
        if not check['ok']:
            check['reason']='named solos do not reconstruct delivered stem through the documented master chain'
            if not diagnostic_bank:
                bank.close()
                return check,None
        bank.attribution_validated=check['ok']
        return check,bank
    except (OSError,ValueError,KeyError,TypeError,OverflowError,FloatingPointError) as exc:
        check['ok']=False
        check['reason']=str(exc)
        return check,None
