"""Isolated SP compiler/material comparison. Run --help; no renderer runs during prepare."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

REFERENCE = '61cf28b'
SP_PATH = 'the-long-dawn/shots/run/sdfppl.py'
OVERRIDES = ('W15_P7', 'W15_CAM', 'W15_OWN_CAM', 'A14_CAM', 'A14_NOUG', 'NIGHT_CLOUD')


def sha(data):
    return hashlib.sha256(data).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def assert_cache_reload(cold, current):
    """A dispatcher cache hit alone does not establish output persistence."""
    for key in ('settings', 'source_sha256', 'harness_sha256', 'outputs'):
        assert cold[key] == current[key], f'Cache reload changed {key}'
    assert len(cold['sp_calls']) == len(current['sp_calls'])
    for old, new in zip(cold['sp_calls'], current['sp_calls']):
        for key in ('mode', 'name', 'repeat', 'call', 'input_hashes', 'img', 'zb', 'depth_changed_mask'):
            assert old[key] == new[key], f'Cache reload changed {key}'
    assert len(cold.get('frames', [])) == len(current.get('frames', []))
    for old, new in zip(cold.get('frames', []), current.get('frames', [])):
        for key in ('mode', 'frame', 'repeat', 'linear', 'finished', 'sp_calls', 'mblur'):
            assert old[key] == new[key], f'Cache reload changed frame {key}'


def assert_warm_frames(frames):
    initial = {}
    for frame in frames:
        key = (frame['mode'], frame['frame'])
        if frame['repeat'] == 0:
            initial[key] = frame
        else:
            for field in ('linear', 'finished'):
                assert frame[field] == initial[key][field], f'Warm full frame changed {field}: {key}'


def frame_requests(mode, frames):
    if mode == 'material':
        return []
    result = []
    for frame in (int(value) for value in frames.split(',')):
        shot = ('A14' if frame <= 4239 else 'A15') if mode == 'suite' else mode
        lo, hi = (3920, 4239) if shot == 'A14' else (4240, 4399)
        if not lo <= frame <= hi:
            raise ValueError(f'{shot} frame {frame} is outside {lo}..{hi}')
        result.append(dict(mode=shot, frame=frame))
    if not result or len({item['frame'] for item in result}) != len(result):
        raise ValueError('Supply a nonempty list of distinct cut frames.')
    return result


def settings(args):
    return dict(mode=args.mode, frames=args.frames, frame_requests=frame_requests(args.mode, args.frames),
                scale=args.scale, ss=args.ss, repeats=args.repeats, material_size=args.size)


def git(repo, *args):
    return subprocess.check_output(['git', '-C', str(repo), *args])


def prepare(args):
    root = args.out.resolve()
    if root.exists():
        raise SystemExit('Choose a fresh output directory; existing snapshots are never overwritten.')
    repo = args.repo.resolve()
    commit = git(repo, 'rev-parse', REFERENCE).decode().strip()
    names = git(repo, 'ls-tree', '-r', '--name-only', commit, 'the-long-dawn/lib',
                'the-long-dawn/shots/run', 'the-long-dawn/shots/montage').decode().splitlines()
    names = [n for n in names if Path(n).suffix in ('.py', '.npy', '.npz') and '/tests/' not in n]
    candidate = (repo / SP_PATH).read_bytes()
    records = {'reference': {}, 'candidate': {}}
    for name in names:
        original = git(repo, 'show', f'{commit}:{name}')
        for variant in records:
            data = candidate if variant == 'candidate' and name == SP_PATH else original
            target = root / 'sources' / variant / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
            records[variant][name] = sha(data)
    changed = [n for n in names if records['reference'][n] != records['candidate'][n]]
    if changed != [SP_PATH]:
        raise SystemExit(f'Expected only candidate SP to differ; found {changed}. Snapshot not runnable.')
    write_json(root / 'sources.json', dict(reference_commit=commit, source_sha256=records,
        changed_files=changed, harness_sha256=sha(Path(__file__).read_bytes()),
        candidate_worktree_head=git(repo, 'rev-parse', 'HEAD').decode().strip()))
    print(f'Prepared {len(names)} pinned files per variant at {root}', flush=True)


def run(args):
    root = args.out.resolve()
    result = root / 'results' / args.mode / args.variant / args.phase
    result.mkdir(parents=True, exist_ok=False)
    cache = root / 'caches' / args.mode / args.variant
    if args.phase == 'cold' and cache.exists() and any(cache.rglob('*')):
        raise SystemExit('Cold run requires an empty cache directory.')
    if args.phase == 'cache-hit' and not list(cache.rglob('*.nbc')):
        raise SystemExit('Cache-hit run requires the same variant/mode cold run first.')
    if args.phase == 'cache-hit':
        cold = json.loads((result.parent / 'cold' / 'receipt.json').read_text())
        requested = settings(args)
        if cold['status'] != 'complete' or cold['settings'] != requested:
            raise SystemExit('Cache-hit run must repeat a completed cold run with identical settings.')
    cache.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    for key in ('NUMBA_NUM_THREADS', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                'VECLIB_MAXIMUM_THREADS'):
        env[key] = '1'
    env.update(NUMBA_CACHE_DIR=str(cache), PYTHONDONTWRITEBYTECODE='1')
    env.pop('NUMBA_DISABLE_JIT', None)
    env.pop('PYTHONPATH', None)
    for key in OVERRIDES:
        env.pop(key, None)
    command = [sys.executable, '-B', str(Path(__file__).resolve()), '_worker', '--out', str(root),
               '--variant', args.variant, '--mode', args.mode, '--phase', args.phase,
               '--frames', args.frames, '--scale', str(args.scale), '--ss', str(args.ss),
               '--repeats', str(args.repeats), '--size', str(args.size)]
    started = time.perf_counter()
    child = subprocess.Popen(command, env=env, start_new_session=True)
    try:
        code = child.wait(timeout=args.timeout)
    except (subprocess.TimeoutExpired, KeyboardInterrupt):
        os.killpg(child.pid, signal.SIGTERM)
        try:
            child.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait()
        write_json(result / 'launcher.json', dict(status='interrupted_or_timeout', timeout=args.timeout,
            child_returncode=child.returncode, elapsed_seconds=time.perf_counter() - started))
        raise
    write_json(result / 'launcher.json', dict(status='complete' if code == 0 else 'failed',
        child_returncode=code, elapsed_seconds=time.perf_counter() - started))
    raise SystemExit(code)


def material_cases(SP, RC, np, size):
    """Diagnostic fixtures, not film look proposals; unchanged for both source variants."""
    def scene(kind, depth):
        sc = SP.Scene()
        if kind == 'cloth':
            sc.begin(rgb=(.18, .075, .045))
            sc.bell(np.array([0., .65, depth]), np.array([0., -.6, depth]), .23, .44,
                    np.array([0., 0., 1.]), depth=.055, ell=.25, wind=.025, phase=.7)
            sc.end()
        elif kind == 'iron_wood':
            for x, mat in ((-.32, 1), (.32, 2)):
                sc.begin()
                sc.box(np.array([x, 0., depth]), (.23, .62, .18), yaw=.3, rnd=.04, mat=mat)
                sc.end()
        elif kind in ('horn', 'glass'):
            sc.begin()
            sc.box(np.array([0., 0., depth + .7]), (.65, .85, .1), mat=2)
            sc.end()
            SP.lantern_v3(sc, np.array([0., -.2, depth]), .23, (1., .46, .13), .9, k=1.4)
            if kind == 'glass':
                sc.O[-1][14] = 0.0  # same shell with ordinary glass accumulation
        elif kind == 'small_glass':
            sc.begin(emit=.6, glass=(1., .45, .1))
            sc.cone(np.array([0., -.35, depth]), np.array([0., .35, depth]), .24, mat=3)
            sc.end()
        elif kind == 'stone_ring':
            SP.stone_ring(sc, np.array([0., -.25, depth]), seed=734, n=7, r0=.52)
        elif kind == 'rough_rock':
            sc.begin()
            sc.box(np.array([0., 0., depth]), (.6, .4, .35), yaw=.24, pitch=.15,
                   rnd=.05, mat=6, rough=.065, rough_f=13.)
            sc.end()
            sc.O[-1][14] = -1.0
        return sc.arrays()

    camera = RC.SrcCam(np.zeros(3), 0., size * 1.4, -size / 2, size / 2, -size / 2, size / 2)
    moon = np.array([-.3, .8, -.52, .5, .65, .9, .65])
    moon[:3] /= np.linalg.norm(moon[:3])
    LT = np.array([[1.1, 1.5, 1.5, 1., .55, .18, 8., .4]], np.float64)
    amb = np.array([.12, .15, .2])
    specs = [(kind, 3.8, False, .025) for kind in
             ('cloth', 'iron_wood', 'horn', 'glass', 'small_glass', 'stone_ring', 'rough_rock')]
    specs += [('cloth', 16., False, .025), ('cloth', 3.8, True, .025), ('cloth', 3.8, False, 0.)]
    for kind, distance, blocked, fog in specs:
        P, O = scene(kind, distance)
        img = np.full((size, size, 3), (.021, .026, .035), np.float32)
        zb = np.full((size, size), 1e9, np.float32)
        if blocked:
            zb[:, :size // 2] = 1.0
        fogp = np.array([fog, .002, 0., 0., 0., .09, .12, .18])
        name = f'{kind}-z{distance:g}-block{int(blocked)}-fog{fog:g}'
        yield name, (img, zb, camera.params(), P, O, LT, moon, amb, fogp, float(camera.pos[1]))
        if kind == 'glass':
            unlit = O.copy()
            unlit[:, 9] = 0.0
            yield 'glass-no-emission', (img, zb, camera.params(), P, unlit, LT, moon, amb, fogp, 0.)
        if kind == 'rough_rock':
            smooth = P.copy()
            smooth[:, 14:16] = 0.0  # same object bounds, only displacement disabled
            yield 'rough_rock-no-displacement', (img, zb, camera.params(), smooth, O, LT, moon, amb, fogp, 0.)


def worker(args):
    import importlib.metadata
    import resource
    import traceback
    import numpy as np
    import cv2
    import numba

    root = args.out.resolve()
    metadata = json.loads((root / 'sources.json').read_text())
    tree = root / 'sources' / args.variant
    result = root / 'results' / args.mode / args.variant / args.phase
    expected = metadata['source_sha256'][args.variant]
    for path, digest in expected.items():
        if sha((tree / path).read_bytes()) != digest:
            raise RuntimeError(f'Frozen input changed: {path}')
    sys.path[:0] = [str(tree / 'the-long-dawn/shots/run'), str(tree / 'the-long-dawn/shots/montage')]
    receipt = dict(status='starting', variant=args.variant, phase=args.phase,
        reference_commit=metadata['reference_commit'], source_sha256=expected,
        harness_sha256=sha(Path(__file__).read_bytes()), created_utc=datetime.now(timezone.utc).isoformat(),
        settings=settings(args),
        environment={k: os.environ.get(k) for k in (*OVERRIDES, 'NUMBA_NUM_THREADS', 'NUMBA_CACHE_DIR',
            'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS')},
        versions={k: importlib.metadata.version(k) for k in ('numpy', 'numba', 'llvmlite')},
        python=sys.version, platform=sys.platform, sp_calls=[], frames=[], outputs={})
    first = time.perf_counter()
    def save():
        write_json(result / 'receipt.json', receipt)
    def info(a):
        a = np.asarray(a)
        return dict(shape=list(a.shape), dtype=str(a.dtype), sha256=sha(np.ascontiguousarray(a).tobytes()),
                    finite=bool(np.isfinite(a).all()))
    def output(name, a):
        if not np.isfinite(a).all():
            raise RuntimeError(f'Nonfinite output: {name}')
        path = result / f'{name}.npy'
        np.save(path, a, allow_pickle=False)
        receipt['outputs'][name] = dict(file=path.name, file_sha256=sha(path.read_bytes()), **info(a))
    save()
    try:
        import sdfppl as SP
        import rcam as RC
        from mt import noise
        if args.mode != 'material':
            import watchers_a as WA
            BR = WA.br()
        cv2.setNumThreads(0)
        numba.set_num_threads(1)
        if numba.config.DISABLE_JIT:
            raise RuntimeError('This benchmark requires compiled renderers.')
        original_render = SP.render
        assert not original_render.signatures, 'SP render was primed before timing'
        if args.variant == 'reference' and SP.gnoise3 is not noise.gnoise3:
            raise RuntimeError('Reference unexpectedly uses a local noise wrapper.')
        if args.variant == 'candidate' and SP.gnoise3 is noise.gnoise3:
            raise RuntimeError('Candidate local noise wrapper is absent.')
        receipt.update(import_seconds=time.perf_counter() - first, numba_threads=numba.get_num_threads(),
            opencv_requested_threads=0, opencv_reported_threads=cv2.getNumThreads(),
            opencv_version=cv2.__version__, jit_disabled=False,
            render_options=original_render.targetoptions, noise_options=SP.gnoise3.targetoptions,
            noise_is_shared=SP.gnoise3 is noise.gnoise3,
            noise_signatures_before=[str(s) for s in SP.gnoise3.signatures],
            cache_files_before=[str(p.relative_to(Path(os.environ['NUMBA_CACHE_DIR'])))
                                for p in Path(os.environ['NUMBA_CACHE_DIR']).rglob('*') if p.is_file()])
        state = {'mode': 'material', 'name': '', 'repeat': 0, 'calls': 0}
        def timed_render(*values):
            before_depth = values[1].copy()
            call = dict(mode=state['mode'], name=state['name'], repeat=state['repeat'], call=state['calls'],
                input_hashes=[info(v) for v in values],
                prior_signatures=[str(s) for s in original_render.signatures])
            state['calls'] += 1
            receipt['sp_calls'].append(call)
            save()
            print(f'SP enter {call["name"]} repeat={call["repeat"]} signatures={len(original_render.signatures)}', flush=True)
            started = time.perf_counter()
            returned = original_render(*values)
            kernel_seconds = time.perf_counter() - started
            depth_changed = values[1] != before_depth
            call.update(seconds=kernel_seconds,
                signatures=[str(s) for s in original_render.signatures],
                img=info(values[0]), zb=info(values[1]),
                depth_changed_mask=info(depth_changed), depth_changed_pixels=int(depth_changed.sum()),
                cache_hits={str(k): int(v) for k, v in original_render.stats.cache_hits.items()},
                cache_misses={str(k): int(v) for k, v in original_render.stats.cache_misses.items()})
            call['new_signatures'] = [s for s in call['signatures'] if s not in call['prior_signatures']]
            if state['repeat'] == 0:
                stem = f'{state["name"]}-sp{call["call"]}'
                output(stem + '-img', values[0])
                output(stem + '-zb', values[1])
                output(stem + '-depth-changed-mask', depth_changed)
            save()
            print(f'SP exit {call["name"]}: {call["seconds"]:.6f}s', flush=True)
            return returned
        SP.render = timed_render
        if args.mode in ('material', 'suite'):
            cases = list(material_cases(SP, RC, np, args.size))
            first_outputs = {}
            for name, values in cases:
                for repeat in range(args.repeats + 1):
                    state.update(name=name, repeat=repeat, calls=0)
                    fresh = tuple(v.copy() if isinstance(v, np.ndarray) else v for v in values)
                    timed_render(*fresh)
                    changed = np.any(fresh[0] != values[0], axis=2)
                    receipt['sp_calls'][-1].update(changed_pixels=int(changed.sum()),
                        depth_hit_pixels=int(np.count_nonzero(fresh[1] != values[1])))
                    if not changed.any():
                        raise RuntimeError(f'Empty material probe: {name}')
                    if repeat == 0:
                        first_outputs[name] = fresh[:2]
            def case(name):
                return first_outputs[name]
            near = case('cloth-z3.8-block0-fog0.025')
            blocked = case('cloth-z3.8-block1-fog0.025')
            inputs = dict(cases)['cloth-z3.8-block1-fog0.025']
            half = args.size // 2
            live = dict(
                horn_changes_glass_rgb=bool(np.any(case('horn-z3.8-block0-fog0.025')[0] != case('glass-z3.8-block0-fog0.025')[0])),
                ordinary_glass_emits=bool(np.any(case('glass-z3.8-block0-fog0.025')[0] != case('glass-no-emission')[0])),
                displacement_changes_depth=bool(np.any(case('rough_rock-z3.8-block0-fog0.025')[1] != case('rough_rock-no-displacement')[1])),
                fog_changes_cloth_rgb=bool(np.any(near[0] != case('cloth-z3.8-block0-fog0')[0])),
                blocker_preserves_foreground=bool(np.array_equal(blocked[0][:, :half], inputs[0][:, :half])
                    and np.array_equal(blocked[1][:, :half], inputs[1][:, :half])),
                visible_half_matches_unblocked=bool(np.array_equal(blocked[0][:, half:], near[0][:, half:])))
            metal_wood = case('iron_wood-z3.8-block0-fog0.025')[1] < 1e9
            # Both box X envelopes stay on their respective side of zero.
            live.update(iron_has_hits=bool(metal_wood[:, :half].any()),
                        wood_has_hits=bool(metal_wood[:, half:].any()))
            receipt['material_liveness'] = live
            if not all(live.values()):
                raise RuntimeError(f'Material coverage control failed: {live}')
            # Gamma display is only a preview; all metrics use the unchanged float arrays.
            tile = args.size + 28
            preview = np.zeros((((len(cases)+3)//4)*tile, 4*args.size, 3), np.uint8)
            for index, (name, _) in enumerate(cases):
                y, x = (index//4)*tile, (index%4)*args.size
                rgb = case(name)[0]
                preview[y+28:y+tile, x:x+args.size] = np.rint(np.clip(rgb, 0., 1.)**(1/2.2)*255).astype(np.uint8)
                cv2.putText(preview, name.split('-')[0], (x+2, y+12), cv2.FONT_HERSHEY_PLAIN, .65, (255,255,255), 1)
                cv2.putText(preview, name.split('-',1)[1], (x+2, y+23), cv2.FONT_HERSHEY_PLAIN, .55, (255,255,255), 1)
            if not cv2.imwrite(str(result/'material-preview.png'), preview[..., ::-1]):
                raise OSError('Could not write material preview.')
            receipt['preview_transfer'] = 'clip linear RGB to [0,1], power 1/2.2, round to uint8; display only'
        if args.mode != 'material':
            for request in receipt['settings']['frame_requests']:
                frame, shot = request['frame'], request['mode']
                name = f'{shot}-{frame}'
                for repeat in range(args.repeats + 1):
                    state.update(mode=shot, name=name, repeat=repeat, calls=0)
                    started = time.perf_counter()
                    if shot == 'A14':
                        linear, _cam, _dist = BR.render(frame, scale=args.scale, ss=args.ss, mblur=True)
                        finished = WA.PI.look.finish(linear, **BR.FINISH)
                    else:
                        linear = WA.render(frame, scale=args.scale, ss=args.ss)
                        finished = WA.finish(linear)
                    item = dict(mode=shot, frame=frame, repeat=repeat, seconds_including_instrumentation=time.perf_counter()-started,
                        linear=info(linear), finished=info(finished), sp_calls=state['calls'], mblur=shot == 'A14')
                    receipt['frames'].append(item)
                    if not state['calls']:
                        raise RuntimeError('Actual frame did not invoke SP.render')
                    if repeat == 0:
                        output(name + '-linear', linear)
                        output(name + '-finished', finished)
        receipt.update(status='complete', total_seconds=time.perf_counter() - first,
            max_rss_native=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            max_rss_units='bytes' if sys.platform == 'darwin' else 'KiB',
            noise_signatures_after=[str(s) for s in SP.gnoise3.signatures],
            render_signatures=[str(s) for s in original_render.signatures],
            signature_changes=[dict(mode=c['mode'], name=c['name'], repeat=c['repeat'],
                                    new_signatures=c['new_signatures']) for c in receipt['sp_calls'] if c['new_signatures']],
            limits='Finite sampled material fixtures/frames only. Cold SP call includes compilation and rendering; '
                   'later same-signature calls are warm. Full-frame timing includes other kernels and capture overhead. '
                   'Cache-hit phase is a fresh process and must show dispatcher cache hits; no CPU pinning or load isolation.')
        initial_call = receipt['sp_calls'][0]
        if args.phase == 'cache-hit' and not sum(initial_call['cache_hits'].values()):
            raise RuntimeError('Requested cache-hit run compiled instead of loading SP.render cache.')
        if args.phase == 'cold' and sum(initial_call['cache_hits'].values()):
            raise RuntimeError('Cold run unexpectedly loaded an SP.render cache.')
        initial = {}
        for call in receipt['sp_calls']:
            key = (call['name'], call['call'])
            if call['repeat'] == 0:
                initial[key] = call
            else:
                for field in ('input_hashes', 'img', 'zb', 'depth_changed_mask'):
                    if call[field] != initial[key][field]:
                        raise RuntimeError(f'Warm repeat changed {field}: {key}')
        receipt['warm_repeats_exact'] = True
        assert_warm_frames(receipt['frames'])
        receipt['full_frame_warm_repeats_exact'] = True if receipt['frames'] else None
        if args.phase == 'cache-hit':
            cold_dir = result.parent / 'cold'
            cold = json.loads((cold_dir / 'receipt.json').read_text())
            for item in cold['outputs'].values():
                assert sha((cold_dir / item['file']).read_bytes()) == item['file_sha256']
            assert_cache_reload(cold, receipt)
            receipt['cache_reload_exact'] = True
    except BaseException:
        receipt.update(status='failed', traceback=traceback.format_exc())
        raise
    finally:
        save()


def compare(args):
    import numpy as np
    a, b = [json.loads((p / 'receipt.json').read_text()) for p in (args.reference, args.candidate)]
    assert a['status'] == b['status'] == 'complete'
    assert a['settings'] == b['settings'] and a['versions'] == b['versions']
    assert a['phase'] == b['phase']
    assert a['python'] == b['python'] and a['platform'] == b['platform']
    assert a['opencv_version'] == b['opencv_version']
    assert a['warm_repeats_exact'] and b['warm_repeats_exact']
    assert a['harness_sha256'] == b['harness_sha256']
    assert a['numba_threads'] == b['numba_threads'] == 1
    assert a['opencv_reported_threads'] == b['opencv_reported_threads']
    assert a['reference_commit'] == b['reference_commit']
    assert set(a['source_sha256']) == set(b['source_sha256'])
    differing_sources = [k for k in a['source_sha256'] if a['source_sha256'][k] != b['source_sha256'].get(k)]
    assert differing_sources == [SP_PATH], differing_sources
    assert len(a['sp_calls']) == len(b['sp_calls']) > 0
    assert list(a['outputs']) == list(b['outputs'])
    calls = []
    for x, y in zip(a['sp_calls'], b['sp_calls']):
        assert (x['mode'], x['name'], x['repeat'], x['call']) == (y['mode'], y['name'], y['repeat'], y['call'])
        assert x['input_hashes'] == y['input_hashes'], f'Inputs differ: {x["name"]}'
        calls.append(dict(mode=x['mode'], name=x['name'], repeat=x['repeat'], reference_seconds=x['seconds'],
            candidate_seconds=y['seconds'], img_exact=x['img']==y['img'], zb_exact=x['zb']==y['zb'],
            depth_changed_masks_exact=x['depth_changed_mask']==y['depth_changed_mask'],
            reference_depth_changed_pixels=x['depth_changed_pixels'],
            candidate_depth_changed_pixels=y['depth_changed_pixels']))
    outputs = []
    for name in a['outputs']:
        arrays = []
        for directory, record in ((args.reference, a['outputs'][name]), (args.candidate, b['outputs'][name])):
            path = directory / record['file']
            assert sha(path.read_bytes()) == record['file_sha256']
            data = np.load(path, allow_pickle=False)
            assert np.isfinite(data).all()
            arrays.append(data)
        x, y = arrays
        assert x.shape == y.shape and x.dtype == y.dtype
        diff = np.abs(x.astype(np.float64)-y.astype(np.float64))
        outputs.append(dict(name=name, exact=np.array_equal(x,y), bytes_exact=x.tobytes()==y.tobytes(),
            changed_values=int(np.count_nonzero(x != y)), max_abs_difference=float(diff.max()) if diff.size else 0.))
        if name.endswith('-depth-changed-mask'):
            outputs[-1]['mask_disagreement_pixels'] = int(np.count_nonzero(x != y))
    report = dict(reference=str(args.reference), candidate=str(args.candidate), calls=calls, outputs=outputs,
                  all_exact=all(o['bytes_exact'] for o in outputs) and all(c['img_exact'] and c['zb_exact']
                    and c['depth_changed_masks_exact'] for c in calls))
    write_json(args.output, report)
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report['all_exact'] else 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest='action', required=True)
    p = subs.add_parser('prepare'); p.add_argument('--repo', type=Path, required=True); p.add_argument('--out', type=Path, required=True)
    for name in ('run', '_worker'):
        p = subs.add_parser(name)
        p.add_argument('--out', type=Path, required=True)
        p.add_argument('--variant', choices=('reference', 'candidate'), required=True)
        p.add_argument('--mode', choices=('material', 'A14', 'A15', 'suite'), required=True)
        p.add_argument('--phase', choices=('cold', 'cache-hit'), default='cold')
        p.add_argument('--frames')
        p.add_argument('--scale', type=float, default=.5)
        p.add_argument('--ss', type=float, default=1.5)
        p.add_argument('--size', type=int, default=96)
        p.add_argument('--repeats', type=int, default=3)
        p.add_argument('--timeout', type=float, default=1200.)
    p = subs.add_parser('compare')
    p.add_argument('--reference', type=Path, required=True); p.add_argument('--candidate', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if hasattr(args, 'frames'):
        if args.mode == 'suite' and args.frames is None:
            parser.error('Suite mode requires an explicit --frames list.')
        args.frames = args.frames or '4240'
        try:
            frame_requests(args.mode, args.frames)
        except ValueError as exc:
            parser.error(str(exc))
    if hasattr(args, 'repeats') and (args.repeats < 1 or args.size < 16 or args.scale <= 0 or args.ss <= 0):
        parser.error('Use repeats>=1, size>=16 and positive scale/ss.')
    {'prepare': prepare, 'run': run, '_worker': worker, 'compare': compare}[args.action](args)


if __name__ == '__main__':
    main()
