#!/usr/bin/env python3
"""Farm adapter for ADOPTION.md's ld_candidate; native PNGs, unchanged render APIs.

Each process owns one renderer family. The farm may split --range into multiple
processes; page RGB and matte are emitted by the same process. Setup uses
--prepare followed by --equal FRAME --scale 1. No farm API is called here.
"""
import argparse
import gc
import hashlib
import importlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import resource
import sys
import tempfile
import time

# Set before importing numpy/numba/OpenCV, including when used as a library.
for _name in ('NUMBA_NUM_THREADS', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
              'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_name] = '1'

ROOT = Path(__file__).resolve().parents[1]
RANGES = {'reveal': (2880, 3119), 'watch': (4480, 4719), 'scroll': (0, 319),
          'beaconrun': (3920, 4239), 'watchers': (4240, 4399),
          'trap': (2320, 2639), 'map': (3440, 3839),
          'deep': (4240, 4479), 'cold': (3816, 3999), 'unfinished': (4000, 4239),
          'pen': (5440, 5679), 't1': (320, 559), 'crossing': (4880, 5839),
          'falsedawn': (80, 559), 'sweep': (1680, 1687)}
OPTIONS = {'reveal': 'night-fire', 'watch': 'night-fire', 'scroll': 'night-fire',
           'beaconrun': 'linked-fires', 'watchers': 'linked-fires',
           'trap': 'front_smoke_near', 'map': 'beacon-falloff',
           'deep': 'leaned_ladders', 'cold': 'lead24', 'unfinished': 'in-step',
           'pen': 'soft_spine_metal', 't1': 'current-words', 'crossing': 'both',
           'falsedawn': 'clear_high_deck', 'sweep': 'soft-entry'}
# Explicit review alternatives leave OPTIONS and omitted renderer defaults unchanged.
MORE = {'crossing': ('both_decal', 'both_decal_cap'), 't1': ('current-words-held',),
        'cold': ('in-step',)}
PAGES = ('deep', 'pen', 't1', 'sweep')
_FAMILY = None


def imports(kind):
    global _FAMILY
    family = ('run' if kind in ('reveal', 'watch', 'scroll', 'crossing', 'beaconrun', 'watchers', 'falsedawn') else
              'embers' if kind in ('trap', 'cold', 'unfinished') else 'map')
    if _FAMILY is not None and _FAMILY != family:
        raise RuntimeError('Use a fresh process for each renderer family')
    _FAMILY = family
    sys.path.insert(0, str(ROOT / 'shots' / family))
    sys.path.append(str(ROOT / 'lib'))


def threads():
    import cv2
    import numba
    cv2.setNumThreads(0)
    numba.set_num_threads(1)
    if cv2.getNumThreads() != 1 or numba.get_num_threads() != 1:
        raise RuntimeError('Renderer thread limit changed')


def checked(render, frame, scale=1.0):
    import numpy as np
    import cv2
    import numba
    threads()
    result = render(frame)
    if cv2.getNumThreads() != 1 or numba.get_num_threads() != 1:
        raise RuntimeError('Renderer changed its thread limit')
    if result['rgb'].shape != (int(804 * scale), int(1920 * scale), 3):
        raise ValueError('Unexpected RGB dimensions')
    if not all(np.isfinite(a).all() for a in result.values()):
        raise ValueError('Non-finite renderer output')
    return result


def book_output(module, result):
    hdr, alpha = result
    rgb = module.look.finish(hdr, exposure=1.15, bloom_strength=.06,
                             bloom_threshold=1.2, vignette_amount=.32)
    return dict(hdr=hdr, alpha=alpha, rgb=rgb)


def build(kind, route, option=None, scale=1.0):
    if kind not in RANGES or route not in ('original', 'default', 'shared', 'candidate'):
        raise ValueError('Unknown kind or route')
    if not math.isfinite(scale) or not 0 < scale <= 1 or int(804 * scale) < 1:
        raise ValueError('Scale must produce positive dimensions, at most native')
    if route == 'candidate' and option not in (OPTIONS[kind],) + MORE.get(kind, ()):
        raise ValueError('Unsupported candidate')
    imports(kind)
    W, H = int(1920 * scale), int(804 * scale)
    page_kind = kind in PAGES
    # route distinguishes direct original, omitted default, shared accepted and candidate.
    if kind in ('beaconrun', 'watchers'):
        # Import both renderers HERE, before checked() resets the pools: lib/look.py sets cv2.setNumThreads(2) when it
        # is imported, and beaconrun_candidates imports these modules lazily inside its first render, so the first
        # frame changed the limit after the reset and checked() refused it (farm 29 Sep 20:52, every unit).
        import beaconrun_a  # noqa: F401
        import watchers_a  # noqa: F401
        if route == 'original':
            if kind == 'beaconrun':
                import beaconrun_a as BR
                call = lambda f: BR.PI.look.finish(BR.render(f, scale=scale, ss=1.5)[0], **BR.FINISH)
            else:
                import watchers_a as WA
                call = lambda f: WA.finish(WA.render(f, scale=scale, ss=1.5))
        else:
            import beaconrun_candidates as X
            if route == 'default':
                call = lambda f: X.render(f, kind=kind, scale=scale, ss=1.5)
            else:
                choice = 'accepted' if route == 'shared' else option
                call = lambda f: X.render(f, kind=kind, candidate=choice, scale=scale, ss=1.5)
        return lambda f: dict(rgb=call(f))
    if kind == 'falsedawn':
        import falsedawn as FD

        def call(f):
            kwargs = dict(design='arc', scale=scale, ss=1.5)
            if route == 'candidate':
                kwargs['sky_candidate'] = option
            hdr = FD.render(f - FD.CUT0, **kwargs)
            return dict(hdr=hdr, rgb=FD.PI.look.finish(hdr, **FD.FINISH))

        return call
    if kind in ('reveal', 'watch', 'scroll'):
        import beacon_night_candidates as X
        driver = getattr(X, kind)
        if kind == 'scroll':
            import render_ink as RI
            # The accepted scroll uses local 0-319 (C5 3120-3439), including
            # its seven timed catches and the already-lit horizon beacon.
            shot = RI.Shot('scroll')
        else:
            shot = driver.make_shot()
        if route == 'original':
            call = lambda f: driver.render(f, shot, scale=scale, ss=2.)
        elif route == 'shared':
            call = lambda f: X.render_variants(f, shot, kind=kind, scale=scale,
                                               ss=2., variants=('accepted',))['accepted']
        elif route == 'default':
            call = lambda f: X.render(f, shot, kind=kind, scale=scale, ss=2.)
        else:
            call = lambda f: X.render(f, shot, kind=kind, variant=option, scale=scale, ss=2.)
        return lambda f: dict(rgb=call(f))
    if kind in ('cold', 'unfinished'):
        import c_v5_cold_leadin as lead
        import c_v5_instep as X
        if route == 'original':
            # The accepted edit includes the lead24 extension at C3816..3839.
            scene = lead.Scene('lead24') if kind == 'cold' else X.base.Scene()
        elif route == 'default':
            scene = X.Scene()
        elif route == 'shared':
            scene = X.Scene(variant='accepted')
        elif option == 'lead24':
            scene = lead.Scene(variant='lead24')
        else:
            scene = X.Scene(variant=option)
        return lambda f: dict(rgb=scene.frame(f, scale=scale))
    if kind == 'trap':
        X = importlib.import_module('c_v5_trap_candidates')
        base = X.C
        scene = base.Scene() if route == 'original' else (X.Scene() if route == 'default' else X.Scene(variant=option))
        return lambda f: dict(rgb=scene.frame(f, scale=scale))
    if kind == 'map':
        import last_beacon_candidates as X
        base = X.BASE
        manifest = base.source_manifest()
        identity = dict(world=manifest['world'], sources=manifest['sources'])
        key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16]
        world = base.CACHE.parent / 'world' / key / f"terra_{manifest['world']}.npz"
        if not world.is_file():
            raise RuntimeError('Existing source-pinned Map world is required; no bake')
        base.validate_atlas(base.atlas_dir(), manifest)
        shot = base.LastBeacon() if route == 'original' else X.make_shot()
        if route == 'original':
            call = lambda f: shot.render_c(f, scale)
        elif route == 'shared':
            call = lambda f: X.render_variants(f, shot, scale=scale, variants=('accepted',))['accepted']
        elif route == 'default':
            call = lambda f: X.render(f, shot, scale=scale)
        else:
            call = lambda f: X.render(f, shot, candidate=option, scale=scale)
        return lambda f: dict(rgb=call(f))
    if kind == 'sweep':
        import sweep_entry_candidates as X
        # C 1680-1687 keeps the filmed scorch and baked race. These inputs are
        # already display RGB; another book finish would grade them twice.
        if route == 'original':
            return lambda f: X.render_original(f, scale=scale)
        if route == 'default':
            return X.make_renderer(scale=scale)
        return X.make_renderer('accepted' if route == 'shared' else option, scale=scale)
    if page_kind:
        name = {'deep': 'book_c_v5_deep_candidates', 'pen': 'book_c_v5_pen_candidates',
                't1': 'book_c_t1_candidates'}[kind]
        X = importlib.import_module(name)
        if kind == 't1':
            BC = X.BC
            scene = (BC.Book3(W, H, with_fire=False) if route == 'original' else
                     X.make_renderer(scale=scale, with_fire=False) if route == 'default' else
                     X.make_renderer(option, scale=scale, with_fire=False))
            scene.no_burn = True
        else:
            BC = X.V.BC
            shot = 'deep_abandoned' if kind == 'deep' else 'pen'
            scene = (X.V.PagesV5(shot, W, H, 110) if route == 'original' else
                     X.make_renderer(scale=scale, ppc=110) if route == 'default' else
                     X.make_renderer(option, scale=scale, ppc=110))
        return lambda f: book_output(BC, scene.frame(f))
    import crossing_candidates as X
    import crossing_rock_candidates as rock
    cr = X.load_renderer()     # rejects a missing fire cache or changed baseline
    def call(f):
        if route == 'original':
            hdr = cr.render(f - 4880, scale=scale, ss=1.5, variant='main', trail=True)
        elif route == 'default':
            hdr = X.render_cut(f, renderer=cr, scale=scale, ss=1.5, variant='main', trail=True)
        else:
            modifier = (lambda scene, renderer, cfg: rock.apply_to_scene(
                scene, renderer, cfg, candidate='low_shoulders')) if option in ('rock', 'both', 'both_decal', 'both_decal_cap') else None
            # Only the cap candidate changes terrain; preserve every existing
            # route's keyword arguments for the accepted/default equality gate.
            terrain = {'terrain': 'round_cap'} if option == 'both_decal_cap' else {}
            hdr = X.render_cut(f, renderer=cr, scale=scale, ss=1.5, variant='main', trail=True,
                               rope=('snow_decal' if option in ('both_decal', 'both_decal_cap') else
                                     'snow_clearance' if option in ('rope', 'both') else 'accepted'),
                               rock='low_shoulders' if modifier else 'accepted', rock_modifier=modifier,
                               **terrain)
        return dict(hdr=hdr, rgb=cr.PI.look.finish(hdr, **cr.FINISH))
    return call


def digest(a):
    return dict(dtype=a.dtype.str, shape=list(a.shape), sha256=hashlib.sha256(a.tobytes()).hexdigest())


def equal(kind, frame, scale=1.0, candidate=None):
    """ADOPTION's original/default/shared proof, including HDR and page alpha.

    Explicit exceptions keep the gate active even under python -O. This proves
    API equality on this machine, not equality to archived delivery JPEGs.
    """
    import numpy as np
    validate_frames(kind, frame, frame)
    if candidate is not None and (kind != 'cold' or candidate != 'in-step' or frame >= 3848):
        raise ValueError('Candidate equality requires cold in-step before C3848')
    original = build(kind, 'original', scale=scale)
    expected = checked(original, frame, scale)
    del original
    gc.collect()
    proofs = {}
    routes = ('default', 'shared') if kind in ('reveal', 'watch', 'scroll', 'map', 'beaconrun', 'watchers', 'sweep', 'cold', 'unfinished') else ('default',)
    if candidate is not None:
        routes += ('candidate',)
    for route in routes:
        render = (build(kind, route, candidate, scale=scale) if route == 'candidate' else
                  build(kind, route, scale=scale))
        actual = checked(render, frame, scale)
        if actual.keys() != expected.keys():
            raise RuntimeError(f'Equality failed: {kind} {route} array keys')
        for key in expected:
            if actual[key].dtype != expected[key].dtype or not np.array_equal(actual[key], expected[key]):
                raise RuntimeError(f'Equality failed: {kind} {route} {key} frame {frame}')
        proofs[route] = {key: digest(a) for key, a in actual.items()}
        del render, actual
        gc.collect()
    receipt = dict(kind=kind, frame=frame, scale=scale, exact_equal=True, proofs=proofs)
    if candidate is not None:
        receipt['candidate'] = candidate
    print(json.dumps(receipt), flush=True)
    return receipt


def validate_frames(kind, first, last):
    if kind not in RANGES or not RANGES[kind][0] <= first <= last <= RANGES[kind][1]:
        raise ValueError('Frame range is outside this shot')


def writable_renders():
    """Refuse the owner symlink before any module with import-time cache writes."""
    directory = ROOT / 'renders'
    if directory.is_symlink() or directory.resolve().parent != ROOT.resolve():
        raise ValueError('Refusing a linked renders directory; use a farm checkout or isolated test copy')
    return directory


def output_dirs(kind, option, out):
    directory = writable_renders()
    out = Path(out)
    if not out.is_absolute():
        out = ROOT / out
    expected = f'cand_{kind}_{option}'
    if os.environ.get('LD_OPEN_RING') == '1':
        # The open band must get fresh frames; keep the accepted candidates intact.
        expected = {('trap', 'front_smoke_near'): 'embers_C5_trap_open',
                    ('cold', 'in-step'): 'cand_cold_instep_open',
                    ('unfinished', 'in-step'): 'cand_unfinished_instep_open'}.get((kind, option), expected)
    if out.parent.resolve() != directory.resolve() or out.name != expected:
        if os.environ.get('LD_OPEN_RING') == '1':
            raise ValueError(f'Use renders/{expected} for this candidate')
        raise ValueError('Use renders/cand_<kind>_<option> for this candidate')
    directories = [out] + ([out.with_name(out.name + '_matte')] if kind in PAGES else [])
    if any(p.is_symlink() or (p.exists() and not p.is_dir()) for p in directories):
        raise ValueError('Output must be a real candidate directory')
    return directories


def save_png(path, array):
    """Same pre-JPEG quantization as ld_candidate; publish only complete PNGs."""
    import numpy as np
    from PIL import Image
    path = Path(path)
    pixels = np.rint(np.clip(array, 0, 1) * 255).astype(np.uint8)
    # Farm creates/reuses directories and removes requested stale frames. Each
    # lane owns disjoint frame numbers; refuse an unexpected existing frame.
    with tempfile.NamedTemporaryFile(prefix='.' + path.name + '.', suffix='.tmp',
                                     dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            Image.fromarray(pixels).save(stream, format='PNG')
            stream.flush()
            os.fsync(stream.fileno())
            os.link(temporary, path)  # atomic publication, no overwrite
        finally:
            temporary.unlink(missing_ok=True)


STACK = {'numpy': '2.5.3', 'numba': '0.67.0', 'llvmlite': '0.49.0',
         'scipy': '1.18.1', 'opencv-python-headless': '4.10.0.84',
         'pillow': '12.3.0', 'fonttools': '4.66.0'}
ASSETS = {
    'reveal': ('shots/run/summits.npy',),
    'watch': ('shots/run/summits.npy',),
    'scroll': ('shots/run/summits.npy',),
    'beaconrun': ('shots/run/summits.npy', 'shots/run/beaconrun_a_chain.npy',
                  'shots/run/reveal_a_fires.npy', 'shots/run/watchers_a_extra.npy',
                  'shots/run/watchers_a_figs.npy'),
    'watchers': ('shots/run/summits.npy', 'shots/run/beaconrun_a_chain.npy',
                 'shots/run/reveal_a_fires.npy', 'shots/run/watchers_a_extra.npy',
                 'shots/run/watchers_a_figs.npy'),
    'trap': ('assets/ring/inscription_outer.png', 'assets/ring/inscription_inner.png'),
    'cold': ('assets/ring/inscription_outer.png', 'assets/ring/inscription_inner.png',
             'music/v3/barmap_C5P2.json'),
    'unfinished': ('assets/ring/inscription_outer.png', 'assets/ring/inscription_inner.png',
                   'music/v3/barmap_C5P2.json'),
    't1': ('assets/fonts/EBGaramond-Italic.ttf',),
    'crossing': ('shots/run/crossing.py', 'shots/run/crossing_fires.npy'),
}


def prepare(kind):
    """Farm-only setup: verify stack/assets; rebuild Map caches from source.

    Other procedural caches are built by the following equality command. No
    downloader, substitute font, or unpinned dependency fallback is used.
    """
    writable_renders()
    versions = {name: importlib.metadata.version(name) for name in STACK}
    if versions != STACK or sys.version_info[:2] != (3, 12):
        raise RuntimeError(f'Farm stack mismatch: Python {sys.version}; packages {versions}; expected {STACK}')
    assets = {}
    for name in ASSETS.get(kind, ()):
        path = ROOT / name
        if not path.is_file():
            raise RuntimeError(f'Required tracked input is missing: {name}')
        assets[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    receipt = dict(kind=kind, python=sys.version, packages=versions, assets=assets)
    if kind == 'sweep':
        imports(kind)
        import sweep_entry_candidates as X
        # The farm checkout has no local delivered renders. Require the pinned
        # input bundle before equality or any candidate frame can be produced.
        receipt['inputs'] = X.validate_inputs()
    print(json.dumps(receipt), flush=True)
    if kind == 'crossing':
        # Check without importing the large renderer; load_renderer repeats it.
        text = (ROOT / 'shots/run/crossing_candidates.py').read_text()
        expected = re.search(r"BASELINE_SHA256\s*=\s*['\"]([a-f0-9]{64})['\"]", text)
        if not expected or assets['shots/run/crossing.py'] != expected.group(1):
            raise RuntimeError('Crossing baseline source hash mismatch')
    if kind == 'map':
        imports(kind)
        import last_beacon as base
        threads()
        base.build_atlas()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('kind', choices=RANGES)
    parser.add_argument('option', nargs='?')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--prepare', action='store_true')
    mode.add_argument('--equal', type=int, metavar='FRAME')
    mode.add_argument('--range', dest='frame_range', metavar='A-B')
    parser.add_argument('--out', type=Path)
    parser.add_argument('--scale', type=float, default=1.0, help='equality only; farm uses 1')
    parser.add_argument('--candidate', help='also prove this candidate is unchanged at --equal FRAME')
    args = parser.parse_args(argv)
    if args.prepare:
        if args.option or args.out or args.scale != 1.0 or args.candidate:
            parser.error('--prepare takes only a kind')
        prepare(args.kind)
    elif args.equal is not None:
        if args.option or args.out or args.scale not in (.5, 1.0):
            parser.error('--equal takes a kind, a frame, and --scale .5 or 1')
        writable_renders()
        if args.candidate:
            equal(args.kind, args.equal, args.scale, candidate=args.candidate)
        else:
            equal(args.kind, args.equal, args.scale)
    else:
        if args.option not in (OPTIONS[args.kind],) + MORE.get(args.kind, ()) or not args.frame_range or not args.out or args.scale != 1.0 or args.candidate:
            parser.error('render requires this kind\'s candidate, --range A-B, --out DIR, native scale')
        match = re.fullmatch(r'(\d+)-(\d+)', args.frame_range)
        if not match:
            parser.error('--range must be an inclusive A-B range')
        first, last = map(int, match.groups())
        validate_frames(args.kind, first, last)
        directories = output_dirs(args.kind, args.option, args.out)
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
        render = build(args.kind, 'candidate', args.option)
        import numpy as np
        for frame in range(first, last + 1):
            started = time.perf_counter()
            result = checked(render, frame)
            filename = f'f_{frame:05d}.png'
            save_png(directories[0] / filename, result['rgb'])
            if args.kind in PAGES:
                if result['alpha'].shape != (804, 1920):
                    raise ValueError('Unexpected matte dimensions')
                matte = np.repeat(np.clip(result['alpha'], 0, 1)[..., None], 3, axis=2)
                save_png(directories[1] / filename, matte)
                del matte
            print(json.dumps(dict(frame=frame, kind=args.kind, option=args.option,
                                  seconds=time.perf_counter() - started)), flush=True)
            del result
            gc.collect()
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print(json.dumps(dict(process_peak_rss_bytes=peak if sys.platform == 'darwin' else peak * 1024)), flush=True)


if __name__ == '__main__':
    main()
