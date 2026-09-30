"""Source-derived ROI guides only; no picture pixels or renderer are evaluated.

The geometry definitions are compiled unchanged from an explicit source
allowlist. Crossing's render(), build_scene(), frame objects and rasterizers
are never imported or compiled. The largest geometry array is the original
path search (1,060 by 113 terrain positions); no image-size arrays are made.
Run through the owner-night onepy wrapper, including this geometry-only probe.
"""
import argparse
import ast
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import sys
from types import SimpleNamespace


FUNCTIONS = set('''smoothstep uv build_arete build_ranges ground_many _build_path
path at s_of_u variant_cfg _hh walk_k lantern_s layout _sched step_event step_h
smin _ease_on line_s stride_of foot_cycle walk_phase plant _rate sky_angle _ss
heart_pos _swing camera _fire_sites _finish_set wfs wf_burn _shoulders bearers'''.split())
CONSTANTS = set('''FPS NFR UP C0 YAW_E _a E3 S3 KNOTS PASS_FRAMES PASS_BEYOND
CR0 CR GR_V GR_U0 GR_KNOTS GR_ROWS GR_CREST _PATH SPEED S_L0 T_GO TAU_GO
N_GROUP_SIZES _LAYOUT STEP_T STEP_H STEP_HOLD _STEP STANCE SKY_DEG T_WHEEL
T_WIDE _CLK LANT_K H0 RING _D0 _BUILDING _WF'''.split())
FORBIDDEN = {'render', 'render_figures', 'build_scene', 'Frame', 'march', 'shade',
             'flame', 'halo', 'warp', 'warp_maps'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def target_names(node):
    if isinstance(node, ast.Name):
        return {node.id}
    if isinstance(node, (ast.Tuple, ast.List)):
        return set().union(*(target_names(e) for e in node.elts))
    if isinstance(node, ast.Subscript):
        return target_names(node.value)
    return set()


def ensure_geometry_only(tree):
    """Reject a rasterizer call before any extracted source executes."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            callee = node.func
            name = callee.id if isinstance(callee, ast.Name) else getattr(callee, 'attr', '')
            if name in FORBIDDEN:
                raise ValueError(f'forbidden non-geometry call: {name}')


def selected_tree(path, functions, constants):
    source = ast.parse(Path(path).read_text(), filename=str(path))
    nodes = []
    found = set()
    for node in source.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in functions:
            nodes.append(node)
            found.add(node.name)
        elif isinstance(node, (ast.Assign, ast.AugAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            names = set().union(*(target_names(t) for t in targets))
            if names and names <= constants:
                nodes.append(node)
                found.update(names)
    missing = (functions | constants) - found
    if missing:
        raise ValueError(f'missing source definitions: {sorted(missing)}')
    tree = ast.Module(body=nodes, type_ignores=[])
    ensure_geometry_only(tree)
    return tree


def self_test():
    ensure_geometry_only(ast.parse('camera(10).project(p)'))
    for bad in ('render(0)', 'build_scene(0, cfg)', 'PI.Frame(cam)', 'WD.march(p)'):
        try:
            ensure_geometry_only(ast.parse(bad))
        except ValueError as exc:
            print(f'negative control rejected {bad}: {exc}')
        else:
            raise AssertionError(f'negative control incorrectly accepted: {bad}')


def beacon_geometry(root):
    """Beacons use only existing geometry; no render function is called."""
    import numpy as np
    import beaconrun_a as BR
    import watchers_a as WA
    import nighta as NA
    # Fail loudly if a future geometry helper starts calling a render routine.
    def forbid_raster_calls(frame, event, arg):
        if event == 'call' and frame.f_code.co_name in FORBIDDEN:
            raise RuntimeError(f'forbidden renderer call: {frame.f_code.co_name}')
    sys.setprofile(forbid_raster_calls)
    frames = []
    try:
        offset7 = WA.seventh_hearth(BR.CHAIN[6]).fire_base_offset
        up = np.array([0., 1., 0.])
        for cut in range(3920, 4240):
            cam = BR.camera(cut, 960, 402)
            row = {'frame': cut, 'chain': []}
            for k, source in enumerate(BR.CHAIN):
                x, y, z, ignition, size, seed = source
                height, radius = NA.fire_dims(size)[:2]
                base = np.array([x, y + (offset7 if k == 6 else 0.), z])
                points = {'base': base, 'center': base + up * height * .5,
                          'top': base + up * height,
                          'catch_center': base + up * height * (.12 if k == 6 else .5)}
                fire = {'id': k + 1, 'source_ignition': int(ignition),
                        'nominal_height_m': float(height),
                        'distance_m': float(np.linalg.norm(base - cam.pos))}
                fire.update({name: [float(v) for v in cam.project(p)] for name, p in points.items()})
                fire['nominal_halfwidth_px'] = float(radius * cam.f / fire['base'][2])
                row['chain'].append(fire)
            frames.append(row)
    finally:
        sys.setprofile(None)
    run = root / 'shots/run'
    return {'kind': 'source-derived ROI aids; not measured pixels or visibility',
            'method': 'Original beacon CHAIN, camera, fire_dims and seventh hearth-seat geometry. No rendering.',
            'resolution': [960, 402], 'frame_count': len(frames),
            'source_sha256': {str(p.relative_to(root)): sha(p) for p in
                             (run / 'beaconrun_a.py', run / 'beaconrun_a_chain.npy',
                              run / 'watchers_a.py', run / 'nighta.py', run / 'hearth_a.py',
                              run / 'rcam.py', run / 'world.py', run / 'falsedawn.py',
                              Path(__file__).resolve())},
            'limitations': ['No occlusion test or picture measurement.',
                            'Nominal flame extent excludes flicker, blur, bloom and grade.'],
            'frames': frames}


def ridge_geometry(root):
    """Extract the reveal's cameras and summit offset, without its renderer."""
    import numpy as np
    import world as WD
    run, hills = root / 'shots/run', root / 'shots/hills'
    shared = {'np': np, 'math': math}
    look_ctx = {}
    exec(compile(selected_tree(root / 'lib/look.py', set(), {'W', 'H'}),
                 str(root / 'lib/look.py'), 'exec'), look_ctx)
    core = dict(shared, look=SimpleNamespace(W=look_ctx['W'], H=look_ctx['H']))
    exec(compile(selected_tree(hills / 'core.py',
                 {'Camera', 'clamp', 'smoothstep', 'fnoise1', '_pn1', '_h1'}, {'FULL_W', 'FULL_H'}),
                 str(hills / 'core.py'), 'exec'), core)
    reveal_source = ast.parse((run / 'reveal_a.py').read_text())
    # This source assertion follows the original timing configuration. Reading
    # its literal avoids importing beacon (and avoids assuming its raw ROAR).
    timing_guard = next(n for n in reveal_source.body if isinstance(n, ast.Assert)
                        and 'BK.ROAR' in ast.unparse(n))
    roar_guard = next(n for n in ast.walk(timing_guard) if isinstance(n, ast.Compare)
                      and ast.unparse(n.left) == 'BK.ROAR')
    if not (len(roar_guard.comparators) == 1 and isinstance(roar_guard.ops[0], ast.Eq)):
        raise ValueError('reveal timing assertion changed')
    roar = ast.literal_eval(roar_guard.comparators[0])
    ctx = dict(shared, Camera=core['Camera'], fnoise1=core['fnoise1'], smoothstep=core['smoothstep'],
               BK=SimpleNamespace(ROAR=roar))
    exec(compile(selected_tree(run / 'reveal_a.py', {'h1_camera', '_s5', 'crane_e', 'camera'},
                 {'CUT0', 'CUT1', 'SRC_OFF', 'TAKE_END', 'SRC1', 'FPS', 'END_POS', 'END_YAW',
                  'END_PITCH', 'END_HFOV', 'EASE_A'}), str(run / 'reveal_a.py'), 'exec'), ctx)
    peaks_source = ast.parse((hills / 'peaks.py').read_text())
    branch = next(n for n in peaks_source.body if isinstance(n, ast.If)
                  and ast.unparse(n.test) == 'S1_WORLD')
    offset_nodes = [n for n in branch.body if isinstance(n, ast.Assign)
                    and set().union(*(target_names(t) for t in n.targets)) <= {'_SUM', 'OFFX', 'OFFY', 'OFFZ'}]
    offset_tree = ast.Module(body=offset_nodes, type_ignores=[])
    ensure_geometry_only(offset_tree)
    offset_ctx = dict(shared, _S1=WD.S1)
    print('ridge: original summit geometry query', flush=True)
    exec(compile(offset_tree, str(hills / 'peaks.py'), 'exec'), offset_ctx)
    off = np.array([offset_ctx[k] for k in ('OFFX', 'OFFY', 'OFFZ')])
    dims = dict(shared)
    exec(compile(selected_tree(run / 'nighta.py', {'fire_dims'}, {'POOL_I'}),
                 str(run / 'nighta.py'), 'exec'), dims)
    fire_path = root / 'shots/montage/mt/fire.py'
    ign = dict(shared)
    exec(compile(selected_tree(fire_path, {'ignite_env'}, set()), str(fire_path), 'exec'), ign)
    fire_table_path = run / 'reveal_a_fires.npy'
    fires = np.load(fire_table_path)
    if fires.ndim != 2 or fires.shape[1] != 6:
        raise ValueError('unexpected source fire table shape')
    rows = []
    for cut in range(3660, 3800):
        if (cut-3660)%80 == 0:
            print('ridge geometry frame',cut,'RSS',resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,flush=True)
        cam, _, _ = ctx['camera'](cut - ctx['SRC_OFF'], .5)
        row = {'frame': cut, 'fires': []}
        for k, p in enumerate(fires):
            x, y, z, ignition, size, seed = p
            height = dims['fire_dims'](size)[0]
            live = ign['ignite_env'](cut / ctx['FPS'], ignition / ctx['FPS'])[0]
            # nighta.fire_layer's original base offset and unresolved hot-point
            # vertical offset; source-space vertical displacement equals world-up.
            base = np.array([x, y + .45 * height, z]) - off
            points = {'base': base, 'hot_point': base + np.array([0., .3 * height * live, 0.]),
                      'top': base + np.array([0., height * live, 0.])}
            pred = {'source_index': k, 'source_seed': int(seed),
                    'source_ignition': float(ignition), 'source_size_multiplier': float(live)}
            pred.update({name: [float(v) for v in cam.project(q)] for name, q in points.items()})
            row['fires'].append(pred)
        rows.append(row)
    source_paths = [run / 'reveal_a.py', fire_table_path, hills / 'core.py', hills / 'peaks.py',
                    root / 'lib/look.py', root / 'shots/montage/s1_peak.py', fire_path,
                    run / 'nighta.py', run / 'world.py', Path(__file__).resolve()]
    return {'kind': 'source-derived ROI aids; not measured pixels or visibility',
            'method': 'AST-selected original reveal/hills cameras; original s1 summit geometry; no reveal/beacon/peaks imports or grids.',
            'resolution': [960, 402], 'frame_count': len(rows), 'source_fire_count': len(fires),
            'source_sha256': {str(p.relative_to(root)): sha(p) for p in source_paths},
            'world_offset': off.tolist(), 'timing_guard_roar': roar,
            'limitations': ['Predicted projections do not establish a catch, occlusion or visibility.',
                            'Source ignition is a model parameter and is not reported as a picture catch.',
                            'No raster sampling, blur or finish included. Pixel-centre coordinates may differ by half a pixel.'],
            'frames': rows}


def write_result(output, destination):
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    output['peak_rss_bytes'] = int(rss if sys.platform == 'darwin' else rss * 1024)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp = destination.with_suffix(destination.suffix + '.tmp')
    temp.write_text(json.dumps(output, indent=2, allow_nan=False) + '\n')
    temp.replace(destination)
    print(json.dumps({'out': str(destination), 'frames': output['frame_count'],
                      'peak_rss_bytes': output['peak_rss_bytes']}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--self-test', action='store_true')
    parser.add_argument('--beacon-only', action='store_true')
    parser.add_argument('--ridge-only', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if args.out is None:
        parser.error('--out is required unless --self-test')
    print('geometry helper acquired slot', 'ridge' if args.ridge_only else 'beacon' if args.beacon_only else 'crossing', flush=True)
    os.environ.setdefault('NUMBA_NUM_THREADS', '1')
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
    root = Path(__file__).resolve().parents[2]
    run = root / 'shots/run'
    sys.path.insert(0, str(run))
    if args.beacon_only:
        write_result(beacon_geometry(root), args.out)
        return
    if args.ridge_only:
        write_result(ridge_geometry(root), args.out)
        return
    import numpy as np
    import world as WD
    import rcam as RC

    src = run / 'crossing.py'
    tree = selected_tree(src, FUNCTIONS, CONSTANTS)
    ctx = {'np': np, 'WD': WD, 'RC': RC, 'math': math}
    exec(compile(tree, str(src), 'exec'), ctx)
    if 'render' in ctx or 'build_scene' in ctx:
        raise AssertionError('renderer unexpectedly present')
    # Extract the exact flame's base and nominal height expressions, without
    # compiling or calling draw_fires. Projected extent is not visible extent.
    source_tree = ast.parse(src.read_text())
    draw = next(n for n in source_tree.body if isinstance(n, ast.FunctionDef) and n.name == 'draw_fires')
    flame = next(n for n in ast.walk(draw) if isinstance(n, ast.Call)
                 and isinstance(n.func, ast.Attribute) and n.func.attr == 'flame')
    expressions = [compile(ast.Expression(flame.args[i]), str(src), 'eval') for i in (3, 4, 5)]
    print('crossing: original bounded path and watch-fire geometry query',flush=True)
    fires = ctx['wfs']()
    fixed_heart = ctx['heart_pos']((5180 - 4880) / ctx['FPS']).copy()
    def project(cam, point):
        x, y, z = cam.project(point)
        return [float(x), float(y), float(z)]
    frames = []
    for cut in range(4880, 5840):
        if (cut-4880)%80 == 0:
            print('crossing geometry frame',cut,'RSS',resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,flush=True)
        local = cut - 4880
        t = local / ctx['FPS']
        cam = ctx['camera'](local, 960, 402)
        row = {'frame': cut, 'heart_ideal': project(cam, ctx['heart_pos'](t)),
               'heart_fixed_at_5180': project(cam, fixed_heart), 'watchfires': []}
        for k, p in enumerate(fires):
            b, _ = ctx['wf_burn'](t, k)
            env = {'p': p, 'UP': ctx['UP'], 'b': b}
            base, height, radius = [eval(code, {}, env) for code in expressions]
            row['watchfires'].append({'id': k + 1, 'base': project(cam, base),
                'center': project(cam, base + ctx['UP'] * height * 0.5),
                'top': project(cam, base + ctx['UP'] * height),
                'nominal_halfwidth_px': float(radius * cam.f / cam.project(base)[2]),
                'nominal_height_m': float(height)})
        frames.append(row)
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    rss_bytes = int(rss if sys.platform == 'darwin' else rss * 1024)
    output = {
        'kind': 'source-derived ROI aids; not measured pixels or visibility',
        'method': 'Unchanged AST allowlist of source geometry; no crossing import, no render/build_scene, no frames.',
        'source_sha256': {str(p.relative_to(root)): sha(p) for p in
                          (src, run / 'world.py', run / 'rcam.py', Path(__file__).resolve())},
        'resolution': [960, 402], 'frame_count': len(frames),
        'peak_rss_bytes': rss_bytes,
        'flame_source_expressions': [ast.unparse(flame.args[i]) for i in (3, 4, 5)],
        'limitations': ['No occlusion test, flicker, camera-warp sampling, motion blur, finish or pixel segmentation.',
                        'heart_ideal is the camera target, not the swinging rasterized lantern heart.',
                        'Fixed-heart projection is a stationary-path counterfactual, not a rendered negative control.'],
        'frames': frames,
    }
    write_result(output, args.out)


if __name__ == '__main__':
    main()
