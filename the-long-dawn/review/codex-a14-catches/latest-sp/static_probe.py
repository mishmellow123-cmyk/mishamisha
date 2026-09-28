#!/usr/bin/env python3
"""Read-only static reachability check; imports no project code and performs no JIT/rendering."""
import argparse
import ast
import datetime
import hashlib
import json
from pathlib import Path
import subprocess


def sha(s):
    return hashlib.sha256(s.encode()).hexdigest()


def git(repo, *args):
    return subprocess.check_output(['git', '-C', str(repo), *args], text=True)


def function_map(tree):
    result = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            result[node.name] = node
        elif isinstance(node, ast.ClassDef):
            result.update({node.name + '.' + n.name: n for n in node.body if isinstance(n, ast.FunctionDef)})
    return result


def digest(n):
    return sha(ast.dump(n, include_attributes=False))


def record(n):
    return dict(line=n.lineno, ast_sha256=digest(n), decorators=[ast.unparse(d) for d in n.decorator_list])


def defaults(n):
    return {a.arg: ast.literal_eval(d) for a, d in zip(n.args.args[-len(n.args.defaults):], n.args.defaults)
            if isinstance(d, (ast.Constant, ast.Tuple, ast.List))}


def calls(n, prefix):
    return [c for c in ast.walk(n) if isinstance(c, ast.Call) and ast.unparse(c.func).startswith(prefix)]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo', required=True)
    p.add_argument('--before', default='6ae71801bae7b188a8f2d9533761c7e13824fc06')
    p.add_argument('--after', default='70d102b148ff225947f252f3183aca9503a6d4fd')
    p.add_argument('--out', required=True)
    a = p.parse_args()
    repo = Path(a.repo).resolve()
    refs = {k: git(repo, 'rev-parse', getattr(a, k) + '^{commit}').strip() for k in ('before', 'after')}
    sp = 'the-long-dawn/shots/run/sdfppl.py'
    texts = {k: git(repo, 'show', r + ':' + sp) for k, r in refs.items()}
    maps = {k: function_map(ast.parse(s)) for k, s in texts.items()}
    assert maps['before'].keys() == maps['after'].keys()
    compiled = [n for n, f in maps['after'].items() if f.decorator_list]
    assert len(compiled) == 11
    changed = [n for n in maps['after'] if digest(maps['before'][n]) != digest(maps['after'][n])]
    assert set(changed) == {'Scene.box', '_prim', '_horn', 'render', 'lantern_v3', 'stone_ring'}
    assert {n for n in changed if n in compiled} == {'_prim', '_horn', 'render'}
    box = maps['after']['Scene.box']
    assert defaults(box)['rough'] == 0.0 and defaults(box)['rough_f'] == 0.0
    assert defaults(box)['glass'] is False
    box_guards = [ast.unparse(n.test) for n in ast.walk(box) if isinstance(n, ast.If)]
    assert box_guards == ['rough > 0.0 and (not glass)']
    begin = maps['after']['Scene.begin']
    object_row = next(n.value for n in begin.body if isinstance(n, ast.Assign)
                      and ast.unparse(n.targets[0]) == 'self._o')
    assert ast.literal_eval(object_row.elts[14]) == 0.0
    traveller = maps['after']['traveller']
    box_calls = calls(traveller, 'sc.box')
    assert box_calls and all(not any(k.arg in {'rough', 'rough_f', 'glass'} for k in c.keywords) for c in box_calls)
    primitive_calls = []
    material_positions = {'sc.cone': 4, 'sc.box': 5, 'sc.bell': 8}
    for c in calls(traveller, 'sc.'):
        name = ast.unparse(c.func)
        if name not in material_positions:
            continue
        kws = {k.arg: k.value for k in c.keywords}
        mi = material_positions[name]
        material = kws.get('mat', c.args[mi] if len(c.args) > mi else ast.Constant(0))
        m = ast.literal_eval(material)
        assert m in (0, 2)
        primitive_calls.append(dict(line=c.lineno, method=name, material=m, expression=ast.unparse(c)))
    run = repo / 'the-long-dawn/shots/run'
    wa_text = (run / 'watchers_a.py').read_text()
    wa = function_map(ast.parse(wa_text))
    sp_calls = sorted(set(ast.unparse(c.func) for c in calls(ast.parse(wa_text), 'SP.')))
    assert sp_calls == ['SP.Scene', 'SP.render', 'SP.traveller']
    figure_calls = calls(wa['_figure'], 'SP.traveller')
    assert len(figure_calls) == 1
    kw = {k.arg for k in figure_calls[0].keywords}
    assert not kw.intersection({'cloth', 'pack', 'bedroll', 'lantern_side', 'lantern_mode'})
    assert defaults(traveller)['cloth'] is True and defaults(traveller)['lantern_side'] == 0.0
    wrap_calls = calls(wa['_figure'], 'sc.bell')
    assert len(wrap_calls) == 1 and ast.literal_eval(wrap_calls[0].args[8]) == 0
    assert any(ast.unparse(c.func) == 'HA.draw' for c in calls(wa['draw_figures'], 'HA.'))
    assert 'SP.' not in (run / 'hearth_a.py').read_text()
    unchanged_inputs = {}
    for path in ['watchers_a.py', 'beaconrun_a.py', 'nighta.py', 'hearth_a.py', 'fire_near_a.py']:
        full = 'the-long-dawn/shots/run/' + path
        atext, btext = (git(repo, 'show', refs[k] + ':' + full) for k in ('before', 'after'))
        assert atext == btext
        unchanged_inputs[full] = dict(before_source_sha256=sha(atext), after_source_sha256=sha(btext), identical=True)
    result = dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  method='Standard-library AST/source inspection only; no project import, scene generation, JIT or pixels',
                  script_sha256=sha(Path(__file__).read_text()), repo=str(repo), commits=refs,
                  observed_worktree_head=git(repo, 'rev-parse', 'HEAD').strip(),
                  sdfppl={k: dict(source_sha256=sha(texts[k]), functions={n: record(f) for n, f in maps[k].items()})
                          for k in refs},
                  compiled_function_count=len(compiled), compiled_functions=compiled,
                  changed_functions=changed, changed_compiled_functions=[n for n in changed if n in compiled],
                  unchanged_production_shot_inputs=unchanged_inputs,
                  watcher_worktree_source_sha256=sha(wa_text), watcher_sp_calls=sp_calls,
                  traveller_box_calls=[dict(line=c.lineno, expression=ast.unparse(c)) for c in box_calls],
                  traveller_primitive_material_calls=primitive_calls,
                  static_branch_assessment=[
                      dict(branch='rough box displacement', gate='primitive type 1 AND P[14] > 0',
                           result='not selected by current watcher calls',
                           reason='traveller box calls omit rough; its default is zero, and Scene.box initializes rows to zero. Bells have type4, so their positive P[14] fold depth does not satisfy box type.'),
                      dict(branch='horn transmission', gate='glass hit AND O[14] > 0.5',
                           result='not selected by current watcher calls',
                           reason='Scene.begin initializes O[14] to 0; watcher does not call lantern_v3 or emit glass primitives.'),
                      dict(branch='rough stone shading', gate='material >= 5.5',
                           result='not selected by current watcher calls',
                           reason='traveller primitive material arguments are 0 or 2 and direct wrap is material 0. Hearth uses the separate HA.draw tracer, without calling SP.stone_ring.'),
                      dict(branch='lantern_v3 bar position and stone_ring geometry',
                           result='not called by current watcher scene',
                           reason='watchers_a SP call set is Scene/traveller/render only; A14 invokes WA.draw_figures.')],
                  limits=['The compiled _prim, _horn and render functions have changed; prior compiled-function identity does not apply.',
                          'Reachability follows current source call arguments; actual geometry arrays are checked separately by another agent.',
                          'Unselected source branches do not establish machine-code identity or pixel equivalence after recompilation.',
                          'The older pixel render retains its original source provenance and does not validate this newer shader.'])
    out = Path(a.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(receipt=str(out), changed_compiled_functions=result['changed_compiled_functions'],
                          watcher_sp_calls=sp_calls,
                          traveller_materials=sorted(set(v['material'] for v in primitive_calls))), indent=2))


if __name__ == '__main__':
    main()
