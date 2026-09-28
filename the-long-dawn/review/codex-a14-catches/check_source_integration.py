#!/usr/bin/env python3
"""Static integration receipt. Standard library only; never imports shot/farm code.

Reads pinned Git blobs and local job/renderer files. Writes only --out.
Run with explicit --repo, --reference, --production, --since and --out.
"""
import argparse
import ast
import datetime
import difflib
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
from types import SimpleNamespace


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def ah(node):
    return sha(ast.dump(node, include_attributes=False))


def git(repo, *args):
    return subprocess.check_output(['git', '-C', str(repo), *args], text=True)


def functions(tree):
    result = {}
    def visit(node, prefix=''):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                name = prefix + child.name
                if not isinstance(child, ast.ClassDef):
                    decorators = [ast.unparse(d) for d in child.decorator_list]
                    compiled = any((d.func if isinstance(d, ast.Call) else d).id in {'njit', 'jit', 'vectorize', 'guvectorize'}
                                   for d in child.decorator_list
                                   if isinstance(d.func if isinstance(d, ast.Call) else d, ast.Name))
                    result[name] = dict(line=child.lineno, decorators=decorators,
                                        compiled=compiled, ast_sha256=ah(child))
                visit(child, name + '.')
            else:
                visit(child, prefix)
    visit(tree)
    return result


def top_nodes(tree):
    result = {}
    for i, node in enumerate(tree.body):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            key = type(node).__name__ + ':' + node.name
        else:
            key = f'{i}:{type(node).__name__}:' + ast.unparse(node).splitlines()[0][:160]
        result[key] = dict(line=node.lineno, ast_sha256=ah(node))
    return result


def source_report(text):
    tree = ast.parse(text)
    return dict(source_sha256=sha(text), module_ast_sha256=ah(tree),
                functions=functions(tree), top_level_nodes=top_nodes(tree))


def restricted_value(node, env):
    """Evaluate only the renderer's path assignment expressions, not its code."""
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        return env[node.id]
    if isinstance(node, ast.Attribute):
        return getattr(restricted_value(node.value, env), node.attr)
    if isinstance(node, ast.IfExp):
        return restricted_value(node.body if restricted_value(node.test, env) else node.orelse, env)
    if isinstance(node, ast.Compare) and len(node.ops) == 1:
        left, right = restricted_value(node.left, env), restricted_value(node.comparators[0], env)
        if isinstance(node.ops[0], ast.Is):
            return left is right
        if isinstance(node.ops[0], ast.IsNot):
            return left is not right
    if isinstance(node, ast.Call) and ast.unparse(node.func) in {'os.path.join', 'os.path.isabs'}:
        assert not node.keywords
        return restricted_value(node.func, env)(*[restricted_value(a, env) for a in node.args])
    raise ValueError('Unsupported path expression: ' + ast.dump(node))


def option(tokens, name, default=None):
    return tokens[tokens.index(name) + 1] if name in tokens else default


def frame_range(value):
    a, b = map(int, value.split('-'))
    assert a <= b
    return list(range(a, b + 1))


def job_report(repo, path, text):
    job = json.loads(text)
    root = repo / 'the-long-dawn'
    assert len(job['render']) == 1
    tokens = shlex.split(job['render'][0])
    renderer = next(t for t in tokens if t.startswith('shots/') and t.endswith('.py'))
    renderer_text = (root / renderer).read_text()
    tree = ast.parse(renderer_text)
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
    assignments = {n.targets[0].id: n.value for n in main.body
                   if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name)
                   and n.targets[0].id in {'base', 'out'}}
    assert 'out' in assignments
    env = dict(ROOT=str(root), PI=SimpleNamespace(CM=SimpleNamespace(ROOT=str(root))),
               a=SimpleNamespace(out=option(tokens, '--out')), os=os)
    for name, expr in assignments.items():
        env[name] = restricted_value(expr, env)
    resolved = os.path.normpath(env['out'])
    collected = os.path.normpath(str(root / job['out_dir']))
    wanted = frame_range(job['frames'])
    commanded = frame_range(option(tokens, '--range'))
    limits = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
                  and any(isinstance(t, ast.Tuple) and [ast.unparse(e) for e in t.elts] == ['FR0', 'NFR']
                          for t in n.targets))
    assert wanted == commanded, 'Job frames differ from renderer arguments'
    assert wanted == list(range(limits[0], limits[0] + limits[1])), 'Job does not cover entire shot'
    assert resolved == collected, 'Renderer/collector output mismatch'
    assert '--skip' in tokens and option(tokens, '--procs') == '2'
    assert 'NUMBA_NUM_THREADS=1' in tokens
    assert option(tokens, '--scale', '1.0') == '1.0' and job['ship'] == 'jpg'
    warm = shlex.split(job['setup'][-1])
    warmenv = dict(env, a=SimpleNamespace(out=option(warm, '--out')))
    for name, expr in assignments.items():
        warmenv[name] = restricted_value(expr, warmenv)
    warmout = os.path.normpath(warmenv['out'])
    assert warmout != collected and not warmout.startswith(collected + os.sep)
    return dict(path=path, job_source_sha256=sha(text), name=job['name'], legacy_output_branch=job['branch'],
                renderer=renderer, renderer_source_sha256=sha(renderer_text), frame_count=len(wanted),
                first_frame=wanted[0], last_frame=wanted[-1], complete_frame_list=wanted,
                renderer_limits=list(limits), path_expressions={k: ast.unparse(v) for k, v in assignments.items()},
                resolved_writer_directory=resolved, resolved_collector_directory=collected,
                warmup_directory=warmout, writer_matches_collector=True, covers_entire_shot=True,
                output_directory_exists_locally=Path(collected).exists(), json=job)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for flag in ('repo', 'reference', 'production', 'since', 'out'):
        ap.add_argument('--' + flag, required=True)
    args = ap.parse_args()
    repo = Path(args.repo).resolve()
    out = Path(args.out).resolve()
    refs = {k: git(repo, 'rev-parse', getattr(args, k) + '^{commit}').strip()
            for k in ('reference', 'production', 'since')}
    sp = 'the-long-dawn/shots/run/sdfppl.py'
    texts = {k: git(repo, 'show', f'{v}:{sp}') for k, v in refs.items()}
    sp_reports = {k: source_report(v) for k, v in texts.items()}
    before, after = sp_reports['reference'], sp_reports['production']
    assert before['functions'].keys() == after['functions'].keys()
    compiled = [name for name, record in before['functions'].items() if record['compiled']]
    assert compiled, 'No compiled functions found; empty comparison must not pass'
    assert compiled == [name for name, record in after['functions'].items() if record['compiled']]
    changed_functions = [name for name in before['functions']
                         if before['functions'][name]['ast_sha256'] != after['functions'][name]['ast_sha256']]
    assert changed_functions == ['_traveller_v2'], changed_functions
    assert all(before['functions'][name] == after['functions'][name] for name in compiled)
    changed_nodes = [name for name in before['top_level_nodes']
                     if before['top_level_nodes'][name] != after['top_level_nodes'].get(name)]
    assert before['top_level_nodes'].keys() == after['top_level_nodes'].keys()
    assert changed_nodes == ['FunctionDef:_traveller_v2'], changed_nodes
    old = "out['reach_hand'] = arm(shB, shB + d * reach_len * h, -sl)"
    new = "out['reach_hand'] = arm(shB, shB + d * 0.62 * h, -sl)"
    legacy = next(n for n in ast.parse(texts['reference']).body
                  if isinstance(n, ast.FunctionDef) and n.name == '_traveller_v2')
    lines = texts['reference'].splitlines(True)
    segment = ''.join(lines[legacy.lineno - 1:legacy.end_lineno])
    assert segment.count(old) == 1
    replaced = ''.join(lines[:legacy.lineno - 1]) + segment.replace(old, new, 1) + ''.join(lines[legacy.end_lineno:])
    assert replaced == texts['production'], 'Other SP source changes exist'
    changes = git(repo, 'diff', '--name-status', refs['since'], refs['production']).splitlines()
    run_changes = [line for line in changes if '\tthe-long-dawn/shots/run/' in line]
    changed_run_sources = []
    for row in run_changes:
        status, path = row.split('\t')
        record = dict(status=status, path=path)
        for key, ref in refs.items():
            proc = subprocess.run(['git', '-C', str(repo), 'show', f'{ref}:{path}'], capture_output=True, text=True)
            record[key + '_source_sha256'] = sha(proc.stdout) if proc.returncode == 0 else None
        changed_run_sources.append(record)
    names = ('watchers_a_catches3', 'beaconrun_a_catches3')
    jobs = {}
    local_source_texts = {}
    for name in names:
        path = f'the-long-dawn/cloud/jobs/{name}.json'
        text = (repo / path).read_text()
        jobs[name] = job_report(repo, path, text)
        local_source_texts[path] = sha(text)
    baseline_path = 'the-long-dawn/cloud/jobs/watchers_a_hearth3.json'
    baseline_text = git(repo, 'show', refs['production'] + ':' + baseline_path)
    baseline = job_report(repo, baseline_path + '@' + refs['production'], baseline_text)
    a15 = jobs['watchers_a_catches3']
    assert json.loads(baseline_text.replace('hearth3', 'catches3')) == a15['json']
    assert a15['resolved_writer_directory'] != baseline['resolved_writer_directory']
    assert len({j['resolved_writer_directory'] for j in jobs.values()} |
               {baseline['resolved_writer_directory']}) == 3
    consumer_paths = ['the-long-dawn/cloud/farm.py', 'the-long-dawn/cloud/farm_node.py',
                      'the-long-dawn/cloud/run_job.py']
    consumers = {p: sha((repo / p).read_text()) for p in consumer_paths}
    receipt = dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                   method='Python standard-library AST, JSON, shlex and Git blob reads; no project imports, JIT, rendering or farm calls',
                   repo=str(repo), commits=refs, observed_worktree_head=git(repo, 'rev-parse', 'HEAD').strip(),
                   script_sha256=sha(Path(__file__).read_text()),
                   sdfppl=dict(path=sp, reports=sp_reports, compiled_function_count=len(compiled),
                               compiled_functions=compiled, all_compiled_function_records_identical=True,
                               changed_function_asts=changed_functions, changed_top_level_nodes=changed_nodes,
                               exact_single_line_substitution_proved=True,
                               reference_to_production_diff=''.join(difflib.unified_diff(
                                   texts['reference'].splitlines(True), texts['production'].splitlines(True),
                                   fromfile=refs['reference'] + ':' + sp, tofile=refs['production'] + ':' + sp))),
                   all_production_changes_since_prior_tip=changes,
                   run_source_and_test_changes_since_prior_tip=changed_run_sources,
                   current_job_reports=jobs, production_a15_base_job=baseline,
                   job_namespace_only_change_from_a15_base=True, three_distinct_output_directories=True,
                   consumer_source_sha256=consumers,
                   consumer_contract={'farm_output': 'ROOT/out_dir; farm.py Job.__init__',
                                      'farm_source_branch': 'claude/long-dawn-v2; farm_node.py BRANCH',
                                      'branch_json_meaning': 'legacy run_job.py output branch, not farm source selection',
                                      'farm_skip': 'node clears both requested PNG/JPEG before render; farm --missing recognizes both on receiver',
                                      'cli_skip': 'A15 recognizes PNG/JPEG; A14 recognizes PNG only',
                                      'legacy_resume_limit': 'run_job.py discovers PNG only; A15 JPEG-only resume skips rendering but is not collected'},
                   limits=['Static identity is not a compiled binary, pixel or visual-quality test.',
                           'Job files are local worktree snapshots; hashes identify their exact content.',
                           'Distinct output namespaces do not prove those paths are empty on remote workers.',
                           'No cloud run, source checkout, receipt or rendered frame was queried.'])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(dict(receipt=str(out), commits=refs, compiled_functions=len(compiled),
                          changed_functions=changed_functions,
                          jobs={k: {'count': v['frame_count'], 'output': v['resolved_writer_directory']}
                                for k, v in jobs.items()}), indent=2))


if __name__ == '__main__':
    main()
