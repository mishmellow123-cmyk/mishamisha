"""Config gate for cloud sessions working on THE LONG DAWN.

Every cloud session must run as model claude-opus-5-5 at effort max. This reads the session's own
transcript (newest ~/.claude/projects/*/*.jsonl) and prints one verdict line, then the evidence:

    GATE PASS           every assistant turn so far is claude-opus-5-5 at effort max
    GATE FAIL           some assistant turn used another model or effort -> stop, do no work
    GATE UNVERIFIABLE   no transcript / no effort field found -> report what was found

    python3 the-long-dawn/cloud/verify_session.py      (exit code 0 only on PASS)
"""
import collections
import glob
import json
import os
import subprocess
import sys

WANT_MODEL, WANT_EFFORT = 'claude-opus-5-5', 'max'


def transcripts():
    found = glob.glob(os.path.expanduser('~/.claude/projects/*/*.jsonl'))
    if not found:
        try:
            found = subprocess.run(['find', '/', '-xdev', '-name', '*.jsonl', '-path', '*/.claude/projects/*',
                                    '-mmin', '-600'], capture_output=True, text=True, timeout=90).stdout.split()
        except Exception:
            found = []
    return sorted(found, key=os.path.getmtime, reverse=True)


def main():
    env = {k: v for k, v in os.environ.items() if 'EFFORT' in k.upper()}
    ts = transcripts()
    if not ts:
        print('GATE UNVERIFIABLE: no session transcript found')
        print('effort env:', env)
        return 2
    path = ts[0]
    models, efforts, notes = collections.Counter(), collections.Counter(), []
    with open(path, errors='replace') as fh:
        for line in fh:
            try:
                e = json.loads(line)
            except ValueError:
                continue
            if e.get('type') == 'assistant':
                m = (e.get('message') or {}).get('model')
                if m and not m.startswith('<'):
                    models[m] += 1
                for k in ('effort', 'perTurnEffort'):
                    if e.get(k):
                        efforts[f'{k}={e[k]}'] += 1
            elif len(notes) < 6:
                s = json.dumps(e)
                if 'effort' in s.lower() and len(s) < 4000:   # e.g. the /effort command's own output
                    notes.append(s[:300])
    print_evidence = lambda: (print('transcript:', path, f'({len(ts)} found)'), print('models:', dict(models)),
                              print('efforts:', dict(efforts)), print('effort env:', env),
                              [print('note:', n) for n in notes])
    bad_model = [m for m in models if m != WANT_MODEL]
    bad_effort = [k for k in efforts if k.split('=', 1)[1] != WANT_EFFORT]
    if not models or not efforts:
        print('GATE UNVERIFIABLE: transcript has no model/effort fields on assistant turns')
        print_evidence()
        return 2
    if bad_model or bad_effort:
        print(f'GATE FAIL: wrong model {bad_model} / wrong effort {bad_effort} -- do no further work')
        print_evidence()
        return 1
    print(f'GATE PASS: {sum(models.values())} assistant turns, all {WANT_MODEL} at effort {WANT_EFFORT}')
    print_evidence()
    return 0


if __name__ == '__main__':
    sys.exit(main())
