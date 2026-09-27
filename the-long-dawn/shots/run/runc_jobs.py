"""Write RUN-C's cloud jobs: each shot's frames cut into even blocks of at most N frames (one 4-vCPU box each).

    python3 shots/run/runc_jobs.py 16 32 24      # frames per job: reveal, scroll, illum (27 Sep: <= 45 min each)

Every existing cloud/jobs/runC_*.json is replaced."""
import json, os, sys, string
J = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'cloud', 'jobs')
SHOTS = dict(reveal=(0, 239), scroll=(0, 319), illum=(2398, 2877))
per = dict(reveal=int(sys.argv[1]), scroll=int(sys.argv[2]), illum=int(sys.argv[3]))
for f in os.listdir(J):
    if f.startswith('runC_') and f.endswith('.json'):
        os.remove(os.path.join(J, f))
out = []
for shot, (s, e) in SHOTS.items():
    n = e - s + 1
    k = -(-n // per[shot])
    base, extra = divmod(n, k)                      # even blocks
    a = s
    for i in range(k):
        b = a + base + (1 if i < extra else 0) - 1
        tag = string.ascii_lowercase[i] if k <= 26 else f'{i:02d}'
        name = f'runC_{shot}_{tag}'
        job = {
            'name': name,
            'branch': f'claude/render-runC-{shot}-{tag}',
            'setup': [
                'python3 -m pip install -q numba==0.67.0 scipy opencv-python-headless==4.10.0.84 numpy || '
                'python3 -m pip install -q numba numpy scipy opencv-python-headless',
                f'NUMBA_NUM_THREADS=4 python3 shots/run/ink_final.py --shot {shot} --frames {a} --scale 0.25 --threads 4 --out warm',
            ],
            'render': [f'python3 shots/run/ink_final.py --shot {shot} --range {a}-{b} --procs 2 --threads 2 --skip'],
            'out_dir': f'renders/runC_{shot}',
            'frames': f'{a}-{b}',
            'push_every': 300,
            'ship': 'jpg',
        }
        with open(os.path.join(J, name + '.json'), 'w') as fh:
            json.dump(job, fh, indent=1)
            fh.write('\n')
        out.append((name, a, b))
        a = b + 1
for name, a, b in out:
    print(name, f'{a}-{b}', b - a + 1)
print(len(out), 'jobs')
