#!/usr/bin/env python3
"""RUN-A2: write the final FARM jobs for THE CROSSING (A18) and THE BLUE HOUR + A20's title sky (A19-A20), each
sized to finish in <= ~40 minutes on one cpu-8 node (8 single-threaded render processes).

    python3 the-long-dawn/cloud/jobs/make_runA2_jobs.py --x-spf 520 --b-spf 700    # measured single-thread s/frame

Jobs: crossing_a2_NN.json (cut 4880-5839 -> renders/crossing_A) and bluehour_a2_NN.json (cut 5840-6479 ->
renders/bluehour_A). Frames are interleaved within each job's contiguous slice by the render's own --procs pool. No --skip: parked
nodes keep old renders on disk, and a re-render after a code change must never ship them (use farm.py --missing).
The old cloud-lane jobs (crossing_a_1..8, bluehour_a_1..5) are superseded and deleted by this script.
"""
import argparse
import glob
import json
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))
BUDGET_S = 40 * 60            # wall seconds per job on one node
PROCS = 8


def write(prefix, script, a, b, out_dir, spf, extra=''):
    per_job = max(PROCS, int(BUDGET_S * PROCS / spf) // PROCS * PROCS)
    n = math.ceil((b - a + 1) / per_job)
    size = math.ceil((b - a + 1) / n)
    names = []
    for k in range(n):
        f0 = a + k * size
        f1 = min(b, f0 + size - 1)
        name = f'{prefix}_{k + 1:02d}'
        job = {
            'name': name,
            'branch': f'claude/render-{name.replace("_", "-")}',
            'setup': [],
            'render': [f'python3 shots/run/{script} --range {f0}-{f1} --procs {PROCS}{extra}'],
            'out_dir': out_dir,
            'frames': f'{f0}-{f1}',
            'ship': 'jpg',
        }
        with open(os.path.join(HERE, name + '.json'), 'w') as fh:
            json.dump(job, fh, indent=1)
            fh.write('\n')
        names.append((name, f0, f1, (f1 - f0 + 1) * spf / PROCS / 60.0))
    return names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--x-spf', type=float, required=True, help='crossing: single-thread seconds per full-res frame')
    ap.add_argument('--b-spf', type=float, required=True, help='blue hour: single-thread seconds per full-res frame')
    a = ap.parse_args()
    for old in glob.glob(os.path.join(HERE, 'crossing_a_[0-9]*.json')) + \
            glob.glob(os.path.join(HERE, 'bluehour_a_[0-9]*.json')) + \
            glob.glob(os.path.join(HERE, 'crossing_a2_*.json')) + glob.glob(os.path.join(HERE, 'bluehour_a2_*.json')):
        os.remove(old)
    rows = write('crossing_a2', 'crossing.py', 4880, 5839, 'renders/crossing_A', a.x_spf)
    rows += write('bluehour_a2', 'bluehour.py', 5840, 6479, 'renders/bluehour_A', a.b_spf)
    for name, f0, f1, mins in rows:
        print(f'{name}: {f0}-{f1} ({f1 - f0 + 1} f), est {mins:.0f} min on one cpu-8 node')


if __name__ == '__main__':
    main()
