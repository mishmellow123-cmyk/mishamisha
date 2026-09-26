"""Cloud render runner for THE LONG DAWN: render a job's frames, check every frame, push them in batches.

    cd the-long-dawn && python3 cloud/run_job.py cloud/jobs/<job>.json

Job file (JSON):
    name        short id, e.g. "run_v2_b"
    branch      where frames are pushed, e.g. "claude/render-run_v2_b"
    setup       shell commands run first, in order (cwd = the-long-dawn/); any failure aborts
    render      shell commands run CONCURRENTLY (e.g. interleaved frame lists); each gets its own log
    out_dir     output folder relative to the-long-dawn/, e.g. "renders/run_v2"
    frames      frames this job must produce: "1560-1599,1640-1679" (global numbers, files f_%05d.png)
    shape       optional [height, width] every frame must have (default [804, 1920])
    push_every  seconds between batch pushes (default 300)

Frames are written atomically by lib/look.save_png, so any f_*.png that exists is complete. Each one is
decoded before it's pushed (a bad frame is reported and deleted so it can't reach the edit). Progress
lines go to stdout, and the last line is JOB COMPLETE or JOB ENDED (missing: ...).
"""
import json
import os
import subprocess
import sys
import time

import cv2

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))      # the-long-dawn/


def frames_of(spec):
    out = []
    for part in str(spec).split(','):
        part = part.strip()
        if '-' in part:
            a, b = part.split('-')
            out += range(int(a), int(b) + 1)
        elif part:
            out.append(int(part))
    return sorted(set(out))


def log(msg):
    print(time.strftime('%H:%M:%S'), msg, flush=True)


def sh(cmd, **kw):
    return subprocess.run(cmd, shell=True, cwd=HERE, **kw)


def git_push(paths, branch, n_total, n_done):
    if not paths:
        return True
    for i in range(0, len(paths), 200):                  # keep argv short
        sh('git add -f -- ' + ' '.join(paths[i:i + 200]), check=True)
    sh(f'git -c user.name=Claude -c user.email=noreply@anthropic.com commit -q -m '
       f'"cloud render {os.path.basename(branch)}: {n_done}/{n_total} frames"', check=True)
    for attempt in range(4):
        r = sh(f'git push -q origin HEAD:{branch}', capture_output=True, text=True)
        if r.returncode == 0:
            return True
        log(f'push failed (attempt {attempt + 1}): {r.stderr.strip()[-300:]}')
        time.sleep(20 * (attempt + 1))
    return False


def main():
    job = json.load(open(sys.argv[1]))
    name, branch, out_dir = job['name'], job['branch'], job['out_dir']
    want = frames_of(job['frames'])
    shape = tuple(job.get('shape', [804, 1920]))
    push_every = job.get('push_every', 300)
    os.makedirs(os.path.join(HERE, 'cloud_logs'), exist_ok=True)
    os.makedirs(os.path.join(HERE, out_dir), exist_ok=True)
    log(f'JOB {name}: {len(want)} frames -> {out_dir}, pushing to {branch}')

    for cmd in job.get('setup', []):
        log(f'setup: {cmd}')
        t = time.time()
        r = sh(cmd)
        if r.returncode != 0:
            log(f'ERROR setup failed (exit {r.returncode}): {cmd}')
            log('JOB ENDED (setup failed)')
            sys.exit(2)
        log(f'setup ok in {time.time() - t:.0f}s')

    procs = []
    for i, cmd in enumerate(job['render']):
        lf = open(os.path.join(HERE, 'cloud_logs', f'{name}_{i}.log'), 'w')
        procs.append((subprocess.Popen(cmd, shell=True, cwd=HERE, stdout=lf, stderr=subprocess.STDOUT), cmd, lf))
        log(f'render[{i}] started: {cmd}')

    t0 = time.time()
    pushed, bad = set(), set()
    last_push = time.time()
    while True:
        running = [p for p, _, _ in procs if p.poll() is None]
        ready = []
        for f in want:
            if f in pushed or f in bad:
                continue
            p = os.path.join(HERE, out_dir, f'f_{f:05d}.png')
            if os.path.exists(p):
                im = cv2.imread(p)
                if im is None or im.shape[:2] != shape:
                    log(f'ERROR bad frame {f}: {None if im is None else im.shape}; deleted')
                    os.remove(p)
                    bad.add(f)
                    continue
                ready.append(f)
        done = len(pushed) + len(ready)
        if ready and (time.time() - last_push >= push_every or not running or done == len(want)):
            paths = [os.path.join(out_dir, f'f_{f:05d}.png') for f in ready]
            if git_push(paths, branch, len(want), done):
                pushed.update(ready)
                el = time.time() - t0
                log(f'pushed {len(ready)} (total {len(pushed)}/{len(want)}; {el / max(1, len(pushed)):.1f}s/frame wall)')
            last_push = time.time()
        if not running and not ready:
            for p, cmd, lf in procs:
                lf.close()
                if p.returncode != 0:
                    log(f'ERROR render exited {p.returncode}: {cmd} (see cloud_logs/{name}_*.log)')
            missing = [f for f in want if f not in pushed]
            if missing:
                log(f'JOB ENDED (missing {len(missing)}: {missing[:40]}{"..." if len(missing) > 40 else ""})')
                sys.exit(1)
            log(f'JOB COMPLETE: {len(pushed)} frames pushed to {branch} in {(time.time() - t0) / 60:.1f} min')
            return
        time.sleep(30)


if __name__ == '__main__':
    main()
