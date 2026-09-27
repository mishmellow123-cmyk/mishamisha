#!/usr/bin/env python3
"""THE LONG DAWN render farm: run cloud/jobs/*.json on Autoresearch nodes; the frames stream back to this Mac.

    python3 the-long-dawn/cloud/farm.py the-long-dawn/cloud/jobs/<job>.json [more.json ...]
            [--nodes N] [--gpu] [--frames A-B[,C-D]] [--test [K]] [--local-out DIR] [--missing] [--dry-run]
    python3 the-long-dawn/cloud/farm.py status        # farm nodes, who holds them, mission spend, token expiry
    python3 the-long-dawn/cloud/farm.py stop-idle     # stop every farm node that no live farm.py is using

What a run does:
  * plans units: each job's frames are split across --nodes N units. Every unit keeps the job's own lanes (its
    render commands), each lane cut to a contiguous slice of its frames, so a node runs the job exactly as the
    department designed it, on fewer frames. A lane that leaves the node's CPUs mostly idle is cut into more
    processes (sized to the node's cpu_limit, never to os.cpu_count()). A job whose frame arguments farm.py
    can't read runs whole on one node.
  * nodes: cpu-8 for numba work, h100-1 (--gpu, automatic for Blender jobs) for Cycles. It reuses the farm's
    parked nodes first (their disks keep the venvs, numba caches and the Cycles kernel cache), then creates
    new ones up to the caps. Each node gets: the repo at the branch tip (git fetch --depth 1 + reset --hard
    before every unit, so PUSH YOUR CODE FIRST), the pinned numba stack, and on GPU nodes bpy 4.5.14.
  * frames: finished PNGs become decode-checked JPEG q95 4:4:4 on the node, stream back over the node's own
    HTTPS endpoint as they land, are checked again here (sha256 + decode at the job's shape, default
    804x1920), and are swapped into the-long-dawn/<out_dir> atomically. A landed JPEG supersedes an older PNG.
    Nothing is pushed to git and no credential ever leaves this Mac.
  * cost: every node stops the moment it has no more work (and stops itself after 10 idle minutes if this
    process dies). Spend is refused past LDFARM_SPEND_STOP (default $200) of the farm mission.

Modes:
  --test [K]   look-dev: K frames (default 4) spread over the job (or over --frames), on one node unless
               --nodes; lands in renders/_farmtest/<job>/ unless --local-out, so a test never overwrites a
               finished render.
  --missing    render only the frames this Mac doesn't already have.
  --dry-run    print the plan (units and rewritten commands) and touch nothing.

Credential: ~/.config/longdawn-farm/token (chmod 600; a workspace-scoped infra token), or $LDFARM_TOKEN.
Limits (env): LDFARM_MAX_CPU (30), LDFARM_MAX_GPU (2), LDFARM_MAX_ENDPOINTS (15: one per streaming node).
"""
import argparse
import hashlib
import io
import json
import math
import os
import re
import signal
import sys
import tarfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor

VENV_PY = os.path.expanduser('~/.venvs/longdawn/bin/python')
try:
    import cv2
    import numpy as np
except ImportError:                                   # system python3: re-run under the project venv (has cv2)
    if os.path.exists(VENV_PY) and os.path.realpath(sys.executable) != os.path.realpath(VENV_PY):
        os.execv(VENV_PY, [VENV_PY] + sys.argv)
    raise

HERE = os.path.dirname(os.path.abspath(__file__))                 # the-long-dawn/cloud
ROOT = os.path.dirname(HERE)                                      # the-long-dawn
AGENT = os.path.join(HERE, 'farm_node.py')
API = os.environ.get('LDFARM_API', 'https://autoresearch.sfcompute.com/preview')
CFG = os.path.expanduser('~/.config/longdawn-farm')
CACHE = os.path.expanduser('~/.cache/ldfarm')
LEASES = os.path.join(CACHE, 'leases')
MISSION = os.environ.get('LDFARM_MISSION', 'long-dawn-render-farm')
PORT = 8700
MAX_CPU = int(os.environ.get('LDFARM_MAX_CPU', '30'))
MAX_GPU = int(os.environ.get('LDFARM_MAX_GPU', '2'))
MAX_STREAM = int(os.environ.get('LDFARM_MAX_ENDPOINTS', '15'))
SPEND_WARN = float(os.environ.get('LDFARM_SPEND_WARN', '150'))
SPEND_STOP = float(os.environ.get('LDFARM_SPEND_STOP', '200'))
KINDS = {'cpu': dict(chip='cpu-8', prefix='ldf-c', cpu=8, cap=MAX_CPU),
         'gpu': dict(chip='h100-1', prefix='ldf-g', cpu=14, cap=MAX_GPU)}
BATCH = 16                                                        # frames per download request

BOOT = r'''set -e
export PATH=$HOME/.local/bin:$PATH
mkdir -p ~/ld/runs && cd ~/ld
pkill -f "farm_node[.]py agent" || true
command -v uv >/dev/null 2>&1 || (curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1)
[ -d mishamisha/.git ] || git clone -q --single-branch --branch claude/long-dawn-v2 --depth 1 --filter=blob:none https://github.com/mishmellow123-cmyk/mishamisha
git -C mishamisha fetch -q --depth 1 origin claude/long-dawn-v2 && git -C mishamisha reset -q --hard FETCH_HEAD
[ -x venv/bin/python ] || uv venv -q --seed --python 3.12 venv
venv/bin/python -c "import numba, numpy, scipy, cv2, PIL, fontTools, pip" 2>/dev/null || uv pip install -q --python venv/bin/python pip numba==0.67.0 llvmlite==0.49.0 numpy==2.5.3 scipy==1.18.1 opencv-python-headless==4.10.0.84 pillow==12.3.0 fonttools==4.66.0
if command -v nvidia-smi >/dev/null 2>&1; then
  [ -x bpyenv/bin/python ] || uv venv -q --python 3.11 bpyenv
  [ -d bpyenv/lib/python3.11/site-packages/bpy ] || uv pip install -q --python bpyenv/bin/python bpy==4.5.14
  bpyenv/bin/python -c "import bpy" >/dev/null 2>&1 || { sudo apt-get update -qq >/dev/null 2>&1; sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq --no-install-recommends libx11-6 libxrender1 libxxf86vm1 libxfixes3 libxi6 libxkbcommon0 libsm6 libice6 libxext6 libgl1 libegl1 libglu1-mesa >/dev/null 2>&1; }
  bpyenv/bin/python -c "import bpy" >/dev/null 2>&1 && echo "bpy ok"
  ln -sfn $HOME/ld/bpyenv $HOME/bpyenv
fi
echo "BOOT OK $(git -C mishamisha rev-parse --short HEAD)"
'''

STOP = threading.Event()
PRINT = threading.Lock()


def log(msg):
    with PRINT:
        print(time.strftime('%H:%M:%S') + ' ' + str(msg), flush=True)


# ------------------------------------------------------------------ platform API (the token never leaves here)

class ApiError(Exception):
    def __init__(self, status, code, message):
        super().__init__(f'{status} {code}: {message}')
        self.status, self.code, self.message = status, code, message or ''


def _token():
    t = os.environ.get('LDFARM_TOKEN')
    if not t:
        p = os.path.join(CFG, 'token')
        if not os.path.exists(p):
            sys.exit(f'farm: no credential at {p} (ask FARM to vend one)')
        t = open(p).read().strip()
    return t


def _meta():
    try:
        return json.load(open(os.path.join(CFG, 'token.meta.json')))
    except (OSError, ValueError):
        return {}


WORKSPACE = os.environ.get('LDFARM_WORKSPACE') or _meta().get('workspace') or 'long-dawn-farm'


def api(method, path, body=None, params=None, timeout=90, tries=6):
    params = dict(params or {})
    params['workspace'] = WORKSPACE
    url = API + path + '?' + urllib.parse.urlencode(params)
    data = json.dumps(body).encode() if body is not None else None
    last = None
    for attempt in range(tries):
        req = urllib.request.Request(url, data=data, method=method,
                                     headers={'Authorization': 'Bearer ' + _token(), 'Content-Type': 'application/json'})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                raw = r.read()
                return json.loads(raw) if raw else {}
        except urllib.error.HTTPError as e:
            txt = e.read().decode('utf-8', 'replace')
            try:
                err = json.loads(txt).get('error', {})
            except ValueError:
                err = {'message': txt[:300]}
            if e.code == 401:
                sys.exit('farm: the farm credential was refused (expired?). Ask FARM to vend a new token.')
            if e.code in (429, 503) or e.code >= 500:
                last = ApiError(e.code, err.get('code'), err.get('message'))
                time.sleep(min(float(e.headers.get('Retry-After') or (2 + 3 * attempt)), 60))
                continue
            raise ApiError(e.code, err.get('code'), err.get('message'))
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
            last = ApiError(0, 'network', str(e)[:200])
            time.sleep(2 + 3 * attempt)
    raise last or ApiError(0, 'unreachable', path)


def get_node(name):
    try:
        return api('GET', f'/nodes/{name}')
    except ApiError as e:
        if e.status == 404 or 'no node named' in e.message:
            return None
        raise


def list_nodes():
    return api('GET', '/nodes').get('items', [])


def mission_cost():
    try:
        m = api('GET', f'/missions/{MISSION}')
    except ApiError:
        return None
    for k in ('cost', 'receipt', 'spend'):
        v = m.get(k)
        if isinstance(v, dict):
            for kk in ('usd', 'total_usd', 'spend_usd'):
                if isinstance(v.get(kk), (int, float)):
                    return float(v[kk])
        elif isinstance(v, (int, float)):
            return float(v)
    for k in ('cost_usd', 'total_usd', 'spend_usd'):
        if isinstance(m.get(k), (int, float)):
            return float(m[k])
    return None


# ------------------------------------------------------------------ leases: concurrent farm.py runs never share a node

def _alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def lease(name, run_id):
    os.makedirs(LEASES, exist_ok=True)
    p = os.path.join(LEASES, name + '.json')
    for _ in range(2):
        try:
            fd = os.open(p, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError:
            try:
                info = json.load(open(p))
            except (OSError, ValueError):
                info = {}
            if info.get('pid') and _alive(info['pid']):
                return False
            try:
                os.remove(p)
            except OSError:
                pass
            continue
        with os.fdopen(fd, 'w') as fh:
            json.dump({'pid': os.getpid(), 'run': run_id, 't': time.time()}, fh)
        return True
    return False


def release(name):
    p = os.path.join(LEASES, name + '.json')
    try:
        if json.load(open(p)).get('pid') == os.getpid():
            os.remove(p)
    except (OSError, ValueError):
        pass


def live_leases():
    out = {}
    if os.path.isdir(LEASES):
        for fn in os.listdir(LEASES):
            try:
                info = json.load(open(os.path.join(LEASES, fn)))
            except (OSError, ValueError):
                continue
            if info.get('pid') and _alive(info['pid']):
                out[fn[:-5]] = info
    return out


# ------------------------------------------------------------------ frame arguments: read and rewrite

def frames_of(spec):
    out = []
    for part in str(spec).split(','):
        part = part.strip()
        step = 1
        if ':' in part:
            part, st = part.split(':')
            step = int(st)
        if '-' in part:
            a, b = part.split('-')
            out += range(int(a), int(b) + 1, step)
        elif part:
            out.append(int(part))
    return sorted(set(out))


def compact(frames):
    frames = sorted(frames)
    out, i = [], 0
    while i < len(frames):
        j = i
        while j + 1 < len(frames) and frames[j + 1] == frames[j] + 1:
            j += 1
        out.append(str(frames[i]) if i == j else f'{frames[i]}-{frames[j]}')
        i = j + 1
    return ','.join(out)


RX = [('range', re.compile(r'(--range[ =])(\d+)-(\d+)(?=\s|$|[;)&|])')),
      ('frames', re.compile(r'(--frames[ =])(\d[\d,:\-]*)(?=\s|$|[;)&|])')),
      ('pos', re.compile(r'(\brender\.py\s+)(\d[\d,:\-]*)(?=\s|$|[;)&|])'))]
RX_STEP = re.compile(r'--step[ =](\d+)')


class LaneCmd:
    """One render command whose frame argument farm.py can rewrite. seq = its frames in its own order."""

    def __init__(self, cmd):
        self.cmd = cmd
        hits = [(kind, m) for kind, rx in RX for m in rx.finditer(cmd)]
        if len(hits) != 1:
            raise ValueError(f'{len(hits)} frame arguments')
        self.kind, m = hits[0]
        self.span = m.span(2) if self.kind != 'range' else (m.start(2), m.end(3))
        ms = RX_STEP.search(cmd)
        self.step = int(ms.group(1)) if ms and self.kind in ('range', 'frames') else 1
        v = m.group(2) if self.kind != 'range' else f'{m.group(2)}-{m.group(3)}'
        if self.kind == 'range':
            a, b = int(m.group(2)), int(m.group(3))
            self.seq = list(range(a, b + 1, self.step))
            self.style = 'ap'
        elif re.fullmatch(r'\d+-\d+', v):
            a, b = map(int, v.split('-'))
            self.seq = list(range(a, b + 1, self.step))
            self.style = 'ap'
        elif re.fullmatch(r'\d+-\d+:\d+', v) and self.kind == 'pos':
            ab, st = v.split(':')
            a, b = map(int, ab.split('-'))
            self.step = int(st)
            self.seq = list(range(a, b + 1, self.step))
            self.style = 'ap:'
        else:
            self.seq = sorted(set(frames_of(v)))
            self.style = 'list'
        t = re.search(r'NUMBA_NUM_THREADS=(\d+)', cmd) or re.search(r'--threads[ =](\d+)', cmd)
        p = re.search(r'--(?:procs|workers)[ =](\d+)', cmd)
        self.threads = int(t.group(1)) if t else 1
        self.procs = int(p.group(1)) if p else 1

    def demand(self, n_frames):
        return self.threads * max(1, min(self.procs, n_frames))

    def runs(self, sub):
        """Split a subset of seq into pieces this command's own syntax can express."""
        sub = [f for f in self.seq if f in set(sub)]
        if not sub:
            return []
        if self.style == 'list':
            return [sub]
        out, cur = [], [sub[0]]
        for f in sub[1:]:
            if f == cur[-1] + self.step:
                cur.append(f)
            else:
                out.append(cur)
                cur = [f]
        return out + [cur]

    def render(self, piece):
        a, b = self.span
        if self.style == 'list':
            val = ','.join(map(str, piece))
        elif self.style == 'ap:':
            val = f'{piece[0]}-{piece[-1]}:{self.step}'
        else:
            val = f'{piece[0]}-{piece[-1]}'
        return self.cmd[:a] + val + self.cmd[b:]


def split_even(seq, k):
    n = len(seq)
    return [seq[round(i * n / k):round((i + 1) * n / k)] for i in range(k)]


# ------------------------------------------------------------------ jobs and units

SKIP_SETUP = re.compile(r'apt-get|-m venv\b.*bpy|bpy==')


class Job:
    def __init__(self, path, args):
        self.path = path
        self.spec = json.load(open(path))
        self.name = self.spec['name']
        self.shape = tuple(self.spec.get('shape', [804, 1920]))
        outs = self.spec.get('outputs') or [{'out_dir': self.spec['out_dir'], 'frames': self.spec['frames']}]
        self.outputs = [dict(key=i, out_dir=o['out_dir'], frames=frames_of(o['frames'])) for i, o in enumerate(outs)]
        cmds = self.spec.get('render', [])
        self.gpu = args.gpu or any(re.search(r'MT3D_BLENDER|bpyenv', c) for c in cmds)
        self.setup = [c for c in self.spec.get('setup', []) if not SKIP_SETUP.search(c)]
        self.skipped_setup = [c for c in self.spec.get('setup', []) if SKIP_SETUP.search(c)]
        try:
            self.lanes = [LaneCmd(c) for c in cmds]
        except ValueError:
            self.lanes = None
        # local destination per output
        if args.local_out:
            base = os.path.abspath(os.path.expanduser(args.local_out))
        elif args.test is not None:
            base = os.path.join(ROOT, 'renders', '_farmtest', self.name)
        else:
            base = None
        for o in self.outputs:
            if base is None:
                o['dest'] = os.path.join(ROOT, o['out_dir'])
            elif len(self.outputs) == 1:
                o['dest'] = base
            else:
                o['dest'] = os.path.join(base, os.path.basename(o['out_dir'].rstrip('/')))
        # which frames to render
        allf = sorted({f for o in self.outputs for f in o['frames']})
        want = set(allf)
        if args.frames:
            want &= set(frames_of(args.frames))
        if args.test is not None:
            pool = sorted(want)
            k = args.test or 4
            if len(pool) > k:
                pool = [pool[round(i * (len(pool) - 1) / (k - 1))] for i in range(k)] if k > 1 else [pool[len(pool) // 2]]
            want = set(pool)
        if args.missing:
            have = set()
            for o in self.outputs:
                for f in o['frames']:
                    p = os.path.join(o['dest'], f'f_{f:05d}')
                    if os.path.exists(p + '.png') or os.path.exists(p + '.jpg'):
                        have.add((o['key'], f))
            want = {f for f in want if any((o['key'], f) not in have and f in o['frames'] for o in self.outputs)}
        self.want = want
        self.whole = self.lanes is None
        if self.whole and (args.frames or args.test is not None or args.missing):
            log(f'[{self.name}] note: its frame arguments are not rewritable, so it runs whole (all frames)')
            self.want = set(allf)

    def units(self, n_units, cpu_limit, retry_of=None, frames=None):
        want = set(frames) if frames is not None else self.want
        if not want:
            return []
        if self.whole:
            return [self._unit(0, [dict(cmd=c) for c in self.spec['render']], want, len(self.spec['render']), retry_of)]
        lanes = [(ln, [f for f in ln.seq if f in want]) for ln in self.lanes]
        lanes = [(ln, fs) for ln, fs in lanes if fs]
        disjoint = sum(len(fs) for _, fs in lanes) == len({f for _, fs in lanes for f in fs})
        n_units = max(1, min(n_units, max(len(fs) for _, fs in lanes)))
        if disjoint:                          # every unit gets a contiguous slice of EVERY lane: balanced nodes
            per_unit = [[(ln, part) for ln, part in ((ln, split_even(fs, n_units)[j]) for ln, fs in lanes) if part]
                        for j in range(n_units)]
        else:                                 # lanes share frames: a frame must live in exactly one unit
            chunks = split_even(sorted({f for _, fs in lanes for f in fs}), n_units)
            per_unit = [[(ln, [f for f in fs if f in set(ch)]) for ln, fs in lanes] for ch in chunks]
            per_unit = [[(ln, p) for ln, p in u if p] for u in per_unit]
        out = []
        for j, parts in enumerate(per_unit):
            if not parts:
                continue
            items, frames_u = [], set()
            demand = sum(ln.demand(len(p)) for ln, p in parts)
            k = max(1, int(cpu_limit // max(1, demand))) if (not self.gpu and demand * 2 <= cpu_limit) else 1
            for ln, p in parts:
                frames_u |= set(p)
                for run in ln.runs(p):
                    for piece in (split_even(run, min(k, len(run))) if k > 1 else [run]):
                        if piece:
                            items.append(dict(cmd=ln.render(piece), n=len(piece), d=ln.demand(len(piece))))
            d_avg = sum(i['d'] for i in items) / len(items)
            conc = len(items) if sum(i['d'] for i in items) <= 2 * cpu_limit else max(1, round(cpu_limit / d_avg))
            if self.gpu:
                conc = min(conc, max(1, len(self.spec['render'])))
            out.append(self._unit(j, [dict(cmd=i['cmd']) for i in items], frames_u, conc, retry_of))
        return out

    def _unit(self, j, items, frames, conc, retry_of):
        uid = f'{self.name}-{time.strftime("%m%d%H%M%S")}-{j}' + (f'-r{retry_of}' if retry_of else '')
        outputs = [dict(key=o['key'], out_dir=o['out_dir'], frames=sorted(set(o['frames']) & set(frames)))
                   for o in self.outputs]
        outputs = [o for o in outputs if o['frames']]
        return dict(id=re.sub(r'[^A-Za-z0-9_.-]', '_', uid), job=self.name, setup=self.setup, items=items,
                    concurrency=conc, outputs=outputs, shape=list(self.shape), env={},
                    _job=self, _retry=retry_of or 0)


# ------------------------------------------------------------------ one node: boot, agent, endpoint

class Node:
    def __init__(self, name, kind):
        self.name, self.kind = name, kind
        self.url = self.tok = None
        self.cpu = KINDS[kind]['cpu']
        self.state = 'booting'
        self.unit = None           # current unit dict
        self.seq = 0               # last ready seq seen
        self.bad_polls = 0
        self.last_ok = None
        self.exists = False
        self.t_ready = None
        self.error = None
        self.region = None

    def agent(self, method, path, body=None, timeout=60, raw=False):
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(self.url.rstrip('/') + path, data=data, method=method,
                                     headers={'Authorization': 'Bearer ' + self.tok, 'Content-Type': 'application/json'})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            b = r.read()
            return b if raw else json.loads(b or b'{}')

    def boot(self, farm):
        kind = KINDS[self.kind]
        info = get_node(self.name)
        if info is None or str(info.get('status', '')).startswith('failed'):
            self._guard()
            log(f'{self.name}: creating {kind["chip"]}')
            info = api('POST', '/nodes', dict(chip=kind['chip'], name=self.name, mission=MISSION, max_wait='2h'))
        self.exists = True
        t0 = time.time()
        woke = False
        while not STOP.is_set():
            st = str(info.get('status', ''))
            if st.startswith('running'):
                break
            if st.startswith('stopped') and not woke:
                log(f'{self.name}: waking its parked disk')
                try:
                    api('POST', f'/nodes/{self.name}/commands', dict(command='true', timeout=30, max_wait='2h'), timeout=60)
                except ApiError as e:
                    if 'queued' not in e.message and 'waking' not in e.message:
                        log(f'{self.name}: wake answered {e.message[:160]}')
                woke = True
            elif st.startswith(('queued',)) and int(time.time() - t0) % 120 < 5:
                log(f'{self.name}: queued (position {info.get("queue_position") or info.get("position")}, '
                    f'est {info.get("estimated_ready_seconds")} s)')
            if 'unrestorable' in st or 'lost' in st:
                raise RuntimeError(f'{self.name} is {st}')
            time.sleep(5)
            info = get_node(self.name) or {}
        self.cpu = int((info.get('resources') or {}).get('cpu_limit') or kind['cpu'])
        self.region = info.get('region')
        self._guard()
        r = api('POST', f'/nodes/{self.name}/commands', dict(command=BOOT, timeout=280), timeout=320)
        out = (r.get('stdout') or '') + (r.get('stderr') or '')
        if r.get('exit_code') != 0 or 'BOOT OK' not in out:
            raise RuntimeError(f'{self.name}: bootstrap failed: {out[-600:]}')
        if self.kind == 'gpu' and 'bpy ok' not in out:
            raise RuntimeError(f'{self.name}: bpy does not import: {out[-400:]}')
        self._guard()
        api('POST', f'/nodes/{self.name}/files', dict(path='ld/farm_node.py', content=open(AGENT).read()))
        self._guard()
        api('POST', f'/nodes/{self.name}/commands',
            dict(command=f'cd ~/ld && exec ~/ld/venv/bin/python -u ~/ld/farm_node.py agent --port {PORT}', detach=True))
        self._guard()
        self.expose()
        for _ in range(40):
            try:
                self.agent('GET', '/status', timeout=15)
                break
            except Exception:                   # noqa: BLE001 - the agent is still starting
                time.sleep(3)
        else:
            raise RuntimeError(f'{self.name}: agent did not answer on its endpoint')
        self.state = 'ready'
        self.t_ready = time.time()
        log(f'{self.name}: ready ({self.kind}, cpu_limit {self.cpu}, {self.region}, {out.split("BOOT OK")[-1].strip()}) '
            f'in {time.time() - t0:.0f}s')

    def _guard(self):
        """Every write/exec below wakes a stopped node, so bail out once the run is over or the node was stopped."""
        if STOP.is_set() or self.state == 'stopped':
            raise RuntimeError('run interrupted during boot')

    def expose(self):
        r = {}
        for attempt in range(2):
            r = api('POST', f'/nodes/{self.name}/endpoints', dict(port=PORT, auth='bearer'))
            tok = r.get('token') or r.get('bearer_token') or r.get('auth_token')
            if not tok and isinstance(r.get('bearer'), dict):
                tok = r['bearer'].get('token')
            if r.get('url') and tok:
                self.url, self.tok = r['url'], tok
                return
            try:                                    # an older endpoint whose token we never saw: replace it
                api('DELETE', f'/nodes/{self.name}/endpoints/{PORT}')
            except ApiError:
                pass
        raise RuntimeError(f'{self.name}: expose_port gave no bearer token (fields: {sorted(r.keys())})')

    def stop(self, why=''):
        if self.state == 'stopped':
            return
        try:
            if self.url:
                self.agent('POST', '/quit', {}, timeout=10)
        except Exception:                           # noqa: BLE001
            pass
        if not self.exists:                         # never created: nothing is billing
            self.state = 'stopped'
            release(self.name)
            return
        try:
            api('POST', f'/nodes/{self.name}/stop', dict(mission=MISSION), timeout=60)
            log(f'{self.name}: stopped{" (" + why + ")" if why else ""}')
        except ApiError as e:
            log(f'{self.name}: stop failed: {e.message[:200]} -- run `farm.py stop-idle`')
        self.state = 'stopped'
        release(self.name)


# ------------------------------------------------------------------ the run

class Farm:
    def __init__(self, jobs, args):
        self.jobs, self.args = jobs, args
        self.run_id = time.strftime('%m%d-%H%M%S') + f'-{os.getpid()}'
        self.queue = []
        self.nodes = {}
        self.landed = {}           # unit id -> set((key, frame))
        self.results = []          # finished unit summaries
        self.pool = ThreadPoolExecutor(max_workers=6)
        self.t0 = time.time()
        self.total = sum(len(set(o['frames']) & j.want) for j in jobs for o in j.outputs)
        self.n_landed = 0
        self.boot_failures = 0
        self.lock = threading.Lock()

    def plan(self):
        n = self.args.nodes or 1
        weights = {j.name: len(j.want) for j in self.jobs}
        tot = sum(weights.values()) or 1
        for j in self.jobs:
            kind = 'gpu' if j.gpu else 'cpu'
            u = 1 if self.args.test is not None and not self.args.nodes else max(1, round(n * weights[j.name] / tot))
            self.queue += j.units(u, KINDS[kind]['cpu'])
        self.queue.sort(key=lambda u: -sum(len(o['frames']) for o in u['outputs']))

    def print_plan(self, verbose=False):
        for u in self.queue:
            j = u['_job']
            fr = sorted({f for o in u['outputs'] for f in o['frames']})
            log(f'unit {u["id"]} [{"gpu" if j.gpu else "cpu"}] {len(fr)} frames ({compact(fr)[:80]}), '
                f'{len(u["items"])} processes, {u["concurrency"]} at once')
            if verbose or len(self.queue) <= 4:
                for it in u['items'][:12]:
                    log(f'    {it["cmd"][:220]}')
                if len(u['items']) > 12:
                    log(f'    ... {len(u["items"]) - 12} more')
        for j in self.jobs:
            for c in j.skipped_setup:
                log(f'[{j.name}] setup step skipped (the farm image provides it): {c[:120]}')
            for o in j.outputs:
                log(f'[{j.name}] output {o["out_dir"]} -> {o["dest"]}')

    def acquire(self):
        need = {'cpu': 0, 'gpu': 0}
        for u in self.queue:
            need['gpu' if u['_job'].gpu else 'cpu'] += 1
        want_n = {k: min(v, self.args.nodes or (1 if self.args.test is not None else v)) for k, v in need.items()}
        existing = {x['name']: x for x in list_nodes()}
        held = live_leases()
        for kind, n in want_n.items():
            if not n:
                continue
            kd = KINDS[kind]
            mine = sorted((x for x in existing.values() if x['name'].startswith(kd['prefix'])),
                          key=lambda x: (0 if str(x.get('state', '')).startswith('running') else 1, x['name']))
            got = []
            for x in mine:
                st = str(x.get('state', ''))
                if len(got) >= n or 'unrestorable' in st or 'lost' in st or st.startswith('failed'):
                    continue
                if len(held) + len(got) + len(self.nodes) >= MAX_STREAM:
                    break
                if lease(x['name'], self.run_id):
                    got.append(x['name'])
            i = 1
            while len(got) < n and sum(1 for k in held if k.startswith(kd['prefix'])) + len(got) < kd['cap'] \
                    and len(held) + len(got) + len(self.nodes) < MAX_STREAM:
                name = f'{kd["prefix"]}{i:02d}'
                i += 1
                if name in existing:
                    continue
                if lease(name, self.run_id):
                    got.append(name)
            if len(got) < n:
                log(f'{kind}: {len(got)} of {n} nodes available now (caps: {kd["cap"]} {kind}, {MAX_STREAM} streaming; '
                    f'{len(held)} held by other farm runs); the rest of the work queues')
            for name in got:
                nd = Node(name, kind)
                self.nodes[name] = nd
                self.pool.submit(self._boot, nd)
        if not self.nodes:
            raise SystemExit('farm: no node is free right now (every farm node is held by another farm.py run)')

    def _boot(self, nd):
        try:
            nd.boot(self)
        except Exception as e:                      # noqa: BLE001
            nd.error = str(e)[:400]
            if not STOP.is_set():
                self.boot_failures += 1
                log(f'{nd.name}: FAILED to boot: {nd.error}')
            nd.stop('boot failed')

    # -- frames
    def fetch(self, nd, unit, entries):
        j = unit['_job']
        byname = {f'{e["key"]}/{e["name"]}': e for e in entries}
        names = list(byname)
        got = []
        for i in range(0, len(names), BATCH):
            chunk = names[i:i + BATCH]
            q = urllib.parse.urlencode({'f': ','.join(chunk)})
            data = nd.agent('GET', f'/tar/{unit["id"]}?{q}', timeout=120, raw=True)
            with tarfile.open(fileobj=io.BytesIO(data)) as tf:
                for m in tf.getmembers():
                    e = byname.get(m.name)
                    if e is None:
                        continue
                    b = tf.extractfile(m).read()
                    if hashlib.sha256(b).hexdigest() != e['sha']:
                        log(f'[{j.name}] {m.name} from {nd.name}: sha256 mismatch, will ask again')
                        continue
                    im = cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR)
                    if im is None or im.shape[:2] != j.shape:
                        log(f'[{j.name}] {m.name} from {nd.name}: bad decode {None if im is None else im.shape}')
                        continue
                    o = next(o for o in j.outputs if o['key'] == e['key'])
                    os.makedirs(o['dest'], exist_ok=True)
                    dst = os.path.join(o['dest'], e['name'])
                    tmp = dst + f'.farm{os.getpid()}.tmp'
                    with open(tmp, 'wb') as fh:
                        fh.write(b)
                    os.replace(tmp, dst)
                    if dst.endswith('.jpg') and os.path.exists(dst[:-4] + '.png'):
                        os.remove(dst[:-4] + '.png')          # the farm JPEG supersedes the older PNG
                    got.append(m.name)
                    with self.lock:
                        self.landed.setdefault(unit['id'], set()).add((e['key'], e['frame']))
                        self.n_landed += 1
        if got:
            nd.agent('POST', '/ack', dict(unit=unit['id'], files=got), timeout=30)
            fr = sorted(int(n.split('f_')[1][:5]) for n in got)
            log(f'[{j.name}] +{len(got)} ({compact(fr)[:60]}) from {nd.name} | {self.n_landed}/{self.total} landed')
        return got

    def tick(self, nd):
        """Poll one ready node: land new frames, notice a finished unit, hand out the next one."""
        try:
            st = nd.agent('GET', f'/status?since={nd.seq}', timeout=30)
            nd.bad_polls = 0
            nd.last_ok = time.time()
        except Exception as e:                      # noqa: BLE001
            nd.bad_polls += 1
            if nd.bad_polls in (3, 10, 30):
                log(f'{nd.name}: endpoint not answering ({str(e)[:120]}); {nd.bad_polls} polls')
            if time.time() - (nd.last_ok or nd.t_ready or time.time()) > 240:   # give its work to another node
                self.node_lost(nd, 'endpoint silent for 4 min')
            return
        new = st.get('ready', [])
        if new:
            nd.seq = max(nd.seq, max(e['seq'] for e in new))
            byunit = {}
            for e in new:
                byunit.setdefault(e['unit'], []).append(e)
            for uid, entries in byunit.items():
                if nd.unit and nd.unit['id'] == uid:
                    try:
                        self.fetch(nd, nd.unit, entries)
                    except Exception as e:          # noqa: BLE001
                        log(f'{nd.name}: download failed ({str(e)[:160]}); retrying')
                        nd.seq = min(x['seq'] for x in entries) - 1
                        return
        if nd.unit:
            us = next((u for u in st.get('units', []) if u['id'] == nd.unit['id']), None)
            if us and us['state'] in ('done', 'failed', 'cancelled') and not st.get('more'):
                self.finish_unit(nd, us)
        if nd.unit is None:
            if self.queue_for(nd):
                u = self.queue_for(nd)[0]
                spec = {k: v for k, v in u.items() if not k.startswith('_')}
                nd.agent('POST', '/unit', spec, timeout=30)     # raises -> the unit stays queued
                self.queue.remove(u)
                nd.unit = u
                u['_t0'] = time.time()
                fr = sum(len(o['frames']) for o in u['outputs'])
                log(f'{nd.name}: unit {u["id"]} ({fr} frames, {len(u["items"])} processes, {u["concurrency"]} at once)')
            else:
                nd.stop('no more work')

    def queue_for(self, nd):
        return [u for u in self.queue if (u['_job'].gpu) == (nd.kind == 'gpu')]

    def finish_unit(self, nd, us):
        u = nd.unit
        j = u['_job']
        want = {(o['key'], f) for o in u['outputs'] for f in o['frames']}
        got = self.landed.get(u['id'], set())
        missing = sorted(want - got)
        t = us.get('t', {})
        n = len(got)
        render_s = (t.get('end', 0) - t.get('setup_done', t.get('start', 0))) if t.get('end') else None
        self.results.append(dict(job=j.name, unit=u['id'], node=nd.name, kind=nd.kind, frames=n, state=us['state'],
                                 wall=round(time.time() - u['_t0']), setup=round(t.get('setup_done', 0) - t.get('start', 0)) if t.get('setup_done') else None,
                                 render=round(render_s) if render_s else None, commit=us.get('commit'),
                                 s_per_frame=round(render_s / n, 2) if render_s and n else None))
        msg = f'{nd.name}: unit {u["id"]} {us["state"]} at {us.get("commit")}: {n}/{len(want)} frames'
        if render_s and n:
            msg += f', {render_s / n:.2f} s/frame on the node (setup {self.results[-1]["setup"]}s)'
        log(msg)
        if us.get('error'):
            log(f'    error: {us["error"]}')
        for i, lines in (us.get('fail_tails') or {}).items():
            log(f'    process {i} tail:')
            for ln in lines[-12:]:
                log(f'      | {ln[:200]}')
        for ln in (us.get('setup_tail') or [])[-15:]:
            log(f'      setup| {ln[:200]}')
        if missing:
            frames = sorted({f for _, f in missing})
            if u['_retry'] < 1 and us['state'] != 'cancelled' and not STOP.is_set():
                log(f'[{j.name}] {len(frames)} frame(s) missing ({compact(frames)[:80]}): one retry queued')
                self.queue += j.units(1, KINDS['gpu' if j.gpu else 'cpu']['cpu'], retry_of=u['_retry'] + 1, frames=frames)
            else:
                log(f'[{j.name}] GAVE UP on {len(frames)} frame(s): {compact(frames)}')
                self.results[-1]['missing'] = compact(frames)
        nd.unit = None

    def node_lost(self, nd, why):
        log(f'{nd.name}: LOST ({why})')
        if nd.unit:
            u = nd.unit
            j = u['_job']
            want = {(o['key'], f) for o in u['outputs'] for f in o['frames']}
            missing = sorted({f for _, f in want - self.landed.get(u['id'], set())})
            if missing:
                self.queue += j.units(1, KINDS[nd.kind]['cpu'], retry_of=u['_retry'] + 1, frames=missing)
            nd.unit = None
        nd.stop(why)

    def run(self):
        spend = mission_cost()
        if spend is not None:
            if spend >= SPEND_STOP and not os.environ.get('LDFARM_SPEND_OK'):
                raise SystemExit(f'farm: mission spend ${spend:.2f} >= ${SPEND_STOP:.0f}: stopping here (tell the director; '
                                 f'LDFARM_SPEND_OK=1 overrides)')
            if spend >= SPEND_WARN:
                log(f'WARNING: farm mission spend is ${spend:.2f} (warn ${SPEND_WARN:.0f}, stop ${SPEND_STOP:.0f})')
        self.acquire()
        last_note = time.time()
        while not STOP.is_set():
            live = [nd for nd in self.nodes.values() if nd.state not in ('stopped', 'failed')]
            ready = [nd for nd in live if nd.state == 'ready']
            futures = [self.pool.submit(self.tick, nd) for nd in ready]
            for f in futures:
                try:
                    f.result()
                except Exception as e:              # noqa: BLE001
                    log(f'poll error: {str(e)[:200]}')
            busy = [nd for nd in self.nodes.values() if nd.unit is not None]
            booting = [nd for nd in live if nd.state == 'booting']
            if not self.queue and not busy and not booting:
                break
            if self.queue and not live:
                if self.boot_failures >= 3:
                    log('nodes keep failing to boot; giving up. Units left: ' + ', '.join(u['id'] for u in self.queue))
                    break
                self.acquire()
            if time.time() - last_note > 120:
                last_note = time.time()
                log(f'-- {self.n_landed}/{self.total} landed, {len(busy)} node(s) rendering, {len(booting)} booting, '
                    f'{len(self.queue)} unit(s) queued, {(time.time() - self.t0) / 60:.1f} min')
            time.sleep(3)
        self.shutdown()

    def shutdown(self):
        for nd in list(self.nodes.values()):
            if nd.state not in ('stopped',):
                nd.stop('run over' if not STOP.is_set() else 'interrupted')
        spend = mission_cost()
        log(f'RUN {self.run_id} {"INTERRUPTED" if STOP.is_set() else "DONE"}: {self.n_landed}/{self.total} frames landed '
            f'in {(time.time() - self.t0) / 60:.1f} min' + (f'; farm mission spend ${spend:.2f}' if spend is not None else ''))
        for r in self.results:
            log(f'    {r["job"]:<22} {r["node"]} {r["kind"]} {r["frames"]:>4} frames  setup {r["setup"]}s  render {r["render"]}s '
                f' {r["s_per_frame"]} s/frame  wall {r["wall"]}s  {r["state"]}' + (f'  MISSING {r["missing"]}' if r.get('missing') else ''))
        os.makedirs(CACHE, exist_ok=True)
        with open(os.path.join(CACHE, 'runs.jsonl'), 'a') as fh:
            fh.write(json.dumps(dict(run=self.run_id, jobs=[j.name for j in self.jobs], landed=self.n_landed,
                                     total=self.total, results=self.results, t=time.time())) + '\n')
        for j in self.jobs:
            for o in j.outputs:
                log(f'[{j.name}] frames in {o["dest"]}')


# ------------------------------------------------------------------ commands

def cmd_status():
    held = live_leases()
    meta = _meta()
    try:
        items = list_nodes()
    except ApiError as e:
        sys.exit(f'farm: {e}')
    print(f'workspace {WORKSPACE}, mission {MISSION}, token expires {meta.get("expires_at", "?")}')
    spend = mission_cost()
    if spend is not None:
        print(f'farm mission spend ${spend:.2f} (warn ${SPEND_WARN:.0f}, refuse ${SPEND_STOP:.0f}, workspace cap $230)')
    for x in sorted(items, key=lambda x: x['name']):
        h = held.get(x['name'])
        print(f'  {x["name"]:<10} {x.get("chip", "?"):<7} {str(x.get("state", "?")):<34} '
              f'{("held by farm.py pid " + str(h["pid"]) + " run " + h["run"]) if h else ""}')
    if not items:
        print('  (no farm nodes yet)')


def cmd_stop_idle():
    held = live_leases()
    for x in list_nodes():
        st = str(x.get('state', ''))
        if x['name'] in held or not st.startswith(('running', 'provisioning', 'queued', 'waking')):
            continue
        try:
            api('POST', f'/nodes/{x["name"]}/stop', dict(mission=MISSION))
            print(f'stopped {x["name"]} (was {st})')
        except ApiError as e:
            print(f'{x["name"]}: {e.message[:200]}')


def main():
    if len(sys.argv) > 1 and sys.argv[1] == 'status':
        return cmd_status()
    if len(sys.argv) > 1 and sys.argv[1] == 'stop-idle':
        return cmd_stop_idle()
    ap = argparse.ArgumentParser(description='THE LONG DAWN render farm', usage=__doc__)
    ap.add_argument('jobs', nargs='+')
    ap.add_argument('--nodes', type=int, default=None)
    ap.add_argument('--gpu', action='store_true')
    ap.add_argument('--frames', default=None)
    ap.add_argument('--test', type=int, nargs='?', const=0, default=None)
    ap.add_argument('--local-out', default=None)
    ap.add_argument('--missing', action='store_true')
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()
    args.nodes_retry = True
    jobs = []
    for p in args.jobs:
        if not os.path.exists(p) and os.path.exists(os.path.join(ROOT, 'cloud', 'jobs', p)):
            p = os.path.join(ROOT, 'cloud', 'jobs', p)
        jobs.append(Job(p, args))
    for j in jobs:
        if not j.want:
            log(f'[{j.name}] nothing to render (all wanted frames already here?)')
    jobs = [j for j in jobs if j.want]
    if not jobs:
        return
    farm = Farm(jobs, args)
    farm.plan()
    log(f'plan: {len(farm.queue)} unit(s), {farm.total} frames, jobs: ' + ', '.join(
        f'{j.name}{" (gpu)" if j.gpu else ""}' for j in jobs))
    farm.print_plan(verbose=args.dry_run)
    if args.dry_run:
        return

    def on_signal(sig, frame):
        if STOP.is_set():
            os._exit(1)
        log('interrupt: stopping the farm nodes (press again to abandon)')
        STOP.set()
    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)
    try:
        farm.run()
    finally:
        for nd in farm.nodes.values():
            release(nd.name)


if __name__ == '__main__':
    main()
