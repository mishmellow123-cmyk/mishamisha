#!/usr/bin/env python3
"""THE LONG DAWN render farm: run cloud/jobs/*.json on Autoresearch nodes; the frames stream back to this Mac.

    python3 the-long-dawn/cloud/farm.py the-long-dawn/cloud/jobs/<job>.json [more.json ...]
            [--nodes N] [--gpu] [--frames A-B[,C-D]] [--test [K]] [--local-out DIR] [--missing] [--dry-run]
            [--detach] [--direct]
    python3 the-long-dawn/cloud/farm.py status          # the queue, the farm nodes, spend, token expiry
    python3 the-long-dawn/cloud/farm.py cancel <request>
    python3 the-long-dawn/cloud/farm.py stop-idle       # stop every farm node that nothing is using

One shared queue: every farm.py call is a REQUEST to a single farm daemon on this Mac (started on demand, gone
after 5 idle minutes). The daemon drives the whole node pool: --test requests go ahead of finals, each request
uses at most --nodes N nodes at once, finals leave 2 slots free for look-dev, and a node that finishes a unit
takes the next one from any request (else it stops at once). The calling farm.py just follows its request's
log and exits when the request is done (exit 0), incomplete/failed (1) or cancelled (2). Ctrl-C (or killing
it) cancels the request. --detach submits and returns. --direct runs in-process without the daemon.

What a request does:
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
  * cost: a node stops the moment no queued unit wants it (and stops itself after 10 idle minutes if the
    daemon dies). Requests are refused past LDFARM_SPEND_STOP (default $200) of farm spend.

Modes:
  --test [K]   look-dev: K frames (default 4) spread over the job (or over --frames), on one node unless
               --nodes; lands in renders/_farmtest/<job>/ unless --local-out, so a test never overwrites a
               finished render.
  --missing    render only the frames this Mac doesn't already have.
  --dry-run    print the plan (units and rewritten commands) and touch nothing.

Files: ~/.cache/ldfarm/ (inbox/, req/<id>.log + .json, daemon.log, leases/). Credential:
~/.config/longdawn-farm/token (chmod 600; a workspace-scoped infra token), or $LDFARM_TOKEN.
Limits (env): LDFARM_MAX_CPU (30), LDFARM_MAX_GPU (2), LDFARM_MAX_ENDPOINTS (15: one per streaming node),
LDFARM_TEST_RESERVE (2).
"""
import argparse
import fcntl
import hashlib
import io
import itertools
import json
import math
import os
import random
import re
import signal
import subprocess
import sys
import tarfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor

VENV_PY = os.path.expanduser('~/.venvs/longdawn/bin/python')
# cv2/numpy are imported only where frames land, so a run that is waiting for a node stays small on this Mac

HERE = os.path.dirname(os.path.abspath(__file__))                 # the-long-dawn/cloud
ROOT = os.path.dirname(HERE)                                      # the-long-dawn
AGENT = os.path.join(HERE, 'farm_node.py')
API = os.environ.get('LDFARM_API', 'https://autoresearch.sfcompute.com/preview')
CFG = os.path.expanduser('~/.config/longdawn-farm')
CACHE = os.path.expanduser('~/.cache/ldfarm')
LEASES = os.path.join(CACHE, 'leases')
INBOX = os.path.join(CACHE, 'inbox')
REQDIR = os.path.join(CACHE, 'req')
DONEDIR = os.path.join(CACHE, 'done')
DLOCK = os.path.join(CACHE, 'daemon.lock')
DLOG = os.path.join(CACHE, 'daemon.log')
DPID = os.path.join(CACHE, 'daemon.pid')
DAEMON_IDLE_EXIT = int(os.environ.get('LDFARM_DAEMON_IDLE_EXIT', '300'))
MISSION = os.environ.get('LDFARM_MISSION', 'long-dawn-render-farm')
PORT = 8700
MAX_CPU = int(os.environ.get('LDFARM_MAX_CPU', '30'))
MAX_GPU = int(os.environ.get('LDFARM_MAX_GPU', '2'))
SSH_GPU_AFTER_S = int(os.environ.get('LDFARM_SSH_GPU_AFTER_S', '180'))  # M4 Metal takes Cycles units after this wait
FINAL_SHARE = float(os.environ.get('LDFARM_FINAL_SHARE', '0.34'))   # running finals keep ~1 node in 3
FINAL_AGING_S = int(os.environ.get('LDFARM_FINAL_AGING_S', '300'))   # finals never wait longer than this behind tests
MAX_QUEUED = int(os.environ.get('LDFARM_MAX_QUEUED', '4'))      # nodes per pool waiting in a capacity queue
MAX_BOOTING = int(os.environ.get('LDFARM_MAX_BOOTING', '4'))  # nodes creating/queued/waking at once
MAX_SPILL = int(os.environ.get('LDFARM_MAX_SPILL', '10'))    # extra h100-1 nodes that take CPU units while cpu-8 has no room
MAX_STREAM = int(os.environ.get('LDFARM_MAX_ENDPOINTS', '15'))
TEST_RESERVE = int(os.environ.get('LDFARM_TEST_RESERVE', '2'))  # slots finals leave free so look-dev never waits long
SPEND_WARN = float(os.environ.get('LDFARM_SPEND_WARN', '150'))
SPEND_STOP = float(os.environ.get('LDFARM_SPEND_STOP', '200'))
KINDS = {'cpu': dict(chip='cpu-8', prefix='ldf-c', cpu=8, cap=MAX_CPU),
         'gpu': dict(chip='h100-1', prefix='ldf-g', cpu=14, cap=MAX_GPU),
         'ssh': dict(chip='ssh', prefix='ldf-m', cpu=8, cap=0)}      # our own machines over ssh (ssh_nodes.json)
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

SSH_BOOT = r'''set -e
LDF=$HOME/ldfarm
mkdir -p "$LDF/bin" "$LDF/config" "$LDF/runs"
# everything uv writes stays under ~/ldfarm: binary, managed pythons, caches, receipt; no shell profile is touched
export UV_INSTALL_DIR="$LDF/bin" INSTALLER_NO_MODIFY_PATH=1 UV_NO_MODIFY_PATH=1 XDG_CONFIG_HOME="$LDF/config" \
       XDG_CACHE_HOME="$LDF/cache" XDG_DATA_HOME="$LDF/share" UV_CACHE_DIR="$LDF/uv-cache" \
       UV_PYTHON_INSTALL_DIR="$LDF/uv-python" UV_PYTHON_BIN_DIR="$LDF/uv-python-bin" \
       UV_TOOL_DIR="$LDF/uv-tools" UV_TOOL_BIN_DIR="$LDF/uv-tools-bin" UV_NO_CONFIG=1
export PATH="$LDF/bin:$PATH"
pkill -f "ldfarm/farm_node[.]py agent" || true
[ -x "$LDF/bin/uv" ] || (curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1)
cd "$LDF"
[ -d mishamisha/.git ] || git clone -q --single-branch --branch claude/long-dawn-v2 --depth 1 --filter=blob:none https://github.com/mishmellow123-cmyk/mishamisha
git -C mishamisha fetch -q --depth 1 origin claude/long-dawn-v2 && git -C mishamisha reset -q --hard FETCH_HEAD
[ -x venv/bin/python ] || uv venv -q --seed --python 3.12 venv
venv/bin/python -c "import numba, numpy, scipy, cv2, PIL, fontTools, pip" 2>/dev/null || uv pip install -q --python venv/bin/python pip numba==0.67.0 llvmlite==0.49.0 numpy==2.5.3 scipy==1.18.1 opencv-python-headless==4.10.0.84 pillow==12.3.0 fonttools==4.66.0
if [ -n "$LDFARM_WANT_BPY" ]; then
  [ -x bpyenv/bin/python ] || uv venv -q --python 3.11 bpyenv
  [ -d bpyenv/lib/python3.11/site-packages/bpy ] || uv pip install -q --python bpyenv/bin/python bpy==4.5.14
  bpyenv/bin/python -c "import bpy" >/dev/null 2>&1 && echo "bpy ok"
fi
echo "BOOT OK $(git -C mishamisha rev-parse --short HEAD)"
'''
SSH_CFG = os.path.join(CFG, 'ssh_nodes.json')


def ssh_specs():
    try:
        return json.load(open(SSH_CFG))
    except (OSError, ValueError):
        return []


STOP = threading.Event()
UNIT_SERIAL = itertools.count(1)
PRINT = threading.Lock()


def log(msg):
    with PRINT:
        print(time.strftime('%H:%M:%S') + ' ' + str(msg), flush=True)


# ------------------------------------------------------------------ platform API (the token never leaves here)

class ApiError(Exception):
    def __init__(self, status, code, message):
        super().__init__(f'{status} {code}: {message}')
        self.status, self.code, self.message = status, code, message or ''


class Pool:
    """One Autoresearch account/workspace the farm can rent from: its own token, endpoint cap, create queue, rate
    limit, credit and refusal threshold. Node names carry the pool's prefix, so leases stay unique across pools."""

    def __init__(self, name, token_file, field, workspace, mission, spend_stop, prefix, max_stream, cap_note=''):
        self.name, self.token_file, self.field, self.workspace = name, token_file, field, workspace
        self.mission, self.spend_stop, self.prefix, self.max_stream = mission, spend_stop, prefix, max_stream
        self.cap_note = cap_note
        self._tok = None
        self._spend = (0, None)
        self.mission_ok = False

    def token(self):
        if self._tok is None:
            raw = open(self.token_file).read()
            self._tok = (json.loads(raw)[self.field] if self.field else raw).strip()
        return self._tok

    def meta(self):
        try:
            if self.field:
                d = json.load(open(self.token_file))
                return {'expires_at': d.get('expires_at'), 'workspace': self.workspace}
            return json.load(open(os.path.join(os.path.dirname(self.token_file), 'token.meta.json')))
        except (OSError, ValueError):
            return {}

    def api(self, method, path, body=None, params=None, timeout=90, tries=14):
        params = dict(params or {})
        params['workspace'] = self.workspace
        url = API + path + '?' + urllib.parse.urlencode(params)
        data = json.dumps(body).encode() if body is not None else None
        last = None
        for attempt in range(tries):
            req = urllib.request.Request(url, data=data, method=method, headers={
                'Authorization': 'Bearer ' + self.token(), 'Content-Type': 'application/json'})
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
                    raise ApiError(401, 'unauthenticated', f'pool {self.name}: the credential was refused (expired?)')
                if e.code in (429, 503) or e.code >= 500:
                    last = ApiError(e.code, err.get('code'), err.get('message'))
                    wait = float(e.headers.get('Retry-After') or (2 + 3 * attempt))
                    time.sleep(min(max(wait, 2.0) + random.random() * 3, 60))
                    continue
                raise ApiError(e.code, err.get('code'), err.get('message'))
            except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
                last = ApiError(0, 'network', str(e)[:200])
                time.sleep(2 + 3 * attempt)
        raise last or ApiError(0, 'unreachable', path)

    def get_node(self, name):
        try:
            return self.api('GET', f'/nodes/{name}')
        except ApiError as e:
            if e.status == 404 or 'no node named' in e.message:
                return None
            raise

    def list_nodes(self):
        return self.api('GET', '/nodes').get('items', [])

    def ensure_mission(self):
        if not self.mission_ok:
            try:
                self.api('POST', '/missions', dict(name=self.mission, title='THE LONG DAWN render farm'))
                self.mission_ok = True
            except ApiError as e:
                log(f'pool {self.name}: mission {self.mission}: {e.message[:120]}')

    def mission_cost(self):
        try:
            m = self.api('GET', f'/missions/{self.mission}')
        except ApiError:
            return None
        v = (m.get('cost') or {}).get('total_usd')
        return float(v) if isinstance(v, (int, float)) else None

    def spend(self):
        t, v = self._spend
        if time.time() - t > 60:
            v = self.mission_cost()
            self._spend = (time.time(), v)
        return v

    def over_budget(self):
        v = self.spend()
        return v is not None and v >= self.spend_stop and not os.environ.get('LDFARM_SPEND_OK')

    def pfx(self, kind):
        return self.prefix + ('g' if kind == 'gpu' else 'c')

    def owns(self, name):
        return name.startswith((self.prefix + 'c', self.prefix + 'g'))


def load_pools():
    pools = [Pool('p1', os.path.join(CFG, 'token'), None, os.environ.get('LDFARM_WORKSPACE', 'long-dawn-farm'),
                  MISSION, SPEND_STOP, 'ldf-', MAX_STREAM, 'workspace cap $230')]
    p2 = os.path.expanduser('~/.config/longdawn-farm2/token.json')
    if os.path.exists(p2) and not os.environ.get('LDFARM_NO_POOL2'):
        pools.append(Pool('p2', p2, 'token', 'default', MISSION, float(os.environ.get('LDFARM_SPEND_STOP_P2', '450')),
                          'lf2-', int(os.environ.get('LDFARM_MAX_ENDPOINTS_P2', '15')), 'org byeron98: default cap $1,060'))
    return pools


POOLS = []


def api(*a, **k):
    return POOLS[0].api(*a, **k)


def get_node(name):
    return pool_of(name).get_node(name)


def list_nodes():
    return [x for p in POOLS for x in p.list_nodes()]


def mission_cost():
    return POOLS[0].mission_cost()


def pool_of(name):
    for p in POOLS:
        if p.owns(name):
            return p
    return POOLS[0]


# ------------------------------------------------------------------ leases: concurrent farm.py runs never share a node

_ALIVE = {}


def _alive(pid):
    """True for a running process; a zombie (exited, not yet reaped) counts as dead. Cached for 5 s."""
    hit = _ALIVE.get(pid)
    if hit and time.time() - hit[0] < 5:
        return hit[1]
    try:
        os.kill(pid, 0)
        st = subprocess.run(['ps', '-o', 'stat=', '-p', str(pid)], capture_output=True, text=True).stdout.strip()
        ok = bool(st) and not st.startswith('Z')
    except OSError:
        ok = False
    _ALIVE[pid] = (time.time(), ok)
    return ok


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
    def __init__(self, path, args, say=None):
        self.say = say or log
        self.tag = ''
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
            self.say(f'[{self.name}] note: its frame arguments are not rewritable, so it runs whole (all frames)')
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
            out.append(self._unit(j, [dict(cmd=i['cmd']) for i in items], frames_u, conc, retry_of, d=d_avg))
        return out

    def _unit(self, j, items, frames, conc, retry_of, d=2):
        uid = f'{self.tag}-{self.name}-{j}' + (f'-r{retry_of}.{next(UNIT_SERIAL)}' if retry_of else '')
        outputs = [dict(key=o['key'], out_dir=o['out_dir'], frames=sorted(set(o['frames']) & set(frames)))
                   for o in self.outputs]
        outputs = [o for o in outputs if o['frames']]
        return dict(id=re.sub(r'[^A-Za-z0-9_.-]', '_', uid), job=self.name, setup=self.setup, items=items,
                    concurrency=conc, outputs=outputs, shape=list(self.shape), env={},
                    _job=self, _retry=retry_of or 0, _d=max(1, round(d)))


# ------------------------------------------------------------------ one node: boot, agent, endpoint

class Interrupted(RuntimeError):
    """The scheduler stopped this node during its boot on purpose (run over, or trimmed from a capacity queue)."""


class CapacityError(RuntimeError):
    """The platform has no room for this shape right now (a failed create or wake)."""


class Node:
    def __init__(self, name, kind, sched, pool=None):
        self.name, self.kind, self.sched = name, kind, sched
        self.pool = pool or (pool_of(name) if kind != 'ssh' else None)
        self.url = self.tok = None
        self.cpu = KINDS[kind]['cpu']
        self.state = 'booting'
        self.unit = None           # current unit dict
        self.seq = 0               # last ready seq seen
        self.bad_polls = 0
        self.ticking = False
        self.last_ok = None
        self.exists = False
        self.t_ready = None
        self.error = None
        self.region = None
        self.last_job = None
        self.queued = False
        self.t_queued = None
        self.created_new = False
        self.req = None            # the request whose work made us start this node (boot news goes to its log)

    def say(self, msg):
        if self.req is not None:
            self.req.log(msg)
        else:
            log(msg)

    def agent(self, method, path, body=None, timeout=60, raw=False):
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(self.url.rstrip('/') + path, data=data, method=method,
                                     headers={'Authorization': 'Bearer ' + self.tok, 'Content-Type': 'application/json'})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            b = r.read()
            return b if raw else json.loads(b or b'{}')

    def _guard(self):
        """Every write/exec below wakes a stopped node, so bail out once the run is over or the node was stopped."""
        if STOP.is_set() or self.state == 'stopped':
            raise Interrupted('interrupted during boot')

    def boot(self):
        kind = KINDS[self.kind]
        pool = self.pool
        pool.ensure_mission()
        info = pool.get_node(self.name)
        if info is None or str(info.get('status', '')).startswith('failed'):
            self._guard()
            self.say(f'{self.name}: creating {kind["chip"]}')
            self.created_new = True
            info = pool.api('POST', '/nodes', dict(chip=kind['chip'], name=self.name, mission=pool.mission, max_wait='2h'))
        self.exists = True                          # from here on, any failure must stop the node (never leak one)
        t0 = time.time()
        woke = None
        last_q = 0
        while True:
            self._guard()
            st = str(info.get('status', ''))
            if st.startswith('running'):
                break
            if st.startswith('failed'):
                raise CapacityError(f'create failed: {str(info.get("error", ""))[:160]}')
            if st.startswith('stopped'):
                stp = info.get('stopped') or {}
                if woke is None:
                    self.say(f'{self.name}: waking its parked disk')
                    woke = time.time()
                    try:
                        pool.api('POST', f'/nodes/{self.name}/commands', dict(command='true', timeout=30, max_wait='2h'), timeout=60)
                    except ApiError as e:
                        if 'queued' not in e.message and 'waking' not in e.message:
                            self.say(f'{self.name}: wake answered {e.message[:160]}')
                elif time.time() - woke > 20 and stp.get('reason') == 'wake_failed':
                    raise CapacityError(f'wake failed: {str((info.get("last_failure") or {}).get("error", ""))[:160]}')
                elif time.time() - woke > 240:
                    raise CapacityError('the wake never started')
            self.queued = st.startswith('queued')
            if self.queued and not self.t_queued:
                self.t_queued = time.time()
            if self.queued and time.time() - last_q > 120:
                last_q = time.time()
                self.say(f'{self.name}: queued for capacity (est {info.get("estimated_ready_seconds")} s); its place '
                         f'is kept (queued demand is what starts machines)')
            if 'unrestorable' in st or 'lost' in st:
                raise RuntimeError(f'{self.name} is {st}')
            for _ in range(4 if self.queued else 1):   # a queue moves in minutes: poll it every 20 s
                self._guard()
                time.sleep(5)
            info = pool.get_node(self.name) or {}
        self.queued = False
        self.cpu = int((info.get('resources') or {}).get('cpu_limit') or kind['cpu'])
        self.region = info.get('region')
        self._guard()
        r = pool.api('POST', f'/nodes/{self.name}/commands', dict(command=BOOT, timeout=280), timeout=320)
        out = (r.get('stdout') or '') + (r.get('stderr') or '')
        if r.get('exit_code') != 0 or 'BOOT OK' not in out:
            raise RuntimeError(f'{self.name}: bootstrap failed (exit {r.get("exit_code")}, fields '
                               f'{sorted(k for k in r if k not in ("stdout", "stderr"))}): {out[-500:]}')
        if self.kind == 'gpu' and 'bpy ok' not in out:
            raise RuntimeError(f'{self.name}: bpy does not import: {out[-400:]}')
        self._guard()
        pool.api('POST', f'/nodes/{self.name}/files', dict(path='ld/farm_node.py', content=open(AGENT).read()))
        self._guard()
        pool.api('POST', f'/nodes/{self.name}/commands',
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
        self.t_ready = self.last_ok = time.time()
        self.state = 'ready'
        self.say(f'{self.name}: ready ({self.kind}, cpu_limit {self.cpu}, {self.region}, '
                 f'{out.split("BOOT OK")[-1].strip()}) in {time.time() - t0:.0f}s')

    def expose(self):
        r = {}
        for attempt in range(2):
            r = self.pool.api('POST', f'/nodes/{self.name}/endpoints', dict(port=PORT, auth='bearer'))
            tok = r.get('token') or r.get('bearer_token') or r.get('auth_token')
            if not tok and isinstance(r.get('bearer'), dict):
                tok = r['bearer'].get('token')
            if r.get('url') and tok:
                self.url, self.tok = r['url'], tok
                return
            try:                                    # an older endpoint whose token we never saw: replace it
                self.pool.api('DELETE', f'/nodes/{self.name}/endpoints/{PORT}')
            except ApiError:
                pass
        raise RuntimeError(f'{self.name}: expose_port gave no bearer token (fields: {sorted(r.keys())})')

    def stop(self, why='', api_stop=True):
        if self.state == 'stopped':
            return
        self.state = 'stopped'
        if not api_stop:
            self.exists = False
        try:
            if self.url:
                self.agent('POST', '/quit', {}, timeout=10)
        except Exception:                           # noqa: BLE001
            pass
        if self.exists:
            try:
                self.pool.api('POST', f'/nodes/{self.name}/stop', dict(mission=self.pool.mission), timeout=60)
                log(f'{self.name}: stopped{" (" + why + ")" if why else ""}')
            except ApiError as e:
                log(f'{self.name}: stop failed: {e.message[:200]} -- run `farm.py stop-idle`')
        release(self.name)


class SshNode(Node):
    """A machine of ours reached over ssh (the M4): same agent, bound to 127.0.0.1 there, reached through an ssh
    tunnel from this Mac. It lives in ~/<home> on that machine and nowhere else; caffeinate keeps it awake while
    the agent runs, and the agent exits after 10 idle minutes. Free, so it takes CPU units before any cloud node."""

    def __init__(self, spec, sched):
        super().__init__(spec['name'], 'ssh', sched)
        self.host = spec['host']
        self.home = spec.get('home', 'ldfarm')
        self.cpu = int(spec.get('cpu', 8))
        self.lport = int(spec.get('local_port', 18701))
        self.bpy = bool(spec.get('bpy'))
        self.tunnel = None
        self.exists = True
        self.checked = 0
        self.ok = True

    def ssh(self, cmd, stdin=None, timeout=900):
        return subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', '-o', 'ServerAliveInterval=15',
                               self.host, cmd], input=stdin, capture_output=True, text=True, timeout=timeout)

    def ok_to_work(self):
        """Someone's own laptop: work only on AC power, and only when its owner isn't running heavy jobs of their
        own (1-min load, measured while we have nothing running there, under 40% of its cores)."""
        if time.time() - self.checked < 120:
            return self.ok
        self.checked = time.time()
        try:
            r = self.ssh("pmset -g batt | head -1; sysctl -n vm.loadavg hw.ncpu", timeout=20)
            out = r.stdout or ''
            ac = 'AC Power' in out
            nums = out.split('{')[-1].split('}')
            load1 = float(nums[0].split()[0]) if len(nums) > 1 else 99.0
            ncpu = int(nums[-1].split()[0]) if len(nums) > 1 else 10
            busy_owner = self.unit is None and load1 > 0.4 * ncpu
            self.ok = ac and not busy_owner
            if not self.ok:
                self.say(f'{self.name}: stepping aside ({"on battery" if not ac else ""}'
                         f'{" and " if not ac and busy_owner else ""}{"its owner is busy, load %.0f" % load1 if busy_owner else ""})')
        except Exception:                           # noqa: BLE001
            self.ok = False
        return self.ok

    def open_tunnel(self):
        if self.tunnel and self.tunnel.poll() is None:
            return
        self.tunnel = subprocess.Popen(['ssh', '-N', '-o', 'BatchMode=yes', '-o', 'ExitOnForwardFailure=yes',
                                        '-o', 'ServerAliveInterval=15', '-o', 'ServerAliveCountMax=3',
                                        '-L', f'127.0.0.1:{self.lport}:127.0.0.1:{PORT}', self.host],
                                       stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                       start_new_session=True)
        time.sleep(1.5)

    def agent(self, method, path, body=None, timeout=60, raw=False):
        self.open_tunnel()
        return super().agent(method, path, body, timeout, raw)

    def boot(self):
        t0 = time.time()
        self._guard()
        self.say(f'{self.name}: preparing {self.host} over ssh')
        r = self.ssh(('LDFARM_WANT_BPY=1 ' if self.bpy else '') + '/bin/bash -s', stdin=SSH_BOOT, timeout=1500)
        out = (r.stdout or '') + (r.stderr or '')
        if r.returncode != 0 or 'BOOT OK' not in out:
            raise RuntimeError(f'{self.name}: ssh bootstrap failed: {out[-400:]}')
        self._guard()
        r = self.ssh(f'cat > ~/{self.home}/farm_node.py', stdin=open(AGENT).read(), timeout=60)
        if r.returncode != 0:
            raise RuntimeError(f'{self.name}: could not copy the agent: {(r.stderr or "")[-200:]}')
        env = (f'LDFARM_HOME=$HOME/{self.home} LDFARM_SSH=1 LDFARM_BIND=127.0.0.1 LDFARM_CPU_LIMIT={self.cpu} '
               f'LDFARM_BPYENV=$HOME/{self.home}/bpyenv LDFARM_UNIT_HOME=$HOME/{self.home}/home NUMBA_NUM_THREADS={self.cpu}')
        self.ssh(f'cd ~/{self.home} && {env} nohup caffeinate -dimsu venv/bin/python -u farm_node.py agent '
                 f'--port {PORT} > agent.log 2>&1 < /dev/null &', timeout=60)
        self.url, self.tok = f'http://127.0.0.1:{self.lport}', 'ssh-tunnel'
        for _ in range(40):
            try:
                self.agent('GET', '/status', timeout=10)
                break
            except Exception:                   # noqa: BLE001 - agent or tunnel still starting
                if self.tunnel and self.tunnel.poll() is not None:
                    self.tunnel = None
                time.sleep(3)
        else:
            raise RuntimeError(f'{self.name}: agent did not answer through the ssh tunnel')
        self.t_ready = self.last_ok = time.time()
        self.state = 'ready'
        self.say(f'{self.name}: ready ({self.host}, {self.cpu} threads, {out.split("BOOT OK")[-1].strip()}) '
                 f'in {time.time() - t0:.0f}s')

    def stop(self, why='', api_stop=True):
        if self.state == 'stopped':
            return
        self.state = 'stopped'
        try:
            if self.url:
                Node.agent(self, 'POST', '/quit', {}, timeout=10)
        except Exception:                           # noqa: BLE001
            pass
        if self.tunnel and self.tunnel.poll() is None:
            self.tunnel.terminate()
        log(f'{self.name}: released{" (" + why + ")" if why else ""}')
        release(self.name)


# ------------------------------------------------------------------ requests: one per farm.py call

TERMINAL = ('done', 'incomplete', 'failed', 'cancelled')


def _paths(rid):
    return (os.path.join(REQDIR, rid + '.log'), os.path.join(REQDIR, rid + '.json'), os.path.join(REQDIR, rid + '.cancel'))


class Request:
    def __init__(self, rid, argv, submitted, direct=False):
        self.id, self.argv, self.submitted, self.direct = rid, argv, submitted, direct
        self.args = parse_run_args(argv)
        self.short = hashlib.sha1(rid.encode()).hexdigest()[:6]
        self.jobs = []
        self.state = 'queued'
        self.units_total = self.units_open = 0
        self.landed = self.total = 0
        self.landed_set = set()     # (job, output key, frame) this request itself landed
        self.results = []
        self.gave_up = []
        self.cancelled = False
        self.t_start = self.t_end = None
        self.prio = 0 if self.args.test is not None else 1
        self.budget = 1
        self.note = ''
        self.logp, self.statep, self.cancelp = _paths(rid)
        self.logf = None if direct else open(self.logp, 'a')

    def log(self, msg):
        line = time.strftime('%H:%M:%S') + ' ' + str(msg)
        if self.direct:
            with PRINT:
                print(line, flush=True)
        else:
            self.logf.write(line + '\n')
            self.logf.flush()
            with PRINT:
                print(f'[{self.id}] {msg}', flush=True)

    def save(self):
        if self.direct:
            return
        d = dict(id=self.id, state=self.state, kind='test' if self.prio == 0 else 'final', note=self.note,
                 jobs=[j.name for j in self.jobs] or [os.path.basename(p) for p in self.args.jobs],
                 landed=self.landed, total=self.total, units_total=self.units_total, units_open=self.units_open,
                 budget=self.budget, submitted=self.submitted, t_start=self.t_start, t_end=self.t_end,
                 gave_up=self.gave_up, dests=sorted({o['dest'] for j in self.jobs for o in j.outputs}),
                 landed_frames=sorted(self.landed_set))
        tmp = self.statep + '.tmp'
        with open(tmp, 'w') as fh:
            json.dump(d, fh)
        os.replace(tmp, self.statep)

    def finish(self, state, note=''):
        self.state, self.note, self.t_end = state, note, time.time()
        took = (self.t_end - (self.t_start or self.submitted)) / 60
        self.log(f'REQUEST {state.upper()}: {self.landed}/{self.total} frames landed in {took:.1f} min'
                 + (f' ({note})' if note else ''))
        for r in self.results:
            self.log(f'    {r["job"]:<22} {r["node"]} {r["kind"]} {r["frames"]:>4} frames  setup {r["setup"]}s  '
                     f'render {r["render"]}s  {r["s_per_frame"]} s/frame  {r["state"]}'
                     + (f'  MISSING {r["missing"]}' if r.get('missing') else ''))
        for j in self.jobs:
            for o in j.outputs:
                if set(o['frames']) & j.want:
                    self.log(f'[{j.name}] frames in {o["dest"]}')
        self.save()
        if not self.direct:
            try:
                os.makedirs(DONEDIR, exist_ok=True)
                os.replace(os.path.join(INBOX, self.id + '.json'), os.path.join(DONEDIR, self.id + '.json'))
            except OSError:
                pass
            with open(os.path.join(CACHE, 'runs.jsonl'), 'a') as fh:
                fh.write(json.dumps(dict(req=self.id, state=state, landed=self.landed, total=self.total,
                                         results=self.results, t=time.time())) + '\n')
            self.logf.close()


# ------------------------------------------------------------------ the scheduler (the daemon, or one --direct run)

def ukind(u):
    return 'gpu' if u['_job'].gpu else 'cpu'


class Scheduler:
    def __init__(self):
        self.reqs = {}
        self.queue = []
        self.nodes = {}
        self.landed = {}           # unit id -> set((key, frame))
        self.lock = threading.RLock()
        self.pool = ThreadPoolExecutor(max_workers=12)       # polls/downloads: short
        self.boot_pool = ThreadPoolExecutor(max_workers=64)  # boots: may wait in a capacity queue for many minutes
        self.seq = 0
        self.boot_failures = 0     # consecutive
        self.last_grow = 0
        self.waiting_noted = False
        self.last_spend = (0, None)
        self.backoff = {}           # kind -> (retry_at, delay) after a capacity failure
        self.cpu_blocked_until = 0
        self.known = set()
        self.budget_noted = {}
        self.last_sweep = 0
        self.last_spend_post = 0
        self.last_hourly = time.time() - 3000
        self.alerted = {}
        self.bad_names = {}          # node name -> skip until (a node that failed its bootstrap)
        self.last_report = 0
        self.unit_secs = []          # recent unit wall times, for ETAs

    # -- intake
    def add(self, req):
        try:
            for p in req.args.jobs:
                j = Job(p, req.args, say=req.log)
                j.tag = req.short
                req.jobs.append(j)
        except Exception as e:                      # noqa: BLE001
            self.reqs[req.id] = req
            return req.finish('failed', f'bad job file: {e}')
        self.reqs[req.id] = req
        resumed = 0
        try:                                        # a restarted daemon resumes: skip only what THIS request landed
            prev = json.load(open(req.statep)).get('landed_frames', [])
        except (OSError, ValueError):
            prev = []
        mine = {(jn, int(k), int(f)) for jn, k, f in prev}
        req.landed_set |= mine
        req.landed = len(req.landed_set)
        for j in req.jobs:
            done_here = {f for f in j.want
                         if all((j.name, o['key'], f) in mine for o in j.outputs if f in o['frames'])}
            if done_here:
                j.want -= done_here
                resumed += len(done_here)
        if resumed:
            req.log(f'resuming: {resumed} frame(s) of this request already landed; planning the rest')
        jobs = [j for j in req.jobs if j.want]
        for j in req.jobs:
            if not j.want:
                req.log(f'[{j.name}] nothing to render (every wanted frame is already here?)')
        units = plan_units(jobs, req.args)
        req.total = sum(len(set(o['frames']) & j.want) for j in jobs for o in j.outputs) + len(req.landed_set)
        req.units_total = req.units_open = len(units)
        req.budget = req.args.nodes or (1 if req.args.test is not None else max(1, len(units)))
        if not units:
            return req.finish('done', 'nothing to render')
        if req.args.dry_run:
            req.log(f'plan: {len(units)} unit(s), {req.total} frames, up to {req.budget} node(s) (dry run: nothing starts)')
            return print_plan(units, req.log, verbose=True)
        if all(p.over_budget() for p in POOLS):
            return req.finish('failed', 'every pool is past its spend threshold: ask the director')
        with self.lock:
            for u in units:
                u['_req'], u['_prio'], u['_seq'] = req, req.prio, self.seq
                self.seq += 1
            self.queue += units
        waiting = sum(1 for u in self.queue if u['_req'] is not req)
        req.log(f'queued: {len(units)} unit(s), {req.total} frames, '
                f'{"TEST (goes first)" if req.prio == 0 else "final"}, up to {req.budget} node(s); '
                f'{waiting} other unit(s) in the queue')
        print_plan(units, req.log, verbose=bool(req.args.dry_run))
        req.save()

    def spend(self):
        t, v = self.last_spend
        if time.time() - t > 60:
            v = mission_cost()
            self.last_spend = (time.time(), v)
            if v is not None and v >= SPEND_WARN:
                log(f'WARNING: farm spend ${v:.2f} (warn ${SPEND_WARN:.0f}, stop ${SPEND_STOP:.0f})')
        return v

    def cancel(self, req, why='cancelled by its caller'):
        with self.lock:
            if req.cancelled or req.state in TERMINAL:
                return
            req.cancelled = True
            mine = [u for u in self.queue if u['_req'] is req]
            for u in mine:
                self.queue.remove(u)
            req.units_open -= len(mine)
            running = [nd for nd in self.nodes.values() if nd.unit and nd.unit['_req'] is req]
        req.log(f'cancelling ({why}): {len(mine)} queued unit(s) dropped, {len(running)} running unit(s) stopping')
        for nd in running:
            try:
                nd.agent('POST', '/cancel', dict(unit=nd.unit['id']), timeout=20)
            except Exception:                       # noqa: BLE001
                pass
        self.maybe_finish(req)

    def maybe_finish(self, req):
        if req.state in TERMINAL or req.units_open > 0:
            return
        if req.cancelled:
            req.finish('cancelled')
        elif req.gave_up:
            req.finish('incomplete', f'{sum(len(frames_of(g)) for g in req.gave_up)} frame(s) never landed')
        else:
            req.finish('done')

    # -- nodes
    def trim_queues(self):
        """Keep at most MAX_QUEUED nodes per pool waiting in a capacity queue (cpu gets only 2 while spilling, so h100
        wakes and creates can queue too). Extra waiters are cancelled - newest first - which frees their slots."""
        for pool in POOLS:
            waiting = [nd for nd in self.nodes.values() if nd.pool is pool and nd.state == 'booting' and nd.queued]
            cpu_w = sorted((nd for nd in waiting if nd.kind == 'cpu'), key=lambda nd: nd.t_queued or 0)
            keep = 2 if self.spilling() else MAX_QUEUED
            extra = cpu_w[keep:] + [nd for nd in waiting if nd.kind == 'gpu'][max(0, MAX_QUEUED - min(len(cpu_w), keep)):]
            for nd in extra:
                nd.say(f'{nd.name}: leaving the capacity queue ({len(waiting)} of this pool\'s nodes were waiting)')
                self.bad_names[nd.name] = time.time() + 600
                nd.stop('trimmed from the capacity queue')

    def spilling(self):
        """cpu-8 has no room: h100-1 nodes take CPU units too (6.6x the price per node, so only then)."""
        if MAX_SPILL <= 0:
            return False
        if time.time() < self.cpu_blocked_until:
            return True
        return any(nd.kind == 'cpu' and nd.queued and nd.state == 'booting' for nd in self.nodes.values())

    def dispatchable(self, node_kind, bpy=False):
        """Units a node of this kind may start now, best first: its own kind before spilled CPU work, --test before
        finals, then FIFO. A request never runs on more than its --nodes budget at once."""
        kinds = (['gpu', 'cpu'] if node_kind == 'gpu' and self.spilling() else
                 (['cpu', 'gpu'] if bpy else ['cpu']) if node_kind == 'ssh' else [node_kind])
        running = {}
        for nd in self.nodes.values():
            if nd.unit:
                running[nd.unit['_req'].id] = running.get(nd.unit['_req'].id, 0) + 1
        out, taken = [], {}
        now = time.time()

        def eff(u):                                 # choose() applies the finals guarantees
            return u['_prio']
        for u in sorted(self.queue, key=lambda u: (ukind(u) != node_kind, eff(u), u['_seq'])):
            r = u['_req']
            if ukind(u) not in kinds or r.cancelled:
                continue
            if node_kind == 'ssh' and ukind(u) == 'gpu' and now - r.submitted < SSH_GPU_AFTER_S:
                continue                            # an h100 gets first claim on Cycles work
            if running.get(r.id, 0) + taken.get(r.id, 0) >= r.budget:
                continue
            taken[r.id] = taken.get(r.id, 0) + 1
            out.append(u)
        return out

    def acquire(self):
        now = time.time()
        with self.lock:
            live = [nd for nd in self.nodes.values() if nd.state in ('booting', 'ready')]
            if any(nd.kind == 'cpu' and nd.queued and now - (nd.t_queued or now) > 60 for nd in live):
                self.cpu_blocked_until = max(self.cpu_blocked_until, now + 300)   # cpu-8 queue isn't moving

            def idle(kind):
                return [nd for nd in live if nd.kind == kind and nd.unit is None and not nd.queued]
            cpu_units = self.dispatchable('cpu')
            if cpu_units:                                  # our own ssh machines first: they cost nothing
                for spec in ssh_specs():
                    nd = self.nodes.get(spec['name'])
                    if (nd is None or nd.state == 'stopped') and now >= self.backoff.get(spec['name'], (0, 0))[0] \
                            and lease(spec['name'], 'farmd'):
                        nd = SshNode(spec, self)
                        nd.req = cpu_units[0]['_req']
                        self.nodes[spec['name']] = nd
                        self.boot_pool.submit(self._adopt_ssh, nd)
                live = [nd for nd in self.nodes.values() if nd.state in ('booting', 'ready')]
                cpu_units = cpu_units[len(idle('ssh')) + sum(1 for nd in live if nd.kind == 'ssh' and nd.state == 'booting'):]
            gpu_own = [u for u in self.dispatchable('gpu') if ukind(u) == 'gpu']
            need = {'cpu': max(0, len(cpu_units) - len(idle('cpu'))), 'gpu': max(0, len(gpu_own) - len(idle('gpu')))}
            if need['cpu'] and self.spilling():
                gpu_live = sum(1 for nd in live if nd.kind == 'gpu')
                spare = max(0, len(idle('gpu')) - len(gpu_own))
                spill = max(0, min(need['cpu'] - spare, MAX_GPU + MAX_SPILL - gpu_live - need['gpu']))
                need['gpu'] += spill
                need['cpu'] = min(need['cpu'], 1)          # keep probing cpu-8 with one create at a time
            has_test = any(u['_prio'] == 0 for u in self.queue)
        self.trim_queues()
        added = 0
        for kind, n in need.items():
            for pool in sorted(POOLS, key=lambda p: sum(1 for k in live_leases() if p.owns(k))):
                if n <= 0:
                    break
                if now < self.backoff.get((pool.name, kind), (0, 0))[0]:
                    continue                               # that pool's shape is backing off
                if pool.over_budget():
                    if not self.budget_noted.get(pool.name):
                        log(f'pool {pool.name}: spend ${pool.spend():.2f} >= ${pool.spend_stop:.0f}; no new nodes there')
                        self.budget_noted[pool.name] = True
                    continue
                kd = KINDS[kind]
                cap_kind = kd['cap'] + (MAX_SPILL if kind == 'gpu' and self.spilling() else 0)
                cap_total = pool.max_stream if has_test else pool.max_stream - TEST_RESERVE
                waiting = {nm for nm, nd in self.nodes.items() if nd.queued and nd.state == 'booting'}
                held = [k for k in live_leases() if pool.owns(k) and k not in waiting]   # queued: no endpoint yet
                room = min(cap_total - len(held), cap_kind - sum(1 for k in held if k.startswith(pool.pfx(kind))))
                booting = [nd for nd in self.nodes.values() if nd.pool is pool and nd.state == 'booting']
                room = min(room, MAX_QUEUED + 2 - len(booting))
                if room <= 0:
                    continue
                creating = sum(1 for nd in self.nodes.values()
                               if nd.state == 'booting' and nd.created_new and nd.pool is pool)
                names = self.lease_names(pool, kind, min(n, room), max_new=max(0, MAX_BOOTING - creating))
                with self.lock:
                    disp = self.dispatchable(kind)
                    for i, name in enumerate(names):
                        nd = Node(name, kind, self, pool)
                        nd.req = disp[min(i, len(disp) - 1)]['_req'] if disp else None
                        self.nodes[name] = nd
                        self.boot_pool.submit(self._boot, nd)
                added += len(names)
                n -= len(names)
            if n > 0 and not self.waiting_noted:
                log(f'{kind}: every pool is at its cap or backing off; {n} more node(s) wanted, work waits its turn')
                self.waiting_noted = True
        if added:
            self.waiting_noted = False
        return added

    def lease_names(self, pool, kind, n, max_new=99):
        """Parked or running farm nodes first (a wake is not a create), then at most max_new brand-new names: the
        platform holds only 4 queued creates per account."""
        prefix = pool.pfx(kind)
        existing = {x['name']: x for x in pool.list_nodes()}
        self.known = set(existing)
        held = live_leases()
        got = []
        for x in sorted((x for x in existing.values() if x['name'].startswith(prefix)),
                        key=lambda x: (0 if str(x.get('state', '')).startswith('running') else 1, x['name'])):
            st = str(x.get('state', ''))
            if len(got) >= n:
                break
            if x['name'] in held or x['name'] in self.nodes and self.nodes[x['name']].state != 'stopped':
                continue
            if 'unrestorable' in st or 'lost' in st or self.bad_names.get(x['name'], 0) > time.time():
                continue
            if st.startswith('running'):            # a live agent may be mid-unit: sweep_orphans re-attaches it
                continue
            if lease(x['name'], 'farmd'):
                got.append(x['name'])
        i = 1
        new = 0
        while len(got) < n and i < 100 and new < max_new:
            name = f'{prefix}{i:02d}'
            i += 1
            if name in existing or name in held or self.bad_names.get(name, 0) > time.time():
                continue
            if lease(name, 'farmd'):
                got.append(name)
                new += 1
        return got

    def _boot(self, nd):
        try:
            nd.boot()
            self.boot_failures = 0
            if nd.pool:
                self.backoff.pop((nd.pool.name, nd.kind), None)
        except Interrupted:
            return
        except CapacityError as e:
            if 'wake' in str(e):
                self.bad_names[nd.name] = time.time() + 900
                nd.say(f'{nd.name}: its parked disk can\'t wake right now ({e}); using other nodes for 15 min')
                return nd.stop('wake failed')
            key = (nd.pool.name, nd.kind)
            until, delay = self.backoff.get(key, (0, 30))
            delay = min(300, delay * 2)
            self.backoff[key] = (time.time() + delay, delay)
            if nd.kind == 'cpu':
                self.cpu_blocked_until = time.time() + 900
            nd.say(f'{nd.name}: no {KINDS[nd.kind]["chip"]} capacity ({e}); that shape retries in {delay}s'
                   + ('; CPU units spill onto h100-1 meanwhile' if nd.kind == 'cpu' and MAX_SPILL else ''))
            nd.stop('no capacity', api_stop='create failed' not in str(e))
        except ApiError as e:                       # any refusal (queued-creates max, capacity, rate limit): wait, retry
            key = (nd.pool.name if nd.pool else nd.name, nd.kind)
            until, delay = self.backoff.get(key, (0, 15))
            delay = min(180, delay * 2)
            self.backoff[key] = (time.time() + delay, delay)
            if 'queued' in e.message or 'capacity' in e.message or 'carve' in e.message:
                if nd.kind == 'cpu':
                    self.cpu_blocked_until = time.time() + 900
            nd.say(f'{nd.name}: platform said "{e.message[:110]}"; waiting {delay}s, then retrying (the unit stays queued)')
            nd.stop('platform refusal', api_stop=nd.exists)
        except Exception as e:                      # noqa: BLE001 - this NODE is broken: skip it for a while
            nd.error = str(e)[:400]
            if nd.kind == 'ssh':
                self.backoff[nd.name] = (time.time() + 300, 300)
                nd.say(f'{nd.name}: not available ({nd.error[:200]}); retrying in 5 min')
                return nd.stop('ssh boot failed')
            if not STOP.is_set():
                self.boot_failures += 1
                self.bad_names[nd.name] = time.time() + 900
                nd.say(f'{nd.name}: failed to boot ({nd.error[:240]}); trying another node (the unit stays queued)')
                if self.boot_failures in (4, 10):
                    log(f'ALERT: {self.boot_failures} node boots in a row failed; last: {nd.error[:200]}')
            nd.stop('boot failed')

    # -- frames
    def fetch(self, nd, unit, entries):
        import cv2
        import numpy as np
        j, req = unit['_job'], unit['_req']
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
                        req.log(f'[{j.name}] {m.name} from {nd.name}: sha256 mismatch, will ask again')
                        continue
                    im = cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR)
                    if im is None or im.shape[:2] != j.shape:
                        req.log(f'[{j.name}] {m.name} from {nd.name}: bad decode {None if im is None else im.shape}')
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
                        req.landed_set.add((j.name, e['key'], e['frame']))
                        req.landed = len(req.landed_set)
        if got:
            nd.agent('POST', '/ack', dict(unit=unit['id'], files=got), timeout=30)
            fr = sorted(int(n.split('f_')[1][:5]) for n in got)
            req.log(f'[{j.name}] +{len(got)} ({compact(fr)[:60]}) from {nd.name} | {req.landed}/{req.total} landed')
            req.save()
        return got

    def tick(self, nd):
        """Poll one ready node: land new frames, notice a finished unit, hand out the next unit or stop the node."""
        try:
            st = nd.agent('GET', f'/status?since={nd.seq}', timeout=30)
            nd.bad_polls = 0
            nd.last_ok = time.time()
        except Exception as e:                      # noqa: BLE001
            nd.bad_polls += 1
            if nd.bad_polls in (3, 10, 30):
                log(f'{nd.name}: endpoint not answering ({str(e)[:120]}); {nd.bad_polls} polls')
            if time.time() - (nd.last_ok or time.time()) > 240:
                self.node_lost(nd, 'endpoint silent for 4 min')
            return
        new = st.get('ready', [])
        if new and nd.unit:
            mine = [e for e in new if e['unit'] == nd.unit['id']]
            if mine:
                try:
                    self.fetch(nd, nd.unit, mine)
                except Exception as e:              # noqa: BLE001
                    log(f'{nd.name}: download failed ({str(e)[:160]}); retrying')
                    return                           # nd.seq unchanged: the same frames are offered again
            nd.seq = max(nd.seq, max(e['seq'] for e in new))
        elif new:
            nd.seq = max(nd.seq, max(e['seq'] for e in new))
        if nd.unit:
            us = next((u for u in st.get('units', []) if u['id'] == nd.unit['id']), None)
            if us and us['state'] in ('done', 'failed', 'cancelled') and not st.get('more'):
                self.finish_unit(nd, us)
        if nd.unit is None:
            self.dispatch(nd)

    def _tick(self, nd):
        try:
            self.tick(nd)
        except Exception as e:                      # noqa: BLE001
            log(f'{nd.name}: poll error: {str(e)[:200]}')
        finally:
            nd.ticking = False

    def choose(self, nd, disp):
        """Tests first, with two guarantees for finals: an approved final's first unit starts within FINAL_AGING_S,
        and running finals keep about 1 node in 3 (FINAL_SHARE). Within a class: FIFO, with warm-setup affinity
        among the next 3 in line."""
        if not disp:
            return None
        now = time.time()
        tests = [u for u in disp if u['_prio'] == 0]
        finals = [u for u in disp if u['_prio'] == 1]
        running = [x.unit for x in self.nodes.values() if x.unit]
        fin_run = sum(1 for u in running if u['_prio'] == 1)
        unstarted = [u for u in finals if u['_req'].t_start is None and now - u['_req'].submitted > FINAL_AGING_S]
        if unstarted:
            pick = unstarted
        elif finals and (not tests or fin_run < max(1, round((len(running) + 1) * FINAL_SHARE))):
            pick = finals
        else:
            pick = tests or finals
        tier = pick[:3]
        return next((u for u in tier if u['_job'].name == nd.last_job), tier[0])

    def dispatch(self, nd):
        if nd.kind == 'ssh' and not nd.ok_to_work():
            nd.stop('owner needs the machine')
            self.backoff[nd.name] = (time.time() + 900, 900)
            return
        with self.lock:
            u = self.choose(nd, self.dispatchable(nd.kind, bpy=getattr(nd, 'bpy', False)))
            if u is not None:
                self.queue.remove(u)
                nd.unit = u
        if u is None:
            nd.stop('no queued work')
            return
        spec = {k: v for k, v in u.items() if not k.startswith('_')}
        if not u['_job'].gpu and nd.cpu > KINDS['cpu']['cpu']:     # a bigger node than planned for: use its cores
            spec['concurrency'] = min(len(spec['items']), max(spec['concurrency'], nd.cpu // u.get('_d', 2)))
        if nd.kind == 'ssh':                        # someone's laptop: total threads stay within its budget
            per = max(2, u.get('_d', 2))
            spec['concurrency'] = max(1, min(spec['concurrency'], nd.cpu // per))
            asked = [int(m) for it in spec['items'] for m in re.findall(r'--threads[ =](\d+)', it['cmd'])]
            spec['env'] = dict(spec.get('env') or {}, NUMBA_NUM_THREADS=str(max([nd.cpu // spec['concurrency']] + asked)))
        try:
            nd.agent('POST', '/unit', spec, timeout=30)
        except Exception as e:                      # noqa: BLE001
            with self.lock:
                nd.unit = None
                self.queue.append(u)
            log(f'{nd.name}: could not hand over {u["id"]} ({str(e)[:120]}); requeued')
            return
        req = u['_req']
        u['_t0'] = time.time()
        nd.last_job = u['_job'].name
        if req.t_start is None:
            req.t_start = time.time()
            req.state = 'running'
            req.save()
        fr = sum(len(o['frames']) for o in u['outputs'])
        req.log(f'{nd.name}: unit {u["id"]} ({fr} frames, {len(u["items"])} processes, {u["concurrency"]} at once)')

    def finish_unit(self, nd, us):
        u = nd.unit
        j, req = u['_job'], u['_req']
        want = {(o['key'], f) for o in u['outputs'] for f in o['frames']}
        got = self.landed.get(u['id'], set())
        missing = sorted(want - got)
        t = us.get('t', {})
        n = len(got)
        render_s = (t.get('end', 0) - t.get('setup_done', t.get('start', 0))) if t.get('end') else None
        res = dict(job=j.name, unit=u['id'], node=nd.name, kind=nd.kind, frames=n, state=us['state'],
                   wall=round(time.time() - u.get('_t0', time.time())), commit=us.get('commit'),
                   setup=round(t['setup_done'] - t['start']) if t.get('setup_done') and t.get('start') else None,
                   render=round(render_s) if render_s else None,
                   s_per_frame=round(render_s / n, 2) if render_s and n else None)
        req.results.append(res)
        self.unit_secs.append(res['wall'])
        msg = f'{nd.name}: unit {u["id"]} {us["state"]} at {us.get("commit")}: {n}/{len(want)} frames'
        if render_s and n:
            msg += f', {render_s / n:.2f} s/frame on the node (setup {res["setup"]}s)'
        req.log(msg)
        if us.get('error'):
            req.log(f'    error: {us["error"]}')
        for i, lines in (us.get('fail_tails') or {}).items():
            req.log(f'    process {i} tail:')
            for ln in lines[-12:]:
                req.log(f'      | {ln[:200]}')
        for ln in (us.get('setup_tail') or [])[-15:]:
            req.log(f'      setup| {ln[:200]}')
        if missing and not us.get('fail_tails') and not us.get('setup_tail'):
            for i in range(min(2, len(u['items']))):   # exit 0 but no frames: show what the process said
                try:
                    lines = nd.agent('GET', f'/log/{u["id"]}/item_{i}.log?n=40', timeout=20).get('lines', [])
                except Exception:                   # noqa: BLE001
                    lines = []
                key = [ln for ln in lines if any(w in ln for w in ('Error', 'error', 'Traceback', 'exit ', 'rror:'))]
                req.log(f'    process {i} exited 0 but its frames never appeared; its log says:')
                for ln in (key or lines)[-12:]:
                    req.log(f'      | {ln[:200]}')
        with self.lock:
            nd.unit = None
            req.units_open -= 1
            if missing and not req.cancelled:
                frames = sorted({f for _, f in missing})
                if u['_retry'] < 1 and us['state'] != 'cancelled' and not STOP.is_set():
                    req.log(f'[{j.name}] {len(frames)} frame(s) missing ({compact(frames)[:80]}): one retry queued')
                    for r in j.units(1, KINDS[ukind(u)]['cpu'], retry_of=u['_retry'] + 1, frames=frames):
                        r['_req'], r['_prio'], r['_seq'] = req, req.prio, self.seq
                        self.seq += 1
                        self.queue.append(r)
                        req.units_open += 1
                else:
                    req.log(f'[{j.name}] GAVE UP on {len(frames)} frame(s): {compact(frames)}')
                    res['missing'] = compact(frames)
                    req.gave_up.append(compact(frames))
        req.save()
        self.maybe_finish(req)

    def node_lost(self, nd, why):
        log(f'{nd.name}: LOST ({why})')
        with self.lock:
            u, nd.unit = nd.unit, None
        if u:
            req, j = u['_req'], u['_job']
            want = {(o['key'], f) for o in u['outputs'] for f in o['frames']}
            missing = sorted({f for _, f in want - self.landed.get(u['id'], set())})
            with self.lock:
                req.units_open -= 1
                if missing and not req.cancelled:
                    for r in j.units(1, KINDS[nd.kind]['cpu'], retry_of=u['_retry'] + 1, frames=missing):
                        r['_req'], r['_prio'], r['_seq'] = req, req.prio, self.seq
                        self.seq += 1
                        self.queue.append(r)
                        req.units_open += 1
            req.log(f'{nd.name} was lost ({why}); {len(missing)} frame(s) requeued')
            self.maybe_finish(req)
        nd.stop(why)

    # -- the loop
    def step(self):
        self.sweep_orphans()
        live = [nd for nd in self.nodes.values() if nd.state in ('booting', 'ready')]
        if self.queue and (time.time() - self.last_grow > 10 or not live):
            self.last_grow = time.time()
            self.acquire()                          # units are never dropped: they wait until a node takes them
            self.report_waiting()
        for nd in [nd for nd in self.nodes.values() if nd.state == 'ready' and not nd.ticking]:
            nd.ticking = True
            self.pool.submit(self._tick, nd)
        try:
            self.report_spend()
        except Exception as e:                      # noqa: BLE001
            log(f'spend report failed: {str(e)[:120]}')
        for name in [n for n, nd in self.nodes.items() if nd.state == 'stopped']:
            del self.nodes[name]

    def sweep_orphans(self):
        """A running farm node that no farm process holds (an old daemon's, a crashed run's) is adopted when work waits
        for its kind and stopped at once otherwise: nothing idles on the meter."""
        if time.time() - self.last_sweep < 60:
            return
        self.last_sweep = time.time()
        held = live_leases()
        items = []
        for pool in POOLS:
            try:
                items += [(pool, x) for x in pool.list_nodes()]
            except ApiError:
                pass
        for pool, x in items:
            name, st = x['name'], str(x.get('state', ''))
            if not pool.owns(name) or name in held or not st.startswith('running'):
                continue
            if name in self.nodes and self.nodes[name].state != 'stopped':
                continue
            kind = 'gpu' if name.startswith(pool.pfx('gpu')) else 'cpu'
            with self.lock:
                work = [u for u in self.queue if ukind(u) == kind or (kind == 'gpu' and ukind(u) == 'cpu')]
            if work and lease(name, 'farmd'):
                if kind == 'gpu' and all(ukind(u) == 'cpu' for u in work):
                    self.cpu_blocked_until = max(self.cpu_blocked_until, time.time() + 300)
                nd = Node(name, kind, self, pool)
                nd.req = work[0]['_req']
                self.nodes[name] = nd
                self.boot_pool.submit(self._adopt, nd)
                log(f'{name}: adopting a running node nobody held ({len(work)} unit(s) waiting)')
            elif not work:
                try:
                    pool.api('POST', f'/nodes/{name}/stop', dict(mission=pool.mission))
                    log(f'{name}: stopped (running, held by no farm process, no work waiting)')
                except ApiError as e:
                    log(f'{name}: orphan stop failed: {e.message[:120]}')

    def rebind(self, nd, st):
        """The agent's in-flight (or finished-but-uncollected) unit becomes this node's unit again, by its id."""
        live = [u for u in st.get('units', []) if u['state'] not in ('done', 'failed', 'cancelled')
                or u.get('frames_ready', 0) > u.get('acked', 0)]
        with self.lock:
            for au in live[-1:]:
                q = next((u for u in self.queue if u['id'] == au['id']), None)
                if q is not None:
                    self.queue.remove(q)
                    nd.unit, q['_t0'] = q, time.time()
                    nd.last_job = q['_job'].name
                    q['_req'].log(f'{nd.name}: re-attached to its running unit {q["id"]} ({au["state"]})')

    def _adopt_ssh(self, nd):
        try:
            nd.url, nd.tok = f'http://127.0.0.1:{nd.lport}', 'ssh-tunnel'
            st = nd.agent('GET', '/status', timeout=10)
            if int(st.get('version', 0)) < 5:
                raise RuntimeError('old agent')
            self.rebind(nd, st)
            nd.t_ready = nd.last_ok = time.time()
            nd.state = 'ready'
            nd.say(f'{nd.name}: re-attached to its running agent')
        except Exception:                           # noqa: BLE001 - no live agent there: boot it
            self._boot(nd)

    def _adopt(self, nd):
        """Re-attach to the agent already running there (fresh endpoint token); if it answers, rebind its in-flight
        unit to the matching queued unit and carry on. Otherwise boot the node from scratch."""
        try:
            nd.exists = True
            info = nd.pool.get_node(nd.name) or {}
            nd.cpu = int((info.get('resources') or {}).get('cpu_limit') or nd.cpu)
            nd.region = info.get('region')
            try:
                nd.pool.api('DELETE', f'/nodes/{nd.name}/endpoints/{PORT}')
            except ApiError:
                pass
            nd.expose()
            st = None
            for _ in range(5):
                try:
                    st = nd.agent('GET', '/status', timeout=15)
                    break
                except Exception:                   # noqa: BLE001
                    time.sleep(3)
            if not st or int(st.get('version', 0)) < 4:
                raise RuntimeError('no current agent answers there')
            self.rebind(nd, st)
            nd.t_ready = nd.last_ok = time.time()
            nd.state = 'ready'
            nd.say(f'{nd.name}: adopted without a reboot ({nd.kind}, cpu_limit {nd.cpu}, {nd.region})')
        except Interrupted:
            return
        except Exception as e:                      # noqa: BLE001
            nd.say(f'{nd.name}: re-attach failed ({str(e)[:120]}); booting it fresh')
            self._boot(nd)

    def report_waiting(self):
        if time.time() - self.last_report < 60:
            return
        self.last_report = time.time()
        with self.lock:
            order = sorted(self.queue, key=lambda u: (u['_prio'], u['_seq']))
            busy = sum(1 for nd in self.nodes.values() if nd.unit)
            live = sum(1 for nd in self.nodes.values() if nd.state in ('booting', 'ready'))
        avg = sum(self.unit_secs[-20:]) / len(self.unit_secs[-20:]) if self.unit_secs else 240
        seen = set()
        for pos, u in enumerate(order, 1):
            r = u['_req']
            if r.id in seen:
                continue
            seen.add(r.id)
            eta = (pos / max(1, live)) * avg / 60
            r.log(f'waiting: queue position {pos} of {len(order)} ({"test" if r.prio == 0 else "final"}), '
                  f'{busy} node(s) rendering, {live} up or booting; ETA to start ~{eta:.0f} min')

    def report_spend(self):
        """While busy: an hourly per-pool node count + spend line on each pool's mission page, and a one-time alert
        when a pool reaches 80% of its spend stop."""
        now = time.time()
        if now - self.last_spend_post < 300:
            return
        self.last_spend_post = now
        lines = []
        for pool in POOLS:
            sp = pool.spend()
            nodes = [nd for nd in self.nodes.values() if nd.pool is pool and nd.state in ('ready', 'booting')]
            gpu = sum(1 for nd in nodes if nd.kind == 'gpu' and nd.state == 'ready')
            cpu = sum(1 for nd in nodes if nd.kind == 'cpu' and nd.state == 'ready')
            wait = sum(1 for nd in nodes if nd.queued)
            lines.append(f'pool {pool.name}: {gpu} h100-1 + {cpu} cpu-8 up ({wait} waiting for capacity), spend '
                         f'${sp if sp is not None else float("nan"):.2f} of its ${pool.spend_stop:.0f} stop')
            if sp is not None and sp >= 0.8 * pool.spend_stop and not self.alerted.get(pool.name):
                self.alerted[pool.name] = True
                try:
                    pool.api('POST', f'/missions/{pool.mission}/updates', dict(
                        kind='alert', title=f'Farm pool {pool.name} at ${sp:.2f}: 80% of its ${pool.spend_stop:.0f} stop',
                        body='New nodes stop in this pool at the threshold; running work finishes. Raise it or accept.'))
                except ApiError:
                    pass
        if now - self.last_hourly < 3600 or not self.nodes:
            return
        self.last_hourly = now
        ssh_up = sum(1 for nd in self.nodes.values() if nd.kind == 'ssh' and nd.state == 'ready')
        queued = len(self.queue)
        body = '\n'.join('- ' + ln for ln in lines) + f'\n- M4 (ssh): {"up" if ssh_up else "idle"}\n- {queued} unit(s) queued'
        for pool in POOLS:
            try:
                pool.api('POST', f'/missions/{pool.mission}/updates', dict(
                    title=f'Farm hourly: {sum(1 for nd in self.nodes.values() if nd.state == "ready")} nodes up, {queued} queued',
                    body=body, kind='note'))
            except ApiError:
                pass

    def active(self):
        return any(r.state not in TERMINAL for r in self.reqs.values())

    def shutdown(self):
        for nd in list(self.nodes.values()):
            nd.stop('farm shutting down')


def plan_units(jobs, args):
    n = args.nodes or 1
    weights = {j.name: len(j.want) for j in jobs}
    tot = sum(weights.values()) or 1
    units = []
    for j in jobs:
        kind = 'gpu' if j.gpu else 'cpu'
        u = 1 if args.test is not None and not args.nodes else max(1, round(n * weights[j.name] / tot))
        units += j.units(u, KINDS[kind]['cpu'])
    units.sort(key=lambda u: -sum(len(o['frames']) for o in u['outputs']))
    return units


def print_plan(units, say, verbose=False):
    for u in units:
        j = u['_job']
        fr = sorted({f for o in u['outputs'] for f in o['frames']})
        say(f'unit {u["id"]} [{"gpu" if j.gpu else "cpu"}] {len(fr)} frames ({compact(fr)[:80]}), '
            f'{len(u["items"])} processes, {u["concurrency"]} at once')
        if verbose or len(units) <= 4:
            for it in u['items'][:12]:
                say(f'    {it["cmd"][:220]}')
            if len(u['items']) > 12:
                say(f'    ... {len(u["items"]) - 12} more')
    for j in {u['_job'] for u in units}:
        for c in j.skipped_setup:
            say(f'[{j.name}] setup step skipped (the farm image provides it): {c[:120]}')
        for o in j.outputs:
            if set(o['frames']) & j.want:
                say(f'[{j.name}] output {o["out_dir"]} -> {o["dest"]}')


# ------------------------------------------------------------------ daemon and client

def daemon_alive():
    os.makedirs(CACHE, exist_ok=True)
    fh = open(DLOCK, 'a+')
    try:
        fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        fh.close()
        return True
    fcntl.flock(fh, fcntl.LOCK_UN)
    fh.close()
    return False


def ensure_daemon():
    if daemon_alive():
        return
    # double fork: the daemon's parent is launchd, never a farm.py client (so it can't linger as a zombie)
    subprocess.Popen(['/bin/sh', '-c', 'nohup "$0" "$1" daemon >> "$2" 2>&1 < /dev/null &',
                      VENV_PY, os.path.abspath(__file__), DLOG], cwd=os.path.dirname(ROOT)).wait()
    for _ in range(40):
        time.sleep(0.25)
        if daemon_alive():
            return


def cmd_daemon():
    os.makedirs(INBOX, exist_ok=True)
    os.makedirs(REQDIR, exist_ok=True)
    lockf = open(DLOCK, 'a+')
    try:
        fcntl.flock(lockf, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        return                                      # another daemon has the pool
    with open(DPID, 'w') as fh:
        fh.write(str(os.getpid()))
    upgrade = threading.Event()
    signal.signal(signal.SIGTERM, lambda *a: STOP.set())
    signal.signal(signal.SIGINT, lambda *a: STOP.set())
    signal.signal(signal.SIGUSR1, lambda *a: (upgrade.set(), STOP.set()))
    log(f'farm daemon {os.getpid()} up (streaming nodes {MAX_STREAM}, test reserve {TEST_RESERVE})')
    sched = Scheduler()
    idle_since = time.time()
    try:
        while not STOP.is_set():
            for fn in sorted(os.listdir(INBOX)):
                rid = fn[:-5]
                if not fn.endswith('.json') or rid in sched.reqs:
                    continue
                try:
                    r = json.load(open(os.path.join(INBOX, fn)))
                    req = Request(rid, r['argv'], r['submitted'])
                except Exception as e:              # noqa: BLE001
                    log(f'bad request {fn}: {e}')
                    os.replace(os.path.join(INBOX, fn), os.path.join(INBOX, fn + '.bad'))
                    continue
                sched.add(req)
            for req in list(sched.reqs.values()):
                if req.state not in TERMINAL and os.path.exists(req.cancelp):
                    sched.cancel(req)
            sched.step()
            if sched.active() or sched.nodes:
                idle_since = time.time()
            elif time.time() - idle_since > DAEMON_IDLE_EXIT:
                log('no requests for a while: daemon exits (the next farm.py call starts a new one)')
                break
            for rid in [rid for rid, r in sched.reqs.items() if r.state in TERMINAL and time.time() - r.t_end > 600]:
                del sched.reqs[rid]
            time.sleep(2)
    finally:
        if upgrade.is_set():
            log('farm daemon exits for an upgrade: nodes and requests stay for the next daemon')
            os._exit(0)
        sched.shutdown()
        log('farm daemon down')


def new_request_id(args):
    base = re.sub(r'[^A-Za-z0-9]', '', os.path.basename(args.jobs[0]).replace('.json', ''))[:24]
    return f'{time.strftime("%m%d-%H%M%S")}-{base}-{os.getpid() % 100000:05d}'


def cmd_submit(argv):
    args = parse_run_args(argv)
    if args.dry_run or args.direct:
        return cmd_direct(argv, args)
    fixed = []
    for i, a in enumerate(argv):                    # absolute paths: the daemon runs elsewhere
        prev = argv[i - 1] if i else ''
        if a.endswith('.json') and not a.startswith('-'):
            p = a if os.path.exists(a) else os.path.join(ROOT, 'cloud', 'jobs', a)
            if not os.path.exists(p):
                sys.exit(f'farm: no such job file: {a}')
            json.load(open(p))                      # a broken job file fails here, not in the daemon
            a = os.path.abspath(p)
        elif prev == '--local-out':
            a = os.path.abspath(os.path.expanduser(a))
        fixed.append(a)
    rid = new_request_id(args)
    os.makedirs(INBOX, exist_ok=True)
    os.makedirs(REQDIR, exist_ok=True)
    logp, statep, cancelp = _paths(rid)
    tmp = os.path.join(INBOX, rid + '.tmp')
    with open(tmp, 'w') as fh:
        json.dump(dict(argv=fixed, submitted=time.time(), client_pid=os.getpid(), cwd=os.getcwd()), fh)
    os.replace(tmp, os.path.join(INBOX, rid + '.json'))
    print(time.strftime('%H:%M:%S') + f' submitted {rid} to the farm queue', flush=True)
    ensure_daemon()
    if args.detach:
        print(f'follow it with: tail -f {logp}   (state: {statep})')
        return 0
    return follow(rid)


def follow(rid):
    logp, statep, cancelp = _paths(rid)
    state = {'cancel': False}

    def on_signal(sig, frame):
        if state['cancel']:
            os._exit(130)
        state['cancel'] = True
        open(cancelp, 'w').close()
        print(time.strftime('%H:%M:%S') + ' cancelling this request (again to leave it running)...', flush=True)
    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)
    pos, last_check = 0, time.time()
    while True:
        if os.path.exists(logp):
            with open(logp) as fh:
                fh.seek(pos)
                chunk = fh.read()
                pos = fh.tell()
            if chunk:
                sys.stdout.write(chunk)
                sys.stdout.flush()
        st = {}
        try:
            st = json.load(open(statep))
        except (OSError, ValueError):
            pass
        if st.get('state') in TERMINAL:
            with open(logp) as fh:
                fh.seek(pos)
                sys.stdout.write(fh.read())
            return 0 if st['state'] == 'done' else (2 if st['state'] == 'cancelled' else 1)
        if time.time() - last_check > 20:
            last_check = time.time()
            if not daemon_alive():
                print(time.strftime('%H:%M:%S') + ' the farm daemon is not running: restarting it', flush=True)
                ensure_daemon()
        time.sleep(1)


def cmd_direct(argv, args):
    """In-process run (--direct, and --dry-run): the same scheduler, no daemon."""
    req = Request(new_request_id(args), argv, time.time(), direct=True)
    sched = Scheduler()
    sched.add(req)
    if args.dry_run or req.state in TERMINAL:
        return 0

    def on_signal(sig, frame):
        if STOP.is_set():
            os._exit(1)
        req.log('interrupt: stopping the farm nodes (press again to abandon)')
        STOP.set()
    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)
    try:
        while req.state not in TERMINAL and not STOP.is_set():
            sched.step()
            time.sleep(2)
    finally:
        sched.shutdown()
        if req.state not in TERMINAL:
            req.finish('cancelled', 'interrupted')
    return 0 if req.state == 'done' else 1


def cmd_status():
    held = live_leases()
    alive = daemon_alive()
    print(f'farm daemon: {"running (pid " + open(DPID).read().strip() + ")" if alive and os.path.exists(DPID) else "not running (starts on the next farm.py call)"}')
    for pool in POOLS:
        sp = pool.mission_cost()
        print(f'pool {pool.name}: workspace {pool.workspace}, token expires {pool.meta().get("expires_at", "?")}, '
              f'spend ${sp if sp is not None else float("nan"):.2f} (refuses new nodes at ${pool.spend_stop:.0f}; {pool.cap_note})')
    rows = []
    if os.path.isdir(REQDIR):
        for fn in os.listdir(REQDIR):
            if fn.endswith('.json'):
                try:
                    rows.append(json.load(open(os.path.join(REQDIR, fn))))
                except (OSError, ValueError):
                    pass
    act = sorted([r for r in rows if r.get('state') not in TERMINAL], key=lambda r: r.get('submitted', 0))
    done = sorted([r for r in rows if r.get('state') in TERMINAL], key=lambda r: r.get('t_end') or 0)[-6:]
    print(f'requests: {len(act)} active')
    for r in act + done:
        age = (time.time() - r.get('submitted', time.time())) / 60
        print(f'  {r["id"]:<44} {r.get("kind", "?"):<5} {r.get("state", "?"):<10} {r.get("landed", 0)}/{r.get("total", 0)} frames, '
              f'{r.get("units_total", 0) - r.get("units_open", 0)}/{r.get("units_total", 0)} units, {age:.0f} min ago')
    items = []
    for pool in POOLS:
        try:
            items += pool.list_nodes()
        except ApiError as e:
            print(f'pool {pool.name} nodes: {e}')
    print(f'nodes: {sum(1 for x in items if str(x.get("state", "")).startswith("running"))} running, {len(held)} held')
    for x in sorted(items, key=lambda x: x['name']):
        h = held.get(x['name'])
        print(f'  {x["name"]:<10} {x.get("chip", "?"):<7} {str(x.get("state", "?")):<34} '
              f'{("held by pid " + str(h["pid"]) + " (" + h["run"] + ")") if h else ""}')
    for sp in ssh_specs():
        h = held.get(sp['name'])
        print(f'  {sp["name"]:<10} {"ssh":<7} {sp["host"]:<34} {("in use by pid " + str(h["pid"])) if h else "idle (starts when CPU work is queued)"}')


def cmd_stop_idle():
    held = live_leases()
    for pool in POOLS:
        for x in pool.list_nodes():
            st = str(x.get('state', ''))
            if x['name'] in held or not pool.owns(x['name']) or not st.startswith(('running', 'provisioning', 'queued', 'waking')):
                continue
            try:
                pool.api('POST', f'/nodes/{x["name"]}/stop', dict(mission=pool.mission))
                print(f'stopped {x["name"]} (was {st})')
            except ApiError as e:
                print(f'{x["name"]}: {e.message[:200]}')


def cmd_cancel(rid):
    logp, statep, cancelp = _paths(rid)
    if not os.path.exists(statep) and not os.path.exists(os.path.join(INBOX, rid + '.json')):
        sys.exit(f'farm: no request {rid} (farm.py status lists them)')
    open(cancelp, 'w').close()
    print(f'cancel requested for {rid}')


def parse_run_args(argv):
    ap = argparse.ArgumentParser(prog='farm.py', description='THE LONG DAWN render farm', usage=__doc__)
    ap.add_argument('jobs', nargs='+')
    ap.add_argument('--nodes', type=int, default=None)
    ap.add_argument('--gpu', action='store_true')
    ap.add_argument('--frames', default=None)
    ap.add_argument('--test', type=int, nargs='?', const=0, default=None)
    ap.add_argument('--local-out', default=None)
    ap.add_argument('--missing', action='store_true')
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--direct', action='store_true')
    ap.add_argument('--detach', action='store_true')
    args = ap.parse_args(argv)
    args.jobs = [p if os.path.exists(p) or not os.path.exists(os.path.join(ROOT, 'cloud', 'jobs', p))
                 else os.path.join(ROOT, 'cloud', 'jobs', p) for p in args.jobs]
    return args


def ensure_venv():
    import importlib.util
    if importlib.util.find_spec('cv2') is None:
        if os.path.exists(VENV_PY) and os.path.realpath(sys.executable) != os.path.realpath(VENV_PY):
            os.execv(VENV_PY, [VENV_PY] + sys.argv)
        sys.exit('farm: needs cv2 (the ~/.venvs/longdawn venv)')


def main():
    POOLS[:] = load_pools()
    argv = sys.argv[1:]
    cmd = argv[0] if argv else ''
    if cmd == 'status':
        return cmd_status()
    if cmd == 'stop-idle':
        return cmd_stop_idle()
    if cmd == 'cancel' and len(argv) == 2:
        return cmd_cancel(argv[1])
    if cmd == 'daemon':
        ensure_venv()
        return cmd_daemon()
    if not argv or cmd in ('-h', '--help', 'help'):
        print(__doc__)
        return 0
    if '--direct' in argv:
        ensure_venv()
    return cmd_submit(argv)


if __name__ == '__main__':
    sys.exit(main() or 0)
