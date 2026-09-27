#!/usr/bin/env python3
"""THE LONG DAWN render farm: the agent that runs ON a farm node (cloud/farm.py on the Mac drives it).

    ~/ld/venv/bin/python ~/ld/farm_node.py agent --port 8700

farm.py uploads this file to each node, starts one agent per node session, and reaches it through the
platform's expose_port endpoint (bearer auth at the edge). The agent speaks a small HTTP API:

    GET  /status?since=N           agent + unit states, and every ready frame with seq > N
    POST /unit                     queue a unit (JSON spec below); units run one at a time, in order
    GET  /tar/<unit>?f=k/name,...  an uncompressed tar of ready JPEGs (the Mac lands them, then acks)
    POST /ack                      {"unit": id, "files": ["0/f_01040.jpg", ...]}: collected on the Mac
    POST /cancel                   {"unit": id}: kill a unit's processes
    GET  /log/<unit>/<name>?n=60   tail of a unit log (setup.log, item_3.log)
    POST /quit                     {"stop": true} also stops this node through the metadata service

A unit is one slice of one job:
    {"id", "job", "setup": [cmd], "items": [{"cmd", "env"}], "concurrency", "env",
     "outputs": [{"key", "out_dir", "frames": [..]}], "shape": [h, w]}
For each unit the agent brings the repo to the branch tip (git fetch --depth 1 + reset --hard, so lanes must
push their code first), deletes stale copies of the unit's frames, runs setup (skipped when this node already
ran the same setup at the same commit), runs the items (at most `concurrency` at once), and turns every
finished frame into a decode-checked JPEG q95 4:4:4 under ~/ld/runs/<unit>/ready/<key>/.
Watchdog: with no unit queued or running and no request for IDLE_STOP_S seconds, the node stops itself
(files park on its disk; nothing idles on the meter).
"""
import hashlib
import io
import json
import os
import shutil
import signal
import subprocess
import sys
import tarfile
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

VERSION = 4
HOME = os.path.expanduser('~')
LD = os.path.join(HOME, 'ld')
REPO = os.path.join(LD, 'mishamisha')
ROOT = os.path.join(REPO, 'the-long-dawn')
RUNS = os.path.join(LD, 'runs')
BRANCH = 'claude/long-dawn-v2'
IDLE_STOP_S = int(os.environ.get('LDFARM_IDLE_STOP_S', '600'))
CPU_LIMIT = int(os.environ.get('GMN_CPU_LIMIT') or 8)

LOCK = threading.RLock()
UNITS = {}            # id -> unit dict (state lives here)
ORDER = []            # unit ids in arrival order
READY = []            # [{seq, unit, key, frame, name, size, sha}]
SEQ = [0]
LAST_REQ = [time.time()]
SETUP_DONE = set()    # (setup hash, commit)
STOPPING = [False]
T0 = time.time()


def now():
    return time.time()


def tail(path, n=40):
    try:
        with open(path, 'rb') as fh:
            fh.seek(0, 2)
            size = fh.tell()
            fh.seek(max(0, size - 64 * 1024))
            return fh.read().decode('utf-8', 'replace').splitlines()[-n:]
    except OSError:
        return []


def base_env(extra=None):
    env = dict(os.environ)
    env['PATH'] = os.path.join(LD, 'venv', 'bin') + ':' + os.path.join(HOME, '.local', 'bin') + ':' + env.get('PATH', '')
    env['PYTHONUNBUFFERED'] = '1'
    env.setdefault('CUDA_CACHE_MAXSIZE', '4294967296')     # keep Cycles' one-time sm_90 JIT in ~/.nv (4 GiB cap)
    env.setdefault('NUMBA_NUM_THREADS', str(CPU_LIMIT))     # numba's pool ceiling = this node's budget, not the host
    for k, v in (extra or {}).items():
        env[str(k)] = str(v)
    return env


def git_update(u):
    """Bring the repo to the branch tip; returns the short commit. Three tries, then the unit fails."""
    err = ''
    for attempt in range(3):
        r = subprocess.run(f'git -C {REPO} fetch -q --depth 1 origin {BRANCH} && git -C {REPO} reset -q --hard FETCH_HEAD',
                           shell=True, capture_output=True, text=True)
        if r.returncode == 0:
            c = subprocess.run(['git', '-C', REPO, 'rev-parse', '--short', 'HEAD'], capture_output=True, text=True)
            return c.stdout.strip()
        err = (r.stderr or r.stdout)[-400:]
        time.sleep(5 * (attempt + 1))
    raise RuntimeError('git update failed: ' + err)


def frame_files(out_dir, f):
    p = os.path.join(ROOT, out_dir, f'f_{f:05d}.png')
    return p, p[:-4] + '.jpg'


class Unit:
    def __init__(self, spec):
        self.spec = spec
        self.id = spec['id']
        self.dir = os.path.join(RUNS, self.id)
        self.ready_dir = os.path.join(self.dir, 'ready')
        self.state = 'queued'
        self.error = None
        self.commit = None
        self.t = {'queued': now()}
        self.items = [dict(i=i, state='waiting', exit=None, t0=None, t1=None) for i in range(len(spec['items']))]
        self.procs = {}
        self.todo = {}        # (key, frame) -> first-seen info while waiting for a stable file
        self.done = set()     # (key, frame) converted
        self.bad = {}         # (key, frame) -> reason
        self.acked = set()    # (key, name)
        self.cancelled = False
        self.shape = tuple(spec.get('shape') or (804, 1920))
        self.want = {(o['key'], f) for o in spec['outputs'] for f in o['frames']}
        self.outdir = {o['key']: o['out_dir'] for o in spec['outputs']}

    def public(self, full=False):
        d = dict(id=self.id, job=self.spec.get('job'), state=self.state, error=self.error, commit=self.commit, t=self.t,
                 items=self.items, frames_total=len(self.want), frames_ready=len(self.done), bad=len(self.bad),
                 acked=len(self.acked))
        if self.state in ('done', 'failed', 'cancelled') or full:
            d['missing'] = sorted(f'{k}:{f}' for k, f in self.want - self.done - set(self.bad))[:400]
            d['bad_frames'] = {f'{k}:{f}': r for (k, f), r in list(self.bad.items())[:100]}
            fails = [it['i'] for it in self.items if it['exit'] not in (0, None)]
            d['fail_tails'] = {str(i): tail(os.path.join(self.dir, f'item_{i}.log'), 30) for i in fails[:4]}
            if self.error and 'setup' in (self.error or ''):
                d['setup_tail'] = tail(os.path.join(self.dir, 'setup.log'), 40)
        return d


# ------------------------------------------------------------------ frames -> JPEG

def scan(u, final=False):
    """Convert every newly finished frame of unit u. A PNG is taken once its size is stable across two scans
    (look.save_png is atomic anyway); an undecodable or wrong-shape file is retried for 30 s, then marked bad."""
    import cv2
    import numpy as np
    for key, f in sorted(u.want - u.done - set(u.bad)):
        png, jpg = frame_files(u.outdir[key], f)
        src = png if os.path.exists(png) else (jpg if os.path.exists(jpg) else None)
        if src is None:
            continue
        try:
            st = os.stat(src)
        except OSError:
            continue
        sig = (src, st.st_size, st.st_mtime)
        prev = u.todo.get((key, f))
        if prev is None or prev['sig'] != sig:
            u.todo[(key, f)] = {'sig': sig, 'seen': now()}
            if not final:
                continue
        try:
            data = open(src, 'rb').read()
        except OSError:
            continue
        im = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if im is None or im.shape[:2] != u.shape:
            age = now() - u.todo[(key, f)]['seen']
            if age < 30 and not final:
                continue
            u.bad[(key, f)] = f'unreadable' if im is None else f'shape {im.shape[:2]} != {u.shape}'
            try:
                os.replace(src, src + '.bad')
            except OSError:
                pass
            continue
        if src.endswith('.png'):
            ok, enc = cv2.imencode('.jpg', im, [cv2.IMWRITE_JPEG_QUALITY, 95,
                                                cv2.IMWRITE_JPEG_SAMPLING_FACTOR, cv2.IMWRITE_JPEG_SAMPLING_FACTOR_444])
            if not ok:
                u.bad[(key, f)] = 'jpeg encode failed'
                continue
            data = enc.tobytes()
        name = f'f_{f:05d}.jpg'
        d = os.path.join(u.ready_dir, str(key))
        os.makedirs(d, exist_ok=True)
        tmp = os.path.join(d, name + '.tmp')
        with open(tmp, 'wb') as fh:
            fh.write(data)
        os.replace(tmp, os.path.join(d, name))
        with LOCK:
            SEQ[0] += 1
            READY.append(dict(seq=SEQ[0], unit=u.id, key=key, frame=f, name=name, size=len(data),
                              sha=hashlib.sha256(data).hexdigest(), t=round(now(), 2)))
            u.done.add((key, f))
            if 'first_frame' not in u.t:
                u.t['first_frame'] = now()
        u.todo.pop((key, f), None)


# ------------------------------------------------------------------ running a unit

def run_unit(u):
    os.makedirs(u.ready_dir, exist_ok=True)
    spec = u.spec
    env = base_env(spec.get('env'))
    try:
        u.state = 'updating'
        u.t['start'] = now()
        u.commit = git_update(u)
        os.makedirs(os.path.join(ROOT, 'cloud_logs'), exist_ok=True)
        for key, f in u.want:                      # stale copies from an earlier run on this disk must never ship
            for p in frame_files(u.outdir[key], f):
                if os.path.exists(p):
                    os.remove(p)
        for o in spec['outputs']:
            os.makedirs(os.path.join(ROOT, o['out_dir']), exist_ok=True)
        setup = spec.get('setup') or []
        sh = hashlib.sha256(json.dumps(setup).encode()).hexdigest()[:16]
        if setup and (sh, u.commit) not in SETUP_DONE:
            u.state = 'setup'
            with open(os.path.join(u.dir, 'setup.log'), 'w') as lf:
                for cmd in setup:
                    if u.cancelled:
                        raise RuntimeError('cancelled')
                    lf.write(f'$ {cmd}\n')
                    lf.flush()
                    t = now()
                    r = subprocess.run(cmd, shell=True, cwd=ROOT, env=env, stdout=lf, stderr=subprocess.STDOUT)
                    lf.write(f'[exit {r.returncode} in {now() - t:.0f}s]\n')
                    lf.flush()
                    if r.returncode != 0:
                        raise RuntimeError(f'setup failed (exit {r.returncode}): {cmd[:200]}')
            SETUP_DONE.add((sh, u.commit))
        u.t['setup_done'] = now()
        u.state = 'rendering'
        conc = max(1, int(spec.get('concurrency') or len(spec['items'])))
        pending = list(range(len(spec['items'])))
        while (pending or u.procs) and not u.cancelled:
            while pending and len(u.procs) < conc:
                i = pending.pop(0)
                it = spec['items'][i]
                lf = open(os.path.join(u.dir, f'item_{i}.log'), 'w')
                lf.write(f'$ {it["cmd"]}\n')
                lf.flush()
                p = subprocess.Popen(it['cmd'], shell=True, cwd=ROOT, env=base_env({**(spec.get('env') or {}), **(it.get('env') or {})}),
                                     stdout=lf, stderr=subprocess.STDOUT, start_new_session=True)
                u.procs[i] = (p, lf)
                u.items[i].update(state='running', t0=now())
                with open(os.path.join(RUNS, 'pgids'), 'a') as fh:     # a later agent kills these on takeover
                    fh.write(f'{p.pid}\n')
            for i, (p, lf) in list(u.procs.items()):
                if p.poll() is not None:
                    lf.close()
                    u.items[i].update(state='exited', exit=p.returncode, t1=now())
                    del u.procs[i]
            scan(u)
            time.sleep(2)
        if u.cancelled:
            raise RuntimeError('cancelled')
        u.state = 'finishing'
        scan(u)
        time.sleep(2)
        scan(u, final=True)
        fails = [it for it in u.items if it['exit'] != 0]
        missing = u.want - u.done - set(u.bad)
        u.state = 'done'
        if fails or missing or u.bad:
            u.error = (f'{len(fails)} item(s) exited nonzero; ' if fails else '') + \
                      (f'{len(missing)} frame(s) missing; ' if missing else '') + (f'{len(u.bad)} bad frame(s)' if u.bad else '')
    except Exception as e:                          # noqa: BLE001 - the unit reports its own failure
        u.error = str(e)[:500]
        u.state = 'cancelled' if u.cancelled else 'failed'
        for i, (p, lf) in list(u.procs.items()):
            kill(p)
            lf.close()
            u.items[i].update(state='killed', exit=-9, t1=now())
        u.procs.clear()
        try:
            scan(u, final=True)                     # whatever did finish still ships
        except Exception:                           # noqa: BLE001
            pass
    for k, fn in list(u.acked):                     # sources of frames the Mac already holds: free the disk
        try:
            for p in frame_files(u.outdir[int(k)], int(fn[2:7])):
                if os.path.exists(p):
                    os.remove(p)
        except (ValueError, KeyError, OSError):
            pass
    u.t['end'] = now()


def kill(p):
    try:
        os.killpg(p.pid, signal.SIGTERM)
    except OSError:
        return
    for _ in range(20):
        if p.poll() is not None:
            return
        time.sleep(0.5)
    try:
        os.killpg(p.pid, signal.SIGKILL)
    except OSError:
        pass


def worker():
    while not STOPPING[0]:
        u = None
        with LOCK:
            for uid in ORDER:
                if UNITS[uid].state == 'queued':
                    u = UNITS[uid]
                    break
        if u is None:
            time.sleep(1)
            continue
        run_unit(u)


def busy():
    return any(u.state not in ('done', 'failed', 'cancelled') for u in UNITS.values())


def self_stop():
    try:
        req = urllib.request.Request('http://169.254.42.1/v1/stop', method='POST', data=b'',
                                     headers={'Metadata-Flavor': 'givemeanode'})
        urllib.request.urlopen(req, timeout=20).read()
    except Exception as e:                          # noqa: BLE001
        print('self-stop failed:', e, flush=True)


def watchdog():
    while not STOPPING[0]:
        time.sleep(20)
        if not busy() and now() - LAST_REQ[0] > IDLE_STOP_S:
            print(f'idle {IDLE_STOP_S}s with no unit and no request: stopping this node', flush=True)
            STOPPING[0] = True
            self_stop()
            os._exit(0)


# ------------------------------------------------------------------ HTTP API

class H(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'

    def log_message(self, fmt, *args):
        pass

    def _send(self, code, body, ctype='application/json'):
        if isinstance(body, (dict, list)):
            body = json.dumps(body).encode()
        elif isinstance(body, str):
            body = body.encode()
        self.send_response(code)
        self.send_header('Content-Type', ctype)
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(body)

    def _body(self):
        n = int(self.headers.get('Content-Length') or 0)
        return json.loads(self.rfile.read(n) or b'{}') if n else {}

    def do_GET(self):
        LAST_REQ[0] = now()
        url = urlparse(self.path)
        q = parse_qs(url.query)
        parts = [p for p in url.path.split('/') if p]
        try:
            if parts == ['status'] or not parts:
                since = int(q.get('since', ['0'])[0])
                with LOCK:
                    ready = [r for r in READY if r['seq'] > since]
                    units = [UNITS[i].public() for i in ORDER]
                self._send(200, dict(version=VERSION, up=round(now() - T0), busy=busy(), cpu_limit=CPU_LIMIT,
                                     seq=SEQ[0], units=units, ready=ready[:2000], more=len(ready) > 2000,
                                     idle_stop_s=IDLE_STOP_S))
            elif parts[0] == 'tar' and len(parts) == 2:
                u = UNITS.get(parts[1])
                if u is None:
                    return self._send(404, {'error': 'no such unit'})
                names = [n for n in ','.join(q.get('f', [''])).split(',') if n]
                buf = io.BytesIO()
                total = 0
                with tarfile.open(fileobj=buf, mode='w') as tf:
                    for n in names[:200]:
                        k, _, fn = n.partition('/')
                        p = os.path.join(u.ready_dir, os.path.basename(k), os.path.basename(fn))
                        if os.path.exists(p) and total < 48 * 2 ** 20:     # the Mac asks again for the rest
                            tf.add(p, arcname=f'{k}/{fn}')
                            total += os.path.getsize(p)
                self._send(200, buf.getvalue(), 'application/x-tar')
            elif parts[0] == 'log' and len(parts) == 3:
                u = UNITS.get(parts[1])
                n = int(q.get('n', ['60'])[0])
                p = os.path.join(RUNS, os.path.basename(parts[1]), os.path.basename(parts[2]))
                self._send(200 if u else 404, {'lines': tail(p, n)})
            else:
                self._send(404, {'error': 'unknown route'})
        except Exception as e:                      # noqa: BLE001
            self._send(500, {'error': str(e)[:300]})

    def do_POST(self):
        LAST_REQ[0] = now()
        parts = [p for p in urlparse(self.path).path.split('/') if p]
        try:
            b = self._body()
            if parts == ['unit']:
                with LOCK:
                    if b['id'] in UNITS:
                        return self._send(200, {'ok': True, 'dup': True})
                    u = Unit(b)
                    os.makedirs(u.dir, exist_ok=True)
                    with open(os.path.join(u.dir, 'spec.json'), 'w') as fh:
                        json.dump(b, fh)
                    UNITS[u.id] = u
                    ORDER.append(u.id)
                self._send(200, {'ok': True})
            elif parts == ['ack']:
                u = UNITS.get(b.get('unit'))
                if u is None:
                    return self._send(404, {'error': 'no such unit'})
                for n in b.get('files', []):
                    k, _, fn = n.partition('/')
                    try:
                        os.remove(os.path.join(u.ready_dir, os.path.basename(k), os.path.basename(fn)))
                    except OSError:
                        pass
                    u.acked.add((k, fn))
                    try:                            # the source frame has shipped: free the disk
                        f = int(fn[2:7])
                        for p in frame_files(u.outdir[int(k)], f):
                            if os.path.exists(p) and u.state in ('done', 'failed', 'cancelled'):
                                os.remove(p)
                    except (ValueError, KeyError):
                        pass
                self._send(200, {'ok': True, 'acked': len(u.acked)})
            elif parts == ['cancel']:
                u = UNITS.get(b.get('unit'))
                if u is None:
                    return self._send(404, {'error': 'no such unit'})
                u.cancelled = True
                for p, _ in list(u.procs.values()):
                    threading.Thread(target=kill, args=(p,), daemon=True).start()
                self._send(200, {'ok': True})
            elif parts == ['quit']:
                self._send(200, {'ok': True})
                STOPPING[0] = True
                for u in UNITS.values():
                    u.cancelled = True
                    for p, _ in list(u.procs.values()):
                        kill(p)
                if b.get('stop'):
                    self_stop()
                threading.Thread(target=lambda: (time.sleep(1), os._exit(0)), daemon=True).start()
            else:
                self._send(404, {'error': 'unknown route'})
        except Exception as e:                      # noqa: BLE001
            self._send(500, {'error': str(e)[:300]})


def main():
    if len(sys.argv) < 2 or sys.argv[1] != 'agent':
        print(__doc__)
        sys.exit(2)
    port = int(sys.argv[sys.argv.index('--port') + 1]) if '--port' in sys.argv else 8700
    os.makedirs(RUNS, exist_ok=True)
    srv = ThreadingHTTPServer(('0.0.0.0', port), H)
    srv.daemon_threads = True
    threading.Thread(target=worker, daemon=True).start()
    threading.Thread(target=watchdog, daemon=True).start()
    print(f'farm agent v{VERSION} on :{port} (cpu_limit {CPU_LIMIT}, idle stop {IDLE_STOP_S}s)', flush=True)
    srv.serve_forever()


if __name__ == '__main__':
    main()
