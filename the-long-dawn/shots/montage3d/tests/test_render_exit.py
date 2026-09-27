"""Exercise the real driver's process/thread contract without Blender or image dependencies.

    python3 -m unittest discover -s shots/montage3d/tests -p 'test_render_exit.py'

MONTAGE_DRIVER_UNDER_TEST can point at an earlier render.py to negative-test these regressions.
Only the shot/image work is substituted; the driver launches a real child process and its real poster thread.
"""
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import unittest


DRIVER = Path(os.environ.get('MONTAGE_DRIVER_UNDER_TEST') or Path(__file__).resolve().parents[1] / 'render.py')

RUNNER = r'''
import importlib.util, os, pathlib, sys, types
driver, root, case = sys.argv[1:]
for name in ('numpy', 'exr', 'fireparts', 'look'):
    sys.modules[name] = types.ModuleType(name)
shot = types.ModuleType('contract_shot')
shot.START, shot.END, shot.SAMPLES = 10, 11, 1
shot.flame_specs = lambda: []
shot.timing = lambda frames: {}
sys.modules[shot.__name__] = shot
spec = importlib.util.spec_from_file_location('driver_under_test', driver)
render = importlib.util.module_from_spec(spec)
spec.loader.exec_module(render)
render.HERE = render.ROOT = root
render.FINAL_DIR = str(pathlib.Path(root) / 'out')
render.LOCK = str(pathlib.Path(root) / 'unowned.lock')
render.BLENDER = sys.executable
def post(shot, frame, exr_dir, out_dir, scene, keep_exr):
    if case == 'post_error' and frame == 11:
        raise RuntimeError('injected post failure')
    if case == 'post_exit':
        raise SystemExit(0)
    pathlib.Path(out_dir, f'f_{frame - render.OUT_OFFSET:05d}.png').write_text('fresh')
render.post_frame = post
if case == 'queue_draining':
    import threading
    poster_released = threading.Event()
    producer_done = threading.Event()
    main_thread = threading.current_thread()
    class ScheduledLock:
        def __init__(self):
            self.lock = threading.Lock()
            self.paused = False
        def __enter__(self):
            self.lock.acquire()
            return self
        def __exit__(self, *exc):
            self.lock.release()
            if threading.current_thread() is not main_thread and not self.paused:
                self.paused = True
                poster_released.set()
                if not producer_done.wait(5):
                    raise RuntimeError('producer did not finish during scheduled pause')
    class ScheduledThread(threading.Thread):
        def start(self):
            super().start()
            if not poster_released.wait(5):
                raise RuntimeError('poster did not reach scheduled pause')
        def join(self, *args, **kwargs):
            producer_done.set()
            return super().join(*args, **kwargs)
    # Force the consumer to yield after its first empty-queue inspection; the producer
    # then queues every frame and finishes before the consumer resumes. No wall-clock race.
    render.threading = types.SimpleNamespace(Lock=ScheduledLock, Thread=ScheduledThread, Event=threading.Event)
sys.argv = ['render.py', 'contract_shot', '--final', '--force', '--frames', '10,11', '--out-offset', '10']
render.main()
'''

CHILD = r'''
import json, os, pathlib, sys, time
job = json.load(open(sys.argv[-1]))
root = pathlib.Path(os.environ['CONTRACT_ROOT'])
root.joinpath('child.pid').write_text(str(os.getpid()))
pathlib.Path(job['exr_dir'], 'scene.json').write_text('{}')
case = os.environ['CONTRACT_CASE']
if case == 'child_error':
    sys.exit(7)
if case == 'malformed_frame':
    print('FRAME invalid invalid', flush=True)
    time.sleep(30)
    sys.exit(0)
frames = job['frames']
if case in ('partial', 'partial_error'):
    frames = frames[:1]
elif case == 'stale':
    frames = []
elif case == 'unexpected':
    frames = frames + [99]
for frame in frames:
    print(f'FRAME {frame} 0.01s', flush=True)
sys.exit(7 if case == 'partial_error' else 0)
'''


class RenderExitTests(unittest.TestCase):
    def run_driver(self, case, stale=False):
        with tempfile.TemporaryDirectory(prefix='montage-contract-') as directory:
            root = Path(directory)
            root.joinpath('bl_main.py').write_text(CHILD)
            root.joinpath('unowned.lock').write_text('other renderer')
            if stale:
                root.joinpath('out').mkdir()
                for frame in (0, 1):
                    root.joinpath('out', f'f_{frame:05d}.png').write_text('stale')
            env = dict(os.environ, MT3D_BLENDER=sys.executable, CONTRACT_ROOT=directory, CONTRACT_CASE=case)
            proc = subprocess.Popen([sys.executable, '-c', RUNNER, str(DRIVER), directory, case],
                                    env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                    start_new_session=True)
            try:
                output, _ = proc.communicate(timeout=15)
                pid_file = root / 'child.pid'
                child_alive = False
                if pid_file.exists():
                    try:
                        os.kill(int(pid_file.read_text()), 0)
                        child_alive = True
                    except ProcessLookupError:
                        pass
                frames = {p.name: p.read_text() for p in root.glob('out/f_*.png')}
                return proc.returncode, output, frames, child_alive, root.joinpath('unowned.lock').exists()
            finally:
                # The old driver can orphan its child; the negative test must still clean it up.
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                proc.wait()

    def assert_failure(self, case, stale=False):
        code, output, frames, child_alive, _ = self.run_driver(case, stale=stale)
        self.assertNotEqual(code, 0, output)
        self.assertFalse(child_alive, 'driver exited without reaping its Blender child')
        return output, frames

    def test_child_failure_is_nonzero(self):
        output, _ = self.assert_failure('child_error')
        self.assertIn('exit 7', output)

    def test_partial_child_failure_keeps_completed_frames(self):
        _, frames = self.assert_failure('partial_error')
        self.assertEqual(frames, {'f_00000.png': 'fresh'})

    def test_successful_child_with_missing_frame_fails(self):
        self.assert_failure('partial')

    def test_stale_outputs_do_not_count_as_this_runs_frames(self):
        self.assert_failure('stale', stale=True)

    def test_post_failure_is_nonzero_even_with_old_output(self):
        output, frames = self.assert_failure('post_error', stale=True)
        self.assertIn('POST FAILED 11', output)
        self.assertEqual(frames['f_00000.png'], 'fresh')
        self.assertEqual(frames['f_00001.png'], 'stale')

    def test_worker_system_exit_is_nonzero(self):
        self.assert_failure('post_exit')

    def test_unexpected_frame_fails(self):
        self.assert_failure('unexpected')

    def test_malformed_notification_reaps_child(self):
        self.assert_failure('malformed_frame')

    def test_complete_render_succeeds(self):
        code, output, frames, child_alive, _ = self.run_driver('complete')
        self.assertEqual(code, 0, output)
        self.assertEqual(frames, {'f_00000.png': 'fresh', 'f_00001.png': 'fresh'})
        self.assertFalse(child_alive)

    def test_poster_drains_frames_queued_during_its_idle_yield(self):
        code, output, frames, child_alive, _ = self.run_driver('queue_draining')
        self.assertEqual(code, 0, output)
        self.assertEqual(frames, {'f_00000.png': 'fresh', 'f_00001.png': 'fresh'})
        self.assertFalse(child_alive)

    def test_farm_driver_preserves_other_renderers_lock(self):
        _, _, _, _, lock_exists = self.run_driver('complete')
        self.assertTrue(lock_exists)


if __name__ == '__main__':
    unittest.main()
