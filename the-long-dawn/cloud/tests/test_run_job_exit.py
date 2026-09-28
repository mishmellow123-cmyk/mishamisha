"""Exercise runner completion with simulated renderers/uploads; no cloud or image imports."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock


RUNNER_PATH = Path(__file__).resolve().parents[1] / 'run_job.py'
SPEC = importlib.util.spec_from_file_location('run_job_exit_test_target', RUNNER_PATH)
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


class SimulatedRenderer:
    def __init__(self, exit_code, running_polls=1):
        self.exit_code = exit_code
        self.running_polls = running_polls
        self.poll_count = 0
        self.returncode = None

    def poll(self):
        self.poll_count += 1
        if self.poll_count > self.running_polls:
            self.returncode = self.exit_code
        return self.returncode


class RunJobExitTests(unittest.TestCase):
    def run_job(self, codes, present=(10, 11), running_polls=None):
        processes = [SimulatedRenderer(code, (running_polls or [1] * len(codes))[i])
                     for i, code in enumerate(codes)]
        statuses, pushes, upload_poll_counts = [], [], []
        stdout = io.StringIO()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / 'renders/exit-test'
            output.mkdir(parents=True)
            for frame in present:
                # Decode is simulated below; no real image/renderer dependency is needed.
                (output / f'f_{frame:05d}.png').write_bytes(b'synthetic frame fixture')
            job = dict(name='exit-test', branch='codex/test-render-exit',
                       out_dir='renders/exit-test', frames='10-11', shape=[2, 3],
                       render=[f'simulated-render-{i}' for i in range(len(codes))])
            job_path = root / 'job.json'
            job_path.write_text(json.dumps(job))

            def upload(paths, branch, total, done):
                pushes.append((list(paths), total, done))
                for path in paths:
                    if path.endswith('_status.txt'):
                        statuses.append((root / path).read_text())
                if any(path.endswith('.png') for path in paths):
                    upload_poll_counts.append([p.poll_count for p in processes])
                return True

            fake_cv2 = SimpleNamespace(imread=mock.Mock(return_value=SimpleNamespace(shape=(2, 3, 3))))
            with contextlib.ExitStack() as stack:
                stack.enter_context(mock.patch.object(RUNNER, 'HERE', str(root)))
                stack.enter_context(mock.patch.object(RUNNER, 'LOG', []))
                stack.enter_context(mock.patch.object(sys, 'argv', [str(RUNNER_PATH), str(job_path)]))
                stack.enter_context(mock.patch.dict(sys.modules, {'cv2': fake_cv2}))
                popen = stack.enter_context(mock.patch.object(RUNNER.subprocess, 'Popen', side_effect=processes))
                stack.enter_context(mock.patch.object(RUNNER, 'git_push', side_effect=upload))
                stack.enter_context(mock.patch.object(RUNNER, 'sh', side_effect=AssertionError('unexpected shell call')))
                stack.enter_context(mock.patch.object(RUNNER.time, 'sleep'))
                stack.enter_context(contextlib.redirect_stdout(stdout))
                code = 0
                try:
                    RUNNER.main()
                except SystemExit as exc:
                    code = exc.code
                closed = [call.kwargs['stdout'].closed for call in popen.call_args_list]
        return SimpleNamespace(code=code, text=stdout.getvalue(), statuses=statuses,
                               pushes=pushes, processes=processes, closed=closed,
                               upload_poll_counts=upload_poll_counts)

    def test_nonzero_exit_fails_even_after_all_frames_were_pushed(self):
        for exit_code in (7, -9):
            with self.subTest(exit_code=exit_code):
                result = self.run_job([exit_code])
                self.assertEqual(result.code, 1)
                self.assertIn('pushed 2 (total 2/2;', result.text)
                self.assertEqual(result.upload_poll_counts, [[1]])
                self.assertEqual(result.processes[0].poll_count, 2)
                self.assertIn(f'ERROR render exited {exit_code}: simulated-render-0', result.text)
                self.assertIn('JOB ENDED', result.text)
                self.assertNotIn('JOB COMPLETE', result.text)
                self.assertIn(f'ERROR render exited {exit_code}', result.statuses[-1])
                self.assertIn('JOB ENDED', result.statuses[-1])
                self.assertEqual(result.pushes[-1][1:], (2, 2))
                self.assertEqual(result.closed, [True])

    def test_one_failed_renderer_fails_after_all_concurrent_renderers_finish(self):
        result = self.run_job([8, 0], running_polls=[1, 2])
        self.assertEqual(result.code, 1)
        self.assertIn('pushed 2 (total 2/2;', result.text)
        self.assertIn('ERROR render exited 8: simulated-render-0', result.text)
        self.assertNotIn('ERROR render exited 0', result.text)
        self.assertEqual([p.poll_count for p in result.processes], [3, 3])
        self.assertEqual(result.closed, [True, True])
        self.assertNotIn('JOB COMPLETE', result.text)

    def test_successful_renderers_and_all_frames_complete(self):
        result = self.run_job([0, 0])
        self.assertEqual(result.code, 0)
        self.assertIn('JOB COMPLETE: 2 frames pushed', result.text)
        self.assertNotIn('JOB ENDED', result.text)
        self.assertNotIn('ERROR render', result.text)
        self.assertEqual(result.upload_poll_counts, [[1, 1]])
        self.assertEqual(result.closed, [True, True])

    def test_missing_frame_still_fails_after_successful_renderer(self):
        result = self.run_job([0], present=(10,))
        self.assertEqual(result.code, 1)
        self.assertIn('JOB ENDED (missing 1:', result.text)
        self.assertIn('exit-test:11', result.text)
        self.assertNotIn('JOB COMPLETE', result.text)
        self.assertIn('JOB ENDED (missing 1:', result.statuses[-1])
        self.assertEqual(result.closed, [True])

    def test_failed_renderer_and_missing_frame_both_remain_visible(self):
        result = self.run_job([4], present=(10,))
        self.assertEqual(result.code, 1)
        self.assertIn('ERROR render exited 4: simulated-render-0', result.text)
        self.assertIn('JOB ENDED (missing 1:', result.text)
        self.assertIn('exit-test:11', result.text)
        self.assertIn('ERROR render exited 4', result.statuses[-1])
        self.assertIn('JOB ENDED (missing 1:', result.statuses[-1])
        self.assertNotIn('JOB COMPLETE', result.text)
        self.assertEqual(result.closed, [True])


if __name__ == '__main__':
    unittest.main()
