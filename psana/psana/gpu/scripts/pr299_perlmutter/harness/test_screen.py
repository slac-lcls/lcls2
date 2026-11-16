import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from common import write_json
import screen


class ScreenTests(unittest.TestCase):
    def test_gpu_discovery_isolated_with_unchanged_environment(self):
        identity = dict(complete=True, gpu_uuid='GPU-test', gpu_lines=['GPU-test, A100, 40960'])
        response = screen.subprocess.CompletedProcess([], 0, json.dumps(identity), '')
        with tempfile.TemporaryDirectory() as directory, \
                mock.patch.dict(screen.os.environ, {'SLURM_JOB_GPUS': '2', 'CUDA_VISIBLE_DEVICES': '0'}), \
                mock.patch.object(screen.subprocess, 'run', return_value=response) as query:
            path = Path(directory) / 'gpu.json'
            self.assertEqual(screen.assigned_gpu(path), ('GPU-test', identity['gpu_lines']))
            self.assertEqual(query.call_args.args[0],
                             [screen.sys.executable, '-B', str(screen.ROOT / 'harness/gpu_identity.py')])
            self.assertNotIn('env', query.call_args.kwargs)
            self.assertEqual(screen.os.environ['CUDA_VISIBLE_DEVICES'], '0')
            self.assertEqual(json.loads(path.read_text())['identity'], identity)
            with mock.patch.dict(screen.os.environ, {'LD_LIBRARY_PATH': '/runtime/lib'}):
                environment, install = screen.environment('head', [[0]], 'GPU-test')
                self.assertEqual(environment['SCREEN_GPU_UUID'], 'GPU-test')

    def test_helper_failures_retain_diagnostics(self):
        responses = [screen.subprocess.CompletedProcess([], 1, json.dumps(dict(complete=False, error='CUDA error 100')), 'driver stderr'),
                     screen.subprocess.CompletedProcess([], -11, '', 'crashed'),
                     screen.subprocess.CompletedProcess([], 0, 'not json', 'bad output')]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'gpu.json'
            for response in responses:
                with mock.patch.object(screen.subprocess, 'run', return_value=response):
                    with self.assertRaises((RuntimeError, ValueError)):
                        screen.assigned_gpu(path)
                evidence = json.loads(path.read_text())
                self.assertEqual(evidence['returncode'], response.returncode)
                self.assertEqual(evidence['stdout'], response.stdout)
                self.assertEqual(evidence['stderr'], response.stderr)

    def test_helper_timeout_retains_diagnostics(self):
        with tempfile.TemporaryDirectory() as directory, \
                mock.patch.object(screen.subprocess, 'run', side_effect=screen.subprocess.TimeoutExpired(
                    ['helper'], 45, output=b'partial', stderr=b'driver stderr')):
            path = Path(directory) / 'gpu.json'
            with self.assertRaisesRegex(RuntimeError, 'timed out'):
                screen.assigned_gpu(path)
            evidence = json.loads(path.read_text())
            self.assertTrue(evidence['timeout'])
            self.assertIn('partial', evidence['stdout'])
            self.assertIn('driver stderr', evidence['stderr'])

    def test_pair(self):
        left = dict(case='case', label='A', timestamp_sha256='same', events=10000,
                    events_per_second=100, loop_seconds=100)
        right = dict(left, label='B', events_per_second=110, loop_seconds=10000 / 110)
        result = screen.paired_result(left, right)
        self.assertAlmostEqual(result['throughput_change_percent'], 10)
        self.assertLess(result['loop_change_percent'], 0)
        for wrong in (dict(right, events=9999), dict(right, timestamp_sha256='other')):
            with self.assertRaises(RuntimeError):
                screen.paired_result(left, wrong)

    def test_gpu_samples(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'gpu.csv'
            path.write_text('time, GPU-test, 1024, 30, 40\ntime, GPU-test, 2048, 50, 42\n')
            self.assertEqual(screen.gpu_samples(path)['peak_used_mib'], 2048)
            for invalid in ('', 'no GPU\n', 'time,GPU-test,N/A,50,42\n'):
                path.write_text(invalid)
                with self.assertRaises((RuntimeError, ValueError)):
                    screen.gpu_samples(path)

    def test_harness_change_rejected(self):
        with mock.patch.object(screen, 'tree_hashes', return_value={'worker.py': 'new'}):
            with self.assertRaisesRegex(RuntimeError, 'harness changed'):
                screen.check_frozen({'harness': {'worker.py': 'reviewed'}})

    def test_install_change_rejected(self):
        with mock.patch.object(screen, 'tree_hashes', side_effect=[{}, {'library': 'new'}]):
            with self.assertRaisesRegex(RuntimeError, 'candidate install changed'):
                screen.check_frozen({'harness': {}, 'head_install': {'library': 'reviewed'}})

    def test_missing_review_stops_before_execution(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_json(root / 'manifest.json', {})
            with mock.patch.object(screen, 'ROOT', root), mock.patch.object(screen, 'run_command') as command:
                with self.assertRaises(FileNotFoundError):
                    screen.execute()
                command.assert_not_called()

    def test_json_roundtrip(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'roundtrip.json'
            value = dict(events=[[1, 0], [2, 1]])
            write_json(path, value)
            self.assertEqual(json.loads(path.read_text()), value)
            self.assertFalse(path.with_suffix('.json.tmp').exists())


if __name__ == '__main__':
    unittest.main()
