import ctypes
import json
import unittest
from unittest import mock

import gpu_identity


GPU_UUID = 'GPU-c66a5240-c553-4f3b-8faf-888aadef5938'
OTHER_UUID = 'GPU-11111111-2222-3333-4444-555555555555'


class GPUIdentityTests(unittest.TestCase):
    def setUp(self):
        self.environment = dict(SLURM_JOB_ID='123', SLURM_JOB_GPUS='2', CUDA_VISIBLE_DEVICES='0')
        self.identity = dict(count=1, ordinal=0, uuid=GPU_UUID, pci_bus_id='0000:C1:00.0')
        self.row = f'{GPU_UUID}, 00000000:C1:00.0, NVIDIA A100-SXM4-40GB, 40960, Disabled\n'

    def discover(self, row=None, code=0, environment=None, identity=None):
        record = dict(environment=self.environment if environment is None else environment)
        response = gpu_identity.subprocess.CompletedProcess([], code, self.row if row is None else row, 'stderr evidence')
        with mock.patch.object(gpu_identity, 'cuda_identity', return_value=self.identity if identity is None else identity), \
                mock.patch.object(gpu_identity.subprocess, 'run', return_value=response) as query:
            gpu_identity.discover(record)
        return record, query

    def test_global_two_visible_zero_queries_cuda_uuid(self):
        record, query = self.discover()
        self.assertEqual(record['gpu_uuid'], GPU_UUID)
        self.assertEqual(query.call_args.args[0][2], GPU_UUID)
        self.assertNotIn('env', query.call_args.kwargs)

    def test_step_assignment_without_job_assignment(self):
        environment = dict(SLURM_JOB_ID='123', SLURM_STEP_GPUS='2', CUDA_VISIBLE_DEVICES='0')
        self.assertEqual(self.discover(environment=environment)[0]['gpu_uuid'], GPU_UUID)

    def test_uuid_visibility_and_assignment(self):
        environment = dict(self.environment, SLURM_JOB_GPUS=GPU_UUID, CUDA_VISIBLE_DEVICES=GPU_UUID)
        self.assertEqual(self.discover(environment=environment)[0]['gpu_uuid'], GPU_UUID)

    def test_missing_multiple_malformed_and_mig_visibility(self):
        for variable in ('SLURM_JOB_GPUS', 'CUDA_VISIBLE_DEVICES'):
            for value in (None, '', '0,1', '0-1', '-1', 'MIG-test', 'GPU-test', ' 0'):
                with self.subTest(variable=variable, value=value), \
                        mock.patch.object(gpu_identity, 'cuda_identity') as query:
                    with self.assertRaisesRegex(RuntimeError, 'exactly one full GPU'):
                        gpu_identity.discover(dict(environment=dict(self.environment, **{variable: value})))
                    query.assert_not_called()

    def test_missing_job_rejected(self):
        with self.assertRaises(RuntimeError):
            self.discover(environment=dict(self.environment, SLURM_JOB_ID=None))

    def test_explicit_uuid_mismatch(self):
        for variable in ('SLURM_JOB_GPUS', 'SLURM_STEP_GPUS', 'CUDA_VISIBLE_DEVICES'):
            with self.subTest(variable=variable), self.assertRaisesRegex(RuntimeError, 'differs'):
                self.discover(environment=dict(self.environment, **{variable: OTHER_UUID}))

    def test_invalid_cuda_identity(self):
        for identity in (dict(self.identity, count=0), dict(self.identity, count=2),
                         dict(self.identity, uuid='MIG-test'), dict(self.identity, uuid='GPU-bad')):
            with self.subTest(identity=identity), self.assertRaises(RuntimeError):
                self.discover(identity=identity)

    def test_wrong_device_mig_or_incomplete_row(self):
        for row in ('', self.row + self.row, self.row.replace('40960', '163840'),
                    self.row.replace('40960', 'nan'), self.row.replace('A100', 'V100'),
                    self.row.replace('Disabled', 'Enabled'), self.row.replace('Disabled', 'N/A'),
                    self.row.replace(GPU_UUID, OTHER_UUID), self.row.replace('C1:00', '81:00'),
                    self.row.replace('00000000:C1:00.0', 'bad-bus'), self.row.replace(', Disabled', '')):
            with self.subTest(row=row), self.assertRaises((RuntimeError, ValueError)):
                self.discover(row=row)

    def test_failed_query_retains_command_and_output(self):
        record = dict(environment=self.environment)
        response = gpu_identity.subprocess.CompletedProcess([], 6, 'No devices found', 'NVML failure')
        with mock.patch.object(gpu_identity, 'cuda_identity', return_value=self.identity), \
                mock.patch.object(gpu_identity.subprocess, 'run', return_value=response):
            with self.assertRaisesRegex(RuntimeError, 'failed \(6\)'):
                gpu_identity.discover(record)
        self.assertEqual(record['nvidia_smi']['returncode'], 6)
        self.assertEqual(record['nvidia_smi']['stdout'], 'No devices found')
        self.assertEqual(record['nvidia_smi']['stderr'], 'NVML failure')

    def test_query_timeout_retains_output(self):
        record = dict(environment=self.environment)
        with mock.patch.object(gpu_identity, 'cuda_identity', return_value=self.identity), \
                mock.patch.object(gpu_identity.subprocess, 'run', side_effect=gpu_identity.subprocess.TimeoutExpired(
                    ['nvidia-smi'], 20, output=b'partial', stderr=b'NVML stderr')):
            with self.assertRaisesRegex(RuntimeError, 'timed out'):
                gpu_identity.discover(record)
        self.assertTrue(record['nvidia_smi']['timeout'])
        self.assertEqual(record['nvidia_smi']['stderr'], 'NVML stderr')

    def test_cuda_error_status(self):
        driver = mock.Mock()
        driver.cuInit.return_value = 100
        with mock.patch.object(ctypes, 'CDLL', return_value=driver):
            with self.assertRaisesRegex(RuntimeError, 'cuInit returned CUDA error 100'):
                gpu_identity.cuda_identity()
        driver.cuDeviceGetCount.assert_not_called()

    def test_cuda_actual_device_count_must_be_one(self):
        for count in (0, 2):
            driver = mock.Mock()
            driver.cuInit.return_value = 0

            def set_count(pointer):
                ctypes.cast(pointer, ctypes.POINTER(ctypes.c_int))[0] = count
                return 0

            driver.cuDeviceGetCount.side_effect = set_count
            with mock.patch.object(ctypes, 'CDLL', return_value=driver):
                with self.assertRaisesRegex(RuntimeError, f'got {count}'):
                    gpu_identity.cuda_identity()
            driver.cuDeviceGet.assert_not_called()

    def test_main_retains_driver_failure(self):
        with mock.patch.dict(gpu_identity.os.environ, self.environment, clear=True), \
                mock.patch.object(gpu_identity, 'cuda_identity', side_effect=OSError('missing driver')), \
                mock.patch('builtins.print') as output:
            self.assertEqual(gpu_identity.main(), 1)
        record = json.loads(output.call_args.args[0])
        self.assertFalse(record['complete'])
        self.assertIn('missing driver', record['error'])
        self.assertEqual(record['environment']['CUDA_VISIBLE_DEVICES'], '0')


if __name__ == '__main__':
    unittest.main()
