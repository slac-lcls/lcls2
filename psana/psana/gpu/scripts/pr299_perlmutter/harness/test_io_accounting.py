import builtins
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from io_accounting import ReaderIoAccounting, observe_reader_io, save_io_evidence, validate_io
import worker


def reader_class():
    class Reader:
        def __init__(self):
            self.stats = dict(total_bytes=0, total_requests=0, requested_bytes=0,
                              useful_bytes=0, bulk_read=True, compat_mode=True,
                              io_path='CPU-fallback')
            self.pending = False

        def io_stats(self):
            return dict(self.stats)

        def reset_io_stats(self):
            if self.pending:
                raise RuntimeError('pending reads')
            for field in ('total_bytes', 'total_requests', 'requested_bytes', 'useful_bytes'):
                self.stats[field] = 0

        def complete(self, size=100):
            self.stats['total_bytes'] += size
            self.stats['total_requests'] += 1
            self.stats['requested_bytes'] += size
            self.stats['useful_bytes'] += size

    return Reader


class AccountingTests(unittest.TestCase):
    def setUp(self):
        self.reader_type = reader_class()
        self.original_reset = self.reader_type.reset_io_stats
        self.accounting = ReaderIoAccounting(self.reader_type)
        self.addCleanup(self.accounting.finish)
        self.reader = self.reader_type()

    def test_reset_before_first_delivery_and_empty_tail(self):
        self.reader.reset_io_stats()
        self.reader.complete()
        self.reader.reset_io_stats()
        self.assertEqual(self.reader.io_stats()['total_bytes'], 0)
        for unused in range(3):
            self.reader.reset_io_stats()
        summary = self.accounting.finish(self.reader)
        validate_io(summary)
        self.assertEqual(summary['total_bytes'], 100)
        self.assertEqual(summary['total_requests'], 1)
        self.assertEqual(summary['positive_windows'], 1)
        self.assertEqual(summary['reset_windows'], 5)
        self.assertEqual(summary['tail_windows'], 1)

    def test_final_window_included_once(self):
        self.reader.complete()
        self.reader.reset_io_stats()
        self.reader.complete(250)
        summary = self.accounting.finish(self.reader)
        self.assertEqual(summary['total_bytes'], 350)
        self.assertEqual(summary['total_requests'], 2)
        self.assertEqual(summary, self.accounting.finish(self.reader))
        self.assertIs(self.reader_type.reset_io_stats, self.original_reset)

    def test_no_reset_still_counts_tail(self):
        self.reader.complete()
        summary = self.accounting.finish(self.reader)
        validate_io(summary)
        self.assertEqual(summary['total_bytes'], 100)
        self.assertEqual(summary['reset_windows'], 0)

    def test_multiple_readers_retained_until_finish(self):
        second = self.reader_type()
        self.reader.complete()
        self.reader.reset_io_stats()
        second.complete(250)
        second.reset_io_stats()
        second.complete(50)
        summary = self.accounting.finish(self.reader)
        self.assertEqual(summary['total_bytes'], 400)
        self.assertEqual(summary['readers'], 2)
        self.assertEqual(summary['tail_windows'], 2)

    def test_failed_reset_does_not_double_count(self):
        self.reader.complete()
        self.reader.pending = True
        with self.assertRaisesRegex(RuntimeError, 'pending reads'):
            self.reader.reset_io_stats()
        summary = self.accounting.finish()
        self.assertEqual(summary['reset_windows'], 0)
        self.assertEqual(summary['total_bytes'], 100)
        self.assertIs(self.reader_type.reset_io_stats, self.original_reset)

    def test_no_reads_rejected_with_saved_evidence(self):
        self.reader.reset_io_stats()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'io.json'
            summary = save_io_evidence(path, self.accounting, self.reader, delivered_events=120)
            with self.assertRaisesRegex(RuntimeError, 'bulk reads'):
                validate_io(summary)
            self.assertEqual(json.loads(path.read_text())['cumulative_io'], summary)

    def test_missing_reader_rejected(self):
        with self.assertRaisesRegex(RuntimeError, 'CPU-fallback'):
            validate_io(self.accounting.finish())

    def test_bad_modes_in_earlier_window_are_not_hidden(self):
        for field, value in [('bulk_read', False), ('compat_mode', False), ('io_path', 'GDS')]:
            with self.subTest(field=field):
                reader_type = reader_class()
                accounting = ReaderIoAccounting(reader_type)
                reader = reader_type()
                original_value = reader.stats[field]
                reader.stats[field] = value
                reader.complete()
                reader.reset_io_stats()
                reader.stats[field] = original_value
                with self.assertRaises(RuntimeError):
                    validate_io(accounting.finish(reader))

    def test_restores_patch_if_snapshot_fails(self):
        with mock.patch.object(self.reader, 'io_stats', side_effect=RuntimeError('stats failed')):
            with self.assertRaisesRegex(RuntimeError, 'stats failed'):
                self.accounting.finish(self.reader)
        self.assertIs(self.reader_type.reset_io_stats, self.original_reset)

    def test_cpu_observer_does_not_import_gpu(self):
        with mock.patch.object(builtins, '__import__', side_effect=AssertionError('unexpected import')):
            with observe_reader_io(False) as accounting:
                self.assertIsNone(accounting)

    def test_context_restores_after_exception(self):
        module = SimpleNamespace(KvikioGpuReader=reader_class())
        original = module.KvikioGpuReader.reset_io_stats
        with mock.patch.dict('sys.modules', {'psana.gpu.gpu_kvikio_read': module}):
            with self.assertRaisesRegex(RuntimeError, 'worker failed'):
                with observe_reader_io(True) as accounting:
                    reader = module.KvikioGpuReader()
                    reader.reset_io_stats()
                    reader.complete()
                    raise RuntimeError('worker failed')
        self.assertIs(module.KvikioGpuReader.reset_io_stats, original)
        self.assertEqual(accounting.finish()['total_bytes'], 100)


class WorkerEvidenceTests(unittest.TestCase):
    def exercise(self, iterator, fail=False, empty=False):
        reader_type = reader_class()
        accounting = ReaderIoAccounting(reader_type)
        self.addCleanup(accounting.finish)
        reader = reader_type()

        def events():
            reader.reset_io_stats()
            if not empty:
                reader.complete()
            reader.reset_io_stats()
            if fail:
                raise RuntimeError('iteration failed')
            yield SimpleNamespace(timestamp=42)
            reader.reset_io_stats()

        run = SimpleNamespace(
            comms=SimpleNamespace(node_type=lambda: 'bd'),
            dsparms=SimpleNamespace(n_gpu_streams=1, gpu_memory_budget_gb=8),
            _gpu_manager_for_placement=SimpleNamespace(gpu_reader=reader),
            events=events, steps=lambda: iter([SimpleNamespace(events=events)]),
            close_shared_memory=lambda: None)
        datasource = SimpleNamespace(runs=lambda: iter([run]))
        comm = mock.Mock()
        comm.Get_rank.return_value = 2
        gpu = SimpleNamespace(cuda=SimpleNamespace(runtime=SimpleNamespace(deviceSynchronize=lambda: None)))
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'worker.json'
            network = Path(directory) / 'network.json'
            network.write_text('{}')
            args = SimpleNamespace(mode='gpu', iterator=iterator, output=str(output),
                                   nic_inventory=str(network), check_input=False)
            with mock.patch.object(worker, 'runtime_provenance', return_value={'mock':True}), mock.patch.dict('os.environ', {'SLURM_LOCALID':'2','SCREEN_GPU_UUID':'GPU-test'}), mock.patch.dict('sys.modules', {'psana': SimpleNamespace(DataSource=lambda **options: datasource), 'mpi4py': SimpleNamespace(MPI=SimpleNamespace(Get_library_version=lambda: 'CRAY MPICH')), 'gpu_identity': SimpleNamespace(cuda_identity=lambda: {'uuid':'GPU-test'})}):
                if fail or empty:
                    with self.assertRaisesRegex(RuntimeError, 'iteration failed' if fail else 'bulk reads'):
                        worker.execute_worker(args, comm, [[], [], []], {}, gpu, {}, [], accounting)
                else:
                    worker.execute_worker(args, comm, [[], [], []], {}, gpu, {}, [], accounting)
                    row = comm.gather.call_args.args[0]
                    self.assertEqual(row['first_batch_io']['total_bytes'], 0)
                    self.assertEqual(row['last_batch_io']['total_bytes'], 0)
                    self.assertEqual(row['cumulative_io']['total_bytes'], 100)
            evidence = json.loads((Path(directory) / 'io-rank-2.json').read_text())
            self.assertEqual(evidence['loop_complete'], not fail)
            self.assertEqual(evidence['cumulative_io']['total_bytes'], 0 if empty else 100)
            self.assertIs(reader_type.reset_io_stats, accounting.original_reset)

    def test_worker_events(self):
        self.exercise('events')

    def test_worker_steps(self):
        self.exercise('steps')

    def test_worker_retains_evidence_on_iteration_failure(self):
        self.exercise('events', fail=True)

    def test_worker_retains_evidence_on_validation_failure(self):
        self.exercise('events', empty=True)


if __name__ == '__main__':
    unittest.main()
