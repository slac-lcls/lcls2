"""Offline adapter-contract tests; no psana, MPI launch or CUDA initialization."""
import copy
import importlib.util
import json
import mmap
import os
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch, Mock

import campaign
import native_contract as n
import native_worker as w
import perlmutter_adapter as a


# Load the real classifier without importing psana's MPI/native package setup.
_spec = importlib.util.spec_from_file_location(
    'regression_transitionid', Path(__file__).resolve().parents[4] / 'psexp' / 'transitionid.py')
_transitionid = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_transitionid)
TransitionId = _transitionid.TransitionId


class NativeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.config = n.read(Path(__file__).with_name('profile.json'))
        self.config['dataset'].update(events=4, event_sha256=n.digest([100, 101, 102, 103]), requested_bytes=2000)
        self.config['warmup']['events'] = 2
        self.config['dataset']['sites']['perlmutter'] = str(self.root)
        self.config['runtimes'] = {k: dict(commit='a'*40, install='/install') for k in ('baseline', 'candidate')}
        self.reference = dict(schema='run51-raw-reference-v1', events=[[100+i, 0, 500] for i in range(4)])
        self.add_artifact('event_reference', self.reference)
        files = []
        for stream in (3, 5, 7, 8, 9):
            row = dict(stream=stream)
            for key, suffix in (('bigdata', '.xtc2'), ('smd', '.smd.xtc2')):
                parent = self.root / 'smalldata' if key == 'smd' else self.root
                parent.mkdir(exist_ok=True)
                path = parent / ('mfx100848724-r0051-s%03d-c000%s' % (stream, suffix))
                path.write_bytes(b'x' * 500)
                st = path.stat()
                entry = dict(path=str(path), size=st.st_size, mtime_ns=st.st_mtime_ns, device=st.st_dev, inode=st.st_ino, sha256=n.file_hash(path))
                if key == 'bigdata':
                    entry.update(range_start=100, range_bytes=400)
                row[key] = entry
            files.append(row)
        self.manifest = dict(schema='run51-private-input-v1', private_copy=True, root=str(self.root), files=files,
                             event_reference_sha256=self.config['artifacts']['event_reference']['sha256'])
        self.add_artifact('dataset_manifest', self.manifest)
        self.site = dict(schema='perlmutter-raw-site-v1', job_id='123', node=os.uname().nodename,
                         gpu_uuid='GPU-11111111-1111-1111-1111-111111111111', step_timeout_seconds=30,
                         cpu_affinity={'1': [[0], [1], [2]], '2': [[0], [1], [2], [3]]}, nic_interfaces=['hsn0'])
        self.add_artifact('native_site', self.site)
        self.runtime = dict(schema='psana-native-runtime-v1', commit='a'*40, install='/install', api_base='199575205',
                            files={'x.py': 'b'*64}, python='/python', python_version='3.11.0', dependencies={}, native_libraries={},
                            cupy_version='14.0.1', kvikio_version='26.10.0', cuda_runtime_version=12090, cuda_driver_version=12090, pythonpath='/install/lib/python/site-packages',
                            environment={k: '/p' for k in ('PATH', 'LD_LIBRARY_PATH', 'CUDA_HOME', 'CUDA_PATH', 'CONDA_PREFIX', 'LCLS_CALIB_HTTP')})
        for key in ('baseline_runtime', 'candidate_runtime'):
            self.add_artifact(key, self.runtime)
        worker = self.root / 'worker.py'
        worker.write_text('pass\n')
        self.config['artifacts']['native_worker'] = dict(path=str(worker), sha256=n.file_hash(worker))
        self.request = campaign.make_request(self.config, {'profile_sha256': n.digest(self.config)}, campaign.schedule(self.config, 'correctness')[0])
        self.env = patch.dict(os.environ, {'SLURM_JOB_ID': '123'})
        self.env.start()

    def tearDown(self):
        self.env.stop()
        self.temp.cleanup()

    def add_artifact(self, key, value):
        path = self.root / (key + '.json')
        path.write_text(json.dumps(value))
        self.config['artifacts'][key] = dict(path=str(path), sha256=n.file_hash(path))

    def rows(self, bds=1, gpu=False):
        rows = []
        for rank in range(bds + 2):
            events = [100+i for i in range(4) if i % bds == rank-2] if rank >= 2 else []
            rows.append(dict(rank=rank, role='smd0' if rank == 0 else 'eb' if rank == 1 else 'bd',
                             physical_cores=[[0, rank]], timestamps=events, steps=[[t, 0] for t in events],
                             cpu_raw_bytes=500*len(events), callback_events=len(events),
                             io=dict(requested_bytes=500*len(events), useful_bytes=500*len(events), total_bytes=500*len(events),
                                     total_requests=len(events)*5, fallback=True, bulk=True) if gpu and rank >= 2 else None))
        return rows

    def test_supported_contract_and_reject_unqualified_settings(self):
        n.load_contract(self.request)
        self.request['contract']['settings']['n_gpu_streams'] = 2
        self.request['contract_sha256'] = n.digest(self.request['contract'])
        with self.assertRaisesRegex(ValueError, 'unsupported settings'):
            n.load_contract(self.request)

    def test_v2_disjoint_ranges_validate_without_transition_bytes(self):
        self.manifest['schema'] = 'run51-private-input-v2'
        for item in self.manifest['files']:
            item['bigdata'].pop('range_start')
            item['bigdata']['ranges'] = [[50, 200], [300, 200]]
        self.add_artifact('dataset_manifest', self.manifest)
        n.load_contract(self.request)
        self.manifest['files'][0]['bigdata']['ranges'][1][0] = 249
        self.add_artifact('dataset_manifest', self.manifest)
        with self.assertRaisesRegex(ValueError, 'overlapping'):
            n.load_contract(self.request)

    def test_changed_input_and_shared_hardlink_rejected(self):
        path = Path(self.manifest['files'][0]['bigdata']['path'])
        os.link(path, self.root / 'alias')
        with self.assertRaisesRegex(ValueError, 'private copies'):
            n.load_contract(self.request)
        (self.root / 'alias').unlink()
        path.write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'input stat changed'):
            n.load_contract(self.request)

    def test_reference_digest_and_byte_disagreement(self):
        self.request['contract']['dataset']['requested_bytes'] += 1
        self.request['contract_sha256'] = n.digest(self.request['contract'])
        with self.assertRaisesRegex(ValueError, 'byte denominator'):
            n.load_contract(self.request)

    def test_ambiguous_runtime_rejected(self):
        other = copy.deepcopy(self.runtime)
        other['files'] = {'other.py': 'c'*64}
        self.add_artifact('candidate_runtime', other)
        with self.assertRaisesRegex(ValueError, 'uniquely pinned'):
            n.load_contract(self.request)

    def test_changed_shared_runtime_and_out_of_allocation_gpu_rejected(self):
        other = copy.deepcopy(self.runtime)
        other['python'] = '/other'
        self.add_artifact('candidate_runtime', other)
        with self.assertRaisesRegex(ValueError, 'runtime environment differs'):
            n.load_contract(self.request)
        env = dict(SLURM_JOB_GPUS='0,1,2,3', SLURM_STEP_GPUS='2', CUDA_VISIBLE_DEVICES='0')
        before = dict(env)
        self.assertEqual(w.step_gpu_environment(env), before)
        self.assertEqual(env, before)
        for change in ({'SLURM_STEP_GPUS':'4'}, {'CUDA_VISIBLE_DEVICES':'0,1'}, {'SLURM_STEP_GPUS':''}):
            with self.assertRaises(ValueError):
                w.step_gpu_environment(dict(env, **change))

    def test_sdf_and_wrong_allocation_rejected(self):
        self.request['site'] = 'sdf'
        with self.assertRaisesRegex(ValueError, 'Perlmutter'):
            n.load_contract(self.request)
        self.request['site'] = 'perlmutter'
        with patch.dict(os.environ, {'SLURM_JOB_ID': '999'}):
            with self.assertRaisesRegex(ValueError, 'allocation differs'):
                n.load_contract(self.request)

    def test_mixed_runtime_path_policies_rejected(self):
        other = copy.deepcopy(self.runtime)
        other['path_policy'] = 'isolated-install-v1'
        self.add_artifact('candidate_runtime', other)
        with self.assertRaisesRegex(ValueError, 'path policies differ'):
            n.load_contract(self.request)

    def test_environment_scrubs_stale_overrides(self):
        with patch.dict(os.environ, dict(PS_FAKESTEP_FLAG='1', PSANA_TEST='bad', KVIKIO_AUTO_DIRECT_IO='ON', CUDA_VISIBLE_DEVICES='3',
                                         LD_PRELOAD='/bad', PYTHONPATH='/other', OMP_NUM_THREADS='99')):
            env = n.native_environment(self.request, self.runtime)
        self.assertEqual(env['PS_FAKESTEP_FLAG'], '0')
        self.assertEqual(env['OMP_NUM_THREADS'], '1')
        self.assertEqual(env['SLURM_JOB_ID'], '123')
        for key in ('PSANA_TEST', 'KVIKIO_AUTO_DIRECT_IO', 'CUDA_VISIBLE_DEVICES', 'LD_PRELOAD'):
            self.assertNotIn(key, env)

    def test_cpu_and_gpu_work_match_independent_reference(self):
        case = dict(bds=2, workload='cpu-raw', iterator='steps')
        result = n.check_rows(self.rows(2), self.reference['events'], case)
        self.assertEqual(result['requested_bytes'], 2000)
        case.update(workload='gpu-raw', iterator='events')
        n.check_rows(self.rows(2, True), self.reference['events'], case)

    def test_duplicate_events_idle_bd_and_physical_overlap_rejected(self):
        case = dict(bds=2, workload='cpu-raw', iterator='events')
        for field, value in [('timestamps', [100, 100]), ('timestamps', []), ('physical_cores', [[0, 0]])]:
            rows = self.rows(2)
            rows[2][field] = value
            with self.assertRaises(ValueError):
                n.check_rows(rows, self.reference['events'], case)

    def test_gpu_skipped_callback_or_incomplete_read_rejected(self):
        case = dict(bds=1, workload='gpu-raw', iterator='events')
        rows = self.rows(1, True)
        rows[2]['callback_events'] -= 1
        with self.assertRaisesRegex(ValueError, 'callback work'):
            n.check_rows(rows, self.reference['events'], case)
        rows = self.rows(1, True)
        rows[2]['io']['total_bytes'] -= 1
        with self.assertRaisesRegex(ValueError, 'incomplete'):
            n.check_rows(rows, self.reference['events'], case)

    def test_partial_page_residency_counts_range_bytes(self):
        path = self.root / 'pages'
        with path.open('w+b') as stream:
            stream.write(b'x' * (mmap.PAGESIZE*3))
            stream.flush()
            mapping = mmap.mmap(stream.fileno(), 0)
            self.assertEqual(mapping[0], ord('x'))
            self.assertEqual(mapping[mmap.PAGESIZE], ord('x'))
            manifest = dict(files=[dict(bigdata=dict(path=str(path), range_start=17, range_bytes=mmap.PAGESIZE))])
            observed = n.cache_ranges(manifest)
            mapping.close()
        self.assertEqual(observed['measured_bytes'], mmap.PAGESIZE)
        self.assertEqual(observed['resident_bytes_before'], mmap.PAGESIZE)

    def test_fadvise_alone_never_proves_cold(self):
        with patch.object(n, 'cache_ranges', return_value={'resident_bytes_before': 1}):
            with self.assertRaisesRegex(ValueError, 'not client-cold'):
                n.cold_cache(self.manifest)

    def test_nic_reset_and_unavailable_not_zero(self):
        before = dict(interfaces={'hsn0': {'rx': 10, 'tx': 30}}, unavailable={})
        after = dict(interfaces={'hsn0': {'rx': 20, 'tx': 35}}, unavailable={})
        self.assertEqual(n.nic_delta(before, after), dict(rx_bytes=10, tx_bytes=5))
        after['interfaces']['hsn0']['rx'] = 1
        with self.assertRaisesRegex(ValueError, 'reset'):
            n.nic_delta(before, after)
        after['unavailable'] = {'hsn0': 'missing'}
        self.assertIsNone(n.nic_delta(before, after)['rx_bytes'])

    def test_io_reset_windows_tail_and_idempotence(self):
        class Reader:
            def __init__(self):
                self.value = 7
            def io_stats(self):
                row = {k: self.value for k in w.ReaderAccounting.fields}
                row.update(compat_mode=True, io_path='CPU-fallback', bulk_read=True)
                return row
            def reset_io_stats(self):
                self.value = 0
        original = Reader.reset_io_stats
        reader = Reader()
        accounting = w.ReaderAccounting(Reader)
        reader.reset_io_stats()
        reader.value = 11
        first = accounting.finish(reader)
        self.assertEqual(first['total_bytes'], 18)
        self.assertEqual(accounting.finish(reader), first)
        self.assertIs(Reader.reset_io_stats, original)

    def test_step_filter_never_cancels_another_sample(self):
        with patch.object(a, 'command_output', return_value='123.1|ours\n123.2|theirs\n'):
            self.assertEqual(a.matching_steps('123', 'ours'), ['123.1'])
        with patch.object(a, 'command_output', return_value='999.1|ours\n'):
            with self.assertRaisesRegex(ValueError, 'unexpected step'):
                a.matching_steps('123', 'ours')

    def test_timeout_records_cleanup_and_produces_no_success(self):
        process = Mock()
        process.wait.side_effect = [subprocess.TimeoutExpired(['srun'], 1), 0]
        process.poll.return_value = None
        request_path = self.root / 'request.json'
        folder = self.root / 'timeout'
        with patch.object(a, 'matching_steps', return_value=[]), patch.object(a, 'cancel_step', return_value={'no_active_steps': True}) as cancel, patch.object(a.subprocess, 'Popen', return_value=process):
            with self.assertRaises(subprocess.TimeoutExpired):
                a.launch(self.request, request_path, folder, 'measured', self.site, self.runtime)
        cancel.assert_called_once()
        self.assertTrue(n.read(folder / 'failure.json')['cleanup']['no_active_steps'])
        self.assertFalse((folder / 'scheduler.json').exists())

    def test_assembled_receipt_passes_controller_and_checks_native_index(self):
        rows = self.rows()
        for row in rows:
            row.update(node=self.site['node'], affinity=[row['rank']], gpu=None,
                       observed_settings={'batch_size': 20},
                       environment=dict(self.request['contract']['environment'], PS_SMD_N_EVENTS='1000', PS_PARALLEL='mpi'),
                       provenance_before={}, provenance_after={}, setup_seconds=1,
                       first_batch_seconds=2 if row['rank'] >= 2 else None, loop_seconds=5)
        cache = dict(policy='client-cold', verified=True, measured_bytes=2000, resident_bytes_before=0)
        measured = dict(rows=rows, diagnostic=False, work=dict(events=4, event_sha256=n.digest([100,101,102,103]), requested_bytes=2000),
                        cache={'observations': [cache, cache]}, nic={'rx_bytes': None, 'tx_bytes': None}, nic_before={}, nic_after={})
        warmup = dict(status='pass', stage='warmup', rows=rows)
        for invalid in ('diagnostic', 'warmup', 'gpu', 'cache'):
            bad, warm = copy.deepcopy(measured), copy.deepcopy(warmup)
            if invalid == 'diagnostic':
                bad['diagnostic'] = True
            elif invalid == 'warmup':
                warm['rows'][2]['affinity'] = [99]
            elif invalid == 'gpu':
                bad['rows'][2]['gpu'] = warm['rows'][2]['gpu'] = {'uuid': self.site['gpu_uuid']}
            else:
                bad['cache']['observations'][-1]['resident_bytes_before'] = 1
            with self.assertRaises(ValueError):
                a.assemble(self.request, bad, warm, {'measured': {'seconds': 6}}, 7, self.root, self.site)
        row = a.assemble(self.request, measured, warmup, {'measured': {'seconds': 6}}, 7, self.root, self.site)
        campaign.validate_result(self.config, self.request, row, self.root)
        (self.root / 'measured').mkdir()
        (self.root / 'warmup').mkdir()
        native = self.root / 'measured' / 'diagnostic.json'
        native.write_text('original')
        a.seal_native_index(self.root, row)
        campaign.validate_result(self.config, self.request, row, self.root)
        native.write_text('changed')
        with self.assertRaisesRegex(ValueError, 'native indexed file changed'):
            campaign.validate_result(self.config, self.request, row, self.root)

    def delivery_stub(self):
        class Events:
            def __init__(self):
                self._batch_source = iter(['packet0', 'packet1', 'packet2'])
                self._batch_event_count, self._batch_start_time = 0, None
                self.envelopes = iter([])
            def __next__(self):
                service = next(self.envelopes)
                self._batch_event_count += 1
                return SimpleNamespace(service=service)
            def __iter__(self):
                return self
            def _emit_batch_end(self):
                if self._batch_start_time is not None:
                    self._batch_event_count, self._batch_start_time = 0, None
        return Events

    def test_delivery_windows_and_finish_drain_are_distinct(self):
        Events = self.delivery_stub()
        original, original_next, original_init = Events._emit_batch_end, Events.__next__, Events.__init__
        accounting = w.DeliveryAccounting(Events, lambda: 2.5, lambda e: e.service, TransitionId.isEvent)
        try:
            events = Events()
            next(events._batch_source)
            events._batch_start_time = 1
            events._emit_batch_end()  # first GPU packet submits without retirement
            next(events._batch_source)
            events._batch_start_time, events.envelopes = 2, iter([12, 11])
            list(events)
            events._emit_batch_end()  # next packet yields earlier work
            events.envelopes = iter([12, 11])
            list(events)
            events._emit_batch_end()  # finish drain has no start marker
            events._emit_batch_end()  # repeated snapshot must not double count
            result = accounting.result(4)
            self.assertEqual(result['eb_deliveries'], 2)
            self.assertEqual(result['l1_per_delivery_window'], [0, 2])
            self.assertEqual(result['drain_l1'], 2)
            self.assertEqual(result['drain_transitions'], {})
            self.assertEqual(result['first_l1_yield_window_seconds'], 2.5)
            self.assertEqual(result['transitions_per_delivery_window'], [{}, {}])
            with self.assertRaisesRegex(ValueError, 'do not cover'):
                accounting.result(5)
        finally:
            accounting.restore()
        self.assertIs(Events._emit_batch_end, original)
        self.assertIs(Events.__next__, original_next)
        self.assertIs(Events.__init__, original_init)

    def test_cpu_transitions_do_not_count_as_delivered_l1_or_first_batch(self):
        # Events and steps consume the same native envelope stream; steps returns
        # at EndStep and resumes the SAME iterator. This tests only our observer.
        for mode in ('events', 'steps'):
            Events = self.delivery_stub()
            clock = [1.0]
            accounting = w.DeliveryAccounting(Events, lambda: clock[0], lambda e: e.service, TransitionId.isEvent)
            delivered = []
            try:
                events = Events()
                for index, services in enumerate(([6, 8], [11, 7], [6, 8, 12, 11, 7, 5])):
                    next(events._batch_source)
                    events._batch_start_time = index + 1
                    events.envelopes = iter(services)
                    if mode == 'events':
                        delivered.extend(e for e in events if e.service in (11, 12))
                    else:
                        while True:
                            for envelope in events:
                                if envelope.service in (11, 12):
                                    delivered.append(envelope)
                                if envelope.service == 7:
                                    break
                            else:
                                break
                    events._emit_batch_end()
                    if index == 0:
                        self.assertIsNone(accounting.first_nonempty)
                    clock[0] += 1
                result = accounting.result(len(delivered))
                self.assertEqual(result['l1_per_delivery_window'], [0, 1, 2])
                self.assertEqual(result['transitions_per_delivery_window'], [{'6':1,'8':1}, {'7':1}, {'6':1,'8':1,'7':1,'5':1}])
                self.assertEqual(result['drain_l1'], 0)
                self.assertEqual(result['first_l1_yield_window_seconds'], 2.0)
                with self.assertRaisesRegex(ValueError, 'do not cover'):
                    accounting.result(len(delivered)-1)
            finally:
                accounting.restore()

    def test_actual_smd_path_order_and_flat_layout_rejected(self):
        datasource = SimpleNamespace(xtc_files=[f['bigdata']['path'] for f in self.manifest['files']],
                                     smd_files=[f['smd']['path'] for f in self.manifest['files']])
        n.check_selected_files(datasource, self.manifest)
        datasource.smd_files.reverse()
        with self.assertRaisesRegex(ValueError, 'selected smd_files'):
            n.check_selected_files(datasource, self.manifest)
        self.manifest['files'][0]['smd']['path'] = str(self.root / Path(self.manifest['files'][0]['smd']['path']).name)
        self.add_artifact('dataset_manifest', self.manifest)
        with self.assertRaisesRegex(ValueError, 'input layout'):
            n.load_contract(self.request)

    def test_nonzero_signal_cleanup_failure_and_step_identity(self):
        for kind in ('exit', 'signal', 'cleanup', 'identity'):
            folder = self.root / ('failure-' + kind)
            process = Mock()
            process.poll.return_value = None
            if kind in ('signal', 'cleanup'):
                process.wait.side_effect = [InterruptedError('SIGTERM'), 0]
            else:
                process.wait.return_value = 17 if kind == 'exit' else 0
            def popen(*args, **kwargs):
                if kind == 'identity':
                    n.write(folder / 'step.json', dict(job='999', step='1', node=self.site['node']))
                return process
            with patch.object(a, 'matching_steps', return_value=[]), patch.object(a.subprocess, 'Popen', side_effect=popen), patch.object(a, 'cancel_step', side_effect=RuntimeError('remains active') if kind == 'cleanup' else None, return_value={'no_active_steps': True}):
                with self.assertRaises((ValueError, InterruptedError)):
                    a.launch(self.request, self.root/'request.json', folder, 'measured', self.site, self.runtime)
            failure = n.read(folder/'failure.json')
            self.assertEqual(failure['cleanup']['no_active_steps'], kind != 'cleanup')
            self.assertFalse((folder/'scheduler.json').exists())

    def test_cleanup_deadline_is_bounded(self):
        with patch.object(a.time, 'monotonic', side_effect=[0, 0.1, 7]), patch.object(a, 'matching_steps', return_value=['123.1']), patch.object(a.subprocess, 'run') as cancel:
            with self.assertRaisesRegex(TimeoutError, 'cleanup deadline'):
                a.cancel_step('123', 'ours')
            cancel.assert_not_called()

    def test_process_timer_excludes_scheduler_confirmation(self):
        folder = self.root / 'success'
        process = Mock(wait=Mock(return_value=0))
        def popen(*args, **kwargs):
            n.write(folder/'step.json', dict(job='123', step='2', node=self.site['node']))
            n.write(folder/'complete.json', dict(status='pass', stage='measured'))
            return process
        with patch.object(a, 'matching_steps', return_value=[]), patch.object(a.subprocess, 'Popen', side_effect=popen), patch.object(a, 'command_output', return_value='123.2|COMPLETED|0:0'), patch.object(a.time, 'monotonic', side_effect=[0, 5, 9]):
            _, receipt = a.launch(self.request, self.root/'request.json', folder, 'measured', self.site, self.runtime)
        self.assertEqual(receipt['seconds'], 5)
        self.assertEqual(receipt['scheduler_confirmation_seconds'], 4)

    def test_controller_timeout_requires_native_cleanup_headroom(self):
        self.request['timeout_seconds'] = 180
        with self.assertRaisesRegex(ValueError, 'cleanup headroom'):
            n.load_contract(self.request)

    def test_unsuccessful_scheduler_accounting_rejects_native_pass(self):
        folder = self.root / 'scheduler-failure'
        def popen(*args, **kwargs):
            n.write(folder / 'step.json', dict(job='123', step='9', node=self.site['node']))
            return Mock(wait=Mock(return_value=0))
        with patch.object(a, 'matching_steps', return_value=[]), patch.object(a, 'cancel_step', return_value={'no_active_steps': True}), patch.object(a.subprocess, 'Popen', side_effect=popen), patch.object(a, 'command_output', return_value='123.9|FAILED|1:0\n'), patch.object(a.time, 'sleep'):
            with self.assertRaisesRegex(RuntimeError, 'scheduler did not confirm'):
                a.launch(self.request, self.root/'request.json', folder, 'measured', self.site, self.runtime)


if __name__ == '__main__':
    unittest.main()
