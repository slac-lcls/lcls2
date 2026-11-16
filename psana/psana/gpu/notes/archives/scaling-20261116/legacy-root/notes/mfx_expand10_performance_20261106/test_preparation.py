"""Host-only boundary, ownership, fixture-binding and profiling regression tests."""
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from timing import Recorder, Window, admission_counts
from perf_cube import Capture, Cube
import fixture_check


class Clock:
    def __init__(self): self.now = 0.
    def __call__(self): return self.now
    def advance(self, amount): self.now += amount


class Tests(unittest.TestCase):
    def test_callback_rounding_and_completed_tail(self):
        clock = Clock()
        w = Window(5, 10, clock)
        joined = []
        def join():
            joined.append(w.submitted)
            clock.advance(2)
        for size in (3,3,4):
            w.before(join)
            clock.advance(1)
            w.after(size)
        clock.advance(3)  # EOF/delivery/explicit join included in denominator.
        result = w.finish()
        self.assertEqual(joined, [6])
        self.assertEqual(result['warmup_events'], 6)
        self.assertEqual(result['steady_events'], 4)
        self.assertEqual(result['boundary_join_seconds'], 2)
        self.assertEqual(result['submission_seconds'], 1)
        self.assertEqual(result['tail_completion_seconds'], 3)
        self.assertEqual(result['completed_events_per_second'], 1)

    def test_failed_join_never_starts_timer(self):
        w = Window(2, 4)
        w.after(2)
        def fail(): raise RuntimeError('CUDA error')
        with self.assertRaises(RuntimeError): w.before(fail)
        self.assertIsNone(w.start)
        with self.assertRaises(RuntimeError): w.finish()

    def test_no_steady_window_and_incomplete_rejected(self):
        w = Window(9, 10)
        w.after(10)
        with self.assertRaises(RuntimeError): w.finish()
        w = Window(2, 10)
        w.after(2); w.before(lambda: None); w.after(3)
        with self.assertRaises(RuntimeError): w.finish()
        with self.assertRaises(ValueError): w.after(6)
        for warmup in (0,10,11):
            with self.assertRaises(ValueError): Window(warmup,10)

    def test_generator_excludes_consumer_and_preserves_close(self):
        clock, closed = Clock(), []
        recorder = Recorder(True)
        cp = SimpleNamespace(cuda=SimpleNamespace(nvtx=SimpleNamespace(RangePush=lambda _:None, RangePop=lambda:None)))
        def source():
            try:
                clock.advance(2); yield 1
                clock.advance(3); yield 2
            finally:
                closed.append(True)
        with patch.dict('sys.modules', {'cupy':cp}), patch('timing.time.perf_counter',clock):
            gen = recorder.wrap('flush',source,True)()
            self.assertEqual(next(gen),1)
            clock.advance(100)
            self.assertEqual(next(gen),2)
            gen.close()
        self.assertEqual([r['seconds'] for r in recorder.rows], [2,3,0])
        self.assertEqual(closed,[True])

    def test_clean_mode_creates_no_cuda_events_or_publications(self):
        recorder = Recorder(False)
        with patch.dict('sys.modules', {'cupy':None}):
            with recorder.cpu('unused'):
                self.assertEqual(recorder.device_call('unused', None, lambda:42),42)
                stream=object()
                self.assertIs(recorder.device_call('kernel',stream,lambda **kw:kw['stream'],stream=stream),stream)
        self.assertEqual(recorder.rows,[])
        self.assertEqual(recorder.cuda,[])
        owners=[]
        capture=Capture(SimpleNamespace(keepalive=lambda *values:owners.extend(values)))
        obj=object(); capture.publish('sim10.errors',obj)
        self.assertIs(capture.arrays['sim10.errors'],obj)
        self.assertEqual(owners,[obj])

    def test_framework_hooks_match_runtime_and_restore_on_failure(self):
        from psana.gpu.gpu_stream import EventPool
        from psana.gpu.gpudgram.batch import GpuXtcBatchPool
        from psana.gpu.gpu_task_batch import BatchInputContext
        original=(EventPool.flush,GpuXtcBatchPool.parse_groups,BatchInputContext._ensure_metadata)
        with self.assertRaisesRegex(RuntimeError,'injected'):
            with Recorder(True).framework():
                self.assertIsNot(EventPool.flush,original[0])
                self.assertIsNot(GpuXtcBatchPool.parse_groups,original[1])
                self.assertIsNot(BatchInputContext._ensure_metadata,original[2])
                raise RuntimeError('injected')
        self.assertEqual(original,(EventPool.flush,GpuXtcBatchPool.parse_groups,BatchInputContext._ensure_metadata))
        sealed=SimpleNamespace(_metadata_ready=False,size=1,_metadata_open=False,_stream=None)
        with Recorder(True).framework():
            with self.assertRaisesRegex(RuntimeError,'during the callback'):
                BatchInputContext._ensure_metadata(sealed)

    def test_pressure_counter_no_clocks_and_restores(self):
        from psana.gpu.gpu_events import GpuEventManager
        clock=Clock(); window=Window(2,4,clock)
        with patch.object(GpuEventManager,'_trim_gpu_caches',lambda manager:17):
            original=GpuEventManager._trim_gpu_caches
            with self.assertRaisesRegex(RuntimeError,'injected'):
                with admission_counts(window) as counts, patch('timing.time.perf_counter',side_effect=AssertionError('timer used')):
                    self.assertEqual(GpuEventManager._trim_gpu_caches(None),17)
                    window.after(2); window.before(lambda:None)
                    self.assertEqual(GpuEventManager._trim_gpu_caches(None),17)
                    raise RuntimeError('injected')
            self.assertEqual(counts,{'warmup':1,'steady':1})
            self.assertIs(GpuEventManager._trim_gpu_caches,original)

    def test_prepare_rejects_repeat_or_existing_events_without_cuda(self):
        cube=Cube(range(4),warmup=2,recorder=Recorder(False),nbins=3,active_bins=2,
                  shape=(2,3,5),max_state_bytes=1<<20)
        cube.kernel=object()
        with patch.dict('sys.modules',{'cupy':None}):
            with self.assertRaisesRegex(RuntimeError,'Prepare once'):cube.prepare()
            cube.kernel=None; cube.seen.add(0)
            with self.assertRaisesRegex(RuntimeError,'Prepare once'):cube.prepare()

    def test_10k_manifest_receipt_size_and_optional_fresh_hash(self):
        def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); notes=root/'notes'; root=root/'data'
            root.mkdir(); (root/'smalldata').mkdir()
            (notes/'mfx_expand10_10k_20261106').mkdir(parents=True)
            streams=[]; files=[]
            for stream in range(10):
                mapping={}
                for kind in ('bd','smd'):
                    path=(root if kind=='bd' else root/'smalldata')/f'{stream}-{kind}'
                    path.write_bytes(b'abc')
                    record=dict(path=str(path),sha256=sha(path))
                    mapping[kind]=record; files.append(dict(bytes=3,**record))
                streams.append(dict(stream=stream,files=mapping))
            manifest=dict(event_count=10000,streams=streams,
                          schedule=[dict(service=12,timestamp=i) for i in range(10000)])
            (root/'generation.json').write_text(json.dumps(manifest))
            receipt=dict(root=str(root),generation_sha256=sha(root/'generation.json'),files=files)
            (notes/'mfx_expand10_10k_20261106/files.json').write_text(json.dumps(receipt))
            with patch.object(fixture_check,'NOTES',notes):
                fixture_check.check_10k(root,sha,full_hash=True)
                Path(files[0]['path']).write_bytes(b'def')
                # Size-only mode makes no claim to detect same-size mutations.
                fixture_check.check_10k(root,sha)
                with self.assertRaises(ValueError): fixture_check.check_10k(root,sha,full_hash=True)
                Path(files[0]['path']).write_bytes(b'longer')
                with self.assertRaises(ValueError): fixture_check.check_10k(root,sha)
                (root/'generation.json').write_text('{}')
                with self.assertRaises(ValueError): fixture_check.check_10k(root,sha)


if __name__ == '__main__': unittest.main()
