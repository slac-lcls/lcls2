"""Task-local batch-capacity extension; no full-image publication."""
import importlib.util
import os
from pathlib import Path
import time
import weakref

import numpy as np

from reference import SHAPE, PIXELS, require, digest
from contracts import admission_from_budget

HISTORICAL = Path(__file__).resolve().parents[1] / 'mfx_expand10_stage2_20261105/adapter.py'
spec = importlib.util.spec_from_file_location('compute_seed_adapter', HISTORICAL)
adapter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adapter)
spec = importlib.util.spec_from_file_location('compute_pinned_calibration',
        Path(os.environ['SIM10_PR_SOURCE']) / 'psana/psana/gpu/examples/jungfrau_calibration.py')
calibration_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(calibration_module)


class StageTask:
    inputs = (('jungfrau', 'sim10', 'seed'),)

    def __init__(self, stage, batch, timestamps, acceptance=None, diagnostic=False):
        self.stage, self.batch, self.acceptance = stage, batch, acceptance
        self.calibration = calibration_module.JungfrauCalibration(use_offset=True)
        self.calibconst = self.calibration.calibconst if stage in ('EC', 'ECC') else ()
        self.callbacks, self.memory, self.previous = [], [], []
        self.stream_ptr = None
        self.cube = None
        self.timestamps = timestamps
        self.kernel = None
        self.manager = None
        self.initialized = False
        self.diagnostic = diagnostic
        self.intervals = []

    def measure(self, name, execution_stream, function, *args, **kwargs):
        if not self.diagnostic:
            return function(*args, **kwargs)
        import cupy as cp
        start, finish = cp.cuda.Event(), cp.cuda.Event()
        self.intervals.append((name, start, finish))
        start.record(execution_stream)
        result = function(*args, **kwargs)
        finish.record(execution_stream)
        return result

    def interval_results(self):
        import cupy as cp
        result = []
        for name, start, finish in self.intervals:
            finish.synchronize()
            result.append(dict(name=name, milliseconds=float(cp.cuda.get_elapsed_time(start, finish))))
        return dict(intervals=result, interpretation='Instrumented pilot-only inclusive CUDA intervals; do not sum overlapping intervals or call these clean throughput.')

    def sample(self, label):
        import cupy as cp
        free, total = cp.cuda.runtime.memGetInfo()
        pool = cp.get_default_memory_pool()
        row = dict(label=label, free=int(free), total=int(total), used=int(pool.used_bytes()),
                   reserved=int(pool.total_bytes()), at=time.monotonic())
        require(free >= 4 << 30, 'Less than 4 GiB actual device safety headroom')
        self.memory.append(row)
        return row

    def prepare(self, manager, constant_records):
        import cupy as cp
        require(not self.initialized, 'Task prepared twice')
        self.manager = weakref.proxy(manager)
        probe = cp.empty(1, cp.uint8)
        probe_reference = weakref.ref(probe)
        require(probe_reference() is probe, 'CuPy ndarray weakref unsupported')
        del probe
        require(probe_reference() is None, 'CuPy ndarray weakref retains allocation')
        self.weakref_probe = dict(status='pass', dtype='uint8', elements=1,
                                 released=True, cupy=cp.__version__)
        sample = self.sample('before_task_setup')
        self.ledger = admission_from_budget(self.stage, self.batch, sample['free'], manager._gpu_budget)
        self.constant_checks = {}
        for detector, key in self.calibconst:
            value = manager._task_constants.get(detector, key)
            expected = constant_records[key]
            require(list(value.shape) == expected['shape'] and str(value.dtype) == expected['dtype'], 'GPU constant layout')
            hasher = __import__('hashlib').sha256()
            for plane in range(value.shape[0]):
                for panel in range(value.shape[1]):
                    hasher.update(memoryview(value[plane, panel].get()).cast('B'))
            require(hasher.hexdigest() == expected['sha256'], 'GPU constant hash after MPI distribution')
            self.constant_checks[key] = expected
        if self.stage != 'S':
            self.kernel = cp.RawKernel(adapter.kernel_source(), 'expand10', options=('--std=c++17',))
            self.kernel.compile()
        if self.calibconst:
            types = {key: calibration_module._FLOAT_TYPES[np.dtype(record['dtype'])]
                     for key, record in constant_records.items()}
            types = {key: types[key] for _, key in self.calibconst}
            kernel = cp.RawKernel(self.calibration._source(tuple(range(32)), types, with_validity=False),
                                  'calibrate', options=('--std=c++17', '--fmad=false'))
            kernel.compile()
            self.calibration._kernels[(0, tuple(range(32)), tuple(types.items()), False)] = kernel
        if self.stage == 'ECC':
            from mpi_cube import Cube
            self.cube = Cube(self.timestamps, max_batch_events=self.batch)
            self.cube.prepare(manager.event_pool.next_stream)
        self.initialized = True
        self.sample('after_task_setup')

    def __call__(self, batch, stream):
        import cupy as cp
        require(self.initialized and 1 <= batch.size <= self.batch, 'Unqualified callback capacity')
        if self.cube is not None:
            self.cube.drain()
        require(not any(owner() is not None for owner in self.previous), 'Previous full-image owner retained across callbacks')
        pointer = int(stream.ptr)
        require(self.stream_ptr in (None, pointer), 'Unexpected second CUDA stream')
        self.stream_ptr = pointer
        field = batch.field(*self.inputs[0])
        require(tuple(batch.segment_ids('jungfrau')) == tuple(range(32)) and
                tuple(field.segment_ids) == tuple(range(32)) and
                field.rows.shape == (batch.size, 32, adapter.FIELD_WORDS) and
                field.rows.dtype == cp.uint64, 'Descriptor ABI/panel order')
        self.callbacks.append(dict(size=batch.size, timestamps=list(map(int, batch.timestamps)),
                                   stream=pointer, step_generation=int(batch.step_generation)))
        self.sample('callback_entry')
        if self.stage == 'S':
            self.sample('after_submission')
            return
        with stream:
            raw = cp.empty((batch.size, *SHAPE), cp.uint16)
            presence = cp.empty((batch.size, 32), cp.uint8)
            errors = cp.empty((batch.size, 32), cp.uint32)
            batch.keepalive(raw)
            batch.keepalive(presence)
            batch.publish('sim10.errors', errors)
            self.measure('expansion', stream, self.kernel,
                         (batch.size * 32 * ((PIXELS + 255) // 256),), (256,),
                         (field.rows, raw, presence, errors, np.uint64(batch.size * 32)), stream=stream)
            calibrated = None
            if self.calibconst:
                calibrated, _ = self.measure('calibration', stream, self.calibration.calibrate,
                    adapter.ExpandedBatch(batch, raw, presence), stream, publish=False)
            if self.cube is not None:
                self.measure('cube_count_and_sum', stream, self.cube.add,
                             calibrated, errors, batch.timestamps, stream, batch.keepalive, batch.step_generation)
            self.previous = [weakref.ref(raw)] + ([weakref.ref(calibrated)] if calibrated is not None else [])
            self.sample('after_submission')
            if self.acceptance is not None:
                stream.synchronize()
                self.acceptance(batch.timestamps, raw, calibrated, errors)


class CopyAudit:
    """Observe actual destination decisions without replacing the copy algorithm."""
    def __init__(self, copier):
        self.bytes = self.groups = self.fallbacks = self.pinned_capacity_high = 0
        original_destination, original_enqueue = copier._destination, copier.enqueue

        def destination(shape, dtype, nbytes):
            host, block = original_destination(shape, dtype, nbytes)
            self.bytes += nbytes
            self.groups += 1
            self.fallbacks += int(nbytes > 0 and block is None)
            self.pinned_capacity_high = max(self.pinned_capacity_high, copier.pinned_bytes)
            require(not nbytes or block is not None, 'Ordinary-host publication fallback forbidden')
            return host, block

        def enqueue(record):
            publications = record.publication_batches
            require(all(item.name == 'sim10.errors' and item.dtype == np.uint32 and
                        item.shape[1:] == (32,) for item in publications), 'Unexpected image/metadata publication')
            require(sum(item.nbytes for item in publications) <= 64 << 10, 'Publication cap exceeded')
            return original_enqueue(record)

        copier._destination, copier.enqueue = destination, enqueue

    def result(self):
        return dict(bytes=self.bytes, groups=self.groups, ordinary_host_fallbacks=self.fallbacks,
                    pinned_capacity_high=self.pinned_capacity_high,
                    method='harness wrappers of every enqueue/_destination decision, not CUDA profiling')
