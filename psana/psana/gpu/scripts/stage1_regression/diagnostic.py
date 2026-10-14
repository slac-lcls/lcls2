"""Separate preflight diagnostics; no instrumentation in timed samples.

Host submission durations are inclusive, not device durations. Kernel factories
are wrapped to count actual launches without adding synchronization. Steady
labels start after the first delivered event on each BD. CUDA copies/syncs in
Cython require the separate Nsight capture, not these Python counters.
"""
from collections import defaultdict
from functools import wraps
import time


class Diagnostic:
    def __init__(self, workload):
        from psana.gpu.gpudgram import parser
        from psana.gpu import gpu_detector as detector
        from psana.gpu.gpu_stream import EventPool
        from psana.gpu.gpu_kvikio_read import KvikioGpuReader
        self.active = self.steady = False
        self.stats = defaultdict(lambda: dict(calls=0, host_ns=0))
        for module, factory, label in (
            (parser, '_walk_kernel', 'walk'),
            (parser, '_init_locators_kernel', 'init_locators'),
            (parser, '_locate_fields_kernel', 'locate_fields'),
            (detector, '_batched_gather_kernel', 'gather'),
        ):
            self.kernel(module, factory, label)
        if workload == 'calib':
            from psana.gpu import gpu_calib
            self.kernel(gpu_calib, '_jungfrau_calib_kernel', 'calib')
            self.kernel(detector, '_zero_missing_kernel', 'zero_missing')
        for owner, name, label in (
            (EventPool, 'submit', 'pool.submit'),
            (EventPool, 'begin_retire_next', 'pool.retire'),
            (EventPool, 'finish_retire_next', 'pool.release'),
            (KvikioGpuReader, 'wait_batch', 'read.wait'),
        ):
            setattr(owner, name, self.measured(getattr(owner, name), label))

    def measured(self, fn, label):
        @wraps(fn)
        def call(*args, **kwargs):
            if not self.active:
                return fn(*args, **kwargs)
            key = ('steady/' if self.steady else 'startup/') + label
            start = time.perf_counter_ns()
            try:
                return fn(*args, **kwargs)
            finally:
                row = self.stats[key]
                row['calls'] += 1
                row['host_ns'] += time.perf_counter_ns()-start
        return call

    def kernel(self, module, name, label):
        factory = getattr(module, name)
        @wraps(factory)
        def wrapped(*args, **kwargs):
            return self.measured(factory(*args, **kwargs), 'launch.'+label)
        setattr(module, name, wrapped)

    def snapshot(self):
        return dict(self.stats)
