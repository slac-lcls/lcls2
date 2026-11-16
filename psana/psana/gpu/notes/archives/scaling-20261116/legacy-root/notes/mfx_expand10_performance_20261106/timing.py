"""Opt-in CPU/CUDA/NVTX spans; resolve CUDA events only after the final drain."""
from contextlib import contextmanager, ExitStack
from functools import wraps
import time
from unittest.mock import patch


class Window:
    """Warmup ends at a callback boundary after joining prior GPU submissions."""
    def __init__(self, warmup, total, clock=time.perf_counter):
        if not 0 < warmup < total:
            raise ValueError('Require 0 < warmup < total events')
        self.target, self.total, self.clock = warmup, total, clock
        self.submitted = self.warm_events = 0
        self.start = self.last_submission = self.end = None
        self.boundary_join_seconds = 0.

    @property
    def phase(self):
        return 'steady' if self.start is not None else 'warmup'

    def before(self, join):
        if self.start is None and self.submitted >= self.target:
            t = self.clock()
            join()
            self.boundary_join_seconds = self.clock() - t
            self.warm_events = self.submitted
            self.start = self.clock()

    def after(self, size):
        if size <= 0 or self.submitted + size > self.total:
            raise ValueError('Invalid submitted event count')
        self.submitted += size
        self.last_submission = self.clock()

    def finish(self):
        if self.submitted != self.total or self.start is None:
            raise RuntimeError('Incomplete run or no steady window; reduce warmup')
        self.end = self.clock()
        duration = self.end - self.start
        if duration <= 0:
            raise RuntimeError('Nonpositive timing window')
        return dict(warmup_events=self.warm_events, steady_events=self.total-self.warm_events,
                    boundary_join_seconds=self.boundary_join_seconds,
                    submission_seconds=self.last_submission-self.start,
                    tail_completion_seconds=self.end-self.last_submission,
                    completed_window_seconds=duration,
                    completed_events_per_second=(self.total-self.warm_events)/duration)


class Recorder:
    def __init__(self, enabled=False):
        self.enabled = enabled
        self.rows, self.cuda = [], []
        self.phase = lambda: 'setup'
        self.callback = None

    def tags(self):
        return dict(phase=self.phase(), callback=self.callback)

    @contextmanager
    def cpu(self, name):
        if not self.enabled:
            yield
            return
        import cupy as cp
        tags = self.tags()
        cp.cuda.nvtx.RangePush(name)
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter()-start
            cp.cuda.nvtx.RangePop()
            self.rows.append(dict(stage=name, seconds=elapsed, **tags))

    def device_call(self, name, execution_stream, function, *args, **kwargs):
        if not self.enabled:
            return function(*args, **kwargs)
        import cupy as cp
        begin, end = cp.cuda.Event(), cp.cuda.Event()
        row = dict(stage=name, stream=int(execution_stream.ptr), begin=begin, end=end,
                   complete=False, **self.tags())
        self.cuda.append(row)  # Own before any asynchronous recording.
        begin.record(execution_stream)
        with self.cpu(name + '.enqueue'):
            result = function(*args, **kwargs)
        end.record(execution_stream)
        row['complete'] = True
        return result

    def summary(self):
        import cupy as cp
        import numpy as np
        intervals = []
        for row in self.cuda:
            if not row['complete']:
                raise RuntimeError('Incomplete CUDA timing span')
            row['end'].synchronize()
            intervals.append(dict(stage=row['stage'], stream=row['stream'], phase=row['phase'],
                                  callback=row['callback'], seconds=float(cp.cuda.get_elapsed_time(row['begin'], row['end']))/1000))
        def aggregate(rows):
            groups = {}
            for row in rows:
                groups.setdefault((row['phase'], row['stage']), []).append(row['seconds'])
            return [dict(phase=k[0], stage=k[1], calls=len(v), total_seconds=sum(v),
                         p50_seconds=float(np.percentile(v,50)), p95_seconds=float(np.percentile(v,95)))
                    for k,v in sorted(groups.items())]
        return dict(cpu=aggregate(self.rows), cuda=aggregate(intervals),
                    cpu_intervals=self.rows, cuda_intervals=intervals,
                    interpretation='CPU spans are inclusive/nested. CUDA call spans include host submission gaps and stream waits; kernel spans bracket individual launches. Concurrent/nested intervals must not be summed into loop wall time.')

    def wrap(self, name, function, generator=False):
        if generator:
            @wraps(function)
            def iterate(*args, **kwargs):
                source = function(*args, **kwargs)
                try:
                    while True:
                        try:
                            with self.cpu(name):
                                item = next(source)
                        except StopIteration:
                            return
                        yield item  # Consumer time is deliberately outside the span.
                finally:
                    with self.cpu(name + '.close'):
                        source.close()
            return iterate
        @wraps(function)
        def call(*args, **kwargs):
            with self.cpu(name):
                return function(*args, **kwargs)
        return call

    @contextmanager
    def framework(self):
        if not self.enabled:
            yield
            return
        from psana.gpu.gpu_events import GpuEventManager
        from psana.gpu.gpu_stream import EventPool
        from psana.gpu.gpu_task_batch import BatchInputContext
        from psana.gpu.gpudgram.batch import GpuXtcBatchPool
        from psana.gpu.gpu_kvikio_read import KvikioGpuReader
        from psana.psexp.event_manager import EventManager
        hooks = ((GpuEventManager, '_next_batch', 'smd_eventbuilder_next', False),
                 (GpuEventManager, '_trim_gpu_caches', 'cache_trim_after_pressure', False),
                 (EventManager, '_read', 'cpu_ancillary_read', False),
                 (EventManager, '_get_next_dgrams', 'cpu_dgram_construction', False),
                 (EventPool, 'submit', 'slot_submit_including_null_wait', False),
                 (EventPool, 'begin_retire_next', 'producer_retirement_wait', False),
                 (EventPool, 'finish_retire_next', 'consumer_retirement_release', False),
                 (EventPool, 'flush', 'slot_flush_active', True),
                 (KvikioGpuReader, '_submit_read', 'kvikio_submit', False),
                 (KvikioGpuReader, 'wait_batch', 'kvikio_wait', False))
        with ExitStack() as stack:
            for cls, attr, name, generator in hooks:
                stack.enter_context(patch.object(cls, attr, self.wrap(name, getattr(cls, attr), generator)))
            for attr in ('parse_window','parse_groups'):
                original = getattr(GpuXtcBatchPool, attr)
                def parse(pool, reads, stream, *args, _original=original, _name=attr, **kwargs):
                    return self.device_call(_name + '_call', stream, _original, pool, reads, stream, *args, **kwargs)
                stack.enter_context(patch.object(GpuXtcBatchPool, attr, parse))
            original_metadata = BatchInputContext._ensure_metadata
            def metadata(batch):
                if batch._metadata_ready or not batch.size or not batch._metadata_open:
                    return original_metadata(batch)
                return self.device_call('descriptor_build_upload_call', batch._stream, original_metadata, batch)
            stack.enter_context(patch.object(BatchInputContext, '_ensure_metadata', metadata))
            yield


@contextmanager
def admission_counts(window):
    """Count the frozen runtime's pressure-only trim path; no clocks or CUDA."""
    from psana.gpu.gpu_events import GpuEventManager
    original = GpuEventManager._trim_gpu_caches
    counts = dict(warmup=0, steady=0)
    def counted(manager):
        counts[window.phase] += 1
        return original(manager)
    with patch.object(GpuEventManager, '_trim_gpu_caches', counted):
        yield counts


class Kernel:
    def __init__(self, kernel, name, recorder):
        self.kernel, self.name, self.recorder = kernel, name, recorder

    def __call__(self, *args, **kwargs):
        return self.recorder.device_call(self.name, kwargs['stream'], self.kernel, *args, **kwargs)


class KernelCache(dict):
    """Wrap lazy calibration kernel insertion without changing calibration code."""
    def __init__(self, recorder):
        super().__init__()
        self.recorder = recorder

    def __setitem__(self, key, kernel):
        # The first call uses the local unwrapped variable; compile it outside the
        # steady window during warmup. Subsequent lookups time actual launches.
        super().__setitem__(key, Kernel(kernel, 'calibration_kernel', self.recorder))
