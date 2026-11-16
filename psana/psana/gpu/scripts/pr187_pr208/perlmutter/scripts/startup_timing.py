"""Inclusive host timers and one CUDA-complete first-batch boundary."""
from contextlib import contextmanager
from functools import wraps
import time


class Recorder:
    def __init__(self, nvtx=None, clock=time.perf_counter):
        self.nvtx = nvtx
        self.clock = clock
        self.phase = 'setup'
        self.records = {}
        self.installed = []

    @contextmanager
    def region(self, name):
        if self.nvtx is not None:
            self.nvtx.RangePush(name)
        try:
            yield
        finally:
            if self.nvtx is not None:
                self.nvtx.RangePop()

    def mark(self, name):
        if self.nvtx is not None:
            self.nvtx.Mark(name)

    def wrap(self, owner, method, label=None, byte_count=None):
        original = getattr(owner, method)
        name = label or owner.__name__ + '.' + method
        self.installed.append(name)

        @wraps(original)
        def measured(*args, **kwargs):
            phase = self.phase
            nbytes = int(byte_count(args, kwargs)) if byte_count else 0
            began = self.clock()
            failed = False
            try:
                with self.region(name):
                    return original(*args, **kwargs)
            except BaseException:
                failed = True
                raise
            finally:
                elapsed = self.clock() - began
                for key in (name, phase + '/' + name):
                    record = self.records.setdefault(key, dict(count=0, total_s=0., max_s=0., bytes=0, failures=0))
                    record['count'] += 1
                    record['total_s'] += elapsed
                    record['max_s'] = max(record['max_s'], elapsed)
                    record['bytes'] += nbytes
                    record['failures'] += int(failed)
        setattr(owner, method, measured)


class FirstBatch:
    """Retain a completion event until the first batch's last event is delivered."""
    def __init__(self):
        self.timestamps = None
        self.done = None
        self.seconds = None
        self.events = None

    def submitted(self, timestamps, stream, event_factory):
        if self.timestamps is None:
            self.timestamps = tuple(int(t) for t in timestamps)
            assert len(self.timestamps) == 20
            self.done = event_factory(disable_timing=True)
            self.done.record(stream)

    def delivered(self, stamps, loop_start, clock=time.perf_counter):
        if self.seconds is None and self.timestamps is not None and stamps[-1] == self.timestamps[-1]:
            assert tuple(stamps) == self.timestamps
            self.done.synchronize()
            assert self.done.done
            self.seconds = clock() - loop_start
            self.events = len(stamps)
            return True
        return False


def install(recorder, variant, worker):
    from psana.psexp.mpi_ds import RunParallel
    if variant == 'B':
        recorder.wrap(RunParallel, '_discover_gpu_placement_impl')
    if not worker:
        return
    from psana.gpu.gpu_events import GpuEventManager
    from psana.gpu.gpu_task import RequestedConstants
    for method in ('_setup_gpu_pipeline', 'close'):
        recorder.wrap(GpuEventManager, method)
    recorder.wrap(RequestedConstants, 'refresh')
    if variant == 'B':
        from psana.gpu import gpu_shared_constants as shared
        for method in ('refresh', '_scan', '_agree_on_set', '_establish', '_publish',
                       '_subscribe', '_settle', '_barrier', '_sync_device', 'close',
                       '_upload_private', '_release_shared'):
            recorder.wrap(shared.SharedRequestedConstants, method)
        recorder.wrap(shared, 'digest_of', 'digest_of', lambda a, k: a[0].nbytes)
        recorder.wrap(shared._OwnedBlock, 'handle', 'IPC.export_handle')
        recorder.wrap(shared._ImportedBlock, '__init__', 'IPC.open_mapping')
        recorder.wrap(shared._ImportedBlock, 'close', 'IPC.close_mapping')
