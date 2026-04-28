from __future__ import annotations

from psana.gpu.backends import make_gpu_backend
from psana.gpu.execution import make_gpu_execution_backend
from psana.gpu.pipeline import GpuPipeline
from psana.gpu.runtime.base import GpuRuntime


class CupyThreeStageRuntime(GpuRuntime):
    runtime_name = 'cupy'
    pipeline_name = '3stage'

    def __init__(self, run, profiler=None):
        super().__init__(run, profiler=profiler)
        self.execution = make_gpu_execution_backend(
            self.runtime_name,
            run=run,
            profiler=self.profiler,
            logger=getattr(run, 'logger', None),
        )
        self.backend = make_gpu_backend(
            run.dsparms.gpu_detectors,
            run=run,
            execution_backend=self.execution,
        )
        self.pipeline = GpuPipeline(
            backend=self.backend,
            queue_depth=run.dsparms.gpu_queue_depth,
            profiler=self.profiler,
        )

    def make_detector(self, name, accept_missing=False, **kwargs):
        return self.backend.make_detector(name, accept_missing=accept_missing, **kwargs)

    def handle_transition(self, rec):
        return self.pipeline.handle_transition(rec)

    def submit_l1(self, rec):
        return self.pipeline.submit_l1(rec)

    def pop_ready(self):
        yield from self.pipeline.pop_ready()

    def has_free_slot(self):
        return self.pipeline.has_free_slot()

    def wait_ready(self):
        yield from self.pipeline.wait_ready()

    def flush(self):
        yield from self.pipeline.flush()

    def drain(self):
        self.pipeline.drain()

    def finalize(self):
        if self.profiler is not None:
            self.profiler.flush_summary()
