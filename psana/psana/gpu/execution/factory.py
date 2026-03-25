from __future__ import annotations

from psana.gpu.execution.cupy import CupyExecutionBackend


SUPPORTED_GPU_EXECUTION_BACKENDS = ('cupy',)


def make_gpu_execution_backend(name, run, profiler=None, logger=None):
    normalized = (name or 'cupy').strip().lower()
    if normalized == 'cupy':
        return CupyExecutionBackend(run=run, profiler=profiler, logger=logger)
    raise ValueError(
        f'Unsupported gpu execution backend={name!r}; expected one of {SUPPORTED_GPU_EXECUTION_BACKENDS}'
    )
