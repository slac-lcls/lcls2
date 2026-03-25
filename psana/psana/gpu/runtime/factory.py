from __future__ import annotations

from psana.gpu.runtime.cupy_three_stage import CupyThreeStageRuntime

DEFAULT_GPU_RUNTIME = 'default'
DEFAULT_GPU_PIPELINE = 'default'
SUPPORTED_GPU_RUNTIMES = (DEFAULT_GPU_RUNTIME, 'cupy')
SUPPORTED_GPU_PIPELINES = (DEFAULT_GPU_PIPELINE, '3stage')


def normalize_gpu_runtime_name(name):
    normalized = (name or DEFAULT_GPU_RUNTIME).strip().lower()
    if normalized not in SUPPORTED_GPU_RUNTIMES:
        raise ValueError(
            f'Unsupported gpu_runtime={name!r}; expected one of {SUPPORTED_GPU_RUNTIMES}'
        )
    return 'cupy' if normalized == DEFAULT_GPU_RUNTIME else normalized


def normalize_gpu_pipeline_name(name):
    normalized = (name or DEFAULT_GPU_PIPELINE).strip().lower()
    if normalized not in SUPPORTED_GPU_PIPELINES:
        raise ValueError(
            f'Unsupported gpu_pipeline={name!r}; expected one of {SUPPORTED_GPU_PIPELINES}'
        )
    return '3stage' if normalized == DEFAULT_GPU_PIPELINE else normalized


def make_gpu_runtime(run, profiler=None):
    runtime_name = normalize_gpu_runtime_name(getattr(run.dsparms, 'gpu_runtime', DEFAULT_GPU_RUNTIME))
    pipeline_name = normalize_gpu_pipeline_name(getattr(run.dsparms, 'gpu_pipeline', DEFAULT_GPU_PIPELINE))

    if runtime_name == 'cupy' and pipeline_name == '3stage':
        return CupyThreeStageRuntime(run=run, profiler=profiler)

    raise ValueError(
        f'Unsupported gpu runtime/pipeline combination: runtime={runtime_name!r} pipeline={pipeline_name!r}'
    )
