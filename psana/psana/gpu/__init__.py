"""
psana.gpu — GPU acceleration for psana2 BD ranks.

Public API
----------
    from psana.gpu import gpu_events, GPUResult, GpuEventContext

    for ctx in gpu_events(smd_glob, gpu_det='jungfrau'):
        calib  = ctx.get('jungfrau.calib').on_gpu   # CuPy, stays on GPU
        energy = ctx.raw('gmd').energy               # CPU, unchanged
        n_hit  = int(cp.sum(calib > 5.0))
        if n_hit > 100:
            save(ctx.get('jungfrau.calib').on_cpu, energy)

Module map
----------
context.py       GPUResult, GpuEventContext         — user-visible API types
gpu_events.py    gpu_events()                        — generic event iterator
gpu_calib.py     GPUDetector, fused_calib_gpu, ...  — calibration pipeline
gpu_stream.py    StreamPool                          — CUDA stream management
gpu_kvikio_read  KvikioGpuReader, PendingBatch       — GDS read layer
gpu_batch.py     GpuBatchView                        — GPUBAT1 wire format
gpudgramlite.py  extract_dgram_info                  — GPU dgram header parse
"""

# context.py and gpu_stream.py have no heavy psana dependencies — safe to
# import eagerly.  gpu_events.py imports psana.psexp.ds_base which would
# create a circular dependency if imported here at module load time (because
# psana.eventbuilder already imports psana.gpu.gpu_batch at load time).
# Use a lazy wrapper so the heavy imports only run when gpu_events() is called.

from psana.gpu.context import GPUResult, GpuEventContext
from psana.gpu.gpu_stream import StreamPool, EventPool
from psana.gpu.detector_router import DetectorRouter
from psana.gpu.gpu_kernel_registry import (
    GPUKernel, GPUFileKernel, GPUKernelRegistry,
    JungfrauCalibKernel, SimpleAreaCalibKernel,
    default_registry, gpu_kernel_from_file,
)
from psana.gpu.gpu_calib import optimal_kernel_batch_size


def gpu_events(smd_glob_or_files, gpu_det, **kwargs):
    """Yield one GpuEventContext per L1Accept event.

    Lazy entry-point: the actual implementation is imported on first call to
    avoid the circular import that would occur if gpu_events.py were imported
    at psana.gpu module load time.

    See psana.gpu.gpu_events.gpu_events for full documentation.
    """
    from psana.gpu.gpu_events import gpu_events as _impl
    return _impl(smd_glob_or_files, gpu_det, **kwargs)


__all__ = [
    # User-facing event loop and result types
    'gpu_events',
    'GPUResult',
    'GpuEventContext',
    # Kernel registry
    'GPUKernel',
    'GPUFileKernel',
    'GPUKernelRegistry',
    'JungfrauCalibKernel',
    'SimpleAreaCalibKernel',
    'default_registry',
    'gpu_kernel_from_file',
    'optimal_kernel_batch_size',
    # Detector routing
    'DetectorRouter',
    # Stream management
    'StreamPool',
    'EventPool',
]
