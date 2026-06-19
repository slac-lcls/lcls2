"""
psana.gpu — GPU acceleration for psana2 BD ranks.

Public API
----------
    from psana import DataSource

    ds = DataSource(exp='mfx100852324', run=77, gpu_det='jungfrau')
    run = next(ds.runs())

    for ctx in run.events():
        calib  = ctx.get('calib').on_gpu            # CuPy, stays on GPU
        energy = ctx.raw('gmd').energy               # CPU, unchanged
        n_hit  = int(cp.sum(calib > 5.0))
        if n_hit > 100:
            save(ctx.get('jungfrau.calib').on_cpu, energy)

Standalone prototype
--------------------
    from psana.gpu.gpu_events_prototype import gpu_events

Module map
----------
context.py       GPUResult, GpuEventContext         — user-visible API types
gpu_events.py    GpuEvents                           — integrated RunSerial iterator
gpu_events_prototype.py gpu_events()                 — standalone prototype iterator
gpu_calib.py     GPUDetector, fused_calib_gpu, ...  — calibration pipeline
gpu_stream.py    StreamPool                          — CUDA stream management
gpu_kvikio_read  KvikioGpuReader, PendingBatch       — GDS read layer
gpu_batch.py     GpuBatchView                        — GPUBAT1 wire format
gpudgramlite.py  extract_dgram_info                  — GPU dgram header parse
"""

# context.py and gpu_stream.py have no heavy psana dependencies — safe to
# import eagerly.  Keep integrated/prototype event iterators lazy because they
# import psana reader internals.

from psana.gpu.context import GPUResult, GpuEventContext
from psana.gpu.gpu_stream import StreamPool, EventPool
from psana.gpu.detector_router import DetectorRouter
from psana.gpu.gpu_kernel_registry import (
    GPUKernel, GPUFileKernel, GPUKernelRegistry,
    JungfrauCalibKernel, SimpleAreaCalibKernel,
    default_registry, gpu_kernel_from_file,
)
from psana.gpu.gpu_calib import optimal_kernel_batch_size


__all__ = [
    # User-facing event result types
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
