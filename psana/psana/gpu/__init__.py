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

Module map
----------
context.py       GPUResult, GpuEventContext         — user-visible API types
gpu_events.py    GpuEvents                           — integrated RunSerial iterator
gpu_calib.py     GPUDetector, fused_calib_gpu, ...  — calibration pipeline
gpu_stream.py    EventPool                           — CUDA stream management
gpu_kvikio_read  KvikioGpuReader, PendingBatch       — GDS read layer
gpu_batch.py     GpuBatchView                        — GPUBAT1 wire format
gpu_mpi.py       init_gpu_rank, create_gpu_communicators,
                 verify_gpu_pinning, gpu_error_handler  — MPI + GPU rank setup
"""

# context.py and gpu_stream.py have no heavy psana dependencies and are safe to
# import eagerly.  The integrated event iterator remains internal to Run.
#
# gpu_mpi.py has no CuPy or heavy psana dependency — always safe to import.

from psana.gpu.context import GPUResult, GpuEventContext
from psana.gpu.gpu_stream import EventPool
from psana.gpu.detector_router import DetectorRouter
from psana.gpu.gpu_kernel_registry import (
    GPUKernel, GPUFileKernel, GPUKernelRegistry,
    JungfrauCalibKernel, SimpleAreaCalibKernel,
    default_registry, gpu_kernel_from_file,
)
from psana.gpu.gpu_calib import optimal_kernel_batch_size
from psana.gpu.gpu_mpi import (
    init_gpu_rank,
    verify_gpu_pinning,
    create_gpu_communicators,
    gpu_error_handler,
    log_gpu_mem,
    share_calib_between_gpu_peers,
)


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
    'EventPool',
    # MPI + GPU rank management (no CuPy dependency — safe to import early)
    'init_gpu_rank',
    'verify_gpu_pinning',
    'create_gpu_communicators',
    'gpu_error_handler',
]
