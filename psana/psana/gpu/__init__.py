"""GPU input routing and parsed-field access; no automatic calibration.

Use evt.gpu.detector(name).field(algorithm, field) for leased input access.
User callback support is under development.
"""

from psana.gpu.context import GPUResult, GpuEventState
from psana.gpu.gpu_input import GpuFieldData, GpuFieldResult
from psana.gpu.gpu_mpi import init_gpu_rank


__all__ = [
    "GPUResult",
    "GpuFieldData",
    "GpuFieldResult",
    "GpuEventState",
    "init_gpu_rank",
]
