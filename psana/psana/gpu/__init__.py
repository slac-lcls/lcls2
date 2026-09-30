"""GPU input routing and parsed-field access; no automatic calibration.

Use evt.gpu.detector(name).field(algorithm, field) for leased input access.
GpuTask callbacks process execution subbatches and publish named host results.
"""

from psana.gpu.context import GPUResult, GpuEventState
from psana.gpu.gpu_input import GpuFieldData, GpuFieldResult
from psana.gpu.gpu_mpi import init_gpu_rank
from psana.gpu.gpu_task import GpuTask


__all__ = [
    "GpuTask",
    "GPUResult",
    "GpuFieldData",
    "GpuFieldResult",
    "GpuEventState",
    "init_gpu_rank",
]
