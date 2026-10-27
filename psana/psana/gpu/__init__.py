"""GPU input routing and parsed-field access; no automatic calibration.

Use evt.gpu.detector(name).field(algorithm, field) for leased input access.
GpuTask callbacks process execution subbatches and publish named host results.
"""

from psana.gpu.context import GPUResult, GpuEventState
from psana.gpu.gpu_input import GpuFieldData, GpuFieldResult
from psana.gpu.gpu_task import GpuTask

# Re-exported from psexp.run so an external framework can type-check
# `run.gpu_device` without importing from psana.psexp, which is internal.
from psana.psexp.run import GpuDevice


__all__ = [
    "GpuTask",
    "GPUResult",
    "GpuFieldData",
    "GpuFieldResult",
    "GpuEventState",
    "GpuDevice",
]
