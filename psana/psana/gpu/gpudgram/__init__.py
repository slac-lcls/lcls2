"""Detector-independent GPU views over XTC dgrams."""

from .parser import GPUDgramBatch
from .schema import GpuNamesSchema

__all__ = ["GPUDgramBatch", "GpuNamesSchema"]
