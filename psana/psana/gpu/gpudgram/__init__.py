"""Detector-independent, device-resident XTC parsing primitives."""

from .config import (
    DeviceConfigTables,
    GpuFieldHandle,
    GpuStreamConfigTable,
)
from .parser import DeviceFieldLocators, GpuEventBatch

__all__ = [
    "DeviceConfigTables",
    "DeviceFieldLocators",
    "GpuEventBatch",
    "GpuFieldHandle",
    "GpuStreamConfigTable",
]
