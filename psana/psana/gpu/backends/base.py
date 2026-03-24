from __future__ import annotations

from abc import ABC

from psana.gpu.cache import GpuResidencyCache
from psana.psexp.run import Run


class GpuDetectorBackend(ABC):
    """Detector-specific hooks for the generic GPU pipeline."""

    detector_name = None

    def __init__(self, run):
        self.run = run
        self.device_cache = GpuResidencyCache(
            profiler=getattr(run, "profiler", None),
            logger=getattr(run, "logger", None),
        )

    def make_detector(self, name, accept_missing=False, **kwargs):
        return Run.Detector(self.run, name, accept_missing=accept_missing, **kwargs)

    def allocate_slot_buffers(self, slot):
        return None

    def pack_l1_to_host(self, rec, slot):
        return None

    def ensure_device_cache(self, rec, slot):
        return None

    def transfer_to_device(self, rec, slot):
        return None

    def launch_compute(self, rec, slot):
        return None

    def on_transition(self, rec, state_version):
        return None


def make_gpu_backend(detector_name, run):
    if detector_name == "jungfrau":
        from psana.gpu.backends.jungfrau import GpuJungfrauBackend

        return GpuJungfrauBackend(run)

    raise ValueError(f"Unsupported gpu_detector={detector_name!r}")
