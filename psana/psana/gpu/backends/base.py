from __future__ import annotations

from abc import ABC

from psana.gpu.execution import make_gpu_execution_backend
from psana.psexp.run import Run


class GpuDetectorBackend(ABC):
    """Detector-specific hooks for the generic GPU pipeline."""

    detector_name = None

    def __init__(self, run, execution_backend=None):
        self.run = run
        self.execution = execution_backend or make_gpu_execution_backend(
            'cupy',
            run=run,
            profiler=getattr(run, 'profiler', None),
            logger=getattr(run, 'logger', None),
        )
        self.device_cache = self.execution.make_residency_cache()

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


def make_gpu_backend(detector_name, run, execution_backend=None):
    if detector_name == "jungfrau":
        from psana.gpu.backends.jungfrau import GpuJungfrauBackend

        return GpuJungfrauBackend(run, execution_backend=execution_backend)

    raise ValueError(f"Unsupported gpu_detector={detector_name!r}")
