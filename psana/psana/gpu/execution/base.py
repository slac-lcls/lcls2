from __future__ import annotations

import time
from abc import ABC

import numpy as np

from psana.gpu.cache import GpuResidencyCache


class GpuExecutionBackend(ABC):
    runtime_name = None

    def __init__(self, run, profiler=None, logger=None):
        self.run = run
        self.profiler = profiler
        self.logger = logger or getattr(run, 'logger', None)

    def make_residency_cache(self):
        return GpuResidencyCache(
            profiler=self.profiler,
            logger=self.logger,
            uploader=self.upload_cache_array,
        )

    def upload_cache_array(self, host_array):
        return np.ascontiguousarray(host_array), 'host'

    def allocate_slot_stream(self, slot):
        return None

    def allocate_slot_buffers(self, slot, raw_shape, raw_dtype):
        slot.dev_raw = None
        slot.dev_out = None

    def empty_host_buffer(self, shape, dtype):
        return np.empty(shape, dtype=dtype)

    def copy_raw_to_execution(self, host_raw, slot):
        start = time.perf_counter()
        arr = np.asarray(host_raw)
        if self.profiler is not None:
            self.profiler.record_copy(time.perf_counter() - start)
        return arr, 'host'

    def compute_jungfrau_calib_v3(self, raw_obj, ccons, slot):
        return None, None

    def to_host_array(self, arr):
        return np.asarray(arr)

    def array_from_evt_data(self, arr, storage, copy=False):
        return np.array(arr, copy=copy)
