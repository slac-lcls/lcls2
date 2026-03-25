from __future__ import annotations

import time
from typing import Any

import numpy as np


class GpuResidencyCache:
    """Lazy per-process cache for arrays resident on the selected GPU backend.

    The cache tries to upload to CuPy device memory when available. If CuPy or a
    working CUDA context is unavailable, it falls back to storing NumPy arrays so
    the rest of the prototype can continue to run.
    """

    def __init__(self, profiler=None, logger=None, uploader=None):
        self.profiler = profiler
        self.logger = logger
        self._entries: dict[Any, Any] = {}
        self._storage_kind: dict[Any, str] = {}
        self._uploader = uploader
        self._cupy = None
        self._cupy_checked = False

    def get(self, key, default=None):
        return self._entries.get(key, default)

    def get_kind(self, key, default=None):
        return self._storage_kind.get(key, default)

    def get_or_upload(self, key, host_array):
        if key in self._entries:
            return self._entries[key]

        start = time.perf_counter()
        arr, kind = self._upload(host_array)
        dt_s = time.perf_counter() - start

        self._entries[key] = arr
        self._storage_kind[key] = kind
        if self.profiler is not None:
            self.profiler.record_cache_upload(dt_s)
        return arr

    def invalidate(self, predicate=None):
        if predicate is None:
            self._entries.clear()
            self._storage_kind.clear()
            return

        to_remove = [key for key in self._entries if predicate(key)]
        for key in to_remove:
            self._entries.pop(key, None)
            self._storage_kind.pop(key, None)

    def _upload(self, host_array):
        if self._uploader is not None:
            return self._uploader(host_array)

        cp = self._get_cupy()
        np_array = np.ascontiguousarray(host_array)
        if cp is None:
            return np_array, "host"

        try:
            return cp.asarray(np_array), "device"
        except Exception as exc:
            if self.logger is not None:
                self.logger.debug(
                    "Falling back to host cache storage after CuPy upload failed: %s",
                    exc,
                    exc_info=True,
                )
            return np_array, "host"

    def _get_cupy(self):
        if self._cupy_checked:
            return self._cupy

        self._cupy_checked = True
        try:
            import cupy as cp
        except Exception:
            cp = None
        self._cupy = cp
        return self._cupy
