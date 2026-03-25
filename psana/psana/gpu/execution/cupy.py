from __future__ import annotations

import time

import numpy as np

from psana.gpu.execution.base import GpuExecutionBackend


class CupyExecutionBackend(GpuExecutionBackend):
    runtime_name = 'cupy'

    def __init__(self, run, profiler=None, logger=None):
        super().__init__(run, profiler=profiler, logger=logger)
        self._cp = None
        self._cupyx = None
        self._gpu_import_checked = False
        self._pixel_index_cache = {}

    def upload_cache_array(self, host_array):
        cp = self._get_cupy()
        np_array = np.ascontiguousarray(host_array)
        if cp is None:
            return np_array, 'host'
        try:
            return cp.asarray(np_array), 'device'
        except Exception as exc:
            if self.logger is not None:
                self.logger.debug(
                    'Falling back to host cache storage after CuPy upload failed: %s',
                    exc,
                    exc_info=True,
                )
            return np_array, 'host'

    def allocate_slot_stream(self, slot):
        cp = self._get_cupy()
        if cp is not None and slot.stream is None:
            try:
                slot.stream = cp.cuda.Stream(non_blocking=True)
            except Exception:
                slot.stream = None
        return None

    def allocate_slot_buffers(self, slot, raw_shape, raw_dtype):
        cp = self._get_cupy()
        if cp is not None:
            try:
                slot.dev_raw = cp.empty(raw_shape, dtype=raw_dtype)
                slot.dev_out = cp.empty(raw_shape, dtype=np.float32)
                return
            except Exception as exc:
                if self.logger is not None:
                    self.logger.debug(
                        'Failed to allocate CuPy raw buffers for Jungfrau slot %d: %s',
                        slot.slot_id,
                        exc,
                        exc_info=True,
                    )
        slot.dev_raw = None
        slot.dev_out = None

    def empty_host_buffer(self, shape, dtype):
        cupyx = self._get_cupyx()
        if cupyx is not None:
            try:
                return cupyx.empty_pinned(shape, dtype=dtype)
            except Exception as exc:
                if self.logger is not None:
                    self.logger.debug(
                        'Failed to allocate pinned host buffer for Jungfrau raw staging: %s',
                        exc,
                        exc_info=True,
                    )
        return np.empty(shape, dtype=dtype)

    def copy_raw_to_execution(self, host_raw, slot):
        copy_start = time.perf_counter()
        cp = self._get_cupy()
        if cp is not None and slot.dev_raw is not None:
            copy_start_evt = cp.cuda.Event()
            copy_stop_evt = cp.cuda.Event()
            try:
                if slot.stream is not None:
                    with slot.stream:
                        copy_start_evt.record()
                        slot.dev_raw.set(host_raw)
                        copy_stop_evt.record()
                else:
                    copy_start_evt.record()
                    slot.dev_raw.set(host_raw)
                    copy_stop_evt.record()
                copy_stop_evt.synchronize()
                if self.profiler is not None:
                    self.profiler.record_copy(cp.cuda.get_elapsed_time(copy_start_evt, copy_stop_evt) / 1e3)
                return slot.dev_raw, 'device'
            except Exception as exc:
                if self.logger is not None:
                    self.logger.debug(
                        'Falling back to host-backed raw slot after CuPy copy failed: %s',
                        exc,
                        exc_info=True,
                    )
        if self.profiler is not None:
            self.profiler.record_copy(time.perf_counter() - copy_start)
        return host_raw, 'host'

    def compute_jungfrau_calib_v3(self, raw_obj, ccons, slot):
        cp = self._get_cupy()
        if cp is None or slot.dev_out is None or ccons is None:
            return None, None
        if not isinstance(raw_obj, cp.ndarray) or not isinstance(ccons, cp.ndarray):
            return None, None

        kernel_start = cp.cuda.Event()
        kernel_stop = cp.cuda.Event()
        try:
            if slot.stream is not None:
                with slot.stream:
                    kernel_start.record()
                    self._jungfrau_calib_v3(raw_obj, ccons, slot.dev_out)
                    kernel_stop.record()
            else:
                kernel_start.record()
                self._jungfrau_calib_v3(raw_obj, ccons, slot.dev_out)
                kernel_stop.record()

            kernel_stop.synchronize()
            if self.profiler is not None:
                self.profiler.record_kernel(cp.cuda.get_elapsed_time(kernel_start, kernel_stop) / 1e3)
            return cp.array(slot.dev_out, copy=True), 'device'
        except Exception as exc:
            if self.logger is not None:
                self.logger.debug(
                    'Falling back to host Jungfrau calibration after GPU compute failed: %s',
                    exc,
                    exc_info=True,
                )
            return None, None

    def to_host_array(self, arr):
        cp = self._get_cupy()
        if cp is not None and isinstance(arr, cp.ndarray):
            return cp.asnumpy(arr)
        return np.asarray(arr)

    def array_from_evt_data(self, arr, storage, copy=False):
        cp = self._get_cupy()
        if storage == 'device' and cp is not None:
            return cp.array(arr, copy=copy)
        return np.array(arr, copy=copy)

    def _jungfrau_calib_v3(self, raw, ccons, out):
        cp = self._get_cupy()
        raw_flat = raw.reshape(-1)
        out_flat = out.reshape(-1)
        size = int(raw_flat.size)
        base_idx = self._pixel_index(size)
        gain_idx = (raw_flat >> 14).astype(cp.int32, copy=False)
        cc_idx = 2 * (base_idx + size * gain_idx)
        pedoff = cp.take(ccons, cc_idx)
        gain = cp.take(ccons, cc_idx + 1)
        out_flat[...] = (cp.bitwise_and(raw_flat, 0x3FFF).astype(cp.float32) - pedoff) * gain
        out.shape = raw.shape
        return out

    def _pixel_index(self, size):
        cp = self._get_cupy()
        if cp is None:
            return None
        idx = self._pixel_index_cache.get(size)
        if idx is None:
            idx = cp.arange(size, dtype=cp.int32)
            self._pixel_index_cache[size] = idx
        return idx

    def _get_cupy(self):
        if self._gpu_import_checked:
            return self._cp

        self._gpu_import_checked = True
        try:
            import cupy as cp
        except Exception:
            cp = None
        try:
            import cupyx
        except Exception:
            cupyx = None

        if cp is not None:
            try:
                cp.cuda.runtime.getDeviceCount()
            except Exception as exc:
                if self.logger is not None:
                    self.logger.debug(
                        'Disabling CuPy execution backend because CUDA runtime is unavailable: %s',
                        exc,
                        exc_info=True,
                    )
                cp = None
                cupyx = None

        self._cp = cp
        self._cupyx = cupyx
        return self._cp

    def _get_cupyx(self):
        if not self._gpu_import_checked:
            self._get_cupy()
        return self._cupyx
