from __future__ import annotations

import time

import numpy as np
from psana.psexp.run import Run

from psana.gpu.backends.base import GpuDetectorBackend


class GpuJungfrauBackend(GpuDetectorBackend):
    """Prototype 1 Jungfrau backend with raw staging and lazy cache uploads."""

    detector_name = "jungfrau"

    def __init__(self, run):
        super().__init__(run)
        self._resolved_det_name = None
        self._det_raw = None
        self._host_calib = None
        self._cache_version = None
        self._cp = None
        self._cupyx = None
        self._gpu_import_checked = False
        self._pixel_index_cache = {}

    def make_detector(self, name, accept_missing=False, **kwargs):
        requested_name = name
        if name == self.detector_name:
            resolved = self._resolve_detector_name()
            if resolved is not None:
                name = resolved
        det = Run.Detector(self.run, name, accept_missing=accept_missing, **kwargs)
        if requested_name == self.detector_name or name == self._resolve_detector_name():
            return GpuJungfrauDetector(det, self)
        return det

    def allocate_slot_buffers(self, slot):
        cp = self._get_cupy()
        if cp is not None and slot.stream is None:
            try:
                slot.stream = cp.cuda.Stream(non_blocking=True)
            except Exception:
                slot.stream = None
        return None

    def pack_l1_to_host(self, rec, slot):
        det_raw = self._get_det_raw()
        if det_raw is None:
            return None

        segs = det_raw._segments(rec.event)
        if segs is None:
            return None

        seg_ids = tuple(det_raw._segment_numbers)
        first_seg = segs[seg_ids[0]].raw
        raw_shape = first_seg.shape if len(seg_ids) == 1 else (len(seg_ids),) + first_seg.shape
        raw_dtype = first_seg.dtype
        self._ensure_slot_raw_buffers(slot, raw_shape, raw_dtype)

        if len(seg_ids) == 1:
            np.copyto(slot.host_raw, first_seg, casting="no")
        else:
            for idx, seg_id in enumerate(seg_ids):
                np.copyto(slot.host_raw[idx], segs[seg_id].raw, casting="no")

        slot.metadata["raw_shape"] = tuple(raw_shape)
        slot.metadata["raw_dtype"] = str(raw_dtype)
        slot.metadata["segment_ids"] = seg_ids
        slot.metadata["det_name"] = self._resolve_detector_name()
        return slot.host_raw

    def ensure_device_cache(self, rec, slot):
        host_calib = self._ensure_host_calib(rec)
        if not host_calib:
            return None

        cache_version = self._cache_identity(host_calib)
        if self._cache_version != cache_version:
            self.device_cache.invalidate()
            self._cache_version = cache_version

        device_cache = {}
        storage = {}
        for name, host_array in host_calib.items():
            key = cache_version + (name,)
            device_cache[name] = self.device_cache.get_or_upload(key, host_array)
            storage[name] = self.device_cache.get_kind(key)

        slot.metadata["device_cache"] = device_cache
        slot.metadata["device_cache_storage"] = storage
        slot.metadata["device_cache_key"] = cache_version
        return device_cache

    def transfer_to_device(self, rec, slot):
        if slot.host_raw is None:
            return None

        copy_start = time.perf_counter()
        cp = self._get_cupy()
        if cp is not None and slot.dev_raw is not None:
            copy_start_evt = cp.cuda.Event()
            copy_stop_evt = cp.cuda.Event()
            try:
                if slot.stream is not None:
                    with slot.stream:
                        copy_start_evt.record()
                        slot.dev_raw.set(slot.host_raw)
                        copy_stop_evt.record()
                else:
                    copy_start_evt.record()
                    slot.dev_raw.set(slot.host_raw)
                    copy_stop_evt.record()
                copy_stop_evt.synchronize()
                if getattr(self.run, "profiler", None) is not None:
                    self.run.profiler.record_copy(cp.cuda.get_elapsed_time(copy_start_evt, copy_stop_evt) / 1e3)
                storage = "device"
                raw_obj = slot.dev_raw
            except Exception as exc:
                self.run.logger.debug(
                    "Falling back to host-backed raw slot after CuPy copy failed: %s",
                    exc,
                    exc_info=True,
                )
                storage = "host"
                raw_obj = slot.host_raw
                if getattr(self.run, "profiler", None) is not None:
                    self.run.profiler.record_copy(time.perf_counter() - copy_start)
        else:
            storage = "host"
            raw_obj = slot.host_raw
            if getattr(self.run, "profiler", None) is not None:
                self.run.profiler.record_copy(time.perf_counter() - copy_start)

        slot.metadata["raw_storage"] = storage
        slot.metadata["raw_device"] = raw_obj
        rec.event._gpu_raw = raw_obj
        rec.event._gpu_raw_storage = storage
        rec.event._gpu_cache = slot.metadata.get("device_cache")
        return raw_obj

    def launch_compute(self, rec, slot):
        raw_obj = slot.metadata.get("raw_device")
        if raw_obj is None:
            return None

        cp = self._get_cupy()
        device_cache = slot.metadata.get("device_cache") or {}
        ccons = device_cache.get("ccons")

        if cp is not None and slot.metadata.get("raw_storage") == "device" and ccons is not None:
            kernel_start = cp.cuda.Event()
            kernel_stop = cp.cuda.Event()
            try:
                if slot.stream is not None:
                    with slot.stream:
                        kernel_start.record()
                        self._gpu_calib_v3(raw_obj, ccons, slot.dev_out)
                        kernel_stop.record()
                else:
                    kernel_start.record()
                    self._gpu_calib_v3(raw_obj, ccons, slot.dev_out)
                    kernel_stop.record()

                kernel_stop.synchronize()
                if getattr(self.run, "profiler", None) is not None:
                    self.run.profiler.record_kernel(cp.cuda.get_elapsed_time(kernel_start, kernel_stop) / 1e3)
                calib = cp.array(slot.dev_out, copy=True)
                storage = "device"
            except Exception as exc:
                self.run.logger.debug(
                    "Falling back to host Jungfrau calibration after GPU compute failed: %s",
                    exc,
                    exc_info=True,
                )
                calib = self._cpu_calib_v3(slot.host_raw, device_cache)
                storage = "host"
        else:
            kernel_start = time.perf_counter()
            calib = self._cpu_calib_v3(slot.host_raw, device_cache)
            storage = "host"
            if getattr(self.run, "profiler", None) is not None:
                self.run.profiler.record_kernel(time.perf_counter() - kernel_start)

        slot.metadata["calib_storage"] = storage
        slot.metadata["calib_result"] = calib
        rec.event._gpu_calib = calib
        rec.event._gpu_calib_storage = storage
        return calib

    def on_transition(self, rec, state_version):
        # Conservative Prototype 1 policy: force cache refresh after transitions.
        self.device_cache.invalidate()
        self._host_calib = None
        self._cache_version = None

    def _ensure_host_calib(self, rec):
        if self._host_calib is not None:
            return self._host_calib

        det_raw = self._get_det_raw()
        if det_raw is None:
            return None

        shared = getattr(det_raw, "_jf_shared", None)
        arrays = None
        if shared:
            arrays = self._arrays_from_shared(shared)

        if not arrays:
            arrays = self._arrays_from_odc(det_raw, rec.event)

        self._host_calib = arrays
        return self._host_calib

    def _get_det_raw(self):
        if self._det_raw is not None:
            return self._det_raw

        det_name = self._resolve_detector_name()
        if det_name is None:
            return None

        det = Run.Detector(self.run, det_name)
        self._det_raw = getattr(det, "raw", None)
        return self._det_raw

    def _resolve_detector_name(self):
        if self._resolved_det_name is not None:
            return self._resolved_det_name

        for (det_name, xface_name), drp_class in self.run.dsparms.det_classes["normal"].items():
            class_name = getattr(drp_class, "__name__", "").lower()
            module_name = getattr(drp_class, "__module__", "").lower()
            if xface_name != "raw":
                continue
            if class_name.startswith("jungfrau_raw") or module_name.endswith(".jungfrau"):
                self._resolved_det_name = det_name
                return self._resolved_det_name

        self.run.logger.warning(
            "Unable to resolve a Jungfrau detector name from gpu_detector=%s",
            self.detector_name,
        )
        return None

    def _arrays_from_shared(self, shared):
        arrays = {}
        for name in ("ccons", "poff", "gfac", "gmask", "mask"):
            value = shared.get(name)
            if value is not None:
                arrays[name] = value
        return arrays or None

    def _arrays_from_odc(self, det_raw, evt):
        import psana.detector.UtilsJungfrau as uj

        odc = getattr(det_raw, "_odc", None)
        if odc is None or getattr(odc, "cversion", None) != uj.CALIB_CPP_V3:
            odc = uj.DetCache(det_raw, evt, cversion=uj.CALIB_CPP_V3)
            det_raw._odc = odc

        arrays = {}
        for name in ("ccons", "poff", "gfac", "gmask", "mask"):
            value = getattr(odc, name, None)
            if value is not None:
                arrays[name] = value
        return arrays or None

    def _cache_identity(self, arrays):
        ccons = arrays.get("ccons")
        shape = tuple(getattr(ccons, "shape", ()))
        dtype = str(getattr(ccons, "dtype", "unknown"))
        return (
            self.detector_name,
            self.run.runnum,
            self._resolve_detector_name(),
            shape,
            dtype,
        )


    def _gpu_calib_v3(self, raw, ccons, out):
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

    def _cpu_calib_v3(self, raw, device_cache):
        import psana.pycalgos.utilsdetector as ud

        if raw is None:
            return None

        ccons = device_cache.get("ccons")
        if ccons is None:
            return np.asarray(raw)

        host_ccons = self._to_host_array(ccons)
        out = np.empty(raw.shape, dtype=np.float32)
        return ud.calib_jungfrau_v3(np.asarray(raw), host_ccons, 512 * 1024, out)

    def _pixel_index(self, size):
        cp = self._get_cupy()
        if cp is None:
            return None
        idx = self._pixel_index_cache.get(size)
        if idx is None:
            idx = cp.arange(size, dtype=cp.int32)
            self._pixel_index_cache[size] = idx
        return idx

    def _to_host_array(self, arr):
        cp = self._get_cupy()
        if cp is not None and isinstance(arr, cp.ndarray):
            return cp.asnumpy(arr)
        return np.asarray(arr)

    def _ensure_slot_raw_buffers(self, slot, raw_shape, raw_dtype):
        if slot.raw_shape == tuple(raw_shape) and slot.raw_dtype == raw_dtype and slot.host_raw is not None:
            return

        slot.raw_shape = tuple(raw_shape)
        slot.raw_dtype = raw_dtype
        slot.host_raw = self._empty_host_buffer(raw_shape, raw_dtype)

        cp = self._get_cupy()
        if cp is not None:
            try:
                slot.dev_raw = cp.empty(raw_shape, dtype=raw_dtype)
                slot.dev_out = cp.empty(raw_shape, dtype=np.float32)
            except Exception as exc:
                self.run.logger.debug(
                    "Failed to allocate CuPy raw buffers for Jungfrau slot %d: %s",
                    slot.slot_id,
                    exc,
                    exc_info=True,
                )
                slot.dev_raw = None
                slot.dev_out = None
        else:
            slot.dev_raw = None
            slot.dev_out = None

    def _empty_host_buffer(self, shape, dtype):
        cupyx = self._get_cupyx()
        if cupyx is not None:
            try:
                return cupyx.empty_pinned(shape, dtype=dtype)
            except Exception as exc:
                self.run.logger.debug(
                    "Failed to allocate pinned host buffer for Jungfrau raw staging: %s",
                    exc,
                    exc_info=True,
                )
        return np.empty(shape, dtype=dtype)

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

        self._cp = cp
        self._cupyx = cupyx
        return self._cp

    def _get_cupyx(self):
        if not self._gpu_import_checked:
            self._get_cupy()
        return self._cupyx


class GpuJungfrauDetector:
    def __init__(self, cpu_detector, backend):
        self._cpu_detector = cpu_detector
        self._backend = backend
        self.raw = GpuJungfrauRaw(cpu_detector.raw, backend)

    def __getattr__(self, name):
        return getattr(self._cpu_detector, name)


class GpuJungfrauRaw:
    def __init__(self, cpu_raw, backend):
        self._cpu_raw = cpu_raw
        self._backend = backend

    def raw(self, evt, copy=True):
        gpu_raw = getattr(evt, "_gpu_raw", None)
        if gpu_raw is None:
            return self._cpu_raw.raw(evt, copy=copy)

        storage = getattr(evt, "_gpu_raw_storage", "host")
        if storage == "device":
            cp = self._backend._get_cupy()
            if cp is not None:
                return cp.array(gpu_raw, copy=copy)
        return np.array(gpu_raw, copy=copy)

    def calib(self, evt, **kwargs):
        if self._kwargs_require_cpu(kwargs):
            return self._cpu_raw.calib(evt, **kwargs)

        gpu_calib = getattr(evt, "_gpu_calib", None)
        if gpu_calib is not None:
            return gpu_calib
        return self._cpu_raw.calib(evt, **kwargs)

    def _kwargs_require_cpu(self, kwargs):
        if not kwargs:
            return False

        cversion = kwargs.get("cversion", 3)
        size_blk = kwargs.get("size_blk", 512 * 1024)
        allowed = set(kwargs.keys()) <= {"cversion", "size_blk"}
        return (not allowed) or cversion != 3 or size_blk != 512 * 1024

    def __getattr__(self, name):
        return getattr(self._cpu_raw, name)
