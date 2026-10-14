"""GPU detector processing, raw assembly, and result-buffer ownership."""

from dataclasses import dataclass
from functools import lru_cache
from typing import Iterator

import numpy as np

from psana.gpu.gpu_calib import (
    assemble_image as assemble_calib_image,
    fused_calib_gpu,
    prepare_geometry,
    prepare_geometry_from_arrays,
)
from psana.gpu.gpu_input import GpuDetectorBinding
from psana.gpu.gpudgram.batch import (
    LOC_NBYTES,
    LOC_NCOLS,
    LOC_OFFSET,
    LOC_RANK,
    LOC_STATUS,
    LOC_TYPE,
)
from psana.gpu.gpudgram.config import GpuFieldHandle
from psana.gpu.gpudgram.parser import STATUS_FOUND

_GATHER_U16_KERNEL_NAME = "gather_locator_u16_kernel"
_GATHER_F32_KERNEL_NAME = "gather_locator_f32_kernel"
_ZERO_MISSING_KERNEL_NAME = "zero_missing_rows_kernel"


def optimal_kernel_batch_size(det_shape, threads_per_block=256,
                               min_events=1, max_events=256):
    """Compute how many L1Accept events should be batched into one kernel launch
    to fully saturate the current GPU.

    The calibration kernel is a flat 1-D loop: one thread per pixel.  For a
    given detector shape the number of thread-blocks launched is:

        blocks_per_event = ceil(n_pixels_per_event / threads_per_block)

    The GPU can execute at most:

        gpu_capacity = n_SMs × min(max_blocks_per_SM,
                                    max_threads_per_SM / threads_per_block)

    blocks concurrently.  Batching multiple events into a single array and
    launching one kernel gives the GPU enough work to fill all SMs when:

        n_events ≥ ceil(gpu_capacity / blocks_per_event)

    For large detectors (Jungfrau 4M: 9.96M pixels → 38 912 blocks) a single
    event already exceeds the A100's 864-block capacity, so the optimal batch
    is 1.  For small detectors (ePix100a: 71K pixels → 279 blocks) batching
    3–4 events is needed to saturate the GPU.

    Parameters
    ----------
    det_shape         : tuple  (n_segs, nrows, ncols) for the GPU-path segments
    threads_per_block : int    CUDA block size (default 256, matches the kernel)
    min_events        : int    lower bound (default 1)
    max_events        : int    upper bound — prevents unreasonably large batches
                               on tiny detectors (default 256)

    Returns
    -------
    int  — optimal number of L1Accept events per kernel launch
    """
    import numpy as np

    n_pixels = int(np.prod(det_shape))
    if n_pixels <= 0:
        return max(1, min_events)

    blocks_per_event = (n_pixels + threads_per_block - 1) // threads_per_block

    try:
        import cupy as cp
        attrs = cp.cuda.Device(0).attributes
        n_sms              = attrs.get('MultiProcessorCount',        108)
        max_blocks_per_sm  = attrs.get('MaxBlocksPerMultiprocessor',  32)
        max_threads_per_sm = attrs.get('MaxThreadsPerMultiProcessor', 2048)
        # A block needs threads_per_block thread-slots; also limited by the
        # hardware block-count cap.
        blocks_per_sm = min(max_blocks_per_sm,
                            max_threads_per_sm // threads_per_block)
        gpu_capacity = n_sms * blocks_per_sm
    except Exception:
        # No GPU available (e.g. login node) — return a reasonable default.
        gpu_capacity = 108 * 8   # A100 estimate

    # ceil(gpu_capacity / blocks_per_event)
    optimal = (gpu_capacity + blocks_per_event - 1) // blocks_per_event
    return int(max(min_events, min(optimal, max_events)))


@dataclass
class EventContext:
    """GPU-resident result for one L1Accept event.

    Attributes
    ----------
    timestamp : int
        64-bit LCLS timestamp matching the CPU event.
    calib_gpu : cp.ndarray
        Calibrated float32 array on device, shape (n_segs, nrows, ncols).
        Stays on GPU; call .get() only for validation.
    raw_gpu : cp.ndarray or None
        Raw uint16 ADC values on device, same shape as calib_gpu.
        None when raw extraction was skipped.
    image_gpu : cp.ndarray or None
        Assembled 2-D detector image on device, shape (nrows_image, ncols_image).
        None when geometry was not loaded or unavailable.
    """
    timestamp: int
    calib_gpu: object           # cp.ndarray float32
    raw_gpu:   object = None    # cp.ndarray uint16 or None
    image_gpu: object = None    # cp.ndarray float32 or None


class GPUDetector:
    """Per-event GPU calibration fed by GPU XTC field locators.

    Configure-derived field handles identify one array payload per physical
    segment.  Event locators provide the byte offset for each handle in each
    dgram, so extraction does not depend on detector-specific XTC layout,
    recursive CPU parsing, fixed panel strides, or L1 child order.

    Parameters
    ----------
    det_shape        : tuple  (n_segs, nrows, ncols) for the full detector,
                       read from calibconst e.g. ``peds.shape[1:]``.
    peds_gpu         : cp.ndarray float32, flat, length 3 * prod(det_shape)
    gmask_gpu        : cp.ndarray float32, flat, same length
    binding : GpuDetectorBinding
        Run-scoped detector membership, field handles, and canonical segment
        order. The binding is independent of calibration and payload shape.
    """

    def __init__(self, det_shape, peds_gpu, gmask_gpu,
                 binding,
                 cmpars=None,
                 n_slots=2,
                 budget=None,
                 passthrough=False):
        self.det_shape         = tuple(det_shape)
        self.peds_gpu          = peds_gpu
        self.gmask_gpu         = gmask_gpu
        self._n_segs_calib     = int(det_shape[0])
        self._nrows            = int(det_shape[1])
        self._ncols            = int(det_shape[2])
        self._n_pix_seg        = self._nrows * self._ncols
        if not isinstance(binding, GpuDetectorBinding):
            raise TypeError("binding must be a GpuDetectorBinding")
        self.binding = binding
        self._canonical_segment_ids = binding.canonical_segment_ids
        if len(self._canonical_segment_ids) != self._n_segs_calib:
            raise ValueError(
                "canonical_segment_ids must contain one entry per detector "
                f"segment: got {len(self._canonical_segment_ids)}, "
                f"expected {self._n_segs_calib}"
            )
        self._canonical_segment_rows = binding.canonical_segment_rows
        self._budget = budget  # _GpuBudget | None

        self._field_handles_by_segment = binding.field_handles_by_segment
        self._sources_by_stream = binding.sources_by_stream
        for segment_id, handle in self._field_handles_by_segment.items():
            if handle.rank <= 0:
                raise ValueError(
                    f"segment {segment_id} handle must select an array field"
                )
        # Passthrough mode: bigdata is already calibrated float32 from the DRP.
        # Skip fused_calib_gpu entirely; just read and reshape the float32 pixels.
        self._passthrough = bool(passthrough)
        # Bytes per pixel in the bigdata stream.
        # Determined from drp_class_name via run.detinfo — the same source the
        # CPU path uses (Name::DataType in the XTC Names Configure container):
        #   drp_class_name == 'raw'  → uint16 (2 bytes)  → passthrough=False
        #   drp_class_name == 'fex'  → float32 (4 bytes) → passthrough=True
        # This is always known at construction time; no bigdata inspection needed.
        self._pixel_bytes = 4 if passthrough else 2
        incompatible = {
            segment_id: handle.element_size
            for segment_id, handle in self._field_handles_by_segment.items()
            if handle.element_size != self._pixel_bytes
        }
        if incompatible:
            raise ValueError(
                "Configure field element sizes do not match detector mode: "
                f"expected {self._pixel_bytes}, got {incompatible}"
            )
        # CPU-side cache for beginstep() change detection.
        self._peds_cpu_cache   = None
        self._gmask_cpu_cache  = None
        # Geometry scatter map for image assembly (set by setup_geometry()).
        self._scatter_ix   = None   # cp.ndarray int64, flat
        self._scatter_iy   = None   # cp.ndarray int64, flat
        self._image_shape  = None   # (nrows_img, ncols_img)
        # True when peds_gpu/gmask_gpu are shared views owned by another rank
        # (set by share_calib_between_gpu_peers() for follower BD ranks).
        # beginstep() skips the H→D write on followers to avoid a race with
        # the leader writing to the same shared GPU memory.
        self._is_calib_follower = False
        # Per-slot canonical raw and calibrated buffers. Both cover the whole
        # batch because EventContext exposes per-event views after all detector
        # work has been queued. Their EventPool lease prevents overwrite until
        # downstream consumers finish.
        # One buffer per EventPool slot, grown lazily to fit the first batch.
        # Reused across batches to prevent CuPy pool fragmentation that causes
        # OOM with large batch sizes.  Each slot's buffer is written by the GPU
        # calibration kernel and protected by the EventPool lease until its
        # registered terminal consumer completes.  on_gpu returns an independent
        # device copy; on_gpu_view keeps the slot leased through user GPU work.
        self._n_slots         = int(n_slots)
        self._raw_slot_bufs   = [None] * self._n_slots   # uint16 per slot
        self._calib_slot_bufs = [None] * self._n_slots   # cp.ndarray per slot
        self._present_slot_bufs = [None] * self._n_slots  # uint8 per slot
        # Common-mode correction — not yet implemented on GPU.
        if cmpars is not None:
            raise NotImplementedError(
                "Common-mode correction (cmpars) is not yet implemented for "
                "the GPU calibration path.  Pass cmpars=None (default) or "
                "omit the argument.  Implement Phase F3 (common-mode CUDA "
                "kernel) before using cmpars with GPUDetector."
            )

    @property
    def canonical_segment_ids(self):
        return self._canonical_segment_ids

    # ------------------------------------------------------------------
    # Geometry — image assembly
    # ------------------------------------------------------------------

    def setup_geometry(self, det):
        """Build the GPU image-scatter map from a psana detector."""
        geometry = prepare_geometry(det, self._canonical_segment_ids)
        if geometry is not None:
            self._scatter_ix, self._scatter_iy, self._image_shape = geometry

    def setup_geometry_from_arrays(self, ix_all, iy_all):
        """Build the GPU image-scatter map from coordinate-index arrays."""
        geometry = prepare_geometry_from_arrays(
            ix_all,
            iy_all,
            self._canonical_segment_ids,
        )
        if geometry is not None:
            self._scatter_ix, self._scatter_iy, self._image_shape = geometry

    def assemble_image(self, calib_gpu, stream=None):
        """Scatter canonical calibrated segments into a 2-D GPU image."""
        return assemble_calib_image(
            calib_gpu,
            self._scatter_ix,
            self._scatter_iy,
            self._image_shape,
            stream=stream,
        )

    # BeginStep hook
    # ------------------------------------------------------------------

    def beginstep(self, peds_flat, gmask_flat):
        """Refresh GPU calibration constants in-place after a BeginStep.

        Updates peds_gpu and gmask_gpu using CuPy ndarray.set(), which
        overwrites the existing device buffers without changing their GPU
        addresses.  This is required for future CUDA-graph compatibility
        (graphs capture buffer addresses at build time; in-place writes keep
        them valid across steps).

        Change detection: if the new constants are identical to the cached CPU
        arrays from the previous call, the H→D transfer is skipped.  This
        makes beginstep() a cheap no-op for single-gain-mode runs where
        constants don't change across steps.

        In passthrough mode (pre-calibrated bigdata) there are no calibration
        constants — this method is a no-op.

        Parameters
        ----------
        peds_flat  : np.ndarray float32, flat, length 3 * prod(det_shape)
            New pedestals from _compute_calib_constants_cpu().
        gmask_flat : np.ndarray float32, flat, same length
            New gain*mask from _compute_calib_constants_cpu().
        """
        if self._passthrough:
            return   # no calibration constants in passthrough mode

        # Compare against cached CPU arrays to skip unnecessary H->D transfers.
        if (self._peds_cpu_cache is not None
                and np.array_equal(peds_flat, self._peds_cpu_cache)
                and np.array_equal(gmask_flat, self._gmask_cpu_cache)):
            return   # no change — skip H->D

        if self._is_calib_follower:
            # peds_gpu/gmask_gpu are shared views into the leader's GPU
            # memory.  The leader's beginstep() will write the new values;
            # doing so here too would race-write to shared memory.
            self._peds_cpu_cache  = peds_flat.copy()
            self._gmask_cpu_cache = gmask_flat.copy()
            return

        # In-place update: same GPU buffer addresses (CUDA-graph-safe).
        self.peds_gpu.set(np.ascontiguousarray(peds_flat))
        self.gmask_gpu.set(np.ascontiguousarray(gmask_flat))

        # Cache the new CPU arrays for next comparison.
        self._peds_cpu_cache  = peds_flat.copy()
        self._gmask_cpu_cache = gmask_flat.copy()

    # ------------------------------------------------------------------
    # Production API
    # ------------------------------------------------------------------

    def memory_bytes(self) -> dict:
        """Return current VRAM usage broken down by category.

        All values are bytes on the GPU device.  Used by
        GpuEventManager.log_memory() for Phase-0 accounting.

        Categories
        ----------
        constants   peds_gpu + gmask_gpu (calibration constants)
        geometry    scatter_ix + scatter_iy (pixel coordinate maps)
        routing     reserved for run-scoped detector routing metadata
        calib_slots sum of allocated per-slot calibrated-output buffers
        raw_slots   sum of allocated per-slot raw-gather buffers
        total       sum of the above
        """
        def _nb(arr):
            return int(arr.nbytes) if arr is not None else 0

        constants   = _nb(self.peds_gpu) + _nb(self.gmask_gpu)
        geometry    = _nb(self._scatter_ix) + _nb(self._scatter_iy)
        routing     = 0
        calib_slots = sum(_nb(b) for b in (self._calib_slot_bufs or []))
        raw_slots   = (
            sum(_nb(b) for b in self._raw_slot_bufs)
            + sum(_nb(b) for b in self._present_slot_bufs)
        )
        total       = constants + geometry + routing + calib_slots + raw_slots
        return {
            'constants':   constants,
            'geometry':    geometry,
            'routing':     routing,
            'calib_slots': calib_slots,
            'raw_slots':   raw_slots,
            'total':       total,
        }

    def estimate_subbatch_bytes(self, n_events: int) -> int:
        """Estimate device VRAM needed for calibration of n_events events.

        Accounts for the two dominant variable allocations per batch:
          - Calibrated output buffer (float32): n_events × n_segs × nrows × ncols × 4
          - Raw-gather scratch buffer (uint16): n_events × n_segs × nrows × ncols × 2

        Calibration constants and geometry scatter maps are fixed allocations
        excluded from this per-subbatch estimate. GpuEventManager subtracts
        their measured bytes when deriving the default allowance; they are not
        currently reserved in _GpuBudget.committed().

        Parameters
        ----------
        n_events : int
            Number of L1Accept events in the proposed subbatch.

        Returns
        -------
        int  — estimated bytes, always >= 0.
        """
        if n_events <= 0:
            return 0
        n_segs = len(self._field_handles_by_segment)
        n_pix_per_event = n_segs * self._nrows * self._ncols
        # float32 calib output: 4 bytes/pixel in both modes.
        # Normal (uint16) mode also needs a raw-gather scratch buffer: +2 bytes/pixel.
        # Passthrough mode skips the scratch (bigdata is already float32).
        if self._passthrough:
            bytes_per_event = n_pix_per_event * 4
        else:
            bytes_per_event = n_pix_per_event * (4 + 2)
        return int(n_events * bytes_per_event)

    def _slot_buffer(self, buffers, slot, shape, dtype, label):
        """Return a reusable slot view, growing its backing array only."""
        cp = _cupy()
        nitems = int(np.prod(shape))
        needed = nitems * np.dtype(dtype).itemsize
        buf = buffers[slot]
        old_size = int(buf.nbytes) if buf is not None else 0
        if old_size < needed:
            delta = needed - old_size
            if self._budget is not None:
                self._budget.reserve(delta)
            try:
                new_buf = cp.empty(nitems, dtype=dtype)
            except Exception:
                if self._budget is not None:
                    self._budget.release(delta)
                raise
            buffers[slot] = new_buf
            buf = new_buf
            if __import__('os').environ.get('PSANA_GPU_MEM_DEBUG'):
                free_b, _ = cp.cuda.Device().mem_info
                print(
                    f'[GPU-MEM] {label} slot grow: '
                    f'need={needed/1e9:.1f}GB free={free_b/1e9:.1f}GB',
                    flush=True,
                )
        return buf[:nitems].reshape(shape)

    def process_batch(self, gpu_events,
                      stream=None, slot_id=None) -> Iterator[EventContext]:
        """Assemble canonical detector arrays from device field locators.

        ``GpuEventDgrams`` maps event streams to dense dgram indices once for
        every detector consumer. Configure-selected handles and GPU-produced
        locator rows provide every payload address; pixel bytes remain on the
        device.
        """
        cp = _cupy()
        events_info = [
            event_dgrams
            for event_dgrams in gpu_events
            if self.binding.has_sources(event_dgrams)
        ]

        if not events_info:
            return
        if slot_id is None:
            raise ValueError("GPUDetector.process_batch requires an EventPool slot_id")

        slot = int(slot_id) % self._n_slots
        batch_shape = (
            len(events_info) * self._n_segs_calib,
            self._nrows,
            self._ncols,
        )
        calib_slot = self._slot_buffer(
            self._calib_slot_bufs, slot, batch_shape, np.float32, "calib"
        )
        raw_slot = None
        if not self._passthrough:
            raw_slot = self._slot_buffer(
                self._raw_slot_bufs, slot, batch_shape, np.uint16, "raw"
            )
        present_slot = self._slot_buffer(
            self._present_slot_bufs,
            slot,
            (len(events_info), self._n_segs_calib),
            np.uint8,
            "field-presence",
        )

        sctx = stream if stream is not None else cp.cuda.Stream.null
        for event_index, event_dgrams in enumerate(events_info):
            lo = event_index * self._n_segs_calib
            hi = lo + self._n_segs_calib
            calib_out = calib_slot[lo:hi]
            raw_out = None if raw_slot is None else raw_slot[lo:hi]
            present = present_slot[event_index]

            with sctx:
                target = calib_out if self._passthrough else raw_out
                target.fill(0)
                present.fill(0)
                for dgram, output_row, _segment_id, handle in (
                    self.binding.iter_sources(event_dgrams)
                ):
                    locator_rows = dgram.locate(
                        handle, stream=sctx
                    ).wait_on(sctx)
                    _gather_locator_field_gpu(
                        dgram.data_gpu,
                        locator_rows,
                        dgram_index=dgram.dgram_index,
                        handle=handle,
                        output_row=output_row,
                        pixels_per_segment=self._n_pix_seg,
                        out=target,
                        present=present,
                    )

                if not self._passthrough:
                    fused_calib_gpu(
                        raw_out, self.peds_gpu, self.gmask_gpu, out=calib_out
                    )
                    _zero_missing_rows_gpu(calib_out, present)

            yield EventContext(
                timestamp=event_dgrams.timestamp,
                calib_gpu=calib_out,
                raw_gpu=raw_out,
            )


def _gather_locator_field_gpu(
    data_gpu,
    locator_rows,
    dgram_index,
    handle,
    output_row,
    pixels_per_segment,
    out,
    present,
    threads=256,
):
    """Copy one located field into its canonical detector row."""
    cp = _cupy()
    if data_gpu.dtype != cp.uint8 or data_gpu.ndim != 1:
        raise TypeError("data_gpu must be a 1-dimensional uint8 array")
    if locator_rows.dtype != cp.uint64 or locator_rows.ndim != 2:
        raise TypeError("locator_rows must be a 2-dimensional uint64 array")
    if locator_rows.shape[1] != LOC_NCOLS:
        raise ValueError("locator_rows has the wrong column count")
    if not isinstance(handle, GpuFieldHandle):
        raise TypeError("handle must be a GpuFieldHandle")
    expected_dtype = cp.float32 if handle.element_size == 4 else cp.uint16
    if handle.element_size not in (2, 4) or out.dtype != expected_dtype:
        raise TypeError(
            f"field element_size={handle.element_size} is incompatible with "
            f"output dtype {out.dtype}"
        )
    if present.dtype != cp.uint8 or present.ndim != 1:
        raise TypeError("present must be a 1-dimensional uint8 array")

    blocks = (int(pixels_per_segment) + threads - 1) // threads
    kernel = (
        _gather_u16_kernel() if out.dtype == cp.uint16
        else _gather_f32_kernel()
    )
    kernel(
        (blocks,),
        (threads,),
        (
            data_gpu,
            np.uint64(data_gpu.nbytes),
            locator_rows,
            np.uint64(dgram_index),
            out.ravel(),
            present,
            np.uint64(output_row),
            np.uint64(pixels_per_segment),
            np.uint64(handle.type),
            np.uint64(handle.rank),
        ),
    )
    return out


def _zero_missing_rows_gpu(out, present, threads=256):
    """Restore zero output for fields absent or rejected by the locator copy."""
    n_segments = int(out.shape[0])
    pixels_per_segment = int(np.prod(out.shape[1:]))
    blocks = (pixels_per_segment + threads - 1) // threads
    _zero_missing_kernel()(
        (blocks, n_segments),
        (threads,),
        (out.ravel(), present, np.uint64(pixels_per_segment)),
    )
    return out


@lru_cache(maxsize=1)
def _cupy():
    import cupy as cp

    return cp


@lru_cache(maxsize=1)
def _gather_u16_kernel():
    cp = _cupy()
    return cp.RawKernel(
        _gather_kernel_source(),
        _GATHER_U16_KERNEL_NAME,
        options=("--std=c++17",),
    )


@lru_cache(maxsize=1)
def _gather_f32_kernel():
    cp = _cupy()
    return cp.RawKernel(
        _gather_kernel_source(),
        _GATHER_F32_KERNEL_NAME,
        options=("--std=c++17",),
    )


@lru_cache(maxsize=1)
def _zero_missing_kernel():
    cp = _cupy()
    return cp.RawKernel(
        _gather_kernel_source(),
        _ZERO_MISSING_KERNEL_NAME,
        options=("--std=c++17",),
    )


@lru_cache(maxsize=1)
def _gather_kernel_source():
    return f"""

namespace {{

template <typename T>
__device__ __forceinline__ void gather_locator_field(
    const unsigned char* data,
    unsigned long long data_nbytes,
    const unsigned long long* locators,
    unsigned long long dgram_index,
    T* out,
    unsigned char* present,
    unsigned long long output_row,
    unsigned long long pixels_per_segment,
    unsigned long long expected_type,
    unsigned long long expected_rank)
{{
    const unsigned long long* locator =
        locators + dgram_index * {LOC_NCOLS};
    const unsigned long long nbytes = locator[{LOC_NBYTES}];
    const unsigned long long offset = locator[{LOC_OFFSET}];
    const unsigned long long expected_nbytes =
        pixels_per_segment * sizeof(T);
    if (locator[{LOC_STATUS}] != {STATUS_FOUND} ||
        locator[{LOC_TYPE}] != expected_type ||
        locator[{LOC_RANK}] != expected_rank ||
        nbytes != expected_nbytes ||
        offset > data_nbytes || nbytes > data_nbytes - offset) {{
        return;
    }}

    const unsigned long long pixel =
        (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (pixel >= pixels_per_segment) return;
    const T* src = reinterpret_cast<const T*>(data + offset);
    out[output_row * pixels_per_segment + pixel] = src[pixel];
    if (pixel == 0) present[output_row] = 1;
}}

}} // namespace

extern "C" __global__
void {_GATHER_U16_KERNEL_NAME}(
    const unsigned char* data,
    unsigned long long data_nbytes,
    const unsigned long long* locators,
    unsigned long long dgram_index,
    unsigned short* out,
    unsigned char* present,
    unsigned long long output_row,
    unsigned long long pixels_per_segment,
    unsigned long long expected_type,
    unsigned long long expected_rank)
{{
    gather_locator_field<unsigned short>(
        data, data_nbytes, locators, dgram_index, out, present, output_row,
        pixels_per_segment, expected_type, expected_rank);
}}

extern "C" __global__
void {_GATHER_F32_KERNEL_NAME}(
    const unsigned char* data,
    unsigned long long data_nbytes,
    const unsigned long long* locators,
    unsigned long long dgram_index,
    float* out,
    unsigned char* present,
    unsigned long long output_row,
    unsigned long long pixels_per_segment,
    unsigned long long expected_type,
    unsigned long long expected_rank)
{{
    gather_locator_field<float>(
        data, data_nbytes, locators, dgram_index, out, present, output_row,
        pixels_per_segment, expected_type, expected_rank);
}}

extern "C" __global__
void {_ZERO_MISSING_KERNEL_NAME}(
    float* out,
    const unsigned char* present,
    unsigned long long pixels_per_segment)
{{
    const unsigned long long pixel =
        (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned long long segment = blockIdx.y;
    if (pixel >= pixels_per_segment || present[segment]) return;
    out[segment * pixels_per_segment + pixel] = 0.0f;
}}
"""
