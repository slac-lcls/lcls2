"""
gpu_calib.py — GPU-accelerated Jungfrau calibration (Step 5a).

Public API
----------
fused_calib_gpu(raw_gpu, peds_gpu, gmask_gpu) -> calib_gpu

    Runs the jungfrau_calib_kernel on GPU.  All three input arrays must
    already live on device.  Returns a float32 CuPy array with the same
    shape as raw_gpu.

prep_calib_constants(det) -> (peds_gpu, gmask_gpu)

    Helper: extract pedestals and gain*mask from a psana Detector object,
    compute gmask = (1/pixel_gain) * mask on CPU, transfer both to GPU.
    Call once per run; pass the results to fused_calib_gpu() per event.

Notes
-----
- Only the calibration step is implemented (no common-mode correction,
  no decompression).  Intended for uncompressed Jungfrau data.
- Calibration constant layout (both peds_gpu and gmask_gpu):
    flat float32, length 3 * npixels, mode-major C order.
    Index: mode * npixels + pixel_index
    where npixels = nsegs * nrows * ncols.
- Gain bit mapping (top 2 bits of each raw uint16 value):
    0 -> gain mode 0 (g0 / low-gain)
    1 -> gain mode 1 (g1 / medium-gain)
    3 -> gain mode 2 (g2 / high-gain)
    2 -> bad pixel -> output 0.0
"""

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterator

import numpy as np

from psana.gpu.gpu_kvikio_read import DESC_DEVICE_OFFSET, DESC_READ_SIZE, DESC_STREAM_ID

# Named constants for the Jungfrau 0.5M panel layout — used for documentation
# and as explicit labels in Jungfrau-specific code.  GPUDetector auto-detects
# these values from the XTC tree at runtime so they are NOT used as defaults.
JUNGFRAU_SEG_STRIDE_BYTES = 1048658   # XTC child extent for one 0.5M panel
JUNGFRAU_DGRAM_RAW_OFFSET = 80        # bytes from dgram start to raw uint16 pixels

_KERNEL_NAME = "jungfrau_calib_kernel"


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


def _detect_dgram_layout(dgram_bytes):
    """Walk the XTC tree of one bigdata L1Accept dgram to determine the
    per-segment stride and the byte offset of the raw pixel data.

    Works for any uncompressed area detector whose bigdata dgram is laid out
    as: Dgram-header (24 B) + N × child-XTC, where each child-XTC holds a
    small metadata XTC followed by the raw pixel data leaf.

    Parameters
    ----------
    dgram_bytes : bytes-like, at least 512 bytes from the dgram start

    Returns
    -------
    seg_stride_bytes : int
        Bytes from the start of one child-XTC to the start of the next.
        Equals the first child-XTC's ``extent`` field.
    raw_data_offset : int
        Byte offset from the dgram start to the first raw pixel.

    Raises
    ------
    RuntimeError if no data leaf is found within the first child-XTC.
    """
    n = len(dgram_bytes)

    # First child-XTC starts at offset 24 (after the 24-byte Dgram header).
    # Its ``extent`` field lives at bytes [32:36] of the dgram.
    seg_stride = int.from_bytes(dgram_bytes[32:36], 'little')

    # Walk nested XTCs inside the first child-XTC payload (starts at 36).
    # The data leaf is the first nested XTC whose payload exceeds half the
    # child-XTC's total size — i.e., it contains the bulk of the pixel data.
    walk_end = min(24 + seg_stride, n)
    pos = 36
    while pos + 12 <= walk_end:
        extent = int.from_bytes(dgram_bytes[pos + 8: pos + 12], 'little')
        if extent <= 12:
            break
        if (extent - 12) * 2 > seg_stride:   # large leaf → raw data
            return seg_stride, pos + 12
        pos += extent

    raise RuntimeError(
        f"_detect_dgram_layout: no raw-data leaf found "
        f"(seg_stride={seg_stride}, walked bytes 36..{pos})"
    )


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
        None when raw extraction was skipped (e.g. calibrate() shortcut).
    image_gpu : cp.ndarray or None
        Assembled 2-D detector image on device, shape (nrows_image, ncols_image).
        None when geometry was not loaded or unavailable.
    stream : cp.cuda.Stream or None
        The CUDA stream on which the arrays above were produced.
    """
    timestamp: int
    calib_gpu: object           # cp.ndarray float32
    raw_gpu:   object = None    # cp.ndarray uint16 or None
    image_gpu: object = None    # cp.ndarray float32 or None
    stream:    object = None    # cp.cuda.Stream or None


class GPUDetector:
    """Per-event GPU calibration for an uncompressed Jungfrau detector.

    Handles both single-segment test fixtures and multi-segment real bigdata
    dgrams.  The XTC header overhead (raw_data_offset) and per-segment stride
    (seg_stride_bytes) are auto-detected from the first bigdata dgram seen,
    so no detector-specific constants need to be hard-coded by the caller.

    Parameters
    ----------
    det_shape        : tuple  (n_segs, nrows, ncols) for the full detector,
                       read from calibconst e.g. ``peds.shape[1:]``.
    peds_gpu         : cp.ndarray float32, flat, length 3 * prod(det_shape)
    gmask_gpu        : cp.ndarray float32, flat, same length
    raw_data_offset  : int or None
        Bytes from the dgram start to the first raw pixel.  None (default)
        means auto-detect from the XTC tree on the first batch.
    seg_stride_bytes : int or None
        Bytes between consecutive segment starts inside a multi-segment
        bigdata dgram.  None (default) means auto-detect.
    """

    def __init__(self, det_shape, peds_gpu, gmask_gpu,
                 raw_data_offset=None,
                 seg_stride_bytes=None,
                 stream_seg_map=None,
                 cmpars=None,
                 n_slots=2,
                 budget=None):
        self.det_shape         = tuple(det_shape)
        self.peds_gpu          = peds_gpu
        self.gmask_gpu         = gmask_gpu
        self._raw_data_offset  = None if raw_data_offset is None else int(raw_data_offset)
        self._seg_stride_bytes = None if seg_stride_bytes is None else int(seg_stride_bytes)
        self._n_segs_calib     = int(det_shape[0])
        self._nrows            = int(det_shape[1])
        self._ncols            = int(det_shape[2])
        self._n_pix_seg        = self._nrows * self._ncols
        # {stream_id: [seg_ids]}
        self._stream_seg_map   = stream_seg_map  # type: dict | None
        # Number of calibration gain modes from calibconst.
        # Derived from peds_gpu.size = n_modes * n_segs_calib * n_pix_seg.
        # Used by _extract_and_calibrate to avoid hardcoding 3 (Jungfrau).
        n_pix_total            = self._n_segs_calib * self._n_pix_seg
        self._n_modes_calib    = (int(peds_gpu.size) // n_pix_total
                                  if peds_gpu is not None and n_pix_total > 0
                                  else 3)
        # CPU-side cache for beginstep() change detection.
        self._peds_cpu_cache   = None
        self._gmask_cpu_cache  = None
        # Geometry scatter map for image assembly (set by setup_geometry()).
        self._scatter_ix   = None   # cp.ndarray int64, flat
        self._scatter_iy   = None   # cp.ndarray int64, flat
        self._image_shape  = None   # (nrows_img, ncols_img)
        # Per-stream calibconst cache: pre-computed peds/gmask slices for each
        # stream_id so _extract_and_calibrate() avoids cp.concatenate per event.
        # Computed once on first process_batch() call (after stream_seg_map is
        # known) and reused for all subsequent events — eliminates the 20+ GB
        # of pool allocations that caused OOM at large batch sizes.
        self._stream_peds  = {}     # {stream_key: cp.ndarray float32 flat}
        self._stream_gmask = {}     # {stream_key: cp.ndarray float32 flat}
        # True when peds_gpu/gmask_gpu are shared views owned by another rank
        # (set by share_calib_between_gpu_peers() for follower BD ranks).
        # beginstep() skips the H→D write on followers to avoid a race with
        # the leader writing to the same shared GPU memory.
        self._is_calib_follower = False
        # Per-(slot,stream) pre-allocated raw uint16 buffers.
        # Eliminates cp.stack() allocation inside _extract_and_calibrate()
        # (5 MB × 500 calls/batch × 20 batches = 50 GB pool growth at bs=50).
        self._raw_slot_bufs = {}    # {(slot_id, stream_key): cp.ndarray uint16}
        # Per-slot pre-allocated calib_gpu buffers (Option E).
        # One buffer per EventPool slot, grown lazily to fit the first batch.
        # Reused across batches to prevent CuPy pool fragmentation that causes
        # OOM with large batch sizes.  Each slot's buffer is written by the GPU
        # calibration kernel and read by the user via GpuEventContext.on_gpu.
        #
        # Safety guarantee: the EventPool recycles slot N only after N+n_slots
        # batches, by which point the user has consumed all GpuEventContext
        # objects from that slot.  on_gpu arrays are views into this buffer —
        # users who need to retain results beyond one event-loop cycle should
        # call .on_cpu() to get an independent NumPy copy, then call
        # free_calib_bufs() to reclaim GPU memory.
        self._n_slots         = int(n_slots)
        self._budget          = budget  # _GpuBudget | None
        self._calib_slot_bufs = [None] * self._n_slots   # cp.ndarray per slot
        # Common-mode correction — not yet implemented on GPU.
        if cmpars is not None:
            raise NotImplementedError(
                "Common-mode correction (cmpars) is not yet implemented for "
                "the GPU calibration path.  Pass cmpars=None (default) or "
                "omit the argument.  Implement Phase F3 (common-mode CUDA "
                "kernel) before using cmpars with GPUDetector."
            )

    # Expose detected values as read-only properties for inspection / testing.
    @property
    def raw_data_offset(self):
        return self._raw_data_offset

    @property
    def seg_stride_bytes(self):
        return self._seg_stride_bytes

    # ------------------------------------------------------------------
    # Geometry — image assembly
    # ------------------------------------------------------------------

    def setup_geometry(self, det):
        """Build the GPU scatter map from psana pixel coordinate indices.

        Calls ``det.raw._pixel_coord_indexes(all_segs=True)`` to get the
        (n_all_segs, nrows, ncols) integer arrays that map each pixel in
        the stacked segment array to its position in the assembled 2-D
        detector image.  Selects the rows corresponding to the GPU-routed
        segment IDs (from stream_seg_map) and transfers to GPU.

        After this call ``assemble_image(calib_gpu)`` is available.

        Parameters
        ----------
        det : psana Detector object (returned by run.Detector(det_name))
        """
        import cupy as cp

        try:
            # Full-detector scatter indices: (n_all_segs, nrows, ncols) uint64.
            ix_all, iy_all = det.raw._pixel_coord_indexes(all_segs=True)
        except Exception as exc:
            import warnings
            warnings.warn(
                f'GPUDetector.setup_geometry: could not load pixel coordinate '
                f'indices ({exc}). ctx.get("*.image") will return None.'
            )
            return

        # Build the ordered list of segment IDs in the same order that
        # process_batch() concatenates segments (stream_id ascending, then
        # L1Accept child-XTC order within each stream).
        if self._stream_seg_map:
            all_seg_ids = []
            for sid in sorted(self._stream_seg_map):
                all_seg_ids.extend(self._stream_seg_map[sid])
        else:
            # No seg map: assume rows 0..n_segs_calib-1.
            all_seg_ids = list(range(self._n_segs_calib))

        # Select and flatten scatter indices for the GPU-routed segments.
        try:
            ix = ix_all[all_seg_ids].astype(np.int64)  # (n_segs, nrows, ncols)
            iy = iy_all[all_seg_ids].astype(np.int64)
        except IndexError as exc:
            import warnings
            warnings.warn(
                f'GPUDetector.setup_geometry: segment index out of range '
                f'({exc}). ctx.get("*.image") will return None.'
            )
            return

        nrows_img = int(ix.max()) + 1
        ncols_img = int(iy.max()) + 1

        try:
            self._scatter_ix  = cp.asarray(np.ascontiguousarray(ix.ravel()))
            self._scatter_iy  = cp.asarray(np.ascontiguousarray(iy.ravel()))
            self._image_shape = (nrows_img, ncols_img)
        except Exception as exc:
            import warnings
            warnings.warn(
                f'GPUDetector.setup_geometry: could not transfer scatter '
                f'indices to GPU ({exc}).  ctx.get("*.image") will return '
                f'None.  This typically occurs when the detector has many '
                f'segments (e.g. full Jungfrau 16M) and GPU memory is limited.'
            )

    def setup_geometry_from_arrays(self, ix_all, iy_all):
        """Build GPU scatter map from pre-computed coordinate arrays.

        Equivalent to setup_geometry(det) but takes the pixel coordinate
        index arrays directly rather than calling det.raw._pixel_coord_indexes().

        Used in MPI BD mode where _pixel_coord_indexes() cannot be called
        lazily during the event loop (it triggers a shmem collective that
        requires all MPI ranks to participate, but smd0/EB are already in
        their own event loops by that point).

        Instead, _setup_gpu_geometry() in RunParallel calls
        _pixel_coord_indexes() during __init__ while all ranks are still
        synchronising, stores the numpy arrays, and passes them here.

        Parameters
        ----------
        ix_all : np.ndarray, shape (n_all_segs, nrows, ncols), dtype int64
        iy_all : np.ndarray, shape (n_all_segs, nrows, ncols), dtype int64
            Full-detector row/col scatter indices for every segment,
            as returned by det.raw._pixel_coord_indexes(all_segs=True).
        """
        import cupy as cp

        if self._stream_seg_map:
            all_seg_ids = []
            for sid in sorted(self._stream_seg_map):
                all_seg_ids.extend(self._stream_seg_map[sid])
        else:
            all_seg_ids = list(range(self._n_segs_calib))

        try:
            ix = ix_all[all_seg_ids].astype(np.int64)
            iy = iy_all[all_seg_ids].astype(np.int64)
        except IndexError as exc:
            import warnings
            warnings.warn(
                f'GPUDetector.setup_geometry_from_arrays: segment index out '
                f'of range ({exc}). ctx.get("*.image") will return None.'
            )
            return

        nrows_img = int(ix.max()) + 1
        ncols_img = int(iy.max()) + 1

        try:
            self._scatter_ix  = cp.asarray(np.ascontiguousarray(ix.ravel()))
            self._scatter_iy  = cp.asarray(np.ascontiguousarray(iy.ravel()))
            self._image_shape = (nrows_img, ncols_img)
        except Exception as exc:
            import warnings
            warnings.warn(
                f'GPUDetector.setup_geometry_from_arrays: GPU transfer failed '
                f'({exc}). ctx.get("*.image") will return None.'
            )

    def assemble_image(self, calib_gpu, stream=None):
        """Scatter calibrated segments onto a 2-D detector image on GPU.

        Requires setup_geometry() to have been called first.

        Parameters
        ----------
        calib_gpu : cp.ndarray float32, shape (n_segs, nrows, ncols)
        stream    : cp.cuda.Stream or None

        Returns
        -------
        cp.ndarray float32, shape image_shape, or None if geometry not loaded.
        """
        if self._scatter_ix is None or self._image_shape is None:
            return None

        cp    = _cupy()
        nrows, ncols = self._image_shape
        ctx   = stream if stream is not None else cp.cuda.Stream.null
        try:
            with ctx:
                image_gpu = cp.zeros((nrows, ncols), dtype=cp.float32)
                image_gpu[self._scatter_ix, self._scatter_iy] = calib_gpu.ravel()
            return image_gpu
        except Exception:
            # OOM or other GPU error — image assembly unavailable this event.
            return None

    # ------------------------------------------------------------------
    # Layout auto-detection
    # ------------------------------------------------------------------

    def _ensure_layout(self, sample_bytes):
        """Auto-detect raw_data_offset and seg_stride_bytes if not yet set."""
        if self._raw_data_offset is None or self._seg_stride_bytes is None:
            self._seg_stride_bytes, self._raw_data_offset = \
                _detect_dgram_layout(bytes(sample_bytes))

    # ------------------------------------------------------------------
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

        Parameters
        ----------
        peds_flat  : np.ndarray float32, flat, length 3 * prod(det_shape)
            New pedestals from _compute_calib_constants_cpu().
        gmask_flat : np.ndarray float32, flat, same length
            New gain*mask from _compute_calib_constants_cpu().
        """
        # Compare against cached CPU arrays to skip unnecessary H->D transfers.
        if (self._peds_cpu_cache is not None
                and np.array_equal(peds_flat, self._peds_cpu_cache)
                and np.array_equal(gmask_flat, self._gmask_cpu_cache)):
            return   # no change — skip H->D

        if self._is_calib_follower:
            # peds_gpu/gmask_gpu are shared views into the leader's GPU
            # memory.  The leader's beginstep() will write the new values;
            # doing so here too would race-write to shared memory.
            # We only need to clear derived caches so slices get recomputed
            # from the (already-updated-by-leader) shared arrays.
            self._stream_peds.clear()
            self._stream_gmask.clear()
            self._peds_cpu_cache  = peds_flat.copy()
            self._gmask_cpu_cache = gmask_flat.copy()
            return

        # In-place update: same GPU buffer addresses (CUDA-graph-safe).
        self.peds_gpu.set(np.ascontiguousarray(peds_flat))
        self.gmask_gpu.set(np.ascontiguousarray(gmask_flat))

        # Invalidate per-stream peds/gmask cache.
        self._stream_peds.clear()
        self._stream_gmask.clear()

        # Cache the new CPU arrays for next comparison.
        self._peds_cpu_cache  = peds_flat.copy()
        self._gmask_cpu_cache = gmask_flat.copy()

    # ------------------------------------------------------------------
    # Production API
    # ------------------------------------------------------------------

    def _build_stream_calib_cache(self, stream_id, seg_ids):
        """Pre-compute and cache peds/gmask slices for a stream.

        Called once per stream_id on the first event that needs it.
        Subsequent events for the same stream reuse the cached arrays,
        eliminating the per-event cp.concatenate() calls that accumulated
        20+ GB of pool entries and caused OOM at large batch sizes.
        """
        cp      = _cupy()
        n_total = self._n_segs_calib * self._n_pix_seg
        n_modes = self._n_modes_calib
        n_segs  = len(seg_ids)
        peds_parts = []
        gmask_parts = []
        for m in range(n_modes):
            for s in seg_ids:
                lo = m * n_total + s * self._n_pix_seg
                hi = lo + self._n_pix_seg
                peds_parts.append(self.peds_gpu[lo:hi])
                gmask_parts.append(self.gmask_gpu[lo:hi])
        # One allocation per stream per run — not per event.
        self._stream_peds[stream_id]  = cp.concatenate(peds_parts)
        self._stream_gmask[stream_id] = cp.concatenate(gmask_parts)

    def free_calib_bufs(self):
        """Release all pre-allocated per-slot calib_gpu buffers.

        Call this when:
          - You accumulate GpuEventContext objects across event-loop cycles
            (e.g. ``results = list(run.events())``), in which case the
            on_gpu arrays would alias into buffers that get overwritten.
          - You want to reclaim GPU VRAM at the end of a run.

        After calling this, process_batch() falls back to dynamic
        allocation (one cp.empty() per batch), restoring normal behaviour
        at the cost of CuPy pool growth over long runs.

        Safe to call at any time — in-flight kernels have already read
        from their slot's buffer before it is freed here.
        """
        self._calib_slot_bufs = [None] * self._n_slots

    def memory_bytes(self) -> dict:
        """Return current VRAM usage broken down by category.

        All values are bytes on the GPU device.  Used by
        GpuEvents.log_memory() for Phase-0 accounting.

        Categories
        ----------
        constants   peds_gpu + gmask_gpu (calibration constants)
        geometry    scatter_ix + scatter_iy (pixel coordinate maps)
        calib_slots sum of allocated per-slot calibrated-output buffers
        raw_slots   sum of allocated per-slot raw-gather buffers
        total       sum of the above
        """
        def _nb(arr):
            return int(arr.nbytes) if arr is not None else 0

        constants   = _nb(self.peds_gpu) + _nb(self.gmask_gpu)
        geometry    = _nb(self._scatter_ix) + _nb(self._scatter_iy)
        calib_slots = sum(_nb(b) for b in (self._calib_slot_bufs or []))
        raw_slots   = sum(_nb(b) for b in self._raw_slot_bufs.values())
        total       = constants + geometry + calib_slots + raw_slots
        return {
            'constants':   constants,
            'geometry':    geometry,
            'calib_slots': calib_slots,
            'raw_slots':   raw_slots,
            'total':       total,
        }

    def process_batch(self, gpu_view, gpu_read,
                      stream=None, slot_id=None,
                      compute_raw=False,
                      compute_image=False) -> Iterator[EventContext]:
        """Yield one EventContext per L1Accept event in the batch.

        Reads desc_table (CPU NumPy) for device_offset and read_size per
        event.  Derives the number of segments in the stream's bigdata dgram
        from read_size and gathers them into a contiguous raw buffer before
        calling fused_calib_gpu().

        Only suitable for uncompressed bigdata.

        Parameters
        ----------
        gpu_view : GpuBatchView
        gpu_read : KvikioBatchRead with ``data_gpu`` populated
        stream   : cp.cuda.Stream or None
            CUDA stream on which to run calibration kernels.  When None the
            CuPy default stream is used.  EventPool supplies a non-blocking
            stream to overlap batches and avoid default-stream serialisation.
            The yielded EventContext stores the stream so the caller can
            synchronise before reading results.
        """
        cp         = _cupy()
        data_u16   = gpu_read.data_gpu.view(cp.uint16)
        desc_table = gpu_read.desc_table   # NumPy CPU array — no D2H needed

        # Auto-detect layout on the first batch only.  The guard is outside the
        # .get() call because Python evaluates arguments before calling the
        # function — without the guard, data_gpu[:512].get() (a 512-byte D→H
        # transfer) would fire on every process_batch() call even after layout
        # is already known.
        if self._raw_data_offset is None or self._seg_stride_bytes is None:
            self._ensure_layout(gpu_read.data_gpu[:512].get())

        # ── Phase 1: pre-scan all events ─────────────────────────────────────
        # Collect descriptor rows and segment counts for every non-empty event
        # so we can size the slot buffer to hold the ENTIRE batch in one shot.
        # This is required to give each event a unique, non-overlapping slice:
        # with batch_size > 1 all events share the same det shape, so the old
        # per-event resize check would reuse (and overwrite) the same buffer
        # for every event in the batch — all timestamps would alias the last
        # event's calibration result.
        events_info = []   # list of (GpuBatchEvent, desc_rows, seg_counts, total_segs)
        for event in gpu_view.iter_events():
            if event.n_desc == 0:
                continue
            desc_rows = [desc_table[event.first_desc + i]
                         for i in range(int(event.n_desc))]
            seg_counts = [
                max(1, (int(row[DESC_READ_SIZE]) - 24) // self._seg_stride_bytes)
                for row in desc_rows
            ]
            events_info.append((event, desc_rows, seg_counts, sum(seg_counts)))

        if not events_info:
            return

        # ── Phase 2: allocate / grow the slot buffer for the whole batch ──────
        # The slot buffer covers ALL events: shape = (total_segs_batch, nrows, ncols).
        # Each event's calib output lands in a unique non-overlapping slice, so
        # no event's data can be overwritten by a later event in the same batch.
        total_segs_batch = sum(e[3] for e in events_info)
        batch_shape      = (total_segs_batch, self._nrows, self._ncols)

        slot    = (int(slot_id) % self._n_slots
                   if slot_id is not None else None)
        slot_buf = None
        if self._calib_slot_bufs is not None and slot is not None:
            buf = self._calib_slot_bufs[slot]
            if buf is None or buf.shape != batch_shape:
                needed   = int(batch_shape[0] * batch_shape[1] * batch_shape[2]) * 4
                old_size = int(buf.nbytes) if buf is not None else 0
                if self._budget is not None:
                    # Update budget regardless of grow vs shrink so _committed
                    # never drifts.  reserve() raises GpuMemoryPressureError if
                    # the new allocation would exceed the per-BD limit.
                    if old_size:
                        self._budget.release(old_size)
                    self._budget.reserve(needed)
                else:
                    # No budget object — flush pool before a large allocation.
                    cp.get_default_memory_pool().free_all_blocks()
                if __import__('os').environ.get('PSANA_GPU_MEM_DEBUG'):
                    free_b, _ = cp.cuda.Device().mem_info
                    print('[GPU-MEM] calib slot grow: need=%.1fGB free=%.1fGB'
                          % (needed/1e9, free_b/1e9), flush=True)
                self._calib_slot_bufs[slot] = cp.empty(batch_shape, dtype=cp.float32)
            slot_buf = self._calib_slot_bufs[slot]

        # ── Phase 3: per-event calibration into non-overlapping slices ────────
        sctx         = stream if stream is not None else cp.cuda.Stream.null
        batch_offset = 0   # segment offset into slot_buf for the current event

        for event, desc_rows, seg_counts, total_segs in events_info:
            raw_segs = [] if compute_raw else None

            # Each event gets a UNIQUE slice of the batch slot buffer so that
            # calibrating event i+1 cannot overwrite event i's result.
            out_buf = (slot_buf[batch_offset : batch_offset + total_segs]
                       if slot_buf is not None else None)

            with sctx:
                seg_offset = 0
                for i, desc_row in enumerate(desc_rows):
                    device_offset = int(desc_row[DESC_DEVICE_OFFSET])
                    read_size     = int(desc_row[DESC_READ_SIZE])
                    stream_id     = int(desc_row[DESC_STREAM_ID])
                    seg_ids       = (self._stream_seg_map.get(stream_id)
                                     if self._stream_seg_map else None)
                    n_segs        = seg_counts[i]
                    # Pass the pre-allocated slice so the kernel writes
                    # directly into it — no allocation inside calibrate().
                    out_slice = (out_buf[seg_offset:seg_offset + n_segs]
                                 if out_buf is not None else None)
                    calib, raw = self._extract_and_calibrate(
                        data_u16, device_offset, read_size, seg_ids,
                        out=out_slice, slot_id=slot, stream=stream,
                    )
                    if out_buf is None:
                        if i == 0:
                            calib_gpu = calib
                        else:
                            calib_gpu = cp.concatenate(
                                [calib_gpu, calib], axis=0
                            )
                    if raw_segs is not None:
                        raw_segs.append(raw)
                    seg_offset += n_segs

                if out_buf is not None:
                    calib_gpu = out_buf   # unique view into slot_buf, no copy

                if raw_segs is None:
                    raw_gpu = None
                elif len(raw_segs) == 1:
                    raw_gpu = raw_segs[0]
                else:
                    raw_gpu = cp.concatenate(raw_segs, axis=0)
                image_gpu = (self.assemble_image(calib_gpu, stream=stream)
                             if compute_image else None)

            yield EventContext(timestamp=event.timestamp,
                               calib_gpu=calib_gpu,
                               raw_gpu=raw_gpu,
                               image_gpu=image_gpu,
                               stream=stream)

            batch_offset += total_segs   # advance to next event's region

    # ------------------------------------------------------------------
    # Test / validation API
    # ------------------------------------------------------------------

    def calibrate(self, data_gpu, device_offset=0):
        """Calibrate a single dgram already resident in data_gpu.

        Infers the number of segments from the dgram size so this entry point
        works for any uncompressed area detector without detector-specific
        configuration.

        Parameters
        ----------
        data_gpu      : cp.ndarray uint8, the raw bigdata dgram bytes
        device_offset : int, byte offset of the dgram start within data_gpu
                        (default 0 when the buffer holds exactly one dgram)

        Returns
        -------
        cp.ndarray float32, shape (n_segs, nrows, ncols)
        """
        cp        = _cupy()
        data_u16  = data_gpu.view(cp.uint16)
        read_size = int(data_gpu.nbytes) - device_offset
        # Auto-detect layout from first 512 bytes of this dgram.
        if self._raw_data_offset is None or self._seg_stride_bytes is None:
            self._ensure_layout(data_gpu[device_offset:device_offset + 512].get())
        calib, _ = self._extract_and_calibrate(data_u16, device_offset, read_size)
        return calib

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_and_calibrate(self, data_u16, device_offset, read_size,
                               seg_ids=None, out=None, slot_id=None,
                               stream=None):
        """Gather raw uint16 pixels and run fused_calib_gpu.

        Parameters
        ----------
        data_u16      : cp.ndarray uint16 view of the full data_gpu buffer
        device_offset : byte offset of this dgram within the original buffer
        read_size     : size in bytes of this dgram
        seg_ids       : list of int or None
            Calibconst row indices for this stream's physical segments, in
            the same order as the child XTCs in the bigdata dgram.  When
            provided, the calibration
            constants are pulled from the correct rows rather than naively
            using rows 0..n_segs-1.  Build this list with
            build_stream_seg_map().  None falls back to the first-N
            approximation (incorrect for mixed-segment detectors).
        stream        : cp.cuda.Stream or None
            Stream on which the raw gather and calibration must execute in
            order.  Passing None uses CuPy's default stream.

        _ensure_layout() must have been called before this method.
        """
        cp = _cupy()

        n_segs = max(1, (read_size - 24) // self._seg_stride_bytes)

        stream_key = tuple(seg_ids) if seg_ids is not None else None
        raw_key    = (slot_id, stream_key, n_segs)
        if n_segs == 1:
            pix_start = (device_offset + self._raw_data_offset) // 2
            raw_u16   = data_u16[pix_start:pix_start + self._n_pix_seg].reshape(
                1, self._nrows, self._ncols
            )
        else:
            # Gather all N segments in one kernel launch instead of N separate
            # device-to-device copies.  Segments sit at evenly-spaced offsets
            # in data_gpu (seg_stride_bytes apart).  as_strided exposes them
            # as a 3-D view (n_segs, nrows, ncols) with no copy; assigning
            # that view into the contiguous pre-allocated buf triggers one
            # gather kernel rather than N element-wise copy kernels.
            cp = _cupy()
            buf = self._raw_slot_bufs.get(raw_key)
            if buf is None or buf.shape != (n_segs, self._nrows, self._ncols):
                buf = cp.empty((n_segs, self._nrows, self._ncols), dtype=cp.uint16)
                self._raw_slot_bufs[raw_key] = buf
            pix_start  = (device_offset + self._raw_data_offset) // 2
            stride_u16 = self._seg_stride_bytes // 2
            span_u16   = (n_segs - 1) * stride_u16 + self._n_pix_seg
            src_view = cp.lib.stride_tricks.as_strided(
                data_u16[pix_start : pix_start + span_u16],
                shape=(n_segs, self._nrows, self._ncols),
                strides=(self._seg_stride_bytes, self._ncols * 2, 2),
            )
            buf[:] = src_view   # one gather kernel, not N copy kernels
            raw_u16 = buf

        n_total = self._n_segs_calib * self._n_pix_seg

        n_modes = self._n_modes_calib
        if seg_ids is not None:
            # Use pre-computed per-stream calibconst slices.
            # _build_stream_calib_cache() allocates once per unique stream_id;
            # subsequent events reuse the cached arrays — no per-event
            # cp.concatenate() and no pool fragmentation.
            stream_key = tuple(seg_ids)
            if stream_key not in self._stream_peds:
                self._build_stream_calib_cache(stream_key, seg_ids)
            peds  = self._stream_peds[stream_key]
            gmask = self._stream_gmask[stream_key]
        elif n_segs != self._n_segs_calib:
            # Fallback: first n_segs rows (approximation).
            n_target = n_segs * self._n_pix_seg
            peds  = cp.concatenate([
                self.peds_gpu[m * n_total : m * n_total + n_target]
                for m in range(n_modes)
            ])
            gmask = cp.concatenate([
                self.gmask_gpu[m * n_total : m * n_total + n_target]
                for m in range(n_modes)
            ])
        else:
            peds  = self.peds_gpu
            gmask = self.gmask_gpu

        calib = fused_calib_gpu(raw_u16, peds, gmask, out=out)

        return calib, raw_u16   # (calib_gpu, raw_gpu) tuple


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fused_calib_gpu(raw_gpu, peds_gpu, gmask_gpu, threads=256, out=None):
    """Run Jungfrau calibration on GPU.

    Parameters
    ----------
    raw_gpu   : cp.ndarray, dtype uint16, any shape (total npixels elements)
    peds_gpu  : cp.ndarray, dtype float32, flat, length 3 * npixels
    gmask_gpu : cp.ndarray, dtype float32, flat, length 3 * npixels
    threads   : CUDA threads per block

    Returns
    -------
    calib_gpu : cp.ndarray, dtype float32, same shape as raw_gpu
    """
    cp = _cupy()

    npixels = int(raw_gpu.size)

    if raw_gpu.dtype != cp.uint16:
        raise TypeError(f"raw_gpu must be uint16, got {raw_gpu.dtype}")
    if peds_gpu.dtype != cp.float32:
        raise TypeError(f"peds_gpu must be float32, got {peds_gpu.dtype}")
    if gmask_gpu.dtype != cp.float32:
        raise TypeError(f"gmask_gpu must be float32, got {gmask_gpu.dtype}")
    if peds_gpu.size != 3 * npixels:
        raise ValueError(
            f"peds_gpu length {peds_gpu.size} != 3 * npixels ({3 * npixels})"
        )
    if gmask_gpu.size != 3 * npixels:
        raise ValueError(
            f"gmask_gpu length {gmask_gpu.size} != 3 * npixels ({3 * npixels})"
        )

    # Use the caller's pre-allocated output buffer when provided (Option E).
    # Avoids a cp.empty() allocation per batch — the caller passes a view into
    # the EventPool's slot buffer so no new VRAM is consumed.
    if out is not None and out.size >= npixels and out.dtype == cp.float32:
        calib_gpu = out.ravel()[:npixels]
    else:
        calib_gpu = cp.empty(npixels, dtype=cp.float32)

    blocks = (npixels + threads - 1) // threads
    _jungfrau_calib_kernel()(
        (blocks,),
        (threads,),
        (
            raw_gpu.ravel(),
            peds_gpu.ravel(),
            gmask_gpu.ravel(),
            calib_gpu,
            np.uint64(npixels),
        ),
    )

    return calib_gpu.reshape(raw_gpu.shape)


def _compute_calib_constants_cpu(det):
    """Compute calibration constants on CPU.

    Shared by prep_calib_constants() (first run) and GPUDetector.beginstep()
    (step refresh).

    Returns
    -------
    peds_flat  : np.ndarray float32, contiguous, length 3 * npixels
    gmask_flat : np.ndarray float32, contiguous, same length
    """
    cc   = det.calibconst
    peds = cc["pedestals"][0].astype(np.float32)   # (3, n_all_segs, nrows, ncols)
    gain = cc["pixel_gain"][0].astype(np.float32)
    # Apply pixel_offset calibration constant (same shape as pedestals).
    # On CPU: DetCache.poff = pedestals + pixel_offset; we replicate that here.
    try:
        offset = cc.get("pixel_offset", [None])[0]
        if offset is not None:
            peds = peds + offset.astype(np.float32)
    except Exception:
        pass   # pixel_offset absent or wrong shape — silently skip

    # Per-pixel mask (1 = good, 0 = bad).  Priority:
    #   1. det.raw._mask(all_segs=True) — uses all calibration quality flags.
    #   2. pixel_status calibconst mode-0 (0=good) — fallback when _mask()
    #      shape mismatches (fewer active segments than calibconst covers).
    #   3. All-ones — last resort.
    expected_shape = peds.shape[1:]   # (n_all_segs, nrows, ncols)
    mask = None
    try:
        m = det.raw._mask(all_segs=True)
        if m is not None and m.shape == expected_shape:
            mask = m
    except Exception:
        pass
    if mask is None:
        try:
            status = cc['pixel_status'][0]        # (3, n_all_segs, nrows, ncols)
            mask = (status[0] == 0).astype(np.float32)  # mode-0, 1=good
        except Exception:
            pass                                   # all-ones below

    gfac = np.where(gain != 0, np.float32(1.0) / gain, np.float32(0.0))
    if mask is not None:
        gmask = (gfac * mask[np.newaxis]).astype(np.float32)
    else:
        gmask = gfac.astype(np.float32)

    return (np.ascontiguousarray(peds.ravel()),
            np.ascontiguousarray(gmask.ravel()))


def prep_calib_constants(det):
    """Transfer calibration constants for det to GPU.

    Computes gmask = (1 / pixel_gain) * mask on CPU, then copies both
    pedestals and gmask to the GPU.  Call once per run (BeginRun).
    For mid-run refresh after a BeginStep use GPUDetector.beginstep().

    Parameters
    ----------
    det : psana Detector object with calibconst loaded

    Returns
    -------
    peds_gpu  : cp.ndarray float32, flat, length 3 * npixels
    gmask_gpu : cp.ndarray float32, flat, same length
    """
    import cupy as cp
    peds_flat, gmask_flat = _compute_calib_constants_cpu(det)
    return cp.asarray(peds_flat), cp.asarray(gmask_flat)


def _segment_ids_in_l1_order(dgram, det_name):
    """Return detector segment IDs in XTC traversal order for one event.

    ``dgram.cc`` walks ShapesData records in payload order and inserts a
    detector dictionary entry the first time it encounters each segment.
    Python dictionaries preserve that insertion order, so the keys give the
    child-XTC order needed by the GPU's fixed-stride raw-pixel gather.

    Segment identity itself still comes from Configure: dgram.cc joins each
    event ShapesData NamesId to its Configure Names record before inserting
    the corresponding ``Names.segment()`` key.
    """
    detector_data = getattr(dgram, det_name, None)
    if detector_data is None:
        return []
    return [int(segment_id) for segment_id in detector_data.keys()]


def build_stream_seg_map(stream_bd_files, det_name='jungfrau'):
    """Build the segment-ID map for GPU-routed bigdata streams.

    Opens each detector-bearing bigdata stream and reads its first L1Accept.
    The event ShapesData records are joined to Configure by psana's normal
    dgram parser, yielding physical segment IDs in actual child-XTC order.
    The resulting map tells GPUDetector which calibconst row belongs to each
    strided raw-panel position.

    Parameters
    ----------
    stream_bd_files : dict  {stream_id: str}
        Maps each GPU-routed stream ID to its bigdata file path.
        Build this from ``smd_to_xtc_file()`` for each relevant stream.
    det_name : str
        Detector name (default 'jungfrau').

    Returns
    -------
    dict {stream_id: List[int]}
        Segment IDs for each stream in L1Accept child-XTC order.  These IDs
        are direct calibconst row indices; they are intentionally not sorted.

    Example
    -------
    >>> seg_map = build_stream_seg_map({6: '/path/to/s006.xtc2',
    ...                                 8: '/path/to/s008.xtc2'})
    >>> # seg_map = {6: [3,7,0,1,2], 8: [17,13,9,5,29,25,21]}
    """
    from psana.dgrammanager import DgramManager
    from psana.psexp.transitionid import TransitionId

    seg_map = {}
    for stream_id, bd_file in stream_bd_files.items():
        dm = None
        try:
            dm = DgramManager([str(bd_file)])

            # Do not scan non-detector streams to EOF.  Configure is enough
            # to determine whether this stream can contain the detector.
            carries_detector = any(
                hasattr(getattr(config, 'software', None), det_name)
                for config in dm.configs
            )
            if not carries_detector:
                continue

            for dgrams in dm:
                dg = dgrams[0] if dgrams else None
                if dg is None or not TransitionId.isEvent(dg.service()):
                    continue
                segment_ids = _segment_ids_in_l1_order(dg, det_name)
                if segment_ids:
                    seg_map[int(stream_id)] = segment_ids
                    break

            if int(stream_id) not in seg_map:
                import warnings
                warnings.warn(
                    f"build_stream_seg_map: stream {stream_id} Configure "
                    f"contains {det_name!r}, but no detector L1Accept was found"
                )
        except Exception as exc:
            import warnings
            warnings.warn(
                f"build_stream_seg_map: could not read stream {stream_id} "
                f"({bd_file}): {exc}"
            )
        finally:
            if dm is not None:
                dm.close()
    return seg_map


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _cupy():
    import cupy as cp
    return cp


@lru_cache(maxsize=1)
def _jungfrau_calib_kernel():
    cp = _cupy()
    return cp.RawKernel(
        _kernel_source(),
        _KERNEL_NAME,
        options=("--std=c++17",),
    )


@lru_cache(maxsize=1)
def _kernel_source():
    header_path = Path(__file__).with_name("cuda") / "fused_calib.cuh"
    header = header_path.read_text()

    return header + f"""

extern "C" __global__
void {_KERNEL_NAME}(
    const unsigned short* raw,
    const float*          peds,
    const float*          gmask,
    float*                calib,
    unsigned long long    npixels)
{{
    const unsigned long long i =
        (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= npixels)
        return;
    calib[i] = psana_gpu::jungfrau_calib_pixel(raw[i], peds, gmask, i, npixels);
}}
"""
