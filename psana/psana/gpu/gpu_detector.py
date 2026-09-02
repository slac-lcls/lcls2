"""GPU detector processing, raw assembly, and result-buffer ownership."""

from dataclasses import dataclass
from functools import lru_cache
from typing import Iterator

import numpy as np

from psana.gpu.dgram_layout import detect_dgram_layout
from psana.gpu.gpu_calib import (
    assemble_image as assemble_calib_image,
    fused_calib_gpu,
    prepare_geometry,
    prepare_geometry_from_arrays,
)
from psana.gpu.gpu_kvikio_read import DESC_DEVICE_OFFSET, DESC_READ_SIZE, DESC_STREAM_ID

_GATHER_U16_KERNEL_NAME = "gather_strided_u16_rows_kernel"
_GATHER_F32_KERNEL_NAME = "gather_strided_f32_rows_kernel"


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
    canonical_segment_ids : sequence[int] or None
        Segment order expected by the normal psana detector API. GPU input is
        gathered in L1 child-XTC order and reordered before it is exposed.
    """

    def __init__(self, det_shape, peds_gpu, gmask_gpu,
                 raw_data_offset=None,
                 seg_stride_bytes=None,
                 stream_seg_map=None,
                 canonical_segment_ids=None,
                 cmpars=None,
                 n_slots=2,
                 budget=None,
                 passthrough=False):
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
        self._canonical_segment_ids = tuple(
            range(self._n_segs_calib)
            if canonical_segment_ids is None else canonical_segment_ids
        )
        if len(self._canonical_segment_ids) != self._n_segs_calib:
            raise ValueError(
                "canonical_segment_ids must contain one entry per detector "
                f"segment: got {len(self._canonical_segment_ids)}, "
                f"expected {self._n_segs_calib}"
            )
        if len(set(self._canonical_segment_ids)) != len(self._canonical_segment_ids):
            raise ValueError("canonical_segment_ids contains duplicates")
        self._canonical_segment_rows = {
            segment_id: row
            for row, segment_id in enumerate(self._canonical_segment_ids)
        }
        self._budget = budget  # _GpuBudget | None

        # Stream dgrams group panels in L1 child order. Cache the mapping from
        # each stream-local input row to the canonical detector row. Raw
        # assembly is the only stage that knows this routing; calibration and
        # downstream user kernels consume an ordinary canonical detector array.
        self._canonical_rows_by_stream = {}
        routed_rows = set()
        for stream_id, segment_ids in (self._stream_seg_map or {}).items():
            unknown = set(segment_ids) - set(self._canonical_segment_rows)
            if unknown:
                raise ValueError(
                    f"stream {stream_id} contains segments absent from "
                    f"canonical_segment_ids: {sorted(unknown)}"
                )
            rows = tuple(self._canonical_segment_rows[s] for s in segment_ids)
            duplicates = routed_rows.intersection(rows)
            if duplicates:
                raise ValueError(
                    "detector segments appear in more than one GPU stream: "
                    f"canonical rows {sorted(duplicates)}"
                )
            routed_rows.update(rows)
            self._canonical_rows_by_stream[int(stream_id)] = rows

        # These maps are tiny (one uint32 per panel), fixed for the run, and
        # explicitly budgeted.  Keeping them on device avoids rebuilding an
        # advanced-index array for every event.
        self._canonical_rows_gpu_by_stream = {}
        self._routing_map_bytes = sum(
            len(rows) * np.dtype(np.uint32).itemsize
            for rows in self._canonical_rows_by_stream.values()
        )
        if self._routing_map_bytes:
            if self._budget is not None:
                self._budget.reserve(self._routing_map_bytes)
            try:
                cp = _cupy()
                self._canonical_rows_gpu_by_stream = {
                    stream_id: cp.asarray(rows, dtype=cp.uint32)
                    for stream_id, rows in self._canonical_rows_by_stream.items()
                }
            except Exception:
                if self._budget is not None:
                    self._budget.release(self._routing_map_bytes)
                raise
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

    # Layout auto-detection
    # ------------------------------------------------------------------

    def _ensure_layout(self, sample_bytes):
        """Auto-detect seg_stride_bytes and raw_data_offset from the first bigdata dgram.

        The pixel dtype (uint16 vs float32) is NOT derived here — it comes from
        the Configure-dgram drp_class_name and is already set as _pixel_bytes in
        __init__ via the passthrough flag.
        """
        if self._raw_data_offset is None or self._seg_stride_bytes is None:
            self._seg_stride_bytes, self._raw_data_offset = \
                detect_dgram_layout(bytes(sample_bytes))

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
        routing     canonical output-row maps
        calib_slots sum of allocated per-slot calibrated-output buffers
        raw_slots   sum of allocated per-slot raw-gather buffers
        total       sum of the above
        """
        def _nb(arr):
            return int(arr.nbytes) if arr is not None else 0

        constants   = _nb(self.peds_gpu) + _nb(self.gmask_gpu)
        geometry    = _nb(self._scatter_ix) + _nb(self._scatter_iy)
        routing     = sum(_nb(m) for m in self._canonical_rows_gpu_by_stream.values())
        calib_slots = sum(_nb(b) for b in (self._calib_slot_bufs or []))
        raw_slots   = sum(_nb(b) for b in self._raw_slot_bufs)
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
        that are excluded here (they are already committed in _GpuBudget).

        Uses stream_seg_map when available to count only the GPU-routed
        segments (not the full calibconst segment count which includes
        CPU-routed segments).

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
        # Count only GPU-routed segments (stream_seg_map keys) if available.
        if self._stream_seg_map:
            n_segs = sum(len(segs) for segs in self._stream_seg_map.values())
        else:
            n_segs = self._n_segs_calib
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

    def process_batch(self, gpu_view, gpu_read,
                      stream=None, slot_id=None) -> Iterator[EventContext]:
        """Assemble canonical detector arrays and yield one result per event.

        The reader controls where each logical dgram lands in ``data_gpu``.
        This method honors every explicit ``DESC_DEVICE_OFFSET`` and therefore
        does not require one physical read per dgram or tightly packed input.
        Stream/segment routing ends at canonical raw assembly; Jungfrau
        calibration consumes the same canonical raw array exposed to users.

        Parameters
        ----------
        gpu_view : GpuBatchView
        gpu_read : KvikioBatchRead with ``data_gpu`` populated
        stream   : cp.cuda.Stream or None
            CUDA stream on which to run calibration kernels.  When None the
            CuPy default stream is used.  EventPool supplies a non-blocking
            stream to overlap batches and avoid default-stream serialisation.
        """
        cp         = _cupy()
        data_gpu   = gpu_read.data_gpu
        desc_table = gpu_read.desc_table   # NumPy CPU array — no D2H needed

        relevant_rows = [
            row for row in desc_table
            if (not self._stream_seg_map
                or int(row[DESC_STREAM_ID]) in self._stream_seg_map)
        ]
        if not relevant_rows:
            return

        # Detect from this detector's first logical dgram, not data_gpu[0]. A
        # future coalesced reader may leave gaps in the physical input buffer.
        if self._raw_data_offset is None or self._seg_stride_bytes is None:
            sample_offset = int(relevant_rows[0][DESC_DEVICE_OFFSET])
            self._ensure_layout(
                data_gpu[sample_offset:sample_offset + 512].get()
            )

        # ── Phase 1: pre-scan all events ─────────────────────────────────────
        # Collect descriptor rows and segment counts for every non-empty event
        # so we can size the slot buffer to hold the ENTIRE batch in one shot.
        # This is required to give each event a unique, non-overlapping slice:
        # with batch_size > 1 all events share the same det shape, so the old
        # per-event resize check would reuse (and overwrite) the same buffer
        # for every event in the batch — all timestamps would alias the last
        # event's calibration result.
        events_info = []   # (GpuBatchEvent, desc_rows, seg_counts, segment_ids)
        for event in gpu_view.iter_events():
            if event.n_desc == 0:
                continue
            desc_rows = [desc_table[event.first_desc + i]
                         for i in range(int(event.n_desc))
                         if (not self._stream_seg_map
                             or int(desc_table[event.first_desc + i][DESC_STREAM_ID])
                             in self._stream_seg_map)]
            if not desc_rows:
                continue
            seg_counts = [
                max(1, (int(row[DESC_READ_SIZE]) - 24) // self._seg_stride_bytes)
                for row in desc_rows
            ]
            segment_ids = []
            for row, n_segs in zip(desc_rows, seg_counts):
                stream_id = int(row[DESC_STREAM_ID])
                stream_segment_ids = (
                    self._stream_seg_map.get(stream_id)
                    if self._stream_seg_map else None
                )
                if stream_segment_ids is None:
                    stream_segment_ids = list(range(n_segs))
                if len(stream_segment_ids) != n_segs:
                    raise RuntimeError(
                        f"GPU stream {stream_id} contains {n_segs} segments, "
                        "but Configure/L1 metadata identifies "
                        f"{len(stream_segment_ids)}: {stream_segment_ids}"
                    )
                segment_ids.extend(stream_segment_ids)
            unknown = set(segment_ids) - set(self._canonical_segment_rows)
            if unknown:
                raise RuntimeError(
                    "GPU batch contains detector segments absent from "
                    f"Configure: {sorted(unknown)}"
                )
            if len(set(segment_ids)) != len(segment_ids):
                raise RuntimeError(
                    "GPU batch contains duplicate detector segments: "
                    f"{segment_ids}"
                )
            events_info.append((event, desc_rows, seg_counts, segment_ids))

        if not events_info:
            return

        # Both result slots cover the whole batch. Per-event views remain valid
        # until EventPool retires this execution slot.
        batch_shape = (
            len(events_info) * self._n_segs_calib,
            self._nrows,
            self._ncols,
        )

        if slot_id is None:
            raise ValueError("GPUDetector.process_batch requires an EventPool slot_id")
        slot = int(slot_id) % self._n_slots
        calib_slot = self._slot_buffer(
            self._calib_slot_bufs, slot, batch_shape, np.float32, "calib"
        )
        raw_slot = None
        if not self._passthrough:
            raw_slot = self._slot_buffer(
                self._raw_slot_bufs, slot, batch_shape, np.uint16, "raw"
            )

        sctx = stream if stream is not None else cp.cuda.Stream.null
        for event_index, event_info in enumerate(events_info):
            event, desc_rows, seg_counts, segment_ids = event_info
            lo = event_index * self._n_segs_calib
            hi = lo + self._n_segs_calib
            calib_out = calib_slot[lo:hi]
            raw_out = None if raw_slot is None else raw_slot[lo:hi]
            complete = len(segment_ids) == self._n_segs_calib

            with sctx:
                target = calib_out if self._passthrough else raw_out
                if not complete:
                    target.fill(0)

                for desc_row, n_segs in zip(desc_rows, seg_counts):
                    stream_id = int(desc_row[DESC_STREAM_ID])
                    seg_ids = (
                        self._stream_seg_map.get(stream_id)
                        if self._stream_seg_map else list(range(n_segs))
                    )
                    output_rows = self._canonical_rows_gpu_by_stream.get(stream_id)
                    if output_rows is None and (
                        n_segs != self._n_segs_calib
                        or tuple(seg_ids) != self._canonical_segment_ids
                    ):
                        raise RuntimeError(
                            f"GPU stream {stream_id} has no canonical output-row map"
                        )

                    device_offset = int(desc_row[DESC_DEVICE_OFFSET])
                    if self._passthrough:
                        src = data_gpu.view(cp.float32)
                        pixel_bytes = 4
                    else:
                        src = data_gpu.view(cp.uint16)
                        pixel_bytes = 2
                    _gather_strided_rows_gpu(
                        src,
                        pix_start=(device_offset + self._raw_data_offset)
                            // pixel_bytes,
                        stride_pixels=self._seg_stride_bytes // pixel_bytes,
                        n_segments=n_segs,
                        pixels_per_segment=self._n_pix_seg,
                        output_rows=output_rows,
                        out=target,
                    )

                if not self._passthrough:
                    fused_calib_gpu(
                        raw_out, self.peds_gpu, self.gmask_gpu, out=calib_out
                    )
                    # Missing raw rows contain zero, which would calibrate to
                    # -pedestal. Preserve the established zero-result behavior.
                    if not complete:
                        present_rows = {
                            self._canonical_segment_rows[s]
                            for s in segment_ids
                        }
                        for row in set(range(self._n_segs_calib)) - present_rows:
                            calib_out[row].fill(0)

            yield EventContext(
                timestamp=event.timestamp,
                calib_gpu=calib_out,
                raw_gpu=raw_out,
            )

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
        read_size = int(data_gpu.nbytes) - device_offset
        # Auto-detect layout from first 512 bytes of this dgram.
        if self._raw_data_offset is None or self._seg_stride_bytes is None:
            self._ensure_layout(data_gpu[device_offset:device_offset + 512].get())
        calib, _ = self._extract_and_calibrate(data_gpu, device_offset, read_size)
        return calib

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_and_calibrate(self, data_gpu, device_offset, read_size,
                               out=None):
        """Gather and calibrate one full-detector validation dgram.

        In normal mode, gathers raw uint16 ADC pixels and calls fused_calib_gpu.
        In passthrough mode (pre-calibrated float32 bigdata from the DRP),
        reads the float32 pixels directly and skips the calibration kernel.

        Parameters
        ----------
        data_gpu      : cp.ndarray uint8 — the full raw bigdata GPU buffer.
                        A uint16 or float32 view is created internally based
                        on self._passthrough.
        device_offset : byte offset of this dgram within data_gpu
        read_size     : size in bytes of this dgram
        _ensure_layout() must have been called before this method.
        """
        cp = _cupy()

        n_segs = max(1, (read_size - 24) // self._seg_stride_bytes)

        # ── Passthrough mode: pre-calibrated float32 — no kernel needed ──────
        if self._passthrough:
            data_f32  = data_gpu.view(cp.float32)
            pix_start = (device_offset + self._raw_data_offset) // 4
            if out is not None:
                _gather_strided_rows_gpu(
                    data_f32,
                    pix_start=pix_start,
                    stride_pixels=self._seg_stride_bytes // 4,
                    n_segments=n_segs,
                    pixels_per_segment=self._n_pix_seg,
                    output_rows=None,
                    out=out,
                )
                return out, None
            if n_segs == 1:
                src = data_f32[pix_start:pix_start + self._n_pix_seg].reshape(
                    1, self._nrows, self._ncols
                )
            else:
                stride_f32 = self._seg_stride_bytes // 4
                span_f32   = (n_segs - 1) * stride_f32 + self._n_pix_seg
                src = cp.lib.stride_tricks.as_strided(
                    data_f32[pix_start:pix_start + span_f32],
                    shape=(n_segs, self._nrows, self._ncols),
                    strides=(self._seg_stride_bytes, self._ncols * 4, 4),
                )
            if out is not None:
                out[:] = src
                return out, None
            # No pre-allocated output slot — copy to prevent aliasing with the
            # reader's slot buffer, which will be overwritten on the next batch.
            return src.copy(), None

        # Build canonical raw first, then calibrate it exactly as any user
        # kernel would consume it. This validation entry point is only
        # unambiguous when one dgram covers the full detector.
        if n_segs != self._n_segs_calib:
            raise ValueError(
                "calibrate() requires a full-detector dgram; production "
                "multi-stream assembly is handled by process_batch()"
            )
        raw_u16 = cp.empty(
            (self._n_segs_calib, self._nrows, self._ncols), dtype=cp.uint16
        )
        _gather_strided_rows_gpu(
            data_gpu.view(cp.uint16),
            pix_start=(device_offset + self._raw_data_offset) // 2,
            stride_pixels=self._seg_stride_bytes // 2,
            n_segments=n_segs,
            pixels_per_segment=self._n_pix_seg,
            output_rows=None,
            out=raw_u16,
        )
        calib = fused_calib_gpu(
            raw_u16, self.peds_gpu, self.gmask_gpu, out=out
        )
        return calib, raw_u16


def _gather_strided_rows_gpu(
    src,
    pix_start,
    stride_pixels,
    n_segments,
    pixels_per_segment,
    output_rows,
    out,
    threads=256,
):
    """Gather one logical dgram's panels into canonical detector rows.

    ``pix_start`` is the logical payload location within the reader-owned GPU
    buffer. It may contain arbitrary leading or inter-dgram gaps, which keeps
    raw assembly independent of future physical-read coalescing.
    """
    cp = _cupy()
    if src.dtype not in (cp.uint16, cp.float32):
        raise TypeError(f"src must be uint16 or float32, got {src.dtype}")
    if out.dtype != src.dtype:
        raise TypeError(f"out dtype {out.dtype} does not match src {src.dtype}")
    if output_rows is not None and output_rows.dtype != cp.uint32:
        raise TypeError(f"output_rows must be uint32, got {output_rows.dtype}")
    if output_rows is not None and int(output_rows.size) != int(n_segments):
        raise ValueError(
            f"output_rows length {output_rows.size} != n_segments {n_segments}"
        )
    source_end = (
        int(pix_start)
        + (int(n_segments) - 1) * int(stride_pixels)
        + int(pixels_per_segment)
    )
    if source_end > int(src.size):
        raise ValueError(
            f"strided source ends at {source_end}, buffer has {src.size} elements"
        )

    if output_rows is None:
        itemsize = int(src.dtype.itemsize)
        src_view = cp.lib.stride_tricks.as_strided(
            src[int(pix_start):source_end],
            shape=(int(n_segments), int(pixels_per_segment)),
            strides=(int(stride_pixels) * itemsize, itemsize),
        )
        out.reshape(out.shape[0], -1)[:n_segments] = src_view
        return out

    blocks_per_segment = (pixels_per_segment + threads - 1) // threads
    kernel = (
        _gather_u16_kernel() if src.dtype == cp.uint16
        else _gather_f32_kernel()
    )
    kernel(
        (blocks_per_segment, n_segments),
        (threads,),
        (
            src,
            output_rows,
            out.ravel(),
            np.uint64(pix_start),
            np.uint64(stride_pixels),
            np.uint64(pixels_per_segment),
        ),
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
def _gather_kernel_source():
    return f"""

extern "C" __global__
void {_GATHER_U16_KERNEL_NAME}(
    const unsigned short* src,
    const unsigned int*   output_rows,
    unsigned short*       out,
    unsigned long long    pix_start,
    unsigned long long    stride_pixels,
    unsigned long long    pixels_per_segment)
{{
    const unsigned long long pixel =
        (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned long long input_segment = blockIdx.y;
    if (pixel >= pixels_per_segment)
        return;

    const unsigned long long input_index =
        pix_start + input_segment * stride_pixels + pixel;
    const unsigned long long output_index =
        (unsigned long long)output_rows[input_segment] * pixels_per_segment + pixel;
    out[output_index] = src[input_index];
}}

extern "C" __global__
void {_GATHER_F32_KERNEL_NAME}(
    const float*          src,
    const unsigned int*   output_rows,
    float*                out,
    unsigned long long    pix_start,
    unsigned long long    stride_pixels,
    unsigned long long    pixels_per_segment)
{{
    const unsigned long long pixel =
        (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned long long input_segment = blockIdx.y;
    if (pixel >= pixels_per_segment)
        return;

    const unsigned long long input_index =
        pix_start + input_segment * stride_pixels + pixel;
    const unsigned long long output_index =
        (unsigned long long)output_rows[input_segment] * pixels_per_segment + pixel;
    out[output_index] = src[input_index];
}}
"""
