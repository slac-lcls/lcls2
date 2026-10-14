"""Reusable CUDA stream slots for the integrated GPU event path."""


class EventPool:
    """Keep N GPU calibration batches in flight simultaneously.

    While batch N's calibration kernel runs on stream[N % n], batch N+1's
    GDS reads and CPU work proceed concurrently.  Results from batch N are
    retired (and its stream synchronised) before slot N % n is recycled for
    batch N + n.  The caller yields the retired result before submitting the
    replacement because calibrated arrays alias reusable per-slot buffers.

    Timeline with n=2 (two batches overlapping)
    -------------------------------------------
    Batch 0: submit → launch calib on stream_0 (non-blocking)
                       store (results_0, evts_0) in slot 0
                       nothing to return yet (slot was empty)
    Batch 1: submit → launch calib on stream_1 (non-blocking)
                       store (results_1, evts_1) in slot 1
    Batch 2: retire_next → sync stream_0, yield results_0
             submit     → launch calib on stream_0 (now safe to recycle)
                          store (results_2, evts_2) in slot 0
    flush() → sync remaining slots in order, yield results

    Benefit
    -------
    For detectors whose single-batch calibration kernel does not fully saturate
    the GPU (small detectors, short kernels), n=2 or n=4 allows consecutive
    batches' kernels to execute concurrently on separate SMs, improving GPU
    utilisation.  For large detectors like Jungfrau on an A100 — where a single
    2.6 M-pixel batch already saturates the GPU — the benefit is smaller but
    the pattern still reduces CPU-side synchronisation overhead.

    Parameters
    ----------
    n : int
        Number of batches to keep in flight.  2 is the practical default
        (two streams: compute for batch N, prefetch for batch N+1).
        4 is appropriate when single-batch kernels are very short (<0.5 ms).

    Guide reference: §12 (EventPool — N events in flight).
    """

    def __init__(self, n: int = 2):
        import cupy as cp
        self._n = n
        self._streams = [cp.cuda.Stream(non_blocking=True) for _ in range(n)]
        # Each slot: (gpu_results_by_ts, cpu_evts, stream) | None
        self._slots: list = [None] * n
        self._write_idx = 0

    # ------------------------------------------------------------------
    # Main interface
    # ------------------------------------------------------------------

    @property
    def next_slot_id(self) -> int:
        """Slot that the next submitted batch will occupy."""
        return self._write_idx % self._n

    def retire_next(self):
        """Synchronise and remove the batch occupying the next write slot.

        The returned arrays remain valid only until the caller submits the
        replacement batch into this slot.  GpuEvents therefore yields them
        before calling :meth:`submit`.
        """
        slot = self.next_slot_id
        old = self._slots[slot]
        if old is None:
            return None

        old_results, old_evts, old_stream = old
        old_stream.synchronize()
        self._slots[slot] = None
        return old_results, old_evts

    def submit(self, gv, gpu_read, cpu_evts: list, gpu_detectors: dict):
        """Launch calibration into the already-retired next slot.

        Calibration kernels are queued without a host synchronisation, so
        work in other slots can overlap.  Call :meth:`retire_next` before this
        method to make both the calibrated-output and raw-input slot safe to
        reuse.

        Parameters
        ----------
        gv            : GpuBatchView
        gpu_read      : KvikioBatchRead with ``data_gpu`` populated
        cpu_evts      : list of psana2 Event objects for this batch
        gpu_detectors : dict  {det_name: (psana_det, GPUDetector)}

        Returns
        -------
        None.  Retired results are returned by :meth:`retire_next`.
        """
        slot    = self.next_slot_id
        if self._slots[slot] is not None:
            raise RuntimeError(
                f"EventPool slot {slot} was submitted without retire_next()"
            )
        stream  = self._streams[slot]    # ← no synchronize() here

        # Launch calibration on this slot's stream (non-blocking).
        gpu_results_by_ts: dict = {}
        for det_name, det_info in gpu_detectors.items():
            # det_info is (psana_det, gpu_det_obj) or
            # (psana_det, gpu_det_obj, cpu_seg_map) for D1 combined routing.
            gpu_det_obj = det_info[1]
            for ec in gpu_det_obj.process_batch(gv, gpu_read, stream=stream,
                                                slot_id=slot):
                ts_dict = gpu_results_by_ts.setdefault(ec.timestamp, {})
                ts_dict[f'{det_name}.calib'] = ec.calib_gpu
                if ec.raw_gpu is not None:
                    ts_dict[f'{det_name}.raw'] = ec.raw_gpu
                if ec.image_gpu is not None:
                    ts_dict[f'{det_name}.image'] = ec.image_gpu

        if __import__('os').environ.get('PSANA_GPU_MEM_DEBUG'):
            try:
                from psana.gpu.gpu_mpi import log_gpu_mem
                log_gpu_mem(f'EventPool.submit slot={slot} write={self._write_idx}')
            except Exception:
                pass
        self._slots[slot] = (gpu_results_by_ts, list(cpu_evts), stream)
        self._write_idx += 1

        return None

    def flush(self):
        """Drain all remaining in-flight slots in submission order.

        Yields (gpu_results_by_ts, cpu_evts) for each non-empty slot,
        synchronising the slot's stream before yielding.

        Call this after the main batch loop to ensure no results are lost.
        """
        for i in range(self._n):
            slot = (self._write_idx + i) % self._n
            if self._slots[slot] is None:
                continue
            results, evts, stream = self._slots[slot]
            stream.synchronize()
            yield results, evts
            self._slots[slot] = None

    # ------------------------------------------------------------------
    # Inspection
    # ------------------------------------------------------------------

    @property
    def depth(self) -> int:
        """Number of batches that can be in flight simultaneously."""
        return self._n

    def __len__(self) -> int:
        return self._n
