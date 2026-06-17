"""
gpu_stream.py — StreamPool for CUDA stream management.

All GPU operations in the prototype default to CuPy's default CUDA stream
(stream 0).  This is convenient but causes two problems at scale:

  1. Serialisation when multiple MPI ranks share one GPU — the default stream
     is shared across CUDA contexts; non-blocking streams are rank-private.
  2. No overlap between H→D transfers and compute — different work must be on
     different streams to run concurrently on separate hardware engines.

StreamPool pre-allocates a fixed set of non-blocking streams and round-robins
through them.  acquire() synchronises the selected slot before returning it,
ensuring no prior work on that stream is still in flight.

Usage
-----
    pool   = StreamPool(size=2)       # two streams: compute + prefetch
    stream = pool.acquire()           # synchronises previous work on this slot
    with stream:
        calib = fused_calib_gpu(raw_gpu, peds_gpu, gmask_gpu)
    # ... other work on CPU or a different stream ...
    # calib is safe to read after stream.synchronize() or pool.acquire() on
    # the same slot.

Guide reference: §7 (CUDA Streams and the StreamPool) and §9 (Multiple Ranks
on One GPU — stream isolation).
"""


class StreamPool:
    """Reusable pool of non-blocking CUDA/HIP streams.

    Parameters
    ----------
    size : int
        Number of streams to pre-allocate.  2 is optimal for a simple
        compute/prefetch ping-pong (S3DF benchmark: 2 streams beat 4 for
        ePix10k calibration — kernels too short for additional overlap).
        4 is appropriate when using EventPool with N events in flight.
    """

    def __init__(self, size: int = 4):
        import cupy as cp
        self._streams = [cp.cuda.Stream(non_blocking=True)
                         for _ in range(size)]
        self._idx = 0

    def acquire(self):
        """Return the next stream from the pool.

        Synchronises the selected stream before returning it to ensure any
        previously submitted work on this slot has completed.  The caller
        can then submit new work without worrying about ordering.

        Returns
        -------
        cp.cuda.Stream
        """
        stream = self._streams[self._idx % len(self._streams)]
        stream.synchronize()   # wait for previous work on this slot
        self._idx += 1
        return stream

    def synchronize_all(self):
        """Block until all streams in the pool have completed their work."""
        for s in self._streams:
            s.synchronize()

    def __len__(self) -> int:
        return len(self._streams)


class EventPool:
    """Keep N GPU calibration batches in flight simultaneously.

    While batch N's calibration kernel runs on stream[N % n], batch N+1's
    GDS reads and CPU work proceed concurrently.  Results from batch N are
    returned (and its stream synchronised) when slot N % n is recycled for
    batch N + n, guaranteeing they are ready to yield to the user.

    Timeline with n=2 (two batches overlapping)
    -------------------------------------------
    Batch 0: submit → launch calib on stream_0 (non-blocking)
                       store (results_0, evts_0) in slot 0
                       nothing to return yet (slot was empty)
    Batch 1: submit → launch calib on stream_1 (non-blocking)
                       store (results_1, evts_1) in slot 1
                       return slot 0: sync stream_0, yield results_0
    Batch 2: submit → launch calib on stream_0 (recycled)
                       store (results_2, evts_2) in slot 0
                       return slot 1: sync stream_1, yield results_1
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

    def submit(self, gv, gpu_read, cpu_evts: list, gpu_detectors: dict):
        """Launch calibration for this batch and return old results if ready.

        Calibration kernels are queued on the slot's stream WITHOUT calling
        synchronize() first — this allows the new kernel to overlap with work
        on other streams.  The slot that is being recycled (n batches old) is
        synchronised here before its results are returned.

        Parameters
        ----------
        gv            : GpuBatchView
        gpu_read      : KvikioBatchRead (data_gpu populated, compute_digest=False)
        cpu_evts      : list of psana2 Event objects for this batch
        gpu_detectors : dict  {det_name: (psana_det, GPUDetector)}

        Returns
        -------
        (gpu_results_by_ts: dict, cpu_evts: list) from n batches ago, or None.
        """
        slot    = self._write_idx % self._n
        old     = self._slots[slot]
        stream  = self._streams[slot]    # ← no synchronize() here

        # Launch calibration on this slot's stream (non-blocking).
        gpu_results_by_ts: dict = {}
        for det_name, det_info in gpu_detectors.items():
            # det_info is (psana_det, gpu_det_obj) or
            # (psana_det, gpu_det_obj, cpu_seg_map) for D1 combined routing.
            gpu_det_obj = det_info[1]
            for ec in gpu_det_obj.process_batch(gv, gpu_read, stream=stream):
                ts_dict = gpu_results_by_ts.setdefault(ec.timestamp, {})
                ts_dict[f'{det_name}.calib'] = ec.calib_gpu
                if ec.raw_gpu is not None:
                    ts_dict[f'{det_name}.raw'] = ec.raw_gpu
                if ec.image_gpu is not None:
                    ts_dict[f'{det_name}.image'] = ec.image_gpu

        self._slots[slot] = (gpu_results_by_ts, list(cpu_evts), stream)
        self._write_idx += 1

        # Return the recycled slot's results, synchronised.
        if old is not None:
            old_results, old_evts, old_stream = old
            old_stream.synchronize()    # wait for old batch's kernel
            return old_results, old_evts
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
