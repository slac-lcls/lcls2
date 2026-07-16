"""Reusable CUDA stream slots for the integrated GPU event path.

EventPool manages N in-flight calibration batches.  Each slot follows
the state machine from gpu_memory_backpressure_and_async_join.md:

    FREE → READING/COMPUTING → RESULT_READY → D2H_IN_FLIGHT → FREE

Phase 1 of the design: slot leases and CUDA completion tokens.
The calibration stream is no longer the only completion signal — a slot
may not be recycled until every consumer (D2H or downstream GPU kernel)
has registered completion via SlotLease.register_d2h_done().
"""


class EventPool:
    """Keep N GPU calibration batches in flight simultaneously.

    For each submitted batch:
      1. submit()      — launch calibration kernels on the slot's stream;
                         record one calib_done CUDA event covering all
                         kernels; create one SlotLease per event.
      2. retire_next() — wait for each SlotLease's consumer to complete
                         (e.g. D→H), then synchronise the calib stream
                         and yield the results for the caller to process.

    The caller (GpuEvents) must call retire_next() before submit() so
    the outgoing slot is fully drained before its buffers are overwritten.

    Parameters
    ----------
    n : int
        Number of batches to keep in flight.  2 is a practical default.
    """

    def __init__(self, n: int = 2):
        import cupy as cp
        self._n = n
        self._streams = [cp.cuda.Stream(non_blocking=True) for _ in range(n)]
        # Each slot: (gpu_results_by_ts, cpu_evts, stream, leases) | None
        # leases is a flat list of all SlotLease objects for that slot.
        self._slots: list = [None] * n
        self._write_idx = 0

    # ------------------------------------------------------------------
    # Main interface
    # ------------------------------------------------------------------

    @property
    def next_slot_id(self) -> int:
        """Slot index that the next submitted batch will occupy."""
        return self._write_idx % self._n

    def retire_next(self):
        """Wait for all consumers to finish, then release the next slot.

        Phase 1: before synchronising the calibration stream, wait for
        every SlotLease in the outgoing slot to signal that its D→H (or
        other downstream consumer) has completed.  This guarantees the
        slot buffers are not overwritten while data is still in transit.

        Returns (gpu_results_by_ts, cpu_evts) or None if the slot is empty.
        The returned arrays remain valid until submit() is called next.
        """
        slot = self.next_slot_id
        old  = self._slots[slot]
        if old is None:
            return None

        old_results, old_evts, old_stream, old_leases, old_leases_by_ts = old

        # Wait for every consumer (D→H or downstream GPU kernel) that
        # registered a completion token on one of this slot's leases.
        for lease in old_leases:
            lease.wait_until_safe_to_reuse()

        # Now safe to synchronise the calibration stream and recycle.
        old_stream.synchronize()
        self._slots[slot] = None
        return old_results, old_evts, old_leases_by_ts

    def submit(self, gv, gpu_read, cpu_evts: list, gpu_detectors: dict):
        """Queue calibration into the already-retired next slot.

        Records a calib_done CUDA event after all calibration kernels
        are queued, then creates one SlotLease per event so downstream
        consumers can issue async D→H and release the slot when done.

        Returns None.  Results come back via retire_next().
        """
        import cupy as cp
        from psana.gpu.context import SlotLease

        slot   = self.next_slot_id
        if self._slots[slot] is not None:
            raise RuntimeError(
                f"EventPool slot {slot} was submitted without retire_next()"
            )
        stream = self._streams[slot]

        # Synchronise the null (default) stream before launching the
        # calibration kernel.  Any on_gpu D→D copies issued by the user
        # in the previous iteration run on the null stream; without this
        # sync they could race with the new calib kernel which writes to
        # the same slot buffer (Race 1).  The sync is a no-op if no
        # null-stream work is pending, so it adds negligible overhead.
        try:
            cp.cuda.Stream.null.synchronize()
        except Exception:
            pass

        # Launch calibration on this slot's non-blocking stream.
        gpu_results_by_ts: dict = {}
        for det_name, det_info in gpu_detectors.items():
            gpu_det_obj = det_info[1]
            for ec in gpu_det_obj.process_batch(gv, gpu_read, stream=stream,
                                                slot_id=slot):
                ts_dict = gpu_results_by_ts.setdefault(ec.timestamp, {})
                ts_dict[f'{det_name}.calib'] = ec.calib_gpu
                if ec.raw_gpu is not None:
                    ts_dict[f'{det_name}.raw'] = ec.raw_gpu
                if ec.image_gpu is not None:
                    ts_dict[f'{det_name}.image'] = ec.image_gpu

        # Record ONE calib_done event after all kernels are queued.
        # All events in this batch share this single event — they all
        # ran on the same stream so any one of them completing means all
        # preceding work is done.
        calib_done = cp.cuda.Event(disable_timing=True)
        calib_done.record(stream)

        # Create one SlotLease per (timestamp, det, result_type) — each
        # gets the shared calib_done event but its own view (array slice).
        leases_by_ts: dict = {}   # {ts: {key: SlotLease}}
        all_leases: list  = []
        for ts, ts_dict in gpu_results_by_ts.items():
            ts_leases = {}
            for key, arr in ts_dict.items():
                lease = SlotLease(slot, calib_done, arr)
                ts_leases[key] = lease
                all_leases.append(lease)
            leases_by_ts[ts] = ts_leases

        if __import__('os').environ.get('PSANA_GPU_MEM_DEBUG'):
            try:
                from psana.gpu.gpu_mpi import log_gpu_mem
                log_gpu_mem(f'EventPool.submit slot={slot} '
                            f'write={self._write_idx}')
            except Exception:
                pass

        self._slots[slot] = (gpu_results_by_ts, list(cpu_evts),
                             stream, all_leases, leases_by_ts)
        self._write_idx += 1
        return None

    def flush(self):
        """Drain all remaining in-flight slots in submission order.

        Waits for every consumer lease before synchronising each stream.
        Yields (gpu_results_by_ts, cpu_evts) for each non-empty slot.
        """
        for i in range(self._n):
            slot = (self._write_idx + i) % self._n
            if self._slots[slot] is None:
                continue
            results, evts, stream, leases, leases_by_ts = self._slots[slot]
            for lease in leases:
                lease.wait_until_safe_to_reuse()
            stream.synchronize()
            yield results, evts, leases_by_ts
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
