"""Reusable CUDA stream slots for the integrated GPU event path.

EventPool manages N in-flight calibration batches.  Each slot follows
the state machine from gpu_memory_backpressure_and_async_join.md:

    FREE → READING/COMPUTING → RESULT_READY → CONSUMER_IN_FLIGHT → FREE

Slot leases and CUDA completion tokens connect the producer to its terminal
consumer.  A slot may not be recycled until every consumer (automatic D2H or
a downstream GPU kernel) has registered and completed its work.
"""

import os
from dataclasses import dataclass, field


@dataclass
class _EventSlot:
    """One occupied execution slot and its eventual host-result handles."""

    slot_id: int
    gpu_results_by_ts: dict
    cpu_evts: list
    stream: object
    leases: list
    leases_by_ts: dict
    pending_d2h_by_ts: dict = field(default_factory=dict)
    cached_cpu_results_by_ts: dict = field(default_factory=dict)


class EventPool:
    """Keep N GPU calibration batches in flight simultaneously.

    For each submitted batch:
      1. submit()      — launch calibration kernels on the slot's stream;
                         finalize routed results; record one result-ready
                         event; create one SlotLease per result.
      2. automatic D2H may be armed immediately against that event.
      3. begin_retire_next() — synchronise the producer stream but retain
                               ownership of the outgoing slot.
      4. finish_retire_next() — wait for each registered terminal consumer,
                                then release the slot for reuse.

    GpuEventManager must complete both retirement phases before submit()
    so the outgoing slot is fully drained before overwrite.  External-GPU mode
    yields between the phases so user work can register its completion event;
    automatic-D2H mode may finish retirement before yielding a host result.

    Parameters
    ----------
    n : int
        Number of batches to keep in flight.  2 is a practical default.
    """

    def __init__(self, n: int = 2):
        import cupy as cp
        self._n = n
        self._streams = [cp.cuda.Stream(non_blocking=True) for _ in range(n)]
        # Each slot is an _EventSlot or None.  leases is a flat list of all
        # SlotLease objects that protect buffers owned by that execution slot.
        self._slots: list = [None] * n
        self._write_idx = 0
        # Slot currently exposed between begin_retire_next() and
        # finish_retire_next().  Keeping it in _slots preserves ownership
        # while the yielded event registers an external completion token.
        self._retiring = None

    # ------------------------------------------------------------------
    # Main interface
    # ------------------------------------------------------------------

    @property
    def next_slot_id(self) -> int:
        """Slot index that the next submitted batch will occupy."""
        return self._write_idx % self._n

    def begin_retire_next(self):
        """Synchronize the outgoing producer but retain its slot lease.

        This is phase one of retirement.  The returned result is safe to
        expose to the caller, but its slot remains occupied so a later submit
        cannot overwrite it.  Before calling finish_retire_next(), the caller
        must ensure that any external consumer has registered its completion
        event.  Automatic consumers may already have registered at submission.

        Returns the occupied _EventSlot, or None if the slot is empty.  Its
        arrays remain valid through finish_retire_next().
        """
        if self._retiring is not None:
            raise RuntimeError("EventPool retirement already in progress")

        slot = self.next_slot_id
        old  = self._slots[slot]
        if old is None:
            return None

        old.stream.synchronize()
        self._retiring = old
        return old

    def finish_retire_next(self):
        """Wait for consumers registered after begin, then release the slot."""
        if self._retiring is None:
            return

        old = self._retiring
        # This lookup happens after the caller has consumed the yielded
        # result.  In particular, on_gpu_view().__exit__ may have registered
        # its external-kernel completion event during that interval.
        try:
            for lease in old.leases:
                lease.wait_until_safe_to_reuse()
        except Exception:
            # Leave the slot occupied because consumer completion was not
            # confirmed, but release the in-progress latch so retirement can
            # be retried instead of permanently locking the pool.
            self._retiring = None
            raise

        self._slots[old.slot_id] = None
        self._retiring = None

    def submit(self, gv, gpu_read, cpu_evts: list, gpu_detectors: dict,
               finalize_results=None):
        """Queue calibration into the already-retired next slot.

        Records a result-ready CUDA event after calibration and final routing
        are queued, then creates one SlotLease per result so downstream
        consumers can release the slot when done.

        ``finalize_results`` may enqueue routing or assembly work on the same
        producer stream before the final result-ready event is recorded.

        Returns the occupied _EventSlot.  Automatic consumers such as D2H may
        attach their completion tokens immediately; results are delivered later
        by begin_retire_next().
        """
        import cupy as cp
        from psana.gpu.context import SlotLease

        slot   = self.next_slot_id
        if self._slots[slot] is not None:
            raise RuntimeError(
                f"EventPool slot {slot} was submitted before retirement finished"
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

        if finalize_results is not None:
            gpu_results_by_ts = finalize_results(
                gpu_results_by_ts, cpu_evts, stream
            )

        # Record ONE result-ready event after calibration and any final routing
        # work are queued.  All results share this event because they run on the
        # same slot stream.
        result_ready = cp.cuda.Event(disable_timing=True)
        result_ready.record(stream)

        # Create one SlotLease per (timestamp, det, result_type) — each
        # gets the shared result_ready event but its own view (array slice).
        leases_by_ts: dict = {}   # {ts: {key: SlotLease}}
        all_leases: list  = []
        for ts, ts_dict in gpu_results_by_ts.items():
            ts_leases = {}
            for key, arr in ts_dict.items():
                lease = SlotLease(result_ready)
                ts_leases[key] = lease
                all_leases.append(lease)
            leases_by_ts[ts] = ts_leases

        if os.environ.get('PSANA_GPU_MEM_DEBUG'):
            try:
                from psana.gpu.gpu_mpi import log_gpu_mem
                log_gpu_mem(f'EventPool.submit slot={slot} '
                            f'write={self._write_idx}')
            except Exception:
                pass

        record = _EventSlot(
            slot_id=slot,
            gpu_results_by_ts=gpu_results_by_ts,
            cpu_evts=list(cpu_evts),
            stream=stream,
            leases=all_leases,
            leases_by_ts=leases_by_ts,
        )
        self._slots[slot] = record
        self._write_idx += 1
        return record

    def flush(self):
        """Drain all remaining in-flight slots in submission order.

        Synchronizes each producer, yields its _EventSlot so consumers can
        register completion, then waits for those consumers before clearing
        the slot.
        """
        for i in range(self._n):
            slot = (self._write_idx + i) % self._n
            if self._slots[slot] is None:
                continue
            record = self._slots[slot]
            record.stream.synchronize()
            try:
                yield record
            finally:
                # The yield above is the registration window.  This finally
                # also protects generator close/early loop termination.
                for lease in record.leases:
                    lease.wait_until_safe_to_reuse()
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
