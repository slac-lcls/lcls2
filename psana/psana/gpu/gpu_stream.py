"""Reusable CUDA stream slots for the integrated GPU event path.

EventPool manages N in-flight GPU subbatches. Each slot follows the
state machine documented in docs/memory_backpressure_and_results.md:

    FREE → READING/COMPUTING → RESULT_READY → CONSUMER_IN_FLIGHT → FREE

Slot leases and CUDA completion tokens connect the producer to terminal
consumers. The intended rule is that a slot cannot be recycled until all of
them complete. Input and result leases collect every terminal consumer and
prevent reuse while a registered view is still open.
"""

import os
from dataclasses import dataclass, field


@dataclass
class _EventSlot:
    """One occupied execution slot and its eventual host-result handles."""

    slot_id: int
    gpu_results_by_ts: dict
    event_envelopes: list
    stream: object
    leases: list
    leases_by_ts: dict
    xtc_batch: object = None
    input_windows: tuple = ()
    gpu_event_dgrams: tuple = ()
    input_dgrams_by_ts: dict = field(default_factory=dict)
    input_leases_by_ts: dict = field(default_factory=dict)
    pending_d2h_by_ts: dict = field(default_factory=dict)
    cached_cpu_results_by_ts: dict = field(default_factory=dict)


class EventPool:
    """Keep N GPU calibration batches in flight simultaneously.

    For each submitted batch:
      1. submit()      — launch detector work on the slot's stream; record one
                         result-ready event; create one SlotLease per result.
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
        # Each slot is an _EventSlot or None. Its leases include terminal
        # result leases and references to independently owned input windows.
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

    @property
    def next_stream(self):
        """Producer stream for parsing the next execution's transient inputs."""
        return self._streams[self.next_slot_id]

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
        old.gpu_results_by_ts = {ts: dict.fromkeys(results) for ts, results in old.gpu_results_by_ts.items()}
        old.input_dgrams_by_ts = {}
        old.input_leases_by_ts = {}
        old.gpu_event_dgrams = ()
        old.xtc_batch = None
        old.input_windows = ()
        self._retiring = None

    def submit(
        self, gv, gpu_read, event_envelopes: list, gpu_detectors: dict,
        xtc_parser=None, *, input_windows=None, batch_id=0,
    ):
        """Queue execution using owned inputs, independently of its slot ID.

        The default creates one input window for the existing subbatch. An
        internal caller may instead supply resident and transient windows;
        that caller controls when those windows close to new planned uses.
        """
        import cupy as cp
        from psana.gpu.context import SlotLease
        from psana.gpu.gpu_input import GpuEventDgrams, InputSlotLease

        slot = self.next_slot_id
        if self._slots[slot] is not None:
            raise RuntimeError(
                f"EventPool slot {slot} was submitted before retirement finished"
            )
        stream = self._streams[slot]
        null = getattr(cp.cuda.Stream, 'null', None)
        if null is not None:
            null.synchronize()

        owned_window = None
        if input_windows is None:
            if xtc_parser is not None:
                owned_window = xtc_parser.parse_window(gpu_read, stream, batch_id=batch_id)
            input_windows = () if owned_window is None else (owned_window,)
        windows = tuple(input_windows)
        all_leases = []
        try:
            execution_inputs = InputSlotLease(None, windows)
            if windows:
                all_leases.append(execution_inputs)
            for window in windows:
                window.wait_ready(stream)
            gpu_event_dgrams = (
                GpuEventDgrams.from_windows(gv, windows, batch_id=batch_id)
                if gv is not None and windows else ()
            )

            gpu_results_by_ts = {}
            for det_name, det_info in gpu_detectors.items():
                for ec in det_info[1].process_batch(
                    gpu_event_dgrams, stream=stream, slot_id=slot
                ):
                    results = gpu_results_by_ts.setdefault(ec.timestamp, {})
                    results[f'{det_name}.calib'] = ec.calib_gpu
                    if ec.raw_gpu is not None:
                        results[f'{det_name}.raw'] = ec.raw_gpu
                    if ec.image_gpu is not None:
                        results[f'{det_name}.image'] = ec.image_gpu

            result_ready = cp.cuda.Event(disable_timing=True)
            result_ready.record(stream)
            execution_inputs.result_ready = result_ready
            leases_by_ts = {}
            for ts, results in gpu_results_by_ts.items():
                leases_by_ts[ts] = {key: SlotLease(result_ready) for key in results}
                all_leases.extend(leases_by_ts[ts].values())

            input_dgrams_by_ts, input_leases_by_ts = {}, {}
            for event in gpu_event_dgrams:
                lease = InputSlotLease(result_ready, event.input_windows)
                event.bind_lease(lease)
                input_dgrams_by_ts[event.timestamp] = event
                input_leases_by_ts[event.timestamp] = lease
                all_leases.append(lease)
            if owned_window is not None:
                owned_window.close()  # leases keep it alive through delivery

            if os.environ.get('PSANA_GPU_MEM_DEBUG'):
                from psana.gpu.gpu_mpi import log_gpu_mem
                log_gpu_mem(f'EventPool.submit slot={slot} write={self._write_idx}')

            record = _EventSlot(
                slot_id=slot, gpu_results_by_ts=gpu_results_by_ts,
                event_envelopes=list(event_envelopes), stream=stream,
                leases=all_leases, leases_by_ts=leases_by_ts,
                xtc_batch=windows[0].batch if len(windows) == 1 else None,
                input_windows=windows, gpu_event_dgrams=gpu_event_dgrams,
                input_dgrams_by_ts=input_dgrams_by_ts,
                input_leases_by_ts=input_leases_by_ts,
            )
        except BaseException:
            # Preserve every owner if CUDA completion cannot be established.
            # A subsequent close/flush can retry the same synchronization.
            failed = _EventSlot(slot, {}, [], stream, all_leases, {}, input_windows=windows)
            try:
                stream.synchronize()
                for lease in all_leases:
                    lease.wait_until_safe_to_reuse()
            except BaseException:
                self._slots[slot] = failed
                self._write_idx += 1
                raise
            finally:
                if owned_window is not None:
                    owned_window.close()
            raise
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
                record.gpu_results_by_ts = {ts: dict.fromkeys(results) for ts, results in record.gpu_results_by_ts.items()}
                record.input_dgrams_by_ts = {}
                record.input_leases_by_ts = {}
                record.gpu_event_dgrams = ()
                record.xtc_batch = None
                record.input_windows = ()

    # ------------------------------------------------------------------
    # Inspection
    # ------------------------------------------------------------------

    @property
    def depth(self) -> int:
        """Number of batches that can be in flight simultaneously."""
        return self._n

    @property
    def active_count(self) -> int:
        """Occupied executions, including one exposed during retirement."""
        return sum(record is not None for record in self._slots)

    def __len__(self) -> int:
        return self._n
