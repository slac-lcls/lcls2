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


# Unknown completion must retain user owners even if an exception causes the
# caller to drop its manager. Successful close/retirement removes this root.
_failed_execution_pools = set()


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
    prepared_inputs: dict = field(default_factory=dict)
    producer_owners: list = field(default_factory=list)
    publications_by_ts: dict = field(default_factory=dict)
    publication_batches: list = field(default_factory=list)
    batch_inputs: object = None

    def release_storage(self):
        """Detach device references only after all terminal leases finish."""
        if self.batch_inputs is not None:
            self.batch_inputs.close()
            self.batch_inputs = None
        self.gpu_results_by_ts = {ts: dict.fromkeys(results)
                                 for ts, results in self.gpu_results_by_ts.items()}
        self.input_dgrams_by_ts = {}
        self.input_leases_by_ts = {}
        self.gpu_event_dgrams = ()
        self.prepared_inputs = {}
        self.producer_owners.clear()
        self.publications_by_ts.clear()
        self.publication_batches.clear()
        self.xtc_batch = None
        self.input_windows = ()


class EventPool:
    """Keep N GPU input batches in flight simultaneously.

    Submission records input completion. Retirement yields parsed inputs so
    consumers can register completion events, then drains every lease before
    the slot can be reused.

    Parameters
    ----------
    n : int
        Number of batches to keep in flight.  2 is a practical default.
    """

    def __init__(self, n: int = 2, *, budget=None):
        import cupy as cp
        self._n = n
        self._budget = budget
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

        try:
            old.stream.synchronize()
        except BaseException:
            _failed_execution_pools.add(self)
            raise
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
        except BaseException:
            # Leave the slot occupied because consumer completion was not
            # confirmed, but release the in-progress latch so retirement can
            # be retried instead of permanently locking the pool.
            self._retiring = None
            _failed_execution_pools.add(self)
            raise

        self._slots[old.slot_id] = None
        old.release_storage()
        self._retiring = None
        self._release_quarantine_if_drained()

    def submit(
        self, gv, gpu_read, event_envelopes: list, input_preparers=None,
        xtc_parser=None, *, input_windows=None, input_uses=None, batch_id=0,
        task=None, detector_bindings=None, task_constants=None, run=None,
        step_generation=0,
    ):
        """Queue execution using owned inputs, independently of its slot ID.

        The default creates one input window for the existing subbatch. An
        internal caller may instead supply resident and transient windows;
        that caller controls when those windows close to new planned uses.
        Internal task dispatch records publications but does not deliver them;
        public task processing remains gated until Stage 4 host delivery.
        """
        import cupy as cp
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
        prepared, publications, owners = {}, {}, []
        publication_batches = []
        producer_lease = None
        batch_inputs = None
        try:
            if task is not None:
                from .context import SlotLease
                # Result consumers must finish before any backing input lease
                # can retire: a publication may be a borrowed input view.
                producer_lease = SlotLease(None)
                all_leases.append(producer_lease)
                owners.extend((input_preparers or {}).values())
            execution_inputs = InputSlotLease(None, windows, planned_uses=input_uses)
            if windows:
                all_leases.append(execution_inputs)
            for window in windows:
                window.wait_ready(stream)
            gpu_event_dgrams = (
                GpuEventDgrams.from_windows(gv, windows, batch_id=batch_id)
                if gv is not None and windows else ()
            )

            gpu_results_by_ts = {}
            selected = gpu_event_dgrams
            if task is not None:
                from .gpu_task_batch import select_task_events, BatchInputContext
                selected = select_task_events(gpu_event_dgrams, event_envelopes)
            for name, preparer in (input_preparers or {}).items():
                if task is None:
                    prepared[name] = preparer.prepare_batch(selected, stream=stream, slot_id=slot)
                else:
                    prepared[name] = preparer.prepare_batch(selected, stream=stream,
                                                           slot_id=slot, aligned=True)
            if task is not None:
                batch_inputs = BatchInputContext(
                    selected, task, prepared, detector_bindings or {}, task_constants,
                    stream, owners, budget=self._budget, batch_id=batch_id,
                    run=run, step_generation=step_generation)
                from .gpu_producer import dispatch_task
                dispatch_task(task, batch_inputs, detector_bindings or {}, stream,
                              owners, publications, publication_batches, producer_lease)
            result_ready = cp.cuda.Event(disable_timing=True)
            result_ready.record(stream)
            if producer_lease is not None:
                producer_lease.result_ready = result_ready
            execution_inputs.result_ready = result_ready
            leases_by_ts = {}
            input_dgrams_by_ts, input_leases_by_ts = {}, {}
            for event in gpu_event_dgrams:
                lease = InputSlotLease(result_ready, event.input_windows, planned_uses=input_uses)
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
                prepared_inputs=prepared,
                producer_owners=owners, publications_by_ts=publications,
                publication_batches=publication_batches,
                batch_inputs=batch_inputs,
            )
        except BaseException:
            # Preserve every owner if CUDA completion cannot be established.
            # A subsequent close/flush can retry the same synchronization.
            failed = _EventSlot(slot, {}, [], stream, all_leases, {},
                                input_windows=windows, prepared_inputs=prepared,
                                producer_owners=owners, publications_by_ts=publications,
                                publication_batches=publication_batches,
                                batch_inputs=batch_inputs)
            try:
                stream.synchronize()
                for lease in all_leases:
                    lease.wait_until_safe_to_reuse()
                failed.release_storage()
            except BaseException:
                self._slots[slot] = failed
                self._write_idx += 1
                _failed_execution_pools.add(self)
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
            try:
                record.stream.synchronize()
            except BaseException:
                _failed_execution_pools.add(self)
                raise
            try:
                yield record
            finally:
                # The yield above is the registration window.  This finally
                # also protects generator close/early loop termination.
                try:
                    for lease in record.leases:
                        lease.wait_until_safe_to_reuse()
                except BaseException:
                    _failed_execution_pools.add(self)
                    raise
                self._slots[slot] = None
                record.release_storage()
                self._release_quarantine_if_drained()

    def _release_quarantine_if_drained(self):
        if self in _failed_execution_pools and not self.active_count:
            _failed_execution_pools.discard(self)

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

    def pinned_bytes(self):
        """Include metadata uploads retained after an unproven failed drain."""
        from .gpu_task_batch import _MetadataUpload
        return sum(owner.pinned_nbytes for record in self._slots if record is not None
                   for owner in record.producer_owners if isinstance(owner, _MetadataUpload))

    def __len__(self) -> int:
        return self._n
