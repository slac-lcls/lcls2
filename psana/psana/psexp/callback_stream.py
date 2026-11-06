"""Bounded, same-thread production batches from a run-scoped SMD callback."""

from array import array
from contextvars import copy_context
from operator import index

from . import TransitionId
from .callback_iterator import CallbackIterator


class _TransitionSink:
    """RunSmallData/Step append transitions here while the callback executes."""

    def __init__(self, batches):
        self.batches = batches

    def append(self, record):
        self.batches._append(record)

    def clear(self):
        self.batches._reset()


class StreamingCallbackBatches:
    """Suspend the callback stack when a batch is ready, even without L1 yields.

    All callback, reader and MPI work stays on the caller's thread. No producer
    runs ahead while the consumer processes a batch. Serialized pending records
    are bounded by count and bytes (one oversized record is permitted). User-held
    events and upstream input chunks are outside this output-buffer bound.
    """

    def __init__(self, builders, configs, dsparms, run_serial=False,
                 max_records=1000, max_bytes=8 * 1024 * 1024):
        # Only callback users need this dependency at import/run time.
        from greenlet import greenlet

        if max_records <= 0 or max_bytes <= 0:
            raise ValueError("Callback batch limits must be positive")
        self.run_serial = run_serial
        self.max_records = max_records
        self.max_bytes = max_bytes
        self.batch_size = dsparms.batch_size if dsparms.batch_size > 0 else max_records
        self.iterator = CallbackIterator(builders, configs, dsparms,
                                         dsparms.smd_callback, drain_remaining=True)
        self.iterator.run.proxy_events = _TransitionSink(self)
        self._closed = False
        self._reset()
        self._worker = greenlet(self._produce)
        self._worker.gr_context = copy_context()

    def _reset(self):
        self._records = []
        self._nbytes = 0
        self._l1 = 0
        self._destination = None

    def _append(self, record):
        is_event = TransitionId.isEvent(record.service)
        dest = 0 if self.run_serial else index(record.destination)
        if is_event and dest < 0:
            raise ValueError("Callback destination must be a non-negative integer")
        packet = record.as_bytearray()
        # Flush before adding a record that would exceed a bound or mix default
        # and explicit routing. A batch has a single event destination; step
        # history supplies these transitions to other BD clients as needed.
        if self._records and (
            self._nbytes + len(packet) > self.max_bytes
            or (is_event and self._destination is not None and dest != self._destination)
        ):
            self._flush()
        self._records.append((is_event, packet))
        self._nbytes += len(packet)
        if is_event:
            self._destination = dest
            self._l1 += 1
        if (len(self._records) >= self.max_records or self._nbytes >= self.max_bytes
                or self._l1 >= self.batch_size):
            self._flush()

    @staticmethod
    def _pack(packets):
        sizes = [len(packet) for packet in packets]
        result = bytearray().join(packets)
        if sizes:
            result.extend(array("I", sizes + [len(sizes)]))
        return result, sizes

    def _flush(self):
        if not self._records:
            return
        if self._closed:
            raise GeneratorExit
        dest = self._destination if self._destination is not None else 0
        batch = {dest: self._pack([packet for _, packet in self._records])}
        steps = {} if self.run_serial else {
            dest: self._pack([packet for is_event, packet in self._records if not is_event])
        }
        self._reset()
        self._worker.parent.switch((batch, steps))

    def _produce(self):
        try:
            for evt in self.iterator:
                self._append(evt._proxy_evt)
            self._flush()
        except BaseException:
            try:
                self.iterator.close()
            except BaseException:
                pass
            raise
        else:
            self.iterator.close()

    def __iter__(self):
        return self

    def __next__(self):
        if self._closed or self._worker.dead:
            raise StopIteration
        try:
            result = self._worker.switch()
        except BaseException:
            self._closed = True
            raise
        if self._worker.dead:
            self._closed = True
            raise StopIteration
        return result

    def close(self):
        if self._closed:
            return
        self._closed = True
        if not self._worker.dead:
            try:
                self._worker.throw(GeneratorExit)
            except GeneratorExit:
                pass
        self.iterator.close()
        if not self._worker.dead:
            raise RuntimeError("Callback ignored cancellation")
