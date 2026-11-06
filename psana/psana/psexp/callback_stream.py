"""Synchronous production batches from a run-scoped SMD callback.

Output batch sizes are bounded. Pending transitions during a callback call that
does not yield are not bounded; prompt delivery for that case is deferred.
"""

from array import array
from collections import deque
from operator import index

from . import TransitionId
from .callback_iterator import CallbackIterator


class _TransitionSink:
    """RunSmallData/Step append transitions here while the callback executes."""

    def __init__(self, batches):
        self.batches = batches

    def append(self, record):
        self.batches._pending.append(self.batches._serialize(record))

    def clear(self):
        self.batches._pending.clear()
        self.batches._reset()


class StreamingCallbackBatches:
    """Keep one callback alive, collecting output when it yields or returns.

    All work executes synchronously on the caller's thread. The callback must
    yield or return before the consumer can receive another batch. Transitions
    serialize immediately, but their pending queue can grow during a long gap
    without a callback yield. Count/byte limits apply to emitted batches only
    (one oversized record is permitted), not to that queue.
    """

    def __init__(self, builders, configs, dsparms, run_serial=False,
                 max_records=1000, max_bytes=8 * 1024 * 1024):
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
        self._pending = deque()
        self._reset()
        self._batches = self._produce()

    def _reset(self):
        self._records = []
        self._nbytes = 0
        self._l1 = 0
        self._destination = None

    def _serialize(self, record):
        is_event = TransitionId.isEvent(record.service)
        dest = 0 if self.run_serial else index(record.destination)
        if is_event and dest < 0:
            raise ValueError("Callback destination must be a non-negative integer")
        return is_event, dest, record.as_bytearray()

    @staticmethod
    def _pack(packets):
        sizes = [len(packet) for packet in packets]
        result = bytearray().join(packets)
        if sizes:
            result.extend(array("I", sizes + [len(sizes)]))
        return result, sizes

    def _take_batch(self):
        dest = self._destination if self._destination is not None else 0
        batch = {dest: self._pack([packet for _, packet in self._records])}
        steps = {} if self.run_serial else {
            dest: self._pack([packet for is_event, packet in self._records if not is_event])
        }
        self._reset()
        return batch, steps

    def _produce(self):
        try:
            finished = False
            while not finished or self._pending:
                if not self._pending:
                    try:
                        evt = next(self.iterator)
                    except StopIteration:
                        finished = True
                    else:
                        self._pending.append(self._serialize(evt._proxy_evt))
                    # next() may have queued transitions before yielding an
                    # event or reaching run end. Preserve acquisition order.
                    if not self._pending:
                        break
                is_event, dest, packet = self._pending.popleft()
                if self._records and (
                    self._nbytes + len(packet) > self.max_bytes
                    or (is_event and self._destination is not None and dest != self._destination)
                ):
                    yield self._take_batch()
                self._records.append((is_event, packet))
                self._nbytes += len(packet)
                if is_event:
                    self._destination = dest
                    self._l1 += 1
                if (len(self._records) >= self.max_records or self._nbytes >= self.max_bytes
                        or self._l1 >= self.batch_size):
                    yield self._take_batch()
            if self._records:
                yield self._take_batch()
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
        if self._closed:
            raise StopIteration
        try:
            return next(self._batches)
        except BaseException:
            self._closed = True
            raise

    def close(self):
        if self._closed:
            return
        self._closed = True
        try:
            self._batches.close()
        finally:
            self.iterator.close()
