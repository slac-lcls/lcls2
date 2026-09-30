"""Bounded host delivery of contiguous user-publication groups.

One copy per group and one terminal event per execution. Host tokens never own
user device arrays. Pinned capacity includes cached and token-held blocks;
pressure uses a synchronous ordinary-host copy, without waiting for delivery.
"""
from math import prod
import os
from threading import RLock
from weakref import WeakSet

import numpy as np


DEFAULT_PINNED_BYTES = 64 << 20


def validate_pinned_bytes(value):
    if type(value) is not int:
        raise TypeError('gpu_d2h_pinned_bytes must be an int')
    if value < 0:
        raise ValueError('gpu_d2h_pinned_bytes must be nonnegative')
    return value


class _CopyCompletion:
    """Root destinations before enqueue; use stream drain if recording fails."""
    def __init__(self, stream):
        self._stream, self._event = stream, None
        self._owners = []
        self._complete = False
        self._lock = RLock()

    def retain(self, owner):
        self._owners.append(owner)

    def arm(self, event):
        self._event = event

    def _release(self):
        self._complete = True
        self._owners.clear()
        self._stream = self._event = None

    def synchronize(self):
        with self._lock:
            if not self._complete:
                (self._event if self._event is not None else self._stream).synchronize()
                self._release()

    def query(self):
        with self._lock:
            if not self._complete:
                ready = self._event.query() if self._event is not None else self._stream.done
                if ready:
                    self._release()
            return self._complete


class _HostGroup:
    def __init__(self, host, done):
        self.host, self.done = host, done
        self.tokens = WeakSet()
        self.lock = RLock()

    def reusable(self):
        with self.lock:
            if self.tokens:
                return False
            if self.done is not None and not self.done.query():
                return False
            self.host = self.done = None
            return True

    def materialize(self):
        with self.lock:
            if self.done is not None:
                self.done.synchronize()
            for token in list(self.tokens):
                token.get()
            self.host = self.done = None


class HostResult:
    """One host row with independent cached NumPy storage on first access."""
    def __init__(self, group, row, shape, dtype):
        self._group, self._row = group, row
        self.shape, self.dtype = shape, dtype
        self._cache = None
        group.tokens.add(self)

    def get(self):
        # Keep the group reference stable: close and user access can race while
        # NumPy's copy releases the GIL. The shared lock serializes both paths.
        group = self._group
        with group.lock:
            if self._cache is None:
                if group.done is not None:
                    group.done.synchronize()
                self._cache = group.host[self._row:self._row+1].reshape(self.shape).copy()
                group.tokens.discard(self)
                if not group.tokens:
                    group.host = group.done = None
            return self._cache


class _PinnedBlock:
    def __init__(self, capacity):
        import cupy as cp
        # Direct allocation bypasses CuPy's global pinned pool and its retained
        # rounded blocks. Account whole OS pages, not only logical payload.
        self.owner = cp.cuda.PinnedMemoryPointer(cp.cuda.PinnedMemory(capacity), 0)
        self.capacity = capacity
        self.group = None


class PublicationD2H:
    """One run/BD's aggregate output staging cap, shared across all names.

    Free blocks remain charged. No cache eviction or user-GPU allocation occurs.
    cap=0 and oversized groups use blocking copies into ordinary NumPy memory.
    """
    def __init__(self, pinned_bytes=DEFAULT_PINNED_BYTES):
        self.limit = validate_pinned_bytes(pinned_bytes)
        self._capacity = 0
        self._blocks = []
        self._groups = WeakSet()
        self._stream = None
        self._closed = False
        self._page_size = os.sysconf('SC_PAGE_SIZE')

    @property
    def pinned_bytes(self):
        return self._capacity

    def _destination(self, shape, dtype, nbytes):
        if not nbytes:
            return np.empty(shape, dtype), None
        for block in self._blocks:
            if block.capacity >= nbytes and (block.group is None or block.group.reusable()):
                return np.ndarray(shape, dtype, buffer=block.owner), block
        capacity = ((nbytes + self._page_size-1) // self._page_size) * self._page_size
        if self._capacity + capacity <= self.limit:
            self._capacity += capacity
            try:
                block = _PinnedBlock(capacity)
            except BaseException:
                self._capacity -= capacity
                raise
            self._blocks.append(block)
            return np.ndarray(shape, dtype, buffer=block.owner), block
        return np.empty(shape, dtype), None

    def enqueue(self, record):
        if self._closed:
            raise RuntimeError('publication delivery is closed')
        publications = record.publication_batches
        if not publications:
            return
        # Validate all metadata before issuing any copies, including mutations
        # to an array after publish() registered its original layout.
        for pub in publications:
            array = pub.array
            if (tuple(array.shape) != pub.shape or np.dtype(array.dtype) != pub.dtype
                    or int(array.nbytes) != pub.nbytes or not array.flags.c_contiguous
                    or pub.nbytes != prod(pub.shape) * pub.dtype.itemsize):
                raise ValueError(f'publication {pub.name!r} metadata changed after registration')
        publications = [pub for pub in publications if pub.timestamps]
        if not publications:
            return
        lease = publications[0].lease
        if any(pub.lease is not lease for pub in publications):
            raise ValueError('publication groups must share one producer lease')
        producer_done = lease.result_ready
        pending, results = {}, {}
        needs_copy = any(pub.nbytes for pub in publications)
        done = producer_done
        if needs_copy:
            import cupy as cp
            if self._stream is None:
                self._stream = cp.cuda.Stream(non_blocking=True)
            done = _CopyCompletion(self._stream)
            # Registration precedes every copy command. EventPool retains this
            # guard and the producer's owners if any later drain is unproven.
            lease.register_consumer_done(done)
        try:
            if needs_copy:
                self._stream.wait_event(producer_done)
            for pub in publications:
                host, block = self._destination(pub.shape, pub.dtype, pub.nbytes)
                if needs_copy:
                    done.retain(host)
                group = _HostGroup(host, done)
                self._groups.add(group)
                if block is not None:
                    block.group = group
                if pub.nbytes:
                    pub.array.get(out=host, stream=self._stream, blocking=block is None)
                for row, ts in enumerate(pub.timestamps):
                    pending.setdefault(ts, {})[pub.name] = HostResult(group, row, pub.shape[1:], pub.dtype)
                    results.setdefault(ts, {})[pub.name] = None
            if needs_copy:
                event = cp.cuda.Event(disable_timing=True)
                event.record(self._stream)
                done.arm(event)
        except BaseException:
            if needs_copy:
                done.synchronize()
            raise
        # Publish the handoff atomically only after successful terminal record.
        record.pending_d2h_by_ts = pending
        record.gpu_results_by_ts = results

    def close(self):
        if self._closed:
            return
        # Retained events must not pin this run's cache indefinitely. Materialize
        # only still-live rows, then detach every pinned alias before credits.
        for group in list(self._groups):
            group.materialize()
        for block in self._blocks:
            block.group = None
        self._blocks.clear()
        self._capacity = 0
        self._stream = None
        self._closed = True
