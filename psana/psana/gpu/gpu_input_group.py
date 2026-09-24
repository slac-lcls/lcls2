"""Bounded independent input owners, opt-in until Stage 3 scheduling.

The pool owns a dedicated KvikioGpuReader's slots. A small stream keeps its
credit until its group's last planned/active/CUDA consumer finishes, including
across EB batches. Polling examines all groups rather than a retirement queue.
"""
from dataclasses import dataclass, field

import numpy as np


@dataclass
class _GroupState:
    key: tuple
    group: object
    slot: int
    pending: object = None
    read: object = None
    release_raw: object = None
    window: object = None
    planned: dict = field(default_factory=dict)
    error: object = None


class InputGroupPool:
    """One BD-thread owner of bounded raw slots and group lifecycle metadata.

    Caller selects eligible requests from a plan and submits each exactly once.
    The pool enforces slot/small-stream availability and the reader's allocation
    budget. Parser/compute growth reservations remain the caller's obligation.
    No production scheduling policy or kernel batching is installed here.
    """

    def __init__(self, reader):
        if not reader.bulk_read or reader._n_slots <= 0:
            raise ValueError('group pool requires a bulk-capable reader with slots')
        if reader._pending or any(reader._input_holds.values()):
            raise ValueError('group pool requires a reader without live inputs')
        self.reader = reader
        self._groups = {}
        self._small = {}
        self._closed = False

    @property
    def live_keys(self):
        return tuple(self._groups)

    def issue(self, batch_id, group):
        """Return (batch, group) key, or None for slot/small-stream backpressure.

        No GPU wait is introduced here. Poll completed groups before choosing
        a slot; budget exhaustion remains an explicit pre-allocation exception.
        """
        if self._closed:
            raise RuntimeError('input group pool is closed')
        self.poll()
        key = (int(batch_id), group.group_id)
        if key in self._groups:
            raise ValueError(f'duplicate live input group {key}')
        if group.small and group.stream_id in self._small:
            return None
        busy = {g.slot for g in self._groups.values()}
        free = [i for i in range(self.reader._n_slots) if i not in busy]
        if not free:
            return None
        # Prefer a reusable fitting buffer to allocating another generation.
        fits = [i for i in free if self.reader._slot_bufs[i] is not None
                and self.reader._slot_bufs[i].nbytes >= group.size]
        slot = min(fits, key=lambda i: self.reader._slot_bufs[i].nbytes) if fits else free[0]
        state = _GroupState(key, group, slot)
        self._groups[key] = state
        if group.small:
            self._small[group.stream_id] = key
        try:
            state.pending = self.reader.issue_group(group, slot_id=slot)
        except BaseException:
            # issue_batch drains any partial submission before raising and
            # poisons the reader on I/O failure. Allocation failures occur
            # before submission. Both leave no newly pending future here.
            self._forget(state)
            raise
        return key

    def read(self, key):
        """Collect KvikIO completion and pin raw storage until ownership transfers."""
        state = self._groups[key]
        if state.error is not None:
            raise RuntimeError('input group failed') from state.error
        if state.read is None:
            try:
                state.read = self.reader.wait_batch(state.pending)
                state.release_raw = state.read.retain_input()
            except BaseException as exc:
                state.error = exc
                raise
        return state.read

    def bind(self, key, window):
        """Attach parsed storage and seal one planned use for each present event.

        The supplied window must own this read via retain_input() and use
        deferred retirement. Separate windows may share a batched parser
        allocation through its release callback; the group API does not require
        a parser launch per read request.
        """
        state = self._groups[key]
        if state.window is not None:
            raise RuntimeError('input group is already bound')
        read = self.read(key)
        if (window.batch_id != key[0] or not window._defer_retirement
                or window.batch.data_gpu is not read.data_gpu
                or not np.array_equal(window.desc_table, read.desc_table)):
            raise ValueError('window does not match this group or lacks deferred retirement')
        state.window = window
        # Retain the pool's raw hold as well as the parser's, until poll removes
        # the group. This makes the backing safe even through partial setup.
        for d in state.group.dgrams:
            state.planned[d.batch_event_index] = window.acquire()
        window.close()  # existing planned references can still fork consumers
        return window

    def parse(self, key, parser, stream):
        """Convenience for ownership tests; runtime may batch parsing then bind."""
        read = self.read(key)
        window = parser.parse_window(read, stream, batch_id=key[0], defer_retirement=True)
        return self.bind(key, window)

    def take_use(self, key, event_index):
        """Transfer one pre-registered use to the caller; caller must release it.

        Consumers can fork this use before releasing it, including after the
        window has closed to new root acquisitions. Untaken uses block reuse.
        """
        state = self._groups[key]
        if state.window is None:
            raise RuntimeError('input group is not parsed')
        return state.planned.pop(event_index)

    def poll(self):
        """Reclaim every ready group, including later ones; never synchronize CUDA."""
        reclaimed, error = [], None
        for state in tuple(self._groups.values()):
            if state.window is None:
                continue
            try:
                if state.window.poll():
                    self._forget(state)
                    reclaimed.append(state.key)
            except BaseException as exc:
                if error is None:
                    error = exc
        if error is not None:
            raise error
        return tuple(reclaimed)

    def _forget(self, state):
        if state.release_raw is not None:
            state.release_raw()
            state.release_raw = None
        self._groups.pop(state.key, None)
        if self._small.get(state.group.stream_id) == state.key:
            del self._small[state.group.stream_id]

    def close(self):
        """Stop submission, drain I/O, cancel untaken planned uses, drain CUDA.

        Active uses transferred to callers cannot be cancelled here. They keep
        groups alive and cause an explicit error; release them and retry close.
        Caller must also close parser pools to drain failed parser submissions.
        """
        self._closed = True
        error = None
        for state in tuple(self._groups.values()):
            try:
                if state.pending is not None and not state.pending.completed:
                    self.read(state.key)
                if state.window is not None:
                    for use in tuple(state.planned.values()):
                        use.wait_until_safe_to_reuse()
                    state.planned.clear()
                    if not state.window.drain():
                        raise RuntimeError(f'input group {state.key} still has live consumers')
                self._forget(state)
            except BaseException as exc:
                if error is None:
                    error = exc
        try:
            self.reader.close()
        except BaseException as exc:
            if error is None:
                error = exc
        if error is not None:
            raise error
