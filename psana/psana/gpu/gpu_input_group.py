"""Bounded independent input owners, opt-in until Stage 3 scheduling.

The pool owns a dedicated KvikioGpuReader's slots. A small stream keeps its
credit until its group's last planned/active/CUDA consumer finishes, including
across EB batches. Only unreferenced windows enter the retirement queue.
"""
from bisect import bisect_left
from dataclasses import dataclass, field
from queue import Empty, SimpleQueue

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
        self._busy_slots = set()
        self._retirement_queue = SimpleQueue()
        self._candidates = {}
        self._closed = False

    @property
    def live_keys(self):
        return tuple(self._groups)

    def plan_slots(self, groups):
        """Choose distinct available slots before a combined allocation hold."""
        self.poll()
        busy = set(self._busy_slots)
        small = set(self._small)
        # Snapshot availability/capacity once per plan. Sorting by slot as the
        # secondary key preserves the old lowest-ID tie break. Keep the index
        # local: growth, trimming and completion are reflected on the next call.
        free = [i for i in range(self.reader._n_slots) if i not in busy]
        capacities = {i: self.reader._slot_bufs[i].nbytes for i in free
                      if self.reader._slot_bufs[i] is not None}
        cached = sorted((capacity, i) for i, capacity in capacities.items())
        first_free = 0
        result = []
        for group in groups:
            if group.small and group.stream_id in small:
                return None
            if len(result) == len(free):
                return None
            fit = bisect_left(cached, (group.size, -1))
            if fit < len(cached):
                _, slot = cached.pop(fit)
            else:
                # No cached buffer fits: grow the lowest free slot, even if
                # another slot has a larger (but still undersized) allocation.
                while free[first_free] in busy:
                    first_free += 1
                slot = free[first_free]
                if slot in capacities:
                    cached.pop(bisect_left(cached, (capacities[slot], slot)))
            result.append(slot)
            busy.add(slot)
            if group.small:
                small.add(group.stream_id)
        return tuple(result)

    def issue(self, batch_id, group, *, slot_id=None):
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
        if slot_id is None:
            slots = self.plan_slots((group,))
            if slots is None:
                return None
            slot = slots[0]
        else:
            slot = int(slot_id)
        if not 0 <= slot < self.reader._n_slots:
            raise IndexError(slot)
        if slot in self._busy_slots or (group.small and group.stream_id in self._small):
            return None
        return self._issue_reserved(key, group, slot)

    def issue_groups(self, batch_id, groups, *, slots):
        """Submit a complete slot plan without per-group polling or pool scans.

        plan_slots() collects retirement candidates before the allocation hold.
        Validate all keys/credits/slots before any I/O. A partial I/O failure
        leaves successful submissions owned by this pool for close() to drain.
        """
        if self._closed:
            raise RuntimeError('input group pool is closed')
        groups, slots = tuple(groups), tuple(map(int, slots))
        if len(groups) != len(slots):
            raise ValueError('input groups and slots differ in length')
        keys = tuple((int(batch_id), g.group_id) for g in groups)
        if len(set(keys)) != len(keys) or any(k in self._groups for k in keys):
            raise ValueError('duplicate live input group')
        if any(not 0 <= s < self.reader._n_slots for s in slots):
            raise IndexError('input slot outside reader pool')
        small = [g.stream_id for g in groups if g.small]
        if (len(set(slots)) != len(slots) or self._busy_slots.intersection(slots)
                or len(set(small)) != len(small) or any(s in self._small for s in small)):
            return None
        for key, group, slot in zip(keys, groups, slots):
            self._issue_reserved(key, group, slot)
        return keys

    def _issue_reserved(self, key, group, slot):
        state = _GroupState(key, group, slot)
        self._groups[key] = state
        self._busy_slots.add(slot)
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
        a parser launch per read request. Bind only after notification registration
        succeeds, so every bound window has a path into the candidate queue.
        """
        state = self._groups[key]
        if state.window is not None:
            raise RuntimeError('input group is already bound')
        read = self.read(key)
        if (window.batch_id != key[0] or not window._defer_retirement
                or window.batch.data_gpu is not read.data_gpu
                or not np.array_equal(window.desc_table, read.desc_table)):
            raise ValueError('window does not match this group or lacks deferred retirement')
        # Notification may come from a consumer thread. It only enqueues; the
        # BD thread alone mutates pool state and releases raw-slot ownership.
        queue = self._retirement_queue
        window.notify_when_retirable(lambda: queue.put(state))
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

    def window(self, key):
        window = self._groups[key].window
        if window is None:
            raise RuntimeError('input group is not parsed')
        return window

    def parse_groups(self, keys, parser, stream):
        keys = tuple(keys)
        if not keys:
            return ()
        if len({key[0] for key in keys}) != 1:
            raise ValueError('cannot parse groups from different EB batches together')
        windows = parser.parse_groups([self.read(key) for key in keys], stream,
                                      batch_id=keys[0][0])
        try:
            for key, window in zip(keys, windows):
                self.bind(key, window)
        finally:
            # Unbound windows still belong to the parser pool if setup fails;
            # close root acquisitions so parser.close can drain them.
            for window in windows:
                window.close()
        return windows

    def _collect_candidates(self):
        while True:
            try:
                state = self._retirement_queue.get_nowait()
            except Empty:
                break
            # Identity check also rejects a stale notification after key reuse.
            if self._groups.get(state.key) is state:
                self._candidates[state.key] = state

    def poll(self):
        """Reclaim eligible windows only; never synchronize or scan live uses."""
        self._collect_candidates()
        reclaimed, error = [], None
        for state in tuple(self._candidates.values()):
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

    def drain_idle(self):
        """Pressure/shutdown fallback after execution delivery has drained.

        Planned/live consumers stay protected. Normal collection uses poll().
        """
        for state in tuple(self._groups.values()):
            if state.window is not None and not state.window.references:
                state.window.drain()
        return self.poll()

    def _forget(self, state):
        if state.release_raw is not None:
            state.release_raw()
            state.release_raw = None
        self._groups.pop(state.key, None)
        self._busy_slots.discard(state.slot)
        self._candidates.pop(state.key, None)
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
        self._collect_candidates()  # discard stale notifications from shutdown
        try:
            self.reader.close()
        except BaseException as exc:
            if error is None:
                error = exc
        if error is not None:
            raise error
