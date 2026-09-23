"""Explicit ownership of parsed GPU inputs, independent of execution slots."""

from threading import RLock

import numpy as np

from .gpu_kvikio_read import DESC_EVENT_INDEX, DESC_STREAM_ID, DESC_TIMESTAMP, DESC_NCOLS


class InputWindow:
    """Own raw bytes and parser tables until planned uses and CUDA work finish.

    ``close`` stops new acquisitions. Existing uses may finish registering
    their consumers; retirement starts only after every use has been released.
    The release callback returns backing storage to its input pools, not to an
    execution slot. There is deliberately no garbage-collection release path.
    """

    def __init__(self, batch_id, window_id, batch, desc_table, *, release=None):
        self.batch_id = int(batch_id)
        self.window_id = int(window_id)
        self.batch = batch
        table = np.asarray(desc_table)
        if table.dtype != np.uint64 or table.ndim != 2 or table.shape[1] != DESC_NCOLS:
            raise ValueError("input descriptors must be a dense uint64 descriptor table")
        if len(table) != batch.n_dgrams:
            raise ValueError("input descriptors do not match parsed dgram count")
        self.desc_table = table.copy()
        self.desc_table.flags.writeable = False
        self.rows_by_event = {}
        for index, row in enumerate(self.desc_table):
            event = int(row[DESC_EVENT_INDEX])
            stream = int(row[DESC_STREAM_ID])
            rows = self.rows_by_event.setdefault(event, {})
            if stream in rows:
                raise ValueError(f"duplicate input event/stream: {(event, stream)}")
            rows[stream] = (index, int(row[DESC_TIMESTAMP]))
        self._lock = RLock()
        self._uses = set()
        self._closed = False
        self._retiring = False
        self._released = False
        self._release = release
        self._ready = [getattr(batch, 'walk_done', None)]
        if getattr(batch, '_configured_backing', None) is not None:
            self._ready.append(batch.configured_locations().ready)
        self._ready.extend(loc.ready for loc in getattr(batch, '_locators', {}).values())

    @property
    def released(self):
        with self._lock:
            return self._released

    @property
    def references(self):
        with self._lock:
            return len(self._uses)

    def require_storage(self):
        with self._lock:
            if self._retiring or self._released:
                raise RuntimeError("input window is retiring or released")

    def acquire(self):
        """Reserve an execution, event consumer, or planned future use."""
        with self._lock:
            if self._closed:
                raise RuntimeError("input window is closed to new uses")
            use = InputWindowUse(self)
            self._uses.add(use)
            return use

    def wait_ready(self, stream):
        with self._lock:
            self.require_storage()
            for event in self._ready:
                if event is not None:
                    stream.wait_event(event)

    def locate(self, handle, *, stream=None):
        with self._lock:
            self.require_storage()
            result = self.batch.locate(handle, stream=stream)
            # A lazily requested locator is itself a consumer of raw/parser
            # storage, even before its caller registers a kernel completion.
            self._ready.append(result.ready)
            return result

    def close(self):
        """Stop new uses; return True once all backing storage is reusable."""
        with self._lock:
            self._closed = True
        return self._try_retire()

    def _try_retire(self):
        with self._lock:
            if self._released:
                return True
            if not self._closed or self._uses or self._retiring:
                return False
            self._retiring = True
            ready = tuple(self._ready)
        try:
            for event in ready:
                if event is not None:
                    event.synchronize()
            if self._release is not None:
                self._release()
        except BaseException:
            with self._lock:
                self._retiring = False  # closed, but completion can be retried
            raise
        with self._lock:
            retire = getattr(self.batch, 'retire', None)
            if retire is not None:
                retire()
            self.batch = None
            self._ready.clear()
            self._released = True
            self._retiring = False
            self._release = None
        return True


class InputWindowUse:
    """One planned/live reference, with its own terminal CUDA consumers."""

    def __init__(self, window):
        self.window = window
        self._done = []
        self._closing = False
        self._released = False

    def fork(self):
        """Split an already reserved use, including after root acquisition closes."""
        with self.window._lock:
            if self._closing or self._released or self.window._retiring:
                raise RuntimeError("input use is retiring or released")
            child = InputWindowUse(self.window)
            self.window._uses.add(child)
            return child

    def register_consumer_done(self, event):
        with self.window._lock:
            if self._closing or self._released:
                raise RuntimeError("input use is retiring or released")
            self._done.append(event)

    def wait_until_safe_to_reuse(self):
        # Transfer completion dependencies to the owner. Releasing one slow
        # execution does not wait for all other uses of a resident fast input.
        with self.window._lock:
            if not self._released:
                self._closing = True
                self.window._ready.extend(self._done)
                self._done.clear()
                self.window._uses.remove(self)
                self._released = True
        return self.window._try_retire()
