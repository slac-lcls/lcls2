"""Run-scoped callback iteration over a sequence of EventBuilder chunks.

This is the input/lifetime component for streaming callbacks. Production batch
routing still uses CallbackBatchBuilder; integration belongs to the next step.
"""

from array import array

from . import TransitionId
from .run import RunSmallData


class _OwnedProxyEvent:
    """Routing metadata backed by the owned dgrams returned by EB.events().

    Native ProxyEvent/PyDgram objects borrow pointers into the chunk and cannot
    survive refill. These records intentionally do not use the native proxy
    recycling API; a batch sink must serialize them through as_bytearray().
    """

    def __init__(self, dgrams, proxy):
        self._dgrams = dgrams
        self.service = proxy.service
        self.timestamp = proxy.timestamp
        self.destination = proxy.destination

    def set_destination(self, destination):
        self.destination = destination

    def as_bytearray(self):
        footer = array("I", [0] * (len(self._dgrams) + 1))
        footer[-1] = len(self._dgrams)
        result = bytearray()
        for index, dgram in enumerate(self._dgrams):
            if dgram is not None:
                data = memoryview(dgram)
                footer[index] = data.nbytes
                result.extend(data)
        result.extend(footer)
        return result


class _CallbackRun(RunSmallData):
    def __init__(self, events, configs, dsparms):
        self._stream_events = events
        super().__init__(None, configs, dsparms)

    def _iter_events(self):
        # Both run.events() and run.steps() share this single advancing stream.
        return self._stream_events


class CallbackIterator:
    """Invoke one callback for one run on the current executing rank.

    ``event_builders`` is a run-scoped iterable of native EventBuilders. Advancing
    it may fetch/refill and overwrite the preceding builder's input buffers.
    Chunk boundaries are invisible to the callback; EndStep still ends a Step.
    Callback events and accumulated transitions own their data across refills.

    Selected Events are yielded, while non-L1 records accumulate in
    ``run.proxy_events`` as in RunSmallData. This component does not route or
    bound that transition queue: the later batch integration must drain it
    during input consumption, including when the callback rejects all events.
    It must also drain/terminate the upstream MPI protocol after early return.
    Keeping this component unwired avoids changing those production semantics.

    An exhausted/returned callback is never restarted. Explicit close discards
    pending transitions and closes the owned input iterator; it does not drain
    further chunks. A new run must use a new CallbackIterator. State is local to
    a rank, not a global count across multiple event-builder ranks.
    """

    def __init__(self, event_builders, configs, dsparms, callback):
        self._builders = iter(event_builders)
        self._callback = callback
        self._callback_iter = None
        self._finished = False
        self._closed = False
        self._input_closed = False
        self.chunks_read = 0
        self._events = self._iter_events()
        self.run = _CallbackRun(self._events, configs, dsparms)

    def _iter_events(self):
        for builder in self._builders:
            self.chunks_read += 1
            while builder.has_more():
                for dgrams, proxy in builder.events():
                    # EB.events() already copies dgrams. Reuse those owned
                    # buffers rather than retain borrowed native proxy pointers.
                    owned_proxy = _OwnedProxyEvent(dgrams, proxy)
                    yield dgrams, owned_proxy
                    if owned_proxy.service == TransitionId.EndRun:
                        return

    def __iter__(self):
        return self

    def __next__(self):
        if self._finished:
            raise StopIteration
        try:
            if self._callback_iter is None:
                self._callback_iter = iter(self._callback(self.run))
            return next(self._callback_iter)
        except StopIteration:
            self._finished = True
            # Leave transition records available to the eventual batch sink.
            self._close_input()
            raise
        except BaseException:
            self.close()
            raise

    def _close_input(self):
        if self._input_closed:
            return
        self._input_closed = True
        try:
            self._events.close()
        finally:
            close = getattr(self._builders, "close", None)
            if close is not None:
                close()

    def close(self):
        if self._closed:
            return
        self._closed = True
        self._finished = True
        try:
            close = getattr(self._callback_iter, "close", None)
            if close is not None:
                close()
        finally:
            try:
                self._close_input()
            finally:
                self.run.proxy_events.clear()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
