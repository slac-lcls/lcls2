"""Check synchronous delivery and cleanup without loading MPI in pytest."""

from contextvars import ContextVar
from types import SimpleNamespace
import threading
import importlib.abc
import sys


class NoGreenlet(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "greenlet" or fullname.startswith("greenlet."):
            raise ImportError("This test must run without greenlet")


sys.meta_path.insert(0, NoGreenlet())

from psana.psexp import TransitionId
from psana.psexp.callback_stream import StreamingCallbackBatches


class Record:
    service = TransitionId.BeginStep
    destination = 0

    def __init__(self, size=100):
        self.size = size

    def as_bytearray(self):
        return bytearray(self.size)


class Source:
    def __init__(self, fail_close=False):
        self.closes = 0
        self.fail_close = fail_close

    def __iter__(self):
        return self

    def __next__(self):
        raise StopIteration

    def close(self):
        self.closes += 1
        if self.fail_close:
            raise RuntimeError("cleanup failure")


def stress(close_early=False, before_start=False, record_size=100):
    seen = 0
    calls = 0
    cleaned = False
    thread = threading.get_ident()
    context = ContextVar("callback_test_context")
    context.set("inherited")

    def callback(run):
        nonlocal seen, calls, cleaned
        calls += 1
        assert threading.get_ident() == thread
        assert context.get() == "inherited"
        try:
            for _ in range(10000):
                seen += 1
                run.proxy_events.append(Record(record_size))
            if False:
                yield
        finally:
            cleaned = True

    source = Source()
    parms = SimpleNamespace(batch_size=2, smd_callback=callback)
    stream = StreamingCallbackBatches(source, [], parms, max_records=5, max_bytes=320)
    delivered = 0
    if not before_start:
        for batch, steps in stream:
            packet, sizes = batch[0]
            assert steps[0] == batch[0]
            assert len(sizes) <= 5
            assert sum(sizes) <= 320 or len(sizes) == 1
            assert len(packet) == sum(sizes) + 4 * (len(sizes) + 1)
            delivered += len(sizes)
            # The synchronous callback finishes before its first transition
            # batch is delivered. Prompt bounded delivery is a deferred issue.
            assert seen == 10000
            assert cleaned
            if close_early:
                break
    stream.close()
    stream.close()
    assert source.closes == 1
    assert not stream._pending
    assert not stream._records
    assert list(stream) == []
    assert calls == (0 if before_start else 1)
    assert cleaned == (not before_start)
    if not close_early and not before_start:
        assert delivered == 10000


def failure():
    def callback(run):
        raise ValueError("callback failure")
        yield

    source = Source(fail_close=True)
    stream = StreamingCallbackBatches(source, [], SimpleNamespace(batch_size=1, smd_callback=callback))
    try:
        next(stream)
    except ValueError as exc:
        assert str(exc) == "callback failure"
    else:
        raise AssertionError("callback failure was lost")
    assert source.closes == 1
    assert not stream._pending
    assert not stream._records


if __name__ == "__main__":
    stress()
    stress(close_early=True)
    stress(before_start=True)
    stress(record_size=1000)
    failure()
    print("CALLBACK_STREAM_STRESS_OK")
