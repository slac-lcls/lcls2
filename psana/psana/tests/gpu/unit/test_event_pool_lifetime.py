"""Unit coverage for EventPool reusable-slot lifetime."""

import sys
from types import SimpleNamespace

import pytest

from psana.gpu.gpu_stream import EventPool


class _FakeStream:
    def __init__(self, non_blocking=True):
        self.non_blocking = non_blocking
        self.synchronize_calls = 0

    def synchronize(self):
        self.synchronize_calls += 1


class _FakeDetector:
    def process_batch(self, *args, **kwargs):
        return iter(())


def test_slot_must_be_retired_before_reuse(monkeypatch):
    fake_cupy = SimpleNamespace(
        cuda=SimpleNamespace(Stream=_FakeStream),
    )
    monkeypatch.setitem(sys.modules, "cupy", fake_cupy)

    pool = EventPool(n=1)
    detectors = {"jungfrau": (None, _FakeDetector())}

    pool.submit(None, None, ["event-0"], detectors)
    with pytest.raises(RuntimeError, match="without retire_next"):
        pool.submit(None, None, ["event-1"], detectors)

    results, events = pool.retire_next()
    assert results == {}
    assert events == ["event-0"]
    assert pool._streams[0].synchronize_calls == 1

    # Reuse is valid only after the old result has been handed back.
    pool.submit(None, None, ["event-1"], detectors)
