"""Input windows outlive execution slots without permitting early reuse."""
from threading import Event, Thread
from types import SimpleNamespace as NS
import sys

import numpy as np
import pytest

from psana.gpu.gpu_input import GpuEventDgrams, InputSlotLease
from psana.gpu.gpu_input_window import InputWindow
from psana.gpu.gpu_kvikio_read import DESC_NCOLS
from psana.gpu.gpu_stream import EventPool


class Token:
    def __init__(self, **kwargs):
        self.waits = 0
        self.fail = False

    def record(self, stream):
        pass

    def synchronize(self):
        self.waits += 1
        if self.fail:
            raise RuntimeError('injected completion failure')


class Stream:
    def __init__(self, **kwargs):
        self.waited = []

    def wait_event(self, event):
        self.waited.append(event)

    def synchronize(self):
        pass


def window(stream=0, batch_id=7, release=None):
    table = np.zeros((1, DESC_NCOLS), dtype=np.uint64)
    table[0] = [1, stream, 100, 0, 4, 0]
    batch = NS(data_gpu=np.array([1, 2, 3, 4], dtype=np.uint8),
               stream_ids_by_dgram=np.array([stream]), n_dgrams=1,
               walk_done=Token())
    return InputWindow(batch_id, stream, batch, table, release=release)


def event_view():
    return NS(iter_events=lambda: iter((NS(batch_event_index=1, timestamp=100,
                                         first_desc=0, n_desc=2),)))


def test_configured_ready_is_required_without_locator_wrappers():
    configured = Token()
    configured.fail = True
    batch = NS(n_dgrams=0, walk_done=Token(), _locators={},
               _configured_backing=object(),
               configured_locations=lambda: NS(ready=configured))
    released = []
    owner = InputWindow(0, 0, batch, np.zeros((0, DESC_NCOLS), np.uint64),
                        release=lambda: released.append(True))
    stream = Stream()
    owner.wait_ready(stream)
    assert configured in stream.waited
    with pytest.raises(RuntimeError, match='completion failure'):
        owner.close()
    assert not released and not owner.released
    configured.fail = False
    assert owner.close() and released == [True]


def test_shared_locator_readiness_is_deduplicated_without_losing_consumers():
    owner = window()
    shared, fallback, consumer = Token(), Token(), Token()
    owner.batch.locate = lambda handle, **kw: NS(ready=shared if handle == 0 else fallback)
    owner.locate(0)
    owner.locate(0)
    owner.locate(1)
    use = owner.acquire()
    use.register_consumer_done(consumer)
    use.register_consumer_done(consumer)
    owner.close()
    use.wait_until_safe_to_reuse()
    assert shared.waits == fallback.waits == consumer.waits == 1


def test_fast_input_survives_repeated_execution_retirement(monkeypatch):
    monkeypatch.setitem(sys.modules, 'cupy', NS(cuda=NS(Stream=Stream, Event=Token)))
    releases = []
    fast = window(release=lambda: releases.append('fast'))
    planned = fast.acquire()
    early = Token()
    planned.register_consumer_done(early)
    pool = EventPool(n=1)
    for i in range(3):
        slow = window(1, release=lambda: releases.append('slow'))
        record = pool.submit(event_view(), None, [], {}, input_windows=(fast, slow), batch_id=7)
        assert record.input_dgrams_by_ts[100][0].owner is fast
        assert record.input_dgrams_by_ts[100][1].owner is slow
        assert pool._streams[0].waited
        slow.close()
        pool.begin_retire_next()
        pool.finish_retire_next()
        assert slow.released and not fast.released
        assert early.waits == 0
        np.testing.assert_array_equal(fast.batch.data_gpu, [1, 2, 3, 4])
    assert releases == ['slow'] * 3
    assert not fast.close()
    planned.wait_until_safe_to_reuse()
    assert fast.released and early.waits == 1
    assert releases[-1] == 'fast'


def test_composition_rejects_cross_batch_duplicate_and_missing_inputs():
    fast, slow = window(), window(1)
    views = GpuEventDgrams.from_windows(event_view(), (fast, slow), batch_id=7)
    assert views[0].batch is None
    assert views[0].batch_event_index == 1
    assert views[0][0].dgram_index == views[0][1].dgram_index == 0
    with pytest.raises(ValueError, match='different EB batches'):
        GpuEventDgrams.from_windows(event_view(), (fast, window(1, batch_id=8)), batch_id=7)
    with pytest.raises(ValueError, match='duplicate'):
        GpuEventDgrams.from_windows(event_view(), (fast, window()), batch_id=7)
    with pytest.raises(ValueError, match='do not cover'):
        GpuEventDgrams.from_windows(event_view(), (fast,), batch_id=7)


def test_field_reference_can_finish_after_event_lease_closes():
    released = []
    owner = window(release=lambda: released.append(True))
    event = InputSlotLease(Token(), (owner,))
    owner.close()
    view = event.acquire_view()  # splits an already reserved reference
    event.wait_until_safe_to_reuse()
    assert not owner.released
    with pytest.raises(RuntimeError, match='retiring or released'):
        event.acquire_view()
    done = Token()
    view.register_consumer_done(done)
    view.wait_until_safe_to_reuse()
    assert owner.released and done.waits == 1 and released == [True]


def test_failed_completion_keeps_storage_and_rejects_late_registration():
    released = []
    owner = window(release=lambda: released.append(True))
    use = owner.acquire()
    done = Token()
    done.fail = True
    use.register_consumer_done(done)
    owner.close()
    with pytest.raises(RuntimeError, match='completion failure'):
        use.wait_until_safe_to_reuse()
    assert not released and not owner.released
    with pytest.raises(RuntimeError, match='closed to new uses'):
        owner.acquire()
    with pytest.raises(RuntimeError, match='retiring or released'):
        use.register_consumer_done(Token())
    done.fail = False
    assert owner.close()
    assert owner.close()
    assert released == [True]


def test_retirement_blocks_new_references_while_cuda_completion_is_pending():
    waiting, complete = Event(), Event()
    released, errors = [], []

    class Gate:
        def synchronize(self):
            waiting.set()
            assert complete.wait(5)

    owner = window(release=lambda: released.append(True))
    use = owner.acquire()
    use.register_consumer_done(Gate())
    owner.close()

    def retire():
        try:
            use.wait_until_safe_to_reuse()
        except BaseException as exc:
            errors.append(exc)

    worker = Thread(target=retire)
    worker.start()
    try:
        assert waiting.wait(5)
        assert not released
        with pytest.raises(RuntimeError, match='closed to new uses'):
            owner.acquire()
        with pytest.raises(RuntimeError, match='retiring or released'):
            use.fork()
    finally:
        complete.set()
        worker.join(5)
    assert not worker.is_alive() and not errors
    assert released == [True]


def test_parser_failure_retains_raw_and_tables_until_completion_can_be_retried():
    from psana.gpu.gpudgram.batch import GpuXtcBatchPool
    pool = GpuXtcBatchPool.__new__(GpuXtcBatchPool)
    pool._owners = [None]
    pool._failed_inputs = []
    pool._next_window_id = 0
    released = []
    read = NS(data_gpu=None, desc_table=None,
              retain_input=lambda: lambda: released.append(True))
    stream = Token()
    stream.fail = True

    def fail(*args):
        raise ValueError('parser submission failed')

    pool.parse = fail
    with pytest.raises(RuntimeError, match='completion failure'):
        pool.parse_window(read, stream, batch_id=7)
    assert pool._owners == [stream] and not released
    with pytest.raises(RuntimeError, match='no free GPU input'):
        pool.parse_window(read, stream, batch_id=7)
    stream.fail = False
    pool.close()
    assert pool._owners == [None] and released == [True]


def test_failed_execution_keeps_its_input_reference_until_stream_drains(monkeypatch):
    monkeypatch.setitem(sys.modules, 'cupy', NS(cuda=NS(Stream=Stream, Event=Token)))
    fast, slow = window(), window(1)
    pool = EventPool(n=1)
    failed_stream = Token()
    failed_stream.wait_event = lambda event: None
    failed_stream.fail = True
    pool._streams[0] = failed_stream

    class Detector:
        def process_batch(self, *args, **kwargs):
            raise ValueError('detector submission failed')

    with pytest.raises(RuntimeError, match='completion failure'):
        pool.submit(event_view(), None, [], {'det': (None, Detector())},
                    input_windows=(fast, slow), batch_id=7)
    assert not fast.close() and not slow.close()
    assert pool._slots[0] is not None
    failed_stream.fail = False
    list(pool.flush())
    assert fast.released and slow.released


def test_partial_multi_owner_acquisition_returns_prior_reference(monkeypatch):
    released = []
    fast = window(release=lambda: released.append('fast'))
    slow = window(1, release=lambda: released.append('slow'))
    def fail():
        assert fast.references == 1
        raise RuntimeError('injected second owner acquisition failure')
    monkeypatch.setattr(slow, 'acquire', fail)
    with pytest.raises(RuntimeError, match='second owner acquisition failure'):
        InputSlotLease(Token(), (fast, slow))
    assert fast.references == slow.references == 0
    assert not fast.released and not slow.released
    assert fast.close() and slow.close()
    assert released == ['fast', 'slow']


def test_partial_multi_owner_view_fork_preserves_parent_and_returns_child(monkeypatch):
    fast, slow = window(), window(1)
    ready = Token()
    parent = InputSlotLease(ready, (fast, slow))
    assert not fast.close() and not slow.close()
    def fail():
        assert fast.references == 2
        raise RuntimeError('injected second owner fork failure')
    monkeypatch.setattr(parent._uses[1], 'fork', fail)
    with pytest.raises(RuntimeError, match='second owner fork failure'):
        parent.acquire_view()
    assert fast.references == slow.references == 1
    assert not fast.released and not slow.released and ready.waits == 0
    parent.require_active()
    parent.wait_until_safe_to_reuse()
    assert fast.released and slow.released
    assert fast.references == slow.references == 0
