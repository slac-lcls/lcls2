"""Real CUDA consumers retain field-copy storage and its allocation charge."""
from types import SimpleNamespace as NS
import weakref

import numpy as np
import pytest

from psana.gpu.gpu_allocation import owned_empty
from psana.gpu.gpu_budget import _GpuBudget
from psana.gpu.gpu_input import GpuFieldResult, InputSlotLease, _GpuFieldViewContext
from psana.gpu.gpu_input_window import InputWindow
from psana.gpu.gpu_kvikio_read import DESC_NCOLS
from gpu_group_fixture import available

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(
    not available(), reason='no CUDA device')]


@pytest.mark.parametrize('child_before_copy', [False, True])
def test_retired_cache_keeps_copy_for_two_delayed_streams(monkeypatch, child_before_copy):
    import cupy as cp

    budget = _GpuBudget(4096)
    source = cp.arange(256, dtype=cp.uint16)
    source_ready = cp.cuda.Event(disable_timing=True)
    source_ready.record()
    owner = InputWindow(0, 0, NS(n_dgrams=0, walk_done=source_ready, data_gpu=source),
                        np.zeros((0, DESC_NCOLS), np.uint64), defer_retirement=True)
    parent = InputSlotLease(source_ready, (owner,))
    result = GpuFieldResult(NS(segment_ids=(0,)), NS(), parent)
    streams = [cp.cuda.Stream(non_blocking=True) for _ in range(2)]
    consumers_done = [cp.cuda.Event(disable_timing=True) for _ in streams]
    outputs = [cp.empty_like(source) for _ in streams]
    copy_stream = cp.cuda.Stream(non_blocking=True)
    copy_ready = cp.cuda.Event(disable_timing=True)
    kernel = cp.RawKernel(r'''
    extern "C" __global__ void delayed_copy(const unsigned short* src,
        unsigned short* dst, int n, unsigned long long ticks) {
        unsigned long long start = clock64();
        while (clock64() - start < ticks) {}
        for (int i = 0; i < n; ++i) dst[i] = src[i];
    }
    ''', 'delayed_copy')
    kernel.compile()
    contexts = []
    if child_before_copy:
        early = parent.acquire_view()
    copied = owned_empty(cp, source.shape, source.dtype, budget, 'field-copy')
    saved = weakref.ref(copied)
    parent.retain_copy(copied)
    copy_stream.wait_event(source_ready)
    kernel((1,), (1,), (source, copied, np.int32(source.size), np.uint64(0)), stream=copy_stream)
    copy_ready.record(copy_stream)
    parent.register_consumer_done(copy_ready)
    result._views = {0: copied}
    monkeypatch.setattr(GpuFieldResult, '_slot_views', lambda self, stream: self._views)
    try:
        for stream, output, consumer_done in zip(streams, outputs, consumers_done):
            context = _GpuFieldViewContext(result, stream)
            values = context.__enter__()
            contexts.append(context)
            stream.wait_event(copy_ready)
            kernel((1,), (1,), (values[0], output, np.int32(source.size),
                               np.uint64(2000000000)), stream=stream)
            consumer_done.record(stream)
        del values, copied
        owner.close()
        parent.wait_until_safe_to_reuse()
        assert not all(event.done for event in consumers_done)
        assert result._views is None and saved() is not None
        assert budget.committed() == 512 and not owner.released
        contexts[0].__exit__(None, None, None)
        assert saved() is not None and budget.committed() == 512
        contexts[1].__exit__(None, None, None)
        if child_before_copy:
            assert saved() is not None and budget.committed() == 512
            early.wait_until_safe_to_reuse()
        assert saved() is None and budget.committed() == 0
        assert owner.poll()
        for output in outputs:
            np.testing.assert_array_equal(output.get(), np.arange(256, dtype=np.uint16))
    finally:
        copy_stream.synchronize()
        for stream in streams:
            stream.synchronize()
        for context in contexts:
            context.__exit__(None, None, None)
        if child_before_copy:
            early.wait_until_safe_to_reuse()
        parent.wait_until_safe_to_reuse()
        owner.drain()


def test_pending_copy_keeps_input_and_charge_until_producer_finishes():
    import cupy as cp

    budget = _GpuBudget(4096)
    source = cp.arange(256, dtype=cp.uint16)
    cp.cuda.get_current_stream().synchronize()
    stream = cp.cuda.Stream(non_blocking=True)
    done = cp.cuda.Event(disable_timing=True)
    released = []

    def release():
        assert done.done
        released.append(True)

    owner = InputWindow(0, 0, NS(n_dgrams=0, data_gpu=source),
                        np.zeros((0, DESC_NCOLS), np.uint64), release=release)
    parent = InputSlotLease(None, (owner,))
    copied = owned_empty(cp, source.shape, source.dtype, budget, 'field-copy')
    saved = weakref.ref(copied)
    parent.retain_copy(copied)
    with stream:
        cp.copyto(copied, source)
    done.record(stream)
    parent.register_consumer_done(done)
    del copied
    owner.close()
    assert not released and saved() is not None and budget.committed() == 512
    parent.wait_until_safe_to_reuse()
    assert released == [True] and saved() is None and budget.committed() == 0
