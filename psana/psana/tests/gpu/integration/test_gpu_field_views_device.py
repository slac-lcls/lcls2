"""Actual-address field reads and asynchronous copy/cache ownership on CUDA."""
from pathlib import Path
import sys
import weakref

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from field_view_fixture import DTYPES, field_case, retire
from gpu_group_fixture import available

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(
    not available(), reason='no CUDA device')]


@pytest.mark.parametrize('type_id', range(10))
@pytest.mark.parametrize('shape', [(), (3, 7), (0,)])
@pytest.mark.parametrize('base_shift,offset', [(0, i) for i in range(8)] + [(1, 8), (1, 7)])
def test_device_field_bits_alignment_and_cache(type_id, shape, base_shift, offset):
    import cupy as cp
    case = field_case(cp, type_id=type_id, shapes=(shape,),
                      base_shift=base_shift, offset=offset)
    stream = cp.cuda.Stream(non_blocking=True)
    dtype = np.dtype(DTYPES[type_id])
    src_ptr = case.sources[0].data_gpu.data.ptr + offset
    aligned = src_ptr % dtype.itemsize == 0
    output = cp.empty(shape, dtype=dtype)
    ctype = {1: 'unsigned char', 2: 'unsigned short', 4: 'unsigned int',
             8: 'unsigned long long'}[dtype.itemsize]
    # A typed load under memcheck, including signed/floating object bits.
    kernel = cp.RawKernel('extern "C" __global__ void load(const '+ctype+
        '* src, '+ctype+'* dst, int n) { int i=threadIdx.x; if(i<n) dst[i]=src[i]; }', 'load')
    kernel.compile()
    try:
        with case.result.on_gpu_view(stream) as values:
            value = values.only()
            if value.size:
                assert value.data.ptr % dtype.itemsize == 0
                assert (value.data.ptr == src_ptr) == aligned
            kernel((1,), (32,), (value, output, np.int32(value.size)), stream=stream)
            ptr = value.data.ptr
        charge = case.budget.committed()
        with case.detector.field('raw', 'value', segment=9).on_gpu_view(stream) as cached:
            assert cached.only().data.ptr == ptr
        assert case.budget.committed() == charge
        assert charge == (0 if aligned else ((value.nbytes + 511) // 512) * 512)
        stream.synchronize()
        assert output.get().tobytes() == case.expected[9].tobytes()
        assert value.shape == shape and value.dtype == dtype
        del value, values, cached
        retire(case)
        assert case.budget.committed() == 0
    finally:
        stream.synchronize()
        retire(case)


def test_delayed_materialization_second_stream_and_open_child_retirement():
    import cupy as cp
    case = field_case(cp, type_id=3, shapes=((257,),), offset=1)
    producer, first, second = [cp.cuda.Stream(non_blocking=True) for _ in range(3)]
    outputs = [cp.empty(257, cp.uint64) for _ in range(2)]
    done = [cp.cuda.Event(disable_timing=True) for _ in outputs]
    delay = cp.RawKernel(r'''
    extern "C" __global__ void delay(unsigned long long ticks,
        unsigned char* data, int n, unsigned char value) {
        unsigned long long start=clock64(); while(clock64()-start<ticks) {}
        for (int i=1; i<n; ++i) data[i]=value;
    }''', 'delay')
    delay.compile()
    source = case.sources[0].data_gpu
    # Cold cudaMalloc / first kernel loading may synchronize all streams.
    # Warm those before the delay so this test isolates copy-ready ordering.
    # Fill inside the raw kernel: CuPy scalar assignment can block the host.
    warm_copy = cp.empty(257, cp.uint64)
    delay((1,), (1,), (np.uint64(0), source, np.int32(source.size), np.uint8(0)),
          stream=producer)
    cp.cuda.get_current_stream().synchronize()
    producer.synchronize()
    del warm_copy
    producer.wait_event(case.lease.result_ready)
    # Changing source after a delayed producer proves entry waits for readiness.
    delay((1,), (1,), (np.uint64(2000000000), source, np.int32(source.size), np.uint8(37)),
          stream=producer)
    producer.record(case.lease.result_ready)
    contexts = [case.result.on_gpu_view(first), case.result.on_gpu_view(second)]
    try:
        values = contexts[0].__enter__()
        saved = weakref.ref(values.only())
        assert not case.result._views_ready[0].done
        with first:
            cp.copyto(outputs[0], values.only())
        first.record(done[0])
        cached = contexts[1].__enter__()
        assert cached.only().data.ptr == values.only().data.ptr
        with second:
            cp.copyto(outputs[1], cached.only())
        second.record(done[1])
        del values, cached
        case.owner.close()
        case.lease.wait_until_safe_to_reuse()
        assert case.result._views is None and case.lease._field_copies is None
        assert saved() is not None and case.budget.committed() == 2560
        assert not case.owner.released
        contexts[0].__exit__(None, None, None)
        assert saved() is not None and case.budget.committed() == 2560
        contexts[1].__exit__(None, None, None)
        assert saved() is None and case.budget.committed() == 0
        assert all(event.done for event in done)
        case.owner.drain()
        assert case.owner.released
        for output in outputs:
            assert output.get().tobytes() == bytes([37]) * (257 * 8)
    finally:
        for stream in (producer, first, second):
            stream.synchronize()
        for context in contexts:
            context.__exit__(None, None, None)
        retire(case)


def test_large_ragged_absent_and_scalar_copies_share_budget():
    import cupy as cp
    case = field_case(cp, type_id=9, shapes=((512, 1024), (), (3,)), absent=(17,))
    try:
        with case.result.on_gpu_view() as values:
            assert values.segment_ids == (9, 4)
            assert values[9].shape == (512, 1024) and values[4].shape == ()
        cp.cuda.Stream.null.synchronize()
        assert all(values[s].get().tobytes() == case.expected[s].tobytes() for s in values)
        assert case.budget.committed() == 512 * 1024 * 8 + 512
        assert len(case.budget.allocation_snapshot()) == 2
        del values
        retire(case)
        assert case.budget.committed() == 0
    finally:
        cp.cuda.Stream.null.synchronize()
        retire(case)
