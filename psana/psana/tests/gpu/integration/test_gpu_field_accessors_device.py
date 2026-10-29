"""Copy-property compatibility on packed input, distinct from borrowed views."""
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from field_view_fixture import DTYPES, field_case, retire
from gpu_group_fixture import available

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(
    not available(), reason='no CUDA device')]


@pytest.mark.parametrize('accessor', ['on_cpu', 'on_gpu'])
@pytest.mark.parametrize('type_id', [1, 2, 9])
@pytest.mark.parametrize('shape', [(), (2, 3), (0, 3)])
@pytest.mark.parametrize('base_shift,offset', [(0, 8), (1, 8), (1, 7)])
def test_copy_accessor_exact_bits_and_retirement(accessor, type_id, shape, base_shift, offset):
    import cupy as cp
    case = field_case(cp, type_id=type_id, shapes=(shape,),
                      base_shift=base_shift, offset=offset)
    stream = cp.cuda.Stream(non_blocking=True)
    try:
        with stream:
            output = getattr(case.result, accessor)
        # This synchronization is caller ordering for the owned GPU copy.
        stream.synchronize()
        value = output.only()
        host = value if accessor == 'on_cpu' else value.get()
        assert host.dtype == np.dtype(DTYPES[type_id]) and host.shape == shape
        assert host.tobytes() == case.expected[9].tobytes()
        if accessor == 'on_gpu' and value.size:
            assert value.data.ptr % value.dtype.itemsize == 0
            assert value.data.ptr != case.sources[0].data_gpu.data.ptr + offset
        retire(case)
        assert case.owner.released and case.budget.committed() == 0
        # Simulate input reuse after retirement; independent outputs survive.
        with stream:
            case.sources[0].data_gpu.fill(0)
        stream.synchronize()
        host = value if accessor == 'on_cpu' else value.get()
        assert host.tobytes() == case.expected[9].tobytes()
        if accessor == 'on_cpu':
            assert case.result.on_cpu is output
        else:
            with pytest.raises(RuntimeError, match='retiring'):
                case.result.on_gpu
    finally:
        stream.synchronize()
        retire(case)


@pytest.mark.parametrize('offset', [1, 8])
def test_gpu_copies_are_independent_and_outside_field_budget(offset):
    import cupy as cp
    case = field_case(cp, type_id=3, shapes=((2, 3), (), (5,)),
                      offset=offset, absent=(17,))
    stream = cp.cuda.Stream(non_blocking=True)
    try:
        with stream:
            first, second = case.result.on_gpu, case.result.on_gpu
            first[9].fill(0)
            first[4].fill(0)
        stream.synchronize()
        assert first.segment_ids == second.segment_ids == (9, 4)
        for segment in second:
            assert first[segment].data.ptr != second[segment].data.ptr
            assert second[segment].get().tobytes() == case.expected[segment].tobytes()
        with case.result.on_gpu_view(stream) as values:
            for segment in values:
                # Already synchronized above; no implicit cross-stream claim.
                assert values[segment].get().tobytes() == case.expected[segment].tobytes()
        del values
        for index, segment in enumerate((9, 4)):
            assert case.sources[index].data_gpu[offset:].get().tobytes() == case.expected[segment].tobytes()
        assert case.budget.committed() == (1024 if offset == 1 else 0)
        retire(case)
        assert case.budget.committed() == 0
        assert all(second[s].get().tobytes() == case.expected[s].tobytes() for s in second)
    finally:
        stream.synchronize()
        retire(case)


@pytest.mark.parametrize('accessor', ['on_cpu', 'on_gpu'])
@pytest.mark.parametrize('offset', [1, 8])
def test_copy_accessor_waits_for_producer_on_current_stream(accessor, offset):
    import cupy as cp
    case = field_case(cp, type_id=3, shapes=((257,),), offset=offset)
    producer, consumer = [cp.cuda.Stream(non_blocking=True) for _ in range(2)]
    write = cp.RawKernel(r'''
    extern "C" __global__ void write(unsigned char* data, int offset, int n,
                                    unsigned long long ticks) {
        unsigned long long start=clock64(); while(clock64()-start<ticks) {}
        for (int i=offset; i<n; ++i) data[i]=37;
    }''', 'write')
    write.compile()
    source = case.sources[0].data_gpu
    try:
        producer.wait_event(case.lease.result_ready)
        write((1,), (1,), (source, np.int32(offset), np.int32(source.size),
                          np.uint64(200000000)), stream=producer)
        producer.record(case.lease.result_ready)
        with consumer:
            output = getattr(case.result, accessor)
        if accessor == 'on_cpu':
            # No explicit host wait before checking the synchronous CPU result.
            assert output.only().tobytes() == bytes([37]) * (257 * 8)
        else:
            consumer.synchronize()
            assert output.only().get().tobytes() == bytes([37]) * (257 * 8)
    finally:
        producer.synchronize()
        consumer.synchronize()
        retire(case)
