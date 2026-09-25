"""Real CuPy memory-backing lifetime and reservation checks."""
import numpy as np
import pytest

from psana.gpu.gpu_allocation import owned_empty, allocation_capacity, allocation_requirement
from psana.gpu.gpu_budget import _GpuBudget, GpuMemoryPressureError


def available():
    try:
        import cupy as cp
        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


pytestmark = [pytest.mark.gpu, pytest.mark.skipif(not available(), reason='no CUDA device')]


def test_cupy_views_memory_pointer_and_dlpack_keep_the_charge():
    import cupy as cp
    baseline = cp.get_default_memory_pool().used_bytes()
    budget = _GpuBudget(4096)
    root = owned_empty(cp, 130, cp.uint32, budget, 'input')
    root.fill(7)
    cp.cuda.Stream.null.synchronize()
    assert budget.committed() == 1024
    sliced = root[2:18].reshape(4, 4)
    pointer_view = cp.ndarray((16,), dtype=cp.uint32, memptr=sliced.data)
    exported = cp.from_dlpack(pointer_view)
    del root, sliced, pointer_view
    assert budget.committed() == 1024
    assert cp.get_default_memory_pool().used_bytes()-baseline == 1024
    np.testing.assert_array_equal(cp.asnumpy(exported), np.full(16, 7, np.uint32))
    del exported
    assert budget.committed() == 0
    assert budget.allocation_snapshot() == ()
    assert cp.get_default_memory_pool().used_bytes() == baseline


def test_same_rounded_capacity_growth_still_requires_full_replacement():
    import cupy as cp
    budget = _GpuBudget(1024)
    old = owned_empty(cp, 10, cp.uint8, budget, 'parser')
    view = old[:]
    need, previous = allocation_requirement(cp, 20, old)
    assert (need, previous) == (512, 0)
    hold = budget.hold(need)
    with hold:
        replacement = owned_empty(cp, 20, cp.uint8, budget, 'parser')
        del old
        assert budget.committed() == 1024 and hold.remaining == 0
    hold.close()
    with pytest.raises(GpuMemoryPressureError):
        owned_empty(cp, 1, cp.uint8, budget, 'parser')
    del view
    assert budget.committed() == 512
    del replacement
    assert budget.committed() == 0


def test_wrapper_failure_releases_backing_and_restores_original_hold(monkeypatch):
    import cupy as cp
    baseline = cp.get_default_memory_pool().used_bytes()
    budget = _GpuBudget(2048)
    hold = budget.hold(1024)
    def fail(*args, **kwargs):
        raise RuntimeError('injected wrapper failure')
    monkeypatch.setattr(cp.cuda, 'UnownedMemory', fail)
    with hold, pytest.raises(RuntimeError, match='injected wrapper failure'):
        owned_empty(cp, 520, cp.uint8, budget, 'input')
    assert hold.remaining == budget._held == 1024 and budget.committed() == 0
    assert cp.get_default_memory_pool().used_bytes() == baseline
    hold.close()


def test_unsupported_allocator_is_rejected_before_allocating():
    import cupy as cp
    budget = _GpuBudget(1024)
    other = cp.cuda.MemoryPool()
    with cp.cuda.using_allocator(other.malloc):
        with pytest.raises(ValueError, match='default CuPy'):
            owned_empty(cp, 1, cp.uint8, budget, 'input')
    assert budget.committed() == 0


def test_device_zero_size_allocation():
    import cupy as cp
    budget = _GpuBudget(0)
    arr = owned_empty(cp, (0, 10), cp.uint64, budget, 'parser')
    assert arr.shape == (0, 10) and budget.committed() == 0
    del arr
    assert budget.allocation_snapshot() == ()


def test_reader_and_parser_charges_survive_retirement_and_trim(tmp_path):
    import cupy as cp
    from pathlib import Path
    import struct
    from types import SimpleNamespace as NS
    from psana import dgram
    from psana.gpu.gpu_batch import GpuReadDesc
    from psana.gpu.gpu_kvikio_read import KvikioGpuReader
    from psana.gpu.gpudgram.batch import GpuXtcBatchPool
    from psana.gpu.gpudgram.config import GpuStreamConfigTable

    data = (Path(__file__).resolve().parents[2]/'test_data/chunking/xpptut15-r0014-s000-c000.xtc2').read_bytes()
    config = dgram.Dgram(view=memoryview(data), offset=0)
    offset = config._size
    while True:
        size = 12 + struct.unpack_from('<I', data, offset+20)[0]
        if (struct.unpack_from('<I', data, offset+8)[0] >> 24) & 15 == 12:
            break
        offset += size
    timestamp = struct.unpack_from('<Q', data, offset)[0]
    path = tmp_path/'retention.xtc2'
    path.write_bytes(data)
    dm = NS(xtc_files=[path], get_chunk_id=lambda _: 0)
    descs = [GpuReadDesc(0, timestamp, 0, offset, size, 0, 1)]
    view = NS(iter_read_descs=lambda _: iter(descs))
    configs = GpuStreamConfigTable.from_configs([config])
    handle = configs.resolve('xppcspad', 1, 'raw', 'arrayRaw', stream_id=0)
    budget = _GpuBudget(16*1024**2)
    reader = KvikioGpuReader(n_slots=1, budget=budget, bulk_read=False)
    parser = GpuXtcBatchPool(configs, field_handles=[handle], n_slots=1, budget=budget)
    fixed = budget.committed()
    stream = cp.cuda.Stream(non_blocking=True)
    def parse():
        read = reader.wait_batch(reader.issue_batch(view, dm))
        return parser.parse_window(read, stream, batch_id=1)
    window = parse()
    stream.synchronize()
    raw_alias = window.batch.data_gpu[1:3]
    locator_alias = window.batch.locate(handle).rows_gpu[:]
    raw_capacity = int(raw_alias.data.mem.size)
    locator_capacity = int(locator_alias.data.mem.size)
    full_charge = budget.committed()
    assert full_charge == sum(r['capacity'] for r in budget.allocation_snapshot())
    assert window.close()
    reader.trim_free_buffers()
    parser.trim_free_buffers()
    assert reader.memory_bytes()['raw_input_slots'] == parser.memory_bytes()['batch_slots'] == 0
    assert budget.committed() == fixed+raw_capacity+locator_capacity < full_charge
    assert window.batch is None
    # The retired batch/slot dies; escaped aliases still own their two blocks.
    del window
    assert budget.committed() == fixed+raw_capacity+locator_capacity
    del raw_alias
    assert budget.committed() == fixed+locator_capacity
    del locator_alias
    assert budget.committed() == fixed
    assert all(a['category'] == 'fixed' for a in budget.allocation_snapshot())
    parser.close()
    reader.close()


def test_result_retirement_joins_two_streams_and_releases_only_unaliased_storage():
    import cupy as cp
    from psana.gpu.context import GPUResult, SlotLease
    budget = _GpuBudget(4096)
    array = owned_empty(cp, 64, cp.uint32, budget, 'detector')
    array.fill(7)
    ready = cp.cuda.Event()
    ready.record()
    lease = SlotLease(ready)
    result = GPUResult(array, lease)
    slow, fast = cp.cuda.Stream(non_blocking=True), cp.cuda.Stream(non_blocking=True)
    outputs = [cp.empty_like(array), cp.empty_like(array)]
    kernel = cp.RawKernel(r'''
    extern "C" __global__ void delayed_copy(const unsigned int* src, unsigned int* dst,
                                            unsigned long long delay) {
        unsigned long long start = clock64();
        while (clock64() - start < delay) {}
        dst[threadIdx.x] = src[threadIdx.x];
    }''', 'delayed_copy')
    with result.on_gpu_view(slow) as escaped:
        kernel((1,), (64,), (escaped, outputs[0], np.uint64(20000000)), stream=slow)
    with result.on_gpu_view(fast) as second:
        kernel((1,), (64,), (second, outputs[1], np.uint64(1)), stream=fast)
    del array, second
    done = tuple(lease._consumer_done)
    lease.wait_until_safe_to_reuse()
    assert len(done) == 2 and all(e.done for e in done)
    for output in outputs:
        np.testing.assert_array_equal(cp.asnumpy(output), np.full(64, 7, np.uint32))
    assert result._arr is None and budget.committed() == 512
    with pytest.raises(RuntimeError, match='released'):
        result.on_gpu
    del escaped
    assert budget.committed() == 0


def test_fixed_uploads_and_output_growth_keep_all_live_generations_charged():
    import cupy as cp
    from psana.gpu.gpu_allocation import upload_owned
    from psana.gpu.gpu_detector import GPUDetector
    budget = _GpuBudget(4096)
    fixed, = upload_owned(cp, [np.arange(16, dtype=np.float32)], budget)
    fixed_alias = fixed[1:3]
    del fixed
    detector = GPUDetector.__new__(GPUDetector)
    detector._budget = budget
    buffers = [None]
    old = detector._slot_buffer(buffers, 0, (100,), np.float32, 'calib')
    new = detector._slot_buffer(buffers, 0, (200,), np.float32, 'calib')
    assert budget.committed() == 512 + 512 + 1024
    buffers[0] = None
    del new
    assert budget.committed() == 1024
    del old
    assert budget.committed() == 512
    del fixed_alias
    assert budget.committed() == 0


def test_repeated_growth_reuse_and_pressure_with_retained_generations(monkeypatch):
    import cupy as cp
    from psana.gpu.gpu_detector import GPUDetector
    baseline = cp.get_default_memory_pool().used_bytes()
    budget = _GpuBudget(512 * 16)
    detector = GPUDetector.__new__(GPUDetector)
    detector._budget = budget
    buffers, aliases = [None], []
    capacities = []
    for count in (1, 4, 2, 8):
        previous = buffers[0]
        previous_ptr = None if previous is None else previous.data.ptr
        array = detector._slot_buffer(buffers, 0, (count * 128,), np.float32, 'calib')
        if count == 2:
            assert array.data.ptr == previous_ptr
        else:
            array.fill(count)
            cp.cuda.get_current_stream().synchronize()
            aliases.append(array[:1])
            capacities.append(count * 512)
        del previous, array
        assert budget.committed() == sum(capacities)
        assert cp.get_default_memory_pool().used_bytes()-baseline == sum(capacities)
        assert len(budget.allocation_snapshot()) == len(capacities)
    # Cache trim cannot release aliases or spend their credit on replacement.
    buffers[0] = None
    allocations = []
    original = cp.empty
    def observed(*args, **kwargs):
        allocations.append(True)
        return original(*args, **kwargs)
    monkeypatch.setattr(cp, 'empty', observed)
    with pytest.raises(GpuMemoryPressureError):
        detector._slot_buffer(buffers, 0, (4 * 128,), np.float32, 'calib')
    assert not allocations
    for expected in (1, 4, 8):
        alias = aliases.pop(0)
        np.testing.assert_array_equal(cp.asnumpy(alias), [expected])
        del alias
        capacities.pop(0)
        assert budget.committed() == sum(capacities)
        assert cp.get_default_memory_pool().used_bytes()-baseline == sum(capacities)
    assert budget.allocation_snapshot() == ()


@pytest.mark.parametrize('fail_at', [1, 2, 3])
def test_fixed_upload_allocation_failure_drains_prior_device_uploads(monkeypatch, fail_at):
    import cupy as cp
    from psana.gpu import gpu_allocation
    baseline = cp.get_default_memory_pool().used_bytes()
    budget = _GpuBudget(4096)
    original = cp.empty
    calls = []
    def failing(*args, **kwargs):
        calls.append(True)
        if len(calls) == fail_at:
            raise MemoryError('injected device allocation failure')
        return original(*args, **kwargs)
    monkeypatch.setattr(cp, 'empty', failing)
    with pytest.raises(MemoryError, match='injected device allocation failure'):
        gpu_allocation.upload_owned(cp, [np.arange(130, dtype=np.uint32)] * 3, budget)
    assert len(calls) == fail_at
    assert budget.committed() == budget._held == 0
    assert budget.allocation_snapshot() == ()
    assert budget._failed_allocations == []
    assert cp.get_default_memory_pool().used_bytes() == baseline


def test_failed_upload_survives_lost_budget_owner_and_gc():
    import gc
    import weakref
    from types import SimpleNamespace as NS
    import cupy as cp
    # Collect unrelated cyclic owners before measuring this test's allocation.
    gc.collect()
    baseline = cp.get_default_memory_pool().used_bytes()
    budget = _GpuBudget(1024)
    array = owned_empty(cp, 128, cp.uint32, budget, 'fixed')
    def fail():
        raise RuntimeError('unproven upload completion')
    stream = NS(synchronize=fail)
    budget.quarantine_upload(stream, (array,), (np.arange(128, dtype=np.uint32),))
    ref = weakref.ref(budget)
    del array, budget
    gc.collect()
    try:
        assert ref() is not None
        assert ref().committed() == 512
        assert cp.get_default_memory_pool().used_bytes()-baseline == 512
        with pytest.raises(RuntimeError, match='unproven upload completion'):
            ref().drain_failed_allocations()
        gc.collect()
        assert ref().committed() == 512
    finally:
        if ref() is not None:
            stream.synchronize = lambda: None
            ref().drain_failed_allocations()
    assert ref() is None
    assert cp.get_default_memory_pool().used_bytes() == baseline
