"""A retained parsed input survives independent input/execution slot reuse."""
from pathlib import Path
import struct
from types import SimpleNamespace as NS

import numpy as np
import pytest

from psana.gpu.gpu_batch import GpuReadDesc
from psana.gpu.gpu_file_epochs import GpuFileEpochs
from psana.gpu.gpu_kvikio_read import KvikioGpuReader
from psana.gpu.gpu_stream import EventPool
from psana.gpu.gpudgram.batch import GpuXtcBatchPool
from psana.gpu.gpudgram.config import GpuStreamConfigTable


def available():
    try:
        import cupy as cp
        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


@pytest.mark.gpu
@pytest.mark.skipif(not available(), reason='no CUDA device')
def test_retained_input_bytes_and_locators_survive_slow_slot_reuse(tmp_path):
    import cupy as cp
    from psana import dgram

    fixture = Path(__file__).resolve().parents[2] / 'test_data/chunking/xpptut15-r0014-s000-c000.xtc2'
    data = fixture.read_bytes()
    config = dgram.Dgram(view=memoryview(data), offset=0)
    records, offset = [], config._size
    while offset < len(data) and len(records) < 2:
        size = 12 + struct.unpack_from('<I', data, offset + 20)[0]
        service = (struct.unpack_from('<I', data, offset + 8)[0] >> 24) & 15
        if service == 12:
            records.append((offset, size, struct.unpack_from('<Q', data, offset)[0]))
        offset += size
    assert len(records) == 2
    paths = [tmp_path / 'fast.xtc2', tmp_path / 'slow.xtc2']
    for path in paths:
        path.write_bytes(data)
    dm = NS(xtc_files=paths, get_chunk_id=lambda _: 0)
    fast_desc = [GpuReadDesc(i, ts, 0, off, size, 0, 1)
                 for i, (off, size, ts) in enumerate(records)]
    off, size, ts = records[1]
    slow_desc = [GpuReadDesc(1, ts, 1, off, size, 0, 1)]
    epochs = GpuFileEpochs(dm).resolve(fast_desc + slow_desc, [])
    configs = GpuStreamConfigTable.from_configs([config, config])
    handles = [configs.resolve('xppcspad', 1, 'raw', 'arrayRaw', stream_id=i) for i in range(2)]
    reader = KvikioGpuReader(n_slots=2)
    parser = GpuXtcBatchPool(configs, field_handles=handles, n_slots=2)
    pool = EventPool(n=1)
    parse_stream = cp.cuda.Stream(non_blocking=True)

    def read(descs, slot):
        return reader.wait_batch(reader.issue_batch(
            NS(iter_read_descs=lambda _: iter(descs)), dm,
            slot_id=slot, file_epochs=epochs))

    fast = parser.parse_window(read(fast_desc, 0), parse_stream, batch_id=9)
    planned = fast.acquire()
    parse_stream.synchronize()
    raw_before = cp.asnumpy(fast.batch.data_gpu)
    locators = fast.batch.locate(handles[0]).rows_gpu
    loc_before = cp.asnumpy(locators)
    locator_ptr = locators.data.ptr
    consumer = cp.cuda.Stream(non_blocking=True)
    observed = cp.zeros(1, dtype=cp.uint8)
    delayed = cp.RawKernel(r'''
    extern "C" __global__ void delayed_read(const unsigned char* input,
        unsigned long long index, unsigned char* output) {
        unsigned long long start = clock64();
        while (clock64() - start < 1000000000ULL) {}
        output[0] = input[index];
    }
    ''', 'delayed_read')
    delayed.compile()
    cp.cuda.Stream.null.synchronize()
    consumer.wait_event(fast.batch.walk_done)
    delayed((1,), (1,), (fast.batch.data_gpu, np.uint64(len(raw_before)-1), observed), stream=consumer)
    done = cp.cuda.Event(disable_timing=True)
    done.record(consumer)
    planned.register_consumer_done(done)
    gv = NS(iter_events=lambda: iter((NS(batch_event_index=1, timestamp=ts, first_desc=0, n_desc=2),)))
    try:
        assert not done.done  # actual outstanding CUDA consumer
        for _ in range(3):
            slow = parser.parse_window(read(slow_desc, 1), parse_stream, batch_id=9)
            record = pool.submit(gv, None, [], {}, input_windows=(fast, slow), batch_id=9)
            assert record.input_dgrams_by_ts[ts][0].owner is fast
            slow.close()
            pool.begin_retire_next()
            pool.finish_retire_next()
            assert slow.released and not fast.released
            assert fast.batch.locate(handles[0]).rows_gpu.data.ptr == locator_ptr
            np.testing.assert_array_equal(cp.asnumpy(locators), loc_before)
            np.testing.assert_array_equal(cp.asnumpy(fast.batch.data_gpu), raw_before)
        with pytest.raises(RuntimeError, match='owned by an input window'):
            read(fast_desc, 0)
        with pytest.raises(RuntimeError, match='owned by an input window'):
            parser.parse(0, fast.batch.data_gpu, fast.desc_table, parse_stream)
        assert not fast.close()
        planned.wait_until_safe_to_reuse()
        assert fast.released and done.done
        assert int(cp.asnumpy(observed)[0]) == int(raw_before[-1])
        # Both input pools can reuse their storage after the last dependency.
        replacement = parser.parse_window(read(fast_desc, 0), parse_stream, batch_id=10)
        assert replacement.close()
    finally:
        for _ in pool.flush():
            pass
        planned.wait_until_safe_to_reuse()
        fast.close()
        parser.close()
        reader.close()
