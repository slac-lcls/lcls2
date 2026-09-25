"""Independent group reclamation with real KvikIO, XTC parsing and CUDA work."""
from pathlib import Path
import struct

import numpy as np
import pytest

from psana.gpu.gpu_input_group import InputGroupPool
from psana.gpu.gpu_kvikio_read import KvikioGpuReader
from psana.gpu.gpu_read_plan import ResolvedDgram, ResolvedFile
from psana.gpu.gpu_stream_read_plan import StreamReadGroup
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
@pytest.mark.parametrize('batched', [False, True])
def test_later_group_reuses_storage_while_earlier_cuda_consumers_run(tmp_path, batched):
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
            records.append(data[offset:offset + size])
        offset += size
    assert len(records) == 2
    # Coalesced two-event input and independently reusable one-event inputs.
    groups = []
    for stream in (0, 1):
        path = tmp_path / f's{stream}.xtc2'
        path.write_bytes(b''.join(records))
        file = ResolvedFile(str(path), 0)
        rows, offset = [], 0
        for event, record in enumerate(records):
            rows.append(ResolvedDgram(event, struct.unpack_from('<Q', record)[0],
                                      stream, file, offset, len(record)))
            offset += len(record)
        if stream == 0:
            groups.append(StreamReadGroup(0, stream, 0, file, 0, offset, tuple(rows), True))
        else:
            groups.extend(StreamReadGroup(i + 1, stream, 0, file, r.file_offset,
                                          r.size, (r,), False) for i, r in enumerate(rows))
    configs = GpuStreamConfigTable.from_configs([config, config])
    handles = [configs.resolve('xppcspad', 1, 'raw', 'arrayRaw', stream_id=i) for i in (0, 1)]
    reader = KvikioGpuReader(n_slots=3)
    inputs = InputGroupPool(reader)
    parser = GpuXtcBatchPool(configs, field_handles=handles, n_slots=3)
    parsing = cp.cuda.Stream(non_blocking=True)
    consumer = cp.cuda.Stream(non_blocking=True)
    delayed = cp.RawKernel(r'''
    extern "C" __global__ void delayed_read(const unsigned char* a,
        const unsigned char* b, unsigned char* out) {
        unsigned long long start = clock64();
        while (clock64() - start < 1000000000ULL) {}
        out[0] = a[0]; out[1] = b[0];
    }
    ''', 'delayed_read')
    delayed.compile()
    observed = cp.empty(2, dtype=cp.uint8)
    done = cp.cuda.Event(disable_timing=True)
    try:
        keys = [inputs.issue(0, g) for g in groups]
        windows = (inputs.parse_groups(keys, parser, parsing) if batched
                   else [inputs.parse(k, parser, parsing) for k in keys])
        parsing.synchronize()
        raw = windows[0].batch.data_gpu
        locators = windows[0].locate(handles[0]).rows_gpu
        loc_before = cp.asnumpy(locators)
        reusable_ptr = windows[2].batch.data_gpu.data.ptr
        cp.cuda.Stream.null.synchronize()
        delayed((1,), (1,), (raw, windows[1].batch.data_gpu, observed), stream=consumer)
        done.record(consumer)
        for key, events in ((keys[0], (0, 1)), (keys[1], (0,))):
            for event in events:
                use = inputs.take_use(key, event)
                use.register_consumer_done(done)
                use.wait_until_safe_to_reuse()
        inputs.take_use(keys[2], 1).wait_until_safe_to_reuse()
        assert not done.done
        assert inputs.poll() == (keys[2],)
        assert not done.done  # collection did not synchronize the slow consumer
        assert not windows[0].released and not windows[1].released
        replacement = inputs.issue(1, groups[2])
        new_window = inputs.parse(replacement, parser, parsing)
        assert new_window.batch.data_gpu.data.ptr == reusable_ptr
        parsing.synchronize()
        np.testing.assert_array_equal(cp.asnumpy(raw), np.frombuffer(b''.join(records), dtype=np.uint8))
        np.testing.assert_array_equal(cp.asnumpy(locators), loc_before)
        inputs.take_use(replacement, 1).wait_until_safe_to_reuse()
        inputs.poll()
        done.synchronize()
        assert set(inputs.poll()) == set(keys[:2])
        assert all(w.released for w in windows)
        np.testing.assert_array_equal(cp.asnumpy(observed), [records[0][0], records[0][0]])
    finally:
        inputs.close()
        parser.close()
