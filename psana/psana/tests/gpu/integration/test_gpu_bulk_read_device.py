"""Real KvikIO/CUDA coalescing and parser parity on a tracked XTC fixture."""

from pathlib import Path
import struct
from types import SimpleNamespace as NS

import numpy as np
import pytest

from psana.gpu.gpu_batch import GpuReadDesc
from psana.gpu.gpu_file_epochs import GpuFileEpochs
from psana.gpu.gpu_kvikio_read import KvikioGpuReader, DESC_DEVICE_OFFSET, DESC_READ_SIZE
from psana.gpu.gpudgram.batch import GpuXtcBatchPool
from psana.gpu.gpudgram.config import GpuStreamConfigTable


def available():
    try:
        import cupy as cp
        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


@pytest.mark.gpu
@pytest.mark.skipif(not available(), reason="no CUDA device")
def test_coalesced_kvikio_bytes_and_parsed_fields_match_per_dgram_reads(tmp_path):
    import cupy as cp
    from psana import dgram

    fixture = Path(__file__).resolve().parents[2] / "test_data/chunking/xpptut15-r0014-s000-c000.xtc2"
    data = fixture.read_bytes()
    config = dgram.Dgram(view=memoryview(data), offset=0)
    # Independent fixture framing; production uses GPUBAT1 descriptors.
    records = []
    offset = config._size
    while offset < len(data):
        size = 12 + struct.unpack_from("<I", data, offset + 20)[0]
        service = (struct.unpack_from("<I", data, offset + 8)[0] >> 24) & 15
        if service == 12:
            records.append((offset, size, struct.unpack_from("<Q", data, offset)[0]))
        offset += size
    records = records[:4]
    assert len(records) == 4
    paths = [tmp_path / "z.xtc2", tmp_path / "a.xtc2"]
    for path in paths:
        path.write_bytes(data)
    dm = NS(xtc_files=paths, get_chunk_id=lambda _: 0)
    descriptors = [GpuReadDesc(i, ts, stream, off, size, 0, 1)
                   for i, (off, size, ts) in enumerate(records) for stream in range(2)]
    view = NS(iter_read_descs=lambda _: iter(descriptors))
    epochs = GpuFileEpochs(dm).resolve(descriptors, [])
    configs = GpuStreamConfigTable.from_configs([config, config])
    handles = [configs.resolve("xppcspad", 1, "raw", "arrayRaw", stream_id=stream)
               for stream in range(2)]
    results, requests = [], []
    for bulk in (False, True):
        reader = KvikioGpuReader(bulk_read=bulk)
        try:
            read = reader.wait_batch(reader.issue_batch(view, dm, file_epochs=epochs))
            pool = GpuXtcBatchPool(configs, field_handles=handles, n_slots=1)
            stream = cp.cuda.Stream(non_blocking=True)
            batch = pool.parse(0, read.data_gpu, read.desc_table, stream)
            stream.synchronize()
            payloads, fields = [], []
            for i, (desc, row) in enumerate(zip(descriptors, read.desc_table)):
                off, size = int(row[DESC_DEVICE_OFFSET]), int(row[DESC_READ_SIZE])
                payload = cp.asnumpy(read.data_gpu[off:off + size]).tobytes()
                assert payload == data[desc.offset:desc.offset + desc.size]
                payloads.append(payload)
                from psana.gpu.gpudgram.batch import LOC_OFFSET, LOC_NBYTES, LOC_STATUS
                from psana.gpu.gpudgram.parser import STATUS_FOUND
                locator = cp.asnumpy(batch.locate(handles[desc.stream_id]).rows_gpu[i])
                assert locator[LOC_STATUS] == STATUS_FOUND
                field_off, field_size = int(locator[LOC_OFFSET]), int(locator[LOC_NBYTES])
                fields.append(cp.asnumpy(read.data_gpu[field_off:field_off + field_size]).tobytes())
            results.append((payloads, fields))
            requests.append(reader.io_stats()["total_requests"])
        finally:
            reader.close()
    assert results[0] == results[1]
    assert requests[0] == 8
    assert requests[1] < requests[0]
