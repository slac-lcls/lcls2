"""Device-only GPU XTC parser correctness on the xpptut test stream."""

import os
import struct

import numpy as np
import pytest

from psana.gpu.gpudgram.batch import (
    DGRAM_EVENT_INDEX,
    DGRAM_NCOLS,
    DGRAM_OFFSET,
    DGRAM_SIZE,
    DGRAM_STATUS,
    DGRAM_STREAM_ID,
    GpuXtcBatchPool,
)
from psana.gpu.gpudgram.parser import (
    LOC_DIM0,
    LOC_NBYTES,
    LOC_NCOLS,
    LOC_OFFSET,
    LOC_RANK,
    LOC_STATUS,
    STATUS_FOUND,
    STATUS_NOT_PRESENT,
    STATUS_OK,
    GpuEventBatch,
)
from psana.gpu.gpudgram.config import GpuStreamConfigTable
from psana.gpu.gpu_kvikio_read import (
    DESC_DEVICE_OFFSET,
    DESC_EVENT_INDEX,
    DESC_NCOLS,
    DESC_READ_SIZE,
    DESC_STREAM_ID,
)


_XTC = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    ".tmp",
    "xpptut15-r0014-s000-c000.xtc2",
)


def _gpu_available():
    try:
        import cupy as cp

        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


requires_gpu = pytest.mark.skipif(
    not _gpu_available(), reason="no CUDA device available"
)
requires_data = pytest.mark.skipif(
    not os.path.isfile(_XTC), reason=f"test data not found: {_XTC}"
)


def _index_post_config(data, config_nbytes):
    records = []
    offset = int(config_nbytes)
    event_index = 0
    while offset < len(data):
        if len(data) - offset < 24:
            raise AssertionError("partial dgram header in test data")
        extent = struct.unpack_from("<I", data, offset + 20)[0]
        size = 12 + extent
        if extent < 12 or size > len(data) - offset:
            raise AssertionError("invalid dgram extent in test data")
        row = np.zeros(DGRAM_NCOLS, dtype=np.uint64)
        row[DGRAM_EVENT_INDEX] = event_index
        row[DGRAM_STREAM_ID] = 0
        row[DGRAM_OFFSET] = offset - config_nbytes
        row[DGRAM_SIZE] = size
        records.append(row)
        event_index += 1
        offset += size
    return np.stack(records)


def _consume_uint16(cp, data_gpu, locators, dgram_index, n_values):
    kernel = cp.RawKernel(
        f"""
        extern "C" __global__
        void consume(const unsigned char* data,
                     const unsigned long long* locators,
                     unsigned long long dgram_index,
                     unsigned long long n_values,
                     unsigned short* output)
        {{
            const unsigned long long i =
                (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
            if (i >= n_values) return;
            const unsigned long long* locator =
                locators + dgram_index * {LOC_NCOLS};
            if (locator[{LOC_STATUS}] != {STATUS_FOUND}) return;
            const unsigned short* field = reinterpret_cast<const unsigned short*>(
                data + locator[{LOC_OFFSET}]);
            output[i] = field[i];
        }}
        """,
        "consume",
    )
    output = cp.full(int(n_values), 0xFFFF, dtype=cp.uint16)
    consumer_stream = cp.cuda.Stream(non_blocking=True)
    locator_rows = locators.wait_on(consumer_stream)
    with consumer_stream:
        kernel(
            ((int(n_values) + 127) // 128,),
            (128,),
            (
                data_gpu,
                locator_rows,
                np.uint64(dgram_index),
                np.uint64(n_values),
                output,
            ),
            stream=consumer_stream,
        )
    consumer_stream.synchronize()
    return output


@pytest.mark.gpu
@requires_gpu
@requires_data
def test_xpptut_field_is_located_and_consumed_without_metadata_round_trip():
    import cupy as cp

    from psana import dgram

    file_bytes = open(_XTC, "rb").read()
    config = dgram.Dgram(view=memoryview(file_bytes), offset=0)
    config_nbytes = int(config._size)
    post_config = file_bytes[config_nbytes:]
    records = _index_post_config(file_bytes, config_nbytes)

    # CPU parsing is only the independent expected-value oracle in this test.
    target_record = 3
    target_offset = config_nbytes + int(records[target_record, DGRAM_OFFSET])
    target_size = int(records[target_record, DGRAM_SIZE])
    cpu_dgram = dgram.Dgram(
        config=config,
        view=memoryview(file_bytes),
        offset=target_offset,
        size=target_size,
    )
    expected = np.asarray(cpu_dgram.xppcspad[1].raw.arrayRaw).copy()

    configs = GpuStreamConfigTable.from_config(config)
    handle = configs.resolve("xppcspad", 1, "raw", "arrayRaw")
    data_gpu = cp.asarray(np.frombuffer(post_config, dtype=np.uint8))
    records_gpu = cp.asarray(records)
    batch = GpuEventBatch(data_gpu, configs.to_device(cp), records_gpu)
    located = batch.locate(handle)
    actual = _consume_uint16(
        cp, data_gpu, located, target_record, expected.size
    ).reshape(expected.shape)

    locator = cp.asnumpy(located.rows_gpu[target_record])
    dgram_status = cp.asnumpy(records_gpu[:, DGRAM_STATUS])
    assert np.all(dgram_status == STATUS_OK)
    assert locator[LOC_STATUS] == STATUS_FOUND
    rank = int(locator[LOC_RANK])
    shape = tuple(int(value) for value in locator[LOC_DIM0 : LOC_DIM0 + rank])
    nbytes = int(locator[LOC_NBYTES])

    assert shape == expected.shape
    assert nbytes == expected.nbytes
    np.testing.assert_array_equal(cp.asnumpy(actual), expected)


@pytest.mark.gpu
@requires_gpu
@requires_data
def test_same_names_id_is_resolved_in_its_own_stream_config_table():
    import cupy as cp

    from psana import dgram

    file_bytes = open(_XTC, "rb").read()
    config = dgram.Dgram(view=memoryview(file_bytes), offset=0)
    config_nbytes = int(config._size)
    records = _index_post_config(file_bytes, config_nbytes)
    target_record = 3
    begin = config_nbytes + int(records[target_record, DGRAM_OFFSET])
    size = int(records[target_record, DGRAM_SIZE])
    event_bytes = file_bytes[begin : begin + size]

    configs = GpuStreamConfigTable(
        {0: config.config_names(), 1: config.config_names()}
    )
    handle0, handle1 = configs.resolve_all(
        "xppcspad", 1, "raw", "arrayRaw"
    )
    assert handle0.names_id == handle1.names_id == 0x10C
    assert handle0.config_names_index != handle1.config_names_index

    two_records = np.zeros((2, DGRAM_NCOLS), dtype=np.uint64)
    two_records[:, DGRAM_EVENT_INDEX] = 0
    two_records[:, DGRAM_STREAM_ID] = [0, 1]
    two_records[:, DGRAM_OFFSET] = [0, size]
    two_records[:, DGRAM_SIZE] = size
    data_gpu = cp.asarray(
        np.frombuffer(event_bytes + event_bytes, dtype=np.uint8)
    )
    records_gpu = cp.asarray(two_records)
    batch = GpuEventBatch(data_gpu, configs.to_device(cp), records_gpu)

    rows0 = cp.asnumpy(batch.locate(handle0).rows_gpu[:, LOC_STATUS])
    rows1 = cp.asnumpy(batch.locate(handle1).rows_gpu[:, LOC_STATUS])
    assert rows0.tolist() == [STATUS_FOUND, STATUS_NOT_PRESENT]
    assert rows1.tolist() == [STATUS_NOT_PRESENT, STATUS_FOUND]


@pytest.mark.gpu
@requires_gpu
@requires_data
def test_slot_pool_reuses_device_parser_tables_without_metadata_round_trip():
    import cupy as cp

    from psana import dgram
    from psana.gpu.gpu_budget import _GpuBudget

    file_bytes = open(_XTC, "rb").read()
    config = dgram.Dgram(view=memoryview(file_bytes), offset=0)
    config_nbytes = int(config._size)
    post_config = file_bytes[config_nbytes:]
    records = _index_post_config(file_bytes, config_nbytes)
    desc = np.zeros((len(records), DESC_NCOLS), dtype=np.uint64)
    desc[:, DESC_EVENT_INDEX] = records[:, DGRAM_EVENT_INDEX]
    desc[:, DESC_STREAM_ID] = records[:, DGRAM_STREAM_ID]
    desc[:, DESC_READ_SIZE] = records[:, DGRAM_SIZE]
    desc[:, DESC_DEVICE_OFFSET] = records[:, DGRAM_OFFSET]

    configs = GpuStreamConfigTable.from_config(config)
    handle = configs.resolve("xppcspad", 1, "raw", "arrayRaw")
    budget = _GpuBudget(limit_bytes=1024**3)
    pool = GpuXtcBatchPool(
        configs,
        field_handles=[handle],
        n_slots=1,
        budget=budget,
    )
    data_gpu = cp.asarray(np.frombuffer(post_config, dtype=np.uint8))
    stream = cp.cuda.Stream(non_blocking=True)

    first = pool.parse(0, data_gpu, desc, stream)
    stream.synchronize()
    first_ptrs = (
        first.dgram_records_gpu.data.ptr,
        first.shape_counts_gpu.data.ptr,
        first.shape_refs_gpu.data.ptr,
        first.locate(handle).rows_gpu.data.ptr,
    )
    memory_after_first = pool.memory_bytes()

    second = pool.parse(0, data_gpu, desc, stream)
    stream.synchronize()
    second_ptrs = (
        second.dgram_records_gpu.data.ptr,
        second.shape_counts_gpu.data.ptr,
        second.shape_refs_gpu.data.ptr,
        second.locate(handle).rows_gpu.data.ptr,
    )

    assert second_ptrs == first_ptrs
    assert pool.memory_bytes() == memory_after_first
    assert budget.committed() == memory_after_first["total"]
    assert cp.asnumpy(second.dgram_records_gpu[:, DGRAM_STATUS]).tolist() == [
        STATUS_OK
    ] * len(records)
