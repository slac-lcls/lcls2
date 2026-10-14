"""Real read/parse/gather allocations fit the pre-I/O admission reservation."""

from pathlib import Path
import struct
from types import SimpleNamespace as NS

import numpy as np
import pytest

from psana.gpu.gpu_admission import AdmissionEvent
from psana.gpu.gpu_batch import GpuReadDesc
from psana.gpu.gpu_budget import _GpuBudget, GpuMemoryPressureError, allocation_growth_bytes
from psana.gpu.gpu_calib import _upload_fixed_arrays
from psana.gpu.gpu_detector import GPUDetector
from psana.gpu.gpu_events import GpuEventManager
from psana.gpu.gpu_file_epochs import GpuFileEpochs
from psana.gpu.gpu_input import GpuDetectorBinding
from psana.gpu.gpu_kvikio_read import KvikioGpuReader
from psana.gpu.gpu_stream import EventPool
from psana.gpu.gpudgram import GpuStreamConfigTable, GpuXtcBatchPool


def available():
    try:
        import cupy as cp
        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


@pytest.mark.gpu
@pytest.mark.skipif(not available(), reason='no CUDA device')
def test_read_parse_and_detector_growth_fit_exact_admission(tmp_path):
    import cupy as cp
    from psana import dgram

    fixture = Path(__file__).resolve().parents[2] / 'test_data/chunking/xpptut15-r0014-s000-c000.xtc2'
    data = fixture.read_bytes()
    config = dgram.Dgram(view=memoryview(data), offset=0)
    records, offset = [], config._size
    while offset < len(data) and len(records) < 4:
        size = 12 + struct.unpack_from('<I', data, offset + 20)[0]
        if (struct.unpack_from('<I', data, offset + 8)[0] >> 24) & 15 == 12:
            records.append((offset, size, struct.unpack_from('<Q', data, offset)[0]))
        offset += size
    assert len(records) == 4
    path = tmp_path / 'admission.xtc2'
    path.write_bytes(data)
    configs = GpuStreamConfigTable.from_configs([config])
    handle = configs.resolve('xppcspad', 1, 'raw', 'arrayRaw', stream_id=0)
    binding = GpuDetectorBinding('xppcspad', canonical_segment_ids=(1,),
                                 field_handles_by_segment={1: handle},
                                 field_handles_by_name={('raw', 'arrayRaw'): {1: handle}})
    budget = _GpuBudget(1024**2)
    peds, gain = _upload_fixed_arrays((np.zeros(54, np.float32), np.ones(54, np.float32)), budget)
    detector = GPUDetector((1, 3, 6), peds, gain, binding, n_slots=1, budget=budget)
    parser = GpuXtcBatchPool(configs, field_handles=(handle,), n_slots=1, budget=budget)
    detector.configure_gather(parser.handle_indices)
    reader = KvikioGpuReader(n_slots=1, budget=budget)
    manager = GpuEventManager.__new__(GpuEventManager)
    manager.dm = NS(xtc_files=[path], get_chunk_id=lambda _: 0)
    manager.dsparms = NS(gpu_bulk_read=True)
    manager.gpu_reader, manager.gpu_xtc_parser = reader, parser
    manager.gpu_detectors = {'xppcspad': (None, detector)}
    manager.event_pool = EventPool(n=1)
    manager._gpu_budget, manager._admission_margin = budget, 0
    manager._pending_gpu_read = None
    manager._d2h_pipelines = {}
    try:
        for batch_id, count in enumerate((1, 4, 2)):
            descs = [GpuReadDesc(i, ts, 0, off, size, 0, 1)
                     for i, (off, size, ts) in enumerate(records[:count])]
            view = NS(iter_read_descs=lambda _: iter(descs),
                      total_read_bytes=sum(d.size for d in descs),
                      iter_events=lambda: iter(NS(batch_event_index=i, timestamp=d.timestamp,
                                                 first_desc=i, n_desc=1) for i, d in enumerate(descs)))
            manager._event_memory = lambda _: [AdmissionEvent(((0, d.size),),
                                                               detector.estimate_subbatch_bytes(1)) for d in descs]
            manager._input_batch_id = batch_id
            manager._gpu_read_files = GpuFileEpochs(manager.dm).resolve(descs, [])
            requirements = (reader.allocation_requirements(view.total_read_bytes, 0)
                            + parser.allocation_requirements(count)
                            + detector.allocation_requirements(count, 0))
            growth = allocation_growth_bytes(requirements)
            budget._limit = budget.committed() + growth
            pending = manager._issue_gpu_read(view, 0)
            read = manager._wait_gpu_read(pending)
            record = manager._submit_gpu(view, read, [])
            assert budget._held == 0
            record.stream.synchronize()
            for desc in descs:
                cpu = dgram.Dgram(config=config, view=memoryview(data), offset=desc.offset)
                raw = cp.asnumpy(record.gpu_results_by_ts[desc.timestamp]['xppcspad.raw'])[0]
                np.testing.assert_array_equal(raw, cpu.xppcspad[1].raw.arrayRaw)
            owned = reader.memory_bytes()['raw_input_slots'] + parser.memory_bytes()['total'] + detector.memory_bytes()['total']
            assert budget.committed() == owned <= budget.limit()
            manager.event_pool.begin_retire_next()
            manager.event_pool.finish_retire_next()
        # Cached capacity is still charged after leases finish, then trimming
        # returns only variable storage; constants and Configure stay charged.
        fixed = (parser.memory_bytes()['config'] + detector.memory_bytes()['constants']
                 + detector.memory_bytes()['routing'])
        assert budget.committed() > fixed
        # These public read/record aliases keep reader/parser backing charged
        # even after retirement. Remove them before asserting cache-only cost.
        del pending, read, record
        manager._trim_gpu_caches()
        assert budget.committed() == fixed
        budget._limit = fixed
        requests_before = reader.io_stats()['total_requests']
        with pytest.raises(GpuMemoryPressureError):
            manager._issue_gpu_read(view, 0)
        assert reader.io_stats()['total_requests'] == requests_before
    finally:
        manager._drain_pending_gpu_read()
        for _ in manager.event_pool.flush():
            pass
        parser.close()
        reader.close()
