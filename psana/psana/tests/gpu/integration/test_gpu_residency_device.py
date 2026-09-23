"""Production mixed-rate schedule: one fast input and five slow executions."""
from pathlib import Path
import struct
from types import SimpleNamespace as NS

import numpy as np
import pytest

from psana.gpu.gpu_budget import _GpuBudget, allocation_growth_bytes
from psana.gpu.gpu_allocation import allocation_capacity
from psana.gpu.gpu_calib import _upload_fixed_arrays
from psana.gpu.gpu_detector import GPUDetector
from psana.gpu.gpu_events import GpuEventManager
from psana.gpu.gpu_file_epochs import GpuFileEpochs
from psana.gpu.gpu_input import GpuDetectorBinding
from psana.gpu.gpu_kvikio_read import KvikioGpuReader
from psana.gpu.gpu_stream import EventPool
from psana.gpu.gpudgram import GpuStreamConfigTable, GpuXtcBatchPool
from psana.gpu.gpudgram.parser import LOC_OFFSET, LOC_STATUS, STATUS_FOUND


def available():
    try:
        import cupy as cp
        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


def residency_case(tmp_path, mixed_packet, fast_padding=0, slow_padding=1024**2):
    """Real reader/parser/detector fixture shared with lifecycle acceptance."""
    import cupy as cp
    from psana import dgram

    data = (Path(__file__).resolve().parents[2]
            / 'test_data/chunking/xpptut15-r0014-s000-c000.xtc2').read_bytes()
    config = dgram.Dgram(view=memoryview(data), offset=0)
    offset = config._size
    while True:
        size = 12 + struct.unpack_from('<I', data, offset + 20)[0]
        if (struct.unpack_from('<I', data, offset + 8)[0] >> 24) & 15 == 12:
            break
        offset += size
    template = bytearray(data[offset:offset + size])
    cpu = dgram.Dgram(config=config, view=memoryview(template), offset=0)
    expected = cpu.xppcspad[1].raw.arrayRaw.copy()
    raw_offset = cpu.xppcspad[1].raw.arrayRaw.ctypes.data - np.frombuffer(template, np.uint8).ctypes.data
    assert 0 <= raw_offset <= len(template) - expected.nbytes
    timestamp_base = int(cpu.timestamp())
    fast_size, slow_size = size + fast_padding, size + slow_padding

    def record(i, slow=False):
        result = bytearray(template)
        struct.pack_into('<Q', result, 0, timestamp_base + i)
        struct.pack_into('<H', result, raw_offset, 1000 + i if slow else i)
        padding = slow_padding if slow else fast_padding
        if padding:
            # A valid opaque Data sibling enlarges either input without
            # changing its configured raw field. The XTC walker skips it.
            result += struct.pack('<IHHI', 0, 0, 3, padding) + bytes(padding - 12)
            struct.pack_into('<I', result, 20, len(result) - 12)
        return result

    paths = [tmp_path / 'fast.xtc2', tmp_path / 'slow.xtc2']
    paths[0].write_bytes(b''.join(record(i) for i in range(1000)))
    paths[1].write_bytes(b''.join(record(i, True) for i in range(99, 1000, 100)))
    configs = GpuStreamConfigTable.from_configs([config, config])
    handles = [configs.resolve('xppcspad', 1, 'raw', 'arrayRaw', stream_id=s) for s in (0, 1)]
    binding = GpuDetectorBinding('slow', canonical_segment_ids=(1,),
                                 field_handles_by_segment={1: handles[1]})
    budget = _GpuBudget(32 * 1024**2)
    peds, gain = _upload_fixed_arrays((np.zeros(54, np.float32), np.ones(54, np.float32)), budget)
    detector = GPUDetector((1, 3, 6), peds, gain, binding, n_slots=2, budget=budget)
    parser = GpuXtcBatchPool(configs, field_handles=handles, n_slots=3, budget=budget)
    detector.configure_gather(parser.handle_indices)
    per_dgram = parser.estimate_batch_bytes(1)
    resident_bytes = 1000 * (fast_size + per_dgram)
    slow_cost = slow_size + per_dgram + detector.estimate_subbatch_bytes(1)
    capacity = resident_bytes + 4 * slow_cost
    # Keep the same logical residency plan and explicitly budget the pool's
    # rounded blocks: one resident input/parser set and two transient sets.
    def rounding(n, dgram_size):
        physical = (allocation_capacity(cp, n * dgram_size)
                    + allocation_growth_bytes(parser.allocation_requirements(n)))
        if n == 2:
            physical += allocation_growth_bytes(detector.allocation_requirements(n, 0))
            return physical - n * (dgram_size + per_dgram + detector.estimate_subbatch_bytes(1))
        return physical - n * (dgram_size + per_dgram)
    pool_rounding = rounding(1000, fast_size) + 2 * rounding(2, slow_size)
    budget._limit = budget.committed() + capacity + pool_rounding
    m = GpuEventManager.__new__(GpuEventManager)
    m.dm = NS(xtc_files=paths, get_chunk_id=lambda _: 0, fds=[0, 1])
    m.dsparms = NS(gpu_bulk_read=True, n_gpu_streams=2, max_events=0)
    m._gpu_budget, m._admission_margin = budget, 0
    m._admission_capacity, m._subbatch_budget_bytes = capacity, capacity // 2
    m.gpu_reader = KvikioGpuReader(n_slots=3, budget=budget)
    m.gpu_xtc_parser, m.event_pool = parser, EventPool(n=2)
    m.gpu_detectors, m.gpu_det_names = {'slow': (None, detector)}, ['slow']
    m._gpu_file_epochs = GpuFileEpochs(m.dm)
    m.configs, m._d2h_pipelines = [config, config], {}
    m._first_batch_logged, m._done, m._closed = True, False, False
    m._n_events, m._pending_gpu_read = 0, None
    packet = mixed_packet(fast_size=fast_size, slow_size=slow_size,
                          timestamp_base=timestamp_base)
    return NS(manager=m, packet=packet, expected=expected, handles=handles,
              fast_size=fast_size, slow_size=slow_size,
              resident_bytes=resident_bytes, per_dgram=per_dgram,
              timestamp_base=timestamp_base)


@pytest.mark.gpu
@pytest.mark.skipif(not available(), reason='no CUDA device')
@pytest.mark.parametrize('fast_padding,slow_padding', [
    (0, 1024**2),
    (16 * 1024, 64 * 1024),  # frequent small dgrams have the larger total footprint
])
def test_resident_fast_and_five_slow_reads_match_cpu(
        tmp_path, mixed_packet, fast_padding, slow_padding):
    import cupy as cp
    case = residency_case(tmp_path, mixed_packet, fast_padding, slow_padding)
    m, packet, expected, handles = case.manager, case.packet, case.expected, case.handles
    budget = m._gpu_budget
    fast_size, slow_size = case.fast_size, case.slow_size
    resident_bytes, per_dgram = case.resident_bytes, case.per_dgram
    fast_owner, fast_bytes, rows, slow_owners, observed = None, None, None, set(), []
    try:
        for envelope in m._process_batch({}, {0: (packet, [])}, {}):
            state = envelope.gpu_state
            dgrams = state._event_dgrams
            i = dgrams.batch_event_index
            owner = dgrams[0].owner
            if fast_owner is None:
                fast_owner = owner
                fast_bytes = cp.asnumpy(owner.batch.data_gpu)
                rows = cp.asnumpy(owner.batch.locate(handles[0]).rows_gpu)
            assert owner is fast_owner and not owner.released
            row = dgrams[0].dgram_index
            assert rows[row, LOC_STATUS] == STATUS_FOUND
            start = int(rows[row, LOC_OFFSET])
            raw = np.frombuffer(fast_bytes, np.uint16, expected.size, start).reshape(expected.shape)
            reference = expected.copy()
            reference.flat[0] = i
            np.testing.assert_array_equal(raw, reference)
            if 1 in dgrams:
                slow_owners.add(id(dgrams[1].owner))
                reference.flat[0] = 1000 + i
                np.testing.assert_array_equal(cp.asnumpy(state._gpu_results['slow.raw'])[0], reference)
                np.testing.assert_array_equal(cp.asnumpy(state._gpu_results['slow.calib'])[0], reference.astype(np.float32))
            observed.append(i)
            if i == 999:
                # Re-read device storage after all five slow executions. A host
                # snapshot alone would hide accidental resident-buffer reuse.
                np.testing.assert_array_equal(cp.asnumpy(owner.batch.data_gpu), fast_bytes)
        assert observed == list(range(1000))
        plan = m._last_admission_plan
        assert plan.resident_streams == (0,)
        if fast_padding:
            assert 1000 * fast_size > 10 * slow_size
            assert resident_bytes > 10 * (slow_size + per_dgram)
        assert [(d.candidate.stream_id, d.admitted) for d in plan.residency_decisions] == [(0, True), (1, False)]
        assert len(slow_owners) == 5 and fast_owner.released
        stats = m.gpu_reader.io_stats()
        assert stats['total_requests'] == 6
        assert stats['requested_bytes'] == 1000 * fast_size + 10 * slow_size
        assert budget._held == 0 and budget.committed() <= budget.limit()
        assert m.event_pool.active_count == 0 and not any(m.gpu_reader._input_holds.values())
    finally:
        m.close()
