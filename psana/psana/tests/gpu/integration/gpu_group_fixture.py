"""Real mixed-rate XTC input fixture for production group scheduling."""
from pathlib import Path
import struct
from types import SimpleNamespace as NS

import numpy as np

from psana.gpu.gpu_budget import _GpuBudget
from psana.gpu.gpu_calib import _upload_fixed_arrays
from psana.gpu.gpu_detector import GPUDetector
from psana.gpu.gpu_events import GpuEventManager
from psana.gpu.gpu_file_epochs import GpuFileEpochs
from psana.gpu.gpu_input import GpuDetectorBinding
from psana.gpu.gpu_input_group import InputGroupPool
from psana.gpu.gpu_kvikio_read import KvikioGpuReader
from psana.gpu.gpu_stream import EventPool
from psana.gpu.gpudgram import GpuStreamConfigTable, GpuXtcBatchPool


def available():
    try:
        import cupy as cp
        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


def group_case(tmp_path, mixed_packet, fast_padding=0, slow_padding=1024**2):
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
    parser = GpuXtcBatchPool(configs, field_handles=handles, n_slots=5, budget=budget)
    detector.configure_gather(parser.handle_indices)
    per_dgram = parser.estimate_batch_bytes(1)
    capacity = 16 * 1024**2
    budget._limit = 64 * 1024**2
    m = GpuEventManager.__new__(GpuEventManager)
    m.dm = NS(xtc_files=paths, get_chunk_id=lambda _: 0, fds=[0, 1])
    m.dsparms = NS(gpu_bulk_read=True, n_gpu_streams=2, max_events=0)
    m._gpu_budget, m._admission_margin = budget, 0
    m._admission_capacity, m._subbatch_budget_bytes = capacity, capacity // 2
    m.gpu_reader = KvikioGpuReader(n_slots=2020, budget=budget)
    m._group_inputs = InputGroupPool(m.gpu_reader)
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
              per_dgram=per_dgram,
              timestamp_base=timestamp_base)
