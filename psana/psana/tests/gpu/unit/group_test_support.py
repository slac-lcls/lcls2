"""CPU controller scaffolding shared by group-scheduling tests."""
import sys
from types import SimpleNamespace as NS
from psana.gpu.gpu_events import GpuEventManager
from psana.gpu.gpu_budget import _GpuBudget
from psana.gpu.gpu_file_epochs import GpuFileEpochs
from psana.gpu.gpu_kvikio_read import KvikioGpuReader
from psana.gpu.gpu_stream import EventPool
from test_gpu_input_window import Stream, Token


def manager(io, capacity=6004, *, max_events=0):
    cp = sys.modules['cupy']
    cp.cuda = NS(Stream=Stream, Event=Token)
    m = GpuEventManager.__new__(GpuEventManager)
    m.dm = NS(xtc_files=['/fast', '/slow'], get_chunk_id=lambda _: 0, fds=[0, 1])
    m.dsparms = NS(gpu_bulk_read=True, n_gpu_streams=2, max_events=max_events)
    m._gpu_budget = _GpuBudget(capacity)
    m._admission_capacity, m._admission_margin = capacity, 0
    m._subbatch_budget_bytes = capacity // 2
    m.gpu_reader = KvikioGpuReader(n_slots=3, budget=m._gpu_budget)
    m.gpu_xtc_parser = None
    m.event_pool = EventPool(n=2)
    m.gpu_detectors, m.gpu_det_names, m._d2h_pipelines = {}, [], {}
    m.configs = [None, None]
    m._gpu_file_epochs = GpuFileEpochs(m.dm)
    m._first_batch_logged, m._done, m._closed = True, False, False
    m._n_events, m._pending_gpu_read = 0, None
    m.run = NS(_handle_transition=lambda _: None)
    io.files = {'/fast': bytes(i % 256 for i in range(1000)),
                '/slow': b'x' * 10000}
    return m
