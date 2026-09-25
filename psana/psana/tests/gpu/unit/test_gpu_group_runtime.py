"""Run the controller and reader with injected CUDA completion and tiny I/O."""
from types import SimpleNamespace as NS

import pytest

from test_gpu_bulk_read import io
from test_gpu_residency import manager
from test_gpu_input_window import Token
from psana.gpu.gpu_input_group import InputGroupPool
from psana.gpu.gpu_input_window import InputWindow
from psana.gpu.gpu_kvikio_read import KvikioGpuReader


class Parser:
    def __init__(self):
        self.windows = []
        self.launch_batches = []

    def estimate_batch_bytes(self, n):
        return n

    def allocation_requirements(self, n, *, groups=False):
        return []

    def parse_groups(self, reads, stream, *, batch_id):
        self.launch_batches.append(len(reads))
        windows = []
        for read in reads:
            batch = NS(data_gpu=read.data_gpu, n_dgrams=len(read.desc_table), walk_done=Token())
            window = InputWindow(batch_id, len(self.windows), batch, read.desc_table,
                                 release=read.retain_input(), defer_retirement=True)
            self.windows.append(window)
            windows.append(window)
        return windows

    def trim_free_buffers(self):
        pass

    def close(self):
        assert all(w.drain() for w in self.windows)


def build(io, monkeypatch):
    monkeypatch.setattr(Token, 'done', property(lambda self: True), raising=False)
    m = manager(io, capacity=4 * 1024**2)
    m.gpu_reader.close()
    m.gpu_reader = KvikioGpuReader(n_slots=30, budget=m._gpu_budget)
    m._group_inputs = InputGroupPool(m.gpu_reader)
    m.gpu_xtc_parser = Parser()
    io.files = {'/fast': bytes(range(120)), '/slow': b'x' * (4 * 1024**2)}
    return m


def test_controller_interleaves_requests_and_releases_all_group_inputs(io, monkeypatch, mixed_packet):
    m = build(io, monkeypatch)
    packet = mixed_packet(n_events=12, fast_size=10, slow_size=1024**2, interval=3)
    observed = []
    try:
        for envelope in m._process_batch({}, {0: (packet, [])}, {}):
            observed.append(envelope.gpu_state._event_dgrams.batch_event_index)
        for envelope in m.finish():
            observed.append(envelope.gpu_state._event_dgrams.batch_event_index)
        assert observed == list(range(12))
        assert m.gpu_reader.io_stats()['total_requests'] == 5
        assert m.gpu_xtc_parser.launch_batches[0] == 2
        assert not m._group_inputs.live_keys and m._gpu_budget._held == 0
    finally:
        m.close()


def test_early_close_cancels_future_small_group_uses(io, monkeypatch, mixed_packet):
    m = build(io, monkeypatch)
    packet = mixed_packet(n_events=12, fast_size=10, slow_size=1024**2, interval=3)
    loop = m._process_batch({}, {0: (packet, [])}, {})
    next(loop)
    loop.close()
    m.close()
    assert not m._group_inputs.live_keys
    assert not any(m.gpu_reader._input_holds.values())
    assert m._gpu_budget._held == 0
