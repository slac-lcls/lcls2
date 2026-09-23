"""Transitions must drain active multi-owner work before changing constants."""
from types import SimpleNamespace as NS

import numpy as np
import pytest

from test_batched_gather import _gpu_available
from test_multiowner_gather import fixture_inputs
from psana.gpu import gpu_events
from psana.gpu.gpu_stream import EventPool
from psana.psexp import TransitionId

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(
    not _gpu_available(), reason='no CUDA device available')]


@pytest.mark.parametrize('service', [TransitionId.BeginStep, TransitionId.EndRun])
def test_transition_drains_pending_multiowner_consumers_before_dispatch(monkeypatch, service):
    import cupy as cp
    _, detector, _, specs, expected, owner = fixture_inputs(cp)
    producer = cp.cuda.Stream(non_blocking=True)
    windows = tuple(owner((i,), range(4), producer) for i in range(2))
    m = gpu_events.GpuEventManager.__new__(gpu_events.GpuEventManager)
    m.event_pool = EventPool(n=1)
    m._first_batch_logged = True
    m.gpu_detectors, m.configs = {'camera': (None, detector)}, []
    record = m.event_pool.submit(NS(iter_events=lambda: iter(specs)), None, [],
                                 m.gpu_detectors, input_windows=windows, batch_id=7)
    output = record.gpu_results_by_ts[specs[0].timestamp]['camera.calib']
    lease = record.leases_by_ts[specs[0].timestamp]['camera.calib']
    consumer = cp.cuda.Stream(non_blocking=True)
    delay = cp.RawKernel('''extern "C" __global__ void delay(unsigned long long ticks) {
        unsigned long long start = clock64();
        while (clock64() - start < ticks) {}
    }''', 'delay')
    delay.compile()
    with consumer:
        consumer.wait_event(lease.result_ready)
        delay((1,), (1,), (np.uint64(60000000),))
        copied = output.copy()
        done = cp.cuda.Event(disable_timing=True)
        done.record(consumer)
    order = []

    class Completion:
        def synchronize(self):
            done.synchronize()
            order.append('consumer')
    lease.register_consumer_done(Completion())
    for window in windows:
        assert not window.close()
    assert m.event_pool.active_count == 1
    monkeypatch.setattr(gpu_events, '_iter_step_events', lambda packet, configs: iter(packet))

    def constants(*args, **kwargs):
        assert done.done and order == ['consumer']
        assert not m.event_pool.active_count and all(w.released for w in windows)
        order.append('constants')
        return np.full(2700, 11, np.float32), np.full(2700, 3, np.float32)
    monkeypatch.setattr(gpu_events, '_compute_calib_constants_cpu', constants)

    def dispatch(dgrams):
        assert done.done and not m.event_pool.active_count
        assert all(w.released for w in windows)
        order.append('dispatch')
    m.run = NS(_handle_transition=dispatch)
    dg = NS(service=lambda: service)
    try:
        assert list(m._handle_steps({0: ([(service, [dg])], [])})) == []
        assert order == (['consumer', 'constants', 'dispatch'] if service == TransitionId.BeginStep
                         else ['consumer', 'dispatch'])
        np.testing.assert_array_equal(copied.get(), (expected[0].astype(np.float32) - 7) * 2)
        if service == TransitionId.BeginStep:
            np.testing.assert_array_equal(detector.peds_gpu.get(), np.full(2700, 11, np.float32))
    finally:
        for _ in m.event_pool.flush():
            pass
        for window in windows:
            window.close()
