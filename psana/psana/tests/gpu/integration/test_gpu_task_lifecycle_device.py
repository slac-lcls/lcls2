"""Stage 6: real task, transfer and input ownership across lifecycle boundaries."""
from types import SimpleNamespace as NS

import numpy as np
import pytest

from test_gpu_producer_device import fixture, bindings, envelopes
from test_gpu_allocation_device import available
from psana.gpu import GpuTask, gpu_events
from psana.gpu.gpu_d2h import PublicationD2H
from psana.gpu.gpu_stream import EventPool
from psana.gpu.gpu_task import RequestedConstants
from psana.psexp import TransitionId

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(not available(), reason='no CUDA device')]


def manager(pool, mapping):
    m = gpu_events.GpuEventManager.__new__(gpu_events.GpuEventManager)
    m.event_pool = pool
    m._first_batch_logged = True
    m.gpu_det_names = ['camera']
    m.gpu_detector_bindings = mapping
    m.configs = []
    m._step_generation = 0
    return m


@pytest.mark.parametrize('cap', [0, 8192])
@pytest.mark.parametrize('service', [TransitionId.BeginStep, TransitionId.EndRun])
def test_transition_drains_task_publications_before_constant_change(monkeypatch, cap, service):
    import cupy as cp
    parser, detector, budget, specs, expected, window = fixture(cp, ((0,),) * 3)
    mapping = bindings(parser, detector)
    pool = EventPool(n=1, budget=budget)
    pipeline = PublicationD2H(cap)
    constants = RequestedConstants([('camera', 'gain')], budget)
    host = np.array(1, np.uint16)
    constants.refresh({'camera': {'gain': host}})
    old_gain = constants.get('camera', 'gain')
    m = manager(pool, mapping)
    m.dsparms = NS(calibconst={'camera': {'gain': host}})
    m._task_constants = constants
    m._trim_gpu_caches = lambda: None
    m._compute_subbatch_budget = lambda: 4096
    calls, held, order = [], [], []
    delay = cp.RawKernel('''extern "C" __global__ void delay(unsigned long long ticks) {
        unsigned long long t=clock64(); while(clock64()-t<ticks) {} }''', 'delay')
    delay.compile()

    def callback(batch, stream):
        calls.append((batch.size, batch.step_generation))
        out = cp.empty_like(batch.input('camera.raw'))
        batch.publish('value', out)
        delay((1,), (1,), (np.uint64(60000000),), stream=stream)
        cp.add(batch.input('camera.raw'), batch.calibconst('camera', 'gain'), out=out)

    def submit():
        rec = pool.submit(NS(iter_events=lambda: iter(specs)), None, envelopes(specs),
                          {'camera.raw': detector}, input_windows=(window,), batch_id=7,
                          task=GpuTask(callback, ['camera.raw'], [('camera', 'gain')]),
                          detector_bindings=mapping, task_constants=constants,
                          step_generation=m._step_generation)
        pipeline.enqueue(rec)
        return rec

    def dispatch(dgrams):
        assert pool.active_count == 0
        assert constants.get('camera', 'gain') is old_gain
        assert len(held) == 3
        for i, state in enumerate(held):
            np.testing.assert_array_equal(state.get('value').on_cpu, expected[i] + 1)
        host[...] = 2
        order.append('dispatch')

    m.run = NS(_handle_transition=dispatch)
    monkeypatch.setattr(gpu_events, '_iter_step_events', lambda packet, configs: iter(packet))
    try:
        submit()
        dg = NS(service=lambda: service)
        for envelope in m._handle_steps({0: ([(service, [dg])], [])}):
            held.append(envelope.gpu_state)
            order.append('delivery')
        assert order == ['delivery'] * 3 + ['dispatch']
        assert cp.asnumpy(old_gain) == 1
        if service == TransitionId.BeginStep:
            submit()
            for i, envelope in enumerate(m._flush_event_pool()):
                np.testing.assert_array_equal(envelope.gpu_state.get('value').on_cpu, expected[i] + 2)
            assert calls == [(3, 0), (3, 1)]
        else:
            assert calls == [(3, 0)] and m._step_generation == 0
        for i, state in enumerate(held):
            np.testing.assert_array_equal(state.get('value').on_cpu, expected[i] + 1)
    finally:
        list(pool.flush())
        pipeline.close()
        constants.close()
        assert window.close()


@pytest.mark.parametrize('depth', [1, 2])
def test_owned_copy_mutation_preserves_borrowed_inputs_constants_and_public_fields(depth):
    import cupy as cp
    from psana.gpu.gpu_detector import DenseInputPreparer
    parser, initial, budget, specs, expected, window = fixture(cp, ((0,),) * 3)
    detector = DenseInputPreparer(initial.det_shape, initial.binding, n_slots=depth, budget=budget)
    detector.configure_gather(parser.handle_indices)
    mapping = bindings(parser, detector)
    pool = EventPool(n=depth, budget=budget)
    pipeline = PublicationD2H(8192)
    constants = RequestedConstants([('camera', 'gain')], budget)
    host = np.array(7, np.uint16)
    constants.refresh({'camera': {'gain': host}})
    m = manager(pool, mapping)
    held = []

    def callback(batch, stream):
        raw = batch.input('camera.raw')
        owned = cp.empty_like(raw)
        batch.keepalive(owned)
        cp.copyto(owned, raw)
        owned.fill(99)
        gain_copy = cp.empty_like(batch.calibconst('camera', 'gain'))
        batch.keepalive(gain_copy)
        cp.copyto(gain_copy, batch.calibconst('camera', 'gain'))
        gain_copy.fill(123)
        batch.publish('changed', owned)
        batch.publish('original', raw)
        batch.publish('gain', cp.broadcast_to(batch.calibconst('camera', 'gain'), (batch.size,)).copy())

    def check(ready):
        for i, envelope in enumerate(m._yield_ready(ready)):
            state = envelope.gpu_state
            fields = state.detector('camera').field('raw', 'pixels').on_cpu
            np.testing.assert_array_equal(fields[4], expected[i][1])
            held.append((state, expected[i].copy()))

    try:
        for n in (3, 2, 1, 3):
            ready = pool.begin_retire_next()
            if ready is not None:
                check(ready)
            pool.finish_retire_next()
            rec = pool.submit(NS(iter_events=lambda: iter(specs)), None, envelopes(specs[:n]),
                              {'camera.raw': detector}, input_windows=(window,), batch_id=7,
                              task=GpuTask(callback, ['camera.raw'], [('camera', 'gain')]),
                              detector_bindings=mapping, task_constants=constants)
            pipeline.enqueue(rec)
        for ready in pool.flush():
            check(ready)
        assert len(held) == 9 and host == 7 and cp.asnumpy(constants.get('camera', 'gain')) == 7
        for state, want in held:
            np.testing.assert_array_equal(state.get('original').on_cpu, want)
            np.testing.assert_array_equal(state.get('changed').on_cpu, np.full_like(want, 99))
            assert state.get('gain').on_cpu == 7
    finally:
        list(pool.flush())
        pipeline.close()
        constants.close()
        assert window.close()


def test_noncorrupted_damage_reaches_task_and_corrupted_pixels_are_absent():
    import cupy as cp
    from test_batched_locators import _configs, _input
    from psana.gpu.gpudgram.batch import GpuXtcBatchPool
    from psana.gpu.gpu_input import GpuDetectorBinding
    from psana.gpu.gpu_input_window import InputWindow
    from psana.gpu.gpu_detector import DenseInputPreparer
    from psana.gpu.gpu_kvikio_read import DESC_TIMESTAMP
    configs = _configs()
    parser = GpuXtcBatchPool(configs, field_handles=configs.field_handles(), n_slots=1)
    binding = GpuDetectorBinding('camera', canonical_segment_ids=(0,),
                                field_handles_by_segment={0: configs.resolve('camera', 0, 'raw', 'pixels', stream_id=0)})
    detector = DenseInputPreparer((1, 2, 3), binding, n_slots=1)
    detector.configure_gather(parser.handle_indices)
    data, desc = _input(cp, ['ok', 'damaged', 'corrupted'], [0, 0, 0])
    desc[:, DESC_TIMESTAMP] = [100, 101, 102]
    batch = parser.parse(0, data, desc, cp.cuda.Stream(non_blocking=True))
    window = InputWindow(7, 0, batch, desc)
    specs = tuple(NS(first_desc=i, n_desc=1, timestamp=100+i, batch_event_index=7+3*i) for i in range(3))
    pool = EventPool(n=1)
    pipeline = PublicationD2H(4096)
    calls = []
    def callback(batch, stream):
        calls.append(batch.size)
        batch.publish('raw', batch.input('camera.raw'))
        batch.publish('present', batch.present('camera.raw'))
    try:
        rec = pool.submit(NS(iter_events=lambda: iter(specs)), None, envelopes(specs),
                          {'camera.raw': detector}, input_windows=(window,), batch_id=7,
                          task=GpuTask(callback, ['camera.raw']), detector_bindings={'camera': binding})
        pipeline.enqueue(rec)
        pending = dict(rec.pending_d2h_by_ts)
        list(pool.flush())
        assert calls == [3]
        for ts in (100, 101):
            np.testing.assert_array_equal(pending[ts]['raw'].get(), np.arange(1, 7, dtype=np.uint16).reshape(1, 2, 3))
            assert pending[ts]['present'].get().all()
        assert not pending[102]['raw'].get().any()
        assert not pending[102]['present'].get().any()
    finally:
        list(pool.flush())
        pipeline.close()
        assert window.close()
