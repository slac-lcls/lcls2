"""New production scheduling: bounded small groups, batched multi-base parsing."""
import numpy as np
import pytest

from test_gpu_residency_device import available, residency_case
from psana.gpu.gpu_input_group import InputGroupPool
from psana.gpu.gpu_kvikio_read import KvikioGpuReader
from psana.gpu.gpudgram import parser as parser_module
from psana.gpu.gpudgram.parser import LOC_OFFSET, LOC_STATUS, STATUS_FOUND


def group_case(tmp_path, mixed_packet, fast_padding=0):
    case = residency_case(tmp_path, mixed_packet, fast_padding=fast_padding)
    m = case.manager
    m.gpu_reader.close()
    m.gpu_reader = KvikioGpuReader(n_slots=2020, budget=m._gpu_budget)
    m._group_inputs = InputGroupPool(m.gpu_reader)
    # Allow two execution arenas and two arenas retained by small streams.
    from psana.gpu.gpudgram.batch import GpuXtcBatchPool
    m.gpu_xtc_parser.close()
    m.gpu_xtc_parser = GpuXtcBatchPool(
        m.gpu_xtc_parser.configs, field_handles=case.handles, n_slots=5, budget=m._gpu_budget)
    for _, det in m.gpu_detectors.values():
        det._gather_plan = None
        det.configure_gather(m.gpu_xtc_parser.handle_indices)
    m._gpu_budget._limit = 64 * 1024**2
    m._admission_capacity = 16 * 1024**2
    return case


@pytest.mark.gpu
@pytest.mark.skipif(not available(), reason='no CUDA device')
@pytest.mark.parametrize('fast_padding', [0, 16 * 1024])
def test_production_groups_match_pixels_and_keep_batched_parser_launches(
        tmp_path, mixed_packet, monkeypatch, fast_padding):
    import cupy as cp

    case = group_case(tmp_path, mixed_packet, fast_padding)
    m = case.manager
    launches = dict(walk=0, init=0, locate=0)
    parses = []
    for name, label in (('_walk_kernel', 'walk'), ('_init_locators_kernel', 'init'),
                        ('_locate_fields_kernel', 'locate')):
        factory = getattr(parser_module, name)
        def counted(*args, factory=factory, label=label):
            kernel = factory(*args)
            def launch(*args, **kwargs):
                launches[label] += 1
                return kernel(*args, **kwargs)
            return launch
        monkeypatch.setattr(parser_module, name, counted)
    original_parse = m.gpu_xtc_parser.parse_groups
    def parse(reads, *args, **kwargs):
        if reads:
            parses.append(len(reads))
        return original_parse(reads, *args, **kwargs)
    monkeypatch.setattr(m.gpu_xtc_parser, 'parse_groups', parse)
    observed, snapshots = [], {}
    try:
        def check(envelope):
            state = envelope.gpu_state
            dgrams = state._event_dgrams
            event = dgrams.batch_event_index
            owner = dgrams[0].owner
            if owner not in snapshots:
                snapshots[owner] = (cp.asnumpy(owner.batch.data_gpu),
                                     cp.asnumpy(owner.locate(case.handles[0]).rows_gpu))
            raw, rows = snapshots[owner]
            row = rows[dgrams[0].dgram_index]
            assert row[LOC_STATUS] == STATUS_FOUND
            actual = np.frombuffer(raw, np.uint16, case.expected.size,
                                   int(row[LOC_OFFSET])).reshape(case.expected.shape)
            expected = case.expected.copy()
            expected.flat[0] = event
            np.testing.assert_array_equal(actual, expected)
            if 1 in dgrams:
                expected.flat[0] = 1000 + event
                np.testing.assert_array_equal(cp.asnumpy(state._gpu_results['slow.raw'])[0], expected)
            observed.append(event)

        for envelope in m._process_batch({}, {0: (case.packet, [])}, {}):
            check(envelope)
        for envelope in m.finish():
            check(envelope)
        assert observed == list(range(1000))
        assert launches == {label: len(parses) for label in launches}
        assert max(parses) > 1  # independent requests shared each parser launch
        expected_requests = sum(bool(g.size) for g in m._group_schedule.plan.groups)
        assert m.gpu_reader.io_stats()['total_requests'] == expected_requests
        assert not m._group_inputs.live_keys
        assert m._gpu_budget._held == 0
        assert all(owner.released for owner in snapshots)
    finally:
        m.close()


@pytest.mark.gpu
@pytest.mark.skipif(not available(), reason='no CUDA device')
@pytest.mark.parametrize('transition', ['BeginStep', 'EndRun'])
def test_group_transition_drains_deferred_field_consumer(tmp_path, mixed_packet, monkeypatch, transition):
    from types import SimpleNamespace as NS
    from psana.psexp import TransitionId
    from psana.gpu import gpu_events
    import cupy as cp

    case = group_case(tmp_path, mixed_packet)
    m = case.manager
    service = getattr(TransitionId, transition)
    consumer = cp.cuda.Stream(non_blocking=True)
    delayed = cp.RawKernel(r'''
    extern "C" __global__ void delayed_read(const unsigned char* src,
                                           unsigned char* dst) {
        unsigned long long start = clock64();
        while (clock64() - start < 150000000ULL) {}
        dst[0] = src[0];
    }
    ''', 'delayed_read')
    delayed.compile()
    observed = cp.empty(1, dtype=cp.uint8)
    done = cp.cuda.Event(disable_timing=True)
    held, delivered, dispatched = [], [], []

    def consume(envelopes):
        for envelope in envelopes:
            state = envelope.gpu_state
            index = state._event_dgrams.batch_event_index
            delivered.append(index)
            if index == 999:
                owner = state._event_dgrams[0].owner
                raw = owner.batch.data_gpu
                expected = int(raw[0].get())
                child = state._input_lease.acquire_view()
                consumer.wait_event(child.result_ready)
                delayed((1,), (1,), (raw, observed), stream=consumer)
                done.record(consumer)
                child.register_consumer_done(done)
                child.wait_until_safe_to_reuse()
                assert not done.done
                held.append((owner, expected))

    def dispatch(dgrams):
        assert done.done and not m.event_pool.active_count
        assert held[0][0].released
        assert not m._group_inputs.live_keys
        dispatched.append(dgrams[0].service())

    # This test isolates transition ordering; the fixture's detector has no
    # CPU calibration service. Existing multi-owner tests check new constants.
    m.run = NS(_handle_transition=dispatch)
    monkeypatch.setattr(m, '_dispatch_transition', lambda service, dgrams: dispatch(dgrams))
    monkeypatch.setattr(gpu_events, '_iter_step_events', lambda packet, configs: iter(packet))
    dg = NS(service=lambda: service)
    try:
        consume(m._process_batch({}, {0: (case.packet, [])}, {}))
        consume(m._handle_steps({0: ([(service, [dg])], [])}))
        assert delivered == list(range(1000))
        assert dispatched == [service]
        assert int(observed.get()[0]) == held[0][1]
        assert m._gpu_budget._held == 0
    finally:
        consumer.synchronize()
        m.close()


@pytest.mark.gpu
@pytest.mark.skipif(not available(), reason='no CUDA device')
@pytest.mark.parametrize('fail_drain', [False, True])
def test_partial_group_window_setup_detaches_created_children(monkeypatch, fail_drain):
    """A later constructor failure must not leave GC-dependent parser aliases."""
    from types import SimpleNamespace as NS
    from test_batched_gather import _setup, _input
    from psana.gpu import gpu_input_window as iw
    from psana.gpu.gpu_kvikio_read import DESC_DEVICE_OFFSET, DESC_READ_SIZE
    import cupy as cp

    parser, detector, dtype = _setup(cp, False)
    stream = cp.cuda.Stream(non_blocking=True)
    batch, _, _ = _input(cp, parser, stream, [(0,), (1,)], dtype)
    stream.synchronize()
    data, table = batch.data_gpu.get(), batch._test_descriptors.copy()
    batch.retire()
    holds = [0, 0]
    reads = []
    for i, row in enumerate(table):
        start, size = int(row[DESC_DEVICE_OFFSET]), int(row[DESC_READ_SIZE])
        raw = cp.asarray(data[start:start + size])
        row[DESC_DEVICE_OFFSET] = 0
        def retain(i=i):
            holds[i] += 1
            live = True
            def release():
                nonlocal live
                if live:
                    holds[i] -= 1
                    live = False
            return release
        reads.append(NS(data_gpu=raw, desc_table=row.reshape(1, -1), retain_input=retain))
    cp.cuda.Stream.null.synchronize()
    original = iw.InputWindow
    created = []
    def construct(*args, **kwargs):
        if created:
            raise ValueError('injected second window failure')
        window = original(*args, **kwargs)
        created.append(window)
        if fail_drain:
            window.drain = lambda: (_ for _ in ()).throw(RuntimeError('injected drain failure'))
        return window
    monkeypatch.setattr(iw, 'InputWindow', construct)
    try:
        with pytest.raises((ValueError, RuntimeError), match='injected'):
            parser.parse_groups(reads, stream, batch_id=1)
        if fail_drain:
            assert holds == [1, 1] and parser._failed_inputs
            created[0].drain = original.drain.__get__(created[0])
            parser.close()
        assert holds == [0, 0]
        assert created[0].released and created[0].batch is None
        assert all(owner is None for owner in parser._owners)
        assert not parser._failed_inputs
    finally:
        if created:
            created[0].drain = original.drain.__get__(created[0])
        parser.close()
