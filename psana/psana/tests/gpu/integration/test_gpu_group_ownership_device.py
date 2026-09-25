"""Production group ownership under public field views and byte pressure."""
import numpy as np
import pytest

from psana.gpu import gpu_events
from psana.gpu.gpu_budget import GpuMemoryPressureError
from psana.gpu.gpu_input import GpuDetectorBinding
from psana.gpu.gpu_stream import EventPool
from gpu_group_fixture import available, group_case

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(
    not available(), reason='no CUDA device')]


def ownership_case(tmp_path, mixed_packet, monkeypatch, depth, mib):
    case = group_case(tmp_path, mixed_packet)
    m = case.manager
    m.event_pool = EventPool(n=depth)
    m.dsparms.n_gpu_streams = depth
    m.gpu_detector_bindings = {
        name: GpuDetectorBinding(name, canonical_segment_ids=(1,),
            field_handles_by_segment={1: handle},
            field_handles_by_name={('raw', 'arrayRaw'): {1: handle}})
        for name, handle in zip(('fast', 'slow'), case.handles)}
    budget = m._gpu_budget
    budget._limit = budget.committed() + mib * 1024**2
    m._admission_capacity = mib * 1024**2
    samples = []
    for name in ('reserve', 'hold'):
        original = getattr(budget, name)

        def checked(*args, original=original, **kwargs):
            result = original(*args, **kwargs)
            used = budget.committed() + budget._held
            assert used <= budget.limit()
            samples.append(used)
            return result
        monkeypatch.setattr(budget, name, checked)
    case.budget_samples = samples
    return case


def packet(case, mixed_packet, n, base=0):
    assert base % 100 == 0
    return mixed_packet(n_events=n, fast_size=case.fast_size,
        slow_size=case.slow_size, timestamp_base=case.timestamp_base + base,
        offsets=(base * case.fast_size, (base // 100) * case.slow_size))


def field(state, name='fast'):
    return state.detector(name).field('raw', 'arrayRaw')


def expected(case, event, slow=False):
    result = case.expected.copy()
    result.flat[0] = event + (1000 if slow else 0)
    return result


def assert_closed(m):
    assert m._closed and not m.event_pool.active_count
    assert not m._group_inputs.live_keys and not m.gpu_reader._pending
    assert not any(m.gpu_reader._input_holds.values())
    assert not m._gpu_budget._held
    assert m._gpu_budget.committed() <= m._gpu_budget.limit()


@pytest.mark.parametrize('depth', [1, 2])
@pytest.mark.parametrize('mib', [4, 8])
@pytest.mark.parametrize('d2h', [0, 7])
def test_tight_budget_two_batches_preserve_pixels_and_partial_tail(
        tmp_path, mixed_packet, monkeypatch, depth, mib, d2h):
    case = ownership_case(tmp_path, mixed_packet, monkeypatch, depth, mib)
    m = case.manager
    if d2h:
        m._d2h_pipelines = {'slow.calib': gpu_events._D2hPipeline('slow.calib', d2h)}
    seen, slow_seen, windows = [], [], []
    original = m.gpu_xtc_parser.parse_groups

    def track(*args, **kwargs):
        result = original(*args, **kwargs)
        windows.extend(result)
        return result
    monkeypatch.setattr(m.gpu_xtc_parser, 'parse_groups', track)

    def consume(envelopes):
        for envelope in envelopes:
            event = int(envelope.dgrams[0].timestamp()) - case.timestamp_base
            state = envelope.gpu_state
            seen.append(event)
            if event % 100 in (0, 99) or event == 502:
                np.testing.assert_array_equal(field(state).on_cpu[1], expected(case, event))
            if event % 100 == 99:
                np.testing.assert_array_equal(state.get('slow.calib').on_cpu[0],
                                              expected(case, event, True).astype(np.float32))
                slow_seen.append(event)

    try:
        consume(m._process_batch({}, {0: (packet(case, mixed_packet, 300), [])}, {}))
        consume(m._process_batch({}, {0: (packet(case, mixed_packet, 203, 300), [])}, {}))
        consume(m.finish())
        assert seen == list(range(503)) and slow_seen == [99, 199, 299, 399, 499]
        assert len({w.batch_id for w in windows}) == 2
        assert all(w.released and w.batch is None for w in windows)
        assert case.budget_samples and max(case.budget_samples) <= m._gpu_budget.limit()
        assert_closed(m)
        m.close()
        assert_closed(m)
    finally:
        m.close()


@pytest.mark.parametrize('depth', [1, 2])
@pytest.mark.parametrize('mib', [4, 8])
@pytest.mark.parametrize('recovery', ['resume', 'close'])
def test_open_public_field_context_preserves_owners_under_pressure(
        tmp_path, mixed_packet, monkeypatch, depth, mib, recovery):
    import cupy as cp
    case = ownership_case(tmp_path, mixed_packet, monkeypatch, depth, mib)
    m = case.manager
    stream = cp.cuda.Stream(non_blocking=True)
    context = values = None
    owners, seen = [], []

    def consume(envelopes):
        nonlocal context, values
        for envelope in envelopes:
            index = envelope.gpu_state._event_dgrams.batch_event_index
            seen.append(index)
            if index == 99 and context is None:
                state = envelope.gpu_state
                owners.extend(state._event_dgrams.input_windows)
                context = field(state).on_gpu_view(stream)
                values = context.__enter__()

    try:
        batch = packet(case, mixed_packet, 203)
        consume(m._process_batch({}, {0: (batch, [])}, {}))
        consume(m._flush_event_pool())
        assert seen == list(range(203)) and len(owners) == 2
        # A field context conservatively retains both event input owners.
        # Advancing delivery, trimming, and failed admission cannot revoke it.
        with pytest.raises(GpuMemoryPressureError, match='stream credit'):
            list(m._process_batch({}, {0: (batch, [])}, {}))
        m._trim_gpu_caches()
        assert all(not w.released for w in owners)
        np.testing.assert_array_equal(values[1].get(), expected(case, 99))
        assert m._gpu_budget.committed() > 0 and not m._gpu_budget._held
        if recovery == 'close':
            with pytest.raises(RuntimeError, match='live consumers'):
                m.close()
            assert not m._closed and all(not w.released for w in owners)
        context.__exit__(None, None, None)
        values = None
        stream.synchronize()
        m._group_inputs.poll()
        assert all(w.released for w in owners)
        if recovery == 'resume':
            resumed = list(m._process_batch({}, {0: (batch, [])}, {}))
            resumed.extend(m._flush_event_pool())
            assert [e.dgrams[0].timestamp() for e in resumed] == list(
                range(case.timestamp_base, case.timestamp_base + 203))
        m.close()
        assert_closed(m)
    finally:
        if context is not None:
            context.__exit__(None, None, None)
        values = None
        stream.synchronize()
        m.close()


@pytest.mark.parametrize('depth', [1, 2])
def test_public_field_consumers_on_two_streams_do_not_order_group_reclamation(
        tmp_path, mixed_packet, monkeypatch, depth):
    import cupy as cp
    case = ownership_case(tmp_path, mixed_packet, monkeypatch, depth, 8)
    m = case.manager
    streams = [cp.cuda.Stream(non_blocking=True) for _ in range(2)]
    done = [cp.cuda.Event(disable_timing=True) for _ in streams]
    output = [cp.empty(case.expected.shape, dtype=cp.uint16) for _ in streams]
    delay = cp.RawKernel(r'''
    extern "C" __global__ void delayed_copy(const unsigned short* src,
        unsigned short* dst, int n, unsigned long long ticks) {
        unsigned long long start = clock64();
        while (clock64() - start < ticks) {}
        for (int i = 0; i < n; ++i) dst[i] = src[i];
    }
    ''', 'delayed_copy')
    delay.compile()
    windows, retained, seen = [], [], []
    parse = m.gpu_xtc_parser.parse_groups

    def track(*args, **kwargs):
        result = parse(*args, **kwargs)
        windows.extend(result)
        return result
    monkeypatch.setattr(m.gpu_xtc_parser, 'parse_groups', track)

    def consume(envelopes):
        for envelope in envelopes:
            state = envelope.gpu_state
            index = state._event_dgrams.batch_event_index
            seen.append(index)
            if index == 99:
                retained.extend(state._event_dgrams.input_windows)
                for i, stream in enumerate(streams):
                    with field(state).on_gpu_view(stream) as values:
                        delay((1,), (1,), (values[1], output[i], np.int32(case.expected.size),
                            np.uint64(2000000000 if i == 0 else 0)), stream=stream)
                    done[i].record(stream)
                    values = None  # Python's with target otherwise retains an alias.

    try:
        consume(m._process_batch({}, {0: (packet(case, mixed_packet, 203), [])}, {}))
        consume(m._flush_event_pool())
        assert seen == list(range(203)) and len(retained) == 2
        done[1].synchronize()
        assert not done[0].done
        m._group_inputs.poll()
        assert not done[0].done  # poll must not wait on the older consumer
        assert all(not w.released for w in retained)
        later = [w for w in windows if w not in retained]
        assert later and all(w.released for w in later)
        np.testing.assert_array_equal(output[1].get(), expected(case, 99))
        done[0].synchronize()
        m._group_inputs.poll()
        assert all(w.released for w in windows)
        np.testing.assert_array_equal(output[0].get(), expected(case, 99))
        m.close()
        assert_closed(m)
    finally:
        for stream in streams:
            stream.synchronize()
        m.close()
