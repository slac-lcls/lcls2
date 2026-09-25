"""Integrated bulk parser/gather cleanup through production manager boundaries."""
from types import SimpleNamespace as NS

import numpy as np
import pytest

from test_gpu_residency_device import available, residency_case
from psana.gpu import gpu_events
from psana.gpu.gpu_detector import _CanonicalGatherPlan
from psana.psexp import TransitionId

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(
    not available(), reason='no CUDA device')]


@pytest.mark.parametrize('stop', ['max_events', 'generator_close', 'read_failure', 'gather_failure'])
@pytest.mark.parametrize('d2h', [0, 7])
def test_resident_cleanup_with_real_parser_and_gather(tmp_path, mixed_packet, monkeypatch, stop, d2h):
    case = residency_case(tmp_path, mixed_packet)
    m = case.manager
    if d2h:
        m._d2h_pipelines = {'slow.calib': gpu_events._D2hPipeline('slow.calib', d2h)}
    windows = []
    parse = m.gpu_xtc_parser.parse_window

    def track(*args, **kwargs):
        window = parse(*args, **kwargs)
        windows.append(window)
        return window
    monkeypatch.setattr(m.gpu_xtc_parser, 'parse_window', track)
    if stop == 'max_events':
        m.dsparms.max_events = 203
    elif stop == 'read_failure':
        issue = m.gpu_reader._submit_read
        calls = []

        class FailedFuture:
            def __init__(self, future):
                self.future = future

            def get(self):
                self.future.get()  # real I/O completes before reporting failure
                raise OSError('injected completed read failure')

        def fail_second_read(*args, **kwargs):
            pending = issue(*args, **kwargs)
            calls.append(len(pending.futures))
            if len(calls) == 2:
                r, size, future = pending.futures[0]
                pending.futures[0] = (r, size, FailedFuture(future))
            return pending
        monkeypatch.setattr(m.gpu_reader, '_submit_read', fail_second_read)
    elif stop == 'gather_failure':
        gather = _CanonicalGatherPlan.gather

        def fail_after_gather(self, *args, **kwargs):
            gather(self, *args, **kwargs)
            raise RuntimeError('injected queued gather failure')
        monkeypatch.setattr(_CanonicalGatherPlan, 'gather', fail_after_gather)

    iterator = m._process_batch({}, {0: (case.packet, [])}, {})
    saved = []
    try:
        if stop.endswith('failure'):
            with pytest.raises((OSError, RuntimeError), match='injected'):
                list(iterator)
        elif stop == 'generator_close':
            saved.append(next(iterator).gpu_state)
            iterator.close()
        else:
            timestamps = []
            for envelope in iterator:
                timestamps.append(envelope.dgrams[0].timestamp())
                saved.append(envelope.gpu_state)
            assert len(timestamps) == 203
            assert timestamps == list(range(timestamps[0], timestamps[0] + 203))
        m.close()
        m.close()  # idempotent after both normal and error retirement
        assert windows and all(w.released and w.batch is None for w in windows)
        assert not m.event_pool.active_count and not m.gpu_reader._pending
        assert not any(m.gpu_reader._input_holds.values())
        assert m._gpu_budget._held == 0
        assert m._gpu_budget.committed() <= m._gpu_budget.limit()
        for state in saved:
            with pytest.raises(RuntimeError):
                state._input_lease.require_active()
    finally:
        iterator.close()
        m.close()


def test_beginstep_updates_after_drain_and_endrun_finishes_once(
        tmp_path, mixed_packet, monkeypatch):
    import cupy as cp
    case = residency_case(tmp_path, mixed_packet)
    m, detector = case.manager, case.manager.gpu_detectors['slow'][1]
    saved, windows, transitions = [], [], []
    parse = m.gpu_xtc_parser.parse_window

    def track(*args, **kwargs):
        window = parse(*args, **kwargs)
        windows.append(window)
        return window
    monkeypatch.setattr(m.gpu_xtc_parser, 'parse_window', track)
    monkeypatch.setattr(gpu_events, '_iter_step_events', lambda packet, configs: iter(packet))
    peds, gains = np.full(54, 7, np.float32), np.full(54, 2, np.float32)

    def constants(*args, **kwargs):
        assert all(w.released for w in windows)
        assert not m.event_pool.active_count
        return peds, gains
    monkeypatch.setattr(gpu_events, '_compute_calib_constants_cpu', constants)

    def dispatch(dgrams):
        assert all(w.released for w in windows)
        assert not m.event_pool.active_count
        transitions.append(dgrams[0].service())
    m.run = NS(_handle_transition=dispatch)

    def process(packet):
        samples = []
        for envelope in m._process_batch({}, {0: (packet, [])}, {}):
            state = envelope.gpu_state
            if state._gpu_results:
                i = state._event_dgrams.batch_event_index
                raw = case.expected.copy()
                raw.flat[0] = 1000 + i
                expected = raw.astype(np.float32)
                if transitions:
                    expected = (expected - 7) * 2
                np.testing.assert_array_equal(cp.asnumpy(state._gpu_results['slow.calib'])[0], expected)
                samples.append(i)
            saved.append(state)
        return samples

    def transition(service):
        dg = NS(timestamp=lambda: 999999, service=lambda: service)
        return list(m._process_batch({}, {}, {0: ([(service, [dg])], [])}))

    try:
        assert process(case.packet) == list(range(99, 1000, 100))
        assert transition(TransitionId.BeginStep) == []
        # A smaller EB tail uses updated constants with the same configured plan.
        tail = mixed_packet(n_events=203, fast_size=case.fast_size,
                            slow_size=case.slow_size,
                            timestamp_base=case.timestamp_base)
        assert process(tail) == [99, 199]
        assert transition(TransitionId.EndRun) == []
        assert m._done and all(w.released for w in windows)
        assert list(m.finish()) == [] and list(m.finish()) == []
        assert transitions == [TransitionId.BeginStep, TransitionId.EndRun]
        assert m._gpu_budget._held == 0
        for state in saved:
            with pytest.raises(RuntimeError):
                state._input_lease.require_active()
    finally:
        m.close()
