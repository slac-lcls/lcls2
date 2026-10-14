"""Integrated bulk parser/gather cleanup through production manager boundaries."""
from types import SimpleNamespace as NS

import numpy as np
import pytest

from gpu_group_fixture import available, group_case
from psana.gpu import gpu_events
from psana.gpu.gpu_input import GpuEventDgrams
from psana.psexp import TransitionId

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(
    not available(), reason='no CUDA device')]


@pytest.mark.parametrize('stop', ['max_events', 'generator_close', 'read_failure', 'input_failure'])
def test_group_cleanup_with_real_parser_and_gather(tmp_path, mixed_packet, monkeypatch, stop):
    case = group_case(tmp_path, mixed_packet)
    m = case.manager
    # Force several executions so early close and a later read failure occur
    # while earlier parsed inputs and deliveries are live.
    m._admission_capacity = 4 * 1024**2
    windows = []
    parse = m.gpu_xtc_parser.parse_groups

    def track(*args, **kwargs):
        window = parse(*args, **kwargs)
        windows.extend(window)
        return window
    monkeypatch.setattr(m.gpu_xtc_parser, 'parse_groups', track)
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
            if windows and not any(c is False for c in calls):
                calls.append(False)
                r, size, future = pending.futures[0]
                pending.futures[0] = (r, size, FailedFuture(future))
            return pending
        monkeypatch.setattr(m.gpu_reader, '_submit_read', fail_second_read)
    elif stop == 'input_failure':
        original = GpuEventDgrams.from_windows
        def fail_after_inputs(*args, **kwargs):
            original(*args, **kwargs)
            raise RuntimeError('injected input binding failure')
        monkeypatch.setattr(GpuEventDgrams, 'from_windows', fail_after_inputs)

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
    case = group_case(tmp_path, mixed_packet)
    m = case.manager
    saved, windows, transitions = [], [], []
    parse = m.gpu_xtc_parser.parse_groups

    def track(*args, **kwargs):
        window = parse(*args, **kwargs)
        windows.extend(window)
        return window
    monkeypatch.setattr(m.gpu_xtc_parser, 'parse_groups', track)
    monkeypatch.setattr(gpu_events, '_iter_step_events', lambda packet, configs: iter(packet))
    def dispatch(dgrams):
        assert all(w.released for w in windows)
        assert not m.event_pool.active_count
        transitions.append(dgrams[0].service())
    m.run = NS(_handle_transition=dispatch)

    def process(packet):
        samples = []
        from itertools import chain
        for envelope in chain(m._process_batch({}, {0: (packet, [])}, {}), m._flush_event_pool()):
            state = envelope.gpu_state
            assert not state._gpu_results
            i = state._event_dgrams.batch_event_index
            if i % 100 == 99:
                raw = case.expected.copy()
                raw.flat[0] = 1000 + i
                np.testing.assert_array_equal(state.detector('slow').field('raw', 'arrayRaw').on_cpu[1], raw)
                samples.append(i)
            saved.append(state)
        return samples

    def transition(service):
        dg = NS(timestamp=lambda: 999999, service=lambda: service)
        return list(m._process_batch({}, {}, {0: ([(service, [dg])], [])}))

    try:
        assert process(case.packet) == list(range(99, 1000, 100))
        assert transition(TransitionId.BeginStep) == []
        # A smaller EB tail keeps the raw input unchanged with the same configured plan.
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
