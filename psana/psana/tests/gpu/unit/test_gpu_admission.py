"""Admission reserves progress before I/O and accounts for buffer growth."""

from types import SimpleNamespace as NS

import numpy as np
import pytest

from psana.gpu.gpu_admission import AdmissionEvent, plan_admission, _resident_candidates
from psana.gpu.gpu_budget import _GpuBudget, GpuMemoryPressureError, allocation_growth_bytes
from psana.gpu.gpu_allocation import owned_empty
from psana.gpu.gpu_events import GpuEventManager
from psana.gpu.gpu_kvikio_read import KvikioGpuReader
from psana.gpu import gpu_calib


def mixed_events():
    # Fast stream 0 each event; slow stream 1 every 100th event. Byte units
    # deliberately small so the expected capacity arithmetic is reviewable.
    return [AdmissionEvent(((0, 1), (1, 100)) if i % 100 == 99 else ((0, 1),),
                           200 if i % 100 == 99 else 0)
            for i in range(1000)]


def _assert_admission_fits(events, plan, capacity, parser_bytes=0):
    assert [i for a, b in plan.execution_ranges for i in range(a, b)] == list(range(len(events)))
    expected_resident = sum(n + parser_bytes for e in events for s, n in e.streams
                            if s in plan.resident_streams)
    assert plan.resident_bytes == expected_resident
    for a, b in plan.execution_ranges:
        cost = sum(e.detector_bytes + sum(n + parser_bytes for s, n in e.streams
                                          if s not in plan.resident_streams)
                   for e in events[a:b])
        assert cost <= plan.per_execution_bytes
    assert plan.resident_bytes + plan.inflight * plan.per_execution_bytes <= capacity


@pytest.mark.parametrize('small_stream,large_stream', [(0, 1), (7, 2)])
@pytest.mark.parametrize('capacity,inflight', [(2600, 2), (2300, 1)])
def test_small_dgrams_win_even_when_sparse_large_stream_costs_less(
        small_stream, large_stream, capacity, inflight):
    # Small: 1000 input + 1000 parser; large: 700 input + 10 parser.
    # Both would fit individually as resident, but not together. The old
    # total-footprint order chose the large-dgram stream and excluded the small.
    events = [AdmissionEvent(((small_stream, 1), (large_stream, 70))
                             if i % 100 == 99 else ((small_stream, 1),),
                             200 if i % 100 == 99 else 0)
              for i in range(1000)]
    candidates = _resident_candidates(events, 1)
    assert [c.stream_id for c in candidates] == [small_stream, large_stream]
    assert candidates[0].resident_bytes > candidates[1].resident_bytes
    plan = plan_admission(events, capacity, parser_bytes_per_dgram=1,
                          max_inflight=2, allow_residency=True)
    assert plan.resident_streams == (small_stream,)
    assert plan.inflight == inflight
    _assert_admission_fits(events, plan, capacity, 1)


def test_priority_uses_mean_present_size_not_minimum_or_batch_event_count():
    events = [AdmissionEvent(((0, 1), (1, 60), (2, 40)), 0),
              AdmissionEvent(((0, 99),), 0)] + [AdmissionEvent((), 0)] * 100
    candidates = _resident_candidates(events, 3)
    assert [c.stream_id for c in candidates] == [2, 0, 1]
    assert [c.average_dgram_bytes for c in candidates] == [40, 50, 60]


def test_priority_preserves_fractional_mean_and_large_integer_precision():
    large = 2**60
    events = [AdmissionEvent(((0, large + 1), (1, large)), 0),
              AdmissionEvent(((1, large),), 0)]
    assert [c.stream_id for c in _resident_candidates(events, 0)] == [1, 0]
    events = [AdmissionEvent(((0, 1), (1, 1)), 0),
              AdmissionEvent(((0, 2), (1, 1)), 0)]
    assert [c.stream_id for c in _resident_candidates(events, 0)] == [1, 0]


def test_equal_mean_ties_use_full_footprint_then_stream_id():
    events = [AdmissionEvent(((2, 4), (1, 4), (0, 4)), 0),
              AdmissionEvent(((1, 4),), 0)]
    candidates = _resident_candidates(events, 3)
    assert [c.stream_id for c in candidates] == [0, 2, 1]
    assert [c.resident_bytes for c in candidates] == [7, 7, 14]
    reordered = [AdmissionEvent(tuple(reversed(e.streams)), e.detector_bytes)
                 for e in reversed(events)]
    assert candidates == _resident_candidates(reordered, 3)


def test_zero_size_rows_do_not_improve_priority_but_keep_parser_charge():
    events = [AdmissionEvent(((0, 0), (1, 3), (2, 0)), 0),
              AdmissionEvent(((0, 4), (2, 0)), 0)]
    candidates = _resident_candidates(events, 2)
    assert [c.stream_id for c in candidates] == [1, 0]
    assert [c.average_dgram_bytes for c in candidates] == [3, 4]
    assert [c.resident_bytes for c in candidates] == [5, 8]
    plan = plan_admission(events, 100, parser_bytes_per_dgram=2, allow_residency=True)
    assert plan.resident_streams == (1, 0)
    _assert_admission_fits(events, plan, 100, 2)


def test_equal_mean_and_input_bytes_tie_includes_parser_footprint():
    events = [AdmissionEvent(((0, 4), (1, 4)), 0), AdmissionEvent(((0, 0),), 0)]
    candidates = _resident_candidates(events, 2)
    assert [c.stream_id for c in candidates] == [1, 0]
    assert [c.resident_bytes for c in candidates] == [6, 8]


def test_all_empty_stream_remains_execution_scoped():
    events = [AdmissionEvent(((0, 0),), 0)] * 3
    assert _resident_candidates(events, 2) == ()
    plan = plan_admission(events, 8, parser_bytes_per_dgram=2, allow_residency=True)
    assert plan.resident_streams == ()
    assert plan.execution_ranges == ((0, 2), (2, 3))
    _assert_admission_fits(events, plan, 8, 2)


def test_unaffordable_small_stream_does_not_block_later_candidate():
    events = [AdmissionEvent(((0, 1), (1, 10)), 5)] + [AdmissionEvent(((0, 1),), 0)] * 99
    plan = plan_admission(events, 30, max_inflight=2, allow_residency=True)
    assert plan.resident_streams == (1,)
    _assert_admission_fits(events, plan, 30)


def test_priority_does_not_rescue_an_oversized_complete_event():
    # One easy event must not hide the later oversized joint-detector event.
    events = [AdmissionEvent(((0, 1),), 0), AdmissionEvent(((0, 1), (1, 100)), 200)]
    with pytest.raises(GpuMemoryPressureError, match='event 1 cannot fit alone'):
        plan_admission(events, 300, allow_residency=True)


def test_residency_disabled_keeps_execution_only_accounting():
    events = mixed_events()
    plan = plan_admission(events, 2200, allow_residency=False)
    assert plan.resident_streams == () and plan.resident_bytes == 0
    _assert_admission_fits(events, plan, 2200)


@pytest.mark.parametrize('capacity,expected_depth,expected_reasons', [
    (2200, 2, ('fits_with_execution', 'insufficient_capacity')),
    (1400, 1, ('fits_with_execution', 'insufficient_capacity')),
    (800, 2, ('insufficient_capacity', 'insufficient_capacity')),
    (3000, 2, ('fits_with_execution', 'fits_with_execution')),
])
def test_residency_decisions_record_real_fit_and_concurrency(
        capacity, expected_depth, expected_reasons):
    plan = plan_admission(mixed_events(), capacity, allow_residency=True)
    decisions = plan.residency_decisions
    assert tuple(d.reason for d in decisions) == expected_reasons
    assert tuple(d.candidate.stream_id for d in decisions if d.admitted) == plan.resident_streams
    assert plan.inflight == expected_depth
    resident, depth = 0, 2
    for d in decisions:
        assert d.resident_bytes_before == resident
        assert d.inflight_before == depth
        assert d.capacity_bytes == capacity
        assert d.admitted == (d.required_bytes <= capacity)
        assert d.required_bytes == resident + d.candidate.resident_bytes + d.inflight * d.working_bytes
        if d.admitted:
            resident += d.candidate.resident_bytes
        depth = d.inflight
    assert resident == plan.resident_bytes and depth == plan.inflight


def test_disabled_or_empty_residency_has_no_candidate_decisions():
    assert plan_admission(mixed_events(), 2200).residency_decisions == ()
    assert plan_admission([], 0, allow_residency=True).residency_decisions == ()
    assert plan_admission([AdmissionEvent(((0, 0),), 0)], 10,
                          parser_bytes_per_dgram=1, allow_residency=True).residency_decisions == ()


def test_full_fast_admission_with_two_slow_events_per_execution():
    plan = plan_admission(mixed_events(), 2200, max_inflight=2, allow_residency=True)
    assert plan.resident_streams == (0,)
    assert plan.resident_bytes == 1000
    assert plan.inflight == 2 and plan.per_execution_bytes == 600
    assert len(plan.execution_ranges) == 5
    assert plan.execution_ranges[0][0] == 0 and plan.execution_ranges[-1][1] == 1000
    assert all(sum(i % 100 == 99 for i in range(a, b)) == 2
               for a, b in plan.execution_ranges)


def test_tight_budget_keeps_order_and_shortens_input_groups():
    events = mixed_events()
    plan = plan_admission(events, 800, max_inflight=2, allow_residency=True)
    assert plan.resident_streams == ()
    assert len(plan.execution_ranges) > 5
    assert [i for a, b in plan.execution_ranges for i in range(a, b)] == list(range(1000))


def test_full_fast_residency_can_reduce_overlap_to_preserve_progress():
    plan = plan_admission(mixed_events(), 1400, max_inflight=2, allow_residency=True)
    assert plan.resident_streams == (0,) and plan.inflight == 1
    assert plan.per_execution_bytes == 400


def test_reduce_overlap_before_rejecting_minimum_event():
    events = [AdmissionEvent(((0, 40),), 60)] * 3
    plan = plan_admission(events, 150, max_inflight=2)
    assert plan.inflight == 1 and plan.execution_ranges == ((0, 1), (1, 2), (2, 3))
    with pytest.raises(GpuMemoryPressureError, match='input=40, parser=8, detector=60'):
        plan_admission(events, 100, parser_bytes_per_dgram=8)


def test_resident_inputs_include_parser_and_preserve_fast_only_events():
    events = [AdmissionEvent(((0, 1),), 0)] + [AdmissionEvent(((0, 1), (1, 40)), 80)] * 2
    plan = plan_admission(events, 163, parser_bytes_per_dgram=10, max_inflight=1,
                          allow_residency=True)
    assert plan.resident_streams == (0,) and plan.resident_bytes == 33
    assert plan.execution_ranges == ((0, 2), (2, 3))


def test_no_floor_and_no_calibration_required_for_admission():
    manager = GpuEventManager.__new__(GpuEventManager)
    manager._gpu_budget = _GpuBudget(100)
    manager._gpu_budget.reserve(30)  # fixed/config ownership, once
    manager.dsparms = NS(n_gpu_streams=2)
    manager.gpu_detectors = {}
    assert manager._compute_subbatch_budget() == 30
    assert manager._admission_capacity == 60


def test_growth_peak_and_future_allocations_are_reserved():
    budget = _GpuBudget(1100)
    budget.reserve(200)  # two existing 100-byte arrays
    assert allocation_growth_bytes([(300, 100), (500, 100), (20, 100)]) == 800
    hold = budget.hold(800, margin=100)
    assert budget.available() == 100
    # Unrelated allocation cannot spend the parser/detector progress credit.
    with pytest.raises(GpuMemoryPressureError):
        budget.reserve(101)
    with hold:
        budget.reserve(300)
        assert budget.committed() == 500  # old allocation remains during growth
        budget.release(100)
    assert budget.committed() == 400 and budget._held == 600
    with hold:
        budget.reserve(500)
        assert budget.committed() == 900
        budget.release(100)
    hold.close()
    assert budget.committed() == 800 and budget.available() == 300
    hold.close()
    assert budget.available() == 300


def test_failed_allocation_rolls_back_into_admission_credit():
    budget = _GpuBudget(100)
    hold = budget.hold(80)
    with hold:
        budget.reserve(80)
        budget.release(80)  # allocator failure
    assert budget.committed() == 0 and budget.available() == 20
    hold.close()
    assert budget.available() == 100


def test_existing_capacity_is_not_charged_twice():
    budget = _GpuBudget(100)
    budget.reserve(90)
    hold = budget.hold(allocation_growth_bytes([(40, 90)]), margin=10)
    with hold:
        pass
    hold.close()
    assert budget.committed() == 90


def test_growth_refuses_old_plus_new_peak_before_allocator():
    calls = []
    reader = KvikioGpuReader.__new__(KvikioGpuReader)
    reader._slot_bufs = [np.zeros(60, dtype=np.uint8)]
    reader._budget = _GpuBudget(100)
    reader._budget.reserve(60)
    reader.cp = NS(uint8=np.uint8, empty=lambda *a, **k: calls.append(True))
    with pytest.raises(GpuMemoryPressureError):
        reader._ensure_slot_buffer(0, 80)
    assert not calls and reader._budget.committed() == 60


def test_reader_trimming_preserves_live_input_and_io():
    reader = KvikioGpuReader.__new__(KvikioGpuReader)
    reader._input_holds = {0: 1}
    reader._pending = [NS(slot_id=1)]
    reader._budget = _GpuBudget(100)
    reader._slot_bufs = [owned_empty(np, 10, np.uint8, reader._budget, 'reader')
                        for _ in range(3)]
    reader.trim_free_buffers()
    assert reader._slot_bufs[0] is not None and reader._slot_bufs[1] is not None
    assert reader._slot_bufs[2] is None and reader._budget.committed() == 20


def test_trimming_checks_occupied_executions_including_retirement():
    from psana.gpu.gpu_stream import EventPool
    pool = EventPool.__new__(EventPool)
    pool._n, pool._slots = 2, [None, object()]
    manager = GpuEventManager.__new__(GpuEventManager)
    manager.event_pool = pool
    calls = []
    manager.gpu_reader = NS(trim_free_buffers=lambda: calls.append('reader'))
    manager.gpu_xtc_parser, manager.gpu_detectors = None, {}
    assert len(pool) == 2 and pool.active_count == 1
    with pytest.raises(RuntimeError, match='active executions'):
        manager._trim_gpu_caches()
    assert not calls
    pool._slots[1] = None
    assert len(pool) == 2 and pool.active_count == 0
    manager._trim_gpu_caches()
    assert calls == ['reader']


def test_parser_trimming_retains_owned_rows_and_fixed_charge():
    from psana.gpu.gpudgram.batch import GpuXtcBatchPool, _GpuXtcSlotBuffers
    parser = GpuXtcBatchPool.__new__(GpuXtcBatchPool)
    parser.cp, parser._budget = np, _GpuBudget(1000)
    parser._budget.reserve(40)  # fixed configuration remains legacy in Stage 2
    parser._owners = [object(), None]
    parser._slots = [_GpuXtcSlotBuffers(np, parser._budget,
                                       dgram_records=owned_empty(np, (1, 10), np.uint64,
                                                                parser._budget, 'parser'))
                     for _ in range(2)]
    live = parser._slots[0]
    parser.trim_free_buffers()
    assert parser._slots[0] is live and live.memory_bytes == 80
    assert parser._slots[1].memory_bytes == 0
    assert parser._slots[1].budget is parser._budget
    assert parser._budget.committed() == 120


def test_fixed_upload_reserves_before_transfer_and_rolls_back(monkeypatch):
    budget = _GpuBudget(100)
    calls = []
    def upload(target, array):
        assert budget.committed() + budget._held == 80
        calls.append(True)
        if len(calls) == 2:
            raise RuntimeError('upload failed')
        return array.copy()
    drained = []
    monkeypatch.setattr(gpu_calib, '_cupy', lambda: NS(
        empty=np.empty, cuda=NS(get_current_stream=lambda: NS(synchronize=lambda: drained.append(True)))))
    monkeypatch.setattr(np, "copyto", upload)
    with pytest.raises(RuntimeError, match='upload failed'):
        gpu_calib._upload_fixed_arrays((np.zeros(10, np.float32),) * 2, budget)
    assert drained == [True] and budget.committed() == 0


def test_fixed_upload_failure_with_unproven_completion_stays_charged(monkeypatch):
    budget = _GpuBudget(100)
    def fail(*args):
        raise RuntimeError('CUDA failure')
    monkeypatch.setattr(gpu_calib, '_cupy', lambda: NS(
        empty=np.empty, cuda=NS(get_current_stream=lambda: NS(synchronize=fail))))
    with pytest.raises(RuntimeError):
        gpu_calib._upload_fixed_arrays((np.zeros(10, np.float32),), budget)
    assert budget.committed() == 40 and len(budget._failed_allocations) == 1
    budget._failed_allocations[0][0].synchronize = lambda: None
    budget.drain_failed_allocations()
    budget.drain_failed_allocations()  # retry is idempotent
    assert budget.committed() == 0 and not budget._failed_allocations


def test_ipc_follower_does_not_double_subtract_shared_constants():
    manager = GpuEventManager.__new__(GpuEventManager)
    manager._gpu_budget = _GpuBudget(1000)
    manager._gpu_budget.reserve(20)  # only this rank's Configure tables
    manager.gpu_detectors = {'jf': (None, NS(
        _is_calib_follower=True, memory_bytes=lambda: {'constants': 400, 'geometry': 0}))}
    manager.dsparms = NS(n_gpu_streams=2)
    assert manager._compute_subbatch_budget() == 440  # (1000-20-100)/2


def test_source_presence_controls_detector_cost():
    from psana.gpu.gpu_batch import GpuBatchView
    from test_core import _make_batch
    manager = GpuEventManager.__new__(GpuEventManager)
    manager.gpu_detectors = {
        'fast': (None, NS(binding=NS(has_sources=lambda s: 0 in s), estimate_subbatch_bytes=lambda n: 10*n)),
        'slow': (None, NS(binding=NS(has_sources=lambda s: 1 in s), estimate_subbatch_bytes=lambda n: 1000*n)),
    }
    view = GpuBatchView(_make_batch(2, descs_per_event=1, stream_ids=[0], bd_size=5))
    costs = manager._event_memory(view)
    assert [e.detector_bytes for e in costs] == [10, 10]


def test_admission_failure_precedes_issue_and_releases_failed_issue_hold():
    manager = GpuEventManager.__new__(GpuEventManager)
    manager._pending_gpu_read = None
    manager.dsparms = NS(gpu_bulk_read=False)
    manager.dm = None
    budget = _GpuBudget(100)
    calls = []
    manager.gpu_reader = NS(issue_batch=lambda *a, **kw: calls.append(True))
    manager._reserve_gpu_subbatch = lambda *a: budget.hold(101)
    with pytest.raises(GpuMemoryPressureError):
        manager._issue_gpu_read(None, 0)
    assert not calls and budget._held == 0
    manager._reserve_gpu_subbatch = lambda *a: budget.hold(90)
    def fail(*args, **kwargs):
        raise RuntimeError('I/O submission failure')
    manager.gpu_reader.issue_batch = fail
    with pytest.raises(RuntimeError, match='I/O submission failure'):
        manager._issue_gpu_read(None, 0)
    assert budget._held == 0 and manager._pending_gpu_read is None


def test_pressure_drains_consumer_delivery_before_trimming_and_retry():
    manager = GpuEventManager.__new__(GpuEventManager)
    log, attempts = [], []
    manager.event_pool = NS(begin_retire_next=lambda: 'old',
                            finish_retire_next=lambda: log.append('finish old'), next_slot_id=0)
    manager._d2h_pipelines = {}
    manager._yield_ready = lambda *a, **k: iter(('old event',))
    def flush():
        yield 'other event'
        log.append('other consumers done')
    manager._flush_event_pool = flush
    def trim():
        assert log[-1] == 'other consumers done'
        log.append('trim')
    manager._trim_gpu_caches = trim
    pending = object()
    def issue(*args):
        attempts.append(True)
        if len(attempts) == 1:
            raise GpuMemoryPressureError('cached capacity')
        assert log[-1] == 'trim'
        return pending
    manager._issue_gpu_read = issue
    processing = manager._retire_issue_and_yield(None)
    assert next(processing) == 'old event' and log == []
    assert next(processing) == 'other event' and log == ['finish old']
    with pytest.raises(StopIteration) as finished:
        next(processing)
    assert finished.value.value is pending and len(attempts) == 2


def test_wait_and_submission_failures_return_unused_progress_credit():
    manager = GpuEventManager.__new__(GpuEventManager)
    budget = _GpuBudget(100)
    pending = object()
    manager._pending_gpu_read = pending
    manager._gpu_read_reservation = budget.hold(80)
    def fail(*args, **kwargs):
        raise RuntimeError('injected failure')
    manager.gpu_reader = NS(wait_batch=fail)
    with pytest.raises(RuntimeError):
        manager._wait_gpu_read(pending)
    assert budget.available() == 100 and manager._pending_gpu_read is None
    manager._gpu_read_reservation = budget.hold(80)
    manager.gpu_detectors, manager.gpu_xtc_parser = {}, None
    def submit(*args, **kwargs):
        budget.reserve(20)  # allocated storage retained by failed parser owner
        raise RuntimeError('injected failure')
    manager.event_pool = NS(submit=submit)
    with pytest.raises(RuntimeError):
        manager._submit_gpu(None, None, [])
    assert budget.committed() == 20 and budget.available() == 80
