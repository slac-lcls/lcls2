"""Admission reserves progress before I/O and accounts for buffer growth."""

from types import SimpleNamespace as NS

import numpy as np
import pytest

from psana.gpu.gpu_admission import AdmissionEvent, plan_admission
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


@pytest.mark.parametrize('capacity', [301, 800, 1400, 2200])
@pytest.mark.parametrize('depth', [1, 2])
def test_complete_event_ranges_fit_execution_budget(capacity, depth):
    events = mixed_events()
    plan = plan_admission(events, capacity, max_inflight=depth)
    assert [i for a, b in plan.execution_ranges for i in range(a, b)] == list(range(1000))
    assert plan.inflight * plan.per_execution_bytes <= capacity
    for a, b in plan.execution_ranges:
        assert sum(e.detector_bytes + sum(n for _, n in e.streams)
                   for e in events[a:b]) <= plan.per_execution_bytes


def test_empty_and_zero_byte_events_keep_parser_costs():
    assert plan_admission([], 0).execution_ranges == ()
    events = [AdmissionEvent(((0, 0),), 0)] * 3
    assert plan_admission(events, 8, parser_bytes_per_dgram=2).execution_ranges == ((0, 2), (2, 3))


def test_reduce_overlap_before_rejecting_minimum_event():
    events = [AdmissionEvent(((0, 40),), 60)] * 3
    plan = plan_admission(events, 150, max_inflight=2)
    assert plan.inflight == 1 and plan.execution_ranges == ((0, 1), (1, 2), (2, 3))
    with pytest.raises(GpuMemoryPressureError, match='input=40, parser=8, detector=60'):
        plan_admission(events, 100, parser_bytes_per_dgram=8)


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
