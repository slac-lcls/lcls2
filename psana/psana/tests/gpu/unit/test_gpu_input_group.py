"""Independent group reuse, completion polling and failure-drain invariants."""
from types import SimpleNamespace as NS

import numpy as np
import pytest

from psana.gpu.gpu_budget import _GpuBudget, GpuMemoryPressureError
from psana.gpu.gpu_input_group import InputGroupPool
from psana.gpu.gpu_input_window import InputWindow
from psana.gpu.gpu_kvikio_read import KvikioGpuReader
from psana.gpu.gpu_read_plan import ResolvedDgram, ResolvedFile
from psana.gpu.gpu_stream_read_plan import build_stream_read_plan
from test_gpu_bulk_read import io  # shared injected KvikIO backend
from input_group_checks import assert_pool_invariants


class Token:
    def __init__(self, ready=True):
        self.ready, self.waits, self.queries, self.fail = ready, 0, 0, False

    @property
    def done(self):
        self.queries += 1
        if self.fail:
            raise RuntimeError('injected query failure')
        return self.ready

    def synchronize(self):
        self.waits += 1
        if self.fail:
            raise RuntimeError('injected wait failure')
        self.ready = True


def setup(io, slots=3, capacity=100):
    io.files = {'/fee':bytes(range(64)), '/jf':bytes(range(64))}
    rows = [ResolvedDgram(i, 100+i, s, ResolvedFile(path,0), i*n,n)
            for i in range(3) for s,path,n in ((0,'/fee',2),(5,'/jf',16))]
    plan = build_stream_read_plan(rows,n_events=3,small_target_bytes=8,input_capacity_bytes=32)
    reader = KvikioGpuReader(n_slots=slots,budget=_GpuBudget(capacity))
    return InputGroupPool(reader), plan.groups


def bind(pool, key, producer=None):
    read = pool.read(key)
    batch = NS(data_gpu=read.data_gpu,n_dgrams=len(read.desc_table),walk_done=producer or Token())
    window = InputWindow(key[0],key[1],batch,read.desc_table,
                         release=read.retain_input(),defer_retirement=True)
    pool.bind(key, window)
    return window


@pytest.mark.parametrize('corruption,diagnostic', [
    ('missing_busy_slot', 'busy slots'),
    ('extra_busy_slot', 'busy slots'),
    ('orphan_candidate', 'candidate keys'),
    ('stale_candidate', 'candidate identity'),
])
def test_pool_invariant_checker_rejects_bookkeeping_drift(corruption, diagnostic):
    key = (0, 0)
    state = NS(slot=0)
    pool = NS(_groups={key: state}, _busy_slots={0}, _candidates={})
    assert_pool_invariants(pool)
    pool._candidates[key] = state
    assert_pool_invariants(pool)
    if corruption == 'missing_busy_slot':
        pool._busy_slots.clear()
    elif corruption == 'extra_busy_slot':
        pool._busy_slots.add(1)
    elif corruption == 'orphan_candidate':
        pool._candidates[(0, 1)] = NS(slot=1)
    else:
        pool._candidates[key] = NS(slot=0)
    with pytest.raises(AssertionError, match=diagnostic):
        assert_pool_invariants(pool)


def test_later_jf_reclaims_while_fee_and_earlier_jf_are_busy(io):
    pool, groups = setup(io)
    assert_pool_invariants(pool)
    fee, jf0, jf1 = [pool.issue(0,g) for g in groups[:3]]
    assert_pool_invariants(pool)
    assert all(f.gets == 0 for f in io.futures)
    windows = [bind(pool,k) for k in (fee,jf0,jf1)]
    delayed_fee, delayed_jf = Token(False), Token(False)
    for i in range(3):
        use = pool.take_use(fee,i)
        use.register_consumer_done(delayed_fee)
        use.wait_until_safe_to_reuse()
    old = pool.take_use(jf0,0)
    old.register_consumer_done(delayed_jf)
    old.wait_until_safe_to_reuse()
    later = pool.take_use(jf1,1)
    later.wait_until_safe_to_reuse()
    assert pool.poll() == (jf1,)
    assert_pool_invariants(pool)
    assert delayed_fee.waits == delayed_jf.waits == 0
    assert not windows[0].released and not windows[1].released
    assert bytes(windows[0].batch.data_gpu) == bytes(range(6))
    replacement = pool.issue(0,groups[3])
    assert replacement is not None
    assert pool._groups[replacement].slot == 2  # later slot reused first
    assert_pool_invariants(pool)
    assert pool.reader.memory_bytes()['raw_input_slots'] == 38
    delayed_fee.ready = delayed_jf.ready = True
    assert set(pool.poll()) == {fee,jf0}
    assert_pool_invariants(pool)
    pool.close()  # drains replacement I/O even without parsing
    assert_pool_invariants(pool)
    assert all(f.gets == 1 for f in io.futures)


def test_small_credit_spans_batches_and_waits_for_all_planned_uses(io):
    pool, groups = setup(io)
    key = pool.issue(3,groups[0]); window = bind(pool,key)
    use = pool.take_use(key,0)
    use.wait_until_safe_to_reuse()
    assert pool.issue(4,groups[0]) is None
    assert window.references == 2  # future uses prevent a false zero-consumer gap
    for event in (1,2):
        pool.take_use(key,event).wait_until_safe_to_reuse()
    assert pool.issue(4,groups[0]) is not None
    pool.close()


def test_retained_child_blocks_reuse_after_parent_releases(io):
    pool, groups = setup(io,slots=1)
    key=pool.issue(0,groups[1]); window=bind(pool,key)
    parent=pool.take_use(key,0); child=parent.fork()
    parent.wait_until_safe_to_reuse()
    assert not pool.poll() and pool.issue(0,groups[2]) is None
    with pytest.raises(RuntimeError,match='live consumers'):
        pool.close()
    assert not window.released
    child.wait_until_safe_to_reuse()
    pool.close()
    assert window.released and not pool.live_keys


def test_query_failure_does_not_recycle_failed_owner_or_block_other_reclamation(io):
    pool, groups = setup(io)
    a,b=[pool.issue(0,g) for g in groups[1:3]]
    wa,wb=bind(pool,a),bind(pool,b)
    failure=Token(False)
    use=pool.take_use(a,0);use.register_consumer_done(failure)
    use.wait_until_safe_to_reuse()
    pool.take_use(b,1).wait_until_safe_to_reuse()
    failure.fail=True
    with pytest.raises(RuntimeError,match='query failure'):pool.poll()
    assert_pool_invariants(pool)
    assert a in pool.live_keys and b not in pool.live_keys
    assert not wa.released and wb.released and failure.waits == 0
    failure.fail=False;failure.ready=True
    assert pool.poll() == (a,)
    assert_pool_invariants(pool)
    pool.close()
    assert_pool_invariants(pool)


def test_producer_and_all_consumer_tokens_must_complete(io):
    pool,groups=setup(io)
    key=pool.issue(0,groups[1]);producer=Token(False);window=bind(pool,key,producer)
    first,second=Token(False),Token(False)
    use=pool.take_use(key,0)
    use.register_consumer_done(first);use.register_consumer_done(second)
    use.wait_until_safe_to_reuse()
    producer.ready=first.ready=True
    assert not pool.poll() and not window.released
    second.ready=True
    assert pool.poll() == (key,)
    assert producer.waits == first.waits == second.waits == 0
    pool.close()


def test_budget_failure_precedes_submission_and_returns_group_credit(io):
    pool,groups=setup(io,capacity=21)
    fee=pool.issue(0,groups[0])  # 6 bytes leave only 15 for a 16-byte JF group
    with pytest.raises(GpuMemoryPressureError):pool.issue(0,groups[1])
    assert pool.live_keys == (fee,) and len(io.futures)==1
    pool.close()


@pytest.mark.parametrize('failure',['fail_submit','fail_get','short'])
def test_io_failure_drains_every_started_future_and_prevents_reuse(io,failure):
    pool,groups=setup(io)
    setattr(io,failure,0)
    with pytest.raises(RuntimeError):
        key=pool.issue(0,groups[1]);pool.read(key)
    assert all(f.gets==1 for f in io.futures)
    with pytest.raises(RuntimeError,match='closed or failed'):
        pool.issue(1,groups[1])
    pool.close()


def test_close_cancels_only_untaken_uses_and_drains_tokens(io):
    pool,groups=setup(io)
    key=pool.issue(0,groups[0]);window=bind(pool,key)
    use=pool.take_use(key,0);token=Token(False)
    use.register_consumer_done(token);use.wait_until_safe_to_reuse()
    assert token.waits==0
    pool.close()
    assert_pool_invariants(pool)
    assert token.waits==1 and window.released and not pool.live_keys
    pool.close()
    assert_pool_invariants(pool)
    assert token.waits==1


def test_old_receipt_cannot_retain_reused_generation(io):
    pool,groups=setup(io,slots=1)
    key=pool.issue(0,groups[1]);old=pool.read(key);bind(pool,key)
    pool.take_use(key,0).wait_until_safe_to_reuse();pool.poll()
    pool.issue(0,groups[2])
    with pytest.raises(RuntimeError,match='obsolete'):old.retain_input()
    pool.close()


def test_shutdown_wait_failure_keeps_backing_until_retry(io):
    pool, groups = setup(io, slots=1)
    key = pool.issue(0, groups[1])
    token = Token(False)
    window = bind(pool, key, token)
    token.fail = True
    with pytest.raises(RuntimeError):
        pool.close()
    assert_pool_invariants(pool)
    assert not window.released and pool.reader._input_holds[0] == 2
    token.fail = False
    pool.close()
    assert_pool_invariants(pool)
    assert window.released and not pool.live_keys
    assert pool.reader._input_holds[0] == 0


def test_batch_detaches_before_callback_and_is_not_retired_twice(io):
    pool, groups = setup(io, slots=1)
    key = pool.issue(0, groups[1])
    read = pool.read(key)
    actions = []
    release = read.retain_input()

    def retire():
        actions.append('detach')

    def fail_once():
        assert actions[0] == 'detach'
        actions.append('release')
        if len(actions) == 2:
            raise RuntimeError('injected release failure')
        release()

    batch = NS(data_gpu=read.data_gpu, n_dgrams=1, walk_done=Token(), retire=retire)
    window = InputWindow(0, 0, batch, read.desc_table, release=fail_once,
                         defer_retirement=True)
    pool.bind(key, window)
    with pytest.raises(RuntimeError, match='release failure'):
        pool.take_use(key, 0).wait_until_safe_to_reuse()
    assert not window.released and pool.live_keys == (key,)
    assert pool.poll() == (key,)
    assert actions == ['detach', 'release', 'release']
    pool.close()


def test_failed_parser_keeps_reader_hold_through_group_pool_shutdown(io):
    from psana.gpu.gpudgram.batch import GpuXtcBatchPool

    pool, groups = setup(io, slots=1)
    key = pool.issue(0, groups[1])
    parser = GpuXtcBatchPool.__new__(GpuXtcBatchPool)
    parser._owners, parser._failed_inputs, parser._next_window_id = [None], [], 0
    stream = Token(False)
    stream.fail = True

    def fail(*args):
        raise ValueError('injected parser failure')

    parser.parse = fail
    with pytest.raises(RuntimeError, match='wait failure'):
        pool.parse(key, parser, stream)
    assert pool.reader._input_holds[0] == 2
    pool.close()
    assert pool.reader._input_holds[0] == 1  # quarantined parser owns raw input
    pool.reader.trim_free_buffers()
    assert pool.reader.memory_bytes()['raw_input_slots'] == 16
    stream.fail = False
    parser.close()
    assert pool.reader._input_holds[0] == 0
    pool.reader.trim_free_buffers()
    assert pool.reader.memory_bytes()['raw_input_slots'] == 0


def test_batch_submission_uses_plan_without_repolling_and_preserves_reads(io, monkeypatch):
    pool, groups = setup(io)
    selected = groups[1:]
    slots = pool.plan_slots(selected)
    assert_pool_invariants(pool)

    def unexpected_poll():
        raise AssertionError('batch submission repeated retirement polling')

    monkeypatch.setattr(pool, 'poll', unexpected_poll)
    keys = pool.issue_groups(0, selected, slots=slots)
    assert_pool_invariants(pool)
    assert len(keys) == len(selected) == len(io.futures) == 3
    assert pool._busy_slots == set(slots)
    assert [call[2:] for call in io.calls if call[0] == 'submit'] == [(0, 16), (16, 16), (32, 16)]
    pool.close()
    assert_pool_invariants(pool)
    assert not pool._busy_slots and all(f.gets == 1 for f in io.futures)


def test_batch_conflicts_are_rejected_before_any_io(io):
    pool, groups = setup(io)
    assert pool.issue_groups(0, groups[1:3], slots=(0, 0)) is None
    assert_pool_invariants(pool)
    assert not io.futures
    with pytest.raises(ValueError, match='duplicate'):
        pool.issue_groups(0, (groups[1], groups[1]), slots=(0, 1))
    assert_pool_invariants(pool)
    with pytest.raises(IndexError):
        pool.issue_groups(0, groups[1:3], slots=(0, 99))
    assert_pool_invariants(pool)
    assert not io.futures
    key = pool.issue(0, groups[1], slot_id=0)
    assert pool.issue_groups(1, groups[1:3], slots=(0, 1)) is None
    assert_pool_invariants(pool)
    assert len(io.futures) == 1 and pool.live_keys == (key,)
    pool.close()
    assert_pool_invariants(pool)


def test_partial_batch_submission_failure_keeps_started_reads_owned(io):
    pool, groups = setup(io)
    io.fail_submit = 1
    with pytest.raises(RuntimeError, match='submission'):
        pool.issue_groups(0, groups[1:], slots=(0, 1, 2))
    assert_pool_invariants(pool)
    assert len(io.futures) == 1 and pool._busy_slots == {0}
    assert pool.live_keys == ((0, groups[1].group_id),)
    pool.close()
    assert_pool_invariants(pool)
    assert not pool._busy_slots and not pool.live_keys
    assert all(f.gets == 1 for f in io.futures)


def test_retirement_candidates_skip_live_windows_and_retry_delayed_cuda(io, monkeypatch):
    pool, groups = setup(io)
    a, b = pool.issue_groups(0, groups[1:3], slots=(0, 1))
    assert_pool_invariants(pool)
    wa, wb = bind(pool, a), bind(pool, b)
    original = wa.poll

    def still_owned():
        raise AssertionError('polled a window with a live planned reference')

    monkeypatch.setattr(wa, 'poll', still_owned)
    done = Token(False)
    use = pool.take_use(b, 1)
    use.register_consumer_done(done)
    use.wait_until_safe_to_reuse()
    assert not pool.poll()
    assert set(pool._candidates) == {b}
    assert_pool_invariants(pool)
    assert done.waits == 0
    done.ready = True
    assert pool.poll() == (b,) and not pool._candidates
    assert_pool_invariants(pool)
    assert not wa.released and wb.released
    monkeypatch.setattr(wa, 'poll', original)
    pool.take_use(a, 0).wait_until_safe_to_reuse()
    assert pool.poll() == (a,)
    assert_pool_invariants(pool)
    pool.close()
    assert_pool_invariants(pool)


def test_consumer_thread_only_enqueues_and_stale_notification_cannot_reclaim_new_group(io):
    from concurrent.futures import ThreadPoolExecutor
    pool, groups = setup(io, slots=1)
    key = pool.issue(0, groups[1])
    bind(pool, key)
    old = pool._groups[key]
    use = pool.take_use(key, 0)
    with ThreadPoolExecutor(max_workers=1) as executor:
        executor.submit(use.wait_until_safe_to_reuse).result()
    assert pool.live_keys == (key,) and not pool._candidates
    assert_pool_invariants(pool)
    assert pool.poll() == (key,)
    assert_pool_invariants(pool)
    assert pool.issue(0, groups[1]) == key
    assert_pool_invariants(pool)
    bind(pool, key)
    pool._retirement_queue.put(old)
    assert not pool.poll() and not pool._candidates
    assert pool.live_keys == (key,) and pool._busy_slots == {0}
    assert_pool_invariants(pool)
    pool.close()
    assert_pool_invariants(pool)
    assert pool._retirement_queue.empty() and not pool._candidates


def test_raw_hold_release_failure_preserves_candidate_and_busy_slot_until_retry(io):
    pool, groups = setup(io, slots=1)
    key = pool.issue(0, groups[1])
    bind(pool, key)
    state = pool._groups[key]
    release = state.release_raw
    calls = []

    def fail_once():
        calls.append('release')
        if len(calls) == 1:
            raise RuntimeError('raw hold release failed')
        release()

    state.release_raw = fail_once
    pool.take_use(key, 0).wait_until_safe_to_reuse()
    with pytest.raises(RuntimeError, match='raw hold release failed'):
        pool.poll()
    assert_pool_invariants(pool)
    assert pool._groups[key] is state and pool._candidates[key] is state
    assert pool._busy_slots == {0} and state.release_raw is fail_once
    assert pool.reader._input_holds[0] == 1
    assert pool.poll() == (key,)
    assert_pool_invariants(pool)
    assert not pool._groups and not pool._candidates and not pool._busy_slots
    assert pool.reader._input_holds[0] == 0
    pool.close()
    assert_pool_invariants(pool)


def test_failed_notification_registration_does_not_leave_untracked_bound_window(io):
    pool, groups = setup(io, slots=1)
    key = pool.issue(0, groups[1])
    read = pool.read(key)
    batch = NS(data_gpu=read.data_gpu, n_dgrams=1, walk_done=Token())
    rejected = InputWindow(key[0], key[1], batch, read.desc_table,
        release=read.retain_input(), defer_retirement=True)
    rejected.notify_when_retirable(lambda: None)
    with pytest.raises(RuntimeError, match='already registered'):
        pool.bind(key, rejected)
    assert_pool_invariants(pool)
    assert pool._groups[key].window is None
    assert rejected.close()
    window = bind(pool, key)
    pool.take_use(key, 0).wait_until_safe_to_reuse()
    assert pool.poll() == (key,) and window.released
    assert_pool_invariants(pool)
    pool.close()
    assert_pool_invariants(pool)
