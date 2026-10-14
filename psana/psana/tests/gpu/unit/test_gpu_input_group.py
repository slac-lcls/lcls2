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


def test_later_jf_reclaims_while_fee_and_earlier_jf_are_busy(io):
    pool, groups = setup(io)
    fee, jf0, jf1 = [pool.issue(0,g) for g in groups[:3]]
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
    assert delayed_fee.waits == delayed_jf.waits == 0
    assert not windows[0].released and not windows[1].released
    assert bytes(windows[0].batch.data_gpu) == bytes(range(6))
    replacement = pool.issue(0,groups[3])
    assert replacement is not None
    assert pool._groups[replacement].slot == 2  # later slot reused first
    assert pool.reader.memory_bytes()['raw_input_slots'] == 38
    delayed_fee.ready = delayed_jf.ready = True
    assert set(pool.poll()) == {fee,jf0}
    pool.close()  # drains replacement I/O even without parsing
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
    assert a in pool.live_keys and b not in pool.live_keys
    assert not wa.released and wb.released and failure.waits == 0
    failure.fail=False;failure.ready=True
    assert pool.poll() == (a,)
    pool.close()


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
    assert token.waits==1 and window.released and not pool.live_keys
    pool.close()
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
    assert not window.released and pool.reader._input_holds[0] == 2
    token.fail = False
    pool.close()
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
