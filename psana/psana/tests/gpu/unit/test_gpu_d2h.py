"""Host delivery accounting, exact row mapping, and failed-copy ownership."""
import gc
import sys
import weakref
from types import SimpleNamespace as NS

import numpy as np
import pytest

from psana.gpu.context import GpuEventState, SlotLease
from psana.gpu.gpu_d2h import PublicationD2H, validate_pinned_bytes
from psana.gpu.gpu_producer import PublicationBatch


class Event:
    def __init__(self, log): self.log = log
    def record(self, stream):
        self.log.append('record')
        if stream.fail_record: raise RuntimeError('record failed')
    def query(self): return True
    def synchronize(self): self.log.append('event-sync')


class Stream:
    def __init__(self, log):
        self.log = log
        self.fail_record = self.fail_sync = False
        self.done = True
    def wait_event(self, event): self.log.append('wait')
    def synchronize(self):
        self.log.append('stream-sync')
        if self.fail_sync: raise RuntimeError('drain failed')


class Array:
    def __init__(self, data, log):
        self.data, self.log = data, log
        self.shape, self.dtype, self.nbytes = data.shape, data.dtype, data.nbytes
        self.flags = data.flags
        self.fail = False
    def get(self, *, out, stream, blocking):
        self.log.append(('copy', self.shape, blocking))
        out[...] = self.data
        if self.fail: raise RuntimeError('copy failed')


@pytest.fixture
def cuda(monkeypatch):
    log, allocations = [], []
    stream = Stream(log)
    def pinned(n):
        allocations.append(n)
        return bytearray(n)
    monkeypatch.setitem(sys.modules, 'cupy', NS(cuda=NS(
        Stream=lambda **kw:stream, Event=lambda **kw:Event(log),
        PinnedMemory=pinned, PinnedMemoryPointer=lambda owner,offset:owner)))
    return log, allocations, stream


def record(cuda, specifications):
    log, _, _ = cuda
    ready = Event(log)
    lease = SlotLease(ready)
    pubs = []
    for name, data, stamps in specifications:
        a = Array(np.asarray(data), log)
        pubs.append(PublicationBatch(name, a, a.shape, a.dtype, a.nbytes, lease,
                                    tuple(range(len(stamps))), tuple(stamps)))
    return NS(publication_batches=pubs, pending_d2h_by_ts={}, gpu_results_by_ts={}, lease=lease)


def test_one_copy_per_group_one_event_exact_sparse_mixed_rows(cuda):
    log, allocations, _ = cuda
    r = record(cuda, [('count', np.array([7,9],np.uint32), [102,100]),
                      ('count', np.arange(3,dtype=np.float64).reshape(1,3), [101]),
                      ('mask', np.ones((3,2),np.uint8), [100,101,102]),
                      ('empty', np.empty((3,0,4),np.int16), [100,101,102])])
    pipeline = PublicationD2H(3*4096)
    pipeline.enqueue(r)
    assert len([x for x in log if isinstance(x,tuple)]) == 3
    assert log.count('wait') == log.count('record') == 1
    assert allocations == [4096]*3 and pipeline.pinned_bytes == 3*4096
    assert all(v is None for values in r.gpu_results_by_ts.values() for v in values.values())
    states = {ts:GpuEventState(keys, ['jf'], pending_d2h=r.pending_d2h_by_ts[ts])
              for ts,keys in r.gpu_results_by_ts.items()}
    v=states[100].get('count').on_cpu
    assert isinstance(v,np.ndarray) and v.shape==() and v.dtype==np.uint32 and v==9
    assert states[100].get('count').on_cpu is v
    np.testing.assert_array_equal(states[101].get('count').on_cpu,[0,1,2])
    assert states[102].get('empty').on_cpu.shape == (0,4)
    with pytest.raises(RuntimeError,match='host-delivered'):states[100].get('count').on_gpu
    with pytest.raises(KeyError):states[100].get('jf.count')
    r.lease.wait_until_safe_to_reuse()
    pipeline.close()
    assert pipeline.pinned_bytes == 0
    assert states[102].get('count').on_cpu == 7


@pytest.mark.parametrize('cap', [0,4095,4096])
def test_aggregate_cap_retained_rows_and_oversize_fallback(cuda,cap):
    log, allocations, _=cuda
    pipeline=PublicationD2H(cap)
    r=record(cuda,[('a',np.ones((2,8)),[100,101]),('b',np.ones((2,8)),[100,101])])
    pipeline.enqueue(r)
    first=r.pending_d2h_by_ts[100]['a']
    del r.pending_d2h_by_ts[101]
    copies=[v for v in log if isinstance(v,tuple)]
    assert [v[2] for v in copies] == ([False,True] if cap==4096 else [True,True])
    assert pipeline.pinned_bytes <= cap
    r.lease.wait_until_safe_to_reuse()
    r2=record(cuda,[('new',np.ones((1,2000)),[103])]);pipeline.enqueue(r2)
    assert [v for v in log if isinstance(v,tuple)][-1][2] is True
    assert first.get().shape == (8,)
    r2.lease.wait_until_safe_to_reuse();pipeline.close()
    assert pipeline.pinned_bytes==0


def test_ignored_rows_reuse_capacity_without_extra_allocation(cuda):
    pipeline=PublicationD2H(4096)
    for i in range(5):
        r=record(cuda,[('a',np.full((3,4),i,np.uint8),[100,101,102])])
        pipeline.enqueue(r)
        r.lease.wait_until_safe_to_reuse()
        r.pending_d2h_by_ts.clear()
        gc.collect()
    assert cuda[1]==[4096] and pipeline.pinned_bytes==4096
    pipeline.close()


def test_empty_and_absent_publications_do_not_touch_cuda(cuda,monkeypatch):
    r=record(cuda,[('empty',np.empty((2,0,3),np.float32),[101,100])])
    monkeypatch.setitem(sys.modules,'cupy',None)
    pipeline=PublicationD2H()
    pipeline.enqueue(r)
    assert not cuda[1] and pipeline._stream is None
    assert r.pending_d2h_by_ts[100]['empty'].get().shape==(0,3)
    pipeline.close()
    assert r.pending_d2h_by_ts[101]['empty'].get().dtype==np.float32
    pipeline=PublicationD2H();pipeline.enqueue(NS(publication_batches=[]));pipeline.close()


@pytest.mark.parametrize('failure', ['copy','record','drain'])
def test_partial_failure_retains_destinations_and_never_installs_handoff(cuda,failure):
    pipeline=PublicationD2H(4096)
    r=record(cuda,[('a',np.ones((2,4)),[100,101]),('b',np.ones((2,4)),[100,101])])
    if failure in ('copy','drain'):r.publication_batches[1].array.fail=True
    if failure=='record':cuda[2].fail_record=True
    if failure=='drain':cuda[2].fail_sync=True
    with pytest.raises(RuntimeError,match='failed'):pipeline.enqueue(r)
    assert not r.pending_d2h_by_ts and not r.gpu_results_by_ts
    guard=r.lease._consumer_done[0]
    if failure=='drain':
        assert len(guard._owners)==2
        refs=[weakref.ref(x) for x in guard._owners]
        with pytest.raises(RuntimeError,match='drain failed'):r.lease.wait_until_safe_to_reuse()
        gc.collect();assert all(ref() is not None for ref in refs)
        cuda[2].fail_sync=False
    r.lease.wait_until_safe_to_reuse()
    assert not guard._owners
    pipeline.close();assert pipeline.pinned_bytes==0


def test_changed_metadata_rejected_before_any_transfer(cuda):
    r=record(cuda,[('a',np.ones((2,4)),[100,101])])
    r.publication_batches[0].array.shape=(2,2,2)
    pipeline=PublicationD2H()
    with pytest.raises(ValueError,match='metadata changed'):pipeline.enqueue(r)
    assert not cuda[0] and not cuda[1] and not r.lease._consumer_done


def test_retained_host_token_does_not_retain_publication_or_device_array(cuda):
    r=record(cuda,[('a',np.ones((2,4)),[100,101])]);pipeline=PublicationD2H()
    refs=[weakref.ref(r.publication_batches[0]),weakref.ref(r.publication_batches[0].array)]
    pipeline.enqueue(r);token=r.pending_d2h_by_ts[100]['a']
    r.lease.wait_until_safe_to_reuse();r.publication_batches.clear();gc.collect()
    assert all(ref() is None for ref in refs)
    pipeline.close();assert pipeline.pinned_bytes==0
    np.testing.assert_array_equal(token.get(),np.ones(4))


@pytest.mark.parametrize('bad', [-1,True,1.5,'12',None])
def test_invalid_pinned_cap(bad):
    with pytest.raises((TypeError,ValueError)):validate_pinned_bytes(bad)


def test_retained_results_and_close_are_safe_during_concurrent_get(cuda):
    from concurrent.futures import ThreadPoolExecutor
    pipeline=PublicationD2H(4096)
    r=record(cuda,[('a',np.arange(300).reshape(3,100),[100,101,102])]);pipeline.enqueue(r)
    tokens=[r.pending_d2h_by_ts[ts]['a'] for ts in (100,101,102)]
    with ThreadPoolExecutor(4) as pool:
        futures=[pool.submit(t.get) for t in tokens]*1+[pool.submit(pipeline.close)]
        for future in futures:future.result()
    for i,t in enumerate(tokens):np.testing.assert_array_equal(t.get(),np.arange(300).reshape(3,100)[i])
    assert pipeline.pinned_bytes==0
