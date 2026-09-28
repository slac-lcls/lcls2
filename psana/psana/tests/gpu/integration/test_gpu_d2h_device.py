"""Real copies, delayed terminal ownership, and public batched task delivery."""
import gc
import weakref
from dataclasses import replace
from types import SimpleNamespace as NS

import numpy as np
import pytest

from test_gpu_producer_device import fixture, bindings, envelopes
from test_gpu_allocation_device import available
from psana.gpu import GpuTask
from psana.gpu.context import GpuEventState
from psana.gpu.gpu_d2h import PublicationD2H
from psana.gpu.gpu_stream import EventPool

pytestmark=[pytest.mark.gpu,pytest.mark.skipif(not available(),reason='no CUDA device')]


def submit(pool, specs, window, callback, **kwargs):
    return pool.submit(NS(iter_events=lambda:iter(specs)),None,envelopes(specs),
        input_windows=(window,),batch_id=7,task=GpuTask(callback),**kwargs)


@pytest.mark.parametrize('cap',[0,4096,65536])
def test_real_sparse_mixed_scalar_empty_names_and_bounded_retention(monkeypatch,cap):
    import cupy as cp
    _,_,_,specs,_,window=fixture(cp,((0,),)*3)
    pipeline=PublicationD2H(cap);pool=EventPool(n=1)
    held=[];calls=[];copies=[];completions=[];refs=[]
    class Counted:
        def __init__(self,array):self.array=array
        def __getattr__(self,name):return getattr(self.array,name)
        def get(self,**kw):copies.append((self.shape,kw['blocking']));return self.array.get(**kw)
    real_event=cp.cuda.Event
    for generation in range(3):
        def callback(batch,stream):
            calls.append(batch.size)
            value=cp.array([generation+2,generation+4],dtype=cp.uint32)
            refs.append(weakref.ref(value))
            batch.publish('threshold',value,event_indices=[2,0])
            batch.publish('threshold',cp.full((1,generation+1),generation,dtype=cp.float64),event_indices=[1])
            batch.publish('mask',cp.full((3,2,4),generation%2,dtype=cp.uint8))
            batch.publish('empty',cp.empty((3,0,2),cp.float16))
            if generation==2:batch.publish('new',cp.ones((1,2),cp.complex64),event_indices=[0])
        rec=submit(pool,specs,window,callback)
        rec.publication_batches[:]=[replace(p,array=Counted(p.array)) for p in rec.publication_batches]
        def counted(**kw):completions.append(1);return real_event(**kw)
        with monkeypatch.context() as m:
            m.setattr(cp.cuda,'Event',counted)
            pipeline.enqueue(rec)
        held.append({ts:GpuEventState(keys,detector_names=['camera','other'],pending_d2h=rec.pending_d2h_by_ts[ts])
                     for ts,keys in rec.gpu_results_by_ts.items()})
        list(pool.flush());gc.collect()
        assert pipeline.pinned_bytes<=cap and refs[-1]() is None
    assert calls==[3,3,3] and len(copies)==10 and len(completions)==3
    if cap==0:assert all(blocking for _,blocking in copies)
    pipeline.close();assert pipeline.pinned_bytes==0
    for generation,states in enumerate(held):
        assert states[102].get('threshold').on_cpu==generation+2
        assert states[100].get('threshold').on_cpu.shape==()
        v=states[101].get('threshold').on_cpu
        assert v.shape==(generation+1,) and v.dtype==np.float64 and np.all(v==generation)
        assert states[100].get('empty').on_cpu.shape==(0,2)
        assert states[100].get('mask').on_cpu.dtype==np.uint8
        with pytest.raises(RuntimeError,match='host-delivered'):states[100].get('mask').on_gpu_view()
        with pytest.raises(KeyError):states[100].get('camera.threshold')
    assert window.close()


def test_delayed_copy_joins_before_preallocated_output_reuse():
    import cupy as cp
    _,_,_,specs,_,window=fixture(cp,((0,),)*3)
    pool=EventPool(n=1);pipeline=PublicationD2H(8192)
    pipeline._stream=cp.cuda.Stream(non_blocking=True)
    delay=cp.RawKernel('''extern "C" __global__ void delay(unsigned long long ticks) {
        unsigned long long start=clock64(); while(clock64()-start<ticks) {} }''','delay')
    delay.compile()
    output=cp.empty((3,32),cp.uint32)
    def callback(batch,stream):
        batch.publish('value',output)
        output.fill(1)
    rec=submit(pool,specs,window,callback)
    delay((1,),(1,),(np.uint64(300000000),),stream=pipeline._stream)
    pipeline.enqueue(rec)
    old=rec.pending_d2h_by_ts[100]['value']
    # Reuse must wait for copy completion, even though the producer is done.
    pool.begin_retire_next();pool.finish_retire_next()
    def callback2(batch,stream):
        batch.publish('value',output)
        output.fill(2)
    rec=submit(pool,specs,window,callback2);pipeline.enqueue(rec)
    list(pool.flush())
    np.testing.assert_array_equal(old.get(),np.ones(32,np.uint32))
    np.testing.assert_array_equal(rec.pending_d2h_by_ts[100]['value'].get(),np.full(32,2,np.uint32))
    pipeline.close();assert window.close()


def test_copy_record_and_drain_failure_quarantines_real_device_owners(monkeypatch):
    import cupy as cp
    _,_,_,specs,_,window=fixture(cp,((0,),))
    pool=EventPool(n=1);pipeline=PublicationD2H(4096);refs=[]
    def callback(batch,stream):
        value=cp.full((1,20),17,cp.uint32);refs.append(weakref.ref(value));batch.publish('x',value)
    rec=submit(pool,specs,window,callback)
    real=cp.cuda.Stream(non_blocking=True)
    pipeline._stream=real
    class RetryStream:
        fail=True
        def __getattr__(self,name):return getattr(real,name)
        def synchronize(self):
            if self.fail:raise RuntimeError('copy drain unproven')
            real.synchronize()
    retry=RetryStream()
    # CuPy get requires its native stream, so install the failed drain wrapper
    # only after payload submission, when terminal event recording fails.
    class FailRecord:
        def record(self,stream):
            pipeline._stream=retry
            rec.leases[0]._consumer_done[0]._stream=retry
            raise ValueError('copy record failed')
    with monkeypatch.context() as m:
        m.setattr(cp.cuda,'Event',lambda **kw:FailRecord())
        with pytest.raises(RuntimeError,match='copy drain unproven'):pipeline.enqueue(rec)
    with pytest.raises(RuntimeError,match='copy drain unproven'):list(pool.flush())
    assert not window.close() and refs[0]() is not None
    pool_ref=weakref.ref(pool);del pool;gc.collect();assert pool_ref() is not None
    retry.fail=False
    pool=pool_ref();list(pool.flush());pipeline.close()
    assert window.released and refs[0]() is None


@pytest.mark.parametrize('cap',[0,4096])
def test_serial_public_delivery_exact_keys_retained_events_and_input_access(monkeypatch,cap):
    import cupy as cp
    from psana import DataSource
    from psana.psexp.run import Run
    gain=np.array([3],np.uint32)
    monkeypatch.setattr(Run,'_setup_run_calibconst',lambda run:setattr(run.dsparms,'calibconst',{'jungfrau':{'pixel_gain':(gain,{})}}))
    calls=[]
    def callback(batch,stream):
        calls.append((batch.size,batch.step_generation))
        raw=batch.input('jungfrau.raw')
        batch.publish('timestamp',batch.timestamps_gpu)
        batch.publish('count',cp.full((batch.size,),7,cp.uint32))
        batch.publish('pixel',raw.reshape(batch.size,-1)[:,0].copy())
        assert batch.calibconst('jungfrau','pixel_gain').shape==(1,)
    task=GpuTask(callback,['jungfrau.raw'],[('jungfrau','pixel_gain')])
    ds=DataSource(exp='mfx100848724',run=51,dir='/sdf/data/lcls/ds/prj/public01/xtc',
                  detectors=['jungfrau'],max_events=7,batch_size=3,gpu_det='jungfrau',gpu_fn=task,
                  gpu_d2h_pinned_bytes=cap,n_gpu_streams=2)
    run=next(ds.runs());manager=run._evt_iter;held=[]
    try:
        for event in run.events():
            assert event.gpu.get('count').on_cpu==7
            fields=event.gpu.detector('jungfrau').field('raw','raw').on_cpu
            assert event.gpu.get('pixel').on_cpu==fields[fields.segment_ids[0]].flat[0]
            assert manager._output_d2h.pinned_bytes<=cap
            with pytest.raises(RuntimeError,match='host-delivered'):event.gpu.get('count').on_gpu
            held.append(event)
        assert len(held)==7 and sum(n for n,_ in calls)==7 and max(n for n,_ in calls)<=3
        assert any(n>1 for n,_ in calls) and calls[-1][0]==1
        assert manager._closed and manager._output_d2h.pinned_bytes==0
        for event in held:assert event.gpu.get('timestamp').on_cpu==event.timestamp
    finally:manager.close()
