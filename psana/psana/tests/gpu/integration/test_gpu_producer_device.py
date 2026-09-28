"""Real producer stream ordering, selected identities and terminal ownership."""
import gc
import weakref
from types import SimpleNamespace as NS

import numpy as np
import pytest

from test_batched_gather import _setup, _input, _gpu_available
from test_multiowner_gather import fixture_inputs
from psana.gpu import GpuTask
from psana.gpu.gpu_budget import _GpuBudget
from psana.gpu.gpu_detector import DenseInputPreparer
from psana.gpu.gpu_input import GpuDetectorBinding
from psana.gpu.gpu_input_window import InputWindow
from psana.gpu.gpu_stream import EventPool
from psana.gpu.gpu_task import RequestedConstants

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(not _gpu_available(), reason='no CUDA device')]


def envelopes(specs):
    return [NS(dgrams=[NS(timestamp=lambda ts=s.timestamp: ts)]) for s in specs]


def bindings(parser, detector):
    return {'camera': GpuDetectorBinding('camera',
        canonical_segment_ids=detector.canonical_segment_ids,
        field_handles_by_segment={}, field_handles_by_name={
            (alg, field): {s: parser.configs.resolve('camera', s, alg, field)
                           for s in detector.canonical_segment_ids}
            for alg,field in [('raw','pixels'),('raw','counter')]})}


def fixture(cp, selections=((1,0), (2,), (0,), (), (1,0))):
    budget = _GpuBudget(16*1024**2)
    parser, detector, dtype = _setup(cp, budget=budget)
    producer = cp.cuda.Stream(non_blocking=True)
    batch, events, expected = _input(cp, parser, producer, selections, dtype)
    window = InputWindow(7, 0, batch, batch._test_descriptors)
    specs = tuple(e.event for e in events)
    return parser, detector, budget, specs, expected, window


def test_selected_tail_multiple_inputs_constants_fields_and_launches(monkeypatch):
    import cupy as cp
    from psana.gpu import gpu_detector as gd
    parser, camera, budget, specs, expected, window = fixture(cp)
    mapping = bindings(parser, camera)
    other = GpuDetectorBinding('other', canonical_segment_ids=(0,),
        field_handles_by_segment={0: parser.configs.resolve('other',0,'raw','pixels')})
    mapping['other'] = other
    preparers = {'camera.raw': camera,
                 'other.raw': DenseInputPreparer((1,3,100), other, budget=budget, n_slots=1)}
    constants = RequestedConstants([('camera','gain')], budget)
    constants.refresh({'camera': {'gain': np.array(2, np.uint32)}})
    counter = cp.RawKernel('''extern "C" __global__ void counter(
        const unsigned char* raw, unsigned long long size,
        const unsigned long long* loc, unsigned long long index, unsigned int* out) {
        const unsigned long long* r=loc+index*11;
        *out=0xffffffff;
        if(r[10]==1 && r[1]==2 && r[2]==0 && r[9]==4 && r[8]<=size && 4<=size-r[8])
            *out=*(const unsigned int*)(raw+r[8]);
    }''', 'counter')
    counter.compile()
    calls, contexts, gathers = [], [], []
    gather = gd._batched_gather_kernel
    def counted(dtype):
        kernel = gather(dtype)
        def launch(*args, **kw):
            gathers.append(1)
            return kernel(*args, **kw)
        return launch
    monkeypatch.setattr(gd, '_batched_gather_kernel', counted)
    def callback(evt, stream):
        assert cp.cuda.get_current_stream().ptr == stream.ptr
        calls.append((evt.timestamp,evt.batch_event_index,evt.batch_id,evt.run,evt.step_generation))
        contexts.append(evt)
        assert evt.segment_ids('camera') == (9,4,8)
        evt.publish('gain_alias', evt.calibconst('camera','gain'))
        raw = evt.input('camera.raw')
        if raw is None:
            assert evt.present('camera.raw') is None
            evt.publish('other_pixels', evt.input('other.raw'))
        else:
            assert evt.input('other.raw') is None
            evt.publish('pixels', raw)
            evt.publish('presence', evt.present('camera.raw'))
        fields = evt.field('camera','raw','counter')
        assert fields is evt.field('camera','raw','counter')
        for field in fields:
            out = cp.empty((), cp.uint32)
            evt.publish('counter_'+str(field.segment_id), out)
            counter((1,), (1,), (field.data_gpu,np.uint64(field.raw_nbytes),
                    field.locator_rows,np.uint64(field.row),out), stream=stream)
    task = GpuTask(callback, ['camera.raw','other.raw',('camera','raw','counter')],
                   [('camera','gain')])
    pool = EventPool(n=1)
    record = pool.submit(NS(iter_events=lambda: iter(specs)), None, envelopes(specs[:4]),
        preparers, input_windows=(window,), batch_id=7, task=task,
        detector_bindings=mapping, task_constants=constants, run=51, step_generation=3)
    assert not window.close()
    assert calls == [(100,7,7,51,3),(101,10,7,51,3),(102,13,7,51,3)]
    assert len(gathers)==2 and window.batch._locators=={}
    assert pool.begin_retire_next() is record
    for ts, outputs in record.publications_by_ts.items():
        assert cp.asnumpy(outputs['gain_alias'].array)==2
        for name,pub in outputs.items():
            if name.startswith('counter_'):assert cp.asnumpy(pub.array)==ts-100
        if ts!=101:np.testing.assert_array_equal(outputs['pixels'].array.get(),expected[0 if ts==100 else 1])
    for ctx in contexts:
        with pytest.raises(RuntimeError,match='during its callback'):ctx.input('camera.raw')
    pool.finish_retire_next()
    assert window.released and not record.producer_owners and not record.publications_by_ts
    constants.close()


@pytest.mark.parametrize('failure', ['none', 'callback', 'record'])
def test_delayed_scratch_only_callback_and_exception_drain(monkeypatch, failure):
    import cupy as cp
    _, detector, _, specs, _, window = fixture(cp, ((0,),))
    kernel = cp.RawKernel('''extern "C" __global__ void work(unsigned int* x,
        unsigned long long ticks) { unsigned long long t=clock64();
        while(clock64()-t<ticks) {} *x=37; }''', 'work')
    kernel.compile()
    refs, observed = [], []
    class Owner:
        def __init__(self):self.array=cp.empty((),cp.uint32)
        def __del__(self):observed.append(int(cp.asnumpy(self.array)))
    def callback(evt, stream):
        owner = Owner()
        refs.append(weakref.ref(owner))
        evt.keepalive(owner)
        kernel((1,),(1,),(owner.array,np.uint64(60000000)),stream=stream)
        if failure=='callback':raise ValueError('callback after launch')
    pool=EventPool(n=1)
    args=dict(input_windows=(window,), batch_id=7, task=GpuTask(callback))
    if failure=='record':
        def fail_record(*args):raise ValueError('completion record after launch')
        monkeypatch.setattr(cp.cuda,'Event',lambda **kw:NS(record=fail_record))
    if failure!='none':
        with pytest.raises(ValueError,match='after launch'):
            pool.submit(NS(iter_events=lambda:iter(specs)),None,envelopes(specs),**args)
        gc.collect()
        assert pool.active_count==0 and observed==[37] and refs[0]() is None
    else:
        record=pool.submit(NS(iter_events=lambda:iter(specs)),None,envelopes(specs),**args)
        gc.collect()
        assert refs[0]() is not None and observed==[] and record.publications_by_ts=={}
        list(pool.flush())
        assert observed==[37] and refs[0]() is None
    assert window.close()


def test_failed_completion_keeps_registered_owners_and_inputs_until_retry():
    import cupy as cp
    _, _, _, specs, _, window=fixture(cp, ((0,),))
    pool=EventPool(n=1)
    real=pool.next_stream
    class RetryStream:
        fail=True
        def __getattr__(self,name):return getattr(real,name)
        def __enter__(self):return real.__enter__()
        def __exit__(self,*args):return real.__exit__(*args)
        def synchronize(self):
            if self.fail:raise RuntimeError('unproven completion')
            real.synchronize()
    stream=RetryStream();pool._streams[0]=stream
    refs=[]
    def callback(evt, stream):
        array=cp.empty(5,cp.uint32)
        refs.append(weakref.ref(array))
        evt.publish('out',array)
        array.fill(37)
        raise ValueError('after launch')
    with pytest.raises(RuntimeError,match='unproven completion'):
        pool.submit(NS(iter_events=lambda:iter(specs)),None,envelopes(specs),
                    input_windows=(window,),batch_id=7,task=GpuTask(callback))
    gc.collect()
    assert pool.active_count==1 and refs[0]() is not None
    assert not window.close()
    pool_ref=weakref.ref(pool)
    del pool
    gc.collect()
    assert pool_ref() is not None and refs[0]() is not None
    pool=pool_ref()
    stream.fail=False
    list(pool.flush());gc.collect()
    assert window.released and pool.active_count==0 and refs[0]() is None
    from psana.gpu.gpu_stream import _failed_execution_pools
    assert pool not in _failed_execution_pools


def test_publication_metadata_validation_and_borrowed_input_consumer_lease():
    import cupy as cp
    parser,detector,_,specs,expected,window=fixture(cp,((0,),))
    def callback(evt,stream):
        evt.publish('scalar',cp.array(4,cp.uint32))
        evt.publish('empty',cp.empty((0,3),cp.float64))
        with pytest.raises(ValueError,match='duplicate'):evt.publish('scalar',cp.empty(1))
        with pytest.raises(ValueError,match='reserved'):evt.publish('camera.raw',cp.empty(1))
        with pytest.raises(TypeError,match='CuPy'):evt.publish('host',np.empty(1))
        with pytest.raises(ValueError,match='contiguous'):evt.publish('strided',cp.empty(8)[::2])
        evt.publish('borrowed',evt.input('camera.raw'))
    pool=EventPool(n=1)
    record=pool.submit(NS(iter_events=lambda:iter(specs)),None,envelopes(specs),
        {'camera.raw':detector},input_windows=(window,),batch_id=7,
        task=GpuTask(callback,['camera.raw']),detector_bindings=bindings(parser,detector))
    assert not window.close()
    pubs=record.publications_by_ts[100]
    assert (pubs['scalar'].shape,pubs['scalar'].nbytes,pubs['empty'].shape,pubs['empty'].nbytes)==((),4,(0,3),0)
    copied=cp.empty_like(pubs['borrowed'].array)
    consumer=cp.cuda.Stream(non_blocking=True)
    with consumer:
        consumer.wait_event(pubs['borrowed'].lease.result_ready)
        cp.copyto(copied,pubs['borrowed'].array)
        done=cp.cuda.Event();done.record(consumer)
    class RetryDone:
        fail=True
        def synchronize(self):
            if self.fail:raise RuntimeError('consumer pending')
            done.synchronize()
    completion=RetryDone()
    pubs['borrowed'].lease.register_consumer_done(completion)
    pool.begin_retire_next()
    with pytest.raises(RuntimeError,match='consumer pending'):pool.finish_retire_next()
    assert pool.active_count==1 and not window.released and record.producer_owners
    completion.fail=False
    pool.begin_retire_next();pool.finish_retire_next()
    assert window.released and not pool.active_count
    np.testing.assert_array_equal(copied.get(),expected[0])


def test_generic_fields_from_independent_input_bases():
    import cupy as cp
    parser,detector,_,specs,_,owner=fixture_inputs(cp)
    stream=cp.cuda.Stream(non_blocking=True)
    windows=tuple(owner((i,),range(4),stream) for i in range(2))
    seen=[]
    def callback(evt,stream):
        fields=evt.field('camera','raw','pixels')
        seen.append(tuple((f.segment_id,f.raw_ptr,f.row) for f in fields))
        assert all(f.rank==2 and f.xtc_type==1 and f.element_size==2 for f in fields)
        for f in fields:evt.publish('locator_'+str(f.segment_id),f.locator_rows[f.row])
    pool=EventPool(n=1)
    rec=pool.submit(NS(iter_events=lambda:iter(specs)),None,envelopes(specs),
        input_windows=windows,batch_id=7,task=GpuTask(callback,[('camera','raw','pixels')]),
        detector_bindings=bindings(parser,detector))
    assert len({ptr for _,ptr,_ in seen[0]})==2
    assert [s for s,_,_ in seen[0]]==[9,4,8]
    assert all(w.batch._locators=={} for w in windows)
    pool.begin_retire_next()
    for pubs in rec.publications_by_ts.values():
        for pub in pubs.values():assert int(pub.array.get()[10])==1
    for w in windows:assert not w.close()
    pool.finish_retire_next()
    assert all(w.released for w in windows)
