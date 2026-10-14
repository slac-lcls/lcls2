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
    calls, contexts, gathers = [], [], []
    gather = gd._batched_gather_kernel
    def counted(dtype):
        kernel = gather(dtype)
        def launch(*args, **kw):
            gathers.append(1)
            return kernel(*args, **kw)
        return launch
    monkeypatch.setattr(gd, '_batched_gather_kernel', counted)
    def callback(batch, stream):
        assert cp.cuda.get_current_stream().ptr == stream.ptr
        calls.append((batch.timestamps,batch.batch_event_indices,batch.batch_id,batch.run,batch.step_generation))
        contexts.append(batch)
        assert batch.segment_ids('camera') == (9,4,8)
        batch.publish('gain_alias', batch.calibconst('camera','gain').reshape(1),event_indices=[1])
        batch.publish('pixels', batch.input('camera.raw'))
        batch.publish('presence', batch.present('camera.raw'))
        batch.publish('other_pixels', batch.input('other.raw'))
        fields=batch.field('camera','raw','counter')
        assert fields is batch.field('camera','raw','counter')
        batch.publish('field_rows',fields.rows)
    task = GpuTask(callback, ['camera.raw','other.raw',('camera','raw','counter')],
                   [('camera','gain')])
    pool = EventPool(n=1,budget=budget)
    record = pool.submit(NS(iter_events=lambda: iter(specs)), None, envelopes(specs[:4]),
        preparers, input_windows=(window,), batch_id=7, task=task,
        detector_bindings=mapping, task_constants=constants, run=51, step_generation=3)
    assert not window.close()
    assert calls == [((100,101,102),(7,10,13),7,51,3)]
    assert len(gathers)==2 and window.batch._locators=={}
    assert len(record.publication_batches)==5
    assert pool.begin_retire_next() is record
    for ts, outputs in record.publications_by_ts.items():
        if ts==101:
            assert cp.asnumpy(outputs['gain_alias'].array)==2
            assert not outputs['pixels'].array.any()
        else:
            np.testing.assert_array_equal(outputs['pixels'].array.get(),expected[0 if ts==100 else 1])
    for ctx in contexts:
        with pytest.raises(RuntimeError,match='during its callback'):ctx.input('camera.raw')
    pool.finish_retire_next()
    assert window.released and not record.producer_owners and not record.publications_by_ts
    assert not record.publication_batches
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
        array=cp.empty((1,5),cp.uint32)
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
        evt.publish('scalar',cp.array([4],cp.uint32))
        evt.publish('empty',cp.empty((1,0,3),cp.float64))
        with pytest.raises(ValueError,match='duplicate'):evt.publish('scalar',cp.empty(1))
        with pytest.raises(ValueError,match='reserved'):evt.publish('camera.raw',cp.empty(1))
        with pytest.raises(TypeError,match='CuPy'):evt.publish('host',np.empty(1))
        with pytest.raises(ValueError,match='contiguous'):evt.publish('strided',cp.empty((1,8))[:,::2])
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


@pytest.mark.parametrize('depth', [1,2])
def test_one_scratch_allocation_and_launch_per_subbatch_with_tail_reuse(monkeypatch,depth):
    import cupy as cp
    parser,initial,budget,specs,expected,window=fixture(cp, ((0,),)*20)
    detector=DenseInputPreparer(initial.det_shape,initial.binding,n_slots=depth,budget=budget)
    detector.configure_gather(parser.handle_indices)
    kernel=cp.RawKernel('''extern "C" __global__ void scratch(
        const unsigned short* raw, unsigned short* out, unsigned long long n) {
        unsigned long long i=blockIdx.x*blockDim.x+threadIdx.x;
        if(i<n) out[i]=raw[i]+1;
    }''','scratch')
    kernel.compile()
    counts=dict(callback=0,allocate=0,launch=0,completion=0)
    real_allocate=cp.empty_like
    def allocate(*a,**kw):
        counts['allocate']+=1
        return real_allocate(*a,**kw)
    monkeypatch.setattr(cp,'empty_like',allocate)
    real_event=cp.cuda.Event
    def completion(**kw):
        counts['completion']+=1
        return real_event(**kw)
    monkeypatch.setattr(cp.cuda,'Event',completion)
    contexts=[]
    def callback(batch,stream):
        counts['callback']+=1
        contexts.append(batch)
        raw=batch.input('camera.raw')
        out=cp.empty_like(raw)
        batch.keepalive(out)
        kernel(((raw.size+255)//256,),(256,),(raw,out,np.uint64(raw.size)),stream=stream)
        counts['launch']+=1
    pool=EventPool(n=depth,budget=budget)
    def check(record):
        if record is None:return
        arrays=[a for a in record.producer_owners if isinstance(a,cp.ndarray)]
        assert len(arrays)==1 and not record.publications_by_ts and not record.publication_batches
        n=record.batch_inputs.size
        np.testing.assert_array_equal(arrays[0].get(),np.asarray(expected[:n])+1)
    for n in (20,3,1):
        check(pool.begin_retire_next());pool.finish_retire_next()
        pool.submit(NS(iter_events=lambda:iter(specs)),None,envelopes(specs[:n]),
            {'camera.raw':detector},input_windows=(window,),batch_id=7,
            task=GpuTask(callback,['camera.raw']),detector_bindings=bindings(parser,detector))
    assert not window.close()
    for record in pool.flush():check(record)
    assert counts==dict(callback=3,allocate=3,launch=3,completion=3)
    assert window.released
    for context in contexts:
        with pytest.raises(RuntimeError,match='during its callback'):context.timestamps_gpu


def test_sparse_mixed_publication_groups_and_shared_consumer_completion():
    import cupy as cp
    parser,detector,budget,specs,expected,window=fixture(cp,((0,),)*3)
    saved=[]
    def callback(batch,stream):
        raw=batch.input('camera.raw')
        batch.publish('out',raw[1:3],event_indices=[2,0])
        scalar=cp.array([42],cp.uint32)
        batch.publish('out',scalar,event_indices=[1])
        batch.publish('empty',cp.empty((3,0,2),cp.float32))
        with pytest.raises(ValueError,match='duplicate'):
            batch.publish('out',raw,event_indices=[0,1,2])
        saved.append(batch)
    pool=EventPool(n=1,budget=budget)
    rec=pool.submit(NS(iter_events=lambda:iter(specs)),None,envelopes(specs),
        {'camera.raw':detector},input_windows=(window,),batch_id=7,
        task=GpuTask(callback,['camera.raw']),detector_bindings=bindings(parser,detector))
    assert len(rec.publication_batches)==3
    p0,p1,p2=[rec.publications_by_ts[t]['out'] for t in (100,101,102)]
    assert p0.batch is p2.batch and p1.batch is not p0.batch
    assert p0.row==1 and p2.row==0 and p1.shape==()
    assert p0.array.data.ptr==p0.batch.array.data.ptr+p0.nbytes
    copy=cp.empty_like(p0.array)
    consumer=cp.cuda.Stream(non_blocking=True)
    with consumer:
        consumer.wait_event(p0.lease.result_ready)
        cp.copyto(copy,p0.array)
        done=cp.cuda.Event();done.record(consumer)
    p0.lease.register_consumer_done(done)
    pool.begin_retire_next()
    np.testing.assert_array_equal(p1.array.get(),42)
    assert all(pub['empty'].shape==(0,2) and pub['empty'].nbytes==0
               for pub in rec.publications_by_ts.values())
    assert not window.close()
    pool.finish_retire_next()
    np.testing.assert_array_equal(copy.get(),expected[2])
    assert window.released and not rec.publication_batches
