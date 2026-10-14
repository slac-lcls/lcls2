"""A100 acceptance for aligned task inputs and bulk field/identity metadata."""
import gc
import weakref
from types import SimpleNamespace as NS

import numpy as np
import pytest

from test_batched_gather import _gpu_available
from test_gpu_producer_device import fixture, bindings, envelopes
from test_multiowner_gather import fixture_inputs
from psana.gpu import GpuTask
from psana.gpu.gpu_detector import DenseInputPreparer
from psana.gpu.gpu_input import GpuDetectorBinding
from psana.gpu.gpu_stream import EventPool
from psana.gpu.gpu_task import RequestedConstants
from psana.gpu.gpu_task_batch import metadata_bytes, FIELD_SOURCE_PRESENT, FIELD_RAW_PTR

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(not _gpu_available(), reason='no CUDA device')]


@pytest.mark.parametrize('selections,limit', [(((1,0),(2,),(0,),(),(1,0)), 4),
                                            (((2,), (2,)), 2), (((0,),), 0)])
def test_aligned_inputs_absent_detector_selection_and_bulk_upload(monkeypatch, selections, limit):
    import cupy as cp
    from psana.gpu import gpu_detector as gd, gpu_task_batch as tb
    parser, camera, budget, specs, expected, window = fixture(cp, selections)
    mapping = bindings(parser, camera)
    other = GpuDetectorBinding('other', canonical_segment_ids=(0,),
        field_handles_by_segment={0: parser.configs.resolve('other',0,'raw','pixels')})
    mapping['other'] = other
    other_preparer = DenseInputPreparer((1,3,100), other, budget=budget, n_slots=1)
    other_preparer.configure_gather(parser.handle_indices)
    constants = RequestedConstants([('camera', 'gain')], budget)
    constants.refresh({'camera': {'gain': np.array(2, np.uint32)}})
    gathers, uploads = [], []
    original_gather, original_allocate = gd._batched_gather_kernel, tb.owned_empty
    def counted(dtype):
        kernel = original_gather(dtype)
        def launch(*a, **kw):
            gathers.append(1)
            return kernel(*a, **kw)
        return launch
    class UploadCounter:
        def __init__(self, array): self.array = array
        def __getitem__(self, key): return self.array[key]
        def set(self, *a, **kw):
            uploads.append(1)
            return self.array.set(*a, **kw)
    monkeypatch.setattr(gd, '_batched_gather_kernel', counted)
    monkeypatch.setattr(tb, 'owned_empty', lambda *a, **kw: UploadCounter(original_allocate(*a, **kw)))
    def request_metadata(batch, stream):
        # All GPU metadata is requested before producer completion is recorded.
        assert batch.timestamps_gpu is not None
        assert batch.batch_event_indices_gpu is not None
        assert batch.field('camera','raw','counter') is not None
    task = GpuTask(request_metadata, ['camera.raw', 'other.raw', ('camera','raw','counter')],
                   [('camera','gain')])
    pool = EventPool(n=1, budget=budget)
    rec = pool.submit(NS(iter_events=lambda:iter(specs)), None, envelopes(specs[:limit]),
        {'camera.raw':camera, 'other.raw':other_preparer}, input_windows=(window,),
        batch_id=7, task=task, detector_bindings=mapping, task_constants=constants,
        run=51, step_generation=3)
    selected = [i for i in range(limit) if selections[i]]
    batch = rec.batch_inputs
    assert (batch.batch_id, batch.run, batch.step_generation) == (7,51,3)
    assert batch.timestamps == tuple(100+i for i in selected)
    assert batch.batch_event_indices == tuple(7+3*i for i in selected)
    assert len(gathers) == (2 if selected else 0)
    assert len(uploads) == bool(selected)
    assert pool.pinned_bytes() == batch.pinned_nbytes
    assert pool.pinned_bytes() >= metadata_bytes(task, mapping, len(selected))
    pool.begin_retire_next()
    if selected:
        np.testing.assert_array_equal(batch.timestamps_gpu.get(), batch.timestamps)
        np.testing.assert_array_equal(batch.batch_event_indices_gpu.get(), batch.batch_event_indices)
        assert batch.input('camera.raw').shape == (len(selected),3,3,100)
        assert batch.input('other.raw').shape == (len(selected),1,3,100)
        assert batch.calibconst('camera','gain') is constants.get('camera','gain')
        for row, i in enumerate(selected):
            for name, segments, streams in [('camera',(9,4,8),(1,0,1)),('other',(0,),(2,))]:
                wanted = np.zeros((len(segments),3,100), np.uint16)
                present = []
                for j,(segment,stream) in enumerate(zip(segments,streams)):
                    present.append(stream in selections[i])
                    if present[-1]: wanted[j] = np.arange(300).reshape(3,100)+400*i+segment
                np.testing.assert_array_equal(batch.input(name+'.raw')[row].get(), wanted)
                np.testing.assert_array_equal(batch.present(name+'.raw')[row].get(), present)
        table = batch.field('camera','raw','counter')
        assert table.segment_ids == (9,4,8)
        np.testing.assert_array_equal(table.rows.get()[:,:,FIELD_SOURCE_PRESENT],
                                     [[s in selections[i] for s in (1,0,1)] for i in selected])
    else:
        assert batch.input('camera.raw') is None and batch.timestamps_gpu is None
    assert window.batch._locators == {}
    assert not window.close()
    pool.finish_retire_next()
    assert window.released and pool.pinned_bytes() == 0
    with pytest.raises(RuntimeError, match='closed'): batch.input('camera.raw')
    constants.close()


def test_batched_field_kernel_independent_windows():
    import cupy as cp
    parser, camera, budget, specs, _, owner = fixture_inputs(cp)
    stream = cp.cuda.Stream(non_blocking=True)
    windows = tuple(owner((i,),range(4),stream) for i in range(2))
    pool = EventPool(n=1, budget=budget)
    kernel = cp.RawKernel('''extern "C" __global__ void counters(
        const unsigned long long* fields, unsigned int* out, unsigned long long n) {
        unsigned long long i=blockIdx.x*blockDim.x+threadIdx.x;
        if(i>=n) return;
        const unsigned long long* f=fields+8*i;
        out[i]=0xffffffff;
        if(!f[7]) return;
        const unsigned long long* loc=(const unsigned long long*)f[2]+f[3]*11;
        if(loc[10]!=1 || loc[1]!=f[4] || loc[2]!=f[5] || f[4]!=2 || f[5]!=0 ||
           f[6]!=4 || loc[9]!=4 || loc[8]>f[1] || 4>f[1]-loc[8]) return;
        out[i]=*(const unsigned int*)((const unsigned char*)f[0]+loc[8]);
    }''','counters')
    calls=[]
    def callback(batch, producer):
        field=batch.field('camera','raw','counter')
        out=cp.empty((batch.size,3),cp.uint32)
        batch.publish('counters',out)
        calls.append(batch.size)
        # One kernel across all rows, before producer completion is recorded.
        kernel((1,), (128,), (field.rows,out,np.uint64(out.size)),stream=producer)
    rec = pool.submit(NS(iter_events=lambda:iter(specs)),None,envelopes(specs),
        input_windows=windows,batch_id=7,
        task=GpuTask(callback,[('camera','raw','counter')]),
        detector_bindings=bindings(parser,camera))
    assert calls==[4]
    batch=rec.batch_inputs
    field=batch.field('camera','raw','counter')
    out=rec.publication_batches[0].array
    pool.begin_retire_next()
    wanted = np.array([[0,0,0], [0xffffffff,1,0xffffffff],
                       [2,2,2], [3,0xffffffff,3]], np.uint32)
    np.testing.assert_array_equal(out.get(), wanted)
    assert len(set(field.rows.get()[0,:,FIELD_RAW_PTR])) == 2
    assert all(w.batch._locators == {} for w in windows)
    for w in windows: assert not w.close()
    pool.finish_retire_next()
    assert all(w.released for w in windows)


def test_failed_metadata_upload_retains_pinned_source_and_charge_until_retry(monkeypatch):
    import cupy as cp
    from psana.gpu import gpu_task_batch as tb
    _, _, budget, specs, _, window = fixture(cp, ((0,),))
    pool = EventPool(n=1, budget=budget)
    real = pool.next_stream
    class RetryStream:
        fail = True
        def __getattr__(self, name): return getattr(real,name)
        def __enter__(self): return real.__enter__()
        def __exit__(self, *args): return real.__exit__(*args)
        def synchronize(self):
            if self.fail: raise RuntimeError('unproven metadata upload')
            real.synchronize()
    pool._streams[0] = stream = RetryStream()
    refs = []
    allocate = tb.owned_empty
    class FailingUpload:
        def __init__(self, array): self.array = array
        def __getitem__(self,key): return self.array[key]
        def set(self, source, **kwargs):
            refs.append(weakref.ref(source))
            self.array.set(source,**kwargs)
            raise ValueError('after upload')
    monkeypatch.setattr(tb,'owned_empty',lambda *a,**kw:FailingUpload(allocate(*a,**kw)))
    with pytest.raises(RuntimeError,match='unproven metadata upload'):
        pool.submit(NS(iter_events=lambda:iter(specs)),None,envelopes(specs),
                    input_windows=(window,),batch_id=7,task=GpuTask(lambda batch, stream:batch.timestamps_gpu))
    gc.collect()
    assert refs[0]() is not None and refs[0]().nbytes==16
    assert pool.pinned_bytes()>=16 and pool.active_count==1
    assert sum(a['requested'] for a in budget.allocation_snapshot()
               if a['category']=='task-metadata') == 16
    assert not window.close()
    stream.fail=False
    list(pool.flush())
    gc.collect()
    assert refs[0]() is None and pool.pinned_bytes()==0 and window.released
