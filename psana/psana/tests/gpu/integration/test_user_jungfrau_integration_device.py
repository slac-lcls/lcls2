"""Batched integration correctness, counts, streams, and actual launch counts."""
import numpy as np
import pytest
from test_gpu_allocation_device import available
from test_user_jungfrau_calibration_device import Batch
from psana.gpu.examples.jungfrau_azimuthal_integration import JungfrauAzimuthalIntegration
from psana.tests.gpu.user_calibration_reference import calibrate
from psana.tests.gpu.user_integration_reference import integrate
pytestmark = [pytest.mark.gpu, pytest.mark.skipif(not available(), reason='no CUDA device')]


class OwnedBatch(Batch):
    def __init__(self, *args):
        super().__init__(*args)
        self.owners = []
    def keepalive(self, *owners): self.owners.extend(owners)


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
@pytest.mark.parametrize('empty', [False, True])
def test_histograms_valid_zero_missing_masks_invalid_gain_and_empty_bins(dtype, empty, monkeypatch):
    import cupy as cp
    rng = np.random.default_rng(17)
    shape = (3,4,3,7)
    constants = dict(pedestals=rng.uniform(0,20,shape).astype(dtype),
        pixel_gain=rng.uniform(.1,5,shape).astype(dtype), pixel_offset=np.zeros(shape,dtype),
        pixel_status=np.zeros(shape,np.uint64), status_extra=np.zeros(shape,np.uint64))
    constants['pixel_gain'][:,3,0,0] = 0
    constants['pixel_gain'][:,3,0,1] = np.inf
    constants['pedestals'][:,3,0,2] = np.nan
    constants['pixel_status'][2,1,0,3] = 1
    constants['status_extra'][1,1,0,4] = 1 << 40
    constants['pedestals'][:,3,0,5] = 0  # Valid zero intensity must count.
    bins = rng.integers(0,4,(4,3,7),dtype=np.int32)
    bins[3,0,5] = 4  # Dedicated bin for valid zeros; bin 5 stays empty.
    if empty: bins.fill(-1)
    raw = rng.integers(0,65536,(5,2,3,7),dtype=np.uint16)
    raw[:,0,0,5] = 0
    present = np.ones((5,2),np.uint8);present[1,1]=0;present[3]=0
    segments=(3,1)
    policy=dict(use_offset=True,status_bits=(1<<64)-1,stextra_bits=(1<<64)-1)
    with np.errstate(invalid='ignore'):
        calibrated = calibrate(raw,present,segments,constants,**policy)
    expected = integrate(calibrated,raw,present,segments,constants,bins,6,
                         status_bits=policy['status_bits'],stextra_bits=policy['stextra_bits'])
    launches=[]
    original=cp.RawKernel
    def counted(src,name,**kw):
        kernel=original(src,name,**kw)
        def launch(grid,block,args,**kwargs):
            launches.append((name,kwargs['stream'].ptr))
            return kernel(grid,block,args,**kwargs)
        return launch
    monkeypatch.setattr(cp,'RawKernel',counted)
    user=JungfrauAzimuthalIntegration(bins,6,**policy)
    stream=cp.cuda.Stream(non_blocking=True)
    with stream:
        batch=OwnedBatch(cp.asarray(raw),cp.asarray(present),segments,{k:cp.asarray(v) for k,v in constants.items()})
        user(batch,stream)
    stream.synchronize()
    actual=batch.outputs[user.output].get()
    np.testing.assert_array_equal(actual[:,2],expected[:,2])
    np.testing.assert_allclose(actual[:,:2],expected[:,:2],rtol=1e-12,atol=1e-9)
    assert launches==[('calibrate',stream.ptr),('integrate',stream.ptr)]
    assert list(batch.outputs)==[user.output] and len(batch.owners)>=3
    if not empty:
        np.testing.assert_array_equal(actual[:,2,4],[1,1,1,0,1])
        np.testing.assert_array_equal(actual[:,0,4],0)


def test_cross_stream_table_upload_tail_and_replaced_constants():
    import cupy as cp
    user=JungfrauAzimuthalIntegration(np.zeros((1,2,3),np.int32),2)
    delay=cp.RawKernel(r'''extern "C" __global__ void delay(unsigned long long n) {
        auto t=clock64(); while(clock64()-t<n) {} }''','delay')
    saved=[]
    for n,ped in ((5,1),(3,7),(1,20)):
        stream=cp.cuda.Stream(non_blocking=True)
        with stream:
            constants=dict(pedestals=cp.full((3,1,2,3),ped,cp.float32),pixel_gain=cp.full((3,1,2,3),2,cp.float32))
            batch=OwnedBatch(cp.full((n,1,2,3),20,cp.uint16),cp.ones((n,1),cp.uint8),(0,),constants)
            delay((1,),(1,),(np.uint64(20000000),),stream=stream)
            user(batch,stream)
            saved.append((stream,batch,(20-ped)/2))
    for stream,batch,value in reversed(saved):
        stream.synchronize()
        actual=batch.outputs[user.output].get()
        np.testing.assert_array_equal(actual[:,0,0],value)
        np.testing.assert_array_equal(actual[:,1,0],value*6)
        np.testing.assert_array_equal(actual[:,2,0],6)
        np.testing.assert_array_equal(actual[:,:,1],0)
    assert len(user._tables)==len(user._reductions)==1
    assert user.calls==3 and user.events==9
