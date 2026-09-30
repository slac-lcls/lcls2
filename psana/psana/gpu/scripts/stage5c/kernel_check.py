"""Isolated hot-buffer sanity check, not an end-to-end performance sample."""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import pickle
import statistics
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--source',type=Path,required=True)
    a=p.parse_args()
    from psana import dgram
    from work import LocalBatch
    from psana.gpu.examples.jungfrau_azimuthal_integration import JungfrauAzimuthalIntegration
    import cupy as cp
    panels={};timestamps=[]
    for name in json.loads((a.root/'reference.json').read_text())['10000']['stage_bytes']:
        fd=os.open(a.source/name,os.O_RDONLY)
        try:
            cfg=dgram.Dgram(file_descriptor=fd)
            while True:
                evt=dgram.Dgram(config=cfg)
                if evt.service()==12:break
            timestamps.append(int(evt.timestamp()))
            for segment,value in evt.jungfrau.items():
                assert segment not in panels
                panels[segment]=np.asarray(value.raw.raw).reshape(512,1024).copy()
        finally:os.close(fd)
    assert len(set(timestamps))==1 and len(panels)==32
    segments=tuple(sorted(panels))
    raw=np.stack([panels[s] for s in segments])
    with gzip.open(a.root/'constants.pkl.gz','rb') as f:host=pickle.load(f)['jungfrau']
    constants={k:cp.asarray(host[k][0]) for k in ('pedestals','pixel_gain','pixel_offset','pixel_status')}
    with np.load(a.root/'bins.npz') as bins:
        user=JungfrauAzimuthalIntegration(bins['bin_ids'],len(bins['edges'])-1,use_offset=True,status_bits=(1<<64)-1)
    stream=cp.cuda.Stream(non_blocking=True)
    with stream:
        raw20=cp.broadcast_to(cp.asarray(raw),(20,)+raw.shape).copy()
        presence=cp.ones((20,len(segments)),cp.uint8)
    stream.synchronize()
    captured={};original=cp.RawKernel
    def record(source,name,**kw):
        kernel=original(source,name,**kw);kernel.compile()
        def launch(grid,block,args,**kwargs):
            captured[name]=(kernel,grid,block,args)
            return kernel(grid,block,args,**kwargs)
        return launch
    cp.RawKernel=record
    cases={}
    try:
        for n in (1,20):
            batch=LocalBatch(raw20[:n],presence[:n],segments,constants,[timestamps[0]]*n)
            user(batch,stream);stream.synchronize()
            cases[n]=(batch,dict(captured))
    finally:cp.RawKernel=original
    expected=cases[1][0].outputs[user.output].get()[0]
    np.testing.assert_array_equal(cases[20][0].outputs[user.output].get(),np.repeat(expected[None],20,axis=0))
    timings={str(n):{name:[] for name in ('calibrate','integrate')} for n in cases}
    for repeat in range(12):
        for n in ((1,20) if repeat%2==0 else (20,1)):
            for name in ('calibrate','integrate'):
                kernel,grid,block,args=cases[n][1][name]
                for _ in range(3):kernel(grid,block,args,stream=stream)
                before,after=cp.cuda.Event(),cp.cuda.Event()
                before.record(stream)
                for _ in range(10):kernel(grid,block,args,stream=stream)
                after.record(stream);after.synchronize()
                timings[str(n)][name].append(cp.cuda.get_elapsed_time(before,after)/10)
    prop=cp.cuda.runtime.getDeviceProperties(0)
    result=dict(kind='hot buffers, repeated first real event, no I/O/allocation/D2H in kernel intervals',
        timestamp=timestamps[0],shape=list(raw.shape),segments=segments,
        device={k:prop[k] for k in ('multiProcessorCount','memoryClockRate','memoryBusWidth')},
        histogram_sha256=hashlib.sha256(expected.tobytes()).hexdigest(),
        total_valid_pixels=int(expected[2].sum()),nonzero_sums=int(np.count_nonzero(expected[1])),
        raw_gain_codes={str(i):int(np.count_nonzero(raw>>14==i)) for i in range(4)},
        timings_ms=timings,median_ms={n:{k:statistics.median(v) for k,v in t.items()} for n,t in timings.items()})
    print('KERNEL_CHECK_RESULT '+json.dumps(result,sort_keys=True),flush=True)


if __name__=='__main__':main()
