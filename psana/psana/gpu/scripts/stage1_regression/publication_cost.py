"""Public Run.events() scheduling comparison on immutable parsed GPU fixtures.

No DataSource setup, I/O or calibration is timed. The per-event reference uses
benchmark-only dense GPUResult input injection, then actual on_gpu_view and
user-loop launches/copies. The batch path uses GpuTask and automatic D2H with
actual on_cpu access. Both return independent NumPy rows. Compact outputs use
64 MiB pinned capacity; default full-image outputs use cap=0 on both sides.
image_fresh_host also allocates the reference ordinary-host destination per event;
image_pinned uses a 1.5 GiB aggregate cap on both sides, enough for two N=20 groups. Fixture
injection is benchmark plumbing, not a new product input/result API.
"""
import argparse
import gc
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace as NS

import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--dispatch',choices=('event','batch'),required=True)
    p.add_argument('--profile',choices=('micro','jungfrau'),required=True)
    p.add_argument('--submissions',type=int,default=1000)
    p.add_argument('--reverse',action='store_true')
    p.add_argument('--modes',nargs='+',choices=('compact','compact_prealloc','image','image_fresh_host','image_pinned'),default=['compact','compact_prealloc','image'])
    p.add_argument('--batch-sizes',nargs='+',type=int,choices=(1,3,20),default=[1,3,20])
    p.add_argument('--depths',nargs='+',type=int,choices=(1,2),default=[1,2])
    a=p.parse_args()
    import cupy as cp
    import psana
    from psana.event import EventEnvelope
    from psana.psexp.run import Run
    from psana.gpu import GpuTask, gpu_detector as gd
    from psana.gpu.context import SlotLease
    from psana.gpu.gpu_budget import _GpuBudget
    from psana.gpu.gpu_events import GpuEventManager,_GpuOnlyDgram
    from psana.gpu.gpu_input_window import InputWindow
    from psana.gpu.gpu_stream import EventPool
    from psana.gpu.gpudgram import parser as parser_module
    if a.dispatch=='batch':from psana.gpu.gpu_d2h import PublicationD2H
    sys.path.insert(0,str(Path(psana.__file__).parent/'tests/gpu/integration'))
    from test_batched_gather import _setup,_input
    from callback_fixture import jungfrau_fixture
    from callback_cost import count_framework
    compact=cp.RawKernel('''extern "C" __global__ void compact(const unsigned short* x,
        unsigned int* y,unsigned long long n,unsigned long long stride) {
        unsigned long long i=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x;
        if(i<n) y[i]=x[i*stride+300]+1;
    }''','compact')
    image=cp.RawKernel('''extern "C" __global__ void image(const unsigned short* x,
        unsigned short* y,unsigned long long n) {
        unsigned long long i=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x;
        if(i<n) y[i]=x[i]+1;
    }''','image')
    compact.compile();image.compile()
    result=dict(complete=False,scope=__doc__,dispatch=a.dispatch,profile=a.profile,
        psana=psana.__file__,cupy=cp.__version__,cuda=cp.cuda.runtime.runtimeGetVersion(),
        device=str(cp.cuda.runtime.getDeviceProperties(0)['name']),affinity=sorted(os.sched_getaffinity(0)),samples=[],preflights=[])
    modes=list(a.modes)
    if a.reverse:modes.reverse()
    for size in a.batch_sizes:
        budget=_GpuBudget(8<<30)
        if a.profile=='jungfrau':
            parser,initial,dtype,batch,events,expected=jungfrau_fixture(cp,budget,size)
        else:
            parser,initial,dtype=_setup(cp,budget=budget)
            stream=cp.cuda.Stream(non_blocking=True)
            batch,events,expected=_input(cp,parser,stream,[(0,)]*size,dtype)
            stream.synchronize();expected=np.asarray(expected)
        specs=tuple(e.event for e in events)
        indexes={e.timestamp:i for i,e in enumerate(specs)}
        envelopes=[EventEnvelope(dgrams=[_GpuOnlyDgram(e.timestamp)]) for e in specs]
        gv=NS(iter_events=lambda:iter(specs))
        stride=int(np.prod(initial.det_shape))
        class ImmutableInput:
            def __getattr__(self,name):return getattr(batch,name)
            def retire(self):pass
        for depth in a.depths:
            preparer=gd.DenseInputPreparer(initial.det_shape,initial.binding,n_slots=depth,budget=budget)
            preparer.configure_gather(parser.handle_indices)
            for mode in modes:
                is_image=mode.startswith('image');prealloc=mode=='compact_prealloc'
                cap=(1536<<20) if mode=='image_pinned' else (0 if is_image else 64<<20)
                pool=EventPool(n=depth,budget=budget)
                delivery=PublicationD2H(cap) if a.dispatch=='batch' else None
                manager=GpuEventManager.__new__(GpuEventManager)
                manager._first_batch_logged=True;manager.gpu_det_names=['camera']
                manager.gpu_detector_bindings={}
                consumer=cp.cuda.Stream(non_blocking=True)
                buffers=[cp.empty((size,),cp.uint32) for _ in range(depth)] if prealloc and a.dispatch=='batch' else []
                event_buffer=cp.empty((1,),cp.uint32) if prealloc and a.dispatch=='event' else None
                host_shape=initial.det_shape if is_image else ()
                host_dtype=np.uint16 if is_image else np.uint32
                # Reference has one reusable host destination because each
                # event synchronously returns its independent CPU result.
                pin_capacity=((int(np.prod(host_shape))*np.dtype(host_dtype).itemsize+4095)//4096)*4096 if cap and a.dispatch=='event' else 0
                pin=cp.cuda.PinnedMemoryPointer(cp.cuda.PinnedMemory(pin_capacity),0) if pin_capacity else None
                host=np.ndarray(host_shape,host_dtype,buffer=pin) if pin is not None else (None if mode=='image_fresh_host' else np.empty(host_shape,host_dtype))
                counts={};checking=False
                def callback(ctx,stream):
                    if checking:counts['callbacks']+=1
                    raw=ctx.input('camera.raw')
                    out=buffers[pool.next_slot_id] if prealloc else cp.empty(raw.shape if is_image else (ctx.size,),host_dtype)
                    if checking:
                        counts['allocations']+=not prealloc;counts['kernels']+=1
                    ctx.publish('value',out)
                    if is_image:image(((raw.size+255)//256,),(256,),(raw,out,np.uint64(raw.size)),stream=stream)
                    else:compact(((ctx.size+255)//256,),(256,),(raw,out,np.uint64(ctx.size),np.uint64(stride)),stream=stream)
                task=GpuTask(callback,['camera.raw']) if a.dispatch=='batch' else None
                def handoff(record):
                    if a.dispatch=='event':
                        # Use a real result lease: InputSlotLease is a different
                        # API. Join view completion before all input releases.
                        ready=next(iter(record.input_leases_by_ts.values())).result_ready
                        lease=SlotLease(ready);record.leases.insert(0,lease)
                        data=record.prepared_inputs['camera.raw'].data
                        for i,event in enumerate(specs):
                            record.gpu_results_by_ts[event.timestamp]={'bench.input':data[i]}
                            record.leases_by_ts[event.timestamp]={'bench.input':lease}
                    else:
                        delivery.enqueue(record)
                        if checking:
                            counts['copies']+=len(record.publication_batches)
                            counts['output_pinned_peak']=max(counts['output_pinned_peak'],delivery.pinned_bytes)
                def source(n):
                    for _ in range(n):
                        old=pool.begin_retire_next()
                        yield from manager._yield_ready(old)
                        pool.finish_retire_next()
                        window=InputWindow(7,0,ImmutableInput(),batch._test_descriptors)
                        try:
                            rec=pool.submit(gv,None,envelopes,{'camera.raw':preparer},
                                input_windows=(window,),batch_id=7,task=task,
                                detector_bindings={'camera':initial.binding})
                            handoff(rec)
                        finally:window.close()
                    for rec in pool.flush():yield from manager._yield_ready(rec)
                def run(n,check=False):
                    nonlocal checking,counts
                    checking=check
                    counts=dict(callbacks=0,allocations=0,kernels=0,copies=0,events=0,output_pinned_peak=pin_capacity)
                    run=Run.__new__(Run);run._run_ctx=None
                    run._handle_transition=lambda dgrams:False
                    run._evt_iter=source(n)
                    start=time.perf_counter_ns()
                    for event in run.events():
                        if a.dispatch=='event':
                            with event.gpu.get('bench.input').on_gpu_view(consumer) as raw:
                                out=event_buffer if prealloc else cp.empty(host_shape if is_image else (1,),host_dtype)
                                if check:
                                    counts['kernels']+=1;counts['allocations']+=not prealloc
                                if is_image:image(((raw.size+255)//256,),(256,),(raw,out,np.uint64(raw.size)),stream=consumer)
                                else:compact((1,),(1,),(raw,out,np.uint64(1),np.uint64(stride)),stream=consumer)
                                destination=np.empty(host_shape,host_dtype) if mode=='image_fresh_host' else host
                                (out if is_image else out.reshape(host_shape)).get(out=destination,stream=consumer,blocking=True)
                                value=destination.copy()
                                if check:counts['copies']+=1
                        else:value=event.gpu.get('value').on_cpu
                        if check:
                            counts['events']+=1
                            row=expected[indexes[event.timestamp]]
                            reference=row+np.uint16(1) if is_image else np.asarray(int(row.flat[300])+1,np.uint32)
                            np.testing.assert_array_equal(value,reference)
                    elapsed=time.perf_counter_ns()-start
                    assert not pool.active_count
                    if check:
                        operations=n*(size if a.dispatch=='event' else 1)
                        assert counts['kernels']==counts['copies']==operations,counts
                        assert counts['allocations']==(0 if prealloc else operations),counts
                        assert counts['callbacks']==(n if a.dispatch=='batch' else 0),counts
                        assert counts['events']==n*size and counts['output_pinned_peak']<=cap
                    return elapsed
                # Numerical and counted preflights, excluded from all timing.
                with count_framework(cp,gd,parser_module,'batch') as framework:
                    run(depth*2+1,True)
                result['preflights'].append(dict(mode=mode,batch_size=size,depth=depth,user=counts,framework=framework))
                assert framework['gather']==depth*2+1 and framework['task_metadata_uploads']==0
                assert framework['completion_events']==(depth*2+1)*(size+1 if a.dispatch=='event' else 2),framework
                run(32 if is_image and a.profile=='jungfrau' else 256)
                gc.collect()
                elapsed=run(a.submissions)
                result['samples'].append(dict(mode=mode,batch_size=size,depth=depth,
                    submissions=a.submissions,events=a.submissions*size,
                    loop_us_per_subbatch=elapsed/a.submissions/1000,
                    loop_us_per_event=elapsed/(a.submissions*size)/1000,
                    output_pinned_cap=cap,output_pinned_after_loop=(delivery.pinned_bytes if delivery else pin_capacity),
                    user_output_bytes_per_subbatch=size*(stride*2 if is_image else 4),
                    budget_committed_after_drain=budget.committed(),cupy_pool_used_after_drain=cp.get_default_memory_pool().used_bytes()))
                if delivery:delivery.close()
                del run,source,handoff,callback,pool,delivery,manager,buffers,event_buffer,host,pin
            del preparer
        batch.retire();parser.close()
        del batch,parser,initial,events,expected
        gc.collect()
    result['complete']=True
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
