"""Current frozen-runtime JF dense input staging, no calibration or user task."""
import argparse
from contextlib import closing
import gc
import importlib
import json
import os
from pathlib import Path
import platform
import time
from common import HERE,REF,require,sha,ts_hash,begin,end,sources,CacheGateError,cache_unavailable
rank_hint=int(os.environ['OMPI_COMM_WORLD_RANK'])
os.environ['CUDA_VISIBLE_DEVICES']=os.environ['JF_GPU_UUID'] if rank_hint>=2 else ''
os.sched_setaffinity(0,{int(x) for x in os.environ['BENCH_CPU_AFFINITY'].split(',')})
from mpi4py import MPI
from psana import DataSource
from psana.psexp.run import Run
comm=MPI.COMM_WORLD

def main(a):
    require(comm.size==a.bds+2 and comm.rank==rank_hint,'Rank mismatch')
    is_bd=comm.rank>=2
    pixels={r['timestamp']:dict(timestamp=r['timestamp'],raw=r['raw']) for r in json.loads((HERE/'pixels.json').read_text())}
    def calibration(run):
        run._clear_calibconst();run._calib_const={name:{} for name in run.dsparms.configinfo_dict};run.dsparms.calibconst=run._calib_const
    Run._setup_run_calibconst=calibration
    checks=[];managers=[]
    counters=dict(bytes=0,requests=0,read_wait_s=0.,peak_owned_and_held=0,budget_limit=0,cpu_bd_reads=0,pressure_drains=0)
    device=None
    if is_bd:
        from input_adapter import install
        install(checks,pixels if a.preflight else {})
        import cupy as cp
        import kvikio
        import kvikio.defaults as defaults
        from psana.gpu.gpu_budget import _GpuBudget
        from psana.gpu.gpu_events import GpuEventManager
        from psana.gpu.gpu_kvikio_read import KvikioGpuReader
        from psana.psexp.event_manager import EventManager
        defaults.set(dict(compat_mode=kvikio.CompatMode.ON,num_threads=8,task_size=1<<20,bounce_buffer_size=16<<20,auto_direct_io_read=False))
        require(defaults.get('compat_mode')==kvikio.CompatMode.ON,'Fallback required')
        effective={k:defaults.get(k) for k in ('num_threads','task_size','bounce_buffer_size','auto_direct_io_read')}
        require(effective==dict(num_threads=8,task_size=1<<20,bounce_buffer_size=16<<20,auto_direct_io_read=False),'Settings mismatch')
        wait=KvikioGpuReader.wait_batch
        def waited(reader,pending):
            fresh=not pending.completed;before=reader._total_bytes_read,reader._total_io_ns
            result=wait(reader,pending)
            if fresh:
                counters['requests']+=len(pending.futures);counters['bytes']+=reader._total_bytes_read-before[0];counters['read_wait_s']+=(reader._total_io_ns-before[1])/1e9
            return result
        KvikioGpuReader.wait_batch=waited
        for name in ('reserve','hold'):
            original=getattr(_GpuBudget,name)
            def charged(budget,*args,original=original,**kwargs):
                result=original(budget,*args,**kwargs);used=budget.committed()+budget._held
                require(used<=budget.limit(),'Budget exceeded');counters['peak_owned_and_held']=max(counters['peak_owned_and_held'],used);counters['budget_limit']=budget.limit();return result
            setattr(_GpuBudget,name,charged)
        setup=GpuEventManager._setup_gpu_pipeline
        def configured(m,*args,**kwargs):setup(m,*args,**kwargs);managers.append(m)
        GpuEventManager._setup_gpu_pipeline=configured
        trim=GpuEventManager._trim_gpu_caches
        def pressure(m):counters['pressure_drains']+=1;return trim(m)
        GpuEventManager._trim_gpu_caches=pressure
        read=EventManager._read
        def cpu_read(m,*args,**kwargs):counters['cpu_bd_reads']+=1;return read(m,*args,**kwargs)
        EventManager._read=cpu_read
        require(cp.cuda.runtime.getDeviceCount()==1,'One visible GPU required')
        device=dict(mask=os.environ['CUDA_VISIBLE_DEVICES'],pci=cp.cuda.runtime.deviceGetPCIBusId(0),name=cp.cuda.runtime.getDeviceProperties(0)['name'].decode(),effective=effective,compat_mode='ON',cupy=cp.__version__,kvikio=kvikio.__version__)
    start=time.perf_counter()
    ds=DataSource(exp='mfx101210926',run=387,dir=a.directory,detectors=['jungfrau'],gpu_det=['jungfrau'],gpu_bulk_read=True,gpu_bulk_target_bytes=1<<20,max_events=a.events,batch_size=20,n_gpu_streams=1,gpu_memory_budget_gb=6,gpu_d2h_chunk_size=0,skip_calib_load='all',log_level='ERROR')
    run=next(ds.runs());setup_seconds=time.perf_counter()-start
    require(run.comms.node_type()==('smd0' if comm.rank==0 else 'eb' if comm.rank==1 else 'bd'),'Wrong role')
    if is_bd:cp.cuda.Device().synchronize()
    comm.Barrier();observation=None
    if comm.rank==0:
        try:observation=begin(a.directory,a.cache)
        except CacheGateError as exc:
            cache_unavailable(a.output,vars(a),exc);comm.Abort(42)
    start=time.perf_counter();comm.Barrier();rank_start=time.perf_counter();stamps=[]
    with closing(run.events()) as events:
        for evt in events:stamps.append(int(evt.timestamp))
    if is_bd:cp.cuda.Device().synchronize()
    rank_seconds=time.perf_counter()-rank_start
    comm.Barrier()
    if comm.rank==0:end(observation,a.directory,time.perf_counter()-start)
    if is_bd:
        require(managers and all(m._closed and m._gpu_budget._held==0 and not m.gpu_reader._pending and not m.event_pool.active_count for m in managers),'Undrained resources')
        require(counters['cpu_bd_reads']==counters['pressure_drains']==0,'Unexpected CPU read or pressure drain')
        require(counters['requests']==5*len(stamps) and counters['bytes']==33557176*len(stamps),'Local byte/count mismatch')
        require([m._n_bd_per_gpu for m in managers]==[a.bds],'Wrong peers')
    modules=['psana','psana.gpu.gpu_events','psana.gpu.gpu_detector','psana.gpu.gpu_kvikio_read']
    row=dict(rank=comm.rank,is_bd=is_bd,host=platform.node(),timestamps=stamps,checks=checks,counts=counters,device=device,affinity=sorted(os.sched_getaffinity(0)),setup_seconds=setup_seconds,loop_seconds=rank_seconds,runtime_files={n:dict(path=importlib.import_module(n).__file__,sha256=sha(importlib.import_module(n).__file__)) for n in modules})
    rows=comm.gather(row,root=0)
    if comm.rank==0:
        bds=[r for r in rows if r['is_bd']];ts=[t for r in bds for t in r['timestamps']]
        require(len(ts)==len(set(ts))==a.events and ts_hash(sorted(ts))==REF[str(a.events)]['timestamp_sha256'],'Global event identities')
        require(all(r['timestamps']==sorted(r['timestamps']) for r in bds) and all(not r['timestamps'] for r in rows if not r['is_bd']),'Event ordering/roles')
        require(len({r['device']['pci'] for r in bds})==1 and all(r['device']['mask']==os.environ['JF_GPU_UUID'] for r in bds),'GPU mismatch')
        got=sorted([x for r in bds for x in r['checks']],key=lambda x:x['timestamp'])
        require(got==([pixels[t] for t in sorted(pixels)] if a.preflight else []),'Raw pixel hashes')
        total=sum(r['counts']['bytes'] for r in bds);require(total==REF[str(a.events)]['payload_bytes'],'Global bytes')
        result=dict(status='pass',workload='psana-dense-input',settings=vars(a),job=os.getenv('SLURM_JOB_ID'),host=platform.node(),rows=rows,measurement=observation,bytes=total,events=a.events,gib_s=total/(1<<30)/observation['loop_seconds'],events_per_second=a.events/observation['loop_seconds'],source_sha256=sources())
        with open(a.output,'x') as f:json.dump(result,f,indent=2);f.write('\n')
        print(json.dumps(dict(status='pass',gib_s=result['gib_s'],output=a.output)),flush=True)
    comm.Barrier();evt=None;run.close_shared_memory();managers.clear();del run,ds;gc.collect();comm.Barrier()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--directory',required=True);p.add_argument('--output',required=True);p.add_argument('--cache',choices=['cold','warm'],required=True);p.add_argument('--bds',type=int,required=True);p.add_argument('--events',type=int,choices=[200,10000],default=10000);p.add_argument('--preflight',action='store_true')
    try:main(p.parse_args())
    except BaseException:
        import traceback;traceback.print_exc();comm.Abort(1)
