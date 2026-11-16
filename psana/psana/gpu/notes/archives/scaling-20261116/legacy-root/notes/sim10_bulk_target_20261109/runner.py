"""DataSource MPI loop with raw descriptor or dense-raw staging, no user kernels."""
import argparse,gc,importlib,json,os,platform,subprocess,sys,time
from pathlib import Path
from contextlib import closing
from support import HERE,NOTES,raw,require,sha,cache,physical_counters,counter_delta,affinity,source_hashes,write
rank_hint=int(os.environ['OMPI_COMM_WORLD_RANK'])
os.environ['CUDA_VISIBLE_DEVICES']=os.environ['JF_GPU_UUID'] if rank_hint==2 else ''
chosen=affinity(rank_hint)
from mpi4py import MPI
from psana import DataSource
from psana.gpu import GpuTask
comm=MPI.COMM_WORLD

class StagingTask:
    def __init__(self,mode,preflight):
        self.mode=mode;self.callbacks=[];self.checks=[]
        self.pixels={r['timestamp']:r['raw'] for r in json.loads((raw.HERE/'pixels.json').read_text())} if preflight else {}
        self.inputs=('jungfrau.raw',) if mode=='raw-dense' else (('jungfrau','sim10','seed') if mode=='sim10-field' else ('jungfrau','raw','raw'),)
    def __call__(self,batch,stream):
        if self.mode=='raw-dense':
            a=batch.input('jungfrau.raw')
            require(a.shape==(batch.size,32,512,1024) and str(a.dtype)=='uint16','Dense shape/type')
            for i,ts in enumerate(batch.timestamps):
                if ts in self.pixels:
                    stream.synchronize();require(bool(batch.present('jungfrau.raw')[i].get().all()),'Missing panel')
                    actual=raw.digest(a[i].get());require(actual==self.pixels[ts],'Raw CPU/GPU hash mismatch')
                    self.checks.append(dict(timestamp=ts,raw=actual))
        else:
            field=batch.field(*self.inputs[0]);require(tuple(field.segment_ids)==tuple(range(32)) and field.rows.shape==(batch.size,32,8),'Descriptor shape/panels')
        self.callbacks.append(dict(timestamps=list(batch.timestamps),size=batch.size,stream=int(stream.ptr)))

def main(a):
    require(comm.size==3 and comm.rank==rank_hint,'Require SMD0, EB, one BD')
    dataset='sim10' if a.mode=='sim10-field' else 'raw'
    info=json.loads(Path(a.inputs).read_text())[dataset];stamps=info['timestamps'][:a.events]
    require(len(stamps)==a.events,'Event reference')
    if dataset=='sim10':
        sys.path.insert(0,str(NOTES/'mfx_expand10_stage2_20261105'));import adapter;adapter.register_reader()
    task=StagingTask(a.mode,a.preflight);managers=[];cpu_reads=[0];device=None
    if rank_hint==2:
        import cupy as cp
        import kvikio
        import kvikio.defaults as defaults
        from psana.gpu.gpu_events import GpuEventManager
        from psana.psexp.event_manager import EventManager
        sys.path.insert(0,str(NOTES/'mfx_pipeline_scaling_20261109'));from bulk_accounting import accounting
        settings=dict(compat_mode=kvikio.CompatMode.ON,num_threads=8,task_size=1<<20,bounce_buffer_size=16<<20,auto_direct_io_read=False)
        defaults.set(settings);require(all(defaults.get(k)==v for k,v in settings.items()),'Settings mismatch')
        configure=GpuEventManager._setup_gpu_pipeline
        def configured(m,*args,**kwargs):configure(m,*args,**kwargs);managers.append(m)
        GpuEventManager._setup_gpu_pipeline=configured
        original=EventManager._read
        def read(m,*args,**kwargs):cpu_reads[0]+=1;return original(m,*args,**kwargs)
        EventManager._read=read
        require(cp.cuda.runtime.getDeviceCount()==1,'One GPU required')
        device=dict(mask=os.environ['CUDA_VISIBLE_DEVICES'],pci=cp.cuda.runtime.deviceGetPCIBusId(0),name=cp.cuda.runtime.getDeviceProperties(0)['name'].decode(),cupy=cp.__version__,kvikio=kvikio.__version__,compat_mode='ON',effective={k:defaults.get(k) for k in settings if k!='compat_mode'})
    ds=DataSource(exp=info['exp'],run=info['run'],dir=info['directory'],detectors=['jungfrau'],gpu_det='jungfrau',live=False,max_events=a.events,
        skip_calib_load='all',gpu_bulk_read=True,gpu_bulk_target_bytes=a.bulk_target,batch_size=a.batch_size,n_gpu_streams=1,gpu_memory_budget_gb=6,gpu_d2h_chunk_size=0,
        gpu_fn=GpuTask(task,inputs=task.inputs,calibconst=()),log_level='ERROR')
    run=next(ds.runs());require(run.comms.node_type()==('smd0' if rank_hint==0 else 'eb' if rank_hint==1 else 'bd'),'Roles')
    if rank_hint==2:cp.cuda.Device().synchronize()
    comm.Barrier();observation=None
    if rank_hint==0:
        before=physical_counters();time.sleep(1);after=physical_counters()
        files=list(map(Path,info['files']));cache(files,True);initial=cache(files)
        require(all(x['resident_pages']==0 for x in initial),'Cold gate')
        observation=dict(initial_cache=initial,idle=dict(before=before,after=after,delta=counter_delta(before,after)),nic_before=physical_counters())
        start=time.perf_counter()
    comm.Barrier();delivered=[];io=None
    if rank_hint==2:
        with accounting() as io:
            with closing(run.events()) as events:
                for event in events:delivered.append(int(event.timestamp))
            cp.cuda.Device().synchronize()
    else:
        with closing(run.events()) as events:
            for event in events:delivered.append(int(event.timestamp))
    comm.Barrier()
    if rank_hint==0:
        seconds=time.perf_counter()-start;observation['seconds']=seconds;observation['nic_after']=physical_counters()
        observation['nic_delta']=counter_delta(observation['nic_before'],observation['nic_after'],info['event_bytes']*a.events)
        require(all(x['seconds']>=seconds for x in observation['nic_delta'].values()),'NIC bracket')
        observation['final_cache']=cache(files)
    if rank_hint==2:
        require(delivered==stamps,'Exact delivered timestamps')
        require([t for c in task.callbacks for t in c['timestamps']]==delivered,'Callbacks vs delivery')
        require(io['requests']==io['completed_requests']==info['expected_reads'][str(a.events)][str(a.bulk_target)]['count'] and io['requested_bytes']==io['completed_bytes']==info['event_bytes']*a.events,'Exact GPU I/O')
        require(io['read_size_histogram']==info['expected_reads'][str(a.events)][str(a.bulk_target)]['histogram'],'Read size histogram')
        require(cpu_reads[0]==io['pressure_drains']==0,'CPU payload or admission drain')
        require(len(managers)==1 and all(m._closed and not m.gpu_reader._pending and not m.event_pool.active_count and m._gpu_budget._held==0 for m in managers),'Drained resources')
        require(managers[0]._n_bd_per_gpu==1,'One BD peer')
        if a.preflight:require({r['timestamp'] for r in task.checks}==set(task.pixels),'Preflight hashes')
    drain=[dict(closed=m._closed,pending=len(m.gpu_reader._pending),active=m.event_pool.active_count,held=m._gpu_budget._held,peers=m._n_bd_per_gpu) for m in managers]
    row=dict(drain=drain,rank=rank_hint,host=platform.node(),affinity=chosen,device=device,timestamps=delivered,callbacks=task.callbacks,checks=task.checks,io=io,cpu_reads=cpu_reads[0],
        numa_policy=subprocess.run(['numactl','--show'],capture_output=True,text=True,check=True).stdout)
    require('policy: interleave' in row['numa_policy'],'Expected interleaved page allocations')
    rows=comm.gather(row,root=0)
    if rank_hint==0:
        require(all(not r['timestamps'] for r in rows[:2]),'Helper events')
        modules=('psana','psana.gpu.gpu_events','psana.gpu.gpu_detector','psana.gpu.gpu_task','psana.gpu.gpu_kvikio_read')
        result=dict(status='pass',settings=vars(a),events=a.events,bytes=info['event_bytes']*a.events,seconds=seconds,gib_s=info['event_bytes']*a.events/(1<<30)/seconds,hz=a.events/seconds,
            rows=rows,measurement=observation,job=os.environ['SLURM_JOB_ID'],host=platform.node(),source_sha256=source_hashes(),
            framework=dict(batch_size=a.batch_size,streams=1,budget_gib=6,bulk_target_bytes=a.bulk_target,smd_events=int(os.environ['PS_SMD_N_EVENTS'])),
            runtime_files={n:dict(path=importlib.import_module(n).__file__,sha256=sha(importlib.import_module(n).__file__)) for n in modules})
        write(a.output,result);print(json.dumps(dict(mode=a.mode,events=a.events,gib_s=result['gib_s'],hz=result['hz'])),flush=True)
    comm.Barrier();run.close_shared_memory();managers.clear();del run,ds;gc.collect();comm.Barrier()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--inputs',required=True);p.add_argument('--output',required=True);p.add_argument('--mode',choices=('sim10-field',),required=True);p.add_argument('--events',type=int,choices=(200,10000),default=10000);p.add_argument('--preflight',action='store_true');p.add_argument('--batch-size',type=int,choices=(100,),required=True)
    p.add_argument('--bulk-target',type=int,choices=(524288,4194304,8388608),required=True)
    try:main(p.parse_args())
    except BaseException:
        import traceback;traceback.print_exc();comm.Abort(1)
