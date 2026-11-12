"""Cold FFB, one physical GPU, one BD x one/two streams; descriptor staging only."""
import argparse,gc,importlib,json,os,platform,subprocess,sys,time
from pathlib import Path
from contextlib import closing
from support import NOTES,raw,require,sha,cache,physical_counters,counter_delta,affinity,source_hashes,write,histogram,EVENT_BYTES,BULK,BATCH,STREAMS,BUDGET,WORKERS
rank_hint=int(os.environ['OMPI_COMM_WORLD_RANK'])
os.environ['CUDA_VISIBLE_DEVICES']=os.environ['JF_GPU_UUID'] if rank_hint>=2 else ''
chosen=affinity()
from runtime import activate, identity
activate()
from mpi4py import MPI
from psana import DataSource
from psana.gpu import GpuTask
comm=MPI.COMM_WORLD

class StagingTask:
    inputs=(('jungfrau','sim10','seed'),)
    def __init__(self):self.callbacks=[]
    def __call__(self,batch,stream):
        field=batch.field(*self.inputs[0])
        require(tuple(field.segment_ids)==tuple(range(32)) and field.rows.shape==(batch.size,32,8),'Descriptor shape/panels')
        self.callbacks.append(dict(timestamps=list(batch.timestamps),size=batch.size,stream=int(stream.ptr)))

def main(a):
    require(comm.size==a.bds+2 and comm.rank==rank_hint,'Require SMD0, EB, N BDs')
    require(os.environ.get('PS_FAKESTEP_FLAG')=='0' and os.environ.get('PS_PARALLEL')=='mpi','Runtime flags')
    info=json.loads(Path(a.inputs).read_text());stamps=info['timestamps'][:a.events]
    require(len(stamps)==a.events,'Event reference')
    sys.path.insert(0,str(NOTES/'mfx_expand10_stage2_20261105'));import adapter;adapter.register_reader()
    task=StagingTask();managers=[];cpu_reads=[0];device=None;memory=None;io=None
    if rank_hint>=2:
        import cupy as cp
        import kvikio
        import kvikio.defaults as defaults
        from psana.gpu.gpu_events import GpuEventManager
        from psana.psexp.event_manager import EventManager
        from bulk_accounting import accounting
        settings=dict(compat_mode=kvikio.CompatMode.ON,num_threads=WORKERS,task_size=1<<20,bounce_buffer_size=16<<20,auto_direct_io_read=False)
        defaults.set(settings);require(all(defaults.get(k)==v for k,v in settings.items()),'Settings mismatch')
        configure=GpuEventManager._setup_gpu_pipeline
        def configured(m,*args,**kwargs):configure(m,*args,**kwargs);managers.append(m)
        GpuEventManager._setup_gpu_pipeline=configured
        original=EventManager._read
        def read(m,*args,**kwargs):cpu_reads[0]+=1;return original(m,*args,**kwargs)
        EventManager._read=read
        require(cp.cuda.runtime.getDeviceCount()==1,'One visible GPU required')
        device=dict(mask=os.environ['CUDA_VISIBLE_DEVICES'],pci=cp.cuda.runtime.deviceGetPCIBusId(0),name=cp.cuda.runtime.getDeviceProperties(0)['name'].decode(),cupy=cp.__version__,kvikio=kvikio.__version__,cuda_runtime=cp.cuda.runtime.runtimeGetVersion(),compat_mode='ON',effective={k:defaults.get(k) for k in settings if k!='compat_mode'})
    ds=DataSource(exp=info['exp'],run=info['run'],dir=info['directory'],detectors=['jungfrau'],gpu_det='jungfrau',live=False,max_events=a.events,
        skip_calib_load='all',gpu_bulk_read=True,gpu_bulk_target_bytes=BULK,batch_size=BATCH,n_gpu_streams=a.streams,gpu_memory_budget_gb=BUDGET,gpu_d2h_chunk_size=0,
        gpu_fn=GpuTask(task,inputs=task.inputs,calibconst=()),log_level='ERROR')
    run=next(ds.runs());role=run.comms.node_type();require(role==('smd0' if rank_hint==0 else 'eb' if rank_hint==1 else 'bd'),'Roles')
    if rank_hint>=2:
        cp.cuda.Device().synchronize();free,total=cp.cuda.runtime.memGetInfo();memory=dict(before_free=free,total=total)
    comm.Barrier();observation=None
    if rank_hint==0:
        before=physical_counters();time.sleep(1);after=physical_counters()
        files=list(map(Path,info['files']));conditioning=cache(files,True);initial=cache(files)
        require(all(x['resident_pages']==0 for x in initial),'Cold gate')
        observation=dict(conditioning=conditioning,initial_cache=initial,idle=dict(before=before,after=after,delta=counter_delta(before,after)),nic_before=physical_counters())
        start=time.perf_counter()
    comm.Barrier();delivered=[]
    if rank_hint>=2:
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
        observation['nic_delta']=counter_delta(observation['nic_before'],observation['nic_after'],EVENT_BYTES*a.events)
        require(all(x['seconds']>=seconds for x in observation['nic_delta'].values()),'NIC bracket')
        observation['final_cache']=cache(files)
    if rank_hint>=2:
        n=len(delivered);require(n>0 and delivered==sorted(set(delivered)),'Local ordered unique events')
        require([t for c in task.callbacks for t in c['timestamps']]==delivered,'Callbacks vs delivery')
        require(len(task.callbacks)==a.events//BATCH and all(c['size']==BATCH for c in task.callbacks),'Exact callback count and size')
        require(len({c['stream'] for c in task.callbacks})==a.streams,'Requested execution streams must be exercised')
        require(all(0<c['size']<=BATCH and c['size']==len(c['timestamps']) for c in task.callbacks),'Callback sizes')
        require(io['requests']==io['completed_requests']==5*n and io['requested_bytes']==io['completed_bytes']==EVENT_BYTES*n,'Exact GPU I/O')
        require(io['read_size_histogram']==histogram(n),'Read size histogram')
        require(cpu_reads[0]==io['pressure_drains']==0,'CPU payload or admission drain')
        require(len(managers)==1 and all(m._closed and not m.gpu_reader._pending and not m.event_pool.active_count and m._gpu_budget._held==0 and m._n_bd_per_gpu==a.bds for m in managers),'Drained resources/peers')
        free,total=cp.cuda.runtime.memGetInfo();memory.update(after_free=free,pool_used=cp.get_default_memory_pool().used_bytes(),pool_total=cp.get_default_memory_pool().total_bytes())
    drain=[dict(closed=m._closed,pending=len(m.gpu_reader._pending),active=m.event_pool.active_count,held=m._gpu_budget._held,peers=m._n_bd_per_gpu) for m in managers]
    modules=('psana.gpu.gpu_input_group','psana.gpu.gpu_input_window','psana.dgram','psana.eventbuilder','psana.smdreader','psana','psana.gpu.gpu_events','psana.gpu.gpu_detector','psana.gpu.gpu_task','psana.gpu.gpu_kvikio_read','psana.gpu.gpu_placement','psana.gpu.gpu_stream_read_plan','psana.gpu.gpu_group_schedule')
    runtime=identity(modules)
    row=dict(drain=drain,rank=rank_hint,role=role,host=platform.node(),affinity=chosen,device=device,memory=memory,timestamps=delivered,callbacks=task.callbacks,io=io,cpu_reads=cpu_reads[0],runtime_files=runtime,
        numa_policy=subprocess.run(['numactl','--show'],capture_output=True,text=True,check=True).stdout)
    require('policy: interleave' in row['numa_policy'],'Expected interleaved page allocations')
    rows=comm.gather(row,root=0)
    if rank_hint==0:
        bds=rows[2:];merged=[t for r in bds for t in r['timestamps']]
        require(sorted(merged)==stamps and len(set(merged))==a.events,'Global exact event partition')
        require(all(not r['timestamps'] for r in rows[:2]),'Helper events')
        require(all(r['device']==bds[0]['device'] for r in bds),'Physical GPU/settings mismatch')
        require(sum(r['io']['completed_bytes'] for r in bds)==EVENT_BYTES*a.events,'Global bytes')
        result=dict(status='pass',settings=vars(a),events=a.events,bytes=EVENT_BYTES*a.events,seconds=seconds,gib_s=EVENT_BYTES*a.events/(1<<30)/seconds,hz=a.events/seconds,
            rows=rows,measurement=observation,job=os.environ['SLURM_JOB_ID'],host=platform.node(),source_sha256=source_hashes(),
            framework=dict(batch_size=BATCH,streams=a.streams,budget_gib=BUDGET,bulk_target_bytes=BULK,smd_events=int(os.environ['PS_SMD_N_EVENTS'])))
        write(a.output,result);print(json.dumps(dict(bds=a.bds,events=a.events,gib_s=result['gib_s'],hz=result['hz'])),flush=True)
    comm.Barrier();run.close_shared_memory();managers.clear();del run,ds;gc.collect();comm.Barrier()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--inputs',required=True);p.add_argument('--output',required=True);p.add_argument('--bds',type=int,choices=(1,),required=True);p.add_argument('--events',type=int,choices=(1000,10000),default=10000)
    p.add_argument('--streams',type=int,choices=(1,2),required=True)
    try:main(p.parse_args())
    except BaseException:
        import traceback;traceback.print_exc();comm.Abort(1)
