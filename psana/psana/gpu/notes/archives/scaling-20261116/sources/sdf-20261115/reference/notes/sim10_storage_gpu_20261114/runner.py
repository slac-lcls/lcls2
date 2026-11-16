"""Paired cold Weka/NVMe, four fixed BDs across one, two or four GPUs."""
from plan import BASE
import argparse,gc,importlib,json,os,platform,subprocess,sys,time
from pathlib import Path
from contextlib import closing
from support import NOTES,raw,require,sha,cache,physical_counters,counter_delta,affinity,source_hashes,write,histogram,EVENT_BYTES,BULK,STREAMS,BUDGET
from cpu_plan import BATCH_SIZES,validate_case
from plan import source_hashes, placement
from storage import cache, counters, evidence
from task_runtime import kvikio_identity
from host_memory import snapshot, validate_host_memory, demand
rank_hint=int(os.environ['OMPI_COMM_WORLD_RANK'])
gpu_plan=json.loads(os.environ['SIM10_GPU_PLAN'])
require(gpu_plan==placement(gpu_plan['allocation_uuids'],gpu_plan['gpus']),'GPU plan')
os.environ['CUDA_VISIBLE_DEVICES']=gpu_plan['rank_uuids'][rank_hint]
peers=gpu_plan['peers'].get(os.environ['CUDA_VISIBLE_DEVICES'],0)
inherited_affinity=sorted(os.sched_getaffinity(0))
chosen=affinity()
os.environ['KVIKIO_NTHREADS']=str(json.loads(os.environ['SIM10_CPU_PLAN'])['workers_per_bd'])
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

def main(args):
    validate_case(args.bds,args.workers,args.batch,args.bounce_mib,args.streams,args.events)
    require(args.bds==4 and args.workers==4 and args.batch==200 and args.bounce_mib==16,'Fixed controls')
    require(args.gpus==gpu_plan['gpus'],'GPU count mismatch')
    cpu_assignment=json.loads(os.environ['SIM10_CPU_PLAN'])
    require(cpu_assignment['bds']==args.bds and cpu_assignment['workers_per_bd']==args.workers,'Worker settings/CPU plan mismatch')
    require(comm.size==args.bds+2 and comm.rank==rank_hint,'Require SMD0, EB, N BDs')
    require(os.environ.get('PS_FAKESTEP_FLAG')=='0' and os.environ.get('PS_PARALLEL')=='mpi','Runtime flags')
    info=json.loads(Path(args.inputs).read_text());stamps=info['timestamps'][:args.events]
    require(info['media']==args.media,'Storage selection mismatch')
    files=list(map(Path,info['files']));stage=Path(info['directory']);leaves=info['observed_leaves']
    require(len(stamps)==args.events,'Event reference')
    sys.path.insert(0,str(NOTES/'mfx_expand10_stage2_20261105'));import adapter;adapter.register_reader()
    host_memory=dict(startup=snapshot(), demand=demand(args.bds,args.workers,args.bounce_mib))
    task=StagingTask();managers=[];cpu_reads=[0];device=None;memory=None;io=None
    if rank_hint>=2:
        import cupy as cp
        import kvikio
        import kvikio.defaults as defaults
        require(kvikio.__version__.startswith('26.10.'),'Require KvikIO26.10')
        kvikio_files=kvikio_identity()
        require(kvikio_files==json.loads(Path(os.environ['SIM10_NATIVE_RECEIPT']).read_text())['kvikio_identity'],'KvikIO runtime identity mismatch')
        from psana.gpu.gpu_events import GpuEventManager
        from psana.psexp.event_manager import EventManager
        from bulk_accounting import accounting
        require(int(os.environ['KVIKIO_TASK_SIZE'])==1<<20,'Task environment mismatch')
        require(int(os.environ['KVIKIO_BOUNCE_BUFFER_SIZE'])==args.bounce_mib<<20,'Bounce environment mismatch')
        settings=dict(compat_mode=kvikio.CompatMode.ON,num_threads=args.workers,task_size=1<<20,bounce_buffer_size=args.bounce_mib<<20,auto_direct_io_read=False)
        defaults.set(settings);require(all(defaults.get(key)==value for key,value in settings.items()),'Settings mismatch')
        configure=GpuEventManager._setup_gpu_pipeline
        def configured(manager,*args,**kwargs):
            configure(manager,*args,**kwargs)
            require(manager.gpu_reader.task_size in (None,1<<20),'Reader task override mismatch')
            managers.append(manager)
        GpuEventManager._setup_gpu_pipeline=configured
        original=EventManager._read
        def read(manager,*args,**kwargs):cpu_reads[0]+=1;return original(manager,*args,**kwargs)
        EventManager._read=read
        require(cp.cuda.runtime.getDeviceCount()==1,'One visible GPU required')
        device=dict(kvikio_identity=kvikio_files,mask=os.environ['CUDA_VISIBLE_DEVICES'],pci=cp.cuda.runtime.deviceGetPCIBusId(0),name=cp.cuda.runtime.getDeviceProperties(0)['name'].decode(),cupy=cp.__version__,kvikio=kvikio.__version__,cuda_runtime=cp.cuda.runtime.runtimeGetVersion(),compat_mode='ON',effective={key:defaults.get(key) for key in settings if key!='compat_mode'})
    ds=DataSource(exp=info['exp'],run=info['run'],dir=info['directory'],detectors=['jungfrau'],gpu_det='jungfrau',live=False,max_events=args.events,
        skip_calib_load='all',gpu_bulk_read=True,gpu_bulk_target_bytes=BULK,batch_size=args.batch,n_gpu_streams=args.streams,gpu_memory_budget_gb=BUDGET,gpu_d2h_chunk_size=0,
        gpu_fn=GpuTask(task,inputs=task.inputs,calibconst=()),log_level='ERROR')
    run=next(ds.runs());role=run.comms.node_type();require(role==('smd0' if rank_hint==0 else 'eb' if rank_hint==1 else 'bd'),'Roles')
    if rank_hint>=2:
        cp.cuda.Device().synchronize();free,total=cp.cuda.runtime.memGetInfo();memory=dict(before_free=free,total=total)
        require(free >= (peers*BUDGET+4)*(1<<30),'Insufficient physical-GPU headroom for aggregate per-BD budgets')
    from cpu_plan import thread_snapshot
    threads_before=thread_snapshot()
    host_memory['before_loop']=snapshot()
    require(host_memory['before_loop']['mem_available_bytes']>=host_memory['demand']['aggregate_estimate_bytes']+(8<<30),'Host headroom for bounce buffers')
    comm.Barrier();observation=None
    if rank_hint==0:
        idle_before=counters(leaves);time.sleep(2);idle_after=counters(leaves)
        conditioning=cache(files,stage,args.media,True);initial=cache(files,stage,args.media)
        require(all(item['resident_pages']==0 for item in initial),'Cold gate')
        observation=dict(conditioning=conditioning,initial_cache=initial,idle_before=idle_before,idle_after=idle_after,before=counters(leaves))
        start=time.perf_counter()
    comm.Barrier();delivered=[]
    trace_window=None
    if rank_hint>=2:
        tracer_pid=int(next(line.split(':')[1] for line in Path('/proc/self/status').read_text().splitlines() if line.startswith('TracerPid:')))
        require((tracer_pid>0)==(args.events<100000),'Only pilots must be traced; full timing runs must not be traced')
        trace_window=dict(tracer_pid=tracer_pid,start=time.time())
        with accounting() as io:
            with closing(run.events()) as events:
                for event in events:delivered.append(int(event.timestamp))
            cp.cuda.Device().synchronize()
        trace_window['end']=time.time()
    else:
        with closing(run.events()) as events:
            for event in events:delivered.append(int(event.timestamp))
    comm.Barrier()
    if rank_hint==0:
        seconds=time.perf_counter()-start;observation['seconds']=seconds;observation['after']=counters(leaves)
        observation['storage_evidence']=evidence(observation,args.media,EVENT_BYTES*args.events,seconds)
        observation['final_cache']=cache(files,stage,args.media)
    
    comm.Barrier()
    if rank_hint>=2:
        count=len(delivered);require(count>0 and delivered==sorted(set(delivered)),'Local ordered unique events')
        require([timestamp for callback in task.callbacks for timestamp in callback['timestamps']]==delivered,'Callbacks vs delivery')
        require(len(task.callbacks)==count//args.batch and all(callback['size']==args.batch for callback in task.callbacks),'Unexpected execution splitting: callback count/size differs from nominal batch')
        require(len({callback['stream'] for callback in task.callbacks})==args.streams,'Requested execution streams must be exercised')
        require(all(0<callback['size']<=args.batch and callback['size']==len(callback['timestamps']) for callback in task.callbacks),'Callback sizes')
        require(io['requests']==io['completed_requests']==5*count and io['requested_bytes']==io['completed_bytes']==EVENT_BYTES*count,'Exact GPU I/O')
        require(io['read_size_histogram']==histogram(count),'Read size histogram')
        require(io['issue_groups_calls']==len(task.callbacks),'PR298 batch issue count')
        require(cpu_reads[0]==io['pressure_drains']==0,'CPU payload or admission drain')
        require(len(managers)==1 and all(manager._closed and not manager.gpu_reader._pending and not manager.event_pool.active_count and manager._gpu_budget._held==0 and manager._n_bd_per_gpu==peers for manager in managers),'Drained resources/peers')
        free,total=cp.cuda.runtime.memGetInfo();memory.update(after_free=free,pool_used=cp.get_default_memory_pool().used_bytes(),pool_total=cp.get_default_memory_pool().total_bytes())
    host_memory['after_loop']=snapshot()
    validate_host_memory(host_memory,args.bds,args.workers,args.bounce_mib)
    if rank_hint>=2:
        require(all(defaults.get(key)==value for key,value in settings.items()),'Settings changed during loop')
    drain=[dict(closed=manager._closed,pending=len(manager.gpu_reader._pending),active=manager.event_pool.active_count,held=manager._gpu_budget._held,peers=manager._n_bd_per_gpu) for manager in managers]
    modules=('psana.gpu.gpu_input_group','psana.gpu.gpu_input_window','psana.dgram','psana.eventbuilder','psana.smdreader','psana','psana.gpu.gpu_events','psana.gpu.gpu_detector','psana.gpu.gpu_task','psana.gpu.gpu_kvikio_read','psana.gpu.gpu_placement','psana.gpu.gpu_stream_read_plan','psana.gpu.gpu_group_schedule')
    runtime=identity(modules)
    row=dict(host_memory=host_memory,reader_task_sizes=[manager.gpu_reader.task_size for manager in managers],trace_window=trace_window,cpu_plan=json.loads(os.environ['SIM10_CPU_PLAN']),threads_before=threads_before,threads_after=thread_snapshot(),drain=drain,rank=rank_hint,role=role,host=platform.node(),inherited_affinity=inherited_affinity,affinity=chosen,device=device,memory=memory,timestamps=delivered,callbacks=task.callbacks,io=io,cpu_reads=cpu_reads[0],runtime_files=runtime,
        numa_policy=subprocess.run(['numactl','--show'],capture_output=True,text=True,check=True).stdout)
    require('policy: interleave' in row['numa_policy'],'Expected interleaved page allocations')
    rows=comm.gather(row,root=0)
    if rank_hint==0:
        bds=rows[2:];merged=[timestamp for record in bds for timestamp in record['timestamps']]
        require(sorted(merged)==stamps and len(set(merged))==args.events,'Global exact event partition')
        require(all(not record['timestamps'] for record in rows[:2]),'Helper events')
        require([record['device']['mask'] for record in bds]==gpu_plan['rank_uuids'][2:],'Physical GPU assignment mismatch')
        require(len({record['device']['pci'] for record in bds})==args.gpus,'Physical GPU count mismatch')
        require(all(record['device']['effective']==bds[0]['device']['effective'] for record in bds),'KvikIO settings mismatch')
        require(sum(record['io']['completed_bytes'] for record in bds)==EVENT_BYTES*args.events,'Global bytes')
        require(sum(len(record['callbacks']) for record in bds)==args.events//args.batch,'Global batch count')
        result=dict(status='pass',settings=vars(args),events=args.events,bytes=EVENT_BYTES*args.events,seconds=seconds,gib_s=EVENT_BYTES*args.events/(1<<30)/seconds,hz=args.events/seconds,
            gpu_plan=gpu_plan,rows=rows,measurement=observation,job=os.environ['SLURM_JOB_ID'],host=platform.node(),source_sha256=source_hashes(),
            framework=dict(batch_size=args.batch,streams=args.streams,budget_gib=BUDGET,bulk_target_bytes=BULK,smd_events=int(os.environ['PS_SMD_N_EVENTS'])))
        write(args.output,result);print(json.dumps(dict(bds=args.bds,events=args.events,gib_s=result['gib_s'],hz=result['hz'])),flush=True)
    comm.Barrier();run.close_shared_memory();managers.clear();del run,ds;gc.collect();comm.Barrier()

if __name__=='__main__':
    cpu_assignment=argparse.ArgumentParser();cpu_assignment.add_argument('--inputs',required=True);cpu_assignment.add_argument('--output',required=True);cpu_assignment.add_argument('--bds',type=int,choices=(1,4),required=True);cpu_assignment.add_argument('--events',type=int,choices=(1000,2000,3000,4000,100000),default=100000)
    cpu_assignment.add_argument('--media',choices=('weka','nvme'),required=True);cpu_assignment.add_argument('--gpus',type=int,choices=(1,2,4),required=True)
    cpu_assignment.add_argument('--bounce-mib',type=int,choices=(16,64),required=True)
    cpu_assignment.add_argument('--workers',type=int,choices=(4,8),required=True)
    cpu_assignment.add_argument('--streams',type=int,choices=(1,),default=1);cpu_assignment.add_argument('--batch',type=int,choices=BATCH_SIZES,required=True)
    try:main(cpu_assignment.parse_args())
    except BaseException:
        import traceback;traceback.print_exc();comm.Abort(1)
