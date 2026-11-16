"""Exact PR298 staging: configurable stripe,depth,BDs; fixed batch/KvikIO controls."""
import argparse,gc,hashlib,json,os,platform,struct,subprocess,sys,time
from pathlib import Path
from contextlib import closing
from support import ROOT,SHA,require,sha,write,load_inputs,view,affinity,runtime,cold_gate,cache,memory,EVENT_BYTES,histogram,diagnostics
rank_hint=int(os.environ['SLURM_PROCID']);inherited_mask=os.environ.get('CUDA_VISIBLE_DEVICES','')
require(inherited_mask,'Missing allocation GPU mask')
inherited_uuids=subprocess.check_output(['nvidia-smi','-i',inherited_mask,'--query-gpu=uuid','--format=csv,noheader'],text=True).strip().splitlines()
require(os.environ['JF_GPU_UUID'] in inherited_uuids,'Selected GPU not granted')
os.environ['CUDA_VISIBLE_DEVICES']=os.environ['JF_GPU_UUID'] if rank_hint>=2 else ''
chosen=affinity()
from mpi4py import MPI
from psana import DataSource
from psana.gpu import GpuTask
from adapter import register_reader
from instrument import accounting,bindings,budget_peaks
comm=MPI.COMM_WORLD

class Task:
    inputs=(('jungfrau','sim10','seed'),)
    def __init__(self,witness):self.callbacks=[];self.witness=witness;self.witness_values=0;self.kernel=None
    def __call__(self,batch,stream):
        field=batch.field(*self.inputs[0]);require(tuple(field.segment_ids)==tuple(range(32)) and field.rows.shape==(batch.size,32,8),'Field ABI/panels')
        self.callbacks.append({'timestamps':list(batch.timestamps),'size':batch.size,'stream':int(stream.ptr)})
        if self.witness is not None:
            import cupy as cp,numpy as np
            from psana.gpu.gpudgram.batch import LOC_NCOLS,LOC_OFFSET,LOC_NBYTES,LOC_STATUS,LOC_DIM0
            from psana.gpu.gpudgram.parser import STATUS_FOUND
            code=r'''extern "C" __global__ void witness(const unsigned long long* f,unsigned short* out,unsigned int* errors,int n){
                int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;const auto* q=f+i*8;
                if(!q[7]||!q[0]||!q[2]){errors[i]=1;return;}
                const auto* loc=(const unsigned long long*)q[2]+q[3]*@NC@;
                unsigned long long offset=loc[@OFF@];
                if(loc[@STATUS@]!=@FOUND@||loc[@DIM@]!=52429||loc[@BYTES@]!=104858||offset>q[1]||104858>q[1]-offset){errors[i]=2;return;}
                const auto* raw=(const unsigned char*)q[0]+offset;int indices[3]={0,26214,52428};
                for(int j=0;j<3;j++){int k=indices[j]*2;out[i*3+j]=(unsigned short)raw[k]|((unsigned short)raw[k+1]<<8);}errors[i]=0;
            }'''
            for k,v in {'NC':LOC_NCOLS,'OFF':LOC_OFFSET,'BYTES':LOC_NBYTES,'STATUS':LOC_STATUS,'DIM':LOC_DIM0,'FOUND':STATUS_FOUND}.items():code=code.replace('@'+k+'@',str(v))
            with stream:
                if self.kernel is None:self.kernel=cp.RawKernel(code,'witness',options=('--std=c++17',))
                out=cp.empty((batch.size,32,3),cp.uint16);errors=cp.empty((batch.size,32),cp.uint32)
                batch.keepalive(out);batch.keepalive(errors)
                self.kernel(((batch.size*32+127)//128,),(128,),(field.rows,out,errors,np.int32(batch.size*32)),stream=stream)
                stream.synchronize();got=cp.asnumpy(out);err=cp.asnumpy(errors)
                require(not np.any(err),'GPU seed locator error')
                expected=np.array([[self.witness[str(t)][str(p)] for p in range(32)] for t in batch.timestamps],dtype=np.uint16)
                require(np.array_equal(got,expected),'GPU seed values differ from CPU');self.witness_values+=int(got.size)

def main(a):
    require(comm.rank==rank_hint and comm.size==a.bds+2,'MPI size/role')
    pins=json.loads((Path(__file__).parent/'external-pins.json').read_text())
    require(all(sha(path)==h for path,h in pins.items()),'External input/build manifest changed')
    info=load_inputs();g=view(info,a.count);stamps=info['timestamps'][:a.events]
    register_reader();task=Task(json.loads(Path(a.witness).read_text()) if a.witness else None)
    ranges=json.loads(Path(a.ranges).read_text());ts_index={t:i for i,t in enumerate(info['timestamps'])}
    managers=[];cpu_reads=[0];device=None;mem=None;io=None;warmup=None
    if rank_hint>=2:
        import cupy as cp,kvikio,kvikio.defaults as defaults
        from psana.gpu.gpu_events import GpuEventManager
        from psana.psexp.event_manager import EventManager
        settings=dict(compat_mode=kvikio.CompatMode.ON,num_threads=4,task_size=1<<20,bounce_buffer_size=16<<20,auto_direct_io_read=False)
        defaults.set(settings);require(all(defaults.get(k)==v for k,v in settings.items()),'KvikIO defaults')
        setup=GpuEventManager._setup_gpu_pipeline
        def configured(m,*args,**kw):setup(m,*args,**kw);managers.append(m)
        GpuEventManager._setup_gpu_pipeline=configured
        old=EventManager._read
        def read(m,*args,**kw):cpu_reads[0]+=1;return old(m,*args,**kw)
        EventManager._read=read
        require(cp.cuda.runtime.getDeviceCount()==1,'OneGPUvisible')
        from psana.gpu.gpudgram import parser
        warm_start=time.time()
        for factory in (parser._walk_kernel,parser._locate_kernel,parser._init_locators_kernel,parser._locate_fields_kernel):factory().compile()
        warmup={'started':warm_start,'finished':time.time(),'kernels':['walk_xtc','locate_field','init_locators','locate_fields'],'payload_reads':0}
        props=cp.cuda.runtime.getDeviceProperties(0)
        require('A100' in props['name'].decode() and 39*(1<<30)<props['totalGlobalMem']<41*(1<<30),'RequireA10040GB')
        device={'uuid':os.environ['JF_GPU_UUID'],'pci':cp.cuda.runtime.deviceGetPCIBusId(0),'name':props['name'].decode(),'cupy':cp.__version__,'kvikio':kvikio.__version__,'cuda_runtime':cp.cuda.runtime.runtimeGetVersion(),'compat_mode':'ON','effective':{k:defaults.get(k) for k in settings if k!='compat_mode'}}
    with budget_peaks() as peaks, bindings(g['files']) as handles:
        ds=DataSource(exp=info['exp'],run=info['run'],dir=g['directory'],detectors=['jungfrau'],gpu_det='jungfrau',live=False,max_events=a.events,skip_calib_load='all',gpu_bulk_read=True,gpu_bulk_target_bytes=524288,batch_size=200,n_gpu_streams=a.streams,gpu_memory_budget_gb=6,gpu_d2h_chunk_size=0,gpu_fn=GpuTask(task,inputs=task.inputs,calibconst=()),log_level='ERROR')
        run=next(ds.runs());role=run.comms.node_type();require(role==('smd0' if rank_hint==0 else 'eb' if rank_hint==1 else 'bd'),'Rankrole')
        if rank_hint>=2:cp.cuda.Device().synchronize();free,total=cp.cuda.runtime.memGetInfo();mem={'before_free':free,'total':total}
        comm.Barrier();observation=None
        if rank_hint==0:
            gate_start=time.time();gate=cold_gate(info,a.count);gate_end=time.time()
            observation={'cache':gate,'gate_started':gate_start,'gate_finished':gate_end,'diagnostics_before':diagnostics(),'memory_before':memory(),'started_unix':time.time()}
        comm.Barrier();start=time.perf_counter();released=time.time();delivered=[]
        if rank_hint>=2:
            with accounting() as io:
                with closing(run.events()) as events:
                    for event in events:delivered.append(int(event.timestamp))
                cp.cuda.Device().synchronize()
        else:
            with closing(run.events()) as events:
                for event in events:delivered.append(int(event.timestamp))
        completed=time.time();comm.Barrier();elapsed=time.perf_counter()-start
        if rank_hint==0:observation.update(finished_unix=time.time(),memory_after=memory(),diagnostics_after=diagnostics())
        if rank_hint>=2:
            n=len(delivered)
            if a.kind in ('sizing','screen'):require(n>0,'Full case must exercise every BD')
            require(delivered==sorted(set(delivered)),'Localorder')
            require([t for c in task.callbacks for t in c['timestamps']]==delivered,'Callback identities')
            if a.kind!='binding' and (a.kind in ('sizing','screen') and n or len(task.callbacks)>=a.streams):require(len({c['stream'] for c in task.callbacks})==a.streams,'Exercise requested streams')
            require(all(0<c['size']<=200 and len(c['timestamps'])==c['size'] for c in task.callbacks),'Callbacksize')
            require(io['requests']==io['completed_requests']==5*n and io['requested_bytes']==io['completed_bytes']==EVENT_BYTES*n and io['read_size_histogram']==(histogram(n) if n else {}),'Exactbytes/requests')
            wanted={}
            for r in g['files']:
                name=Path(r['path']).name;ref=ranges[name];h=hashlib.sha256()
                for t in delivered:h.update(struct.pack('<QQQ',t,ref['offsets'][ts_index[t]],ref['size']))
                if n:wanted[r['path']]=h.hexdigest()
            require(io['range_sha256']==wanted,'Physical range identity digest')
            require(cpu_reads[0]==io['pressure_drains']==0,'CPU reads/pressure drains')
            require((len(managers)==1 if n else len(managers)<=1) and all(m._closed and not m.gpu_reader._pending and not m.event_pool.active_count and m._gpu_budget._held==0 and m._n_bd_per_gpu==a.bds for m in managers),'Resources/peers')
            require(set(x['primary']['path'] for x in handles['kvikio'])==set(wanted),'Five actual KvikIO files')
            require(set(x['path'] for x in handles['dgram'])==set(r['path'] for r in g['files']),'Five actual DgramManager files')
            free,total=cp.cuda.runtime.memGetInfo();mem.update(after_free=free,pool_used=cp.get_default_memory_pool().used_bytes(),pool_total=cp.get_default_memory_pool().total_bytes())
        admission=[{'capacity':m._admission_capacity,'margin':m._admission_margin,'budget':m._gpu_budget.limit(),'peers':m._n_bd_per_gpu,'placement':vars(m._placement).copy()} for m in managers]
        # Placement can contain communicators; serialize scalar evidence only.
        for r in admission:r['placement']={k:v for k,v in r['placement'].items() if v is None or isinstance(v,(str,int,float,bool))}
        row={'rank':rank_hint,'role':role,'released_unix':released,'completed_unix':completed,'compile_warmup':warmup,'elapsed':elapsed,'host':platform.node(),'affinity':chosen,'inherited_mask':inherited_mask,'inherited_uuids':inherited_uuids,'device':device,'memory':mem,'timestamps':delivered,'callbacks':task.callbacks,'io':io,'cpu_reads':cpu_reads[0],'handles':handles,'admission':admission,'drain':[{'closed':m._closed,'pending':len(m.gpu_reader._pending),'active':m.event_pool.active_count,'held':m._gpu_budget._held} for m in managers],'runtime':runtime(),'budget_high_water':list(peaks.values()),'witness_values':task.witness_values,'mpi':MPI.Get_library_version(),'numa':subprocess.check_output(['numactl','--show'],text=True)}
        require('CRAY MPICH' in row['mpi'].upper() and 'policy: interleave' in row['numa'],'MPI/NUMA')
        comm.Barrier();run.close_shared_memory();managers.clear();del run,ds;gc.collect();comm.Barrier()
        row['cleanup_complete']=True
    rows=comm.gather(row,root=0)
    if rank_hint==0:
        bds=rows[2:];merged=[t for r in bds for t in r['timestamps']];require(sorted(merged)==stamps and len(set(merged))==a.events,'Global events once')
        require(all(not r['timestamps'] for r in rows[:2]),'Helper events')
        require(all(r['device']==bds[0]['device'] for r in bds),'PhysicalGPUsettings')
        require({str(r['rank']):r['affinity'] for r in rows}==json.loads(os.environ['JF_CPU_AFFINITY']),'Rank affinity differs')
        require(any(len({c['stream'] for c in r['callbacks']})==a.streams for r in bds),'Requested depth never exercised')
        sizes=[c['size'] for r in bds for c in r['callbacks']]
        require(sorted(sizes)==sorted([200]*(a.events//200)+([a.events%200] if a.events%200 else [])),'Global callback sizes')
        seconds=max(r['elapsed'] for r in rows)
        write(a.output,{'state':'pass','parameters':dict(vars(a),batch_size=200,workers_per_bd=4),'execution_sha':SHA,'events':a.events,'bytes':EVENT_BYTES*a.events,'seconds':seconds,'gib_s':EVENT_BYTES*a.events/(1<<30)/seconds,'events_s':a.events/seconds,'rows':rows,'measurement':observation,'job':os.environ['SLURM_JOB_ID'],'host':platform.node(),'scope':'descriptor staging; local page cache cold; backend/remote traffic unverified'})
        print(json.dumps({'count':a.count,'bds':a.bds,'streams':a.streams,'events':a.events,'seconds':seconds,'gib_s':EVENT_BYTES*a.events/(1<<30)/seconds}),flush=True)
    comm.Barrier()
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--count',type=int,choices=(1,2,4,8,16,32),required=True);p.add_argument('--bds',type=int,choices=(4,),required=True);p.add_argument('--streams',type=int,choices=(1,),required=True);p.add_argument('--events',type=int,choices=(100,1000,1003,100000),required=True);p.add_argument('--kind',choices=('binding','tail','sizing','screen'),required=True);p.add_argument('--output',required=True);p.add_argument('--ranges',required=True);p.add_argument('--witness')
    try:main(p.parse_args())
    except BaseException:
        import traceback;traceback.print_exc();comm.Abort(1)
