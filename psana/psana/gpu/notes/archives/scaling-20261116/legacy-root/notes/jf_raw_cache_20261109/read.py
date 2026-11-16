"""Kernel-free KvikIO ring over exactly the JF event datagrams."""
import argparse,json,os,platform,time
from pathlib import Path
from common import require,paths,requests,run_queue,begin,end,sources,CacheGateError,cache_unavailable

def main(a):
    os.environ['CUDA_VISIBLE_DEVICES']=os.environ['JF_GPU_UUID']
    cores=[int(x) for x in os.environ['BENCH_CPU_AFFINITY'].split(',')];os.sched_setaffinity(0,cores[:a.threads])
    import cupy as cp
    import kvikio
    import kvikio.defaults as defaults
    settings=dict(compat_mode=kvikio.CompatMode.ON,num_threads=a.threads,task_size=a.task_mib<<20,bounce_buffer_size=16<<20,auto_direct_io_read=False)
    defaults.set(settings);require(all(defaults.get(k)==v for k,v in settings.items()),'Settings mismatch')
    require(cp.cuda.runtime.getDeviceCount()==1,'One GPU required')
    plan,stamps=requests(a.directory);files=paths(a.directory);depth=35
    buffers=[cp.empty(max(r[2] for r in plan),cp.uint8) for _ in range(depth)]
    handles=[kvikio.CuFile(str(p),'r') for p in files]
    def submit(slot,r):return handles[r[0]].pread(buffers[slot],size=r[2],file_offset=r[1],task_size=a.task_mib<<20)
    try:
        run_queue(plan[:5],5,submit,lambda f:int(f.get()));cp.cuda.Device().synchronize()
        try:observation=begin(a.directory,a.cache)
        except CacheGateError as exc:
            cache_unavailable(a.output,vars(a),exc);raise SystemExit(42)
        start=time.perf_counter()
        total,count,last=run_queue(plan,depth,submit,lambda f:int(f.get()));cp.cuda.Device().synchronize();seconds=time.perf_counter()-start
        end(observation,a.directory,seconds);require(count==50000 and total==335571760000,'Incomplete reads')
        witnesses=[]
        for slot,(i,offset,size) in sorted(last.items()):
            fd=os.open(files[i],os.O_RDONLY)
            try:
                for at in (0,size//2-32,size-64):
                    actual=buffers[slot][at:at+64].get().tobytes();expected=os.pread(fd,64,offset+at)
                    require(actual==expected and len(actual)==64,'GPU/CPU witness mismatch');witnesses.append(dict(slot=slot,file=i,offset=offset+at,size=64))
            finally:os.close(fd)
        result=dict(status='pass',workload='standalone-kvikio',settings=vars(a),host=platform.node(),job=os.getenv('SLURM_JOB_ID'),bytes=total,requests=count,events=10000,gib_s=total/(1<<30)/seconds,measurement=observation,witnesses=witnesses,depth=depth,device_buffer_bytes=sum(b.nbytes for b in buffers),cpu_affinity=sorted(os.sched_getaffinity(0)),device=dict(mask=os.environ['CUDA_VISIBLE_DEVICES'],pci=cp.cuda.runtime.deviceGetPCIBusId(0),name=cp.cuda.runtime.getDeviceProperties(0)['name'].decode(),cupy=cp.__version__,kvikio=kvikio.__version__,compat_mode='ON',effective={k:defaults.get(k) for k in settings if k!='compat_mode'}),source_sha256=sources())
        with open(a.output,'x') as f:json.dump(result,f,indent=2);f.write('\n')
        print(json.dumps(dict(status='pass',gib_s=result['gib_s'],output=a.output)),flush=True)
    finally:
        cp.cuda.Device().synchronize()
        for h in handles:h.close()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--directory',required=True);p.add_argument('--output',required=True);p.add_argument('--cache',choices=['cold','warm'],required=True);p.add_argument('--threads',type=int,choices=[4,8],required=True);p.add_argument('--task-mib',type=int,choices=[1,4],required=True);main(p.parse_args())
