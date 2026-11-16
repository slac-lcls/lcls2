"""Cold Weka FFB, matched kernel-free JF event reads. No NVMe input stage."""
import json, os, platform, shutil, signal, subprocess, sys, time
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'jf_raw_cache_20261109'))
import common as raw
from plan_reads import load_fixture, run_queue
from network_watch import physical_counters,counter_delta
require=raw.require
FFB=Path('/sdf/data/lcls/drpsrcf/ffb/users/monarin')

def write(path,value):path.write_text(json.dumps(value,indent=2)+'\n')

def cache(files,evict=False):
    rows=[]
    for p in files:
        require(p.resolve().is_relative_to(FFB) and p.stat().st_uid==os.geteuid(),'Foreign cache file')
        if evict:
            fd=os.open(p,os.O_RDONLY)
            try:
                os.fsync(fd)
                for _ in range(20):
                    os.posix_fadvise(fd,0,0,os.POSIX_FADV_DONTNEED)
                    if raw.prefix_residency(p,p.stat().st_size)['resident_pages']==0:break
                    time.sleep(.1)
            finally:os.close(fd)
        rows.append(raw.prefix_residency(p,p.stat().st_size))
    if evict:require(all(r['resident_pages']==0 for r in rows),'Cold gate failed')
    return rows

def main(output):
    output.mkdir(parents=True,exist_ok=False)
    stage=FFB/('jf-cold-sim10-'+os.environ['SLURM_JOB_ID'])
    stage.mkdir(mode=0o700)
    signal.signal(signal.SIGTERM,lambda *_:sys.exit(143))
    try:
        # Keep all CUDA work, setup, identity verification and cache conditioning untimed.
        inventory=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,pci.bus_id,name','--format=csv,noheader'],text=True)
        os.environ['CUDA_VISIBLE_DEVICES']=inventory.splitlines()[0].split(',')[0].strip()
        cores=[];seen=set()
        for cpu in sorted(os.sched_getaffinity(0)):
            topology=Path('/sys/devices/system/cpu/cpu%d/topology'%cpu)
            key=((topology/'physical_package_id').read_text(),(topology/'core_id').read_text())
            if key not in seen:cores.append(cpu);seen.add(key)
        require(len(cores)>=4,'Need four physical cores');os.sched_setaffinity(0,cores[:4])
        import cupy as cp
        import kvikio
        import kvikio.defaults as defaults
        settings=dict(compat_mode=kvikio.CompatMode.ON,num_threads=4,task_size=4<<20,bounce_buffer_size=16<<20,auto_direct_io_read=False)
        defaults.set(settings)
        require(all(defaults.get(k)==v for k,v in settings.items()),'Settings mismatch')
        require(cp.cuda.runtime.getDeviceCount()==1,'Need one visible GPU')
        rawfiles=raw.paths(raw.SOURCE);rawplan,stamps=raw.requests(raw.SOURCE)
        expected=json.loads((raw.HERE/'expected-sha256.json').read_text())
        for p in rawfiles:require(raw.sha(p)==expected[p.name],'Raw SHA mismatch')
        print('Uncompressed identity verified',flush=True)
        sources,sizes,smallplan,identity=load_fixture()
        require(shutil.disk_usage(stage).free>sum(sizes)+(2<<30),'FFB capacity')
        smallfiles=[]
        for p,row in zip(sources,identity['files']):
            target=stage/p.name;shutil.copyfile(p,target)
            require(raw.sha(target)==row['sha256'],'Sim10 copy SHA mismatch');smallfiles.append(target)
        mounts={label:subprocess.check_output(['findmnt','-J','-T',str(path)],text=True) for label,path in [('raw',raw.SOURCE),('sim10',stage)]}
        require(json.loads(mounts['raw'])['filesystems'][0]['source']==json.loads(mounts['sim10'])['filesystems'][0]['source'],'Different filesystem')
        provenance=dict(raw_sha256=expected,raw_timestamp_sha256=raw.ts_hash(stamps),sim10_identity=identity,sim10_paths=list(map(str,smallfiles)),mounts=mounts,
            host=platform.node(),job=os.environ['SLURM_JOB_ID'],cpu_affinity=sorted(os.sched_getaffinity(0)),gpu_inventory=inventory,gpu_mask=os.environ['CUDA_VISIBLE_DEVICES'],
            cupy=cp.__version__,kvikio=kvikio.__version__,effective={k:defaults.get(k) for k in settings if k!='compat_mode'},compat_mode='ON',
            source_sha256={str(p):raw.sha(p) for p in [Path(__file__),HERE/'run.sbatch',raw.HERE/'common.py',HERE.parent/'mfx_kvikio_read_20261108/plan_reads.py',HERE.parent/'mfx_kvikio_read_20261108/network_watch.py']})
        write(output/'provenance.json',provenance)
        cases=[]
        for index,label in enumerate(['raw','sim10','sim10','raw']):
            files,plan,expected_bytes=(rawfiles,rawplan,335571760000) if label=='raw' else (smallfiles,smallplan,33582000000)
            require(len(plan)==50000 and sum(r[2] for r in plan)==expected_bytes,'Plan totals')
            buffers=[cp.empty(max(r[2] for r in plan),cp.uint8) for _ in range(35)]
            handles=[kvikio.CuFile(str(p),'r') for p in files]
            def submit(slot,r):return handles[r[0]].pread(buffers[slot],size=r[2],file_offset=r[1],task_size=4<<20)
            try:
                run_queue(plan[:5],5,submit,lambda f:int(f.get()));cp.cuda.Device().synchronize()
                idle0=physical_counters();time.sleep(1);idle1=physical_counters()
                cache(files,True);initial=cache(files)
                require(all(r['resident_pages']==0 for r in initial),'Cache repopulated before read')
                before=physical_counters();start=time.perf_counter()
                total,count,last=run_queue(plan,35,submit,lambda f:int(f.get()))
                cp.cuda.Device().synchronize();seconds=time.perf_counter()-start;after=physical_counters()
                delta=counter_delta(before,after,total)
                require(count==50000 and total==expected_bytes,'Short reads')
                require(all(r['seconds']>=seconds for r in delta.values()),'NIC bracket')
                final=cache(files);witnesses=[]
                for slot,(i,offset,size) in sorted(last.items()):
                    fd=os.open(files[i],os.O_RDONLY)
                    try:
                        for at in (0,size//2-32,size-64):
                            require(buffers[slot][at:at+64].get().tobytes()==os.pread(fd,64,offset+at),'GPU witness')
                            witnesses.append(dict(slot=slot,file=i,offset=offset+at,size=64))
                    finally:os.close(fd)
                result=dict(status='pass',dataset=label,events=10000,requests=count,bytes=total,seconds=seconds,gib_s=total/(1<<30)/seconds,hz=10000/seconds,
                    depth=35,device_buffer_bytes=sum(b.nbytes for b in buffers),initial_cache=initial,final_cache=final,nic_before=before,nic_after=after,nic_delta=delta,
                    idle=dict(before=idle0,after=idle1,delta=counter_delta(idle0,idle1)),witnesses=witnesses)
                name='%02d-%s.json'%(index,label);write(output/name,result);cases.append(name)
                print(json.dumps(dict(case=name,gib_s=result['gib_s'],hz=result['hz'])),flush=True)
            finally:
                cp.cuda.Device().synchronize()
                for h in handles:h.close()
                del buffers
                cp.get_default_memory_pool().free_all_blocks()
        write(output/'complete.json',dict(status='pass',cases=cases))
    finally:
        shutil.rmtree(stage)
        write(output/'cleanup.json',dict(stage=str(stage),removed=not stage.exists()))

if __name__=='__main__':main(Path(sys.argv[1]))
