"""Bounded KvikIO bounce-buffer screen with one GPU and one CUDA stream per BD."""
import json,os,shutil,signal,subprocess,sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from support import HERE,ffb,require,sha,fixture,write,source_hashes,raw,EVENT_BYTES,ROOT
from cpu_plan import discover,plan,case_matrix,slurm_request,CONFIGURATIONS,plan_key
from hardware import inventory as hardware_inventory
from task_evidence import pilot_evidence

def run_owned(cmd,log,timeout):
    # Stop and reap the MPI process group before removing its staged inputs.
    child=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    try:
        rc=child.wait(timeout=timeout)
        if rc:raise subprocess.CalledProcessError(rc,cmd)
    except BaseException:
        if child.poll() is None:
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid,signal.SIGKILL);child.wait()
        raise

def main(output):
    output.mkdir(parents=True,exist_ok=False)
    stage=ffb.FFB/('sim10-kvikio-bounce-'+os.environ['SLURM_JOB_ID']);stage.mkdir(mode=0o700)
    signal.signal(signal.SIGTERM,lambda *_:sys.exit(143))
    try:
        allocation_mask=os.environ['CUDA_VISIBLE_DEVICES'];require(bool(allocation_mask),'Scheduler GPU allocation required')
        # Exclusive Slurm nodes allocate all GPUs even when requesting one.
        # Select one device from that allocation, then expose only its UUID.
        mask=allocation_mask.split(',')[0]
        inventory=subprocess.check_output(['nvidia-smi','-i',mask,'--query-gpu=uuid,pci.bus_id,name','--format=csv,noheader'],text=True)
        require(len(inventory.splitlines())==1 and 'A100-SXM4-40GB' in inventory,'Expected one A100')
        resources=slurm_request()
        require(int(os.environ['SLURM_NTASKS'])==resources['ntasks'] and int(os.environ['SLURM_CPUS_PER_TASK'])==resources['cpus_per_task'],'Scheduler task/CPU request mismatch')
        topology=discover()
        plans={plan_key(b,w):plan(topology,b,w) for b,w,_,_ in CONFIGURATIONS}
        require(len({(x['socket'],x['core']) for x in topology})>=resources['requested_physical_cores'],'Allocated physical CPU request not met')
        os.environ['JF_GPU_UUID']=inventory.splitlines()[0].split(',')[0].strip()
        hardware=hardware_inventory(os.environ['JF_GPU_UUID'])
        hardware['strace_version']=subprocess.check_output(['strace','--version'],text=True).splitlines()[0]
        write(output/'hardware.json',hardware)
        with (output/'gpu-tests.log').open('x') as log:
            run_owned([sys.executable,str(HERE/'gpu_tests.py'),str(output/'gpu-tests.json'),str(output/'pytest-tmp')],log,300)
        manifest,streams,stamps=fixture()
        require(shutil.disk_usage(stage).free>sum(v['bytes'] for s in streams for v in s['files'].values())+(5<<30),'FFB capacity')
        mount=json.loads(subprocess.check_output(['findmnt','-J','-T',str(stage)],text=True))
        require(mount['filesystems'][0]['source']=='drpsrcf/ffb','Require Weka FFB')
        (stage/'smalldata').mkdir()
        def copy(item):
            s,kind,r=item;source=Path(r['path']);target=(stage if kind=='bd' else stage/'smalldata')/source.name
            require(not target.exists(),'Existing stage target');shutil.copyfile(source,target)
            require(target.stat().st_size==r['bytes'] and sha(target)==r['sha256'],'Staged SHA/size mismatch')
            print('Verified staged',target.name,flush=True)
            return dict(stream=s,kind=kind,source=str(source),target=str(target),bytes=r['bytes'],sha256=r['sha256'])
        items=[(s['stream'],k,v) for s in streams for k,v in s['files'].items()]
        with ThreadPoolExecutor(max_workers=4) as pool:copied=list(pool.map(copy,items))
        info=dict(exp='mfx101555026',run=10,directory=str(stage),files=[x['target'] for x in copied if x['kind']=='bd'],timestamps=stamps,event_bytes=EVENT_BYTES,
            generation_sha256=sha(ROOT/'generation.json'),timestamp_sha256=raw.ts_hash(stamps))
        write(output/'inputs.json',info)
        write(output/'provenance.json',dict(staged=copied,mount=mount,allocation_gpu_mask=allocation_mask,gpu_inventory=inventory,gpu_uuid=os.environ['JF_GPU_UUID'],source_sha256=source_hashes(),cpu_topology=topology,cpu_plans=plans,slurm_request=slurm_request(),slurm_environment={k:v for k,v in os.environ.items() if k.startswith('SLURM_') and any(t in k for t in ('CPU','TASK','NODELIST','JOB_ID'))},hardware_sha256=sha(output/'hardware.json'),native_receipt=json.loads(Path(os.environ['SIM10_NATIVE_RECEIPT']).read_text())))
        cases=[]
        def execute(name,bds,workers,streams,batch,bounce_mib,events):
            os.environ['SIM10_CPU_PLAN']=json.dumps(plans[plan_key(bds,workers)])
            os.environ['KVIKIO_NTHREADS']=str(workers)
            os.environ['KVIKIO_TASK_SIZE']=str(1<<20)
            os.environ['KVIKIO_BOUNCE_BUFFER_SIZE']=str(bounce_mib<<20)
            cmd=['numactl','--interleave=all','mpiexec','-n',str(bds+2),'--bind-to','none',sys.executable,*([str(HERE/'trace_runner.py'),str(HERE/'runner.py')] if events<100000 else [str(HERE/'runner.py')]),'--inputs',str(output/'inputs.json'),'--output',str(output/(name+'.json')),'--bds',str(bds),'--workers',str(workers),'--events',str(events),'--streams',str(streams),'--batch',str(batch),'--bounce-mib',str(bounce_mib)]
            print('Starting',name,flush=True)
            with (output/(name+'.log')).open('x') as log:run_owned(cmd,log,900)
            r=json.loads((output/(name+'.json')).read_text());require(r['status']=='pass','Case failure')
            if events<100000:
                from support import histogram
                write(output/(name+'-io-evidence.json'),pilot_evidence(output/(name+'.json'),r,events,histogram(events)))
            cases.append(dict(name=name,bds=bds,workers=workers,streams=streams,batch=batch,bounce_mib=bounce_mib,events=events,command=cmd,gib_s=r['gib_s'],hz=r['hz']))
            write(output/'progress.json',cases);print(json.dumps(cases[-1]),flush=True)
        for case in case_matrix():execute(*case)
        write(output/'complete.json',dict(status='pass',cases=cases))
    finally:
        shutil.rmtree(stage);write(output/'cleanup.json',dict(stage=str(stage),removed=not stage.exists()))

if __name__=='__main__':
    output=Path(sys.argv[1]);main(output)
    with (output/'audit.json').open('x') as receipt:
        subprocess.run([sys.executable,str(HERE/'audit.py'),str(output)],stdout=receipt,check=True)
