"""Bounded 1 BD/one GPU/1..2 streams cold-FFB sweep over the accepted 100k sim10 fixture."""
import json,os,shutil,signal,subprocess,sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from support import HERE,ffb,require,sha,fixture,write,source_hashes,raw,EVENT_BYTES,ROOT,affinity

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
    stage=ffb.FFB/('sim10-ada-100k-'+os.environ['SLURM_JOB_ID']);stage.mkdir(mode=0o700)
    signal.signal(signal.SIGTERM,lambda *_:sys.exit(143))
    try:
        allocation_mask=os.environ['CUDA_VISIBLE_DEVICES'];require(bool(allocation_mask),'Scheduler GPU allocation required')
        # Exclusive Slurm nodes allocate all GPUs even when requesting one.
        # Select one device from that allocation, then expose only its UUID.
        mask=allocation_mask.split(',')[0]
        inventory=subprocess.check_output(['nvidia-smi','-i',mask,'--query-gpu=uuid,pci.bus_id,name','--format=csv,noheader'],text=True)
        require(len(inventory.splitlines())==1 and 'L40S' in inventory,'Expected one Ada L40S')
        cores=affinity()
        os.environ['JF_GPU_UUID']=inventory.splitlines()[0].split(',')[0].strip()
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
        write(output/'provenance.json',dict(staged=copied,mount=mount,allocation_gpu_mask=allocation_mask,gpu_inventory=inventory,gpu_uuid=os.environ['JF_GPU_UUID'],source_sha256=source_hashes(),cpu_affinity=cores,native_receipt=json.loads(Path(os.environ['SIM10_NATIVE_RECEIPT']).read_text())))
        cases=[]
        def execute(name,streams,events):
            bds=1
            cmd=['numactl','--interleave=all','mpiexec','-n',str(bds+2),'--bind-to','none',sys.executable,str(HERE/'runner.py'),'--inputs',str(output/'inputs.json'),'--output',str(output/(name+'.json')),'--bds',str(bds),'--events',str(events),'--streams',str(streams)]
            print('Starting',name,flush=True)
            with (output/(name+'.log')).open('x') as log:run_owned(cmd,log,900)
            r=json.loads((output/(name+'.json')).read_text());require(r['status']=='pass','Case failure')
            cases.append(dict(name=name,bds=bds,streams=streams,events=events,command=cmd,gib_s=r['gib_s'],hz=r['hz']))
            write(output/'progress.json',cases);print(json.dumps(cases[-1]),flush=True)
        for streams in (1,2):execute('preflight-stream%d'%streams,streams,1000)
        for i,streams in enumerate((1,2,2,1)):execute('%02d-stream%d'%(i,streams),streams,100000)
        write(output/'complete.json',dict(status='pass',cases=cases))
    finally:
        shutil.rmtree(stage);write(output/'cleanup.json',dict(stage=str(stage),removed=not stage.exists()))

if __name__=='__main__':main(Path(sys.argv[1]))
