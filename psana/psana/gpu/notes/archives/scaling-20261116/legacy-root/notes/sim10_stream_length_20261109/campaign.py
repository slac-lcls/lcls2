"""Within-node Lustre local-cold pilots and reversed stream/duration matrix."""
import json,os,signal,subprocess,sys,time
from pathlib import Path
from support import HERE,require,sha,fixture,write,source_hashes,ts_hash,EVENT_BYTES,FIXTURE,ROOT,MATRIX,mount,MODE

def launch(tasks,program,*args):
    return ['srun','--mpi=cray_shasta','--nodes=1','--ntasks='+str(tasks),'--ntasks-per-node='+str(tasks),'--cpus-per-task=16','--gpus-per-node=1','--gpu-bind=none','--cpu-bind=none','--kill-on-bad-exit=1','numactl','--interleave=all',sys.executable,str(HERE/program),*map(str,args)]
def main(output):
    output.mkdir(parents=True,exist_ok=False);started=time.time()
    signal.signal(signal.SIGTERM,lambda *_:sys.exit(143))
    try:
        inventory=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,pci.bus_id,name,memory.total','--format=csv,noheader'],text=True)
        allocated=os.environ.get('CUDA_VISIBLE_DEVICES','').split(',')
        require(allocated and all(x not in ('','-1') for x in allocated),'Slurm allocated GPUs required')
        os.environ['JF_GPU_UUID']=subprocess.check_output(['nvidia-smi','-i',allocated[0],'--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
        require(os.environ['JF_GPU_UUID'].startswith('GPU-') and '\n' not in os.environ['JF_GPU_UUID'],'Allocated GPU UUID')
        cmd=launch(1,'validate_input.py',output/'validation.json')
        write(output/'validation-command.json',cmd)
        with (output/'validation.log').open('x') as log:subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=2400)
        validation=json.loads((output/'validation.json').read_text());require(validation['status']=='pass','Input validation')
        manifest,streams,stamps=fixture()
        files=[dict(stream=s['stream'],kind=k,source=r['source_path'],target=r['path'],bytes=r['bytes'],sha256=r['sha256']) for s in streams for k,r in s['files'].items()]
        info=dict(exp='mfx101555026',run=10,directory=str(ROOT),files=[r['target'] for r in files if r['kind']=='bd'],timestamps=stamps,event_bytes=EVENT_BYTES,generation_sha256=sha(ROOT/'generation.json'),timestamp_sha256=ts_hash(stamps))
        write(output/'inputs.json',info)
        write(output/'provenance.json',dict(staged=files,staging='transferred fixture reused in place',mode=MODE,mount=mount(),allocated_mask=','.join(allocated),gpu_inventory=inventory,gpu_uuid=os.environ['JF_GPU_UUID'],source_sha256=source_hashes(),acceptance=json.loads((FIXTURE/'acceptance.json').read_text()),validation_sha256=sha(output/'validation.json'),job=os.environ['SLURM_JOB_ID'],started=started))
        cases=[]
        for name,streams,events in MATRIX:
            cmd=launch(3,'runner.py','--inputs',output/'inputs.json','--output',output/(name+'.json'),'--bds',1,'--events',events,'--streams',streams)
            print('Starting',name,flush=True)
            with (output/(name+'.log')).open('x') as log:subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=900)
            r=json.loads((output/(name+'.json')).read_text());require(r['status']=='pass','Case failure')
            cases.append(dict(name=name,bds=1,streams=streams,events=events,command=cmd,gib_s=r['gib_s'],hz=r['hz']))
            write(output/'progress.json',cases);print(json.dumps(cases[-1]),flush=True)
        write(output/'complete.json',dict(status='pass',cases=cases,job=os.environ['SLURM_JOB_ID'],started=started,finished=time.time()))
    finally:
        write(output/'cleanup.json',dict(processes_exited=True,input_preserved=ROOT.exists(),temporary_input_created=False))
    from audit import audit
    write(output/'audit.json',audit(output))
    with (output/'test-audit.log').open('x') as log:subprocess.run([sys.executable,str(HERE/'test_audit.py'),str(output)],stdout=log,stderr=subprocess.STDOUT,check=True)
if __name__=='__main__':main(Path(sys.argv[1]))
