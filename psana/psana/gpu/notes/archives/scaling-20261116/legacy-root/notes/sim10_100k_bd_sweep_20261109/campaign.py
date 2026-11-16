"""Bounded 1..4 BD/one-GPU cold-FFB sweep over the accepted 100k sim10 fixture."""
import json,os,shutil,signal,subprocess,sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from support import HERE,ffb,require,sha,fixture,write,source_hashes,raw,EVENT_BYTES,FIXTURE,ROOT

def main(output):
    output.mkdir(parents=True,exist_ok=False)
    stage=ffb.FFB/('sim10-100k-bds-'+os.environ['SLURM_JOB_ID']);stage.mkdir(mode=0o700)
    signal.signal(signal.SIGTERM,lambda *_:sys.exit(143))
    try:
        inventory=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,pci.bus_id,name','--format=csv,noheader'],text=True)
        os.environ['JF_GPU_UUID']=inventory.splitlines()[0].split(',')[0].strip()
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
        write(output/'provenance.json',dict(staged=copied,mount=mount,gpu_inventory=inventory,gpu_uuid=os.environ['JF_GPU_UUID'],source_sha256=source_hashes(),acceptance=json.loads((FIXTURE/'acceptance.json').read_text())))
        cases=[]
        def execute(name,bds,events):
            cmd=['numactl','--interleave=all','mpiexec','-n',str(bds+2),'--bind-to','none',sys.executable,str(HERE/'runner.py'),'--inputs',str(output/'inputs.json'),'--output',str(output/(name+'.json')),'--bds',str(bds),'--events',str(events)]
            print('Starting',name,flush=True)
            with (output/(name+'.log')).open('x') as log:subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=900)
            r=json.loads((output/(name+'.json')).read_text());require(r['status']=='pass','Case failure')
            cases.append(dict(name=name,bds=bds,events=events,command=cmd,gib_s=r['gib_s'],hz=r['hz']))
            write(output/'progress.json',cases);print(json.dumps(cases[-1]),flush=True)
        for bds in (1,2,3,4):execute('preflight-bd%d'%bds,bds,1000*bds)
        for i,bds in enumerate((1,2,3,4,4,3,2,1)):execute('%02d-bd%d'%(i,bds),bds,100000)
        write(output/'complete.json',dict(status='pass',cases=cases))
    finally:
        shutil.rmtree(stage);write(output/'cleanup.json',dict(stage=str(stage),removed=not stage.exists()))

if __name__=='__main__':main(Path(sys.argv[1]))
