"""Compute-node acceptance of the exact transferred JF-only subset."""
import json,os,platform,sys,time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from support import HERE,NOTES,ROOT,TRANSFER,fixture,require,sha,ts_hash,write,EVENT_BYTES,mount

def main(output):
    require(os.environ.get('SLURM_JOB_ID') and os.environ.get('SLURM_PROCID')=='0','Compute step required')
    os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['PS_PARALLEL']='none'
    from psana import DataSource,dgram
    import numpy as np
    sys.path.insert(0,str(NOTES/'synthetic_mfx_stage3_20261104'))
    import xtc_fixture as x
    sys.path.insert(0,str(NOTES/'mfx_expand10_stage2_20261105'))
    import adapter
    adapter.register_reader()
    start=time.time();manifest,streams,stamps=fixture();mount_info=mount()
    records=json.loads(TRANSFER.read_text())['files']
    def verify(r):
        p=Path(r['destination_path']);actual=sha(p)
        require(actual==r['sha256'],'Transferred SHA '+str(p))
        print('SHA verified',p.name,flush=True)
        return dict(path=str(p),bytes=p.stat().st_size,sha256=actual)
    with ThreadPoolExecutor(max_workers=5) as pool:hashes=list(pool.map(verify,records))
    write(output.with_name('hashes.json'),dict(status='pass',files=hashes,job=os.environ['SLURM_JOB_ID']))
    expected=[(r['timestamp'],r['service']) for r in manifest['plan']];checks=[]
    for stream in streams:
        bdpath=Path(stream['files']['bd']['path']);smdpath=Path(stream['files']['smd']['path'])
        with bdpath.open('rb') as bd,smdpath.open('rb') as smd:
            bcfg=x.read_record(bd,0);scfg=x.read_record(smd,0);schemas=x.schemas(bcfg);ss=x.schemas(scfg)
            jf=[s for s in schemas.values() if s['detname']=='jungfrau' and s['alg']=='sim10']
            fields=[('seed',1,1),('frame_cnt',3,0),('timestamp',3,0),('hotPixelThresh',1,0),('numHotPixels',2,0),('maxHotPixels',2,0)]
            require([s['segment'] for s in jf]==stream['panels'],'Configure panel order')
            require(all(s['alg']=='sim10' and s['fields']==fields for s in jf),'Configure Names fields')
            cfg=dgram.Dgram(view=bcfg)
            meta=cfg.sim10meta[stream['stream']].config
            require(all(int(getattr(meta,k))==v for k,v in dict(factor=10,rows=512,columns=1024,dtype_code=1,encoding_version=1).items()),'Expansion metadata')
            pos=offset=events=payload=0
            for timestamp,service in expected:
                raw=x.read_record(smd,pos);pos+=len(raw)
                require((x.header(raw)[0],x.service(raw))==(timestamp,service),'SMD timeline')
                bd.seek(offset);head=bd.read(24);require(len(head)==24,'Short BD header')
                h=x.HEADER.unpack(head);size=h[-1]+12
                require((h[0],(h[1]>>24)&15)==(timestamp,service) and size>=24,'BD timeline/header')
                if service==12:
                    require([v for _,v in x.smd_offsets(raw,ss)]==[offset,size],'SMD offset/size')
                    require(size==len(stream['panels'])*104940+24,'Event dgram bytes')
                    events+=1;payload+=size
                offset+=size
            require(pos==smdpath.stat().st_size and offset==bdpath.stat().st_size and events==100000,'Stream completeness')
            checks.append(dict(stream=stream['stream'],events=events,records=len(expected),payload_bytes=payload,panels=stream['panels'],schemas=jf))
        print('SMD/config verified',stream['stream'],flush=True)
    require(sum(r['payload_bytes'] for r in checks)==EVENT_BYTES*100000,'Total payload')
    ds=DataSource(exp='mfx101555026',run=10,dir=str(ROOT),detectors=['jungfrau'],live=False,skip_calib_load='all')
    observed=[];steps=[];runs=0
    for run in ds.runs():
        runs+=1;require(run.expt=='mfx101555026' and run.runnum==10,'Run identity')
        for step in run.steps():
            count=0
            for event in step.events():
                index=len(observed);require(index<len(stamps) and int(event.timestamp)==stamps[index],'DataSource timestamps')
                require(len(event._dgrams)==5 and all(d is not None for d in event._dgrams),'DataSource subset')
                panels=[]
                for dg,stream in zip(event._dgrams,streams):
                    require(int(dg.timestamp())==stamps[index] and list(dg.jungfrau)==stream['panels'],'DataSource physical panel order')
                    for panel,obj in dg.jungfrau.items():
                        require(obj.sim10.seed.shape==(52429,) and obj.sim10.seed.dtype==np.uint16,'Seed layout');panels.append(panel)
                require(sorted(panels)==list(range(32)),'Panel coverage')
                observed.append(int(event.timestamp));count+=1
                if count%10000==0:print('DataSource checked',count,flush=True)
            steps.append(count)
        run.close_shared_memory()
    require(runs==1 and steps==[100000] and observed==stamps,'DataSource complete')
    write(output,dict(status='pass',job=os.environ['SLURM_JOB_ID'],host=platform.node(),started=start,finished=time.time(),files=hashes,mount=mount_info,
        event_count=len(observed),steps=steps,runs=runs,streams=checks,timestamp_sha256=ts_hash(observed),generation_sha256=sha(ROOT/'generation.json'),transfer_sha256=sha(TRANSFER),
        limitations='Relocation and JF-only subset accepted; unchanged original manifest; no calibration or expansion performed.'))
if __name__=='__main__':main(Path(sys.argv[1]))
