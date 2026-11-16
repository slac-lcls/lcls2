"""Parallel independent SMD/header checks and CPU seed witnesses; no GPU reads."""
import os,sys,json,time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['PS_PARALLEL']='none'
from support import load_inputs,verify_payloads,sha,require,write,runtime
from adapter import register_reader
register_reader()
from psana import dgram
import xtc_fixture as x

def validate_one(g,r,expected,streams,witness_stamps):
    started=time.time();require(sha(r['smd'])==r['smd_sha256'],'SMD bytes changed')
    stream=int(Path(r['path']).name.split('-s')[1][:3]);panels=streams[stream]['panels'];witness={}
    with open(r['path'],'rb',buffering=0) as bd,open(r['smd'],'rb') as smd:
        os.posix_fadvise(bd.fileno(),0,0,os.POSIX_FADV_RANDOM)
        bcfg=x.read_record(bd,0);scfg=x.read_record(smd,0);schemas=x.schemas(bcfg);ss=x.schemas(scfg)
        jf=[s for s in schemas.values() if s['detname']=='jungfrau' and s['alg']=='sim10']
        require([s['segment'] for s in jf]==panels,'Panel membership/order')
        cfg=dgram.Dgram(view=bcfg);pos=offset=events=0;offsets=[]
        for timestamp,service in expected:
            raw=x.read_record(smd,pos);pos+=len(raw)
            require((x.header(raw)[0],x.service(raw))==(timestamp,service),'SMD timeline')
            head=os.pread(bd.fileno(),24,offset);require(len(head)==24,'Short header');h=x.HEADER.unpack(head);size=h[-1]+12
            require((h[0],(h[1]>>24)&15)==(timestamp,service) and size>=24,'BD timeline')
            if service==12:
                require([v for _,v in x.smd_offsets(raw,ss)]==[offset,size],'SMD offsets')
                require(size==len(panels)*104940+24,'Event size');events+=1;offsets.append(offset)
                if g['count']==2 and timestamp in witness_stamps:
                    data=os.pread(bd.fileno(),size,offset);require(len(data)==size,'Witness short read')
                    dg=dgram.Dgram(view=data,config=cfg);values={}
                    for panel in panels:
                        seed=dg.jungfrau[panel].sim10.seed;require(seed.shape==(52429,),'Seed shape')
                        values[str(panel)]=[int(seed[i]) for i in (0,26214,52428)]
                    witness[str(timestamp)]=values
            offset+=size
        require(events==100000 and offset==r['bytes'] and pos==Path(r['smd']).stat().st_size,'Full timeline/EOF')
    return {'count':g['count'],'stream':stream,'events':events,'panels':panels,'started':started,'finished':time.time(),'header_calls':len(expected)}, offsets, witness

def main(output):
    started=time.time();io_before=Path('/proc/self/io').read_text();info=load_inputs();verify_payloads(info);manifest=json.loads(Path(info['manifest']).read_text())
    expected=[(r['timestamp'],r['service']) for r in manifest['plan']];streams={s['stream']:s for s in manifest['streams'] if s['panels']}
    checks=[];ranges={};witness={str(t):{} for t in info['timestamps'][:1003]};witness_stamps=set(info['timestamps'][:1003])
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures={executor.submit(validate_one,g,r,expected,streams,witness_stamps):(g,r) for g in info['sets'] for r in g['files']}
        for future in as_completed(futures):
            g,r=futures[future];check,offsets,values=future.result();checks.append(check)
            if g['count']==2:
                ranges[Path(r['path']).name]={'offsets':offsets,'size':len(streams[check['stream']]['panels'])*104940+24}
                for ts,panels in values.items():
                    require(not set(witness[ts]).intersection(panels),'Duplicate witness panel');witness[ts].update(panels)
            write(output.with_name('validation-progress.json'),{'started':started,'completed':checks,'workers':8})
            print(json.dumps(check),flush=True)
    require(all(set(v)==set(map(str,range(32))) for v in witness.values()),'Witness panel coverage')
    write(output.with_name('witness.json'),witness);write(output.with_name('ranges.json'),ranges)
    write(output,{'state':'pass','checks':sorted(checks,key=lambda r:(r['count'],r['stream'])),'seconds':time.time()-started,'workers':8,'header_io':'unbuffered pread(24), POSIX_FADV_RANDOM; storage-wire bytes unmeasured','proc_io_before':io_before,'proc_io_after':Path('/proc/self/io').read_text(),'ranges_sha256':sha(output.with_name('ranges.json')),'witness_sha256':sha(output.with_name('witness.json')),'runtime':runtime(),'scope':'All100k SMD ranges and BD headers in all30files; sampled seed values for1003events,count2; historical full content hashes'})
if __name__=='__main__':main(Path(sys.argv[1]))
