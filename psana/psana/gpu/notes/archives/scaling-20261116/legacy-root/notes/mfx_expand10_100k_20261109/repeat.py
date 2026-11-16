"""Project-scratch sample10 repetition writer; preserve payloads, rewrite outer times/SMD offsets."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import struct
import sys

S2=Path(__file__).resolve().parents[1]/'mfx_expand10_stage2_20261105'
sys.path.insert(0,str(S2))
import fixture
from xtc_fixture import HEADER, from_ns, to_ns


def timeline(rows, count, gap_ns=1000):
    fixture.require(type(count) is int and 1<=count<=100000,'Count must be 1..100000')
    fixture.require(type(gap_ns) is int and gap_ns>0,'Positive integer seam gap required')
    services=[r['service'] for r in rows]
    fixture.require([s for s in services if s not in (10,12)]==[2,4,6,8,9,7,5], 'Expected one native enabled step')
    fixture.require(all(to_ns(a['timestamp'])<to_ns(b['timestamp']) for a,b in zip(rows,rows[1:])), 'Unordered base timeline')
    enable,disable=services.index(8),services.index(9)
    fixture.require(services[:enable+1]==[2,4,6,8] and services[disable:]==[9,7,5], 'Unsupported transition placement')
    body=list(range(enable+1,disable)); events=[i for i in body if services[i]==12]
    fixture.require(events and all(services[i] in (10,12) for i in body),'Invalid enabled body')
    period=to_ns(rows[body[-1]]['timestamp'])-to_ns(rows[body[0]]['timestamp'])+gap_ns
    def entry(i,cycle,ts,event=None):
        return {'source_index':i,'cycle':cycle,'timestamp':from_ns(ts),'service':services[i],'output_event_index':event}
    plan=[entry(i,0,to_ns(rows[i]['timestamp'])) for i in range(enable+1)]
    completed=0;cycle=0
    while completed<count:
        for i in body:
            # A complete cycle retains trailing SlowUpdates; a partial stops at its last event.
            event=completed if services[i]==12 else None
            plan.append(entry(i,cycle,to_ns(rows[i]['timestamp'])+cycle*period,event))
            completed+=services[i]==12
            if completed==count and count%len(events):break
        cycle+=1
    last=to_ns(plan[-1]['timestamp'])
    plan += [entry(i,cycle-1,last+(j+1)*gap_ns) for j,i in enumerate(range(disable,len(rows)))]
    fixture.require(sum(r['service']==12 for r in plan)==count,'Wrong repeated event count')
    fixture.require(all(a['timestamp']<b['timestamp'] for a,b in zip(plan,plan[1:])), 'Nonmonotonic output')
    return plan,period


def index_source(root):
    manifest=json.loads((root/'generation.json').read_text())
    fixture.require(manifest['format']=='sample10-v1' and len(manifest['streams'])==10,'Invalid base format')
    result=[]
    for stream,record in enumerate(manifest['streams']):
        fixture.require(record['stream']==stream,'Stream order mismatch')
        paths={k:Path(v['path']) for k,v in record['files'].items()}
        fixture.require(set(paths)=={'bd','smd'},'Invalid file set')
        for k,path in paths.items():
            fixture.require(path.parent==(root if k=='bd' else root/'smalldata'),'Base escapes root')
            fixture.require(path.stat().st_size==record['files'][k]['bytes'] and fixture.file_sha(path)==record['files'][k]['sha256'],'Base digest mismatch')
        rows=[]; offset=0
        with paths['bd'].open('rb') as bd, paths['smd'].open('rb') as smd:
            smd_config=fixture.read_record(smd,0); sc=fixture.schemas(smd_config)
            pos=0
            while pos<paths['smd'].stat().st_size:
                raw=fixture.read_record(smd,pos); ts=fixture.header(raw)[0]; svc=fixture.service(raw)
                if svc==12:
                    (p,_),(q,_)=fixture.smd_offsets(raw,sc)
                    off,size=struct.unpack_from('<Q',raw,p)[0],struct.unpack_from('<Q',raw,q)[0]
                    fixture.require(off==offset,'Noncontiguous base BD offset')
                else:
                    bd.seek(offset); head=bd.read(24); fixture.require(len(head)==24,'Missing BD transition')
                    size=HEADER.unpack(head)[-1]+12
                rows.append({'timestamp':ts,'service':svc,'bd_offset':offset,'bd_size':size,'smd_offset':pos,'smd_size':len(raw)})
                offset+=size;pos+=len(raw)
        fixture.require(offset==paths['bd'].stat().st_size,'BD index does not reach EOF')
        result.append({'paths':paths,'rows':rows,'schemas':sc,'record':record})
    ref=[(r['timestamp'],r['service']) for r in result[0]['rows']]
    fixture.require(all([(r['timestamp'],r['service']) for r in s['rows']]==ref for s in result),'Unaligned base streams')
    fixture.require(sum(s==12 for _,s in ref)==manifest['event_count'],'Base count mismatch')
    timeline(result[0]['rows'],1)  # Validate one-step structural contract.
    return manifest,result


def rewrite(raw,item,*,smd=False,schemas=None,bd_offset=0,bd_size=0):
    out=bytearray(raw);struct.pack_into('<Q',out,0,item['timestamp'])
    if smd and item['service']==12:
        (p,_),(q,_)=fixture.smd_offsets(out,schemas)
        struct.pack_into('<Q',out,p,bd_offset);struct.pack_into('<Q',out,q,bd_size)
    fixture.header(out)
    return out


def make(source,root,count,gap_ns=1000):
    manifest,streams=index_source(source)
    plan,period=timeline(streams[0]['rows'],count,gap_ns)
    total=sum(s['rows'][r['source_index']]['bd_size']+s['rows'][r['source_index']]['smd_size'] for s in streams for r in plan)
    parent=root.parent;stat=os.statvfs(parent)
    fixture.require(stat.f_bavail*stat.f_frsize>total+2_000_000_000,'Insufficient filesystem headroom')
    root.mkdir(mode=0o770,exist_ok=False);(root/'smalldata').mkdir()
    report={'format':'sample10-v1','source_base':str(source),'source_generation_sha256':fixture.file_sha(source/'generation.json'),
            'event_count':count,'source_events':manifest['event_count'],'period_ns':period,'gap_ns':gap_ns,
            'identity_policy':'Source experiment/run in project scratch; repeated payloads, unique outer timestamps, rewritten SMD offsets; not a published run',
            'plan':plan,'streams':[],'total_file_bytes':total,'metadata':manifest['metadata'],
            'calibration_manifest_sha256':manifest['calibration_manifest_sha256'],'code_sha256':{str(Path(__file__)):fixture.file_sha(Path(__file__))}}
    # Do not expose a successful generation manifest until every file is complete.
    for number,s in enumerate(streams):
        paths={'bd':root/s['paths']['bd'].name,'smd':root/'smalldata'/s['paths']['smd'].name}
        hashes={k:hashlib.sha256() for k in paths}
        with s['paths']['bd'].open('rb') as ib, s['paths']['smd'].open('rb') as ism, paths['bd'].open('xb') as ob, paths['smd'].open('xb') as osm:
            for ordinal,item in enumerate(plan):
                row=s['rows'][item['source_index']]; offset=ob.tell()
                bd=fixture.read_record(ib,row['bd_offset']); smd=fixture.read_record(ism,row['smd_offset'])
                fixture.require(fixture.header(bd)[0]==row['timestamp'] and fixture.service(bd)==item['service'],'Base BD timeline mismatch')
                fixture.require(len(bd)==row['bd_size'] and len(smd)==row['smd_size'],'Base extent mismatch')
                outputs={'bd':rewrite(bd,item),'smd':rewrite(smd,item,smd=True,schemas=s['schemas'],bd_offset=offset,bd_size=len(bd))}
                for key,f in [('bd',ob),('smd',osm)]:f.write(outputs[key]);hashes[key].update(outputs[key])
                if ordinal and ordinal%10000==0:print('write stream',number,'records',ordinal,flush=True)
            for f in (ob,osm):f.flush();os.fsync(f.fileno())
        report['streams'].append({'stream':number,'panels':s['record']['panels'],'files':{k:{'path':str(p),'bytes':p.stat().st_size,'sha256':hashes[k].hexdigest()} for k,p in paths.items()}})
        print('Repeated stream',number,flush=True)
    fixture.require(sum(x['bytes'] for s in report['streams'] for x in s['files'].values())==total,'Unexpected total bytes')
    fixture.save_json(root/'generation.json',report)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--root',type=Path,required=True);p.add_argument('--events',type=int,required=True);p.add_argument('--gap-ns',type=int,default=1000)
    a=p.parse_args();make(a.source,a.root,a.events,a.gap_ns)
