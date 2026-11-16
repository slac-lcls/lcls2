"""Independent full byte readback and serial DataSource acceptance for repetition."""
import hashlib,json,os,struct
from pathlib import Path
import numpy as np
from psana import DataSource,dgram
from repeat import fixture as f,index_source
from xtc_fixture import to_ns,from_ns


def timestamps_hash(times):return f.sha(struct.pack('<%dQ'%len(times),*times))


def reference_plan(rows,count,gap):
    """Build expected complete cycles plus a sliced partial tail, independently."""
    services=[r['service'] for r in rows];begin=services.index(8)+1;end=services.index(9)
    body=list(range(begin,end));n=sum(services[i]==12 for i in body)
    cycles,tail=divmod(count,n);period=to_ns(rows[body[-1]]['timestamp'])-to_ns(rows[body[0]]['timestamp'])+gap
    pairs=[(i,0) for i in range(begin)]+[(i,c) for c in range(cycles) for i in body]
    if tail:
        stop=[j for j,i in enumerate(body) if services[i]==12][tail-1]+1
        pairs += [(i,cycles) for i in body[:stop]]
    expected=[(i,c,from_ns(to_ns(rows[i]['timestamp'])+c*period),services[i]) for i,c in pairs]
    last=to_ns(expected[-1][2]);cycle=(count-1)//n
    expected += [(i,cycle,from_ns(last+(j+1)*gap),services[i]) for j,i in enumerate(range(end,len(rows)))]
    return expected,period


def normalized(raw,sc,smd):
    out=bytearray(raw)
    if smd and f.service(raw)==12:
        for p,_ in f.smd_offsets(raw,sc):struct.pack_into('<Q',out,p,0)
    return f.sha(memoryview(out)[8:])


def verify(source,root):
    base,streams=index_source(source);saved=json.loads((root/'generation.json').read_text());count=saved['event_count']
    f.require(saved['source_generation_sha256']==f.file_sha(source/'generation.json'),'Source receipt changed')
    f.require(saved['metadata']==base['metadata'] and saved['calibration_manifest_sha256']==base['calibration_manifest_sha256'],'Metadata/calibration changed')
    expected,period=reference_plan(streams[0]['rows'],count,saved['gap_ns'])
    f.require(period==saved['period_ns'],'Period mismatch')
    f.require([(x['source_index'],x['cycle'],x['timestamp'],x['service']) for x in saved['plan']]==expected,'Repeated timeline mismatch')
    times=[t for _,_,t,svc in expected if svc==12]
    f.require(len(times)==count and all(a[2]<b[2] for a,b in zip(expected,expected[1:])),'Timestamp identity/order')
    f.require([s for _,_,_,s in expected if s not in (10,12)]==[2,4,6,8,9,7,5],'Transition order')
    base_event_rows=[i for i,r in enumerate(streams[0]['rows']) if r['service']==12];n=len(base_event_rows)
    f.require([i for i,_,_,svc in expected if svc==12]==[base_event_rows[i%n] for i in range(count)],'Event mapping')
    witnesses=set()
    for start in range(0,count,n):
        witnesses.update(i for i in (start,start+1,min(start+n,count)-2,min(start+n,count)-1) if start<=i<min(start+n,count))
    base_witness_rows={base_event_rows[i%n] for i in witnesses};seed_hashes={};checks=[];file_bytes=0
    f.require([s['stream'] for s in saved['streams']]==list(range(10)),'Output stream set/order')
    for number,(source_stream,output) in enumerate(zip(streams,saved['streams'])):
        sc=source_stream['schemas'];ref={'bd':[],'smd':[]}
        for kind in ref:
            with source_stream['paths'][kind].open('rb') as handle:
                config=None
                for index,row in enumerate(source_stream['rows']):
                    raw=f.read_record(handle,row[kind+'_offset'])
                    f.require(len(raw)==row[kind+'_size'],'Base size')
                    ref[kind].append(normalized(raw,sc,kind=='smd'))
                    if kind=='bd' and row['service']==2:config=dgram.Dgram(view=raw)
                    if kind=='bd' and index in base_witness_rows:
                        decoded=dgram.Dgram(config=config,view=raw)
                        for panel,obj in getattr(decoded,'jungfrau',{}).items():seed_hashes[index,panel]=f.sha(obj.sim10.seed.tobytes())
        f.require(output['panels']==source_stream['record']['panels'],'Panel identities changed')
        paths={k:Path(v['path']) for k,v in output['files'].items()}
        f.require(set(paths)=={'bd','smd'},'Output file set')
        for kind,path in paths.items():
            f.require(path.resolve().parent==(root if kind=='bd' else root/'smalldata').resolve(),'Output path escapes fixture')
            f.require(path.name==source_stream['paths'][kind].name,'Output filename changed')
        cursors={'bd':0,'smd':0};hashes={k:hashlib.sha256() for k in cursors};state='start';events=0
        with paths['bd'].open('rb') as bd,paths['smd'].open('rb') as smd:
            for ordinal,(index,cycle,ts,svc) in enumerate(expected):
                raw={k:f.read_record(handle,cursors[k]) for k,handle in [('bd',bd),('smd',smd)]}
                for kind,data in raw.items():
                    f.require((f.header(data)[0],f.service(data))==(ts,svc),'BD/SMD timeline mismatch')
                    f.require(len(data)==source_stream['rows'][index][kind+'_size'],'Output size changed')
                    f.require(normalized(data,sc,kind=='smd')==ref[kind][index],'Unexpected payload/header mutation')
                    hashes[kind].update(data)
                if svc==12:
                    refs=f.smd_offsets(raw['smd'],sc)
                    f.require([v for _,v in refs]==[cursors['bd'],len(raw['bd'])],'Wrong SMD reference')
                    f.require(state=='enabled','Event outside enabled state');events+=1
                elif svc==10:f.require(state=='enabled','SlowUpdate outside enabled state')
                else:
                    states={2:('start','configured'),4:('configured','run'),6:('run','step'),8:('step','enabled'),9:('enabled','step'),7:('step','run'),5:('run','done')}
                    f.require(state==states[svc][0],'State machine');state=states[svc][1]
                for kind in cursors:cursors[kind]+=len(raw[kind])
                if ordinal and ordinal%20000==0:print('verify stream',number,'records',ordinal,flush=True)
            f.require(bd.read(1)==smd.read(1)==b'','Unaccounted output tail')
        f.require(state=='done' and events==count,'Stream completeness')
        for kind,path in paths.items():
            f.require(path.stat().st_size==cursors[kind]==output['files'][kind]['bytes'],'Output file size')
            f.require(hashes[kind].hexdigest()==output['files'][kind]['sha256'],'Output file SHA mismatch');file_bytes+=cursors[kind]
        checks.append(dict(stream=number,events=events,records=len(expected),files=output['files']))
        print('verified stream',number,flush=True)
    f.require(file_bytes==saved['total_file_bytes'],'Total footprint')
    f.require(os.environ.get('PS_PARALLEL')=='none' and os.environ.get('PS_FAKESTEP_FLAG')=='0','Serial real-step validation required')
    ds=DataSource(exp='mfx101555026',run=10,dir=str(root),live=False,skip_calib_load='all')
    runs=0;steps=[];observed=[];panel_checks=0;witness_checks=0
    for run in ds.runs():
        runs+=1;f.require(run.expt=='mfx101555026' and run.runnum==10,'Run identity')
        for step in run.steps():
            step_count=0
            for event in step.events():
                ordinal=len(observed);f.require(ordinal<count and int(event.timestamp)==times[ordinal],'DataSource timestamp')
                observed.append(int(event.timestamp));seen=set();f.require(len(event._dgrams)==10 and all(x is not None for x in event._dgrams),'Missing DataSource stream')
                for dg in event._dgrams:
                    f.require(int(dg.timestamp())==times[ordinal],'DataSource stream alignment')
                    for panel,obj in getattr(dg,'jungfrau',{}).items():
                        f.require(panel not in seen,'Duplicate panel');seen.add(panel);seed=obj.sim10.seed
                        f.require(seed.shape==(52429,) and seed.dtype==np.uint16,'Seed layout');panel_checks+=1
                        if ordinal in witnesses:
                            f.require(f.sha(seed.tobytes())==seed_hashes[base_event_rows[ordinal%n],panel],'DataSource seed witness mismatch');witness_checks+=1
                f.require(seen==set(range(32)),'DataSource panel membership');step_count+=1
                if step_count%10000==0:print('DataSource checked',step_count,flush=True)
            steps.append(step_count)
    f.require(runs==1 and steps==[count] and observed==times and panel_checks==32*count and witness_checks==32*len(witnesses),'DataSource traversal completeness')
    return dict(status='pass',event_count=count,source_event_count=n,steps=steps,runs=runs,streams=checks,total_file_bytes=file_bytes,
        timestamp_sha256=timestamps_hash(times),first_timestamp=times[0],last_timestamp=times[-1],records_per_stream=len(expected),
        all_datagram_payloads_verified=True,panel_layout_checks=panel_checks,seed_witness_events=sorted(witnesses),seed_witness_checks=witness_checks,
        generation_sha256=f.file_sha(root/'generation.json'),limitations='All payload bytes checked against accepted base; DataSource shape/panels/timestamps checked every event; seed values checked around every cycle seam/tail. No GPU performance or new scientific-calibration claim.')
