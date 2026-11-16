"""Independent post-run acceptance and compact throughput summary."""
import argparse,hashlib,json,math,statistics,struct
from pathlib import Path

def need(ok,why):
    if not ok:raise ValueError(why)

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def gate(m,kind):
    need(m['conditioning']['kind']==kind,'Conditioning kind')
    c=m['conditioning']['cache'];files=c['files']
    need(len(files)==5 and len({x['path'] for x in files})==5,'Five cache files required')
    need(sum(x['pages'] for x in files)==c['pages'] and sum(x['resident_pages'] for x in files)==c['resident_pages'],'Cache accounting')
    need(all(x['resident_pages']==0 if kind=='cold' else x['resident_pages']/x['pages']>=.999 for x in files),'Cache gate')
    need(m['nic_before'] and m['nic_before'].keys()==m['nic_after'].keys()==m['nic_delta'].keys(),'NIC inventory')
    for name,a in m['nic_before'].items():
        b=m['nic_after'][name];d=m['nic_delta'][name];dt=b['monotonic']-a['monotonic'];rx=b['rx_bytes_phy']-a['rx_bytes_phy'];tx=b['tx_bytes_phy']-a['tx_bytes_phy']
        need(a['device']==b['device'] and rx>=0 and tx>=0 and dt>=m['loop_seconds']>0,'NIC reset/bracket')
        need(d['seconds']==dt and d['rx_bytes']==rx and d['tx_bytes']==tx,'NIC arithmetic')
        need(math.isclose(d['rx_gib_s'],rx/(1<<30)/dt,rel_tol=1e-12),'NIC rate')

def check(r,refs):
    need(r['status']=='pass','Failed report');a=r['settings'];m=r['measurement'];gate(m,a['cache'])
    n=r['events'];need(r['bytes']==33557176*n and math.isclose(r['gib_s'],r['bytes']/(1<<30)/m['loop_seconds'],rel_tol=1e-12),'Rate/bytes')
    if r['workload']=='psana-dense-input':
        rows=r['rows'];bd=[x for x in rows if x['is_bd']];ts=[t for x in bd for t in x['timestamps']]
        need(len(bd)==a['bds'] and len(rows)==a['bds']+2,'Ranks')
        need(len(ts)==len(set(ts))==n and hashlib.sha256(struct.pack('<%dQ'%n,*sorted(ts))).hexdigest()==refs[str(n)]['timestamp_sha256'],'Event identities')
        need(all(not x['timestamps'] for x in rows if not x['is_bd']),'Helper events')
        need(len({x['device']['pci'] for x in bd})==len({x['device']['mask'] for x in bd})==1,'Device identity')
        need(all(x['affinity']==rows[0]['affinity'] for x in rows),'CPU placement')
        for x in bd:
            c=x['counts'];need(c['requests']==5*len(x['timestamps']) and c['bytes']==33557176*len(x['timestamps']),'Rank bytes')
            need(c['cpu_bd_reads']==c['pressure_drains']==0 and 0<c['peak_owned_and_held']<=c['budget_limit']<=6*(1<<30),'Budget/read/drain')
            need(x['timestamps']==sorted(x['timestamps']),'Local order')
        got=sorted([v for x in bd for v in x['checks']],key=lambda v:v['timestamp'])
        pixels=json.loads((Path(__file__).resolve().parents[1]/'pixels.json').read_text())
        want=sorted([dict(timestamp=x['timestamp'],raw=x['raw']) for x in pixels],key=lambda v:v['timestamp']) if a['preflight'] else []
        need(got==want,'Preflight pixels')
        devices=[x['device'] for x in bd]
    else:
        need(r['workload']=='standalone-kvikio' and r['requests']==50000 and n==10000 and len(r['witnesses'])==105 and r['depth']==35,'Standalone acceptance')
        devices=[r['device']]
    for d in devices:
        e=d['effective'];need(d['compat_mode']=='ON' and e['bounce_buffer_size']==16<<20 and not e['auto_direct_io_read'],'Fallback settings')
        need(e['num_threads']==(8 if r['workload']=='psana-dense-input' else a['threads']),'Workers')
        need(e['task_size']==(1<<20 if r['workload']=='psana-dense-input' else a['task_mib']<<20),'Task size')
    return devices[0]

def unavailable(r,cache):
    need(r['status']=='cache-unavailable' and r['settings']['cache']=='warm','Unavailable classification')
    need('gib_s' not in r and 'measurement' not in r,'Failed warm gate has a rate')
    need(cache['status']=='fail' and cache['kind']=='warm','Failed cache evidence')
    need(len(cache['history'])==10 and cache['history'][-1]['stage']=='repair-8','Bounded repairs missing')
    files=cache['cache']['files']
    need(len(files)==5 and any(x['resident_pages']/x['pages']<.999 for x in files),'Warm cache actually passed')


def audit(root):
    here=Path(__file__).resolve().parents[1];refs=json.loads((here/'reference.json').read_text());complete=json.loads((root/'complete.json').read_text())
    need(complete['status'] in ('pass','partial') and len(complete['cases'])==20,'Campaign incomplete')
    expected={'preflight-b1','preflight-b4'}|{'psana-%s-b%d-%s-r%d'%(storage,bds,cache,rep) for storage,bds in [('weka',1),('weka',4),('local',4)] for cache in ('cold','warm') for rep in (1,2)}|{'read-%s-t%d-%s'%(storage,threads,cache) for storage,threads in [('weka',4),('weka',8),('local',8)] for cache in ('cold','warm')}
    need({x['name'] for x in complete['cases']}==expected,'Unexpected cases')
    artifacts={};rows=[];groups={};source=None;device=None;host=None;missing=[]
    for c in complete['cases']:
        p=root/(c['name']+'.json');r=json.loads(p.read_text())
        if r['status']=='cache-unavailable':
            q=root/(c['name']+'.cache.json');cache=json.loads(q.read_text());unavailable(r,cache)
            need(c['status']=='cache-unavailable' and c['gib_s'] is None and sha(q)==r['cache_sha256'],'Unavailable record mismatch')
            need(r['host']==host and r['source_sha256']==source,'Unavailable source/host mismatch')
            artifacts[str(p)]=sha(p);artifacts[str(q)]=sha(q)
            missing.append(dict(name=c['name'],cache=cache['cache'],conditioning_seconds=cache['seconds']))
            continue
        d=check(r,refs)
        if source is None:source=r['source_sha256'];device=d['mask'];host=r['host']
        need(r['source_sha256']==source and d['mask']==device and r['host']==host,'Campaign identity')
        need(c['gib_s']==r['gib_s'],'Progress mismatch');artifacts[str(p)]=sha(p)
        row=dict(name=c['name'],gib_s=r['gib_s'],seconds=r['measurement']['loop_seconds'],nic=r['measurement']['nic_delta'],cache_before=r['measurement']['conditioning']['cache']['resident_pages'],cache_pages=r['measurement']['conditioning']['cache']['pages'],cache_after=r['measurement']['cache_after'],conditioning_seconds=r['measurement']['conditioning']['seconds'],bytes=r['bytes'])
        if 'rows' in r:row['per_bd']=[{k:x[k] for k in ('rank','counts','loop_seconds')} for x in r['rows'] if x['is_bd']]
        rows.append(row)
        if not c['name'].startswith('preflight'):groups.setdefault(c['name'].rsplit('-r',1)[0],[]).append(r['gib_s'])
    for p,h in source.items():need(sha(p)==h,'Source changed')
    stage=json.loads((root/'staging.json').read_text());expected_hash=json.loads((here/'expected-sha256.json').read_text())
    need(len(stage)==5 and all(x['sha256']==expected_hash[Path(x['source']).name] for x in stage),'Stage identity')
    need(json.loads((root/'cleanup.json').read_text())['removed'],'Local cleanup')
    need(json.loads((root/'root-cleanup.json').read_text())['removed'],'Root cleanup')
    need(complete['status']==('partial' if missing else 'pass'),'Completion classification')
    return dict(status='partial' if missing else 'pass',unavailable_warm_cases=missing,host=host,device_mask=device,artifacts=artifacts,source_verified=source,cases=rows,summary={k:dict(median=statistics.median(v),minimum=min(v),maximum=max(v),repeats=len(v)) for k,v in groups.items()})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    with a.output.open('x') as f:json.dump(audit(a.root),f,indent=2);f.write('\n')
