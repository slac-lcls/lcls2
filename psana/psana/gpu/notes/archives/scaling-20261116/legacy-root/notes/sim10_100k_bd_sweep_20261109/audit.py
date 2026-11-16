"""Recompute topology, cold-cache, exact I/O and rate acceptance from native rows."""
import json,math,sys
from pathlib import Path
from support import require as need,raw,source_hashes,counter_delta,FIXTURE,EVENT_BYTES,BULK,BATCH,STREAMS,BUDGET,histogram

def audit(root):
    def read(name):return json.loads((root/name).read_text())
    info=read('inputs.json');p=read('provenance.json');complete=read('complete.json');cleanup=read('cleanup.json')
    need(complete['status']=='pass' and cleanup['removed'],'Incomplete sweep/cleanup')
    acceptance=json.loads((FIXTURE/'acceptance.json').read_text());v=json.loads((FIXTURE/'verification.json').read_text())
    need(info['generation_sha256']==acceptance['generation_sha256'] and raw.ts_hash(info['timestamps'])==info['timestamp_sha256']==v['timestamp_sha256'],'Fixture identity')
    need(len(info['timestamps'])==100000 and info['event_bytes']==EVENT_BYTES,'Fixture count/bytes')
    expected_sources=source_hashes();need(p['source_sha256']==expected_sources,'Source provenance')
    need(p['mount']['filesystems'][0]['source']=='drpsrcf/ffb','FFB mount')
    need(p['acceptance']==acceptance,'Acceptance receipt')
    need(len(p['staged'])==10 and [r['target'] for r in p['staged'] if r['kind']=='bd']==info['files'],'Staged file set')
    sizes={r['target']:r['bytes'] for r in p['staged']}
    frozen=json.loads((FIXTURE/'files.json').read_text());refs={r['path']:r for r in frozen['files']}
    for r in p['staged']:
        need(r['bytes']==refs[r['source']]['bytes'] and r['sha256']==refs[r['source']]['sha256'],'Copy receipt')
    matrix=[('preflight-bd%d'%b,b,1000*b) for b in (1,2,3,4)]+[('%02d-bd%d'%(i,b),b,100000) for i,b in enumerate((1,2,3,4,4,3,2,1))]
    need([(x['name'],x['bds'],x['events']) for x in complete['cases']]==matrix,'Case matrix')
    host=device=runtime=placement=None;results=[]
    for name,bds,n in matrix:
        r=read(name+'.json');total=EVENT_BYTES*n
        need(r['status']=='pass' and r['events']==n and r['bytes']==total and r['settings']['bds']==bds and r['settings']['events']==n,'Case identity')
        need(r['framework']==dict(batch_size=BATCH,streams=STREAMS,budget_gib=BUDGET,bulk_target_bytes=BULK,smd_events=1000),'Framework settings')
        need(r['seconds']>0 and math.isclose(r['gib_s'],total/2**30/r['seconds']) and math.isclose(r['hz'],n/r['seconds']),'Rates')
        need(r['source_sha256']==expected_sources,'Case code')
        if host is None:host=r['host']
        need(r['host']==host,'Host changed')
        rows=r['rows'];need([x['rank'] for x in rows]==list(range(bds+2)) and [x['role'] for x in rows]==['smd0','eb']+['bd']*bds,'Rank roles')
        need(all(not x['timestamps'] and x['device'] is None and not x['callbacks'] for x in rows[:2]),'Helper data')
        workers=rows[2:];events=[t for x in workers for t in x['timestamps']]
        need(sorted(events)==info['timestamps'][:n] and len(set(events))==n,'Global partition')
        for x in rows:
            need(x['host']==host and 'policy: interleave' in x['numa_policy'],'Placement')
            if placement is None:placement=x['affinity'];runtime=x['runtime_files']
            need(x['affinity']==placement and len(placement)>=40 and x['runtime_files']==runtime,'CPU/runtime changed')
        for x in workers:
            stamps=x['timestamps'];k=len(stamps);io=x['io']
            need(k>0 and stamps==sorted(set(stamps)),'Local partition')
            need([t for c in x['callbacks'] for t in c['timestamps']]==stamps,'Callback identity')
            need(all(0<c['size']<=BATCH and c['size']==len(c['timestamps']) for c in x['callbacks']),'Callback sizes')
            need(len({c['stream'] for c in x['callbacks']})==STREAMS,'Execution streams')
            need(x['drain']==[dict(closed=True,pending=0,active=0,held=0,peers=bds)],'Drain/peer discovery')
            if device is None:device=x['device']
            need(x['device']==device and device['mask']==p['gpu_uuid'],'GPU identity')
            need(device['compat_mode']=='ON' and device['effective']==dict(num_threads=8,task_size=1<<20,bounce_buffer_size=16<<20,auto_direct_io_read=False),'KvikIO settings')
            need(io['requests']==io['completed_requests']==5*k and io['requested_bytes']==io['completed_bytes']==EVENT_BYTES*k,'I/O count')
            need(io['read_size_histogram']==histogram(k),'I/O histogram')
            need(io['pressure_drains']==x['cpu_reads']==0,'Pressure/CPU reads')
        m=r['measurement'];need(math.isclose(m['seconds'],r['seconds']),'Timer identity')
        for key in ('conditioning','initial_cache','final_cache'):
            records=m[key];need([x['path'] for x in records]==info['files'],'Cache files')
            for x in records:
                need(x['bytes']==sizes[x['path']] and x['pages']==(x['bytes']+4095)//4096 and 0<=x['resident_pages']<=x['pages'],'Cache bounds')
                if key!='final_cache':need(x['resident_pages']==0,'Cold gate')
        delta=counter_delta(m['nic_before'],m['nic_after'],total)
        need(delta==m['nic_delta'] and all(x['seconds']>=r['seconds'] for x in delta.values()),'NIC bracket/delta')
        need(counter_delta(m['idle']['before'],m['idle']['after'])==m['idle']['delta'],'Idle counters')
        callbacks=[c for x in workers for c in x['callbacks']]
        results.append(dict(name=name,bds=bds,events=n,seconds=r['seconds'],gib_s=r['gib_s'],hz=r['hz'],reader_calls=sum(x['io']['requests'] for x in workers),callbacks=len(callbacks),
            subbatch_histogram={str(s):sum(c['size']==s for c in callbacks) for s in sorted({c['size'] for c in callbacks})},bd_events=[len(x['timestamps']) for x in workers],
            pending_requests_per_bd=[x['io']['peak_outstanding_requests'] for x in workers],pending_bytes_per_bd=[x['io']['peak_outstanding_bytes'] for x in workers],
            nic_rx_gib={k:v['rx_bytes']/2**30 for k,v in delta.items()},nic_envelope_excess_seconds={k:v['seconds']-r['seconds'] for k,v in delta.items()},
            final_resident_fraction=sum(x['resident_pages'] for x in m['final_cache'])/sum(x['pages'] for x in m['final_cache']),memory=[x['memory'] for x in workers]))
    return dict(status='pass',host=host,device=device,physical_cpus=placement,runtime_files=runtime,cases=results)
if __name__=='__main__':print(json.dumps(audit(Path(sys.argv[1])),indent=2))
