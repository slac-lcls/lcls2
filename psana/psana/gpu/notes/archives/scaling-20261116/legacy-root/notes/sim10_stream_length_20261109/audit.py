"""Recompute topology, cold-cache, exact I/O and rate acceptance from native rows."""
import json,math,sys
from pathlib import Path
from support import require as need,ts_hash,sha,MODE,HERE,TRANSFER,source_hashes,FIXTURE,ROOT,EVENT_BYTES,BULK,BATCH,BUDGET,MATRIX,histogram

def audit(root):
    def read(name):return json.loads((root/name).read_text())
    info=read('inputs.json');p=read('provenance.json');complete=read('complete.json');cleanup=read('cleanup.json')
    need(complete['status']=='pass' and cleanup==dict(processes_exited=True,input_preserved=True,temporary_input_created=False),'Incomplete sweep/cleanup')
    acceptance=json.loads((FIXTURE/'acceptance.json').read_text());v=json.loads((FIXTURE/'verification.json').read_text())
    need(info['generation_sha256']==acceptance['generation_sha256'] and ts_hash(info['timestamps'])==info['timestamp_sha256']==v['timestamp_sha256'],'Fixture identity')
    need(len(info['timestamps'])==100000 and info['event_bytes']==EVENT_BYTES,'Fixture count/bytes')
    expected_sources=source_hashes();need(p['source_sha256']==expected_sources,'Source provenance')
    need(p['mount']['filesystems'][0]['fstype']=='lustre' and p['mode']==MODE,'Lustre measurement mode')
    need(p['acceptance']==acceptance,'Acceptance receipt')
    need(len(p['staged'])==10 and [r['target'] for r in p['staged'] if r['kind']=='bd']==info['files'],'Staged file set')
    sizes={r['target']:r['bytes'] for r in p['staged']}
    frozen=json.loads((FIXTURE/'files.json').read_text());refs={r['path']:r for r in frozen['files']}
    for r in p['staged']:
        need(r['bytes']==refs[r['source']]['bytes'] and r['sha256']==refs[r['source']]['sha256'],'Copy receipt')
    validation=read('validation.json')
    need(sha(root/'validation.json')==p['validation_sha256'] and validation['status']=='pass' and validation['event_count']==100000 and validation['steps']==[100000] and validation['runs']==1,'Native input acceptance')
    need(validation['timestamp_sha256']==info['timestamp_sha256'] and validation['generation_sha256']==info['generation_sha256'] and validation['transfer_sha256']==sha(TRANSFER),'Native input identity')
    transfer=json.loads(TRANSFER.read_text())
    need(info['directory']==str(ROOT) and info['exp']=='mfx101555026' and info['run']==10,'Input location')
    need(p['staged']==[dict(stream=t['stream'],kind=t['kind'],source=t['source_path'],target=t['destination_path'],bytes=t['bytes'],sha256=t['sha256']) for t in transfer['files'] if t['kind']!='manifest'],'Relocated file mapping')
    need(validation['files']==[dict(path=t['destination_path'],bytes=t['bytes'],sha256=t['sha256']) for t in transfer['files']],'Native file hashes')
    need([t['stream'] for t in validation['streams']]==[5,6,7,8,9] and all(t['events']==100000 for t in validation['streams']),'Native stream counts')
    matrix=MATRIX
    need([(x['name'],x['streams'],x['events']) for x in complete['cases']]==matrix and all(x['bds']==1 for x in complete['cases']),'Case matrix')
    host=device=runtime=placement=None;results=[]
    for name,streams,n in matrix:
        bds=1
        r=read(name+'.json');total=EVENT_BYTES*n
        need(r['status']=='pass' and r['events']==n and r['bytes']==total and r['settings']['bds']==bds and r['settings']['events']==n and r['settings']['streams']==streams,'Case identity')
        need(r['framework']==dict(batch_size=BATCH,streams=streams,budget_gib=BUDGET,bulk_target_bytes=BULK,smd_events=1000),'Framework settings')
        need(r['seconds']>0 and math.isclose(r['gib_s'],total/2**30/r['seconds']) and math.isclose(r['hz'],n/r['seconds']),'Rates')
        need(r['source_sha256']==expected_sources,'Case code')
        if host is None:host=r['host']
        need(r['host']==host,'Host changed')
        need(r['job']==p['job']==complete['job']==validation['job'],'Job identity')
        rows=r['rows'];need([x['rank'] for x in rows]==list(range(bds+2)) and [x['role'] for x in rows]==['smd0','eb']+['bd']*bds,'Rank roles')
        need(all(not x['timestamps'] and x['device'] is None and not x['callbacks'] for x in rows[:2]),'Helper data')
        workers=rows[2:];events=[t for x in workers for t in x['timestamps']]
        need(sorted(events)==info['timestamps'][:n] and len(set(events))==n,'Global partition')
        for x in rows:
            need(x['host']==host and 'policy: interleave' in x['numa_policy'],'Placement')
            need(x['cleanup_complete'] and 'CRAY MPICH' in x['mpi'].upper(),'Cleanup/MPI')
            need(p['gpu_uuid'] in x['inherited_uuids'] and len(x['inherited_uuids'])==1,'Step GPU allocation')
            need(x['mask']==(p['gpu_uuid'] if x['rank']==2 else ''),'Rank GPU masks')
            need(x['runtime_files']==json.loads((HERE/'perlmutter-runtime.json').read_text())['modules'],'Frozen runtime')
            if placement is None:placement=x['affinity'];runtime=x['runtime_files']
            need(x['affinity']==placement and len(placement)>=12 and x['runtime_files']==runtime,'CPU/runtime changed')
        for x in workers:
            stamps=x['timestamps'];k=len(stamps);io=x['io']
            need(len(x['admission'])==1,'Admission count')
            ad=x['admission'][0]
            need(ad['budget']==BUDGET*2**30 and ad['margin']==ad['budget']//10 and 0<ad['capacity']<=ad['budget']-ad['margin'] and ad['subbatch_budget']==ad['capacity']//streams,'Admission budget')
            need(x['memory']['total']>BUDGET*2**30 and 0<=x['memory']['pool_used']<=x['memory']['pool_total'],'GPU memory')
            need(k>0 and stamps==sorted(set(stamps)),'Local partition')
            need([t for c in x['callbacks'] for t in c['timestamps']]==stamps,'Callback identity')
            need(len(x['callbacks'])==k//BATCH and all(c['size']==BATCH and len(c['timestamps'])==BATCH for c in x['callbacks']),'Callback sizes/histogram')
            need(len({c['stream'] for c in x['callbacks']})==streams,'Execution streams')
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
                need(x['bytes']==sizes[x['path']] and x['page_size']==4096 and x['pages']==(x['bytes']+x['page_size']-1)//x['page_size'] and 0<=x['resident_pages']<=x['pages'],'Cache bounds')
                if key!='final_cache':need(x['resident_pages']==0,'Cold gate')
        need(m['mode']==MODE and m['diagnostic_before']['mode']==m['diagnostic_after']['mode']==MODE,'Measurement mode')
        need(m['diagnostic_after']['monotonic']-m['diagnostic_before']['monotonic']>=r['seconds'] and m['wall_end']>=m['wall_start']>=p['started'],'Diagnostic bracket')
        callbacks=[c for x in workers for c in x['callbacks']]
        results.append(dict(name=name,bds=bds,streams=streams,events=n,seconds=r['seconds'],gib_s=r['gib_s'],hz=r['hz'],reader_calls=sum(x['io']['requests'] for x in workers),callbacks=len(callbacks),
            subbatch_histogram={str(s):sum(c['size']==s for c in callbacks) for s in sorted({c['size'] for c in callbacks})},bd_events=[len(x['timestamps']) for x in workers],
            pending_requests_per_bd=[x['io']['peak_outstanding_requests'] for x in workers],pending_bytes_per_bd=[x['io']['peak_outstanding_bytes'] for x in workers],
            remote_traffic_verified=False,wall_start=m['wall_start'],wall_end=m['wall_end'],admission=[x['admission'] for x in workers],
            final_resident_fraction=sum(x['resident_pages'] for x in m['final_cache'])/sum(x['pages'] for x in m['final_cache']),memory=[x['memory'] for x in workers]))
    return dict(status='pass',mode=MODE,job=p['job'],host=host,device=device,physical_cpus=placement,runtime_files=runtime,cases=results)
if __name__=='__main__':print(json.dumps(audit(Path(sys.argv[1])),indent=2))
