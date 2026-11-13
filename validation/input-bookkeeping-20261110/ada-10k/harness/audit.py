"""Independent evidence checks; never accept rates alone."""
import json,math,sys
from pathlib import Path
from support import require as need,EVENT_BYTES,histogram,counter_delta,raw

def audit(root):
    read=lambda n:json.loads((root/n).read_text())
    p=read('provenance.json');info=read('inputs.json');done=read('complete.json');tests=read('gpu-tests.json')
    need(tests['exitcode']==0 and tests['passed']>0 and tests['failed']==tests['skipped']==0,'GPU tests')
    need(done['status']=='pass' and read('cleanup.json')['removed'],'Completion/cleanup')
    need(p['mount']['filesystems'][0]['source']=='drpsrcf/ffb','FFB mount')
    need(p['native_receipt']['status']=='pass' and p['native_receipt']['pr_sha']=='67c327a148a08b2aaf972e67e1e02e367c2b298f','PR identity')
    need(len(info['timestamps'])==10000 and info['timestamp_sha256']==raw.ts_hash(info['timestamps']),'Input identity')
    matrix=[('preflight-stream1',1,1000),('preflight-stream2',2,1000)]+[('%02d-stream%d'%(i,s),s,10000) for i,s in enumerate((1,2,2,1))]
    need([(c['name'],c['streams'],c['events']) for c in done['cases']]==matrix,'Matrix')
    result=[];host=device=runtime=None
    for name,streams,n in matrix:
        r=read(name+'.json');rows=r['rows'];bd=rows[2];io=bd['io'];m=r['measurement']
        need(r['status']=='pass' and r['events']==n and r['bytes']==EVENT_BYTES*n,'Case counts')
        need(r['framework']==dict(batch_size=100,streams=streams,budget_gib=6,bulk_target_bytes=524288,smd_events=1000),'Settings')
        need([x['role'] for x in rows]==['smd0','eb','bd'],'Roles')
        need(bd['timestamps']==info['timestamps'][:n] and all(not x['timestamps'] for x in rows[:2]),'Exact event identities')
        need(len(bd['callbacks'])==n//100 and all(c['size']==100 for c in bd['callbacks']),'Callbacks')
        need([t for c in bd['callbacks'] for t in c['timestamps']]==bd['timestamps'],'Callback identities')
        need(len({c['stream'] for c in bd['callbacks']})==streams,'Stream count')
        need(io['requests']==io['completed_requests']==5*n and io['requested_bytes']==io['completed_bytes']==EVENT_BYTES*n,'Read accounting')
        need(io['read_size_histogram']==histogram(n) and io['pressure_drains']==bd['cpu_reads']==0,'Read histogram/pressure')
        need(bd['drain']==[dict(closed=True,pending=0,active=0,held=0,peers=1)],'Drain')
        if host is None:host=r['host'];device=bd['device'];runtime=bd['runtime_files']
        need(r['host']==host and bd['device']==device,'Host/GPU unchanged')
        need(device['mask']==p['gpu_uuid'] and 'L40S' in device['name'],'Assigned GPU')
        need(device['compat_mode']=='ON' and device['effective']==dict(num_threads=8,task_size=1<<20,bounce_buffer_size=16<<20,auto_direct_io_read=False),'KvikIO')
        need(all(x['affinity']==p['cpu_affinity'] and len(x['affinity'])==24 and 'policy: interleave' in x['numa_policy'] and x['runtime_files']==runtime for x in rows),'CPU/runtime placement')
        need(r['source_sha256']==p['source_sha256'],'Harness identity')
        receipt=p['native_receipt'];source=Path(receipt['source'])/'psana/psana'
        for item in bd['runtime_files'].values():
            path=Path(item['path'])
            if path.suffix=='.py':
                need(path.is_relative_to(source),'Python path outside pinned PR')
                expected=receipt['python'].get(str(path.relative_to(source)))
            else:expected=receipt['native'].get(str(path))
            need(expected is not None and item['sha256']==expected,'Runtime receipt mismatch')
        for key in ('conditioning','initial_cache'):
            need([x['path'] for x in m[key]]==info['files'] and all(x['resident_pages']==0 for x in m[key]),'Cold cache')
        need(m['seconds']==r['seconds'] and r['seconds']>0 and math.isclose(r['gib_s'],r['bytes']/2**30/r['seconds']),'Rate')
        delta=counter_delta(m['nic_before'],m['nic_after'],r['bytes'])
        need(delta==m['nic_delta'] and all(v['seconds']>=r['seconds'] for v in delta.values()),'NIC bracket')
        need(counter_delta(m['idle']['before'],m['idle']['after'])==m['idle']['delta'],'Idle bracket')
        result.append(dict(name=name,streams=streams,events=n,seconds=r['seconds'],gib_s=r['gib_s'],hz=r['hz'],reader_calls=io['requests'],callbacks=len(bd['callbacks']),nic_delta=delta,idle=m['idle']['delta'],memory=bd['memory']))
    return dict(status='pass',host=host,device=device,cases=result)
if __name__=='__main__':print(json.dumps(audit(Path(sys.argv[1])),indent=2))
