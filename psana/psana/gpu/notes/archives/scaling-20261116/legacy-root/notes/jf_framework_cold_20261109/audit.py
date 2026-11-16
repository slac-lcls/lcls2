"""Recompute acceptance for the complete psana-GPU cold framework matrix."""
import json,math,sys
from pathlib import Path
from support import require as need,raw,source_hashes,counter_delta

def audit(root):
    def read(name):return json.loads((root/name).read_text())
    inputs=read('inputs.json');provenance=read('provenance.json');completion=read('complete.json');cleanup=read('cleanup.json')
    need(completion['status']=='pass' and cleanup['removed'],'Incomplete campaign/cleanup')
    modes=['raw-dense','sim10-field','raw-field','sim10-field','raw-dense','raw-dense','sim10-field','raw-field']
    names=['preflight-raw-dense','preflight-sim10-field']+['%02d-%s'%(i,m) for i,m in enumerate(modes[2:])]
    need([r['name'] for r in completion['cases']]==names,'Case matrix')
    expected_sources=source_hashes();need(provenance['source_sha256']==expected_sources,'Source provenance changed')
    need(all(v['filesystems'][0]['source']=='drpsrcf/ffb' for v in provenance['mounts'].values()),'Storage')
    need(raw.ts_hash(inputs['raw']['timestamps'])==raw.REF['10000']['timestamp_sha256'],'Raw reference')
    results=[];host=None;device=None;runtime=None;placement=None
    for index,(name,mode) in enumerate(zip(names,modes)):
        r=read(name+'.json');n=200 if index<2 else 10000;dataset='sim10' if mode=='sim10-field' else 'raw';info=inputs[dataset]
        total=(3358200 if dataset=='sim10' else 33557176)*n
        need(r['status']=='pass' and r['events']==n and r['settings']['mode']==mode and r['bytes']==total,'Case identity/bytes')
        need(r['seconds']>0 and math.isclose(r['gib_s'],total/(1<<30)/r['seconds']) and math.isclose(r['hz'],n/r['seconds']),'Rates')
        need(r['source_sha256']==expected_sources,'Case code changed')
        if host is None:host=r['host'];runtime=r['runtime_files']
        need(r['host']==host and r['runtime_files']==runtime,'Host/runtime changed')
        rows=r['rows'];need([x['rank'] for x in rows]==[0,1,2],'Ranks')
        need(not rows[0]['timestamps'] and not rows[1]['timestamps'],'Helper delivery')
        bd=rows[2];need(bd['timestamps']==info['timestamps'][:n],'Exact timestamps')
        need([t for c in bd['callbacks'] for t in c['timestamps']]==bd['timestamps'],'Callback delivery')
        need(all(0<c['size']<=7 and c['size']==len(c['timestamps']) for c in bd['callbacks']),'Subbatch size')
        need(len({c['stream'] for c in bd['callbacks']})==2,'Two execution streams exercised')
        need(all('policy: interleave' in x['numa_policy'] for x in rows),'NUMA policy')
        cpus=[x['affinity'] for x in rows];need(list(map(len,cpus))==[1,1,4] and len({c for xs in cpus for c in xs})==6,'CPU placement')
        if placement is None:placement=cpus
        need(cpus==placement,'CPU placement changed')
        if device is None:device=bd['device']
        need(bd['device']==device,'GPU/settings changed')
        need(device['compat_mode']=='ON' and device['effective']==dict(num_threads=4,task_size=4<<20,bounce_buffer_size=16<<20,auto_direct_io_read=False),'KvikIO settings')
        io=bd['io'];need(io['requests']==io['completed_requests']==5*n and io['requested_bytes']==io['completed_bytes']==total,'I/O count')
        need(io['pressure_drains']==bd['cpu_reads']==0,'Pressure/CPU read')
        m=r['measurement'];need(math.isclose(m['seconds'],r['seconds']),'Timing identity')
        for key in ('initial_cache','final_cache'):
            cache=m[key];need([x['path'] for x in cache]==info['files'] and len(cache)==5,'Cache files')
            for x in cache:
                need(x['pages']==(x['bytes']+4095)//4096 and 0<=x['resident_pages']<=x['pages'],'Cache bounds')
                if key=='initial_cache':need(x['resident_pages']==0,'Cold gate')
        delta=counter_delta(m['nic_before'],m['nic_after'],total)
        need(delta==m['nic_delta'] and all(v['seconds']>=r['seconds'] for v in delta.values()),'NIC bracket/delta')
        need(counter_delta(m['idle']['before'],m['idle']['after'])==m['idle']['delta'],'Idle NIC')
        if index==0:
            expected=[dict(timestamp=x['timestamp'],raw=x['raw']) for x in json.loads((raw.HERE/'pixels.json').read_text())]
            need(sorted(bd['checks'],key=lambda x:x['timestamp'])==sorted(expected,key=lambda x:x['timestamp']),'Raw pixel hashes')
        else:need(not bd['checks'],'Unexpected validation D2H')
        final=m['final_cache'];results.append(dict(name=name,mode=mode,events=n,seconds=r['seconds'],gib_s=r['gib_s'],hz=r['hz'],callbacks=len(bd['callbacks']),
            final_resident_fraction=sum(x['resident_pages'] for x in final)/sum(x['pages'] for x in final),nic_rx_gib={k:v['rx_bytes']/(1<<30) for k,v in delta.items()}))
    return dict(status='pass',host=host,device=device,cases=results)
if __name__=='__main__':print(json.dumps(audit(Path(sys.argv[1])),indent=2))
