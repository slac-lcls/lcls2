"""Recompute acceptance for the complete psana-GPU cold framework matrix."""
import json,math,sys
from pathlib import Path
from support import HERE,require as need,raw,source_hashes,counter_delta

def audit(root):
    def read(name):return json.loads((root/name).read_text())
    inputs=read('inputs.json');provenance=read('provenance.json');completion=read('complete.json');cleanup=read('cleanup.json')
    need(completion['status']=='pass' and cleanup['removed'],'Incomplete campaign/cleanup')
    targets=[524288,4194304,8388608,524288,4194304,8388608,8388608,4194304,524288]
    modes=['sim10-field']*len(targets);batches=[100]*len(targets)
    names=['preflight-%d'%t for t in targets[:3]]+['%02d-target-%d'%(i,t) for i,t in enumerate(targets[3:])]
    need([r['name'] for r in completion['cases']]==names,'Case matrix')
    expected_sources=source_hashes();need(provenance['source_sha256']==expected_sources,'Source provenance changed')
    need(all(v['filesystems'][0]['source']=='drpsrcf/ffb' for v in provenance['mounts'].values()),'Storage')
    frozen=json.loads((raw.HERE.parent/'jf_framework_cold_20261109/sim10_identity_check.json').read_text())
    need(raw.ts_hash(inputs['sim10']['timestamps'])==frozen['timestamp_sha256'],'Sim10 reference')
    need(inputs['sim10']['expected_reads']==json.loads((HERE/'expected-reference.json').read_text()),'Expected read reference')
    results=[];host=None;device=None;runtime=None;placement=None
    for index,(name,mode) in enumerate(zip(names,modes)):
        batch=batches[index]
        r=read(name+'.json');n=200 if index<3 else 10000;dataset='sim10' if mode=='sim10-field' else 'raw';info=inputs[dataset]
        total=(3358200 if dataset=='sim10' else 33557176)*n
        need(r['status']=='pass' and r['events']==n and r['settings']['mode']==mode and r['bytes']==total,'Case identity/bytes')
        need(r['settings']['bulk_target']==targets[index] and r['settings']['batch_size']==batch and r['framework']==dict(batch_size=batch,streams=1,budget_gib=6,bulk_target_bytes=targets[index],smd_events=1000),'Framework settings')
        need(r['seconds']>0 and math.isclose(r['gib_s'],total/(1<<30)/r['seconds']) and math.isclose(r['hz'],n/r['seconds']),'Rates')
        need(r['source_sha256']==expected_sources,'Case code changed')
        if host is None:host=r['host'];runtime=r['runtime_files']
        need(r['host']==host and r['runtime_files']==runtime,'Host/runtime changed')
        rows=r['rows'];need([x['rank'] for x in rows]==[0,1,2],'Ranks')
        need(not rows[0]['timestamps'] and not rows[1]['timestamps'],'Helper delivery')
        bd=rows[2];need(bd['timestamps']==info['timestamps'][:n],'Exact timestamps')
        need([t for c in bd['callbacks'] for t in c['timestamps']]==bd['timestamps'],'Callback delivery')
        need(all(0<c['size']<=batch and c['size']==len(c['timestamps']) for c in bd['callbacks']),'Subbatch size')
        need(len({c['stream'] for c in bd['callbacks']})==1,'One execution stream exercised')
        need(all('policy: interleave' in x['numa_policy'] for x in rows),'NUMA policy')
        need(bd['drain']==[dict(closed=True,pending=0,active=0,held=0,peers=1)],'Resource drain')
        if targets[index]==524288:need(max(c['size'] for c in bd['callbacks'])==batch,'Requested batch reached GPU')
        cpus=[x['affinity'] for x in rows];need(cpus[0]==cpus[1]==cpus[2] and len(cpus[0])>=8,'CPU placement')
        if placement is None:placement=cpus
        need(cpus==placement,'CPU placement changed')
        if device is None:device=bd['device']
        need(bd['device']==device,'GPU/settings changed')
        need(device['compat_mode']=='ON' and device['effective']==dict(num_threads=8,task_size=1<<20,bounce_buffer_size=16<<20,auto_direct_io_read=False),'KvikIO settings')
        io=bd['io'];need(io['requests']==io['completed_requests']==info['expected_reads'][str(n)][str(targets[index])]['count'] and io['requested_bytes']==io['completed_bytes']==total,'I/O count')
        need(io['read_size_histogram']==info['expected_reads'][str(n)][str(targets[index])]['histogram'],'Read histogram')
        need(sum(int(k)*v for k,v in io['read_size_histogram'].items())==total and sum(io['read_size_histogram'].values())==io['requests'],'Histogram totals')
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
        need(not bd['checks'],'Unexpected validation D2H')
        final=m['final_cache'];results.append(dict(name=name,mode=mode,batch=batch,bulk_target=targets[index],events=n,seconds=r['seconds'],gib_s=r['gib_s'],hz=r['hz'],callbacks=len(bd['callbacks']),
            subbatch_histogram={str(size):sum(c['size']==size for c in bd['callbacks']) for size in sorted({c['size'] for c in bd['callbacks']})},io=io,
            final_resident_fraction=sum(x['resident_pages'] for x in final)/sum(x['pages'] for x in final),nic_rx_gib={k:v['rx_bytes']/(1<<30) for k,v in delta.items()}))
    return dict(status='pass',host=host,device=device,cases=results)
if __name__=='__main__':print(json.dumps(audit(Path(sys.argv[1])),indent=2))
