"""Audit the KvikIO bounce-buffer screen, including physical CPU demand and cold NIC evidence."""
import json
from collections import Counter
import math
from pathlib import Path
import sys
from prepare_runtime import validate_configs, PR_SHA
from support import require as need,EVENT_BYTES,histogram,counter_delta,raw,sha
from cpu_plan import case_matrix,plan,slurm_request,plan_key
from hardware import validate as validate_hardware
from task_evidence import pilot_evidence
from host_memory import validate_host_memory


def audit(root):
    read=lambda name:json.loads((root/name).read_text())
    p=read('provenance.json');info=read('inputs.json');done=read('complete.json');tests=read('gpu-tests.json')
    need(tests['exitcode']==0 and tests['passed']==31 and tests['failed']==tests['skipped']==0,'GPU tests')
    need(done['status']=='pass' and read('cleanup.json')['removed'],'Completion/cleanup')
    need(p['mount']['filesystems'][0]['source']=='drpsrcf/ffb','FFB mount')
    need(sha(root/'hardware.json')==p['hardware_sha256'],'Hardware receipt identity')
    hardware=read('hardware.json');validate_hardware(hardware,p['gpu_uuid'])
    receipt=p['native_receipt']
    need(receipt['status']=='pass' and receipt['pr_sha']==PR_SHA,'PR identity')
    need(validate_configs(receipt['build_configurations'])==receipt['config_mismatches'],'Build compatibility')
    need(len(info['timestamps'])==100000 and info['timestamp_sha256']==raw.ts_hash(info['timestamps']),'Input identity')
    matrix=case_matrix()
    need([(c['name'],c['bds'],c['workers'],c['streams'],c['batch'],c['bounce_mib'],c['events']) for c in done['cases']]==matrix,'Matrix')
    need(p['slurm_request']==slurm_request(),'Scheduler CPU demand')
    env=p['slurm_environment']
    request=slurm_request()
    need(int(env['SLURM_NTASKS'])==request['ntasks'] and int(env['SLURM_CPUS_PER_TASK'])==request['cpus_per_task'],'Scheduler task/CPU request mismatch')
    need(len({(x['socket'],x['core']) for x in p['cpu_topology']})>=request['requested_physical_cores'],'Allocated physical cores')
    result=[];host=device=runtime=None
    for name,bds,num_workers,streams,batch,bounce_mib,n in matrix:
        r=read(name+'.json');rows=r['rows'];workers=rows[2:];m=r['measurement']
        expected_plan=plan(p['cpu_topology'],bds,num_workers)
        need(p['cpu_plans'][plan_key(bds,num_workers)]==expected_plan,'Saved CPU plan')
        need(r['status']=='pass' and r['events']==n and r['bytes']==EVENT_BYTES*n,'Case counts')
        need(r['settings']['bds']==bds and r['settings']['workers']==num_workers and r['settings']['streams']==streams and r['settings']['batch']==batch and r['settings']['bounce_mib']==bounce_mib,'Case topology')
        need(r['framework']==dict(batch_size=batch,streams=streams,budget_gib=6,bulk_target_bytes=524288,smd_events=1000),'Settings')
        need([x['rank'] for x in rows]==list(range(bds+2)) and [x['role'] for x in rows]==['smd0','eb']+['bd']*bds,'Roles')
        delivered=[t for x in workers for t in x['timestamps']]
        need(sorted(delivered)==info['timestamps'][:n] and len(set(delivered))==n,'Exact global partition')
        need(all(not x['timestamps'] and not x['callbacks'] and x['device'] is None for x in rows[:2]),'Service ranks')
        callbacks=[c for x in workers for c in x['callbacks']]
        need(len(callbacks)==n//batch and all(c['size']==batch for c in callbacks),'Unexpected batch splitting/global callback count')
        case_device=workers[0]['device']
        physical_device={k:v for k,v in case_device.items() if k!='effective'}
        if host is None:host=r['host'];device=physical_device;runtime=workers[0]['runtime_files']
        need(physical_device==device and hardware['host']==host,'Hardware changed between cases')
        need(r['host']==host and device['mask']==p['gpu_uuid'] and 'A100-SXM4-40GB' in device['name'] and device['kvikio'].startswith('26.10.'),'Assigned GPU/host')
        need(device['compat_mode']=='ON' and case_device['effective']==dict(num_threads=num_workers,task_size=1<<20,bounce_buffer_size=bounce_mib<<20,auto_direct_io_read=False),'KvikIO')
        need(r['source_sha256']==p['source_sha256'],'Harness identity')
        need(device['kvikio_identity']==receipt['kvikio_identity'],'KvikIO binary/header identity')
        if n<100000:
            need(read(name+'-io-evidence.json')==pilot_evidence(root/(name+'.json'),r,n,histogram(n)),'Pilot syscall evidence')
        for i,x in enumerate(rows):
            validate_host_memory(x['host_memory'],bds,num_workers,bounce_mib)
            need(x['host']==host and x['runtime_files']==runtime and 'policy: interleave' in x['numa_policy'],'Placement/runtime')
            need(x['cpu_plan']==expected_plan and x['affinity']==expected_plan['ranks'][i],'CPU budget/binding')
            need(set(x['affinity'])<=set(x['inherited_affinity']),'Inherited CPU affinity')
            for sample in ('threads_before','threads_after'):
                need(x[sample]['clock_ticks']>0 and x[sample]['threads'],'Thread inventory')
        source=(Path(receipt['source'])/'psana/psana').resolve()
        for item in runtime.values():
            path=Path(item['path'])
            if path.suffix=='.py':
                need(path.is_relative_to(source),'Python path outside pinned PR')
                expected=receipt['python'].get(str(path.relative_to(source)))
            else:expected=receipt['native'].get(str(path))
            need(expected is not None and item['sha256']==expected,'Runtime receipt mismatch')
        for x in workers:
            k=len(x['timestamps']);io=x['io']
            need(x['reader_task_sizes'] in ([None],[1<<20]),'Reader task override')
            need((x['trace_window']['tracer_pid']>0)==(n<100000),'Trace mode')
            need(0<x['trace_window']['start']<x['trace_window']['end'],'Trace window')
            need(k>0 and x['timestamps']==sorted(set(x['timestamps'])),'Local ordered partition')
            need([t for c in x['callbacks'] for t in c['timestamps']]==x['timestamps'],'Callback identities')
            need(len(x['callbacks'])==k//batch and all(c['size']==len(c['timestamps'])==batch for c in x['callbacks']),'Local callback sizes/count')
            need(len({c['stream'] for c in x['callbacks']})==streams,'Stream count')
            need(io['requests']==io['completed_requests']==5*k and io['requested_bytes']==io['completed_bytes']==EVENT_BYTES*k,'Read accounting')
            need(io['issue_groups_calls']==len(x['callbacks']),'PR298 batch issue count')
            need(io['read_size_histogram']==histogram(k) and io['pressure_drains']==x['cpu_reads']==0,'Read histogram/pressure')
            need(x['drain']==[dict(closed=True,pending=0,active=0,held=0,peers=bds)],'Drain')
            need(x['device']==case_device,'Shared physical GPU')
            need(x['memory']['before_free']>=(bds*6+4)*(1<<30),'Aggregate GPU headroom')
        for key in ('conditioning','initial_cache'):
            need([x['path'] for x in m[key]]==info['files'] and all(x['resident_pages']==0 for x in m[key]),'Cold cache')
        need(m['seconds']==r['seconds'] and r['seconds']>0 and math.isclose(r['gib_s'],r['bytes']/2**30/r['seconds']) and math.isclose(r['hz'],n/r['seconds']),'Rate')
        delta=counter_delta(m['nic_before'],m['nic_after'],r['bytes'])
        need(delta==m['nic_delta'] and all(v['seconds']>=r['seconds'] for v in delta.values()),'NIC bracket')
        need(counter_delta(m['idle']['before'],m['idle']['after'])==m['idle']['delta'],'Idle bracket')
        rx=sum(v['rx_bytes'] for v in delta.values())
        idle=sum(m['idle']['delta'][nic]['rx_bytes']/m['idle']['delta'][nic]['seconds']*v['seconds'] for nic,v in delta.items())
        need(rx-idle>=0.90*r['bytes'],'Insufficient NIC RX to corroborate network-served input')
        result.append(dict(name=name,bds=bds,workers=num_workers,total_workers=bds*num_workers,required_physical_cores=expected_plan['required_physical_cores'],streams=streams,batch=batch,bounce_mib=bounce_mib,events=n,seconds=r['seconds'],gib_s=r['gib_s'],hz=r['hz'],reader_calls=sum(x['io']['requests'] for x in workers),issue_groups_calls=sum(x['io']['issue_groups_calls'] for x in workers),callbacks=len(callbacks),callback_size_histogram=dict(Counter(c['size'] for c in callbacks)),bd_events=[len(x['timestamps']) for x in workers],cpu_plan=expected_plan,nic_delta=delta,idle=m['idle']['delta'],idle_adjusted_rx_payload_ratio=(rx-idle)/r['bytes'],memory=[x['memory'] for x in workers],host_memory=[x['host_memory'] for x in rows],task_evidence=read(name+'-io-evidence.json') if n<100000 else None))
    return dict(status='pass',host=host,device=device,cases=result)


if __name__=='__main__':print(json.dumps(audit(Path(sys.argv[1])),indent=2))
