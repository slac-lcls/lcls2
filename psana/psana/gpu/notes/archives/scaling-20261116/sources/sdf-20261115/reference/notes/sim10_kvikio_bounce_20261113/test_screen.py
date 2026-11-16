"""Reject resource underprovisioning and corrupted distributed evidence."""
import copy,json,subprocess,sys
from pathlib import Path
import pytest
from cpu_plan import plan,case_matrix,slurm_request,plan_key,CONFIGURATIONS,validate_case
from audit import audit
from prepare_runtime import ALLOWED_CONFIG_MISMATCHES,PR_SHA
from support import raw,counter_delta,EVENT_BYTES,histogram,sha
from task_evidence import pilot_evidence
from host_memory import demand


def topology():
    return [dict(cpu=i+56*s,socket=i//28,core=i%28) for s in (0,1) for i in range(56)]


def test_cpu_budget_uses_physical_cores():
    for b,c in [(1,17),(2,28),(3,39),(4,50)]:
        p=plan(topology(),b)
        assert p['required_physical_cores']==c
        assert p['available_physical_cores']==56
        assert [len(r) for r in p['ranks']]==[3,3]+[11]*b
        ids=[cpu for r in p['ranks'] for cpu in r]
        assert len(set(ids))==c and max(ids)<56
    small=[dict(cpu=i,socket=0,core=i%36) for i in range(72)]
    with pytest.raises(ValueError,match='only 36 available'):plan(small,4)
    assert slurm_request()['requested_physical_cores']==36


def fixture(root):
    stamps=list(range(1,100001));files=['/ffb/'+s for s in 'abcde']
    native=dict(kvikio_identity={'binary':'hash'},status='pass',pr_sha=PR_SHA,source='/pr',build_configurations=copy.deepcopy(ALLOWED_CONFIG_MISMATCHES),config_mismatches=copy.deepcopy(ALLOWED_CONFIG_MISMATCHES),python={'__init__.py':'pythonhash'},native={'/install/dgram.so':'nativehash'})
    modules={'psana':dict(path='/pr/psana/psana/__init__.py',sha256='pythonhash'),'psana.dgram':dict(path='/install/dgram.so',sha256='nativehash')}
    p=dict(mount={'filesystems':[{'source':'drpsrcf/ffb'}]},gpu_uuid='GPU-test',cpu_topology=topology(),cpu_plans={plan_key(b,w):plan(topology(),b,w) for b,w,_,_ in CONFIGURATIONS},slurm_request=slurm_request(),slurm_environment=dict(SLURM_NTASKS='6',SLURM_CPUS_PER_TASK='6',SLURM_JOB_CPUS_PER_NODE='56'),source_sha256={'harness':'hash'},native_receipt=native)
    info=dict(timestamps=stamps,timestamp_sha256=raw.ts_hash(stamps),files=files)
    device=dict(kvikio_identity={'binary':'hash'},mask='GPU-test',name='NVIDIA A100-SXM4-40GB',kvikio='26.10.0',compat_mode='ON',effective=dict(num_threads=8,task_size=1<<20,bounce_buffer_size=16<<20,auto_direct_io_read=False))
    data={'provenance.json':p,'inputs.json':info,'gpu-tests.json':dict(exitcode=0,passed=31,failed=0,skipped=0),'cleanup.json':dict(removed=True)}
    for name,b,w,s,batch,t,n in case_matrix():
        before={'eth0':dict(monotonic=1.,rx_bytes_phy=100,tx_bytes_phy=10,device='/pci/1')}
        after={'eth0':dict(monotonic=12.,rx_bytes_phy=100+n*EVENT_BYTES,tx_bytes_phy=20,device='/pci/1')}
        idle_after={'eth0':dict(monotonic=2.,rx_bytes_phy=110,tx_bytes_phy=11,device='/pci/1')}
        m=dict(seconds=10.,conditioning=[dict(path=f,resident_pages=0) for f in files],initial_cache=[dict(path=f,resident_pages=0) for f in files],nic_before=before,nic_after=after,nic_delta=counter_delta(before,after,n*EVENT_BYTES),idle=dict(before=before,after=idle_after,delta=counter_delta(before,idle_after)))
        cp=p['cpu_plans'][plan_key(b,w)];rows=[]
        device['effective']['num_threads']=w
        device['effective']['bounce_buffer_size']=t<<20
        for rank in range(b+2):
            calls=[] if rank<2 else [dict(size=batch,stream=j%s,timestamps=stamps[i*batch:(i+1)*batch]) for j,i in enumerate(range(rank-2,n//batch,b))]
            delivered=[t for c in calls for t in c['timestamps']];k=len(delivered)
            sample=dict(clock_ticks=100,threads=[dict(tid=1,state='R',user_ticks=0,system_ticks=0)])
            hm=dict(mem_available_bytes=100<<30,VmRSS=1<<20,VmHWM=2<<20,VmLck=0,VmPin=0)
            host_memory=dict(startup=hm,before_loop=hm,after_loop=hm,demand=demand(b,w,t))
            rows.append(dict(host_memory=copy.deepcopy(host_memory),reader_task_sizes=[] if rank<2 else [None],trace_window=None if rank<2 else dict(tracer_pid=123 if n<100000 else 0,start=101,end=103),rank=rank,role='smd0' if rank==0 else 'eb' if rank==1 else 'bd',host='ampere',cpu_plan=copy.deepcopy(cp),affinity=list(cp['ranks'][rank]),inherited_affinity=list(range(56)),numa_policy='policy: interleave',runtime_files=copy.deepcopy(modules),timestamps=delivered,callbacks=calls,threads_before=sample,threads_after=sample,io=None if rank<2 else dict(requests=5*k,completed_requests=5*k,requested_bytes=k*EVENT_BYTES,completed_bytes=k*EVENT_BYTES,read_size_histogram=histogram(k),pressure_drains=0,issue_groups_calls=len(calls)),cpu_reads=0,drain=[] if rank<2 else [dict(closed=True,pending=0,active=0,held=0,peers=b)],device=None if rank<2 else copy.deepcopy(device),memory=None if rank<2 else dict(before_free=39*(1<<30))))
        data[name+'.json']=dict(status='pass',settings=dict(bds=b,workers=w,streams=s,batch=batch,bounce_mib=t),events=n,bytes=n*EVENT_BYTES,rows=rows,measurement=m,host='ampere',framework=dict(batch_size=batch,streams=s,budget_gib=6,bulk_target_bytes=524288,smd_events=1000),source_sha256=p['source_sha256'],seconds=10.,gib_s=n*EVENT_BYTES/2**30/10,hz=n/10)
    data['complete.json']=dict(status='pass',cases=[dict(name=name,bds=b,workers=w,streams=s,batch=batch,bounce_mib=t,events=n) for name,b,w,s,batch,t,n in case_matrix()])
    def save():
        for name,value in data.items():(root/name).write_text(json.dumps(value))
    hardware=dict(host='ampere',gpu_uuid='GPU-test',commands={k:dict(returncode=0,stdout='test',stderr='') for k in ('lscpu','gpu','gpu_topology')},pci_devices=[],nvme_controllers=[])
    (root/'hardware.json').write_text(json.dumps(hardware))
    p['hardware_sha256']=sha(root/'hardware.json')
    for name,b,w,s,batch,t,n in case_matrix():
        if n==100000:continue
        result=data[name+'.json']
        for row in result['rows'][2:]:
            counts=histogram(len(row['timestamps']))
            trace=root/(name+'.json.rank%d.strace'%row['rank'])
            trace.write_text(''.join('123 102.0 pread64(7, ""..., %s, 0) = %s\n'%(size,size) for size,count in counts.items() for _ in range(count)))
        data[name+'-io-evidence.json']=pilot_evidence(root/(name+'.json'),result,n,histogram(n))
    save();return data,save


def test_accept_complete_evidence(tmp_path):
    fixture(tmp_path)
    assert len(audit(tmp_path)['cases'])==12


@pytest.mark.parametrize('kind',['duplicate_events','readcount','issue_calls','smt','cpu_cap','affinity_overlap','nic_zero','cache','runtime','stream','budget','cleanup','gpu','matrix','batch','split','inherited','slurm','effective_workers','planned_workers','saved_workers','task_size','bounce_size','host_missing','host_demand','host_headroom','reader_override','trace_full','trace_pilot','kvikio_identity','trace_changed','gpu_change','hardware'])
def test_reject_corrupted_evidence(tmp_path,kind):
    d,save=fixture(tmp_path);r=d['02-bd4-workers4-batch200-bounce16.json'];bd=r['rows'][2]
    if kind=='duplicate_events':r['rows'][3]['timestamps'][0]=bd['timestamps'][0]
    elif kind=='readcount':bd['io']['completed_requests']-=1
    elif kind=='issue_calls':bd['io']['issue_groups_calls']*=500
    elif kind=='smt':
        for row in d['provenance.json']['cpu_topology']:row['socket']=0
    elif kind=='cpu_cap':bd['cpu_plan']['required_physical_cores']=24
    elif kind=='affinity_overlap':r['rows'][3]['affinity']=bd['affinity']
    elif kind=='nic_zero':
        m=r['measurement'];m['nic_after']['eth0']['rx_bytes_phy']=m['nic_before']['eth0']['rx_bytes_phy'];m['nic_delta']=counter_delta(m['nic_before'],m['nic_after'],r['bytes'])
    elif kind=='cache':r['measurement']['initial_cache'][0]['resident_pages']=1
    elif kind=='runtime':bd['runtime_files']['psana']['sha256']='wrong'
    elif kind=='stream':bd['callbacks'][1]['stream']=1
    elif kind=='budget':bd['memory']['before_free']=20*(1<<30)
    elif kind=='cleanup':d['cleanup.json']['removed']=False
    elif kind=='gpu':d['gpu-tests.json']['skipped']=1
    elif kind=='matrix':d['complete.json']['cases'].pop()
    elif kind=='batch':r['settings']['batch']=100
    elif kind=='split':
        c=bd['callbacks'][0];half=len(c['timestamps'])//2
        bd['callbacks'][:1]=[dict(size=half,stream=c['stream'],timestamps=c['timestamps'][:half]),dict(size=half,stream=c['stream'],timestamps=c['timestamps'][half:])]
        bd['io']['issue_groups_calls']+=1
    elif kind=='inherited':bd['inherited_affinity']=[]
    elif kind=='effective_workers':bd['device']['effective']['num_threads']=8
    elif kind=='planned_workers':bd['cpu_plan']['workers_per_bd']=8
    elif kind=='saved_workers':r['settings']['workers']=8
    elif kind=='task_size':bd['device']['effective']['task_size']=4<<20
    elif kind=='bounce_size':bd['device']['effective']['bounce_buffer_size']=64<<20
    elif kind=='host_missing':del bd['host_memory']['after_loop']
    elif kind=='host_demand':bd['host_memory']['demand']['aggregate_estimate_bytes']=0
    elif kind=='host_headroom':bd['host_memory']['before_loop']['mem_available_bytes']=1
    elif kind=='reader_override':bd['reader_task_sizes']=[4<<20]
    elif kind=='trace_full':bd['trace_window']['tracer_pid']=123
    elif kind=='trace_pilot':d['preflight-bd1-workers8-batch50-bounce16.json']['rows'][2]['trace_window']['tracer_pid']=0
    elif kind=='kvikio_identity':bd['device']['kvikio_identity']={'binary':'changed'}
    elif kind=='trace_changed':
        path=tmp_path/'preflight-bd1-workers8-batch50-bounce16.json.rank2.strace';path.write_text(path.read_text()+'123 104.0 +++ exited with 0 +++\n')
    elif kind=='gpu_change':
        for x in r['rows'][2:]:x['device']['mask']='GPU-other'
    elif kind=='hardware':d['provenance.json']['hardware_sha256']='wrong'
    elif kind=='slurm':d['provenance.json']['slurm_environment']['SLURM_CPUS_PER_TASK']='4'
    save()
    with pytest.raises((ValueError,KeyError)):audit(tmp_path)


def test_imports_survive_helper_path_changes():
    folder=Path(__file__).parent
    code="import campaign, audit, runtime; from pathlib import Path; p=Path(%r); assert Path(audit.__file__).parent==p; assert Path(runtime.__file__).parent==p"%str(folder)
    subprocess.run([sys.executable,'-c',code],cwd=folder,check=True)


def test_worker_cpu_demand_and_no_cap():
    for b,w,cores in [(1,4,13),(1,8,17),(1,16,25),(2,8,28),(4,4,34),(4,8,50)]:
        p=plan(topology(),b,w)
        assert p['required_physical_cores']==cores
        assert [len(r) for r in p['ranks']]==[3,3]+[w+3]*b
        assert p['total_workers']==b*w
    with pytest.raises(ValueError,match='Need 82 physical cores'):plan(topology(),4,16)
    larger=slurm_request(((4,16,100),))
    assert larger['required_physical_cores']==82 and larger['requested_physical_cores']==84
    # A high-worker, low-rank case can set the maximum independently of max BDs.
    mixed=slurm_request(((1,64,100),(4,4,100)))
    assert mixed['ntasks']==6 and mixed['required_physical_cores']==73 and mixed['requested_physical_cores']==78


def test_screen_controls_and_bounce_sizes():
    assert set(CONFIGURATIONS)=={(1,8,50,16),(1,8,50,64),(4,4,200,16),(4,4,200,64)}
    cases=case_matrix()
    assert len(cases)==12 and len({c[0] for c in cases})==12
    for _,b,w,s,batch,t,n in cases:validate_case(b,w,batch,t,s,n)
    assert [c[1:] for c in cases[4:8]]==[c[1:] for c in cases[8:]][::-1]
    for case in [(4,8,200,16,1,100000),(1,8,50,16,2,100000),(1,8,50,32,1,100000),(4,4,200,64,1,1000)]:
        with pytest.raises(ValueError):validate_case(*case)


def test_hardware_requires_core_inventory():
    from hardware import validate
    record=dict(host='ampere',gpu_uuid='GPU-test',commands={k:dict(returncode=0,stdout='test') for k in ('lscpu','gpu','gpu_topology')})
    validate(record,'GPU-test')
    record['commands']['gpu_topology']['returncode']=1
    with pytest.raises(ValueError,match='gpu_topology'):validate(record,'GPU-test')
