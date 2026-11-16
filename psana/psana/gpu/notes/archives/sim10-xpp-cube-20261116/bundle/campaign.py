"""Sequential, bounded same-node Perlmutter S/E/EC/ECC comparison."""
import datetime
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
import shutil
from support import HERE, ROOT, TASK_ROOT, require, sha, write, verify_inputs, gpu_inventory
from plan import ATTEMPT, WALLTIME, TIME_LIMIT, BUDGET, OUTPUT_RESERVE, cases, placement
from qualified_support import select_affinity, diagnostics
from memory_watch import Watch, validate_memory
from audit import audit_case

def launch(tasks, program, args, timeout):
    return ['srun','--mpi=cray_shasta','--nodes=1','--ntasks='+str(tasks),
            '--ntasks-per-node='+str(tasks),'--cpus-per-task='+str(120//tasks),
            '--gpus-per-node=4','--gpu-bind=none','--cpu-bind=none','--kill-on-bad-exit=1',
            '--time='+str(math.ceil(timeout/60)), 'numactl','--interleave=all',sys.executable,
            str(HERE/program),*map(str,args)]

def allocation(text):
    def field(name):
        matches=re.findall(r'(?:^|\s)'+name+r'=(\S+)',text)
        require(len(matches)==1, 'Missing Slurm '+name)
        return matches[0]
    require(field('Account')=='lcls_g' and field('QOS') in ('premium','gpu_premium'), 'Account/QoS')
    require(field('NumNodes')=='1' and field('TimeLimit')==TIME_LIMIT, 'Bounded allocation')
    deadline=datetime.datetime.fromisoformat(field('EndTime')).replace(tzinfo=datetime.timezone.utc).timestamp()
    require(0 < deadline-time.time() <= WALLTIME+2, 'Allocation deadline outside remaining walltime')
    return deadline

def runstep(output, name, command, timeout, deadline):
    require(deadline-time.time() >= timeout+120, 'Insufficient Slurm time for next whole step')
    write(output/(name+'.command.json'), command)
    begin=time.time()
    try:
        with (output/(name+'.log')).open('x') as stream:
            process=subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                ret=process.wait(timeout=timeout)
                require(ret==0, 'Step failed: '+name)
            except BaseException:
                if process.poll() is None:
                    os.killpg(process.pid,signal.SIGTERM)
                    try:process.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid,signal.SIGKILL);process.wait(timeout=15)
                raise
    finally:
        write(output/(name+'.timing.json'),dict(started=begin,finished=time.time(),seconds=time.time()-begin))

def verify_bundle(attempt):
    plan=json.loads((attempt/'plan.json').read_text())
    require(plan['walltime_seconds']==WALLTIME and plan['budget']==BUDGET and plan['cases']==cases(), 'Frozen comparison plan')
    require(plan['bundle_sha256']==sha(HERE/'bundle.json'), 'Bundle manifest changed')
    bundle=json.loads((HERE/'bundle.json').read_text())
    require(bundle['commit']==plan['source_commit'], 'Bundle source commit')
    require(all(sha(HERE/path)==digest for path,digest in bundle['files'].items()), 'Bundle changed')
    require(all(sha(path)==digest for path,digest in bundle['external'].items()), 'External inputs changed')
    return bundle

def main(attempt):
    require(attempt==TASK_ROOT/ATTEMPT and HERE==attempt/'bundle', 'Prepared immutable attempt only')
    deadline=allocation((attempt/'slurm-job.txt').read_text())
    require(os.environ.get('SLURM_JOB_NUM_NODES')=='1', 'Single node')
    output=attempt/'results';output.mkdir()
    bundle=verify_bundle(attempt)
    info=json.loads((attempt/'inputs.json').read_text())
    ranges=json.loads((attempt/'ranges.json').read_text())
    verify_inputs(info)
    require(shutil.disk_usage(attempt).free >= OUTPUT_RESERVE, 'Need 4 GiB report/log reserve')
    state=dict(status='running',job=os.environ['SLURM_JOB_ID'],completed=[],source_commit=bundle['commit'])
    write(output/'status.json',state)
    # Hold a cooperative lock across all input cache conditioning and reads.
    import fcntl
    lock=(TASK_ROOT/'count2-use.lock').open('a')
    try:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        runstep(output,'runtime-recheck',launch(1,'check_reuse.py',[output/'runtime-recheck.json'],180),180,deadline)
        current=json.loads((output/'runtime-recheck.json').read_text())
        frozen=json.loads((attempt/'reuse-check.json').read_text())
        require(current['kvikio_identity']==frozen['kvikio_identity'] and current['cupy_identity']==frozen['cupy_identity'], 'GPU-library identity changed')
        uuids=subprocess.check_output(['nvidia-smi','-i',os.environ['CUDA_VISIBLE_DEVICES'],'--query-gpu=uuid','--format=csv,noheader'],text=True).strip().splitlines()
        require(len(uuids)==len(set(uuids))==4, 'Four granted GPUs')
        inventory=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,pci.bus_id','--format=csv,noheader'],text=True)
        pci_by_uuid=gpu_inventory(inventory)
        require(set(pci_by_uuid)==set(uuids), 'Inventory matches granted GPUs')
        write(output/'hardware.json',dict(uuids=uuids, inventory=inventory, pci_by_uuid=pci_by_uuid,
              topology=subprocess.check_output(['nvidia-smi','topo','-m'],text=True), numa=subprocess.check_output(['numactl','--hardware'],text=True), diagnostics=diagnostics()))
        probes=[]
        for tasks in (3,6):
            runstep(output,'affinity-%d'%tasks,launch(tasks,'affinity_probe.py',[output,tasks],30),30,deadline)
            probes.extend(json.loads((output/('affinity-%d-r%d.json'%(tasks,rank))).read_text()) for rank in range(tasks))
        common=set.intersection(*(set(p['cpus']) for p in probes))
        six=[dict(p,cpus=sorted(common)) for p in probes[3:]]
        selected=select_affinity(six)
        mapping=[selected[str(rank)] for rank in range(6)]
        write(output/'affinity.json',dict(mapping=mapping,probes=probes))
        for index,uuid in enumerate(uuids):
            os.environ['JF_GPU_UUID']=uuid
            runstep(output,'device-%d'%index,launch(1,'device_check.py',[output/('device-%d.json'%index)],120),120,deadline)
        for case in cases():
            gpu_plan=placement(uuids,case)
            gpu_plan['pci_by_uuid']=pci_by_uuid
            cpu_plan=mapping[:case['bds']+2]
            os.environ['SIM10_GPU_PLAN']=json.dumps(gpu_plan)
            os.environ['SIM10_CPU_PLAN']=json.dumps(cpu_plan)
            state['current']=case;write(output/'status.json',state)
            args=['--inputs',attempt/'inputs.json','--output',output/(case['name']+'.json')]
            for key in ('bds','batch','stage','events','kind'):args+=['--'+key,case[key]]
            watches=[Watch(uuid) for uuid in sorted(set(gpu_plan['rank_uuids'][2:]))]
            for watch in watches:watch.start()
            try:runstep(output,case['name'],launch(case['bds']+2,'runner.py',args,case['timeout']),case['timeout'],deadline)
            finally:
                memory=[watch.stop() for watch in watches]
                write(output/(case['name']+'.memory.json'),memory)
            for receipt in memory:validate_memory(receipt)
            result=json.loads((output/(case['name']+'.json')).read_text())
            audit_context=output/(case['name']+'.audit-context.json')
            write(audit_context,dict(case=case,inputs=str(attempt/'inputs.json'),ranges=str(attempt/'ranges.json'),
                                    mapping=cpu_plan,gpu_plan=gpu_plan))
            audit_output=output/(case['name']+'.audit.json')
            audit_timeout=20
            runstep(output,case['name']+'-audit',[sys.executable,str(HERE/'audit.py'),str(audit_context),
                    str(output/(case['name']+'.json')),str(audit_output)],audit_timeout,deadline)
            verdict=json.loads(audit_output.read_text())
            state['completed'].append(dict(case=case,audit=verdict,result_sha256=sha(output/(case['name']+'.json')),
                                           seconds=result['seconds'],payload_gib_s=result['payload_gib_s']))
            write(output/'status.json',state)
        runstep(output,'comparison-summary',[sys.executable,str(HERE/'comparison.py'),str(attempt)],
                BUDGET['summary'],deadline)
        state['summary_sha256']=sha(output/'comparison.json')
        verify_inputs(info);verify_bundle(attempt)
        state.pop('current',None);state['status']='passed_pending_slurm_and_independent_result_review'
        write(output/'complete.json',state);write(output/'status.json',state)
    except BaseException as error:
        state.update(status='failed',error=repr(error));write(output/'status.json',state);raise
    finally:
        write(output/'diagnostics-after.json',diagnostics())
        lock.close()

if __name__=='__main__':
    main(Path(sys.argv[1]).resolve())
