"""Native ownership acceptance, then matched 10k old/new staging comparisons."""
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent
BASE=HERE.parent/'sim10_stream_length_20261109'
sys.path.insert(0,str(BASE))
from support import fixture,mount,require,sha,ts_hash,write,ROOT,EVENT_BYTES
from prepare_overlay import prepare
spec=importlib.util.spec_from_file_location('accepted_trace_checks',HERE.parent/'sim10_trace_20261110/campaign.py')
checks=importlib.util.module_from_spec(spec)
spec.loader.exec_module(checks)


def main(out):
    out.mkdir(parents=True,exist_ok=False)
    original=checks.file_stats()
    _,streams,stamps=fixture()
    gpu=subprocess.check_output(['nvidia-smi','-i',os.environ['CUDA_VISIBLE_DEVICES'].split(',')[0],
        '--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
    require(gpu.startswith('GPU-') and '\n' not in gpu,'GPU identity')
    os.environ['JF_GPU_UUID']=gpu
    overlay=out.parent/'overlay'
    prepare(overlay)
    os.environ['SIM10_OVERLAY']=str(overlay)
    os.environ['SIM10_VARIANT']='new'
    write(out/'inputs.json',dict(exp='mfx101555026',run=10,directory=str(ROOT),
        files=[s['files']['bd']['path'] for s in streams],timestamps=stamps,
        event_bytes=EVENT_BYTES,generation_sha256=sha(ROOT/'generation.json'),timestamp_sha256=ts_hash(stamps)))
    source=list(HERE.glob('*.py'))+list(HERE.glob('*.sbatch'))
    source += [HERE.parents[1]/'psana/psana/gpu'/(n+'.py') for n in ('gpu_events','gpu_input_group','gpu_input_window')]
    write(out/'provenance.json',dict(job=os.environ['SLURM_JOB_ID'],gpu=gpu,mount=mount(),
        input_metadata=original,overlay=json.loads((overlay/'manifest.json').read_text()),
        git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=HERE,text=True).strip(),
        source_sha256={str(p):sha(p) for p in source},started=time.time()))
    def launch(name,args,ranks=3,timeout=180):
        cmd=['srun','--mpi=cray_shasta','--nodes=1','--ntasks='+str(ranks),
            '--ntasks-per-node='+str(ranks),'--cpus-per-task=16','--gpus-per-node=1',
            '--gpu-bind=none','--cpu-bind=none','--kill-on-bad-exit=1',
            'numactl','--interleave=all',sys.executable,*map(str,args)]
        write(out/(name+'.command.json'),cmd)
        print('Starting',name,flush=True)
        with (out/(name+'.log')).open('x') as log:
            subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=timeout)
    launch('baseline-check',[HERE/'baseline_check.py',out/'baseline-check.json',out.parent/'baseline-test-tmp'],ranks=1)
    require(json.loads((out/'baseline-check.json').read_text())['expected_failure_reproduced'],'Baseline diagnostic')
    launch('gpu-tests',[HERE/'gpu_tests.py',out/'gpu-tests.json',out.parent/'pytest-tmp'],ranks=1,timeout=240)
    tests=json.loads((out/'gpu-tests.json').read_text())
    require(tests['exitcode']==tests['failed']==tests['skipped']==0 and tests['passed']>0,'Native acceptance')
    cases=[]
    clean=[('old',1),('new',1),('old',2),('new',2)]
    schedule=[('clean',v,n) for v,n in clean+list(reversed(clean))]
    schedule += [('timing',v,n) for v,n in [('old',1),('new',1),('old',2),('new',2)]]
    for index,(mode,variant,nstreams) in enumerate(schedule):
        name='%02d-%s-%s-s%d'%(index,mode,variant,nstreams)
        os.environ['SIM10_VARIANT']=variant
        program=HERE/('runner.py' if mode=='clean' else 'trace_runner.py')
        launch(name,[program,'--inputs',out/'inputs.json','--output',out/(name+'.json'),
            '--events','10000','--streams',str(nstreams)])
        result=json.loads((out/(name+'.json')).read_text())
        checks.validate(result,stamps,nstreams)
        require(result['job']==os.environ['SLURM_JOB_ID'],'Job identity')
        require(result['rows'][2]['device']['mask']==gpu,'GPU changed')
        if cases:
            require(result['host']==cases[0]['host'],'Host changed')
            require(result['rows'][2]['affinity']==cases[0]['affinity'],'CPU affinity changed')
        if mode=='timing':
            phases=json.loads((out/(name+'.json.phases.json')).read_text())
            reader=phases['counters']['reader']
            require(reader['total_requests']==50000 and reader['total_bytes']==33582000000,'Timing I/O')
            require(all(p['self_ns']>=0 for p in phases['phases'].values()),'Phase accounting')
        cases.append(dict(name=name,mode=mode,variant=variant,streams=nstreams,
            seconds=result['seconds'],gib_s=result['gib_s'],host=result['host'],affinity=result['rows'][2]['affinity']))
        write(out/'progress.json',cases)
        print(json.dumps(cases[-1]),flush=True)
    require(checks.file_stats()==original,'Input metadata changed')
    write(out/'complete.json',dict(status='pass',cases=cases,tests=tests,input_unchanged=True,finished=time.time()))


if __name__=='__main__':
    output=Path(sys.argv[1])
    def terminated(*_):raise RuntimeError('Slurm termination signal')
    signal.signal(signal.SIGTERM,terminated)
    try:main(output)
    except BaseException as error:
        if output.is_dir():write(output/'failed.json',dict(error=repr(error),time=time.time()))
        raise
