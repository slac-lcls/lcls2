"""Private-input, balanced Stage 5c campaign; stop on any acceptance failure."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time
from common import cache_inputs, records, sha
from pairs import validate_pair

POINTS=((1,1,20,2),(1,1,5,2),(1,1,20,1),(1,2,20,2),(1,4,20,2),(2,2,20,2),(2,4,20,2),(4,4,20,2))
DIAGNOSTIC_EVENTS=1000


def save(p,value):
    tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2,sort_keys=True)+'\n');tmp.replace(p)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--preflight-only',action='store_true')
    a=p.parse_args();root=a.root.resolve();job=os.environ['SLURM_JOB_ID']
    for key,value in (('PS_EB_NODES','1'),('PS_SRV_NODES','0'),('PS_PARALLEL','mpi')):
        assert os.environ.get(key)==value, ('missing or incorrect launch environment',key)
    output=root/f'job-{job}';output.mkdir(exist_ok=False)
    hashes=json.loads((root/'hashes.json').read_text())
    def verify():
        for name,h in hashes.items():assert sha(name)==h,name
    verify()
    reference=json.loads((root/'reference.json').read_text())
    stage=Path('/lscratch/monarin')/f'user-kernel-stage5c-{job}'
    stage.mkdir(parents=True,exist_ok=False)
    manifest=reference['10000'];sizes=manifest['stage_bytes']
    assert shutil.disk_usage(stage).free>sum(sizes.values())+20*1024**3
    env=dict(os.environ,BENCH_CPU_AFFINITY=','.join(map(str,sorted(os.sched_getaffinity(0)))))
    gpu_info=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,pci.bus_id,name,memory.total','--format=csv,noheader'],text=True)
    assert {'0','1'}.issubset(set(os.environ['SLURM_JOB_GPUS'].split(','))), 'requires physical GPUs 0 and 1'
    provenance=dict(job=job,host=os.uname().nodename,base=(root/'source-commit.txt').read_text().strip(),
        gpu_info=gpu_info,stage=str(stage),source=str(a.source),points=POINTS,
        batch_work='calibration + validity + sorted radial integration, float64 histogram',
        bins_provenance='Fixed validated run-51 physical-layout radial map used as a performance workload on run 387; not run-387 q/beam calibration.',
        variants=['event_loop','batched_task'],events=10000,bulk='on',
        diagnostic_events=DIAGNOSTIC_EVENTS,complete=False)
    rows=[];pairs=[]
    start=time.monotonic()
    try:
        def copy(name):
            src=a.source/name;dst=stage/name;before=src.stat();remaining=sizes[name];h=hashlib.sha256()
            with src.open('rb',buffering=0) as f,dst.open('xb',buffering=0) as g:
                while remaining:
                    block=f.read(min(16<<20,remaining));assert block
                    assert g.write(block)==len(block);h.update(block);remaining-=len(block)
                os.fsync(g.fileno())
            after=src.stat();assert (before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns)
            print('STAGED',name,flush=True)
            return dict(name=name,bytes=sizes[name],sha256=h.hexdigest())
        with ThreadPoolExecutor(max_workers=5) as pool:provenance['staged']=list(pool.map(copy,sizes))
        (stage/'smalldata').mkdir()
        for name in sizes:
            smd=name.replace('.xtc2','.smd.xtc2');shutil.copy2(a.source/'smalldata'/smd,stage/'smalldata'/smd)
            assert sha(stage/'smalldata'/smd)==manifest['smd_hashes'][smd]
        save(output/'provenance.json',provenance)
        def sample(point,cache,round_id,variant,diagnostic=False):
            g,b,batch,depth=point
            tag=f'g{g}-bd{b}-n{batch}-d{depth}-{cache}-r{round_id}-{variant}'+('-diagnostic' if diagnostic else '')
            before=None if diagnostic else cache_inputs(stage,cache,ranges=manifest['prefixes'])
            command=['mpiexec','--oversubscribe','--bind-to','none','-n',str(b+2),sys.executable,
                str(root/'scripts/stage5c/bench.py'),'--directory',str(stage),
                '--reference',str(root/'reference.json'),'--pixels',str(root/'pixels.json'),
                '--constants',str(root/'constants.pkl.gz'),'--bins',str(root/'bins.npz'),
                '--bulk','on','--events',str(DIAGNOSTIC_EVENTS) if diagnostic else '10000','--workload','input',
                '--variant',variant,'--batch-size',str(batch),'--depth',str(depth)]
            if diagnostic:command.append('--check-pixels')
            print('BEGIN',tag,flush=True);begin=time.monotonic()
            with (output/(tag+'-gpu.csv')).open('w') as gpu:
                monitor=subprocess.Popen(['nvidia-smi','--query-gpu=timestamp,index,memory.used,utilization.gpu,utilization.memory',
                    '--format=csv,noheader,nounits','-lms','250'],stdout=gpu)
                try:
                    with (output/(tag+'.log')).open('w') as log:
                        subprocess.run(command,env=dict(env,SLURM_GPUS_ON_NODE=str(g)),stdout=log,stderr=subprocess.STDOUT,check=True,timeout=900)
                finally:monitor.terminate();monitor.wait(timeout=10)
            parsed=records((output/(tag+'.log')).read_text(),'STAGE5C_RESULT ');assert len(parsed)==1
            row=parsed[0];assert row['events']==(DIAGNOSTIC_EVENTS if diagnostic else 10000)
            assert (row['ngpus'],row['nbds'],row['batch_size'],row['depth'],row['variant'])==(g,b,batch,depth,variant)
            row.update(cache=cache,round=round_id,tag=tag,wall_s=time.monotonic()-begin,before=before,
                after=None if diagnostic else cache_inputs(stage,cache,prepare=False,ranges=manifest['prefixes']),
                log_sha256=sha(output/(tag+'.log')))
            rows.append(row);save(output/'results.json',rows)
            print('PASS',tag,round(row['events_per_s'],2),'events/s',flush=True)
            return row
        # Every topology/layout must pass actual launch counts and histogram equivalence before timing.
        for point in POINTS:
            baseline=sample(point,'warm',0,'event_loop',True)
            batched=sample(point,'warm',0,'batched_task',True)
            validate_pair(baseline,batched)
        print('ALL_PREFLIGHTS_PASS',flush=True)
        if not a.preflight_only:
            schedule=[(point,'warm',r) for r in range(1,7) for point in (POINTS if r%2 else tuple(reversed(POINTS)))
                      if r<=4 or point==POINTS[0]]
            schedule += [(POINTS[0],'cold',r) for r in range(1,5)]
            for point,cache,r in schedule:
                order=('event_loop','batched_task') if r%2 else ('batched_task','event_loop')
                values={v:sample(point,cache,r,v) for v in order}
                before,after=values['event_loop'],values['batched_task']
                comparison=validate_pair(before,after)
                pairs.append(dict(point=point,cache=cache,round=r,event_loop_s=before['loop_s'],batched_s=after['loop_s'],
                    **comparison))
                save(output/'pairs.json',pairs)
        summary=[]
        for point,cache in [(p,'warm') for p in POINTS]+[(POINTS[0],'cold')]:
            selected=[r for r in pairs if tuple(r['point'])==point and r['cache']==cache]
            if selected:summary.append(dict(point=point,cache=cache,pairs=len(selected),
                median_paired_percent=statistics.median(r['percent'] for r in selected),
                median_paired_delta_s=statistics.median(r['delta_s'] for r in selected),
                event_median_s=statistics.median(r['event_loop_s'] for r in selected),
                batched_median_s=statistics.median(r['batched_s'] for r in selected)))
        verify();provenance['complete']=True;provenance['wall_s']=time.monotonic()-start
        save(output/'summary.json',dict(complete=True,summary=summary,preflight_only=a.preflight_only))
        print('STAGE5C_CAMPAIGN_COMPLETE',flush=True)
    finally:
        assert stage==Path('/lscratch/monarin')/f'user-kernel-stage5c-{job}'
        shutil.rmtree(stage);provenance['removed_stage']=str(stage);save(output/'provenance.json',provenance)


if __name__=='__main__':main()
