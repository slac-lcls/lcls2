"""Bounded adaptive screen followed by three fresh-process finalist repeats."""
import argparse
import json
import os
from pathlib import Path
import random
import statistics
import subprocess
import sys
from plan_reads import require


def config(name, **kw):
    result=dict(name=name,pattern='events',chunk_mib=0,threads=4,task_mib=4,bounce_mib=16,
                depth=35,direct=False,cpus=16)
    result.update(kw)
    return result


def campaign(output):
    output.mkdir(mode=0o770,exist_ok=False)
    results=[]
    rng=random.Random(20261108)
    def execute(c, phase):
        dest=output/(c['name']+'.json')
        with (output/(c['name']+'.log')).open('x') as log:
            subprocess.run([sys.executable,str(Path(__file__).with_name('read_case.py')),
                            '--config',json.dumps(c),'--output',str(dest)],
                           stdout=log,stderr=subprocess.STDOUT,check=True)
        r=json.loads(dest.read_text())
        require(r['status']=='pass','Failed measurement')
        r=dict(config=c,phase=phase,path=str(dest),gib_s=r['gib_s'],seconds=r['seconds'],
               passes=len(r['passes']),total_bytes=r['total_bytes'])
        results.append(r)
        (output/'progress.json').write_text(json.dumps(results,indent=2)+'\n')
        print(json.dumps(r),flush=True)
        return r
    anchor=config('anchor-first',cpus=4)
    execute(anchor,'initial-control')
    screen=[config('event-t%d'%t,threads=t) for t in (1,4,8,16)]
    screen += [config('event-q%d'%q,depth=q) for q in (5,140)]
    screen += [config('slab-c%d-t%d'%(c,t),pattern='slabs',chunk_mib=c,threads=t,depth=10)
               for c in (4,16,64) for t in (4,8,16)]
    rng.shuffle(screen)
    for c in screen: execute(c,'screen')
    best_slab=max((r for r in results if r['config']['pattern']=='slabs'),key=lambda r:r['gib_s'])['config']
    tune=[]
    for name,key,values in [('task','task_mib',(1,16)),('queue','depth',(5,40)),('bounce','bounce_mib',(1,64))]:
        for value in values:
            c=dict(best_slab);c.update(name='tune-%s%d'%(name,value));c[key]=value;tune.append(c)
    rng.shuffle(tune)
    for c in tune:execute(c,'tune')
    best_event=max((r for r in results if r['config']['pattern']=='events'),key=lambda r:r['gib_s'])['config']
    best_slab=max((r for r in results if r['config']['pattern']=='slabs'),key=lambda r:r['gib_s'])['config']
    for c in (best_event,best_slab):
        direct=dict(c);direct.update(name='direct-'+c['pattern'],direct=True)
        execute(direct,'direct-probe')
    finalists={
        'anchor':anchor,
        'best-events':max((r for r in results if r['config']['pattern']=='events'),key=lambda r:r['gib_s'])['config'],
        'best-slabs':max((r for r in results if r['config']['pattern']=='slabs'),key=lambda r:r['gib_s'])['config'],
    }
    confirmation=[]
    for repeat in range(1,4):
        names=list(finalists);rng.shuffle(names)
        for label in names:
            c=dict(finalists[label]);c['name']='confirm-%s-r%d'%(label,repeat)
            r=execute(c,'confirmation');confirmation.append((label,r))
    summary={label:dict(config=c,median_gib_s=statistics.median(r['gib_s'] for n,r in confirmation if n==label),
                        min_gib_s=min(r['gib_s'] for n,r in confirmation if n==label),
                        max_gib_s=max(r['gib_s'] for n,r in confirmation if n==label))
             for label,c in finalists.items()}
    (output/'summary.json').write_text(json.dumps(dict(status='pass',results=results,confirmation=summary,
        interpretation='Adaptive screening, not full factorial; finalists selected by screening peak, medians use only independent confirmation cases. Cache/order/system effects uncontrolled. Best observed, not universal max.'),indent=2)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output',type=Path)
    args=parser.parse_args()
    campaign(args.output)
